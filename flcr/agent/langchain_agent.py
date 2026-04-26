from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from typing import Any

from pydantic import BaseModel, Field

try:
    from langchain.agents import create_agent
except ImportError:  # pragma: no cover - optional dependency
    create_agent = None

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:  # pragma: no cover - optional dependency
    MultiServerMCPClient = None

try:
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover - optional dependency
    ChatOllama = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - optional dependency
    ChatGoogleGenerativeAI = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None

try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover - optional dependency
    ChatGroq = None


DEFAULT_AGENT_PROVIDER = os.environ.get("FLCR_AGENT_PROVIDER", "groq").lower()
DEFAULT_OLLAMA_MODEL = os.environ.get("FLCR_OLLAMA_MODEL", "qwen3:8b")
DEFAULT_GEMINI_MODEL = os.environ.get("FLCR_GEMINI_MODEL", "gemini-2.5-flash-lite")
DEFAULT_OPENAI_MODEL = os.environ.get("FLCR_OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_GROQ_MODEL = os.environ.get("FLCR_GROQ_MODEL", "qwen/qwen3-32b")
DEFAULT_OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_AGENT_CANDIDATE_K = int(os.environ.get("FLCR_AGENT_CANDIDATE_K", "30"))
DEFAULT_AGENT_MAX_RESULTS = int(os.environ.get("FLCR_AGENT_MAX_RESULTS", "10"))
DEFAULT_MCP_SERVER_URL = os.environ.get("FLCR_MCP_SERVER_URL", "http://127.0.0.1:8000/agent-tools/mcp")


class AgentSearchResponse(BaseModel):
    selected_titles: list[str] = Field(
        description=(
            "A filtered and ordered list of movie titles chosen from the tool output. "
            "Return only the titles that should actually be shown to the user."
        )
    )
    summary: str = Field(
        description=(
            "A short overall summary of the final selected list. "
            "Mention the strongest matches and describe the list in a balanced way."
        )
    )


def _fastmcp_available() -> bool:
    return importlib.util.find_spec("fastmcp") is not None


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _provider_label() -> str:
    if DEFAULT_AGENT_PROVIDER == "gemini":
        return f"gemini:{DEFAULT_GEMINI_MODEL}"
    if DEFAULT_AGENT_PROVIDER == "openai":
        return f"openai:{DEFAULT_OPENAI_MODEL}"
    if DEFAULT_AGENT_PROVIDER == "groq":
        return f"groq:{DEFAULT_GROQ_MODEL}"
    return f"ollama:{DEFAULT_OLLAMA_MODEL}"


def agent_is_available() -> tuple[bool, str | None]:
    if create_agent is None:
        return False, "LangChain is not installed."
    if MultiServerMCPClient is None:
        return False, "langchain-mcp-adapters is not installed."
    if not _fastmcp_available():
        return False, "fastmcp is not installed."
    if DEFAULT_AGENT_PROVIDER == "gemini":
        if ChatGoogleGenerativeAI is None:
            return False, "langchain-google-genai is not installed."
        if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            return False, "GOOGLE_API_KEY or GEMINI_API_KEY is not set."
        return True, None
    if DEFAULT_AGENT_PROVIDER == "openai":
        if ChatOpenAI is None:
            return False, "langchain-openai is not installed."
        if not os.environ.get("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY is not set."
        return True, None
    if DEFAULT_AGENT_PROVIDER == "groq":
        if ChatGroq is None:
            return False, "langchain-groq is not installed."
        if not os.environ.get("GROQ_API_KEY"):
            return False, "GROQ_API_KEY is not set."
        return True, None
    if ChatOllama is None:
        return False, "langchain-ollama is not installed."
    return True, None


def _build_llm():
    if DEFAULT_AGENT_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(model=DEFAULT_GEMINI_MODEL, temperature=0)
    if DEFAULT_AGENT_PROVIDER == "openai":
        return ChatOpenAI(model=DEFAULT_OPENAI_MODEL, temperature=0)
    if DEFAULT_AGENT_PROVIDER == "groq":
        return ChatGroq(model=DEFAULT_GROQ_MODEL, temperature=0)
    return ChatOllama(model=DEFAULT_OLLAMA_MODEL, temperature=0, base_url=DEFAULT_OLLAMA_BASE_URL)


def _extract_tool_payload(messages: list[Any]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if getattr(message, "type", "") != "tool":
            continue
        text = _message_text(message)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "recommendations" in payload:
            return payload
    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    payload_text = (text or "").strip()
    if not payload_text:
        return None

    candidates = [payload_text]
    match = __import__("re").search(r"\{.*\}", payload_text, __import__("re").DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


async def _build_agent(mcp_server_url: str = DEFAULT_MCP_SERVER_URL):
    resolved_mcp_url = mcp_server_url or DEFAULT_MCP_SERVER_URL
    llm = _build_llm()
    client = MultiServerMCPClient(
        {
            "cineseek_search": {
                "transport": "http",
                "url": resolved_mcp_url,
            }
        }
    )
    tools = await client.get_tools()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You help users search for movies. "
            "If the user's query is vague, rewrite it into a clearer movie search query before calling the tool. "
            "Always call the search_movies tool exactly once. "
            "After seeing results, choose the titles that should actually be shown to the user and order them from best match to weaker match. "
            "Use only titles that appear in the tool output. "
            f"If the query looks like a precise known-item search, return only 1 to 3 titles. "
            f"If the query is broader or exploratory, return 5 to {DEFAULT_AGENT_MAX_RESULTS} titles. "
            f"Never return more than {DEFAULT_AGENT_MAX_RESULTS} titles. "
            "Exclude clearly irrelevant candidates instead of keeping them just to fill space. "
            "Write a short overall summary of the final selected list. "
            "The summary should describe the strongest matches, briefly characterize the list as a whole, and avoid focusing only on the top rank. "
            "Keep the summary under 90 words. "
            "Return only valid JSON with this exact shape: "
            '{"selected_titles":["Title 1","Title 2"],"summary":"..."} '
            "Do not wrap the JSON in markdown fences."
        ),
    )
    return agent


async def _agent_recommend_async(raw_query: str, mcp_server_url: str = DEFAULT_MCP_SERVER_URL) -> dict[str, Any]:
    available, reason = agent_is_available()
    if not available:
        raise RuntimeError(reason or "Agent is unavailable.")

    agent = await _build_agent(mcp_server_url=mcp_server_url)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": raw_query}]})
    messages = result.get("messages", [])
    final_message = messages[-1] if messages else None
    summary = ""
    selected_titles: list[str] = []
    if final_message is not None:
        final_text = _message_text(final_message)
        payload = _extract_json_object(final_text)
        if payload is not None:
            selected_titles = [str(title) for title in payload.get("selected_titles", []) if str(title).strip()]
            summary = str(payload.get("summary", "")).strip()
        if not summary:
            summary = final_text

    tool_payload = _extract_tool_payload(messages) or {}
    query_used = tool_payload.get("query_used") or raw_query
    from apps.demo import network

    recommendations = network.direct_recommend(query_used, k=DEFAULT_AGENT_CANDIDATE_K)["recommendations"]
    if selected_titles:
        title_to_items: dict[str, list[dict[str, Any]]] = {}
        for item in recommendations:
            title_to_items.setdefault(item["title"], []).append(item)
        selected = []
        for title in selected_titles:
            items = title_to_items.get(title)
            if items:
                selected.append(items.pop(0))
        if selected:
            recommendations = selected
    recommendations = recommendations[:DEFAULT_AGENT_MAX_RESULTS]
    return {
        "query_used": query_used,
        "recommendations": recommendations,
        "agent_summary": summary,
        "agent_model": _provider_label(),
    }


def agent_recommend(raw_query: str, mcp_server_url: str = DEFAULT_MCP_SERVER_URL) -> dict[str, Any]:
    return asyncio.run(_agent_recommend_async(raw_query, mcp_server_url=mcp_server_url))
