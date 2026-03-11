from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

try:
    from langchain.agents import create_agent
    from langchain.tools import tool
except ImportError:  # pragma: no cover - optional dependency
    create_agent = None
    tool = None

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


DEFAULT_AGENT_PROVIDER = os.environ.get("FLCR_AGENT_PROVIDER", "gemini").lower()
DEFAULT_OLLAMA_MODEL = os.environ.get("FLCR_OLLAMA_MODEL", "qwen3:8b")
DEFAULT_GEMINI_MODEL = os.environ.get("FLCR_GEMINI_MODEL", "gemini-2.5-flash-lite")
DEFAULT_OPENAI_MODEL = os.environ.get("FLCR_OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


class AgentSearchResponse(BaseModel):
    reranked_titles: list[str] = Field(
        description="Movie titles reordered from best match to weaker match using only titles returned by the search tool."
    )
    summary: str = Field(description="A short overall summary of the returned list that mentions the strongest matches and characterizes the rest of the list.")


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
    return f"ollama:{DEFAULT_OLLAMA_MODEL}"


def agent_is_available() -> tuple[bool, str | None]:
    if create_agent is None or tool is None:
        return False, "LangChain is not installed."
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
    if ChatOllama is None:
        return False, "langchain-ollama is not installed."
    return True, None


def _build_llm():
    if DEFAULT_AGENT_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(model=DEFAULT_GEMINI_MODEL, temperature=0)
    if DEFAULT_AGENT_PROVIDER == "openai":
        return ChatOpenAI(model=DEFAULT_OPENAI_MODEL, temperature=0)
    return ChatOllama(model=DEFAULT_OLLAMA_MODEL, temperature=0, base_url=DEFAULT_OLLAMA_BASE_URL)


@lru_cache(maxsize=1)
def _build_agent():
    llm = _build_llm()

    @tool
    def search_movies(query: str) -> str:
        """Search the movie index for the user's query and return the top matches."""
        from apps.demo import network

        result = network.direct_recommend(query, k=8)
        _build_agent.tool_state["query_used"] = result["query_used"]
        _build_agent.tool_state["recommendations"] = result["recommendations"]
        payload = []
        for idx, item in enumerate(result["recommendations"], start=1):
            payload.append(
                {
                    "rank": idx,
                    "title": item["title"],
                    "score": round(float(item["score"]), 4),
                    "metadata": item["metadata"][:400],
                }
            )
        return json.dumps(payload, ensure_ascii=False)

    agent = create_agent(
        model=llm,
        tools=[search_movies],
        response_format=AgentSearchResponse,
        system_prompt=(
            "You help users search for movies. "
            "If the user's query is vague, rewrite it into a clearer movie search query before calling the tool. "
            "Always call the search_movies tool exactly once. "
            "After seeing results, reorder the returned titles from best match to weaker match. "
            "Use only titles that appear in the tool output. "
            "Write a short overall summary of the returned list. "
            "The summary should describe the strongest matches, briefly characterize the list as a whole, and avoid focusing only on the top rank. "
            "Keep the summary under 90 words."
        ),
    )
    return agent


_build_agent.tool_state = {"query_used": None, "recommendations": None}


def agent_recommend(raw_query: str) -> dict[str, Any]:
    available, reason = agent_is_available()
    if not available:
        raise RuntimeError(reason or "Agent is unavailable.")

    _build_agent.tool_state = {"query_used": None, "recommendations": None}
    agent = _build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": raw_query}]})
    messages = result.get("messages", [])
    final_message = messages[-1] if messages else None
    structured = result.get("structured_response")
    summary = ""
    reranked_titles: list[str] = []
    if structured is not None:
        summary = structured.summary
        reranked_titles = list(structured.reranked_titles or [])
    elif final_message is not None:
        summary = _message_text(final_message)

    recommendations = _build_agent.tool_state.get("recommendations") or []
    if reranked_titles:
        title_to_items: dict[str, list[dict[str, Any]]] = {}
        for item in recommendations:
            title_to_items.setdefault(item["title"], []).append(item)
        reranked = []
        for title in reranked_titles:
            items = title_to_items.get(title)
            if items:
                reranked.append(items.pop(0))
        remaining = []
        for items in title_to_items.values():
            remaining.extend(items)
        recommendations = reranked + remaining
    query_used = _build_agent.tool_state.get("query_used") or raw_query
    return {
        "query_used": query_used,
        "recommendations": recommendations,
        "agent_summary": summary,
        "agent_model": _provider_label(),
    }
