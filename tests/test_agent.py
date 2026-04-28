"""Tests for agent timeout and fallback-sensitive behavior."""

from __future__ import annotations

import asyncio

import pytest

from flcr.agent import langchain_agent


class _SlowAgent:
    async def ainvoke(self, payload):
        await asyncio.sleep(0.05)
        return {"messages": []}


class TestAgentTimeout:
    def test_agent_call_respects_global_timeout(self, monkeypatch):
        monkeypatch.setattr(langchain_agent, "DEFAULT_AGENT_TIMEOUT", 0)
        monkeypatch.setattr(langchain_agent, "agent_is_available", lambda: (True, None))

        async def fake_build_agent(mcp_server_url: str = ""):
            return _SlowAgent()

        monkeypatch.setattr(langchain_agent, "_build_agent", fake_build_agent)

        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(langchain_agent._agent_recommend_async("test query"))
