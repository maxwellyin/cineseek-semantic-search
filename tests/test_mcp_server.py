"""Tests for the MCP search tool server (apps/demo/search_mcp_server.py).

These tests verify the MCP tool's output format contract without requiring
a full FAISS index. They mock the network.direct_recommend function.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from apps.demo.search_mcp_server import search_movies


def _mock_recommend(query: str, k: int = 12):
    return {
        "query_used": query,
        "recommendations": [
            {
                "title": f"Movie {i}",
                "score": 0.9 - i * 0.05,
                "structured": {
                    "release_year": "2020",
                    "genres": ["Sci-Fi", "Action"],
                    "overview": "A great movie about interesting things.",
                    "tags": ["exciting", "thoughtful"],
                },
            }
            for i in range(min(k, 5))
        ],
    }


class TestSearchMoviesTool:
    @patch("apps.demo.search_mcp_server.network.direct_recommend", side_effect=_mock_recommend)
    def test_returns_valid_json(self, mock_fn):
        result = search_movies("sci-fi movies", k=5)
        payload = json.loads(result)
        assert "query_used" in payload
        assert "recommendations" in payload
        assert isinstance(payload["recommendations"], list)

    @patch("apps.demo.search_mcp_server.network.direct_recommend", side_effect=_mock_recommend)
    def test_recommendation_fields(self, mock_fn):
        result = search_movies("test query", k=3)
        payload = json.loads(result)
        rec = payload["recommendations"][0]
        expected_keys = {"rank", "title", "score", "year", "genres", "overview", "tags"}
        assert expected_keys.issubset(set(rec.keys()))

    @patch("apps.demo.search_mcp_server.network.direct_recommend", side_effect=_mock_recommend)
    def test_rank_is_sequential(self, mock_fn):
        result = search_movies("test", k=5)
        payload = json.loads(result)
        ranks = [rec["rank"] for rec in payload["recommendations"]]
        assert ranks == list(range(1, len(ranks) + 1))

    @patch("apps.demo.search_mcp_server.network.direct_recommend", side_effect=_mock_recommend)
    def test_genres_capped(self, mock_fn):
        result = search_movies("test", k=1)
        payload = json.loads(result)
        for rec in payload["recommendations"]:
            assert len(rec["genres"]) <= 4

    @patch("apps.demo.search_mcp_server.network.direct_recommend", side_effect=_mock_recommend)
    def test_overview_truncated(self, mock_fn):
        result = search_movies("test", k=1)
        payload = json.loads(result)
        for rec in payload["recommendations"]:
            assert len(rec["overview"]) <= 220
