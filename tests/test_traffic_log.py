"""Tests for lazy traffic DB initialization and request tracking."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from apps.demo import traffic_log


def _fake_request(path: str = "/", query: str | None = None, use_agent: str | None = None):
    params = {}
    if query is not None:
        params["text"] = query
    if use_agent is not None:
        params["use_agent"] = use_agent
    return SimpleNamespace(
        method="GET",
        url=SimpleNamespace(path=path),
        headers={"user-agent": "pytest"},
        query_params=params,
        client=SimpleNamespace(host="203.0.113.10"),
    )


class TestTrafficLog:
    def test_record_request_initializes_db_lazily(self, monkeypatch, tmp_path):
        db_path = tmp_path / "traffic.sqlite3"
        monkeypatch.setattr(traffic_log, "TRAFFIC_DB_PATH", db_path)
        monkeypatch.setattr(traffic_log, "_db_initialized", False)
        monkeypatch.setattr(traffic_log, "_last_cleanup_at", None)
        monkeypatch.setattr(traffic_log, "_cached_geolocation", lambda ip: "Test Region")

        assert not db_path.exists()

        traffic_log.record_request(_fake_request("/", None, None), 200)

        assert db_path.exists()
        dashboard = traffic_log.fetch_dashboard(limit=10)
        assert dashboard["summary"]["last_7d_visits"] == 1

    def test_search_request_is_recorded(self, monkeypatch, tmp_path):
        db_path = tmp_path / "traffic.sqlite3"
        monkeypatch.setattr(traffic_log, "TRAFFIC_DB_PATH", db_path)
        monkeypatch.setattr(traffic_log, "_db_initialized", False)
        monkeypatch.setattr(traffic_log, "_last_cleanup_at", None)
        monkeypatch.setattr(traffic_log, "_cached_geolocation", lambda ip: "Test Region")

        traffic_log.record_request(_fake_request("/search/results", "dark sci-fi", "1"), 200)
        dashboard = traffic_log.fetch_dashboard(limit=10)

        assert dashboard["summary"]["last_7d_searches"] == 1
        assert dashboard["recent_events"][0]["query_text"] == "dark sci-fi"
