"""Smoke tests for FastAPI app routes (apps/demo/app.py).

These tests verify that the app can be imported, routes exist,
and basic endpoints respond correctly without requiring a loaded
FAISS index or sentence model.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.demo.app import app


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


class TestAppRoutes:
    def test_expected_routes_exist(self):
        paths = {getattr(route, "path", "") for route in app.routes}
        expected = {"/", "/about", "/search", "/search/results", "/movie", "/health", "/ops/traffic", "/api/config", "/api/search", "/api/movie"}
        missing = expected - paths
        assert not missing, f"Missing routes: {missing}"

    def test_home_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "CineSeek" in response.text

    def test_search_page(self, client):
        response = client.get("/search")
        assert response.status_code == 200
        assert "Search" in response.text

    def test_about_page(self, client):
        response = client.get("/about")
        assert response.status_code == 200
        assert "CineSeek" in response.text

    def test_head_home(self, client):
        response = client.head("/")
        assert response.status_code == 200

    def test_favicon_redirect(self, client):
        response = client.get("/favicon.ico", follow_redirects=False)
        assert response.status_code == 308
        assert "/favicon.svg" in response.headers.get("location", "")


class TestMCPAuth:
    def test_public_mcp_rejects_without_token(self, client):
        response = client.get("/mcp/search/mcp")
        assert response.status_code == 401

    def test_public_mcp_rejects_wrong_token(self, client):
        response = client.get(
            "/mcp/search/mcp",
            headers={"Authorization": "Bearer wrong_token"},
        )
        assert response.status_code == 401
