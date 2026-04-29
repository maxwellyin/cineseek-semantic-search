"""Tests for network-layer retrieval logic (apps/demo/network.py).

These tests focus on the title-matching, reranking, and diversification
logic which does not require a loaded FAISS index or sentence model.
"""

from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from apps.demo.network import (
    candidate_bucket,
    content_tokens,
    diversify_recommendations,
    needs_diversity,
    normalize_title_text,
    parse_item_metadata,
    rerank_with_title_signal,
    similar_movies,
    title_match_score,
    title_signal_weight,
)


class TestParseItemMetadata:
    def test_empty_input(self):
        result = parse_item_metadata("")
        assert result["display_title"] == ""
        assert result["genres"] == []
        assert result["overview"] == ""

    def test_full_metadata(self):
        text = (
            "Inception (2010) "
            "genres: Sci-Fi, Thriller "
            "overview: A thief who steals secrets through dream-sharing. "
            "tags: mind-bending, dreams "
            "director: Christopher Nolan "
            "actors: Leonardo DiCaprio, Tom Hardy "
            "characters: Cobb, Eames"
        )
        result = parse_item_metadata(text)
        assert "Inception" in result["display_title"]
        assert result["release_year"] == "2010"
        assert "Sci-Fi" in result["genres"]
        assert result["director"] == "Christopher Nolan"
        assert len(result["actors"]) >= 2

    def test_missing_sections(self):
        text = "The Matrix (1999) genres: Action, Sci-Fi"
        result = parse_item_metadata(text)
        assert result["release_year"] == "1999"
        assert "Action" in result["genres"]
        assert result["overview"] == ""


class TestTitleMatchScore:
    def test_exact_match(self):
        assert title_match_score("inception", "Inception") == 1.0

    def test_no_match(self):
        assert title_match_score("inception", "The Godfather") == 0.0

    def test_substring_match(self):
        score = title_match_score("batman", "Batman Begins (2005)")
        assert score > 0.5

    def test_empty_strings(self):
        assert title_match_score("", "something") == 0.0
        assert title_match_score("something", "") == 0.0


class TestTitleSignalWeight:
    def test_exploratory_query_returns_zero(self):
        assert title_signal_weight("movies similar to inception") == 0.0
        assert title_signal_weight("recommend me something like batman") == 0.0

    def test_short_query_returns_nonzero(self):
        assert title_signal_weight("the matrix") > 0.0


class TestNeedsDiversity:
    def test_superhero_query(self):
        assert needs_diversity("superhero movies with marvel characters")

    def test_normal_query(self):
        assert not needs_diversity("romantic comedies about love")


class TestCandidateBucket:
    def test_marvel_movie(self):
        item = {
            "title": "Spider-Man: Homecoming",
            "structured": {"overview": "A young Spider-Man", "tags": [], "genres": [], "characters": ["Peter Parker"]},
        }
        assert candidate_bucket(item) == "spider_man"

    def test_non_franchise(self):
        item = {
            "title": "Inception",
            "structured": {"overview": "Dream heist", "tags": [], "genres": [], "characters": []},
        }
        assert candidate_bucket(item) is None


class TestDiversifyRecommendations:
    def _make_item(self, title, tags=None, characters=None):
        return {
            "title": title,
            "score": 1.0,
            "structured": {
                "overview": "",
                "tags": tags or [],
                "genres": [],
                "characters": characters or [],
            },
        }

    def test_no_diversity_needed(self):
        items = [self._make_item(f"Movie {i}") for i in range(10)]
        result = diversify_recommendations("romantic comedies", items, k=5)
        assert len(result) == 5

    def test_limits_same_franchise(self):
        items = [
            self._make_item("Spider-Man: Homecoming", characters=["Peter Parker"]),
            self._make_item("Spider-Man: Far From Home", characters=["Peter Parker"]),
            self._make_item("Spider-Man: No Way Home", characters=["Peter Parker"]),
            self._make_item("Batman Begins", characters=["Bruce Wayne"]),
            self._make_item("The Dark Knight", characters=["Bruce Wayne"]),
            self._make_item("Wonder Woman", characters=["Diana Prince"]),
        ]
        result = diversify_recommendations("superhero movies", items, k=4)
        titles = [r["title"] for r in result]
        spider_count = sum(1 for t in titles if "Spider-Man" in t)
        assert spider_count <= 2  # diversity should limit same franchise


class TestRerankWithTitleSignal:
    def test_preserves_all_items(self):
        items = [
            {"title": "The Matrix", "score": 0.8},
            {"title": "Inception", "score": 0.9},
        ]
        result = rerank_with_title_signal("inception", items)
        assert len(result) == 2
        assert all("semantic_score" in r for r in result)
        assert all("title_match_score" in r for r in result)


class TestSimilarMovies:
    def test_uses_item_neighbors(self, monkeypatch):
        dataset = {
            "item_titles": {1: "Seed Movie", 2: "Neighbor One", 3: "Neighbor Two"},
            "item_metadata_texts": {
                1: "Seed Movie genres: drama overview: Seed overview tags: emotional director: A actors: X characters: Y",
                2: "Neighbor One genres: drama overview: Neighbor overview tags: moody director: B actors: X characters: Y",
                3: "Neighbor Two genres: thriller overview: Another overview tags: dark director: C actors: X characters: Y",
            },
        }
        poster_urls = {}
        item_embeddings = torch.randn(3, 8)
        fake_index = SimpleNamespace(ntotal=3)

        monkeypatch.setattr(
            "apps.demo.network.load_assets",
            lambda: (dataset, fake_index, None, poster_urls, item_embeddings),
        )
        monkeypatch.setattr(
            "apps.demo.network.lookup_movie",
            lambda title: {"title": "Seed Movie", "item_idx": 1, "structured": {}, "poster_url": "", "metadata": ""},
        )
        monkeypatch.setattr(
            "apps.demo.network.search_index",
            lambda index, queries, k: (
                torch.tensor([[1.0, 0.9, 0.8]]).numpy(),
                torch.tensor([[0, 1, 2]]).numpy(),
            ),
        )

        result = similar_movies("Seed Movie", k=2)
        assert [item["title"] for item in result] == ["Neighbor One", "Neighbor Two"]
