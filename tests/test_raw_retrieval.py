"""Tests for raw retrieval embedding construction (flcr/raw_retrieval.py)."""

from __future__ import annotations

import pytest
import torch

from flcr.raw_retrieval import build_raw_item_embeddings, build_raw_query_embeddings, normalize_embeddings


class TestNormalizeEmbeddings:
    def test_unit_norm(self):
        embeddings = torch.randn(10, 32)
        normalized = normalize_embeddings(embeddings)
        norms = torch.norm(normalized, dim=-1)
        torch.testing.assert_close(norms, torch.ones(10), atol=1e-5, rtol=0)

    def test_preserves_direction(self):
        embeddings = torch.tensor([[3.0, 4.0]])
        normalized = normalize_embeddings(embeddings)
        expected = torch.tensor([[0.6, 0.8]])
        torch.testing.assert_close(normalized, expected, atol=1e-5, rtol=0)


class TestBuildRawItemEmbeddings:
    @pytest.fixture()
    def mock_dataset(self):
        num_items = 11  # index 0 is padding
        dim = 16
        return {
            "item_title_embeddings": torch.randn(num_items, dim),
            "item_metadata_embeddings": torch.randn(num_items, dim),
        }

    def test_title_mode(self, mock_dataset):
        result = build_raw_item_embeddings(mock_dataset, mode="title")
        assert result.shape == (10, 16)  # excludes index 0

    def test_metadata_mode(self, mock_dataset):
        result = build_raw_item_embeddings(mock_dataset, mode="metadata")
        assert result.shape == (10, 16)

    def test_title_metadata_avg_mode(self, mock_dataset):
        result = build_raw_item_embeddings(mock_dataset, mode="title_metadata_avg")
        assert result.shape == (10, 16)
        norms = torch.norm(result, dim=-1)
        torch.testing.assert_close(norms, torch.ones(10), atol=1e-5, rtol=0)

    def test_fallback_to_overview(self):
        dataset = {
            "item_title_embeddings": torch.randn(5, 8),
            "item_overview_embeddings": torch.randn(5, 8),
        }
        result = build_raw_item_embeddings(dataset, mode="overview")
        assert result.shape == (4, 8)

    def test_invalid_mode(self, mock_dataset):
        with pytest.raises(ValueError, match="Unsupported"):
            build_raw_item_embeddings(mock_dataset, mode="invalid")


class TestBuildRawQueryEmbeddings:
    def test_output_normalized(self):
        queries = torch.randn(5, 16)
        result = build_raw_query_embeddings(queries)
        norms = torch.norm(result, dim=-1)
        torch.testing.assert_close(norms, torch.ones(5), atol=1e-5, rtol=0)
