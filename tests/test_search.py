"""Tests for the FAISS search module (flcr/search.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flcr.search import build_index, load_index, save_index, search_index, to_faiss_array


class TestToFaissArray:
    def test_from_numpy(self):
        arr = np.random.randn(5, 8).astype(np.float64)
        result = to_faiss_array(arr)
        assert result.dtype == np.float32
        assert result.flags["C_CONTIGUOUS"]
        assert result.shape == (5, 8)

    def test_from_torch_tensor(self):
        tensor = torch.randn(3, 4)
        result = to_faiss_array(tensor)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (3, 4)


class TestBuildAndSearchIndex:
    @pytest.fixture()
    def sample_vectors(self):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((20, 16)).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

    def test_build_index_size(self, sample_vectors):
        index = build_index(sample_vectors)
        assert index.ntotal == 20

    def test_search_returns_correct_shape(self, sample_vectors):
        index = build_index(sample_vectors)
        query = sample_vectors[:2]
        scores, idxes = search_index(index, query, k=5)
        assert scores.shape == (2, 5)
        assert idxes.shape == (2, 5)

    def test_self_retrieval(self, sample_vectors):
        """Each vector should retrieve itself as the top-1 result."""
        index = build_index(sample_vectors)
        scores, idxes = search_index(index, sample_vectors, k=1)
        for i in range(len(sample_vectors)):
            assert idxes[i, 0] == i

    def test_k_capped_at_ntotal(self, sample_vectors):
        index = build_index(sample_vectors)
        scores, idxes = search_index(index, sample_vectors[:1], k=9999)
        assert idxes.shape[1] == 20

    def test_save_and_load(self, sample_vectors, tmp_path):
        index = build_index(sample_vectors)
        path = tmp_path / "test.faiss"
        save_index(index, path)
        loaded = load_index(path)
        assert loaded.ntotal == index.ntotal

        scores_orig, idxes_orig = search_index(index, sample_vectors[:1], k=5)
        scores_loaded, idxes_loaded = search_index(loaded, sample_vectors[:1], k=5)
        np.testing.assert_array_equal(idxes_orig, idxes_loaded)
