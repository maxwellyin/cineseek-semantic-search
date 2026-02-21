from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
import torch

try:
    faiss.omp_set_num_threads(1)
except AttributeError:
    pass


def to_faiss_array(vectors) -> np.ndarray:
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().cpu().numpy()
    return np.ascontiguousarray(vectors, dtype=np.float32)


def build_index(vectors) -> faiss.Index:
    matrix = to_faiss_array(vectors)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))


def search_index(index: faiss.Index, queries, k: int):
    query_matrix = to_faiss_array(queries)
    capped_k = min(k, index.ntotal)
    scores, idxes = index.search(query_matrix, capped_k)
    return scores, idxes
