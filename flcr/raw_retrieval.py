from __future__ import annotations

import torch
import torch.nn.functional as F


DEFAULT_RAW_MODE = "title_overview_avg"


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings.float(), dim=-1)


def build_raw_item_embeddings(dataset: dict, mode: str = DEFAULT_RAW_MODE) -> torch.Tensor:
    if mode == "title":
        return normalize_embeddings(dataset["item_title_embeddings"][1:])
    if mode in {"overview", "metadata"}:
        key = "item_metadata_embeddings" if "item_metadata_embeddings" in dataset else "item_overview_embeddings"
        return normalize_embeddings(dataset[key][1:])
    if mode in {"title_overview_avg", "title_metadata_avg"}:
        title = normalize_embeddings(dataset["item_title_embeddings"][1:])
        metadata_key = "item_metadata_embeddings" if "item_metadata_embeddings" in dataset else "item_overview_embeddings"
        metadata = normalize_embeddings(dataset[metadata_key][1:])
        return normalize_embeddings((title + metadata) / 2.0)
    raise ValueError(f"Unsupported raw embedding mode: {mode}")


def build_raw_query_embeddings(query_embeddings: torch.Tensor) -> torch.Tensor:
    return normalize_embeddings(query_embeddings)
