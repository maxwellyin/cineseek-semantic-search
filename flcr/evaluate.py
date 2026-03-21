from __future__ import annotations

import math

import torch

from flcr.config import DATASET_PATH, INDEX_PATH
from flcr.raw_retrieval import build_raw_query_embeddings
from flcr.search import load_index, search_index


def _singleton_positive_ids(target_ids: torch.Tensor) -> list[list[int]]:
    return [[int(target_id)] for target_id in target_ids.tolist()]


def evaluate_topk(query_embeddings: torch.Tensor, positive_ids: list[list[int]], index, k_values=(10, 50, 100)):
    hit_counts = {k: 0 for k in k_values}
    reciprocal_ranks = []
    ndcgs = []
    total = 0

    with torch.no_grad():
        query_repr = build_raw_query_embeddings(query_embeddings).cpu()
        _, idxes = search_index(index, query_repr, k=max(k_values))
        predicted_ids = idxes + 1

    for ranked_tensor, positives in zip(predicted_ids, positive_ids):
        ranked = ranked_tensor.tolist()
        positive_set = set(int(item_id) for item_id in positives)
        total += 1
        for k in k_values:
            if any(item_id in positive_set for item_id in ranked[:k]):
                hit_counts[k] += 1
        matching_ranks = [rank + 1 for rank, item_id in enumerate(ranked) if item_id in positive_set]
        if matching_ranks:
            best_rank = min(matching_ranks)
            reciprocal_ranks.append(1.0 / best_rank)
            ndcgs.append(1.0 / math.log2(best_rank + 1))
        else:
            reciprocal_ranks.append(0.0)
            ndcgs.append(0.0)

    metrics = {f"recall@{k}": hit_counts[k] / total for k in k_values}
    metrics["mrr"] = sum(reciprocal_ranks) / total
    metrics["ndcg"] = sum(ndcgs) / total
    return metrics


def main():
    dataset = torch.load(DATASET_PATH, map_location="cpu")
    index = load_index(INDEX_PATH)

    val_positive_ids = dataset.get("val_positive_ids")
    if val_positive_ids is None:
        val_positive_ids = _singleton_positive_ids(dataset["val_target_ids"])
    test_positive_ids = dataset.get("test_positive_ids")
    if test_positive_ids is None:
        test_positive_ids = _singleton_positive_ids(dataset["test_target_ids"])

    val_metrics = evaluate_topk(dataset["val_query_embeddings"], positive_ids=val_positive_ids, index=index)
    test_metrics = evaluate_topk(dataset["test_query_embeddings"], positive_ids=test_positive_ids, index=index)
    print("Validation:", val_metrics)
    print("Test:", test_metrics)


if __name__ == "__main__":
    main()
