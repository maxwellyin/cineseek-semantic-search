from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from flcr.config import DATASET_PATH


def evaluate_topk(query_embeddings: torch.Tensor, item_embeddings: torch.Tensor, positive_ids: list[list[int]], k_values=(10, 50, 100)):
    query_repr = F.normalize(query_embeddings.float(), dim=-1)
    item_repr = F.normalize(item_embeddings[1:].float(), dim=-1)
    scores = query_repr @ item_repr.T
    _, idxes = torch.topk(scores, k=max(k_values), dim=1)
    predicted_ids = idxes + 1

    hit_counts = {k: 0 for k in k_values}
    reciprocal_ranks = []
    ndcgs = []
    total = 0

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


def item_embeddings_for_mode(dataset: dict, mode: str) -> torch.Tensor:
    if mode == "metadata":
        if "item_metadata_embeddings" in dataset:
            return dataset["item_metadata_embeddings"]
        return dataset["item_overview_embeddings"]
    if mode == "overview":
        return dataset["item_overview_embeddings"]
    if mode == "title":
        return dataset["item_title_embeddings"]
    if mode in {"title_metadata_avg", "title_overview_avg"}:
        title = F.normalize(dataset["item_title_embeddings"].float(), dim=-1)
        metadata_key = "item_metadata_embeddings" if "item_metadata_embeddings" in dataset else "item_overview_embeddings"
        metadata = F.normalize(dataset[metadata_key].float(), dim=-1)
        fused = F.normalize((title + metadata) / 2.0, dim=-1)
        return fused
    raise ValueError(f"Unsupported mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["metadata", "overview", "title", "title_metadata_avg", "title_overview_avg"], default="metadata")
    args = parser.parse_args()

    dataset = torch.load(DATASET_PATH, map_location="cpu")
    item_embeddings = item_embeddings_for_mode(dataset, args.mode)
    val_metrics = evaluate_topk(dataset["val_query_embeddings"], item_embeddings, dataset["val_positive_ids"])
    test_metrics = evaluate_topk(dataset["test_query_embeddings"], item_embeddings, dataset["test_positive_ids"])
    print(f"Raw sentence-transformer baseline ({args.mode})")
    print("Validation:", val_metrics)
    print("Test:", test_metrics)


if __name__ == "__main__":
    main()
