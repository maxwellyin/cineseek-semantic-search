from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from flcr.config import BATCH_SIZE, CHECKPOINT_PATH, DATASET_PATH, DEVICE, INDEX_PATH, NUM_WORKERS
from flcr.model import TextItemTrainDataset
from flcr.search import load_index, search_index
from flcr.train import build_model


def _singleton_positive_ids(target_ids: torch.Tensor) -> list[list[int]]:
    return [[int(target_id)] for target_id in target_ids.tolist()]


def evaluate_topk(model, query_embeddings: torch.Tensor, positive_ids: list[list[int]], index, k_values=(10, 50, 100)):
    loader = DataLoader(
        TextItemTrainDataset(query_embeddings, torch.arange(query_embeddings.shape[0], dtype=torch.long)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    hit_counts = {k: 0 for k in k_values}
    reciprocal_ranks = []
    ndcgs = []
    total = 0

    model.eval()
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch_query_embeddings = batch["query_embeddings"].to(DEVICE)
            batch_positive_ids = positive_ids[offset : offset + batch_query_embeddings.shape[0]]
            offset += batch_query_embeddings.shape[0]
            query_repr = model.encode_queries(batch_query_embeddings).cpu()

            _, idxes = search_index(index, query_repr, k=max(k_values))
            predicted_ids = idxes + 1

            for i, positives in enumerate(batch_positive_ids):
                ranked = predicted_ids[i].tolist()
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
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = build_model(dataset)
    model.load_state_dict(checkpoint["state_dict"])
    index = load_index(INDEX_PATH)

    val_positive_ids = dataset.get("val_positive_ids")
    if val_positive_ids is None:
        val_positive_ids = _singleton_positive_ids(dataset["val_target_ids"])
    test_positive_ids = dataset.get("test_positive_ids")
    if test_positive_ids is None:
        test_positive_ids = _singleton_positive_ids(dataset["test_target_ids"])

    val_metrics = evaluate_topk(model, dataset["val_query_embeddings"], positive_ids=val_positive_ids, index=index)
    test_metrics = evaluate_topk(model, dataset["test_query_embeddings"], positive_ids=test_positive_ids, index=index)
    print("Validation:", val_metrics)
    print("Test:", test_metrics)


if __name__ == "__main__":
    main()
