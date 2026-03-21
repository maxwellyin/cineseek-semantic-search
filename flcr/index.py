from __future__ import annotations

import torch

from flcr.config import DATASET_PATH, INDEX_METADATA_PATH, INDEX_PATH
from flcr.raw_retrieval import DEFAULT_RAW_MODE, build_raw_item_embeddings
from flcr.search import build_index, save_index


def main():
    dataset = torch.load(DATASET_PATH, map_location="cpu")
    embeddings = build_raw_item_embeddings(dataset, mode=DEFAULT_RAW_MODE)

    index = build_index(embeddings)
    save_index(index, INDEX_PATH)

    torch.save(
        {
            "item_titles": dataset["item_titles"],
            "idx_to_item_id": dataset["idx_to_item_id"],
            "embeddings_shape": tuple(embeddings.shape),
            "retriever": "raw_sentence_transformer",
            "mode": DEFAULT_RAW_MODE,
        },
        INDEX_METADATA_PATH,
    )
    print(f"Saved index to {INDEX_PATH}")
    print(f"Saved index metadata to {INDEX_METADATA_PATH}")


if __name__ == "__main__":
    main()
