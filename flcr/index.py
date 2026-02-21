from __future__ import annotations

import torch

from flcr.config import CHECKPOINT_PATH, DATASET_PATH, DEVICE, INDEX_METADATA_PATH, INDEX_PATH
from flcr.search import build_index, save_index
from flcr.train import build_model


def encode_all_items(model, num_items: int, batch_size: int = 2048):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for start in range(1, num_items + 1, batch_size):
            stop = min(start + batch_size, num_items + 1)
            item_ids = torch.arange(start, stop, device=DEVICE)
            all_embeddings.append(model.encode_items(item_ids).cpu())
    return torch.cat(all_embeddings, dim=0)


def main():
    dataset = torch.load(DATASET_PATH, map_location="cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    model = build_model(dataset)
    model.load_state_dict(checkpoint["state_dict"])
    embeddings = encode_all_items(model, dataset["num_items"])

    index = build_index(embeddings)
    save_index(index, INDEX_PATH)

    torch.save(
        {
            "item_titles": dataset["item_titles"],
            "idx_to_item_id": dataset["idx_to_item_id"],
            "embeddings_shape": tuple(embeddings.shape),
        },
        INDEX_METADATA_PATH,
    )
    print(f"Saved index to {INDEX_PATH}")
    print(f"Saved index metadata to {INDEX_METADATA_PATH}")


if __name__ == "__main__":
    main()
