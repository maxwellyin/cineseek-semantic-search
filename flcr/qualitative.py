from __future__ import annotations

import argparse

import torch
from sentence_transformers import SentenceTransformer

from flcr.config import CHECKPOINT_PATH, DATASET_PATH, DEVICE, INDEX_PATH, SENTENCE_MODEL_DIR, SENTENCE_TRANSFORMER_DEVICE
from flcr.search import load_index, search_index
from flcr.train import build_model


def load_assets():
    dataset = torch.load(DATASET_PATH, map_location="cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = build_model(dataset)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    index = load_index(INDEX_PATH)
    sentence_model = SentenceTransformer(str(SENTENCE_MODEL_DIR), device=SENTENCE_TRANSFORMER_DEVICE)
    return dataset, model, index, sentence_model


def recommend(query: str, k: int = 10):
    dataset, model, index, sentence_model = load_assets()
    query_embedding = sentence_model.encode(
        [query],
        batch_size=1,
        show_progress_bar=False,
        convert_to_tensor=True,
        normalize_embeddings=False,
    ).to(DEVICE)
    with torch.no_grad():
        query_repr = model.encode_queries(query_embedding).cpu()
    _, idxes = search_index(index, query_repr, k=k)
    return [dataset["item_titles"][idx + 1] for idx in idxes[0].tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="the matrix movie")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    print("Query:", args.query)
    for rank, title in enumerate(recommend(args.query, k=args.k), start=1):
        print(f"{rank}. {title}")


if __name__ == "__main__":
    main()
