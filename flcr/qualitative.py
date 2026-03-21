from __future__ import annotations

import argparse

import torch
from sentence_transformers import SentenceTransformer

from flcr.config import DATASET_PATH, INDEX_PATH, SENTENCE_MODEL_DIR, SENTENCE_TRANSFORMER_DEVICE
from flcr.raw_retrieval import build_raw_query_embeddings
from flcr.search import load_index, search_index


def load_assets():
    dataset = torch.load(DATASET_PATH, map_location="cpu")
    index = load_index(INDEX_PATH)
    sentence_model = SentenceTransformer(str(SENTENCE_MODEL_DIR), device=SENTENCE_TRANSFORMER_DEVICE)
    return dataset, index, sentence_model


def recommend(query: str, k: int = 10):
    dataset, index, sentence_model = load_assets()
    query_embedding = sentence_model.encode(
        [query],
        batch_size=1,
        show_progress_bar=False,
        convert_to_tensor=True,
        normalize_embeddings=False,
    )
    with torch.no_grad():
        query_repr = build_raw_query_embeddings(query_embedding).cpu()
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
