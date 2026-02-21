from __future__ import annotations

from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

from flcr.config import CHECKPOINT_PATH, DATASET_PATH, DEVICE, INDEX_PATH, SENTENCE_MODEL_DIR, SENTENCE_TRANSFORMER_DEVICE
from flcr.search import load_index, search_index
from flcr.train import build_model


@lru_cache(maxsize=1)
def load_assets():
    dataset = torch.load(DATASET_PATH, map_location="cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = build_model(dataset)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    index = load_index(INDEX_PATH)
    sentence_model = SentenceTransformer(str(SENTENCE_MODEL_DIR), device=SENTENCE_TRANSFORMER_DEVICE)
    return dataset, model, index, sentence_model


def recommend(raw_text: str, k: int = 12):
    dataset, model, index, sentence_model = load_assets()
    query_embedding = sentence_model.encode(
        [raw_text],
        batch_size=1,
        show_progress_bar=False,
        convert_to_tensor=True,
        normalize_embeddings=False,
    ).to(DEVICE)
    with torch.no_grad():
        query_repr = model.encode_queries(query_embedding).cpu()
    scores, idxes = search_index(index, query_repr, k=k)

    recommendations = []
    for score, idx in zip(scores[0].tolist(), idxes[0].tolist()):
        item_idx = idx + 1
        recommendations.append(
            {
                "title": dataset["item_titles"][item_idx],
                "metadata": dataset["item_metadata_texts"][item_idx],
                "score": float(score),
            }
        )
    return {"recommendations": recommendations}
