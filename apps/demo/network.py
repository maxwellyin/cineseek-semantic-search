from __future__ import annotations

from functools import lru_cache
import re

import torch
from sentence_transformers import SentenceTransformer

from flcr.agent.langchain_agent import agent_is_available, agent_recommend
from flcr.config import CHECKPOINT_PATH, DATASET_PATH, DEVICE, INDEX_PATH, SENTENCE_MODEL_DIR, SENTENCE_TRANSFORMER_DEVICE
from flcr.search import load_index, search_index
from flcr.train import build_model


METADATA_LABELS = ["genres", "overview", "tags", "director", "actors", "characters"]


def parse_item_metadata(raw_text: str) -> dict[str, object]:
    text = (raw_text or "").strip()
    if not text:
        return {
            "display_title": "",
            "release_year": "",
            "overview": "",
            "genres": [],
            "tags": [],
            "director": "",
            "actors": [],
            "characters": [],
            "raw_text": "",
        }

    pattern = re.compile(r"(genres|overview|tags|director|actors|characters):")
    matches = list(pattern.finditer(text))
    title_part = text[: matches[0].start()].strip(" .") if matches else text

    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[match.group(1)] = text[start:end].strip(" .")

    year_match = re.search(r"\((\d{4})\)\s*$", title_part)
    release_year = year_match.group(1) if year_match else ""
    display_title = title_part

    def split_csv(value: str) -> list[str]:
        return [part.strip() for part in value.split(",") if part.strip()]

    return {
        "display_title": display_title,
        "release_year": release_year,
        "overview": sections.get("overview", ""),
        "genres": split_csv(sections.get("genres", "")),
        "tags": split_csv(sections.get("tags", ""))[:8],
        "director": sections.get("director", ""),
        "actors": split_csv(sections.get("actors", ""))[:6],
        "characters": split_csv(sections.get("characters", ""))[:6],
        "raw_text": text,
    }


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


def direct_recommend(raw_text: str, k: int = 12):
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
                "structured": parse_item_metadata(dataset["item_metadata_texts"][item_idx]),
                "score": float(score),
            }
        )
    return {"query_used": raw_text, "recommendations": recommendations}


def recommend(raw_text: str, k: int = 12, use_agent: bool = False):
    if use_agent:
        available, reason = agent_is_available()
        if available:
            try:
                return agent_recommend(raw_text)
            except Exception as exc:  # pragma: no cover - defensive runtime fallback
                fallback = direct_recommend(raw_text, k=k)
                fallback["agent_error"] = f"Agent fallback: {exc}"
                return fallback
        fallback = direct_recommend(raw_text, k=k)
        fallback["agent_error"] = reason
        return fallback
    return direct_recommend(raw_text, k=k)
