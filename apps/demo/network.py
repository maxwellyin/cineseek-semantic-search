from __future__ import annotations

import csv
from functools import lru_cache
import re

import torch
from sentence_transformers import SentenceTransformer

from flcr.agent.langchain_agent import agent_is_available, agent_recommend
from flcr.config import DATASET_PATH, INDEX_PATH, ITEM_TABLE_PATH, SENTENCE_MODEL_DIR, SENTENCE_TRANSFORMER_DEVICE
from flcr.raw_retrieval import DEFAULT_RAW_MODE, build_raw_item_embeddings, build_raw_query_embeddings
from flcr.search import load_index, search_index


METADATA_LABELS = ["genres", "overview", "tags", "director", "actors", "characters"]
TITLE_STOPWORDS = {"a", "an", "and", "film", "films", "for", "movie", "movies", "of", "the", "to", "with"}
EXPLORATORY_QUERY_MARKERS = {
    "about",
    "characters",
    "featuring",
    "from",
    "genre",
    "like",
    "recommend",
    "recommendation",
    "recommendations",
    "similar",
    "superhero",
    "superheroes",
    "with",
}
DIVERSITY_QUERY_MARKERS = {
    "characters",
    "comic",
    "comics",
    "dc",
    "featuring",
    "franchise",
    "marvel",
    "superhero",
    "superheroes",
}

FRANCHISE_BUCKET_RULES = {
    "justice_league": ["justice league", "teen titans"],
    "batman": ["batman", "bruce wayne", "dark knight", "gotham"],
    "superman": ["superman", "clark kent", "krypton", "man of steel"],
    "wonder_woman": ["wonder woman", "diana prince", "amazon princess"],
    "aquaman": ["aquaman", "arthur curry", "atlantis"],
    "green_lantern": ["green lantern", "hal jordan", "john stewart"],
    "shazam": ["shazam"],
    "flash": ["flash", "barry allen"],
    "spider_man": ["spider man", "spider-man", "peter parker"],
    "iron_man": ["iron man", "tony stark"],
    "captain_america": ["captain america", "steve rogers"],
    "thor": ["thor", "asgard"],
    "avengers": ["avengers"],
    "x_men": ["x men", "x-men", "wolverine", "mutant"],
    "black_panther": ["black panther", "wakanda"],
    "guardians": ["guardians of the galaxy", "star lord", "star-lord"],
    "hulk": ["hulk", "bruce banner"],
    "deadpool": ["deadpool", "wade wilson"],
    "daredevil": ["daredevil", "matt murdock"],
}
DC_BUCKETS = {"aquaman", "batman", "flash", "green_lantern", "justice_league", "shazam", "superman", "wonder_woman"}
MARVEL_BUCKETS = {
    "avengers",
    "black_panther",
    "captain_america",
    "daredevil",
    "deadpool",
    "guardians",
    "hulk",
    "iron_man",
    "spider_man",
    "thor",
    "x_men",
}


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


def normalize_title_text(value: str) -> str:
    text = re.sub(r"\(\d{4}\)\s*$", "", value or "")
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(text.split())


def content_tokens(value: str) -> set[str]:
    return {token for token in normalize_title_text(value).split() if token not in TITLE_STOPWORDS and len(token) > 1}


def title_match_score(query: str, title: str) -> float:
    normalized_query = normalize_title_text(query)
    normalized_title = normalize_title_text(title)
    if not normalized_query or not normalized_title:
        return 0.0
    if normalized_query == normalized_title:
        return 1.0
    if normalized_query in normalized_title or normalized_title in normalized_query:
        return 0.72

    query_tokens = content_tokens(query)
    title_tokens = content_tokens(title)
    if not query_tokens or not title_tokens:
        return 0.0
    overlap = len(query_tokens & title_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(query_tokens)
    recall = overlap / len(title_tokens)
    return 0.5 * precision + 0.5 * recall


def title_signal_weight(query: str) -> float:
    normalized_query = normalize_title_text(query)
    tokens = set(normalized_query.split())
    content = content_tokens(query)
    if tokens & EXPLORATORY_QUERY_MARKERS:
        return 0.0
    if len(content) <= 4:
        return 0.18
    return 0.0


def rerank_with_title_signal(raw_text: str, recommendations: list[dict[str, object]]) -> list[dict[str, object]]:
    lexical_weight = title_signal_weight(raw_text)
    reranked = []
    for item in recommendations:
        lexical_score = title_match_score(raw_text, str(item["title"]))
        item["semantic_score"] = float(item["score"])
        item["title_match_score"] = lexical_score
        effective_weight = lexical_weight if lexical_score >= 0.6 else 0.0
        item["score"] = float(item["score"]) + (effective_weight * lexical_score)
        reranked.append(item)
    return sorted(reranked, key=lambda item: item["score"], reverse=True)


def needs_diversity(query: str) -> bool:
    tokens = set(normalize_title_text(query).split())
    return bool(tokens & DIVERSITY_QUERY_MARKERS)


def candidate_bucket(item: dict[str, object]) -> str | None:
    structured = item.get("structured") or {}
    parts = [
        str(item.get("title", "")),
        str(structured.get("overview", "")),
        " ".join(structured.get("tags", []) or []),
        " ".join(structured.get("genres", []) or []),
        " ".join(structured.get("characters", []) or []),
    ]
    text = normalize_title_text(" ".join(parts))
    padded_text = f" {text} "
    for bucket, markers in FRANCHISE_BUCKET_RULES.items():
        if any(f" {normalize_title_text(marker)} " in padded_text for marker in markers):
            return bucket
    return None


def diversify_recommendations(query: str, recommendations: list[dict[str, object]], k: int) -> list[dict[str, object]]:
    if not needs_diversity(query):
        return recommendations[:k]

    selected = []
    delayed = []
    bucket_counts: dict[str, int] = {}
    query_tokens = set(normalize_title_text(query).split())

    for item in recommendations:
        bucket = candidate_bucket(item)
        if ("dc" in query_tokens and bucket in MARVEL_BUCKETS) or ("marvel" in query_tokens and bucket in DC_BUCKETS):
            delayed.append(item)
            continue
        limit = 1 if len(selected) < max(4, k // 2) else 2
        if bucket and bucket_counts.get(bucket, 0) >= limit:
            delayed.append(item)
            continue
        selected.append(item)
        if bucket:
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if len(selected) >= k:
            break

    if len(selected) < k:
        selected.extend(delayed[: k - len(selected)])

    return selected[:k]


def load_poster_urls() -> dict[int, str]:
    poster_urls: dict[int, str] = {}
    with ITEM_TABLE_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            item_idx = row.get("item_idx")
            poster_url = (row.get("poster_url") or "").strip()
            if item_idx and poster_url:
                poster_urls[int(item_idx)] = poster_url
    return poster_urls


@lru_cache(maxsize=1)
def load_assets():
    dataset = torch.load(DATASET_PATH, map_location="cpu")
    index = load_index(INDEX_PATH)
    sentence_model = SentenceTransformer(str(SENTENCE_MODEL_DIR), device=SENTENCE_TRANSFORMER_DEVICE)
    poster_urls = load_poster_urls()
    item_embeddings = build_raw_item_embeddings(dataset, mode=DEFAULT_RAW_MODE).cpu()
    return dataset, index, sentence_model, poster_urls, item_embeddings


def build_recommendation(dataset: dict, poster_urls: dict[int, str], item_idx: int, score: float) -> dict[str, object]:
    metadata = dataset["item_metadata_texts"][item_idx]
    return {
        "title": dataset["item_titles"][item_idx],
        "metadata": metadata,
        "structured": parse_item_metadata(metadata),
        "poster_url": poster_urls.get(item_idx, ""),
        "score": float(score),
    }


def lookup_movie(title: str) -> dict[str, object] | None:
    dataset, _, _, poster_urls, _ = load_assets()
    wanted = (title or "").strip()
    if not wanted:
        return None

    normalized_wanted = normalize_title_text(wanted)
    fallback_match = None

    for item_idx in range(1, len(dataset["item_titles"])):
        item_title = dataset["item_titles"][item_idx]
        if item_title == wanted:
            movie = build_recommendation(dataset, poster_urls, item_idx, 1.0)
            movie["item_idx"] = item_idx
            return movie
        if fallback_match is None and normalize_title_text(item_title) == normalized_wanted:
            fallback_match = item_idx

    if fallback_match is None:
        return None

    movie = build_recommendation(dataset, poster_urls, fallback_match, 1.0)
    movie["item_idx"] = fallback_match
    return movie


def similar_movies(title: str, k: int = 6) -> list[dict[str, object]]:
    movie = lookup_movie(title)
    if movie is None:
        return []

    dataset, index, _, poster_urls, item_embeddings = load_assets()
    seed_idx = int(movie["item_idx"])
    seed_vector = item_embeddings[seed_idx - 1 : seed_idx]
    scores, idxes = search_index(index, seed_vector, k=min(max(k + 1, 8), index.ntotal))

    recommendations = []
    for score, idx in zip(scores[0].tolist(), idxes[0].tolist()):
        item_idx = idx + 1
        if item_idx == seed_idx:
            continue
        recommendations.append(build_recommendation(dataset, poster_urls, item_idx, float(score)))
        if len(recommendations) >= k:
            break

    return recommendations


def direct_recommend(raw_text: str, k: int = 12):
    dataset, index, sentence_model, poster_urls, _ = load_assets()
    query_embedding = sentence_model.encode(
        [raw_text],
        batch_size=1,
        show_progress_bar=False,
        convert_to_tensor=True,
        normalize_embeddings=False,
    )
    with torch.no_grad():
        query_repr = build_raw_query_embeddings(query_embedding).cpu()
    search_k = min(max(k, 50), index.ntotal)
    scores, idxes = search_index(index, query_repr, k=search_k)

    recommendations = []
    for score, idx in zip(scores[0].tolist(), idxes[0].tolist()):
        item_idx = idx + 1
        recommendations.append(build_recommendation(dataset, poster_urls, item_idx, float(score)))
    recommendations = rerank_with_title_signal(raw_text, recommendations)
    recommendations = diversify_recommendations(raw_text, recommendations, k)
    return {"query_used": raw_text, "recommendations": recommendations}


def health_status() -> dict[str, object]:
    try:
        dataset, index, sentence_model, poster_urls, _ = load_assets()
        agent_available, agent_reason = agent_is_available()
        return {
            "status": "ok",
            "retriever": {
                "type": "raw_sentence_transformer",
                "mode": DEFAULT_RAW_MODE,
                "dataset_items": int(dataset.get("num_items", len(dataset.get("item_titles", {})))),
                "index_items": int(index.ntotal),
                "poster_items": len(poster_urls),
                "sentence_model": str(SENTENCE_MODEL_DIR.name),
            },
            "agent": {
                "available": agent_available,
                "reason": agent_reason,
            },
        }
    except Exception as exc:  # pragma: no cover - runtime diagnostic endpoint
        return {
            "status": "error",
            "error": str(exc),
        }


def format_agent_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "resource_exhausted" in lowered or "quota exceeded" in lowered or "429" in lowered or "rate limit" in lowered:
        return "The current LLM quota or rate limit has been reached. Showing direct retrieval results instead."
    if "google_api_key" in lowered or "gemini_api_key" in lowered:
        return "Gemini is not configured on this deployment. Showing direct retrieval results instead."
    if "groq_api_key" in lowered:
        return "Groq is not configured on this deployment. Showing direct retrieval results instead."
    if "openai_api_key" in lowered:
        return "OpenAI is not configured on this deployment. Showing direct retrieval results instead."
    if "timed out" in lowered or "timeout" in lowered:
        return "The agent took too long to respond. Showing direct retrieval results instead."
    return "The agent is temporarily unavailable. Showing direct retrieval results instead."


def recommend(raw_text: str, k: int = 12, use_agent: bool = False, mcp_server_url: str | None = None):
    if use_agent:
        available, reason = agent_is_available()
        if available:
            try:
                return agent_recommend(raw_text, mcp_server_url=mcp_server_url or "")
            except Exception as exc:  # pragma: no cover - defensive runtime fallback
                fallback = direct_recommend(raw_text, k=k)
                fallback["agent_error"] = format_agent_error(exc)
                return fallback
        fallback = direct_recommend(raw_text, k=k)
        fallback["agent_error"] = reason
        return fallback
    return direct_recommend(raw_text, k=k)
