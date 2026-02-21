from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from flcr.config import (
    DATASET_PATH,
    ITEM_METADATA_EMBEDDINGS_PATH,
    ITEM_TABLE_PATH,
    ITEM_TITLE_EMBEDDINGS_PATH,
    MAX_METADATA_CHARS,
    MIN_OVERVIEW_CHARS,
    MIN_QUERY_CHARS,
    QUERY_EMBEDDINGS_PATH,
    QUERY_TABLE_PATH,
    RANDOM_SEED,
    RAW_MSRD_MOVIES_PATH,
    RAW_MSRD_QUERIES_PATH,
    SENTENCE_MODEL_DIR,
    SENTENCE_TRANSFORMER_DEVICE,
    ensure_directories,
)


def load_sentence_model() -> SentenceTransformer:
    return SentenceTransformer(str(SENTENCE_MODEL_DIR), device=SENTENCE_TRANSFORMER_DEVICE)


def sanitize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    return " ".join(text.split())


def build_title_text(title: str, year) -> str:
    year_text = sanitize_text(year)
    if year_text and year_text != "0":
        return f"{title} ({year_text})"
    return title


def build_metadata_text(row: pd.Series) -> str:
    pieces = [
        row["title_text"],
        f"genres: {row['genres']}" if row["genres"] else "",
        f"overview: {row['overview']}" if row["overview"] else "",
        f"tags: {row['tags']}" if row["tags"] else "",
        f"director: {row['director']}" if row["director"] else "",
        f"actors: {row['actors']}" if row["actors"] else "",
        f"characters: {row['characters']}" if row["characters"] else "",
    ]
    text = " ".join(piece for piece in pieces if piece).strip()
    if len(text) > MAX_METADATA_CHARS:
        text = text[:MAX_METADATA_CHARS].rsplit(" ", 1)[0].strip()
    return text


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 256) -> torch.Tensor:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=False,
    ).cpu()


def load_movies() -> pd.DataFrame:
    movies = pd.read_csv(RAW_MSRD_MOVIES_PATH, sep="\t", compression="gzip")
    for column in ["title", "overview", "tags", "genres", "director", "actors", "characters"]:
        movies[column] = movies[column].map(sanitize_text)
    movies = movies[movies["title"].map(bool)].copy()
    movies["title_text"] = movies.apply(lambda row: build_title_text(row["title"], row["year"]), axis=1)
    movies["metadata_text"] = movies.apply(build_metadata_text, axis=1)
    movies = movies.reset_index(drop=True)
    movies["item_idx"] = np.arange(1, len(movies) + 1)
    return movies


def load_query_groups(item_id_to_idx: dict[int, int], max_queries: int | None) -> pd.DataFrame:
    queries = pd.read_csv(RAW_MSRD_QUERIES_PATH, sep="\t", compression="gzip")
    queries["query"] = queries["query"].map(sanitize_text)
    queries = queries[(queries["label"] > 0) & (queries["query"].str.len() >= MIN_QUERY_CHARS)].copy()
    queries["target_item_id"] = queries["id"].map(item_id_to_idx)
    queries = queries[queries["target_item_id"].notna()].copy()
    queries["target_item_id"] = queries["target_item_id"].astype(int)

    grouped = (
        queries.groupby("query", as_index=False)["target_item_id"]
        .agg(lambda values: sorted(set(int(value) for value in values)))
    )
    grouped = grouped[grouped["target_item_id"].map(len) > 0].copy()

    if max_queries is not None and max_queries < len(grouped):
        grouped = grouped.sample(n=max_queries, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        grouped = grouped.reset_index(drop=True)

    return grouped


def load_or_build_item_embeddings(
    model: SentenceTransformer,
    item_table: pd.DataFrame,
    refresh_embeddings: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        not refresh_embeddings
        and ITEM_TITLE_EMBEDDINGS_PATH.exists()
        and ITEM_METADATA_EMBEDDINGS_PATH.exists()
    ):
        title_embeddings = torch.load(ITEM_TITLE_EMBEDDINGS_PATH, map_location="cpu")
        metadata_embeddings = torch.load(ITEM_METADATA_EMBEDDINGS_PATH, map_location="cpu")
        if title_embeddings.shape[0] == len(item_table) + 1 and metadata_embeddings.shape[0] == len(item_table) + 1:
            return title_embeddings, metadata_embeddings

    encoded_titles = encode_texts(model, item_table["title_text"].tolist())
    encoded_metadata = encode_texts(model, item_table["metadata_text"].tolist())
    title_embeddings = torch.zeros((len(item_table) + 1, encoded_titles.shape[1]), dtype=torch.float32)
    metadata_embeddings = torch.zeros((len(item_table) + 1, encoded_metadata.shape[1]), dtype=torch.float32)
    title_embeddings[1:] = encoded_titles.float()
    metadata_embeddings[1:] = encoded_metadata.float()
    torch.save(title_embeddings, ITEM_TITLE_EMBEDDINGS_PATH)
    torch.save(metadata_embeddings, ITEM_METADATA_EMBEDDINGS_PATH)
    return title_embeddings, metadata_embeddings


def load_or_build_query_embeddings(
    model: SentenceTransformer,
    query_table: pd.DataFrame,
    refresh_embeddings: bool = False,
) -> torch.Tensor:
    if not refresh_embeddings and QUERY_EMBEDDINGS_PATH.exists():
        query_embeddings = torch.load(QUERY_EMBEDDINGS_PATH, map_location="cpu")
        if query_embeddings.shape[0] == len(query_table):
            return query_embeddings

    query_embeddings = encode_texts(model, query_table["query_text"].tolist(), batch_size=128).float()
    torch.save(query_embeddings, QUERY_EMBEDDINGS_PATH)
    return query_embeddings


def build_splits(num_queries: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(num_queries)
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(indices)
    train_end = int(num_queries * 0.8)
    val_end = int(num_queries * 0.9)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def build_train_examples(
    query_embeddings: torch.Tensor,
    query_table: pd.DataFrame,
    indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_query_embeddings = []
    train_target_ids = []
    for idx in indices.tolist():
        positive_ids = query_table.iloc[idx]["positive_item_ids"]
        for item_id in positive_ids:
            train_query_embeddings.append(query_embeddings[idx])
            train_target_ids.append(int(item_id))
    return torch.stack(train_query_embeddings, dim=0), torch.tensor(train_target_ids, dtype=torch.long)


def build_eval_split(
    query_embeddings: torch.Tensor,
    query_table: pd.DataFrame,
    indices: np.ndarray,
) -> tuple[torch.Tensor, list[list[int]], list[str]]:
    positive_item_ids = [list(map(int, query_table.iloc[idx]["positive_item_ids"])) for idx in indices.tolist()]
    query_texts = query_table.iloc[indices]["query_text"].tolist()
    return query_embeddings[indices], positive_item_ids, query_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-embeddings", action="store_true")
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    ensure_directories()
    movies = load_movies()
    movies = movies[(movies["overview"].str.len() >= MIN_OVERVIEW_CHARS) | movies["tags"].map(bool)].copy()
    movies = movies.reset_index(drop=True)
    movies["item_idx"] = np.arange(1, len(movies) + 1)
    item_id_to_idx = {int(movie_id): int(item_idx) for movie_id, item_idx in zip(movies["id"], movies["item_idx"])}

    query_groups = load_query_groups(item_id_to_idx, max_queries=args.max_queries)
    query_groups = query_groups.rename(columns={"query": "query_text", "target_item_id": "positive_item_ids"})

    sentence_model = load_sentence_model()
    item_title_embeddings, item_metadata_embeddings = load_or_build_item_embeddings(
        sentence_model,
        movies,
        refresh_embeddings=args.refresh_embeddings,
    )
    query_embeddings = load_or_build_query_embeddings(
        sentence_model,
        query_groups,
        refresh_embeddings=args.refresh_embeddings,
    )

    train_idx, val_idx, test_idx = build_splits(len(query_groups))
    train_query_embeddings, train_target_ids = build_train_examples(query_embeddings, query_groups, train_idx)
    val_loss_query_embeddings, val_loss_target_ids = build_train_examples(query_embeddings, query_groups, val_idx)
    val_query_embeddings, val_positive_ids, val_query_texts = build_eval_split(query_embeddings, query_groups, val_idx)
    test_query_embeddings, test_positive_ids, test_query_texts = build_eval_split(query_embeddings, query_groups, test_idx)

    dataset = {
        "dataset_name": "msrd",
        "num_items": len(movies),
        "sentence_embedding_dim": int(item_title_embeddings.shape[1]),
        "query_embedding_dim": int(query_embeddings.shape[1]),
        "item_id_to_idx": item_id_to_idx,
        "idx_to_item_id": dict(zip(movies["item_idx"], movies["id"])),
        "item_titles": dict(zip(movies["item_idx"], movies["title_text"])),
        "item_title_texts": dict(zip(movies["item_idx"], movies["title_text"])),
        "item_metadata_texts": dict(zip(movies["item_idx"], movies["metadata_text"])),
        "item_title_embeddings": item_title_embeddings,
        "item_metadata_embeddings": item_metadata_embeddings,
        "train_query_embeddings": train_query_embeddings,
        "train_target_ids": train_target_ids,
        "val_loss_query_embeddings": val_loss_query_embeddings,
        "val_loss_target_ids": val_loss_target_ids,
        "val_query_embeddings": val_query_embeddings,
        "val_positive_ids": val_positive_ids,
        "val_query_texts": val_query_texts,
        "test_query_embeddings": test_query_embeddings,
        "test_positive_ids": test_positive_ids,
        "test_query_texts": test_query_texts,
        "sample_queries": test_query_texts[: min(128, len(test_query_texts))],
    }

    query_table = query_groups.copy()
    query_table["num_positive_items"] = query_table["positive_item_ids"].map(len)

    torch.save(dataset, DATASET_PATH)
    movies.to_csv(ITEM_TABLE_PATH, index=False)
    query_table.to_csv(QUERY_TABLE_PATH, index=False)
    print(f"Saved dataset to {DATASET_PATH}")
    print(f"Saved item table to {ITEM_TABLE_PATH}")
    print(f"Saved query table to {QUERY_TABLE_PATH}")
    print(
        "Split sizes:",
        {
            "train_queries": int(len(train_idx)),
            "val_queries": int(len(val_idx)),
            "test_queries": int(len(test_idx)),
            "train_examples": int(train_target_ids.shape[0]),
        },
    )


if __name__ == "__main__":
    main()
