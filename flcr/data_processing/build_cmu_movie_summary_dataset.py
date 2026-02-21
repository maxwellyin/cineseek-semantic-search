from __future__ import annotations

import argparse
import ast
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from flcr.config import (
    ITEM_METADATA_EMBEDDINGS_PATH,
    ITEM_TABLE_PATH,
    ITEM_TITLE_EMBEDDINGS_PATH,
    MAX_METADATA_CHARS,
    MIN_QUERY_CHARS,
    MIN_SUMMARY_CHARS,
    RANDOM_SEED,
    RAW_CMU_ARCHIVE,
    SENTENCE_MODEL_DIR,
    SENTENCE_TRANSFORMER_DEVICE,
    ensure_directories,
    get_dataset_path,
    get_query_embeddings_path,
    get_query_table_path,
)


def load_sentence_model() -> SentenceTransformer:
    return SentenceTransformer(str(SENTENCE_MODEL_DIR), device=SENTENCE_TRANSFORMER_DEVICE)


def sanitize_text(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    return " ".join(text.split())


def parse_freebase_dict(raw_value: str) -> list[str]:
    raw_value = sanitize_text(raw_value)
    if not raw_value:
        return []
    try:
        parsed = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, dict):
        return [sanitize_text(value) for value in parsed.values() if sanitize_text(value)]
    return []


def _open_archive_member_text(filename: str):
    archive = tarfile.open(RAW_CMU_ARCHIVE, "r:gz")
    try:
        try:
            member = archive.getmember(filename)
        except KeyError:
            member = archive.getmember(f"MovieSummaries/{filename}")
        handle = archive.extractfile(member)
        if handle is None:
            raise FileNotFoundError(filename)
        return archive, handle
    except Exception:
        archive.close()
        raise


def load_plot_summaries() -> pd.DataFrame:
    rows = []
    archive, raw_handle = _open_archive_member_text("plot_summaries.txt")
    try:
        fp = (line.decode("utf-8", errors="ignore") for line in raw_handle)
        for line in fp:
            line = line.rstrip("\n")
            if not line:
                continue
            wiki_id, summary = line.split("\t", 1)
            summary = sanitize_text(summary)
            if len(summary) < MIN_SUMMARY_CHARS:
                continue
            rows.append({"wiki_movie_id": int(wiki_id), "plot_summary": summary})
    finally:
        raw_handle.close()
        archive.close()
    return pd.DataFrame(rows)


def load_movie_metadata() -> pd.DataFrame:
    columns = [
        "wiki_movie_id",
        "freebase_movie_id",
        "movie_name",
        "movie_release_date",
        "movie_box_office_revenue",
        "movie_runtime",
        "movie_languages",
        "movie_countries",
        "movie_genres",
    ]
    rows = []
    archive, raw_handle = _open_archive_member_text("movie.metadata.tsv")
    try:
        fp = (line.decode("utf-8", errors="ignore") for line in raw_handle)
        for line in fp:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(columns):
                continue
            row = dict(zip(columns, parts))
            row["wiki_movie_id"] = int(row["wiki_movie_id"])
            row["movie_name"] = sanitize_text(row["movie_name"])
            row["movie_release_date"] = sanitize_text(row["movie_release_date"])
            row["genres"] = parse_freebase_dict(row["movie_genres"])
            row["languages"] = parse_freebase_dict(row["movie_languages"])
            row["countries"] = parse_freebase_dict(row["movie_countries"])
            if not row["movie_name"]:
                continue
            rows.append(row)
    finally:
        raw_handle.close()
        archive.close()
    return pd.DataFrame(rows)


def extract_release_year(release_date: str) -> str:
    match = re.search(r"(19|20)\d{2}", release_date or "")
    return match.group(0) if match else ""


def split_sentences(summary: str) -> list[str]:
    sentences = [
        sanitize_text(sentence).strip(" -.,;:")
        for sentence in re.split(r"(?<=[.!?])\s+", sanitize_text(summary))
    ]
    return [sentence for sentence in sentences if len(sentence.split()) >= 4]


def build_query_text(summary: str) -> str:
    return sanitize_text(summary)


def build_metadata_text(title: str, year: str, genres: list[str], summary: str, max_chars: int) -> str:
    pieces = [
        title,
        f"release year: {year}" if year else "",
        f"genres: {', '.join(genres)}" if genres else "",
        f"plot summary: {summary}",
    ]
    text = " ".join(piece for piece in pieces if piece).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 256) -> torch.Tensor:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=False,
    ).cpu()


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
    query_mode: str,
    refresh_embeddings: bool = False,
) -> torch.Tensor:
    query_embeddings_path = get_query_embeddings_path(query_mode)
    if not refresh_embeddings and query_embeddings_path.exists():
        query_embeddings = torch.load(query_embeddings_path, map_location="cpu")
        if query_embeddings.shape[0] == len(query_table):
            return query_embeddings

    if query_mode == "full_summary":
        query_embeddings = encode_texts(model, query_table["query_text"].tolist(), batch_size=128).float()
    elif query_mode == "sentence_mean":
        sentence_groups = [split_sentences(text) or [sanitize_text(text)] for text in query_table["query_text"].tolist()]
        flat_sentences = [sentence for group in sentence_groups for sentence in group]
        encoded = encode_texts(model, flat_sentences, batch_size=128).float()
        offsets = [0]
        for group in sentence_groups:
            offsets.append(offsets[-1] + len(group))
        pooled = []
        for start, stop in zip(offsets[:-1], offsets[1:]):
            pooled.append(encoded[start:stop].mean(dim=0))
        query_embeddings = torch.stack(pooled, dim=0)
    else:
        raise ValueError(f"Unsupported query_mode: {query_mode}")

    torch.save(query_embeddings, query_embeddings_path)
    return query_embeddings


def build_splits(num_examples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(num_examples)
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(indices)
    train_end = int(num_examples * 0.8)
    val_end = int(num_examples * 0.9)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-embeddings", action="store_true")
    parser.add_argument(
        "--query-mode",
        choices=["full_summary", "sentence_mean"],
        default="full_summary",
    )
    args = parser.parse_args()

    ensure_directories()
    dataset_path = get_dataset_path(args.query_mode)
    query_table_path = get_query_table_path(args.query_mode)
    plot_summaries = load_plot_summaries()
    movie_metadata = load_movie_metadata()
    merged = plot_summaries.merge(movie_metadata, on="wiki_movie_id", how="inner")
    merged = merged.drop_duplicates("wiki_movie_id").reset_index(drop=True)

    merged["release_year"] = merged["movie_release_date"].map(extract_release_year)
    merged["title"] = merged["movie_name"]
    merged["title_text"] = merged.apply(
        lambda row: f"{row['title']} ({row['release_year']})" if row["release_year"] else row["title"],
        axis=1,
    )
    merged["query_text"] = merged["plot_summary"].map(build_query_text)
    merged = merged[merged["query_text"].str.len() >= MIN_QUERY_CHARS].copy()
    merged["metadata_text"] = merged.apply(
        lambda row: build_metadata_text(
            title=row["title"],
            year=row["release_year"],
            genres=row["genres"],
            summary=row["plot_summary"],
            max_chars=MAX_METADATA_CHARS,
        ),
        axis=1,
    )
    merged = merged.reset_index(drop=True)
    merged["item_idx"] = np.arange(1, len(merged) + 1)

    item_table = merged[
        [
            "wiki_movie_id",
            "item_idx",
            "title",
            "title_text",
            "metadata_text",
            "release_year",
            "plot_summary",
            "query_text",
        ]
    ].copy()
    item_id_to_idx = dict(zip(item_table["wiki_movie_id"], item_table["item_idx"]))
    query_table = merged[["wiki_movie_id", "query_text", "plot_summary"]].copy()
    query_table["target_item_id"] = query_table["wiki_movie_id"].map(item_id_to_idx)

    sentence_model = load_sentence_model()
    item_title_embeddings, item_metadata_embeddings = load_or_build_item_embeddings(
        sentence_model,
        item_table,
        refresh_embeddings=args.refresh_embeddings,
    )
    query_embeddings = load_or_build_query_embeddings(
        sentence_model,
        query_table,
        query_mode=args.query_mode,
        refresh_embeddings=args.refresh_embeddings,
    )

    train_idx, val_idx, test_idx = build_splits(len(query_table))
    dataset = {
        "num_items": len(item_table),
        "sentence_embedding_dim": int(item_title_embeddings.shape[1]),
        "query_embedding_dim": int(query_embeddings.shape[1]),
        "item_id_to_idx": item_id_to_idx,
        "idx_to_item_id": dict(zip(item_table["item_idx"], item_table["wiki_movie_id"])),
        "item_titles": dict(zip(item_table["item_idx"], item_table["title_text"])),
        "item_title_texts": dict(zip(item_table["item_idx"], item_table["title_text"])),
        "item_metadata_texts": dict(zip(item_table["item_idx"], item_table["metadata_text"])),
        "item_title_embeddings": item_title_embeddings,
        "item_metadata_embeddings": item_metadata_embeddings,
        "train_query_embeddings": query_embeddings[train_idx],
        "train_target_ids": torch.tensor(query_table.iloc[train_idx]["target_item_id"].to_numpy(), dtype=torch.long),
        "val_query_embeddings": query_embeddings[val_idx],
        "val_target_ids": torch.tensor(query_table.iloc[val_idx]["target_item_id"].to_numpy(), dtype=torch.long),
        "test_query_embeddings": query_embeddings[test_idx],
        "test_target_ids": torch.tensor(query_table.iloc[test_idx]["target_item_id"].to_numpy(), dtype=torch.long),
        "sample_queries": query_table.iloc[test_idx[: min(128, len(test_idx))]]["query_text"].tolist(),
        "query_mode": args.query_mode,
    }
    torch.save(dataset, dataset_path)
    item_table.to_csv(ITEM_TABLE_PATH, index=False)
    query_table.to_csv(query_table_path, index=False)
    print(f"Saved dataset to {dataset_path}")
    print(f"Saved item table to {ITEM_TABLE_PATH}")
    print(f"Saved query table to {query_table_path}")
    print(f"Query mode: {args.query_mode}")
    print(
        "Split sizes:",
        {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
    )


if __name__ == "__main__":
    main()
