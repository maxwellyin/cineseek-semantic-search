from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np
import torch


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "msrd"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
RUNS_DIR = ARTIFACTS_DIR / "runs"

RAW_MSRD_MOVIES_PATH = RAW_DATA_DIR / "movies.csv.gz"
RAW_MSRD_QUERIES_PATH = RAW_DATA_DIR / "queries.csv.gz"

ITEM_TABLE_PATH = PROCESSED_DIR / "msrd_items.csv"
QUERY_TABLE_PATH = PROCESSED_DIR / "msrd_queries.csv"
DATASET_PATH = PROCESSED_DIR / "msrd_text2item_dataset.pt"
ITEM_TITLE_EMBEDDINGS_PATH = PROCESSED_DIR / "msrd_title_embeddings.pt"
ITEM_METADATA_EMBEDDINGS_PATH = PROCESSED_DIR / "msrd_metadata_embeddings.pt"
QUERY_EMBEDDINGS_PATH = PROCESSED_DIR / "msrd_query_embeddings.pt"

INDEX_PATH = CHECKPOINT_DIR / "msrd_items.faiss"
INDEX_METADATA_PATH = CHECKPOINT_DIR / "msrd_index_metadata.pt"

SENTENCE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_MODEL_DIR = MODELS_DIR / "all-MiniLM-L6-v2"

MAX_METADATA_CHARS = 1400
MIN_QUERY_CHARS = 3
MIN_OVERVIEW_CHARS = 20
RANDOM_SEED = 7

SENTENCE_EMBED_DIM = 384


def get_device() -> torch.device:
    forced_device = os.environ.get("FLCR_DEVICE")
    if forced_device:
        return torch.device(forced_device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


def get_sentence_transformer_device() -> str:
    if DEVICE.type == "mps":
        return "mps"
    if DEVICE.type == "cuda":
        return "cuda"
    return "cpu"


SENTENCE_TRANSFORMER_DEVICE = get_sentence_transformer_device()


def ensure_directories() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
