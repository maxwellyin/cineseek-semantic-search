# Text-to-Title Retrieval System

A local-first retrieval system that maps real movie search queries to movie titles.

- dual-tower retrieval with PyTorch
- sentence-transformer caching for both query and item text
- Apple Silicon friendly training with `mps` fallback to CPU
- FAISS ANN serving for low-latency text-to-item retrieval
- FastAPI demo for interactive testing

## Overview

The system learns two embedding spaces:

- a query tower that projects cached sentence-transformer search-query embeddings
- an item tower that fuses separate cached sentence-transformer embeddings for title and metadata

Sentence-transformer outputs are cached locally, so later training runs only optimize the lightweight MLP dual towers.

## Dataset

The default data source is `MSRD` (Movie Search Ranking Dataset), which provides:

- movie metadata sourced from MovieLens and TMDB
- 28,320 real movie search queries
- crowd-labeled query-to-movie relevance judgments

This matches the target task directly: `movie search query -> recommended title`.

## Pipeline

1. Cache the sentence-transformer model locally
2. Download movie metadata and query relevance labels from MSRD
3. Build cached query and item embeddings
4. Train the dual-tower retriever
5. Encode all catalog items and build a FAISS index
6. Evaluate retrieval quality
7. Launch the FastAPI demo

## Quick Start

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Cache the sentence-transformer and build a processed dataset:

```bash
python -m flcr.data_processing.download_sentence_transformer
python -m flcr.data_processing.download_msrd
python -m flcr.data_processing.build_msrd_dataset
```

Train and build the index:

```bash
python -m flcr.train
env FLCR_DEVICE=cpu KMP_DUPLICATE_LIB_OK=TRUE python -m flcr.index
env KMP_DUPLICATE_LIB_OK=TRUE python -m flcr.evaluate
python -m flcr.qualitative --query "the matrix movie"
```

Optional Weights & Biases logging:

```bash
python -m flcr.train --epochs 100 --batch-size 512 --wandb --wandb-project retrieval-system
```

Long training options:

```bash
python -m flcr.train --epochs 100 --batch-size 512 --save-every 10
python -m flcr.train --epochs 100 --batch-size 512 --resume-from artifacts/checkpoints/msrd_text_retriever_latest.pt
```

Run the demo:

```bash
uvicorn apps.demo.app:app --reload
```

Then open [http://127.0.0.1:8000/demo](http://127.0.0.1:8000/demo).

## Repository Structure

```text
flcr/
  ├── config.py
  ├── model.py
  ├── train.py
  ├── index.py
  ├── evaluate.py
  ├── qualitative.py
  ├── search.py
  └── data_processing/

apps/
  └── demo/

artifacts/
data/
```

## Notes

The repository does not redistribute MSRD contents. The build scripts download the public raw files locally and write only processed local artifacts into `data/processed/`.
