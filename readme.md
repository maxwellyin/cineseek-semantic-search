# CineSeek

**CineSeek** is a local-first semantic movie search project built to show a full retrieval stack, not just a chat wrapper.

It maps real user-style movie queries to titles using a trained dual-tower retriever, serves candidates with FAISS, and adds an optional LangChain + Ollama layer for query rewrite, reranking, and explanation.

## Highlights

- **Real search task** using MSRD query-to-movie relevance judgments
- **Dual-tower retrieval** trained in PyTorch with cached sentence embeddings
- **Low-latency search** served through FAISS
- **Agent layer** powered by LangChain + Gemini by default, with optional Ollama / OpenAI backends
- **Portfolio-friendly demo** built with FastAPI

## Why This Project Exists

Most portfolio projects stop at vector search or a lightweight prompt demo. CineSeek is meant to show the full retrieval loop:

- training on a real search relevance dataset
- caching sentence-transformer embeddings for efficient iteration
- learning a lightweight retrieval head with PyTorch
- serving low-latency ANN search with FAISS
- layering a local agent on top without replacing the retrieval system

The result is a project that is easy to demo, easy to extend, and still grounded in retrieval engineering.

## What It Does

- **Query-to-movie retrieval**
  - The system is trained on **MSRD** (Movie Search Ranking Dataset), which contains real movie search queries and relevance labels.
- **Dual-tower ranking**
  - A query tower projects search-query embeddings.
  - An item tower fuses title and metadata embeddings.
- **Fast local serving**
  - FAISS handles candidate retrieval from the movie catalog.
- **Agent-enhanced search**
  - A local LangChain + Ollama layer can rewrite vague queries, rerank the top candidates, and explain the result set.

## Architecture

```text
user query
  -> sentence-transformer embedding
  -> query tower
  -> FAISS retrieval over item tower embeddings
  -> top-k candidates
  -> optional LangChain + Ollama reranker / explainer
  -> final result list
```

## Tech Stack

- **PyTorch** for dual-tower training
- **Sentence-Transformers** for cached text embeddings
- **FAISS** for ANN retrieval
- **FastAPI + Jinja** for the web interface
- **LangChain + Gemini / Ollama / OpenAI** for rewrite, rerank, and explanation
- **Weights & Biases** for training runs and checkpoint comparisons

## Dataset

The current mainline uses **MSRD**:

- movie metadata sourced from MovieLens and TMDB
- 28,320 real movie search queries
- crowd-labeled query-to-movie relevance judgments

This makes the task match the product surface directly:

**movie search query -> movie title**

## Quick Start

Create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare the retriever:

```bash
python -m flcr.data_processing.download_sentence_transformer
python -m flcr.data_processing.download_msrd
python -m flcr.data_processing.build_msrd_dataset
python -m flcr.train
env FLCR_DEVICE=cpu KMP_DUPLICATE_LIB_OK=TRUE python -m flcr.index
```

Run the app:

```bash
uvicorn apps.demo.app:app --reload
```

Then open [http://127.0.0.1:8000/search](http://127.0.0.1:8000/search).

## Docker Deployment

This repo includes a production-oriented Docker setup for low-cost VPS deployment. The container bakes in:

- the processed MSRD dataset
- the cached sentence-transformer model
- the selected retriever checkpoint
- the FAISS index

Pull and run from GHCR:

```bash
docker pull ghcr.io/maxwellyin/cineseek-semantic-search:latest
docker run -p 8000:8000 \
  -e GOOGLE_API_KEY=your_gemini_api_key \
  ghcr.io/maxwellyin/cineseek-semantic-search:latest
```

Or build locally:

```bash
docker build -t cineseek-semantic-search .
docker run -p 8000:8000 \
  -e GOOGLE_API_KEY=your_gemini_api_key \
  cineseek-semantic-search
```

Or use Docker Compose:

```bash
cp .env.example .env
# edit .env and set GOOGLE_API_KEY
docker compose up -d --build
```

This setup is intentionally deployment-friendly for small CPU servers: retrieval artifacts are pre-baked into the image so the server only needs to run the app, not rebuild the dataset or retrain the model.

## Agent Providers

CineSeek now defaults to a **Gemini-backed agent** for query rewrite, reranking, and explanation.

Set an API key:

```bash
export GOOGLE_API_KEY=...
```

or

```bash
export GEMINI_API_KEY=...
```

Default settings:

```bash
export FLCR_AGENT_PROVIDER=gemini
export FLCR_GEMINI_MODEL=gemini-2.5-flash
```

Optional alternatives:

```bash
export FLCR_AGENT_PROVIDER=ollama
export FLCR_OLLAMA_MODEL=qwen3:8b
```

```bash
export FLCR_AGENT_PROVIDER=openai
export OPENAI_API_KEY=...
```

## Project Layout

```text
apps/demo/          FastAPI UI
flcr/train.py       training loop
flcr/model.py       dual-tower retriever
flcr/index.py       FAISS index builder
flcr/evaluate.py    retrieval metrics
flcr/search.py      search utilities
flcr/agent/         LangChain agent layer
flcr/data_processing/
```

## Notes

- This repository does **not** redistribute MSRD data.
- Raw files are downloaded locally and processed into local caches.
- The LLM layer is intentionally optional; the retrieval system remains the core product.
