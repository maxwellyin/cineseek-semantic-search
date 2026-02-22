# CineSeek

**CineSeek** is a local-first semantic movie search project built to show a full retrieval stack, not just a chat wrapper.

It maps real user-style movie queries to titles using a trained dual-tower retriever, serves candidates with FAISS, and adds an optional LangChain + Ollama layer for query rewrite, reranking, and explanation.

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
- **LangChain + Ollama** for local agent-based rewrite / rerank / explanation
- **Weights & Biases** for training runs and checkpoint comparisons

## Dataset

The current mainline uses **MSRD**:

- movie metadata sourced from MovieLens and TMDB
- 28,320 real movie search queries
- crowd-labeled query-to-movie relevance judgments

This makes the task match the product surface directly:

**movie search query -> movie title**

## Running It

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

## Local Agent Mode

CineSeek defaults to a **local Ollama-backed agent** for search enhancement.

Install and start Ollama:

```bash
brew install ollama
brew services start ollama
ollama pull qwen3:8b
```

Then launch the app and keep the agent option enabled in the UI.

Optional overrides:

```bash
export FLCR_AGENT_PROVIDER=ollama
export FLCR_OLLAMA_MODEL=qwen3:8b
```

If you later want a hosted provider instead:

```bash
export FLCR_AGENT_PROVIDER=openai
export OPENAI_API_KEY=...
```

## Repository Layout

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
