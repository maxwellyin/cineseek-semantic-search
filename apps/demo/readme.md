This is the main text-to-title retrieval demo.

Current status:

- web framework: FastAPI
- retrieval backend: FAISS
- model style: dual-tower retrieval over movie search queries and movie metadata
- optional agent layer: LangChain + Ollama for local query rewrite and result explanation

Run from the repository root:

```bash
uvicorn apps.demo.app:app --reload
```

Before starting the app, build the dataset, train the model, and create the index:

```bash
python -m flcr.data_processing.download_sentence_transformer
python -m flcr.data_processing.download_msrd
python -m flcr.data_processing.build_msrd_dataset
python -m flcr.train
env FLCR_DEVICE=cpu KMP_DUPLICATE_LIB_OK=TRUE python -m flcr.index
```

Optional agent mode:

- install `langchain`, `langchain-ollama`
- install and run Ollama locally
- pull a local model such as `qwen3:8b`
- check `Use LangChain agent enhancements` in the demo UI

Example local setup:

```bash
brew install ollama
ollama serve
ollama pull qwen3:8b
```

Optional provider override:

- `FLCR_AGENT_PROVIDER=openai` plus `OPENAI_API_KEY`
- `FLCR_AGENT_PROVIDER=ollama` plus `FLCR_OLLAMA_MODEL=qwen3:8b`
