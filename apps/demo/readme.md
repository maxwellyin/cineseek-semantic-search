This is the main text-to-title retrieval demo.

Current status:

- web framework: FastAPI
- retrieval backend: FAISS
- model style: raw sentence-transformer retrieval over movie search queries, titles, and overviews
- optional agent layer: LangChain + Groq / Gemini / Ollama / OpenAI for query rewrite and result explanation

Run from the repository root:

```bash
uvicorn apps.demo.app:app --reload
```

Before starting the app, build the dataset and create the raw embedding index:

```bash
python -m flcr.data_processing.download_sentence_transformer
python -m flcr.data_processing.download_msrd
python -m flcr.data_processing.build_msrd_dataset
env FLCR_DEVICE=cpu KMP_DUPLICATE_LIB_OK=TRUE python -m flcr.index
```

Optional agent mode:

- install `langchain`, `langchain-google-genai`, or `langchain-groq`
- set `GROQ_API_KEY`
- or set `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- check `Use LangChain agent enhancements` in the demo UI

Default provider:

- `FLCR_AGENT_PROVIDER=groq`
- `FLCR_GROQ_MODEL=qwen/qwen3-32b`

Optional provider override:

- `FLCR_AGENT_PROVIDER=gemini` plus `GOOGLE_API_KEY` and `FLCR_GEMINI_MODEL=gemini-2.5-flash-lite`
- `FLCR_AGENT_PROVIDER=ollama` plus `FLCR_OLLAMA_MODEL=qwen3:8b`
- `FLCR_AGENT_PROVIDER=openai` plus `OPENAI_API_KEY`
