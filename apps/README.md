Application entry points and interactive demos.

- `demo/`: the sole text-to-title retrieval demo, backed by cached sentence-transformer embeddings and a FAISS index

Current demo apps run on FastAPI.

Typical startup command from the repository root:

```bash
uvicorn apps.demo.app:app --reload
```

Notes:

- the app renders Jinja templates from its local `templates/` directory
- the app loads the processed dataset and FAISS index produced by `python -m flcr.index`
