This is the main text-to-title retrieval demo.

Current status:

- web framework: FastAPI
- retrieval backend: FAISS
- model style: dual-tower retrieval over movie search queries and movie metadata

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
