MSRD data processing pipeline:

1. `download_sentence_transformer.py` downloads and caches the sentence-transformer model used for query and item text embeddings.
2. `download_msrd.py` downloads the MSRD `movies.csv.gz` and `queries.csv.gz` files.
3. `build_msrd_dataset.py` reads movie metadata and query relevance labels, caches sentence-transformer outputs for query and item title/metadata, then writes train/validation/test tensors for the dual-tower retriever.
