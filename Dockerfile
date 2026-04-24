ARG ASSET_IMAGE=ghcr.io/maxwellyin/cineseek-semantic-search:latest
FROM ${ASSET_IMAGE} AS asset_source

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLCR_DEVICE=cpu \
    FLCR_AGENT_PROVIDER=groq \
    FLCR_GROQ_MODEL=qwen/qwen3-32b

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && grep -v '^torch$' /app/requirements.txt > /app/requirements.docker.txt \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r /app/requirements.docker.txt

COPY apps /app/apps
COPY flcr /app/flcr
COPY scripts /app/scripts
COPY readme.md /app/readme.md
RUN mkdir -p /app/artifacts/checkpoints
COPY --from=asset_source /app/data/processed /app/data/processed
COPY --from=asset_source /app/data/models /app/data/models
COPY --from=asset_source /app/artifacts/checkpoints/msrd_items.faiss /app/artifacts/checkpoints/msrd_items.faiss
COPY --from=asset_source /app/artifacts/checkpoints/msrd_index_metadata.pt /app/artifacts/checkpoints/msrd_index_metadata.pt

EXPOSE 8000

CMD ["uvicorn", "apps.demo.app:app", "--host", "0.0.0.0", "--port", "8000"]
