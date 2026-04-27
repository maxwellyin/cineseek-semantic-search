ARG ASSET_BUNDLE_URL=https://github.com/maxwellyin/cineseek-semantic-search/releases/download/assets-current/cineseek-assets.tar.gz
FROM python:3.11-slim AS asset_source

ARG ASSET_BUNDLE_URL

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl tar \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /asset-source

RUN test -n "${ASSET_BUNDLE_URL}" \
    && curl -fsSL --retry 5 --retry-all-errors "${ASSET_BUNDLE_URL}" -o /tmp/cineseek-assets.tar.gz \
    && tar -xzf /tmp/cineseek-assets.tar.gz -C /asset-source \
    && rm -f /tmp/cineseek-assets.tar.gz

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
COPY --from=asset_source /asset-source/data/processed /app/data/processed
COPY --from=asset_source /asset-source/data/models /app/data/models
COPY --from=asset_source /asset-source/artifacts/checkpoints /app/artifacts/checkpoints

EXPOSE 8000

CMD ["uvicorn", "apps.demo.app:app", "--host", "0.0.0.0", "--port", "8000"]
