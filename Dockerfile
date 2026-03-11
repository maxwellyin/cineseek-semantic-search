FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLCR_DEVICE=cpu \
    FLCR_AGENT_PROVIDER=gemini \
    FLCR_GEMINI_MODEL=gemini-2.5-flash-lite

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
COPY data/processed /app/data/processed
COPY data/models /app/data/models
COPY artifacts/checkpoints /app/artifacts/checkpoints

EXPOSE 8000

CMD ["uvicorn", "apps.demo.app:app", "--host", "0.0.0.0", "--port", "8000"]
