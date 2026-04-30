#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/root}"
IMAGE="${IMAGE:-ghcr.io/maxwellyin/cineseek-semantic-search:latest}"
CONTAINER="${CONTAINER:-cineseek}"
PORT="${PORT:-8000:8000}"
ENV_FILE="${ENV_FILE:-.env}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/health}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./scripts/cineseek_deploy.sh

Expected server-side behavior:
  - pull the latest CineSeek image
  - stop and remove the old container if it exists
  - start a new container with --env-file
  - wait for the /health endpoint to pass
  - prune dangling/unused image layers

Environment overrides:
  APP_DIR     Default: /root
  IMAGE       Default: ghcr.io/maxwellyin/cineseek-semantic-search:latest
  CONTAINER   Default: cineseek
  PORT        Default: 8000:8000
  ENV_FILE    Default: .env
  HEALTH_URL  Default: http://127.0.0.1:8000/health
EOF
  exit 0
fi

cd "$APP_DIR"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: ${APP_DIR}/${ENV_FILE}" >&2
  exit 1
fi

echo "==> Pulling latest image..."
docker pull "$IMAGE"

echo "==> Stopping old container (if exists)..."
docker stop "$CONTAINER" 2>/dev/null || true

echo "==> Removing old container (if exists)..."
docker rm "$CONTAINER" 2>/dev/null || true

echo "==> Starting new container..."
docker run -d \
  --name "$CONTAINER" \
  --restart=always \
  -p "$PORT" \
  --env-file "$ENV_FILE" \
  "$IMAGE"

echo "==> Waiting for health check..."
HEALTH_OK=0

for _ in $(seq 1 60); do
  if curl -fsS "$HEALTH_URL" >/dev/null; then
    HEALTH_OK=1
    echo "==> Health check passed."
    break
  fi

  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "Container exited. Recent logs:" >&2
    docker logs --tail=120 "$CONTAINER" >&2
    exit 1
  fi

  sleep 2
done

if [[ "$HEALTH_OK" != "1" ]]; then
  echo "Health check timed out. Recent logs:" >&2
  docker logs --tail=120 "$CONTAINER" >&2
  exit 1
fi

echo
echo "==> Deployment finished."
docker ps --filter "name=${CONTAINER}" || true

echo
echo "==> Pruning unused old images..."
docker image prune -f
