#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_NAME="${IMAGE_NAME:-ghcr.io/maxwellyin/cineseek-semantic-search}"
PLATFORM="${PLATFORM:-linux/amd64}"
GIT_SHA="$(git rev-parse --short HEAD)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./scripts/publish_docker.sh [extra-tag ...]

Builds and pushes the CineSeek Docker image to GHCR with:
  - latest
  - current git SHA
  - any optional extra tags you pass
Publishes only the requested target platform and disables extra provenance/SBOM manifests.

Environment overrides:
  IMAGE_NAME   Default: ghcr.io/maxwellyin/cineseek-semantic-search
  PLATFORM     Default: linux/amd64
EOF
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed or not on PATH" >&2
  exit 1
fi

if ! docker buildx version >/dev/null 2>&1; then
  echo "docker buildx is required" >&2
  exit 1
fi

TAGS=(
  "-t" "${IMAGE_NAME}:latest"
  "-t" "${IMAGE_NAME}:${GIT_SHA}"
)

if [[ $# -gt 0 ]]; then
  for extra_tag in "$@"; do
    TAGS+=("-t" "${IMAGE_NAME}:${extra_tag}")
  done
fi

echo "Publishing Docker image"
echo "  image:    ${IMAGE_NAME}"
echo "  platform: ${PLATFORM}"
echo "  tags:     latest ${GIT_SHA}${*:+ $*}"

docker buildx build \
  --platform "${PLATFORM}" \
  --provenance=false \
  --sbom=false \
  "${TAGS[@]}" \
  --push \
  .

echo
echo "Done."
echo "Pushed:"
echo "  ${IMAGE_NAME}:latest"
echo "  ${IMAGE_NAME}:${GIT_SHA}"
if [[ $# -gt 0 ]]; then
  for extra_tag in "$@"; do
    echo "  ${IMAGE_NAME}:${extra_tag}"
  done
fi
