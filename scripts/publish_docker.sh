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

echo
echo "--- GHCR Image Pruning ---"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Skip pruning: GITHUB_TOKEN environment variable is not set."
  exit 0
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "Skip pruning: jq is not installed."
  exit 0
fi

if [[ "${IMAGE_NAME}" == ghcr.io/* ]]; then
  REPO_PATH="${IMAGE_NAME#ghcr.io/}"
  GH_OWNER="$(echo "$REPO_PATH" | cut -d'/' -f1)"
  GH_PKG="$(echo "$REPO_PATH" | cut -d'/' -f2-)"
  
  # URL encode package name
  GH_PKG_ENCODED=$(echo "$GH_PKG" | sed 's/\//%2f/g')

  echo "Checking historical images for ${GH_OWNER}/${GH_PKG}..."

  API_URL="https://api.github.com/user/packages/container/${GH_PKG_ENCODED}/versions"
  
  VERSIONS_JSON=$(curl -s -H "Accept: application/vnd.github.v3+json" -H "Authorization: Bearer ${GITHUB_TOKEN}" "${API_URL}")
  
  if echo "$VERSIONS_JSON" | jq -e '.message' >/dev/null 2>&1; then
    ERR_MSG=$(echo "$VERSIONS_JSON" | jq -r '.message')
    echo "Skip pruning: API Error - ${ERR_MSG}"
    exit 0
  fi

  # Parse out version IDs, sort descending by creation date, take everything after the first 15
  IDS_TO_DELETE=$(echo "$VERSIONS_JSON" | jq -r 'sort_by(.created_at) | reverse | .[15:] | .[].id')
  
  if [[ -n "$IDS_TO_DELETE" ]]; then
    count=$(echo "$IDS_TO_DELETE" | wc -w)
    echo "Found ${count} old image version(s) to prune (keeping latest 15)."
    for version_id in $IDS_TO_DELETE; do
      echo "  -> Deleting version ID: ${version_id}..."
      curl -s -X DELETE \
        -H "Accept: application/vnd.github.v3+json" \
        -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        "${API_URL}/${version_id}"
    done
    echo "Pruning complete."
  else
    echo "No old images to prune (15 or fewer total)."
  fi
else
  echo "Skip pruning: IMAGE_NAME does not start with ghcr.io/"
fi
