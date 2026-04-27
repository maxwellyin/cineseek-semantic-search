#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_PATH="${1:-artifacts/release/cineseek-assets.tar.gz}"

REQUIRED_PATHS=(
  "data/processed"
  "data/models"
  "artifacts/checkpoints/msrd_items.faiss"
  "artifacts/checkpoints/msrd_index_metadata.pt"
)

OPTIONAL_PATHS=(
  "artifacts/checkpoints/msrd_text_retriever.pt"
  "artifacts/checkpoints/msrd_text_retriever_latest.pt"
  "artifacts/checkpoints/epochs"
)

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./scripts/build_asset_bundle.sh [output-path]

Builds a compressed asset bundle containing the static retrieval artifacts
needed for Docker/CI builds:
  - data/processed
  - data/models
  - FAISS index + index metadata
  - current retriever checkpoints when present

Default output:
  artifacts/release/cineseek-assets.tar.gz
EOF
  exit 0
fi

for path in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required asset path: $path" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$OUTPUT_PATH")"

TAR_PATHS=("${REQUIRED_PATHS[@]}")
for path in "${OPTIONAL_PATHS[@]}"; do
  if [[ -e "$path" ]]; then
    TAR_PATHS+=("$path")
  fi
done

echo "Building CineSeek asset bundle"
echo "  output: ${OUTPUT_PATH}"

tar -czf "$OUTPUT_PATH" "${TAR_PATHS[@]}"

echo
echo "Done."
du -sh "$OUTPUT_PATH"
