#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RELEASE_TAG="${RELEASE_TAG:-assets-current}"
ASSET_NAME="${ASSET_NAME:-cineseek-assets.tar.gz}"
REPO="${REPO:-maxwellyin/cineseek-semantic-search}"
OUTPUT_PATH="${OUTPUT_PATH:-artifacts/release/${ASSET_NAME}}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./scripts/publish_asset_release.sh

Builds the static retrieval asset bundle and uploads it to a dedicated GitHub
release that Docker/CI can download from.

Environment overrides:
  RELEASE_TAG  Default: assets-current
  ASSET_NAME   Default: cineseek-assets.tar.gz
  REPO         Default: maxwellyin/cineseek-semantic-search
  OUTPUT_PATH  Default: artifacts/release/cineseek-assets.tar.gz
EOF
  exit 0
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required" >&2
  exit 1
fi

"$ROOT_DIR/scripts/build_asset_bundle.sh" "$OUTPUT_PATH"

if ! gh release view "$RELEASE_TAG" --repo "$REPO" >/dev/null 2>&1; then
  gh release create "$RELEASE_TAG" \
    --repo "$REPO" \
    --title "CineSeek Assets" \
    --notes "Static retrieval assets for Docker and CI builds."
fi

gh release upload "$RELEASE_TAG" "$OUTPUT_PATH" \
  --repo "$REPO" \
  --clobber

echo
echo "Uploaded ${ASSET_NAME} to release ${RELEASE_TAG}"
