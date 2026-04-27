Auxiliary scripts retained for dataset preparation and export tasks.

- `public/data_export/`: scripts for public-data export and formatting

Representative script names use descriptive verbs such as `build_`, `export_`, `reformat_`, and `preview_` to make their roles easier to scan.
Utility scripts for manual validation.

- `demo_cases.py`: runs the exact same retrieval path used by `apps/demo` against preset or custom movie search queries so you can inspect demo-quality results from the terminal. Supports `--case`, `--query`, and `--k`.
- `publish_docker.sh`: builds and pushes the `linux/amd64` Docker image to GHCR with `latest` and the current git SHA tags. Optional extra tags can be passed as arguments. GHCR cleanup is handled separately by a GitHub Actions workflow.
- `build_asset_bundle.sh`: packages the static retrieval artifacts into `artifacts/release/cineseek-assets.tar.gz` for release-based Docker/CI builds.
- `publish_asset_release.sh`: builds the asset bundle and uploads it to the dedicated GitHub release tag (`assets-current`) that Docker and GitHub Actions download from.
