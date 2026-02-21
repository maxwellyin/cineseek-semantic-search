Auxiliary scripts retained for dataset preparation and export tasks.

- `public/data_export/`: scripts for public-data export and formatting

Representative script names use descriptive verbs such as `build_`, `export_`, `reformat_`, and `preview_` to make their roles easier to scan.
Utility scripts for manual validation.

- `demo_cases.py`: runs the exact same retrieval path used by `apps/demo` against preset or custom movie search queries so you can inspect demo-quality results from the terminal. Supports `--case`, `--query`, and `--k`.
