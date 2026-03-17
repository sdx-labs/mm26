# Data Pipeline Planning and Initial Usage

The first stages of a beginning plan to set this up are as follows:

## General Rules
- fully examine the rules of the kaggle competition and the data files as outlined by the files within .agents/kaggle/*. 
- you are to update those files with all pertinent findings to further improve your usage and continually improve your understanding of the competition and the dataa

## First Steps
- fully understand the competition data currently hosted in the ./data folder. it contains a wide array of files. these should be treated as silver layer files for data. read the competition objectives, then piece together a gold table, one that's human-readable, that consists of the following:
- team table: with all relevant statistics
- game table: with all relevant statistics

## Next Steps
- the user plans to integrate cbbd api to join the data into the predictive model. be aware of this while you are making your initial plans.

## Stage 1 Implementation Contract (March 2026)
- Single entrypoint: run `python main.py run --mode manual --target-season 2026` from repo root.
- Layered outputs are written to `artifacts/runs/<run_id>/` with `artifacts/latest` symlinked to newest run.
- Bronze:
  - Kaggle source CSVs are converted into immutable parquet snapshots under `bronze/kaggle/`.
  - CBBD men-only lines data is written to `bronze/cbbd/lines.parquet` (empty table if API key missing/error).
- Silver:
  - `team_dim.parquet`
  - `game_fact.parquet` (canonical team-game grain)
  - `team_id_map.parquet`
  - `quality_flags.parquet`
- Gold:
  - `team_season_features.parquet`
  - `game_features.parquet`
  - `pairwise_features.parquet`
  - `submission.csv` in Kaggle `ID,Pred` format
- Reports:
  - `reports/validation_report.json`
  - `reports/team_id_map_report.csv`
  - `reports/team_id_map_summary.json`
  - `reports/gold_quality_summary.json`
  - top-level `run_manifest.json`

## Mapping Governance
- CBBD mapping mismatches are allowed during development runs.
- Every run must emit mismatch counts and percent.
- Model training and scoring logic must only consume rows passing required feature-quality checks.

## Notebook Compatibility
- Notebooks are consumers of pipeline artifacts only.
- Notebook workflows should read from `artifacts/latest/gold/*` and `artifacts/latest/reports/*`.
- Notebooks should not mutate pipeline state or write over layered outputs.

## Next steps
- pull in more historical data for a more robust machine learning model
- pull in cbbd data and ensure that it tests appropriately
