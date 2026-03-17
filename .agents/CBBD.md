# CBBD API

the college basketball data api (CBBD API) is going to be used to set up a data pipeline to pull men's college basketball data. 

the repository can be found at https://github.com/CFBD/cbbd-python/tree/main. 

you are to open a separate environment and extensively search the guidelines and files within that repository every time that the user asks questions regarding the cbbd api.

you are to come back to this file .agents/cbbd.md and update this file with all relevant new information regarding the appropriate usage, the appropriate usage of this api. 

the api key is stored in .env file under CBBD_API_KEY

## Stage 1 Usage in This Repo
- Scope: men-only enrichment in Stage 1.
- Pipeline call site: `src/mm26/pipeline.py`, function `ingest_cbbd_lines`.
- Runtime behavior:
  - If `CBBD_API_KEY` is present, pipeline attempts to fetch lines for `target_season`.
  - If key is missing or API call fails, pipeline writes an empty bronze CBBD parquet and continues.
  - Run manifest records CBBD status as `ok`, `skipped`, or `error`.

## TeamID Mapping Behavior
- CBBD team names are normalized and mapped to Kaggle TeamID using:
  - `MTeams.csv` team names
  - `MTeamSpellings.csv` alternative spellings
- Mapping outputs:
  - `silver/team_id_map.parquet`
  - `reports/team_id_map_report.csv`
  - `reports/team_id_map_summary.json` (includes mapped/unmapped and mapped_pct)

## Safety
- Do not print API secrets in logs or reports.
- Treat CBBD as optional at ingest stage, but require good feature rows for model training/prediction.
