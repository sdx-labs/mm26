# CBBD API

The College Basketball Data API (CBBD API) is used in this repo to enrich the men's pipeline with historical game metadata, team box scores, and betting lines.

The client repository referenced by the team is:
- https://github.com/CFBD/cbbd-python/tree/main

The API key is stored in `.env` as `CBBD_API_KEY`.

## Stage 1 Usage in This Repo

- Scope: men-only enrichment in Stage 1.
- Pipeline call site: `src/mm26/pipeline.py`
- Runtime behavior:
  - `manual` mode backfills the men-only historical CBBD window aligned to the Kaggle detailed-results era requested by the project: seasons `2006-2025`.
  - `daily` mode fetches only recent completed games plus today and upcoming lines for `target_season`.
  - If `CBBD_API_KEY` is missing, the pipeline writes empty bronze CBBD parquet outputs and continues.
  - If the API call fails, the pipeline records per-dataset CBBD status in the run manifest instead of crashing the whole ingest stage.

## Installed Client Facts

- Authentication in the installed `cbbd` client uses `Configuration(access_token=api_key)`.
- The generated client emits the `Authorization: Bearer <token>` header from `access_token`.
- Do not rely on `configuration.api_key["Authorization"]` in this repo.

## Relevant Endpoints for This Repo

- `GamesApi.get_games`
  - Used for men's game metadata.
- `GamesApi.get_game_teams`
  - Used for men's team box score rows.
  - The returned model is already one row per team-game and includes nested `teamStats` and `opponentStats`.
- `LinesApi.get_lines`
  - Used for raw provider lines and game-level consensus spread features.

## API Response Constraints

- `get_games` returns the first 3000 matching games.
- `get_game_teams` returns the first 3000 matching games.
- `get_lines` returns the first 3000 matching games.
- Because of these caps, this repo should use date-windowed fetches rather than broad full-history requests.
- Historical backfill should iterate by season and date windows.
- Daily refresh should use narrow date windows around the current date.

## Line Data Notes

- The installed line model exposes:
  - `provider`
  - `spread`
  - `spreadOpen`
  - `overUnder`
  - `overUnderOpen`
  - `homeMoneyline`
  - `awayMoneyline`
- The installed client does not expose a full timestamped line-history series.
- For this repo, the v1 "accurate point spread" policy is:
  - store raw provider rows
  - compute a game-level consensus spread as the median of current provider spreads
  - store `spreadOpen` separately when available

## TeamID Mapping Behavior

- CBBD team names are normalized and mapped to Kaggle TeamID using:
  - `MTeams.csv`
  - `MTeamSpellings.csv`
- Mapping outputs:
  - `silver/team_id_map.parquet`
  - `reports/team_id_map_report.csv`
  - `reports/team_id_map_summary.json`
- CBBD team IDs are also preserved in the mapping output so the CBBD branch can keep native identifiers internally before bridging to Kaggle `TeamID`.

## Historical Alignment With Kaggle

- Kaggle men's detailed team box scores begin in season `2003`.
- This repo currently aligns the CBBD backfill to the last 20 complete seasons requested for model development:
  - train window target: `2006-2024`
  - holdout validation season: `2025`
  - current prediction season: `2026`

## Safety

- Do not print API secrets in logs or reports.
- Treat CBBD as optional at ingest stage.
- The enriched men's path should surface coverage and mapping quality so we can tell when CBBD data is too sparse to trust.
