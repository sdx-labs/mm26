from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
LATEST = ROOT / "artifacts" / "latest"

team = pl.read_parquet(LATEST / "gold" / "team_season_features.parquet")
pair = pl.read_parquet(LATEST / "gold" / "pairwise_features.parquet")
sub = pl.read_csv(LATEST / "submission.csv")

print({
    "team_season_rows": team.height,
    "pairwise_rows": pair.height,
    "submission_rows": sub.height,
    "pred_min": float(sub["Pred"].min()),
    "pred_max": float(sub["Pred"].max()),
})
