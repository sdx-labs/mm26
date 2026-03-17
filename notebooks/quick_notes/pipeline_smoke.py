from __future__ import annotations

import json
from pathlib import Path
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
LATEST = ROOT / "artifacts" / "latest"

TEAM_PATH = LATEST / "gold" / "team_season_features.parquet"
PAIR_PATH = LATEST / "gold" / "pairwise_features.parquet"
SUB_PATH = LATEST / "submission.csv"

required_paths = [TEAM_PATH, PAIR_PATH, SUB_PATH]
missing = [str(path) for path in required_paths if not path.exists()]
if missing:
    print(
        json.dumps(
            {
                "error": "missing_artifacts",
                "message": "Run `python main.py run --mode manual --target-season 2026` first.",
                "missing_paths": missing,
            },
            indent=2,
        )
    )
    sys.exit(1)

team = pl.read_parquet(TEAM_PATH)
pair = pl.read_parquet(PAIR_PATH)
sub = pl.read_csv(SUB_PATH)

pred_min: float | None = None
pred_max: float | None = None
if "Pred" in sub.columns and sub.height > 0:
    stats = sub.select(pl.col("Pred").min().alias("pred_min"), pl.col("Pred").max().alias("pred_max")).row(0)
    pred_min = float(stats[0]) if stats[0] is not None else None
    pred_max = float(stats[1]) if stats[1] is not None else None

print(
    json.dumps(
        {
            "team_season_rows": team.height,
            "pairwise_rows": pair.height,
            "submission_rows": sub.height,
            "pred_min": pred_min,
            "pred_max": pred_max,
        },
        indent=2,
    )
)
