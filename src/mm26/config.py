"""Shared configuration, IO helpers, and small utilities used across all pipeline modules."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl


@dataclass
class PipelineConfig:
    project_root: Path
    data_dir: Path = field(init=False)
    source_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    run_id: str = field(init=False)
    target_season: int = 2026
    mode: str = "manual"
    cbbd_history_start: int = 2003
    cbbd_history_end: int = 2025
    holdout_season: int = 2025
    daily_completed_lookback_days: int = 3
    daily_upcoming_days: int = 3

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.source_dir = self.project_root / "src"
        self.artifacts_dir = self.project_root / "artifacts"
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        digest = hashlib.sha1(f"{stamp}:{self.mode}".encode("utf-8")).hexdigest()[:8]
        self.run_id = f"{stamp}_{digest}"

    @property
    def run_dir(self) -> Path:
        return self.artifacts_dir / "runs" / self.run_id

    @property
    def bronze_dir(self) -> Path:
        return self.run_dir / "bronze"

    @property
    def silver_dir(self) -> Path:
        return self.run_dir / "silver"

    @property
    def gold_dir(self) -> Path:
        return self.run_dir / "gold"

    @property
    def reports_dir(self) -> Path:
        return self.run_dir / "reports"


def normalize_name(value: str | None) -> str:
    if value is None:
        return ""
    text = value.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _infer_sex_from_team_id(team_id: int) -> str:
    return "M" if team_id < 2000 else "W"


def _invert_wloc_expr(col: str) -> pl.Expr:
    return (
        pl.when(pl.col(col) == "H")
        .then(pl.lit("A"))
        .when(pl.col(col) == "A")
        .then(pl.lit("H"))
        .otherwise(pl.col(col))
    )


def _required_kaggle_schemas() -> dict[str, list[str]]:
    return {
        "MTeams": ["TeamID", "TeamName", "FirstD1Season", "LastD1Season"],
        "WTeams": ["TeamID", "TeamName"],
        "MRegularSeasonDetailedResults": ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc", "NumOT"],
        "WRegularSeasonDetailedResults": ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc", "NumOT"],
        "MNCAATourneyCompactResults": ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"],
        "WNCAATourneyCompactResults": ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"],
        "MTeamSpellings": ["TeamNameSpelling", "TeamID"],
        "MMasseyOrdinals": ["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"],
        "SampleSubmissionStage2": ["ID", "Pred"],
    }


def _read_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def _write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _validation_split_metadata(config: PipelineConfig) -> dict[str, Any]:
    return {
        "train_start_season": config.cbbd_history_start,
        "train_end_season": config.holdout_season - 1,
        "holdout_season": config.holdout_season,
        "prediction_season": config.target_season,
    }
