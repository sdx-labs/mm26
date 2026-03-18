from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover
    LogisticRegression = None

try:
    from sklearn.isotonic import IsotonicRegression
except Exception:  # pragma: no cover
    IsotonicRegression = None


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


def _empty_cbbd_games_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "game_id": pl.Int64,
            "season": pl.Int64,
            "season_type": pl.Utf8,
            "status": pl.Utf8,
            "start_date": pl.Utf8,
            "home_team_id": pl.Int64,
            "home_team": pl.Utf8,
            "home_conference": pl.Utf8,
            "home_score": pl.Float64,
            "away_team_id": pl.Int64,
            "away_team": pl.Utf8,
            "away_conference": pl.Utf8,
            "away_score": pl.Float64,
            "neutral_site": pl.Boolean,
            "conference_game": pl.Boolean,
            "notes": pl.Utf8,
            "home_elo_start": pl.Float64,
            "home_elo_end": pl.Float64,
            "away_elo_start": pl.Float64,
            "away_elo_end": pl.Float64,
            "home_seed": pl.Float64,
            "away_seed": pl.Float64,
            "excitement": pl.Float64,
        }
    )


def _empty_cbbd_game_teams_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "game_id": pl.Int64,
            "season": pl.Int64,
            "season_type": pl.Utf8,
            "start_date": pl.Utf8,
            "team_id": pl.Int64,
            "team": pl.Utf8,
            "conference": pl.Utf8,
            "opponent_id": pl.Int64,
            "opponent": pl.Utf8,
            "opponent_conference": pl.Utf8,
            "neutral_site": pl.Boolean,
            "is_home": pl.Boolean,
            "conference_game": pl.Boolean,
            "game_type": pl.Utf8,
            "game_minutes": pl.Float64,
            "pace": pl.Float64,
        }
    )


def _empty_cbbd_lines_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "game_id": pl.Int64,
            "season": pl.Int64,
            "season_type": pl.Utf8,
            "start_date": pl.Utf8,
            "home_team_id": pl.Int64,
            "home_team": pl.Utf8,
            "away_team_id": pl.Int64,
            "away_team": pl.Utf8,
            "home_score": pl.Float64,
            "away_score": pl.Float64,
            "provider": pl.Utf8,
            "spread": pl.Float64,
            "spread_open": pl.Float64,
            "over_under": pl.Float64,
            "over_under_open": pl.Float64,
            "home_moneyline": pl.Float64,
            "away_moneyline": pl.Float64,
        }
    )


def _build_cbbd_configuration(api_key: str) -> Any:
    import cbbd

    return cbbd.Configuration(host="https://api.collegebasketballdata.com", access_token=api_key)


def _load_env_value(project_root: Path, key: str) -> str | None:
    env_path = project_root / ".env"
    if not env_path.exists():
        return None
    pattern = re.compile(rf"^{re.escape(key)}\s*=\s*['\"]?(.*?)['\"]?\s*$")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            value = match.group(1).strip()
            return value or None
    return None


def _to_utc_datetime(value: date, end_of_day: bool = False) -> datetime:
    clock = time(23, 59, 59) if end_of_day else time(0, 0, 0)
    return datetime.combine(value, clock, tzinfo=UTC)


def _historical_cbbd_seasons(config: PipelineConfig) -> list[int]:
    return list(range(config.cbbd_history_start, config.cbbd_history_end + 1))


def _historical_cbbd_windows(config: PipelineConfig) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for season in _historical_cbbd_seasons(config):
        season_start = date(season - 1, 11, 1)
        season_end = date(season, 4, 30)
        cursor = season_start
        while cursor <= season_end:
            window_end = min(cursor + timedelta(days=34), season_end)
            windows.append(
                {
                    "season": season,
                    "start": _to_utc_datetime(cursor),
                    "end": _to_utc_datetime(window_end, end_of_day=True),
                    "label": f"{season}:{cursor.isoformat()}:{window_end.isoformat()}",
                }
            )
            cursor = window_end + timedelta(days=1)
    return windows


def _daily_cbbd_windows(config: PipelineConfig) -> dict[str, Any]:
    today = datetime.now(UTC).date()
    completed_start = today - timedelta(days=config.daily_completed_lookback_days)
    upcoming_end = today + timedelta(days=config.daily_upcoming_days)
    return {
        "completed_start": _to_utc_datetime(completed_start),
        "completed_end": _to_utc_datetime(today, end_of_day=True),
        "lines_start": _to_utc_datetime(today),
        "lines_end": _to_utc_datetime(upcoming_end, end_of_day=True),
        "today": today.isoformat(),
    }


def _snake_case(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return value.replace("-", "_").lower()


def _flatten_nested(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}_{_snake_case(key)}" if prefix else _snake_case(key)
            _flatten_nested(child_prefix, item, output)
        return
    output[prefix] = value


def _coerce_str(value: Any) -> str | None:
    """Convert enum-like API values to plain strings."""
    if value is None:
        return None
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _normalize_api_payload(item: Any) -> dict[str, Any]:
    if hasattr(item, "to_dict"):
        return item.to_dict()
    return dict(item)


def _normalize_game_record(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "game_id": payload.get("id"),
        "season": payload.get("season"),
        "season_type": _coerce_str(payload.get("seasonType")),
        "status": _coerce_str(payload.get("status")),
        "start_date": str(payload.get("startDate")) if payload.get("startDate") is not None else None,
        "home_team_id": payload.get("homeTeamId"),
        "home_team": payload.get("homeTeam"),
        "home_conference": payload.get("homeConference"),
        "home_score": payload.get("homePoints"),
        "away_team_id": payload.get("awayTeamId"),
        "away_team": payload.get("awayTeam"),
        "away_conference": payload.get("awayConference"),
        "away_score": payload.get("awayPoints"),
        "neutral_site": payload.get("neutralSite"),
        "conference_game": payload.get("conferenceGame"),
        "notes": payload.get("gameNotes"),
        "home_elo_start": payload.get("homeTeamEloStart"),
        "home_elo_end": payload.get("homeTeamEloEnd"),
        "away_elo_start": payload.get("awayTeamEloStart"),
        "away_elo_end": payload.get("awayTeamEloEnd"),
        "home_seed": payload.get("homeSeed"),
        "away_seed": payload.get("awaySeed"),
        "excitement": payload.get("excitement"),
    }


def _normalize_game_team_record(payload: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "game_id": payload.get("gameId"),
        "season": payload.get("season"),
        "season_type": _coerce_str(payload.get("seasonType")),
        "start_date": str(payload.get("startDate")) if payload.get("startDate") is not None else None,
        "team_id": payload.get("teamId"),
        "team": payload.get("team"),
        "conference": payload.get("conference"),
        "opponent_id": payload.get("opponentId"),
        "opponent": payload.get("opponent"),
        "opponent_conference": payload.get("opponentConference"),
        "neutral_site": payload.get("neutralSite"),
        "is_home": payload.get("isHome"),
        "conference_game": payload.get("conferenceGame"),
        "game_type": payload.get("gameType"),
        "game_minutes": payload.get("gameMinutes"),
        "pace": payload.get("pace"),
    }
    _flatten_nested("team_stats", payload.get("teamStats") or {}, row)
    _flatten_nested("opponent_stats", payload.get("opponentStats") or {}, row)
    return row


def _normalize_line_record(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = {
        "game_id": payload.get("gameId"),
        "season": payload.get("season"),
        "season_type": _coerce_str(payload.get("seasonType")),
        "start_date": str(payload.get("startDate")) if payload.get("startDate") is not None else None,
        "home_team_id": payload.get("homeTeamId"),
        "home_team": payload.get("homeTeam"),
        "away_team_id": payload.get("awayTeamId"),
        "away_team": payload.get("awayTeam"),
        "home_score": payload.get("homeScore"),
        "away_score": payload.get("awayScore"),
    }
    lines = payload.get("lines") or []
    if not lines:
        rows.append(
            {
                **base,
                "provider": None,
                "spread": None,
                "spread_open": None,
                "over_under": None,
                "over_under_open": None,
                "home_moneyline": None,
                "away_moneyline": None,
            }
        )
        return rows
    for line in lines:
        line_payload = _normalize_api_payload(line)
        rows.append(
            {
                **base,
                "provider": line_payload.get("provider"),
                "spread": line_payload.get("spread"),
                "spread_open": line_payload.get("spreadOpen"),
                "over_under": line_payload.get("overUnder"),
                "over_under_open": line_payload.get("overUnderOpen"),
                "home_moneyline": line_payload.get("homeMoneyline"),
                "away_moneyline": line_payload.get("awayMoneyline"),
            }
        )
    return rows


def ingest_kaggle(config: PipelineConfig) -> dict[str, Any]:
    kaggle_dir = config.bronze_dir / "kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"files": {}, "source_dir": str(config.data_dir)}

    csv_files = sorted(config.data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No Kaggle CSV files found under {config.data_dir}")

    for csv_path in csv_files:
        stem = csv_path.stem
        df = pl.read_csv(csv_path)
        output_path = kaggle_dir / f"{stem}.parquet"
        _write_parquet(df, output_path)
        manifest["files"][stem] = {
            "source": str(csv_path),
            "artifact": str(output_path),
            "rows": df.height,
            "columns": df.columns,
        }

    return manifest


def _write_empty_cbbd_outputs(cbbd_dir: Path) -> dict[str, str]:
    paths = {
        "games": str(cbbd_dir / "games.parquet"),
        "game_teams": str(cbbd_dir / "game_teams.parquet"),
        "lines": str(cbbd_dir / "lines.parquet"),
    }
    _write_parquet(_empty_cbbd_games_df(), Path(paths["games"]))
    _write_parquet(_empty_cbbd_game_teams_df(), Path(paths["game_teams"]))
    _write_parquet(_empty_cbbd_lines_df(), Path(paths["lines"]))
    return paths


def _collect_cbbd_responses(config: PipelineConfig, api_bundle: Any, fetcher: Any) -> dict[str, list[Any]]:
    responses: dict[str, list[Any]] = {"games": [], "game_teams": [], "lines": []}

    if config.mode == "manual":
        for window in _historical_cbbd_windows(config):
            kwargs = {
                "season": window["season"],
                "start_date_range": window["start"],
                "end_date_range": window["end"],
            }
            responses["games"].extend(fetcher(api_bundle, "get_games", **kwargs))
            responses["game_teams"].extend(fetcher(api_bundle, "get_game_teams", **kwargs))
            responses["lines"].extend(fetcher(api_bundle, "get_lines", **kwargs))
        return responses

    daily = _daily_cbbd_windows(config)
    completed_kwargs = {
        "season": config.target_season,
        "start_date_range": daily["completed_start"],
        "end_date_range": daily["completed_end"],
    }
    lines_kwargs = {
        "season": config.target_season,
        "start_date_range": daily["lines_start"],
        "end_date_range": daily["lines_end"],
    }
    responses["games"].extend(fetcher(api_bundle, "get_games", **completed_kwargs))
    responses["game_teams"].extend(fetcher(api_bundle, "get_game_teams", **completed_kwargs))
    responses["lines"].extend(fetcher(api_bundle, "get_lines", **lines_kwargs))
    return responses


def ingest_cbbd(
    config: PipelineConfig,
    api_key: str | None = None,
    client_factory: Any = None,
) -> dict[str, Any]:
    cbbd_dir = config.bronze_dir / "cbbd"
    cbbd_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = _write_empty_cbbd_outputs(cbbd_dir)
    manifest: dict[str, Any] = {
        "historical_seasons": _historical_cbbd_seasons(config),
        "daily_refresh_window": _daily_cbbd_windows(config),
        "datasets": {
            name: {
                "status": "skipped",
                "reason": None,
                "rows": 0,
                "artifact": artifact,
            }
            for name, artifact in artifact_paths.items()
        },
    }

    resolved_api_key = api_key or os.getenv("CBBD_API_KEY") or _load_env_value(config.project_root, "CBBD_API_KEY")
    if not resolved_api_key:
        for dataset in manifest["datasets"].values():
            dataset["reason"] = "CBBD_API_KEY missing"
        return manifest

    def _fetch(api_bundle: Any, name: str, **kwargs: Any) -> list[Any]:
        if hasattr(api_bundle, name):
            return list(getattr(api_bundle, name)(**kwargs))
        raise AttributeError(name)

    try:
        if client_factory is None:
            import cbbd

            configuration = _build_cbbd_configuration(resolved_api_key)
            with cbbd.ApiClient(configuration) as api_client:
                games_api = cbbd.GamesApi(api_client)
                lines_api = cbbd.LinesApi(api_client)
                api_bundle = type(
                    "ApiBundle",
                    (),
                    {
                        "get_games": games_api.get_games,
                        "get_game_teams": games_api.get_game_teams,
                        "get_lines": lines_api.get_lines,
                    },
                )()
                api_responses = _collect_cbbd_responses(config, api_bundle, _fetch)
        else:
            api_bundle = client_factory(resolved_api_key)
            api_responses = _collect_cbbd_responses(config, api_bundle, _fetch)
    except Exception as exc:  # pragma: no cover
        for dataset in manifest["datasets"].values():
            dataset["status"] = "error"
            dataset["reason"] = str(exc)
        return manifest

    games_rows = [_normalize_game_record(_normalize_api_payload(item)) for item in api_responses["games"]]
    game_team_rows = [_normalize_game_team_record(_normalize_api_payload(item)) for item in api_responses["game_teams"]]
    line_rows = [
        row
        for item in api_responses["lines"]
        for row in _normalize_line_record(_normalize_api_payload(item))
    ]

    games_df = pl.DataFrame(games_rows, schema=_empty_cbbd_games_df().schema) if games_rows else _empty_cbbd_games_df()
    game_teams_df = pl.DataFrame(game_team_rows, schema=_empty_cbbd_game_teams_df().schema) if game_team_rows else _empty_cbbd_game_teams_df()
    lines_df = pl.DataFrame(line_rows, schema=_empty_cbbd_lines_df().schema) if line_rows else _empty_cbbd_lines_df()

    if games_df.height > 0:
        games_df = games_df.unique(subset=["game_id"], keep="last")
    if game_teams_df.height > 0:
        game_teams_df = game_teams_df.unique(subset=["game_id", "team_id"], keep="last")
    if lines_df.height > 0:
        lines_df = lines_df.unique(subset=["game_id", "provider"], keep="last")

    _write_parquet(games_df, Path(artifact_paths["games"]))
    _write_parquet(game_teams_df, Path(artifact_paths["game_teams"]))
    _write_parquet(lines_df, Path(artifact_paths["lines"]))

    for name, df in [("games", games_df), ("game_teams", game_teams_df), ("lines", lines_df)]:
        manifest["datasets"][name]["status"] = "ok"
        manifest["datasets"][name]["rows"] = df.height

    return manifest


def run_ingest(config: PipelineConfig) -> dict[str, Any]:
    kaggle_manifest = ingest_kaggle(config)
    cbbd_manifest = ingest_cbbd(config)
    return {"kaggle": kaggle_manifest, "cbbd": cbbd_manifest}


def run_validations(config: PipelineConfig, ingest_manifest: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {"checks": [], "failed": False, "data_dir": str(config.data_dir)}
    files = ingest_manifest["kaggle"]["files"]
    required = _required_kaggle_schemas()

    for dataset_name, required_columns in required.items():
        entry = files.get(dataset_name)
        if entry is None:
            results["checks"].append({"dataset": dataset_name, "ok": False, "error": "missing dataset"})
            results["failed"] = True
            continue

        found_columns = set(entry["columns"])
        missing = sorted(set(required_columns) - found_columns)
        ok = not missing and entry["rows"] > 0
        results["checks"].append(
            {
                "dataset": dataset_name,
                "ok": ok,
                "missing_columns": missing,
                "rows": entry["rows"],
            }
        )
        if not ok:
            results["failed"] = True

    sample = _read_parquet(Path(files["SampleSubmissionStage2"]["artifact"]))
    id_parts = sample.with_columns(
        pl.col("ID").str.split("_").list.get(0).cast(pl.Int64).alias("Season")
    )
    latest_season = int(id_parts["Season"].max())
    freshness_ok = latest_season >= config.target_season
    results["checks"].append(
        {
            "dataset": "SampleSubmissionStage2",
            "ok": freshness_ok,
            "latest_season": latest_season,
            "expected_target": config.target_season,
        }
    )
    if not freshness_ok:
        results["failed"] = True

    results["cbbd_manifest_shape"] = sorted(ingest_manifest["cbbd"]["datasets"].keys())
    if config.mode == "daily":
        cbbd_datasets = ingest_manifest["cbbd"]["datasets"]
        games_ok = cbbd_datasets.get("games", {}).get("status") == "ok" and cbbd_datasets.get("games", {}).get("rows", 0) > 0
        lines_ok = cbbd_datasets.get("lines", {}).get("status") == "ok" and cbbd_datasets.get("lines", {}).get("rows", 0) > 0
        results["checks"].append(
            {
                "dataset": "CBBD_daily_games_lines",
                "ok": bool(games_ok and lines_ok),
                "games_rows": cbbd_datasets.get("games", {}).get("rows", 0),
                "lines_rows": cbbd_datasets.get("lines", {}).get("rows", 0),
            }
        )
    report_path = config.reports_dir / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")

    if results["failed"]:
        raise ValueError(f"Validation failed. See {report_path}")

    return results


def _expand_team_games(detailed: pl.DataFrame, sex: str) -> pl.DataFrame:
    # Determine if detailed box-score columns are present
    has_detail = "WFGM" in detailed.columns

    base_win = [
        pl.lit(sex).alias("sex"),
        pl.col("Season").cast(pl.Int64).alias("season"),
        pl.col("DayNum").cast(pl.Int64).alias("day_num"),
        pl.col("WTeamID").cast(pl.Int64).alias("team_id"),
        pl.col("LTeamID").cast(pl.Int64).alias("opp_team_id"),
        pl.col("WScore").cast(pl.Float64).alias("team_score"),
        pl.col("LScore").cast(pl.Float64).alias("opp_score"),
        pl.col("WLoc").alias("team_loc"),
        pl.col("NumOT").cast(pl.Int64).alias("num_ot"),
        pl.lit(1).alias("win"),
    ]
    base_loss = [
        pl.lit(sex).alias("sex"),
        pl.col("Season").cast(pl.Int64).alias("season"),
        pl.col("DayNum").cast(pl.Int64).alias("day_num"),
        pl.col("LTeamID").cast(pl.Int64).alias("team_id"),
        pl.col("WTeamID").cast(pl.Int64).alias("opp_team_id"),
        pl.col("LScore").cast(pl.Float64).alias("team_score"),
        pl.col("WScore").cast(pl.Float64).alias("opp_score"),
        _invert_wloc_expr("WLoc").alias("team_loc"),
        pl.col("NumOT").cast(pl.Int64).alias("num_ot"),
        pl.lit(0).alias("win"),
    ]

    if has_detail:
        detail_win = [
            pl.col("WFGM").cast(pl.Float64).alias("fgm"),
            pl.col("WFGA").cast(pl.Float64).alias("fga"),
            pl.col("WFGM3").cast(pl.Float64).alias("fgm3"),
            pl.col("WFGA3").cast(pl.Float64).alias("fga3"),
            pl.col("WFTM").cast(pl.Float64).alias("ftm"),
            pl.col("WFTA").cast(pl.Float64).alias("fta"),
            pl.col("WOR").cast(pl.Float64).alias("oreb"),
            pl.col("WDR").cast(pl.Float64).alias("dreb"),
            pl.col("WAst").cast(pl.Float64).alias("ast"),
            pl.col("WTO").cast(pl.Float64).alias("tov"),
            pl.col("WStl").cast(pl.Float64).alias("stl"),
            pl.col("WBlk").cast(pl.Float64).alias("blk"),
            pl.col("WPF").cast(pl.Float64).alias("pf"),
            pl.col("LFGM").cast(pl.Float64).alias("opp_fgm"),
            pl.col("LFGA").cast(pl.Float64).alias("opp_fga"),
            pl.col("LFGM3").cast(pl.Float64).alias("opp_fgm3"),
            pl.col("LFGA3").cast(pl.Float64).alias("opp_fga3"),
            pl.col("LFTM").cast(pl.Float64).alias("opp_ftm"),
            pl.col("LFTA").cast(pl.Float64).alias("opp_fta"),
            pl.col("LOR").cast(pl.Float64).alias("opp_oreb"),
            pl.col("LDR").cast(pl.Float64).alias("opp_dreb"),
            pl.col("LAst").cast(pl.Float64).alias("opp_ast"),
            pl.col("LTO").cast(pl.Float64).alias("opp_tov"),
            pl.col("LStl").cast(pl.Float64).alias("opp_stl"),
            pl.col("LBlk").cast(pl.Float64).alias("opp_blk"),
            pl.col("LPF").cast(pl.Float64).alias("opp_pf"),
        ]
        detail_loss = [
            pl.col("LFGM").cast(pl.Float64).alias("fgm"),
            pl.col("LFGA").cast(pl.Float64).alias("fga"),
            pl.col("LFGM3").cast(pl.Float64).alias("fgm3"),
            pl.col("LFGA3").cast(pl.Float64).alias("fga3"),
            pl.col("LFTM").cast(pl.Float64).alias("ftm"),
            pl.col("LFTA").cast(pl.Float64).alias("fta"),
            pl.col("LOR").cast(pl.Float64).alias("oreb"),
            pl.col("LDR").cast(pl.Float64).alias("dreb"),
            pl.col("LAst").cast(pl.Float64).alias("ast"),
            pl.col("LTO").cast(pl.Float64).alias("tov"),
            pl.col("LStl").cast(pl.Float64).alias("stl"),
            pl.col("LBlk").cast(pl.Float64).alias("blk"),
            pl.col("LPF").cast(pl.Float64).alias("pf"),
            pl.col("WFGM").cast(pl.Float64).alias("opp_fgm"),
            pl.col("WFGA").cast(pl.Float64).alias("opp_fga"),
            pl.col("WFGM3").cast(pl.Float64).alias("opp_fgm3"),
            pl.col("WFGA3").cast(pl.Float64).alias("opp_fga3"),
            pl.col("WFTM").cast(pl.Float64).alias("opp_ftm"),
            pl.col("WFTA").cast(pl.Float64).alias("opp_fta"),
            pl.col("WOR").cast(pl.Float64).alias("opp_oreb"),
            pl.col("WDR").cast(pl.Float64).alias("opp_dreb"),
            pl.col("WAst").cast(pl.Float64).alias("opp_ast"),
            pl.col("WTO").cast(pl.Float64).alias("opp_tov"),
            pl.col("WStl").cast(pl.Float64).alias("opp_stl"),
            pl.col("WBlk").cast(pl.Float64).alias("opp_blk"),
            pl.col("WPF").cast(pl.Float64).alias("opp_pf"),
        ]
        base_win.extend(detail_win)
        base_loss.extend(detail_loss)

    win_side = detailed.select(base_win)
    loss_side = detailed.select(base_loss)
    return pl.concat([win_side, loss_side], how="vertical")


def _build_team_id_map(config: PipelineConfig, files: dict[str, Any]) -> tuple[pl.DataFrame, dict[str, Any]]:
    mteams = _read_parquet(Path(files["MTeams"]["artifact"]))
    spellings = _read_parquet(Path(files["MTeamSpellings"]["artifact"]))
    cbbd_games = _read_parquet(config.bronze_dir / "cbbd" / "games.parquet")
    cbbd_game_teams = _read_parquet(config.bronze_dir / "cbbd" / "game_teams.parquet")
    cbbd_lines = _read_parquet(config.bronze_dir / "cbbd" / "lines.parquet")

    kaggle_names = pl.concat(
        [
            mteams.select(
                pl.col("TeamID").cast(pl.Int64).alias("team_id"),
                pl.col("TeamName").cast(pl.Utf8).alias("team_name_raw"),
            ),
            spellings.select(
                pl.col("TeamID").cast(pl.Int64).alias("team_id"),
                pl.col("TeamNameSpelling").cast(pl.Utf8).alias("team_name_raw"),
            ),
        ],
        how="vertical",
    ).with_columns(pl.col("team_name_raw").map_elements(normalize_name, return_dtype=pl.Utf8).alias("normalized_name"))

    kaggle_map = kaggle_names.group_by("normalized_name").agg(pl.col("team_id").min().alias("team_id"))

    cbbd_name_frames: list[pl.DataFrame] = []
    if cbbd_games.height > 0:
        cbbd_name_frames.extend(
            [
                cbbd_games.select(
                    pl.col("home_team").cast(pl.Utf8).alias("cbbd_team_name"),
                    pl.col("home_team_id").cast(pl.Int64).alias("cbbd_team_id"),
                ),
                cbbd_games.select(
                    pl.col("away_team").cast(pl.Utf8).alias("cbbd_team_name"),
                    pl.col("away_team_id").cast(pl.Int64).alias("cbbd_team_id"),
                ),
            ]
        )
    if cbbd_game_teams.height > 0:
        cbbd_name_frames.append(
            cbbd_game_teams.select(
                pl.col("team").cast(pl.Utf8).alias("cbbd_team_name"),
                pl.col("team_id").cast(pl.Int64).alias("cbbd_team_id"),
            )
        )
    if cbbd_lines.height > 0:
        cbbd_name_frames.extend(
            [
                cbbd_lines.select(
                    pl.col("home_team").cast(pl.Utf8).alias("cbbd_team_name"),
                    pl.col("home_team_id").cast(pl.Int64).alias("cbbd_team_id"),
                ),
                cbbd_lines.select(
                    pl.col("away_team").cast(pl.Utf8).alias("cbbd_team_name"),
                    pl.col("away_team_id").cast(pl.Int64).alias("cbbd_team_id"),
                ),
            ]
        )

    if not cbbd_name_frames:
        mapping = pl.DataFrame(
            schema={
                "cbbd_team_name": pl.Utf8,
                "cbbd_team_id": pl.Int64,
                "normalized_name": pl.Utf8,
                "team_id": pl.Int64,
                "mapped": pl.Boolean,
            }
        )
        summary = {"total": 0, "mapped": 0, "unmapped": 0, "mapped_pct": 0.0}
        return mapping, summary

    cbbd_names = pl.concat(cbbd_name_frames, how="vertical").drop_nulls("cbbd_team_name").unique()
    mapping = cbbd_names.with_columns(
        pl.col("cbbd_team_name").map_elements(normalize_name, return_dtype=pl.Utf8).alias("normalized_name")
    ).join(kaggle_map, on="normalized_name", how="left")
    mapping = mapping.with_columns(pl.col("team_id").is_not_null().alias("mapped"))

    total = mapping.height
    mapped = int(mapping.filter(pl.col("mapped")).height)
    unmapped = total - mapped
    mapped_pct = 0.0 if total == 0 else mapped / total
    summary = {"total": total, "mapped": mapped, "unmapped": unmapped, "mapped_pct": mapped_pct}
    return mapping, summary


def _aggregate_consensus_lines(lines_df: pl.DataFrame) -> pl.DataFrame:
    if lines_df.height == 0:
        return pl.DataFrame(
            schema={
                "game_id": pl.Int64,
                "season": pl.Int64,
                "start_date": pl.Utf8,
                "home_team_id": pl.Int64,
                "away_team_id": pl.Int64,
                "consensus_home_spread": pl.Float64,
                "consensus_home_spread_open": pl.Float64,
                "consensus_over_under": pl.Float64,
                "provider_count": pl.Int64,
            }
        )

    return (
        lines_df.group_by(["game_id", "season", "start_date", "home_team_id", "away_team_id"])
        .agg(
            pl.col("spread").drop_nulls().median().alias("consensus_home_spread"),
            pl.col("spread_open").drop_nulls().median().alias("consensus_home_spread_open"),
            pl.col("over_under").drop_nulls().median().alias("consensus_over_under"),
            pl.col("provider").drop_nulls().n_unique().alias("provider_count"),
        )
        .sort(["season", "start_date", "game_id"])
    )


def _attach_kaggle_team_ids_to_consensus(consensus: pl.DataFrame, team_map: pl.DataFrame) -> pl.DataFrame:
    if consensus.height == 0:
        return consensus.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("kaggle_home_team_id"),
            pl.lit(None, dtype=pl.Int64).alias("kaggle_away_team_id"),
            pl.lit(None, dtype=pl.Int64).alias("team_low"),
            pl.lit(None, dtype=pl.Int64).alias("team_high"),
            pl.lit(None, dtype=pl.Float64).alias("consensus_low_spread"),
        )

    cbbd_map = team_map.select(
        pl.col("cbbd_team_id").cast(pl.Int64),
        pl.col("team_id").cast(pl.Int64).alias("kaggle_team_id"),
    ).drop_nulls().unique()
    home_map = cbbd_map.rename({"cbbd_team_id": "home_team_id", "kaggle_team_id": "kaggle_home_team_id"})
    away_map = cbbd_map.rename({"cbbd_team_id": "away_team_id", "kaggle_team_id": "kaggle_away_team_id"})

    return (
        consensus.join(home_map, on="home_team_id", how="left")
        .join(away_map, on="away_team_id", how="left")
        .with_columns(
            pl.min_horizontal("kaggle_home_team_id", "kaggle_away_team_id").alias("team_low"),
            pl.max_horizontal("kaggle_home_team_id", "kaggle_away_team_id").alias("team_high"),
        )
        .with_columns(
            pl.when(pl.col("team_low").is_null() | pl.col("team_high").is_null())
            .then(None)
            .when(pl.col("team_low") == pl.col("kaggle_home_team_id"))
            .then(pl.col("consensus_home_spread"))
            .otherwise(-pl.col("consensus_home_spread"))
            .alias("consensus_low_spread")
        )
    )


def _build_cbbd_game_fact(config: PipelineConfig, team_map: pl.DataFrame) -> pl.DataFrame:
    game_teams = _read_parquet(config.bronze_dir / "cbbd" / "game_teams.parquet")
    if game_teams.height == 0:
        return game_teams

    mapped = team_map.select(
        pl.col("cbbd_team_id").cast(pl.Int64).alias("team_id"),
        pl.col("team_id").cast(pl.Int64).alias("kaggle_team_id"),
    ).drop_nulls().unique()
    opponent_map = mapped.rename({"team_id": "opponent_id", "kaggle_team_id": "kaggle_opp_team_id"})

    return (
        game_teams.join(mapped, on="team_id", how="left")
        .join(opponent_map, on="opponent_id", how="left")
        .with_columns(
            pl.lit("M").alias("sex"),
            pl.col("start_date").str.slice(0, 10).alias("game_date"),
        )
        .sort(["season", "start_date", "game_id", "team_id"])
    )


def _build_cbbd_games_clean(config: PipelineConfig, team_map: pl.DataFrame) -> pl.DataFrame:
    games = _read_parquet(config.bronze_dir / "cbbd" / "games.parquet")
    if games.height == 0:
        return games

    cbbd_map = team_map.select(
        pl.col("cbbd_team_id").cast(pl.Int64),
        pl.col("team_id").cast(pl.Int64).alias("kaggle_team_id"),
    ).drop_nulls().unique()
    home_map = cbbd_map.rename({"cbbd_team_id": "home_team_id", "kaggle_team_id": "kaggle_home_team_id"})
    away_map = cbbd_map.rename({"cbbd_team_id": "away_team_id", "kaggle_team_id": "kaggle_away_team_id"})

    return (
        games.join(home_map, on="home_team_id", how="left")
        .join(away_map, on="away_team_id", how="left")
        .with_columns(
            pl.col("start_date").str.slice(0, 10).alias("game_date"),
            pl.min_horizontal("kaggle_home_team_id", "kaggle_away_team_id").alias("team_low"),
            pl.max_horizontal("kaggle_home_team_id", "kaggle_away_team_id").alias("team_high"),
        )
        .sort(["season", "start_date", "game_id"])
    )


def _build_cbbd_lines_clean(config: PipelineConfig, team_map: pl.DataFrame) -> pl.DataFrame:
    lines = _read_parquet(config.bronze_dir / "cbbd" / "lines.parquet")
    if lines.height == 0:
        return lines

    cbbd_map = team_map.select(
        pl.col("cbbd_team_id").cast(pl.Int64),
        pl.col("team_id").cast(pl.Int64).alias("kaggle_team_id"),
    ).drop_nulls().unique()
    home_map = cbbd_map.rename({"cbbd_team_id": "home_team_id", "kaggle_team_id": "kaggle_home_team_id"})
    away_map = cbbd_map.rename({"cbbd_team_id": "away_team_id", "kaggle_team_id": "kaggle_away_team_id"})

    return (
        lines.join(home_map, on="home_team_id", how="left")
        .join(away_map, on="away_team_id", how="left")
        .with_columns(
            pl.col("start_date").str.slice(0, 10).alias("game_date"),
            pl.min_horizontal("kaggle_home_team_id", "kaggle_away_team_id").alias("team_low"),
            pl.max_horizontal("kaggle_home_team_id", "kaggle_away_team_id").alias("team_high"),
        )
        .sort(["season", "start_date", "game_id", "provider"])
    )


def _build_cbbd_coverage_summary(cbbd_game_fact: pl.DataFrame) -> dict[str, Any]:
    if cbbd_game_fact.height == 0:
        return {"seasons": [], "rows": 0}
    summary_rows = (
        cbbd_game_fact.group_by("season")
        .agg(
            pl.len().alias("team_game_rows"),
            pl.col("game_id").n_unique().alias("games"),
            pl.col("kaggle_team_id").drop_nulls().n_unique().alias("mapped_teams"),
        )
        .sort("season")
        .to_dicts()
    )
    return {"rows": cbbd_game_fact.height, "seasons": summary_rows}


def run_transform(config: PipelineConfig, ingest_manifest: dict[str, Any]) -> dict[str, Any]:
    files = ingest_manifest["kaggle"]["files"]

    mteams = _read_parquet(Path(files["MTeams"]["artifact"]))
    wteams = _read_parquet(Path(files["WTeams"]["artifact"]))

    team_dim_m = mteams.select(
        pl.lit("M").alias("sex"),
        pl.col("TeamID").cast(pl.Int64).alias("team_id"),
        pl.col("TeamName").cast(pl.Utf8).alias("team_name"),
        pl.col("FirstD1Season").cast(pl.Int64).alias("first_d1_season"),
        pl.col("LastD1Season").cast(pl.Int64).alias("last_d1_season"),
    )
    team_dim_w = wteams.select(
        pl.lit("W").alias("sex"),
        pl.col("TeamID").cast(pl.Int64).alias("team_id"),
        pl.col("TeamName").cast(pl.Utf8).alias("team_name"),
        pl.lit(None, dtype=pl.Int64).alias("first_d1_season"),
        pl.lit(None, dtype=pl.Int64).alias("last_d1_season"),
    )
    team_dim = pl.concat([team_dim_m, team_dim_w], how="vertical")

    m_reg_detail = _read_parquet(Path(files["MRegularSeasonDetailedResults"]["artifact"]))
    w_reg_detail = _read_parquet(Path(files["WRegularSeasonDetailedResults"]["artifact"]))

    team_games = (
        pl.concat(
            [
                _expand_team_games(m_reg_detail, "M"),
                _expand_team_games(w_reg_detail, "W"),
            ],
            how="vertical",
        )
        .with_columns(
            pl.min_horizontal(pl.col("team_id"), pl.col("opp_team_id")).alias("team_low"),
            pl.max_horizontal(pl.col("team_id"), pl.col("opp_team_id")).alias("team_high"),
        )
        .with_columns(
            (
                pl.col("sex")
                + pl.lit("_")
                + pl.col("season").cast(pl.Utf8)
                + pl.lit("_")
                + pl.col("day_num").cast(pl.Utf8)
                + pl.lit("_")
                + pl.col("team_low").cast(pl.Utf8)
                + pl.lit("_")
                + pl.col("team_high").cast(pl.Utf8)
            ).alias("game_key")
        )
    )

    quality_flags = team_games.select(
        "sex",
        "season",
        "day_num",
        "team_id",
        "opp_team_id",
        "game_key",
        (
            pl.col("team_score").is_not_null()
            & pl.col("opp_score").is_not_null()
            & pl.col("team_id").is_not_null()
            & pl.col("opp_team_id").is_not_null()
        ).alias("row_quality_pass")
    )

    team_id_map, map_summary = _build_team_id_map(config, files)
    cbbd_game_fact = _build_cbbd_game_fact(config, team_id_map)
    cbbd_lines = _read_parquet(config.bronze_dir / "cbbd" / "lines.parquet")
    cbbd_line_consensus = _attach_kaggle_team_ids_to_consensus(_aggregate_consensus_lines(cbbd_lines), team_id_map)
    cbbd_games_clean = _build_cbbd_games_clean(config, team_id_map)
    cbbd_lines_clean = _build_cbbd_lines_clean(config, team_id_map)
    coverage_summary = _build_cbbd_coverage_summary(cbbd_game_fact)

    _write_parquet(team_dim, config.silver_dir / "team_dim.parquet")
    _write_parquet(team_games, config.silver_dir / "game_fact.parquet")
    _write_parquet(team_id_map, config.silver_dir / "team_id_map.parquet")
    _write_parquet(quality_flags, config.silver_dir / "quality_flags.parquet")
    _write_parquet(cbbd_game_fact, config.silver_dir / "cbbd_game_fact.parquet")
    _write_parquet(cbbd_line_consensus, config.silver_dir / "cbbd_line_consensus.parquet")
    _write_parquet(cbbd_games_clean, config.silver_dir / "cbbd_games.parquet")
    _write_parquet(cbbd_lines_clean, config.silver_dir / "cbbd_lines.parquet")

    team_id_map.write_csv(config.reports_dir / "team_id_map_report.csv")
    (config.reports_dir / "team_id_map_summary.json").write_text(json.dumps(map_summary, indent=2, default=_json_default), encoding="utf-8")
    (config.reports_dir / "cbbd_coverage_summary.json").write_text(json.dumps(coverage_summary, indent=2, default=_json_default), encoding="utf-8")

    return {
        "team_dim_rows": team_dim.height,
        "game_fact_rows": team_games.height,
        "team_id_map_rows": team_id_map.height,
        "team_id_map_summary": map_summary,
        "cbbd_game_fact_rows": cbbd_game_fact.height,
        "cbbd_line_consensus_rows": cbbd_line_consensus.height,
        "cbbd_games_rows": cbbd_games_clean.height,
        "cbbd_lines_rows": cbbd_lines_clean.height,
        "cbbd_coverage_summary": coverage_summary,
    }


# ---------------------------------------------------------------------------
# Phase 1 – ELO Engine
# ---------------------------------------------------------------------------

def _compute_elo_ratings(game_fact: pl.DataFrame, k_factor: float = 20.0,
                         home_advantage: float = 100.0,
                         carry_over: float = 0.33) -> pl.DataFrame:
    """Compute game-by-game ELO ratings with MOV multiplier, home court, and carry-over."""
    games = (
        game_fact.filter(pl.col("team_id") == pl.col("team_low"))
        .sort(["sex", "season", "day_num", "game_key"])
        .select("sex", "season", "day_num", "game_key", "team_low", "team_high",
                "team_score", "opp_score", "win", "team_loc")
    )
    rows: list[dict[str, Any]] = []
    elo: dict[tuple[str, int, int], float] = {}
    prev_season_elo: dict[tuple[str, int], float] = {}

    for r in games.iter_rows(named=True):
        sex, season = r["sex"], r["season"]
        t_low, t_high = r["team_low"], r["team_high"]

        # Season carry-over: start from 1500 + carry_over * (prev - 1500)
        default_low = 1500.0 + carry_over * (prev_season_elo.get((sex, t_low), 1500.0) - 1500.0)
        default_high = 1500.0 + carry_over * (prev_season_elo.get((sex, t_high), 1500.0) - 1500.0)

        elo_low = elo.get((sex, season, t_low), default_low)
        elo_high = elo.get((sex, season, t_high), default_high)

        # Home court adjustment for expected win probability
        loc = r["team_loc"] if r["team_loc"] else "N"
        if loc == "H":
            elo_adj_low = elo_low + home_advantage
            elo_adj_high = elo_high
        elif loc == "A":
            elo_adj_low = elo_low
            elo_adj_high = elo_high + home_advantage
        else:
            elo_adj_low = elo_low
            elo_adj_high = elo_high

        expected_low = 1.0 / (1.0 + 10.0 ** (-(elo_adj_low - elo_adj_high) / 400.0))
        expected_margin = 25.0 * (expected_low * 2.0 - 1.0)

        actual_margin_low = float(r["team_score"] - r["opp_score"])
        actual_win_low = int(r["win"])

        # Margin-of-victory multiplier (FiveThirtyEight-style)
        elo_diff = abs(elo_low - elo_high)
        mov_mult = np.log(abs(actual_margin_low) + 1.0) * (2.2 / (elo_diff * 0.001 + 2.2))

        k_eff = k_factor * mov_mult

        elo_low_new = elo_low + k_eff * (actual_win_low - expected_low)
        elo_high_new = elo_high + k_eff * ((1 - actual_win_low) - (1 - expected_low))

        elo[(sex, season, t_low)] = elo_low_new
        elo[(sex, season, t_high)] = elo_high_new
        prev_season_elo[(sex, t_low)] = elo_low_new
        prev_season_elo[(sex, t_high)] = elo_high_new

        rows.append({
            "sex": sex, "season": season, "day_num": r["day_num"],
            "game_key": r["game_key"], "team_id": t_low,
            "elo_before": elo_low, "elo_after": elo_low_new,
            "expected_win_prob": expected_low, "expected_margin": expected_margin,
            "actual_win": actual_win_low, "actual_margin": actual_margin_low,
        })
        rows.append({
            "sex": sex, "season": season, "day_num": r["day_num"],
            "game_key": r["game_key"], "team_id": t_high,
            "elo_before": elo_high, "elo_after": elo_high_new,
            "expected_win_prob": 1.0 - expected_low, "expected_margin": -expected_margin,
            "actual_win": 1 - actual_win_low, "actual_margin": -actual_margin_low,
        })

    if not rows:
        return pl.DataFrame(schema={
            "sex": pl.Utf8, "season": pl.Int64, "day_num": pl.Int64,
            "game_key": pl.Utf8, "team_id": pl.Int64,
            "elo_before": pl.Float64, "elo_after": pl.Float64,
            "expected_win_prob": pl.Float64, "expected_margin": pl.Float64,
            "actual_win": pl.Int64, "actual_margin": pl.Float64,
        })
    return pl.DataFrame(rows).sort(["sex", "season", "day_num", "game_key", "team_id"])


# ---------------------------------------------------------------------------
# Phase 2 – Heat Score Engine
# ---------------------------------------------------------------------------

def _compute_heat_scores(elo_ratings: pl.DataFrame) -> pl.DataFrame:
    """Rolling over-performance relative to ELO expectations."""
    if elo_ratings.height == 0:
        return pl.DataFrame(schema={
            "sex": pl.Utf8, "season": pl.Int64, "day_num": pl.Int64,
            "team_id": pl.Int64, "heat_delta": pl.Float64,
            "heat_1g": pl.Float64, "heat_3g": pl.Float64, "heat_5g": pl.Float64,
        })

    base = (
        elo_ratings.with_columns(
            (pl.col("actual_margin").cast(pl.Float64) - pl.col("expected_margin")).alias("heat_delta")
        )
        .sort(["sex", "season", "team_id", "day_num"])
    )

    lagged = base
    for i in range(1, 6):
        lagged = lagged.with_columns(
            pl.col("heat_delta").shift(i).over(["sex", "season", "team_id"]).alias(f"_lag{i}")
        )

    return lagged.with_columns(
        pl.col("_lag1").alias("heat_1g"),
        pl.mean_horizontal("_lag1", "_lag2", "_lag3").alias("heat_3g"),
        pl.mean_horizontal("_lag1", "_lag2", "_lag3", "_lag4", "_lag5").alias("heat_5g"),
    ).select("sex", "season", "day_num", "team_id", "heat_delta", "heat_1g", "heat_3g", "heat_5g")


def _get_pre_tournament_heat(heat_scores: pl.DataFrame, tourney_cutoff_day: int = 132) -> pl.DataFrame:
    """Last heat row per team per season before the tournament."""
    return (
        heat_scores.filter(pl.col("day_num") <= tourney_cutoff_day)
        .sort(["sex", "season", "team_id", "day_num"])
        .group_by(["sex", "season", "team_id"])
        .last()
    )


def _load_seed_map(ingest_manifest: dict[str, Any]) -> pl.DataFrame:
    """Load tournament seeds for M and W, extracting numeric seed."""
    files = ingest_manifest["kaggle"]["files"]
    frames: list[pl.DataFrame] = []
    for key, sex in [("MNCAATourneySeeds", "M"), ("WNCAATourneySeeds", "W")]:
        if key not in files:
            continue
        df = _read_parquet(Path(files[key]["artifact"]))
        if "Seed" not in df.columns or "TeamID" not in df.columns:
            continue
        df = df.select(
            pl.lit(sex).alias("sex"),
            pl.col("Season").cast(pl.Int64).alias("season"),
            pl.col("TeamID").cast(pl.Int64).alias("team_id"),
            pl.col("Seed").str.extract(r"(\d+)", 1).cast(pl.Int64).alias("seed_num"),
        )
        frames.append(df)
    if not frames:
        return pl.DataFrame(schema={"sex": pl.Utf8, "season": pl.Int64, "team_id": pl.Int64, "seed_num": pl.Int64})
    return pl.concat(frames, how="vertical")


def run_elo_and_heat(config: PipelineConfig) -> dict[str, Any]:
    """Compute ELO ratings and heat scores from silver game_fact."""
    game_fact = _read_parquet(config.silver_dir / "game_fact.parquet")

    elo_ratings = _compute_elo_ratings(game_fact)
    heat_scores = _compute_heat_scores(elo_ratings)

    _write_parquet(elo_ratings, config.silver_dir / "elo_ratings.parquet")
    _write_parquet(heat_scores, config.silver_dir / "heat_scores.parquet")
    _write_parquet(elo_ratings, config.gold_dir / "elo_ratings.parquet")
    _write_parquet(heat_scores, config.gold_dir / "heat_scores.parquet")

    return {"elo_rows": elo_ratings.height, "heat_rows": heat_scores.height}


# ---------------------------------------------------------------------------
# Phase 5 – Monte Carlo Bracket Simulation helpers
# ---------------------------------------------------------------------------

def _build_prob_lookup(submission: pl.DataFrame) -> dict[tuple[int, int], float]:
    """Build {(team_low, team_high): pred_low_wins} lookup."""
    lookup: dict[tuple[int, int], float] = {}
    for row in submission.iter_rows(named=True):
        parts = row["ID"].split("_")
        lookup[(int(parts[1]), int(parts[2]))] = float(row["Pred"])
    return lookup


def _simulate_bracket(
    seeds_df: pl.DataFrame,
    slots_df: pl.DataFrame,
    prob_lookup: dict[tuple[int, int], float],
    n_sims: int = 100_000,
) -> dict[tuple[int, int], float]:
    """Vectorised Monte Carlo bracket simulation.  Returns {(t_low, t_high): frac_low_wins}."""
    if seeds_df.height == 0 or slots_df.height == 0:
        return {}

    seed_to_team: dict[str, int] = {}
    for row in seeds_df.iter_rows(named=True):
        seed_to_team[row["Seed"]] = int(row["TeamID"])
    if not seed_to_team:
        return {}

    slots = sorted(
        [{"slot": r["Slot"], "strong": r["StrongSeed"], "weak": r["WeakSeed"]}
         for r in slots_df.iter_rows(named=True)],
        key=lambda s: s["slot"],
    )

    all_keys: set[str] = set(seed_to_team.keys())
    for s in slots:
        all_keys.update([s["slot"], s["strong"], s["weak"]])
    key_to_idx = {k: i for i, k in enumerate(sorted(all_keys))}

    team_at = np.zeros((n_sims, len(key_to_idx)), dtype=np.int32)
    for seed, tid in seed_to_team.items():
        team_at[:, key_to_idx[seed]] = tid

    rng = np.random.default_rng(42)
    draws = rng.random((n_sims, len(slots)))

    win_counts: dict[tuple[int, int], int] = {}
    game_counts: dict[tuple[int, int], int] = {}

    for game_idx, slot_info in enumerate(slots):
        si = key_to_idx[slot_info["strong"]]
        wi = key_to_idx[slot_info["weak"]]
        oi = key_to_idx[slot_info["slot"]]

        ta = team_at[:, si]
        tb = team_at[:, wi]
        t_low = np.minimum(ta, tb)
        t_high = np.maximum(ta, tb)

        unique_matchups = np.unique(np.column_stack([t_low, t_high]), axis=0)

        for matchup in unique_matchups:
            ml, mh = int(matchup[0]), int(matchup[1])
            if ml == 0 or mh == 0:
                continue
            pair = (ml, mh)
            prob_low = prob_lookup.get(pair, 0.5)
            mask = (t_low == ml) & (t_high == mh)
            count = int(mask.sum())
            game_counts[pair] = game_counts.get(pair, 0) + count

            low_wins = draws[:, game_idx] < prob_low
            win_counts[pair] = win_counts.get(pair, 0) + int((low_wins & mask).sum())

            team_at[:, oi] = np.where(
                mask, np.where(low_wins, t_low, t_high), team_at[:, oi]
            )

    return {pair: win_counts.get(pair, 0) / cnt for pair, cnt in game_counts.items() if cnt > 0}


# ---------------------------------------------------------------------------
# Phase 3 – Enhanced team-season feature builder
# ---------------------------------------------------------------------------

def _build_team_season_features(
    team_games: pl.DataFrame,
    elo_ratings: pl.DataFrame | None = None,
    heat_scores: pl.DataFrame | None = None,
) -> pl.DataFrame:
    ordered = team_games.sort(["sex", "season", "team_id", "day_num"])

    has_detail = "fgm" in ordered.columns

    base_aggs = [
        pl.len().alias("games_played"),
        pl.sum("win").alias("wins"),
        (pl.len() - pl.sum("win")).alias("losses"),
        pl.mean("team_score").alias("avg_pts_for"),
        pl.mean("opp_score").alias("avg_pts_against"),
        (pl.col("team_score") - pl.col("opp_score")).mean().alias("avg_margin"),
        pl.mean("num_ot").alias("avg_num_ot"),
        pl.col("team_score").tail(5).mean().alias("last5_avg_pts_for"),
        pl.col("opp_score").tail(5).mean().alias("last5_avg_pts_against"),
        (pl.col("team_score") - pl.col("opp_score")).tail(5).mean().alias("last5_avg_margin"),
        pl.col("win").tail(5).mean().alias("last5_win_rate"),
    ]

    if has_detail:
        detail_aggs = [
            # Shooting efficiency
            (pl.sum("fgm") / pl.sum("fga")).alias("fg_pct"),
            (pl.sum("fgm3") / pl.sum("fga3")).alias("fg3_pct"),
            (pl.sum("ftm") / pl.sum("fta")).alias("ft_pct"),
            # Opponent shooting
            (pl.sum("opp_fgm") / pl.sum("opp_fga")).alias("opp_fg_pct"),
            # Rebounding
            ((pl.sum("oreb") + pl.sum("dreb")) - (pl.sum("opp_oreb") + pl.sum("opp_dreb"))).cast(pl.Float64).alias("total_reb_margin"),
            pl.mean("oreb").alias("avg_oreb"),
            pl.mean("dreb").alias("avg_dreb"),
            # Ball handling
            pl.mean("ast").alias("avg_ast"),
            pl.mean("tov").alias("avg_tov"),
            pl.mean("stl").alias("avg_stl"),
            pl.mean("blk").alias("avg_blk"),
            # Opponent turnovers (defensive forcing)
            pl.mean("opp_tov").alias("avg_opp_tov"),
            # Tempo proxy: possessions ~ FGA - OREB + TO + 0.475 * FTA
            (pl.col("fga") - pl.col("oreb") + pl.col("tov") + pl.col("fta") * 0.475).mean().alias("avg_possessions"),
            # Last-5 shooting trend
            (pl.col("fgm").tail(5).sum() / pl.col("fga").tail(5).sum()).alias("last5_fg_pct"),
        ]
        base_aggs.extend(detail_aggs)

    features = ordered.group_by(["sex", "season", "team_id"]).agg(
        base_aggs
    ).with_columns((pl.col("wins") / pl.col("games_played")).alias("win_rate"))

    if has_detail:
        # Compute derived features
        features = features.with_columns(
            ((pl.col("total_reb_margin")) / pl.col("games_played")).alias("avg_reb_margin"),
            (pl.col("avg_ast") / pl.col("avg_tov").clip(0.1, None)).alias("ast_to_ratio"),
            (pl.col("avg_stl") + pl.col("avg_blk")).alias("avg_stl_blk"),
        ).drop("total_reb_margin")
    else:
        features = features.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("fg_pct"),
            pl.lit(None, dtype=pl.Float64).alias("fg3_pct"),
            pl.lit(None, dtype=pl.Float64).alias("ft_pct"),
            pl.lit(None, dtype=pl.Float64).alias("opp_fg_pct"),
            pl.lit(None, dtype=pl.Float64).alias("avg_oreb"),
            pl.lit(None, dtype=pl.Float64).alias("avg_dreb"),
            pl.lit(None, dtype=pl.Float64).alias("avg_ast"),
            pl.lit(None, dtype=pl.Float64).alias("avg_tov"),
            pl.lit(None, dtype=pl.Float64).alias("avg_stl"),
            pl.lit(None, dtype=pl.Float64).alias("avg_blk"),
            pl.lit(None, dtype=pl.Float64).alias("avg_opp_tov"),
            pl.lit(None, dtype=pl.Float64).alias("avg_possessions"),
            pl.lit(None, dtype=pl.Float64).alias("avg_reb_margin"),
            pl.lit(None, dtype=pl.Float64).alias("ast_to_ratio"),
            pl.lit(None, dtype=pl.Float64).alias("avg_stl_blk"),
        )

    # Join season-end ELO (last elo_after before day 133)
    if elo_ratings is not None and elo_ratings.height > 0:
        season_end_elo = (
            elo_ratings.filter(pl.col("day_num") <= 132)
            .sort(["sex", "season", "team_id", "day_num"])
            .group_by(["sex", "season", "team_id"])
            .agg(pl.col("elo_after").last().alias("season_end_elo"))
        )
        features = features.join(season_end_elo, on=["sex", "season", "team_id"], how="left")

        # Strength of Schedule: average opponent ELO
        opp_elo = (
            elo_ratings.filter(pl.col("day_num") <= 132)
            .sort(["sex", "season", "team_id", "day_num"])
            .group_by(["sex", "season", "team_id"])
            .agg(pl.col("elo_after").last().alias("opp_elo"))
        )
        # Join opponent ELO via game_fact to compute SOS
        if team_games.height > 0:
            sos = (
                team_games.filter(pl.col("day_num") <= 132)
                .select("sex", "season", "team_id", "opp_team_id")
                .join(
                    opp_elo.rename({"team_id": "opp_team_id"}),
                    on=["sex", "season", "opp_team_id"],
                    how="left",
                )
                .group_by(["sex", "season", "team_id"])
                .agg(pl.mean("opp_elo").alias("sos"))
            )
            features = features.join(sos, on=["sex", "season", "team_id"], how="left")
        else:
            features = features.with_columns(pl.lit(None, dtype=pl.Float64).alias("sos"))
    else:
        features = features.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("season_end_elo"),
            pl.lit(None, dtype=pl.Float64).alias("sos"),
        )

    # Join pre-tournament heat scores
    if heat_scores is not None and heat_scores.height > 0:
        pre_heat = _get_pre_tournament_heat(heat_scores)
        features = features.join(
            pre_heat.select("sex", "season", "team_id", "heat_1g", "heat_3g", "heat_5g").rename({
                "heat_1g": "pre_tourney_heat_1g",
                "heat_3g": "pre_tourney_heat_3g",
                "heat_5g": "pre_tourney_heat_5g",
            }),
            on=["sex", "season", "team_id"],
            how="left",
        )
        # Enhanced heat: heat_trend (heating up/cooling off), abs_heat
        features = features.with_columns(
            (pl.col("pre_tourney_heat_5g") - pl.col("pre_tourney_heat_1g")).alias("heat_trend"),
            pl.col("pre_tourney_heat_5g").abs().alias("abs_heat_5g"),
        )
    else:
        features = features.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("pre_tourney_heat_1g"),
            pl.lit(None, dtype=pl.Float64).alias("pre_tourney_heat_3g"),
            pl.lit(None, dtype=pl.Float64).alias("pre_tourney_heat_5g"),
            pl.lit(None, dtype=pl.Float64).alias("heat_trend"),
            pl.lit(None, dtype=pl.Float64).alias("abs_heat_5g"),
        )

    return features


def _line_features_for_pairs(sample: pl.DataFrame, cbbd_line_consensus: pl.DataFrame, season: int) -> pl.DataFrame:
    parsed = sample.with_columns(
        pl.col("ID").str.split("_").list.get(0).cast(pl.Int64).alias("season"),
        pl.col("ID").str.split("_").list.get(1).cast(pl.Int64).alias("team_low"),
        pl.col("ID").str.split("_").list.get(2).cast(pl.Int64).alias("team_high"),
    )
    if cbbd_line_consensus.height == 0:
        return parsed.select(
            "ID",
            "season",
            "team_low",
            "team_high",
            pl.lit(None, dtype=pl.Float64).alias("consensus_low_spread"),
            pl.lit(None, dtype=pl.Float64).alias("consensus_over_under"),
            pl.lit(0, dtype=pl.Int64).alias("provider_count"),
        )

    latest = (
        cbbd_line_consensus.filter(pl.col("season") == season)
        .filter(pl.col("team_low").is_not_null() & pl.col("team_high").is_not_null())
        .sort(["start_date", "game_id"])
        .group_by(["season", "team_low", "team_high"])
        .agg(
            pl.col("consensus_low_spread").last().alias("consensus_low_spread"),
            pl.col("consensus_over_under").last().alias("consensus_over_under"),
            pl.col("provider_count").last().alias("provider_count"),
        )
    )
    return parsed.join(latest, on=["season", "team_low", "team_high"], how="left").with_columns(
        pl.col("provider_count").fill_null(0)
    )


def _team_feature_rename_map(columns: list[str]) -> dict[str, str]:
    """Build rename mapping for team_features → low/high suffixed columns."""
    skip = {"sex", "season"}
    mapping: dict[str, str] = {}
    for col in columns:
        if col in skip:
            continue
        if col == "team_id":
            mapping["team_id"] = "team_low"  # will be overridden for high
        else:
            mapping[col] = f"{col}_low"
    # Build both low and high
    low_map = dict(mapping)
    high_map = {}
    for col in columns:
        if col in skip:
            continue
        if col == "team_id":
            high_map["team_id"] = "team_high"
        else:
            high_map[col] = f"{col}_high"
    # Return combined (caller splits by suffix)
    combined = {}
    for col in columns:
        if col in skip:
            continue
        if col == "team_id":
            combined[col] = f"{col}__BOTH"  # placeholder
        else:
            combined[col] = f"{col}_low"  # default to low
    return combined


def _build_side_rename(columns: list[str], side: str) -> dict[str, str]:
    """Rename team features columns to _low or _high suffix."""
    skip = {"sex", "season"}
    mapping: dict[str, str] = {}
    for col in columns:
        if col in skip:
            continue
        if col == "team_id":
            mapping[col] = f"team_{side}"
        else:
            mapping[col] = f"{col}_{side}"
    return mapping


def _compute_diff_features(paired: pl.DataFrame) -> pl.DataFrame:
    """Compute all pairwise difference features from paired team data."""
    diffs = [
        (pl.col("win_rate_low") - pl.col("win_rate_high")).alias("diff_win_rate"),
        (pl.col("avg_margin_low") - pl.col("avg_margin_high")).alias("diff_avg_margin"),
        (pl.col("avg_pts_for_low") - pl.col("avg_pts_for_high")).alias("diff_avg_pts_for"),
        (pl.col("avg_pts_against_high") - pl.col("avg_pts_against_low")).alias("diff_defense_proxy"),
        (pl.col("last5_win_rate_low") - pl.col("last5_win_rate_high")).alias("diff_last5_win_rate"),
        (pl.col("last5_avg_margin_low") - pl.col("last5_avg_margin_high")).alias("diff_last5_avg_margin"),
        (pl.col("season_end_elo_low").fill_null(1500.0) - pl.col("season_end_elo_high").fill_null(1500.0)).alias("diff_elo"),
        (pl.col("pre_tourney_heat_1g_low").fill_null(0.0) - pl.col("pre_tourney_heat_1g_high").fill_null(0.0)).alias("diff_heat_1g"),
        (pl.col("pre_tourney_heat_3g_low").fill_null(0.0) - pl.col("pre_tourney_heat_3g_high").fill_null(0.0)).alias("diff_heat_3g"),
        (pl.col("pre_tourney_heat_5g_low").fill_null(0.0) - pl.col("pre_tourney_heat_5g_high").fill_null(0.0)).alias("diff_heat_5g"),
        (pl.col("seed_low").fill_null(8).cast(pl.Float64) - pl.col("seed_high").fill_null(8).cast(pl.Float64)).alias("diff_seed"),
    ]

    # Enhanced heat features
    if "heat_trend_low" in paired.columns:
        diffs.append(
            (pl.col("heat_trend_low").fill_null(0.0) - pl.col("heat_trend_high").fill_null(0.0)).alias("diff_heat_trend")
        )
        diffs.append(
            (pl.col("abs_heat_5g_low").fill_null(0.0) - pl.col("abs_heat_5g_high").fill_null(0.0)).alias("diff_abs_heat_5g")
        )

    # SOS
    if "sos_low" in paired.columns:
        diffs.append(
            (pl.col("sos_low").fill_null(1500.0) - pl.col("sos_high").fill_null(1500.0)).alias("diff_sos")
        )

    # Box score features
    if "fg_pct_low" in paired.columns:
        diffs.extend([
            (pl.col("fg_pct_low").fill_null(0.45) - pl.col("fg_pct_high").fill_null(0.45)).alias("diff_fg_pct"),
            (pl.col("fg3_pct_low").fill_null(0.33) - pl.col("fg3_pct_high").fill_null(0.33)).alias("diff_fg3_pct"),
            (pl.col("ft_pct_low").fill_null(0.70) - pl.col("ft_pct_high").fill_null(0.70)).alias("diff_ft_pct"),
            (pl.col("opp_fg_pct_low").fill_null(0.45) - pl.col("opp_fg_pct_high").fill_null(0.45)).alias("diff_opp_fg_pct"),
            (pl.col("avg_reb_margin_low").fill_null(0.0) - pl.col("avg_reb_margin_high").fill_null(0.0)).alias("diff_reb_margin"),
            (pl.col("ast_to_ratio_low").fill_null(1.0) - pl.col("ast_to_ratio_high").fill_null(1.0)).alias("diff_ast_to_ratio"),
            (pl.col("avg_stl_blk_low").fill_null(0.0) - pl.col("avg_stl_blk_high").fill_null(0.0)).alias("diff_stl_blk"),
            (pl.col("avg_possessions_low").fill_null(65.0) - pl.col("avg_possessions_high").fill_null(65.0)).alias("diff_possessions"),
        ])

    # Seed interaction features
    diffs.append(
        (pl.col("seed_low").fill_null(8).cast(pl.Float64) * pl.col("seed_high").fill_null(8).cast(pl.Float64)).alias("seed_product")
    )

    # Consensus spread (fill with 0 when missing)
    if "consensus_low_spread" in paired.columns:
        diffs.append(pl.col("consensus_low_spread").fill_null(0.0).alias("consensus_low_spread_filled"))

    return paired.with_columns(diffs)


def _pair_features_from_ids(
    ids: pl.DataFrame,
    team_features: pl.DataFrame,
    cbbd_line_consensus: pl.DataFrame,
    seed_map: pl.DataFrame | None = None,
) -> pl.DataFrame:
    parsed = ids.with_columns(
        pl.col("ID").str.split("_").list.get(0).cast(pl.Int64).alias("season"),
        pl.col("ID").str.split("_").list.get(1).cast(pl.Int64).alias("team_low"),
        pl.col("ID").str.split("_").list.get(2).cast(pl.Int64).alias("team_high"),
    ).with_columns(
        pl.col("team_low").map_elements(_infer_sex_from_team_id, return_dtype=pl.Utf8).alias("sex")
    )

    rename_map = _team_feature_rename_map(team_features.columns)
    low_features = team_features.rename(_build_side_rename(team_features.columns, "low"))
    high_features = team_features.rename(_build_side_rename(team_features.columns, "high"))

    paired = parsed.join(low_features, on=["sex", "season", "team_low"], how="left").join(
        high_features, on=["sex", "season", "team_high"], how="left"
    )
    max_season = int(parsed["season"].max()) if parsed.height else 0
    line_features = _line_features_for_pairs(ids, cbbd_line_consensus, max_season) if parsed.height else pl.DataFrame()
    paired = paired.join(
        line_features.select("ID", "consensus_low_spread", "consensus_over_under", "provider_count"),
        on="ID",
        how="left",
    )

    # Join seed features
    if seed_map is not None and seed_map.height > 0:
        seed_low = seed_map.rename({"team_id": "team_low", "seed_num": "seed_low"}).select("sex", "season", "team_low", "seed_low")
        seed_high = seed_map.rename({"team_id": "team_high", "seed_num": "seed_high"}).select("sex", "season", "team_high", "seed_high")
        paired = paired.join(seed_low, on=["sex", "season", "team_low"], how="left")
        paired = paired.join(seed_high, on=["sex", "season", "team_high"], how="left")
    else:
        paired = paired.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("seed_low"),
            pl.lit(None, dtype=pl.Int64).alias("seed_high"),
        )

    return _compute_diff_features(paired)


def _build_training_pairs(
    ingest_manifest: dict[str, Any],
    team_features: pl.DataFrame,
    target_season: int,
    seed_map: pl.DataFrame | None = None,
) -> pl.DataFrame:
    files = ingest_manifest["kaggle"]["files"]
    m_tourney = _read_parquet(Path(files["MNCAATourneyCompactResults"]["artifact"]))
    w_tourney = _read_parquet(Path(files["WNCAATourneyCompactResults"]["artifact"]))

    def _normalize_tourney(df: pl.DataFrame, sex: str) -> pl.DataFrame:
        return df.select(
            pl.lit(sex).alias("sex"),
            pl.col("Season").cast(pl.Int64).alias("season"),
            pl.min_horizontal(pl.col("WTeamID"), pl.col("LTeamID")).cast(pl.Int64).alias("team_low"),
            pl.max_horizontal(pl.col("WTeamID"), pl.col("LTeamID")).cast(pl.Int64).alias("team_high"),
            (pl.col("WTeamID") < pl.col("LTeamID")).cast(pl.Int64).alias("target_low_wins"),
        )

    labels = pl.concat([_normalize_tourney(m_tourney, "M"), _normalize_tourney(w_tourney, "W")], how="vertical")
    labels = labels.filter(pl.col("season") < target_season)

    low_features = team_features.rename(_build_side_rename(team_features.columns, "low"))
    high_features = team_features.rename(_build_side_rename(team_features.columns, "high"))

    paired = labels.join(low_features, on=["sex", "season", "team_low"], how="left").join(
        high_features, on=["sex", "season", "team_high"], how="left"
    )

    # Join seed features
    if seed_map is not None and seed_map.height > 0:
        seed_low = seed_map.rename({"team_id": "team_low", "seed_num": "seed_low"}).select("sex", "season", "team_low", "seed_low")
        seed_high = seed_map.rename({"team_id": "team_high", "seed_num": "seed_high"}).select("sex", "season", "team_high", "seed_high")
        paired = paired.join(seed_low, on=["sex", "season", "team_low"], how="left")
        paired = paired.join(seed_high, on=["sex", "season", "team_high"], how="left")
    else:
        paired = paired.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("seed_low"),
            pl.lit(None, dtype=pl.Int64).alias("seed_high"),
        )

    # Add consensus_low_spread placeholder filled with 0 (not available for training)
    paired = paired.with_columns(pl.lit(0.0, dtype=pl.Float64).alias("consensus_low_spread"))

    return _compute_diff_features(paired)


def _train_model(train_df: pl.DataFrame, feature_cols: list[str], n_estimators: int = 500,
                 max_depth: int = 5, learning_rate: float = 0.04,
                 subsample: float = 0.8, colsample_bytree: float = 0.7,
                 min_child_weight: int = 3, gamma: float = 0.1,
                 reg_alpha: float = 0.1, reg_lambda: float = 2.0) -> Any:
    if train_df.height == 0:
        return None
    cleaned = train_df.select(
        [pl.col(c).fill_null(0.0) for c in feature_cols] + [pl.col("target_low_wins")]
    ).drop_nulls()
    if cleaned.height == 0:
        return None
    y_values = cleaned.select("target_low_wins").to_series().to_list()
    if len(set(y_values)) < 2:
        return None
    x_values = cleaned.select(feature_cols).to_numpy()

    if XGBClassifier is not None:
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(x_values, y_values)
        return model

    if LogisticRegression is not None:
        model = LogisticRegression(max_iter=1000)
        model.fit(x_values, y_values)
        return model

    return None


def _time_series_cv_brier(training: pl.DataFrame, feature_cols: list[str],
                          sex: str = "M", cv_start: int = 2019,
                          cv_end: int = 2025, **model_kwargs: Any) -> tuple[float, list[float]]:
    """Time-series cross-validation returning average Brier and per-fold Brier scores."""
    fold_briers: list[float] = []
    sex_data = training.filter(pl.col("sex") == sex)

    for holdout_year in range(cv_start, cv_end + 1):
        train_fold = sex_data.filter(pl.col("season") < holdout_year)
        test_fold = sex_data.filter(pl.col("season") == holdout_year)
        if train_fold.height < 50 or test_fold.height == 0:
            continue

        model = _train_model(train_fold, feature_cols, **model_kwargs)
        if model is None:
            continue

        preds = _predict_with_model_raw(model, test_fold, feature_cols)
        actuals = test_fold.select("target_low_wins").to_series().to_numpy().astype(np.float64)
        brier = float(np.mean((preds - actuals) ** 2))
        fold_briers.append(brier)

    avg_brier = float(np.mean(fold_briers)) if fold_briers else 1.0
    return avg_brier, fold_briers


def _predict_with_model_raw(model: Any, features_df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Get raw model probabilities without clipping."""
    if features_df.height == 0:
        return np.array([], dtype=np.float64)
    if model is None:
        return np.full(features_df.height, 0.5, dtype=np.float64)
    filled = features_df.select([pl.col(c).fill_null(0.0) for c in feature_cols])
    return model.predict_proba(filled.to_numpy())[:, 1].astype(np.float64)


def _fit_calibration(model: Any, training: pl.DataFrame, feature_cols: list[str],
                     sex: str = "M", cv_start: int = 2019, cv_end: int = 2025) -> Any:
    """Fit isotonic calibration on out-of-fold predictions from time-series CV."""
    if IsotonicRegression is None:
        return None

    all_preds: list[float] = []
    all_actuals: list[float] = []
    sex_data = training.filter(pl.col("sex") == sex)

    for holdout_year in range(cv_start, cv_end + 1):
        train_fold = sex_data.filter(pl.col("season") < holdout_year)
        test_fold = sex_data.filter(pl.col("season") == holdout_year)
        if train_fold.height < 50 or test_fold.height == 0:
            continue

        fold_model = _train_model(train_fold, feature_cols)
        if fold_model is None:
            continue

        preds = _predict_with_model_raw(fold_model, test_fold, feature_cols)
        actuals = test_fold.select("target_low_wins").to_series().to_numpy().astype(np.float64)
        all_preds.extend(preds.tolist())
        all_actuals.extend(actuals.tolist())

    if len(all_preds) < 20:
        return None

    calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    calibrator.fit(np.array(all_preds), np.array(all_actuals))
    return calibrator


def _dynamic_clip(preds: np.ndarray, paired_df: pl.DataFrame) -> np.ndarray:
    """Apply seed-matchup-aware probability clipping.

    Extreme seed matchups (1v16, 2v15) get wider bounds when model is confident.
    """
    clipped = preds.copy()
    if "seed_low" not in paired_df.columns or "seed_high" not in paired_df.columns:
        return np.clip(clipped, 0.025, 0.975)

    seed_low = paired_df["seed_low"].fill_null(8).to_numpy()
    seed_high = paired_df["seed_high"].fill_null(8).to_numpy()
    seed_diff = np.abs(seed_low - seed_high)

    # Default bounds
    lo = np.full_like(clipped, 0.025)
    hi = np.full_like(clipped, 0.975)

    # Widen bounds for extreme seed matchups
    mask_1v16 = seed_diff >= 15  # 1v16
    mask_2v15 = (seed_diff >= 13) & (seed_diff < 15)  # 2v15
    mask_3v14 = (seed_diff >= 11) & (seed_diff < 13)  # 3v14

    lo[mask_1v16] = 0.005
    hi[mask_1v16] = 0.995
    lo[mask_2v15] = 0.01
    hi[mask_2v15] = 0.99
    lo[mask_3v14] = 0.015
    hi[mask_3v14] = 0.985

    return np.clip(clipped, lo, hi)


def _predict_with_model(model: Any, features_df: pl.DataFrame, feature_cols: list[str],
                        calibrator: Any = None) -> pl.Series:
    if features_df.height == 0:
        return pl.Series("pred", [], dtype=pl.Float64)

    if model is None:
        return pl.Series("pred", [0.5] * features_df.height, dtype=pl.Float64)

    preds = _predict_with_model_raw(model, features_df, feature_cols)

    # Apply isotonic calibration if available
    if calibrator is not None:
        preds = calibrator.predict(preds)

    # Apply dynamic clipping based on seed matchups
    clipped = _dynamic_clip(preds, features_df)
    return pl.Series("pred", clipped.tolist(), dtype=pl.Float64)


def _validation_split_metadata(config: PipelineConfig) -> dict[str, Any]:
    return {
        "train_start_season": config.cbbd_history_start,
        "train_end_season": config.holdout_season - 1,
        "holdout_season": config.holdout_season,
        "prediction_season": config.target_season,
    }


def run_gold_and_model(config: PipelineConfig, ingest_manifest: dict[str, Any]) -> dict[str, Any]:
    game_fact = _read_parquet(config.silver_dir / "game_fact.parquet")
    elo_ratings = _read_parquet(config.silver_dir / "elo_ratings.parquet")
    heat_scores = _read_parquet(config.silver_dir / "heat_scores.parquet")
    cbbd_line_consensus = _read_parquet(config.silver_dir / "cbbd_line_consensus.parquet")
    sample = _read_parquet(Path(ingest_manifest["kaggle"]["files"]["SampleSubmissionStage2"]["artifact"]))

    seed_map = _load_seed_map(ingest_manifest)
    team_features = _build_team_season_features(game_fact, elo_ratings, heat_scores)
    pairwise_features = _pair_features_from_ids(sample.select("ID"), team_features, cbbd_line_consensus, seed_map)
    training = _build_training_pairs(ingest_manifest, team_features, config.target_season, seed_map)

    split_meta = _validation_split_metadata(config)
    feature_cols = [
        "diff_win_rate",
        "diff_avg_margin",
        "diff_avg_pts_for",
        "diff_defense_proxy",
        "diff_last5_win_rate",
        "diff_last5_avg_margin",
        "diff_elo",
        "diff_heat_1g",
        "diff_heat_3g",
        "diff_heat_5g",
        "diff_heat_trend",
        "diff_abs_heat_5g",
        "diff_seed",
        "diff_sos",
        "diff_fg_pct",
        "diff_fg3_pct",
        "diff_ft_pct",
        "diff_opp_fg_pct",
        "diff_reb_margin",
        "diff_ast_to_ratio",
        "diff_stl_blk",
        "diff_possessions",
        "seed_product",
        "consensus_low_spread_filled",
    ]
    # Keep only feature columns that actually exist in both training and prediction data
    available_train_cols = set(training.columns) if training.height > 0 else set()
    available_pred_cols = set(pairwise_features.columns) if pairwise_features.height > 0 else set()
    feature_cols = [c for c in feature_cols if c in available_train_cols and c in available_pred_cols]

    predictions = []
    model_stats: dict[str, Any] = {}

    for sex in ["M", "W"]:
        train_sex = training.filter(pl.col("sex") == sex)
        pred_sex = pairwise_features.filter(pl.col("sex") == sex)

        if sex == "M":
            train_fit = train_sex.filter(pl.col("season") <= split_meta["train_end_season"])
            holdout = train_sex.filter(pl.col("season") == split_meta["holdout_season"])
            n_est = 500
        else:
            train_fit = train_sex
            holdout = pl.DataFrame(schema=train_sex.schema)
            n_est = 400

        model = _train_model(train_fit, feature_cols, n_estimators=n_est)

        # Fit isotonic calibration on time-series CV out-of-fold predictions
        calibrator = _fit_calibration(model, training, feature_cols, sex=sex)

        pred_values = _predict_with_model(model, pred_sex, feature_cols, calibrator=calibrator)
        pred_df = pred_sex.select("ID").with_columns(pred_values.alias("Pred"))
        predictions.append(pred_df)

        # Holdout Brier score
        holdout_brier = None
        cv_brier = None
        if holdout.height > 0 and model is not None:
            holdout_preds = _predict_with_model(model, holdout, feature_cols, calibrator=calibrator)
            actuals = holdout.select("target_low_wins").to_series().to_numpy().astype(np.float64)
            holdout_brier = float(np.mean((holdout_preds.to_numpy() - actuals) ** 2))

        # Time-series CV Brier
        if sex == "M":
            cv_brier_val, cv_folds = _time_series_cv_brier(training, feature_cols, sex=sex)
            cv_brier = {"mean": cv_brier_val, "folds": cv_folds}

        model_type = "flat_fallback"
        if model is not None:
            model_type = "xgboost" if XGBClassifier is not None and isinstance(model, XGBClassifier) else "logistic_regression"

        model_stats[sex] = {
            "train_rows": train_fit.height,
            "holdout_rows": holdout.height,
            "predict_rows": pred_sex.height,
            "model_type": model_type,
            "good_train_rows": train_fit.select(feature_cols).drop_nulls().height,
            "holdout_brier": holdout_brier,
            "cv_brier": cv_brier,
            "calibrated": calibrator is not None,
        }

    submission = pl.concat(predictions, how="vertical")
    submission = sample.select("ID").join(submission, on="ID", how="left").with_columns(
        pl.col("Pred").fill_null(0.5)
    )

    # Monte Carlo bracket simulation
    sim_meta: dict[str, Any] = {"simulated": False}
    files = ingest_manifest["kaggle"]["files"]
    prob_lookup = _build_prob_lookup(submission)

    for sex, seeds_key, slots_key in [("M", "MNCAATourneySeeds", "MNCAATourneySlots"), ("W", "WNCAATourneySeeds", "WNCAATourneySlots")]:
        if seeds_key not in files or slots_key not in files:
            continue
        all_seeds = _read_parquet(Path(files[seeds_key]["artifact"]))
        all_slots = _read_parquet(Path(files[slots_key]["artifact"]))
        target_seeds = all_seeds.filter(pl.col("Season") == config.target_season)
        target_slots = all_slots.filter(pl.col("Season") == config.target_season)
        if target_seeds.height == 0 or target_slots.height == 0:
            continue

        sim_preds = _simulate_bracket(target_seeds, target_slots, prob_lookup, n_sims=100_000)

        if sim_preds:
            sim_meta["simulated"] = True
            sim_meta[f"{sex}_matchups_simulated"] = len(sim_preds)

            blend_updates = [
                {"ID": f"{config.target_season}_{t_low}_{t_high}", "sim_pred": sim_prob}
                for (t_low, t_high), sim_prob in sim_preds.items()
            ]
            if blend_updates:
                sim_df = pl.DataFrame(blend_updates)
                submission = submission.join(sim_df, on="ID", how="left").with_columns(
                    pl.when(pl.col("sim_pred").is_not_null())
                    .then(pl.col("Pred") * 0.6 + pl.col("sim_pred") * 0.4)
                    .otherwise(pl.col("Pred"))
                    .alias("Pred")
                ).drop("sim_pred")

    # Write outputs
    game_features = game_fact.group_by(["sex", "season", "game_key"]).agg(
        pl.mean("team_score").alias("avg_team_points"),
        pl.mean("opp_score").alias("avg_opp_points"),
        pl.sum("win").alias("sum_win_rows"),
    )

    _write_parquet(team_features, config.gold_dir / "team_season_features.parquet")
    _write_parquet(game_features, config.gold_dir / "game_features.parquet")
    _write_parquet(pairwise_features, config.gold_dir / "pairwise_features.parquet")
    submission.write_csv(config.run_dir / "submission.csv")

    quality = pairwise_features.with_columns(
        pl.all_horizontal([pl.col(c).is_not_null() for c in feature_cols]).alias("feature_row_quality_pass")
    )
    quality_summary = {
        "total_rows": quality.height,
        "good_rows": int(quality.filter(pl.col("feature_row_quality_pass")).height),
        "bad_rows": int(quality.filter(~pl.col("feature_row_quality_pass")).height),
    }

    perf_summary = {
        "model_stats": model_stats,
        "simulation": sim_meta,
        "feature_cols": feature_cols,
    }
    (config.reports_dir / "gold_quality_summary.json").write_text(json.dumps(quality_summary, indent=2, default=_json_default), encoding="utf-8")
    (config.reports_dir / "validation_split_summary.json").write_text(json.dumps(split_meta, indent=2, default=_json_default), encoding="utf-8")
    (config.reports_dir / "model_performance_summary.json").write_text(json.dumps(perf_summary, indent=2, default=_json_default), encoding="utf-8")

    return {
        "team_season_rows": team_features.height,
        "game_feature_rows": game_features.height,
        "pairwise_rows": pairwise_features.height,
        "submission_rows": submission.height,
        "model_stats": model_stats,
        "quality_summary": quality_summary,
        "validation_split": split_meta,
        "simulation": sim_meta,
    }


def _write_run_manifest(config: PipelineConfig, payload: dict[str, Any]) -> Path:
    manifest_path = config.run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

    latest_dir = config.artifacts_dir / "latest"
    if latest_dir.exists() or latest_dir.is_symlink():
        if latest_dir.is_symlink() or latest_dir.is_file():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)
    try:
        latest_dir.symlink_to(config.run_dir, target_is_directory=True)
    except OSError:
        shutil.copytree(config.run_dir, latest_dir)
    return manifest_path


def run_pipeline(project_root: Path, mode: str = "manual", target_season: int = 2026) -> dict[str, Any]:
    config = PipelineConfig(project_root=project_root, mode=mode, target_season=target_season)
    for path in [config.bronze_dir, config.silver_dir, config.gold_dir, config.reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(UTC).isoformat()
    ingest_manifest = run_ingest(config)
    validation = run_validations(config, ingest_manifest)
    transform = run_transform(config, ingest_manifest)
    elo_heat = run_elo_and_heat(config)
    gold = run_gold_and_model(config, ingest_manifest)
    finished_at = datetime.now(UTC).isoformat()

    payload = {
        "run_id": config.run_id,
        "mode": mode,
        "target_season": target_season,
        "started_at": started_at,
        "finished_at": finished_at,
        "artifacts": {
            "run_dir": str(config.run_dir),
            "submission": str(config.run_dir / "submission.csv"),
        },
        "ingest": ingest_manifest,
        "validation": validation,
        "transform": transform,
        "elo_heat": elo_heat,
        "gold": gold,
    }
    _write_run_manifest(config, payload)
    return payload


def run_model_only(project_root: Path, target_season: int = 2026) -> dict[str, Any]:
    """Reuse bronze/silver from artifacts/latest, recompute ELO + heat + gold/model."""
    latest_dir = project_root / "artifacts" / "latest"
    manifest_path = latest_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No previous run found at {manifest_path}. Run full pipeline first.")

    prev_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ingest_manifest = prev_manifest["ingest"]

    config = PipelineConfig(project_root=project_root, mode="manual", target_season=target_season)
    for path in [config.bronze_dir, config.silver_dir, config.gold_dir, config.reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    # Copy bronze and silver from latest run
    prev_run_dir = Path(prev_manifest["artifacts"]["run_dir"])
    for layer in ["bronze", "silver"]:
        src = prev_run_dir / layer
        dst = config.run_dir / layer
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    # Rewrite ingest artifact paths to point at this run's copies
    for dataset_info in ingest_manifest.get("kaggle", {}).get("files", {}).values():
        old_artifact = dataset_info.get("artifact", "")
        if old_artifact:
            dataset_info["artifact"] = old_artifact.replace(str(prev_run_dir), str(config.run_dir))

    started_at = datetime.now(UTC).isoformat()
    elo_heat = run_elo_and_heat(config)
    gold = run_gold_and_model(config, ingest_manifest)
    finished_at = datetime.now(UTC).isoformat()

    payload = {
        "run_id": config.run_id,
        "mode": "model_only",
        "target_season": target_season,
        "started_at": started_at,
        "finished_at": finished_at,
        "artifacts": {
            "run_dir": str(config.run_dir),
            "submission": str(config.run_dir / "submission.csv"),
        },
        "ingest": ingest_manifest,
        "elo_heat": elo_heat,
        "gold": gold,
    }
    _write_run_manifest(config, payload)
    return payload


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="MM26 Stage 1 pipeline orchestrator")
    parser.add_argument("command", choices=["run"], nargs="?", default="run")
    parser.add_argument("--mode", choices=["manual", "daily"], default="manual")
    parser.add_argument("--target-season", type=int, default=2026)
    parser.add_argument("--stage", choices=["full", "model"], default="full",
                        help="'model' reuses existing bronze/silver from latest run")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    if args.stage == "model":
        result = run_model_only(project_root=project_root, target_season=args.target_season)
    else:
        result = run_pipeline(project_root=project_root, mode=args.mode, target_season=args.target_season)
    print(json.dumps({"run_id": result["run_id"], "submission": result["artifacts"]["submission"]}, indent=2))


if __name__ == "__main__":
    cli_main()
