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

import polars as pl

try:
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover
    LogisticRegression = None


@dataclass
class PipelineConfig:
    project_root: Path
    data_dir: Path = field(init=False)
    source_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    run_id: str = field(init=False)
    target_season: int = 2026
    mode: str = "manual"
    cbbd_history_start: int = 2016
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
    win_side = detailed.select(
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
    )
    loss_side = detailed.select(
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
    )
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


def _build_team_season_features(team_games: pl.DataFrame) -> pl.DataFrame:
    ordered = team_games.sort(["sex", "season", "team_id", "day_num"])
    return ordered.group_by(["sex", "season", "team_id"]).agg(
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
    ).with_columns((pl.col("wins") / pl.col("games_played")).alias("win_rate"))


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


def _pair_features_from_ids(ids: pl.DataFrame, team_features: pl.DataFrame, cbbd_line_consensus: pl.DataFrame) -> pl.DataFrame:
    parsed = ids.with_columns(
        pl.col("ID").str.split("_").list.get(0).cast(pl.Int64).alias("season"),
        pl.col("ID").str.split("_").list.get(1).cast(pl.Int64).alias("team_low"),
        pl.col("ID").str.split("_").list.get(2).cast(pl.Int64).alias("team_high"),
    ).with_columns(
        pl.col("team_low").map_elements(_infer_sex_from_team_id, return_dtype=pl.Utf8).alias("sex")
    )

    low_features = team_features.rename(
        {
            "team_id": "team_low",
            "games_played": "games_low",
            "wins": "wins_low",
            "losses": "losses_low",
            "avg_pts_for": "avg_pts_for_low",
            "avg_pts_against": "avg_pts_against_low",
            "avg_margin": "avg_margin_low",
            "avg_num_ot": "avg_num_ot_low",
            "win_rate": "win_rate_low",
            "last5_avg_pts_for": "last5_avg_pts_for_low",
            "last5_avg_pts_against": "last5_avg_pts_against_low",
            "last5_avg_margin": "last5_avg_margin_low",
            "last5_win_rate": "last5_win_rate_low",
        }
    )
    high_features = team_features.rename(
        {
            "team_id": "team_high",
            "games_played": "games_high",
            "wins": "wins_high",
            "losses": "losses_high",
            "avg_pts_for": "avg_pts_for_high",
            "avg_pts_against": "avg_pts_against_high",
            "avg_margin": "avg_margin_high",
            "avg_num_ot": "avg_num_ot_high",
            "win_rate": "win_rate_high",
            "last5_avg_pts_for": "last5_avg_pts_for_high",
            "last5_avg_pts_against": "last5_avg_pts_against_high",
            "last5_avg_margin": "last5_avg_margin_high",
            "last5_win_rate": "last5_win_rate_high",
        }
    )

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

    return paired.with_columns(
        (pl.col("win_rate_low") - pl.col("win_rate_high")).alias("diff_win_rate"),
        (pl.col("avg_margin_low") - pl.col("avg_margin_high")).alias("diff_avg_margin"),
        (pl.col("avg_pts_for_low") - pl.col("avg_pts_for_high")).alias("diff_avg_pts_for"),
        (pl.col("avg_pts_against_high") - pl.col("avg_pts_against_low")).alias("diff_defense_proxy"),
        (pl.col("last5_win_rate_low") - pl.col("last5_win_rate_high")).alias("diff_last5_win_rate"),
        (pl.col("last5_avg_margin_low") - pl.col("last5_avg_margin_high")).alias("diff_last5_avg_margin"),
    )


def _build_training_pairs(ingest_manifest: dict[str, Any], team_features: pl.DataFrame, target_season: int) -> pl.DataFrame:
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

    low_features = team_features.rename(
        {
            "team_id": "team_low",
            "games_played": "games_low",
            "wins": "wins_low",
            "losses": "losses_low",
            "avg_pts_for": "avg_pts_for_low",
            "avg_pts_against": "avg_pts_against_low",
            "avg_margin": "avg_margin_low",
            "avg_num_ot": "avg_num_ot_low",
            "win_rate": "win_rate_low",
            "last5_avg_pts_for": "last5_avg_pts_for_low",
            "last5_avg_pts_against": "last5_avg_pts_against_low",
            "last5_avg_margin": "last5_avg_margin_low",
            "last5_win_rate": "last5_win_rate_low",
        }
    )
    high_features = team_features.rename(
        {
            "team_id": "team_high",
            "games_played": "games_high",
            "wins": "wins_high",
            "losses": "losses_high",
            "avg_pts_for": "avg_pts_for_high",
            "avg_pts_against": "avg_pts_against_high",
            "avg_margin": "avg_margin_high",
            "avg_num_ot": "avg_num_ot_high",
            "win_rate": "win_rate_high",
            "last5_avg_pts_for": "last5_avg_pts_for_high",
            "last5_avg_pts_against": "last5_avg_pts_against_high",
            "last5_avg_margin": "last5_avg_margin_high",
            "last5_win_rate": "last5_win_rate_high",
        }
    )

    return labels.join(low_features, on=["sex", "season", "team_low"], how="left").join(
        high_features, on=["sex", "season", "team_high"], how="left"
    ).with_columns(
        (pl.col("win_rate_low") - pl.col("win_rate_high")).alias("diff_win_rate"),
        (pl.col("avg_margin_low") - pl.col("avg_margin_high")).alias("diff_avg_margin"),
        (pl.col("avg_pts_for_low") - pl.col("avg_pts_for_high")).alias("diff_avg_pts_for"),
        (pl.col("avg_pts_against_high") - pl.col("avg_pts_against_low")).alias("diff_defense_proxy"),
        (pl.col("last5_win_rate_low") - pl.col("last5_win_rate_high")).alias("diff_last5_win_rate"),
        (pl.col("last5_avg_margin_low") - pl.col("last5_avg_margin_high")).alias("diff_last5_avg_margin"),
    )


def _train_model(train_df: pl.DataFrame, feature_cols: list[str]) -> Any:
    if train_df.height == 0 or LogisticRegression is None:
        return None
    cleaned = train_df.select(feature_cols + ["target_low_wins"]).drop_nulls()
    if cleaned.height == 0:
        return None
    y_values = cleaned.select("target_low_wins").to_series().to_list()
    if len(set(y_values)) < 2:
        return None
    x_values = cleaned.select(feature_cols).to_numpy()

    model = LogisticRegression(max_iter=1000)
    model.fit(x_values, y_values)
    return model


def _predict_with_model(model: Any, features_df: pl.DataFrame, feature_cols: list[str]) -> pl.Series:
    if features_df.height == 0:
        return pl.Series("pred", [], dtype=pl.Float64)

    score_expr = (
        0.5
        + 0.35 * pl.col("diff_win_rate").fill_null(0.0)
        + 0.03 * pl.col("diff_avg_margin").fill_null(0.0)
        + 0.015 * pl.col("diff_last5_avg_margin").fill_null(0.0)
        - 0.02 * pl.col("consensus_low_spread").fill_null(0.0)
    )

    if model is None:
        return features_df.select(score_expr.clip(0.025, 0.975).alias("pred"))["pred"]

    selected = features_df.select(feature_cols)
    row_ok_expr = pl.all_horizontal([pl.col(c).is_not_null() for c in feature_cols]).alias("row_ok")
    marked = selected.with_columns(row_ok_expr)
    preds = [0.5] * marked.height

    good = marked.filter(pl.col("row_ok")).drop("row_ok")
    if good.height > 0:
        good_idx = marked.with_row_index("idx").filter(pl.col("row_ok")).select("idx").to_series().to_list()
        proba = model.predict_proba(good.to_numpy())[:, 1]
        for idx, value in zip(good_idx, proba, strict=False):
            preds[int(idx)] = float(value)

    clipped = [max(0.025, min(0.975, p)) for p in preds]
    return pl.Series("pred", clipped, dtype=pl.Float64)


def _validation_split_metadata(config: PipelineConfig) -> dict[str, Any]:
    return {
        "train_start_season": config.cbbd_history_start,
        "train_end_season": config.holdout_season - 1,
        "holdout_season": config.holdout_season,
        "prediction_season": config.target_season,
    }


def run_gold_and_model(config: PipelineConfig, ingest_manifest: dict[str, Any]) -> dict[str, Any]:
    game_fact = _read_parquet(config.silver_dir / "game_fact.parquet")
    cbbd_line_consensus = _read_parquet(config.silver_dir / "cbbd_line_consensus.parquet")
    sample = _read_parquet(Path(ingest_manifest["kaggle"]["files"]["SampleSubmissionStage2"]["artifact"]))

    team_features = _build_team_season_features(game_fact)
    pairwise_features = _pair_features_from_ids(sample.select("ID"), team_features, cbbd_line_consensus)
    training = _build_training_pairs(ingest_manifest, team_features, config.target_season)

    split_meta = _validation_split_metadata(config)
    feature_cols = [
        "diff_win_rate",
        "diff_avg_margin",
        "diff_avg_pts_for",
        "diff_defense_proxy",
        "diff_last5_win_rate",
        "diff_last5_avg_margin",
    ]

    predictions = []
    model_stats: dict[str, Any] = {}

    for sex in ["M", "W"]:
        train_sex = training.filter(pl.col("sex") == sex)
        pred_sex = pairwise_features.filter(pl.col("sex") == sex)

        if sex == "M":
            train_fit = train_sex.filter(pl.col("season") <= split_meta["train_end_season"])
            holdout = train_sex.filter(pl.col("season") == split_meta["holdout_season"])
        else:
            train_fit = train_sex
            holdout = pl.DataFrame(schema=train_sex.schema)

        model = _train_model(train_fit, feature_cols)
        pred_values = _predict_with_model(model, pred_sex, feature_cols)
        pred_df = pred_sex.select("ID").with_columns(pred_values.alias("Pred"))
        predictions.append(pred_df)

        model_stats[sex] = {
            "train_rows": train_fit.height,
            "holdout_rows": holdout.height,
            "predict_rows": pred_sex.height,
            "model_type": "logistic_regression" if model is not None else "heuristic_fallback",
            "good_train_rows": train_fit.select(feature_cols).drop_nulls().height,
        }

    submission = pl.concat(predictions, how="vertical")
    submission = sample.select("ID").join(submission, on="ID", how="left").with_columns(
        pl.col("Pred").fill_null(0.5).clip(0.025, 0.975)
    )

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
    (config.reports_dir / "gold_quality_summary.json").write_text(json.dumps(quality_summary, indent=2, default=_json_default), encoding="utf-8")
    (config.reports_dir / "validation_split_summary.json").write_text(json.dumps(split_meta, indent=2, default=_json_default), encoding="utf-8")

    return {
        "team_season_rows": team_features.height,
        "game_feature_rows": game_features.height,
        "pairwise_rows": pairwise_features.height,
        "submission_rows": submission.height,
        "model_stats": model_stats,
        "quality_summary": quality_summary,
        "validation_split": split_meta,
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
        "gold": gold,
    }
    _write_run_manifest(config, payload)
    return payload


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="MM26 Stage 1 pipeline orchestrator")
    parser.add_argument("command", choices=["run"], nargs="?", default="run")
    parser.add_argument("--mode", choices=["manual", "daily"], default="manual")
    parser.add_argument("--target-season", type=int, default=2026)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    result = run_pipeline(project_root=project_root, mode=args.mode, target_season=args.target_season)
    print(json.dumps({"run_id": result["run_id"], "submission": result["artifacts"]["submission"]}, indent=2))


if __name__ == "__main__":
    cli_main()
