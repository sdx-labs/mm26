"""Kaggle CSV and CBBD API data ingestion (bronze layer)."""

from __future__ import annotations

import os
import re
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from .config import PipelineConfig, _read_parquet, _required_kaggle_schemas, _write_parquet


# ---------------------------------------------------------------------------
# Empty CBBD schema helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CBBD configuration & env helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# API payload normalization helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Top-level ingestion functions
# ---------------------------------------------------------------------------

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
