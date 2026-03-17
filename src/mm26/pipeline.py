from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
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

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data" / "march-machine-learning-mania-2026"
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
        "MRegularSeasonDetailedResults": ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"],
        "WRegularSeasonDetailedResults": ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"],
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


def ingest_cbbd_lines(config: PipelineConfig) -> dict[str, Any]:
    cbbd_dir = config.bronze_dir / "cbbd"
    cbbd_dir.mkdir(parents=True, exist_ok=True)
    output_path = cbbd_dir / "lines.parquet"

    status = {
        "status": "skipped",
        "reason": None,
        "rows": 0,
        "artifact": str(output_path),
    }

    api_key = os.getenv("CBBD_API_KEY")
    if not api_key:
        df = pl.DataFrame(
            {
                "season": pl.Series([], dtype=pl.Int64),
                "start_date": pl.Series([], dtype=pl.Utf8),
                "home_team": pl.Series([], dtype=pl.Utf8),
                "away_team": pl.Series([], dtype=pl.Utf8),
                "home_score": pl.Series([], dtype=pl.Int64),
                "away_score": pl.Series([], dtype=pl.Int64),
                "spread": pl.Series([], dtype=pl.Float64),
                "over_under": pl.Series([], dtype=pl.Float64),
                "provider": pl.Series([], dtype=pl.Utf8),
            }
        )
        _write_parquet(df, output_path)
        status["reason"] = "CBBD_API_KEY missing"
        return status

    try:
        import cbbd

        configuration = cbbd.Configuration(host="https://api.collegebasketballdata.com")
        configuration.api_key["Authorization"] = api_key
        rows: list[dict[str, Any]] = []

        with cbbd.ApiClient(configuration) as api_client:
            api_instance = cbbd.LinesApi(api_client)
            response = api_instance.get_lines(season=config.target_season)

            for game in response:
                payload = game.to_dict() if hasattr(game, "to_dict") else dict(game)
                lines = payload.get("lines") or []
                if not lines:
                    rows.append(
                        {
                            "season": payload.get("season"),
                            "start_date": str(payload.get("start_date")),
                            "home_team": payload.get("home_team"),
                            "away_team": payload.get("away_team"),
                            "home_score": payload.get("home_score"),
                            "away_score": payload.get("away_score"),
                            "spread": None,
                            "over_under": None,
                            "provider": None,
                        }
                    )
                    continue

                for line in lines:
                    line_info = line.to_dict() if hasattr(line, "to_dict") else dict(line)
                    rows.append(
                        {
                            "season": payload.get("season"),
                            "start_date": str(payload.get("start_date")),
                            "home_team": payload.get("home_team"),
                            "away_team": payload.get("away_team"),
                            "home_score": payload.get("home_score"),
                            "away_score": payload.get("away_score"),
                            "spread": line_info.get("spread"),
                            "over_under": line_info.get("over_under"),
                            "provider": line_info.get("provider"),
                        }
                    )

        df = pl.DataFrame(rows) if rows else pl.DataFrame(
            {
                "season": pl.Series([], dtype=pl.Int64),
                "start_date": pl.Series([], dtype=pl.Utf8),
                "home_team": pl.Series([], dtype=pl.Utf8),
                "away_team": pl.Series([], dtype=pl.Utf8),
                "home_score": pl.Series([], dtype=pl.Int64),
                "away_score": pl.Series([], dtype=pl.Int64),
                "spread": pl.Series([], dtype=pl.Float64),
                "over_under": pl.Series([], dtype=pl.Float64),
                "provider": pl.Series([], dtype=pl.Utf8),
            }
        )

        _write_parquet(df, output_path)
        status["status"] = "ok"
        status["rows"] = df.height
    except Exception as exc:  # pragma: no cover
        df = pl.DataFrame(
            {
                "season": pl.Series([], dtype=pl.Int64),
                "start_date": pl.Series([], dtype=pl.Utf8),
                "home_team": pl.Series([], dtype=pl.Utf8),
                "away_team": pl.Series([], dtype=pl.Utf8),
                "home_score": pl.Series([], dtype=pl.Int64),
                "away_score": pl.Series([], dtype=pl.Int64),
                "spread": pl.Series([], dtype=pl.Float64),
                "over_under": pl.Series([], dtype=pl.Float64),
                "provider": pl.Series([], dtype=pl.Utf8),
            }
        )
        _write_parquet(df, output_path)
        status["status"] = "error"
        status["reason"] = str(exc)

    return status


def run_ingest(config: PipelineConfig) -> dict[str, Any]:
    kaggle_manifest = ingest_kaggle(config)
    cbbd_manifest = ingest_cbbd_lines(config)
    return {"kaggle": kaggle_manifest, "cbbd": cbbd_manifest}


def run_validations(config: PipelineConfig, ingest_manifest: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {"checks": [], "failed": False}
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
        check_result = {
            "dataset": dataset_name,
            "ok": ok,
            "missing_columns": missing,
            "rows": entry["rows"],
        }
        results["checks"].append(check_result)
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

    report_path = config.reports_dir / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

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

    cbbd_names = pl.concat(
        [
            cbbd_lines.select(pl.col("home_team").cast(pl.Utf8).alias("cbbd_team_name")),
            cbbd_lines.select(pl.col("away_team").cast(pl.Utf8).alias("cbbd_team_name")),
        ],
        how="vertical",
    ).drop_nulls().unique()

    if cbbd_names.height == 0:
        mapping = pl.DataFrame(
            {
                "cbbd_team_name": pl.Series([], dtype=pl.Utf8),
                "normalized_name": pl.Series([], dtype=pl.Utf8),
                "team_id": pl.Series([], dtype=pl.Int64),
                "mapped": pl.Series([], dtype=pl.Boolean),
            }
        )
        summary = {"total": 0, "mapped": 0, "unmapped": 0, "mapped_pct": 0.0}
        return mapping, summary

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

    team_games = pl.concat(
        [
            _expand_team_games(m_reg_detail, "M"),
            _expand_team_games(w_reg_detail, "W"),
        ],
        how="vertical",
    ).with_columns(
        pl.min_horizontal(pl.col("team_id"), pl.col("opp_team_id")).alias("team_low"),
        pl.max_horizontal(pl.col("team_id"), pl.col("opp_team_id")).alias("team_high"),
    ).with_columns(
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

    _write_parquet(team_dim, config.silver_dir / "team_dim.parquet")
    _write_parquet(team_games, config.silver_dir / "game_fact.parquet")
    _write_parquet(team_id_map, config.silver_dir / "team_id_map.parquet")
    _write_parquet(quality_flags, config.silver_dir / "quality_flags.parquet")

    team_id_map.write_csv(config.reports_dir / "team_id_map_report.csv")
    (config.reports_dir / "team_id_map_summary.json").write_text(json.dumps(map_summary, indent=2), encoding="utf-8")

    return {
        "team_dim_rows": team_dim.height,
        "game_fact_rows": team_games.height,
        "team_id_map_rows": team_id_map.height,
        "team_id_map_summary": map_summary,
    }


def _build_team_season_features(team_games: pl.DataFrame) -> pl.DataFrame:
    return team_games.group_by(["sex", "season", "team_id"]).agg(
        pl.count().alias("games_played"),
        pl.sum("win").alias("wins"),
        (pl.count() - pl.sum("win")).alias("losses"),
        pl.mean("team_score").alias("avg_pts_for"),
        pl.mean("opp_score").alias("avg_pts_against"),
        (pl.col("team_score") - pl.col("opp_score")).mean().alias("avg_margin"),
        pl.mean("num_ot").alias("avg_num_ot"),
    ).with_columns((pl.col("wins") / pl.col("games_played")).alias("win_rate"))


def _pair_features_from_ids(ids: pl.DataFrame, team_features: pl.DataFrame) -> pl.DataFrame:
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
        }
    )

    paired = parsed.join(low_features, on=["sex", "season", "team_low"], how="left").join(
        high_features, on=["sex", "season", "team_high"], how="left"
    )

    return paired.with_columns(
        (pl.col("win_rate_low") - pl.col("win_rate_high")).alias("diff_win_rate"),
        (pl.col("avg_margin_low") - pl.col("avg_margin_high")).alias("diff_avg_margin"),
        (pl.col("avg_pts_for_low") - pl.col("avg_pts_for_high")).alias("diff_avg_pts_for"),
        (pl.col("avg_pts_against_high") - pl.col("avg_pts_against_low")).alias("diff_defense_proxy"),
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
        }
    )

    dataset = labels.join(low_features, on=["sex", "season", "team_low"], how="left").join(
        high_features, on=["sex", "season", "team_high"], how="left"
    ).with_columns(
        (pl.col("win_rate_low") - pl.col("win_rate_high")).alias("diff_win_rate"),
        (pl.col("avg_margin_low") - pl.col("avg_margin_high")).alias("diff_avg_margin"),
        (pl.col("avg_pts_for_low") - pl.col("avg_pts_for_high")).alias("diff_avg_pts_for"),
        (pl.col("avg_pts_against_high") - pl.col("avg_pts_against_low")).alias("diff_defense_proxy"),
    )

    return dataset


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
    )

    if model is None:
        return features_df.select(score_expr.clip(0.025, 0.975).alias("pred"))["pred"]

    selected = features_df.select(feature_cols)
    row_ok_expr = pl.all_horizontal([pl.col(c).is_not_null() for c in feature_cols]).alias("row_ok")
    marked = selected.with_columns(row_ok_expr)
    preds = [0.5] * marked.height

    good = marked.filter(pl.col("row_ok")).drop("row_ok")
    if good.height > 0:
        good_idx = (
            marked.with_row_index("idx")
            .filter(pl.col("row_ok"))
            .select("idx")
            .to_series()
            .to_list()
        )
        proba = model.predict_proba(good.to_numpy())[:, 1]
        for idx, value in zip(good_idx, proba, strict=False):
            preds[int(idx)] = float(value)

    clipped = [max(0.025, min(0.975, p)) for p in preds]
    return pl.Series("pred", clipped, dtype=pl.Float64)


def run_gold_and_model(config: PipelineConfig, ingest_manifest: dict[str, Any]) -> dict[str, Any]:
    game_fact = _read_parquet(config.silver_dir / "game_fact.parquet")
    sample = _read_parquet(Path(ingest_manifest["kaggle"]["files"]["SampleSubmissionStage2"]["artifact"]))

    team_features = _build_team_season_features(game_fact)
    pairwise_features = _pair_features_from_ids(sample.select("ID"), team_features)

    training = _build_training_pairs(ingest_manifest, team_features, config.target_season)

    feature_cols = ["diff_win_rate", "diff_avg_margin", "diff_avg_pts_for", "diff_defense_proxy"]

    predictions = []
    model_stats: dict[str, Any] = {}

    for sex in ["M", "W"]:
        train_sex = training.filter(pl.col("sex") == sex)
        pred_sex = pairwise_features.filter(pl.col("sex") == sex)

        model = _train_model(train_sex, feature_cols)
        pred_values = _predict_with_model(model, pred_sex, feature_cols)

        pred_df = pred_sex.select("ID").with_columns(pred_values.alias("Pred"))
        predictions.append(pred_df)

        model_stats[sex] = {
            "train_rows": train_sex.height,
            "predict_rows": pred_sex.height,
            "model_type": "logistic_regression" if model is not None else "heuristic_fallback",
            "good_train_rows": train_sex.select(feature_cols).drop_nulls().height,
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
        pl.all_horizontal(
            [pl.col(c).is_not_null() for c in feature_cols]
        ).alias("feature_row_quality_pass")
    )
    quality_summary = {
        "total_rows": quality.height,
        "good_rows": int(quality.filter(pl.col("feature_row_quality_pass")).height),
        "bad_rows": int(quality.filter(~pl.col("feature_row_quality_pass")).height),
    }
    (config.reports_dir / "gold_quality_summary.json").write_text(json.dumps(quality_summary, indent=2), encoding="utf-8")

    return {
        "team_season_rows": team_features.height,
        "game_feature_rows": game_features.height,
        "pairwise_rows": pairwise_features.height,
        "submission_rows": submission.height,
        "model_stats": model_stats,
        "quality_summary": quality_summary,
    }


def _write_run_manifest(config: PipelineConfig, payload: dict[str, Any]) -> Path:
    manifest_path = config.run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    latest_dir = config.artifacts_dir / "latest"
    if latest_dir.exists() or latest_dir.is_symlink():
        if latest_dir.is_symlink() or latest_dir.is_file():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)
    latest_dir.symlink_to(config.run_dir, target_is_directory=True)
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
