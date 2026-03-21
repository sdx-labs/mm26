"""Silver-layer transformations: expand games, team ID mapping, consensus lines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from .config import (
    PipelineConfig,
    _invert_wloc_expr,
    _json_default,
    _read_parquet,
    _write_parquet,
    normalize_name,
)


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
