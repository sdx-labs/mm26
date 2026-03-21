"""Team-season feature engineering, pairwise diffs, and training pair construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from .config import _infer_sex_from_team_id, _read_parquet
from .ratings import _get_pre_tournament_heat


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
            (pl.sum("fgm") / pl.sum("fga")).alias("fg_pct"),
            (pl.sum("fgm3") / pl.sum("fga3")).alias("fg3_pct"),
            (pl.sum("ftm") / pl.sum("fta")).alias("ft_pct"),
            (pl.sum("opp_fgm") / pl.sum("opp_fga")).alias("opp_fg_pct"),
            ((pl.sum("oreb") + pl.sum("dreb")) - (pl.sum("opp_oreb") + pl.sum("opp_dreb"))).cast(pl.Float64).alias("total_reb_margin"),
            pl.mean("oreb").alias("avg_oreb"),
            pl.mean("dreb").alias("avg_dreb"),
            pl.mean("ast").alias("avg_ast"),
            pl.mean("tov").alias("avg_tov"),
            pl.mean("stl").alias("avg_stl"),
            pl.mean("blk").alias("avg_blk"),
            pl.mean("opp_tov").alias("avg_opp_tov"),
            (pl.col("fga") - pl.col("oreb") + pl.col("tov") + pl.col("fta") * 0.475).mean().alias("avg_possessions"),
            (pl.col("fgm").tail(5).sum() / pl.col("fga").tail(5).sum()).alias("last5_fg_pct"),
        ]
        base_aggs.extend(detail_aggs)

    features = ordered.group_by(["sex", "season", "team_id"]).agg(
        base_aggs
    ).with_columns((pl.col("wins") / pl.col("games_played")).alias("win_rate"))

    if has_detail:
        features = features.with_columns(
            ((pl.col("total_reb_margin")) / pl.col("games_played")).alias("avg_reb_margin"),
            (pl.col("avg_ast") / pl.col("avg_tov").clip(0.1, None)).alias("ast_to_ratio"),
            (pl.col("avg_stl") + pl.col("avg_blk")).alias("avg_stl_blk"),
            (100.0 * pl.col("avg_pts_for") / pl.col("avg_possessions").clip(40.0, None)).alias("off_rating"),
            (100.0 * pl.col("avg_pts_against") / pl.col("avg_possessions").clip(40.0, None)).alias("def_rating"),
        ).with_columns(
            (pl.col("off_rating") - pl.col("def_rating")).alias("net_rating"),
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
            pl.lit(None, dtype=pl.Float64).alias("off_rating"),
            pl.lit(None, dtype=pl.Float64).alias("def_rating"),
            pl.lit(None, dtype=pl.Float64).alias("net_rating"),
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

        heat_cols_to_join = ["sex", "season", "team_id", "heat_1g", "heat_3g", "heat_5g"]
        heat_rename = {
            "heat_1g": "pre_tourney_heat_1g",
            "heat_3g": "pre_tourney_heat_3g",
            "heat_5g": "pre_tourney_heat_5g",
        }

        if "sw_heat_5g" in pre_heat.columns:
            heat_cols_to_join.extend(["sw_heat_1g", "sw_heat_3g", "sw_heat_5g", "heat_volatility_5g"])
            heat_rename.update({
                "sw_heat_1g": "pre_tourney_sw_heat_1g",
                "sw_heat_3g": "pre_tourney_sw_heat_3g",
                "sw_heat_5g": "pre_tourney_sw_heat_5g",
                "heat_volatility_5g": "pre_tourney_heat_volatility_5g",
            })

        features = features.join(
            pre_heat.select(heat_cols_to_join).rename(heat_rename),
            on=["sex", "season", "team_id"],
            how="left",
        )
        features = features.with_columns(
            (pl.col("pre_tourney_heat_5g") - pl.col("pre_tourney_heat_1g")).alias("heat_trend"),
            pl.col("pre_tourney_heat_5g").abs().alias("abs_heat_5g"),
        )
    else:
        features = features.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("pre_tourney_heat_1g"),
            pl.lit(None, dtype=pl.Float64).alias("pre_tourney_heat_3g"),
            pl.lit(None, dtype=pl.Float64).alias("pre_tourney_heat_5g"),
            pl.lit(None, dtype=pl.Float64).alias("pre_tourney_sw_heat_1g"),
            pl.lit(None, dtype=pl.Float64).alias("pre_tourney_sw_heat_3g"),
            pl.lit(None, dtype=pl.Float64).alias("pre_tourney_sw_heat_5g"),
            pl.lit(None, dtype=pl.Float64).alias("pre_tourney_heat_volatility_5g"),
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
    """Build rename mapping for team_features -> low/high suffixed columns."""
    skip = {"sex", "season"}
    mapping: dict[str, str] = {}
    for col in columns:
        if col in skip:
            continue
        if col == "team_id":
            mapping["team_id"] = "team_low"
        else:
            mapping[col] = f"{col}_low"
    low_map = dict(mapping)
    high_map = {}
    for col in columns:
        if col in skip:
            continue
        if col == "team_id":
            high_map["team_id"] = "team_high"
        else:
            high_map[col] = f"{col}_high"
    combined = {}
    for col in columns:
        if col in skip:
            continue
        if col == "team_id":
            combined[col] = f"{col}__BOTH"
        else:
            combined[col] = f"{col}_low"
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

    if "heat_trend_low" in paired.columns:
        diffs.append(
            (pl.col("heat_trend_low").fill_null(0.0) - pl.col("heat_trend_high").fill_null(0.0)).alias("diff_heat_trend")
        )
        diffs.append(
            (pl.col("abs_heat_5g_low").fill_null(0.0) - pl.col("abs_heat_5g_high").fill_null(0.0)).alias("diff_abs_heat_5g")
        )

    if "pre_tourney_sw_heat_5g_low" in paired.columns:
        diffs.append(
            (pl.col("pre_tourney_sw_heat_5g_low").fill_null(0.0) - pl.col("pre_tourney_sw_heat_5g_high").fill_null(0.0)).alias("diff_sw_heat_5g")
        )
    if "pre_tourney_heat_volatility_5g_low" in paired.columns:
        diffs.append(
            (pl.col("pre_tourney_heat_volatility_5g_low").fill_null(0.0) - pl.col("pre_tourney_heat_volatility_5g_high").fill_null(0.0)).alias("diff_heat_volatility")
        )

    if "sos_low" in paired.columns:
        diffs.append(
            (pl.col("sos_low").fill_null(1500.0) - pl.col("sos_high").fill_null(1500.0)).alias("diff_sos")
        )

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

    if "off_rating_low" in paired.columns:
        diffs.extend([
            (pl.col("off_rating_low").fill_null(100.0) - pl.col("off_rating_high").fill_null(100.0)).alias("diff_off_rating"),
            (pl.col("def_rating_high").fill_null(100.0) - pl.col("def_rating_low").fill_null(100.0)).alias("diff_def_rating"),
            (pl.col("net_rating_low").fill_null(0.0) - pl.col("net_rating_high").fill_null(0.0)).alias("diff_net_rating"),
        ])

    if "massey_avg_rank_low" in paired.columns:
        diffs.extend([
            (pl.col("massey_avg_rank_high").fill_null(175.0) - pl.col("massey_avg_rank_low").fill_null(175.0)).alias("diff_massey_rank"),
            (pl.col("massey_best_rank_high").fill_null(175.0) - pl.col("massey_best_rank_low").fill_null(175.0)).alias("diff_massey_best"),
        ])

    diffs.append(
        (pl.col("seed_low").fill_null(8).cast(pl.Float64) * pl.col("seed_high").fill_null(8).cast(pl.Float64)).alias("seed_product")
    )

    if "consensus_low_spread" in paired.columns:
        diffs.append(pl.col("consensus_low_spread").fill_null(0.0).alias("consensus_low_spread_filled"))

    if "quality_low" in paired.columns:
        diffs.append(
            (pl.col("quality_low").fill_null(0.0) - pl.col("quality_high").fill_null(0.0)).alias("diff_quality")
        )
        if "season_end_elo_low" in paired.columns:
            diffs.append(
                ((pl.col("quality_low").fill_null(0.0) - pl.col("quality_high").fill_null(0.0))
                 - (pl.col("season_end_elo_low").fill_null(1500.0) - pl.col("season_end_elo_high").fill_null(1500.0)) / 100.0)
                .alias("diff_quality_minus_elo")
            )

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

    paired = paired.with_columns(pl.lit(0.0, dtype=pl.Float64).alias("consensus_low_spread"))

    return _compute_diff_features(paired)
