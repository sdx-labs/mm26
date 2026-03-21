"""ELO ratings, heat scores, quality scores, Massey ordinals, and seed map loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .config import _read_parquet

try:
    from sklearn.linear_model import Ridge
except Exception:  # pragma: no cover
    Ridge = None


# ---------------------------------------------------------------------------
# ELO Engine
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
# Heat Score Engine
# ---------------------------------------------------------------------------

def _compute_heat_scores(elo_ratings: pl.DataFrame) -> pl.DataFrame:
    """Rolling over-performance relative to ELO expectations with surprise weighting."""
    empty_schema = {
        "sex": pl.Utf8, "season": pl.Int64, "day_num": pl.Int64,
        "team_id": pl.Int64, "heat_delta": pl.Float64,
        "heat_1g": pl.Float64, "heat_3g": pl.Float64, "heat_5g": pl.Float64,
        "sw_heat_delta": pl.Float64,
        "sw_heat_1g": pl.Float64, "sw_heat_3g": pl.Float64, "sw_heat_5g": pl.Float64,
        "heat_volatility_5g": pl.Float64,
    }
    if elo_ratings.height == 0:
        return pl.DataFrame(schema=empty_schema)

    base = (
        elo_ratings.with_columns(
            (pl.col("actual_margin").cast(pl.Float64) - pl.col("expected_margin")).alias("heat_delta"),
            pl.when(pl.col("actual_win") == 1)
            .then(-pl.col("expected_win_prob").clip(0.02, 0.98).log(base=2))
            .otherwise(-(1.0 - pl.col("expected_win_prob").clip(0.02, 0.98)).log(base=2))
            .alias("_surprise_weight"),
        )
        .with_columns(
            (pl.col("heat_delta") * pl.col("_surprise_weight")).alias("sw_heat_delta")
        )
        .sort(["sex", "season", "team_id", "day_num"])
    )

    lagged = base
    for i in range(1, 6):
        lagged = lagged.with_columns(
            pl.col("heat_delta").shift(i).over(["sex", "season", "team_id"]).alias(f"_lag{i}"),
            pl.col("sw_heat_delta").shift(i).over(["sex", "season", "team_id"]).alias(f"_sw_lag{i}"),
        )

    result = lagged.with_columns(
        pl.col("_lag1").alias("heat_1g"),
        pl.mean_horizontal("_lag1", "_lag2", "_lag3").alias("heat_3g"),
        pl.mean_horizontal("_lag1", "_lag2", "_lag3", "_lag4", "_lag5").alias("heat_5g"),
        pl.col("_sw_lag1").alias("sw_heat_1g"),
        pl.mean_horizontal("_sw_lag1", "_sw_lag2", "_sw_lag3").alias("sw_heat_3g"),
        pl.mean_horizontal("_sw_lag1", "_sw_lag2", "_sw_lag3", "_sw_lag4", "_sw_lag5").alias("sw_heat_5g"),
    )

    sw_lag_cols = [f"_sw_lag{i}" for i in range(1, 6)]
    mean_expr = pl.mean_horizontal(*sw_lag_cols)
    var_expr = pl.mean_horizontal(*[(pl.col(c) - mean_expr) ** 2 for c in sw_lag_cols])
    result = result.with_columns(var_expr.sqrt().alias("heat_volatility_5g"))

    return result.select(
        "sex", "season", "day_num", "team_id",
        "heat_delta", "heat_1g", "heat_3g", "heat_5g",
        "sw_heat_delta", "sw_heat_1g", "sw_heat_3g", "sw_heat_5g",
        "heat_volatility_5g",
    )


def _get_pre_tournament_heat(heat_scores: pl.DataFrame, tourney_cutoff_day: int = 132) -> pl.DataFrame:
    """Last heat row per team per season before the tournament."""
    return (
        heat_scores.filter(pl.col("day_num") <= tourney_cutoff_day)
        .sort(["sex", "season", "team_id", "day_num"])
        .group_by(["sex", "season", "team_id"])
        .last()
    )


def _compute_quality_scores(game_fact: pl.DataFrame, alpha: float = 1.0,
                            recency_gamma: float = 0.5) -> pl.DataFrame:
    """Ridge-regularized schedule-adjusted quality metric."""
    if Ridge is None or game_fact.height == 0:
        return pl.DataFrame(schema={
            "sex": pl.Utf8, "season": pl.Int64, "team_id": pl.Int64,
            "quality": pl.Float64, "quality_rank": pl.Int64,
        })

    reg = (
        game_fact.filter(pl.col("day_num") <= 132)
        .filter(pl.col("team_id") == pl.col("team_low"))
        .select("sex", "season", "day_num", "team_low", "team_high",
                "team_score", "opp_score", "team_loc")
    )

    results: list[dict[str, Any]] = []

    for (sex, season), group_df in reg.group_by(["sex", "season"]):
        rows = group_df.to_dicts()
        if len(rows) < 30:
            continue

        team_ids: set[int] = set()
        for r in rows:
            team_ids.add(int(r["team_low"]))
            team_ids.add(int(r["team_high"]))
        tid_list = sorted(team_ids)
        tid_to_idx = {t: i for i, t in enumerate(tid_list)}
        n_teams = len(tid_list)

        n_games = len(rows)
        X = np.zeros((n_games, n_teams + 1), dtype=np.float64)
        y = np.zeros(n_games, dtype=np.float64)
        w = np.zeros(n_games, dtype=np.float64)

        max_day = max(r["day_num"] for r in rows)
        for g, r in enumerate(rows):
            t_low_idx = tid_to_idx[int(r["team_low"])]
            t_high_idx = tid_to_idx[int(r["team_high"])]
            X[g, t_low_idx] = 1.0
            X[g, t_high_idx] = -1.0

            loc = r["team_loc"] if r["team_loc"] else "N"
            if loc == "H":
                X[g, n_teams] = 1.0
            elif loc == "A":
                X[g, n_teams] = -1.0

            y[g] = float(r["team_score"]) - float(r["opp_score"])
            day_frac = float(r["day_num"]) / max_day if max_day > 0 else 1.0
            w[g] = np.exp(-recency_gamma * (1.0 - day_frac))

        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(X, y, sample_weight=w)

        team_coefs = model.coef_[:n_teams]
        for tid, idx in tid_to_idx.items():
            results.append({
                "sex": sex, "season": season, "team_id": tid,
                "quality": float(team_coefs[idx]),
            })

    if not results:
        return pl.DataFrame(schema={
            "sex": pl.Utf8, "season": pl.Int64, "team_id": pl.Int64,
            "quality": pl.Float64, "quality_rank": pl.Int64,
        })

    df = pl.DataFrame(results)
    df = df.with_columns(
        pl.col("quality").rank(descending=True).over(["sex", "season"]).cast(pl.Int64).alias("quality_rank")
    )
    return df


def _load_massey_features(ingest_manifest: dict[str, Any]) -> pl.DataFrame:
    """Load Massey ordinal rankings and compute per-team aggregate features."""
    files = ingest_manifest["kaggle"]["files"]
    if "MMasseyOrdinals" not in files:
        return pl.DataFrame(schema={
            "sex": pl.Utf8, "season": pl.Int64, "team_id": pl.Int64,
            "massey_avg_rank": pl.Float64, "massey_median_rank": pl.Float64,
            "massey_best_rank": pl.Float64, "massey_system_count": pl.Int64,
        })

    ordinals = _read_parquet(Path(files["MMasseyOrdinals"]["artifact"]))
    pre_tourney = ordinals.filter(pl.col("RankingDayNum") <= 133)
    latest_day = (
        pre_tourney.group_by(["Season", "SystemName"])
        .agg(pl.col("RankingDayNum").max().alias("max_day"))
    )
    final = (
        pre_tourney.join(latest_day, on=["Season", "SystemName"])
        .filter(pl.col("RankingDayNum") == pl.col("max_day"))
        .drop("max_day")
    )

    agg = (
        final.group_by(["Season", "TeamID"])
        .agg(
            pl.col("OrdinalRank").mean().alias("massey_avg_rank"),
            pl.col("OrdinalRank").median().alias("massey_median_rank"),
            pl.col("OrdinalRank").min().alias("massey_best_rank"),
            pl.col("OrdinalRank").count().alias("massey_system_count"),
        )
        .with_columns(
            pl.lit("M").alias("sex"),
            pl.col("Season").alias("season"),
            pl.col("TeamID").cast(pl.Int64).alias("team_id"),
        )
        .select("sex", "season", "team_id", "massey_avg_rank", "massey_median_rank",
                "massey_best_rank", "massey_system_count")
    )
    return agg


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
