"""Pipeline orchestration: stage runners, bracket simulation, and CLI entry-point."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .config import PipelineConfig, _json_default, _read_parquet, _validation_split_metadata, _write_parquet
from .features import _build_team_season_features, _pair_features_from_ids, _build_training_pairs
from .ingest import ingest_cbbd, ingest_kaggle
from .model import (
    LogisticRegression,
    XGBClassifier,
    _EnsembleModel,
    _fit_calibration,
    _predict_with_model,
    _time_series_cv_brier,
    _train_ensemble,
)
from .ratings import _compute_elo_ratings, _compute_heat_scores, _compute_quality_scores, _load_massey_features, _load_seed_map
from .transform import run_transform
from .validate import run_validations


# ---------------------------------------------------------------------------
# Ingest orchestrator
# ---------------------------------------------------------------------------

def run_ingest(config: PipelineConfig) -> dict[str, Any]:
    kaggle_manifest = ingest_kaggle(config)
    cbbd_manifest = ingest_cbbd(config)
    return {"kaggle": kaggle_manifest, "cbbd": cbbd_manifest}


# ---------------------------------------------------------------------------
# ELO + Heat orchestrator
# ---------------------------------------------------------------------------

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
# Monte Carlo Bracket Simulation helpers
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
# Gold + Model orchestrator
# ---------------------------------------------------------------------------

def run_gold_and_model(config: PipelineConfig, ingest_manifest: dict[str, Any]) -> dict[str, Any]:
    game_fact = _read_parquet(config.silver_dir / "game_fact.parquet")
    elo_ratings = _read_parquet(config.silver_dir / "elo_ratings.parquet")
    heat_scores = _read_parquet(config.silver_dir / "heat_scores.parquet")
    cbbd_line_consensus = _read_parquet(config.silver_dir / "cbbd_line_consensus.parquet")
    sample = _read_parquet(Path(ingest_manifest["kaggle"]["files"]["SampleSubmissionStage2"]["artifact"]))

    seed_map = _load_seed_map(ingest_manifest)
    team_features = _build_team_season_features(game_fact, elo_ratings, heat_scores)

    # Compute Ridge quality scores and join into team features
    quality_scores = _compute_quality_scores(game_fact)
    if quality_scores.height > 0:
        team_features = team_features.join(
            quality_scores.select("sex", "season", "team_id", "quality"),
            on=["sex", "season", "team_id"],
            how="left",
        )
    else:
        team_features = team_features.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("quality"),
        )

    # Load and join Massey ordinal features (Men only for now)
    massey_features = _load_massey_features(ingest_manifest)
    if massey_features is not None and massey_features.height > 0:
        team_features = team_features.join(
            massey_features.select("sex", "season", "team_id",
                                   "massey_avg_rank", "massey_median_rank",
                                   "massey_best_rank", "massey_system_count"),
            on=["sex", "season", "team_id"],
            how="left",
        )
    else:
        team_features = team_features.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("massey_avg_rank"),
            pl.lit(None, dtype=pl.Float64).alias("massey_median_rank"),
            pl.lit(None, dtype=pl.Float64).alias("massey_best_rank"),
            pl.lit(None, dtype=pl.Float64).alias("massey_system_count"),
        )

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
        "diff_sw_heat_5g",
        "diff_heat_volatility",
        "diff_quality",
        "diff_quality_minus_elo",
        "diff_off_rating",
        "diff_def_rating",
        "diff_net_rating",
        "diff_massey_rank",
        "diff_massey_best",
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

        model = _train_ensemble(train_fit, feature_cols, blend_alpha=0.7, n_estimators=n_est)

        # Fit isotonic calibration on time-series CV out-of-fold predictions
        calibrator = _fit_calibration(model, training, feature_cols, sex=sex)

        pred_values = _predict_with_model(model, pred_sex, feature_cols, calibrator=calibrator)
        pred_df = pred_sex.select("ID").with_columns(pred_values.alias("Pred"))
        predictions.append(pred_df)

        # Holdout Brier score
        holdout_brier = None
        holdout_diagnostics: dict[str, Any] = {}
        cv_brier = None
        if holdout.height > 0 and model is not None:
            holdout_preds = _predict_with_model(model, holdout, feature_cols, calibrator=calibrator)
            actuals = holdout.select("target_low_wins").to_series().to_numpy().astype(np.float64)
            preds_np = holdout_preds.to_numpy()
            holdout_brier = float(np.mean((preds_np - actuals) ** 2))

            # ECE (Expected Calibration Error) — 10 equal-width bins
            n_bins = 10
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
            ece = 0.0
            bin_details = []
            for i in range(n_bins):
                mask = (preds_np >= bin_edges[i]) & (preds_np < bin_edges[i + 1])
                if i == n_bins - 1:
                    mask = (preds_np >= bin_edges[i]) & (preds_np <= bin_edges[i + 1])
                count = int(mask.sum())
                if count > 0:
                    avg_pred = float(preds_np[mask].mean())
                    avg_actual = float(actuals[mask].mean())
                    ece += abs(avg_pred - avg_actual) * count
                    bin_details.append({"bin": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}", "count": count,
                                        "avg_pred": round(avg_pred, 4), "avg_actual": round(avg_actual, 4)})
            ece = ece / len(actuals) if len(actuals) > 0 else None
            holdout_diagnostics["ece"] = round(ece, 6) if ece is not None else None
            holdout_diagnostics["calibration_bins"] = bin_details

            # Per-seed Brier (using lower-seeded team's seed)
            if "seed_low" in holdout.columns:
                seed_brier = {}
                seeds = holdout.select("seed_low").to_series().to_numpy()
                for s in sorted(set(seeds)):
                    s_mask = seeds == s
                    if s_mask.sum() > 0:
                        seed_brier[int(s)] = round(float(np.mean((preds_np[s_mask] - actuals[s_mask]) ** 2)), 6)
                holdout_diagnostics["per_seed_brier"] = seed_brier

            # Brier decomposition: reliability, resolution, uncertainty
            mean_actual = float(actuals.mean())
            uncertainty = mean_actual * (1.0 - mean_actual)
            holdout_diagnostics["brier_decomposition"] = {
                "uncertainty": round(uncertainty, 6),
                "base_rate": round(mean_actual, 4),
            }

        # Time-series CV Brier
        if sex == "M":
            cv_brier_val, cv_folds = _time_series_cv_brier(training, feature_cols, sex=sex)
            cv_brier = {"mean": cv_brier_val, "folds": cv_folds}
        elif sex == "W" and train_sex.height > 0:
            cv_brier_val, cv_folds = _time_series_cv_brier(training, feature_cols, sex=sex)
            if cv_brier_val is not None:
                cv_brier = {"mean": cv_brier_val, "folds": cv_folds}

        model_type = "flat_fallback"
        if model is not None:
            if isinstance(model, _EnsembleModel):
                model_type = "ensemble_xgb_lr"
            elif XGBClassifier is not None and isinstance(model, XGBClassifier):
                model_type = "xgboost"
            else:
                model_type = "logistic_regression"

        # Feature importance
        feature_importance = {}
        if model is not None and hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            if importances is not None:
                feature_importance = {
                    fc: round(float(imp), 6)
                    for fc, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1])
                }

        model_stats[sex] = {
            "train_rows": train_fit.height,
            "holdout_rows": holdout.height,
            "predict_rows": pred_sex.height,
            "model_type": model_type,
            "good_train_rows": train_fit.select(feature_cols).drop_nulls().height,
            "holdout_brier": holdout_brier,
            "holdout_diagnostics": holdout_diagnostics if holdout_diagnostics else None,
            "cv_brier": cv_brier,
            "calibrated": calibrator is not None,
            "feature_importance": feature_importance if feature_importance else None,
        }

    submission = pl.concat(predictions, how="vertical")
    submission = sample.select("ID").join(submission, on="ID", how="left").with_columns(
        pl.col("Pred").fill_null(0.5)
    )

    # Simulation blending removed — pure model output (calibrated + clipped)
    sim_meta: dict[str, Any] = {"simulated": False, "blend_applied": False}

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


# ---------------------------------------------------------------------------
# Run manifest & full pipeline
# ---------------------------------------------------------------------------

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
