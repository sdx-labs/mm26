"""Backward-compatible shim — re-exports all public symbols from sub-modules.

Consumers that ``from mm26.pipeline import X`` will continue to work.
New code should import from the specific sub-module directly.
"""
from __future__ import annotations

# ── config ──────────────────────────────────────────────────────────────
from .config import (
    PipelineConfig,
    normalize_name,
    _infer_sex_from_team_id,
    _invert_wloc_expr,
    _json_default,
    _read_parquet,
    _required_kaggle_schemas,
    _validation_split_metadata,
    _write_parquet,
)

# ── ingest ──────────────────────────────────────────────────────────────
from .ingest import (
    _build_cbbd_configuration,
    _load_env_value,
    _normalize_game_team_record,
    ingest_cbbd,
    ingest_kaggle,
)

# ── validate ────────────────────────────────────────────────────────────
from .validate import run_validations

# ── transform ───────────────────────────────────────────────────────────
from .transform import (
    _aggregate_consensus_lines,
    _build_cbbd_games_clean,
    _build_cbbd_lines_clean,
    run_transform,
)

# ── ratings ─────────────────────────────────────────────────────────────
from .ratings import (
    _compute_elo_ratings,
    _compute_heat_scores,
    _compute_quality_scores,
    _get_pre_tournament_heat,
    _load_massey_features,
    _load_seed_map,
)

# ── features ────────────────────────────────────────────────────────────
from .features import (
    _build_team_season_features,
    _build_training_pairs,
    _pair_features_from_ids,
)

# ── model ───────────────────────────────────────────────────────────────
from .model import (
    _EnsembleModel,
    _dynamic_clip,
    _feature_ablation_cv,
    _fit_calibration,
    _predict_with_model,
    _predict_with_model_raw,
    _time_series_cv_brier,
    _train_ensemble,
    _train_model,
    _tune_hyperparameters,
)

# ── orchestrate ─────────────────────────────────────────────────────────
from .orchestrate import (
    _build_prob_lookup,
    _simulate_bracket,
    _write_run_manifest,
    cli_main,
    run_elo_and_heat,
    run_gold_and_model,
    run_ingest,
    run_model_only,
    run_pipeline,
)

__all__ = [
    # config
    "PipelineConfig",
    "normalize_name",
    # ingest
    "ingest_kaggle",
    "ingest_cbbd",
    # validate
    "run_validations",
    # transform
    "run_transform",
    # ratings
    "_compute_elo_ratings",
    "_compute_heat_scores",
    "_compute_quality_scores",
    "_get_pre_tournament_heat",
    "_load_massey_features",
    "_load_seed_map",
    # features
    "_build_team_season_features",
    "_build_training_pairs",
    # model
    "_train_model",
    "_train_ensemble",
    "_predict_with_model",
    "_fit_calibration",
    "_time_series_cv_brier",
    # orchestrate
    "_build_prob_lookup",
    "_simulate_bracket",
    "run_ingest",
    "run_elo_and_heat",
    "run_gold_and_model",
    "run_pipeline",
    "run_model_only",
    "cli_main",
]
