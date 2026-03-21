"""Model training, ensembling, calibration, prediction, and hyperparameter tuning."""

from __future__ import annotations

from itertools import product
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


# ---------------------------------------------------------------------------
# Core training / prediction
# ---------------------------------------------------------------------------

def _train_model(train_df: pl.DataFrame, feature_cols: list[str], n_estimators: int = 800,
                 max_depth: int = 4, learning_rate: float = 0.03,
                 subsample: float = 0.8, colsample_bytree: float = 0.6,
                 min_child_weight: int = 5, gamma: float = 0.2,
                 reg_alpha: float = 0.15, reg_lambda: float = 3.0) -> Any:
    if train_df.height == 0:
        return None
    cleaned = train_df.select(
        [pl.col(c).fill_null(0.0) for c in feature_cols] + [pl.col("target_low_wins")]
    ).drop_nulls()
    if cleaned.height == 0:
        return None
    y_values = cleaned.select("target_low_wins").to_series().to_numpy().astype(np.float64)
    if len(set(y_values.tolist())) < 2:
        return None
    x_values = cleaned.select(feature_cols).to_numpy()

    if XGBClassifier is not None:
        n = len(y_values)
        use_early_stopping = n >= 200  # Need enough data for a meaningful val split

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
            early_stopping_rounds=50 if use_early_stopping else None,
            verbosity=0,
        )

        if use_early_stopping:
            n_val = max(30, int(n * 0.15))
            rng = np.random.RandomState(42)
            indices = rng.permutation(n)
            train_idx, val_idx = indices[n_val:], indices[:n_val]
            model.fit(
                x_values[train_idx], y_values[train_idx],
                eval_set=[(x_values[val_idx], y_values[val_idx])],
                verbose=False,
            )
        else:
            model.fit(x_values, y_values)
        return model

    if LogisticRegression is not None:
        model = LogisticRegression(max_iter=1000)
        model.fit(x_values, y_values)
        return model

    return None


class _EnsembleModel:
    """Blends XGBoost and Logistic Regression predictions for smoother Brier scores."""

    def __init__(self, xgb_model: Any, lr_model: Any, blend_alpha: float = 0.7):
        self.xgb_model = xgb_model
        self.lr_model = lr_model
        self.blend_alpha = blend_alpha

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        xgb_proba = self.xgb_model.predict_proba(x)[:, 1] if self.xgb_model is not None else np.full(x.shape[0], 0.5)
        lr_proba = self.lr_model.predict_proba(x)[:, 1] if self.lr_model is not None else np.full(x.shape[0], 0.5)
        blended = self.blend_alpha * xgb_proba + (1.0 - self.blend_alpha) * lr_proba
        # Return in sklearn format (2-column)
        return np.column_stack([1.0 - blended, blended])

    @property
    def feature_importances_(self) -> np.ndarray | None:
        if self.xgb_model is not None and hasattr(self.xgb_model, "feature_importances_"):
            return self.xgb_model.feature_importances_
        return None


def _train_ensemble(train_df: pl.DataFrame, feature_cols: list[str],
                    blend_alpha: float = 0.7, **xgb_kwargs: Any) -> Any:
    """Train an XGBoost + Logistic Regression ensemble."""
    if train_df.height == 0:
        return None
    cleaned = train_df.select(
        [pl.col(c).fill_null(0.0) for c in feature_cols] + [pl.col("target_low_wins")]
    ).drop_nulls()
    if cleaned.height == 0:
        return None
    y_values = cleaned.select("target_low_wins").to_series().to_numpy().astype(np.float64)
    if len(set(y_values.tolist())) < 2:
        return None

    from sklearn.preprocessing import StandardScaler

    x_values = cleaned.select(feature_cols).to_numpy()

    xgb_model = _train_model(train_df, feature_cols, **xgb_kwargs)

    lr_model = None
    if LogisticRegression is not None:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_values)

        class _ScaledLR:
            def __init__(self, lr: Any, sc: Any):
                self.lr = lr
                self.scaler = sc

            def predict_proba(self, x: np.ndarray) -> np.ndarray:
                return self.lr.predict_proba(self.scaler.transform(x))

        lr = LogisticRegression(max_iter=2000, C=0.5)
        lr.fit(x_scaled, y_values)
        lr_model = _ScaledLR(lr, scaler)

    if xgb_model is None and lr_model is None:
        return None

    return _EnsembleModel(xgb_model, lr_model, blend_alpha=blend_alpha)


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

        fold_model = _train_ensemble(train_fold, feature_cols)
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


# ---------------------------------------------------------------------------
# Cross-validation & tuning
# ---------------------------------------------------------------------------

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

        model = _train_ensemble(train_fold, feature_cols, **model_kwargs)
        if model is None:
            continue

        preds = _predict_with_model_raw(model, test_fold, feature_cols)
        actuals = test_fold.select("target_low_wins").to_series().to_numpy().astype(np.float64)
        brier = float(np.mean((preds - actuals) ** 2))
        fold_briers.append(brier)

    avg_brier = float(np.mean(fold_briers)) if fold_briers else 1.0
    return avg_brier, fold_briers


def _feature_ablation_cv(training: pl.DataFrame, candidate_features: list[str],
                         sex: str = "M", **model_kwargs: Any) -> dict[str, dict[str, float]]:
    """Leave-one-out feature ablation using time-series CV Brier.

    Returns {"feature_name": {"brier_without": float, "delta": float}} where
    delta = brier_without - brier_all (positive = feature helps, drop hurts).
    """
    baseline_brier, _ = _time_series_cv_brier(training, candidate_features, sex=sex, **model_kwargs)
    results: dict[str, dict[str, float]] = {"__baseline__": {"brier_without": baseline_brier, "delta": 0.0}}

    for feat in candidate_features:
        reduced = [f for f in candidate_features if f != feat]
        if not reduced:
            continue
        feat_brier, _ = _time_series_cv_brier(training, reduced, sex=sex, **model_kwargs)
        results[feat] = {"brier_without": feat_brier, "delta": feat_brier - baseline_brier}

    return results


def _tune_hyperparameters(training: pl.DataFrame, feature_cols: list[str],
                          sex: str = "M") -> dict[str, Any]:
    """Grid search over focused XGBoost hyperparameter space using time-series CV Brier."""
    param_grid = {
        "n_estimators": [300, 500, 700],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.02, 0.04, 0.06],
        "min_child_weight": [1, 3, 5],
    }

    keys = list(param_grid.keys())
    best_brier = float("inf")
    best_params: dict[str, Any] = {}
    all_results: list[dict[str, Any]] = []

    for combo in product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        brier, folds = _time_series_cv_brier(training, feature_cols, sex=sex, **params)
        all_results.append({**params, "cv_brier": brier, "folds": folds})
        if brier < best_brier:
            best_brier = brier
            best_params = params

    return {
        "best_params": best_params,
        "best_cv_brier": best_brier,
        "all_results": sorted(all_results, key=lambda x: x["cv_brier"]),
    }
