# Brier Score Optimization v1 — Feature & Diagnostics Upgrade

This ExecPlan is a living document. The sections Progress, Surprises & Discoveries, Decision Log, and Outcomes & Retrospective must be kept up to date as work proceeds.

This document must be maintained in accordance with .agents/PLANS.md.

## Purpose / Big Picture

The current men's holdout Brier is 0.138, CV mean is 0.202, and women's model has no holdout validation at all. This plan targets a holdout Brier under 0.120 (men's) and establishes a women's baseline by adding high-signal features, improving efficiency metrics, and building proper diagnostics.

After this change, the pipeline produces:

- Massey ordinal-derived features (average ranking from top systems like POM, SAG, etc.)
- Tempo-free offensive and defensive efficiency ratings
- Conference-adjusted strength signals
- Enhanced model performance diagnostics (ECE, per-seed Brier, Brier decomposition, reliability curves)
- Women's time-series CV and holdout evaluation
- Feature ablation results identifying which features actually help

To see it working: run `python main.py run --stage model` and compare the new `model_performance_summary.json` against baseline.

## Progress

- [x] (2026-03-18 16:00Z) Read and analyzed full pipeline.py (2370 lines)
- [x] (2026-03-18 16:00Z) Identified baseline metrics: M holdout=0.138, CV=0.202, W=no eval
- [x] (2026-03-18 16:00Z) Inventoried available data: MMasseyOrdinals.csv is rich but unused
- [ ] Implement Massey ordinals integration (_load_massey_ordinals, diff_massey features)
- [ ] Implement tempo-free efficiency metrics (ORtg, DRtg per 100 possessions)
- [ ] Enhance diagnostics (ECE, Brier decomposition, per-seed analysis, reliability)
- [ ] Add women's CV and holdout evaluation
- [ ] Run feature ablation and prune weak features
- [ ] Calibration upgrade (comparison of isotonic vs Platt)
- [ ] Full pipeline run and holdout comparison vs baseline

## Surprises & Discoveries

(To be populated during implementation)

## Decision Log

- Decision: Start with Massey ordinals as highest-impact new feature source.
  Rationale: MMasseyOrdinals.csv contains rankings from 100+ systems per season since 2003. Composite rankings (Pomeroy, Sagarin, RPI) are among the strongest single predictors of tournament outcomes. Adding diff_massey_avg_rank can encode expert consensus directly.
  Date: 2026-03-18

- Decision: Add tempo-free efficiency as second priority feature.
  Rationale: Current features use raw scoring averages which are pace-dependent. Teams that play fast score more but also allow more. ORtg/DRtg per 100 possessions isolates true efficiency.
  Date: 2026-03-18

- Decision: Add diagnostics before calibration changes so we can measure impact.
  Rationale: Cannot optimize calibration without knowing where the model is miscalibrated.
  Date: 2026-03-18

## Outcomes & Retrospective

(To be populated at completion)

## Context and Orientation

Key files:
- src/mm26/pipeline.py — all pipeline logic (2370 lines), edit target
- data/MMasseyOrdinals.csv — rich ranking data, currently unused
- artifacts/latest/reports/model_performance_summary.json — baseline metrics
- tests/test_pipeline_contracts.py — contract tests
- tests/test_pipeline_utils.py — unit tests

The pipeline processes data in phases: Ingest (bronze) → Transform (silver) → ELO/Heat → Gold/Model. The model phase (run_gold_and_model) trains XGBoost on historical tournament outcomes, calibrates with isotonic regression, and produces submission.csv.

Current feature set (28 features): diff_win_rate, diff_avg_margin, diff_avg_pts_for, diff_defense_proxy, diff_last5_win_rate, diff_last5_avg_margin, diff_elo, diff_heat_{1g,3g,5g}, diff_heat_trend, diff_abs_heat_5g, diff_seed, diff_sos, diff_fg_pct, diff_fg3_pct, diff_ft_pct, diff_opp_fg_pct, diff_reb_margin, diff_ast_to_ratio, diff_stl_blk, diff_possessions, seed_product, consensus_low_spread_filled, diff_sw_heat_5g, diff_heat_volatility, diff_quality, diff_quality_minus_elo.

## Plan of Work

### Milestone 1: Massey Ordinals Integration

Add a function `_load_massey_ordinals(files, target_season)` that:
1. Reads MMasseyOrdinals.csv from bronze
2. Filters to final pre-tournament rankings (RankingDayNum == 133 or highest available)
3. For each (season, team): computes average ordinal rank across all systems
4. Joins into team_season_features as "massey_avg_rank"
5. Adds "diff_massey_rank" as pairwise feature

### Milestone 2: Tempo-Free Efficiency Metrics

In `_build_team_season_features`, add:
1. ORtg = 100 * team_score / possessions
2. DRtg = 100 * opp_score / possessions
3. NetRtg = ORtg - DRtg
4. Add diff_net_rating, diff_off_rating, diff_def_rating as pairwise features

### Milestone 3: Enhanced Diagnostics

In `run_gold_and_model`, after computing model_stats, add:
1. ECE (Expected Calibration Error) computed on holdout
2. Per-seed-tier Brier scores
3. Brier decomposition (calibration + discrimination + uncertainty)
4. Reliability curve data (10 decile bins)
5. Women's time-series CV evaluation

### Milestone 4: Feature Ablation

Wire in `_feature_ablation_cv` call during model run:
1. Run ablation for M on time-series CV
2. Identify features with positive delta (removing them helps)
3. Prune those features
4. Report results

## Validation and Acceptance

Run `python main.py run --stage model` and verify:
1. model_performance_summary.json contains new diagnostic fields
2. Men's holdout Brier improves (target: < 0.130)
3. Men's CV Brier mean improves (target: < 0.195)
4. Women's CV Brier is now reported (non-null)
5. Feature count changes and ablation results are recorded
6. All existing tests pass: `python -m pytest tests/ -v`
