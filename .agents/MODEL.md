# March Machine Learning Madness Model

the underlying thesis of the machine learning modeling behind project is the "cinderalla system" - an attempt to create a "heat" variable which can be used in training to better increase the accuracy of the model. 

The "heat" variable is going to be used as a proxy for teams who are outperforming expectations. So first we must establish an "expectations" baseline by creating a holistic and thorough model, then compare that to the teams past performance on a 1 game, a 3 game, and a 5 game basis - teams who are "overperforming", or winning games that they have not been "expected" to win based on the original holistic model, should be assigned a "heat" variable which should have a significant impact on the outcome of the model.

Then we are going to run a model which performs many simulations of the current 2026 bracket (and all potential 2026 march madness matchups) - and compose a composite prediction, in accordance with Kaggle guidelines outlined in the .agents/kaggle/* folder. the composite prediction should be influenced by both the initial prediciton model, based on a derived ELO, and the "heat" variable, along with standard statistics included in the datasets available. Hopefully, the "heat" variable helps us predict some potential cinderalla's - across the many, many simulations, some teams may be consistently overperforming their expected outcome, and their heat variable would then rise, and influence their overall expected win probability.

rememebr that the end result of the prediction should be completely and totally optimized for the brier score, as outlined in the .agents/kaggle/ folders.

the "heat" variable is calculated in multiple parts:

1. the first is to create a model based on standard ELO - using only the data available in the kaggle data set, we are going to estimate the probabiltiy of a win using an XGBoost model based on stats available.
2. then we are going to compute an initial expected win probability for the low team to beat the high team.
3. then we are going to use this to create a 1 game, a 3 game, and a 5 game "heat" score which compares the teams actual performance over the last 5 games to the performance that was initially predicted.
4. then we are going to run many simulations which are going to result in an end predicition of the bracket, and a submission file acceptable for the rules and optimized for the brier score.

## Updates

so we are going to implement a few updates to the current pipeline based on the following notes.

on a high level, this is a multi-step process:
    - original xgboost model based on a calculated ELO
    - predict probability of winning for that matchup - avoid all leakage across the pipeline and the CV folds
    - use that predicted probability to create the "heat" variable - which is then used as a feature in each team's *next* game
    - we are going to get rid of the simulations, for now, as they seem to only worsen the brier score

the plan is to read the 24.md and the 25.md, and the notes below, compare them to the above strategy, and beginning implementing changes. each change should be validated that it increases the brier score. we are going to use the notes to improve our ELO calculation and the math behind it by improving the quality of the features that are used. we are also going to add a mathematically rigorous way to scale the "heat" variable (e.g., the core idea is to identify potential "cinderalla's" who could make a run, a team with a 10% chance of winning, who actually wins, should be rewarded in their next game with an increased probability of winning)

then we are going to begin a more rigorous feature selection and hyperparameter tuning on the XGBoost model with the upgraded ELO and the

### Notes on 25.md

You want a deep breakdown of that prior “quality” metric plus mathematically solid upgrades grounded in your available columns; I’ll first inspect the pipeline feature construction so the improvement ideas are concrete to this repo’s data.

Read [](file:///c%3A/Users/sdiehl/Desktop/seandiehlprojects/mm26/src/mm26/pipeline.py#1311-1311), lines 1311 to 1719

Read [](file:///c%3A/Users/sdiehl/Desktop/seandiehlprojects/mm26/src/mm26/pipeline.py#1060-1060), lines 1060 to 1299

Read [](file:///c%3A/Users/sdiehl/Desktop/seandiehlprojects/mm26/README.md)

I’ve pulled the exact ELO/heat and pairwise feature math from the code, and next I’ll inspect the upstream game/line columns plus the model report so the improvement ideas map directly to what you can compute today.

Read [](file:///c%3A/Users/sdiehl/Desktop/seandiehlprojects/mm26/src/mm26/pipeline.py#625-625), lines 625 to 933

Read [](file:///c%3A/Users/sdiehl/Desktop/seandiehlprojects/mm26/artifacts/latest/reports/gold_quality_summary.json)

Read [](file:///c%3A/Users/sdiehl/Desktop/seandiehlprojects/mm26/artifacts/latest/reports/model_performance_summary.json)

Your attached quality metric is a classic schedule-adjusted team-strength model, and it is a strong idea. Here is what it is doing, the math behind it, and several mathematically rigorous upgrades that fit your current pipeline/data.

Code context:
- Prior metric snippet: 25.md
- Your current feature/rating pipeline: pipeline.py
- Current model feature set: model_performance_summary.json

**What The Existing “Quality” Strategy Is Doing**
1. It creates season-team keys like Season/TeamID so strength is estimated separately each season.
2. It defines a modeling pool:
- All tournament teams.
- Plus non-tournament teams that beat tournament teams at least once.
3. It fits, for each season and sex, a linear model on point differential:
- Formula in your file: $PointDiff \sim -1 + T1\_TeamID + T2\_TeamID$
4. It extracts the T1 team coefficients as quality scores, then joins those into tournament matchup rows and uses difference:
- $diff\_quality = quality_{T1} - quality_{T2}$

Interpretation:
- This is a fixed-effects strength model (not truly mixed effects, despite the comment).
- Team coefficients capture schedule-adjusted strength because every game connects two teams in one system.
- Using point differential gives richer signal than binary win/loss.

**Math Behind It (Clean Formulation)**
Let game $g$ be between teams $i$ and $j$ with observed margin $y_g = score_i - score_j$.

A canonical version of your model is:
$$
y_g = \theta_i - \theta_j + \epsilon_g,\quad \epsilon_g \sim \mathcal{N}(0,\sigma^2)
$$

In matrix form:
$$
\mathbf{y} = \mathbf{X}\boldsymbol{\theta} + \boldsymbol{\epsilon}
$$
where each row has +1 at team $i$ and -1 at team $j$.

Estimated strengths:
$$
\hat{\boldsymbol{\theta}} = \arg\min_{\theta}\|\mathbf{y}-\mathbf{X}\theta\|_2^2
$$
(or pseudo-inverse/regularized equivalent when rank-deficient).

Matchup feature:
$$
diff\_quality_{ij} = \hat{\theta}_i - \hat{\theta}_j
$$

Why it works:
- Opponent adjustment is implicit in the linear system.
- Blowouts and close losses are treated continuously via margin.

**Important Statistical Caveats In The Current Snippet**
1. Identifiability/rank issues:
- No intercept + many team dummies can be non-identifiable unless constrained.
- The “0000” bucket helps but is ad hoc and can distort interpretation.
2. Selection bias from the st filter:
- Restricting teams to “tournament or beat tournament team” changes the game graph and can inflate variance.
3. Homoskedastic Gaussian assumption:
- Margin variance differs by matchup quality and pace.
4. No explicit home/away correction:
- Your current production ELO does include home adjustment in [src/mm26/pipeline.py](src/mm26/pipeline.py), but this snippet does not.
5. No recency dynamics:
- One season-level coefficient ignores late-season form unless manually windowed.

**How This Maps To Your Existing MM26 Data**
Your pipeline already has columns that can support stronger versions:
- Game-level outcomes + location + box stats from detailed Kaggle game tables, transformed in [_expand_team_games] within [src/mm26/pipeline.py](src/mm26/pipeline.py)
- ELO and heat deltas from [_compute_elo_ratings] and [_compute_heat_scores] in [src/mm26/pipeline.py](src/mm26/pipeline.py)
- Consensus market lines and provider count from CBBD line aggregation in [src/mm26/pipeline.py](src/mm26/pipeline.py)
- Current 24 pairwise features in [artifacts/latest/reports/model_performance_summary.json](artifacts/latest/reports/model_performance_summary.json)

So you can add a quality-style feature without changing your data contracts.

**Mathematically Rigorous Upgrade Ideas**
Here are 5 strong options, ordered from easiest/high-impact to more advanced.

1. Ridge-regularized adjusted margin quality (direct upgrade of your snippet)
- Model:
$$
y_g = \theta_i - \theta_j + \beta_{home}H_g + \beta_{ot}OT_g + \epsilon_g
$$
with ridge penalty:
$$
\min_{\theta,\beta}\sum_g w_g\left(y_g-\theta_i+\theta_j-\beta^\top z_g\right)^2 + \lambda\|\theta\|_2^2
$$
- Use recency weights $w_g=\exp(-\gamma\Delta days)$.
- Benefit: stable team strengths, explicit home effect, fewer singularity issues.
- Output features to add:
  - diff_quality_ridge
  - diff_quality_recent (fit on last N games only)
  - quality_uncertainty_diff (if you compute SEs)

2. Bradley-Terry with margin augmentation (probability-native strength)
- Win-probability model:
$$
P(i\text{ beats }j)=\sigma(\theta_i-\theta_j+\beta_{home}H_g)
$$
- Augment with margin channel (two-part model):
  - Logistic for win/loss + Gaussian for conditional margin.
- Benefit: directly aligned with Brier/logloss objectives; can still derive a single quality score $\theta$.
- Features:
  - diff_bt_strength
  - bt_upset_propensity from residuals versus expected win probabilities

3. Possession-adjusted latent efficiency quality
- Use box columns to model offensive/defensive efficiencies:
$$
ORtg_i = 100\cdot \frac{Pts_i}{Poss_i},\quad DRtg_i = 100\cdot \frac{Pts_{opp}}{Poss_{opp}}
$$
- Define latent net quality:
$$
Q_i = \alpha\cdot AdjORtg_i - \beta\cdot AdjDRtg_i
$$
where adjustment is via opponent strength fixed effects.
- Benefit: separates pace from quality; often better tournament transferability.
- Features:
  - diff_adj_netrtg
  - diff_shot_quality (eFG-like proxy from FG/3P/FT columns)
  - diff_turnover_pressure (tov and opp_tov structure)

4. Bayesian dynamic strength (state-space quality + heat consistency)
- Team strength evolves over season:
$$
\theta_{i,t}=\theta_{i,t-1}+\eta_{i,t},\quad \eta_{i,t}\sim\mathcal{N}(0,\tau^2)
$$
- Observation:
$$
y_{g,t}\sim\mathcal{N}((\theta_{i,t}-\theta_{j,t})+\beta_{home}H_{g,t},\sigma^2)
$$
- Benefit: principled uncertainty + true momentum instead of ad hoc windowing.
- Features:
  - diff_theta_pretourney
  - diff_theta_trend_last14d
  - diff_posterior_sd (uncertainty-aware upset signal)

5. Market-calibrated quality fusion (lines as noisy strength observations)
- Treat consensus spread as an external measurement of latent strength gap:
$$
Spread_g = c(\theta_i-\theta_j)+\nu_g
$$

- Jointly estimate strengths from outcomes and spread observations (multi-task objective).
- Benefit: fuses wisdom-of-crowds with team stats; robust when box data are sparse.
- Features:
  - diff_quality_market_fused
  - market_disagreement = modeled_gap - implied_gap_from_spread
  - provider_reliability_weighted_spread using provider_count

**Most Practical Path For Your Current Repo**
Given your existing architecture, I would prioritize:
1. Add one new feature family first: ridge adjusted quality.
2. Add uncertainty and recency variants of that same quality.
3. Re-run current XGBoost stack and check holdout Brier change versus baseline in [artifacts/latest/reports/model_performance_summary.json](artifacts/latest/reports/model_performance_summary.json).

Minimal feature additions that are likely to help:
- diff_quality_ridge
- diff_quality_recent
- diff_quality_minus_elo (captures where quality disagrees with ELO)
- diff_quality_minus_seed (captures “mis-seeded” teams)
- abs_diff_quality_interaction_with_heat:
$$
|diff\_quality|\times diff\_heat\_5g
$$

**Validation Design To Keep It Rigorous**
Use your existing time-series CV pattern and evaluate:
- Mean Brier improvement
- Fold-by-fold stability (not just one lucky season)
- Calibration drift after isotonic (ECE/Brier decomposition if possible)

Quick acceptance rule:
- Keep new quality features only if they improve mean CV Brier and do not worsen worst-fold Brier materially.

If you want, I can next sketch the exact equations-to-columns mapping for a ridge adjusted quality builder that plugs into your existing team_season_features and pairwise feature pipeline without changing notebook contracts.

### Notes on 24.md

rategy breakdown for .agents/kaggle/winning_strategies/24.md:

It imports a prebuilt submission from an external package (goto-conversion), then writes it out.
It computes, for each possible matchup ID in submission.csv, whether that matchup is:
not a tournament matchup,
or a possible Round 1/2/3/4/5/6/7 matchup based only on seeds and region structure.
It allows a manual “risk team” override:
for all potential matchups involving one chosen team up to a chosen round,
force Pred to 1.0 (if risk team is lower ID) or 0.0 (if higher ID).
So this is a leverage/risk-tilting strategy layered on top of an existing forecast, not a predictive model trained from game data.

Math behind the round logic:

The slot ordering list in the file
[1,16,8,9,5,12,4,13,6,11,3,14,7,10,2,15]
is a bracket tree embedding for one region.

For two seeds s1, s2 in the same region:

Let i1 = index(s1), i2 = index(s2) in that list.
Earliest meeting round is found by block membership:
floor(i1/2) == floor(i2/2) -> Round of 64 game pairing
floor(i1/4) == floor(i2/4) -> Round of 32
floor(i1/8) == floor(i2/8) -> Sweet 16
otherwise Elite 8 within region
Across regions, it hard-codes Final Four vs Championship pairing using WX and YZ region groups.

Math behind the risk override:

For matchup m = (a,b), baseline model gives q_m = P(a beats b).
The override sets:

q_m = 1 if chosen team must win and is team a
q_m = 0 if chosen team must win and is team b
This is equivalent to imposing deterministic constraints on uncertain events:
P(risk team wins early rounds) = 1

Under common proper scoring rules this is usually dangerous:

If log loss:
L(y,q) = -( y log q + (1-y) log(1-q) )
Setting q to 0 or 1 creates catastrophic penalty when wrong (infinite without clipping, huge with clipping).

If Brier:
B(y,q) = (y-q)^2
Setting q to 0 or 1 maximizes variance and often worsens expected score unless true win probability is already near 0 or 1.

So the strategy is mathematically a high-variance utility hack for contest differentiation, not an expected-score optimizer.

What is clever in it:

The bracket-round detection from seed structure is conceptually correct for mapping potential matchup depth.
It implements intentional risk preference in a controllable way (team, maximum round).
Main weaknesses:

No use of game data, ELO, heat, box stats, spreads, or calibration.
Hard 0/1 overrides are statistically brittle.
Region pairing is hard-coded and not future-proof to bracket format changes.
Women detection via team1 + team2 > 6000 is heuristic, not principled.
First Four seed suffix handling is acknowledged as fragile in comments.
Rigorous upgrades (same spirit, much better math):

Soft tilt instead of hard override
Replace 0/1 with logit shift:
q'm = sigmoid( logit(q_m) + delta{team,round} )
Choose delta by optimization, not by hand.
Constrained utility optimization
Solve for adjusted probabilities q' maximizing variance/leverage while bounding expected score drop:
maximize U(q')
subject to E[score(q')] >= E[score(q)] - epsilon
and KL(q' || q) <= B
This gives controlled aggressiveness rather than all-or-nothing bets.
Bracket-consistent path probabilities
Use slots and pairwise probabilities to compute each team’s advancement probability by round via dynamic programming or Monte Carlo.
Then derive matchup probabilities consistent with path structure, rather than independent manual edits.
Risk-aware objective for winner-take-all contests
If your true goal is probability of finishing first (not mean logloss), optimize that directly via simulation against a field distribution of opponent submissions.
This is a game-theoretic optimization, and can justify selective contrarian shifts mathematically.
Proper clipping and calibration
Enforce q in [epsilon, 1-epsilon], epsilon around 0.01 to 0.025.
Recalibrate after tilt (isotonic/Platt on historical folds) so shifted forecasts remain probabilistically coherent.
How to connect this to your current pipeline:

Your existing stack in pipeline.py already has calibrated pairwise probabilities, ELO, heat, seeds, and bracket simulation. The best version of “strategy 24” is:

start from calibrated q,
apply small logit tilts tied to seed/round/team narratives,
re-evaluate via time-split backtests and simulated leaderboard utility,
keep only tilts that improve your chosen objective.
If you want, I can next propose a concrete optimization recipe with equations and parameter ranges (delta grid, epsilon, KL budget, and evaluation protocol) tailored to your current artifacts.

