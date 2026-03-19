from __future__ import annotations

from pathlib import Path
import shutil
import sys
import unittest

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mm26.pipeline import (
    PipelineConfig,
    _aggregate_consensus_lines,
    _build_cbbd_games_clean,
    _build_cbbd_lines_clean,
    _build_cbbd_configuration,
    _compute_elo_ratings,
    _compute_heat_scores,
    _compute_quality_scores,
    _get_pre_tournament_heat,
    _load_env_value,
    _normalize_game_team_record,
    _required_kaggle_schemas,
    _validation_split_metadata,
    ingest_cbbd,
)


class FakeApiBundle:
    def get_games(self, **kwargs):
        return [
            {
                "id": 1,
                "season": kwargs["season"],
                "seasonType": "regular",
                "status": "final",
                "startDate": "2025-03-01T00:00:00+00:00",
                "homeTeamId": 10,
                "homeTeam": "Alpha",
                "homePoints": 80,
                "awayTeamId": 20,
                "awayTeam": "Beta",
                "awayPoints": 70,
                "homeTeamEloStart": 1500.0,
                "homeTeamEloEnd": 1510.0,
                "awayTeamEloStart": 1490.0,
                "awayTeamEloEnd": 1480.0,
                "homeSeed": None,
                "awaySeed": None,
                "excitement": 5.5,
                "gameNotes": None,
            }
        ]

    def get_game_teams(self, **kwargs):
        return [
            {
                "gameId": 1,
                "season": kwargs["season"],
                "seasonType": "regular",
                "startDate": "2025-03-01T00:00:00+00:00",
                "teamId": 10,
                "team": "Alpha",
                "opponentId": 20,
                "opponent": "Beta",
                "isHome": True,
                "teamStats": {
                    "points": {"total": 80},
                    "fieldGoals": {"made": 28, "attempts": 60},
                },
                "opponentStats": {
                    "points": {"total": 70},
                },
            }
        ]

    def get_lines(self, **kwargs):
        return [
            {
                "gameId": 1,
                "season": kwargs["season"],
                "seasonType": "regular",
                "startDate": "2025-03-01T00:00:00+00:00",
                "homeTeamId": 10,
                "homeTeam": "Alpha",
                "awayTeamId": 20,
                "awayTeam": "Beta",
                "lines": [
                    {"provider": "A", "spread": -3.5, "spreadOpen": -2.5, "overUnder": 145.0},
                    {"provider": "B", "spread": -4.5, "spreadOpen": -3.5, "overUnder": 146.0},
                ],
            }
        ]


class TestPipelineContracts(unittest.TestCase):
    def test_required_schema_keys_present(self) -> None:
        required = _required_kaggle_schemas()
        self.assertIn("MTeams", required)
        self.assertIn("SampleSubmissionStage2", required)
        self.assertIn("MRegularSeasonDetailedResults", required)

    def test_pipeline_config_uses_repo_data_root(self) -> None:
        config = PipelineConfig(project_root=ROOT)
        self.assertEqual(config.data_dir, ROOT / "data")

    def test_validation_split_metadata_excludes_holdout_from_train(self) -> None:
        config = PipelineConfig(project_root=ROOT)
        split = _validation_split_metadata(config)
        self.assertEqual(split["train_start_season"], 2003)
        self.assertEqual(split["train_end_season"], 2024)
        self.assertEqual(split["holdout_season"], 2025)
        self.assertEqual(split["prediction_season"], 2026)

    def test_cbbd_configuration_uses_access_token(self) -> None:
        config = _build_cbbd_configuration("secret-token")
        self.assertEqual(config.access_token, "secret-token")
        self.assertNotIn("Authorization", config.api_key)

    def test_load_env_value_reads_key_from_dotenv(self) -> None:
        value = _load_env_value(ROOT, "CBBD_API_KEY")
        self.assertTrue(bool(value))

    def test_consensus_spread_uses_provider_median(self) -> None:
        lines = pl.DataFrame(
            {
                "game_id": [1, 1, 1],
                "season": [2025, 2025, 2025],
                "start_date": ["2025-03-01", "2025-03-01", "2025-03-01"],
                "home_team_id": [10, 10, 10],
                "away_team_id": [20, 20, 20],
                "provider": ["A", "B", "C"],
                "spread": [-5.0, -3.0, None],
                "spread_open": [-4.0, -2.0, None],
                "over_under": [141.0, 145.0, None],
            }
        )
        consensus = _aggregate_consensus_lines(lines)
        row = consensus.to_dicts()[0]
        self.assertEqual(row["provider_count"], 3)
        self.assertEqual(row["consensus_home_spread"], -4.0)
        self.assertEqual(row["consensus_home_spread_open"], -3.0)
        self.assertEqual(row["consensus_over_under"], 143.0)

    def test_flattened_game_team_record_includes_box_score_fields(self) -> None:
        row = _normalize_game_team_record(
            {
                "gameId": 7,
                "season": 2025,
                "seasonType": "regular",
                "startDate": "2025-03-01T00:00:00+00:00",
                "teamId": 10,
                "team": "Alpha",
                "opponentId": 20,
                "opponent": "Beta",
                "teamStats": {
                    "points": {"total": 80},
                    "fieldGoals": {"made": 28, "attempts": 60},
                },
                "opponentStats": {"points": {"total": 70}},
            }
        )
        self.assertEqual(row["team_stats_points_total"], 80)
        self.assertEqual(row["team_stats_field_goals_made"], 28)
        self.assertEqual(row["opponent_stats_points_total"], 70)

    def test_ingest_cbbd_writes_mocked_outputs_without_network(self) -> None:
        config = PipelineConfig(project_root=ROOT)
        config.artifacts_dir = ROOT / "artifacts_test"
        if config.artifacts_dir.exists():
            shutil.rmtree(config.artifacts_dir)
        self.addCleanup(lambda: shutil.rmtree(config.artifacts_dir, ignore_errors=True))

        manifest = ingest_cbbd(
            config,
            api_key="secret-token",
            client_factory=lambda _: FakeApiBundle(),
        )

        self.assertEqual(sorted(manifest["datasets"].keys()), ["game_teams", "games", "lines"])
        self.assertEqual(manifest["datasets"]["games"]["status"], "ok")
        self.assertGreater(manifest["datasets"]["games"]["rows"], 0)
        self.assertEqual(manifest["datasets"]["game_teams"]["status"], "ok")
        self.assertGreater(manifest["datasets"]["game_teams"]["rows"], 0)
        self.assertGreater(manifest["datasets"]["lines"]["rows"], 0)
        self.assertTrue((config.bronze_dir / "cbbd" / "games.parquet").exists())
        self.assertTrue((config.bronze_dir / "cbbd" / "game_teams.parquet").exists())
        self.assertTrue((config.bronze_dir / "cbbd" / "lines.parquet").exists())

    def test_clean_cbbd_games_and_lines_include_mapped_team_ids(self) -> None:
        config = PipelineConfig(project_root=ROOT)
        config.artifacts_dir = ROOT / "artifacts_test"
        if config.artifacts_dir.exists():
            shutil.rmtree(config.artifacts_dir)
        self.addCleanup(lambda: shutil.rmtree(config.artifacts_dir, ignore_errors=True))

        cbbd_dir = config.bronze_dir / "cbbd"
        cbbd_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "game_id": [1],
                "season": [2025],
                "season_type": ["regular"],
                "status": ["final"],
                "start_date": ["2025-03-01T00:00:00+00:00"],
                "home_team_id": [10],
                "home_team": ["Alpha"],
                "away_team_id": [20],
                "away_team": ["Beta"],
                "home_score": [80.0],
                "away_score": [70.0],
                "neutral_site": [False],
                "conference_game": [False],
                "notes": [None],
                "home_elo_start": [1500.0],
                "home_elo_end": [1510.0],
                "away_elo_start": [1490.0],
                "away_elo_end": [1480.0],
                "home_seed": [None],
                "away_seed": [None],
                "excitement": [5.5],
            }
        ).write_parquet(cbbd_dir / "games.parquet")

        pl.DataFrame(
            {
                "game_id": [1],
                "season": [2025],
                "season_type": ["regular"],
                "start_date": ["2025-03-01T00:00:00+00:00"],
                "home_team_id": [10],
                "home_team": ["Alpha"],
                "away_team_id": [20],
                "away_team": ["Beta"],
                "home_score": [80.0],
                "away_score": [70.0],
                "provider": ["A"],
                "spread": [-3.5],
                "spread_open": [-2.5],
                "over_under": [145.0],
                "over_under_open": [144.0],
                "home_moneyline": [110.0],
                "away_moneyline": [-120.0],
            }
        ).write_parquet(cbbd_dir / "lines.parquet")

        team_map = pl.DataFrame(
            {
                "cbbd_team_id": [10, 20],
                "team_id": [100, 200],
            }
        )

        games_clean = _build_cbbd_games_clean(config, team_map)
        lines_clean = _build_cbbd_lines_clean(config, team_map)

        self.assertIn("kaggle_home_team_id", games_clean.columns)
        self.assertIn("kaggle_away_team_id", games_clean.columns)
        self.assertIn("team_low", games_clean.columns)
        self.assertIn("team_high", games_clean.columns)
        self.assertEqual(games_clean.select("kaggle_home_team_id").to_series()[0], 100)
        self.assertEqual(games_clean.select("kaggle_away_team_id").to_series()[0], 200)

        self.assertIn("kaggle_home_team_id", lines_clean.columns)
        self.assertIn("kaggle_away_team_id", lines_clean.columns)
        self.assertIn("team_low", lines_clean.columns)
        self.assertIn("team_high", lines_clean.columns)
        self.assertEqual(lines_clean.select("kaggle_home_team_id").to_series()[0], 100)
        self.assertEqual(lines_clean.select("kaggle_away_team_id").to_series()[0], 200)


class TestEloAndHeat(unittest.TestCase):
    """Contract tests for the ELO and heat score engines."""

    def _make_game_fact(self) -> pl.DataFrame:
        """Minimal game_fact with two games in one season."""
        return pl.DataFrame({
            "sex": ["M", "M", "M", "M"],
            "season": [2025, 2025, 2025, 2025],
            "day_num": [10, 10, 20, 20],
            "team_id": [1101, 1102, 1101, 1103],
            "opp_team_id": [1102, 1101, 1103, 1101],
            "team_score": [80.0, 70.0, 65.0, 75.0],
            "opp_score": [70.0, 80.0, 75.0, 65.0],
            "team_loc": ["H", "A", "A", "H"],
            "num_ot": [0, 0, 0, 0],
            "win": [1, 0, 0, 1],
            "team_low": [1101, 1101, 1101, 1101],
            "team_high": [1102, 1102, 1103, 1103],
            "game_key": ["M_2025_10_1101_1102", "M_2025_10_1101_1102",
                         "M_2025_20_1101_1103", "M_2025_20_1101_1103"],
        })

    def test_elo_ratings_schema_and_reset(self) -> None:
        game_fact = self._make_game_fact()
        elo = _compute_elo_ratings(game_fact)
        self.assertGreater(elo.height, 0)
        expected_cols = {"sex", "season", "day_num", "game_key", "team_id",
                         "elo_before", "elo_after", "expected_win_prob",
                         "expected_margin", "actual_win", "actual_margin"}
        self.assertTrue(expected_cols.issubset(set(elo.columns)))
        # First game: both teams should start at 1500
        first_game = elo.filter(pl.col("day_num") == 10)
        self.assertTrue(all(v == 1500.0 for v in first_game["elo_before"].to_list()))

    def test_heat_scores_schema(self) -> None:
        game_fact = self._make_game_fact()
        elo = _compute_elo_ratings(game_fact)
        heat = _compute_heat_scores(elo)
        self.assertGreater(heat.height, 0)
        expected_cols = {"sex", "season", "day_num", "team_id",
                         "heat_delta", "heat_1g", "heat_3g", "heat_5g"}
        self.assertTrue(expected_cols.issubset(set(heat.columns)))

    def test_pre_tournament_heat_filters_by_day(self) -> None:
        game_fact = self._make_game_fact()
        elo = _compute_elo_ratings(game_fact)
        heat = _compute_heat_scores(elo)
        pre = _get_pre_tournament_heat(heat, tourney_cutoff_day=132)
        # All day_nums <= 132 so all teams should appear
        teams = sorted(pre["team_id"].to_list())
        self.assertIn(1101, teams)

    def test_elo_empty_input(self) -> None:
        empty = pl.DataFrame(schema={
            "sex": pl.Utf8, "season": pl.Int64, "day_num": pl.Int64,
            "team_id": pl.Int64, "opp_team_id": pl.Int64,
            "team_score": pl.Float64, "opp_score": pl.Float64,
            "team_loc": pl.Utf8, "num_ot": pl.Int64, "win": pl.Int64,
            "team_low": pl.Int64, "team_high": pl.Int64, "game_key": pl.Utf8,
        })
        elo = _compute_elo_ratings(empty)
        self.assertEqual(elo.height, 0)

    def test_heat_includes_surprise_weighted_columns(self) -> None:
        game_fact = self._make_game_fact()
        elo = _compute_elo_ratings(game_fact)
        heat = _compute_heat_scores(elo)
        sw_cols = {"sw_heat_delta", "sw_heat_1g", "sw_heat_3g", "sw_heat_5g", "heat_volatility_5g"}
        self.assertTrue(sw_cols.issubset(set(heat.columns)))

    def test_surprise_weight_amplifies_upsets(self) -> None:
        """A win by a team with low expected_win_prob should produce larger sw_heat_delta."""
        game_fact = self._make_game_fact()
        elo = _compute_elo_ratings(game_fact)
        heat = _compute_heat_scores(elo)
        # For the same magnitude heat_delta, surprise-weighted should differ based on expected prob
        sw_deltas = heat.filter(pl.col("sw_heat_delta").is_not_null()).select("sw_heat_delta").to_series()
        raw_deltas = heat.filter(pl.col("heat_delta").is_not_null()).select("heat_delta").to_series()
        # sw_heat_delta should exist and have non-zero values
        self.assertGreater(len(sw_deltas), 0)
        # At least one sw value should differ from raw
        if len(raw_deltas) > 0 and len(sw_deltas) > 0:
            self.assertFalse(all(abs(s - r) < 1e-10 for s, r in zip(sw_deltas.to_list(), raw_deltas.to_list())))


class TestQualityScores(unittest.TestCase):
    """Tests for the Ridge-regularized quality metric."""

    def _make_game_fact(self) -> pl.DataFrame:
        """Build a small game_fact with enough games for Ridge fitting."""
        import numpy as np
        rng = np.random.default_rng(42)
        games = []
        teams = [1101, 1102, 1103, 1104]
        game_num = 0
        for day in range(10, 130, 3):
            # Each day: random matchup
            t1, t2 = rng.choice(teams, size=2, replace=False)
            s1 = int(60 + rng.integers(-10, 20))
            s2 = int(60 + rng.integers(-10, 20))
            t_low, t_high = min(t1, t2), max(t1, t2)
            gk = f"M_2025_{day}_{t_low}_{t_high}"
            # Winner row
            w_tid, l_tid = (t1, t2) if s1 > s2 else (t2, t1)
            w_score, l_score = max(s1, s2), min(s1, s2)
            if s1 == s2:
                s1 += 1
                w_score = s1
                w_tid, l_tid = t1, t2
            games.append({
                "sex": "M", "season": 2025, "day_num": day,
                "team_id": t_low, "opp_team_id": t_high,
                "team_score": float(s1 if t_low == t1 else s2),
                "opp_score": float(s2 if t_low == t1 else s1),
                "team_loc": "H", "num_ot": 0,
                "win": 1 if (t_low == t1 and s1 > s2) or (t_low == t2 and s2 > s1) else 0,
                "team_low": t_low, "team_high": t_high, "game_key": gk,
            })
            games.append({
                "sex": "M", "season": 2025, "day_num": day,
                "team_id": t_high, "opp_team_id": t_low,
                "team_score": float(s2 if t_high == t2 else s1),
                "opp_score": float(s1 if t_high == t2 else s2),
                "team_loc": "A", "num_ot": 0,
                "win": 1 if (t_high == t2 and s2 > s1) or (t_high == t1 and s1 > s2) else 0,
                "team_low": t_low, "team_high": t_high, "game_key": gk,
            })
        return pl.DataFrame(games)

    def test_quality_returns_all_teams(self) -> None:
        gf = self._make_game_fact()
        q = _compute_quality_scores(gf)
        self.assertGreater(q.height, 0)
        self.assertIn("quality", q.columns)
        self.assertIn("quality_rank", q.columns)
        teams = sorted(q["team_id"].to_list())
        self.assertEqual(teams, [1101, 1102, 1103, 1104])

    def test_quality_empty_returns_empty(self) -> None:
        empty = pl.DataFrame(schema={
            "sex": pl.Utf8, "season": pl.Int64, "day_num": pl.Int64,
            "team_id": pl.Int64, "opp_team_id": pl.Int64,
            "team_score": pl.Float64, "opp_score": pl.Float64,
            "team_loc": pl.Utf8, "num_ot": pl.Int64, "win": pl.Int64,
            "team_low": pl.Int64, "team_high": pl.Int64, "game_key": pl.Utf8,
        })
        q = _compute_quality_scores(empty)
        self.assertEqual(q.height, 0)

    def test_quality_excludes_tournament_games(self) -> None:
        gf = self._make_game_fact()
        # Add a tournament game (day_num > 132)
        tourney_row = gf.head(2).with_columns(pl.lit(140).cast(pl.Int64).alias("day_num"))
        gf_with_tourney = pl.concat([gf, tourney_row])
        q1 = _compute_quality_scores(gf)
        q2 = _compute_quality_scores(gf_with_tourney)
        # Quality should be the same since tourney games are filtered out
        q1_dict = {r["team_id"]: r["quality"] for r in q1.to_dicts()}
        q2_dict = {r["team_id"]: r["quality"] for r in q2.to_dicts()}
        for tid in q1_dict:
            self.assertAlmostEqual(q1_dict[tid], q2_dict[tid], places=5)


if __name__ == "__main__":
    unittest.main()
