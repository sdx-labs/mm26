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
        self.assertEqual(split["train_start_season"], 2016)
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


if __name__ == "__main__":
    unittest.main()
