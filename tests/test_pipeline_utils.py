from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import unittest

from mm26.pipeline import (
    _infer_sex_from_team_id,
    _build_prob_lookup,
    _simulate_bracket,
    _train_model,
    _predict_with_model,
    normalize_name,
)


class TestPipelineUtils(unittest.TestCase):
    def test_normalize_name_basic(self) -> None:
        self.assertEqual(normalize_name("Texas A&M"), "texas a and m")
        self.assertEqual(normalize_name("  St. John's "), "st john s")

    def test_infer_sex_from_team_id(self) -> None:
        self.assertEqual(_infer_sex_from_team_id(1181), "M")
        self.assertEqual(_infer_sex_from_team_id(3181), "W")


class TestProbLookupAndSimulation(unittest.TestCase):
    def test_build_prob_lookup(self) -> None:
        import polars as pl
        sub = pl.DataFrame({
            "ID": ["2026_1101_1102", "2026_1101_1103"],
            "Pred": [0.6, 0.4],
        })
        lookup = _build_prob_lookup(sub)
        self.assertAlmostEqual(lookup[(1101, 1102)], 0.6)
        self.assertAlmostEqual(lookup[(1101, 1103)], 0.4)

    def test_simulate_bracket_empty_seeds(self) -> None:
        import polars as pl
        seeds = pl.DataFrame(schema={"Season": pl.Int64, "Seed": pl.Utf8, "TeamID": pl.Int64})
        slots = pl.DataFrame(schema={"Season": pl.Int64, "Slot": pl.Utf8, "StrongSeed": pl.Utf8, "WeakSeed": pl.Utf8})
        result = _simulate_bracket(seeds, slots, {}, n_sims=10)
        self.assertEqual(result, {})

    def test_simulate_bracket_simple(self) -> None:
        import polars as pl
        seeds = pl.DataFrame({
            "Season": [2026, 2026],
            "Seed": ["W01", "W02"],
            "TeamID": [1101, 1102],
        })
        slots = pl.DataFrame({
            "Season": [2026],
            "Slot": ["R1W1"],
            "StrongSeed": ["W01"],
            "WeakSeed": ["W02"],
        })
        lookup = {(1101, 1102): 0.9}
        result = _simulate_bracket(seeds, slots, lookup, n_sims=10_000)
        self.assertIn((1101, 1102), result)
        # With 90% prob, team_low should win ~90% of sims
        self.assertGreater(result[(1101, 1102)], 0.8)
        self.assertLess(result[(1101, 1102)], 1.0)


class TestTrainAndPredict(unittest.TestCase):
    def test_train_model_returns_model(self) -> None:
        import polars as pl
        df = pl.DataFrame({
            "f1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "target_low_wins": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        })
        model = _train_model(df, ["f1"], n_estimators=10)
        self.assertIsNotNone(model)

    def test_predict_with_model_none_returns_half(self) -> None:
        import polars as pl
        df = pl.DataFrame({
            "f1": [0.1, 0.2],
        })
        preds = _predict_with_model(None, df, ["f1"])
        self.assertEqual(preds.to_list(), [0.5, 0.5])

    def test_predict_with_model_clips(self) -> None:
        import polars as pl
        df = pl.DataFrame({
            "f1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "target_low_wins": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        })
        model = _train_model(df, ["f1"], n_estimators=10)
        # Without seed columns, default clip bounds [0.025, 0.975] apply
        preds = _predict_with_model(model, df, ["f1"])
        self.assertTrue(all(0.025 <= v <= 0.975 for v in preds.to_list()))

    def test_predict_with_model_dynamic_clip_wide(self) -> None:
        import polars as pl
        df = pl.DataFrame({
            "f1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "target_low_wins": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "seed_low": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "seed_high": [16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
        })
        model = _train_model(df, ["f1"], n_estimators=10)
        # With 1v16, clip bounds widen to [0.005, 0.995]
        preds = _predict_with_model(model, df, ["f1"])
        self.assertTrue(all(0.005 <= v <= 0.995 for v in preds.to_list()))


if __name__ == "__main__":
    unittest.main()
