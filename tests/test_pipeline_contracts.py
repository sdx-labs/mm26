from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import unittest

from mm26.pipeline import _required_kaggle_schemas


class TestPipelineContracts(unittest.TestCase):
    def test_required_schema_keys_present(self) -> None:
        required = _required_kaggle_schemas()
        self.assertIn("MTeams", required)
        self.assertIn("SampleSubmissionStage2", required)
        self.assertIn("MRegularSeasonDetailedResults", required)


if __name__ == "__main__":
    unittest.main()
