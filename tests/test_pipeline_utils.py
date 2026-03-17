from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import unittest

from mm26.pipeline import _infer_sex_from_team_id, normalize_name


class TestPipelineUtils(unittest.TestCase):
    def test_normalize_name_basic(self) -> None:
        self.assertEqual(normalize_name("Texas A&M"), "texas a and m")
        self.assertEqual(normalize_name("  St. John's "), "st john s")

    def test_infer_sex_from_team_id(self) -> None:
        self.assertEqual(_infer_sex_from_team_id(1181), "M")
        self.assertEqual(_infer_sex_from_team_id(3181), "W")


if __name__ == "__main__":
    unittest.main()
