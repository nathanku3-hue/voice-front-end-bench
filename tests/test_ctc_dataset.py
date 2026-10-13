"""Offline-only checks for CTC evaluation-data selection."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_ctc_eval import freeze_split


class CtcDatasetTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(id=f"U{i:03d}", speaker_id=f"S{i%8:02d}", reference="你好",
                          wav=f"audio/U{i:03d}.wav") for i in range(80)]

    def test_freeze_is_reproducible_and_disjoint(self):
        dev, test = freeze_split(self.rows, dev=30, test=50, seed=20261012)
        self.assertEqual((len(dev), len(test)), (30, 50))
        self.assertTrue(all(row["split"] == "dev" for row in dev))
        self.assertTrue(all(row["split"] == "test" for row in test))
        self.assertFalse({x["id"] for x in dev} & {x["id"] for x in test})
        other_dev, other_test = freeze_split(list(reversed(self.rows)), dev=30, test=50, seed=20261012)
        self.assertEqual(dev, other_dev)
        self.assertEqual(test, other_test)

    def test_reject_duplicates_or_missing_rows(self):
        with self.assertRaises(ValueError):
            freeze_split(self.rows[:-1], dev=30, test=50)
        repeated = list(self.rows)
        repeated[0] = dict(repeated[1])
        with self.assertRaises(ValueError):
            freeze_split(repeated, dev=30, test=50)


if __name__ == "__main__":
    unittest.main()
