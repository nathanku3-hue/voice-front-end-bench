"""Reject provenance, edit-count and UTF-8 mismatches in Round 2 audit."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ctc_round2_audit import check_records


class Round2EvidenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.source = {
            "dev_a": {"id": "dev_a", "reference": "你好", "speaker_id": "S1"},
            "dev_b": {"id": "dev_b", "reference": "天地", "speaker_id": "S2"},
        }
        self.rows = [
            {"id": "dev_a", "reference": "你好", "hypothesis": "你好",
             "speaker_id": "S1", "frames": 5, "decoder_median_ms": .1,
             "utf8_replacement_chars": 0,
             "s": 0, "d": 0, "i": 0, "n": 2, "cer": 0.0},
            {"id": "dev_b", "reference": "天地", "hypothesis": "天�",
             "speaker_id": "S2", "frames": 7, "decoder_median_ms": .2,
             "utf8_replacement_chars": 1,
             "s": 1, "d": 0, "i": 0, "n": 2, "cer": .5},
        ]

    def check(self, rows):
        return check_records(rows, self.source, total_chars=4, total_frames=12)

    def test_valid_aggregates_and_replacements(self):
        summary, by_id = self.check(self.rows)
        self.assertEqual((summary["substitutions"], summary["reference_chars"]), (1, 4))
        self.assertEqual(set(by_id), {"dev_a", "dev_b"})

    def test_reject_modified_edits(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["s"] = 0
        with self.assertRaises(ValueError):
            self.check(rows)

    def test_reject_erased_replacement_diagnostic(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["utf8_replacement_chars"] = 0
        with self.assertRaises(ValueError):
            self.check(rows)

    def test_reject_reference_and_id_leak(self):
        for mutate in ("reference", "id", "speaker_id"):
            rows = copy.deepcopy(self.rows)
            rows[1][mutate] = "not-dev"
            with self.subTest(field=mutate), self.assertRaises(ValueError):
                self.check(rows)

    def test_reject_frame_or_latency_drift(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["frames"] = 123
        with self.assertRaises(ValueError):
            self.check(rows)
        rows = copy.deepcopy(self.rows)
        rows[0]["decoder_median_ms"] = 0.0
        with self.assertRaises(ValueError):
            self.check(rows)


if __name__ == "__main__":
    unittest.main()
