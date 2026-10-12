"""Offline, model-free tests for paired stream-time endpoint accounting."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from endpoint_timing_audit import (THRESHOLDS, load_runs, paired_measurements,
                                   paired_summary, quantile, audit)


class EndpointTimingTests(unittest.TestCase):
    def test_quantile_and_constant_difference(self):
        self.assertEqual(quantile([0, 100, 200], .5), 100)
        rows = [
            dict(id="x", speaker_id="a", delta_ms=320, deletion_reduction=2,
                 error_reduction=1, reference_chars=5),
            dict(id="y", speaker_id="b", delta_ms=320, deletion_reduction=1,
                 error_reduction=1, reference_chars=5),
        ]
        summary = paired_summary(rows, reps=100, seed=1)
        self.assertAlmostEqual(summary["mean_delta_ms"], 320)
        self.assertEqual(summary["mean_delta_cluster_bootstrap95_ms"], [320, 320])
        self.assertEqual(summary["deletion_reduction"], 3)
        self.assertEqual(summary["positive_n"], 2)

    def _write_runs(self, target: Path):
        for t in THRESHOLDS:
            rows = [
                {"id": "a", "speaker_id": "sp1", "wav_sha256": "x",
                 "reference": "甲", "reference_chars": 1, "speech_end_s": 1.0,
                 "audio_duration_s": 2.0, "endpoint_detected": True,
                 "endpoint_audio_time_s": 3.0 + t / 1000,
                 "s": 0, "d": int(t == 300), "i": 0},
                {"id": "b", "speaker_id": "sp2", "wav_sha256": "y",
                 "reference": "乙", "reference_chars": 1, "speech_end_s": 1.0,
                 "audio_duration_s": 2.0, "endpoint_detected": True,
                 "endpoint_audio_time_s": 3.5 + t / 1000,
                 "s": 0, "d": int(t == 300), "i": 0},
            ]
            payload = {
                "model": {"file_sha256": {"encoder": "fixed"}},
                "config": {"rule2_trailing_silence_ms": t, "chunk_ms": 160,
                           "threads": 2}, "utterances": rows,
            }
            (target / f"rule2_{t}ms.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_audit_and_hypothesis_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_runs(root)
            report = audit(root, reps=100)
            self.assertAlmostEqual(report["pairs"]["700_to_1200"]["mean_delta_ms"], 500)
            self.assertEqual(report["pairs"]["300_to_700"]["deletion_reduction"], 2)
            runs = load_runs(root)
            self.assertEqual(len(paired_measurements(runs, 700, 1200)), 2)
            corrupted = json.loads((root / "rule2_900ms.json").read_text(encoding="utf-8"))
            corrupted["config"]["chunk_ms"] = 40
            (root / "rule2_900ms.json").write_text(json.dumps(corrupted), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Non-threshold"):
                audit(root, reps=100)

    def test_real_sweep_invariants(self):
        root = ROOT / "results/endpoint_sweep"
        report = audit(root, reps=100)
        self.assertEqual(report["pairs"]["700_to_1200"]["n"], 30)
        self.assertEqual(report["pairs"]["700_to_1200"]["speakers"], 11)
        self.assertEqual(report["pairs"]["300_to_1200"]["deletion_reduction"], 237)
        self.assertGreater(report["pairs"]["700_to_1200"]["mean_delta_ms"], 500)


if __name__ == "__main__":
    unittest.main()
