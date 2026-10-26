"""Check that the flow-control and beam-width artifacts back the reported claims.

These tests do not re-measure anything. They recompute every ratio and every
selection decision from the published artifacts, so a stale or hand-edited JSON
cannot keep an unsupported resume claim alive.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOAK = ROOT / "results/backpressure_soak.json"
PARETO = ROOT / "results/ctc_beam_pareto.json"
BASELINE = ROOT / "results/ctc_baseline_dev.json"

sys.path.insert(0, str(ROOT / "scripts"))


class BackpressureSoakTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SOAK.is_file():
            raise unittest.SkipTest("run scripts/run_backpressure_soak.py first")
        cls.report = json.loads(SOAK.read_text(encoding="utf-8"))

    def test_scope_and_protocol(self):
        self.assertEqual(self.report["status"], "MEASURED_BACKPRESSURE_SOAK")
        self.assertEqual(self.report["method"]["consumer_rate_ratio"], 0.5)
        self.assertAlmostEqual(self.report["headline"]["audio_seconds"], 3600.0)

    def test_every_frame_compared_and_bit_exact(self):
        headline = self.report["headline"]
        self.assertEqual(headline["max_abs_diff"], 0.0)
        self.assertEqual(headline["mean_abs_diff"], 0.0)
        # 80 mel bins per frame; a smaller count means the audit skipped frames.
        self.assertEqual(headline["compared_values"], headline["frames"] * 80)

    def test_watermarks_are_bounded_by_configuration(self):
        for case in [self.report["headline"], *self.report["capacity_sweep"]]:
            capacity = case["config"]["capacity_frames"]
            with self.subTest(capacity=capacity):
                self.assertLessEqual(case["max_frames_held"], capacity)
                self.assertLessEqual(case["max_pcm_backlog_samples"],
                                     case["config"]["chunk_samples"])

    def test_watermark_tracks_capacity_not_elapsed_time(self):
        for case in self.report["capacity_sweep"]:
            with self.subTest(capacity=case["config"]["capacity_frames"]):
                # A ring that never fills would mean the audit never applied
                # backpressure; a ring that overflows would mean it lost data.
                self.assertEqual(case["max_frames_held"], case["config"]["capacity_frames"])

    def test_no_heap_or_rss_growth(self):
        tolerance = self.report["method"]["heap_tolerance_bytes"]
        for case in [self.report["headline"], *self.report["capacity_sweep"]]:
            with self.subTest(capacity=case["config"]["capacity_frames"]):
                growth = case["heap_live_bytes"]["end"] - case["heap_live_bytes"]["warm"]
                self.assertLessEqual(growth, tolerance)
                rss_growth = case["rss_bytes"]["end"] - case["rss_bytes"]["warm"]
                self.assertLessEqual(rss_growth, 8 * 1024 * 1024)

    def test_backpressure_actually_engaged(self):
        headline = self.report["headline"]
        self.assertGreater(headline["pause_events"], 0)
        self.assertEqual(headline["pause_events"], headline["resume_events"])
        # Half the ticks must be blocked for a 50% consumer; a rate-matched
        # harness would show ~0 and would not test backpressure at all.
        self.assertGreater(headline["derived"]["pause_fraction"], 0.4)
        self.assertLess(headline["derived"]["pause_fraction"], 0.6)
        self.assertGreater(headline["frames_after_resume"], 0)


class BeamParetoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PARETO.is_file():
            raise unittest.SkipTest("run scripts/sweep_beam_pareto.py first")
        cls.report = json.loads(PARETO.read_text(encoding="utf-8"))
        cls.rows = cls.report["results"]

    def test_widths_and_scope(self):
        self.assertEqual(self.report["status"], "BEAM_WIDTH_PARETO_DEV_ONLY")
        self.assertEqual([r["beam_width"] for r in self.rows], [1, 2, 4, 8, 16, 32])
        self.assertEqual(self.report["provenance"]["dev_count"], 30)
        self.assertEqual(self.report["provenance"]["reference_chars"], 442)

    def test_cer_matches_the_frozen_greedy_baseline(self):
        frozen = json.loads(BASELINE.read_text(encoding="utf-8"))["corpus"]
        for row in self.rows:
            with self.subTest(width=row["beam_width"]):
                self.assertAlmostEqual(row["cer"], frozen["cer"], places=12)
                self.assertEqual(row["reference_chars"], frozen["reference_chars"])

    def test_every_width_produces_the_same_tokens(self):
        # This is the reason the CER curve is flat: not detokenization aliasing.
        for row in self.rows:
            with self.subTest(width=row["beam_width"]):
                self.assertEqual(row["hypotheses_differing_from_widest"], 0)
                self.assertEqual(row["hypotheses_differing_from_greedy"], 0)

    def test_latency_is_monotonic_in_width(self):
        latencies = [row["decoder_mean_ms"] for row in self.rows]
        self.assertEqual(latencies, sorted(latencies))
        self.assertGreater(latencies[-1] / latencies[0], 2.0)

    def test_reported_speedups_recompute(self):
        widest = max(self.rows, key=lambda r: r["beam_width"])
        for row in self.rows:
            with self.subTest(width=row["beam_width"]):
                expected = widest["decoder_sum_ms"] / row["decoder_sum_ms"]
                self.assertAlmostEqual(row["speedup_vs_widest_beam"], expected, places=12)

    def test_knee_follows_the_declared_rule(self):
        from sweep_beam_pareto import select_knee

        rule = self.report["selection_rule"]
        chosen = select_knee(self.rows, rule["cer_tolerance_absolute"])
        self.assertIsNotNone(chosen)
        self.assertEqual(self.report["knee"]["beam_width"], chosen["beam_width"])

    def test_posterior_concentration_explains_flat_cer(self):
        stats = self.report["posterior_concentration"]
        self.assertGreater(stats["median_top1_probability"], 0.99)
        self.assertGreater(stats["fraction_frames_margin_above_0_9"], 0.8)


if __name__ == "__main__":
    unittest.main()
