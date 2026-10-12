"""Offline P3 regression checks for cost-envelope math and paired 2x2 reports."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_sla_tradeoff import build_report, cost, envelope, winners
from run_frontend_sensitivity import CONDITIONS, MODES, summarize


class CostModelTests(unittest.TestCase):
    def test_lambdas_units_and_observed_switches(self):
        report = build_report(ROOT / "results/endpoint_sweep", [0, 20, 400, 500, 800])
        rec = report["observations"]
        self.assertEqual(len(rec), 5)
        self.assertEqual([r["deletions"] for r in rec], [244, 101, 66, 36, 7])
        self.assertEqual([w["winners_threshold_ms"] for w in report["scenarios"]],
                         [[300], [300], [500], [900], [1200]])
        intervals = report["open_intervals"]
        self.assertEqual([r["winner_threshold_ms"] for r in intervals],
                         [[300], [500], [900], [1200]])
        self.assertAlmostEqual(intervals[1]["lambda_min_exclusive"], 332.531, delta=.002)
        self.assertAlmostEqual(intervals[2]["lambda_min_exclusive"], 494.344, delta=.002)
        self.assertAlmostEqual(intervals[3]["lambda_min_exclusive"], 758.866, delta=.002)
        self.assertAlmostEqual(
            (rec[4]["mean_endpoint_stream_time_ms"] - rec[2]["mean_endpoint_stream_time_ms"]) /
            (rec[2]["deletions_per_utterance"] - rec[4]["deletions_per_utterance"]),
            599.617, delta=.002,
        )

    def test_cost_dimensions_and_invalid_lambda(self):
        row = {"mean_endpoint_stream_time_ms": 1000, "deletions_per_utterance": 2}
        self.assertEqual(cost(row, 10), 1020)
        with self.assertRaises(ValueError):
            cost(row, -1)

    def test_dominated_configuration(self):
        rows = [
            {"threshold_ms": 1, "mean_endpoint_stream_time_ms": 0, "deletions_per_utterance": 10},
            {"threshold_ms": 2, "mean_endpoint_stream_time_ms": 100, "deletions_per_utterance": 9},
            {"threshold_ms": 3, "mean_endpoint_stream_time_ms": 110, "deletions_per_utterance": 7},
        ]
        self.assertEqual(winners(rows, 0), [1])
        self.assertEqual(winners(rows, 100), [3])
        self.assertNotIn(2, [r["winner_threshold_ms"][0] for r in envelope(rows)])


class FrontendSensitivityTests(unittest.TestCase):
    def test_native_cli_exposes_options_and_rejects_invalid_cutoff(self):
        script = str(ROOT / "scripts/asr_stream_bench.py")
        help_result = subprocess.run([sys.executable, script, "--help"],
                                     capture_output=True, text=True, check=True)
        self.assertIn("--snip-edges", help_result.stdout)
        self.assertIn("--high-freq", help_result.stdout)
        invalid = subprocess.run([sys.executable, script, "--wav", "nonexistent.wav",
                                  "--model-dir", ".", "--high-freq", "nan"],
                                 capture_output=True, text=True)
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("must be finite", invalid.stderr)


    def _files(self, root: Path):
        result = {}
        for mode in MODES:
            result[mode] = {}
            for variant, snip, high in CONDITIONS:
                output = root / f"{mode}_{variant}.json"
                info = {"model": {"file_sha256": {"encoder": "fixed"}},
                        "manifest_sha256": "manifest_fixed",
                        "config": {
                            "chunk_ms": 160, "tail_ms": 2500, "threads": 2,
                            "rule2_trailing_silence_ms": 1200,
                            "endpoint_enabled": mode == "rule2_1200",
                            "snip_edges": snip, "high_freq": high,
                        },
                        "utterances": [
                            {"id": f"utt{i}", "wav_sha256": f"hash{i}",
                             "speaker_id": f"sp{i%11}", "reference": "甲" * (14 if i < 13 else 13),
                             "reference_chars": 14 if i < 13 else 13,
                             "s": 0, "d": 1 if (i == 0 and variant == "native") else 0,
                             "i": 0, "cer": 0.0,
                             "hypothesis": "甲" if variant == "native" else "乙",
                             "endpoint_detected": mode == "rule2_1200",
                             "endpoint_audio_time_s": 3 if mode == "rule2_1200" else None}
                            for i in range(30)
                        ]}
                # 13*14 + 17*13 = 403 chars
                output.write_text(json.dumps(info), encoding="utf-8")
                result[mode][variant] = output
        return result

    def test_2x2_report_and_strict_custody_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._files(Path(tmp))
            report = summarize(paths)
            self.assertEqual(len(report["modes"]), 2)
            self.assertEqual(report["modes"]["rule2_1200"]["conditions"]["native"]["d"], 1)
            self.assertEqual(report["modes"]["rule2_1200"]["conditions"]
                             ["matched_cpp_options"]["changed_transcripts_vs_native"], 30)
            with paths["no_endpoint"]["cutoff_only"].open(encoding="utf-8") as fp:
                data = json.load(fp)
            data["config"]["chunk_ms"] = 120
            paths["no_endpoint"]["cutoff_only"].write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Configuration mismatch"):
                summarize(paths)
    def test_real_p3_evidence_roundtrip(self):
        path = ROOT / "results/p3_frontend_runs"
        mapping = {
            mode: {name: path / f"{mode}_{name}.json" for name, _, _ in CONDITIONS}
            for mode in MODES
        }
        recalculated = summarize(mapping)
        committed = json.loads((ROOT / "results/p3_frontend_summary.json")
                               .read_text(encoding="utf-8"))
        for mode in MODES:
            self.assertEqual(recalculated["modes"][mode]["conditions"],
                             committed["modes"][mode]["conditions"])
        no_endpoint = recalculated["modes"]["no_endpoint"]["conditions"]
        with_endpoint = recalculated["modes"]["rule2_1200"]["conditions"]
        self.assertEqual((no_endpoint["native"]["s"], no_endpoint["native"]["d"],
                          no_endpoint["native"]["i"]), (46, 3, 2))
        self.assertEqual((with_endpoint["native"]["s"], with_endpoint["native"]["d"],
                          with_endpoint["native"]["i"]), (46, 7, 2))
        self.assertEqual(with_endpoint["matched_cpp_options"]
                         ["changed_transcripts_vs_native"], 9)
        self.assertGreater(with_endpoint["matched_cpp_options"]["cer"],
                           with_endpoint["native"]["cer"])
        confidence = with_endpoint["matched_cpp_options"]["cer_delta_cluster_bootstrap95"]
        self.assertEqual(confidence["speakers"], 11)
        self.assertLess(confidence["approx_95_ci_pp"][0], 0)
        self.assertGreater(confidence["approx_95_ci_pp"][1], 0)


if __name__ == "__main__":
    unittest.main()
