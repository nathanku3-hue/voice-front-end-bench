"""Sweep shape, separation, and guarded merging without model downloads."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_ctc_dev_sweep import BEAMS, THRESHOLDS, STRATEGIES, grid, config_id, merge_parts


class DevSweepContractTests(unittest.TestCase):
    def test_exact_36_unique_conditions_and_independent_greedy(self):
        names = [config_id(w, strategy, threshold)
                 for w in BEAMS for strategy, threshold in grid(w)]
        self.assertEqual(len(names), 36)
        self.assertEqual(len(set(names)), 36)
        self.assertEqual(sum("unpruned" in name for name in names), 4)
        self.assertEqual(sum("forced_blank" in name for name in names), 16)
        self.assertEqual(sum("hard_skip" in name for name in names), 16)
        self.assertNotIn("framewise_greedy", names)

    @staticmethod
    def fake_part(width):
        configs = []
        for strategy, tau in grid(width):
            configs.append(dict(
                id=config_id(width, strategy, tau), strategy=strategy,
                beam_width=width, blank_threshold=tau,
                utterances=[dict(id=f"dev{i}") for i in range(30)],
                corpus=dict(labeled_utterances=30, reference_chars=442, cer=.05),
                timing=dict(decoder_sum_of_utterance_medians_ms=10.0),
                nonblank_arcs_expanded=1000,
            ))
        return dict(
            status="measured_from_dev_only", width=width,
            frozen_input_contract=dict(reference_chars=442, held_out_test_count=50),
            timing_protocol=dict(clock="perf_counter_ns", repeats=3),
            framewise_greedy=(
                dict(corpus=dict(labeled_utterances=30, cer=.04),
                     timing=dict(decoder_sum_of_utterance_medians_ms=.5))
                if width == 1 else None
            ),
            configs=configs,
        )

    def test_merge_requires_all_widths_and_exact_grid(self):
        parts = [self.fake_part(w) for w in BEAMS]
        merged = merge_parts(parts)
        self.assertEqual(merged["grid"]["measured_configurations"], 36)
        self.assertEqual(merged["hypothesis_selection_scope"],
                         "30 Dev only; no held-out Test CER or hypotheses generated")
        self.assertAlmostEqual(merged["configurations"][0]["cer_delta_vs_framewise_greedy"], .01)
        self.assertEqual(len({r["id"] for r in merged["configurations"]}), 36)
        for bad in [parts[:3], parts[:3] + [self.fake_part(1)]]:
            with self.assertRaises(ValueError):
                merge_parts(bad)
        broken = [self.fake_part(w) for w in BEAMS]
        broken[0]["configs"].pop()
        with self.assertRaises(ValueError):
            merge_parts(broken)
        broken = [self.fake_part(w) for w in BEAMS]
        broken[1]["frozen_input_contract"]["reference_chars"] = 123
        with self.assertRaises(ValueError):
            merge_parts(broken)


if __name__ == "__main__":
    unittest.main()
