"""Pure-function Round 3 invariants. Does not access held-out Test logits."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_ctc_test_eval import (
    CONDITIONS, CSV_COLUMNS, check_output_available, exclusive_claim,
    pareto_ids, project_csv, select_badcases, verify_condition_contract, execute,
)


class Round3ContractTests(unittest.TestCase):
    def test_exact_four_locked_conditions(self):
        verify_condition_contract()
        self.assertEqual(len(CONDITIONS), 4)
        self.assertEqual(len({c[0] for c in CONDITIONS}), 4)
        self.assertEqual(sum(c[2] == "forced_blank" for c in CONDITIONS), 2)
        self.assertNotIn("hard_skip", [c[2] for c in CONDITIONS])
        self.assertEqual([c[1] for c in CONDITIONS], [None, 16, 16, 4])

    def test_pareto_allows_single_and_multiple_winners(self):
        def c(cid, t, cer):
            return dict(id=cid, timing=dict(decoder_mean_ms=t), corpus=dict(cer=cer))
        one = [c("greedy", .02, .03), c("a", 5, .03), c("b", 8, .04)]
        self.assertEqual(pareto_ids(one), ["greedy"])
        tradeoff = [c("greedy", .02, .05), c("a", 5, .04), c("b", 8, .05)]
        self.assertEqual(pareto_ids(tradeoff), ["greedy", "a"])
        tie = [c("greedy", .02, .05), c("a", .02, .05)]
        self.assertEqual(pareto_ids(tie), ["greedy", "a"])

    def test_fixed_badcase_selection_and_evidence(self):
        def u(uid, s=0, d=0, i=0):
            return dict(id=uid, reference="ABC", hypothesis="ABC", tokens=[4],
                        speaker_id="S1", audio_relpath="audio/clip.wav", duration_s=1.0, frames=12,
                        avg_frame_max_p=.95, avg_blank_p=.8,
                        fraction_blank_p_ge_095=.7, s=s, d=d, i=i,
                        utf8_replacement_chars=0)
        greedy = [u("case_b", s=2), u("case_a", d=2), u("case_c", i=1), u("perfect")]
        confs = [dict(id=cid, utterances=[dict(r) for r in greedy]) for cid, *_ in CONDITIONS]
        selected = select_badcases(confs)
        self.assertEqual([r["id"] for r in selected], ["case_a", "case_b", "case_c"])
        self.assertEqual(set(selected[0]["hypotheses"]), {c[0] for c in CONDITIONS})

    def test_claim_is_exclusive(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            protocol = root / "protocol.md"
            protocol.write_text("four conditions", encoding="utf-8")
            path = root / "claim.json"
            check_output_available(path, root / "report.json", root / "report.csv")
            first = exclusive_claim(path, {"example": "synthetic"}, protocol)
            self.assertEqual(first["status"], "CLAIMED_BEFORE_FIRST_TEST_DECODE")
            with self.assertRaises(FileExistsError):
                exclusive_claim(path, {}, protocol)
            with self.assertRaises(FileExistsError):
                check_output_available(path, root / "report.json", root / "report.csv")
            self.assertEqual(json.loads(path.read_text())["conditions"],
                             [c[0] for c in CONDITIONS])

    def test_preflight_cannot_load_arrays_or_create_claim(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            args = SimpleNamespace(
                preflight=True, execute=False, model_dir=root,
                claim=root / "claim.json", json_output=root / "result.json",
                csv_output=root / "result.csv", test_root=root,
                data_root=root, inventory=root / "inventory.json",
                dev_report=root / "dev.json",
            )
            with patch("run_ctc_test_eval.prepare_inputs",
                       return_value=([object()] * 50, [], {}, {"synthetic": True})) as mock:
                execute(args)
            self.assertFalse(args.claim.exists())
            self.assertFalse(args.json_output.exists())
            self.assertFalse(args.csv_output.exists())
            self.assertIs(mock.call_args.kwargs["load_arrays"], False)

    def test_summary_projection_has_four_rows_and_fixed_columns(self):
        confs = []
        for cid, beam, strategy, threshold in CONDITIONS:
            confs.append(dict(id=cid, beam_width=beam, strategy=strategy,
                              blank_threshold=threshold,
                              corpus=dict(cer=.03, substitutions=3, deletions=0,
                                          insertions=0, reference_chars=100),
                              timing=dict(decoder_mean_ms=2., decoder_median_ms=2.,
                                          decoder_p95_ms=2.5,
                                          decoder_sum_of_utterance_medians_ms=100),
                              utf8_replacement_chars=0, triggered_blank_frame_rate=0,
                              forced_blank_frame_rate=0, actual_hard_skip_frame_rate=0,
                              hypotheses_different_from_greedy=0,
                              tokens_different_from_greedy=0))
        rows = project_csv(dict(configurations=confs))
        self.assertEqual(len(rows), 4)
        self.assertEqual(tuple(rows[0]), CSV_COLUMNS)
        self.assertEqual(rows[2]["blank_threshold"], .95)


if __name__ == "__main__":
    unittest.main()
