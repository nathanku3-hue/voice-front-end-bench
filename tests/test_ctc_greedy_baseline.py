"""CTC baseline contract tests; synthetic data are not benchmark measurements."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ctc_greedy_baseline import greedy_ids, validate_log_probs, run


class CtcGreedyTests(unittest.TestCase):
    def test_ctc_collapse_before_remove_blank(self):
        x = np.log(np.array([[.05, .95, 0.0], [.05, .95, 0.0], [.05, .95, 0.0]]) + 1e-20)
        self.assertEqual(greedy_ids(x, 0), [1])
        x = np.log(np.array([[.05, .95, 0.0], [.95, .05, 0.0], [.05, .95, 0.0]]) + 1e-20)
        self.assertEqual(greedy_ids(x, 0), [1, 1])

    def test_invalid_contract(self):
        with self.assertRaises(ValueError):
            validate_log_probs(np.array([[1.0, 0.0]]), 2)
        with self.assertRaises(ValueError):
            validate_log_probs(np.array([[float("nan"), 0.0]]), 2)
        with self.assertRaises(ValueError):
            validate_log_probs(np.zeros((2, 3)), 2)

    def test_end_to_end_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p / "tokens.txt").write_text("<blk>\n你\n好\n", encoding="utf-8")
            (p / "model.json").write_text(json.dumps(dict(model_name="fixture-only",
                model_sha256="synthetic", ctc_head="synthetic", frame_stride_ms=20)))
            paths = [[1, 0, 2], [1, 1, 0]]
            for i, ids in enumerate(paths):
                probs = np.full((len(ids), 3), 0.01)
                for j, id_ in enumerate(ids):
                    probs[j, id_] = 0.98
                np.save(p / f"{i}.npy", np.log(probs))
            items = [
                dict(id="one", reference="你好", log_probs="0.npy"),
                dict(id="two", reference="你好", log_probs="1.npy"),
            ]
            (p / "manifest.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))
            result = run(p / "manifest.jsonl", p / "tokens.txt", 0, p / "out.json", p / "model.json")
            self.assertEqual(result["corpus"]["reference_chars"], 4)
            self.assertEqual(result["corpus"]["deletions"], 1)
            self.assertEqual(result["corpus"]["cer"], .25)
            self.assertTrue((p / "out.json").is_file())

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p / "tokens").write_text("<blk>\n你\n")
            (p / "model.json").write_text(json.dumps(dict(model_name="fixture",
                model_sha256="synthetic", ctc_head="synthetic", frame_stride_ms=20)))
            (p / "manifest.jsonl").write_text(json.dumps(dict(id="one", reference="你", log_probs="../out.npy")) + "\n")
            with self.assertRaises(ValueError):
                run(p / "manifest.jsonl", p / "tokens", 0, p / "out.json", p / "model.json")


if __name__ == "__main__":
    unittest.main()
