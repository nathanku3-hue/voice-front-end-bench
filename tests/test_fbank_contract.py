"""Numeric comparison tests; independent of sherpa binaries and ONNX downloads."""
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_fbank_contract import compare


class FeatureContractTests(unittest.TestCase):
    def test_identical_features(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        outcome = compare(a, a.copy())
        self.assertEqual(outcome["frames_compared"], 2)
        self.assertEqual(outcome["feature_dim"], 2)
        self.assertEqual(outcome["mae"], 0)
        self.assertAlmostEqual(outcome["flattened_feature_cosine"], 1.0)

    def test_nonidentical_feature_errors_are_reported(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[2.0, 2.0], [3.0, 4.0]])
        outcome = compare(a, b)
        self.assertEqual(outcome["max_abs"], 1.0)
        self.assertAlmostEqual(outcome["mae"], 0.25)
        with self.assertRaisesRegex(ValueError, "unequal"):
            compare(a, b[:1])


if __name__ == "__main__":
    unittest.main()
