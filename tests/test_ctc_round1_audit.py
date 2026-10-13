"""Pure contract checks for the frozen CTC Round-1 evidence auditor."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ctc_round1_audit import unique_ids


class CtcRound1AuditTests(unittest.TestCase):
    def test_unique_ids_return_exact_set(self):
        rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        self.assertEqual(unique_ids(rows), {"a", "b", "c"})

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            unique_ids([{"id": "a"}, {"id": "a"}])


if __name__ == "__main__":
    unittest.main()
