"""Run with: python -m unittest discover -s tests -p 'test_asr_*.py'"""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from asr_metrics import normalize, evaluate, corpus

class TestCer(unittest.TestCase):
    def test_unicode(self):
        self.assertEqual(normalize(" 前方，３００m！ "), "前方300m")
    def test_edits(self):
        self.assertEqual((evaluate("导航回家", "导航去家").s, evaluate("导航回家", "导航去家").cer), (1, 0.25))
        self.assertEqual((evaluate("导航回家", "导航家").s, evaluate("导航回家", "导航家").d), (0, 1))
        self.assertEqual(evaluate("导航家", "导航回家").i, 1)
        self.assertEqual(evaluate("ABC", "abc").cer, 0)
    def test_empty_hypothesis(self):
        c = evaluate("你好", "")
        self.assertEqual((c.s,c.d,c.i,c.n,c.cer), (0,2,0,2,1))
    def test_empty_reference_rejected(self):
        with self.assertRaises(ValueError):
            evaluate("。，", "你好")
    def test_corpus_weighted(self):
        a = evaluate("甲乙", "甲")
        b = evaluate("丙丁戊", "丙丁戊")
        total = corpus([a,b])
        self.assertEqual((total["deletions"], total["reference_chars"], total["cer"]), (1,5,0.2))
        self.assertIsNone(corpus([])["cer"])

if __name__ == "__main__":
    unittest.main()
