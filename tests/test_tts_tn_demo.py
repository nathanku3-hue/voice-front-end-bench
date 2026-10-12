"""Round 3 deterministic TN regression, no ASR model or TTS dependencies."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from tts_tn_demo import DEMO_CASES, add_pause_tags, integer_to_hanzi, normalize_text, transform


class TextNormalizationTests(unittest.TestCase):
    def test_all_curated_cases(self):
        self.assertGreaterEqual(len(DEMO_CASES), 15)
        self.assertLessEqual(len(DEMO_CASES), 25)
        for source, normalized, pauses in DEMO_CASES:
            with self.subTest(source=source):
                self.assertEqual(transform(source), {
                    "input": source, "normalized": normalized, "pause_tags": pauses
                })

    def test_cardinal_boundaries(self):
        self.assertEqual(integer_to_hanzi(0), "零")
        self.assertEqual(integer_to_hanzi(10), "十")
        self.assertEqual(integer_to_hanzi(101), "一百零一")
        self.assertEqual(integer_to_hanzi(1001), "一千零一")
        self.assertEqual(integer_to_hanzi(2024), "两千零二十四")
        self.assertEqual(integer_to_hanzi(9999), "九千九百九十九")

    def test_time_boundary_and_leading_zero(self):
        self.assertEqual(normalize_text("00:00"), "零点整")
        self.assertEqual(normalize_text("23:59"), "二十三点五十九分")
        self.assertEqual(normalize_text("07:03"), "七点零三分")
        for invalid in ("24:00", "12:60", "99:99"):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    normalize_text(invalid)

    def test_decimals_zero_and_full_width(self):
        self.assertEqual(normalize_text("油耗0.5升"), "油耗零点五升")
        self.assertEqual(normalize_text("油耗0.5L"), "油耗0.5L")
        self.assertEqual(normalize_text("前方３００ｍ右转"), "前方三百米右转")
        self.assertEqual(normalize_text("车内温度２６℃"), "车内温度二十六摄氏度")
        self.assertEqual(normalize_text("停车费10.05元"), "停车费十点零五元")
        self.assertEqual(normalize_text("距离2m"), "距离两米")

    def test_pauses_are_explicit_and_deterministic(self):
        self.assertEqual(add_pause_tags("前方三百米右转，注意行人。"), "前方三百米 | 右转 | 注意行人 ||")
        self.assertEqual(add_pause_tags("导航回家"), "导航回家")

    def test_out_of_scope_numbers_fail_loudly(self):
        for bad in ("10000", "1.234", "-5℃"):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    normalize_text(bad)

    def test_cli_json_single(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/tts_tn_demo.py"),
                                 "--text", "前方300m右转", "--json"],
                                check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout)["pause_tags"], "前方三百米 | 右转")


if __name__ == "__main__":
    unittest.main()
