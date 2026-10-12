"""Check that portfolio claims and relative Markdown links are backed by files/data."""
import json
import re
import unittest
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOCS = (
    ROOT / "README.md",
    ROOT / "CV.md",
    ROOT / "RESUME_BULLETS.md",
    ROOT / "docs/PROJECT_PLAN.md",
    ROOT / "docs/ROUND2_SUMMARY.md",
    ROOT / "docs/ROUND3_VALIDATION_REPORT.md",
    ROOT / "docs/INTERVIEW_GRILL.md",
    ROOT / "docs/P3_DECISION_REPORT.md",
    ROOT / "docs/ROUND3_CLOSURE.md",
)
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


class PortfolioEvidenceTests(unittest.TestCase):
    def test_internal_markdown_links_exist(self):
        checked_links = 0
        for doc in DOCS:
            self.assertTrue(doc.is_file(), doc)
            for raw in LINK_RE.findall(doc.read_text(encoding="utf-8")):
                if raw.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                checked_links += 1
                target = (doc.parent / unquote(raw.split("#", 1)[0])).resolve()
                with self.subTest(doc=doc.name, link=raw):
                    self.assertTrue(target.is_relative_to(ROOT.resolve()))
                    self.assertTrue(target.exists(), f"Missing local link in {doc}: {raw}")
        self.assertGreaterEqual(checked_links, 10, "Markdown link audit matched too few targets")

    def test_reported_cer_and_paired_difference(self):
        summary = json.loads((ROOT / "results/endpoint_summary.json").read_text(encoding="utf-8"))
        rows = summary["rows"]
        self.assertEqual([r["rule2_ms"] for r in rows], [300, 500, 700, 900, 1200])
        self.assertTrue(all(r["n"] == 30 and r["reference_chars"] == 403 for r in rows))
        for r in rows:
            with self.subTest(threshold=r["rule2_ms"]):
                self.assertAlmostEqual(r["cer"], (r["s"] + r["d"] + r["i"]) / 403)
        self.assertEqual((rows[0]["d"], rows[-1]["d"]), (244, 7))
        paired = json.loads((ROOT / "results/paired_endpoint_ci.json").read_text(encoding="utf-8"))
        self.assertEqual(paired["sampled_speakers"], 11)
        self.assertEqual(paired["bootstrap_reps"], 5000)
        self.assertAlmostEqual(
            paired["cer_difference_high_minus_low"], rows[-1]["cer"] - rows[0]["cer"]
        )


if __name__ == "__main__":
    unittest.main()
