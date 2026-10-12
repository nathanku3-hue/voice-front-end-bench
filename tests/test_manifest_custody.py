"""Source-custody comparison stays offline and fails closed on mismatches."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_round2_manifest import verify


class ManifestVerifierTests(unittest.TestCase):
    def test_match_and_audio_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "audio").mkdir()
            content = b"RIFF-fake-PCM-test-bytes"
            (root / "audio/s.wav").write_bytes(content)
            row = {
                "id": "BAC0000000000001", "speaker_id": "S001",
                "source_split": "AISHELL-1/test", "source_row_idx": 3,
                "source_wavpath": "test/S001/BAC0000000000001.wav",
                "wav": "audio/s.wav", "reference": "导航回家",
                "wav_sha256": hashlib.sha256(content).hexdigest(),
            }
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"
            reference.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            candidate.write_text(reference.read_text(encoding="utf-8"), encoding="utf-8")
            self.assertEqual(verify(reference, candidate), 1)
            (root / "audio/s.wav").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify(reference, candidate)

    def test_order_identity_and_label_are_not_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = {
                "id": "BAC0000000000001", "speaker_id": "S001",
                "source_split": "AISHELL-1/test", "source_row_idx": 3,
                "source_wavpath": "test/S001/a.wav",
                "wav": "audio/a.wav", "reference": "导航回家",
                "wav_sha256": "ab" * 32,
            }
            a = root / "source.jsonl"
            b = root / "downloaded.jsonl"
            a.write_text(json.dumps(base, ensure_ascii=False) + "\n", encoding="utf-8")
            for field, value in (("reference", "导航去哪"), ("wav_sha256", "cd" * 32),
                                 ("source_row_idx", 4)):
                row = dict(base, **{field: value})
                b.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        verify(a, b, check_wav=False)


if __name__ == "__main__":
    unittest.main()
