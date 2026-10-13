"""CTC tokenization and real-model export contract regression tests.

Synthetic contract fixtures must never be reported as actual model CER.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ctc_tokenization import load_tokens, BYTE_ALPHABET, decode_encoded_bytes, build_detokenizer
from ctc_export_logits import forward_log_probs, checked_source_rows


class CtcTokenizationTests(unittest.TestCase):
    def test_byte_alphabet_is_bijective(self):
        self.assertEqual(len(BYTE_ALPHABET), 256)
        self.assertEqual(len(set(BYTE_ALPHABET.values())), 256)
        self.assertEqual(BYTE_ALPHABET[ord("A")], "A")
        self.assertEqual(BYTE_ALPHABET[32], " ")

    def test_utf8_icefall_byte_roundtrip(self):
        text = "年龄大跑者的步幅明显短于年轻人"
        encoded = "".join(BYTE_ALPHABET[b] for b in text.encode("utf-8"))
        self.assertEqual(decode_encoded_bytes(encoded), text)
        with self.assertRaises(ValueError):
            decode_encoded_bytes("❤")
        with self.assertRaises(ValueError):
            decode_encoded_bytes(BYTE_ALPHABET[0xe4])  # Truncated UTF-8 multibyte symbol
        self.assertEqual(decode_encoded_bytes(BYTE_ALPHABET[0xe4], errors="replace"), "\ufffd")
        with self.assertRaises(ValueError):
            decode_encoded_bytes(BYTE_ALPHABET[0xe4], errors="ignore")

    def test_explicit_and_implicit_token_table(self):
        with tempfile.TemporaryDirectory() as temp:
            f = Path(temp) / "tokens.txt"
            f.write_text("<blk> 0\n<sos/eos> 1\n你 2\n", encoding="utf-8")
            self.assertEqual(load_tokens(f), ["<blk>", "<sos/eos>", "你"])
            f.write_text("<blk>\n<sos/eos>\n你\n", encoding="utf-8")
            self.assertEqual(load_tokens(f), ["<blk>", "<sos/eos>", "你"])
            f.write_text("<blk> 0\n你 2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_tokens(f)
            f.write_text("<blk> 0\n你 0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_tokens(f)
            f.write_text("<blk> 0\n你\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_tokens(f)

    def test_special_tokens_are_not_silently_removed(self):
        decode = build_detokenizer(["<blk>", "<unk>", "你", "好"])
        self.assertEqual(decode([2, 3]), "你好")
        with self.assertRaises(ValueError):
            decode([1, 2])

    def test_model_output_is_trimmed_to_actual_length(self):
        class FakeSession:
            def run(self, names, values):
                self.names, self.values = names, values
                logits = np.log(np.array([[
                    [.7, .2, .1], [.1, .8, .1],
                    [.1, .1, .8], [.3, .3, .4]]], dtype=np.float32))
                return [logits, np.array([3], dtype=np.int64)]
        session = FakeSession()
        out, details = forward_log_probs(session, np.zeros((40, 80), dtype=np.float32), 3)
        self.assertEqual(out.shape, (3, 3))
        self.assertEqual(details["raw_ctc_frames"], 4)
        self.assertEqual(details["valid_ctc_frames"], 3)
        self.assertEqual(session.names, ["log_probs", "log_probs_len"])
        self.assertEqual(session.values["x_lens"].tolist(), [40])

    def test_source_hash_checked_before_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "test.wav").write_bytes(b"invalid WAV for negative test")
            (root / "manifest.jsonl").write_text(json.dumps(dict(
                id="test", wav="test.wav", reference="你好", wav_sha256="wrong")) + "\n")
            with self.assertRaises(ValueError):
                checked_source_rows(root / "manifest.jsonl", 1)
            (root / "manifest.jsonl").write_text(json.dumps(dict(
                id="test", wav="../outside.wav", reference="你好", wav_sha256="wrong")) + "\n")
            with self.assertRaises(ValueError):
                checked_source_rows(root / "manifest.jsonl", 1)


if __name__ == "__main__":
    unittest.main()
