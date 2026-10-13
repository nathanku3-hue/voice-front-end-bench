"""SentencePiece API contract: beam prefixes are tuples, decoder expects ID lists."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ctc_tokenization import build_detokenizer


class SentencePieceBoundaryTests(unittest.TestCase):
    def test_tuple_beam_ids_are_converted_to_list_at_decode_boundary(self):
        tokens = ["<blk>", "A", "B"]
        class MockSentencePiece:
            def __init__(self, model_file):
                self.model_file = model_file

            def get_piece_size(self):
                return len(tokens)

            def id_to_piece(self, i):
                return tokens[i]

            def decode(self, ids):
                if not isinstance(ids, list):
                    raise TypeError("SentencePiece needs list[int]")
                return "".join(tokens[i] for i in ids)

        fake = SimpleNamespace(SentencePieceProcessor=MockSentencePiece)
        with patch.dict(sys.modules, {"sentencepiece": fake}):
            detokenize = build_detokenizer(
                tokens, kind="bbpe", bpe_model="synthetic-not-a-real-model",
                byte_errors="replace",
            )
            self.assertEqual(detokenize((1, 2)), "AB")
            with self.assertRaises(ValueError):
                detokenize((0, 1))


if __name__ == "__main__":
    unittest.main()
