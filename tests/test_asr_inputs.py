"""Input-contract unit tests; no model downloads needed."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from asr_stream_bench import manifest_records, read_wav


def write_wav(path, rate=16000, channels=1):
    import struct
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<3h", -32768, 0, 32767))


class TestAudio(unittest.TestCase):
    def test_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ok.wav"
            write_wav(path)
            audio = read_wav(path)
            self.assertEqual(audio.dtype.name, "float32")
            self.assertEqual(audio.tolist(), [-1.0, 0.0, 32767 / 32768])
    def test_rate_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.wav"
            write_wav(path, rate=8000)
            with self.assertRaises(ValueError):
                read_wav(path)
    def test_manifest_relative_paths_and_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            wav = d / "audio.wav"
            write_wav(wav)
            p = d / "examples.jsonl"
            p.write_text(json.dumps({"id": "a", "wav": "audio.wav", "reference": "你好", "speech_end_s": 0.1},
                                    ensure_ascii=False) + "\n", encoding="utf-8")
            rows = manifest_records(argparse.Namespace(wav=None, manifest=p))
            self.assertEqual(rows[0]["wav"], str(wav.resolve()))
            self.assertEqual(rows[0]["reference"], "你好")
            p.write_text((p.read_text(encoding="utf-8") * 2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                manifest_records(argparse.Namespace(wav=None, manifest=p))

if __name__ == "__main__":
    unittest.main()
