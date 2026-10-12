import tempfile
import unittest
from pathlib import Path
import wave
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from endpoint_vad import estimate_speech_end_s, estimate_speech_end_details


class TestEndpointVad(unittest.TestCase):
    @staticmethod
    def write_wav(path, samples):
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((samples * 32767).astype("<i2").tobytes())

    def test_tail_silence_estimate(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.wav"
            speech = np.ones(16000, dtype=np.float32) * 0.1
            silence = np.zeros(8000, dtype=np.float32)
            pcm = np.concatenate([speech, silence])
            self.write_wav(p, pcm)
            end = estimate_speech_end_s(p)
            self.assertGreater(end, 0.9)
            self.assertLess(end, 1.2)
            details = estimate_speech_end_details(p)
            self.assertEqual(details["status"], "estimated")
            self.assertEqual(details["relative_threshold_db"], -35.0)
            self.assertGreater(details["trailing_margin_s"], 0.3)
            self.assertIn("peak_relative", details["speech_end_method"])

    def test_silent_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "silent.wav"
            self.write_wav(p, np.zeros(16000, dtype=np.float32))
            details = estimate_speech_end_details(p)
            self.assertIsNone(details["speech_end_s"])
            self.assertEqual(details["status"], "silent")

    def test_no_clear_tail_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "no_tail.wav"
            self.write_wav(p, np.full(16000, 0.05, dtype=np.float32))
            details = estimate_speech_end_details(p)
            self.assertEqual(details["status"], "no_clear_tail")
            self.assertLess(details["trailing_margin_s"], 0.05)

    def test_invalid_threshold_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.wav"
            self.write_wav(p, np.zeros(16000, dtype=np.float32))
            with self.assertRaises(ValueError):
                estimate_speech_end_details(p, relative_db=0)

if __name__ == "__main__":
    unittest.main()
