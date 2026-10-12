#!/usr/bin/env python3
"""Compare C++ StreamingFbank to ACTUAL sherpa-onnx OnlineStream.get_frames().

This is a negative/positive contract audit, not a model inference integration.
Same WAV/normalized float waveform is sent to both frontends, but their declared
window placement and high-frequency cutoffs intentionally remain unchanged to
test drop-in compatibility AS IMPLEMENTED. It does not assert feature parity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

MODEL_FILES = {
    "tokens": "tokens.txt",
    "encoder": "encoder-epoch-99-avg-1.int8.onnx",
    "decoder": "decoder-epoch-99-avg-1.onnx",
    "joiner": "joiner-epoch-99-avg-1.int8.onnx",
}


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getcomptype()) != (
                16000, 1, 2, "NONE"):
            raise ValueError("Require 16kHz mono 16-bit uncompressed WAV")
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768


def native_online_features(wave: np.ndarray, model: Path, frames: int, *,
                           snip_edges: bool, high_freq: float) -> np.ndarray:
    import sherpa_onnx
    names = {k: model / v for k, v in MODEL_FILES.items()}
    missing = [str(p) for p in names.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model assets: {missing}")
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        **{key: str(path) for key, path in names.items()},
        sample_rate=16000, feature_dim=80, low_freq=20.0,
        high_freq=high_freq, dither=0.0, normalize_samples=True,
        snip_edges=snip_edges, num_threads=2, decoding_method="greedy_search",
    )
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, wave)
    stream.input_finished()
    flat = np.asarray(stream.get_frames(0, frames), dtype=np.float64)
    if flat.size != frames * 80:
        raise RuntimeError(f"Unexpected native GetFrames size: {flat.size}")
    result = flat.reshape(frames, 80)
    if not np.all(np.isfinite(result)):
        raise ValueError("Native features include non-finite values")
    return result


def custom_features(wave: np.ndarray, binary: Path) -> np.ndarray:
    if not binary.is_file():
        raise FileNotFoundError(f"C++ fbank_dump binary missing: {binary}")
    with tempfile.TemporaryDirectory() as tmp:
        raw, csv_path = Path(tmp) / "wave.raw", Path(tmp) / "fbank.csv"
        wave.astype("<f4").tofile(raw)
        subprocess.run([str(binary.resolve()), str(raw), str(csv_path), "160"],
                       check=True, capture_output=True, text=True)
        with csv_path.open(encoding="utf-8", newline="") as fp:
            matrix = np.asarray([[float(value) for value in row] for row in csv.reader(fp)],
                                dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 80 or not np.all(np.isfinite(matrix)):
        raise ValueError("Invalid C++ features")
    return matrix


def compare(cpp: np.ndarray, native: np.ndarray) -> dict:
    if cpp.shape != native.shape or cpp.ndim != 2:
        raise ValueError("Cannot compare unequal feature shapes")
    diff = cpp - native
    a, b = cpp.ravel(), native.ravel()
    return {
        "frames_compared": int(cpp.shape[0]),
        "feature_dim": int(cpp.shape[1]),
        "mae": float(np.mean(np.abs(diff))),
        "p95_abs": float(np.percentile(np.abs(diff), 95)),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_signed_difference": float(np.mean(diff)),
        "flattened_feature_cosine": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
        "cpp_mean": float(np.mean(cpp)),
        "native_mean": float(np.mean(native)),
    }


def audit(model_dir: Path, binary: Path, wav: Path) -> dict:
    samples = read_wav(wav)
    cpp = custom_features(samples, binary)
    native = native_online_features(samples, model_dir, len(cpp),
                                    snip_edges=False, high_freq=-400.0)
    matched_native = native_online_features(samples, model_dir, len(cpp),
                                            snip_edges=True, high_freq=0.0)
    return {
        "scope": "Actual sherpa-onnx OnlineStream.get_frames vs C++ StreamingFbank, SAME normalized float32 waveform",
        "wav": str(wav),
        "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
        "model_path": str(model_dir),
        "custom_contract": {
            "sample_rate": 16000, "sample_input_scale": "float32 normalized [-1,1]",
            "snip_edges": True, "high_freq_hz": 8000, "fft_size": 512,
            "feature_dim": 80, "precision": "FFTW float64 internal; output CSV float64",
        },
        "native_contract": {
            "sample_rate": 16000, "sample_input_scale": "float32 normalized [-1,1] (normalize_samples=True)",
            "snip_edges": False, "high_freq_hz": 7600,
            "high_freq_setting": "-400.0 offset from 8000Hz Nyquist",
            "feature_dim": 80, "precision": "native knf float32 features",
        },
        "comparison": compare(cpp, native),
        "matched_reference_contract": {
            "note": "Diagnostic rerun changing sherpa frontend settings only, NOT Round 2 recognizer settings",
            "snip_edges": True, "high_freq_hz": 8000,
            "normalize_samples": True, "dither": 0.0, "feature_dim": 80,
        },
        "comparison_with_matched_reference": compare(cpp, matched_native),
        "interpretation": (
            "Per-index comparison is diagnostic only: different snip_edges means "
            "frame centers are not time aligned; differing mel upper cutoff means "
            "filterbanks are not equivalent. The matched-reference diagnostic isolates "
            "precision/implementation differences from those contract mismatches. "
            "Cosine here is over FEATURES, not encoder logits."
        ),
        "limitations": [
            "Native get_frames queried only for count produced by snip_edges=True C++",
            "Changing sherpa frontend parameters in a diagnostic does not demonstrate native model accuracy or end-to-end equivalence",
            "No ONNX encoder input injection or decoded text comparison performed",
            "No claim that feature or model-output cosine similarity meets any threshold",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("build/fbank_dump"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit(args.model_dir, args.binary, args.wav)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
