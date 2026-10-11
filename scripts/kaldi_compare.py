#!/usr/bin/env python3
"""Round-1 numerical audit vs torchaudio's Kaldi-compatible fbank implementation.

Do NOT describe this as parity with an official Kaldi executable or bit-exact.
Inputs are explicit PCM-amplitude float32 (not normalized [-1, 1]).
"""
import argparse
import csv
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi

SAMPLE_RATE = 16000
KALDI_OPTIONS = dict(
    sample_frequency=16000.0,
    frame_length=25.0,
    frame_shift=10.0,
    num_mel_bins=80,
    low_freq=20.0,
    high_freq=0.0,
    dither=0.0,
    remove_dc_offset=True,
    preemphasis_coefficient=0.97,
    window_type="povey",
    snip_edges=True,
    use_power=True,
    use_log_fbank=True,
    use_energy=False,
    subtract_mean=False,
    round_to_power_of_two=True,
)


def signal_cases():
    seconds = 1.25
    n = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float64)
    t = n / SAMPLE_RATE
    rng = np.random.default_rng(20261011)
    multitone = 3000 * np.sin(2 * math.pi * 300 * t) + 1700 * np.sin(2 * math.pi * 1030 * t) + 500 * np.sin(2 * math.pi * 2600 * t)
    noisy = 2200 * np.sin(2 * math.pi * 720 * t) + rng.normal(0, 850, size=n.size)
    chirp = 2200 * np.sin(2 * math.pi * (240 * t + 0.5 * 2800 / seconds * t * t))
    silence = np.zeros_like(t)
    return {"multitone": multitone.astype("<f4"), "noisy": noisy.astype("<f4"), "chirp": chirp.astype("<f4"), "silence": silence.astype("<f4")}


def stats(actual, expected):
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: C++={actual.shape}; torchaudio={expected.shape}")
    err = actual - expected
    frame, band = np.unravel_index(int(np.argmax(np.abs(err))), err.shape)
    return dict(frames=int(actual.shape[0]), max_abs=float(np.max(np.abs(err))),
                mae=float(np.mean(np.abs(err))), rmse=float(np.sqrt(np.mean(err * err))),
                worst_frame=int(frame), worst_mel_band=int(band),
                value_cpp=float(actual[frame, band]), value_ref=float(expected[frame, band]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build/fbank_dump", type=Path)
    parser.add_argument("--output", default="results/kaldi_compatible_parity.json", type=Path)
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error(f"Binary not found: {args.binary}")
    rows = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for name, pcm in signal_cases().items():
            raw = td / f"{name}.raw"
            out = td / f"{name}.csv"
            pcm.tofile(raw)
            subprocess.run([str(args.binary.resolve()), str(raw), str(out), "160"], check=True, capture_output=True, text=True)
            with out.open(newline="") as fp:
                actual = np.array([[float(v) for v in row] for row in csv.reader(fp)], dtype=np.float64).reshape(-1, 80)
            for precision in ("float32", "float64"):
                dtype = torch.float32 if precision == "float32" else torch.float64
                waveform = torch.from_numpy(pcm.copy()).to(dtype=dtype).unsqueeze(0)
                expected = kaldi.fbank(waveform, **KALDI_OPTIONS).cpu().numpy().astype(np.float64)
                row = {"signal": name, "reference_precision": precision, **stats(actual, expected)}
                rows.append(row)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    report = {
        "scope": "C++ (FFTW double) vs torchaudio.compliance.kaldi only, not official Kaldi binaries",
        "input_format": "raw float32 PCM, Kaldi-style unnormalized amplitude in synthetic cases",
        "reference": {"torchaudio": torchaudio_version(), "torch": torch.__version__, "numpy": np.__version__, "settings": KALDI_OPTIONS},
        "results": rows,
        "warnings": ["float32/float64 reference epsilons differ on silence and low-energy bins", "not bit-exact", "not measured on real speech"],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Signal       Reference    Frames  Max abs     MAE         RMSE")
    for row in rows:
        print(f"{row['signal']:<12} {row['reference_precision']:<10} {row['frames']:<6} {row['max_abs']:<11.5g} {row['mae']:<11.5g} {row['rmse']:<11.5g}")
    print(f"Recorded: {args.output}")


def torchaudio_version():
    import torchaudio
    return torchaudio.__version__


if __name__ == "__main__":
    main()
