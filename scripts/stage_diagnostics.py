#!/usr/bin/env python3
"""Numerical stage tracing: C++ FFTW-double vs torchaudio.compliance.kaldi.

Run on the same Linux installation as build/fbank_dump. The script checks that
its PyTorch stage reconstruction reproduces the public torchaudio.fbank output,
then compares pre-emphasis, windowed samples, FFT power, Mel energies and logs.
This is an opt-in diagnostic, NOT a new front-end algorithm.
"""
import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from kaldi_compare import KALDI_OPTIONS, signal_cases


STAGE_FILES = (
    "after_preemphasis",
    "windowed_padded",
    "power_spectrum",
    "mel_energy",
    "log_fbank",
)


def reference_stages(waveform, precision):
    dtype = torch.float32 if precision == "float32" else torch.float64
    audio = torch.from_numpy(waveform.copy()).to(dtype=dtype)
    frames = kaldi._get_strided(audio, 400, 160, True)
    frames = frames - frames.mean(dim=1, keepdim=True)
    padded_left = F.pad(frames.unsqueeze(0), (1, 0), mode="replicate").squeeze(0)
    after_preemphasis = frames - 0.97 * padded_left[:, :-1]

    window = kaldi._feature_window_function("povey", 400, 0.42, audio.device, dtype)
    windowed_padded = F.pad(after_preemphasis * window, (0, 112))
    power_spectrum = torch.fft.rfft(windowed_padded).abs().pow(2)

    # torchaudio constructs the triangular weights in float32, then casts to
    # the waveform's dtype. Keep that source behavior even for float64.
    mel_filters, _ = kaldi.get_mel_banks(80, 512, 16000.0, 20.0, 8000.0, 100.0, -500.0, 1.0)
    mel_filters = F.pad(mel_filters.to(device=audio.device, dtype=dtype), (0, 1))
    mel_energy = torch.mm(power_spectrum, mel_filters.T)
    log_fbank = torch.max(mel_energy, kaldi._get_epsilon(audio.device, dtype)).log()

    public = kaldi.fbank(audio.unsqueeze(0), **KALDI_OPTIONS)
    reconstruction_error = float(torch.max(torch.abs(public - log_fbank)).item())
    # This is a numerical sanity check, NOT an assertion of C++/Kaldi parity.
    if reconstruction_error > (2e-5 if precision == "float32" else 1e-10):
        raise AssertionError(
            f"Torchaudio stage reconstruction differs from public fbank: "
            f"{precision} error={reconstruction_error}"
        )
    stages = dict(
        after_preemphasis=after_preemphasis,
        windowed_padded=windowed_padded,
        power_spectrum=power_spectrum,
        mel_energy=mel_energy,
        log_fbank=log_fbank,
    )
    return stages, reconstruction_error


def compare_stage(actual, reference):
    ref = reference.detach().cpu().numpy().astype(np.float64)
    if actual.shape != ref.shape:
        raise AssertionError(f"Stage shape mismatch: C++ {actual.shape}, reference {ref.shape}")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(ref)):
        raise AssertionError("Non-finite stage values detected")
    delta = actual - ref
    index = int(np.argmax(np.abs(delta)))
    ref_value = float(ref[index])
    actual_value = float(actual[index])
    return {
        "size": int(ref.size),
        "max_abs": float(np.max(np.abs(delta))),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "worst_index": index,
        "cpp_at_worst": actual_value,
        "reference_at_worst": ref_value,
        "absolute_relative_error_at_worst": float(abs(delta[index]) / abs(ref_value)) if ref_value else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/fbank_dump"))
    parser.add_argument("--signal", choices=tuple(signal_cases()), default="chirp")
    parser.add_argument("--frame", type=int, default=5)
    parser.add_argument("--band", type=int, default=78)
    parser.add_argument("--output", type=Path, default=Path("results/stage_diagnostics.json"))
    parser.add_argument("--keep-npy-dir", type=Path, default=None,
                        help="Optional directory for retaining the five raw stage arrays")
    args = parser.parse_args()
    pcm = signal_cases()[args.signal]
    total_frames = 0 if pcm.size < 400 else 1 + (pcm.size - 400) // 160
    if not (0 <= args.frame < total_frames):
        parser.error(f"Frame index must be in [0, {total_frames - 1}]")
    if not (0 <= args.band < 80):
        parser.error("Mel band index must be in [0, 79]")
    if not args.binary.is_file():
        parser.error(f"Missing C++ binary: {args.binary}")

    with tempfile.TemporaryDirectory() as tempdir:
        temp = Path(tempdir)
        raw = temp / "input.raw"
        csv_path = temp / "features.csv"
        trace_dir = temp / "trace"
        pcm.tofile(raw)
        cmd = [
            str(args.binary.resolve()), str(raw), str(csv_path), "160",
            "--dump-stage", str(trace_dir), "--frame", str(args.frame),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        cpp = {name: np.load(trace_dir / f"{name}.npy") for name in STAGE_FILES}
        if args.keep_npy_dir is not None:
            args.keep_npy_dir.mkdir(parents=True, exist_ok=True)
            for name in STAGE_FILES:
                shutil.copyfile(trace_dir / f"{name}.npy", args.keep_npy_dir / f"{name}.npy")
        with csv_path.open(newline="") as f:
            all_frames = np.asarray([[float(v) for v in row] for row in csv.reader(f)], dtype=np.float64)
        if all_frames.shape != (total_frames, 80):
            raise AssertionError(f"Unexpected C++ feature shape: {all_frames.shape}")
        cpp_csv_trace_diff = float(np.max(np.abs(all_frames[args.frame] - cpp["log_fbank"])))
        if cpp_csv_trace_diff > 1e-12:
            raise AssertionError(f"Trace changed Fbank output: {cpp_csv_trace_diff}")

        report = {
            "reference": "torchaudio.compliance.kaldi (NOT official Kaldi binary)",
            "torch_version": torch.__version__,
            "torchaudio_version": torchaudio.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "signal": args.signal,
            "frame_index_zero_based": args.frame,
            "frame_start_sample": args.frame * 160,
            "input": "float32 samples, unnormalized PCM-count amplitude",
            "cpp_trace_matches_csv_max_abs": cpp_csv_trace_diff,
            "stages": {},
            "target_mel_band": args.band,
        }
        references = {}
        for precision in ("float32", "float64"):
            ref_stages, internal_error = reference_stages(pcm, precision)
            references[precision] = ref_stages
            results = {
                "torchaudio_stage_reconstruction_vs_public_max_abs": internal_error,
                "comparison": {},
            }
            for name in STAGE_FILES:
                results["comparison"][name] = compare_stage(cpp[name], ref_stages[name][args.frame])

            target = args.band
            mel_cpp = float(cpp["mel_energy"][target])
            mel_ref = float(ref_stages["mel_energy"][args.frame][target].item())
            floor = float(kaldi._get_epsilon(torch.device("cpu"), torch.float32 if precision == "float32" else torch.float64))
            predicted_log_diff = math.log(max(mel_cpp, floor)) - math.log(max(mel_ref, floor))
            results["target_band"] = {
                "band_index_zero_based": target,
                "cpp_energy": mel_cpp,
                "reference_energy": mel_ref,
                "floor": floor,
                "energy_ratio_cpp_over_reference": mel_cpp / mel_ref if mel_ref != 0 else None,
                "cpp_log_energy": float(cpp["log_fbank"][target]),
                "reference_log_energy": float(ref_stages["log_fbank"][args.frame][target].item()),
                "predicted_log_difference_from_energy_ratio_and_floor": predicted_log_diff,
                "actual_log_difference": float(cpp["log_fbank"][target] - ref_stages["log_fbank"][args.frame][target].item()),
            }
            report["stages"][precision] = results

        # Cross over one stage at a time, using the SAME float32-derived Mel
        # weights and float64 weighted accumulation. This separates window
        # quantization from FFT arithmetic and from Mel accumulation.
        b = report["target_mel_band"]
        weights = kaldi.get_mel_banks(80, 512, 16000.0, 20.0, 8000.0, 100.0, -500.0, 1.0)[0]
        weight_band = np.asarray(F.pad(weights, (0, 1))[b].numpy(), dtype=np.float64)

        def integrated_energy(power):
            return float(np.dot(np.asarray(power, dtype=np.float64), weight_band))

        def fft_power_on_window(window, dtype):
            # The 'window' here is pre-FFT; only FFT+power dtype changes.
            return torch.fft.rfft(window.to(dtype=dtype)).abs().pow(2).to(dtype=torch.float64).numpy()

        w32 = references["float32"]["windowed_padded"][args.frame]
        w64 = references["float64"]["windowed_padded"][args.frame]
        cpp_window = torch.from_numpy(cpp["windowed_padded"])
        p32 = references["float32"]["power_spectrum"][args.frame]
        p64 = references["float64"]["power_spectrum"][args.frame]

        report["precision_crossover_energy_target_band"] = {
            "note": "All mel integration here uses exactly the same torchaudio-generated float32 mel weights promoted to float64 and a float64 dot-product.",
            "cpp_power_double_accum": integrated_energy(cpp["power_spectrum"]),
            "torch_ref32_power_double_accum": integrated_energy(p32.numpy()),
            "torch_ref64_power_double_accum": integrated_energy(p64.numpy()),
            "torch_float32_window_refft_float64": integrated_energy(fft_power_on_window(w32, torch.float64)),
            "torch_float64_window_refft_float32": integrated_energy(fft_power_on_window(w64, torch.float32)),
            "cpp_window_cast_float32_refft_float32": integrated_energy(fft_power_on_window(cpp_window, torch.float32)),
            "cpp_window_refft_float64": integrated_energy(fft_power_on_window(cpp_window, torch.float64)),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Stage diagnostics {args.signal} frame={args.frame}, band={args.band}")
    for precision in ("float32", "float64"):
        data = report["stages"][precision]
        print(precision, " public reconstruction:", data["torchaudio_stage_reconstruction_vs_public_max_abs"])
        for name in STAGE_FILES:
            s = data["comparison"][name]
            print(f"  {name:<21} max_abs={s['max_abs']:.7g} mae={s['mae']:.7g} worst_idx={s['worst_index']}")
        b = data["target_band"]
        print(f"  band{args.band}: Ecpp={b['cpp_energy']:.9g}, Eref={b['reference_energy']:.9g}, "
              f"log_diff={b['actual_log_difference']:.9g}, predicted={b['predicted_log_difference_from_energy_ratio_and_floor']:.9g}")
    print("Energy crossover (same Mel weights and float64 accumulation):")
    for name, value in report["precision_crossover_energy_target_band"].items():
        if name != "note":
            print(f"  {name:<40} {value:.10g}")
    print("Report:", args.output)


if __name__ == "__main__":
    main()
