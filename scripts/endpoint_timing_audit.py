#!/usr/bin/env python3
"""Paired endpoint stream-time analysis, independent of estimated speech-end labels.

Uses previously recorded waveform replay timestamps. These are stream-time
finalization events, including synthetic tail-silence when present, NOT
microphone-to-response latency or directly comparable human speech-end offsets.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

THRESHOLDS = (300, 500, 700, 900, 1200)
PAIRS = ((300, 700), (700, 900), (700, 1200), (300, 1200))


def quantile(data: list[float], q: float) -> float:
    if not data or not 0 <= q <= 1:
        raise ValueError("Quantile requires a non-empty set and 0 <= q <= 1")
    data = sorted(data)
    at = (len(data) - 1) * q
    lo = int(at)
    return data[lo] + (data[min(lo + 1, len(data) - 1)] - data[lo]) * (at - lo)


def load_runs(directory: Path, thresholds=THRESHOLDS) -> dict[int, dict]:
    runs = {}
    for threshold in thresholds:
        filename = directory / f"rule2_{threshold}ms.json"
        payload = json.loads(filename.read_text(encoding="utf-8"))
        rows = payload["utterances"]
        config = payload["config"]
        if int(config["rule2_trailing_silence_ms"]) != threshold:
            raise ValueError(f"{filename}: threshold does not match payload config")
        if not rows or len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"{filename}: missing or duplicate utterances")
        for row in rows:
            if (row.get("endpoint_detected") is not True
                    or not isinstance(row.get("endpoint_audio_time_s"), (int, float))
                    or row["endpoint_audio_time_s"] < 0
                    or not row.get("speaker_id")):
                raise ValueError(f"{filename}: missing endpoint time or speaker")
        runs[threshold] = payload

    first = runs[thresholds[0]]
    # Narrow causal comparison: all non-threshold config fields and model
    # hashes must remain identical; same ordered files, labels and speakers.
    common_config = {k: v for k, v in first["config"].items()
                     if k != "rule2_trailing_silence_ms"}
    keys = ("id", "wav_sha256", "reference", "reference_chars",
            "speaker_id", "speech_end_s", "audio_duration_s")
    expected = [tuple(r.get(k) for k in keys) for r in first["utterances"]]
    for threshold in thresholds[1:]:
        current = runs[threshold]
        if current["model"].get("file_sha256") != first["model"].get("file_sha256"):
            raise ValueError("Model hash differs across settings")
        if {k: v for k, v in current["config"].items()
            if k != "rule2_trailing_silence_ms"} != common_config:
            raise ValueError("Non-threshold replay config differs across settings")
        if [tuple(r.get(k) for k in keys) for r in current["utterances"]] != expected:
            raise ValueError("Paired utterance/audio/reference mismatch")
    return runs


def paired_measurements(runs: dict[int, dict], low: int, high: int) -> list[dict]:
    if low >= high:
        raise ValueError("Require ordered low < high thresholds")
    a = runs[low]["utterances"]
    b = runs[high]["utterances"]
    return [{
        "id": x["id"],
        "speaker_id": x["speaker_id"],
        "delta_ms": (y["endpoint_audio_time_s"] - x["endpoint_audio_time_s"]) * 1000,
        "deletion_reduction": x["d"] - y["d"],
        "error_reduction": (x["s"] + x["d"] + x["i"])
                           - (y["s"] + y["d"] + y["i"]),
        "reference_chars": x["reference_chars"],
    } for x, y in zip(a, b)]


def paired_summary(rows: list[dict], reps: int, seed: int) -> dict:
    if not rows or reps < 100:
        raise ValueError("Need paired utterances and >=100 bootstrap reps")
    changes = [r["delta_ms"] for r in rows]
    speakers = sorted({r["speaker_id"] for r in rows})
    grouped = {s: [r for r in rows if r["speaker_id"] == s] for s in speakers}
    rng = random.Random(seed)
    boot = []
    for _ in range(reps):
        sampled = [r for spk in rng.choices(speakers, k=len(speakers))
                   for r in grouped[spk]]
        boot.append(statistics.mean(r["delta_ms"] for r in sampled))
    return {
        "n": len(rows), "speakers": len(speakers),
        "mean_delta_ms": statistics.mean(changes),
        "median_delta_ms": statistics.median(changes),
        "p10_delta_ms": quantile(changes, .1),
        "p90_delta_ms": quantile(changes, .9),
        "min_delta_ms": min(changes),
        "max_delta_ms": max(changes),
        "positive_n": sum(v > 1e-7 for v in changes),
        "zero_n": sum(abs(v) <= 1e-7 for v in changes),
        "negative_n": sum(v < -1e-7 for v in changes),
        "mean_delta_cluster_bootstrap95_ms": [quantile(boot, .025), quantile(boot, .975)],
        "deletion_reduction": sum(r["deletion_reduction"] for r in rows),
        "total_error_reduction": sum(r["error_reduction"] for r in rows),
        "reps": reps, "seed": seed,
    }


def audit(directory: Path, reps=5000, seed=20261011) -> dict:
    runs = load_runs(directory)
    result = {}
    for low, high in PAIRS:
        data = paired_measurements(runs, low, high)
        result[f"{low}_to_{high}"] = paired_summary(data, reps, seed)
    return {
        "source": str(directory),
        "units": "ms of unpaced waveform stream-time, including replayed synthetic tail silence",
        "reference_independence": "Paired endpoint times require no estimated speech_end_s; source WAV/end reference held fixed",
        "non_claims": [
            "Not measured wall-clock post-speech latency",
            "Not a proof that 700ms or any threshold meets a product SLA",
            "Unpaced stream-time is affected by 160ms chunks and last partial audio block",
            "Human speech-end annotations are absent; 28/30 tails are unclear",
            "Small 30-utterance/11-speaker dataset, not in-car command performance",
        ],
        "pairs": result,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results/endpoint_sweep"))
    p.add_argument("--output", type=Path, default=None,
                   help="Optional JSON output; prefer ignored build/ when rechecking")
    p.add_argument("--reps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261011)
    args = p.parse_args()
    report = audit(args.results, reps=args.reps, seed=args.seed)
    print("| threshold pair | mean Δ (ms) | median Δ | P10–P90 | cluster-95% CI of mean | Δ D reduced |")
    print("|---|---:|---:|---:|---:|---:|")
    for pair, row in report["pairs"].items():
        lo, hi = row["mean_delta_cluster_bootstrap95_ms"]
        print(f"| {pair.replace('_to_', '→')} | {row['mean_delta_ms']:.1f} | "
              f"{row['median_delta_ms']:.1f} | {row['p10_delta_ms']:.1f}–"
              f"{row['p90_delta_ms']:.1f} | [{lo:.1f}, {hi:.1f}] | "
              f"{row['deletion_reduction']} |")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"Saved {args.output}")
    print("Stream-time only; no manually validated human speech-end or product SLA.")


if __name__ == "__main__":
    main()
