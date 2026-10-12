#!/usr/bin/env python3
"""Paired speaker-cluster bootstrap on already-computed AISHELL endpoint JSONs.

Does NOT rerun ASR or claim population-level causal significance.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values)-1, int(q * len(values)))]


def grouped_condition(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for row in payload["utterances"]:
        if row["cer"] is None or not row.get("speaker_id"):
            raise ValueError("Missing labeled speaker data")
        groups[row["speaker_id"]].append(row)
    return groups, payload["model"]["file_sha256"]


def metrics(grouped, selected):
    sample_rows = [row for speaker in selected for row in grouped[speaker]]
    numer = sum(r["s"] + r["d"] + r["i"] for r in sample_rows)
    denom = sum(r["reference_chars"] for r in sample_rows)
    early = sum(r["endpoint_audio_time_s"] is not None and
                r["endpoint_audio_time_s"] + 0.03 < r["speech_end_s"]
                for r in sample_rows)
    return numer / denom, early / len(sample_rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results/endpoint_sweep"))
    p.add_argument("--low", type=int, default=300)
    p.add_argument("--high", type=int, default=1200)
    p.add_argument("--output", type=Path, default=Path("results/paired_endpoint_ci.json"))
    p.add_argument("--seed", type=int, default=20261011)
    p.add_argument("--reps", type=int, default=5000)
    args = p.parse_args()
    low, sha1 = grouped_condition(args.results / f"rule2_{args.low}ms.json")
    high, sha2 = grouped_condition(args.results / f"rule2_{args.high}ms.json")
    if sha1 != sha2 or set(low) != set(high):
        raise ValueError("Model hashes or speakers mismatch")
    for speaker in low:
        first = [(r["id"], r["wav_sha256"], r["reference"]) for r in low[speaker]]
        second = [(r["id"], r["wav_sha256"], r["reference"]) for r in high[speaker]]
        if first != second:
            raise ValueError(f"Paired input mismatch for {speaker}")
    speakers = sorted(low)
    observed_low = metrics(low, speakers)
    observed_high = metrics(high, speakers)
    rng = random.Random(args.seed)
    differences = [[], []]
    for _ in range(args.reps):
        chosen = rng.choices(speakers, k=len(speakers))
        l = metrics(low, chosen)
        h = metrics(high, chosen)
        differences[0].append(h[0] - l[0])
        differences[1].append(h[1] - l[1])
    report = {
        "method": "paired speaker-cluster bootstrap (resample 11 speakers with replacement)",
        "source": "AISHELL-1 test small stratified pilot",
        "model_sha256": sha1,
        "sampled_speakers": len(speakers),
        "original_utterances": sum(len(low[s]) for s in speakers),
        "bootstrap_reps": args.reps,
        "seed": args.seed,
        "low_rule2_ms": args.low, "high_rule2_ms": args.high,
        "low_cer": observed_low[0], "high_cer": observed_high[0],
        "cer_difference_high_minus_low": observed_high[0]-observed_low[0],
        "cer_difference_bootstrap95": [percentile(differences[0], .025), percentile(differences[0], .975)],
        "low_early_proxy": observed_low[1], "high_early_proxy": observed_high[1],
        "early_proxy_difference_high_minus_low": observed_high[1]-observed_low[1],
        "early_proxy_difference_bootstrap95": [percentile(differences[1], .025), percentile(differences[1], .975)],
        "limitations": [
            "Only 11 observed speakers and 30 clips; extrapolation to population/cars invalid",
            "Early endpoint comparison uses noisy, non-human speech-end proxy",
            "Confidence intervals describe this evaluation construction, not independent replicated training runs"
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.low}->{args.high}ms, CER change={report['cer_difference_high_minus_low']:.2%}, "
          f"speaker bootstrap95={report['cer_difference_bootstrap95']}, "
          f"early-proxy change={report['early_proxy_difference_high_minus_low']:.2%}", flush=True)


if __name__ == "__main__":
    main()
