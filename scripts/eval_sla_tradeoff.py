#!/usr/bin/env python3
"""Illustrative deletion-vs-stream-time cost envelope from saved Round 2 JSON.

J(theta; lambda) = mean(endpoint stream-time in ms per utterance)
                   + lambda_ms_per_char * (deletions / utterances).
Units match: both terms are ms per utterance. The cost weight is hypothetical.
This is NOT a user-facing latency SLA or car-command failure-cost estimate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from endpoint_timing_audit import THRESHOLDS, load_runs


def observations(runs: dict[int, dict]) -> list[dict]:
    data = []
    reference_n = len(runs[THRESHOLDS[0]]["utterances"])
    for threshold in THRESHOLDS:
        rows = runs[threshold]["utterances"]
        if len(rows) != reference_n:
            raise ValueError("Different number of utterances")
        d = sum(r["d"] for r in rows)
        s = sum(r["s"] for r in rows)
        i = sum(r["i"] for r in rows)
        count = sum(r["reference_chars"] for r in rows)
        data.append({
            "threshold_ms": threshold,
            "utterances": reference_n,
            "mean_endpoint_stream_time_ms": 1000 * mean(r["endpoint_audio_time_s"] for r in rows),
            "deletions": d,
            "deletions_per_utterance": d / reference_n,
            "s": s, "i": i, "reference_chars": count,
            "cer": (s + d + i) / count,
        })
    return data


def cost(record: dict, lam: float) -> float:
    if lam < 0:
        raise ValueError("lambda must be nonnegative")
    return record["mean_endpoint_stream_time_ms"] + lam * record["deletions_per_utterance"]


def winners(records: list[dict], lam: float) -> list[int]:
    evaluated = [(r["threshold_ms"], cost(r, lam)) for r in records]
    best = min(x[1] for x in evaluated)
    return sorted(t for t, score in evaluated if abs(score - best) <= 1e-8 * max(1, abs(best)))


def breakpoints(records: list[dict]) -> list[float]:
    possible = [0.0]
    for j, x in enumerate(records):
        for y in records[j + 1:]:
            a = x["deletions_per_utterance"] - y["deletions_per_utterance"]
            b = y["mean_endpoint_stream_time_ms"] - x["mean_endpoint_stream_time_ms"]
            if abs(a) <= 1e-12:
                continue
            crossing = b / a
            if crossing > 0:
                possible.append(crossing)
    return sorted(set(round(v, 11) for v in possible))


def envelope(records: list[dict]) -> list[dict]:
    cuts = breakpoints(records)
    intervals = []
    for n, left in enumerate(cuts):
        right = cuts[n + 1] if n + 1 < len(cuts) else None
        probe = (left + right) / 2 if right is not None else left + max(1000.0, left)
        chosen = winners(records, probe)
        if intervals and intervals[-1]["winner_threshold_ms"] == chosen:
            intervals[-1]["lambda_max_exclusive"] = right
        else:
            intervals.append({
                "lambda_min_exclusive": left,
                "lambda_max_exclusive": right,
                "winner_threshold_ms": chosen,
            })
    # Thresholds are equal-cost exactly at some boundaries: check with winners().
    return intervals


def build_report(root: Path, weights: list[float]) -> dict:
    records = observations(load_runs(root))
    base = records[0]["mean_endpoint_stream_time_ms"]
    for row in records:
        row["extra_mean_stream_time_vs_300_ms"] = row["mean_endpoint_stream_time_ms"] - base
    scenarios = []
    for weight in weights:
        scenarios.append({
            "lambda_ms_per_deleted_char": weight,
            "winners_threshold_ms": winners(records, weight),
            "cost_ms_per_utterance_by_threshold": {
                str(row["threshold_ms"]): cost(row, weight) - base for row in records
            }
        })
    return {
        "method": "observational linear sensitivity model, NOT a fitted business loss",
        "source": str(root),
        "objective": "mean_endpoint_stream_time_ms + lambda_ms_per_deleted_char * deletions_per_utterance",
        "lambda_units": "ms added to expected per-utterance cost per deletion edit",
        "stream_time_caveat": "unpaced file replay stream-time, NOT speech-end-to-user-response latency",
        "unverified_assumption": "one deletion edit costs the same independent of context and utility; λ is illustrative",
        "notes": [
            "30 matched AISHELL-1 read-speech utterances, 11 speakers; not real car commands",
            "Missing human-verified speech ends and realistic application SLA",
            "lambda=0 selects fastest observed endpoint; no optimal production threshold is claimed",
            "Strict interior winner intervals; exact boundaries may have tied configurations",
            "Mean across identical utterances, unlike conditional on-time mean delay",
        ],
        "observations": records,
        "open_intervals": envelope(records),
        "scenarios": scenarios,
    }


def plot(records: list[dict], path: Path, max_weight=1000) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    xs = np.linspace(0, max_weight, 600)
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=150)
    base = records[0]["mean_endpoint_stream_time_ms"]
    for r in records:
        values = [cost(r, float(lam)) - base for lam in xs]
        ax.plot(xs, values, label=f'{r["threshold_ms"]}ms')
    ax.set(xlabel="Hypothetical deletion penalty λ (ms per deleted character)",
           ylabel="Relative mean cost vs. 300ms stream-time (ms/utterance)",
           title="Illustrative endpoint decision cost, NOT product SLA")
    ax.legend(title="Rule2 threshold", ncol=3)
    ax.grid(alpha=0.25)
    fig.text(.5, .015, "30 AISHELL-1 read utterances; stream-time replay; deletion utility is an assumption",
             ha="center", size=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results/endpoint_sweep"))
    p.add_argument("--output", type=Path, default=Path("build/p3_cost_tradeoff.json"))
    p.add_argument("--csv", type=Path, default=Path("build/p3_cost_observations.csv"))
    p.add_argument("--plot", type=Path, default=None)
    p.add_argument("--lambdas", nargs="+", type=float,
                   default=[0, 15, 20, 300, 400, 500, 600, 800, 1000])
    args = p.parse_args()
    if any(x < 0 for x in args.lambdas):
        p.error("All lambda values must be >= 0")
    report = build_report(args.results, args.lambdas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["observations"][0]))
        writer.writeheader()
        writer.writerows(report["observations"])
    if args.plot is not None:
        plot(report["observations"], args.plot)
    print("Open-interval optimum by hypothetical lambda (ms/deletion):")
    for section in report["open_intervals"]:
        lower, upper = section["lambda_min_exclusive"], section["lambda_max_exclusive"]
        print(f"  {lower:.3f} < λ < {upper:.3f}" if upper is not None
              else f"  λ > {lower:.3f}", "->", section["winner_threshold_ms"])
    print("Scenario winners:", [(s["lambda_ms_per_deleted_char"], s["winners_threshold_ms"])
                                for s in report["scenarios"]])
    print("Not a deployment recommendation; no human speech-end latency or business utility verified.")


if __name__ == "__main__":
    main()
