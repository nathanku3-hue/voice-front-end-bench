#!/usr/bin/env python3
"""Run the slow-consumer backpressure audit and publish the evidence artifact.

The audited binary is tests/test_bounded_fbank_bridge.cpp. This runner only
orchestrates it, re-checks its invariants independently, and plots the result;
it never recomputes or smooths a number the binary produced.

What the audit establishes, and what it does not:

* establishes: with a consumer at 50% of the input rate, the caller-owned ring
  saturates, the producer is held back, and neither resident memory nor live
  heap grows across a long continuous stream; every feature frame emitted
  across a pause/resume boundary is bit-identical to an unthrottled baseline.
* does not establish: real-time behaviour, OS scheduling, or device latency.
  The stream is processed as fast as the host allows, not in real time.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

STATUS = "MEASURED_BACKPRESSURE_SOAK"


def run_case(binary, *, audio_seconds, capacity, chunk, out_path):
    cmd = [str(binary), "--soak-seconds", str(audio_seconds),
           "--soak-capacity", str(capacity), "--soak-chunk", str(chunk),
           "--json", str(out_path)]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"audit binary failed ({completed.returncode})\n"
                         f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    if not out_path.exists():
        raise SystemExit("audit binary did not write its JSON report")
    return json.loads(out_path.read_text(encoding="utf-8"))


def check(case, *, tolerance_bytes):
    """Fail closed: the artifact is only written when every claim holds."""
    problems = []
    if case["max_abs_diff"] != 0.0 or case["mean_abs_diff"] != 0.0:
        problems.append(f"max_abs_diff={case['max_abs_diff']} mean_abs_diff={case['mean_abs_diff']}")
    if case["compared_values"] != case["frames"] * 80:
        problems.append("not every feature value was compared")
    if case["pause_events"] <= 0 or case["resume_events"] <= 0:
        problems.append("no backpressure cycle occurred")
    if case["max_frames_held"] > case["config"]["capacity_frames"]:
        problems.append("frame watermark exceeded capacity")
    if case["max_pcm_backlog_samples"] > case["config"]["chunk_samples"]:
        problems.append("PCM backlog exceeded one chunk")
    heap_growth = case["heap_live_bytes"]["end"] - case["heap_live_bytes"]["warm"]
    if heap_growth > tolerance_bytes:
        problems.append(f"live heap grew by {heap_growth} bytes")
    rss_growth = case["rss_bytes"]["end"] - case["rss_bytes"]["warm"]
    if rss_growth > 8 * 1024 * 1024:
        problems.append(f"RSS grew by {rss_growth} bytes")
    if problems:
        raise SystemExit("audit invariants violated: " + "; ".join(problems))
    case["derived"] = dict(
        heap_growth_bytes=heap_growth,
        rss_growth_bytes=rss_growth,
        pause_fraction=case["pause_events"] / case["ticks"],
        realtime_factor=case["audio_seconds"] / case["wall_seconds"],
    )
    return case


def plot(cases, headline, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)

    ax = left
    series = headline["rss_series"]
    hours = [point[0] / 3600.0 for point in series]
    mib = [point[1] / (1024 * 1024) for point in series]
    ax.plot(hours, mib, color="#1f6feb", lw=1.6)
    warm = headline["rss_bytes"]["warm"] / (1024 * 1024)
    peak = headline["rss_bytes"]["peak"] / (1024 * 1024)
    ax.axhline(warm, color="#2f9e44", ls="--", lw=1.2,
               label=f"post-warm-up baseline {warm:.2f} MiB")
    ax.axhspan(headline["rss_bytes"]["min"] / (1024 * 1024),
               headline["rss_bytes"]["max"] / (1024 * 1024),
               color="#1f6feb", alpha=0.12, lw=0,
               label="observed band after warm-up")
    # Full-scale axis on purpose: the point is that the line is flat, and a
    # zoomed axis would turn ordinary working-set trimming into a cliff.
    ax.set_ylim(0, peak * 1.15)
    growth = (headline["rss_bytes"]["end"] - headline["rss_bytes"]["warm"]) / 1024.0
    ax.text(0.02, 0.94,
            f"warm-up {warm:.2f} MiB → end "
            f"{headline['rss_bytes']['end'] / (1024 * 1024):.2f} MiB "
            f"({growth:+.0f} KiB)\n"
            f"band {headline['rss_bytes']['min'] / (1024 * 1024):.2f}"
            f"–{headline['rss_bytes']['max'] / (1024 * 1024):.2f} MiB, "
            f"start-up peak {peak:.2f} MiB",
            transform=ax.transAxes, va="top", fontsize=8.5, color="#495057",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9fa", ec="#dee2e6"))
    ax.set_xlabel("Continuous audio streamed (hours)")
    ax.set_ylabel("Resident set size (MiB)")
    ax.set_title(f"RSS over {headline['audio_seconds'] / 3600:.1f} h of audio "
                 f"({headline['frames']:,} frames)")
    ax.grid(True, alpha=0.25, ls=":")
    ax.legend(loc="lower right", fontsize=9)

    ax = right
    capacities = [case["config"]["capacity_frames"] for case in cases]
    held = [case["max_frames_held"] for case in cases]
    backlog = [case["max_pcm_backlog_samples"] for case in cases]
    ax.plot(capacities, held, "o-", color="#1f6feb", lw=2, ms=7, label="max frames held")
    ax.plot(capacities, [c / 160.0 for c in backlog], "s--", color="#d1495b", lw=2, ms=7,
            label="max PCM backlog (frames-equivalent)")
    ax.plot(capacities, capacities, ":", color="#868e96", lw=1.2, label="capacity (y=x)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(capacities)
    ax.set_xticklabels([str(c) for c in capacities])
    ax.set_xlabel("Caller-owned output capacity (frames)")
    ax.set_ylabel("Observed high-water mark (frames)")
    ax.set_title("Watermark tracks capacity, not elapsed time")
    ax.grid(True, alpha=0.25, ls=":")
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Bounded Fbank bridge: backpressure flow control and leak-free audit", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path,
                        default=Path("build/win-zig/test_bounded_fbank_bridge.exe"))
    parser.add_argument("--headline-seconds", type=float, default=3600.0)
    parser.add_argument("--capacity-sweep-seconds", type=float, default=60.0)
    parser.add_argument("--capacities", type=int, nargs="+", default=[1, 4, 64])
    parser.add_argument("--chunk", type=int, default=320)
    parser.add_argument("--heap-tolerance", type=int, default=65536)
    parser.add_argument("--json-output", type=Path,
                        default=Path("results/backpressure_soak.json"))
    parser.add_argument("--png-output", type=Path,
                        default=Path("results/backpressure_watermark.png"))
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.exists():
        raise SystemExit(f"missing audit binary: {binary}")
    scratch = Path("build/soak-runs")
    scratch.mkdir(parents=True, exist_ok=True)

    cases = []
    for capacity in args.capacities:
        out = scratch / f"soak_capacity{capacity}.json"
        case = run_case(binary, audio_seconds=args.capacity_sweep_seconds,
                        capacity=capacity, chunk=args.chunk, out_path=out)
        cases.append(check(case, tolerance_bytes=args.heap_tolerance))
        print(f"capacity {capacity:>3}: frames held {case['max_frames_held']}, "
              f"pause fraction {case['derived']['pause_fraction']:.3f}, "
              f"heap delta {case['derived']['heap_growth_bytes']} B")

    headline_path = scratch / "soak_headline.json"
    headline = run_case(binary, audio_seconds=args.headline_seconds, capacity=4,
                        chunk=args.chunk, out_path=headline_path)
    headline = check(headline, tolerance_bytes=args.heap_tolerance)
    print(f"headline: {headline['audio_seconds']:.0f} s audio in "
          f"{headline['wall_seconds']:.1f} s wall, "
          f"{headline['frames']:,} frames, MAE {headline['mean_abs_diff']}, "
          f"heap delta {headline['derived']['heap_growth_bytes']} B")

    report = dict(
        status=STATUS,
        scope="C++ bounded Fbank bridge; consumer throttled to 50% of input rate",
        environment=dict(
            platform=platform.platform(), python=platform.python_version(),
            binary=str(binary), binary_sha256=__import__("hashlib").sha256(
                binary.read_bytes()).hexdigest(),
            binary_compiler=headline["compiler"], binary_platform=headline["platform"],
        ),
        method=dict(
            consumer_rate_ratio=0.5,
            producer_chunk_samples=args.chunk,
            headline_audio_seconds=args.headline_seconds,
            capacity_sweep_audio_seconds=args.capacity_sweep_seconds,
            capacity_sweep=args.capacities,
            exactness_check="every emitted frame compared element-wise against an "
                            "unthrottled baseline fed the identical samples in fixed 10 ms blocks",
            memory_check="live bytes counted at operator new/delete plus process RSS sampled "
                         "throughout; no Valgrind/ASan on this host",
            heap_tolerance_bytes=args.heap_tolerance,
        ),
        headline=headline,
        capacity_sweep=cases,
        caveats=[
            "Processed as fast as the host allows; this is not real-time or device latency.",
            "Windows/MSVC-free build via zig cc; RSS granularity is the OS page size.",
            "Valgrind is Linux-only and AddressSanitizer is unavailable on this Windows host, "
            "so the leak audit uses exact live-byte accounting instead.",
            "Single-threaded harness: OS scheduling and lock contention are out of scope.",
        ],
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    plot(cases, headline, args.png_output)
    print(json.dumps(dict(status=STATUS, json=str(args.json_output),
                          png=str(args.png_output)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
