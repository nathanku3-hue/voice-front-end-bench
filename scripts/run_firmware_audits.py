#!/usr/bin/env python3
"""Run the three firmware-facing C++ audits and publish their artifacts.

Each audit is a separate binary that measures one property and asserts it:

  test_zero_alloc_steady_state   hot path performs zero heap allocations
  test_hysteresis_flow           dual watermark cuts pause/resume handshakes
  fbank_stage_profile            where the per-frame time actually goes, and what
                                 the SIMD + fast-log arms buy, measured interleaved

This runner orchestrates them, re-checks the invariants independently of the
binaries' own asserts, and draws the two figures. It never recomputes a number
the binaries produced; every figure is a rendering of the JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ZERO_ALLOC = "build/win-zig/test_zero_alloc_steady_state.exe"
HYSTERESIS = "build/win-zig/test_hysteresis_flow.exe"
STAGE = "build/win-zig/fbank_stage_profile.exe"
MEL = "build/win-zig/test_mel_sparsity.exe"


def run(cmd: list[str], *, expect_zero: bool = True) -> str:
    completed = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if expect_zero and completed.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed ({completed.returncode})\n"
                         f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    return completed.stdout


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_zero_alloc(report: dict) -> dict:
    problems = []
    if report["window_allocations"] != 0:
        problems.append(f"{report['window_allocations']} allocations in the audited window")
    if report["window_denied_allocations"] != 0:
        problems.append(f"{report['window_denied_allocations']} denied allocations")
    if report["window_live_delta_bytes"] != 0:
        problems.append(f"live heap moved by {report['window_live_delta_bytes']} bytes")
    if report["max_abs_diff"] != 0.0:
        problems.append(f"max |diff| = {report['max_abs_diff']}")
    if not report["probe_control_passed"]:
        problems.append("the probe never observed a deliberate allocation")
    if report["pause_events"] <= 0:
        problems.append("no backpressure in the audited window")
    if problems:
        raise SystemExit("zero-allocation audit violated: " + "; ".join(problems))
    report["derived"] = dict(
        allocations_per_1000_frames=0.0,
        microseconds_per_frame=1e6 * report["wall_seconds"] / report["frames"],
    )
    return report


def check_hysteresis(report: dict) -> dict:
    arms = report["arms"]
    control = arms[0]
    if control["mode"] != "single":
        raise SystemExit("first arm must be the single-watermark control")
    for arm in arms:
        if arm["max_abs_diff"] != 0.0:
            raise SystemExit(f"{arm['mode']} low={arm['low_watermark']}: max |diff| != 0")
        if arm["dropped_frames"] != 0:
            raise SystemExit("frames were dropped: backpressure must never drop")
        if arm["max_occupancy"] > arm["capacity_frames"]:
            raise SystemExit("occupancy exceeded the ring capacity")
        if arm["compared_values"] != arm["frames"] * 80:
            raise SystemExit("not every feature value was compared")
        arm["transition_reduction_fraction"] = (
            1.0 - arm["transitions"] / control["transitions"])
        arm["stall_increase_ticks"] = arm["max_stall_ticks"] - control["max_stall_ticks"]
    return report


def check_stage(report: dict, dense: dict | None = None) -> dict:
    if not report["stages"]:
        raise SystemExit("stage profile has no rows")
    total_share = sum(stage["share_percent"] for stage in report["stages"])
    if abs(total_share - 100.0) > 1e-6:
        raise SystemExit(f"stage shares sum to {total_share}, not 100")
    hottest = max(report["stages"], key=lambda s: s["total_ms"])
    # The gap between the whole pass and the sum of stages is not a bug: it is
    # the six clock reads plus the ring copy and the frame push/pop, which no
    # stage owns. Naming it "outside stages" keeps it from being read as
    # instrumentation error.
    report["derived"] = dict(
        isa=report.get("isa"),
        sample_domain_stages=report.get("sample_domain_stages"),
        log_mode=report.get("log_mode"),
        frame_buffer_offset_mod64=report.get("frame_buffer_offset_mod64"),
        fft_input_offset_mod64=report.get("fft_input_offset_mod64"),
        hottest_stage=hottest["name"],
        hottest_share_percent=hottest["share_percent"],
        outside_stages_percent=100.0 * (report["wall_ms"] - report["instrumented_total_ms"])
        / report["wall_ms"],
    )

    if dense is not None:
        frames = report["frames"]
        sparse_fixed = 1e3 * (report["wall_ms"] - report["instrumented_total_ms"]) / frames
        dense_fixed = 1e3 * (dense["wall_ms"] - dense["instrumented_total_ms"]) / frames
        # The un-instrumented per-frame cost (clock reads, ring copy, queue
        # bookkeeping) is the same code in both modes. If it is not, the two
        # runs are not comparable and the before/after table would be fiction.
        drift = abs(sparse_fixed - dense_fixed) / max(sparse_fixed, dense_fixed)
        if drift > 0.30:
            raise SystemExit(f"fixed per-frame overhead differs by {100 * drift:.0f}% "
                             f"between the dense and sparse runs; comparison is invalid")
        report["dense_reference"] = dict(
            mel_projection=dense.get("mel_projection"),
            stages=dense["stages"],
            wall_ms=dense["wall_ms"],
            wall_us_per_frame=dense["wall_us_per_frame"],
            realtime_factor=dense["realtime_factor"],
            instrumented_total_ms=dense["instrumented_total_ms"],
            wall_spread_ms=dense["wall_spread_ms"],
        )
        report["derived"]["fixed_overhead_us_per_frame"] = dict(
            sparse=sparse_fixed, dense=dense_fixed, relative_drift=drift)
        report["derived"]["mel_stage_ratio"] = (
            next(s["us_per_frame"] for s in dense["stages"] if s["name"] == "mel_projection")
            / next(s["us_per_frame"] for s in report["stages"] if s["name"] == "mel_projection"))
    return report


class HostTooNoisy(RuntimeError):
    """The control stages disagreed, so the arms are not comparable."""


def check_interleave(report: dict, strict: bool = True) -> dict:
    """Verify the three-arm interleaved comparison is internally consistent.

    The point of interleaving is that no arm gets a privileged slice of host
    time: the arms run round-robin inside one process, so a bimodal scheduler or
    a cold FFTW plan cannot be attributed to the change under test. What is
    checked here is only that the arms are comparable at all -- same frame
    count, shares summing to 100, and an ordering that matches the code.
    """
    arms = report["arms"]
    names = [arm["name"] for arm in arms]
    for expected in ("baseline", "simd", "simd_fastlog"):
        if expected not in names:
            raise SystemExit(f"interleaved run is missing the {expected} arm")
    frames = {arm["frames"] if "frames" in arm else report["frames"] for arm in arms}
    if len(frames) != 1:
        raise SystemExit(f"arms measured different frame counts: {sorted(frames)}")
    for arm in arms:
        share = sum(stage["share_percent"] for stage in arm["stages"])
        if abs(share - 100.0) > 1e-6:
            raise SystemExit(f"{arm['name']}: stage shares sum to {share}, not 100")

    by_name = {arm["name"]: arm for arm in arms}
    base, simd, fast = by_name["baseline"], by_name["simd"], by_name["simd_fastlog"]
    if not (base["wall_us_per_frame"] > simd["wall_us_per_frame"] > fast["wall_us_per_frame"]):
        raise SystemExit("interleaved arms are not ordered baseline > simd > simd_fastlog; "
                         "the host was too noisy for this comparison to mean anything")

    def stage_us(arm: dict, name: str) -> float:
        return next(stage["us_per_frame"] for stage in arm["stages"] if stage["name"] == name)

    # Control stages: identical code in the arms named. If they disagree the
    # host moved underneath the run, and any difference in the stages that DID
    # change is unattributable. This is the check that stops a noisy host from
    # being reported as a speedup.
    controls = [
        ("fftw_execute", "baseline", "simd"),
        ("power_spectrum", "baseline", "simd"),
        ("mel_projection", "baseline", "simd"),
        ("log_compress", "baseline", "simd"),            # both arms use std::log
        ("dc_remove", "simd", "simd_fastlog"),           # both arms are vectorised
        ("preemphasis_window", "simd", "simd_fastlog"),
    ]
    drift = {}
    for stage_name, left, right in controls:
        a = stage_us(by_name[left], stage_name)
        b = stage_us(by_name[right], stage_name)
        drift[f"{stage_name}:{left}_vs_{right}"] = 100.0 * abs(a - b) / max((a + b) / 2.0, 1e-12)
    worst_control = max(drift.values())
    if worst_control > 25.0 and strict:
        worst_key = max(drift, key=drift.get)
        raise HostTooNoisy(
            f"control stages disagree by {worst_control:.0f}% ({worst_key}): the host was "
            f"noisy during the interleaved run, so the arm comparison cannot be attributed "
            f"to the code change")

    dc_pre = {arm["name"]: stage_us(arm, "dc_remove") + stage_us(arm, "preemphasis_window")
              for arm in arms}
    log_us = {arm["name"]: stage_us(arm, "log_compress") for arm in arms}

    report["derived"] = dict(
        speedup_simd=base["wall_us_per_frame"] / simd["wall_us_per_frame"],
        speedup_simd_fastlog=base["wall_us_per_frame"] / fast["wall_us_per_frame"],
        sample_domain_us_per_frame=dc_pre,
        sample_domain_reduction_percent={
            name: 100.0 * (1.0 - value / dc_pre["baseline"]) for name, value in dc_pre.items()},
        log_us_per_frame=log_us,
        log_reduction_percent={
            name: 100.0 * (1.0 - value / log_us["baseline"]) for name, value in log_us.items()},
        control_stage_drift_percent=drift,
        worst_control_stage_drift_percent=worst_control,
    )
    return report


def plot_interleave(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = report["arms"]
    derived = report["derived"]
    palette = ["#1f6feb", "#2f9e44", "#d1495b", "#f08c00", "#7048e8", "#0b7285"]
    names = [stage["name"] for stage in arms[0]["stages"]]

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.6, 5.6), constrained_layout=True,
                                      gridspec_kw=dict(width_ratios=[1.35, 1.0]))

    ax = left
    for row, arm in enumerate(arms):
        offset = 0.0
        for index, stage in enumerate(arm["stages"]):
            share = stage["share_percent"]
            ax.barh([row], [share], left=offset, color=palette[index % len(palette)],
                    height=0.55, label=stage["name"] if row == 0 else None)
            if share >= 8.0:
                ax.text(offset + share / 2, row, f"{share:.0f}%", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            offset += share
    labels = [f"{arm['name']}\n{arm['wall_us_per_frame']:.2f} us/frame  "
              f"({arm['realtime_factor']:.0f}x)" for arm in arms]
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of instrumented frame time (%)")
    ax.set_title("Three arms, round-robin inside one process\n"
                 "same frames, same host slice, only the code differs", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=8.5,
              frameon=False)

    ax = right
    groups = ["dc_remove", "preemphasis_window", "log_compress"]
    width = 0.26
    for index, arm in enumerate(arms):
        values = [next(stage["us_per_frame"] for stage in arm["stages"] if stage["name"] == name)
                  for name in groups]
        positions = [i + (index - 1) * width for i in range(len(groups))]
        bars = ax.bar(positions, values, width=width, color=palette[index],
                      label=arm["name"])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.03, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([name.replace("_", "\n") for name in groups], fontsize=9)
    ax.set_ylabel("Microseconds per 10 ms frame")
    ax.set_ylim(0, max(max(next(s["us_per_frame"] for s in arm["stages"]
                                if s["name"] == name) for name in groups) for arm in arms) * 1.42)
    ax.set_title(f"Sample-domain stages: "
                 f"{derived['sample_domain_reduction_percent']['simd']:.0f}% off\n"
                 f"log stage: {derived['log_reduction_percent']['simd_fastlog']:.0f}% off",
                 fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, ls=":")
    ax.legend(fontsize=8.5, frameon=False)

    fig.suptitle("SIMD + fast log: the same frontend, measured three ways", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def check_mel(report: dict) -> dict:
    if report["bit_mismatches"] != 0:
        raise SystemExit(f"{report['bit_mismatches']} features differ between dense and sparse")
    if report["max_abs_diff"] != 0.0:
        raise SystemExit(f"max |diff| = {report['max_abs_diff']}")
    if report["compared_values"] != report["frames"] * 80:
        raise SystemExit("not every feature value was compared")
    if report["nonzeros"] >= report["dense_entries"]:
        raise SystemExit("the compressed table is not smaller than the dense one")
    return report


def plot_mel(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 3.4), constrained_layout=True)
    labels = ["dense\n80 x 257 doubles", "compressed CSR\n501 taps"]
    values = [report["dense_us_per_frame"], report["sparse_us_per_frame"]]
    bars = ax.barh(labels, values, color=["#868e96", "#1f6feb"], height=0.55)
    for bar, value, extra in zip(bars, values,
                                 [f"{report['dense_realtime_factor']:.0f}x realtime",
                                  f"{report['sparse_realtime_factor']:.0f}x realtime"]):
        ax.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f} us/frame  ({extra})", va="center", fontsize=9)
    ax.set_xlim(0, max(values) * 1.55)
    ax.set_xlabel("Whole-pass cost per 10 ms frame (us)")
    ax.set_title(f"Mel projection: {report['nonzeros']:,} non-zero taps of "
                 f"{report['dense_entries']:,} ({100 * report['nonzeros'] / report['dense_entries']:.1f}%), "
                 f"{report['speedup']:.2f}x faster, {report['bit_mismatches']} bit differences")
    ax.grid(True, axis="x", alpha=0.25, ls=":")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_hysteresis(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = report["arms"]
    labels = []
    for arm in arms:
        labels.append(f"single\nlow={arm['low_watermark']}" if arm["mode"] == "single"
                      else f"dual\nlow={arm['low_watermark']}")

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)

    ax = left
    rates = [arm["transitions_per_1000_ticks"] for arm in arms]
    colors = ["#868e96"] + ["#1f6feb"] * (len(arms) - 1)
    bars = ax.bar(range(len(arms)), rates, color=colors, width=0.62)
    for i, (bar, arm) in enumerate(zip(bars, arms)):
        reduction = arm["transition_reduction_fraction"]
        tag = f"{arm['transitions']:,}"
        if i:
            tag += f"\n−{100 * reduction:.0f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02, tag,
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pause + resume transitions per 1,000 ticks")
    ax.set_title(f"Handshake rate, ring depth {arms[0]['capacity_frames']}, "
                 f"{arms[0]['frames']:,} frames")
    ax.grid(True, axis="y", alpha=0.25, ls=":")
    ax.set_ylim(0, max(rates) * 1.22)

    ax = right
    stalls = [arm["max_stall_ticks"] for arm in arms]
    mean_occ = [arm["mean_occupancy"] for arm in arms]
    ax.bar(range(len(arms)), stalls, color="#d1495b", width=0.62, label="longest stall (ticks)")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Longest continuous producer stall (ticks)")
    ax.set_ylim(0, max(stalls) * 1.35)
    for i, value in enumerate(stalls):
        ax.text(i, value + 0.08, str(value), ha="center", va="bottom", fontsize=9)
    twin = ax.twinx()
    twin.plot(range(len(arms)), mean_occ, "o--", color="#2f9e44", lw=1.8, ms=6,
              label="mean occupancy (frames)")
    twin.set_ylabel("Mean ring occupancy (frames)", color="#2f9e44")
    twin.tick_params(axis="y", colors="#2f9e44")
    twin.set_ylim(0, arms[0]["capacity_frames"] * 1.1)
    for i, value in enumerate(mean_occ):
        twin.annotate(f"{value:.2f}", (i, value), textcoords="offset points",
                      xytext=(0, 9), ha="center", fontsize=8, color="#2f9e44",
                      bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_title("The cost: longer stalls, more FIFO headroom")
    ax.grid(True, axis="y", alpha=0.25, ls=":")
    # No legend: both axes are labelled, and a box would sit on the first
    # occupancy annotation.

    fig.suptitle("Dual-watermark hysteresis: fewer handshakes at bounded occupancy",
                 fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_stage(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = report["stages"]
    names = [s["name"] for s in stages]
    shares = [s["share_percent"] for s in stages]
    per_frame = [s["us_per_frame"] for s in stages]
    palette = ["#1f6feb", "#2f9e44", "#d1495b", "#f08c00", "#7048e8", "#0b7285"]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12.6, 7.4), constrained_layout=True,
                                      gridspec_kw=dict(height_ratios=[1, 2.1]))

    ax = top
    rows = []
    if "dense_reference" in report:
        dense = report["dense_reference"]
        rows.append((f"before (dense)\n{dense['wall_us_per_frame']:.1f} us/frame",
                     dense["stages"]))
    rows.append((f"after (sparse)\n{report['wall_us_per_frame']:.1f} us/frame", stages))
    for row, (label, row_stages) in enumerate(rows):
        left = 0.0
        for i, stage in enumerate(row_stages):
            share = stage["share_percent"]
            ax.barh([row], [share], left=left, color=palette[i % len(palette)], height=0.5,
                    label=stage["name"] if row == 0 else None)
            if share >= 5.0:
                ax.text(left + share / 2, row, f"{share:.0f}%", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            left += share
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of instrumented frame time (%)")
    ax.set_title("Same code, two filterbank layouts")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=len(names), fontsize=9,
              frameon=False)

    ax = bottom
    order = sorted(range(len(stages)), key=lambda i: per_frame[i])
    bars = ax.barh([names[i] for i in order], [per_frame[i] for i in order],
                   color=[palette[i % len(palette)] for i in order], height=0.6)
    for bar, i in zip(bars, order):
        ax.text(bar.get_width() * 1.03, bar.get_y() + bar.get_height() / 2,
                f"{per_frame[i]:.3f} us", va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Microseconds per 10 ms frame (log scale)")
    ax.set_title(f"Per-stage cost, {report['realtime_factor']:.0f}x realtime on "
                 f"{report['audio_seconds']:.1f} s of audio")
    ax.grid(True, axis="x", alpha=0.25, ls=":")
    ax.set_xlim(min(per_frame) * 0.4, max(per_frame) * 3.2)

    fig.suptitle("Fbank stage latency budget (instrumented, median of "
                 f"{report['repeats']} passes)", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zero-alloc-frames", type=int, default=10000)
    parser.add_argument("--hysteresis-frames", type=int, default=20000)
    parser.add_argument("--hysteresis-capacity", type=int, default=4)
    parser.add_argument("--stage-seconds", type=float, default=20.0)
    parser.add_argument("--stage-repeats", type=int, default=5)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    for binary in (ZERO_ALLOC, HYSTERESIS, STAGE, MEL):
        if not Path(binary).exists():
            raise SystemExit(f"missing binary: {binary}")

    args.results.mkdir(parents=True, exist_ok=True)
    scratch = Path("build/firmware-audits")
    scratch.mkdir(parents=True, exist_ok=True)

    # 1) zero-allocation steady state -----------------------------------------
    zero_path = scratch / "zero_alloc.json"
    run([ZERO_ALLOC, "--frames", str(args.zero_alloc_frames), "--json", str(zero_path)])
    zero = check_zero_alloc(json.loads(zero_path.read_text(encoding="utf-8")))
    (args.results / "zero_alloc_steady_state.json").write_text(
        json.dumps(zero, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"zero-alloc: {zero['frames']:,} frames, {zero['window_allocations']} allocations, "
          f"{zero['compared_values']:,} values bit-exact, "
          f"{zero['pause_events']:,} pause / {zero['resume_events']:,} resume")

    # 2) hysteresis ------------------------------------------------------------
    hyst_path = scratch / "hysteresis.json"
    run([HYSTERESIS, "--sweep", "--frames", str(args.hysteresis_frames),
         "--capacity", str(args.hysteresis_capacity), "--json", str(hyst_path)])
    hyst = check_hysteresis(json.loads(hyst_path.read_text(encoding="utf-8")))
    hyst["binaries"] = dict(
        zero_alloc_sha256=sha256(Path(ZERO_ALLOC)),
        hysteresis_sha256=sha256(Path(HYSTERESIS)),
        stage_sha256=sha256(Path(STAGE)),
    )
    hyst["environment"] = dict(platform=platform.platform(),
                               python=platform.python_version())
    (args.results / "hysteresis_flow.json").write_text(
        json.dumps(hyst, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for arm in hyst["arms"]:
        print(f"  {arm['mode']:>6} low={arm['low_watermark']}: {arm['transitions']:,} transitions "
              f"({-100 * arm['transition_reduction_fraction']:+.1f}%), "
              f"max stall {arm['max_stall_ticks']}, mean occupancy {arm['mean_occupancy']:.2f}")

    # 3) stage budget ----------------------------------------------------------
    stage_path = scratch / "stage.json"
    run([STAGE, "--seconds", str(args.stage_seconds), "--repeats", str(args.stage_repeats),
         "--json", str(stage_path)])
    dense_path = scratch / "stage_dense.json"
    run([STAGE, "--seconds", str(args.stage_seconds), "--repeats", str(args.stage_repeats),
         "--dense-mel", "--json", str(dense_path)])
    stage = check_stage(json.loads(stage_path.read_text(encoding="utf-8")),
                        json.loads(dense_path.read_text(encoding="utf-8")))
    (args.results / "stage_latency_budget.json").write_text(
        json.dumps(stage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"stage budget: {stage['wall_us_per_frame']:.2f} us/frame, "
          f"{stage['realtime_factor']:.0f}x realtime, hottest "
          f"{stage['derived']['hottest_stage']} at "
          f"{stage['derived']['hottest_share_percent']:.1f}%")

    # 4) the same budget under three configurations, interleaved -------------
    # More rounds is the only honest answer to a noisy host: the median over
    # more passes converges, whereas loosening the control tolerance would just
    # publish an unattributable number. Three attempts, then give up loudly.
    # Each pass at 20 s of audio is only ~8 ms of wall time, so a single
    # scheduler hiccup dominates it. Longer passes average over bursts, which
    # is the fix that actually works; more rounds only average over the gaps.
    inter_seconds = max(args.stage_seconds, 120.0)
    inter_path = scratch / "stage_interleave.json"
    inter = None
    for attempt, rounds in enumerate((9, 15, 25), 1):
        run([STAGE, "--interleave", "--seconds", str(inter_seconds),
             "--repeats", str(rounds), "--json", str(inter_path)])
        candidate = json.loads(inter_path.read_text(encoding="utf-8"))
        try:
            inter = check_interleave(candidate, strict=True)
        except HostTooNoisy as noise:
            print(f"  attempt {attempt} with {rounds} rounds rejected: {noise}")
            inter = check_interleave(candidate, strict=False)
            continue
        break
    if inter is None or inter["derived"]["worst_control_stage_drift_percent"] > 25.0:
        raise SystemExit(
            "interleaved arms remain unattributable after 29 rounds; the host is too busy "
            "to publish this comparison. Re-run when it is idle.")
    (args.results / "stage_interleaved_budget.json").write_text(
        json.dumps(inter, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"interleaved: " + ", ".join(
        f"{arm['name']} {arm['wall_us_per_frame']:.2f} us/frame" for arm in inter["arms"]))
    print(f"  sample domain {inter['derived']['sample_domain_reduction_percent']['simd']:.1f}% off, "
          f"log {inter['derived']['log_reduction_percent']['simd_fastlog']:.1f}% off, "
          f"{inter['derived']['speedup_simd_fastlog']:.2f}x overall, "
          f"worst control-stage drift "
          f"{inter['derived']['worst_control_stage_drift_percent']:.1f}%")

    # 5) mel sparsity: the budget's hot spot, measured before and after ------
    mel_path = scratch / "mel.json"
    run([MEL, "--seconds", str(args.stage_seconds), "--repeats", str(args.stage_repeats),
         "--json", str(mel_path)])
    mel = check_mel(json.loads(mel_path.read_text(encoding="utf-8")))
    (args.results / "mel_sparsity.json").write_text(
        json.dumps(mel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"mel sparsity: {mel['nonzeros']:,} of {mel['dense_entries']:,} taps, "
          f"{mel['dense_us_per_frame']:.2f} -> {mel['sparse_us_per_frame']:.2f} us/frame "
          f"({mel['speedup']:.2f}x), {mel['bit_mismatches']} bit mismatches")

    plot_hysteresis(hyst, args.results / "hysteresis_flow.png")
    plot_interleave(inter, args.results / "stage_simd_arms.png")
    plot_stage(stage, args.results / "stage_latency_budget.png")
    plot_mel(mel, args.results / "mel_sparsity.png")
    print(json.dumps(dict(status="ok",
                          json=[str(args.results / "zero_alloc_steady_state.json"),
                                str(args.results / "hysteresis_flow.json"),
                                str(args.results / "stage_latency_budget.json"),
                                str(args.results / "stage_interleaved_budget.json"),
                                str(args.results / "mel_sparsity.json")],
                          png=[str(args.results / "hysteresis_flow.png"),
                               str(args.results / "stage_latency_budget.png"),
                               str(args.results / "stage_simd_arms.png"),
                               str(args.results / "mel_sparsity.png")]),
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
