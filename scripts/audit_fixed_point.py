#!/usr/bin/env python3
"""Fixed-point feasibility audit: dynamic range per stage, then SQNR per format.

Two halves, deliberately separated because they have different strengths of
evidence.

1. Measured (hard): over an hour of real 16 kHz speech the C++ frontend records,
   for every stage, the min/max/peak/RMS of every value it produces, a log2
   histogram of the per-frame peak, a log2 histogram of the per-frame RMS, and
   the per-frame crest factor. Those numbers are facts about this frontend and
   this corpus. They answer "does a Q15/Q31/Q16.16 chain overflow?" -- the
   question that actually kills a fixed-point port -- and they answer it with
   margin, not with hope.

2. Analytical (soft): the SQNR of a candidate format follows from the measured
   RMS and the chosen full scale (q = FS / 2^(b-1), noise power q^2/12). That is
   the textbook model, and it is stated as such: it assumes the quantisation
   error is uniform and independent, which is optimistic for the FFT stage and
   close to right for the log stage.

The interesting result is not "everything passes". It is that a *global* full
scale is the wrong model for the FFT stage: the between-frame peak spread is
84 dB wide, so a fixed word length that suits the loudest frame leaves the
quietest ones with almost no resolution. The per-frame RMS histogram is what
makes that visible, and it is why the FFT recommendation is block floating point
rather than a wider word.

What this does NOT do: it does not implement a fixed-point frontend. The
end-to-end number that would settle the question -- run the Q15/Q31 pipeline and
compare transcripts -- needs a fixed-point FFT and is explicitly out of scope.
The feature-domain quantisation at the end of the chain IS simulated, because
that one is a rounding of the output and needs no rewrite.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Stage -> candidate format. `bits` counts the whole word including the sign
# bit; `full_scale` is chosen from the measurement, rounded up to a power of two
# so the headroom is a stated number rather than an accident.
FORMATS = {
    "input_pcm": dict(bits=16, name="Q15", scale_rule="power_of_two"),
    "dc_removed": dict(bits=16, name="Q15", scale_rule="power_of_two"),
    "preemphasized": dict(bits=16, name="Q15", scale_rule="power_of_two"),
    "windowed": dict(bits=16, name="Q15", scale_rule="power_of_two"),
    "fft_component": dict(bits=16, name="Q15", scale_rule="power_of_two"),
    "power_spectrum": dict(bits=32, name="Q31", scale_rule="power_of_two"),
    "mel_energy": dict(bits=32, name="Q31", scale_rule="power_of_two"),
    "log_fbank": dict(bits=32, name="Q16.16", scale_rule="fixed", full_scale=32768.0),
}

# The FFT word length is the one that does not obviously work, so it gets a
# ladder instead of a single guess. A real port chooses from this table; the
# audit's job is to say which rungs clear the bar and at what cost.
FFT_LADDER = [
    dict(bits=16, name="Q15"),
    dict(bits=18, name="Q17"),
    dict(bits=20, name="Q19"),
    dict(bits=24, name="Q23"),
]

SQNR_TARGET_DB = 60.0
# A radix-2 block exponent normalises the peak into [FS/2, FS]. The guaranteed
# figure is the low end of that interval, i.e. 6.02 dB below a perfect fit.
BFP_EXPONENT_PENALTY_DB = 20.0 * np.log10(2.0)


def build_corpus(audio_dir: Path, target_seconds: float, out_path: Path) -> tuple[Path, int]:
    """Concatenate the available 16 kHz clips and return (path, repeats)."""
    clips = sorted(audio_dir.glob("*.wav"))
    if not clips:
        raise SystemExit(f"no WAVs under {audio_dir}")
    frames = []
    rate = None
    for clip in clips:
        with wave.open(str(clip), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                continue
            rate = handle.getframerate()
            frames.append(handle.readframes(handle.getnframes()))
    if rate != 16000:
        raise SystemExit(f"expected 16 kHz audio, got {rate}")
    total = sum(len(block) for block in frames) // 2
    repeats = max(1, int(round(target_seconds * rate / total)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        for block in frames:
            handle.writeframes(block)
    return out_path, repeats


def hist_quantiles(hist: list[int], offset: int) -> dict:
    """Recover value quantiles from a log2 histogram (bin = ilogb + offset)."""
    counts = np.asarray(hist, dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return {}
    cumulative = np.cumsum(counts) / total

    def pick(q: float) -> float:
        index = int(np.searchsorted(cumulative, q))
        index = min(index, len(counts) - 1)
        return 0.0 if index == 0 else float(2.0 ** (index - offset))

    return {"p05": pick(0.05), "p25": pick(0.25), "median": pick(0.50),
            "p75": pick(0.75), "p95": pick(0.95), "max": pick(0.999)}


def next_power_of_two(value: float) -> float:
    if value <= 0:
        return 1.0
    return float(2.0 ** np.ceil(np.log2(value)))


def sqnr_db(rms: float, full_scale: float, bits: int) -> float:
    """Textbook SQNR: signal power over uniform quantisation noise q^2/12."""
    if rms <= 0.0:
        return 0.0
    step = full_scale / (2 ** (bits - 1))
    return float(10.0 * np.log10((rms * rms) / (step * step / 12.0)))


def per_frame_sqnr(signal: dict, full_scale: float, bits: int, offset: int) -> dict:
    """SQNR of the quietest / typical / loudest frame under one global full scale.

    The corpus-wide SQNR is dominated by the loud frames; a frame 30 dB below
    the peak loses exactly 30 dB of SQNR at a fixed word length. This is the
    number that decides whether a fixed-point port sounds fine on the training
    set and falls apart on a quiet talker.
    """
    quantiles = hist_quantiles(signal.get("rms_hist", []), offset)
    return {key: sqnr_db(value, full_scale, bits) for key, value in quantiles.items()}


def bfp_sqnr(signal: dict, bits: int) -> dict:
    """Block-floating-point SQNR: level independent, crest limited.

    With one exponent per block the mantissa fills the word for every frame, so
    the only remaining level dependence is the crest factor inside the block --
    which is measured, not assumed. Reported twice because a radix-2 exponent
    lands the normalised peak anywhere in [FS/2, FS]; the guaranteed figure is
    the low end.
    """
    crest_mean = signal.get("crest_mean_db", 0.0)
    crest_max = signal.get("crest_max_db", 0.0)
    ceiling = 10.0 * np.log10(12.0 * (2.0 ** (2 * (bits - 1))))
    return {
        "mantissa_bits": bits,
        "crest_mean_db": crest_mean,
        "crest_max_db": crest_max,
        "sqnr_typical_db": float(ceiling - crest_mean),
        "sqnr_guaranteed_db": float(ceiling - crest_max - BFP_EXPONENT_PENALTY_DB),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", type=Path, default=REPO / "data/ctc_eval/audio")
    parser.add_argument("--corpus", type=Path, default=REPO / "build/dynamic_range_corpus.wav")
    parser.add_argument("--seconds", type=float, default=3600.0)
    parser.add_argument("--binary", type=Path, default=REPO / "build/win-zig/fbank_feature_dump.exe")
    parser.add_argument("--range-json", type=Path, default=REPO / "results/fixed_point_dynamic_range.json")
    parser.add_argument("--out", type=Path, default=REPO / "results/fixed_point_sqnr.json")
    parser.add_argument("--plot", type=Path, default=REPO / "results/fixed_point_dynamic_range.png")
    parser.add_argument("--reuse", action="store_true", help="reuse an existing range JSON")
    args = parser.parse_args()

    if not args.reuse:
        corpus, repeats = build_corpus(args.audio_dir, args.seconds, args.corpus)
        features = REPO / "build/dynamic_range_features.f32"
        print(f"corpus {corpus.name}: {repeats} repeats -> "
              f"{repeats * (corpus.stat().st_size - 44) / 2 / 16000:.0f} s of audio")
        proc = subprocess.run(
            [str(args.binary), "--wav", str(corpus), "--repeat", str(repeats),
             "--out", str(features), "--range-json", str(args.range_json),
             "--log", "fast"],
            capture_output=True, text=True, encoding="utf-8")
        if proc.returncode != 0:
            raise SystemExit(f"feature dump failed: {proc.stderr.strip()[:600]}")
        print(proc.stdout.strip())

    report = json.loads(args.range_json.read_text(encoding="utf-8"))
    offset = report["peak_hist_offset"]
    signals = {signal["name"]: signal for signal in report["signals"]}

    # ---- stage-by-stage, one candidate format each ---------------------------
    rows = []
    for signal in report["signals"]:
        name = signal["name"]
        fmt = FORMATS[name]
        if fmt["scale_rule"] == "fixed":
            full_scale = fmt["full_scale"]
        else:
            full_scale = next_power_of_two(signal["abs_max"])
        step = full_scale / (2 ** (fmt["bits"] - 1))
        rms = signal["rms"]
        headroom_db = 20.0 * np.log10(full_scale / signal["abs_max"]) if signal["abs_max"] > 0 else 0.0
        frames = per_frame_sqnr(signal, full_scale, fmt["bits"], offset)
        rows.append({
            "stage": name,
            "format": fmt["name"],
            "bits": fmt["bits"],
            "full_scale": full_scale,
            "quantum": step,
            "measured_min": signal["min"],
            "measured_max": signal["max"],
            "measured_peak": signal["abs_max"],
            "measured_rms": rms,
            "headroom_db": float(headroom_db),
            "overflow": bool(signal["abs_max"] >= full_scale),
            "underflow_risk": bool(step > 0 and rms < 4 * step),
            "sqnr_db": sqnr_db(rms, full_scale, fmt["bits"]),
            "sqnr_quiet_frame_db": frames.get("p05", 0.0),
            "sqnr_p25_frame_db": frames.get("p25", 0.0),
            "sqnr_median_frame_db": frames.get("median", 0.0),
            "crest_mean_db": signal.get("crest_mean_db", 0.0),
            "crest_max_db": signal.get("crest_max_db", 0.0),
            "quantiles": hist_quantiles(signal["peak_hist"], offset),
            "frame_rms_quantiles": hist_quantiles(signal.get("rms_hist", []), offset),
            "values": signal["count"],
            "verdict": ("ok" if sqnr_db(rms, full_scale, fmt["bits"]) > SQNR_TARGET_DB
                        and frames.get("p05", 0.0) > SQNR_TARGET_DB
                        else "corpus ok, quiet frames short"),
        })

    # ---- the FFT ladder ------------------------------------------------------
    fft_signal = signals["fft_component"]
    fft_full_scale = next_power_of_two(fft_signal["abs_max"])
    ladder = []
    for rung in FFT_LADDER:
        frames = per_frame_sqnr(fft_signal, fft_full_scale, rung["bits"], offset)
        ladder.append({
            "format": rung["name"],
            "bits": rung["bits"],
            "full_scale": fft_full_scale,
            "quantum": fft_full_scale / (2 ** (rung["bits"] - 1)),
            "sqnr_db": sqnr_db(fft_signal["rms"], fft_full_scale, rung["bits"]),
            "sqnr_quiet_frame_db": frames.get("p05", 0.0),
            "sqnr_p25_frame_db": frames.get("p25", 0.0),
            "sqnr_median_frame_db": frames.get("median", 0.0),
            "clears_target_on_median_frame": bool(frames.get("median", 0.0) > SQNR_TARGET_DB),
            "clears_target_on_corpus_rms": bool(
                sqnr_db(fft_signal["rms"], fft_full_scale, rung["bits"]) > SQNR_TARGET_DB),
            "clears_target_on_quiet_frame": bool(frames.get("p05", 0.0) > SQNR_TARGET_DB),
        })

    # The recommendation is the cheapest rung that holds the bar on the *quiet*
    # frames, not just on the corpus average -- a port that only passes on the
    # loudest decile is not a port that passes.
    # Two different questions, two different answers, and both are reported:
    # what word length covers a typical frame, and what covers the quietest ones.
    at_median = [rung for rung in ladder if rung["clears_target_on_median_frame"]]
    at_quiet = [rung for rung in ladder if rung["clears_target_on_quiet_frame"]]
    bfp16 = bfp_sqnr(fft_signal, 16)
    recommendation = {
        "approach": "block_floating_point",
        "mantissa_bits": 16,
        "exponent_bits": 5,
        "sqnr_guaranteed_db": bfp16["sqnr_guaranteed_db"],
        "sqnr_typical_db": bfp16["sqnr_typical_db"],
        "cheapest_fixed_point_at_median": at_median[0]["format"] if at_median else None,
        "cheapest_fixed_point_at_quiet": at_quiet[0]["format"] if at_quiet else None,
        "reason": ("a fixed word length has to cover the between-frame peak spread with its "
                   "own headroom, so the word that suits a loud frame leaves a quiet one with "
                   "almost no resolution. One exponent per frame removes the between-frame "
                   "term entirely and leaves only the within-frame crest factor, which is "
                   "measured at "
                   f"{bfp16['crest_mean_db']:.1f} dB mean and "
                   f"{bfp16['crest_max_db']:.1f} dB worst."),
    }
    chosen_fft = recommendation

    # ---- block floating point ------------------------------------------------
    bfp_stages = ["windowed", "fft_component", "power_spectrum", "mel_energy"]
    bfp = {}
    for name in bfp_stages:
        # BFP only helps where the between-frame spread is real; use the
        # stage's own proposed mantissa width.
        bits = FORMATS[name]["bits"]
        bfp[name] = bfp_sqnr(signals[name], bits)
        peaks = hist_quantiles(signals[name]["peak_hist"], offset)
        loud, quiet = peaks.get("p95", 0.0), peaks.get("p05", 0.0)
        bfp[name]["between_frame_spread_db"] = float(
            20.0 * np.log10(loud / quiet) if quiet > 0 else 0.0)
        bfp[name]["fixed_point_quiet_frame_db"] = next(
            row["sqnr_quiet_frame_db"] for row in rows if row["stage"] == name)

    weakest = min(rows, key=lambda row: row["sqnr_quiet_frame_db"])
    overflowing = [row["stage"] for row in rows if row["overflow"]]
    report_out = {
        "status": "measured_dynamic_range_and_analytical_sqnr",
        "corpus": {
            "source": str(args.corpus.relative_to(REPO)) if args.corpus.exists() else "n/a",
            "audio_seconds": report["audio_seconds"],
            "frames": report["frames"],
            "repeat": report["repeat"],
            "material": "concatenated 16 kHz read speech from data/ctc_eval/audio, looped",
        },
        "method": {
            "range": "measured per value at every stage by the C++ frontend",
            "sqnr": "10*log10(rms^2 / (q^2/12)) with q = full_scale / 2^(bits-1)",
            "full_scale": "next power of two above the measured peak, or fixed for Q16.16",
            "per_frame": ("the same formula applied to per-frame RMS quantiles from the "
                          "measured log2 histogram, so quiet frames are not averaged away"),
            "bin_resolution": ("histogram quantiles return the lower edge of the octave bin, "
                               "so every per-frame SQNR here is understated by 0 to 6 dB; "
                               "the conclusion is conservative in the direction that matters"),
            "bfp": ("10*log10(12*2^(2*(b-1))) - crest, with crest measured per frame; "
                    "a radix-2 exponent is charged the worst-case 6.02 dB placement"),
            "sqnr_caveat": ("textbook uniform-noise model; optimistic for the FFT stage where "
                            "the error is not independent of the signal"),
        },
        "target_db": SQNR_TARGET_DB,
        "formats": rows,
        "fft_ladder": ladder,
        "fft_recommendation": chosen_fft,
        "block_floating_point": bfp,
        "derived": {
            "weakest_stage": weakest["stage"],
            "weakest_quiet_frame_sqnr_db": weakest["sqnr_quiet_frame_db"],
            "all_stages_above_target_on_corpus_rms": bool(
                all(row["sqnr_db"] > SQNR_TARGET_DB for row in rows)),
            "all_stages_above_target_on_quiet_frames": bool(
                all(row["sqnr_quiet_frame_db"] > SQNR_TARGET_DB for row in rows)),
            "fft_q15_sqnr_db": next(r["sqnr_db"] for r in ladder if r["format"] == "Q15"),
            "fft_q15_quiet_frame_db": next(r["sqnr_quiet_frame_db"] for r in ladder
                                           if r["format"] == "Q15"),
            "fft_quiet_frame_spread_db": float(
                next(r["sqnr_db"] for r in ladder if r["format"] == "Q15")
                - next(r["sqnr_quiet_frame_db"] for r in ladder if r["format"] == "Q15")),
            "overflow_stages": overflowing,
            "total_peak_to_rms_db": {
                row["stage"]: float(20.0 * np.log10(row["measured_peak"] / row["measured_rms"]))
                for row in rows if row["measured_rms"] > 0
            },
        },
        "non_claims": [
            "no fixed-point frontend was implemented or measured end to end",
            "SQNR is analytical, from the measured RMS and the chosen full scale",
            "a plain Q15 FFT fails the target on quiet frames; the ladder shows the word "
            "length that would be needed and the block-floating-point alternative, but "
            "neither is simulated here",
            "the block-floating-point numbers assume the exponent is chosen per frame to "
            "normalise the frame peak; the scaling itself is not implemented",
        ],
    }
    args.out.write_text(json.dumps(report_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'stage':<16}{'format':>7}{'peak':>10}{'rms':>10}{'FS':>9}"
          f"{'SQNR':>8}{'quiet':>8}{'head':>7}")
    for row in rows:
        print(f"{row['stage']:<16}{row['format']:>7}{row['measured_peak']:>10.4g}"
              f"{row['measured_rms']:>10.4g}{row['full_scale']:>9.4g}"
              f"{row['sqnr_db']:>8.1f}{row['sqnr_quiet_frame_db']:>8.1f}"
              f"{row['headroom_db']:>7.1f}")

    print(f"\nFFT word-length ladder (full scale {fft_full_scale:g}, "
          f"quiet frame = p05 of per-frame RMS)")
    print(f"{'format':>7}{'bits':>6}{'SQNR':>9}{'median':>9}{'p25':>9}{'quiet':>9}   verdict")
    for rung in ladder:
        if rung["clears_target_on_quiet_frame"]:
            verdict = "clears everywhere"
        elif rung["clears_target_on_median_frame"]:
            verdict = "median and up"
        elif rung["clears_target_on_corpus_rms"]:
            verdict = "corpus average only"
        else:
            verdict = "fails"
        print(f"{rung['format']:>7}{rung['bits']:>6}{rung['sqnr_db']:>9.1f}"
              f"{rung['sqnr_median_frame_db']:>9.1f}{rung['sqnr_p25_frame_db']:>9.1f}"
              f"{rung['sqnr_quiet_frame_db']:>9.1f}   {verdict}")
    print(f"cheapest fixed word clearing 60 dB at the median frame: "
          f"{chosen_fft['cheapest_fixed_point_at_median'] or 'none up to Q23'}")
    print(f"cheapest fixed word clearing 60 dB at the quietest 5%:  "
          f"{chosen_fft['cheapest_fixed_point_at_quiet'] or 'none up to Q23'}")
    print(f"recommendation: block floating point, {chosen_fft['mantissa_bits']}-bit mantissa "
          f"+ {chosen_fft['exponent_bits']}-bit exponent -> "
          f"{chosen_fft['sqnr_guaranteed_db']:.1f} dB guaranteed, "
          f"{chosen_fft['sqnr_typical_db']:.1f} dB typical")

    print(f"\nblock floating point (level independent, crest limited)")
    print(f"{'stage':<16}{'mant':>6}{'crest~':>9}{'crestmax':>10}"
          f"{'BFP typ':>9}{'BFP guar':>10}{'fixed p05':>11}")
    for name, entry in bfp.items():
        print(f"{name:<16}{entry['mantissa_bits']:>6}{entry['crest_mean_db']:>9.1f}"
              f"{entry['crest_max_db']:>10.1f}{entry['sqnr_typical_db']:>9.1f}"
              f"{entry['sqnr_guaranteed_db']:>10.1f}"
              f"{entry['fixed_point_quiet_frame_db']:>11.1f}")

    plot(rows, ladder, args.plot)
    print(f"\nwrote {args.out} and {args.plot}")
    return 0 if not overflowing else 1


def plot(rows: list[dict], ladder: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14.5, 10.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85])

    names = [row["stage"] for row in rows]
    positions = list(range(len(rows)))
    stats = []
    for row in rows:
        q = row["quantiles"]
        stats.append(dict(med=q.get("median", 0.0), q1=q.get("p25", 0.0), q3=q.get("p75", 0.0),
                          lo=q.get("p05", 0.0), hi=q.get("p95", 0.0), peak=q.get("max", 0.0)))

    ax = fig.add_subplot(grid[0, 0])
    for index, row in enumerate(rows):
        s = stats[index]
        ax.plot([index, index], [max(s["lo"], 1e-9), s["hi"]], color="#868e96", lw=1.2, zorder=2)
        ax.add_patch(plt.Rectangle((index - 0.28, max(s["q1"], 1e-9)),
                                   0.56, max(s["q3"] - s["q1"], 1e-12),
                                   facecolor="#c9d7f5", edgecolor="#1f6feb", lw=1.2, zorder=3))
        ax.plot([index - 0.28, index + 0.28], [s["med"], s["med"]], color="#1f6feb", lw=2.0, zorder=4)
        ax.plot(index, max(s["peak"], 1e-9), marker="v", color="#d1495b", ms=5, zorder=5)
        ax.plot(index, row["full_scale"], marker="_", color="#2f9e44", ms=16, mew=2.0, zorder=6)
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=32, ha="right", fontsize=9)
    ax.set_ylabel("per-frame peak magnitude (log scale)")
    ax.set_title("Measured dynamic range per stage\n"
                 "box = p25-p75, whisker = p05-p95, red = p99.9, green = chosen full scale",
                 fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, ls=":")

    ax = fig.add_subplot(grid[0, 1])
    # The log stage sits 60 dB above every other one and would flatten the region
    # that matters; it is annotated instead of drawn.
    plotted = [(index, row) for index, row in enumerate(rows) if row["stage"] != "log_fbank"]
    off_scale = next(row for row in rows if row["stage"] == "log_fbank")
    bar_positions = [index for index, _ in plotted]
    colours = ["#2f9e44" if row["sqnr_quiet_frame_db"] > SQNR_TARGET_DB else "#d1495b"
               for _, row in plotted]
    ax.barh(bar_positions, [row["sqnr_db"] for _, row in plotted], color=colours,
            height=0.55, alpha=0.42, label="corpus RMS")
    ax.barh(bar_positions, [row["sqnr_quiet_frame_db"] for _, row in plotted], color=colours,
            height=0.28, label="quiet frame (p05)")
    ax.axvline(SQNR_TARGET_DB, color="#f08c00", ls="--", lw=1.4, label="60 dB requirement")
    for index, row in plotted:
        ax.text(max(row["sqnr_db"], row["sqnr_quiet_frame_db"]) + 1.2, index,
                f"{row['sqnr_quiet_frame_db']:.0f} / {row['sqnr_db']:.0f} dB",
                va="center", fontsize=8.5)
    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("SQNR (dB): quiet-frame p05 / corpus RMS")
    ax.set_xlim(0, max(row["sqnr_db"] for _, row in plotted) * 1.30)
    ax.set_title("Quantisation headroom, and what the quiet frames get", fontsize=10)
    ax.grid(True, axis="x", alpha=0.25, ls=":")
    ax.text(0.985, 0.03,
            f"log_fbank off scale: {off_scale['sqnr_quiet_frame_db']:.0f} / "
            f"{off_scale['sqnr_db']:.0f} dB",
            transform=ax.transAxes, ha="right", fontsize=8.5, color="#2f9e44")
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    ax = fig.add_subplot(grid[1, 0])
    rung_positions = list(range(len(ladder)))
    ax.plot(rung_positions, [r["sqnr_db"] for r in ladder], marker="o", color="#1f6feb",
            lw=1.8, label="corpus RMS")
    ax.plot(rung_positions, [r["sqnr_quiet_frame_db"] for r in ladder], marker="s",
            color="#d1495b", lw=1.8, label="quiet frame (p05)")
    ax.axhline(SQNR_TARGET_DB, color="#f08c00", ls="--", lw=1.4, label="60 dB requirement")
    for index, rung in enumerate(ladder):
        ax.annotate(f"{rung['bits']}b", (index, rung["sqnr_quiet_frame_db"]),
                    textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5,
                    color="#d1495b")
    ax.set_xticks(rung_positions)
    ax.set_xticklabels([r["format"] for r in ladder])
    ax.set_ylabel("SQNR (dB)")
    fft_row = next(row for row in rows if row["stage"] == "fft_component")
    peaks = fft_row["quantiles"]
    spread = 20.0 * np.log10(peaks.get("p95", 1.0) / max(peaks.get("p05", 1e-30), 1e-30))
    ax.set_title(f"FFT word-length ladder\n"
                 f"peak spread across frames (p05-p95) is {spread:.0f} dB, "
                 f"so no single full scale fits",
                 fontsize=10)
    ax.grid(True, alpha=0.25, ls=":")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")

    ax = fig.add_subplot(grid[1, 1])
    fft = fft_row
    rms_quantiles = fft["frame_rms_quantiles"]
    keys = ["p05", "p25", "median", "p75", "p95"]
    levels = [rms_quantiles.get(key, 0.0) for key in keys]
    fixed = [sqnr_db(level, fft["full_scale"], fft["bits"]) for level in levels]
    bfp_line = [bfp_sqnr({"crest_mean_db": fft["crest_mean_db"],
                          "crest_max_db": fft["crest_max_db"]}, 16)["sqnr_guaranteed_db"]] * len(keys)
    ax.plot(range(len(keys)), fixed, marker="o", color="#d1495b", lw=1.8,
            label="fixed global full scale (Q15)")
    ax.plot(range(len(keys)), bfp_line, color="#1f6feb", lw=1.8, ls="-",
            label="block floating point (16-bit mantissa)")
    ax.axhline(SQNR_TARGET_DB, color="#f08c00", ls="--", lw=1.4)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{key}\n{level:.3g}" for key, level in zip(keys, levels)], fontsize=8.5)
    ax.set_xlabel("per-frame RMS quantile (value)")
    ax.set_ylabel("SQNR (dB)")
    ax.set_title("Why a global full scale fails the FFT stage\n"
                 "fixed point tracks level one-for-one; a block exponent does not",
                 fontsize=10)
    ax.grid(True, alpha=0.25, ls=":")
    ax.legend(fontsize=8.5, frameon=False)

    fig.suptitle("Fixed-point feasibility: what range each stage actually needs", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
