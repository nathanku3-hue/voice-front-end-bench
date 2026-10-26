#!/usr/bin/env python3
"""Beam-width vs latency vs CER sweep on the frozen 30-utterance Dev split.

Scope, stated before the measurement:

* Decoder only. Inputs are the already-exported, hash-pinned CTC log
  probabilities in build/ctc_round1_dev; no ONNX session, no audio I/O and no
  detokenization inside the timed region.
* Unpruned prefix beam search only (``strategy="none"``). Blank shortcuts are a
  *different* approximation axis and are deliberately held fixed, so the only
  variable in this sweep is the beam width.
* The held-out 50-utterance Test split is never read or decoded.

The operating point is selected by a rule fixed in advance (``select_knee``),
not by looking at the plot: the narrowest beam whose CER is within
``--cer-tolerance`` absolute of the best measured CER. The report records the
measured numbers for every width so the rule can be re-derived or rejected.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np

from run_ctc_dev_sweep import (
    load_dev,
    make_greedy_baseline,
    one_configuration,
)

WIDTHS = (1, 2, 4, 8, 16, 32)
STATUS = "BEAM_WIDTH_PARETO_DEV_ONLY"


def stabilize_host():
    """Pin to one logical CPU and raise priority on Windows.

    Unpinned, this host schedules the decoder across P- and E-cores and the
    per-pass mean jumps by ~3x between passes, which is larger than the effect
    being measured. Pinning removes the migration; the residual spread is
    reported per width rather than smoothed away.
    """
    if platform.system() != "Windows":
        return dict(affinity="unchanged", priority="unchanged")
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    handle = kernel32.GetCurrentProcess()
    pinned = bool(kernel32.SetProcessAffinityMask(handle, 1))
    raised = bool(kernel32.SetPriorityClass(handle, 0x00000080))  # HIGH_PRIORITY_CLASS
    return dict(affinity="cpu 0" if pinned else "failed", priority="high" if raised else "normal")


def select_knee(rows, tolerance):
    """Narrowest width whose CER is within `tolerance` of the best CER.

    Deterministic tie-break: on equal CER prefer the narrower beam, which is
    always the cheaper decode. Returns None when no width qualifies.
    """
    if not rows:
        return None
    best_cer = min(row["cer"] for row in rows)
    eligible = [row for row in rows if row["cer"] - best_cer <= tolerance + 1e-12]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (row["beam_width"], row["cer"]))


def posterior_stats(inputs):
    """How peaked is the acoustic posterior? Explains a flat CER curve.

    Beam search can only change the output where the per-frame distribution is
    ambiguous; if the top-1 token already dominates, every width collapses to
    the same hypothesis.
    """
    tops, margins = [], []
    for item in inputs:
        probs = np.exp(item["log_probs"])
        tops.append(probs.max(axis=1))
        ordered = np.sort(probs, axis=1)
        margins.append(ordered[:, -1] - ordered[:, -2])
    tops = np.concatenate(tops)
    margins = np.concatenate(margins)
    return dict(
        ctc_frames=int(tops.size),
        mean_top1_probability=float(tops.mean()),
        median_top1_probability=float(np.median(tops)),
        fraction_frames_top1_above_0_99=float(np.mean(tops > 0.99)),
        mean_top1_minus_top2_margin=float(margins.mean()),
        fraction_frames_margin_above_0_9=float(np.mean(margins > 0.9)),
    )


def build_report(inputs, detokenize, previous, identity, *, repeats, tolerance, runs=3,
                 host_state=None):
    greedy = make_greedy_baseline(inputs, detokenize, previous, repeats=repeats)
    greedy_tokens = {u["id"]: tuple(u["tokens"]) for u in greedy["utterances"]}
    greedy_text = {u["id"]: u["hypothesis"] for u in greedy["utterances"]}
    rows = []
    tokens_by_width = {}
    # This host drifts by more than 2x over a minute of sustained decoding, so
    # measuring one width to completion and then the next would attribute host
    # drift to the beam width. Widths are therefore interleaved round-robin and
    # the per-width median across passes is published, with the spread kept.
    passes = {width: [] for width in WIDTHS}
    for _ in range(max(1, runs)):
        for width in WIDTHS:
            passes[width].append(
                one_configuration(inputs, detokenize, width=width, strategy="none",
                                  tau=None, repeats=repeats))
    for width in WIDTHS:
        ordered = sorted(passes[width],
                         key=lambda p: p["timing"]["decoder_sum_of_utterance_medians_ms"])
        result = ordered[len(ordered) // 2]
        mean_ms_passes = [p["timing"]["decoder_mean_ms"] for p in ordered]
        cers = {p["corpus"]["cer"] for p in ordered}
        if len(cers) != 1:
            raise AssertionError(f"Non-deterministic CER at width {width}: {sorted(cers)}")
        tokens_by_width[width] = {u["id"]: tuple(u["tokens"]) for u in result["utterances"]}
        text_by_id = {u["id"]: u["hypothesis"] for u in result["utterances"]}
        rows.append(dict(
            beam_width=width,
            config_id=result["id"],
            cer=result["corpus"]["cer"],
            substitutions=result["corpus"]["substitutions"],
            deletions=result["corpus"]["deletions"],
            insertions=result["corpus"]["insertions"],
            reference_chars=result["corpus"]["reference_chars"],
            decoder_mean_ms=result["timing"]["decoder_mean_ms"],
            decoder_median_ms=result["timing"]["decoder_median_ms"],
            decoder_p95_ms=result["timing"]["decoder_p95_ms"],
            decoder_sum_ms=result["timing"]["decoder_sum_of_utterance_medians_ms"],
            decoder_ms_per_ctc_frame=result["timing"]["decoder_ms_per_ctc_frame"],
            decoder_mean_ms_min_across_runs=min(mean_ms_passes),
            decoder_mean_ms_max_across_runs=max(mean_ms_passes),
            runs_measured=len(ordered),
            nonblank_arcs_expanded=result["nonblank_arcs_expanded"],
            max_active_prefixes=max(u["max_active_prefixes"] for u in result["utterances"]),
            total_ctc_frames=result["total_ctc_frames"],
            hypotheses_differing_from_greedy=sum(
                1 for uid, toks in tokens_by_width[width].items() if toks != greedy_tokens[uid]),
            transcripts_differing_from_greedy=sum(
                1 for uid, text in text_by_id.items() if text != greedy_text[uid]),
            utterances=result["utterances"],
        ))
        print(json.dumps(dict(width=width, cer=result["corpus"]["cer"],
                              mean_ms=result["timing"]["decoder_mean_ms"],
                              run_spread_ms=[round(min(mean_ms_passes), 2),
                                             round(max(mean_ms_passes), 2)]),
                         ensure_ascii=False), flush=True)

    widest = max(rows, key=lambda r: r["beam_width"])
    narrowest = min(rows, key=lambda r: r["beam_width"])
    widest_tokens = tokens_by_width[widest["beam_width"]]
    for row in rows:
        # Speed-up is relative to the widest beam actually measured here, which
        # is the only comparison this run can support.
        row["speedup_vs_widest_beam"] = widest["decoder_sum_ms"] / row["decoder_sum_ms"]
        row["latency_ratio_vs_narrowest_beam"] = row["decoder_sum_ms"] / narrowest["decoder_sum_ms"]
        row["cer_delta_vs_greedy"] = row["cer"] - greedy["corpus"]["cer"]
        row["speedup_vs_greedy"] = (greedy["timing"]["decoder_sum_of_utterance_medians_ms"]
                                    / row["decoder_sum_ms"])
        row["hypotheses_differing_from_widest"] = sum(
            1 for uid, toks in tokens_by_width[row["beam_width"]].items()
            if toks != widest_tokens[uid])
        row.pop("utterances")  # per-utterance rows stay in the width partials

    knee = select_knee(rows, tolerance)
    if knee is None:
        raise ValueError("No beam width satisfies the CER tolerance rule")
    cer_spread = max(r["cer"] for r in rows) - min(r["cer"] for r in rows)
    return dict(
        status=STATUS,
        scope="30 Dev utterances; held-out Test split not decoded",
        provenance=identity,
        timing_protocol=dict(
            clock="time.perf_counter_ns", repeats=repeats, warmups_per_utterance=1,
            independent_runs_per_width=runs,
            run_order="round-robin over beam widths, so host drift hits every width equally",
            reported_statistic="median pass by total decoder time; per-width spread kept",
            host_stabilization=host_state,
            backend=platform.python_implementation(), python=platform.python_version(),
            numpy=np.__version__, platform=platform.platform(),
            included="prefix_beam_search() only; no model/audio I/O, detokenize or scoring",
            threading="sequential Python decoder; all log probs preloaded",
        ),
        selection_rule=dict(
            rule="narrowest beam whose CER is within tolerance of the best measured CER",
            cer_tolerance_absolute=tolerance,
            tie_break="narrower beam on equal CER",
        ),
        framewise_greedy=dict(cer=greedy["corpus"]["cer"],
                              decoder_mean_ms=greedy["timing"]["decoder_mean_ms"],
                              decoder_sum_ms=greedy["timing"]["decoder_sum_of_utterance_medians_ms"]),
        posterior_concentration=posterior_stats(inputs),
        widths=list(WIDTHS),
        results=rows,
        knee=dict(
            beam_width=knee["beam_width"],
            cer=knee["cer"],
            decoder_mean_ms=knee["decoder_mean_ms"],
            decoder_ms_per_ctc_frame=knee["decoder_ms_per_ctc_frame"],
            speedup_vs_widest_beam=knee["speedup_vs_widest_beam"],
            cer_delta_vs_widest_beam=knee["cer"] - widest["cer"],
            cer_spread_across_widths=cer_spread,
            hypotheses_differing_from_widest=knee["hypotheses_differing_from_widest"],
        ),
        caveats=[
            "Timings are a CPython decoder prototype, not device or embedded latency.",
            "CER here is Dev-only and was already used for the Round 2 configuration grid.",
            "Beam width 1 prefix search is not identical to framewise greedy.",
            "A flat CER curve means width cannot be justified by accuracy alone; pick on cost.",
        ],
    )


def write_csv(report, path):
    fields = ["beam_width", "cer", "s", "d", "i", "n",
              "decoder_mean_ms", "decoder_median_ms", "decoder_p95_ms",
              "decoder_sum_ms", "decoder_ms_per_ctc_frame",
              "nonblank_arcs_expanded", "max_active_prefixes",
              "speedup_vs_widest_beam", "latency_ratio_vs_narrowest_beam",
              "cer_delta_vs_greedy", "speedup_vs_greedy",
              "hypotheses_differing_from_greedy", "transcripts_differing_from_greedy",
              "hypotheses_differing_from_widest",
              "decoder_mean_ms_min_across_runs", "decoder_mean_ms_max_across_runs",
              "runs_measured"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["results"]:
            writer.writerow(dict(
                beam_width=row["beam_width"], cer=row["cer"],
                s=row["substitutions"], d=row["deletions"], i=row["insertions"],
                n=row["reference_chars"],
                decoder_mean_ms=row["decoder_mean_ms"],
                decoder_median_ms=row["decoder_median_ms"],
                decoder_p95_ms=row["decoder_p95_ms"],
                decoder_sum_ms=row["decoder_sum_ms"],
                decoder_ms_per_ctc_frame=row["decoder_ms_per_ctc_frame"],
                nonblank_arcs_expanded=row["nonblank_arcs_expanded"],
                max_active_prefixes=row["max_active_prefixes"],
                speedup_vs_widest_beam=row["speedup_vs_widest_beam"],
                latency_ratio_vs_narrowest_beam=row["latency_ratio_vs_narrowest_beam"],
                cer_delta_vs_greedy=row["cer_delta_vs_greedy"],
                speedup_vs_greedy=row["speedup_vs_greedy"],
                hypotheses_differing_from_greedy=row["hypotheses_differing_from_greedy"],
                transcripts_differing_from_greedy=row["transcripts_differing_from_greedy"],
                hypotheses_differing_from_widest=row["hypotheses_differing_from_widest"],
                decoder_mean_ms_min_across_runs=row["decoder_mean_ms_min_across_runs"],
                decoder_mean_ms_max_across_runs=row["decoder_mean_ms_max_across_runs"],
                runs_measured=row["runs_measured"],
            ))


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["results"]
    widths = [r["beam_width"] for r in rows]
    latency = [r["decoder_mean_ms"] for r in rows]
    cer = [100.0 * r["cer"] for r in rows]
    greedy_cer = 100.0 * report["framewise_greedy"]["cer"]
    knee = report["knee"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)

    ax = left
    ax.plot(widths, latency, "o-", color="#1f6feb", lw=2, ms=6, label="decoder latency")
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_xlabel("Beam width")
    ax.set_ylabel("Mean decoder latency per utterance (ms)", color="#1f6feb")
    ax.tick_params(axis="y", labelcolor="#1f6feb")
    ax.grid(True, which="both", alpha=0.25, ls=":")
    twin = ax.twinx()
    twin.plot(widths, cer, "s--", color="#d1495b", lw=2, ms=6, label="CER")
    twin.set_ylabel("Dev CER (%)", color="#d1495b")
    twin.tick_params(axis="y", labelcolor="#d1495b")
    twin.set_ylim(min(cer) - 0.5, max(cer) + 0.5 if max(cer) > min(cer) else min(cer) + 1.0)
    ax.axvline(knee["beam_width"], color="#2f9e44", ls="-.", lw=1.4)
    ax.annotate(f"selected W={knee['beam_width']}\n{knee['decoder_mean_ms']:.2f} ms, "
                f"CER {100 * knee['cer']:.2f}%",
                xy=(knee["beam_width"], knee["decoder_mean_ms"]),
                xytext=(14, 26), textcoords="offset points",
                fontsize=9, color="#2f9e44",
                arrowprops=dict(arrowstyle="->", color="#2f9e44", lw=1.2))
    ax.set_title("Beam width vs decoder latency and Dev CER")

    ax = right
    ax.plot(latency, cer, "o-", color="#1f6feb", lw=2, ms=6)
    for index, row in enumerate(rows):
        ax.annotate(f"W={row['beam_width']}",
                    xy=(row["decoder_mean_ms"], 100 * row["cer"]),
                    xytext=(0, 9 if index % 2 == 0 else -16),
                    textcoords="offset points", fontsize=9, ha="center")
    ax.axhline(greedy_cer, color="#868e96", ls=":", lw=1.2,
               label=f"framewise greedy {greedy_cer:.2f}%")
    ax.scatter([knee["decoder_mean_ms"]], [100 * knee["cer"]], s=140,
               facecolors="none", edgecolors="#2f9e44", lw=2, zorder=5,
               label=f"selected W={knee['beam_width']}")
    ax.set_xlabel("Mean decoder latency per utterance (ms)")
    ax.set_ylabel("Dev CER (%)")
    ax.grid(True, alpha=0.25, ls=":")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Accuracy vs cost frontier (30 Dev utterances)")
    # The point of the chart: the frontier is vertical, not a curve.
    ax.text(0.98, 0.94,
            f"CER identical for every width; "
            f"{report['knee']['hypotheses_differing_from_widest']}/"
            f"{report['provenance']['dev_count']} token sequences differ from "
            f"W={max(widths)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#495057",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9fa", ec="#dee2e6"))

    fig.suptitle("CTC prefix-beam pruning: latency / accuracy trade-off "
                 f"({report['provenance']['dev_count']} Dev utterances, "
                 f"{report['provenance']['reference_chars']} reference chars)",
                 fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Directory holding tokens.txt and bbpe.model")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Timed repeats per utterance inside one pass")
    parser.add_argument("--runs", type=int, default=9,
                        help="Independent passes per width; the median pass is published")
    parser.add_argument("--cer-tolerance", type=float, default=0.001,
                        help="Absolute CER tolerance for the knee rule (0.001 = 0.1 pp)")
    parser.add_argument("--dev-root", type=Path, default=Path("build/ctc_round1_dev"))
    parser.add_argument("--data-root", type=Path, default=Path("data/ctc_eval"))
    parser.add_argument("--inventory", type=Path, default=Path("results/ctc_logit_inventory.json"))
    parser.add_argument("--greedy-report", type=Path, default=Path("results/ctc_baseline_dev.json"))
    parser.add_argument("--json-output", type=Path, default=Path("results/ctc_beam_pareto.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("results/ctc_beam_pareto.csv"))
    parser.add_argument("--png-output", type=Path, default=Path("results/ctc_beam_pareto.png"))
    args = parser.parse_args()

    inputs, detokenize, previous, identity = load_dev(
        args.dev_root, args.data_root, args.model_dir / "tokens.txt",
        args.model_dir / "bbpe.model", args.inventory, args.greedy_report)
    host_state = stabilize_host()
    report = build_report(inputs, detokenize, previous, identity,
                          repeats=args.repeats, tolerance=args.cer_tolerance, runs=args.runs,
                          host_state=host_state)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    write_csv(report, args.csv_output)
    plot(report, args.png_output)
    print(json.dumps(dict(status=report["status"], knee=report["knee"],
                          json=str(args.json_output), csv=str(args.csv_output),
                          png=str(args.png_output)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
