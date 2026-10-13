#!/usr/bin/env python3
"""Draw Dev and held-out Test CER vs *decoder-only* ms/utterance (Round 3).

Does not decode Test logits. Test frontier is computed on Test results only.
All four frozen conditions are drawn; no post-hoc configuration is removed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_ctc_test_eval import TEST_STATUS, CONDITIONS, pareto_ids

LABELS = {
    "framewise_greedy": "Framewise Greedy",
    "prefix_w16_unpruned": "Beam 16, unpruned",
    "prefix_w16_forced_blank_t0.95": "Beam 16, Forced .95",
    "prefix_w04_forced_blank_t0.95": "Beam 4, Forced .95",
}


def validate_results(result):
    if result.get("status") != TEST_STATUS:
        raise ValueError("Only completed Test results may be plotted")
    expected = [c[0] for c in CONDITIONS]
    conf = result["configurations"]
    dev = result["dev_comparison"]
    if [r["id"] for r in conf] != expected or [r["id"] for r in dev] != expected:
        raise ValueError("Exactly four pre-registered Dev/Test conditions required")
    actual_pareto = pareto_ids(conf)
    if result["test_pareto_frontier_ids"] != actual_pareto:
        raise ValueError("Test Pareto annotation inconsistent with measured results")
    return conf, dev, actual_pareto


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tests, devs, frontier = validate_results(result)
    fig, ax = plt.subplots(figsize=(10.8, 6.8), constrained_layout=True)
    for i, (test, dev) in enumerate(zip(tests, devs)):
        label = LABELS[test["id"]]
        tx, ty = test["timing"]["decoder_mean_ms"], 100 * test["corpus"]["cer"]
        dx, dy = dev["timing"]["decoder_mean_ms"], 100 * dev["corpus"]["cer"]
        if min(tx, dx) <= 0:
            raise ValueError("Decoder timings must be strictly positive")
        # Same condition uses one color across both splits; marker indicates split.
        segment, = ax.plot([dx, tx], [dy, ty], linestyle=":", linewidth=1, alpha=0.65)
        color = segment.get_color()  # matplotlib's default cycle, matched by condition
        ax.scatter([dx], [dy], marker="o", s=80, facecolors="none",
                   edgecolors=color, linewidths=2, zorder=3)
        ax.scatter([tx], [ty], marker="x", s=88, color=color, linewidths=2, zorder=4)
        ax.annotate(f"{label}\nTest {ty:.2f}%", (tx, ty),
                    xytext=(7, -13 if i % 2 else 12), textcoords="offset points",
                    fontsize=8.5, color=color)
    lookup = {c["id"]: c for c in tests}
    ordered = sorted((lookup[cid] for cid in frontier),
                     key=lambda c: c["timing"]["decoder_mean_ms"])
    if ordered:
        ax.plot([c["timing"]["decoder_mean_ms"] for c in ordered],
                [100*c["corpus"]["cer"] for c in ordered],
                linewidth=1.5, linestyle="--", label="Test nondominated frontier")
    from matplotlib.lines import Line2D
    legend = [Line2D([], [], linestyle="none", marker="o", markersize=8,
                     markerfacecolor="none", label="Dev (open circle)"),
              Line2D([], [], linestyle="none", marker="x", markersize=8,
                     label="Held-out Test (cross)"),
              Line2D([], [], linestyle="--",
                     label="Test nondominated frontier")]
    ax.legend(handles=legend, loc="best", frameon=False)
    ax.set_xscale("log")
    ax.set_xlabel("Decoder-only mean of per-utterance medians (ms/utterance, log scale)")
    ax.set_ylabel("Corpus CER (%) — lower is better")
    ax.set_title("Pre-registered CTC Round 3: Dev vs held-out Test (4 conditions)")
    ax.grid(alpha=0.22, which="both")
    fig.text(0.01, 0.015,
             "Python decoder only; no WAV/Fbank/ONNX/detokenization timing. "
             "Dev n=30, Test n=50; not speaker-disjoint.", fontsize=8.3)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Not overwriting existing plot: {output}")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-results", type=Path,
                   default=Path("results/ctc_test_final_results.json"))
    p.add_argument("--output", type=Path,
                   default=Path("results/ctc_test_pareto.png"))
    args = p.parse_args()
    result = json.loads(args.test_results.read_text(encoding="utf-8"))
    print(plot(result, args.output))


if __name__ == "__main__":
    main()
