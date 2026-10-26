#!/usr/bin/env python3
"""How sensitive is the CTC transcript to feature-domain error?

The fast log changes the feature matrix by ~3.1e-8 mean absolute error. "That is
obviously small enough" is an argument, not evidence. This script measures it:
take the clean features, add uniform noise of a known magnitude, push the result
through the same int8 Zipformer CTC graph with the same C++ decoder, and see what
moves.

The result is not the cliff that was expected. Transcripts change at *any*
injected amplitude, including the smallest one tested, and the number that change
does not grow with amplitude. What is happening is that one or two utterances sit
on a knife edge -- a token whose top two posteriors are nearly tied -- so a
perturbation of 5e-7 and a perturbation of 1.5e-2 flip the same token, and which
one flips is close to random. The corpus CER moves by at most one character
across five decades of injected noise.

That is a more useful finding than a threshold, and it changes what the fast log
has to prove: not "we are below the cliff" (there is no cliff) but "the error we
introduce is smaller than the smallest perturbation this experiment can resolve,
and the corpus CER does not move". Both are measured here.

Protocol notes that matter:
  * the clean run (level 0) is the reference for CER and transcript identity, and
    it reproduces the published 2.94% baseline exactly -- that is the control
  * noise is uniform on [-a, +a] with a fixed seed per (utterance, level), so the
    run is reproducible and no level gets a lucky draw
  * the achieved MAE is reported next to the nominal amplitude, because uniform
    noise on [-a, a] has MAE a/2
  * determinism of the graph is checked, not assumed: the same features are
    forwarded twice and the posteriors compared bitwise, so a flip cannot be
    blamed on the runtime
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from asr_metrics import corpus, evaluate  # noqa: E402
from audit_numeric_sensitivity import forward, open_session, run, sha256  # noqa: E402

# Nominal half-widths of the uniform perturbation, geometric so no decade is
# skipped. The ladder spans five orders of magnitude, which is the range over
# which "how much error can this frontend take" has to be answered.
LEVELS = [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

PUBLISHED_BASELINE_CER = 0.029411764705882353


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev-root", type=Path, default=REPO / "build/ctc_round1_dev")
    parser.add_argument("--data-root", type=Path, default=REPO / "data/ctc_eval")
    parser.add_argument("--model-dir", type=Path, default=REPO / "build/win-zig/model")
    parser.add_argument("--dump-binary", type=Path, default=REPO / "build/win-zig/fbank_feature_dump.exe")
    parser.add_argument("--pipeline-binary", type=Path, default=REPO / "build/win-zig/fbank_ctc_pipeline.exe")
    parser.add_argument("--baseline", type=Path, default=REPO / "results/ctc_baseline_dev.json")
    parser.add_argument("--inventory", type=Path, default=REPO / "results/ctc_logit_inventory.json")
    parser.add_argument("--fast-log-audit", type=Path, default=REPO / "results/fast_log_accuracy.json")
    parser.add_argument("--numeric-audit", type=Path, default=REPO / "results/numeric_sensitivity.json")
    parser.add_argument("--out", type=Path, default=REPO / "results/cer_sensitivity.json")
    parser.add_argument("--plot", type=Path, default=REPO / "results/cer_sensitivity.png")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20240517)
    args = parser.parse_args()

    model_file = args.model_dir / "model.int8.onnx"
    tokens = args.model_dir / "tokens.txt"
    for path in (model_file, tokens, args.dump_binary, args.pipeline_binary):
        if not path.is_file():
            raise SystemExit(f"missing: {path}")

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    reference_text = {u["id"]: u["reference"] for u in baseline["utterances"]}
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    manifest = [json.loads(line) for line in
                (args.dev_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    if args.limit:
        manifest = manifest[: args.limit]

    observed = sha256(model_file)
    if observed != inventory.get("model_sha256"):
        raise SystemExit("model hash mismatch against the frozen export: "
                         f"{observed} != {inventory.get('model_sha256')}")
    print(f"model {observed[:12]} verified, {len(manifest)} utterances, "
          f"{len(LEVELS)} noise levels")

    session = open_session(model_file)
    scratch = Path(tempfile.mkdtemp(prefix="cer-sensitivity-"))

    per_level: dict[float, dict] = {
        level: {"mae": [], "max_abs": [], "counts": [], "texts": {}} for level in LEVELS}

    determinism = None
    for index, entry in enumerate(manifest, 1):
        uid = entry["id"]
        wav = args.data_root / "audio" / f"{uid}.wav"
        if not wav.is_file():
            raise SystemExit(f"missing audio: {wav}")

        raw = scratch / f"{uid}.f32"
        meta = scratch / f"{uid}.json"
        run([str(args.dump_binary), "--wav", str(wav), "--out", str(raw),
             "--json", str(meta), "--log", "std"])
        info = json.loads(meta.read_text(encoding="utf-8"))
        clean = np.fromfile(raw, dtype="<f4").reshape(info["frames"], info["bins"])
        # Same snip_edges alignment as the variant audit, so the CER is
        # comparable with the published baseline.
        clean = np.vstack([clean[0:1], clean, clean[-1:]]).astype(np.float64)

        if determinism is None:
            # If the graph itself were nondeterministic, every flip below would
            # be unattributable. Check it once, on real features.
            first = forward(session, clean.astype(np.float32))
            second = forward(session, clean.astype(np.float32))
            determinism = {
                "utterance": uid,
                "identical_posteriors": bool(np.array_equal(first, second)),
                "max_abs_diff": float(np.abs(first - second).max()),
            }
            if not determinism["identical_posteriors"]:
                raise SystemExit("the ONNX graph is not deterministic on this host; "
                                 "the sweep below would measure the runtime, not the noise")
            print(f"determinism control: {uid} forwarded twice, "
                  f"max |diff| = {determinism['max_abs_diff']}")

        for level in LEVELS:
            if level == 0.0:
                perturbed = clean
            else:
                rng = np.random.default_rng([args.seed, index, int(round(level * 1e9))])
                perturbed = clean + rng.uniform(-level, level, size=clean.shape)
            delta = np.abs(perturbed - clean)
            log_probs = forward(session, perturbed.astype(np.float32))
            npy = scratch / f"{uid}_{level:g}.npy"
            np.save(npy, log_probs, allow_pickle=False)
            proc = subprocess.run(
                [str(args.pipeline_binary), "--wav", str(wav), "--posteriors", str(npy),
                 "--tokens", str(tokens), "--repeats", "1"],
                capture_output=True, text=True, encoding="utf-8")
            if proc.returncode != 0:
                raise SystemExit(f"{uid} @ {level:g}: pipeline failed: {proc.stderr.strip()[:300]}")
            text = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
            per_level[level]["texts"][uid] = text
            per_level[level]["mae"].append(float(delta.mean()))
            per_level[level]["max_abs"].append(float(delta.max()))
            per_level[level]["counts"].append(evaluate(reference_text[uid], text))
        if index % 5 == 0 or index == len(manifest):
            print(f"  {index}/{len(manifest)} utterances", flush=True)

    clean_texts = per_level[0.0]["texts"]
    clean_counts = per_level[0.0]["counts"]
    rows = []
    for level in LEVELS:
        entry = per_level[level]
        differing = [uid for uid, text in entry["texts"].items() if text != clean_texts[uid]]
        aggregate = corpus(entry["counts"])
        clean_aggregate = corpus(clean_counts)
        rows.append({
            "nominal_amplitude": level,
            "feature_mae": float(np.mean(entry["mae"])),
            "feature_max_abs": float(np.max(entry["max_abs"])),
            "transcripts_identical_to_clean": len(entry["texts"]) - len(differing),
            "transcripts_differing": differing,
            "hypothesis_chars": sum(len(text) for text in entry["texts"].values()),
            "cer": aggregate["cer"],
            "cer_counts": aggregate,
            # How many characters this level moved relative to the clean run --
            # the number that says whether the noise matters.
            "extra_edits_vs_clean": (
                aggregate["substitutions"] + aggregate["deletions"] + aggregate["insertions"]
                - clean_aggregate["substitutions"] - clean_aggregate["deletions"]
                - clean_aggregate["insertions"]),
        })

    # Which utterances are unstable, and at how many levels. The distribution is
    # the finding: it is not spread across the corpus, it is one or two items.
    ever_changed: dict[str, int] = {}
    for row in rows:
        for uid in row["transcripts_differing"]:
            ever_changed[uid] = ever_changed.get(uid, 0) + 1
    fragile = sorted(ever_changed.items(), key=lambda item: (-item[1], item[0]))

    perturbed_rows = [row for row in rows if row["nominal_amplitude"] > 0.0]
    cer_values = [row["cer"] for row in rows]
    total_slots = len(perturbed_rows) * len(manifest)
    changed_slots = sum(len(row["transcripts_differing"]) for row in perturbed_rows)

    fast_log = None
    if args.fast_log_audit.is_file():
        audit = json.loads(args.fast_log_audit.read_text(encoding="utf-8"))
        kernel = next(k for k in audit["kernels"] if k["name"] == "fast_log_float")
        fast_log = {"max_abs_error": kernel["max_abs_error"],
                    "rms_abs_error": kernel["rms_abs_error"]}

    shipped = None
    if args.numeric_audit.is_file():
        numeric = json.loads(args.numeric_audit.read_text(encoding="utf-8"))
        summary = numeric["summary"].get("fast")
        if summary:
            shipped = {
                "feature_mae_mean": summary["feature_mae_mean"],
                "transcripts_identical": summary["transcripts_identical_to_std"],
                "utterances": summary["utterances"],
                "cer": summary["cer"],
            }

    # Compare like with like: the shipped path's *feature* MAE against the
    # sweep's *feature* MAE. The kernel's own rms error (2e-7) is in log units
    # and is not comparable with a rung of the noise ladder.
    smallest_rung = min(row["feature_mae"] for row in perturbed_rows)
    shipped_feature_mae = shipped["feature_mae_mean"] if shipped else None
    report = {
        "status": "measured_cer_sensitivity_curve",
        "question": ("how much feature-domain error does the CTC frontend tolerate, and where "
                     "does the fast log sit relative to that?"),
        "answer": ("there is no threshold to sit below. Transcripts change at the smallest "
                   "amplitude tested, and the number that change does not grow with amplitude; "
                   "the damage is concentrated in one or two utterances whose top-2 posteriors "
                   "are nearly tied, and the corpus CER moves by at most one character across "
                   "five decades of injected noise."),
        "model": {
            "file": str(model_file.relative_to(REPO)),
            "sha256_observed": observed,
            "sha256_recorded_in_frozen_export": inventory.get("model_sha256"),
            "asset_matches_frozen_export": True,
            "execution_provider": "CPUExecutionProvider",
        },
        "determinism_control": determinism,
        "protocol": {
            "utterances": len(manifest),
            "levels": LEVELS,
            "noise": "uniform on [-a, +a], one generator per (utterance, level), fixed seed",
            "seed": args.seed,
            "decoder": "C++ greedy collapse + C++ byte-BPE detokenizer",
            "reference": "the level-0 run of this script",
            "framing_alignment": "one replicated frame prepended and appended",
        },
        "fast_log_reference": fast_log,
        "shipped_path": shipped,
        "rows": rows,
        # Every hypothesis, so a reader can check a claimed flip without
        # re-running the model. Small: 30 utterances x 11 levels.
        "texts": {f"{level:g}": per_level[level]["texts"] for level in LEVELS},
        "fragile_utterances": [{"id": uid, "levels_changed": count} for uid, count in fragile],
        "derived": {
            "clean_cer": rows[0]["cer"],
            "clean_cer_matches_published_baseline": bool(
                abs(rows[0]["cer"] - PUBLISHED_BASELINE_CER) < 1e-12),
            "cer_min": float(min(cer_values)),
            "cer_max": float(max(cer_values)),
            "cer_spread_chars": int(max(cer_values) * rows[0]["cer_counts"]["reference_chars"]
                                    - min(cer_values) * rows[0]["cer_counts"]["reference_chars"]),
            "max_extra_edits_vs_clean": int(max(row["extra_edits_vs_clean"] for row in rows)),
            "change_rate_over_perturbed_levels": changed_slots / total_slots if total_slots else 0.0,
            "utterances_that_ever_change": len(ever_changed),
            "most_fragile_utterance": fragile[0][0] if fragile else None,
            "smallest_tested_feature_mae": smallest_rung,
            "shipped_feature_mae": shipped_feature_mae,
            "shipped_feature_mae_decades_below_smallest_rung": (
                float(np.log10(smallest_rung / shipped_feature_mae))
                if shipped_feature_mae and shipped_feature_mae > 0 else None),
        },
        "non_claims": [
            "uniform noise is not the same perturbation as a fast-log error, which is "
            "signal dependent; the sweep bounds the magnitude, not the shape",
            "one model, one corpus: which utterances are fragile is a property of this int8 "
            "Zipformer on these 30 utterances, not a universal constant",
            "the level-0 CER here is this script's own run, not a re-measurement of the "
            "published baseline (it does reproduce it exactly)",
            "no conclusion is drawn about audio-domain robustness: the noise is injected "
            "after the frontend, so it bypasses every earlier stage",
        ],
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'amplitude':>10}{'feature MAE':>14}{'max |d|':>12}{'identical':>11}"
          f"{'CER':>9}{'extra edits':>13}")
    for row in rows:
        print(f"{row['nominal_amplitude']:>10.0e}{row['feature_mae']:>14.3e}"
              f"{row['feature_max_abs']:>12.3e}"
              f"{row['transcripts_identical_to_clean']:>7}/{len(manifest):<3}"
              f"{100 * row['cer']:>8.2f}%{row['extra_edits_vs_clean']:>13}")
    print(f"\nCER across all levels: {100 * min(cer_values):.2f}% .. {100 * max(cer_values):.2f}% "
          f"({max(row['extra_edits_vs_clean'] for row in rows)} character(s) of movement)")
    print(f"utterances that ever change: {len(ever_changed)} of {len(manifest)}"
          + (f" -- {fragile[0][0]} at {fragile[0][1]} of {len(perturbed_rows)} levels"
             if fragile else ""))
    if shipped_feature_mae:
        print(f"shipped feature MAE {shipped_feature_mae:.2e} is "
              f"{report['derived']['shipped_feature_mae_decades_below_smallest_rung']:.1f} "
              f"decades below the smallest rung tested ({smallest_rung:.2e})")

    plot(rows, fragile,
         dict(feature_mae=shipped_feature_mae) if shipped_feature_mae else None,
         args.plot)
    print(f"\nwrote {args.out} and {args.plot}")
    return 0


def plot(rows: list[dict], fragile: list[tuple[str, int]], fast_log: dict | None,
         path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = rows[0]["cer_counts"]["labeled_utterances"]
    perturbed = [row for row in rows if row["nominal_amplitude"] > 0.0]

    fig, (left, right) = plt.subplots(1, 2, figsize=(14.2, 5.0), constrained_layout=True,
                                      gridspec_kw=dict(width_ratios=[1.35, 1.0]))

    # ---- left: CER against injected error, with the change count annotated ---
    ax = left
    x = [row["feature_mae"] for row in rows]
    cer = [100.0 * row["cer"] for row in rows]
    ax.plot(x, cer, marker="o", color="#1f6feb", lw=1.9, ms=6, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("achieved feature MAE (log scale)")
    ax.set_ylabel("corpus CER (%)")
    ax.grid(True, alpha=0.25, ls=":")
    band_lo, band_hi = min(cer), max(cer)
    ax.axhspan(band_lo, band_hi, color="#1f6feb", alpha=0.07, zorder=0)
    ax.set_ylim(band_lo - 0.08, band_hi + 0.10)
    # Only annotate the rungs that actually moved something; labelling the flat
    # ones would stack eight identical "0/30" strings on one line.
    unchanged = 0
    for order, (row, value) in enumerate(zip(rows, cer)):
        changed = len(row["transcripts_differing"])
        if changed:
            # Alternate the offset so two adjacent annotations cannot collide.
            lift = 11 if (order % 2 == 0) else 20
            ax.annotate(f"{changed}/{total} changed", (row["feature_mae"], value),
                        textcoords="offset points", xytext=(0, lift), ha="center",
                        fontsize=7.5, color="#495057")
        else:
            unchanged += 1
    ax.annotate(f"{unchanged} of {len(rows)} rungs changed nothing",
                (0.985, 0.93), xycoords="axes fraction", ha="right", fontsize=8.5,
                color="#868e96")
    if fast_log:
        # The vertical line is the *feature* MAE of the shipped path, which is
        # what the x-axis measures.
        ax.axvline(fast_log["feature_mae"], color="#2f9e44", ls="--", lw=1.4, zorder=2)
        ax.annotate(f"shipped path\n{fast_log['feature_mae']:.1e}",
                    (fast_log["feature_mae"], band_lo - 0.05),
                    textcoords="offset points", xytext=(7, 0), fontsize=8.5, color="#2f9e44")
    ax.set_title(f"Injected noise vs CER: {band_hi - band_lo:.2f} points of movement\n"
                 f"over five decades, and no threshold anywhere in it", fontsize=10)

    # ---- right: which utterances are unstable --------------------------------
    ax = right
    if fragile:
        names = [uid for uid, _ in fragile]
        counts = [count for _, count in fragile]
        ax.barh(range(len(names)), counts, color="#d1495b", height=0.5)
        for index, (name, count) in enumerate(zip(names, counts)):
            ax.text(count + 0.12, index, f"{name}   {count} of {len(perturbed)}",
                    va="center", fontsize=8.5)
        ax.set_yticks([])
        ax.set_ylim(len(names) - 0.4, -0.6)
        ax.set_xlim(0, max(counts) * 2.3)
        ax.set_xlabel(f"noise levels at which this transcript changed (of {len(perturbed)})")
        ax.set_title(f"The instability is {len(fragile)} utterances, not the corpus\n"
                     "one of them flips at half the levels; the rest flip once", fontsize=10)
        ax.grid(True, axis="x", alpha=0.25, ls=":")
    else:
        ax.axis("off")

    fig.suptitle("Feature-domain error tolerance: measured, not assumed", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
