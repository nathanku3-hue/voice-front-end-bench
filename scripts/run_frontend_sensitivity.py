#!/usr/bin/env python3
"""Run the fixed 2x2 sherpa-native Fbank configuration sensitivity experiment.

Eight matched runs: 2x2 (snip_edges false/true x high_freq -400/0) for
endpoint disabled and for Rule2 1200ms enabled. We do not inject C++ features
into the ONNX encoder. No results or hypotheses are hardcoded.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

CONDITIONS = (
    ("native", False, -400.0),
    ("snip_only", True, -400.0),
    ("cutoff_only", False, 0.0),
    ("matched_cpp_options", True, 0.0),
)
MODES = ("no_endpoint", "rule2_1200")



def paired_cluster_cer_delta(reference: list[dict], candidate: list[dict],
                             reps: int = 5000, seed: int = 20261011) -> dict:
    """Paired 11-speaker resampling of corpus CER(candidate) - CER(reference).

    Describes uncertainty for this *observed cluster sample*, not an unseen-car
    population or repeated-model-training variance.
    """
    if len(reference) != len(candidate) or not reference or reps < 100:
        raise ValueError("Need matched nonempty utterance rows and >=100 resamples")
    groups = {}
    for index, (a, b) in enumerate(zip(reference, candidate)):
        if (a["id"], a["speaker_id"], a["reference_chars"]) != (
                b["id"], b["speaker_id"], b["reference_chars"]):
            raise ValueError("Candidate/source utterance pairing differs")
        groups.setdefault(a["speaker_id"], []).append(index)
    speakers = sorted(groups)
    rng = random.Random(seed)
    differences = []
    for _ in range(reps):
        indices = [i for speaker in rng.choices(speakers, k=len(speakers))
                   for i in groups[speaker]]
        denominator = sum(reference[i]["reference_chars"] for i in indices)
        if denominator <= 0:
            raise ValueError("Invalid zero reference characters")
        delta = sum(
            candidate[i]["s"] + candidate[i]["d"] + candidate[i]["i"] -
            reference[i]["s"] - reference[i]["d"] - reference[i]["i"]
            for i in indices
        ) / denominator
        differences.append(delta * 100)
    differences.sort()
    n = len(differences)
    return {
        "method": "paired speaker-cluster bootstrap, candidate minus native CER percentage points",
        "speakers": len(speakers), "reps": reps, "seed": seed,
        "approx_95_ci_pp": [differences[int(.025 * (n - 1))],
                            differences[int(.975 * (n - 1))]],
    }


def summarize(files: dict[str, dict[str, Path]]) -> dict:
    report = {"scope": "Native sherpa-onnx 2x2 parameter sensitivity, not C++ Fbank injection",
              "limits": [
                  "30 AISHELL-1 read speech files; small speaker sample, not car commands",
                  "One public pretrained model, identical per-condition files/hashes/chunks/threads",
                  "snip_edges and Mel cutoff changed in sherpa's OWN native frontend only",
                  "No ONNX Encoder cache state or C++ feature input path tested",
                  "No-endpoint runs use identical padded tail then input_finished, not measured latency",
                  "Rule2 1200ms runs use approximate stream-time; no human speech-end timestamps",
              ], "modes": {}}
    all_runs = {}
    for mode, mapping in files.items():
        all_runs[mode] = {}
        first = None
        for name, snip, high in CONDITIONS:
            data = json.loads(mapping[name].read_text(encoding="utf-8"))
            cfg = data["config"]
            if (cfg["snip_edges"] != snip or cfg["high_freq"] != high
                    or cfg["chunk_ms"] != 160 or cfg["threads"] != 2
                    or cfg["endpoint_enabled"] != (mode == "rule2_1200")
                    or cfg["rule2_trailing_silence_ms"] != 1200
                    or cfg["tail_ms"] != 2500):
                raise ValueError(f"Configuration mismatch: {mode}/{name}")
            rows = data["utterances"]
            if len(rows) != 30 or any(row["cer"] is None or row["speaker_id"] is None for row in rows):
                raise ValueError(f"Expected 30 scored speaker-identified WAVs in {mode}/{name}")
            identity = [(r["id"], r["wav_sha256"], r["reference"],
                         r["speaker_id"], r["reference_chars"]) for r in rows]
            control = (
                data["model"]["file_sha256"],
                data["manifest_sha256"],
                {k: v for k, v in cfg.items() if k not in ("snip_edges", "high_freq")},
                identity,
            )
            if first is None:
                first = control
            elif control != first:
                raise ValueError(f"Model, inputs or non-frontend settings mismatch: {mode}/{name}")
            if len(set(x[0] for x in identity)) != len(identity):
                raise ValueError(f"Duplicate WAV IDs in {mode}/{name}")
            all_runs[mode][name] = data

        reference_chars = sum(x["reference_chars"] for x in all_runs[mode]["native"]["utterances"])
        if reference_chars != 403:
            raise ValueError(f"Unexpected reference chars: {reference_chars}")
        baseline = all_runs[mode]["native"]["utterances"]
        computed = {}
        for name, snip, high in CONDITIONS:
            data = all_runs[mode][name]
            rows = data["utterances"]
            s, d, i = (sum(r[k] for r in rows) for k in ("s", "d", "i"))
            comparisons = [(a, b) for a, b in zip(baseline, rows)]
            computed[name] = {
                "snip_edges": snip, "high_freq": high,
                "mel_upper_hz": 7600 if high == -400.0 else 8000,
                "n": len(rows), "reference_chars": reference_chars,
                "s": s, "d": d, "i": i, "cer": (s + d + i) / reference_chars,
                "changed_transcripts_vs_native": sum(a["hypothesis"] != b["hypothesis"]
                                                     for a, b in comparisons),
                "deletions_difference_vs_native": d - sum(x["d"] for x in baseline),
                "cer_delta_vs_native_pp": 100 * ((s + d + i) / reference_chars -
                    sum(x["s"] + x["d"] + x["i"] for x in baseline) / reference_chars),
                "cer_delta_cluster_bootstrap95": paired_cluster_cer_delta(baseline, rows),
                "endpoint_observed_n": sum(bool(r["endpoint_detected"]) for r in rows),
                "mean_endpoint_stream_time_ms": (
                    sum(r["endpoint_audio_time_s"] * 1000 for r in rows) / len(rows)
                    if all(r["endpoint_audio_time_s"] is not None for r in rows) else None
                ),
            }
        a, b, c, d = (computed[key]["cer"] for key in
                      ("native", "snip_only", "cutoff_only", "matched_cpp_options"))
        report["modes"][mode] = {
            "conditions": computed,
            "interaction_cer_pp": 100 * (d - b - c + a),
            "observed_best_cer_configs": [
                name for name, row in computed.items()
                if abs(row["cer"] - min(x["cer"] for x in computed.values())) < 1e-12
            ],
            "model_hashes": all_runs[mode]["native"]["model"]["file_sha256"],
            "manifest_sha256": all_runs[mode]["native"]["manifest_sha256"],
            "file_paths": {name: str(mapping[name]) for name, _, _ in CONDITIONS},
        }
    if len(all_runs) == 2:
        sample_a = all_runs[MODES[0]]["native"]
        sample_b = all_runs[MODES[1]]["native"]
        if (sample_a["model"]["file_sha256"] != sample_b["model"]["file_sha256"]
                or sample_a["manifest_sha256"] != sample_b["manifest_sha256"]):
            raise ValueError("Different model / manifest across endpoint modes")
    return report


def execute(manifest: Path, model_dir: Path, out: Path) -> dict[str, dict[str, Path]]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    bench = Path(__file__).with_name("asr_stream_bench.py").resolve()
    for mode in MODES:
        paths[mode] = {}
        for name, snip, high in CONDITIONS:
            destination = out / f"{mode}_{name}.json"
            cmd = [sys.executable, str(bench), "--manifest", str(manifest.resolve()),
                   "--model-dir", str(model_dir.resolve()), "--chunk-ms", "160",
                   "--tail-ms", "2500", "--threads", "2", "--rule2-ms", "1200",
                   "--high-freq", str(high), "--output", str(destination.resolve())]
            if snip:
                cmd.append("--snip-edges")
            if mode == "rule2_1200":
                cmd.append("--endpoint")
            print(f"Running {mode}/{name}: snip={snip}, high_freq={high}", flush=True)
            # Capture per-utterance chatter, but report the diagnostic if a run fails.
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode:
                raise RuntimeError(f"{mode}/{name} failed:\n{done.stdout[-2500:]}\n{done.stderr[-2500:]}")
            paths[mode][name] = destination
    return paths


def write_summary(report: dict, dest: Path, csv_path: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    records = []
    for mode, info in report["modes"].items():
        for name, values in info["conditions"].items():
            records.append({"mode": mode, "condition": name, **values})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("| mode | variant | CER | S/D/I | changed transcripts vs native |")
    print("|---|---|---:|---:|---:|")
    for r in records:
        print(f"| {r['mode']} | {r['condition']} | {r['cer']:.2%} | "
              f"{r['s']}/{r['d']}/{r['i']} | {r['changed_transcripts_vs_native']}/30 |")
    for mode, data in report["modes"].items():
        print(f"{mode}: interaction={data['interaction_cer_pp']:+.3f} pp, "
              f"best observed={data['observed_best_cer_configs']}")
    print("Native frontend reconfiguration only — NOT C++→ONNX drop-in compatibility.")


def plot_summary(report: dict, target: Path) -> None:
    """Render the two replay-mode CER groups; no arbitrary performance target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [name for name, _, _ in CONDITIONS]
    x = list(range(len(labels)))
    width = .34
    fig, ax = plt.subplots(figsize=(10.5, 5.3), dpi=160)
    for shift, mode, caption in ((-width / 2, "no_endpoint", "Endpoint disabled"),
                                 (width / 2, "rule2_1200", "Rule2 = 1200 ms")):
        values = [report["modes"][mode]["conditions"][name]["cer"] * 100
                  for name in labels]
        bars = ax.bar([i + shift for i in x], values, width, label=caption)
        for rect, value in zip(bars, values):
            ax.annotate(f"{value:.2f}%", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, ["Native\n7600Hz", "Snip only\n7600Hz",
                       "Cutoff only\n8000Hz", "Matched options\n8000Hz"])
    ax.set_ylabel("Corpus CER (%) — 403 reference characters")
    ax.set_ylim(0, 19)
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    ax.set_title("Same-model native frontend contract sensitivity (30 AISHELL utterances)")
    fig.text(.5, .01, "Native sherpa frontend settings only; no C++ Fbank injected; not vehicle speech",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Required unless --summarize-only")
    parser.add_argument("--model-dir", type=Path, help="Required unless --summarize-only")
    parser.add_argument("--output-root", type=Path, default=Path("build/p3_frontend_runs"))
    parser.add_argument("--summary", type=Path, default=Path("build/p3_frontend_summary.json"))
    parser.add_argument("--csv", type=Path, default=Path("build/p3_frontend_summary.csv"))
    parser.add_argument("--plot", type=Path, default=None, help="Optional comparison chart")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Validate and summarize existing 8 run files without inference")
    args = parser.parse_args()
    if args.summarize_only:
        paths = {
            mode: {name: args.output_root / f"{mode}_{name}.json" for name, _, _ in CONDITIONS}
            for mode in MODES
        }
    else:
        if args.manifest is None or args.model_dir is None:
            parser.error("--manifest and --model-dir required unless --summarize-only")
        paths = execute(args.manifest, args.model_dir, args.output_root)
    report = summarize(paths)
    write_summary(report, args.summary, args.csv)
    if args.plot is not None:
        plot_summary(report, args.plot)
        print(f'Saved {args.plot}')


if __name__ == "__main__":
    main()
