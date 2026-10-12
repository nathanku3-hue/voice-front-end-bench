#!/usr/bin/env python3
"""Paired 5-setting endpoint sweep; stream-time, not real-microphone latency."""
import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

DEFAULT_THRESHOLDS = (300, 500, 700, 900, 1200)


def bootstrap_cer(rows, seed=20261011, replicates=1000):
    rng = random.Random(seed)
    observed = []
    for _ in range(replicates):
        draw = rng.choices(rows, k=len(rows))
        numerator = sum(x["s"] + x["d"] + x["i"] for x in draw)
        denominator = sum(x["reference_chars"] for x in draw)
        observed.append(numerator / denominator)
    observed.sort()
    return observed[int(replicates * 0.025)], observed[int(replicates * 0.975)]


def summarize(rows, tolerance_s=0.03):
    if not rows or any(x["cer"] is None or x["speech_end_s"] is None or
                       not x.get("speech_end_method") for x in rows):
        raise ValueError("Every sample needs verified reference text and a described endpoint method")
    s, d, i = (sum(x[name] for x in rows) for name in ("s", "d", "i"))
    n_chars = sum(x["reference_chars"] for x in rows)
    observed = [x for x in rows if x["endpoint_audio_time_s"] is not None]
    premature = [x for x in observed if x["endpoint_audio_time_s"] + tolerance_s < x["speech_end_s"]]
    delays = [1000 * (x["endpoint_audio_time_s"] - x["speech_end_s"]) for x in observed]
    ontime = [v for v in delays if v >= -1000 * tolerance_s]
    lo, hi = bootstrap_cer(rows)
    return {
        "n": len(rows), "reference_chars": n_chars, "s": s, "d": d, "i": i,
        "cer": (s+d+i)/n_chars, "cer_ci95_low": lo, "cer_ci95_high": hi,
        "endpoints_observed": len(observed), "endpoints_missing": len(rows) - len(observed),
        "premature_proxy_count": len(premature),
        "premature_proxy_rate": len(premature) / len(rows),
        "mean_signed_delay_ms": sum(delays)/len(delays) if delays else None,
        "mean_ontime_delay_ms": sum(ontime)/len(ontime) if ontime else None,
        "ontime_count": len(ontime),
        "uncertain_vad_end_count": sum(x.get("speech_end_status") != "estimated" for x in rows),
    }


def chart(table, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [x["rule2_ms"] for x in table]
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=150)
    a, = ax.plot(xs, [x["mean_ontime_delay_ms"] for x in table], "-o", label="On-time delay (ms)")
    ax.set_xlabel("Trailing-silence threshold (ms)")
    ax.set_xticks(xs)
    ax.set_ylabel("Mean delay from estimated speech end (ms)")
    ax.grid(alpha=0.2)
    ax2 = ax.twinx()
    b, = ax2.plot(xs, [100*x["cer"] for x in table], "-s", label="CER (%)")
    c, = ax2.plot(xs, [100*x["premature_proxy_rate"] for x in table], "-^", label="Early endpoint proxy (%)")
    ax2.set_ylabel("CER / early endpoint proxy (%)")
    ax2.set_ylim(bottom=0)
    ax.legend(handles=[a,b,c], loc="upper center")
    ax.set_title("AISHELL-1 TEST: endpointing ablation (30-utterance pilot)")
    fig.text(0.5, 0.01, "VAD endpoints are estimated, NOT manual truth; stream-time only, NOT product latency",
             fontsize=8, ha="center")
    fig.tight_layout(rect=(0,0.035,1,1))
    target.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(target)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--thresholds", nargs="+", type=int, default=DEFAULT_THRESHOLDS)
    ap.add_argument("--chunk-ms", type=int, default=160)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tail-ms", type=int, default=2500)
    ap.add_argument("--paced", action="store_true")
    ap.add_argument("--output-root", type=Path, default=Path("results"))
    args = ap.parse_args()
    manifest = [json.loads(x) for x in args.manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not (20 <= len(manifest) <= 30) or len({x["id"] for x in manifest}) != len(manifest):
        ap.error("20-30 unique manifest entries required")
    if any(x.get("source_split") != "AISHELL-1/test" for x in manifest):
        ap.error("Only AISHELL-1 test rows supported")
    if not args.thresholds or min(args.thresholds) <= 0 or len(set(args.thresholds)) != len(args.thresholds):
        ap.error("Unique positive thresholds required")
    target = args.output_root / "endpoint_sweep"
    target.mkdir(parents=True, exist_ok=True)
    expected = None
    table = []
    for threshold in args.thresholds:
        run_json = target / f"rule2_{threshold}ms.json"
        command = [sys.executable, str(Path(__file__).with_name("asr_stream_bench.py").resolve()),
                   "--manifest", str(args.manifest.resolve()), "--model-dir", str(args.model_dir.resolve()),
                   "--endpoint", "--rule2-ms", str(threshold), "--chunk-ms", str(args.chunk_ms),
                   "--tail-ms", str(args.tail_ms), "--threads", str(args.threads),
                   "--output", str(run_json.resolve())]
        if args.paced:
            command.append("--paced")
        print(f"RUN rule2={threshold}ms", flush=True)
        subprocess.run(command, check=True)
        obj = json.loads(run_json.read_text(encoding="utf-8"))
        utterances = obj["utterances"]
        identifiers = [(x["id"], x["wav_sha256"], x["reference"], x["speech_end_s"]) for x in utterances]
        if expected is None:
            expected = identifiers
        elif expected != identifiers:
            raise RuntimeError("Model's evaluated dataset changed between thresholds")
        if [x["id"] for x in utterances] != [x["id"] for x in manifest]:
            raise RuntimeError("Missing or reordered examples")
        row = summarize(utterances)
        row["rule2_ms"] = threshold
        table.append(row)
        print(f"rule2={threshold}: CER={row['cer']:.2%}; early proxy="
              f"{row['premature_proxy_rate']:.2%}; mean ontime delay="
              f"{row['mean_ontime_delay_ms']}; missing endpoints="
              f"{row['endpoints_missing']}", flush=True)
    table.sort(key=lambda x:x["rule2_ms"])
    args.output_root.mkdir(exist_ok=True, parents=True)
    with (args.output_root / "endpoint_summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    summary = dict(dataset="AISHELL-1/test", count=len(manifest), model_input="sherpa-onnx waveform API",
                   speech_end="reverse RMS -35dB relative to peak: automatic proxy, not manual truth",
                   early_proxy="endpoint_audio_time + 30ms < estimated_speech_end_s",
                   timing="paced file replay" if args.paced else "unpaced stream-time",
                   limits="30 utterances small; intervals are naive utterance bootstrap not speaker-aware",
                   rows=table)
    (args.output_root / "endpoint_summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    try:
        chart(table, args.output_root / "endpoint_tradeoff.png")
    except ModuleNotFoundError as e:
        if e.name != "matplotlib":
            raise
        print("Install matplotlib to render endpoint_tradeoff.png",flush=True)
    print(f"DONE {args.output_root / 'endpoint_summary.csv'}",flush=True)


if __name__=="__main__":
    main()
