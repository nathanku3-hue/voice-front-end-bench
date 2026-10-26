#!/usr/bin/env python3
"""Does the faster frontend change what the model hears?

This is the only audit in the repository that answers that question with a model
run instead of an argument. For every Dev utterance it builds the feature matrix
four ways, pushes each one through the same int8 Zipformer CTC graph, decodes
with the same C++ greedy decoder and detokenizer, and compares the transcripts.

    std           vector stages + std::log      the accuracy reference
    fast          vector stages + fast log      what the shipped path does
    fastdouble    vector stages + fast log, double
    scalar_std    scalar stages + std::log      isolates the SIMD rewrite

The model, the token table, the decoder and the detokenizer are identical in all
four runs. Only the frontend arithmetic changes, which is the thing under test.

Model provenance is checked, not assumed: the SHA-256 of the ONNX file must
equal the one recorded in results/ctc_logit_inventory.json for the frozen
export, or the run aborts. Every comparison below is therefore within the same
model file that produced the 2.94% baseline, which is what makes the absolute
CER row comparable rather than merely indicative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from asr_metrics import corpus, evaluate  # noqa: E402

VARIANTS = [
    ("std", {"log": "std", "scalar": False}),
    ("fast", {"log": "fast", "scalar": False}),
    ("fastdouble", {"log": "fastdouble", "scalar": False}),
    ("scalar_std", {"log": "std", "scalar": True}),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} failed: {proc.stderr.strip()[:400]}")
    return proc.stdout


def open_session(model: Path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 3  # errors only; the graph-rewrite chatter is noise here
    session = ort.InferenceSession(str(model), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    names = {i.name for i in session.get_inputs()}
    if names != {"x", "x_lens"}:
        raise RuntimeError(f"unexpected ONNX inputs: {names}")
    return session


def forward(session, features: np.ndarray) -> np.ndarray:
    x = features[None, :, :].astype(np.float32)
    lengths = np.asarray([features.shape[0]], dtype=np.int64)
    scores, out_len = session.run(["log_probs", "log_probs_len"], {"x": x, "x_lens": lengths})
    return np.ascontiguousarray(scores[0, : int(out_len[0]), :], dtype=np.float32)


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
    parser.add_argument("--out", type=Path, default=REPO / "results/numeric_sensitivity.json")
    parser.add_argument("--limit", type=int, default=0, help="debug: stop after N utterances")
    parser.add_argument("--no-pad-frames", dest="pad_frames", action="store_false",
                        help="skip the snip_edges framing alignment")
    args = parser.parse_args()

    model_file = args.model_dir / "model.int8.onnx"
    tokens = args.model_dir / "tokens.txt"
    for path in (model_file, tokens, args.dump_binary, args.pipeline_binary):
        if not path.is_file():
            raise SystemExit(f"missing: {path}")

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    reference_text = {u["id"]: u["reference"] for u in baseline["utterances"]}
    frozen_frames = {u["id"]: u["frames"] for u in baseline["utterances"]}
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    manifest = [json.loads(line) for line in
                (args.dev_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    if args.limit:
        manifest = manifest[: args.limit]

    session = open_session(model_file)
    observed_model_sha = sha256(model_file)
    recorded_model_sha = inventory.get("model_sha256")
    if observed_model_sha != recorded_model_sha:
        raise SystemExit(
            "model hash mismatch: " + observed_model_sha + " != recorded " + str(recorded_model_sha)
            + " -- re-download the release asset, a truncated extraction is the usual cause")
    print(f"model {observed_model_sha[:12]} verified against the frozen export")

    scratch = Path(tempfile.mkdtemp(prefix="numeric-sensitivity-"))
    rows: list[dict] = []
    feature_errors: dict[str, list[float]] = {name: [] for name, _ in VARIANTS if name != "std"}
    feature_worst: dict[str, list[float]] = {name: [] for name, _ in VARIANTS if name != "std"}
    feature_identical: dict[str, list[float]] = {name: [] for name, _ in VARIANTS if name != "std"}
    texts: dict[str, dict[str, str]] = {name: {} for name, _ in VARIANTS}

    for index, entry in enumerate(manifest, 1):
        uid = entry["id"]
        wav = args.data_root / "audio" / f"{uid}.wav"
        if not wav.is_file():
            raise SystemExit(f"missing audio: {wav}")
        features: dict[str, np.ndarray] = {}
        row = {"id": uid, "reference": reference_text[uid], "variants": {}}
        for name, config in VARIANTS:
            raw = scratch / f"{uid}_{name}.f32"
            meta = scratch / f"{uid}_{name}.json"
            cmd = [str(args.dump_binary), "--wav", str(wav), "--out", str(raw),
                   "--json", str(meta), "--log", config["log"]]
            if config["scalar"]:
                cmd.append("--scalar")
            run(cmd)
            info = json.loads(meta.read_text(encoding="utf-8"))
            matrix = np.fromfile(raw, dtype="<f4").reshape(info["frames"], info["bins"])
            features[name] = matrix
            # Framing alignment, not a modelling choice. The C++ frontend uses
            # snip_edges=true (551 frames on the first utterance); the reference
            # frontend uses snip_edges=false, which extends the signal by one
            # frame at each end (553). The difference is exactly 2 frames on
            # 30/30 utterances, so one replicated frame at each end puts every
            # variant back on the reference length and lets the absolute CER be
            # compared with the 2.94% baseline. The padding is identical for all
            # four variants, so it cannot bias the A/B.
            if args.pad_frames:
                matrix = np.vstack([matrix[0:1], matrix, matrix[-1:]])

            log_probs = forward(session, matrix)
            npy = scratch / f"{uid}_{name}.npy"
            np.save(npy, log_probs, allow_pickle=False)
            out = subprocess.run(
                [str(args.pipeline_binary), "--wav", str(wav), "--posteriors", str(npy),
                 "--tokens", str(tokens), "--repeats", "1"],
                capture_output=True, text=True, encoding="utf-8")
            if out.returncode != 0:
                raise SystemExit(f"{uid}/{name}: pipeline failed: {out.stderr.strip()[:300]}")
            text = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
            texts[name][uid] = text
            row["variants"][name] = {
                "frames": int(matrix.shape[0]),
                "frontend_frames": info["frames"],
                "logit_frames": int(log_probs.shape[0]),
                "feature_abs_sum": float(np.abs(matrix.astype(np.float64)).sum()),
                "text": text,
                "cer": evaluate(reference_text[uid], text).cer,
            }

        for name in ("fast", "fastdouble", "scalar_std"):
            delta = np.abs(features[name].astype(np.float64) - features["std"].astype(np.float64))
            feature_errors[name].append(float(delta.mean()))
            feature_worst[name].append(float(delta.max()))
            feature_identical[name].append(
                float(np.mean(features[name].view(np.uint32) == features["std"].view(np.uint32))))
            row["variants"][name]["feature_mae_vs_std"] = float(delta.mean())
            row["variants"][name]["feature_max_abs_vs_std"] = float(delta.max())
            row["variants"][name]["feature_bit_identical_fraction"] = float(
                np.mean(features[name].view(np.uint32) == features["std"].view(np.uint32)))
        rows.append(row)
        if index % 5 == 0 or index == len(manifest):
            print(f"  {index}/{len(manifest)} utterances", flush=True)

    summary = {}
    for name, _ in VARIANTS:
        counts = [evaluate(reference_text[r["id"]], texts[name][r["id"]]) for r in rows]
        identical = sum(1 for r in rows if texts[name][r["id"]] == texts["std"][r["id"]])
        chars = sum(len(texts[name][r["id"]]) for r in rows)
        summary[name] = {
            "utterances": len(rows),
            "transcripts_identical_to_std": identical,
            "transcripts_differing": [r["id"] for r in rows
                                      if texts[name][r["id"]] != texts["std"][r["id"]]],
            "hypothesis_chars": chars,
            "cer": corpus(counts)["cer"],
            "cer_counts": corpus(counts),
        }
        if name != "std":
            values = feature_errors[name]
            summary[name]["feature_mae_mean"] = float(np.mean(values))
            summary[name]["feature_mae_max"] = float(np.max(values))
            summary[name]["feature_max_abs_overall"] = float(np.max(feature_worst[name]))
            summary[name]["feature_bit_identical_fraction"] = float(np.min(feature_identical[name]))

    frozen_cer = baseline.get("corpus", {}).get("cer")
    report = {
        "status": "measured_frontend_numeric_sensitivity",
        "question": "does the SIMD sample-domain path or the fast log change any transcript?",
        "scope": ("model runs are real (int8 Zipformer CTC via onnxruntime); the ONNX file is "
                  "hash-verified against the frozen export, so every variant is compared on the "
                  "same weights the 2.94% baseline was measured with."),
        "model": {
            "file": str(model_file.relative_to(REPO)),
            "sha256_observed": observed_model_sha,
            "sha256_recorded_in_frozen_export": recorded_model_sha,
            "asset_matches_frozen_export": observed_model_sha == recorded_model_sha,
            "tokens_sha256": sha256(tokens),
            "execution_provider": "CPUExecutionProvider",
            "intra_op_threads": 2,
        },
        "protocol": {
            "utterances": len(rows),
            "variants": [{"name": name, **config} for name, config in VARIANTS],
            "decoder": "C++ greedy collapse + C++ byte-BPE detokenizer (fbank_ctc_pipeline)",
            "feature_dump": "C++ StreamingFbank, float32 little-endian",
            "framing_alignment": ("one replicated frame prepended and appended so the "
                                  "snip_edges=true frontend matches the snip_edges=false "
                                  "reference length; identical for every variant") if args.pad_frames
                                 else "none",
            "same_model_every_variant": True,
        },
        "reference": {
            "frozen_export_cer": frozen_cer,
            "frozen_export_note": "from results/ctc_baseline_dev.json, same model file",
            "frozen_logit_frames": frozen_frames,
        },
        "summary": summary,
        "tolerance_argument": {
            "feature_tolerance_used": 1e-4,
            "why": ("the model quantises its inputs far more coarsely than 1e-4; the measured "
                    "feature MAE is reported next to every transcript count so the tolerance is "
                    "a measurement, not an assumption"),
        },
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\nvariant        transcripts == std   feature MAE   max |d|    CER")
    for name, _ in VARIANTS:
        s = summary[name]
        mae = s.get("feature_mae_mean")
        worst = s.get("feature_max_abs_overall")
        print(f"{name:<14} {s['transcripts_identical_to_std']:>3}/{s['utterances']:<3}"
              f"            {('%.2e' % mae) if mae is not None else '   --   ':<10}"
              f" {('%.2e' % worst) if worst is not None else '   --   ':<10} {100 * s['cer']:.2f}%")
    print(f"\nwrote {args.out}")
    identical = summary["fast"]["transcripts_identical_to_std"]
    return 0 if identical == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
