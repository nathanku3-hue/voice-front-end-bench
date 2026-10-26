#!/usr/bin/env python3
"""Experimental REAL C++ Fbank -> ONNX Zipformer-CTC -> text bridge.

Uses the bounded C++ fbank_bounded_dump binary, not sherpa's native feature extractor,
to feed a real public offline CTC model. Runs the unmodified upstream-native
Kaldi/torchaudio frontend on the *same* samples as a control.

IMPORTANT: Round-1 C++ Fbank defaults to snip_edges=true / mel_high=8000 Hz;
the public model's reference features use snip_edges=false / mel_high=7600 Hz.
This is a measured integration experiment, not production compatibility,
speaker-independent CER, endpoint optimization, or a real-time app.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from asr_metrics import evaluate, corpus
from ctc_export_logits import (
    checked_source_rows,
    extract_features,
    forward_log_probs,
    open_ctc_session,
    read_wav,
    sha256,
)
from ctc_greedy_baseline import greedy_ids
from ctc_tokenization import build_detokenizer, load_tokens


def cpp_features(binary: Path, pcm: np.ndarray, chunk_samples: int) -> np.ndarray:
    """Execute actual C++ frontend with a bounded-length feed to its CLI."""
    if chunk_samples < 1:
        raise ValueError("chunk_samples must be positive")
    if not binary.is_file():
        raise FileNotFoundError(f"Build the C++ fbank_dump executable first: {binary}")
    with tempfile.TemporaryDirectory(prefix="cpp_ctc_bridge_") as temp:
        raw = Path(temp) / "samples.raw"
        csv = Path(temp) / "features.csv"
        np.ascontiguousarray(pcm, dtype="<f4").tofile(raw)
        result = subprocess.run(
            [str(binary), str(raw), str(csv), str(chunk_samples)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"C++ Fbank failed: {result.stderr[-2000:]}")
        features = np.loadtxt(csv, delimiter=",", dtype=np.float64, ndmin=2)
        if features.ndim != 2 or features.shape[1] != 80:
            raise ValueError(f"Bad C++ feature output: {features.shape}")
        if not np.isfinite(features).all():
            raise ValueError("C++ frontend emitted NaN/Inf")
        return np.ascontiguousarray(features, dtype=np.float32)


def run(model_dir: Path, manifest: Path, binary: Path, limit: int,
        chunk_samples: int, output: Path) -> dict:
    if not 1 <= limit <= 50:
        raise ValueError("Use 1..50 public evaluation utterances")
    model_dir = model_dir.resolve()
    manifest = manifest.resolve()
    binary = binary.resolve()
    tokens_file = model_dir / "tokens.txt"
    bpe_file = model_dir / "bbpe.model"
    model_file = model_dir / "model.int8.onnx"
    for path in (tokens_file, bpe_file, model_file):
        if not path.is_file():
            raise FileNotFoundError(f"Missing public CTC asset: {path}")
    tokens = load_tokens(tokens_file)
    if tokens[0] != "<blk>":
        raise ValueError("CTC blank ID must be 0")
    detokenize = build_detokenizer(tokens, kind="bbpe", bpe_model=bpe_file,
                                   byte_errors="replace")
    model = open_ctc_session(model_file)
    rows = checked_source_rows(manifest, limit)
    report_rows = []
    native_counts, cpp_counts = [], []
    for item, wav in rows:
        samples = read_wav(wav)
        native = extract_features(samples)
        custom = cpp_features(binary, samples, chunk_samples)
        native_logits, native_shape = forward_log_probs(model, native, len(tokens))
        cpp_logits, cpp_shape = forward_log_probs(model, custom, len(tokens))
        native_text = detokenize(greedy_ids(native_logits, 0))
        cpp_text = detokenize(greedy_ids(cpp_logits, 0))
        native_edits = evaluate(item["reference"], native_text)
        cpp_edits = evaluate(item["reference"], cpp_text)
        native_counts.append(native_edits)
        cpp_counts.append(cpp_edits)
        report_rows.append({
            "id": item["id"], "source_wav_sha256": item["wav_sha256"],
            "reference": item["reference"],
            "model_native": {"hypothesis": native_text, "s": native_edits.s,
                             "d": native_edits.d, "i": native_edits.i,
                             "cer": native_edits.cer, "feature_frames": len(native),
                             "ctc_frames": native_shape["valid_ctc_frames"]},
            "cpp_frontend": {"hypothesis": cpp_text, "s": cpp_edits.s,
                             "d": cpp_edits.d, "i": cpp_edits.i,
                             "cer": cpp_edits.cer, "feature_frames": len(custom),
                             "ctc_frames": cpp_shape["valid_ctc_frames"]},
            "transcripts_match": native_text == cpp_text,
        })
        print(json.dumps({"id": item["id"], "transcripts_match": native_text == cpp_text,
                          "cpp_cer": cpp_edits.cer, "native_cer": native_edits.cer},
                         ensure_ascii=False), flush=True)
    report = {
        "status": "experimental_cpp_fbank_to_real_ctc_inference",
        "model_sha256": sha256(model_file),
        "source_manifest_sha256": sha256(manifest),
        "cpp_binary": str(binary),
        "cpp_frontend_process": "bounded stream-to-CSV subprocess; offline CTC inference",
        "chunk_samples": chunk_samples,
        "source_count": len(rows),
        "native_corpus": corpus(native_counts),
        "cpp_corpus": corpus(cpp_counts),
        "transcripts_matching": sum(r["transcripts_match"] for r in report_rows),
        "conditions": {
            "cpp": {"snip_edges": True, "mel_high_hz": 8000},
            "native": {"snip_edges": False, "mel_high_hz": 7600},
        },
        "limitation": "Raw C++ source features injected into real offline CTC ONNX; "
                      "frontend configurations not yet aligned. No product or streaming "
                      "Transducer equivalence and no live audio latency claim.",
        "utterances": report_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path,
                        default=Path("data/ctc_eval/dev.jsonl"))
    parser.add_argument("--binary", type=Path, default=Path("build/bridge-debug/fbank_bounded_dump"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--chunk-samples", type=int, default=113)
    parser.add_argument("--output", type=Path,
                        default=Path("build/cpp_ctc_integration_pilot.json"))
    args = parser.parse_args()
    r = run(args.model_dir, args.source_manifest, args.binary,
            args.limit, args.chunk_samples, args.output)
    print(json.dumps({"cpp_corpus": r["cpp_corpus"],
                      "native_corpus": r["native_corpus"],
                      "transcripts_matching": r["transcripts_matching"],
                      "source_count": r["source_count"],
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
