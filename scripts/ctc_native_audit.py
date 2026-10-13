#!/usr/bin/env python3
"""Audit CTC export against sherpa-onnx native greedy decoding, DEV only.

Decoding uses the same model and WAV but potentially different native Fbank
implementations. Therefore this is a transcript-level smoke audit, NOT
feature/encoder activation parity. Never select decoding params from Test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from asr_metrics import corpus, evaluate, normalize
from ctc_export_logits import read_wav, sha256


def audit(model_dir, source_manifest, exported_report, output_path):
    import sherpa_onnx
    model_dir = Path(model_dir).resolve()
    source_manifest = Path(source_manifest).resolve()
    exported_report = json.loads(Path(exported_report).read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows or any(row.get("split") != "dev" for row in rows):
        raise ValueError("Only explicitly designated Dev data may be decoded in this audit")
    original = {r["id"]: r for r in exported_report["utterances"]}
    if len(original) != len(rows) or set(original) != {row["id"] for row in rows}:
        raise ValueError("Exported and source manifests contain different utterances")
    model_file = model_dir / "model.int8.onnx"
    tokens_file = model_dir / "tokens.txt"
    if sha256(model_file) != exported_report["model"]["model_sha256"]:
        raise ValueError("Model hash mismatch")
    if sha256(tokens_file) != exported_report["token_table_sha256"]:
        raise ValueError("Token hash mismatch")
    recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
        model=str(model_file), tokens=str(tokens_file), num_threads=2,
        sample_rate=16000, feature_dim=80, decoding_method="greedy_search")

    details, edits = [], []
    for row in rows:
        wave = (source_manifest.parent / row["wav"]).resolve()
        if not wave.is_relative_to(source_manifest.parent):
            raise ValueError("WAV path escapes dataset directory")
        if sha256(wave) != row["wav_sha256"]:
            raise ValueError("Audio hash mismatch for " + row["id"])
        if original[row["id"]]["wav_sha256"] != row["wav_sha256"]:
            raise ValueError("Exported row audio hash mismatch")
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, read_wav(wave))
        recognizer.decode_stream(stream)
        native = stream.result.text
        counts = evaluate(row["reference"], native)
        edits.append(counts)
        own = original[row["id"]]
        details.append(dict(
            id=row["id"], reference=row["reference"],
            exported_greedy=own["hypothesis"], native_greedy=native,
            match=normalize(own["hypothesis"]) == normalize(native),
            exported_utf8_replacement_chars=own["utf8_replacement_chars"],
            native_s=counts.s, native_d=counts.d, native_i=counts.i, native_cer=counts.cer,
        ))
    result = dict(
        scope="Dev-only native sherpa-onnx transcript cross-check (not feature parity)",
        source_manifest_sha256=sha256(source_manifest),
        model_sha256=sha256(model_file), rows=len(rows),
        normalized_hypotheses_equal=sum(row["match"] for row in details),
        normalized_hypotheses_different=sum(not row["match"] for row in details),
        exported_corpus=exported_report["corpus"], native_corpus=corpus(edits),
        utf8_replacement_chars=exported_report["utf8_replacement_chars"],
        note="Native frontend options and torchaudio Fbank need not match exactly; inspect discordant rows.",
        utterances=details,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--exported-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("build/ctc_round1_dev/native_audit.json"))
    args = parser.parse_args()
    result = audit(args.model_dir, args.source_manifest, args.exported_report, args.output)
    print(json.dumps(dict(rows=result["rows"],
                          equal=result["normalized_hypotheses_equal"],
                          different=result["normalized_hypotheses_different"],
                          native_corpus=result["native_corpus"]),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
