#!/usr/bin/env python3
"""Reproducible CTC greedy baseline from *real, externally exported* log probabilities.

Input JSON manifest records: {"id": "...", "reference": "...", "log_probs": "relative.npy"}.
Token file: one token per line, integer ID = zero-based line index. Blank ID must
be explicit. This tool does not pretend a Transducer's Joiner scores are CTC logits.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from asr_metrics import evaluate, corpus
from ctc_tokenization import build_detokenizer, load_tokens


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_log_probs(array, vocab_size, *, atol=2e-3):
    scores = np.asarray(array)
    if scores.ndim != 2 or not scores.shape[0] or scores.shape[1] != vocab_size:
        raise ValueError(f"Expected nonempty [T, {vocab_size}] log_probs, got {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Nonfinite log probabilities")
    if np.max(scores) > atol:
        raise ValueError("Positive values inconsistent with normalized log probabilities")
    # Verify normalized probabilities via stable log-sum-exp.
    maxima = np.max(scores.astype(np.float64), axis=1)
    lse = maxima + np.log(np.exp(scores.astype(np.float64) - maxima[:, None]).sum(axis=1))
    if not np.allclose(lse, 0.0, atol=atol, rtol=0):
        raise ValueError("Rows are not normalized log probabilities; apply log_softmax upstream")
    return scores


def greedy_ids(log_probs, blank_id):
    path = np.argmax(log_probs, axis=1)
    output = []
    previous = None
    for token_id in path:
        token_id = int(token_id)
        if token_id != blank_id and token_id != previous:
            output.append(token_id)
        previous = token_id
    return output


def decode(ids, tokens):
    return "".join(tokens[i] for i in ids)


def run(manifest_path, tokens_path, blank_id, output_path, model_info, *, tokenizer="char", bpe_model=None, byte_errors="strict"):
    manifest_path = Path(manifest_path).resolve()
    tokens_path = Path(tokens_path).resolve()
    tokens = load_tokens(tokens_path)
    if not 0 <= blank_id < len(tokens):
        raise ValueError("Invalid blank ID")
    detokenize = build_detokenizer(tokens, kind=tokenizer, bpe_model=bpe_model, byte_errors=byte_errors)
    model_info = Path(model_info).resolve()
    meta = json.loads(model_info.read_text(encoding="utf-8"))
    for field in ("model_name", "model_sha256", "ctc_head", "frame_stride_ms"):
        if field not in meta:
            raise ValueError(f"Missing model metadata: {field}")
    if not meta["ctc_head"] or not isinstance(meta["frame_stride_ms"], (int, float)) or meta["frame_stride_ms"] <= 0:
        raise ValueError("Invalid CTC model declaration")
    for field, value in (("blank_id", blank_id), ("tokenizer", tokenizer),
                         ("token_table_sha256", sha256(tokens_path)),
                         ("bbpe_model_sha256", sha256(bpe_model) if bpe_model else None)):
        if field in meta and meta[field] != value:
            raise ValueError(f"Model metadata disagrees with actual {field}")
    records = []
    counts = []
    used = set()
    for line_num, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        uid = str(item["id"])
        if uid in used:
            raise ValueError(f"Duplicate utterance ID: {uid}")
        used.add(uid)
        tensor_path = (manifest_path.parent / item["log_probs"]).resolve()
        if not tensor_path.is_relative_to(manifest_path.parent):
            raise ValueError("Log probabilities escape manifest directory")
        if tensor_path.suffix != ".npy":
            raise ValueError("Expected a .npy file")
        scores = validate_log_probs(np.load(tensor_path, allow_pickle=False), len(tokens))
        ids = greedy_ids(scores, blank_id)
        hypothesis = detokenize(ids)
        edits = evaluate(item["reference"], hypothesis)
        counts.append(edits)
        records.append(dict(id=uid, reference=item["reference"], hypothesis=hypothesis,
                            tokens=ids, frames=int(scores.shape[0]), logits_sha256=sha256(tensor_path),
                            wav_sha256=item.get("wav_sha256"), speaker_id=item.get("speaker_id"),
                            source_split=item.get("source_split"),
                            utf8_replacement_chars=hypothesis.count("\ufffd"),
                            s=edits.s, d=edits.d, i=edits.i, n=edits.n, cer=edits.cer))
    if not records:
        raise ValueError("No evaluation records")
    report = dict(status="measured_from_provided_logits", model=meta,
                  token_table_sha256=sha256(tokens_path), blank_id=blank_id,
                  tokenizer=tokenizer, byte_errors=byte_errors,
                  bpe_model_sha256=sha256(bpe_model) if bpe_model else None,
                  manifest_sha256=sha256(manifest_path), corpus=corpus(counts),
                  utf8_replacement_chars=sum(x["utf8_replacement_chars"] for x in records),
                  utterances_with_utf8_replacement=sum(x["utf8_replacement_chars"] > 0 for x in records),
                  utterances=records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--blank-id", type=int, required=True)
    parser.add_argument("--model-info", type=Path, required=True)
    parser.add_argument("--tokenizer", choices=("char", "bbpe"), default="char")
    parser.add_argument("--bpe-model", type=Path)
    parser.add_argument("--byte-errors", choices=("strict", "replace"), default="strict",
                        help="replace preserves malformed Byte-BPE as U+FFFD; report replacement count")
    parser.add_argument("--output", type=Path, default=Path("build/ctc_baseline_greedy.json"))
    args = parser.parse_args()
    report = run(args.manifest, args.tokens, args.blank_id, args.output, args.model_info,
                 tokenizer=args.tokenizer, bpe_model=args.bpe_model, byte_errors=args.byte_errors)
    print(json.dumps(dict(corpus=report["corpus"], output=str(args.output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
