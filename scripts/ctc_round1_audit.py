#!/usr/bin/env python3
"""Audit frozen CTC Round-1 assets without decoding or scoring held-out Test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ctc_export_logits import sha256
from ctc_greedy_baseline import validate_log_probs


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def unique_ids(rows):
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate ID in manifest")
    return set(ids)


def audit(data_root, dev_export, test_export, old_manifest, model_file, tokens, bbpe):
    data_root, dev_export, test_export = map(
        lambda x: Path(x).resolve(), (data_root, dev_export, test_export))
    source = json.loads((data_root / "provenance.json").read_text(encoding="utf-8"))
    old = jsonl(old_manifest)
    old_ids = unique_ids(old)
    splits = {"dev": jsonl(data_root / "dev.jsonl"),
              "test": jsonl(data_root / "test.jsonl")}
    expected_counts = {"dev": 30, "test": 50}
    for split, rows in splits.items():
        if len(rows) != expected_counts[split] or any(r.get("split") != split for r in rows):
            raise ValueError(f"Wrong {split} split count/tag")
    dev_ids, test_ids = unique_ids(splits["dev"]), unique_ids(splits["test"])
    if dev_ids & test_ids or old_ids & (dev_ids | test_ids):
        raise ValueError("Dev/Test/old endpoint ID overlap")
    if unique_ids(jsonl(data_root / "manifest.jsonl")) != dev_ids | test_ids:
        raise ValueError("Complete frozen manifest differs")
    for key, filename in (("complete_manifest_sha256", "manifest.jsonl"),
                          ("dev_manifest_sha256", "dev.jsonl"),
                          ("test_manifest_sha256", "test.jsonl")):
        if sha256(data_root / filename) != source[key]:
            raise ValueError("Frozen manifest hash mismatch")
    if sha256(old_manifest) != source["prior_endpoint_manifest_sha256"]:
        raise ValueError("Old endpoint source identity changed")
    all_source_hashes = set()
    for row in splits["dev"] + splits["test"]:
        path = (data_root / row["wav"]).resolve()
        if not path.is_relative_to(data_root):
            raise ValueError("WAV path escapes dataset root")
        if sha256(path) != row["wav_sha256"]:
            raise ValueError(f"WAV identity mismatch: {row['id']}")
        if row["wav_sha256"] in all_source_hashes:
            raise ValueError("Duplicate source WAV bytes")
        all_source_hashes.add(row["wav_sha256"])
    receipts = {}
    model_meta = None
    for split, folder in (("dev", dev_export), ("test", test_export)):
        summary = json.loads((folder / "export_summary.json").read_text(encoding="utf-8"))
        meta = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        if model_meta is None:
            model_meta = meta
        elif model_meta != meta:
            raise ValueError("Dev/Test model contracts differ")
        if summary["num_utterances"] != expected_counts[split]:
            raise ValueError("Incorrect exported count")
        if summary["source_manifest_sha256"] != sha256(data_root / (split + ".jsonl")):
            raise ValueError("Export did not use frozen source manifest")
        manifest = jsonl(folder / "manifest.jsonl")
        if unique_ids(manifest) != unique_ids(splits[split]):
            raise ValueError(f"Different exported {split} IDs")
        by_id = {r["id"]: r for r in splits[split]}
        by_diag = {r["id"]: r for r in summary["diagnostics"]}
        if set(by_diag) != set(by_id):
            raise ValueError("Missing diagnostic rows")
        lengths = []
        for exported in manifest:
            uid = exported["id"]
            original = by_id[uid]
            if (exported["wav_sha256"] != original["wav_sha256"] or
                exported["reference"] != original["reference"]):
                raise ValueError("Exported metadata diverges from frozen dataset")
            tensor = (folder / exported["log_probs"]).resolve()
            if not tensor.is_relative_to(folder):
                raise ValueError("Tensor path escapes export")
            actual_hash = sha256(tensor)
            diag = by_diag[uid]
            if actual_hash != diag["log_probs_sha256"]:
                raise ValueError(f"Tensor hash mismatch: {uid}")
            scores = validate_log_probs(np.load(tensor, allow_pickle=False),
                                        diag["vocab_size"])
            if tuple(scores.shape) != (diag["valid_ctc_frames"], diag["vocab_size"]):
                raise ValueError("Tensor has invalid [T,V] shape")
            lengths.append(scores.shape[0])
        receipts[split] = dict(count=len(manifest), min_ctc_frames=min(lengths),
                               max_ctc_frames=max(lengths),
                               source_manifest_sha256=sha256(data_root / (split + ".jsonl")),
                               export_summary_sha256=sha256(folder / "export_summary.json"))
    for field, value in (("model_sha256", sha256(model_file)),
                         ("token_table_sha256", sha256(tokens)),
                         ("bbpe_model_sha256", sha256(bbpe))):
        if model_meta[field] != value:
            raise ValueError(f"Frozen model asset mismatch: {field}")
    if (model_meta.get("ctc_head") != "log_probs" or
        model_meta.get("output_length_head") != "log_probs_len" or
        model_meta.get("blank_id") != 0 or
        model_meta.get("tokenizer") != "bbpe"):
        raise ValueError("Model CTC output/tokenizer contract has changed")
    # Check that our own workflow has not produced held-out recognition scores.
    if (test_export / "greedy.json").exists() or (test_export / "greedy_replace.json").exists():
        raise ValueError("Held-out Test has already been greedily scored")
    return dict(status="PASS", corpus="AISHELL-1/test",
                protocol="30 dev exploratory, 50 held-out test, utterance-disjoint",
                note="This validates logits/assets, not held-out test accuracy",
                model_sha256=model_meta["model_sha256"],
                full_manifest_sha256=sha256(data_root / "manifest.jsonl"),
                source_speakers=len(source["speaker_counts"]), splits=receipts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/ctc_eval"))
    parser.add_argument("--dev-export", type=Path, default=Path("build/ctc_round1_dev"))
    parser.add_argument("--test-export", type=Path, default=Path("build/ctc_round1_test"))
    parser.add_argument("--old-manifest", type=Path,
                        default=Path("data/round2_endpoint_eval/manifest.jsonl"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("build/ctc_round1_audit.json"))
    args = parser.parse_args()
    output = audit(args.data_root, args.dev_export, args.test_export,
                   args.old_manifest, args.model_dir / "model.int8.onnx",
                   args.model_dir / "tokens.txt", args.model_dir / "bbpe.model")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
