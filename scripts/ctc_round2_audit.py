#!/usr/bin/env python3
"""Audit every Round-2 Dev sweep record, CSV, and the held-out Test seal.

Does NOT read, decode, or score the 50 Test .npy files. The audit recomputes
all Dev S/D/I totals from the actual stored hypotheses and source references.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from asr_metrics import corpus, evaluate, normalize
from run_ctc_dev_sweep import BEAMS, THRESHOLDS, grid, config_id, csv_data


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines()
            if s.strip()]


def check_records(utterances, source, *, total_chars, total_frames):
    by_id = {row["id"]: row for row in utterances}
    if len(by_id) != len(source) or set(by_id) != set(source) or len(utterances) != len(source):
        raise ValueError("Dev utterance IDs missing, duplicated or contaminated")
    counts = []
    for uid, row in by_id.items():
        official = source[uid]
        if row["reference"] != official["reference"] or row["speaker_id"] != official["speaker_id"]:
            raise ValueError(f"Reference/speaker drift: {uid}")
        edit = evaluate(row["reference"], row["hypothesis"])
        if (edit.s, edit.d, edit.i, edit.n) != (row["s"], row["d"], row["i"], row["n"]):
            raise ValueError(f"Stored S/D/I is inconsistent: {uid}")
        if not math.isclose(edit.cer, row["cer"], abs_tol=1e-12):
            raise ValueError("Stored per-row CER mismatch")
        if row["utf8_replacement_chars"] != row["hypothesis"].count("\ufffd"):
            raise ValueError("UTF-8 replacement count inconsistent")
        if row["decoder_median_ms"] <= 0 or not math.isfinite(row["decoder_median_ms"]):
            raise ValueError("Nonpositive or nonfinite decode timing")
        counts.append(edit)
    aggregate = corpus(counts)
    if aggregate["reference_chars"] != total_chars:
        raise ValueError("Reference character total changed")
    if sum(row["frames"] for row in by_id.values()) != total_frames:
        raise ValueError("CTC frame total changed")
    return aggregate, by_id


def audit(result_path, csv_path, dev_source, test_source, baseline_path,
          dev_export_root, test_export_root):
    p = json.loads(Path(result_path).read_text(encoding="utf-8"))
    dev_rows, test_rows = jsonl(dev_source), jsonl(test_source)
    source = {x["id"]: x for x in dev_rows}
    test_ids = {x["id"] for x in test_rows}
    if len(source) != 30 or len(test_ids) != 50 or source.keys() & test_ids:
        raise ValueError("30-Dev / 50-Test split compromised")
    if any(row.get("split") != "dev" for row in dev_rows) or any(row.get("split") != "test" for row in test_rows):
        raise ValueError("Split role tags invalid")
    if p["status"] != "ROUND2_DEV_GRID_COMPLETE_TEST_SEALED":
        raise ValueError("Sweep status invalid")
    provenance = p["provenance"]
    if (provenance["dev_source_manifest_sha256"] != sha(dev_source)
        or provenance["test_source_manifest_sha256"] != sha(test_source)
        or provenance["dev_count"] != 30
        or provenance["held_out_test_count"] != 50
        or provenance["utf8_byte_errors"] != "replace"):
        raise ValueError("Frozen split/policy identity changed")
    old = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    if (provenance["historical_greedy_report_sha256"] != sha(baseline_path)
        or old["model"]["model_sha256"] != provenance["model_sha256"]
        or old["byte_errors"] != "replace"):
        raise ValueError("Baseline identity drift")
    if provenance["dev_logits_manifest_sha256"] != sha(Path(dev_export_root) / "manifest.jsonl"):
        raise ValueError("Frozen Dev tensor index identity drift")
    greedy = p["framewise_greedy"]
    n_frames = greedy["total_ctc_frames"]
    n_chars = old["corpus"]["reference_chars"]
    corpus_now, greedy_ids = check_records(greedy["utterances"], source,
                                           total_chars=n_chars, total_frames=n_frames)
    if corpus_now != old["corpus"] or greedy["corpus"] != old["corpus"]:
        raise ValueError("Framewise greedy baseline score no longer matches Round 1")
    prev_ids = {u["id"]: u for u in old["utterances"]}
    for uid, item in greedy_ids.items():
        if (tuple(item["tokens"]) != tuple(prev_ids[uid]["tokens"])
            or normalize(item["hypothesis"]) != normalize(prev_ids[uid]["hypothesis"])):
            raise ValueError("Framewise greedy hypothesis or token sequence changed")

    expected_ids = {config_id(width, kind, tau) for width in BEAMS
                    for kind, tau in grid(width)}
    conditions = p["configurations"]
    if len(conditions) != 36 or {r["id"] for r in conditions} != expected_ids:
        raise ValueError("36-condition matrix incomplete, duplicated or unexpected")
    if p["grid"]["measured_configurations"] != 36 or p["timing_protocol"]["repeats"] < 2:
        raise ValueError("Sweep/timing protocol invalid")
    if p["timing_protocol"]["warmups_per_utterance"] != 1:
        raise ValueError("Missing declared warmups")
    by_key = {r["id"]: r for r in conditions}
    derived = {}
    for width in BEAMS:
        ref_by_id = {r["id"]: r for r in by_key[config_id(width, "none", None)]["utterances"]}
        for kind, threshold in grid(width):
            row = by_key[config_id(width, kind, threshold)]
            aggregate, items = check_records(row["utterances"], source,
                                             total_chars=n_chars, total_frames=n_frames)
            if aggregate != row["corpus"] or row["total_ctc_frames"] != n_frames:
                raise ValueError("Wrong corpus aggregate or CTC frame total")
            triggered = sum(x["triggered_blank_frames"] for x in row["utterances"])
            skips = sum(x["hard_skipped_frames"] for x in row["utterances"])
            forced = sum(x["forced_blank_frames"] for x in row["utterances"])
            if ((triggered, skips, forced) !=
                (row["triggered_blank_frames"], row["hard_skipped_frames"],
                 row["forced_blank_frames"])):
                raise ValueError("Blank shortcut counters differ from utterances")
            if kind == "none" and (triggered or skips or forced):
                raise ValueError("Unpruned beam unexpectedly shortcut frames")
            if kind == "forced_blank" and (forced != triggered or skips != 0):
                raise ValueError("Forced Blank counters violate semantics")
            if kind == "hard_skip" and (skips != triggered or forced != 0):
                raise ValueError("Hard Skip counters violate semantics")
            if not math.isclose(row["actual_hard_skip_frame_rate"], skips / n_frames, abs_tol=1e-12):
                raise ValueError("Hard Skip rate incorrect")
            if not math.isclose(row["forced_blank_frame_rate"], forced / n_frames, abs_tol=1e-12):
                raise ValueError("Forced Blank rate incorrect")
            if not math.isclose(row["triggered_blank_frame_rate"], triggered / n_frames, abs_tol=1e-12):
                raise ValueError("Threshold trigger rate incorrect")
            timings = [x["decoder_median_ms"] for x in row["utterances"]]
            reported = row["timing"]
            for actual, expected in (
                (reported["decoder_sum_of_utterance_medians_ms"], sum(timings)),
                (reported["decoder_mean_ms"], statistics.fmean(timings)),
                (reported["decoder_median_ms"], statistics.median(timings)),
            ):
                if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8):
                    raise ValueError("Decoder timing aggregation inconsistent")
            replacements = sum(x["utf8_replacement_chars"] for x in row["utterances"])
            if row["utf8_replacement_chars"] != replacements:
                raise ValueError("Byte-BPE replacement aggregation invalid")
            different_from_greedy = sum(normalize(item["hypothesis"]) != normalize(greedy_ids[uid]["hypothesis"])
                                        for uid, item in items.items())
            different_from_same_width = sum(normalize(item["hypothesis"]) != normalize(ref_by_id[uid]["hypothesis"])
                                            for uid, item in items.items())
            different_tokens_from_same_width = sum(tuple(item["tokens"]) != tuple(ref_by_id[uid]["tokens"])
                                                   for uid, item in items.items())
            derived[row["id"]] = dict(
                id=row["id"], cer=aggregate["cer"], s=aggregate["substitutions"],
                d=aggregate["deletions"], i=aggregate["insertions"],
                normalized_hypotheses_different_from_framewise_greedy=different_from_greedy,
                normalized_hypotheses_different_from_same_width_unpruned=different_from_same_width,
                token_prefixes_different_from_same_width_unpruned=different_tokens_from_same_width,
                utf8_replacement_chars=replacements,
                median_decoder_ms=reported["decoder_median_ms"],
                speedup_vs_same_width_unpruned=row["decoder_speedup_vs_same_beam_unpruned"],
                threshold_triggered_frames=triggered, hard_skipped_frames=skips,
                forced_blank_frames=forced,
            )
    # Independently verify CSV is a faithful projection of source JSON.
    expected_fields, expected_table = csv_data(p)
    with Path(csv_path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != expected_fields:
            raise ValueError("CSV header/schema differs from JSON")
        actual_table = list(reader)
    if len(actual_table) != 36:
        raise ValueError("CSV row count wrong")
    for actual, expected in zip(actual_table, expected_table):
        if any(actual[col] != str(expected[col]) for col in expected_fields):
            raise ValueError(f"CSV/JSON mismatch: {expected['id']}")
    # The canonical Test export is permitted to contain .npy tensors,
    # but must NOT contain any transcript or CER result.
    if any((Path(test_export_root) / name).exists() for name in
           ("greedy.json", "greedy_replace.json", "native_audit.json",
            "beam.json", "beam_results.json", "decoding_experiments.json")):
        raise ValueError("Held-out Test decoding evidence unexpectedly exists")

    summaries = list(derived.values())
    report = dict(
        status="PASS", scope="CTC Round 2 exploratory Dev study; Test remains unscored",
        source_json_sha256=sha(result_path), source_csv_sha256=sha(csv_path),
        model_sha256=provenance["model_sha256"],
        dev_source_manifest_sha256=sha(dev_source),
        held_out_test_manifest_sha256=sha(test_source),
        actual_conditions=len(summaries), dev_utterances=30,
        source_reference_chars=n_chars, total_ctc_frames=n_frames,
        same_corpus_cer_all_conditions=len({r["cer"] for r in summaries}) == 1,
        corpus_cer_values=sorted({r["cer"] for r in summaries}),
        independent_framewise_greedy_cer=old["corpus"]["cer"],
        comparison_summary=summaries,
        limitations=[
            "Same corpus CER does not imply identical utterance hypotheses or no frame-level approximation.",
            "Framewise argmax Greedy is not Prefix Beam width=1.",
            "Forced Blank computes blank transition; it is not a Hard Skip and not lossless.",
            "Only the Dev subset was scored. The 50 held-out Test tensors are not decoded or scored.",
            "No product/encoder latency or unseen-speaker generalization inferred.",
        ]
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/ctc_dev_sweep_results.json"))
    parser.add_argument("--csv", type=Path, default=Path("results/ctc_dev_sweep_results.csv"))
    parser.add_argument("--dev-source", type=Path, default=Path("data/ctc_eval/dev.jsonl"))
    parser.add_argument("--test-source", type=Path, default=Path("data/ctc_eval/test.jsonl"))
    parser.add_argument("--baseline", type=Path, default=Path("results/ctc_baseline_dev.json"))
    parser.add_argument("--dev-export-root", type=Path, default=Path("build/ctc_round1_dev"))
    parser.add_argument("--test-export-root", type=Path, default=Path("build/ctc_round1_test"))
    parser.add_argument("--output", type=Path, default=Path("results/ctc_round2_audit.json"))
    args = parser.parse_args()
    report = audit(args.results, args.csv, args.dev_source, args.test_source,
                   args.baseline, args.dev_export_root, args.test_export_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], conditions=report["actual_conditions"],
                          all_equal_cer=report["same_corpus_cer_all_conditions"],
                          heldout_sealed=True, output=str(args.output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
