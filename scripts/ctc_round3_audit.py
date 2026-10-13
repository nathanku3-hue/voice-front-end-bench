#!/usr/bin/env python3
"""Read-only independent audit of the completed 4 x 50 Test result; no decoding.

Checks custody, preregistration, 200 per-utterance scores, timing aggregates,
blank-transition counters, transcript drift, CSV projection and Pareto selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

import run_ctc_test_eval as runner

from asr_metrics import corpus, evaluate, normalize
from run_ctc_test_eval import (
    CONDITIONS, CSV_COLUMNS, TEST_STATUS, file_sha, pareto_ids,
    project_csv, rows_jsonl, select_badcases,
)


def require_close(actual, expected, what):
    if not (math.isfinite(float(actual)) and math.isfinite(float(expected)) and
            math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-9)):
        raise ValueError(f"Inconsistent {what}: {actual} != {expected}")


def audit(json_path, csv_path, claim_path, test_source,
          protocol_path=Path("docs/CTC_ROUND3_PROTOCOL.md"),
          dev_report_path=Path("results/ctc_dev_sweep_results.json")):
    result = json.loads(Path(json_path).read_text(encoding="utf-8"))
    claim = json.loads(Path(claim_path).read_text(encoding="utf-8"))
    if result.get("status") != TEST_STATUS or claim.get("status") != "COMPLETED":
        raise ValueError("Round 3 result/claim incomplete")
    if file_sha(json_path) != claim.get("result_json_sha256") or \
            file_sha(csv_path) != claim.get("result_csv_sha256"):
        raise ValueError("Final Test results modified after completion")
    if claim.get("script_sha256") != file_sha(Path(runner.__file__)):
        raise ValueError("Pre-registered decoder evaluation script was modified")
    wanted = [c[0] for c in CONDITIONS]
    frozen = result["frozen_input_contract"]
    if (claim.get("conditions") != wanted or
            claim.get("frozen_input_contract") != frozen or
            result["preregistration"].get("locked_conditions") != wanted or
            claim.get("protocol_sha256") != result["preregistration"].get("sha256") or
            claim["protocol_sha256"] != file_sha(protocol_path)):
        raise ValueError("Pre-registration or claim identity mismatch")
    if file_sha(dev_report_path) != frozen["round2_dev_report_sha256"]:
        raise ValueError("Round 2 Dev report changed")
    dev_report = json.loads(Path(dev_report_path).read_text(encoding="utf-8"))
    dev_by_id = {dev_report["framewise_greedy"]["id"]: dev_report["framewise_greedy"]}
    dev_by_id.update({c["id"]: c for c in dev_report["configurations"]})
    dev = result["dev_comparison"]
    if [r["id"] for r in dev] != wanted:
        raise ValueError("Dev comparison changed conditions")
    for d in dev:
        if (d["corpus"] != dev_by_id[d["id"]]["corpus"] or
                d["timing"] != dev_by_id[d["id"]]["timing"]):
            raise ValueError("Dev comparison does not match sealed Dev report")

    source = rows_jsonl(Path(test_source))
    by_id = {r["id"]: r for r in source}
    if (len(source) != 50 or len(by_id) != 50 or
            any(r.get("split") != "test" for r in source)):
        raise ValueError("Wrong held-out Test source or role")
    if file_sha(test_source) != frozen["source_manifest_sha256"]:
        raise ValueError("Test source hash mismatch")
    configurations = result["configurations"]
    if len(configurations) != 4 or [c["id"] for c in configurations] != wanted:
        raise ValueError("Wrong/preselected Test conditions")
    row_maps = {}
    for conf, (cid, width, strategy, threshold) in zip(configurations, CONDITIONS):
        if (conf["id"], conf["beam_width"], conf["strategy"], conf["blank_threshold"]) != \
                (cid, width, strategy, threshold):
            raise ValueError("Unregistered decoder configuration")
        rows = conf["utterances"]
        uid_map = {r["id"]: r for r in rows}
        if len(rows) != 50 or len(uid_map) != 50 or set(uid_map) != set(by_id):
            raise ValueError("Missing, repeated, or extra Test utterance")
        edits = []
        for uid, row in uid_map.items():
            official = by_id[uid]
            if (row["reference"] != official["reference"] or
                    row["speaker_id"] != official["speaker_id"] or
                    row["audio_relpath"] != official["wav"] or
                    row["wav_sha256"] != official["wav_sha256"]):
                raise ValueError(f"Reference/speaker/audio mismatch: {uid}")
            require_close(row["duration_s"], official["duration_s"], f"duration {uid}")
            evaluation = evaluate(row["reference"], row["hypothesis"])
            if (evaluation.s, evaluation.d, evaluation.i, evaluation.n) != \
                    (row["s"], row["d"], row["i"], row["n"]):
                raise ValueError(f"S/D/I audit mismatch: {uid}")
            require_close(row["cer"], evaluation.cer, f"CER {uid}")
            if row["utf8_replacement_chars"] != row["hypothesis"].count("\ufffd"):
                raise ValueError("UTF-8 replacement diagnostic inconsistent")
            times = row["decoder_repeats_ms"]
            if (len(times) != 3 or any(t < 0 or not math.isfinite(t) for t in times) or
                    row["frames"] <= 0):
                raise ValueError("Timing/frame protocol inconsistent")
            require_close(row["decoder_median_ms"], statistics.median(times),
                          f"timing median {uid}")
            nframes = row["frames"]
            triggered = row["triggered_blank_frames"]
            forced = row["forced_blank_frames"]
            skipped = row["hard_skipped_frames"]
            if not 0 <= triggered <= nframes or not 0 <= forced <= nframes or skipped != 0:
                raise ValueError("Blank trigger/counter bounds violated")
            if strategy == "forced_blank" and forced != triggered:
                raise ValueError("Forced Blank semantics violated")
            if strategy != "forced_blank" and (forced or triggered):
                raise ValueError("Unexpected blank shortcut")
            edits.append(evaluation)
        if corpus(edits) != conf["corpus"]:
            raise ValueError("Corpus CER/S/D/I incorrect")
        times = [r["decoder_median_ms"] for r in rows]
        t = conf["timing"]
        for key, observed in (
            ("decoder_mean_ms", statistics.fmean(times)),
            ("decoder_median_ms", statistics.median(times)),
            ("decoder_p95_ms", float(np.percentile(times, 95))),
            ("decoder_sum_of_utterance_medians_ms", sum(times)),
            ("decoder_ms_per_ctc_frame", sum(times)/sum(r["frames"] for r in rows)),
        ):
            require_close(t[key], observed, f"timing aggregate {key}")
        for key in ("utf8_replacement_chars", "triggered_blank_frames",
                    "forced_blank_frames", "hard_skipped_frames"):
            row_key = ("utf8_replacement_chars" if key == "utf8_replacement_chars" else key)
            if conf[key] != sum(r[row_key] for r in rows):
                raise ValueError(f"Incorrect aggregate: {key}")
        if conf["total_ctc_frames"] != sum(r["frames"] for r in rows):
            raise ValueError("Frame total differs")
        for rate, counter in (("triggered_blank_frame_rate", "triggered_blank_frames"),
                              ("forced_blank_frame_rate", "forced_blank_frames"),
                              ("actual_hard_skip_frame_rate", "hard_skipped_frames")):
            require_close(conf[rate], conf[counter] / conf["total_ctc_frames"], rate)
        row_maps[cid] = uid_map
    baseline = row_maps[wanted[0]]
    unpruned = row_maps[wanted[1]]
    for conf in configurations:
        rows = row_maps[conf["id"]]
        hyp_delta = sum(normalize(r["hypothesis"]) != normalize(baseline[uid]["hypothesis"])
                        for uid, r in rows.items())
        token_delta = sum(r["tokens"] != baseline[uid]["tokens"] for uid, r in rows.items())
        beam_delta = sum(normalize(r["hypothesis"]) != normalize(unpruned[uid]["hypothesis"])
                         for uid, r in rows.items())
        if (hyp_delta, token_delta, beam_delta) != (
            conf["hypotheses_different_from_greedy"],
            conf["tokens_different_from_greedy"],
            conf["hypotheses_different_from_beam16_unpruned"],
        ):
            raise ValueError("Observed hypothesis drift count incorrect")
    if result["test_pareto_frontier_ids"] != pareto_ids(configurations):
        raise ValueError("Pareto frontier calculation mismatch")
    if result["badcase_triage"] != select_badcases(configurations, limit=3):
        raise ValueError("Bad-case selection/reconstruction mismatch")
    with Path(csv_path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(CSV_COLUMNS):
            raise ValueError("Summary CSV schema inconsistent")
        actual = list(reader)
    expected = project_csv(result)
    if len(actual) != 4:
        raise ValueError("Summary CSV does not have four rows")
    for got, want in zip(actual, expected):
        for key in CSV_COLUMNS:
            if got[key] != ("" if want[key] is None else str(want[key])):
                raise ValueError(f"Summary CSV mismatch: {key}")
    return dict(status="PASS", held_out_utterances=50,
                configurations=len(configurations), hypotheses_audited=200,
                pareto_frontier=result["test_pareto_frontier_ids"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", type=Path, default=Path("results/ctc_test_final_results.json"))
    p.add_argument("--csv", type=Path, default=Path("results/ctc_test_final_results.csv"))
    p.add_argument("--claim", type=Path, default=Path("results/ctc_round3_unseal_claim.json"))
    p.add_argument("--test-source", type=Path, default=Path("data/ctc_eval/test.jsonl"))
    p.add_argument("--protocol", type=Path, default=Path("docs/CTC_ROUND3_PROTOCOL.md"))
    p.add_argument("--dev-report", type=Path, default=Path("results/ctc_dev_sweep_results.json"))
    args = p.parse_args()
    print(json.dumps(audit(args.json, args.csv, args.claim, args.test_source,
                           args.protocol, args.dev_report), indent=2))


if __name__ == "__main__":
    main()
