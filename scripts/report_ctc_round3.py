#!/usr/bin/env python3
"""After the one-shot run and independent audit, write a factual Round 3 report.

Does NOT decode Test logits or fabricate audio causality. CV claims should only
be copied into the portfolio after reviewing this report and listening to audio.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctc_round3_audit import audit
from run_ctc_test_eval import CONDITIONS, TEST_STATUS


def document(report, audit_result):
    if report["status"] != TEST_STATUS or audit_result["status"] != "PASS":
        raise ValueError("Only audited completed Test results can be documented")
    confs = report["configurations"]
    devs = report["dev_comparison"]
    rows = [
        "# CTC Round 3 — held-out Test evaluation status",
        "",
        "**Status:** Held-out Test one-shot completed and machine-audited; "
        "manual audio-level bad-case attribution remains a separate review.",
        "",
        "## Protocol and custody",
        "",
        f"- Pre-registration SHA-256: `{report['preregistration']['sha256']}`.",
        f"- Frozen Test manifest SHA-256: `{report['frozen_input_contract']['source_manifest_sha256']}`.",
        f"- Model SHA-256: `{report['frozen_input_contract']['model_sha256']}`.",
        "- Exactly four conditions fixed from the 30 Dev samples before evaluating 50 Test samples.",
        "- Independent audit: 4 × 50 utterance hypotheses, corpus edit counts, UTF-8 replacements, "
        "timings, four-row CSV, claim file, and Pareto identities: PASS.",
        "- Data are utterance-disjoint, **not** speaker-disjoint; original training overlap unverified.",
        "",
        "## Test vs Dev — observed, not forecast",
        "",
        "| Locked decoder | Dev corpus CER | Test corpus CER | Test S/D/I | "
        "Test mean decoder-only ms/utt | Test median decoder-only ms/utt | Replacement chars |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c, dev in zip(confs, devs):
        e, t = c["corpus"], c["timing"]
        rows.append(
            f"| `{c['id']}` | {dev['corpus']['cer']*100:.2f}% | {e['cer']*100:.2f}% | "
            f"{e['substitutions']}/{e['deletions']}/{e['insertions']} | "
            f"{t['decoder_mean_ms']:.3f} | {t['decoder_median_ms']:.3f} | "
            f"{c['utf8_replacement_chars']} |"
        )
    frontier = report["test_pareto_frontier_ids"]
    base = confs[0]
    full = confs[1]
    rows += [
        "", "## Measured decision evidence", "",
        "The comparisons below use sums of per-utterance medians (not ratios of group medians).",
    ]
    for c in confs[1:]:
        ratio = c["timing"]["decoder_sum_of_utterance_medians_ms"] / base["timing"]["decoder_sum_of_utterance_medians_ms"]
        rows.append(
            f"- `{c['id']}`: {ratio:.1f}× Greedy decoder time; "
            f"{c['hypotheses_different_from_greedy']}/50 normalized hypotheses differ "
            f"and {c['tokens_different_from_greedy']}/50 token sequences differ."
        )
    for c in confs[2:]:
        speedup = full["timing"]["decoder_sum_of_utterance_medians_ms"] / c["timing"]["decoder_sum_of_utterance_medians_ms"]
        rows.append(
            f"- `{c['id']}` vs Beam 16 unpruned: {speedup:.2f}× decoder-only "
            f"speedup (different beam width if W=4); "
            f"{c['hypotheses_different_from_beam16_unpruned']}/50 normalized hypotheses differ; "
            f"Forced Blank triggered on {100*c['triggered_blank_frame_rate']:.2f}% of frames."
        )
    rows += [
        "", "## Pareto assessment", "",
        f"Test non-dominated condition(s), using mean of utterance decoder medians "
        f"vs corpus CER: **{', '.join('`'+x+'`' for x in frontier)}**.",
        "All four conditions remain in the reported plot and results. "
        "CER equality does not imply identical token sequences or normalized hypotheses.",
        "", "## Deterministically selected bad-case evidence", "",
        "Selection rule: highest *Greedy* absolute edit count (S+D+I), then utterance ID. "
        "This is an audit queue, not a confirmed causal acoustic diagnosis.", "",
    ]
    for case in report["badcase_triage"]:
        rows += [
            f"### `{case['id']}`", "",
            f"- Reference: {case['reference'].rstrip()}",
            f"- Source audio: `data/ctc_eval/{case['audio_relpath']}`; "
            f"duration: {case['duration_s']:.3f} s; CTC frames: {case['frames']}; "
            f"mean blank posterior: {case['avg_blank_p']:.4f}; "
            f"mean maximum frame posterior: {case['avg_frame_max_p']:.4f}.",
        ]
        for cid, hypothesis in case["hypotheses"].items():
            rows.append(f"- `{cid}`: {hypothesis['hypothesis']} "
                        f"(S/D/I={hypothesis['s']}/{hypothesis['d']}/{hypothesis['i']}, "
                        f"U+FFFD={hypothesis['utf8_replacement_chars']}).")
        rows += ["- **Manual listening status:** not performed by this script; "
                 "do not claim homophone/deletion/acoustic cause until verified.", ""]
    if not report["badcase_triage"]:
        rows += ["No Greedy errors in these 50 held-out utterances; "
                 "the pre-registered selection rule therefore yielded no error cases.", ""]
    rows += [
        "## Measurement boundary", "",
        "Same local sequential Python decoder-only timing as Round 2: "
        "one warmup and three repetitions per utterance/config; exclude WAV, Fbank, "
        "ONNX inference, disk I/O, BBPE text reconstruction and CER scoring.",
        "No Language Model or hidden token Top-K. Forced Blank is an approximate restriction.",
        "No universal accuracy result, lossless pruning, deployment RTF, "
        "speaker-disjoint generalization, or independent training-data overlap audit is claimed.",
        "", "## Evidence", "",
        "- `docs/CTC_ROUND3_PROTOCOL.md`",
        "- `results/ctc_round3_unseal_claim.json`",
        "- `results/ctc_test_final_results.json` and `.csv`",
        "- `results/ctc_test_pareto.png`",
        "- `scripts/ctc_round3_audit.py`",
    ]
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results/ctc_test_final_results.json"))
    p.add_argument("--csv", type=Path, default=Path("results/ctc_test_final_results.csv"))
    p.add_argument("--claim", type=Path, default=Path("results/ctc_round3_unseal_claim.json"))
    p.add_argument("--test-source", type=Path, default=Path("data/ctc_eval/test.jsonl"))
    p.add_argument("--output", type=Path, default=Path("docs/CTC_ROUND3_STATUS.md"))
    args = p.parse_args()
    report = json.loads(args.results.read_text(encoding="utf-8"))
    evidence = audit(args.results, args.csv, args.claim, args.test_source)
    if args.output.exists():
        raise FileExistsError(f"Not overwriting Round 3 status: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document(report, evidence)+"\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
