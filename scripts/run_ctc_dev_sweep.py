#!/usr/bin/env python3
"""DEV-ONLY CTC decoding experiment: 4 beam widths x 9 strategies = 36 configs.

Inputs are existing, hashed, frozen CTC log probabilities. Test .npy files are
never loaded or decoded. Timing surrounds *only* decoder functions, excluding
numpy I/O, detokenization, error scoring and ONNX inference. All 1000 tokens
remain eligible on each non-shortcut frame; there is no hidden acoustic Top-K.

Run each width separately to bound wall time, then --merge-only for the final
matrix and CSV. File payloads are reproducible except machine-dependent timings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from asr_metrics import corpus, evaluate, normalize
from ctc_greedy_baseline import greedy_ids, validate_log_probs
from ctc_prefix_beam_search import prefix_beam_search
from ctc_tokenization import build_detokenizer, load_tokens

BEAMS = (1, 4, 8, 16)
THRESHOLDS = (0.90, 0.95, 0.98, 0.99)
POLICY = "replace"
STRATEGIES = ("forced_blank", "hard_skip")


def file_sha(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def config_id(width, strategy, tau):
    if strategy == "none":
        return f"prefix_w{width:02d}_unpruned"
    return f"prefix_w{width:02d}_{strategy}_t{tau:.2f}"


def grid(width):
    if width not in BEAMS:
        raise ValueError(f"Unexpected width: {width}")
    return [("none", None)] + [(strategy, tau) for strategy in STRATEGIES
                               for tau in THRESHOLDS]


def load_dev(dev_root, data_root, tokens_path, bpe_path, inventory_path, previous_greedy_path):
    """Fail closed unless source and export are the *frozen Dev* contract."""
    dev_root, data_root = Path(dev_root).resolve(), Path(data_root).resolve()
    inventory_path, previous_greedy_path = Path(inventory_path).resolve(), Path(previous_greedy_path).resolve()
    tokens_path, bpe_path = Path(tokens_path).resolve(), Path(bpe_path).resolve()
    dev_src = data_root / "dev.jsonl"
    test_src = data_root / "test.jsonl"
    src = rows_jsonl(dev_src)
    sealed_test_ids = {r["id"] for r in rows_jsonl(test_src)}  # IDs only. Never read test Logits.
    if len(src) != 30 or len(sealed_test_ids) != 50 or any(r.get("split") != "dev" for r in src):
        raise ValueError("Missing 30-Dev / 50-Test frozen study split")
    if any(r["id"] in sealed_test_ids for r in src):
        raise ValueError("Held-out Test ID leaked into Dev")
    manifest = rows_jsonl(dev_root / "manifest.jsonl")
    if len(manifest) != len(src) or {r["id"] for r in manifest} != {r["id"] for r in src}:
        raise ValueError("Export/source Dev IDs differ")
    if len({r["id"] for r in src}) != 30:
        raise ValueError("Duplicated Dev IDs")
    source_map = {r["id"]: r for r in src}
    inv = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inv.get("status") != "frozen_real_ctc_log_probabilities":
        raise ValueError("Unrecognized frozen Logits inventory")
    entries = {r["id"]: r for r in inv["splits"]["dev"]["items"]}
    if set(entries) != set(source_map):
        raise ValueError("Wrong frozen Dev Logits inventory")
    if inv["splits"]["dev"]["source_manifest_sha256"] != file_sha(dev_src):
        raise ValueError("Dev source Manifest hash mismatch")
    source_provenance = json.loads((data_root / "provenance.json").read_text(encoding="utf-8"))
    if source_provenance["dev_manifest_sha256"] != file_sha(dev_src):
        raise ValueError("Dev source provenance changed")
    if source_provenance["test_manifest_sha256"] != file_sha(test_src):
        raise ValueError("Held-out source manifest changed")
    model = json.loads((dev_root / "model.json").read_text(encoding="utf-8"))
    for field, expected in (("model_sha256", inv["model_sha256"]),
                            ("token_table_sha256", file_sha(tokens_path)),
                            ("bbpe_model_sha256", file_sha(bpe_path)),
                            ("blank_id", 0), ("tokenizer", "bbpe"),
                            ("ctc_head", "log_probs")):
        if model.get(field) != expected:
            raise ValueError(f"Frozen model contract mismatch: {field}")
    if file_sha(tokens_path) != inv["token_table_sha256"] or file_sha(bpe_path) != inv["bbpe_model_sha256"]:
        raise ValueError("Token/BPE fingerprints changed")
    tokens = load_tokens(tokens_path)
    if len(tokens) != 1000 or tokens[0] != "<blk>":
        raise ValueError("Unexpected CTC blank/vocabulary contract")
    detokenize = build_detokenizer(tokens, kind="bbpe", bpe_model=bpe_path, byte_errors=POLICY)
    previous = json.loads(previous_greedy_path.read_text(encoding="utf-8"))
    if (previous.get("manifest_sha256") != file_sha(dev_root / "manifest.jsonl") or
        previous.get("model", {}).get("model_sha256") != model["model_sha256"] or
        previous.get("byte_errors") != POLICY or
        previous.get("blank_id") != 0):
        raise ValueError("Frozen DEV baseline identity/policy changed")
    prev_by_id = {r["id"]: r for r in previous["utterances"]}
    if set(prev_by_id) != set(source_map):
        raise ValueError("Recorded baseline Dev IDs diverge")

    inputs = []
    for item in manifest:
        uid = item["id"]
        source = source_map[uid]
        if item["reference"] != source["reference"] or item["wav_sha256"] != source["wav_sha256"]:
            raise ValueError("Manifest reference/audio SHA changed")
        tensor = (dev_root / item["log_probs"]).resolve()
        if not tensor.is_relative_to(dev_root) or tensor.suffix != ".npy":
            raise ValueError("Dev Logits path escapes frozen export")
        if file_sha(tensor) != entries[uid]["log_probs_sha256"]:
            raise ValueError(f"Dev Logits SHA-256 mismatch: {uid}")
        scores = validate_log_probs(np.load(tensor, allow_pickle=False), 1000)
        if scores.shape[0] != entries[uid]["ctc_frames"]:
            raise ValueError("Frozen CTC output frame count changed")
        inputs.append(dict(id=uid, reference=source["reference"], speaker_id=source["speaker_id"],
                           log_probs=scores, wav_sha256=source["wav_sha256"],
                           logits_sha256=entries[uid]["log_probs_sha256"],
                           previous=prev_by_id[uid]))
    identity = dict(
        dataset="AISHELL-1/test-subset; exploratory Dev only",
        dev_count=30, held_out_test_count=50, reference_chars=previous["corpus"]["reference_chars"],
        model_sha256=model["model_sha256"],
        dev_source_manifest_sha256=file_sha(dev_src),
        test_source_manifest_sha256=file_sha(test_src),
        dev_logits_manifest_sha256=file_sha(dev_root / "manifest.jsonl"),
        token_table_sha256=file_sha(tokens_path),
        bpe_model_sha256=file_sha(bpe_path),
        frozen_inventory_sha256=file_sha(inventory_path),
        historical_greedy_report_sha256=file_sha(previous_greedy_path),
        blank_id=0, tokenizer="bbpe", utf8_byte_errors=POLICY,
        acoustic_token_pruning="none; all 999 nonblank tokens eligible per expanded frame",
        decoding_scope="CTC decoder only; no ONNX, audio I/O or UTF-8 detokenization in timing",
    )
    return inputs, detokenize, previous, identity


def measurement(fn, *, repeats):
    # One unmeasured warmup for *each utterance/config*, to reduce first-call
    # allocation/import effects; repeat and verify deterministic output.
    first = fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        value = fn()
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        if value.tokens != first.tokens:
            raise AssertionError("Non-deterministic decoder tokens")
        times.append(elapsed_ms)
    return first, statistics.median(times), times


def summarize_timing(per_row):
    times = [row["decoder_median_ms"] for row in per_row]
    frames = [row["frames"] for row in per_row]
    return dict(
        decoder_mean_ms=statistics.fmean(times),
        decoder_median_ms=statistics.median(times),
        decoder_p95_ms=float(np.percentile(times, 95)),
        decoder_sum_of_utterance_medians_ms=float(sum(times)),
        decoder_ms_per_ctc_frame=float(sum(times) / sum(frames)),
    )


def make_greedy_baseline(inputs, detokenize, previous, *, repeats):
    outcomes = []
    for item in inputs:
        scores = item["log_probs"]
        def decode():
            return greedy_ids(scores, 0)
        # Greedy decoder is not a prefix search; timing is argmax+collapse.
        first = tuple(decode())
        _ = decode()  # warmup
        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            ids = tuple(decode())
            elapsed.append((time.perf_counter_ns() - start) / 1e6)
            if ids != first:
                raise AssertionError("Non-deterministic Greedy")
        hyp = detokenize(first)
        prior = item["previous"]
        if first != tuple(prior["tokens"]) or normalize(hyp) != normalize(prior["hypothesis"]):
            raise ValueError("Framewise Greedy fails to reproduce Round 1 baseline")
        edits = evaluate(item["reference"], hyp)
        outcomes.append(dict(id=item["id"], reference=item["reference"], hypothesis=hyp,
                             tokens=list(first), speaker_id=item["speaker_id"],
                             frames=int(scores.shape[0]),
                             decoder_median_ms=statistics.median(elapsed),
                             utf8_replacement_chars=hyp.count("\ufffd"),
                             s=edits.s, d=edits.d, i=edits.i, n=edits.n, cer=edits.cer))
    result = dict(
        id="framewise_greedy", decoder="Framewise argmax -> CTC collapse",
        strategy="independent", beam_width=None, blank_threshold=None,
        corpus=corpus([evaluate(row["reference"], row["hypothesis"]) for row in outcomes]),
        timing=summarize_timing(outcomes),
        total_ctc_frames=sum(row["frames"] for row in outcomes),
        utf8_replacement_chars=sum(row["utf8_replacement_chars"] for row in outcomes),
        utterances=outcomes,
    )
    if result["corpus"] != previous["corpus"]:
        raise ValueError("New Greedy baseline differs from frozen Round 1 S/D/I")
    return result


def one_configuration(inputs, detokenize, *, width, strategy, tau, repeats):
    utterances, counts = [], []
    for item in inputs:
        scores = item["log_probs"]
        def decode():
            return prefix_beam_search(scores, beam_width=width, blank_id=0,
                                      strategy=strategy, blank_threshold=tau)
        output, elapsed, _ = measurement(decode, repeats=repeats)
        hyp = detokenize(output.tokens)  # deliberately outside timed region
        edits = evaluate(item["reference"], hyp)
        counts.append(edits)
        utterances.append(dict(
            id=item["id"], speaker_id=item["speaker_id"],
            reference=item["reference"], hypothesis=hyp,
            tokens=list(output.tokens), frames=output.frames,
            decoder_median_ms=elapsed, s=edits.s, d=edits.d, i=edits.i,
            n=edits.n, cer=edits.cer, utf8_replacement_chars=hyp.count("\ufffd"),
            triggered_blank_frames=output.triggered_blank_frames,
            forced_blank_frames=output.forced_blank_frames,
            hard_skipped_frames=output.hard_skipped_frames,
            expanded_nonblank_arcs=output.expanded_nonblank_arcs,
            candidate_states_considered=output.candidate_states_considered,
            max_active_prefixes=output.max_active_prefixes,
        ))
    total_frames = sum(row["frames"] for row in utterances)
    triggered = sum(row["triggered_blank_frames"] for row in utterances)
    hard_skips = sum(row["hard_skipped_frames"] for row in utterances)
    forced = sum(row["forced_blank_frames"] for row in utterances)
    report = dict(
        id=config_id(width, strategy, tau),
        decoder="CTC log-space Prefix Beam Search",
        beam_width=width, strategy=strategy, blank_threshold=tau,
        corpus=corpus(counts), timing=summarize_timing(utterances),
        total_ctc_frames=total_frames,
        triggered_blank_frames=triggered,
        triggered_blank_frame_rate=triggered / total_frames,
        hard_skipped_frames=hard_skips,
        actual_hard_skip_frame_rate=hard_skips / total_frames,
        forced_blank_frames=forced,
        forced_blank_frame_rate=forced / total_frames,
        nonblank_arcs_expanded=sum(row["expanded_nonblank_arcs"] for row in utterances),
        candidate_states_considered=sum(row["candidate_states_considered"] for row in utterances),
        utf8_replacement_chars=sum(row["utf8_replacement_chars"] for row in utterances),
        utterances=utterances,
    )
    return report


def measure_width(inputs, detokenize, previous, identity, width, repeats):
    if repeats < 1 or repeats > 20:
        raise ValueError("Expected repeats 1..20")
    results = []
    for strategy, tau in grid(width):
        result = one_configuration(inputs, detokenize, width=width,
                                   strategy=strategy, tau=tau, repeats=repeats)
        results.append(result)
        print(json.dumps(dict(
            id=result["id"], cer=result["corpus"]["cer"],
            decoder_median_ms=result["timing"]["decoder_median_ms"],
            hard_skip_rate=result["actual_hard_skip_frame_rate"],
            triggered_frame_rate=result["triggered_blank_frame_rate"],
        ), ensure_ascii=False), flush=True)
    return dict(
        status="measured_from_dev_only",
        width=width,
        timing_protocol=dict(clock="time.perf_counter_ns", repeats=repeats,
                             warmups_per_utterance=1, backend=platform.python_implementation(),
                             python=platform.python_version(),
                             numpy=np.__version__, platform=platform.platform(),
                             threading="sequential Python decoder; all log probs preloaded",
                             included="prefix_beam_search() only; no model/audio file I/O, detokenize, scoring"),
        frozen_input_contract=identity,
        framewise_greedy=(make_greedy_baseline(inputs, detokenize, previous, repeats=repeats)
                          if width == BEAMS[0] else None),
        configs=results,
    )


def merge_parts(parts):
    if len(parts) != len(BEAMS):
        raise ValueError("Need all four beam-width partials")
    by_width = {}
    identity = None
    clock = None
    for part in parts:
        width = part["width"]
        if width in by_width or width not in BEAMS:
            raise ValueError("Duplicate/invalid width")
        if part["status"] != "measured_from_dev_only" or len(part["configs"]) != 9:
            raise ValueError("Incomplete width partial")
        if identity is None:
            identity = part["frozen_input_contract"]
            clock = part["timing_protocol"]
        elif part["frozen_input_contract"] != identity or part["timing_protocol"] != clock:
            raise ValueError("Incompatible input or timing protocol")
        by_width[width] = part
    if set(by_width) != set(BEAMS):
        raise ValueError("Missing width")
    greedy = by_width[BEAMS[0]]["framewise_greedy"]
    if not greedy or greedy["corpus"]["labeled_utterances"] != 30:
        raise ValueError("Missing measured independent Greedy baseline")
    results = []
    for width in BEAMS:
        part = by_width[width]
        expected_ids = {config_id(width, strategy, tau) for strategy, tau in grid(width)}
        if {r["id"] for r in part["configs"]} != expected_ids:
            raise ValueError("Wrong/incomplete configuration matrix")
        unpruned = next(r for r in part["configs"] if r["strategy"] == "none")
        for config in part["configs"]:
            if len(config["utterances"]) != 30 or config["corpus"]["labeled_utterances"] != 30:
                raise ValueError("Incomplete Dev sample results")
            if config["corpus"]["reference_chars"] != identity["reference_chars"]:
                raise ValueError("Dev character count differs")
            config["cer_delta_vs_framewise_greedy"] = config["corpus"]["cer"] - greedy["corpus"]["cer"]
            config["decoder_time_ratio_vs_framewise_greedy"] = (
                config["timing"]["decoder_sum_of_utterance_medians_ms"] /
                greedy["timing"]["decoder_sum_of_utterance_medians_ms"]
            )
            config["decoder_time_ratio_vs_same_beam_unpruned"] = (
                config["timing"]["decoder_sum_of_utterance_medians_ms"] /
                unpruned["timing"]["decoder_sum_of_utterance_medians_ms"]
            )
            config["decoder_speedup_vs_same_beam_unpruned"] = (
                unpruned["timing"]["decoder_sum_of_utterance_medians_ms"] /
                config["timing"]["decoder_sum_of_utterance_medians_ms"]
            )
            config["nonblank_arcs_avoided_vs_same_beam_unpruned"] = (
                1.0 - config["nonblank_arcs_expanded"] / unpruned["nonblank_arcs_expanded"]
            )
            results.append(config)
    # Be explicit: this grid is exploratory; final Test has NOT been decoded.
    return dict(
        status="ROUND2_DEV_GRID_COMPLETE_TEST_SEALED",
        hypothesis_selection_scope="30 Dev only; no held-out Test CER or hypotheses generated",
        provenance=identity, timing_protocol=clock,
        grid=dict(beam_widths=list(BEAMS), blank_thresholds=[None, *THRESHOLDS],
                  shortcut_strategies=list(STRATEGIES), measured_configurations=len(results)),
        framewise_greedy=greedy, configurations=results,
        caveats=[
            "Beam width 1 Prefix Beam Search is NOT framewise Greedy.",
            "Both blank shortcut policies approximate the full CTC probability distribution.",
            "Only hard_skip leaves blank/nonblank states unchanged; forced_blank takes a restricted blank transition.",
            "All 999 nonblank token types are eligible on each expanded frame; no hidden acoustic Top-K.",
            "Timings are Python decoder-only on this host; do not claim end-to-end RTF or device latency.",
            "Byte-BPE uses UTF-8 replacement policy; malformed sequences remain counted in CER.",
            "Dev/Test are utterance-disjoint but not speaker-disjoint; underlying model training overlap is unknown.",
        ],
    )


def csv_data(result):
    fields = ["id", "beam_width", "strategy", "blank_threshold",
              "cer", "s", "d", "i", "n",
              "decoder_mean_ms", "decoder_median_ms", "decoder_p95_ms",
              "decoder_sum_utterance_medians_ms", "decoder_ms_per_ctc_frame",
              "triggered_blank_frame_rate", "actual_hard_skip_frame_rate",
              "forced_blank_frame_rate", "nonblank_arcs_expanded",
              "nonblank_arcs_avoided_vs_same_beam_unpruned",
              "decoder_time_ratio_vs_framewise_greedy",
              "decoder_time_ratio_vs_same_beam_unpruned",
              "decoder_speedup_vs_same_beam_unpruned",
              "cer_delta_vs_framewise_greedy", "utf8_replacement_chars"]
    table = []
    for c in result["configurations"]:
        table.append(dict(
            id=c["id"], beam_width=c["beam_width"], strategy=c["strategy"],
            blank_threshold=c["blank_threshold"] if c["blank_threshold"] is not None else "None",
            cer=c["corpus"]["cer"], s=c["corpus"]["substitutions"],
            d=c["corpus"]["deletions"], i=c["corpus"]["insertions"],
            n=c["corpus"]["reference_chars"],
            decoder_mean_ms=c["timing"]["decoder_mean_ms"],
            decoder_median_ms=c["timing"]["decoder_median_ms"],
            decoder_p95_ms=c["timing"]["decoder_p95_ms"],
            decoder_sum_utterance_medians_ms=c["timing"]["decoder_sum_of_utterance_medians_ms"],
            decoder_ms_per_ctc_frame=c["timing"]["decoder_ms_per_ctc_frame"],
            triggered_blank_frame_rate=c["triggered_blank_frame_rate"],
            actual_hard_skip_frame_rate=c["actual_hard_skip_frame_rate"],
            forced_blank_frame_rate=c["forced_blank_frame_rate"],
            nonblank_arcs_expanded=c["nonblank_arcs_expanded"],
            nonblank_arcs_avoided_vs_same_beam_unpruned=c["nonblank_arcs_avoided_vs_same_beam_unpruned"],
            decoder_time_ratio_vs_framewise_greedy=c["decoder_time_ratio_vs_framewise_greedy"],
            decoder_time_ratio_vs_same_beam_unpruned=c["decoder_time_ratio_vs_same_beam_unpruned"],
            decoder_speedup_vs_same_beam_unpruned=c["decoder_speedup_vs_same_beam_unpruned"],
            cer_delta_vs_framewise_greedy=c["cer_delta_vs_framewise_greedy"],
            utf8_replacement_chars=c["utf8_replacement_chars"],
        ))
    return fields, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, help="External CTC model dir, required for measuring")
    parser.add_argument("--part-width", type=int, choices=BEAMS)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dev-root", type=Path, default=Path("build/ctc_round1_dev"))
    parser.add_argument("--data-root", type=Path, default=Path("data/ctc_eval"))
    parser.add_argument("--inventory", type=Path, default=Path("results/ctc_logit_inventory.json"))
    parser.add_argument("--greedy-report", type=Path, default=Path("results/ctc_baseline_dev.json"))
    parser.add_argument("--partial-dir", type=Path, default=Path("build"))
    parser.add_argument("--json-output", type=Path, default=Path("results/ctc_dev_sweep_results.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("results/ctc_dev_sweep_results.csv"))
    args = parser.parse_args()
    if args.merge_only:
        parts = []
        for width in BEAMS:
            part_path = args.partial_dir / f"ctc_round2_width_{width}.json"
            parts.append(json.loads(part_path.read_text(encoding="utf-8")))
        result = merge_parts(parts)
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        columns, table = csv_data(result)
        with args.csv_output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(table)
        print(json.dumps(dict(status=result["status"], configurations=len(table),
                              json=str(args.json_output), csv=str(args.csv_output)), ensure_ascii=False))
        return
    if args.part_width not in BEAMS or args.model_dir is None:
        parser.error("When measuring, require --part-width and --model-dir")
    inputs, detokenize, previous, identity = load_dev(
        args.dev_root, args.data_root, args.model_dir / "tokens.txt",
        args.model_dir / "bbpe.model", args.inventory, args.greedy_report)
    part = measure_width(inputs, detokenize, previous, identity, args.part_width, args.repeats)
    args.partial_dir.mkdir(parents=True, exist_ok=True)
    output = args.partial_dir / f"ctc_round2_width_{args.part_width}.json"
    output.write_text(json.dumps(part, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=part["status"], width=args.part_width,
                          experiments=len(part["configs"]), partial=str(output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
