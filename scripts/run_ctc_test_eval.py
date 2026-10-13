#!/usr/bin/env python3
"""Round 3: frozen four-way, ONE-SHOT 50-utterance held-out Test CTC evaluation.

--preflight verifies all identities and archived Test tensor hashes but emits NO
Test hypotheses, CER, timing or result artifacts. --execute consumes the Test
set ONCE with an exclusive claim file, guarding against accidental retuning.

Run from the repository root in the existing WSL2 voice-bench environment.
Dependencies are existing scripts/asr_metrics.py, ctc_greedy_baseline.py,
ctc_prefix_beam_search.py, ctc_tokenization.py, run_ctc_dev_sweep.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from asr_metrics import corpus, evaluate, normalize
from ctc_greedy_baseline import greedy_ids, validate_log_probs
from ctc_prefix_beam_search import prefix_beam_search
from ctc_tokenization import build_detokenizer, load_tokens
from run_ctc_dev_sweep import file_sha, rows_jsonl, summarize_timing, config_id

# PRE-REGISTERED, IMMUTABLE FOUR-WAY COMPARISON. Do not add conditions after Test.
CONDITIONS = (
    ("framewise_greedy", None, "independent", None),
    ("prefix_w16_unpruned", 16, "none", None),
    ("prefix_w16_forced_blank_t0.95", 16, "forced_blank", 0.95),
    ("prefix_w04_forced_blank_t0.95", 4, "forced_blank", 0.95),
)
EXPECTED_DEV_STATUS = "ROUND2_DEV_GRID_COMPLETE_TEST_SEALED"
TEST_STATUS = "ROUND3_HELD_OUT_TEST_ONCE_COMPLETE"
POLICY = "replace"
REPEATS = 3
CSV_COLUMNS = (
    "id", "split", "beam_width", "strategy", "blank_threshold", "cer",
    "s", "d", "i", "n", "decoder_mean_ms", "decoder_median_ms",
    "decoder_p95_ms", "decoder_sum_of_utterance_medians_ms",
    "utf8_replacement_chars", "triggered_blank_frame_rate",
    "forced_blank_frame_rate", "actual_hard_skip_frame_rate",
    "hypotheses_different_from_greedy", "tokens_different_from_greedy",
)


def verify_condition_contract():
    frozen = (
        ("framewise_greedy", None, "independent", None),
        (config_id(16, "none", None), 16, "none", None),
        (config_id(16, "forced_blank", 0.95), 16, "forced_blank", 0.95),
        (config_id(4, "forced_blank", 0.95), 4, "forced_blank", 0.95),
    )
    if CONDITIONS != frozen:
        raise ValueError("Round 3 condition matrix changed from preregistration")


def unique_by_id(rows, *, expected_count, label):
    if len(rows) != expected_count:
        raise ValueError(f"{label}: expected {expected_count} rows, got {len(rows)}")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label}: duplicate IDs")
    return {r["id"]: r for r in rows}


def prepare_inputs(*, test_root, data_root, inventory_path, dev_report_path,
                   tokens_path, bpe_path, load_arrays=True):
    """Fail closed before any Test hypothesis or metric calculation.

    When load_arrays=False, still SHA-verify every `.npy` file but do not parse
    numeric arrays or call any CTC decoder. This is a preflight-only operation.
    """
    verify_condition_contract()
    test_root, data_root = Path(test_root).resolve(), Path(data_root).resolve()
    inventory_path = Path(inventory_path).resolve()
    dev_report_path = Path(dev_report_path).resolve()
    tokens_path, bpe_path = Path(tokens_path).resolve(), Path(bpe_path).resolve()
    test_source = data_root / "test.jsonl"
    dev_source = data_root / "dev.jsonl"
    test = rows_jsonl(test_source)
    dev = rows_jsonl(dev_source)
    test_by_id = unique_by_id(test, expected_count=50, label="Test source")
    dev_by_id = unique_by_id(dev, expected_count=30, label="Dev source")
    if set(test_by_id) & set(dev_by_id):
        raise ValueError("Dev/Test overlap")
    if any(r.get("split") != "test" for r in test) or any(r.get("split") != "dev" for r in dev):
        raise ValueError("Frozen source split tags changed")

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("status") != "frozen_real_ctc_log_probabilities":
        raise ValueError("Unexpected frozen logit inventory")
    dev_report = json.loads(dev_report_path.read_text(encoding="utf-8"))
    if dev_report.get("status") != EXPECTED_DEV_STATUS:
        raise ValueError("Round 2 Dev report not closed")
    prov = dev_report["provenance"]
    data_prov = json.loads((data_root / "provenance.json").read_text(encoding="utf-8"))
    if data_prov.get("dev_count") != 30 or data_prov.get("test_count") != 50:
        raise ValueError("Frozen data provenance count mismatch")
    for name, path in (("dev", dev_source), ("test", test_source)):
        digest = file_sha(path)
        if (inventory["splits"][name]["source_manifest_sha256"] != digest or
                data_prov[f"{name}_manifest_sha256"] != digest or
                prov[f"{name}_source_manifest_sha256"] != digest):
            raise ValueError(f"Frozen {name} manifest SHA mismatch")
    if file_sha(inventory_path) != prov["frozen_inventory_sha256"]:
        raise ValueError("Frozen inventory SHA no longer equals Round 2 contract")
    if file_sha(tokens_path) != inventory["token_table_sha256"] or \
            file_sha(bpe_path) != inventory["bbpe_model_sha256"]:
        raise ValueError("BPE/tokenizer SHA mismatch")
    if any((prov["model_sha256"] != inventory["model_sha256"],
            prov["token_table_sha256"] != inventory["token_table_sha256"],
            prov["bpe_model_sha256"] != inventory["bbpe_model_sha256"],
            prov["blank_id"] != 0, prov["tokenizer"] != "bbpe",
            prov["utf8_byte_errors"] != POLICY)):
        raise ValueError("Round 2 model/tokenization/policy drift")

    model = json.loads((test_root / "model.json").read_text(encoding="utf-8"))
    for key, expected in (("model_sha256", inventory["model_sha256"]),
                          ("token_table_sha256", inventory["token_table_sha256"]),
                          ("bbpe_model_sha256", inventory["bbpe_model_sha256"]),
                          ("blank_id", 0), ("tokenizer", "bbpe"),
                          ("ctc_head", "log_probs")):
        if model.get(key) != expected:
            raise ValueError(f"Test export model {key} mismatch")
    tokens = load_tokens(tokens_path)
    if len(tokens) != 1000 or tokens[0] != "<blk>":
        raise ValueError("Wrong frozen CTC vocabulary or blank index")

    export = rows_jsonl(test_root / "manifest.jsonl")
    export_by_id = unique_by_id(export, expected_count=50, label="Test export")
    entries = unique_by_id(inventory["splits"]["test"]["items"],
                           expected_count=50, label="Test inventory")
    if set(export_by_id) != set(test_by_id) or set(entries) != set(test_by_id):
        raise ValueError("Export/inventory Test IDs do not match frozen source")
    inputs = []
    for meta in export:  # stable frozen order
        uid = meta["id"]
        source, inv = test_by_id[uid], entries[uid]
        for key in ("reference", "wav_sha256", "speaker_id"):
            if meta.get(key) != source.get(key):
                raise ValueError(f"Test {key} mismatch: {uid}")
        if meta.get("source_split") != "AISHELL-1/test":
            raise ValueError(f"Test source split metadata drift: {uid}")
        if (source["wav_sha256"] != inv["wav_sha256"] or
                inv["vocab_size"] != len(tokens)):
            raise ValueError(f"Test frozen inventory metadata mismatch: {uid}")
        tensor = (test_root / meta["log_probs"]).resolve()
        if not tensor.is_relative_to(test_root) or tensor.suffix != ".npy":
            raise ValueError(f"Unsafe Test tensor path: {uid}")
        if file_sha(tensor) != inv["log_probs_sha256"]:
            raise ValueError(f"Test tensor SHA-256 drift: {uid}")
        scores = None
        if load_arrays:
            scores = validate_log_probs(np.load(tensor, allow_pickle=False), len(tokens))
            if scores.shape[0] != inv["ctc_frames"]:
                raise ValueError(f"Frozen Test frame count mismatch: {uid}")
        inputs.append(dict(
            id=uid, reference=source["reference"],
            speaker_id=source["speaker_id"], audio_relpath=source["wav"],
            wav_sha256=source["wav_sha256"],
            duration_s=source["duration_s"], logits_sha256=inv["log_probs_sha256"],
            frames=int(inv["ctc_frames"]), log_probs=scores,
        ))

    identity = dict(
        dataset="AISHELL-1/test, held-out from CTC Round 2 Dev selection",
        utterances=50, model_sha256=inventory["model_sha256"],
        source_manifest_sha256=file_sha(test_source),
        export_manifest_sha256=file_sha(test_root / "manifest.jsonl"),
        inventory_sha256=file_sha(inventory_path),
        round2_dev_report_sha256=file_sha(dev_report_path),
        token_table_sha256=file_sha(tokens_path),
        bbpe_model_sha256=file_sha(bpe_path),
        blank_id=0, tokenizer="bbpe", byte_errors=POLICY,
    )
    return inputs, tokens, dev_report, identity


def timed_decode(scores, *, width, strategy, threshold):
    if width is None:
        def decode():
            return tuple(greedy_ids(scores, 0))
        def tokens(out):
            return out
    else:
        def decode():
            return prefix_beam_search(scores, beam_width=width, blank_id=0,
                                      strategy=strategy, blank_threshold=threshold)
        def tokens(out):
            return out.tokens
    first = decode()  # untimed warmup
    initial = tuple(tokens(first))
    durations = []
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        out = decode()
        ms = (time.perf_counter_ns() - start) / 1e6
        if tuple(tokens(out)) != initial:
            raise AssertionError("Non-deterministic held-out decoder hypothesis")
        durations.append(ms)
    return initial, first, statistics.median(durations), durations


def measure_one(inputs, detokenize, condition):
    cid, width, strategy, threshold = condition
    utterances, edits = [], []
    for item in inputs:
        scores = item["log_probs"]
        token_ids, diagnostic, median_ms, reps = timed_decode(
            scores, width=width, strategy=strategy, threshold=threshold)
        hypothesis = detokenize(token_ids)
        counts = evaluate(item["reference"], hypothesis)
        edits.append(counts)
        blank_p = np.exp(np.asarray(scores[:, 0], dtype=np.float64))
        max_logp = np.max(scores, axis=1)
        row = dict(
            id=item["id"], speaker_id=item["speaker_id"],
            reference=item["reference"], hypothesis=hypothesis,
            tokens=list(token_ids), audio_relpath=item["audio_relpath"],
            wav_sha256=item["wav_sha256"],
            logits_sha256=item["logits_sha256"], duration_s=item["duration_s"],
            frames=item["frames"], decoder_median_ms=median_ms,
            decoder_repeats_ms=reps,
            s=counts.s, d=counts.d, i=counts.i, n=counts.n, cer=counts.cer,
            utf8_replacement_chars=hypothesis.count("\ufffd"),
            # Posterior evidence for Bad Case triage; never assert causal diagnosis.
            avg_frame_max_p=float(np.mean(np.exp(max_logp))),
            avg_blank_p=float(np.mean(blank_p)),
            fraction_blank_p_ge_095=float(np.mean(blank_p >= 0.95)),
            triggered_blank_frames=0 if width is None else diagnostic.triggered_blank_frames,
            forced_blank_frames=0 if width is None else diagnostic.forced_blank_frames,
            hard_skipped_frames=0 if width is None else diagnostic.hard_skipped_frames,
            expanded_nonblank_arcs=0 if width is None else diagnostic.expanded_nonblank_arcs,
        )
        utterances.append(row)
    total_frames = sum(r["frames"] for r in utterances)
    return dict(
        id=cid, decoder=("Framewise argmax -> CTC collapse" if width is None else
                         "Full-vocabulary log-domain CTC Prefix Beam Search"),
        beam_width=width, strategy=strategy, blank_threshold=threshold,
        corpus=corpus(edits), timing=summarize_timing(utterances),
        total_ctc_frames=total_frames,
        utf8_replacement_chars=sum(r["utf8_replacement_chars"] for r in utterances),
        triggered_blank_frames=sum(r["triggered_blank_frames"] for r in utterances),
        triggered_blank_frame_rate=sum(r["triggered_blank_frames"] for r in utterances)/total_frames,
        forced_blank_frames=sum(r["forced_blank_frames"] for r in utterances),
        forced_blank_frame_rate=sum(r["forced_blank_frames"] for r in utterances)/total_frames,
        hard_skipped_frames=sum(r["hard_skipped_frames"] for r in utterances),
        actual_hard_skip_frame_rate=sum(r["hard_skipped_frames"] for r in utterances)/total_frames,
        utterances=utterances,
    )


def compare_hypotheses(configs):
    baseline = {u["id"]: u for u in configs[0]["utterances"]}
    for conf in configs:
        conf["hypotheses_different_from_greedy"] = sum(
            normalize(u["hypothesis"]) != normalize(baseline[u["id"]]["hypothesis"])
            for u in conf["utterances"]
        )
        conf["tokens_different_from_greedy"] = sum(
            u["tokens"] != baseline[u["id"]]["tokens"]
            for u in conf["utterances"]
        )
    unpruned = {u["id"]: u for u in configs[1]["utterances"]}
    for conf in configs:
        conf["hypotheses_different_from_beam16_unpruned"] = sum(
            normalize(u["hypothesis"]) != normalize(unpruned[u["id"]]["hypothesis"])
            for u in conf["utterances"]
        )


def pareto_ids(configs):
    """Exact nondominated indices in mean decoder-ms vs corpus CER space."""
    frontier = []
    for i, a in enumerate(configs):
        x, y = a["timing"]["decoder_mean_ms"], a["corpus"]["cer"]
        if not any((b["timing"]["decoder_mean_ms"] <= x and b["corpus"]["cer"] <= y
                    and (b["timing"]["decoder_mean_ms"] < x or b["corpus"]["cer"] < y))
                   for j, b in enumerate(configs) if i != j):
            frontier.append(a["id"])
    return frontier


def select_badcases(configs, limit=3):
    """Fixed rule from protocol: highest Greedy (S+D+I), then ID."""
    greedy = configs[0]["utterances"]
    chosen = sorted(greedy, key=lambda r: (-(r["s"]+r["d"]+r["i"]), r["id"]))
    if not any(r["s"]+r["d"]+r["i"] for r in chosen):
        return []
    selected = []
    for case in chosen:
        if len(selected) >= limit or case["s"]+case["d"]+case["i"] == 0:
            break
        selected.append(dict(
            id=case["id"], reference=case["reference"],
            speaker_id=case["speaker_id"], audio_relpath=case["audio_relpath"],
            duration_s=case["duration_s"],
            frames=case["frames"],
            avg_frame_max_p=case["avg_frame_max_p"],
            avg_blank_p=case["avg_blank_p"],
            fraction_blank_p_ge_095=case["fraction_blank_p_ge_095"],
            note="Triage evidence only; homophones/deletions require manual audio review",
            hypotheses={conf["id"]: next(dict(
                hypothesis=u["hypothesis"], tokens=u["tokens"],
                s=u["s"], d=u["d"], i=u["i"],
                utf8_replacement_chars=u["utf8_replacement_chars"])
                for u in conf["utterances"] if u["id"] == case["id"])
                for conf in configs},
        ))
    return selected


def project_csv(result):
    rows = []
    for conf in result["configurations"]:
        c, t = conf["corpus"], conf["timing"]
        rows.append(dict(id=conf["id"], split="held_out_test", beam_width=conf["beam_width"],
                         strategy=conf["strategy"], blank_threshold=conf["blank_threshold"],
                         cer=c["cer"], s=c["substitutions"], d=c["deletions"],
                         i=c["insertions"], n=c["reference_chars"],
                         decoder_mean_ms=t["decoder_mean_ms"],
                         decoder_median_ms=t["decoder_median_ms"],
                         decoder_p95_ms=t["decoder_p95_ms"],
                         decoder_sum_of_utterance_medians_ms=t["decoder_sum_of_utterance_medians_ms"],
                         utf8_replacement_chars=conf["utf8_replacement_chars"],
                         triggered_blank_frame_rate=conf["triggered_blank_frame_rate"],
                         forced_blank_frame_rate=conf["forced_blank_frame_rate"],
                         actual_hard_skip_frame_rate=conf["actual_hard_skip_frame_rate"],
                         hypotheses_different_from_greedy=conf["hypotheses_different_from_greedy"],
                         tokens_different_from_greedy=conf["tokens_different_from_greedy"]))
    return rows


def check_output_available(claim_path, json_path, csv_path):
    for path in (claim_path, json_path, csv_path):
        if path.exists():
            raise FileExistsError(f"One-shot Test evaluation blocked: {path} already exists")


def exclusive_claim(path, identity, protocol_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    claimed = dict(status="CLAIMED_BEFORE_FIRST_TEST_DECODE",
                   utc_time=datetime.now(timezone.utc).isoformat(),
                   protocol_sha256=file_sha(protocol_path),
                   script_sha256=file_sha(Path(__file__)),
                   conditions=[c[0] for c in CONDITIONS],
                   frozen_input_contract=identity,
                   note="A failed or interrupted run must not be silently repeated")
    # Atomic O_EXCL create; a second invocation fails even with identical inputs.
    with path.open("x", encoding="utf-8") as f:
        json.dump(claimed, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    return claimed


def execute(args):
    # Fail if results/claim already exist. No scoring or decoding before lock.
    check_output_available(args.claim, args.json_output, args.csv_output)
    inputs, tokens, dev_report, identity = prepare_inputs(
        test_root=args.test_root, data_root=args.data_root,
        inventory_path=args.inventory, dev_report_path=args.dev_report,
        tokens_path=args.model_dir / "tokens.txt", bpe_path=args.model_dir / "bbpe.model",
        load_arrays=not args.preflight)
    if args.preflight:
        print(json.dumps(dict(status="PREFLIGHT_ONLY_NO_TEST_DECODE", count=len(inputs),
                              frozen_input_contract=identity,
                              conditions=[x[0] for x in CONDITIONS]), ensure_ascii=False))
        return
    if not args.execute:
        raise ValueError("Choose --preflight or --execute; there is no implicit unseal")
    if not args.protocol.is_file():
        raise FileNotFoundError(f"Pre-registration protocol missing: {args.protocol}")
    # Import/init detokenizer before claiming, but DO NOT decode any Test logits.
    detokenize = build_detokenizer(tokens, kind="bbpe",
                                   bpe_model=args.model_dir / "bbpe.model",
                                   byte_errors=POLICY)
    claim = exclusive_claim(args.claim, identity, args.protocol)
    configs = [measure_one(inputs, detokenize, cond) for cond in CONDITIONS]
    compare_hypotheses(configs)
    dev_by_id = {dev_report["framewise_greedy"]["id"]: dev_report["framewise_greedy"]}
    dev_by_id.update({c["id"]: c for c in dev_report["configurations"]})
    dev_comparison = [dict(id=cid, corpus=dev_by_id[cid]["corpus"],
                           timing=dev_by_id[cid]["timing"]) for cid, *_ in CONDITIONS]
    result = dict(
        status=TEST_STATUS,
        preregistration=dict(path=str(args.protocol), sha256=claim["protocol_sha256"],
                             locked_conditions=[c[0] for c in CONDITIONS]),
        frozen_input_contract=identity,
        timing_protocol=dict(repeats=REPEATS, warmups_per_utterance=1,
                             clock="time.perf_counter_ns", python=platform.python_version(),
                             numpy=np.__version__, platform=platform.platform(),
                             included="decoder-only; includes CTC argmax/collapse or beam function",
                             excluded="WAV/Fbank/ONNX/disk/Byte-BPE/scoring/report I/O"),
        dev_comparison=dev_comparison,
        configurations=configs,
        test_pareto_frontier_ids=pareto_ids(configs),
        badcase_triage=select_badcases(configs, limit=3),
        limitations=[
            "Held-out relative to Round 2 configuration selection, not speaker-disjoint",
            "Potential overlap with the public model's pretraining data is unknown",
            "Forced Blank is heuristic and not mathematically lossless",
            "Python decoder-only timings are local and not end-to-end speech latency",
            "Automated error triage is not a verified causal acoustic diagnosis",
        ],
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    if args.json_output.exists() or args.csv_output.exists():
        raise FileExistsError("Results appeared after one-shot claim; refusing overwrite")
    with args.json_output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with args.csv_output.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(project_csv(result))
    claim["status"] = "COMPLETED"
    claim["result_json_sha256"] = file_sha(args.json_output)
    claim["result_csv_sha256"] = file_sha(args.csv_output)
    args.claim.write_text(json.dumps(claim, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(status=TEST_STATUS,
                          frontier=result["test_pareto_frontier_ids"],
                          configurations=[dict(id=c["id"], cer=c["corpus"]["cer"],
                                               mean_decoder_ms=c["timing"]["decoder_mean_ms"])
                                          for c in configs],
                          json=str(args.json_output), csv=str(args.csv_output)),
                     ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="Hash-check Test inputs without decoding")
    mode.add_argument("--execute", action="store_true", help="One-shot held-out unseal and decode")
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Existing frozen external Zipformer CTC model directory")
    p.add_argument("--test-root", type=Path, default=Path("build/ctc_round1_test"))
    p.add_argument("--data-root", type=Path, default=Path("data/ctc_eval"))
    p.add_argument("--inventory", type=Path, default=Path("results/ctc_logit_inventory.json"))
    p.add_argument("--dev-report", type=Path, default=Path("results/ctc_dev_sweep_results.json"))
    p.add_argument("--protocol", type=Path, default=Path("docs/CTC_ROUND3_PROTOCOL.md"))
    p.add_argument("--claim", type=Path, default=Path("results/ctc_round3_unseal_claim.json"))
    p.add_argument("--json-output", type=Path, default=Path("results/ctc_test_final_results.json"))
    p.add_argument("--csv-output", type=Path, default=Path("results/ctc_test_final_results.csv"))
    args = p.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
