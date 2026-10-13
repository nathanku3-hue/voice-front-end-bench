# CTC Round 3 — pre-registered held-out protocol

**Scope:** Frozen 50-utterance AISHELL-1 Test split; a single evaluation pass. These conditions are fixed using only the completed 30-utterance Dev results, before examining any Test hypotheses or CER. Dev/Test are utterance-disjoint but **not speaker-disjoint**. Overlap with the underlying model's training corpus has not been independently established.

## Locked conditions

| ID | Decoder | Width | Strategy | Threshold |
|---|---|---:|---|---:|
| `framewise_greedy` | Framewise argmax + CTC collapse | — | independent | — |
| `prefix_w16_unpruned` | Full-vocabulary Prefix Beam | 16 | none | — |
| `prefix_w16_forced_blank_t0.95` | Full-vocabulary Prefix Beam | 16 | forced_blank | 0.95 |
| `prefix_w04_forced_blank_t0.95` | Full-vocabulary Prefix Beam | 4 | forced_blank | 0.95 |

No additional candidate, threshold, beam width, tokenizer policy, reference normalization, or model adjustment may be selected using Test data. Report all four conditions, including neutral or negative outcomes. Do not tune and rerun on the same Test set.

## Data, timing, scoring and plotting contract

- Verify exactly 50 frozen Test IDs, 30 disjoint Dev IDs, source/export manifests, model/token/BPE SHA-256 identities, inventory SHA-256 of each `[T,1000]` Test tensor, and blank ID 0 before any decoder call.
- Reuse Byte-BPE `byte_errors='replace'` and report U+FFFD replacement characters. Apply existing `asr_metrics.py` NFKC/casefold/whitespace/punctuation normalization and corpus-weighted CER/S/D/I.
- Model and real log probabilities are fixed; no training, external LM, acoustic token Top-K, or new feature extraction. Forced Blank restricts CTC transitions; it is **not** mathematically lossless.
- Sequential CPU Python decoder-only `perf_counter_ns` timing, 1 untimed warmup + 3 timed calls for **each** utterance and configuration. First preload/validate all 50 tensors. Detokenization, scoring, I/O, Fbank and ONNX inference are **not** timed.
- Report mean, median, P95 and sum of utterance median decoder times. Primary Pareto x-axis = **mean of per-utterance medians** (ms/utterance); y-axis = **corpus CER percentage**. Lower is better for both. All points appear even if Greedy is sole nondominated choice. Plot Dev and Test distinctly, and compute a Test-only Pareto frontier.
- Diagnostic cases are deterministically selected by highest **Greedy absolute edit count**, ties resolved by utterance ID. Report top 2–3, with all four hypotheses, S/D/I, replacement counts, CTC frame count, source audio duration, and blank/posterior confidence summaries. Do not assign acoustic causes from logits alone; confirmation requires actual audio listening.

## One-shot protection and interpretation

After preflight validation, execution creates `results/ctc_round3_unseal_claim.json` exclusively; it refuses a second invocation if this marker or either final result exists. A crash after unsealing is a failed attempt, not permission to tune or silently retry. Resume/recovery decisions require a separately disclosed protocol amendment. A preflight mode validates hashes without decoding or scoring held-out Test examples.

Expected outputs: `results/ctc_test_final_results.json`, `.csv`, a CER–latency plot (`results/ctc_test_pareto.png`) and an evidence-qualified `docs/CTC_ROUND3_STATUS.md`. CV/resume edits are contingent on real observed Test outcomes.

**Limits:** No universal CER gain, lossless blank pruning, speaker-disjoint robustness, training-data independence, production RTF or voice-product response latency is claimed.
