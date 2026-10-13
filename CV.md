# CV.md — ASR Portfolio Positioning & Interview Evidence

Status:
- VoiceFront-End-Bench: **Verified baseline + diagnostic experiments**.
- Streaming ASR Endpoint Evaluation: **Round 2 CLOSED (30 utterances / 5 thresholds; exploratory only)**.
- Mandarin TTS TN demo: **Round 3 rule-based examples only; no audio/MOS results**.
- CTC-Decoding-Bench: **Rounds 1–3 evaluated**: 80 hashed CTC Logits, 30-utterance exploratory Dev sweep (36 Prefix Beam configurations), followed by a **single pre-registered 50-utterance held-out Test evaluation** (4 conditions); independent 200-hypothesis audit PASS. Manual audio-based Bad Case attribution remains open.

The portfolio goal is not to list ASR keywords. It demonstrates measurable engineering decisions: signal correctness, streaming behavior, numerical diagnosis, latency/accuracy trade-offs, and evidence-based scope control.

---

# 1. VoiceFront-End-Bench (locked project)

## Purpose
Demonstrate speech front-end DSP engineering and streaming system correctness.

## Resume-level evidence
- Implemented a C++17 stateful streaming log-Mel Fbank extractor with deterministic frame generation under arbitrary input chunk boundaries.
- Verified packet-boundary invariance through boundary-focused tests instead of assuming network/audio packet behavior.
- Compared Kaldi-compatible Fbank behavior against an external reference and diagnosed numerical limitations, including float precision sensitivity in low-energy spectral regions.
- Produced reproducible experiment reports covering feature correctness, edge cases, precision limitations, and implementation boundaries.

## Evidence
- `src/streaming_fbank.cpp`
- `tests/test_streaming.cpp`
- `docs/EXPERIMENT_REPORT.md`
- `results/kaldi_parity_wsl.json`
- `results/stage_chirp_f5_b78.json`

## Do not claim
- Custom production AEC/NS algorithms.
- Full ASR model training.
- Universal CER improvement from Fbank changes.
- Production real-time guarantees without device-level benchmark.

---

# 2. Streaming ASR Endpoint Evaluation (Round 2 exploratory benchmark)

## Purpose
Isolate an online recognizer **finalization** decision and quantify deletion-risk versus later endpoint commitment.

## Boundary
The sherpa-onnx 1.13.8 Chinese 14M Zipformer online Transducer model and its own native waveform frontend are fixed. The independent C++ Fbank is **not** connected, so no Fbank-driven CER benefit is claimed.

## Evidence
- `scripts/asr_stream_bench.py`, `scripts/asr_metrics.py`, `scripts/run_endpoint_sweep.py`, `scripts/verify_round2_manifest.py`
- `data/round2_endpoint_eval/manifest.jsonl` and `data/round2_endpoint_eval/provenance.json` (official transcript crosscheck, per-WAV SHA-256)
- `results/endpoint_sweep/rule2_{300,500,700,900,1200}ms.json`
- `results/endpoint_summary.csv`, `results/endpoint_tradeoff.png`
- `results/paired_endpoint_ci.json`, `scripts/paired_endpoint_ci.py`
- `docs/ROUND2_SUMMARY.md`, `docs/ROUND2_EVALUATION_REPORT.md`

## Demonstrated
- 30 AISHELL-1 test utterances / 11 speakers / 403 reference characters, five Rule2 thresholds on one fixed model and 160 ms waveform chunks.
- Corpus CER **64.76% → 13.65%**, deletions **244 → 7** for 300→1200 ms thresholds (within these data); speaker-cluster paired bootstrap **−51.12 percentage points**, approximate 95% interval **[−59.10, −42.55] percentage points**.
- Per-utterance S/D/I attribution and uncertainty accounting for 11 observed speaker clusters, rather than reporting a single unqualified accuracy score.

## Follow-up independent audits
- [Paired finalization stream-time](docs/ROUND3_VALIDATION_REPORT.md): 700→1200 ms Rule2 showed mean **+1179.2 ms**, median **+320 ms** across the same 30 utterances, independent of the uncertain speech-end estimate. Still not product latency.
- [Native Fbank contract](docs/ROUND3_VALIDATION_REPORT.md): C++ vs actual sherpa features MAE **0.9372** under mismatched default frame/Mel settings; after explicitly aligning sherpa's options to C++ the feature MAE fell to **4.58e-6** on 559 frames. **No** ONNX Encoder or decoder equivalence claimed.
- [P3 frontend sensitivity](docs/P3_DECISION_REPORT.md): held 30 WAVs and model constant over a **2×2 native sherpa** framing/cutoff ablation, with and without Rule2 endpointing. Matched-to-C++ **native settings**, not custom injected features, changed **9/30** transcripts and moved 1200ms CER **13.65% → 14.89%**.
- [P3 cost sensitivity](docs/P3_DECISION_REPORT.md): built a unit-consistent hypothetical waiting-vs-deletion penalty envelope; switch points **332.5/494.3/758.9 ms per deleted character** yielded **300→500→900→1200ms** winners. **700ms never minimized this specific linear objective**; no calibrated SLA or deployment choice.

## Limitations
- Measured early-cutoff and post-speech delay are based on an **automatic reverse-RMS speech-end proxy**; **28/30** clips have unclear silence tails. Do not claim human-annotated endpoint delay, production latency or an optimal 1200 ms vehicle threshold.
- Unpaced file replay, read Mandarin—not real in-cabin control speech, noise, barge-in or double-talk.
- Training overlap with evaluated AISHELL clips not independently established. R2-C WebRTC NS remains deferred.

---

# 2b. Mandarin TN mini-demo (Round 3)

- **Evidence:** `scripts/tts_tn_demo.py`, `tests/test_tts_tn_demo.py`, and top-level reproducible CLI.
- **Demonstrated:** 21 deterministic inputs, rule-based times/distances/speed/temperature/percentages, and visible pause hints with regression expectations.
- **Limit:** This is front-end text transformation, **not** a TTS generator, SSML duration predictor or acoustic quality/MOS optimization.

---

# 3. CTC-Decoding-Bench (Rounds 1–3; held-out Test measured once)

## Purpose
Demonstrate ASR decoder correctness, evaluation methodology, and optimization trade-offs.

## Verified Round 1 evidence
- Fixed an independent public 2025 Chinese **offline Zipformer-CTC** INT8 ONNX model (not the Project 1 Transducer). Inspected actual `[1,T,80]` input, `[1,T_out,1000]` CTC log-prob output and true output length. Model, indexed Byte-BPE vocabulary and model binary have SHA-256 identities.
- Froze **80 source-checked AISHELL-1 test-split utterances / 20 speakers** in a separate corpus: **30 exploratory Dev** and **50 held-out Test**, with no utterance IDs from the previous endpoint corpus. Held-out Test was not inspected for tuning and was evaluated once in Round 3. Splits are utterance-disjoint but *not speaker-disjoint*; original model training overlap is unverified.
- Exported and hashed **80 real `[T,V]` log-probability tensors**. Asset and normalized-score integrity audit passed.
- Measured **30-utterance Dev Greedy CER 2.94%** (**12 substitutions, 0 deletions, 1 insertion / 442 characters**) with an explicitly recorded Byte-BPE invalid-UTF-8 replacement policy (**two replacement characters**).
- Independently compared with native sherpa-onnx: **28/30 normalized hypotheses agree**, and native corpus CER is also **2.94%**, although native drops the two incomplete symbols and thus has **10 S / 2 D / 1 I**. This is a transcript check, **not model-feature or Logits parity**.

## Verified Round 2 — algorithm and Dev-only measurements
- Implemented an independent **log-domain Prefix Beam Search** using Pb/Pnb states, exact full-vocabulary beam candidate ranking (no hidden acoustic Top-K), deterministic tie resolution and two **distinct approximate** Blank policies.
- Numerically validated **all reachable prefix scores**, not only best output, against independent exhaustive Alignment Oracles (`T=3,V=4` and `T=4,V=3`, absolute tolerance `1e-6`); compared finite-beam vectorization against a literal recurrence.
- Executed **36 configurations** on the same 30 Dev utterances: Beam widths 1/4/8/16, thresholds None/0.90/0.95/0.98/0.99, Forced Blank and Hard Skip. All 36 yielded **2.94% corpus CER** (13 edits / 442 characters), **not** an improvement over separate Framewise Greedy.
- With **Beam 16**, Hard Skip at θ=0.90 bypassed **80.4%** of CTC frames and measured **3.68× lower decoder-only Python time** relative to **unpruned Beam 16**, *not relative to Framewise Greedy or end-to-end inference*. Median decoder time was **14.37 ms → 3.76 ms/utterance** on this host; group-median ratio is **not** the reported speedup statistic.
- Hard Skip changed **2/30 normalized transcripts** in every width/threshold combination, while Forced Blank matched the unpruned transcript on this small Dev. One Hard Skip change introduced `�` in `平衡`; another corrected an insertion. Thus equal corpus CER does **not** mean lossless pruning.
- Independent evidence audit recomputed S/D/I from all **36 × 30 Dev hypotheses** and checked data hashes, timing summaries, CSV/JSON equality and strict Test separation.

## Verified Round 3 — once-only held-out Test (50 utterances)
- Pre-registered exactly four decoders **before** Test scoring: Framewise Greedy; Beam 16 unpruned; Beam 16 Forced Blank θ=0.95; Beam 4 Forced Blank θ=0.95. Verified the frozen 50 tensor hashes, source/export manifests, and actual ONNX/token-table/BBPE file hashes before release.
- All four returned **Test CER 1.6461% (11 substitutions / 0 deletions / 1 insertion; 12 edits / 729 reference characters)** with **identical normalized transcripts and token sequences on all 50 utterances**; no UTF-8 replacement characters on this Test. This empirical zero drift does **not** make Forced Blank mathematically lossless.
- Mean of per-utterance median decoder-only timings in the local WSL2 Python environment: **Greedy 0.0264 ms**, **Beam 16 unpruned 15.5411 ms**, **Beam 16 Forced .95 5.1219 ms**, **Beam 4 Forced .95 3.2940 ms**. Against the same-width unpruned Beam 16, Forced .95 is **3.03× faster**, but still approximately **194× slower than Greedy**. Unpruned Beam 16 is approximately **589× slower than Greedy** (ratios of sums of per-utterance medians).
- **Greedy is the sole nondominated Test CER–decoder-time Pareto choice** among these four conditions. Independent re-scoring and provenance/timing/CSV checks passed for all **4 × 50 = 200** hypotheses. Three deterministic Bad Case candidates were documented with reference, hypotheses, frame confidence and audio path; **manual listening/causal acoustic diagnosis not yet performed**.

## Evidence files
- Round 1: `docs/CTC_ROUND1_STATUS.md`, `data/ctc_eval/provenance.json`, `results/ctc_round1_audit.json`
- Real assets and Greedy: `results/ctc_logit_inventory.json`, `results/ctc_baseline_dev.json`, `results/ctc_native_audit_dev.json`
- Round 2: `docs/CTC_ROUND2_STATUS.md`, `scripts/ctc_prefix_beam_search.py`, `tests/test_ctc_prefix_beam.py`, `scripts/run_ctc_dev_sweep.py`
- Measured grid and independent audit: `results/ctc_dev_sweep_results.json`, `results/ctc_dev_sweep_results.csv`, `results/ctc_round2_audit.json`
- Round 3 pre-registration, one-shot custody, 50-Test results, Pareto plot and bad-case triage: `docs/CTC_ROUND3_PROTOCOL.md`, `results/ctc_round3_unseal_claim.json`, `results/ctc_test_final_results.json`, `results/ctc_test_final_results.csv`, `results/ctc_test_pareto.png`, `docs/CTC_ROUND3_STATUS.md`; independently checked using `scripts/ctc_round3_audit.py`.

## Boundary / do not claim
- CTC is a **separate decoder study**, not the decoder in Project 1's online Transducer.
- No Beam CER improvement, acceleration over independent Framewise Greedy, lossless Blank pruning, speaker-independent generalization or production latency has been established.
- No custom C++ Fbank-to-CTC integration, retrained weights, or blanket model-weight redistribution permission claimed.

---

# Evidence Map

| Claim | Evidence |
|---|---|
| C++ streaming Fbank correctness | `src/streaming_fbank.cpp`, `tests/test_streaming.cpp`, `docs/EXPERIMENT_REPORT.md` |
| Numerical precision diagnosis | `results/stage_chirp_f5_b78.json`, `results/stage_multitone_f109_b79.json` |
| Streaming ASR model boundary | `README.md`, `scripts/asr_stream_bench.py` |
| Streaming ASR endpoint trade-off | `results/endpoint_summary.csv`, `results/endpoint_tradeoff.png`, `docs/ROUND2_EVALUATION_REPORT.md` |
| Speaker-aware paired uncertainty | `scripts/paired_endpoint_ci.py`, `results/paired_endpoint_ci.json` |
| Transcript provenance | `data/round2_endpoint_eval/manifest.jsonl`, `data/round2_endpoint_eval/provenance.json` |
| CER/S-D-I evaluation | `scripts/asr_metrics.py`, `results/endpoint_sweep/` |
| Rule-based Mandarin TN / pause hints | `scripts/tts_tn_demo.py`, `tests/test_tts_tn_demo.py` |
| Independent CTC input contract and 80 tensor hashes | `scripts/ctc_export_logits.py`, `results/ctc_logit_inventory.json`, `results/ctc_round1_audit.json` |
| Exploratory CTC Dev Greedy CER / native audit | `results/ctc_baseline_dev.json`, `results/ctc_native_audit_dev.json`, `docs/CTC_ROUND1_STATUS.md` |
| Oracle-verified CTC Prefix Beam and two Blank policies | `scripts/ctc_prefix_beam_search.py`, `tests/test_ctc_prefix_beam.py`, `docs/CTC_ROUND2_STATUS.md` |
| 36-configuration Dev decoder trade-off and evidence audit | `results/ctc_dev_sweep_results.json`, `results/ctc_dev_sweep_results.csv`, `results/ctc_round2_audit.json` |
| Pre-registered 50-Test CER, decoder-only Pareto, 200-row independent audit | `docs/CTC_ROUND3_PROTOCOL.md`, `results/ctc_test_final_results.json`, `results/ctc_test_pareto.png`, `docs/CTC_ROUND3_STATUS.md` |

---

# Interview narrative

Answer every project question through:

1. What problem was optimized?
2. What metric changed?
3. What trade-off was accepted?
4. Why was a simpler solution chosen over a more complex one?

A measured negative result is also an engineering conclusion. The portfolio objective is evidence-driven ASR engineering, not AI keyword collection.
