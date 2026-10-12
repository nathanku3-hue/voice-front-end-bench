# CV.md — ASR Portfolio Positioning & Interview Evidence

Status:
- VoiceFront-End-Bench: **Verified baseline + diagnostic experiments**.
- Streaming ASR Endpoint Evaluation: **Round 2 CLOSED (30 utterances / 5 thresholds; exploratory only)**.
- Mandarin TTS TN demo: **Round 3 rule-based examples only; no audio/MOS results**.
- CTC-Decoding-Bench: **planned sibling project; results not yet claimed**.

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

# 3. CTC-Decoding-Bench (planned sibling project)

## Purpose
Demonstrate ASR decoder correctness, evaluation methodology, and optimization trade-offs.

## Planned evidence
- Fixed CTC acoustic model baseline with explicit input contract.
- Greedy decoding vs log-domain Prefix Beam Search.
- CER/S-D-I evaluation.
- Blank/non-blank transitions, stable `log_add`, duplicate-token collapse tests, optional exhaustive alignment oracle.
- Decoder latency vs end-to-end RTF analysis.

## Boundary
CTC-Decoding-Bench is a separate decoder study and is not the decoder used by the current Zipformer Transducer benchmark.

## Do not claim before measurement
- CER reduction.
- Decoder speedup.
- Lossless blank pruning.
- Seamless Fbank-to-model integration.

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

---

# Interview narrative

Answer every project question through:

1. What problem was optimized?
2. What metric changed?
3. What trade-off was accepted?
4. Why was a simpler solution chosen over a more complex one?

A measured negative result is also an engineering conclusion. The portfolio objective is evidence-driven ASR engineering, not AI keyword collection.
