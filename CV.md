# CV.md — ASR Portfolio Positioning & Interview Evidence

Status:
- VoiceFront-End-Bench: **Verified baseline + diagnostic experiments**.
- Streaming ASR Pilot: **Exploratory benchmark (measured, limited scope)**.
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

# 2. Streaming ASR Pilot (Round 2 exploratory benchmark)

## Purpose
Demonstrate practical ASR integration, endpointing evaluation, and accuracy-latency trade-off analysis.

## Boundary
This benchmark uses a fixed sherpa-onnx streaming Transducer pipeline. It is separate from the C++ Fbank implementation; no claim is made that custom Fbank improves ASR performance.

## Evidence
- `scripts/asr_stream_bench.py`
- `scripts/asr_metrics.py`
- `results/round2_pilot.json`
- `docs/ROUND2_STATUS.md`

## Demonstrated
- Reproducible streaming ASR evaluation with CER/S-D-I measurement.
- Endpoint threshold analysis: aggressive trailing-silence cutoff can reduce response delay but increase deletion errors when speech continuation is truncated.
- Controlled replay experiments comparing endpoint settings under the same model and audio condition, exposing the latency-completeness trade-off.

## Limitations
- Exploratory pilot, not a vehicle-command benchmark.
- CER observations are based on a single source-matched pilot sample and are not a general ASR benchmark.
- Replay latency measurements are controlled experiments, not product microphone-to-response latency.
- No AEC/NS integration.
- No production accuracy or latency claim.

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
| Streaming ASR endpoint trade-off | `scripts/asr_stream_bench.py`, `results/round2_pilot.json`, `docs/ROUND2_STATUS.md` |
| CER/S-D-I evaluation | `scripts/asr_metrics.py` |

---

# Interview narrative

Answer every project question through:

1. What problem was optimized?
2. What metric changed?
3. What trade-off was accepted?
4. Why was a simpler solution chosen over a more complex one?

A measured negative result is also an engineering conclusion. The portfolio objective is evidence-driven ASR engineering, not AI keyword collection.
