# Round 3 follow-up — Endpoint timing & feature-contract audit

**Measured:** 2026-10-12, WSL2 Ubuntu, existing Round 2 result JSON and public 14M Zipformer model. **Status:** local audit complete; not a production SLA study or an end-to-end ONNX compatibility result.

## Why these two tests?

A skeptical reviewer can attack two specific gaps: (1) endpoint-time claims used unverified reverse-RMS speech-end estimates, and (2) custom C++ Fbank had not been compared to the **actual** feature frames consumed by sherpa-onnx. These tests address both without changing the recognizer's trained weights or replaying the 30-utterance sweep.

## P0. Paired endpoint stream-time (no speech-end annotation required)

**Method:** From each of the five committed `results/endpoint_sweep/rule2_*.json` files, match every trial by ordered ID, WAV SHA-256, reference text, speaker and model hash. Require identical replay settings except the Rule2 threshold. Compute for each utterance:

```text
paired_delta_ms(i, a, b) =
    1000 * (endpoint_audio_time_s(i, b) - endpoint_audio_time_s(i, a))
```

The same unknown `speech_end_s(i)` cancels from the difference **algebraically**; no reverse-RMS label is used. Report paired mean, median, P10–P90, count of positive/zero/negative differences and an approximate **speaker-cluster bootstrap 95% interval of the mean** (11 speakers, 5,000 resamples, seed 20261011). The intervals use replacement sampling at the speaker level, preserving each speaker's utterances.

| Rule2 low→high | Mean extra stream-time | Median | P10–P90 | 95% cluster-bootstrap CI of mean | Fewer D edits |
|---|---:|---:|---:|---:|---:|
| 300→700 ms | **2210.5 ms** | 2212 ms | 640–3512.5 ms | [1862.9, 2621.4] ms | 178 |
| 700→900 ms | **445.7 ms** | 0 ms | 0–570.1 ms | [33.1, 1074.3] ms | 30 |
| 700→1200 ms | **1179.2 ms** | 320 ms | 320–3300.4 ms | [397.2, 2020.3] ms | 59 |
| 300→1200 ms | **3389.7 ms** | 3372 ms | 979–5427.4 ms | [2702.0, 4163.2] ms | 237 |

**Finding:** A 500 ms increase to the *configuration threshold* (700→1200) did **not** produce a uniform 500 ms increase in observed finalization stream-time. Its **median paired difference was 320 ms**, and **mean 1179.2 ms**, driven by heterogeneous endpoint behavior. 700→900 has a **zero median but positive mean**, also highlighting nonuniform outcomes.

**What this does NOT prove:** These are **waveform stream-time** deltas in unpaced replay, sometimes extending into synthetic tail silence; they are **not** human-annotated post-speech delays, real device response latency, product SLA exceedance rates, or proof that 700 ms is a Pareto knee. Chunk submission is 160 ms except for partial file-ending chunks, and no additional human speech-end labels were introduced. 28/30 original tails are ambiguous by the existing reverse-RMS quality flag. The data are 30 read-Mandarin utterances, not in-car commands.

**To reproduce P0 without a model, WAVs or new dependencies:**

```bash
python scripts/endpoint_timing_audit.py --reps 5000 --seed 20261011 --output build/endpoint_timing_audit.json
```

## P1. C++ Fbank against ACTUAL sherpa-onnx online features

**Method:** Use the downloaded `sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23` public model's WAV `test_wavs/0.wav`, SHA-256 `668bf8df51a10027b84d5d8816a1ce11ae93545538dc05cfe2aa6811d399c250`. Convert PCM16 to **the same normalized float32 waveform** (`sample / 32768`) for both paths. Compute custom C++ features via `build/fbank_dump`, and retrieve genuine sherpa features through `OnlineRecognizer.create_stream()`, `OnlineStream.accept_waveform()`, `input_finished()`, and the **native debugging API** `get_frames(0, n)`.

Compare the **first 559 common frames × 80 Mel bands** as per-index diagnostic arrays under two distinct configurations:

| Config | C++ custom | Native Round 2 | Native settings matched to C++ (diagnostic only) |
|---|---|---|---|
| Input | float32 normalized | Same | Same |
| `snip_edges` | **true** | **false** | **true** |
| Mel upper cutoff | **8000 Hz** | **7600 Hz** (`high_freq=-400`) | **8000 Hz** (`high_freq=0`) |
| Mel bins | 80 | 80 | 80 |
| Numerical path | FFTW double | sherpa native | sherpa native |

| Comparison | Mean absolute feature error | P95 absolute error | Max absolute error | Flattened **feature** cosine |
|---|---:|---:|---:|---:|
| C++ vs *actual Round 2* native setup | **0.937197** | 2.486956 | 8.834443 | 0.988513 |
| C++ vs *matched settings* native setup | **0.000004580** | 0.000013245 | 0.00021941 | ~1.000000 |

**Finding:** The C++ frontend **as implemented is not a drop-in equivalent of Round 2's native frontend**: frame placement (`snip_edges`) and Mel upper cutoff differ. But when the actual sherpa native extractor is given the corresponding C++ settings, its feature-level differences become very small on this real WAV. This is a controlled compatibility diagnosis, not a new optimization or proof that any particular recognizer's decoding will remain the same.

**Crucial distinctions:**
- The first MAE is **not** an isolated numerical-precision error: per-index frames do not share identical frame centers under different `snip_edges`, and their Mel filterbanks differ.
- The second comparison changes **sherpa's feature settings only as a diagnostic**; it does not change Round 2 results or C++ code.
- The measured cosine is over **log-Mel feature vectors**, **not** Encoder activations or Joiner logits. No input injection into `encoder.onnx`, state/cache transition equivalence, token-level parity or end-to-end CER match was tested. High cosine alone would not establish those claims.

**To reproduce P1 locally in the recorded WSL2 Python environment:** build the existing C++ executable, ensure the public model archive from [Round 2 setup](ROUND2_STATUS.md) has been downloaded, then from repo root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
MODEL="$HOME/.cache/voice-front-end/models/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23"
~/.venvs/voice-bench/bin/python scripts/audit_fbank_contract.py \
  --model-dir "$MODEL" --wav "$MODEL/test_wavs/0.wav" \
  --binary build/fbank_dump --output build/fbank_contract_audit.json
```

The script uses `sherpa-onnx==1.13.8` and NumPy. It intentionally **does not use onnxruntime**, because this step compares actual native features rather than running the Encoder. The original full system still needs a separate adapter if model inference parity is required.

## P3 extension: decision cost envelope and decoder sensitivity

The subsequent [P3 decision experiments](P3_DECISION_REPORT.md) add a hypothetical deletion-vs-stream-time cost envelope (where 700ms is never selected for the modelled linear penalty), plus **eight real ASR runs** varying sherpa's own snip/framing and Mel cutoff independently under endpoint-disabled and Rule2=1200 modes. Matching sherpa's settings to the C++ options resulted in **14.89% CER vs 13.65% native** at Rule2=1200, with 9/30 transcripts changed. This is **sensitivity to sherpa configuration**, not C++ feature injection.

## Engineering decision (not marketing)

- **Round 2 remains valid as an isolated endpoint policy ablation** because its own native frontend was held fixed throughout.
- **Round 1 validates a fixed C++ Kaldi-style Fbank configuration**, not strict drop-in compatibility with a *different* online recognizer's frame placement and high-frequency cutoff.
- **No production operating point selected**: define false-cutoff cost, latency SLA and human end-of-speech references before choosing a threshold. The current paired stream-time results allow a more honest discussion of responsiveness costs.
- **Future optional Gate 2:** a stateful Encoder cache/input contract check; **Gate 3:** Decoder+Joiner token and CER comparison. Neither is claimed completed.

[Original R1 numeric crossover](EXPERIMENT_REPORT.md) · [Round 2 full endpoint report](ROUND2_EVALUATION_REPORT.md) · [One-page Round 2 summary](ROUND2_SUMMARY.md)
