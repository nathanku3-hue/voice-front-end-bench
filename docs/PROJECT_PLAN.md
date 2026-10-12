# VoiceFront-End-Bench — LOCKED PLAN v1.0

**Change controlled / updated 2026-10-12.** This document is the project contract. Round 1 closed after both frozen-scope implementation and stage-wise precision diagnosis (baseline `085a6da`, diagnostics `a3527f6`). Round 2 was explicitly authorized and is now **CLOSED for the controlled exploratory R2-A/R2-B scope**, based on the committed 30-utterance evaluation (`fe075ae`). Round 3 was explicitly authorized on 2026-10-12 to build a narrow TN demo and publish inspectable evidence. Closure does **not** establish production performance; all unsupported measurements remain hypotheses.

## Project goal
Demonstrate sound DSP mathematics, reproducible C++17/Linux engineering and familiarity with Kaldi-style speech front ends. Build a compact **stateful streaming log-Mel Fbank extractor** and quantify numerical correctness, boundary behavior and processing cost. Not an ASR model, AEC algorithm or production real-time system.

## Round 1 — CLOSED (baseline locked)

* C++17, CMake, minimal public `StreamingFbank` API accepting arbitrary lengths of 16 kHz mono float PCM.
* Fixed 25 ms (400 sample) frames, 10 ms (160 sample) shift, 512-point FFT via a maintained external numerical library, 80 log-Mel bins.
* Kaldi-style windowing: per-frame DC removal, per-frame pre-emphasis (0.97), Povey window, power spectrum, low cutoff 20 Hz, unwarped Mel bands, log with epsilon floor. Dither disabled and `snip_edges=true` for deterministic parity.
* Strict stream chunk-invariance at 1 / 128 / 160 / 240 / 300 / 1024 / irregular sample blocks, including edge-length audio. Clearly define end-of-stream behavior.
* Numerical comparison to **torchaudio.compliance.kaldi** in both float32 and float64; report MAE, RMSE, max error, and reference settings. This is **Kaldi-compatible reference validation**, not official Kaldi executable parity or bit-exactness. Never claim `1e-5` until measured.
* Linux build/test commands, minimal README, experimental evidence and known limitations. Tests must execute under Release (`assert` alone is insufficient).

### Round 1 acceptance gates
1. `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure` succeeds in recorded environment.
2. Different chunkings produce identical frame counts and timestamps, and finite feature outputs within explicitly documented numerical tolerance.
3. Parity script produces measured error tables on deterministic multi-tone, noisy, chirp and silence inputs. Any exception is documented rather than suppressed.
4. All stated implementation and validation claims are reproducible from commands provided; no fabricated accuracy/latency outcomes.

## Round 2 — CLOSED (exploratory): streaming ASR endpoint evaluation

**Goal:** quantify Chinese read-speech transcription correctness and exploratory endpoint trade-offs using **one fixed public streaming ASR model**, without modifying the model or C++ Fbank. The existing standalone C++ Fbank **is not connected** to sherpa-onnx's waveform API; never claim Fbank-to-ASR performance gains.

### Phase R2-A (first / required)
1. Pin a sherpa-onnx Python runtime and a compatible, downloadable **Chinese online Zipformer transducer model** (record model URL/identity and SHA-256 for reproducibility). The model is an application dependency and must not be committed.
2. Implement CLI that accepts a manifest of 16kHz mono WAVs with **real transcripts** and timestamps where available. Input is streamed in configurable chunks through the model's online waveform API; no offline recognizer substitution.
3. Output per-utterance hypothesis, S/D/I character edits, corpus CER, decode CPU timing and reproducible run metadata. Distinguish a **faster-than-real-time playback simulation** from pacing at wall clock, and distinguish wall-clock response delay from model algorithmic lookahead.
4. Include self-checks for CER implementation (insertions / deletions / substitutions, empty references, Unicode normalization) that run without model downloads. Do not manufacture a command dataset or success claims.

### Phase R2-B (after functional ASR baseline)
A controlled sweep of **documented and supported endpointing settings** (such as rule2 trailing silence) on the exact same recorded speech data. Measure final-result availability relative to labeled end of speech, false/early endpoints, and CER. Fix audio, model, machine, thread count and segmentation throughout; report accuracy-latency trade-off with actual data, not a presumed Pareto curve.

### Phase R2-C (optional, not prerequisite)
WebRTC NS on/off under matched noise/audio conditions, using a **verified runnable interface** and same model. NS quality or CER improvements are neither assumed nor guaranteed. Do not add AEC or neural retraining.

### Round 2 gates
- R2-A passes only when a streaming model has been loaded, at least one documented WAV decoded end to end on this machine, model versions recorded, and numerical CER tests pass. A runnable but unexecuted script is **in progress**, not complete.
- R2-B passes only with paired per-sample results under >=2 endpoint configurations and an honest latency definition. If suitable labeled audio is unavailable, report the blocker, do not invent results.
- R2-C may be deferred; report status separately.
- Changes remain narrow: data manifest, scripts, tests and results. No self-written ASR/AEC/TTS kernels, no speculative RTF/CER claims.

### Execution checkpoint 2026-10-11

- R2-A: **PASS as an integration smoke test** — sherpa-onnx 1.13.8, public Chinese 14M model (release archive SHA-256 recorded), incremental waveform decoding succeeded on local Ubuntu WSL2, and Python CER/input tests passed.
- R2-B: **EXPLORATORY PILOT ONLY** — 400/800/1200ms endpoint settings tested on one candidate source-matched Chinese speech sample. Full text, stream-time, paced replay times and provisional CER recorded in [results/round2_pilot.json](../results/round2_pilot.json). No independent speech-end annotation, no command corpus, and cross-package WAV identity has not been cryptographically verified. At this 2026-10-11 checkpoint, no generalizable Pareto/endpoint response claim was possible and R2-B was still open; see the closure checkpoint below for the later 30-utterance study.
- R2-C: **NOT STARTED** — optional NS ablation deferred until controlled labeled speech data exists.
- Full execution notes: [docs/ROUND2_STATUS.md](ROUND2_STATUS.md).
- This entry records the historical Round 2 pilot state; later work is separately documented below.

### Round 2 closure checkpoint — 2026-10-12

- **R2-A DONE**: sherpa-onnx 1.13.8 streaming integration, public 14M Chinese online Transducer, working incremental waveform replay, independent CER/input checks; model archive and components recorded with SHA-256.
- **R2-B DONE (exploratory)**: 30 AISHELL-1 test utterances, 11 speakers, 403 independently matched reference characters, Rule2 {300, 500, 700, 900, 1200} ms, held model/chunks/threads/audio fixed. CER at endpoints 64.76%, 32.01%, 25.56%, 20.10%, 13.65%; deletions 244→7, and speaker-cluster paired CER difference −51.12 percentage points (5,000 resamples; 95% interval −59.10 to −42.55 percentage points). See [one-page summary](ROUND2_SUMMARY.md) and [full report](ROUND2_EVALUATION_REPORT.md).
- **Latency acceptance qualification**: `speech_end_s` uses a reverse-RMS **estimate**, 28/30 trailing boundaries unclear; delay and early-cutoff values are only proxies, and *not* verified realistic in-car response times. Consequently **R2-B closes only as the bounded exploratory study**, not as a product-latency validation.
- **R2-C DEFERRED**: optional matched-noise WebRTC NS on/off was not run; this is not a blocker for Round 3.
- **Frozen boundary**: standalone C++ Fbank remains independent of the sherpa-onnx internal waveform frontend.

## Round 3 — IMPLEMENTED LOCALLY (release pending): evidence-first portfolio

**Goal:** turn recorded work into a reviewable, reproducible engineering portfolio without inventing product improvements.

1. Deliver a self-contained `scripts/tts_tn_demo.py` with **15–20+** transparent high-frequency Mandarin vehicle messages and unit tests. Cover numerical cardinals, distances/speeds, clock times, temperature, battery percentages and deterministic **visual pause hints**. No audio synthesis, SSML timing or measured TTS quality.
2. Rewrite `README.md` with the **actual** endpoint trade-off figure and full 5-condition CER/S/D/I numbers above the fold, plus an honest explanation of conditional delay/estimated speech end. State the separate C++-Fbank and sherpa-onnx native pipelines explicitly.
3. Ship an evidence map and concise interview narrative, a pure-offline check for the committed bootstrap JSON, and reproducible optional full dataset/model commands that specify provenance and downloaded assets.
4. Prior DSP labs are *optional links only*: do not create fictitious or private links when those projects are not in this repository.
5. Gate release on passing unit tests, Git diff inspection and scoped claim audit. **Tag/push only by explicit authorization after validation**, not as an implicit consequence of writing documentation.

**Round 3 acceptance:** TN regression examples and Python tests pass; Markdown links to local artifacts resolve; no fabricated per-condition latency improvement, official-Kaldi parity, vehicle-domain CER, synthesized voice quality, or falsely claimed DSP-lab artifacts.

### Round 3 local validation checkpoint — 2026-10-12

- `scripts/tts_tn_demo.py`: 21 curated examples; CLI JSON and contextual number-reading tests passed.
- `python -m unittest discover -s tests -p 'test_*.py' -q`: **PASS (21/21)**, including docs-link and recorded CER consistency checks; NumPy is needed for legacy Round 2 tests.
- Ubuntu/WSL2 CMake Release build + `ctest --test-dir build --output-on-failure`: **PASS (1/1)**.
- `scripts/paired_endpoint_ci.py` on committed Round 2 JSON reproduced **−51.12 pp**, 95% speaker-cluster bootstrap interval **[−59.10, −42.55] pp**.
- `git diff --check`: no whitespace errors (Git warned about configured Windows CRLF conversion, not a diff validation failure).
- **Release still pending:** no Git commit, release tag or push made by this implementation task; only perform publication when explicitly requested.

## Explicitly excluded
Custom FFT kernel, custom ASR/TTS training, AEC3 integration, neural RES, Kaldi HCLG building, invented throughput improvements, assumed zero-heap audio callback safety, and blanket production/SOTA claims.

## Change control
Scope unfreezing recorded above as explicitly requested. Round 1 baseline remains frozen, Round 2 exploratory experiment is closed with R2-C deferred, and Round 3 is authorized for documentation and rule-based TN only. Additional feature families require a deliberate plan edit and measured justification. Do not add unrelated user files to a commit.
