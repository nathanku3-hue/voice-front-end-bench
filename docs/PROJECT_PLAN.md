# VoiceFront-End-Bench — LOCKED PLAN v1.0

**Change controlled / updated 2026-10-11.** This document is the project contract. Round 1 closed after both frozen-scope implementation and stage-wise precision diagnosis, with baseline commit `085a6da` and diagnostics commit `a3527f6`. Round 2 is now **ACTIVE** by explicit authorization. All proposed measurements remain hypotheses until observed.

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

## Round 2 — ACTIVE: streaming ASR product benchmark

**Goal:** quantify Chinese command transcription correctness and realistic endpoint trade-offs using **one fixed public streaming ASR model**, without modifying the model or C++ Fbank. The existing standalone C++ Fbank **is not connected** to sherpa-onnx's waveform API; never claim Fbank-to-ASR performance gains.

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
- R2-B: **EXPLORATORY PILOT ONLY** — 400/800/1200ms endpoint settings tested on one candidate source-matched Chinese speech sample. Full text, stream-time, paced replay times and provisional CER recorded in [results/round2_pilot.json](../results/round2_pilot.json). No independent speech-end annotation, no command corpus, and cross-package WAV identity has not been cryptographically verified. Therefore no generalizable Pareto/endpoint response claim and R2-B remains open.
- R2-C: **NOT STARTED** — optional NS ablation deferred until controlled labeled speech data exists.
- Full execution notes: [docs/ROUND2_STATUS.md](ROUND2_STATUS.md).
- Unrelated workspace files (e.g. untracked CV.md) must not be edited, staged, or committed as part of this project.

## Round 3 — *not* authorized for implementation in Round 1
One small TTS text-normalization demonstration, documentation, interview explanation, optional links to prior DSP labs.

## Explicitly excluded
Custom FFT kernel, custom ASR/TTS training, AEC3 integration, neural RES, Kaldi HCLG building, invented throughput improvements, assumed zero-heap audio callback safety, and blanket production/SOTA claims.

## Change control
Scope unfreezing recorded above as explicitly requested. Round 1 is closed and its original code remains the baseline. Round 2 is authorized only to the extent described here; Round 3 remains frozen. New feature families require a deliberate plan edit and measured justification. Do not add unrelated user files to a commit.
