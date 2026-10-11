# VoiceFront-End-Bench — LOCKED PLAN v1.0

**Frozen scope / 2026-10-11.** This document is the project contract. Round 1 must be validated before pursuing later rounds. Proposed metrics are not results.

## Project goal
Demonstrate sound DSP mathematics, reproducible C++17/Linux engineering and familiarity with Kaldi-style speech front ends. Build a compact **stateful streaming log-Mel Fbank extractor** and quantify numerical correctness, boundary behavior and processing cost. Not an ASR model, AEC algorithm or production real-time system.

## Round 1 — implementation only (current)

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

## Round 2 — *not* authorized for implementation in Round 1
Fixed-model streaming ASR with true waveform ingestion; CER (S/D/I), endpoint response latency, and optional WebRTC NS on/off ablation. Independent C++ Fbank comparison **does not imply it feeds the ASR model**.

## Round 3 — *not* authorized for implementation in Round 1
One small TTS text-normalization demonstration, documentation, interview explanation, optional links to prior DSP labs.

## Explicitly excluded
Custom FFT kernel, custom ASR/TTS training, AEC3 integration, neural RES, Kaldi HCLG building, invented throughput improvements, assumed zero-heap audio callback safety, and blanket production/SOTA claims.

## Change control
A scope change requires editing this contract first, marking exactly what changed and why. Current request locks Round 1; defer any features that do not improve one of its four acceptance gates.
