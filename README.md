# VoiceFront-End-Bench — Round 1

**Status:** Round 1 reproducible C++17 streaming feature-extractor prototype. **Locked plan:** [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md).

A small, reviewable experiment for a speech-algorithm interview: implement Kaldi-style log-Mel Fbank, verify arbitrary-chunk streaming invariance, and measure numerical differences against an explicit Kaldi-compatible reference. **This repo does not implement ASR, AEC, NS, TTS or an allocation-free real-time callback.**

## Implemented

- C++17 stateful `StreamingFbank`: mono `float32` input in arbitrary chunk sizes, frames emitted incrementally, 16 kHz / 400-sample frame / 160-sample hop.
- Per-frame DC removal and pre-emphasis, symmetric Povey window, zero-pad 400 -> 512, FFT via external FFTW3 backend, 80 triangular Mel-frequency bands from 20 Hz to Nyquist, power spectrum and log-energy with float epsilon floor.
- `snip_edges=true`: last incomplete frame intentionally not emitted; sample count / frame ordering deterministic regardless of input segmentation.
- CTest in Release, including non-`assert` checks for odd-sized and random chunk boundaries; Python parity report using `torchaudio.compliance.kaldi` as a **compatible**, not official-binary, reference.
- FFTW double-precision backend is deliberately distinguished from Kaldi's common float32 arithmetic: precision-dependent differences at near-null spectral bins are a **known limitation**, not a claim of bit-exactness.

## Build (Linux)

Prerequisites: CMake >= 3.16, C++17 compiler, and FFTW3 (`sudo apt-get install cmake g++ libfftw3-dev`). The original baseline was tested under isolated Debian Linux / GCC 14.2; **the code was subsequently built and tested on the host in Ubuntu 24.04 / WSL2, GCC 13.3.0, CMake 3.28.3, FFTW3 3.3.10**. On Windows this source folder is accessible from WSL as `/mnt/e/Code/voice`. A native Windows/MSVC build has **not** been validated.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Numerical reference additionally requires Python 3, NumPy, CPU PyTorch and torchaudio of compatible versions. The host WSL run used an isolated Python 3.12 venv under `~/.venvs/voice-bench` and `torch==2.10.0+cpu`, `torchaudio==2.10.0+cpu`, `numpy==2.3.5`:

```bash
python scripts/kaldi_compare.py --binary build/fbank_dump --output results/kaldi_parity_wsl.json
python scripts/stage_diagnostics.py --binary build/fbank_dump --signal chirp --frame 5 --band 78 \
  --output results/stage_chirp_f5_b78.json --keep-npy-dir build/chirp_trace
python scripts/stage_diagnostics.py --binary build/fbank_dump --signal multitone --frame 109 --band 79 \
  --output results/stage_multitone_f109_b79.json
```

Use `~/.venvs/voice-bench/bin/python` in place of `python` if the venv is not activated. The parity script tests synthetic multi-tone, noisy, chirp and silence cases; signals are **Kaldi-style PCM counts stored as float32**, not normalized [-1, 1] samples. The `--keep-npy-dir` option retains five debug arrays (post-preemphasis, windowed, FFT power, Mel energy, log-Fbank). No private datasets or ASR models are needed.

The earlier isolated-environment measurements are preserved in `results/kaldi_compatible_parity.json`; the **host WSL2** measurements are separately recorded in [`results/kaldi_parity_wsl.json`](results/kaldi_parity_wsl.json). Do not conflate these slightly different numerical runtimes.

## Streaming API

```cpp
#include "streaming_fbank.h"
voice::StreamingFbank fbank;
fbank.AcceptWaveform(samples, count);  // count may be any nonnegative size
while (fbank.NumFramesReady() > 0) {
  auto mel80 = fbank.GetFrame();        // consumes oldest frame
}
```

No explicit padding is performed at EOF. For 16,000 samples, `snip_edges=true` produces `1 + floor((16000 - 400) / 160) = 98` frames. Fbank frame index `t` corresponds to audio start index `160*t`. **It does not feed a Sherpa-ONNX model**; that would need feature API integration that is outside Round 1.

## What the tests prove (and do not prove)

| Check | Evidence | Interpretation |
|---|---|---|
| Chunk invariance | CTest `streaming_invariance` compares whole input to 1, 128, 160, 240, 300, 1024 and seeded irregular block sizes, including 399/400/560 sample edges | Verifies deterministic framing/output order under different packet boundaries; not a multi-thread real-time guarantee |
| Numerical reference | `scripts/kaldi_compare.py` reports MAE/RMSE/max error vs torchaudio in float32 and float64 | Verification against an external, Kaldi-compatible implementation, **not** official Kaldi binary or bitwise matching |
| Linux build | CMake/GCC/CTests and full command lines, including local WSL2 | Toolchain integration, not a native Windows or embedded platform certification |
| Opt-in stage tracing | CLI `--dump-stage` and `scripts/stage_diagnostics.py` | Isolates rounding differences by numerical stage; proves no product ASR improvement |

The precision audit's concrete numbers, outliers, WSL2 build verification, and stage-wise crossovers are in [`docs/EXPERIMENT_REPORT.md`](docs/EXPERIMENT_REPORT.md). No arbitrary `1e-5` target is claimed. The low-energy Float32 mismatch is **not caused by an active log floor** in the diagnosed chirp case; both Float32 preprocessing/window rounding and FFT arithmetic contribute.

## Scope and next steps

Frozen exclusions: custom FFT implementation, AEC3/NS, online CMVN, neural models, full Kaldi ASR pipeline, TTS, and SOTA/production-grade claims. A future Round 1 follow-up may compare with an **official Kaldi executable** configured with matching framing/window/dither/input scaling; that test **has not run** here. Round 2 (ASR CER/endpoint timing) is a separate gated workstream, not part of this deliverable.

## License/dependency notice

FFTW3 is a separately installed FFT backend with its own GPL license terms. This repository does **not** bundle FFTW3 code or binaries. Check its license suitability before distributing a linked application; a permissively licensed FFT alternative can be evaluated as a distinct future change.
