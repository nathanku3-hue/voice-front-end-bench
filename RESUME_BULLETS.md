# Resume Bullets — ASR Portfolio

## VoiceFront-End-Bench — C++ Streaming Speech Front-End

**Focus:** DSP correctness, streaming behavior, numerical validation.

- Implemented a C++17 stateful streaming log-Mel Fbank extractor with deterministic frame generation under arbitrary PCM chunk boundaries; verified packet-boundary invariance through Release-mode tests covering fixed and irregular segmentation patterns.

- Benchmarked Kaldi-compatible Fbank behavior against an external reference implementation and performed stage-wise numerical diagnosis of low-energy spectral discrepancies, isolating float-precision sensitivity in preprocessing/windowing and FFT computation rather than assuming a single error source.

- Built reproducible speech-processing experiments and documentation covering edge cases, numerical limitations, and validation boundaries; avoided unsupported claims of bit-exact Kaldi parity or production real-time guarantees.

---

## Streaming ASR Pilot — Online Transducer Evaluation

**Focus:** ASR integration, endpointing, accuracy-latency trade-offs.

- Integrated a fixed Chinese streaming Transducer ASR pipeline and built a reproducible evaluation harness for streaming waveform input, CER/S-D-I measurement, model metadata tracking, and controlled endpoint experiments.

- Evaluated trailing-silence endpoint configurations under controlled replay conditions, demonstrating the engineering trade-off between faster response commitment and increased deletion errors caused by premature speech truncation.

- Designed evaluation boundaries to separate model-native frontend behavior from custom C++ Fbank experiments, avoiding attribution of ASR changes to unconnected frontend components.

---

## CTC-Decoding-Bench — Planned

**Status:** Planned sibling study; no completed metrics claimed.

Target topics:

- Log-domain CTC Prefix Beam Search correctness.
- Blank/non-blank state transitions and numerical stability.
- Greedy vs beam decoding trade-offs.
- Decoder latency vs recognition accuracy evaluation.

No resume claim should be made until implementation, validation, and reproducible measurements are complete.

---

## Interview Themes

- Correctness before optimization.
- Measure latency and accuracy together instead of optimizing one metric blindly.
- Prefer a simpler verified solution over a more complex unmeasured approach.
- Treat negative experimental results as engineering conclusions when supported by controlled evaluation.
