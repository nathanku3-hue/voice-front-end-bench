# Resume Bullets — ASR Portfolio

## VoiceFront-End-Bench — C++ Streaming Speech Front-End

**Focus:** DSP correctness, streaming behavior, numerical validation.

- Implemented a C++17 stateful streaming log-Mel Fbank extractor with deterministic frame generation under arbitrary PCM chunk boundaries; verified packet-boundary invariance through Release-mode tests covering fixed and irregular segmentation patterns.

- Benchmarked Kaldi-compatible Fbank behavior against an external reference implementation and performed stage-wise numerical diagnosis of low-energy spectral discrepancies, isolating float-precision sensitivity in preprocessing/windowing and FFT computation rather than assuming a single error source.

- Built reproducible speech-processing experiments and documentation covering edge cases, numerical limitations, and validation boundaries; avoided unsupported claims of bit-exact Kaldi parity or production real-time guarantees.

---

## Streaming ASR Endpoint Evaluation — Online Transducer (Round 2)

**Focus:** ASR integration, endpointing, accuracy-latency trade-offs.

- Integrated a fixed Chinese streaming Transducer ASR pipeline and built a reproducible evaluation harness for streaming waveform input, CER/S-D-I measurement, model metadata tracking, and controlled endpoint experiments.

- Benchmarked 30 AISHELL-1 test utterances (11 speakers, 403 independently matched reference characters) across five Rule2 trailing-silence thresholds on a fixed online Transducer: corpus CER declined from **64.76% (300 ms)** to **13.65% (1200 ms)** and deletion edits from **244 to 7**, at the cost of more conservative finalization; this is **not** a verified real-time product latency gain.

- Audited per-utterance S/D/I errors and computed a paired **11-speaker-cluster bootstrap** for the observed **−51.12 percentage-point CER difference** (95% interval **−59.10 to −42.55 points**), while documenting that reverse-RMS speech-end estimates (unclear for 28/30 clips) are **not** manually verified product latency.

- Kept sherpa-onnx's native waveform frontend separate from the C++ Fbank implementation; did not attribute the endpoint CER gains to unconnected DSP code.

- Audited paired finalization stream-time on 30 matched utterances (700→1200 ms Rule2: median **+320 ms**, mean **+1179 ms**) without using noisy end-of-speech labels; isolated a native-vs-C++ Fbank **configuration mismatch** and demonstrated close **feature-level** agreement after matching framing and Mel cutoff (MAE **4.58e-6**, 559 frames). No model-output parity or real microphone latency claimed.

- Executed a controlled **2×2 sherpa-native frontend** configuration study on 30 matched test WAVs, measuring CER under disabled and fixed-1200ms endpointing: changing framing/Mel cutoff to the C++ *settings* altered **9/30 transcripts** and increased the 1200ms CER **13.65% → 14.89%**; also built an explicit hypothetical deletion-vs-waiting cost envelope showing why a visually appealing **700ms knee** is not optimal for the assumed loss. No real product SLA or C++→ONNX inference demonstrated.

---

## Mandarin Text-Normalization Demo (Round 3)

- Built a standalone rule-based Mandarin vehicle-message normalization CLI with 21 curated regression examples (distance, speed, clock time, temperature, percentages) and explicit **visual** phrase-break hints; **no** acoustic TTS synthesis or MOS/audio quality measurement claimed.

---

## CTC-Decoding-Bench — Oracle validation, Dev sweep, held-out Test (Rounds 1–3)

**Status:** One pre-registered 4-condition evaluation on 50 held-out Test utterances complete; independent 200-hypothesis audit PASS. Dev/Test are utterance-disjoint but not speaker-disjoint; model training overlap unverified.

Measured, evidence-qualified bullets:

- Fixed and hashed a separate Chinese Zipformer-CTC ONNX model and exported 80 real `[T,1000]` log-probability tensors from source-checked AISHELL-1 utterances (30 exploratory Dev / 50 held-out Test); implemented source-hashed Greedy, Byte-BPE and CER/S/D/I evaluation (30-Dev Greedy corpus CER **2.94% over 442 characters**).
- Implemented log-domain CTC Prefix Beam Search with Blank/Non-Blank state transitions and a full-token exact beam ranking; verified **all prefix probabilities** against independent exhaustive Alignment Oracles and checked finite-width behavior against a literal reference algorithm.
- Measured 36 Prefix Beam/Blank-shortcut Dev configurations: **all had 2.94% CER**. At Beam 16, Hard Skip θ=0.90 bypassed **80.4% of CTC frames** and was **3.68× faster decoder-only than unpruned Beam 16** on the local Python/WSL2 benchmark, but altered **2/30 transcripts**, despite equal aggregate CER. It remained far slower than separate Framewise Greedy.

- Pre-registered and completed a single evaluation of 50 held-out Mandarin utterances (**729 reference characters**) using Greedy, Beam 16 unpruned, Beam 16 Forced Blank θ=0.95 and Beam 4 Forced Blank θ=0.95. All four gave **1.6461% corpus CER (S/D/I=11/0/1)** and exactly the same token sequences on all 50 Test utterances; independently audited **200 hypotheses**.
- Quantified the cost of unhelpful search without a language model: **0.0264 ms/utterance Greedy** versus **15.5411 ms Beam 16 unpruned** (~**589×** decoder-only time), with no Test CER or hypothesis benefit. Beam 16 Forced Blank θ=0.95 reduced that beam's time to **5.1219 ms/utterance** (~**3.03×** faster than unpruned Beam 16), yet remained ~**194×** slower than Greedy. **Greedy alone occupies the Test CER–latency Pareto frontier** among the four fixed options. Values are local WSL2 Python decoder-only timings, **not** product response latency.

**Pending:** Manual audio listening to validate the three deterministic Bad Case candidates; no acoustic causal conclusion from posterior summaries alone.

**Do not claim:** universal Greedy superiority, lossless Forced Blank, general Beam CER gain, speaker-independent robustness, end-to-end ASR latency or redistribution rights for external model weights. See `docs/CTC_ROUND3_STATUS.md` and `results/ctc_test_final_results.json`.

---

## Interview Themes

- Correctness before optimization.
- Measure latency and accuracy together instead of optimizing one metric blindly.
- Prefer a simpler verified solution over a more complex unmeasured approach.
- Treat negative experimental results as engineering conclusions when supported by controlled evaluation.
