# Resume Bullets — ASR Portfolio

## VoiceFront-End-Bench — C++ Streaming Speech Front-End

**Focus:** DSP correctness, streaming behavior, numerical validation.

- Implemented a C++17 stateful streaming log-Mel Fbank extractor with deterministic frame generation under arbitrary PCM chunk boundaries; verified packet-boundary invariance through Release-mode tests covering fixed and irregular segmentation patterns.

- Benchmarked Kaldi-compatible Fbank behavior against an external reference implementation and performed stage-wise numerical diagnosis of low-energy spectral discrepancies, isolating float-precision sensitivity in preprocessing/windowing and FFT computation rather than assuming a single error source.

- Built reproducible speech-processing experiments and documentation covering edge cases, numerical limitations, and validation boundaries; avoided unsupported claims of bit-exact Kaldi parity or production real-time guarantees.

- Added caller-owned bounded output and backpressure flow control to the streaming extractor: a **1-hour (57.6 M-sample, 359,998-frame) soak at a consumer rate of exactly 50%** held the frame watermark at exactly the configured capacity (1/4/64 frames), kept **live-heap growth at 0 bytes** (RSS inside a 12 KiB band for the whole hour), and remained **bit-identical to an unthrottled baseline across 179,995 pause/resume cycles** (max |diff| = 0, MAE = 0 over 28,799,840 feature values). Leak audit uses exact `operator new` byte accounting, **not** Valgrind/ASan, which are unavailable on this host; the stream is processed 376× faster than real time, so **no real-time or device claim** is made.

- Eliminated allocation from the streaming hot path: replaced the per-frame `std::deque` with a caller-sized fixed ring that **refuses** an overflowing batch instead of growing or dropping, then proved it with a counting replacement of global `operator new` — **0 allocations, 0 denied requests and a 0-byte live delta across 10,000 frames processed under real backpressure**, with all 800,000 feature values bit-identical to an unthrottled reference. Bridge footprint is **5,992 B**; the 7 constructor allocations are reported as init, not hidden.

- Designed a **dual-watermark (hysteresis) flow-control policy** to stop FIFO chatter: at a 4-frame ring the single-watermark control produced **39,988 pause/resume transitions** over 20,000 frames (one handshake every two 10 ms ticks), while low watermarks of 2/1/0 cut that by **67%/80%/86%** at bounded occupancy, zero dropped frames and bit-identical output — and the audit publishes the cost (longest stall 2 → 7 ticks) instead of hiding it.

- Built a **pure C++17 decode pipeline** (WAV → bounded Fbank → CTC argmax/collapse → byte-BPE → UTF-8) with no Python in the process, reproducing the Python reference transcript **30/30** on the Dev set and implementing CPython's exact UTF-8 replacement policy for the two utterances whose greedy tail ends mid-character. Profiling the first version showed the naive argmax was **4× slower than the numpy call it replaced** (a serial index-carrying loop); splitting it into a max reduction plus a first-index search and using four independent accumulators made it **3× faster (1.43 → 0.46 µs per logit frame)**. The acoustic model is explicitly **not** in the binary — posteriors are a frozen export — so no C++ inference claim is made.

- Profiled the frontend with a **per-stage RAII timer** and found the bottleneck was not the FFT: a dense 80 × 257 mel filterbank consumed **88.5%** of frame time because it re-read **164 KB of coefficients every 10 ms frame**, while only **2.4%** of them are non-zero. Replacing it with a CSR of 501 taps built once at construction was **bit-exact** (0 of 159,840 values differ) and cut that stage **60×** and the whole frame **6.7×** (25.5 → 4.1 µs/frame, 392× → 2,445× realtime on this host); the report also states the 31% instrumentation overhead at that scale and why the absolute µs are host-local.

- Attacked the remaining **43.1%** of frame time (DC removal + pre-emphasis/window) by removing the modulo indexing from the inner loops and then vectorising: a 64-byte-aligned staging buffer with `__builtin_assume_aligned` + `_mm256_load_ps` (emitting `vmovaps`, not `vmovups`) and per-function `target("avx2,fma")` dispatch via a hand-written CPUID probe. **Verified in the emitted assembly, not in the source**: pre-emphasis went **16.0 → 1.938 instructions/sample** (80 `ymm` ops, 4 FMAs per iteration), the running mean **10.5 → 0.688**, and log compression **49.0 → 3.875** with no `logf` call left in the loop. Three arms measured **round-robin inside one process** (this host's per-pass wall time is bimodal at ~3×) with a published **control-stage drift of 6.75%** between stages that run identical code — so the **−81.7%** log-stage result is **13× the noise floor** of that stage — giving **3.971 → 3.080 → 2.729 µs/frame** (2,519× → 3,664× realtime), **−67.2%** on the sample-domain stages. The 4 remaining unaligned moves in the pre-emphasis stencil are counted and explained rather than hidden; instruction counts are static and bound no cycles on any specific core.

- Replaced the 80 `std::log` calls per frame with an **IEEE-754 exponent-decomposition kernel** (`x = 2^e·m`, fold into `[1/√2, √2)`, `log m = 2·atanh(u)` truncated after `u⁷`) plus an 8-wide AVX2 variant, and bounded it instead of asserting it: **6.29e-7 max absolute / 1.65e-7 max relative error** over 2M log-spaced points across `[FLT_EPSILON, 1e6]`. The double variant is only **21×** better, which localised the limit as *series truncation* (~3e-8) rather than mantissa width — buying accuracy takes another term, not another register — and the "second order is enough" strawman measured **97× worse**, which is why four terms ship. End-to-end on the frozen int8 Zipformer CTC: **30/30 transcripts identical**, feature MAE **3.14e-8**, CER **2.94% → 2.94%**, while the scalar-SIMD rewrite alone was **bit-identical** (0.0 error).

- Measured, rather than assumed, how much feature-domain error the model tolerates: injected uniform noise over **five orders of magnitude** into clean features and re-ran the same hash-verified model and C++ decoder. There is **no cliff** — transcripts flip at the *smallest* amplitude tested (5e-7 MAE) and the count does not grow with amplitude: five orders of magnitude of extra noise buys **two characters** of CER, and 6 of 10 perturbed rungs move the CER not at all. The instability is **three utterances out of 30**, one of which (`BAC009S0902W0222`, near-tied top-2 posteriors) flips at **5 of the 10** levels — a token decision moving, not accumulated damage. A determinism control confirmed the graph itself is bit-reproducible (`max |diff| = 0`), and the shipped path's **3.14e-8** feature MAE sits **1.2 decades below the smallest resolvable rung** while flipping **nothing**. This reframed the claim from "we are below the threshold" (there is none) to "the frontend is not the fragile part".

- Ran a **fixed-point feasibility audit** over 3,703 s of real speech (370,327 frames), recording per-stage min/max/peak/RMS plus log2 histograms of per-frame peak, per-frame RMS and per-frame crest factor. **No stage overflows** its proposed format (Q15 sample domain, Q31 power/mel, Q16.16 log) — the tightest headroom is 0.9 dB — but a plain **Q15 FFT reaches only 55.4 dB SQNR**, and the audit explains why instead of widening the word: the between-frame peak spread is **54 dB**, so a global full scale leaves quiet frames with almost no resolution (10.8 dB at p05), and even a **24-bit Q23** still gives the quietest 5% only 59 dB. The recommendation is **block floating point**: a 16-bit mantissa plus a 5-bit exponent gives **70.6 dB guaranteed** (including the 6.02 dB worst-case radix-2 exponent placement) because one exponent per frame removes the between-frame term entirely and leaves only the measured within-frame crest factor (17.1 dB mean). SQNR is analytical from measured RMS; no fixed-point frontend was implemented, so this is a range argument, not an end-to-end measurement.

- Modelled the frontend as an **AXI4-Stream slave/master pair** with a 64-byte burst policy: `TLAST` on every 16th beat (a cache line, so the DMA retires per line) and `TUSER` on the beat completing the 640-byte feature frame, rather than forcing a 160-beat packet open across ten lines. Verified the handshake rules that actually break designs — `TVALID` never derived from `TREADY`, `TREADY` a pure function of FIFO occupancy, transfer iff both high — across three `TREADY` scripts and 420,000 cycles: **0 protocol violations**, **398/398 frames delivered bit-identically** in every arm, `S_AXIS_TREADY` first dropping at cycle 1,232 and staying low for 31,744 of the 140,000 cycles with the 512-sample FIFO saturating exactly, **zero allocations** in the tick loop, and **stall-invariant output** (backpressure changes *when*, not *what*). Emitted a **VCD** and rendered the waveform showing the beat held in the register and the backpressure walking upstream. Cycle counts are model cycles; `TVALID` is combinational here and this is not RTL.


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
- Swept beam width **1/2/4/8/16/32** on the frozen 30-utterance Dev logits (nine round-robin passes, host pinned to one CPU): CER stayed at **2.94% with 0/30 token sequences differing from W=32 at any width**, while decoder time grew **4.4×** (0.077 → 0.338 ms per CTC frame). The posterior is near-one-hot (median top-1 = 0.9988; top1−top2 margin > 0.9 on 91% of frames), so beam width is a **cost knob, not an accuracy knob**; W=8 is dominated by W≤4. Values are a CPython decoder-only prototype, not device latency, and the 4.4× ratio drifted ~±25% between sweeps before pinning.
- Quantified the cost of unhelpful search without a language model: **0.0264 ms/utterance Greedy** versus **15.5411 ms Beam 16 unpruned** (~**589×** decoder-only time), with no Test CER or hypothesis benefit. Beam 16 Forced Blank θ=0.95 reduced that beam's time to **5.1219 ms/utterance** (~**3.03×** faster than unpruned Beam 16), yet remained ~**194×** slower than Greedy. **Greedy alone occupies the Test CER–latency Pareto frontier** among the four fixed options. Values are local WSL2 Python decoder-only timings, **not** product response latency.

**Pending:** Manual audio listening to validate the three deterministic Bad Case candidates; no acoustic causal conclusion from posterior summaries alone.

**Do not claim:** universal Greedy superiority, lossless Forced Blank, general Beam CER gain, speaker-independent robustness, end-to-end ASR latency or redistribution rights for external model weights. See `docs/CTC_ROUND3_STATUS.md` and `results/ctc_test_final_results.json`.

---

## Interview Themes

- Correctness before optimization.
- Measure latency and accuracy together instead of optimizing one metric blindly.
- Prefer a simpler verified solution over a more complex unmeasured approach.
- Treat negative experimental results as engineering conclusions when supported by controlled evaluation.
