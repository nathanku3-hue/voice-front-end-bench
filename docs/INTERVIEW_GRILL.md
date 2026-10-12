# Interview Grill — Speech Systems Engineering Evidence

Use this as a **question → decision → measurement → counterfactual** guide, not a memorized defense. Everything attributed to measured code has a file path; a proposed extension is explicitly labeled.

## 1. What exactly did you build?

**Answer:** Three *separate* and inspectable works, not a shipped speech assistant:
- Independent C++17 Kaldi-style streaming 80-bin Fbank with chunk-invariance and stage traces ([source](../src/streaming_fbank.cpp), [tests](../tests/test_streaming.cpp)).
- Fixed public Chinese sherpa-onnx online Transducer evaluated under five finalization/endpoint thresholds on the same 30 AISHELL-1 test utterances ([sweep](../scripts/run_endpoint_sweep.py), [full report](ROUND2_EVALUATION_REPORT.md)).
- Small Chinese vehicle-message **text-normalization** and visual pause-hint demo; not text-to-speech synthesis ([demo](../scripts/tts_tn_demo.py)).

**Do not say:** the C++ Fbank feeds sherpa, that the TN demo is NLU extraction, or that this is a single deployed production pipeline.

## 2. Why implement Fbank instead of training an ASR model?

The question was **deterministic feature correctness and numerical error localization**, not model generalization. C++ Fbank exposes framing, windowing, FFT, Mel accumulation and log as testable stages. Model training would make attribution harder, without testing this low-level invariant.

**Counterfactual:** If matched reference features were not reproducible across packetization, a model-level CER change could not tell whether a streaming feature bug was responsible.

## 3. Why didn't you stop at a single Fbank MAE?

The large float32 log-Mel outlier arose from low-energy pre-log differences, not an active floor on the diagnosed chirp band. Holding Mel weights and accumulator fixed while swapping float32 window/FFT stages localized the arithmetic contributions ([stage crossover report](EXPERIMENT_REPORT.md)). **Do not** extrapolate the stage attribution to every band and real speech.

**Stress test:** Reconstruct `ln(E_float32 / E_float64)` from measured pre-log energies, then check that both exceed epsilon before declaring a log-floor problem.

## 4. Why wasn't your C++ Fbank connected to ASR?

For the **endpoint experiment**, leaving the recognizer's native frontend fixed isolates the Rule2 change. But that alone did not prove *compatibility*. We separately tested native sherpa `OnlineStream.get_frames` against the C++ frontend.

Under real Round 2 frontend settings: `snip_edges=false`, high Mel cutoff 7600 Hz. C++: `snip_edges=true`, high cutoff 8000 Hz. Naive per-index MAE: **0.937197** on 559 real-speech frames. After matching the **native feature settings** to the C++ contract, MAE fell to **4.58e-6**, with a max difference of ~**2.19e-4** ([script](../scripts/audit_fbank_contract.py), [audit](ROUND3_VALIDATION_REPORT.md)).

**Counterfactual:** If matched configs still had large differences, investigate input scaling, window definition, FFT precision and per-bin weights before claiming interchangeability. **Not tested:** model Encoder caches, logits or final transcription using the custom Fbank.

## 5. What did Round 2 optimize?

A finalization **policy**, not the ASR weights. Corpus CER in this small fixed study moved from **64.76% at 300 ms** to **13.65% at 1200 ms** as deletions fell from **244 to 7**. Speaker-cluster paired bootstrap described uncertainty inside the observed 11 speakers ([data and caveats](ROUND2_EVALUATION_REPORT.md)). A lower early-endpoint proxy is not a human-verified product improvement.

**Stress test:** Explain S/D/I separately. Do not label all deletion edits as proven natural-pause cuts unless actual audio and hypotheses are inspected for that mechanism.

## 6. What's the real cost of increasing the threshold?

The threshold increment is **not** the measured finalization delay. We used the same utterance ID's `endpoint_audio_time_s` in two runs and subtracted. For **700→1200 ms**, mean paired extra *stream-time* **1179.2 ms**, median **320 ms**. Human `speech_end_s` reference cancels from this comparison; it is unpaced stream replay, **not** microphone-to-response time ([audit](ROUND3_VALIDATION_REPORT.md)).

**Counterfactual:** A true product recommendation requires a declared SLA, human-labeled speech ends and realistic paced capture.

## 7. Why not declare 700 ms the Pareto knee?

Because an optimal point requires an explicit objective and latency/failure-cost constraint. The observed 700→1200 ms change still removed **59 deletion edits**, and time effects are heterogeneous across utterances. Neither a common annotation nor a deployment SLA has yet been measured. There is **no verified production winner**.

**Measured follow-up (P3-B):** We explicitly defined a hypothetical per-utterance objective `mean endpoint stream-time + λ × (deletions/30)`, with λ in ms/deleted character. The computed decision-envelope switch points were ~**332.53**, **494.34**, **758.87** ms/char, with winning thresholds **300→500→900→1200 ms**; **700 ms was never selected**. This overturned the intuitive knee suggestion, but **does not set a vehicle SLA**. [Cost/assumptions](P3_DECISION_REPORT.md).

**Good follow-up:** Define cost of late response, cost of premature finalization and maximum acceptable latency tail; then test candidate thresholds on **independently annotated commands**.

## 8. Does matching Fbank features imply identical recognition?

**No.** We built a 2×2 experiment varying *native sherpa* `snip_edges` and `high_freq` independently, under both endpoint-disabled and fixed Rule2=1200 conditions. The native baseline at 1200ms was **13.65% CER (46/7/2)**, but the native setup using the C++ *options* produced **14.89% (47/7/6)**, with **9/30 transcripts changed**. Changing framing alone yielded **16.13%**, so the parameters interact. These are controlled outputs on the same 30 read-speech files, not generalized model robustness. [All eight runs](P3_DECISION_REPORT.md).

**Counterfactual:** Genuine C++→ONNX drop-in proof requires injecting custom feature frames and verifying encoder states, Joiner and decoded text; neither was done.

## 9. Why not add NS, AEC, beam search or SIMD now?

These answer **different** questions. Noise processing changes the acoustic input, decoder changes search behavior, SIMD changes computational performance. They can be valuable in separate controlled studies, but do not improve causal isolation of the current finalization sweep without a new hypothesis.

**Follow-up gating:** Profile feature-extraction bottlenecks before SIMD; define noise/SNR data before NS; validate model state and feature alignment before any ONNX encoder injection. Manual speech-end labels are not required to calculate *paired stream-time deltas*, but are required to claim true post-speech latency and false-cutoff performance. Avoid unsupported fixed per-frame CPU timing or model-CPU share claims. CTC prefix beam search is an explicitly separate planned sibling project, not the decoding algorithm in this Transducer study.

## 10. What result would invalidate your best claims?

- Identical ordered WAVs/model hashes fail to reproduce measured endpoint stream-time patterns → question evaluation reproducibility.
- Independent human endpoint labels reveal large systematic error in the RMS proxy → withdraw post-speech latency interpretations, retain only direct stream-time comparisons.
- Feature agreement collapses after reproducing the exact shared Fbank configuration → revisit precision/parity diagnosis, do not claim drop-in.
- An unseen-command or in-cabin dataset reverses threshold ordering → restrict the observed CER conclusion to this AISHELL-1 test subset.

**Bottom line:** The hiring signal is not “I used a complex model.” It's “I know which change was controlled, how to measure its consequence, how to test alternative explanations, and when evidence is insufficient for deployment.”
