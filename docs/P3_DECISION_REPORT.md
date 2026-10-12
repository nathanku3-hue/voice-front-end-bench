# P3 — Decisions, not tuning: endpoint cost and frontend sensitivity

**Executed:** 2026-10-12 (local WSL2); previous Round 2/P0/P1 measurements preserved. **Status:** complete bounded experiment, **not** a vehicle SLA, C++→ONNX injection, or production operating-point selection.

## Problem / evidence / decision

1. **Which Rule2 threshold should a product choose?** A threshold cannot be called "optimal" without a value assigned to waiting and transcription failure. We evaluate an **explicit hypothetical cost function** using the already-recorded five-threshold experiment. The result is that **700 ms never minimizes that specific linear objective** on the 30-utterance pilot.
2. **Does a frontend configuration change affect decoding?** Previous feature-contract parity was *feature-level only*. We now vary `snip_edges` and Mel upper cutoff as a **2×2 ablation in sherpa's OWN waveform frontend**, measure actual corpus CER/S/D/I under endpoint disabled and endpoint enabled, and compare decoded strings.
3. **Why not write our own Transducer/ONNX adapter?** This experiment asks whether the model is sensitive to changes in **native** frontend settings. It does not ask if the custom C++ Fbank can be injected into the encoder; that is a separate and unverified engineering gate.

## P3-B: Hypothetical endpoint deletion-vs-waiting decision model

![Measured-source cost sensitivity, not a product SLA](../results/p3_cost_tradeoff.png)

Define the *per-utterance* cost of a threshold `theta` as:

```text
J(theta; lambda) =
    mean_{30 WAVs}(endpoint_audio_time_s * 1000)
    + lambda_ms_per_deleted_char * (sum(deletion_edits) / 30)
```

Both terms have the same units: **ms/utterance**. The first term includes stream-time taken until finalization (including synthetic silence appended during replay). The second is a **hypothetical** and **uniform** per-character deletion penalty. Lambda has units of ms per deleted character. **No real business-value, navigation-intent or user-perceived delay labels were measured.**

| Rule2 threshold | Mean endpoint stream-time | Relative to 300 ms | Deletions / 30 utterances | CER |
|---|---:|---:|---:|---:|
| 300 ms | 2153.7 ms | 0 ms | 244/30 | 64.76% |
| 500 ms | 3738.8 ms | +1585.1 ms | 101/30 | 32.01% |
| 700 ms | 4364.2 ms | +2210.5 ms | 66/30 | 25.56% |
| 900 ms | 4809.9 ms | +2656.1 ms | 36/30 | 20.10% |
| 1200 ms | 5543.4 ms | +3389.7 ms | 7/30 | 13.65% |

The pairwise line-crossings alone do not determine the winner; evaluate the **lower envelope** of all five straight-line costs. The following are *open intervals* (exact boundaries have tied scores):

| Hypothetical lambda | Cost-optimal observed threshold |
|---|---:|
| 0 ≤ λ < **332.531** ms/char | 300 ms |
| **332.531** < λ < **494.344** ms/char | 500 ms |
| **494.344** < λ < **758.866** ms/char | 900 ms |
| λ > **758.866** ms/char | 1200 ms |

**Finding:** **700 ms is not selected for any positive lambda** under this particular linear cost. Although comparing only 700→1200 yields a pairwise crossover of ~599.6 ms/deletion, at that lambda **900 ms beats both** on this fixed sample. Likewise, the suggested "λ = 15–20 ms/char selects 700–1200" is incompatible with the recorded inputs.

**Counterfactual / limits:** The 300 ms condition often finalizes before the speech is complete; its low stream-time is therefore not equivalent to providing a correct response sooner. The cost function does **not** penalize substitutions/insertions, entire command failures, tail latency, or abstentions. Sampling 30 read-Mandarin clips does not estimate in-car intent utility; lambda values are hypothetical, not calibrated. **No production SLA or chosen threshold is justified.**

**Offline reproduction (no model/audio download):**

```bash
python scripts/eval_sla_tradeoff.py --output build/p3_cost_recheck.json --csv build/p3_cost_recheck.csv
# Optional chart (requires matplotlib):
python scripts/eval_sla_tradeoff.py --plot build/p3_cost_recheck.png
```

Source: [code](../scripts/eval_sla_tradeoff.py) · [machine-readable decision envelope](../results/p3_cost_tradeoff.json) · [numeric summary CSV](../results/p3_cost_observations.csv) · [original paired JSON](../results/endpoint_sweep/).

## P3-A: 2×2 sherpa-native frontend configuration sensitivity

![Corpus CER under two replay modes and four native frontend configurations](../results/p3_frontend_cer.png)

**Control:** 30 exact AISHELL-1 test WAVs (11 speakers, 403 independently cross-checked reference characters) with matching WAV SHA-256, identical order/manifest SHA, same public 14M Chinese streaming Transducer and model-file SHA-256, two CPU threads, greedy decoding and 160 ms chunks. The only manipulated **feature** variables are:
- `snip_edges=False/True` (native default / match custom C++ framing);
- `high_freq=-400/0` (upper Mel cutoff 7600/8000 Hz in sherpa).

Two separate replay modes test downstream context:
- `no_endpoint`: endpoint detection disabled; file receives the same 2500 ms padded tail, then `input_finished()`. **Not an automatically verified perfect-transcript oracle.**
- `rule2_1200`: endpoint enabled with Rule2 1200 ms; other endpoint rules disabled as in Round 2; same tail padding available to every variant. No real microphone clock.

| Native feature settings | Endpoint disabled: CER / S-D-I | Rule2 1200 ms: CER / S-D-I | Changed decoded strings vs native (of 30) |
|---|---|---|---:|
| **Native**: snip false, 7600 Hz | **12.66%** / 46-3-2 | **13.65%** / 46-7-2 | 0 |
| **Framing only**: snip true, 7600 Hz | 15.14% / 54-3-4 | 16.13% / 54-7-4 | 10 |
| **Cutoff only**: snip false, 8000 Hz | 12.90% / 44-3-5 | 13.90% / 44-7-5 | 8 |
| **Matched to C++ settings**: snip true, 8000 Hz | 13.90% / 47-3-6 | 14.89% / 47-7-6 | 9 |

All eight conditions actually ran. Reproduced native Rule2=1200 baseline (**13.65%, S/D/I 46/7/2**) matches the archived Round 2 baseline.

**Findings:**

- **Front-end config matters for ASR:** matching sherpa's *native* frame/cutoff settings to the C++ options increases CER from **13.65% to 14.89%** (+1.24 percentage points) under Rule2=1200 in this corpus; **9/30** transcripts changed. The best observed of the four conditions is the **native configuration** in both modes.
- For these exact WAVs, the four 1200-ms endpoint runs had **7 deletion edits each**, whereas the four no-endpoint runs had **3 each**. The feature configuration effect appeared in **substitutions/insertions**, not in the aggregate deletion count. This is descriptive, not a generalized invariance claim.
- Observed 2×2 CER interaction `CER(both) - CER(snip_only) - CER(cutoff_only) + CER(native)` is **−1.489 percentage points** in both replay modes. Four configurations on 30 utterances do not establish population significance.
- This is **not** C++→ONNX model inference. The C++ feature configuration and sherpa matching configuration previously had feature MAE **4.58×10⁻⁶** on one WAV ([P1 audit](ROUND3_VALIDATION_REPORT.md)), but we never fed the C++-generated features to the Encoder. Numeric feature agreement is not proof of identical decoded words.

**Full re-run (requires downloaded WAVs/model, sherpa-onnx and NumPy):**

```bash
MODEL="$HOME/.cache/voice-front-end/models/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23"
~/.venvs/voice-bench/bin/python scripts/run_frontend_sensitivity.py \
  --manifest data/round2_endpoint_eval/manifest.jsonl --model-dir "$MODEL" \
  --output-root build/p3_frontend_recheck \
  --summary build/p3_frontend_recheck/summary.json \
  --csv build/p3_frontend_recheck/summary.csv \
  --plot build/p3_frontend_recheck/cer.png
```

**Offline audit of the already recorded eight run JSONs** (does not load the model, but the plot requires matplotlib):

```bash
python scripts/run_frontend_sensitivity.py --summarize-only \
  --output-root results/p3_frontend_runs --summary build/p3_summary_recheck.json \
  --csv build/p3_summary_recheck.csv
```

Artifacts: [runner](../scripts/run_frontend_sensitivity.py) · [native ASR CLI exposing the two parameters](../scripts/asr_stream_bench.py) · [raw 8 runs](../results/p3_frontend_runs/) · [summary JSON](../results/p3_frontend_summary.json) · [summary CSV](../results/p3_frontend_summary.csv).

## Interview: what did you choose NOT to build, and why?

| Out of scope **for this experiment** | Engineering decision and re-entry condition |
|---|---|
| Custom stateful Zipformer ONNX Decoder/Joiner integration | Not necessary to measure *native frontend parameter sensitivity*. Required if the question becomes true C++ Fbank→model equivalence; **not proved here**. |
| Human speech-end labels in this paired-delta cost analysis | The stream-time difference is computed within the **same** WAV; unknown `speech_end_s` cancels. Human annotations still **mandatory** if claiming post-speech product latency or false cutoff; not "unnecessary data work" globally. |
| Neural VAD, NS or AEC | These change the problem and may be useful as independent factorial studies if noise/voice-activity reliability is the target. They should not be mixed into the fixed-frontend endpoint ablation. |
| Handwritten SIMD / NEON | Do not spend platform-specific optimization effort until profiling shows Fbank is a bottleneck on the target SoC. No unsupported "95% CPU is inference" assumption. |

**Best 30-second interviewer response:**

> I fixed the ASR model and tested two independent decisions. First, I made the endpoint deletion/waiting trade-off explicit with a unit-consistent hypothetical cost: 700 ms, our earlier intuitive "knee," was never cost-optimal on the observed sweep. Second, I isolated native feature framing from Mel cutoff in a 2×2 ASR ablation. Matching sherpa's frontend settings to my C++ frontend changed 9 of 30 decoded transcripts and raised CER by 1.24 percentage points in the 1200-ms condition. I did not pretend that was custom-feature ONNX integration, or that a replay-time cost model establishes a production SLA.

[Round 2 source and limitations](ROUND2_EVALUATION_REPORT.md) · [Prior feature-contract audit](ROUND3_VALIDATION_REPORT.md) · [Interview grill](INTERVIEW_GRILL.md)
