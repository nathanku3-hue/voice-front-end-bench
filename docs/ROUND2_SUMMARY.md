# Round 2 — one-page outcome and interview story

**Closure:** R2-A (streaming recognizer + CER tests) and R2-B (five-setting paired endpoint sweep) are complete as a *small, controlled experiment*. R2-C (WebRTC NS) is optional and **deferred**. Closure does **not** mean production-grade latency or in-car accuracy is validated.

## One concrete engineering question

When a streaming ASR recognizer finalizes a hypothesis after trailing silence, is an earlier endpoint worth the risk of dropping speech that continues? **Optimization knob:** sherpa-onnx Rule2 trailing-silence threshold, *not* ASR training, model weights or C++ Fbank settings.

**Controlled setup:** 30 AISHELL-1 test read-speech recordings, 11 speakers and 403 independently cross-checked reference characters; sherpa-onnx 1.13.8 public 14M Chinese online Zipformer Transducer; 160 ms waveform chunks, 2 CPU threads, unpaced replay; Rule2 = {300, 500, 700, 900, 1200} ms. WAV SHA-256 values and source IDs are in the [manifest](../data/round2_endpoint_eval/manifest.jsonl).

![Round 2 endpoint sweep](../results/endpoint_tradeoff.png)

| Threshold | CER | S / D / I | Estimated early-endpoint proxy | Mean delay for **non-early only** (n) |
|---|---:|---:|---:|---:|
| 300 ms | 64.76% | 17 / **244** / 0 | **27/30** | −1 ms (3) |
| 500 ms | 32.01% | 27 / 101 / 1 | 11/30 | 352 ms (19) |
| 700 ms | 25.56% | 35 / 66 / 2 | 7/30 | 604 ms (23) |
| 900 ms | 20.10% | 43 / 36 / 2 | 5/30 | 665 ms (25) |
| 1200 ms | **13.65%** | 46 / **7** / 2 | **1/30** | 977 ms (29) |

## What changed, what was sacrificed?

- **Visible gain within the set:** 300→1200 ms gives **−51.12 percentage points CER**, with 237 fewer deletion edits. Deletion dominates the measured improvement, suggesting truncation from premature finalization instead of merely poor lexical recognition.
- **The trade:** increasing trailing-silence requirements waits longer to commit a final transcript. The chart's delay series is **not a paired, same-population product-latency curve**: non-early counts vary from 3 to 29.
- **Uncertainty:** paired **speaker-cluster bootstrap**, 5,000 resamples of 11 speakers, yielded a 95% interval of **[−59.10, −42.55] percentage points** for CER(1200)−CER(300). This quantifies sensitivity within this observed set, **not** a claim for car drivers or independently trained systems.

## Critically important limitations

1. Speech-end labels were algorithmically estimated by reverse RMS at −35 dB **relative to each recording's peak**. **28/30 clips** lack a clear trailing-silence region. Therefore the “early-endpoint” metric is a **proxy**, and the chart's delay is **not manual ground-truth latency**.
2. The stream-time replay was **not paced** as live microphone input; no end-to-end response latency, barge-in, double-talk or actual in-car commands were measured.
3. References were independently transcript-crosschecked, but overlap between model training data and test audio has **not** been separately audited.
4. The custom Round 1 C++ Fbank was **not connected** to the sherpa-onnx waveform frontend; the endpoint results are not Fbank gains.
5. **No production threshold is selected** from this experiment alone. Required next step: human speech-end annotation, realistic command/cabin recordings and paced paired trials.

## Why this is more than “ran an ASR repo”

The work offers an **audit trail for a system-design decision**: controlled input hashes and reference custody → fixed-model, five-knob sweep → per-utterance S/D/I failure attribution → speaker-aware *paired* uncertainty → documented scope and latency limits. The central insight is that an endpoint rule can masquerade as an acoustic recognition problem when deletions explode.

**30-second interview explanation:**

> I isolated the endpoint finalization threshold in a fixed streaming Chinese ASR pipeline. On 30 AISHELL test utterances, using 300 ms trailing silence produced 244 deletion errors and 64.76% CER; at 1200 ms those values were 7 and 13.65%. I checked source transcript identity, audited per-speaker paired variation, and made the latency caveat explicit: my current speech-end markers are reverse-RMS estimates, not ground truth. So this is a repeatable reliability-vs-waiting experiment—not a claim that I optimized real car UX.

**Re-run speaker-aware analysis with no model/WAV download:**

```bash
python scripts/paired_endpoint_ci.py --low 300 --high 1200 --reps 5000 --output build/paired_ci_recheck.json
```

[New: paired timing and real native Fbank comparison](ROUND3_VALIDATION_REPORT.md) · [Full report](ROUND2_EVALUATION_REPORT.md) · [source-manifest verifier](../scripts/verify_round2_manifest.py) · [raw summary CSV](../results/endpoint_summary.csv) · [paired bootstrap JSON](../results/paired_endpoint_ci.json) · [dataset method](ROUND2_DATASET.md) · [runner](../scripts/run_endpoint_sweep.py)
