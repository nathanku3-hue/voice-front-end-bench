# Round 2 — AISHELL-1 Test Dataset / Endpointing Evaluation

For the complete results, caveats, plot and reproducibility commands, see [**ROUND2_EVALUATION_REPORT.md**](ROUND2_EVALUATION_REPORT.md) and [the plotted chart](../results/endpoint_tradeoff.png).

## Dataset: 30 real test WAVs, 11 speakers

Status: **30/30 accepted and independently transcript-matched, test-only**. The original [OpenSLR SLR33](https://www.openslr.org/33/) archive is about 15 GB; this reproducible small evaluation uses per-row WAV downloads from the [TwinkStart AISHELL-1 test viewer](https://huggingface.co/datasets/TwinkStart/AISHELL-1), **7,176 test rows**, and crosschecks text against the [AISHELL/AISHELL-1 official transcript](https://huggingface.co/datasets/AISHELL/AISHELL-1/tree/main/data_aishell/transcript). All selected entries matched the original transcript after Unicode NFKC, case folding, whitespace/punctuation removal. Model-generated text was never used to label the data.

Fixed seed: `20261011`; selected sample count: **30**, distinct speaker IDs: **11**. Transcript SHA-256: `b5f33b9e0b47548e20a5ea4e504297f80df41d559133924c4e5b7b544c15b5c4`. Individual WAV SHA-256, source row index and speaker ID are recorded.

Local artifacts:
- `data/round2_endpoint_eval/manifest.jsonl`: reference, WAV-relative path, speaker, SHA-256, estimated speech end, estimate method and quality flag;
- `data/round2_endpoint_eval/provenance.json`: corpus source, seed, page offsets, 11-speaker counts and transcript checksum;
- `data/round2_endpoint_eval/audio/*.wav`: raw source audio, **Git-ignored**;
- `scripts/fetch_aishell_eval.py`: deterministic selection/download with transcript mismatch rejection.

## Speech-end reference is NOT manually annotated

We use reverse short-time RMS with **25ms frame and 10ms hop**, threshold **35dB below the maximum 25ms frame RMS** (NOT an absolute -35dBFS). A last above-threshold frame defines `speech_end_s`. A tail margin below 50ms is flagged `no_clear_tail`: **28/30 samples** received this flag. Consequently, `speech_end_s` is a **boundary-sensitive proxy**, not independently annotated ground truth. Any endpoint latency computed from it must be qualified; do not call it human or product latency. The original AISHELL transcript only supplies text, not word/utterance endpoint timestamps.

The matching source input, file hashes and labels are authentic, but model training overlap with AISHELL is **not independently audited**. The sherpa-onnx model documentation states this 14M model was trained on WenetSpeech; this does not prove zero overlap with all AISHELL clips.

## Paired Endpointing Sweep — measured outcomes

Fixed online model: **sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23**, sherpa-onnx 1.13.8, two CPU threads, greedy search, 160ms chunks, unpaced waveform replay (not physical microphone capture). Input corpus: **30 utterances, 403 reference characters**, Rule2 trailing silence varied only.

| Rule2 threshold | CER (S+D+I)/403 | Premature endpoint proxy | Mean on-time stream delay | On-time n |
|---|---:|---:|---:|---:|
| 300ms | 64.76% | 27/30 = 90.0% | -1ms | 3 |
| 500ms | 32.01% | 11/30 = 36.7% | 352ms | 19 |
| 700ms | 25.56% | 7/30 = 23.3% | 604ms | 23 |
| 900ms | 20.10% | 5/30 = 16.7% | 665ms | 25 |
| 1200ms | 13.65% | 1/30 = 3.3% | 977ms | 29 |

Early endpoint proxy: `endpoint_audio_time_s + 0.03 < speech_end_s`. Every condition observed 30/30 endpoint events. The on-time mean uses only samples not classified as early, so changing sample counts make those means not fully comparable. CER was computed with independently crosschecked original transcripts. Numerical confidence bounds in `results/endpoint_summary.json` are naive utterance-bootstrap estimates and **do not account for speaker clustering**. A separate paired **11-speaker cluster bootstrap (5,000 deterministic resamples)** was run for 300ms vs 1200ms: observed CER change **−51.12 percentage points**, approximate 95% cluster-bootstrap interval **−59.10 to −42.55 percentage points**. This reflects this small fixed speaker sample, not a population-wide performance claim. Details: `results/paired_endpoint_ci.json`; command: `python scripts/paired_endpoint_ci.py --low 300 --high 1200 --reps 5000`.

Artifact paths: `results/endpoint_sweep/rule2_{300,500,700,900,1200}ms.json`, `results/endpoint_summary.json`, `results/endpoint_summary.csv`, `results/endpoint_tradeoff.png`.

## Reproduction

On the host WSL2 Ubuntu installation, with `~/.venvs/voice-bench` already containing NumPy, sherpa-onnx and matplotlib:

```bash
cd /mnt/e/Code/voice
~/.venvs/voice-bench/bin/python scripts/fetch_aishell_eval.py --output-root data/round2_endpoint_eval --count 30 --seed 20261011 --pages 10
~/.venvs/voice-bench/bin/python scripts/run_endpoint_sweep.py --manifest data/round2_endpoint_eval/manifest.jsonl --model-dir "$HOME/.cache/voice-front-end/models/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23" --thresholds 300 500 700 900 1200 --chunk-ms 160 --threads 2
```

This is **real Mandarin speech, not Chinese car commands**, and **not a verified product-latency benchmark**. The 30-utterance sample is a small controlled study; higher thresholds improved text retention within this fixed data, but results must not be generalized to in-car barge-in, unseen cabin noise or arbitrary model variants. A credible real post-speech-latency claim requires separately human-labeled speech-end times and paced capture.

Next quality gate (deferred): manual boundary review of these 30 clips, separate command audio, a paired re-run with human timestamp labels and speaker-aware uncertainty estimates. WebRTC NS/AEC/TTS remain outside this dataset task.
