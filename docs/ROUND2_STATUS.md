# Round 2 — Integration Status and Endpoint Pilot

**Current status (2026-10-12): Round 2 CLOSED as an exploratory R2-A/R2-B evaluation; independent human speech-end latency validation is still NOT DONE. Optional R2-C Noise Suppression is DEFERRED.** The original 2026-10-11 single-file pilot below is retained as history. Current multi-sample analysis: [Round 2 Evaluation Report](ROUND2_EVALUATION_REPORT.md) and [endpoint plot](../results/endpoint_tradeoff.png). See [locked plan](PROJECT_PLAN.md).

## Verified system

- Model: sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23, online Transducer (not self-trained); version sherpa-onnx 1.13.8.
- Model archive: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2
- Archive SHA-256: 2cbd71b640d9c37d3784f29367333a4577b0398b62e9deeed418170b081cba8b
- Machine: Ubuntu 24.04 WSL2, Python 3.12, NumPy 2.3.5, CPU, 2 inference threads, 16kHz mono PCM16 -> normalized float32; chunk size 160ms.
- Audio: official model archive test_wavs/0.wav, 5.6115 seconds, SHA-256 recorded in [pilot JSON](../results/round2_pilot.json).
- Data flow: WAV -> sherpa-onnx OnlineRecognizer waveform API -> partial/final transcript. The independent C++ Fbank from Round 1 is **not connected** to this recognizer.

## Reference provenance limitation

The EvalScope WenetSpeech dev example #0 contains the distinctive sentence:

> 对我做了介绍啊那么我想说的是呢大家如果对我的研究感兴趣呢嗯

Source: https://evalscope.readthedocs.io/zh-cn/latest/benchmarks/wenet_speech.html . It also reports approximately 175.3 KB of audio, consistent with the model archive sample. **Identity of the actual audio bytes across those two distributors has not been independently verified**. The CER below is *provisional* and must not be presented as a certified WenetSpeech benchmark, let alone vehicle-command performance. CER normalizes NFKC/casefold and strips Unicode punctuation and whitespace, but does not translate or normalize word readings.

The model's full decoded hypothesis without endpoint detection is:

> 对我做了介绍那么我想说的是大家如果对我的研究感兴趣

Against the provisional reference: 0 substitutions, 4 deletions, 0 insertions among 29 reference characters, yielding CER 13.79%.

## Single-file endpointing study (paced file replay)

The same fixed model, file and hardware were used. Rule1 trailing-silence=20s and rule3 utterance length=120s disable those endpoint triggers for this clip; only rule2 threshold changes. Each 160ms audio chunk is submitted **after its last sample is available**. All timing values are measured from the start of file replay.

| Rule2 trailing silence | Endpoint stream-time | Endpoint wall-time | Provisional CER | Error counts S/D/I |
|---|---:|---:|---:|---:|
| 400ms | 2.080s | 2.086s | 79.31% | 0/23/0 |
| 800ms | 5.9315s | 5.936s | 13.79% | 0/4/0 |
| 1200ms | 6.2515s | 6.257s | 13.79% | 0/4/0 |

At 400ms the recognizer stopped after “对我做了介绍”, while the WAV continued to 5.6115s. At 800ms and 1200ms it kept the same full hypothesis; the 800ms condition endpointed 320ms earlier on the stream timeline. First non-empty partial output occurred around 0.800s stream-time / 0.806s paced replay wall-time.

These results are **not** end-of-speech response latencies: we have no independently annotated speech start/end in the WAV. File replay is not a physical microphone or product pipeline. The sample is not representative of in-car Chinese commands. One-point CER/latency trade-off is illustrative, **not** a validated Pareto curve.

## Reproduction (WSL2)

Requirements for Round 2 only: Python 3.12, packages pinned in [requirements-round2.txt](../requirements-round2.txt). In WSL, with the repo mounted:

    cd /mnt/e/Code/voice
    python3 -m venv ~/.venvs/voice-r2
    ~/.venvs/voice-r2/bin/python -m pip install -r requirements-round2.txt
    mkdir -p ~/.cache/voice-front-end/models
    cd ~/.cache/voice-front-end/models
    curl -fL https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2 -o model.tar.bz2
    sha256sum model.tar.bz2
    tar -xjf model.tar.bz2
    cd /mnt/e/Code/voice
    MODEL="$HOME/.cache/voice-front-end/models/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23"
    WAV="$MODEL/test_wavs/0.wav"
    ~/.venvs/voice-r2/bin/python scripts/asr_stream_bench.py --model-dir "$MODEL" --wav "$WAV" --chunk-ms 160 --output build/r2_smoke.json

For the **source-matched provisional pilot** append --reference '对我做了介绍啊那么我想说的是呢大家如果对我的研究感兴趣呢嗯' and optionally --endpoint --rule2-ms 400/800/1200 --paced (one value per run).

For a real labeled corpus, use a JSONL manifest with one object per row, such as:

    {"id":"sample-001","wav":"audio/001.wav","reference":"导航回家","speech_end_s":2.18}

The reference must be an independently verified human transcript. The speech_end_s is in seconds from the WAV start; it must also be independently annotated. This example is schematic and **not** a supplied audio file.

## Status and next acceptance gate

- Integration: model loaded, incremental decode succeeded on local WSL2, source audio and model SHA-256 recorded.
- Tests: 8 Python unit tests passed (character edits, normalization, manifest, WAV I/O); Round 1 CTest 1/1 remains green.
- Evaluation: endpoint threshold pilot saved in [round2_pilot.json](../results/round2_pilot.json); it is not a valid multi-condition CER-vs-latency claim.
- Next: acquire 30–50 independently annotated Mandarin commands with speech-end markers and noise metadata, then repeat conditions on exactly the same utterances. Without that, no claim of product improvement, truncation rate, or statistically robust threshold choice.
- 2026-10-11 dataset extension: 30 AISHELL-1 test WAVs from 11 speakers, original transcript crosschecks and per-WAV SHA-256; five Rule2 thresholds completed. CER 64.76%, 32.01%, 25.56%, 20.10%, 13.65% for 300/500/700/900/1200ms, respectively. A separately computed paired speaker-bootstrap interval exists. **28/30 VAD-estimated speech ends are boundary-sensitive and not manually annotated.** See [standalone report](ROUND2_EVALUATION_REPORT.md) for all qualifications. This is a *read-Mandarin* experiment, not a car-command product latency benchmark.
- NS ablation remains explicitly optional and is **not** done; AEC/TTS remain frozen.
