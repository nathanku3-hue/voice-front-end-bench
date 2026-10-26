# Bounded C++ frontend → public CTC recognizer | local engineering evidence

**Status:** working local engineering extension / resume **candidate**, NOT production ASR and NOT the locked JD resume in E:\\code\\job.

## Why extend this work?

The original C++ streaming Fbank provided frame-consistent features under irregular input chunks, but its output queue could grow if the consumer did not read. The independent CTC study used a third-party feature extractor, so our custom C++ implementation was not actually part of model inference.

The new slice runs real source-checked 16-kHz WAV → bounded C++ PCM/Fbank streaming → float feature CSV → public offline Zipformer-CTC ONNX → greedy Byte-BPE → Chinese text. The C++ interface stops consuming inputs when no frame-output space remains, leaving unconsumed PCM with its caller. The CLI reads bounded blocks, stages its output and publishes the final CSV only on success. This does **not** prove streaming ONNX inference or user-perceived latency.

## Concrete implementation and independent upstream boundary

- include/bounded_fbank_bridge.h + src/bounded_fbank_bridge.cpp: caller-owned bounded output and flow control
- src/fbank_bounded_dump.cpp: bounded PCM reader, one-frame output, cleanup on invalid input, no accidental overwrite
- scripts/run_cpp_ctc_bridge.py: sends our actual features to the public offline CTC model; same-source native frontend control
- tests/test_bounded_fbank_bridge.cpp + tests/test_bounded_fbank_cli.py: random input splits, capacity 1/2/7, pause/reset, corrupt audio, exact output comparison across block sizes

The pretrained ONNX weights, tokenizer, and model design are upstream **k2-fsa/sherpa-onnx** work, not authored here. No upstream PR or model training is claimed.

## Measured result (2026-10-20)

- WSL2 Ubuntu, Debug CMake build: **3/3 CTest passed**. CLI emits byte-identical features to the original extractor with read blocks of **1, 113, and 4096 samples** on a deterministic test stream.
- The bounded custom frontend ran a real 30-utterance AISHELL-1 **Dev** corpus (442 reference characters) into the public ONNX CTC model. **30/30 final transcriptions matched the model-native frontend**. Both paths measured **2.94% CER** on these utterances (12 substitutions, 0 deletions, 1 insertion).
- Source-manifest and model SHA-256 identities, individual hypotheses and scores: build/cpp_ctc_bounded_dev30.json (ignored local evidence; regenerate via command below).

**Scope limitations:** model-native Fbank uses snip_edges=false, mel_high=7600 Hz; this custom C++ implementation uses snip_edges=true, mel_high=8000 Hz. Transcript matching is **not** numerical feature equivalence or generalization. The prior frozen 50-Test evaluation was not reopened. No microphone, real-time guarantees, streaming ASR state, AEC/NS, end-to-end latency, or production deployment has been tested.

## Reproduce using the existing WSL2 model and Python environment

```bash
cmake -S . -B build/bridge-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/bridge-debug --parallel 2
ctest --test-dir build/bridge-debug --output-on-failure
~/.venvs/voice-bench/bin/python scripts/run_cpp_ctc_bridge.py \
  --model-dir ~/.cache/voice-front-end/models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03 \
  --source-manifest data/ctc_eval/dev.jsonl --limit 30 \
  --chunk-samples 113 --output build/cpp_ctc_bounded_dev30.json
```

## Resume wording for human review (not approved)

「將自寫 C++ 音訊特徵模組接入開源中文語音辨識模型，讓它從獨立測試走到實際轉錄；加入背壓處理，讓下游處理不及時可暫停讀入，並在 30 段開發集錄音中得到與原生前端相同的辨識文字。」

This is an offline CTC integration proof; do not present it as deployed real-time ASR. Promotion depends on target JD and verifying what was authored personally versus AI-assisted or third-party code.
