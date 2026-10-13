#!/usr/bin/env python3
"""Export real offline Zipformer-CTC log probabilities on verified 16-kHz WAVs.

Example (WSL, from repository root):
  python scripts/ctc_export_logits.py \
    --model-dir ~/.cache/voice-front-end/models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03 \
    --source-manifest data/round2_endpoint_eval/manifest.jsonl \
    --limit 1 --output-root build/ctc_pilot

This is a CTC model's independent input path; it does not inject the C++
StreamingFbank and does not reuse the Transducer model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import wave
from pathlib import Path

import numpy as np
from ctc_greedy_baseline import validate_log_probs
from ctc_tokenization import load_tokens

SAMPLE_RATE = 16000
FBANK = dict(sample_frequency=16000.0, num_mel_bins=80, frame_length=25.0,
             frame_shift=10.0, low_freq=20.0, high_freq=-400.0,
             dither=0.0, snip_edges=False, preemphasis_coefficient=0.97,
             remove_dc_offset=True, window_type="povey", use_power=True,
             use_log_fbank=True, use_energy=False, subtract_mean=False,
             round_to_power_of_two=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_wav(path):
    with wave.open(str(path), "rb") as fp:
        if (fp.getnchannels(), fp.getsampwidth(), fp.getframerate(), fp.getcomptype()) != (
                1, 2, SAMPLE_RATE, "NONE"):
            raise ValueError(f"Require 16-kHz mono PCM16 WAV: {path}")
        samples = np.frombuffer(fp.readframes(fp.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if len(samples) < SAMPLE_RATE // 10:
        raise ValueError("WAV too short for CTC inference")
    return samples


def extract_features(samples):
    import torch
    import torchaudio.compliance.kaldi as kaldi
    waveform = torch.from_numpy(np.asarray(samples, dtype=np.float32).copy()).unsqueeze(0)
    features = kaldi.fbank(waveform, **FBANK).numpy()
    if features.ndim != 2 or features.shape[1] != 80 or not np.all(np.isfinite(features)):
        raise ValueError("Invalid fbank features")
    return np.ascontiguousarray(features, dtype=np.float32)


def open_ctc_session(model_file):
    import onnxruntime as ort
    config = ort.SessionOptions()
    config.intra_op_num_threads = 2
    session = ort.InferenceSession(str(model_file), sess_options=config,
                                   providers=["CPUExecutionProvider"])
    inputs = {i.name: i for i in session.get_inputs()}
    outputs = {i.name: i for i in session.get_outputs()}
    if (set(inputs) != {"x", "x_lens"} or
        inputs["x"].type != "tensor(float)" or
        inputs["x_lens"].type != "tensor(int64)" or
        len(inputs["x"].shape) != 3 or inputs["x"].shape[2] != 80 or
        not {"log_probs", "log_probs_len"}.issubset(outputs)):
        raise ValueError(f"Unexpected ONNX CTC contract: {inputs}, {outputs}")
    if session.get_modelmeta().custom_metadata_map.get("model_type") != "zipformer2_ctc":
        raise ValueError("ONNX file is not identified as zipformer2_ctc")
    return session


def forward_log_probs(session, features, vocab_size):
    x = np.expand_dims(features, 0)
    length = np.asarray([len(features)], dtype=np.int64)
    scores, lengths = session.run(["log_probs", "log_probs_len"],
                                  {"x": x, "x_lens": length})
    scores = np.asarray(scores)
    lengths = np.asarray(lengths)
    if scores.ndim != 3 or scores.shape[0] != 1 or scores.shape[2] != vocab_size:
        raise ValueError(f"Unexpected CTC shape: {scores.shape} for V={vocab_size}")
    if lengths.shape != (1,) or int(lengths[0]) <= 0 or int(lengths[0]) > scores.shape[1]:
        raise ValueError(f"Bad returned output length: {lengths}; logits={scores.shape}")
    out = np.ascontiguousarray(scores[0, :int(lengths[0]), :], dtype=np.float32)
    validate_log_probs(out, vocab_size)  # output is already LogSoftmax; reject, do not alter
    return out, dict(features=len(features), raw_ctc_frames=int(scores.shape[1]),
                     valid_ctc_frames=int(lengths[0]), vocab_size=vocab_size)


def checked_source_rows(source_manifest, limit):
    source_manifest = Path(source_manifest).resolve()
    result, seen = [], set()
    for line in source_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        uid = str(item["id"])
        if uid in seen:
            raise ValueError("Duplicate source ID")
        seen.add(uid)
        path = (source_manifest.parent / item["wav"]).resolve()
        if not path.is_relative_to(source_manifest.parent) or not path.is_file():
            raise ValueError(f"Missing/outside source WAV: {item['wav']}")
        expected_hash = item.get("wav_sha256")
        if not expected_hash or sha256(path) != expected_hash:
            raise ValueError(f"Source WAV hash mismatch: {uid}")
        if not item.get("reference"):
            raise ValueError(f"Missing independent reference for {uid}")
        result.append((item, path))
        if len(result) == limit:
            break
    if len(result) < limit:
        raise ValueError(f"Requested {limit} but found {len(result)} valid source entries")
    return result


def export(model_dir, source_manifest, output_root, limit=1):
    model_dir = Path(model_dir).resolve()
    source_manifest = Path(source_manifest).resolve()
    output_root = Path(output_root).resolve()
    model_file = model_dir / "model.int8.onnx"
    token_file = model_dir / "tokens.txt"
    bpe_file = model_dir / "bbpe.model"
    for p in (model_file, token_file, bpe_file):
        if not p.is_file():
            raise FileNotFoundError(p)
    tokens = load_tokens(token_file)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(bpe_file))
    if sp.get_piece_size() != len(tokens) or any(sp.id_to_piece(i) != t for i, t in enumerate(tokens)):
        raise ValueError("Byte-BPE model and token table disagree")
    if tokens[0] != "<blk>":
        raise ValueError("Blank ID is not verified as zero")
    rows = checked_source_rows(source_manifest, limit)
    session = open_ctc_session(model_file)
    output_root.mkdir(parents=True, exist_ok=True)
    logits_dir = output_root / "log_probs"
    logits_dir.mkdir(exist_ok=True)
    artifacts = []
    diagnostics = []
    for item, audio_path in rows:
        samples = read_wav(audio_path)
        features = extract_features(samples)
        log_probs, shapes = forward_log_probs(session, features, len(tokens))
        target = logits_dir / (item["id"] + ".npy")
        np.save(target, log_probs, allow_pickle=False)
        artifacts.append(dict(id=item["id"], reference=item["reference"],
                              log_probs=str(target.relative_to(output_root)),
                              wav_sha256=item["wav_sha256"],
                              speaker_id=item.get("speaker_id"), source_split=item.get("source_split")))
        diagnostics.append(dict(id=item["id"], **shapes, log_probs_sha256=sha256(target),
                                audio_sha256=item["wav_sha256"],
                                source_wav=str(audio_path)))
        print(json.dumps(dict(id=item["id"], **shapes), ensure_ascii=False), flush=True)
    model_meta = dict(model_name=model_dir.name,
                      model_sha256=sha256(model_file),
                      model_archive_source="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2",
                      token_table_sha256=sha256(token_file),
                      bbpe_model_sha256=sha256(bpe_file),
                      ctc_head="log_probs",
                      output_length_head="log_probs_len",
                      blank_id=0, tokenizer="bbpe",
                      frame_stride_ms=40,
                      frame_stride_basis="nominal Zipformer CTC 4x subsampling of 10ms Fbank, not real-time latency",
                      frontend="torchaudio.compliance.kaldi.fbank",
                      frontend_config=FBANK)
    (output_root / "model.json").write_text(json.dumps(model_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in artifacts), encoding="utf-8")
    summary = dict(status="real_ctc_logits_exported", num_utterances=len(artifacts),
                   source_manifest_sha256=sha256(source_manifest),
                   model_meta=model_meta, diagnostics=diagnostics,
                   note="No claims of acoustic frontend parity, test CER or product latency")
    (output_root / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("build/ctc_pilot"))
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        parser.error("Require --limit 1..100")
    report = export(args.model_dir, args.source_manifest, args.output_root, args.limit)
    print(json.dumps(dict(artifacts=report["num_utterances"], output=str(args.output_root.resolve())),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
