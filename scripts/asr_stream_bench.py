#!/usr/bin/env python3
"""Round 2: controlled waveform->online ASR benchmark.

No claim that the standalone C++ Fbank feeds this model. Model input is native
sherpa-onnx's normalized waveform path. CER requires externally labeled text.
Audio-time events in unpaced mode are NOT end-user wall-clock latencies.
"""
import argparse
import datetime
import hashlib
import json
import platform
import time
import wave
from pathlib import Path

import numpy as np
from asr_metrics import NORMALIZATION, evaluate, corpus

MODEL_FILES = {
    "tokens": "tokens.txt",
    "encoder": "encoder-epoch-99-avg-1.int8.onnx",
    "decoder": "decoder-epoch-99-avg-1.onnx",
    "joiner": "joiner-epoch-99-avg-1.int8.onnx",
}
SAMPLE_RATE = 16000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_wav(path: Path) -> np.ndarray:
    """Use strict 16-kHz mono signed int16 WAV; don't hide resampling."""
    with wave.open(str(path), "rb") as reader:
        sample_rate = reader.getframerate()
        if reader.getnchannels() != 1 or reader.getsampwidth() != 2 or sample_rate != SAMPLE_RATE:
            raise ValueError(f"{path}: require 16kHz mono PCM16 WAV; got "
                             f"{sample_rate} Hz, channels={reader.getnchannels()}, "
                             f"width={reader.getsampwidth()}")
        if reader.getcomptype() != "NONE":
            raise ValueError(f"{path}: only uncompressed PCM accepted")
        raw = reader.readframes(reader.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def manifest_records(args) -> list[dict]:
    if args.wav:
        return [{"id": args.wav.stem, "wav": str(args.wav.resolve()),
                 "reference": args.reference, "speech_end_s": args.speech_end_s,
                 "speech_end_method": "user_supplied_unverified" if args.speech_end_s is not None else None,
                 "speech_end_status": None, "speaker_id": None,
                 "expected_wav_sha256": None, "source_split": None}]
    records = []
    seen = set()
    with args.manifest.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            name = str(row["id"])
            if not name or name in seen:
                raise ValueError(f"Duplicate/empty manifest id line {lineno}")
            seen.add(name)
            wav_path = Path(row["wav"])
            if not wav_path.is_absolute():
                wav_path = args.manifest.parent / wav_path
            reference = row.get("reference")
            if reference is not None and not str(reference).strip():
                raise ValueError(f"Missing reference for {name}; use null instead")
            end_s = row.get("speech_end_s")
            records.append({"id": name, "wav": str(wav_path.resolve()),
                            "reference": reference,
                            "speech_end_s": float(end_s) if end_s is not None else None,
                            "speech_end_method": row.get("speech_end_method") if end_s is not None else None,
                            "speech_end_status": row.get("speech_end_status"),
                            "speaker_id": row.get("speaker_id"),
                            "expected_wav_sha256": row.get("wav_sha256"),
                            "source_split": row.get("source_split")})
    if not records:
        raise ValueError("Empty manifest")
    return records


def online_model(model_dir: Path, args):
    import sherpa_onnx
    model_paths = {key: model_dir / relative for key, relative in MODEL_FILES.items()}
    missing = [str(p) for p in model_paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing model files: " + ", ".join(missing))
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        **{key: str(value) for key, value in model_paths.items()},
        num_threads=args.threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        normalize_samples=True,
        dither=0.0,
        decoding_method="greedy_search",
        enable_endpoint_detection=args.endpoint,
        rule1_min_trailing_silence=20.0,  # don't split initial silence
        rule2_min_trailing_silence=args.rule2_ms / 1000,
        rule3_min_utterance_length=120.0,
    )
    import importlib.metadata
    return recognizer, {name: sha256_file(path) for name, path in model_paths.items()}, importlib.metadata.version("sherpa-onnx")


def decode_one(recognizer, samples: np.ndarray, args) -> dict:
    stream = recognizer.create_stream()
    frame_count = len(samples)
    total_duration_s = frame_count / SAMPLE_RATE
    chunk_samples = int(args.chunk_ms * SAMPLE_RATE / 1000)
    if chunk_samples < 1:
        raise ValueError("chunk too small")
    start = time.perf_counter()
    first_partial = None
    partial_at_audio_s = None
    last_text = ""
    endpoint_audio_s = None
    endpoint_wall_s = None
    chunks = decoder_calls = 0
    compute_wall_s = 0.0
    consumed = 0

    def accept(chunk: np.ndarray):
        nonlocal consumed, first_partial, partial_at_audio_s, last_text
        nonlocal endpoint_audio_s, endpoint_wall_s, decoder_calls, compute_wall_s, chunks
        if args.paced:
            # Do not hand the recognizer samples from the future. A live
            # capture callback only has the block after its last sample has
            # arrived. Schedule to the block END, not its start.
            deadline = start + (consumed + len(chunk)) / SAMPLE_RATE
            sleep_for = deadline - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
        t0 = time.perf_counter()
        stream.accept_waveform(SAMPLE_RATE, chunk)
        consumed += len(chunk)
        chunks += 1
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
            decoder_calls += 1
            hypothesis = recognizer.get_result_all(stream).text
            if hypothesis != last_text:
                last_text = hypothesis
            if hypothesis and first_partial is None:
                first_partial = time.perf_counter() - start
                partial_at_audio_s = consumed / SAMPLE_RATE
        if args.endpoint and recognizer.is_endpoint(stream):
            endpoint_audio_s = consumed / SAMPLE_RATE
            endpoint_wall_s = time.perf_counter() - start
        compute_wall_s += time.perf_counter() - t0
        return endpoint_audio_s is not None

    for start_index in range(0, frame_count, chunk_samples):
        chunk = samples[start_index:start_index + chunk_samples]
        if accept(chunk):
            break

    # An utterance may not end with enough silence to emit all final tokens.
    # Supplement controlled tail silence, separately accounted from WAV duration.
    if endpoint_audio_s is None:
        pad_samples = int(args.tail_ms * SAMPLE_RATE / 1000)
        for start_index in range(0, pad_samples, chunk_samples):
            chunk = np.zeros(min(chunk_samples, pad_samples - start_index), dtype=np.float32)
            if accept(chunk):
                break

    t0 = time.perf_counter()
    if endpoint_audio_s is None:
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
            decoder_calls += 1
    final_text = recognizer.get_result_all(stream).text
    compute_wall_s += time.perf_counter() - t0
    duration_wall_s = time.perf_counter() - start

    return {
        "hypothesis": final_text,
        "audio_duration_s": total_duration_s,
        "audio_consumed_s": consumed / SAMPLE_RATE,
        "tail_added_s": max(0.0, consumed / SAMPLE_RATE - total_duration_s),
        "decoder_calls": decoder_calls,
        "chunks_accepted": chunks,
        "first_nonempty_partial_wall_s": first_partial,
        "first_nonempty_partial_audio_time_s": partial_at_audio_s,
        "endpoint_detected": endpoint_audio_s is not None,
        "endpoint_audio_time_s": endpoint_audio_s,
        "endpoint_wall_time_s": endpoint_wall_s if args.paced else None,
        "decode_and_accept_compute_wall_s": compute_wall_s,
        "processing_rtf_vs_original_audio": (compute_wall_s / total_duration_s if total_duration_s else None),
        "replay_elapsed_wall_s": duration_wall_s,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="JSONL rows: id, wav, optional reference, optional speech_end_s")
    source.add_argument("--wav", type=Path, help="Single 16kHz mono PCM16 WAV, unlabeled unless --reference supplied")
    parser.add_argument("--reference", help="Manually verified transcription of --wav, never ASR-generated pseudo-label")
    parser.add_argument("--speech-end-s", type=float, help="Ground-truth speech end timestamp, only with --wav")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--tail-ms", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--endpoint", action="store_true")
    parser.add_argument("--rule2-ms", type=int, default=800)
    parser.add_argument("--paced", action="store_true", help="Schedule input at audio rate; not a physical microphone")
    parser.add_argument("--output", type=Path, default=Path("results/asr_baseline.json"))
    args = parser.parse_args()
    if args.chunk_ms < 1 or args.tail_ms < 0 or args.threads < 1 or args.rule2_ms < 1:
        parser.error("chunk_ms / threads / rule2_ms must be positive; tail_ms must be nonnegative")
    if args.wav is None and (args.reference is not None or args.speech_end_s is not None):
        parser.error("--reference and --speech-end-s work only with --wav")
    if args.reference is not None and not args.reference.strip():
        parser.error("reference cannot be blank")
    records = manifest_records(args)
    recognizer, hashes, version = online_model(args.model_dir, args)
    output = []
    scored = []
    for row in records:
        path = Path(row["wav"])
        audio = read_wav(path)
        observed_sha = sha256_file(path)
        if row["expected_wav_sha256"] and observed_sha != row["expected_wav_sha256"]:
            raise ValueError(f"Waveform SHA-256 mismatch for {row['id']}; dataset changed")
        if row["speech_end_s"] is not None and not (0 <= row["speech_end_s"] <= len(audio) / SAMPLE_RATE):
            raise ValueError(f"Invalid speech_end_s for {row['id']}")
        measured = decode_one(recognizer, audio, args)
        counts = None
        if row["reference"] is not None:
            counts = evaluate(str(row["reference"]), measured["hypothesis"])
            scored.append(counts)
        if row["speech_end_s"] is not None and measured["endpoint_audio_time_s"] is not None:
            measured["endpoint_delay_from_speech_end_reference_s"] = (
                measured["endpoint_audio_time_s"] - row["speech_end_s"]
            )
            measured["endpoint_before_speech_end_reference"] = (
                measured["endpoint_audio_time_s"] < row["speech_end_s"]
            )
        else:
            measured["endpoint_delay_from_speech_end_reference_s"] = None
            measured["endpoint_before_speech_end_reference"] = None
        result = {
            "id": row["id"], "wav": str(path), "wav_sha256": observed_sha,
            "reference": row["reference"],
            "speech_end_s": row["speech_end_s"],
            "speech_end_method": row["speech_end_method"],
            "speech_end_status": row["speech_end_status"],
            "speaker_id": row["speaker_id"], "source_split": row["source_split"],
            **measured, "cer": counts.cer if counts else None,
            "s": counts.s if counts else None, "d": counts.d if counts else None,
            "i": counts.i if counts else None, "reference_chars": counts.n if counts else None,
        }
        output.append(result)
        print(f"{row['id']}: {measured['hypothesis']!r} "
              f"(CER={result['cer']}, endpoint={result['endpoint_detected']})")
    final = {
        "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": {"type": "sherpa-onnx OnlineRecognizer transducer",
                  "name": args.model_dir.name, "file_sha256": hashes, "sherpa_onnx_version": version},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "machine": platform.machine(), "numpy": np.__version__},
        "config": {"chunk_ms": args.chunk_ms, "tail_ms": args.tail_ms, "threads": args.threads,
                   "endpoint_enabled": args.endpoint, "rule2_trailing_silence_ms": args.rule2_ms,
                   "replay": "paced_file_replay" if args.paced else "as_fast_as_possible",
                   "cer_normalization": NORMALIZATION},
        "timing_note": "Unpaced wall-time is not user-facing latency. Endpoint audio_time is the stream-time including synthetic tail. Report measured decode wall only as a machine-specific proxy. Source file has no real-time capture clocks.",
        "cer_summary": corpus(scored),
        "manifest_sha256": sha256_file(args.manifest) if args.manifest is not None else None,
        "utterances": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(output)} utterances -> {args.output}, scored: {len(scored)}, "
          f"corpus CER={final['cer_summary']['cer']}")


if __name__ == "__main__":
    main()
