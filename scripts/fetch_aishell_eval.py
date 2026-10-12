#!/usr/bin/env python3
"""Acquire 20-30 AISHELL-1 TEST WAVs via Hugging Face viewer /rows.

Every mirror transcript is checked against the official AISHELL transcript.
The endpoint is a peak-relative RMS ESTIMATE, not human-annotated ground truth.
"""
import argparse
import hashlib
import json
import random
import re
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from asr_metrics import normalize
from endpoint_vad import estimate_speech_end_details, read_pcm16

API = "https://datasets-server.huggingface.co/rows"
TRANS_URL = ("https://huggingface.co/datasets/AISHELL/AISHELL-1/resolve/main/"
             "data_aishell/transcript/aishell_transcript_v0.8.txt")
UID = re.compile(r"^BAC[0-9A-Z]{12,}$")


def download(url):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in {"huggingface.co", "datasets-server.huggingface.co"}:
        raise ValueError("Unexpected host for dataset URL")
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VoiceFrontendBench/1.0"})
            with urllib.request.urlopen(req, timeout=45) as f:
                return f.read()
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def viewer(offset):
    query = urllib.parse.urlencode(dict(dataset="TwinkStart/AISHELL-1", config="default",
                                        split="test", offset=offset, length=100))
    output = json.loads(download(API + "?" + query))
    if output.get("num_rows_total", 0) < 7000:
        raise ValueError("Unexpected split/row count")
    return output


def load_transcript(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    file = cache_dir / "aishell_transcript_v0.8.txt"
    if not file.exists():
        data = download(TRANS_URL)
        if len(data) < 1_000_000:
            raise ValueError("Transcript download unexpectedly short")
        file.write_bytes(data)
    raw = file.read_bytes()
    transcripts = {}
    for line in raw.decode("utf-8-sig").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2:
            transcripts[fields[0]] = fields[1]
    if len(transcripts) < 10000:
        raise ValueError("Official transcript contains too few lines")
    return transcripts, hashlib.sha256(raw).hexdigest()


def balanced(count, seed, pages, max_per_speaker):
    first = viewer(0)
    total = int(first["num_rows_total"])
    offsets = sorted(set(round(i * (total - 100) / (pages - 1)) for i in range(pages)))
    speakers = defaultdict(list)
    other_offsets = [offset for offset in offsets if offset]
    responses = {0: first}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for offset, data in zip(other_offsets, executor.map(viewer, other_offsets)):
            responses[offset] = data
    for offset in offsets:
        results = responses[offset]
        for record in results["rows"]:
            item = record["row"]
            uid = item.get("name", "")
            wavpath = item.get("WavPath", "")
            sections = Path(wavpath).parts
            audio = item.get("audio")
            if (not UID.fullmatch(uid) or len(sections) != 3 or sections[0] != "test" or
                sections[-1] != uid + ".wav" or not isinstance(audio, list) or
                len(audio) != 1 or not audio[0].get("src")):
                continue
            speakers[sections[1]].append({
                "id": uid, "speaker": sections[1], "wavpath": wavpath,
                "text": item.get("text", ""), "row_idx": record["row_idx"],
                "src": audio[0]["src"],
            })
    rng = random.Random(seed)
    for arr in speakers.values():
        rng.shuffle(arr)
    seen, result = set(), []
    for i in range(max_per_speaker):
        for speaker in sorted(speakers):
            if i < len(speakers[speaker]):
                item = speakers[speaker][i]
                if item["id"] not in seen:
                    result.append(item)
                    seen.add(item["id"])
    if len(result) < count:
        raise ValueError(f"Too few balanced candidates ({len(result)} for {count})")
    return result, total, offsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data/round2_endpoint_eval"))
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/voice-front-end/datasets/aishell1")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--pages", type=int, default=10)
    parser.add_argument("--max-per-speaker", type=int, default=5)
    args = parser.parse_args()
    if not (20 <= args.count <= 30 and args.pages >= 3 and args.max_per_speaker >= 1):
        parser.error("Require count 20–30, pages >=3 and max-per-speaker >=1")
    trans, transcript_sha = load_transcript(args.cache)
    print(f"Official transcripts={len(trans)} sha256={transcript_sha}", flush=True)
    candidates, total, offsets = balanced(args.count, args.seed, args.pages, args.max_per_speaker)
    print(f"Viewer test rows={total}, eligible candidate rows={len(candidates)}", flush=True)
    base = args.output_root.resolve()
    (base / "audio").mkdir(parents=True, exist_ok=True)
    # Download a bounded reserve concurrently, but preserve candidate ordering
    # for deterministic selection and audit the exact accepted hashes.
    def prefetch(candidate):
        wav = base / "audio" / (candidate["id"] + ".wav")
        if wav.exists():
            return
        try:
            blob = download(candidate["src"])
            if len(blob) >= 16000 and blob[:4] == b"RIFF" and blob[8:12] == b"WAVE":
                wav.write_bytes(blob)
        except (OSError, ValueError):
            pass  # Main validation loop retries/report failures in detail.

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(prefetch, candidates[:min(len(candidates), args.count + 12)]))
    selected, rejected = [], []
    for candidate in candidates:
        if len(selected) >= args.count:
            break
        uid = candidate["id"]
        reference = trans.get(uid)
        if reference is None or normalize(reference) != normalize(candidate["text"]):
            rejected.append({"id": uid, "reason": "official_transcript_mismatch_or_missing"})
            continue
        wav = base / "audio" / f"{uid}.wav"
        try:
            if not wav.is_file():
                blob = download(candidate["src"])
                if len(blob) < 16000 or blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
                    raise ValueError("Invalid or short RIFF WAVE")
                wav.write_bytes(blob)
            signal = read_pcm16(wav)
            duration = len(signal) / 16000
            if not 1.8 <= duration <= 10.0:
                raise ValueError(f"Duration {duration:.3f} outside 1.8–10 seconds")
            vad = estimate_speech_end_details(wav)
            if vad["speech_end_s"] is None:
                raise ValueError("Cannot estimate speech endpoint")
            selected.append({
                "id": uid, "wav": f"audio/{uid}.wav", "reference": reference,
                "speech_end_s": vad["speech_end_s"],
                "speech_end_method": vad["speech_end_method"],
                "speech_end_status": vad["status"],
                "peak_rms_dbfs": vad["peak_rms_dbfs"],
                "trailing_margin_s": vad["trailing_margin_s"],
                "duration_s": round(duration, 6),
                "speaker_id": candidate["speaker"],
                "source_split": "AISHELL-1/test",
                "source_row_idx": candidate["row_idx"],
                "source_wavpath": candidate["wavpath"],
                "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                "transcript_source": "AISHELL/AISHELL-1 aishell_transcript_v0.8.txt",
                "transcript_crosscheck": "normalized_exact_match",
            })
            print(f"{len(selected):02d}/{args.count} {uid} speaker={candidate['speaker']} "
                  f"duration={duration:.2f}s end={vad['speech_end_s']:.2f}s "
                  f"status={vad['status']}", flush=True)
        except (ValueError, OSError) as e:
            rejected.append({"id": uid, "reason": str(e)})
    if len(selected) != args.count:
        raise RuntimeError(f"Insufficient validated WAVs: {len(selected)} / {args.count}, rejected={rejected}")
    speaker_counts = dict(Counter(x["speaker_id"] for x in selected))
    if len(speaker_counts) < 5:
        raise RuntimeError("Fewer than 5 speakers in selected data")
    (base / "manifest.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in selected), encoding="utf-8")
    (base / "selected_ids.txt").write_text(
        "\n".join(x["id"] for x in selected) + "\n", encoding="utf-8")
    receipt = dict(source_mirror="TwinkStart/AISHELL-1", mirror_revision="2a509ef",
                   split="test", original_transcript_url=TRANS_URL,
                   original_transcript_sha256=transcript_sha,
                   seed=args.seed, pages=args.pages, page_offsets=offsets,
                   total_test_rows=total, n=len(selected), speaker_counts=speaker_counts,
                   rejected=rejected, reference_policy="original transcripts matched to mirror independently of ASR",
                   endpoint_policy="peak-relative RMS -35dB estimate; NOT human ground truth",
                   possible_training_overlap="unverified")
    (base / "provenance.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(f"Manifest ready: {base / 'manifest.jsonl'} ({len(selected)} items; "
          f"{len(speaker_counts)} speakers)", flush=True)


if __name__ == "__main__":
    main()
