#!/usr/bin/env python3
"""Freeze a *separate* AISHELL-1 CTC evaluation corpus (30 dev / 50 held-out test).

Reuses the already-audited source/mirror/transcript functions from the endpoint
benchmark, without editing or consuming its 30 historical evaluation records.
Does not run ASR or report held-out test CER.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from asr_metrics import normalize
from fetch_aishell_eval import balanced, download, load_transcript

SEED = 20261012
DEV_COUNT = 30
TEST_COUNT = 50
SAMPLE_RATE = 16000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_split(rows, *, dev=DEV_COUNT, test=TEST_COUNT, seed=SEED):
    """Stable, utterance-disjoint split. Does not claim speaker-disjointness."""
    if len(rows) != dev + test:
        raise ValueError("Incorrect number of rows to freeze")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate utterance IDs")
    shuffled = sorted(rows, key=lambda row: row["id"])
    random.Random(seed).shuffle(shuffled)
    dev_rows = shuffled[:dev]
    test_rows = shuffled[dev:]
    if set(row["id"] for row in dev_rows) & set(row["id"] for row in test_rows):
        raise AssertionError("Dev/test overlap")
    return (
        [dict(row, split="dev") for row in dev_rows],
        [dict(row, split="test") for row in test_rows],
    )


def validate_wav(path):
    with wave.open(str(path), "rb") as fp:
        contract = (fp.getnchannels(), fp.getsampwidth(), fp.getframerate(), fp.getcomptype())
        duration = fp.getnframes() / SAMPLE_RATE
        if contract != (1, 2, SAMPLE_RATE, "NONE") or not 1.8 <= duration <= 10.0:
            raise ValueError(f"Invalid 16-kHz mono PCM16 WAV or duration: {path}")
    return round(duration, 6)


def acquire(output_root, *, count=80, seed=SEED, pages=16, max_per_speaker=20,
            exclude_manifest=None):
    if count != DEV_COUNT + TEST_COUNT:
        raise ValueError("Requires exactly 30 dev + 50 test utterances")
    output_root = Path(output_root).resolve()
    if output_root.exists() and any((output_root / name).exists() for name in
                                    ("manifest.jsonl", "dev.jsonl", "test.jsonl")):
        raise FileExistsError("Dataset already frozen; refuse overwrite or re-selection")
    exclude = set()
    if exclude_manifest:
        for line in Path(exclude_manifest).read_text(encoding="utf-8").splitlines():
            if line.strip():
                exclude.add(json.loads(line)["id"])
    official, transcript_hash = load_transcript(Path.home() / ".cache/voice-front-end/datasets/aishell1")
    candidates, total_rows, offsets = balanced(count + min(len(exclude), 30),
                                                 seed, pages, max_per_speaker)
    eligible = [item for item in candidates
                if item["id"] not in exclude
                and item["id"] in official
                and normalize(official[item["id"]]) == normalize(item["text"])]
    if len(eligible) < count:
        raise RuntimeError(f"Not enough eligible, source-matched and non-overlapping candidates: {len(eligible)}")
    audio_dir = output_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Acquire more than needed as a bounded reserve. A failed/malformed WAV
    # is rejected, not silently counted.
    reserve = eligible[:min(len(eligible), count + 35)]
    def fetch_one(candidate):
        path = audio_dir / (candidate["id"] + ".wav")
        if not path.exists():
            path.write_bytes(download(candidate["src"]))
        return candidate, path

    rows, rejected = [], []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for candidate, path in pool.map(fetch_one, reserve):
            try:
                duration = validate_wav(path)
                if not path.is_file():
                    raise FileNotFoundError(path)
                rows.append(dict(
                    id=candidate["id"], wav="audio/" + path.name,
                    reference=official[candidate["id"]], wav_sha256=sha256(path),
                    duration_s=duration, speaker_id=candidate["speaker"],
                    source_split="AISHELL-1/test", source_row_idx=candidate["row_idx"],
                    source_wavpath=candidate["wavpath"],
                    transcript_source="AISHELL/AISHELL-1 aishell_transcript_v0.8.txt",
                    transcript_crosscheck="normalized_exact_match",
                ))
                if len(rows) >= count:
                    break
            except (ValueError, OSError) as exc:
                rejected.append(dict(id=candidate["id"], error=str(exc)))
    if len(rows) != count:
        raise RuntimeError(f"Only {len(rows)} valid WAVs / {count}, rejected={rejected}")

    dev_rows, test_rows = freeze_split(rows, seed=seed)
    if exclude.intersection({x["id"] for x in dev_rows + test_rows}):
        raise AssertionError("Overlaps pre-existing endpoint corpus")
    speaker_counts = dict(Counter(row["speaker_id"] for row in rows))
    if len(speaker_counts) < 5:
        raise ValueError("Too few distinct speakers")
    all_rows = dev_rows + test_rows
    for filename, section in (("manifest.jsonl", all_rows),
                              ("dev.jsonl", dev_rows), ("test.jsonl", test_rows)):
        (output_root / filename).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in section), encoding="utf-8")
    provenance = dict(
        project="CTC-Decoding-Bench", corpus="AISHELL-1/test",
        note="Utterance-disjoint dev/test; speakers may overlap, training overlap not established",
        selection_seed=seed, split_seed=seed, dev_count=len(dev_rows), test_count=len(test_rows),
        source_mirror="TwinkStart/AISHELL-1", mirror_revision="2a509ef",
        original_transcript_sha256=transcript_hash,
        source_test_rows=total_rows, page_offsets=offsets,
        prior_endpoint_manifest_sha256=sha256(exclude_manifest) if exclude_manifest else None,
        excluded_prior_ids_count=len(exclude), speaker_counts=speaker_counts,
        dev_speakers=sorted(set(row["speaker_id"] for row in dev_rows)),
        test_speakers=sorted(set(row["speaker_id"] for row in test_rows)),
        rejected=rejected,
        dev_manifest_sha256=sha256(output_root / "dev.jsonl"),
        test_manifest_sha256=sha256(output_root / "test.jsonl"),
        complete_manifest_sha256=sha256(output_root / "manifest.jsonl"),
    )
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/ctc_eval"))
    parser.add_argument("--exclude-manifest", type=Path,
                        default=Path("data/round2_endpoint_eval/manifest.jsonl"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pages", type=int, default=16)
    parser.add_argument("--max-per-speaker", type=int, default=20)
    args = parser.parse_args()
    result = acquire(args.output_root, seed=args.seed, pages=args.pages,
                     max_per_speaker=args.max_per_speaker,
                     exclude_manifest=args.exclude_manifest)
    print(json.dumps(dict(dev_count=result["dev_count"], test_count=result["test_count"],
                          manifest_hash=result["complete_manifest_sha256"],
                          speakers=result["speaker_counts"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
