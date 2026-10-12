#!/usr/bin/env python3
"""Verify independently re-fetched Round 2 manifest against committed source custody.

No ASR model needed. Fails if IDs, ordered rows, references, source identity or
WAV checksums changed. Also checks candidate WAV bytes against claimed SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FIELDS = (
    "id", "speaker_id", "source_split", "source_row_idx",
    "source_wavpath", "reference", "wav_sha256",
)


def load_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"Manifest missing: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid JSONL manifest {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate utterance id in {path}")
    for row in rows:
        for field in FIELDS:
            if not isinstance(row.get(field), (str, int)):
                raise ValueError(f"Missing/invalid {field} in {path}")
    return rows


def verify(reference: Path, candidate: Path, check_wav: bool = True) -> int:
    original = load_manifest(reference)
    replay = load_manifest(candidate)
    if len(original) != len(replay):
        raise ValueError(f"Row count differs: {len(original)} vs {len(replay)}")
    for index, (a, b) in enumerate(zip(original, replay), 1):
        for name in FIELDS:
            if a[name] != b[name]:
                raise ValueError(f"Row {index} ({a['id']}): {name} mismatch")
        if check_wav:
            relative_path = Path(str(b.get("wav", "")))
            if (not relative_path.parts or relative_path.is_absolute()
                    or ".." in relative_path.parts):
                raise ValueError(f"Unsafe/missing relative WAV path for {b['id']}")
            wav = (candidate.parent / relative_path).resolve()
            if not wav.is_relative_to(candidate.parent.resolve()) or not wav.is_file():
                raise ValueError(f"Missing local replay WAV: {wav}")
            checksum = hashlib.sha256(wav.read_bytes()).hexdigest()
            if checksum != b["wav_sha256"]:
                raise ValueError(f"WAV checksum mismatch for {b['id']}")
    return len(original)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
                        default=Path("data/round2_endpoint_eval/manifest.jsonl"))
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = verify(args.reference, args.candidate)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"FAIL source custody: {exc}\n")
    print(f"PASS: {count} ordered utterances match source/label/WAV SHA-256 custody")


if __name__ == "__main__":
    main()
