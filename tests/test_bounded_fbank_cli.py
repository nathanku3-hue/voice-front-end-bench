#!/usr/bin/env python3
"""Compare bounded and legacy feature output; fail closed on damaged PCM."""
import argparse
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def call(executable, raw, destination, chunk):
    return subprocess.run([str(executable), str(raw), str(destination), str(chunk)],
                          capture_output=True, text=True, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--bounded", type=Path, required=True)
    args = parser.parse_args()
    for exe in (args.baseline, args.bounded):
        if not exe.is_file():
            raise SystemExit(f"missing compiled binary: {exe}")

    with tempfile.TemporaryDirectory(prefix="bounded_fbank_ctest_") as directory:
        folder = Path(directory)
        pcm = folder / "clean.raw"
        samples = [0.35 * math.sin(i * 0.019) + 0.1 * math.cos(i * 0.087)
                   for i in range(40000)]
        pcm.write_bytes(b"".join(struct.pack("<f", x) for x in samples))
        baseline_csv = folder / "reference.csv"
        original = call(args.baseline, pcm, baseline_csv, 253)
        if original.returncode != 0:
            raise AssertionError("baseline failed: " + original.stderr)
        expected = baseline_csv.read_bytes()
        if len(expected.splitlines()) != (len(samples) - 400) // 160 + 1:
            raise AssertionError("wrong expected frame count")
        for chunk in (1, 113, 4096):
            csv = folder / f"bounded_{chunk}.csv"
            result = call(args.bounded, pcm, csv, chunk)
            if result.returncode != 0:
                raise AssertionError(f"bounded chunk={chunk}: {result.stderr}")
            if csv.read_bytes() != expected:
                raise AssertionError(f"feature mismatch at chunk={chunk}")

        malformed = folder / "bad_length.raw"
        malformed.write_bytes(pcm.read_bytes() + b"x")
        damaged_pcm = folder / "nonfinite.raw"
        corrupted = samples[:2000]
        corrupted[-20] = float("nan")
        damaged_pcm.write_bytes(b"".join(struct.pack("<f", v) for v in corrupted))
        for label, invalid in (("bad_length", malformed), ("nonfinite", damaged_pcm)):
            target = folder / f"{label}.csv"
            result = call(args.bounded, invalid, target, 1024)
            if result.returncode == 0 or target.exists() or Path(str(target) + ".part").exists():
                raise AssertionError(f"damaged PCM left a success-looking artifact: {label}")

        existing = folder / "existing.csv"
        existing.write_text("protected", encoding="utf-8")
        result = call(args.bounded, pcm, existing, 113)
        if result.returncode == 0 or existing.read_text() != "protected":
            raise AssertionError("existing result was overwritten")

    print("bounded CLI PASS: byte-identical features under three chunk sizes; "
          "partial output cleanup and no overwrite on failure")


if __name__ == "__main__":
    main()
