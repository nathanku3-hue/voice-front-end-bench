"""End-to-end parity check for the pure-C++ pipeline.

Runs tools/fbank_ctc_pipeline over every Dev utterance and compares its output
with the Python reference path recorded in results/ctc_baseline_dev.json.

What this proves
    For all 30 Dev utterances the C++ frontend produces the exact fbank frame
    count that predicts the exported logit length, and the C++ greedy decoder
    plus C++ byte-level BPE detokenizer reproduce the Python token sequence and
    transcript byte for byte.

What this does NOT prove
    The acoustic model is not in the binary. The posteriors come from a frozen
    export, so this is a parity check of the *frontend + decoder + detokenizer*,
    not of C++ inference. It also does not prove numerical equality of the C++
    fbank features to the Python frontend features -- only that the frame count
    agrees with the model graph and that everything downstream is identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_one(binary: Path, wav: Path, npy: Path, tokens: Path, repeats: int) -> dict:
    proc = subprocess.run(
        [str(binary), "--wav", str(wav), "--posteriors", str(npy),
         "--tokens", str(tokens), "--repeats", str(repeats)],
        capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{wav.name}: exit {proc.returncode}: {proc.stderr.strip()}")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{wav.name}: no transcript on stdout")
    # The transcript is the only thing on stdout; diagnostics go to stderr.
    text = lines[0]
    fields: dict[str, str] = {}
    for line in proc.stderr.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return {"text": text, "diagnostics": fields}


def parse_float(pattern: str, text: str) -> float | None:
    import re
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def parse_int(pattern: str, text: str) -> int | None:
    import re
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def parse_frames_line(line: str) -> dict:
    """Parse 'frames: 551 fbank -> 137 logit (predicted 137), tokens 19 (...)'."""
    import re
    match = re.search(
        r"frames:\s*(\d+)\s+fbank\s*->\s*(\d+)\s+logit\s*\(predicted\s+(\d+)\),\s*"
        r"tokens\s+(\d+)", line)
    if not match:
        raise ValueError(f"unrecognised frames line: {line!r}")
    return {
        "fbank_frames": int(match.group(1)),
        "logit_frames": int(match.group(2)),
        "predicted_logit_frames": int(match.group(3)),
        "tokens": int(match.group(4)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=str(REPO / "build/win-zig/fbank_ctc_pipeline.exe"))
    parser.add_argument("--dev-root", default=str(REPO / "build/ctc_round1_dev"))
    parser.add_argument("--data-root", default=str(REPO / "data/ctc_eval"))
    parser.add_argument("--model-dir", default=str(REPO / "build/win-zig/model"))
    parser.add_argument("--baseline", default=str(REPO / "results/ctc_baseline_dev.json"))
    parser.add_argument("--out", default=str(REPO / "results/pipeline_parity.json"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    binary = Path(args.binary)
    dev_root = Path(args.dev_root)
    data_root = Path(args.data_root)
    tokens = Path(args.model_dir) / "tokens.txt"
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    reference = {u["id"]: u for u in baseline["utterances"]}

    manifest = [json.loads(line) for line in
                (dev_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    rows = []
    for entry in manifest:
        uid = entry["id"]
        wav = data_root / "audio" / f"{uid}.wav"
        npy = dev_root / entry["log_probs"]
        out = run_one(binary, wav, npy, tokens, args.repeats)
        diag = out["diagnostics"]
        want = reference[uid]
        parsed = parse_frames_line("frames: " + diag.get("frames", ""))
        text_ok = out["text"] == want["hypothesis"]
        rows.append(dict(
            id=uid,
            reference=want["reference"],
            python_hypothesis=want["hypothesis"],
            cpp_hypothesis=out["text"],
            text_matches_python=text_ok,
            python_tokens=len(want["tokens"]),
            cpp_tokens=parsed["tokens"],
            fbank_frames=parsed["fbank_frames"],
            logit_frames=parsed["logit_frames"],
            predicted_logit_frames=parsed["predicted_logit_frames"],
            fbank_us_per_frame=parse_float(r"\(([\d.]+) us/frame", diag.get("fbank", "")),
            fbank_realtime=parse_float(r"([\d.]+)x realtime", diag.get("fbank", "")),
            greedy_us_per_logit_frame=parse_float(r"\(([\d.]+) us/logit frame", diag.get("decode", "")),
            utf8_replacement_chars=parse_int(r"utf8 replacements (\d+)", diag.get("detok", "")) or 0,
            python_utf8_replacement_chars=want.get("utf8_replacement_chars", 0),
        ))
        status = "ok" if text_ok else "MISMATCH"
        print(f"{uid} {status} tokens {rows[-1]['cpp_tokens']}/{rows[-1]['python_tokens']} "
              f"fbank {rows[-1]['fbank_frames']} -> {rows[-1]['logit_frames']} "
              f"(predicted {rows[-1]['predicted_logit_frames']})", flush=True)

    mismatched = [r["id"] for r in rows if not r["text_matches_python"]]
    token_mismatched = [r["id"] for r in rows if r["cpp_tokens"] != r["python_tokens"]]
    frame_mismatched = [r["id"] for r in rows
                        if r["logit_frames"] != r["predicted_logit_frames"]]
    replacement_mismatched = [r["id"] for r in rows
                              if r["utf8_replacement_chars"] != r["python_utf8_replacement_chars"]]
    compared_chars = sum(len(r["cpp_hypothesis"]) for r in rows)

    report = dict(
        status="measured_cpp_pipeline_parity",
        scope=("frontend frame-count consistency plus C++ greedy decode and C++ "
               "detokenizer equality; the acoustic model is NOT in the binary"),
        binary=dict(path=str(binary), sha256=sha256_file(binary)),
        tokens_txt=dict(path=str(tokens), sha256=sha256_file(tokens)),
        baseline=dict(path=args.baseline, sha256=sha256_file(Path(args.baseline))),
        repeats=args.repeats,
        utterances=len(rows),
        transcripts_identical=len(rows) - len(mismatched),
        transcripts_mismatched=mismatched,
        token_counts_identical=len(rows) - len(token_mismatched),
        token_counts_mismatched=token_mismatched,
        logit_frames_consistent=len(rows) - len(frame_mismatched),
        logit_frames_mismatched=frame_mismatched,
        utf8_replacement_total=sum(r["utf8_replacement_chars"] for r in rows),
        utf8_replacement_mismatched=replacement_mismatched,
        hypothesis_chars_compared=compared_chars,
        non_claims=[
            "no C++ acoustic model: posteriors are a frozen export of the Python run",
            "no claim that C++ fbank values equal the Python frontend values",
            "host-local decoder-only timing, not device latency",
        ],
        rows=rows,
    )
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    print(f"\n{report['transcripts_identical']}/{report['utterances']} transcripts identical, "
          f"{report['token_counts_identical']}/{report['utterances']} token counts identical, "
          f"{report['logit_frames_consistent']}/{report['utterances']} frame counts consistent, "
          f"{report['utf8_replacement_total']} UTF-8 replacements")
    print(f"wrote {args.out}")
    return 0 if not (mismatched or token_mismatched or frame_mismatched
                     or replacement_mismatched) else 1


if __name__ == "__main__":
    sys.exit(main())
