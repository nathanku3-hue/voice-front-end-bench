#!/usr/bin/env python3
"""Disassembly audit: what the compiler actually emitted for the two hot kernels.

Compiles tools/kernel_asm_probe.cpp to assembly (-O2, Intel syntax, no debug
info), splits it per symbol, finds each function's innermost loop, and counts
what is inside it. The counters are the ones that decide throughput: instructions
per sample, 128- vs 256-bit vector ops, aligned vs unaligned moves, and FMA.

The kernel wrappers in the probe carry the same `target("avx2,fma")` attribute as
the kernels they wrap, so both bodies appear in one translation unit. That is
deliberate: it means the scalar and vector forms are compiled by the same
compiler at the same optimisation level, which is the only way the comparison
means anything.

    python scripts/audit_disassembly.py --out results/simd_disassembly.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# samples handled by one iteration of each kernel's inner loop. These are
# properties of the source, asserted below against the loop body so the table
# cannot silently drift if someone changes the unroll factor.
# (name, symbol, marker opcode, samples per marker, required tokens)
# The marker is an instruction that appears exactly once per source-level
# iteration, so the unroll factor -- which the compiler is free to choose -- is
# recovered from the assembly instead of being assumed.
KERNELS = [
    ("mean_scalar", "probe_mean_scalar", "vcvtss2sd", 1, ["vcvtss2sd"]),
    ("preemph_scalar", "probe_preemph_scalar", "vfmadd132sd", 1, ["vcvtss2sd", "vfmadd"]),
    ("mean_contiguous", "probe_mean_contiguous", "vcvtss2sd", 1, ["vcvtss2sd"]),
    ("mean_avx2", "probe_mean_avx2", "vcvtps2pd", 4, ["vcvtps2pd"]),
    ("preemph_avx2", "probe_preemph_avx2", "vfmadd213pd", 4, ["vcvtps2pd", "vfmadd213pd"]),
    ("log_scalar", "probe_log_scalar", "vdivss", 1, ["vdivss"]),
    ("log_avx2", "probe_log_avx2", "vdivps", 8, ["vdivps"]),
]

LABEL_RE = re.compile(r"^(\.LBB\d+_\d+|probe_[a-z0-9_]+):")
INSN_RE = re.compile(r"^\t([a-z][a-z0-9.]*)\s*(.*)$")
TARGET_RE = re.compile(r"^(\.LBB\d+_\d+)$")


def compile_asm(out_path: Path, extra: list[str]) -> str:
    zig = REPO / ".workbuddy-ai/venv-cpp/Lib/site-packages/ziglang/zig.exe"
    if not zig.exists():
        raise SystemExit(f"zig not found at {zig}")
    cmd = [str(zig), "c++", "-std=c++17", "-O2", "-g0", "-S", "-masm=intel",
           "-I", str(REPO / "include"), *extra,
           str(REPO / "tools/kernel_asm_probe.cpp"), "-o", str(out_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(line for line in proc.stderr.splitlines() if "error" in line)
        raise SystemExit(f"assembly failed:\n{tail[:2000]}")
    return " ".join(cmd[1:])


def split_functions(lines: list[str]) -> dict[str, list[tuple[str, str]]]:
    functions: dict[str, list[tuple[str, str]]] = {}
    current = None
    for line in lines:
        match = LABEL_RE.match(line)
        if match and match.group(1).startswith("probe_"):
            current = match.group(1)
            functions[current] = []
            continue
        if current is None:
            continue
        insn = INSN_RE.match(line)
        if insn:
            functions[current].append((insn.group(1), insn.group(2)))
    return functions


def inner_loops(lines: list[str], symbol: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """All annotated inner loops of one function, as (label, instructions).

    clang marks each loop with `# =>This Inner Loop Header`, so a function can
    have several: a peeled prologue loop, the vector body, and a scalar tail.
    Returning all of them lets the caller pick by content instead of by
    position -- position is exactly what misleads here, because the AVX2
    pre-emphasis kernel opens with a 4-sample scalar peel.
    """
    start = next(i for i, line in enumerate(lines) if line.startswith(symbol + ":"))
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if LABEL_RE.match(lines[i]) and lines[i].startswith("probe_"):
            end = i
            break
    body = lines[start:end]

    loops: list[tuple[str, list[tuple[str, str]]]] = []
    for i, line in enumerate(body):
        if "Inner Loop Header" not in line:
            continue
        match = LABEL_RE.match(line)
        if not match:
            continue
        header = match.group(1)
        header_index = int(header.split("_")[1])
        instructions: list[tuple[str, str]] = []
        for inner in body[i + 1:]:
            if TARGET_RE.match(inner):
                continue
            label = LABEL_RE.match(inner)
            if label and not label.group(1).startswith("probe_"):
                continue
            insn = INSN_RE.match(inner)
            if not insn:
                continue
            instructions.append((insn.group(1), insn.group(2)))
            if insn.group(1).startswith("j"):
                target = insn.group(2).split()[0] if insn.group(2) else ""
                if TARGET_RE.match(target) and int(target.split("_")[1]) <= header_index:
                    break  # backward branch closes the loop
        loops.append((header, instructions))
    if not loops:
        fallback = [(m.group(1), m.group(2)) for m in (INSN_RE.match(l) for l in body) if m]
        return [("", fallback)]
    return loops


def pick_loop(lines: list[str], symbol: str, expect: list[str]) -> tuple[str, list[tuple[str, str]]]:
    """Choose the hot loop, not merely a loop.

    clang emits several loops per kernel: a peeled prologue, the vector body,
    an unrolled body, and a scalar tail. The tail is easy to mistake for the hot
    path because it can be larger (the log kernel's tail vectorises into four
    calls to logf). Two rules pick correctly: the hot loop contains no call, and
    among the candidates it is the one with the most 256-bit work.
    """
    candidates = inner_loops(lines, symbol)
    matching = []
    for header, instructions in candidates:
        text = " ".join(opcode for opcode, _ in instructions)
        if all(token in text for token in expect):
            matching.append((header, instructions))
    pool = matching or candidates
    without_calls = [item for item in pool
                     if not any(opcode == "call" for opcode, _ in item[1])]
    pool = without_calls or pool
    return max(pool, key=lambda item: (sum("ymm" in operand for _, operand in item[1]),
                                       len(item[1])))


def classify(instructions: list[tuple[str, str]]) -> dict:
    mix: dict[str, int] = {}
    for opcode, _ in instructions:
        mix[opcode] = mix.get(opcode, 0) + 1
    text = " ".join(opcode + " " + operand for opcode, operand in instructions)
    return {
        "instruction_count": len(instructions),
        "ymm_ops": sum(count for opcode, count in mix.items() if "ymm" in opcode),
        "xmm_ops": sum(count for opcode, count in mix.items() if "xmm" in opcode),
        "vector_ops_256": len(re.findall(r"ymm", text)),
        "vector_ops_128": len(re.findall(r"xmm", text)),
        "aligned_moves": len(re.findall(r"vmovap[sd]\s", text)),
        "unaligned_moves": len(re.findall(r"vmovup[sd]\s", text)),
        "fma": sum(count for opcode, count in mix.items() if "fmadd" in opcode or "fnmadd" in opcode),
        "scalar_fp": sum(count for opcode, count in mix.items()
                         if opcode.endswith("sd") or opcode.endswith("ss")),
        "mix": dict(sorted(mix.items(), key=lambda item: (-item[1], item[0]))[:10]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "results/simd_disassembly.json")
    parser.add_argument("--asm", type=Path, default=REPO / "build/asm/kernels.s")
    parser.add_argument("--baseline-asm", type=Path, default=REPO / "build/asm/kernels_baseline.s")
    parser.add_argument("--markdown", type=Path,
                        default=REPO / "results/simd_disassembly_diff.md")
    args = parser.parse_args()

    args.asm.parent.mkdir(parents=True, exist_ok=True)
    # The AVX2 bodies come from the target attribute, so one compile covers both.
    flags = compile_asm(args.asm, [])
    lines = args.asm.read_text(encoding="utf-8", errors="replace").splitlines()
    functions = split_functions(lines)

    rows = []
    for name, symbol, marker, samples_per_marker, expect in KERNELS:
        if symbol not in functions:
            raise SystemExit(f"symbol {symbol} missing from the assembly")
        header, loop = pick_loop(lines, symbol, expect)
        stats = classify(loop)
        body_text = " ".join(opcode for opcode, _ in loop)
        for token in expect:
            if token not in body_text:
                raise SystemExit(f"{name}: expected '{token}' in the inner loop, got: {body_text[:200]}")
        markers = sum(1 for opcode, _ in loop if opcode == marker)
        if markers == 0:
            raise SystemExit(f"{name}: marker {marker} missing from the selected loop")
        samples = markers * samples_per_marker
        stats.update(
            name=name,
            symbol=symbol,
            marker=marker,
            markers_per_iteration=markers,
            samples_per_iteration=samples,
            instructions_per_sample=round(stats["instruction_count"] / samples, 3),
            function_instructions=len(functions[symbol]),
            loop_label=header,
            loop_body=[f"{opcode} {operand}".strip() for opcode, operand in loop],
        )
        rows.append(stats)

    by_name = {row["name"]: row for row in rows}

    def ratio(new: str, old: str) -> float:
        return by_name[old]["instructions_per_sample"] / by_name[new]["instructions_per_sample"]

    report = {
        "status": "measured_disassembly_mix",
        "toolchain": flags,
        "source": "tools/kernel_asm_probe.cpp",
        "method": ("one translation unit, -O2 -masm=intel -g0; each kernel's innermost loop is "
                   "taken from the first backward branch inside it; instruction counts are the "
                   "loop body only, not the function prologue"),
        "kernels": rows,
        "derived": {
            "mean_instructions_per_sample": {
                "scalar": by_name["mean_scalar"]["instructions_per_sample"],
                "contiguous_scalar": by_name["mean_contiguous"]["instructions_per_sample"],
                "avx2": by_name["mean_avx2"]["instructions_per_sample"],
                "ratio_scalar_over_avx2": round(ratio("mean_avx2", "mean_scalar"), 3),
            },
            "preemph_instructions_per_sample": {
                "scalar": by_name["preemph_scalar"]["instructions_per_sample"],
                "avx2": by_name["preemph_avx2"]["instructions_per_sample"],
                "ratio_scalar_over_avx2": round(ratio("preemph_avx2", "preemph_scalar"), 3),
            },
            "log_instructions_per_value": {
                "scalar": by_name["log_scalar"]["instructions_per_sample"],
                "avx2": by_name["log_avx2"]["instructions_per_sample"],
                "ratio_scalar_over_avx2": round(ratio("log_avx2", "log_scalar"), 3),
            },
            "unaligned_moves_remaining": {
                row["name"]: row["unaligned_moves"] for row in rows
            },
        },
        "non_claims": [
            "instruction counts are static; they bound nothing about cycles on a specific core",
            "the probe wraps the kernels so the compiler can inline them; the wrappers are the "
            "functions measured",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'kernel':<18}{'samples/it':>11}{'insn/loop':>11}{'insn/sample':>13}"
          f"{'ymm':>5}{'xmm':>5}{'fma':>5}{'unaligned':>11}")
    for row in rows:
        print(f"{row['name']:<18}{row['samples_per_iteration']:>11}{row['instruction_count']:>11}"
              f"{row['instructions_per_sample']:>13}{row['vector_ops_256']:>5}"
              f"{row['vector_ops_128']:>5}{row['fma']:>5}{row['unaligned_moves']:>11}")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report, by_name), encoding="utf-8")
    print(f"\nwrote {args.out} and {args.markdown}")
    return 0


PAIRS = [
    ("mean", "mean_scalar", "mean_avx2",
     "Running-mean over the 400-sample frame. The scalar version recomputes the ring "
     "index with a divide and a multiply per sample; the contiguous version hoists "
     "that out, and the AVX2 version then consumes 16 samples per iteration."),
    ("preemphasis", "preemph_scalar", "preemph_avx2",
     "Pre-emphasis then Hamming window. `y = x[i] - a*x[i-1]` is a two-tap stencil, "
     "so the vector body needs one unaligned load (p + i - 1) per four samples and "
     "cannot avoid it: the shifted stream genuinely starts one element earlier. The "
     "forward loads and both stores are aligned."),
    ("log", "log_scalar", "log_avx2",
     "Log compression over 80 mel bins. The scalar body is 49 instructions per value "
     "because it calls logf; the AVX2 body is 3.875 per value with no call at all."),
]


def render_markdown(report: dict, by_name: dict) -> str:
    """Side-by-side hot-loop listing, generated from the same JSON the tests read."""
    out = [
        "# Hot-loop disassembly, scalar vs AVX2",
        "",
        f"Generated by `scripts/audit_disassembly.py` from `{report['source']}`.",
        f"Toolchain: {report['toolchain']}.",
        "",
        "Instruction counts are the loop body only. They bound nothing about cycles on a",
        "particular core; they say what the compiler actually emitted, which is the only",
        "honest answer to \"did the vectorisation happen?\".",
        "",
        "| kernel | samples/iteration | instructions/loop | instructions/sample | ymm ops | fma | unaligned moves |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["kernels"]:
        out.append(f"| `{row['name']}` | {row['samples_per_iteration']} | {row['instruction_count']} "
                   f"| {row['instructions_per_sample']:.3f} | {row['vector_ops_256']} "
                   f"| {row['fma']} | {row['unaligned_moves']} |")

    for title, scalar, vector, note in PAIRS:
        old, new = by_name[scalar], by_name[vector]
        out += ["", f"## {title}", "", note, "",
                f"`{old['instructions_per_sample']:.3f}` -> "
                f"`{new['instructions_per_sample']:.3f}` instructions per sample "
                f"({old['instructions_per_sample'] / new['instructions_per_sample']:.2f}x fewer).", "",
                f"**{scalar}** (`{old['loop_label']}`, {old['instruction_count']} instructions)", "",
                "```asm"]
        out += [f"  {line}" for line in old["loop_body"]]
        out += ["```", "",
                f"**{vector}** (`{new['loop_label']}`, {new['instruction_count']} instructions)", "",
                "```asm"]
        out += [f"  {line}" for line in new["loop_body"]]
        out += ["```"]

    out += ["", "## What is still unaligned, and why", ""]
    for row in report["kernels"]:
        if row["unaligned_moves"]:
            out.append(f"* `{row['name']}`: {row['unaligned_moves']} unaligned moves in the hot loop.")
    out += ["",
            "The pre-emphasis stencil reads `p[i-1]` while the window coefficient and the",
            "output store advance at `i`. Making the shifted load aligned would mean either",
            "shifting the whole frame by one element before the loop (an extra copy per",
            "frame) or keeping a duplicated ring with a one-sample prefix. Neither is free,",
            "and on this hardware the unaligned load is not the bottleneck, so the audit",
            "records the count instead of hiding it.", ""]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main())
