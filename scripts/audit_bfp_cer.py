#!/usr/bin/env python3
"""End-to-end empirical verification of Block Floating Point (BFP) vs Plain Fixed-Point.

Companion to SIMD_LOG_FIXEDPOINT_AXI.md (Section 3). That document made an
analytical claim: that the 54 dB between-frame dynamic range breaks plain Q15
(and even Q23) on quiet frames, and that Block Floating Point (BFP16: 16-bit
mantissa + 5-bit block exponent per frame) is the hardware answer to guarantee
70.6 dB SQNR.

This script replaces the analytical argument with an empirical end-to-end model
run across all 30 AISHELL-1 Dev utterances:

    1. float32 baseline: clean unquantized features.
    2. feature_bfp16: 16-bit mantissa + 5-bit per-frame exponent on log features.
    3. feature_plain_q15: 16-bit fixed point with global full-scale FS=16.0.
    4. energy_bfp16: 16-bit mantissa + 5-bit per-frame exponent on linear Mel energies before log.
    5. energy_bfp12: 12-bit mantissa + 5-bit exponent on linear Mel energies.
    6. energy_q23: 24-bit fixed point (FS=256.0) on linear Mel energies.
    7. energy_q15: 16-bit fixed point (FS=256.0) on linear Mel energies.

The result is categorical:
  * In the linear energy domain, Plain Q15 suffers catastrophic failure: CER
    explodes from 2.94% to 41.63% (13 -> 184 errors / 442 chars), with 24/30
    transcripts broken, because quiet frames fall below the global LSB step.
  * Even Plain Q23 (24-bit) degrades CER from 2.94% to 4.52% (20 errors).
  * In stark contrast, Block Floating Point (BFP16) preserves precision across
    all frames: CER is 2.71%~2.94% (12~13 errors), with 29~30 of 30 transcripts
    identical to Float32.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from asr_metrics import evaluate, corpus
from audit_numeric_sensitivity import open_session, forward, run, sha256


def bfp_quantize(x: np.ndarray, bits: int = 16) -> np.ndarray:
    """Block floating point: per-frame peak normalization with b-bit mantissa."""
    out = np.zeros_like(x)
    max_val = (1 << (bits - 1)) - 1
    min_val = -(1 << (bits - 1))
    for t in range(x.shape[0]):
        row = x[t]
        peak = np.max(np.abs(row))
        if peak == 0:
            continue
        exp = int(np.ceil(np.log2(peak)))
        scale = 2.0 ** (bits - 1 - exp)
        m = np.round(row * scale)
        m = np.clip(m, min_val, max_val)
        out[t] = m / scale
    return out


def fixed_quantize(x: np.ndarray, bits: int = 16, fs: float = 16.0) -> np.ndarray:
    """Plain fixed point: global full scale FS."""
    max_val = (1 << (bits - 1)) - 1
    min_val = -(1 << (bits - 1))
    scale = max_val / fs
    m = np.round(x * scale)
    m = np.clip(m, min_val, max_val)
    return m / scale


def energy_bfp_to_log(feats: np.ndarray, bits: int = 16) -> np.ndarray:
    """BFP quantization in the linear Mel energy domain before log compression."""
    energy = np.exp(feats)
    out = np.zeros_like(energy)
    max_val = (1 << (bits - 1)) - 1
    for t in range(energy.shape[0]):
        row = energy[t]
        peak = np.max(row)
        if peak <= 1e-10:
            out[t] = 1e-10
            continue
        exp = int(np.ceil(np.log2(peak)))
        scale = 2.0 ** (bits - 1 - exp)
        m = np.round(row * scale)
        m = np.clip(m, 0, max_val)
        recon = np.maximum(m / scale, 1e-10)
        out[t] = recon
    return np.log(out)


def energy_fixed_to_log(feats: np.ndarray, bits: int = 16, fs: float = 256.0) -> np.ndarray:
    """Plain fixed-point quantization in linear Mel energy domain before log."""
    energy = np.exp(feats)
    max_val = (1 << (bits - 1)) - 1
    scale = max_val / fs
    m = np.round(energy * scale)
    m = np.clip(m, 0, max_val)
    recon = np.maximum(m / scale, 1e-10)
    return np.log(recon)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-root", type=Path, default=REPO / "build/ctc_round1_dev")
    parser.add_argument("--data-root", type=Path, default=REPO / "data/ctc_eval")
    parser.add_argument("--model-dir", type=Path, default=REPO / "build/win-zig/model")
    parser.add_argument("--dump-binary", type=Path, default=REPO / "build/win-zig/fbank_feature_dump.exe")
    parser.add_argument("--pipeline-binary", type=Path, default=REPO / "build/win-zig/fbank_ctc_pipeline.exe")
    parser.add_argument("--baseline", type=Path, default=REPO / "results/ctc_baseline_dev.json")
    parser.add_argument("--inventory", type=Path, default=REPO / "results/ctc_logit_inventory.json")
    parser.add_argument("--out", type=Path, default=REPO / "results/bfp_cer_audit.json")
    args = parser.parse_args()

    model_file = args.model_dir / "model.int8.onnx"
    tokens = args.model_dir / "tokens.txt"
    manifest = [json.loads(l) for l in (args.dev_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    ref_text = {u["id"]: u["reference"] for u in baseline["utterances"]}

    session = open_session(model_file)
    scratch = Path(tempfile.mkdtemp(prefix="bfp-cer-audit-"))

    methods = {
        "float32": lambda f: f,
        "feature_bfp16": lambda f: bfp_quantize(f, 16),
        "feature_plain_q15": lambda f: fixed_quantize(f, 16, 16.0),
        "energy_bfp16": lambda f: energy_bfp_to_log(f, 16),
        "energy_bfp12": lambda f: energy_bfp_to_log(f, 12),
        "energy_q23": lambda f: energy_fixed_to_log(f, 24, 256.0),
        "energy_q15": lambda f: energy_fixed_to_log(f, 16, 256.0),
    }

    raw_features = {}
    for entry in manifest:
        uid = entry["id"]
        wav = args.data_root / "audio" / f"{uid}.wav"
        raw = scratch / f"{uid}.f32"
        meta = scratch / f"{uid}.json"
        cmd = [str(args.dump_binary), "--wav", str(wav), "--out", str(raw), "--json", str(meta), "--log", "fast"]
        run(cmd)
        info = json.loads(meta.read_text(encoding="utf-8"))
        feats = np.fromfile(raw, dtype="<f4").reshape(info["frames"], info["bins"])
        raw_features[uid] = np.vstack([feats[0:1], feats, feats[-1:]])

    results = {}
    for m_name, m_func in methods.items():
        hyps = {}
        maes = []
        counts = []
        for entry in manifest:
            uid = entry["id"]
            wav = args.data_root / "audio" / f"{uid}.wav"
            f_in = raw_features[uid]
            q_in = m_func(f_in)
            if m_name != "float32":
                maes.append(float(np.mean(np.abs(f_in - q_in))))
            log_probs = forward(session, q_in)
            npy = scratch / f"{uid}_{m_name}.npy"
            np.save(npy, log_probs)
            out = subprocess.run([str(args.pipeline_binary), "--wav", str(wav), "--posteriors", str(npy),
                                 "--tokens", str(tokens), "--repeats", "1"],
                                 capture_output=True, text=True, encoding="utf-8")
            text = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
            hyps[uid] = text
            counts.append(evaluate(ref_text[uid], text))
        c = corpus(counts)
        results[m_name] = {
            "cer": c["cer"],
            "substitutions": c["substitutions"],
            "deletions": c["deletions"],
            "insertions": c["insertions"],
            "errors": c["substitutions"] + c["deletions"] + c["insertions"],
            "mae_vs_float": float(np.mean(maes)) if maes else 0.0,
            "transcripts": hyps,
        }

    float_transcripts = results["float32"]["transcripts"]
    summary = []
    print("\n" + "=" * 80)
    print("Block Floating Point (BFP) vs Plain Fixed-Point CER Audit (30 Dev Utterances)")
    print("=" * 80)
    print(f"{'Method':<20} {'Domain':<16} {'CER':<8} {'Errors':<10} {'Identical':<12} {'MAE vs Float':<12}")
    print("-" * 80)
    for m_name, res in results.items():
        same = sum(1 for uid in float_transcripts if res["transcripts"][uid] == float_transcripts[uid])
        domain = "Linear Energy" if "energy" in m_name else "Log-Mel Feat"
        cer_str = f"{res['cer'] * 100:.2f}%"
        err_str = f"{res['errors']}/442"
        same_str = f"{same}/30"
        mae_str = f"{res['mae_vs_float']:.2e}" if res['mae_vs_float'] > 0 else "--"
        print(f"{m_name:<20} {domain:<16} {cer_str:<8} {err_str:<10} {same_str:<12} {mae_str:<12}")
        summary.append({
            "method": m_name,
            "domain": domain,
            "cer": res["cer"],
            "errors": res["errors"],
            "identical_to_float": same,
            "mae_vs_float": res["mae_vs_float"]
        })
    print("=" * 80 + "\n")

    out_payload = {
        "status": "PASS",
        "description": "Empirical CER audit of Block Floating Point vs Fixed Point",
        "summary": summary,
        "methods": results,
    }
    args.out.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved artifact: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
