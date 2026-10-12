#!/usr/bin/env python3
"""Estimate an acoustic endpoint by reverse short-time RMS energy.

WARNING: This is an automatic PROXY for speech end, not a hand-labeled
ground-truth timestamp. 'relative_db=-35' means 35 dB below the maximum
25-ms frame RMS, NOT -35 dBFS. Silent input yields no endpoint.
"""
from __future__ import annotations

import wave
from pathlib import Path
import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 25
HOP_MS = 10
METHOD = "reverse_rms_25ms_10ms_peak_relative_minus35dB_v2"


def read_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getcomptype() != "NONE" or w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != SAMPLE_RATE:
            raise ValueError(f"Require uncompressed 16 kHz mono 16-bit PCM WAV: {path}")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0


def estimate_speech_end_details(path: Path, relative_db: float = -35.0) -> dict:
    if not (-100.0 < relative_db < 0):
        raise ValueError("relative_db must be strictly between -100 and 0")
    samples = read_pcm16(path)
    duration = len(samples) / SAMPLE_RATE
    frame = round(FRAME_MS * SAMPLE_RATE / 1000)
    hop = round(HOP_MS * SAMPLE_RATE / 1000)
    if len(samples) < frame:
        return dict(speech_end_s=None, speech_end_method=METHOD, relative_threshold_db=relative_db,
                    duration_s=duration, peak_rms_dbfs=None, effective_threshold_dbfs=None,
                    trailing_margin_s=None, status="too_short")
    # Frame RMS uses float64 to avoid overflow or accidental float32 rounding.
    starts = np.arange(0, len(samples) - frame + 1, hop, dtype=np.int64)
    power = np.array([np.mean(samples[s:s + frame] ** 2) for s in starts])
    rms = np.sqrt(power)
    peak = float(np.max(rms))
    if peak < 1e-5:  # below -100dBFS peak RMS; avoid inventing speech in silence.
        return dict(speech_end_s=None, speech_end_method=METHOD, relative_threshold_db=relative_db,
                    duration_s=duration, peak_rms_dbfs=20 * np.log10(max(peak, 1e-15)),
                    effective_threshold_dbfs=None, trailing_margin_s=None, status="silent")
    # A peak-relative criterion (not absolute -35 dBFS), with a small
    # absolute minimum only to keep digital-level rounding bounded.
    threshold_rms = max(peak * 10 ** (relative_db / 20), 1e-6)
    active = np.flatnonzero(rms >= threshold_rms)
    if not len(active):
        return dict(speech_end_s=None, speech_end_method=METHOD, relative_threshold_db=relative_db,
                    duration_s=duration, peak_rms_dbfs=20 * np.log10(peak),
                    effective_threshold_dbfs=20 * np.log10(threshold_rms),
                    trailing_margin_s=None, status="no_active_frame")
    last = int(active[-1])
    # The end of the last above-threshold 25ms RMS frame, not the start.
    endpoint = min((int(starts[last]) + frame) / SAMPLE_RATE, duration)
    margin = max(0.0, duration - endpoint)
    return dict(speech_end_s=round(endpoint, 6), speech_end_method=METHOD,
                relative_threshold_db=relative_db, duration_s=round(duration, 6),
                peak_rms_dbfs=round(20 * np.log10(peak), 4),
                effective_threshold_dbfs=round(20 * np.log10(threshold_rms), 4),
                trailing_margin_s=round(margin, 6),
                status=("no_clear_tail" if margin < 0.05 else "estimated"))


def estimate_speech_end_s(path: Path, threshold_db: float = -35.0) -> float | None:
    """Compatibility wrapper; may return None for silent/too-short clips."""
    return estimate_speech_end_details(path, relative_db=threshold_db)["speech_end_s"]
