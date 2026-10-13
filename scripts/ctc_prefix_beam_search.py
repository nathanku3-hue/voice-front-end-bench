#!/usr/bin/env python3
"""Log-domain CTC prefix beam search, independent of ASR inference/tokenizer.

Each prefix (tuple of token IDs) has log-probabilities (Pb, Pnb) for
alignments ending in blank / nonblank. No language model or top-K acoustic
candidate pruning is performed. Exact prefix probabilities are guaranteed
only if no beam truncation and no blank shortcut are active.

Blank shortcuts intentionally represent *different* approximations:
  forced_blank: use only the blank transition (with log p(blank)), force Pnb=-inf;
  hard_skip:    ignore the entire frame, preserving Pb/Pnb unchanged.
Neither shortcut is lossless with respect to the original full CTC model.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NEG_INF = -math.inf
STRATEGIES = ("none", "forced_blank", "hard_skip")


def log_add(a: float, b: float) -> float:
    """Stable log(exp(a) + exp(b)); supports -inf exactly."""
    # Fail closed before the -inf early returns: (-inf, nan) must not
    # bypass numerical contract validation.
    if math.isnan(a) or math.isnan(b) or a == math.inf or b == math.inf:
        raise ValueError("Expected finite scores or -inf")
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    hi = a if a >= b else b
    return hi + math.log1p(math.exp(-abs(a - b)))


def _ranked(states, beam_width):
    """Deterministic top beam_width based on summed prefix probability."""
    if len(states) <= beam_width:
        return states
    # Negative score for descending probability, then lexical tuple for ties.
    return dict(sorted(
        states.items(),
        key=lambda item: (-log_add(item[1][0], item[1][1]), item[0])
    )[:beam_width])


def _exact_full_vocab_step(states, logp, blank_id, beam_width):
    """Vectorize all B*(V-1) token arcs without acoustic token pre-pruning.

    Every new prefix q=(parent,c) has exactly one possible parent prefix.
    Only prefixes already in the incoming beam can have *other* sources
    (blank self-loop, repeated-symbol self-loop). We merge those before
    ranking; among remaining new prefixes the top W single-source scores
    suffice for an exact global top-W because no discarded new candidate
    can receive another contribution. Tie-breaking is lexical, including
    ties on the Wth score. This is an exact beam update, NOT token Top-K.
    """
    prefixes = list(states)
    v = len(logp)
    raw = np.asarray(logp, dtype=np.float64)
    pb = np.asarray([states[p][0] for p in prefixes], dtype=np.float64)
    pnb = np.asarray([states[p][1] for p in prefixes], dtype=np.float64)
    totals = np.logaddexp(pb, pnb)
    positions = {p: i for i, p in enumerate(prefixes)}
    blank_score = float(logp[blank_id])

    # Merge all sources into any previously active prefix: its blank
    # transition, its repeat (no-growth) transition, and an append from
    # its unique parent if that parent was also present.
    existing = {}
    for i, prefix in enumerate(prefixes):
        next_pb = float(totals[i] + blank_score)
        next_pnb = float(pnb[i] + raw[prefix[-1]]) if prefix else NEG_INF
        if prefix:
            c = prefix[-1]
            parent_i = positions.get(prefix[:-1])
            if parent_i is not None:
                base = pb[parent_i] if prefixes[parent_i] and prefixes[parent_i][-1] == c else totals[parent_i]
                next_pnb = log_add(next_pnb, float(base + raw[c]))
        if next_pb != NEG_INF or next_pnb != NEG_INF:
            existing[prefix] = [next_pb, next_pnb]

    # Dense candidate arc scores, computed for *all* vocab symbols.
    candidates = totals[:, None] + raw[None, :]
    for i, prefix in enumerate(prefixes):
        if prefix:
            c = prefix[-1]
            candidates[i, c] = pb[i] + raw[c]
    candidates[:, blank_id] = NEG_INF
    for prefix in prefixes:
        if prefix:
            parent_i = positions.get(prefix[:-1])
            if parent_i is not None:
                # Already merged above; never count it again.
                candidates[parent_i, prefix[-1]] = NEG_INF

    flat = candidates.ravel()
    finite = np.isfinite(flat)
    possible_new = int(np.count_nonzero(finite))
    possible_states = possible_new + len(existing)  # before selecting top new arcs
    if possible_new <= beam_width:
        chosen = np.flatnonzero(finite).tolist()
    else:
        # Partition by score, then resolve boundary ties by prefix order.
        cutoff = float(np.partition(flat, len(flat) - beam_width)[-beam_width])
        above = np.flatnonzero(flat > cutoff).tolist()
        slots = beam_width - len(above)
        tied = np.flatnonzero(flat == cutoff).tolist()
        if len(tied) > slots:
            tied.sort(key=lambda idx: prefixes[idx // v] + (int(idx % v),))
            tied = tied[:slots]
        chosen = above + tied

    for idx in chosen:
        parent_i, c = divmod(idx, v)
        new_prefix = prefixes[parent_i] + (int(c),)
        existing[new_prefix] = [NEG_INF, float(flat[idx])]

    return (_ranked(existing, beam_width),
            int(np.count_nonzero(np.isfinite(raw))) - int(np.isfinite(raw[blank_id])),
            possible_states)


@dataclass
class BeamResult:
    tokens: tuple[int, ...]
    log_probability: float
    prefix_scores: dict[tuple[int, ...], float]
    frames: int
    triggered_blank_frames: int
    hard_skipped_frames: int
    forced_blank_frames: int
    expanded_nonblank_arcs: int
    candidate_states_considered: int
    max_active_prefixes: int


def prefix_beam_search(log_probs, *, beam_width: int, blank_id: int = 0,
                       blank_threshold: float | None = None,
                       strategy: str = "none") -> BeamResult:
    """Decode [T,V] log scores, allowing -inf for constrained toy distributions.

    Threshold is a probability in (0,1). If no threshold, use strategy='none'.
    Scores are presumed log probabilities. The caller validates normalization
    on real CTC logits (this function also allows -inf constrained oracles).
    """
    scores = np.asarray(log_probs)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("Expected a nonempty [T,V>=2] matrix")
    if not np.issubdtype(scores.dtype, np.number):
        raise ValueError("Log scores must be numeric")
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("NaN and +inf are not valid log probabilities")
    if np.max(scores) > 1e-7:
        raise ValueError("Positive log probability")
    if not isinstance(beam_width, int) or isinstance(beam_width, bool) or beam_width < 1:
        raise ValueError("beam_width must be an integer >= 1")
    vocab = int(scores.shape[1])
    if not isinstance(blank_id, int) or not (0 <= blank_id < vocab):
        raise ValueError("Blank ID out of bounds")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown blank strategy: {strategy}")
    if blank_threshold is None:
        if strategy != "none":
            raise ValueError("A shortcut strategy requires a threshold")
        threshold_log = None
    else:
        if strategy == "none" or not (0.0 < blank_threshold < 1.0):
            raise ValueError("Threshold must be in (0,1), with a shortcut strategy")
        threshold_log = math.log(blank_threshold)

    # Starting empty prefix has Pblank=1 and Pnonblank=0.
    states: dict[tuple[int, ...], list[float]] = {(): [0.0, NEG_INF]}
    triggered = hard_skipped = forced = arcs = state_count = 0
    maximum_active = 1
    for row in scores:
        logp = row.tolist()
        blank_score = float(logp[blank_id])
        shortcut = threshold_log is not None and blank_score >= threshold_log
        if shortcut:
            triggered += 1
            if strategy == "hard_skip":
                hard_skipped += 1
                continue
            forced += 1
            # Restricted CTC step: only the blank path is allowed. All
            # old Pnb mass is merged into new Pb, never carried as Pnb.
            states = {
                prefix: [log_add(pb, pnb) + blank_score, NEG_INF]
                for prefix, (pb, pnb) in states.items()
            }
            continue

        count_active = len(states)
        states, finite_nonblank_count, reachable = _exact_full_vocab_step(
            states, logp, blank_id, beam_width
        )
        if not states:
            raise ValueError("No reachable CTC prefix (all alignment mass zero)")
        arcs += count_active * finite_nonblank_count
        state_count += reachable
        maximum_active = max(maximum_active, len(states))

    totals = {prefix: log_add(pb, pnb) for prefix, (pb, pnb) in states.items()}
    if not totals:
        raise ValueError("No CTC prefixes survived")
    best = min(totals, key=lambda prefix: (-totals[prefix], prefix))
    return BeamResult(
        tokens=best,
        log_probability=totals[best],
        prefix_scores=totals,
        frames=int(scores.shape[0]),
        triggered_blank_frames=triggered,
        hard_skipped_frames=hard_skipped,
        forced_blank_frames=forced,
        expanded_nonblank_arcs=arcs,
        candidate_states_considered=state_count,
        max_active_prefixes=maximum_active,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-probs", type=Path, required=True)
    parser.add_argument("--beam-width", type=int, required=True)
    parser.add_argument("--blank-id", type=int, default=0)
    parser.add_argument("--strategy", choices=STRATEGIES, default="none")
    parser.add_argument("--blank-threshold", type=float)
    args = parser.parse_args()
    logits = np.load(args.log_probs, allow_pickle=False)
    result = prefix_beam_search(
        logits, beam_width=args.beam_width, blank_id=args.blank_id,
        blank_threshold=args.blank_threshold, strategy=args.strategy,
    )
    print(json.dumps({
        "tokens": result.tokens, "log_probability": result.log_probability,
        "frames": result.frames,
        "triggered_blank_frames": result.triggered_blank_frames,
        "forced_blank_frames": result.forced_blank_frames,
        "hard_skipped_frames": result.hard_skipped_frames,
        "expanded_nonblank_arcs": result.expanded_nonblank_arcs,
        "max_active_prefixes": result.max_active_prefixes,
    }))


if __name__ == "__main__":
    main()
