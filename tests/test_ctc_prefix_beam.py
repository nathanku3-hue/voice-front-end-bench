"""Numerical CTC prefix-beam tests with an independently enumerated oracle.

Keep exhaustive enumeration here rather than reusing production transition
logic. These tests run in CI without downloaded models or WAVs.
"""
import itertools
import math
import sys
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ctc_prefix_beam_search import log_add, prefix_beam_search
from ctc_greedy_baseline import greedy_ids


def independent_collapse(alignment, blank):
    # Independent implementation: group consecutive symbols, then remove blank.
    collapsed = [key for key, group in itertools.groupby(alignment)]
    return tuple(symbol for symbol in collapsed if symbol != blank)


def exhaustive_log_oracle(log_probs, blank=0):
    """Enumerate V^T alignment paths, then logaddexp by collapsed token tuple."""
    matrix = np.asarray(log_probs, dtype=np.float64)
    t, v = matrix.shape
    grouped = {}
    for alignment in itertools.product(range(v), repeat=t):
        score = float(sum(matrix[frame, label] for frame, label in enumerate(alignment)))
        if score == -math.inf:
            continue
        prefix = independent_collapse(alignment, blank)
        grouped[prefix] = float(np.logaddexp(grouped.get(prefix, -math.inf), score))
    return grouped


def slow_pruned_reference(log_probs, width, blank=0):
    """Deliberately literal non-vectorized prefix recurrence and top-W pruning."""
    states = {(): [0., -math.inf]}
    for row in np.asarray(log_probs):
        nxt = {}
        def record(prefix, index, val):
            if val == -math.inf:
                return
            if prefix not in nxt:
                nxt[prefix] = [-math.inf, -math.inf]
            nxt[prefix][index] = float(np.logaddexp(nxt[prefix][index], val))

        for prefix, (pb, pnb) in states.items():
            total = float(np.logaddexp(pb, pnb))
            record(prefix, 0, total + row[blank])
            for c, lp in enumerate(row):
                if c == blank:
                    continue
                if prefix and c == prefix[-1]:
                    record(prefix, 1, pnb + lp)
                    record(prefix + (c,), 1, pb + lp)
                else:
                    record(prefix + (c,), 1, total + lp)
        ranked = sorted(
            nxt.items(),
            key=lambda item: (-float(np.logaddexp(*item[1])), item[0])
        )
        states = dict(ranked[:width])
    return {prefix: float(np.logaddexp(*value)) for prefix, value in states.items()}


class LogMathTests(unittest.TestCase):
    def test_log_add_extremes(self):
        self.assertEqual(log_add(-math.inf, -math.inf), -math.inf)
        self.assertEqual(log_add(-math.inf, -31.2), -31.2)
        self.assertEqual(log_add(-31.2, -math.inf), -31.2)
        for a, b in [(0., 0.), (-0.1, -30.), (-1e6, -1e6-200), (-900., -901.),
                     (-5., -50.), (-2.123, -2.456)]:
            self.assertAlmostEqual(log_add(a, b), float(np.logaddexp(a, b)), places=10)
        for a, b in ((0., math.inf), (-math.inf, math.inf),
                     (-math.inf, math.nan), (math.nan, -math.inf)):
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                log_add(a, b)

    def test_invalid_inputs(self):
        x = np.log(np.array([[.8, .2], [.2, .8]]))
        for kwargs in (dict(beam_width=0), dict(beam_width=2, blank_id=2),
                       dict(beam_width=2, strategy="hard_skip"),
                       dict(beam_width=2, strategy="none", blank_threshold=.9),
                       dict(beam_width=2, strategy="forced_blank", blank_threshold=1.),
                       dict(beam_width=2, strategy="invalid", blank_threshold=.9)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                prefix_beam_search(x, **kwargs)
        with self.assertRaises(ValueError):
            prefix_beam_search(np.array([[math.nan, 0.]]), beam_width=2)
        with self.assertRaises(ValueError):
            prefix_beam_search(np.array([[.1, -.1]]), beam_width=2)


class PrefixOracleTests(unittest.TestCase):
    def assert_oracle_equal(self, matrix, blank):
        t, v = matrix.shape
        expected = exhaustive_log_oracle(matrix, blank)
        result = prefix_beam_search(matrix, beam_width=v ** t, blank_id=blank)
        self.assertEqual(set(result.prefix_scores), set(expected))
        for prefix, oracle_log in expected.items():
            self.assertAlmostEqual(
                result.prefix_scores[prefix], oracle_log, delta=1e-6,
                msg=f"prefix={prefix}, blank={blank}"
            )
        self.assertEqual(result.tokens, min(expected, key=lambda p: (-expected[p], p)))
        self.assertEqual(result.triggered_blank_frames, 0)
        self.assertEqual(result.hard_skipped_frames, 0)
        self.assertEqual(result.forced_blank_frames, 0)

    def test_3x4_and_4x3_all_prefix_scores(self):
        for shape, blanks in [((3, 4), (0, 2)), ((4, 3), (0, 1))]:
            for seed in (3, 19, 2026):
                generator = np.random.default_rng(seed)
                probs = generator.uniform(.01, 1., size=shape)
                probs /= probs.sum(axis=1, keepdims=True)
                for blank in blanks:
                    with self.subTest(shape=shape, seed=seed, blank=blank):
                        self.assert_oracle_equal(np.log(probs), blank)

    def test_oracle_with_constrained_zero_paths(self):
        # Zero-probability paths are excluded; code must correctly retain -inf.
        x = np.array([[.1, .9, 0.], [.95, .05, 0.], [.05, .95, 0.]])
        with np.errstate(divide="ignore"):
            self.assert_oracle_equal(np.log(x), 0)

    def test_vectorized_top_beam_equals_literal_recurrence(self):
        # In addition to no-pruning oracle checks, verify EXACT top-W
        # selection after acoustic-vectorization even when W << V**T.
        for seed in (1, 7, 100):
            rng = np.random.default_rng(seed)
            p = rng.uniform(0.005, 1., size=(4, 4))
            p /= p.sum(axis=1, keepdims=True)
            for blank in (0, 2):
                for width in (1, 2, 3, 5, 12):
                    with self.subTest(seed=seed, blank=blank, width=width):
                        fast = prefix_beam_search(np.log(p), beam_width=width, blank_id=blank)
                        slow = slow_pruned_reference(np.log(p), width, blank)
                        self.assertEqual(set(fast.prefix_scores), set(slow))
                        for prefix in slow:
                            self.assertAlmostEqual(fast.prefix_scores[prefix], slow[prefix], delta=1e-6)
        tied = np.log(np.full((3, 4), .25))
        for width in (1, 2, 3):
            fast = prefix_beam_search(tied, beam_width=width)
            slow = slow_pruned_reference(tied, width)
            self.assertEqual(set(fast.prefix_scores), set(slow))

    def test_repeated_and_blank_transitions(self):
        for path, expected in [
            ([1, 1, 1], (1,)), ([1, 0, 1], (1, 1)),
            ([0, 0, 0], ()), ([1, 2, 1], (1, 2, 1)),
            ([1, 0, 1, 0, 1], (1, 1, 1))
        ]:
            distribution = np.full((len(path), 3), .00001, dtype=np.float64)
            for t, token in enumerate(path):
                distribution[t, token] = .99998
            result = prefix_beam_search(np.log(distribution), beam_width=100, blank_id=0)
            self.assertEqual(result.tokens, expected)
            self.assertEqual(tuple(greedy_ids(distribution, 0)), expected)

    def test_tie_break_is_stable(self):
        x = np.log([[.1, .45, .45]])
        result = prefix_beam_search(x, beam_width=1)
        self.assertEqual(result.tokens, (1,))

    def test_beam_width_one_is_not_defined_as_framewise_argmax(self):
        # Model's framewise greedy operation is external to Prefix Beam Search;
        # a one-prefix search should never silently switch implementation.
        x = np.log(np.array([[.26, .42, .32], [.42, .25, .33],
                             [.34, .32, .34], [.40, .45, .15]]))
        actual = prefix_beam_search(x, beam_width=1)
        self.assertTrue(actual.prefix_scores)
        self.assertEqual(len(actual.prefix_scores), 1)


class BlankShortcutTests(unittest.TestCase):
    def test_forced_blank_matches_restricted_oracle(self):
        probs = np.array([[.1, .85, .05], [.97, .02, .01], [.1, .89, .01]])
        x = np.log(probs)
        constrained = x.copy()
        constrained[1, 1:] = -math.inf
        predicted = prefix_beam_search(x, beam_width=100, blank_id=0,
                                       strategy="forced_blank", blank_threshold=.95)
        expected = exhaustive_log_oracle(constrained, blank=0)
        self.assertEqual(set(predicted.prefix_scores), set(expected))
        for prefix, value in expected.items():
            self.assertAlmostEqual(predicted.prefix_scores[prefix], value, delta=1e-6)
        self.assertEqual(predicted.tokens, (1, 1))
        self.assertEqual(predicted.triggered_blank_frames, 1)
        self.assertEqual(predicted.forced_blank_frames, 1)
        self.assertEqual(predicted.hard_skipped_frames, 0)

    def test_hard_skip_loses_repeated_token_separator(self):
        # A blank in between identical tokens enables AA.
        p = np.array([[.001, .998, .001], [.998, .001, .001],
                      [.001, .998, .001]])
        x = np.log(p)
        forced = prefix_beam_search(x, beam_width=100, blank_id=0,
                                    strategy="forced_blank", blank_threshold=.95)
        skipped = prefix_beam_search(x, beam_width=100, blank_id=0,
                                     strategy="hard_skip", blank_threshold=.95)
        self.assertEqual(forced.tokens, (1, 1))
        self.assertEqual(skipped.tokens, (1,))
        self.assertEqual(skipped.hard_skipped_frames, 1)
        self.assertEqual(skipped.forced_blank_frames, 0)

    def test_non_triggered_approximations_equal_unpruned(self):
        p = np.array([[.3, .4, .3], [.4, .3, .3], [.3, .3, .4]])
        x = np.log(p)
        standard = prefix_beam_search(x, beam_width=100)
        for kind in ("hard_skip", "forced_blank"):
            result = prefix_beam_search(x, beam_width=100, strategy=kind,
                                        blank_threshold=.99)
            self.assertEqual(result.prefix_scores, standard.prefix_scores)
            self.assertEqual(result.triggered_blank_frames, 0)

    def test_blanks_all_around(self):
        x = np.log(np.array([[.99, .01], [.99, .01], [.99, .01]]))
        full = prefix_beam_search(x, beam_width=2)
        forced = prefix_beam_search(x, beam_width=2, strategy="forced_blank",
                                    blank_threshold=.9)
        skipped = prefix_beam_search(x, beam_width=2, strategy="hard_skip",
                                     blank_threshold=.9)
        self.assertEqual(full.tokens, ())
        self.assertEqual(forced.tokens, ())
        self.assertEqual(skipped.tokens, ())
        self.assertEqual((forced.forced_blank_frames, skipped.hard_skipped_frames), (3, 3))


if __name__ == "__main__":
    unittest.main()
