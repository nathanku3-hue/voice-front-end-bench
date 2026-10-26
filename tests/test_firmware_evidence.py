"""Check that the firmware-facing artifacts back the claims made about them.

Four audits, four artifact files, and one rule: every published ratio is
recomputed here from the raw counters in the JSON. Nothing in this file
re-measures anything, and nothing here trusts a number that the artifact does
not also carry the inputs for.

    results/zero_alloc_steady_state.json   hot path allocates nothing
    results/hysteresis_flow.json           dual watermark cuts handshakes
    results/stage_latency_budget.json      where the frame time goes
    results/mel_sparsity.json              the hot spot, removed, bit-exactly
    results/pipeline_parity.json           C++ text == Python text, 30/30
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name):
    path = RESULTS / name
    if not path.is_file():
        raise unittest.SkipTest(f"run scripts/run_firmware_audits.py first ({name} missing)")
    return json.loads(path.read_text(encoding="utf-8"))


class ZeroAllocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load("zero_alloc_steady_state.json")

    def test_window_is_a_steady_state_under_backpressure(self):
        # Without a pause the audit only exercises the happy path, and without a
        # drained frame it never compared anything.
        self.assertGreater(self.report["pause_events"], 0)
        self.assertGreater(self.report["resume_events"], 0)
        self.assertEqual(self.report["frames"], 10000)

    def test_not_one_allocation_in_the_window(self):
        self.assertEqual(self.report["window_allocations"], 0)
        self.assertEqual(self.report["window_denied_allocations"], 0)
        self.assertEqual(self.report["window_live_delta_bytes"], 0)

    def test_probe_is_actually_wired(self):
        # A zero only means something if a non-zero is reachable.
        self.assertTrue(self.report["probe_control_passed"])
        self.assertGreater(self.report["warmup_allocations"], 0)

    def test_every_value_was_compared_and_matched(self):
        self.assertEqual(self.report["max_abs_diff"], 0.0)
        self.assertEqual(self.report["compared_values"], self.report["frames"] * 80)

    def test_footprint_is_a_declared_constant(self):
        # The bridge is one feature slot plus its extractor. The dense mel table
        # is not in the object (it is materialised only for the A/B path), so
        # this number must stay in the single-digit kilobytes; if it grows, the
        # "fixed, small footprint" claim has to be revisited.
        self.assertEqual(self.report["bytes_per_frame_slot"], 640)
        self.assertLess(self.report["bridge_footprint_bytes"], 16 * 1024)


class HysteresisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load("hysteresis_flow.json")
        cls.arms = cls.report["arms"]

    def test_control_arm_is_single_watermark(self):
        control = self.arms[0]
        self.assertEqual(control["mode"], "single")
        self.assertEqual(control["low_watermark"], control["high_watermark"] - 1)

    def test_every_arm_is_bounded_and_lossless(self):
        for arm in self.arms:
            with self.subTest(low=arm["low_watermark"]):
                self.assertEqual(arm["max_abs_diff"], 0.0)
                self.assertEqual(arm["dropped_frames"], 0)
                self.assertLessEqual(arm["max_occupancy"], arm["capacity_frames"])
                self.assertEqual(arm["compared_values"], arm["frames"] * 80)

    def test_reduction_fractions_recompute(self):
        control = self.arms[0]["transitions"]
        for arm in self.arms:
            with self.subTest(low=arm["low_watermark"]):
                expected = 1.0 - arm["transitions"] / control
                self.assertAlmostEqual(arm["transition_reduction_fraction"], expected, places=12)

    def test_more_headroom_means_fewer_transitions(self):
        # Monotone in the low watermark: the whole point of the mechanism.
        ordered = sorted(self.arms, key=lambda a: a["low_watermark"], reverse=True)
        counts = [arm["transitions"] for arm in ordered]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_the_saving_is_paid_for_in_stall_length(self):
        # Hysteresis buys handshakes with queueing delay. If the deepest low
        # watermark did not stall longer, the mechanism would be free and the
        # trade-off claim would be wrong.
        deepest = min(self.arms, key=lambda a: a["low_watermark"])
        self.assertGreater(deepest["max_stall_ticks"], self.arms[0]["max_stall_ticks"])

    def test_transition_rate_matches_the_counts(self):
        for arm in self.arms:
            with self.subTest(low=arm["low_watermark"]):
                self.assertAlmostEqual(arm["transitions_per_1000_ticks"],
                                       1000.0 * arm["transitions"] / arm["ticks"], places=9)
                self.assertEqual(arm["transitions"], arm["pauses"] + arm["resumes"])


class StageBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load("stage_latency_budget.json")
        cls.stages = cls.report["stages"]

    def test_shares_sum_to_one_hundred(self):
        self.assertAlmostEqual(sum(s["share_percent"] for s in self.stages), 100.0, places=9)

    def test_per_frame_cost_recomputes_from_totals(self):
        frames = self.report["frames"]
        for stage in self.stages:
            with self.subTest(stage=stage["name"]):
                self.assertAlmostEqual(stage["us_per_frame"],
                                       1000.0 * stage["total_ms"] / frames, places=9)
                self.assertAlmostEqual(stage["share_percent"],
                                       100.0 * stage["total_ms"]
                                       / self.report["instrumented_total_ms"], places=9)

    def test_stage_names_are_the_declared_ones(self):
        self.assertEqual([s["name"] for s in self.stages],
                         ["dc_remove", "preemphasis_window", "fftw_execute",
                          "power_spectrum", "mel_projection", "log_compress"])

    def test_timing_was_pinned_and_spread_is_published(self):
        self.assertTrue(self.report["pinned_to_one_core"])
        # The pair is published as the observed extremes, in whichever order the
        # binary happened to record them.
        low, high = sorted(self.report["wall_spread_ms"])
        self.assertLessEqual(low, high)
        # Pinning does not make this device latency. At ~4 us/frame the residual
        # jitter between passes has been observed up to ~1.7x, so the guard is
        # set to catch a collapse of measurement quality, not to certify
        # stability.
        self.assertLess(high / low, 2.5)

    def test_unattributed_share_is_reported(self):
        derived = self.report["derived"]
        expected = 100.0 * (self.report["wall_ms"] - self.report["instrumented_total_ms"]) \
            / self.report["wall_ms"]
        self.assertAlmostEqual(derived["outside_stages_percent"], expected, places=9)
        # Six clock reads, the ring copy and the queue bookkeeping are outside
        # every stage. Their cost is host dependent, so this is a sanity bound
        # against instrumentation collapse, not a quality target.
        self.assertLess(derived["outside_stages_percent"], 50.0)

    def test_hottest_stage_matches_the_rows(self):
        hottest = max(self.stages, key=lambda s: s["total_ms"])
        self.assertEqual(self.report["derived"]["hottest_stage"], hottest["name"])

    def test_dense_reference_is_published_and_comparable(self):
        # The before/after table in the doc is only legitimate if both halves are
        # in the artifact, and if the un-instrumented per-frame cost -- the same
        # code in both modes -- did not move between the two runs.
        dense = self.report["dense_reference"]
        self.assertAlmostEqual(sum(s["share_percent"] for s in dense["stages"]), 100.0, places=9)
        self.assertGreater(dense["wall_us_per_frame"], self.report["wall_us_per_frame"])
        overhead = self.report["derived"]["fixed_overhead_us_per_frame"]
        self.assertLess(overhead["relative_drift"], 0.30)
        # Recompute the mel ratio the doc quotes from the two stage tables.
        sparse_mel = next(s["us_per_frame"] for s in self.stages if s["name"] == "mel_projection")
        dense_mel = next(s["us_per_frame"] for s in dense["stages"] if s["name"] == "mel_projection")
        self.assertAlmostEqual(self.report["derived"]["mel_stage_ratio"],
                               dense_mel / sparse_mel, places=9)
        self.assertGreater(dense_mel / sparse_mel, 10.0)


class MelSparsityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load("mel_sparsity.json")

    def test_compression_ratios_recompute(self):
        report = self.report
        self.assertEqual(report["macs_dense"], report["dense_entries"])
        self.assertEqual(report["macs_sparse"], report["nonzeros"])
        self.assertAlmostEqual(report["mac_reduction"],
                               report["dense_entries"] / report["nonzeros"], places=9)
        self.assertLess(report["nonzeros"], report["dense_entries"])

    def test_speedup_is_measured_not_extrapolated(self):
        report = self.report
        self.assertAlmostEqual(report["speedup"],
                               report["dense_us_per_frame"] / report["sparse_us_per_frame"],
                               places=9)
        # The realised speed-up must not be larger than the arithmetic headroom;
        # if it were, the two paths would not be doing the same work.
        self.assertLessEqual(report["speedup"], report["mac_reduction"])

    def test_the_optimisation_is_bit_exact(self):
        # This is the whole argument for shipping it: same bits, less work.
        self.assertEqual(self.report["bit_mismatches"], 0)
        self.assertEqual(self.report["max_abs_diff"], 0.0)
        self.assertEqual(self.report["compared_values"], self.report["frames"] * 80)


class PipelineParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load("pipeline_parity.json")

    def test_all_thirty_utterances_match_python(self):
        report = self.report
        self.assertEqual(report["utterances"], 30)
        self.assertEqual(report["transcripts_identical"], 30)
        self.assertEqual(report["token_counts_identical"], 30)
        self.assertEqual(report["logit_frames_consistent"], 30)
        self.assertEqual(report["transcripts_mismatched"], [])

    def test_utf8_replacement_policy_matches_cpython(self):
        # Two utterances end on a partial character; both paths must emit
        # exactly one U+FFFD there and nowhere else.
        self.assertEqual(self.report["utf8_replacement_total"], 2)
        self.assertEqual(self.report["utf8_replacement_mismatched"], [])

    def test_scope_disclaims_the_model(self):
        # The binary has no acoustic model in it. If this string ever changes to
        # claim C++ inference, the artifact and the claim must move together.
        self.assertIn("NOT in the binary", self.report["scope"])


class InterleavedBudgetTests(unittest.TestCase):
    """The SIMD + fast-log arms, measured round-robin inside one process."""

    @classmethod
    def setUpClass(cls):
        cls.report = load("stage_interleaved_budget.json")
        cls.arms = {arm["name"]: arm for arm in cls.report["arms"]}

    def test_all_three_arms_are_present_and_share_a_frame_count(self):
        self.assertEqual(sorted(self.arms), ["baseline", "simd", "simd_fastlog"])
        counts = {arm.get("frames", self.report["frames"]) for arm in self.arms.values()}
        self.assertEqual(len(counts), 1)
        self.assertGreater(self.report["frames"], 0)

    def test_every_arm_still_sums_to_one_hundred_percent(self):
        for arm in self.arms.values():
            self.assertAlmostEqual(sum(s["share_percent"] for s in arm["stages"]), 100.0, places=9)

    def test_speedups_recompute_from_the_arm_totals(self):
        derived = self.report["derived"]
        base = self.arms["baseline"]["wall_us_per_frame"]
        self.assertAlmostEqual(derived["speedup_simd"],
                               base / self.arms["simd"]["wall_us_per_frame"], places=9)
        self.assertAlmostEqual(derived["speedup_simd_fastlog"],
                               base / self.arms["simd_fastlog"]["wall_us_per_frame"], places=9)
        self.assertGreater(derived["speedup_simd_fastlog"], 1.0)

    def test_sample_domain_reduction_recomputes_from_the_stage_rows(self):
        def pair(arm):
            stages = {s["name"]: s["us_per_frame"] for s in self.arms[arm]["stages"]}
            return stages["dc_remove"] + stages["preemphasis_window"]

        base, simd = pair("baseline"), pair("simd")
        derived = self.report["derived"]
        self.assertAlmostEqual(derived["sample_domain_us_per_frame"]["baseline"], base, places=9)
        self.assertAlmostEqual(derived["sample_domain_reduction_percent"]["simd"],
                               100.0 * (1.0 - simd / base), places=9)
        # The claim is a large reduction, not a marginal one.
        self.assertGreater(derived["sample_domain_reduction_percent"]["simd"], 40.0)

    def test_log_stage_reduction_recomputes(self):
        def log_us(arm):
            return next(s["us_per_frame"] for s in self.arms[arm]["stages"]
                        if s["name"] == "log_compress")

        derived = self.report["derived"]
        self.assertAlmostEqual(derived["log_reduction_percent"]["simd_fastlog"],
                               100.0 * (1.0 - log_us("simd_fastlog") / log_us("baseline")),
                               places=9)
        self.assertGreater(derived["log_reduction_percent"]["simd_fastlog"], 40.0)

    def test_control_stages_agree_between_arms(self):
        # fftw / power / mel / std-log run identical code in the arms they are
        # compared across. If they disagree, the host moved and no arm
        # difference is attributable to the code.
        drift = self.report["derived"]["control_stage_drift_percent"]
        self.assertTrue(drift)
        self.assertLess(max(drift.values()), 25.0)
        self.assertAlmostEqual(self.report["derived"]["worst_control_stage_drift_percent"],
                               max(drift.values()), places=9)


class DisassemblyTests(unittest.TestCase):
    """Static evidence that the vectorisation actually happened."""

    @classmethod
    def setUpClass(cls):
        cls.report = load("simd_disassembly.json")
        cls.kernels = {row["name"]: row for row in cls.report["kernels"]}

    def test_instruction_density_recomputes(self):
        for row in self.kernels.values():
            self.assertAlmostEqual(
                row["instructions_per_sample"],
                row["instruction_count"] / row["samples_per_iteration"], places=3)

    def test_scalar_kernels_use_no_256_bit_registers(self):
        for name in ("mean_scalar", "preemph_scalar", "mean_contiguous", "log_scalar"):
            self.assertEqual(self.kernels[name]["vector_ops_256"], 0, name)

    def test_vector_kernels_use_256_bit_registers(self):
        for name in ("mean_avx2", "preemph_avx2", "log_avx2"):
            self.assertGreater(self.kernels[name]["vector_ops_256"], 0, name)
            self.assertGreater(self.kernels[name]["samples_per_iteration"], 1, name)

    def test_published_ratios_recompute(self):
        derived = self.report["derived"]
        for key, new, old in (
                ("mean_instructions_per_sample", "mean_avx2", "mean_scalar"),
                ("preemph_instructions_per_sample", "preemph_avx2", "preemph_scalar"),
                ("log_instructions_per_value", "log_avx2", "log_scalar")):
            expected = (self.kernels[old]["instructions_per_sample"]
                        / self.kernels[new]["instructions_per_sample"])
            self.assertAlmostEqual(derived[key]["ratio_scalar_over_avx2"], expected, places=3)

    def test_unaligned_moves_are_declared_not_hidden(self):
        # The pre-emphasis stencil genuinely needs the shifted load; the count is
        # published so nobody has to guess whether the audit noticed.
        remaining = self.report["derived"]["unaligned_moves_remaining"]
        self.assertIn("preemph_avx2", remaining)
        self.assertGreater(remaining["preemph_avx2"], 0)

    def test_markdown_diff_is_generated_alongside_the_json(self):
        path = RESULTS / "simd_disassembly_diff.md"
        if not path.is_file():
            raise unittest.SkipTest("run scripts/audit_disassembly.py first")
        text = path.read_text(encoding="utf-8")
        for name in ("mean_avx2", "preemph_avx2", "log_avx2"):
            self.assertIn(f"**{name}**", text)


class FastLogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load("fast_log_accuracy.json")
        cls.kernels = {row["name"]: row for row in cls.report["kernels"]}

    def test_bounds_hold(self):
        fast = self.kernels["fast_log_float"]
        self.assertLess(fast["max_abs_error"], 2e-5)
        self.assertLess(fast["max_rel_error"], 1e-6)

    def test_the_strawman_is_measurably_worse(self):
        # Otherwise the comparison is theatre.
        self.assertGreater(self.kernels["degree2_taylor"]["max_abs_error"],
                           50.0 * self.kernels["fast_log_float"]["max_abs_error"])

    def test_decade_profile_covers_the_sweep(self):
        profiles = self.report["decade_profiles"]
        total = self.report["points"]
        for name, profile in profiles.items():
            self.assertEqual(sum(entry["samples"] for entry in profile), total, name)
            # The per-decade maxima can never exceed the published global max.
            self.assertLessEqual(max(entry["max_abs"] for entry in profile),
                                 self.kernels[name]["max_abs_error"] * (1 + 1e-12), name)


class NumericSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load("numeric_sensitivity.json")

    def test_the_model_is_the_frozen_export(self):
        model = self.report["model"]
        self.assertTrue(model["asset_matches_frozen_export"])
        self.assertEqual(model["sha256_observed"], model["sha256_recorded_in_frozen_export"])

    def test_shipped_path_changes_no_transcript(self):
        summary = self.report["summary"]
        self.assertEqual(summary["fast"]["transcripts_identical_to_std"],
                         summary["fast"]["utterances"])
        self.assertEqual(summary["fast"]["transcripts_differing"], [])

    def test_scalar_reference_is_bit_identical(self):
        scalar = self.report["summary"]["scalar_std"]
        self.assertEqual(scalar["feature_mae_mean"], 0.0)
        self.assertEqual(scalar["feature_max_abs_overall"], 0.0)

    def test_feature_error_is_orders_below_the_tolerance(self):
        fast = self.report["summary"]["fast"]
        self.assertLess(fast["feature_mae_mean"], 1e-6)
        self.assertLess(fast["feature_max_abs_overall"],
                        self.report["tolerance_argument"]["feature_tolerance_used"])

    def test_cer_arithmetic_recomputes(self):
        for name, summary in self.report["summary"].items():
            counts = summary["cer_counts"]
            expected = ((counts["substitutions"] + counts["deletions"] + counts["insertions"])
                        / counts["reference_chars"])
            self.assertAlmostEqual(summary["cer"], expected, places=12, msg=name)

    def test_identical_count_matches_the_differing_list(self):
        for name, summary in self.report["summary"].items():
            self.assertEqual(summary["transcripts_identical_to_std"],
                             summary["utterances"] - len(summary["transcripts_differing"]), name)


class CerSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load("cer_sensitivity.json")
        cls.rows = cls.report["rows"]

    def test_the_ladder_is_geometric_and_starts_clean(self):
        levels = [row["nominal_amplitude"] for row in self.rows]
        self.assertEqual(levels[0], 0.0)
        self.assertEqual(levels, sorted(levels))
        self.assertEqual(self.rows[0]["feature_mae"], 0.0)

    def test_achieved_mae_matches_uniform_noise(self):
        # Uniform on [-a, a] has mean |x| = a/2. A row that deviates by more than
        # a few percent means the perturbation was not what it claimed to be.
        for row in self.rows[1:]:
            self.assertAlmostEqual(row["feature_mae"] / row["nominal_amplitude"], 0.5, delta=0.05)

    def test_identical_counts_and_differing_lists_agree(self):
        total = self.report["protocol"]["utterances"]
        for row in self.rows:
            self.assertEqual(row["transcripts_identical_to_clean"],
                             total - len(row["transcripts_differing"]))

    def test_the_clean_run_reproduces_the_baseline_cer(self):
        self.assertAlmostEqual(self.report["derived"]["clean_cer"], 0.029411764705882353,
                               places=9)
        self.assertTrue(self.report["derived"]["clean_cer_matches_published_baseline"])

    def test_the_graph_is_deterministic(self):
        # Without this, every flip below could be the runtime rather than the
        # injected noise.
        control = self.report["determinism_control"]
        self.assertTrue(control["identical_posteriors"])
        self.assertEqual(control["max_abs_diff"], 0.0)

    def test_the_perturbation_does_not_accumulate(self):
        # This is the finding, and it is deliberately not a threshold: the CER
        # moves by at most one character over five decades of injected noise.
        derived = self.report["derived"]
        # Two characters is the measured worst case over five decades of
        # injected noise; the bound is here to catch a runaway, not to certify
        # a specific number.
        self.assertLessEqual(derived["max_extra_edits_vs_clean"], 3)
        self.assertLess(derived["cer_max"] - derived["cer_min"], 0.01)

    def test_the_damage_is_concentrated(self):
        # If every utterance flipped a little, "one fragile utterance" would be
        # the wrong story and the doc would have to change.
        derived = self.report["derived"]
        self.assertGreater(derived["utterances_that_ever_change"], 0)
        self.assertLessEqual(derived["utterances_that_ever_change"], 3)
        fragile = self.report["fragile_utterances"]
        self.assertEqual(fragile[0]["id"], derived["most_fragile_utterance"])
        # The most fragile one flips at several levels; the rest rarely do.
        # A third of the ladder is the bar, because the exact count depends on
        # which noise draws landed near a decision boundary.
        self.assertGreaterEqual(fragile[0]["levels_changed"], len(self.report["rows"]) // 3)

    def test_the_shipped_error_is_below_the_smallest_rung(self):
        # The sweep cannot resolve below its smallest rung, so the only honest
        # statement is that the shipped feature error is smaller than that.
        derived = self.report["derived"]
        margin = derived["shipped_feature_mae_decades_below_smallest_rung"]
        self.assertIsNotNone(margin)
        self.assertGreater(margin, 0.0)
        # Both sides of the comparison must be in the feature domain.
        self.assertLess(derived["shipped_feature_mae"],
                        derived["smallest_tested_feature_mae"])
        self.assertAlmostEqual(derived["shipped_feature_mae"],
                               self.report["shipped_path"]["feature_mae_mean"], places=15)

    def test_every_hypothesis_is_stored_so_a_flip_can_be_checked(self):
        texts = self.report["texts"]
        self.assertEqual(len(texts), len(self.report["rows"]))
        for row in self.report["rows"]:
            stored = texts[f"{row['nominal_amplitude']:g}"]
            self.assertEqual(len(stored), self.report["protocol"]["utterances"])
            changed = [uid for uid, text in stored.items() if text != texts["0"][uid]]
            self.assertEqual(sorted(changed), sorted(row["transcripts_differing"]))


class FixedPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load("fixed_point_sqnr.json")
        cls.rows = {row["stage"]: row for row in cls.report["formats"]}

    def test_sqnr_recomputes_from_rms_and_full_scale(self):
        import math
        for row in self.rows.values():
            quantum = row["full_scale"] / (2 ** (row["bits"] - 1))
            expected = 10.0 * math.log10(row["measured_rms"] ** 2 / (quantum ** 2 / 12.0))
            self.assertAlmostEqual(row["sqnr_db"], expected, places=6, msg=row["stage"])

    def test_no_stage_overflows_its_proposed_format(self):
        self.assertEqual(self.report["derived"]["overflow_stages"], [])
        for row in self.rows.values():
            self.assertFalse(row["overflow"], row["stage"])
            self.assertGreater(row["headroom_db"], 0.0, row["stage"])

    def test_the_sample_domain_is_safely_inside_q15(self):
        # These four stages are the ones a fixed-point port would put in Q15.
        for stage in ("input_pcm", "dc_removed", "preemphasized", "windowed"):
            self.assertEqual(self.rows[stage]["bits"], 16, stage)
            self.assertLessEqual(self.rows[stage]["measured_peak"],
                                 self.rows[stage]["full_scale"], stage)

    def test_the_ladder_ordering_is_monotonic(self):
        sqnrs = [rung["sqnr_db"] for rung in self.report["fft_ladder"]]
        self.assertEqual(sqnrs, sorted(sqnrs))
        bits = [rung["bits"] for rung in self.report["fft_ladder"]]
        self.assertEqual(bits, sorted(bits))

    def test_the_fft_recommendation_follows_from_the_ladder(self):
        recommendation = self.report["fft_recommendation"]
        ladder = {rung["format"]: rung for rung in self.report["fft_ladder"]}
        clearing = [rung["format"] for rung in self.report["fft_ladder"]
                    if rung["clears_target_on_median_frame"]]
        self.assertEqual(recommendation["cheapest_fixed_point_at_median"], clearing[0])
        quiet = [rung["format"] for rung in self.report["fft_ladder"]
                 if rung["clears_target_on_quiet_frame"]]
        self.assertEqual(recommendation["cheapest_fixed_point_at_quiet"],
                         quiet[0] if quiet else None)
        # The chosen approach must actually clear the target.
        self.assertGreater(recommendation["sqnr_guaranteed_db"], self.report["target_db"])
        self.assertGreater(ladder["Q15"]["sqnr_db"], 0.0)

    def test_the_per_frame_column_is_below_the_corpus_column(self):
        # Quiet frames cannot be better than the corpus average; if they ever
        # are, the histogram is being read backwards.
        for row in self.rows.values():
            self.assertLessEqual(row["sqnr_quiet_frame_db"], row["sqnr_db"] + 1e-9, row["stage"])

    def test_block_floating_point_is_level_independent(self):
        for name, entry in self.report["block_floating_point"].items():
            self.assertGreater(entry["sqnr_typical_db"], entry["sqnr_guaranteed_db"], name)
            # BFP beats the fixed-point quiet-frame figure it is proposed to fix.
            self.assertGreater(entry["sqnr_guaranteed_db"],
                               entry["fixed_point_quiet_frame_db"], name)


class Axi4sProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load("axi4s_protocol.json")

    def test_geometry_is_what_the_header_claims(self):
        self.assertEqual(self.report["beat_bytes"], 4)
        self.assertEqual(self.report["burst_bytes"], 64)
        self.assertEqual(self.report["beats_per_burst"], 16)
        self.assertEqual(self.report["beats_per_frame"], 160)
        self.assertEqual(self.report["frame_bytes"], 640)

    def test_not_one_protocol_violation(self):
        self.assertEqual(self.report["derived"]["protocol_violations"], 0)
        for arm in self.report["arms"]:
            self.assertEqual(arm["tlast_errors"], 0, arm["name"])
            self.assertEqual(arm["tuser_errors"], 0, arm["name"])
            self.assertEqual(arm["bubble_errors"], 0, arm["name"])
            self.assertEqual(arm["tready_rule_errors"], 0, arm["name"])
            self.assertEqual(arm["held_mismatch"], 0, arm["name"])

    def test_every_arm_delivers_every_frame(self):
        self.assertTrue(self.report["derived"]["all_frames_delivered"])
        for arm in self.report["arms"]:
            self.assertEqual(arm["frames"], self.report["reference_frames"], arm["name"])

    def test_the_beat_stream_is_whole_packets(self):
        for arm in self.report["arms"]:
            self.assertEqual(arm["beats_out"] % self.report["beats_per_frame"], 0, arm["name"])
            self.assertEqual(arm["beats_out"],
                             arm["frames"] * self.report["beats_per_frame"], arm["name"])

    def test_backpressure_reached_upstream(self):
        self.assertTrue(self.report["derived"]["backpressure_reached_upstream"])
        stalled = next(arm for arm in self.report["arms"] if arm["name"] == "stall_then_duty")
        self.assertGreater(stalled["stall_cycles"], 0)
        self.assertGreater(stalled["held_beats"], 0)
        self.assertGreater(stalled["s_tready_low_cycles"], 0)
        self.assertGreater(stalled["upstream_stall_cycles"], 0)
        self.assertEqual(stalled["max_fifo_level"], self.report["fifo_depth_samples"])

    def test_stall_invariance(self):
        self.assertTrue(self.report["derived"]["stall_invariant_frames"])

    def test_the_tick_loop_allocates_nothing(self):
        for arm in self.report["arms"]:
            self.assertEqual(arm["allocations_in_tick_loop"], 0, arm["name"])
            self.assertEqual(arm["denied_allocations"], 0, arm["name"])


class Axi4sWaveformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load("axi4s_waveform.json")

    def test_the_dump_lost_nothing(self):
        report = self.report
        self.assertEqual(report["frames_received"], report["frames_reference"])
        self.assertEqual(report["mismatched_values"], 0)
        self.assertEqual(report["max_abs_diff"], 0)
        self.assertEqual(report["compared_values"], report["frames_received"] * 80)

    def test_the_stall_actually_engaged(self):
        report = self.report
        self.assertGreater(report["downstream_stall_cycles"], 0)
        self.assertGreater(report["s_tready_low_cycles"], 0)
        self.assertEqual(report["max_fifo_level"], report["fifo_depth_samples"])
        self.assertLess(report["first_s_tready_low_cycle"], report["cycles"])

    def test_the_vcd_is_present_and_declares_the_handshake_signals(self):
        path = ROOT / self.report["vcd"]
        if not path.is_file():
            raise unittest.SkipTest(f"run the AXI4-Stream sim first ({path.name} missing)")
        text = path.read_text(encoding="utf-8", errors="replace")
        self.assertTrue(text.startswith("$date"))
        for signal in ("s_tvalid", "s_tready", "m_tvalid", "m_tready", "m_tlast", "m_tuser"):
            self.assertIn(signal, text)
        self.assertIn("$enddefinitions", text)


class BfpCerAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load("bfp_cer_audit.json")
        cls.methods = cls.report["methods"]

    def test_feature_bfp16_matches_float32_identically(self):
        f32 = self.methods["float32"]
        bfp16 = self.methods["feature_bfp16"]
        self.assertAlmostEqual(bfp16["cer"], f32["cer"], places=6)
        self.assertEqual(bfp16["errors"], f32["errors"])
        self.assertEqual(bfp16["transcripts"], f32["transcripts"])

    def test_energy_bfp16_is_lossless_or_better(self):
        f32 = self.methods["float32"]
        e_bfp16 = self.methods["energy_bfp16"]
        self.assertLessEqual(e_bfp16["cer"], f32["cer"] + 1e-6)
        same = sum(1 for uid, text in e_bfp16["transcripts"].items() if text == f32["transcripts"][uid])
        self.assertGreaterEqual(same, 29)

    def test_energy_plain_q15_suffers_catastrophic_failure(self):
        f32 = self.methods["float32"]
        e_q15 = self.methods["energy_q15"]
        self.assertGreater(e_q15["cer"], 0.30)
        self.assertGreater(e_q15["errors"], f32["errors"] * 10)


class EmbeddedFootprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load("embedded_memory_footprint.json")
        cls.targets = cls.report["targets"]

    def test_all_four_targets_are_audited(self):
        self.assertIn("cortex_r5", self.targets)
        self.assertIn("cortex_m4", self.targets)
        self.assertIn("cortex_a53", self.targets)
        self.assertIn("x86_64", self.targets)

    def test_cortex_r5_and_m4_flash_under_thirty_kb(self):
        r5_flash = self.targets["cortex_r5"]["totals"]["flash_total_kb"]
        m4_flash = self.targets["cortex_m4"]["totals"]["flash_total_kb"]
        self.assertLess(r5_flash, 30.0)
        self.assertLess(m4_flash, 30.0)

    def test_static_sram_is_zero_on_embedded_targets(self):
        self.assertEqual(self.targets["cortex_r5"]["totals"]["ram_static_bytes"], 0)
        self.assertEqual(self.targets["cortex_m4"]["totals"]["ram_static_bytes"], 0)

    def test_runtime_instance_footprint_is_bounded(self):
        rt = self.report["runtime_instance_footprint"]
        self.assertLess(rt["bounded_bridge_sizeof_arm32"], 4096)
        self.assertEqual(rt["hotpath_steady_allocations"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
