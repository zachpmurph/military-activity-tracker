"""
Tests for intelligence/calibration.py

Covers:
  - _pearson()                      – Pearson correlation edge cases
  - _feature_vector_map()           – key/value consistency with _PROFILES
  - evaluate()                      – all metric computations (incl. signal fields)
  - _compute_violations()           – each threshold check; signal gating
  - detect_failures()               – each of the six rules; signal gating
  - propose_adjustments()           – delta bounded; dominant feature reduced; redistribution
  - apply_adjustments()             – profiles updated; sums preserved
  - run_calibration_loop()          – terminates; gating; stagnation guard
  - diff_profiles()                 – change summary format
  - SignalSufficiencyTests          – compute_signal_sufficiency / compute_input_diversity
  - SignalGatingTests               – legitimate vs pathological clustering
  - LowSignalEnvironmentTests       – end-to-end: ROUTINE batch skips calibration
  - ANOMALY score-cluster mode      – the described aircraft=3-5/spike/no-change-data collapse
  - FeatureCollapseAnomalyTests     – FEATURE_COLLAPSE_ANOMALY rule
  - CheckConvergenceTests           – narrowed convergence: only 4 critical violations block
"""

import copy
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import intelligence.classifier as _cls_module
from intelligence.classifier import (
    Classification,
    RegionFeatures,
    RegionIntelligence,
    build_features,
    classify_regions,
)
from intelligence.calibration import (
    THRESHOLDS,
    CONVERGENCE_CRITERIA,
    _CONVERGENCE_VIOLATION_PREFIXES,
    _CONVERGENCE_VIOLATION_PREFIX_STR,
    CalibrationReport,
    CalibrationFailure,
    IterationRecord,
    WeightAdjustment,
    _pearson,
    _feature_vector_map,
    compute_signal_sufficiency,
    compute_input_diversity,
    compute_feature_contributions,
    evaluate,
    detect_failures,
    propose_adjustments,
    apply_adjustments,
    check_convergence,
    run_calibration_loop,
    diff_profiles,
)


# ─── shared helpers ───────────────────────────────────────────────────────────

_ALL_CLS_NAMES = {c.value for c in Classification}


def _make_intel(
    cls: Classification,
    score: float,
    confidence: int,
    *,
    all_scores: dict | None = None,
    aircraft_count: int = 10,
    lat: float = 0.0,
    lon: float = 0.0,
    spike_flag: bool = False,
    change_score: float = 0.0,
    inflow_count: int = 0,
) -> RegionIntelligence:
    """
    Construct a RegionIntelligence directly (bypassing classify_regions) so
    that all_scores, confidence, and aircraft_count are exactly controlled.
    """
    if all_scores is None:
        runner_up = max(score - 20.0, 0.0)
        all_scores = {c.value: runner_up for c in Classification}
        all_scores[cls.value] = score
    features = RegionFeatures(
        lat=lat,
        lon=lon,
        aircraft_count=aircraft_count,
        spike_flag=spike_flag,
        change_score=change_score,
        inflow_count=inflow_count,
    )
    return RegionIntelligence(
        lat=lat,
        lon=lon,
        classification=cls,
        confidence=confidence,
        score=score,
        all_scores=all_scores,
        features=features,
        explanation="test",
    )


def _tight_all_scores(winner: Classification, score: float, gap: float) -> dict:
    """
    Build an all_scores dict where:
      winner  → score
      first non-winner → score - gap        (the runner-up)
      all others       → score - gap - 1.0
    """
    scores = {c.value: score - gap - 1.0 for c in Classification}
    scores[winner.value] = score
    runner = next(c for c in Classification if c != winner)
    scores[runner.value] = score - gap
    return scores


# ─── _pearson ─────────────────────────────────────────────────────────────────

class PearsonTests(unittest.TestCase):

    def test_perfect_positive_correlation(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertAlmostEqual(_pearson(xs, ys), 1.0, places=9)

    def test_perfect_negative_correlation(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [50.0, 40.0, 30.0, 20.0, 10.0]
        self.assertAlmostEqual(_pearson(xs, ys), -1.0, places=9)

    def test_zero_variance_y_returns_zero(self):
        # denominator → 0 when y is constant
        self.assertEqual(_pearson([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]), 0.0)

    def test_too_few_points_returns_zero(self):
        self.assertEqual(_pearson([], []), 0.0)
        self.assertEqual(_pearson([1.0], [2.0]), 0.0)
        self.assertEqual(_pearson([1.0, 2.0], [3.0, 4.0]), 0.0)

    def test_three_point_minimum_works(self):
        r = _pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(r, 1.0, places=9)

    def test_result_in_minus_one_to_one(self):
        import random
        random.seed(42)
        xs = [random.random() for _ in range(20)]
        ys = [random.random() for _ in range(20)]
        r = _pearson(xs, ys)
        self.assertGreaterEqual(r, -1.0)
        self.assertLessEqual(r, 1.0)


# ─── _feature_vector_map ──────────────────────────────────────────────────────

class FeatureVectorMapTests(unittest.TestCase):
    """
    _feature_vector_map must mirror _score_region exactly so that
    _compute_feature_dominance produces accurate fractions.
    """

    def _default_context(self):
        return {"max_inflow": 5, "max_change": 40.0}

    def test_all_profile_features_present_in_map(self):
        """Every key used in any _PROFILES weight dict must appear in fv."""
        f = RegionFeatures(lat=0, lon=0, aircraft_count=10)
        fv = _feature_vector_map(f, self._default_context())
        all_profile_features: set = set()
        for profile in _cls_module._PROFILES.values():
            all_profile_features |= set(profile.keys())
        missing = all_profile_features - set(fv.keys())
        self.assertEqual(
            missing, set(),
            f"Feature(s) in _PROFILES missing from _feature_vector_map: {missing}",
        )

    def test_all_values_in_unit_range(self):
        """Every feature value must be in [0, 1]."""
        f = RegionFeatures(
            lat=0, lon=0, aircraft_count=15,
            spike_flag=True, coordination_flag=True,
            inflow_count=3, outflow_count=2,
            military_count=5, new_aircraft=3,
            change_score=20.0,
        )
        fv = _feature_vector_map(f, self._default_context())
        for name, val in fv.items():
            self.assertGreaterEqual(val, 0.0, f"{name}={val} below 0")
            self.assertLessEqual(val, 1.0, f"{name}={val} above 1")

    def test_spike_flag_reflects_bool(self):
        f_spike = RegionFeatures(lat=0, lon=0, aircraft_count=5, spike_flag=True)
        f_quiet = RegionFeatures(lat=0, lon=0, aircraft_count=5, spike_flag=False)
        ctx = self._default_context()
        self.assertEqual(_feature_vector_map(f_spike, ctx)["spike_flag"], 1.0)
        self.assertEqual(_feature_vector_map(f_quiet, ctx)["spike_flag"], 0.0)
        self.assertEqual(_feature_vector_map(f_spike, ctx)["no_spike"],   0.0)
        self.assertEqual(_feature_vector_map(f_quiet, ctx)["no_spike"],   1.0)

    def test_inflow_norm_capped_at_one(self):
        # inflow_count > max_inflow → should cap at 1.0
        f = RegionFeatures(lat=0, lon=0, aircraft_count=5, inflow_count=100)
        fv = _feature_vector_map(f, {"max_inflow": 10, "max_change": 40.0})
        self.assertLessEqual(fv["inflow_norm"], 1.0)


# ─── evaluate ─────────────────────────────────────────────────────────────────

class EvaluateTests(unittest.TestCase):

    def test_empty_list_returns_zeroed_report(self):
        report = evaluate([])
        self.assertEqual(report.n_regions, 0)
        self.assertEqual(report.mean_margin, 0.0)
        self.assertEqual(report.low_margin_fraction, 0.0)
        self.assertEqual(report.max_cluster_fraction, 0.0)

    def test_n_regions_correct(self):
        regions = [_make_intel(Classification.STAGING, 80, 70, lat=float(i))
                   for i in range(7)]
        self.assertEqual(evaluate(regions).n_regions, 7)

    def test_mean_margin_computed_correctly(self):
        # rA: STAGING=70, runner-up=50 → margin=20
        # rB: ROUTINE=60, runner-up=55 → margin=5  → mean=12.5
        rA = _make_intel(Classification.STAGING, 70, 60, all_scores={
            "STAGING": 70, "PROJECTION": 50, "ROUTINE": 45,
            "ANOMALY": 40, "COORDINATED_ACTIVITY": 35,
        })
        rB = _make_intel(Classification.ROUTINE, 60, 40, all_scores={
            "STAGING": 30, "PROJECTION": 30, "ROUTINE": 60,
            "ANOMALY": 55, "COORDINATED_ACTIVITY": 20,
        })
        report = evaluate([rA, rB])
        self.assertAlmostEqual(report.mean_margin, 12.5, places=1)

    def test_low_margin_fraction_all_below_threshold(self):
        # All 3 regions have margin < 10 → fraction=1.0
        regions = [
            _make_intel(Classification.ANOMALY, 50, 20,
                        all_scores=_tight_all_scores(Classification.ANOMALY, 50, 2.0),
                        lat=float(i))
            for i in range(3)
        ]
        report = evaluate(regions)
        self.assertAlmostEqual(report.low_margin_fraction, 1.0, places=4)

    def test_score_cluster_fraction_reflects_identical_buckets(self):
        # 5 ANOMALY regions all with score=75 → one bucket owns 100%
        regions = [
            _make_intel(Classification.ANOMALY, 75, 30, all_scores={
                "STAGING": 30, "PROJECTION": 30, "ROUTINE": 30,
                "ANOMALY": 75, "COORDINATED_ACTIVITY": 30,
            }, lat=float(i))
            for i in range(5)
        ]
        report = evaluate(regions)
        self.assertAlmostEqual(report.max_cluster_fraction, 1.0, places=4)

    def test_class_fractions_sum_to_one(self):
        regions = [
            _make_intel(Classification.STAGING,    80, 70, lat=0),
            _make_intel(Classification.ROUTINE,    70, 50, lat=1),
            _make_intel(Classification.ANOMALY,    60, 40, lat=2),
        ]
        report = evaluate(regions)
        # Each fraction is rounded to 4 decimal places; three thirds sum to 0.9999,
        # so we only require 3 places of precision here.
        self.assertAlmostEqual(sum(report.class_fractions.values()), 1.0, places=3)

    def test_class_fractions_keys_cover_all_classifications(self):
        regions = [_make_intel(Classification.STAGING, 80, 70)]
        report = evaluate(regions)
        self.assertEqual(set(report.class_fractions.keys()), _ALL_CLS_NAMES)

    def test_dominant_class_identified(self):
        regions = (
            [_make_intel(Classification.ANOMALY, 75, 40, lat=float(i)) for i in range(4)]
            + [_make_intel(Classification.STAGING, 80, 60, lat=10.0)]
        )
        report = evaluate(regions)
        self.assertEqual(report.dominant_class, "ANOMALY")
        self.assertAlmostEqual(report.dominant_class_fraction, 4 / 5, places=4)

    def test_conf_margin_corr_positive_when_aligned(self):
        """Confidence increases with margin → positive Pearson r."""
        def _r(score, runner_up_gap, conf):
            return _make_intel(Classification.STAGING, score, conf, all_scores={
                "STAGING": score, "PROJECTION": score - runner_up_gap,
                "ROUTINE": score - runner_up_gap - 5,
                "ANOMALY": score - runner_up_gap - 10,
                "COORDINATED_ACTIVITY": score - runner_up_gap - 15,
            })
        regions = [
            _r(80, 40, 80),   # margin=40, conf=80
            _r(60, 20, 50),   # margin=20, conf=50
            _r(50,  3, 20),   # margin=3,  conf=20
        ]
        report = evaluate(regions)
        self.assertGreater(report.conf_margin_corr, 0.5)

    def test_small_data_high_conf_fraction(self):
        # aircraft_count=3 + confidence=80 → triggers (both below/above caps)
        r_trigger = _make_intel(Classification.ANOMALY, 75, 80, aircraft_count=3)
        r_safe    = _make_intel(Classification.ROUTINE,  70, 40, aircraft_count=20, lat=1)
        report = evaluate([r_trigger, r_safe])
        self.assertAlmostEqual(report.small_data_high_conf_fraction, 0.5, places=4)

    def test_score_std_computed_for_multi_member_class(self):
        # Population std of [60, 90] = 15
        regions = [
            _make_intel(Classification.ANOMALY, 60, 30, lat=0),
            _make_intel(Classification.ANOMALY, 90, 60, lat=1),
        ]
        report = evaluate(regions)
        self.assertAlmostEqual(report.score_std_by_class["ANOMALY"], 15.0, places=1)

    def test_score_std_sentinel_for_single_member(self):
        r = _make_intel(Classification.STAGING, 80, 70)
        report = evaluate([r])
        # Classes with only one member receive the -1.0 sentinel
        self.assertEqual(report.score_std_by_class["STAGING"], -1.0)

    def test_feature_dominance_keys_match_all_classes(self):
        regions = [_make_intel(Classification.STAGING, 80, 70) for _ in range(3)]
        report = evaluate(regions)
        self.assertEqual(set(report.feature_dominance.keys()), _ALL_CLS_NAMES)

    def test_violations_populated_in_post_init(self):
        # Signal-rich low-margin regions → mean_margin & low_margin_fraction
        # must both fire.  spike_flag=True puts signal_sufficiency at 0.35 > 0.10
        # so the gate opens and the gated violations are active.
        regions = [
            _make_intel(Classification.STAGING, 50, 20,
                        all_scores=_tight_all_scores(Classification.STAGING, 50, 1.0),
                        spike_flag=True,
                        lat=float(i))
            for i in range(4)
        ]
        report = evaluate(regions)
        self.assertIn("mean_margin", report.violations)
        self.assertIn("low_margin_fraction", report.violations)


# ─── violation detection ───────────────────────────────────────────────────────

class ViolationDetectionTests(unittest.TestCase):

    def test_mean_margin_violation(self):
        # All margins ≈ 1.0 → mean << 15
        regions = [
            _make_intel(Classification.STAGING, 50, 20,
                        all_scores=_tight_all_scores(Classification.STAGING, 50, 1.0),
                        lat=float(i))
            for i in range(5)
        ]
        self.assertIn("mean_margin", evaluate(regions).violations)

    def test_score_cluster_violation(self):
        # Signal-rich ANOMALY@75 cluster → score_cluster must fire.
        # spike_flag=True → signal_sufficiency=0.35 > 0.10, so is_low_signal=False
        # → is_legitimate_clustering=False → gate opens → violation fires.
        regions = [
            _make_intel(Classification.ANOMALY, 75, 30, all_scores={
                "STAGING": 30, "PROJECTION": 30, "ROUTINE": 30,
                "ANOMALY": 75, "COORDINATED_ACTIVITY": 30,
            }, spike_flag=True, lat=float(i))
            for i in range(6)
        ]
        self.assertIn("score_cluster", evaluate(regions).violations)

    def test_class_skew_violation(self):
        # 5/6 = 83% ANOMALY → exceeds 55% cap
        regions = (
            [_make_intel(Classification.ANOMALY, 75, 40, lat=float(i)) for i in range(5)]
            + [_make_intel(Classification.STAGING, 80, 60, lat=10.0)]
        )
        self.assertIn("class_skew", evaluate(regions).violations)

    def test_low_margin_fraction_violation(self):
        # Signal-rich regions with tight margins → fraction=1.0 > 0.10 threshold.
        # spike_flag=True keeps signal_sufficiency above the gate threshold.
        regions = [
            _make_intel(Classification.ANOMALY, 50, 20,
                        all_scores=_tight_all_scores(Classification.ANOMALY, 50, 2.0),
                        spike_flag=True,
                        lat=float(i))
            for i in range(4)
        ]
        self.assertIn("low_margin_fraction", evaluate(regions).violations)

    def test_is_converged_returns_false_with_violations(self):
        r = _make_intel(Classification.STAGING, 80, 70)
        report = evaluate([r])
        # At minimum, conf_margin_corr is 0 (n<3 → pearson returns 0) → violation
        self.assertFalse(report.is_converged())

    def test_summary_string_contains_status_word(self):
        r = _make_intel(Classification.STAGING, 80, 70)
        text = evaluate([r]).summary()
        self.assertTrue("VIOLATIONS" in text or "CONVERGED" in text)

    def test_check_convergence_secondary_violation_does_not_block(self):
        """
        check_convergence() only cares about the 4 critical violations.
        A report with only secondary violations (class_skew) must return True
        even though is_converged() is False.
        """
        r = _make_intel(Classification.STAGING, 80, 70)
        report = evaluate([r])
        # Inject a secondary-only violation set to isolate the behaviour
        report.violations = ["class_skew"]
        self.assertFalse(report.is_converged(),  "Sanity: is_converged sees the violation")
        self.assertTrue(check_convergence(report), "check_convergence must ignore secondary violations")


# ─── detect_failures ──────────────────────────────────────────────────────────

class DetectFailuresTests(unittest.TestCase):

    def test_empty_returns_empty(self):
        self.assertEqual(detect_failures([]), [])

    def test_low_margin_rule_fires(self):
        # spike_flag=True → signal_sufficiency=0.35 > 0.10 → gate opens
        r = _make_intel(
            Classification.STAGING, 50, 30,
            all_scores=_tight_all_scores(Classification.STAGING, 50, 2.0),
            spike_flag=True,
        )
        rules = {f.rule for f in detect_failures([r])}
        self.assertIn("LOW_MARGIN", rules)

    def test_score_cluster_rule_fires(self):
        # Signal-rich: spike_flag=True → sufficiency=0.35 > 0.10 → not low-signal
        # → not legitimate clustering → SCORE_CLUSTER fires.
        # 10 ANOMALY@75 → bucket count 10 ≥ max(3, 10*0.20=2) = 3
        regions = [
            _make_intel(Classification.ANOMALY, 75, 30, all_scores={
                "STAGING": 30, "PROJECTION": 30, "ROUTINE": 30,
                "ANOMALY": 75, "COORDINATED_ACTIVITY": 30,
            }, spike_flag=True, lat=float(i))
            for i in range(10)
        ]
        rules = {f.rule for f in detect_failures(regions)}
        self.assertIn("SCORE_CLUSTER", rules)

    def test_high_conf_low_data_rule_fires(self):
        r = _make_intel(
            Classification.ANOMALY, 75, 80,
            aircraft_count=3,          # ≤ small_data_cap(5)
        )
        rules = {f.rule for f in detect_failures([r])}
        self.assertIn("HIGH_CONF_LOW_DATA", rules)

    def test_high_conf_low_data_rule_does_not_fire_for_large_fleet(self):
        r = _make_intel(Classification.ANOMALY, 75, 80, aircraft_count=50)
        rules = {f.rule for f in detect_failures([r])}
        self.assertNotIn("HIGH_CONF_LOW_DATA", rules)

    def test_class_dominance_rule_fires(self):
        # 5/6 = 83% ANOMALY → exceeds 55% threshold
        regions = (
            [_make_intel(Classification.ANOMALY, 75, 40, lat=float(i)) for i in range(5)]
            + [_make_intel(Classification.STAGING, 80, 60, lat=10.0)]
        )
        rules = {f.rule for f in detect_failures(regions)}
        self.assertIn("CLASS_DOMINANCE", rules)

    def test_anomaly_overfit_rule_fires(self):
        # 4 small ANOMALY + 1 small STAGING → 4/5 = 80% of small regions are ANOMALY
        regions = (
            [_make_intel(Classification.ANOMALY, 75, 30,
                         aircraft_count=4, lat=float(i)) for i in range(4)]
            + [_make_intel(Classification.STAGING, 80, 60, aircraft_count=4, lat=10.0)]
        )
        rules = {f.rule for f in detect_failures(regions)}
        self.assertIn("ANOMALY_OVERFIT", rules)

    def test_anomaly_overfit_rule_does_not_fire_below_threshold(self):
        # Only 1 ANOMALY out of 5 small-volume regions → 20% < 40% threshold
        regions = (
            [_make_intel(Classification.STAGING, 80, 60,
                         aircraft_count=4, lat=float(i)) for i in range(4)]
            + [_make_intel(Classification.ANOMALY, 75, 30, aircraft_count=4, lat=10.0)]
        )
        rules = {f.rule for f in detect_failures(regions)}
        self.assertNotIn("ANOMALY_OVERFIT", rules)

    def test_failures_sorted_by_severity_descending(self):
        regions = (
            [_make_intel(Classification.ANOMALY, 75, 40, lat=float(i)) for i in range(5)]
            + [_make_intel(Classification.STAGING, 80, 60, lat=10.0)]
        )
        failures = detect_failures(regions)
        sevs = [f.severity for f in failures]
        self.assertEqual(sevs, sorted(sevs, reverse=True))

    def test_every_failure_has_non_empty_detail_and_fix(self):
        r = _make_intel(
            Classification.STAGING, 50, 20,
            all_scores=_tight_all_scores(Classification.STAGING, 50, 1.0),
        )
        for f in detect_failures([r]):
            self.assertIsInstance(f.detail, str)
            self.assertTrue(f.detail.strip(), "detail must not be blank")
            self.assertIsInstance(f.suggested_fix, str)
            self.assertTrue(f.suggested_fix.strip(), "suggested_fix must not be blank")

    def test_severity_in_zero_to_one(self):
        regions = (
            [_make_intel(Classification.ANOMALY, 75, 40, lat=float(i)) for i in range(5)]
            + [_make_intel(Classification.STAGING, 80, 60, lat=10.0)]
        )
        for f in detect_failures(regions):
            self.assertGreaterEqual(f.severity, 0.0)
            self.assertLessEqual(f.severity, 1.0)


# ─── propose_adjustments ──────────────────────────────────────────────────────

class ProposeAdjustmentsTests(unittest.TestCase):
    """
    Use a custom ANOMALY profile with spike_flag=0.70 to guarantee
    feature dominance, then verify adjustment is bounded and correctly targeted.
    """

    def _dominant_profiles(self) -> dict:
        profiles = copy.deepcopy(_cls_module._PROFILES)
        profiles[Classification.ANOMALY] = {
            "spike_flag":        0.70,
            "novelty":           0.10,
            "low_persistence":   0.07,
            "change_score_norm": 0.07,
            "unmilitary":        0.06,
        }
        return profiles

    def _classify_with(self, profiles: dict) -> list:
        """Classify a set of pure-spike regions using a monkey-patched profile."""
        features = build_features(
            spikes=[
                (50.0, 10.0, 6, 0),
                (51.0, 10.0, 5, 0),
                (52.0, 10.0, 7, 0),
            ]
        )
        original = _cls_module._PROFILES
        _cls_module._PROFILES = profiles
        try:
            return classify_regions(features, min_aircraft=1)
        finally:
            _cls_module._PROFILES = original

    def test_adjustment_proposed_for_dominant_feature(self):
        profiles    = self._dominant_profiles()
        results     = self._classify_with(profiles)
        report      = evaluate(results, profiles)
        adjustments = propose_adjustments(results, profiles, report)
        cls_names   = {adj.classification for adj in adjustments}
        self.assertIn("ANOMALY", cls_names)

    def test_delta_bounded_by_max_weight_delta(self):
        # FEATURE_DOMINANCE_IMBALANCE adjustments use ×0.9 (a different strategy)
        # and are NOT subject to max_weight_delta.  Only the legacy
        # max_feature_dominance (0.40–0.60 range) adjustments are bounded.
        # _dominant_profiles() puts spike_flag=0.70 which triggers the ×0.9 path,
        # so we use a profile with spike_flag=0.50 (between 0.40 and 0.60) here.
        profiles = copy.deepcopy(_cls_module._PROFILES)
        profiles[Classification.ANOMALY] = {
            "spike_flag":        0.50,
            "novelty":           0.15,
            "low_persistence":   0.12,
            "change_score_norm": 0.13,
            "unmilitary":        0.10,
        }
        results     = self._classify_with(profiles)
        report      = evaluate(results, profiles)
        adjustments = propose_adjustments(results, profiles, report)
        legacy_adjs = [a for a in adjustments
                       if "FEATURE_DOMINANCE_IMBALANCE" not in a.reason]
        for adj in legacy_adjs:
            self.assertLessEqual(
                abs(adj.delta),
                THRESHOLDS["max_weight_delta"] + 1e-9,
                f"{adj.feature}: |delta|={abs(adj.delta)} exceeds max_weight_delta",
            )

    def test_dominant_feature_weight_reduced(self):
        profiles    = self._dominant_profiles()
        results     = self._classify_with(profiles)
        report      = evaluate(results, profiles)
        adjustments = propose_adjustments(results, profiles, report)
        spike_adj = next(
            (a for a in adjustments
             if a.classification == "ANOMALY" and a.feature == "spike_flag"),
            None,
        )
        self.assertIsNotNone(spike_adj, "Expected an adjustment targeting ANOMALY.spike_flag")
        self.assertLess(spike_adj.new_weight, spike_adj.old_weight)

    def test_redistribution_adjustments_positive(self):
        # Recipients of redistributed weight should have delta > 0
        profiles    = self._dominant_profiles()
        results     = self._classify_with(profiles)
        report      = evaluate(results, profiles)
        adjustments = propose_adjustments(results, profiles, report)
        pos_adjs = [a for a in adjustments if a.delta > 0]
        self.assertGreater(len(pos_adjs), 0, "Expected at least one receiving adjustment")

    def test_no_exception_when_no_dominant_feature(self):
        # Healthy fleet: diverse classes, large margins
        regions = [
            _make_intel(Classification.STAGING,    90, 80, lat=0),
            _make_intel(Classification.PROJECTION, 85, 75, lat=1),
            _make_intel(Classification.ROUTINE,    80, 70, lat=2),
        ]
        report = evaluate(regions)
        # Must not raise even if no features dominate
        adjustments = propose_adjustments(regions, report=report)
        for adj in adjustments:
            self.assertLessEqual(abs(adj.delta), THRESHOLDS["max_weight_delta"] + 1e-9)


# ─── apply_adjustments ────────────────────────────────────────────────────────

class ApplyAdjustmentsTests(unittest.TestCase):

    def test_profiles_sum_to_one_after_application(self):
        # Cut spike_flag by 0.05, add it back to novelty
        adjs = [
            WeightAdjustment("ANOMALY", "spike_flag", 0.30, 0.25, -0.05, "cut"),
            WeightAdjustment("ANOMALY", "novelty",    0.20, 0.25, +0.05, "add"),
        ]
        new_profiles = apply_adjustments(adjs)
        for cls_enum, profile in new_profiles.items():
            self.assertAlmostEqual(sum(profile.values()), 1.0, places=5,
                                   msg=f"{cls_enum.value} weights do not sum to 1.0")

    def test_targeted_weight_is_changed(self):
        adjs = [WeightAdjustment("ANOMALY", "spike_flag", 0.30, 0.22, -0.08, "test")]
        new_profiles = apply_adjustments(adjs)
        # After renormalisation spike_flag should be less than original 0.30
        self.assertLess(new_profiles[Classification.ANOMALY]["spike_flag"], 0.30)

    def test_unchanged_profiles_preserved(self):
        # An adjustment to ANOMALY must not alter ROUTINE weights
        original_routine = dict(_cls_module._PROFILES[Classification.ROUTINE])
        adjs = [WeightAdjustment("ANOMALY", "spike_flag", 0.30, 0.25, -0.05, "test")]
        new_profiles = apply_adjustments(adjs)
        for feat, w in original_routine.items():
            self.assertAlmostEqual(
                new_profiles[Classification.ROUTINE][feat], w, places=5,
                msg=f"ROUTINE.{feat} was modified unexpectedly",
            )

    def test_unknown_class_name_is_silently_ignored(self):
        adjs = [WeightAdjustment("NONEXISTENT_CLASS", "spike_flag", 0.30, 0.22, -0.08, "test")]
        new_profiles = apply_adjustments(adjs)
        for cls_enum, profile in new_profiles.items():
            self.assertAlmostEqual(sum(profile.values()), 1.0, places=5)

    def test_weight_never_drops_below_minimum(self):
        # Apply a very large reduction; result should still be > 0 (clamped to 0.01)
        adjs = [WeightAdjustment("ANOMALY", "spike_flag", 0.30, 0.01, -0.29, "extreme")]
        new_profiles = apply_adjustments(adjs)
        self.assertGreater(new_profiles[Classification.ANOMALY]["spike_flag"], 0.0)


# ─── calibration loop ─────────────────────────────────────────────────────────

class CalibrationLoopTests(unittest.TestCase):

    def _spike_features(self, n: int = 5) -> list:
        return build_features(
            spikes=[(float(i), 10.0, 5, 0) for i in range(n)]
        )

    def test_terminates_within_max_iterations(self):
        _, history = run_calibration_loop(
            self._spike_features(5), max_iterations=5, min_aircraft=1, verbose=False,
        )
        self.assertLessEqual(len(history), 5)

    def test_final_profiles_have_correct_structure(self):
        final, _ = run_calibration_loop(
            self._spike_features(3), max_iterations=3, min_aircraft=1, verbose=False,
        )
        for cls in Classification:
            self.assertIn(cls, final,
                          f"{cls.value} missing from final profiles")
            self.assertAlmostEqual(sum(final[cls].values()), 1.0, places=5,
                                   msg=f"{cls.value} weights do not sum to 1.0")

    def test_history_records_correct_types(self):
        _, history = run_calibration_loop(
            self._spike_features(3), max_iterations=3, min_aircraft=1, verbose=False,
        )
        self.assertGreater(len(history), 0)
        for rec in history:
            self.assertIsInstance(rec, IterationRecord)
            self.assertIsInstance(rec.report, CalibrationReport)
            self.assertIsInstance(rec.converged, bool)
            self.assertIsInstance(rec.adjustments, list)

    def test_exits_early_on_empty_feature_list(self):
        """No results → no adjustments proposed → stops after 1 iteration."""
        _, history = run_calibration_loop(
            [], max_iterations=10, min_aircraft=1, verbose=False,
        )
        self.assertEqual(len(history), 1)

    def test_module_profiles_not_mutated(self):
        """
        The loop monkey-patches _cls_module._PROFILES temporarily;
        after it returns the module-level profiles must be restored.
        """
        before = {cls: dict(p) for cls, p in _cls_module._PROFILES.items()}
        run_calibration_loop(
            self._spike_features(3), max_iterations=3, min_aircraft=1, verbose=False,
        )
        for cls in Classification:
            for feat, w in before[cls].items():
                self.assertAlmostEqual(
                    _cls_module._PROFILES[cls][feat], w, places=9,
                    msg=f"Module profile {cls.value}.{feat} was permanently altered",
                )

    def test_caller_profiles_dict_not_mutated(self):
        """run_calibration_loop deep-copies the supplied profiles; the original must be unchanged."""
        original = copy.deepcopy(_cls_module._PROFILES)
        run_calibration_loop(
            self._spike_features(3), profiles=original,
            max_iterations=2, min_aircraft=1, verbose=False,
        )
        for cls in Classification:
            for feat in original[cls]:
                self.assertAlmostEqual(
                    original[cls][feat],
                    _cls_module._PROFILES[cls][feat],
                    places=6,
                    msg=f"Caller-supplied profile {cls.value}.{feat} was mutated",
                )

    def test_iteration_numbers_are_sequential(self):
        _, history = run_calibration_loop(
            self._spike_features(3), max_iterations=5, min_aircraft=1, verbose=False,
        )
        for i, rec in enumerate(history, start=1):
            self.assertEqual(rec.iteration, i)


# ─── diff_profiles ────────────────────────────────────────────────────────────

class DiffProfilesTests(unittest.TestCase):

    def test_identical_profiles_report_no_changes(self):
        p = copy.deepcopy(_cls_module._PROFILES)
        self.assertIn("no changes", diff_profiles(p, p))

    def test_changed_weight_appears_in_output(self):
        before = copy.deepcopy(_cls_module._PROFILES)
        after  = copy.deepcopy(_cls_module._PROFILES)
        after[Classification.ANOMALY]["spike_flag"] += 0.05
        after[Classification.ANOMALY]["novelty"]    -= 0.05
        result = diff_profiles(before, after)
        self.assertIn("ANOMALY",    result)
        self.assertIn("spike_flag", result)
        self.assertIn("novelty",    result)

    def test_output_is_string(self):
        before = copy.deepcopy(_cls_module._PROFILES)
        after  = copy.deepcopy(_cls_module._PROFILES)
        self.assertIsInstance(diff_profiles(before, after), str)

    def test_unchanged_class_not_in_output(self):
        before = copy.deepcopy(_cls_module._PROFILES)
        after  = copy.deepcopy(_cls_module._PROFILES)
        # Only change ANOMALY
        after[Classification.ANOMALY]["spike_flag"] += 0.05
        after[Classification.ANOMALY]["novelty"]    -= 0.05
        result = diff_profiles(before, after)
        self.assertNotIn("ROUTINE", result)
        self.assertNotIn("STAGING", result)


# ─── ANOMALY score-cluster failure mode ───────────────────────────────────────

class AnomalyScoreClusterTests(unittest.TestCase):
    """
    Reproduce the described production failure mode:
    aircraft_count=3–5, spike_flag=True, no persistent history, no change_score.
    All regions should land in a tight ANOMALY score band because:
      - change_score_norm=0  (wastes the 0.20 weight slot)
      - spike_flag drives ~38–40% of every ANOMALY score
    The calibration harness must detect this collapse.
    """

    def _uniform_spike_features(self) -> list:
        return build_features(
            spikes=[
                (55.0, 25.0, 3, 0),
                (56.0, 26.0, 4, 0),
                (57.0, 27.0, 5, 0),
                (58.0, 28.0, 3, 0),
                (59.0, 29.0, 4, 0),
            ]
        )

    def test_all_regions_classified_as_anomaly(self):
        features = self._uniform_spike_features()
        results  = classify_regions(features, min_aircraft=1)
        self.assertTrue(results, "Expected at least one classified region")
        for r in results:
            self.assertEqual(
                r.classification, Classification.ANOMALY,
                f"Expected ANOMALY but got {r.classification.value} for {r}",
            )

    def test_anomaly_scores_form_a_tight_cluster(self):
        """
        All five regions should produce the same ANOMALY score because the
        ANOMALY profile contains no aircraft_density term and all features
        (spike_flag, novelty, low_persistence, unmilitary) are identical.
        """
        features = self._uniform_spike_features()
        results  = classify_regions(features, min_aircraft=1)
        anomaly_scores = [r.score for r in results if r.classification == Classification.ANOMALY]
        self.assertGreater(len(anomaly_scores), 1, "Need ≥2 ANOMALY results to test spread")
        score_range = max(anomaly_scores) - min(anomaly_scores)
        self.assertLess(
            score_range, 5.0,
            f"ANOMALY scores should be tight; range={score_range:.1f}",
        )

    def test_score_cluster_fraction_high(self):
        """evaluate() should report a high cluster fraction."""
        features = self._uniform_spike_features()
        results  = classify_regions(features, min_aircraft=1)
        report   = evaluate(results)
        self.assertGreater(
            report.max_cluster_fraction,
            0.3,
            f"Expected high cluster fraction; got {report.max_cluster_fraction:.2f}",
        )

    def test_within_class_score_std_is_low(self):
        """Score std for ANOMALY should be below min_score_std threshold."""
        features = self._uniform_spike_features()
        results  = classify_regions(features, min_aircraft=1)
        report   = evaluate(results)
        anomaly_std = report.score_std_by_class.get("ANOMALY", -1.0)
        if anomaly_std >= 0:   # skip if only one ANOMALY result (sentinel = -1)
            self.assertLess(
                anomaly_std, THRESHOLDS["min_score_std"],
                f"ANOMALY std={anomaly_std:.2f} should be below min_score_std={THRESHOLDS['min_score_std']}",
            )

    def test_cluster_violation_in_report(self):
        """The CalibrationReport.violations list should flag the score collapse."""
        features = self._uniform_spike_features()
        results  = classify_regions(features, min_aircraft=1)
        report   = evaluate(results)
        # Either score_cluster or a score_std violation should fire
        cluster_violations = [v for v in report.violations
                               if "score_cluster" in v or "score_std" in v]
        self.assertTrue(
            cluster_violations,
            f"Expected cluster/std violation; violations={report.violations}",
        )

    def test_anomaly_overfit_or_score_cluster_failure_detected(self):
        """detect_failures() must flag the ANOMALY_OVERFIT or SCORE_CLUSTER rule."""
        features = self._uniform_spike_features()
        results  = classify_regions(features, min_aircraft=1)
        failures = detect_failures(results)
        rules    = {f.rule for f in failures}
        self.assertTrue(
            "ANOMALY_OVERFIT" in rules or "SCORE_CLUSTER" in rules,
            f"Expected ANOMALY_OVERFIT or SCORE_CLUSTER; got rules={rules}",
        )

    def test_change_score_norm_is_zero_for_all_regions(self):
        """
        Verify the root cause: with no activity_changes data every region
        has change_score=0 → change_score_norm=0 → 20% of ANOMALY weight wasted.
        """
        features = self._uniform_spike_features()
        for f in features:
            self.assertEqual(
                f.change_score, 0.0,
                f"Expected change_score=0 for spike-only features; got {f.change_score}",
            )


# ─── compute_signal_sufficiency / compute_input_diversity ────────────────────

class SignalSufficiencyTests(unittest.TestCase):
    """Unit tests for the two new signal-environment measurement functions."""

    def test_no_signal_returns_zero(self):
        """All default features (no spike, no coordination, no new entries, no change)."""
        r = _make_intel(Classification.ROUTINE, 60, 40)
        self.assertAlmostEqual(compute_signal_sufficiency([r]), 0.0, places=6)

    def test_empty_list_returns_zero(self):
        self.assertEqual(compute_signal_sufficiency([]), 0.0)
        self.assertEqual(compute_input_diversity([]), 0.0)

    def test_spike_only_returns_point35(self):
        """spike_flag=True alone contributes weight 0.35."""
        r = _make_intel(Classification.ANOMALY, 75, 40, spike_flag=True)
        self.assertAlmostEqual(compute_signal_sufficiency([r]), 0.35, places=6)

    def test_all_signals_active_returns_one(self):
        """spike + coordination + new_entry_ratio=1.0 + change_score at max → 1.0."""
        features = RegionFeatures(
            lat=0, lon=0,
            aircraft_count=10,
            spike_flag=True,
            coordination_flag=True,
            new_aircraft=10,          # new_entry_ratio = 10/10 = 1.0
            change_score=100.0,       # will be used as norm_denom → change_score_norm=1.0
        )
        r = RegionIntelligence(
            lat=0, lon=0,
            classification=Classification.ANOMALY,
            confidence=80, score=90,
            all_scores={c.value: 10.0 for c in Classification},
            features=features, explanation="test",
        )
        self.assertAlmostEqual(compute_signal_sufficiency([r]), 1.0, places=6)

    def test_batch_mean_is_average_of_per_region_scores(self):
        """Two regions: one with spike (0.35), one without (0.0) → mean = 0.175."""
        r_spike = _make_intel(Classification.ANOMALY, 75, 40, spike_flag=True,  lat=0)
        r_quiet = _make_intel(Classification.ROUTINE,  60, 30, spike_flag=False, lat=1)
        result = compute_signal_sufficiency([r_spike, r_quiet])
        self.assertAlmostEqual(result, 0.175, places=6)

    def test_input_diversity_zero_for_identical_regions(self):
        """All regions carry the same signal → std = 0."""
        regions = [
            _make_intel(Classification.ROUTINE, 60, 30, spike_flag=False, lat=float(i))
            for i in range(4)
        ]
        self.assertAlmostEqual(compute_input_diversity(regions), 0.0, places=6)

    def test_input_diversity_positive_for_mixed_regions(self):
        """Half with spike (0.35), half without (0.0) → std > 0."""
        regions = (
            [_make_intel(Classification.ANOMALY, 75, 40, spike_flag=True,  lat=float(i))
             for i in range(3)]
            + [_make_intel(Classification.ROUTINE,  60, 30, spike_flag=False, lat=float(i + 10))
               for i in range(3)]
        )
        self.assertGreater(compute_input_diversity(regions), 0.0)

    def test_input_diversity_sentinel_for_single_region(self):
        """Single region → std is undefined; function returns 0.0."""
        r = _make_intel(Classification.ROUTINE, 60, 30)
        self.assertEqual(compute_input_diversity([r]), 0.0)


# ─── signal gating: evaluate() and detect_failures() ─────────────────────────

class SignalGatingTests(unittest.TestCase):
    """
    Verify that score_cluster, class_skew (ROUTINE), and low_margin_fraction
    are suppressed when inputs are uniformly low-signal, but fire when
    signal-rich inputs produce the same patterns.
    """

    # ── helpers ──────────────────────────────────────────────────────────────

    def _routine_cluster(self, spike: bool) -> list:
        """Six ROUTINE@60 regions, optionally with spike_flag set."""
        return [
            _make_intel(Classification.ROUTINE, 60, 30, all_scores={
                "STAGING": 20, "PROJECTION": 20, "ROUTINE": 60,
                "ANOMALY": 25, "COORDINATED_ACTIVITY": 20,
            }, spike_flag=spike, lat=float(i))
            for i in range(6)
        ]

    def _tight_margin_batch(self, spike: bool) -> list:
        """Four regions with margin=2 (below the 10-point threshold)."""
        return [
            _make_intel(Classification.STAGING, 50, 20,
                        all_scores=_tight_all_scores(Classification.STAGING, 50, 2.0),
                        spike_flag=spike, lat=float(i))
            for i in range(4)
        ]

    # ── is_low_signal / is_legitimate_clustering ──────────────────────────────

    def test_is_low_signal_true_when_sufficiency_below_threshold(self):
        regions = self._routine_cluster(spike=False)   # sufficiency = 0.0
        report = evaluate(regions)
        self.assertTrue(report.is_low_signal)

    def test_is_low_signal_false_when_spike_present(self):
        regions = self._routine_cluster(spike=True)    # sufficiency = 0.35 > 0.10
        report = evaluate(regions)
        self.assertFalse(report.is_low_signal)

    def test_is_legitimate_clustering_requires_both_conditions(self):
        # Low signal BUT high diversity → not legitimate (only one condition met)
        r_spike = _make_intel(Classification.ROUTINE, 60, 30, spike_flag=True,  lat=0)
        r_quiet = _make_intel(Classification.ROUTINE, 60, 30, spike_flag=False, lat=1)
        report = evaluate([r_spike, r_quiet])
        # sufficiency = 0.175 > 0.10 → is_low_signal = False → not legitimate
        self.assertFalse(report.is_legitimate_clustering)

    def test_is_legitimate_clustering_true_for_uniform_no_signal(self):
        regions = self._routine_cluster(spike=False)
        report = evaluate(regions)
        self.assertTrue(report.is_legitimate_clustering)

    # ── score_cluster violation gating ────────────────────────────────────────

    def test_score_cluster_suppressed_in_low_signal(self):
        """ROUTINE cluster with no signal → legitimate → score_cluster must NOT fire."""
        report = evaluate(self._routine_cluster(spike=False))
        self.assertNotIn("score_cluster", report.violations)

    def test_score_cluster_fires_for_signal_rich_cluster(self):
        """Same cluster but with spike_flag=True → pathological → must fire."""
        report = evaluate(self._routine_cluster(spike=True))
        self.assertIn("score_cluster", report.violations)

    # ── class_skew (ROUTINE) gating ───────────────────────────────────────────

    def test_routine_class_skew_suppressed_in_low_signal(self):
        """ROUTINE dominating with no signal → expected → class_skew must NOT fire."""
        regions = [
            _make_intel(Classification.ROUTINE, 60, 30, spike_flag=False, lat=float(i))
            for i in range(6)
        ] + [_make_intel(Classification.STAGING, 80, 60, spike_flag=False, lat=10.0)]
        report = evaluate(regions)
        self.assertTrue(report.is_low_signal)
        self.assertNotIn("class_skew", report.violations)

    def test_non_routine_class_skew_fires_even_in_low_signal(self):
        """ANOMALY dominating at 83% must fire even when signal is zero."""
        regions = (
            [_make_intel(Classification.ANOMALY, 75, 40, spike_flag=False, lat=float(i))
             for i in range(5)]
            + [_make_intel(Classification.STAGING, 80, 60, spike_flag=False, lat=10.0)]
        )
        report = evaluate(regions)
        self.assertTrue(report.is_low_signal)
        self.assertIn("class_skew", report.violations)

    def test_routine_class_skew_fires_when_signal_rich(self):
        """ROUTINE dominating in a signal-rich env is suspicious → must fire."""
        regions = [
            _make_intel(Classification.ROUTINE, 60, 30, spike_flag=True, lat=float(i))
            for i in range(6)
        ] + [_make_intel(Classification.STAGING, 80, 60, spike_flag=True, lat=10.0)]
        report = evaluate(regions)
        self.assertFalse(report.is_low_signal)
        self.assertIn("class_skew", report.violations)

    # ── low_margin_fraction gating ────────────────────────────────────────────

    def test_low_margin_suppressed_in_low_signal(self):
        """Tight margins with no signal → expected → low_margin_fraction must NOT fire."""
        report = evaluate(self._tight_margin_batch(spike=False))
        self.assertTrue(report.is_low_signal)
        self.assertNotIn("low_margin_fraction", report.violations)

    def test_low_margin_fires_in_signal_rich_env(self):
        """Tight margins with spike_flag=True → suspicious → must fire."""
        report = evaluate(self._tight_margin_batch(spike=True))
        self.assertFalse(report.is_low_signal)
        self.assertIn("low_margin_fraction", report.violations)

    # ── detect_failures gating ────────────────────────────────────────────────

    def test_score_cluster_failure_suppressed_in_low_signal(self):
        regions = self._routine_cluster(spike=False)
        rules = {f.rule for f in detect_failures(regions)}
        self.assertNotIn("SCORE_CLUSTER", rules)

    def test_low_margin_failure_suppressed_in_low_signal(self):
        regions = self._tight_margin_batch(spike=False)
        rules = {f.rule for f in detect_failures(regions)}
        self.assertNotIn("LOW_MARGIN", rules)

    def test_class_dominance_failure_suppressed_for_routine_in_low_signal(self):
        regions = [
            _make_intel(Classification.ROUTINE, 60, 30, spike_flag=False, lat=float(i))
            for i in range(7)
        ]
        rules = {f.rule for f in detect_failures(regions)}
        self.assertNotIn("CLASS_DOMINANCE", rules)

    def test_class_dominance_failure_fires_for_anomaly_in_low_signal(self):
        """CLASS_DOMINANCE is only suppressed for ROUTINE; ANOMALY dominance always fires."""
        regions = (
            [_make_intel(Classification.ANOMALY, 75, 40, spike_flag=False, lat=float(i))
             for i in range(5)]
            + [_make_intel(Classification.STAGING, 80, 60, spike_flag=False, lat=10.0)]
        )
        rules = {f.rule for f in detect_failures(regions)}
        self.assertIn("CLASS_DOMINANCE", rules)

    def test_high_conf_low_data_always_fires_regardless_of_signal(self):
        """Rule 3 is never gated — it applies in any environment."""
        r = _make_intel(Classification.ROUTINE, 60, 80, aircraft_count=3, spike_flag=False)
        rules = {f.rule for f in detect_failures([r])}
        self.assertIn("HIGH_CONF_LOW_DATA", rules)

    def test_summary_contains_signal_info(self):
        report = evaluate(self._routine_cluster(spike=False))
        text = report.summary()
        self.assertIn("signal=", text)
        self.assertIn("LOW-SIGNAL", text)


# ─── low-signal environment: end-to-end loop gating ──────────────────────────

class LowSignalEnvironmentTests(unittest.TestCase):
    """
    End-to-end test of the described production scenario:
    no spikes, no new entries, no coordination, no activity-change data.
    The classifier correctly produces ROUTINE scores ~59–61.
    The calibration loop must detect the low-signal environment and exit
    WITHOUT modifying any weights.
    """

    def _routine_features(self) -> list:
        """Five long-running, non-military, non-spike regions."""
        return build_features(
            recurring=[
                (51.5,  -0.1,  10, 30, 2),
                (52.0,  -0.5,   8, 25, 1),
                (53.0,   1.0,   9, 28, 0),
                (48.8,   2.3,  11, 32, 3),
                (40.7, -74.0,   7, 20, 0),
            ]
        )

    def test_all_classified_as_routine(self):
        features = self._routine_features()
        results  = classify_regions(features, min_aircraft=1)
        for r in results:
            self.assertEqual(
                r.classification, Classification.ROUTINE,
                f"Expected ROUTINE but got {r.classification.value}",
            )

    def test_signal_sufficiency_is_zero(self):
        features = self._routine_features()
        results  = classify_regions(features, min_aircraft=1)
        sufficiency = compute_signal_sufficiency(results)
        self.assertAlmostEqual(sufficiency, 0.0, places=6)

    def test_report_is_low_signal(self):
        features = self._routine_features()
        results  = classify_regions(features, min_aircraft=1)
        report   = evaluate(results)
        self.assertTrue(report.is_low_signal)

    def test_report_is_legitimate_clustering(self):
        features = self._routine_features()
        results  = classify_regions(features, min_aircraft=1)
        report   = evaluate(results)
        self.assertTrue(report.is_legitimate_clustering)

    def test_gated_violations_absent(self):
        """score_cluster, class_skew (ROUTINE), and low_margin_fraction must be absent."""
        features = self._routine_features()
        results  = classify_regions(features, min_aircraft=1)
        report   = evaluate(results)
        for blocked in ("score_cluster", "class_skew", "low_margin_fraction"):
            self.assertNotIn(
                blocked, report.violations,
                f"Violation '{blocked}' should be suppressed in low-signal env; "
                f"violations={report.violations}",
            )

    def test_calibration_loop_exits_after_one_iteration(self):
        """
        The loop must detect the low-signal environment at iteration 1, record
        the state, and exit without proposing any weight adjustments.
        """
        features = self._routine_features()
        final_profiles, history = run_calibration_loop(
            features, max_iterations=10, min_aircraft=1, verbose=False,
        )
        self.assertEqual(len(history), 1,
                         f"Expected exit after 1 iteration; got {len(history)}")

    def test_calibration_loop_leaves_weights_unchanged(self):
        """
        When the gate fires, no WeightAdjustments should be recorded in history.
        The final profiles must equal the starting profiles exactly.
        """
        features = self._routine_features()
        starting = copy.deepcopy(_cls_module._PROFILES)
        final, history = run_calibration_loop(
            features, profiles=starting, max_iterations=10,
            min_aircraft=1, verbose=False,
        )
        self.assertEqual(history[0].adjustments, [],
                         "Gate iteration must record zero adjustments")
        for cls in Classification:
            for feat in _cls_module._PROFILES[cls]:
                self.assertAlmostEqual(
                    final[cls][feat],
                    _cls_module._PROFILES[cls][feat],
                    places=6,
                    msg=f"{cls.value}.{feat} was modified despite low-signal gate",
                )

    def test_calibration_loop_does_not_alter_module_profiles(self):
        """Module-level _PROFILES must be restored even when the gate fires."""
        before = {cls: dict(p) for cls, p in _cls_module._PROFILES.items()}
        run_calibration_loop(
            self._routine_features(), max_iterations=5, min_aircraft=1, verbose=False,
        )
        for cls in Classification:
            for feat, w in before[cls].items():
                self.assertAlmostEqual(
                    _cls_module._PROFILES[cls][feat], w, places=9,
                    msg=f"Module {cls.value}.{feat} was permanently altered",
                )


# ─── FEATURE_COLLAPSE_ANOMALY rule ───────────────────────────────────────────

class FeatureCollapseAnomalyTests(unittest.TestCase):
    """
    Tests for detect_failures() Rule 6: FEATURE_COLLAPSE_ANOMALY.

    The rule fires when an ANOMALY-profile feature has non-trivial weight
    (≥ 0.10) but near-zero activation across all ANOMALY-classified regions
    (mean < 0.50 AND std < 0.05).  Root cause: the feature is structurally
    inactive (e.g. change_score_norm=0 always), so another feature dominates.
    """

    def _make_anomaly_profiles(
        self, *, collapse_feature: str = "change_score_norm", weight: float = 0.20
    ) -> dict:
        """
        Return profiles where `collapse_feature` has `weight` in the ANOMALY
        profile (and the remaining weight is split evenly among other features).
        """
        profiles = copy.deepcopy(_cls_module._PROFILES)
        anomaly  = profiles[Classification.ANOMALY]

        # Zero out the collapse feature first, redistribute remaining weight
        other_features = [f for f in anomaly if f != collapse_feature]
        if other_features:
            current_other = sum(anomaly[f] for f in other_features)
            if current_other > 0:
                scale = (1.0 - weight) / current_other
                for f in other_features:
                    anomaly[f] = round(anomaly[f] * scale, 6)
        anomaly[collapse_feature] = weight
        return profiles

    def _anomaly_results_no_change(self, n: int = 3) -> list:
        """
        n ANOMALY-classified regions, all with change_score=0 and spike_flag=True.
        We construct RegionIntelligence directly to control classification.
        """
        results = []
        for i in range(n):
            f = RegionFeatures(
                lat=float(i), lon=0.0,
                aircraft_count=4,
                spike_flag=True,
                change_score=0.0,
            )
            r = RegionIntelligence(
                lat=float(i), lon=0.0,
                classification=Classification.ANOMALY,
                confidence=65, score=75,
                all_scores={c.value: (75.0 if c == Classification.ANOMALY else 20.0)
                            for c in Classification},
                features=f, explanation="test",
            )
            results.append(r)
        return results

    def test_rule_fires_for_inactive_feature(self):
        """change_score_norm=0 always, weight=0.20 → rule must fire."""
        profiles = self._make_anomaly_profiles(
            collapse_feature="change_score_norm", weight=0.20
        )
        results  = self._anomaly_results_no_change(3)
        rules    = {f.rule for f in detect_failures(results, profiles)}
        self.assertIn("FEATURE_COLLAPSE_ANOMALY", rules)

    def test_rule_does_not_fire_for_active_feature(self):
        """spike_flag=True always (mean=1.0) → above feature_collapse_max_mean → not flagged.

        Other collapsed features may still fire; we only assert spike_flag itself
        does not appear in any collapse failure detail.
        """
        profiles = self._make_anomaly_profiles(
            collapse_feature="spike_flag", weight=0.40
        )
        results  = self._anomaly_results_no_change(3)
        failures = detect_failures(results, profiles)
        spike_collapse = [
            f for f in failures
            if f.rule == "FEATURE_COLLAPSE_ANOMALY" and "'spike_flag'" in f.detail
        ]
        self.assertEqual(
            spike_collapse, [],
            "spike_flag (mean=1.0) must not appear in FEATURE_COLLAPSE_ANOMALY failures",
        )

    def test_rule_does_not_fire_without_anomaly_results(self):
        """No ANOMALY classifications → rule is naturally skipped."""
        profiles = self._make_anomaly_profiles(
            collapse_feature="change_score_norm", weight=0.20
        )
        results = [
            _make_intel(Classification.ROUTINE, 60, 30, lat=float(i))
            for i in range(3)
        ]
        rules = {f.rule for f in detect_failures(results, profiles)}
        self.assertNotIn("FEATURE_COLLAPSE_ANOMALY", rules)

    def test_rule_does_not_fire_without_profiles_argument(self):
        """profiles=None → rule 6 is skipped entirely (no KeyError, no fire)."""
        results = self._anomaly_results_no_change(3)
        rules   = {f.rule for f in detect_failures(results)}  # profiles=None default
        self.assertNotIn("FEATURE_COLLAPSE_ANOMALY", rules)

    def test_rule_skipped_for_single_anomaly_result(self):
        """len(anomaly_results) < 2 → not enough data to compute std → skipped."""
        profiles = self._make_anomaly_profiles(
            collapse_feature="change_score_norm", weight=0.20
        )
        results = self._anomaly_results_no_change(1)
        rules   = {f.rule for f in detect_failures(results, profiles)}
        self.assertNotIn("FEATURE_COLLAPSE_ANOMALY", rules)

    def test_rule_respects_min_weight_threshold(self):
        """Feature with weight below feature_collapse_min_weight (0.10) is ignored."""
        profiles = self._make_anomaly_profiles(
            collapse_feature="change_score_norm", weight=0.05   # below 0.10
        )
        results = self._anomaly_results_no_change(3)
        rules   = {f.rule for f in detect_failures(results, profiles)}
        self.assertNotIn("FEATURE_COLLAPSE_ANOMALY", rules)

    def test_severity_proportional_to_weight(self):
        """Higher weight → higher severity (capped at 1.0)."""
        profiles_lo = self._make_anomaly_profiles(
            collapse_feature="change_score_norm", weight=0.10
        )
        profiles_hi = self._make_anomaly_profiles(
            collapse_feature="change_score_norm", weight=0.40
        )
        results = self._anomaly_results_no_change(3)

        sev_lo = max(
            (f.severity for f in detect_failures(results, profiles_lo)
             if f.rule == "FEATURE_COLLAPSE_ANOMALY"),
            default=None,
        )
        sev_hi = max(
            (f.severity for f in detect_failures(results, profiles_hi)
             if f.rule == "FEATURE_COLLAPSE_ANOMALY"),
            default=None,
        )
        self.assertIsNotNone(sev_lo, "Expected failures at weight=0.10")
        self.assertIsNotNone(sev_hi, "Expected failures at weight=0.40")
        self.assertGreater(sev_hi, sev_lo)

    def test_self_gated_in_low_signal_routine_batch(self):
        """
        In a ROUTINE-only low-signal batch there are no ANOMALY results, so the
        rule naturally never fires — self-gating without an explicit env check.
        """
        features = build_features(
            recurring=[
                (51.5, -0.1,  10, 30, 2),
                (52.0, -0.5,   8, 25, 1),
                (53.0,  1.0,   9, 28, 0),
            ]
        )
        results  = classify_regions(features, min_aircraft=1)
        profiles = copy.deepcopy(_cls_module._PROFILES)
        rules    = {f.rule for f in detect_failures(results, profiles)}
        self.assertNotIn("FEATURE_COLLAPSE_ANOMALY", rules)


# ─── check_convergence: narrowed to 4 critical violations ────────────────────

class CheckConvergenceTests(unittest.TestCase):
    """
    check_convergence() must return True when only secondary violations are
    present, and False when any of the 4 operationally critical violations fire.

    Critical violations:
      • mean_margin
      • conf_margin_corr
      • score_cluster
      • score_std[<class>]   (any class; threshold = 3.0)

    Secondary (informational only):
      • class_skew
      • feature_dominance[*]
      • low_margin_fraction
      • small_data_high_conf
    """

    def _report_with_violations(self, violations: list) -> CalibrationReport:
        """Build a minimal CalibrationReport and forcibly set its violations list."""
        r = _make_intel(Classification.STAGING, 80, 70)
        report = evaluate([r])
        report.violations = violations
        return report

    def test_empty_violations_converged(self):
        report = self._report_with_violations([])
        self.assertTrue(check_convergence(report))

    def test_secondary_violations_only_still_converged(self):
        """class_skew, feature_dominance, low_margin_fraction, small_data_high_conf
        must NOT block convergence."""
        secondary = [
            "class_skew",
            "feature_dominance[STAGING.spike_flag]",
            "low_margin_fraction",
            "small_data_high_conf",
        ]
        for v in secondary:
            report = self._report_with_violations([v])
            self.assertTrue(
                check_convergence(report),
                f"Secondary violation '{v}' must not block convergence",
            )

    def test_multiple_secondary_violations_still_converged(self):
        report = self._report_with_violations([
            "class_skew", "low_margin_fraction", "small_data_high_conf"
        ])
        self.assertTrue(check_convergence(report))

    def test_mean_margin_blocks_convergence(self):
        report = self._report_with_violations(["mean_margin"])
        self.assertFalse(check_convergence(report))

    def test_conf_margin_corr_blocks_convergence(self):
        report = self._report_with_violations(["conf_margin_corr"])
        self.assertFalse(check_convergence(report))

    def test_score_cluster_blocks_convergence(self):
        report = self._report_with_violations(["score_cluster"])
        self.assertFalse(check_convergence(report))

    def test_score_std_blocks_convergence(self):
        """score_std[ANOMALY] must block — any class name in the bracket."""
        for cls_name in ("ANOMALY", "STAGING", "ROUTINE"):
            report = self._report_with_violations([f"score_std[{cls_name}]"])
            self.assertFalse(
                check_convergence(report),
                f"score_std[{cls_name}] must block convergence",
            )

    def test_mixed_secondary_and_critical_blocks(self):
        """Secondary + critical: must be False because critical is present."""
        report = self._report_with_violations(["class_skew", "mean_margin"])
        self.assertFalse(check_convergence(report))

    def test_is_converged_still_strict(self):
        """is_converged() remains strict — secondary violations do block it."""
        report = self._report_with_violations(["class_skew"])
        self.assertFalse(report.is_converged(),
                         "is_converged() should remain strict (checks all violations)")

    def test_convergence_criteria_contains_exactly_4_entries(self):
        """CONVERGENCE_CRITERIA dict must reflect the 4 critical criteria."""
        self.assertEqual(len(CONVERGENCE_CRITERIA), 4)

    def test_convergence_criteria_keys(self):
        """Verify the 4 expected keys are present."""
        keys = set(CONVERGENCE_CRITERIA.keys())
        self.assertIn("mean_margin",               keys)
        self.assertIn("conf_margin_corr",           keys)
        self.assertIn("max_cluster_fraction",       keys)
        self.assertIn("min_score_std[each class]",  keys)

    def test_score_std_threshold_is_three(self):
        """The within-class std threshold was relaxed from 5.0 to 3.0."""
        self.assertEqual(THRESHOLDS["min_score_std"], 3.0)

    def test_convergence_violation_prefixes_frozenset(self):
        """_CONVERGENCE_VIOLATION_PREFIXES must be a frozenset of the 3 exact prefixes."""
        expected = frozenset({"mean_margin", "conf_margin_corr", "score_cluster"})
        self.assertEqual(_CONVERGENCE_VIOLATION_PREFIXES, expected)

    def test_convergence_violation_prefix_str(self):
        """_CONVERGENCE_VIOLATION_PREFIX_STR must be the score_std prefix."""
        self.assertEqual(_CONVERGENCE_VIOLATION_PREFIX_STR, "score_std[")


# ─── FEATURE_DOMINANCE_IMBALANCE rule ────────────────────────────────────────

class FeatureDominanceImbalanceTests(unittest.TestCase):
    """
    Tests for FEATURE_DOMINANCE_IMBALANCE (Rule 7) and the associated
    feature_dominance:<class>:<feature>:<ratio> violation.

    The rule fires when one feature contributes > 60% of the class score for
    ≥ 60% of regions in that class, in a signal-rich environment.

    Test strategy: use a custom ROUTINE profile with persistence_score=0.75
    and classify regions that have persistence_score=1.0 with spike_flag=True
    (so signal_sufficiency > min_signal_sufficiency).
    """

    def _dominant_routine_profiles(self) -> dict:
        """ROUTINE profile where persistence_score holds 75% of the weight."""
        profiles = copy.deepcopy(_cls_module._PROFILES)
        profiles[Classification.ROUTINE] = {
            "persistence_score":    0.75,
            "unmilitary":           0.08,
            "low_new_entry_ratio":  0.08,
            "no_spike":             0.04,
            "any_flow":             0.03,
            "aircraft_density":     0.02,
        }
        return profiles

    def _classify_with(self, profiles: dict, features_list: list) -> list:
        """Classify using monkey-patched profiles so scores are self-consistent."""
        original = _cls_module._PROFILES
        _cls_module._PROFILES = profiles
        try:
            return classify_regions(features_list, min_aircraft=1)
        finally:
            _cls_module._PROFILES = original

    def _high_persistence_features(self, n: int = 4) -> list:
        """
        n regions: recurring 12 appearances (persistence_score=1.0),
        plus a spike at the same location (spike_flag=True → signal-rich).
        """
        recurring = [(float(i), 0.0, 12, 20, 0) for i in range(n)]
        spikes    = [(float(i), 0.0, 20, 0)       for i in range(n)]
        return build_features(recurring=recurring, spikes=spikes)

    # ── detection ────────────────────────────────────────────────────────────

    def test_dominance_failure_fires(self):
        """persistence_score at 75% weight + max activation → rule fires."""
        profiles = self._dominant_routine_profiles()
        features = self._high_persistence_features(4)
        results  = self._classify_with(profiles, features)
        rules    = {f.rule for f in detect_failures(results, profiles)}
        self.assertIn("FEATURE_DOMINANCE_IMBALANCE", rules)

    def test_dominance_violation_in_report(self):
        """evaluate() records feature_dominance:<class>:<feat>:<ratio> violation."""
        profiles = self._dominant_routine_profiles()
        features = self._high_persistence_features(4)
        results  = self._classify_with(profiles, features)
        report   = evaluate(results, profiles)
        fd_viols = [v for v in report.violations if v.startswith("feature_dominance:")]
        self.assertTrue(
            fd_viols,
            f"Expected feature_dominance: violation; got violations={report.violations}",
        )
        self.assertTrue(
            any("ROUTINE" in v and "persistence_score" in v for v in fd_viols),
            f"Violation must name ROUTINE and persistence_score; got {fd_viols}",
        )

    def test_not_triggered_when_balanced(self):
        """Equal-weight profile with multiple active features → no dominance."""
        profiles = copy.deepcopy(_cls_module._PROFILES)
        feat_list = list(profiles[Classification.ROUTINE].keys())
        eq_w      = round(1.0 / len(feat_list), 6)
        profiles[Classification.ROUTINE] = {f: eq_w for f in feat_list}
        features = self._high_persistence_features(4)
        results  = self._classify_with(profiles, features)
        rules    = {f.rule for f in detect_failures(results, profiles)}
        self.assertNotIn("FEATURE_DOMINANCE_IMBALANCE", rules)

    def test_not_triggered_in_low_signal(self):
        """Dominance with no signal events → low-signal env → suppressed."""
        profiles = self._dominant_routine_profiles()
        # recurring only, no spikes → spike_flag=False → signal_sufficiency=0
        low_signal_features = build_features(
            recurring=[(float(i), 0.0, 12, 20, 0) for i in range(4)]
        )
        results = self._classify_with(profiles, low_signal_features)
        report  = evaluate(results, profiles)
        self.assertTrue(report.is_low_signal, "Expected low-signal environment")
        rules = {f.rule for f in detect_failures(results, profiles)}
        self.assertNotIn("FEATURE_DOMINANCE_IMBALANCE", rules)
        fd_viols = [v for v in report.violations if v.startswith("feature_dominance:")]
        self.assertFalse(fd_viols, "feature_dominance: violation must be suppressed in low-signal")

    # ── adjustment ───────────────────────────────────────────────────────────

    def test_adjustment_reduces_dominant_feature(self):
        """propose_adjustments() must reduce persistence_score weight (×0.9)."""
        profiles = self._dominant_routine_profiles()
        features = self._high_persistence_features(4)
        results  = self._classify_with(profiles, features)
        report   = evaluate(results, profiles)
        adjustments = propose_adjustments(results, profiles, report)
        pers_adj = next(
            (a for a in adjustments
             if a.classification == "ROUTINE" and a.feature == "persistence_score"),
            None,
        )
        self.assertIsNotNone(pers_adj,
            "Expected adjustment targeting ROUTINE.persistence_score")
        self.assertLess(pers_adj.new_weight, pers_adj.old_weight)
        self.assertAlmostEqual(
            pers_adj.new_weight, pers_adj.old_weight * 0.9, places=3,
            msg="Weight must be reduced by ×0.9",
        )

    def test_adjustment_redistributes_and_sums_to_one(self):
        """Removed weight is redistributed; profile sums to 1.0 after apply."""
        profiles    = self._dominant_routine_profiles()
        features    = self._high_persistence_features(4)
        results     = self._classify_with(profiles, features)
        report      = evaluate(results, profiles)
        adjustments = propose_adjustments(results, profiles, report)
        new_profiles = apply_adjustments(adjustments, profiles)
        total = sum(new_profiles[Classification.ROUTINE].values())
        self.assertAlmostEqual(
            total, 1.0, places=4,
            msg=f"ROUTINE weight sum must be 1.0 after redistribution; got {total:.6f}",
        )
        # Redistribution adjustments must all be positive
        redis_adjs = [
            a for a in adjustments
            if a.classification == "ROUTINE"
            and a.feature != "persistence_score"
            and "FEATURE_DOMINANCE_IMBALANCE" in a.reason
        ]
        self.assertTrue(redis_adjs, "Expected redistribution adjustments")
        for a in redis_adjs:
            self.assertGreater(a.delta, 0, f"{a.feature} should receive redistributed weight")

    def test_weights_clamped_to_bounds(self):
        """All new weights from the FEATURE_DOMINANCE_IMBALANCE block must be in [0.05, 0.80]."""
        profiles    = self._dominant_routine_profiles()
        features    = self._high_persistence_features(4)
        results     = self._classify_with(profiles, features)
        report      = evaluate(results, profiles)
        adjustments = propose_adjustments(results, profiles, report)
        imbalance_adjs = [a for a in adjustments if "FEATURE_DOMINANCE_IMBALANCE" in a.reason]
        for a in imbalance_adjs:
            self.assertGreaterEqual(a.new_weight, 0.05,
                f"{a.feature}: new_weight {a.new_weight} below 0.05 clamp")
            self.assertLessEqual(a.new_weight, 0.80,
                f"{a.feature}: new_weight {a.new_weight} above 0.80 clamp")

    # ── feature_contributions_summary ────────────────────────────────────────

    def test_feature_contributions_summary_populated(self):
        """evaluate() populates feature_contributions_summary with per-class data."""
        profiles = self._dominant_routine_profiles()
        features = self._high_persistence_features(4)
        results  = self._classify_with(profiles, features)
        report   = evaluate(results, profiles)
        self.assertIsNotNone(report.feature_contributions_summary)
        self.assertIn("ROUTINE", report.feature_contributions_summary)
        routine = report.feature_contributions_summary["ROUTINE"]
        self.assertIn("persistence_score", routine)
        top_feat = max(routine, key=routine.__getitem__)
        self.assertEqual(top_feat, "persistence_score",
            "persistence_score must be the dominant contributor in the summary")

    def test_compute_feature_contributions_sums_to_one(self):
        """compute_feature_contributions() must produce per-class sums = 1.0
        for any class whose raw contributions are non-zero.
        Classes where all features evaluate to 0 (score would be 0) return
        all-zero breakdowns and are skipped."""
        profiles = self._dominant_routine_profiles()
        features = self._high_persistence_features(1)
        results  = self._classify_with(profiles, features)
        self.assertTrue(results, "Need at least one result")
        contribs = compute_feature_contributions(results[0], profiles)
        for cls_name, breakdown in contribs.items():
            if not breakdown:
                continue
            total = sum(breakdown.values())
            if total == 0.0:
                continue   # class has zero activation — normalization is undefined
            self.assertAlmostEqual(
                total, 1.0, places=3,
                msg=f"Contributions for {cls_name} sum to {total:.4f}, expected 1.0",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
