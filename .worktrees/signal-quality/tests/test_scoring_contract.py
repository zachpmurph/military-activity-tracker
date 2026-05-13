"""
Scoring contract validation tests.

Enforces the invariants, formulas, and thresholds documented in
docs/02_scoring_contract.md using only build_features() and classify_regions()
as the public surface.  Classifier code is never modified.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intelligence.classifier import (
    Classification,
    RegionFeatures,
    _EXT_BOOST_MAX,
    _EXT_BOOST_PER_STRENGTH,
    _EXT_BYPASS_INTENSITY,
    _EXT_BYPASS_MIN_STRENGTH,
    _EXT_COORD_BOOST_MAX,
    _EXT_CROSS_DOMAIN_BONUS,
    build_features,
    classify_regions,
)
from intelligence.external_features import ExternalSignal, SignalType

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_LAT, _LON = 0.0, 0.0


def _find_intel(intel, lat=_LAT, lon=_LON):
    for r in intel:
        if abs(r.lat - lat) < 0.15 and abs(r.lon - lon) < 0.15:
            return r
    return None


def _signal(stype, lat=_LAT, lon=_LON, intensity=1.0):
    return ExternalSignal(
        signal_type=stype, lat=lat, lon=lon, intensity=intensity, metadata={}
    )


# Minimal 10-field activity_changes row: only [0],[1],[2],[7],[9] are read.
def _change_row(lat, lon, change_score, level="LOW", persistence="SHORT"):
    return (lat, lon, change_score, 0, 0, 0.0, "", level, "", persistence)


# ---------------------------------------------------------------------------
# 1. Feature bound invariants — all values must stay in documented ranges
# ---------------------------------------------------------------------------


class FeatureBoundTests(unittest.TestCase):
    def test_core_props_in_unit_range(self):
        features = build_features(
            coordinated=[(0, 0, 10, 6)],
            spikes=[(0, 0, 10, 6)],
            recurring=[(0, 0, 8, 10, 6)],
            new_entries=[(0, 0, 4, 2)],
        )
        f = features[0]
        for name, val in (
            ("military_ratio", f.military_ratio),
            ("new_entry_ratio", f.new_entry_ratio),
            ("persistence_score", f.persistence_score),
            ("aircraft_density", f.aircraft_density),
            ("novelty", f.novelty),
        ):
            self.assertGreaterEqual(val, 0.0, f"{name} below 0")
            self.assertLessEqual(val, 1.0, f"{name} above 1")

    def test_flow_balance_in_signed_range(self):
        # inflow > outflow → negative balance; outflow > inflow → positive
        features = build_features(
            spikes=[(0, 0, 10, 0)],
            movements=[
                (0, 0, 1, 0, 8, 0, 1.0),   # outflow from (0,0)
                (2, 0, 0, 0, 12, 0, 1.0),  # inflow to (0,0)
            ],
        )
        f = features[0]
        self.assertGreaterEqual(f.flow_balance, -1.0)
        self.assertLessEqual(f.flow_balance, 1.0)

    def test_novelty_capped_at_one(self):
        # Maximise every novelty component: all new, spike, zero persistence.
        features = build_features(
            spikes=[(0, 0, 10, 0)],
            new_entries=[(0, 0, 10, 0)],
        )
        f = features[0]
        self.assertLessEqual(f.novelty, 1.0)
        self.assertGreaterEqual(f.novelty, 0.0)

    def test_persistence_score_capped_at_one(self):
        features = build_features(recurring=[(0, 0, 12, 10, 0)])
        f = features[0]
        self.assertAlmostEqual(f.persistence_score, 1.0)


# ---------------------------------------------------------------------------
# 2. Score and confidence bounds — [0, 100]
# ---------------------------------------------------------------------------


class ScoreBoundTests(unittest.TestCase):
    def _build_mixed(self):
        return build_features(
            coordinated=[(0, 0, 10, 6)],
            spikes=[(1, 1, 5, 0)],
            recurring=[(2, 2, 10, 20, 0)],
            staging=[(3, 3, 8, 6)],
            movements=[(3, 3, 4, 3, 7, 5, 1.5)],
        )

    def test_all_class_scores_bounded(self):
        intel = classify_regions(self._build_mixed(), min_aircraft=1)
        self.assertTrue(intel)
        for r in intel:
            for cls, score in r.all_scores.items():
                self.assertGreaterEqual(score, 0.0, f"{cls} below 0")
                self.assertLessEqual(score, 100.0, f"{cls} above 100")

    def test_confidence_bounded(self):
        intel = classify_regions(self._build_mixed(), min_aircraft=1)
        for r in intel:
            self.assertGreaterEqual(r.confidence, 0)
            self.assertLessEqual(r.confidence, 100)

    def test_scores_bounded_with_external_boost(self):
        # Max external boost scenario: external sets spike, both NO_FLY + MARITIME present.
        features = build_features(
            new_entries=[(0, 0, 5, 0)],
            external_signals=[
                _signal(SignalType.NO_FLY,   intensity=1.0),
                _signal(SignalType.MARITIME, intensity=1.0),
            ],
        )
        intel = classify_regions(features)
        r = _find_intel(intel)
        self.assertIsNotNone(r)
        for cls, score in r.all_scores.items():
            self.assertLessEqual(score, 100.0, f"{cls} exceeded 100 after boost")


# ---------------------------------------------------------------------------
# 3. Determinism — identical inputs must produce identical outputs
# ---------------------------------------------------------------------------


class DeterminismTests(unittest.TestCase):
    _KWARGS = dict(
        coordinated=[(0, 0, 8, 4)],
        spikes=[(0, 0, 8, 4)],
        recurring=[(0, 0, 6, 8, 4)],
        external_signals=[
            ExternalSignal(SignalType.NO_FLY, 0, 0, 0.8, {}),
        ],
    )

    def test_repeated_calls_identical(self):
        results = [
            classify_regions(build_features(**self._KWARGS))[0]
            for _ in range(3)
        ]
        for r in results[1:]:
            self.assertEqual(r.classification, results[0].classification)
            self.assertAlmostEqual(r.score, results[0].score)
            self.assertEqual(r.confidence, results[0].confidence)

    def test_external_signal_order_independent(self):
        sig_a = _signal(SignalType.NO_FLY,   intensity=0.9)
        sig_b = _signal(SignalType.MARITIME, intensity=0.7)
        features_ab = build_features(
            new_entries=[(0, 0, 5, 2)], external_signals=[sig_a, sig_b]
        )
        features_ba = build_features(
            new_entries=[(0, 0, 5, 2)], external_signals=[sig_b, sig_a]
        )
        r_ab = _find_intel(classify_regions(features_ab))
        r_ba = _find_intel(classify_regions(features_ba))
        self.assertIsNotNone(r_ab)
        self.assertIsNotNone(r_ba)
        self.assertEqual(r_ab.classification, r_ba.classification)
        self.assertAlmostEqual(r_ab.score, r_ba.score)


# ---------------------------------------------------------------------------
# 4. Additive-only external signals — values may only increase
# ---------------------------------------------------------------------------


class AdditiveOnlyTests(unittest.TestCase):
    def test_spike_flag_not_cleared_by_external(self):
        # Detection sets spike_flag=True; external signal must not clear it.
        base = build_features(spikes=[(0, 0, 5, 0)])[0]
        self.assertTrue(base.spike_flag)

        with_signal = build_features(
            spikes=[(0, 0, 5, 0)],
            external_signals=[_signal(SignalType.NO_FLY, intensity=0.3)],
        )[0]
        self.assertTrue(with_signal.spike_flag)

    def test_change_score_only_increases(self):
        base = build_features(
            activity_changes=[_change_row(0, 0, 15.0)]
        )[0]
        with_signal = build_features(
            activity_changes=[_change_row(0, 0, 15.0)],
            external_signals=[_signal(SignalType.NO_FLY, intensity=1.0)],
        )[0]
        self.assertGreaterEqual(with_signal.change_score, base.change_score)

    def test_inflow_only_increases(self):
        base = build_features(
            spikes=[(0, 0, 5, 0)],
            movements=[(1, 0, 0, 0, 3, 0, 1.0)],  # 3 aircraft inflow to (0,0)
        )[0]
        with_signal = build_features(
            spikes=[(0, 0, 5, 0)],
            movements=[(1, 0, 0, 0, 3, 0, 1.0)],
            external_signals=[_signal(SignalType.MARITIME, intensity=1.0)],
        )[0]
        self.assertGreaterEqual(with_signal.inflow_count, base.inflow_count)

    def test_coordination_flag_not_cleared(self):
        # Detection sets coordination_flag=True; satellite signal must not clear it.
        base = build_features(coordinated=[(0, 0, 6, 3)])[0]
        self.assertTrue(base.coordination_flag)

        with_signal = build_features(
            coordinated=[(0, 0, 6, 3)],
            external_signals=[_signal(SignalType.SATELLITE, intensity=1.0)],
        )[0]
        self.assertTrue(with_signal.coordination_flag)


# ---------------------------------------------------------------------------
# 5. Monotonicity — increasing inputs → non-decreasing derived values
# ---------------------------------------------------------------------------


class MonotonicityTests(unittest.TestCase):
    def test_military_ratio_monotone(self):
        ratios = []
        for mil in range(0, 11):
            f = RegionFeatures(lat=0, lon=0, aircraft_count=10, military_count=mil)
            ratios.append(f.military_ratio)
        self.assertEqual(ratios, sorted(ratios))

    def test_persistence_score_monotone(self):
        scores = []
        for appearances in range(0, 13):
            f = RegionFeatures(lat=0, lon=0, recurring_appearances=appearances)
            scores.append(f.persistence_score)
        self.assertEqual(scores, sorted(scores))

    def test_staging_score_increases_with_flag(self):
        # Without staging_flag (use new_entries to get aircraft without the flag)
        features_no_flag = build_features(new_entries=[(0, 0, 10, 8)])
        r_no_flag = _find_intel(classify_regions(features_no_flag, min_aircraft=1))

        # With staging_flag
        features_flag = build_features(staging=[(0, 0, 10, 8)])
        r_flag = _find_intel(classify_regions(features_flag, min_aircraft=1))

        self.assertIsNotNone(r_no_flag)
        self.assertIsNotNone(r_flag)
        self.assertGreater(
            r_flag.all_scores["STAGING"],
            r_no_flag.all_scores["STAGING"],
        )

    def test_routine_score_increases_with_persistence(self):
        scores = []
        for appearances in (4, 8, 12):
            features = build_features(recurring=[(0, 0, appearances, 10, 0)])
            intel = classify_regions(features, min_aircraft=1)
            r = _find_intel(intel)
            self.assertIsNotNone(r)
            scores.append(r.all_scores["ROUTINE"])
        self.assertEqual(scores, sorted(scores))
        # All three must be strictly increasing
        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[1], scores[2])


# ---------------------------------------------------------------------------
# 6. Coordination / spike flag thresholds — boundary is eff > 0.5
# ---------------------------------------------------------------------------


class FlagThresholdTests(unittest.TestCase):
    """Contract: flag is set when effective_intensity > 0.5 (strictly greater).

    For a colocated signal (distance=0) the weight is 1.0, so eff == intensity.
    """

    def _maritime_features(self, intensity):
        return build_features(
            new_entries=[(0, 0, 5, 0)],
            external_signals=[_signal(SignalType.MARITIME, intensity=intensity)],
        )

    def _nofly_features(self, intensity):
        return build_features(
            new_entries=[(0, 0, 5, 0)],
            external_signals=[_signal(SignalType.NO_FLY, intensity=intensity)],
        )

    def test_coordination_flag_not_set_below_half(self):
        f = self._maritime_features(0.49)[0]
        self.assertFalse(f.coordination_flag)

    def test_coordination_flag_not_set_at_exactly_half(self):
        f = self._maritime_features(0.50)[0]
        self.assertFalse(f.coordination_flag)

    def test_coordination_flag_set_above_half(self):
        f = self._maritime_features(0.51)[0]
        self.assertTrue(f.coordination_flag)

    def test_spike_flag_not_set_at_exactly_half_via_nofly(self):
        # Detection did NOT set spike; external NO_FLY at eff=0.50 must not set it.
        f = self._nofly_features(0.50)[0]
        self.assertFalse(f.spike_flag)

    def test_spike_flag_set_above_half_via_nofly(self):
        f = self._nofly_features(0.51)[0]
        self.assertTrue(f.spike_flag)


# ---------------------------------------------------------------------------
# 7. Min-aircraft bypass conditions
# ---------------------------------------------------------------------------


class BypassConditionTests(unittest.TestCase):
    """Region with aircraft_count=1 is below the default min_aircraft=2 guard.

    The bypass requires BOTH:
      - external_signal_max_intensity >= _EXT_BYPASS_INTENSITY (0.75)
      - external_signal_strength      >= _EXT_BYPASS_MIN_STRENGTH (0.10)
    """

    def _one_aircraft(self, signals):
        return build_features(
            new_entries=[(0, 0, 1, 0)], external_signals=signals
        )

    def test_no_bypass_below_intensity_threshold(self):
        # intensity=0.74 < 0.75 → max_intensity check fails
        features = self._one_aircraft([_signal(SignalType.NO_FLY, intensity=0.74)])
        intel = classify_regions(features)  # default min_aircraft=2
        self.assertIsNone(_find_intel(intel))

    def test_no_bypass_weak_effective_strength(self):
        # intensity=1.0 but at distance≈1.45° → weight≈0.033 → strength≈0.033 < 0.10
        features = self._one_aircraft(
            [_signal(SignalType.NO_FLY, lat=1.45, lon=0.0, intensity=1.0)]
        )
        intel = classify_regions(features)
        self.assertIsNone(_find_intel(intel))

    def test_bypass_activates_with_restricted_colocated(self):
        # intensity=0.75 (RESTRICTED) colocated → both conditions met
        features = self._one_aircraft([_signal(SignalType.NO_FLY, intensity=0.75)])
        intel = classify_regions(features)
        self.assertIsNotNone(_find_intel(intel))

    def test_bypass_activates_with_prohibited_colocated(self):
        # intensity=1.0 (PROHIBITED) colocated
        features = self._one_aircraft([_signal(SignalType.NO_FLY, intensity=1.0)])
        intel = classify_regions(features)
        self.assertIsNotNone(_find_intel(intel))


# ---------------------------------------------------------------------------
# 8. Score decomposition — computed scores must match documented formulas
# ---------------------------------------------------------------------------


class ScoreDecompositionTests(unittest.TestCase):
    def test_anomaly_pure_spike_formula(self):
        """ANOMALY from a pure detection spike with no military, no persistence.

        Contract formula (weights from doc):
          novelty        = 0×0.5 + 0.3 + 1.0×0.2 = 0.5   (spike, no persist)
          ANOMALY score  = (0.30×1 + 0.20×0 + 0.20×0.5 + 0.15×1 + 0.15×1) × 100 = 70.0
        """
        features = build_features(spikes=[(0, 0, 5, 0)])
        intel = classify_regions(features, min_aircraft=1)
        r = _find_intel(intel)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r.all_scores["ANOMALY"], 70.0, delta=0.5)

    def test_routine_dominated_by_persistence(self):
        # Full persistence (12/12) with no anomalous signals → ROUTINE should win.
        features = build_features(recurring=[(0, 0, 12, 20, 0)])
        intel = classify_regions(features, min_aircraft=1)
        r = _find_intel(intel)
        self.assertIsNotNone(r)
        self.assertEqual(r.classification, Classification.ROUTINE)
        routine_score = r.all_scores["ROUTINE"]
        for cls, score in r.all_scores.items():
            if cls != "ROUTINE":
                self.assertGreater(routine_score, score, f"ROUTINE not > {cls}")

    def test_staging_flag_plus_outflow_beats_flag_alone(self):
        """STAGING score is higher when both staging_flag and outflow are present."""
        # Flag only: no outflow
        features_flag = build_features(staging=[(0, 0, 10, 8)])
        r_flag = _find_intel(classify_regions(features_flag, min_aircraft=1))

        # Flag + outflow
        features_both = build_features(
            staging=[(0, 0, 10, 8)],
            movements=[(0, 0, 1, 0, 9, 8, 1.0)],
        )
        r_both = _find_intel(classify_regions(features_both, min_aircraft=1))

        self.assertIsNotNone(r_flag)
        self.assertIsNotNone(r_both)
        self.assertGreater(
            r_both.all_scores["STAGING"], r_flag.all_scores["STAGING"]
        )

    def test_change_score_norm_floor_at_40(self):
        """change_score_norm denominator is floored at 40.0 (FM-3 fix).

        For a single region:
          change=20 → norm = 20/max(20,40) = 0.5
          change=40 → norm = 40/max(40,40) = 1.0

        Without the floor both would equal 1.0 (20/20 == 40/40) and
        ANOMALY scores would be identical.  With the floor they differ.
        """
        features_20 = build_features(
            new_entries=[(0, 0, 5, 0)],
            activity_changes=[_change_row(0, 0, 20.0)],
        )
        features_40 = build_features(
            new_entries=[(0, 0, 5, 0)],
            activity_changes=[_change_row(0, 0, 40.0)],
        )
        intel_20 = classify_regions(features_20, min_aircraft=1)
        intel_40 = classify_regions(features_40, min_aircraft=1)
        r20 = _find_intel(intel_20)
        r40 = _find_intel(intel_40)
        self.assertIsNotNone(r20)
        self.assertIsNotNone(r40)
        self.assertGreater(
            r40.all_scores["ANOMALY"], r20.all_scores["ANOMALY"]
        )


# ---------------------------------------------------------------------------
# 9. Confidence formula — margin + volume + coverage components
# ---------------------------------------------------------------------------


class ConfidenceFormulaTests(unittest.TestCase):
    def test_confidence_increases_with_clear_winner(self):
        """A region with a decisive classification should have higher confidence
        than one with ambiguous, competing signals."""
        # Clear STAGING: flag + high military + outflow all aligned
        features_clear = build_features(
            staging=[(0, 0, 10, 8)],
            movements=[(0, 0, 1, 0, 9, 8, 1.0)],
        )
        r_clear = _find_intel(classify_regions(features_clear, min_aircraft=1))

        # Weak STAGING: flag alone, low military
        features_weak = build_features(staging=[(0, 0, 5, 1)])
        r_weak = _find_intel(classify_regions(features_weak, min_aircraft=1))

        self.assertIsNotNone(r_clear)
        self.assertIsNotNone(r_weak)
        self.assertGreater(r_clear.confidence, r_weak.confidence)

    def test_confidence_increases_with_aircraft_count(self):
        """Higher aircraft count increases volume_pts, raising confidence
        when everything else is equal."""
        # Both have identical ratios and flags; only total count differs.
        features_low = build_features(
            staging=[(0, 0, 2, 2)],
            movements=[(0, 0, 1, 0, 2, 2, 1.0)],
        )
        features_high = build_features(
            staging=[(0, 0, 20, 20)],
            movements=[(0, 0, 1, 0, 18, 18, 1.0)],
        )
        r_low  = _find_intel(classify_regions(features_low,  min_aircraft=1))
        r_high = _find_intel(classify_regions(features_high, min_aircraft=1))
        self.assertIsNotNone(r_low)
        self.assertIsNotNone(r_high)
        self.assertGreater(r_high.confidence, r_low.confidence)

    def test_confidence_formula_components_sum_correctly(self):
        """Verify confidence == int(margin_pts + volume_pts + coverage_pts).

        Construct a known region, extract all the raw values from r.features,
        compute each formula component independently, and assert the result
        matches r.confidence within ±1 (integer truncation).
        """
        import math

        features = build_features(
            staging=[(0, 0, 10, 8)],
            movements=[(0, 0, 1, 0, 9, 8, 1.0)],
            recurring=[(0, 0, 4, 10, 8)],
        )
        intel = classify_regions(features, min_aircraft=1)
        r = _find_intel(intel)
        self.assertIsNotNone(r)

        f = r.features
        winner_score = r.score
        all_scores = list(r.all_scores.values())
        all_scores_sorted = sorted(all_scores, reverse=True)
        runner_up = all_scores_sorted[1] if len(all_scores_sorted) > 1 else 0.0
        margin = winner_score - runner_up

        margin_pts   = min(margin / 40.0, 1.0) * 70.0
        volume_pts   = f.aircraft_density * 20.0
        bool_signals = sum([
            f.spike_flag,
            f.coordination_flag,
            f.staging_flag,
            f.projection_flag,
            f.outflow_count > 0,
            f.inflow_count > 0,
            f.new_aircraft > 0,
            f.recurring_appearances > 0,
        ])
        coverage_pts = (bool_signals / 8) * 10.0
        expected = int(margin_pts + volume_pts + coverage_pts)

        self.assertAlmostEqual(r.confidence, expected, delta=1)


# ---------------------------------------------------------------------------
# 10. External boost formulas
# ---------------------------------------------------------------------------


class ExternalBoostTests(unittest.TestCase):
    def test_spike_not_set_by_external_when_detection_fires_first(self):
        """When detection sets spike_flag, external signal must not claim credit."""
        features = build_features(
            spikes=[(0, 0, 5, 0)],
            external_signals=[_signal(SignalType.NO_FLY, intensity=1.0)],
        )
        f = features[0]
        self.assertFalse(f.spike_set_by_external)

    def test_spike_set_by_external_when_detection_has_no_spike(self):
        """External NO_FLY triggers spike_flag when detection had not set it."""
        features = build_features(
            new_entries=[(0, 0, 5, 0)],
            external_signals=[_signal(SignalType.NO_FLY, intensity=0.60)],
        )
        f = features[0]
        self.assertTrue(f.spike_flag)
        self.assertTrue(f.spike_set_by_external)

    def test_anomaly_boost_applied_only_when_spike_set_by_external(self):
        """Boosted ANOMALY (external spike) must exceed un-boosted (detection spike)
        when all other external contributions are held equal."""
        shared_signal = _signal(SignalType.NO_FLY, intensity=0.60)

        # Boost active: external sets spike
        features_ext = build_features(
            new_entries=[(0, 0, 5, 0)],
            external_signals=[shared_signal],
        )
        r_ext = _find_intel(classify_regions(features_ext))

        # No boost: detection already set spike
        features_det = build_features(
            spikes=[(0, 0, 5, 0)],
            external_signals=[shared_signal],
        )
        r_det = _find_intel(classify_regions(features_det))

        self.assertIsNotNone(r_ext)
        self.assertIsNotNone(r_det)
        self.assertGreater(r_ext.all_scores["ANOMALY"], r_det.all_scores["ANOMALY"])

    def test_anomaly_boost_formula_matches_contract(self):
        """Verify: boosted_score == min(100, base × (1 + min(MAX, strength × PER))).

        Uses a single NO_FLY signal at intensity=0.60 so that:
          strength = 0.60, boost factor = 1.0 + min(0.25, 0.06) = 1.06
        The test computes the expected base score from the contract weight table
        and checks that the final score matches within ±0.5.
        """
        features = build_features(
            new_entries=[(0, 0, 5, 0)],
            external_signals=[_signal(SignalType.NO_FLY, intensity=0.60)],
        )
        f = features[0]
        self.assertTrue(f.spike_set_by_external)

        intel = classify_regions(features)
        r = _find_intel(intel)
        self.assertIsNotNone(r)

        # Reconstruct base score from contract weight table.
        import math
        ac = f.aircraft_count
        mil = f.military_count
        new_ac = f.new_aircraft
        persist = f.recurring_appearances / 12.0
        new_ratio = new_ac / max(ac, 1)
        novelty = min(new_ratio * 0.5 + 0.3 + (1 - persist) * 0.2, 1.0)
        unmilitary = 1.0 - mil / max(ac, 1)
        low_persist = 1.0 - persist
        # max_change floored at 40.0 for single-region batch
        max_change = max(f.change_score, 40.0)
        change_norm = min(f.change_score / max_change, 1.0)

        base = (
            0.30 * 1.0           # spike_flag
            + 0.20 * change_norm
            + 0.20 * novelty
            + 0.15 * unmilitary
            + 0.15 * low_persist
        ) * 100.0

        strength = f.external_signal_strength
        boost = 1.0 + min(_EXT_BOOST_MAX, strength * _EXT_BOOST_PER_STRENGTH)
        expected = min(100.0, base * boost)

        self.assertAlmostEqual(r.all_scores["ANOMALY"], expected, delta=0.5)

    def test_coord_score_boosted_by_maritime(self):
        """COORDINATED_ACTIVITY score must be higher when a MARITIME signal
        is present (maritime_signal_strength > 0 triggers the boost)."""
        base_features = build_features(coordinated=[(0, 0, 8, 4)])
        features_with = build_features(
            coordinated=[(0, 0, 8, 4)],
            external_signals=[_signal(SignalType.MARITIME, intensity=0.8)],
        )
        r_base = _find_intel(classify_regions(base_features, min_aircraft=1))
        r_with = _find_intel(classify_regions(features_with, min_aircraft=1))
        self.assertIsNotNone(r_base)
        self.assertIsNotNone(r_with)
        self.assertGreater(
            r_with.all_scores["COORDINATED_ACTIVITY"],
            r_base.all_scores["COORDINATED_ACTIVITY"],
        )

    def test_cross_domain_bonus_raises_anomaly_score(self):
        """Adding MARITIME alongside NO_FLY must raise ANOMALY score above
        the NO_FLY-only case due to cross-domain bonus (+0.05 each)."""
        sig_nf = _signal(SignalType.NO_FLY,   intensity=0.60)
        sig_mt = _signal(SignalType.MARITIME, intensity=0.60)

        features_nf_only = build_features(
            new_entries=[(0, 0, 5, 0)],
            external_signals=[sig_nf],
        )
        features_both = build_features(
            new_entries=[(0, 0, 5, 0)],
            external_signals=[sig_nf, sig_mt],
        )
        r_nf   = _find_intel(classify_regions(features_nf_only))
        r_both = _find_intel(classify_regions(features_both))
        self.assertIsNotNone(r_nf)
        self.assertIsNotNone(r_both)
        self.assertGreater(
            r_both.all_scores["ANOMALY"],
            r_nf.all_scores["ANOMALY"],
        )


if __name__ == "__main__":
    unittest.main()
