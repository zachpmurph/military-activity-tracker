"""
Contract Validation Tests — docs/02_scoring_contract.md
=======================================================

Three test classes, each tied to a contract section:

  InvariantTests        — feature bounds, score bounds, determinism,
                          additive-only external signals, FM-3 floor
  ThresholdTests        — coordination_flag boundary, bypass tiers,
                          margin/confidence relationship
  ScoreDecompositionTests — formula verification for each profile,
                             boost rules, composite scores (RMA, GT)
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from intelligence.classifier import (
    Classification,
    RegionFeatures,
    _score_region,
    build_features,
    classify_regions,
)
from intelligence.external_features import (
    ExternalSignal,
    SignalType,
    _PROXIMITY_RADIUS,
)

_LAT = 35.0
_LON = 36.0

# Standard context after FM-3 fix: max_change floored at 40.0
_CTX = {"max_inflow": 1, "max_change": 40.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rf(**kw) -> RegionFeatures:
    """Return a RegionFeatures at (_LAT, _LON) with keyword field overrides."""
    f = RegionFeatures(lat=_LAT, lon=_LON)
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def _sig(stype: SignalType, intensity: float, offset: float = 0.0) -> ExternalSignal:
    return ExternalSignal(
        signal_type=stype, lat=_LAT + offset, lon=_LON, intensity=intensity
    )


def _change_row(lat: float, lon: float, score: float, level: str = "HIGH") -> tuple:
    """Minimal 10-element activity_changes row (indices 0,1,2,7,9 used)."""
    return (lat, lon, score, 0, 0, 0, 0, level, 0, "SHORT")


def _rma(s: dict) -> float:
    """Contract §6: Regional Military Action."""
    return (
        0.40 * s["STAGING"]
        + 0.35 * s["PROJECTION"]
        + 0.25 * s["COORDINATED_ACTIVITY"]
    )


def _tension(s: dict) -> float:
    """Contract §6: per-region Global Tension contribution."""
    return (
        0.40 * s["ANOMALY"]
        + 0.35 * s["COORDINATED_ACTIVITY"]
        + 0.25 * max(s["STAGING"], s["PROJECTION"])
    )


# ===========================================================================
# 1.  INVARIANTS
# ===========================================================================

class InvariantTests(unittest.TestCase):

    # --- feature bounds ------------------------------------------------------

    def test_derived_features_stay_in_unit_interval(self):
        """
        Contract §2: all derived properties must be in [0,1].
        flow_balance is the sole exception: [-1, +1].
        Overflowing raw counts (military > aircraft, recurrence > 12) must be clamped.
        """
        f = _rf(
            aircraft_count=100,
            military_count=200,       # would overflow ratio without min(..., 1.0)
            new_aircraft=200,
            recurring_appearances=20, # exceeds 12-bucket max
            outflow_count=50,
            inflow_count=0,
            spike_flag=True,
        )
        for name, lo, hi in [
            ("military_ratio",    0.0, 1.0),
            ("new_entry_ratio",   0.0, 1.0),
            ("persistence_score", 0.0, 1.0),
            ("aircraft_density",  0.0, 1.0),
            ("novelty",           0.0, 1.0),
            ("flow_balance",     -1.0, 1.0),
        ]:
            val = getattr(f, name)
            self.assertGreaterEqual(val, lo, f"{name}={val} below {lo}")
            self.assertLessEqual   (val, hi, f"{name}={val} above {hi}")

    def test_zero_aircraft_does_not_raise_and_returns_zero_ratios(self):
        """
        Division-by-zero guard: aircraft_count=0 must produce 0.0 for ratio features.
        """
        f = _rf(aircraft_count=0, military_count=0, new_aircraft=0)
        self.assertEqual(f.military_ratio,  0.0)
        self.assertEqual(f.new_entry_ratio, 0.0)
        self.assertEqual(f.aircraft_density, 0.0)

    # --- score bounds --------------------------------------------------------

    def test_all_five_class_scores_bounded_zero_to_hundred(self):
        """
        Contract §3: score = dot(weights, fv) × 100.  Must be in [0, 100] for
        any valid input.  Tested with every flag set and all counts at maximum.
        """
        f = _rf(
            aircraft_count=100, military_count=100, new_aircraft=100,
            recurring_appearances=12,
            spike_flag=True, coordination_flag=True,
            staging_flag=True, projection_flag=True,
            outflow_count=10, inflow_count=10,
            change_score=40.0,
        )
        for cls, score in _score_region(f, _CTX).items():
            self.assertGreaterEqual(score,   0.0, f"{cls.value} below 0")
            self.assertLessEqual   (score, 100.0, f"{cls.value} above 100")

    def test_confidence_bounded_zero_to_hundred(self):
        """Contract §4: confidence must not exceed 100."""
        f = _rf(
            aircraft_count=100, military_count=50, recurring_appearances=12,
            staging_flag=True, outflow_count=10,
        )
        r = classify_regions([f], min_aircraft=2)[0]
        self.assertGreaterEqual(r.confidence,   0)
        self.assertLessEqual   (r.confidence, 100)

    # --- determinism ---------------------------------------------------------

    def test_identical_inputs_produce_identical_outputs(self):
        """Contract §3: pipeline must be deterministic."""
        kw = dict(
            spikes=   [(_LAT, _LON, 8, 4)],
            recurring=[(_LAT, _LON, 5, 8, 4)],
        )
        r1 = classify_regions(build_features(**kw))[0]
        r2 = classify_regions(build_features(**kw))[0]
        self.assertEqual(r1.classification, r2.classification)
        self.assertEqual(r1.score,          r2.score)
        self.assertEqual(r1.confidence,     r2.confidence)

    # --- additive-only signals -----------------------------------------------

    def test_weak_satellite_signal_cannot_reduce_existing_change_score(self):
        """
        Contract §5 merge rule: external signals only ADD evidence.
        SATELLITE at intensity=0.50 -> eff=0.50 -> change_score_proxy = 0.50×0.70×40 = 14.
        Existing change_score=30 must not be lowered.
        """
        row     = _change_row(_LAT, _LON, 30.0)
        signal  = _sig(SignalType.SATELLITE, intensity=0.50)
        f = build_features(activity_changes=[row], external_signals=[signal])[0]
        self.assertGreaterEqual(f.change_score, 30.0,
            "Weak SATELLITE must not reduce detection-layer change_score")

    def test_maritime_signal_cannot_clear_detection_spike_flag(self):
        """
        Contract §5: MARITIME has no spike_flag mapping; must never clear
        a spike_flag that the detection layer already set.
        """
        f = build_features(
            spikes=[(_LAT, _LON, 5, 0)],
            external_signals=[_sig(SignalType.MARITIME, intensity=1.0)],
        )[0]
        self.assertTrue(f.spike_flag,
            "Detection-layer spike_flag must survive MARITIME merge")

    def test_stronger_satellite_raises_change_score(self):
        """
        Additive rule in the raising direction: SATELLITE intensity=1.0 at center
        -> change_score_proxy=1.0×0.70×40=28 must raise an existing change_score=20.
        """
        row    = _change_row(_LAT, _LON, 20.0)
        signal = _sig(SignalType.SATELLITE, intensity=1.0)
        f = build_features(activity_changes=[row], external_signals=[signal])[0]
        self.assertGreater(f.change_score, 20.0,
            "Stronger SATELLITE must raise existing change_score")

    # --- FM-3 floor ----------------------------------------------------------

    def test_change_score_norm_uses_floor_of_40(self):
        """
        Contract §2 / FM-3: max_change denominator must be >= 40.

        With change_score=5 (well below the 40 ceiling), the old behaviour
        (max_change=batch_max=5) produced change_score_norm=1.0 and ANOMALY~64.
        After the fix, max_change=40, change_score_norm=0.125, ANOMALY~46.5.

        Region: 5 aircraft all new (new_entry_ratio=1), no spike, no recurrence.
        ANOMALY = 0.30×0 + 0.20×novelty + 0.15×1 + 0.20×(5/40) + 0.15×1
          novelty = 1×0.5 + 0 + 1×0.2 = 0.70
          = 0 + 0.14 + 0.15 + 0.025 + 0.15 = 46.5
        Old denominator (5): +0.20×(5/5-5/40) = +0.175 -> 64.0
        """
        features = build_features(
            new_entries=[(_LAT, _LON, 5, 0)],
            activity_changes=[_change_row(_LAT, _LON, 5.0, "LOW")],
        )
        results = classify_regions(features, min_aircraft=2)
        anomaly = results[0].all_scores["ANOMALY"]
        self.assertLess(anomaly, 55.0,
            "ANOMALY must use max_change=40 floor; old denominator of 5 would give ~64")
        self.assertGreater(anomaly, 35.0,
            "Sanity: ANOMALY must still be nonzero with change_score=5")


# ===========================================================================
# 2.  THRESHOLD BEHAVIOR
# ===========================================================================

class ThresholdTests(unittest.TestCase):

    # --- coordination_flag boundary ------------------------------------------

    def test_maritime_eff_above_half_sets_coordination_flag(self):
        """
        Contract §5: coordination_flag set when effective MARITIME intensity > 0.5.
        intensity=0.60, offset=0 -> eff=0.60 > 0.5 -> flag True.
        """
        f = build_features(
            new_entries=[(_LAT, _LON, 5, 0)],
            external_signals=[_sig(SignalType.MARITIME, intensity=0.60)],
        )[0]
        self.assertTrue(f.coordination_flag,
            "MARITIME eff=0.60 must set coordination_flag")

    def test_maritime_eff_below_half_does_not_set_coordination_flag(self):
        """
        Contract §5: coordination_flag NOT set when effective MARITIME intensity <= 0.5.
        intensity=0.40, offset=0 -> eff=0.40 <= 0.5 -> flag False.
        """
        f = build_features(
            new_entries=[(_LAT, _LON, 5, 0)],
            external_signals=[_sig(SignalType.MARITIME, intensity=0.40)],
        )[0]
        self.assertFalse(f.coordination_flag,
            "MARITIME eff=0.40 must not set coordination_flag")

    def test_coordination_flag_boundary_by_chebyshev_distance(self):
        """
        Contract §2: weight = 1 - (distance / 1.5).
        intensity=1.0, offset=0.74 -> eff = 1 - 0.74/1.5 = 0.5067 > 0.5 -> set.
        intensity=1.0, offset=0.76 -> eff = 1 - 0.76/1.5 = 0.4933 < 0.5 -> not set.
        """
        near = build_features(
            new_entries=[(_LAT, _LON, 5, 0)],
            external_signals=[
                ExternalSignal(SignalType.MARITIME, _LAT + 0.74, _LON, 1.0)
            ],
        )[0]
        far = build_features(
            new_entries=[(_LAT, _LON, 5, 0)],
            external_signals=[
                ExternalSignal(SignalType.MARITIME, _LAT + 0.76, _LON, 1.0)
            ],
        )[0]
        self.assertTrue (near.coordination_flag,
            "MARITIME offset=0.74 (eff=0.507) must set coordination_flag")
        self.assertFalse(far.coordination_flag,
            "MARITIME offset=0.76 (eff=0.493) must not set coordination_flag")

    # --- bypass tiers --------------------------------------------------------

    def test_advisory_intensity_cannot_bypass_min_aircraft(self):
        """
        Contract §4: ADVISORY (0.25) raw < 0.75 threshold -> no bypass.
        """
        features = build_features(
            new_entries=[(_LAT, _LON, 0, 0)],
            external_signals=[_sig(SignalType.NO_FLY, intensity=0.25)],
        )
        self.assertEqual(len(classify_regions(features, min_aircraft=2)), 0,
            "ADVISORY 0.25 must not bypass min_aircraft guard")

    def test_warning_intensity_cannot_bypass_min_aircraft(self):
        """
        Contract §4: WARNING (0.50) raw < 0.75 threshold -> no bypass.
        """
        features = build_features(
            new_entries=[(_LAT, _LON, 0, 0)],
            external_signals=[_sig(SignalType.NO_FLY, intensity=0.50)],
        )
        self.assertEqual(len(classify_regions(features, min_aircraft=2)), 0,
            "WARNING 0.50 must not bypass min_aircraft guard")

    def test_restricted_intensity_at_center_bypasses_min_aircraft(self):
        """
        Contract §4: RESTRICTED (0.75) at center satisfies both bypass conditions:
          raw=0.75 >= 0.75  AND  effective_strength=0.75 >= 0.10.
        """
        features = build_features(
            new_entries=[(_LAT, _LON, 0, 0)],
            external_signals=[_sig(SignalType.NO_FLY, intensity=0.75)],
        )
        self.assertEqual(len(classify_regions(features, min_aircraft=2)), 1,
            "RESTRICTED 0.75 at center must bypass min_aircraft guard")

    def test_restricted_at_edge_blocked_by_strength_floor(self):
        """
        Contract §4 / FM-4: ghost bypass blocked.
        intensity=0.75, offset=1.499:
          raw=0.75 >= 0.75 (passes intensity check)
          weight = 1 - 1.499/1.5 = 0.000667  ->  strength=0.0005 < 0.10 (fails strength check)
        """
        ghost = _sig(SignalType.NO_FLY, intensity=0.75, offset=_PROXIMITY_RADIUS - 0.001)
        features = build_features(new_entries=[(_LAT, _LON, 0, 0)], external_signals=[ghost])
        self.assertEqual(len(classify_regions(features, min_aircraft=2)), 0,
            "Ghost at edge must be blocked by effective-strength floor (FM-4)")

    # --- margin and confidence -----------------------------------------------

    def test_larger_score_margin_produces_higher_confidence(self):
        """
        Contract §4: margin contributes 70 of 100 confidence points.
        A region with a 40+ pt margin must outrank one with a ~10 pt margin,
        given equal aircraft volume.
        """
        # High-margin: staging base with definitive structural evidence
        wide = _rf(
            aircraft_count=10, military_count=8,
            staging_flag=True, outflow_count=10, inflow_count=0,
        )
        # Low-margin: ambiguous mix — features spread across classes
        narrow = _rf(
            aircraft_count=10, military_count=5,
            recurring_appearances=3, change_score=10.0,
        )
        conf_wide   = classify_regions([wide],   min_aircraft=2)[0].confidence
        conf_narrow = classify_regions([narrow], min_aircraft=2)[0].confidence
        self.assertGreater(conf_wide, conf_narrow,
            "Larger score margin must produce higher confidence (margin is 70/100 pts)")


# ===========================================================================
# 3.  SCORE DECOMPOSITION
# ===========================================================================

class ScoreDecompositionTests(unittest.TestCase):

    # --- profile formula verification ----------------------------------------

    def test_anomaly_score_matches_weighted_formula(self):
        """
        Contract §3 ANOMALY: 0.30×spike + 0.20×novelty + 0.15×low_pers
                             + 0.20×change_norm + 0.15×unmilitary.

        Inputs: spike=True, new_aircraft=0, recurrence=0, change_score=20, military=0.
          novelty = 0×0.5 + 0.3 (spike) + 1.0×0.2 = 0.5
          ANOMALY = 0.30×1 + 0.20×0.5 + 0.15×1 + 0.20×0.5 + 0.15×1 = 80.0
        """
        f = _rf(aircraft_count=10, spike_flag=True, change_score=20.0)
        self.assertAlmostEqual(
            _score_region(f, _CTX)[Classification.ANOMALY], 80.0, delta=0.2,
            msg="ANOMALY score must match profile dot-product formula",
        )

    def test_routine_score_matches_weighted_formula(self):
        """
        Contract §3 ROUTINE: 0.50×persist + 0.15×no_spike + 0.15×low_new_entry
                             + 0.05×density + 0.15×unmilitary.

        Inputs: recurring=10, aircraft=10, no spike, no military, no new_aircraft.
          persist=10/12=0.8333, density=log1p(10)/log1p(100)=0.5196
          ROUTINE = 0.50×0.8333 + 0.15×1 + 0.15×1 + 0.05×0.5196 + 0.15×1 = 89.3
        """
        f = _rf(aircraft_count=10, recurring_appearances=10)
        self.assertAlmostEqual(
            _score_region(f, _CTX)[Classification.ROUTINE], 89.3, delta=0.3,
            msg="ROUTINE score must match profile dot-product formula",
        )

    def test_staging_score_matches_weighted_formula(self):
        """
        Contract §3 / Scenario D STAGING: 0.30×staging_flag + 0.30×pos_flow
                                          + 0.25×mil_ratio + 0.10×coord + 0.05×persist.

        Inputs: aircraft=20, military=16 (ratio=0.8), recurring=4 (persist=0.333),
                staging_flag=True, outflow=8, inflow=2 (flow_balance=0.6).
          STAGING = 0.30×1 + 0.30×0.6 + 0.25×0.8 + 0 + 0.05×0.333 = 69.7
        """
        f = _rf(
            aircraft_count=20, military_count=16, recurring_appearances=4,
            staging_flag=True, outflow_count=8, inflow_count=2,
        )
        self.assertAlmostEqual(
            _score_region(f, {"max_inflow": 2, "max_change": 40.0})[Classification.STAGING],
            69.7, delta=0.3,
            msg="STAGING score must match profile dot-product formula",
        )

    # --- ANOMALY boost gating ------------------------------------------------

    def test_detection_spike_prevents_anomaly_boost(self):
        """
        Contract §5: ANOMALY boost fires ONLY when spike_set_by_external=True.
        If the detection layer set spike_flag first, spike_set_by_external stays
        False and the boost multiplier must NOT be applied.
        """
        features = build_features(
            spikes=[(_LAT, _LON, 5, 0)],
            external_signals=[_sig(SignalType.NO_FLY, intensity=0.80)],
        )
        f = features[0]
        self.assertFalse(f.spike_set_by_external,
            "Detection-layer spike must keep spike_set_by_external=False")

        # Score must equal raw _score_region output (no boost factor applied)
        expected = _score_region(f, _CTX)[Classification.ANOMALY]
        actual   = classify_regions(features, min_aircraft=2)[0].all_scores["ANOMALY"]
        self.assertAlmostEqual(actual, expected, delta=0.2,
            msg="No boost: ANOMALY score must equal _score_region output when detection set spike")

    def test_external_only_spike_triggers_anomaly_boost(self):
        """
        Contract §5: when NO_FLY is the first setter of spike_flag,
        spike_set_by_external=True and the boost multiplier is applied.

        Region: 5 new aircraft (no prior spike), NO_FLY at center intensity=0.75.
        After merge: spike_flag=True, change_score=24.0, strength=0.75.
        boost = 1 + min(0.25, 0.75×0.10) = 1.075
        ANOMALY_final = _score_region(merged_f) × 1.075
        """
        features = build_features(
            new_entries=[(_LAT, _LON, 5, 0)],
            external_signals=[_sig(SignalType.NO_FLY, intensity=0.75)],
        )
        f = features[0]
        self.assertTrue(f.spike_set_by_external,
            "NO_FLY-only spike must set spike_set_by_external=True")

        boost          = 1.0 + min(0.25, f.external_signal_strength * 0.10)
        base           = _score_region(f, _CTX)[Classification.ANOMALY]
        expected       = min(100.0, base * boost)
        actual         = classify_regions(features, min_aircraft=2)[0].all_scores["ANOMALY"]
        self.assertAlmostEqual(actual, expected, delta=0.3,
            msg="ANOMALY_final must equal base × (1 + min(0.25, strength×0.10))")

    # --- COORDINATED_ACTIVITY boost ------------------------------------------

    def test_maritime_signal_boosts_coordinated_activity(self):
        """
        Contract §5: when maritime_signal_strength > 0, COORDINATED_ACTIVITY
        is multiplied by coord_boost = 1 + min(0.20, maritime_strength×0.10).
        The boosted score must exceed the pre-external baseline.
        """
        no_ext  = classify_regions(
            build_features(coordinated=[(_LAT, _LON, 10, 5)]),
            min_aircraft=2,
        )[0].all_scores["COORDINATED_ACTIVITY"]

        with_ext = classify_regions(
            build_features(
                coordinated=[(_LAT, _LON, 10, 5)],
                external_signals=[_sig(SignalType.MARITIME, intensity=0.80)],
            ),
            min_aircraft=2,
        )[0].all_scores["COORDINATED_ACTIVITY"]

        self.assertGreater(with_ext, no_ext,
            "MARITIME signal must boost COORDINATED_ACTIVITY score above baseline")

    # --- cross-domain bonus --------------------------------------------------

    def test_cross_domain_bonus_raises_anomaly_above_nofly_only(self):
        """
        Contract §5: NO_FLY + MARITIME -> +0.05 multi-type + +0.05 cross-domain
        bonuses added to ANOMALY boost multiplier.

        Setup: stable region (no detection spike).  NO_FLY sets spike externally.
        Compare ANOMALY with NO_FLY alone vs NO_FLY+MARITIME (same intensities).

        NO_FLY only:   boost = 1 + min(0.25, s×0.10)             [1 type]
        With MARITIME: boost = 1 + min(0.25, s×0.10) + 0.05 + 0.05  [2 types, cross-domain]
        -> ANOMALY_cross > ANOMALY_nofly_only
        """
        region = [(_LAT, _LON, 5, 8, 4)]  # recurring: 5 appearances, 8 aircraft, 4 military

        features_nofly = build_features(
            recurring=region,
            external_signals=[_sig(SignalType.NO_FLY, intensity=0.60)],
        )
        features_cross = build_features(
            recurring=region,
            external_signals=[
                _sig(SignalType.NO_FLY,  intensity=0.60),
                _sig(SignalType.MARITIME, intensity=0.60),
            ],
        )

        anomaly_nofly = classify_regions(features_nofly, min_aircraft=2)[0].all_scores["ANOMALY"]
        anomaly_cross = classify_regions(features_cross, min_aircraft=2)[0].all_scores["ANOMALY"]

        self.assertGreater(anomaly_cross, anomaly_nofly,
            "NO_FLY+MARITIME cross-domain must produce higher ANOMALY than NO_FLY alone")

    # --- composite score formulas --------------------------------------------

    def test_rma_formula_applied_to_staging_scenario(self):
        """
        Contract §6: RMA = 0.40×STAGING + 0.35×PROJECTION + 0.25×COORD.
        Scenario D (staging base): all three scores are nonzero; RMA must fall
        in the moderate tier (41-60) — STAGING dominates, no PROJECTION evidence.
        """
        f = _rf(
            aircraft_count=20, military_count=16, recurring_appearances=4,
            staging_flag=True, outflow_count=8, inflow_count=2,
        )
        s   = classify_regions([f], min_aircraft=2)[0].all_scores
        rma = _rma(s)

        self.assertGreater(rma, 40.0,
            "Staging base must reach moderate RMA tier (41-60)")
        self.assertLess(rma, 80.0,
            "Single-domain staging with no PROJECTION evidence must stay below high tier")

    def test_global_tension_contribution_above_background_for_anomaly_region(self):
        """
        Contract §6: T_i = 0.40×ANOMALY + 0.35×COORD + 0.25×max(STAG, PROJ).
        An ANOMALY-classified region with spike_flag=True, change_score=20 must
        produce T_i > 15 (elevated above background threshold of §6).
        A quiet ROUTINE region must produce T_i <= 30 (not falsely elevated).
        """
        anomaly_f = _rf(
            aircraft_count=8, spike_flag=True, change_score=20.0, recurring_appearances=0,
        )
        routine_f = RegionFeatures(lat=_LAT + 10, lon=_LON)
        routine_f.aircraft_count        = 20
        routine_f.recurring_appearances = 10

        results = classify_regions([anomaly_f, routine_f], min_aircraft=2)
        self.assertEqual(len(results), 2)

        anomaly_r = next(r for r in results if abs(r.lat - _LAT) < 0.01)
        routine_r = next(r for r in results if abs(r.lat - (_LAT + 10)) < 0.01)

        t_anomaly = _tension(anomaly_r.all_scores)
        t_routine = _tension(routine_r.all_scores)

        self.assertGreater(t_anomaly, 15.0,
            "ANOMALY region must contribute above background GT threshold (15)")
        self.assertLess(t_routine, 30.0,
            "Quiet ROUTINE region must stay below elevated GT threshold (30)")
        self.assertGreater(t_anomaly, t_routine,
            "ANOMALY region must contribute more GT than ROUTINE region")

    def test_gt_values_individually_bounded(self):
        """
        Contract §6: each per-region T_i is derived from 0-100 scores with
        weights summing to 1.0, so T_i must be in [0, 100].
        Verified for both extreme (all flags set) and minimal (no flags) regions.
        """
        extreme = _rf(
            aircraft_count=100, military_count=100,
            spike_flag=True, coordination_flag=True,
            staging_flag=True, projection_flag=True,
            change_score=40.0, recurring_appearances=12,
        )
        minimal = _rf(aircraft_count=5)

        for f in (extreme, minimal):
            r = classify_regions([f], min_aircraft=2)
            if not r:
                continue
            t = _tension(r[0].all_scores)
            self.assertGreaterEqual(t,   0.0, f"T_i below 0 for {f}")
            self.assertLessEqual   (t, 100.0, f"T_i above 100 for {f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
