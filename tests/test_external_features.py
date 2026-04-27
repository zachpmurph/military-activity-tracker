"""
Tests for intelligence/external_features.py

Covers:
  - ExternalSignal        — construction, intensity clamping, metadata
  - map_external_signals_to_features()
      • NO_FLY, MARITIME, SATELLITE mappings
      • No signals / no nearby signals → empty dict
      • Multiple same-type signals aggregate by max
      • Conflicting signal types sharing a feature → max wins
      • All output values bounded [0, 1]
  - merge_external_features()
      • Flags set correctly (spike_flag, coordination_flag)
      • change_score computed from change_score_norm
      • inflow_count computed from inflow_norm
      • novelty proxy via new_aircraft
      • Never reduces existing field values
      • No-op on empty ext_map
  - build_features() integration
      • signals=None → identical to baseline (no-op)
      • signals provided → fields augmented
      • Augmented fields feed into scoring normally
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intelligence.classifier import (
    Classification,
    RegionFeatures,
    build_features,
    classify_regions,
)
from intelligence.external_features import (
    ExternalSignal,
    SignalType,
    _CHANGE_SCORE_MAX,
    _INFLOW_MAX,
    _PROXIMITY_RADIUS,
    _clamp,
    _signals_near,
    map_external_signals_to_features,
    merge_external_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _region(lat: float = 0.0, lon: float = 0.0, **kwargs) -> RegionFeatures:
    """Construct a RegionFeatures with known defaults."""
    return RegionFeatures(lat=lat, lon=lon, aircraft_count=10, **kwargs)


def _signal(
    stype: SignalType,
    lat: float = 0.0,
    lon: float = 0.0,
    intensity: float = 1.0,
    **meta,
) -> ExternalSignal:
    return ExternalSignal(signal_type=stype, lat=lat, lon=lon, intensity=intensity,
                          metadata=meta)


# ---------------------------------------------------------------------------
# ExternalSignal construction
# ---------------------------------------------------------------------------

class ExternalSignalTests(unittest.TestCase):

    def test_intensity_stored_correctly(self):
        s = _signal(SignalType.NO_FLY, intensity=0.7)
        self.assertAlmostEqual(s.intensity, 0.7)

    def test_intensity_clamped_above_one(self):
        s = _signal(SignalType.NO_FLY, intensity=2.5)
        self.assertEqual(s.intensity, 1.0)

    def test_intensity_clamped_below_zero(self):
        s = _signal(SignalType.MARITIME, intensity=-0.3)
        self.assertEqual(s.intensity, 0.0)

    def test_intensity_zero_allowed(self):
        s = _signal(SignalType.SATELLITE, intensity=0.0)
        self.assertEqual(s.intensity, 0.0)

    def test_metadata_stored(self):
        s = ExternalSignal(
            signal_type=SignalType.NO_FLY,
            lat=10.0, lon=20.0, intensity=0.5,
            metadata={"source": "notam", "id": "A1234/24"},
        )
        self.assertEqual(s.metadata["source"], "notam")
        self.assertEqual(s.metadata["id"], "A1234/24")

    def test_default_metadata_is_empty_dict(self):
        s = _signal(SignalType.SATELLITE)
        self.assertEqual(s.metadata, {})

    def test_signal_type_enum_values(self):
        self.assertEqual(SignalType.NO_FLY.value,    "NO_FLY")
        self.assertEqual(SignalType.MARITIME.value,  "MARITIME")
        self.assertEqual(SignalType.SATELLITE.value, "SATELLITE")


# ---------------------------------------------------------------------------
# _signals_near helper
# ---------------------------------------------------------------------------

class SignalsNearTests(unittest.TestCase):

    def test_exact_match_included(self):
        s = _signal(SignalType.NO_FLY, lat=10.0, lon=20.0)
        self.assertEqual(_signals_near([s], 10.0, 20.0), [s])

    def test_within_radius_included(self):
        s = _signal(SignalType.NO_FLY, lat=10.4, lon=20.4)
        self.assertEqual(_signals_near([s], 10.0, 20.0), [s])

    def test_outside_radius_excluded(self):
        s = _signal(SignalType.NO_FLY, lat=10.6, lon=20.0)
        self.assertEqual(_signals_near([s], 10.0, 20.0), [])

    def test_custom_radius(self):
        s = _signal(SignalType.NO_FLY, lat=11.0, lon=20.0)
        self.assertEqual(_signals_near([s], 10.0, 20.0, radius=1.5), [s])
        self.assertEqual(_signals_near([s], 10.0, 20.0, radius=0.9), [])

    def test_empty_list_returns_empty(self):
        self.assertEqual(_signals_near([], 0.0, 0.0), [])

    def test_mixed_proximity(self):
        near = _signal(SignalType.NO_FLY, lat=0.3, lon=0.0)
        far  = _signal(SignalType.NO_FLY, lat=5.0, lon=0.0)
        result = _signals_near([near, far], 0.0, 0.0)
        self.assertIn(near, result)
        self.assertNotIn(far, result)


# ---------------------------------------------------------------------------
# map_external_signals_to_features
# ---------------------------------------------------------------------------

class MapSignalsTests(unittest.TestCase):

    def test_empty_signals_returns_empty_dict(self):
        region = _region()
        self.assertEqual(map_external_signals_to_features([], region), {})

    def test_no_nearby_signals_returns_empty_dict(self):
        s      = _signal(SignalType.NO_FLY, lat=90.0, lon=0.0)
        region = _region(lat=0.0, lon=0.0)
        self.assertEqual(map_external_signals_to_features([s], region), {})

    # ── NO_FLY mapping ───────────────────────────────────────────────────────

    def test_no_fly_sets_spike_flag(self):
        s    = _signal(SignalType.NO_FLY, intensity=1.0)
        feat = map_external_signals_to_features([s], _region())
        self.assertIn("spike_flag", feat)
        self.assertAlmostEqual(feat["spike_flag"], 1.0)

    def test_no_fly_sets_change_score_norm(self):
        s    = _signal(SignalType.NO_FLY, intensity=1.0)
        feat = map_external_signals_to_features([s], _region())
        self.assertIn("change_score_norm", feat)
        self.assertAlmostEqual(feat["change_score_norm"], 0.8, places=5)

    def test_no_fly_intensity_scaled(self):
        s    = _signal(SignalType.NO_FLY, intensity=0.5)
        feat = map_external_signals_to_features([s], _region())
        self.assertAlmostEqual(feat["spike_flag"],        0.5,  places=5)
        self.assertAlmostEqual(feat["change_score_norm"], 0.4,  places=5)

    # ── MARITIME mapping ─────────────────────────────────────────────────────

    def test_maritime_sets_coordination_flag(self):
        s    = _signal(SignalType.MARITIME, intensity=1.0)
        feat = map_external_signals_to_features([s], _region())
        self.assertIn("coordination_flag", feat)
        self.assertAlmostEqual(feat["coordination_flag"], 1.0)

    def test_maritime_sets_inflow_norm(self):
        s    = _signal(SignalType.MARITIME, intensity=1.0)
        feat = map_external_signals_to_features([s], _region())
        self.assertIn("inflow_norm", feat)
        self.assertAlmostEqual(feat["inflow_norm"], 0.6, places=5)

    def test_maritime_intensity_scaled(self):
        s    = _signal(SignalType.MARITIME, intensity=0.5)
        feat = map_external_signals_to_features([s], _region())
        self.assertAlmostEqual(feat["coordination_flag"], 0.5,  places=5)
        self.assertAlmostEqual(feat["inflow_norm"],       0.3,  places=5)

    # ── SATELLITE mapping ────────────────────────────────────────────────────

    def test_satellite_sets_change_score_norm(self):
        s    = _signal(SignalType.SATELLITE, intensity=1.0)
        feat = map_external_signals_to_features([s], _region())
        self.assertIn("change_score_norm", feat)
        self.assertAlmostEqual(feat["change_score_norm"], 0.7, places=5)

    def test_satellite_sets_novelty(self):
        s    = _signal(SignalType.SATELLITE, intensity=1.0)
        feat = map_external_signals_to_features([s], _region())
        self.assertIn("novelty", feat)
        self.assertAlmostEqual(feat["novelty"], 0.5, places=5)

    def test_satellite_intensity_scaled(self):
        s    = _signal(SignalType.SATELLITE, intensity=0.4)
        feat = map_external_signals_to_features([s], _region())
        self.assertAlmostEqual(feat["change_score_norm"], 0.28, places=5)
        self.assertAlmostEqual(feat["novelty"],           0.20, places=5)

    # ── All output values bounded ────────────────────────────────────────────

    def test_all_output_values_in_0_1(self):
        signals = [
            _signal(SignalType.NO_FLY,    intensity=1.0),
            _signal(SignalType.MARITIME,  intensity=1.0),
            _signal(SignalType.SATELLITE, intensity=1.0),
        ]
        feat = map_external_signals_to_features(signals, _region())
        for key, val in feat.items():
            self.assertGreaterEqual(val, 0.0, f"{key} below 0")
            self.assertLessEqual(val,   1.0, f"{key} above 1")

    # ── Multiple same-type signals aggregate by max ───────────────────────────

    def test_multiple_no_fly_signals_take_max(self):
        """Two NO_FLY signals near the region; only the stronger one should win."""
        s_lo = _signal(SignalType.NO_FLY, intensity=0.3)
        s_hi = _signal(SignalType.NO_FLY, intensity=0.9)
        feat = map_external_signals_to_features([s_lo, s_hi], _region())
        self.assertAlmostEqual(feat["spike_flag"], 0.9, places=5)

    def test_multiple_maritime_signals_take_max(self):
        signals = [
            _signal(SignalType.MARITIME, intensity=0.2),
            _signal(SignalType.MARITIME, intensity=0.7),
            _signal(SignalType.MARITIME, intensity=0.5),
        ]
        feat = map_external_signals_to_features(signals, _region())
        self.assertAlmostEqual(feat["coordination_flag"], 0.7, places=5)

    # ── Conflicting signal types sharing a feature ───────────────────────────

    def test_conflicting_change_score_norm_takes_max(self):
        """
        NO_FLY (intensity=0.5) → change_score_norm = 0.40
        SATELLITE (intensity=1.0) → change_score_norm = 0.70
        Expected: max → 0.70
        """
        signals = [
            _signal(SignalType.NO_FLY,    intensity=0.5),
            _signal(SignalType.SATELLITE, intensity=1.0),
        ]
        feat = map_external_signals_to_features(signals, _region())
        self.assertAlmostEqual(feat["change_score_norm"], 0.70, places=5)

    def test_conflicting_strong_no_fly_beats_weak_satellite(self):
        """
        NO_FLY (intensity=1.0) → change_score_norm = 0.80
        SATELLITE (intensity=0.5) → change_score_norm = 0.35
        Expected: max → 0.80
        """
        signals = [
            _signal(SignalType.NO_FLY,    intensity=1.0),
            _signal(SignalType.SATELLITE, intensity=0.5),
        ]
        feat = map_external_signals_to_features(signals, _region())
        self.assertAlmostEqual(feat["change_score_norm"], 0.80, places=5)

    def test_all_three_types_together(self):
        """All three types present; each feature resolves to its max contribution."""
        signals = [
            _signal(SignalType.NO_FLY,    intensity=0.8),
            _signal(SignalType.MARITIME,  intensity=0.6),
            _signal(SignalType.SATELLITE, intensity=0.5),
        ]
        feat = map_external_signals_to_features(signals, _region())
        # spike_flag: only NO_FLY contributes → 0.8
        self.assertAlmostEqual(feat["spike_flag"],        0.8,  places=5)
        # coordination_flag: only MARITIME → 0.6
        self.assertAlmostEqual(feat["coordination_flag"], 0.6,  places=5)
        # change_score_norm: NO_FLY=0.64, SATELLITE=0.35 → max=0.64
        self.assertAlmostEqual(feat["change_score_norm"], 0.64, places=5)
        # inflow_norm: only MARITIME → 0.36
        self.assertAlmostEqual(feat["inflow_norm"],       0.36, places=5)
        # novelty: only SATELLITE → 0.25
        self.assertAlmostEqual(feat["novelty"],           0.25, places=5)


# ---------------------------------------------------------------------------
# merge_external_features
# ---------------------------------------------------------------------------

class MergeExternalFeaturesTests(unittest.TestCase):

    def test_empty_ext_map_is_noop(self):
        region = _region(lat=0.0, lon=0.0, aircraft_count=10)
        before_spike = region.spike_flag
        before_cs    = region.change_score
        merge_external_features(region, {})
        self.assertEqual(region.spike_flag,   before_spike)
        self.assertEqual(region.change_score, before_cs)

    # ── spike_flag ───────────────────────────────────────────────────────────

    def test_spike_flag_set_when_value_above_threshold(self):
        region = _region()
        self.assertFalse(region.spike_flag)
        merge_external_features(region, {"spike_flag": 0.9})
        self.assertTrue(region.spike_flag)

    def test_spike_flag_not_set_below_threshold(self):
        region = _region()
        merge_external_features(region, {"spike_flag": 0.4})
        self.assertFalse(region.spike_flag)

    def test_spike_flag_never_cleared(self):
        """A region with spike_flag=True keeps it True regardless of ext_map value."""
        region = _region(spike_flag=True)
        merge_external_features(region, {"spike_flag": 0.0})
        self.assertTrue(region.spike_flag)

    # ── coordination_flag ────────────────────────────────────────────────────

    def test_coordination_flag_set_when_value_above_threshold(self):
        region = _region()
        self.assertFalse(region.coordination_flag)
        merge_external_features(region, {"coordination_flag": 0.8})
        self.assertTrue(region.coordination_flag)

    def test_coordination_flag_never_cleared(self):
        region = _region(coordination_flag=True)
        merge_external_features(region, {"coordination_flag": 0.0})
        self.assertTrue(region.coordination_flag)

    # ── change_score ─────────────────────────────────────────────────────────

    def test_change_score_computed_from_norm(self):
        region = _region()
        merge_external_features(region, {"change_score_norm": 0.5})
        expected = 0.5 * _CHANGE_SCORE_MAX
        self.assertAlmostEqual(region.change_score, expected, places=6)

    def test_change_score_takes_max_not_overwrite(self):
        """Existing higher change_score is preserved."""
        existing = 30.0
        region   = _region(change_score=existing)
        merge_external_features(region, {"change_score_norm": 0.5})   # → 20.0
        self.assertAlmostEqual(region.change_score, existing, places=6)

    def test_change_score_updated_when_external_is_higher(self):
        region = _region(change_score=5.0)
        merge_external_features(region, {"change_score_norm": 1.0})   # → 40.0
        self.assertAlmostEqual(region.change_score, _CHANGE_SCORE_MAX, places=6)

    def test_change_score_bounded_by_change_score_max(self):
        """change_score_norm=1.0 must produce exactly _CHANGE_SCORE_MAX."""
        region = _region()
        merge_external_features(region, {"change_score_norm": 1.0})
        self.assertLessEqual(region.change_score, _CHANGE_SCORE_MAX)

    # ── inflow_count ─────────────────────────────────────────────────────────

    def test_inflow_count_computed_from_norm(self):
        region = _region()
        merge_external_features(region, {"inflow_norm": 1.0})
        self.assertEqual(region.inflow_count, _INFLOW_MAX)

    def test_inflow_count_proportional_to_norm(self):
        region = _region()
        merge_external_features(region, {"inflow_norm": 0.5})
        self.assertEqual(region.inflow_count, int(0.5 * _INFLOW_MAX))

    def test_inflow_count_takes_max(self):
        """Existing higher inflow_count is preserved."""
        region = _region(inflow_count=8)
        merge_external_features(region, {"inflow_norm": 0.3})   # → int(0.3*10)=3
        self.assertEqual(region.inflow_count, 8)

    # ── novelty proxy (new_aircraft) ─────────────────────────────────────────

    def test_novelty_raises_new_aircraft(self):
        region = _region(aircraft_count=10, new_aircraft=0)
        merge_external_features(region, {"novelty": 0.5})
        self.assertEqual(region.new_aircraft, 5)   # int(0.5 * 10)

    def test_novelty_capped_at_aircraft_count(self):
        region = _region(aircraft_count=10)
        merge_external_features(region, {"novelty": 1.0})
        self.assertLessEqual(region.new_aircraft, region.aircraft_count)

    def test_novelty_skipped_when_no_aircraft(self):
        """aircraft_count=0 → no proxy update (would divide by zero)."""
        region = RegionFeatures(lat=0.0, lon=0.0, aircraft_count=0, new_aircraft=0)
        merge_external_features(region, {"novelty": 1.0})
        self.assertEqual(region.new_aircraft, 0)

    def test_novelty_takes_max_new_aircraft(self):
        region = _region(aircraft_count=10, new_aircraft=8)
        merge_external_features(region, {"novelty": 0.3})   # → int(0.3*10)=3
        self.assertEqual(region.new_aircraft, 8)

    # ── general: never reduces ────────────────────────────────────────────────

    def test_no_field_is_ever_reduced(self):
        """After merge, every field must be ≥ its pre-merge value."""
        region_before = _region(
            aircraft_count=10, change_score=15.0,
            inflow_count=4, new_aircraft=2,
            spike_flag=True, coordination_flag=True,
        )
        ext_map = {
            "spike_flag":        0.0,
            "coordination_flag": 0.0,
            "change_score_norm": 0.1,   # → 4.0, below existing 15.0
            "inflow_norm":       0.2,   # → 2, below existing 4
            "novelty":           0.1,   # → 1, below existing 2
        }
        merge_external_features(region_before, ext_map)
        self.assertTrue(region_before.spike_flag)
        self.assertTrue(region_before.coordination_flag)
        self.assertAlmostEqual(region_before.change_score, 15.0)
        self.assertEqual(region_before.inflow_count, 4)
        self.assertEqual(region_before.new_aircraft, 2)


# ---------------------------------------------------------------------------
# build_features integration
# ---------------------------------------------------------------------------

class BuildFeaturesIntegrationTests(unittest.TestCase):

    def _spike_region_features(self) -> list:
        return build_features(
            spikes=[(10.0, 20.0, 8, 2)],
        )

    def test_no_signals_is_noop(self):
        """build_features with signals=None must produce identical output."""
        baseline = self._spike_region_features()
        with_none = build_features(
            spikes=[(10.0, 20.0, 8, 2)],
            external_signals=None,
        )
        self.assertEqual(len(baseline), len(with_none))
        b, w = baseline[0], with_none[0]
        self.assertEqual(b.spike_flag,        w.spike_flag)
        self.assertEqual(b.coordination_flag, w.coordination_flag)
        self.assertAlmostEqual(b.change_score, w.change_score)
        self.assertEqual(b.inflow_count,      w.inflow_count)

    def test_external_signal_sets_coordination_flag(self):
        """MARITIME signal near a spike region adds coordination_flag=True."""
        result = build_features(
            spikes=[(10.0, 20.0, 8, 2)],
            external_signals=[
                ExternalSignal(
                    signal_type=SignalType.MARITIME,
                    lat=10.1, lon=20.1, intensity=0.9,
                )
            ],
        )
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].coordination_flag)

    def test_external_signal_raises_change_score(self):
        """SATELLITE signal near a region raises change_score."""
        baseline = build_features(spikes=[(10.0, 20.0, 8, 2)])
        augmented = build_features(
            spikes=[(10.0, 20.0, 8, 2)],
            external_signals=[
                ExternalSignal(
                    signal_type=SignalType.SATELLITE,
                    lat=10.0, lon=20.0, intensity=1.0,
                )
            ],
        )
        self.assertGreater(augmented[0].change_score, baseline[0].change_score)

    def test_distant_signal_has_no_effect(self):
        """Signal far from the region must not alter its features."""
        baseline  = build_features(spikes=[(10.0, 20.0, 8, 2)])
        augmented = build_features(
            spikes=[(10.0, 20.0, 8, 2)],
            external_signals=[
                ExternalSignal(
                    signal_type=SignalType.NO_FLY,
                    lat=50.0, lon=80.0, intensity=1.0,
                )
            ],
        )
        b, a = baseline[0], augmented[0]
        self.assertEqual(b.spike_flag,   a.spike_flag)
        self.assertAlmostEqual(b.change_score, a.change_score)

    def test_no_fly_signal_sets_spike_flag(self):
        """NO_FLY signal with intensity > 0.5 must set spike_flag."""
        result = build_features(
            recurring=[(10.0, 20.0, 5, 15, 0)],   # non-spike region
            external_signals=[
                ExternalSignal(
                    signal_type=SignalType.NO_FLY,
                    lat=10.0, lon=20.0, intensity=0.8,
                )
            ],
        )
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].spike_flag)

    def test_multiple_signals_all_applied(self):
        """Multiple signal types near the same region all contribute."""
        result = build_features(
            recurring=[(10.0, 20.0, 3, 10, 0)],
            external_signals=[
                ExternalSignal(SignalType.NO_FLY,    10.0, 20.0, 0.9),
                ExternalSignal(SignalType.MARITIME,  10.1, 20.1, 0.8),
                ExternalSignal(SignalType.SATELLITE, 10.2, 20.2, 0.7),
            ],
        )
        r = result[0]
        self.assertTrue(r.spike_flag)
        self.assertTrue(r.coordination_flag)
        self.assertGreater(r.change_score, 0.0)
        self.assertGreater(r.inflow_count, 0)

    def test_augmented_features_flow_into_scoring(self):
        """
        A region that is ROUTINE with only recurring data should become
        COORDINATED_ACTIVITY or have a higher coordination score once a
        MARITIME + NO_FLY signal is injected (spot-check: score > baseline).
        """
        features_base = build_features(
            recurring=[(10.0, 20.0, 3, 10, 0)],
        )
        features_ext = build_features(
            recurring=[(10.0, 20.0, 3, 10, 0)],
            external_signals=[
                ExternalSignal(SignalType.MARITIME,  10.0, 20.0, 1.0),
                ExternalSignal(SignalType.NO_FLY,    10.0, 20.0, 1.0),
                ExternalSignal(SignalType.SATELLITE, 10.0, 20.0, 1.0),
            ],
        )
        results_base = classify_regions(features_base, min_aircraft=1)
        results_ext  = classify_regions(features_ext,  min_aircraft=1)
        self.assertEqual(len(results_base), 1)
        self.assertEqual(len(results_ext),  1)
        # At minimum the all_scores dict for COORDINATED_ACTIVITY should be
        # higher in the externally-augmented version.
        base_ca = results_base[0].all_scores.get("COORDINATED_ACTIVITY", 0.0)
        ext_ca  = results_ext[0].all_scores.get("COORDINATED_ACTIVITY",  0.0)
        self.assertGreater(ext_ca, base_ca,
            "External MARITIME+NO_FLY signals should raise COORDINATED_ACTIVITY score")

    def test_empty_external_signals_list_is_noop(self):
        """Passing an empty list is equivalent to passing None."""
        with_none  = build_features(spikes=[(10.0, 20.0, 8, 2)], external_signals=None)
        with_empty = build_features(spikes=[(10.0, 20.0, 8, 2)], external_signals=[])
        b, e = with_none[0], with_empty[0]
        self.assertEqual(b.spike_flag,        e.spike_flag)
        self.assertEqual(b.coordination_flag, e.coordination_flag)
        self.assertAlmostEqual(b.change_score, e.change_score)


# ---------------------------------------------------------------------------
# Safeguard / boundary tests
# ---------------------------------------------------------------------------

class SafeguardTests(unittest.TestCase):

    def test_change_score_norm_bounded_at_1(self):
        """Even with intensity=1.0, change_score ≤ _CHANGE_SCORE_MAX."""
        region = _region()
        ext_map = map_external_signals_to_features(
            [_signal(SignalType.NO_FLY, intensity=1.0)], region
        )
        merge_external_features(region, ext_map)
        self.assertLessEqual(region.change_score, _CHANGE_SCORE_MAX)

    def test_inflow_count_bounded_at_inflow_max(self):
        """Even with intensity=1.0, inflow_count ≤ _INFLOW_MAX."""
        region = _region()
        ext_map = map_external_signals_to_features(
            [_signal(SignalType.MARITIME, intensity=1.0)], region
        )
        merge_external_features(region, ext_map)
        self.assertLessEqual(region.inflow_count, _INFLOW_MAX)

    def test_new_aircraft_bounded_by_aircraft_count(self):
        """novelty proxy must never exceed aircraft_count."""
        region = _region(aircraft_count=5)
        ext_map = map_external_signals_to_features(
            [_signal(SignalType.SATELLITE, intensity=1.0)], region
        )
        merge_external_features(region, ext_map)
        self.assertLessEqual(region.new_aircraft, region.aircraft_count)

    def test_map_output_always_in_0_1(self):
        """map_external_signals_to_features output values must all be in [0, 1]."""
        for stype in SignalType:
            s    = _signal(stype, intensity=1.0)
            feat = map_external_signals_to_features([s], _region())
            for key, val in feat.items():
                self.assertGreaterEqual(val, 0.0, f"{stype.value}/{key} below 0")
                self.assertLessEqual(val,   1.0, f"{stype.value}/{key} above 1")

    def test_existing_tests_unaffected(self):
        """
        Calling build_features with no external_signals argument must produce
        the same result as before the feature was added (regression guard).
        """
        features = build_features(
            coordinated=[(34.1, -117.2, 6, 3)],
            spikes     =[(34.1, -117.2, 8, 4)],
            recurring  =[(34.1, -117.2, 5, 15, 5)],
        )
        self.assertEqual(len(features), 1)
        f = features[0]
        self.assertTrue(f.coordination_flag)
        self.assertTrue(f.spike_flag)
        self.assertEqual(f.recurring_appearances, 5)
        self.assertEqual(f.aircraft_count, 15)
        self.assertEqual(f.military_count, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
