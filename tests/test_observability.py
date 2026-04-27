"""
Tests for external-signal observability instrumentation.

Covers:
  - RegionFeatures default observability fields
      • external_signal_count  == 0
      • external_signal_types  == set()
      • external_signal_max_intensity == 0.0
      • _pre_external_snapshot is None
  - Tracking fields populated correctly by build_features()
      • count, types, max_intensity for a region near signals
      • region far from signals remains untouched
  - ENABLE_EXTERNAL_DEBUG flag
      • Flipping it True/False does not change any feature value
      • Pre-merge snapshot is only set when the flag is True
      • Debug does not alter classify_regions output
  - _compute_external_summary() (query.py)
      • Correct totals, affected count, percentage
      • Correct average intensity
      • Correct count_by_type breakdown
      • Zero-signal edge case returns sensible defaults
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import intelligence.classifier as classifier_mod
from intelligence.classifier import (
    ENABLE_EXTERNAL_DEBUG,
    RegionFeatures,
    build_features,
    classify_regions,
)
from intelligence.external_features import ExternalSignal, SignalType
from intelligence.notam_ingestion import (
    NotamSeverity,
    Notam,
    fetch_notam_signals,
)
from query import _compute_external_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LAT = 34.5   # NOTAM centre used in several tests
_LON = 33.5

def _signal(lat=_LAT, lon=_LON, intensity=0.75, stype=SignalType.NO_FLY):
    return ExternalSignal(signal_type=stype, lat=lat, lon=lon, intensity=intensity)


def _spike(lat=_LAT, lon=_LON, ac=5, mil=2):
    return [(lat, lon, ac, mil)]


def _region_after(signals, lat=_LAT, lon=_LON):
    """Run build_features with a spike at (lat, lon) + provided signals; return the region."""
    features = build_features(spikes=_spike(lat, lon), external_signals=signals)
    return next(
        f for f in features
        if abs(f.lat - lat) < 0.15 and abs(f.lon - lon) < 0.15
    )


# ---------------------------------------------------------------------------
# 1. RegionFeatures default values
# ---------------------------------------------------------------------------

class RegionFeaturesDefaultsTests(unittest.TestCase):

    def test_external_signal_count_default_zero(self):
        r = RegionFeatures(lat=0.0, lon=0.0)
        self.assertEqual(r.external_signal_count, 0)

    def test_external_signal_types_default_empty_set(self):
        r = RegionFeatures(lat=0.0, lon=0.0)
        self.assertIsInstance(r.external_signal_types, set)
        self.assertEqual(len(r.external_signal_types), 0)

    def test_external_signal_max_intensity_default_zero(self):
        r = RegionFeatures(lat=0.0, lon=0.0)
        self.assertAlmostEqual(r.external_signal_max_intensity, 0.0)

    def test_pre_external_snapshot_default_none(self):
        r = RegionFeatures(lat=0.0, lon=0.0)
        self.assertIsNone(r._pre_external_snapshot)

    def test_types_field_is_not_shared_between_instances(self):
        """Mutable default_factory must not be shared across instances."""
        r1 = RegionFeatures(lat=0.0, lon=0.0)
        r2 = RegionFeatures(lat=1.0, lon=1.0)
        r1.external_signal_types.add(SignalType.NO_FLY)
        self.assertEqual(len(r2.external_signal_types), 0)

    def test_new_fields_excluded_from_equality(self):
        """compare=False — observability fields must not affect == checks."""
        r1 = RegionFeatures(lat=0.0, lon=0.0)
        r2 = RegionFeatures(lat=0.0, lon=0.0)
        r1.external_signal_count = 3
        r1.external_signal_types = {SignalType.MARITIME}
        r1.external_signal_max_intensity = 0.9
        self.assertEqual(r1, r2)


# ---------------------------------------------------------------------------
# 2. Tracking fields populated by build_features()
# ---------------------------------------------------------------------------

class TrackingFieldsTests(unittest.TestCase):

    def test_count_for_region_near_one_signal(self):
        region = _region_after([_signal()])
        self.assertEqual(region.external_signal_count, 1)

    def test_count_for_region_near_two_signals(self):
        signals = [_signal(), _signal(intensity=0.5)]
        region = _region_after(signals)
        self.assertEqual(region.external_signal_count, 2)

    def test_types_contains_no_fly_for_notam_signal(self):
        region = _region_after([_signal(stype=SignalType.NO_FLY)])
        self.assertIn(SignalType.NO_FLY, region.external_signal_types)

    def test_types_contains_maritime_signal(self):
        region = _region_after([_signal(stype=SignalType.MARITIME)])
        self.assertIn(SignalType.MARITIME, region.external_signal_types)

    def test_types_contains_multiple_signal_types(self):
        signals = [
            _signal(stype=SignalType.NO_FLY),
            _signal(stype=SignalType.MARITIME),
        ]
        region = _region_after(signals)
        self.assertIn(SignalType.NO_FLY,   region.external_signal_types)
        self.assertIn(SignalType.MARITIME, region.external_signal_types)

    def test_max_intensity_single_signal(self):
        region = _region_after([_signal(intensity=0.75)])
        self.assertAlmostEqual(region.external_signal_max_intensity, 0.75)

    def test_max_intensity_picks_highest(self):
        signals = [_signal(intensity=0.25), _signal(intensity=0.90)]
        region = _region_after(signals)
        self.assertAlmostEqual(region.external_signal_max_intensity, 0.90)

    def test_max_intensity_single_prohibited_notam(self):
        notam = Notam(notam_id="T1", lat=_LAT, lon=_LON, radius_km=50.0,
                      severity=NotamSeverity.PROHIBITED)
        signals = fetch_notam_signals([notam])
        region = _region_after(signals)
        self.assertAlmostEqual(region.external_signal_max_intensity, 1.00)

    def test_region_far_from_signals_has_zero_count(self):
        """Region at (0, 0) must not be influenced by signal at (34.5, 33.5)."""
        far_features = build_features(
            spikes=_spike(0.0, 0.0),
            external_signals=[_signal(_LAT, _LON, 1.0)],
        )
        far_region = next(
            f for f in far_features
            if abs(f.lat) < 0.15 and abs(f.lon) < 0.15
        )
        self.assertEqual(far_region.external_signal_count, 0)

    def test_region_far_from_signals_has_empty_types(self):
        far_features = build_features(
            spikes=_spike(0.0, 0.0),
            external_signals=[_signal(_LAT, _LON, 1.0)],
        )
        far_region = next(
            f for f in far_features
            if abs(f.lat) < 0.15 and abs(f.lon) < 0.15
        )
        self.assertEqual(far_region.external_signal_types, set())

    def test_region_far_from_signals_has_zero_max_intensity(self):
        far_features = build_features(
            spikes=_spike(0.0, 0.0),
            external_signals=[_signal(_LAT, _LON, 1.0)],
        )
        far_region = next(
            f for f in far_features
            if abs(f.lat) < 0.15 and abs(f.lon) < 0.15
        )
        self.assertAlmostEqual(far_region.external_signal_max_intensity, 0.0)

    def test_no_signals_passed_leaves_all_tracking_at_defaults(self):
        features = build_features(spikes=_spike(), external_signals=None)
        for f in features:
            self.assertEqual(f.external_signal_count, 0)
            self.assertEqual(f.external_signal_types, set())
            self.assertAlmostEqual(f.external_signal_max_intensity, 0.0)


# ---------------------------------------------------------------------------
# 3. ENABLE_EXTERNAL_DEBUG flag
# ---------------------------------------------------------------------------

class ExternalDebugFlagTests(unittest.TestCase):
    """Flipping the debug flag must never change any computed feature value."""

    def _features_with_debug(self, debug_on: bool):
        """Return feature list produced with the given debug setting."""
        original = classifier_mod.ENABLE_EXTERNAL_DEBUG
        try:
            classifier_mod.ENABLE_EXTERNAL_DEBUG = debug_on
            return build_features(
                spikes=_spike(),
                external_signals=[_signal(intensity=1.0)],
            )
        finally:
            classifier_mod.ENABLE_EXTERNAL_DEBUG = original

    def test_spike_flag_unchanged_by_debug(self):
        off = self._features_with_debug(False)
        on  = self._features_with_debug(True)
        self.assertEqual(
            [f.spike_flag for f in off],
            [f.spike_flag for f in on],
        )

    def test_change_score_unchanged_by_debug(self):
        off = self._features_with_debug(False)
        on  = self._features_with_debug(True)
        self.assertEqual(
            [f.change_score for f in off],
            [f.change_score for f in on],
        )

    def test_external_signal_count_unchanged_by_debug(self):
        off = self._features_with_debug(False)
        on  = self._features_with_debug(True)
        self.assertEqual(
            [f.external_signal_count for f in off],
            [f.external_signal_count for f in on],
        )

    def test_snapshot_is_none_when_debug_off(self):
        features = self._features_with_debug(False)
        for f in features:
            self.assertIsNone(f._pre_external_snapshot)

    def test_snapshot_is_set_when_debug_on(self):
        """Snapshot must be set for every region when debug is True."""
        features = self._features_with_debug(True)
        # At least one region should exist
        self.assertGreater(len(features), 0)
        for f in features:
            # Every region gets a snapshot dict (even those with no nearby signals)
            self.assertIsNotNone(f._pre_external_snapshot)

    def test_snapshot_contains_expected_keys(self):
        features = self._features_with_debug(True)
        expected_keys = {
            "spike_flag", "coordination_flag",
            "change_score", "inflow_count", "new_aircraft",
        }
        for f in features:
            self.assertEqual(set(f._pre_external_snapshot.keys()), expected_keys)

    def test_classify_regions_output_unchanged_by_debug(self):
        """Debug output in classify_regions must not affect classification results."""
        signals = [_signal(intensity=1.0)]
        spikes  = _spike()

        original = classifier_mod.ENABLE_EXTERNAL_DEBUG
        try:
            classifier_mod.ENABLE_EXTERNAL_DEBUG = False
            features_off = build_features(spikes=spikes, external_signals=signals)
            intel_off    = classify_regions(features_off)

            classifier_mod.ENABLE_EXTERNAL_DEBUG = True
            features_on  = build_features(spikes=spikes, external_signals=signals)
            intel_on     = classify_regions(features_on)
        finally:
            classifier_mod.ENABLE_EXTERNAL_DEBUG = original

        self.assertEqual(len(intel_off), len(intel_on))
        for r_off, r_on in zip(intel_off, intel_on):
            self.assertEqual(r_off.classification, r_on.classification)
            self.assertAlmostEqual(r_off.score, r_on.score)
            self.assertEqual(r_off.confidence,   r_on.confidence)


# ---------------------------------------------------------------------------
# 4. _compute_external_summary()
# ---------------------------------------------------------------------------

class ComputeExternalSummaryTests(unittest.TestCase):

    def _make_feature(self, signal_count=0):
        """RegionFeatures stub with a given external_signal_count."""
        f = RegionFeatures(lat=0.0, lon=0.0)
        f.external_signal_count = signal_count
        return f

    # --- Zero-signal edge case ---

    def test_no_signals_total_is_zero(self):
        summary = _compute_external_summary([], [self._make_feature(0)])
        self.assertEqual(summary["total_signals_processed"], 0)

    def test_no_signals_regions_affected_is_zero(self):
        summary = _compute_external_summary([], [self._make_feature(0)])
        self.assertEqual(summary["regions_affected"], 0)

    def test_no_signals_pct_is_zero(self):
        summary = _compute_external_summary([], [self._make_feature(0)])
        self.assertAlmostEqual(summary["pct_regions_affected"], 0.0)

    def test_no_signals_average_intensity_is_zero(self):
        summary = _compute_external_summary([], [self._make_feature(0)])
        self.assertAlmostEqual(summary["average_intensity"], 0.0)

    def test_no_signals_count_by_type_is_empty(self):
        summary = _compute_external_summary([], [self._make_feature(0)])
        self.assertEqual(summary["count_by_type"], {})

    # --- Total signals processed ---

    def test_total_signals_processed(self):
        signals = [_signal(), _signal(), _signal()]
        summary = _compute_external_summary(signals, [])
        self.assertEqual(summary["total_signals_processed"], 3)

    # --- Regions affected ---

    def test_regions_affected_counts_only_touched_regions(self):
        features = [
            self._make_feature(signal_count=2),   # affected
            self._make_feature(signal_count=0),   # not affected
            self._make_feature(signal_count=1),   # affected
        ]
        summary = _compute_external_summary([_signal()], features)
        self.assertEqual(summary["regions_affected"], 2)

    def test_regions_affected_when_all_affected(self):
        features = [self._make_feature(1), self._make_feature(3)]
        summary = _compute_external_summary([_signal()], features)
        self.assertEqual(summary["regions_affected"], 2)

    def test_regions_affected_when_none_affected(self):
        features = [self._make_feature(0), self._make_feature(0)]
        summary = _compute_external_summary([], features)
        self.assertEqual(summary["regions_affected"], 0)

    # --- Percentage ---

    def test_pct_regions_affected_half(self):
        features = [self._make_feature(1), self._make_feature(0)]
        summary = _compute_external_summary([_signal()], features)
        self.assertAlmostEqual(summary["pct_regions_affected"], 50.0)

    def test_pct_regions_affected_all(self):
        features = [self._make_feature(1), self._make_feature(1)]
        summary = _compute_external_summary([_signal()], features)
        self.assertAlmostEqual(summary["pct_regions_affected"], 100.0)

    def test_pct_rounded_to_one_decimal(self):
        """2/3 ≈ 66.7%"""
        features = [
            self._make_feature(1),
            self._make_feature(1),
            self._make_feature(0),
        ]
        summary = _compute_external_summary([_signal()], features)
        self.assertAlmostEqual(summary["pct_regions_affected"], 66.7)

    # --- Average intensity ---

    def test_average_intensity_single_signal(self):
        summary = _compute_external_summary([_signal(intensity=0.8)], [])
        self.assertAlmostEqual(summary["average_intensity"], 0.800)

    def test_average_intensity_multiple_signals(self):
        signals = [_signal(intensity=0.25), _signal(intensity=0.75)]
        summary = _compute_external_summary(signals, [])
        self.assertAlmostEqual(summary["average_intensity"], 0.500)

    def test_average_intensity_rounded_to_three_decimals(self):
        # 0.10 + 0.20 + 0.30 = 0.60 / 3 = 0.200 exactly
        signals = [
            _signal(intensity=0.10),
            _signal(intensity=0.20),
            _signal(intensity=0.30),
        ]
        summary = _compute_external_summary(signals, [])
        self.assertAlmostEqual(summary["average_intensity"], 0.200, places=3)

    # --- Count by type ---

    def test_count_by_type_single_no_fly(self):
        summary = _compute_external_summary([_signal(stype=SignalType.NO_FLY)], [])
        self.assertEqual(summary["count_by_type"], {"NO_FLY": 1})

    def test_count_by_type_mixed(self):
        signals = [
            _signal(stype=SignalType.NO_FLY),
            _signal(stype=SignalType.NO_FLY),
            _signal(stype=SignalType.MARITIME),
            _signal(stype=SignalType.SATELLITE),
        ]
        summary = _compute_external_summary(signals, [])
        self.assertEqual(summary["count_by_type"]["NO_FLY"],    2)
        self.assertEqual(summary["count_by_type"]["MARITIME"],  1)
        self.assertEqual(summary["count_by_type"]["SATELLITE"], 1)

    def test_count_by_type_uses_enum_value_strings(self):
        summary = _compute_external_summary([_signal(stype=SignalType.MARITIME)], [])
        self.assertIn("MARITIME", summary["count_by_type"])

    # --- Integration: real build_features output ---

    def test_summary_with_real_build_features_output(self):
        """End-to-end: signals affect exactly the nearby region."""
        signals = [_signal(_LAT, _LON, intensity=1.0, stype=SignalType.NO_FLY)]
        features = build_features(
            spikes=_spike(_LAT, _LON),        # one region near the signal
            external_signals=signals,
        )
        summary = _compute_external_summary(signals, features)
        self.assertEqual(summary["total_signals_processed"], 1)
        self.assertGreaterEqual(summary["regions_affected"], 1)
        self.assertAlmostEqual(summary["average_intensity"], 1.0)
        self.assertEqual(summary["count_by_type"].get("NO_FLY", 0), 1)

    def test_summary_with_mock_notam_signals(self):
        """fetch_notam_signals() mock produces a plausible summary."""
        signals  = fetch_notam_signals()
        features = build_features(external_signals=signals)
        summary  = _compute_external_summary(signals, features)
        self.assertEqual(summary["total_signals_processed"], len(signals))
        self.assertGreaterEqual(summary["pct_regions_affected"], 0.0)
        self.assertLessEqual(summary["pct_regions_affected"], 100.0)
        self.assertIn("NO_FLY", summary["count_by_type"])


if __name__ == "__main__":
    unittest.main()
