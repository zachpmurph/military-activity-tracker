"""
Tests for intelligence/maritime_ingestion.py
─────────────────────────────────────────────
Covers:
  - _normalize_density()
      • zero count → 0.0
      • proportional values at various counts
      • count at max → 1.0
      • count above max → clamped to 1.0
  - _parse_gfw_response()
      • valid entry → one signal per grid cell
      • signal type is MARITIME
      • lat/lon binned to _GRID_RESOLUTION grid
      • metadata: source="maritime", vessel_count present
      • missing position key → skipped
      • null position → skipped
      • out-of-range lat/lon → skipped
      • empty entries list → empty result
      • missing "entries" key → empty result
      • multiple entries same cell → one signal with summed count
      • multiple entries different cells → separate signals
      • intensity derived from normalised count
  - _load_fallback_signals()
      • returns non-empty list
      • all MARITIME type
      • intensities in [0, 1]
      • metadata has source and vessel_count
      • items with bad coords are skipped (via patched file)
  - fetch_maritime_signals()
      • no GFW_API_KEY → returns non-empty list (from fallback)
      • returns list of ExternalSignal
      • all MARITIME type
      • all intensities in [0, 1]
  - ExternalSignal contract
      • fallback signals satisfy full contract
  - Integration with build_features()
      • MARITIME signal sets coordination_flag=True for nearby region
      • MARITIME signal raises inflow_count > 0 for nearby region
      • region far from all signals is unaffected
      • combined NOTAM + MARITIME triggers cross-domain potential
        (both NO_FLY and MARITIME in region.external_signal_types)
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intelligence.maritime_ingestion import (
    _normalize_density,
    _parse_gfw_response,
    _load_fallback_signals,
    fetch_maritime_signals,
    _MAX_VESSELS_PER_CELL,
    _GRID_RESOLUTION,
)
from intelligence.external_features import ExternalSignal, SignalType
from intelligence.classifier import build_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gfw_entry(lat: float, lon: float) -> dict:
    """Minimal GFW Events API entry with a valid position."""
    return {
        "type":     "fishing",
        "position": {"lat": lat, "lon": lon},
    }


def _gfw_response(*entries) -> dict:
    return {"entries": list(entries)}


# ---------------------------------------------------------------------------
# _normalize_density() tests
# ---------------------------------------------------------------------------

class NormalizeDensityTests(unittest.TestCase):

    def test_zero_count_returns_zero(self):
        self.assertEqual(_normalize_density(0), 0.0)

    def test_negative_count_returns_zero(self):
        self.assertEqual(_normalize_density(-5), 0.0)

    def test_half_max_count_returns_half(self):
        self.assertAlmostEqual(_normalize_density(_MAX_VESSELS_PER_CELL // 2), 0.5)

    def test_at_max_count_returns_one(self):
        self.assertAlmostEqual(_normalize_density(_MAX_VESSELS_PER_CELL), 1.0)

    def test_above_max_clamped_to_one(self):
        self.assertAlmostEqual(_normalize_density(_MAX_VESSELS_PER_CELL * 3), 1.0)

    def test_single_vessel_small_intensity(self):
        result = _normalize_density(1)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 0.5)

    def test_custom_max_count(self):
        self.assertAlmostEqual(_normalize_density(5, max_count=10), 0.5)

    def test_result_always_in_unit_range(self):
        for count in range(0, _MAX_VESSELS_PER_CELL * 3):
            result = _normalize_density(count)
            self.assertGreaterEqual(result, 0.0, f"count={count}")
            self.assertLessEqual(result, 1.0, f"count={count}")


# ---------------------------------------------------------------------------
# _parse_gfw_response() tests
# ---------------------------------------------------------------------------

class ParseGfwResponseTests(unittest.TestCase):

    def test_valid_entry_returns_one_signal(self):
        result = _parse_gfw_response(_gfw_response(_gfw_entry(34.0, 56.0)))
        self.assertEqual(len(result), 1)

    def test_signal_type_is_maritime(self):
        result = _parse_gfw_response(_gfw_response(_gfw_entry(10.0, 20.0)))
        self.assertEqual(result[0].signal_type, SignalType.MARITIME)

    def test_lat_lon_binned_to_grid_resolution(self):
        result = _parse_gfw_response(_gfw_response(_gfw_entry(34.6, 56.4)))
        # 34.6 / 1.0 rounded to nearest int = 35; 56.4 → 56
        self.assertAlmostEqual(result[0].lat, round(34.6 / _GRID_RESOLUTION) * _GRID_RESOLUTION)
        self.assertAlmostEqual(result[0].lon, round(56.4 / _GRID_RESOLUTION) * _GRID_RESOLUTION)

    def test_metadata_source_is_maritime(self):
        result = _parse_gfw_response(_gfw_response(_gfw_entry(0.0, 0.0)))
        self.assertEqual(result[0].metadata.get("source"), "maritime")

    def test_metadata_vessel_count_present(self):
        result = _parse_gfw_response(_gfw_response(_gfw_entry(0.0, 0.0)))
        self.assertIn("vessel_count", result[0].metadata)

    def test_missing_position_key_skipped(self):
        entry = {"type": "fishing"}  # no "position"
        result = _parse_gfw_response({"entries": [entry]})
        self.assertEqual(result, [])

    def test_null_position_skipped(self):
        result = _parse_gfw_response({"entries": [{"position": None}]})
        self.assertEqual(result, [])

    def test_null_lat_skipped(self):
        result = _parse_gfw_response({"entries": [{"position": {"lat": None, "lon": 10.0}}]})
        self.assertEqual(result, [])

    def test_null_lon_skipped(self):
        result = _parse_gfw_response({"entries": [{"position": {"lat": 10.0, "lon": None}}]})
        self.assertEqual(result, [])

    def test_lat_out_of_range_skipped(self):
        result = _parse_gfw_response(_gfw_response(_gfw_entry(999.0, 0.0)))
        self.assertEqual(result, [])

    def test_lon_out_of_range_skipped(self):
        result = _parse_gfw_response(_gfw_response(_gfw_entry(0.0, 999.0)))
        self.assertEqual(result, [])

    def test_empty_entries_returns_empty(self):
        result = _parse_gfw_response({"entries": []})
        self.assertEqual(result, [])

    def test_missing_entries_key_returns_empty(self):
        result = _parse_gfw_response({})
        self.assertEqual(result, [])

    def test_multiple_entries_same_cell_produce_one_signal(self):
        # 34.1 and 34.4 both round to 34; 56.1 and 56.4 both round to 56
        data = _gfw_response(_gfw_entry(34.1, 56.1), _gfw_entry(34.4, 56.4))
        result = _parse_gfw_response(data)
        self.assertEqual(len(result), 1)

    def test_same_cell_vessel_count_is_summed(self):
        data = _gfw_response(
            _gfw_entry(34.1, 56.1),
            _gfw_entry(34.4, 56.4),
            _gfw_entry(34.2, 56.3),
        )
        result = _parse_gfw_response(data)
        self.assertEqual(result[0].metadata["vessel_count"], 3)

    def test_different_cells_produce_separate_signals(self):
        # 10.0 → cell 10; 50.0 → cell 50
        data = _gfw_response(_gfw_entry(10.0, 10.0), _gfw_entry(50.0, 50.0))
        result = _parse_gfw_response(data)
        self.assertEqual(len(result), 2)

    def test_intensity_derived_from_count(self):
        # _MAX_VESSELS_PER_CELL entries in one cell → intensity 1.0
        entries = [_gfw_entry(0.0, 0.0) for _ in range(_MAX_VESSELS_PER_CELL)]
        result = _parse_gfw_response({"entries": entries})
        self.assertAlmostEqual(result[0].intensity, 1.0)

    def test_intensity_proportional_to_count(self):
        # 1 entry in one cell, 10 entries in another; second should have higher intensity
        data = {
            "entries": (
                [_gfw_entry(10.0, 10.0)]
                + [_gfw_entry(50.0, 50.0)] * 10
            )
        }
        result = _parse_gfw_response(data)
        by_lat = {round(s.lat): s for s in result}
        self.assertGreater(by_lat[50].intensity, by_lat[10].intensity)

    def test_valid_mixed_with_invalid_only_valid_returned(self):
        data = {
            "entries": [
                _gfw_entry(10.0, 20.0),    # valid
                {"position": None},         # invalid
                _gfw_entry(30.0, 40.0),    # valid
                {"position": {"lat": 999.0, "lon": 0.0}},  # out of range
            ]
        }
        result = _parse_gfw_response(data)
        self.assertEqual(len(result), 2)

    def test_signal_is_external_signal_instance(self):
        result = _parse_gfw_response(_gfw_response(_gfw_entry(10.0, 20.0)))
        self.assertIsInstance(result[0], ExternalSignal)

    def test_intensity_clamped_to_unit_range(self):
        # 100 entries in one cell — far above max → clamped to 1.0
        entries = [_gfw_entry(0.0, 0.0) for _ in range(100)]
        result = _parse_gfw_response({"entries": entries})
        self.assertLessEqual(result[0].intensity, 1.0)
        self.assertGreaterEqual(result[0].intensity, 0.0)


# ---------------------------------------------------------------------------
# _load_fallback_signals() tests
# ---------------------------------------------------------------------------

class LoadFallbackSignalsTests(unittest.TestCase):

    def test_returns_list(self):
        result = _load_fallback_signals()
        self.assertIsInstance(result, list)

    def test_returns_nonempty_list(self):
        result = _load_fallback_signals()
        self.assertGreater(len(result), 0)

    def test_all_signals_are_maritime_type(self):
        for sig in _load_fallback_signals():
            self.assertEqual(sig.signal_type, SignalType.MARITIME)

    def test_all_intensities_in_unit_range(self):
        for sig in _load_fallback_signals():
            self.assertGreaterEqual(sig.intensity, 0.0)
            self.assertLessEqual(sig.intensity, 1.0)

    def test_metadata_source_is_maritime(self):
        for sig in _load_fallback_signals():
            self.assertEqual(sig.metadata.get("source"), "maritime")

    def test_metadata_vessel_count_present_and_positive(self):
        for sig in _load_fallback_signals():
            count = sig.metadata.get("vessel_count")
            self.assertIsNotNone(count)
            self.assertGreater(count, 0)

    def test_all_lat_lon_in_valid_range(self):
        for sig in _load_fallback_signals():
            self.assertGreaterEqual(sig.lat, -90.0)
            self.assertLessEqual(sig.lat,  90.0)
            self.assertGreaterEqual(sig.lon, -180.0)
            self.assertLessEqual(sig.lon,  180.0)

    def test_all_items_are_external_signal_instances(self):
        for sig in _load_fallback_signals():
            self.assertIsInstance(sig, ExternalSignal)

    def test_higher_vessel_count_gives_higher_intensity(self):
        signals = _load_fallback_signals()
        # Sort by vessel count; verify intensity ordering follows
        sorted_sigs = sorted(signals, key=lambda s: s.metadata["vessel_count"])
        intensities = [s.intensity for s in sorted_sigs]
        self.assertEqual(intensities, sorted(intensities))

    def test_missing_file_returns_empty(self):
        with patch("intelligence.maritime_ingestion._FALLBACK_JSON",
                   Path("/nonexistent/path/maritime_fallback.json")):
            result = _load_fallback_signals()
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# fetch_maritime_signals() tests
# ---------------------------------------------------------------------------

class FetchMaritimeSignalsTests(unittest.TestCase):

    def test_without_api_key_returns_nonempty_list(self):
        """No GFW_API_KEY configured → pipeline falls through to fallback → non-empty."""
        import os
        saved = os.environ.pop("GFW_API_KEY", None)
        try:
            result = fetch_maritime_signals()
            self.assertGreater(len(result), 0)
        finally:
            if saved is not None:
                os.environ["GFW_API_KEY"] = saved

    def test_returns_list_of_external_signals(self):
        for sig in fetch_maritime_signals():
            self.assertIsInstance(sig, ExternalSignal)

    def test_all_signals_are_maritime_type(self):
        for sig in fetch_maritime_signals():
            self.assertEqual(sig.signal_type, SignalType.MARITIME)

    def test_all_intensities_in_unit_range(self):
        for sig in fetch_maritime_signals():
            self.assertGreaterEqual(sig.intensity, 0.0)
            self.assertLessEqual(sig.intensity, 1.0)

    def test_all_metadata_source_is_maritime(self):
        for sig in fetch_maritime_signals():
            self.assertEqual(sig.metadata.get("source"), "maritime")

    def test_all_metadata_has_vessel_count(self):
        for sig in fetch_maritime_signals():
            self.assertIn("vessel_count", sig.metadata)


# ---------------------------------------------------------------------------
# Integration with build_features()
# ---------------------------------------------------------------------------

class BuildFeaturesIntegrationTests(unittest.TestCase):
    """MARITIME signals influence build_features() output for nearby regions."""

    # Strait of Hormuz: NOTAM A0002/24 at (26.8, 56.2), maritime at (26.5, 56.2)
    _LAT = 26.8
    _LON = 56.2

    def _maritime_signal(self, lat=None, lon=None, intensity=1.0) -> ExternalSignal:
        return ExternalSignal(
            signal_type = SignalType.MARITIME,
            lat         = lat if lat is not None else self._LAT,
            lon         = lon if lon is not None else self._LON,
            intensity   = intensity,
            metadata    = {"source": "maritime", "vessel_count": 20},
        )

    def _no_fly_signal(self, lat=None, lon=None, intensity=1.0) -> ExternalSignal:
        return ExternalSignal(
            signal_type = SignalType.NO_FLY,
            lat         = lat if lat is not None else self._LAT,
            lon         = lon if lon is not None else self._LON,
            intensity   = intensity,
            metadata    = {},
        )

    def _spike_row(self, lat=None, lon=None):
        return [(lat or self._LAT, lon or self._LON, 8, 3)]

    def _find_region(self, features):
        for f in features:
            if abs(f.lat - self._LAT) < 0.15 and abs(f.lon - self._LON) < 0.15:
                return f
        return None

    def test_maritime_signal_sets_coordination_flag(self):
        features = build_features(
            spikes           = self._spike_row(),
            external_signals = [self._maritime_signal()],
        )
        region = self._find_region(features)
        self.assertIsNotNone(region)
        self.assertTrue(region.coordination_flag,
                        "MARITIME signal should set coordination_flag=True for nearby region")

    def test_maritime_signal_raises_inflow_count(self):
        features = build_features(
            spikes           = self._spike_row(),
            external_signals = [self._maritime_signal()],
        )
        region = self._find_region(features)
        self.assertIsNotNone(region)
        self.assertGreater(region.inflow_count, 0,
                           "MARITIME signal should raise inflow_count above 0")

    def test_maritime_signal_strength_populated(self):
        features = build_features(
            spikes           = self._spike_row(),
            external_signals = [self._maritime_signal(intensity=0.8)],
        )
        region = self._find_region(features)
        self.assertIsNotNone(region)
        self.assertGreater(region.maritime_signal_strength, 0.0,
                           "maritime_signal_strength should be > 0 when MARITIME present")

    def test_region_far_from_signal_is_unaffected(self):
        far_lat, far_lon = 0.0, 0.0
        baseline   = build_features(spikes=[(far_lat, far_lon, 5, 2)])
        with_signal = build_features(
            spikes           = [(far_lat, far_lon, 5, 2)],
            external_signals = [self._maritime_signal()],
        )
        def _region(features):
            return next(
                f for f in features
                if abs(f.lat - far_lat) < 0.15 and abs(f.lon - far_lon) < 0.15
            )
        r_base = _region(baseline)
        r_sig  = _region(with_signal)
        self.assertEqual(r_base.coordination_flag, r_sig.coordination_flag)
        self.assertEqual(r_base.inflow_count,      r_sig.inflow_count)

    def test_maritime_signal_type_recorded_in_region(self):
        """ExternalSignal proximity tracking should record MARITIME type."""
        features = build_features(
            spikes           = self._spike_row(),
            external_signals = [self._maritime_signal()],
        )
        region = self._find_region(features)
        self.assertIsNotNone(region)
        self.assertIn(SignalType.MARITIME, region.external_signal_types)

    def test_combined_notam_maritime_triggers_cross_domain(self):
        """
        NO_FLY + MARITIME near the same region should produce
        external_signal_types containing both SignalType.NO_FLY and
        SignalType.MARITIME, which drives the cross-domain bonus in
        classify_regions.
        """
        features = build_features(
            spikes           = self._spike_row(),
            external_signals = [
                self._no_fly_signal(),
                self._maritime_signal(),
            ],
        )
        region = self._find_region(features)
        self.assertIsNotNone(region)
        self.assertIn(SignalType.NO_FLY,   region.external_signal_types,
                      "NO_FLY must be recorded in external_signal_types")
        self.assertIn(SignalType.MARITIME, region.external_signal_types,
                      "MARITIME must be recorded in external_signal_types")

    def test_combined_signals_raise_coordination_score(self):
        """
        NO_FLY + MARITIME together should produce a higher
        COORDINATED_ACTIVITY score than NO_FLY alone.
        """
        from intelligence.classifier import classify_regions

        no_fly_only = build_features(
            spikes           = self._spike_row(),
            external_signals = [self._no_fly_signal()],
        )
        combined = build_features(
            spikes           = self._spike_row(),
            external_signals = [self._no_fly_signal(), self._maritime_signal()],
        )

        def _ca_score(features):
            for r in classify_regions(features):
                if abs(r.lat - self._LAT) < 0.15 and abs(r.lon - self._LON) < 0.15:
                    return r.all_scores.get("COORDINATED_ACTIVITY", 0.0)
            return None

        score_nf   = _ca_score(no_fly_only)
        score_comb = _ca_score(combined)
        self.assertIsNotNone(score_nf)
        self.assertIsNotNone(score_comb)
        self.assertGreater(score_comb, score_nf,
                           f"Combined signals should raise CA score: {score_comb:.1f} vs {score_nf:.1f}")

    def test_fallback_maritime_signals_integrate_without_error(self):
        """fetch_maritime_signals() output must not raise when passed to build_features."""
        signals = fetch_maritime_signals()
        try:
            build_features(external_signals=signals)
        except Exception as exc:
            self.fail(f"build_features raised {exc!r} with fallback maritime signals")

    def test_fallback_maritime_signals_affect_at_least_one_region(self):
        """At least one region should show external_signal_count > 0 when maritime data present."""
        signals = fetch_maritime_signals()
        features = build_features(
            spikes           = self._spike_row(lat=26.5, lon=56.2),
            external_signals = signals,
        )
        affected = [f for f in features if f.external_signal_count > 0]
        self.assertGreater(len(affected), 0,
                           "At least one region should be affected by fallback maritime signals")


if __name__ == "__main__":
    unittest.main()
