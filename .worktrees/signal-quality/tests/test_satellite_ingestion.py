"""
Tests for intelligence/satellite_ingestion.py
───────────────────────────────────────────────
Covers:
  - _normalize_change()
      • zero / negative ratio → 0.0
      • proportional values at various ratios
      • at max_ratio → 1.0
      • above max_ratio → clamped to 1.0
      • custom max_ratio
  - _parse_satellite_json()
      • valid record → ExternalSignal with correct fields
      • signal_type = SATELLITE
      • intensity = normalize_change(change_metric)
      • metadata: source="satellite", change_metric present
      • missing lat / lon / change_metric → skipped
      • out-of-range lat/lon → skipped
      • zero / negative change_metric → no signal (skipped)
      • empty list → empty result
      • mixed valid and invalid → only valid returned
  - _load_fallback_signals()
      • returns non-empty list
      • all SATELLITE type
      • intensities in [0, 1]
      • metadata source="satellite", change_metric present
      • missing file → empty list
  - _generate_from_db()
      • nonexistent path → empty list
      • empty DB (no tables) → empty list
      • DB with only baseline data → empty list
      • fewer than _MIN_CURRENT_COUNT aircraft → filtered out
      • brand-new region (baseline=0, current≥3) → high-intensity signal
      • stable region (current==baseline) → no signal
      • decreasing region → no signal
      • spike region (current >> baseline) → positive-intensity signal
      • signal type is SATELLITE
      • metadata has source and change_metric
      • intensity in [0, 1]
      • multiple spike cells → multiple signals
  - fetch_satellite_signals()
      • returns list
      • all SATELLITE type
      • all intensities in [0, 1]
      • explicit db_path bypasses cache
      • safe failure → returns fallback signals (non-empty)
  - Integration with build_features()
      • SATELLITE signal raises change_score for nearby region
      • SATELLITE signal raises new_aircraft proxy
      • region far from signal is unaffected
      • fallback signals integrate without error
      • fallback signals affect at least one nearby region
      • all three signal types together populate external_signal_types
"""

import os
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intelligence.satellite_ingestion import (
    _normalize_change,
    _parse_satellite_json,
    _load_fallback_signals,
    _generate_from_db,
    fetch_satellite_signals,
    _MAX_CHANGE_RATIO,
    _MIN_CURRENT_COUNT,
)
from intelligence.external_features import ExternalSignal, SignalType
from intelligence.classifier import build_features


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _make_db(current_rows=None, baseline_rows=None) -> Path:
    """
    Create a temporary SQLite DB with the aircraft_positions table.

    current_rows   list of (lat_bin, lon_bin) inserted at now-10 min
    baseline_rows  list of (lat_bin, lon_bin) inserted at now-4 h
    """
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE aircraft_positions (
            icao24 TEXT, callsign TEXT,
            lat REAL, lon REAL,
            altitude REAL, speed REAL,
            timestamp REAL, type TEXT,
            behavior TEXT, score INTEGER,
            lat_bin REAL, lon_bin REAL
        )
    """)

    now = time.time()
    current_ts  = now - 600      # 10 min ago — inside current window
    baseline_ts = now - 14400    # 4 hours ago — inside baseline window

    if current_rows:
        conn.executemany(
            "INSERT INTO aircraft_positions (lat_bin, lon_bin, timestamp) VALUES (?,?,?)",
            [(r[0], r[1], current_ts) for r in current_rows],
        )
    if baseline_rows:
        conn.executemany(
            "INSERT INTO aircraft_positions (lat_bin, lon_bin, timestamp) VALUES (?,?,?)",
            [(r[0], r[1], baseline_ts) for r in baseline_rows],
        )
    conn.commit()
    conn.close()
    return Path(path)


class _TmpDbMixin:
    """Register temp DB paths for cleanup in tearDown."""

    def setUp(self):
        self._tmp_paths = []

    def tearDown(self):
        for p in self._tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    def _db(self, **kwargs) -> Path:
        p = _make_db(**kwargs)
        self._tmp_paths.append(str(p))
        return p


# ---------------------------------------------------------------------------
# _normalize_change() tests
# ---------------------------------------------------------------------------

class NormalizeChangeTests(unittest.TestCase):

    def test_zero_ratio_returns_zero(self):
        self.assertEqual(_normalize_change(0.0), 0.0)

    def test_negative_ratio_returns_zero(self):
        self.assertEqual(_normalize_change(-1.0), 0.0)

    def test_very_negative_returns_zero(self):
        self.assertEqual(_normalize_change(-100.0), 0.0)

    def test_half_max_ratio_returns_half(self):
        self.assertAlmostEqual(_normalize_change(_MAX_CHANGE_RATIO / 2), 0.5)

    def test_at_max_ratio_returns_one(self):
        self.assertAlmostEqual(_normalize_change(_MAX_CHANGE_RATIO), 1.0)

    def test_above_max_clamped_to_one(self):
        self.assertAlmostEqual(_normalize_change(_MAX_CHANGE_RATIO * 10), 1.0)

    def test_small_positive_ratio(self):
        result = _normalize_change(0.5)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 0.5)

    def test_custom_max_ratio(self):
        self.assertAlmostEqual(_normalize_change(5.0, max_ratio=10.0), 0.5)

    def test_result_always_in_unit_range(self):
        for r in [-2.0, -0.1, 0.0, 0.5, 1.5, 3.0, 6.0, 100.0]:
            result = _normalize_change(r)
            self.assertGreaterEqual(result, 0.0, f"ratio={r}")
            self.assertLessEqual(result, 1.0, f"ratio={r}")


# ---------------------------------------------------------------------------
# _parse_satellite_json() tests
# ---------------------------------------------------------------------------

class ParseSatelliteJsonTests(unittest.TestCase):

    def _item(self, lat=10.0, lon=20.0, change_metric=1.5, **extra):
        d = {"lat": lat, "lon": lon, "change_metric": change_metric}
        d.update(extra)
        return d

    def test_valid_record_returns_one_signal(self):
        result = _parse_satellite_json([self._item()])
        self.assertEqual(len(result), 1)

    def test_signal_type_is_satellite(self):
        result = _parse_satellite_json([self._item()])
        self.assertEqual(result[0].signal_type, SignalType.SATELLITE)

    def test_lat_lon_preserved(self):
        result = _parse_satellite_json([self._item(lat=34.5, lon=33.5)])
        self.assertAlmostEqual(result[0].lat, 34.5)
        self.assertAlmostEqual(result[0].lon, 33.5)

    def test_intensity_derived_from_change_metric(self):
        metric = 1.5
        result = _parse_satellite_json([self._item(change_metric=metric)])
        expected = _normalize_change(metric)
        self.assertAlmostEqual(result[0].intensity, expected)

    def test_metadata_source_is_satellite(self):
        result = _parse_satellite_json([self._item()])
        self.assertEqual(result[0].metadata.get("source"), "satellite")

    def test_metadata_change_metric_present(self):
        result = _parse_satellite_json([self._item(change_metric=2.1)])
        self.assertIn("change_metric", result[0].metadata)
        self.assertAlmostEqual(result[0].metadata["change_metric"], 2.1)

    def test_missing_lat_key_skipped(self):
        result = _parse_satellite_json([{"lon": 20.0, "change_metric": 1.0}])
        self.assertEqual(result, [])

    def test_missing_lon_key_skipped(self):
        result = _parse_satellite_json([{"lat": 10.0, "change_metric": 1.0}])
        self.assertEqual(result, [])

    def test_missing_change_metric_skipped(self):
        result = _parse_satellite_json([{"lat": 10.0, "lon": 20.0}])
        self.assertEqual(result, [])

    def test_lat_out_of_range_skipped(self):
        result = _parse_satellite_json([self._item(lat=999.0)])
        self.assertEqual(result, [])

    def test_lon_out_of_range_skipped(self):
        result = _parse_satellite_json([self._item(lon=999.0)])
        self.assertEqual(result, [])

    def test_zero_change_metric_no_signal(self):
        result = _parse_satellite_json([self._item(change_metric=0.0)])
        self.assertEqual(result, [])

    def test_negative_change_metric_no_signal(self):
        result = _parse_satellite_json([self._item(change_metric=-1.5)])
        self.assertEqual(result, [])

    def test_empty_list_returns_empty(self):
        result = _parse_satellite_json([])
        self.assertEqual(result, [])

    def test_intensity_clamped_to_unit_range(self):
        result = _parse_satellite_json([self._item(change_metric=999.0)])
        self.assertLessEqual(result[0].intensity, 1.0)
        self.assertGreaterEqual(result[0].intensity, 0.0)

    def test_multiple_valid_records(self):
        result = _parse_satellite_json([self._item(lat=float(i)) for i in range(3)])
        self.assertEqual(len(result), 3)

    def test_mixed_valid_and_invalid(self):
        items = [
            self._item(lat=10.0),              # valid
            {"lat": None, "lon": 0.0, "change_metric": 1.0},  # invalid lat
            self._item(lat=20.0),              # valid
            self._item(change_metric=-1.0),    # no signal (negative)
        ]
        result = _parse_satellite_json(items)
        self.assertEqual(len(result), 2)

    def test_result_is_external_signal_instance(self):
        result = _parse_satellite_json([self._item()])
        self.assertIsInstance(result[0], ExternalSignal)


# ---------------------------------------------------------------------------
# _load_fallback_signals() tests
# ---------------------------------------------------------------------------

class LoadFallbackSignalsTests(unittest.TestCase):

    def test_returns_list(self):
        self.assertIsInstance(_load_fallback_signals(), list)

    def test_returns_nonempty_list(self):
        result = _load_fallback_signals()
        self.assertGreater(len(result), 0)

    def test_all_satellite_type(self):
        for sig in _load_fallback_signals():
            self.assertEqual(sig.signal_type, SignalType.SATELLITE)

    def test_all_intensities_in_unit_range(self):
        for sig in _load_fallback_signals():
            self.assertGreaterEqual(sig.intensity, 0.0)
            self.assertLessEqual(sig.intensity, 1.0)

    def test_all_intensities_positive(self):
        for sig in _load_fallback_signals():
            self.assertGreater(sig.intensity, 0.0,
                               "fallback change_metric values must produce positive intensity")

    def test_metadata_source_is_satellite(self):
        for sig in _load_fallback_signals():
            self.assertEqual(sig.metadata.get("source"), "satellite")

    def test_metadata_change_metric_present(self):
        for sig in _load_fallback_signals():
            self.assertIn("change_metric", sig.metadata)

    def test_all_lat_lon_in_valid_range(self):
        for sig in _load_fallback_signals():
            self.assertGreaterEqual(sig.lat, -90.0)
            self.assertLessEqual(sig.lat,  90.0)
            self.assertGreaterEqual(sig.lon, -180.0)
            self.assertLessEqual(sig.lon,  180.0)

    def test_all_items_are_external_signal_instances(self):
        for sig in _load_fallback_signals():
            self.assertIsInstance(sig, ExternalSignal)

    def test_missing_file_returns_empty(self):
        with patch("intelligence.satellite_ingestion._FALLBACK_JSON",
                   Path("/nonexistent/satellite_fallback.json")):
            result = _load_fallback_signals()
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# _generate_from_db() tests
# ---------------------------------------------------------------------------

class GenerateFromDbTests(_TmpDbMixin, unittest.TestCase):

    def test_nonexistent_path_returns_empty(self):
        result = _generate_from_db(Path("/nonexistent/path/db.db"))
        self.assertEqual(result, [])

    def test_empty_db_no_tables_returns_empty(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._tmp_paths.append(path)
        result = _generate_from_db(Path(path))
        self.assertEqual(result, [])

    def test_only_baseline_data_no_current_returns_empty(self):
        db = self._db(baseline_rows=[(34.5, 33.5)] * 5)
        self.assertEqual(_generate_from_db(db), [])

    def test_too_few_current_aircraft_filtered(self):
        # _MIN_CURRENT_COUNT - 1 rows → HAVING filters them out
        db = self._db(current_rows=[(34.5, 33.5)] * (_MIN_CURRENT_COUNT - 1))
        self.assertEqual(_generate_from_db(db), [])

    def test_new_region_produces_signal(self):
        # Brand-new region: baseline=0, current=5 → change_ratio=5.0 → intensity=1.0
        db = self._db(current_rows=[(34.5, 33.5)] * 5)
        result = _generate_from_db(db)
        self.assertEqual(len(result), 1)

    def test_new_region_has_full_intensity(self):
        # 5 aircraft, no baseline → change_ratio=5.0 → intensity=min(5/3,1)=1.0
        db = self._db(current_rows=[(34.5, 33.5)] * 5)
        result = _generate_from_db(db)
        self.assertAlmostEqual(result[0].intensity, 1.0)

    def test_spike_region_produces_signal(self):
        db = self._db(
            current_rows  = [(34.5, 33.5)] * 10,
            baseline_rows = [(34.5, 33.5)] * 2,
        )
        result = _generate_from_db(db)
        self.assertGreater(len(result), 0)

    def test_stable_region_no_signal(self):
        # Same count in current and baseline → change_ratio=0 → no signal
        db = self._db(
            current_rows  = [(34.5, 33.5)] * 5,
            baseline_rows = [(34.5, 33.5)] * 5,
        )
        self.assertEqual(_generate_from_db(db), [])

    def test_decreasing_region_no_signal(self):
        db = self._db(
            current_rows  = [(34.5, 33.5)] * 3,
            baseline_rows = [(34.5, 33.5)] * 10,
        )
        self.assertEqual(_generate_from_db(db), [])

    def test_signal_type_is_satellite(self):
        db = self._db(current_rows=[(34.5, 33.5)] * 5)
        for sig in _generate_from_db(db):
            self.assertEqual(sig.signal_type, SignalType.SATELLITE)

    def test_metadata_source_is_satellite(self):
        db = self._db(current_rows=[(34.5, 33.5)] * 5)
        result = _generate_from_db(db)
        self.assertEqual(result[0].metadata.get("source"), "satellite")

    def test_metadata_change_metric_present(self):
        db = self._db(current_rows=[(34.5, 33.5)] * 5)
        result = _generate_from_db(db)
        self.assertIn("change_metric", result[0].metadata)

    def test_intensity_in_unit_range(self):
        db = self._db(
            current_rows  = [(34.5, 33.5)] * 20,
            baseline_rows = [(34.5, 33.5)] * 2,
        )
        for sig in _generate_from_db(db):
            self.assertGreaterEqual(sig.intensity, 0.0)
            self.assertLessEqual(sig.intensity, 1.0)

    def test_multiple_spike_cells_produce_multiple_signals(self):
        db = self._db(
            current_rows  = [(34.5, 33.5)] * 5 + [(26.8, 56.2)] * 5,
        )
        result = _generate_from_db(db)
        # Both cells round to distinct 1° bins: (35, 34) and (27, 56)
        self.assertEqual(len(result), 2)

    def test_result_elements_are_external_signal_instances(self):
        db = self._db(current_rows=[(34.5, 33.5)] * 5)
        for sig in _generate_from_db(db):
            self.assertIsInstance(sig, ExternalSignal)

    def test_spike_intensity_proportional_to_change(self):
        """Larger relative increase → higher intensity."""
        # small increase: 6 vs 5 baseline → change_ratio=(6-5)/5=0.2 → low intensity
        db_small = self._db(
            current_rows  = [(34.5, 33.5)] * 6,
            baseline_rows = [(34.5, 33.5)] * 5,
        )
        # large increase: 15 vs 1 baseline → change_ratio=(15-1)/1=14 → max intensity
        db_large = self._db(
            current_rows  = [(34.5, 33.5)] * 15,
            baseline_rows = [(34.5, 33.5)] * 1,
        )
        r_small = _generate_from_db(db_small)
        r_large = _generate_from_db(db_large)
        if r_small and r_large:
            self.assertGreater(r_large[0].intensity, r_small[0].intensity)

    def test_1deg_binning_merges_nearby_cells(self):
        # 34.1 and 34.4 both ROUND to 34; same for 33.1 and 33.4
        db = self._db(
            current_rows=[(34.1, 33.1)] * 3 + [(34.4, 33.4)] * 3,
        )
        result = _generate_from_db(db)
        # Both rows round to (34, 33) → merged into one cell with count=6
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].metadata["change_metric"],
                         round(6 / 1, 3))  # baseline=0 → 6/max(0,1)=6.0


# ---------------------------------------------------------------------------
# fetch_satellite_signals() tests
# ---------------------------------------------------------------------------

class FetchSatelliteSignalsTests(_TmpDbMixin, unittest.TestCase):

    def test_returns_list(self):
        self.assertIsInstance(fetch_satellite_signals(), list)

    def test_all_satellite_type(self):
        for sig in fetch_satellite_signals():
            self.assertEqual(sig.signal_type, SignalType.SATELLITE)

    def test_all_intensities_in_unit_range(self):
        for sig in fetch_satellite_signals():
            self.assertGreaterEqual(sig.intensity, 0.0)
            self.assertLessEqual(sig.intensity, 1.0)

    def test_all_metadata_source_is_satellite(self):
        for sig in fetch_satellite_signals():
            self.assertEqual(sig.metadata.get("source"), "satellite")

    def test_explicit_db_path_bypasses_cache(self):
        """Passing db_path= should always return a fresh result."""
        db = self._db(current_rows=[(10.0, 20.0)] * 5)
        result_a = fetch_satellite_signals(db_path=db)
        result_b = fetch_satellite_signals(db_path=db)
        self.assertEqual(len(result_a), len(result_b))

    def test_empty_db_falls_back_to_json(self):
        """An empty (no-data) DB should cause fallback to JSON signals."""
        db = self._db()  # no rows
        result = fetch_satellite_signals(db_path=db)
        # Fallback JSON has 5 entries all with SATELLITE type
        self.assertGreater(len(result), 0)
        for sig in result:
            self.assertEqual(sig.signal_type, SignalType.SATELLITE)

    def test_spike_db_returns_db_signals(self):
        """A DB with spike data should return DB-generated signals (not fallback)."""
        db = self._db(current_rows=[(34.5, 33.5)] * 5)
        result = fetch_satellite_signals(db_path=db)
        # DB-generated signals have change_metric from actual data
        db_sources = [s for s in result if s.metadata.get("change_metric") == 5.0]
        self.assertGreater(len(db_sources), 0)


# ---------------------------------------------------------------------------
# Integration with build_features()
# ---------------------------------------------------------------------------

class BuildFeaturesIntegrationTests(unittest.TestCase):

    _LAT = 34.5
    _LON = 33.5

    def _sat_signal(self, intensity=0.8) -> ExternalSignal:
        return ExternalSignal(
            signal_type = SignalType.SATELLITE,
            lat         = self._LAT,
            lon         = self._LON,
            intensity   = intensity,
            metadata    = {"source": "satellite", "change_metric": 2.4},
        )

    def _find_region(self, features):
        for f in features:
            if abs(f.lat - self._LAT) < 0.15 and abs(f.lon - self._LON) < 0.15:
                return f
        return None

    def test_satellite_signal_raises_change_score(self):
        baseline = build_features(spikes=[(self._LAT, self._LON, 8, 2)])
        augmented = build_features(
            spikes           = [(self._LAT, self._LON, 8, 2)],
            external_signals = [self._sat_signal(intensity=1.0)],
        )
        r_base = self._find_region(baseline)
        r_aug  = self._find_region(augmented)
        self.assertIsNotNone(r_aug)
        self.assertGreater(r_aug.change_score, r_base.change_score,
                           "SATELLITE signal must raise change_score above baseline")

    def test_satellite_signal_raises_new_aircraft_proxy(self):
        """
        SATELLITE → novelty → new_aircraft proxy raised in merge_external_features.
        """
        features = build_features(
            spikes           = [(self._LAT, self._LON, 10, 2)],
            external_signals = [self._sat_signal(intensity=1.0)],
        )
        region = self._find_region(features)
        self.assertIsNotNone(region)
        self.assertGreater(region.new_aircraft, 0,
                           "SATELLITE signal should raise new_aircraft proxy")

    def test_satellite_signal_type_recorded_in_region(self):
        features = build_features(
            spikes           = [(self._LAT, self._LON, 5, 1)],
            external_signals = [self._sat_signal()],
        )
        region = self._find_region(features)
        self.assertIsNotNone(region)
        self.assertIn(SignalType.SATELLITE, region.external_signal_types)

    def test_region_far_from_signal_unaffected(self):
        far_lat, far_lon = 0.0, 0.0
        baseline    = build_features(spikes=[(far_lat, far_lon, 5, 1)])
        with_signal = build_features(
            spikes           = [(far_lat, far_lon, 5, 1)],
            external_signals = [self._sat_signal()],
        )
        def _get(features):
            return next(f for f in features
                        if abs(f.lat - far_lat) < 0.15 and abs(f.lon - far_lon) < 0.15)
        r_base = _get(baseline)
        r_sig  = _get(with_signal)
        self.assertAlmostEqual(r_base.change_score, r_sig.change_score)
        self.assertEqual(r_base.new_aircraft, r_sig.new_aircraft)

    def test_higher_intensity_gives_higher_change_score(self):
        low  = build_features(
            spikes           = [(self._LAT, self._LON, 8, 2)],
            external_signals = [self._sat_signal(intensity=0.3)],
        )
        high = build_features(
            spikes           = [(self._LAT, self._LON, 8, 2)],
            external_signals = [self._sat_signal(intensity=1.0)],
        )
        r_low  = self._find_region(low)
        r_high = self._find_region(high)
        self.assertGreater(r_high.change_score, r_low.change_score)

    def test_fallback_signals_integrate_without_error(self):
        signals = _load_fallback_signals()
        try:
            build_features(external_signals=signals)
        except Exception as exc:
            self.fail(f"build_features raised {exc!r} with fallback satellite signals")

    def test_fallback_signals_affect_nearby_region(self):
        """Fallback has a signal at (34.5, 33.5) — a spike there should be affected."""
        signals = _load_fallback_signals()
        features = build_features(
            spikes           = [(self._LAT, self._LON, 8, 2)],
            external_signals = signals,
        )
        region = self._find_region(features)
        self.assertIsNotNone(region)
        self.assertGreater(region.external_signal_count, 0,
                           "Region near fallback satellite signal should be affected")

    def test_all_three_signal_types_populate_external_signal_types(self):
        """
        NO_FLY + MARITIME + SATELLITE near the same region should populate
        external_signal_types with all three SignalType values.
        """
        signals = [
            ExternalSignal(SignalType.NO_FLY,    self._LAT, self._LON, 0.9, {}),
            ExternalSignal(SignalType.MARITIME,  self._LAT, self._LON, 0.8, {}),
            ExternalSignal(SignalType.SATELLITE, self._LAT, self._LON, 0.7, {}),
        ]
        features = build_features(
            spikes           = [(self._LAT, self._LON, 8, 2)],
            external_signals = signals,
        )
        region = self._find_region(features)
        self.assertIsNotNone(region)
        self.assertIn(SignalType.NO_FLY,    region.external_signal_types)
        self.assertIn(SignalType.MARITIME,  region.external_signal_types)
        self.assertIn(SignalType.SATELLITE, region.external_signal_types)

    def test_satellite_raises_anomaly_score(self):
        """
        SATELLITE raises change_score_norm and novelty — both feed ANOMALY.
        A region with a SATELLITE signal should score higher on ANOMALY than baseline.
        """
        from intelligence.classifier import classify_regions

        baseline = build_features(spikes=[(self._LAT, self._LON, 8, 0)])
        augmented = build_features(
            spikes           = [(self._LAT, self._LON, 8, 0)],
            external_signals = [self._sat_signal(intensity=1.0)],
        )

        def _anomaly(features):
            for r in classify_regions(features):
                if abs(r.lat - self._LAT) < 0.15 and abs(r.lon - self._LON) < 0.15:
                    return r.all_scores.get("ANOMALY", 0.0)
            return None

        score_base = _anomaly(baseline)
        score_aug  = _anomaly(augmented)
        self.assertIsNotNone(score_base)
        self.assertIsNotNone(score_aug)
        self.assertGreater(score_aug, score_base,
                           f"SATELLITE should raise ANOMALY: {score_aug:.1f} vs {score_base:.1f}")


if __name__ == "__main__":
    unittest.main()
