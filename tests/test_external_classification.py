"""
Tests for external-signal influence on classification outcomes.

Covers:
  Goal 1 — min_aircraft bypass
      • 1 aircraft + PROHIBITED (1.0) → region appears in classify_regions output
      • 1 aircraft + RESTRICTED (0.75) → bypasses (boundary, intensity == threshold)
      • 1 aircraft + WARNING   (0.50) → does NOT bypass (below threshold)
      • 1 aircraft + ADVISORY  (0.25) → does NOT bypass (well below threshold)
      • 0 external signals          → does NOT bypass

  Goal 2 — ANOMALY score boost
      • spike_set_by_external=True when external merge set the flag
      • spike_set_by_external=False when detection layer set the flag first
      • spike_set_by_external=False with no external signals
      • ANOMALY score is 1.15x higher when externally-set vs detection-set spike
      • ANOMALY score never exceeds 100 after boost
      • ROUTINE score unaffected (only ANOMALY is boosted)
      • No boost when spike_flag set by detection layer

  Goal 3 — explanation tagging
      • "external constraints detected" appears when signals present
      • Tag absent when no external signals
      • Tag appears for every classification type

  Goal 4 — external_influence field in _log_classify_impact
      • external_influence=True when winner changes
      • external_influence=True when |delta| > _EXT_INFLUENCE_THRESHOLD
      • external_influence=False when delta is small and winner unchanged

  No-regression
      • signals=None → same output as baseline
      • signals=[] → same output as baseline
      • Non-nearby signals have no effect on distant region
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import intelligence.classifier as classifier_mod
from intelligence.classifier import (
    Classification,
    RegionFeatures,
    _EXT_BOOST_MAX,
    _EXT_BOOST_MULTI_TYPE,
    _EXT_BOOST_PER_STRENGTH,
    _EXT_BYPASS_INTENSITY,
    _EXT_INFLUENCE_THRESHOLD,
    _EXT_STRENGTH_CAP,
    _log_classify_impact,
    _score_region,
    build_features,
    classify_regions,
)
from intelligence.external_features import ExternalSignal, SignalType
from intelligence.notam_ingestion import NotamSeverity, Notam, fetch_notam_signals


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_LAT, _LON = 37.2, -76.3   # Hampton Roads area; not near any mock NOTAM


def _no_fly(lat=_LAT, lon=_LON, intensity=1.0) -> ExternalSignal:
    return ExternalSignal(
        signal_type=SignalType.NO_FLY, lat=lat, lon=lon,
        intensity=intensity, metadata={},
    )


def _features_with_signal(intensity=1.0, aircraft=5, military=0,
                           use_new_entries=True):
    """
    Build a region at (_LAT, _LON) with the given aircraft count.
    By default uses new_entries (so spike_flag=False before external merge).
    Pass use_new_entries=False to use spikes input (spike_flag=True from detection).
    """
    if use_new_entries:
        return build_features(
            new_entries=[(  _LAT, _LON, aircraft, military)],
            external_signals=[_no_fly(intensity=intensity)],
        )
    else:
        return build_features(
            spikes=[(  _LAT, _LON, aircraft, military)],
            external_signals=[_no_fly(intensity=intensity)],
        )


def _find_region(features, lat=_LAT, lon=_LON):
    """Return the RegionFeatures nearest to (lat, lon), or None."""
    for f in features:
        if abs(f.lat - lat) < 0.15 and abs(f.lon - lon) < 0.15:
            return f
    return None


def _find_intel(intel, lat=_LAT, lon=_LON):
    """Return the RegionIntelligence nearest to (lat, lon), or None."""
    for r in intel:
        if abs(r.lat - lat) < 0.15 and abs(r.lon - lon) < 0.15:
            return r
    return None


def _anomaly_score(features, lat=_LAT, lon=_LON):
    """Classify features and return the ANOMALY score for the target region."""
    intel = classify_regions(features)
    r = _find_intel(intel, lat, lon)
    return r.all_scores.get("ANOMALY", 0.0) if r else None


# ---------------------------------------------------------------------------
# Goal 1 — min_aircraft bypass
# ---------------------------------------------------------------------------

class MinAircraftBypassTests(unittest.TestCase):

    def test_prohibited_intensity_bypasses_min_aircraft(self):
        """1 aircraft + PROHIBITED (1.0) → region must appear after bypass."""
        features = build_features(
            coordinated=[(  _LAT, _LON, 1, 0)],   # 1 aircraft only
            external_signals=[_no_fly(intensity=1.0)],
        )
        intel = classify_regions(features, min_aircraft=2)
        r = _find_intel(intel)
        self.assertIsNotNone(r, "Region with 1 aircraft + PROHIBITED should appear")

    def test_restricted_intensity_bypasses_min_aircraft(self):
        """RESTRICTED (0.75) is exactly on the threshold — must bypass."""
        features = build_features(
            coordinated=[(  _LAT, _LON, 1, 0)],
            external_signals=[_no_fly(intensity=_EXT_BYPASS_INTENSITY)],  # 0.75
        )
        intel = classify_regions(features, min_aircraft=2)
        r = _find_intel(intel)
        self.assertIsNotNone(r, "RESTRICTED intensity == threshold should bypass")

    def test_warning_intensity_does_not_bypass(self):
        """WARNING (0.50) < 0.75 threshold — must NOT bypass min_aircraft."""
        features = build_features(
            coordinated=[(  _LAT, _LON, 1, 0)],
            external_signals=[_no_fly(intensity=0.50)],
        )
        intel = classify_regions(features, min_aircraft=2)
        r = _find_intel(intel)
        self.assertIsNone(r, "WARNING intensity < threshold should NOT bypass")

    def test_advisory_intensity_does_not_bypass(self):
        """ADVISORY (0.25) — well below threshold — must NOT bypass."""
        features = build_features(
            coordinated=[(  _LAT, _LON, 1, 0)],
            external_signals=[_no_fly(intensity=0.25)],
        )
        intel = classify_regions(features, min_aircraft=2)
        r = _find_intel(intel)
        self.assertIsNone(r, "ADVISORY intensity should NOT bypass")

    def test_no_external_signals_no_bypass(self):
        """No external signals → standard min_aircraft filter applies."""
        features = build_features(
            coordinated=[(  _LAT, _LON, 1, 0)],
        )
        intel = classify_regions(features, min_aircraft=2)
        r = _find_intel(intel)
        self.assertIsNone(r, "No signals → filter should block 1-aircraft region")

    def test_bypass_region_receives_valid_classification(self):
        """Bypassed region must have a non-None, valid Classification."""
        features = build_features(
            coordinated=[(  _LAT, _LON, 1, 0)],
            external_signals=[_no_fly(intensity=1.0)],
        )
        intel = classify_regions(features, min_aircraft=2)
        r = _find_intel(intel)
        self.assertIsNotNone(r)
        self.assertIsInstance(r.classification, Classification)

    def test_bypass_region_has_confidence_in_valid_range(self):
        features = build_features(
            coordinated=[(  _LAT, _LON, 1, 0)],
            external_signals=[_no_fly(intensity=1.0)],
        )
        intel = classify_regions(features, min_aircraft=2)
        r = _find_intel(intel)
        self.assertIsNotNone(r)
        self.assertGreaterEqual(r.confidence, 0)
        self.assertLessEqual(r.confidence, 100)

    def test_region_above_threshold_always_included(self):
        """Region with aircraft_count >= min_aircraft is unaffected by bypass logic."""
        features = build_features(
            coordinated=[(  _LAT, _LON, 5, 0)],
        )
        intel = classify_regions(features, min_aircraft=2)
        r = _find_intel(intel)
        self.assertIsNotNone(r, "Normal region above min_aircraft must always appear")

    def test_distant_signal_does_not_bypass_for_unaffected_region(self):
        """Signal far away must not trigger bypass for a region it cannot reach."""
        # Signal at (0, 0), region at (37.2, -76.3) — 37+ degrees apart
        features = build_features(
            coordinated=[(  _LAT, _LON, 1, 0)],
            external_signals=[_no_fly(lat=0.0, lon=0.0, intensity=1.0)],
        )
        intel = classify_regions(features, min_aircraft=2)
        r = _find_intel(intel)
        self.assertIsNone(r, "Distant signal must not bypass min_aircraft for unrelated region")


# ---------------------------------------------------------------------------
# Goal 2 — ANOMALY score boost
# ---------------------------------------------------------------------------

class AnomalyScoreBoostTests(unittest.TestCase):

    def _detection_spike_features(self):
        """
        Spike set by detection layer — no boost should apply.

        Uses a persistent region (recurring_appearances=8) with very few new
        aircraft (2 out of 50) so that new_entry_ratio is low and the raw
        ANOMALY score stays well below 87 — leaving room for the 1.15x boost
        to be measurable without hitting the 100-cap.
        """
        return build_features(
            coordinated=[(  _LAT, _LON, 50, 0)],
            recurring=[(    _LAT, _LON,  8, 50, 0)],
            new_entries=[(  _LAT, _LON,  2, 0)],
            spikes=[(       _LAT, _LON, 50, 0)],   # sets spike_flag=True from detection
            external_signals=[_no_fly(intensity=1.0)],
        )

    def _external_spike_features(self):
        """
        Spike set exclusively by external signal — 1.15x boost must apply.

        Identical region to _detection_spike_features but WITHOUT the spikes
        input, so spike_flag starts False and is flipped by the external merge.
        """
        return build_features(
            coordinated=[(  _LAT, _LON, 50, 0)],
            recurring=[(    _LAT, _LON,  8, 50, 0)],
            new_entries=[(  _LAT, _LON,  2, 0)],
            # No spike from detection — external NO_FLY (1.0 > 0.5) sets it.
            external_signals=[_no_fly(intensity=1.0)],
        )

    # --- spike_set_by_external attribute ---

    def test_spike_set_by_external_true_when_external_caused_spike(self):
        features = self._external_spike_features()
        region = _find_region(features)
        self.assertIsNotNone(region)
        self.assertTrue(region.spike_set_by_external)

    def test_spike_set_by_external_false_when_detection_set_spike_first(self):
        features = self._detection_spike_features()
        region = _find_region(features)
        self.assertIsNotNone(region)
        self.assertFalse(region.spike_set_by_external)

    def test_spike_set_by_external_false_with_no_signals(self):
        features = build_features(spikes=[(  _LAT, _LON, 10, 0)])
        region = _find_region(features)
        self.assertIsNotNone(region)
        self.assertFalse(region.spike_set_by_external)

    def test_spike_set_by_external_false_with_low_intensity_signal(self):
        """NO_FLY at 0.25 intensity won't set spike_flag (threshold 0.5) → False."""
        features = build_features(
            new_entries=[(  _LAT, _LON, 10, 0)],
            external_signals=[_no_fly(intensity=0.25)],
        )
        region = _find_region(features)
        self.assertIsNotNone(region)
        self.assertFalse(region.spike_set_by_external)

    # --- ANOMALY score is actually higher ---

    def test_anomaly_score_higher_for_externally_set_spike(self):
        """
        Externally-set spike → 1.15x boost → higher ANOMALY than detection-set spike.
        Feature vectors are otherwise identical (both have spike_flag=True).
        """
        ext_score  = _anomaly_score(self._external_spike_features())
        det_score  = _anomaly_score(self._detection_spike_features())

        self.assertIsNotNone(ext_score)
        self.assertIsNotNone(det_score)
        # External: boosted; detection: un-boosted — ext must be strictly higher.
        self.assertGreater(ext_score, det_score,
                           f"Expected ext({ext_score:.1f}) > det({det_score:.1f})")

    def test_anomaly_boost_factor_single_signal(self):
        """
        With a single NO_FLY signal at intensity=1.0:
          strength=1.0, 1 type → boost = 1.0 + min(0.25, 1.0×0.10) = 1.10.
        Score ratio between boosted and un-boosted ANOMALY should equal
        1.0 + _EXT_BOOST_PER_STRENGTH.
        """
        ext_score  = _anomaly_score(self._external_spike_features())
        det_score  = _anomaly_score(self._detection_spike_features())

        self.assertIsNotNone(ext_score)
        self.assertIsNotNone(det_score)
        expected_boost = 1.0 + _EXT_BOOST_PER_STRENGTH   # 1.10
        if det_score > 0:
            ratio = ext_score / det_score
            self.assertAlmostEqual(
                ratio, expected_boost, places=2,
                msg=f"Expected ratio {expected_boost}, got {ratio:.3f}",
            )

    def test_anomaly_score_capped_at_100(self):
        """Even after 1.15x boost, ANOMALY score must not exceed 100."""
        # High-novelty region: no persistence, spike from external, high change
        features = build_features(
            new_entries=[(  _LAT, _LON, 50, 0)],
            external_signals=[_no_fly(intensity=1.0)],
        )
        intel = classify_regions(features)
        r = _find_intel(intel)
        if r is not None:
            self.assertLessEqual(r.all_scores.get("ANOMALY", 0.0), 100.0)

    def test_all_scores_bounded_after_boost(self):
        """No score for any class should exceed 100 after external boost."""
        features = self._external_spike_features()
        intel = classify_regions(features)
        r = _find_intel(intel)
        if r is not None:
            for cls, score in r.all_scores.items():
                self.assertLessEqual(score, 100.0,
                                     f"{cls} score {score:.1f} exceeds 100")

    # --- ROUTINE score is unaffected ---

    def test_routine_score_unaffected_by_external_boost(self):
        """The 1.15x multiplier must NOT touch ROUTINE score."""
        ext_features = self._external_spike_features()
        det_features = self._detection_spike_features()

        ext_intel = classify_regions(ext_features)
        det_intel = classify_regions(det_features)

        r_ext = _find_intel(ext_intel)
        r_det = _find_intel(det_intel)

        self.assertIsNotNone(r_ext)
        self.assertIsNotNone(r_det)
        self.assertAlmostEqual(
            r_ext.all_scores.get("ROUTINE", 0.0),
            r_det.all_scores.get("ROUTINE", 0.0),
            places=1,
            msg="ROUTINE score must be identical with/without external spike boost",
        )

    def test_no_boost_applied_without_external_signals(self):
        """Baseline region (no external) should not be affected by any boost."""
        feat_ext  = build_features(
            new_entries=[(  _LAT, _LON, 10, 0)],
            external_signals=[_no_fly(intensity=1.0)],
        )
        feat_base = build_features(
            new_entries=[(  _LAT, _LON, 10, 0)],
            spikes=[(        _LAT, _LON, 10, 0)],   # spike from detection, no boost
        )
        # The boosted region should score strictly higher
        ext_score  = _anomaly_score(feat_ext)
        base_score = _anomaly_score(feat_base)
        self.assertGreater(ext_score, base_score)


# ---------------------------------------------------------------------------
# Goal 3 — explanation tagging
# ---------------------------------------------------------------------------

class ExplanationTaggingTests(unittest.TestCase):

    _TAG = "external constraints detected (e.g., no-fly restriction)"

    def test_tag_present_with_prohibited_signal(self):
        features = _features_with_signal(intensity=1.0)
        intel = classify_regions(features)
        r = _find_intel(intel)
        self.assertIsNotNone(r)
        self.assertIn(self._TAG, r.explanation)

    def test_tag_present_with_restricted_signal(self):
        features = _features_with_signal(intensity=0.75)
        intel = classify_regions(features)
        r = _find_intel(intel)
        self.assertIsNotNone(r)
        self.assertIn(self._TAG, r.explanation)

    def test_tag_present_with_advisory_signal(self):
        """Even low-intensity signals that enter a region trigger the tag."""
        features = _features_with_signal(intensity=0.25)
        intel = classify_regions(features)
        r = _find_intel(intel)
        self.assertIsNotNone(r)
        self.assertIn(self._TAG, r.explanation)

    def test_tag_absent_with_no_signals(self):
        features = build_features(new_entries=[(  _LAT, _LON, 10, 0)])
        intel = classify_regions(features)
        r = _find_intel(intel)
        self.assertIsNotNone(r)
        self.assertNotIn(self._TAG, r.explanation)

    def test_tag_absent_for_all_regions_with_no_signals(self):
        features = build_features(
            spikes=[(1.0, 1.0, 10, 0), (2.0, 2.0, 10, 0)],
        )
        intel = classify_regions(features)
        for r in intel:
            self.assertNotIn(self._TAG, r.explanation)

    def test_tag_absent_for_far_region_with_nearby_signal(self):
        """Signal at (0, 0) must not tag a region at (37.2, -76.3)."""
        features = build_features(
            new_entries=[(  _LAT, _LON, 10, 0)],
            external_signals=[_no_fly(lat=0.0, lon=0.0, intensity=1.0)],
        )
        intel = classify_regions(features)
        r = _find_intel(intel)
        self.assertIsNotNone(r)
        self.assertNotIn(self._TAG, r.explanation)

    def test_tag_appended_after_flight_evidence(self):
        """The external tag must come last, not displace core evidence."""
        features = _features_with_signal(intensity=1.0, aircraft=20)
        intel = classify_regions(features)
        r = _find_intel(intel)
        self.assertIsNotNone(r)
        # Tag is last clause
        self.assertTrue(
            r.explanation.endswith(self._TAG),
            f"Expected tag at end of: {r.explanation!r}",
        )

    def test_bypassed_region_also_gets_tag(self):
        """1-aircraft bypassed region must include the external tag."""
        features = build_features(
            coordinated=[(  _LAT, _LON, 1, 0)],
            external_signals=[_no_fly(intensity=1.0)],
        )
        intel = classify_regions(features, min_aircraft=2)
        r = _find_intel(intel)
        self.assertIsNotNone(r)
        self.assertIn(self._TAG, r.explanation)


# ---------------------------------------------------------------------------
# Goal 4 — external_influence field in _log_classify_impact
# ---------------------------------------------------------------------------

class LogClassifyImpactTests(unittest.TestCase):
    """
    Test _log_classify_impact by capturing stdout and inspecting the output.
    """

    def _capture_log(self, scores, baseline_scores, spike_from_ext=True):
        """Run _log_classify_impact and return the printed line."""
        import io, contextlib
        f = RegionFeatures(lat=_LAT, lon=_LON)
        f.external_signal_count = 1
        f.external_signal_types = {SignalType.NO_FLY}
        f.external_signal_max_intensity = 1.0
        f.spike_set_by_external = spike_from_ext

        winner = max(scores, key=scores.__getitem__)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _log_classify_impact(f, scores, winner, baseline_scores)
        return buf.getvalue()

    def test_external_influence_true_when_winner_changed(self):
        """Winner changed from ROUTINE to ANOMALY → external_influence=True."""
        scores = {
            Classification.ANOMALY:              70.0,
            Classification.ROUTINE:              30.0,
            Classification.STAGING:              10.0,
            Classification.PROJECTION:            5.0,
            Classification.COORDINATED_ACTIVITY: 20.0,
        }
        baseline = {
            Classification.ANOMALY:              30.0,
            Classification.ROUTINE:              70.0,
            Classification.STAGING:              10.0,
            Classification.PROJECTION:            5.0,
            Classification.COORDINATED_ACTIVITY: 20.0,
        }
        line = self._capture_log(scores, baseline)
        self.assertIn("external_influence=True", line)

    def test_external_influence_true_when_large_delta(self):
        """Winner unchanged but delta > _EXT_INFLUENCE_THRESHOLD → True."""
        big_delta = _EXT_INFLUENCE_THRESHOLD + 1.0
        scores = {
            Classification.ANOMALY:              50.0 + big_delta,
            Classification.ROUTINE:              40.0,
            Classification.STAGING:               5.0,
            Classification.PROJECTION:            3.0,
            Classification.COORDINATED_ACTIVITY: 10.0,
        }
        baseline = {
            Classification.ANOMALY:              50.0,
            Classification.ROUTINE:              40.0,
            Classification.STAGING:               5.0,
            Classification.PROJECTION:            3.0,
            Classification.COORDINATED_ACTIVITY: 10.0,
        }
        line = self._capture_log(scores, baseline)
        self.assertIn("external_influence=True", line)

    def test_external_influence_false_when_small_delta_same_winner(self):
        """Small delta, winner unchanged → external_influence=False."""
        small_delta = _EXT_INFLUENCE_THRESHOLD - 1.0  # just below threshold
        scores = {
            Classification.ANOMALY:              50.0 + small_delta,
            Classification.ROUTINE:              40.0,
            Classification.STAGING:               5.0,
            Classification.PROJECTION:            3.0,
            Classification.COORDINATED_ACTIVITY: 10.0,
        }
        baseline = {
            Classification.ANOMALY:              50.0,
            Classification.ROUTINE:              40.0,
            Classification.STAGING:               5.0,
            Classification.PROJECTION:            3.0,
            Classification.COORDINATED_ACTIVITY: 10.0,
        }
        line = self._capture_log(scores, baseline)
        self.assertIn("external_influence=False", line)

    def test_log_line_contains_winner_change_note(self):
        """[WINNER CHANGED: ...] appended when winner differs from baseline."""
        scores = {
            Classification.ANOMALY:              80.0,
            Classification.ROUTINE:              20.0,
            Classification.STAGING:               5.0,
            Classification.PROJECTION:            3.0,
            Classification.COORDINATED_ACTIVITY: 10.0,
        }
        baseline = {
            Classification.ANOMALY:              10.0,
            Classification.ROUTINE:              80.0,
            Classification.STAGING:               5.0,
            Classification.PROJECTION:            3.0,
            Classification.COORDINATED_ACTIVITY: 10.0,
        }
        line = self._capture_log(scores, baseline)
        self.assertIn("WINNER CHANGED", line)

    def test_log_line_no_winner_change_note_when_unchanged(self):
        scores = {
            Classification.ANOMALY:              60.0,
            Classification.ROUTINE:              20.0,
            Classification.STAGING:               5.0,
            Classification.PROJECTION:            3.0,
            Classification.COORDINATED_ACTIVITY: 10.0,
        }
        baseline = {
            Classification.ANOMALY:              55.0,
            Classification.ROUTINE:              20.0,
            Classification.STAGING:               5.0,
            Classification.PROJECTION:            3.0,
            Classification.COORDINATED_ACTIVITY: 10.0,
        }
        line = self._capture_log(scores, baseline)
        self.assertNotIn("WINNER CHANGED", line)


# ---------------------------------------------------------------------------
# No-regression tests
# ---------------------------------------------------------------------------

class NoRegressionTests(unittest.TestCase):
    """External signal machinery must not alter output when signals=None/[]."""

    def _baseline(self):
        return build_features(
            spikes=[(  _LAT, _LON, 10, 0)],
            new_entries=[(  _LAT, _LON, 5, 0)],
        )

    def test_no_regression_signals_none(self):
        baseline  = self._baseline()
        no_signal = build_features(
            spikes=[(      _LAT, _LON, 10, 0)],
            new_entries=[( _LAT, _LON,  5, 0)],
            external_signals=None,
        )
        intel_base = classify_regions(baseline)
        intel_none = classify_regions(no_signal)
        self.assertEqual(len(intel_base), len(intel_none))
        for r_b, r_n in zip(intel_base, intel_none):
            self.assertEqual(r_b.classification, r_n.classification)
            self.assertAlmostEqual(r_b.score, r_n.score, places=5)

    def test_no_regression_signals_empty_list(self):
        baseline  = self._baseline()
        empty_sig = build_features(
            spikes=[(      _LAT, _LON, 10, 0)],
            new_entries=[( _LAT, _LON,  5, 0)],
            external_signals=[],
        )
        intel_base  = classify_regions(baseline)
        intel_empty = classify_regions(empty_sig)
        self.assertEqual(len(intel_base), len(intel_empty))
        for r_b, r_e in zip(intel_base, intel_empty):
            self.assertEqual(r_b.classification, r_e.classification)
            self.assertAlmostEqual(r_b.score, r_e.score, places=5)

    def test_non_nearby_signal_no_effect_on_distant_region(self):
        """Signal at (0, 0) must not change anything for region at (37.2, -76.3)."""
        lat, lon = _LAT, _LON
        baseline = build_features(new_entries=[(lat, lon, 10, 0)])
        with_far = build_features(
            new_entries=[(lat, lon, 10, 0)],
            external_signals=[_no_fly(lat=0.0, lon=0.0, intensity=1.0)],
        )
        r_base = _find_region(baseline, lat, lon)
        r_far  = _find_region(with_far, lat, lon)
        self.assertIsNotNone(r_base)
        self.assertIsNotNone(r_far)
        self.assertEqual(r_base.spike_flag,   r_far.spike_flag)
        self.assertEqual(r_base.change_score, r_far.change_score)
        self.assertFalse(r_far.spike_set_by_external)

    def test_mock_notam_signals_no_crash(self):
        """fetch_notam_signals() into classify_regions must not raise."""
        signals  = fetch_notam_signals()
        features = build_features(
            spikes=[(  _LAT, _LON, 10, 0)],
            external_signals=signals,
        )
        try:
            classify_regions(features)
        except Exception as exc:
            self.fail(f"classify_regions raised {exc!r} with mock NOTAM signals")

    def test_spike_set_by_external_defaults_false_for_all_regions(self):
        """Without external signals, no region should have spike_set_by_external=True."""
        features = build_features(
            spikes=[(1.0, 1.0, 10, 0), (2.0, 2.0, 10, 0)],
        )
        for f in features:
            self.assertFalse(f.spike_set_by_external,
                             f"Region ({f.lat},{f.lon}) unexpected spike_set_by_external")


# ---------------------------------------------------------------------------
# Signal strength / dynamic boost tests
# ---------------------------------------------------------------------------

def _make_signal(signal_type=SignalType.NO_FLY, lat=_LAT, lon=_LON, intensity=1.0):
    return ExternalSignal(
        signal_type=signal_type, lat=lat, lon=lon,
        intensity=intensity, metadata={},
    )


def _low_novelty_features(signals):
    """
    Persistent region with low novelty so the raw ANOMALY score ≈ 77.7,
    leaving room for any valid boost (up to 1.25×) to be measurable
    without hitting the 100-cap (max ≈ 97.1).

    Spike is NOT set by detection — external signals may flip it.
    """
    return build_features(
        coordinated=[(  _LAT, _LON, 50, 0)],
        recurring=[(    _LAT, _LON,  8, 50, 0)],
        new_entries=[(  _LAT, _LON,  2, 0)],
        external_signals=signals,
    )


class SignalStrengthBoostTests(unittest.TestCase):
    """Dynamic boost formula and external_signal_strength field."""

    # --- Field defaults ---

    def test_external_signal_strength_default_is_zero(self):
        """Without external signals the field must be 0.0."""
        features = build_features(spikes=[(  _LAT, _LON, 10, 0)])
        region = _find_region(features)
        self.assertIsNotNone(region)
        self.assertEqual(region.external_signal_strength, 0.0)

    # --- Strength accumulation ---

    def test_single_signal_strength_equals_intensity(self):
        """One signal at intensity=0.8 → strength=0.8."""
        features = _low_novelty_features([_make_signal(intensity=0.8)])
        region = _find_region(features)
        self.assertIsNotNone(region)
        self.assertAlmostEqual(region.external_signal_strength, 0.8, places=5)

    def test_two_same_type_signals_strength_is_sum(self):
        """Two NO_FLY signals at 0.6 each → strength = 1.2 (< cap)."""
        features = _low_novelty_features([
            _make_signal(intensity=0.6),
            _make_signal(intensity=0.6),
        ])
        region = _find_region(features)
        self.assertIsNotNone(region)
        self.assertAlmostEqual(region.external_signal_strength, 1.2, places=5)

    def test_two_mixed_type_signals_strength_is_sum(self):
        """NO_FLY(0.8) + MARITIME(0.7) → strength = 1.5 (< cap)."""
        features = _low_novelty_features([
            _make_signal(signal_type=SignalType.NO_FLY,    intensity=0.8),
            _make_signal(signal_type=SignalType.MARITIME,  intensity=0.7),
        ])
        region = _find_region(features)
        self.assertIsNotNone(region)
        self.assertAlmostEqual(region.external_signal_strength, 1.5, places=5)

    def test_strength_capped_at_ext_strength_cap(self):
        """Three signals at 1.0 → raw sum=3.0 but field capped at _EXT_STRENGTH_CAP."""
        features = _low_novelty_features([
            _make_signal(signal_type=SignalType.NO_FLY,    intensity=1.0),
            _make_signal(signal_type=SignalType.MARITIME,  intensity=1.0),
            _make_signal(signal_type=SignalType.SATELLITE, intensity=1.0),
        ])
        region = _find_region(features)
        self.assertIsNotNone(region)
        self.assertAlmostEqual(region.external_signal_strength, _EXT_STRENGTH_CAP, places=5)

    # --- Boost increases with signal count ---

    def test_two_signals_produce_higher_anomaly_than_one(self):
        """Two signals → higher strength → bigger boost → higher ANOMALY score."""
        one_signal  = _low_novelty_features([_make_signal(intensity=1.0)])
        two_signals = _low_novelty_features([
            _make_signal(intensity=1.0),
            _make_signal(signal_type=SignalType.MARITIME, intensity=1.0),
        ])
        score_one = _anomaly_score(one_signal)
        score_two = _anomaly_score(two_signals)
        self.assertIsNotNone(score_one)
        self.assertIsNotNone(score_two)
        self.assertGreater(score_two, score_one,
                           f"Two signals ({score_two:.1f}) should beat one ({score_one:.1f})")

    # --- Multi-type bonus ---

    def test_mixed_types_produce_higher_anomaly_than_same_type(self):
        """Two different SignalTypes get +_EXT_BOOST_MULTI_TYPE; same type does not."""
        same_type = _low_novelty_features([
            _make_signal(signal_type=SignalType.NO_FLY, intensity=1.0),
            _make_signal(signal_type=SignalType.NO_FLY, intensity=1.0),
        ])
        mixed_type = _low_novelty_features([
            _make_signal(signal_type=SignalType.NO_FLY,   intensity=1.0),
            _make_signal(signal_type=SignalType.MARITIME, intensity=1.0),
        ])
        score_same  = _anomaly_score(same_type)
        score_mixed = _anomaly_score(mixed_type)
        self.assertIsNotNone(score_same)
        self.assertIsNotNone(score_mixed)
        self.assertGreater(score_mixed, score_same,
                           f"Mixed types ({score_mixed:.1f}) should beat same type ({score_same:.1f})")

    # --- Formula verification ---

    def test_boost_formula_single_signal(self):
        """
        Verify that classify_regions applies exactly the expected boost factor
        for a single NO_FLY signal at intensity=1.0.

        strength=1.0, 1 type → boost = 1.0 + min(_EXT_BOOST_MAX, 1.0 × _EXT_BOOST_PER_STRENGTH)
                                      = 1.0 + 0.10 = 1.10.
        """
        no_ext = _low_novelty_features(None)          # no signals → no boost
        one_sig = _low_novelty_features([_make_signal(intensity=1.0)])

        # Recompute baseline: region with same flags but spike set from detection
        # (so no boost applied) — use the detection-set fixture to get the un-boosted score.
        det_features = build_features(
            coordinated=[(  _LAT, _LON, 50, 0)],
            recurring=[(    _LAT, _LON,  8, 50, 0)],
            new_entries=[(  _LAT, _LON,  2, 0)],
            spikes=[(       _LAT, _LON, 50, 0)],   # detection sets spike → no boost
            external_signals=[_make_signal(intensity=1.0)],
        )
        ext_score = _anomaly_score(one_sig)
        det_score = _anomaly_score(det_features)

        self.assertIsNotNone(ext_score)
        self.assertIsNotNone(det_score)
        expected_boost = 1.0 + min(_EXT_BOOST_MAX, 1.0 * _EXT_BOOST_PER_STRENGTH)
        if det_score > 0:
            ratio = ext_score / det_score
            self.assertAlmostEqual(
                ratio, expected_boost, places=2,
                msg=f"Expected boost {expected_boost:.2f}, got ratio {ratio:.3f}",
            )

    def test_boost_formula_max_strength_two_types(self):
        """
        Two different types at 1.0 each → strength=2.0 (= cap), 2 types.
        boost = 1.0 + min(_EXT_BOOST_MAX, 2.0 × _EXT_BOOST_PER_STRENGTH) + _EXT_BOOST_MULTI_TYPE
              = 1.0 + 0.20 + 0.05 = 1.25.
        """
        det_features = build_features(
            coordinated=[(  _LAT, _LON, 50, 0)],
            recurring=[(    _LAT, _LON,  8, 50, 0)],
            new_entries=[(  _LAT, _LON,  2, 0)],
            spikes=[(       _LAT, _LON, 50, 0)],   # detection sets spike → no boost
            external_signals=[
                _make_signal(signal_type=SignalType.NO_FLY,   intensity=1.0),
                _make_signal(signal_type=SignalType.MARITIME,  intensity=1.0),
            ],
        )
        ext_features = _low_novelty_features([
            _make_signal(signal_type=SignalType.NO_FLY,   intensity=1.0),
            _make_signal(signal_type=SignalType.MARITIME,  intensity=1.0),
        ])
        det_score = _anomaly_score(det_features)
        ext_score = _anomaly_score(ext_features)

        self.assertIsNotNone(ext_score)
        self.assertIsNotNone(det_score)
        strength      = min(_EXT_STRENGTH_CAP, 2.0)   # 2.0
        expected_boost = (
            1.0
            + min(_EXT_BOOST_MAX, strength * _EXT_BOOST_PER_STRENGTH)
            + _EXT_BOOST_MULTI_TYPE
        )
        if det_score > 0:
            ratio = ext_score / det_score
            self.assertAlmostEqual(
                ratio, expected_boost, places=2,
                msg=f"Expected boost {expected_boost:.2f}, got ratio {ratio:.3f}",
            )

    def test_no_boost_without_spike_set_by_external(self):
        """strength field is set but if spike was detection-layer, no boost is applied."""
        det_features = build_features(
            coordinated=[(  _LAT, _LON, 50, 0)],
            recurring=[(    _LAT, _LON,  8, 50, 0)],
            new_entries=[(  _LAT, _LON,  2, 0)],
            spikes=[(       _LAT, _LON, 50, 0)],
            external_signals=[_make_signal(intensity=1.0)],
        )
        region = _find_region(det_features)
        self.assertIsNotNone(region)
        # spike was from detection: strength is set but boost should not apply
        self.assertFalse(region.spike_set_by_external)
        self.assertGreater(region.external_signal_strength, 0.0,
                           "strength field should still be populated")


if __name__ == "__main__":
    unittest.main()
