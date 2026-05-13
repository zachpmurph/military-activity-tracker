"""
Exp-4: Bypass Ghost Classification (FM-4)
==========================================

FAILURE MODE
------------
The min-aircraft bypass in classify_regions() uses RAW signal intensity to
decide whether a sub-threshold region should be scored:

    classifier.py:757   external_signal_max_intensity = max(s.intensity ...)
    classifier.py:810   f.external_signal_max_intensity >= _EXT_BYPASS_INTENSITY

This compares the raw intensity stored in the ExternalSignal object, not the
EFFECTIVE intensity (intensity × distance_weight) that actually influences
the feature vector.

A signal placed just inside the proximity radius at the bypass threshold
intensity qualifies for the bypass but contributes almost nothing to
features because its Chebyshev distance weight ≈ 0:

    offset = 1.499°  ->  weight = 1 − (1.499 / 1.5) ≈ 0.000667
    raw_intensity   = 0.75   (meets bypass threshold: 0.75 ≥ 0.75)
    effective       = 0.75 × 0.000667 ≈ 0.0005   (effectively zero)

The ghost signal grants a region access to scoring but then contributes
nothing meaningful to the feature vector. Classification is driven entirely
by absence-of-signal default scores rather than real evidence.

SCENARIO
--------
Region (35.0, 36.0): aircraft_count=0 — below min_aircraft=2.

Reference (center signal, offset=0.0°):
  effective=0.75, spike_flag=True, strength=0.750 -> ANOMALY wins (legitimate)

Ghost (edge signal, offset=1.499°):
  effective≈0.0005, spike_flag=False, strength≈0.0005 -> ROUTINE wins (ghost)

EXPECTED BEHAVIOUR
------------------
Any region that bypasses the min-aircraft guard should have a meaningful
effective signal (strength > 0.1) — the signal that granted the bypass
should actually influence the classification.

WHAT CONSTITUTES FAILURE
-------------------------
* test_ghost_signal_bypasses_with_near_zero_strength — FAILS:
    the region enters scoring (bypass fires) but external_signal_strength ≈ 0,
    meaning the bypass was granted by a ghost that changed nothing.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from intelligence.classifier import (
    Classification,
    build_features,
    classify_regions,
    _EXT_BYPASS_INTENSITY,
    _EXT_BYPASS_MIN_STRENGTH,
)
from intelligence.external_features import ExternalSignal, SignalType, _PROXIMITY_RADIUS

_LAT = 35.0
_LON = 36.0

# Just inside the proximity boundary: Chebyshev distance = 1.499°
# weight = 1 − (1.499 / 1.5) ≈ 0.000667 -> effective ≈ 0.0005
_GHOST_OFFSET = _PROXIMITY_RADIUS - 0.001


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_signal(intensity: float, offset: float) -> ExternalSignal:
    """Place a NO_FLY signal at (_LAT + offset, _LON)."""
    return ExternalSignal(
        signal_type=SignalType.NO_FLY,
        lat=_LAT + offset,
        lon=_LON,
        intensity=intensity,
        metadata={"scenario": "exp4"},
    )


def make_empty_region_detection():
    """
    Minimal detection input that registers a region at (_LAT, _LON) in
    build_features with aircraft_count=0 — below the min_aircraft=2 guard.
    """
    # new_entries: (lat, lon, new_aircraft, new_military)
    # new_ac=0 -> aircraft_count stays 0 after max(existing, 0)
    return [(round(_LAT, 1), round(_LON, 1), 0, 0)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class BypassGhostTests(unittest.TestCase):

    # --- baseline: guard works without signals -------------------------------

    def test_no_signal_region_is_filtered(self):
        """Without any signal, aircraft_count=0 region never reaches scoring."""
        features = build_features(new_entries=make_empty_region_detection())
        results  = classify_regions(features, min_aircraft=2)

        self.assertEqual(len(results), 0,
            "Region with 0 aircraft and no signals must be filtered by min_aircraft guard")

    # --- reference: legitimate bypass works as designed ---------------------

    def test_center_signal_is_a_legitimate_bypass(self):
        """
        A signal at the region center (offset=0) with intensity=0.75 is a
        genuine bypass: effective=0.75 sets spike_flag=True and produces
        meaningful external_signal_strength.  The region's classification
        is driven by real signal evidence.
        """
        signal   = make_signal(intensity=_EXT_BYPASS_INTENSITY, offset=0.0)
        features = build_features(
            new_entries=make_empty_region_detection(),
            external_signals=[signal],
        )
        results = classify_regions(features, min_aircraft=2)

        self.assertEqual(len(results), 1,
            "Center signal at threshold intensity must trigger a legitimate bypass")

        f = features[0]
        print(f"\n[Exp-4 CENTER] aircraft={f.aircraft_count}  "
              f"raw_intensity={f.external_signal_max_intensity:.4f}  "
              f"effective_strength={f.external_signal_strength:.4f}  "
              f"spike_flag={f.spike_flag}  "
              f"winner={results[0].classification.value}")

        self.assertTrue(f.spike_flag,
            "Center signal (eff=0.75 > 0.5 threshold) must set spike_flag")
        self.assertGreater(f.external_signal_strength, 0.5,
            "Legitimate bypass must produce meaningful effective signal strength")

    # --- FM-4 probe: ghost bypass --------------------------------------------

    def test_ghost_signal_rejected_by_effective_strength_floor(self):
        """
        FM-4 fix verification: the dual-condition bypass now rejects a ghost
        signal whose raw intensity meets the threshold but effective strength
        does not.

        Fix applied (classifier.py):
            high_conf_ext = (
                external_signal_count > 0
                AND external_signal_max_intensity >= 0.75   # raw: RESTRICTED+
                AND external_signal_strength      >= 0.10   # effective: non-trivial
            )

        Ghost signal (offset=1.499°, intensity=0.75):
          raw_intensity   = 0.75  (still passes first condition)
          effective       = 0.75 * 0.000667 ≈ 0.0005  (fails second condition)
          -> region correctly filtered out

        Before fix: len(results) == 1, effective_strength ≈ 0  (bug)
        After fix:  len(results) == 0  (correct)
        """
        signal   = make_signal(intensity=_EXT_BYPASS_INTENSITY, offset=_GHOST_OFFSET)
        features = build_features(
            new_entries=make_empty_region_detection(),
            external_signals=[signal],
        )
        results  = classify_regions(features, min_aircraft=2)

        f = features[0]
        print(f"\n[Exp-4 GHOST FIX] aircraft={f.aircraft_count}  "
              f"raw_intensity={f.external_signal_max_intensity:.4f}  "
              f"effective_strength={f.external_signal_strength:.6f}  "
              f"bypassed={len(results) > 0}")

        # Confirm the raw-intensity condition still matches (this is why old code was wrong)
        self.assertGreaterEqual(
            f.external_signal_max_intensity, _EXT_BYPASS_INTENSITY,
            "Raw intensity still meets old bypass threshold — confirming the old bug was real",
        )
        # Confirm the effective-strength condition fails (this is what the fix catches)
        self.assertLess(
            f.external_signal_strength, _EXT_BYPASS_MIN_STRENGTH,
            "Effective strength below new floor — confirming why the fix rejects this signal",
        )
        # The fix: ghost is correctly rejected despite passing the raw-intensity check
        self.assertEqual(len(results), 0,
            "FM-4 FIX VERIFIED: ghost signal correctly rejected by effective-strength floor")

    def test_ghost_region_is_absent_from_results(self):
        """
        Supplementary verification: a sub-threshold region with a ghost signal
        must not appear in classify_regions output after the fix.
        """
        signal   = make_signal(intensity=_EXT_BYPASS_INTENSITY, offset=_GHOST_OFFSET)
        features = build_features(
            new_entries=make_empty_region_detection(),
            external_signals=[signal],
        )
        results  = classify_regions(features, min_aircraft=2)

        f = features[0]
        print(f"\n[Exp-4 GHOST ABSENT] aircraft={f.aircraft_count}  "
              f"raw={f.external_signal_max_intensity:.4f}  "
              f"strength={f.external_signal_strength:.6f}  "
              f"len(results)={len(results)}")

        self.assertEqual(len(results), 0,
            "Sub-threshold region with ghost signal must not appear in results")
        self.assertLess(f.change_score, 0.1,
            "Ghost signal contributes negligible change_score (eff x 0.80 x 40 ≈ 0.016)")

    # --- boundary: outside radius cannot bypass ------------------------------

    def test_signal_outside_radius_does_not_bypass(self):
        """
        A signal 1 meter outside the proximity radius is excluded from
        _signals_near_weighted entirely — external_signal_count stays 0,
        so the bypass cannot fire regardless of raw intensity.
        """
        signal   = make_signal(intensity=1.0, offset=_PROXIMITY_RADIUS + 0.001)
        features = build_features(
            new_entries=make_empty_region_detection(),
            external_signals=[signal],
        )
        results  = classify_regions(features, min_aircraft=2)

        self.assertEqual(len(results), 0,
            "Signal just outside proximity radius must not trigger bypass "
            "(excluded from _signals_near_weighted, external_signal_count=0)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
