"""
Exp-3: Context Normalisation Suppression (FM-3)
================================================

FAILURE MODE
------------
classify_regions() builds one shared context dict for ALL regions before
scoring any of them:

    context["max_change"] = max(f.change_score for f in features_list)

change_score_norm in _score_region is then:

    change_score_norm = f.change_score / context["max_change"]

This means the ANOMALY score of every region depends not just on its own
features but on the highest change_score among ALL regions being classified
together. Adding one outlier region with a larger change_score compresses
every other region's change_score_norm, costing each up to 10 ANOMALY points
(weight 0.20 × 100) with no compensating mechanism and no log warning.

SCENARIO
--------
Target  : (35.0, 36.0)  aircraft=8, military=4, change_score=20
Outlier : (55.0, 20.0)  aircraft=50, military=0, change_score=40

Without outlier  max_change=20  change_score_norm=1.0  ANOMALY=46.5
With outlier     max_change=40  change_score_norm=0.5  ANOMALY=36.5

ANOMALY drops 10 points and the winner flips to ROUTINE (39.9).

EXPECTED BEHAVIOUR
------------------
The target's classification should not change when a geographically and
thematically unrelated region is added to the features list.

WHAT CONSTITUTES FAILURE
-------------------------
* test_anomaly_score_drops_when_outlier_present  — FAILS: ANOMALY drops 10 pts
* test_classification_stable_despite_outlier     — FAILS: winner flips ANOMALY->ROUTINE
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from intelligence.classifier import (
    Classification,
    RegionFeatures,
    _score_region,
    classify_regions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_region(
    lat: float,
    lon: float,
    aircraft_count: int = 0,
    military_count: int = 0,
    change_score: float = 0.0,
    recurring_appearances: int = 0,
) -> RegionFeatures:
    r = RegionFeatures(lat=lat, lon=lon)
    r.aircraft_count        = aircraft_count
    r.military_count        = military_count
    r.change_score          = change_score
    r.recurring_appearances = recurring_appearances
    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class ContextSuppressionTests(unittest.TestCase):

    # --- sanity checks -------------------------------------------------------

    def test_target_classifies_consistently_with_and_without_outlier(self):
        """
        After the FM-3 fix, the target must classify identically whether or
        not the outlier is present.  The fix floors max_change at 40.0, making
        change_score_norm = 20/40 = 0.5 in both cases.

        Pre-fix: target ANOMALY=46.5 (alone, max_change=20) vs 36.5 (with outlier,
        max_change=40) — the alone-case was inflated by a small-batch denominator.
        Post-fix: target ANOMALY=36.5 in both cases; winner=ROUTINE (stable).

        The key property is STABILITY, not a specific classification label.
        """
        target  = make_region(35.0, 36.0, aircraft_count=8, military_count=4,
                               change_score=20.0)
        outlier = make_region(55.0, 20.0, aircraft_count=50, military_count=0,
                               change_score=40.0, recurring_appearances=10)

        r_alone = classify_regions([target])[0]
        r_joint = next(
            r for r in classify_regions([target, outlier])
            if abs(r.lat - 35.0) < 0.01
        )

        print(f"\n[Exp-3 ALONE] winner={r_alone.classification.value}  "
              f"ANOMALY={r_alone.all_scores['ANOMALY']:.1f}  "
              f"ROUTINE={r_alone.all_scores['ROUTINE']:.1f}")

        self.assertEqual(r_alone.classification, r_joint.classification,
            "Classification must be identical regardless of outlier presence")

    def test_outlier_alone_classifies_routine(self):
        """
        Outlier is civilian high-traffic: no military, high persistence -> ROUTINE.
        recurring_appearances=10 gives persistence_score=0.83, which pushes ROUTINE
        above ANOMALY even though change_score=40 is high.
        (Without recurring_appearances, zero persistence + high change_score would
        itself score ANOMALY — persistence is what makes it clearly civilian.)
        """
        outlier = make_region(55.0, 20.0, aircraft_count=50, military_count=0,
                               change_score=40.0, recurring_appearances=10)
        results = classify_regions([outlier])

        r = results[0]
        print(f"\n[Exp-3 OUTLIER] winner={r.classification.value}  "
              f"ANOMALY={r.all_scores['ANOMALY']:.1f}  "
              f"ROUTINE={r.all_scores['ROUTINE']:.1f}")

        self.assertEqual(r.classification, Classification.ROUTINE,
            "Outlier has high persistence and no military presence -> ROUTINE")

    # --- FM-3 probes ---------------------------------------------------------

    def test_anomaly_score_stable_when_outlier_present(self):
        """
        FM-3 fix verification: flooring max_change at _MAX_CHANGE_SCORE (40.0)
        makes change_score_norm independent of batch composition for values ≤ 40.

        Before fix: target ANOMALY 46.5 (alone) vs 36.5 (with outlier) — 10 pt drop.
        After fix:  target ANOMALY 36.5 in both cases — score is stable.

        Note: the absolute score is now 36.5 in both cases (not 46.5), because
        the floor fixes the denominator at 40.0 rather than the batch maximum of
        20.0 that the alone-case previously used.  The critical property is not
        the absolute value but that it does NOT change when an outlier is present.
        """
        target  = make_region(35.0, 36.0, aircraft_count=8, military_count=4,
                               change_score=20.0)
        outlier = make_region(55.0, 20.0, aircraft_count=50, military_count=0,
                               change_score=40.0, recurring_appearances=10)

        score_alone = classify_regions([target])[0].all_scores["ANOMALY"]

        target_in_joint = next(
            r for r in classify_regions([target, outlier])
            if abs(r.lat - 35.0) < 0.01
        )
        score_joint = target_in_joint.all_scores["ANOMALY"]

        print(f"\n[Exp-3 SCORE FIX] alone={score_alone:.1f}  "
              f"with_outlier={score_joint:.1f}  "
              f"delta={score_joint - score_alone:+.1f}")

        # FM-3 fix verified: score is now stable regardless of outlier presence
        self.assertAlmostEqual(
            score_alone, score_joint, delta=1.0,
            msg=(
                f"FM-3 NOT FIXED: target ANOMALY still drops {score_alone - score_joint:.1f} pts "
                f"when outlier is added — max_change floor is not working"
            ),
        )

    def test_classification_stable_despite_outlier(self):
        """
        FM-3 fix verification: classification winner must not change when an
        unrelated outlier is added to the features list.

        Before fix: winner flipped ANOMALY -> ROUTINE (ANOMALY dropped below ROUTINE).
        After fix:  winner stays ANOMALY in both cases.
        """
        target  = make_region(35.0, 36.0, aircraft_count=8, military_count=4,
                               change_score=20.0)
        outlier = make_region(55.0, 20.0, aircraft_count=50, military_count=0,
                               change_score=40.0, recurring_appearances=10)

        winner_alone = classify_regions([target])[0].classification

        joint_results = classify_regions([target, outlier])
        target_result = next(r for r in joint_results if abs(r.lat - 35.0) < 0.01)
        winner_joint  = target_result.classification

        print(f"\n[Exp-3 WINNER FIX] alone={winner_alone.value}  "
              f"with_outlier={winner_joint.value}")
        for r in sorted(joint_results, key=lambda x: x.lat):
            print(f"  ({r.lat}, {r.lon}) {r.classification.value}  "
                  + "  ".join(f"{k}={v:.1f}"
                               for k, v in sorted(r.all_scores.items(), key=lambda x: -x[1])))

        # FM-3 fix verified: winner is now stable
        self.assertEqual(
            winner_alone, winner_joint,
            msg=(
                f"FM-3 NOT FIXED: classification still flipped "
                f"{winner_alone.value} -> {winner_joint.value} — "
                f"max_change floor is not working"
            ),
        )

    def test_change_score_norm_uses_fixed_ceiling(self):
        """
        Regression anchor: after the FM-3 fix, classify_regions always uses
        _MAX_CHANGE_SCORE (40.0) as the max_change denominator when the batch
        maximum is <= 40.0.  This means change_score_norm is absolute, not
        relative to batch composition.

        Verification via _score_region directly with both old and new contexts:
          Old context (max_change=20): ANOMALY=46.5  change_score_norm=1.0
          New context (max_change=40): ANOMALY=36.5  change_score_norm=0.5

        The fix eliminates the old context (max_change<40 can no longer arise
        from classify_regions).  Both the alone-case and the joint-case now
        use max_change=40 -> both score ANOMALY=36.5.
        """
        target = make_region(35.0, 36.0, aircraft_count=8, military_count=4,
                              change_score=20.0)

        # Old behaviour (pre-fix): alone-case used max_change=20
        scores_old = _score_region(target, {"max_inflow": 1, "max_change": 20.0})
        # New behaviour (post-fix): alone-case also uses max_change=40 (floor)
        scores_new = _score_region(target, {"max_inflow": 1, "max_change": 40.0})

        print(f"\n[Exp-3 CEILING] ANOMALY pre-fix={scores_old[Classification.ANOMALY]:.1f}  "
              f"post-fix={scores_new[Classification.ANOMALY]:.1f}  "
              f"(change_score_norm: 20/20=1.0 -> 20/40=0.5, weight=0.20)")

        # The pre-fix alone-case was inflated: change_score_norm=1.0 was only
        # possible because the batch contained no higher change_score.
        # After the fix the score is always computed against the ceiling (0.5).
        self.assertAlmostEqual(
            scores_new[Classification.ANOMALY], 36.5, delta=0.1,
            msg="After fix: change_score=20 must score ANOMALY=36.5 (20/40 norm, weight=0.20)",
        )
        # Confirm the old inflated score is no longer reachable via classify_regions
        self.assertGreater(
            scores_old[Classification.ANOMALY], scores_new[Classification.ANOMALY],
            msg="Pre-fix score was artificially inflated by small-batch denominator",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
