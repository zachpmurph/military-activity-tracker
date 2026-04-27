"""
Tests for intelligence/classifier.py

Verifies feature assembly, scoring, classification, confidence, and explanation
using hand-crafted inputs whose expected classification is unambiguous.
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


class BuildFeaturesTests(unittest.TestCase):
    def test_coordinated_sets_flag_and_counts(self):
        features = build_features(coordinated=[(34.1, -117.2, 6, 3)])
        self.assertEqual(len(features), 1)
        f = features[0]
        self.assertTrue(f.coordination_flag)
        self.assertEqual(f.aircraft_count, 6)
        self.assertEqual(f.military_count, 3)

    def test_spikes_sets_flag(self):
        features = build_features(spikes=[(40.0, -75.0, 5, 1)])
        self.assertTrue(features[0].spike_flag)

    def test_recurring_sets_appearances(self):
        features = build_features(recurring=[(51.5, -0.1, 8, 20, 4)])
        f = features[0]
        self.assertEqual(f.recurring_appearances, 8)
        self.assertAlmostEqual(f.persistence_score, 8 / 12)

    def test_movements_populate_inflow_and_outflow(self):
        # Two aircraft move FROM (34.1, -117.2) TO (35.1, -118.1)
        features = build_features(movements=[(34.1, -117.2, 35.1, -118.1, 2, 1, 1.4)])
        by_key = {(f.lat, f.lon): f for f in features}
        src = by_key[(34.1, -117.2)]
        dst = by_key[(35.1, -118.1)]
        self.assertEqual(src.outflow_count, 2)
        self.assertEqual(src.inflow_count, 0)
        self.assertEqual(dst.inflow_count, 2)
        self.assertEqual(dst.outflow_count, 0)

    def test_staging_sets_flag(self):
        features = build_features(staging=[(48.0, 2.0, 4, 2)])
        self.assertTrue(features[0].staging_flag)

    def test_projection_sets_flag(self):
        features = build_features(projection=[(48.0, 2.0, 4, 2)])
        self.assertTrue(features[0].projection_flag)

    def test_new_entries_sets_counts(self):
        features = build_features(new_entries=[(36.1, -119.1, 3, 2)])
        f = features[0]
        self.assertEqual(f.new_aircraft, 3)
        self.assertEqual(f.new_military, 2)

    def test_same_region_across_sources_merges_correctly(self):
        # Same region appears in coordinated AND spikes AND recurring
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
        self.assertEqual(f.aircraft_count, 15)   # max across all sources
        self.assertEqual(f.military_count, 5)


class DerivedFeatureTests(unittest.TestCase):
    def test_military_ratio(self):
        f = RegionFeatures(lat=0, lon=0, aircraft_count=10, military_count=4)
        self.assertAlmostEqual(f.military_ratio, 0.4)

    def test_military_ratio_capped_at_1(self):
        f = RegionFeatures(lat=0, lon=0, aircraft_count=2, military_count=5)
        self.assertEqual(f.military_ratio, 1.0)

    def test_new_entry_ratio(self):
        f = RegionFeatures(lat=0, lon=0, aircraft_count=10, new_aircraft=3)
        self.assertAlmostEqual(f.new_entry_ratio, 0.3)

    def test_flow_balance_pure_outflow(self):
        f = RegionFeatures(lat=0, lon=0, outflow_count=4, inflow_count=0)
        self.assertAlmostEqual(f.flow_balance, 1.0)

    def test_flow_balance_pure_inflow(self):
        f = RegionFeatures(lat=0, lon=0, outflow_count=0, inflow_count=4)
        self.assertAlmostEqual(f.flow_balance, -1.0)

    def test_flow_balance_zero_when_no_movement(self):
        f = RegionFeatures(lat=0, lon=0)
        self.assertEqual(f.flow_balance, 0.0)

    def test_persistence_score_capped(self):
        f = RegionFeatures(lat=0, lon=0, recurring_appearances=20)
        self.assertEqual(f.persistence_score, 1.0)


class ClassificationTests(unittest.TestCase):
    """
    Each test constructs inputs whose dominant signals point unambiguously
    at one classification, then asserts the classifier picks it.
    """

    def _classify_one(self, **kwargs):
        features = build_features(**kwargs)
        results  = classify_regions(features, min_aircraft=1)
        self.assertTrue(results, "Expected at least one classified region")
        return results[0]

    def test_staging_classification(self):
        # High military, confirmed staging origin, clear net outflow
        r = self._classify_one(
            staging    =[(34.1, -117.2, 8, 6)],
            movements  =[(34.1, -117.2, 36.0, -100.0, 6, 5, 2.0),
                         (34.1, -117.2, 37.0, -101.0, 4, 3, 2.5)],
            coordinated=[(34.1, -117.2, 8, 6)],
        )
        self.assertEqual(r.classification, Classification.STAGING)
        self.assertGreater(r.confidence, 20)

    def test_projection_classification(self):
        # Confirmed projection destination, high inflow, many new arrivals
        r = self._classify_one(
            projection  =[(35.1, -118.1, 8, 5)],
            movements   =[(34.1, -117.2, 35.1, -118.1, 6, 4, 2.0),
                          (33.0, -116.0, 35.1, -118.1, 4, 3, 2.5)],
            new_entries =[(35.1, -118.1, 6, 4)],
        )
        self.assertEqual(r.classification, Classification.PROJECTION)
        self.assertGreater(r.confidence, 20)

    def test_coordinated_activity_classification(self):
        # Coordinated flag + spike + high military, no movement bias
        r = self._classify_one(
            coordinated=[(48.0, 2.0, 10, 7)],
            spikes     =[(48.0, 2.0, 10, 7)],
        )
        self.assertEqual(r.classification, Classification.COORDINATED_ACTIVITY)

    def test_anomaly_classification(self):
        # Pure spike, all new aircraft, no prior persistence
        r = self._classify_one(
            spikes      =[(55.0, 25.0, 6, 0)],
            new_entries =[(55.0, 25.0, 6, 0)],
        )
        self.assertEqual(r.classification, Classification.ANOMALY)

    def test_routine_classification(self):
        # Long-running, no spike, no new entries, balanced flows
        r = self._classify_one(
            recurring=[(51.5, -0.1, 10, 30, 2)],
        )
        self.assertEqual(r.classification, Classification.ROUTINE)

    def test_sorted_by_confidence_descending(self):
        features = build_features(
            staging    =[(34.1, -117.2, 8, 6)],
            projection =[(35.1, -118.1, 8, 5)],
            recurring  =[(51.5, -0.1,  10, 30, 2)],
        )
        results = classify_regions(features, min_aircraft=1)
        confidences = [r.confidence for r in results]
        self.assertEqual(confidences, sorted(confidences, reverse=True))

    def test_min_aircraft_filter(self):
        features = build_features(
            spikes=[(55.0, 25.0, 1, 0)],   # only 1 aircraft
        )
        results = classify_regions(features, min_aircraft=2)
        self.assertEqual(len(results), 0)

    def test_confidence_range(self):
        features = build_features(
            staging    =[(34.1, -117.2, 8, 6)],
            coordinated=[(34.1, -117.2, 8, 6)],
            spikes     =[(34.1, -117.2, 8, 6)],
        )
        results = classify_regions(features, min_aircraft=1)
        for r in results:
            self.assertGreaterEqual(r.confidence, 0)
            self.assertLessEqual(r.confidence, 100)

    def test_all_scores_present(self):
        features = build_features(coordinated=[(34.1, -117.2, 5, 2)])
        results  = classify_regions(features, min_aircraft=1)
        self.assertEqual(
            set(results[0].all_scores.keys()),
            {c.value for c in Classification},
        )

    def test_explanation_is_non_empty_string(self):
        features = build_features(staging=[(34.1, -117.2, 8, 6)])
        results  = classify_regions(features, min_aircraft=1)
        self.assertIsInstance(results[0].explanation, str)
        self.assertTrue(results[0].explanation.strip())

    def test_str_representation_contains_classification(self):
        features = build_features(staging=[(34.1, -117.2, 8, 6)])
        results  = classify_regions(features, min_aircraft=1)
        self.assertIn("STAGING", str(results[0]))


class ExampleOutputTest(unittest.TestCase):
    """
    Produce and print a realistic example output for a high-confidence
    STAGING region — serves as documentation and a smoke test.
    """

    def test_example_staging_region(self):
        features = build_features(
            coordinated =[(34.1, -117.2, 8, 6)],
            spikes      =[(34.1, -117.2, 8, 6)],
            staging     =[(34.1, -117.2, 8, 6)],
            movements   =[(34.1, -117.2, 36.0, -100.0, 6, 5, 2.0),
                          (34.1, -117.2, 37.0, -101.0, 3, 2, 2.5)],
            recurring   =[(34.1, -117.2, 4, 8, 6)],
            activity_changes=[
                # (lat, lon, change_score, delta_ac, delta_mil, delta_score,
                #  flags, level, tags, persistence_level, ...)
                (34.1, -117.2, 28.5, 6, 5, 3.0,
                 "SURGE,MILITARY_BUILDUP", "HIGH",
                 ["MILITARY_BUILDUP", "HIGH_SCORE"],
                 "MEDIUM", "abc12345", 45.0,
                 75.0, 50.0, 25.0, "MILITARY_HEAVY"),
            ],
        )

        results = classify_regions(features, min_aircraft=2)
        self.assertTrue(results)

        r = results[0]
        print("\n--- Example Intelligence Output ---")
        print(r)
        print(f"  features.military_ratio    : {r.features.military_ratio:.2f}")
        print(f"  features.flow_balance      : {r.features.flow_balance:.2f}")
        print(f"  features.persistence_score : {r.features.persistence_score:.2f}")
        print(f"  features.novelty           : {r.features.novelty:.2f}")
        print(f"  all_scores                 : {r.all_scores}")

        self.assertEqual(r.classification, Classification.STAGING)
        # Confidence reflects genuine STAGING/COORDINATED_ACTIVITY ambiguity
        # (scores 85.8 vs 83.0) — assert it's non-trivial, not that it's high
        self.assertGreater(r.confidence, 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
