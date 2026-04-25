import ast
import io
import sqlite3
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import query


class RecurringRegionsTests(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.cursor = self.connection.cursor()
        self.cursor.execute("""
            CREATE TABLE aircraft_positions (
                icao24 TEXT,
                callsign TEXT,
                lat REAL,
                lon REAL,
                altitude REAL,
                speed REAL,
                timestamp REAL,
                type TEXT,
                behavior TEXT,
                score INTEGER
            )
        """)

    def tearDown(self):
        self.connection.close()

    def insert_position(self, icao24, lat, lon, timestamp, aircraft_type, score):
        self.cursor.execute("""
            INSERT INTO aircraft_positions
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            icao24,
            icao24,
            lat,
            lon,
            30000,
            400,
            timestamp,
            aircraft_type,
            "NORMAL",
            score,
        ))

    def test_recurring_regions_prints_top_recurring_cluster(self):
        now = 1_800_000
        region_rows = [
            ("alpha1", 34.11, -117.21, now - 300, "US_CARGO", 7),
            ("alpha2", 34.12, -117.19, now - 300, "UK_CARGO", 6),
            ("alpha3", 34.13, -117.18, now - 300, "UNKNOWN", 5),
            ("bravo1", 34.14, -117.22, now - 2_100, "US_CARGO", 6),
            ("bravo2", 34.12, -117.24, now - 2_100, "UNKNOWN", 4),
            ("bravo3", 34.10, -117.23, now - 2_100, "TANKER", 8),
            ("charlie1", 34.14, -117.20, now - 3_900, "UNKNOWN", 5),
            ("charlie2", 34.13, -117.21, now - 3_900, "US_CARGO", 7),
            ("charlie3", 34.12, -117.24, now - 3_900, "UNKNOWN", 6),
            ("delta1", 40.01, -75.02, now - 300, "UNKNOWN", 5),
            ("delta2", 40.03, -75.01, now - 300, "UNKNOWN", 5),
            ("delta3", 40.02, -75.04, now - 300, "UNKNOWN", 5),
            ("echo1", 40.02, -75.03, now - 2_100, "UNKNOWN", 5),
            ("echo2", 40.04, -75.04, now - 2_100, "UNKNOWN", 5),
            ("echo3", 40.01, -75.00, now - 2_100, "UNKNOWN", 5),
        ]

        for row in region_rows:
            self.insert_position(*row)

        self.connection.commit()

        output = io.StringIO()
        with patch.object(query.time, "time", return_value=now):
            with redirect_stdout(output):
                query.recurring_regions(self.cursor)

        lines = [line.strip() for line in output.getvalue().splitlines() if line.strip()]

        self.assertEqual(lines[0], "--- Recurring Activity Regions (Last 6 Hours) ---")
        self.assertIn("(34.1, -117.2, 3, 9, 5)", lines)
        self.assertNotIn("(40.0, -75.0, 2, 6, 0)", lines)


class RankRegionsTests(unittest.TestCase):
    def test_rank_regions_combines_region_signals(self):
        def fake_coordinated_activity(cursor):
            return [(34.12, -117.24, 4, 2), (40.01, -75.01, 2, 0), (42.01, -78.02, 5, 0)]

        def fake_detect_spikes(cursor):
            return [(34.08, -117.18, 6, 4), (41.0, -76.0, 1, 0), (42.03, -78.04, 5, 0)]

        def fake_recurring_regions(cursor):
            return [(34.1, -117.2, 4, 9, 5), (39.0, -77.0, 1, 2, 0), (42.0, -78.0, 3, 5, 0)]

        output = io.StringIO()
        with patch.object(query, "coordinated_activity", side_effect=fake_coordinated_activity, create=True):
            with patch.object(query, "detect_spikes", side_effect=fake_detect_spikes, create=True):
                with patch.object(query, "recurring_regions", side_effect=fake_recurring_regions):
                    with redirect_stdout(output):
                        query.rank_regions(None)

        lines = [line.strip() for line in output.getvalue().splitlines() if line.strip()]

        self.assertEqual(lines[0], "--- PRIORITY REGIONS ---")
        self.assertIn("(34.1, -117.2, 34.8, 9, 5, 1, 4)", lines)
        self.assertIn("(42.0, -78.0, 5.0, 5, 0, 1, 3)", lines)
        self.assertNotIn("(39.0, -77.0, 2.5, 2, 0, 0, 1)", lines)
        self.assertNotIn("(40.0, -75.0, 3.0, 2, 0, 0, 0)", lines)
        self.assertNotIn("(41.0, -76.0, 2.1, 1, 0, 1, 0)", lines)


class DetectNewEntriesTests(RecurringRegionsTests):
    def test_detect_new_entries_finds_novel_aircraft_by_region(self):
        now = 100_000

        recent_rows = [
            ("alpha1", 35.11, -118.11, now - 300, "US_CARGO", 6),
            ("alpha2", 35.12, -118.09, now - 600, "UNKNOWN", 4),
            ("bravo1", 36.11, -119.11, now - 400, "MIL_PATROL", 7),
            ("bravo2", 36.14, -119.12, now - 500, "UK_CARGO", 6),
            ("bravo3", 36.13, -119.14, now - 700, "UNKNOWN", 4),
            ("charlie1", 37.11, -120.11, now - 200, "UNKNOWN", 5),
            ("charlie2", 37.12, -120.12, now - 250, "UNKNOWN", 5),
        ]

        baseline_rows = [
            ("old1", 35.10, -118.10, now - 10_000, "UNKNOWN", 3),
            ("charlie1", 37.11, -120.11, now - 10_200, "UNKNOWN", 3),
            ("charlie2", 37.12, -120.12, now - 10_300, "UNKNOWN", 3),
            ("alpha1", 34.10, -117.10, now - 10_100, "US_CARGO", 3),
        ]

        for row in recent_rows + baseline_rows:
            self.insert_position(*row)

        self.connection.commit()

        output = io.StringIO()
        with patch.object(query.time, "time", return_value=now):
            with redirect_stdout(output):
                query.detect_new_entries(self.cursor)

        lines = [line.strip() for line in output.getvalue().splitlines() if line.strip()]

        self.assertEqual(lines[0], "--- New Aircraft Activity (Last 30 Minutes) ---")
        self.assertEqual(lines[1], "(36.1, -119.1, 3, 2)")
        self.assertIn("(35.1, -118.1, 2, 1)", lines)
        self.assertNotIn("(37.1, -120.1, 2, 0)", lines)


class DetectMovementsTests(RecurringRegionsTests):
    def test_detect_movements_aggregates_cross_region_routes(self):
        now = 200_000

        movement_rows = [
            ("move1", 34.11, -117.11, now - 5000, "US_CARGO", 5),
            ("move1", 35.12, -118.12, now - 200, "US_CARGO", 5),
            ("move2", 34.14, -117.14, now - 4900, "MIL_PATROL", 6),
            ("move2", 35.10, -118.10, now - 100, "MIL_PATROL", 6),
            ("move3", 36.11, -119.11, now - 4700, "UNKNOWN", 4),
            ("move3", 37.12, -120.12, now - 120, "UNKNOWN", 4),
            ("move4", 36.10, -119.14, now - 4600, "UNKNOWN", 4),
            ("move4", 37.11, -120.10, now - 110, "UNKNOWN", 4),
            ("tiny1", 39.11, -122.11, now - 1300, "UNKNOWN", 4),
            ("tiny1", 39.31, -122.11, now - 90, "UNKNOWN", 4),
            ("tiny2", 39.12, -122.14, now - 1200, "UNKNOWN", 4),
            ("tiny2", 39.30, -122.10, now - 80, "UNKNOWN", 4),
            ("mid1", 40.11, -123.11, now - 4400, "UNKNOWN", 4),
            ("mid1", 40.51, -123.11, now - 70, "UNKNOWN", 4),
            ("mid2", 40.12, -123.14, now - 4300, "UNKNOWN", 4),
            ("mid2", 40.50, -123.10, now - 60, "UNKNOWN", 4),
            ("stay1", 38.11, -121.11, now - 1000, "UNKNOWN", 3),
            ("stay1", 38.12, -121.12, now - 100, "UNKNOWN", 3),
            ("oldmove", 34.11, -117.11, now - 8000, "US_CARGO", 5),
            ("oldmove", 35.12, -118.12, now - 7500, "US_CARGO", 5),
        ]

        for row in movement_rows:
            self.insert_position(*row)

        self.connection.commit()

        output = io.StringIO()
        with patch.object(query.time, "time", return_value=now):
            with redirect_stdout(output):
                query.detect_movements(self.cursor)

        lines = [line.strip() for line in output.getvalue().splitlines() if line.strip()]

        self.assertEqual(lines[0], "--- Cross-Region Movements (Last 2 Hours) ---")
        self.assertEqual(lines[1], "(34.1, -117.1, 35.1, -118.1, 2, 2, 1.4)")
        self.assertIn("(36.1, -119.1, 37.1, -120.1, 2, 0, 1.4)", lines)
        self.assertNotIn("(39.1, -122.1, 39.3, -122.1, 2, 0, 0.2)", lines)
        self.assertNotIn("(40.1, -123.1, 40.5, -123.1, 2, 0, 0.4)", lines)
        self.assertNotIn("(38.1, -121.1, 38.1, -121.1, 1, 0, 0.0)", lines)


class DetectStagingAndProjectionTests(RecurringRegionsTests):
    def setUp(self):
        super().setUp()
        self.cursor.execute("""
            CREATE TABLE aircraft_tracks (
                icao24 TEXT,
                first_seen REAL,
                last_seen REAL,
                start_lat REAL,
                start_lon REAL,
                end_lat REAL,
                end_lon REAL,
                max_distance REAL,
                type TEXT,
                PRIMARY KEY (icao24)
            )
        """)

    def insert_track(self, icao24, first_seen, last_seen, start_lat, start_lon, end_lat, end_lon, max_distance, aircraft_type):
        self.cursor.execute("""
            INSERT INTO aircraft_tracks
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            icao24,
            first_seen,
            last_seen,
            start_lat,
            start_lon,
            end_lat,
            end_lon,
            max_distance,
            aircraft_type,
        ))

    def test_detect_staging_and_projection_reports_regions_and_flows(self):
        now = 300_000
        tracks = [
            ("t1", now - 20_000, now - 100, 34.11, -117.11, 35.11, -118.11, 1.4, "US_CARGO"),
            ("t2", now - 19_500, now - 120, 34.12, -117.14, 35.13, -118.12, 1.4, "MIL_PATROL"),
            ("t3", now - 19_000, now - 200, 34.09, -117.10, 36.11, -119.11, 2.8, "MIL_PATROL"),
            ("t4", now - 18_500, now - 140, 33.11, -116.11, 35.12, -118.14, 2.3, "UNKNOWN"),
            ("t5", now - 18_000, now - 160, 33.12, -116.10, 35.14, -118.11, 2.2, "UK_CARGO"),
            ("t6", now - 17_500, now - 180, 33.10, -116.12, 37.11, -120.11, 5.6, "UNKNOWN"),
            ("old1", now - 30_000, now - 25_000, 34.11, -117.11, 35.11, -118.11, 1.4, "US_CARGO"),
            ("small1", now - 5_000, now - 100, 34.11, -117.11, 34.31, -117.11, 0.4, "US_CARGO"),
        ]

        for track in tracks:
            self.insert_track(*track)

        self.connection.commit()

        output = io.StringIO()
        with patch.object(query.time, "time", return_value=now):
            with redirect_stdout(output):
                query.detect_staging_and_projection(self.cursor)

        lines = [line.strip() for line in output.getvalue().splitlines() if line.strip()]

        self.assertIn("--- Staging Regions (Last 6 Hours) ---", lines)
        self.assertIn("(34.1, -117.1, 3, 3)", lines)
        self.assertIn("(33.1, -116.1, 3, 1)", lines)
        self.assertIn("--- Projection Regions (Last 6 Hours) ---", lines)
        self.assertIn("(35.1, -118.1, 4, 3)", lines)
        self.assertIn("--- Major Flow Routes (Last 6 Hours) ---", lines)
        self.assertIn("(34.1, -117.1, 35.1, -118.1, 2, 2, 7.0)", lines)
        self.assertIn("(33.1, -116.1, 35.1, -118.1, 2, 1, 4.5)", lines)
        self.assertNotIn("(34.1, -117.1, 34.3, -117.1, 1, 1, 3.5)", lines)


class DetectActivityChangesTests(RecurringRegionsTests):
    def test_detect_activity_changes_reports_surge_buildup_and_emerging_regions(self):
        # Time windows (now=600_000):
        #   recent : timestamp >= 598_200
        #   prior  : 594_600 <= timestamp < 598_200
        now = 600_000

        # --- Growth region: prior 5 UNKNOWN, recent adds 6 US_CARGO ---
        # growth_score = sqrt(6) + 6*2.5 + (24/11)*1.5 + 2.0 persistence = 2.449+15.0+3.272+2.0 = 22.7
        for i in range(5):
            self.insert_position(f"prior-{i}", 35.01, -120.01, now - 4_000, "UNKNOWN", 3)
        for i in range(5):
            self.insert_position(f"rec-u-{i}", 35.01, -120.01, now - 500, "UNKNOWN", 3)
        for i in range(6):
            self.insert_position(f"rec-m-{i}", 35.01, -120.01, now - 300, "US_CARGO", 7)

        # --- Emerging military region: 36 aircraft (5 military), no prior ---
        # emergence_score = min(sqrt(36), 6) + 5*2.5 = 6.0 + 12.5 = 18.5
        for i in range(31):
            self.insert_position(f"emerg-u-{i}", 48.01, 2.01, now - 300, "UNKNOWN", 5)
        for i in range(5):
            self.insert_position(f"emerg-m-{i}", 48.01, 2.01, now - 200, "US_CARGO", 5)

        # --- Small region: 4 UNKNOWN, no prior → emergence_score = sqrt(4) = 2.0 < 15 ---
        for i in range(4):
            self.insert_position(f"small-{i}", 42.01, -80.01, now - 300, "UNKNOWN", 3)

        self.connection.commit()

        output = io.StringIO()
        with patch.object(query.time, "time", return_value=now):
            with redirect_stdout(output):
                query.detect_activity_changes(self.cursor)

        lines = [line.strip() for line in output.getvalue().splitlines()
                 if line.strip() and not line.strip().startswith("[TIMER]")]

        self.assertEqual(lines[0], "--- Activity Changes (Last 90 Minutes) ---")
        # Emerging region ranks first: 6.0+12.5+5.0+3.0 military_bonus=26.5 > growth 25.7; both HIGH
        self.assertEqual(lines[1], "(48.0, 2.0, 26.5, 36, 5, 5.0, 'SURGE,MILITARY_BUILDUP,EMERGING_REGION,ESCALATION', 'HIGH', ['MILITARY_BUILDUP', 'EMERGING_REGION', 'HIGH_SCORE'], 'SHORT', 'b8afacf0', 0.0, 13.9, 0.0, 100.0, 'MIXED')")
        # Growth region second: 22.7+3.0 military_bonus=25.7; prior matched but no 6hr data → MEDIUM
        self.assertIn("(35.0, -120.0, 25.7, 6, 6, 2.2, 'SURGE,MILITARY_BUILDUP,ESCALATION', 'HIGH', ['MILITARY_BUILDUP', 'PERSISTENT_ACTIVITY', 'HIGH_SCORE'], 'MEDIUM', '5af8d654', 0.0, 54.5, 45.5, 54.5, 'MILITARY_HEAVY')", lines)
        # Small UNKNOWN-only region is filtered (score 2.0 < threshold 15)
        self.assertFalse(any("42." in l for l in lines), "Small UNKNOWN region must be filtered out")


class DetectActivityChangesCapsTests(RecurringRegionsTests):
    def test_civilian_surge_bonus_scales_and_military_ranks_higher(self):
        """Civilian bonus is capped at 1.5 (scaled * 0.5) so civilian-only regions
        never reach HIGH. 100-aircraft flood at score=8 scores 6+0+8+1.5=15.5 (LOW),
        while a 36-aircraft military emergence scores 23.5 (HIGH) and ranks above it."""
        now = 500_000

        # 100 UNKNOWN at score=8 — avg_score=8.0 meets guard>=6.0; bonus=min(1.5,√100*0.5)=1.5; total=15.5
        for i in range(100):
            self.insert_position(f"flood-{i}", 50.01, 10.01, now - 300, "UNKNOWN", 8)

        # 36 aircraft, 5 military — emergence = 6+12.5+5 = 23.5, no civilian bonus
        for i in range(31):
            self.insert_position(f"mil-u-{i}", 51.01, 11.01, now - 200, "UNKNOWN", 5)
        for i in range(5):
            self.insert_position(f"mil-m-{i}", 51.01, 11.01, now - 200, "US_CARGO", 5)

        self.connection.commit()

        output = io.StringIO()
        with patch.object(query.time, "time", return_value=now):
            with redirect_stdout(output):
                query.detect_activity_changes(self.cursor)

        result_lines = [
            l.strip() for l in output.getvalue().splitlines()
            if l.strip().startswith("(")
        ]

        mil_lines      = [l for l in result_lines if "51." in l]
        summary_lines  = [l for l in result_lines if "'LOW_ACTIVITY_SUMMARY'" in l]

        # Civilian flood is LOW — compressed into summary, not an individual row
        self.assertEqual(len(summary_lines), 1)
        summary = ast.literal_eval(summary_lines[0])
        self.assertEqual(summary[1], 1)      # 1 LOW region
        self.assertEqual(summary[2], 15.5)   # min score
        self.assertEqual(summary[3], 15.5)   # max score

        # Military emergence is HIGH, printed individually
        self.assertEqual(len(mil_lines), 1)
        mil_row = ast.literal_eval(mil_lines[0])
        self.assertEqual(mil_row[2], 26.5)
        self.assertIn("EMERGING_REGION", mil_row[6])
        self.assertEqual(mil_row[7], "HIGH")

        # HIGH row appears before LOW summary
        self.assertLess(result_lines.index(mil_lines[0]), result_lines.index(summary_lines[0]))


class EdgeCaseTests(RecurringRegionsTests):
    def _run(self, now):
        output = io.StringIO()
        with patch.object(query.time, "time", return_value=now):
            with redirect_stdout(output):
                query.detect_activity_changes(self.cursor)
        return [l.strip() for l in output.getvalue().splitlines() if l.strip().startswith("(")]

    def test_large_civilian_flood_no_military_is_low_never_high(self):
        # 120 UNKNOWN, score=8, no prior: 6+0+8+min(1.5,√120*0.5)=15.5, LOW
        # Civilian-only emergence cannot exceed 15.5 — never reaches MEDIUM or HIGH.
        now = 500_000
        for i in range(120):
            self.insert_position(f"flood-{i}", 40.01, -75.01, now - 300, "UNKNOWN", 8)
        self.connection.commit()

        lines = self._run(now)
        # LOW region compressed into summary — no individual row printed
        self.assertEqual(len(lines), 1)
        summary = ast.literal_eval(lines[0])
        self.assertEqual(summary[0], "LOW_ACTIVITY_SUMMARY")
        self.assertEqual(summary[1], 1)
        self.assertEqual(summary[2], 15.5)   # min score
        self.assertEqual(summary[3], 15.5)   # max score (same; only one LOW)
        self.assertNotIn("HIGH", lines[0])

    def test_small_military_buildup_is_medium(self):
        # 4 US_CARGO, score=5, no prior: √4+4*2.5+5=17.0, MEDIUM
        now = 500_000
        for i in range(4):
            self.insert_position(f"mil-{i}", 55.01, 25.01, now - 300, "US_CARGO", 5)
        self.connection.commit()

        lines = self._run(now)
        self.assertEqual(len(lines), 1)
        row = ast.literal_eval(lines[0])
        self.assertEqual(row[7], "MEDIUM")
        self.assertIn("MILITARY_BUILDUP", row[8])

    def test_persistent_low_activity_is_filtered(self):
        # prior 3 UNKNOWN → recent 4 UNKNOWN, same score: aircraft_delta=1, score_delta=0
        # total = √1+0+0+2.0 persistence = 3.0 < threshold
        now = 500_000
        for i in range(3):
            self.insert_position(f"prior-{i}", 45.01, 10.01, now - 4_000, "UNKNOWN", 3)
        for i in range(4):
            self.insert_position(f"rec-{i}", 45.01, 10.01, now - 300, "UNKNOWN", 3)
        self.connection.commit()

        lines = self._run(now)
        self.assertFalse(any("45." in l for l in lines))

    def test_emerging_high_score_region_is_high(self):
        # 8 US_CARGO, score=8, no prior: √8+8*2.5+8=30.8, HIGH; HIGH_SCORE tag (>=25)
        now = 500_000
        for i in range(8):
            self.insert_position(f"emerg-{i}", 60.01, 30.01, now - 300, "US_CARGO", 8)
        self.connection.commit()

        lines = self._run(now)
        self.assertEqual(len(lines), 1)
        row = ast.literal_eval(lines[0])
        self.assertEqual(row[7], "HIGH")
        self.assertIn("HIGH_SCORE", row[8])
        self.assertIn("EMERGING_REGION", row[8])

    def test_random_noise_is_filtered(self):
        # 2 UNKNOWN, score=2, no prior: √2+0+2=3.4 < threshold
        now = 500_000
        for i in range(2):
            self.insert_position(f"noise-{i}", 33.01, -90.01, now - 300, "UNKNOWN", 2)
        self.connection.commit()

        lines = self._run(now)
        self.assertFalse(any("33." in l for l in lines))

    def test_results_sorted_descending_and_stable_on_equal_score(self):
        # Region A (61.): 6 US_CARGO score=5, no prior → sqrt(6)+6*2.5+5 = 22.4, HIGH
        # Region D (62.): 4 US_CARGO + 5 UNKNOWN score=4, no prior → 3+10+4 = 17.0, 9 aircraft
        # Region E (63.): 4 US_CARGO score=5, no prior → 2+10+5 = 17.0, 4 aircraft
        # D and E tie at 17.0; D inserted first (9 > 4 aircraft) so stable sort keeps D before E.
        now = 500_000

        for i in range(6):
            self.insert_position(f"a-{i}", 61.01, 20.01, now - 300, "US_CARGO", 5)

        for i in range(4):
            self.insert_position(f"d-m-{i}", 62.01, 20.01, now - 300, "US_CARGO", 4)
        for i in range(5):
            self.insert_position(f"d-u-{i}", 62.01, 20.01, now - 300, "UNKNOWN", 4)

        for i in range(4):
            self.insert_position(f"e-{i}", 63.01, 20.01, now - 300, "US_CARGO", 5)

        self.connection.commit()

        lines = self._run(now)

        a_lines = [l for l in lines if "61." in l]
        d_lines = [l for l in lines if "62." in l]
        e_lines = [l for l in lines if "63." in l]

        self.assertEqual(len(a_lines), 1)
        self.assertEqual(len(d_lines), 1)
        self.assertEqual(len(e_lines), 1)

        a_row = ast.literal_eval(a_lines[0])
        d_row = ast.literal_eval(d_lines[0])
        e_row = ast.literal_eval(e_lines[0])

        self.assertGreater(a_row[2], d_row[2], "A must rank above D")
        self.assertEqual(d_row[2], e_row[2], "D and E must have equal scores")

        a_idx = lines.index(a_lines[0])
        d_idx = lines.index(d_lines[0])
        e_idx = lines.index(e_lines[0])
        self.assertLess(a_idx, d_idx, "A must appear before D")
        self.assertLess(d_idx, e_idx, "D must appear before E on equal score (stable sort)")

    def test_threshold_boundary_inclusion_exclusion(self):
        # Below (12.5):  prior 6 UNKNOWN score=0, recent 6 UNKNOWN score=7
        #   → aircraft_delta=0, score_delta=7: 0+0+7*1.5+2.0=12.5 < 13, filtered
        # Low (15.0):    prior 5 UNKNOWN score=0, recent 6 UNKNOWN score=8
        #   → aircraft_delta=1, score_delta=8: 1+0+12+2=15.0, included, LOW
        # Medium (17.0): no prior, 4 US_CARGO score=5
        #   → sqrt(4)+4*2.5+5=17.0, included, MEDIUM
        # Anchor (17.0): no prior, 4 US_CARGO score=5 at 74.0 — keeps total >= 3
        #   so fill logic never pulls in the 12.5 region
        now = 500_000

        for i in range(6):
            self.insert_position(f"bel-p-{i}", 70.01, 20.01, now - 4_000, "UNKNOWN", 0)
        for i in range(6):
            self.insert_position(f"bel-r-{i}", 70.01, 20.01, now - 300, "UNKNOWN", 7)

        for i in range(5):
            self.insert_position(f"at-p-{i}", 71.01, 20.01, now - 4_000, "UNKNOWN", 0)
        for i in range(6):
            self.insert_position(f"at-r-{i}", 71.01, 20.01, now - 300, "UNKNOWN", 8)

        for i in range(4):
            self.insert_position(f"abv-{i}", 72.01, 20.01, now - 300, "US_CARGO", 5)

        for i in range(4):
            self.insert_position(f"anc-{i}", 74.01, 20.01, now - 300, "US_CARGO", 5)

        self.connection.commit()

        lines = self._run(now)

        below_lines   = [l for l in lines if "70." in l]
        summary_lines = [l for l in lines if "'LOW_ACTIVITY_SUMMARY'" in l]
        above_lines   = [l for l in lines if "72." in l]

        self.assertEqual(len(below_lines), 0, "score 12.5 must be filtered")

        # low region (15.0) compressed into summary
        self.assertEqual(len(summary_lines), 1)
        summary = ast.literal_eval(summary_lines[0])
        self.assertEqual(summary[1], 1)      # 1 LOW region
        self.assertEqual(summary[2], 15.0)   # min score
        self.assertEqual(summary[3], 15.0)   # max score

        self.assertEqual(len(above_lines), 1)
        above_row = ast.literal_eval(above_lines[0])
        self.assertEqual(above_row[2], 17.0)
        self.assertEqual(above_row[7], "MEDIUM")


if __name__ == "__main__":
    unittest.main()
