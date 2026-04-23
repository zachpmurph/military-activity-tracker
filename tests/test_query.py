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
            print("\n--- Coordinated Activity Regions ---")
            print((34.12, -117.24, 4, 2))
            print((40.01, -75.01, 2, 0))
            print((42.01, -78.02, 5, 0))

        def fake_detect_spikes(cursor):
            print("\n--- Activity Spikes ---")
            print((34.08, -117.18, 6, 4))
            print((41.0, -76.0, 1, 0))
            print((42.03, -78.04, 5, 0))

        def fake_recurring_regions(cursor):
            print("\n--- Recurring Activity Regions (Last 6 Hours) ---")
            print((34.1, -117.2, 4, 9, 5))
            print((39.0, -77.0, 1, 2, 0))
            print((42.0, -78.0, 3, 5, 0))

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


if __name__ == "__main__":
    unittest.main()
