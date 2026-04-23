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


if __name__ == "__main__":
    unittest.main()
