import math
import sqlite3
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ingest.persistence as persistence
import ingest.normalization as normalization


def setup_test_db():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS aircraft_positions (
            icao24 TEXT,
            callsign TEXT,
            lat REAL,
            lon REAL,
            altitude REAL,
            speed REAL,
            timestamp REAL,
            type TEXT,
            behavior TEXT,
            score INTEGER,
            lat_bin REAL,
            lon_bin REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS aircraft_tracks (
            icao24 TEXT,
            first_seen REAL,
            last_seen REAL,
            start_lat REAL,
            start_lon REAL,
            end_lat REAL,
            end_lon REAL,
            max_distance REAL,
            PRIMARY KEY (icao24)
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_time ON aircraft_positions (timestamp)")
    conn.commit()
    return conn, cursor


class AircraftTracksTests(unittest.TestCase):
    def setUp(self):
        conn, cursor = setup_test_db()
        persistence.conn = conn
        persistence.cursor = cursor
        normalization.history.clear()

    def tearDown(self):
        persistence.conn.close()

    def make_aircraft(self, icao24, lat, lon):
        return {
            "icao24": icao24,
            "callsign": "RCH123",
            "lat": lat,
            "lon": lon,
            "altitude": 30000,
            "velocity": 400,
            "source": "adsb",
        }

    def test_store_aircraft_updates_persistent_tracks(self):
        persistence.cursor.execute("""
            INSERT INTO aircraft_tracks
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("stale1", 1, 2, 10.0, 10.0, 10.0, 10.0, 0.0))
        persistence.conn.commit()

        with patch.object(persistence.time, "time", return_value=100_000):
            persistence.store_aircraft([self.make_aircraft("abc123", 34.0, -117.0)])

        row = persistence.cursor.execute("""
            SELECT first_seen, last_seen, start_lat, start_lon, end_lat, end_lon, max_distance
            FROM aircraft_tracks
            WHERE icao24 = 'abc123'
        """).fetchone()

        self.assertEqual(row, (100_000, 100_000, 34.0, -117.0, 34.0, -117.0, 0.0))

        with patch.object(persistence.time, "time", return_value=100_600):
            persistence.store_aircraft([self.make_aircraft("abc123", 34.6, -116.2)])

        updated_row = persistence.cursor.execute("""
            SELECT first_seen, last_seen, start_lat, start_lon, end_lat, end_lon, max_distance
            FROM aircraft_tracks
            WHERE icao24 = 'abc123'
        """).fetchone()

        self.assertEqual(updated_row[:6], (100_000, 100_600, 34.0, -117.0, 34.6, -116.2))
        self.assertAlmostEqual(updated_row[6], math.sqrt((0.6 ** 2) + (0.8 ** 2)))

        stale_row = persistence.cursor.execute("""
            SELECT icao24
            FROM aircraft_tracks
            WHERE icao24 = 'stale1'
        """).fetchone()
        self.assertIsNone(stale_row)


if __name__ == "__main__":
    unittest.main()
