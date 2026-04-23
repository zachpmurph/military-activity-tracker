import math
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_ingest_module():
    ingest_path = Path(__file__).resolve().parents[1] / "src" / "ingest.py"
    source = ingest_path.read_text()
    source = source.replace(
        "from sources.adsb import fetch as fetch_adsb\n",
        "def fetch_adsb():\n    return []\n",
    )
    source = source.replace(
        "from sources.opensky import fetch as fetch_opensky\n",
        "def fetch_opensky():\n    return []\n",
    )
    source = source.replace('sqlite3.connect("data/aircraft.db")', 'sqlite3.connect(":memory:")')
    source = source.split("# --- MAIN LOOP ---")[0]

    module = types.ModuleType("ingest_under_test")
    exec(source, module.__dict__)
    return module


class AircraftTracksTests(unittest.TestCase):
    def setUp(self):
        self.ingest = load_ingest_module()

    def tearDown(self):
        self.ingest.conn.close()

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
        self.ingest.cursor.execute("""
            INSERT INTO aircraft_tracks
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("stale1", 1, 2, 10.0, 10.0, 10.0, 10.0, 0.0))
        self.ingest.conn.commit()

        with patch.object(self.ingest.time, "time", return_value=100_000):
            self.ingest.store_aircraft([self.make_aircraft("abc123", 34.0, -117.0)])

        row = self.ingest.cursor.execute("""
            SELECT first_seen, last_seen, start_lat, start_lon, end_lat, end_lon, max_distance
            FROM aircraft_tracks
            WHERE icao24 = 'abc123'
        """).fetchone()

        self.assertEqual(row, (100_000, 100_000, 34.0, -117.0, 34.0, -117.0, 0.0))

        with patch.object(self.ingest.time, "time", return_value=100_600):
            self.ingest.store_aircraft([self.make_aircraft("abc123", 34.6, -116.2)])

        updated_row = self.ingest.cursor.execute("""
            SELECT first_seen, last_seen, start_lat, start_lon, end_lat, end_lon, max_distance
            FROM aircraft_tracks
            WHERE icao24 = 'abc123'
        """).fetchone()

        self.assertEqual(updated_row[:6], (100_000, 100_600, 34.0, -117.0, 34.6, -116.2))
        self.assertAlmostEqual(updated_row[6], math.sqrt((0.6 ** 2) + (0.8 ** 2)))

        stale_row = self.ingest.cursor.execute("""
            SELECT icao24
            FROM aircraft_tracks
            WHERE icao24 = 'stale1'
        """).fetchone()
        self.assertIsNone(stale_row)


if __name__ == "__main__":
    unittest.main()
