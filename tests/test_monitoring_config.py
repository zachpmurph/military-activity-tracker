import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core import monitoring_config


class TheaterForPointTests(unittest.TestCase):
    def test_theater_for_point_prefers_previous_theater_on_small_boundary_shift(self):
        theaters = [
            {
                "theater_id": "alpha",
                "label": "Alpha",
                "center_lat": 0.0,
                "center_lon": 0.0,
                "radius_km": 30.0,
            },
            {
                "theater_id": "bravo",
                "label": "Bravo",
                "center_lat": 0.3,
                "center_lon": 0.0,
                "radius_km": 30.0,
            },
        ]

        with patch.object(monitoring_config, "THEATERS", theaters):
            self.assertEqual(monitoring_config.theater_for_point(0.2, 0.0), "bravo")
            self.assertEqual(
                monitoring_config.theater_for_point(
                    0.2,
                    0.0,
                    previous_theater_id="alpha",
                    previous_lat=0.1,
                    previous_lon=0.0,
                ),
                "alpha",
            )


if __name__ == "__main__":
    unittest.main()
