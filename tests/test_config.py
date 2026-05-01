import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.config import classify_aircraft


class CallsignClassificationTests(unittest.TestCase):

    def _ac(self, callsign, icao24="123456"):
        return {"callsign": callsign, "icao24": icao24}

    def test_reach_is_military(self):
        self.assertEqual(classify_aircraft(self._ac("REACH123")), "MILITARY")

    def test_spar_is_military(self):
        self.assertEqual(classify_aircraft(self._ac("SPAR10")), "MILITARY")

    def test_venus_is_military(self):
        self.assertEqual(classify_aircraft(self._ac("VENUS01")), "MILITARY")

    def test_knife_is_military(self):
        self.assertEqual(classify_aircraft(self._ac("KNIFE11")), "MILITARY")

    def test_ascot_is_military(self):
        self.assertEqual(classify_aircraft(self._ac("ASCOT456")), "MILITARY")

    def test_tartan_is_military(self):
        self.assertEqual(classify_aircraft(self._ac("TARTAN5")), "MILITARY")

    def test_magma_is_military(self):
        self.assertEqual(classify_aircraft(self._ac("MAGMA7")), "MILITARY")

    def test_evac_is_military(self):
        self.assertEqual(classify_aircraft(self._ac("EVAC22")), "MILITARY")

    def test_rch_still_us_cargo(self):
        self.assertEqual(classify_aircraft(self._ac("RCH101")), "US_CARGO")

    def test_rrr_still_uk_cargo(self):
        self.assertEqual(classify_aircraft(self._ac("RRR55")), "UK_CARGO")

    def test_aal_still_civilian(self):
        self.assertEqual(classify_aircraft(self._ac("AAL123")), "CIVILIAN")

    def test_unknown_callsign_no_mil_hex_still_unknown(self):
        self.assertEqual(classify_aircraft(self._ac("XYZ999", "A12345")), "UNKNOWN")

    def test_lowercase_callsign_normalized(self):
        self.assertEqual(classify_aircraft(self._ac("reach123")), "MILITARY")


if __name__ == "__main__":
    unittest.main()
