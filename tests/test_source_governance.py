import sys
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intelligence.external_features import ExternalSignal, SignalType
from intelligence.source_governance import collect_operational_external_signals


class SourceGovernanceTests(unittest.TestCase):
    @patch("intelligence.source_governance.fetch_satellite_signals", return_value=[])
    @patch("intelligence.source_governance.fetch_notam_signals", return_value=[])
    @patch("intelligence.source_governance.fetch_maritime_signals", return_value=[])
    @patch("intelligence.source_governance.fetch_ais_signals", return_value=[])
    def test_operational_collection_excludes_notam_and_satellite(
        self,
        mock_ais,
        mock_maritime,
        mock_notam,
        mock_satellite,
    ):
        with patch.dict(
            "os.environ",
            {"AIS_API_KEY": "ais-key", "GFW_API_KEY": "gfw-key"},
            clear=False,
        ):
            collect_operational_external_signals()

        mock_ais.assert_called_once_with(bounding_boxes=ANY, allow_fallback=False)
        mock_maritime.assert_called_once_with(allow_fallback=False)
        mock_notam.assert_not_called()
        mock_satellite.assert_not_called()

    @patch("intelligence.source_governance.fetch_maritime_signals", return_value=[])
    @patch("intelligence.source_governance.fetch_ais_signals", return_value=[])
    def test_development_mode_allows_fallbacks(self, mock_ais, mock_maritime):
        collect_operational_external_signals(development_mode=True)

        mock_ais.assert_called_once_with(bounding_boxes=ANY, allow_fallback=True)
        mock_maritime.assert_called_once_with(allow_fallback=True)

    @patch(
        "intelligence.source_governance.fetch_maritime_signals",
        return_value=[ExternalSignal(SignalType.MARITIME, 10.0, 20.0, 0.7, {"source": "maritime"})],
    )
    @patch(
        "intelligence.source_governance.fetch_ais_signals",
        return_value=[ExternalSignal(SignalType.MARITIME, 11.0, 21.0, 0.8, {"source": "ais"})],
    )
    def test_collection_returns_signals_and_source_reports(self, mock_ais, mock_maritime):
        with patch.dict(
            "os.environ",
            {"AIS_API_KEY": "ais-key", "GFW_API_KEY": "gfw-key"},
            clear=False,
        ):
            signals, reports = collect_operational_external_signals()

        self.assertEqual(len(signals), 2)
        report_names = {report["source_name"] for report in reports}
        self.assertEqual(report_names, {"aisstream", "gfw"})
        self.assertTrue(all(report["source_tier"] in {"primary_live", "secondary_live"} for report in reports))


if __name__ == "__main__":
    unittest.main()
