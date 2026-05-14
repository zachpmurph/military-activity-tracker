import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intelligence.validation_cli import load_validation_report, summarize_validation_report, main


class ValidationCliTests(unittest.TestCase):
    def test_load_and_summarize_validation_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "latest_validation_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "summary": {"headline": "Validation warning: test headline"},
                        "recommended_actions": ["restore_primary_live_sources", "review_civilian_noise_thresholds"],
                    }
                ),
                encoding="utf-8",
            )

            payload = load_validation_report(report_path)
            summary = summarize_validation_report(payload)

            self.assertEqual(payload["summary"]["headline"], "Validation warning: test headline")
            self.assertIn("Validation warning: test headline", summary)
            self.assertIn("restore_primary_live_sources", summary)
            self.assertIn("review_civilian_noise_thresholds", summary)

    def test_main_uses_default_export_path_when_no_arg_is_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            default_path = Path(tmpdir) / "exports" / "latest_validation_report.json"
            default_path.parent.mkdir(parents=True, exist_ok=True)
            default_path.write_text(
                json.dumps({"summary": {"headline": "Validation pass: default path works"}}),
                encoding="utf-8",
            )

            with patch("intelligence.validation_cli.resolve_db_path", return_value=Path(tmpdir) / "aircraft.db"):
                with patch("builtins.print") as mock_print:
                    exit_code = main([])

            self.assertEqual(exit_code, 0)
            printed = "\n".join(" ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list)
            self.assertIn("Validation pass: default path works", printed)


if __name__ == "__main__":
    unittest.main()
