"""
Tests for intelligence/runner.py
─────────────────────────────────
Covers:
  - run_once() executes a full detection → signal → classification pass
    without raising, using mocked detection and ingestion functions.
  - main() calls run_once() at least once before sleeping.
"""

import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import intelligence.runner as runner_mod
from intelligence.runner import run_once, INTEL_INTERVAL_SEC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_cursor():
    """Return a MagicMock that behaves like an sqlite3 cursor."""
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = None
    return cursor


def _make_fake_region():
    """Return a minimal object that looks like RegionIntelligence."""
    r = MagicMock()
    r.external_signal_count = 0
    return r


# ---------------------------------------------------------------------------
# run_once() smoke tests
# ---------------------------------------------------------------------------

class RunOnceTests(unittest.TestCase):

    @patch("intelligence.runner.classify_regions", return_value=[])
    @patch("intelligence.runner.build_features",    return_value=[])
    @patch("intelligence.runner.fetch_satellite_signals", return_value=[])
    @patch("intelligence.runner.fetch_maritime_signals",  return_value=[])
    @patch("intelligence.runner.fetch_notam_signals",     return_value=[])
    @patch("intelligence.runner.detect_activity_changes", return_value=[])
    @patch("intelligence.runner.detect_spikes",           return_value=[])
    @patch("intelligence.runner.coordinated_activity",    return_value=[])
    @patch("intelligence.runner.detect_staging_and_projection", return_value=([], []))
    @patch("intelligence.runner.detect_movements",        return_value=[])
    @patch("intelligence.runner.detect_new_entries",      return_value=[])
    @patch("intelligence.runner.recurring_regions",       return_value=[])
    def test_run_once_does_not_raise(self, *_mocks):
        """run_once() should complete without exception when all dependencies succeed."""
        cursor  = _empty_cursor()
        db_path = Path("/fake/aircraft.db")
        try:
            run_once(cursor, db_path)
        except Exception as exc:
            self.fail(f"run_once raised unexpectedly: {exc!r}")

    @patch("intelligence.runner.classify_regions", return_value=[])
    @patch("intelligence.runner.build_features",    return_value=[])
    @patch("intelligence.runner.fetch_satellite_signals", return_value=[])
    @patch("intelligence.runner.fetch_maritime_signals",  return_value=[])
    @patch("intelligence.runner.fetch_notam_signals",     return_value=[])
    @patch("intelligence.runner.detect_activity_changes", return_value=[])
    @patch("intelligence.runner.detect_spikes",           return_value=[])
    @patch("intelligence.runner.coordinated_activity",    return_value=[])
    @patch("intelligence.runner.detect_staging_and_projection", return_value=([], []))
    @patch("intelligence.runner.detect_movements",        return_value=[])
    @patch("intelligence.runner.detect_new_entries",      return_value=[])
    @patch("intelligence.runner.recurring_regions",       return_value=[])
    def test_run_once_calls_all_detection_functions(self, mock_recurring, mock_new,
                                                     mock_movements, mock_staging,
                                                     mock_coord, mock_spikes,
                                                     mock_changes, *_rest):
        cursor  = _empty_cursor()
        db_path = Path("/fake/aircraft.db")
        run_once(cursor, db_path)
        mock_recurring.assert_called_once()
        mock_new.assert_called_once()
        mock_movements.assert_called_once()
        mock_staging.assert_called_once()
        mock_coord.assert_called_once()
        mock_spikes.assert_called_once()
        mock_changes.assert_called_once()

    @patch("intelligence.runner.classify_regions", return_value=[])
    @patch("intelligence.runner.build_features",    return_value=[])
    @patch("intelligence.runner.fetch_satellite_signals", return_value=[])
    @patch("intelligence.runner.fetch_maritime_signals",  return_value=[])
    @patch("intelligence.runner.fetch_notam_signals",     return_value=[])
    @patch("intelligence.runner.detect_activity_changes", return_value=[])
    @patch("intelligence.runner.detect_spikes",           return_value=[])
    @patch("intelligence.runner.coordinated_activity",    return_value=[])
    @patch("intelligence.runner.detect_staging_and_projection", return_value=([], []))
    @patch("intelligence.runner.detect_movements",        return_value=[])
    @patch("intelligence.runner.detect_new_entries",      return_value=[])
    @patch("intelligence.runner.recurring_regions",       return_value=[])
    def test_run_once_calls_all_signal_fetchers(self, *_detection,
                                                 mock_changes=None, mock_spikes=None,
                                                 mock_coord=None, mock_staging=None,
                                                 mock_movements=None, mock_new=None,
                                                 mock_recurring=None):
        # Re-patch cleanly so we can inspect signal mocks
        with patch("intelligence.runner.fetch_notam_signals", return_value=[]) as mn, \
             patch("intelligence.runner.fetch_maritime_signals", return_value=[]) as mm, \
             patch("intelligence.runner.fetch_satellite_signals", return_value=[]) as ms:
            run_once(_empty_cursor(), Path("/fake/aircraft.db"))
            if runner_mod.ENABLE_EXTERNAL_SIGNALS:
                mn.assert_called_once()
                mm.assert_called_once()
                ms.assert_called_once()

    @patch("intelligence.runner.classify_regions", return_value=[])
    @patch("intelligence.runner.build_features",    return_value=[])
    @patch("intelligence.runner.fetch_satellite_signals", return_value=[])
    @patch("intelligence.runner.fetch_maritime_signals",  return_value=[])
    @patch("intelligence.runner.fetch_notam_signals",     return_value=[])
    @patch("intelligence.runner.detect_activity_changes", return_value=[])
    @patch("intelligence.runner.detect_spikes",           return_value=[])
    @patch("intelligence.runner.coordinated_activity",    return_value=[])
    @patch("intelligence.runner.detect_staging_and_projection", return_value=([], []))
    @patch("intelligence.runner.detect_movements",        return_value=[])
    @patch("intelligence.runner.detect_new_entries",      return_value=[])
    @patch("intelligence.runner.recurring_regions",       return_value=[])
    def test_run_once_passes_db_path_to_satellite(self, *_mocks):
        """fetch_satellite_signals receives the db_path argument."""
        db_path = Path("/my/aircraft.db")
        with patch("intelligence.runner.fetch_satellite_signals", return_value=[]) as ms:
            run_once(_empty_cursor(), db_path)
            if runner_mod.ENABLE_EXTERNAL_SIGNALS:
                ms.assert_called_once_with(db_path)

    @patch("intelligence.runner.classify_regions", return_value=[])
    @patch("intelligence.runner.build_features",    return_value=[])
    @patch("intelligence.runner.fetch_satellite_signals", return_value=[])
    @patch("intelligence.runner.fetch_maritime_signals",  return_value=[])
    @patch("intelligence.runner.fetch_notam_signals",     return_value=[])
    @patch("intelligence.runner.detect_activity_changes", return_value=None)
    @patch("intelligence.runner.detect_spikes",           return_value=[])
    @patch("intelligence.runner.coordinated_activity",    return_value=[])
    @patch("intelligence.runner.detect_staging_and_projection", return_value=([], []))
    @patch("intelligence.runner.detect_movements",        return_value=[])
    @patch("intelligence.runner.detect_new_entries",      return_value=[])
    @patch("intelligence.runner.recurring_regions",       return_value=[])
    def test_run_once_handles_none_activity_changes(self, *_mocks):
        """detect_activity_changes() may return None — run_once must not crash."""
        try:
            run_once(_empty_cursor(), Path("/fake/aircraft.db"))
        except Exception as exc:
            self.fail(f"run_once raised with None activity_changes: {exc!r}")


# ---------------------------------------------------------------------------
# main() loop test
# ---------------------------------------------------------------------------

class MainLoopTests(unittest.TestCase):

    @patch("intelligence.runner.time.sleep", side_effect=[None, KeyboardInterrupt])
    @patch("intelligence.runner.run_once")
    @patch("intelligence.runner._migrate")
    @patch("intelligence.runner.sqlite3.connect")
    def test_main_calls_run_once_before_sleeping(self, mock_connect, mock_migrate,
                                                   mock_run_once, mock_sleep):
        """main() must call run_once() at least once before sleeping."""
        mock_connect.return_value.__enter__ = MagicMock()
        mock_connect.return_value.cursor.return_value = _empty_cursor()
        try:
            runner_mod.main()
        except KeyboardInterrupt:
            pass
        mock_run_once.assert_called()

    @patch("intelligence.runner.time.sleep", side_effect=KeyboardInterrupt)
    @patch("intelligence.runner.run_once")
    @patch("intelligence.runner._migrate")
    @patch("intelligence.runner.sqlite3.connect")
    def test_main_sleeps_for_configured_interval(self, mock_connect, mock_migrate,
                                                   mock_run_once, mock_sleep):
        """main() sleeps for INTEL_INTERVAL_SEC between passes."""
        mock_connect.return_value.cursor.return_value = _empty_cursor()
        try:
            runner_mod.main()
        except KeyboardInterrupt:
            pass
        mock_sleep.assert_called_with(INTEL_INTERVAL_SEC)

    @patch("intelligence.runner.time.sleep", side_effect=[None, KeyboardInterrupt])
    @patch("intelligence.runner.run_once", side_effect=RuntimeError("db error"))
    @patch("intelligence.runner._migrate")
    @patch("intelligence.runner.sqlite3.connect")
    def test_main_continues_after_run_once_exception(self, mock_connect, mock_migrate,
                                                       mock_run_once, mock_sleep):
        """main() must not crash when run_once() raises — it catches and continues."""
        mock_connect.return_value.cursor.return_value = _empty_cursor()
        try:
            runner_mod.main()
        except KeyboardInterrupt:
            pass
        # If we reach here the exception was swallowed — test passes.
        self.assertGreaterEqual(mock_run_once.call_count, 1)


if __name__ == "__main__":
    unittest.main()
