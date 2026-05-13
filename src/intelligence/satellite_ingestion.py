"""
intelligence/satellite_ingestion.py
─────────────────────────────────────
Satellite-based (synthetic) change detection pipeline.

Produces SATELLITE ExternalSignal objects that drive change_score and
novelty features in the classification pipeline.  Signal intensity
represents the magnitude of unexpected aircraft activity increase in a
grid cell relative to a recent baseline.

Data source
───────────
  Primary   The existing aircraft.db SQLite database is queried for count
            deltas per 1° grid cell between two time windows:
              current window  — last 30 min
              baseline window — 2–6 h ago
            A meaningful positive delta represents activity appearing where
            there was little or none before, analogous to what satellite
            SAR or optical change detection observes on the ground.

  Fallback  src/data/satellite_fallback.json is used when the database is
            absent, unavailable, or produces no change signals.

Change metric
─────────────
  change_ratio  = (current_count − baseline_count) / max(baseline_count, 1)
  intensity     = min(change_ratio / _MAX_CHANGE_RATIO, 1.0)

  _MAX_CHANGE_RATIO  = 3.0  (3× increase or brand-new region → intensity 1.0)
  _MIN_CURRENT_COUNT = 3    (suppress single-aircraft noise)

  Only positive deltas produce signals; decreases are ignored.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

from core.db import resolve_db_path
from intelligence.external_features import ExternalSignal, SignalType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DB_PATH       = resolve_db_path()
_FALLBACK_JSON = Path(__file__).resolve().parent.parent / "data" / "satellite_fallback.json"

_CURRENT_WINDOW_SECS = 30 * 60    # 30 minutes  — "now" snapshot
_BASELINE_START_SECS = 6 * 3600   # 6 hours ago — start of baseline window
_BASELINE_END_SECS   = 2 * 3600   # 2 hours ago — end of baseline window
_MIN_CURRENT_COUNT   = 3          # minimum aircraft to avoid noise-floor signals
_MAX_CHANGE_RATIO    = 3.0        # ratio at which intensity saturates to 1.0

# Module-level cache: populated on the first no-arg call to fetch_satellite_signals.
_SIGNAL_CACHE: Optional[List[ExternalSignal]] = None


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _normalize_change(change_ratio: float, max_ratio: float = _MAX_CHANGE_RATIO) -> float:
    """
    Normalise a change ratio to [0, 1].

    Negative or zero ratios return 0.0 — only positive change produces a signal.
    """
    if change_ratio <= 0.0:
        return 0.0
    return min(change_ratio / max_ratio, 1.0)


# ---------------------------------------------------------------------------
# Synthetic generation from aircraft database
# ---------------------------------------------------------------------------

def _generate_from_db(db_path: Optional[Path] = None) -> List[ExternalSignal]:
    """
    Query the aircraft position database for per-cell activity deltas.

    Aircraft positions are aggregated into 1° grid cells (via
    ROUND(lat_bin) / ROUND(lon_bin)).  For each cell whose current-window
    count is ≥ _MIN_CURRENT_COUNT, the count delta versus the baseline
    window is converted to a normalised intensity and wrapped in a SATELLITE
    ExternalSignal.

    Returns an empty list when:
      - the database does not exist
      - the database schema is missing lat_bin/lon_bin columns
      - no cells exceed _MIN_CURRENT_COUNT in the current window
      - all deltas are zero or negative (stable or declining regions)
      - any database error occurs

    Parameters
    ──────────
    db_path  Path to the SQLite database.  Defaults to _DB_PATH.
    """
    path = db_path or _DB_PATH
    if not path.exists():
        return []

    try:
        conn   = sqlite3.connect(str(path))
        cursor = conn.cursor()
        now    = time.time()

        current_start  = now - _CURRENT_WINDOW_SECS
        baseline_start = now - _BASELINE_START_SECS
        baseline_end   = now - _BASELINE_END_SECS

        # Current counts — 1° bins, above noise floor only
        cursor.execute("""
            SELECT ROUND(lat_bin) AS lat1, ROUND(lon_bin) AS lon1, COUNT(*) AS cnt
            FROM aircraft_positions
            WHERE timestamp > ?
              AND lat_bin IS NOT NULL
            GROUP BY lat1, lon1
            HAVING cnt >= ?
        """, (current_start, _MIN_CURRENT_COUNT))
        current_counts: Dict[tuple, int] = {
            (row[0], row[1]): row[2] for row in cursor.fetchall()
        }

        # Baseline counts — comparison window
        cursor.execute("""
            SELECT ROUND(lat_bin) AS lat1, ROUND(lon_bin) AS lon1, COUNT(*) AS cnt
            FROM aircraft_positions
            WHERE timestamp BETWEEN ? AND ?
              AND lat_bin IS NOT NULL
            GROUP BY lat1, lon1
        """, (baseline_start, baseline_end))
        baseline_counts: Dict[tuple, int] = {
            (row[0], row[1]): row[2] for row in cursor.fetchall()
        }

        conn.close()

        signals: List[ExternalSignal] = []
        for (lat, lon), current in current_counts.items():
            baseline     = baseline_counts.get((lat, lon), 0)
            change_ratio = (current - baseline) / max(baseline, 1)
            intensity    = _normalize_change(change_ratio)
            if intensity <= 0.0:
                continue
            signals.append(ExternalSignal(
                signal_type = SignalType.SATELLITE,
                lat         = float(lat),
                lon         = float(lon),
                intensity   = intensity,
                metadata    = {
                    "source":        "satellite",
                    "change_metric": round(change_ratio, 3),
                },
            ))
        return signals

    except Exception:
        return []


# ---------------------------------------------------------------------------
# Static JSON fallback
# ---------------------------------------------------------------------------

def _parse_satellite_json(items: list) -> List[ExternalSignal]:
    """
    Parse a list of satellite fallback JSON records into ExternalSignal objects.

    Each record must supply "lat", "lon", and "change_metric".
    Records with missing, null, or out-of-range coordinates are skipped.
    Negative or zero change_metric values produce no signal (skipped).
    """
    signals: List[ExternalSignal] = []
    for item in items:
        try:
            lat           = float(item["lat"])
            lon           = float(item["lon"])
            change_metric = float(item["change_metric"])
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue
            intensity = _normalize_change(change_metric)
            if intensity <= 0.0:
                continue
            signals.append(ExternalSignal(
                signal_type = SignalType.SATELLITE,
                lat         = lat,
                lon         = lon,
                intensity   = intensity,
                metadata    = {
                    "source":        "satellite",
                    "change_metric": round(change_metric, 3),
                },
            ))
        except (KeyError, ValueError, TypeError):
            continue
    return signals


def _load_fallback_signals() -> List[ExternalSignal]:
    """
    Load SATELLITE ExternalSignal objects from the repo-local static JSON file.

    Returns an empty list on any I/O or parse error.
    """
    try:
        with open(_FALLBACK_JSON, encoding="utf-8") as fh:
            items = json.load(fh)
        return _parse_satellite_json(items)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_satellite_signals(db_path: Optional[Path] = None) -> List[ExternalSignal]:
    """
    Return SATELLITE ExternalSignal objects representing activity change events.

    Fetch priority
    ──────────────
    When *db_path* is provided the cache is bypassed and the given database
    is queried directly (intended for testing / custom paths).

    When *db_path* is None (normal production use):
      1. Returns the module-level cache if already populated.
      2. Generates signals from the aircraft.db database by computing
         per-cell count deltas (current 30 min vs. baseline 2–6 h ago).
      3. Falls back to src/data/satellite_fallback.json when the database
         is inaccessible or produces no positive-delta cells.

    Returns
    ───────
    List of ExternalSignal objects with signal_type=SATELLITE.
    All intensities are guaranteed to be in [0, 1].
    """
    global _SIGNAL_CACHE

    if db_path is not None:
        signals = _generate_from_db(db_path)
        if not signals:
            signals = _load_fallback_signals()
        return signals

    if _SIGNAL_CACHE is not None:
        return _SIGNAL_CACHE

    signals = _generate_from_db()
    if not signals:
        signals = _load_fallback_signals()

    _SIGNAL_CACHE = signals
    return _SIGNAL_CACHE
