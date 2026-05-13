"""
intelligence/runner.py
──────────────────────
Continuous intelligence classification loop.

Runs a full detection + external-signal + classification pass on a fixed
interval so classifications are updated automatically alongside the ingest
pipeline, without requiring manual invocation of query.py.

Usage
─────
  # Terminal 1 — ingest
  python src/ingest/pipeline.py

  # Terminal 2 — intelligence (auto-classifies every 5 minutes)
  python src/intelligence/runner.py

Configuration
─────────────
  INTEL_INTERVAL_SEC   seconds between passes (default: 300)
  ENABLE_EXTERNAL_SIGNALS  imported from query.py; set False to skip signals
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

# Ensure src/ is on sys.path when this module is invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.clusters import (
    coordinated_activity,
    detect_new_entries,
    detect_spikes,
    recurring_regions,
)
from detection.changes import detect_activity_changes
from detection.movements import detect_movements
from detection.staging import detect_staging_and_projection
from intelligence.classifier import build_features, classify_regions
from intelligence.notam_ingestion import fetch_notam_signals
from intelligence.maritime_ingestion import fetch_maritime_signals
from intelligence.satellite_ingestion import fetch_satellite_signals
from intelligence.ais_ingestion import fetch_ais_signals
from query import (
    ENABLE_EXTERNAL_SIGNALS,
    _compute_external_summary,
    _migrate,
    _print_external_summary,
)


INTEL_INTERVAL_SEC: int = 300   # 5 minutes between passes


def run_once(cursor: sqlite3.Cursor, db_path: Path) -> None:
    """
    Execute one full intelligence pass and print the results.

    Mirrors the orchestration in query.py main() (lines 326-368) exactly,
    so both entry points produce identical output for the same DB state.
    """
    # Detection layer
    _recurring    = recurring_regions(cursor)
    _new_entries  = detect_new_entries(cursor)
    _movements    = detect_movements(cursor)
    _staging, _projection = detect_staging_and_projection(cursor)
    _coordinated  = coordinated_activity(cursor)
    _spikes       = detect_spikes(cursor)
    _changes      = detect_activity_changes(cursor) or []

    # External signals
    if ENABLE_EXTERNAL_SIGNALS:
        _notam_signals     = fetch_notam_signals()
        _maritime_signals  = fetch_maritime_signals()
        _satellite_signals = fetch_satellite_signals(db_path)
        _ais_signals       = fetch_ais_signals()
    else:
        _notam_signals = _maritime_signals = _satellite_signals = _ais_signals = []
    _external = _notam_signals + _maritime_signals + _satellite_signals + _ais_signals

    # Feature extraction + classification
    features = build_features(
        coordinated      = _coordinated,
        spikes           = _spikes,
        recurring        = _recurring,
        new_entries      = _new_entries,
        movements        = _movements,
        staging          = _staging,
        projection       = _projection,
        activity_changes = _changes,
        external_signals = _external or None,
    )
    intelligence = classify_regions(features)

    print("\n--- INTELLIGENCE ASSESSMENT ---")
    for region in intelligence[:10]:
        print(region)

    _print_external_summary(_compute_external_summary(_external, features))


def main() -> None:
    db_path = Path(__file__).resolve().parent.parent / "data" / "aircraft.db"
    conn = sqlite3.connect(db_path)
    _migrate(conn)
    cursor = conn.cursor()

    print(f"[runner] Intelligence loop started — interval {INTEL_INTERVAL_SEC}s")
    while True:
        try:
            run_once(cursor, db_path)
        except Exception as exc:
            print(f"[runner] Pass failed: {exc!r}")
        time.sleep(INTEL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
