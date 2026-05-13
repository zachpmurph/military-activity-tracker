"""
intelligence/runner.py
——————————————————————
Continuous intelligence classification loop.

Runs a full detection + external-signal + classification pass on a fixed
interval so classifications are updated automatically alongside the ingest
pipeline, without requiring manual invocation of query.py.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import resolve_db_path
from detection.clusters import (
    coordinated_activity,
    detect_new_entries,
    detect_spikes,
    recurring_regions,
)
from detection.changes import detect_activity_changes
from detection.flows import flow_routes_last_6h
from detection.movements import detect_movements
from detection.staging import detect_staging_and_projection
from intelligence.classifier import build_features, classify_regions
from intelligence.monitoring_backend import export_monitoring_snapshot, persist_monitoring_state
from intelligence.source_governance import collect_operational_external_signals
from query import (
    ENABLE_EXTERNAL_SIGNALS,
    _compute_external_summary,
    _migrate,
    _print_external_summary,
)


INTEL_INTERVAL_SEC: int = 300


def run_once(cursor: sqlite3.Cursor, db_path: Path) -> None:
    snapshot_time = time.time()

    _recurring = recurring_regions(cursor)
    _new_entries = detect_new_entries(cursor)
    _movements = detect_movements(cursor)
    _staging, _projection = detect_staging_and_projection(cursor)
    _flow_routes = flow_routes_last_6h(cursor)
    _coordinated = coordinated_activity(cursor)
    _spikes = detect_spikes(cursor)
    _changes = detect_activity_changes(cursor) or []

    if ENABLE_EXTERNAL_SIGNALS:
        _external, source_reports = collect_operational_external_signals(db_path=db_path)
    else:
        _external, source_reports = [], []

    features = build_features(
        coordinated=_coordinated,
        spikes=_spikes,
        recurring=_recurring,
        new_entries=_new_entries,
        movements=_movements,
        staging=_staging,
        projection=_projection,
        activity_changes=_changes,
        external_signals=_external or None,
    )
    intelligence = classify_regions(features)

    backend_summary = persist_monitoring_state(
        cursor,
        intelligence=intelligence,
        route_rows=_flow_routes,
        source_reports=source_reports,
        snapshot_time=snapshot_time,
    )
    cursor.connection.commit()
    export_paths = export_monitoring_snapshot(cursor, db_path.parent / "exports", snapshot_time)

    print("\n--- INTELLIGENCE ASSESSMENT ---")
    for region in intelligence[:10]:
        print(region)

    if ENABLE_EXTERNAL_SIGNALS:
        _print_external_summary(_compute_external_summary(_external, features))

    print("\n--- MONITORING BACKEND SUMMARY ---")
    print(
        {
            "region_count": backend_summary["region_count"],
            "route_count": backend_summary["route_count"],
            "alert_count": backend_summary["alert_count"],
            "exports": export_paths,
        }
    )


def main() -> None:
    db_path = resolve_db_path()
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
