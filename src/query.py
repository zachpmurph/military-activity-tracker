import sqlite3
import time
from pathlib import Path

from core.db import (
    _ensure_aircraft_position_columns,
    _ensure_aircraft_track_type,
    _ensure_monitoring_tables,
    _seed_theaters,
    resolve_db_path,
)
from detection.clusters import (
    coordinated_activity,
    detect_new_entries,
    detect_spikes,
    recurring_regions,
)
from detection.changes import detect_activity_changes, detect_linked_regions
from detection.movements import detect_movements
from detection.staging import detect_staging_and_projection
from intelligence.classifier import build_features, classify_regions
from intelligence.source_governance import collect_operational_external_signals


# ---------------------------------------------------------------------------
# Feature toggles
# ---------------------------------------------------------------------------
# Set ENABLE_EXTERNAL_SIGNALS = False to skip all external signal fetching
# (useful for A/B comparisons, profiling, or when a data source is suspect).
# When False, build_features receives no external_signals and behaves exactly
# as it did before the NOTAM pipeline was added.
ENABLE_EXTERNAL_SIGNALS: bool = True


def timed(fn):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = fn(*args, **kwargs)
        duration = time.time() - start
        print(f"[TIMER] {fn.__name__}: {duration:.3f}s")
        return result
    return wrapper

# ----------------------------
# Basic queries
# ----------------------------

@timed
def most_suspicious(cursor):
    cutoff = time.time() - 21600
    cursor.execute("""
    SELECT callsign, type, behavior, AVG(score) as avg_score, COUNT(*) as sightings
    FROM aircraft_positions
    WHERE timestamp > ?
    GROUP BY icao24
    ORDER BY avg_score DESC, sightings DESC
    LIMIT 10
    """, (cutoff,))
    return cursor.fetchall()


@timed
def recent_activity(cursor):
    cutoff = time.time() - 600

    cursor.execute("""
    SELECT callsign, type, behavior, COUNT(*) as sightings
    FROM aircraft_positions
    WHERE timestamp > ?
    GROUP BY icao24
    ORDER BY sightings DESC
    LIMIT 10
    """, (cutoff,))

    return cursor.fetchall()


@timed
def loitering(cursor):
    cutoff = time.time() - 21600
    cursor.execute("""
    SELECT callsign, COUNT(*) as sightings
    FROM aircraft_positions
    WHERE behavior = 'LOITERING' AND timestamp > ?
    GROUP BY icao24
    ORDER BY sightings DESC
    LIMIT 10
    """, (cutoff,))
    return cursor.fetchall()


@timed
def persistent_aircraft(cursor):
    print("\n--- Persistent Aircraft (Last 30 Minutes) ---")

    cutoff = time.time() - 1800

    cursor.execute("""
        SELECT
            icao24,
            callsign,
            type,
            COUNT(*) as sightings,
            AVG(score) as avg_score
        FROM aircraft_positions
        WHERE timestamp > ?
        AND callsign != ''
        AND (
            type IN ('US_CARGO', 'UK_CARGO', 'TANKER', 'MILITARY')
            OR score >= 6
            OR behavior = 'LOITERING'
        )
        GROUP BY icao24
        HAVING sightings >= 3
        ORDER BY avg_score DESC, sightings DESC
        LIMIT 10;
    """, (cutoff,))

    for row in cursor:
        print(row)


@timed
def military_cluster(cursor):
    print("\n--- Military Activity Cluster (Last 30 Minutes) ---")

    cutoff = time.time() - 1800

    cursor.execute("""
        SELECT
            TRIM(type) as type,
            COUNT(*) as total_sightings,
            COUNT(DISTINCT icao24) as unique_aircraft
        FROM aircraft_positions
        WHERE timestamp > ?
          AND type IN ('US_CARGO', 'UK_CARGO', 'MILITARY')
        GROUP BY type
        ORDER BY total_sightings DESC;
    """, (cutoff,))

    results = cursor.fetchall()

    if not results:
        print("No military clustering detected.")
        return

    for row in results:
        print(row)

# ----------------------------
# Region ranking (stays here so test patches on this module's globals work)
# ----------------------------

@timed
def rank_regions(cursor):
    print("\n--- PRIORITY REGIONS ---")

    regions = {}

    def get_region(lat, lon):
        key = (round(lat, 1), round(lon, 1))
        if key not in regions:
            regions[key] = {
                "aircraft_count": 0,
                "military_count": 0,
                "spike_flag": 0,
                "recurring_count": 0,
                "coordinated_flag": 0,
            }
        return regions[key]

    for lat, lon, aircraft_count, military_count in coordinated_activity(cursor):
        region = get_region(lat, lon)
        region["aircraft_count"] = max(region["aircraft_count"], aircraft_count)
        region["military_count"] = max(region["military_count"], military_count)
        region["coordinated_flag"] = 1

    for lat, lon, aircraft_count, military_count in detect_spikes(cursor):
        region = get_region(lat, lon)
        region["aircraft_count"] = max(region["aircraft_count"], aircraft_count)
        region["military_count"] = max(region["military_count"], military_count)
        region["spike_flag"] = 1

    for lat, lon, appearances, total_aircraft, military_presence in recurring_regions(cursor):
        region = get_region(lat, lon)
        region["aircraft_count"] = max(region["aircraft_count"], total_aircraft)
        region["military_count"] = max(region["military_count"], military_presence)
        region["recurring_count"] = max(region["recurring_count"], appearances)

    ranked_regions = []
    for (lat, lon), region in regions.items():
        score = (
            (region["military_count"] * 4)
            + (region["aircraft_count"] * 0.2)
            + (region["spike_flag"] * 5)
            + (region["recurring_count"] * 1.5)
            + (region["coordinated_flag"] * 2)
        )

        if region["military_count"] == 0:
            score = score * 0.4

        if region["military_count"] < 1 and region["spike_flag"] != 1:
            continue

        if score < 5:
            continue

        ranked_regions.append((
            lat,
            lon,
            round(score, 1),
            region["aircraft_count"],
            region["military_count"],
            region["spike_flag"],
            region["recurring_count"],
        ))

    ranked_regions.sort(key=lambda row: row[2], reverse=True)

    for row in ranked_regions[:10]:
        print(row)

# ----------------------------
# External signal summary
# ----------------------------

def _compute_external_summary(signals, features):
    """
    Return a dict of summary statistics for external signal influence.

    Pure function (no I/O) — suitable for unit testing.

    Keys
    ────
    total_signals_processed   int     – total ExternalSignal objects consumed
    regions_affected          int     – regions where external_signal_count > 0
    pct_regions_affected      float   – percentage (0–100, 1 d.p.)
    average_intensity         float   – mean signal intensity (0.0 when no signals)
    count_by_type             dict    – signal_type.value → count
    """
    total      = len(signals)
    affected   = sum(1 for f in features if f.external_signal_count > 0)
    n_regions  = max(len(features), 1)
    pct        = round((affected / n_regions) * 100, 1)
    avg_intens = round(sum(s.intensity for s in signals) / total, 3) if total else 0.0

    by_type: dict = {}
    for s in signals:
        key = s.signal_type.value
        by_type[key] = by_type.get(key, 0) + 1

    return {
        "total_signals_processed": total,
        "regions_affected":        affected,
        "pct_regions_affected":    pct,
        "average_intensity":       avg_intens,
        "count_by_type":           by_type,
    }


def _print_external_summary(summary):
    """Format and print the summary dict produced by _compute_external_summary."""
    print("\n--- EXTERNAL SIGNAL SUMMARY ---")
    print(f"  total_signals_processed : {summary['total_signals_processed']}")
    print(f"  regions_affected        : {summary['regions_affected']}")
    print(f"  % of regions affected   : {summary['pct_regions_affected']:.1f}%")
    print(f"  average_intensity       : {summary['average_intensity']:.3f}")
    print(f"  count by type           :")
    for stype, count in sorted(summary["count_by_type"].items()):
        print(f"    {stype:<12s}: {count}")


# ----------------------------
# Main
# ----------------------------

def _migrate(conn):
    """Ensure the monitoring-era schema exists on the active database.

    Safe to call on every startup: column adds are caught, tables are
    idempotent, and the theater seed upserts.
    """
    cur = conn.cursor()
    _ensure_aircraft_position_columns(cur)
    _ensure_aircraft_track_type(cur)
    _ensure_monitoring_tables(cur)
    _seed_theaters(cur)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_time_bins
        ON aircraft_positions (timestamp, lat_bin, lon_bin)
    """)
    conn.commit()


@timed
def main():
    db_path = resolve_db_path()
    conn = sqlite3.connect(db_path)
    _migrate(conn)
    cursor = conn.cursor()

    cursor.execute("SELECT timestamp FROM aircraft_positions ORDER BY timestamp DESC LIMIT 5;")
    print("\n--- Latest Timestamps ---")
    print(cursor.fetchall())

    print("\n--- Most Suspicious ---")
    for row in most_suspicious(cursor):
        print(row)

    print("\n--- Recent Activity ---")
    for row in recent_activity(cursor):
        print(row)

    print("\n--- Loitering Aircraft ---")
    for row in loitering(cursor):
        print(row)

    persistent_aircraft(cursor)
    military_cluster(cursor)

    # Collect detection outputs for the intelligence layer
    _recurring   = recurring_regions(cursor)
    rank_regions(cursor)
    _new_entries = detect_new_entries(cursor)
    _movements   = detect_movements(cursor)
    _staging, _projection = detect_staging_and_projection(cursor)
    _coordinated = coordinated_activity(cursor)
    _spikes      = detect_spikes(cursor)
    _changes     = detect_activity_changes(cursor) or []
    detect_linked_regions(cursor)

    # External signals (NOTAM + maritime vessel traffic + satellite change detection).
    # Skipped entirely when ENABLE_EXTERNAL_SIGNALS is False so the pipeline
    # behaves identically to pre-external-signal behaviour.
    if ENABLE_EXTERNAL_SIGNALS:
        _external_signals, _source_reports = collect_operational_external_signals(db_path=db_path)
    else:
        _external_signals = []

    # Intelligence classification
    features = build_features(
        coordinated      = _coordinated,
        spikes           = _spikes,
        recurring        = _recurring,
        new_entries      = _new_entries,
        movements        = _movements,
        staging          = _staging,
        projection       = _projection,
        activity_changes = _changes,
        external_signals = _external_signals or None,
    )
    intelligence = classify_regions(features)

    print("\n--- INTELLIGENCE ASSESSMENT ---")
    for region in intelligence[:10]:
        print(region)

    # External signal influence summary
    _print_external_summary(_compute_external_summary(_external_signals, features))

    conn.close()


if __name__ == "__main__":
    main()
