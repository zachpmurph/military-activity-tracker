import sqlite3
import time
from pathlib import Path

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
            type IN ('US_CARGO', 'UK_CARGO', 'TANKER')
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
          AND type IN ('US_CARGO', 'UK_CARGO')
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
# Main
# ----------------------------

def _migrate(conn):
    """Ensure lat_bin / lon_bin columns and the composite index exist.

    Safe to call on every startup: ADD COLUMN is no-op if the column already
    exists (caught), and CREATE INDEX IF NOT EXISTS is always idempotent.
    """
    cur = conn.cursor()
    for col in ("lat_bin", "lon_bin"):
        try:
            cur.execute(f"ALTER TABLE aircraft_positions ADD COLUMN {col} REAL")
        except sqlite3.OperationalError:
            pass  # column already present
    cur.execute("""
        UPDATE aircraft_positions
        SET lat_bin = ROUND(lat, 1),
            lon_bin = ROUND(lon, 1)
        WHERE lat_bin IS NULL
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_time_bins
        ON aircraft_positions (timestamp, lat_bin, lon_bin)
    """)
    conn.commit()


@timed
def main():
    db_path = Path(__file__).resolve().parent / "data" / "aircraft.db"
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
    )
    intelligence = classify_regions(features)

    print("\n--- INTELLIGENCE ASSESSMENT ---")
    for region in intelligence[:10]:
        print(region)

    conn.close()


if __name__ == "__main__":
    main()
