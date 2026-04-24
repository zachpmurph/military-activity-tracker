import ast
import io
import sqlite3
import time
from contextlib import redirect_stdout

from detection.clusters import (
    coordinated_activity,
    detect_new_entries,
    detect_spikes,
    recurring_regions,
)
from detection.changes import detect_activity_changes, detect_linked_regions
from detection.movements import detect_movements
from detection.staging import detect_staging_and_projection

# ----------------------------
# Basic queries
# ----------------------------

def most_suspicious(cursor):
    cursor.execute("""
    SELECT callsign, type, behavior, AVG(score) as avg_score, COUNT(*) as sightings
    FROM aircraft_positions
    GROUP BY icao24
    ORDER BY avg_score DESC, sightings DESC
    LIMIT 10
    """)
    return cursor.fetchall()


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


def loitering(cursor):
    cursor.execute("""
    SELECT callsign, COUNT(*) as sightings
    FROM aircraft_positions
    WHERE behavior = 'LOITERING'
    GROUP BY icao24
    ORDER BY sightings DESC
    LIMIT 10
    """)
    return cursor.fetchall()


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

    for row in cursor.fetchall():
        print(row)


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

def _capture_region_rows(query_func, cursor):
    output = io.StringIO()

    with redirect_stdout(output):
        query_func(cursor)

    rows = []
    for line in output.getvalue().splitlines():
        line = line.strip()
        if not line.startswith("("):
            continue

        try:
            row = ast.literal_eval(line)
        except (SyntaxError, ValueError):
            continue

        if isinstance(row, tuple):
            rows.append(row)

    return rows


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

    for lat, lon, aircraft_count, military_count in _capture_region_rows(coordinated_activity, cursor):
        region = get_region(lat, lon)
        region["aircraft_count"] = max(region["aircraft_count"], aircraft_count)
        region["military_count"] = max(region["military_count"], military_count)
        region["coordinated_flag"] = 1

    for lat, lon, aircraft_count, military_count in _capture_region_rows(detect_spikes, cursor):
        region = get_region(lat, lon)
        region["aircraft_count"] = max(region["aircraft_count"], aircraft_count)
        region["military_count"] = max(region["military_count"], military_count)
        region["spike_flag"] = 1

    for lat, lon, appearances, total_aircraft, military_presence in _capture_region_rows(recurring_regions, cursor):
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

def main():
    conn = sqlite3.connect("data/aircraft.db")
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
    recurring_regions(cursor)
    rank_regions(cursor)
    detect_new_entries(cursor)
    detect_movements(cursor)
    detect_staging_and_projection(cursor)
    detect_activity_changes(cursor)
    detect_linked_regions(cursor)

    conn.close()


if __name__ == "__main__":
    main()
