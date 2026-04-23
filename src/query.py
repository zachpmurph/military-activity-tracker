import ast
import io
import sqlite3
import time
from contextlib import redirect_stdout

# ----------------------------
# Queries
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

    cutoff = time.time() - 1800  # 30 minutes

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

    import time
    cutoff = time.time() - 1800  # 30 minutes

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

    cutoff = time.time() - 1800  # 30 min


    for row in cursor.fetchall():
        print(row)


def recurring_regions(cursor):
    print("\n--- Recurring Activity Regions (Last 6 Hours) ---")

    cutoff = time.time() - (6 * 3600)

    cursor.execute("""
        WITH bucket_clusters AS (
            SELECT
                ROUND(lat, 1) AS lat_bin,
                ROUND(lon, 1) AS lon_bin,
                CAST(timestamp / 1800 AS INTEGER) AS bucket,
                COUNT(DISTINCT icao24) AS aircraft_count,
                COUNT(DISTINCT CASE
                    WHEN type IN ('US_CARGO', 'UK_CARGO', 'TANKER') THEN icao24
                END) AS military_count,
                AVG(score) AS avg_score
            FROM aircraft_positions
            WHERE timestamp > ?
            GROUP BY bucket, lat_bin, lon_bin
            HAVING aircraft_count >= 3
        )
        SELECT
            lat_bin,
            lon_bin,
            COUNT(DISTINCT bucket) AS appearances,
            SUM(aircraft_count) AS total_aircraft,
            SUM(military_count) AS military_presence
        FROM bucket_clusters
        GROUP BY lat_bin, lon_bin
        HAVING appearances >= 3
        ORDER BY appearances DESC, military_presence DESC
        LIMIT 10;
    """, (cutoff,))

    for row in cursor.fetchall():
        print(row)


def coordinated_activity(cursor):
    print("\n--- Coordinated Activity Regions (Last 30 Minutes) ---")

    cutoff = time.time() - 1800

    cursor.execute("""
        SELECT
            ROUND(lat, 1) AS lat_bin,
            ROUND(lon, 1) AS lon_bin,
            COUNT(DISTINCT icao24) AS aircraft_count,
            COUNT(DISTINCT CASE
                WHEN type IN ('US_CARGO', 'UK_CARGO', 'TANKER') THEN icao24
            END) AS military_count
        FROM aircraft_positions
        WHERE timestamp > ?
        GROUP BY lat_bin, lon_bin
        HAVING aircraft_count >= 3
        ORDER BY military_count DESC, aircraft_count DESC
        LIMIT 10;
    """, (cutoff,))

    for row in cursor.fetchall():
        print(row)


def detect_spikes(cursor):
    print("\n--- Activity Spikes (Last 30 Minutes) ---")

    now = time.time()
    current_cutoff = now - 1800
    previous_cutoff = now - 3600

    cursor.execute("""
        WITH current_window AS (
            SELECT
                ROUND(lat, 1) AS lat_bin,
                ROUND(lon, 1) AS lon_bin,
                COUNT(DISTINCT icao24) AS aircraft_count,
                COUNT(DISTINCT CASE
                    WHEN type IN ('US_CARGO', 'UK_CARGO', 'TANKER') THEN icao24
                END) AS military_count
            FROM aircraft_positions
            WHERE timestamp > ?
            GROUP BY lat_bin, lon_bin
        ),
        previous_window AS (
            SELECT
                ROUND(lat, 1) AS lat_bin,
                ROUND(lon, 1) AS lon_bin,
                COUNT(DISTINCT icao24) AS aircraft_count
            FROM aircraft_positions
            WHERE timestamp > ? AND timestamp <= ?
            GROUP BY lat_bin, lon_bin
        )
        SELECT
            current_window.lat_bin,
            current_window.lon_bin,
            current_window.aircraft_count,
            current_window.military_count
        FROM current_window
        LEFT JOIN previous_window
            ON current_window.lat_bin = previous_window.lat_bin
           AND current_window.lon_bin = previous_window.lon_bin
        WHERE current_window.aircraft_count >= 3
          AND current_window.aircraft_count > COALESCE(previous_window.aircraft_count, 0)
        ORDER BY
            (current_window.aircraft_count - COALESCE(previous_window.aircraft_count, 0)) DESC,
            current_window.military_count DESC
        LIMIT 10;
    """, (current_cutoff, previous_cutoff, current_cutoff))

    for row in cursor.fetchall():
        print(row)


def detect_new_entries(cursor):
    print("\n--- New Aircraft Activity (Last 30 Minutes) ---")

    recent_cutoff = time.time() - 1800
    baseline_start = recent_cutoff - (6 * 3600)
    baseline_end = recent_cutoff - (2 * 3600)

    cursor.execute("""
        WITH recent_presence AS (
            SELECT
                icao24,
                ROUND(lat, 1) AS lat_bin,
                ROUND(lon, 1) AS lon_bin,
                MAX(type) AS type
            FROM aircraft_positions
            WHERE timestamp >= ?
            GROUP BY icao24, lat_bin, lon_bin
        ),
        baseline_presence AS (
            SELECT
                icao24,
                ROUND(lat, 1) AS lat_bin,
                ROUND(lon, 1) AS lon_bin
            FROM aircraft_positions
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY icao24, lat_bin, lon_bin
        ),
        new_entries AS (
            SELECT
                recent_presence.icao24,
                recent_presence.lat_bin,
                recent_presence.lon_bin,
                recent_presence.type
            FROM recent_presence
            LEFT JOIN baseline_presence
                ON recent_presence.icao24 = baseline_presence.icao24
               AND recent_presence.lat_bin = baseline_presence.lat_bin
               AND recent_presence.lon_bin = baseline_presence.lon_bin
            WHERE baseline_presence.icao24 IS NULL
        )
        SELECT
            lat_bin,
            lon_bin,
            COUNT(DISTINCT icao24) AS new_aircraft,
            COUNT(DISTINCT CASE
                WHEN type LIKE '%CARGO%' OR type LIKE '%MIL%' THEN icao24
            END) AS new_military
        FROM new_entries
        GROUP BY lat_bin, lon_bin
        HAVING new_aircraft >= 2
        ORDER BY new_military DESC, new_aircraft DESC
        LIMIT 10;
    """, (recent_cutoff, baseline_start, baseline_end))

    for row in cursor.fetchall():
        print(row)


def detect_movements(cursor):
    print("\n--- Cross-Region Movements (Last 2 Hours) ---")

    end_time = time.time()
    start_time = end_time - (2 * 3600)

    cursor.execute("""
        WITH recent_positions AS (
            SELECT
                icao24,
                ROUND(lat, 1) AS lat_bin,
                ROUND(lon, 1) AS lon_bin,
                timestamp,
                type
            FROM aircraft_positions
            WHERE timestamp BETWEEN ? AND ?
        ),
        aircraft_times AS (
            SELECT
                icao24,
                MIN(timestamp) AS first_seen,
                MAX(timestamp) AS last_seen
            FROM recent_positions
            GROUP BY icao24
        ),
        movements AS (
            SELECT
                aircraft_times.icao24,
                start_pos.lat_bin AS from_lat,
                start_pos.lon_bin AS from_lon,
                end_pos.lat_bin AS to_lat,
                end_pos.lon_bin AS to_lon,
                MAX(end_pos.type) AS type,
                SQRT(
                    ((end_pos.lat_bin - start_pos.lat_bin) * (end_pos.lat_bin - start_pos.lat_bin)) +
                    ((end_pos.lon_bin - start_pos.lon_bin) * (end_pos.lon_bin - start_pos.lon_bin))
                ) AS movement_distance,
                CASE
                    WHEN SQRT(
                        ((end_pos.lat_bin - start_pos.lat_bin) * (end_pos.lat_bin - start_pos.lat_bin)) +
                        ((end_pos.lon_bin - start_pos.lon_bin) * (end_pos.lon_bin - start_pos.lon_bin))
                    ) >= 1.0 THEN 1
                    ELSE 0
                END AS long_move
            FROM aircraft_times
            JOIN recent_positions AS start_pos
                ON aircraft_times.icao24 = start_pos.icao24
               AND aircraft_times.first_seen = start_pos.timestamp
            JOIN recent_positions AS end_pos
                ON aircraft_times.icao24 = end_pos.icao24
               AND aircraft_times.last_seen = end_pos.timestamp
            WHERE start_pos.lat_bin != end_pos.lat_bin
               OR start_pos.lon_bin != end_pos.lon_bin
            GROUP BY
                aircraft_times.icao24,
                start_pos.lat_bin,
                start_pos.lon_bin,
                end_pos.lat_bin,
                end_pos.lon_bin
        ),
        filtered_movements AS (
            SELECT
                icao24,
                from_lat,
                from_lon,
                to_lat,
                to_lon,
                type,
                movement_distance,
                long_move
            FROM movements
            WHERE movement_distance >= 0.3
              AND (
                  movement_distance >= 0.5
                  OR type LIKE '%CARGO%'
                  OR type LIKE '%MIL%'
              )
        )
        SELECT
            from_lat,
            from_lon,
            to_lat,
            to_lon,
            COUNT(DISTINCT icao24) AS aircraft_moved,
            COUNT(DISTINCT CASE
                WHEN type LIKE '%CARGO%' OR type LIKE '%MIL%' THEN icao24
            END) AS military_moved,
            ROUND(AVG(movement_distance), 1) AS avg_distance
        FROM filtered_movements
        GROUP BY from_lat, from_lon, to_lat, to_lon
        HAVING aircraft_moved >= 2
           AND (
               military_moved >= 1
               OR AVG(movement_distance) >= 0.5
               OR SUM(long_move) >= 1
           )
        ORDER BY military_moved DESC, avg_distance DESC, aircraft_moved DESC
        LIMIT 10;
    """, (start_time, end_time))

    for row in cursor.fetchall():
        print(row)


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
    import sqlite3
    import time

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

    # 🔥 NEW LINE
    military_cluster(cursor)
    recurring_regions(cursor)
    rank_regions(cursor)
    detect_new_entries(cursor)
    detect_movements(cursor)

    conn.close()    # ✅ Use ONE consistent database path
    conn = sqlite3.connect("data/aircraft.db")
    cursor = conn.cursor()


if __name__ == "__main__":
    main()
