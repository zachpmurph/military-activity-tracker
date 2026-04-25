import time


def recurring_regions(cursor):
    start = time.time()
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

    rows = cursor.fetchall()
    for row in rows:
        print(row)
    print(f"[TIMER] recurring_regions: {time.time() - start:.3f}s")
    return rows


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

    rows = cursor.fetchall()
    for row in rows:
        print(row)
    return rows


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

    rows = cursor.fetchall()
    for row in rows:
        print(row)
    return rows


def detect_new_entries(cursor):
    start = time.time()
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
    print(f"[TIMER] detect_new_entries: {time.time() - start:.3f}s")


