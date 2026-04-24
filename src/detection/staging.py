import time

from detection.flows import detect_flow_routes


def detect_staging_and_projection(cursor):
    current_time = time.time()
    cutoff = current_time - 21600

    track_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(aircraft_tracks)").fetchall()
    }
    type_expr = "type" if "type" in track_columns else "''"

    print("\n--- Staging Regions (Last 6 Hours) ---")
    cursor.execute(f"""
        SELECT
            ROUND(start_lat, 1) AS lat_bin,
            ROUND(start_lon, 1) AS lon_bin,
            COUNT(DISTINCT icao24) AS aircraft_count,
            COUNT(DISTINCT CASE
                WHEN {type_expr} LIKE '%CARGO%' OR {type_expr} LIKE '%MIL%' THEN icao24
            END) AS military_count
        FROM aircraft_tracks
        WHERE last_seen > ?
          AND max_distance >= 0.5
        GROUP BY lat_bin, lon_bin
        HAVING aircraft_count >= 3
        ORDER BY military_count DESC, aircraft_count DESC
        LIMIT 10;
    """, (cutoff,))

    for row in cursor.fetchall():
        print(row)

    print("\n--- Projection Regions (Last 6 Hours) ---")
    cursor.execute(f"""
        SELECT
            ROUND(end_lat, 1) AS lat_bin,
            ROUND(end_lon, 1) AS lon_bin,
            COUNT(DISTINCT icao24) AS aircraft_count,
            COUNT(DISTINCT CASE
                WHEN {type_expr} LIKE '%CARGO%' OR {type_expr} LIKE '%MIL%' THEN icao24
            END) AS military_count
        FROM aircraft_tracks
        WHERE last_seen > ?
          AND max_distance >= 0.5
        GROUP BY lat_bin, lon_bin
        HAVING aircraft_count >= 3
        ORDER BY military_count DESC, aircraft_count DESC
        LIMIT 10;
    """, (cutoff,))

    for row in cursor.fetchall():
        print(row)

    detect_flow_routes(cursor, cutoff, type_expr)
