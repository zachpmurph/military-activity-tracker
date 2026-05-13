import time


def flow_routes_last_6h(cursor, cutoff=None):
    current_time = time.time()
    if cutoff is None:
        cutoff = current_time - 21600

    track_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(aircraft_tracks)").fetchall()
    }
    type_expr = "type" if "type" in track_columns else "''"

    rows = cursor.execute(
        f"""
        SELECT
            ROUND(start_lat, 1) AS origin_lat,
            ROUND(start_lon, 1) AS origin_lon,
            ROUND(end_lat, 1) AS dest_lat,
            ROUND(end_lon, 1) AS dest_lon,
            COUNT(DISTINCT icao24) AS aircraft_count,
            COUNT(DISTINCT CASE
                WHEN {type_expr} LIKE '%CARGO%' OR {type_expr} LIKE '%MIL%' THEN icao24
            END) AS military_count,
            ROUND(AVG(max_distance), 1) AS avg_distance,
            ROUND(
                (COUNT(DISTINCT icao24) * 1.0) +
                (COUNT(DISTINCT CASE
                    WHEN {type_expr} LIKE '%CARGO%' OR {type_expr} LIKE '%MIL%' THEN icao24
                END) * 2.5),
                1
            ) AS flow_score
        FROM aircraft_tracks
        WHERE last_seen > ?
          AND max_distance >= 0.5
        GROUP BY origin_lat, origin_lon, dest_lat, dest_lon
        HAVING aircraft_count >= 2
        ORDER BY flow_score DESC, aircraft_count DESC
        LIMIT 10;
        """,
        (cutoff,),
    ).fetchall()
    return rows


def detect_flow_routes(cursor, cutoff, type_expr=None):
    del type_expr
    print("\n--- Major Flow Routes (Last 6 Hours) ---")
    rows = flow_routes_last_6h(cursor, cutoff=cutoff)
    for row in rows:
        print((row[0], row[1], row[2], row[3], row[4], row[5], row[7]))
    return rows
