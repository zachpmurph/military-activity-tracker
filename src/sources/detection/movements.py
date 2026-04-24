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
