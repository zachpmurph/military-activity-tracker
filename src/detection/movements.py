import time


def detect_movements(cursor):
    start = time.time()
    print("\n--- Cross-Region Movements (Last 2 Hours) ---")

    end_time   = time.time()
    start_time = end_time - (2 * 3600)

    # Single table scan replaces the O(n²) CTE self-join.
    # The old query materialised recent_positions as an unindexed CTE then
    # joined it to itself twice; each join was a full O(n) scan per aircraft.
    # Here we do one O(n) Python pass to extract first/last position per
    # aircraft, then one O(aircraft) pass to aggregate routes.
    cursor.execute("""
        SELECT icao24, ROUND(lat, 1), ROUND(lon, 1), timestamp, type
        FROM aircraft_positions
        WHERE timestamp BETWEEN ? AND ?
    """, (start_time, end_time))

    # entry: [from_lat, from_lon, from_ts, to_lat, to_lon, to_ts, to_type]
    aircraft = {}
    for icao24, lat, lon, ts, typ in cursor:
        entry = aircraft.get(icao24)
        if entry is None:
            aircraft[icao24] = [lat, lon, ts, lat, lon, ts, typ]
        else:
            if ts < entry[2]:
                entry[0] = lat; entry[1] = lon; entry[2] = ts
            if ts > entry[5]:
                entry[3] = lat; entry[4] = lon; entry[5] = ts; entry[6] = typ

    # Aggregate by (from, to) route.
    # Each icao24 contributes to exactly one route, so plain int counters
    # are equivalent to COUNT(DISTINCT icao24).
    # route -> [ac_count, mil_count, dist_sum, dist_count, long_count]
    routes = {}

    for icao24, entry in aircraft.items():
        from_lat = entry[0];  from_lon = entry[1]
        to_lat   = entry[3];  to_lon   = entry[4]
        typ      = entry[6]

        if from_lat == to_lat and from_lon == to_lon:
            continue

        dlat    = to_lat - from_lat
        dlon    = to_lon - from_lon
        dist_sq = dlat * dlat + dlon * dlon

        if dist_sq < 0.09:              # movement_distance < 0.3 → skip
            continue

        is_mil = bool(typ and ('CARGO' in typ or 'MIL' in typ))
        if dist_sq < 0.25 and not is_mil:   # dist < 0.5 and not military → skip
            continue

        dist = dist_sq ** 0.5           # SQRT computed once, only for passing rows
        key  = (from_lat, from_lon, to_lat, to_lon)
        r    = routes.get(key)
        if r is None:
            routes[key] = [1, 1 if is_mil else 0, dist, 1, 1 if dist_sq >= 1.0 else 0]
        else:
            r[0] += 1
            if is_mil:        r[1] += 1
            r[2]             += dist
            r[3]             += 1
            if dist_sq >= 1.0: r[4] += 1

    # Filter, sort, and print — mirrors the original HAVING / ORDER BY / LIMIT.
    results = []
    for (from_lat, from_lon, to_lat, to_lon), r in routes.items():
        ac_count, mil_count, dist_sum, dist_count, long_count = r

        if ac_count < 2:
            continue

        avg_dist = dist_sum / dist_count
        if not (mil_count >= 1 or avg_dist >= 0.5 or long_count >= 1):
            continue

        results.append((from_lat, from_lon, to_lat, to_lon,
                        ac_count, mil_count, round(avg_dist, 1)))

    # ORDER BY military_moved DESC, avg_distance DESC, aircraft_moved DESC
    results.sort(key=lambda r: (-r[5], -r[6], -r[4]))

    for row in results[:10]:
        print(row)
    print(f"[TIMER] detect_movements: {time.time() - start:.3f}s")
