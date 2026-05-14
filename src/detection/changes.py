import hashlib
import math
import time

from geo.clustering import merge_regions

_MILITARY_CAP = 10
_SCORE_CAP = 8.0
_MERGE_RADIUS = 0.3
_MIN_AIRCRAFT = 2
_CHANGE_SCORE_THRESHOLD = 13.0
_FILL_MIN_SCORE = 7.0

_CIVILIAN_SURGE_MIN_AIRCRAFT = 50
_MIN_CIVILIAN_SCORE = 6.0
_PERSISTENCE_BONUS = 2.0
_MILITARY_BONUS_THRESHOLD = 5
_MILITARY_BONUS = 3.0
_CIVILIAN_PROMOTION_MIN_AIRCRAFT = 75
_CIVILIAN_PROMOTION_MAX_MILITARY_PCT = 5.0
_CIVILIAN_PROMOTION_MAX_MILITARY_DELTA = 1

_LEVEL_HIGH = 22.0
_LEVEL_MEDIUM = 16.0


def _region_id(lat, lon):
    snapped_lat = round(lat * 2) / 2
    snapped_lon = round(lon * 2) / 2
    return hashlib.md5(f"{snapped_lat:.1f},{snapped_lon:.1f}".encode()).hexdigest()[:8]


def _classify_level(change_score):
    if change_score >= _LEVEL_HIGH:
        return "HIGH"
    if change_score >= _LEVEL_MEDIUM:
        return "MEDIUM"
    return "LOW"


def detect_activity_changes(cursor, include_breakdown=False):
    start = time.time()
    print("\n--- Activity Changes (Last 90 Minutes) ---")

    now = time.time()
    recent_cutoff = now - 1800
    prior_cutoff = now - 5400

    def fetch_snapshot(start_time, end_time=None):
        if end_time is None:
            cursor.execute("""
                SELECT
                    lat_bin,
                    lon_bin,
                    COUNT(DISTINCT icao24),
                    COUNT(DISTINCT CASE
                        WHEN type LIKE '%CARGO%' OR type LIKE '%MIL%' THEN icao24
                    END),
                    SUM(score),
                    GROUP_CONCAT(DISTINCT behavior)
                FROM aircraft_positions
                WHERE timestamp >= ?
                GROUP BY lat_bin, lon_bin
            """, (start_time,))
        else:
            cursor.execute("""
                SELECT
                    lat_bin,
                    lon_bin,
                    COUNT(DISTINCT icao24),
                    COUNT(DISTINCT CASE
                        WHEN type LIKE '%CARGO%' OR type LIKE '%MIL%' THEN icao24
                    END),
                    SUM(score),
                    GROUP_CONCAT(DISTINCT behavior)
                FROM aircraft_positions
                WHERE timestamp >= ? AND timestamp < ?
                GROUP BY lat_bin, lon_bin
            """, (start_time, end_time))

        # Stream rows directly from the cursor — avoids materialising the full
        # result set into a list before building the snapshot dicts.
        return [
            {
                "lat_bin": lat,
                "lon_bin": lon,
                "aircraft_count": ac,
                "military_count": mc,
                "total_score": ts or 0,
                "behaviors": set((beh or "").split(",")) - {""},
            }
            for lat, lon, ac, mc, ts, beh in cursor
        ]

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS region_activity (
            region_id TEXT PRIMARY KEY,
            first_seen REAL,
            last_seen REAL,
            total_detections INTEGER
        )
    """)

    t0 = time.time()
    _snap_recent = fetch_snapshot(recent_cutoff)
    print(f"[TIMER] snapshot_recent: {time.time() - t0:.3f}s")
    t0 = time.time()
    recent_regions = merge_regions(_snap_recent, radius=_MERGE_RADIUS)
    print(f"[TIMER] merge_recent: {time.time() - t0:.3f}s")

    t0 = time.time()
    _snap_prior = fetch_snapshot(prior_cutoff, recent_cutoff)
    print(f"[TIMER] snapshot_prior: {time.time() - t0:.3f}s")
    t0 = time.time()
    prior_regions = merge_regions(_snap_prior, radius=_MERGE_RADIUS)
    print(f"[TIMER] merge_prior: {time.time() - t0:.3f}s")

    t0 = time.time()
    _snap_older = fetch_snapshot(now - 21600, prior_cutoff)
    print(f"[TIMER] snapshot_older: {time.time() - t0:.3f}s")
    t0 = time.time()
    older_regions = merge_regions(_snap_older, radius=_MERGE_RADIUS)
    print(f"[TIMER] merge_older: {time.time() - t0:.3f}s")

    def region_key(r):
        return (round(r["lat"], 1), round(r["lon"], 1))

    prior_lookup  = {region_key(r): r for r in prior_regions}
    older_lookup  = {region_key(r) for r in older_regions}

    t0 = time.time()
    cursor.execute("SELECT region_id, first_seen, last_seen FROM region_activity")
    activity_cache = {rid: fs for rid, fs, _ls in cursor}
    print(f"[TIMER] db_cache_read: {time.time() - t0:.3f}s")

    results        = []
    fill_candidates = []
    suppressed_rows = []

    t0 = time.time()
    for region in sorted(recent_regions, key=lambda r: r["aircraft_count"], reverse=True):
        # --- unpack region fields once ---
        region_aircraft = region["aircraft_count"]
        if region_aircraft < _MIN_AIRCRAFT:
            continue

        region_military  = region["military_count"]
        region_avg_score = region["avg_score"]

        key          = region_key(region)
        prior_region = prior_lookup.get(key)

        if prior_region is None:
            prior_aircraft  = 0
            prior_military  = 0
            prior_avg_score = 0.0
            emerging_flag   = True
        else:
            prior_aircraft  = prior_region["aircraft_count"]
            prior_military  = prior_region["military_count"]
            prior_avg_score = prior_region["avg_score"]
            emerging_flag   = False

        aircraft_delta = region_aircraft - prior_aircraft
        military_delta = region_military  - prior_military
        score_delta    = region_avg_score  - prior_avg_score

        if prior_aircraft == 0 and region_aircraft >= 3:
            emerging_flag = True

        # --- score computation (sqrt computed once, reused for civilian_bonus) ---
        sqrt_delta       = math.sqrt(max(aircraft_delta, 0))
        aircraft_component = min(sqrt_delta, 6.0)
        military_component = min(military_delta, _MILITARY_CAP) * 2.5

        if prior_aircraft == 0:
            score_component = min(region_avg_score, _SCORE_CAP)
        else:
            score_component = min(score_delta, _SCORE_CAP) * 1.5

        change_score = aircraft_component + military_component + score_component

        persistence_bonus = 0.0
        if prior_aircraft > 0:
            persistence_bonus = _PERSISTENCE_BONUS
            change_score     += persistence_bonus

        civilian_bonus = 0.0
        if (military_delta == 0
                and aircraft_delta >= _CIVILIAN_SURGE_MIN_AIRCRAFT
                and region_avg_score >= _MIN_CIVILIAN_SCORE):
            civilian_bonus = min(1.5, sqrt_delta * 0.5)   # reuse sqrt_delta
            change_score  += civilian_bonus

        military_bonus = 0.0
        if military_delta >= _MILITARY_BONUS_THRESHOLD:
            military_bonus = _MILITARY_BONUS
            change_score  += military_bonus

        density_penalty = 1.0
        if military_delta < _MILITARY_BONUS_THRESHOLD:
            if region_aircraft > 5000:
                density_penalty = 0.6
            elif region_aircraft > 1000:
                density_penalty = 0.8
        if density_penalty < 1.0:
            change_score *= density_penalty

        change_score = round(change_score, 1)

        # --- early exit: skip all metadata work for non-qualifying rows ---
        if change_score < _FILL_MIN_SCORE:
            continue

        # --- metadata (only reached by qualifying rows) ---
        in_older = key in older_lookup
        if prior_aircraft > 0 and in_older:
            persistence_level = "LONG"
        elif prior_aircraft > 0:
            persistence_level = "MEDIUM"
        else:
            persistence_level = "SHORT"

        rid          = _region_id(region["lat"], region["lon"])
        first_seen   = activity_cache.get(rid, now)
        lifetime_minutes = round((now - first_seen) / 60, 1)

        # one division, three multiplications
        inv_aircraft      = 100.0 / region_aircraft
        percent_military  = round(region_military * inv_aircraft, 1)
        percent_persistent = round(min(prior_aircraft, region_aircraft) * inv_aircraft, 1)
        percent_new       = round(max(0, aircraft_delta) * inv_aircraft, 1)

        if percent_military > 30:
            region_type = "MILITARY_HEAVY"
        elif percent_military < 5:
            region_type = "CIVILIAN_HEAVY"
        else:
            region_type = "MIXED"

        flags = []
        if aircraft_delta >= 3:     flags.append("SURGE")
        if military_delta >= 2:     flags.append("MILITARY_BUILDUP")
        if emerging_flag:           flags.append("EMERGING_REGION")
        if score_delta >= 1.5:      flags.append("ESCALATION")

        tags = []
        if military_delta > 0:      tags.append("MILITARY_BUILDUP")
        if civilian_bonus > 0:      tags.append("CIVILIAN_SURGE")
        if persistence_bonus > 0:   tags.append("PERSISTENT_ACTIVITY")
        if prior_aircraft == 0:     tags.append("EMERGING_REGION")
        if change_score >= 25:      tags.append("HIGH_SCORE")
        if region_aircraft > 2000:  tags.append("HIGH_DENSITY_REGION")

        level = "HIGH" if change_score >= _LEVEL_HIGH else "MEDIUM" if change_score >= _LEVEL_MEDIUM else "LOW"

        row = (
            region["lat"],
            region["lon"],
            change_score,
            aircraft_delta,
            military_delta,
            round(score_delta, 1),
            ",".join(flags),
            level,
            tags,
            persistence_level,
            rid,
            lifetime_minutes,
            percent_military,
            percent_persistent,
            percent_new,
            region_type,
        )

        suppress_civilian_promotion = (
            region_type == "CIVILIAN_HEAVY"
            and region_aircraft >= _CIVILIAN_PROMOTION_MIN_AIRCRAFT
            and percent_military <= _CIVILIAN_PROMOTION_MAX_MILITARY_PCT
            and military_delta <= _CIVILIAN_PROMOTION_MAX_MILITARY_DELTA
        )

        if suppress_civilian_promotion:
            suppressed_rows.append(row)
            continue

        if change_score >= _CHANGE_SCORE_THRESHOLD:
            results.append(row)
        else:
            fill_candidates.append(row)

    print(f"[TIMER] main_loop: {time.time() - t0:.3f}s")

    t0 = time.time()
    fill_needed = max(0, 3 - len(results))
    if fill_needed > 0:
        fill_candidates.sort(key=lambda r: r[2], reverse=True)
        results.extend(fill_candidates[:fill_needed])

    sorted_results = sorted(results, key=lambda r: r[2], reverse=True)

    # single pass to partition by level (was three separate list comprehensions)
    high_rows, medium_rows, low_rows = [], [], []
    for r in sorted_results:
        lv = r[7]
        if lv == "HIGH":
            high_rows.append(r)
        elif lv == "MEDIUM":
            medium_rows.append(r)
        else:
            low_rows.append(r)
    low_rows.extend(sorted(suppressed_rows, key=lambda r: r[2]))

    for row in high_rows:
        print(row)

    for row in medium_rows[:5]:
        print(row)

    if low_rows:
        print((
            "LOW_ACTIVITY_SUMMARY",
            len(low_rows),
            low_rows[-1][2],
            low_rows[0][2],
            [r[10] for r in low_rows],
        ))

    cursor.connection.commit()
    print(f"[TIMER] sort_and_print: {time.time() - t0:.3f}s")
    print(f"[TIMER] detect_activity_changes: {time.time() - start:.3f}s")
    return sorted_results

def detect_linked_regions(cursor):
    start = time.time()
    print("\n--- Linked Regions (Last 2 Hours) ---")

    cutoff = time.time() - 7200

    # Single table scan replaces the three-CTE self-join.
    # The old query materialised `positions` as an unindexed CTE, joined
    # `bounds` back to it twice (first_seen and last_seen), then grouped —
    # each join was an O(n) scan per aircraft.
    # Here we do one O(n) Python pass to extract first/last grid cell per
    # aircraft, then one O(aircraft) pass to count routes.
    cursor.execute("""
        SELECT icao24,
               ROUND(lat * 2, 0) / 2.0,
               ROUND(lon * 2, 0) / 2.0,
               timestamp
        FROM aircraft_positions
        WHERE timestamp >= ?
    """, (cutoff,))

    # entry: [src_lat, src_lon, src_ts, dst_lat, dst_lon, dst_ts]
    aircraft = {}
    for icao24, lat, lon, ts in cursor:
        entry = aircraft.get(icao24)
        if entry is None:
            aircraft[icao24] = [lat, lon, ts, lat, lon, ts]
        else:
            if ts < entry[2]:
                entry[0] = lat; entry[1] = lon; entry[2] = ts
            if ts > entry[5]:
                entry[3] = lat; entry[4] = lon; entry[5] = ts

    # Count how many distinct aircraft share each (src, dst) route pair.
    # Each icao24 contributes to exactly one route, so a plain int counter
    # is equivalent to COUNT(DISTINCT icao24).
    routes = {}
    for entry in aircraft.values():
        src_lat = entry[0]; src_lon = entry[1]
        dst_lat = entry[3]; dst_lon = entry[4]
        if src_lat == dst_lat and src_lon == dst_lon:
            continue
        key = (src_lat, src_lon, dst_lat, dst_lon)
        r = routes.get(key)
        routes[key] = 1 if r is None else r + 1

    # Filter (HAVING movement_count >= 2), sort (ORDER BY DESC), limit 10.
    results = [
        (src_lat, src_lon, dst_lat, dst_lon, cnt)
        for (src_lat, src_lon, dst_lat, dst_lon), cnt in routes.items()
        if cnt >= 2
    ]
    results.sort(key=lambda r: r[4], reverse=True)

    if not results:
        print("No linked regions detected.")
        print(f"[TIMER] detect_linked_regions: {time.time() - start:.3f}s")
        return

    for src_lat, src_lon, dst_lat, dst_lon, movement_count in results[:10]:
        print((_region_id(src_lat, src_lon), _region_id(dst_lat, dst_lon), movement_count))
    print(f"[TIMER] detect_linked_regions: {time.time() - start:.3f}s")
