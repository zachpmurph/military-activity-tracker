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

_CIVILIAN_SURGE_MIN_AIRCRAFT = 25
_MIN_CIVILIAN_SCORE = 4.5
_PERSISTENCE_BONUS = 2.0
_MILITARY_BONUS_THRESHOLD = 5
_MILITARY_BONUS = 3.0

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
    print("\n--- Activity Changes (Last 90 Minutes) ---")

    now = time.time()
    recent_cutoff = now - 1800
    prior_cutoff = now - 5400

    # ----------------------------
    # Snapshot fetch (unchanged logic, optimized execution)
    # ----------------------------
    def fetch_snapshot(start_time, end_time=None):
        if end_time is None:
            cursor.execute("""
                SELECT
                    ROUND(lat, 1),
                    ROUND(lon, 1),
                    COUNT(DISTINCT icao24),
                    COUNT(DISTINCT CASE
                        WHEN type LIKE '%CARGO%' OR type LIKE '%MIL%' THEN icao24
                    END),
                    SUM(score),
                    GROUP_CONCAT(DISTINCT behavior)
                FROM aircraft_positions
                WHERE timestamp >= ?
                GROUP BY 1,2
            """, (start_time,))
        else:
            cursor.execute("""
                SELECT
                    ROUND(lat, 1),
                    ROUND(lon, 1),
                    COUNT(DISTINCT icao24),
                    COUNT(DISTINCT CASE
                        WHEN type LIKE '%CARGO%' OR type LIKE '%MIL%' THEN icao24
                    END),
                    SUM(score),
                    GROUP_CONCAT(DISTINCT behavior)
                FROM aircraft_positions
                WHERE timestamp >= ? AND timestamp < ?
                GROUP BY 1,2
            """, (start_time, end_time))

        return [
            {
                "lat_bin": lat,
                "lon_bin": lon,
                "aircraft_count": ac,
                "military_count": mc,
                "total_score": ts or 0,
                "behaviors": set((beh or "").split(",")) - {""},
            }
            for lat, lon, ac, mc, ts, beh in cursor.fetchall()
        ]

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS region_activity (
            region_id TEXT PRIMARY KEY,
            first_seen REAL,
            last_seen REAL,
            total_detections INTEGER
        )
    """)

    # ----------------------------
    # Build snapshots
    # ----------------------------
    recent_regions = merge_regions(fetch_snapshot(recent_cutoff), radius=_MERGE_RADIUS)
    prior_regions = merge_regions(fetch_snapshot(prior_cutoff, recent_cutoff), radius=_MERGE_RADIUS)
    older_regions = merge_regions(fetch_snapshot(now - 21600, prior_cutoff), radius=_MERGE_RADIUS)

    # ----------------------------
    # FAST LOOKUP MAPS (huge speedup)
    # ----------------------------
    def region_key(r):
        return (round(r["lat"], 1), round(r["lon"], 1))

    prior_lookup = {region_key(r): r for r in prior_regions}
    older_lookup = set(region_key(r) for r in older_regions)

    results = []
    fill_candidates = []

    # ----------------------------
    # Batch fetch region_activity (avoid per-loop queries)
    # ----------------------------
    cursor.execute("SELECT region_id, first_seen, last_seen FROM region_activity")
    activity_cache = {rid: (fs, ls) for rid, fs, ls in cursor.fetchall()}

    # ----------------------------
    # Main loop (optimized)
    # ----------------------------
    for region in sorted(recent_regions, key=lambda r: r["aircraft_count"], reverse=True):
        if region["aircraft_count"] < _MIN_AIRCRAFT:
            continue

        key = region_key(region)
        prior_region = prior_lookup.get(key)

        if prior_region is None:
            prior_region = {
                "aircraft_count": 0,
                "military_count": 0,
                "avg_score": 0,
                "distinct_behaviors": 0,
            }
            emerging_flag = True
        else:
            emerging_flag = False

        aircraft_delta = region["aircraft_count"] - prior_region["aircraft_count"]
        military_delta = region["military_count"] - prior_region["military_count"]
        score_delta = region["avg_score"] - prior_region["avg_score"]

        surge_flag = aircraft_delta >= 3
        military_buildup_flag = military_delta >= 2
        if prior_region["aircraft_count"] == 0 and region["aircraft_count"] >= 3:
            emerging_flag = True
        escalation_flag = score_delta >= 1.5

        aircraft_component = min(math.sqrt(max(aircraft_delta, 0)), 6.0)
        military_component = min(military_delta, _MILITARY_CAP) * 2.5

        if prior_region["aircraft_count"] == 0:
            score_component = min(region["avg_score"], _SCORE_CAP)
        else:
            score_component = min(score_delta, _SCORE_CAP) * 1.5

        change_score = aircraft_component + military_component + score_component

        persistence_bonus = 0.0
        if prior_region["aircraft_count"] > 0:
            persistence_bonus = _PERSISTENCE_BONUS
            change_score += persistence_bonus

        civilian_bonus = 0.0
        if (
            military_delta == 0
            and aircraft_delta >= _CIVILIAN_SURGE_MIN_AIRCRAFT
            and region["avg_score"] >= _MIN_CIVILIAN_SCORE
        ):
            civilian_bonus = min(1.5, math.sqrt(max(aircraft_delta, 0)) * 0.5)
            change_score += civilian_bonus

        military_bonus = 0.0
        if military_delta >= _MILITARY_BONUS_THRESHOLD:
            military_bonus = _MILITARY_BONUS
            change_score += military_bonus

        density_penalty = 1.0
        if military_delta < _MILITARY_BONUS_THRESHOLD:
            if region["aircraft_count"] > 5000:
                density_penalty = 0.6
            elif region["aircraft_count"] > 1000:
                density_penalty = 0.8

        if density_penalty < 1.0:
            change_score *= density_penalty

        change_score = round(change_score, 1)

        # ----------------------------
        # Persistence check (fast set lookup)
        # ----------------------------
        in_older = key in older_lookup

        if prior_region["aircraft_count"] > 0 and in_older:
            persistence_level = "LONG"
        elif prior_region["aircraft_count"] > 0:
            persistence_level = "MEDIUM"
        else:
            persistence_level = "SHORT"

        # ----------------------------
        # Region activity cache (no DB hits per row)
        # ----------------------------
        rid = _region_id(region["lat"], region["lon"])

        if rid in activity_cache:
            first_seen, _ = activity_cache[rid]
        else:
            first_seen = now

        lifetime_minutes = round((now - first_seen) / 60, 1)

        percent_military = round(region["military_count"] / region["aircraft_count"] * 100, 1)
        percent_persistent = round(
            min(prior_region["aircraft_count"], region["aircraft_count"]) / region["aircraft_count"] * 100, 1
        )
        percent_new = round(max(0, aircraft_delta) / region["aircraft_count"] * 100, 1)

        if percent_military > 30:
            region_type = "MILITARY_HEAVY"
        elif percent_military < 5:
            region_type = "CIVILIAN_HEAVY"
        else:
            region_type = "MIXED"

        flags = []
        if surge_flag: flags.append("SURGE")
        if military_buildup_flag: flags.append("MILITARY_BUILDUP")
        if emerging_flag: flags.append("EMERGING_REGION")
        if escalation_flag: flags.append("ESCALATION")

        tags = []
        if military_delta > 0: tags.append("MILITARY_BUILDUP")
        if civilian_bonus > 0: tags.append("CIVILIAN_SURGE")
        if persistence_bonus > 0: tags.append("PERSISTENT_ACTIVITY")
        if prior_region["aircraft_count"] == 0: tags.append("EMERGING_REGION")
        if change_score >= 25: tags.append("HIGH_SCORE")
        if region["aircraft_count"] > 2000: tags.append("HIGH_DENSITY_REGION")

        row = (
            region["lat"],
            region["lon"],
            change_score,
            aircraft_delta,
            military_delta,
            round(score_delta, 1),
            ",".join(flags),
            _classify_level(change_score),
            tags,
            persistence_level,
            rid,
            lifetime_minutes,
            percent_military,
            percent_persistent,
            percent_new,
            region_type,
        )

        if change_score >= _CHANGE_SCORE_THRESHOLD:
            results.append(row)
        elif change_score >= _FILL_MIN_SCORE:
            fill_candidates.append(row)

    # ----------------------------
    # Final output (unchanged)
    # ----------------------------
    fill_needed = max(0, 3 - len(results))
    if fill_needed > 0:
        fill_candidates.sort(key=lambda r: r[2], reverse=True)
        results.extend(fill_candidates[:fill_needed])

    sorted_results = sorted(results, key=lambda r: r[2], reverse=True)

    high_rows   = [r for r in sorted_results if r[7] == "HIGH"]
    medium_rows = [r for r in sorted_results if r[7] == "MEDIUM"]
    low_rows    = [r for r in sorted_results if r[7] == "LOW"]

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

def detect_linked_regions(cursor):
    print("\n--- Linked Regions (Last 2 Hours) ---")

    cutoff = time.time() - 7200

    cursor.execute("""
        WITH positions AS (
            SELECT
                icao24,
                ROUND(lat * 2, 0) / 2.0 AS lat_grid,
                ROUND(lon * 2, 0) / 2.0 AS lon_grid,
                timestamp
            FROM aircraft_positions
            WHERE timestamp >= ?
        ),
        bounds AS (
            SELECT icao24, MIN(timestamp) AS first_seen, MAX(timestamp) AS last_seen
            FROM positions
            GROUP BY icao24
        ),
        links AS (
            SELECT
                src.lat_grid AS src_lat,
                src.lon_grid AS src_lon,
                dst.lat_grid AS dst_lat,
                dst.lon_grid AS dst_lon,
                COUNT(DISTINCT bounds.icao24) AS movement_count
            FROM bounds
            JOIN positions AS src ON bounds.icao24 = src.icao24 AND bounds.first_seen = src.timestamp
            JOIN positions AS dst ON bounds.icao24 = dst.icao24 AND bounds.last_seen  = dst.timestamp
            WHERE src.lat_grid != dst.lat_grid OR src.lon_grid != dst.lon_grid
            GROUP BY src_lat, src_lon, dst_lat, dst_lon
            HAVING movement_count >= 2
        )
        SELECT src_lat, src_lon, dst_lat, dst_lon, movement_count
        FROM links
        ORDER BY movement_count DESC
        LIMIT 10
    """, (cutoff,))

    rows = cursor.fetchall()
    if not rows:
        print("No linked regions detected.")
        return

    for src_lat, src_lon, dst_lat, dst_lon, movement_count in rows:
        print((_region_id(src_lat, src_lon), _region_id(dst_lat, dst_lon), movement_count))