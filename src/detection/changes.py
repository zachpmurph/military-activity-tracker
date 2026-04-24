import hashlib
import math
import time

from geo.clustering import merge_regions

_MILITARY_CAP = 10
_SCORE_CAP = 8.0
_MERGE_RADIUS = 0.3
_MIN_AIRCRAFT = 2
_CHANGE_SCORE_THRESHOLD = 15.0

_CIVILIAN_SURGE_MIN_AIRCRAFT = 25
_MIN_CIVILIAN_SCORE = 5.0
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

    def fetch_snapshot(start_time, end_time=None):
        if end_time is None:
            cursor.execute("""
                SELECT
                    ROUND(lat, 1) AS lat_bin,
                    ROUND(lon, 1) AS lon_bin,
                    COUNT(DISTINCT icao24) AS aircraft_count,
                    COUNT(DISTINCT CASE
                        WHEN type LIKE '%CARGO%' OR type LIKE '%MIL%' THEN icao24
                    END) AS military_count,
                    SUM(score) AS total_score,
                    GROUP_CONCAT(DISTINCT behavior) AS behaviors
                FROM aircraft_positions
                WHERE timestamp >= ?
                GROUP BY lat_bin, lon_bin
            """, (start_time,))
        else:
            cursor.execute("""
                SELECT
                    ROUND(lat, 1) AS lat_bin,
                    ROUND(lon, 1) AS lon_bin,
                    COUNT(DISTINCT icao24) AS aircraft_count,
                    COUNT(DISTINCT CASE
                        WHEN type LIKE '%CARGO%' OR type LIKE '%MIL%' THEN icao24
                    END) AS military_count,
                    SUM(score) AS total_score,
                    GROUP_CONCAT(DISTINCT behavior) AS behaviors
                FROM aircraft_positions
                WHERE timestamp >= ?
                  AND timestamp < ?
                GROUP BY lat_bin, lon_bin
            """, (start_time, end_time))

        snapshot = []
        for lat_bin, lon_bin, aircraft_count, military_count, total_score, behaviors in cursor.fetchall():
            snapshot.append({
                "lat_bin": lat_bin,
                "lon_bin": lon_bin,
                "aircraft_count": aircraft_count,
                "military_count": military_count,
                "total_score": total_score or 0,
                "behaviors": set((behaviors or "").split(",")) - {""},
            })
        return snapshot

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS region_activity (
            region_id TEXT PRIMARY KEY,
            first_seen REAL,
            last_seen REAL,
            total_detections INTEGER
        )
    """)

    recent_regions = merge_regions(fetch_snapshot(recent_cutoff), radius=_MERGE_RADIUS)
    prior_regions = merge_regions(fetch_snapshot(prior_cutoff, recent_cutoff), radius=_MERGE_RADIUS)
    older_regions = merge_regions(fetch_snapshot(now - 21600, prior_cutoff), radius=_MERGE_RADIUS)
    # One-to-one matching: each prior region is consumed at most once
    unmatched_prior = set(range(len(prior_regions)))
    results = []

    for region in sorted(recent_regions, key=lambda item: item["aircraft_count"], reverse=True):
        if region["aircraft_count"] < _MIN_AIRCRAFT:
            continue

        best_match_index = None
        best_distance = None

        for prior_index in unmatched_prior:
            prior_region = prior_regions[prior_index]
            lat_diff = abs(region["lat"] - prior_region["lat"])
            lon_diff = abs(region["lon"] - prior_region["lon"])

            if lat_diff > 0.3 or lon_diff > 0.3:
                continue

            distance = (lat_diff ** 2) + (lon_diff ** 2)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_match_index = prior_index

        if best_match_index is None:
            prior_region = {
                "aircraft_count": 0,
                "military_count": 0,
                "avg_score": 0,
                "distinct_behaviors": 0,
            }
            emerging_flag = True
        else:
            prior_region = prior_regions[best_match_index]
            unmatched_prior.remove(best_match_index)
            emerging_flag = False

        aircraft_delta = region["aircraft_count"] - prior_region["aircraft_count"]
        military_delta = region["military_count"] - prior_region["military_count"]
        score_delta = region["avg_score"] - prior_region["avg_score"]

        surge_flag = aircraft_delta >= 3
        military_buildup_flag = military_delta >= 2
        if prior_region["aircraft_count"] == 0 and region["aircraft_count"] >= 3:
            emerging_flag = True
        escalation_flag = score_delta >= 1.5

        # sqrt scaling compresses large aircraft counts: 100 aircraft = sqrt(100) = 10,
        # capped at 6 so even 10,000-aircraft airports can't dominate the ranking.
        aircraft_component = min(math.sqrt(max(aircraft_delta, 0)), 6.0)
        military_component = min(military_delta, _MILITARY_CAP) * 2.5

        if best_match_index is None:
            # emergence_score: use the region's absolute avg_score (not delta vs 0)
            # to reward high-quality emerging signals without inflating from a zero baseline.
            score_component = min(region["avg_score"], _SCORE_CAP) * 1.0
        else:
            # growth_score: prior baseline is real, so score improvement is a genuine signal.
            score_component = min(score_delta, _SCORE_CAP) * 1.5

        change_score = aircraft_component + military_component + score_component

        persistence_bonus = 0.0
        if best_match_index is not None:
            persistence_bonus = _PERSISTENCE_BONUS
            change_score += persistence_bonus

        # Civilian surge bonus: scaled at 0.5 and capped at 1.5 so civilian-only regions
        # can never reach HIGH (max civilian-only = 20.0 base + 1.5 = 21.5 < _LEVEL_HIGH).
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
        if region["aircraft_count"] > 5000:
            density_penalty = 0.4
        elif region["aircraft_count"] > 1000:
            density_penalty = 0.6
        if density_penalty < 1.0:
            change_score *= density_penalty
        change_score = round(change_score, 1)

        if change_score < _CHANGE_SCORE_THRESHOLD:
            continue

        flags = []
        if surge_flag:
            flags.append("SURGE")
        if military_buildup_flag:
            flags.append("MILITARY_BUILDUP")
        if emerging_flag:
            flags.append("EMERGING_REGION")
        if escalation_flag:
            flags.append("ESCALATION")

        tags = []
        if military_delta > 0:
            tags.append("MILITARY_BUILDUP")
        if civilian_bonus > 0:
            tags.append("CIVILIAN_SURGE")
        if persistence_bonus > 0:
            tags.append("PERSISTENT_ACTIVITY")
        if best_match_index is None:
            tags.append("EMERGING_REGION")
        if change_score >= 25:
            tags.append("HIGH_SCORE")
        if region["aircraft_count"] > 2000:
            tags.append("HIGH_DENSITY_REGION")

        in_older = any(
            abs(region["lat"] - r["lat"]) <= _MERGE_RADIUS
            and abs(region["lon"] - r["lon"]) <= _MERGE_RADIUS
            for r in older_regions
        )
        if best_match_index is not None and in_older:
            persistence_level = "LONG"
        elif best_match_index is not None:
            persistence_level = "MEDIUM"
        else:
            persistence_level = "SHORT"

        rid = _region_id(region["lat"], region["lon"])
        cursor.execute("""
            INSERT INTO region_activity (region_id, first_seen, last_seen, total_detections)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(region_id) DO UPDATE SET
                last_seen = excluded.last_seen,
                total_detections = total_detections + 1
        """, (rid, now, now))
        first_seen, last_seen = cursor.execute(
            "SELECT first_seen, last_seen FROM region_activity WHERE region_id = ?", (rid,)
        ).fetchone()
        lifetime_minutes = round((last_seen - first_seen) / 60, 1)

        percent_military = round(region["military_count"] / region["aircraft_count"] * 100, 1)
        percent_persistent = round(min(prior_region["aircraft_count"], region["aircraft_count"]) / region["aircraft_count"] * 100, 1)
        percent_new = round(max(0, aircraft_delta) / region["aircraft_count"] * 100, 1)

        if percent_military > 30:
            region_type = "MILITARY_HEAVY"
        elif percent_military < 5:
            region_type = "CIVILIAN_HEAVY"
        else:
            region_type = "MIXED"

        row = (
            region["lat"],
            region["lon"],
            round(change_score, 1),
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
        if include_breakdown:
            row += ({
                "aircraft": round(aircraft_component, 3),
                "military": round(military_component, 3),
                "score": round(score_component, 3),
                "civilian_bonus": round(civilian_bonus, 3),
                "persistence_bonus": round(persistence_bonus, 3),
                "military_bonus": round(military_bonus, 3),
                "density_penalty": round(density_penalty, 3),
            },)
        results.append(row)

    cursor.connection.commit()

    sorted_results = sorted(results, key=lambda item: item[2], reverse=True)

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
