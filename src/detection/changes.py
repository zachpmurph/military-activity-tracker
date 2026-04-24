import time

from geo.clustering import merge_regions


def detect_activity_changes(cursor):
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

    recent_regions = merge_regions(fetch_snapshot(recent_cutoff))
    prior_regions = merge_regions(fetch_snapshot(prior_cutoff, recent_cutoff))
    unmatched_prior = set(range(len(prior_regions)))
    results = []

    for region in sorted(recent_regions, key=lambda item: item["aircraft_count"], reverse=True):
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
        change_score = (
            (aircraft_delta * 1.0) +
            (military_delta * 2.5) +
            (score_delta * 1.5)
        )

        if change_score < 3 and not emerging_flag:
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

        results.append((
            region["lat"],
            region["lon"],
            round(change_score, 1),
            aircraft_delta,
            military_delta,
            round(score_delta, 1),
            ",".join(flags),
        ))

    for row in sorted(results, key=lambda item: item[2], reverse=True):
        print(row)
