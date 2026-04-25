import math


def merge_regions(snapshot, radius=0.2):
    if not snapshot:
        return []

    # Pre-bucket every point into a grid cell of size `radius`.
    # Two points can only be within `radius` of each other if their
    # cells are at most 1 step apart in each dimension (9 cells max),
    # so BFS expansion drops from O(n) to O(points-in-9-cells) per node.
    cell_size = radius

    cells = [
        (int(math.floor(c["lat_bin"] / cell_size)),
         int(math.floor(c["lon_bin"] / cell_size)))
        for c in snapshot
    ]

    grid = {}
    for idx, key in enumerate(cells):
        grid.setdefault(key, []).append(idx)

    merged_regions = []
    visited = set()

    for index in range(len(snapshot)):
        if index in visited:
            continue

        pending = [index]
        member_indices = []
        visited.add(index)

        while pending:
            current_index = pending.pop()
            member_indices.append(current_index)
            current_cluster = snapshot[current_index]
            cur_lat = current_cluster["lat_bin"]
            cur_lon = current_cluster["lon_bin"]
            cx, cy = cells[current_index]

            for dcx in (-1, 0, 1):
                for dcy in (-1, 0, 1):
                    for other_index in grid.get((cx + dcx, cy + dcy), ()):
                        if other_index in visited:
                            continue
                        other_cluster = snapshot[other_index]
                        if (
                            abs(cur_lat - other_cluster["lat_bin"]) <= radius
                            and abs(cur_lon - other_cluster["lon_bin"]) <= radius
                        ):
                            visited.add(other_index)
                            pending.append(other_index)

        members = [snapshot[i] for i in member_indices]
        total_aircraft = sum(m["aircraft_count"] for m in members)
        total_military = sum(m["military_count"] for m in members)
        total_score    = sum(m["total_score"]     for m in members)

        behavior_union = set()
        for m in members:
            behavior_union.update(m["behaviors"])

        center_lat = sum(m["lat_bin"] * m["aircraft_count"] for m in members) / total_aircraft
        center_lon = sum(m["lon_bin"] * m["aircraft_count"] for m in members) / total_aircraft

        merged_regions.append({
            "lat":               round(center_lat, 1),
            "lon":               round(center_lon, 1),
            "aircraft_count":    total_aircraft,
            "military_count":    total_military,
            "avg_score":         total_score / total_aircraft if total_aircraft else 0,
            "distinct_behaviors": len(behavior_union),
        })

    return merged_regions
