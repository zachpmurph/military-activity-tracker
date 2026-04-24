def merge_regions(snapshot, radius=0.2):
    merged_regions = []
    visited = set()

    for index, cluster in enumerate(snapshot):
        if index in visited:
            continue

        pending = [index]
        member_indices = []
        visited.add(index)

        while pending:
            current_index = pending.pop()
            member_indices.append(current_index)
            current_cluster = snapshot[current_index]

            for other_index, other_cluster in enumerate(snapshot):
                if other_index in visited:
                    continue

                if (
                    abs(current_cluster["lat_bin"] - other_cluster["lat_bin"]) <= radius
                    and abs(current_cluster["lon_bin"] - other_cluster["lon_bin"]) <= radius
                ):
                    visited.add(other_index)
                    pending.append(other_index)

        members = [snapshot[i] for i in member_indices]
        total_aircraft = sum(member["aircraft_count"] for member in members)
        total_military = sum(member["military_count"] for member in members)
        total_score = sum(member["total_score"] for member in members)
        behavior_union = set()

        for member in members:
            behavior_union.update(member["behaviors"])

        center_lat = sum(member["lat_bin"] * member["aircraft_count"] for member in members) / total_aircraft
        center_lon = sum(member["lon_bin"] * member["aircraft_count"] for member in members) / total_aircraft

        merged_regions.append({
            "lat": round(center_lat, 1),
            "lon": round(center_lon, 1),
            "aircraft_count": total_aircraft,
            "military_count": total_military,
            "avg_score": total_score / total_aircraft if total_aircraft else 0,
            "distinct_behaviors": len(behavior_union),
        })

    return merged_regions
