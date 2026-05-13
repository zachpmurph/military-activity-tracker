def fuse_data(sources):
    fused = {}

    for source in sources:
        for a in source:
            hex_id = a["icao24"]

            if not hex_id:
                continue

            if hex_id not in fused:
                fused[hex_id] = a.copy()
                fused[hex_id]["sources"] = {a.get("source")}
            else:
                existing = fused[hex_id]

                for key in ["callsign", "lat", "lon", "altitude", "velocity"]:
                    if a.get(key) and not existing.get(key):
                        existing[key] = a[key]

                existing["sources"].add(a.get("source"))

    return list(fused.values())
