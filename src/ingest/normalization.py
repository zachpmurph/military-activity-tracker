import time
from collections import defaultdict

from core.config import classify_aircraft

history = defaultdict(list)


def update_history(aircraft):
    now = time.time()

    for a in aircraft:
        history[a["icao24"]].append({
            "time": now,
            "lat": a["lat"],
            "lon": a["lon"]
        })

        history[a["icao24"]] = history[a["icao24"]][-20:]


def detect_behavior(a):
    classification = classify_aircraft(a)

    if classification == "CIVILIAN":
        return "NORMAL"

    altitude = a.get("altitude")
    speed = a.get("velocity")
    points = history[a["icao24"]]

    if len(points) < 3:
        return "INSUFFICIENT_DATA"

    lats = [p["lat"] for p in points]
    lons = [p["lon"] for p in points]

    lat_range = max(lats) - min(lats)
    lon_range = max(lons) - min(lons)

    duration = points[-1]["time"] - points[0]["time"]

    if (
        duration > 900 and
        lat_range < 0.02 and
        lon_range < 0.02 and
        altitude and altitude > 20000 and
        speed and speed < 300
    ):
        return "LOITERING"
    if a["altitude"] and a["altitude"] > 20000:
        return "LOITERING"
    if lat_range > 1 or lon_range > 1:
        return "TRANSIT"

    return "NORMAL"
