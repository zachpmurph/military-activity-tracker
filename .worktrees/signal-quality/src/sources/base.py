# src/sources/base.py

def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_aircraft(raw):
    return {
        "icao24": raw.get("icao24"),
        "callsign": (raw.get("callsign") or "").strip(),
        "lat": to_float(raw.get("lat")),
        "lon": to_float(raw.get("lon")),
        "altitude": to_float(raw.get("altitude")),
        "velocity": to_float(raw.get("velocity")),
        "source": raw.get("source"),
    }