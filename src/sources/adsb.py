import requests
from .base import normalize_aircraft

URLS = [
    # USA West (Seattle)
    "https://api.adsb.lol/v2/lat/47.6/lon/-122.3/dist/300",

    # USA Central
    "https://api.adsb.lol/v2/lat/39.0/lon/-95.0/dist/500",

    # USA East
    "https://api.adsb.lol/v2/lat/40.7/lon/-74.0/dist/300",

    # Europe
    "https://api.adsb.lol/v2/lat/50.0/lon/10.0/dist/600",

    # Middle East (very important region)
    "https://api.adsb.lol/v2/lat/25.0/lon/45.0/dist/600",

    # Pacific (Guam / strategic area)
    "https://api.adsb.lol/v2/lat/13.5/lon/144.8/dist/800",
]

def fetch():
    all_aircraft = []

    for url in URLS:
        try:
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                print(f"ADSB failed ({url}):", response.status_code)
                continue

            data = response.json()
            aircraft_list = data.get("ac") or data.get("aircraft") or []

            for a in aircraft_list:
                raw = {
                    "source": url,
                    "icao24": a.get("hex"),
                    "callsign": (a.get("flight") or "").strip(),
                    "lat": a.get("lat"),
                    "lon": a.get("lon"),
                    "altitude": a.get("alt_baro"),
                    "velocity": a.get("gs")
                }

                if raw["lat"] is None or raw["lon"] is None:
                    continue

                all_aircraft.append(normalize_aircraft(raw))

            print(f"ADSB region success: {len(aircraft_list)} from {url}")

        except Exception as e:
            print("ADSB error:", e)

    return all_aircraft 