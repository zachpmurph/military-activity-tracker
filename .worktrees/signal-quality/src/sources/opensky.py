import requests
from .base import normalize_aircraft

URL = "https://opensky-network.org/api/states/all"

def fetch():
    try:
        response = requests.get(
            URL,
            auth=("zmurph3", "Aku4172007!"),
            timeout=10
        )

        if response.status_code != 200:
            print("OpenSky status:", response.status_code)
            return []
        
        if response.status_code == 429:
            print("Rate limited. Waiting...")
            time.sleep(120)
            return []

        data = response.json()
        states = data.get("states", [])

        aircraft_list = []

        for s in states:
            if s is None:
                continue

            raw = {
                "icao24": s[0],
                "callsign": s[1],
                "lon": s[5],
                "lat": s[6],
                "altitude": s[7],
                "velocity": s[9]
            }

            if raw["lat"] is None or raw["lon"] is None:
                continue

            aircraft = normalize_aircraft(raw)
            aircraft_list.append(aircraft)

        return aircraft_list

    except Exception as e:
        print("OpenSky error:", e)
        return []