import time
import sqlite3
from collections import defaultdict
from turtle import speed
from sources.adsb import fetch as fetch_adsb
from sources.opensky import fetch as fetch_opensky

# --- DATABASE SETUP ---
conn = sqlite3.connect("data/aircraft.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS aircraft_positions (
    icao24 TEXT,
    callsign TEXT,
    lat REAL,
    lon REAL,
    altitude REAL,
    speed REAL,
    timestamp REAL,
    type TEXT,
    behavior TEXT,
    score INTEGER
)
""")

CIVILIAN_PREFIXES = (
    # Major US
    "AAL","UAL","DAL","SWA","FFT","JBU","ASA","SKW",

    # Cargo
    "FDX","UPS",

    # Europe
    "DLH","BAW","AFR","KLM","RYR","EZY","VLG",

    # Middle East / Asia
    "QTR","UAE","ETD","SIA","ANA","JAL",

    # Canada
    "ACA","WJA","JZA",

    # Latin America
    "AMX","CMP","BWA","RAM", 
    "WZZ","EZY","RYR","VLG","IBE","SAS",
    "RPA","SKW","ENY","ASH",
    "CFE","BAW","DLH","KLM",

    "ASA","EJU","WZZ","RPA","CFE",
    "SWR","ITY","SVA","BAW","DLH",
    "AFR","KLM","EZY","RYR"
)

cursor.execute("CREATE INDEX IF NOT EXISTS idx_time ON aircraft_positions (timestamp)")
conn.commit()

# --- MEMORY TRACKING ---
history = defaultdict(list)

# --- CLASSIFICATION ---
def classify_aircraft(a):
    callsign = a["callsign"]

    # --- CIVILIAN AIRLINES ---
    if callsign.startswith(CIVILIAN_PREFIXES):
        return "CIVILIAN"

    # --- MILITARY SIGNALS ---
    if callsign.startswith("RCH"):
        return "US_CARGO"

    if callsign.startswith(("QID", "K35R", "DRAG")):
        return "TANKER"

    if callsign.startswith("RRR"):
        return "UK_CARGO"

    return "UNKNOWN"  

# --- HISTORY ---
def update_history(aircraft):
    now = time.time()

    for a in aircraft:
        history[a["icao24"]].append({
            "time": now,
            "lat": a["lat"],
            "lon": a["lon"]
        })

        history[a["icao24"]] = history[a["icao24"]][-20:]

# --- BEHAVIOR ---
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

# --- SCORING ---
def compute_score(a, behavior):
    source_count = len(a.get("sources", []))
    callsign = a["callsign"]
    altitude = a["altitude"] or 0
    speed = a["velocity"] or 0

    score = 0
    score += source_count
    classification = classify_aircraft(a)

    if classification == "CIVILIAN":
        score -= 2
    elif classification == "UNKNOWN":
        score += 0
    if classification in ("US_CARGO", "TANKER", "UK_CARGO"):
        score += 2
    if not callsign:
        score -= 2

    if callsign:
        score += 1

    if altitude > 25000:
        score += 1

    if altitude > 30000 and speed < 250:
        score += 2

    if behavior == "LOITERING":
        score += 2

    if behavior == "INSUFFICIENT_DATA":
        score -= 1

    return score

# --- STORE ---
def store_aircraft(aircraft_list):
    now = time.time()

    for a in aircraft_list:
        classification = classify_aircraft(a)
        behavior = detect_behavior(a)
        score = compute_score(a, behavior)

        if score < 2:
            continue

        cursor.execute("""
        INSERT INTO aircraft_positions
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            a["icao24"],
            a["callsign"],
            a["lat"],
            a["lon"],
            a["altitude"],
            a["velocity"],
            now,
            classification,
            behavior,
            score
        ))

    conn.commit()

# --- FUSION ---
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

                # Merge intelligently
                for key in ["callsign", "lat", "lon", "altitude", "velocity"]:
                    if a.get(key) and not existing.get(key):
                        existing[key] = a[key]

                # Track sources
                existing["sources"].add(a.get("source"))

    return list(fused.values()) 

    for source in sources:
        for a in source:
            hex_id = a["icao24"]

            if hex_id not in fused:
                fused[hex_id] = a
            else:
                for key in a:
                    if a[key] and not fused[hex_id].get(key):
                        fused[hex_id][key] = a[key]

    return list(fused.values())

# --- MAIN LOOP ---
while True:
    adsb_data = fetch_adsb()
    aircraft = fuse_data([adsb_data])

    update_history(aircraft)
    store_aircraft(aircraft)

    print(f"\n--- Retrieved {len(aircraft)} aircraft (after fusion) ---")  
    time.sleep(30)