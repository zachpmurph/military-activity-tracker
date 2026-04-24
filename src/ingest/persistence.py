import time

from core.config import classify_aircraft
from geo.distance import euclidean_distance
from ingest.normalization import detect_behavior
from scoring.scoring import compute_score

conn = None
cursor = None


def update_aircraft_track(a, now):
    icao24 = a.get("icao24")
    lat = a.get("lat")
    lon = a.get("lon")

    if not icao24 or lat is None or lon is None:
        return

    existing_track = cursor.execute("""
        SELECT first_seen, start_lat, start_lon, max_distance
        FROM aircraft_tracks
        WHERE icao24 = ?
    """, (icao24,)).fetchone()

    if not existing_track:
        cursor.execute("""
            INSERT INTO aircraft_tracks
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (icao24, now, now, lat, lon, lat, lon, 0.0))
        return

    first_seen, start_lat, start_lon, max_distance = existing_track
    distance = euclidean_distance(start_lat, start_lon, lat, lon)

    cursor.execute("""
        UPDATE aircraft_tracks
        SET last_seen = ?,
            end_lat = ?,
            end_lon = ?,
            max_distance = ?
        WHERE icao24 = ?
    """, (now, lat, lon, max(max_distance, distance), icao24))


def cleanup_old_tracks(now):
    cutoff = now - (6 * 3600)
    cursor.execute("""
        DELETE FROM aircraft_tracks
        WHERE last_seen < ?
    """, (cutoff,))


def store_aircraft(aircraft_list):
    now = time.time()

    for a in aircraft_list:
        update_aircraft_track(a, now)

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

    cleanup_old_tracks(now)
    conn.commit()
