import sqlite3
import time

# ----------------------------
# Queries
# ----------------------------

def most_suspicious(cursor):
    cursor.execute("""
    SELECT callsign, type, behavior, AVG(score) as avg_score, COUNT(*) as sightings
    FROM aircraft_positions
    GROUP BY icao24
    ORDER BY avg_score DESC, sightings DESC
    LIMIT 10
    """)
    return cursor.fetchall()


def recent_activity(cursor):
    cutoff = time.time() - 600

    cursor.execute("""
    SELECT callsign, type, behavior, COUNT(*) as sightings
    FROM aircraft_positions
    WHERE timestamp > ?
    GROUP BY icao24
    ORDER BY sightings DESC
    LIMIT 10
    """, (cutoff,))

    return cursor.fetchall()


def loitering(cursor):
    cursor.execute("""
    SELECT callsign, COUNT(*) as sightings
    FROM aircraft_positions
    WHERE behavior = 'LOITERING'
    GROUP BY icao24
    ORDER BY sightings DESC
    LIMIT 10
    """)
    return cursor.fetchall()


def persistent_aircraft(cursor):
    print("\n--- Persistent Aircraft (Last 30 Minutes) ---")

    cutoff = time.time() - 1800  # 30 minutes

    cursor.execute("""
        SELECT 
            icao24,
            callsign,
            type,
            COUNT(*) as sightings,
            AVG(score) as avg_score
        FROM aircraft_positions
        WHERE timestamp > ?
        AND callsign != ''
        AND (
            type IN ('US_CARGO', 'UK_CARGO', 'TANKER')
            OR score >= 6
            OR behavior = 'LOITERING'
        )
        GROUP BY icao24
        HAVING sightings >= 3
        ORDER BY avg_score DESC, sightings DESC
        LIMIT 10;
    """, (cutoff,))

    for row in cursor.fetchall():
        print(row)

def military_cluster(cursor):
    print("\n--- Military Activity Cluster (Last 30 Minutes) ---")

    import time
    cutoff = time.time() - 1800  # 30 minutes

    cursor.execute("""
        SELECT 
            TRIM(type) as type,
            COUNT(*) as total_sightings,
            COUNT(DISTINCT icao24) as unique_aircraft
        FROM aircraft_positions
        WHERE timestamp > ?
          AND type IN ('US_CARGO', 'UK_CARGO')
        GROUP BY type
        ORDER BY total_sightings DESC;
    """, (cutoff,))

    results = cursor.fetchall()

    if not results:
        print("No military clustering detected.")
        return

    for row in results:
        print(row)    

    cutoff = time.time() - 1800  # 30 min


    for row in cursor.fetchall():
        print(row)

# ----------------------------
# Main
# ----------------------------

def main():
    import sqlite3
    import time

    conn = sqlite3.connect("data/aircraft.db")
    cursor = conn.cursor()

    cursor.execute("SELECT timestamp FROM aircraft_positions ORDER BY timestamp DESC LIMIT 5;")
    print("\n--- Latest Timestamps ---")
    print(cursor.fetchall())
    
    print("\n--- Most Suspicious ---")
    for row in most_suspicious(cursor):
        print(row)

    print("\n--- Recent Activity ---")
    for row in recent_activity(cursor):
        print(row)

    print("\n--- Loitering Aircraft ---")
    for row in loitering(cursor):
        print(row)

    persistent_aircraft(cursor)

    # 🔥 NEW LINE
    military_cluster(cursor)

    conn.close()    # ✅ Use ONE consistent database path
    conn = sqlite3.connect("data/aircraft.db")
    cursor = conn.cursor()


if __name__ == "__main__":
    main()