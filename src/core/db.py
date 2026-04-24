import sqlite3


def init_db(path="data/aircraft.db"):
    conn = sqlite3.connect(path)
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

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS aircraft_tracks (
            icao24 TEXT,
            first_seen REAL,
            last_seen REAL,
            start_lat REAL,
            start_lon REAL,
            end_lat REAL,
            end_lon REAL,
            max_distance REAL,
            PRIMARY KEY (icao24)
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_time ON aircraft_positions (timestamp)")
    conn.commit()

    return conn, cursor
