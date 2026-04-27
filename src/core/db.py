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
            score INTEGER,
            lat_bin REAL,
            lon_bin REAL
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

    # Migrate existing databases: add lat_bin / lon_bin columns if absent,
    # then backfill any rows written before the migration.
    for col in ("lat_bin", "lon_bin"):
        try:
            cursor.execute(f"ALTER TABLE aircraft_positions ADD COLUMN {col} REAL")
        except sqlite3.OperationalError:
            pass  # column already exists

    cursor.execute("""
        UPDATE aircraft_positions
        SET lat_bin = ROUND(lat, 1),
            lon_bin = ROUND(lon, 1)
        WHERE lat_bin IS NULL
    """)

    # Composite index covers every query that filters on timestamp and groups
    # by (lat_bin, lon_bin).  Replaces the narrower idx_time(timestamp) index.
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_time_bins
        ON aircraft_positions (timestamp, lat_bin, lon_bin)
    """)
    conn.commit()

    return conn, cursor
