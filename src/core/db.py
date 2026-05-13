import sqlite3
from pathlib import Path

from core.monitoring_config import THEATERS


def _ensure_aircraft_position_columns(cursor):
    for col, col_type in (
        ("lat_bin", "REAL"),
        ("lon_bin", "REAL"),
        ("source_name", "TEXT"),
        ("source_tier", "TEXT"),
    ):
        try:
            cursor.execute(f"ALTER TABLE aircraft_positions ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass

    cursor.execute(
        """
        UPDATE aircraft_positions
        SET lat_bin = ROUND(lat, 1),
            lon_bin = ROUND(lon, 1),
            source_name = COALESCE(source_name, 'unknown'),
            source_tier = COALESCE(source_tier, 'unknown')
        WHERE lat_bin IS NULL
           OR lon_bin IS NULL
           OR source_name IS NULL
           OR source_tier IS NULL
        """
    )


def _ensure_aircraft_track_type(cursor):
    try:
        cursor.execute("ALTER TABLE aircraft_tracks ADD COLUMN type TEXT")
    except sqlite3.OperationalError:
        pass

    cursor.execute(
        """
        UPDATE aircraft_tracks
        SET type = COALESCE(type, 'UNKNOWN')
        WHERE type IS NULL
        """
    )


def _ensure_monitoring_tables(cursor):
    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS theaters (
            theater_id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            center_lat REAL NOT NULL,
            center_lon REAL NOT NULL,
            radius_km REAL NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS source_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT NOT NULL,
            source_tier TEXT NOT NULL,
            started_at REAL NOT NULL,
            finished_at REAL NOT NULL,
            status TEXT NOT NULL,
            latency_seconds REAL NOT NULL,
            item_count INTEGER NOT NULL,
            error_reason TEXT
        );

        CREATE TABLE IF NOT EXISTS source_health (
            source_name TEXT PRIMARY KEY,
            source_tier TEXT NOT NULL,
            last_status TEXT NOT NULL,
            last_started_at REAL NOT NULL,
            last_finished_at REAL NOT NULL,
            last_latency_seconds REAL NOT NULL,
            last_item_count INTEGER NOT NULL,
            last_error_reason TEXT,
            degraded INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS current_region_state (
            region_id TEXT PRIMARY KEY,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            theater_id TEXT NOT NULL,
            classification TEXT NOT NULL,
            confidence INTEGER NOT NULL,
            score REAL NOT NULL,
            operational_severity TEXT NOT NULL,
            escalation_direction TEXT NOT NULL,
            military_action_risk_6h REAL NOT NULL,
            military_action_risk_24h REAL NOT NULL,
            aircraft_count INTEGER NOT NULL,
            military_count INTEGER NOT NULL,
            new_aircraft INTEGER NOT NULL,
            appearance_count_6h INTEGER NOT NULL DEFAULT 1,
            appearance_count_24h INTEGER NOT NULL DEFAULT 1,
            max_aircraft_6h INTEGER NOT NULL DEFAULT 0,
            max_aircraft_24h INTEGER NOT NULL DEFAULT 0,
            max_military_6h INTEGER NOT NULL DEFAULT 0,
            max_military_24h INTEGER NOT NULL DEFAULT 0,
            max_risk_6h REAL NOT NULL DEFAULT 0,
            max_risk_24h REAL NOT NULL DEFAULT 0,
            source_summary TEXT NOT NULL,
            evidence_summary TEXT NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS current_route_state (
            route_id TEXT PRIMARY KEY,
            origin_lat REAL NOT NULL,
            origin_lon REAL NOT NULL,
            dest_lat REAL NOT NULL,
            dest_lon REAL NOT NULL,
            theater_id TEXT NOT NULL,
            aircraft_moved INTEGER NOT NULL,
            military_moved INTEGER NOT NULL,
            avg_distance REAL NOT NULL,
            flow_score REAL NOT NULL DEFAULT 0,
            operational_severity TEXT NOT NULL,
            escalation_direction TEXT NOT NULL,
            trend_direction TEXT NOT NULL DEFAULT 'rising',
            appearance_count INTEGER NOT NULL DEFAULT 1,
            source_count INTEGER NOT NULL DEFAULT 0,
            live_source_count INTEGER NOT NULL DEFAULT 0,
            experimental_source_count INTEGER NOT NULL DEFAULT 0,
            military_action_risk_6h REAL NOT NULL,
            military_action_risk_24h REAL NOT NULL,
            source_summary TEXT NOT NULL,
            evidence_summary TEXT NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS region_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            region_id TEXT NOT NULL,
            snapshot_time REAL NOT NULL,
            classification TEXT NOT NULL,
            confidence INTEGER NOT NULL,
            score REAL NOT NULL,
            operational_severity TEXT NOT NULL,
            escalation_direction TEXT NOT NULL,
            military_action_risk_6h REAL NOT NULL,
            military_action_risk_24h REAL NOT NULL,
            aircraft_count INTEGER NOT NULL,
            military_count INTEGER NOT NULL,
            new_aircraft INTEGER NOT NULL,
            appearance_count_6h INTEGER NOT NULL DEFAULT 1,
            appearance_count_24h INTEGER NOT NULL DEFAULT 1,
            max_aircraft_6h INTEGER NOT NULL DEFAULT 0,
            max_aircraft_24h INTEGER NOT NULL DEFAULT 0,
            max_military_6h INTEGER NOT NULL DEFAULT 0,
            max_military_24h INTEGER NOT NULL DEFAULT 0,
            max_risk_6h REAL NOT NULL DEFAULT 0,
            max_risk_24h REAL NOT NULL DEFAULT 0,
            source_summary TEXT NOT NULL,
            evidence_summary TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS route_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            route_id TEXT NOT NULL,
            snapshot_time REAL NOT NULL,
            aircraft_moved INTEGER NOT NULL,
            military_moved INTEGER NOT NULL,
            avg_distance REAL NOT NULL,
            flow_score REAL NOT NULL DEFAULT 0,
            operational_severity TEXT NOT NULL,
            escalation_direction TEXT NOT NULL,
            trend_direction TEXT NOT NULL DEFAULT 'rising',
            appearance_count INTEGER NOT NULL DEFAULT 1,
            source_count INTEGER NOT NULL DEFAULT 0,
            live_source_count INTEGER NOT NULL DEFAULT 0,
            experimental_source_count INTEGER NOT NULL DEFAULT 0,
            military_action_risk_6h REAL NOT NULL,
            military_action_risk_24h REAL NOT NULL,
            source_summary TEXT NOT NULL,
            evidence_summary TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS alerts (
            alert_id TEXT PRIMARY KEY,
            alert_type TEXT NOT NULL,
            object_type TEXT NOT NULL,
            object_id TEXT NOT NULL,
            theater_id TEXT NOT NULL,
            severity TEXT NOT NULL,
            priority_score REAL NOT NULL,
            confidence INTEGER NOT NULL,
            status TEXT NOT NULL,
            visible INTEGER NOT NULL DEFAULT 1,
            watchlist_match INTEGER NOT NULL DEFAULT 0,
            first_seen REAL NOT NULL,
            last_seen REAL NOT NULL,
            explanation TEXT NOT NULL,
            evidence_summary TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS alert_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id TEXT NOT NULL,
            event_time REAL NOT NULL,
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            priority_score REAL NOT NULL,
            evidence_summary TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS watchlists (
            watch_id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_type TEXT NOT NULL,
            object_id TEXT NOT NULL,
            theater_id TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        );

        CREATE TABLE IF NOT EXISTS suppression_rules (
            suppression_id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_type TEXT NOT NULL,
            object_id TEXT,
            theater_id TEXT,
            alert_type TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        );
        """
    )


def _ensure_route_monitoring_columns(cursor):
    current_route_columns = (
        ("flow_score", "REAL NOT NULL DEFAULT 0"),
        ("trend_direction", "TEXT NOT NULL DEFAULT 'rising'"),
        ("appearance_count", "INTEGER NOT NULL DEFAULT 1"),
        ("source_count", "INTEGER NOT NULL DEFAULT 0"),
        ("live_source_count", "INTEGER NOT NULL DEFAULT 0"),
        ("experimental_source_count", "INTEGER NOT NULL DEFAULT 0"),
    )
    route_snapshot_columns = current_route_columns

    for column_name, column_type in current_route_columns:
        try:
            cursor.execute(
                f"ALTER TABLE current_route_state ADD COLUMN {column_name} {column_type}"
            )
        except sqlite3.OperationalError:
            pass

    for column_name, column_type in route_snapshot_columns:
        try:
            cursor.execute(
                f"ALTER TABLE route_snapshots ADD COLUMN {column_name} {column_type}"
            )
        except sqlite3.OperationalError:
            pass


def _ensure_region_monitoring_columns(cursor):
    current_region_columns = (
        ("appearance_count_6h", "INTEGER NOT NULL DEFAULT 1"),
        ("appearance_count_24h", "INTEGER NOT NULL DEFAULT 1"),
        ("max_aircraft_6h", "INTEGER NOT NULL DEFAULT 0"),
        ("max_aircraft_24h", "INTEGER NOT NULL DEFAULT 0"),
        ("max_military_6h", "INTEGER NOT NULL DEFAULT 0"),
        ("max_military_24h", "INTEGER NOT NULL DEFAULT 0"),
        ("max_risk_6h", "REAL NOT NULL DEFAULT 0"),
        ("max_risk_24h", "REAL NOT NULL DEFAULT 0"),
    )
    region_snapshot_columns = current_region_columns

    for column_name, column_type in current_region_columns:
        try:
            cursor.execute(
                f"ALTER TABLE current_region_state ADD COLUMN {column_name} {column_type}"
            )
        except sqlite3.OperationalError:
            pass

    for column_name, column_type in region_snapshot_columns:
        try:
            cursor.execute(
                f"ALTER TABLE region_snapshots ADD COLUMN {column_name} {column_type}"
            )
        except sqlite3.OperationalError:
            pass


def _ensure_alert_policy_columns(cursor):
    for column_name, column_type in (
        ("visible", "INTEGER NOT NULL DEFAULT 1"),
        ("watchlist_match", "INTEGER NOT NULL DEFAULT 0"),
    ):
        try:
            cursor.execute(f"ALTER TABLE alerts ADD COLUMN {column_name} {column_type}")
        except sqlite3.OperationalError:
            pass


def _seed_theaters(cursor):
    for theater in THEATERS:
        cursor.execute(
            """
            INSERT INTO theaters (theater_id, label, center_lat, center_lon, radius_km, active)
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(theater_id) DO UPDATE SET
                label = excluded.label,
                center_lat = excluded.center_lat,
                center_lon = excluded.center_lon,
                radius_km = excluded.radius_km,
                active = excluded.active
            """,
            (
                theater["theater_id"],
                theater["label"],
                theater["center_lat"],
                theater["center_lon"],
                theater["radius_km"],
            ),
        )


def resolve_db_path(path: str | Path | None = None) -> Path:
    if path is None:
        return Path(__file__).resolve().parents[1] / "data" / "aircraft.db"
    return Path(path)


def init_db(path: str | Path | None = None):
    db_path = resolve_db_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
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
            lon_bin REAL,
            source_name TEXT,
            source_tier TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS aircraft_tracks (
            icao24 TEXT,
            first_seen REAL,
            last_seen REAL,
            start_lat REAL,
            start_lon REAL,
            end_lat REAL,
            end_lon REAL,
            max_distance REAL,
            type TEXT,
            PRIMARY KEY (icao24)
        )
        """
    )

    _ensure_aircraft_position_columns(cursor)
    _ensure_aircraft_track_type(cursor)
    _ensure_monitoring_tables(cursor)
    _ensure_region_monitoring_columns(cursor)
    _ensure_route_monitoring_columns(cursor)
    _ensure_alert_policy_columns(cursor)
    _seed_theaters(cursor)

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_time_bins
        ON aircraft_positions (timestamp, lat_bin, lon_bin)
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_region_snapshots_region_time
        ON region_snapshots (region_id, snapshot_time)
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_route_snapshots_route_time
        ON route_snapshots (route_id, snapshot_time)
        """
    )
    conn.commit()

    return conn, cursor
