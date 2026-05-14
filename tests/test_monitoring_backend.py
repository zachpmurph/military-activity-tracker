import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.db import init_db
from intelligence.classifier import Classification, RegionFeatures, RegionIntelligence
from intelligence.monitoring_backend import (
    export_monitoring_snapshot,
    persist_monitoring_state,
    persist_source_runs,
)


class MonitoringSchemaTests(unittest.TestCase):
    def test_init_db_creates_monitoring_tables_and_track_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "monitoring.db"
            conn, cursor = init_db(str(db_path))

            tables = {
                row[0]
                for row in cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            self.assertTrue(
                {
                    "theaters",
                    "source_runs",
                    "source_health",
                    "current_region_state",
                    "current_route_state",
                    "region_snapshots",
                    "route_snapshots",
                    "alerts",
                    "alert_events",
                    "watchlists",
                    "suppression_rules",
                }.issubset(tables)
            )

            track_columns = {
                row[1]
                for row in cursor.execute("PRAGMA table_info(aircraft_tracks)").fetchall()
            }
            self.assertIn("type", track_columns)

            position_columns = {
                row[1]
                for row in cursor.execute("PRAGMA table_info(aircraft_positions)").fetchall()
            }
            self.assertIn("source_name", position_columns)
            self.assertIn("source_tier", position_columns)

            region_columns = {
                row[1]
                for row in cursor.execute("PRAGMA table_info(current_region_state)").fetchall()
            }
            self.assertTrue(
                {
                    "appearance_count_6h",
                    "appearance_count_24h",
                    "max_aircraft_6h",
                    "max_aircraft_24h",
                    "max_military_6h",
                    "max_military_24h",
                    "max_risk_6h",
                    "max_risk_24h",
                }.issubset(region_columns)
            )

            route_columns = {
                row[1]
                for row in cursor.execute("PRAGMA table_info(current_route_state)").fetchall()
            }
            self.assertTrue(
                {
                    "flow_score",
                    "trend_direction",
                    "appearance_count",
                    "source_count",
                    "live_source_count",
                    "experimental_source_count",
                }.issubset(route_columns)
            )

            alert_columns = {
                row[1]
                for row in cursor.execute("PRAGMA table_info(alerts)").fetchall()
            }
            self.assertTrue({"visible", "watchlist_match"}.issubset(alert_columns))
            conn.close()


class MonitoringPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.cursor = self.conn.cursor()
        self.cursor.executescript(
            """
            CREATE TABLE theaters (
                theater_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                center_lat REAL NOT NULL,
                center_lon REAL NOT NULL,
                radius_km REAL NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE source_runs (
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
            CREATE TABLE source_health (
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
            CREATE TABLE current_region_state (
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
                appearance_count_6h INTEGER NOT NULL,
                appearance_count_24h INTEGER NOT NULL,
                max_aircraft_6h INTEGER NOT NULL,
                max_aircraft_24h INTEGER NOT NULL,
                max_military_6h INTEGER NOT NULL,
                max_military_24h INTEGER NOT NULL,
                max_risk_6h REAL NOT NULL,
                max_risk_24h REAL NOT NULL,
                source_summary TEXT NOT NULL,
                evidence_summary TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE current_route_state (
                route_id TEXT PRIMARY KEY,
                origin_lat REAL NOT NULL,
                origin_lon REAL NOT NULL,
                dest_lat REAL NOT NULL,
                dest_lon REAL NOT NULL,
                theater_id TEXT NOT NULL,
                aircraft_moved INTEGER NOT NULL,
                military_moved INTEGER NOT NULL,
                avg_distance REAL NOT NULL,
                flow_score REAL NOT NULL,
                operational_severity TEXT NOT NULL,
                escalation_direction TEXT NOT NULL,
                trend_direction TEXT NOT NULL,
                appearance_count INTEGER NOT NULL,
                source_count INTEGER NOT NULL,
                live_source_count INTEGER NOT NULL,
                experimental_source_count INTEGER NOT NULL,
                military_action_risk_6h REAL NOT NULL,
                military_action_risk_24h REAL NOT NULL,
                source_summary TEXT NOT NULL,
                evidence_summary TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE region_snapshots (
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
                appearance_count_6h INTEGER NOT NULL,
                appearance_count_24h INTEGER NOT NULL,
                max_aircraft_6h INTEGER NOT NULL,
                max_aircraft_24h INTEGER NOT NULL,
                max_military_6h INTEGER NOT NULL,
                max_military_24h INTEGER NOT NULL,
                max_risk_6h REAL NOT NULL,
                max_risk_24h REAL NOT NULL,
                source_summary TEXT NOT NULL,
                evidence_summary TEXT NOT NULL
            );
            CREATE TABLE route_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                route_id TEXT NOT NULL,
                snapshot_time REAL NOT NULL,
                aircraft_moved INTEGER NOT NULL,
                military_moved INTEGER NOT NULL,
                avg_distance REAL NOT NULL,
                flow_score REAL NOT NULL,
                operational_severity TEXT NOT NULL,
                escalation_direction TEXT NOT NULL,
                trend_direction TEXT NOT NULL,
                appearance_count INTEGER NOT NULL,
                source_count INTEGER NOT NULL,
                live_source_count INTEGER NOT NULL,
                experimental_source_count INTEGER NOT NULL,
                military_action_risk_6h REAL NOT NULL,
                military_action_risk_24h REAL NOT NULL,
                source_summary TEXT NOT NULL,
                evidence_summary TEXT NOT NULL
            );
            CREATE TABLE alerts (
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
            CREATE TABLE alert_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT NOT NULL,
                event_time REAL NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                priority_score REAL NOT NULL,
                evidence_summary TEXT NOT NULL
            );
            CREATE TABLE watchlists (
                watch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_type TEXT NOT NULL,
                object_id TEXT NOT NULL,
                theater_id TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE suppression_rules (
                suppression_id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_type TEXT NOT NULL,
                object_id TEXT,
                theater_id TEXT,
                alert_type TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL DEFAULT 0
            );
            """
        )
        self.cursor.execute(
            """
            INSERT INTO theaters (theater_id, label, center_lat, center_lon, radius_km, active)
            VALUES ('middle_east', 'Middle East', 25.0, 45.0, 900.0, 1)
            """
        )
        self.conn.commit()

    def tearDown(self):
        self.conn.close()

    def _region_intelligence(self):
        features = RegionFeatures(
            lat=25.0,
            lon=45.0,
            aircraft_count=8,
            military_count=4,
            recurring_appearances=4,
            spike_flag=True,
            coordination_flag=True,
            staging_flag=True,
            new_aircraft=5,
            change_score=26.0,
            change_level="HIGH",
        )
        return RegionIntelligence(
            lat=25.0,
            lon=45.0,
            classification=Classification.STAGING,
            confidence=84,
            score=74.0,
            all_scores={
                "STAGING": 74.0,
                "PROJECTION": 28.0,
                "ROUTINE": 6.0,
                "ANOMALY": 58.0,
                "COORDINATED_ACTIVITY": 61.0,
            },
            features=features,
            explanation="persistent military buildup with outbound movement",
        )

    def _low_signal_region_intelligence(self):
        features = RegionFeatures(
            lat=39.0,
            lon=-95.0,
            aircraft_count=3,
            military_count=0,
            recurring_appearances=1,
            new_aircraft=2,
            change_score=2.0,
            change_level="LOW",
        )
        return RegionIntelligence(
            lat=39.0,
            lon=-95.0,
            classification=Classification.ANOMALY,
            confidence=52,
            score=18.0,
            all_scores={
                "STAGING": 5.0,
                "PROJECTION": 6.0,
                "ROUTINE": 12.0,
                "ANOMALY": 18.0,
                "COORDINATED_ACTIVITY": 4.0,
            },
            features=features,
            explanation="light new-entry activity with low confidence",
        )

    def _region_intelligence_with_risk_profile(
        self,
        *,
        lat,
        lon,
        aircraft_count,
        military_count,
        recurring_appearances,
        new_aircraft,
        change_score,
        staging_score,
        projection_score,
        coordinated_score,
        anomaly_score,
        confidence,
        explanation,
    ):
        features = RegionFeatures(
            lat=lat,
            lon=lon,
            aircraft_count=aircraft_count,
            military_count=military_count,
            recurring_appearances=recurring_appearances,
            spike_flag=True,
            coordination_flag=True,
            staging_flag=True,
            new_aircraft=new_aircraft,
            change_score=change_score,
            change_level="HIGH" if change_score >= 16.0 else "MEDIUM",
        )
        return RegionIntelligence(
            lat=lat,
            lon=lon,
            classification=Classification.STAGING,
            confidence=confidence,
            score=staging_score,
            all_scores={
                "STAGING": staging_score,
                "PROJECTION": projection_score,
                "ROUTINE": 6.0,
                "ANOMALY": anomaly_score,
                "COORDINATED_ACTIVITY": coordinated_score,
            },
            features=features,
            explanation=explanation,
        )

    def test_persist_source_runs_updates_source_health(self):
        reports = [
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "started_at": 10.0,
                "finished_at": 12.5,
                "status": "success",
                "latency_seconds": 2.5,
                "item_count": 42,
                "error_reason": "",
            },
            {
                "source_name": "aisstream",
                "source_tier": "primary_live",
                "started_at": 10.0,
                "finished_at": 10.1,
                "status": "missing_credentials",
                "latency_seconds": 0.1,
                "item_count": 0,
                "error_reason": "AIS_API_KEY not set",
            },
        ]

        persist_source_runs(self.cursor, reports)

        self.assertEqual(
            self.cursor.execute("SELECT COUNT(*) FROM source_runs").fetchone()[0],
            2,
        )
        adsb_health = self.cursor.execute(
            "SELECT last_status, degraded FROM source_health WHERE source_name = 'adsb'"
        ).fetchone()
        ais_health = self.cursor.execute(
            "SELECT last_status, degraded FROM source_health WHERE source_name = 'aisstream'"
        ).fetchone()
        self.assertEqual(adsb_health, ("success", 0))
        self.assertEqual(ais_health, ("missing_credentials", 1))

    def test_persist_monitoring_state_upserts_current_rows_and_deduplicates_alerts(self):
        intel = [self._region_intelligence()]
        routes = [(25.0, 45.0, 26.0, 46.0, 3, 2, 1.4)]
        reports = [
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "started_at": 100.0,
                "finished_at": 101.0,
                "status": "success",
                "latency_seconds": 1.0,
                "item_count": 8,
                "error_reason": "",
            }
        ]

        persist_monitoring_state(self.cursor, intel, routes, reports, snapshot_time=100.0)
        persist_monitoring_state(self.cursor, intel, routes, reports, snapshot_time=200.0)

        self.assertEqual(
            self.cursor.execute("SELECT COUNT(*) FROM current_region_state").fetchone()[0],
            1,
        )
        self.assertEqual(
            self.cursor.execute("SELECT COUNT(*) FROM current_route_state").fetchone()[0],
            1,
        )
        self.assertEqual(
            self.cursor.execute("SELECT COUNT(*) FROM region_snapshots").fetchone()[0],
            2,
        )
        self.assertEqual(
            self.cursor.execute("SELECT COUNT(*) FROM route_snapshots").fetchone()[0],
            2,
        )
        self.assertEqual(
            self.cursor.execute(
                """
                SELECT COUNT(*) FROM alerts
                WHERE object_type = 'region' AND alert_type = 'STAGING_REGION'
                """
            ).fetchone()[0],
            1,
        )
        self.assertGreaterEqual(
            self.cursor.execute("SELECT COUNT(*) FROM alert_events").fetchone()[0],
            2,
        )

    def test_persist_monitoring_state_reuses_region_id_when_center_shifts_slightly(self):
        reports = [
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "started_at": 100.0,
                "finished_at": 101.0,
                "status": "success",
                "latency_seconds": 1.0,
                "item_count": 8,
                "error_reason": "",
            }
        ]
        first_region = self._region_intelligence()
        shifted_features = RegionFeatures(
            lat=25.2,
            lon=45.2,
            aircraft_count=9,
            military_count=4,
            recurring_appearances=4,
            spike_flag=True,
            coordination_flag=True,
            staging_flag=True,
            new_aircraft=5,
            change_score=27.0,
            change_level="HIGH",
        )
        shifted_region = RegionIntelligence(
            lat=25.2,
            lon=45.2,
            classification=Classification.STAGING,
            confidence=85,
            score=75.0,
            all_scores={
                "STAGING": 75.0,
                "PROJECTION": 27.0,
                "ROUTINE": 6.0,
                "ANOMALY": 59.0,
                "COORDINATED_ACTIVITY": 62.0,
            },
            features=shifted_features,
            explanation="persistent military buildup shifted slightly east",
        )

        persist_monitoring_state(self.cursor, [first_region], [], reports, snapshot_time=100.0)
        first_region_id = self.cursor.execute(
            "SELECT region_id FROM current_region_state"
        ).fetchone()[0]

        persist_monitoring_state(self.cursor, [shifted_region], [], reports, snapshot_time=200.0)
        region_rows = self.cursor.execute(
            "SELECT region_id, lat, lon FROM current_region_state"
        ).fetchall()

        self.assertEqual(len(region_rows), 1)
        self.assertEqual(region_rows[0][0], first_region_id)
        self.assertEqual(region_rows[0][1:], (25.2, 45.2))

    def test_persist_monitoring_state_tracks_region_rollups_for_6h_and_24h_windows(self):
        reports = [
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "started_at": 100.0,
                "finished_at": 101.0,
                "status": "success",
                "latency_seconds": 1.0,
                "item_count": 8,
                "error_reason": "",
            }
        ]
        first_region = self._region_intelligence()
        second_features = RegionFeatures(
            lat=25.1,
            lon=45.1,
            aircraft_count=10,
            military_count=5,
            recurring_appearances=4,
            spike_flag=True,
            coordination_flag=True,
            staging_flag=True,
            new_aircraft=6,
            change_score=29.0,
            change_level="HIGH",
        )
        second_region = RegionIntelligence(
            lat=25.1,
            lon=45.1,
            classification=Classification.STAGING,
            confidence=86,
            score=76.0,
            all_scores={
                "STAGING": 76.0,
                "PROJECTION": 28.0,
                "ROUTINE": 6.0,
                "ANOMALY": 60.0,
                "COORDINATED_ACTIVITY": 63.0,
            },
            features=second_features,
            explanation="persistent military buildup growing in place",
        )

        persist_monitoring_state(self.cursor, [first_region], [], reports, snapshot_time=1_000.0)
        persist_monitoring_state(self.cursor, [second_region], [], reports, snapshot_time=2_000.0)

        rollup_row = self.cursor.execute(
            """
            SELECT appearance_count_6h, appearance_count_24h,
                   max_aircraft_6h, max_aircraft_24h,
                   max_military_6h, max_military_24h,
                   max_risk_6h, max_risk_24h
            FROM current_region_state
            """
        ).fetchone()
        self.assertEqual(rollup_row, (2, 2, 10, 10, 5, 5, 86.5, 100.0))

        snapshot_rollups = self.cursor.execute(
            """
            SELECT appearance_count_6h, appearance_count_24h,
                   max_aircraft_6h, max_military_6h
            FROM region_snapshots
            ORDER BY snapshot_time
            """
        ).fetchall()
        self.assertEqual(snapshot_rollups, [(1, 1, 8, 4), (2, 2, 10, 5)])

    def test_persist_monitoring_state_accepts_flow_route_rows_from_aircraft_tracks(self):
        intel = [self._region_intelligence()]
        flow_routes = [(25.0, 45.0, 27.0, 47.0, 4, 2, 1.6, 9.0)]
        reports = [
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "started_at": 150.0,
                "finished_at": 151.0,
                "status": "success",
                "latency_seconds": 1.0,
                "item_count": 12,
                "error_reason": "",
            }
        ]

        persist_monitoring_state(self.cursor, intel, flow_routes, reports, snapshot_time=150.0)

        route_row = self.cursor.execute(
            """
            SELECT aircraft_moved, military_moved, avg_distance, flow_score,
                   trend_direction, appearance_count, source_count,
                   live_source_count, experimental_source_count
            FROM current_route_state
            """
        ).fetchone()
        self.assertEqual(route_row, (4, 2, 1.6, 9.0, "rising", 1, 1, 1, 0))

        route_alert_types = {
            row[0]
            for row in self.cursor.execute(
                "SELECT alert_type FROM alerts WHERE object_type = 'route'"
            ).fetchall()
        }
        self.assertIn("MAJOR_FLOW_ROUTE", route_alert_types)

    def test_persist_monitoring_state_tracks_route_trend_and_appearance_count(self):
        intel = [self._region_intelligence()]
        reports = [
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "started_at": 100.0,
                "finished_at": 101.0,
                "status": "success",
                "latency_seconds": 1.0,
                "item_count": 8,
                "error_reason": "",
            },
            {
                "source_name": "satellite",
                "source_tier": "experimental",
                "started_at": 100.0,
                "finished_at": 102.0,
                "status": "success",
                "latency_seconds": 2.0,
                "item_count": 2,
                "error_reason": "",
            },
        ]

        persist_monitoring_state(
            self.cursor,
            intel,
            [(25.0, 45.0, 27.0, 47.0, 2, 1, 1.2, 4.5)],
            reports,
            snapshot_time=100.0,
        )
        persist_monitoring_state(
            self.cursor,
            intel,
            [(25.0, 45.0, 27.0, 47.0, 5, 3, 1.8, 12.5)],
            reports,
            snapshot_time=200.0,
        )

        route_row = self.cursor.execute(
            """
            SELECT flow_score, trend_direction, appearance_count, source_count,
                   live_source_count, experimental_source_count
            FROM current_route_state
            """
        ).fetchone()
        self.assertEqual(route_row, (12.5, "strengthening", 2, 2, 1, 1))

        snapshot_rows = self.cursor.execute(
            """
            SELECT flow_score, trend_direction, appearance_count
            FROM route_snapshots
            ORDER BY snapshot_time
            """
        ).fetchall()
        self.assertEqual(
            snapshot_rows,
            [(4.5, "rising", 1), (12.5, "strengthening", 2)],
        )

    def test_route_evidence_includes_named_live_and_experimental_sources(self):
        intel = [self._region_intelligence()]
        reports = [
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "started_at": 100.0,
                "finished_at": 101.0,
                "status": "success",
                "latency_seconds": 1.0,
                "item_count": 8,
                "error_reason": "",
            },
            {
                "source_name": "gfw",
                "source_tier": "secondary_live",
                "started_at": 100.0,
                "finished_at": 102.0,
                "status": "success",
                "latency_seconds": 2.0,
                "item_count": 3,
                "error_reason": "",
            },
            {
                "source_name": "satellite",
                "source_tier": "experimental",
                "started_at": 100.0,
                "finished_at": 103.0,
                "status": "success",
                "latency_seconds": 3.0,
                "item_count": 1,
                "error_reason": "",
            },
            {
                "source_name": "aisstream",
                "source_tier": "primary_live",
                "started_at": 100.0,
                "finished_at": 104.0,
                "status": "missing_credentials",
                "latency_seconds": 4.0,
                "item_count": 0,
                "error_reason": "AIS_API_KEY not set",
            },
        ]

        persist_monitoring_state(
            self.cursor,
            intel,
            [(25.0, 45.0, 27.0, 47.0, 4, 2, 1.6, 9.0)],
            reports,
            snapshot_time=100.0,
        )

        evidence_summary = self.cursor.execute(
            "SELECT evidence_summary FROM current_route_state"
        ).fetchone()[0]
        evidence = json.loads(evidence_summary)

        self.assertEqual(
            evidence["source_provenance"]["live_source_names"],
            ["adsb", "gfw"],
        )
        self.assertEqual(
            evidence["source_provenance"]["experimental_source_names"],
            ["satellite"],
        )
        self.assertEqual(
            evidence["source_provenance"]["source_names"],
            ["adsb", "gfw", "satellite"],
        )

    def test_route_trend_history_covers_stable_and_cooling(self):
        intel = [self._region_intelligence()]
        reports = [
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "started_at": 100.0,
                "finished_at": 101.0,
                "status": "success",
                "latency_seconds": 1.0,
                "item_count": 8,
                "error_reason": "",
            }
        ]

        persist_monitoring_state(
            self.cursor,
            intel,
            [(25.0, 45.0, 27.0, 47.0, 3, 1, 1.3, 7.0)],
            reports,
            snapshot_time=100.0,
        )
        persist_monitoring_state(
            self.cursor,
            intel,
            [(25.0, 45.0, 27.0, 47.0, 3, 1, 1.4, 8.5)],
            reports,
            snapshot_time=200.0,
        )
        persist_monitoring_state(
            self.cursor,
            intel,
            [(25.0, 45.0, 27.0, 47.0, 2, 1, 1.1, 5.5)],
            reports,
            snapshot_time=300.0,
        )

        snapshot_rows = self.cursor.execute(
            """
            SELECT snapshot_time, trend_direction, appearance_count
            FROM route_snapshots
            ORDER BY snapshot_time
            """
        ).fetchall()
        self.assertEqual(
            snapshot_rows,
            [
                (100.0, "rising", 1),
                (200.0, "stable", 2),
                (300.0, "cooling", 3),
            ],
        )

    def test_route_risk_increases_when_same_flow_repeats(self):
        intel = [self._region_intelligence()]
        reports = [
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "started_at": 100.0,
                "finished_at": 101.0,
                "status": "success",
                "latency_seconds": 1.0,
                "item_count": 8,
                "error_reason": "",
            }
        ]
        route_row = (25.0, 45.0, 27.0, 47.0, 3, 1, 1.3, 7.0)

        persist_monitoring_state(self.cursor, intel, [route_row], reports, snapshot_time=100.0)
        first_risks = self.cursor.execute(
            """
            SELECT military_action_risk_6h, military_action_risk_24h
            FROM current_route_state
            """
        ).fetchone()

        persist_monitoring_state(self.cursor, intel, [route_row], reports, snapshot_time=200.0)
        second_risks = self.cursor.execute(
            """
            SELECT military_action_risk_6h, military_action_risk_24h
            FROM current_route_state
            """
        ).fetchone()

        self.assertGreater(second_risks[0], first_risks[0])
        self.assertGreater(second_risks[1], first_risks[1])

    def test_route_alert_explanation_includes_trend_and_live_sources(self):
        intel = [self._region_intelligence()]
        reports = [
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "started_at": 100.0,
                "finished_at": 101.0,
                "status": "success",
                "latency_seconds": 1.0,
                "item_count": 8,
                "error_reason": "",
            },
            {
                "source_name": "gfw",
                "source_tier": "secondary_live",
                "started_at": 100.0,
                "finished_at": 102.0,
                "status": "success",
                "latency_seconds": 2.0,
                "item_count": 3,
                "error_reason": "",
            },
            {
                "source_name": "satellite",
                "source_tier": "experimental",
                "started_at": 100.0,
                "finished_at": 103.0,
                "status": "success",
                "latency_seconds": 3.0,
                "item_count": 1,
                "error_reason": "",
            },
        ]

        persist_monitoring_state(
            self.cursor,
            intel,
            [(25.0, 45.0, 27.0, 47.0, 2, 1, 1.2, 4.5)],
            reports,
            snapshot_time=100.0,
        )
        persist_monitoring_state(
            self.cursor,
            intel,
            [(25.0, 45.0, 27.0, 47.0, 5, 3, 1.8, 12.5)],
            reports,
            snapshot_time=200.0,
        )

        explanation = self.cursor.execute(
            """
            SELECT explanation
            FROM alerts
            WHERE object_type = 'route' AND alert_type = 'MAJOR_FLOW_ROUTE'
            """
        ).fetchone()[0]
        self.assertIn("strengthening", explanation)
        self.assertIn("adsb, gfw", explanation)
        self.assertIn("experimental: satellite", explanation)

    def test_low_value_region_alerts_do_not_spam_duplicate_events(self):
        reports = [
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "started_at": 100.0,
                "finished_at": 101.0,
                "status": "success",
                "latency_seconds": 1.0,
                "item_count": 8,
                "error_reason": "",
            }
        ]
        region = self._low_signal_region_intelligence()

        persist_monitoring_state(self.cursor, [region], [], reports, snapshot_time=100.0)
        persist_monitoring_state(self.cursor, [region], [], reports, snapshot_time=200.0)

        alert_rows = self.cursor.execute(
            """
            SELECT COUNT(*)
            FROM alerts
            WHERE object_type = 'region' AND alert_type = 'NEW_ENTRY_CLUSTER'
            """
        ).fetchone()[0]
        alert_event_rows = self.cursor.execute(
            """
            SELECT COUNT(*)
            FROM alert_events
            WHERE alert_id IN (
                SELECT alert_id
                FROM alerts
                WHERE object_type = 'region' AND alert_type = 'NEW_ENTRY_CLUSTER'
            )
            """
        ).fetchone()[0]

        self.assertEqual(alert_rows, 1)
        self.assertEqual(alert_event_rows, 1)
        self.assertEqual(
            self.cursor.execute("SELECT COUNT(*) FROM region_snapshots").fetchone()[0],
            2,
        )

    def test_high_value_alerts_do_not_emit_duplicate_updates_when_materially_unchanged(self):
        reports = [
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "started_at": 100.0,
                "finished_at": 101.0,
                "status": "success",
                "latency_seconds": 1.0,
                "item_count": 8,
                "error_reason": "",
            }
        ]
        region = self._region_intelligence()

        persist_monitoring_state(self.cursor, [region], [], reports, snapshot_time=100.0)
        persist_monitoring_state(self.cursor, [region], [], reports, snapshot_time=200.0)

        event_rows = self.cursor.execute(
            """
            SELECT event_type, severity
            FROM alert_events
            WHERE alert_id IN (
                SELECT alert_id
                FROM alerts
                WHERE object_type = 'region' AND alert_type = 'STAGING_REGION'
            )
            ORDER BY event_time
            """
        ).fetchall()
        self.assertEqual(event_rows, [("opened", "CRITICAL")])
        self.assertEqual(
            self.cursor.execute("SELECT COUNT(*) FROM region_snapshots").fetchone()[0],
            2,
        )

    def test_region_alert_severity_transitions_when_evidence_strengthens(self):
        reports = [
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "started_at": 100.0,
                "finished_at": 101.0,
                "status": "success",
                "latency_seconds": 1.0,
                "item_count": 8,
                "error_reason": "",
            }
        ]
        moderate_region = self._region_intelligence_with_risk_profile(
            lat=25.0,
            lon=45.0,
            aircraft_count=4,
            military_count=1,
            recurring_appearances=2,
            new_aircraft=1,
            change_score=6.0,
            staging_score=45.0,
            projection_score=10.0,
            coordinated_score=20.0,
            anomaly_score=15.0,
            confidence=70,
            explanation="moderate staging signal",
        )
        strong_region = self._region_intelligence_with_risk_profile(
            lat=25.0,
            lon=45.0,
            aircraft_count=6,
            military_count=3,
            recurring_appearances=4,
            new_aircraft=3,
            change_score=20.0,
            staging_score=70.0,
            projection_score=20.0,
            coordinated_score=40.0,
            anomaly_score=30.0,
            confidence=88,
            explanation="strong staging escalation",
        )

        persist_monitoring_state(self.cursor, [moderate_region], [], reports, snapshot_time=100.0)
        persist_monitoring_state(self.cursor, [strong_region], [], reports, snapshot_time=200.0)

        alert_row = self.cursor.execute(
            """
            SELECT severity, status, priority_score
            FROM alerts
            WHERE object_type = 'region' AND alert_type = 'STAGING_REGION'
            """
        ).fetchone()
        self.assertEqual(alert_row[:2], ("CRITICAL", "new"))
        self.assertGreater(alert_row[2], 80.0)

        event_rows = self.cursor.execute(
            """
            SELECT event_type, severity
            FROM alert_events
            WHERE alert_id IN (
                SELECT alert_id
                FROM alerts
                WHERE object_type = 'region' AND alert_type = 'STAGING_REGION'
            )
            ORDER BY event_time
            """
        ).fetchall()
        self.assertEqual(
            event_rows,
            [("opened", "WATCH"), ("updated", "CRITICAL")],
        )

    def test_region_risk_increases_when_history_shows_repeat_presence(self):
        reports = [
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "started_at": 100.0,
                "finished_at": 101.0,
                "status": "success",
                "latency_seconds": 1.0,
                "item_count": 8,
                "error_reason": "",
            }
        ]
        repeating_region = self._region_intelligence_with_risk_profile(
            lat=25.0,
            lon=45.0,
            aircraft_count=4,
            military_count=1,
            recurring_appearances=1,
            new_aircraft=1,
            change_score=6.0,
            staging_score=45.0,
            projection_score=10.0,
            coordinated_score=20.0,
            anomaly_score=15.0,
            confidence=70,
            explanation="moderate staging signal",
        )

        persist_monitoring_state(self.cursor, [repeating_region], [], reports, snapshot_time=100.0)
        first_risks = self.cursor.execute(
            """
            SELECT military_action_risk_6h, military_action_risk_24h
            FROM current_region_state
            """
        ).fetchone()

        persist_monitoring_state(self.cursor, [repeating_region], [], reports, snapshot_time=200.0)
        second_risks = self.cursor.execute(
            """
            SELECT military_action_risk_6h, military_action_risk_24h
            FROM current_region_state
            """
        ).fetchone()

        self.assertGreater(second_risks[0], first_risks[0])
        self.assertGreater(second_risks[1], first_risks[1])

    def test_region_escalation_direction_uses_recent_history_window(self):
        reports = [
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "started_at": 100.0,
                "finished_at": 101.0,
                "status": "success",
                "latency_seconds": 1.0,
                "item_count": 8,
                "error_reason": "",
            }
        ]
        region = self._region_intelligence_with_risk_profile(
            lat=25.0,
            lon=45.0,
            aircraft_count=4,
            military_count=1,
            recurring_appearances=1,
            new_aircraft=1,
            change_score=6.0,
            staging_score=45.0,
            projection_score=10.0,
            coordinated_score=20.0,
            anomaly_score=15.0,
            confidence=70,
            explanation="moderate staging signal",
        )

        persist_monitoring_state(self.cursor, [region], [], reports, snapshot_time=100.0)
        persist_monitoring_state(self.cursor, [region], [], reports, snapshot_time=200.0)
        persist_monitoring_state(self.cursor, [region], [], reports, snapshot_time=300.0)

        escalation_direction = self.cursor.execute(
            """
            SELECT escalation_direction
            FROM current_region_state
            """
        ).fetchone()[0]
        self.assertEqual(escalation_direction, "rising")

    def test_region_evidence_summary_includes_forecast_drivers(self):
        reports = [
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "started_at": 100.0,
                "finished_at": 101.0,
                "status": "success",
                "latency_seconds": 1.0,
                "item_count": 8,
                "error_reason": "",
            }
        ]
        region = self._region_intelligence_with_risk_profile(
            lat=25.0,
            lon=45.0,
            aircraft_count=4,
            military_count=1,
            recurring_appearances=1,
            new_aircraft=1,
            change_score=6.0,
            staging_score=45.0,
            projection_score=10.0,
            coordinated_score=20.0,
            anomaly_score=15.0,
            confidence=70,
            explanation="moderate staging signal",
        )

        persist_monitoring_state(self.cursor, [region], [], reports, snapshot_time=100.0)
        persist_monitoring_state(self.cursor, [region], [], reports, snapshot_time=200.0)

        evidence_summary = self.cursor.execute(
            """
            SELECT evidence_summary
            FROM current_region_state
            """
        ).fetchone()[0]
        evidence = json.loads(evidence_summary)

        self.assertEqual(evidence["forecast"]["direction"], "rising")
        self.assertGreater(evidence["forecast"]["military_action_risk_24h"], 38.5)
        self.assertIn("repeat_presence_6h", evidence["forecast"]["drivers"])

    def test_alerts_enter_cooldown_before_resolution_when_evidence_drops_out(self):
        reports = [
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "started_at": 100.0,
                "finished_at": 101.0,
                "status": "success",
                "latency_seconds": 1.0,
                "item_count": 8,
                "error_reason": "",
            }
        ]

        persist_monitoring_state(self.cursor, [self._region_intelligence()], [], reports, snapshot_time=100.0)
        persist_monitoring_state(self.cursor, [], [], reports, snapshot_time=200.0)

        cooldown_row = self.cursor.execute(
            """
            SELECT status, severity
            FROM alerts
            WHERE object_type = 'region' AND alert_type = 'STAGING_REGION'
            """
        ).fetchone()
        self.assertEqual(cooldown_row, ("watching", "PRIORITY"))

        persist_monitoring_state(self.cursor, [], [], reports, snapshot_time=4_000.0)

        resolved_row = self.cursor.execute(
            """
            SELECT status, severity
            FROM alerts
            WHERE object_type = 'region' AND alert_type = 'STAGING_REGION'
            """
        ).fetchone()
        self.assertEqual(resolved_row, ("resolved", "PRIORITY"))

        event_rows = self.cursor.execute(
            """
            SELECT event_type, severity
            FROM alert_events
            WHERE alert_id IN (
                SELECT alert_id
                FROM alerts
                WHERE object_type = 'region' AND alert_type = 'STAGING_REGION'
            )
            ORDER BY event_time
            """
        ).fetchall()
        self.assertEqual(
            event_rows,
            [("opened", "CRITICAL"), ("cooling", "PRIORITY"), ("resolved", "PRIORITY")],
        )

    def test_watchlist_matches_boost_alert_priority_and_mark_alert(self):
        reports = [
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "started_at": 100.0,
                "finished_at": 101.0,
                "status": "success",
                "latency_seconds": 1.0,
                "item_count": 8,
                "error_reason": "",
            }
        ]
        region = self._region_intelligence_with_risk_profile(
            lat=25.0,
            lon=45.0,
            aircraft_count=4,
            military_count=1,
            recurring_appearances=2,
            new_aircraft=1,
            change_score=6.0,
            staging_score=45.0,
            projection_score=10.0,
            coordinated_score=20.0,
            anomaly_score=15.0,
            confidence=70,
            explanation="moderate staging signal",
        )

        self.cursor.execute(
            """
            INSERT INTO watchlists (object_type, object_id, theater_id, active, created_at)
            VALUES ('region', '', 'middle_east', 1, 0)
            """
        )

        persist_monitoring_state(self.cursor, [region], [], reports, snapshot_time=100.0)

        alert_row = self.cursor.execute(
            """
            SELECT priority_score, watchlist_match, visible
            FROM alerts
            WHERE object_type = 'region' AND alert_type = 'STAGING_REGION'
            """
        ).fetchone()
        self.assertGreater(alert_row[0], 41.5)
        self.assertEqual(alert_row[1:], (1, 1))

    def test_suppression_rule_hides_matching_alert_but_keeps_record(self):
        reports = [
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "started_at": 100.0,
                "finished_at": 101.0,
                "status": "success",
                "latency_seconds": 1.0,
                "item_count": 8,
                "error_reason": "",
            }
        ]

        self.cursor.execute(
            """
            INSERT INTO suppression_rules (object_type, object_id, theater_id, alert_type, active, created_at)
            VALUES ('region', '', 'middle_east', 'STAGING_REGION', 1, 0)
            """
        )

        persist_monitoring_state(self.cursor, [self._region_intelligence()], [], reports, snapshot_time=100.0)

        alert_row = self.cursor.execute(
            """
            SELECT status, visible
            FROM alerts
            WHERE object_type = 'region' AND alert_type = 'STAGING_REGION'
            """
        ).fetchone()
        self.assertEqual(alert_row, ("suppressed", 0))
        self.assertEqual(
            self.cursor.execute(
                """
                SELECT COUNT(*)
                FROM alert_events
                WHERE alert_id IN (
                    SELECT alert_id
                    FROM alerts
                    WHERE object_type = 'region' AND alert_type = 'STAGING_REGION'
                )
                """
            ).fetchone()[0],
            1,
        )

    def test_experimental_only_evidence_cannot_generate_critical_alerts(self):
        intel = [self._region_intelligence()]
        routes = []
        reports = [
            {
                "source_name": "satellite",
                "source_tier": "experimental",
                "started_at": 300.0,
                "finished_at": 301.0,
                "status": "success",
                "latency_seconds": 1.0,
                "item_count": 3,
                "error_reason": "",
            }
        ]

        persist_monitoring_state(self.cursor, intel, routes, reports, snapshot_time=300.0)

        severities = {
            row[0]
            for row in self.cursor.execute("SELECT severity FROM alerts").fetchall()
        }
        self.assertNotIn("CRITICAL", severities)


class MonitoringExportTests(unittest.TestCase):
    def test_export_monitoring_snapshot_writes_latest_regions_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "monitoring.db"
            conn, cursor = init_db(str(db_path))
            reports = [
                {
                    "source_name": "adsb",
                    "source_tier": "primary_live",
                    "started_at": 100.0,
                    "finished_at": 101.0,
                    "status": "success",
                    "latency_seconds": 1.0,
                    "item_count": 8,
                    "error_reason": "",
                }
            ]
            features = RegionFeatures(
                lat=25.0,
                lon=45.0,
                aircraft_count=8,
                military_count=4,
                recurring_appearances=4,
                spike_flag=True,
                coordination_flag=True,
                staging_flag=True,
                new_aircraft=5,
                change_score=26.0,
                change_level="HIGH",
            )
            region = RegionIntelligence(
                lat=25.0,
                lon=45.0,
                classification=Classification.STAGING,
                confidence=84,
                score=74.0,
                all_scores={
                    "STAGING": 74.0,
                    "PROJECTION": 28.0,
                    "ROUTINE": 6.0,
                    "ANOMALY": 58.0,
                    "COORDINATED_ACTIVITY": 61.0,
                },
                features=features,
                explanation="persistent military buildup with outbound movement",
            )

            persist_monitoring_state(cursor, [region], [], reports, snapshot_time=100.0)
            conn.commit()
            export_paths = export_monitoring_snapshot(cursor, Path(tmpdir) / "exports", 100.0)

            self.assertIn("regions", export_paths)
            payload = json.loads(Path(export_paths["regions"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["generated_at"], 100.0)
            self.assertEqual(len(payload["regions"]), 1)

            exported_region = payload["regions"][0]
            self.assertEqual(exported_region["region_id"], cursor.execute(
                "SELECT region_id FROM current_region_state"
            ).fetchone()[0])
            self.assertEqual(exported_region["classification"], "STAGING")
            self.assertEqual(exported_region["operational_severity"], "CRITICAL")
            self.assertEqual(exported_region["appearance_count_6h"], 1)
            self.assertEqual(exported_region["max_military_24h"], 4)
            self.assertEqual(exported_region["visible"], True)
            self.assertEqual(
                exported_region["visible_alert_types"],
                ["ESCALATION_REGION", "MILITARY_BUILDUP", "NEW_ENTRY_CLUSTER", "REGION_SURGE", "STAGING_REGION"],
            )
            self.assertEqual(exported_region["suppressed_alert_types"], [])
            self.assertEqual(
                exported_region["evidence_summary"]["rollups"]["max_aircraft_24h"],
                8,
            )
            conn.close()

    def test_export_monitoring_snapshot_writes_latest_routes_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "monitoring.db"
            conn, cursor = init_db(str(db_path))
            reports = [
                {
                    "source_name": "adsb",
                    "source_tier": "primary_live",
                    "started_at": 100.0,
                    "finished_at": 101.0,
                    "status": "success",
                    "latency_seconds": 1.0,
                    "item_count": 8,
                    "error_reason": "",
                },
                {
                    "source_name": "gfw",
                    "source_tier": "secondary_live",
                    "started_at": 100.0,
                    "finished_at": 102.0,
                    "status": "success",
                    "latency_seconds": 2.0,
                    "item_count": 3,
                    "error_reason": "",
                },
            ]

            persist_monitoring_state(
                cursor,
                [],
                [(25.0, 45.0, 27.0, 47.0, 4, 2, 1.6, 9.0)],
                reports,
                snapshot_time=100.0,
            )
            conn.commit()
            export_paths = export_monitoring_snapshot(cursor, Path(tmpdir) / "exports", 100.0)

            self.assertIn("routes", export_paths)
            payload = json.loads(Path(export_paths["routes"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["generated_at"], 100.0)
            self.assertEqual(len(payload["routes"]), 1)

            exported_route = payload["routes"][0]
            self.assertEqual(exported_route["origin"], [25.0, 45.0])
            self.assertEqual(exported_route["destination"], [27.0, 47.0])
            self.assertEqual(exported_route["flow_score"], 9.0)
            self.assertEqual(exported_route["trend_direction"], "rising")
            self.assertEqual(exported_route["appearance_count"], 1)
            self.assertEqual(exported_route["visible"], True)
            self.assertEqual(exported_route["visible_alert_types"], ["MAJOR_FLOW_ROUTE"])
            self.assertEqual(exported_route["suppressed_alert_types"], [])
            self.assertEqual(
                exported_route["evidence_summary"]["source_provenance"]["live_source_names"],
                ["adsb", "gfw"],
            )
            conn.close()

    def test_export_monitoring_snapshot_writes_latest_source_health_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "monitoring.db"
            conn, cursor = init_db(str(db_path))
            reports = [
                {
                    "source_name": "adsb",
                    "source_tier": "primary_live",
                    "started_at": 100.0,
                    "finished_at": 101.0,
                    "status": "success",
                    "latency_seconds": 1.0,
                    "item_count": 8,
                    "error_reason": "",
                },
                {
                    "source_name": "aisstream",
                    "source_tier": "primary_live",
                    "started_at": 100.0,
                    "finished_at": 102.0,
                    "status": "missing_credentials",
                    "latency_seconds": 2.0,
                    "item_count": 0,
                    "error_reason": "AIS_API_KEY not set",
                },
            ]

            persist_source_runs(cursor, reports)
            conn.commit()
            export_paths = export_monitoring_snapshot(cursor, Path(tmpdir) / "exports", 100.0)

            self.assertIn("source_health", export_paths)
            payload = json.loads(Path(export_paths["source_health"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["generated_at"], 100.0)
            self.assertEqual(len(payload["sources"]), 2)

            exported_sources = {row["source_name"]: row for row in payload["sources"]}
            self.assertEqual(exported_sources["adsb"]["degraded"], False)
            self.assertEqual(exported_sources["adsb"]["last_item_count"], 8)
            self.assertEqual(exported_sources["aisstream"]["degraded"], True)
            self.assertEqual(exported_sources["aisstream"]["last_error_reason"], "AIS_API_KEY not set")
            conn.close()

    def test_export_monitoring_snapshot_writes_latest_summary_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "monitoring.db"
            conn, cursor = init_db(str(db_path))
            reports = [
                {
                    "source_name": "adsb",
                    "source_tier": "primary_live",
                    "started_at": 100.0,
                    "finished_at": 101.0,
                    "status": "success",
                    "latency_seconds": 1.0,
                    "item_count": 8,
                    "error_reason": "",
                },
                {
                    "source_name": "aisstream",
                    "source_tier": "primary_live",
                    "started_at": 100.0,
                    "finished_at": 102.0,
                    "status": "missing_credentials",
                    "latency_seconds": 2.0,
                    "item_count": 0,
                    "error_reason": "AIS_API_KEY not set",
                },
            ]
            features = RegionFeatures(
                lat=25.0,
                lon=45.0,
                aircraft_count=8,
                military_count=4,
                recurring_appearances=4,
                spike_flag=True,
                coordination_flag=True,
                staging_flag=True,
                new_aircraft=5,
                change_score=26.0,
                change_level="HIGH",
            )
            region = RegionIntelligence(
                lat=25.0,
                lon=45.0,
                classification=Classification.STAGING,
                confidence=84,
                score=74.0,
                all_scores={
                    "STAGING": 74.0,
                    "PROJECTION": 28.0,
                    "ROUTINE": 6.0,
                    "ANOMALY": 58.0,
                    "COORDINATED_ACTIVITY": 61.0,
                },
                features=features,
                explanation="persistent military buildup with outbound movement",
            )

            self.assertEqual(
                cursor.execute("SELECT region_id FROM current_region_state").fetchall(),
                [],
            )
            cursor.execute(
                """
                INSERT INTO suppression_rules (object_type, object_id, theater_id, alert_type, active, created_at)
                VALUES ('region', '', 'middle_east', 'NEW_ENTRY_CLUSTER', 1, 0)
                """
            )

            persist_source_runs(cursor, reports)
            persist_monitoring_state(
                cursor,
                [region],
                [(25.0, 45.0, 27.0, 47.0, 4, 2, 1.6, 9.0)],
                reports,
                snapshot_time=100.0,
            )
            conn.commit()
            export_paths = export_monitoring_snapshot(cursor, Path(tmpdir) / "exports", 100.0)

            self.assertIn("summary", export_paths)
            payload = json.loads(Path(export_paths["summary"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["generated_at"], 100.0)
            self.assertEqual(payload["active_alert_count"], 6)
            self.assertEqual(payload["visible_alert_count"], 5)
            self.assertEqual(payload["suppressed_alert_count"], 1)
            self.assertEqual(payload["alerts_by_severity"]["CRITICAL"], 6)
            self.assertEqual(payload["theaters"]["middle_east"]["region_count"], 1)
            self.assertEqual(payload["theaters"]["middle_east"]["route_count"], 1)
            self.assertEqual(payload["source_health"]["degraded_count"], 1)
            self.assertEqual(payload["source_health"]["healthy_count"], 1)
            conn.close()

    def test_export_monitoring_snapshot_writes_latest_priority_brief_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "monitoring.db"
            conn, cursor = init_db(str(db_path))
            try:
                reports = [
                    {
                        "source_name": "adsb",
                        "source_tier": "primary_live",
                        "started_at": 100.0,
                        "finished_at": 101.0,
                        "status": "success",
                        "latency_seconds": 1.0,
                        "item_count": 8,
                        "error_reason": "",
                    }
                ]
                middle_east_region = RegionIntelligence(
                    lat=25.0,
                    lon=45.0,
                    classification=Classification.STAGING,
                    confidence=84,
                    score=74.0,
                    all_scores={
                        "STAGING": 74.0,
                        "PROJECTION": 28.0,
                        "ROUTINE": 6.0,
                        "ANOMALY": 58.0,
                        "COORDINATED_ACTIVITY": 61.0,
                    },
                    features=RegionFeatures(
                        lat=25.0,
                        lon=45.0,
                        aircraft_count=8,
                        military_count=4,
                        recurring_appearances=4,
                        spike_flag=True,
                        coordination_flag=True,
                        staging_flag=True,
                        new_aircraft=5,
                        change_score=26.0,
                        change_level="HIGH",
                    ),
                    explanation="persistent military buildup with outbound movement",
                )
                europe_region = RegionIntelligence(
                    lat=50.0,
                    lon=10.0,
                    classification=Classification.ROUTINE,
                    confidence=58,
                    score=34.0,
                    all_scores={
                        "STAGING": 18.0,
                        "PROJECTION": 10.0,
                        "ROUTINE": 34.0,
                        "ANOMALY": 16.0,
                        "COORDINATED_ACTIVITY": 14.0,
                    },
                    features=RegionFeatures(
                        lat=50.0,
                        lon=10.0,
                        aircraft_count=4,
                        military_count=1,
                        recurring_appearances=1,
                        spike_flag=False,
                        coordination_flag=False,
                        staging_flag=False,
                        new_aircraft=1,
                        change_score=5.0,
                        change_level="MEDIUM",
                    ),
                    explanation="limited routine movement",
                )

                persist_monitoring_state(
                    cursor,
                    [middle_east_region, europe_region],
                    [(25.0, 45.0, 27.0, 47.0, 4, 2, 1.6, 9.0)],
                    reports,
                    snapshot_time=100.0,
                )
                conn.commit()
                export_paths = export_monitoring_snapshot(cursor, Path(tmpdir) / "exports", 100.0)

                self.assertIn("priority_brief", export_paths)
                payload = json.loads(Path(export_paths["priority_brief"]).read_text(encoding="utf-8"))
                self.assertEqual(payload["generated_at"], 100.0)
                self.assertEqual(len(payload["top_theaters"]), 2)

                top_theater = payload["top_theaters"][0]
                self.assertEqual(top_theater["theater_id"], "middle_east")
                self.assertEqual(top_theater["label"], "Middle East")
                self.assertGreater(top_theater["priority_score"], payload["top_theaters"][1]["priority_score"])
                self.assertGreaterEqual(top_theater["visible_alert_count"], 1)
                self.assertGreaterEqual(top_theater["rising_region_count"], 1)
                self.assertIn("max risk", top_theater["explanation"])
            finally:
                conn.close()

    def test_export_monitoring_snapshot_priority_brief_includes_top_routes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "monitoring.db"
            conn, cursor = init_db(str(db_path))
            try:
                reports = [
                    {
                        "source_name": "adsb",
                        "source_tier": "primary_live",
                        "started_at": 100.0,
                        "finished_at": 101.0,
                        "status": "success",
                        "latency_seconds": 1.0,
                        "item_count": 8,
                        "error_reason": "",
                    },
                    {
                        "source_name": "gfw",
                        "source_tier": "secondary_live",
                        "started_at": 100.0,
                        "finished_at": 102.0,
                        "status": "success",
                        "latency_seconds": 2.0,
                        "item_count": 3,
                        "error_reason": "",
                    },
                ]
                persist_monitoring_state(
                    cursor,
                    [],
                    [
                        (25.0, 45.0, 27.0, 47.0, 4, 2, 1.6, 9.0),
                        (50.0, 10.0, 52.0, 12.0, 2, 0, 1.1, 4.0),
                    ],
                    reports,
                    snapshot_time=100.0,
                )
                conn.commit()
                export_paths = export_monitoring_snapshot(cursor, Path(tmpdir) / "exports", 100.0)

                payload = json.loads(Path(export_paths["priority_brief"]).read_text(encoding="utf-8"))
                self.assertIn("top_routes", payload)
                self.assertEqual(len(payload["top_routes"]), 1)

                top_route = payload["top_routes"][0]
                self.assertEqual(top_route["origin"], [25.0, 45.0])
                self.assertEqual(top_route["destination"], [27.0, 47.0])
                self.assertEqual(top_route["theater_id"], "middle_east")
                self.assertEqual(top_route["trend_direction"], "rising")
                self.assertEqual(top_route["live_source_count"], 2)
                self.assertGreater(top_route["priority_score"], 0.0)
                self.assertIn("live sources", top_route["explanation"])
            finally:
                conn.close()

    def test_export_monitoring_snapshot_priority_brief_includes_top_alerts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "monitoring.db"
            conn, cursor = init_db(str(db_path))
            try:
                reports = [
                    {
                        "source_name": "adsb",
                        "source_tier": "primary_live",
                        "started_at": 100.0,
                        "finished_at": 101.0,
                        "status": "success",
                        "latency_seconds": 1.0,
                        "item_count": 8,
                        "error_reason": "",
                    }
                ]
                region = RegionIntelligence(
                    lat=25.0,
                    lon=45.0,
                    classification=Classification.STAGING,
                    confidence=84,
                    score=74.0,
                    all_scores={
                        "STAGING": 74.0,
                        "PROJECTION": 28.0,
                        "ROUTINE": 6.0,
                        "ANOMALY": 58.0,
                        "COORDINATED_ACTIVITY": 61.0,
                    },
                    features=RegionFeatures(
                        lat=25.0,
                        lon=45.0,
                        aircraft_count=8,
                        military_count=4,
                        recurring_appearances=4,
                        spike_flag=True,
                        coordination_flag=True,
                        staging_flag=True,
                        new_aircraft=5,
                        change_score=26.0,
                        change_level="HIGH",
                    ),
                    explanation="persistent military buildup with outbound movement",
                )

                persist_monitoring_state(cursor, [region], [], reports, snapshot_time=100.0)
                conn.commit()
                export_paths = export_monitoring_snapshot(cursor, Path(tmpdir) / "exports", 100.0)

                payload = json.loads(Path(export_paths["priority_brief"]).read_text(encoding="utf-8"))
                self.assertIn("top_alerts", payload)
                self.assertGreaterEqual(len(payload["top_alerts"]), 1)

                top_alert = payload["top_alerts"][0]
                self.assertEqual(top_alert["theater_id"], "middle_east")
                self.assertEqual(top_alert["severity"], "CRITICAL")
                self.assertEqual(top_alert["object_type"], "region")
                self.assertTrue(top_alert["visible"])
                self.assertGreater(top_alert["priority_score"], 0.0)
                self.assertIn("military buildup", top_alert["explanation"])
            finally:
                conn.close()

    def test_export_monitoring_snapshot_priority_brief_includes_degraded_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "monitoring.db"
            conn, cursor = init_db(str(db_path))
            try:
                reports = [
                    {
                        "source_name": "adsb",
                        "source_tier": "primary_live",
                        "started_at": 100.0,
                        "finished_at": 101.0,
                        "status": "success",
                        "latency_seconds": 1.0,
                        "item_count": 8,
                        "error_reason": "",
                    },
                    {
                        "source_name": "aisstream",
                        "source_tier": "primary_live",
                        "started_at": 100.0,
                        "finished_at": 102.0,
                        "status": "missing_credentials",
                        "latency_seconds": 2.0,
                        "item_count": 0,
                        "error_reason": "AIS_API_KEY not set",
                    },
                    {
                        "source_name": "satellite",
                        "source_tier": "experimental",
                        "started_at": 100.0,
                        "finished_at": 103.0,
                        "status": "error",
                        "latency_seconds": 3.0,
                        "item_count": 0,
                        "error_reason": "upstream timeout",
                    },
                ]

                persist_source_runs(cursor, reports)
                conn.commit()
                export_paths = export_monitoring_snapshot(cursor, Path(tmpdir) / "exports", 100.0)

                payload = json.loads(Path(export_paths["priority_brief"]).read_text(encoding="utf-8"))
                self.assertIn("degraded_sources", payload)
                self.assertEqual(len(payload["degraded_sources"]), 2)

                degraded_names = [row["source_name"] for row in payload["degraded_sources"]]
                self.assertEqual(degraded_names, ["aisstream", "satellite"])
                self.assertEqual(payload["degraded_sources"][0]["last_status"], "missing_credentials")
                self.assertEqual(payload["degraded_sources"][0]["last_error_reason"], "AIS_API_KEY not set")
            finally:
                conn.close()

    def test_export_monitoring_snapshot_priority_brief_includes_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "monitoring.db"
            conn, cursor = init_db(str(db_path))
            try:
                reports = [
                    {
                        "source_name": "adsb",
                        "source_tier": "primary_live",
                        "started_at": 100.0,
                        "finished_at": 101.0,
                        "status": "success",
                        "latency_seconds": 1.0,
                        "item_count": 8,
                        "error_reason": "",
                    },
                    {
                        "source_name": "aisstream",
                        "source_tier": "primary_live",
                        "started_at": 100.0,
                        "finished_at": 102.0,
                        "status": "missing_credentials",
                        "latency_seconds": 2.0,
                        "item_count": 0,
                        "error_reason": "AIS_API_KEY not set",
                    },
                ]
                region = RegionIntelligence(
                    lat=25.0,
                    lon=45.0,
                    classification=Classification.STAGING,
                    confidence=84,
                    score=74.0,
                    all_scores={
                        "STAGING": 74.0,
                        "PROJECTION": 28.0,
                        "ROUTINE": 6.0,
                        "ANOMALY": 58.0,
                        "COORDINATED_ACTIVITY": 61.0,
                    },
                    features=RegionFeatures(
                        lat=25.0,
                        lon=45.0,
                        aircraft_count=8,
                        military_count=4,
                        recurring_appearances=4,
                        spike_flag=True,
                        coordination_flag=True,
                        staging_flag=True,
                        new_aircraft=5,
                        change_score=26.0,
                        change_level="HIGH",
                    ),
                    explanation="persistent military buildup with outbound movement",
                )

                persist_source_runs(cursor, reports)
                persist_monitoring_state(
                    cursor,
                    [region],
                    [(25.0, 45.0, 27.0, 47.0, 4, 2, 1.6, 9.0)],
                    reports,
                    snapshot_time=100.0,
                )
                conn.commit()
                export_paths = export_monitoring_snapshot(cursor, Path(tmpdir) / "exports", 100.0)

                payload = json.loads(Path(export_paths["priority_brief"]).read_text(encoding="utf-8"))
                self.assertIn("summary", payload)
                self.assertEqual(payload["summary"]["top_theater_id"], "middle_east")
                self.assertGreaterEqual(payload["summary"]["visible_alert_count"], 1)
                self.assertEqual(payload["summary"]["degraded_source_count"], 1)
                self.assertIn("middle_east", payload["summary"]["headline"])
            finally:
                conn.close()

    def test_export_monitoring_snapshot_writes_latest_operator_views_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "monitoring.db"
            conn, cursor = init_db(str(db_path))
            try:
                reports = [
                    {
                        "source_name": "adsb",
                        "source_tier": "primary_live",
                        "started_at": 100.0,
                        "finished_at": 101.0,
                        "status": "success",
                        "latency_seconds": 1.0,
                        "item_count": 8,
                        "error_reason": "",
                    }
                ]
                middle_east_region = RegionIntelligence(
                    lat=25.0,
                    lon=45.0,
                    classification=Classification.STAGING,
                    confidence=84,
                    score=74.0,
                    all_scores={
                        "STAGING": 74.0,
                        "PROJECTION": 28.0,
                        "ROUTINE": 6.0,
                        "ANOMALY": 58.0,
                        "COORDINATED_ACTIVITY": 61.0,
                    },
                    features=RegionFeatures(
                        lat=25.0,
                        lon=45.0,
                        aircraft_count=8,
                        military_count=4,
                        recurring_appearances=4,
                        spike_flag=True,
                        coordination_flag=True,
                        staging_flag=True,
                        new_aircraft=5,
                        change_score=26.0,
                        change_level="HIGH",
                    ),
                    explanation="persistent military buildup with outbound movement",
                )
                europe_region = RegionIntelligence(
                    lat=50.0,
                    lon=10.0,
                    classification=Classification.ROUTINE,
                    confidence=58,
                    score=34.0,
                    all_scores={
                        "STAGING": 18.0,
                        "PROJECTION": 10.0,
                        "ROUTINE": 34.0,
                        "ANOMALY": 16.0,
                        "COORDINATED_ACTIVITY": 14.0,
                    },
                    features=RegionFeatures(
                        lat=50.0,
                        lon=10.0,
                        aircraft_count=4,
                        military_count=1,
                        recurring_appearances=1,
                        spike_flag=False,
                        coordination_flag=False,
                        staging_flag=False,
                        new_aircraft=1,
                        change_score=5.0,
                        change_level="MEDIUM",
                    ),
                    explanation="limited routine movement",
                )

                persist_monitoring_state(
                    cursor,
                    [middle_east_region, europe_region],
                    [(25.0, 45.0, 27.0, 47.0, 4, 2, 1.6, 9.0)],
                    reports,
                    snapshot_time=100.0,
                )
                conn.commit()
                export_paths = export_monitoring_snapshot(cursor, Path(tmpdir) / "exports", 100.0)

                self.assertIn("operator_views", export_paths)
                payload = json.loads(Path(export_paths["operator_views"]).read_text(encoding="utf-8"))
                self.assertEqual(payload["generated_at"], 100.0)
                self.assertIn("top_rising_theaters", payload)
                self.assertEqual(len(payload["top_rising_theaters"]), 2)

                top_theater = payload["top_rising_theaters"][0]
                self.assertEqual(top_theater["theater_id"], "middle_east")
                self.assertEqual(top_theater["label"], "Middle East")
                self.assertGreater(top_theater["selection_score"], payload["top_rising_theaters"][1]["selection_score"])
                self.assertGreaterEqual(top_theater["rising_region_count"], 1)
                self.assertGreaterEqual(top_theater["visible_alert_count"], 1)
            finally:
                conn.close()

    def test_export_monitoring_snapshot_operator_views_include_top_military_corridors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "monitoring.db"
            conn, cursor = init_db(str(db_path))
            try:
                reports = [
                    {
                        "source_name": "adsb",
                        "source_tier": "primary_live",
                        "started_at": 100.0,
                        "finished_at": 101.0,
                        "status": "success",
                        "latency_seconds": 1.0,
                        "item_count": 8,
                        "error_reason": "",
                    },
                    {
                        "source_name": "gfw",
                        "source_tier": "secondary_live",
                        "started_at": 100.0,
                        "finished_at": 102.0,
                        "status": "success",
                        "latency_seconds": 2.0,
                        "item_count": 3,
                        "error_reason": "",
                    },
                ]
                persist_monitoring_state(
                    cursor,
                    [],
                    [
                        (25.0, 45.0, 27.0, 47.0, 4, 2, 1.6, 9.0),
                        (50.0, 10.0, 52.0, 12.0, 2, 0, 1.1, 4.0),
                    ],
                    reports,
                    snapshot_time=100.0,
                )
                conn.commit()
                export_paths = export_monitoring_snapshot(cursor, Path(tmpdir) / "exports", 100.0)

                payload = json.loads(Path(export_paths["operator_views"]).read_text(encoding="utf-8"))
                self.assertIn("top_military_corridors", payload)
                self.assertEqual(len(payload["top_military_corridors"]), 1)

                top_corridor = payload["top_military_corridors"][0]
                self.assertEqual(top_corridor["theater_id"], "middle_east")
                self.assertEqual(top_corridor["origin"], [25.0, 45.0])
                self.assertEqual(top_corridor["destination"], [27.0, 47.0])
                self.assertEqual(top_corridor["military_moved"], 2)
                self.assertEqual(top_corridor["live_source_count"], 2)
                self.assertGreater(top_corridor["selection_score"], 0.0)
            finally:
                conn.close()

    def test_export_monitoring_snapshot_operator_views_include_top_visible_alerts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "monitoring.db"
            conn, cursor = init_db(str(db_path))
            try:
                reports = [
                    {
                        "source_name": "adsb",
                        "source_tier": "primary_live",
                        "started_at": 100.0,
                        "finished_at": 101.0,
                        "status": "success",
                        "latency_seconds": 1.0,
                        "item_count": 8,
                        "error_reason": "",
                    }
                ]
                region = RegionIntelligence(
                    lat=25.0,
                    lon=45.0,
                    classification=Classification.STAGING,
                    confidence=84,
                    score=74.0,
                    all_scores={
                        "STAGING": 74.0,
                        "PROJECTION": 28.0,
                        "ROUTINE": 6.0,
                        "ANOMALY": 58.0,
                        "COORDINATED_ACTIVITY": 61.0,
                    },
                    features=RegionFeatures(
                        lat=25.0,
                        lon=45.0,
                        aircraft_count=8,
                        military_count=4,
                        recurring_appearances=4,
                        spike_flag=True,
                        coordination_flag=True,
                        staging_flag=True,
                        new_aircraft=5,
                        change_score=26.0,
                        change_level="HIGH",
                    ),
                    explanation="persistent military buildup with outbound movement",
                )

                persist_monitoring_state(cursor, [region], [], reports, snapshot_time=100.0)
                conn.commit()
                export_paths = export_monitoring_snapshot(cursor, Path(tmpdir) / "exports", 100.0)

                payload = json.loads(Path(export_paths["operator_views"]).read_text(encoding="utf-8"))
                self.assertIn("top_visible_alerts", payload)
                self.assertGreaterEqual(len(payload["top_visible_alerts"]), 1)

                top_alert = payload["top_visible_alerts"][0]
                self.assertEqual(top_alert["theater_id"], "middle_east")
                self.assertEqual(top_alert["object_type"], "region")
                self.assertEqual(top_alert["severity"], "CRITICAL")
                self.assertTrue(top_alert["visible"])
                self.assertGreater(top_alert["selection_score"], 0.0)
            finally:
                conn.close()

    def test_export_monitoring_snapshot_operator_views_include_degraded_source_constrained_view(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "monitoring.db"
            conn, cursor = init_db(str(db_path))
            try:
                reports = [
                    {
                        "source_name": "adsb",
                        "source_tier": "primary_live",
                        "started_at": 100.0,
                        "finished_at": 101.0,
                        "status": "success",
                        "latency_seconds": 1.0,
                        "item_count": 8,
                        "error_reason": "",
                    },
                    {
                        "source_name": "aisstream",
                        "source_tier": "primary_live",
                        "started_at": 100.0,
                        "finished_at": 102.0,
                        "status": "missing_credentials",
                        "latency_seconds": 2.0,
                        "item_count": 0,
                        "error_reason": "AIS_API_KEY not set",
                    },
                ]
                region = RegionIntelligence(
                    lat=25.0,
                    lon=45.0,
                    classification=Classification.STAGING,
                    confidence=84,
                    score=74.0,
                    all_scores={
                        "STAGING": 74.0,
                        "PROJECTION": 28.0,
                        "ROUTINE": 6.0,
                        "ANOMALY": 58.0,
                        "COORDINATED_ACTIVITY": 61.0,
                    },
                    features=RegionFeatures(
                        lat=25.0,
                        lon=45.0,
                        aircraft_count=8,
                        military_count=4,
                        recurring_appearances=4,
                        spike_flag=True,
                        coordination_flag=True,
                        staging_flag=True,
                        new_aircraft=5,
                        change_score=26.0,
                        change_level="HIGH",
                    ),
                    explanation="persistent military buildup with outbound movement",
                )

                persist_source_runs(cursor, reports)
                persist_monitoring_state(
                    cursor,
                    [region],
                    [(25.0, 45.0, 27.0, 47.0, 4, 2, 1.6, 9.0)],
                    reports,
                    snapshot_time=100.0,
                )
                conn.commit()
                export_paths = export_monitoring_snapshot(cursor, Path(tmpdir) / "exports", 100.0)

                payload = json.loads(Path(export_paths["operator_views"]).read_text(encoding="utf-8"))
                self.assertIn("degraded_source_constrained_view", payload)
                constrained = payload["degraded_source_constrained_view"]
                self.assertEqual(constrained["degraded_source_count"], 1)
                self.assertEqual(constrained["top_theater_id"], "middle_east")
                self.assertGreaterEqual(constrained["resilient_alert_count"], 1)
                self.assertIn("AIS_API_KEY", constrained["headline"])
            finally:
                conn.close()

    def test_export_monitoring_snapshot_writes_latest_validation_report_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "monitoring.db"
            conn, cursor = init_db(str(db_path))
            try:
                reports = [
                    {
                        "source_name": "adsb",
                        "source_tier": "primary_live",
                        "started_at": 100.0,
                        "finished_at": 101.0,
                        "status": "success",
                        "latency_seconds": 1.0,
                        "item_count": 8,
                        "error_reason": "",
                    },
                    {
                        "source_name": "aisstream",
                        "source_tier": "primary_live",
                        "started_at": 100.0,
                        "finished_at": 102.0,
                        "status": "missing_credentials",
                        "latency_seconds": 2.0,
                        "item_count": 0,
                        "error_reason": "AIS_API_KEY not set",
                    },
                ]
                region = RegionIntelligence(
                    lat=25.0,
                    lon=45.0,
                    classification=Classification.STAGING,
                    confidence=84,
                    score=74.0,
                    all_scores={
                        "STAGING": 74.0,
                        "PROJECTION": 28.0,
                        "ROUTINE": 6.0,
                        "ANOMALY": 58.0,
                        "COORDINATED_ACTIVITY": 61.0,
                    },
                    features=RegionFeatures(
                        lat=25.0,
                        lon=45.0,
                        aircraft_count=8,
                        military_count=4,
                        recurring_appearances=4,
                        spike_flag=True,
                        coordination_flag=True,
                        staging_flag=True,
                        new_aircraft=5,
                        change_score=26.0,
                        change_level="HIGH",
                    ),
                    explanation="persistent military buildup with outbound movement",
                )

                persist_source_runs(cursor, reports)
                persist_monitoring_state(
                    cursor,
                    [region],
                    [(25.0, 45.0, 27.0, 47.0, 4, 2, 1.6, 9.0)],
                    reports,
                    snapshot_time=100.0,
                )
                cursor.execute(
                    """
                    UPDATE current_region_state
                    SET evidence_summary = ?
                    """,
                    (
                        json.dumps(
                            {
                                "explanation": "persistent military buildup with outbound movement",
                                "region_type": "CIVILIAN_HEAVY",
                            },
                            sort_keys=True,
                        ),
                    ),
                )
                conn.commit()

                export_paths = export_monitoring_snapshot(cursor, Path(tmpdir) / "exports", 100.0)

                self.assertIn("validation_report", export_paths)
                payload = json.loads(Path(export_paths["validation_report"]).read_text(encoding="utf-8"))
                self.assertEqual(payload["generated_at"], 100.0)
                self.assertEqual(payload["metrics"]["civilian_heavy_visible_region_count"], 1)
                self.assertEqual(payload["metrics"]["degraded_primary_live_source_count"], 1)
                self.assertEqual(payload["metrics"]["visible_route_experimental_only_count"], 0)
                self.assertGreaterEqual(payload["metrics"]["critical_visible_alert_count_under_degradation"], 1)
                self.assertEqual(payload["checks"]["civilian_heavy_visible_regions"], "warn")
                self.assertEqual(payload["checks"]["degraded_primary_live_sources"], "warn")
                self.assertEqual(payload["checks"]["experimental_only_visible_routes"], "pass")
                self.assertEqual(payload["checks"]["critical_alerts_under_degradation"], "warn")
                self.assertIn("AIS_API_KEY", payload["summary"]["headline"])
                self.assertIn("restore_primary_live_sources", payload["recommended_actions"])
                self.assertIn("review_civilian_noise_thresholds", payload["recommended_actions"])
                self.assertIn("validate_critical_alerts_under_degradation", payload["recommended_actions"])
            finally:
                conn.close()

    def test_export_monitoring_snapshot_validation_report_warns_on_experimental_only_visible_routes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "monitoring.db"
            conn, cursor = init_db(str(db_path))
            try:
                reports = [
                    {
                        "source_name": "satellite",
                        "source_tier": "experimental",
                        "started_at": 100.0,
                        "finished_at": 101.0,
                        "status": "success",
                        "latency_seconds": 1.0,
                        "item_count": 3,
                        "error_reason": "",
                    }
                ]
                persist_monitoring_state(
                    cursor,
                    [],
                    [(25.0, 45.0, 27.0, 47.0, 3, 1, 1.6, 8.0)],
                    reports,
                    snapshot_time=100.0,
                )
                conn.commit()

                export_paths = export_monitoring_snapshot(cursor, Path(tmpdir) / "exports", 100.0)
                payload = json.loads(Path(export_paths["validation_report"]).read_text(encoding="utf-8"))

                self.assertEqual(payload["metrics"]["visible_route_experimental_only_count"], 1)
                self.assertEqual(payload["checks"]["experimental_only_visible_routes"], "warn")
                self.assertIn("review_experimental_route_weighting", payload["recommended_actions"])
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
