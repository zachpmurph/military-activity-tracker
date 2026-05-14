from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from core.monitoring_config import theater_for_point
from intelligence.classifier import Classification


_SEVERITY_ORDER = ["INFO", "WATCH", "PRIORITY", "CRITICAL"]
_ALERT_COOLDOWN_SECONDS = 1800.0


def _hash_id(*parts: object) -> str:
    return hashlib.md5("|".join(str(part) for part in parts).encode()).hexdigest()[:12]


def region_state_id(lat: float, lon: float) -> str:
    return _hash_id("region", round(lat, 1), round(lon, 1))


def route_state_id(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float) -> str:
    return _hash_id(
        "route",
        round(origin_lat, 1),
        round(origin_lon, 1),
        round(dest_lat, 1),
        round(dest_lon, 1),
    )


def _severity_from_risk(risk_24h: float) -> str:
    if risk_24h >= 80.0:
        return "CRITICAL"
    if risk_24h >= 60.0:
        return "PRIORITY"
    if risk_24h >= 35.0:
        return "WATCH"
    return "INFO"


def _downgrade_severity(severity: str) -> str:
    index = _SEVERITY_ORDER.index(severity)
    return _SEVERITY_ORDER[max(index - 1, 0)]


def _source_summary(source_reports: Sequence[Dict]) -> str:
    return json.dumps(
        {
            report["source_name"]: {
                "tier": report["source_tier"],
                "status": report["status"],
                "item_count": report["item_count"],
            }
            for report in source_reports
        },
        sort_keys=True,
    )


def _has_live_operational_evidence(source_reports: Sequence[Dict]) -> bool:
    return any(
        report["source_tier"] in {"primary_live", "secondary_live"}
        and report["status"] == "success"
        and report["item_count"] > 0
        for report in source_reports
    )


def _resolve_region_state_id(
    cursor,
    lat: float,
    lon: float,
    *,
    used_region_ids: set[str],
    match_threshold: float = 0.3,
) -> str:
    exact_region_id = region_state_id(lat, lon)
    exact_match = cursor.execute(
        "SELECT region_id FROM current_region_state WHERE region_id = ?",
        (exact_region_id,),
    ).fetchone()
    if exact_match and exact_region_id not in used_region_ids:
        return exact_region_id

    candidate_rows = cursor.execute(
        """
        SELECT region_id, lat, lon
        FROM current_region_state
        WHERE ABS(lat - ?) <= ?
          AND ABS(lon - ?) <= ?
        """,
        (lat, match_threshold, lon, match_threshold),
    ).fetchall()

    best_match = None
    best_distance = None
    for region_id, candidate_lat, candidate_lon in candidate_rows:
        if region_id in used_region_ids:
            continue
        distance = math.sqrt((lat - candidate_lat) ** 2 + (lon - candidate_lon) ** 2)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_match = region_id

    if best_match is not None:
        return best_match
    return exact_region_id


def _compute_region_risks(region, history: Dict[str, float] | None = None) -> Tuple[float, float]:
    scores = region.all_scores
    features = region.features
    staging = scores.get("STAGING", 0.0)
    projection = scores.get("PROJECTION", 0.0)
    coordinated = scores.get("COORDINATED_ACTIVITY", 0.0)
    anomaly = scores.get("ANOMALY", 0.0)

    military_ratio = features.military_ratio
    recurrence_bonus = min(features.recurring_appearances * 3.0, 15.0)
    change_bonus = min(features.change_score, 40.0) * 0.5
    new_entry_bonus = min(features.new_aircraft * 2.5, 15.0)
    movement_bonus = min((features.outflow_count + features.inflow_count) * 2.0, 12.0)

    base = (
        (staging * 0.35)
        + (projection * 0.30)
        + (coordinated * 0.20)
        + (anomaly * 0.15)
    )
    risk_24h = min(
        100.0,
        base
        + (military_ratio * 20.0)
        + recurrence_bonus
        + change_bonus
        + new_entry_bonus
        + movement_bonus,
    )
    risk_6h = min(100.0, (risk_24h * 0.8) + (5.0 if features.spike_flag else 0.0))
    if history:
        history_bonus_24h = min(
            (history.get("appearance_count_6h", 0) * 1.5)
            + (history.get("appearance_count_24h", 0) * 0.5),
            8.0,
        )
        history_bonus_6h = min(history.get("appearance_count_6h", 0) * 1.5, 6.0)
        risk_24h = min(100.0, risk_24h + history_bonus_24h)
        risk_6h = min(100.0, risk_6h + history_bonus_6h)
    return round(risk_6h, 1), round(risk_24h, 1)


def _compute_region_rollups(
    cursor,
    *,
    region_id: str,
    snapshot_time: float,
    aircraft_count: int,
    military_count: int,
    risk_6h: float,
    risk_24h: float,
) -> Dict[str, float]:
    six_hour_row = cursor.execute(
        """
        SELECT COUNT(*), COALESCE(MAX(aircraft_count), 0), COALESCE(MAX(military_count), 0),
               COALESCE(MAX(military_action_risk_6h), 0)
        FROM region_snapshots
        WHERE region_id = ?
          AND snapshot_time >= ?
        """,
        (region_id, snapshot_time - 21600),
    ).fetchone()
    day_row = cursor.execute(
        """
        SELECT COUNT(*), COALESCE(MAX(aircraft_count), 0), COALESCE(MAX(military_count), 0),
               COALESCE(MAX(military_action_risk_24h), 0)
        FROM region_snapshots
        WHERE region_id = ?
          AND snapshot_time >= ?
        """,
        (region_id, snapshot_time - 86400),
    ).fetchone()
    return {
        "appearance_count_6h": int(six_hour_row[0]) + 1,
        "appearance_count_24h": int(day_row[0]) + 1,
        "max_aircraft_6h": max(int(six_hour_row[1]), aircraft_count),
        "max_aircraft_24h": max(int(day_row[1]), aircraft_count),
        "max_military_6h": max(int(six_hour_row[2]), military_count),
        "max_military_24h": max(int(day_row[2]), military_count),
        "max_risk_6h": max(float(six_hour_row[3]), risk_6h),
        "max_risk_24h": max(float(day_row[3]), risk_24h),
    }


def _region_forecast(region, *, rollups: Dict[str, float], risk_6h: float, risk_24h: float, direction: str) -> Dict[str, object]:
    drivers: List[str] = []
    if rollups["appearance_count_6h"] > 1:
        drivers.append("repeat_presence_6h")
    if rollups["appearance_count_24h"] > 1:
        drivers.append("repeat_presence_24h")
    if region.features.military_count >= 1:
        drivers.append("military_presence")
    if region.features.staging_flag:
        drivers.append("staging_signal")
    if region.features.new_aircraft > 0:
        drivers.append("new_aircraft")
    return {
        "direction": direction,
        "military_action_risk_6h": risk_6h,
        "military_action_risk_24h": risk_24h,
        "drivers": drivers,
    }


def _route_forecast(
    *,
    metrics: Dict[str, float],
    appearance_count: int,
    risk_6h: float,
    risk_24h: float,
    direction: str,
    route_provenance: Dict[str, object],
) -> Dict[str, object]:
    drivers: List[str] = []
    if appearance_count > 1:
        drivers.append("repeat_flow_route")
    if metrics["military_moved"] >= 1:
        drivers.append("military_route")
    if route_provenance["live_source_count"] > 0:
        drivers.append("live_source_support")
    return {
        "direction": direction,
        "military_action_risk_6h": risk_6h,
        "military_action_risk_24h": risk_24h,
        "drivers": drivers,
    }


def _route_metrics(route_row: Tuple) -> Dict[str, float]:
    if len(route_row) == 7:
        (
            origin_lat,
            origin_lon,
            dest_lat,
            dest_lon,
            aircraft_moved,
            military_moved,
            avg_distance,
        ) = route_row
        flow_score = round((aircraft_moved * 1.0) + (military_moved * 2.5), 1)
    elif len(route_row) == 8:
        (
            origin_lat,
            origin_lon,
            dest_lat,
            dest_lon,
            aircraft_moved,
            military_moved,
            avg_distance,
            flow_score,
        ) = route_row
    else:
        raise ValueError(f"Unsupported route row shape: {route_row!r}")

    return {
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "dest_lat": dest_lat,
        "dest_lon": dest_lon,
        "aircraft_moved": aircraft_moved,
        "military_moved": military_moved,
        "avg_distance": avg_distance,
        "flow_score": flow_score,
    }


def _compute_route_risks(route_row: Tuple, history: Dict[str, float] | None = None) -> Tuple[float, float]:
    metrics = _route_metrics(route_row)
    aircraft_moved = metrics["aircraft_moved"]
    military_moved = metrics["military_moved"]
    avg_distance = metrics["avg_distance"]
    flow_score = metrics["flow_score"]
    base = (aircraft_moved * 8.0) + (military_moved * 18.0) + (avg_distance * 10.0) + (flow_score * 2.0)
    risk_24h = min(100.0, base)
    risk_6h = min(100.0, (aircraft_moved * 6.0) + (military_moved * 15.0) + (avg_distance * 8.0) + (flow_score * 1.5))
    if history:
        history_bonus_24h = min(history.get("appearance_count", 0) * 2.0, 10.0)
        history_bonus_6h = min(history.get("appearance_count", 0) * 1.5, 8.0)
        risk_24h = min(100.0, risk_24h + history_bonus_24h)
        risk_6h = min(100.0, risk_6h + history_bonus_6h)
    return round(risk_6h, 1), round(risk_24h, 1)


def _route_trend_direction(previous_flow_score: float | None, current_flow_score: float) -> str:
    if previous_flow_score is None:
        return "rising"
    delta = current_flow_score - previous_flow_score
    if delta >= 2.0:
        return "strengthening"
    if delta <= -2.0:
        return "cooling"
    return "stable"


def _direction(previous_risk: float | None, current_risk: float) -> str:
    if previous_risk is None:
        return "rising"
    if current_risk - previous_risk >= 5.0:
        return "rising"
    if previous_risk - current_risk >= 5.0:
        return "cooling"
    return "stable"


def _direction_from_recent_history(
    cursor,
    *,
    table_name: str,
    id_column: str,
    state_id: str,
    current_risk: float,
    previous_risk: float | None,
) -> str:
    rows = cursor.execute(
        f"""
        SELECT military_action_risk_24h
        FROM {table_name}
        WHERE {id_column} = ?
        ORDER BY snapshot_time DESC
        LIMIT 3
        """,
        (state_id,),
    ).fetchall()
    if rows:
        average_risk = sum(row[0] for row in rows) / len(rows)
        rise_threshold = 2.0 if len(rows) == 1 else 3.0
        cool_threshold = 2.0 if len(rows) == 1 else 3.0
        if current_risk - average_risk >= rise_threshold:
            return "rising"
        if average_risk - current_risk >= cool_threshold:
            return "cooling"
    return _direction(previous_risk, current_risk)


def _route_source_provenance(source_reports: Sequence[Dict]) -> Dict[str, object]:
    successful_reports = [
        report
        for report in source_reports
        if report["status"] == "success" and report["item_count"] > 0
    ]
    live_reports = [
        report
        for report in successful_reports
        if report["source_tier"] in {"primary_live", "secondary_live"}
    ]
    experimental_reports = [
        report for report in successful_reports if report["source_tier"] == "experimental"
    ]
    return {
        "source_count": len(successful_reports),
        "live_source_count": len(live_reports),
        "experimental_source_count": len(experimental_reports),
        "source_names": [report["source_name"] for report in successful_reports],
        "live_source_names": [report["source_name"] for report in live_reports],
        "experimental_source_names": [report["source_name"] for report in experimental_reports],
    }


def _alert_policy(cursor, *, object_type: str, object_id: str, theater_id: str, alert_type: str) -> Dict[str, object]:
    watchlist_match = cursor.execute(
        """
        SELECT 1
        FROM watchlists
        WHERE active = 1
          AND object_type = ?
          AND (object_id = '' OR object_id = ?)
          AND (theater_id IS NULL OR theater_id = '' OR theater_id = ?)
        LIMIT 1
        """,
        (object_type, object_id, theater_id),
    ).fetchone() is not None
    suppressed = cursor.execute(
        """
        SELECT 1
        FROM suppression_rules
        WHERE active = 1
          AND object_type = ?
          AND alert_type = ?
          AND (object_id IS NULL OR object_id = '' OR object_id = ?)
          AND (theater_id IS NULL OR theater_id = '' OR theater_id = ?)
        LIMIT 1
        """,
        (object_type, alert_type, object_id, theater_id),
    ).fetchone() is not None
    return {
        "watchlist_match": watchlist_match,
        "priority_boost": 10.0 if watchlist_match else 0.0,
        "visible": 0 if suppressed else 1,
        "status": "suppressed" if suppressed else None,
    }


def persist_source_runs(cursor, source_reports: Sequence[Dict]) -> None:
    for report in source_reports:
        cursor.execute(
            """
            INSERT INTO source_runs (
                source_name, source_tier, started_at, finished_at, status,
                latency_seconds, item_count, error_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report["source_name"],
                report["source_tier"],
                report["started_at"],
                report["finished_at"],
                report["status"],
                report["latency_seconds"],
                report["item_count"],
                report.get("error_reason", ""),
            ),
        )

        degraded = 0 if report["status"] == "success" else 1
        cursor.execute(
            """
            INSERT INTO source_health (
                source_name, source_tier, last_status, last_started_at, last_finished_at,
                last_latency_seconds, last_item_count, last_error_reason, degraded
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_name) DO UPDATE SET
                source_tier = excluded.source_tier,
                last_status = excluded.last_status,
                last_started_at = excluded.last_started_at,
                last_finished_at = excluded.last_finished_at,
                last_latency_seconds = excluded.last_latency_seconds,
                last_item_count = excluded.last_item_count,
                last_error_reason = excluded.last_error_reason,
                degraded = excluded.degraded
            """,
            (
                report["source_name"],
                report["source_tier"],
                report["status"],
                report["started_at"],
                report["finished_at"],
                report["latency_seconds"],
                report["item_count"],
                report.get("error_reason", ""),
                degraded,
            ),
        )


def _upsert_alert(
    cursor,
    *,
    alert_type: str,
    object_type: str,
    object_id: str,
    theater_id: str,
    severity: str,
    priority_score: float,
    confidence: int,
    explanation: str,
    evidence_summary: str,
    snapshot_time: float,
) -> str:
    alert_id = _hash_id(alert_type, object_type, object_id)
    policy = _alert_policy(
        cursor,
        object_type=object_type,
        object_id=object_id,
        theater_id=theater_id,
        alert_type=alert_type,
    )
    effective_priority_score = min(100.0, priority_score + policy["priority_boost"])
    existing = cursor.execute(
        """
        SELECT first_seen, status, severity, priority_score, explanation
        FROM alerts
        WHERE alert_id = ?
        """,
        (alert_id,),
    ).fetchone()
    write_event = True

    if existing is None:
        status = policy["status"] or "new"
        first_seen = snapshot_time
        event_type = "opened"
        cursor.execute(
            """
            INSERT INTO alerts (
                alert_id, alert_type, object_type, object_id, theater_id,
                severity, priority_score, confidence, status, visible, watchlist_match,
                first_seen, last_seen,
                explanation, evidence_summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alert_id,
                alert_type,
                object_type,
                object_id,
                theater_id,
                severity,
                effective_priority_score,
                confidence,
                status,
                policy["visible"],
                1 if policy["watchlist_match"] else 0,
                first_seen,
                snapshot_time,
                explanation,
                evidence_summary,
            ),
        )
    else:
        first_seen, existing_status, existing_severity, existing_priority, existing_explanation = existing
        status = policy["status"] or (existing_status if existing_status not in {"resolved"} else "watching")
        event_type = "updated"
        if (
            existing_severity == severity
            and abs(existing_priority - effective_priority_score) < 3.0
            and existing_explanation == explanation
            and status == existing_status
        ):
            write_event = False
        cursor.execute(
            """
            UPDATE alerts
            SET theater_id = ?,
                severity = ?,
                priority_score = ?,
                confidence = ?,
                status = ?,
                visible = ?,
                watchlist_match = ?,
                last_seen = ?,
                explanation = ?,
                evidence_summary = ?
            WHERE alert_id = ?
            """,
            (
                theater_id,
                severity,
                effective_priority_score,
                confidence,
                status,
                policy["visible"],
                1 if policy["watchlist_match"] else 0,
                snapshot_time,
                explanation,
                evidence_summary,
                alert_id,
            ),
        )

    if write_event:
        cursor.execute(
            """
            INSERT INTO alert_events (
                alert_id, event_time, event_type, severity, priority_score, evidence_summary
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (alert_id, snapshot_time, event_type, severity, effective_priority_score, evidence_summary),
        )
    return alert_id


def _resolve_missing_alerts(cursor, active_alert_ids: Iterable[str], snapshot_time: float) -> None:
    active_alert_ids = set(active_alert_ids)
    rows = cursor.execute(
        """
        SELECT alert_id, severity, priority_score, evidence_summary, status, last_seen
        FROM alerts
        WHERE status IN ('new', 'watching', 'acknowledged')
        """
    ).fetchall()
    for alert_id, severity, priority_score, evidence_summary, status, last_seen in rows:
        if alert_id in active_alert_ids:
            continue
        if snapshot_time - last_seen <= _ALERT_COOLDOWN_SECONDS:
            cooled_severity = _downgrade_severity(severity)
            if status != "watching" or severity != cooled_severity:
                cursor.execute(
                    """
                    UPDATE alerts
                    SET status = 'watching',
                        severity = ?
                    WHERE alert_id = ?
                    """,
                    (cooled_severity, alert_id),
                )
                cursor.execute(
                    """
                    INSERT INTO alert_events (
                        alert_id, event_time, event_type, severity, priority_score, evidence_summary
                    )
                    VALUES (?, ?, 'cooling', ?, ?, ?)
                    """,
                    (alert_id, snapshot_time, cooled_severity, priority_score, evidence_summary),
                )
            continue
        cursor.execute(
            """
            UPDATE alerts
            SET status = 'resolved',
                last_seen = ?
            WHERE alert_id = ?
            """,
            (snapshot_time, alert_id),
        )
        cursor.execute(
            """
            INSERT INTO alert_events (
                alert_id, event_time, event_type, severity, priority_score, evidence_summary
            )
            VALUES (?, ?, 'resolved', ?, ?, ?)
            """,
            (alert_id, snapshot_time, severity, priority_score, evidence_summary),
        )


def _region_alerts(region, risk_24h: float) -> List[Tuple[str, str]]:
    alerts = []
    features = region.features
    if region.classification == Classification.STAGING and features.military_count >= 1:
        alerts.append(("STAGING_REGION", "region classified as staging"))
    if region.classification == Classification.PROJECTION and features.military_count >= 1:
        alerts.append(("PROJECTION_REGION", "region classified as projection"))
    if features.spike_flag and features.aircraft_count >= 3:
        alerts.append(("REGION_SURGE", "sudden regional surge detected"))
    if features.military_count >= 2 and features.change_score >= 16.0:
        alerts.append(("MILITARY_BUILDUP", "military buildup exceeds monitoring threshold"))
    if features.new_aircraft >= 3 and features.recurring_appearances <= 1:
        alerts.append(("EMERGING_REGION", "new aircraft cluster appeared in low-history region"))
    if features.change_level == "HIGH" or risk_24h >= 60.0:
        alerts.append(("ESCALATION_REGION", "regional escalation risk is elevated"))
    if features.new_aircraft >= 2:
        alerts.append(("NEW_ENTRY_CLUSTER", "new entry cluster detected"))
    return alerts


def _route_alert_explanation(metrics: Dict[str, float], trend_direction: str, route_provenance: Dict[str, object]) -> str:
    live_sources = route_provenance["live_source_names"]
    experimental_sources = route_provenance["experimental_source_names"]
    segments = [
        f"{trend_direction} cross-region flow route",
        f"{int(metrics['aircraft_moved'])} aircraft moved",
        f"{int(metrics['military_moved'])} military-linked",
    ]
    if live_sources:
        segments.append(f"live sources: {', '.join(live_sources)}")
    if experimental_sources:
        segments.append(f"experimental: {', '.join(experimental_sources)}")
    return "; ".join(segments)


def _route_alerts(
    route_row: Tuple,
    risk_24h: float,
    trend_direction: str,
    route_provenance: Dict[str, object],
) -> List[Tuple[str, str]]:
    metrics = _route_metrics(route_row)
    aircraft_moved = metrics["aircraft_moved"]
    military_moved = metrics["military_moved"]
    avg_distance = metrics["avg_distance"]
    flow_score = metrics["flow_score"]
    if aircraft_moved >= 2 and (
        military_moved >= 1
        or avg_distance >= 0.8
        or flow_score >= 6.5
        or risk_24h >= 55.0
    ):
        return [
            (
                "MAJOR_FLOW_ROUTE",
                _route_alert_explanation(metrics, trend_direction, route_provenance),
            )
        ]
    return []


def persist_monitoring_state(
    cursor,
    intelligence: Sequence,
    route_rows: Sequence[Tuple],
    source_reports: Sequence[Dict],
    snapshot_time: float,
) -> Dict:
    source_summary = _source_summary(source_reports)
    live_operational = _has_live_operational_evidence(source_reports)
    route_provenance = _route_source_provenance(source_reports)
    active_alert_ids: List[str] = []
    used_region_ids: set[str] = set()

    persist_source_runs(cursor, source_reports)

    for region in intelligence:
        region_id = _resolve_region_state_id(
            cursor,
            region.lat,
            region.lon,
            used_region_ids=used_region_ids,
        )
        used_region_ids.add(region_id)
        previous_region_state = cursor.execute(
            """
            SELECT theater_id, lat, lon, military_action_risk_24h,
                   appearance_count_6h, appearance_count_24h
            FROM current_region_state
            WHERE region_id = ?
            """,
            (region_id,),
        ).fetchone()
        theater_id = theater_for_point(
            region.lat,
            region.lon,
            previous_theater_id=previous_region_state[0] if previous_region_state else None,
            previous_lat=previous_region_state[1] if previous_region_state else None,
            previous_lon=previous_region_state[2] if previous_region_state else None,
        )
        region_history = (
            {
                "appearance_count_6h": previous_region_state[4],
                "appearance_count_24h": previous_region_state[5],
            }
            if previous_region_state
            else None
        )
        risk_6h, risk_24h = _compute_region_risks(region, history=region_history)
        previous_risk = previous_region_state[3] if previous_region_state else None
        escalation_direction = _direction_from_recent_history(
            cursor,
            table_name="region_snapshots",
            id_column="region_id",
            state_id=region_id,
            current_risk=risk_24h,
            previous_risk=previous_risk,
        )
        severity = _severity_from_risk(risk_24h)
        if severity == "CRITICAL" and not live_operational:
            severity = _downgrade_severity(severity)
        rollups = _compute_region_rollups(
            cursor,
            region_id=region_id,
            snapshot_time=snapshot_time,
            aircraft_count=region.features.aircraft_count,
            military_count=region.features.military_count,
            risk_6h=risk_6h,
            risk_24h=risk_24h,
        )

        evidence_payload = {
            "explanation": region.explanation,
            "scores": region.all_scores,
            "military_count": region.features.military_count,
            "new_aircraft": region.features.new_aircraft,
            "change_score": region.features.change_score,
            "rollups": rollups,
            "forecast": _region_forecast(
                region,
                rollups=rollups,
                risk_6h=risk_6h,
                risk_24h=risk_24h,
                direction=escalation_direction,
            ),
        }
        evidence_summary = json.dumps(evidence_payload, sort_keys=True)

        cursor.execute(
            """
            INSERT INTO current_region_state (
                region_id, lat, lon, theater_id, classification, confidence, score,
                operational_severity, escalation_direction, military_action_risk_6h,
                military_action_risk_24h, aircraft_count, military_count, new_aircraft,
                appearance_count_6h, appearance_count_24h, max_aircraft_6h, max_aircraft_24h,
                max_military_6h, max_military_24h, max_risk_6h, max_risk_24h,
                source_summary, evidence_summary, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(region_id) DO UPDATE SET
                lat = excluded.lat,
                lon = excluded.lon,
                theater_id = excluded.theater_id,
                classification = excluded.classification,
                confidence = excluded.confidence,
                score = excluded.score,
                operational_severity = excluded.operational_severity,
                escalation_direction = excluded.escalation_direction,
                military_action_risk_6h = excluded.military_action_risk_6h,
                military_action_risk_24h = excluded.military_action_risk_24h,
                aircraft_count = excluded.aircraft_count,
                military_count = excluded.military_count,
                new_aircraft = excluded.new_aircraft,
                appearance_count_6h = excluded.appearance_count_6h,
                appearance_count_24h = excluded.appearance_count_24h,
                max_aircraft_6h = excluded.max_aircraft_6h,
                max_aircraft_24h = excluded.max_aircraft_24h,
                max_military_6h = excluded.max_military_6h,
                max_military_24h = excluded.max_military_24h,
                max_risk_6h = excluded.max_risk_6h,
                max_risk_24h = excluded.max_risk_24h,
                source_summary = excluded.source_summary,
                evidence_summary = excluded.evidence_summary,
                updated_at = excluded.updated_at
            """,
            (
                region_id,
                region.lat,
                region.lon,
                theater_id,
                region.classification.value,
                region.confidence,
                region.score,
                severity,
                escalation_direction,
                risk_6h,
                risk_24h,
                region.features.aircraft_count,
                region.features.military_count,
                region.features.new_aircraft,
                rollups["appearance_count_6h"],
                rollups["appearance_count_24h"],
                rollups["max_aircraft_6h"],
                rollups["max_aircraft_24h"],
                rollups["max_military_6h"],
                rollups["max_military_24h"],
                rollups["max_risk_6h"],
                rollups["max_risk_24h"],
                source_summary,
                evidence_summary,
                snapshot_time,
            ),
        )
        cursor.execute(
            """
            INSERT INTO region_snapshots (
                region_id, snapshot_time, classification, confidence, score,
                operational_severity, escalation_direction, military_action_risk_6h,
                military_action_risk_24h, aircraft_count, military_count, new_aircraft,
                appearance_count_6h, appearance_count_24h, max_aircraft_6h, max_aircraft_24h,
                max_military_6h, max_military_24h, max_risk_6h, max_risk_24h,
                source_summary, evidence_summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                region_id,
                snapshot_time,
                region.classification.value,
                region.confidence,
                region.score,
                severity,
                escalation_direction,
                risk_6h,
                risk_24h,
                region.features.aircraft_count,
                region.features.military_count,
                region.features.new_aircraft,
                rollups["appearance_count_6h"],
                rollups["appearance_count_24h"],
                rollups["max_aircraft_6h"],
                rollups["max_aircraft_24h"],
                rollups["max_military_6h"],
                rollups["max_military_24h"],
                rollups["max_risk_6h"],
                rollups["max_risk_24h"],
                source_summary,
                evidence_summary,
            ),
        )

        for alert_type, explanation in _region_alerts(region, risk_24h):
            alert_id = _upsert_alert(
                cursor,
                alert_type=alert_type,
                object_type="region",
                object_id=region_id,
                theater_id=theater_id,
                severity=severity,
                priority_score=risk_24h,
                confidence=region.confidence,
                explanation=explanation,
                evidence_summary=evidence_summary,
                snapshot_time=snapshot_time,
            )
            active_alert_ids.append(alert_id)

    for route_row in route_rows:
        metrics = _route_metrics(route_row)
        origin_lat = metrics["origin_lat"]
        origin_lon = metrics["origin_lon"]
        dest_lat = metrics["dest_lat"]
        dest_lon = metrics["dest_lon"]
        aircraft_moved = metrics["aircraft_moved"]
        military_moved = metrics["military_moved"]
        avg_distance = metrics["avg_distance"]
        flow_score = metrics["flow_score"]
        route_id = route_state_id(origin_lat, origin_lon, dest_lat, dest_lon)
        midpoint_lat = (origin_lat + dest_lat) / 2.0
        midpoint_lon = (origin_lon + dest_lon) / 2.0
        theater_id = theater_for_point(midpoint_lat, midpoint_lon)
        previous = cursor.execute(
            """
            SELECT military_action_risk_24h, flow_score, appearance_count
            FROM current_route_state
            WHERE route_id = ?
            """,
            (route_id,),
        ).fetchone()
        previous_risk = previous[0] if previous else None
        previous_flow_score = previous[1] if previous else None
        previous_appearance_count = previous[2] if previous else 0
        route_history = {"appearance_count": previous_appearance_count} if previous else None
        risk_6h, risk_24h = _compute_route_risks(route_row, history=route_history)
        escalation_direction = _direction_from_recent_history(
            cursor,
            table_name="route_snapshots",
            id_column="route_id",
            state_id=route_id,
            current_risk=risk_24h,
            previous_risk=previous_risk,
        )
        trend_direction = _route_trend_direction(previous_flow_score, flow_score)
        appearance_count = previous_appearance_count + 1
        severity = _severity_from_risk(risk_24h)
        if severity == "CRITICAL" and not live_operational:
            severity = _downgrade_severity(severity)

        evidence_payload = {
            "origin": [origin_lat, origin_lon],
            "destination": [dest_lat, dest_lon],
            "aircraft_moved": aircraft_moved,
            "military_moved": military_moved,
            "avg_distance": avg_distance,
            "flow_score": flow_score,
            "trend_direction": trend_direction,
            "appearance_count": appearance_count,
            "source_provenance": route_provenance,
            "forecast": _route_forecast(
                metrics=metrics,
                appearance_count=appearance_count,
                risk_6h=risk_6h,
                risk_24h=risk_24h,
                direction=escalation_direction,
                route_provenance=route_provenance,
            ),
        }
        evidence_summary = json.dumps(evidence_payload, sort_keys=True)

        cursor.execute(
            """
            INSERT INTO current_route_state (
                route_id, origin_lat, origin_lon, dest_lat, dest_lon, theater_id,
                aircraft_moved, military_moved, avg_distance, flow_score,
                operational_severity, escalation_direction, trend_direction,
                appearance_count, source_count, live_source_count, experimental_source_count,
                military_action_risk_6h, military_action_risk_24h, source_summary,
                evidence_summary, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(route_id) DO UPDATE SET
                origin_lat = excluded.origin_lat,
                origin_lon = excluded.origin_lon,
                dest_lat = excluded.dest_lat,
                dest_lon = excluded.dest_lon,
                theater_id = excluded.theater_id,
                aircraft_moved = excluded.aircraft_moved,
                military_moved = excluded.military_moved,
                avg_distance = excluded.avg_distance,
                flow_score = excluded.flow_score,
                operational_severity = excluded.operational_severity,
                escalation_direction = excluded.escalation_direction,
                trend_direction = excluded.trend_direction,
                appearance_count = excluded.appearance_count,
                source_count = excluded.source_count,
                live_source_count = excluded.live_source_count,
                experimental_source_count = excluded.experimental_source_count,
                military_action_risk_6h = excluded.military_action_risk_6h,
                military_action_risk_24h = excluded.military_action_risk_24h,
                source_summary = excluded.source_summary,
                evidence_summary = excluded.evidence_summary,
                updated_at = excluded.updated_at
            """,
            (
                route_id,
                origin_lat,
                origin_lon,
                dest_lat,
                dest_lon,
                theater_id,
                aircraft_moved,
                military_moved,
                avg_distance,
                flow_score,
                severity,
                escalation_direction,
                trend_direction,
                appearance_count,
                route_provenance["source_count"],
                route_provenance["live_source_count"],
                route_provenance["experimental_source_count"],
                risk_6h,
                risk_24h,
                source_summary,
                evidence_summary,
                snapshot_time,
            ),
        )
        cursor.execute(
            """
            INSERT INTO route_snapshots (
                route_id, snapshot_time, aircraft_moved, military_moved, avg_distance,
                flow_score, operational_severity, escalation_direction, trend_direction,
                appearance_count, source_count, live_source_count, experimental_source_count,
                military_action_risk_6h, military_action_risk_24h, source_summary, evidence_summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                route_id,
                snapshot_time,
                aircraft_moved,
                military_moved,
                avg_distance,
                flow_score,
                severity,
                escalation_direction,
                trend_direction,
                appearance_count,
                route_provenance["source_count"],
                route_provenance["live_source_count"],
                route_provenance["experimental_source_count"],
                risk_6h,
                risk_24h,
                source_summary,
                evidence_summary,
            ),
        )

        for alert_type, explanation in _route_alerts(
            route_row,
            risk_24h,
            trend_direction,
            route_provenance,
        ):
            alert_id = _upsert_alert(
                cursor,
                alert_type=alert_type,
                object_type="route",
                object_id=route_id,
                theater_id=theater_id,
                severity=severity,
                priority_score=risk_24h,
                confidence=min(100, 50 + (military_moved * 10) + (aircraft_moved * 5)),
                explanation=explanation,
                evidence_summary=evidence_summary,
                snapshot_time=snapshot_time,
            )
            active_alert_ids.append(alert_id)

    _resolve_missing_alerts(cursor, active_alert_ids, snapshot_time)
    return {
        "region_count": len(intelligence),
        "route_count": len(route_rows),
        "alert_count": len(active_alert_ids),
    }


def export_monitoring_snapshot(cursor, output_dir: Path, snapshot_time: float) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    theaters = [
        {
            "theater_id": row[0],
            "label": row[1],
            "center_lat": row[2],
            "center_lon": row[3],
            "radius_km": row[4],
            "active": bool(row[5]),
        }
        for row in cursor.execute(
            "SELECT theater_id, label, center_lat, center_lon, radius_km, active FROM theaters ORDER BY theater_id"
        ).fetchall()
    ]
    region_alerts: Dict[str, Dict[str, List[str]]] = {}
    for row in cursor.execute(
        """
        SELECT object_id, alert_type, visible
        FROM alerts
        WHERE object_type = 'region'
          AND status != 'resolved'
        ORDER BY alert_type
        """
    ).fetchall():
        region_alerts.setdefault(
            row[0],
            {"visible_alert_types": [], "suppressed_alert_types": []},
        )
        key = "visible_alert_types" if row[2] else "suppressed_alert_types"
        region_alerts[row[0]][key].append(row[1])
    route_alerts: Dict[str, Dict[str, List[str]]] = {}
    for row in cursor.execute(
        """
        SELECT object_id, alert_type, visible
        FROM alerts
        WHERE object_type = 'route'
          AND status != 'resolved'
        ORDER BY alert_type
        """
    ).fetchall():
        route_alerts.setdefault(
            row[0],
            {"visible_alert_types": [], "suppressed_alert_types": []},
        )
        key = "visible_alert_types" if row[2] else "suppressed_alert_types"
        route_alerts[row[0]][key].append(row[1])

    regions = [
        {
            "region_id": row[0],
            "lat": row[1],
            "lon": row[2],
            "theater_id": row[3],
            "classification": row[4],
            "confidence": row[5],
            "score": row[6],
            "operational_severity": row[7],
            "escalation_direction": row[8],
            "military_action_risk_6h": row[9],
            "military_action_risk_24h": row[10],
            "aircraft_count": row[11],
            "military_count": row[12],
            "new_aircraft": row[13],
            "appearance_count_6h": row[14],
            "appearance_count_24h": row[15],
            "max_aircraft_6h": row[16],
            "max_aircraft_24h": row[17],
            "max_military_6h": row[18],
            "max_military_24h": row[19],
            "max_risk_6h": row[20],
            "max_risk_24h": row[21],
            "source_summary": json.loads(row[22]),
            "evidence_summary": json.loads(row[23]),
            "updated_at": row[24],
            "visible": bool(region_alerts.get(row[0], {}).get("visible_alert_types")),
            "visible_alert_types": region_alerts.get(row[0], {}).get("visible_alert_types", []),
            "suppressed_alert_types": region_alerts.get(row[0], {}).get("suppressed_alert_types", []),
        }
        for row in cursor.execute(
            """
            SELECT region_id, lat, lon, theater_id, classification, confidence, score,
                   operational_severity, escalation_direction, military_action_risk_6h,
                   military_action_risk_24h, aircraft_count, military_count, new_aircraft,
                   appearance_count_6h, appearance_count_24h, max_aircraft_6h, max_aircraft_24h,
                   max_military_6h, max_military_24h, max_risk_6h, max_risk_24h,
                   source_summary, evidence_summary, updated_at
            FROM current_region_state
            ORDER BY military_action_risk_24h DESC, updated_at DESC
            """
        ).fetchall()
    ]
    routes = [
        {
            "route_id": row[0],
            "origin": [row[1], row[2]],
            "destination": [row[3], row[4]],
            "theater_id": row[5],
            "aircraft_moved": row[6],
            "military_moved": row[7],
            "avg_distance": row[8],
            "flow_score": row[9],
            "operational_severity": row[10],
            "escalation_direction": row[11],
            "trend_direction": row[12],
            "appearance_count": row[13],
            "source_count": row[14],
            "live_source_count": row[15],
            "experimental_source_count": row[16],
            "military_action_risk_6h": row[17],
            "military_action_risk_24h": row[18],
            "source_summary": json.loads(row[19]),
            "evidence_summary": json.loads(row[20]),
            "updated_at": row[21],
            "visible": bool(route_alerts.get(row[0], {}).get("visible_alert_types")),
            "visible_alert_types": route_alerts.get(row[0], {}).get("visible_alert_types", []),
            "suppressed_alert_types": route_alerts.get(row[0], {}).get("suppressed_alert_types", []),
        }
        for row in cursor.execute(
            """
            SELECT route_id, origin_lat, origin_lon, dest_lat, dest_lon, theater_id,
                   aircraft_moved, military_moved, avg_distance, flow_score,
                   operational_severity, escalation_direction, trend_direction,
                   appearance_count, source_count, live_source_count, experimental_source_count,
                   military_action_risk_6h, military_action_risk_24h, source_summary,
                   evidence_summary, updated_at
            FROM current_route_state
            ORDER BY military_action_risk_24h DESC, updated_at DESC
            """
        ).fetchall()
    ]
    source_health = [
        {
            "source_name": row[0],
            "source_tier": row[1],
            "last_status": row[2],
            "last_started_at": row[3],
            "last_finished_at": row[4],
            "last_latency_seconds": row[5],
            "last_item_count": row[6],
            "last_error_reason": row[7],
            "degraded": bool(row[8]),
        }
        for row in cursor.execute(
            """
            SELECT source_name, source_tier, last_status, last_started_at, last_finished_at,
                   last_latency_seconds, last_item_count, last_error_reason, degraded
            FROM source_health
            ORDER BY source_name
            """
        ).fetchall()
    ]
    active_alert_rows = cursor.execute(
        """
        SELECT theater_id, severity, visible
        FROM alerts
        WHERE status != 'resolved'
        """
    ).fetchall()
    alerts_by_severity: Dict[str, int] = {}
    for _theater_id, severity, _visible in active_alert_rows:
        alerts_by_severity[severity] = alerts_by_severity.get(severity, 0) + 1

    theater_counts: Dict[str, Dict[str, int]] = {}
    for theater_id, count in cursor.execute(
        """
        SELECT theater_id, COUNT(*)
        FROM current_region_state
        GROUP BY theater_id
        """
    ).fetchall():
        theater_counts.setdefault(theater_id, {"region_count": 0, "route_count": 0})
        theater_counts[theater_id]["region_count"] = count
    for theater_id, count in cursor.execute(
        """
        SELECT theater_id, COUNT(*)
        FROM current_route_state
        GROUP BY theater_id
        """
    ).fetchall():
        theater_counts.setdefault(theater_id, {"region_count": 0, "route_count": 0})
        theater_counts[theater_id]["route_count"] = count

    summary = {
        "generated_at": snapshot_time,
        "active_alert_count": len(active_alert_rows),
        "visible_alert_count": sum(1 for _theater_id, _severity, visible in active_alert_rows if visible),
        "suppressed_alert_count": sum(1 for _theater_id, _severity, visible in active_alert_rows if not visible),
        "alerts_by_severity": alerts_by_severity,
        "theaters": theater_counts,
        "source_health": {
            "healthy_count": sum(1 for row in source_health if not row["degraded"]),
            "degraded_count": sum(1 for row in source_health if row["degraded"]),
        },
    }
    alerts = [
        {
            "alert_id": row[0],
            "alert_type": row[1],
            "object_type": row[2],
            "object_id": row[3],
            "theater_id": row[4],
            "severity": row[5],
            "priority_score": row[6],
            "confidence": row[7],
            "status": row[8],
            "first_seen": row[9],
            "last_seen": row[10],
            "explanation": row[11],
            "evidence_summary": json.loads(row[12]),
            "visible": bool(row[13]),
        }
        for row in cursor.execute(
            """
            SELECT alert_id, alert_type, object_type, object_id, theater_id, severity,
                   priority_score, confidence, status, first_seen, last_seen,
                   explanation, evidence_summary, visible
            FROM alerts
            ORDER BY priority_score DESC, last_seen DESC
            """
        ).fetchall()
    ]

    theater_labels = {theater["theater_id"]: theater["label"] for theater in theaters}
    theater_priorities: Dict[str, Dict[str, object]] = {}

    def ensure_theater(theater_id: str) -> Dict[str, object]:
        entry = theater_priorities.get(theater_id)
        if entry is None:
            entry = {
                "theater_id": theater_id,
                "label": theater_labels.get(theater_id, theater_id.replace("_", " ").title()),
                "visible_alert_count": 0,
                "priority_alert_count": 0,
                "critical_alert_count": 0,
                "rising_region_count": 0,
                "rising_route_count": 0,
                "max_risk_24h": 0.0,
            }
            theater_priorities[theater_id] = entry
        return entry

    for theater_id, severity, visible in active_alert_rows:
        entry = ensure_theater(theater_id)
        if visible:
            entry["visible_alert_count"] += 1
        if severity == "PRIORITY":
            entry["priority_alert_count"] += 1
        elif severity == "CRITICAL":
            entry["critical_alert_count"] += 1

    for region in regions:
        entry = ensure_theater(region["theater_id"])
        if region["escalation_direction"] == "rising":
            entry["rising_region_count"] += 1
        entry["max_risk_24h"] = max(entry["max_risk_24h"], region["military_action_risk_24h"])

    for route in routes:
        entry = ensure_theater(route["theater_id"])
        if route["escalation_direction"] == "rising" or route["trend_direction"] in {"rising", "strengthening"}:
            entry["rising_route_count"] += 1
        entry["max_risk_24h"] = max(entry["max_risk_24h"], route["military_action_risk_24h"])

    top_theaters = []
    for entry in theater_priorities.values():
        priority_score = round(
            (entry["visible_alert_count"] * 10.0)
            + (entry["priority_alert_count"] * 12.0)
            + (entry["critical_alert_count"] * 16.0)
            + (entry["rising_region_count"] * 6.0)
            + (entry["rising_route_count"] * 5.0)
            + (entry["max_risk_24h"] * 0.5),
            1,
        )
        explanation = (
            f"{entry['visible_alert_count']} visible alerts, "
            f"{entry['rising_region_count']} rising regions, "
            f"{entry['rising_route_count']} rising routes, "
            f"max risk {entry['max_risk_24h']:.1f}"
        )
        top_theaters.append(
            {
                **entry,
                "priority_score": priority_score,
                "explanation": explanation,
            }
        )
    top_theaters.sort(
        key=lambda theater: (
            theater["priority_score"],
            theater["critical_alert_count"],
            theater["priority_alert_count"],
            theater["max_risk_24h"],
        ),
        reverse=True,
    )
    top_rising_theaters = []
    for theater in top_theaters:
        selection_score = round(
            (theater["rising_region_count"] * 8.0)
            + (theater["rising_route_count"] * 6.0)
            + (theater["visible_alert_count"] * 4.0)
            + (theater["critical_alert_count"] * 5.0)
            + (theater["max_risk_24h"] * 0.4),
            1,
        )
        top_rising_theaters.append(
            {
                "theater_id": theater["theater_id"],
                "label": theater["label"],
                "selection_score": selection_score,
                "rising_region_count": theater["rising_region_count"],
                "rising_route_count": theater["rising_route_count"],
                "visible_alert_count": theater["visible_alert_count"],
                "critical_alert_count": theater["critical_alert_count"],
                "max_risk_24h": theater["max_risk_24h"],
                "explanation": theater["explanation"],
            }
        )
    top_rising_theaters.sort(
        key=lambda theater: (
            theater["selection_score"],
            theater["rising_region_count"],
            theater["rising_route_count"],
            theater["max_risk_24h"],
        ),
        reverse=True,
    )
    top_military_corridors = []
    for route in routes:
        if not route["visible"]:
            continue
        if route["military_moved"] < 1:
            continue
        selection_score = round(
            (route["military_moved"] * 12.0)
            + (route["flow_score"] * 3.0)
            + (route["military_action_risk_24h"] * 0.5)
            + (route["live_source_count"] * 5.0)
            + (6.0 if route["trend_direction"] in {"rising", "strengthening"} else 0.0),
            1,
        )
        top_military_corridors.append(
            {
                "route_id": route["route_id"],
                "theater_id": route["theater_id"],
                "origin": route["origin"],
                "destination": route["destination"],
                "military_moved": route["military_moved"],
                "aircraft_moved": route["aircraft_moved"],
                "flow_score": route["flow_score"],
                "military_action_risk_24h": route["military_action_risk_24h"],
                "live_source_count": route["live_source_count"],
                "trend_direction": route["trend_direction"],
                "selection_score": selection_score,
                "explanation": (
                    f"{route['military_moved']} military-linked movements, "
                    f"flow {route['flow_score']:.1f}, "
                    f"{route['live_source_count']} live sources"
                ),
            }
        )
    top_military_corridors.sort(
        key=lambda route: (
            route["selection_score"],
            route["military_moved"],
            route["military_action_risk_24h"],
        ),
        reverse=True,
    )
    top_visible_alerts = []
    for alert in alerts:
        if not alert["visible"] or alert["status"] == "resolved":
            continue
        severity_weight = (_SEVERITY_ORDER.index(alert["severity"]) + 1) * 10.0
        selection_score = round(
            alert["priority_score"] + severity_weight + (alert["confidence"] * 0.2),
            1,
        )
        top_visible_alerts.append(
            {
                "alert_id": alert["alert_id"],
                "alert_type": alert["alert_type"],
                "object_type": alert["object_type"],
                "object_id": alert["object_id"],
                "theater_id": alert["theater_id"],
                "severity": alert["severity"],
                "visible": alert["visible"],
                "status": alert["status"],
                "confidence": alert["confidence"],
                "selection_score": selection_score,
                "explanation": alert["evidence_summary"].get("explanation", alert["explanation"]),
            }
        )
    top_visible_alerts.sort(
        key=lambda alert: (
            alert["selection_score"],
            alert["confidence"],
        ),
        reverse=True,
    )

    top_routes = []
    for route in routes:
        if not route["visible"]:
            continue
        if not (
            route["military_moved"] >= 1
            or route["flow_score"] >= 6.0
            or route["military_action_risk_24h"] >= 50.0
        ):
            continue
        trend_bonus = 8.0 if route["trend_direction"] == "strengthening" else 5.0 if route["trend_direction"] == "rising" else 0.0
        escalation_bonus = 5.0 if route["escalation_direction"] == "rising" else 0.0
        priority_score = round(
            route["military_action_risk_24h"]
            + (route["flow_score"] * 2.0)
            + (route["live_source_count"] * 4.0)
            + trend_bonus
            + escalation_bonus,
            1,
        )
        top_routes.append(
            {
                "route_id": route["route_id"],
                "theater_id": route["theater_id"],
                "origin": route["origin"],
                "destination": route["destination"],
                "operational_severity": route["operational_severity"],
                "escalation_direction": route["escalation_direction"],
                "trend_direction": route["trend_direction"],
                "flow_score": route["flow_score"],
                "military_action_risk_24h": route["military_action_risk_24h"],
                "live_source_count": route["live_source_count"],
                "priority_score": priority_score,
                "explanation": (
                    f"{route['trend_direction']} corridor, "
                    f"{route['live_source_count']} live sources, "
                    f"flow {route['flow_score']:.1f}, "
                    f"risk {route['military_action_risk_24h']:.1f}"
                ),
            }
        )
    top_routes.sort(
        key=lambda route: (
            route["priority_score"],
            route["military_action_risk_24h"],
            route["flow_score"],
        ),
        reverse=True,
    )

    top_alerts = [
        {
            "alert_id": alert["alert_id"],
            "alert_type": alert["alert_type"],
            "object_type": alert["object_type"],
            "object_id": alert["object_id"],
            "theater_id": alert["theater_id"],
            "severity": alert["severity"],
            "priority_score": alert["priority_score"],
            "confidence": alert["confidence"],
            "status": alert["status"],
            "visible": alert["visible"],
            "first_seen": alert["first_seen"],
            "last_seen": alert["last_seen"],
            "explanation": alert["evidence_summary"].get("explanation", alert["explanation"]),
        }
        for alert in alerts
        if alert["visible"] and alert["status"] != "resolved"
    ]
    top_alerts.sort(
        key=lambda alert: (
            alert["priority_score"],
            _SEVERITY_ORDER.index(alert["severity"]),
            alert["last_seen"],
        ),
        reverse=True,
    )

    source_tier_rank = {
        "primary_live": 0,
        "secondary_live": 1,
        "experimental": 2,
        "fallback_only": 3,
    }
    degraded_sources = [
        {
            "source_name": source["source_name"],
            "source_tier": source["source_tier"],
            "last_status": source["last_status"],
            "last_item_count": source["last_item_count"],
            "last_error_reason": source["last_error_reason"],
            "degraded": source["degraded"],
        }
        for source in source_health
        if source["degraded"]
    ]
    degraded_sources.sort(
        key=lambda source: (
            source_tier_rank.get(source["source_tier"], 99),
            source["source_name"],
        )
    )
    top_degraded_reason = degraded_sources[0]["last_error_reason"] if degraded_sources else ""
    degraded_source_constrained_view = {
        "degraded_source_count": len(degraded_sources),
        "top_theater_id": top_rising_theaters[0]["theater_id"] if top_rising_theaters else "global",
        "resilient_alert_count": len(top_visible_alerts),
        "headline": (
            f"{top_rising_theaters[0]['theater_id'] if top_rising_theaters else 'global'} remains the strongest theater "
            f"despite degraded sources: {top_degraded_reason}"
            if degraded_sources
            else "No degraded live sources are currently constraining the operator view"
        ),
    }
    civilian_heavy_visible_region_count = sum(
        1
        for region in regions
        if region["visible"] and region["evidence_summary"].get("region_type") == "CIVILIAN_HEAVY"
    )
    visible_route_experimental_only_count = sum(
        1
        for route in routes
        if route["visible"] and route["live_source_count"] == 0 and route["experimental_source_count"] > 0
    )
    degraded_primary_live_source_count = sum(
        1
        for source in degraded_sources
        if source["source_tier"] == "primary_live"
    )
    critical_visible_alert_count_under_degradation = (
        sum(1 for alert in top_visible_alerts if alert["severity"] == "CRITICAL")
        if degraded_primary_live_source_count > 0
        else 0
    )
    recommended_actions = []
    if degraded_primary_live_source_count > 0:
        recommended_actions.append("restore_primary_live_sources")
    if civilian_heavy_visible_region_count > 0:
        recommended_actions.append("review_civilian_noise_thresholds")
    if visible_route_experimental_only_count > 0:
        recommended_actions.append("review_experimental_route_weighting")
    if critical_visible_alert_count_under_degradation > 0:
        recommended_actions.append("validate_critical_alerts_under_degradation")
    validation_report = {
        "generated_at": snapshot_time,
        "metrics": {
            "civilian_heavy_visible_region_count": civilian_heavy_visible_region_count,
            "visible_route_experimental_only_count": visible_route_experimental_only_count,
            "degraded_primary_live_source_count": degraded_primary_live_source_count,
            "critical_visible_alert_count_under_degradation": critical_visible_alert_count_under_degradation,
        },
        "checks": {
            "civilian_heavy_visible_regions": "warn" if civilian_heavy_visible_region_count > 0 else "pass",
            "experimental_only_visible_routes": "warn" if visible_route_experimental_only_count > 0 else "pass",
            "degraded_primary_live_sources": "warn" if degraded_primary_live_source_count > 0 else "pass",
            "critical_alerts_under_degradation": "warn" if critical_visible_alert_count_under_degradation > 0 else "pass",
        },
        "recommended_actions": recommended_actions,
        "summary": {
            "headline": (
                f"Validation warning: {civilian_heavy_visible_region_count} civilian-heavy visible regions and "
                f"{degraded_primary_live_source_count} degraded primary-live sources; "
                f"top issue: {degraded_sources[0]['last_error_reason']}"
                if degraded_sources
                else f"Validation pass: {civilian_heavy_visible_region_count} civilian-heavy visible regions and no degraded primary-live sources"
            ),
        },
    }

    top_theater_id = top_theaters[0]["theater_id"] if top_theaters else "global"
    summary_headline = (
        f"{top_theater_id} is the leading theater with "
        f"{len(top_alerts[:10])} visible priority signals and "
        f"{len(degraded_sources[:10])} degraded sources"
    )

    priority_brief = {
        "generated_at": snapshot_time,
        "top_theaters": top_theaters[:10],
        "top_routes": top_routes[:10],
        "top_alerts": top_alerts[:10],
        "degraded_sources": degraded_sources[:10],
        "summary": {
            "top_theater_id": top_theater_id,
            "visible_alert_count": len(top_alerts[:10]),
            "degraded_source_count": len(degraded_sources[:10]),
            "headline": summary_headline,
        },
    }
    operator_views = {
        "generated_at": snapshot_time,
        "top_rising_theaters": top_rising_theaters[:10],
        "top_military_corridors": top_military_corridors[:10],
        "top_visible_alerts": top_visible_alerts[:10],
        "degraded_source_constrained_view": degraded_source_constrained_view,
    }

    theaters_path = output_dir / "latest_theaters.json"
    regions_path = output_dir / "latest_regions.json"
    routes_path = output_dir / "latest_routes.json"
    source_health_path = output_dir / "latest_source_health.json"
    summary_path = output_dir / "latest_summary.json"
    alerts_path = output_dir / "latest_alerts.json"
    priority_brief_path = output_dir / "latest_priority_brief.json"
    operator_views_path = output_dir / "latest_operator_views.json"
    validation_report_path = output_dir / "latest_validation_report.json"
    theaters_path.write_text(json.dumps({"generated_at": snapshot_time, "theaters": theaters}, indent=2), encoding="utf-8")
    regions_path.write_text(json.dumps({"generated_at": snapshot_time, "regions": regions}, indent=2), encoding="utf-8")
    routes_path.write_text(json.dumps({"generated_at": snapshot_time, "routes": routes}, indent=2), encoding="utf-8")
    source_health_path.write_text(
        json.dumps({"generated_at": snapshot_time, "sources": source_health}, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    alerts_path.write_text(json.dumps({"generated_at": snapshot_time, "alerts": alerts}, indent=2), encoding="utf-8")
    priority_brief_path.write_text(json.dumps(priority_brief, indent=2), encoding="utf-8")
    operator_views_path.write_text(json.dumps(operator_views, indent=2), encoding="utf-8")
    validation_report_path.write_text(json.dumps(validation_report, indent=2), encoding="utf-8")
    return {
        "theaters": str(theaters_path),
        "regions": str(regions_path),
        "routes": str(routes_path),
        "source_health": str(source_health_path),
        "summary": str(summary_path),
        "alerts": str(alerts_path),
        "priority_brief": str(priority_brief_path),
        "operator_views": str(operator_views_path),
        "validation_report": str(validation_report_path),
    }
