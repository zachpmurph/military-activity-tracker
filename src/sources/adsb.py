from __future__ import annotations

import time
from typing import Dict, List, Tuple

import requests

from core.monitoring_config import adsb_watch_areas
from .base import normalize_aircraft


def _watch_area_url(area: Dict) -> str:
    adsb = area["adsb"]
    return (
        f"https://api.adsb.lol/v2/lat/{adsb['lat']}/lon/{adsb['lon']}/dist/{adsb['dist_km']}"
    )


def fetch_with_report() -> Tuple[List[dict], dict]:
    started = time.time()
    all_aircraft: List[dict] = []
    area_reports: List[dict] = []

    for area in adsb_watch_areas():
        url = _watch_area_url(area)
        area_started = time.time()
        status = "success"
        error_reason = ""
        aircraft_list = []

        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                status = "http_error"
                error_reason = f"status_code={response.status_code}"
            else:
                data = response.json()
                aircraft_list = data.get("ac") or data.get("aircraft") or []
                for a in aircraft_list:
                    raw = {
                        "source": url,
                        "source_name": "adsb",
                        "source_tier": "primary_live",
                        "theater_id": area["theater_id"],
                        "icao24": a.get("hex"),
                        "callsign": (a.get("flight") or "").strip(),
                        "lat": a.get("lat"),
                        "lon": a.get("lon"),
                        "altitude": a.get("alt_baro"),
                        "velocity": a.get("gs"),
                    }
                    if raw["lat"] is None or raw["lon"] is None:
                        continue
                    all_aircraft.append(normalize_aircraft(raw))
        except Exception as exc:
            status = "exception"
            error_reason = str(exc)

        area_reports.append(
            {
                "source_name": "adsb",
                "source_tier": "primary_live",
                "watch_area_id": area["theater_id"],
                "started_at": area_started,
                "finished_at": time.time(),
                "status": status,
                "latency_seconds": round(time.time() - area_started, 3),
                "item_count": len(aircraft_list),
                "error_reason": error_reason,
            }
        )

    overall_status = "success" if any(r["item_count"] > 0 for r in area_reports) else "degraded"
    if any(r["status"] not in {"success"} for r in area_reports):
        overall_status = "degraded" if overall_status == "success" else "failed"

    report = {
        "source_name": "adsb",
        "source_tier": "primary_live",
        "started_at": started,
        "finished_at": time.time(),
        "status": overall_status,
        "latency_seconds": round(time.time() - started, 3),
        "item_count": len(all_aircraft),
        "error_reason": "; ".join(
            f"{r['watch_area_id']}:{r['error_reason']}"
            for r in area_reports
            if r["error_reason"]
        ),
        "details": area_reports,
    }
    return all_aircraft, report


def fetch() -> List[dict]:
    aircraft, _report = fetch_with_report()
    return aircraft
