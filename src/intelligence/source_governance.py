from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.monitoring_config import ais_bounding_boxes
from intelligence.ais_ingestion import fetch_ais_signals
from intelligence.maritime_ingestion import fetch_maritime_signals
from intelligence.notam_ingestion import fetch_notam_signals
from intelligence.satellite_ingestion import fetch_satellite_signals


DEFERRED_EXPERIMENTAL_SOURCES = [
    "commercial_satellite_api",
    "sar_change_detection_api",
    "military_notice_scraper",
    "airport_port_notice_scraper",
    "public_logistics_scraper",
]


def _report(
    source_name: str,
    source_tier: str,
    started_at: float,
    finished_at: float,
    status: str,
    item_count: int,
    error_reason: str = "",
) -> Dict:
    return {
        "source_name": source_name,
        "source_tier": source_tier,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "latency_seconds": round(finished_at - started_at, 3),
        "item_count": item_count,
        "error_reason": error_reason,
    }


def collect_operational_external_signals(
    db_path: Optional[Path] = None,
    development_mode: bool = False,
) -> Tuple[List, List[Dict]]:
    """
    Return operational external signals and per-source run metadata.

    Operational v1 includes only live/API-backed maritime sources:
      - AISStream (primary_live)
      - Global Fishing Watch (secondary_live)

    NOTAM-like feeds and synthetic satellite signals are deliberately excluded
    from primary scoring in this phase.
    """
    signals = []
    reports: List[Dict] = []

    started = time.time()
    allow_fallback = development_mode
    ais_key = os.environ.get("AIS_API_KEY", "")
    if not ais_key and not allow_fallback:
        reports.append(
            _report(
                "aisstream",
                "primary_live",
                started,
                time.time(),
                "missing_credentials",
                0,
                "AIS_API_KEY not set",
            )
        )
    else:
        ais_started = time.time()
        ais_signals = fetch_ais_signals(
            bounding_boxes=ais_bounding_boxes() or None,
            allow_fallback=allow_fallback,
        )
        ais_status = "success" if ais_signals else "no_live_data"
        reports.append(
            _report(
                "aisstream",
                "primary_live",
                ais_started,
                time.time(),
                ais_status,
                len(ais_signals),
                "" if ais_signals else "No AISStream signals collected",
            )
        )
        signals.extend(ais_signals)

    gfw_started = time.time()
    gfw_key = os.environ.get("GFW_API_KEY", "")
    if not gfw_key and not allow_fallback:
        reports.append(
            _report(
                "gfw",
                "secondary_live",
                gfw_started,
                time.time(),
                "missing_credentials",
                0,
                "GFW_API_KEY not set",
            )
        )
    else:
        gfw_signals = fetch_maritime_signals(allow_fallback=allow_fallback)
        gfw_status = "success" if gfw_signals else "no_live_data"
        reports.append(
            _report(
                "gfw",
                "secondary_live",
                gfw_started,
                time.time(),
                gfw_status,
                len(gfw_signals),
                "" if gfw_signals else "No GFW signals collected",
            )
        )
        signals.extend(gfw_signals)

    # Experimental sources are intentionally excluded from primary scoring in v1.
    _ = db_path
    _ = fetch_notam_signals
    _ = fetch_satellite_signals
    return signals, reports
