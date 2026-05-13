"""
intelligence/maritime_ingestion.py
────────────────────────────────────
Maritime vessel traffic ingestion pipeline.

Produces MARITIME ExternalSignal objects from AIS / vessel-density data so
they can be injected into the feature extraction pipeline alongside NOTAM
signals via build_features(external_signals=...).

Data model
──────────
  fetch_maritime_signals()   Returns MARITIME ExternalSignals for active
                             vessel clusters.

Intensity normalization
───────────────────────
  vessel_count → intensity = min(count / _MAX_VESSELS_PER_CELL, 1.0)

  _MAX_VESSELS_PER_CELL = 20  (20+ vessels in a 1° cell → full intensity)

  This produces the full [0, 1] range:
    1–2 vessels  → 0.05–0.10  (low activity, ADVISORY-equivalent)
    5  vessels   → 0.25
    10 vessels   → 0.50
    15 vessels   → 0.75
    20+ vessels  → 1.00       (dense activity, PROHIBITED-equivalent)

Live fetch
──────────
  Set GFW_API_KEY to enable real-time vessel density data from the
  Global Fishing Watch Events API (https://globalfishingwatch.org/api/).
  When the key is absent or the request fails the pipeline falls back to
  src/data/maritime_fallback.json, then to built-in cluster data.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from intelligence.external_features import ExternalSignal, SignalType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GFW_EVENTS_URL       = "https://gateway.api.globalfishingwatch.org/v3/events"
_GFW_TIMEOUT          = 4              # seconds per request
_FALLBACK_JSON        = Path(__file__).resolve().parent.parent / "data" / "maritime_fallback.json"
_MAX_VESSELS_PER_CELL = 20             # count at which intensity saturates to 1.0
_GRID_RESOLUTION      = 1.0           # degrees — vessels binned into 1° grid cells

# Module-level cache: populated on the first call to fetch_maritime_signals().
_SIGNAL_CACHE: Optional[List[ExternalSignal]] = None

# Built-in cluster data used when both the API and the JSON fallback are
# unavailable.  Mirrors the strategic locations in maritime_fallback.json.
_BUILTIN_CLUSTERS = [
    # (lat, lon, vessel_count)
    (26.5,  56.2, 25),   # Strait of Hormuz
    (25.0,  52.0, 18),   # Persian Gulf
    ( 2.5, 101.5, 28),   # Strait of Malacca
    (16.0, 113.0, 20),   # South China Sea
    (44.0,  33.5, 12),   # Black Sea
    (34.0,  34.0, 10),   # Eastern Mediterranean
]


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _normalize_density(count: int, max_count: int = _MAX_VESSELS_PER_CELL) -> float:
    """
    Normalise a raw vessel count to [0, 1].

    Returns 0.0 for counts ≤ 0 and 1.0 for counts ≥ max_count.
    """
    if count <= 0:
        return 0.0
    return min(float(count) / max_count, 1.0)


# ---------------------------------------------------------------------------
# Real ingestion: Global Fishing Watch Events API
# ---------------------------------------------------------------------------

def _parse_gfw_response(data: dict) -> List[ExternalSignal]:
    """
    Parse a GFW Events API response dict into MARITIME ExternalSignal objects.

    Expects the GFW v3 events envelope::

        {
          "entries": [
            {
              "position": {"lat": 34.5, "lon": 56.2},
              ...
            }, ...
          ]
        }

    Entries are binned into _GRID_RESOLUTION° cells.  One ExternalSignal is
    created per non-empty cell with intensity derived from the event count.
    Entries with missing, null, or out-of-range positions are silently skipped.
    """
    counts: Dict[tuple, int] = {}
    for entry in data.get("entries", []):
        try:
            pos = entry.get("position") or {}
            lat = pos.get("lat")
            lon = pos.get("lon")
            if lat is None or lon is None:
                continue
            lat, lon = float(lat), float(lon)
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue
            cell = (
                round(lat / _GRID_RESOLUTION) * _GRID_RESOLUTION,
                round(lon / _GRID_RESOLUTION) * _GRID_RESOLUTION,
            )
            counts[cell] = counts.get(cell, 0) + 1
        except (TypeError, ValueError, AttributeError):
            continue

    signals: List[ExternalSignal] = []
    for (cell_lat, cell_lon), count in counts.items():
        signals.append(ExternalSignal(
            signal_type = SignalType.MARITIME,
            lat         = cell_lat,
            lon         = cell_lon,
            intensity   = _normalize_density(count),
            metadata    = {
                "source":       "maritime",
                "vessel_count": count,
            },
        ))
    return signals


def _fetch_gfw_vessels() -> Optional[List[ExternalSignal]]:
    """
    Fetch vessel activity from the GFW Events API.

    Reads GFW_API_KEY from the environment.  Returns None immediately when
    the key is absent or on any network / parse error (4 s timeout).
    Queries the last 7 days of public fishing events; the resulting positions
    are gridded to produce density-based MARITIME signals.
    """
    api_key = os.environ.get("GFW_API_KEY", "")
    if not api_key:
        return None

    try:
        end_date   = date.today()
        start_date = end_date - timedelta(days=7)
        params = urllib.parse.urlencode({
            "datasets[0]": "public-global-fishing-events:latest",
            "start-date":  start_date.isoformat(),
            "end-date":    end_date.isoformat(),
            "limit":       "200",
        })
        req = urllib.request.Request(
            f"{_GFW_EVENTS_URL}?{params}",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=_GFW_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
        return _parse_gfw_response(data)
    except Exception:
        return None


def _load_fallback_signals() -> List[ExternalSignal]:
    """
    Load MARITIME ExternalSignal objects from the repo-local static JSON file.

    Items with missing, invalid, or out-of-range lat/lon are skipped.
    Returns an empty list on any I/O or parse error.
    """
    try:
        with open(_FALLBACK_JSON, encoding="utf-8") as fh:
            items = json.load(fh)
        signals: List[ExternalSignal] = []
        for item in items:
            try:
                lat   = float(item["lat"])
                lon   = float(item["lon"])
                count = int(item["vessel_count"])
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    continue
                if count < 0:
                    continue
                signals.append(ExternalSignal(
                    signal_type = SignalType.MARITIME,
                    lat         = lat,
                    lon         = lon,
                    intensity   = _normalize_density(count),
                    metadata    = {
                        "source":       "maritime",
                        "vessel_count": count,
                    },
                ))
            except (KeyError, ValueError, TypeError):
                continue
        return signals
    except Exception:
        return []


def _builtin_signals() -> List[ExternalSignal]:
    """Return ExternalSignals from the built-in cluster constants (last-resort fallback)."""
    return [
        ExternalSignal(
            signal_type = SignalType.MARITIME,
            lat         = lat,
            lon         = lon,
            intensity   = _normalize_density(count),
            metadata    = {"source": "maritime", "vessel_count": count},
        )
        for lat, lon, count in _BUILTIN_CLUSTERS
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_maritime_signals() -> List[ExternalSignal]:
    """
    Return MARITIME ExternalSignal objects from vessel traffic data.

    Fetch priority (no-arg, single-run cached)
    ───────────────────────────────────────────
    1. Returns the module-level cache if already populated.
    2. Attempts a live fetch from the GFW Events API (requires GFW_API_KEY
       env var; 4 s timeout).
    3. Falls back to src/data/maritime_fallback.json on any failure.
    4. Falls back to built-in cluster constants if the file is unavailable.

    Returns
    ───────
    List of ExternalSignal objects with signal_type=MARITIME.
    All intensities are guaranteed to be in [0, 1].
    """
    global _SIGNAL_CACHE
    if _SIGNAL_CACHE is not None:
        return _SIGNAL_CACHE

    signals = _fetch_gfw_vessels()
    if signals is None:
        signals = _load_fallback_signals()
    if not signals:
        signals = _builtin_signals()

    _SIGNAL_CACHE = signals
    return _SIGNAL_CACHE
