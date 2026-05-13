"""
intelligence/ais_ingestion.py
──────────────────────────────
Real-time AIS vessel position ingestion via aisstream.io WebSocket feed.

Produces MARITIME ExternalSignal objects from live AIS data and can be
combined with GFW signals from maritime_ingestion.py for broader coverage.

Data model
──────────
  fetch_ais_signals()   Returns MARITIME ExternalSignals from AIS positions
                        binned into a 1° grid, with density-based intensity.

Intensity normalisation
───────────────────────
  vessel_count → intensity = min(count / _MAX_VESSELS_PER_CELL, 1.0)

  Same scale as maritime_ingestion.py so signals from both sources are
  comparable when combined.

Live fetch
──────────
  Set AIS_API_KEY to enable real-time ingestion from aisstream.io.
  The module connects to the WebSocket, subscribes to the configured
  bounding boxes, collects positions for _COLLECT_SECONDS, then closes.

  When the key is absent, the websocket-client library is missing, or the
  connection fails, the pipeline falls back to src/data/ais_fallback.json
  and then to built-in cluster constants.

Bounding boxes
──────────────
  Defaults cover seven strategic maritime chokepoints.  Override at call
  time or set custom boxes directly in _DEFAULT_BOUNDING_BOXES.
  Format: [[[min_lat, min_lon], [max_lat, max_lon]], ...]
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import websocket as _ws_lib
    _WEBSOCKET_AVAILABLE = True
except ImportError:
    _WEBSOCKET_AVAILABLE = False

from intelligence.external_features import ExternalSignal, SignalType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AIS_URL              = "wss://stream.aisstream.io/v0/stream"
_COLLECT_SECONDS      = 30             # how long to collect AIS messages per call
_SOCKET_TIMEOUT       = 2.0           # per-message recv timeout (seconds)
_GRID_RESOLUTION      = 1.0           # degrees — positions binned into 1° cells
_MAX_VESSELS_PER_CELL = 20            # count at which intensity saturates to 1.0

_FALLBACK_JSON = Path(__file__).resolve().parent.parent / "data" / "ais_fallback.json"

# Module-level cache: populated on the first successful fetch.
_SIGNAL_CACHE: Optional[List[ExternalSignal]] = None

# Default bounding boxes: seven strategic maritime chokepoints.
# Each entry is [[min_lat, min_lon], [max_lat, max_lon]].
_DEFAULT_BOUNDING_BOXES: List[List[List[float]]] = [
    [[ 22.0,  54.0], [ 30.0,  60.0]],  # Strait of Hormuz + Persian Gulf
    [[ -6.0,  95.0], [ 10.0, 110.0]],  # Strait of Malacca
    [[  0.0, 105.0], [ 25.0, 125.0]],  # South China Sea
    [[ 40.0,  27.0], [ 48.0,  42.0]],  # Black Sea
    [[ 30.0,  25.0], [ 42.0,  38.0]],  # Eastern Mediterranean
    [[ 12.0,  32.0], [ 32.0,  44.0]],  # Red Sea / Suez Canal
    [[ 53.0,  10.0], [ 66.0,  30.0]],  # Baltic Sea
]

# Built-in cluster constants — last-resort fallback matching the same
# strategic locations as maritime_ingestion._BUILTIN_CLUSTERS.
_BUILTIN_CLUSTERS: List[Tuple[float, float, int]] = [
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
    if count <= 0:
        return 0.0
    return min(float(count) / max_count, 1.0)


# ---------------------------------------------------------------------------
# Real ingestion: aisstream.io WebSocket
# ---------------------------------------------------------------------------

def _collect_positions(
    api_key: str,
    bounding_boxes: List[List[List[float]]],
    collect_seconds: int,
) -> Optional[List[Tuple[float, float]]]:
    """
    Open a WebSocket to aisstream.io, collect raw (lat, lon) pairs for
    ``collect_seconds``, then close.

    Returns None when the library is unavailable, the key is empty, or
    the connection fails.  Returns an empty list when connected but no
    position messages arrive within the window.
    """
    if not _WEBSOCKET_AVAILABLE or not api_key:
        return None

    try:
        ws = _ws_lib.WebSocket()
        ws.connect(_AIS_URL)
        ws.send(json.dumps({
            "APIKey":        api_key,
            "BoundingBoxes": bounding_boxes,
        }))
        ws.settimeout(_SOCKET_TIMEOUT)

        positions: List[Tuple[float, float]] = []
        deadline = time.time() + collect_seconds

        while time.time() < deadline:
            try:
                raw = ws.recv()
                msg = json.loads(raw)
                meta = msg.get("MetaData", {})
                lat  = meta.get("latitude")
                lon  = meta.get("longitude")
                if lat is None or lon is None:
                    continue
                lat, lon = float(lat), float(lon)
                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    positions.append((lat, lon))
            except _ws_lib.WebSocketTimeoutException:
                # No message in this tick — keep waiting until deadline.
                continue
            except Exception:
                break

        ws.close()
        return positions

    except Exception:
        return None


def _bin_to_signals(positions: List[Tuple[float, float]]) -> List[ExternalSignal]:
    """
    Bin (lat, lon) pairs into a 1° grid and produce one ExternalSignal per
    non-empty cell with intensity proportional to vessel density.
    """
    counts: Dict[Tuple[float, float], int] = {}
    for lat, lon in positions:
        cell = (
            round(lat / _GRID_RESOLUTION) * _GRID_RESOLUTION,
            round(lon / _GRID_RESOLUTION) * _GRID_RESOLUTION,
        )
        counts[cell] = counts.get(cell, 0) + 1

    return [
        ExternalSignal(
            signal_type = SignalType.MARITIME,
            lat         = cell_lat,
            lon         = cell_lon,
            intensity   = _normalize_density(count),
            metadata    = {"source": "ais", "vessel_count": count},
        )
        for (cell_lat, cell_lon), count in counts.items()
    ]


def _fetch_ais_live(
    bounding_boxes: List[List[List[float]]],
    collect_seconds: int,
) -> Optional[List[ExternalSignal]]:
    """Attempt a live aisstream.io fetch; return None on any failure."""
    api_key = os.environ.get("AIS_API_KEY", "")
    positions = _collect_positions(api_key, bounding_boxes, collect_seconds)
    if positions is None:
        return None
    return _bin_to_signals(positions)


# ---------------------------------------------------------------------------
# Fallback loaders
# ---------------------------------------------------------------------------

def _load_fallback_signals() -> List[ExternalSignal]:
    """
    Load MARITIME ExternalSignal objects from the repo-local AIS fallback JSON.

    Expected item schema: {"lat": float, "lon": float, "vessel_count": int}
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
                    metadata    = {"source": "ais", "vessel_count": count},
                ))
            except (KeyError, ValueError, TypeError):
                continue
        return signals
    except Exception:
        return []


def _builtin_signals() -> List[ExternalSignal]:
    """Return ExternalSignals from built-in cluster constants (last-resort fallback)."""
    return [
        ExternalSignal(
            signal_type = SignalType.MARITIME,
            lat         = lat,
            lon         = lon,
            intensity   = _normalize_density(count),
            metadata    = {"source": "ais", "vessel_count": count},
        )
        for lat, lon, count in _BUILTIN_CLUSTERS
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_ais_signals(
    bounding_boxes: Optional[List[List[List[float]]]] = None,
    collect_seconds: int = _COLLECT_SECONDS,
) -> List[ExternalSignal]:
    """
    Return MARITIME ExternalSignal objects from live AIS vessel positions.

    Fetch priority
    ──────────────
    1. Returns the module-level cache if already populated.
    2. Attempts a live fetch from aisstream.io (requires AIS_API_KEY env var).
    3. Falls back to src/data/ais_fallback.json on any failure.
    4. Falls back to built-in cluster constants if the file is unavailable.

    Parameters
    ──────────
    bounding_boxes   Override the default strategic chokepoint boxes.
                     Format: [[[min_lat, min_lon], [max_lat, max_lon]], ...]
    collect_seconds  How long to collect AIS messages per call (default: 30).

    Returns
    ───────
    List of ExternalSignal objects with signal_type=MARITIME.
    All intensities are guaranteed to be in [0, 1].
    """
    global _SIGNAL_CACHE
    if _SIGNAL_CACHE is not None:
        return _SIGNAL_CACHE

    boxes = bounding_boxes if bounding_boxes is not None else _DEFAULT_BOUNDING_BOXES

    signals = _fetch_ais_live(boxes, collect_seconds)
    if signals is None:
        signals = _load_fallback_signals()
    if not signals:
        signals = _builtin_signals()

    _SIGNAL_CACHE = signals
    return _SIGNAL_CACHE
