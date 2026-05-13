"""
intelligence/notam_ingestion.py
────────────────────────────────
NOTAM (Notice to Air Missions) ingestion pipeline.

Produces ExternalSignal objects from NOTAM data so they can be injected into
the feature extraction pipeline via build_features(external_signals=...).

Data model
──────────
  Notam                 Raw NOTAM record (lat/lon centre, radius, severity)
  NotamSeverity         Four-level enum → deterministic intensity mapping
  expand_notam_to_signals(notam)   Converts one Notam → List[ExternalSignal]
  fetch_notam_signals()            Returns all active signals

Severity → intensity mapping
────────────────────────────
  ADVISORY    0.25   Informational; no immediate hazard
  WARNING     0.50   Caution required; potential hazard
  RESTRICTED  0.75   Access limited; military / exercise area
  PROHIBITED  1.00   Hard no-fly; live operations / conflict zone

Live fetch
──────────
  Set FAA_CLIENT_ID and FAA_CLIENT_SECRET environment variables to enable
  real-time ingestion from the FAA NOTAM API (OAuth2 client credentials).
  When credentials are absent or the request fails the pipeline falls back
  to src/data/notams_fallback.json, and then to _MOCK_NOTAMS if the file
  is unavailable.
"""

from __future__ import annotations

import json
import math
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from intelligence.external_features import ExternalSignal, SignalType


# ---------------------------------------------------------------------------
# Severity model
# ---------------------------------------------------------------------------

class NotamSeverity(str, Enum):
    """ICAO-aligned severity levels for a NOTAM."""
    ADVISORY   = "ADVISORY"    # informational — flight planning awareness
    WARNING    = "WARNING"     # caution — possible hazard
    RESTRICTED = "RESTRICTED"  # limited access — military / exercise zone
    PROHIBITED = "PROHIBITED"  # hard exclusion — live operations


# Deterministic mapping from severity to ExternalSignal intensity.
# Values chosen so each level is clearly separated and the full [0,1]
# range is utilised without saturation at the low end.
_SEVERITY_TO_INTENSITY: Dict[NotamSeverity, float] = {
    NotamSeverity.ADVISORY:   0.25,
    NotamSeverity.WARNING:    0.50,
    NotamSeverity.RESTRICTED: 0.75,
    NotamSeverity.PROHIBITED: 1.00,
}


# ---------------------------------------------------------------------------
# Notam dataclass
# ---------------------------------------------------------------------------

@dataclass
class Notam:
    """
    A single NOTAM record.

    Fields
    ──────
    notam_id        Unique identifier (e.g. "A1234/24").
    lat             WGS-84 latitude of the affected area centre.
    lon             WGS-84 longitude of the affected area centre.
    radius_km       Nominal radius of the affected area in kilometres.
                    Stored in signal metadata for downstream consumers;
                    not yet used for spatial tiling.
    severity        How operationally significant the restriction is.
    description     Human-readable summary of the NOTAM content.
    effective_from  ISO-8601 start of validity window (None = already active).
    effective_to    ISO-8601 end of validity window (None = indefinite).
    source          Origin tag written into ExternalSignal metadata:
                    "NOTAM" for static/fallback data,
                    "real_notam" for live FAA API data.
    """

    notam_id:       str
    lat:            float
    lon:            float
    radius_km:      float
    severity:       NotamSeverity
    description:    str                = ""
    effective_from: Optional[str]      = None
    effective_to:   Optional[str]      = None
    source:         str                = "NOTAM"


# ---------------------------------------------------------------------------
# Static fallback data  (used when live API is unavailable)
# ---------------------------------------------------------------------------

_MOCK_NOTAMS: List[Notam] = [
    # Eastern Mediterranean — large-scale military exercise restriction
    Notam(
        notam_id    = "A0001/24",
        lat         = 34.5,
        lon         = 33.5,
        radius_km   = 50.0,
        severity    = NotamSeverity.RESTRICTED,
        description = "Military exercise area — restricted FL050–FL450",
        effective_from = "2024-01-15T06:00Z",
        effective_to   = "2024-01-30T18:00Z",
    ),
    # Strait of Hormuz — active naval operations, hard prohibition
    Notam(
        notam_id    = "A0002/24",
        lat         = 26.8,
        lon         = 56.2,
        radius_km   = 30.0,
        severity    = NotamSeverity.PROHIBITED,
        description = "Prohibited zone — naval live-fire operations",
        effective_from = "2024-01-10T00:00Z",
        effective_to   = "2024-02-10T00:00Z",
    ),
    # Black Sea — active conflict hazard warning
    Notam(
        notam_id    = "A0003/24",
        lat         = 44.5,
        lon         = 33.5,
        radius_km   = 80.0,
        severity    = NotamSeverity.WARNING,
        description = "Warning area — active conflict zone, hazard to aviation",
    ),
    # Baltic Sea — temporary NATO exercise advisory
    Notam(
        notam_id    = "A0004/24",
        lat         = 57.5,
        lon         = 22.0,
        radius_km   = 25.0,
        severity    = NotamSeverity.ADVISORY,
        description = "Temporary restriction — NATO Baltic exercise BALTOPS",
        effective_from = "2024-01-20T08:00Z",
        effective_to   = "2024-01-25T18:00Z",
    ),
    # South China Sea — live-fire exercise, hard prohibition
    Notam(
        notam_id    = "A0005/24",
        lat         = 15.8,
        lon         = 112.5,
        radius_km   = 60.0,
        severity    = NotamSeverity.PROHIBITED,
        description = "Prohibited zone — PLA live-fire exercise",
        effective_from = "2024-01-18T00:00Z",
        effective_to   = "2024-01-22T00:00Z",
    ),
]


# ---------------------------------------------------------------------------
# Real ingestion: FAA NOTAM API
# ---------------------------------------------------------------------------

_FAA_TOKEN_URL = "https://external-api.faa.gov/notamapi/v1/token"
_FAA_NOTAM_URL = "https://external-api.faa.gov/notamapi/v1/notams"
_FAA_TIMEOUT   = 4   # seconds per request

_FALLBACK_JSON = Path(__file__).resolve().parent.parent / "data" / "notams_fallback.json"

# Module-level cache: populated on the first no-arg call to fetch_notam_signals().
_SIGNAL_CACHE: Optional[List[ExternalSignal]] = None

# Ordered (keyword, severity) pairs for deriving severity from NOTAM free text.
_TEXT_SEVERITY = [
    ("PROHIBITED", NotamSeverity.PROHIBITED),
    ("RESTRICTED", NotamSeverity.RESTRICTED),
    ("WARNING",    NotamSeverity.WARNING),
]


def _text_to_severity(text: str) -> NotamSeverity:
    """Derive NotamSeverity from NOTAM free text; defaults to ADVISORY."""
    upper = text.upper()
    for keyword, severity in _TEXT_SEVERITY:
        if keyword in upper:
            return severity
    return NotamSeverity.ADVISORY


def _parse_faa_response(data: dict) -> List[Notam]:
    """
    Parse a FAA NOTAM API geoJSON response dict into Notam objects.

    Expects the FAA v1 geoJSON envelope::

        {
          "items": [
            {
              "geometry": {"type": "Point", "coordinates": [lon, lat]},
              "properties": {
                "coreNOTAMData": {
                  "notam": {
                    "id": "...", "text": "...",
                    "effectiveStart": "...", "effectiveEnd": "..."
                  }
                }
              }
            }, ...
          ]
        }

    Items with missing, null, or out-of-range coordinates are silently skipped.
    Parsed Notam objects carry source="real_notam".
    """
    notams: List[Notam] = []
    for item in data.get("items", []):
        try:
            geom   = item.get("geometry") or {}
            coords = geom.get("coordinates")
            if not coords or len(coords) < 2:
                continue
            lon, lat = float(coords[0]), float(coords[1])
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue

            props      = item.get("properties", {})
            core       = props.get("coreNOTAMData", {})
            notam_data = core.get("notam", {})

            notam_id  = str(notam_data.get("id") or notam_data.get("number") or "")
            text      = str(notam_data.get("text") or "")
            eff_start = notam_data.get("effectiveStart")
            eff_end   = notam_data.get("effectiveEnd")

            notams.append(Notam(
                notam_id       = notam_id,
                lat            = lat,
                lon            = lon,
                radius_km      = 30.0,
                severity       = _text_to_severity(text),
                description    = text[:200],
                effective_from = eff_start,
                effective_to   = eff_end,
                source         = "real_notam",
            ))
        except (TypeError, ValueError, KeyError):
            continue
    return notams


def _fetch_faa_notams() -> Optional[List[Notam]]:
    """
    Fetch NOTAMs from the FAA NOTAM API using OAuth2 client credentials.

    Reads FAA_CLIENT_ID and FAA_CLIENT_SECRET from the environment.
    Returns None immediately when credentials are absent or on any error.
    Timeout per request: _FAA_TIMEOUT seconds.
    """
    client_id     = os.environ.get("FAA_CLIENT_ID", "")
    client_secret = os.environ.get("FAA_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        return None

    try:
        # Step 1: obtain bearer token via client credentials grant
        token_body = urllib.parse.urlencode({
            "client_id":     client_id,
            "client_secret": client_secret,
            "grant_type":    "client_credentials",
        }).encode()
        token_req = urllib.request.Request(
            _FAA_TOKEN_URL,
            data    = token_body,
            headers = {"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(token_req, timeout=_FAA_TIMEOUT) as resp:
            token_json = json.loads(resp.read().decode())
        token = token_json.get("access_token", "")
        if not token:
            return None

        # Step 2: fetch NOTAMs (geoJSON, page-1, 200 items)
        params    = urllib.parse.urlencode({"responseFormat": "geoJson", "pageSize": "200"})
        notam_req = urllib.request.Request(
            f"{_FAA_NOTAM_URL}?{params}",
            headers = {"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(notam_req, timeout=_FAA_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())

        return _parse_faa_response(data)

    except Exception:
        return None


def _load_fallback_notams() -> List[Notam]:
    """
    Load Notam records from the repo-local static JSON fallback file.

    Items with missing or out-of-range lat/lon are skipped.
    Returns an empty list on any I/O or parse error.
    Loaded Notam objects carry the default source="NOTAM".
    """
    try:
        with open(_FALLBACK_JSON, encoding="utf-8") as fh:
            items = json.load(fh)
        notams: List[Notam] = []
        for item in items:
            try:
                lat = float(item["lat"])
                lon = float(item["lon"])
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    continue
                notams.append(Notam(
                    notam_id       = str(item.get("notam_id", "")),
                    lat            = lat,
                    lon            = lon,
                    radius_km      = float(item.get("radius_km", 30.0)),
                    severity       = NotamSeverity(item["severity"]),
                    description    = str(item.get("description", "")),
                    effective_from = item.get("effective_from"),
                    effective_to   = item.get("effective_to"),
                ))
            except (KeyError, ValueError, TypeError):
                continue
        return notams
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Spatial expansion constants
# ---------------------------------------------------------------------------

_KM_PER_DEG   = 111.0   # approximate km per degree of latitude (and longitude at equator)
_GRID_STEP_KM = 55.0    # grid point spacing — ~0.5° resolution, balances coverage vs. count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_notam_to_signals(notam: Notam) -> List[ExternalSignal]:
    """
    Convert a single Notam record into one or more ExternalSignal objects.

    Tiles a uniform grid over the NOTAM's circular area so that the full
    ``radius_km`` coverage is represented by multiple signals.  The existing
    spatial-weighting logic in ``_signals_near_weighted`` then applies linear
    distance attenuation from each grid point, giving realistic coverage.

    Grid spacing is ~55 km (~0.5°).  Points outside ``radius_km`` are excluded.
    For small NOTAMs (radius < one grid step, ~55 km) only the centre point is
    emitted, preserving the original single-signal behaviour.

    Parameters
    ──────────
    notam   A Notam record to convert.

    Returns
    ───────
    List of ExternalSignal objects; at least one (the centre point).
    All signals share the same intensity derived from ``notam.severity``.
    """
    intensity  = _SEVERITY_TO_INTENSITY[notam.severity]
    step_deg   = _GRID_STEP_KM / _KM_PER_DEG
    radius_deg = notam.radius_km / _KM_PER_DEG
    lat_steps  = math.ceil(radius_deg / step_deg)
    lon_steps  = math.ceil(radius_deg / step_deg)

    metadata = {
        "source":    notam.source,
        "id":        notam.notam_id,
        "radius_km": notam.radius_km,
        "severity":  notam.severity.value,
    }

    signals: List[ExternalSignal] = []
    for di in range(-lat_steps, lat_steps + 1):
        for dj in range(-lon_steps, lon_steps + 1):
            dist_km = math.sqrt(
                (di * step_deg * _KM_PER_DEG) ** 2 +
                (dj * step_deg * _KM_PER_DEG) ** 2
            )
            if dist_km <= notam.radius_km:
                signals.append(ExternalSignal(
                    signal_type = SignalType.NO_FLY,
                    lat         = notam.lat + di * step_deg,
                    lon         = notam.lon + dj * step_deg,
                    intensity   = intensity,
                    metadata    = metadata,
                ))

    # Always emit at least the centre point (handles radius=0 edge case).
    if not signals:
        signals.append(ExternalSignal(
            signal_type = SignalType.NO_FLY,
            lat         = notam.lat,
            lon         = notam.lon,
            intensity   = intensity,
            metadata    = metadata,
        ))

    return signals


def fetch_notam_signals(
    notams: Optional[List[Notam]] = None,
) -> List[ExternalSignal]:
    """
    Return ExternalSignal objects derived from the active NOTAM dataset.

    Parameters
    ──────────
    notams  Optional override list of Notam records.  When provided the
            list is expanded directly — bypasses the cache and any live
            fetch.  Pass a custom list in tests or when supplying your
            own NOTAM data.

    Default path (notams=None)
    ──────────────────────────
    1. Returns the module-level cache if already populated.
    2. Attempts a live fetch from the FAA NOTAM API (requires
       FAA_CLIENT_ID and FAA_CLIENT_SECRET env vars; 4 s timeout each).
    3. Falls back to src/data/notams_fallback.json on any failure.
    4. Falls back to _MOCK_NOTAMS if the JSON file is unavailable.

    The result is cached so that repeated no-arg calls within the same
    process incur only one network round-trip.

    Returns
    ───────
    Flat list of ExternalSignal objects.
    Order follows the source list; duplicates are not de-duplicated.
    """
    global _SIGNAL_CACHE

    if notams is not None:
        signals: List[ExternalSignal] = []
        for notam in notams:
            signals.extend(expand_notam_to_signals(notam))
        return signals

    if _SIGNAL_CACHE is not None:
        return _SIGNAL_CACHE

    source_notams = _fetch_faa_notams()
    if source_notams is None:
        source_notams = _load_fallback_notams()
    if not source_notams:
        source_notams = list(_MOCK_NOTAMS)

    result: List[ExternalSignal] = []
    for notam in source_notams:
        result.extend(expand_notam_to_signals(notam))

    _SIGNAL_CACHE = result
    return _SIGNAL_CACHE
