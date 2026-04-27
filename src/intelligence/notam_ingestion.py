"""
intelligence/notam_ingestion.py
────────────────────────────────
NOTAM (Notice to Air Missions) ingestion pipeline.

Produces ExternalSignal objects from NOTAM data so they can be injected into
the feature extraction pipeline via build_features(external_signals=...).

Current state:  mock data only — no live scraping.
Future hooks:   replace _MOCK_NOTAMS with a real fetch, or swap
                expand_notam_to_signals() for polygon / radius tiling once
                a geometry library is available.

Data model
──────────
  Notam                 Raw NOTAM record (lat/lon centre, radius, severity)
  NotamSeverity         Four-level enum → deterministic intensity mapping
  expand_notam_to_signals(notam)   Converts one Notam → List[ExternalSignal]
  fetch_notam_signals()            Returns all active signals from _MOCK_NOTAMS

Severity → intensity mapping
────────────────────────────
  ADVISORY    0.25   Informational; no immediate hazard
  WARNING     0.50   Caution required; potential hazard
  RESTRICTED  0.75   Access limited; military / exercise area
  PROHIBITED  1.00   Hard no-fly; live operations / conflict zone
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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
    """

    notam_id:       str
    lat:            float
    lon:            float
    radius_km:      float
    severity:       NotamSeverity
    description:    str                = ""
    effective_from: Optional[str]      = None
    effective_to:   Optional[str]      = None


# ---------------------------------------------------------------------------
# Mock data  (replace with live fetch when ready)
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
# Public API
# ---------------------------------------------------------------------------

def expand_notam_to_signals(notam: Notam) -> List[ExternalSignal]:
    """
    Convert a single Notam record into one or more ExternalSignal objects.

    Current behaviour (centre-point expansion)
    ────────────────────────────────────────────
    Returns exactly one ExternalSignal positioned at the NOTAM centre.
    The ``radius_km`` is preserved in the signal metadata so that consumers
    can apply their own spatial logic.

    Future extension
    ────────────────
    Replace or extend this function to:
      • tile a grid of signals covering the full radius
      • trace a polygon boundary for irregular shapes
      • split large areas into multiple overlapping signals
    The ``List[ExternalSignal]`` return type already supports all of these
    without changing any call sites.

    Parameters
    ──────────
    notam   A Notam record to convert.

    Returns
    ───────
    List of ExternalSignal objects (currently length 1).
    """
    intensity = _SEVERITY_TO_INTENSITY[notam.severity]
    return [
        ExternalSignal(
            signal_type = SignalType.NO_FLY,
            lat         = notam.lat,
            lon         = notam.lon,
            intensity   = intensity,
            metadata    = {
                "source":    "NOTAM",
                "id":        notam.notam_id,
                "radius_km": notam.radius_km,
                "severity":  notam.severity.value,
            },
        )
    ]


def fetch_notam_signals(
    notams: Optional[List[Notam]] = None,
) -> List[ExternalSignal]:
    """
    Return ExternalSignal objects derived from the active NOTAM dataset.

    Parameters
    ──────────
    notams  Optional override list of Notam records.  Defaults to the
            built-in mock dataset (_MOCK_NOTAMS) when None.  Pass a custom
            list in tests or when a real fetch replaces the mock.

    Returns
    ───────
    Flat list of ExternalSignal objects, one per NOTAM (currently).
    Order follows the input list; duplicates are not de-duplicated.
    """
    source = notams if notams is not None else _MOCK_NOTAMS
    signals: List[ExternalSignal] = []
    for notam in source:
        signals.extend(expand_notam_to_signals(notam))
    return signals
