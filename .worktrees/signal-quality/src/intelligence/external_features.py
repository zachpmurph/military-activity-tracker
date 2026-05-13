"""
intelligence/external_features.py
──────────────────────────────────
Infrastructure for injecting external data-source signals into the feature
extraction pipeline.  This is the infrastructure layer only — no new
calibration weights or profile entries are introduced here.

Signal lifecycle
────────────────
1. Construct ExternalSignal objects from raw source data.
2. Pass them to build_features(external_signals=...) in classifier.py.
3. build_features calls map_external_signals_to_features() for each grid cell,
   then merge_external_features() to apply the result in-place to RegionFeatures.

Supported signal types
──────────────────────
  NO_FLY     Restricted-airspace notification near a grid cell.
             → spike_flag, change_score_norm

  MARITIME   Naval / maritime coordination activity near a grid cell.
             → coordination_flag, inflow_norm

  SATELLITE  Overhead-imagery observation of activity at a grid cell.
             → change_score_norm, novelty (via new_aircraft proxy)

All output feature values are bounded to [0, 1] before merging back to raw
RegionFeatures fields.  External signals can only ADD evidence — they never
reduce values that were already set by the detection pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:                                    # pragma: no cover
    from intelligence.classifier import RegionFeatures


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Spatial proximity window for associating a signal to a grid cell.
# 1.5° ≈ 165 km; grid cells are rounded to 0.1° in build_features.
# Influence decays linearly from 1.0 at the centre to 0.0 at the edge
# (see _signals_near_weighted), so enlarging the radius allows gradual
# influence rather than all-or-nothing coverage.
_PROXIMITY_RADIUS: float = 1.5

# Conversion factors: normalised feature-vector value → raw RegionFeatures
# field units.  These invert the normalisation applied in
# calibration._feature_vector_map so that the round-trip
#   raw → normalised (merge) → normalised (scoring)
# is bounded and idempotent.
_CHANGE_SCORE_MAX: float = 40.0   # mirrors calibration.py default max_change
_INFLOW_MAX: int         = 10     # practical cap for external inflow proxy


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class SignalType(str, Enum):
    """Category of an external data-source signal."""
    NO_FLY    = "NO_FLY"
    MARITIME  = "MARITIME"
    SATELLITE = "SATELLITE"


@dataclass
class ExternalSignal:
    """
    A single observation from an external data source.

    Fields
    ──────
    signal_type  Category of the signal (see SignalType).
    lat          WGS-84 latitude of the signal origin.
    lon          WGS-84 longitude of the signal origin.
    intensity    Normalised signal strength in [0, 1].  Values outside this
                 range are silently clamped on construction.
    metadata     Arbitrary source-specific key/value context
                 (e.g. ``{"source": "notam", "id": "A1234/24"}``).
    """

    signal_type: SignalType
    lat:         float
    lon:         float
    intensity:   float
    metadata:    Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.intensity = _clamp(self.intensity)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Return *value* clamped to [*lo*, *hi*]."""
    return max(lo, min(hi, float(value)))


def _signals_near(
    signals:    List[ExternalSignal],
    region_lat: float,
    region_lon: float,
    radius:     float = _PROXIMITY_RADIUS,
) -> List[ExternalSignal]:
    """Return the subset of *signals* within *radius* degrees of the grid cell."""
    return [
        s for s in signals
        if abs(s.lat - region_lat) <= radius
        and abs(s.lon - region_lon) <= radius
    ]


def _signals_near_weighted(
    signals:    List[ExternalSignal],
    region_lat: float,
    region_lon: float,
    radius:     float = _PROXIMITY_RADIUS,
) -> List[Tuple[ExternalSignal, float]]:
    """
    Return ``(signal, weight)`` pairs for every signal within *radius* degrees.

    Distance metric
    ───────────────
    Chebyshev distance (max of absolute lat/lon deltas), consistent with the
    original binary _signals_near logic.

    Weight function
    ───────────────
    Linear decay from 1.0 at the grid-cell centre to 0.0 at the boundary::

        weight = 1.0 − (distance / radius)

    A signal exactly on top of the region contributes with full intensity;
    one right at the boundary contributes almost nothing (weight ≈ 0).
    Signals outside the radius are excluded entirely.

    Usage
    ─────
    Callers compute *effective intensity* as ``signal.intensity × weight``
    before feeding values into feature mappings.
    """
    result: List[Tuple[ExternalSignal, float]] = []
    for s in signals:
        lat_diff = abs(s.lat - region_lat)
        lon_diff = abs(s.lon - region_lon)
        distance = max(lat_diff, lon_diff)
        if distance <= radius:
            weight = 1.0 - (distance / radius)
            result.append((s, weight))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_external_signals_to_features(
    signals: List[ExternalSignal],
    region:  "RegionFeatures",
) -> Dict[str, float]:
    """
    Map external signals near *region* to a normalised feature dict.

    The returned dict uses the same feature names as
    ``calibration._feature_vector_map`` (e.g. ``"spike_flag"``,
    ``"change_score_norm"``).  All values are guaranteed to be in [0, 1].
    Returns an empty dict when no signals fall within the proximity radius.

    Aggregation strategy
    ────────────────────
    *Per signal type*: the maximum effective_intensity of nearby signals of
    that type, where effective_intensity = signal.intensity × distance_weight.
    *Per output feature*: the maximum contribution across all signal types.
    This means the most salient nearby signal always wins; no averaging hides
    a high-effective-intensity reading behind weaker neighbours.

    Mapping table  (iv = max effective_intensity for that type)
    ─────────────
    NO_FLY    → spike_flag = iv
                change_score_norm = iv × 0.80
    MARITIME  → coordination_flag = iv
                inflow_norm = iv × 0.80
    SATELLITE → change_score_norm = iv × 0.70
                novelty = iv × 0.50

    Parameters
    ──────────
    signals  List of ExternalSignal objects (may be empty).
    region   The RegionFeatures whose lat/lon identifies the target grid cell.

    Returns
    ───────
    ``Dict[str, float]`` — feature name → value in [0, 1].
    """
    nearby_w = _signals_near_weighted(signals, region.lat, region.lon)
    if not nearby_w:
        return {}

    # Reduce to max effective_intensity per signal type.
    # effective_intensity = raw_intensity × distance_weight, so signals at the
    # grid-cell centre contribute fully while those near the boundary barely
    # register.  All values are clamped to [0, 1] before use.
    by_type: Dict[str, float] = {}
    for s, weight in nearby_w:
        eff = _clamp(s.intensity * weight)
        by_type[s.signal_type.value] = max(by_type.get(s.signal_type.value, 0.0), eff)

    out: Dict[str, float] = {}

    # NO_FLY: restricted-airspace activity → spike + elevated change evidence
    if SignalType.NO_FLY.value in by_type:
        iv = by_type[SignalType.NO_FLY.value]
        out["spike_flag"]        = max(out.get("spike_flag",        0.0), _clamp(iv))
        out["change_score_norm"] = max(out.get("change_score_norm", 0.0), _clamp(iv * 0.80))

    # MARITIME: naval coordination → coordination flag + inflow evidence
    if SignalType.MARITIME.value in by_type:
        iv = by_type[SignalType.MARITIME.value]
        out["coordination_flag"] = max(out.get("coordination_flag", 0.0), _clamp(iv))
        out["inflow_norm"]       = max(out.get("inflow_norm",       0.0), _clamp(iv * 0.80))

    # SATELLITE: overhead observation → change evidence + novelty
    if SignalType.SATELLITE.value in by_type:
        iv = by_type[SignalType.SATELLITE.value]
        out["change_score_norm"] = max(out.get("change_score_norm", 0.0), _clamp(iv * 0.70))
        out["novelty"]           = max(out.get("novelty",           0.0), _clamp(iv * 0.50))

    return out


def merge_external_features(
    region:  "RegionFeatures",
    ext_map: Dict[str, float],
) -> None:
    """
    Apply an external feature dict to *region* in-place.

    Converts each normalised feature-vector value back to raw RegionFeatures
    field units using the inverse of the normalisation in
    ``calibration._feature_vector_map``::

        spike_flag        → set True when value > 0.5  (never cleared)
        coordination_flag → set True when value > 0.5  (never cleared)
        change_score      → max(existing, value × _CHANGE_SCORE_MAX)
        inflow_count      → max(existing, int(value × _INFLOW_MAX))
        novelty (proxy)   → new_aircraft raised so new_entry_ratio ≈ ext value
                            (capped at aircraft_count; skipped when count == 0)

    Invariants
    ──────────
    * External signals can only ADD evidence — no existing field is ever
      reduced.
    * ``change_score`` is bounded by _CHANGE_SCORE_MAX × 1.0 = 40.0, so
      ``change_score_norm`` stays ≤ 1.0 during scoring.
    * ``inflow_count`` proxy is bounded by _INFLOW_MAX = 10.

    Parameters
    ──────────
    region   RegionFeatures to update in-place.
    ext_map  Output of map_external_signals_to_features(); may be empty.
    """
    if not ext_map:
        return

    if ext_map.get("spike_flag", 0.0) > 0.5:
        region.spike_flag = True

    if ext_map.get("coordination_flag", 0.0) > 0.5:
        region.coordination_flag = True

    if "change_score_norm" in ext_map:
        region.change_score = max(
            region.change_score,
            _clamp(ext_map["change_score_norm"]) * _CHANGE_SCORE_MAX,
        )

    if "inflow_norm" in ext_map:
        region.inflow_count = max(
            region.inflow_count,
            int(_clamp(ext_map["inflow_norm"]) * _INFLOW_MAX),
        )

    # Novelty proxy: raise new_aircraft so new_entry_ratio tracks ext value.
    # new_entry_ratio = new_aircraft / aircraft_count feeds into the novelty
    # property; capped at aircraft_count to keep the ratio ≤ 1.
    if "novelty" in ext_map and region.aircraft_count > 0:
        proxy = int(_clamp(ext_map["novelty"]) * region.aircraft_count)
        region.new_aircraft = max(region.new_aircraft, proxy)
