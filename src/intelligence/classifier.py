"""
intelligence/classifier.py
──────────────────────────
Feature-based scoring and classification layer.

Consumes the outputs of the detection layer unchanged and produces
RegionIntelligence objects — one per geographic region — each containing:

  • a normalised feature vector   (RegionFeatures)
  • weighted scores per class     (all_scores dict)
  • a winner classification       (Classification enum)
  • a 0–100 confidence score
  • a human-readable explanation  (str)

No detection logic is modified.  All inputs are plain Python lists/tuples
matching the return types already emitted by the detection functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from intelligence.external_features import (
    ExternalSignal,
    map_external_signals_to_features,
    merge_external_features,
)


# ---------------------------------------------------------------------------
# 1. CLASSIFICATION ENUM
# ---------------------------------------------------------------------------

class Classification(str, Enum):
    STAGING              = "STAGING"
    PROJECTION           = "PROJECTION"
    ROUTINE              = "ROUTINE"
    ANOMALY              = "ANOMALY"
    COORDINATED_ACTIVITY = "COORDINATED_ACTIVITY"


# ---------------------------------------------------------------------------
# 2. FEATURE SCHEMA
# ---------------------------------------------------------------------------

@dataclass
class RegionFeatures:
    """
    Normalised feature set for one geographic region.

    Raw counts are stored alongside derived 0–1 floats so callers can
    inspect the underlying signal without re-deriving it.

    Feature index
    ─────────────
    military_ratio      – fraction of aircraft that are military             [0,1]
    new_entry_ratio     – fraction that were absent in the 2–6 h baseline   [0,1]
    persistence_score   – fraction of 30-min buckets (of 12) with ≥3 ac    [0,1]
    flow_balance        – (outflow − inflow) / (outflow + inflow)           [−1,+1]
    aircraft_density    – log-normalised volume (100 ac → 1.0)              [0,1]
    novelty             – composite: new_entry_ratio + spike + low persist  [0,1]
    """

    lat: float
    lon: float

    # --- Volume ---
    aircraft_count:       int   = 0
    military_count:       int   = 0

    # --- Temporal ---
    recurring_appearances: int  = 0   # 30-min buckets with ≥3 aircraft (max 12)
    persistence_level:    str   = "SHORT"   # SHORT | MEDIUM | LONG

    # --- Event flags ---
    spike_flag:           bool  = False
    coordination_flag:    bool  = False
    staging_flag:         bool  = False     # confirmed track departure origin
    projection_flag:      bool  = False     # confirmed track arrival destination

    # --- Movement flows ---
    outflow_count:        int   = 0     # departure routes leaving this region
    inflow_count:         int   = 0     # arrival routes entering this region
    outflow_military:     int   = 0
    inflow_military:      int   = 0

    # --- New entries ---
    new_aircraft:         int   = 0     # not seen in 2–6 h baseline
    new_military:         int   = 0

    # --- Activity change signal ---
    change_score:         float = 0.0
    change_level:         str   = ""    # HIGH | MEDIUM | LOW

    # ------------------------------------------------------------------
    # Derived / normalised features  (always 0–1 except flow_balance)
    # ------------------------------------------------------------------

    @property
    def military_ratio(self) -> float:
        return min(self.military_count / max(self.aircraft_count, 1), 1.0)

    @property
    def new_entry_ratio(self) -> float:
        return min(self.new_aircraft / max(self.aircraft_count, 1), 1.0)

    @property
    def persistence_score(self) -> float:
        return min(self.recurring_appearances / 12.0, 1.0)

    @property
    def flow_balance(self) -> float:
        """
        +1 = pure outflow (staging signal)
        −1 = pure inflow  (projection signal)
         0 = balanced or no movement data
        """
        total = self.outflow_count + self.inflow_count
        if total == 0:
            return 0.0
        return (self.outflow_count - self.inflow_count) / total

    @property
    def aircraft_density(self) -> float:
        """Log-normalised volume.  100 aircraft → 1.0."""
        return min(math.log1p(self.aircraft_count) / math.log1p(100), 1.0)

    @property
    def novelty(self) -> float:
        """
        Composite unexpectedness signal.  Combines new-entry ratio, spike
        presence, and inverse persistence (how surprising this activity is).
        """
        spike_contrib = 0.3 if self.spike_flag else 0.0
        return min(
            self.new_entry_ratio * 0.5
            + spike_contrib
            + (1.0 - self.persistence_score) * 0.2,
            1.0,
        )


# ---------------------------------------------------------------------------
# 3. CLASSIFICATION PROFILES  (weight vectors)
# ---------------------------------------------------------------------------
#
# Each dict maps a feature-name → importance weight.
# Weights within a profile sum to 1.0 so the dot product is already in [0,1];
# multiplying by 100 gives a 0–100 class score.
#
# Feature names beginning with special prefixes are computed inline:
#   neg_*       → max(−feature, 0)      (inverted directional features)
#   low_*       → 1 − feature            (inverted magnitude features)
#   no_*        → float(not feature)     (inverted boolean features)
#   *_norm      → feature / context max  (cross-region normalisation)
# ---------------------------------------------------------------------------

_PROFILES: Dict[Classification, Dict[str, float]] = {
    # STAGING: net outflow + confirmed origin + military composition.
    # pos_flow_balance raised to 0.30 — directional bias is the primary
    # discriminator vs COORDINATED_ACTIVITY.  staging_flag raised to 0.30 as
    # the definitive structural signal.  persistence_score cut to 0.05 (a
    # staging base that happens to be long-running should not inflate ROUTINE).
    Classification.STAGING: {
        "military_ratio":    0.25,   # was 0.30 (−0.05)
        "staging_flag":      0.30,   # was 0.25 (+0.05)
        "pos_flow_balance":  0.30,   # was 0.20 (+0.10)  ← key fix
        "persistence_score": 0.05,   # was 0.10 (−0.05)
        "coordination_flag": 0.10,   # was 0.15 (−0.05)
    },
    # PROJECTION: confirmed destination must dominate.  projection_flag raised
    # to 0.40 — track-level evidence outweighs composition signals.  Cuts to
    # new_entry_ratio and military_ratio prevent ANOMALY from winning when new
    # military aircraft arrive at a known destination.
    Classification.PROJECTION: {
        "projection_flag":   0.40,   # was 0.30 (+0.10)  ← key fix
        "inflow_norm":       0.25,   # unchanged
        "new_entry_ratio":   0.15,   # was 0.20 (−0.05)
        "military_ratio":    0.10,   # was 0.15 (−0.05)
        "neg_flow_balance":  0.10,   # unchanged
    },
    # COORDINATED_ACTIVITY: reduce composition overlap with STAGING.
    # spike_flag raised (temporal coincidence is the defining coordination
    # signal).  aircraft_density raised (coordinated events require volume).
    Classification.COORDINATED_ACTIVITY: {
        "coordination_flag": 0.30,   # was 0.35 (−0.05)
        "military_ratio":    0.20,   # was 0.25 (−0.05)
        "spike_flag":        0.25,   # was 0.20 (+0.05)  ← key fix
        "aircraft_density":  0.15,   # was 0.10 (+0.05)
        "any_flow":          0.10,   # unchanged
    },
    # ANOMALY: eliminate spike double-counting.  novelty encodes spike_flag
    # internally (contributes +0.3 when spike=True), so at weight 0.30 the
    # spike signal was worth ~39 pts total.  Cut novelty to 0.20.  Redirect
    # to change_score_norm (best-calibrated external signal) and unmilitary
    # (military arrivals should route to PROJECTION/STAGING).
    Classification.ANOMALY: {
        "spike_flag":        0.30,   # unchanged
        "novelty":           0.20,   # was 0.30 (−0.10)  ← key fix
        "low_persistence":   0.15,   # was 0.20 (−0.05)
        "change_score_norm": 0.20,   # was 0.10 (+0.10)  ← key fix
        "unmilitary":        0.15,   # was 0.10 (+0.05)
    },
    # ROUTINE: persistence must be the sole primary driver.  The three
    # absence-of-signal features (no_spike, low_new_entry_ratio, aircraft_density)
    # previously produced a ~40-point noise floor on any quiet region,
    # allowing military staging bases to score 70+.  aircraft_density cut from
    # 0.15 to 0.05 removes ~9 free points.  no_spike and low_new_entry_ratio
    # each cut to 0.15.  unmilitary raised so military regions resist ROUTINE.
    Classification.ROUTINE: {
        "persistence_score":   0.50,   # was 0.35 (+0.15)  ← key fix
        "no_spike":            0.15,   # was 0.20 (−0.05)
        "low_new_entry_ratio": 0.15,   # was 0.20 (−0.05)
        "aircraft_density":    0.05,   # was 0.15 (−0.10)  ← key fix
        "unmilitary":          0.15,   # was 0.10 (+0.05)
    },
}

_MAX_CHANGE_SCORE = 40.0   # normalisation ceiling for change_score
_MAX_FLOW        = 10      # normalisation ceiling for flow counts


# ---------------------------------------------------------------------------
# 4. SCORING FUNCTION
# ---------------------------------------------------------------------------

def _score_region(
    f: RegionFeatures,
    context: Dict[str, float],
) -> Dict[Classification, float]:
    """
    Compute a 0–100 score for each Classification given a feature set.

    `context` carries cross-region normalisation denominators:
        max_inflow   – max inflow_count across all regions
        max_change   – max change_score across all regions
    """
    max_inflow = max(context.get("max_inflow", 1), 1)
    max_change = max(context.get("max_change", _MAX_CHANGE_SCORE), 1.0)

    fv: Dict[str, float] = {
        # Direct features
        "military_ratio":    f.military_ratio,
        "staging_flag":      float(f.staging_flag),
        "persistence_score": f.persistence_score,
        "coordination_flag": float(f.coordination_flag),
        "projection_flag":   float(f.projection_flag),
        "new_entry_ratio":   f.new_entry_ratio,
        "spike_flag":        float(f.spike_flag),
        "aircraft_density":  f.aircraft_density,
        "novelty":           f.novelty,
        # Directional
        "pos_flow_balance":  max(f.flow_balance, 0.0),
        "neg_flow_balance":  max(-f.flow_balance, 0.0),
        # Cross-region normalised
        "inflow_norm":       min(f.inflow_count / max_inflow, 1.0),
        "change_score_norm": min(f.change_score / max_change, 1.0),
        # Composite / derived
        "any_flow":          min(
                                 (f.outflow_count + f.inflow_count) / (_MAX_FLOW * 2),
                                 1.0,
                             ),
        # Inverted
        "low_persistence":   1.0 - f.persistence_score,
        "unmilitary":        1.0 - f.military_ratio,
        "no_spike":          float(not f.spike_flag),
        "low_new_entry_ratio": 1.0 - f.new_entry_ratio,
    }

    return {
        cls: round(sum(weights[k] * fv[k] for k in weights) * 100, 1)
        for cls, weights in _PROFILES.items()
    }


# ---------------------------------------------------------------------------
# 5. CONFIDENCE SCORE
# ---------------------------------------------------------------------------

def _confidence(
    scores: Dict[Classification, float],
    winner: Classification,
    f: RegionFeatures,
) -> int:
    """
    0–100 confidence in the winning classification.

    Three additive components:
      Margin   (70 pts) – normalised gap between winner and runner-up
      Volume   (20 pts) – log-normalised aircraft count
      Coverage (10 pts) – fraction of boolean features that are set

    Margin denominator is 40 (not 100) so that a 20-point lead produces
    ~35 margin pts instead of 12.  This means classification separation
    dominates confidence; volume/coverage serve as tie-breakers only.
    """
    sorted_vals = sorted(scores.values(), reverse=True)
    margin      = sorted_vals[0] - (sorted_vals[1] if len(sorted_vals) > 1 else 0.0)
    margin_pts  = min(margin / 40.0, 1.0) * 70   # was / 100.0 * 60

    volume_pts  = f.aircraft_density * 20          # was * 25

    bool_signals = [
        f.spike_flag,
        f.coordination_flag,
        f.staging_flag,
        f.projection_flag,
        f.outflow_count > 0,
        f.inflow_count > 0,
        f.new_aircraft > 0,
        f.recurring_appearances > 0,
    ]
    coverage_pts = (sum(bool_signals) / len(bool_signals)) * 10  # was * 15

    return min(100, int(margin_pts + volume_pts + coverage_pts))


# ---------------------------------------------------------------------------
# 6. EXPLANATION GENERATOR
# ---------------------------------------------------------------------------

def _explain(f: RegionFeatures, classification: Classification) -> str:
    """Return a human-readable sentence explaining the top classification."""
    parts: List[str] = []

    # --- Volume ---
    if f.aircraft_count > 0:
        parts.append(f"{f.aircraft_count} aircraft observed")

    # --- Composition ---
    if f.military_ratio >= 0.5:
        parts.append(f"majority military ({int(f.military_ratio * 100)}%)")
    elif f.military_ratio >= 0.2:
        parts.append(f"mixed composition ({int(f.military_ratio * 100)}% military)")

    # --- Classification-specific evidence ---
    if classification == Classification.STAGING:
        if f.staging_flag:
            parts.append("confirmed track departure origin")
        if f.outflow_count > 0:
            parts.append(f"net outflow on {f.outflow_count} departure route(s)")
        if f.coordination_flag:
            parts.append("coordinated group activity")
        if f.persistence_score >= 0.25:
            parts.append(f"active in {f.recurring_appearances} time window(s)")

    elif classification == Classification.PROJECTION:
        if f.projection_flag:
            parts.append("confirmed track arrival destination")
        if f.inflow_count > 0:
            parts.append(f"net inflow on {f.inflow_count} arrival route(s)")
        if f.new_entry_ratio >= 0.3:
            parts.append(f"{int(f.new_entry_ratio * 100)}% are new arrivals")

    elif classification == Classification.COORDINATED_ACTIVITY:
        if f.coordination_flag:
            parts.append("simultaneous multi-aircraft activity")
        if f.spike_flag:
            parts.append("count spike vs prior 30-min window")
        if f.military_count > 0:
            parts.append(f"{f.military_count} military aircraft in group")
        if f.outflow_count + f.inflow_count > 0:
            parts.append(f"{f.outflow_count + f.inflow_count} linked movement route(s)")

    elif classification == Classification.ANOMALY:
        if f.spike_flag:
            parts.append("unexpected activity spike")
        if f.new_entry_ratio >= 0.3:
            parts.append(
                f"{int(f.new_entry_ratio * 100)}% of aircraft absent from baseline"
            )
        if f.persistence_score < 0.1:
            parts.append("no prior activity in this region")
        if f.change_level in ("HIGH", "MEDIUM"):
            parts.append(f"activity change classifier: {f.change_level}")

    elif classification == Classification.ROUTINE:
        if f.persistence_score >= 0.5:
            parts.append(f"consistent activity across {f.recurring_appearances}/12 windows")
        elif f.persistence_score > 0:
            parts.append(f"recurring activity ({f.recurring_appearances} window(s))")
        if not f.spike_flag and not f.coordination_flag:
            parts.append("stable, predictable pattern")
        if f.persistence_level == "LONG":
            parts.append("long-term persistent region")

    if not parts:
        parts.append("pattern matched by weighted feature profile")

    return "; ".join(parts)


# ---------------------------------------------------------------------------
# 7. OUTPUT TYPE
# ---------------------------------------------------------------------------

@dataclass
class RegionIntelligence:
    lat:            float
    lon:            float
    classification: Classification
    confidence:     int                   # 0–100
    score:          float                 # winning class score (0–100)
    all_scores:     Dict[str, float]      # scores for all 5 classes
    features:       RegionFeatures
    explanation:    str

    def __str__(self) -> str:
        scores = ", ".join(
            f"{k}={v:.0f}" for k, v in sorted(
                self.all_scores.items(), key=lambda x: -x[1]
            )
        )
        return (
            f"({self.lat}, {self.lon}) "
            f"[{self.classification.value}] "
            f"conf={self.confidence} score={self.score} "
            f"| {scores} "
            f"| {self.explanation}"
        )


# ---------------------------------------------------------------------------
# 8. FEATURE ASSEMBLY
# ---------------------------------------------------------------------------

def _region_key(lat: float, lon: float) -> Tuple[float, float]:
    return (round(lat, 1), round(lon, 1))


def _get(regions: dict, lat: float, lon: float) -> RegionFeatures:
    key = _region_key(lat, lon)
    if key not in regions:
        regions[key] = RegionFeatures(lat=key[0], lon=key[1])
    return regions[key]


def build_features(
    coordinated:      List[Tuple] = (),
    spikes:           List[Tuple] = (),
    recurring:        List[Tuple] = (),
    new_entries:      List[Tuple] = (),
    movements:        List[Tuple] = (),
    staging:          List[Tuple] = (),
    projection:       List[Tuple] = (),
    activity_changes: List[Tuple] = (),
    external_signals: Optional[List[ExternalSignal]] = None,
) -> List[RegionFeatures]:
    """
    Merge all detection outputs into one RegionFeatures per geographic region.

    Input formats (matching existing detection function return types):
        coordinated      → (lat, lon, aircraft_count, military_count)
        spikes           → (lat, lon, aircraft_count, military_count)
        recurring        → (lat, lon, appearances, total_aircraft, military_presence)
        new_entries      → (lat, lon, new_aircraft, new_military)
        movements        → (from_lat, from_lon, to_lat, to_lon, ac_count, mil_count, avg_dist)
        staging          → (lat, lon, aircraft_count, military_count)
        projection       → (lat, lon, aircraft_count, military_count)
        activity_changes → full 16-field change tuples:
                           (lat, lon, change_score, ..., level[7], ..., persistence_level[9], ...)
        external_signals → optional list of ExternalSignal objects; when provided,
                           each region is augmented with mapped signal features
                           after all detection-layer inputs are merged.
                           Passing None (default) is a no-op — no existing
                           behaviour is changed.
    """
    regions: Dict[Tuple, RegionFeatures] = {}

    for lat, lon, ac, mil in coordinated:
        r = _get(regions, lat, lon)
        r.aircraft_count    = max(r.aircraft_count, ac)
        r.military_count    = max(r.military_count, mil)
        r.coordination_flag = True

    for lat, lon, ac, mil in spikes:
        r = _get(regions, lat, lon)
        r.aircraft_count = max(r.aircraft_count, ac)
        r.military_count = max(r.military_count, mil)
        r.spike_flag     = True

    # recurring: (lat, lon, appearances, total_aircraft, military_presence)
    for lat, lon, appearances, total_ac, mil_presence in recurring:
        r = _get(regions, lat, lon)
        r.aircraft_count        = max(r.aircraft_count, total_ac)
        r.military_count        = max(r.military_count, mil_presence)
        r.recurring_appearances = max(r.recurring_appearances, appearances)

    # new_entries: (lat, lon, new_aircraft, new_military)
    for lat, lon, new_ac, new_mil in new_entries:
        r = _get(regions, lat, lon)
        r.new_aircraft   = max(r.new_aircraft, new_ac)
        r.new_military   = max(r.new_military, new_mil)
        r.aircraft_count = max(r.aircraft_count, new_ac)

    # movements: (from_lat, from_lon, to_lat, to_lon, ac_count, mil_count, avg_dist)
    for from_lat, from_lon, to_lat, to_lon, ac, mil, _avg in movements:
        src = _get(regions, from_lat, from_lon)
        src.outflow_count    += ac
        src.outflow_military += mil

        dst = _get(regions, to_lat, to_lon)
        dst.inflow_count    += ac
        dst.inflow_military += mil

    for lat, lon, ac, mil in staging:
        r = _get(regions, lat, lon)
        r.aircraft_count = max(r.aircraft_count, ac)
        r.military_count = max(r.military_count, mil)
        r.staging_flag   = True

    for lat, lon, ac, mil in projection:
        r = _get(regions, lat, lon)
        r.aircraft_count  = max(r.aircraft_count, ac)
        r.military_count  = max(r.military_count, mil)
        r.projection_flag = True

    # activity_changes tuple layout (from detect_activity_changes):
    #   [0] lat   [1] lon   [2] change_score   [7] level   [9] persistence_level
    for row in activity_changes:
        if len(row) < 10:
            continue
        r = _get(regions, row[0], row[1])
        r.change_score      = max(r.change_score, float(row[2]))
        r.change_level      = row[7]
        r.persistence_level = row[9]

    # Merge external signals (if provided) into each region's feature set.
    # This is a no-op when external_signals is None or empty — all existing
    # behaviour is preserved unchanged.
    if external_signals:
        for region in regions.values():
            ext_map = map_external_signals_to_features(external_signals, region)
            merge_external_features(region, ext_map)

    return list(regions.values())


# ---------------------------------------------------------------------------
# 9. TOP-LEVEL CLASSIFY FUNCTION
# ---------------------------------------------------------------------------

def classify_regions(
    features_list: List[RegionFeatures],
    min_aircraft:  int = 2,
) -> List[RegionIntelligence]:
    """
    Score and classify every RegionFeatures object.
    Returns results sorted by (confidence DESC, score DESC).
    """
    if not features_list:
        return []

    # Cross-region normalisation context
    context: Dict[str, float] = {
        "max_inflow": max((f.inflow_count  for f in features_list), default=1),
        "max_change": max((f.change_score  for f in features_list), default=_MAX_CHANGE_SCORE),
    }

    results: List[RegionIntelligence] = []

    for f in features_list:
        if f.aircraft_count < min_aircraft:
            continue

        scores = _score_region(f, context)
        winner = max(scores, key=scores.__getitem__)
        conf   = _confidence(scores, winner, f)
        expl   = _explain(f, winner)

        results.append(RegionIntelligence(
            lat            = f.lat,
            lon            = f.lon,
            classification = winner,
            confidence     = conf,
            score          = scores[winner],
            all_scores     = {c.value: s for c, s in scores.items()},
            features       = f,
            explanation    = expl,
        ))

    results.sort(key=lambda r: (r.confidence, r.score), reverse=True)
    return results
