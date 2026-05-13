# Scoring Contract — OSINT Threat Platform

## 1. Overview

The scoring pipeline converts raw aircraft detection data into structured threat intelligence
through four deterministic layers.  Each layer builds on the previous; no layer can modify the
outputs of a lower layer.

```
Layer 1  RegionFeatures        — normalised feature vector (0–1 per field)
Layer 2  Classification Scores — five 0–100 scores per region
Layer 3  Winner + Confidence   — argmax classification, 0–100 confidence
Layer 4  Composite Scores      — Regional Military Action, Global Tension
```

External signals (NO_FLY, MARITIME, SATELLITE) influence Layers 1 and 2 only, and only in
an additive direction.  They never reduce an existing value.

---

## 2. Layer 1 — Feature Vector

All features are normalised to [0, 1] (or [−1, +1] for flow_balance).
The classifier reads only the derived properties; raw counts are for observability.

| Feature | Formula | Range |
|---|---|---|
| `military_ratio` | `military_count / max(aircraft_count, 1)` | [0, 1] |
| `new_entry_ratio` | `new_aircraft / max(aircraft_count, 1)` | [0, 1] |
| `persistence_score` | `recurring_appearances / 12` | [0, 1] |
| `flow_balance` | `(outflow − inflow) / (outflow + inflow)` | [−1, +1] |
| `aircraft_density` | `log(count + 1) / log(101)` | [0, 1] |
| `novelty` | `new_entry_ratio × 0.5 + (0.3 if spike_flag) + (1 − persistence_score) × 0.2` | [0, 1] |
| `spike_flag` | set by detection layer or NO_FLY external signal | bool |
| `coordination_flag` | set by detection layer or MARITIME external signal | bool |
| `staging_flag` | track-level departure origin confirmed | bool |
| `projection_flag` | track-level arrival destination confirmed | bool |

**Context normalisation** — two features are normalised against the batch:

| Feature | Formula | Notes |
|---|---|---|
| `change_score_norm` | `change_score / max_change` | `max_change` is floored at 40.0 (FM-3 fix) |
| `inflow_norm` | `inflow_count / max_inflow` | `max_inflow` = max across all regions in batch |

The floor at 40.0 means `change_score_norm` is always computed against the design ceiling
regardless of batch composition.  A region with `change_score = 20` always produces
`change_score_norm = 0.5`, not 1.0 when it happens to be the only region in the batch.

---

## 3. Layer 2 — Classification Scores

Score formula: `score_c = dot(weights_c, fv) × 100`

All weights within a profile sum to 1.0.  The dot product is in [0, 1]; multiplying by 100
gives a 0–100 score.

### ANOMALY — unexpected activity, no prior context

| Feature | Weight | Rationale |
|---|---|---|
| `spike_flag` | 0.30 | Primary signal — sudden count increase |
| `change_score_norm` | 0.20 | Best-calibrated external signal proxy |
| `novelty` | 0.20 | Composite: new entries + spike + low persistence |
| `unmilitary` | 0.15 | Military arrivals route to STAGING/PROJECTION |
| `low_persistence` | 0.15 | Absence of prior history amplifies surprise |

`novelty` encodes `spike_flag` internally (+0.3 when set).  Combined with the direct
`spike_flag` weight, a pure spike contributes up to 39 ANOMALY points.

### ROUTINE — known, stable, civilian traffic

| Feature | Weight | Rationale |
|---|---|---|
| `persistence_score` | 0.50 | Must be the primary discriminator |
| `no_spike` | 0.15 | Absence of disruption |
| `low_new_entry_ratio` | 0.15 | Stable composition |
| `unmilitary` | 0.15 | Military presence resists ROUTINE |
| `aircraft_density` | 0.05 | Volume (low weight avoids noise floor) |

Persistence is the single largest weight in the system.  A region seen across 10/12 windows
(`persistence_score = 0.833`) scores ROUTINE ≈ 54 from this feature alone.

### STAGING — force accumulation and departure

| Feature | Weight | Rationale |
|---|---|---|
| `staging_flag` | 0.30 | Definitive structural signal |
| `pos_flow_balance` | 0.30 | Net outflow is the directional discriminator |
| `military_ratio` | 0.25 | Composition validation |
| `coordination_flag` | 0.10 | Multi-aircraft coincidence |
| `persistence_score` | 0.05 | Low weight — long-running staging must not inflate ROUTINE |

### PROJECTION — force reception at destination

| Feature | Weight | Rationale |
|---|---|---|
| `projection_flag` | 0.40 | Track-level arrival evidence dominates |
| `inflow_norm` | 0.25 | Volume of arriving routes |
| `new_entry_ratio` | 0.15 | New aircraft entering a known destination |
| `military_ratio` | 0.10 | Composition check |
| `neg_flow_balance` | 0.10 | Directional confirmation (inflow dominant) |

### COORDINATED_ACTIVITY — synchronised multi-aircraft event

| Feature | Weight | Rationale |
|---|---|---|
| `coordination_flag` | 0.30 | Simultaneous multi-aircraft detection |
| `spike_flag` | 0.25 | Temporal coincidence of the coordination event |
| `military_ratio` | 0.20 | Military-dominant events score higher |
| `aircraft_density` | 0.15 | Coordination events require volume |
| `any_flow` | 0.10 | Movement linkage |

---

## 4. Layer 3 — Winner and Confidence

### Winner

```
winner = argmax({STAGING, PROJECTION, ROUTINE, ANOMALY, COORDINATED_ACTIVITY})
```

Ties are broken by the natural dict iteration order of `_PROFILES` (STAGING first).

### Min-aircraft guard

Regions with `aircraft_count < min_aircraft` (default 2) are excluded from scoring unless
a high-confidence external signal is present:

```
bypass = (
    external_signal_count > 0
    AND external_signal_max_intensity >= 0.75   (RESTRICTED or PROHIBITED level)
    AND external_signal_strength      >= 0.10   (non-trivial effective influence)
)
```

The dual condition prevents ghost bypasses: a signal near the proximity boundary (weight ≈ 0)
can satisfy the raw intensity check but fails the effective strength check.

### Confidence formula

Confidence expresses how trustworthy the winner is, not how strong the signal is.

```
margin_pts  = min(margin / 40, 1.0) × 70    where margin = winner_score − runner_up_score
volume_pts  = aircraft_density × 20
coverage_pts = (boolean_signals_set / 8) × 10
confidence  = int(margin_pts + volume_pts + coverage_pts)   capped at 100
```

Boolean signals counted: `spike_flag`, `coordination_flag`, `staging_flag`, `projection_flag`,
`outflow_count > 0`, `inflow_count > 0`, `new_aircraft > 0`, `recurring_appearances > 0`.

A 20-point margin produces ~35 margin_pts.  Classification separation dominates; volume
and coverage are tie-breakers only.

---

## 5. External Signal Influence

### Signal types and feature mappings

| Signal Type | Fields Modified | Formula |
|---|---|---|
| NO_FLY | `spike_flag`, `change_score` | `spike_flag = True`; `Δchange_score = eff × 0.80 × 40` |
| MARITIME | `coordination_flag`, `inflow_count` | `coordination_flag = True`; `Δinflow = eff × 0.80 × 10` |
| SATELLITE | `change_score`, `new_aircraft` | `Δchange_score = eff × 0.70 × 40`; `Δnew_aircraft = eff × 0.50 × count` |

Effective intensity: `eff = clamp(intensity × weight, 0, 1)`
Distance weight: `weight = 1 − (chebyshev_distance / 1.5)` for signals within radius 1.5°

Merge rule: the delta is only applied if it raises the existing value.  Signals never reduce.

### ANOMALY boost

Applied only when `spike_set_by_external = True` (detection layer did not set `spike_flag`; the
external merge did).

```
boost = 1.0 + min(0.25, external_signal_strength × 0.10)
      [+ 0.05  if >= 2 distinct signal types present]
      [+ 0.05  if both NO_FLY and MARITIME present (cross-domain)]

ANOMALY_score = min(100, ANOMALY_score × boost)
```

Maximum boost multiplier: 1.35 (single signal at strength=2.5 with cross-domain bonus).

### COORDINATED_ACTIVITY boost

Applied when `maritime_signal_strength > 0`.

```
coord_boost = 1.0 + min(0.20, maritime_signal_strength × 0.10)
            [+ 0.05  if both NO_FLY and MARITIME present (cross-domain)]

COORDINATED_ACTIVITY_score = min(100, COORDINATED_ACTIVITY_score × coord_boost)
```

### Signal intensity tiers

| Tier | Intensity | Bypass eligible |
|---|---|---|
| ADVISORY | 0.25 | No |
| WARNING | 0.50 | No |
| RESTRICTED | 0.75 | Yes (if effective strength >= 0.10) |
| PROHIBITED | 1.00 | Yes (if effective strength >= 0.10) |

---

## 6. Layer 4 — Composite Scores

These scores are derived from Layer 2 (raw class scores, before winner selection).  They are
not stored in `RegionIntelligence`; they are computed by the caller from `all_scores`.

### Regional Military Action (RMA)

Measures the intensity of structured, deliberate military activity in a single region.

```
RMA = 0.40 × STAGING_score + 0.35 × PROJECTION_score + 0.25 × COORDINATED_ACTIVITY_score
```

| Score | Interpretation |
|---|---|
| 0–20 | No evidence of military movement |
| 21–40 | Low-level activity; confirm with additional sources |
| 41–60 | Moderate activity; indicators of deliberate positioning |
| 61–80 | High activity; structured force movement underway |
| 81–100 | Confirmed military action with strong multi-source evidence |

RMA peaks when all three component classes score simultaneously, which requires:
- `staging_flag` or `projection_flag` set (structural confirmation)
- High `military_ratio`
- `coordination_flag` set with meaningful flow data

### Global Tension (GT)

Aggregates per-region threat signals across the entire active theater.

Per-region tension contribution:

```
T_i = 0.40 × ANOMALY_score_i + 0.35 × COORDINATED_ACTIVITY_score_i
    + 0.25 × max(STAGING_score_i, PROJECTION_score_i)
```

Theater-wide aggregate:

```
GT = mean(T_i for all regions with aircraft_count >= min_aircraft
          OR bypass_active)
     capped at 100
```

| Score | Interpretation |
|---|---|
| 0–15 | Normal background activity |
| 16–30 | Elevated; isolated anomalies present |
| 31–50 | Heightened; multiple anomalous regions active |
| 51–70 | High; coordinated or structured military activity visible |
| 71–100 | Critical; widespread anomalies with multi-domain signals |

GT rises fastest when multiple regions simultaneously show ANOMALY and COORDINATED_ACTIVITY.
A single STAGING region has less effect because its contribution is capped at 0.25 × STAGING_score.

---

## 7. Scoring Scenarios

### Scenario A — Quiet civilian airport (ROUTINE)

```
aircraft_count=24, military_count=0, recurring_appearances=10
spike_flag=False, change_score=0, new_aircraft=2
```

Feature values:
- `military_ratio = 0.0`
- `persistence_score = 10/12 = 0.833`
- `novelty = 0.2 × 0.0 + 0 + 0.167 × 0.2 = 0.033`
- `change_score_norm = 0/40 = 0.0`

Scores:
- ROUTINE: `0.50 × 0.833 + 0.15 × 1.0 + 0.15 × 0.917 + 0.05 × 0.42 + 0.15 × 1.0 ≈ 90.9`
- ANOMALY: `0.30 × 0 + 0.20 × 0 + 0.20 × 0.033 + 0.15 × 1.0 + 0.15 × 0.833 ≈ 27.6`

Winner: ROUTINE (margin ≈ 63 pts → confidence ≈ 90)
GT contribution: T = 0.40 × 27.6 + 0.35 × ~5 + 0.25 × ~5 ≈ 14.1

---

### Scenario B — Anomalous spike, new activity (ANOMALY)

```
aircraft_count=8, military_count=4, recurring_appearances=0
spike_flag=True, change_score=20, new_aircraft=6
```

Feature values:
- `military_ratio = 0.5`
- `persistence_score = 0`
- `new_entry_ratio = 0.75`
- `novelty = 0.75 × 0.5 + 0.3 + 1.0 × 0.2 = 0.875`
- `change_score_norm = 20/40 = 0.5`

Scores:
- ANOMALY: `0.30 × 1.0 + 0.20 × 0.875 + 0.15 × 1.0 + 0.20 × 0.5 + 0.15 × 0.5 = 36.5 × ... ≈ 63.0`
- ROUTINE: `0.50 × 0 + 0.15 × 0 + 0.15 × 0.25 + 0.05 × 0.25 + 0.15 × 0.5 ≈ 15.4`

Winner: ANOMALY
GT contribution (no external): T = 0.40 × 63.0 + 0.35 × ~12 + 0.25 × ~18 ≈ 34.5

---

### Scenario C — Same region with NO_FLY signal (ANOMALY boosted)

Same region as B, plus a NO_FLY signal at offset=0.5° (weight = 1 − 0.5/1.5 = 0.667):

```
intensity=0.80  →  eff = 0.80 × 0.667 = 0.533
external_signal_strength = 0.533
spike_set_by_external = False  (spike_flag was already True from detection layer)
```

Because `spike_set_by_external = False`, the ANOMALY boost does **not** apply.  The signal
raises `change_score` by `0.533 × 0.80 × 40 = 17.1` (if that exceeds the existing value),
which increases `change_score_norm`.

If `spike_set_by_external = True` (detection layer had NOT set spike_flag):
```
boost = 1.0 + min(0.25, 0.533 × 0.10) = 1.0533
ANOMALY_boosted = 63.0 × 1.053 = 66.3
```

---

### Scenario D — Force staging base (STAGING)

```
aircraft_count=20, military_count=16, recurring_appearances=4
staging_flag=True, outflow_count=8, inflow_count=2
```

Feature values:
- `military_ratio = 0.8`
- `persistence_score = 4/12 = 0.333`
- `flow_balance = (8−2)/(8+2) = 0.6`
- `pos_flow_balance = 0.6`

Scores:
- STAGING: `0.30 × 1.0 + 0.30 × 0.6 + 0.25 × 0.8 + 0.10 × 0 + 0.05 × 0.333 ≈ 68.2`
- ROUTINE: `0.50 × 0.333 + 0.15 × 1.0 + 0.15 × 1.0 + 0.05 × 0.35 + 0.15 × 0.2 ≈ 48.2`

Winner: STAGING
RMA = 0.40 × 68.2 + 0.35 × ~10 + 0.25 × ~15 ≈ 34.0 (moderate — no PROJECTION evidence)

---

### Scenario E — Cross-domain: NO_FLY + MARITIME (maximum boost)

```
ANOMALY_base = 55.0
external_signal_strength = 1.8  (two strong signals)
maritime_signal_strength = 0.9
spike_set_by_external = True
cross_domain = True  (NO_FLY + MARITIME both present)
```

ANOMALY boost:
```
boost = 1.0 + min(0.25, 1.8 × 0.10) + 0.05 (multi-type) + 0.05 (cross-domain)
      = 1.0 + 0.18 + 0.05 + 0.05 = 1.28
ANOMALY_final = min(100, 55.0 × 1.28) = 70.4
```

COORDINATED_ACTIVITY boost:
```
coord_boost = 1.0 + min(0.20, 0.9 × 0.10) + 0.05 (cross-domain)
            = 1.0 + 0.09 + 0.05 = 1.14
COORD_final = min(100, base_coord × 1.14)
```

---

## 8. Invariants

These must hold after any code change:

| Invariant | Source |
|---|---|
| All feature values ∈ [0, 1] (flow_balance ∈ [−1, +1]) | `RegionFeatures` properties |
| All class scores ∈ [0, 100] | `_score_region` + boost caps |
| Confidence ∈ [0, 100] | `_confidence` cap |
| External signals never reduce feature values | `merge_external_features` additive-only |
| `change_score_norm` denominator >= 40.0 | FM-3 fix: max_change floor |
| Ghost bypass rejected: effective strength >= 0.10 required | FM-4 fix: dual bypass condition |
| ANOMALY boost applies only when `spike_set_by_external = True` | Prevents double-counting |
| Profile weights sum to 1.0 per class | Validated by calibration harness |
