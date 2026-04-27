# CLAUDE.md — Execution Protocol

## Purpose

Real-time aircraft activity classification using:

* Internal detection features (primary signal)
* External signals (NO_FLY, MARITIME, SATELLITE)

External signals **enhance** detection — never replace it.

---

## Core Pipeline

```text
raw data → build_features() → classify_regions() → explanations
```

* `build_features()` merges detection + external signals
* `classify_regions()` produces scores:

  * ROUTINE
  * ANOMALY
  * COORDINATED_ACTIVITY
  * STAGING
  * PROJECTION

---

## External Signals

### Types

* NO_FLY
* MARITIME
* SATELLITE

### Flow

```text
ExternalSignal → map_external_signals_to_features()
               → merge_external_features()
               → RegionFeatures
```

---

## Spatial Influence (CRITICAL)

* `_PROXIMITY_RADIUS = 1.5`
* Distance: Chebyshev → `max(|Δlat|, |Δlon|)`

### Weight

```text
weight = 1 - (distance / radius)
effective_intensity = clamp(intensity × weight)
```

Rules:

* Center → full effect
* Edge → zero
* Outside → ignored
* All outputs must remain ∈ [0,1]

---

## Feature Mapping Rules

* Aggregate per feature using **max**
* External signals **never reduce** existing values

| Signal    | Effects                   |
| --------- | ------------------------- |
| NO_FLY    | spike_flag, change_score  |
| MARITIME  | coordination_flag, inflow |
| SATELLITE | change_score, novelty     |

---

## External Influence Logic

### Spike Attribution

```text
spike_set_by_external = True
```

Only if:

* was False before merge
* becomes True after merge

---

### Min-Aircraft Bypass

Allow classification if:

```text
external_signal_count > 0
AND external_signal_max_intensity ≥ 0.75
```

---

### Strength Metrics

* `external_signal_strength` = Σ(intensity × weight), capped
* `maritime_signal_strength` = same, MARITIME only

---

### Boosting

#### ANOMALY

```text
boost = 1 + min(MAX, strength × factor)
+ cross_domain_bonus (NO_FLY + MARITIME)
```

Only if:

```text
spike_set_by_external == True
```

---

#### COORDINATED_ACTIVITY

Applied if:

```text
maritime_signal_strength > 0
```

---

## Observability

### Toggle

```text
ENABLE_EXTERNAL_DEBUG = True/False
```

### Logs

* Merge:

```text
signals=[TYPE(int=1.00 w=0.67 eff=0.67)]
delta=[...]
```

* Classification impact:

  * baseline vs post
  * delta
  * winner change
  * external_influence flag

---

## Invariants (DO NOT BREAK)

* Feature values ∈ [0,1]
* Scores ∈ [0,100]
* External signals never reduce values
* APIs unchanged unless required
* Tests define correctness

---

## Development Workflow (MANDATORY)

### 1. Read

* Inspect all relevant files first

### 2. Plan

* Define minimal changes
* Do not refactor unnecessarily

### 3. Implement

* Make surgical edits only

### 4. Test

* Run full test suite
* Fix root cause, not symptoms

### 5. Verify

* Validate behavior via real output
* Check debug logs if applicable

---

## Testing Rules

* Full suite must pass
* No silent behavior changes
* Prefer fixing implementation over tests
* Add tests for:

  * edge cases
  * new logic
  * regressions

---

## Prompting Rules (Claude Code / Ruflo)

Every task must specify:

1. Files to read
2. Exact change required
3. Constraints
4. Expected behavior
5. Test expectations

Avoid:

* vague instructions
* broad rewrites
* unnecessary abstraction

---

## Priority Rules (OVERRIDE ALL)

1. **Do not break invariants**
2. **Do not change behavior without tests**
3. **Prefer minimal diffs**
4. **Preserve determinism**
5. **External signals must remain additive-only**

---

## Current Capabilities

* Weighted spatial external signals
* Dynamic anomaly + coordination boosts
* Cross-domain interaction (NO_FLY + MARITIME)
* Full observability + debug logging
* Complete test coverage

---

## Guiding Rule

> Flight data drives detection. External signals adjust confidence.
