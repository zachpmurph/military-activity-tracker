---
name: persistence-analyzer
description: Analyze time-based aircraft activity, persistent tracks, staging, projection, and movement patterns in this repository. Use when working on src/query.py, src/detection/*, aircraft_tracks-based logic, or tests/test_query.py that depend on repeated activity over time.
---

# Persistence Analyzer

Use this skill when a task depends on persistence over time rather than a single snapshot.

## Workflow

1. Identify the time window the feature actually cares about:
   - 30 minutes
   - 90 minutes
   - 2 hours
   - 6 hours
2. Decide whether the task should read from `aircraft_positions`, `aircraft_tracks`, or both.
3. Group by stable identifiers first:
   - `icao24` for aircraft
   - merged rounded regions for area-level activity
   - origin and destination bins for flow logic
4. Check whether the logic is meant to detect:
   - repetition
   - emergence
   - movement
   - staging/projection
   - escalation
5. Prefer merged-region reasoning over raw bins when adjacent cells would create duplicates.
6. Back every change with a minimal SQLite fixture in `tests/test_query.py`.

## Output

- The persistent pattern being measured
- The threshold or merge rule that makes it meaningful
- A concrete query or test update that captures the intended behavior
