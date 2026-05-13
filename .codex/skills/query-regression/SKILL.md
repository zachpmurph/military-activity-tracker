---
name: query-regression
description: Update or debug SQLite-backed query and detection logic in this repository while preserving output shape and regression coverage. Use when changing src/query.py, src/detection/*, src/geo/*, aircraft_tracks-driven analysis, or tests/test_query.py.
---

# Query Regression

## Overview

Use this skill for region, movement, staging, flow, and change-detection work that is verified through in-memory SQLite fixtures. Favor minimal, test-first changes and preserve the printed tuple format unless the task explicitly asks to change it.

## Workflow

1. Read the target query and the matching section of `tests/test_query.py` first.
2. Decide which persistence source drives the feature:
   - `aircraft_positions`
   - `aircraft_tracks`
   - both
3. Write or update the smallest failing in-memory SQLite test before changing production code.
4. Keep region binning, merge rules, and tuple output explicit in the test.
5. Patch the implementation with the narrowest possible SQL or Python merge logic.
6. Re-run the focused query test first, then the broader query suite.

## Repo Patterns

- Region logic lives in `src/query.py` and `src/detection/*`.
- Spatial helpers live in `src/geo/*`.
- Tests belong in `tests/test_query.py`.
- Printed outputs are part of the contract for many query functions.

## Guardrails

- Prefer SQLite-compatible SQL and simple Python post-processing.
- Use merged regions when adjacent bins would otherwise duplicate a signal.
- Keep output headers and tuple ordering stable unless the task explicitly changes them.
- When a rule is easier to express in Python than SQL, keep the SQL snapshot simple and do the merge or matching step in Python.
