---
name: data-pipeline-debugger
description: Debug ingestion pipelines, SQLite persistence, track-table issues, and data consistency problems in this repository. Use when work touches src/ingest/*, src/core/db.py, src/query.py, src/data/aircraft.db, or tests such as tests/test_ingest.py and tests/test_query.py, especially if rows are missing, timestamps do not line up, or schema assumptions drift.
---

# Data Pipeline Debugger

Use this skill when ingestion succeeds but downstream queries look wrong, when tests fail around persistence, or when `aircraft_positions` and `aircraft_tracks` disagree.

## Workflow

1. Confirm the active database path and the canonical repo root.
2. Inspect the live schema and row counts for `aircraft_positions` and `aircraft_tracks`.
3. Check time-window assumptions:
   - unix timestamps
   - recent vs prior windows
   - cleanup cutoffs
4. Trace the relevant ingest path:
   - `src/ingest/pipeline.py`
   - `src/ingest/persistence.py`
   - `src/ingest/normalization.py`
   - `src/ingest/deduplication.py`
5. Compare stored rows with the query logic in `src/query.py` or `src/detection/*`.
6. Reproduce the problem with the smallest test or SQLite fixture possible before patching.
7. Fix the root cause and re-run the narrowest relevant tests first.

## Output

- Root cause in one sentence
- Exact file or schema mismatch causing it
- Smallest safe fix
- Test command that proves the issue is resolved
