---
name: data-source-integrator
description: Add new aircraft or intelligence data sources and integrate them into this repository's ingestion and normalization pipeline. Use when creating new source modules under src/sources/* or new intelligence ingestion paths under src/intelligence/*, and when new data must map cleanly into persistence, scoring, and classifier inputs.
---

# Data Source Integrator

Use this skill when the project needs broader coverage, better fallback data, or a new external signal feed.

## Workflow

1. Classify the source type first:
   - aircraft position feed
   - maritime feed
   - NOTAM feed
   - satellite or other external signal
2. Decide the integration point:
   - `src/sources/*` for raw aircraft sources
   - `src/intelligence/*_ingestion.py` for external signal pipelines
3. Map source fields into the internal schema and note every missing or approximate field.
4. Normalize early so downstream scoring and persistence do not need source-specific branches.
5. Confirm storage compatibility with:
   - `src/core/db.py`
   - `src/ingest/persistence.py`
   - `src/intelligence/external_features.py`
6. Add focused tests for ingestion and one end-to-end behavior check if the new source changes classification.

## Output

- New source or ingestion module
- Normalization rules
- Test coverage proving the data reaches the expected downstream path
