---
name: data-source-integrator
description: Add new external data sources and integrate them into the ingestion pipeline.
---

# Data Source Integrator

Use this skill when:
- adding a new API or data source
- expanding ingestion coverage

## Workflow

1. Analyze new source format
2. Map fields to internal schema:
   - icao24
   - callsign
   - lat/lon
   - altitude
   - speed
3. Normalize data
4. Create new source file
5. Integrate into ingest pipeline
6. Ensure compatibility with:
   - scoring
   - classification
   - storage

## Output

- new source module
- integration steps
- updated ingest logic