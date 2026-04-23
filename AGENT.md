# Project Rules

## Goal
Detect meaningful military activity patterns from ADS-B data.

## Stack
- Python
- SQLite (data/aircraft.db)
- No paid APIs

## Key Files
- src/ingest.py → data ingestion
- src/query.py → analysis queries
- src/sources/ → data sources

## Rules
- Use Unix timestamps (float)
- Keep logic simple (heuristics > ML)
- Do NOT refactor unrelated files
- Only modify requested functions

## Output Style
- Minimal explanation
- Show only changed code