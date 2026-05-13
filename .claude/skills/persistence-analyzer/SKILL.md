---
name: persistence-analyzer
description: Analyze time-based aircraft activity and detect persistent or anomalous patterns.
---

# Persistence Analyzer

Use this skill when:
- analyzing repeated aircraft activity
- identifying patterns over time
- detecting anomalies

## Workflow

1. Query recent time window
2. Group by stable identity (icao24)
3. Calculate:
   - sightings count
   - average score
4. Identify:
   - persistent aircraft
   - unusual repetition
5. Highlight:
   - military types
   - high-score unknowns

## Output

- Top persistent aircraft
- Potential anomalies
- Suggested thresholds