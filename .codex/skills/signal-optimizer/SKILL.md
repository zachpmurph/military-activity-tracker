---
name: signal-filter-optimizer
description: Improve filtering and scoring of aircraft to reduce noise and highlight meaningful signals.
---

# Signal Filter Optimizer

Use this skill when:
- too many civilian aircraft appear
- signal-to-noise ratio is poor
- classification or scoring needs tuning

## Workflow

1. Analyze current output:
   - most_suspicious
   - persistent_aircraft
2. Identify noise sources:
   - civilian airlines
   - private jets
3. Evaluate classification logic
4. Adjust:
   - type filters
   - score thresholds
   - behavior weighting
5. Suggest updated rules

## Output

- List of filtering issues
- Improved scoring logic
- Updated code snippets