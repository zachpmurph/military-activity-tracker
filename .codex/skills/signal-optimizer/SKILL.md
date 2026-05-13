---
name: signal-filter-optimizer
description: Improve filtering, scoring, and noise suppression for aircraft, regions, and external-signal effects in this repository. Use when outputs are too noisy, civilian traffic dominates, or work touches src/scoring/scoring.py, src/intelligence/*, src/query.py, or tests that validate scoring behavior.
---

# Signal Filter Optimizer

Use this skill when the system is technically working but surfacing the wrong things.

## Workflow

1. Identify the noisy layer:
   - aircraft scoring
   - region ranking
   - movement filtering
   - external-signal boosting
2. Inspect the current thresholds and weights before changing them.
3. Separate civilian-noise suppression from military-signal boosting.
4. Keep external signals additive-only and consistent with `CLAUDE.md`.
5. Prefer small weight or threshold changes over structural rewrites.
6. Add or update regression tests that prove the intended ranking order.

## Output

- Which inputs are creating noise
- Which threshold or weight should move
- The smallest scoring or filter patch that improves ranking without breaking invariants
