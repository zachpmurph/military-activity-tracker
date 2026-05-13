---
name: intelligence-signal-tuning
description: Tune classifier behavior, external-signal influence, calibration, and scoring in this repository without breaking the additive-only external-signal contract. Use when changing src/intelligence/*, src/scoring/scoring.py, tests/test_classifier.py, tests/test_external_features.py, tests/test_external_classification.py, or tests/validation/*.
---

# Intelligence Signal Tuning

## Overview

Use this skill when the system is classifying the wrong winner, over- or under-weighting external signals, or failing scoring-contract tests. Follow `CLAUDE.md` first because it defines non-negotiable invariants for external effects.

## Workflow

1. Read `CLAUDE.md` before changing anything in `src/intelligence/*` or `src/scoring/*`.
2. Identify which layer is wrong:
   - feature building
   - external feature merge
   - calibration
   - scoring
   - final winner selection
3. Preserve these invariants:
   - external signals never reduce values
   - feature values stay in `[0,1]`
   - scores stay in `[0,100]`
   - APIs stay stable unless the task requires otherwise
4. Prefer implementation fixes over weakening tests.
5. Run the narrowest relevant tests first, then the validation tests if external behavior changed.

## File Map

- `src/intelligence/classifier.py`: region classification and winner logic
- `src/intelligence/external_features.py`: external feature merge and attribution
- `src/intelligence/calibration.py`: score adjustment logic
- `src/scoring/scoring.py`: scoring rules
- `tests/test_external_features.py`, `tests/test_external_classification.py`, `tests/test_classifier.py`: unit coverage
- `tests/validation/*`: contract and regression behavior for external signals

## Guardrails

- Do not let external signals replace core flight-data detection.
- Treat debug and observability output as part of the expected behavior when those tests cover it.
- When changing weights, explain whether the fix targets false positives, false negatives, or winner instability.
