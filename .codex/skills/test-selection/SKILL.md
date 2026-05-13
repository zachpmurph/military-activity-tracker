---
name: test-selection
description: Choose the smallest reliable test subset for a code change in this repository, then expand only as needed. Use when work spans multiple modules and it is not obvious which tests should run first or which broader regression checks are required afterward.
---

# Test Selection

## Overview

Use this skill to map changed files to the fastest meaningful tests. Start narrow, then widen to nearby module tests, then run broader validation only if the change crosses subsystem boundaries.

## Test Map

- Query, detection, geo, and region logic:
  - `tests/test_query.py`

- Ingest, persistence, and database behavior:
  - `tests/test_ingest.py`
  - `tests/test_config.py` when config or DB-path behavior changed

- Classifier and scoring behavior:
  - `tests/test_classifier.py`
  - `tests/test_scoring_contract.py`
  - `tests/test_calibration.py`

- External feature and external classification behavior:
  - `tests/test_external_features.py`
  - `tests/test_external_classification.py`
  - `tests/validation/test_scoring_contract.py`
  - `tests/validation/test_exp3_context_suppression.py`
  - `tests/validation/test_exp4_bypass_ghost.py`

- Intelligence ingestion adapters:
  - `tests/test_maritime_ingestion.py`
  - `tests/test_notam_ingestion.py`
  - `tests/test_satellite_ingestion.py`

- Runner and observability behavior:
  - `tests/test_runner.py`
  - `tests/test_observability.py`

## Workflow

1. Start with the file-to-test map above.
2. Run the narrowest test that directly covers the changed module.
3. If the change touches shared logic, run the neighboring suite next.
4. If scoring or external-signal behavior changes, include validation tests before claiming success.
5. Prefer one explicit command over broad “run everything” unless the change really crosses subsystems.

## Default Bias

- `src/query.py` or `src/detection/*` changed: run `tests/test_query.py`
- `src/ingest/*` or DB persistence changed: run `tests/test_ingest.py`
- `src/intelligence/*` or `src/scoring/*` changed: run the matching unit tests and validation tests
- Multiple layers changed: run the targeted suites for each touched layer, then decide if a broader pass is needed
