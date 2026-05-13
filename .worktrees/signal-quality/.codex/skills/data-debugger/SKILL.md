---
name: data-pipeline-debugger
description: Debug ingestion pipelines, database issues, and data consistency problems.
---

# Data Pipeline Debugger

Use this skill when:
- ingestion fails
- database queries return empty results
- timestamps or schema issues occur

## Workflow

1. Check database connection path consistency
2. Inspect schema:
   - tables
   - column types
3. Validate timestamp format:
   - detect unix vs datetime mismatch
4. Run ingestion script and capture errors
5. Check row counts and recent inserts
6. Identify mismatch between ingest and query logic

## Output

- Root cause of issue
- Exact fix
- Code patch if needed