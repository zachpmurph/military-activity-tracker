from __future__ import annotations

import json
import sys
from pathlib import Path

from core.db import resolve_db_path

def load_validation_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_validation_report(report: dict) -> str:
    headline = report.get("summary", {}).get("headline", "Validation report available")
    actions = report.get("recommended_actions", [])
    if not actions:
        return headline
    action_lines = "\n".join(f"- {action}" for action in actions)
    return f"{headline}\nRecommended actions:\n{action_lines}"


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        default_path = resolve_db_path().parent / "exports" / "latest_validation_report.json"
        if not default_path.exists():
            print("Usage: python -m intelligence.validation_cli <validation-report-path>")
            return 1
        args = [str(default_path)]

    report = load_validation_report(args[0])
    print(summarize_validation_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
