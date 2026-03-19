from __future__ import annotations

import json
from pathlib import Path

from nudemo.benchmarks.models import BenchmarkReport


def export_report(report: BenchmarkReport, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)
    return output_path


def load_report(path: str | Path) -> BenchmarkReport:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return BenchmarkReport.from_dict(payload)
