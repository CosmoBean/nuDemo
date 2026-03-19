from __future__ import annotations

import csv
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


def flatten_report(report: BenchmarkReport) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in report.results:
        row: dict[str, object] = {
            "stage": result.stage,
            "backend": result.backend,
            "pattern": result.pattern,
            "sample_count": result.sample_count,
            "elapsed_sec": result.elapsed_sec,
            "status": result.status,
            "error": result.error or "",
        }
        row.update(result.metrics)
        row.update({f"meta_{key}": value for key, value in result.metadata.items()})
        rows.append(row)
    return rows


def export_report_bundle(
    report: BenchmarkReport,
    output_root: str | Path,
) -> tuple[Path, Path, Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    report_path = export_report(report, root / "benchmark_report.json")
    flat_json_path = root / "benchmark_results.json"
    csv_path = root / "benchmark_results.csv"
    rows = flatten_report(report)
    with flat_json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    fieldnames = (
        sorted({key for row in rows for key in row})
        if rows
        else ["stage", "backend", "pattern"]
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return report_path, flat_json_path, csv_path
