from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from nudemo.benchmarks.models import BenchmarkReport, BenchmarkResult


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


def iter_report_paths(reports_root: str | Path) -> list[Path]:
    root = Path(reports_root)
    candidates: list[Path] = []
    direct_report = root / "benchmark_report.json"
    if direct_report.exists():
        candidates.append(direct_report)
    candidates.extend(sorted(root.glob("overnight_runs/*/*/benchmark_report.json")))
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        ordered.append(path)
        seen.add(resolved)
    return ordered


def load_latest_comparison_report(reports_root: str | Path) -> BenchmarkReport:
    selected_reports = load_latest_backend_reports(reports_root)
    if not selected_reports:
        report_path = Path(reports_root) / "benchmark_report.json"
        raise FileNotFoundError(report_path)

    merged_results: list[BenchmarkResult] = []
    created_ats: list[str] = []
    providers: set[str] = set()
    sample_scopes: set[tuple[object, object]] = set()

    for backend in sorted(selected_reports):
        _, report = selected_reports[backend]
        created_ats.append(report.created_at)
        provider = report.dataset.get("provider")
        if provider:
            providers.add(str(provider))
        sample_scopes.add((report.dataset.get("samples"), report.dataset.get("scenes")))
        for result in report.results:
            if result.backend != backend:
                continue
            metadata = dict(result.metadata)
            metadata.setdefault("source_suite_name", report.suite_name)
            metadata.setdefault("source_run_id", report.dataset.get("run_id", ""))
            metadata.setdefault("source_provider", report.dataset.get("provider", ""))
            metadata.setdefault("source_samples", report.dataset.get("samples", ""))
            metadata.setdefault("source_scenes", report.dataset.get("scenes", ""))
            metadata.setdefault("source_created_at", report.created_at)
            merged_results.append(
                BenchmarkResult(
                    stage=result.stage,
                    backend=result.backend,
                    pattern=result.pattern,
                    metrics=dict(result.metrics),
                    metadata=metadata,
                    sample_count=result.sample_count,
                    elapsed_sec=result.elapsed_sec,
                    status=result.status,
                    error=result.error,
                )
            )

    dataset: dict[str, int | float | str] = {
        "provider": providers.pop() if len(providers) == 1 else "mixed",
        "run_id": "combined-latest",
        "comparison_scope": "latest_per_backend",
    }
    if len(sample_scopes) == 1:
        samples, scenes = next(iter(sample_scopes))
        dataset["samples"] = "" if samples is None else samples
        dataset["scenes"] = "" if scenes is None else scenes
    else:
        dataset["samples"] = "mixed"
        dataset["scenes"] = "mixed"

    return BenchmarkReport(
        suite_name="nuDemo latest backend comparison",
        dataset=dataset,
        results=merged_results,
        created_at=max(created_ats, key=_sort_created_at),
    )


def load_latest_backend_reports(
    reports_root: str | Path,
) -> dict[str, tuple[Path, BenchmarkReport]]:
    selected: dict[str, tuple[Path, BenchmarkReport]] = {}
    for report_path in iter_report_paths(reports_root):
        report = load_report(report_path)
        for backend in _storage_backends(report):
            current = selected.get(backend)
            if current is None or _sort_created_at(report.created_at) >= _sort_created_at(
                current[1].created_at
            ):
                selected[backend] = (report_path, report)
    return selected


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


def _storage_backends(report: BenchmarkReport) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for result in report.results:
        if result.backend in {"Kafka", "real"}:
            continue
        if result.backend in seen:
            continue
        ordered.append(result.backend)
        seen.add(result.backend)
    return ordered


def _sort_created_at(value: object) -> datetime:
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)


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
