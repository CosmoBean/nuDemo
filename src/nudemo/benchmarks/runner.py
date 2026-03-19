from __future__ import annotations

import csv
import json
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from nudemo.storage.base import StorageBackend, StorageWriteResult


@dataclass(slots=True)
class BenchmarkRecord:
    backend: str
    pattern: str
    metrics: dict[str, float | int | str]

    def as_row(self) -> dict[str, float | int | str]:
        return {"backend": self.backend, "pattern": self.pattern, **self.metrics}


def benchmark_sequential(
    name: str,
    load_fn: Callable[[], Iterable[object]],
    num_runs: int = 3,
) -> BenchmarkRecord:
    throughputs: list[float] = []
    total_count = 0
    for _ in range(num_runs):
        t0 = time.perf_counter()
        count = 0
        for _ in load_fn():
            count += 1
        elapsed = time.perf_counter() - t0
        total_count = count
        throughputs.append(count / elapsed if elapsed else 0.0)
    return BenchmarkRecord(
        backend=name,
        pattern="sequential_scan",
        metrics={
            "throughput_mean": statistics.mean(throughputs) if throughputs else 0.0,
            "throughput_std": statistics.pstdev(throughputs) if len(throughputs) > 1 else 0.0,
            "total_samples": total_count,
        },
    )


def benchmark_random_access(
    name: str,
    fetch_fn: Callable[[int], object],
    indices: Iterable[int],
    num_runs: int = 3,
) -> BenchmarkRecord:
    latencies_ms: list[float] = []
    index_list = list(indices)
    for _ in range(num_runs):
        for sample_idx in index_list:
            t0 = time.perf_counter()
            fetch_fn(sample_idx)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
    sorted_latencies = sorted(latencies_ms)
    return BenchmarkRecord(
        backend=name,
        pattern="random_access",
        metrics={
            "latency_p50_ms": _percentile(sorted_latencies, 50),
            "latency_p95_ms": _percentile(sorted_latencies, 95),
            "latency_p99_ms": _percentile(sorted_latencies, 99),
            "num_fetches": len(latencies_ms),
        },
    )


def benchmark_curation_query(
    name: str, query_fn: Callable[[], list[int]], num_runs: int = 3
) -> BenchmarkRecord:
    times_ms: list[float] = []
    result_count = 0
    for _ in range(num_runs):
        t0 = time.perf_counter()
        results = query_fn()
        result_count = len(results)
        times_ms.append((time.perf_counter() - t0) * 1000)
    return BenchmarkRecord(
        backend=name,
        pattern="curation_query",
        metrics={
            "query_time_ms_mean": statistics.mean(times_ms) if times_ms else 0.0,
            "query_time_ms_std": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
            "num_results": result_count,
        },
    )


def benchmark_end_to_end_curation(
    name: str,
    query_fn: Callable[[], list[int]],
    fetch_fn: Callable[[int], object],
    num_runs: int = 3,
) -> BenchmarkRecord:
    totals: list[float] = []
    fetched = 0
    for _ in range(num_runs):
        t0 = time.perf_counter()
        indices = query_fn()
        fetched = len(indices)
        for sample_idx in indices:
            fetch_fn(sample_idx)
        totals.append(time.perf_counter() - t0)
    per_sample_ms = 0.0
    if fetched:
        per_sample_ms = statistics.mean(totals) / fetched * 1000
    return BenchmarkRecord(
        backend=name,
        pattern="e2e_curation",
        metrics={
            "total_sec_mean": statistics.mean(totals) if totals else 0.0,
            "num_fetched": fetched,
            "per_sample_ms": per_sample_ms,
        },
    )


def build_write_record(write_result: StorageWriteResult) -> BenchmarkRecord:
    return BenchmarkRecord(
        backend=write_result.backend,
        pattern="write_throughput",
        metrics={
            "samples_written": write_result.samples_written,
            "elapsed_sec": write_result.elapsed_sec,
            "bytes_written": write_result.bytes_written,
            "throughput": write_result.throughput,
        },
    )


def build_disk_record(name: str, size_bytes: int) -> BenchmarkRecord:
    return BenchmarkRecord(
        backend=name,
        pattern="disk_footprint",
        metrics={"disk_bytes": size_bytes, "disk_mb": round(size_bytes / (1024 * 1024), 3)},
    )


def export_records(records: list[BenchmarkRecord], output_root: Path) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "benchmark_results.json"
    csv_path = output_root / "benchmark_results.csv"
    rows = [record.as_row() for record in records]
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


@dataclass(slots=True)
class BenchmarkRunner:
    backends: dict[str, StorageBackend]

    def run_storage_suite(self, random_indices: list[int] | None = None) -> list[BenchmarkRecord]:
        indices = random_indices or [0, 1, 2, 3, 4]
        records: list[BenchmarkRecord] = []
        for backend in self.backends.values():
            records.append(benchmark_sequential(backend.name, backend.sequential_iter, num_runs=1))
            try:
                records.append(
                    benchmark_random_access(backend.name, backend.fetch, indices, num_runs=1)
                )
            except NotImplementedError:
                pass
            try:
                records.append(
                    benchmark_curation_query(backend.name, backend.curation_query, num_runs=1)
                )
            except NotImplementedError:
                pass
            try:
                records.append(
                    benchmark_end_to_end_curation(
                        backend.name,
                        backend.curation_query,
                        backend.fetch,
                        num_runs=1,
                    )
                )
            except NotImplementedError:
                pass
            records.append(build_disk_record(backend.name, backend.disk_footprint()))
        return records


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * (percentile / 100)
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight
