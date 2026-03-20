from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nudemo.benchmarks.models import BenchmarkReport, BenchmarkResult
from nudemo.explorer.app import BenchmarkReportStore


class BenchmarkReportStoreTests(unittest.TestCase):
    def test_fetch_summary_builds_storage_comparison(self) -> None:
        report = BenchmarkReport(
            suite_name="nuDemo live benchmark suite",
            dataset={"samples": 32, "scenes": 4, "provider": "real", "run_id": "run-1"},
            results=[
                BenchmarkResult(
                    stage="storage",
                    backend="Parquet",
                    pattern="write_throughput",
                    metrics={"throughput_samples_per_sec": 18.4},
                ),
                BenchmarkResult(
                    stage="training",
                    backend="Parquet",
                    pattern="sequential_scan",
                    metrics={"throughput_mean": 220.0},
                ),
                BenchmarkResult(
                    stage="evaluation",
                    backend="Parquet",
                    pattern="random_access",
                    metrics={"latency_p50_ms": 8.3},
                ),
                BenchmarkResult(
                    stage="storage",
                    backend="Parquet",
                    pattern="disk_footprint",
                    metrics={"disk_mb": 28.5},
                ),
                BenchmarkResult(
                    stage="curation",
                    backend="Parquet",
                    pattern="curation_query",
                    metrics={"query_time_ms_mean": 4.2},
                ),
                BenchmarkResult(
                    stage="storage",
                    backend="Redis",
                    pattern="write_throughput",
                    metrics={"throughput_samples_per_sec": 41.9},
                ),
                BenchmarkResult(
                    stage="training",
                    backend="Redis",
                    pattern="sequential_scan",
                    metrics={"throughput_mean": 810.0},
                ),
                BenchmarkResult(
                    stage="evaluation",
                    backend="Redis",
                    pattern="random_access",
                    metrics={"latency_p50_ms": 0.7},
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "benchmark_report.json"
            report_path.write_text(json.dumps(report.to_dict()), encoding="utf-8")

            payload = BenchmarkReportStore(Path(tmp_dir)).fetch_summary()

        self.assertEqual(payload["suite_name"], "nuDemo latest backend comparison")
        self.assertEqual(payload["dataset"]["provider"], "real")
        self.assertEqual(payload["recommendations"]["storage"], "Redis")
        self.assertEqual(payload["recommendations"]["random_access"], "Redis")

        formats = {row["backend"]: row for row in payload["storage_formats"]}
        self.assertIn("Parquet", formats)
        self.assertAlmostEqual(formats["Parquet"]["write_samples_per_sec"], 18.4)
        self.assertAlmostEqual(formats["Parquet"]["disk_mb"], 28.5)
        self.assertEqual(formats["Parquet"]["status"], "ok")
        self.assertAlmostEqual(formats["Redis"]["random_access_p50_ms"], 0.7)
        self.assertEqual(payload["comparison_note"], "All backend rows come from the same sample and scene scope.")

    def test_fetch_summary_uses_latest_report_per_backend(self) -> None:
        minio_report = BenchmarkReport(
            suite_name="nuDemo batched ingest study - minio-postgres",
            dataset={"samples": 34149, "scenes": 850, "provider": "real", "run_id": "run-big"},
            created_at="2026-03-20T15:49:11+00:00",
            results=[
                BenchmarkResult(
                    stage="storage",
                    backend="MinIO+PostgreSQL",
                    pattern="write_throughput",
                    metrics={"throughput_samples_per_sec": 8.12},
                ),
                BenchmarkResult(
                    stage="training",
                    backend="MinIO+PostgreSQL",
                    pattern="sequential_scan",
                    metrics={"throughput_mean": 258.52},
                ),
                BenchmarkResult(
                    stage="evaluation",
                    backend="MinIO+PostgreSQL",
                    pattern="random_access",
                    metrics={"latency_p50_ms": 12.69},
                ),
            ],
        )
        redis_report = BenchmarkReport(
            suite_name="nuDemo batched ingest study - redis",
            dataset={"samples": 100, "scenes": 3, "provider": "real", "run_id": "run-redis"},
            created_at="2026-03-20T16:09:44+00:00",
            results=[
                BenchmarkResult(
                    stage="storage",
                    backend="Redis",
                    pattern="write_throughput",
                    metrics={"throughput_samples_per_sec": 183.4},
                ),
                BenchmarkResult(
                    stage="training",
                    backend="Redis",
                    pattern="sequential_scan",
                    metrics={"throughput_mean": 3796.7},
                ),
                BenchmarkResult(
                    stage="evaluation",
                    backend="Redis",
                    pattern="random_access",
                    metrics={"latency_p50_ms": 0.75},
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            big_dir = root / "overnight_runs" / "20260320_091446" / "minio-postgres"
            small_dir = root / "overnight_runs" / "20260320_120903" / "redis"
            big_dir.mkdir(parents=True)
            small_dir.mkdir(parents=True)
            (big_dir / "benchmark_report.json").write_text(
                json.dumps(minio_report.to_dict()),
                encoding="utf-8",
            )
            (small_dir / "benchmark_report.json").write_text(
                json.dumps(redis_report.to_dict()),
                encoding="utf-8",
            )

            payload = BenchmarkReportStore(root).fetch_summary()

        formats = {row["backend"]: row for row in payload["storage_formats"]}
        self.assertEqual(payload["dataset"]["samples"], "mixed")
        self.assertIn("Rows come from the latest completed run for each backend.", payload["comparison_note"])
        self.assertEqual(formats["MinIO+PostgreSQL"]["samples"], 34149)
        self.assertEqual(formats["MinIO+PostgreSQL"]["scenes"], 850)
        self.assertEqual(formats["Redis"]["samples"], 100)
        self.assertEqual(formats["Redis"]["scenes"], 3)
        self.assertAlmostEqual(formats["Redis"]["write_samples_per_sec"], 183.4)


if __name__ == "__main__":
    unittest.main()
