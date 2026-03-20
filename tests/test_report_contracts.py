from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nudemo.benchmarks.export import export_report_bundle, load_report
from nudemo.benchmarks.models import BenchmarkReport
from nudemo.observability import (
    build_run_measurements,
    build_service_measurements,
    build_span_measurements,
)

REPORT_PATH = ROOT / "artifacts" / "reports" / "benchmark_report.json"


class ReportContractTests(unittest.TestCase):
    def test_current_report_shape_and_bundle_exports(self) -> None:
        payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

        self.assertEqual(set(payload), {"suite_name", "dataset", "results", "created_at"})
        self.assertEqual(set(payload["dataset"]), {"samples", "scenes", "provider", "run_id"})

        report = load_report(REPORT_PATH)
        self.assertIsInstance(report, BenchmarkReport)
        self.assertGreater(len(report.results), 0)
        self.assertEqual(report.dataset["provider"], "real")

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path, flat_json_path, csv_path = export_report_bundle(report, Path(tmpdir))

            self.assertTrue(report_path.exists())
            self.assertTrue(flat_json_path.exists())
            self.assertTrue(csv_path.exists())

            flat_rows = json.loads(flat_json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(flat_rows), len(report.results))
            self.assertTrue(all("stage" in row and "backend" in row for row in flat_rows))

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertIn("stage", reader.fieldnames or [])
                self.assertIn("backend", reader.fieldnames or [])
                self.assertIn("pattern", reader.fieldnames or [])

    def test_observability_measurements_match_report_contract(self) -> None:
        report = load_report(REPORT_PATH)
        result = report.results[0]

        run_measurements = build_run_measurements(
            {
                "run_id": report.dataset["run_id"],
                "provider": report.dataset["provider"],
                "status": "ok",
                "simulate": False,
                "elapsed_sec": 12.5,
                "sample_limit": report.dataset["samples"],
                "dataset": report.dataset,
                "summary": {"result_count": len(report.results), "ok_count": len(report.results)},
            }
        )
        self.assertTrue(
            any(attributes["metric_name"] == "dataset_samples" for _, attributes in run_measurements)
        )
        self.assertTrue(
            any(attributes["metric_name"] == "summary_result_count" for _, attributes in run_measurements)
        )

        span_measurements = build_span_measurements(
            str(report.dataset["run_id"]),
            [result.to_dict()],
        )
        self.assertTrue(
            any(attributes["stage"] == result.stage for _, attributes in span_measurements)
        )
        self.assertTrue(
            any(attributes["backend"] == result.backend for _, attributes in span_measurements)
        )

        service_measurements = build_service_measurements(
            str(report.dataset["run_id"]),
            [
                {
                    "service": "kafka",
                    "snapshot_label": "run_start",
                    "cpu_percent": 1.25,
                    "mem_percent": 0.75,
                    "mem_usage_bytes": 1_000_000,
                    "mem_limit_bytes": 2_000_000,
                    "net_input_bytes": 1000,
                    "net_output_bytes": 2000,
                    "block_input_bytes": 3000,
                    "block_output_bytes": 4000,
                    "pids": 12,
                }
            ],
        )
        self.assertTrue(
            any(attributes["service"] == "kafka" for _, attributes in service_measurements)
        )
        self.assertTrue(
            any(
                attributes["metric_name"] == "latest_cpu_percent"
                for _, attributes in service_measurements
            )
        )


if __name__ == "__main__":
    unittest.main()
