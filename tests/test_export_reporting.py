from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nudemo.benchmarks.backends import (
    LanceBackend,
    MinioPostgresBackend,
    ParquetBackend,
    RedisBackend,
    WebDatasetBackend,
)
from nudemo.benchmarks.export import export_report, load_report
from nudemo.benchmarks.orchestrator import BenchmarkOrchestrator
from nudemo.benchmarks.synthetic import SyntheticNuScenesDataset
from nudemo.reporting.dashboard import (
    DashboardApp,
    build_dashboard_html,
    build_recommendation_summary,
)


class ExportReportingTests(unittest.TestCase):
    def test_export_roundtrip_and_dashboard_render(self) -> None:
        dataset = SyntheticNuScenesDataset(sample_count=24, scene_count=4).build()
        backends = [
            MinioPostgresBackend(),
            LanceBackend(),
            ParquetBackend(),
            RedisBackend(),
            WebDatasetBackend(),
        ]
        report = BenchmarkOrchestrator(
            dataset,
            backends,
            suite_name="nuDemo benchmark suite",
        ).run(num_runs=1, random_sample_count=5, batch_size=4, num_workers=(0, 2))

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = export_report(report, Path(tmpdir) / "benchmark_results.json")
            loaded = load_report(json_path)
            dashboard = DashboardApp.from_json(json_path)
            html = dashboard.render()

        self.assertEqual(loaded.suite_name, "nuDemo benchmark suite")
        self.assertIn("Recommendation Summary", html)
        self.assertIn("Throughput vs. num_workers", html)
        self.assertIn("WebDataset", html)
        self.assertIn("Lance", html)
        self.assertIn("Parquet", html)

        recommendations = build_recommendation_summary(loaded)
        self.assertIn("training", recommendations)
        self.assertIn("random_access", recommendations)

        rendered = build_dashboard_html(loaded)
        self.assertIn("Access Pattern Matrix", rendered)
        self.assertIn("Results", rendered)


if __name__ == "__main__":
    unittest.main()
