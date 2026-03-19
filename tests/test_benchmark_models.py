from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nudemo.benchmarks.models import BenchmarkReport, BenchmarkResult


class BenchmarkModelTests(unittest.TestCase):
    def test_result_roundtrip(self) -> None:
        result = BenchmarkResult(
            backend="Lance",
            pattern="random_access",
            metrics={"latency_p50_ms": 1.25, "latency_p95_ms": 3.5},
            metadata={"num_workers": 4},
            sample_count=12,
            elapsed_sec=0.42,
        )

        encoded = json.dumps(result.to_dict())
        decoded = BenchmarkResult.from_dict(json.loads(encoded))

        self.assertEqual(decoded.backend, "Lance")
        self.assertEqual(decoded.pattern, "random_access")
        self.assertEqual(decoded.metadata["num_workers"], 4)
        self.assertAlmostEqual(decoded.metrics["latency_p50_ms"], 1.25)

    def test_report_roundtrip(self) -> None:
        report = BenchmarkReport(
            suite_name="nuDemo",
            dataset={"samples": 404, "scenes": 10},
            results=[
                BenchmarkResult(
                    backend="WebDataset",
                    pattern="dataloader",
                    metrics={"throughput_samples_per_sec": 512.0},
                    metadata={"batch_size": 4, "num_workers": 2},
                ),
                BenchmarkResult(
                    backend="Lance",
                    pattern="random_access",
                    metrics={"latency_p50_ms": 1.75},
                    metadata={"num_workers": 0},
                )
            ],
        )

        encoded = json.dumps(report.to_dict())
        decoded = BenchmarkReport.from_dict(json.loads(encoded))

        self.assertEqual(decoded.dataset["samples"], 404)
        self.assertEqual(decoded.results[0].backend, "WebDataset")
        self.assertEqual(decoded.results[0].metadata["num_workers"], 2)
        best_latency = decoded.best_result("random_access", "latency_p50_ms", high_is_better=False)
        self.assertIsNotNone(best_latency)
        self.assertEqual(best_latency.backend, "Lance")


if __name__ == "__main__":
    unittest.main()
