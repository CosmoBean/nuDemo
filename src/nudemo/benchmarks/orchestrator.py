from __future__ import annotations

from dataclasses import dataclass

from nudemo.benchmarks.models import BenchmarkReport, BenchmarkResult
from nudemo.benchmarks.synthetic import SyntheticSample


@dataclass(slots=True)
class BenchmarkOrchestrator:
    dataset: list[SyntheticSample]
    backends: list[object]
    suite_name: str = "nuDemo benchmark suite"

    def run(
        self,
        *,
        num_runs: int = 1,
        random_sample_count: int = 10,
        batch_size: int = 4,
        num_workers: tuple[int, ...] = (0, 2, 4),
    ) -> BenchmarkReport:
        for backend in self.backends:
            backend.load(self.dataset)

        results: list[BenchmarkResult] = []
        query_predicate = self._default_curation_filter()
        fetch_count = min(len(self.dataset), random_sample_count)

        for backend in self.backends:
            sample_count = len(self.dataset)
            results.append(
                BenchmarkResult(
                    stage="storage",
                    backend=backend.name,
                    pattern="write_throughput",
                    metrics={
                        "throughput_samples_per_sec": backend.profile.write_samples_per_sec,
                    },
                    sample_count=sample_count,
                    elapsed_sec=backend.write_elapsed(sample_count),
                )
            )
            results.append(
                BenchmarkResult(
                    stage="storage",
                    backend=backend.name,
                    pattern="disk_footprint",
                    metrics={"disk_bytes": float(backend.disk_bytes())},
                    sample_count=sample_count,
                )
            )
            if backend.profile.sequential_samples_per_sec:
                results.append(
                    BenchmarkResult(
                        stage="training",
                        backend=backend.name,
                        pattern="sequential_scan",
                        metrics={
                            "throughput_samples_per_sec": backend.profile.sequential_samples_per_sec
                        },
                        sample_count=sample_count,
                        elapsed_sec=sample_count / backend.profile.sequential_samples_per_sec,
                    )
                )
            if backend.supports_random_access:
                results.append(
                    BenchmarkResult(
                        stage="evaluation",
                        backend=backend.name,
                        pattern="random_access",
                        metrics={
                            "latency_p50_ms": backend.profile.random_access_ms,
                            "latency_p95_ms": backend.profile.random_access_ms * 1.4,
                        },
                        metadata={"num_fetches": fetch_count * num_runs},
                        sample_count=fetch_count * num_runs,
                    )
                )
            if backend.supports_query:
                matches = backend.query_indices(query_predicate)
                results.append(
                    BenchmarkResult(
                        stage="curation",
                        backend=backend.name,
                        pattern="curation_query",
                        metrics={"query_time_ms": backend.profile.query_ms},
                        metadata={"num_results": len(matches)},
                        sample_count=len(matches),
                    )
                )
                if backend.supports_payload_fetch:
                    results.append(
                        BenchmarkResult(
                            stage="curation",
                            backend=backend.name,
                            pattern="e2e_curation",
                            metrics={"per_sample_ms": backend.profile.e2e_ms},
                            metadata={"num_results": len(matches)},
                            sample_count=len(matches),
                            elapsed_sec=(backend.profile.e2e_ms / 1000.0) * len(matches),
                        )
                    )
            if backend.profile.dataloader_base:
                for worker_count in num_workers:
                    results.append(
                        BenchmarkResult(
                            stage="training",
                            backend=backend.name,
                            pattern="dataloader",
                            metrics={
                                "throughput_samples_per_sec": backend.dataloader_throughput(
                                    worker_count, batch_size
                                )
                            },
                            metadata={"batch_size": batch_size, "num_workers": worker_count},
                            sample_count=sample_count,
                        )
                    )

        dataset_summary = {
            "samples": len(self.dataset),
            "scenes": len({sample.scene_name for sample in self.dataset}),
        }
        return BenchmarkReport(suite_name=self.suite_name, dataset=dataset_summary, results=results)

    @staticmethod
    def _default_curation_filter():
        return lambda sample: (
            sample.location == "boston-seaport"
            and any("pedestrian" in category for category in sample.categories)
            and sample.num_annotations > 5
        )
