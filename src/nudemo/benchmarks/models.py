from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(slots=True)
class BenchmarkResult:
    backend: str
    pattern: str
    metrics: dict[str, float]
    metadata: dict[str, int | float | str] = field(default_factory=dict)
    sample_count: int = 0
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "pattern": self.pattern,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "sample_count": self.sample_count,
            "elapsed_sec": self.elapsed_sec,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> BenchmarkResult:
        return cls(
            backend=str(payload["backend"]),
            pattern=str(payload["pattern"]),
            metrics=dict(payload.get("metrics", {})),
            metadata=dict(payload.get("metadata", {})),
            sample_count=int(payload.get("sample_count", 0)),
            elapsed_sec=float(payload.get("elapsed_sec", 0.0)),
        )


@dataclass(slots=True)
class BenchmarkReport:
    suite_name: str
    dataset: dict[str, int | float | str]
    results: list[BenchmarkResult]
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, object]:
        return {
            "suite_name": self.suite_name,
            "dataset": self.dataset,
            "results": [result.to_dict() for result in self.results],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> BenchmarkReport:
        return cls(
            suite_name=str(payload["suite_name"]),
            dataset=dict(payload.get("dataset", {})),
            results=[BenchmarkResult.from_dict(item) for item in payload.get("results", [])],
            created_at=str(payload.get("created_at", datetime.now(UTC).isoformat())),
        )

    def best_result(
        self, pattern: str, metric: str, *, high_is_better: bool = True
    ) -> BenchmarkResult | None:
        matches = [
            result
            for result in self.results
            if result.pattern == pattern and metric in result.metrics
        ]
        if not matches:
            return None
        chooser = max if high_is_better else min
        return chooser(matches, key=lambda result: result.metrics[metric])
