from __future__ import annotations

import math
from dataclasses import dataclass, field

from nudemo.benchmarks.synthetic import SyntheticSample


@dataclass(slots=True)
class BackendProfile:
    sequential_samples_per_sec: float
    random_access_ms: float
    query_ms: float
    e2e_ms: float
    write_samples_per_sec: float
    dataloader_base: float
    disk_multiplier: float


@dataclass(slots=True)
class BaseBackend:
    name: str
    profile: BackendProfile
    supports_random_access: bool = True
    supports_query: bool = True
    supports_payload_fetch: bool = True
    _samples: list[SyntheticSample] = field(default_factory=list)

    def load(self, dataset: list[SyntheticSample]) -> None:
        self._samples = list(dataset)

    def sequential_scan(self):
        yield from self._samples

    def query_indices(self, predicate):
        if not self.supports_query:
            raise NotImplementedError(f"{self.name} does not support metadata queries")
        return [sample.sample_idx for sample in self._samples if predicate(sample)]

    def fetch_sample(self, sample_idx: int, payload: bool = True) -> SyntheticSample:
        if payload and not self.supports_payload_fetch:
            raise NotImplementedError(f"{self.name} does not expose payload fetches")
        if not self.supports_random_access:
            raise NotImplementedError(f"{self.name} is sequential only")
        sample = self._samples[sample_idx]
        if payload:
            return sample
        return SyntheticSample(
            sample_idx=sample.sample_idx,
            scene_name=sample.scene_name,
            location=sample.location,
            categories=list(sample.categories),
            num_annotations=sample.num_annotations,
            camera_bytes=b"",
            lidar_bytes=b"",
            metadata=dict(sample.metadata),
        )

    def write_elapsed(self, sample_count: int) -> float:
        return sample_count / self.profile.write_samples_per_sec

    def dataloader_throughput(self, num_workers: int, batch_size: int) -> float:
        scaling = 1.0 + min(num_workers, 4) * 0.18
        return self.profile.dataloader_base * scaling * max(batch_size / 4, 0.5)

    def disk_bytes(self) -> int:
        if not self._samples:
            return 0
        total_payload = sum(
            sample.payload_bytes() + sample.manifest_bytes() for sample in self._samples
        )
        return math.ceil(total_payload * self.profile.disk_multiplier)


class MinioPostgresBackend(BaseBackend):
    def __init__(self) -> None:
        super().__init__(
            name="MinIO+PostgreSQL",
            profile=BackendProfile(
                sequential_samples_per_sec=185.0,
                random_access_ms=12.5,
                query_ms=3.2,
                e2e_ms=18.0,
                write_samples_per_sec=145.0,
                dataloader_base=190.0,
                disk_multiplier=1.16,
            ),
        )


class LanceBackend(BaseBackend):
    def __init__(self) -> None:
        super().__init__(
            name="Lance",
            profile=BackendProfile(
                sequential_samples_per_sec=245.0,
                random_access_ms=2.1,
                query_ms=5.4,
                e2e_ms=5.0,
                write_samples_per_sec=225.0,
                dataloader_base=255.0,
                disk_multiplier=0.98,
            ),
        )


class RedisBackend(BaseBackend):
    def __init__(self) -> None:
        super().__init__(
            name="Redis",
            profile=BackendProfile(
                sequential_samples_per_sec=0.0,
                random_access_ms=0.8,
                query_ms=2.6,
                e2e_ms=0.0,
                write_samples_per_sec=430.0,
                dataloader_base=0.0,
                disk_multiplier=0.21,
            ),
            supports_payload_fetch=False,
        )


class WebDatasetBackend(BaseBackend):
    def __init__(self) -> None:
        super().__init__(
            name="WebDataset",
            profile=BackendProfile(
                sequential_samples_per_sec=330.0,
                random_access_ms=0.0,
                query_ms=0.0,
                e2e_ms=0.0,
                write_samples_per_sec=285.0,
                dataloader_base=340.0,
                disk_multiplier=0.9,
            ),
            supports_random_access=False,
            supports_query=False,
        )
