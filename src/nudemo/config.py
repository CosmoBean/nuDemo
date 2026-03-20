from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


@dataclass(slots=True)
class RuntimePaths:
    dataset_root: Path
    artifacts_root: Path
    formats_root: Path
    reports_root: Path

    def ensure(self) -> None:
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.formats_root.mkdir(parents=True, exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class PipelineSettings:
    dataset_version: str
    sample_limit: int
    synthetic_enabled: bool
    synthetic_scene_count: int
    synthetic_samples_per_scene: int
    synthetic_seed: int
    camera_height: int
    camera_width: int
    lidar_points: int
    radar_points: int


@dataclass(slots=True)
class KafkaSettings:
    bootstrap_servers: str
    raw_topic: str
    refined_topic: str


@dataclass(slots=True)
class MinioSettings:
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    secure: bool


@dataclass(slots=True)
class PostgresSettings:
    host: str
    port: int
    database: str
    user: str
    password: str

    @property
    def dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.database} "
            f"user={self.user} password={self.password}"
        )


@dataclass(slots=True)
class RedisSettings:
    host: str
    port: int
    db: int

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


@dataclass(slots=True)
class LanceSettings:
    dataset_path: Path


@dataclass(slots=True)
class ParquetSettings:
    dataset_path: Path


@dataclass(slots=True)
class WebDatasetSettings:
    shard_pattern: str
    maxcount: int


@dataclass(slots=True)
class ServiceSettings:
    kafka: KafkaSettings
    minio: MinioSettings
    postgres: PostgresSettings
    redis: RedisSettings


@dataclass(slots=True)
class StorageSettings:
    lance: LanceSettings
    parquet: ParquetSettings
    webdataset: WebDatasetSettings


@dataclass(slots=True)
class AppConfig:
    runtime: RuntimePaths
    pipeline: PipelineSettings
    services: ServiceSettings
    storage: StorageSettings

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> AppConfig:
        repo_root = Path(__file__).resolve().parents[2]
        defaults_path = Path(config_path) if config_path else repo_root / "config/defaults.toml"
        with defaults_path.open("rb") as handle:
            raw = tomllib.load(handle)

        runtime_raw = raw["runtime"]
        base_dir = repo_root
        
        def env_path(name: str, default: str) -> Path:
            return _resolve_path(base_dir, os.getenv(name, default))

        runtime = RuntimePaths(
            dataset_root=env_path("NUDEMO_DATASET_ROOT", runtime_raw["dataset_root"]),
            artifacts_root=env_path("NUDEMO_ARTIFACTS_ROOT", runtime_raw["artifacts_root"]),
            formats_root=env_path("NUDEMO_FORMATS_ROOT", runtime_raw["formats_root"]),
            reports_root=env_path("NUDEMO_REPORTS_ROOT", runtime_raw["reports_root"]),
        )

        pipeline_raw = raw["pipeline"]
        pipeline = PipelineSettings(
            dataset_version=os.getenv("NUDEMO_DATASET_VERSION", pipeline_raw["dataset_version"]),
            sample_limit=int(os.getenv("NUDEMO_SAMPLE_LIMIT", pipeline_raw["sample_limit"])),
            synthetic_enabled=os.getenv(
                "NUDEMO_SYNTHETIC_ENABLED", str(pipeline_raw["synthetic_enabled"])
            ).lower()
            in {"1", "true", "yes"},
            synthetic_scene_count=int(
                os.getenv("NUDEMO_SYNTHETIC_SCENE_COUNT", pipeline_raw["synthetic_scene_count"])
            ),
            synthetic_samples_per_scene=int(
                os.getenv(
                    "NUDEMO_SYNTHETIC_SAMPLES_PER_SCENE",
                    pipeline_raw["synthetic_samples_per_scene"],
                )
            ),
            synthetic_seed=int(os.getenv("NUDEMO_SYNTHETIC_SEED", pipeline_raw["synthetic_seed"])),
            camera_height=int(os.getenv("NUDEMO_CAMERA_HEIGHT", pipeline_raw["camera_height"])),
            camera_width=int(os.getenv("NUDEMO_CAMERA_WIDTH", pipeline_raw["camera_width"])),
            lidar_points=int(os.getenv("NUDEMO_LIDAR_POINTS", pipeline_raw["lidar_points"])),
            radar_points=int(os.getenv("NUDEMO_RADAR_POINTS", pipeline_raw["radar_points"])),
        )

        services_raw = raw["services"]
        services = ServiceSettings(
            kafka=KafkaSettings(**services_raw["kafka"]),
            minio=MinioSettings(**services_raw["minio"]),
            postgres=PostgresSettings(**services_raw["postgres"]),
            redis=RedisSettings(**services_raw["redis"]),
        )
        storage_raw = raw["storage"]
        storage = StorageSettings(
            lance=LanceSettings(
                dataset_path=_resolve_path(base_dir, storage_raw["lance"]["dataset_path"])
            ),
            parquet=ParquetSettings(
                dataset_path=_resolve_path(base_dir, storage_raw["parquet"]["dataset_path"])
            ),
            webdataset=WebDatasetSettings(**storage_raw["webdataset"]),
        )
        runtime.ensure()
        return cls(runtime=runtime, pipeline=pipeline, services=services, storage=storage)
