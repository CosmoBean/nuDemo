from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from nudemo.config import AppConfig
from nudemo.domain.models import (
    CAMERAS,
    RADARS,
    AnnotationRecord,
    SyntheticShapeConfig,
    UnifiedSample,
)

ANNOTATION_CATEGORIES = (
    "vehicle.car",
    "vehicle.bus",
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "movable_object.barrier",
    "vehicle.bicycle",
)
LOCATIONS = ("boston-seaport", "singapore-onenorth", "singapore-hollandvillage")


class SampleProvider(Protocol):
    def iter_samples(self, limit: int | None = None) -> Iterator[UnifiedSample]:
        ...


@dataclass(slots=True)
class SyntheticNuScenesProvider:
    shape: SyntheticShapeConfig
    scene_count: int
    samples_per_scene: int
    seed: int = 7

    def iter_samples(self, limit: int | None = None) -> Iterator[UnifiedSample]:
        total = self.scene_count * self.samples_per_scene
        upper = total if limit is None else min(total, limit)
        for sample_idx in range(upper):
            yield self._make_sample(sample_idx)

    def _make_sample(self, sample_idx: int) -> UnifiedSample:
        scene_idx = sample_idx // self.samples_per_scene
        rng = np.random.default_rng(self.seed + sample_idx)

        cameras = {
            camera: rng.integers(
                low=0,
                high=255,
                size=(self.shape.camera_height, self.shape.camera_width, 3),
                dtype=np.uint8,
            )
            for camera in CAMERAS
        }
        lidar = rng.normal(size=(self.shape.lidar_points, 5)).astype(np.float32)
        radars = {
            radar: rng.normal(size=(self.shape.radar_points, 18)).astype(np.float32)
            for radar in RADARS
        }

        annotation_count = int(rng.integers(low=2, high=12))
        annotations = [
            AnnotationRecord(
                category=str(rng.choice(ANNOTATION_CATEGORIES)),
                translation=rng.normal(size=3).round(3).tolist(),
                size=np.abs(rng.normal(loc=[2.0, 4.0, 1.5], scale=0.5, size=3)).round(3).tolist(),
                rotation=rng.normal(size=4).round(6).tolist(),
                num_lidar_pts=int(rng.integers(low=3, high=600)),
                num_radar_pts=int(rng.integers(low=0, high=25)),
            )
            for _ in range(annotation_count)
        ]

        timestamp = 1_532_402_928_669_565 + sample_idx * 500_000
        sample = UnifiedSample(
            token=f"sample-{sample_idx:06d}",
            timestamp=timestamp,
            scene_token=f"scene-token-{scene_idx:04d}",
            scene_name=f"scene-{scene_idx:04d}",
            location=LOCATIONS[scene_idx % len(LOCATIONS)],
            cameras=cameras,
            lidar_top=lidar,
            radars=radars,
            ego_translation=rng.normal(size=3).round(3).tolist(),
            ego_rotation=rng.normal(size=4).round(6).tolist(),
            annotations=annotations,
        )
        sample.validate()
        return sample


@dataclass(slots=True)
class NuScenesProvider:
    dataset_root: Path
    version: str = "v1.0-mini"

    def iter_samples(self, limit: int | None = None) -> Iterator[UnifiedSample]:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
        from PIL import Image

        nusc = NuScenes(version=self.version, dataroot=str(self.dataset_root), verbose=False)
        records = nusc.sample[:limit] if limit else nusc.sample

        for record in records:
            scene = nusc.get("scene", record["scene_token"])
            log = nusc.get("log", scene["log_token"])
            cameras = {}
            for camera in CAMERAS:
                sample_data = nusc.get("sample_data", record["data"][camera])
                image_path = self.dataset_root / sample_data["filename"]
                cameras[camera] = np.asarray(Image.open(image_path))

            lidar_sd = nusc.get("sample_data", record["data"]["LIDAR_TOP"])
            lidar_path = self.dataset_root / lidar_sd["filename"]
            lidar = LidarPointCloud.from_file(str(lidar_path)).points.T.astype(np.float32)

            radars = {}
            for radar in RADARS:
                radar_sd = nusc.get("sample_data", record["data"][radar])
                radar_path = self.dataset_root / radar_sd["filename"]
                radars[radar] = (
                    RadarPointCloud.from_file(str(radar_path)).points.T.astype(np.float32)
                )

            ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
            annotations = []
            for annotation_token in record["anns"]:
                annotation = nusc.get("sample_annotation", annotation_token)
                annotations.append(
                    AnnotationRecord(
                        category=annotation["category_name"],
                        translation=annotation["translation"],
                        size=annotation["size"],
                        rotation=annotation["rotation"],
                        num_lidar_pts=annotation["num_lidar_pts"],
                        num_radar_pts=annotation["num_radar_pts"],
                    )
                )

            sample = UnifiedSample(
                token=record["token"],
                timestamp=record["timestamp"],
                scene_token=record["scene_token"],
                scene_name=scene["name"],
                location=log["location"],
                cameras=cameras,
                lidar_top=lidar,
                radars=radars,
                ego_translation=ego_pose["translation"],
                ego_rotation=ego_pose["rotation"],
                annotations=annotations,
            )
            sample.validate()
            yield sample


def resolve_provider(config: AppConfig, provider_name: str = "auto") -> SampleProvider:
    if provider_name not in {"auto", "real", "synthetic"}:
        raise ValueError(f"unsupported provider {provider_name}")

    metadata_dir = config.runtime.dataset_root / config.pipeline.dataset_version
    real_available = metadata_dir.exists()
    if provider_name == "real" and not real_available:
        raise FileNotFoundError(f"nuScenes metadata directory not found: {metadata_dir}")

    if provider_name == "real" or (provider_name == "auto" and real_available):
        return NuScenesProvider(
            dataset_root=config.runtime.dataset_root,
            version=config.pipeline.dataset_version,
        )

    if provider_name == "synthetic":
        return SyntheticNuScenesProvider(
            shape=SyntheticShapeConfig(
                camera_height=config.pipeline.camera_height,
                camera_width=config.pipeline.camera_width,
                lidar_points=config.pipeline.lidar_points,
                radar_points=config.pipeline.radar_points,
            ),
            scene_count=config.pipeline.synthetic_scene_count,
            samples_per_scene=config.pipeline.synthetic_samples_per_scene,
            seed=config.pipeline.synthetic_seed,
        )

    if not config.pipeline.synthetic_enabled:
        raise RuntimeError("synthetic provider disabled and real dataset unavailable")

    shape = SyntheticShapeConfig(
        camera_height=config.pipeline.camera_height,
        camera_width=config.pipeline.camera_width,
        lidar_points=config.pipeline.lidar_points,
        radar_points=config.pipeline.radar_points,
    )
    return SyntheticNuScenesProvider(
        shape=shape,
        scene_count=config.pipeline.synthetic_scene_count,
        samples_per_scene=config.pipeline.synthetic_samples_per_scene,
        seed=config.pipeline.synthetic_seed,
    )
