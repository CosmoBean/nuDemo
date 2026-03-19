from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

RADARS = (
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
)


@dataclass(slots=True)
class SyntheticShapeConfig:
    camera_height: int
    camera_width: int
    lidar_points: int
    radar_points: int


@dataclass(slots=True)
class AnnotationRecord:
    category: str
    translation: list[float]
    size: list[float]
    rotation: list[float]
    num_lidar_pts: int
    num_radar_pts: int


@dataclass(slots=True)
class SampleBlobRefs:
    camera_paths: dict[str, str]
    sensor_paths: dict[str, str]

    def flattened(self) -> dict[str, str]:
        result = dict(self.camera_paths)
        result.update(self.sensor_paths)
        return result


@dataclass(slots=True)
class SampleMetadata:
    sample_idx: int
    token: str
    timestamp: int
    scene_token: str
    scene_name: str
    location: str
    ego_translation: list[float]
    ego_rotation: list[float]
    num_annotations: int
    annotation_categories: list[str]
    num_lidar_points: int
    blob_refs: dict[str, str] = field(default_factory=dict)

    def as_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass(slots=True)
class UnifiedSample:
    token: str
    timestamp: int
    scene_token: str
    scene_name: str
    location: str
    cameras: dict[str, np.ndarray]
    lidar_top: np.ndarray
    radars: dict[str, np.ndarray]
    ego_translation: list[float]
    ego_rotation: list[float]
    annotations: list[AnnotationRecord]

    def validate(self) -> None:
        missing_cameras = set(CAMERAS) - set(self.cameras)
        missing_radars = set(RADARS) - set(self.radars)
        if missing_cameras or missing_radars:
            raise ValueError(
                f"missing sensors cameras={sorted(missing_cameras)} radars={sorted(missing_radars)}"
            )

    def blob_refs(self, sample_idx: int) -> SampleBlobRefs:
        camera_paths = {camera: f"samples/{sample_idx:04d}/{camera}.jpg" for camera in CAMERAS}
        sensor_paths = {"LIDAR_TOP": f"samples/{sample_idx:04d}/LIDAR_TOP.npy"}
        sensor_paths.update(
            {radar: f"samples/{sample_idx:04d}/{radar}.npy" for radar in RADARS}
        )
        return SampleBlobRefs(camera_paths=camera_paths, sensor_paths=sensor_paths)

    def metadata(self, sample_idx: int) -> SampleMetadata:
        return SampleMetadata(
            sample_idx=sample_idx,
            token=self.token,
            timestamp=self.timestamp,
            scene_token=self.scene_token,
            scene_name=self.scene_name,
            location=self.location,
            ego_translation=self.ego_translation,
            ego_rotation=self.ego_rotation,
            num_annotations=len(self.annotations),
            annotation_categories=[annotation.category for annotation in self.annotations],
            num_lidar_points=int(self.lidar_top.shape[0]),
            blob_refs=self.blob_refs(sample_idx).flattened(),
        )

    def metadata_payload(self, sample_idx: int) -> bytes:
        return self.metadata(sample_idx).as_json().encode("utf-8")

    def to_dict(self, sample_idx: int) -> dict[str, Any]:
        return {
            "sample_idx": sample_idx,
            "token": self.token,
            "timestamp": self.timestamp,
            "scene_token": self.scene_token,
            "scene_name": self.scene_name,
            "location": self.location,
            "ego_translation": self.ego_translation,
            "ego_rotation": self.ego_rotation,
            "num_annotations": len(self.annotations),
            "num_lidar_points": int(self.lidar_top.shape[0]),
            "annotation_categories": [annotation.category for annotation in self.annotations],
        }

