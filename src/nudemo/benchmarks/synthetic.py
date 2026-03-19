from __future__ import annotations

import json
from dataclasses import dataclass

ANNOTATION_CATEGORIES = (
    "vehicle.car",
    "vehicle.bus",
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "movable_object.barrier",
)
LOCATIONS = ("boston-seaport", "singapore-onenorth", "singapore-hollandvillage")


@dataclass(slots=True)
class SyntheticSample:
    sample_idx: int
    scene_name: str
    location: str
    categories: list[str]
    num_annotations: int
    camera_bytes: bytes
    lidar_bytes: bytes
    metadata: dict[str, object]

    def payload_bytes(self) -> int:
        return len(self.camera_bytes) + len(self.lidar_bytes)

    def manifest_bytes(self) -> int:
        return len(json.dumps(self.metadata, sort_keys=True).encode("utf-8"))


@dataclass(slots=True)
class SyntheticNuScenesDataset:
    sample_count: int = 404
    scene_count: int = 10

    def build(self) -> list[SyntheticSample]:
        scene_span = max(self.sample_count // self.scene_count, 1)
        samples: list[SyntheticSample] = []
        for sample_idx in range(self.sample_count):
            scene_idx = min(sample_idx // scene_span, self.scene_count - 1)
            location = LOCATIONS[scene_idx % len(LOCATIONS)]
            num_annotations = 2 + (sample_idx % 10)
            categories = [
                ANNOTATION_CATEGORIES[(sample_idx + offset) % len(ANNOTATION_CATEGORIES)]
                for offset in range(num_annotations % 3 + 1)
            ]
            if sample_idx % 4 == 0 and "human.pedestrian.adult" not in categories:
                categories.append("human.pedestrian.adult")
            metadata = {
                "sample_idx": sample_idx,
                "scene_name": f"scene-{scene_idx:04d}",
                "location": location,
                "num_annotations": num_annotations,
                "categories": categories,
            }
            camera_bytes = (f"camera-{sample_idx:04d}".encode()) * 64
            lidar_bytes = (f"lidar-{sample_idx:04d}".encode()) * 96
            samples.append(
                SyntheticSample(
                    sample_idx=sample_idx,
                    scene_name=f"scene-{scene_idx:04d}",
                    location=location,
                    categories=categories,
                    num_annotations=num_annotations,
                    camera_bytes=camera_bytes,
                    lidar_bytes=lidar_bytes,
                    metadata=metadata,
                )
            )
        return samples

    @staticmethod
    def curation_filter():
        def _match(sample: SyntheticSample) -> bool:
            return (
                sample.location == "boston-seaport"
                and any("pedestrian" in category for category in sample.categories)
                and sample.num_annotations > 5
            )

        return _match

