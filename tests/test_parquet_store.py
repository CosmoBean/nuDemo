from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nudemo.domain.models import (
    CAMERAS,
    RADARS,
    AnnotationRecord,
    UnifiedSample,
)
from nudemo.storage import ParquetBackend
from nudemo.storage.base import array_to_npy_bytes, directory_size, image_to_jpeg_bytes


def make_sample(sample_idx: int, location: str, annotations: list[AnnotationRecord]) -> UnifiedSample:
    cameras = {
        camera: np.full((4, 4, 3), sample_idx + offset, dtype=np.uint8)
        for offset, camera in enumerate(CAMERAS)
    }
    radars = {
        radar: np.full((3, 3), sample_idx + offset, dtype=np.float32)
        for offset, radar in enumerate(RADARS)
    }
    return UnifiedSample(
        token=f"token-{sample_idx}",
        timestamp=1_700_000_000 + sample_idx,
        scene_token=f"scene-{sample_idx // 2}",
        scene_name=f"scene-{sample_idx // 2}",
        location=location,
        cameras=cameras,
        lidar_top=np.full((5, 5), sample_idx, dtype=np.float32),
        radars=radars,
        ego_translation=[float(sample_idx), 0.0, 1.0],
        ego_rotation=[0.0, 0.0, 0.0, 1.0],
        annotations=annotations,
    )


class ParquetBackendTests(unittest.TestCase):
    def test_write_round_trip_and_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            backend = ParquetBackend(Path(tmp_dir) / "dataset")
            samples = [
                make_sample(
                    0,
                    "boston-seaport",
                    [
                        AnnotationRecord(
                            category="human.pedestrian.adult",
                            translation=[0.0, 1.0, 2.0],
                            size=[1.0, 1.5, 2.0],
                            rotation=[0.0, 0.0, 0.0, 1.0],
                            num_lidar_pts=12,
                            num_radar_pts=4,
                        )
                        for _ in range(6)
                    ],
                ),
                make_sample(
                    1,
                    "singapore-onenorth",
                    [
                        AnnotationRecord(
                            category="vehicle.car",
                            translation=[3.0, 4.0, 5.0],
                            size=[1.0, 2.0, 3.0],
                            rotation=[0.0, 0.0, 0.0, 1.0],
                            num_lidar_pts=8,
                            num_radar_pts=2,
                        )
                        for _ in range(3)
                    ],
                ),
            ]

            result = backend.write_samples(samples)

            self.assertEqual(result.backend, "Parquet")
            self.assertEqual(result.samples_written, 2)
            self.assertGreater(result.bytes_written, 0)
            self.assertGreater(result.elapsed_sec, 0)

            seq = list(backend.sequential_iter())
            self.assertEqual([row["idx"] for row in seq], [0, 1])
            self.assertEqual(seq[0]["cam"], image_to_jpeg_bytes(samples[0].cameras["CAM_FRONT"]))
            self.assertEqual(seq[0]["lidar"], array_to_npy_bytes(samples[0].lidar_top))

            fetched = backend.fetch(1)
            self.assertEqual(fetched["cam"], image_to_jpeg_bytes(samples[1].cameras["CAM_FRONT"]))
            self.assertEqual(fetched["lidar"], array_to_npy_bytes(samples[1].lidar_top))

            self.assertEqual(backend.curation_query(), [0])
            self.assertEqual(backend.disk_footprint(), directory_size(Path(tmp_dir) / "dataset"))
            self.assertGreater(backend.disk_footprint(), 0)


if __name__ == "__main__":
    unittest.main()
