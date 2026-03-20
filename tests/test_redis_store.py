import warnings

import numpy as np

from nudemo.config import RedisSettings
from nudemo.domain.models import CAMERAS, RADARS, AnnotationRecord, UnifiedSample
from nudemo.storage.redis_store import RedisBackend


def _sample_with_empty_ranges() -> UnifiedSample:
    cameras = {
        camera: np.zeros((16, 16, 3), dtype=np.uint8)
        for camera in CAMERAS
    }
    radars = {
        radar: np.zeros((0, 18), dtype=np.float32)
        for radar in RADARS
    }
    return UnifiedSample(
        token="sample-empty-radar",
        timestamp=1,
        scene_token="scene-token-0000",
        scene_name="scene-0000",
        location="boston-seaport",
        cameras=cameras,
        lidar_top=np.zeros((0, 5), dtype=np.float32),
        radars=radars,
        ego_translation=[0.0, 0.0, 0.0],
        ego_rotation=[1.0, 0.0, 0.0, 0.0],
        annotations=[
            AnnotationRecord(
                category="human.pedestrian.adult",
                translation=[0.0, 0.0, 0.0],
                size=[1.0, 1.0, 1.0],
                rotation=[1.0, 0.0, 0.0, 0.0],
                num_lidar_pts=0,
                num_radar_pts=0,
            )
        ],
    )


def test_derive_embedding_handles_empty_radar_without_warnings() -> None:
    backend = RedisBackend(RedisSettings(host="localhost", port=6379, db=0))
    sample = _sample_with_empty_ranges()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", RuntimeWarning)
        embedding = backend._derive_embedding(sample)

    assert not caught
    assert embedding.shape == (512,)
    assert np.isfinite(embedding).all()
