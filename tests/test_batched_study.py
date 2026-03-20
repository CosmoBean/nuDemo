from nudemo.config import AppConfig
from nudemo.domain.models import SyntheticShapeConfig
from nudemo.extraction.providers import SyntheticNuScenesProvider
from nudemo.studies.batched_ingest import build_random_indices, iter_sample_batches


def test_build_random_indices_spreads_samples_evenly() -> None:
    indices = build_random_indices(100, 5)

    assert indices == [0, 20, 40, 60, 99]


def test_build_random_indices_handles_small_sample_sets() -> None:
    indices = build_random_indices(3, 10)

    assert indices == [0, 1, 2]


def test_synthetic_provider_scene_limit_caps_to_whole_scenes() -> None:
    provider = SyntheticNuScenesProvider(
        shape=SyntheticShapeConfig(
            camera_height=8,
            camera_width=8,
            lidar_points=4,
            radar_points=2,
        ),
        scene_count=5,
        samples_per_scene=4,
        seed=7,
    )

    samples = list(provider.iter_samples(scene_limit=2))

    assert len(samples) == 8
    assert {sample.scene_name for sample in samples} == {"scene-0000", "scene-0001"}


def test_iter_sample_batches_respects_scene_limit() -> None:
    config = AppConfig.load()
    config.pipeline.synthetic_scene_count = 5
    config.pipeline.synthetic_samples_per_scene = 4

    batches = list(
        iter_sample_batches(
            config=config,
            provider_name="synthetic",
            limit=None,
            scene_limit=2,
            batch_size=3,
        )
    )

    flattened = [sample for _, batch in batches for sample in batch]

    assert [start_idx for start_idx, _ in batches] == [0, 3, 6]
    assert len(flattened) == 8
    assert {sample.scene_name for sample in flattened} == {"scene-0000", "scene-0001"}
