from nudemo.config import AppConfig
from nudemo.domain.models import CAMERAS, RADARS
from nudemo.extraction.providers import resolve_provider


def _synthetic_config(monkeypatch):
    monkeypatch.setenv("NUDEMO_SYNTHETIC_ENABLED", "true")
    return AppConfig.load()


def test_synthetic_provider_matches_schema(monkeypatch):
    config = _synthetic_config(monkeypatch)
    provider = resolve_provider(config, "synthetic")
    sample = next(provider.iter_samples(limit=1))

    assert set(sample.cameras) == set(CAMERAS)
    assert set(sample.radars) == set(RADARS)
    assert sample.lidar_top.shape[1] == 5
    assert sample.metadata(0).num_annotations == len(sample.annotations)


def test_blob_refs_cover_all_modalities(monkeypatch):
    config = _synthetic_config(monkeypatch)
    provider = resolve_provider(config, "synthetic")
    sample = next(provider.iter_samples(limit=1))
    blob_refs = sample.blob_refs(3).flattened()

    assert blob_refs["CAM_FRONT"] == "samples/0003/CAM_FRONT.jpg"
    assert blob_refs["LIDAR_TOP"] == "samples/0003/LIDAR_TOP.npy"
    assert len(blob_refs) == len(CAMERAS) + len(RADARS) + 1
