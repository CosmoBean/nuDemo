import json
import struct

from nudemo.config import AppConfig
from nudemo.extraction.providers import resolve_provider
from nudemo.ingestion.kafka import KafkaPayloadEncoder


def _synthetic_config(monkeypatch):
    monkeypatch.setenv("NUDEMO_SYNTHETIC_ENABLED", "true")
    return AppConfig.load()


def test_metadata_payload_encodes_blob_refs(monkeypatch):
    config = _synthetic_config(monkeypatch)
    sample = next(resolve_provider(config, "synthetic").iter_samples(limit=1))
    payload = KafkaPayloadEncoder(config.services.minio).metadata_only(sample, 0)
    decoded = json.loads(payload.decode("utf-8"))

    assert decoded["sample_idx"] == 0
    assert decoded["blob_refs"]["CAM_FRONT"].startswith(f"{config.services.minio.bucket}/samples/")


def test_full_payload_frames_json_and_image(monkeypatch):
    config = _synthetic_config(monkeypatch)
    sample = next(resolve_provider(config, "synthetic").iter_samples(limit=1))
    payload = KafkaPayloadEncoder(config.services.minio).full_payload(sample)

    metadata_length = struct.unpack("<I", payload[:4])[0]
    assert metadata_length > 0
    metadata = json.loads(payload[4 : 4 + metadata_length].decode("utf-8"))
    assert metadata["token"] == sample.token
