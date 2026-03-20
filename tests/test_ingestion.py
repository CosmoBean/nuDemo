import json
import struct

from nudemo.config import AppConfig
from nudemo.extraction.providers import resolve_provider
from nudemo.ingestion.kafka import KafkaBenchmarker, KafkaPayloadEncoder


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


def test_kafka_benchmarker_uses_backpressure_and_periodic_flush(monkeypatch):
    config = _synthetic_config(monkeypatch)
    sample_iter = resolve_provider(config, "synthetic").iter_samples(limit=9)

    class FakeProducer:
        instance = None

        def __init__(self, _settings):
            self.flush_calls = 0
            self.poll_calls = []
            self.produced = 0
            self._failed_once = False
            FakeProducer.instance = self

        def produce(self, _topic, key=None, value=None):
            assert key is not None
            assert value is not None
            if not self._failed_once:
                self._failed_once = True
                raise BufferError("queue full")
            self.produced += 1

        def poll(self, timeout):
            self.poll_calls.append(timeout)

        def flush(self):
            self.flush_calls += 1
            return 0

    benchmarker = KafkaBenchmarker(
        settings=config.services.kafka,
        encoder=KafkaPayloadEncoder(config.services.minio),
    )
    monkeypatch.setattr(
        KafkaBenchmarker,
        "_load_kafka",
        lambda self: (None, None, FakeProducer, None),
    )

    result = benchmarker.produce_samples(sample_iter, mode="full-payload")

    assert result["messages"] == 9
    assert FakeProducer.instance is not None
    assert FakeProducer.instance.flush_calls >= 2
    assert any(timeout > 0 for timeout in FakeProducer.instance.poll_calls)
