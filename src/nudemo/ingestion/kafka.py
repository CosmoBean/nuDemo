from __future__ import annotations

import io
import json
import struct
import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from nudemo.config import KafkaSettings, MinioSettings
from nudemo.domain.models import UnifiedSample
from nudemo.storage.base import image_to_jpeg_bytes


@dataclass(slots=True)
class KafkaPayloadEncoder:
    minio: MinioSettings

    def metadata_only(self, sample: UnifiedSample, sample_idx: int) -> bytes:
        metadata = sample.metadata(sample_idx)
        metadata.blob_refs = {
            sensor: f"{self.minio.bucket}/{path}" for sensor, path in metadata.blob_refs.items()
        }
        return metadata.as_json().encode("utf-8")

    def full_payload(self, sample: UnifiedSample) -> bytes:
        meta = json.dumps(
            {
                "token": sample.token,
                "timestamp": sample.timestamp,
                "scene_name": sample.scene_name,
                "location": sample.location,
                "num_lidar_points": int(sample.lidar_top.shape[0]),
            },
            sort_keys=True,
        ).encode("utf-8")
        image_bytes = image_to_jpeg_bytes(sample.cameras["CAM_FRONT"])
        lidar_buffer = io.BytesIO()
        np.save(lidar_buffer, sample.lidar_top)
        lidar_bytes = lidar_buffer.getvalue()
        return (
            struct.pack("<I", len(meta))
            + meta
            + struct.pack("<I", len(image_bytes))
            + image_bytes
            + lidar_bytes
        )


@dataclass(slots=True)
class KafkaBenchmarker:
    settings: KafkaSettings
    encoder: KafkaPayloadEncoder

    def _load_kafka(self):
        from confluent_kafka import Consumer, Producer
        from confluent_kafka.admin import AdminClient, NewTopic

        return AdminClient, NewTopic, Producer, Consumer

    def create_topics(self) -> None:
        AdminClient, NewTopic, _, _ = self._load_kafka()
        admin = AdminClient({"bootstrap.servers": self.settings.bootstrap_servers})
        topics = [
            NewTopic(
                self.settings.raw_topic,
                num_partitions=4,
                replication_factor=1,
                config={
                    "max.message.bytes": "10485760",
                    "retention.ms": str(7 * 24 * 3600 * 1000),
                },
            ),
            NewTopic(
                self.settings.refined_topic,
                num_partitions=4,
                replication_factor=1,
                config={"max.message.bytes": "1048576"},
            ),
        ]
        admin.create_topics(topics)

    def produce_samples(
        self,
        samples: Iterable[UnifiedSample],
        mode: str = "metadata-only",
    ) -> dict[str, float]:
        _, _, Producer, _ = self._load_kafka()
        producer = Producer(
            {
                "bootstrap.servers": self.settings.bootstrap_servers,
                "message.max.bytes": 10485760,
                "linger.ms": 50,
                "batch.num.messages": 16,
                "compression.type": "lz4",
            }
        )

        total_messages = 0
        total_bytes = 0
        topic = self.settings.refined_topic if mode == "metadata-only" else self.settings.raw_topic
        t0 = time.perf_counter()
        for sample_idx, sample in enumerate(samples):
            payload = (
                self.encoder.metadata_only(sample, sample_idx)
                if mode == "metadata-only"
                else self.encoder.full_payload(sample)
            )
            producer.produce(topic, key=sample.scene_name.encode("utf-8"), value=payload)
            total_messages += 1
            total_bytes += len(payload)
        producer.flush()
        elapsed = time.perf_counter() - t0
        return {
            "messages": total_messages,
            "total_mb": total_bytes / (1024 * 1024),
            "throughput_msg_sec": total_messages / elapsed if elapsed else 0.0,
            "throughput_mb_sec": (total_bytes / (1024 * 1024)) / elapsed if elapsed else 0.0,
            "elapsed_sec": elapsed,
        }

    def benchmark_consumer(self, topic: str, group_id: str) -> dict[str, float]:
        _, _, _, Consumer = self._load_kafka()
        consumer = Consumer(
            {
                "bootstrap.servers": self.settings.bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "earliest",
                "fetch.max.bytes": 10485760,
            }
        )
        consumer.subscribe([topic])
        total_messages = 0
        total_bytes = 0
        t0 = time.perf_counter()
        while True:
            message = consumer.poll(timeout=1.0)
            if message is None:
                if total_messages:
                    break
                continue
            if message.error():
                continue
            total_messages += 1
            total_bytes += len(message.value())
        elapsed = time.perf_counter() - t0
        consumer.close()
        return {
            "messages": total_messages,
            "total_mb": total_bytes / (1024 * 1024),
            "throughput_msg_sec": total_messages / elapsed if elapsed else 0.0,
            "throughput_mb_sec": (total_bytes / (1024 * 1024)) / elapsed if elapsed else 0.0,
            "elapsed_sec": elapsed,
        }
