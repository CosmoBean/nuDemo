from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from nudemo.config import RedisSettings
from nudemo.domain.models import UnifiedSample
from nudemo.storage.base import StorageWriteResult


@dataclass(slots=True)
class RedisBackend:
    settings: RedisSettings
    name: str = "Redis"

    def _client(self):
        import redis

        return redis.Redis(host=self.settings.host, port=self.settings.port, db=self.settings.db)

    def write_samples(self, samples):
        client = self._client()
        pipeline = client.pipeline()
        for pattern in ("sample:*", "embedding:*", "location:*", "category:*"):
            keys = list(client.scan_iter(pattern))
            if keys:
                pipeline.delete(*keys)
        pipeline.delete("samples_by_timestamp")
        pipeline.execute()

        t0 = time.perf_counter()
        bytes_written = 0
        pipeline = client.pipeline()
        samples_written = 0
        for sample_idx, sample in enumerate(samples):
            sample_key = f"sample:{sample_idx:04d}"
            metadata = sample.metadata(sample_idx)
            categories = json.dumps(metadata.annotation_categories)
            mapping = {
                "token": sample.token,
                "scene_name": sample.scene_name,
                "location": sample.location,
                "num_annotations": metadata.num_annotations,
                "num_lidar_points": metadata.num_lidar_points,
                "categories": categories,
            }
            pipeline.hset(sample_key, mapping=mapping)
            bytes_written += sum(len(str(value)) for value in mapping.values())

            embedding = self._derive_embedding(sample).tobytes()
            pipeline.set(f"embedding:{sample_idx:04d}", embedding)
            bytes_written += len(embedding)

            pipeline.zadd("samples_by_timestamp", {sample_key: sample.timestamp})
            pipeline.sadd(f"location:{sample.location}", sample_key)
            for annotation in sample.annotations:
                pipeline.sadd(f"category:{annotation.category}", sample_key)
            if samples_written and samples_written % 100 == 0:
                pipeline.execute()
            samples_written += 1
        pipeline.execute()
        elapsed = time.perf_counter() - t0
        return StorageWriteResult(
            backend=self.name,
            samples_written=samples_written,
            elapsed_sec=elapsed,
            bytes_written=bytes_written,
        )

    def _derive_embedding(self, sample: UnifiedSample) -> np.ndarray:
        camera = sample.cameras["CAM_FRONT"].astype(np.float32)
        row_idx = np.linspace(0, camera.shape[0] - 1, 12, dtype=int)
        col_idx = np.linspace(0, camera.shape[1] - 1, 12, dtype=int)
        thumbnail = camera[row_idx][:, col_idx].reshape(-1) / 255.0

        lidar = sample.lidar_top.astype(np.float32)
        lidar_stats = self._dense_stats_or_zeros(lidar)

        radar_stats = []
        for radar in sample.radars.values():
            radar_array = radar.astype(np.float32)
            radar_stats.extend(
                [
                    float(radar_array.shape[0]),
                    self._column_mean_or_zero(radar_array, 0),
                    self._column_mean_or_zero(radar_array, 1),
                ]
            )

        features = np.concatenate(
            [
                thumbnail.astype(np.float32),
                lidar_stats.astype(np.float32),
                np.asarray(radar_stats, dtype=np.float32),
                np.asarray(
                    [len(sample.annotations), float(sample.lidar_top.shape[0])],
                    dtype=np.float32,
                ),
            ]
        )
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        embedding = np.zeros(512, dtype=np.float32)
        embedding[: min(embedding.shape[0], features.shape[0])] = features[: embedding.shape[0]]
        return embedding

    @staticmethod
    def _column_mean_or_zero(array: np.ndarray, column_idx: int) -> float:
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] <= column_idx:
            return 0.0
        return float(array[:, column_idx].mean())

    @staticmethod
    def _dense_stats_or_zeros(array: np.ndarray, default_width: int = 5) -> np.ndarray:
        if array.ndim != 2:
            return np.zeros(default_width * 4, dtype=np.float32)
        width = array.shape[1] if array.shape[1] > 0 else default_width
        if array.shape[0] == 0:
            return np.zeros(width * 4, dtype=np.float32)
        return np.concatenate(
            [
                array.mean(axis=0),
                array.std(axis=0),
                array.min(axis=0),
                array.max(axis=0),
            ]
        ).astype(np.float32)

    def sequential_iter(self):
        client = self._client()
        for sample_key in client.zrange("samples_by_timestamp", 0, -1):
            idx = sample_key.decode().split(":")[1]
            yield {
                "idx": int(idx),
                "meta": client.hgetall(sample_key),
                "embedding": client.get(f"embedding:{idx}"),
            }

    def fetch(self, sample_idx: int):
        client = self._client()
        key = f"sample:{sample_idx:04d}"
        return {"meta": client.hgetall(key), "embedding": client.get(f"embedding:{sample_idx:04d}")}

    def curation_query(self):
        client = self._client()
        boston = {key.decode() for key in client.smembers("location:boston-seaport")}
        pedestrians: set[str] = set()
        for key in client.scan_iter("category:human.pedestrian*"):
            pedestrians |= {member.decode() for member in client.smembers(key)}
        results = []
        for key in sorted(boston & pedestrians):
            metadata = client.hgetall(key)
            if int(metadata.get(b"num_annotations", 0)) > 5:
                results.append(int(key.split(":")[1]))
        return results

    def disk_footprint(self) -> int:
        client = self._client()
        memory = client.info("memory")
        return int(memory.get("used_memory", 0))
