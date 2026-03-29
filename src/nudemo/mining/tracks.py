from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any

from nudemo.config import AppConfig
from nudemo.mining.store import TrackStore
from nudemo.storage.track_elasticsearch_store import TrackElasticsearchBackend


@dataclass(slots=True)
class TrackMaterializationResult:
    tracks: int
    observations: int
    sample_count: int
    scene_count: int
    dataset_version: str

    def as_dict(self) -> dict[str, object]:
        return {
            "tracks": self.tracks,
            "observations": self.observations,
            "sample_count": self.sample_count,
            "scene_count": self.scene_count,
            "dataset_version": self.dataset_version,
        }


class TrackMaterializer:
    def __init__(
        self,
        config: AppConfig,
        *,
        track_store: TrackStore | None = None,
        es_backend: TrackElasticsearchBackend | None = None,
    ) -> None:
        self._config = config
        self._track_store = track_store or TrackStore(config.services.postgres)
        self._es = es_backend or TrackElasticsearchBackend(url=config.services.elasticsearch.url)

    def materialize(
        self,
        *,
        limit: int | None = None,
        scene_limit: int | None = None,
    ) -> TrackMaterializationResult:
        from nuscenes.nuscenes import NuScenes

        rows = self._track_store.fetch_loaded_samples(limit=limit, scene_limit=scene_limit)
        if not rows:
            result = TrackMaterializationResult(
                tracks=0,
                observations=0,
                sample_count=0,
                scene_count=0,
                dataset_version=self._config.pipeline.dataset_version,
            )
            self._track_store.replace_tracks(tracks=[], observations=[])
            return result

        nusc = NuScenes(
            version=self._config.pipeline.dataset_version,
            dataroot=str(self._config.runtime.dataset_root),
            verbose=False,
        )

        track_builders: dict[str, dict[str, Any]] = {}
        observations: list[dict[str, object]] = []
        scene_tokens: set[str] = set()

        for row in rows:
            sample_idx = int(row["sample_idx"])
            sample_token = str(row["token"])
            scene_token = str(row["scene_token"])
            sample_record = nusc.get("sample", sample_token)
            scene_tokens.add(scene_token)
            for annotation_token in sample_record["anns"]:
                annotation = nusc.get("sample_annotation", annotation_token)
                track_id = str(annotation.get("instance_token") or annotation_token)
                category = str(annotation.get("category_name") or "")
                builder = track_builders.setdefault(
                    track_id,
                    {
                        "track_id": track_id,
                        "scene_token": scene_token,
                        "scene_name": str(row.get("scene_name") or scene_token),
                        "location": str(row.get("location") or ""),
                        "category": category,
                        "sample_ids": [],
                        "timestamps": [],
                        "num_lidar_pts": [],
                        "num_radar_pts": [],
                        "visibility_tokens": set(),
                    },
                )
                builder["sample_ids"].append(sample_idx)
                builder["timestamps"].append(int(row.get("timestamp") or 0))
                builder["num_lidar_pts"].append(int(annotation.get("num_lidar_pts") or 0))
                builder["num_radar_pts"].append(int(annotation.get("num_radar_pts") or 0))
                visibility_token = str(annotation.get("visibility_token") or "")
                if visibility_token:
                    builder["visibility_tokens"].add(visibility_token)
                observations.append(
                    {
                        "track_id": track_id,
                        "sample_idx": sample_idx,
                        "sample_token": sample_token,
                        "annotation_token": annotation_token,
                        "observation_idx": 0,
                        "timestamp": int(row.get("timestamp") or 0),
                        "category": category,
                        "translation": list(annotation.get("translation") or []),
                        "size": list(annotation.get("size") or []),
                        "rotation": list(annotation.get("rotation") or []),
                        "num_lidar_pts": int(annotation.get("num_lidar_pts") or 0),
                        "num_radar_pts": int(annotation.get("num_radar_pts") or 0),
                        "visibility_token": visibility_token,
                        "attribute_tokens": list(annotation.get("attribute_tokens") or []),
                    }
                )

        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for observation in observations:
            grouped[str(observation["track_id"])].append(observation)

        ordered_observations: list[dict[str, object]] = []
        tracks: list[dict[str, object]] = []
        for track_id, builder in sorted(
            track_builders.items(),
            key=lambda item: (item[1]["scene_name"], item[0]),
        ):
            items = sorted(
                grouped.get(track_id, []),
                key=lambda item: (
                    int(item["timestamp"]),
                    int(item["sample_idx"]),
                    str(item["annotation_token"]),
                ),
            )
            for index, item in enumerate(items):
                item["observation_idx"] = index
                ordered_observations.append(item)
            sample_ids = [int(value) for value in builder["sample_ids"]]
            timestamps = [int(value) for value in builder["timestamps"]]
            lidar_counts = [int(value) for value in builder["num_lidar_pts"]]
            radar_counts = [int(value) for value in builder["num_radar_pts"]]
            tracks.append(
                {
                    "track_id": track_id,
                    "scene_token": builder["scene_token"],
                    "scene_name": builder["scene_name"],
                    "location": builder["location"],
                    "category": builder["category"],
                    "start_timestamp": min(timestamps),
                    "end_timestamp": max(timestamps),
                    "sample_ids": sorted(sample_ids),
                    "sample_count": len(sample_ids),
                    "annotation_count": len(items),
                    "avg_num_lidar_pts": float(mean(lidar_counts)) if lidar_counts else 0.0,
                    "avg_num_radar_pts": float(mean(radar_counts)) if radar_counts else 0.0,
                    "max_num_lidar_pts": max(lidar_counts, default=0),
                    "max_num_radar_pts": max(radar_counts, default=0),
                    "visibility_tokens": sorted(builder["visibility_tokens"]),
                }
            )

        persisted = self._track_store.replace_tracks(
            tracks=tracks,
            observations=ordered_observations,
        )
        return TrackMaterializationResult(
            tracks=int(persisted["tracks"]),
            observations=int(persisted["observations"]),
            sample_count=len(rows),
            scene_count=len(scene_tokens),
            dataset_version=self._config.pipeline.dataset_version,
        )

    def materialize_loaded_tracks(
        self,
        *,
        limit: int | None = None,
        scene_limit: int | None = None,
    ) -> dict[str, object]:
        return self.materialize(limit=limit, scene_limit=scene_limit).as_dict()

    def index_materialized_tracks(
        self,
        *,
        rebuild: bool = True,
        batch_size: int = 250,
        limit: int | None = None,
        scene_limit: int | None = None,
    ) -> dict[str, object]:
        if limit is not None or scene_limit is not None:
            self.materialize(limit=limit, scene_limit=scene_limit)
        if rebuild:
            self._es.clear()
        else:
            self._es.ensure_index()
        indexed = self._es.bulk_index_from_postgres(
            self._config.services.postgres,
            batch_size=batch_size,
        )
        return {
            "indexed": indexed,
            "index": self._es.index,
            "doc_count": self._es.doc_count(),
        }

    def materialize_and_index(
        self,
        *,
        rebuild: bool = True,
        batch_size: int = 250,
        limit: int | None = None,
        scene_limit: int | None = None,
    ) -> dict[str, object]:
        materialized = self.materialize(limit=limit, scene_limit=scene_limit).as_dict()
        indexed = self.index_materialized_tracks(
            rebuild=rebuild,
            batch_size=batch_size,
        )
        return {**materialized, **indexed}
