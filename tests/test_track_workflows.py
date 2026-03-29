from __future__ import annotations

import sys
import types
from pathlib import Path

from nudemo.config import (
    AppConfig,
    ElasticsearchSettings,
    KafkaSettings,
    LanceSettings,
    MinioSettings,
    ParquetSettings,
    PipelineSettings,
    PostgresSettings,
    RedisSettings,
    RuntimePaths,
    ServiceSettings,
    StorageSettings,
    WebDatasetSettings,
)
from nudemo.mining.exports import CohortExportService
from nudemo.mining.store import validate_task_transition
from nudemo.mining.tracks import TrackMaterializer
from nudemo.storage.track_elasticsearch_store import TrackElasticsearchBackend


def _build_config(root: Path) -> AppConfig:
    artifacts = root / "artifacts"
    reports = artifacts / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        runtime=RuntimePaths(
            dataset_root=root / "data",
            artifacts_root=artifacts,
            formats_root=artifacts / "formats",
            reports_root=reports,
        ),
        pipeline=PipelineSettings(
            dataset_version="v1.0-mini",
            sample_limit=404,
            synthetic_enabled=False,
            synthetic_scene_count=10,
            synthetic_samples_per_scene=41,
            synthetic_seed=7,
            camera_height=120,
            camera_width=224,
            lidar_points=2048,
            radar_points=128,
        ),
        services=ServiceSettings(
            kafka=KafkaSettings(
                bootstrap_servers="localhost:9092",
                raw_topic="raw",
                refined_topic="refined",
            ),
            minio=MinioSettings(
                endpoint="localhost:9000",
                access_key="minioadmin",
                secret_key="minioadmin",
                bucket="nuscenes",
                secure=False,
            ),
            postgres=PostgresSettings(
                host="localhost",
                port=5432,
                database="nuscenes",
                user="nuscenes",
                password="nuscenes",
            ),
            redis=RedisSettings(host="localhost", port=6379, db=0),
            elasticsearch=ElasticsearchSettings(url="http://localhost:9200"),
        ),
        storage=StorageSettings(
            lance=LanceSettings(dataset_path=artifacts / "formats/lance/dataset.lance"),
            parquet=ParquetSettings(dataset_path=artifacts / "formats/parquet/dataset"),
            webdataset=WebDatasetSettings(
                shard_pattern="artifacts/formats/webdataset/shard-%04d.tar",
                maxcount=50,
            ),
        ),
    )


def test_validate_task_transition_rejects_invalid_flow() -> None:
    assert validate_task_transition("queued", "assigned") is True
    assert validate_task_transition("queued", "qa_passed") is False


def test_track_materializer_materializes_loaded_samples(monkeypatch, tmp_path: Path) -> None:
    config = _build_config(tmp_path)

    class FakeTrackStore:
        def __init__(self):
            self.persisted = None

        def fetch_loaded_samples(self, *, limit=None, scene_limit=None):
            return [
                {
                    "sample_idx": 12,
                    "token": "sample-a",
                    "scene_token": "scene-token",
                    "scene_name": "scene-0001",
                    "location": "singapore-onenorth",
                    "timestamp": 100,
                },
                {
                    "sample_idx": 13,
                    "token": "sample-b",
                    "scene_token": "scene-token",
                    "scene_name": "scene-0001",
                    "location": "singapore-onenorth",
                    "timestamp": 200,
                },
            ]

        def replace_tracks(self, *, tracks, observations):
            self.persisted = {"tracks": tracks, "observations": observations}
            return {"tracks": len(tracks), "observations": len(observations)}

    class FakeNuScenes:
        def __init__(self, *args, **kwargs):
            self._samples = {
                "sample-a": {"anns": ["ann-1"]},
                "sample-b": {"anns": ["ann-2", "ann-3"]},
            }
            self._annotations = {
                "ann-1": {
                    "instance_token": "track-1",
                    "category_name": "human.pedestrian.adult",
                    "translation": [1.0, 2.0, 3.0],
                    "size": [1.0, 1.0, 1.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "num_lidar_pts": 5,
                    "num_radar_pts": 1,
                    "visibility_token": "4",
                    "attribute_tokens": ["moving"],
                },
                "ann-2": {
                    "instance_token": "track-1",
                    "category_name": "human.pedestrian.adult",
                    "translation": [2.0, 2.0, 3.0],
                    "size": [1.0, 1.0, 1.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "num_lidar_pts": 8,
                    "num_radar_pts": 2,
                    "visibility_token": "3",
                    "attribute_tokens": [],
                },
                "ann-3": {
                    "instance_token": "track-2",
                    "category_name": "vehicle.car",
                    "translation": [5.0, 6.0, 1.0],
                    "size": [4.0, 1.8, 1.4],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "num_lidar_pts": 12,
                    "num_radar_pts": 0,
                    "visibility_token": "",
                    "attribute_tokens": ["parked"],
                },
            }

        def get(self, table: str, token: str):
            if table == "sample":
                return self._samples[token]
            if table == "sample_annotation":
                return self._annotations[token]
            raise KeyError(table)

    monkeypatch.setitem(
        sys.modules,
        "nuscenes.nuscenes",
        types.SimpleNamespace(NuScenes=FakeNuScenes),
    )

    fake_store = FakeTrackStore()
    result = TrackMaterializer(config, track_store=fake_store).materialize()

    assert result.tracks == 2
    assert result.observations == 3
    assert result.sample_count == 2
    assert fake_store.persisted is not None
    assert fake_store.persisted["tracks"][0]["track_id"] == "track-1"
    assert fake_store.persisted["observations"][0]["observation_idx"] == 0


def test_track_elasticsearch_search_builds_identifier_and_filter_queries(monkeypatch) -> None:
    backend = TrackElasticsearchBackend(url="http://localhost:9200")
    calls: list[tuple[str, str, object]] = []

    def fake_req(method: str, path: str, body=None, *, ndjson: bool = False):
        calls.append((method, path, body))
        return {"hits": {"total": {"value": 0}, "hits": []}}

    monkeypatch.setattr(TrackElasticsearchBackend, "_req", staticmethod(fake_req))
    backend.search(
        q="track-123",
        scene_token="scene-token",
        location="singapore-onenorth",
        category="vehicle.car",
        size=10,
        from_=5,
    )

    _method, _path, body = calls[-1]
    filters = body["query"]["bool"]["filter"]
    should = body["query"]["bool"]["must"][0]["bool"]["should"]
    assert {"term": {"scene_token": "scene-token"}} in filters
    assert {"term": {"location.keyword": "singapore-onenorth"}} in filters
    assert {"term": {"category.keyword": "vehicle.car"}} in filters
    assert any("track_id" in clause.get("term", {}) for clause in should)


def test_cohort_export_service_writes_manifest_and_records_export(tmp_path: Path) -> None:
    config = _build_config(tmp_path)

    class FakeSessionStore:
        def get_cohort(self, cohort_id: str):
            assert cohort_id == "cohort-1"
            return {
                "cohort_id": cohort_id,
                "name": "night pedestrians",
                "query": "pedestrian",
                "filters": {"location": "singapore-onenorth"},
                "sample_ids": [12, 13],
            }

    class FakeExportStore:
        def __init__(self):
            self.recorded = None

        def record_export(self, **kwargs):
            self.recorded = kwargs
            return {"export_id": "export-1", **kwargs}

    class FakeTaskStore:
        def __init__(self):
            self.closed = None

        def close_task(self, task_id: str, *, actor: str, note: str):
            self.closed = {"task_id": task_id, "actor": actor, "note": note}

    export_store = FakeExportStore()
    task_store = FakeTaskStore()
    service = CohortExportService(
        config,
        session_store=FakeSessionStore(),
        export_store=export_store,
        task_store=task_store,
    )
    service._fetch_rows = lambda _sample_ids: [  # type: ignore[attr-defined]
        {"sample_idx": 12, "token": "sample-a", "track_ids": ["track-1"]},
        {"sample_idx": 13, "token": "sample-b", "track_ids": ["track-2"]},
    ]

    result = service.export_cohort("cohort-1", task_id="task-1")

    assert result["export_id"] == "export-1"
    assert export_store.recorded is not None
    assert Path(export_store.recorded["output_path"]).exists()
    assert task_store.closed["task_id"] == "task-1"
