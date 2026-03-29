from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

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
from nudemo.explorer.app import create_app


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


def test_api_mining_search_returns_hydrated_items(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("nudemo.explorer.app.ensure_metrics_exporter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("nudemo.explorer.app.install_http_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("nudemo.explorer.app.ElasticsearchBackend.is_available", lambda _self: True)
    monkeypatch.setattr(
        "nudemo.explorer.app.MiningSearchService.search",
        lambda *_args, **_kwargs: {
            "total": 1,
            "hits": [
                {
                    "sample_idx": 12,
                    "score": 1.2345,
                    "score_breakdown": {"lexical": 0.2111, "fused": 0.6888},
                    "dominant_signal": "fused",
                }
            ],
            "aggs": {"scenes": [{"scene_token": "scene-token", "sample_count": 1}]},
            "meta": {"mode": "hybrid", "encoder_backend": "fallback"},
        },
    )
    monkeypatch.setattr(
        "nudemo.explorer.app.ExplorerStore.fetch_samples_by_ids",
        lambda _self, _ids: [
            {
                "sample_idx": 12,
                "token": "sample-token",
                "scene_token": "scene-token",
                "scene_name": "scene-0001",
                "timestamp": 123,
                "location": "singapore-onenorth",
                "num_annotations": 6,
                "num_lidar_points": 3200,
                "annotation_categories": ["vehicle.car"],
                "preview_url": "/api/samples/12/cameras/CAM_FRONT",
            }
        ],
    )

    client = TestClient(create_app(config=_build_config(tmp_path)))
    response = client.post(
        "/api/mining/search",
        json={"q": "scene-0001", "mode": "hybrid", "limit": 12, "offset": 0},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["sample_idx"] == 12
    assert payload["items"][0]["match"]["dominant_signal"] == "fused"
    assert payload["items"][0]["match"]["breakdown"]["fused"] == 0.6888
    assert payload["mining"]["mode"] == "hybrid"


def test_mining_session_and_cohort_routes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("nudemo.explorer.app.ensure_metrics_exporter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("nudemo.explorer.app.install_http_metrics", lambda *_args, **_kwargs: None)

    session = {
        "session_id": "session-1",
        "label": "night pedestrians",
        "query": "pedestrian crossing at night",
        "mode": "hybrid",
        "modality_weights": {},
        "positive_sample_ids": [12],
        "negative_sample_ids": [48],
    }
    cohort = {
        "cohort_id": "cohort-1",
        "session_id": "session-1",
        "name": "night pedestrian cohort",
        "query": "pedestrian crossing at night",
        "filters": {"location": "singapore-onenorth"},
        "sample_ids": [12, 22, 32],
    }

    monkeypatch.setattr(
        "nudemo.explorer.app.MiningSessionStore.create_session",
        lambda _self, **_kwargs: session,
    )
    monkeypatch.setattr(
        "nudemo.explorer.app.MiningSessionStore.get_session",
        lambda _self, _session_id: session,
    )
    monkeypatch.setattr(
        "nudemo.explorer.app.MiningSessionStore.replace_examples",
        lambda _self, _session_id, **_kwargs: session,
    )
    monkeypatch.setattr(
        "nudemo.explorer.app.MiningSessionStore.list_sessions",
        lambda _self, limit=24: [
            {
                "session_id": "session-1",
                "label": "night pedestrians",
                "query": "pedestrian crossing at night",
                "mode": "hybrid",
                "modality_weights": {},
                "positive_count": 1,
                "negative_count": 1,
            }
        ],
    )
    monkeypatch.setattr(
        "nudemo.explorer.app.MiningSessionStore.save_cohort",
        lambda _self, _session_id, **_kwargs: cohort,
    )
    monkeypatch.setattr(
        "nudemo.explorer.app.MiningSessionStore.list_cohorts",
        lambda _self, limit=24: [cohort],
    )
    monkeypatch.setattr(
        "nudemo.explorer.app.MiningSessionStore.get_cohort",
        lambda _self, _cohort_id: cohort,
    )

    client = TestClient(create_app(config=_build_config(tmp_path)))

    created = client.post(
        "/api/mining/sessions",
        json={"label": "night pedestrians", "query": "pedestrian crossing at night"},
    )
    assert created.status_code == 200
    assert created.json()["session_id"] == "session-1"
    assert created.json()["modality_preset"] == "balanced"

    updated = client.put(
        "/api/mining/sessions/session-1/examples",
        json={
            "positive_sample_ids": [12],
            "negative_sample_ids": [48],
            "query": "pedestrian crossing at night",
            "mode": "hybrid",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["positive_count"] == 1
    assert updated.json()["negative_count"] == 1

    overview = client.get("/api/mining/overview")
    assert overview.status_code == 200
    assert overview.json()["sessions"][0]["session_id"] == "session-1"
    assert overview.json()["cohorts"][0]["cohort_id"] == "cohort-1"

    saved = client.post(
        "/api/mining/cohorts",
        json={
            "session_id": "session-1",
            "name": "night pedestrian cohort",
            "query": "pedestrian crossing at night",
            "sample_ids": [12, 22, 32],
        },
    )
    assert saved.status_code == 200
    assert saved.json()["name"] == "night pedestrian cohort"
