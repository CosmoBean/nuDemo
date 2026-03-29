from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from nudemo.mining.service import MiningSearchService


def _unit(values: list[float]) -> np.ndarray:
    vector = np.zeros(512, dtype=np.float32)
    vector[: len(values)] = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector if norm <= 0 else (vector / norm).astype(np.float32)


class _FakeEncoder:
    backend = "fake"
    model_name = "fake-model"

    def __init__(self, vector: np.ndarray | None = None, *, fail_on_text: bool = False) -> None:
        self._vector = vector if vector is not None else _unit([1.0, 0.0, 0.0])
        self._fail_on_text = fail_on_text

    def encode_text(self, _text: str) -> np.ndarray:
        if self._fail_on_text:
            raise AssertionError("encode_text should not be called")
        return self._vector


class _FakeEsBackend:
    def __init__(
        self,
        *,
        lexical_hits: list[dict[str, object]],
        vector_hits: list[dict[str, object]] | None = None,
        documents: dict[int, dict[str, object]] | None = None,
        fail_on_vector: bool = False,
    ) -> None:
        self._lexical_hits = lexical_hits
        self._vector_hits = vector_hits or []
        self._documents = documents or {}
        self._fail_on_vector = fail_on_vector

    def search(self, **_kwargs) -> dict[str, object]:
        return {
            "total": len(self._lexical_hits),
            "hits": list(self._lexical_hits),
            "aggs": {},
        }

    def vector_search(self, **_kwargs) -> dict[str, object]:
        if self._fail_on_vector:
            raise AssertionError("vector_search should not be called")
        return {
            "total": len(self._vector_hits),
            "hits": list(self._vector_hits),
        }

    def fetch_documents(self, sample_ids: list[int], *, include_vectors: bool = False) -> list[dict[str, object]]:
        del include_vectors
        return [self._documents[sample_id] for sample_id in sample_ids if sample_id in self._documents]


def _document(
    sample_idx: int,
    *,
    scene_name: str,
    token: str = "",
    scene_token: str = "",
    fused: list[float] | None = None,
    image: list[float] | None = None,
    lidar: list[float] | None = None,
    radar: list[float] | None = None,
    metadata: list[float] | None = None,
) -> dict[str, object]:
    return {
        "sample_idx": sample_idx,
        "token": token or f"token-{sample_idx}",
        "scene_token": scene_token or f"scene-token-{sample_idx}",
        "scene_name": scene_name,
        "location": "singapore-onenorth",
        "num_annotations": 4,
        "annotations": [{"category": "human.pedestrian.adult"}],
        "image_vec": image or [],
        "lidar_vec": lidar or [],
        "radar_vec": radar or [],
        "metadata_vec": metadata or [],
        "fused_vec": fused or [],
    }


def test_lexical_mode_skips_vector_search() -> None:
    es_backend = _FakeEsBackend(
        lexical_hits=[{"sample_idx": 1, "scene_name": "scene-0001", "score": 57.0}],
        documents={1: _document(1, scene_name="scene-0001")},
        fail_on_vector=True,
    )
    service = MiningSearchService(
        SimpleNamespace(),
        es_backend=es_backend,
        encoder=_FakeEncoder(fail_on_text=True),
    )

    payload = service.search(q="scene-0001", mode="lexical", size=5)

    assert payload["total"] == 1
    assert payload["hits"][0]["sample_idx"] == 1
    assert payload["hits"][0]["score_breakdown"]["lexical"] == 1.0


def test_identifier_queries_prefer_exact_lexical_hits() -> None:
    es_backend = _FakeEsBackend(
        lexical_hits=[
            {"sample_idx": 1, "scene_name": "scene-0001", "token": "token-a", "scene_token": "scene-token-a", "score": 57.0},
            {"sample_idx": 2, "scene_name": "scene-0001", "token": "token-b", "scene_token": "scene-token-b", "score": 57.0},
        ],
        vector_hits=[
            {"sample_idx": 99, "scene_name": "scene-0099", "score": 2.0},
        ],
        documents={
            1: _document(1, scene_name="scene-0001", token="token-a", scene_token="scene-token-a"),
            2: _document(2, scene_name="scene-0001", token="token-b", scene_token="scene-token-b"),
            99: _document(
                99,
                scene_name="scene-0099",
                fused=_unit([1.0, 0.0, 0.0]).tolist(),
                image=_unit([1.0, 0.0, 0.0]).tolist(),
            ),
        },
    )
    service = MiningSearchService(
        SimpleNamespace(),
        es_backend=es_backend,
        encoder=_FakeEncoder(_unit([1.0, 0.0, 0.0])),
    )

    payload = service.search(q="scene-0001", mode="hybrid", size=10)

    assert [hit["sample_idx"] for hit in payload["hits"]] == [1, 2]
    assert all(hit["scene_name"] == "scene-0001" for hit in payload["hits"])


def test_vector_only_queries_abstain_when_semantics_are_too_weak() -> None:
    weak_vector = _unit([0.30, 0.95, 0.0]).tolist()
    es_backend = _FakeEsBackend(
        lexical_hits=[],
        vector_hits=[{"sample_idx": 7, "score": 1.2}],
        documents={
            7: _document(
                7,
                scene_name="scene-0007",
                fused=weak_vector,
                image=weak_vector,
                lidar=[],
                radar=[],
                metadata=[],
            )
        },
    )
    service = MiningSearchService(
        SimpleNamespace(),
        es_backend=es_backend,
        encoder=_FakeEncoder(_unit([1.0, 0.0, 0.0])),
    )

    payload = service.search(q="ostrich", mode="hybrid", size=10)

    assert payload["total"] == 0
    assert payload["hits"] == []
