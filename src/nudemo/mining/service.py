# ruff: noqa: E501

from __future__ import annotations

from collections import defaultdict

import numpy as np
import psycopg
from minio import Minio
from psycopg.rows import dict_row

from nudemo.config import AppConfig
from nudemo.mining.embeddings import (
    EncodedSampleVectors,
    MultimodalEmbeddingEncoder,
    build_metadata_text,
    normalize_modality_weights,
)
from nudemo.storage.elasticsearch_store import ElasticsearchBackend

CAMERA_PATH_COLUMNS = {
    "CAM_FRONT": "cam_front_path",
    "CAM_FRONT_LEFT": "cam_front_left_path",
    "CAM_FRONT_RIGHT": "cam_front_right_path",
    "CAM_BACK": "cam_back_path",
    "CAM_BACK_LEFT": "cam_back_left_path",
    "CAM_BACK_RIGHT": "cam_back_right_path",
}

RADAR_PATH_COLUMNS = {
    "RADAR_FRONT": "radar_front_path",
    "RADAR_FRONT_LEFT": "radar_front_left_path",
    "RADAR_FRONT_RIGHT": "radar_front_right_path",
    "RADAR_BACK_LEFT": "radar_back_left_path",
    "RADAR_BACK_RIGHT": "radar_back_right_path",
}

LIDAR_PATH_COLUMN = "lidar_top_path"


class MiningSearchService:
    def __init__(
        self,
        config: AppConfig,
        *,
        es_backend: ElasticsearchBackend | None = None,
        encoder: MultimodalEmbeddingEncoder | None = None,
    ) -> None:
        self._config = config
        self._es = es_backend or ElasticsearchBackend(url=config.services.elasticsearch.url)
        self._encoder = encoder or MultimodalEmbeddingEncoder()

    @property
    def encoder_backend(self) -> str:
        return self._encoder.backend

    def index_loaded_samples(
        self,
        *,
        rebuild: bool = True,
        batch_size: int = 24,
        limit: int | None = None,
        scene_limit: int | None = None,
    ) -> dict[str, object]:
        if rebuild:
            self._es.clear()
        else:
            self._es.ensure_index()

        client = Minio(
            self._config.services.minio.endpoint,
            access_key=self._config.services.minio.access_key,
            secret_key=self._config.services.minio.secret_key,
            secure=self._config.services.minio.secure,
        )

        total = 0
        batch_rows: list[dict[str, object]] = []
        with psycopg.connect(self._config.services.postgres.dsn, row_factory=dict_row) as connection:
            with connection.cursor(name="mining_index_cursor") as cursor:
                cursor.itersize = max(1, min(batch_size, 128))
                cursor.execute(self._index_sql(limit=limit, scene_limit=scene_limit))
                for row in cursor:
                    batch_rows.append(dict(row))
                    if len(batch_rows) >= batch_size:
                        total += self._index_batch_rows(client, batch_rows)
                        batch_rows = []
                if batch_rows:
                    total += self._index_batch_rows(client, batch_rows)

        return {
            "indexed": total,
            "index": self._es.index,
            "encoder_backend": self._encoder.backend,
            "encoder_model": self._encoder.model_name,
        }

    def _index_batch_rows(self, client: Minio, rows: list[dict[str, object]]) -> int:
        payloads = []
        for row in rows:
            payloads.append(
                {
                    "camera_payloads": {
                        name: self._fetch_object(client, row[column])
                        for name, column in CAMERA_PATH_COLUMNS.items()
                        if row.get(column)
                    },
                    "lidar_payload": (
                        self._fetch_object(client, row[LIDAR_PATH_COLUMN])
                        if row.get(LIDAR_PATH_COLUMN)
                        else None
                    ),
                    "radar_payloads": {
                        name: self._fetch_object(client, row[column])
                        for name, column in RADAR_PATH_COLUMNS.items()
                        if row.get(column)
                    },
                    "metadata_text": build_metadata_text(row),
                }
            )
        vectors = self._encoder.encode_sample_payload_batch(payloads)
        documents = [
            self._build_document(row, vector)
            for row, vector in zip(rows, vectors, strict=False)
        ]
        return self._es.bulk_index_documents(documents)

    def search(
        self,
        *,
        q: str = "",
        scene_token: str = "",
        location: str = "",
        category: str = "",
        min_annotations: int = 0,
        size: int = 20,
        from_: int = 0,
        mode: str = "hybrid",
        modality_weights: dict[str, float] | None = None,
        positive_sample_ids: list[int] | None = None,
        negative_sample_ids: list[int] | None = None,
    ) -> dict[str, object]:
        positive_sample_ids = [int(value) for value in (positive_sample_ids or [])]
        negative_sample_ids = [int(value) for value in (negative_sample_ids or [])]
        weights = normalize_modality_weights(modality_weights)
        normalized_q = q.strip()

        candidate_size = max(40, min(200, (from_ + size) * 4))
        lexical_payload = self._es.search(
            q=normalized_q,
            scene_token=scene_token,
            location=location,
            category=category,
            min_annotations=min_annotations,
            size=candidate_size,
            from_=0,
        )

        search_vector = None
        if normalized_q:
            search_vector = self._encoder.encode_text(normalized_q)

        example_docs = self._es.fetch_documents(
            positive_sample_ids + negative_sample_ids,
            include_vectors=True,
        )
        positive_docs = [doc for doc in example_docs if int(doc["sample_idx"]) in set(positive_sample_ids)]
        negative_docs = [doc for doc in example_docs if int(doc["sample_idx"]) in set(negative_sample_ids)]
        positive_centroid = _centroid([_vector_from_doc(doc, "fused_vec") for doc in positive_docs])
        negative_centroid = _centroid([_vector_from_doc(doc, "fused_vec") for doc in negative_docs])

        if mode == "example-driven" and positive_centroid is not None:
            search_vector = positive_centroid
        elif mode == "hybrid" and positive_centroid is not None and search_vector is not None:
            search_vector = _normalize(search_vector + positive_centroid)
        elif search_vector is None and positive_centroid is not None:
            search_vector = positive_centroid

        vector_payload = {"hits": []}
        if search_vector is not None:
            vector_payload = self._es.vector_search(
                query_vector=search_vector,
                scene_token=scene_token,
                location=location,
                category=category,
                min_annotations=min_annotations,
                size=candidate_size,
            )

        lexical_ranks = {int(hit["sample_idx"]): idx for idx, hit in enumerate(lexical_payload.get("hits", []))}
        vector_ranks = {int(hit["sample_idx"]): idx for idx, hit in enumerate(vector_payload.get("hits", []))}
        candidate_ids = []
        seen: set[int] = set()
        for hit in lexical_payload.get("hits", []) + vector_payload.get("hits", []):
            sample_idx = int(hit["sample_idx"])
            if sample_idx in seen:
                continue
            seen.add(sample_idx)
            candidate_ids.append(sample_idx)

        documents = self._es.fetch_documents(candidate_ids, include_vectors=True)
        ranked_hits: list[dict[str, object]] = []
        for doc in documents:
            sample_idx = int(doc["sample_idx"])
            lexical_component = _rrf(lexical_ranks.get(sample_idx))
            vector_component = _rrf(vector_ranks.get(sample_idx))
            image_component = _cosine(search_vector, _vector_from_doc(doc, "image_vec")) if search_vector is not None else 0.0
            lidar_component = _cosine(search_vector, _vector_from_doc(doc, "lidar_vec")) if search_vector is not None else 0.0
            radar_component = _cosine(search_vector, _vector_from_doc(doc, "radar_vec")) if search_vector is not None else 0.0
            metadata_component = _cosine(search_vector, _vector_from_doc(doc, "metadata_vec")) if search_vector is not None else 0.0
            fused_component = _cosine(search_vector, _vector_from_doc(doc, "fused_vec")) if search_vector is not None else 0.0
            positive_component = _cosine(positive_centroid, _vector_from_doc(doc, "fused_vec")) if positive_centroid is not None else 0.0
            negative_component = _cosine(negative_centroid, _vector_from_doc(doc, "fused_vec")) if negative_centroid is not None else 0.0
            total = (
                (weights["lexical"] * lexical_component)
                + (weights["fused"] * max(fused_component, 0.0))
                + (weights["image"] * max(image_component, 0.0))
                + (weights["lidar"] * max(lidar_component, 0.0))
                + (weights["radar"] * max(radar_component, 0.0))
                + (weights["metadata"] * max(metadata_component, 0.0))
                + (weights["positive"] * max(positive_component, 0.0))
                - (weights["negative"] * max(negative_component, 0.0))
                + (0.1 * vector_component)
            )
            categories = sorted({annotation["category"] for annotation in doc.get("annotations", [])})
            breakdown = {
                "lexical": round(lexical_component, 4),
                "fused": round(fused_component, 4),
                "image": round(image_component, 4),
                "lidar": round(lidar_component, 4),
                "radar": round(radar_component, 4),
                "metadata": round(metadata_component, 4),
                "positive": round(positive_component, 4),
                "negative": round(negative_component, 4),
            }
            dominant_signal = max(
                ("lexical", lexical_component),
                ("image", image_component),
                ("lidar", lidar_component),
                ("radar", radar_component),
                ("metadata", metadata_component),
                ("positive", positive_component),
                key=lambda item: item[1],
            )[0]
            ranked_hits.append(
                {
                    "sample_idx": sample_idx,
                    "token": doc.get("token", ""),
                    "scene_token": doc.get("scene_token", ""),
                    "scene_name": doc.get("scene_name", ""),
                    "location": doc.get("location", ""),
                    "num_annotations": int(doc.get("num_annotations", 0)),
                    "categories": categories,
                    "score": round(total, 6),
                    "score_breakdown": breakdown,
                    "dominant_signal": dominant_signal,
                }
            )

        ranked_hits.sort(key=lambda hit: (-float(hit["score"]), int(hit["sample_idx"])))
        page_hits = ranked_hits[from_: from_ + size]
        return {
            "total": len(ranked_hits),
            "hits": page_hits,
            "aggs": _aggregate_hits(ranked_hits),
            "meta": {
                "mode": mode,
                "encoder_backend": self._encoder.backend,
                "positive_count": len(positive_sample_ids),
                "negative_count": len(negative_sample_ids),
            },
        }

    def _index_sql(self, *, limit: int | None, scene_limit: int | None) -> str:
        scene_cte = ""
        scene_join = ""
        if scene_limit is not None:
            scene_cte = (
                "WITH selected_scenes AS (SELECT scene_token FROM scenes ORDER BY scene_name LIMIT "
                f"{max(1, int(scene_limit))}) "
            )
            scene_join = "JOIN selected_scenes ss ON ss.scene_token = s.scene_token"
        limit_clause = f"LIMIT {max(1, int(limit))}" if limit is not None else ""
        camera_columns = ", ".join(f"s.{column} AS {column}" for column in CAMERA_PATH_COLUMNS.values())
        radar_columns = ", ".join(f"s.{column} AS {column}" for column in RADAR_PATH_COLUMNS.values())
        return f"""
            {scene_cte}
            SELECT
                s.sample_idx,
                s.token,
                s.scene_token,
                sc.scene_name,
                s.location,
                s.timestamp,
                s.num_annotations,
                s.num_lidar_points,
                {camera_columns},
                s.{LIDAR_PATH_COLUMN} AS {LIDAR_PATH_COLUMN},
                {radar_columns},
                COALESCE(
                    json_agg(
                        json_build_object(
                            'category',      a.category,
                            'num_lidar_pts', a.num_lidar_pts,
                            'num_radar_pts', a.num_radar_pts,
                            'size_x',        COALESCE(a.size[1], 0),
                            'size_y',        COALESCE(a.size[2], 0),
                            'size_z',        COALESCE(a.size[3], 0)
                        ) ORDER BY a.id
                    ) FILTER (WHERE a.id IS NOT NULL),
                    '[]'::json
                ) AS annotations
            FROM samples s
            JOIN scenes sc ON sc.scene_token = s.scene_token
            {scene_join}
            LEFT JOIN annotations a ON a.sample_idx = s.sample_idx
            GROUP BY
                s.sample_idx,
                s.token,
                s.scene_token,
                sc.scene_name,
                s.location,
                s.timestamp,
                s.num_annotations,
                s.num_lidar_points,
                {", ".join(f"s.{column}" for column in CAMERA_PATH_COLUMNS.values())},
                s.{LIDAR_PATH_COLUMN},
                {", ".join(f"s.{column}" for column in RADAR_PATH_COLUMNS.values())}
            ORDER BY s.sample_idx
            {limit_clause}
        """

    def _fetch_object(self, client: Minio, object_path: str) -> bytes:
        response = client.get_object(self._config.services.minio.bucket, object_path)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    @staticmethod
    def _build_document(row: dict[str, object], vectors: EncodedSampleVectors) -> dict[str, object]:
        annotations = row.get("annotations") or []
        return {
            "sample_idx": int(row["sample_idx"]),
            "token": row.get("token", ""),
            "scene_token": row.get("scene_token", ""),
            "scene_name": row.get("scene_name", ""),
            "location": row.get("location", ""),
            "timestamp": int(row.get("timestamp") or 0),
            "num_annotations": int(row.get("num_annotations") or 0),
            "annotations": [
                {
                    "category": annotation.get("category", ""),
                    "category_text": str(annotation.get("category", "")).replace(".", " "),
                    "category_group": str(annotation.get("category", "")).split(".")[0],
                    "num_lidar_pts": int(annotation.get("num_lidar_pts") or 0),
                    "num_radar_pts": int(annotation.get("num_radar_pts") or 0),
                    "size_x": float(annotation.get("size_x") or 0.0),
                    "size_y": float(annotation.get("size_y") or 0.0),
                    "size_z": float(annotation.get("size_z") or 0.0),
                }
                for annotation in annotations
            ],
            **vectors.as_document_fields(),
        }


def _vector_from_doc(document: dict[str, object], key: str) -> np.ndarray:
    values = document.get(key) or []
    if not values:
        return np.zeros(512, dtype=np.float32)
    return _normalize(np.asarray(values, dtype=np.float32))


def _normalize(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return None
    return (vector / norm).astype(np.float32)


def _centroid(vectors: list[np.ndarray]) -> np.ndarray | None:
    usable = [vector for vector in vectors if vector is not None and np.linalg.norm(vector) > 0]
    if not usable:
        return None
    return _normalize(np.stack(usable, axis=0).mean(axis=0))


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    return float(np.dot(left, right))


def _rrf(rank: int | None, k: int = 60) -> float:
    if rank is None:
        return 0.0
    return 1.0 / (k + rank + 1)


def _aggregate_hits(hits: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    categories: dict[str, int] = defaultdict(int)
    locations: dict[str, int] = defaultdict(int)
    scenes: dict[str, dict[str, object]] = {}
    for hit in hits[:200]:
        locations[str(hit.get("location") or "")] += 1
        for category in hit.get("categories", []):
            categories[str(category)] += 1
        scene_token = str(hit.get("scene_token") or "")
        bucket = scenes.setdefault(
            scene_token,
            {
                "scene_token": scene_token,
                "scene_name": hit.get("scene_name", ""),
                "location": hit.get("location", ""),
                "sample_count": 0,
            },
        )
        bucket["sample_count"] = int(bucket["sample_count"]) + 1
    return {
        "categories": [
            {"category": category, "count": count}
            for category, count in sorted(categories.items(), key=lambda item: (-item[1], item[0]))[:20]
        ],
        "locations": [
            {"location": location, "count": count}
            for location, count in sorted(locations.items(), key=lambda item: (-item[1], item[0]))[:12]
        ],
        "scenes": list(
            sorted(scenes.values(), key=lambda item: (-int(item["sample_count"]), str(item["scene_name"])))[:24]
        ),
    }
