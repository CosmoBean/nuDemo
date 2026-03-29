# ruff: noqa: E501

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from nudemo.domain.models import CAMERAS, RADARS
from nudemo.mining.embeddings import VECTOR_DIM, build_metadata_text
from nudemo.storage.base import array_to_npy_bytes, image_to_jpeg_bytes

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
VECTOR_FIELDS = ("image_vec", "lidar_vec", "radar_vec", "metadata_vec", "fused_vec")
_QUERY_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_CATEGORY_GROUPS = {"human", "vehicle", "movable_object", "static_object"}
_HEX_IDENTIFIER_PATTERN = re.compile(r"^[0-9a-f]{6,32}$", re.IGNORECASE)
_SCENE_IDENTIFIER_PATTERN = re.compile(r"^scene-\d{1,4}$", re.IGNORECASE)

_INDEX_MAPPING: dict[str, Any] = {
    "mappings": {
        "properties": {
            "sample_idx": {"type": "integer"},
            "token": {"type": "keyword"},
            "scene_token": {"type": "keyword"},
            "scene_name": {"type": "keyword"},
            "location": {"type": "keyword"},
            "timestamp": {"type": "long"},
            "num_annotations": {"type": "integer"},
            "annotation_categories": {"type": "keyword"},
            "encoder_backend": {"type": "keyword"},
            "encoder_model": {"type": "keyword"},
            "has_image": {"type": "boolean"},
            "has_lidar": {"type": "boolean"},
            "has_radar": {"type": "boolean"},
            "has_metadata": {"type": "boolean"},
            "image_vec": {"type": "dense_vector", "dims": VECTOR_DIM, "index": False},
            "lidar_vec": {"type": "dense_vector", "dims": VECTOR_DIM, "index": False},
            "radar_vec": {"type": "dense_vector", "dims": VECTOR_DIM, "index": False},
            "metadata_vec": {"type": "dense_vector", "dims": VECTOR_DIM, "index": False},
            "fused_vec": {"type": "dense_vector", "dims": VECTOR_DIM, "index": False},
            "annotations": {
                "type": "nested",
                "properties": {
                    "category": {"type": "keyword"},
                    "category_text": {"type": "text"},
                    "category_group": {"type": "keyword"},
                    "num_lidar_pts": {"type": "integer"},
                    "num_radar_pts": {"type": "integer"},
                    "size_x": {"type": "float"},
                    "size_y": {"type": "float"},
                    "size_z": {"type": "float"},
                },
            },
        }
    }
}


def _escape_wildcard_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("*", "\\*").replace("?", "\\?")


def _query_tokens(value: str) -> set[str]:
    return {token for token in _QUERY_TOKEN_PATTERN.findall(value.lower()) if token}


def _looks_like_identifier_query(value: str) -> bool:
    normalized = value.strip()
    return bool(
        _HEX_IDENTIFIER_PATTERN.fullmatch(normalized)
        or _SCENE_IDENTIFIER_PATTERN.fullmatch(normalized)
    )


def _looks_structured_category_query(value: str) -> bool:
    return "." in value or "_" in value


def _category_aliases(value: str) -> list[str]:
    tokens = _query_tokens(value)
    aliases: set[str] = set()
    if tokens & {"person", "people", "human", "pedestrian"}:
        aliases.add("human.pedestrian")
    if tokens & {"man", "woman", "adult"}:
        aliases.add("human.pedestrian.adult")
    if tokens & {"child", "kid"}:
        aliases.add("human.pedestrian.child")
    if tokens & {"worker", "construction"}:
        aliases.add("human.pedestrian.construction_worker")
    return sorted(aliases)


def _build_annotation_document(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
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
    ]


def _source_fields(*, include_vectors: bool) -> list[str]:
    base = [
        "sample_idx",
        "token",
        "scene_token",
        "scene_name",
        "location",
        "timestamp",
        "num_annotations",
        "annotation_categories",
        "annotations",
        "encoder_backend",
        "encoder_model",
        "has_image",
        "has_lidar",
        "has_radar",
        "has_metadata",
    ]
    if include_vectors:
        base.extend(VECTOR_FIELDS)
    return base


@dataclass(slots=True)
class ElasticsearchBackend:
    url: str
    index: str = "nudemo-annotations"

    def _req(self, method: str, path: str, body: object = None, *, ndjson: bool = False) -> dict:
        full_url = self.url.rstrip("/") + "/" + path.lstrip("/")
        data = None
        content_type = "application/x-ndjson" if ndjson else "application/json"
        if body is not None:
            data = (body if ndjson else json.dumps(body)).encode("utf-8")
        req = urllib.request.Request(full_url, data=data, method=method)
        req.add_header("Content-Type", content_type)
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = resp.read()
        return json.loads(payload) if payload else {}

    def ensure_index(self) -> None:
        try:
            self._req("HEAD", f"/{self.index}")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            self._req("PUT", f"/{self.index}", _INDEX_MAPPING)

    def clear(self) -> None:
        try:
            self._req("DELETE", f"/{self.index}")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
        self.ensure_index()

    def _build_doc(self, sample, sample_idx: int) -> dict[str, Any]:
        annotations = [
            {
                "category": ann.category,
                "num_lidar_pts": ann.num_lidar_pts,
                "num_radar_pts": ann.num_radar_pts,
                "size_x": float(ann.size[0]) if ann.size else 0.0,
                "size_y": float(ann.size[1]) if ann.size else 0.0,
                "size_z": float(ann.size[2]) if ann.size else 0.0,
            }
            for ann in sample.annotations
        ]
        return {
            "sample_idx": sample_idx,
            "token": sample.token,
            "scene_token": sample.scene_token,
            "scene_name": sample.scene_name,
            "location": sample.location,
            "timestamp": int(sample.timestamp),
            "num_annotations": len(sample.annotations),
            "annotation_categories": sorted({ann["category"] for ann in annotations if ann["category"]}),
            "annotations": _build_annotation_document(annotations),
        }

    def bulk_index_documents(self, documents: list[dict[str, Any]]) -> int:
        if not documents:
            return 0
        self.ensure_index()
        lines: list[str] = []
        for doc in documents:
            sample_idx = int(doc["sample_idx"])
            lines.append(json.dumps({"index": {"_index": self.index, "_id": str(sample_idx)}}))
            lines.append(json.dumps(doc))
        result = self._req("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
        if result.get("errors"):
            errors = sum(1 for item in result.get("items", []) if "error" in item.get("index", {}))
            if errors:
                raise RuntimeError(f"{errors}/{len(documents)} bulk index errors")
        return len(documents)

    def bulk_index(self, samples_with_idx: list[tuple]) -> int:
        documents = [self._build_doc(sample, sample_idx) for sample, sample_idx in samples_with_idx]
        return self.bulk_index_documents(documents)

    def bulk_index_multimodal(self, samples_with_idx: list[tuple], encoder) -> int:
        documents: list[dict[str, Any]] = []
        for sample, sample_idx in samples_with_idx:
            annotations = [
                {
                    "category": ann.category,
                    "num_lidar_pts": ann.num_lidar_pts,
                    "num_radar_pts": ann.num_radar_pts,
                    "size_x": float(ann.size[0]) if ann.size else 0.0,
                    "size_y": float(ann.size[1]) if ann.size else 0.0,
                    "size_z": float(ann.size[2]) if ann.size else 0.0,
                }
                for ann in sample.annotations
            ]
            sample_record = {
                "sample_idx": sample_idx,
                "token": sample.token,
                "scene_token": sample.scene_token,
                "scene_name": sample.scene_name,
                "location": sample.location,
                "timestamp": int(sample.timestamp),
                "num_annotations": len(sample.annotations),
                "num_lidar_points": int(sample.lidar_top.shape[0]),
                "annotations": annotations,
            }
            vectors = encoder.encode_sample_payloads(
                camera_payloads={
                    camera: image_to_jpeg_bytes(sample.cameras[camera])
                    for camera in CAMERAS
                    if camera in sample.cameras
                },
                lidar_payload=array_to_npy_bytes(sample.lidar_top),
                radar_payloads={
                    radar: array_to_npy_bytes(sample.radars[radar])
                    for radar in RADARS
                    if radar in sample.radars
                },
                metadata_text=build_metadata_text(sample_record),
            )
            documents.append(
                {
                    "sample_idx": sample_idx,
                    "token": sample.token,
                    "scene_token": sample.scene_token,
                    "scene_name": sample.scene_name,
                    "location": sample.location,
                    "timestamp": int(sample.timestamp),
                    "num_annotations": len(sample.annotations),
                    "annotation_categories": sorted(
                        {annotation["category"] for annotation in annotations if annotation["category"]}
                    ),
                    "annotations": _build_annotation_document(annotations),
                    **vectors.as_document_fields(),
                }
            )
        return self.bulk_index_documents(documents)

    def bulk_index_from_postgres(self, postgres_settings, batch_size: int = 500) -> int:
        import psycopg
        import psycopg.rows

        sql = """
            SELECT
                s.sample_idx,
                s.token,
                s.scene_token,
                sc.scene_name,
                s.location,
                s.timestamp,
                s.num_annotations,
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
            LEFT JOIN annotations a ON a.sample_idx = s.sample_idx
            GROUP BY s.sample_idx, s.token, s.scene_token, sc.scene_name,
                     s.location, s.timestamp, s.num_annotations
            ORDER BY s.sample_idx
        """

        total = 0
        batch: list[dict[str, Any]] = []

        with psycopg.connect(postgres_settings.dsn, row_factory=psycopg.rows.dict_row) as conn:
            with conn.cursor(name="es_index_cursor") as cur:
                cur.itersize = batch_size
                cur.execute(sql)
                for row in cur:
                    annotations = row["annotations"] if isinstance(row["annotations"], list) else []
                    batch.append(
                        {
                            "sample_idx": row["sample_idx"],
                            "token": row["token"],
                            "scene_token": row["scene_token"],
                            "scene_name": row["scene_name"],
                            "location": row["location"],
                            "timestamp": int(row["timestamp"]),
                            "num_annotations": row["num_annotations"],
                            "annotation_categories": sorted(
                                {annotation.get("category", "") for annotation in annotations if annotation.get("category")}
                            ),
                            "annotations": _build_annotation_document(annotations),
                        }
                    )
                    if len(batch) >= batch_size:
                        total += self.bulk_index_documents(batch)
                        batch = []
        if batch:
            total += self.bulk_index_documents(batch)
        return total

    def bulk_index_multimodal_from_postgres(
        self,
        postgres_settings,
        minio_settings,
        encoder,
        *,
        batch_size: int = 24,
        limit: int | None = None,
        scene_limit: int | None = None,
    ) -> int:
        import psycopg
        from minio import Minio
        from psycopg.rows import dict_row

        self.ensure_index()
        client = Minio(
            minio_settings.endpoint,
            access_key=minio_settings.access_key,
            secret_key=minio_settings.secret_key,
            secure=minio_settings.secure,
        )

        total = 0
        documents: list[dict[str, Any]] = []
        with psycopg.connect(postgres_settings.dsn, row_factory=dict_row) as connection:
            with connection.cursor(name="es_multimodal_index_cursor") as cursor:
                cursor.itersize = max(1, min(batch_size, 128))
                cursor.execute(self._index_sql(limit=limit, scene_limit=scene_limit))
                for row in cursor:
                    annotations = row.get("annotations") or []
                    sample_record = {
                        "sample_idx": int(row["sample_idx"]),
                        "token": row.get("token", ""),
                        "scene_token": row.get("scene_token", ""),
                        "scene_name": row.get("scene_name", ""),
                        "location": row.get("location", ""),
                        "timestamp": int(row.get("timestamp") or 0),
                        "num_annotations": int(row.get("num_annotations") or 0),
                        "num_lidar_points": int(row.get("num_lidar_points") or 0),
                        "annotations": annotations,
                    }
                    vectors = encoder.encode_sample_payloads(
                        camera_payloads={
                            name: self._fetch_object(client, minio_settings.bucket, row[column])
                            for name, column in CAMERA_PATH_COLUMNS.items()
                            if row.get(column)
                        },
                        lidar_payload=(
                            self._fetch_object(client, minio_settings.bucket, row[LIDAR_PATH_COLUMN])
                            if row.get(LIDAR_PATH_COLUMN)
                            else None
                        ),
                        radar_payloads={
                            name: self._fetch_object(client, minio_settings.bucket, row[column])
                            for name, column in RADAR_PATH_COLUMNS.items()
                            if row.get(column)
                        },
                        metadata_text=build_metadata_text(sample_record),
                    )
                    documents.append(
                        {
                            "sample_idx": int(row["sample_idx"]),
                            "token": row.get("token", ""),
                            "scene_token": row.get("scene_token", ""),
                            "scene_name": row.get("scene_name", ""),
                            "location": row.get("location", ""),
                            "timestamp": int(row.get("timestamp") or 0),
                            "num_annotations": int(row.get("num_annotations") or 0),
                            "annotation_categories": sorted(
                                {annotation.get("category", "") for annotation in annotations if annotation.get("category")}
                            ),
                            "annotations": _build_annotation_document(annotations),
                            **vectors.as_document_fields(),
                        }
                    )
                    if len(documents) >= batch_size:
                        total += self.bulk_index_documents(documents)
                        documents = []
        if documents:
            total += self.bulk_index_documents(documents)
        return total

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

    def _fetch_object(self, client, bucket: str, object_path: str) -> bytes:
        response = client.get_object(bucket, object_path)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def _base_filters(
        self,
        *,
        scene_token: str,
        location: str,
        category: str,
        min_annotations: int,
    ) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []
        if scene_token:
            filters.append({"term": {"scene_token": scene_token}})
        if location:
            filters.append({"term": {"location": location}})
        if category:
            filters.append(
                {
                    "nested": {
                        "path": "annotations",
                        "query": {"term": {"annotations.category": category}},
                    }
                }
            )
        if min_annotations > 0:
            filters.append({"range": {"num_annotations": {"gte": min_annotations}}})
        return filters

    def search(
        self,
        q: str = "",
        scene_token: str = "",
        location: str = "",
        category: str = "",
        min_annotations: int = 0,
        size: int = 20,
        from_: int = 0,
    ) -> dict[str, Any]:
        normalized_q = q.strip()
        must: list[dict[str, Any]] = []
        filters = self._base_filters(
            scene_token=scene_token,
            location=location,
            category=category,
            min_annotations=min_annotations,
        )

        if normalized_q:
            wildcard_q = _escape_wildcard_value(normalized_q)
            annotation_query = normalized_q.replace(".", " ").replace("_", " ").strip()
            alias_categories = _category_aliases(normalized_q)
            match_query: dict[str, Any] = {
                "query": annotation_query,
                "operator": "AND",
                "boost": 4.0,
            }
            if len(annotation_query) >= 5:
                match_query["fuzziness"] = "AUTO"
            annotation_should: list[dict[str, Any]] = [
                {
                    "match": {
                        "annotations.category_text": match_query
                    }
                },
                {
                    "term": {
                        "annotations.category": {
                            "value": normalized_q.lower(),
                            "boost": 10.0,
                        }
                    }
                },
            ]
            lowered_q = normalized_q.lower()
            if lowered_q in _CATEGORY_GROUPS:
                annotation_should.append(
                    {
                        "term": {
                            "annotations.category_group": {
                                "value": lowered_q,
                                "boost": 8.0,
                            }
                        }
                    }
                )
            if _looks_structured_category_query(normalized_q):
                annotation_should.append(
                    {
                        "prefix": {
                            "annotations.category": {
                                "value": normalized_q.lower(),
                                "boost": 8.0,
                            }
                        }
                    }
                )
                annotation_should.append(
                    {
                        "match_phrase_prefix": {
                            "annotations.category_text": {
                                "query": annotation_query,
                                "boost": 6.0,
                            }
                        }
                    }
                )
            for alias in alias_categories:
                annotation_should.append(
                    {
                        "prefix": {
                            "annotations.category": {
                                "value": alias,
                                "boost": 8.0,
                            }
                        }
                    }
                )
                annotation_should.append(
                    {
                        "match_phrase": {
                            "annotations.category_text": {
                                "query": alias.replace(".", " "),
                                "boost": 6.0,
                            }
                        }
                    }
                )
            should: list[dict[str, Any]] = [
                {"term": {"scene_name": {"value": normalized_q, "boost": 8.0}}},
                {
                    "wildcard": {
                        "scene_name": {
                            "value": f"*{wildcard_q}*",
                            "case_insensitive": True,
                            "boost": 4.0,
                        }
                    }
                },
                {"term": {"token": {"value": normalized_q, "boost": 12.0}}},
                {"term": {"scene_token": {"value": normalized_q, "boost": 12.0}}},
                {
                    "wildcard": {
                        "location": {
                            "value": f"*{wildcard_q}*",
                            "case_insensitive": True,
                            "boost": 2.0,
                        }
                    }
                },
                {
                    "nested": {
                        "path": "annotations",
                        "score_mode": "max",
                        "query": {
                            "bool": {
                                "should": annotation_should,
                                "minimum_should_match": 1,
                            }
                        },
                    }
                },
            ]
            if len(normalized_q) >= 6 or _looks_like_identifier_query(normalized_q):
                should.extend(
                    [
                        {
                            "wildcard": {
                                "token": {
                                    "value": f"{wildcard_q}*",
                                    "case_insensitive": True,
                                    "boost": 6.0,
                                }
                            }
                        },
                        {
                            "wildcard": {
                                "scene_token": {
                                    "value": f"{wildcard_q}*",
                                    "case_insensitive": True,
                                    "boost": 6.0,
                                }
                            }
                        },
                    ]
                )
            must.append({"bool": {"should": should, "minimum_should_match": 1}})

        es_query = {
            "query": {
                "bool": {
                    "must": must or [{"match_all": {}}],
                    "filter": filters,
                }
            },
            "from": from_,
            "size": size,
            "sort": [{"_score": "desc"}, {"sample_idx": "asc"}],
            "_source": _source_fields(include_vectors=False),
            "aggs": {
                "top_categories": {
                    "nested": {"path": "annotations"},
                    "aggs": {
                        "cats": {"terms": {"field": "annotations.category", "size": 30}}
                    },
                },
                "locations": {"terms": {"field": "location", "size": 20}},
                "scenes": {
                    "terms": {"field": "scene_token", "size": 50},
                    "aggs": {
                        "scene_name": {"terms": {"field": "scene_name", "size": 1}},
                        "location": {"terms": {"field": "location", "size": 1}},
                        "top_score": {"max": {"script": {"source": "_score"}}},
                    },
                },
            },
        }

        result = self._req("POST", f"/{self.index}/_search", es_query)
        hits_raw = result.get("hits", {})
        hits = [self._hit_to_payload(hit, include_vectors=False) for hit in hits_raw.get("hits", [])]

        aggs = result.get("aggregations", {})
        cat_buckets = aggs.get("top_categories", {}).get("cats", {}).get("buckets", [])
        loc_buckets = aggs.get("locations", {}).get("buckets", [])
        scene_buckets = aggs.get("scenes", {}).get("buckets", [])
        scenes = []
        for bucket in scene_buckets:
            name_buckets = bucket.get("scene_name", {}).get("buckets", [])
            location_buckets = bucket.get("location", {}).get("buckets", [])
            scenes.append(
                {
                    "scene_token": bucket["key"],
                    "scene_name": name_buckets[0]["key"] if name_buckets else "",
                    "location": location_buckets[0]["key"] if location_buckets else "",
                    "sample_count": bucket["doc_count"],
                }
            )

        return {
            "total": hits_raw.get("total", {}).get("value", 0),
            "hits": hits,
            "aggs": {
                "categories": [
                    {"category": bucket["key"], "count": bucket["doc_count"]}
                    for bucket in cat_buckets
                ],
                "locations": [
                    {"location": bucket["key"], "count": bucket["doc_count"]}
                    for bucket in loc_buckets
                ],
                "scenes": scenes,
            },
        }

    def fetch_documents(
        self,
        sample_ids: list[int],
        *,
        include_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        if not sample_ids:
            return []
        body = {
            "docs": [
                {
                    "_index": self.index,
                    "_id": str(sample_idx),
                    "_source": _source_fields(include_vectors=include_vectors),
                }
                for sample_idx in sample_ids
            ]
        }
        result = self._req("POST", "/_mget", body)
        ordered: list[dict[str, Any]] = []
        for doc in result.get("docs", []):
            if not doc.get("found"):
                continue
            ordered.append(self._source_to_payload(doc.get("_source", {}), include_vectors=include_vectors))
        return ordered

    def vector_search(
        self,
        *,
        query_vector,
        scene_token: str = "",
        location: str = "",
        category: str = "",
        min_annotations: int = 0,
        size: int = 20,
        field: str = "fused_vec",
    ) -> dict[str, Any]:
        vector = [float(value) for value in query_vector]
        if not any(abs(value) > 1e-8 for value in vector):
            return {"total": 0, "hits": []}
        filters = self._base_filters(
            scene_token=scene_token,
            location=location,
            category=category,
            min_annotations=min_annotations,
        )
        query = {
            "size": size,
            "_source": _source_fields(include_vectors=False),
            "sort": [{"_score": "desc"}, {"sample_idx": "asc"}],
            "query": {
                "script_score": {
                    "query": {"bool": {"filter": filters}},
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, '{field}') + 1.0",
                        "params": {"query_vector": vector},
                    },
                }
            },
        }
        result = self._req("POST", f"/{self.index}/_search", query)
        hits_raw = result.get("hits", {})
        return {
            "total": hits_raw.get("total", {}).get("value", 0),
            "hits": [self._hit_to_payload(hit, include_vectors=False) for hit in hits_raw.get("hits", [])],
        }

    def _hit_to_payload(self, hit: dict[str, Any], *, include_vectors: bool) -> dict[str, Any]:
        payload = self._source_to_payload(hit.get("_source", {}), include_vectors=include_vectors)
        payload["score"] = float(hit.get("_score") or 0.0)
        return payload

    def _source_to_payload(self, source: dict[str, Any], *, include_vectors: bool) -> dict[str, Any]:
        annotations = source.get("annotations") or []
        categories = source.get("annotation_categories") or [
            annotation.get("category", "")
            for annotation in annotations
            if annotation.get("category")
        ]
        payload: dict[str, Any] = {
            "sample_idx": int(source.get("sample_idx") or 0),
            "token": source.get("token", ""),
            "scene_token": source.get("scene_token", ""),
            "scene_name": source.get("scene_name", ""),
            "location": source.get("location", ""),
            "timestamp": int(source.get("timestamp") or 0),
            "num_annotations": int(source.get("num_annotations") or 0),
            "categories": sorted(set(categories)),
            "annotation_categories": sorted(set(categories)),
            "annotations": annotations,
            "encoder_backend": source.get("encoder_backend", ""),
            "encoder_model": source.get("encoder_model", ""),
            "has_image": bool(source.get("has_image", False)),
            "has_lidar": bool(source.get("has_lidar", False)),
            "has_radar": bool(source.get("has_radar", False)),
            "has_metadata": bool(source.get("has_metadata", False)),
        }
        if include_vectors:
            for field in VECTOR_FIELDS:
                payload[field] = source.get(field) or []
        return payload

    def is_available(self) -> bool:
        try:
            self._req("GET", "/")
            return True
        except Exception:
            return False

    def doc_count(self) -> int:
        try:
            result = self._req("GET", f"/{self.index}/_count")
            return int(result.get("count", 0))
        except Exception:
            return 0
