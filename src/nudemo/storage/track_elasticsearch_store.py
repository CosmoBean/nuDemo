from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

_TRACK_INDEX_MAPPING = {
    "settings": {
        "analysis": {
            "normalizer": {
                "folding": {
                    "type": "custom",
                    "char_filter": [],
                    "filter": ["lowercase", "asciifolding"],
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "track_id": {"type": "keyword", "normalizer": "folding"},
            "scene_token": {"type": "keyword", "normalizer": "folding"},
            "scene_name": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "normalizer": "folding"}},
            },
            "location": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "normalizer": "folding"}},
            },
            "category": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "normalizer": "folding"}},
            },
            "summary_text": {"type": "text"},
            "start_timestamp": {"type": "long"},
            "end_timestamp": {"type": "long"},
            "sample_ids": {"type": "integer"},
            "sample_count": {"type": "integer"},
            "annotation_count": {"type": "integer"},
            "avg_num_lidar_pts": {"type": "float"},
            "avg_num_radar_pts": {"type": "float"},
            "max_num_lidar_pts": {"type": "integer"},
            "max_num_radar_pts": {"type": "integer"},
            "visibility_tokens": {"type": "keyword"},
            "preview_sample_idx": {"type": "integer"},
        }
    },
}


def _escape_wildcard_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("*", "\\*").replace("?", "\\?")


@dataclass(slots=True)
class TrackElasticsearchBackend:
    url: str
    index: str = "nudemo-tracks"

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
            self._req("PUT", f"/{self.index}", _TRACK_INDEX_MAPPING)

    def clear(self) -> None:
        try:
            self._req("DELETE", f"/{self.index}")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
        self.ensure_index()

    def bulk_index_documents(self, documents: list[dict[str, Any]]) -> int:
        if not documents:
            return 0
        self.ensure_index()
        lines: list[str] = []
        for doc in documents:
            track_id = str(doc["track_id"])
            lines.append(json.dumps({"index": {"_index": self.index, "_id": track_id}}))
            lines.append(json.dumps(doc))
        result = self._req("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
        if result.get("errors"):
            errors = sum(1 for item in result.get("items", []) if "error" in item.get("index", {}))
            if errors:
                raise RuntimeError(f"{errors}/{len(documents)} bulk index errors")
        return len(documents)

    def bulk_index_from_postgres(self, postgres_settings, *, batch_size: int = 250) -> int:
        import psycopg
        from psycopg.rows import dict_row

        total = 0
        batch: list[dict[str, Any]] = []
        with psycopg.connect(postgres_settings.dsn, row_factory=dict_row) as connection:
            with connection.cursor(name="track_es_cursor") as cursor:
                cursor.itersize = max(1, min(batch_size, 512))
                cursor.execute(
                    """
                    SELECT
                        track_id,
                        scene_token,
                        scene_name,
                        location,
                        category,
                        start_timestamp,
                        end_timestamp,
                        sample_ids,
                        sample_count,
                        annotation_count,
                        avg_num_lidar_pts,
                        avg_num_radar_pts,
                        max_num_lidar_pts,
                        max_num_radar_pts,
                        visibility_tokens,
                        sample_ids[1] AS preview_sample_idx
                    FROM tracks
                    ORDER BY start_timestamp, track_id
                    """
                )
                for row in cursor:
                    batch.append(self._row_to_doc(dict(row)))
                    if len(batch) >= batch_size:
                        total += self.bulk_index_documents(batch)
                        batch = []
        if batch:
            total += self.bulk_index_documents(batch)
        return total

    def search(
        self,
        *,
        q: str = "",
        scene_token: str = "",
        location: str = "",
        category: str = "",
        size: int = 20,
        from_: int = 0,
    ) -> dict[str, Any]:
        normalized_q = q.strip()
        filters: list[dict[str, Any]] = []
        if scene_token:
            filters.append({"term": {"scene_token": scene_token}})
        if location:
            filters.append({"term": {"location.keyword": location}})
        if category:
            filters.append({"term": {"category.keyword": category}})

        must: list[dict[str, Any]] = []
        if normalized_q:
            wildcard_q = _escape_wildcard_value(normalized_q)
            should = [
                {"term": {"track_id": {"value": normalized_q, "boost": 12.0}}},
                {"term": {"scene_token": {"value": normalized_q, "boost": 10.0}}},
                {"term": {"scene_name.keyword": {"value": normalized_q, "boost": 8.0}}},
                {
                    "wildcard": {
                        "scene_name.keyword": {
                            "value": f"*{wildcard_q}*",
                            "case_insensitive": True,
                            "boost": 5.0,
                        }
                    }
                },
                {
                    "wildcard": {
                        "track_id": {
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
                {"match": {"category": {"query": normalized_q, "operator": "AND", "boost": 4.0}}},
                {
                    "match": {
                        "summary_text": {
                            "query": normalized_q,
                            "operator": "AND",
                            "boost": 3.0,
                        }
                    }
                },
                {
                    "wildcard": {
                        "location.keyword": {
                            "value": f"*{wildcard_q}*",
                            "case_insensitive": True,
                            "boost": 2.0,
                        }
                    }
                },
            ]
            must.append({"bool": {"should": should, "minimum_should_match": 1}})

        result = self._req(
            "POST",
            f"/{self.index}/_search",
            {
                "query": {"bool": {"must": must or [{"match_all": {}}], "filter": filters}},
                "from": from_,
                "size": size,
                "sort": [{"_score": "desc"}, {"sample_count": "desc"}, {"start_timestamp": "asc"}],
                "_source": True,
            },
        )
        hits_raw = result.get("hits", {})
        hits = [self._hit_to_payload(hit) for hit in hits_raw.get("hits", [])]
        return {"total": hits_raw.get("total", {}).get("value", 0), "hits": hits}

    def fetch_documents(self, track_ids: list[str]) -> list[dict[str, Any]]:
        if not track_ids:
            return []
        result = self._req(
            "POST",
            "/_mget",
            {
                "docs": [
                    {"_index": self.index, "_id": track_id, "_source": True}
                    for track_id in track_ids
                ]
            },
        )
        ordered: list[dict[str, Any]] = []
        for doc in result.get("docs", []):
            if not doc.get("found"):
                continue
            ordered.append(self._source_to_payload(doc.get("_source", {})))
        return ordered

    def _hit_to_payload(self, hit: dict[str, Any]) -> dict[str, Any]:
        payload = self._source_to_payload(hit.get("_source", {}))
        payload["score"] = float(hit.get("_score") or 0.0)
        return payload

    @staticmethod
    def _source_to_payload(source: dict[str, Any]) -> dict[str, Any]:
        return {
            "track_id": str(source.get("track_id") or ""),
            "scene_token": str(source.get("scene_token") or ""),
            "scene_name": str(source.get("scene_name") or ""),
            "location": str(source.get("location") or ""),
            "category": str(source.get("category") or ""),
            "start_timestamp": int(source.get("start_timestamp") or 0),
            "end_timestamp": int(source.get("end_timestamp") or 0),
            "sample_ids": [int(value) for value in (source.get("sample_ids") or [])],
            "sample_count": int(source.get("sample_count") or 0),
            "annotation_count": int(source.get("annotation_count") or 0),
            "avg_num_lidar_pts": float(source.get("avg_num_lidar_pts") or 0.0),
            "avg_num_radar_pts": float(source.get("avg_num_radar_pts") or 0.0),
            "max_num_lidar_pts": int(source.get("max_num_lidar_pts") or 0),
            "max_num_radar_pts": int(source.get("max_num_radar_pts") or 0),
            "visibility_tokens": list(source.get("visibility_tokens") or []),
            "preview_sample_idx": (
                int(source.get("preview_sample_idx"))
                if source.get("preview_sample_idx") is not None
                else None
            ),
        }

    @staticmethod
    def _row_to_doc(row: dict[str, Any]) -> dict[str, Any]:
        track_id = str(row.get("track_id") or "")
        scene_name = str(row.get("scene_name") or "")
        category = str(row.get("category") or "")
        location = str(row.get("location") or "")
        scene_token = str(row.get("scene_token") or "")
        return {
            "track_id": track_id,
            "scene_token": scene_token,
            "scene_name": scene_name,
            "location": location,
            "category": category,
            "summary_text": " ".join(
                part
                for part in [track_id, scene_name, scene_token, location, category]
                if part
            ),
            "start_timestamp": int(row.get("start_timestamp") or 0),
            "end_timestamp": int(row.get("end_timestamp") or 0),
            "sample_ids": [int(value) for value in (row.get("sample_ids") or [])],
            "sample_count": int(row.get("sample_count") or 0),
            "annotation_count": int(row.get("annotation_count") or 0),
            "avg_num_lidar_pts": float(row.get("avg_num_lidar_pts") or 0.0),
            "avg_num_radar_pts": float(row.get("avg_num_radar_pts") or 0.0),
            "max_num_lidar_pts": int(row.get("max_num_lidar_pts") or 0),
            "max_num_radar_pts": int(row.get("max_num_radar_pts") or 0),
            "visibility_tokens": list(row.get("visibility_tokens") or []),
            "preview_sample_idx": (
                int(row.get("preview_sample_idx"))
                if row.get("preview_sample_idx") is not None
                else None
            ),
        }

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
