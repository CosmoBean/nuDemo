from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

_INDEX_MAPPING: dict = {
    "mappings": {
        "properties": {
            "sample_idx":      {"type": "integer"},
            "token":           {"type": "keyword"},
            "scene_token":     {"type": "keyword"},
            "scene_name":      {"type": "keyword"},
            "location":        {"type": "keyword"},
            "timestamp":       {"type": "long"},
            "num_annotations": {"type": "integer"},
            "annotations": {
                "type": "nested",
                "properties": {
                    "category":       {"type": "keyword"},
                    "category_text":  {"type": "text"},
                    "category_group": {"type": "keyword"},
                    "num_lidar_pts":  {"type": "integer"},
                    "num_radar_pts":  {"type": "integer"},
                    "size_x":         {"type": "float"},
                    "size_y":         {"type": "float"},
                    "size_z":         {"type": "float"},
                },
            },
        }
    }
}


def _escape_wildcard_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("*", "\\*").replace("?", "\\?")


@dataclass(slots=True)
class ElasticsearchBackend:
    """Annotation search index backed by Elasticsearch.

    Stores one document per sample with a nested ``annotations`` array so that
    queries can target individual annotation fields (category, size, sensor
    point counts) while still returning the parent sample.
    """

    url: str
    index: str = "nudemo-annotations"

    # ── Internal HTTP helper ──────────────────────────────────────────────────

    def _req(self, method: str, path: str, body: object = None, *, ndjson: bool = False) -> dict:
        full_url = self.url.rstrip("/") + "/" + path.lstrip("/")
        data = None
        content_type = "application/x-ndjson" if ndjson else "application/json"
        if body is not None:
            data = (body if ndjson else json.dumps(body)).encode("utf-8")
        req = urllib.request.Request(full_url, data=data, method=method)
        req.add_header("Content-Type", content_type)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    # ── Index lifecycle ───────────────────────────────────────────────────────

    def ensure_index(self) -> None:
        """Create the index with the nested annotation mapping if absent."""
        try:
            self._req("HEAD", f"/{self.index}")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            self._req("PUT", f"/{self.index}", _INDEX_MAPPING)

    def clear(self) -> None:
        """Delete the index then recreate it."""
        try:
            self._req("DELETE", f"/{self.index}")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
        self.ensure_index()

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _build_doc(self, sample, sample_idx: int) -> dict:
        return {
            "sample_idx": sample_idx,
            "token": sample.token,
            "scene_token": sample.scene_token,
            "scene_name": sample.scene_name,
            "location": sample.location,
            "timestamp": int(sample.timestamp),
            "num_annotations": len(sample.annotations),
            "annotations": [
                {
                    "category":       ann.category,
                    "category_text":  ann.category.replace(".", " "),
                    "category_group": ann.category.split(".")[0],
                    "num_lidar_pts":  ann.num_lidar_pts,
                    "num_radar_pts":  ann.num_radar_pts,
                    "size_x":         float(ann.size[0]) if ann.size else 0.0,
                    "size_y":         float(ann.size[1]) if ann.size else 0.0,
                    "size_z":         float(ann.size[2]) if ann.size else 0.0,
                }
                for ann in sample.annotations
            ],
        }

    def bulk_index_from_postgres(self, postgres_settings, batch_size: int = 500) -> int:
        """Read all samples + annotations from PostgreSQL and bulk-index them.

        Uses a server-side cursor so the full table never loads into RAM.
        """
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
        batch: list[str] = []

        with psycopg.connect(postgres_settings.dsn, row_factory=psycopg.rows.dict_row) as conn:
            with conn.cursor(name="es_index_cursor") as cur:
                cur.itersize = batch_size
                cur.execute(sql)
                for row in cur:
                    anns = row["annotations"] if isinstance(row["annotations"], list) else []
                    doc = {
                        "sample_idx":      row["sample_idx"],
                        "token":           row["token"],
                        "scene_token":     row["scene_token"],
                        "scene_name":      row["scene_name"],
                        "location":        row["location"],
                        "timestamp":       int(row["timestamp"]),
                        "num_annotations": row["num_annotations"],
                        "annotations": [
                            {
                                "category":       a["category"],
                                "category_text":  a["category"].replace(".", " "),
                                "category_group": a["category"].split(".")[0],
                                "num_lidar_pts":  a["num_lidar_pts"] or 0,
                                "num_radar_pts":  a["num_radar_pts"] or 0,
                                "size_x":         float(a["size_x"] or 0),
                                "size_y":         float(a["size_y"] or 0),
                                "size_z":         float(a["size_z"] or 0),
                            }
                            for a in anns
                        ],
                    }
                    batch.append(
                        json.dumps(
                            {"index": {"_index": self.index, "_id": str(row["sample_idx"])}}
                        )
                    )
                    batch.append(json.dumps(doc))

                    if len(batch) >= batch_size * 2:
                        ndjson_body = "\n".join(batch) + "\n"
                        result = self._req("POST", "/_bulk", ndjson_body, ndjson=True)
                        if result.get("errors"):
                            errors = sum(
                                1
                                for item in result.get("items", [])
                                if "error" in item.get("index", {})
                            )
                            if errors:
                                raise RuntimeError(f"{errors} bulk index errors")
                        total += len(batch) // 2
                        batch = []

        if batch:
            ndjson_body = "\n".join(batch) + "\n"
            result = self._req("POST", "/_bulk", ndjson_body, ndjson=True)
            total += len(batch) // 2

        return total

    def bulk_index(self, samples_with_idx: list[tuple]) -> int:
        """Bulk index a batch of (sample, sample_idx) pairs. Returns count indexed."""
        lines: list[str] = []
        for sample, sample_idx in samples_with_idx:
            lines.append(json.dumps({"index": {"_index": self.index, "_id": str(sample_idx)}}))
            lines.append(json.dumps(self._build_doc(sample, sample_idx)))
        if not lines:
            return 0
        ndjson_body = "\n".join(lines) + "\n"
        result = self._req("POST", "/_bulk", ndjson_body, ndjson=True)
        if result.get("errors"):
            errors = sum(
                1 for item in result.get("items", []) if "error" in item.get("index", {})
            )
            if errors:
                raise RuntimeError(f"{errors}/{len(lines)//2} bulk index errors")
        return len(lines) // 2

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        q: str = "",
        scene_token: str = "",
        location: str = "",
        category: str = "",
        min_annotations: int = 0,
        size: int = 20,
        from_: int = 0,
    ) -> dict:
        normalized_q = q.strip()
        must: list = []
        filter_: list = []

        if normalized_q:
            wildcard_q = _escape_wildcard_value(normalized_q)
            should: list[dict] = [
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
                                "should": [
                                    {
                                        "wildcard": {
                                            "annotations.category": {
                                                "value": f"*{wildcard_q}*",
                                                "case_insensitive": True,
                                            }
                                        }
                                    },
                                    {
                                        "wildcard": {
                                            "annotations.category_group": {
                                                "value": f"*{wildcard_q}*",
                                                "case_insensitive": True,
                                            }
                                        }
                                    },
                                    {
                                        "match": {
                                            "annotations.category_text": {
                                                "query": normalized_q,
                                                "fuzziness": "AUTO",
                                            }
                                        }
                                    },
                                ]
                            }
                        },
                    }
                },
            ]
            if len(normalized_q) >= 6:
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
            must.append({
                "bool": {
                    "should": should,
                    "minimum_should_match": 1,
                }
            })

        if scene_token:
            filter_.append({"term": {"scene_token": scene_token}})
        if location:
            filter_.append({"term": {"location": location}})
        if category:
            filter_.append(
                {
                    "nested": {
                        "path": "annotations",
                        "query": {"term": {"annotations.category": category}},
                    }
                }
            )
        if min_annotations > 0:
            filter_.append({"range": {"num_annotations": {"gte": min_annotations}}})

        es_query = {
            "query": {
                "bool": {
                    "must":   must or [{"match_all": {}}],
                    "filter": filter_,
                }
            },
            "from": from_,
            "size": size,
            "sort": [{"_score": "desc"}, {"sample_idx": "asc"}],
            "_source": [
                "sample_idx",
                "token",
                "scene_token",
                "scene_name",
                "location",
                "num_annotations",
                "annotations.category",
            ],
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
                        "location":   {"terms": {"field": "location",   "size": 1}},
                        "top_score":  {"max": {"script": {"source": "_score"}}},
                    },
                },
            },
        }

        result = self._req("POST", f"/{self.index}/_search", es_query)
        hits_raw = result.get("hits", {})

        hits = []
        for hit in hits_raw.get("hits", []):
            src = hit["_source"]
            cats = list({a["category"] for a in src.get("annotations", [])})
            hits.append({
                "sample_idx":     src["sample_idx"],
                "token":          src.get("token", ""),
                "scene_token":    src.get("scene_token", ""),
                "scene_name":     src.get("scene_name", ""),
                "location":       src.get("location", ""),
                "num_annotations": src.get("num_annotations", 0),
                "categories":     cats,
                "score":          hit.get("_score") or 0.0,
            })

        aggs = result.get("aggregations", {})
        cat_buckets = aggs.get("top_categories", {}).get("cats", {}).get("buckets", [])
        loc_buckets = aggs.get("locations", {}).get("buckets", [])
        scene_buckets = aggs.get("scenes", {}).get("buckets", [])

        scenes = []
        for b in scene_buckets:
            name_buckets = b.get("scene_name", {}).get("buckets", [])
            loc_buckets_inner = b.get("location", {}).get("buckets", [])
            scenes.append({
                "scene_token":  b["key"],
                "scene_name":   name_buckets[0]["key"] if name_buckets else "",
                "location":     loc_buckets_inner[0]["key"] if loc_buckets_inner else "",
                "sample_count": b["doc_count"],
            })

        return {
            "total": hits_raw.get("total", {}).get("value", 0),
            "hits":  hits,
            "aggs": {
                "categories": [
                    {"category": b["key"], "count": b["doc_count"]} for b in cat_buckets
                ],
                "locations": [
                    {"location": b["key"], "count": b["doc_count"]} for b in loc_buckets
                ],
                "scenes": scenes,
            },
        }

    # ── Health ────────────────────────────────────────────────────────────────

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
