# ruff: noqa: E501

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from nudemo.benchmarks.export import load_report
from nudemo.benchmarks.models import BenchmarkReport, BenchmarkResult
from nudemo.config import AppConfig
from nudemo.explorer.media import (
    lidar_payload_to_point_cloud,
    lidar_payload_to_svg,
    process_camera_payload,
)
from nudemo.observability import ensure_metrics_exporter, install_http_metrics
from nudemo.reporting.dashboard import build_recommendation_summary

CAMERA_COLUMN_MAP = {
    "CAM_FRONT": "cam_front_path",
    "CAM_FRONT_LEFT": "cam_front_left_path",
    "CAM_FRONT_RIGHT": "cam_front_right_path",
    "CAM_BACK": "cam_back_path",
    "CAM_BACK_LEFT": "cam_back_left_path",
    "CAM_BACK_RIGHT": "cam_back_right_path",
}

SENSOR_COLUMN_MAP = {
    "LIDAR_TOP": "lidar_top_path",
    "RADAR_FRONT": "radar_front_path",
    "RADAR_FRONT_LEFT": "radar_front_left_path",
    "RADAR_FRONT_RIGHT": "radar_front_right_path",
    "RADAR_BACK_LEFT": "radar_back_left_path",
    "RADAR_BACK_RIGHT": "radar_back_right_path",
}


class ExplorerStore:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def _connection(self):
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self._config.services.postgres.dsn, row_factory=dict_row)

    def _minio(self):
        from minio import Minio

        settings = self._config.services.minio
        return Minio(
            settings.endpoint,
            access_key=settings.access_key,
            secret_key=settings.secret_key,
            secure=settings.secure,
        )

    def fetch_summary(self) -> dict[str, Any]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM samples) AS sample_count,
                    (SELECT COUNT(*) FROM annotations) AS annotation_count,
                    (SELECT COUNT(*) FROM scenes) AS scene_count,
                    (SELECT COALESCE(SUM(num_lidar_points), 0) FROM samples) AS lidar_points
                """
            )
            summary = dict(cursor.fetchone() or {})

            cursor.execute(
                """
                SELECT location, COUNT(*) AS count
                FROM samples
                GROUP BY location
                ORDER BY count DESC, location
                LIMIT 8
                """
            )
            summary["top_locations"] = list(cursor.fetchall())

            cursor.execute(
                """
                SELECT category, COUNT(*) AS count
                FROM annotations
                GROUP BY category
                ORDER BY count DESC, category
                LIMIT 12
                """
            )
            summary["top_categories"] = list(cursor.fetchall())
        return summary

    def fetch_filters(self) -> dict[str, Any]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT location FROM samples ORDER BY location")
            locations = [row["location"] for row in cursor.fetchall()]

            cursor.execute("SELECT DISTINCT category FROM annotations ORDER BY category")
            categories = [row["category"] for row in cursor.fetchall()]

            cursor.execute(
                """
                SELECT scene_token, scene_name, location, num_samples
                FROM scenes
                ORDER BY scene_name, location
                """
            )
            scenes = [dict(row) for row in cursor.fetchall()]

        return {"locations": locations, "categories": categories, "scenes": scenes}

    def search_samples(
        self,
        *,
        q: str | None,
        scene_token: str | None,
        location: str | None,
        category: str | None,
        min_annotations: int,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        pattern = f"%{q.strip()}%" if q and q.strip() else None
        effective_limit = max(1, min(limit, 100))
        effective_offset = max(0, offset)
        effective_min_annotations = max(0, min_annotations)
        params = {
            "scene_token": scene_token or None,
            "location": location or None,
            "category": category or None,
            "pattern": pattern,
            "min_annotations": effective_min_annotations,
            "limit": effective_limit,
            "offset": effective_offset,
        }
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                WITH filtered AS (
                    SELECT
                        s.sample_idx,
                        s.token,
                        s.scene_token,
                        COALESCE(sc.scene_name, s.scene_token) AS scene_name,
                        s.timestamp,
                        s.location,
                        s.num_annotations,
                        s.num_lidar_points,
                        s.cam_front_path,
                        ARRAY_REMOVE(
                            ARRAY_AGG(DISTINCT a.category ORDER BY a.category),
                            NULL
                        ) AS annotation_categories
                    FROM samples s
                    LEFT JOIN scenes sc ON sc.scene_token = s.scene_token
                    LEFT JOIN annotations a ON a.sample_idx = s.sample_idx
                    WHERE (%(scene_token)s::text IS NULL OR s.scene_token = %(scene_token)s::text)
                      AND (%(location)s::text IS NULL OR s.location = %(location)s::text)
                      AND (
                        %(min_annotations)s = 0
                        OR s.num_annotations >= %(min_annotations)s
                      )
                      AND (
                        %(category)s::text IS NULL
                        OR EXISTS (
                            SELECT 1
                            FROM annotations a2
                            WHERE a2.sample_idx = s.sample_idx
                              AND a2.category = %(category)s::text
                        )
                      )
                      AND (
                        %(pattern)s::text IS NULL
                        OR s.token ILIKE %(pattern)s::text
                        OR COALESCE(sc.scene_name, s.scene_token) ILIKE %(pattern)s::text
                        OR s.location ILIKE %(pattern)s::text
                        OR EXISTS (
                            SELECT 1
                            FROM annotations a3
                            WHERE a3.sample_idx = s.sample_idx
                              AND a3.category ILIKE %(pattern)s::text
                        )
                      )
                    GROUP BY
                        s.sample_idx,
                        s.token,
                        s.scene_token,
                        sc.scene_name,
                        s.timestamp,
                        s.location,
                        s.num_annotations,
                        s.num_lidar_points,
                        s.cam_front_path
                ),
                counted AS (
                    SELECT filtered.*, COUNT(*) OVER() AS total_count
                    FROM filtered
                )
                SELECT *
                FROM counted
                ORDER BY sample_idx
                LIMIT %(limit)s
                OFFSET %(offset)s
                """,
                params,
            )
            rows = [dict(row) for row in cursor.fetchall()]

        total = int(rows[0]["total_count"]) if rows else 0
        items = []
        for row in rows:
            categories = row.get("annotation_categories") or []
            items.append(
                {
                    "sample_idx": row["sample_idx"],
                    "token": row["token"],
                    "scene_token": row["scene_token"],
                    "scene_name": row["scene_name"],
                    "timestamp": row["timestamp"],
                    "location": row["location"],
                    "num_annotations": row["num_annotations"],
                    "num_lidar_points": row["num_lidar_points"],
                    "annotation_categories": categories,
                    "preview_url": (
                        f"/api/samples/{row['sample_idx']}/cameras/CAM_FRONT"
                        if row.get("cam_front_path")
                        else None
                    ),
                }
            )
        return {
            "total": total,
            "limit": effective_limit,
            "offset": effective_offset,
            "items": items,
        }

    def fetch_sample_detail(self, sample_idx: int) -> dict[str, Any] | None:
        column_names = (*CAMERA_COLUMN_MAP.values(), *SENSOR_COLUMN_MAP.values())
        column_select = ", ".join(
            f"s.{column} AS {column}" for column in column_names
        )
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    s.sample_idx,
                    s.token,
                    s.scene_token,
                    COALESCE(sc.scene_name, s.scene_token) AS scene_name,
                    s.timestamp,
                    s.location,
                    s.ego_translation,
                    s.ego_rotation,
                    s.num_annotations,
                    s.num_lidar_points,
                    {column_select}
                FROM samples s
                LEFT JOIN scenes sc ON sc.scene_token = s.scene_token
                WHERE s.sample_idx = %s
                """,
                (sample_idx,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            cursor.execute(
                """
                SELECT category, translation, size, rotation, num_lidar_pts, num_radar_pts
                FROM annotations
                WHERE sample_idx = %s
                ORDER BY category
                """,
                (sample_idx,),
            )
            annotations = [dict(annotation) for annotation in cursor.fetchall()]

        detail = dict(row)
        detail["annotations"] = annotations
        detail["camera_urls"] = {
            camera: (
                f"/api/samples/{sample_idx}/cameras/{camera}"
                if detail.get(column_name)
                else None
            )
            for camera, column_name in CAMERA_COLUMN_MAP.items()
        }
        detail["camera_paths"] = {
            camera: detail.get(column_name) for camera, column_name in CAMERA_COLUMN_MAP.items()
        }
        detail["sensor_paths"] = {
            sensor: detail.get(column_name) for sensor, column_name in SENSOR_COLUMN_MAP.items()
        }
        return detail

    def fetch_scene_samples(self, scene_token: str, *, limit: int = 18) -> list[dict[str, Any]]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    s.sample_idx,
                    s.token,
                    s.scene_token,
                    COALESCE(sc.scene_name, s.scene_token) AS scene_name,
                    s.location,
                    s.timestamp,
                    s.num_annotations,
                    s.num_lidar_points,
                    s.cam_front_path
                FROM samples s
                LEFT JOIN scenes sc ON sc.scene_token = s.scene_token
                WHERE s.scene_token = %s
                ORDER BY s.timestamp, s.sample_idx
                LIMIT %s
                """,
                (scene_token, max(1, min(limit, 240))),
            )
            rows = [dict(row) for row in cursor.fetchall()]
        return [
            {
                "sample_idx": row["sample_idx"],
                "token": row["token"],
                "scene_token": row["scene_token"],
                "scene_name": row["scene_name"],
                "location": row["location"],
                "timestamp": row["timestamp"],
                "num_annotations": row["num_annotations"],
                "num_lidar_points": row["num_lidar_points"],
                "preview_url": (
                    f"/api/samples/{row['sample_idx']}/cameras/CAM_FRONT"
                    if row.get("cam_front_path")
                    else None
                ),
            }
            for row in rows
        ]

    def fetch_scene_detail(self, scene_token: str, *, limit: int = 180) -> dict[str, Any] | None:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT scene_token, scene_name, location, num_samples
                FROM scenes
                WHERE scene_token = %s
                """,
                (scene_token,),
            )
            scene = cursor.fetchone()
            if scene is None:
                return None

        detail = dict(scene)
        detail["samples"] = self.fetch_scene_samples(scene_token, limit=max(1, min(limit, 240)))
        return detail

    def fetch_sensor_bytes(self, sample_idx: int, sensor: str) -> bytes:
        column_name = SENSOR_COLUMN_MAP.get(sensor)
        if column_name is None:
            raise KeyError(sensor)

        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT {column_name} AS object_path FROM samples WHERE sample_idx = %s",
                (sample_idx,),
            )
            row = cursor.fetchone()
            if row is None or row["object_path"] is None:
                raise FileNotFoundError(sensor)
            object_path = row["object_path"]

        client = self._minio()
        response = client.get_object(self._config.services.minio.bucket, object_path)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def fetch_camera_bytes(self, sample_idx: int, camera: str) -> bytes:
        column_name = CAMERA_COLUMN_MAP.get(camera)
        if column_name is None:
            raise KeyError(camera)

        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT {column_name} AS object_path FROM samples WHERE sample_idx = %s",
                (sample_idx,),
            )
            row = cursor.fetchone()
            if row is None or row["object_path"] is None:
                raise FileNotFoundError(camera)
            object_path = row["object_path"]

        client = self._minio()
        response = client.get_object(self._config.services.minio.bucket, object_path)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()


class BenchmarkReportStore:
    def __init__(self, reports_root: Path) -> None:
        self._reports_root = reports_root

    def fetch_summary(self) -> dict[str, Any]:
        report_path = self._reports_root / "benchmark_report.json"
        if not report_path.exists():
            raise FileNotFoundError(report_path)

        report = load_report(report_path)
        format_rows = [
            self._build_backend_summary(report, backend) for backend in self._storage_backends(report)
        ]
        return {
            "suite_name": report.suite_name,
            "created_at": report.created_at,
            "dataset": report.dataset,
            "recommendations": build_recommendation_summary(report),
            "storage_formats": format_rows,
        }

    def _storage_backends(self, report: BenchmarkReport) -> list[str]:
        ordered = []
        seen: set[str] = set()
        for result in report.results:
            if result.backend == "Kafka" or result.backend == "real":
                continue
            if result.backend in seen:
                continue
            ordered.append(result.backend)
            seen.add(result.backend)
        return ordered

    def _build_backend_summary(self, report: BenchmarkReport, backend: str) -> dict[str, Any]:
        write_result = self._match(report, backend, "write_throughput")
        scan_result = self._match(report, backend, "sequential_scan")
        random_result = self._match(report, backend, "random_access")
        disk_result = self._match(report, backend, "disk_footprint")
        curation_result = self._match(report, backend, "curation_query")
        return {
            "backend": backend,
            "write_samples_per_sec": self._metric(
                write_result, "throughput_samples_per_sec", "throughput"
            ),
            "sequential_samples_per_sec": self._metric(
                scan_result, "throughput_mean", "throughput_samples_per_sec"
            ),
            "random_access_p50_ms": self._metric(random_result, "latency_p50_ms"),
            "disk_mb": self._metric(disk_result, "disk_mb"),
            "curation_query_ms": self._metric(
                curation_result, "query_time_ms_mean", "query_time_ms"
            ),
            "status": self._status(write_result, scan_result, random_result, disk_result),
        }

    @staticmethod
    def _match(report: BenchmarkReport, backend: str, pattern: str) -> BenchmarkResult | None:
        for result in report.results:
            if result.backend == backend and result.pattern == pattern:
                return result
        return None

    @staticmethod
    def _metric(result: BenchmarkResult | None, *keys: str) -> float | None:
        if result is None:
            return None
        for key in keys:
            value = result.metrics.get(key)
            if isinstance(value, int | float):
                return float(value)
        return None

    @staticmethod
    def _status(*results: BenchmarkResult | None) -> str:
        statuses = [result.status for result in results if result is not None]
        if not statuses:
            return "missing"
        if any(status != "ok" for status in statuses):
            return "degraded"
        return "ok"


def build_browser_home_html() -> str:
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>nuDemo Browser</title>
    <style>
      :root {
        --bg: #0b0b10;
        --panel: #161622;
        --line: #544bb0;
        --ink: #e7e8f3;
        --muted: #b5b0cf;
        --accent: #6b59dd;
        --accent-alt: #4a3da0;
        --accent-soft: #201f31;
        --shadow: 6px 6px 0 #211b52;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Space Grotesk", "IBM Plex Sans", "Avenir Next", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top right, rgba(84, 54, 252, 0.12), transparent 24%),
          linear-gradient(180deg, #0f0f16 0%, var(--bg) 100%);
      }
      main {
        max-width: 1120px;
        margin: 0 auto;
        padding: 48px 20px 64px;
      }
      .hero {
        background: var(--panel);
        border: 3px solid var(--line);
        border-radius: 28px;
        box-shadow: var(--shadow);
        overflow: hidden;
      }
      .hero-inner {
        display: grid;
        grid-template-columns: 1.3fr 1fr;
        gap: 24px;
        padding: 32px;
      }
      h1 {
        margin: 0 0 12px;
        font-size: clamp(2rem, 4vw, 3.4rem);
        line-height: 1.05;
        letter-spacing: -0.04em;
      }
      p {
        margin: 0;
        color: var(--muted);
        line-height: 1.6;
      }
      .pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 12px;
        border-radius: 999px;
        background: var(--accent);
        color: var(--ink);
        border: 2px solid var(--line);
        box-shadow: 4px 4px 0 #211b52;
        font-size: 0.9rem;
        margin-bottom: 16px;
        font-weight: 700;
        text-transform: uppercase;
      }
      .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
      }
      .card {
        background: var(--panel);
        border: 3px solid var(--line);
        border-radius: 22px;
        padding: 18px;
        box-shadow: var(--shadow);
      }
      .card strong {
        display: block;
        margin-bottom: 8px;
        font-size: 1.15rem;
      }
      .card a {
        display: inline-block;
        margin-top: 14px;
        color: var(--ink);
        font-weight: 600;
        text-decoration: none;
        border-bottom: 2px solid var(--line);
      }
      .card a:hover { text-decoration: underline; }
      .links {
        display: grid;
        gap: 12px;
      }
      .link-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        padding: 14px 16px;
        border-radius: 18px;
        background: #201f31;
        border: 3px solid var(--line);
        box-shadow: 4px 4px 0 #211b52;
      }
      .link-row a {
        color: var(--ink);
        font-weight: 600;
        text-decoration: none;
        white-space: nowrap;
      }
      .section-title {
        margin: 28px 0 12px;
        font-size: 1rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--ink);
        font-weight: 800;
      }
      @media (max-width: 860px) {
        .hero-inner { grid-template-columns: 1fr; }
        .link-row { align-items: flex-start; flex-direction: column; }
      }
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <div class="hero-inner">
          <div>
            <div class="pill">Ingested data browser</div>
            <h1>Search the pipeline output, inspect samples, and keep the benchmark views close.</h1>
            <p>
              This browser combines a searchable view of the ingested PostgreSQL and MinIO records
              with the benchmark and telemetry dashboards already generated by the pipeline.
            </p>
          </div>
          <div class="links">
            <div class="link-row">
              <span>Search and preview ingested samples</span>
              <a href="/explorer">Open explorer</a>
            </div>
            <div class="link-row">
              <span>Scene player with 3D LiDAR rendering</span>
              <a href="/scene-studio">Open scene studio</a>
            </div>
            <div class="link-row">
              <span>Benchmark comparison dashboard</span>
              <a href="/benchmark_dashboard.html">Open benchmark</a>
            </div>
            <div class="link-row">
              <span>Telemetry and bottleneck dashboard</span>
              <a href="/telemetry_dashboard.html">Open telemetry</a>
            </div>
            <div class="link-row">
              <span>Grafana observability workspace</span>
              <a href="/grafana/">Open Grafana</a>
            </div>
            <div class="link-row">
              <span>Prometheus scrape and query UI</span>
              <a href="/prometheus/">Open Prometheus</a>
            </div>
            <div class="link-row">
              <span>Static report artifact index</span>
              <a href="/index.html">Open reports</a>
            </div>
          </div>
        </div>
      </section>

      <div class="section-title">Views</div>
      <div class="card-grid">
        <section class="card">
          <strong>Explorer</strong>
          Search by token, scene, location, or annotation category. Compare stored formats against
          Parquet, inspect processed camera images, and preview LiDAR geometry for each sample.
          <a href="/explorer">Go to explorer</a>
        </section>
        <section class="card">
          <strong>Scene Studio</strong>
          Focus on one scene at a time with a scrubber, live sample switching, and a 3D LiDAR
          point-cloud viewer driven by the stored nuScenes payloads.
          <a href="/scene-studio">Open studio</a>
        </section>
        <section class="card">
          <strong>Benchmark Dashboard</strong>
          Compare backend write, read, random access, and dataloader behavior from the most recent
          report bundle.
          <a href="/benchmark_dashboard.html">Open dashboard</a>
        </section>
        <section class="card">
          <strong>Telemetry Dashboard</strong>
          Review span timings, service peaks, and pipeline choke points captured during live runs.
          <a href="/telemetry_dashboard.html">Open telemetry</a>
        </section>
        <section class="card">
          <strong>Grafana + Prometheus</strong>
          Inspect the OpenTelemetry-backed Prometheus metrics derived from the latest benchmark and
          service snapshots.
          <a href="/grafana/">Open observability</a>
        </section>
      </div>
    </main>
  </body>
</html>
"""


def build_explorer_html() -> str:
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>nuDemo Explorer</title>
    <style>
      :root {
        --bg: #0b0b10;
        --panel: #161622;
        --line: #544bb0;
        --ink: #e7e8f3;
        --muted: #b5b0cf;
        --accent: #6b59dd;
        --accent-alt: #4a3da0;
        --accent-soft: #201f31;
        --success-soft: #2a2938;
        --shadow: 6px 6px 0 #211b52;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Space Grotesk", "IBM Plex Sans", "Avenir Next", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top right, rgba(84, 54, 252, 0.12), transparent 20%),
          linear-gradient(180deg, #0f0f16 0%, var(--bg) 100%);
        overflow-x: hidden;
      }
      main {
        max-width: 1480px;
        margin: 0 auto;
        padding: 28px clamp(12px, 1.8vw, 22px) 48px;
      }
      .shell {
        display: grid;
        grid-template-columns: minmax(280px, 320px) minmax(0, 1fr) minmax(320px, 380px);
        gap: 18px;
        align-items: start;
      }
      .panel {
        background: var(--panel);
        border: 3px solid var(--line);
        border-radius: 24px;
        box-shadow: var(--shadow);
        min-width: 0;
      }
      .sidebar, .detail {
        padding: 20px;
        align-self: start;
        position: sticky;
        top: 18px;
      }
      .content {
        min-width: 0;
      }
      h1 {
        margin: 0 0 8px;
        font-size: 2rem;
        line-height: 1;
        letter-spacing: -0.04em;
      }
      h2, h3 {
        margin: 0 0 10px;
      }
      p, label, span, td, th {
        color: var(--muted);
      }
      .subnav {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 18px;
      }
      .subnav a {
        color: var(--ink);
        font-weight: 600;
        text-decoration: none;
        padding: 10px 12px;
        border: 3px solid var(--line);
        border-radius: 999px;
        background: var(--accent-soft);
        box-shadow: 4px 4px 0 #211b52;
      }
      .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 14px;
        margin-bottom: 18px;
      }
      .compare-panel {
        padding: 18px;
        margin-bottom: 18px;
      }
      .compare-panel p {
        margin-bottom: 12px;
      }
      .table-wrap {
        overflow-x: auto;
        border: 3px solid var(--line);
        border-radius: 18px;
        background: #111119;
        box-shadow: 4px 4px 0 #211b52;
      }
      .format-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
      }
      .format-table th, .format-table td {
        padding: 10px 12px;
        border-bottom: 2px dashed var(--line);
        text-align: left;
        vertical-align: top;
        overflow-wrap: anywhere;
        word-break: break-word;
      }
      .format-table th {
        color: var(--ink);
        background: #1b1b28;
      }
      .format-table tbody tr:last-child td {
        border-bottom: 0;
      }
      .format-table .backend-name {
        color: var(--ink);
        font-weight: 700;
      }
      .format-table .parquet-row {
        background: rgba(107, 89, 221, 0.12);
      }
      .format-table .status-ok {
        color: #dcd8ff;
      }
      .format-table .status-degraded,
      .format-table .status-missing {
        color: #f6b7ff;
      }
      .summary-card {
        padding: 16px;
        border-radius: 18px;
        background: var(--panel);
        border: 3px solid var(--line);
        box-shadow: 4px 4px 0 #211b52;
        min-width: 0;
      }
      .summary-card strong {
        display: block;
        margin-top: 6px;
        color: var(--ink);
        font-size: 1.2rem;
        overflow-wrap: anywhere;
        word-break: break-word;
        line-height: 1.1;
      }
      .summary-card span {
        display: block;
        font-size: 0.84rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--muted);
      }
      .field {
        display: grid;
        gap: 6px;
        margin-bottom: 12px;
      }
      input, select, button {
        width: 100%;
        border: 3px solid var(--line);
        border-radius: 14px;
        padding: 11px 12px;
        font: inherit;
        background: #201f31;
        color: var(--ink);
        box-shadow: 4px 4px 0 #211b52;
      }
      button {
        cursor: pointer;
        background: var(--accent);
        color: var(--ink);
        font-weight: 600;
      }
      button.secondary {
        background: var(--accent-soft);
        color: var(--ink);
      }
      .button-row {
        display: flex;
        gap: 10px;
        margin-top: 14px;
      }
      .results-head {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: center;
        margin-bottom: 12px;
      }
      .results-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
        gap: 16px;
      }
      .sample-card {
        overflow: hidden;
        cursor: pointer;
      }
      .sample-card img {
        width: 100%;
        aspect-ratio: 16 / 9;
        object-fit: cover;
        display: block;
        background: #2a2938;
      }
      .sample-card .body {
        padding: 14px;
      }
      .sample-card h3 {
        margin: 0 0 8px;
        font-size: 1rem;
        color: var(--ink);
        overflow-wrap: anywhere;
        word-break: break-word;
      }
      .meta-list {
        display: grid;
        gap: 6px;
        font-size: 0.92rem;
      }
      .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
      }
      .chip {
        display: inline-flex;
        align-items: center;
        padding: 5px 10px;
        border-radius: 999px;
        background: var(--success-soft);
        color: var(--ink);
        border: 2px solid var(--line);
        font-size: 0.8rem;
        overflow-wrap: anywhere;
        word-break: break-word;
      }
      .list {
        list-style: none;
        padding: 0;
        margin: 0;
        display: grid;
        gap: 10px;
      }
      .list li {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 12px;
        align-items: start;
        padding: 12px 0;
        border-bottom: 2px dashed var(--line);
      }
      .list li span {
        min-width: 0;
        overflow-wrap: anywhere;
        word-break: break-word;
      }
      .list li strong {
        white-space: nowrap;
        max-width: 100%;
        justify-self: end;
        align-self: start;
        padding: 3px 8px;
        border: 2px solid var(--line);
        border-radius: 999px;
        background: var(--accent);
        color: var(--ink);
        box-shadow: 3px 3px 0 #211b52;
      }
      .path-list li {
        grid-template-columns: minmax(0, 1fr);
        gap: 8px;
      }
      .path-list li span {
        color: var(--ink);
        font-weight: 700;
      }
      .path-list li code {
        display: block;
        width: 100%;
        padding: 10px 12px;
      }
      .detail-placeholder {
        display: grid;
        place-items: center;
        min-height: 320px;
        text-align: center;
      }
      .camera-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin-top: 14px;
      }
      .camera-frame {
        border: 3px solid var(--line);
        border-radius: 14px;
        overflow: hidden;
        background: var(--panel);
        box-shadow: 4px 4px 0 #211b52;
      }
      .camera-frame img {
        width: 100%;
        display: block;
        aspect-ratio: 16 / 9;
        object-fit: cover;
        background: #2a2938;
      }
      .camera-frame strong {
        display: block;
        padding: 10px 12px 0;
      }
      .camera-frame span {
        display: block;
        padding: 4px 12px 12px;
        font-size: 0.84rem;
        overflow-wrap: anywhere;
        word-break: break-word;
      }
      .camera-compare {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        margin-top: 14px;
      }
      .scene-strip {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
        gap: 10px;
        margin-top: 14px;
      }
      .scene-sample {
        border: 3px solid var(--line);
        border-radius: 16px;
        overflow: hidden;
        background: var(--panel);
        box-shadow: 4px 4px 0 #211b52;
        cursor: pointer;
      }
      .scene-sample img {
        width: 100%;
        aspect-ratio: 16 / 9;
        display: block;
        object-fit: cover;
        background: #2a2938;
      }
      .scene-sample .body {
        padding: 10px 12px 12px;
        font-size: 0.82rem;
      }
      .lidar-card {
        margin-top: 14px;
        border: 3px solid var(--line);
        border-radius: 18px;
        overflow: hidden;
        background: #111119;
        box-shadow: 4px 4px 0 #211b52;
      }
      .lidar-card img {
        display: block;
        width: 100%;
        height: auto;
      }
      .annotation-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 14px;
        font-size: 0.88rem;
        table-layout: fixed;
      }
      .annotation-table th, .annotation-table td {
        padding: 8px 10px;
        border-bottom: 2px dashed var(--line);
        text-align: left;
        vertical-align: top;
        overflow-wrap: anywhere;
        word-break: break-word;
      }
      .pagination {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        align-items: center;
        margin-top: 16px;
      }
      .notice {
        padding: 12px 14px;
        border-radius: 16px;
        background: #201f31;
        color: #e7e8f3;
        border: 3px solid var(--line);
        box-shadow: 4px 4px 0 #211b52;
        margin-bottom: 12px;
      }
      code {
        font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
        background: var(--accent-soft);
        color: var(--ink);
        padding: 2px 6px;
        border-radius: 8px;
        overflow-wrap: anywhere;
        word-break: break-word;
      }
      .results-head span,
      .meta-list span,
      #page_meta,
      #result_meta {
        overflow-wrap: anywhere;
        word-break: break-word;
      }
      @media (max-width: 1380px) {
        .shell { grid-template-columns: minmax(280px, 300px) minmax(0, 1fr); }
        .detail { grid-column: 1 / -1; position: static; }
      }
      @media (max-width: 980px) {
        .shell { grid-template-columns: 1fr; }
        .sidebar { position: static; }
        .camera-grid { grid-template-columns: 1fr; }
        .camera-compare { grid-template-columns: 1fr; }
        .scene-strip { grid-template-columns: repeat(auto-fill, minmax(132px, 1fr)); }
      }
    </style>
  </head>
  <body>
    <main>
      <div class="subnav">
        <a href="/">Browser home</a>
        <a href="/scene-studio">Scene studio</a>
        <a href="/benchmark_dashboard.html">Benchmark dashboard</a>
        <a href="/telemetry_dashboard.html">Telemetry dashboard</a>
        <a href="/grafana/">Grafana</a>
        <a href="/prometheus/">Prometheus</a>
        <a href="/index.html">Report index</a>
      </div>
      <div class="shell">
        <aside class="panel sidebar">
          <h1>Search Ingested Samples</h1>
          <p>Query the samples loaded into PostgreSQL and preview camera images streamed from MinIO.</p>
          <div id="notice" class="notice" hidden></div>
          <div class="field">
            <label for="q">Search</label>
            <input id="q" type="search" placeholder="token, scene, location, or category">
          </div>
          <div class="field">
            <label for="scene">Scene</label>
            <select id="scene"><option value="">All scenes</option></select>
          </div>
          <div class="field">
            <label for="location">Location</label>
            <select id="location"><option value="">All locations</option></select>
          </div>
          <div class="field">
            <label for="category">Category</label>
            <select id="category"><option value="">All categories</option></select>
          </div>
          <div class="field">
            <label for="min_annotations">Minimum annotations</label>
            <input id="min_annotations" type="number" min="0" step="1" value="0">
          </div>
          <div class="field">
            <label for="limit">Page size</label>
            <select id="limit">
              <option value="12">12</option>
              <option value="24" selected>24</option>
              <option value="48">48</option>
              <option value="96">96</option>
            </select>
          </div>
          <div class="button-row">
            <button id="apply">Apply filters</button>
            <button id="reset" class="secondary">Reset</button>
          </div>

          <h2 style="margin-top: 24px;">Popular locations</h2>
          <ul id="top_locations" class="list"></ul>

          <h2 style="margin-top: 24px;">Top categories</h2>
          <ul id="top_categories" class="list"></ul>
        </aside>

        <section class="content">
          <div id="summary" class="summary-grid"></div>
          <section class="panel compare-panel">
            <h2 style="margin-bottom: 4px;">Storage Format Comparison</h2>
            <p id="benchmark_meta">Waiting for the latest benchmark report...</p>
            <div id="benchmark_recommendations" class="chip-row"></div>
            <div class="table-wrap">
              <table class="format-table">
                <thead>
                  <tr>
                    <th>Backend</th>
                    <th>Write samples/s</th>
                    <th>Sequential samples/s</th>
                    <th>Random access p50 ms</th>
                    <th>Disk MB</th>
                    <th>Curation ms</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody id="benchmark_rows">
                  <tr><td colspan="7">No benchmark report loaded yet.</td></tr>
                </tbody>
              </table>
            </div>
          </section>
          <div class="results-head">
            <div>
              <h2 style="margin-bottom: 4px;">Loaded samples</h2>
              <span id="result_meta">Waiting for data...</span>
            </div>
            <span id="page_meta"></span>
          </div>
          <div id="results" class="results-grid"></div>
          <div class="pagination">
            <button id="prev" class="secondary">Previous</button>
            <button id="next" class="secondary">Next</button>
          </div>
        </section>

        <aside class="panel detail" id="detail_panel">
          <div class="detail-placeholder">
            <div>
              <h2>Select a sample</h2>
              <p>The detail view shows scene context, camera and processed previews, a LiDAR top-down render, and stored sensor paths.</p>
            </div>
          </div>
        </aside>
      </div>
    </main>

    <script>
      const initialParams = new URLSearchParams(window.location.search);
      const state = {
        q: initialParams.get("q") || "",
        scene: initialParams.get("scene_token") || "",
        location: initialParams.get("location") || "",
        category: initialParams.get("category") || "",
        minAnnotations: Number(initialParams.get("min_annotations") || 0),
        limit: Number(initialParams.get("limit") || 24),
        offset: Number(initialParams.get("offset") || 0),
        total: 0,
      };

      const el = {
        notice: document.getElementById("notice"),
        q: document.getElementById("q"),
        scene: document.getElementById("scene"),
        location: document.getElementById("location"),
        category: document.getElementById("category"),
        minAnnotations: document.getElementById("min_annotations"),
        limit: document.getElementById("limit"),
        apply: document.getElementById("apply"),
        reset: document.getElementById("reset"),
        summary: document.getElementById("summary"),
        benchmarkMeta: document.getElementById("benchmark_meta"),
        benchmarkRecommendations: document.getElementById("benchmark_recommendations"),
        benchmarkRows: document.getElementById("benchmark_rows"),
        topLocations: document.getElementById("top_locations"),
        topCategories: document.getElementById("top_categories"),
        results: document.getElementById("results"),
        resultMeta: document.getElementById("result_meta"),
        pageMeta: document.getElementById("page_meta"),
        prev: document.getElementById("prev"),
        next: document.getElementById("next"),
        detail: document.getElementById("detail_panel"),
      };

      el.q.value = state.q;
      el.minAnnotations.value = String(state.minAnnotations);
      el.limit.value = String(state.limit);

      function showNotice(message) {
        el.notice.hidden = !message;
        el.notice.textContent = message || "";
      }

      function paramsFromState() {
        const params = new URLSearchParams();
        if (state.q) params.set("q", state.q);
        if (state.scene) params.set("scene_token", state.scene);
        if (state.location) params.set("location", state.location);
        if (state.category) params.set("category", state.category);
        if (state.minAnnotations > 0) params.set("min_annotations", String(state.minAnnotations));
        params.set("limit", String(state.limit));
        params.set("offset", String(state.offset));
        return params;
      }

      async function requestJson(url) {
        const response = await fetch(url);
        if (!response.ok) {
          const payload = await response.json().catch(() => ({ detail: "request failed" }));
          throw new Error(payload.detail || "request failed");
        }
        return response.json();
      }

      function renderSummary(summary) {
        const cards = [
          ["Samples", summary.sample_count ?? 0],
          ["Annotations", summary.annotation_count ?? 0],
          ["Scenes", summary.scene_count ?? 0],
          ["LiDAR points", summary.lidar_points ?? 0],
        ];
        el.summary.innerHTML = cards.map(([label, value]) => `
          <section class="panel summary-card">
            <span>${label}</span>
            <strong class="metric-value">${Number(value).toLocaleString()}</strong>
          </section>
        `).join("");

        el.topLocations.innerHTML = (summary.top_locations || []).map((row) => `
          <li><span>${row.location}</span><strong>${Number(row.count).toLocaleString()}</strong></li>
        `).join("");
        el.topCategories.innerHTML = (summary.top_categories || []).map((row) => `
          <li><span>${row.category}</span><strong>${Number(row.count).toLocaleString()}</strong></li>
        `).join("");
      }

      function renderFilterOptions(filters) {
        el.scene.innerHTML = `<option value="">All scenes</option>` + (filters.scenes || [])
          .map((scene) => `<option value="${scene.scene_token}">${scene.scene_name} · ${scene.location} · ${Number(scene.num_samples || 0).toLocaleString()} samples</option>`)
          .join("");
        el.location.innerHTML = `<option value="">All locations</option>` + filters.locations
          .map((value) => `<option value="${value}">${value}</option>`)
          .join("");
        el.category.innerHTML = `<option value="">All categories</option>` + filters.categories
          .map((value) => `<option value="${value}">${value}</option>`)
          .join("");
        el.scene.value = state.scene;
        el.location.value = state.location;
        el.category.value = state.category;
      }

      function formatMetric(value, digits = 2) {
        if (value === null || value === undefined || Number.isNaN(Number(value))) {
          return "n/a";
        }
        return Number(value).toLocaleString(undefined, {
          minimumFractionDigits: 0,
          maximumFractionDigits: digits,
        });
      }

      function renderBenchmarkSummary(payload) {
        const dataset = payload.dataset || {};
        const formats = payload.storage_formats || [];
        el.benchmarkMeta.textContent = [
          payload.suite_name || "Benchmark report",
          dataset.provider ? `${dataset.provider} provider` : null,
          dataset.samples !== undefined ? `${Number(dataset.samples).toLocaleString()} samples` : null,
          dataset.scenes !== undefined ? `${Number(dataset.scenes).toLocaleString()} scenes` : null,
        ].filter(Boolean).join(" · ");

        const recommendationEntries = Object.entries(payload.recommendations || {});
        el.benchmarkRecommendations.innerHTML = recommendationEntries.map(([stage, backend]) => `
          <span class="chip">${stage.replaceAll("_", " ")}: ${backend}</span>
        `).join("");

        if (!formats.length) {
          el.benchmarkRows.innerHTML = `<tr><td colspan="7">No storage backend metrics were found in the latest report.</td></tr>`;
          return;
        }

        el.benchmarkRows.innerHTML = formats.map((row) => `
          <tr class="${row.backend === "Parquet" ? "parquet-row" : ""}">
            <td class="backend-name">${row.backend}</td>
            <td>${formatMetric(row.write_samples_per_sec)}</td>
            <td>${formatMetric(row.sequential_samples_per_sec)}</td>
            <td>${formatMetric(row.random_access_p50_ms)}</td>
            <td>${formatMetric(row.disk_mb)}</td>
            <td>${formatMetric(row.curation_query_ms)}</td>
            <td class="status-${row.status}">${row.status}</td>
          </tr>
        `).join("");
      }

      function renderResults(payload) {
        state.total = payload.total || 0;
        const start = payload.items.length ? payload.offset + 1 : 0;
        const end = payload.offset + payload.items.length;
        el.resultMeta.textContent = `${state.total} matching samples`;
        el.pageMeta.textContent = start ? `${start}-${end}` : "0";
        el.prev.disabled = payload.offset <= 0;
        el.next.disabled = payload.offset + payload.limit >= payload.total;

        if (!payload.items.length) {
          el.results.innerHTML = `<section class="panel sample-card"><div class="body"><h3>No samples match the current filters.</h3><div class="meta-list">Try reducing the annotation threshold or clearing the category filter.</div></div></section>`;
          return;
        }

        el.results.innerHTML = payload.items.map((sample) => `
          <article class="panel sample-card" data-sample="${sample.sample_idx}">
            ${sample.preview_url ? `<img loading="lazy" src="${sample.preview_url}" alt="Sample ${sample.sample_idx}">` : `<div style="aspect-ratio:16/9;background:#2a2938;"></div>`}
            <div class="body">
              <h3>#${sample.sample_idx} · ${sample.scene_name}</h3>
              <div class="meta-list">
                <span><strong style="color:var(--ink);">Location</strong> ${sample.location}</span>
                <span><strong style="color:var(--ink);">Annotations</strong> ${sample.num_annotations}</span>
                <span><strong style="color:var(--ink);">LiDAR points</strong> ${sample.num_lidar_points}</span>
                <span><strong style="color:var(--ink);">Token</strong> <code>${sample.token.slice(0, 12)}</code></span>
              </div>
              <div class="chip-row">
                ${(sample.annotation_categories || []).slice(0, 6).map((category) => `<span class="chip">${category}</span>`).join("")}
              </div>
            </div>
          </article>
        `).join("");

        document.querySelectorAll(".sample-card[data-sample]").forEach((node) => {
          node.addEventListener("click", () => loadDetail(node.dataset.sample));
        });
      }

      function annotationRow(annotation) {
        return `
          <tr>
            <td>${annotation.category}</td>
            <td>${annotation.num_lidar_pts ?? 0}</td>
            <td>${annotation.num_radar_pts ?? 0}</td>
            <td><code>${JSON.stringify(annotation.translation)}</code></td>
          </tr>
        `;
      }

      function renderDetail(sample) {
        const cameraFrames = Object.entries(sample.camera_urls || {})
          .map(([camera, url]) => url ? `
            <div class="camera-frame">
              <img loading="lazy" src="${url}" alt="${camera}">
              <strong>${camera}</strong>
              <span>${(sample.camera_paths || {})[camera] || "proxied from MinIO"}</span>
            </div>
          ` : "")
          .join("");

        const sensorRows = Object.entries(sample.sensor_paths || {})
          .map(([sensor, path]) => `<li><span>${sensor}</span><code>${path || "missing"}</code></li>`)
          .join("");

        el.detail.innerHTML = `
          <h2>Sample #${sample.sample_idx}</h2>
          <p>${sample.scene_name} · ${sample.location}</p>
          <div class="chip-row" style="margin-top:12px;">
            <span class="chip">${sample.num_annotations} annotations</span>
            <span class="chip">${sample.num_lidar_points} LiDAR points</span>
            <span class="chip">token ${sample.token.slice(0, 12)}</span>
            <span class="chip">scene ${sample.scene_token.slice(0, 12)}</span>
          </div>
          <div class="button-row" style="margin-top:14px;">
            <button class="secondary" onclick="window.location.href='/scene-studio?scene_token=${encodeURIComponent(sample.scene_token)}&sample_idx=${sample.sample_idx}'">Open scene studio</button>
          </div>

          <h3 style="margin-top:18px;">Scene strip</h3>
          <div class="scene-strip" id="scene_strip"><p>Loading scene samples...</p></div>

          <h3 style="margin-top:18px;">Camera previews</h3>
          <div class="camera-grid">${cameraFrames || "<p>No camera blobs available.</p>"}</div>

          <h3 style="margin-top:18px;">Processed camera comparison</h3>
          <div class="camera-compare">
            <div class="camera-frame">
              <img loading="lazy" src="${sample.camera_urls?.CAM_FRONT || ""}" alt="CAM_FRONT original">
              <strong>CAM_FRONT original</strong>
              <span>${(sample.camera_paths || {}).CAM_FRONT || "missing"}</span>
            </div>
            <div class="camera-frame">
              <img loading="lazy" src="/api/samples/${sample.sample_idx}/cameras/CAM_FRONT/processed?mode=edges" alt="CAM_FRONT processed">
              <strong>CAM_FRONT processed</strong>
              <span>Edge-enhanced preview derived on demand from the stored JPEG.</span>
            </div>
          </div>

          <h3 style="margin-top:18px;">LiDAR top-down view</h3>
          <div class="lidar-card">
            <img loading="lazy" src="/api/samples/${sample.sample_idx}/lidar/preview.svg" alt="LiDAR preview for sample ${sample.sample_idx}">
          </div>

          <h3 style="margin-top:18px;">Sensor object paths</h3>
          <ul class="list path-list">${sensorRows}</ul>

          <h3 style="margin-top:18px;">Annotations</h3>
          <table class="annotation-table">
            <thead>
              <tr><th>Category</th><th>LiDAR</th><th>Radar</th><th>Translation</th></tr>
            </thead>
            <tbody>
              ${(sample.annotations || []).map(annotationRow).join("")}
            </tbody>
          </table>
        `;
      }

      function renderSceneSamples(samples) {
        const target = document.getElementById("scene_strip");
        if (!target) return;
        if (!samples.length) {
          target.innerHTML = "<p>No neighboring scene samples were found.</p>";
          return;
        }
        target.innerHTML = samples.map((sample) => `
          <article class="scene-sample" data-scene-sample="${sample.sample_idx}">
            ${sample.preview_url ? `<img loading="lazy" src="${sample.preview_url}" alt="Scene sample ${sample.sample_idx}">` : `<div style="aspect-ratio:16/9;background:#2a2938;"></div>`}
            <div class="body">
              <strong style="display:block;color:var(--ink);margin-bottom:6px;">#${sample.sample_idx}</strong>
              <span>${sample.num_annotations} ann · ${sample.num_lidar_points} pts</span>
            </div>
          </article>
        `).join("");
        target.querySelectorAll("[data-scene-sample]").forEach((node) => {
          node.addEventListener("click", () => loadDetail(node.dataset.sceneSample));
        });
      }

      async function loadSummary() {
        const summary = await requestJson("/api/summary");
        renderSummary(summary);
      }

      async function loadFilters() {
        const filters = await requestJson("/api/filters");
        renderFilterOptions(filters);
      }

      async function loadResults() {
        const payload = await requestJson(`/api/samples?${paramsFromState().toString()}`);
        renderResults(payload);
      }

      async function loadBenchmarkSummary() {
        try {
          const payload = await requestJson("/api/benchmark/summary");
          renderBenchmarkSummary(payload);
        } catch (error) {
          el.benchmarkMeta.textContent = error.message;
          el.benchmarkRecommendations.innerHTML = "";
          el.benchmarkRows.innerHTML = `<tr><td colspan="7">Benchmark comparison is unavailable.</td></tr>`;
        }
      }

      async function loadDetail(sampleIdx) {
        const sample = await requestJson(`/api/samples/${sampleIdx}`);
        renderDetail(sample);
        const scenePayload = await requestJson(`/api/scenes/${encodeURIComponent(sample.scene_token)}/samples`);
        renderSceneSamples(scenePayload.items || []);
      }

      function syncStateFromInputs() {
        state.q = el.q.value.trim();
        state.scene = el.scene.value;
        state.location = el.location.value;
        state.category = el.category.value;
        state.minAnnotations = Number(el.minAnnotations.value || 0);
        state.limit = Number(el.limit.value || 24);
      }

      async function refresh(resetOffset = false) {
        syncStateFromInputs();
        if (resetOffset) state.offset = 0;
        try {
          showNotice("");
          await Promise.all([loadSummary(), loadResults()]);
        } catch (error) {
          showNotice(error.message);
        }
      }

      el.apply.addEventListener("click", () => refresh(true));
      el.reset.addEventListener("click", async () => {
        el.q.value = "";
        el.scene.value = "";
        el.location.value = "";
        el.category.value = "";
        el.minAnnotations.value = "0";
        el.limit.value = "24";
        state.offset = 0;
        await refresh(true);
      });
      el.prev.addEventListener("click", async () => {
        state.offset = Math.max(0, state.offset - state.limit);
        await refresh(false);
      });
      el.next.addEventListener("click", async () => {
        state.offset += state.limit;
        await refresh(false);
      });
      el.q.addEventListener("keydown", (event) => {
        if (event.key === "Enter") refresh(true);
      });

      (async () => {
        try {
          await loadFilters();
          await loadBenchmarkSummary();
          await refresh(true);
        } catch (error) {
          showNotice(error.message);
        }
      })();
    </script>
  </body>
</html>
"""


def build_scene_studio_html() -> str:
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>nuDemo Scene Studio</title>
    <style>
      :root {
        --bg: #09090e;
        --panel: #151520;
        --panel-alt: #101018;
        --line: #544bb0;
        --ink: #ececff;
        --muted: #b7b4d6;
        --accent: #6b59dd;
        --accent-soft: #201f31;
        --shadow: 6px 6px 0 #211b52;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        color: var(--ink);
        font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
        background:
          radial-gradient(circle at top right, rgba(84, 54, 252, 0.14), transparent 24%),
          linear-gradient(180deg, #0f0f16 0%, var(--bg) 100%);
        overflow-x: hidden;
      }
      main {
        max-width: 1580px;
        margin: 0 auto;
        padding: 24px clamp(12px, 1.8vw, 24px) 52px;
      }
      .subnav {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 18px;
      }
      .subnav a {
        color: var(--ink);
        font-weight: 700;
        text-decoration: none;
        padding: 10px 12px;
        border: 3px solid var(--line);
        border-radius: 999px;
        background: var(--accent-soft);
        box-shadow: 4px 4px 0 #211b52;
      }
      .hero {
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(300px, 420px);
        gap: 16px;
        margin-bottom: 18px;
      }
      .panel {
        min-width: 0;
        background: var(--panel);
        border: 3px solid var(--line);
        border-radius: 24px;
        box-shadow: var(--shadow);
      }
      .hero-panel, .hero-meta, .controls, .timeline-panel, .inspector, .camera-panel, .canvas-panel {
        padding: 18px;
      }
      .hero-panel h1 {
        margin: 0 0 10px;
        line-height: 0.98;
        font-size: clamp(2.3rem, 4vw, 4.1rem);
        letter-spacing: -0.05em;
      }
      .hero-panel p, .hero-meta p, .controls p, .inspector p, .camera-label, .status-line, .annotation-list li, .timeline-card span {
        color: var(--muted);
      }
      .hero-meta {
        background: var(--panel-alt);
      }
      .chip-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 12px;
      }
      .chip {
        display: inline-flex;
        align-items: center;
        padding: 6px 10px;
        border-radius: 999px;
        border: 2px solid var(--line);
        background: var(--accent-soft);
        color: var(--ink);
        font-size: 0.82rem;
        font-weight: 700;
      }
      .studio-shell {
        display: grid;
        grid-template-columns: minmax(280px, 320px) minmax(0, 1fr);
        gap: 18px;
        align-items: start;
      }
      .controls {
        position: sticky;
        top: 18px;
      }
      .field {
        display: grid;
        gap: 6px;
        margin-bottom: 12px;
      }
      label, .label {
        color: var(--muted);
        font-size: 0.92rem;
      }
      input, select, button {
        width: 100%;
        border: 3px solid var(--line);
        border-radius: 14px;
        padding: 11px 12px;
        font: inherit;
        background: #201f31;
        color: var(--ink);
        box-shadow: 4px 4px 0 #211b52;
      }
      button {
        cursor: pointer;
        background: var(--accent);
        font-weight: 700;
      }
      button.secondary {
        background: var(--accent-soft);
      }
      .button-row {
        display: flex;
        gap: 10px;
      }
      .studio-main {
        display: grid;
        gap: 18px;
      }
      .viewer-meta {
        display: flex;
        justify-content: space-between;
        gap: 14px;
        align-items: flex-start;
        margin-bottom: 12px;
      }
      .viewer-meta h2 {
        margin: 0 0 4px;
      }
      .status-line {
        font-size: 0.9rem;
      }
      #lidar_canvas {
        display: block;
        width: 100%;
        height: min(68vh, 760px);
        border-radius: 18px;
        background:
          radial-gradient(circle at top left, rgba(107, 89, 221, 0.12), transparent 28%),
          linear-gradient(180deg, #09090e 0%, #0f0f16 100%);
      }
      .studio-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.45fr) minmax(300px, 360px);
        gap: 18px;
      }
      .timeline-strip {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 12px;
        margin-top: 12px;
      }
      .timeline-card {
        border: 3px solid var(--line);
        border-radius: 18px;
        overflow: hidden;
        background: var(--panel-alt);
        box-shadow: 4px 4px 0 #211b52;
        cursor: pointer;
        transition: transform 120ms ease;
      }
      .timeline-card.active {
        transform: translate(-2px, -2px);
        box-shadow: 8px 8px 0 #211b52;
        background: rgba(107, 89, 221, 0.14);
      }
      .timeline-card img {
        width: 100%;
        aspect-ratio: 16 / 9;
        object-fit: cover;
        display: block;
        background: #1f2030;
      }
      .timeline-card .body {
        padding: 10px 12px 12px;
        display: grid;
        gap: 6px;
      }
      .timeline-card strong {
        color: var(--ink);
      }
      .camera-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
      }
      .camera-frame {
        border: 3px solid var(--line);
        border-radius: 18px;
        overflow: hidden;
        background: var(--panel-alt);
        box-shadow: 4px 4px 0 #211b52;
      }
      .camera-frame img {
        width: 100%;
        display: block;
        aspect-ratio: 16 / 9;
        object-fit: cover;
        background: #1f2030;
      }
      .camera-frame strong {
        display: block;
        padding: 10px 12px 0;
      }
      .camera-label {
        display: block;
        padding: 4px 12px 12px;
        font-size: 0.86rem;
      }
      .inspector-grid {
        display: grid;
        gap: 10px;
      }
      .metric-row {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: baseline;
        padding: 10px 0;
        border-bottom: 2px dashed var(--line);
      }
      .metric-row strong {
        color: var(--ink);
      }
      .annotation-list {
        list-style: none;
        padding: 0;
        margin: 12px 0 0;
        display: grid;
        gap: 10px;
      }
      .annotation-list li {
        padding: 10px 12px;
        border: 2px solid var(--line);
        border-radius: 14px;
        background: var(--accent-soft);
        overflow-wrap: anywhere;
      }
      .notice {
        padding: 12px 14px;
        border-radius: 16px;
        background: #201f31;
        color: var(--ink);
        border: 3px solid var(--line);
        box-shadow: 4px 4px 0 #211b52;
        margin-bottom: 12px;
      }
      code {
        font-family: "IBM Plex Mono", monospace;
        background: var(--accent-soft);
        color: var(--ink);
        padding: 2px 6px;
        border-radius: 8px;
        overflow-wrap: anywhere;
      }
      @media (max-width: 1240px) {
        .hero, .studio-shell, .studio-grid {
          grid-template-columns: 1fr;
        }
        .controls {
          position: static;
        }
      }
      @media (max-width: 860px) {
        .camera-grid {
          grid-template-columns: 1fr;
        }
        .timeline-strip {
          grid-template-columns: repeat(auto-fill, minmax(132px, 1fr));
        }
      }
    </style>
  </head>
  <body>
    <main>
      <div class="subnav">
        <a href="/">Browser home</a>
        <a href="/explorer">Explorer</a>
        <a href="/scene-studio">Scene studio</a>
        <a href="/benchmark_dashboard.html">Benchmark dashboard</a>
        <a href="/telemetry_dashboard.html">Telemetry dashboard</a>
        <a href="/grafana/">Grafana</a>
        <a href="/prometheus/">Prometheus</a>
      </div>

      <section class="hero">
        <section class="panel hero-panel">
          <h1>Scene Studio</h1>
          <p>
            Load a real nuScenes scene, scrub across samples, and inspect the LiDAR point cloud in
            3D while keeping the front camera and annotation context in sync.
          </p>
          <div id="scene_chips" class="chip-row"></div>
        </section>
        <aside class="panel hero-meta">
          <h2 style="margin-top:0;">Live browser goals</h2>
          <p>Use this page to demonstrate scene-level understanding, not just single-sample lookup.</p>
          <div class="chip-row">
            <span class="chip">3D LiDAR</span>
            <span class="chip">Scene scrubber</span>
            <span class="chip">Real sample payloads</span>
            <span class="chip">Processed camera diff</span>
          </div>
        </aside>
      </section>

      <div class="studio-shell">
        <aside class="panel controls">
          <div id="studio_notice" class="notice" hidden></div>
          <div class="field">
            <label for="scene_select">Scene</label>
            <select id="scene_select"><option value="">Loading scenes…</option></select>
          </div>
          <div class="field">
            <label for="max_points">3D point budget</label>
            <select id="max_points">
              <option value="8000">8,000</option>
              <option value="16000" selected>16,000</option>
              <option value="32000">32,000</option>
            </select>
          </div>
          <div class="field">
            <label for="processed_mode">Processed camera mode</label>
            <select id="processed_mode">
              <option value="edges" selected>Edges</option>
              <option value="grayscale">Grayscale</option>
              <option value="contrast">Contrast</option>
            </select>
          </div>
          <div class="button-row" style="margin-bottom:12px;">
            <button id="load_scene">Load scene</button>
            <button id="open_search" class="secondary">Back to search</button>
          </div>
          <p>
            The canvas renders the stored <code>LIDAR_TOP.npy</code> payload for the selected
            sample. Orbit, pan, and zoom are enabled, and the timeline below lets you walk the
            scene sample by sample.
          </p>
        </aside>

        <section class="studio-main">
          <section class="panel canvas-panel">
            <div class="viewer-meta">
              <div>
                <h2 id="viewer_title">Waiting for scene data…</h2>
                <div id="viewer_status" class="status-line">Load a scene to start the 3D viewer.</div>
              </div>
              <div class="chip-row" id="viewer_chips"></div>
            </div>
            <canvas id="lidar_canvas"></canvas>
          </section>

          <div class="studio-grid">
            <section class="panel timeline-panel">
              <h2 style="margin-top:0;">Scene timeline</h2>
              <div id="scene_timeline" class="timeline-strip"></div>
            </section>

            <aside class="panel inspector">
              <h2 style="margin-top:0;">Sample inspector</h2>
              <div id="sample_inspector" class="inspector-grid">
                <p>Select a scene sample to inspect its metadata and annotations.</p>
              </div>
            </aside>
          </div>

          <section class="panel camera-panel">
            <h2 style="margin-top:0;">Front camera compare</h2>
            <div id="camera_compare" class="camera-grid">
              <div class="camera-frame">
                <div style="aspect-ratio:16/9;background:#1f2030;"></div>
                <strong>Original</strong>
                <span class="camera-label">Load a sample to preview CAM_FRONT.</span>
              </div>
              <div class="camera-frame">
                <div style="aspect-ratio:16/9;background:#1f2030;"></div>
                <strong>Processed</strong>
                <span class="camera-label">Edges, grayscale, or contrast preview.</span>
              </div>
            </div>
          </section>
        </section>
      </div>
    </main>

    <script type="module">
      import * as THREE from "https://unpkg.com/three@0.180.0/build/three.module.js";
      import { OrbitControls } from "https://unpkg.com/three@0.180.0/examples/jsm/controls/OrbitControls.js";

      const params = new URLSearchParams(window.location.search);

      const state = {
        sceneToken: params.get("scene_token") || "",
        sampleIdx: params.get("sample_idx") ? Number(params.get("sample_idx")) : null,
        maxPoints: Number(params.get("max_points") || 16000),
        processedMode: params.get("processed_mode") || "edges",
        currentScene: null,
        currentSample: null,
      };

      const el = {
        notice: document.getElementById("studio_notice"),
        sceneSelect: document.getElementById("scene_select"),
        maxPoints: document.getElementById("max_points"),
        processedMode: document.getElementById("processed_mode"),
        loadScene: document.getElementById("load_scene"),
        openSearch: document.getElementById("open_search"),
        sceneChips: document.getElementById("scene_chips"),
        viewerTitle: document.getElementById("viewer_title"),
        viewerStatus: document.getElementById("viewer_status"),
        viewerChips: document.getElementById("viewer_chips"),
        timeline: document.getElementById("scene_timeline"),
        inspector: document.getElementById("sample_inspector"),
        cameraCompare: document.getElementById("camera_compare"),
        canvas: document.getElementById("lidar_canvas"),
      };

      const viewer = createViewer(el.canvas);

      function showNotice(message) {
        el.notice.hidden = !message;
        el.notice.textContent = message || "";
      }

      function syncQuery() {
        const next = new URLSearchParams();
        if (state.sceneToken) next.set("scene_token", state.sceneToken);
        if (state.sampleIdx !== null && state.sampleIdx !== undefined) next.set("sample_idx", String(state.sampleIdx));
        next.set("max_points", String(state.maxPoints));
        next.set("processed_mode", state.processedMode);
        const query = next.toString();
        history.replaceState(null, "", query ? `/scene-studio?${query}` : "/scene-studio");
      }

      async function requestJson(url) {
        const response = await fetch(url);
        if (!response.ok) {
          const payload = await response.json().catch(() => ({ detail: "request failed" }));
          throw new Error(payload.detail || "request failed");
        }
        return response.json();
      }

      function formatNumber(value, digits = 0) {
        return Number(value || 0).toLocaleString(undefined, {
          minimumFractionDigits: 0,
          maximumFractionDigits: digits,
        });
      }

      function populateScenes(filters) {
        const scenes = filters.scenes || [];
        el.sceneSelect.innerHTML = scenes
          .map((scene) => `<option value="${scene.scene_token}">${scene.scene_name} · ${scene.location} · ${formatNumber(scene.num_samples)} samples</option>`)
          .join("");

        if (!state.sceneToken && scenes.length) {
          state.sceneToken = scenes[0].scene_token;
        }
        el.sceneSelect.value = state.sceneToken;
      }

      function renderSceneHeader(scene) {
        el.viewerTitle.textContent = `${scene.scene_name} · ${scene.location}`;
        el.sceneChips.innerHTML = `
          <span class="chip">${formatNumber(scene.num_samples)} samples</span>
          <span class="chip">scene ${scene.scene_token.slice(0, 12)}</span>
          <span class="chip">3D LiDAR active</span>
        `;
      }

      function renderTimeline(scene, activeSampleIdx) {
        const samples = scene.samples || [];
        if (!samples.length) {
          el.timeline.innerHTML = "<p>No scene samples available.</p>";
          return;
        }
        el.timeline.innerHTML = samples.map((sample) => `
          <article class="timeline-card ${sample.sample_idx === activeSampleIdx ? "active" : ""}" data-sample="${sample.sample_idx}">
            ${sample.preview_url ? `<img loading="lazy" src="${sample.preview_url}" alt="Sample ${sample.sample_idx}">` : `<div style="aspect-ratio:16/9;background:#1f2030;"></div>`}
            <div class="body">
              <strong>#${sample.sample_idx}</strong>
              <span>${formatNumber(sample.num_annotations)} ann · ${formatNumber(sample.num_lidar_points)} pts</span>
            </div>
          </article>
        `).join("");
        el.timeline.querySelectorAll("[data-sample]").forEach((node) => {
          node.addEventListener("click", () => loadSample(Number(node.dataset.sample)));
        });
      }

      function renderInspector(sample, lidarPayload) {
        const annotations = (sample.annotations || []).slice(0, 8);
        el.inspector.innerHTML = `
          <div class="metric-row"><span>Sample</span><strong>#${sample.sample_idx}</strong></div>
          <div class="metric-row"><span>Timestamp</span><strong>${formatNumber(sample.timestamp)}</strong></div>
          <div class="metric-row"><span>Scene</span><strong>${sample.scene_name}</strong></div>
          <div class="metric-row"><span>Rendered points</span><strong>${formatNumber(lidarPayload.rendered_count)}</strong></div>
          <div class="metric-row"><span>Total LiDAR points</span><strong>${formatNumber(lidarPayload.count)}</strong></div>
          <div class="metric-row"><span>Annotations</span><strong>${formatNumber(sample.num_annotations)}</strong></div>
          <div class="metric-row"><span>Location</span><strong>${sample.location}</strong></div>
          <div class="metric-row"><span>Token</span><strong><code>${sample.token.slice(0, 16)}</code></strong></div>
          <h3 style="margin:12px 0 0;">Top annotations</h3>
          <ul class="annotation-list">
            ${annotations.map((annotation) => `<li><strong style="color:var(--ink);">${annotation.category}</strong><br>LiDAR ${formatNumber(annotation.num_lidar_pts)} · Radar ${formatNumber(annotation.num_radar_pts)}</li>`).join("") || "<li>No annotations available.</li>"}
          </ul>
        `;
      }

      function renderCameraCompare(sample) {
        const originalUrl = sample.camera_urls?.CAM_FRONT || "";
        const processedUrl = originalUrl
          ? `/api/samples/${sample.sample_idx}/cameras/CAM_FRONT/processed?mode=${encodeURIComponent(state.processedMode)}`
          : "";
        const originalLabel = (sample.camera_paths || {}).CAM_FRONT || "CAM_FRONT unavailable";
        const processedLabel = `Processed CAM_FRONT using ${state.processedMode}.`;

        el.cameraCompare.innerHTML = `
          <div class="camera-frame">
            ${originalUrl ? `<img loading="lazy" src="${originalUrl}" alt="CAM_FRONT original">` : `<div style="aspect-ratio:16/9;background:#1f2030;"></div>`}
            <strong>CAM_FRONT original</strong>
            <span class="camera-label">${originalLabel}</span>
          </div>
          <div class="camera-frame">
            ${processedUrl ? `<img loading="lazy" src="${processedUrl}" alt="CAM_FRONT processed">` : `<div style="aspect-ratio:16/9;background:#1f2030;"></div>`}
            <strong>CAM_FRONT processed</strong>
            <span class="camera-label">${processedLabel}</span>
          </div>
        `;
      }

      async function loadScene(sceneToken, preferredSampleIdx = null) {
        state.sceneToken = sceneToken;
        syncQuery();
        const scene = await requestJson(`/api/scenes/${encodeURIComponent(sceneToken)}`);
        state.currentScene = scene;
        renderSceneHeader(scene);

        const firstSample = preferredSampleIdx ?? state.sampleIdx ?? scene.samples?.[0]?.sample_idx ?? null;
        renderTimeline(scene, firstSample);
        if (firstSample !== null) {
          await loadSample(Number(firstSample), { skipSceneReload: true });
        }
      }

      async function loadSample(sampleIdx, options = {}) {
        state.sampleIdx = sampleIdx;
        syncQuery();
        const sample = await requestJson(`/api/samples/${sampleIdx}`);
        const lidar = await requestJson(`/api/samples/${sampleIdx}/lidar/points?max_points=${encodeURIComponent(state.maxPoints)}`);
        state.currentSample = sample;

        el.viewerStatus.textContent = `${formatNumber(lidar.rendered_count)} of ${formatNumber(lidar.count)} points rendered · sample #${sample.sample_idx}`;
        el.viewerChips.innerHTML = `
          <span class="chip">${sample.scene_name}</span>
          <span class="chip">${formatNumber(sample.num_annotations)} annotations</span>
          <span class="chip">${state.maxPoints.toLocaleString()} point budget</span>
        `;

        viewer.setPointCloud(lidar);
        renderInspector(sample, lidar);
        renderCameraCompare(sample);

        if (!options.skipSceneReload && sample.scene_token !== state.sceneToken) {
          await loadScene(sample.scene_token, sample.sample_idx);
          return;
        }

        if (state.currentScene) {
          renderTimeline(state.currentScene, sampleIdx);
        }
      }

      function createViewer(canvas) {
        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x09090e);
        scene.add(new THREE.AmbientLight(0xffffff, 0.9));
        const light = new THREE.DirectionalLight(0xffffff, 1.15);
        light.position.set(16, 18, 10);
        scene.add(light);
        scene.add(new THREE.GridHelper(120, 30, 0x544bb0, 0x24233a));
        scene.add(new THREE.AxesHelper(12));

        const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
        camera.position.set(28, 18, 28);

        const controls = new OrbitControls(camera, canvas);
        controls.enableDamping = true;
        controls.target.set(0, 0, 0);

        let cloud = null;
        let lastWidth = 0;
        let lastHeight = 0;

        function resize() {
          const width = canvas.clientWidth || 1;
          const height = canvas.clientHeight || 1;
          if (lastWidth !== width || lastHeight !== height) {
            renderer.setSize(width, height, false);
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            lastWidth = width;
            lastHeight = height;
          }
        }

        function setPointCloud(payload) {
          if (cloud) {
            scene.remove(cloud);
            cloud.geometry.dispose();
            cloud.material.dispose();
            cloud = null;
          }

          const source = payload.positions || [];
          const intensity = payload.intensity || [];
          const pointCount = Math.floor(source.length / 3);
          const positions = new Float32Array(pointCount * 3);
          const colors = new Float32Array(pointCount * 3);

          let zMin = Infinity;
          let zMax = -Infinity;
          for (let idx = 0; idx < pointCount; idx += 1) {
            const z = source[idx * 3 + 2];
            zMin = Math.min(zMin, z);
            zMax = Math.max(zMax, z);
          }
          const zSpan = Math.max(0.001, zMax - zMin);

          for (let idx = 0; idx < pointCount; idx += 1) {
            const x = source[idx * 3];
            const y = source[idx * 3 + 1];
            const z = source[idx * 3 + 2];

            positions[idx * 3] = x;
            positions[idx * 3 + 1] = z;
            positions[idx * 3 + 2] = -y;

            const normalizedZ = (z - zMin) / zSpan;
            const normalizedI = Math.max(0, Math.min(1, Number(intensity[idx] || 0)));
            const color = new THREE.Color();
            color.setHSL(0.78 - normalizedZ * 0.28, 0.86, 0.48 + normalizedI * 0.18);
            colors[idx * 3] = color.r;
            colors[idx * 3 + 1] = color.g;
            colors[idx * 3 + 2] = color.b;
          }

          const geometry = new THREE.BufferGeometry();
          geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
          geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
          geometry.computeBoundingSphere();

          const material = new THREE.PointsMaterial({
            size: 0.11,
            vertexColors: true,
            transparent: true,
            opacity: 0.95,
          });
          cloud = new THREE.Points(geometry, material);
          scene.add(cloud);

          const radius = Math.max(18, geometry.boundingSphere?.radius || 18);
          camera.position.set(radius * 1.1, radius * 0.58, radius * 1.1);
          controls.target.set(0, 0, 0);
          controls.update();
        }

        function animate() {
          resize();
          controls.update();
          renderer.render(scene, camera);
          requestAnimationFrame(animate);
        }
        animate();

        return { setPointCloud };
      }

      async function bootstrap() {
        try {
          showNotice("");
          el.maxPoints.value = String(state.maxPoints);
          el.processedMode.value = state.processedMode;

          const filters = await requestJson("/api/filters");
          populateScenes(filters);
          if (!state.sceneToken && filters.scenes?.length) {
            state.sceneToken = filters.scenes[0].scene_token;
          }
          if (state.sceneToken) {
            await loadScene(state.sceneToken, state.sampleIdx);
          }
        } catch (error) {
          showNotice(error.message);
        }
      }

      el.sceneSelect.addEventListener("change", async () => {
        state.sceneToken = el.sceneSelect.value;
        state.sampleIdx = null;
        await bootstrap();
      });

      el.maxPoints.addEventListener("change", async () => {
        state.maxPoints = Number(el.maxPoints.value || 16000);
        if (state.sampleIdx !== null) {
          await loadSample(state.sampleIdx, { skipSceneReload: true });
        }
      });

      el.processedMode.addEventListener("change", async () => {
        state.processedMode = el.processedMode.value;
        syncQuery();
        if (state.currentSample) {
          renderCameraCompare(state.currentSample);
        }
      });

      el.loadScene.addEventListener("click", async () => {
        if (el.sceneSelect.value) {
          state.sceneToken = el.sceneSelect.value;
          state.sampleIdx = null;
          await bootstrap();
        }
      });

      el.openSearch.addEventListener("click", () => {
        const next = new URLSearchParams();
        if (state.sceneToken) next.set("scene_token", state.sceneToken);
        window.location.href = next.toString() ? `/explorer?${next}` : "/explorer";
      });

      bootstrap();
    </script>
  </body>
</html>
"""


class ExplorerApplication:
    def __init__(self, app: FastAPI) -> None:
        self._app = app

    def run(self, host: str = "127.0.0.1", port: int = 8788, debug: bool = False) -> None:
        import uvicorn

        uvicorn.run(
            self._app,
            host=host,
            port=port,
            log_level="debug" if debug else "info",
        )


def create_app(
    config: AppConfig | None = None,
    *,
    result_limit: int = 24,
) -> FastAPI:
    config = config or AppConfig.load()
    config.runtime.ensure()
    ensure_metrics_exporter(config.services.postgres)
    store = ExplorerStore(config)
    report_store = BenchmarkReportStore(config.runtime.reports_root)
    default_limit = max(1, min(result_limit, 100))
    app = FastAPI(title="nuDemo Browser", docs_url="/docs", redoc_url=None)
    install_http_metrics(app)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "reports_root": str(config.runtime.reports_root)}

    @app.get("/", response_class=HTMLResponse)
    def browser_home() -> str:
        return build_browser_home_html()

    @app.get("/explorer", response_class=HTMLResponse)
    def explorer_page() -> str:
        return build_explorer_html()

    @app.get("/scene-studio", response_class=HTMLResponse)
    def scene_studio_page() -> str:
        return build_scene_studio_html()

    @app.get("/api/summary")
    def api_summary() -> dict[str, Any]:
        try:
            return store.fetch_summary()
        except Exception as exc:  # pragma: no cover - depends on external services
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/filters")
    def api_filters() -> dict[str, Any]:
        try:
            return store.fetch_filters()
        except Exception as exc:  # pragma: no cover - depends on external services
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/benchmark/summary")
    def api_benchmark_summary() -> dict[str, Any]:
        try:
            return report_store.fetch_summary()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"{exc} was not found") from exc
        except Exception as exc:  # pragma: no cover - depends on external services
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/samples")
    def api_samples(
        q: str | None = None,
        scene_token: str | None = None,
        location: str | None = None,
        category: str | None = None,
        min_annotations: int = Query(default=0, ge=0),
        limit: int = Query(default=default_limit, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        try:
            return store.search_samples(
                q=q,
                scene_token=scene_token,
                location=location,
                category=category,
                min_annotations=min_annotations,
                limit=limit,
                offset=offset,
            )
        except Exception as exc:  # pragma: no cover - depends on external services
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/samples/{sample_idx}")
    def api_sample_detail(sample_idx: int) -> dict[str, Any]:
        try:
            payload = store.fetch_sample_detail(sample_idx)
        except Exception as exc:  # pragma: no cover - depends on external services
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if payload is None:
            raise HTTPException(status_code=404, detail=f"sample {sample_idx} was not found")
        return payload

    @app.get("/api/samples/{sample_idx}/cameras/{camera}")
    def api_sample_camera(sample_idx: int, camera: str) -> Response:
        normalized_camera = camera.upper()
        try:
            payload = store.fetch_camera_bytes(sample_idx, normalized_camera)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"camera {escape(camera)} is not available",
            ) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"camera {normalized_camera} for sample {sample_idx} was not found",
            ) from exc
        except Exception as exc:  # pragma: no cover - depends on external services
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(content=payload, media_type="image/jpeg")

    @app.get("/api/samples/{sample_idx}/cameras/{camera}/processed")
    def api_sample_camera_processed(sample_idx: int, camera: str, mode: str = "edges") -> Response:
        normalized_camera = camera.upper()
        try:
            payload = store.fetch_camera_bytes(sample_idx, normalized_camera)
            processed = process_camera_payload(payload, mode=mode)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"camera {escape(camera)} is not available",
            ) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"camera {normalized_camera} for sample {sample_idx} was not found",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - depends on external services
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(content=processed, media_type="image/jpeg")

    @app.get("/api/samples/{sample_idx}/lidar/preview.svg")
    def api_sample_lidar_preview(
        sample_idx: int,
        width: int = Query(default=720, ge=240, le=1400),
        height: int = Query(default=420, ge=180, le=900),
        max_points: int = Query(default=2500, ge=200, le=12000),
    ) -> Response:
        try:
            payload = store.fetch_sensor_bytes(sample_idx, "LIDAR_TOP")
            svg = lidar_payload_to_svg(
                payload,
                width=width,
                height=height,
                max_points=max_points,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="LiDAR sensor is not available") from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"LiDAR payload for sample {sample_idx} was not found",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - depends on external services
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(content=svg, media_type="image/svg+xml")

    @app.get("/api/samples/{sample_idx}/lidar/points")
    def api_sample_lidar_points(
        sample_idx: int,
        max_points: int = Query(default=16000, ge=1000, le=64000),
    ) -> dict[str, Any]:
        try:
            payload = store.fetch_sensor_bytes(sample_idx, "LIDAR_TOP")
            return lidar_payload_to_point_cloud(payload, max_points=max_points)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="LiDAR sensor is not available") from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"LiDAR payload for sample {sample_idx} was not found",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - depends on external services
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/scenes/{scene_token}/samples")
    def api_scene_samples(
        scene_token: str,
        limit: int = Query(default=18, ge=1, le=36),
    ) -> dict[str, Any]:
        try:
            items = store.fetch_scene_samples(scene_token, limit=limit)
        except Exception as exc:  # pragma: no cover - depends on external services
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"scene_token": scene_token, "items": items}

    @app.get("/api/scenes/{scene_token}")
    def api_scene_detail(
        scene_token: str,
        limit: int = Query(default=180, ge=1, le=240),
    ) -> dict[str, Any]:
        try:
            scene = store.fetch_scene_detail(scene_token, limit=limit)
        except Exception as exc:  # pragma: no cover - depends on external services
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if scene is None:
            raise HTTPException(status_code=404, detail=f"scene {scene_token} was not found")
        return scene

    app.mount("/", StaticFiles(directory=config.runtime.reports_root), name="reports")
    return app


def create_explorer_app(
    config: AppConfig | None = None,
    *,
    result_limit: int = 24,
) -> ExplorerApplication:
    return ExplorerApplication(create_app(config=config, result_limit=result_limit))
