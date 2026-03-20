# ruff: noqa: E501

from __future__ import annotations

from html import escape
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from nudemo.config import AppConfig

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

    def fetch_filters(self) -> dict[str, list[str]]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT location FROM samples ORDER BY location")
            locations = [row["location"] for row in cursor.fetchall()]

            cursor.execute("SELECT DISTINCT category FROM annotations ORDER BY category")
            categories = [row["category"] for row in cursor.fetchall()]

        return {"locations": locations, "categories": categories}

    def search_samples(
        self,
        *,
        q: str | None,
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
                    WHERE (%(location)s::text IS NULL OR s.location = %(location)s::text)
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
        --bg: #050507;
        --panel: #181819;
        --line: #6246fb;
        --ink: #f3f7f7;
        --muted: #aabae5;
        --accent: #5436fc;
        --accent-alt: #3c31bb;
        --accent-soft: #1a1a20;
        --shadow: 8px 8px 0 #2a2170;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Space Grotesk", "IBM Plex Sans", "Avenir Next", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top right, rgba(84, 54, 252, 0.2), transparent 24%),
          linear-gradient(180deg, #09090f 0%, var(--bg) 100%);
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
        box-shadow: 4px 4px 0 #2a2170;
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
        background: #1a1a20;
        border: 3px solid var(--line);
        box-shadow: 6px 6px 0 #2a2170;
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
              <span>Benchmark comparison dashboard</span>
              <a href="/benchmark_dashboard.html">Open benchmark</a>
            </div>
            <div class="link-row">
              <span>Telemetry and bottleneck dashboard</span>
              <a href="/telemetry_dashboard.html">Open telemetry</a>
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
          Search by token, scene, location, or annotation category. Filter the loaded dataset and
          inspect camera images and raw metadata for each sample.
          <a href="/explorer">Go to explorer</a>
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
        --bg: #050507;
        --panel: #181819;
        --line: #6246fb;
        --ink: #f3f7f7;
        --muted: #aabae5;
        --accent: #5436fc;
        --accent-alt: #3c31bb;
        --accent-soft: #1a1a20;
        --success-soft: #26262e;
        --shadow: 8px 8px 0 #2a2170;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Space Grotesk", "IBM Plex Sans", "Avenir Next", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top right, rgba(84, 54, 252, 0.2), transparent 20%),
          linear-gradient(180deg, #09090f 0%, var(--bg) 100%);
      }
      main {
        max-width: 1440px;
        margin: 0 auto;
        padding: 28px 18px 48px;
      }
      .shell {
        display: grid;
        grid-template-columns: 320px 1fr 420px;
        gap: 18px;
      }
      .panel {
        background: var(--panel);
        border: 3px solid var(--line);
        border-radius: 24px;
        box-shadow: var(--shadow);
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
        box-shadow: 4px 4px 0 #2a2170;
      }
      .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 14px;
        margin-bottom: 18px;
      }
      .summary-card {
        padding: 16px;
        border-radius: 18px;
        background: var(--panel);
        border: 3px solid var(--line);
        box-shadow: 6px 6px 0 #2a2170;
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
        background: #1a1a20;
        color: var(--ink);
        box-shadow: 4px 4px 0 #2a2170;
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
        background: #26262e;
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
        box-shadow: 3px 3px 0 #2a2170;
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
        box-shadow: 5px 5px 0 #2a2170;
      }
      .camera-frame img {
        width: 100%;
        display: block;
        aspect-ratio: 16 / 9;
        object-fit: cover;
        background: #26262e;
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
        background: #1a1a20;
        color: #e7eeee;
        border: 3px solid var(--line);
        box-shadow: 5px 5px 0 #2a2170;
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
      @media (max-width: 1260px) {
        .shell { grid-template-columns: 300px 1fr; }
        .detail { grid-column: 1 / -1; position: static; }
      }
      @media (max-width: 900px) {
        .shell { grid-template-columns: 1fr; }
        .sidebar { position: static; }
        .camera-grid { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <main>
      <div class="subnav">
        <a href="/">Browser home</a>
        <a href="/benchmark_dashboard.html">Benchmark dashboard</a>
        <a href="/telemetry_dashboard.html">Telemetry dashboard</a>
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
              <p>The detail view shows all camera streams, annotation rows, and stored sensor paths.</p>
            </div>
          </div>
        </aside>
      </div>
    </main>

    <script>
      const state = {
        q: "",
        location: "",
        category: "",
        minAnnotations: 0,
        limit: 24,
        offset: 0,
        total: 0,
      };

      const el = {
        notice: document.getElementById("notice"),
        q: document.getElementById("q"),
        location: document.getElementById("location"),
        category: document.getElementById("category"),
        minAnnotations: document.getElementById("min_annotations"),
        limit: document.getElementById("limit"),
        apply: document.getElementById("apply"),
        reset: document.getElementById("reset"),
        summary: document.getElementById("summary"),
        topLocations: document.getElementById("top_locations"),
        topCategories: document.getElementById("top_categories"),
        results: document.getElementById("results"),
        resultMeta: document.getElementById("result_meta"),
        pageMeta: document.getElementById("page_meta"),
        prev: document.getElementById("prev"),
        next: document.getElementById("next"),
        detail: document.getElementById("detail_panel"),
      };

      function showNotice(message) {
        el.notice.hidden = !message;
        el.notice.textContent = message || "";
      }

      function paramsFromState() {
        const params = new URLSearchParams();
        if (state.q) params.set("q", state.q);
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
        el.location.innerHTML = `<option value="">All locations</option>` + filters.locations
          .map((value) => `<option value="${value}">${value}</option>`)
          .join("");
        el.category.innerHTML = `<option value="">All categories</option>` + filters.categories
          .map((value) => `<option value="${value}">${value}</option>`)
          .join("");
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
            ${sample.preview_url ? `<img loading="lazy" src="${sample.preview_url}" alt="Sample ${sample.sample_idx}">` : `<div style="aspect-ratio:16/9;background:#26262e;"></div>`}
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
          </div>

          <h3 style="margin-top:18px;">Camera previews</h3>
          <div class="camera-grid">${cameraFrames || "<p>No camera blobs available.</p>"}</div>

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

      async function loadDetail(sampleIdx) {
        const sample = await requestJson(`/api/samples/${sampleIdx}`);
        renderDetail(sample);
      }

      function syncStateFromInputs() {
        state.q = el.q.value.trim();
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
          await refresh(true);
        } catch (error) {
          showNotice(error.message);
        }
      })();
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
    store = ExplorerStore(config)
    default_limit = max(1, min(result_limit, 100))
    app = FastAPI(title="nuDemo Browser", docs_url="/docs", redoc_url=None)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "reports_root": str(config.runtime.reports_root)}

    @app.get("/", response_class=HTMLResponse)
    def browser_home() -> str:
        return build_browser_home_html()

    @app.get("/explorer", response_class=HTMLResponse)
    def explorer_page() -> str:
        return build_explorer_html()

    @app.get("/api/summary")
    def api_summary() -> dict[str, Any]:
        try:
            return store.fetch_summary()
        except Exception as exc:  # pragma: no cover - depends on external services
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/filters")
    def api_filters() -> dict[str, list[str]]:
        try:
            return store.fetch_filters()
        except Exception as exc:  # pragma: no cover - depends on external services
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/samples")
    def api_samples(
        q: str | None = None,
        location: str | None = None,
        category: str | None = None,
        min_annotations: int = Query(default=0, ge=0),
        limit: int = Query(default=default_limit, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        try:
            return store.search_samples(
                q=q,
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

    app.mount("/", StaticFiles(directory=config.runtime.reports_root), name="reports")
    return app


def create_explorer_app(
    config: AppConfig | None = None,
    *,
    result_limit: int = 24,
) -> ExplorerApplication:
    return ExplorerApplication(create_app(config=config, result_limit=result_limit))
