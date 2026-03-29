# ruff: noqa: E501

from __future__ import annotations

import json
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row

from nudemo.config import PostgresSettings

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS mining_sessions (
    session_id VARCHAR(32) PRIMARY KEY,
    label TEXT NOT NULL DEFAULT '',
    query TEXT NOT NULL DEFAULT '',
    mode VARCHAR(24) NOT NULL DEFAULT 'hybrid',
    modality_weights JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mining_session_examples (
    session_id VARCHAR(32) NOT NULL REFERENCES mining_sessions(session_id) ON DELETE CASCADE,
    sample_idx INTEGER NOT NULL REFERENCES samples(sample_idx) ON DELETE CASCADE,
    polarity VARCHAR(16) NOT NULL CHECK (polarity IN ('positive', 'negative')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (session_id, sample_idx)
);

CREATE INDEX IF NOT EXISTS idx_mining_session_examples_polarity
    ON mining_session_examples(session_id, polarity);

CREATE TABLE IF NOT EXISTS mining_cohorts (
    cohort_id VARCHAR(32) PRIMARY KEY,
    session_id VARCHAR(32) REFERENCES mining_sessions(session_id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    query TEXT NOT NULL DEFAULT '',
    filters JSONB NOT NULL DEFAULT '{}'::jsonb,
    sample_ids INTEGER[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mining_cohorts_created_at
    ON mining_cohorts(created_at DESC);

CREATE TABLE IF NOT EXISTS tracks (
    track_id VARCHAR(64) PRIMARY KEY,
    scene_token VARCHAR(64) NOT NULL REFERENCES scenes(scene_token) ON DELETE CASCADE,
    scene_name VARCHAR(64) NOT NULL,
    location VARCHAR(64) NOT NULL,
    category VARCHAR(128) NOT NULL,
    start_timestamp BIGINT NOT NULL,
    end_timestamp BIGINT NOT NULL,
    sample_ids INTEGER[] NOT NULL DEFAULT '{}',
    sample_count INTEGER NOT NULL DEFAULT 0,
    annotation_count INTEGER NOT NULL DEFAULT 0,
    avg_num_lidar_pts DOUBLE PRECISION NOT NULL DEFAULT 0,
    avg_num_radar_pts DOUBLE PRECISION NOT NULL DEFAULT 0,
    max_num_lidar_pts INTEGER NOT NULL DEFAULT 0,
    max_num_radar_pts INTEGER NOT NULL DEFAULT 0,
    visibility_tokens VARCHAR(16)[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tracks_scene_category
    ON tracks(scene_token, category);

CREATE INDEX IF NOT EXISTS idx_tracks_location
    ON tracks(location);

CREATE TABLE IF NOT EXISTS track_observations (
    id BIGSERIAL PRIMARY KEY,
    track_id VARCHAR(64) NOT NULL REFERENCES tracks(track_id) ON DELETE CASCADE,
    sample_idx INTEGER NOT NULL REFERENCES samples(sample_idx) ON DELETE CASCADE,
    sample_token VARCHAR(64) NOT NULL,
    annotation_token VARCHAR(64) NOT NULL,
    observation_idx INTEGER NOT NULL,
    timestamp BIGINT NOT NULL,
    category VARCHAR(128) NOT NULL,
    translation DOUBLE PRECISION[],
    size DOUBLE PRECISION[],
    rotation DOUBLE PRECISION[],
    num_lidar_pts INTEGER NOT NULL DEFAULT 0,
    num_radar_pts INTEGER NOT NULL DEFAULT 0,
    visibility_token VARCHAR(16) NOT NULL DEFAULT '',
    attribute_tokens VARCHAR(64)[] NOT NULL DEFAULT '{}',
    UNIQUE(track_id, sample_idx, annotation_token)
);

CREATE INDEX IF NOT EXISTS idx_track_observations_track_sample
    ON track_observations(track_id, sample_idx);

CREATE INDEX IF NOT EXISTS idx_track_observations_timestamp
    ON track_observations(timestamp);

CREATE TABLE IF NOT EXISTS review_tasks (
    task_id VARCHAR(32) PRIMARY KEY,
    source_type VARCHAR(24) NOT NULL CHECK (source_type IN ('cohort', 'track', 'manual')),
    source_id VARCHAR(64),
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status VARCHAR(24) NOT NULL CHECK (
        status IN ('queued', 'assigned', 'in_progress', 'submitted', 'qa_failed', 'qa_passed', 'closed')
    ),
    assignee TEXT NOT NULL DEFAULT '',
    priority VARCHAR(16) NOT NULL DEFAULT 'normal',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    submitted_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_review_tasks_status_created
    ON review_tasks(status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_review_tasks_source
    ON review_tasks(source_type, source_id);

CREATE TABLE IF NOT EXISTS task_events (
    id BIGSERIAL PRIMARY KEY,
    task_id VARCHAR(32) NOT NULL REFERENCES review_tasks(task_id) ON DELETE CASCADE,
    event_type VARCHAR(32) NOT NULL,
    actor TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_task_events_task_created
    ON task_events(task_id, created_at);

CREATE TABLE IF NOT EXISTS cohort_exports (
    export_id VARCHAR(32) PRIMARY KEY,
    cohort_id VARCHAR(32) REFERENCES mining_cohorts(cohort_id) ON DELETE CASCADE,
    task_id VARCHAR(32) REFERENCES review_tasks(task_id) ON DELETE SET NULL,
    export_format VARCHAR(16) NOT NULL,
    manifest_version VARCHAR(16) NOT NULL DEFAULT 'v1',
    output_path TEXT NOT NULL,
    row_count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cohort_exports_cohort_created
    ON cohort_exports(cohort_id, created_at DESC);
"""

VALID_TASK_STATUSES = {
    "queued",
    "assigned",
    "in_progress",
    "submitted",
    "qa_failed",
    "qa_passed",
    "closed",
}
VALID_TASK_PRIORITIES = {"low", "normal", "high", "critical"}
VALID_SOURCE_TYPES = {"cohort", "track", "manual"}
TASK_TRANSITIONS = {
    "queued": {"assigned", "closed"},
    "assigned": {"in_progress", "submitted", "closed"},
    "in_progress": {"submitted", "closed"},
    "submitted": {"qa_failed", "qa_passed", "closed"},
    "qa_failed": {"assigned", "in_progress", "closed"},
    "qa_passed": {"closed"},
    "closed": set(),
}
TASK_STATUSES = tuple(sorted(VALID_TASK_STATUSES))
TASK_PRIORITIES = tuple(sorted(VALID_TASK_PRIORITIES))


def ensure_schema(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(SCHEMA_SQL)
    connection.commit()


class PostgresStore:
    def __init__(self, settings: PostgresSettings) -> None:
        self._settings = settings

    def _connection(self):
        connection = psycopg.connect(self._settings.dsn, row_factory=dict_row)
        ensure_schema(connection)
        return connection

    @staticmethod
    def ensure_schema(connection) -> None:
        ensure_schema(connection)


class MiningSessionStore(PostgresStore):
    def create_session(
        self,
        *,
        label: str = "",
        query: str = "",
        mode: str = "hybrid",
        modality_weights: dict[str, float] | None = None,
    ) -> dict[str, object]:
        session_id = uuid4().hex[:16]
        payload = json.dumps(modality_weights or {})
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO mining_sessions (session_id, label, query, mode, modality_weights)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                """,
                (session_id, label, query, mode, payload),
            )
            connection.commit()
        return self.get_session(session_id)

    def list_sessions(self, *, limit: int = 24) -> list[dict[str, object]]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    s.session_id,
                    s.label,
                    s.query,
                    s.mode,
                    s.modality_weights,
                    s.created_at,
                    s.updated_at,
                    COUNT(*) FILTER (WHERE e.polarity = 'positive') AS positive_count,
                    COUNT(*) FILTER (WHERE e.polarity = 'negative') AS negative_count
                FROM mining_sessions s
                LEFT JOIN mining_session_examples e ON e.session_id = s.session_id
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
                LIMIT %s
                """,
                (max(1, min(limit, 100)),),
            )
            rows = [dict(row) for row in cursor.fetchall()]
        for row in rows:
            row["modality_weights"] = dict(row.get("modality_weights") or {})
        return rows

    def get_session(self, session_id: str) -> dict[str, object]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT session_id, label, query, mode, modality_weights, created_at, updated_at
                FROM mining_sessions
                WHERE session_id = %s
                """,
                (session_id,),
            )
            session = cursor.fetchone()
            if session is None:
                raise KeyError(session_id)
            cursor.execute(
                """
                SELECT sample_idx, polarity, created_at
                FROM mining_session_examples
                WHERE session_id = %s
                ORDER BY polarity, created_at, sample_idx
                """,
                (session_id,),
            )
            examples = [dict(row) for row in cursor.fetchall()]
        payload = dict(session)
        payload["modality_weights"] = dict(payload.get("modality_weights") or {})
        payload["examples"] = examples
        payload["positive_sample_ids"] = [
            int(row["sample_idx"]) for row in examples if row["polarity"] == "positive"
        ]
        payload["negative_sample_ids"] = [
            int(row["sample_idx"]) for row in examples if row["polarity"] == "negative"
        ]
        return payload

    def set_example(self, session_id: str, *, sample_idx: int, polarity: str) -> dict[str, object]:
        if polarity not in {"positive", "negative"}:
            raise ValueError("polarity must be positive or negative")
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO mining_session_examples (session_id, sample_idx, polarity)
                VALUES (%s, %s, %s)
                ON CONFLICT (session_id, sample_idx)
                DO UPDATE SET polarity = EXCLUDED.polarity, created_at = NOW()
                """,
                (session_id, sample_idx, polarity),
            )
            cursor.execute(
                """
                UPDATE mining_sessions
                SET updated_at = NOW()
                WHERE session_id = %s
                """,
                (session_id,),
            )
            connection.commit()
        return self.get_session(session_id)

    def replace_examples(
        self,
        session_id: str,
        *,
        positive_sample_ids: list[int],
        negative_sample_ids: list[int],
        query: str,
        mode: str,
        modality_weights: dict[str, float] | None,
    ) -> dict[str, object]:
        positive_ids = [int(value) for value in positive_sample_ids]
        negative_ids = [int(value) for value in negative_sample_ids if int(value) not in positive_ids]
        rows = [(session_id, sample_idx, "positive") for sample_idx in positive_ids]
        rows.extend((session_id, sample_idx, "negative") for sample_idx in negative_ids)
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE mining_sessions
                SET query = %s,
                    mode = %s,
                    modality_weights = %s::jsonb,
                    updated_at = NOW()
                WHERE session_id = %s
                """,
                (
                    query,
                    mode,
                    json.dumps(modality_weights or {}),
                    session_id,
                ),
            )
            cursor.execute(
                "DELETE FROM mining_session_examples WHERE session_id = %s",
                (session_id,),
            )
            if rows:
                cursor.executemany(
                    """
                    INSERT INTO mining_session_examples (session_id, sample_idx, polarity)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (session_id, sample_idx)
                    DO UPDATE SET polarity = EXCLUDED.polarity, created_at = NOW()
                    """,
                    rows,
                )
            connection.commit()
        return self.get_session(session_id)

    def remove_example(self, session_id: str, *, sample_idx: int) -> dict[str, object]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM mining_session_examples
                WHERE session_id = %s AND sample_idx = %s
                """,
                (session_id, sample_idx),
            )
            cursor.execute(
                """
                UPDATE mining_sessions
                SET updated_at = NOW()
                WHERE session_id = %s
                """,
                (session_id,),
            )
            connection.commit()
        return self.get_session(session_id)

    def save_cohort(
        self,
        session_id: str | None,
        *,
        name: str,
        query: str,
        filters: dict[str, object],
        sample_ids: list[int],
    ) -> dict[str, object]:
        cohort_id = uuid4().hex[:16]
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO mining_cohorts (cohort_id, session_id, name, query, filters, sample_ids)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::int[])
                """,
                (cohort_id, session_id, name, query, json.dumps(filters), sample_ids),
            )
            connection.commit()
        return self.get_cohort(cohort_id)

    def get_cohort(self, cohort_id: str) -> dict[str, object]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT cohort_id, session_id, name, query, filters, sample_ids, created_at
                FROM mining_cohorts
                WHERE cohort_id = %s
                """,
                (cohort_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(cohort_id)
        payload = dict(row)
        payload["filters"] = dict(payload.get("filters") or {})
        payload["sample_ids"] = [int(value) for value in (payload.get("sample_ids") or [])]
        return payload

    def list_cohorts(self, *, limit: int = 24) -> list[dict[str, object]]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT cohort_id, session_id, name, query, filters, sample_ids, created_at
                FROM mining_cohorts
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (max(1, min(limit, 100)),),
            )
            rows = [dict(row) for row in cursor.fetchall()]
        for row in rows:
            row["filters"] = dict(row.get("filters") or {})
            row["sample_ids"] = [int(value) for value in (row.get("sample_ids") or [])]
        return rows


class TrackStore(PostgresStore):
    def fetch_loaded_samples(
        self,
        *,
        limit: int | None = None,
        scene_limit: int | None = None,
    ) -> list[dict[str, object]]:
        params: list[object] = []
        cte = ""
        join = ""
        if scene_limit is not None:
            cte = (
                "WITH selected_scenes AS ("
                "SELECT scene_token FROM scenes ORDER BY scene_name LIMIT %s"
                ") "
            )
            join = "JOIN selected_scenes ss ON ss.scene_token = s.scene_token"
            params.append(max(1, int(scene_limit)))
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT %s"
            params.append(max(1, int(limit)))
        sql = f"""
            {cte}
            SELECT
                s.sample_idx,
                s.token,
                s.scene_token,
                sc.scene_name,
                s.location,
                s.timestamp
            FROM samples s
            JOIN scenes sc ON sc.scene_token = s.scene_token
            {join}
            ORDER BY s.timestamp, s.sample_idx
            {limit_clause}
        """
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def replace_tracks(
        self,
        *,
        tracks: list[dict[str, object]],
        observations: list[dict[str, object]],
    ) -> dict[str, int]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute("TRUNCATE track_observations, tracks RESTART IDENTITY CASCADE")
            if tracks:
                cursor.executemany(
                    """
                    INSERT INTO tracks (
                        track_id, scene_token, scene_name, location, category,
                        start_timestamp, end_timestamp, sample_ids, sample_count,
                        annotation_count, avg_num_lidar_pts, avg_num_radar_pts,
                        max_num_lidar_pts, max_num_radar_pts, visibility_tokens
                    )
                    VALUES (
                        %(track_id)s, %(scene_token)s, %(scene_name)s, %(location)s, %(category)s,
                        %(start_timestamp)s, %(end_timestamp)s, %(sample_ids)s::int[], %(sample_count)s,
                        %(annotation_count)s, %(avg_num_lidar_pts)s, %(avg_num_radar_pts)s,
                        %(max_num_lidar_pts)s, %(max_num_radar_pts)s, %(visibility_tokens)s::varchar[]
                    )
                    """,
                    tracks,
                )
            if observations:
                cursor.executemany(
                    """
                    INSERT INTO track_observations (
                        track_id, sample_idx, sample_token, annotation_token, observation_idx,
                        timestamp, category, translation, size, rotation,
                        num_lidar_pts, num_radar_pts, visibility_token, attribute_tokens
                    )
                    VALUES (
                        %(track_id)s, %(sample_idx)s, %(sample_token)s, %(annotation_token)s, %(observation_idx)s,
                        %(timestamp)s, %(category)s, %(translation)s, %(size)s, %(rotation)s,
                        %(num_lidar_pts)s, %(num_radar_pts)s, %(visibility_token)s, %(attribute_tokens)s::varchar[]
                    )
                    """,
                    observations,
                )
            connection.commit()
        return {"tracks": len(tracks), "observations": len(observations)}

    def search_tracks(
        self,
        *,
        q: str = "",
        scene_token: str = "",
        location: str = "",
        category: str = "",
        limit: int = 12,
        offset: int = 0,
    ) -> dict[str, object]:
        pattern = f"%{q.strip()}%" if q and q.strip() else None
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                WITH filtered AS (
                    SELECT
                        t.*,
                        t.sample_ids[1] AS preview_sample_idx
                    FROM tracks t
                    WHERE (%(scene_token)s::text = '' OR t.scene_token = %(scene_token)s::text)
                      AND (%(location)s::text = '' OR t.location = %(location)s::text)
                      AND (%(category)s::text = '' OR t.category = %(category)s::text)
                      AND (
                        %(pattern)s::text IS NULL
                        OR t.track_id ILIKE %(pattern)s::text
                        OR t.scene_name ILIKE %(pattern)s::text
                        OR t.scene_token ILIKE %(pattern)s::text
                        OR t.location ILIKE %(pattern)s::text
                        OR t.category ILIKE %(pattern)s::text
                      )
                ),
                counted AS (
                    SELECT filtered.*, COUNT(*) OVER() AS total_count
                    FROM filtered
                )
                SELECT *
                FROM counted
                ORDER BY sample_count DESC, start_timestamp ASC, track_id
                LIMIT %(limit)s
                OFFSET %(offset)s
                """,
                {
                    "scene_token": scene_token,
                    "location": location,
                    "category": category,
                    "pattern": pattern,
                    "limit": max(1, min(limit, 100)),
                    "offset": max(0, offset),
                },
            )
            rows = [dict(row) for row in cursor.fetchall()]
        total = int(rows[0]["total_count"]) if rows else 0
        return {
            "total": total,
            "items": [self._row_to_track_summary(row) for row in rows],
        }

    def hydrate_tracks(self, track_ids: list[str]) -> list[dict[str, object]]:
        if not track_ids:
            return []
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                WITH requested AS (
                    SELECT track_id, ord
                    FROM unnest(%s::varchar[]) WITH ORDINALITY AS ids(track_id, ord)
                )
                SELECT t.*, t.sample_ids[1] AS preview_sample_idx, requested.ord
                FROM requested
                JOIN tracks t ON t.track_id = requested.track_id
                ORDER BY requested.ord
                """,
                (track_ids,),
            )
            rows = [dict(row) for row in cursor.fetchall()]
        return [self._row_to_track_summary(row) for row in rows]

    def fetch_tracks_by_ids(self, track_ids: list[str]) -> list[dict[str, object]]:
        return self.hydrate_tracks(track_ids)

    def get_track(self, track_id: str, *, observation_limit: int = 180) -> dict[str, object]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT t.*, t.sample_ids[1] AS preview_sample_idx
                FROM tracks t
                WHERE t.track_id = %s
                """,
                (track_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(track_id)
            cursor.execute(
                """
                SELECT
                    o.track_id,
                    o.sample_idx,
                    o.sample_token,
                    o.annotation_token,
                    o.observation_idx,
                    o.timestamp,
                    o.category,
                    o.translation,
                    o.size,
                    o.rotation,
                    o.num_lidar_pts,
                    o.num_radar_pts,
                    o.visibility_token,
                    o.attribute_tokens,
                    s.token,
                    s.scene_token,
                    COALESCE(sc.scene_name, s.scene_token) AS scene_name,
                    s.location,
                    s.num_annotations,
                    s.num_lidar_points,
                    s.cam_front_path
                FROM track_observations o
                JOIN samples s ON s.sample_idx = o.sample_idx
                LEFT JOIN scenes sc ON sc.scene_token = s.scene_token
                WHERE o.track_id = %s
                ORDER BY o.observation_idx
                LIMIT %s
                """,
                (track_id, max(1, min(observation_limit, 480))),
            )
            observations = []
            for obs in cursor.fetchall():
                payload = dict(obs)
                payload["preview_url"] = (
                    f"/api/samples/{payload['sample_idx']}/cameras/CAM_FRONT"
                    if payload.get("cam_front_path")
                    else None
                )
                observations.append(payload)
        result = self._row_to_track_summary(dict(row))
        result["observations"] = observations
        return result

    def summary(self) -> dict[str, object]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM tracks) AS track_count,
                    (SELECT COUNT(*) FROM track_observations) AS observation_count
                """
            )
            payload = dict(cursor.fetchone() or {})
            cursor.execute(
                """
                SELECT category, COUNT(*) AS count
                FROM tracks
                GROUP BY category
                ORDER BY count DESC, category
                LIMIT 12
                """
            )
            payload["top_track_categories"] = [dict(row) for row in cursor.fetchall()]
        return payload

    @staticmethod
    def _row_to_track_summary(row: dict[str, object]) -> dict[str, object]:
        sample_ids = [int(value) for value in (row.get("sample_ids") or [])]
        preview_sample_idx = row.get("preview_sample_idx")
        return {
            "track_id": str(row.get("track_id") or ""),
            "scene_token": str(row.get("scene_token") or ""),
            "scene_name": str(row.get("scene_name") or ""),
            "location": str(row.get("location") or ""),
            "category": str(row.get("category") or ""),
            "start_timestamp": int(row.get("start_timestamp") or 0),
            "end_timestamp": int(row.get("end_timestamp") or 0),
            "sample_ids": sample_ids,
            "sample_count": int(row.get("sample_count") or len(sample_ids)),
            "annotation_count": int(row.get("annotation_count") or 0),
            "avg_num_lidar_pts": float(row.get("avg_num_lidar_pts") or 0.0),
            "avg_num_radar_pts": float(row.get("avg_num_radar_pts") or 0.0),
            "max_num_lidar_pts": int(row.get("max_num_lidar_pts") or 0),
            "max_num_radar_pts": int(row.get("max_num_radar_pts") or 0),
            "visibility_tokens": list(row.get("visibility_tokens") or []),
            "preview_sample_idx": int(preview_sample_idx) if preview_sample_idx is not None else None,
            "preview_url": (
                f"/api/samples/{int(preview_sample_idx)}/cameras/CAM_FRONT"
                if preview_sample_idx is not None
                else None
            ),
        }


class ReviewTaskStore(PostgresStore):
    def create_task(
        self,
        *,
        source_type: str,
        source_id: str | None,
        title: str,
        description: str = "",
        priority: str = "normal",
        assignee: str = "",
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        normalized_source_type = source_type.strip().lower()
        normalized_priority = priority.strip().lower()
        if normalized_source_type not in VALID_SOURCE_TYPES:
            raise ValueError(f"invalid source_type: {source_type}")
        if normalized_priority not in VALID_TASK_PRIORITIES:
            raise ValueError(f"invalid priority: {priority}")
        if not title.strip():
            raise ValueError("title must not be empty")
        task_id = uuid4().hex[:16]
        initial_status = "assigned" if assignee else "queued"
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO review_tasks (
                    task_id, source_type, source_id, title, description,
                    status, assignee, priority, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    task_id,
                    normalized_source_type,
                    source_id,
                    title.strip(),
                    description.strip(),
                    initial_status,
                    assignee.strip(),
                    normalized_priority,
                    json.dumps(metadata or {}),
                ),
            )
            self._record_event(
                cursor,
                task_id=task_id,
                event_type="created",
                actor=assignee.strip(),
                payload={"status": initial_status, "metadata": metadata or {}},
            )
            connection.commit()
        return self.get_task(task_id)

    def list_tasks(
        self,
        *,
        status: str | None = None,
        source_type: str | None = None,
        source_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    t.*,
                    COUNT(e.id) AS event_count
                FROM review_tasks t
                LEFT JOIN task_events e ON e.task_id = t.task_id
                WHERE (%s::text IS NULL OR t.status = %s::text)
                  AND (%s::text IS NULL OR t.source_type = %s::text)
                  AND (%s::text IS NULL OR t.source_id = %s::text)
                GROUP BY t.task_id
                ORDER BY
                    CASE t.status
                        WHEN 'in_progress' THEN 0
                        WHEN 'assigned' THEN 1
                        WHEN 'queued' THEN 2
                        WHEN 'submitted' THEN 3
                        WHEN 'qa_failed' THEN 4
                        WHEN 'qa_passed' THEN 5
                        ELSE 6
                    END,
                    t.updated_at DESC
                LIMIT %s
                """,
                (
                    status if status else None,
                    status if status else None,
                    source_type if source_type else None,
                    source_type if source_type else None,
                    source_id if source_id else None,
                    source_id if source_id else None,
                    max(1, min(limit, 200)),
                ),
            )
            rows = [dict(row) for row in cursor.fetchall()]
        return [self._normalize_task_row(row) for row in rows]

    def get_task(self, task_id: str) -> dict[str, object]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM review_tasks
                WHERE task_id = %s
                """,
                (task_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(task_id)
            cursor.execute(
                """
                SELECT task_id, event_type, actor, payload, created_at
                FROM task_events
                WHERE task_id = %s
                ORDER BY created_at
                """,
                (task_id,),
            )
            events = [dict(event) for event in cursor.fetchall()]
        payload = self._normalize_task_row(dict(row))
        payload["events"] = events
        return payload

    def task_summary(self) -> dict[str, object]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM review_tasks
                GROUP BY status
                """
            )
            counts = {row["status"]: int(row["count"]) for row in cursor.fetchall()}
            cursor.execute(
                """
                SELECT
                    AVG(EXTRACT(EPOCH FROM (COALESCE(closed_at, updated_at) - created_at))) FILTER (
                        WHERE status IN ('qa_passed', 'closed')
                    ) AS avg_cycle_time_sec
                FROM review_tasks
                """
            )
            avg_cycle_row = cursor.fetchone() or {}
        return {
            "counts": counts,
            "avg_cycle_time_sec": float(avg_cycle_row.get("avg_cycle_time_sec") or 0.0),
            "total": int(sum(counts.values())),
        }

    def claim_task(self, task_id: str, *, actor: str, assignee: str | None = None) -> dict[str, object]:
        return self._transition_task(
            task_id,
            next_status="assigned",
            event_type="claimed",
            actor=actor,
            assignee=(assignee or actor).strip(),
        )

    def start_task(self, task_id: str, *, actor: str) -> dict[str, object]:
        return self._transition_task(
            task_id,
            next_status="in_progress",
            event_type="started",
            actor=actor,
            assignee=actor,
        )

    def submit_task(
        self,
        task_id: str,
        *,
        actor: str,
        note: str = "",
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return self._transition_task(
            task_id,
            next_status="submitted",
            event_type="submitted",
            actor=actor,
            assignee=actor,
            note=note,
            patch={
                "submitted_at = NOW()",
                "metadata = COALESCE(review_tasks.metadata, '{}'::jsonb) || %s::jsonb",
            },
            patch_params=[json.dumps(metadata or {})],
        )

    def qa_task(
        self,
        task_id: str,
        *,
        actor: str,
        passed: bool,
        note: str = "",
    ) -> dict[str, object]:
        return self._transition_task(
            task_id,
            next_status="qa_passed" if passed else "qa_failed",
            event_type="qa_passed" if passed else "qa_failed",
            actor=actor,
            note=note,
        )

    def close_task(self, task_id: str, *, actor: str, note: str = "") -> dict[str, object]:
        return self._transition_task(
            task_id,
            next_status="closed",
            event_type="closed",
            actor=actor,
            note=note,
            patch={"closed_at = NOW()"},
        )

    def _transition_task(
        self,
        task_id: str,
        *,
        next_status: str,
        event_type: str,
        actor: str,
        assignee: str | None = None,
        note: str = "",
        patch: set[str] | None = None,
        patch_params: list[object] | None = None,
    ) -> dict[str, object]:
        if next_status not in VALID_TASK_STATUSES:
            raise ValueError(f"invalid task status: {next_status}")

        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT task_id, status, assignee
                FROM review_tasks
                WHERE task_id = %s
                FOR UPDATE
                """,
                (task_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(task_id)
            current_status = str(row["status"])
            if next_status == current_status:
                pass
            elif next_status not in TASK_TRANSITIONS.get(current_status, set()):
                raise ValueError(f"cannot transition task {task_id} from {current_status} to {next_status}")

            assignments = ["status = %s", "updated_at = NOW()"]
            params: list[object] = [next_status]
            if assignee is not None:
                assignments.append("assignee = %s")
                params.append(assignee)
            if patch:
                assignments.extend(sorted(patch))
            if patch_params:
                params.extend(patch_params)
            params.append(task_id)
            cursor.execute(
                f"""
                UPDATE review_tasks
                SET {", ".join(assignments)}
                WHERE task_id = %s
                """,
                params,
            )
            self._record_event(
                cursor,
                task_id=task_id,
                event_type=event_type,
                actor=actor,
                payload={
                    "from_status": current_status,
                    "to_status": next_status,
                    "note": note,
                    "assignee": assignee,
                },
            )
            connection.commit()
        return self.get_task(task_id)

    @staticmethod
    def _record_event(
        cursor,
        *,
        task_id: str,
        event_type: str,
        actor: str,
        payload: dict[str, object],
    ) -> None:
        cursor.execute(
            """
            INSERT INTO task_events (task_id, event_type, actor, payload)
            VALUES (%s, %s, %s, %s::jsonb)
            """,
            (task_id, event_type, actor, json.dumps(payload)),
        )

    @staticmethod
    def _normalize_task_row(row: dict[str, object]) -> dict[str, object]:
        payload = dict(row)
        payload["metadata"] = dict(payload.get("metadata") or {})
        payload["event_count"] = int(payload.get("event_count") or 0)
        return payload


class CohortExportStore(PostgresStore):
    def record_export(
        self,
        *,
        cohort_id: str,
        task_id: str | None,
        export_format: str,
        manifest_version: str,
        output_path: str,
        row_count: int,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        export_id = uuid4().hex[:16]
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO cohort_exports (
                    export_id, cohort_id, task_id, export_format, manifest_version,
                    output_path, row_count, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    export_id,
                    cohort_id,
                    task_id,
                    export_format,
                    manifest_version,
                    output_path,
                    int(row_count),
                    json.dumps(metadata or {}),
                ),
            )
            connection.commit()
        return self.get_export(export_id)

    def get_export(self, export_id: str) -> dict[str, object]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM cohort_exports
                WHERE export_id = %s
                """,
                (export_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(export_id)
        payload = dict(row)
        payload["metadata"] = dict(payload.get("metadata") or {})
        payload["row_count"] = int(payload.get("row_count") or 0)
        return payload

    def list_exports(
        self,
        *,
        cohort_id: str | None = None,
        limit: int = 24,
    ) -> list[dict[str, object]]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM cohort_exports
                WHERE (%s::text IS NULL OR cohort_id = %s::text)
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (
                    cohort_id if cohort_id else None,
                    cohort_id if cohort_id else None,
                    max(1, min(limit, 100)),
                ),
            )
            rows = [dict(row) for row in cursor.fetchall()]
        for row in rows:
            row["metadata"] = dict(row.get("metadata") or {})
            row["row_count"] = int(row.get("row_count") or 0)
        return rows


def fetch_workflow_metrics(settings: PostgresSettings) -> dict[str, object]:
    with psycopg.connect(settings.dsn, row_factory=dict_row) as connection:
        ensure_schema(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM tracks) AS track_count,
                    (SELECT COUNT(*) FROM track_observations) AS track_observation_count,
                    (SELECT COUNT(*) FROM review_tasks) AS task_count,
                    (SELECT COUNT(*) FROM cohort_exports) AS cohort_export_count
                """
            )
            summary = dict(cursor.fetchone() or {})
            cursor.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM review_tasks
                GROUP BY status
                """
            )
            summary["task_counts"] = {
                row["status"]: int(row["count"]) for row in cursor.fetchall()
            }
            cursor.execute(
                """
                SELECT
                    AVG(EXTRACT(EPOCH FROM (COALESCE(closed_at, updated_at) - created_at))) FILTER (
                        WHERE status IN ('qa_passed', 'closed')
                    ) AS avg_cycle_time_sec
                FROM review_tasks
                """
            )
            cycle_row = dict(cursor.fetchone() or {})
            summary["avg_cycle_time_sec"] = float(cycle_row.get("avg_cycle_time_sec") or 0.0)
    return summary


def validate_task_transition(current_status: str, next_status: str) -> bool:
    if current_status not in VALID_TASK_STATUSES or next_status not in VALID_TASK_STATUSES:
        return False
    return next_status in TASK_TRANSITIONS.get(current_status, set()) or current_status == next_status
