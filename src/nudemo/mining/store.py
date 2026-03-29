# ruff: noqa: E501

from __future__ import annotations

import json
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row

from nudemo.config import PostgresSettings


class MiningSessionStore:
    def __init__(self, settings: PostgresSettings) -> None:
        self._settings = settings

    def _connection(self):
        connection = psycopg.connect(self._settings.dsn, row_factory=dict_row)
        self.ensure_schema(connection)
        return connection

    @staticmethod
    def ensure_schema(connection) -> None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
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
                """
            )
        connection.commit()

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
