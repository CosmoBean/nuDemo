from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from nudemo.benchmarks.models import BenchmarkResult
from nudemo.config import PostgresSettings
from nudemo.telemetry.docker import capture_service_snapshots

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS telemetry_runs (
    run_id VARCHAR(32) PRIMARY KEY,
    suite_name VARCHAR(128) NOT NULL,
    provider VARCHAR(32) NOT NULL,
    simulate BOOLEAN NOT NULL DEFAULT FALSE,
    sample_limit INTEGER,
    status VARCHAR(32) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    elapsed_sec DOUBLE PRECISION,
    dataset JSONB NOT NULL DEFAULT '{}'::jsonb,
    summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    report_path TEXT,
    json_path TEXT,
    csv_path TEXT,
    dashboard_path TEXT,
    telemetry_dashboard_path TEXT
);

CREATE TABLE IF NOT EXISTS telemetry_spans (
    id BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(32) NOT NULL REFERENCES telemetry_runs(run_id) ON DELETE CASCADE,
    stage VARCHAR(64) NOT NULL,
    backend VARCHAR(128) NOT NULL DEFAULT '',
    pattern VARCHAR(128) NOT NULL DEFAULT '',
    status VARCHAR(32) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    elapsed_sec DOUBLE PRECISION,
    sample_count INTEGER NOT NULL DEFAULT 0,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_telemetry_spans_run_id ON telemetry_spans(run_id);
CREATE INDEX IF NOT EXISTS idx_telemetry_spans_stage ON telemetry_spans(stage);

CREATE TABLE IF NOT EXISTS telemetry_service_snapshots (
    id BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(32) NOT NULL REFERENCES telemetry_runs(run_id) ON DELETE CASCADE,
    snapshot_label VARCHAR(64) NOT NULL,
    service VARCHAR(64) NOT NULL,
    container_name VARCHAR(128) NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    cpu_percent DOUBLE PRECISION,
    mem_percent DOUBLE PRECISION,
    mem_usage_bytes BIGINT,
    mem_limit_bytes BIGINT,
    net_input_bytes BIGINT,
    net_output_bytes BIGINT,
    block_input_bytes BIGINT,
    block_output_bytes BIGINT,
    pids INTEGER
);

CREATE INDEX IF NOT EXISTS idx_telemetry_service_run_id
ON telemetry_service_snapshots(run_id);
"""


@dataclass(slots=True)
class TelemetryRecorder:
    settings: PostgresSettings
    compose_file: Path
    run_id: str
    suite_name: str
    provider: str
    simulate: bool
    sample_limit: int | None
    started_at: datetime
    enabled: bool = True
    errors: list[str] = field(default_factory=list)

    @classmethod
    def start(
        cls,
        *,
        settings: PostgresSettings,
        compose_file: str | Path,
        run_id: str,
        suite_name: str,
        provider: str,
        simulate: bool,
        sample_limit: int | None,
    ) -> TelemetryRecorder:
        recorder = cls(
            settings=settings,
            compose_file=Path(compose_file),
            run_id=run_id,
            suite_name=suite_name,
            provider=provider,
            simulate=simulate,
            sample_limit=sample_limit,
            started_at=datetime.now(UTC),
        )
        try:
            with recorder._connect() as connection, connection.cursor() as cursor:
                ensure_schema(connection)
                cursor.execute(
                    """
                    INSERT INTO telemetry_runs (
                        run_id, suite_name, provider, simulate, sample_limit, status, started_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO NOTHING
                    """,
                    (
                        recorder.run_id,
                        recorder.suite_name,
                        recorder.provider,
                        recorder.simulate,
                        recorder.sample_limit,
                        "running",
                        recorder.started_at,
                    ),
                )
                connection.commit()
        except Exception as exc:  # pragma: no cover - depends on external services
            recorder.enabled = False
            recorder.errors.append(str(exc))
        return recorder

    def _connect(self):
        return psycopg.connect(self.settings.dsn, row_factory=dict_row)

    def snapshot_services(self, snapshot_label: str) -> None:
        if not self.enabled or self.simulate:
            return
        try:
            snapshots = capture_service_snapshots(
                self.compose_file,
                snapshot_label=snapshot_label,
            )
            if not snapshots:
                return
            with self._connect() as connection, connection.cursor() as cursor:
                ensure_schema(connection)
                for snapshot in snapshots:
                    cursor.execute(
                        """
                        INSERT INTO telemetry_service_snapshots (
                            run_id, snapshot_label, service, container_name, observed_at,
                            cpu_percent, mem_percent, mem_usage_bytes, mem_limit_bytes,
                            net_input_bytes, net_output_bytes, block_input_bytes,
                            block_output_bytes, pids
                        ) VALUES (
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, %s,
                            %s, %s
                        )
                        """,
                        (
                            self.run_id,
                            snapshot.snapshot_label,
                            snapshot.service,
                            snapshot.container_name,
                            snapshot.observed_at,
                            snapshot.cpu_percent,
                            snapshot.mem_percent,
                            snapshot.mem_usage_bytes,
                            snapshot.mem_limit_bytes,
                            snapshot.net_input_bytes,
                            snapshot.net_output_bytes,
                            snapshot.block_input_bytes,
                            snapshot.block_output_bytes,
                            snapshot.pids,
                        ),
                    )
                connection.commit()
        except Exception as exc:  # pragma: no cover - depends on docker/postgres
            self.errors.append(str(exc))

    def record_result(self, result: BenchmarkResult) -> None:
        if not self.enabled:
            return
        ended_at = datetime.now(UTC)
        started_at = ended_at - timedelta(seconds=result.elapsed_sec or 0.0)
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                ensure_schema(connection)
                cursor.execute(
                    """
                    INSERT INTO telemetry_spans (
                        run_id, stage, backend, pattern, status, started_at, ended_at,
                        elapsed_sec, sample_count, metrics, metadata, error
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        self.run_id,
                        result.stage,
                        result.backend,
                        result.pattern,
                        result.status,
                        started_at,
                        ended_at,
                        result.elapsed_sec,
                        result.sample_count,
                        Jsonb(result.metrics),
                        Jsonb(result.metadata),
                        result.error,
                    ),
                )
                connection.commit()
        except Exception as exc:  # pragma: no cover - depends on external services
            self.errors.append(str(exc))

    def complete(
        self,
        *,
        status: str,
        dataset: dict[str, object],
        summary: dict[str, object],
        report_path: str | Path | None = None,
        json_path: str | Path | None = None,
        csv_path: str | Path | None = None,
        dashboard_path: str | Path | None = None,
        telemetry_dashboard_path: str | Path | None = None,
    ) -> None:
        if not self.enabled:
            return
        completed_at = datetime.now(UTC)
        elapsed_sec = (completed_at - self.started_at).total_seconds()
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                ensure_schema(connection)
                cursor.execute(
                    """
                    UPDATE telemetry_runs
                    SET status = %s,
                        completed_at = %s,
                        elapsed_sec = %s,
                        dataset = %s,
                        summary = %s,
                        report_path = %s,
                        json_path = %s,
                        csv_path = %s,
                        dashboard_path = %s,
                        telemetry_dashboard_path = %s
                    WHERE run_id = %s
                    """,
                    (
                        status,
                        completed_at,
                        elapsed_sec,
                        Jsonb(dataset),
                        Jsonb(summary),
                        str(report_path) if report_path else None,
                        str(json_path) if json_path else None,
                        str(csv_path) if csv_path else None,
                        str(dashboard_path) if dashboard_path else None,
                        str(telemetry_dashboard_path) if telemetry_dashboard_path else None,
                        self.run_id,
                    ),
                )
                connection.commit()
        except Exception as exc:  # pragma: no cover - depends on external services
            self.errors.append(str(exc))


def ensure_schema(connection: psycopg.Connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(_SCHEMA_SQL)
    connection.commit()


def fetch_recent_runs(
    settings: PostgresSettings,
    *,
    limit: int = 10,
) -> list[dict[str, object]]:
    with psycopg.connect(settings.dsn, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            ensure_schema(connection)
            cursor.execute(
                """
                SELECT run_id, suite_name, provider, simulate, sample_limit, status,
                       started_at, completed_at, elapsed_sec, dataset, summary,
                       report_path, json_path, csv_path, dashboard_path,
                       telemetry_dashboard_path
                FROM telemetry_runs
                ORDER BY started_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]


def fetch_run_bundle(
    settings: PostgresSettings,
    *,
    run_id: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    with psycopg.connect(settings.dsn, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            ensure_schema(connection)
            if run_id is None:
                cursor.execute(
                    """
                    SELECT *
                    FROM telemetry_runs
                    ORDER BY started_at DESC
                    LIMIT 1
                    """
                )
            else:
                cursor.execute(
                    """
                    SELECT *
                    FROM telemetry_runs
                    WHERE run_id = %s
                    """,
                    (run_id,),
                )
            run = cursor.fetchone()
            if run is None:
                raise ValueError("No telemetry run found")

            cursor.execute(
                """
                SELECT run_id, stage, backend, pattern, status, started_at, ended_at,
                       elapsed_sec, sample_count, metrics, metadata, error
                FROM telemetry_spans
                WHERE run_id = %s
                ORDER BY started_at ASC, id ASC
                """,
                (run["run_id"],),
            )
            spans = [dict(row) for row in cursor.fetchall()]

            cursor.execute(
                """
                SELECT run_id, snapshot_label, service, container_name, observed_at,
                       cpu_percent, mem_percent, mem_usage_bytes, mem_limit_bytes,
                       net_input_bytes, net_output_bytes, block_input_bytes,
                       block_output_bytes, pids
                FROM telemetry_service_snapshots
                WHERE run_id = %s
                ORDER BY observed_at ASC, id ASC
                """,
                (run["run_id"],),
            )
            snapshots = [dict(row) for row in cursor.fetchall()]
    return dict(run), spans, snapshots
