from __future__ import annotations

from typing import Any
from uuid import uuid4

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
from psycopg.rows import dict_row

from nudemo.config import AppConfig
from nudemo.mining.store import CohortExportStore, MiningSessionStore, ReviewTaskStore


class CohortExportService:
    def __init__(
        self,
        config: AppConfig,
        *,
        session_store: MiningSessionStore | None = None,
        export_store: CohortExportStore | None = None,
        task_store: ReviewTaskStore | None = None,
    ) -> None:
        self._config = config
        self._session_store = session_store or MiningSessionStore(config.services.postgres)
        self._export_store = export_store or CohortExportStore(config.services.postgres)
        self._task_store = task_store or ReviewTaskStore(config.services.postgres)

    def export_cohort(
        self,
        cohort_id: str,
        *,
        task_id: str | None = None,
        manifest_version: str = "v1",
    ) -> dict[str, object]:
        cohort = self._session_store.get_cohort(cohort_id)
        sample_ids = [int(value) for value in (cohort.get("sample_ids") or [])]
        rows = self._fetch_rows(sample_ids)
        export_dir = self._config.runtime.artifacts_root / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        output_path = export_dir / f"{cohort_id}-{uuid4().hex[:8]}-{manifest_version}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), output_path)
        export = self._export_store.record_export(
            cohort_id=cohort_id,
            task_id=task_id,
            export_format="parquet",
            manifest_version=manifest_version,
            output_path=str(output_path),
            row_count=len(rows),
            metadata={
                "cohort_name": cohort.get("name", ""),
                "query": cohort.get("query", ""),
                "filters": cohort.get("filters") or {},
                "sample_count": len(sample_ids),
            },
        )
        if task_id:
            try:
                self._task_store.close_task(
                    task_id,
                    actor="export",
                    note=f"exported {len(rows)} rows",
                )
            except Exception:
                pass
        return export

    def list_exports(
        self,
        *,
        cohort_id: str | None = None,
        limit: int = 24,
    ) -> list[dict[str, object]]:
        return self._export_store.list_exports(cohort_id=cohort_id, limit=limit)

    def _fetch_rows(self, sample_ids: list[int]) -> list[dict[str, Any]]:
        if not sample_ids:
            return []
        with psycopg.connect(
            self._config.services.postgres.dsn,
            row_factory=dict_row,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    WITH requested AS (
                        SELECT sample_idx, ord
                        FROM unnest(%s::int[]) WITH ORDINALITY AS ids(sample_idx, ord)
                    ),
                    track_membership AS (
                        SELECT
                            sample_idx,
                            ARRAY_AGG(DISTINCT track_id ORDER BY track_id) AS track_ids
                        FROM track_observations
                        WHERE sample_idx = ANY(%s::int[])
                        GROUP BY sample_idx
                    )
                    SELECT
                        requested.ord,
                        s.sample_idx,
                        s.token,
                        s.scene_token,
                        COALESCE(sc.scene_name, s.scene_token) AS scene_name,
                        s.location,
                        s.timestamp,
                        s.num_annotations,
                        s.num_lidar_points,
                        s.cam_front_path,
                        s.cam_front_left_path,
                        s.cam_front_right_path,
                        s.cam_back_path,
                        s.cam_back_left_path,
                        s.cam_back_right_path,
                        s.lidar_top_path,
                        s.radar_front_path,
                        s.radar_front_left_path,
                        s.radar_front_right_path,
                        s.radar_back_left_path,
                        s.radar_back_right_path,
                        COALESCE(tm.track_ids, '{}'::varchar[]) AS track_ids
                    FROM requested
                    JOIN samples s ON s.sample_idx = requested.sample_idx
                    LEFT JOIN scenes sc ON sc.scene_token = s.scene_token
                    LEFT JOIN track_membership tm ON tm.sample_idx = s.sample_idx
                    ORDER BY requested.ord
                    """,
                    (sample_ids, sample_ids),
                )
                rows = [dict(row) for row in cursor.fetchall()]
        for row in rows:
            row["track_ids"] = list(row.get("track_ids") or [])
        return rows
