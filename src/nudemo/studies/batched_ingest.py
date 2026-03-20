from __future__ import annotations

import csv
import io
import json
import shutil
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from nudemo.benchmarks.export import export_report_bundle
from nudemo.benchmarks.models import BenchmarkReport, BenchmarkResult
from nudemo.benchmarks.runner import (
    benchmark_curation_query,
    benchmark_end_to_end_curation,
    benchmark_random_access,
    benchmark_sequential,
    build_disk_record,
    record_to_result,
)
from nudemo.config import AppConfig
from nudemo.domain.models import CAMERAS, RADARS, UnifiedSample
from nudemo.extraction.providers import resolve_provider
from nudemo.reporting.dashboard import build_dashboard_html
from nudemo.storage.base import (
    StorageWriteResult,
    array_to_npy_bytes,
    image_to_jpeg_bytes,
)
from nudemo.storage.lance_store import LanceBackend
from nudemo.storage.minio_postgres import MinioPostgresBackend
from nudemo.storage.parquet_store import ParquetBackend
from nudemo.storage.redis_store import RedisBackend
from nudemo.storage.webdataset_store import WebDatasetBackend
from nudemo.telemetry.dashboard import build_telemetry_dashboard_html
from nudemo.telemetry.store import TelemetryRecorder, fetch_run_bundle


@dataclass(slots=True)
class StudyOptions:
    provider: str = "real"
    limit: int | None = None
    batch_size: int = 32
    random_sample_count: int = 256
    snapshot_every_batches: int = 1
    purge_after_backend: bool = True
    keep_backend: str | None = "minio-postgres"


@dataclass(slots=True)
class BackendStudySummary:
    backend_key: str
    backend_name: str
    run_id: str
    status: str
    samples: int
    scenes: int
    batch_count: int
    ingest_elapsed_sec: float
    batch_p50_sec: float
    batch_p95_sec: float
    ingest_throughput_mean: float
    random_access_p50_ms: float | None
    sequential_throughput: float | None
    curation_query_ms: float | None
    disk_gb: float
    peak_service_cpu: float | None
    peak_service_name: str | None
    benchmark_dashboard: str
    telemetry_dashboard: str
    report_dir: str

    def as_dict(self) -> dict[str, object]:
        return {
            "backend_key": self.backend_key,
            "backend_name": self.backend_name,
            "run_id": self.run_id,
            "status": self.status,
            "samples": self.samples,
            "scenes": self.scenes,
            "batch_count": self.batch_count,
            "ingest_elapsed_sec": round(self.ingest_elapsed_sec, 4),
            "batch_p50_sec": round(self.batch_p50_sec, 4),
            "batch_p95_sec": round(self.batch_p95_sec, 4),
            "ingest_throughput_mean": round(self.ingest_throughput_mean, 4),
            "random_access_p50_ms": _round_or_none(self.random_access_p50_ms),
            "sequential_throughput": _round_or_none(self.sequential_throughput),
            "curation_query_ms": _round_or_none(self.curation_query_ms),
            "disk_gb": round(self.disk_gb, 4),
            "peak_service_cpu": _round_or_none(self.peak_service_cpu),
            "peak_service_name": self.peak_service_name,
            "benchmark_dashboard": self.benchmark_dashboard,
            "telemetry_dashboard": self.telemetry_dashboard,
            "report_dir": self.report_dir,
        }


class BatchWriter:
    backend_key: str
    backend_name: str

    def reset(self) -> None:
        raise NotImplementedError

    def append_batch(self, start_idx: int, samples: Sequence[UnifiedSample]) -> StorageWriteResult:
        raise NotImplementedError

    def reader(self):
        raise NotImplementedError

    def finalize(self) -> None:
        return None

    def cleanup(self) -> None:
        self.reset()


class MinioPostgresBatchWriter(BatchWriter):
    backend_key = "minio-postgres"
    backend_name = "MinIO+PostgreSQL"

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._reader = MinioPostgresBackend(
            minio=config.services.minio,
            postgres=config.services.postgres,
        )

    def _clients(self):
        import psycopg
        from minio import Minio

        minio_client = Minio(
            self._config.services.minio.endpoint,
            access_key=self._config.services.minio.access_key,
            secret_key=self._config.services.minio.secret_key,
            secure=self._config.services.minio.secure,
        )
        connection = psycopg.connect(self._config.services.postgres.dsn)
        return minio_client, connection

    def reset(self) -> None:
        minio_client, connection = self._clients()
        try:
            if not minio_client.bucket_exists(self._config.services.minio.bucket):
                minio_client.make_bucket(self._config.services.minio.bucket)
            self._reader._clear_sample_objects(minio_client)
            with connection, connection.cursor() as cursor:
                cursor.execute("TRUNCATE annotations, samples, scenes RESTART IDENTITY CASCADE")
        finally:
            connection.close()

    def append_batch(self, start_idx: int, samples: Sequence[UnifiedSample]) -> StorageWriteResult:
        minio_client, connection = self._clients()
        if not minio_client.bucket_exists(self._config.services.minio.bucket):
            minio_client.make_bucket(self._config.services.minio.bucket)
        t0 = time.perf_counter()
        bytes_written = 0
        try:
            with connection, connection.cursor() as cursor:
                for offset, sample in enumerate(samples):
                    sample_idx = start_idx + offset
                    cursor.execute(
                        """
                        INSERT INTO scenes (scene_token, scene_name, location, num_samples)
                        VALUES (%s, %s, %s, 1)
                        ON CONFLICT (scene_token)
                        DO UPDATE SET num_samples = scenes.num_samples + 1
                        """,
                        (
                            sample.scene_token,
                            sample.scene_name,
                            sample.location,
                        ),
                    )

                    blob_refs = sample.blob_refs(sample_idx)
                    for camera, path in blob_refs.camera_paths.items():
                        payload = image_to_jpeg_bytes(sample.cameras[camera])
                        bytes_written += len(payload)
                        minio_client.put_object(
                            self._config.services.minio.bucket,
                            path,
                            data=io.BytesIO(payload),
                            length=len(payload),
                            content_type="image/jpeg",
                        )

                    for sensor, path in blob_refs.sensor_paths.items():
                        data = sample.lidar_top if sensor == "LIDAR_TOP" else sample.radars[sensor]
                        payload = array_to_npy_bytes(data)
                        bytes_written += len(payload)
                        minio_client.put_object(
                            self._config.services.minio.bucket,
                            path,
                            data=io.BytesIO(payload),
                            length=len(payload),
                            content_type="application/octet-stream",
                        )

                    flat_refs = blob_refs.flattened()
                    cursor.execute(
                        """
                        INSERT INTO samples (
                            sample_idx, token, scene_token, timestamp, location,
                            ego_translation, ego_rotation, num_annotations, num_lidar_points,
                            cam_front_path, cam_front_left_path, cam_front_right_path,
                            cam_back_path, cam_back_left_path, cam_back_right_path,
                            lidar_top_path, radar_front_path, radar_front_left_path,
                            radar_front_right_path, radar_back_left_path, radar_back_right_path
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (sample_idx) DO UPDATE SET token = EXCLUDED.token
                        """,
                        (
                            sample_idx,
                            sample.token,
                            sample.scene_token,
                            sample.timestamp,
                            sample.location,
                            sample.ego_translation,
                            sample.ego_rotation,
                            len(sample.annotations),
                            int(sample.lidar_top.shape[0]),
                            flat_refs["CAM_FRONT"],
                            flat_refs["CAM_FRONT_LEFT"],
                            flat_refs["CAM_FRONT_RIGHT"],
                            flat_refs["CAM_BACK"],
                            flat_refs["CAM_BACK_LEFT"],
                            flat_refs["CAM_BACK_RIGHT"],
                            flat_refs["LIDAR_TOP"],
                            flat_refs["RADAR_FRONT"],
                            flat_refs["RADAR_FRONT_LEFT"],
                            flat_refs["RADAR_FRONT_RIGHT"],
                            flat_refs["RADAR_BACK_LEFT"],
                            flat_refs["RADAR_BACK_RIGHT"],
                        ),
                    )

                    for annotation in sample.annotations:
                        cursor.execute(
                            """
                            INSERT INTO annotations (
                                sample_idx, category, translation, size, rotation,
                                num_lidar_pts, num_radar_pts
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                sample_idx,
                                annotation.category,
                                annotation.translation,
                                annotation.size,
                                annotation.rotation,
                                annotation.num_lidar_pts,
                                annotation.num_radar_pts,
                            ),
                        )
        finally:
            connection.close()

        elapsed = time.perf_counter() - t0
        return StorageWriteResult(
            backend=self.backend_name,
            samples_written=len(samples),
            elapsed_sec=elapsed,
            bytes_written=bytes_written,
        )

    def reader(self):
        return self._reader


class RedisBatchWriter(BatchWriter):
    backend_key = "redis"
    backend_name = "Redis"

    def __init__(self, config: AppConfig) -> None:
        self._reader = RedisBackend(config.services.redis)

    def reset(self) -> None:
        client = self._reader._client()
        pipeline = client.pipeline()
        for pattern in ("sample:*", "embedding:*", "location:*", "category:*"):
            keys = list(client.scan_iter(pattern))
            if keys:
                pipeline.delete(*keys)
        pipeline.delete("samples_by_timestamp")
        pipeline.execute()

    def append_batch(self, start_idx: int, samples: Sequence[UnifiedSample]) -> StorageWriteResult:
        client = self._reader._client()
        pipeline = client.pipeline()
        t0 = time.perf_counter()
        bytes_written = 0
        for offset, sample in enumerate(samples):
            sample_idx = start_idx + offset
            sample_key = f"sample:{sample_idx:04d}"
            metadata = sample.metadata(sample_idx)
            categories = json.dumps(metadata.annotation_categories)
            mapping = {
                "token": sample.token,
                "scene_name": sample.scene_name,
                "location": sample.location,
                "num_annotations": metadata.num_annotations,
                "num_lidar_points": metadata.num_lidar_points,
                "categories": categories,
            }
            pipeline.hset(sample_key, mapping=mapping)
            bytes_written += sum(len(str(value)) for value in mapping.values())

            embedding = self._reader._derive_embedding(sample).tobytes()
            pipeline.set(f"embedding:{sample_idx:04d}", embedding)
            bytes_written += len(embedding)

            pipeline.zadd("samples_by_timestamp", {sample_key: sample.timestamp})
            pipeline.sadd(f"location:{sample.location}", sample_key)
            for annotation in sample.annotations:
                pipeline.sadd(f"category:{annotation.category}", sample_key)
        pipeline.execute()
        elapsed = time.perf_counter() - t0
        return StorageWriteResult(
            backend=self.backend_name,
            samples_written=len(samples),
            elapsed_sec=elapsed,
            bytes_written=bytes_written,
        )

    def reader(self):
        return self._reader


class LanceBatchWriter(BatchWriter):
    backend_key = "lance"
    backend_name = "Lance"

    def __init__(self, config: AppConfig) -> None:
        self._reader = LanceBackend(config.storage.lance.dataset_path)
        self._mode = "overwrite"

    def reset(self) -> None:
        self._reader.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if self._reader.dataset_path.exists():
            shutil.rmtree(self._reader.dataset_path)
        self._mode = "overwrite"

    def append_batch(self, start_idx: int, samples: Sequence[UnifiedSample]) -> StorageWriteResult:
        import lance
        import pyarrow as pa

        t0 = time.perf_counter()
        bytes_written = 0
        rows = []
        for offset, sample in enumerate(samples):
            sample_idx = start_idx + offset
            row = sample.to_dict(sample_idx)
            for camera in CAMERAS:
                payload = image_to_jpeg_bytes(sample.cameras[camera])
                row[f"{camera}_bytes"] = payload
                bytes_written += len(payload)
            for sensor in ("LIDAR_TOP", *RADARS):
                data = sample.lidar_top if sensor == "LIDAR_TOP" else sample.radars[sensor]
                payload = array_to_npy_bytes(data)
                row[f"{sensor}_bytes"] = payload
                bytes_written += len(payload)
            rows.append(row)
        if rows:
            table = pa.Table.from_pylist(rows)
            lance.write_dataset(table, str(self._reader.dataset_path), mode=self._mode)
            self._mode = "append"
        elapsed = time.perf_counter() - t0
        return StorageWriteResult(
            backend=self.backend_name,
            samples_written=len(samples),
            elapsed_sec=elapsed,
            bytes_written=bytes_written,
        )

    def reader(self):
        return self._reader


class ParquetBatchWriter(BatchWriter):
    backend_key = "parquet"
    backend_name = "Parquet"

    def __init__(self, config: AppConfig) -> None:
        self._reader = ParquetBackend(config.storage.parquet.dataset_path)
        self._part_idx = 0

    def reset(self) -> None:
        self._reader._clear_dataset()
        self._part_idx = 0

    def append_batch(self, start_idx: int, samples: Sequence[UnifiedSample]) -> StorageWriteResult:
        import pyarrow as pa
        import pyarrow.parquet as pq

        t0 = time.perf_counter()
        bytes_written = 0
        rows = []
        for offset, sample in enumerate(samples):
            sample_idx = start_idx + offset
            row = sample.to_dict(sample_idx)
            row["has_pedestrian"] = any(
                "pedestrian" in category for category in row["annotation_categories"]
            )
            for camera in CAMERAS:
                payload = image_to_jpeg_bytes(sample.cameras[camera])
                row[f"{camera}_bytes"] = payload
                bytes_written += len(payload)
            for sensor in ("LIDAR_TOP", *RADARS):
                data = sample.lidar_top if sensor == "LIDAR_TOP" else sample.radars[sensor]
                payload = array_to_npy_bytes(data)
                row[f"{sensor}_bytes"] = payload
                bytes_written += len(payload)
            rows.append(row)
        if rows:
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, self._reader.dataset_path / f"part-{self._part_idx:05d}.parquet")
            self._part_idx += 1
        elapsed = time.perf_counter() - t0
        return StorageWriteResult(
            backend=self.backend_name,
            samples_written=len(samples),
            elapsed_sec=elapsed,
            bytes_written=bytes_written,
        )

    def reader(self):
        return self._reader


class WebDatasetBatchWriter(BatchWriter):
    backend_key = "webdataset"
    backend_name = "WebDataset"

    def __init__(self, config: AppConfig) -> None:
        self._reader = WebDatasetBackend(
            shard_pattern=config.storage.webdataset.shard_pattern,
            maxcount=config.storage.webdataset.maxcount,
        )
        self._sink = None

    def reset(self) -> None:
        import webdataset as wds

        root = self._reader._root_dir()
        if self._sink is not None:
            self._sink.close()
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        self._sink = wds.ShardWriter(self._reader.shard_pattern, maxcount=self._reader.maxcount)

    def append_batch(self, start_idx: int, samples: Sequence[UnifiedSample]) -> StorageWriteResult:
        if self._sink is None:
            self.reset()

        t0 = time.perf_counter()
        bytes_written = 0
        for offset, sample in enumerate(samples):
            sample_idx = start_idx + offset
            record = {"__key__": f"sample_{sample_idx:04d}"}
            for camera in CAMERAS:
                payload = image_to_jpeg_bytes(sample.cameras[camera])
                record[f"{camera}.jpg"] = payload
                bytes_written += len(payload)
            for sensor in ("LIDAR_TOP", *RADARS):
                data = sample.lidar_top if sensor == "LIDAR_TOP" else sample.radars[sensor]
                payload = array_to_npy_bytes(data)
                record[f"{sensor}.npy"] = payload
                bytes_written += len(payload)
            metadata = sample.to_dict(sample_idx)
            record["metadata.json"] = json.dumps(metadata, sort_keys=True).encode("utf-8")
            self._sink.write(record)
        elapsed = time.perf_counter() - t0
        return StorageWriteResult(
            backend=self.backend_name,
            samples_written=len(samples),
            elapsed_sec=elapsed,
            bytes_written=bytes_written,
        )

    def finalize(self) -> None:
        if self._sink is not None:
            self._sink.close()
            self._sink = None

    def cleanup(self) -> None:
        self.finalize()
        root = self._reader._root_dir()
        if root.exists():
            shutil.rmtree(root)

    def reader(self):
        return self._reader


def run_batched_ingest_study(
    config: AppConfig,
    *,
    backends: Sequence[str],
    options: StudyOptions,
    output_root: Path,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[BackendStudySummary] = []
    for backend_key in backends:
        summary = _run_backend_study(
            config=config,
            backend_key=backend_key,
            options=options,
            output_root=output_root,
        )
        summaries.append(summary)

    summary_payload = {
        "provider": options.provider,
        "dataset_version": config.pipeline.dataset_version,
        "batch_size": options.batch_size,
        "limit": options.limit,
        "backends": [summary.as_dict() for summary in summaries],
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    _write_summary_csv(output_root / "summary.csv", summaries)
    dashboard_path = output_root / "summary.html"
    dashboard_path.write_text(
        build_study_summary_html(summary_payload),
        encoding="utf-8",
    )
    return {
        "summary": str(output_root / "summary.json"),
        "csv": str(output_root / "summary.csv"),
        "dashboard": str(dashboard_path),
    }


def build_study_summary_html(payload: dict[str, object]) -> str:
    backend_rows = payload.get("backends") or []
    rows = []
    for item in backend_rows:
        row = dict(item)
        random_access = (
            row["random_access_p50_ms"] if row["random_access_p50_ms"] is not None else "n/a"
        )
        sequential = (
            row["sequential_throughput"] if row["sequential_throughput"] is not None else "n/a"
        )
        curation = row["curation_query_ms"] if row["curation_query_ms"] is not None else "n/a"
        peak_cpu = row["peak_service_cpu"] if row["peak_service_cpu"] is not None else "n/a"
        rows.append(
            "<tr>"
            f"<td>{row['backend_name']}</td>"
            f"<td>{row['status']}</td>"
            f"<td>{row['samples']}</td>"
            f"<td>{row['scenes']}</td>"
            f"<td>{row['batch_count']}</td>"
            f"<td>{row['ingest_elapsed_sec']}</td>"
            f"<td>{row['ingest_throughput_mean']}</td>"
            f"<td>{random_access}</td>"
            f"<td>{sequential}</td>"
            f"<td>{curation}</td>"
            f"<td>{row['disk_gb']}</td>"
            f"<td>{row['peak_service_name'] or 'n/a'} ({peak_cpu}%)</td>"
            f"<td><a href=\"{Path(row['benchmark_dashboard']).name}\">benchmark</a></td>"
            f"<td><a href=\"{Path(row['telemetry_dashboard']).name}\">telemetry</a></td>"
            "</tr>"
        )
    provider = payload.get("provider", "n/a")
    version = payload.get("dataset_version", "n/a")
    batch_size = payload.get("batch_size", "n/a")
    limit = payload.get("limit", "full")
    return f"""
    <html>
      <head>
        <style>
          :root {{
            --bg: #0b0b10;
            --panel: #161622;
            --line: #544bb0;
            --ink: #e7e8f3;
            --muted: #b5b0cf;
            --accent-soft: #201f31;
            --shadow: 6px 6px 0 #211b52;
          }}
          body {{
            font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
            margin: 32px;
            color: var(--ink);
            background:
              radial-gradient(circle at top right, rgba(84, 54, 252, 0.12), transparent 22%),
              linear-gradient(180deg, #0f0f16 0%, var(--bg) 100%);
          }}
          main {{ max-width: 1440px; margin: 0 auto; }}
          .card {{
            border: 3px solid var(--line);
            border-radius: 20px;
            padding: 16px;
            background: var(--panel);
            box-shadow: 4px 4px 0 #211b52;
            margin-bottom: 18px;
          }}
          .table-wrap {{
            overflow-x: auto;
            border: 3px solid var(--line);
            border-radius: 20px;
            background: var(--panel);
            box-shadow: var(--shadow);
          }}
          table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
          }}
          th, td {{
            border: 2px solid var(--line);
            padding: 8px 10px;
            text-align: left;
            vertical-align: top;
            overflow-wrap: anywhere;
          }}
          th {{ background: #6b59dd; }}
          a {{ color: var(--ink); }}
          .muted {{ color: var(--muted); }}
        </style>
      </head>
      <body>
        <main>
          <h1>Overnight Batched Study</h1>
          <div class="card">
            <p class="muted">Provider: <strong>{provider}</strong></p>
            <p class="muted">Dataset version: <strong>{version}</strong></p>
            <p class="muted">Batch size: <strong>{batch_size}</strong></p>
            <p class="muted">Limit: <strong>{limit}</strong></p>
          </div>
          <div class="table-wrap">
            <table>
              <tr>
                <th>Backend</th><th>Status</th><th>Samples</th><th>Scenes</th><th>Batches</th>
                <th>Ingest (s)</th><th>Mean Throughput</th><th>Random p50 (ms)</th>
                <th>Sequential</th><th>Curation (ms)</th><th>Disk (GB)</th>
                <th>Peak Service CPU</th><th>Benchmark</th><th>Telemetry</th>
              </tr>
              {''.join(rows)}
            </table>
          </div>
        </main>
      </body>
    </html>
    """


def _run_backend_study(
    *,
    config: AppConfig,
    backend_key: str,
    options: StudyOptions,
    output_root: Path,
) -> BackendStudySummary:
    writer = _make_writer(config, backend_key)
    backend_dir = output_root / backend_key
    backend_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex[:12]
    recorder = TelemetryRecorder.start(
        settings=config.services.postgres,
        compose_file=Path(__file__).resolve().parents[3] / "config/docker-compose.yml",
        run_id=run_id,
        suite_name=f"nuDemo batched ingest study - {backend_key}",
        provider=options.provider,
        simulate=False,
        sample_limit=options.limit,
    )
    recorder.snapshot_services("run_start")

    results: list[BenchmarkResult] = []
    batch_elapsed: list[float] = []
    batch_throughputs: list[float] = []
    scene_tokens: set[str] = set()
    total_samples = 0
    batch_count = 0
    status = "ok"

    writer.reset()
    try:
        for start_idx, batch in iter_sample_batches(
            config=config,
            provider_name=options.provider,
            limit=options.limit,
            batch_size=options.batch_size,
        ):
            batch_count += 1
            scene_tokens.update(sample.scene_token for sample in batch)
            write_result = writer.append_batch(start_idx, batch)
            total_samples += write_result.samples_written
            batch_elapsed.append(write_result.elapsed_sec)
            batch_throughputs.append(write_result.throughput)
            result = BenchmarkResult(
                stage="storage",
                backend=writer.backend_name,
                pattern="batch_ingest",
                metrics={
                    "throughput_samples_per_sec": float(write_result.throughput),
                    "bytes_written": float(write_result.bytes_written),
                    "batch_size": float(write_result.samples_written),
                    "cumulative_samples": float(total_samples),
                },
                metadata={
                    "batch_index": batch_count,
                    "start_idx": start_idx,
                    "end_idx": start_idx + write_result.samples_written - 1,
                },
                sample_count=write_result.samples_written,
                elapsed_sec=write_result.elapsed_sec,
            )
            results.append(result)
            recorder.record_result(result)
            if (
                options.snapshot_every_batches > 0
                and batch_count % options.snapshot_every_batches == 0
            ):
                recorder.snapshot_services(f"batch_{batch_count:04d}")

        writer.finalize()
        reader = writer.reader()
        evaluation_results = _measure_reader(
            reader=reader,
            sample_count=total_samples,
            random_sample_count=options.random_sample_count,
        )
        for result in evaluation_results:
            results.append(result)
            recorder.record_result(result)
        recorder.snapshot_services("post_evaluation")
    except Exception as exc:
        status = "error"
        error_result = BenchmarkResult(
            stage="storage",
            backend=writer.backend_name,
            pattern="batch_ingest",
            metrics={},
            sample_count=0,
            elapsed_sec=0.0,
            status="error",
            error=str(exc),
        )
        results.append(error_result)
        recorder.record_result(error_result)
    finally:
        writer.finalize()

    dataset = {
        "samples": total_samples,
        "scenes": len(scene_tokens),
        "provider": options.provider,
        "run_id": run_id,
        "batch_size": options.batch_size,
    }
    report = BenchmarkReport(
        suite_name=f"nuDemo batched ingest study - {backend_key}",
        dataset=dataset,
        results=results,
    )
    report_path, json_path, csv_path = export_report_bundle(report, backend_dir)
    dashboard_path = backend_dir / "benchmark_dashboard.html"
    dashboard_path.write_text(build_dashboard_html(report), encoding="utf-8")

    summary = _result_summary(results)
    telemetry_dashboard_path = backend_dir / "telemetry_dashboard.html"
    recorder.complete(
        status="partial" if summary["error_count"] else status,
        dataset=dataset,
        summary=summary,
        report_path=report_path,
        json_path=json_path,
        csv_path=csv_path,
        dashboard_path=dashboard_path,
        telemetry_dashboard_path=telemetry_dashboard_path,
    )

    run, spans, snapshots = fetch_run_bundle(config.services.postgres, run_id=run_id)
    telemetry_dashboard_path.write_text(
        build_telemetry_dashboard_html(run, spans, snapshots),
        encoding="utf-8",
    )

    peak_service_name, peak_service_cpu = _peak_service_cpu(snapshots)
    summary_row = BackendStudySummary(
        backend_key=backend_key,
        backend_name=writer.backend_name,
        run_id=run_id,
        status=run["status"],
        samples=total_samples,
        scenes=len(scene_tokens),
        batch_count=batch_count,
        ingest_elapsed_sec=sum(batch_elapsed),
        batch_p50_sec=_percentile(batch_elapsed, 50),
        batch_p95_sec=_percentile(batch_elapsed, 95),
        ingest_throughput_mean=_safe_mean(batch_throughputs),
        random_access_p50_ms=_metric_from_results(results, "random_access", "latency_p50_ms"),
        sequential_throughput=_metric_from_results(results, "sequential_scan", "throughput_mean"),
        curation_query_ms=_metric_from_results(
            results,
            "curation_query",
            "query_time_ms_mean",
        ),
        disk_gb=(_metric_from_results(results, "disk_footprint", "disk_bytes") or 0.0)
        / (1024 * 1024 * 1024),
        peak_service_cpu=peak_service_cpu,
        peak_service_name=peak_service_name,
        benchmark_dashboard=str(dashboard_path),
        telemetry_dashboard=str(telemetry_dashboard_path),
        report_dir=str(backend_dir),
    )

    if options.purge_after_backend and options.keep_backend != backend_key:
        writer.cleanup()
    return summary_row


def _measure_reader(
    reader,
    *,
    sample_count: int,
    random_sample_count: int,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    sequential = record_to_result(
        benchmark_sequential(reader.name, reader.sequential_iter, num_runs=1)
    )
    results.append(sequential)

    indices = build_random_indices(sample_count, random_sample_count)
    if indices:
        try:
            results.append(
                record_to_result(
                    benchmark_random_access(reader.name, reader.fetch, indices, num_runs=1)
                )
            )
        except NotImplementedError:
            pass

    try:
        results.append(
            record_to_result(
                benchmark_curation_query(reader.name, reader.curation_query, num_runs=1)
            )
        )
    except NotImplementedError:
        pass

    try:
        results.append(
            record_to_result(
                benchmark_end_to_end_curation(
                    reader.name,
                    reader.curation_query,
                    reader.fetch,
                    num_runs=1,
                )
            )
        )
    except NotImplementedError:
        pass

    results.append(record_to_result(build_disk_record(reader.name, reader.disk_footprint())))
    return results


def iter_sample_batches(
    *,
    config: AppConfig,
    provider_name: str,
    limit: int | None,
    batch_size: int,
) -> Iterator[tuple[int, list[UnifiedSample]]]:
    provider = resolve_provider(config, provider_name)
    current: list[UnifiedSample] = []
    start_idx = 0
    for sample in provider.iter_samples(limit=limit):
        current.append(sample)
        if len(current) >= batch_size:
            yield start_idx, current
            start_idx += len(current)
            current = []
    if current:
        yield start_idx, current


def build_random_indices(sample_count: int, random_sample_count: int) -> list[int]:
    if sample_count <= 0 or random_sample_count <= 0:
        return []
    target = min(sample_count, random_sample_count)
    if target == sample_count:
        return list(range(sample_count))
    step = max(sample_count // target, 1)
    indices = list(range(0, sample_count, step))[:target]
    if indices[-1] != sample_count - 1:
        indices[-1] = sample_count - 1
    return sorted(set(indices))


def _make_writer(config: AppConfig, backend_key: str) -> BatchWriter:
    writers: dict[str, type[BatchWriter]] = {
        "minio-postgres": MinioPostgresBatchWriter,
        "redis": RedisBatchWriter,
        "lance": LanceBatchWriter,
        "parquet": ParquetBatchWriter,
        "webdataset": WebDatasetBatchWriter,
    }
    try:
        return writers[backend_key](config)
    except KeyError as exc:
        raise ValueError(f"unsupported backend {backend_key}") from exc


def _write_summary_csv(path: Path, summaries: Sequence[BackendStudySummary]) -> None:
    rows = [summary.as_dict() for summary in summaries]
    fieldnames = list(rows[0].keys()) if rows else ["backend_key", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_from_results(
    results: Sequence[BenchmarkResult],
    pattern: str,
    metric_name: str,
) -> float | None:
    for result in results:
        if result.pattern == pattern and metric_name in result.metrics:
            return float(result.metrics[metric_name])
    return None


def _peak_service_cpu(snapshots: Sequence[dict[str, Any]]) -> tuple[str | None, float | None]:
    peak_service = None
    peak_cpu = None
    for snapshot in snapshots:
        cpu = snapshot.get("cpu_percent")
        if cpu is None:
            continue
        value = float(cpu)
        if peak_cpu is None or value > peak_cpu:
            peak_cpu = value
            peak_service = str(snapshot.get("service", "unknown"))
    return peak_service, peak_cpu


def _result_summary(results: Sequence[BenchmarkResult]) -> dict[str, int]:
    error_count = sum(1 for result in results if result.status != "ok")
    return {
        "result_count": len(results),
        "ok_count": len(results) - error_count,
        "error_count": error_count,
    }


def _safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * (percentile / 100)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 4)
