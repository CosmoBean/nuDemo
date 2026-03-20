from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

from nudemo.benchmarks.backends import (
    LanceBackend as SimulatedLanceBackend,
)
from nudemo.benchmarks.backends import (
    MinioPostgresBackend as SimulatedMinioPostgresBackend,
)
from nudemo.benchmarks.backends import (
    ParquetBackend as SimulatedParquetBackend,
)
from nudemo.benchmarks.backends import (
    RedisBackend as SimulatedRedisBackend,
)
from nudemo.benchmarks.backends import (
    WebDatasetBackend as SimulatedWebDatasetBackend,
)
from nudemo.benchmarks.export import export_report, export_report_bundle
from nudemo.benchmarks.models import BenchmarkResult
from nudemo.benchmarks.orchestrator import BenchmarkOrchestrator
from nudemo.benchmarks.runner import (
    BenchmarkRunner,
    build_live_report,
    record_to_result,
)
from nudemo.benchmarks.synthetic import SyntheticNuScenesDataset
from nudemo.config import AppConfig
from nudemo.domain.models import CAMERAS, RADARS
from nudemo.explorer import create_explorer_app
from nudemo.extraction.providers import resolve_provider
from nudemo.ingestion.kafka import KafkaBenchmarker, KafkaPayloadEncoder
from nudemo.rendering import render_sample_contact_sheet, render_scene_gif
from nudemo.reporting.dashboard import build_dashboard_html
from nudemo.reporting.dashboard import main as dashboard_main
from nudemo.storage.lance_store import LanceBackend
from nudemo.storage.minio_postgres import MinioPostgresBackend
from nudemo.storage.parquet_store import ParquetBackend
from nudemo.storage.redis_store import RedisBackend
from nudemo.storage.webdataset_store import WebDatasetBackend
from nudemo.telemetry.dashboard import build_telemetry_dashboard_html
from nudemo.telemetry.store import TelemetryRecorder, fetch_recent_runs, fetch_run_bundle


def _iter_samples(config: AppConfig, provider_name: str, limit: int | None):
    provider = resolve_provider(config, provider_name)
    effective_limit = limit or config.pipeline.sample_limit
    return provider.iter_samples(limit=effective_limit)


def _compose_file() -> Path:
    return Path(__file__).resolve().parents[2] / "config/docker-compose.yml"


def make_backends(config: AppConfig) -> dict[str, object]:
    return {
        "minio-postgres": MinioPostgresBackend(
            minio=config.services.minio,
            postgres=config.services.postgres,
        ),
        "redis": RedisBackend(config.services.redis),
        "lance": LanceBackend(config.storage.lance.dataset_path),
        "parquet": ParquetBackend(config.storage.parquet.dataset_path),
        "webdataset": WebDatasetBackend(
            shard_pattern=config.storage.webdataset.shard_pattern,
            maxcount=config.storage.webdataset.maxcount,
        ),
    }


def make_simulated_backends() -> dict[str, object]:
    return {
        "minio-postgres": SimulatedMinioPostgresBackend(),
        "redis": SimulatedRedisBackend(),
        "lance": SimulatedLanceBackend(),
        "parquet": SimulatedParquetBackend(),
        "webdataset": SimulatedWebDatasetBackend(),
    }


def _benchmark_extraction(
    config: AppConfig,
    provider_name: str,
    limit: int | None,
) -> tuple[BenchmarkResult, dict[str, int | float | str]]:
    t0 = time.perf_counter()
    sample_count = 0
    scene_names: set[str] = set()
    total_payload_bytes = 0
    total_annotations = 0
    missing_sensor_count = 0
    for sample in _iter_samples(config, provider_name, limit):
        sample_count += 1
        scene_names.add(sample.scene_name)
        total_annotations += len(sample.annotations)
        total_payload_bytes += sum(camera.nbytes for camera in sample.cameras.values())
        total_payload_bytes += int(sample.lidar_top.nbytes)
        total_payload_bytes += sum(radar.nbytes for radar in sample.radars.values())
        missing_sensor_count += len(set(CAMERAS) - set(sample.cameras))
        missing_sensor_count += len(set(RADARS) - set(sample.radars))
    elapsed = time.perf_counter() - t0
    metrics = {
        "throughput_samples_per_sec": sample_count / elapsed if elapsed else 0.0,
        "avg_payload_mb_per_sample": (
            (total_payload_bytes / sample_count) / (1024 * 1024) if sample_count else 0.0
        ),
        "avg_annotations_per_sample": total_annotations / sample_count if sample_count else 0.0,
        "missing_sensor_count": float(missing_sensor_count),
    }
    summary = {
        "samples": sample_count,
        "scenes": len(scene_names),
        "provider": provider_name,
    }
    return (
        BenchmarkResult(
            stage="extraction",
            backend=provider_name,
            pattern="extract_summary",
            metrics=metrics,
            sample_count=sample_count,
            elapsed_sec=elapsed,
        ),
        summary,
    )


def _benchmark_kafka(
    config: AppConfig,
    provider_name: str,
    limit: int | None,
    result_callback: Callable[[BenchmarkResult], None] | None = None,
) -> list[BenchmarkResult]:
    def avg_message_kb(payload: dict[str, float | int]) -> float:
        return float(payload["total_mb"]) * 1024 / max(float(payload["messages"]), 1.0)

    suffix = uuid4().hex[:8]
    kafka_settings = replace(
        config.services.kafka,
        raw_topic=f"{config.services.kafka.raw_topic}.{suffix}",
        refined_topic=f"{config.services.kafka.refined_topic}.{suffix}",
    )
    benchmarker = KafkaBenchmarker(
        settings=kafka_settings,
        encoder=KafkaPayloadEncoder(config.services.minio),
    )
    benchmarker.create_topics()
    results: list[BenchmarkResult] = []
    for mode, pattern_base, topic, group_prefix in [
        ("metadata-only", "kafka_metadata", kafka_settings.refined_topic, "meta"),
        ("full-payload", "kafka_full_payload", kafka_settings.raw_topic, "full"),
    ]:
        produce = benchmarker.produce_samples(
            _iter_samples(config, provider_name, limit),
            mode=mode,
        )
        produce_result = BenchmarkResult(
            stage="ingestion",
            backend="Kafka",
            pattern=f"{pattern_base}_produce",
            metrics={
                "throughput_msg_sec": float(produce["throughput_msg_sec"]),
                "throughput_mb_sec": float(produce["throughput_mb_sec"]),
                "total_mb": float(produce["total_mb"]),
                "avg_message_kb": avg_message_kb(produce),
            },
            metadata={"topic": topic, "mode": mode},
            sample_count=int(produce["messages"]),
            elapsed_sec=float(produce["elapsed_sec"]),
        )
        results.append(produce_result)
        if result_callback is not None:
            result_callback(produce_result)
        consume = benchmarker.benchmark_consumer(topic, f"{group_prefix}-{suffix}")
        consume_result = BenchmarkResult(
            stage="ingestion",
            backend="Kafka",
            pattern=f"{pattern_base}_consume",
            metrics={
                "throughput_msg_sec": float(consume["throughput_msg_sec"]),
                "throughput_mb_sec": float(consume["throughput_mb_sec"]),
                "total_mb": float(consume["total_mb"]),
                "avg_message_kb": avg_message_kb(consume),
            },
            metadata={"topic": topic, "mode": mode},
            sample_count=int(consume["messages"]),
            elapsed_sec=float(consume["elapsed_sec"]),
        )
        results.append(consume_result)
        if result_callback is not None:
            result_callback(consume_result)
    return results


def _run_live_benchmark(
    config: AppConfig,
    args: argparse.Namespace,
    recorder: TelemetryRecorder | None = None,
) -> tuple[dict[str, int | float | str], list[BenchmarkResult]]:
    selected_names = args.backends or ["minio-postgres", "redis", "lance", "parquet", "webdataset"]
    dataset_result, dataset_summary = _benchmark_extraction(config, args.provider, args.limit)
    results: list[BenchmarkResult] = [dataset_result]
    if recorder is not None:
        recorder.record_result(dataset_result)
        recorder.snapshot_services("post_extraction")
    try:
        kafka_results = _benchmark_kafka(
            config,
            args.provider,
            args.limit,
            result_callback=recorder.record_result if recorder is not None else None,
        )
        results.extend(kafka_results)
        if recorder is not None:
            recorder.snapshot_services("post_kafka")
    except Exception as exc:  # pragma: no cover - network/service failures depend on environment
        error_result = BenchmarkResult(
            stage="ingestion",
            backend="Kafka",
            pattern="kafka_benchmark",
            metrics={},
            status="error",
            error=str(exc),
        )
        results.append(error_result)
        if recorder is not None:
            recorder.record_result(error_result)
            recorder.snapshot_services("post_kafka")

    available_backends = make_backends(config)
    candidate_backends = {name: available_backends[name] for name in selected_names}
    successful_backends: dict[str, object] = {}
    for name, backend in candidate_backends.items():
        try:
            write_result = backend.write_samples(_iter_samples(config, args.provider, args.limit))
        except Exception as exc:  # pragma: no cover - depends on external services
            error_result = BenchmarkResult(
                stage="storage",
                backend=backend.name,
                pattern="write_throughput",
                metrics={},
                status="error",
                error=str(exc),
            )
            results.append(error_result)
            if recorder is not None:
                recorder.record_result(error_result)
                recorder.snapshot_services(f"post_storage_{name.replace('-', '_')}")
            continue

        successful_backends[name] = backend
        storage_result = BenchmarkResult(
            stage="storage",
            backend=backend.name,
            pattern="write_throughput",
            metrics={
                "throughput_samples_per_sec": float(write_result.throughput),
                "bytes_written": float(write_result.bytes_written),
            },
            sample_count=int(write_result.samples_written),
            elapsed_sec=float(write_result.elapsed_sec),
        )
        results.append(storage_result)
        if recorder is not None:
            recorder.record_result(storage_result)
            recorder.snapshot_services(f"post_storage_{name.replace('-', '_')}")

    runner = BenchmarkRunner(backends=successful_backends)
    random_indices = list(range(min(dataset_summary["samples"], args.random_sample_count)))
    if successful_backends:
        suite_results: list[BenchmarkResult] = []

        def capture_record(record) -> None:
            result = record_to_result(record)
            suite_results.append(result)
            if recorder is not None:
                recorder.record_result(result)

        runner.run_storage_suite(random_indices=random_indices, record_callback=capture_record)
        results.extend(suite_results)
    return dataset_summary, results


def _result_summary(results: list[BenchmarkResult]) -> dict[str, int]:
    error_count = sum(1 for result in results if result.status != "ok")
    return {
        "result_count": len(results),
        "ok_count": len(results) - error_count,
        "error_count": error_count,
    }


def _write_telemetry_dashboard(
    config: AppConfig,
    *,
    run_id: str,
) -> Path:
    run, spans, snapshots = fetch_run_bundle(config.services.postgres, run_id=run_id)
    output_path = config.runtime.reports_root / "telemetry_dashboard.html"
    output_path.write_text(
        build_telemetry_dashboard_html(run, spans, snapshots),
        encoding="utf-8",
    )
    return output_path


def command_telemetry_runs(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    try:
        runs = fetch_recent_runs(config.services.postgres, limit=args.limit)
    except Exception as exc:  # pragma: no cover - depends on external services
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    print(json.dumps(runs, indent=2, default=str))
    return 0


def command_telemetry_dashboard(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    run_id = None if args.latest else args.run_id
    try:
        run, spans, snapshots = fetch_run_bundle(config.services.postgres, run_id=run_id)
    except Exception as exc:  # pragma: no cover - depends on external services
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    output_path = config.runtime.reports_root / f"telemetry_dashboard_{run['run_id']}.html"
    output_path.write_text(
        build_telemetry_dashboard_html(run, spans, snapshots),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run["run_id"],
                "dashboard": str(output_path),
                "span_count": len(spans),
                "snapshot_count": len(snapshots),
            },
            indent=2,
            default=str,
        )
    )
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    report = {
        "dataset_root": str(config.runtime.dataset_root),
        "dataset_present": (config.runtime.dataset_root / config.pipeline.dataset_version).exists(),
        "synthetic_enabled": config.pipeline.synthetic_enabled,
        "artifacts_root": str(config.runtime.artifacts_root),
    }
    if report["dataset_present"]:
        try:
            from nuscenes.nuscenes import NuScenes

            nusc = NuScenes(
                version=config.pipeline.dataset_version,
                dataroot=str(config.runtime.dataset_root),
                verbose=False,
            )
            report["dataset_summary"] = {
                "scenes": len(nusc.scene),
                "samples": len(nusc.sample),
                "sample_data": len(nusc.sample_data),
                "annotations": len(nusc.sample_annotation),
            }
        except ModuleNotFoundError:
            report["dataset_summary"] = {"error": "nuscenes-devkit is not installed"}
    print(json.dumps(report, indent=2))
    return 0


def command_extract(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    sample_iter = _iter_samples(config, args.provider, args.limit)
    first_sample = next(sample_iter, None)
    sample_count = 0 if first_sample is None else 1 + sum(1 for _ in sample_iter)
    payload = {
        "provider": args.provider,
        "samples": sample_count,
        "first_sample": first_sample.metadata(0).as_json() if first_sample else None,
    }
    print(json.dumps(payload, indent=2))
    return 0


def command_kafka(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    benchmarker = KafkaBenchmarker(
        settings=config.services.kafka,
        encoder=KafkaPayloadEncoder(config.services.minio),
    )
    if args.create_topics:
        benchmarker.create_topics()
    result = benchmarker.produce_samples(
        _iter_samples(config, args.provider, args.limit),
        mode=args.mode,
    )
    print(json.dumps(result, indent=2))
    return 0


def command_storage(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    backend = make_backends(config)[args.backend]
    result = backend.write_samples(_iter_samples(config, args.provider, args.limit))
    payload = asdict(result)
    payload["throughput"] = result.throughput
    print(json.dumps(payload, indent=2))
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    selected_names = args.backends or ["minio-postgres", "redis", "lance", "parquet", "webdataset"]
    run_id = uuid4().hex[:12]
    recorder = TelemetryRecorder.start(
        settings=config.services.postgres,
        compose_file=_compose_file(),
        run_id=run_id,
        suite_name="nuDemo benchmark suite" if args.simulate else "nuDemo live benchmark suite",
        provider="synthetic" if args.simulate else args.provider,
        simulate=args.simulate,
        sample_limit=args.limit,
    )
    recorder.snapshot_services("run_start")

    if args.simulate:
        dataset = SyntheticNuScenesDataset(
            sample_count=args.limit or config.pipeline.sample_limit,
            scene_count=config.pipeline.synthetic_scene_count,
        ).build()
        backends = [make_simulated_backends()[name] for name in selected_names]
        report = BenchmarkOrchestrator(
            dataset,
            backends,
            suite_name="nuDemo benchmark suite",
        ).run(
            num_runs=args.num_runs,
            random_sample_count=args.random_sample_count,
            batch_size=args.batch_size,
            num_workers=tuple(args.num_workers),
        )
        report.dataset["run_id"] = run_id
        report_path = export_report(report, config.runtime.reports_root / "benchmark_report.json")
        dashboard_path = config.runtime.reports_root / "benchmark_dashboard.html"
        dashboard_path.write_text(build_dashboard_html(report), encoding="utf-8")
        for result in report.results:
            recorder.record_result(result)
        telemetry_dashboard_path = None
        telemetry_summary = _result_summary(report.results)
        if recorder.enabled:
            recorder.complete(
                status="partial" if telemetry_summary["error_count"] else "ok",
                dataset=report.dataset,
                summary=telemetry_summary,
                report_path=report_path,
                dashboard_path=dashboard_path,
            )
            try:
                telemetry_dashboard_path = _write_telemetry_dashboard(config, run_id=run_id)
            except Exception as exc:  # pragma: no cover - depends on external services
                recorder.errors.append(str(exc))
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "dashboard": str(dashboard_path),
                    "telemetry_dashboard": (
                        str(telemetry_dashboard_path) if telemetry_dashboard_path else None
                    ),
                    "telemetry_run_id": run_id,
                    "results": len(report.results),
                },
                indent=2,
            )
        )
        return 0

    dataset_summary, results = _run_live_benchmark(config, args, recorder=recorder)
    dataset_summary["run_id"] = run_id
    report = build_live_report(
        suite_name="nuDemo live benchmark suite",
        dataset=dataset_summary,
        results=results,
    )
    report_path, flat_json_path, csv_path = export_report_bundle(
        report,
        config.runtime.reports_root,
    )
    dashboard_path = config.runtime.reports_root / "benchmark_dashboard.html"
    dashboard_path.write_text(build_dashboard_html(report), encoding="utf-8")
    telemetry_dashboard_path = None
    telemetry_summary = _result_summary(results)
    if recorder.enabled:
        recorder.snapshot_services("run_end")
        telemetry_dashboard_path = config.runtime.reports_root / "telemetry_dashboard.html"
        recorder.complete(
            status="partial" if telemetry_summary["error_count"] else "ok",
            dataset=dataset_summary,
            summary=telemetry_summary,
            report_path=report_path,
            json_path=flat_json_path,
            csv_path=csv_path,
            dashboard_path=dashboard_path,
            telemetry_dashboard_path=telemetry_dashboard_path,
        )
        try:
            telemetry_dashboard_path = _write_telemetry_dashboard(config, run_id=run_id)
        except Exception as exc:  # pragma: no cover - depends on external services
            recorder.errors.append(str(exc))
            telemetry_dashboard_path = None
    print(
        json.dumps(
            {
                "report": str(report_path),
                "json": str(flat_json_path),
                "csv": str(csv_path),
                "dashboard": str(dashboard_path),
                "telemetry_dashboard": (
                    str(telemetry_dashboard_path) if telemetry_dashboard_path else None
                ),
                "telemetry_run_id": run_id,
                "records": len(results),
            },
            indent=2,
        )
    )
    return 0


def command_dashboard(args: argparse.Namespace) -> int:
    results_path = Path(args.results_path)
    return dashboard_main(results_path)


def command_explore(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    app = create_explorer_app(config, result_limit=args.limit)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


def command_render_sample(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    output_path = Path(args.output) if args.output else None
    artifact = render_sample_contact_sheet(
        config,
        sample_idx=args.sample_idx,
        provider_name=args.provider,
        output_path=output_path,
    )
    print(
        json.dumps(
            {
                "artifact_type": artifact.artifact_type,
                "output": str(artifact.output_path),
                "metadata": artifact.metadata,
            },
            indent=2,
        )
    )
    return 0


def command_render_scene(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    output_path = Path(args.output) if args.output else None
    artifact = render_scene_gif(
        config,
        scene_name=args.scene_name,
        camera=args.camera,
        max_frames=args.max_frames,
        step=args.step,
        fps=args.fps,
        output_path=output_path,
    )
    print(
        json.dumps(
            {
                "artifact_type": artifact.artifact_type,
                "output": str(artifact.output_path),
                "metadata": artifact.metadata,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nudemo")
    parser.add_argument("--config", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor")
    doctor.set_defaults(func=command_doctor)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--provider", default="real", choices=["auto", "real", "synthetic"])
    extract.add_argument("--limit", type=int, default=None)
    extract.set_defaults(func=command_extract)

    kafka = subparsers.add_parser("kafka")
    kafka.add_argument("--provider", default="real", choices=["auto", "real", "synthetic"])
    kafka.add_argument("--limit", type=int, default=None)
    kafka.add_argument("--mode", default="metadata-only", choices=["metadata-only", "full-payload"])
    kafka.add_argument("--create-topics", action="store_true")
    kafka.set_defaults(func=command_kafka)

    storage = subparsers.add_parser("storage")
    storage.add_argument(
        "backend",
        choices=["minio-postgres", "redis", "lance", "parquet", "webdataset"],
    )
    storage.add_argument("--provider", default="real", choices=["auto", "real", "synthetic"])
    storage.add_argument("--limit", type=int, default=None)
    storage.set_defaults(func=command_storage)

    benchmark = subparsers.add_parser("benchmark")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)

    run = benchmark_sub.add_parser("run")
    run.add_argument("--provider", default="real", choices=["auto", "real", "synthetic"])
    run.add_argument("--limit", type=int, default=None)
    run.add_argument(
        "--simulate",
        action="store_true",
        help="Run the in-memory synthetic benchmark suite instead of live service backends.",
    )
    run.add_argument(
        "--backends",
        nargs="*",
        choices=["minio-postgres", "redis", "lance", "parquet", "webdataset"],
        default=None,
    )
    run.add_argument("--num-runs", type=int, default=1)
    run.add_argument("--random-sample-count", type=int, default=10)
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--num-workers", nargs="*", type=int, default=[0, 2, 4])
    run.set_defaults(func=command_benchmark)

    dashboard = benchmark_sub.add_parser("dashboard")
    dashboard.add_argument("--results-path", required=True)
    dashboard.set_defaults(func=command_dashboard)

    explore = subparsers.add_parser("explore")
    explore.add_argument("--host", default="127.0.0.1")
    explore.add_argument("--port", type=int, default=8788)
    explore.add_argument("--limit", type=int, default=200)
    explore.add_argument("--debug", action="store_true")
    explore.set_defaults(func=command_explore)

    render = subparsers.add_parser("render")
    render_sub = render.add_subparsers(dest="render_command", required=True)

    render_sample = render_sub.add_parser("sample")
    render_sample.add_argument("--provider", default="real", choices=["auto", "real", "synthetic"])
    render_sample.add_argument("--sample-idx", type=int, default=0)
    render_sample.add_argument("--output", default=None)
    render_sample.set_defaults(func=command_render_sample)

    render_scene = render_sub.add_parser("scene")
    render_scene.add_argument("--scene-name", default=None)
    render_scene.add_argument("--camera", default="CAM_FRONT", choices=list(CAMERAS))
    render_scene.add_argument("--max-frames", type=int, default=24)
    render_scene.add_argument("--step", type=int, default=1)
    render_scene.add_argument("--fps", type=int, default=2)
    render_scene.add_argument("--output", default=None)
    render_scene.set_defaults(func=command_render_scene)

    telemetry = subparsers.add_parser("telemetry")
    telemetry_sub = telemetry.add_subparsers(dest="telemetry_command", required=True)

    telemetry_runs = telemetry_sub.add_parser("runs")
    telemetry_runs.add_argument("--limit", type=int, default=10)
    telemetry_runs.set_defaults(func=command_telemetry_runs)

    telemetry_dashboard = telemetry_sub.add_parser("dashboard")
    telemetry_dashboard.add_argument("--run-id", default=None)
    telemetry_dashboard.add_argument("--latest", action="store_true")
    telemetry_dashboard.set_defaults(func=command_telemetry_dashboard)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)
