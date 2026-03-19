from __future__ import annotations

import argparse
import json
from pathlib import Path

from nudemo.benchmarks.backends import (
    LanceBackend as SimulatedLanceBackend,
)
from nudemo.benchmarks.backends import (
    MinioPostgresBackend as SimulatedMinioPostgresBackend,
)
from nudemo.benchmarks.backends import (
    RedisBackend as SimulatedRedisBackend,
)
from nudemo.benchmarks.backends import (
    WebDatasetBackend as SimulatedWebDatasetBackend,
)
from nudemo.benchmarks.export import export_report
from nudemo.benchmarks.orchestrator import BenchmarkOrchestrator
from nudemo.benchmarks.runner import BenchmarkRunner, build_write_record, export_records
from nudemo.benchmarks.synthetic import SyntheticNuScenesDataset
from nudemo.config import AppConfig
from nudemo.extraction.providers import resolve_provider
from nudemo.ingestion.kafka import KafkaBenchmarker, KafkaPayloadEncoder
from nudemo.reporting.dashboard import build_dashboard_html
from nudemo.reporting.dashboard import main as dashboard_main
from nudemo.storage.lance_store import LanceBackend
from nudemo.storage.minio_postgres import MinioPostgresBackend
from nudemo.storage.redis_store import RedisBackend
from nudemo.storage.webdataset_store import WebDatasetBackend


def _iter_samples(config: AppConfig, provider_name: str, limit: int | None):
    provider = resolve_provider(config, provider_name)
    effective_limit = limit or config.pipeline.sample_limit
    return provider.iter_samples(limit=effective_limit)


def make_backends(config: AppConfig) -> dict[str, object]:
    return {
        "minio-postgres": MinioPostgresBackend(
            minio=config.services.minio,
            postgres=config.services.postgres,
        ),
        "redis": RedisBackend(config.services.redis),
        "lance": LanceBackend(config.storage.lance.dataset_path),
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
        "webdataset": SimulatedWebDatasetBackend(),
    }


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
    print(json.dumps(result.__dict__, indent=2))
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    selected_names = args.backends or ["minio-postgres", "redis", "lance", "webdataset"]

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
        report_path = export_report(report, config.runtime.reports_root / "benchmark_report.json")
        dashboard_path = config.runtime.reports_root / "benchmark_dashboard.html"
        dashboard_path.write_text(build_dashboard_html(report), encoding="utf-8")
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "dashboard": str(dashboard_path),
                    "results": len(report.results),
                },
                indent=2,
            )
        )
        return 0

    selected_backends = {name: make_backends(config)[name] for name in selected_names}

    records = []
    for backend in selected_backends.values():
        write_result = backend.write_samples(_iter_samples(config, args.provider, args.limit))
        records.append(build_write_record(write_result))

    runner = BenchmarkRunner(backends=selected_backends)
    records.extend(runner.run_storage_suite())
    json_path, csv_path = export_records(records, config.runtime.reports_root)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "records": len(records),
            },
            indent=2,
        )
    )
    return 0


def command_dashboard(args: argparse.Namespace) -> int:
    results_path = Path(args.results_path)
    return dashboard_main(results_path)


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
    storage.add_argument("backend", choices=["minio-postgres", "redis", "lance", "webdataset"])
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
        choices=["minio-postgres", "redis", "lance", "webdataset"],
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
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)
