from __future__ import annotations

import argparse
import json
from pathlib import Path

from nudemo.benchmarks.runner import BenchmarkRunner, build_write_record, export_records
from nudemo.config import AppConfig
from nudemo.extraction.providers import resolve_provider
from nudemo.ingestion.kafka import KafkaBenchmarker, KafkaPayloadEncoder
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
    selected_backends = make_backends(config)
    if args.backends:
        selected_backends = {name: selected_backends[name] for name in args.backends}

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
        "--backends",
        nargs="*",
        choices=["minio-postgres", "redis", "lance", "webdataset"],
        default=None,
    )
    run.set_defaults(func=command_benchmark)

    dashboard = benchmark_sub.add_parser("dashboard")
    dashboard.add_argument("--results-path", required=True)
    dashboard.set_defaults(func=command_dashboard)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)
