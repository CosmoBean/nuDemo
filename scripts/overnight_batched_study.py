#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from nudemo.config import AppConfig
from nudemo.studies.batched_ingest import StudyOptions, run_batched_ingest_study


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="overnight_batched_study")
    parser.add_argument("--config", default=None)
    parser.add_argument("--provider", default="real", choices=["auto", "real", "synthetic"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scene-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-sample-count", type=int, default=256)
    parser.add_argument("--snapshot-every-batches", type=int, default=1)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["minio-postgres", "redis", "lance", "parquet", "webdataset"],
        choices=["minio-postgres", "redis", "lance", "parquet", "webdataset"],
    )
    parser.add_argument("--keep-backend", default="minio-postgres")
    parser.add_argument("--purge-after-backend", action="store_true", default=False)
    parser.add_argument("--output-root", default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = AppConfig.load(args.config)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root)
        if args.output_root
        else config.runtime.reports_root / "overnight_runs" / stamp
    )
    payload = run_batched_ingest_study(
        config,
        backends=args.backends,
        options=StudyOptions(
            provider=args.provider,
            limit=args.limit,
            scene_limit=args.scene_limit,
            batch_size=args.batch_size,
            random_sample_count=args.random_sample_count,
            snapshot_every_batches=args.snapshot_every_batches,
            purge_after_backend=args.purge_after_backend,
            keep_backend=args.keep_backend,
        ),
        output_root=output_root,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
