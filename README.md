# nuDemo

`nuDemo` is a local execution scaffold for the `project_spec_v2.md` nuScenes pipeline:

- Real-path code for nuScenes extraction, Kafka ingestion, and MinIO/PostgreSQL, Redis, Lance, and WebDataset storage backends.
- A synthetic benchmark suite that runs without Docker or the real dataset and produces report artifacts immediately.
- Reporting hooks that export JSON/CSV-style benchmark data and render a lightweight dashboard HTML summary.

## Quickstart

Bootstrap a pinned Python 3.12 environment:

```bash
./scripts/bootstrap_env.sh
```

Check the runtime and dataset visibility:

```bash
.venv/bin/nudemo doctor
```

Run the service-free synthetic benchmark suite:

```bash
.venv/bin/nudemo benchmark run --simulate --provider synthetic --limit 24
```

This writes:

- `artifacts/reports/benchmark_report.json`
- `artifacts/reports/benchmark_dashboard.html`

## Real Pipeline Paths

The real integration points are implemented under `src/nudemo/`:

- `extraction/` loads either nuScenes-mini or a deterministic synthetic fallback.
- `ingestion/` encodes metadata-only and full-payload Kafka messages and exposes topic/bootstrap helpers.
- `storage/` contains live backends for MinIO+PostgreSQL, Redis, Lance, and WebDataset.
- `benchmarks/` contains an in-memory orchestration layer for fast local validation and CI.

Local infrastructure definitions live in `config/docker-compose.yml` and `config/init.sql`.

## Recommended Architecture

The scaffold follows the architecture in the spec:

- Ingestion: Kafka as the replayable event bus.
- Blob storage: MinIO or S3-compatible storage for camera, LiDAR, and radar payloads.
- Queryable metadata: PostgreSQL for curation filters and annotation joins.
- Hot-path retrieval: Redis for embeddings and lightweight sample metadata.
- Random-access dataset: Lance for evaluation and curation workflows.
- Sequential training export: WebDataset shards for high-throughput loading.

## Current Status

- Verified locally: synthetic benchmark generation, JSON export, dashboard HTML generation, typed data models, and CLI flows.
- Implemented but not exercised on this machine: Docker-backed services and live nuScenes ingestion, because `docker` is not installed here and the real dataset is not present.
- Target runtime: Python `3.12`; the repo bootstraps that version explicitly because the external stack is not yet a clean Python 3.13 target.
