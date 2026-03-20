# nuDemo

`nuDemo` is a local execution scaffold for a nuScenes data engineering and benchmarking pipeline:

- Real-path code for nuScenes extraction, Kafka ingestion, and MinIO/PostgreSQL, Redis, Lance, and WebDataset storage backends.
- A synthetic benchmark suite that runs without Docker or the real dataset and produces report artifacts immediately.
- Reporting hooks that export JSON/CSV-style benchmark data and render a lightweight dashboard HTML summary.
- Persisted telemetry for benchmark runs, including stage spans and Docker service snapshots stored in PostgreSQL when the Docker-backed stack is available.

## Quickstart

Bootstrap a pinned Python 3.12 environment:

```bash
./scripts/bootstrap_env.sh
```

Check the runtime and dataset visibility:

```bash
.venv/bin/nudemo doctor
```

For real-data runs, place the official `nuScenes v1.0-mini` dataset under `data/nuscenes` so that
`data/nuscenes/v1.0-mini` exists, or set `NUDEMO_DATASET_ROOT` to a different extracted dataset root.

Run the service-free synthetic benchmark suite:

```bash
.venv/bin/nudemo benchmark run --simulate --provider synthetic --limit 24
```

This writes:

- `artifacts/reports/benchmark_report.json`
- `artifacts/reports/benchmark_dashboard.html`

If PostgreSQL is available, synthetic runs also persist telemetry history and can emit
`artifacts/reports/telemetry_dashboard.html`.

Run the live benchmark suite against the Docker-backed services:

```bash
make deps
make benchmark-real PROVIDER=real LIMIT=16 BACKENDS="minio-postgres redis lance webdataset"
```

`benchmark-real` reloads the selected backends with exactly the requested `LIMIT`, then benchmarks
extraction, Kafka ingest/consume, storage writes, sequential scans, random access, curation queries,
and disk footprint.

This writes:

- `artifacts/reports/benchmark_report.json`
- `artifacts/reports/benchmark_results.json`
- `artifacts/reports/benchmark_results.csv`
- `artifacts/reports/benchmark_dashboard.html`
- `artifacts/reports/telemetry_dashboard.html`

Inspect persisted telemetry history:

```bash
make telemetry-runs
make telemetry-dashboard
make reports-index
make serve-reports
make data-explorer
```

`make telemetry-dashboard` renders the latest run. To render a specific run, pass its
`telemetry_run_id` back to the CLI or Make:

```bash
make telemetry-dashboard RUN_ID=<run_id>
```

`make reports-index` renders `artifacts/reports/index.html`, and `make serve-reports` serves
`artifacts/reports/` over `http://127.0.0.1:8787/` by default so the generated HTML, JSON, and CSV
artifacts can be viewed from a browser or fronted by a reverse proxy or tunnel.

The same browser entry point now also serves a searchable ingested-data explorer at `/explorer`.
It reads sample metadata from PostgreSQL and proxies camera images from MinIO, so it requires the
`minio-postgres` backend to be loaded with real or synthetic samples first:

```bash
make storage-minio-postgres PROVIDER=real LIMIT=64
make data-explorer
```

From the running browser host you can then use:

- `/`
- `/explorer`
- `/benchmark_dashboard.html`
- `/telemetry_dashboard.html`

`make data-explorer` starts a live browser app on `http://127.0.0.1:8788/` for the ingested
records. It supports text search over token, scene, location, and annotation category;
location/category filters; minimum-annotation filtering; summary cards; and `CAM_*` previews
sourced from MinIO when the camera blobs are available.

## Visual Inspection

nuScenes is scene-based. A scene is a temporal driving sequence, while each sample is one
synchronized sensor snapshot across the six cameras, LiDAR, and radars. This repo now exports both:

```bash
make render-sample PROVIDER=real SAMPLE_IDX=0
make render-scene SCENE_NAME=scene-0061 CAMERA=CAM_FRONT MAX_FRAMES=24 FPS=2
make reports-index
```

These commands write visual artifacts under `artifacts/reports/renders/`, which are also served by
the report browser host. `render-sample` creates a camera contact sheet for one sample. `render-scene`
creates an animated GIF for one camera channel across a scene so you can inspect temporal continuity.

## Real Pipeline Paths

The real integration points are implemented under `src/nudemo/`:

- `extraction/` loads either nuScenes data or a deterministic synthetic fallback.
- `ingestion/` encodes metadata-only and full-payload Kafka messages and exposes topic/bootstrap helpers.
- `storage/` contains live backends for MinIO+PostgreSQL, Redis, Lance, and WebDataset.
- `benchmarks/` contains an in-memory orchestration layer for fast local validation and CI.
- `rendering.py` exports sample contact sheets and scene GIFs into browser-visible artifacts.
- `telemetry/` persists run history, stage spans, and service snapshots into PostgreSQL and renders a bottleneck dashboard.

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

- Verified locally: real nuScenes-mini extraction, Kafka ingestion, MinIO/PostgreSQL, Redis, Lance, and WebDataset benchmark runs, JSON/CSV export, benchmark dashboard HTML generation, telemetry ingestion into PostgreSQL, and telemetry dashboard HTML generation.
- Recent benchmark runs can be queried with `nudemo telemetry runs` and re-rendered with `nudemo telemetry dashboard`.
- Telemetry is additive. If PostgreSQL or Docker snapshots are unavailable, the benchmark still writes the benchmark report and dashboard artifacts.
- Target runtime: Python `3.12`; the repo bootstraps that version explicitly because the external stack is not yet a clean Python 3.13 target.
