# nuDemo

`nuDemo` is a local execution scaffold for a nuScenes data engineering and benchmarking pipeline:

- Real-path code for nuScenes extraction, Kafka ingestion, and MinIO/PostgreSQL, Redis, Lance, Parquet, and WebDataset storage backends.
- A synthetic benchmark suite that runs without Docker or the real dataset and produces report artifacts immediately.
- Reporting hooks that export JSON/CSV-style benchmark data and render a lightweight dashboard HTML summary.
- Persisted telemetry for benchmark runs, including stage spans and periodic Docker service snapshots stored in PostgreSQL when the Docker-backed stack is available.
- A browser UI for scene search, processed camera previews, LiDAR visualization, storage-format comparison, and Prometheus/Grafana observability views.

## Quickstart

Bootstrap a pinned Python 3.12 environment:

```bash
./scripts/bootstrap_env.sh
```

Check the runtime and dataset visibility:

```bash
.venv/bin/nudemo doctor
```

For real-data runs, place an extracted nuScenes dataset under `data/nuscenes` so that
`data/nuscenes/<dataset-version>` exists, or set `NUDEMO_DATASET_ROOT` to a different extracted dataset root.
The repo still defaults to `v1.0-mini`, but you can target full trainval with
`DATASET_VERSION=v1.0-trainval`.

To pull the official full trainval keyframe release from the Motional nuScenes AWS Open Data mirror:

```bash
make download-trainval
```

This downloads and extracts `v1.0-trainval_meta.tgz`, the ten `v1.0-trainval##_keyframes.tgz`
archives, and `nuScenes-map-expansion-v1.3.zip` into `data/nuscenes`, while maintaining resumable
checkpoints under `artifacts/checkpoints/nuscenes-trainval/`. The default profile is the full
trainval keyframe sample set, which is about `42.6 GiB` compressed and matches the current
pipeline's sample-level extraction path. If you later want the much larger sweep/blob archives,
run `make download-trainval-full`.

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
make benchmark-real DATASET_VERSION=v1.0-trainval PROVIDER=real LIMIT=16 BACKENDS="minio-postgres redis lance parquet webdataset"
```

`benchmark-real` reloads the selected backends with exactly the requested `LIMIT`, then benchmarks
extraction, Kafka ingest/consume, storage writes, sequential scans, random access, curation queries,
and disk footprint. Live runs now persist total elapsed timing for the fetch-side stages as well, so
the telemetry/Prometheus views can compare ingestion cost against scan and fetch latency more directly.

This writes:

- `artifacts/reports/benchmark_report.json`
- `artifacts/reports/benchmark_results.json`
- `artifacts/reports/benchmark_results.csv`
- `artifacts/reports/benchmark_dashboard.html`
- `artifacts/reports/telemetry_dashboard.html`

For long-running real-data studies, prefer the batched overnight runner instead of the one-shot
all-backend benchmark. It streams the selected dataset through one backend at a time in fixed-size
batches, persists per-batch telemetry into PostgreSQL, renders a per-backend benchmark dashboard,
renders a per-backend telemetry dashboard, and writes an overall summary bundle. This is the safer
path for `v1.0-trainval` because it avoids holding a full byte-for-byte copy of every backend on disk
at the same time.

```bash
make overnight-study DATASET_VERSION=v1.0-trainval STUDY_BATCH_SIZE=32 BACKENDS="minio-postgres redis lance parquet webdataset"
```

By default the overnight runner keeps the final `minio-postgres` load in place for the explorer and
purges the other backends after they are benchmarked. It writes timestamped artifacts under
`artifacts/reports/overnight_runs/<stamp>/` plus a matching log under `artifacts/logs/`.
If you need a smaller, scene-bounded study window, pass `SCENE_LIMIT=<n>`. For example, `SCENE_LIMIT=200`
targets the first `200` real scenes instead of all `850`.

## Runtime Expectations

On this machine, the full `v1.0-trainval` `minio-postgres` overnight study has already completed
successfully over the real dataset:

- Dataset size: `850` scenes, `34,149` samples, `1,166,187` annotations.
- Batch size: `32` samples.
- Ingest wall-clock: `4,197.6s` (`69m 58s`).
- End-to-end study wall-clock: `8,800.1s` (`2h 26m 40s`).
- Mean ingest throughput: `8.15 samples/sec`.

That measured run is the best anchor for planning. If you run the same full study again for just
`minio-postgres`, budget about `2.5 hours`. If you run all five backends sequentially with
`overnight-study`, budget an actual overnight window, roughly `8-12 hours` on this host. That
full-sweep estimate is a planning number, not a completed five-backend trainval measurement yet.
If you need results in roughly `2-3 hours`, use `SCENE_LIMIT=200` and consider setting
`SNAPSHOT_EVERY_BATCHES=8` so you still capture service pressure without paying the cost of a Docker
snapshot after every single batch.

The explorer reads the `minio-postgres` backend only. If that backend has been loaded with the full
trainval set, the explorer and scene studio can show all `850` scenes immediately without waiting
for Redis, Lance, Parquet, or WebDataset to finish.

Inspect persisted telemetry history:

```bash
make telemetry-runs
make telemetry-dashboard
make serve-reports
make data-explorer
```

`make telemetry-dashboard` renders the latest run. To render a specific run, pass its
`telemetry_run_id` back to the CLI or Make:

```bash
make telemetry-dashboard RUN_ID=<run_id>
```

`make serve-reports` serves the browser app plus the generated report artifacts over
`http://127.0.0.1:8787/` by default. The browser host also renders `/benchmark_dashboard.html`
from the latest completed report for each backend, so large MinIO+PostgreSQL runs and newer
single-backend benchmark runs stay visible without manually rewriting the root artifact bundle.

The same browser entry point now also serves a searchable ingested-data explorer at `/explorer`.
It reads sample metadata from PostgreSQL and proxies camera images from MinIO, so it requires the
`minio-postgres` backend to be loaded with real or synthetic samples first:

```bash
make storage-minio-postgres DATASET_VERSION=v1.0-trainval PROVIDER=real LIMIT=512
make data-explorer
```

From the running browser host you can then use:

- `/`
- `/explorer`
- `/scene-studio`
- `/benchmark_dashboard.html`
- `/grafana-dashboard`

`make data-explorer` starts a live browser app on `http://127.0.0.1:8788/` for the ingested
records. It supports text search over token, scene, location, and annotation category;
dedicated scene/location/category filters; minimum-annotation filtering; summary cards; `CAM_*`
previews sourced from MinIO when the camera blobs are available; on-demand processed camera
comparisons; an inline LiDAR top-down preview rendered from the stored `.npy` payloads; a
storage-format comparison panel that highlights Parquet, Redis, Lance, WebDataset, and
MinIO+PostgreSQL metrics from the latest completed run for each backend, with inline comparison
graphs; and a dedicated `/scene-studio` surface with scene scrubbing plus browser-side 3D LiDAR
rendering from the stored `LIDAR_TOP` payloads.

`make deps` also starts Prometheus and Grafana:

- Grafana dashboard: `http://127.0.0.1:3000/grafana/d/nudemo-observability/nudemo-observability`
- Grafana root: `http://127.0.0.1:3000/grafana/`

The report browser exports OpenTelemetry-backed Prometheus metrics from the latest persisted
benchmark run on port `9464` by default, and Prometheus scrapes that endpoint so Grafana can show
run elapsed time, bottleneck spans, service peaks, and browser request metrics.

## Visual Inspection

nuScenes is scene-based. A scene is a temporal driving sequence, while each sample is one
synchronized sensor snapshot across the six cameras, LiDAR, and radars. This repo now exports both:

```bash
make render-sample DATASET_VERSION=v1.0-trainval PROVIDER=real SAMPLE_IDX=0
make render-scene DATASET_VERSION=v1.0-trainval SCENE_NAME=scene-0061 CAMERA=CAM_FRONT MAX_FRAMES=24 FPS=2
make reports-index
```

These commands write visual artifacts under `artifacts/reports/renders/`, which are also served by
the report browser host. `render-sample` creates a camera contact sheet for one sample. `render-scene`
creates an animated GIF for one camera channel across a scene so you can inspect temporal continuity.

## Real Pipeline Paths

The real integration points are implemented under `src/nudemo/`:

- `extraction/` loads either nuScenes data or a deterministic synthetic fallback.
- `ingestion/` encodes metadata-only and full-payload Kafka messages and exposes topic/bootstrap helpers.
- `storage/` contains live backends for MinIO+PostgreSQL, Redis, Lance, Parquet, and WebDataset.
- `benchmarks/` contains an in-memory orchestration layer for fast local validation and CI.
- `rendering.py` exports sample contact sheets and scene GIFs into browser-visible artifacts.
- `telemetry/` persists run history, stage spans, and service snapshots into PostgreSQL and renders a bottleneck dashboard.
- `observability/` exports the latest persisted telemetry as Prometheus metrics for Grafana.

Local infrastructure definitions live in `config/docker-compose.yml` and `config/init.sql`.
Prometheus and Grafana provisioning live under `config/prometheus/` and `config/grafana/`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    DATA SOURCE                                              │
│                                                                                             │
│   nuScenes v1.0-mini (404 samples, 10 scenes)   /   v1.0-trainval (34,149 samples)         │
│   12 sensors per sample: CAM_FRONT/BACK/FRONT_LEFT/FRONT_RIGHT/BACK_LEFT/BACK_RIGHT,       │
│   LIDAR_TOP, RADAR_FRONT/FRONT_LEFT/FRONT_RIGHT/BACK_LEFT/BACK_RIGHT                      │
└──────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              EXTRACTION LAYER  (src/nudemo/extraction/)                     │
│                                                                                             │
│   NuScenesProvider ──► UnifiedSample records in memory                                      │
│   SyntheticProvider  (no dataset required — deterministic fallback for CI / smoke tests)    │
└──────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                       │  UnifiedSample stream
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              INGESTION LAYER  (src/nudemo/ingestion/)                       │
│                                                                                             │
│   Kafka broker (localhost:9092)                                                             │
│   ├── drivelogs.raw      ← full-payload messages (camera JPEGs + LiDAR/radar .npy)         │
│   └── drivelogs.refined  ← metadata-only messages (~4 KB each)                             │
│                                                                                             │
│   Measured: 20.4 msg/sec produce  │  3.4 msg/sec consume (metadata mode, 16-sample run)    │
└──────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                       │  32-sample study batches
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              STORAGE LAYER  (src/nudemo/storage/)                           │
│                                                                                             │
│  ┌──────────────────────────┐  ┌────────────────┐  ┌──────────┐  ┌─────────┐  ┌─────────┐ │
│  │   MinIO + PostgreSQL     │  │     Redis       │  │  Lance   │  │ Parquet │  │WebData  │ │
│  │  (canonical blob+meta)   │  │  (hot-path /   │  │ (Arrow   │  │(columnar│  │  set    │ │
│  │                          │  │   embeddings)   │  │ random   │  │ scans)  │  │  tars)  │ │
│  │  MinIO: 6 cam JPEGs,     │  │                 │  │ access)  │  │         │  │         │ │
│  │   1 LIDAR .npy,          │  │  metadata hash  │  │          │  │         │  │sequential│ │
│  │   5 radar .npy per sample│  │  + 512-d embed  │  │ CAM_FRONT│  │CAM_FRONT│  │ only    │ │
│  │  PostgreSQL: scenes,     │  │  per sample     │  │ LIDAR_TOP│  │LIDAR_TOP│  │         │ │
│  │   samples, annotations   │  │                 │  │  columnar│  │ columnar│  │         │ │
│  │                          │  │  [localhost      │  │          │  │         │  │         │ │
│  │  8.15 samp/sec ingest    │  │   only + auth]  │  │36.9 s/s  │  │35.8 s/s │  │36.5 s/s │ │
│  │  263.6 samp/sec scan     │  │  183 samp/sec   │  │779 s/s   │  │605 s/s  │  │635 s/s  │ │
│  │  12.6 ms rand p50        │  │  0.76 ms p50    │  │0.85ms p50│  │19.6ms p5│  │n/a      │ │
│  │  68.6 GB (full trainval) │  │  0.002 GB       │  │0.20 GB   │  │0.19 GB  │  │0.20 GB  │ │
│  └──────────┬───────────────┘  └────────────────┘  └──────────┘  └─────────┘  └─────────┘ │
└─────────────┼───────────────────────────────────────────────────────────────────────────────┘
              │                         │  all backends
              │                         ▼
              │  ┌─────────────────────────────────────────────────────────────────────────┐
              │  │                BENCHMARK & STUDY LAYER  (src/nudemo/benchmarks/, studies/)│
              │  │                                                                          │
              │  │  BenchmarkOrchestrator ── per-backend:                                  │
              │  │    sequential_scan · random_access · curation_query · disk_footprint     │
              │  │  OvernightBatchedStudy ── streams dataset in 32-sample batches,          │
              │  │    one backend at a time, purges after benchmarking                      │
              │  └─────────────────────────┬───────────────────────────────────────────────┘
              │                            │
              ▼                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                         TELEMETRY & OBSERVABILITY                                          │
│                                                                                            │
│  TelemetryRecorder (src/nudemo/telemetry/)                                                 │
│  └── PostgreSQL: run records · stage spans · Docker service snapshots (CPU/mem/net/IO)     │
│                                                                                            │
│  PrometheusExporter (src/nudemo/observability/)  :9464                                     │
│  └── republishes latest run/span/service values from PostgreSQL                            │
│       nudemo_latest_run_metric_value                                                       │
│       nudemo_latest_span_metric_value                                                      │
│       nudemo_latest_service_metric_value                                                   │
│                                                                                            │
│  Prometheus :9090  ──scrapes──►  Grafana :3000/grafana/                                    │
└─────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                         BROWSER SERVING LAYER  (src/nudemo/explorer/)                      │
│                                                                                            │
│  Report server  :8787                         Explorer  :8788                              │
│  ├── /                 dashboard index        ├── /explorer      scene search + cam preview │
│  ├── /benchmark_dashboard.html                ├── /scene-studio  3D LiDAR scrubber         │
│  ├── /telemetry_dashboard.html                └── (reads PostgreSQL + proxies MinIO blobs) │
│  └── /grafana-dashboard  iframe proxy                                                      │
│                                                                                            │
│  Rendering (src/nudemo/rendering.py)                                                       │
│  └── artifacts/reports/renders/  camera contact sheets · scene GIFs                       │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Backend quick-reference

| Backend | Best fit | Ingest | Seq scan | Rand p50 | Curation | Disk (trainval) |
|---|---|---|---|---|---|---|
| MinIO + PostgreSQL | Explorer, mixed retrieval | 8.2 s/s | 263.6 s/s | 12.6 ms | 227.9 ms | 68.6 GB |
| Redis | Hot embeddings, metadata cache | 183 s/s | 3,796 s/s | 0.76 ms | 3.6 ms | ~0.002 GB |
| Lance | Random-access eval, curation | 36.9 s/s | 779 s/s | 0.85 ms | 13.7 ms | ~0.20 GB |
| Parquet | Analytical scans, interchange | 35.8 s/s | 605 s/s | 19.6 ms | 1.9 ms | ~0.19 GB |
| WebDataset | Sequential training export | 36.5 s/s | 635 s/s | — | — | ~0.20 GB |

*Mini-set (100-sample) numbers for Redis/Lance/Parquet/WebDataset; full trainval for MinIO+PostgreSQL.*

For `minio-postgres`, one real sample fans out into:

- `12` object writes to MinIO:
  `6` camera JPEGs, `1` LiDAR `.npy`, `5` radar `.npy`
- `1` scene upsert in PostgreSQL
- `1` sample row insert in PostgreSQL
- `N` annotation row inserts in PostgreSQL

On the completed full trainval run, the dataset averaged `34.15` annotations per sample. At the
measured `8.15 samples/sec` ingest rate, that maps to roughly:

- `97.75` MinIO object `PUT`s/sec
- `294.46` PostgreSQL write statements/sec
- `16.73 MB/sec` of persisted payload bytes

Those are single-runner application-level rates. They are not cluster maximums, and they are not
synthetic HTTP load-test numbers.

The backends are meant for different access shapes:

- `minio-postgres`: canonical blob + metadata path; best fit for the explorer and mixed retrieval.
- `redis`: hot metadata and embedding retrieval; no full image/LiDAR blob persistence.
- `lance`: Arrow-native random access and curation-friendly local dataset.
- `parquet`: columnar export for analytical scans and cheap interchange.
- `webdataset`: tar-sharded sequential training path; no point lookup or query support.

Read-side behavior differs by backend. In the current benchmark suite:

- `minio-postgres` sequential scan means one ordered SQL cursor plus `2` MinIO `GET`s per sample
  (`CAM_FRONT` and `LIDAR_TOP`).
- `minio-postgres` random access means one SQL point lookup plus `2` MinIO `GET`s for that sample.
- `redis` reads metadata hashes and embeddings only.
- `lance` and `parquet` read `CAM_FRONT` and `LIDAR_TOP` from local columnar files.
- `webdataset` is sequential only and deliberately skips random-access and curation-query tests.

## How Numbers Are Calculated

The benchmark and telemetry numbers are direct wall-clock calculations from the code paths under
`src/nudemo/benchmarks/`, `src/nudemo/storage/`, and `src/nudemo/studies/`:

| Metric | Calculation |
| --- | --- |
| Storage write throughput | `samples_written / elapsed_sec` |
| Batch ingest p50 / p95 | Percentiles over the per-batch `elapsed_sec` values |
| Sequential scan throughput | `total_samples / elapsed_sec` |
| Random access latency | p50 / p95 / p99 of individual fetch latencies in milliseconds |
| Curation query latency | Mean and stddev of query-only elapsed time |
| End-to-end curation | Query time plus all matching fetches, plus `per_sample_ms` |
| Disk footprint | Backend-reported bytes on disk or in object storage |
| Service pressure | Peak Docker CPU, memory, network, block I/O, and PID counts from persisted snapshots |

The concrete implementations are:

- `StorageWriteResult.throughput` in `src/nudemo/storage/base.py`
- `benchmark_sequential()` in `src/nudemo/benchmarks/runner.py`
- `benchmark_random_access()` in `src/nudemo/benchmarks/runner.py`
- `benchmark_curation_query()` in `src/nudemo/benchmarks/runner.py`
- `benchmark_end_to_end_curation()` in `src/nudemo/benchmarks/runner.py`
- `run_batched_ingest_study()` in `src/nudemo/studies/batched_ingest.py`
- `TelemetryRecorder` in `src/nudemo/telemetry/store.py`

The Prometheus exporter does not invent separate numbers. It republishes the latest persisted run,
span, and service-snapshot values from PostgreSQL as:

- `nudemo_latest_run_metric_value`
- `nudemo_latest_span_metric_value`
- `nudemo_latest_service_metric_value`

That means the HTML dashboards, Prometheus, and Grafana are all reading the same persisted telemetry
source of truth.

## Real-World Interpretation

This repo measures `samples/sec`, `messages/sec`, and fetch/query latency because a nuScenes sample
is a composite unit, not a single blob request. When you present the system, frame the numbers like
this:

- Ingest throughput tells you how fast the pipeline can materialize synchronized multimodal samples.
- Random-access latency tells you how fast one evaluation or curation fetch returns a sample view.
- Sequential throughput tells you how well a backend behaves as a training/evaluation reader.
- Curation latency tells you how expensive metadata filtering is before any sample bytes are fetched.
- Service pressure tells you whether the bottleneck is infrastructure saturation or client-side code.

A good example from the completed full `minio-postgres` trainval run:

- Ingest: `8.15 samples/sec`
- Sequential scan: `263.64 samples/sec`
- Random access p50: `12.64 ms`
- Curation query mean: `227.87 ms`
- Disk footprint: `68.59 GB`
- Peak service CPU: `96.83%` on `minio`

That is the kind of backend-specific operating profile you can compare against Redis, Lance,
Parquet, and WebDataset in the overnight study summaries.

## Current Status

- Verified locally: real nuScenes-mini extraction, Kafka ingestion, MinIO/PostgreSQL, Redis, Lance, Parquet, and WebDataset benchmark runs, JSON/CSV export, benchmark dashboard HTML generation, telemetry ingestion into PostgreSQL, telemetry dashboard HTML generation, browser-based sample exploration, and official trainval archive discovery from the AWS Open Data mirror.
- Recent benchmark runs can be queried with `nudemo telemetry runs` and re-rendered with `nudemo telemetry dashboard`.
- Telemetry is additive. If PostgreSQL or Docker snapshots are unavailable, the benchmark still writes the benchmark report and dashboard artifacts.
- Target runtime: Python `3.12`; the repo bootstraps that version explicitly because the external stack is not yet a clean Python 3.13 target.
