# nuDemo

Storage backend benchmark for autonomous driving sensor data. Extracts nuScenes samples (cameras, LiDAR, radar, annotations) and measures ingest throughput, sequential scan, random access latency, curation query time, and disk footprint across five backends.

## Results — v1.0-trainval (34,149 samples, 850 scenes)

| Backend | Ingest | Seq scan | Rand p50 | Curation | Disk |
|---|---|---|---|---|---|
| MinIO + PostgreSQL | 8.2 s/s | 264 s/s | 12.6 ms | 228 ms | 68.6 GB |
| Lance | 37.6 s/s | 2,598 s/s | 3.6 ms | 749 ms | ~68 GB |
| Parquet | 35.8 s/s | 1,909 s/s | 135 ms | 242 ms | ~65 GB |
| WebDataset | 36.0 s/s | 387 s/s | — | — | ~69 GB |
| Redis¹ | 184.9 s/s | 3,705 s/s | 0.87 ms | 3,045 ms | 142 MB |

¹ Redis stores metadata + embeddings only (~5 KB/sample). Not a blob storage replacement — used as a hot-path index in front of a full-fidelity backend.

All numbers are wall-clock measurements from real runs on this host, not synthetic profiles.

## Quickstart

```bash
./scripts/bootstrap_env.sh          # Python 3.12 venv
.venv/bin/nudemo doctor             # check environment

# No-Docker smoke test (synthetic data)
.venv/bin/nudemo benchmark run --simulate --provider synthetic --limit 24

# Start Docker services (Kafka, MinIO, PostgreSQL, Redis, Prometheus, Grafana)
make infra-up

# Load the searchable multimodal index from the already-ingested MinIO+PostgreSQL corpus
make multimodal-index BATCH_SIZE=24

# Materialize tracks from the loaded nuScenes corpus and build the track search index
make track-index BATCH_SIZE=250

# Search tracks or export a saved cohort manifest
make track-search EXTRA_ARGS='--q pedestrian'
make export-cohort EXTRA_ARGS='cohort_id_here'

# Download nuScenes v1.0-trainval (~42 GB keyframes)
make download-trainval

# Run full overnight study (one backend at a time, purges between runs)
make overnight-study DATASET_VERSION=v1.0-trainval BACKENDS="minio-postgres redis lance parquet webdataset"
```

Budget ~2.5 hours per blob backend on this host. Running all five sequentially takes an overnight window (~10–12 hours).

## Browser UI

```bash
make data-explorer    # http://127.0.0.1:8788
```

| Route | What |
|---|---|
| `/` | Home — links to all views |
| `/compare` | Backend comparison — charts and ranked table |
| `/explorer` | Unified lexical + multimodal sample search, mining sessions, saved cohorts |
| `/scene-studio` | 3D LiDAR scrubber per scene |
| `/tasks` | Review task board for cohorts and tracks |
| `/open-grafana` | Grafana dashboards redirect |

Requires the `minio-postgres` backend to be loaded. Multimodal search requires `make multimodal-index` after ingest. Track review requires `make track-index` after the sample corpus is loaded. Grafana at `http://127.0.0.1:3000/grafana/`.

## Architecture

```
nuScenes → Extraction → Storage backends (swappable) → Benchmarks / Explorer
                  ↓                         ↓
           Kafka refined topic        MinIO + PostgreSQL
                  ↓                         ↓
                Redis cache       Multimodal sample index → Elasticsearch
                                             ↓
                        Explorer search / mining / cohorts / task creation
                                             ↓
            Raw nuScenes metadata → Track materializer → Postgres tracks + track index
                                             ↓
                        Scene Studio track review / Tasks board / Cohort export
```

Hot write path: Extraction → direct write to backend (35–37 s/s for Lance/Parquet/WebDataset).
Kafka full-payload consume measured at ~3.4 msg/s — not in the write path.

Multimodal search path:
- `make storage-minio-postgres ...` loads the full-fidelity browser corpus.
- `make multimodal-index` builds one Elasticsearch document per sample with lexical fields plus image, LiDAR, radar, metadata, and fused vectors.
- `/explorer` keeps one search bar. The backend transparently upgrades text search into hybrid retrieval and lets you steer with positive and negative examples.
- Mining sessions and saved cohorts live in PostgreSQL.

Track-aware review path:
- `make track-index` walks the raw nuScenes metadata for the currently loaded corpus, materializes `tracks` and `track_observations` into PostgreSQL, and indexes track summaries into Elasticsearch.
- `/explorer` can surface track hits alongside sample hits from the same query.
- `/scene-studio?track_id=...` opens temporal review around a specific track.
- `/tasks` manages `queued → assigned → in_progress → submitted → qa_failed|qa_passed → closed`.
- `make export-cohort EXTRA_ARGS='<cohort_id>'` writes a Parquet manifest and records it in `cohort_exports`.

## Stack

| Service | Purpose |
|---|---|
| MinIO | Object storage for camera JPEGs, LiDAR/radar arrays |
| PostgreSQL | Sample metadata, annotations, tracks, review tasks, exports, telemetry history |
| Redis | Metadata index, embedding cache |
| Kafka | Async metadata bus (refined topic → Redis warmup) |
| Elasticsearch | Lexical + multimodal sample retrieval plus track retrieval |
| Prometheus | Scrapes `:9464` — run metrics, span metrics, service pressure |
| Grafana | Backend comparison, query latency monitor, stage summary, workflow/task metrics |

## Development

```bash
make test     # pytest
make lint     # ruff
make doctor   # runtime + dataset check
```

Python 3.12 only.
