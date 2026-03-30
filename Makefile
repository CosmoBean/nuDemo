.DEFAULT_GOAL := help

# Common overrides understood by the wrapper scripts in scripts/.
# Example:
#   make benchmark-real SCENE_LIMIT=200 BACKENDS="redis lance parquet" NUM_WORKERS="0 4"
#   make storage BACKEND=minio-postgres LIMIT=1024

UV ?= uv
PYTHON ?= 3.12
CONFIG ?=
PROVIDER ?= real
DATASET_VERSION ?=
DATASET_ROOT ?=
RAW_ROOT ?= $(CURDIR)/data/raw/v1.0-trainval
LIMIT ?=
SCENE_LIMIT ?=
BACKEND ?= lance
BACKENDS ?=
MODE ?= metadata-only
RESULTS ?= artifacts/reports/benchmark_report.json
RUN_ID ?=
NUM_RUNS ?= 1
RANDOM_SAMPLE_COUNT ?= 10
BATCH_SIZE ?= 4
STUDY_BATCH_SIZE ?= 32
SNAPSHOT_EVERY_BATCHES ?= 1
NUM_WORKERS ?= 0 2 4
REPORTS_HOST ?= 127.0.0.1
REPORTS_PORT ?= 8787
REPORTS_ROOT ?= $(CURDIR)/artifacts/reports
EXPLORER_HOST ?= 127.0.0.1
EXPLORER_PORT ?= 8788
EXPLORER_LIMIT ?= 200
SAMPLE_IDX ?= 0
SCENE_NAME ?=
CAMERA ?= CAM_FRONT
MAX_FRAMES ?= 24
FRAME_STEP ?= 1
FPS ?= 2
OUTPUT ?=
COHORT_ID ?=
TASK_ID ?=
Q ?=
SOURCE ?= elasticsearch
STATUS ?=
SOURCE_TYPE ?=
SOURCE_ID ?=
OFFSET ?=
EXTRA_ARGS ?=
NUSCENES_PROFILE ?= keyframes
DOWNLOAD_MODE ?= all
KEEP_ARCHIVES ?= 1
DOCKER_COMPOSE ?= docker compose -f config/docker-compose.yml
CREATE_TOPICS ?= 0
APPEND ?= 0
MATERIALIZE_ONLY ?= 0
INDEX_ONLY ?= 0

export UV PYTHON CONFIG PROVIDER DATASET_VERSION DATASET_ROOT RAW_ROOT LIMIT SCENE_LIMIT BACKEND \
	BACKENDS MODE RESULTS RUN_ID NUM_RUNS RANDOM_SAMPLE_COUNT BATCH_SIZE STUDY_BATCH_SIZE \
	SNAPSHOT_EVERY_BATCHES NUM_WORKERS REPORTS_HOST REPORTS_PORT REPORTS_ROOT EXPLORER_HOST \
	EXPLORER_PORT EXPLORER_LIMIT SAMPLE_IDX SCENE_NAME CAMERA MAX_FRAMES FRAME_STEP FPS OUTPUT \
	COHORT_ID TASK_ID Q SOURCE STATUS SOURCE_TYPE SOURCE_ID OFFSET EXTRA_ARGS NUSCENES_PROFILE \
	DOWNLOAD_MODE KEEP_ARCHIVES DOCKER_COMPOSE CREATE_TOPICS APPEND MATERIALIZE_ONLY INDEX_ONLY

.PHONY: help bootstrap bootstrap-legacy check-env deps doctor cli extract extract-synthetic \
	kafka kafka-topics kafka-metadata kafka-full storage storage-minio-postgres storage-redis \
	storage-lance storage-parquet storage-webdataset benchmark-sim benchmark-real dashboard render-sample \
	render-scene download-trainval download-trainval-full telemetry-runs multimodal-index \
	telemetry-dashboard reports-index serve-reports data-explorer track-index track-search \
	export-cohort tasks lint lint-pylint test clean infra-up infra-down infra-ps infra-logs overnight-study

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "%-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

bootstrap: ## Create/update the pinned Python 3.12 environment
	@bash ./scripts/bootstrap_env.sh

bootstrap-legacy: ## Run the older bootstrap helper
	@bash ./scripts/bootstrap.sh

check-env: ## Validate local tooling and project config
	@bash ./scripts/check_env.sh

deps: infra-up ## Start all compose-backed project dependencies, including Prometheus and Grafana

doctor: ## Inspect runtime, dataset visibility, and dataset counts
	@$(UV) run --python $(PYTHON) nudemo $(if $(CONFIG),--config $(CONFIG),) doctor

cli: ## Run an arbitrary nudemo CLI command via ARGS="..."
	@$(UV) run --python $(PYTHON) nudemo $(if $(CONFIG),--config $(CONFIG),) $(ARGS)

extract: ## Extract sample metadata from the configured provider
	@bash ./scripts/run_extract.sh $(EXTRA_ARGS)

extract-synthetic: ## Extract one synthetic sample for smoke testing
	@PROVIDER=synthetic bash ./scripts/run_extract.sh $(EXTRA_ARGS)

kafka: ## Produce Kafka messages with MODE=metadata-only|full-payload
	@bash ./scripts/run_kafka.sh $(EXTRA_ARGS)

kafka-topics: ## Create Kafka topics for the live pipeline
	@CREATE_TOPICS=1 bash ./scripts/run_kafka.sh $(EXTRA_ARGS)

kafka-metadata: ## Produce metadata-only Kafka messages
	@MODE=metadata-only bash ./scripts/run_kafka.sh $(EXTRA_ARGS)

kafka-full: ## Produce full-payload Kafka benchmark messages
	@MODE=full-payload bash ./scripts/run_kafka.sh $(EXTRA_ARGS)

storage: ## Write samples to one backend via BACKEND=minio-postgres|redis|lance|parquet|webdataset
	@bash ./scripts/run_storage.sh $(EXTRA_ARGS)

storage-minio-postgres: ## Write samples to MinIO + PostgreSQL
	@BACKEND=minio-postgres bash ./scripts/run_storage.sh $(EXTRA_ARGS)

storage-redis: ## Write samples to Redis
	@BACKEND=redis bash ./scripts/run_storage.sh $(EXTRA_ARGS)

storage-lance: ## Write samples to Lance
	@BACKEND=lance bash ./scripts/run_storage.sh $(EXTRA_ARGS)

storage-parquet: ## Write samples to Parquet
	@BACKEND=parquet bash ./scripts/run_storage.sh $(EXTRA_ARGS)

storage-webdataset: ## Write samples to WebDataset
	@BACKEND=webdataset bash ./scripts/run_storage.sh $(EXTRA_ARGS)

benchmark-sim: ## Run the in-memory synthetic benchmark suite
	@bash ./scripts/run_benchmark.sh sim $(EXTRA_ARGS)

benchmark-real: ## Run live backend benchmarks against the configured provider
	@bash ./scripts/run_benchmark.sh real $(EXTRA_ARGS)

dashboard: ## Render the benchmark dashboard from RESULTS=...
	@$(UV) run --python $(PYTHON) nudemo $(if $(CONFIG),--config $(CONFIG),) benchmark dashboard --results-path "$(RESULTS)"

render-sample: ## Render a sample contact sheet to artifacts/reports/renders; use SAMPLE_IDX=<n>
	@bash ./scripts/run_render.sh sample $(EXTRA_ARGS)

render-scene: ## Render a scene GIF to artifacts/reports/renders; optional SCENE_NAME=... CAMERA=... MAX_FRAMES=... FRAME_STEP=... FPS=...
	@bash ./scripts/run_render.sh scene $(EXTRA_ARGS)

download-trainval: ## Download/extract official nuScenes v1.0-trainval keyframes from AWS Open Data
	@NUSCENES_PROFILE="$(NUSCENES_PROFILE)" NUSCENES_MODE="$(DOWNLOAD_MODE)" NUSCENES_KEEP_ARCHIVES="$(KEEP_ARCHIVES)" \
		NUSCENES_RAW_ROOT="$(RAW_ROOT)" NUSCENES_DATASET_ROOT="$(if $(DATASET_ROOT),$(DATASET_ROOT),$(CURDIR)/data/nuscenes)" \
		bash ./scripts/download_nuscenes_trainval.sh

download-trainval-full: ## Download/extract trainval keyframes plus sweep blobs; requires substantially more disk
	@$(MAKE) download-trainval NUSCENES_PROFILE=full DOWNLOAD_MODE="$(DOWNLOAD_MODE)" KEEP_ARCHIVES="$(KEEP_ARCHIVES)" RAW_ROOT="$(RAW_ROOT)" DATASET_ROOT="$(DATASET_ROOT)"

telemetry-runs: ## Show recent telemetry runs stored in PostgreSQL
	@bash ./scripts/run_telemetry.sh runs $(EXTRA_ARGS)

multimodal-index: ## Build the multimodal Elasticsearch index from loaded samples
	@bash ./scripts/run_indexing.sh multimodal $(EXTRA_ARGS)

track-index: ## Materialize track tables and index them into Elasticsearch
	@bash ./scripts/run_indexing.sh track $(EXTRA_ARGS)

track-search: ## Search materialized tracks; use Q=... CATEGORY=... SOURCE=... for tuning
	@bash ./scripts/run_indexing.sh track-search $(EXTRA_ARGS)

export-cohort: ## Export a saved cohort as Parquet; set COHORT_ID and optional TASK_ID
	@bash ./scripts/run_indexing.sh export-cohort $(EXTRA_ARGS)

tasks: ## Operate on review tasks; pass EXTRA_ARGS like "list" or "create --title ..."
	@bash ./scripts/run_tasks.sh $(EXTRA_ARGS)

telemetry-dashboard: ## Render telemetry dashboard from PostgreSQL; optional RUN_ID=<id>
	@bash ./scripts/run_telemetry.sh dashboard $(EXTRA_ARGS)

reports-index: ## Render artifacts/reports/index.html for browser-friendly navigation
	@python3 ./scripts/render_reports_index.py "$(REPORTS_ROOT)"

serve-reports: ## Serve artifacts/reports over localhost for browser or tunnel access
	@python3 ./scripts/render_reports_index.py "$(REPORTS_ROOT)" >/dev/null
	@NUDEMO_REPORTS_ROOT="$(REPORTS_ROOT)" NUDEMO_REPORTS_HOST="$(REPORTS_HOST)" NUDEMO_REPORTS_PORT="$(REPORTS_PORT)" \
		bash ./scripts/serve_reports.sh

data-explorer: ## Serve the searchable ingested-data explorer
	@NUDEMO_EXPLORER_HOST="$(EXPLORER_HOST)" NUDEMO_EXPLORER_PORT="$(EXPLORER_PORT)" NUDEMO_EXPLORER_LIMIT="$(EXPLORER_LIMIT)" \
		bash ./scripts/serve_explorer.sh

overnight-study: ## Run the sequential full-data batched ingest study; override env vars for tuning
	@env \
		NUDEMO_STUDY_LIMIT="$(LIMIT)" \
		NUDEMO_STUDY_SCENE_LIMIT="$(SCENE_LIMIT)" \
		NUDEMO_STUDY_BACKENDS="$(BACKENDS)" \
		NUDEMO_STUDY_PROVIDER="$(PROVIDER)" \
		NUDEMO_STUDY_RANDOM_SAMPLE_COUNT="$(RANDOM_SAMPLE_COUNT)" \
		NUDEMO_STUDY_BATCH_SIZE="$(STUDY_BATCH_SIZE)" \
		NUDEMO_SNAPSHOT_EVERY_BATCHES="$(SNAPSHOT_EVERY_BATCHES)" \
		bash ./scripts/run_overnight_batched_study.sh $(EXTRA_ARGS)

lint: ## Run Ruff over src/ and tests/
	@$(UV) run --python $(PYTHON) ruff check src tests

lint-pylint: ## Run pylint over src/nudemo
	@bash ./scripts/run_pylint.sh src/nudemo

test: ## Run the test suite
	@$(UV) run --python $(PYTHON) pytest

clean: ## Remove common generated artifacts
	@rm -rf .pytest_cache .ruff_cache .mypy_cache build dist artifacts *.egg-info
	@find src tests -type d -name '__pycache__' -prune -exec rm -rf {} +

infra-up: ## Start Kafka, MinIO, PostgreSQL, Redis, Prometheus, and Grafana via docker compose
	@if ! command -v docker >/dev/null 2>&1; then echo "docker is not installed"; exit 1; fi
	@$(DOCKER_COMPOSE) up -d

infra-down: ## Stop local infrastructure containers
	@if ! command -v docker >/dev/null 2>&1; then echo "docker is not installed"; exit 1; fi
	@$(DOCKER_COMPOSE) down

infra-ps: ## Show compose service status
	@if ! command -v docker >/dev/null 2>&1; then echo "docker is not installed"; exit 1; fi
	@$(DOCKER_COMPOSE) ps

infra-logs: ## Tail compose logs; use EXTRA_ARGS="service-name"
	@if ! command -v docker >/dev/null 2>&1; then echo "docker is not installed"; exit 1; fi
	@$(DOCKER_COMPOSE) logs -f $(EXTRA_ARGS)
