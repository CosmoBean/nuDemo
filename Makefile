.DEFAULT_GOAL := help

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
MODE ?= metadata-only
BACKENDS ?=
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
EXTRA_ARGS ?=
NUSCENES_PROFILE ?= keyframes
DOWNLOAD_MODE ?= all
KEEP_ARCHIVES ?= 1
DOCKER_COMPOSE ?= docker compose -f config/docker-compose.yml

RUN_ENV_VARS := $(strip $(if $(DATASET_VERSION),NUDEMO_DATASET_VERSION=$(DATASET_VERSION)) $(if $(DATASET_ROOT),NUDEMO_DATASET_ROOT=$(DATASET_ROOT)))

ifdef CONFIG
CONFIG_ARGS := --config $(CONFIG)
else
CONFIG_ARGS :=
endif

ifdef LIMIT
LIMIT_ARGS := --limit $(LIMIT)
else
LIMIT_ARGS :=
endif

ifdef SCENE_LIMIT
SCENE_LIMIT_ARGS := --scene-limit $(SCENE_LIMIT)
else
SCENE_LIMIT_ARGS :=
endif

ifdef BACKENDS
BACKENDS_ARGS := --backends $(BACKENDS)
else
BACKENDS_ARGS :=
endif

.PHONY: help bootstrap bootstrap-legacy check-env deps doctor cli extract extract-synthetic \
	kafka kafka-topics kafka-metadata kafka-full storage storage-minio-postgres storage-redis \
	storage-lance storage-parquet storage-webdataset benchmark-sim benchmark-real dashboard render-sample \
	render-scene download-trainval download-trainval-full telemetry-runs multimodal-index \
	telemetry-dashboard reports-index serve-reports data-explorer lint test clean infra-up \
	infra-down infra-ps infra-logs overnight-study

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
	@env $(RUN_ENV_VARS) $(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) doctor

cli: ## Run an arbitrary nudemo CLI command via ARGS="..."
	@env $(RUN_ENV_VARS) $(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) $(ARGS)

extract: ## Extract sample metadata from the configured provider
	@env $(RUN_ENV_VARS) $(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) extract --provider $(PROVIDER) $(LIMIT_ARGS) $(SCENE_LIMIT_ARGS) $(EXTRA_ARGS)

extract-synthetic: ## Extract one synthetic sample for smoke testing
	@env $(RUN_ENV_VARS) $(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) extract --provider synthetic $(LIMIT_ARGS) $(SCENE_LIMIT_ARGS) $(EXTRA_ARGS)

kafka: ## Produce Kafka messages with MODE=metadata-only|full-payload
	@env $(RUN_ENV_VARS) $(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) kafka --provider $(PROVIDER) $(LIMIT_ARGS) $(SCENE_LIMIT_ARGS) --mode $(MODE) $(EXTRA_ARGS)

kafka-topics: ## Create Kafka topics for the live pipeline
	@env $(RUN_ENV_VARS) $(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) kafka --provider $(PROVIDER) $(LIMIT_ARGS) $(SCENE_LIMIT_ARGS) --create-topics $(EXTRA_ARGS)

kafka-metadata: ## Produce metadata-only Kafka messages
	@env $(RUN_ENV_VARS) $(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) kafka --provider $(PROVIDER) $(LIMIT_ARGS) $(SCENE_LIMIT_ARGS) --mode metadata-only $(EXTRA_ARGS)

kafka-full: ## Produce full-payload Kafka benchmark messages
	@env $(RUN_ENV_VARS) $(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) kafka --provider $(PROVIDER) $(LIMIT_ARGS) $(SCENE_LIMIT_ARGS) --mode full-payload $(EXTRA_ARGS)

storage: ## Write samples to one backend via BACKEND=minio-postgres|redis|lance|parquet|webdataset
	@env $(RUN_ENV_VARS) $(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) storage $(BACKEND) --provider $(PROVIDER) $(LIMIT_ARGS) $(SCENE_LIMIT_ARGS) $(EXTRA_ARGS)

storage-minio-postgres: ## Write samples to MinIO + PostgreSQL
	@$(MAKE) storage BACKEND=minio-postgres PROVIDER=$(PROVIDER) LIMIT="$(LIMIT)" EXTRA_ARGS="$(EXTRA_ARGS)"

storage-redis: ## Write samples to Redis
	@$(MAKE) storage BACKEND=redis PROVIDER=$(PROVIDER) LIMIT="$(LIMIT)" EXTRA_ARGS="$(EXTRA_ARGS)"

storage-lance: ## Write samples to Lance
	@$(MAKE) storage BACKEND=lance PROVIDER=$(PROVIDER) LIMIT="$(LIMIT)" EXTRA_ARGS="$(EXTRA_ARGS)"

storage-parquet: ## Write samples to Parquet
	@$(MAKE) storage BACKEND=parquet PROVIDER=$(PROVIDER) LIMIT="$(LIMIT)" EXTRA_ARGS="$(EXTRA_ARGS)"

storage-webdataset: ## Write samples to WebDataset
	@$(MAKE) storage BACKEND=webdataset PROVIDER=$(PROVIDER) LIMIT="$(LIMIT)" EXTRA_ARGS="$(EXTRA_ARGS)"

benchmark-sim: ## Run the in-memory synthetic benchmark suite
	@env $(RUN_ENV_VARS) $(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) benchmark run --simulate --provider synthetic $(LIMIT_ARGS) $(SCENE_LIMIT_ARGS) $(BACKENDS_ARGS) --num-runs $(NUM_RUNS) --random-sample-count $(RANDOM_SAMPLE_COUNT) --batch-size $(BATCH_SIZE) --num-workers $(NUM_WORKERS) $(EXTRA_ARGS)

benchmark-real: ## Run live backend benchmarks against the configured provider
	@env $(RUN_ENV_VARS) $(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) benchmark run --provider $(PROVIDER) $(LIMIT_ARGS) $(SCENE_LIMIT_ARGS) $(BACKENDS_ARGS) --num-runs $(NUM_RUNS) --random-sample-count $(RANDOM_SAMPLE_COUNT) --batch-size $(BATCH_SIZE) --num-workers $(NUM_WORKERS) $(EXTRA_ARGS)

dashboard: ## Render the benchmark dashboard from RESULTS=...
	@env $(RUN_ENV_VARS) $(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) benchmark dashboard --results-path $(RESULTS)

render-sample: ## Render a sample contact sheet to artifacts/reports/renders; use SAMPLE_IDX=<n>
	@env $(RUN_ENV_VARS) $(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) render sample --provider $(PROVIDER) --sample-idx $(SAMPLE_IDX) $(if $(OUTPUT),--output $(OUTPUT),) $(EXTRA_ARGS)

render-scene: ## Render a scene GIF to artifacts/reports/renders; optional SCENE_NAME=... CAMERA=... MAX_FRAMES=... FRAME_STEP=... FPS=...
	@env $(RUN_ENV_VARS) $(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) render scene $(if $(SCENE_NAME),--scene-name $(SCENE_NAME),) --camera $(CAMERA) --max-frames $(MAX_FRAMES) --step $(FRAME_STEP) --fps $(FPS) $(if $(OUTPUT),--output $(OUTPUT),) $(EXTRA_ARGS)

download-trainval: ## Download/extract official nuScenes v1.0-trainval keyframes from AWS Open Data
	@NUSCENES_PROFILE="$(NUSCENES_PROFILE)" NUSCENES_MODE="$(DOWNLOAD_MODE)" NUSCENES_KEEP_ARCHIVES="$(KEEP_ARCHIVES)" \
		NUSCENES_RAW_ROOT="$(RAW_ROOT)" NUSCENES_DATASET_ROOT="$(if $(DATASET_ROOT),$(DATASET_ROOT),$(CURDIR)/data/nuscenes)" \
		bash ./scripts/download_nuscenes_trainval.sh

download-trainval-full: ## Download/extract trainval keyframes plus sweep blobs; requires substantially more disk
	@$(MAKE) download-trainval NUSCENES_PROFILE=full DOWNLOAD_MODE="$(DOWNLOAD_MODE)" KEEP_ARCHIVES="$(KEEP_ARCHIVES)" RAW_ROOT="$(RAW_ROOT)" DATASET_ROOT="$(DATASET_ROOT)"

telemetry-runs: ## Show recent telemetry runs stored in PostgreSQL
	@env $(RUN_ENV_VARS) $(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) telemetry runs --limit $(or $(LIMIT),10)

multimodal-index: ## Build the multimodal Elasticsearch index from loaded samples
	@env $(RUN_ENV_VARS) $(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) multimodal-index $(LIMIT_ARGS) $(SCENE_LIMIT_ARGS) --batch-size $(or $(BATCH_SIZE),24) $(EXTRA_ARGS)

telemetry-dashboard: ## Render telemetry dashboard from PostgreSQL; optional RUN_ID=<id>
	@env $(RUN_ENV_VARS) $(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) telemetry dashboard $(if $(RUN_ID),--run-id $(RUN_ID),--latest)

reports-index: ## Render artifacts/reports/index.html for browser-friendly navigation
	@python3 ./scripts/render_reports_index.py "$(REPORTS_ROOT)"

serve-reports: ## Serve artifacts/reports over localhost for browser or tunnel access
	@python3 ./scripts/render_reports_index.py "$(REPORTS_ROOT)" >/dev/null
	@NUDEMO_REPORTS_ROOT="$(REPORTS_ROOT)" NUDEMO_REPORTS_HOST="$(REPORTS_HOST)" NUDEMO_REPORTS_PORT="$(REPORTS_PORT)" \
		bash ./scripts/serve_reports.sh

data-explorer: ## Serve the searchable ingested-data explorer
	@NUDEMO_EXPLORER_HOST="$(EXPLORER_HOST)" NUDEMO_EXPLORER_PORT="$(EXPLORER_PORT)" NUDEMO_EXPLORER_LIMIT="$(EXPLORER_LIMIT)" \
		bash ./scripts/serve_explorer.sh

overnight-study: ## Run the sequential full-data batched ingest study; override EXTRA_ARGS or env vars for tuning
	@env $(RUN_ENV_VARS) \
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
