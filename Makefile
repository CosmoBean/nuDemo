.DEFAULT_GOAL := help

UV ?= uv
PYTHON ?= 3.12
CONFIG ?=
PROVIDER ?= real
LIMIT ?=
BACKEND ?= lance
MODE ?= metadata-only
BACKENDS ?=
RESULTS ?= artifacts/reports/benchmark_report.json
NUM_RUNS ?= 1
RANDOM_SAMPLE_COUNT ?= 10
BATCH_SIZE ?= 4
NUM_WORKERS ?= 0 2 4
EXTRA_ARGS ?=
DOCKER_COMPOSE ?= docker compose -f config/docker-compose.yml

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

ifdef BACKENDS
BACKENDS_ARGS := --backends $(BACKENDS)
else
BACKENDS_ARGS :=
endif

.PHONY: help bootstrap bootstrap-legacy check-env deps doctor cli extract extract-synthetic \
	kafka kafka-topics kafka-metadata kafka-full storage storage-minio-postgres storage-redis \
	storage-lance storage-webdataset benchmark-sim benchmark-real dashboard lint test \
	clean infra-up infra-down infra-ps infra-logs

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "%-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

bootstrap: ## Create/update the pinned Python 3.12 environment
	@bash ./scripts/bootstrap_env.sh

bootstrap-legacy: ## Run the older bootstrap helper
	@bash ./scripts/bootstrap.sh

check-env: ## Validate local tooling and project config
	@bash ./scripts/check_env.sh

deps: infra-up ## Start all compose-backed project dependencies

doctor: ## Inspect runtime, dataset visibility, and dataset counts
	@$(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) doctor

cli: ## Run an arbitrary nudemo CLI command via ARGS="..."
	@$(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) $(ARGS)

extract: ## Extract sample metadata from the configured provider
	@$(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) extract --provider $(PROVIDER) $(LIMIT_ARGS) $(EXTRA_ARGS)

extract-synthetic: ## Extract one synthetic sample for smoke testing
	@$(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) extract --provider synthetic $(LIMIT_ARGS) $(EXTRA_ARGS)

kafka: ## Produce Kafka messages with MODE=metadata-only|full-payload
	@$(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) kafka --provider $(PROVIDER) $(LIMIT_ARGS) --mode $(MODE) $(EXTRA_ARGS)

kafka-topics: ## Create Kafka topics for the live pipeline
	@$(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) kafka --provider $(PROVIDER) $(LIMIT_ARGS) --create-topics $(EXTRA_ARGS)

kafka-metadata: ## Produce metadata-only Kafka messages
	@$(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) kafka --provider $(PROVIDER) $(LIMIT_ARGS) --mode metadata-only $(EXTRA_ARGS)

kafka-full: ## Produce full-payload Kafka benchmark messages
	@$(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) kafka --provider $(PROVIDER) $(LIMIT_ARGS) --mode full-payload $(EXTRA_ARGS)

storage: ## Write samples to one backend via BACKEND=minio-postgres|redis|lance|webdataset
	@$(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) storage $(BACKEND) --provider $(PROVIDER) $(LIMIT_ARGS) $(EXTRA_ARGS)

storage-minio-postgres: ## Write samples to MinIO + PostgreSQL
	@$(MAKE) storage BACKEND=minio-postgres PROVIDER=$(PROVIDER) LIMIT="$(LIMIT)" EXTRA_ARGS="$(EXTRA_ARGS)"

storage-redis: ## Write samples to Redis
	@$(MAKE) storage BACKEND=redis PROVIDER=$(PROVIDER) LIMIT="$(LIMIT)" EXTRA_ARGS="$(EXTRA_ARGS)"

storage-lance: ## Write samples to Lance
	@$(MAKE) storage BACKEND=lance PROVIDER=$(PROVIDER) LIMIT="$(LIMIT)" EXTRA_ARGS="$(EXTRA_ARGS)"

storage-webdataset: ## Write samples to WebDataset
	@$(MAKE) storage BACKEND=webdataset PROVIDER=$(PROVIDER) LIMIT="$(LIMIT)" EXTRA_ARGS="$(EXTRA_ARGS)"

benchmark-sim: ## Run the in-memory synthetic benchmark suite
	@$(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) benchmark run --simulate --provider synthetic $(LIMIT_ARGS) $(BACKENDS_ARGS) --num-runs $(NUM_RUNS) --random-sample-count $(RANDOM_SAMPLE_COUNT) --batch-size $(BATCH_SIZE) --num-workers $(NUM_WORKERS) $(EXTRA_ARGS)

benchmark-real: ## Run live backend benchmarks against the configured provider
	@$(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) benchmark run --provider $(PROVIDER) $(LIMIT_ARGS) $(BACKENDS_ARGS) --num-runs $(NUM_RUNS) --random-sample-count $(RANDOM_SAMPLE_COUNT) --batch-size $(BATCH_SIZE) --num-workers $(NUM_WORKERS) $(EXTRA_ARGS)

dashboard: ## Render the benchmark dashboard from RESULTS=...
	@$(UV) run --python $(PYTHON) nudemo $(CONFIG_ARGS) benchmark dashboard --results-path $(RESULTS)

lint: ## Run Ruff over src/ and tests/
	@$(UV) run --python $(PYTHON) ruff check src tests

test: ## Run the test suite
	@$(UV) run --python $(PYTHON) pytest

clean: ## Remove common generated artifacts
	@rm -rf .pytest_cache .ruff_cache .mypy_cache build dist artifacts *.egg-info
	@find src tests -type d -name '__pycache__' -prune -exec rm -rf {} +

infra-up: ## Start Kafka, MinIO, PostgreSQL, and Redis via docker compose
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
