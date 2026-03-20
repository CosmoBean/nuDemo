#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${NUDEMO_STUDY_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${NUDEMO_LOG_DIR:-artifacts/logs}"
REPORT_ROOT_BASE="${NUDEMO_STUDY_OUTPUT_BASE:-artifacts/reports/overnight_runs}"
REPORT_ROOT="${NUDEMO_STUDY_OUTPUT_ROOT:-${REPORT_ROOT_BASE}/${STAMP}}"

mkdir -p "${LOG_DIR}"

if ! mkdir -p "${REPORT_ROOT}" 2>/dev/null; then
  FALLBACK_ROOT="artifacts/reports/overnight_runs/${STAMP}"
  echo "Requested output root is not writable: ${REPORT_ROOT}" >&2
  echo "Falling back to ${FALLBACK_ROOT}" >&2
  REPORT_ROOT="${FALLBACK_ROOT}"
  mkdir -p "${REPORT_ROOT}"
fi

LOG_PATH="${LOG_DIR}/overnight_batched_study_${STAMP}.log"

BACKENDS="${NUDEMO_STUDY_BACKENDS:-minio-postgres redis lance parquet webdataset}"
PROVIDER="${NUDEMO_STUDY_PROVIDER:-real}"
BATCH_SIZE="${NUDEMO_STUDY_BATCH_SIZE:-32}"
RANDOM_SAMPLE_COUNT="${NUDEMO_STUDY_RANDOM_SAMPLE_COUNT:-256}"
SNAPSHOT_EVERY_BATCHES="${NUDEMO_SNAPSHOT_EVERY_BATCHES:-1}"
KEEP_BACKEND="${NUDEMO_KEEP_BACKEND:-minio-postgres}"
PURGE_AFTER_BACKEND="${NUDEMO_PURGE_AFTER_BACKEND:-1}"
LIMIT="${NUDEMO_STUDY_LIMIT:-}"
SCENE_LIMIT="${NUDEMO_STUDY_SCENE_LIMIT:-}"

echo "Starting overnight batched study at $(date --iso-8601=seconds)" | tee -a "${LOG_PATH}"
echo "Output root: ${REPORT_ROOT}" | tee -a "${LOG_PATH}"
echo "Backends: ${BACKENDS}" | tee -a "${LOG_PATH}"
echo "Batch size: ${BATCH_SIZE}" | tee -a "${LOG_PATH}"
echo "Dataset version: ${NUDEMO_DATASET_VERSION:-default}" | tee -a "${LOG_PATH}"

ARGS=(
  "--provider" "${PROVIDER}"
  "--batch-size" "${BATCH_SIZE}"
  "--random-sample-count" "${RANDOM_SAMPLE_COUNT}"
  "--snapshot-every-batches" "${SNAPSHOT_EVERY_BATCHES}"
  "--keep-backend" "${KEEP_BACKEND}"
  "--output-root" "${REPORT_ROOT}"
  "--backends"
)

for backend in ${BACKENDS}; do
  ARGS+=("${backend}")
done

if [[ -n "${LIMIT}" ]]; then
  ARGS+=("--limit" "${LIMIT}")
fi

if [[ -n "${SCENE_LIMIT}" ]]; then
  ARGS+=("--scene-limit" "${SCENE_LIMIT}")
fi

if [[ "${PURGE_AFTER_BACKEND}" == "1" ]]; then
  ARGS+=("--purge-after-backend")
fi

uv run --python 3.12 python scripts/overnight_batched_study.py "${ARGS[@]}" "$@" 2>&1 | tee -a "${LOG_PATH}"

echo "Finished overnight batched study at $(date --iso-8601=seconds)" | tee -a "${LOG_PATH}"
