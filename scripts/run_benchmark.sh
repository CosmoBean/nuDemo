#!/usr/bin/env bash
set -euo pipefail

# Copy this file into scripts/custom/ and tune the defaults for your machine.
# Common tunables:
#   PROVIDER=real
#   LIMIT=4096
#   SCENE_LIMIT=200
#   BACKENDS="minio-postgres redis lance parquet webdataset"
#   NUM_RUNS=1
#   RANDOM_SAMPLE_COUNT=128
#   BATCH_SIZE=32
#   NUM_WORKERS="0 2 4"

source "$(dirname "$0")/lib/nudemo_common.sh"

MODE_NAME="${1:-real}"
if [[ "${MODE_NAME}" == "real" || "${MODE_NAME}" == "sim" || "${MODE_NAME}" == "synthetic" ]]; then
  shift
else
  MODE_NAME="${BENCHMARK_MODE:-real}"
fi

: "${PROVIDER:=real}"
: "${NUM_RUNS:=1}"
: "${RANDOM_SAMPLE_COUNT:=10}"
: "${BATCH_SIZE:=4}"
: "${NUM_WORKERS:=0 2 4}"

cmd=(benchmark run)
if [[ "${MODE_NAME}" == "sim" || "${MODE_NAME}" == "synthetic" ]]; then
  cmd+=(--simulate --provider synthetic)
else
  cmd+=(--provider "${PROVIDER}")
fi

append_option_if_set cmd --limit "${LIMIT:-}"
append_option_if_set cmd --scene-limit "${SCENE_LIMIT:-}"
append_word_list_if_set cmd --backends "${BACKENDS:-}"
append_option_if_set cmd --num-runs "${NUM_RUNS}"
append_option_if_set cmd --random-sample-count "${RANDOM_SAMPLE_COUNT}"
append_option_if_set cmd --batch-size "${BATCH_SIZE}"
append_word_list_if_set cmd --num-workers "${NUM_WORKERS}"

run_nudemo "${cmd[@]}" "$@"
