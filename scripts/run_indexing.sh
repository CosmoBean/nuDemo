#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_indexing.sh multimodal
#   bash scripts/run_indexing.sh track
#   bash scripts/run_indexing.sh track-search
#
# Tunables:
#   LIMIT=4096
#   SCENE_LIMIT=200
#   BATCH_SIZE=250
#   Q=pedestrian
#   SOURCE=elasticsearch|postgres
#   COHORT_ID=...
#   TASK_ID=...

source "$(dirname "$0")/lib/nudemo_common.sh"

SUBCOMMAND="${1:-multimodal}"
shift || true

case "${SUBCOMMAND}" in
  multimodal)
    : "${BATCH_SIZE:=24}"
    cmd=(multimodal-index --batch-size "${BATCH_SIZE}")
    append_option_if_set cmd --limit "${LIMIT:-}"
    append_option_if_set cmd --scene-limit "${SCENE_LIMIT:-}"
    append_flag_if_true cmd --append "${APPEND:-0}"
    ;;
  track)
    : "${BATCH_SIZE:=250}"
    cmd=(track-index --batch-size "${BATCH_SIZE}")
    append_option_if_set cmd --limit "${LIMIT:-}"
    append_option_if_set cmd --scene-limit "${SCENE_LIMIT:-}"
    append_flag_if_true cmd --append "${APPEND:-0}"
    append_flag_if_true cmd --materialize-only "${MATERIALIZE_ONLY:-0}"
    append_flag_if_true cmd --index-only "${INDEX_ONLY:-0}"
    ;;
  track-search)
    : "${SOURCE:=elasticsearch}"
    cmd=(track-search --source "${SOURCE}" --limit "${LIMIT:-20}")
    append_option_if_set cmd --q "${Q:-}"
    append_option_if_set cmd --scene-token "${SCENE_TOKEN:-}"
    append_option_if_set cmd --location "${LOCATION:-}"
    append_option_if_set cmd --category "${CATEGORY:-}"
    append_option_if_set cmd --offset "${OFFSET:-}"
    ;;
  export-cohort)
    if [[ -z "${COHORT_ID:-}" ]]; then
      echo "COHORT_ID is required for export-cohort" >&2
      exit 1
    fi
    cmd=(export-cohort "${COHORT_ID}")
    append_option_if_set cmd --task-id "${TASK_ID:-}"
    ;;
  *)
    echo "usage: $0 multimodal|track|track-search|export-cohort [extra nudemo args...]" >&2
    exit 1
    ;;
esac

run_nudemo "${cmd[@]}" "$@"
