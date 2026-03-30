#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_tasks.sh list
#   bash scripts/run_tasks.sh create --title "review"
#   bash scripts/run_tasks.sh claim task_id --actor alice

source "$(dirname "$0")/lib/nudemo_common.sh"

SUBCOMMAND="${1:-list}"
shift || true

cmd=(tasks "${SUBCOMMAND}")

if [[ "${SUBCOMMAND}" == "list" ]]; then
  append_option_if_set cmd --status "${STATUS:-}"
  append_option_if_set cmd --source-type "${SOURCE_TYPE:-}"
  append_option_if_set cmd --source-id "${SOURCE_ID:-}"
  append_option_if_set cmd --limit "${LIMIT:-50}"
fi

run_nudemo "${cmd[@]}" "$@"
