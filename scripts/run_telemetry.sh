#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_telemetry.sh runs
#   bash scripts/run_telemetry.sh dashboard

source "$(dirname "$0")/lib/nudemo_common.sh"

SUBCOMMAND="${1:-runs}"
shift || true

case "${SUBCOMMAND}" in
  runs)
    cmd=(telemetry runs --limit "${LIMIT:-10}")
    ;;
  dashboard)
    cmd=(telemetry dashboard)
    append_option_if_set cmd --run-id "${RUN_ID:-}"
    if [[ -z "${RUN_ID:-}" ]]; then
      cmd+=(--latest)
    fi
    ;;
  *)
    echo "usage: $0 runs|dashboard [extra nudemo args...]" >&2
    exit 1
    ;;
esac

run_nudemo "${cmd[@]}" "$@"
