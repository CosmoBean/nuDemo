#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

REPORTS_ROOT="${NUDEMO_REPORTS_ROOT:-$PWD/artifacts/reports}"
REPORTS_HOST="${NUDEMO_REPORTS_HOST:-127.0.0.1}"
REPORTS_PORT="${NUDEMO_REPORTS_PORT:-8787}"

mkdir -p "$REPORTS_ROOT"

echo "Serving reports from $REPORTS_ROOT at http://$REPORTS_HOST:$REPORTS_PORT/"
exec python3 -m http.server "$REPORTS_PORT" --bind "$REPORTS_HOST" --directory "$REPORTS_ROOT"
