#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

EXPLORER_HOST="${NUDEMO_EXPLORER_HOST:-127.0.0.1}"
EXPLORER_PORT="${NUDEMO_EXPLORER_PORT:-8788}"
EXPLORER_LIMIT="${NUDEMO_EXPLORER_LIMIT:-200}"
export NUDEMO_METRICS_ENABLED="${NUDEMO_METRICS_ENABLED:-0}"

echo "Serving explorer at http://$EXPLORER_HOST:$EXPLORER_PORT/"
exec uv run --python 3.12 nudemo explore --host "$EXPLORER_HOST" --port "$EXPLORER_PORT" --limit "$EXPLORER_LIMIT"
