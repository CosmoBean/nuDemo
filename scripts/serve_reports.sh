#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

REPORTS_ROOT="${NUDEMO_REPORTS_ROOT:-$PWD/artifacts/reports}"
REPORTS_HOST="${NUDEMO_REPORTS_HOST:-127.0.0.1}"
REPORTS_PORT="${NUDEMO_REPORTS_PORT:-8787}"
PYTHON_BIN="${NUDEMO_BROWSER_PYTHON:-$PWD/.venv/bin/python}"
export NUDEMO_METRICS_ENABLED="${NUDEMO_METRICS_ENABLED:-1}"
export NUDEMO_METRICS_HOST="${NUDEMO_METRICS_HOST:-0.0.0.0}"
export NUDEMO_METRICS_PORT="${NUDEMO_METRICS_PORT:-9464}"

mkdir -p "$REPORTS_ROOT"
python3 ./scripts/render_reports_index.py "$REPORTS_ROOT" >/dev/null

echo "Serving nuDemo browser from $REPORTS_ROOT at http://$REPORTS_HOST:$REPORTS_PORT/"
if [[ -x "$PYTHON_BIN" ]]; then
  exec "$PYTHON_BIN" -m uvicorn nudemo.explorer.app:create_app --factory \
    --app-dir "$PWD/src" \
    --host "$REPORTS_HOST" \
    --port "$REPORTS_PORT"
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run --python 3.12 uvicorn nudemo.explorer.app:create_app --factory \
    --app-dir "$PWD/src" \
    --host "$REPORTS_HOST" \
    --port "$REPORTS_PORT"
fi

PYTHON_BIN="$(command -v python3)"
exec "$PYTHON_BIN" -m uvicorn nudemo.explorer.app:create_app --factory \
  --app-dir "$PWD/src" \
  --host "$REPORTS_HOST" \
  --port "$REPORTS_PORT"
