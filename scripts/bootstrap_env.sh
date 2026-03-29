#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv python install 3.12
uv venv --python 3.12
uv sync --extra dataset --extra pipeline --extra reporting --extra search --extra dev

echo "Environment ready at .venv"
