#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/nudemo benchmark run --provider real --limit 64 "$@"
