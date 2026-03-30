#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib/nudemo_common.sh"

nudemo_cd_root
nudemo_apply_runtime_env

TARGET="${1:-src/nudemo}"
shift || true

"${UV}" run --python "${PYTHON}" --extra dev pylint "${TARGET}" "$@"
