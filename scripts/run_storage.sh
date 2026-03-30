#!/usr/bin/env bash
set -euo pipefail

# Tunables:
#   BACKEND=minio-postgres|redis|lance|parquet|webdataset
#   PROVIDER=real
#   LIMIT=1024
#   SCENE_LIMIT=32

source "$(dirname "$0")/lib/nudemo_common.sh"

: "${BACKEND:=lance}"
: "${PROVIDER:=real}"

cmd=(storage "${BACKEND}" --provider "${PROVIDER}")
append_option_if_set cmd --limit "${LIMIT:-}"
append_option_if_set cmd --scene-limit "${SCENE_LIMIT:-}"

run_nudemo "${cmd[@]}" "$@"
