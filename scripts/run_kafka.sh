#!/usr/bin/env bash
set -euo pipefail

# Tunables:
#   PROVIDER=real
#   MODE=metadata-only|full-payload
#   LIMIT=256
#   SCENE_LIMIT=20
#   CREATE_TOPICS=1

source "$(dirname "$0")/lib/nudemo_common.sh"

: "${PROVIDER:=real}"
: "${MODE:=metadata-only}"

cmd=(kafka --provider "${PROVIDER}" --mode "${MODE}")
append_option_if_set cmd --limit "${LIMIT:-}"
append_option_if_set cmd --scene-limit "${SCENE_LIMIT:-}"
append_flag_if_true cmd --create-topics "${CREATE_TOPICS:-0}"

run_nudemo "${cmd[@]}" "$@"
