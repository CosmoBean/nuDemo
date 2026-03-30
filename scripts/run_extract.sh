#!/usr/bin/env bash
set -euo pipefail

# Tunables:
#   PROVIDER=real|synthetic|auto
#   LIMIT=128
#   SCENE_LIMIT=12

source "$(dirname "$0")/lib/nudemo_common.sh"

: "${PROVIDER:=real}"

cmd=(extract --provider "${PROVIDER}")
append_option_if_set cmd --limit "${LIMIT:-}"
append_option_if_set cmd --scene-limit "${SCENE_LIMIT:-}"

run_nudemo "${cmd[@]}" "$@"
