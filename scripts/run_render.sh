#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_render.sh sample
#   bash scripts/run_render.sh scene
#
# Tunables:
#   PROVIDER=real
#   SAMPLE_IDX=0
#   SCENE_NAME=scene-0001
#   CAMERA=CAM_FRONT
#   MAX_FRAMES=24
#   FRAME_STEP=1
#   FPS=2
#   OUTPUT=artifacts/reports/renders/out.gif

source "$(dirname "$0")/lib/nudemo_common.sh"

SUBCOMMAND="${1:-sample}"
shift || true

case "${SUBCOMMAND}" in
  sample)
    : "${PROVIDER:=real}"
    : "${SAMPLE_IDX:=0}"
    cmd=(render sample --provider "${PROVIDER}" --sample-idx "${SAMPLE_IDX}")
    append_option_if_set cmd --output "${OUTPUT:-}"
    ;;
  scene)
    : "${CAMERA:=CAM_FRONT}"
    : "${MAX_FRAMES:=24}"
    : "${FRAME_STEP:=1}"
    : "${FPS:=2}"
    cmd=(render scene --camera "${CAMERA}" --max-frames "${MAX_FRAMES}" --step "${FRAME_STEP}" --fps "${FPS}")
    append_option_if_set cmd --scene-name "${SCENE_NAME:-}"
    append_option_if_set cmd --output "${OUTPUT:-}"
    ;;
  *)
    echo "usage: $0 sample|scene [extra nudemo args...]" >&2
    exit 1
    ;;
esac

run_nudemo "${cmd[@]}" "$@"
