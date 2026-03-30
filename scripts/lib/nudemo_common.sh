#!/usr/bin/env bash

nudemo_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

nudemo_cd_root() {
  cd "$(nudemo_repo_root)"
}

nudemo_apply_runtime_env() {
  export UV="${UV:-uv}"
  export PYTHON="${PYTHON:-3.12}"

  if [[ -n "${DATASET_VERSION:-}" ]]; then
    export NUDEMO_DATASET_VERSION="${DATASET_VERSION}"
  fi

  if [[ -n "${DATASET_ROOT:-}" ]]; then
    export NUDEMO_DATASET_ROOT="${DATASET_ROOT}"
  fi
}

nudemo_base_args() {
  local -n out=$1
  if [[ -n "${CONFIG:-}" ]]; then
    out+=(--config "${CONFIG}")
  fi
}

append_option_if_set() {
  local -n out=$1
  local flag=$2
  local value=${3:-}
  if [[ -n "${value}" ]]; then
    out+=("${flag}" "${value}")
  fi
}

append_flag_if_true() {
  local -n out=$1
  local flag=$2
  local value=${3:-}
  case "${value}" in
    1|true|TRUE|yes|YES|on|ON)
      out+=("${flag}")
      ;;
  esac
}

append_word_list_if_set() {
  local -n out=$1
  local flag=$2
  local value=${3:-}
  local words=()
  if [[ -n "${value}" ]]; then
    read -r -a words <<< "${value}"
    if [[ ${#words[@]} -gt 0 ]]; then
      out+=("${flag}" "${words[@]}")
    fi
  fi
}

run_nudemo() {
  nudemo_cd_root
  nudemo_apply_runtime_env
  local args=()
  nudemo_base_args args
  args+=("$@")
  "${UV}" run --python "${PYTHON}" nudemo "${args[@]}"
}
