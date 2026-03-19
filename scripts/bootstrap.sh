#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not installed." >&2
  exit 1
fi

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "python3.12 not found; asking uv to manage the interpreter." >&2
  uv python install 3.12
fi

uv sync --group pipeline --group training --group dashboard --group dev

if [[ "${START_INFRA:-0}" == "1" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not installed; skipping compose startup." >&2
    exit 0
  fi
  docker compose -f config/docker-compose.yml up -d
fi

