#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

required=(uv)
optional=(docker python3.12)

for cmd in "${required[@]}"; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "missing required command: $cmd" >&2
    exit 1
  fi
done

for cmd in "${optional[@]}"; do
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "$cmd: $( "$cmd" --version 2>/dev/null || true )"
  else
    echo "$cmd: missing"
  fi
done

python3 - <<'PY'
from pathlib import Path
import tomllib

data = tomllib.loads(Path("pyproject.toml").read_text())
assert data["project"]["requires-python"] == ">=3.12,<3.13"
optional = data["project"]["optional-dependencies"]
for group in ("dataset", "pipeline", "reporting", "dev"):
    assert group in optional
print("pyproject: ok")
PY

if command -v docker >/dev/null 2>&1; then
  docker compose -f config/docker-compose.yml config >/dev/null
  echo "compose: ok"
fi
