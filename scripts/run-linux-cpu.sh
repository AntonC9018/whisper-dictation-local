#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="$repo_root/.venv/bin/python"
if [[ ! -x "$python" ]]; then
  echo "Run bash scripts/install-linux-cpu.sh first." >&2
  exit 1
fi
exec "$python" "$repo_root/server.py" "$@"