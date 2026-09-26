#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.10 or newer is required. Install Python and rerun this script." >&2
  exit 1
fi

python_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  echo "Python 3.10 or newer is required. Found $python_version." >&2
  exit 1
fi

if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "The Python venv module is missing. On Ubuntu, install it with: sudo apt-get install python3-venv" >&2
  exit 1
fi

venv_dir="$repo_root/.venv"
python3 -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install -r "$repo_root/requirements.txt"
echo "Dependencies are ready. Start the app with: bash scripts/run-linux-cpu.sh"