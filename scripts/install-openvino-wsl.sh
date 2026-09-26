#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if ! grep -qiE 'microsoft|wsl' /proc/version; then
  echo "Run this installer inside WSL2 Ubuntu." >&2
  exit 1
fi

. /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
  echo "This installer targets Ubuntu 22.04 in WSL2. Found ${PRETTY_NAME:-unknown}." >&2
  exit 1
fi

if [[ "$(dpkg --print-architecture)" != "amd64" ]]; then
  echo "The Intel WSL GPU packages in this script require amd64." >&2
  exit 1
fi

if [[ "$EUID" -eq 0 ]]; then
  sudo_cmd=()
elif command -v sudo >/dev/null 2>&1; then
  sudo_cmd=(sudo)
else
  echo "Run as root or install sudo first." >&2
  exit 1
fi

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT
apt_sources="$work_dir/sources.list"
cat > "$apt_sources" <<'SOURCES'
deb [arch=amd64] http://archive.ubuntu.com/ubuntu jammy main restricted universe multiverse
deb [arch=amd64] http://archive.ubuntu.com/ubuntu jammy-updates main restricted universe multiverse
deb [arch=amd64] http://security.ubuntu.com/ubuntu jammy-security main restricted universe multiverse
SOURCES
apt_options=(-o "Dir::Etc::sourcelist=$apt_sources" -o 'Dir::Etc::sourceparts=-' -o 'APT::Get::List-Cleanup=0')

"${sudo_cmd[@]}" apt-get "${apt_options[@]}" update
"${sudo_cmd[@]}" apt-get "${apt_options[@]}" install -y --no-install-recommends \
  ca-certificates curl gnupg git cmake build-essential \
  python3 python3-venv python3-pip ffmpeg

intel_key="$work_dir/intel-graphics.key"
curl -fsSL https://repositories.intel.com/graphics/intel-graphics.key -o "$intel_key"
"${sudo_cmd[@]}" gpg --dearmor --batch --yes \
  --output /usr/share/keyrings/intel-graphics.gpg "$intel_key"
intel_source='deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc'
printf '%s\n' "$intel_source" > "$work_dir/intel-graphics.list"
"${sudo_cmd[@]}" install -m 0644 "$work_dir/intel-graphics.list" \
  /etc/apt/sources.list.d/whisper-dictation-intel-graphics.list
printf '%s\n' "$intel_source" >> "$apt_sources"
"${sudo_cmd[@]}" apt-get "${apt_options[@]}" update
"${sudo_cmd[@]}" apt-get "${apt_options[@]}" install -y --no-install-recommends \
  intel-opencl-icd intel-level-zero-gpu level-zero

cache_root="${WHISPER_DICTATION_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/whisper-dictation}"
cache_root="$(mkdir -p "$cache_root" && cd "$cache_root" && pwd)"
whisper_cpp_dir="${WHISPER_CPP_DIR:-$cache_root/whisper.cpp}"
whisper_cpp_dir="$(realpath -m "$whisper_cpp_dir")"
repo_real="$(realpath "$repo_root")"
case "$whisper_cpp_dir/" in
  "$repo_real/"*)
    echo "WHISPER_CPP_DIR must be outside the repository so model files stay out of Git." >&2
    exit 1
    ;;
esac

whisper_cpp_commit='d09f61a708f3487afa956ff578e60eae5e7a233c'
if [[ -d "$whisper_cpp_dir/.git" ]]; then
  if [[ -n "$(git -C "$whisper_cpp_dir" status --porcelain)" ]]; then
    echo "The cached whisper.cpp checkout has local changes: $whisper_cpp_dir" >&2
    echo "Move it aside or set WHISPER_CPP_DIR to a clean cache path, then retry." >&2
    exit 1
  fi
  git -C "$whisper_cpp_dir" fetch --depth 1 origin "$whisper_cpp_commit"
  git -C "$whisper_cpp_dir" checkout --detach "$whisper_cpp_commit"
elif [[ -e "$whisper_cpp_dir" ]]; then
  echo "WHISPER_CPP_DIR exists but is not a Git checkout: $whisper_cpp_dir" >&2
  exit 1
else
  mkdir -p "$(dirname "$whisper_cpp_dir")"
  git clone https://github.com/ggml-org/whisper.cpp.git "$whisper_cpp_dir"
  git -C "$whisper_cpp_dir" checkout --detach "$whisper_cpp_commit"
fi

convert_venv="${WHISPER_OPENVINO_VENV:-$cache_root/openvino-convert-venv}"
python3 -m venv "$convert_venv"
convert_python="$convert_venv/bin/python"
"$convert_python" -m pip install --upgrade pip
"$convert_python" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.0+cpu'
"$convert_python" -m pip install -r "$repo_root/openvino/requirements-conversion.txt"

"$convert_python" - <<'PY'
import sys
import openvino as ov

devices = ov.Core().available_devices
print("OpenVINO devices:", ", ".join(devices))
if "GPU" not in devices:
    print(
        "Intel GPU is not visible in WSL. Update the Windows Intel graphics driver, "
        "run 'wsl.exe --update' and 'wsl.exe --shutdown' in PowerShell, then retry.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

models_dir="$whisper_cpp_dir/models"
mkdir -p "$models_dir"
if [[ ! -f "$models_dir/ggml-base.bin" ]]; then
  bash "$models_dir/download-ggml-model.sh" base "$models_dir"
fi
(
  cd "$models_dir"
  "$convert_python" convert-whisper-to-openvino.py --model base
)

openvino_cmake_dir="$($convert_python -c 'import pathlib, openvino; print(pathlib.Path(openvino.__file__).parent / "cmake")')"
cmake -S "$whisper_cpp_dir" -B "$whisper_cpp_dir/build-openvino" \
  -DWHISPER_OPENVINO=ON \
  -DOpenVINO_DIR="$openvino_cmake_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DWHISPER_BUILD_TESTS=OFF \
  -DWHISPER_BUILD_SERVER=OFF
cmake --build "$whisper_cpp_dir/build-openvino" --target whisper-cli --parallel 4

echo "OpenVINO GPU dependencies and the base model are ready."
echo "Run the app with: bash scripts/run-openvino-wsl.sh"