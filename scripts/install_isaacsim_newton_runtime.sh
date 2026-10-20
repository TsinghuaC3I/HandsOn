#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${HANDSON_LOCAL_RUNTIME_ROOT:-$ROOT/../handson-local-state/runtime}"
TOOLCHAIN_ROOT="${HANDSON_ISAACSIM_NEWTON_TOOLCHAIN_ROOT:-$RUNTIME_ROOT/.handson_isaacsim/toolchain-6.0.1}"
PYTHON_ENV="$TOOLCHAIN_ROOT/python-env"
PUBLIC_ENV="$TOOLCHAIN_ROOT/env"
ISAACLAB_SOURCE="$TOOLCHAIN_ROOT/IsaacLab-newton-rtx"
OVERLAY="${HANDSON_NEWTON_GPU_TO_GPU_OVERLAY:-$RUNTIME_ROOT/newton-gpu-to-gpu-v1}"
RECORDING_REQUIREMENTS="$ROOT/scripts/requirements-isaacsim-recording.txt"
READY_MARKER="$TOOLCHAIN_ROOT/.handson-newton-runtime-ready-v2"
LEGACY_READY_MARKER="$TOOLCHAIN_ROOT/.handson-newton-runtime-ready-v1"
LOCK_FILE="$TOOLCHAIN_ROOT/.install.lock"
STATUS_FILE="$TOOLCHAIN_ROOT/install-status.json"

# Collection workers intentionally disable user-site packages.  Apply the same
# isolation while installing and qualifying the runtime so ~/.local cannot hide
# a missing dependency in the published toolchain.
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

ISAACLAB_REPOSITORY="https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_COMMIT="ffff603eafc6b74264a5261cc0183d6a65390d78"
ISAACSIM_VERSION="6.0.1.0"
TORCH_VERSION="2.10.0"

mkdir -p "$TOOLCHAIN_ROOT"
exec 9>"$LOCK_FILE"
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
  echo "HandsOn Newton runtime installation is already running."
  exit 0
fi

write_status() {
  local status="$1"
  local detail="$2"
  local next_status="$STATUS_FILE.next.$$"
  printf '{"status":"%s","detail":"%s","updated_at":"%s"}\n' \
    "$status" "$detail" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$next_status"
  mv -f "$next_status" "$STATUS_FILE"
}

on_exit() {
  local status=$?
  trap - EXIT
  if [[ "$status" -ne 0 ]]; then
    write_status "failed" "installer_exit_$status"
  fi
  exit "$status"
}
trap on_exit EXIT
write_status "installing" "preparing_newton_runtime"

verify_runtime() {
  [[ -x "$PUBLIC_ENV/bin/python" ]] || return 1
  PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}" \
    "$PUBLIC_ENV/bin/python" - <<'PY' >/dev/null
import importlib
import importlib.metadata
from pathlib import Path
import site
import sys

import warp as wp

expected = {
    "isaacsim": "6.0.1.0",
    "isaaclab": "6.1.14",
    "isaaclab-newton": "0.13.6",
    "warp-lang": "1.15.0",
    "newton": "1.4.0",
    "newton-actuators": "0.1.0",
    "mujoco": "3.10.0",
    "mujoco-warp": "3.10.0.3",
    "absl-py": "2.4.0",
    "glfw": "2.10.0",
    "zarr": "2.18.3",
    "numcodecs": "0.15.1",
    "av": "17.1.0",
}
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"Newton runtime mismatch: {actual!r}")
if site.ENABLE_USER_SITE:
    raise SystemExit("Newton runtime verification must disable user-site packages")
prefix = Path(sys.prefix).resolve()
for module_name in ("absl", "glfw", "zarr", "numcodecs", "av"):
    module = importlib.import_module(module_name)
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(prefix):
        raise SystemExit(
            f"{module_name} resolved outside the Newton toolchain: {module_path}"
        )
if not hasattr(wp.DeterministicMode, "GPU_TO_GPU"):
    raise SystemExit("Warp does not expose DeterministicMode.GPU_TO_GPU")
PY
}

install_recording_requirements() {
  local python="$1"

  # Install the five exact direct wheels without consulting installed metadata:
  # pip can otherwise report a ~/.local distribution as satisfied even though
  # collection workers run with PYTHONNOUSERSITE=1.  --no-deps prevents NumPy
  # or any unrelated Isaac package from being reinstalled.  Resolve support
  # dependencies afterwards, when pip can see the direct metadata inside this
  # toolchain and retain Isaac Sim's already-satisfied NumPy.
  "$python" -m pip install \
    --ignore-installed \
    --no-deps \
    --requirement "$RECORDING_REQUIREMENTS"
  "$python" -m pip install --requirement "$RECORDING_REQUIREMENTS"
}

publish_ready_marker() {
  local next_marker="$READY_MARKER.next.$$"
  : >"$next_marker"
  mv -f "$next_marker" "$READY_MARKER"
}

# A v1 runtime may be fully qualified already, or it may only be missing the
# recorder dependencies added by v2.  Verify first, then repair that narrow
# dependency set before falling back to the full bootstrap.  Publish v2 only
# after the isolated runtime passes every version, import, and origin check.
repair_published_runtime=0
if [[ -f "$READY_MARKER" || -f "$LEGACY_READY_MARKER" ]]; then
  repair_published_runtime=1
fi
if verify_runtime; then
  publish_ready_marker
  write_status "ready" "newton_runtime_already_ready"
  trap - EXIT
  echo "HandsOn Newton runtime is already ready at $PUBLIC_ENV."
  exit 0
fi
rm -f "$READY_MARKER"

if [[ "$repair_published_runtime" == "1" && -x "$PUBLIC_ENV/bin/python" ]]; then
  install_recording_requirements "$PUBLIC_ENV/bin/python"
  if verify_runtime; then
    publish_ready_marker
    write_status "ready" "newton_runtime_recording_dependencies_repaired"
    trap - EXIT
    echo "HandsOn Newton runtime recording dependencies repaired at $PUBLIC_ENV."
    exit 0
  fi
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "HandsOn Newton runtime currently supports only Linux x86_64." >&2
  exit 1
fi

MAMBA="${MAMBA_EXE:-$(command -v micromamba || true)}"
if [[ -z "$MAMBA" ]]; then
  MAMBA_ROOT="$TOOLCHAIN_ROOT/micromamba-bootstrap"
  MAMBA="$MAMBA_ROOT/bin/micromamba"
  if [[ ! -x "$MAMBA" ]]; then
    mkdir -p "$MAMBA_ROOT"
    echo "Downloading Micromamba bootstrap..."
    curl --fail --location --retry 5 --retry-all-errors \
      https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj -C "$MAMBA_ROOT" bin/micromamba
  fi
fi

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$TOOLCHAIN_ROOT/mamba-root}"
if [[ -x "$PYTHON_ENV/bin/python" ]]; then
  "$MAMBA" install -y -p "$PYTHON_ENV" python=3.12 pip git cmake
else
  "$MAMBA" create -y -p "$PYTHON_ENV" python=3.12 pip git cmake
fi

PYTHON="$PYTHON_ENV/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES

"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install \
  "torch==$TORCH_VERSION" \
  --index-url https://download.pytorch.org/whl/cu128
"$PYTHON" -m pip install \
  --pre \
  "isaacsim[all,extscache]==$ISAACSIM_VERSION" \
  --extra-index-url https://pypi.nvidia.com
install_recording_requirements "$PYTHON"

if [[ -e "$ISAACLAB_SOURCE" && ! -d "$ISAACLAB_SOURCE/.git" ]]; then
  mv "$ISAACLAB_SOURCE" "$ISAACLAB_SOURCE.incomplete.$(date +%s)"
fi
if [[ ! -d "$ISAACLAB_SOURCE/.git" ]]; then
  git clone --filter=blob:none --no-checkout \
    "$ISAACLAB_REPOSITORY" "$ISAACLAB_SOURCE"
fi
git -C "$ISAACLAB_SOURCE" fetch --depth 1 origin "$ISAACLAB_COMMIT"
git -C "$ISAACLAB_SOURCE" checkout --detach "$ISAACLAB_COMMIT"

(
  export VIRTUAL_ENV="$PYTHON_ENV"
  export CONDA_PREFIX="$PYTHON_ENV"
  export PATH="$PYTHON_ENV/bin:$PATH"
  cd "$ISAACLAB_SOURCE"
  ./isaaclab.sh -i newton
)

"$PYTHON" -m pip install --upgrade "newton-actuators==0.1.0"
HANDSON_ISAACSIM_NEWTON_PYTHON="$PYTHON" \
HANDSON_NEWTON_GPU_TO_GPU_OVERLAY="$OVERLAY" \
  "$ROOT/scripts/install_newton_gpu_to_gpu_overlay.sh"

PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - <<'PY'
import importlib
import importlib.metadata
from pathlib import Path
import site
import sys

import warp as wp

expected = {
    "isaacsim": "6.0.1.0",
    "isaaclab": "6.1.14",
    "isaaclab-newton": "0.13.6",
    "warp-lang": "1.15.0",
    "newton": "1.4.0",
    "newton-actuators": "0.1.0",
    "mujoco": "3.10.0",
    "mujoco-warp": "3.10.0.3",
    "absl-py": "2.4.0",
    "glfw": "2.10.0",
    "zarr": "2.18.3",
    "numcodecs": "0.15.1",
    "av": "17.1.0",
}
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"Newton runtime mismatch: {actual!r}")
if site.ENABLE_USER_SITE:
    raise SystemExit("Newton runtime verification must disable user-site packages")
prefix = Path(sys.prefix).resolve()
for module_name in ("absl", "glfw", "zarr", "numcodecs", "av"):
    module = importlib.import_module(module_name)
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(prefix):
        raise SystemExit(
            f"{module_name} resolved outside the Newton toolchain: {module_path}"
        )
if not hasattr(wp.DeterministicMode, "GPU_TO_GPU"):
    raise SystemExit("Warp does not expose DeterministicMode.GPU_TO_GPU")
print(f"HandsOn Newton runtime ready: {actual}")
PY

if [[ -e "$PUBLIC_ENV" && ! -L "$PUBLIC_ENV" ]]; then
  mv "$PUBLIC_ENV" "$PUBLIC_ENV.incomplete.$(date +%s)"
fi
next_link="$TOOLCHAIN_ROOT/.env.next.$$"
ln -s "python-env" "$next_link"
mv -Tf "$next_link" "$PUBLIC_ENV"
publish_ready_marker
write_status "ready" "newton_runtime_published"
trap - EXIT
echo "HandsOn Newton runtime published at $PUBLIC_ENV."
