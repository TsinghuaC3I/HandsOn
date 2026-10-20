#!/usr/bin/env bash
set -euo pipefail

ROOT="${HANDSON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

load_env_file() {
  local env_file="$1"
  if [[ -r "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
}

load_env_file "/etc/handson/cloud.env"
load_env_file "$HOME/.config/handson/cloud.env"
if [[ -n "${HANDSON_CLOUD_ENV_FILE:-}" ]]; then
  load_env_file "$HANDSON_CLOUD_ENV_FILE"
fi

ENV_FILE="${HANDSON_COLLECTION_ENV_FILE:-$ROOT/.handson_collection.env}"
load_env_file "$ENV_FILE"

# Collection is locked to the single Isaac Sim 5.1 runtime.  Legacy Newton
# diagnostics can still be launched from their dedicated scripts with an
# explicitly separate environment, but the UI must not route episodes there.
export HANDSON_ENABLE_ISAACSIM_NEWTON="${HANDSON_ENABLE_ISAACSIM_NEWTON:-0}"

# Production collection seals V3 candidates at recorder exit. Cloud upload no
# longer waits for local physical replay; keep legacy manual/night modes off so
# the independent uploader can drain both new and historical pending data.
export HANDSON_POSTPROCESS_MANUAL_ONLY=0
export HANDSON_NIGHT_FORCE_RENDER_UPLOAD=0
export HANDSON_IDLE_MAINTENANCE=1
# Upload every sealed PhysX V3 raw replay package and queue rendering for a
# cloud GPU consumer; never start local physical replay, arm, or camera render.
export HANDSON_CLOUD_RENDER_ONLY=1
export HANDSON_REVIEW_AUTO_UPLOAD=0
# Upload is a schedule-independent 60-second scan. It selects immutable PhysX
# V3 episodes from normal and physical-pending roots, including earlier failed
# data, and may run all day while collection continues.
export HANDSON_BACKGROUND_V3_UPLOAD=1

# The former 8092 sitecustomize profile is now part of HandsOn itself.  Ignore
# inherited switcher flags so controller filters are not applied twice.
unset HANDSON_RUNTIME_SWITCHER
unset HANDSON_TELEOP_PROFILE_PATH

clean_pythonpath=""
if [[ -n "${PYTHONPATH:-}" ]]; then
  IFS=':' read -r -a pythonpath_entries <<< "$PYTHONPATH"
  for entry in "${pythonpath_entries[@]}"; do
    if [[ "$entry" == "$HOME/.handson_runtime_switcher"* ]]; then
      continue
    fi
    clean_pythonpath="${clean_pythonpath:+$clean_pythonpath:}$entry"
  done
fi

HOST="${HANDSON_COLLECTION_HOST:-127.0.0.1}"
PORT="${HANDSON_COLLECTION_PORT:-8090}"
OUT_DIR="${HANDSON_COLLECTION_OUT_DIR:-$HOME/handson_demos}"
ADJACENT_RUNTIME_ROOT="$(cd "$ROOT/../handson" 2>/dev/null && pwd || true)"

if [[ -n "${HANDSON_PYTHON:-}" ]]; then
  PY="$HANDSON_PYTHON"
elif [[ -x "$HOME/miniconda3/envs/act/bin/python" ]]; then
  PY="$HOME/miniconda3/envs/act/bin/python"
elif [[ -x "$HOME/anaconda3/envs/act/bin/python" ]]; then
  PY="$HOME/anaconda3/envs/act/bin/python"
elif [[ -x "$HOME/micromamba/envs/handson/bin/python" ]]; then
  PY="$HOME/micromamba/envs/handson/bin/python"
elif [[ -x "$HOME/.micromamba/envs/handson/bin/python" ]]; then
  PY="$HOME/.micromamba/envs/handson/bin/python"
elif [[ -x "$HOME/miniconda3/envs/handson/bin/python" ]]; then
  PY="$HOME/miniconda3/envs/handson/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "No Python interpreter found. Set HANDSON_PYTHON." >&2
  exit 1
fi

if [[ -n "${HANDSON_ISAACSIM_PYTHON:-}" ]]; then
  ISAACSIM_PY="$HANDSON_ISAACSIM_PYTHON"
elif [[ -n "$ADJACENT_RUNTIME_ROOT" && -x "$ADJACENT_RUNTIME_ROOT/.handson_isaacsim/toolchain/env/bin/python" ]]; then
  ISAACSIM_PY="$ADJACENT_RUNTIME_ROOT/.handson_isaacsim/toolchain/env/bin/python"
elif [[ -x "$ROOT/../.handson_isaacsim/toolchain/env/bin/python" ]]; then
  ISAACSIM_PY="$(cd "$ROOT/../.handson_isaacsim/toolchain/env/bin" && pwd)/python"
elif [[ -x "$ROOT/.handson_isaacsim/toolchain/env/bin/python" ]]; then
  ISAACSIM_PY="$ROOT/.handson_isaacsim/toolchain/env/bin/python"
elif [[ -x "$HOME/miniconda3/envs/isaacsim/bin/python" ]]; then
  ISAACSIM_PY="$HOME/miniconda3/envs/isaacsim/bin/python"
elif [[ -x "$HOME/anaconda3/envs/isaacsim/bin/python" ]]; then
  ISAACSIM_PY="$HOME/anaconda3/envs/isaacsim/bin/python"
elif [[ -x "$HOME/.local/share/mamba/envs/isaacsim/bin/python" ]]; then
  ISAACSIM_PY="$HOME/.local/share/mamba/envs/isaacsim/bin/python"
elif [[ -x "$HOME/.micromamba/envs/isaacsim/bin/python" ]]; then
  ISAACSIM_PY="$HOME/.micromamba/envs/isaacsim/bin/python"
else
  ISAACSIM_PY=""
fi

NEWTON_RUNTIME_ROOT="${HANDSON_LOCAL_RUNTIME_ROOT:-$ROOT/../handson-local-state/runtime}"
NEWTON_TOOLCHAIN_ROOT="${HANDSON_ISAACSIM_NEWTON_TOOLCHAIN_ROOT:-$NEWTON_RUNTIME_ROOT/.handson_isaacsim/toolchain-6.0.1}"
NEWTON_EXPECTED_PY="$NEWTON_TOOLCHAIN_ROOT/env/bin/python"
NEWTON_INSTALLER="$ROOT/scripts/install_isaacsim_newton_runtime.sh"
NEWTON_INSTALL_LOG="$NEWTON_RUNTIME_ROOT/newton-runtime-install.log"
NEWTON_READY_MARKER="$NEWTON_TOOLCHAIN_ROOT/.handson-newton-runtime-ready-v2"

if [[ "${HANDSON_ENABLE_ISAACSIM_NEWTON:-1}" =~ ^(0|[Ff][Aa][Ll][Ss][Ee]|[Nn][Oo]|[Oo][Ff][Ff])$ ]]; then
  # Keep the optional Newton/Sim 6 implementation installed but unavailable
  # to this collection UI.  Native PhysX tasks continue through ISAACSIM_PY.
  ISAACSIM_NEWTON_PY=""
elif [[ -x "$NEWTON_EXPECTED_PY" && -f "$NEWTON_READY_MARKER" ]]; then
  ISAACSIM_NEWTON_PY="$NEWTON_EXPECTED_PY"
elif [[ "${HANDSON_ALLOW_EXTERNAL_NEWTON_RUNTIME:-0}" == "1" \
  && -n "${HANDSON_ISAACSIM_NEWTON_PYTHON:-}" \
  && -x "$HANDSON_ISAACSIM_NEWTON_PYTHON" ]]; then
  # External runtimes are intentionally opt-in.  Old per-machine overrides
  # previously made B03 use different Isaac/Newton versions on each computer.
  ISAACSIM_NEWTON_PY="$HANDSON_ISAACSIM_NEWTON_PYTHON"
else
  # Keep the future path in server state so status polling enables Newton as
  # soon as the background installer atomically publishes env/bin/python.
  ISAACSIM_NEWTON_PY="$NEWTON_EXPECTED_PY"
  if [[ "${HANDSON_AUTO_INSTALL_NEWTON:-1}" != "0" && -x "$NEWTON_INSTALLER" ]]; then
    mkdir -p "$NEWTON_RUNTIME_ROOT"
    setsid -f "$NEWTON_INSTALLER" >>"$NEWTON_INSTALL_LOG" 2>&1 < /dev/null
  fi
fi

if [[ -n "${HANDSON_GENESIS_PYTHON:-}" ]]; then
  GENESIS_PY="$HANDSON_GENESIS_PYTHON"
elif [[ -n "$ADJACENT_RUNTIME_ROOT" && -x "$ADJACENT_RUNTIME_ROOT/.handson_genesis/env/bin/python" ]]; then
  GENESIS_PY="$ADJACENT_RUNTIME_ROOT/.handson_genesis/env/bin/python"
elif [[ -x "$ROOT/../.handson_genesis/env/bin/python" ]]; then
  GENESIS_PY="$(cd "$ROOT/../.handson_genesis/env/bin" && pwd)/python"
elif [[ -x "$ROOT/.handson_genesis/env/bin/python" ]]; then
  GENESIS_PY="$ROOT/.handson_genesis/env/bin/python"
else
  GENESIS_PY=""
fi

# libuipc's CUDA backend is linked against CUDA 12.  Record the matching
# wheel library directories here; server.py prepends them only for Genesis
# children, so Isaac Sim's CUDA runtime is left untouched.
if [[ -z "${HANDSON_GENESIS_CUDA12_LIBRARY_PATH:-}" ]]; then
  for nvidia_root in \
    "$HOME/micromamba/envs/handson-dc"/lib/python*/site-packages/nvidia \
    "$HOME/micromamba/envs/handson"/lib/python*/site-packages/nvidia \
    "$HOME/.micromamba/envs/handson-dc"/lib/python*/site-packages/nvidia \
    "$HOME/.micromamba/envs/handson"/lib/python*/site-packages/nvidia; do
    if [[ ! -f "$nvidia_root/cuda_runtime/lib/libcudart.so.12" ]]; then
      continue
    fi
    genesis_cuda12_paths=""
    for component in nvjitlink cuda_runtime cublas cusparse cusolver cuda_nvrtc cufft curand cudnn nccl nvtx; do
      component_lib="$nvidia_root/$component/lib"
      if [[ -d "$component_lib" ]]; then
        genesis_cuda12_paths="${genesis_cuda12_paths:+$genesis_cuda12_paths:}$component_lib"
      fi
    done
    export HANDSON_GENESIS_CUDA12_LIBRARY_PATH="$genesis_cuda12_paths"
    break
  done
fi

export HANDSON_ROKAE_IK_BACKEND="${HANDSON_ROKAE_IK_BACKEND:-haorui_new}"
export HANDSON_GPU_MODE="${HANDSON_GPU_MODE:-auto}"
export HANDSON_ISAACLAB_DEVICE="${HANDSON_ISAACLAB_DEVICE:-cuda:0}"
export HANDSON_ISAACSIM_VIEWPORT_ONLY="${HANDSON_ISAACSIM_VIEWPORT_ONLY:-1}"
export HANDSON_ISAACSIM_TELEMETRY="${HANDSON_ISAACSIM_TELEMETRY:-0}"
export HANDSON_VIDEO_ENCODER="${HANDSON_VIDEO_ENCODER:-auto}"
NEWTON_GPU_TO_GPU_OVERLAY="${HANDSON_NEWTON_GPU_TO_GPU_OVERLAY:-$NEWTON_RUNTIME_ROOT/newton-gpu-to-gpu-v1}"
export HANDSON_NEWTON_GPU_TO_GPU_OVERLAY="$NEWTON_GPU_TO_GPU_OVERLAY"
if [[ -f "$NEWTON_GPU_TO_GPU_OVERLAY/warp/__init__.py" \
  && -f "$NEWTON_GPU_TO_GPU_OVERLAY/newton/__init__.py" \
  && -f "$NEWTON_GPU_TO_GPU_OVERLAY/mujoco_warp/__init__.py" ]]; then
  :
fi
# Keep the overlay out of the Web UI and ordinary MuJoCo processes.  The UI
# injects it only into isaacsim_newton children, whose Python ABI matches the
# compiled MuJoCo-Warp wheel in the overlay.
# Runtime modules live under ``handson/``, while task-development packages
# imported by the production Isaac scene live at the repository root.
export PYTHONPATH="$ROOT:$ROOT/handson${clean_pythonpath:+:$clean_pythonpath}"

cd "$ROOT"
supervisor_lock="${XDG_RUNTIME_DIR:-/tmp}/handson-collection-ui-${PORT}.lock"
exec 9>"$supervisor_lock"
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
  echo "HandsOn collection UI is already supervised on port $PORT."
  exit 0
fi

restart_delay=1
while true; do
  started_at=$SECONDS
  set +e
  # Keep the watchdog in a different process group from the HTTP backend.
  # Collection/device cleanup may terminate the backend group; it must never
  # take the watchdog with it, otherwise the still-open UI becomes a dead page.
  setsid "$PY" "$ROOT/collection_ui/server.py" \
    --host "$HOST" \
    --port "$PORT" \
    --root "$ROOT" \
    --python "$PY" \
    --isaacsim-python "$ISAACSIM_PY" \
    --isaacsim-newton-python "$ISAACSIM_NEWTON_PY" \
    --genesis-python "$GENESIS_PY" \
    --out-dir "$OUT_DIR"
  status=$?
  set -e

  if [[ "$status" -eq 75 ]]; then
    echo "HandsOn collection UI updated; restarting with the published code."
    exec "$ROOT/scripts/run_collection_ui.sh"
  fi
  if (( SECONDS - started_at >= 60 )); then
    restart_delay=1
  fi
  echo "HandsOn collection UI exited unexpectedly with status $status; restarting in ${restart_delay}s." >&2
  sleep "$restart_delay"
  if (( restart_delay < 15 )); then
    restart_delay=$((restart_delay * 2))
  fi
done
