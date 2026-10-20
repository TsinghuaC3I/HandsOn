#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY="${HANDSON_NEWTON_GPU_TO_GPU_OVERLAY:-$ROOT/../handson-local-state/runtime/newton-gpu-to-gpu-v1}"

if [[ -n "${HANDSON_ISAACSIM_NEWTON_PYTHON:-}" ]]; then
  NEWTON_PYTHON="$HANDSON_ISAACSIM_NEWTON_PYTHON"
elif [[ -x "$ROOT/../handson-local-state/runtime/.handson_isaacsim/toolchain-6.0.1/env/bin/python" ]]; then
  NEWTON_PYTHON="$ROOT/../handson-local-state/runtime/.handson_isaacsim/toolchain-6.0.1/env/bin/python"
elif [[ -x "$ROOT/.handson_isaacsim/toolchain-6.0.1/env/bin/python" ]]; then
  NEWTON_PYTHON="$ROOT/.handson_isaacsim/toolchain-6.0.1/env/bin/python"
else
  echo "Isaac Sim 6 Newton Python was not found." >&2
  exit 1
fi

for package in warp newton mujoco_warp; do
  if [[ ! -f "$OVERLAY/$package/__init__.py" ]]; then
    echo "Newton GPU_TO_GPU overlay is incomplete: $OVERLAY" >&2
    echo "Run ./scripts/install_newton_gpu_to_gpu_overlay.sh first." >&2
    exit 1
  fi
done

export HANDSON_NEWTON_GPU_TO_GPU_OVERLAY="$OVERLAY"
export PYTHONPATH="$OVERLAY:$ROOT/handson${PYTHONPATH:+:$PYTHONPATH}"
exec "$NEWTON_PYTHON" "$@"
