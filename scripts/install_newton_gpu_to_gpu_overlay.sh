#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY="${HANDSON_NEWTON_GPU_TO_GPU_OVERLAY:-$ROOT/../handson-local-state/runtime/newton-gpu-to-gpu-v1}"
QD_PROJECTION_PATCH="$ROOT/scripts/patches/newton_gpu_to_gpu_v1/b03_qd_projection.patch"

if ! command -v patch >/dev/null 2>&1; then
  echo "Install the system 'patch' utility before installing the Newton overlay." >&2
  exit 1
fi

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

mkdir -p "$OVERLAY"
"$NEWTON_PYTHON" -m pip install \
  --upgrade \
  --no-deps \
  --target "$OVERLAY" \
  "warp-lang==1.15.0" \
  "newton==1.4.0" \
  "mujoco==3.10.0" \
  "mujoco-warp==3.10.0.3"

if [[ ! -f "$QD_PROJECTION_PATCH" ]]; then
  echo "Missing Newton velocity-projection patch: $QD_PROJECTION_PATCH" >&2
  exit 1
fi
if grep -Fq "HANDSON_B03_NEWTON_QD_PROJECTION_V1" \
    "$OVERLAY/mujoco_warp/_src/forward.py" \
  && grep -Fq "HANDSON_B03_NEWTON_QD_PROJECTION_V1" \
    "$OVERLAY/newton/_src/solvers/mujoco/solver_mujoco.py"; then
  echo "Newton velocity-projection patch is already installed."
else
  patch --batch --forward -p1 -d "$OVERLAY" <"$QD_PROJECTION_PATCH"
fi

for patched_source in \
  "$OVERLAY/mujoco_warp/_src/forward.py" \
  "$OVERLAY/newton/_src/solvers/mujoco/solver_mujoco.py"; do
  if ! grep -Fq "HANDSON_B03_NEWTON_QD_PROJECTION_V1" "$patched_source"; then
    echo "Newton velocity-projection patch did not update: $patched_source" >&2
    exit 1
  fi
done

PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}" "$NEWTON_PYTHON" - <<'PY'
import importlib.metadata
import warp as wp

expected = {
    "isaaclab-newton": "0.13.6",
    "warp-lang": "1.15.0",
    "newton": "1.4.0",
    "newton-actuators": "0.1.0",
    "mujoco": "3.10.0",
    "mujoco-warp": "3.10.0.3",
}
actual = {
    name: importlib.metadata.version(name)
    for name in expected
}
if actual != expected:
    raise SystemExit(f"Newton GPU_TO_GPU overlay mismatch: {actual!r}")
if not hasattr(wp.DeterministicMode, "GPU_TO_GPU"):
    raise SystemExit("Warp does not expose DeterministicMode.GPU_TO_GPU")
print(f"Newton GPU_TO_GPU overlay ready: {actual}")
PY
