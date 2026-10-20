#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_SH="${ISAACLAB_SH:-/home/user/IsaacLab/isaaclab.sh}"
ISAACSIM_ENV_BIN="${ISAACSIM_ENV_BIN:-/home/user/miniconda3/envs/isaacsim/bin}"

export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM="xterm-256color"
fi
export PATH="${ISAACSIM_ENV_BIN}:${PATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

export HANDSON_SIM_BACKEND="${HANDSON_SIM_BACKEND:-isaacsim}"
export HANDSON_AUTO_START_TELEOP="${HANDSON_AUTO_START_TELEOP:-1}"
export HANDSON_E01_GARMENTLAB_PRESET="${HANDSON_E01_GARMENTLAB_PRESET:-2}"
export HANDSON_E01_REST_SHAPE_MODE="${HANDSON_E01_REST_SHAPE_MODE:-tabletop_flat}"

cd "${REPO_ROOT}"
exec "${ISAACLAB_SH}" -p scripts/record_demos_zarr.py \
  --exp_name=e01_fold_garment \
  --sim_backend=isaacsim \
  --preview_only=true \
  --render_mode=human \
  --show_sim_cameras=true \
  --arm_type=rokae \
  --hand_type=wuji1 \
  --randomize=false \
  "$@"
