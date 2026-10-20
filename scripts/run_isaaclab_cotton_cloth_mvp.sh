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

cd "${REPO_ROOT}"
exec "${ISAACLAB_SH}" -p scripts/preview_isaaclab_cotton_cloth_mvp.py "$@"
