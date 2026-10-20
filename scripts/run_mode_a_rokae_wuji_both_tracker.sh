#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANUS_ROOT="${MANUS_WUJI_ROOT:-${REPO_ROOT}/teleoperation/manus/vendor/wuji-hand-teleop}"

export HANDSON_ROOT="${HANDSON_ROOT:-${REPO_ROOT}}"
export HANDSON_PYTHON="${HANDSON_PYTHON:-/home/user/.micromamba/envs/handson/bin/python}"
export ROS2_WS="${ROS2_WS:-${MANUS_ROOT}}"
export MANUS_WUJI_ROOT="${MANUS_WUJI_ROOT:-${MANUS_ROOT}}"
export WUJI_RETARGETING_ROOT="${WUJI_RETARGETING_ROOT:-${REPO_ROOT}/teleoperation/manus/vendor/wuji-retargeting}"
export HANDSON_MANUS_MAPPING_CONFIG="${HANDSON_MANUS_MAPPING_CONFIG:-${REPO_ROOT}/teleoperation/manus/config/handson_manus_mapping.yaml}"

cd "${REPO_ROOT}"
exec scripts/mode_a_handson.sh start both tracker relative "$@"
