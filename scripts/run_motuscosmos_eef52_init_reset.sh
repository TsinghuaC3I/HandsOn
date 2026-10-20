#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 TASK OUTPUT_JSON [SEED] [GPU]" >&2
  exit 2
fi

task=$1
output_json=$2
seed=${3:-3407}
gpu=${4:-0}

repo=/mnt/pfs/wuhongjin/repos/motuscosmos-handson-d905-runtime
isaac_python=/mnt/pfs/luotianwei/envs/isaacsim-5.1-handson/bin/python
compat_libs=/mnt/pfs/luotianwei/isaacsim_runtime_compat/ubuntu2204-driver580
gl_compat_libs=/mnt/pfs/luotianwei/isaacsim_runtime_compat/glvnd
host_tag=$(hostname -s)
runtime_dir="/tmp/wuhongjin-motuscosmos-phase4c-${task}-gpu${gpu}-$$"
cache_root=/mnt/pfs/wuhongjin/cache/motuscosmos-handson/phase4c

mkdir -p "$(dirname "$output_json")" "$runtime_dir" \
  "$cache_root/isaaclab/$task" "$cache_root/xdg/$host_tag-gpu$gpu"
chmod 700 "$runtime_dir"

cd "$repo"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
unset http_proxy https_proxy all_proxy no_proxy
export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu"
export HANDSON_SIM_BACKEND=isaacsim
export HANDSON_ARM_TYPE=rokae
export HANDSON_HAND_TARGET_CONTROL=1
export HANDSON_WUJI_HAND_ACTUATOR=1
export HANDSON_ROKAE_WUJI_ARM_ACTUATOR=1
export HANDSON_ROKAE_WUJI_TABLETOP_RESET=1
export HANDSON_STABLE_RECORDING_ORIGIN=1
export HANDSON_ISAAC_IN_PROCESS_HARD_RESET=1
export HANDSON_ISAACLAB_DEVICE=cuda:0
export HANDSON_ISAACLAB_CONTROL_HZ=30
export HANDSON_ISAACLAB_CONTROL_DT=0.03333333333333333
export HANDSON_ISAACLAB_ARM_IK_BACKEND=dls
export HANDSON_ISAACLAB_ARM_TARGET_ALPHA=1
export HANDSON_ISAACLAB_ARM_MAX_SITE_STEP=0
export HANDSON_ISAACLAB_ARM_TARGET_JUMP_DELTA=0
export HANDSON_ISAACLAB_ARM_TARGET_POS_DEADBAND=0
export HANDSON_ISAACLAB_ARM_TARGET_ROT_JUMP_DEG=0
export HANDSON_ISAACLAB_ARM_TARGET_ROT_DEADBAND_DEG=0
export HANDSON_ISAACLAB_ARM_TARGET_JUMP_LIMIT=0
export HANDSON_ISAACLAB_ARM_MAX_JOINT_STEP=0
export HANDSON_ISAACLAB_DEFER_ARM_RENDER=0
export HANDSON_ISAACLAB_CACHE="$cache_root/isaaclab/$task"
export HANDSON_ISAACSIM_GLOBAL_INIT_LOCK="$cache_root/.phase4c-init.lock"
export HANDSON_ISAACSIM_PORTABLE_ROOT="$runtime_dir/portable"
export HANDSON_ISAACSIM_RENDERER_GPU=0
export XDG_CACHE_HOME="$cache_root/xdg/$host_tag-gpu$gpu"
export XDG_RUNTIME_DIR="$runtime_dir/xdg-runtime"
export PYTHONPYCACHEPREFIX="$runtime_dir/pycache"
export LD_LIBRARY_PATH="$gl_compat_libs:$compat_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo/handson"
mkdir -p "$HANDSON_ISAACSIM_PORTABLE_ROOT" "$XDG_RUNTIME_DIR" \
  "$PYTHONPYCACHEPREFIX"

exec "$isaac_python" scripts/smoke_motuscosmos_eef52_init_reset.py \
  --task="$task" --seed="$seed" --output="$output_json"
