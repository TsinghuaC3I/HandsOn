#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-/mnt/pfs/luotianwei/handson/openpi}"
python="${PYTHON_BIN:-/mnt/pfs/luotianwei/envs/handson-openpi-jax062/bin/python}"
cudnn_lib="${CUDNN_LIB:-/mnt/pfs/luotianwei/env_overlays/cudnn-9.10.2.21-cu12/nvidia/cudnn/lib}"
legacy_lerobot="${LEGACY_LEROBOT:-/mnt/pfs/luotianwei/handson-h200-bench-repo/legacy_lerobot}"
config=physx_v3_actual_eef52_wuji1
dataset="${DATASET_ROOT:?DATASET_ROOT is required}"
label="${TRAIN_LABEL:?TRAIN_LABEL is required}"
assets_base="${ASSETS_BASE_DIR:?ASSETS_BASE_DIR is required}"
base_params="${BASE_PARAMS:-/mnt/pfs/luotianwei/handson/checkpoints/pi05_base_action_dim_52_fresh_v2/params}"
checkpoint_root="${CHECKPOINT_ROOT:-/mnt/pfs/luotianwei/handson/checkpoints/hp5090_physx_v3_actual_eef52_fullft_data_ablation_150k_20260809}"
num_train_steps="${NUM_TRAIN_STEPS:-150000}"
save_interval="${SAVE_INTERVAL:-10000}"
batch_size="${BATCH_SIZE:-32}"
stamp="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
run="physxv3-actualeef52-${label}-fullft${num_train_steps}-fsdp8-b${batch_size}-wandb-${stamp}"
run_dir="${repo}/logs/${run}"
log="${run_dir}/train.log"
status_file="${run_dir}/status.env"
assets="${assets_base}/${config}/local_repo"
mkdir -p "${run_dir}"

read -r episodes frames < <(
  "${python}" - "${dataset}/meta/info.json" <<'PY'
import json
import sys
from pathlib import Path

info = json.loads(Path(sys.argv[1]).read_text())
assert info["features"]["observation.state"]["shape"] == [52]
assert info["features"]["action"]["shape"] == [52]
print(int(info["total_episodes"]), int(info["total_frames"]))
PY
)

test -d "${base_params}"
test -f "${cudnn_lib}/libcudnn.so.9"
test -d "${legacy_lerobot}/lerobot"
test -f "${dataset}/meta/handson_policy_schema.json"
test -f "${assets}/norm_stats.json"
test -f "${assets}/handson_policy_schema.json"

cat >"${status_file}" <<EOF
FORMAT=handson.pi05_fullft_data_ablation
HOST=$(hostname -s)
CONFIG=${config}
DATASET=${dataset}
ASSETS=${assets}
CUDNN_LIB=${cudnn_lib}
LEGACY_LEROBOT=${legacy_lerobot}
POLICY_SCHEMA=handson.policy.actual_eef52.v1
EPISODES=${episodes}
FRAMES=${frames}
RUN=${run}
LOG=${log}
BASE_PARAMS=${base_params}
CHECKPOINT_ROOT=${checkpoint_root}
GPU_COUNT=8
GLOBAL_BATCH_SIZE=${batch_size}
FINETUNE_MODE=full
MAX_STEPS=${num_train_steps}
SAVE_INTERVAL=${save_interval}
STATUS=starting
UPDATED_AT=$(date --iso-8601=seconds)
EOF

cd "${repo}"
set +e
env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.75}" \
  XLA_FLAGS=--xla_gpu_autotune_level=0 \
  LD_LIBRARY_PATH="${cudnn_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  PYTHONPATH="${legacy_lerobot}:${repo}/src" \
  PI05_FINETUNE_MODE=full \
  "${python}" -u scripts/train_selective_finetune.py \
    "${config}" \
    --exp-name "${run}" \
    --checkpoint-base-dir "${checkpoint_root}" \
    --assets-base-dir "${assets_base}" \
    --weight-loader.params-path "${base_params}" \
    --data.root "${dataset}" \
    --batch-size "${batch_size}" \
    --num-workers 16 \
    --num-train-steps "${num_train_steps}" \
    --log-interval 100 \
    --save-interval "${save_interval}" \
    --keep-period "${save_interval}" \
    --fsdp-devices 8 \
    --wandb-enabled \
    >"${log}" 2>&1
rc=$?
set -e

{
  if (( rc == 0 )); then
    echo STATUS=complete
  else
    echo STATUS=failed
  fi
  echo EXIT_CODE=${rc}
  echo UPDATED_AT=$(date --iso-8601=seconds)
} >>"${status_file}"
exit "${rc}"
