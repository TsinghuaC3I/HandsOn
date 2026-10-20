#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/pfs/luotianwei/handson-h200-bench-repo/openpi
python=/mnt/pfs/luotianwei/envs/handson-openpi-jax062/bin/python
config=physx_v3_actual_eef52_wuji1
dataset="${DATASET_ROOT:-/mnt/pfs/luotianwei/handson/datasets/lerobot_v21/physx_v3_actual_eef52}"
assets="${repo}/assets/${config}/local_repo"
base_params=/mnt/pfs/luotianwei/handson/checkpoints/pi05_base_action_dim_52_fresh_v2/params
warm_root=/mnt/pfs/luotianwei/handson/checkpoints/hp5090_physx_v3_actual_eef52_expert_warmup
full_root=/mnt/pfs/luotianwei/handson/checkpoints/hp5090_physx_v3_actual_eef52_freeze_vision
num_train_steps="${NUM_TRAIN_STEPS:-30000}"
stamp="$(date +%Y%m%d-%H%M%S)"

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

label="physxv3-actualeef52-${episodes}ep-${frames}f-controlorder-v1-freshhead"
stage1_run="${label}-expertwarmup2k-fsdp8-b32-wandb-${stamp}"
stage2_run="${label}-freezevision${num_train_steps}-fsdp8-b32-wandb-${stamp}"
pipeline_dir="${repo}/logs/${stage2_run}.pipeline"
stage1_log="${pipeline_dir}/stage1_expert_warmup.log"
stage2_log="${pipeline_dir}/stage2_freeze_vision.log"
status_file="${pipeline_dir}/status.env"
mkdir -p "${pipeline_dir}"

test -d "${base_params}"
test -f "${dataset}/meta/handson_policy_schema.json"
test -f "${assets}/norm_stats.json"
test -f "${assets}/handson_policy_schema.json"

cd "${repo}"
"${python}" scripts/audit_handson_policy_schema.py "${dataset}" "${assets}"

write_status() {
  {
    printf 'FORMAT=handson.physx_v3_actual_eef52_training_pipeline\n'
    printf 'HOST=%s\n' "$(hostname -s)"
    printf 'CONFIG=%s\n' "${config}"
    printf 'DATASET=%s\n' "${dataset}"
    printf 'POLICY_SCHEMA=handson.policy.actual_eef52.v1\n'
    printf 'EPISODES=%s\n' "${episodes}"
    printf 'FRAMES=%s\n' "${frames}"
    printf 'STAGE=%s\n' "$1"
    printf 'STAGE1_RUN=%s\n' "${stage1_run}"
    printf 'STAGE2_RUN=%s\n' "${stage2_run}"
    printf 'STAGE1_LOG=%s\n' "${stage1_log}"
    printf 'STAGE2_LOG=%s\n' "${stage2_log}"
    printf 'BASE_PARAMS=%s\n' "${base_params}"
    printf 'WARM_ROOT=%s\n' "${warm_root}"
    printf 'FULL_ROOT=%s\n' "${full_root}"
    printf 'GPU_COUNT=8\n'
    printf 'GLOBAL_BATCH_SIZE=32\n'
    printf 'MAX_STEPS=%s\n' "${num_train_steps}"
    printf 'UPDATED_AT=%s\n' "$(date --iso-8601=seconds)"
  } >"${status_file}"
}

common_env=(
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
  PYTHONPATH="${repo}/../legacy_lerobot:${repo}/src"
)

write_status stage1_expert_warmup
env "${common_env[@]}" PI05_FINETUNE_MODE=expert_only \
  "${python}" -u scripts/train_selective_finetune.py \
  "${config}" \
  --exp-name "${stage1_run}" \
  --checkpoint-base-dir "${warm_root}" \
  --weight-loader.params-path "${base_params}" \
  --data.root "${dataset}" \
  --batch-size 32 \
  --num-workers 16 \
  --num-train-steps 2000 \
  --log-interval 100 \
  --save-interval 2000 \
  --keep-period 2000 \
  --fsdp-devices 8 \
  --wandb-enabled \
  >"${stage1_log}" 2>&1

warm_params="${warm_root}/${config}/${stage1_run}/1999/params"
test -d "${warm_params}"

write_status stage2_freeze_vision
env "${common_env[@]}" PI05_FINETUNE_MODE=freeze_vision \
  "${python}" -u scripts/train_selective_finetune.py \
  "${config}" \
  --exp-name "${stage2_run}" \
  --checkpoint-base-dir "${full_root}" \
  --weight-loader.params-path "${warm_params}" \
  --data.root "${dataset}" \
  --batch-size 32 \
  --num-workers 16 \
  --num-train-steps "${num_train_steps}" \
  --log-interval 100 \
  --save-interval 10000 \
  --keep-period 10000 \
  --fsdp-devices 8 \
  --wandb-enabled \
  >"${stage2_log}" 2>&1

write_status complete
