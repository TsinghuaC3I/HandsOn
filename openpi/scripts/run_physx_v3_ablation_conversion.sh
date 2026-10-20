#!/usr/bin/env bash
set -euo pipefail

size="${1:?usage: $0 full|10|50|100|200|300|400}"
repo="${REPO_ROOT:-/mnt/pfs/luotianwei/handson/openpi}"
python="${PYTHON_BIN:-/mnt/pfs/luotianwei/envs/handson-openpi-jax062/bin/python}"
raw_root=/mnt/pfs/luotianwei/handson/datasets/new_cloud_physx_v3_all8421_correct_20260809
dataset_parent=/mnt/pfs/luotianwei/handson/datasets/lerobot_v21
assets_base=/mnt/pfs/luotianwei/handson/training_assets/physx_v3_actual_eef52_all8421_correct_20260809
log_root="${raw_root}/_manifests/lerobot_conversion_logs"
mkdir -p "${log_root}"

if [[ "${size}" == full ]]; then
  manifest="${raw_root}/_manifests/new_cloud_physx_v3_sync.json"
  output="${dataset_parent}/physx_v3_actual_eef52_all8421_correct_20260809"
  assets="${assets_base}/physx_v3_actual_eef52_wuji1/local_repo"
  repo_id=handson/physx_v3_actual_eef52_all8421_correct_20260809
else
  case "${size}" in
    10|50|100|200|300|400) ;;
    *) echo "unsupported size: ${size}" >&2; exit 2 ;;
  esac
  manifest="${raw_root}/_manifests/a01_data_ablation/a01_$(printf '%03d' "${size}").json"
  output="${dataset_parent}/physx_v3_actual_eef52_a01_${size}ep_20260809"
  assets="${output}/_conversion_assets/local_repo"
  repo_id="handson/physx_v3_actual_eef52_a01_${size}ep_20260809"
fi

test -f "${manifest}"
if [[ -e "${output}" || -e "${assets}" ]]; then
  echo "refusing to overwrite existing output: ${output} or ${assets}" >&2
  exit 3
fi

log="${log_root}/${size}.log"
cd "${repo}"
exec "${python}" -u scripts/build_physx_v3_actual_eef52_lerobot_v21.py \
  --raw-root "${raw_root}" \
  --manifest "${manifest}" \
  --output-root "${output}" \
  --assets-dir "${assets}" \
  --repo-id "${repo_id}" \
  --include-source-success-without-certification \
  >"${log}" 2>&1
