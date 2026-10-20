#!/usr/bin/env bash
# Fresh, allocated single-GPU GEN1 preview; never modifies the shared runtime.
set -euo pipefail
GEN1_TASK_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$GEN1_TASK_DIR/validation_env.sh"
: "${GEN1_HOST:?Set the supervisor-assigned host}"
: "${GEN1_GPU_INDEX:?Set the supervisor-assigned physical GPU index}"
: "${GEN1_GPU_UUID:?Set the supervisor-assigned full GPU UUID}"
: "${GEN1_RUN_NAME:?Choose a new evidence directory name}"
: "${HANDSON_ISAACSIM_TASK_VISUAL_PROFILE:?Set source or real_v12 before this cold process}"
gen1_prepare_runtime "$GEN1_HOST" "$GEN1_GPU_INDEX" "$GEN1_GPU_UUID" "$GEN1_RUN_NAME"
exec 9>"$GEN1_ROOT/.gen1-cache/visual_gpu_${GEN1_GPU_UUID}.lock"
flock -n 9 || { printf '%s\n' 'GEN1 GPU diagnostic lock is already held' >&2; exit 2; }
GEN1_REFERENCE_ARGS=()
if [[ -n "${GEN1_VISUAL_REFERENCE:-}" ]]; then
  GEN1_REFERENCE_ARGS=(--reference "$GEN1_VISUAL_REFERENCE")
fi
gen1_in_gpu_namespace "$HANDSON_ISAACSIM_PYTHON" -B \
  "$GEN1_TASK_DIR/${GEN1_NATIVE_SCRIPT:-native_visual_validation.py}" \
  --out "$GEN1_RUN/visual_report.json" --profile "$HANDSON_ISAACSIM_TASK_VISUAL_PROFILE" \
  --seed 7 --phase "${GEN1_VISUAL_PHASE:-reset}" "${GEN1_REFERENCE_ARGS[@]}" \
  > "$GEN1_RUN/child_stdout.log" 2>&1
# Kit cleanup can terminate with exit 0 before the Python caller returns.
"$HANDSON_ISAACSIM_PYTHON" -B - "$GEN1_RUN/visual_report.json" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1]); report=json.loads(path.read_text())
assert report.get('complete') and report.get('passed'), report.get('errors', report)
assert set(report.get('images',{})) == {'main','wrist_left','wrist_right'}
print(json.dumps({'report': str(path), 'passed': True}))
PY
