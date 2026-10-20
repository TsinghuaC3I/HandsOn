#!/usr/bin/env bash
set -euo pipefail
# Usage: run_render.sh HOST GPU_INDEX GPU_UUID UNIQUE_RUN source|real_v12 [options]
source "$(dirname -- "${BASH_SOURCE[0]}")/validation_env.sh"
gen3_prepare_runtime "${1:?allocated host required}" "${2:?physical GPU index required}" "${3:?full GPU UUID required}" "${4:?unique run name required}"
GEN3_VISUAL_PROFILE=${5:?source or real_v12 required}
[[ "$GEN3_VISUAL_PROFILE" == source || "$GEN3_VISUAL_PROFILE" == real_v12 ]]
shift 5
export HANDSON_ISAACSIM_TASK_VISUAL_PROFILE="$GEN3_VISUAL_PROFILE"
set +e
gen3_in_gpu_namespace timeout --signal=TERM --kill-after=20s 480s "$HANDSON_ISAACSIM_PYTHON" taskdev/tasks/eval_gen02_open_trash_bin/validate_render.py --out "$GEN3_RUN/render_audit.json" --profile "$GEN3_VISUAL_PROFILE" "$@" > "$GEN3_RUN/child.log" 2>&1
GEN3_EXIT=$?
printf '%s\n' "$GEN3_EXIT" > "$GEN3_RUN/child_exit_code.txt"
# Fast Kit shutdown can exit zero before Python propagates a failed audit.
if [[ "$GEN3_EXIT" == 0 ]]; then
  "$HANDSON_ISAACSIM_PYTHON" -B -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("passed") is True and not r.get("errors") else 1)' "$GEN3_RUN/render_audit.json"
  GEN3_EXIT=$?
fi
printf '%s\n' "$GEN3_EXIT" > "$GEN3_RUN/exit_code.txt"
tail -60 "$GEN3_RUN/child.log"
exit "$GEN3_EXIT"
