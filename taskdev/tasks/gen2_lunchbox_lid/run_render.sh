#!/usr/bin/env bash
set -euo pipefail
# Usage: run_render.sh HOST GPU_INDEX GPU_UUID UNIQUE_RUN source|real_v12 [options]
source "$(dirname -- "${BASH_SOURCE[0]}")/validation_env.sh"
gen2_prepare_runtime "${1:?allocated host required}" "${2:?physical GPU index required}" "${3:?full GPU UUID required}" "${4:?unique run name required}"
GEN2_VISUAL_PROFILE=${5:?source or real_v12 required}
[[ "$GEN2_VISUAL_PROFILE" == source || "$GEN2_VISUAL_PROFILE" == real_v12 ]]
shift 5
export HANDSON_ISAACSIM_TASK_VISUAL_PROFILE="$GEN2_VISUAL_PROFILE"
set +e
gen2_in_gpu_namespace timeout --signal=TERM --kill-after=20s 480s "$HANDSON_ISAACSIM_PYTHON" taskdev/tasks/eval_gen01_cover_lunchbox/validate_render.py --out "$GEN2_RUN/render_audit.json" --profile "$GEN2_VISUAL_PROFILE" "$@" > "$GEN2_RUN/child.log" 2>&1
GEN2_EXIT=$?
printf '%s\n' "$GEN2_EXIT" > "$GEN2_RUN/child_exit_code.txt"
# SimulationApp's fast close may exit the child with zero before Python returns.
# A successful diagnostic therefore also requires its complete positive report.
if [[ "$GEN2_EXIT" == 0 ]]; then
  "$HANDSON_ISAACSIM_PYTHON" -B -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("passed") is True and not r.get("errors") else 1)' "$GEN2_RUN/render_audit.json"
  GEN2_EXIT=$?
fi
printf '%s\n' "$GEN2_EXIT" > "$GEN2_RUN/exit_code.txt"
tail -60 "$GEN2_RUN/child.log"
exit "$GEN2_EXIT"
