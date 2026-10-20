#!/usr/bin/env bash
set -euo pipefail
# Usage: run_render.sh HOST GPU_INDEX GPU_UUID UNIQUE_RUN source|real_v12 [options]
source "$(dirname -- "${BASH_SOURCE[0]}")/validation_env.sh"
gen7_prepare_runtime "${1:?allocated host required}" "${2:?physical GPU index required}" "${3:?full GPU UUID required}" "${4:?unique run name required}"
GEN7_VISUAL_PROFILE=${5:?source or real_v12 required}
[[ "$GEN7_VISUAL_PROFILE" == source || "$GEN7_VISUAL_PROFILE" == real_v12 ]]
shift 5
export HANDSON_ISAACSIM_TASK_VISUAL_PROFILE="$GEN7_VISUAL_PROFILE"
set +e
gen7_in_gpu_namespace timeout --signal=TERM --kill-after=20s 480s "$HANDSON_ISAACSIM_PYTHON" taskdev/tasks/eval_gen06_spray_shoe/validate_render.py --out "$GEN7_RUN/render_audit.json" --profile "$GEN7_VISUAL_PROFILE" "$@" > "$GEN7_RUN/child.log" 2>&1
GEN7_EXIT=$?
printf '%s\n' "$GEN7_EXIT" > "$GEN7_RUN/child_exit_code.txt"
# SimulationApp's fast close may exit the child with zero before Python returns.
# A successful diagnostic therefore also requires its complete positive report.
if [[ "$GEN7_EXIT" == 0 ]]; then
  "$HANDSON_ISAACSIM_PYTHON" -B -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("passed") is True and not r.get("errors") else 1)' "$GEN7_RUN/render_audit.json"
  GEN7_EXIT=$?
fi
printf '%s\n' "$GEN7_EXIT" > "$GEN7_RUN/exit_code.txt"
tail -60 "$GEN7_RUN/child.log"
exit "$GEN7_EXIT"
