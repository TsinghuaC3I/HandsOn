#!/usr/bin/env bash
set -euo pipefail
# Usage: run_live.sh HOST GPU_INDEX GPU_UUID UNIQUE_RUN [validate_live options]
source "$(dirname -- "${BASH_SOURCE[0]}")/validation_env.sh"
gen2_prepare_runtime "${1:?allocated host required}" "${2:?physical GPU index required}" "${3:?full GPU UUID required}" "${4:?unique run name required}"
shift 4
set +e
gen2_in_gpu_namespace timeout --signal=TERM --kill-after=20s 360s "$HANDSON_ISAACSIM_PYTHON" taskdev/tasks/eval_gen01_cover_lunchbox/validate_live.py --out "$GEN2_RUN/audit.json" "$@" > "$GEN2_RUN/child.log" 2>&1
GEN2_EXIT=$?
printf '%s\n' "$GEN2_EXIT" > "$GEN2_RUN/exit_code.txt"
tail -60 "$GEN2_RUN/child.log"
exit "$GEN2_EXIT"
