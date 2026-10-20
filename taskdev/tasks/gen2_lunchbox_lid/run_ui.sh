#!/usr/bin/env bash
set -euo pipefail
# Software smoke only: no-device configuration is not human acceptance.
# Usage: run_ui.sh HOST GPU_INDEX GPU_UUID UNIQUE_RUN PORT
source "$(dirname -- "${BASH_SOURCE[0]}")/validation_env.sh"
GEN2_UI_PORT=${5:?explicit unused UI port required}
[[ "$GEN2_UI_PORT" =~ ^[0-9]+$ && "$GEN2_UI_PORT" -ge 1024 && "$GEN2_UI_PORT" -le 65535 ]] || exit 2
gen2_prepare_runtime "${1:?allocated host required}" "${2:?physical GPU index required}" "${3:?full GPU UUID required}" "${4:?unique run name required}"
# A task-owned Git executable may be needed on minimal collection hosts.
if ! command -v git >/dev/null && [[ -x "$GEN2_TASK_DIR/evidence/tools/usr/bin/git" ]]; then
  export PATH="$GEN2_TASK_DIR/evidence/tools/usr/bin:$PATH"
  export GIT_EXEC_PATH="$GEN2_TASK_DIR/evidence/tools/usr/lib/git-core"
fi
export HANDSON_GRAPHICS_LIBRARY_PATH="$GEN2_GL"
export HANDSON_VULKAN_ICD="$VK_ICD_FILENAMES"
export HANDSON_NEWTON_GPU_TO_GPU_OVERLAY="$GEN2_RUN/unused_newton_overlay"
export HANDSON_BACKGROUND_V3_UPLOAD=0
export HANDSON_REVIEW_AUTO_UPLOAD=0
export HANDSON_IDLE_MAINTENANCE=0
export HANDSON_COLLECTION_CLEAN_STALE_ON_START=0
export HANDSON_WORK_PLAN_DESKTOP_NOTIFICATIONS=0
export HANDSON_ISAACSIM_CACHE_WARMUP=0
export HANDSON_POSTPROCESS_MANUAL_ONLY=1
export HANDSON_COLLECTION_TELEOP_STACK=none
export HANDSON_COLLECTION_REQUIRE_TELEOP_READY=0
export HANDSON_COLLECTION_SIM_BACKEND=isaacsim
export HANDSON_COLLECTION_ARM_TYPE=rokae
export HANDSON_COLLECTION_HAND_TYPE=wuji1
export HANDSON_RELEASE_ROLE=disabled
mkdir -p "$XDG_STATE_HOME" "$XDG_RUNTIME_DIR" "$HANDSON_MODE_A_LOG_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
printf '%s\n' "$$" > "$GEN2_RUN/ui.pid"
exec unshare --mount --propagation private -- /usr/bin/python3 \
  "$GEN2_TASK_DIR/simbench_gpu_namespace_exec.py" --host "$GEN2_HOST" \
  --physical-gpu "$GEN2_GPU_INDEX" --expected-uuid "$GEN2_GPU_UUID" --run-dir "$GEN2_RUN" -- \
  "$HANDSON_ISAACSIM_PYTHON" -B collection_ui/server.py --host 127.0.0.1 --port "$GEN2_UI_PORT" --root "$GEN2_ROOT" --python "$HANDSON_ISAACSIM_PYTHON" --isaacsim-python "$HANDSON_ISAACSIM_PYTHON" --out-dir "$GEN2_RUN/episodes" > "$GEN2_RUN/ui.log" 2>&1
