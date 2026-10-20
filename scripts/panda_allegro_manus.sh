#!/usr/bin/env bash
# One-terminal launcher for the original HandsOn Panda + Allegro collection path
# with MANUS ROS2 hand input.
#
# Usage:
#   ./scripts/panda_allegro_manus.sh [right|both] [task]
#
# Defaults:
#   right hand -> water_plant
#   both hands -> bimanual_assembly

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_ROOT="$ROOT/teleoperation/manus/vendor"
MANUS_ROOT="${MANUS_WUJI_ROOT:-$VENDOR_ROOT/wuji-hand-teleop}"
MANUS_ROOT="$(cd "$MANUS_ROOT" && pwd)"

resolve_ros2_ws() {
  if [[ -n "${ROS2_WS:-}" ]]; then
    echo "$ROS2_WS"
    return
  fi

  local candidate
  for candidate in "$MANUS_ROOT" "$HOME/ros2_ws"; do
    if [[ -f "$candidate/install/setup.bash" ]]; then
      echo "$candidate"
      return
    fi
  done

  echo "$MANUS_ROOT"
}

WS="$(resolve_ros2_ws)"

HAND="${HANDSON_HAND:-right}"
if [[ $# -gt 0 ]]; then
  case "$1" in
    right|left|both)
      HAND="$1"
      shift
      ;;
  esac
fi

DEFAULT_TASK="water_plant"
if [[ "$HAND" == "both" ]]; then
  DEFAULT_TASK="bimanual_assembly"
fi
TASK="${1:-${HANDSON_TASK:-$DEFAULT_TASK}}"

HOST="${HANDSON_HOST:-127.0.0.1}"
USE_TRACKER="${HANDSON_USE_TRACKER:-1}"
TRACKER_SOURCE="${HANDSON_TRACKER_SOURCE:-vive_bridge}"
SWAP_HANDS="${HANDSON_SWAP_HANDS:-1}"
ROS_DOMAIN="${ROS_DOMAIN_ID:-71}"
CONDA_ENV="${HANDSON_CONDA_ENV:-act}"
GEORT_CONDA_ENV="${HANDSON_GEORT_CONDA_ENV:-$CONDA_ENV}"
OUT_DIR="${HANDSON_OUT_DIR:-/home/arx/下载/handson_demos}"
RENDER_MODE="${HANDSON_RENDER_MODE:-human}"
SUCCESSES_NEEDED="${HANDSON_SUCCESSES_NEEDED:-2}"
BRIDGE_FREQUENCY="${HANDSON_BRIDGE_FREQUENCY:-90}"
TELEOP_TARGET_FRAME="${HANDSON_TELEOP_TARGET_FRAME:-tracker}"
GEORT_PYTHON="${GEORT_PYTHON:-}"
RECORD_SCRIPT="${HANDSON_RECORD_SCRIPT:-$ROOT/scripts/record_demos_zarr.py}"
OPENVR_CONFIG="${HANDSON_OPENVR_CONFIG:-$MANUS_ROOT/src/input_devices/openvr_input/config/openvr_input.yaml}"
OPENVR_INPUT_ROOT="$MANUS_ROOT/src/input_devices/openvr_input"
VIVE_PYTHON="${HANDSON_VIVE_PYTHON:-conda run -n $CONDA_ENV python}"
VIVE_RIGHT_SERIAL="${HANDSON_VIVE_RIGHT_SERIAL:-}"
VIVE_LEFT_SERIAL="${HANDSON_VIVE_LEFT_SERIAL:-}"
VIVE_DEVICE_INDEX="${HANDSON_VIVE_DEVICE_INDEX:-}"
VIVE_SECOND_DEVICE_INDEX="${HANDSON_VIVE_SECOND_DEVICE_INDEX:-}"

PID_DIR="${XDG_RUNTIME_DIR:-/tmp}/handson-panda-allegro-manus"
LOG_DIR="$PID_DIR/logs"
ENV_SCRIPT="$PID_DIR/env.sh"
mkdir -p "$PID_DIR" "$LOG_DIR"

PIDS=()
BG_NAMES=()

quote_args() {
  printf "%q " "$@"
}

write_env_script() {
  cat >"$ENV_SCRIPT" <<EOF
#!/usr/bin/env bash
if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  source /opt/ros/jazzy/setup.bash
elif [[ -f /opt/ros/humble/setup.bash ]]; then
  source /opt/ros/humble/setup.bash
else
  echo "[ERROR] ROS2 not found" >&2
  exit 1
fi
source "$WS/install/setup.bash"
export ROS_DOMAIN_ID=$ROS_DOMAIN
export DISPLAY=\${DISPLAY:-:0}
export PYTHONUNBUFFERED=1
export ROS_LOG_DIR=\${ROS_LOG_DIR:-$LOG_DIR/ros}
if [[ -d "$OPENVR_INPUT_ROOT" ]]; then
  export PYTHONPATH="$OPENVR_INPUT_ROOT:\${PYTHONPATH:-}"
fi
mkdir -p "\$ROS_LOG_DIR"
EOF
  chmod +x "$ENV_SCRIPT"
}

start_bg() {
  local name="$1"
  local cmd="$2"
  setsid bash -lc "$cmd" >"$LOG_DIR/$name.log" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  BG_NAMES+=("$name")
  echo "$pid" >"$PID_DIR/$name.pid"
  echo "Started $name (pid $pid, log: $LOG_DIR/$name.log)"
}

check_alive() {
  local name="$1"
  local pid_file="$PID_DIR/$name.pid"
  if [[ ! -f "$pid_file" ]]; then
    return
  fi
  local pid
  pid="$(cat "$pid_file")"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[ERROR] $name exited early. Log tail:" >&2
    tail -n 80 "$LOG_DIR/$name.log" >&2 || true
    exit 1
  fi
}

check_all_alive() {
  local name
  for name in "${BG_NAMES[@]:-}"; do
    check_alive "$name"
  done
}

cleanup() {
  local pid
  trap - EXIT INT TERM
  for pid in "${PIDS[@]:-}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
    kill "$pid" 2>/dev/null || true
  done
  sleep 0.5
  for pid in "${PIDS[@]:-}"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
  done
  rm -f "$PID_DIR"/*.pid 2>/dev/null || true
}
trap cleanup EXIT INT TERM

stop_existing_stack() {
  local patterns=(
    "geort/mocap/rokoko_retarget_send_right.py"
    "geort/mocap/rokoko_retarget_send_left.py"
    "teleoperation/vive_bridge/send_vive_pose.py"
    "teleoperation/manus/manus_geort_ros2_bridge.py"
    "ros2 run manus_ros2 manus_data_publisher"
    "ros2 run manus_input_py manus_input"
    "ros2 run openvr_input openvr_input"
    "$WS/install/manus_ros2/lib/manus_ros2/manus_data_publisher"
    "$WS/install/manus_input_py/lib/manus_input_py/manus_input"
    "$WS/install/openvr_input/lib/openvr_input/openvr_input"
    "$ROOT/scripts/record_demos_zarr.py"
    "$ROOT/scripts/record_demos_zarr_manualsave.py"
  )

  local pattern
  for pattern in "${patterns[@]}"; do
    pkill -TERM -f "$pattern" 2>/dev/null || true
  done
  sleep 1
  for pattern in "${patterns[@]}"; do
    pkill -KILL -f "$pattern" 2>/dev/null || true
  done
}

if [[ "$HAND" != "right" && "$HAND" != "left" && "$HAND" != "both" ]]; then
  echo "[ERROR] hand must be right, left, or both (got: $HAND)" >&2
  exit 1
fi
if [[ "$TRACKER_SOURCE" != "vive_bridge" && "$TRACKER_SOURCE" != "ros2" && "$TRACKER_SOURCE" != "none" ]]; then
  echo "[ERROR] HANDSON_TRACKER_SOURCE must be vive_bridge, ros2, or none (got: $TRACKER_SOURCE)" >&2
  exit 1
fi
if [[ ! -f "$WS/install/setup.bash" ]]; then
  echo "[ERROR] ROS2 workspace not built: $WS/install/setup.bash" >&2
  echo "  Build the vendored workspace:" >&2
  echo "    cd $MANUS_ROOT && colcon build --symlink-install" >&2
  echo "  Or set ROS2_WS to an already built workspace, for example:" >&2
  echo "    export ROS2_WS=$HOME/ros2_ws" >&2
  exit 1
fi
if [[ ! -f "$RECORD_SCRIPT" ]]; then
  echo "[ERROR] Missing record script: $RECORD_SCRIPT" >&2
  exit 1
fi

unset HANDSON_HAND_TARGET_CONTROL
unset HANDSON_HAND_DIRECT_QPOS
unset HANDSON_HANDS_ONLY
unset HANDSON_ROKAE_WUJI_ARM_ACTUATOR
unset HANDSON_ROKAE_WUJI_TABLETOP_RESET
export HANDSON_TELEOP_TARGET_FRAME="$TELEOP_TARGET_FRAME"

"$ROOT/scripts/mode_a_handson.sh" stop >/dev/null 2>&1 || true
if [[ "${HANDSON_STOP_EXISTING:-1}" != "0" ]]; then
  stop_existing_stack
fi
write_env_script

echo "Panda + Allegro MANUS launcher"
echo "  hand: $HAND"
echo "  task: $TASK"
echo "  tracker: $USE_TRACKER"
echo "  tracker source: $TRACKER_SOURCE"
echo "  teleop target frame: $TELEOP_TARGET_FRAME"
echo "  swap hands: $SWAP_HANDS"
if [[ -n "$GEORT_PYTHON" ]]; then
  echo "  geort python: $GEORT_PYTHON"
else
  echo "  geort conda env: $GEORT_CONDA_ENV"
fi
echo "  logs: $LOG_DIR"
echo ""

env_q="$(printf "%q" "$ENV_SCRIPT")"
manus_root_q="$(printf "%q" "$MANUS_ROOT")"
root_q="$(printf "%q" "$ROOT")"

yaml_value() {
  local key="$1"
  local file="$2"
  sed -n "s/^[[:space:]]*$key:[[:space:]]*['\\\"]\\?\\([^'\\\"#]*\\)['\\\"]\\?.*$/\\1/p" "$file" | head -n 1 | tr -d '[:space:]'
}

is_auto_tracker_serial() {
  local value="${1:-}"
  local normalized="${value,,}"
  [[ -z "$normalized" || "$normalized" == "auto" || "$normalized" == "any" || "$normalized" == "*" || "$normalized" == "lhr-xxxxxxxx" ]]
}

yaml_tracker_serial() {
  local value
  value="$(yaml_value "$1" "$2")"
  if is_auto_tracker_serial "$value"; then
    echo ""
  else
    echo "$value"
  fi
}

start_bg "manus_data_publisher" \
  "source $env_q && cd $manus_root_q && exec ros2 run manus_ros2 manus_data_publisher"
sleep 3
check_alive "manus_data_publisher"

start_bg "manus_input" \
  "source $env_q && cd $manus_root_q && exec ros2 run manus_input_py manus_input"
sleep 2
check_alive "manus_input"

if [[ "$USE_TRACKER" != "0" && "$USE_TRACKER" != "false" && "$USE_TRACKER" != "False" && "$TRACKER_SOURCE" == "ros2" ]]; then
  if [[ ! -f "$OPENVR_CONFIG" ]]; then
    echo "[ERROR] Missing OpenVR config: $OPENVR_CONFIG" >&2
    exit 1
  fi
  openvr_config_q="$(printf "%q" "$OPENVR_CONFIG")"
  start_bg "openvr_input" \
    "source $env_q && cd $manus_root_q && exec ros2 run openvr_input openvr_input -c $openvr_config_q"
  sleep 2
  check_alive "openvr_input"
fi

start_vive_bridge() {
  local cmd
  local vive_args=(
    "$ROOT/teleoperation/vive_bridge/send_vive_pose.py"
    --host "$HOST"
    --port 5012
    --frequency "$BRIDGE_FREQUENCY"
    --quiet
  )

  if [[ "$HAND" == "both" ]]; then
    if [[ -z "$VIVE_RIGHT_SERIAL" && -f "$OPENVR_CONFIG" ]]; then
      VIVE_RIGHT_SERIAL="$(yaml_tracker_serial right_wrist "$OPENVR_CONFIG")"
    fi
    if [[ -z "$VIVE_LEFT_SERIAL" && -f "$OPENVR_CONFIG" ]]; then
      VIVE_LEFT_SERIAL="$(yaml_tracker_serial left_wrist "$OPENVR_CONFIG")"
    fi
    vive_args+=(--two-trackers)
    if [[ -n "$VIVE_DEVICE_INDEX" ]]; then
      vive_args+=(--device-index "$VIVE_DEVICE_INDEX")
    elif [[ -n "$VIVE_RIGHT_SERIAL" ]]; then
      vive_args+=(--serial-contains "$VIVE_RIGHT_SERIAL")
    fi
    if [[ -n "$VIVE_SECOND_DEVICE_INDEX" ]]; then
      vive_args+=(--second-device-index "$VIVE_SECOND_DEVICE_INDEX")
    elif [[ -n "$VIVE_LEFT_SERIAL" ]]; then
      vive_args+=(--second-serial-contains "$VIVE_LEFT_SERIAL")
    fi
  else
    local serial="$VIVE_RIGHT_SERIAL"
    if [[ "$HAND" == "left" ]]; then
      serial="$VIVE_LEFT_SERIAL"
    fi
    if [[ -z "$serial" && -f "$OPENVR_CONFIG" ]]; then
      if [[ "$HAND" == "left" ]]; then
        serial="$(yaml_tracker_serial left_wrist "$OPENVR_CONFIG")"
      else
        serial="$(yaml_tracker_serial right_wrist "$OPENVR_CONFIG")"
      fi
    fi
    if [[ -n "$VIVE_DEVICE_INDEX" ]]; then
      vive_args+=(--device-index "$VIVE_DEVICE_INDEX")
    elif [[ -n "$serial" ]]; then
      vive_args+=(--serial-contains "$serial")
    fi
  fi

  cmd="$(quote_args $VIVE_PYTHON "${vive_args[@]}")"
  start_bg "vive_bridge" "cd $root_q && exec $cmd"
}

if [[ "$USE_TRACKER" != "0" && "$USE_TRACKER" != "false" && "$USE_TRACKER" != "False" && "$TRACKER_SOURCE" == "vive_bridge" ]]; then
  start_vive_bridge
  sleep 2
  check_alive "vive_bridge"
fi

bridge_cmd=(
  python3 "$ROOT/teleoperation/manus/manus_geort_ros2_bridge.py"
  --hand "$HAND"
  --host "$HOST"
  --frequency "$BRIDGE_FREQUENCY"
)
if [[ "$USE_TRACKER" != "0" && "$USE_TRACKER" != "false" && "$USE_TRACKER" != "False" && "$TRACKER_SOURCE" == "ros2" ]]; then
  bridge_cmd+=(--send-tracker)
fi
if [[ "$SWAP_HANDS" != "0" && "$SWAP_HANDS" != "false" && "$SWAP_HANDS" != "False" ]]; then
  bridge_cmd+=(--swap-hands)
fi
if [[ "${HANDSON_SWAP_WRIST_TRACKERS:-0}" == "1" ]]; then
  bridge_cmd+=(--swap-wrist-trackers)
fi
if [[ "${HANDSON_ALREADY_CANONICAL:-0}" == "1" ]]; then
  bridge_cmd+=(--already-canonical)
fi
bridge_q="$(quote_args "${bridge_cmd[@]}")"
start_bg "manus_geort_bridge" \
  "source $env_q && cd $root_q && exec $bridge_q"
sleep 2
check_alive "manus_geort_bridge"

start_geort_right() {
  local cmd
  if [[ -n "$GEORT_PYTHON" ]]; then
    cmd="$(quote_args \
      "$GEORT_PYTHON" geort/mocap/rokoko_retarget_send_right.py \
      -ckpt_tag handson_right_default \
      --bind_ip "$HOST" --bind_port 5013 \
      --target_ip "$HOST" --target_port 5014)"
  else
    cmd="$(quote_args \
      conda run -n "$GEORT_CONDA_ENV" python geort/mocap/rokoko_retarget_send_right.py \
      -ckpt_tag handson_right_default \
      --bind_ip "$HOST" --bind_port 5013 \
      --target_ip "$HOST" --target_port 5014)"
  fi
  start_bg "geort_right" "cd $(printf "%q" "$ROOT/teleoperation/GeoRT") && exec $cmd"
}

start_geort_left() {
  local cmd
  if [[ -n "$GEORT_PYTHON" ]]; then
    cmd="$(quote_args \
      "$GEORT_PYTHON" geort/mocap/rokoko_retarget_send_left.py \
      -ckpt_tag handson_left_default \
      --bind_ip "$HOST" --bind_port 5015 \
      --target_ip "$HOST" --target_port 5016)"
  else
    cmd="$(quote_args \
      conda run -n "$GEORT_CONDA_ENV" python geort/mocap/rokoko_retarget_send_left.py \
      -ckpt_tag handson_left_default \
      --bind_ip "$HOST" --bind_port 5015 \
      --target_ip "$HOST" --target_port 5016)"
  fi
  start_bg "geort_left" "cd $(printf "%q" "$ROOT/teleoperation/GeoRT") && exec $cmd"
}

if [[ "$HAND" == "right" || "$HAND" == "both" ]]; then
  start_geort_right
  sleep 2
  check_alive "geort_right"
fi
if [[ "$HAND" == "left" || "$HAND" == "both" ]]; then
  start_geort_left
  sleep 2
  check_alive "geort_left"
fi

sleep 1
check_all_alive

echo ""
echo "Starting HandsOn collection in the foreground."
echo "Press ';' in the MuJoCo window to toggle teleop. Press 'r' to discard/reset."
echo "Background logs stay in: $LOG_DIR"
echo ""

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/handson"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"

conda run -n "$CONDA_ENV" python "$RECORD_SCRIPT" \
  --exp_name="$TASK" \
  --successes_needed="$SUCCESSES_NEEDED" \
  --arm_type=panda \
  --hand_type=allegro \
  --render_mode="$RENDER_MODE" \
  --out_dir="$OUT_DIR" \
  2>&1 | tee "$LOG_DIR/record_demos.log"
