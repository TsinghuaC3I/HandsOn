#!/usr/bin/env bash
# Mode A for HandsOn: MANUS glove -> /hand_input -> HandsOn UDP teleop.
#
# This mirrors wuji-hand-teleop/scripts/mode_a_mujoco.sh, but replaces the
# final standalone MuJoCo viewer with HandsOn's ROKAE + Wuji bridge.
#
# Usage:
#   ./scripts/mode_a_handson.sh start [left|right|both] [tracker] [relative] [wuji1|wuji2]
#   ./scripts/mode_a_handson.sh stop
#   ./scripts/mode_a_handson.sh status
#   ./scripts/mode_a_handson.sh recover-openvr [left|right|both]

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

ACTION="${1:-start}"
HAND_ARGUMENT="${2:-}"
HAND="${HAND_ARGUMENT:-both}"
TRACKER="${3:-}"
TRACKER_MODE="${4:-relative}"
HAND_TYPE="${5:-${HANDSON_HAND_TYPE:-wuji1}}"

PID_DIR="${XDG_RUNTIME_DIR:-/tmp}/handson-mode-a"
LOG_DIR="$PID_DIR/logs"
ENV_SCRIPT="$PID_DIR/env.sh"
MANUS_INPUT_CONFIG="$PID_DIR/manus_input.yaml"
STACK_STATE_FILE="$PID_DIR/stack.env"
STACK_SCOPE_FILE="$PID_DIR/stack.scope"
OPENVR_SCOPE_FILE="$PID_DIR/openvr.scope"
STARTUP_COMPONENTS_MARKER="$PID_DIR/.startup-components-$$"
CALIBRATION_WAIT_TIMEOUT="${HANDSON_CALIBRATION_WAIT_TIMEOUT:-180}"
MANUS_HAND_WAIT_TIMEOUT="${HANDSON_MANUS_HAND_WAIT_TIMEOUT:-20}"
MANUS_USB_WAIT_TIMEOUT="${HANDSON_MANUS_USB_WAIT_TIMEOUT:-15}"
MANUS_USB_SETTLE_DELAY="${HANDSON_MANUS_USB_SETTLE_DELAY:-2}"
MANUS_PUBLISHER_START_ATTEMPTS="${HANDSON_MANUS_PUBLISHER_START_ATTEMPTS:-3}"
MANUS_PUBLISHER_START_TIMEOUT="${HANDSON_MANUS_PUBLISHER_START_TIMEOUT:-8}"
MANUS_PUBLISHER_RETRY_DELAY="${HANDSON_MANUS_PUBLISHER_RETRY_DELAY:-2}"
IGNORE_MANUS_LICENSE="${HANDSON_IGNORE_MANUS_LICENSE:-1}"
MANUS_CONNECTION_MODE="${HANDSON_MANUS_CONNECTION_MODE:-}"
MANUS_CONNECTION_MODES="${HANDSON_MANUS_CONNECTION_MODES:-auto}"
# Keep the two hand streams independent by default.  A disconnected glove
# must not stop the live glove from publishing; the strict all-hands behavior
# remains available with HANDSON_STRICT_MANUS_HANDS=1.
STRICT_MANUS_HANDS="${HANDSON_STRICT_MANUS_HANDS:-0}"
BRIDGE_FREQUENCY="${HANDSON_BRIDGE_FREQUENCY:-60}"
export HANDSON_TRACKER_DIRECT_PASSTHROUGH="${HANDSON_TRACKER_DIRECT_PASSTHROUGH:-1}"
# Rebase a tracker that stays displaced after the reconnect guard's short
# confirmation window.  This runtime-only safety policy must not invalidate
# the calibrated direction cache when it changes.
export HANDSON_TRACKER_RESET_AFTER_PERSISTENT_JUMP="${HANDSON_TRACKER_RESET_AFTER_PERSISTENT_JUMP:-1}"
SWAP_HANDS="${HANDSON_SWAP_HANDS:-0}"
SWAP_WRIST_TRACKERS="${HANDSON_SWAP_WRIST_TRACKERS:-0}"
SWAP_ARMS="${HANDSON_SWAP_ARMS:-0}"
STOP_INT_TIMEOUT="${HANDSON_MODE_A_STOP_INT_TIMEOUT:-2}"
STOP_TERM_TIMEOUT="${HANDSON_MODE_A_STOP_TERM_TIMEOUT:-2}"
STOP_KILL_TIMEOUT="${HANDSON_MODE_A_STOP_KILL_TIMEOUT:-1}"
OPENVR_START_TIMEOUT="${HANDSON_OPENVR_START_TIMEOUT:-90}"
OPENVR_START_ATTEMPTS="${HANDSON_OPENVR_START_ATTEMPTS:-1}"
OPENVR_RETRY_DELAY="${HANDSON_OPENVR_RETRY_DELAY:-2}"
STARTUP_CALIBRATION_PRINTED_LINES=0
STARTUP_CALIBRATION_HEADER_PRINTED=0

BRIDGE_SCRIPT="$ROOT/teleoperation/manus/manus_wuji_ros2_bridge.py"
HANDSON_MAPPING_CONFIG="${HANDSON_MANUS_MAPPING_CONFIG:-$ROOT/teleoperation/manus/config/handson_manus_mapping.yaml}"
OPENVR_TEMPLATE="$MANUS_ROOT/src/input_devices/openvr_input/config/openvr_input.yaml"
OPENVR_MACHINE_CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}/handson/openvr_input.yaml"
OPENVR_CONFIG="${HANDSON_OPENVR_CONFIG:-$OPENVR_MACHINE_CONFIG}"
if [[ "$(realpath -m "$OPENVR_CONFIG")" == "$(realpath -m "$OPENVR_TEMPLATE")" ]]; then
  echo "[WARN] HANDSON_OPENVR_CONFIG points to the repository template; using $OPENVR_MACHINE_CONFIG instead" >&2
  OPENVR_CONFIG="$OPENVR_MACHINE_CONFIG"
fi
OPENVR_CONFIG="$(realpath -m "$OPENVR_CONFIG")"
CALIBRATION_STATE_HOME="${XDG_STATE_HOME:-$HOME/.local/state}"
CALIBRATION_CACHE="${HANDSON_CALIBRATION_CACHE:-$CALIBRATION_STATE_HOME/handson/calibration/v1/mode_a.json}"
CALIBRATION_CACHE_CANDIDATE="${HANDSON_CALIBRATION_CACHE_CANDIDATE:-${CALIBRATION_CACHE}.candidate.$$}"
FORCE_RECALIBRATE="${HANDSON_FORCE_RECALIBRATE:-0}"
OPENVR_INPUT_ROOT="$MANUS_ROOT/src/input_devices/openvr_input"
TRACKER_CONFIG_SCRIPT="$ROOT/scripts/configure_openvr_trackers.py"
INDEPENDENT_LAUNCHER="$ROOT/scripts/run_mode_a_transient.sh"
RETARGET_ROOT="${WUJI_RETARGETING_ROOT:-$MANUS_ROOT/../wuji-retargeting}"
if [[ -d "$RETARGET_ROOT" ]]; then
  RETARGET_ROOT="$(cd "$RETARGET_ROOT" && pwd)"
fi

mkdir -p "$PID_DIR" "$LOG_DIR"

is_enabled() {
  case "${1:-}" in
    0|false|False|FALSE|no|No|NO|off|Off|OFF)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

append_bridge_calibration_args() {
  BRIDGE_CMD+=(
    --openvr-config "$OPENVR_CONFIG"
    --calibration-cache "$CALIBRATION_CACHE"
    --calibration-cache-output "$CALIBRATION_CACHE_CANDIDATE"
  )
  if is_enabled "$FORCE_RECALIBRATE"; then
    BRIDGE_CMD+=(--force-recalibrate)
  fi
}

resolve_bridge_python() {
  if [[ -n "${HANDSON_BRIDGE_PYTHON:-}" ]]; then
    echo "$HANDSON_BRIDGE_PYTHON"
    return
  fi
  if [[ -x /usr/bin/python3 ]] \
    && { [[ -f /opt/ros/jazzy/setup.bash ]] || [[ -f /opt/ros/humble/setup.bash ]]; }; then
    echo /usr/bin/python3
    return
  fi
  if [[ -n "${HANDSON_PYTHON:-}" && -x "${HANDSON_PYTHON:-}" ]]; then
    echo "$HANDSON_PYTHON"
    return
  fi
  if [[ -x "$HOME/.micromamba/envs/handson/bin/python" ]]; then
    echo "$HOME/.micromamba/envs/handson/bin/python"
    return
  fi
  command -v python3
}

BRIDGE_PYTHON="$(resolve_bridge_python)"

enable_manus_autopair() {
  local py="${HANDSON_PYTHON:-}"
  if [[ -z "$py" ]]; then
    py="$(command -v python3 || true)"
  fi
  if [[ -z "$py" ]]; then
    echo "[WARN] python3 not found; cannot enable MANUS Core autoPair" >&2
    return 0
  fi
  "$py" "$ROOT/scripts/set_manus_core_autopair.py" --quiet || {
    echo "[WARN] Failed to enable MANUS Core autoPair" >&2
  }
}

manus_usb_dongle_visible() {
  command -v lsusb >/dev/null 2>&1 || return 2
  lsusb -d 3325: >/dev/null 2>&1
}

reset_manus_usb_dongle() {
  local python_bin="${PYTHON3:-python3}"

  if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "[ERROR] Cannot reset the MANUS Sensor Dongle because python3 is unavailable." >&2
    return 1
  fi

  echo "[WARN] Resetting the stuck MANUS Sensor Dongle USB session (3325:0049)..." >&2
  if ! "$python_bin" - <<'PY'
import fcntl
from pathlib import Path

matches = []
for device in Path("/sys/bus/usb/devices").iterdir():
    try:
        vendor = (device / "idVendor").read_text(encoding="ascii").strip().lower()
        product = (device / "idProduct").read_text(encoding="ascii").strip().lower()
        if (vendor, product) != ("3325", "0049"):
            continue
        bus = int((device / "busnum").read_text(encoding="ascii").strip())
        number = int((device / "devnum").read_text(encoding="ascii").strip())
        matches.append(Path(f"/dev/bus/usb/{bus:03d}/{number:03d}"))
    except (FileNotFoundError, PermissionError, ValueError):
        continue

if len(matches) != 1:
    raise SystemExit(
        "expected exactly one MANUS Sensor Dongle 3325:0049, "
        f"found {len(matches)}"
    )

# USBDEVFS_RESET = _IO('U', 20). Opening the exact device node read/write
# avoids disturbing the Vive tracker dongles on the same host.
with matches[0].open("rb+", buffering=0) as stream:
    fcntl.ioctl(stream, 0x5514, 0)
PY
  then
    echo "[ERROR] Automatic MANUS Sensor Dongle USB reset failed; reconnect only that Dongle and retry." >&2
    return 1
  fi

  echo "[INFO] MANUS Sensor Dongle USB reset completed; waiting for re-enumeration."
  wait_for_manus_usb_dongle
}

wait_for_manus_usb_dongle() {
  local timeout_seconds="$MANUS_USB_WAIT_TIMEOUT"
  local start_time="$SECONDS"
  local visibility_result=0
  [[ "$timeout_seconds" =~ ^[0-9]+$ ]] || timeout_seconds=15

  if ! command -v lsusb >/dev/null 2>&1; then
    echo "[WARN] lsusb is unavailable; skipping the MANUS Dongle preflight" >&2
    return 0
  fi
  echo "[INFO] Waiting for the MANUS license Dongle (USB vendor 3325, timeout=${timeout_seconds}s)..."
  while true; do
    manus_usb_dongle_visible && visibility_result=0 || visibility_result=$?
    if (( visibility_result == 0 )); then
      echo "[INFO] MANUS license Dongle detected; allowing USB enumeration to settle for ${MANUS_USB_SETTLE_DELAY}s."
      sleep "$MANUS_USB_SETTLE_DELAY"
      return 0
    fi
    if (( SECONDS - start_time >= timeout_seconds )); then
      break
    fi
    sleep 0.2
  done

  echo "[ERROR] MANUS license Dongle was not detected (USB vendor 3325; lsusb -d 3325:)." >&2
  echo "Reconnect the Dongle and wait for its light before starting collection." >&2
  return 1
}

preflight_manus_usb_dongle_for_modes() {
  local mode
  for mode in "$@"; do
    case "$mode" in
      integrated|core)
        wait_for_manus_usb_dongle
        return
        ;;
    esac
  done
  echo "[INFO] MANUS remote mode selected; skipping the local license Dongle preflight."
}

write_env_script() {
  local domain="${ROS_DOMAIN_ID:-71}"
  local ros_setup=""
  local python_bin="${HANDSON_PYTHON:-}"
  local python_prefix=""

  if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    ros_setup="/opt/ros/jazzy/setup.bash"
  elif [[ -f /opt/ros/humble/setup.bash ]]; then
    ros_setup="/opt/ros/humble/setup.bash"
  fi

  if [[ -z "$ros_setup" && -n "$python_bin" && -x "$python_bin" ]]; then
    python_prefix="$(cd "$(dirname "$python_bin")/.." && pwd)"
    if [[ -f "$python_prefix/setup.bash" ]]; then
      ros_setup="$python_prefix/setup.bash"
    fi
  fi

  if [[ -z "$ros_setup" ]] && command -v ros2 >/dev/null 2>&1; then
    python_prefix="$(cd "$(dirname "$(command -v ros2)")/.." && pwd)"
    if [[ -f "$python_prefix/setup.bash" ]]; then
      ros_setup="$python_prefix/setup.bash"
    fi
  fi

  cat >"$ENV_SCRIPT" <<EOF
#!/usr/bin/env bash
if [[ -n "$ros_setup" && -f "$ros_setup" ]]; then
  source "$ros_setup"
else
  echo "[ERROR] ROS2 not found" >&2
  exit 1
fi
source "$WS/install/setup.bash"
export ROS_DOMAIN_ID=$domain
export ROS_LOCALHOST_ONLY="\${HANDSON_ROS_LOCALHOST_ONLY:-1}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="\${HANDSON_ROS_DISCOVERY_RANGE:-LOCALHOST}"
export DISPLAY=\${DISPLAY:-:0}
export PYTHONUNBUFFERED=1
unset PYTHONNOUSERSITE
export ROS_LOG_DIR=\${ROS_LOG_DIR:-$LOG_DIR/ros}
export MANUS_WUJI_ROOT="$MANUS_ROOT"
export WUJI_RETARGETING_ROOT="$RETARGET_ROOT"
mkdir -p "\$ROS_LOG_DIR"
if [[ -d "$RETARGET_ROOT" ]]; then
  export PYTHONPATH="$RETARGET_ROOT:\${PYTHONPATH:-}"
fi
if [[ -d "$OPENVR_INPUT_ROOT" ]]; then
  export PYTHONPATH="$OPENVR_INPUT_ROOT:\${PYTHONPATH:-}"
fi
EOF
  chmod +x "$ENV_SCRIPT"
}

prepare_openvr_config() {
  if [[ ! -f "$TRACKER_CONFIG_SCRIPT" ]]; then
    echo "[ERROR] Missing OpenVR tracker binding helper: $TRACKER_CONFIG_SCRIPT" >&2
    exit 1
  fi
  if [[ ! -f "$OPENVR_TEMPLATE" ]]; then
    echo "[ERROR] Missing OpenVR config template: $OPENVR_TEMPLATE" >&2
    exit 1
  fi
  require_vrserver

  local env_script_q
  local -a command=(
    "$BRIDGE_PYTHON" "$TRACKER_CONFIG_SCRIPT"
    --template "$OPENVR_TEMPLATE"
    --output "$OPENVR_CONFIG"
  )
  printf -v env_script_q "%q" "$ENV_SCRIPT"
  bash -lc "source $env_script_q && exec ${command[*]@Q}"
}

require_vrserver() {
  if pgrep -x vrserver >/dev/null 2>&1; then
    return 0
  fi
  echo "[ERROR] SteamVR (vrserver) is required for tracker mode; start SteamVR and retry" >&2
  return 1
}

verify_started_process() {
  local name="$1"
  local pid="$2"
  local log_file="$3"
  if process_alive "$pid" && pid_matches_component "$name" "$pid"; then
    return 0
  fi
  echo "[ERROR] $name exited during startup" >&2
  if [[ -f "$log_file" ]]; then
    tail -n 80 "$log_file" >&2 || true
  fi
  return 1
}

manus_publisher_has_license_error() {
  [[ -r "$LOG_DIR/publisher.log" ]] || return 1
  grep -Eiq \
    "No compatible license found|don't have a valid SDK( Integrated)? license|Please connect a valid license key" \
    "$LOG_DIR/publisher.log"
}

manus_publisher_has_initialization_error() {
  [[ -r "$LOG_DIR/publisher.log" ]] || return 1
  grep -Eiq "Failed to initialize the SDK" "$LOG_DIR/publisher.log"
}

manus_publisher_has_fatal_error() {
  manus_publisher_has_initialization_error && return 0
  manus_publisher_dongle_initialization_failed && return 0
  if manus_publisher_has_license_error; then
    if is_enabled "$IGNORE_MANUS_LICENSE"; then
      return 1
    fi
    return 0
  fi
  return 1
}

report_manus_publisher_fatal_error() {
  if manus_publisher_has_initialization_error; then
    echo "[ERROR] MANUS hand input cannot start because the publisher failed to initialize the SDK." >&2
  elif manus_publisher_dongle_initialization_failed; then
    echo "[ERROR] MANUS Sensor Dongle initialization failed after automatic USB recovery." >&2
  else
    echo "[ERROR] MANUS hand input cannot start because the SDK license is unavailable and license enforcement is enabled." >&2
  fi
  echo "Reconnect the MANUS Sensor Dongle, confirm both gloves are online, then retry." >&2
  if [[ -r "$LOG_DIR/publisher.log" ]]; then
    tail -n 80 "$LOG_DIR/publisher.log" >&2 || true
  fi
}

latest_bridge_hand_status_line() {
  [[ -r "$LOG_DIR/bridge.log" ]] || return 1
  grep -E "/hand_input msgs=" "$LOG_DIR/bridge.log" | tail -n 1
}

hand_status_line_has_nonzero_sent() {
  local status_line="$1"
  local side="$2"
  grep -Eq "sent\\([^)]*${side}=[1-9][0-9]*([[:space:]]|\\))" <<<"$status_line"
}

hand_status_line_is_recent() {
  local status_line="$1"
  local updated_at now
  if [[ "$status_line" =~ status_unix_sec=([0-9]+) ]]; then
    updated_at="${BASH_REMATCH[1]}"
  else
    # Compatibility with older bridge logs. New bridges timestamp the status
    # itself so unrelated log writes cannot make an old hand status fresh.
    updated_at="$(stat -c %Y "$LOG_DIR/bridge.log" 2>/dev/null)" || return 1
  fi
  now="$(date +%s)"
  (( now - updated_at >= -1 && now - updated_at <= 5 ))
}

manus_status_counter() {
  local status_line="$1" key="$2"
  local pattern="(^|[[:space:](])${key}=([0-9]+)"
  if [[ "$status_line" =~ $pattern ]]; then
    echo "${BASH_REMATCH[2]}"
  else
    echo 0
  fi
}

manus_hand_status_progressed() {
  local previous="$1" current="$2" side
  [[ -n "$previous" ]] || return 1
  (( $(manus_status_counter "$current" msgs) > $(manus_status_counter "$previous" msgs) )) || return 1
  local progressed=0 required=1
  local -a sides=("$HAND")
  if [[ "$HAND" == both ]]; then
    sides=(right left)
    if is_enabled "$STRICT_MANUS_HANDS"; then required=2; fi
  fi
  for side in "${sides[@]}"; do
    if (( $(manus_status_counter "$current" "$side") > $(manus_status_counter "$previous" "$side") )); then
      progressed=$((progressed + 1))
    fi
  done
  (( progressed >= required ))
}

print_startup_calibration_header_once() {
  if (( STARTUP_CALIBRATION_HEADER_PRINTED )); then
    return
  fi
  STARTUP_CALIBRATION_HEADER_PRINTED=1
  echo ""
  echo "Startup calibration monitor:"
  echo "  Cached mode: hold the neutral tabletop anchor still for a short safety check."
  echo "  Guided mode: anchor still -> forward(+X) -> left(+Y) -> up(+Z)."
  echo "  Middle-axis mode: anchor still -> rotate around the middle-finger axis."
  echo "  HandsOn collection starts after calibration completes."
  echo "  Log: $LOG_DIR/bridge.log"
}

print_new_startup_calibration_lines() {
  local -a lines=()
  local line_count index
  [[ -f "$LOG_DIR/bridge.log" ]] || return

  mapfile -t lines < <(
    grep -E "Startup calibration|Guided startup|Guided calibration|Pair binding captured" \
      "$LOG_DIR/bridge.log" || true
  )
  line_count="${#lines[@]}"
  if (( line_count < STARTUP_CALIBRATION_PRINTED_LINES )); then
    STARTUP_CALIBRATION_PRINTED_LINES=0
  fi
  for ((index = STARTUP_CALIBRATION_PRINTED_LINES; index < line_count; index++)); do
    echo "${lines[$index]}"
  done
  STARTUP_CALIBRATION_PRINTED_LINES="$line_count"
}

publisher_has_manus_glove_side() {
  local side="$1"
  [[ -r "$LOG_DIR/publisher.log" ]] || return 1
  grep -Fqi "Calibration loaded successfully for ${side} glove" "$LOG_DIR/publisher.log"
}

report_missing_requested_manus_hands() {
  local right_detected=false
  local left_detected=false
  local -a missing=()
  local detected="none"

  if publisher_has_manus_glove_side Right; then
    right_detected=true
  fi
  if publisher_has_manus_glove_side Left; then
    left_detected=true
  fi
  if $right_detected && $left_detected; then
    detected="Right,Left"
  elif $right_detected; then
    detected="Right"
  elif $left_detected; then
    detected="Left"
  fi

  case "$HAND" in
    right)
      $right_detected || missing+=(Right)
      ;;
    left)
      $left_detected || missing+=(Left)
      ;;
    both)
      $right_detected || missing+=(Right)
      $left_detected || missing+=(Left)
      ;;
  esac

  if (( ${#missing[@]} )); then
    local missing_csv
    missing_csv="$(IFS=,; echo "${missing[*]}")"
    echo "[ERROR] Requested MANUS glove(s) not detected: ${missing_csv} (detected: ${detected})." >&2
    echo "Power on or reconnect the missing glove(s), then retry the device stack." >&2
  fi
}

requested_manus_hands_ready() {
  local status_line="${1:-}"
  if [[ -z "$status_line" ]]; then
    status_line="$(latest_bridge_hand_status_line)" || return 1
  fi
  hand_status_line_is_recent "$status_line" || return 1
  grep -Eq "hand=fresh([[:space:]]|$)" <<<"$status_line" || return 1

  if ! is_enabled "$STRICT_MANUS_HANDS"; then
    case "$HAND" in
      right)
        hand_status_line_has_nonzero_sent "$status_line" right
        ;;
      left)
        hand_status_line_has_nonzero_sent "$status_line" left
        ;;
      both)
        hand_status_line_has_nonzero_sent "$status_line" right \
          || hand_status_line_has_nonzero_sent "$status_line" left
        ;;
      *)
        return 1
        ;;
    esac
    return
  fi

  case "$HAND" in
    right)
      hand_status_line_has_nonzero_sent "$status_line" right
      ;;
    left)
      hand_status_line_has_nonzero_sent "$status_line" left
      ;;
    both)
      hand_status_line_has_nonzero_sent "$status_line" right \
        && hand_status_line_has_nonzero_sent "$status_line" left
      ;;
    *)
      return 1
      ;;
  esac
}

report_manus_readiness_diagnosis() {
  local status_line
  echo "MANUS readiness diagnosis:" >&2
  report_missing_requested_manus_hands
  status_line="$(latest_bridge_hand_status_line)" || status_line=""
  if [[ -z "$status_line" ]]; then
    echo "[ERROR] Bridge has not produced a hand-stream status; inspect bridge initialization and callback scheduling." >&2
  elif ! hand_status_line_is_recent "$status_line"; then
    echo "[ERROR] Bridge status stopped updating; inspect executor scheduling, retargeting and raw-recording callback latency." >&2
  elif [[ "$status_line" != *"hand=fresh"* ]]; then
    if [[ "$status_line" =~ raw[01]_age_sec=0\.[0-9]+ ]]; then
      echo "[ERROR] Raw MANUS messages are reaching the bridge, but /hand_input is missing or stale. Check manus_input publishing and bridge input callback scheduling." >&2
    else
      echo "[ERROR] No fresh /hand_input at the bridge. Check publisher -> manus_input -> bridge traffic and the raw receive counters." >&2
    fi
  else
    echo "[ERROR] Requested hand commands did not show sustained progress. Compare per-hand sent counters and retargeting errors." >&2
  fi
  echo "Bridge timing/raw counters: ${status_line:-unavailable}" >&2
  echo "A hand-stream timeout alone does not establish a Sensor Dongle or SDK license fault." >&2
}

wait_for_manus_hand_stream() {
  local bridge_pid="$1"
  local monitor_calibration="${2:-false}"
  local timeout_seconds="$MANUS_HAND_WAIT_TIMEOUT"
  local start_time="$SECONDS"
  local status_line previous_status_line=""
  [[ "$timeout_seconds" =~ ^[0-9]+$ ]] || timeout_seconds=20

  if [[ "$monitor_calibration" == true ]]; then
    print_startup_calibration_header_once
  fi
  echo "[INFO] Waiting for non-zero MANUS finger commands (hand=$HAND, timeout=${timeout_seconds}s)..."
  while (( SECONDS - start_time < timeout_seconds )); do
    if [[ "$monitor_calibration" == true ]]; then
      print_new_startup_calibration_lines
    fi
    if manus_publisher_has_fatal_error; then
      report_manus_publisher_fatal_error
      return 1
    fi
    if ! process_alive "$bridge_pid" || ! pid_matches_component bridge "$bridge_pid"; then
      echo "[ERROR] HandsOn bridge exited while waiting for MANUS hand input" >&2
      tail -n 80 "$LOG_DIR/bridge.log" >&2 || true
      return 1
    fi
    status_line="$(latest_bridge_hand_status_line)" || status_line=""
    if [[ -n "$status_line" ]] && requested_manus_hands_ready "$status_line"; then
      if manus_hand_status_progressed "$previous_status_line" "$status_line"; then
        echo "[INFO] MANUS finger commands are live and progressing for hand=$HAND."
        return 0
      fi
      previous_status_line="$status_line"
    else
      previous_status_line=""
    fi
    sleep 0.2
  done

  if manus_publisher_has_fatal_error; then
    report_manus_publisher_fatal_error
    return 1
  fi
  status_line="$(latest_bridge_hand_status_line)" || status_line=""
  if [[ -n "$status_line" ]] && requested_manus_hands_ready "$status_line" \
    && manus_hand_status_progressed "$previous_status_line" "$status_line"; then
    echo "[INFO] MANUS finger commands are live and progressing for hand=$HAND."
    return 0
  fi
  if [[ "$monitor_calibration" == true ]]; then
    print_new_startup_calibration_lines
  fi

  echo "[ERROR] No non-zero MANUS finger commands arrived for hand=$HAND within ${timeout_seconds}s." >&2
  echo "Publisher log: $LOG_DIR/publisher.log" >&2
  tail -n 40 "$LOG_DIR/publisher.log" >&2 || true
  echo "Input log: $LOG_DIR/input.log" >&2
  tail -n 40 "$LOG_DIR/input.log" >&2 || true
  echo "Bridge log: $LOG_DIR/bridge.log" >&2
  tail -n 40 "$LOG_DIR/bridge.log" >&2 || true
  report_manus_readiness_diagnosis
  return 1
}

component_marker() {
  local name="$1"
  case "$name" in
    publisher) echo "manus_data_publisher" ;;
    input) echo "manus_input" ;;
    openvr|openvr_input) echo "openvr_input" ;;
    bridge) echo "manus_wuji_ros2_bridge.py" ;;
    *) return 1 ;;
  esac
}

pid_cmdline() {
  local pid="$1"
  local path="/proc/$pid/cmdline"
  [[ -r "$path" ]] || return 1
  tr '\0' ' ' <"$path"
}

process_alive() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  (( pid > 1 )) || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  if [[ -r "/proc/$pid/stat" ]] && grep -qE '^.*\) Z ' "/proc/$pid/stat"; then
    return 1
  fi
  return 0
}

pid_matches_component() {
  local name="$1"
  local pid="$2"
  local marker cmdline owner
  marker="$(component_marker "$name")" || return 1
  process_alive "$pid" || return 1
  if [[ -d "/proc/$pid" ]]; then
    owner="$(stat -c '%u' "/proc/$pid" 2>/dev/null || true)"
    [[ -z "$owner" || "$owner" == "$(id -u)" ]] || return 1
  fi
  cmdline="$(pid_cmdline "$pid" 2>/dev/null || true)"
  [[ -n "$cmdline" && "$cmdline" == *"$marker"* ]]
}

read_component_pid() {
  local name="$1"
  local file="$PID_DIR/$name.pid"
  local pid=""
  [[ -r "$file" ]] || return 1
  read -r pid _ <"$file" || return 1
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  (( pid > 1 )) || return 1
  echo "$pid"
}

write_component_pid() {
  local name="$1"
  local pid="$2"
  local temporary="$PID_DIR/.$name.pid.$$"
  printf '%s\n' "$pid" >"$temporary"
  mv -f "$temporary" "$PID_DIR/$name.pid"
}

wait_for_process_exit() {
  local pid="$1"
  local timeout_seconds="$2"
  local iterations index
  [[ "$timeout_seconds" =~ ^[0-9]+$ ]] || timeout_seconds=2
  iterations=$(( timeout_seconds * 10 ))
  for ((index = 0; index < iterations; index += 1)); do
    process_alive "$pid" || return 0
    sleep 0.1
  done
  ! process_alive "$pid"
}

signal_component_pid() {
  local signal_name="$1"
  local pid="$2"
  local pgid=""
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]' || true)"
  if [[ "$pgid" == "$pid" ]]; then
    kill -s "$signal_name" -- "-$pid" 2>/dev/null
  else
    kill -s "$signal_name" "$pid" 2>/dev/null
  fi
}

stop_component() {
  local name="$1"
  local file="$PID_DIR/$name.pid"
  local pid signal_name timeout_seconds

  if [[ ! -e "$file" ]]; then
    return 0
  fi
  if ! pid="$(read_component_pid "$name")"; then
    echo "[WARN] Ignoring invalid $name PID file: $file" >&2
    rm -f "$file"
    return 0
  fi
  if ! process_alive "$pid"; then
    rm -f "$file"
    echo "$name: already stopped (stale pid $pid)"
    return 0
  fi
  if ! pid_matches_component "$name" "$pid"; then
    echo "[WARN] Refusing to signal pid $pid from $file: process identity does not match $name" >&2
    rm -f "$file"
    return 0
  fi

  while read -r signal_name timeout_seconds; do
    if ! process_alive "$pid"; then
      rm -f "$file"
      echo "Stopped $name (pid $pid)"
      return 0
    fi
    if ! pid_matches_component "$name" "$pid"; then
      echo "[WARN] $name pid $pid exited or was reused while stopping; no further signal sent" >&2
      rm -f "$file"
      return 0
    fi
    signal_component_pid "$signal_name" "$pid" || true
    if wait_for_process_exit "$pid" "$timeout_seconds"; then
      rm -f "$file"
      echo "Stopped $name with SIG$signal_name (pid $pid)"
      return 0
    fi
  done <<EOF
INT $STOP_INT_TIMEOUT
TERM $STOP_TERM_TIMEOUT
KILL $STOP_KILL_TIMEOUT
EOF

  echo "[ERROR] $name pid $pid is still alive after SIGINT/SIGTERM/SIGKILL" >&2
  return 1
}

stop_components() {
  local result=0
  local name
  for name in bridge openvr input publisher; do
    stop_component "$name" || result=1
  done
  if (( result == 0 )); then
    rm -f "$STACK_STATE_FILE"
  fi
  return "$result"
}

component_running() {
  local name="$1"
  local pid
  pid="$(read_component_pid "$name" 2>/dev/null)" || return 1
  pid_matches_component "$name" "$pid"
}

require_component_running() {
  local name="$1"
  if component_running "$name"; then
    return 0
  fi
  echo "[ERROR] $name is not running; use a full Mode A start instead" >&2
  return 1
}

write_stack_state() {
  local tracker_required="$1"
  local temporary="$PID_DIR/.stack.env.$$"
  {
    printf 'HAND=%q\n' "$HAND"
    printf 'TRACKER_REQUIRED=%q\n' "$tracker_required"
    printf 'TRACKER_MODE=%q\n' "$TRACKER_MODE"
    printf 'HAND_TYPE=%q\n' "$HAND_TYPE"
    printf 'STRICT_MANUS_HANDS=%q\n' "$STRICT_MANUS_HANDS"
  } >"$temporary"
  mv -f "$temporary" "$STACK_STATE_FILE"
}

stack_requires_openvr() {
  if [[ -r "$STACK_STATE_FILE" ]] && grep -q '^TRACKER_REQUIRED=1$' "$STACK_STATE_FILE"; then
    return 0
  fi
  [[ -e "$PID_DIR/openvr.pid" ]]
}

saved_stack_hand() {
  local saved=""
  if [[ -r "$STACK_STATE_FILE" ]]; then
    saved="$(sed -n -E 's/^HAND=(left|right|both)$/\1/p' "$STACK_STATE_FILE" | head -n 1)"
  fi
  echo "${saved:-both}"
}

component_process_pids() {
  local name="$1"
  local marker pid
  marker="$(component_marker "$name")" || return 1
  while read -r pid; do
    [[ "$pid" == "$$" ]] && continue
    if pid_matches_component "$name" "$pid"; then
      echo "$pid"
    fi
  done < <(pgrep -f -- "$marker" 2>/dev/null || true)
}

replace_existing_components() {
  local name
  local -a remaining=()
  if ! stop_components; then
    echo "[ERROR] Could not stop the existing Mode A components; refusing to launch duplicates" >&2
    return 1
  fi
  for name in publisher input openvr bridge; do
    mapfile -t remaining < <(component_process_pids "$name")
    if (( ${#remaining[@]} > 0 )); then
      echo "[ERROR] $name process(es) still running after stop: ${remaining[*]}; refusing to launch duplicates" >&2
      return 1
    fi
  done
  return 0
}

cleanup_transient_units() {
  if [[ -x "$INDEPENDENT_LAUNCHER" ]]; then
    "$INDEPENDENT_LAUNCHER" cleanup >/dev/null 2>&1 || true
  fi
  rm -f "$STACK_SCOPE_FILE" "$OPENVR_SCOPE_FILE"
}

stop_managed() {
  local result=0
  stop_components || result=1
  cleanup_transient_units
  return "$result"
}

start_with_cleanup() {
  local result=0
  rm -f "$STARTUP_COMPONENTS_MARKER"
  if ( start ); then
    rm -f "$STARTUP_COMPONENTS_MARKER"
    return 0
  else
    result=$?
  fi
  if [[ -e "$STARTUP_COMPONENTS_MARKER" ]]; then
    echo "[ERROR] Mode A startup failed; cleaning all partially started components" >&2
    stop_components >/dev/null 2>&1 || true
    rm -f "$STACK_STATE_FILE"
  else
    echo "[ERROR] Mode A preflight failed; the existing device stack was left untouched" >&2
  fi
  rm -f "$STARTUP_COMPONENTS_MARKER"
  return "$result"
}

recover_openvr_with_cleanup() {
  local result=0
  if ( recover_openvr ); then
    return 0
  else
    result=$?
  fi
  if component_running openvr; then
    echo "[WARN] OpenVR recovery timed out, but its self-healing client remains running and will continue waiting for trackers" >&2
  else
    echo "[ERROR] OpenVR recovery failed and the replacement client exited" >&2
    rm -f "$PID_DIR/openvr.pid"
  fi
  return "$result"
}

write_calibration_receipt() {
  local path="${HANDSON_CALIBRATION_RECEIPT_PATH:-}"
  local token="${HANDSON_CALIBRATION_RECEIPT_TOKEN:-}"
  local temporary=""

  if [[ -z "$path" && -z "$token" ]]; then
    return 0
  fi
  if [[ -z "$path" || -z "$token" ]]; then
    echo "[ERROR] Both HANDSON_CALIBRATION_RECEIPT_PATH and HANDSON_CALIBRATION_RECEIPT_TOKEN are required" >&2
    return 1
  fi
  if ! mkdir -p -- "$(dirname -- "$path")"; then
    echo "[ERROR] Could not create calibration receipt directory for $path" >&2
    return 1
  fi
  if ! temporary="$(mktemp "${path}.tmp.XXXXXX")"; then
    echo "[ERROR] Could not create temporary calibration receipt beside $path" >&2
    return 1
  fi
  if ! printf '%s' "$token" >"$temporary"; then
    rm -f -- "$temporary"
    echo "[ERROR] Could not write calibration receipt $path" >&2
    return 1
  fi
  if ! mv -f -- "$temporary" "$path"; then
    rm -f -- "$temporary"
    echo "[ERROR] Could not publish calibration receipt $path" >&2
    return 1
  fi
  echo "[INFO] Calibration startup receipt written: $path"
}

discard_calibration_cache_candidate() {
  rm -f -- "$CALIBRATION_CACHE_CANDIDATE"
}

discard_calibration_receipt() {
  local path="${HANDSON_CALIBRATION_RECEIPT_PATH:-}"
  if [[ -n "$path" ]]; then
    rm -f -- "$path"
  fi
}

commit_calibration_cache_candidate() {
  if [[ ! -e "$CALIBRATION_CACHE_CANDIDATE" ]]; then
    if [[ -r "$LOG_DIR/bridge.log" ]] \
      && grep -q "Startup calibration ready: source=guided" "$LOG_DIR/bridge.log"; then
      echo "[ERROR] Guided calibration completed, but no cache candidate was saved" >&2
      return 1
    fi
    return 0
  fi
  if [[ "$(realpath -m "$CALIBRATION_CACHE_CANDIDATE")" == "$(realpath -m "$CALIBRATION_CACHE")" ]]; then
    echo "[ERROR] Calibration cache candidate must differ from the permanent cache" >&2
    return 1
  fi
  if ! mkdir -p -- "$(dirname -- "$CALIBRATION_CACHE")"; then
    echo "[ERROR] Could not create calibration cache directory for $CALIBRATION_CACHE" >&2
    return 1
  fi
  if ! mv -f -- "$CALIBRATION_CACHE_CANDIDATE" "$CALIBRATION_CACHE"; then
    echo "[ERROR] Could not commit startup calibration cache to $CALIBRATION_CACHE" >&2
    return 1
  fi
  echo "[INFO] Startup calibration cache saved: $CALIBRATION_CACHE"
}

start_direct_with_receipt() {
  local result=0
  discard_calibration_cache_candidate
  discard_calibration_receipt
  if start_with_cleanup; then
    result=0
  else
    result=$?
    discard_calibration_cache_candidate
    return "$result"
  fi
  if ! status >/dev/null 2>&1; then
    echo "[ERROR] Mode A startup completed, but final component status is unhealthy" >&2
    discard_calibration_cache_candidate
    stop_components >/dev/null 2>&1 || true
    return 1
  fi
  if ! write_calibration_receipt; then
    discard_calibration_cache_candidate
    discard_calibration_receipt
    stop_components >/dev/null 2>&1 || true
    return 1
  fi
  if ! commit_calibration_cache_candidate; then
    discard_calibration_cache_candidate
    discard_calibration_receipt
    stop_components >/dev/null 2>&1 || true
    return 1
  fi
}

run_independent() {
  local action="$1"
  shift
  if ! is_enabled "${HANDSON_MODE_A_DISABLE_SYSTEMD:-0}" && [[ -x "$INDEPENDENT_LAUNCHER" ]]; then
    exec "$INDEPENDENT_LAUNCHER" "$action" "$@"
  fi
  case "$action" in
    start) start_direct_with_receipt ;;
    recover-openvr) recover_openvr_with_cleanup ;;
    *) return 1 ;;
  esac
}

wait_for_startup_calibration() {
  local bridge_pid="$1"
  local start_time="$SECONDS"

  print_startup_calibration_header_once

  while (( SECONDS - start_time < CALIBRATION_WAIT_TIMEOUT )); do
    if ! kill -0 "$bridge_pid" 2>/dev/null; then
      echo "[ERROR] HandsOn bridge exited during startup calibration" >&2
      tail -n 80 "$LOG_DIR/bridge.log" >&2 || true
      exit 1
    fi

    if [[ -f "$LOG_DIR/bridge.log" ]]; then
      print_new_startup_calibration_lines

      if grep -Eq "Startup calibration ready: source=(cache|guided)|(Startup calibration complete|Guided startup calibration complete)" "$LOG_DIR/bridge.log"; then
        echo "Startup calibration done. Open the MuJoCo window next, then press ';'."
        return
      fi

      if grep -q "Bridge ready" "$LOG_DIR/bridge.log" \
        && ! grep -Eq "(Startup calibration enabled|Guided startup calibration enabled|Startup calibration cache validation)" "$LOG_DIR/bridge.log" \
        && (( SECONDS - start_time >= 8 )); then
        echo "[WARN] Bridge started, but startup calibration is not enabled in the mapping config."
        return
      fi
    fi

    sleep 1
  done

  echo "[ERROR] Startup calibration did not complete within ${CALIBRATION_WAIT_TIMEOUT}s" >&2
  echo "Keep this open to inspect live logs:" >&2
  echo "  tail -f $LOG_DIR/bridge.log" >&2
  exit 1
}

check_wuji_retargeting() {
  if [[ "$HAND_TYPE" == "wuji2" ]]; then
    local bridge_python_q
    printf -v bridge_python_q "%q" "$BRIDGE_PYTHON"
    if ! bash -lc "source '$ENV_SCRIPT' && $bridge_python_q -c 'import wuji_sdk'"; then
      echo "[ERROR] Wuji Hand 2 requires wuji-sdk in the bridge Python environment" >&2
      echo "  Install: $BRIDGE_PYTHON -m pip install -r $ROOT/teleoperation/manus/requirements-wuji2.txt" >&2
      exit 1
    fi
    return 0
  fi

  if [[ ! -d "$RETARGET_ROOT/wuji_retargeting" ]]; then
    echo "[ERROR] Missing wuji_retargeting package under: $RETARGET_ROOT" >&2
    echo "  Expected in-repo copy: $VENDOR_ROOT/wuji-retargeting" >&2
    echo "  Or set WUJI_RETARGETING_ROOT to another wuji-retargeting checkout." >&2
    exit 1
  fi

  local bridge_python_q
  printf -v bridge_python_q "%q" "$BRIDGE_PYTHON"
  if ! bash -lc "source '$ENV_SCRIPT' && $bridge_python_q -c 'import wuji_retargeting'"; then
    echo "[ERROR] wuji_retargeting exists but is not importable in the launch environment" >&2
    echo "  WUJI_RETARGETING_ROOT=$RETARGET_ROOT" >&2
    echo "  Bridge Python: $BRIDGE_PYTHON" >&2
    echo "  Env script: $ENV_SCRIPT" >&2
    exit 1
  fi
}

write_manus_input_config() {
  local include_right=false
  local include_left=false
  local swap_left_right=false
  local input_source="${HANDSON_MANUS_INPUT_SOURCE:-}"

  case "$HAND" in
    right)
      include_right=true
      ;;
    left)
      include_left=true
      ;;
    both)
      include_right=true
      include_left=true
      ;;
  esac

  if [[ -n "$input_source" ]]; then
    include_right=false
    include_left=false
    case "$input_source" in
      right)
        include_right=true
        ;;
      left)
        include_left=true
        ;;
      both)
        include_right=true
        include_left=true
        ;;
      *)
        echo "[ERROR] HANDSON_MANUS_INPUT_SOURCE must be right, left, or both (got: $input_source)" >&2
        exit 1
        ;;
    esac
  fi

  if is_enabled "$SWAP_HANDS"; then
    swap_left_right=true
  fi

  local allow_missing_hand=true
  local require_all_requested_updates=false
  local max_hand_update_age_sec=0.0
  if is_enabled "$STRICT_MANUS_HANDS"; then
    allow_missing_hand=false
    require_all_requested_updates=true
    max_hand_update_age_sec=0.10
  fi

  cat >"$MANUS_INPUT_CONFIG" <<EOF
# Generated by mode_a_handson.sh.
publish_rate_hz: 120.0
publish_hand_topic: /hand_input
include_right_hand: $include_right
include_left_hand: $include_left
allow_missing_hand: $allow_missing_hand
require_all_requested_updates: $require_all_requested_updates
max_hand_update_age_sec: $max_hand_update_age_sec
swap_left_right: $swap_left_right
left_glove_id: 0
right_glove_id: 1
EOF

  echo "[INFO] MANUS left/right input swap: $swap_left_right"
  if ! is_enabled "$STRICT_MANUS_HANDS"; then
    echo "[WARN] MANUS strict dual-hand freshness is disabled; missing hands use zero slices and live hands continue. Set HANDSON_STRICT_MANUS_HANDS=1 to require every requested hand."
  fi
  if [[ -n "$input_source" ]]; then
    echo "[INFO] Using MANUS input source override: $input_source"
  fi
}

normalize_manus_connection_mode() {
  local mode="${1,,}"
  case "$mode" in
    integrated|direct)
      echo "integrated"
      ;;
    core|local)
      echo "core"
      ;;
    remote)
      echo "remote"
      ;;
    *)
      echo "[ERROR] MANUS connection mode must be integrated, core, or remote (got: $1)" >&2
      return 1
      ;;
  esac
}

manus_connection_modes() {
  local configured="$MANUS_CONNECTION_MODE"
  local raw mode
  local -a modes=()
  if [[ -z "$configured" ]]; then
    configured="$MANUS_CONNECTION_MODES"
  fi
  if [[ -z "$configured" || "${configured,,}" == "auto" ]]; then
    configured="integrated"
    if [[ -x "$WS/install/manus_ros2/lib/manus_ros2/manus_data_publisher_core" ]]; then
      configured="integrated,core"
    fi
  fi
  configured="${configured//,/ }"
  for raw in $configured; do
    mode="$(normalize_manus_connection_mode "$raw")" || return 1
    modes+=("$mode")
  done
  if (( ${#modes[@]} == 0 )); then
    echo "[ERROR] No MANUS connection modes configured" >&2
    return 1
  fi
  printf '%s\n' "${modes[@]}"
}

archive_startup_attempt_logs() {
  local mode="$1"
  local attempt="$2"
  local archive_dir="$LOG_DIR/archive"
  local name
  mkdir -p "$archive_dir"
  for name in publisher input openvr bridge; do
    if [[ -f "$LOG_DIR/$name.log" ]]; then
      mv -f "$LOG_DIR/$name.log" "$archive_dir/${name}-${attempt}-${mode}.log"
    fi
  done
}

detect_manus_left_glove_ids() {
  python3 - <<'PY'
from pathlib import Path
import json

ids = []
for path in Path.home().joinpath(".config", "Manus", "Core 3").glob("*Calibrations.json"):
    try:
        raw = path.read_bytes()
        start = raw.find(b"{")
        if start < 0:
            continue
        data = json.loads(raw[start:])
    except Exception:
        continue
    profiles = data.get("gloveProfiles", {})
    if not isinstance(profiles, dict):
        continue
    for glove_id, profile in profiles.items():
        if not isinstance(profile, dict):
            continue
        if str(profile.get("side", "")).lower() != "left":
            continue
        if isinstance(glove_id, str) and glove_id.startswith("0x") and glove_id not in ids:
            ids.append(glove_id)
print(",".join(ids))
PY
}

export_manus_pairing_candidates() {
  if [[ -n "${HANDSON_MANUS_PAIR_LEFT_GLOVE_IDS:-}" ]]; then
    export HANDSON_MANUS_PAIR_LEFT_GLOVE_IDS
    echo "[INFO] MANUS left glove pairing candidates: $HANDSON_MANUS_PAIR_LEFT_GLOVE_IDS"
    return 0
  fi

  local left_ids
  left_ids="$(detect_manus_left_glove_ids || true)"
  if [[ -n "$left_ids" ]]; then
    export HANDSON_MANUS_PAIR_LEFT_GLOVE_IDS="$left_ids"
    echo "[INFO] MANUS left glove pairing candidates from calibration history: $left_ids"
  fi
}

manus_publisher_executable() {
  local mode="$1"
  case "$mode" in
    integrated)
      echo "manus_data_publisher"
      ;;
    core|remote)
      echo "manus_data_publisher_core"
      ;;
    *)
      echo "manus_data_publisher"
      ;;
  esac
}

manus_publisher_license_ready() {
  local mode="$1"
  [[ -r "$LOG_DIR/publisher.log" ]] || return 1
  case "$mode" in
    integrated)
      grep -Eq \
        "MANUS landscape: license\(sdk=(true|false) integrated=true|DeviceManager: Second stage initialisation succeeded|is connected as .*Dongle|Calibration loaded successfully for (Right|Left) glove" \
        "$LOG_DIR/publisher.log"
      ;;
    core|remote)
      grep -Eq "MANUS landscape: license\(sdk=true" "$LOG_DIR/publisher.log"
      ;;
    *)
      return 1
      ;;
  esac
}

manus_publisher_dongle_initialization_failed() {
  [[ -r "$LOG_DIR/publisher.log" ]] || return 1
  grep -Eiq "DeviceManager: Second stage initialisation failed" \
    "$LOG_DIR/publisher.log"
}

manus_publisher_license_warning_ignored() {
  is_enabled "$IGNORE_MANUS_LICENSE" || return 1
  manus_publisher_has_license_error || return 1
  manus_publisher_license_ready integrated
}

report_ignored_manus_license_warning() {
  local reason="${1:-the publisher reported an SDK license warning}"
  echo "[WARN] Ignoring the MANUS SDK license gate because HANDSON_IGNORE_MANUS_LICENSE=$IGNORE_MANUS_LICENSE ($reason)." >&2
  echo "[WARN] Startup will still require live, non-zero MANUS finger commands." >&2
}

wait_for_manus_publisher_license() {
  local publisher_pid="$1"
  local mode="$2"
  local timeout_seconds="$MANUS_PUBLISHER_START_TIMEOUT"
  local start_time="$SECONDS"
  [[ "$timeout_seconds" =~ ^[0-9]+$ ]] || timeout_seconds=8

  while (( SECONDS - start_time < timeout_seconds )); do
    if manus_publisher_has_fatal_error; then
      return 2
    fi
    # The setsid/bash launcher may not have exec'd the ROS publisher yet.
    # During that short transition, liveness is authoritative; verify the
    # final component identity only after the publisher reports its license.
    if ! process_alive "$publisher_pid"; then
      return 1
    fi
    if manus_publisher_license_warning_ignored; then
      report_ignored_manus_license_warning "publisher warning"
      return 0
    fi
    if manus_publisher_license_ready "$mode"; then
      return 0
    fi
    sleep 0.2
  done
  if manus_publisher_license_warning_ignored; then
    report_ignored_manus_license_warning "publisher warning"
    return 0
  fi
  if manus_publisher_license_ready "$mode"; then
    return 0
  fi
  if is_enabled "$IGNORE_MANUS_LICENSE" \
    && [[ "$mode" != "integrated" ]] \
    && process_alive "$publisher_pid"; then
    report_ignored_manus_license_warning \
      "no usable-license confirmation arrived within ${timeout_seconds}s"
    return 0
  fi
  return 1
}

archive_manus_publisher_attempt_log() {
  local mode="$1"
  local attempt="$2"
  local archive_dir="$LOG_DIR/archive"
  [[ -f "$LOG_DIR/publisher.log" ]] || return 0
  mkdir -p "$archive_dir"
  mv -f "$LOG_DIR/publisher.log" \
    "$archive_dir/publisher-license-${mode}-${attempt}-$(date +%Y%m%d-%H%M%S).log"
}

start_manus_publisher_once() {
  local manus_mode="$1"
  local publisher_executable="$2"
  setsid bash -lc "source '$ENV_SCRIPT' && cd '$MANUS_ROOT' && export HANDSON_MANUS_CONNECTION_MODE='$manus_mode' && exec ros2 run manus_ros2 '$publisher_executable'" \
    >"$LOG_DIR/publisher.log" 2>&1 </dev/null &
  local publisher_pid=$!
  write_component_pid publisher "$publisher_pid"
  wait_for_manus_publisher_license "$publisher_pid" "$manus_mode" \
    && verify_started_process publisher "$publisher_pid" "$LOG_DIR/publisher.log"
}

start_manus_publisher_with_retries() {
  local manus_mode="$1"
  local publisher_executable="$2"
  local attempts="$MANUS_PUBLISHER_START_ATTEMPTS"
  local attempt result=1 reset_dongle=false
  [[ "$attempts" =~ ^[0-9]+$ ]] || attempts=3
  (( attempts > 0 )) || attempts=1

  for ((attempt = 1; attempt <= attempts; attempt += 1)); do
    echo "[INFO] Starting MANUS publisher license handshake (attempt $attempt/$attempts)..."
    if start_manus_publisher_once "$manus_mode" "$publisher_executable"; then
      if (( attempt > 1 )); then
        echo "[INFO] MANUS SDK license recovered automatically on attempt $attempt/$attempts."
      fi
      return 0
    else
      result=$?
    fi
    if (( attempt < attempts )); then
      reset_dongle=false
      if [[ "$manus_mode" == "integrated" ]] \
        && manus_publisher_dongle_initialization_failed; then
        reset_dongle=true
      fi
      stop_component publisher >/dev/null 2>&1 || true
      archive_manus_publisher_attempt_log "$manus_mode" "$attempt"
      if $reset_dongle; then
        if ! reset_manus_usb_dongle; then
          echo "[WARN] Continuing with a publisher-only retry after the USB reset failed." >&2
        fi
      fi
      echo "[WARN] MANUS SDK license handshake failed; retrying only the publisher in ${MANUS_PUBLISHER_RETRY_DELAY}s." >&2
      sleep "$MANUS_PUBLISHER_RETRY_DELAY"
    fi
  done

  if manus_publisher_has_fatal_error; then
    report_manus_publisher_fatal_error
  else
    echo "[ERROR] MANUS publisher did not confirm a usable SDK license after $attempts attempts." >&2
    tail -n 80 "$LOG_DIR/publisher.log" >&2 || true
  fi
  return "$result"
}

latest_openvr_tracker_state() {
  [[ -r "$LOG_DIR/openvr.log" ]] || return 1
  grep -E "\[OpenVR Input\] Tracker stream (state|ready):" "$LOG_DIR/openvr.log" \
    | tail -n 1
}

openvr_log_has_role() {
  local role="$1"
  local state_line="${2:-}"
  if [[ -z "$state_line" ]]; then
    state_line="$(latest_openvr_tracker_state)" || return 1
  fi
  printf '%s\n' "$state_line" \
    | grep -Eq "(^|[^[:alnum:]_])${role}([^[:alnum:]_]|$)"
}

openvr_required_roles_ready() {
  local state_line="$1"
  [[ -n "$state_line" ]] || return 1
  case "$HAND" in
    right)
      openvr_log_has_role right_wrist "$state_line"
      ;;
    left)
      openvr_log_has_role left_wrist "$state_line"
      ;;
    both)
      openvr_log_has_role right_wrist "$state_line" \
        && openvr_log_has_role left_wrist "$state_line"
      ;;
  esac
}

wait_for_openvr_client() {
  local pid="$1"
  local timeout_seconds="$2"
  local iterations index state_line
  [[ "$timeout_seconds" =~ ^[0-9]+$ ]] || timeout_seconds=90
  iterations=$(( timeout_seconds * 10 ))

  for ((index = 0; index < iterations; index += 1)); do
    if ! process_alive "$pid" || ! pid_matches_component openvr "$pid"; then
      echo "[ERROR] openvr_input exited before becoming ready" >&2
      return 1
    fi
    state_line="$(latest_openvr_tracker_state 2>/dev/null || true)"
    if openvr_required_roles_ready "$state_line"; then
      return 0
    fi
    sleep 0.1
  done

  echo "[ERROR] openvr_input is alive, but valid poses for the required wrist trackers did not become ready within ${timeout_seconds}s" >&2
  return 1
}

launch_openvr_client() {
  : >"$LOG_DIR/openvr.log"
  setsid bash -lc "source '$ENV_SCRIPT' && cd '$MANUS_ROOT' && exec ros2 run openvr_input openvr_input -c '$OPENVR_CONFIG'" \
    >>"$LOG_DIR/openvr.log" 2>&1 </dev/null &
  local openvr_pid=$!
  write_component_pid openvr "$openvr_pid"
  echo "$openvr_pid"
}

start_openvr_with_retries() {
  local attempts="$OPENVR_START_ATTEMPTS"
  local attempt openvr_pid
  [[ "$attempts" =~ ^[0-9]+$ ]] || attempts=1
  (( attempts > 0 )) || attempts=1

  for ((attempt = 1; attempt <= attempts; attempt += 1)); do
    echo "[INFO] Starting openvr_input client (attempt $attempt/$attempts)..."
    openvr_pid="$(launch_openvr_client)"
    if wait_for_openvr_client "$openvr_pid" "$OPENVR_START_TIMEOUT"; then
      echo "[INFO] openvr_input is connected with the required wrist tracker mapping (pid $openvr_pid)"
      return 0
    fi
    if [[ -r "$LOG_DIR/openvr.log" ]]; then
      tail -n 40 "$LOG_DIR/openvr.log" >&2 || true
    fi
    if (( attempt < attempts )); then
      stop_component openvr >/dev/null 2>&1 || true
      echo "[WARN] Vive tracker mapping is not ready; retrying only the OpenVR client in ${OPENVR_RETRY_DELAY}s" >&2
      sleep "$OPENVR_RETRY_DELAY"
    fi
  done

  echo "[ERROR] Vive wrist trackers are not pose-ready after $attempts OpenVR client attempts; move or wake the trackers and confirm base-station visibility, then use recover-openvr (SteamVR restart is not required)" >&2
  return 1
}

recover_openvr() {
  case "$HAND" in
    left|right|both) ;;
    *)
      echo "[ERROR] hand must be left, right, or both (got: $HAND)" >&2
      return 1
      ;;
  esac
  require_vrserver
  require_component_running publisher
  require_component_running input
  require_component_running bridge
  if [[ ! -f "$ENV_SCRIPT" ]]; then
    write_env_script
  fi
  if [[ ! -f "$OPENVR_CONFIG" ]]; then
    echo "[ERROR] Missing $OPENVR_CONFIG; use a full Mode A start to configure tracker bindings" >&2
    return 1
  fi

  mkdir -p "$LOG_DIR/archive"
  if [[ -f "$LOG_DIR/openvr.log" ]]; then
    mv -f "$LOG_DIR/openvr.log" "$LOG_DIR/archive/openvr-$(date +%Y%m%d-%H%M%S).log"
  fi
  echo "[INFO] Recovering openvr_input only; MANUS, bridge, SteamVR, and bridge calibration remain untouched."
  stop_component openvr || return 1
  start_openvr_with_retries
}

start_components_for_manus_mode() {
  local use_tracker="$1"
  local manus_mode="$2"
  local mode_index="$3"
  local mode_count="$4"
  local publisher_executable
  local step=1
  local total=3

  if $use_tracker; then
    total=4
  fi

  if (( mode_count > 1 )); then
    echo "[INFO] MANUS connection attempt $mode_index/$mode_count: $manus_mode"
  else
    echo "[INFO] MANUS connection mode: $manus_mode"
  fi

  publisher_executable="$(manus_publisher_executable "$manus_mode")"
  if [[ "$publisher_executable" == "manus_data_publisher_core" \
    && ! -x "$WS/install/manus_ros2/lib/manus_ros2/$publisher_executable" ]]; then
    echo "[ERROR] MANUS core connection mode is unavailable because $publisher_executable is not built on this machine" >&2
    return 1
  fi

  echo "[$step/$total] Starting manus_data_publisher..."
  if ! start_manus_publisher_with_retries "$manus_mode" "$publisher_executable"; then
    return 1
  fi
  step=$((step + 1))

  echo "[$step/$total] Starting manus_input (hand=$HAND)..."
  setsid bash -lc "source '$ENV_SCRIPT' && cd '$MANUS_ROOT' && exec ros2 run manus_input_py manus_input -c '$MANUS_INPUT_CONFIG'" \
    >"$LOG_DIR/input.log" 2>&1 </dev/null &
  local input_pid=$!
  write_component_pid input "$input_pid"
  sleep 2
  if ! verify_started_process input "$input_pid" "$LOG_DIR/input.log"; then
    return 1
  fi
  step=$((step + 1))

  if $use_tracker; then
    if [[ ! -f "$OPENVR_CONFIG" ]]; then
      echo "[ERROR] Missing $OPENVR_CONFIG" >&2
      return 1
    fi
    echo "[$step/$total] Starting openvr_input (Vive wrist TF)..."
    if ! start_openvr_with_retries; then
      return 1
    fi
    step=$((step + 1))
  fi

  echo "[$step/$total] Starting HandsOn bridge (hand=$HAND, hand_type=$HAND_TYPE)..."
  echo "Bridge Python: $BRIDGE_PYTHON"
  BRIDGE_CMD=(
    "$BRIDGE_PYTHON" "$BRIDGE_SCRIPT"
    --hand "$HAND"
    --hand-type "$HAND_TYPE"
    --manus-wuji-root "$MANUS_ROOT"
    --mapping-config "$HANDSON_MAPPING_CONFIG"
    --frequency "$BRIDGE_FREQUENCY"
  )
  append_bridge_calibration_args
  if $use_tracker; then
    BRIDGE_CMD+=(--send-tracker)
    if is_enabled "$SWAP_WRIST_TRACKERS"; then
      BRIDGE_CMD+=(--swap-wrist-trackers)
    fi
    if is_enabled "$SWAP_ARMS"; then
      BRIDGE_CMD+=(--swap-arms)
    fi
    if [[ "$TRACKER_MODE" == "relative" ]]; then
      BRIDGE_CMD+=(--tracker-relative)
    elif [[ "$TRACKER_MODE" == "absolute" ]]; then
      BRIDGE_CMD+=(--tracker-absolute)
    fi
  fi
  setsid bash -lc "source '$ENV_SCRIPT' && cd '$ROOT' && exec ${BRIDGE_CMD[*]@Q}" \
    >"$LOG_DIR/bridge.log" 2>&1 </dev/null &
  local bridge_pid=$!
  write_component_pid bridge "$bridge_pid"
  sleep 1
  if ! verify_started_process bridge "$bridge_pid" "$LOG_DIR/bridge.log"; then
    return 1
  fi
  local monitor_calibration=false
  if $use_tracker && [[ "$TRACKER_MODE" == "relative" ]]; then
    monitor_calibration=true
  fi
  if ! wait_for_manus_hand_stream "$bridge_pid" "$monitor_calibration"; then
    return 1
  fi

  if $use_tracker && [[ "$TRACKER_MODE" == "relative" ]]; then
    wait_for_startup_calibration "$bridge_pid"
  fi
}

start() {
  local -a manus_modes=()
  local mode mode_index mode_count result

  if [[ "$HAND" != "left" && "$HAND" != "right" && "$HAND" != "both" ]]; then
    echo "[ERROR] hand must be left, right, or both (got: $HAND)" >&2
    exit 1
  fi
  if [[ "$HAND_TYPE" != "wuji1" && "$HAND_TYPE" != "wuji2" ]]; then
    echo "[ERROR] hand type must be wuji1 or wuji2 (got: $HAND_TYPE)" >&2
    exit 1
  fi
  if [[ ! -f "$WS/install/setup.bash" ]]; then
    echo "[ERROR] Workspace not built: $WS/install/setup.bash" >&2
    echo "  Build the vendored workspace:" >&2
    echo "    cd $MANUS_ROOT && colcon build --symlink-install" >&2
    echo "  Or set ROS2_WS to an already built workspace, for example:" >&2
    echo "    export ROS2_WS=$HOME/ros2_ws" >&2
    exit 1
  fi
  if [[ ! -f "$BRIDGE_SCRIPT" ]]; then
    echo "[ERROR] Missing HandsOn bridge: $BRIDGE_SCRIPT" >&2
    exit 1
  fi
  if [[ ! -f "$HANDSON_MAPPING_CONFIG" ]]; then
    echo "[ERROR] Missing HandsOn mapping config: $HANDSON_MAPPING_CONFIG" >&2
    exit 1
  fi
  mapfile -t manus_modes < <(manus_connection_modes) || return 1
  mode_count="${#manus_modes[@]}"
  if (( mode_count == 0 )); then
    return 1
  fi
  if ! preflight_manus_usb_dongle_for_modes "${manus_modes[@]}"; then
    return 1
  fi

  local use_tracker=false
  if [[ "$TRACKER" == "tracker" || "$TRACKER" == "relative" ]]; then
    use_tracker=true
  fi

  # Tracker preflight must complete before stopping a working device stack.
  # A missing vrserver, ambiguous auto binding, or unwritable machine config
  # should leave the currently running stack untouched.
  write_env_script
  if $use_tracker; then
    prepare_openvr_config
  fi

  # Failures before this marker are preflight failures and must not tear down a
  # healthy existing stack. Failures after it own the replacement lifecycle and
  # are cleaned by start_with_cleanup.
  : >"$STARTUP_COMPONENTS_MARKER"
  if ! replace_existing_components; then
    return 1
  fi
  sleep 1

  enable_manus_autopair
  write_manus_input_config
  check_wuji_retargeting

  result=1
  for ((mode_index = 1; mode_index <= mode_count; mode_index += 1)); do
    mode="${manus_modes[$((mode_index - 1))]}"
    if (( mode_index > 1 )); then
      stop_components >/dev/null 2>&1 || true
      sleep 1
    fi
    start_components_for_manus_mode "$use_tracker" "$mode" "$mode_index" "$mode_count"
    result=$?
    if (( result == 0 )); then
      result=0
      break
    fi
    if (( mode_index < mode_count )); then
      archive_startup_attempt_logs "$mode" "$mode_index"
      echo "[WARN] Mode A did not get the requested MANUS hand stream with connection mode '$mode'; retrying with '${manus_modes[$mode_index]}'." >&2
    fi
  done
  if (( result != 0 )); then
    return "$result"
  fi

  if $use_tracker; then
    write_stack_state 1
  else
    write_stack_state 0
  fi

  echo ""
  echo "Mode A HandsOn bridge started. Logs: $LOG_DIR/"
  echo "Bridge frequency: ${BRIDGE_FREQUENCY} Hz (override with HANDSON_BRIDGE_FREQUENCY)."
  echo "Dexterous hand: $HAND_TYPE"
  if $use_tracker; then
    echo "Wrist tracking enabled (HandsOn captures relative pose when you press ';')."
  fi
  if [[ "$TRACKER_MODE" != "relative" && "$TRACKER_MODE" != "absolute" ]]; then
    echo "[WARN] Ignoring unknown tracker mode: $TRACKER_MODE"
  fi
  echo "Run HandsOn collection in another terminal, then press ';' in the MuJoCo window."
  echo "Stop: $0 stop"
}

stop() {
  stop_managed
}

status() {
  local result=0
  local name pid
  local require_openvr=false
  if stack_requires_openvr; then
    require_openvr=true
  fi

  for name in publisher input openvr bridge; do
    if ! pid="$(read_component_pid "$name" 2>/dev/null)"; then
      echo "$name: stopped"
      if [[ "$name" != openvr ]] || $require_openvr; then
        result=1
      fi
    elif ! process_alive "$pid"; then
      echo "$name: stopped (stale pid $pid)"
      if [[ "$name" != openvr ]] || $require_openvr; then
        result=1
      fi
    elif ! pid_matches_component "$name" "$pid"; then
      echo "$name: invalid pid identity (pid $pid)"
      if [[ "$name" != openvr ]] || $require_openvr; then
        result=1
      fi
    elif [[ "$name" == publisher ]] && manus_publisher_has_fatal_error; then
      echo "$name: unhealthy (MANUS SDK license error, pid $pid)"
      result=1
    elif [[ "$name" == publisher ]] && manus_publisher_has_license_error; then
      echo "$name: running (license warning ignored, pid $pid)"
    else
      echo "$name: running (pid $pid)"
    fi
  done
  if [[ -r "$STACK_SCOPE_FILE" ]]; then
    echo "scope: $(<"$STACK_SCOPE_FILE")"
  fi
  if [[ -r "$OPENVR_SCOPE_FILE" ]]; then
    echo "openvr recovery scope: $(<"$OPENVR_SCOPE_FILE")"
  fi
  if [[ -r "$STACK_STATE_FILE" ]]; then
    local saved_hand_type saved_strict_manus_hands
    saved_hand_type="$(sed -n -E 's/^HAND_TYPE=(wuji1|wuji2)$/\1/p' "$STACK_STATE_FILE" | head -n 1)"
    saved_strict_manus_hands="$(sed -n -E 's/^STRICT_MANUS_HANDS=([01])$/\1/p' "$STACK_STATE_FILE" | head -n 1)"
    echo "hand_type: ${saved_hand_type:-wuji1}"
    echo "strict_manus_hands: ${saved_strict_manus_hands:-0}"
  fi
  echo "logs: $LOG_DIR/"
  return "$result"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case "$ACTION" in
    start) run_independent start "$HAND" "$TRACKER" "$TRACKER_MODE" "$HAND_TYPE" ;;
    # The scoped child must publish the durable receipt itself.  A workbench
    # restart kills the outer launcher cgroup while this independent scope is
    # intentionally allowed to finish calibration.
    _start-local) start_direct_with_receipt ;;
    stop) stop ;;
    status) status ;;
    recover-openvr)
      if [[ -z "$HAND_ARGUMENT" ]]; then
        HAND="$(saved_stack_hand)"
      fi
      run_independent recover-openvr "$HAND"
      ;;
    _recover-openvr-local) recover_openvr_with_cleanup ;;
    *)
      echo "Usage: $0 {start|stop|status|recover-openvr} [left|right|both] [tracker] [relative] [wuji1|wuji2]" >&2
      exit 1
      ;;
  esac
fi
