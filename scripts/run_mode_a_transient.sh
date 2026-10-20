#!/usr/bin/env bash
# Launch Mode A in transient user scopes so collection-ui restarts cannot kill
# the device processes. Nothing is installed under ~/.config/systemd.

set -euo pipefail

ROOT="${HANDSON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE_A_SCRIPT="${HANDSON_MODE_A_SCRIPT:-$ROOT/scripts/mode_a_handson.sh}"
SYSTEMD_RUN_BIN="${HANDSON_SYSTEMD_RUN:-systemd-run}"
SYSTEMCTL_BIN="${HANDSON_SYSTEMCTL:-systemctl}"
PID_DIR="${XDG_RUNTIME_DIR:-/tmp}/handson-mode-a"
STACK_SCOPE_FILE="$PID_DIR/stack.scope"
OPENVR_SCOPE_FILE="$PID_DIR/openvr.scope"
ACTION="${1:-}"
shift || true

mkdir -p "$PID_DIR"

user_manager_available() {
  command -v "$SYSTEMD_RUN_BIN" >/dev/null 2>&1 || return 1
  command -v "$SYSTEMCTL_BIN" >/dev/null 2>&1 || return 1
  "$SYSTEMCTL_BIN" --user show-environment >/dev/null 2>&1
}

write_scope_file() {
  local path="$1"
  local unit="$2"
  local temporary="$PID_DIR/.$(basename "$path").$$"
  printf '%s\n' "$unit" >"$temporary"
  mv -f "$temporary" "$path"
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

stop_recorded_unit() {
  local path="$1"
  local unit=""
  if [[ -r "$path" ]]; then
    read -r unit <"$path" || true
  fi
  if [[ "$unit" == handson-mode-a-*.scope ]]; then
    "$SYSTEMCTL_BIN" --user stop "$unit" >/dev/null 2>&1 || true
    "$SYSTEMCTL_BIN" --user reset-failed "$unit" >/dev/null 2>&1 || true
  fi
  rm -f "$path"
}

cleanup_units() {
  if ! user_manager_available; then
    rm -f "$STACK_SCOPE_FILE" "$OPENVR_SCOPE_FILE"
    return 0
  fi
  stop_recorded_unit "$OPENVR_SCOPE_FILE"
  stop_recorded_unit "$STACK_SCOPE_FILE"
  # Covers a launcher crash between transient-unit creation and state-file write.
  "$SYSTEMCTL_BIN" --user stop 'handson-mode-a-*.scope' >/dev/null 2>&1 || true
}

run_local_fallback() {
  local hidden_action="$1"
  local public_action=""
  shift
  echo "[WARN] systemd user manager is unavailable; Mode A will run in the caller cgroup" >&2
  case "$hidden_action" in
    _start-local) public_action=start ;;
    _recover-openvr-local) public_action=recover-openvr ;;
    *) public_action="$hidden_action" ;;
  esac
  exec env HANDSON_MODE_A_DISABLE_SYSTEMD=1 "$MODE_A_SCRIPT" "$public_action" "$@"
}

run_in_scope() {
  local kind="$1"
  local hidden_action="$2"
  local state_file="$3"
  shift 3
  local unit_base="handson-mode-a-${kind}-$(date +%s)-$$"
  local unit="${unit_base}.scope"
  local result=0
  local unit_active=false

  if ! user_manager_available; then
    run_local_fallback "$hidden_action" "$@"
  fi

  echo "[INFO] Launching Mode A $kind in independent user scope $unit"
  if "$SYSTEMD_RUN_BIN" \
    --user \
    --scope \
    --quiet \
    --collect \
    --no-ask-password \
    --same-dir \
    --unit="$unit_base" \
    --description="HandsOn Mode A $kind" \
    --property=TimeoutStopSec=8s \
    "$MODE_A_SCRIPT" "$hidden_action" "$@"; then
    result=0
  else
    result=$?
  fi

  # A timed-out recover-openvr intentionally leaves its self-healing client
  # alive. A successful component status means the scope must be retained even
  # when the foreground recovery command returned non-zero.
  if "$SYSTEMCTL_BIN" --user is-active --quiet "$unit" >/dev/null 2>&1; then
    unit_active=true
  fi
  if $unit_active \
    && "$MODE_A_SCRIPT" status >/dev/null 2>&1 \
    && { (( result == 0 )) || [[ "$kind" == openvr ]]; }; then
    write_scope_file "$state_file" "$unit"
    echo "[INFO] Mode A $kind is isolated in $unit"
    if [[ "$kind" == stack ]] && ! write_calibration_receipt; then
      result=1
    fi
  else
    "$SYSTEMCTL_BIN" --user stop "$unit" >/dev/null 2>&1 || true
    "$SYSTEMCTL_BIN" --user reset-failed "$unit" >/dev/null 2>&1 || true
    if (( result == 0 )); then
      result=1
    fi
  fi
  return "$result"
}

case "$ACTION" in
  start)
    run_in_scope stack _start-local "$STACK_SCOPE_FILE" "$@"
    ;;
  recover-openvr)
    run_in_scope openvr _recover-openvr-local "$OPENVR_SCOPE_FILE" "$@"
    ;;
  cleanup)
    cleanup_units
    ;;
  *)
    echo "Usage: $0 {start|recover-openvr|cleanup} [mode-a arguments...]" >&2
    exit 1
    ;;
esac
