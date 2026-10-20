from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODE_A_SCRIPT = ROOT / "scripts" / "mode_a_handson.sh"
TRANSIENT_LAUNCHER = ROOT / "scripts" / "run_mode_a_transient.sh"


class ModeALifecycleTest(unittest.TestCase):
    def run_sourced(
        self,
        body: str,
        runtime_dir: Path,
    ) -> subprocess.CompletedProcess[str]:
        command = f"source {shlex.quote(str(MODE_A_SCRIPT))}\n{body}"
        env = dict(os.environ)
        env["XDG_RUNTIME_DIR"] = str(runtime_dir)
        return subprocess.run(
            ["bash", "-c", command],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_stop_refuses_reused_pid_without_signalling_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            signal_log = runtime / "signals"
            result = self.run_sourced(
                "\n".join(
                    (
                        "printf '4242\\n' >\"$PID_DIR/openvr.pid\"",
                        "process_alive() { return 0; }",
                        "pid_matches_component() { return 1; }",
                        f"signal_component_pid() {{ printf '%s\\n' \"$1\" >>{shlex.quote(str(signal_log))}; }}",
                        "stop_component openvr",
                        "test ! -e \"$PID_DIR/openvr.pid\"",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(signal_log.exists())
            self.assertIn("Refusing to signal pid 4242", result.stderr)

    def test_stop_escalates_int_term_kill_and_waits_each_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            signal_log = runtime / "signals"
            result = self.run_sourced(
                "\n".join(
                    (
                        "printf '4242\\n' >\"$PID_DIR/openvr.pid\"",
                        "process_alive() { return 0; }",
                        "pid_matches_component() { return 0; }",
                        f"signal_component_pid() {{ LAST_SIGNAL=\"$1\"; printf '%s\\n' \"$1\" >>{shlex.quote(str(signal_log))}; }}",
                        "wait_for_process_exit() { [[ \"$LAST_SIGNAL\" == KILL ]]; }",
                        "stop_component openvr",
                        "test ! -e \"$PID_DIR/openvr.pid\"",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(signal_log.read_text(encoding="utf-8").splitlines(), ["INT", "TERM", "KILL"])

    def test_status_is_nonzero_when_any_required_component_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "printf 'TRACKER_REQUIRED=1\\n' >\"$STACK_STATE_FILE\"\nstatus",
                runtime,
            )
            self.assertEqual(result.returncode, 1)
            for component in ("publisher", "input", "openvr", "bridge"):
                self.assertIn(f"{component}: stopped", result.stdout)

    def test_status_always_reports_all_four_components_without_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced("status", runtime)
            self.assertEqual(result.returncode, 1)
            status_lines = result.stdout.splitlines()
            self.assertEqual(
                status_lines[:4],
                [
                    "publisher: stopped",
                    "input: stopped",
                    "openvr: stopped",
                    "bridge: stopped",
                ],
            )

    def test_status_marks_running_publisher_with_license_error_unhealthy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "IGNORE_MANUS_LICENSE=0",
                        "printf '4242\\n' >\"$PID_DIR/publisher.pid\"",
                        "printf 'No compatible license found. Please connect a license with the SDK component.\\n' >\"$LOG_DIR/publisher.log\"",
                        "process_alive() { return 0; }",
                        "pid_matches_component() { return 0; }",
                        "status",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "publisher: unhealthy (MANUS SDK license error, pid 4242)",
                result.stdout,
            )

    def test_status_marks_ignored_license_warning_running_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        'test "$IGNORE_MANUS_LICENSE" = 1',
                        "printf '4242\\n' >\"$PID_DIR/publisher.pid\"",
                        "printf '%s\\n' \"ERROR: It looks like you don't have a valid SDK Integrated license.\" \"DeviceManager: Second stage initialisation succeeded\" >\"$LOG_DIR/publisher.log\"",
                        "process_alive() { return 0; }",
                        "pid_matches_component() { return 0; }",
                        "stack_requires_openvr() { return 1; }",
                        "status",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "publisher: running (license warning ignored, pid 4242)",
                result.stdout,
            )

    def test_sdk_initialization_failure_remains_fatal_when_license_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "IGNORE_MANUS_LICENSE=1",
                        "printf 'Failed to initialize the SDK\\n' >\"$LOG_DIR/publisher.log\"",
                        "manus_publisher_has_fatal_error",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_transient_integrated_license_message_is_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "printf \"ERROR: It looks like you don't have a valid SDK Integrated license.\\n\" >\"$LOG_DIR/publisher.log\"",
                        "! manus_publisher_has_fatal_error",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_failed_dongle_second_stage_is_fatal_when_license_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "IGNORE_MANUS_LICENSE=1",
                        "printf 'DeviceManager: Second stage initialisation failed\\n' >\"$LOG_DIR/publisher.log\"",
                        "manus_publisher_has_fatal_error",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_manus_usb_preflight_reports_missing_license_dongle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "MANUS_USB_WAIT_TIMEOUT=0",
                        "manus_usb_dongle_visible() { return 1; }",
                        "if wait_for_manus_usb_dongle; then exit 31; fi",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("MANUS license Dongle was not detected", result.stderr)
            self.assertIn("USB vendor 3325", result.stderr)

    def test_manus_usb_preflight_accepts_visible_dongle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "MANUS_USB_SETTLE_DELAY=0",
                        "manus_usb_dongle_visible() { return 0; }",
                        "sleep() { :; }",
                        "wait_for_manus_usb_dongle",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("MANUS license Dongle detected", result.stdout)

    def test_manus_remote_mode_skips_local_usb_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "wait_for_manus_usb_dongle() { exit 31; }",
                        "preflight_manus_usb_dongle_for_modes remote",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("skipping the local license Dongle", result.stdout)

    def test_manus_local_mode_runs_usb_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "dongle_preflight_count=0",
                        "wait_for_manus_usb_dongle() { "
                        "dongle_preflight_count=$((dongle_preflight_count + 1)); }",
                        "preflight_manus_usb_dongle_for_modes integrated remote",
                        'test "$dongle_preflight_count" -eq 1',
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_manus_license_wait_does_not_require_pre_exec_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "MANUS_PUBLISHER_START_TIMEOUT=1",
                        "process_alive() { return 0; }",
                        "pid_matches_component() { return 1; }",
                        "manus_publisher_has_fatal_error() { return 1; }",
                        "manus_publisher_license_ready() { return 0; }",
                        "wait_for_manus_publisher_license 4242 integrated",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_manus_license_wait_can_ignore_sdk_license_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "IGNORE_MANUS_LICENSE=1",
                        "MANUS_PUBLISHER_START_TIMEOUT=1",
                        "printf '%s\\n' \"ERROR: It looks like you don't have a valid SDK Integrated license.\" >\"$LOG_DIR/publisher.log\"",
                        "process_alive() { return 0; }",
                        "manus_publisher_has_fatal_error() { return 1; }",
                        "manus_publisher_license_ready() { return 0; }",
                        "wait_for_manus_publisher_license 4242 integrated",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Ignoring the MANUS SDK license gate", result.stderr)

    def test_manus_license_wait_requires_integrated_dongle_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        'test "$IGNORE_MANUS_LICENSE" = 1',
                        "MANUS_PUBLISHER_START_TIMEOUT=0",
                        "process_alive() { return 0; }",
                        "manus_publisher_license_ready() { return 1; }",
                        "if wait_for_manus_publisher_license 4242 integrated; then exit 31; fi",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("Ignoring the MANUS SDK license gate", result.stderr)

    def test_manus_license_wait_never_ignores_publisher_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "IGNORE_MANUS_LICENSE=1",
                        "MANUS_PUBLISHER_START_TIMEOUT=0",
                        "process_alive() { return 1; }",
                        "manus_publisher_license_ready() { return 1; }",
                        "if wait_for_manus_publisher_license 4242 integrated; then exit 31; fi",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_ignored_license_warning_still_requires_live_hand_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "IGNORE_MANUS_LICENSE=1",
                        "MANUS_HAND_WAIT_TIMEOUT=0",
                        "HAND=both",
                        "printf 'No compatible license found.\\n' >\"$LOG_DIR/publisher.log\"",
                        "requested_manus_hands_ready() { return 1; }",
                        "if wait_for_manus_hand_stream 4242; then exit 31; fi",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("No non-zero MANUS finger commands arrived", result.stderr)

    def test_manus_license_wait_can_enforce_sdk_license_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "IGNORE_MANUS_LICENSE=0",
                        "MANUS_PUBLISHER_START_TIMEOUT=0",
                        "printf '%s\\n' \"ERROR: It looks like you don't have a valid SDK Integrated license.\" >\"$LOG_DIR/publisher.log\"",
                        "manus_publisher_license_ready() { return 1; }",
                        "if wait_for_manus_publisher_license 4242 integrated; then exit 31; fi",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_manus_publisher_retries_transient_license_startup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "MANUS_PUBLISHER_START_ATTEMPTS=3",
                        "MANUS_PUBLISHER_RETRY_DELAY=0",
                        "publisher_attempt_count=0",
                        "start_manus_publisher_once() { "
                        "publisher_attempt_count=$((publisher_attempt_count + 1)); "
                        "(( publisher_attempt_count >= 3 )); }",
                        "stop_component() { :; }",
                        "archive_manus_publisher_attempt_log() { :; }",
                        "sleep() { :; }",
                        "start_manus_publisher_with_retries integrated manus_data_publisher",
                        "test \"$publisher_attempt_count\" -eq 3",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("recovered automatically on attempt 3/3", result.stdout)

    def test_manus_publisher_resets_stuck_integrated_dongle_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "MANUS_PUBLISHER_START_ATTEMPTS=2",
                        "MANUS_PUBLISHER_RETRY_DELAY=0",
                        "publisher_attempt_count=0",
                        "dongle_reset_count=0",
                        "start_manus_publisher_once() { "
                        "publisher_attempt_count=$((publisher_attempt_count + 1)); "
                        "(( publisher_attempt_count >= 2 )); }",
                        "manus_publisher_dongle_initialization_failed() { "
                        "(( publisher_attempt_count == 1 )); }",
                        "reset_manus_usb_dongle() { "
                        "dongle_reset_count=$((dongle_reset_count + 1)); }",
                        "stop_component() { :; }",
                        "archive_manus_publisher_attempt_log() { :; }",
                        "sleep() { :; }",
                        "start_manus_publisher_with_retries integrated manus_data_publisher",
                        "test \"$publisher_attempt_count\" -eq 2",
                        "test \"$dongle_reset_count\" -eq 1",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_manus_publisher_license_ready_requires_requested_license(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "printf '%s\n' "
                        "'MANUS landscape: license(sdk=false integrated=false maxPairs=0)' "
                        '>"$LOG_DIR/publisher.log"',
                        "if manus_publisher_license_ready integrated; then exit 31; fi",
                        "printf '%s\n' "
                        "'MANUS landscape: license(sdk=false integrated=true maxPairs=1)' "
                        '>"$LOG_DIR/publisher.log"',
                        "manus_publisher_license_ready integrated",
                        "printf '%s\n' "
                        "'MANUS landscape: license(sdk=true integrated=false maxPairs=1)' "
                        '>"$LOG_DIR/publisher.log"',
                        "manus_publisher_license_ready core",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_requested_manus_hands_default_allows_one_live_hand_for_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "HAND=both",
                        "STRICT_MANUS_HANDS=0",
                        "printf '/hand_input msgs=10 hand=fresh age=0.01s sent(right=10 left=0 tracker=10)\\n' >\"$LOG_DIR/bridge.log\"",
                        "requested_manus_hands_ready",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_requested_manus_hands_strict_requires_nonzero_counter_for_each_side(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "HAND=both",
                        "STRICT_MANUS_HANDS=1",
                        "printf '/hand_input msgs=10 hand=fresh age=0.01s sent(right=10 left=0 tracker=10)\\n' >\"$LOG_DIR/bridge.log\"",
                        "if requested_manus_hands_ready; then exit 31; fi",
                        "printf '/hand_input msgs=11 hand=fresh age=0.01s sent(right=11 left=1 tracker=11)\\n' >>\"$LOG_DIR/bridge.log\"",
                        "requested_manus_hands_ready",
                        "HAND=right",
                        "printf '/hand_input msgs=1 hand=fresh age=0.01s sent(right=1 tracker=1)\\n' >\"$LOG_DIR/bridge.log\"",
                        "requested_manus_hands_ready",
                        "HAND=left",
                        "printf '/hand_input msgs=1 hand=fresh age=0.01s sent(left=1 tracker=1)\\n' >\"$LOG_DIR/bridge.log\"",
                        "requested_manus_hands_ready",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_requested_manus_hands_reject_latest_stale_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "HAND=both",
                        "printf '/hand_input msgs=10 hand=fresh age=0.01s sent(right=10 left=10 tracker=10)\\n' >\"$LOG_DIR/bridge.log\"",
                        "requested_manus_hands_ready",
                        "printf '/hand_input msgs=10 hand=stale age=1.00s sent(right=10 left=10 tracker=20)\\n' >>\"$LOG_DIR/bridge.log\"",
                        "if requested_manus_hands_ready; then exit 31; fi",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_generated_manus_config_defaults_to_independent_hand_freshness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "HAND=both",
                        "write_manus_input_config",
                        "grep -Fx 'include_right_hand: true' \"$MANUS_INPUT_CONFIG\"",
                        "grep -Fx 'include_left_hand: true' \"$MANUS_INPUT_CONFIG\"",
                        "grep -Fx 'allow_missing_hand: true' \"$MANUS_INPUT_CONFIG\"",
                        "grep -Fx 'require_all_requested_updates: false' \"$MANUS_INPUT_CONFIG\"",
                        "grep -Fx 'max_hand_update_age_sec: 0.0' \"$MANUS_INPUT_CONFIG\"",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_generated_manus_config_can_enable_missing_hand_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "HAND=both",
                        "STRICT_MANUS_HANDS=0",
                        "write_manus_input_config",
                        "grep -Fx 'allow_missing_hand: true' \"$MANUS_INPUT_CONFIG\"",
                        "grep -Fx 'require_all_requested_updates: false' \"$MANUS_INPUT_CONFIG\"",
                        "grep -Fx 'max_hand_update_age_sec: 0.0' \"$MANUS_INPUT_CONFIG\"",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_manus_hand_timeout_runs_partial_component_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            cleanup_marker = runtime / "manus-timeout-cleaned"
            result = self.run_sourced(
                "\n".join(
                    (
                        "HAND=both",
                        "MANUS_HAND_WAIT_TIMEOUT=0",
                        "printf '/hand_input msgs=0 hand=never sent(right=0 left=0 tracker=20)\\n' >\"$LOG_DIR/bridge.log\"",
                        ": >\"$LOG_DIR/publisher.log\"",
                        ": >\"$LOG_DIR/input.log\"",
                        "start() { : >\"$STARTUP_COMPONENTS_MARKER\"; wait_for_manus_hand_stream 4242; }",
                        f"stop_components() {{ : >{shlex.quote(str(cleanup_marker))}; }}",
                        "start_with_cleanup",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 1)
            self.assertTrue(cleanup_marker.exists())
            self.assertIn("No non-zero MANUS finger commands arrived", result.stderr)
            self.assertIn("No fresh /hand_input", result.stderr)
            self.assertNotIn("Reconnect the MANUS Sensor Dongle", result.stderr)
            self.assertIn("cleaning all partially started components", result.stderr)

    def test_manus_wait_surfaces_calibration_progress_without_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "HAND=both",
                        "MANUS_HAND_WAIT_TIMEOUT=5",
                        "CALIBRATION_WAIT_TIMEOUT=1",
                        "printf 'Guided calibration phase: anchor - hold still\\n/hand_input msgs=1 hand=fresh sent(right=1 left=1 tracker=1)\\n' >\"$LOG_DIR/bridge.log\"",
                        ": >\"$LOG_DIR/publisher.log\"",
                        ": >\"$LOG_DIR/input.log\"",
                        "process_alive() { return 0; }",
                        "pid_matches_component() { return 0; }",
                        "sleep() { printf '/hand_input msgs=11 hand=fresh sent(right=11 left=11 tracker=11)\\n' >>\"$LOG_DIR/bridge.log\"; }",
                        "wait_for_manus_hand_stream 4242 true",
                        "printf 'Guided startup calibration complete\\n' >>\"$LOG_DIR/bridge.log\"",
                        "kill() { return 0; }",
                        "wait_for_startup_calibration 4242",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("Startup calibration monitor:"), 1)
            self.assertEqual(result.stdout.count("Guided calibration phase: anchor"), 1)
            self.assertEqual(result.stdout.count("Guided startup calibration complete"), 1)

    def test_old_fresh_status_is_rejected_even_when_log_file_is_updated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_sourced(
                '\n'.join((
                    "HAND=both",
                    "printf '/hand_input msgs=10 hand=fresh sent(right=10 left=10) status_unix_sec=1\\nNew unrelated log line\\n' >\"$LOG_DIR/bridge.log\"",
                    "if requested_manus_hands_ready; then exit 31; fi",
                    "report_manus_readiness_diagnosis",
                )), Path(directory),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Bridge status stopped updating", result.stderr)

    def test_hand_progress_requires_new_input_and_requested_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_sourced(
                '\n'.join((
                    "HAND=both; STRICT_MANUS_HANDS=1",
                    "previous='/hand_input msgs=10 hand=fresh sent(right=10 left=10)'",
                    "if manus_hand_status_progressed \"$previous\" \"$previous\"; then exit 31; fi",
                    "if manus_hand_status_progressed \"$previous\" '/hand_input msgs=10 hand=fresh sent(right=20 left=20)'; then exit 32; fi",
                    "if manus_hand_status_progressed \"$previous\" '/hand_input msgs=11 hand=fresh sent(right=20 left=10)'; then exit 33; fi",
                    "manus_hand_status_progressed \"$previous\" '/hand_input msgs=11 hand=fresh sent(right=20 left=20)'",
                    "STRICT_MANUS_HANDS=0",
                    "manus_hand_status_progressed \"$previous\" '/hand_input msgs=11 hand=fresh sent(right=20 left=10)'",
                )), Path(directory),
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_single_fresh_status_cannot_complete_startup_wait(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_sourced(
                '\n'.join((
                    "MANUS_HAND_WAIT_TIMEOUT=1",
                    "printf '/hand_input msgs=2 hand=fresh sent(right=2 left=2)\\n' >\"$LOG_DIR/bridge.log\"",
                    "process_alive() { return 0; }",
                    "pid_matches_component() { return 0; }",
                    "if wait_for_manus_hand_stream 4242; then exit 31; fi",
                )), Path(directory),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("did not show sustained progress", result.stderr)

    def test_active_raw_input_diagnosis_does_not_blame_license(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_sourced(
                '\n'.join((
                    "printf '%s\\n' 'Calibration loaded successfully for Right glove' 'Calibration loaded successfully for Left glove' >\"$LOG_DIR/publisher.log\"",
                    "printf '/hand_input msgs=2 hand=stale sent(right=2 left=2) raw0_received=240 raw0_age_sec=0.010\\n' >\"$LOG_DIR/bridge.log\"",
                    "report_manus_readiness_diagnosis",
                )), Path(directory),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Raw MANUS messages are reaching the bridge", result.stderr)
            self.assertNotIn("Reconnect", result.stderr)
            self.assertNotIn("not detected", result.stderr)

    def test_calibration_wait_accepts_validated_cache_ready_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "CALIBRATION_WAIT_TIMEOUT=1",
                        "printf 'Startup calibration ready: source=cache\\n' >\"$LOG_DIR/bridge.log\"",
                        "kill() { return 0; }",
                        "wait_for_startup_calibration 4242",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Startup calibration ready: source=cache", result.stdout)
            self.assertIn("Startup calibration done", result.stdout)

    def test_calibration_wait_does_not_treat_pending_cache_as_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "CALIBRATION_WAIT_TIMEOUT=1",
                        "printf '%s\\n' 'Bridge ready' "
                        "'Startup calibration cache validation: hold both gloves still' "
                        '>"$LOG_DIR/bridge.log"',
                        "kill() { return 0; }",
                        "wait_for_startup_calibration 4242",
                    )
                ),
                runtime,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("did not complete", result.stderr)
            self.assertNotIn("not enabled", result.stdout)

    def test_bridge_calibration_args_include_cache_openvr_and_force(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "OPENVR_CONFIG=/tmp/openvr-machine.yaml",
                        "CALIBRATION_CACHE=/tmp/mode-a-cache.json",
                        "CALIBRATION_CACHE_CANDIDATE=/tmp/mode-a-cache.candidate",
                        "FORCE_RECALIBRATE=1",
                        "BRIDGE_CMD=(bridge)",
                        "append_bridge_calibration_args",
                        "printf '%s\\n' \"${BRIDGE_CMD[@]}\"",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                result.stdout.splitlines(),
                [
                    "bridge",
                    "--openvr-config",
                    "/tmp/openvr-machine.yaml",
                    "--calibration-cache",
                    "/tmp/mode-a-cache.json",
                    "--calibration-cache-output",
                    "/tmp/mode-a-cache.candidate",
                    "--force-recalibrate",
                ],
            )

    def test_failed_start_discards_candidate_and_preserves_permanent_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            cache = runtime / "mode-a.json"
            candidate = runtime / "mode-a.candidate"
            cache.write_text("old-cache", encoding="utf-8")
            result = self.run_sourced(
                "\n".join(
                    (
                        f"CALIBRATION_CACHE={shlex.quote(str(cache))}",
                        f"CALIBRATION_CACHE_CANDIDATE={shlex.quote(str(candidate))}",
                        "start_with_cleanup() { printf 'new-cache' >\"$CALIBRATION_CACHE_CANDIDATE\"; return 23; }",
                        "start_direct_with_receipt",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 23, result.stderr)
            self.assertEqual(cache.read_text(encoding="utf-8"), "old-cache")
            self.assertFalse(candidate.exists())

    def test_unhealthy_final_status_discards_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            cache = runtime / "mode-a.json"
            candidate = runtime / "mode-a.candidate"
            cache.write_text("old-cache", encoding="utf-8")
            result = self.run_sourced(
                "\n".join(
                    (
                        f"CALIBRATION_CACHE={shlex.quote(str(cache))}",
                        f"CALIBRATION_CACHE_CANDIDATE={shlex.quote(str(candidate))}",
                        "start_with_cleanup() { printf 'new-cache' >\"$CALIBRATION_CACHE_CANDIDATE\"; }",
                        "status() { return 7; }",
                        "stop_components() { :; }",
                        "start_direct_with_receipt",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(cache.read_text(encoding="utf-8"), "old-cache")
            self.assertFalse(candidate.exists())

    def test_healthy_start_atomically_promotes_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            cache = runtime / "state" / "mode-a.json"
            candidate = runtime / "state" / "mode-a.candidate"
            candidate.parent.mkdir()
            cache.write_text("old-cache", encoding="utf-8")
            result = self.run_sourced(
                "\n".join(
                    (
                        f"CALIBRATION_CACHE={shlex.quote(str(cache))}",
                        f"CALIBRATION_CACHE_CANDIDATE={shlex.quote(str(candidate))}",
                        "start_with_cleanup() { printf 'new-cache' >\"$CALIBRATION_CACHE_CANDIDATE\"; }",
                        "status() { return 0; }",
                        "start_direct_with_receipt",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(cache.read_text(encoding="utf-8"), "new-cache")
            self.assertFalse(candidate.exists())
            self.assertIn("Startup calibration cache saved", result.stdout)

    def test_guided_start_without_candidate_fails_and_removes_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            cache = runtime / "mode-a.json"
            candidate = runtime / "mode-a.candidate"
            receipt = runtime / "calibration.receipt"
            cache.write_text("old-cache", encoding="utf-8")
            result = self.run_sourced(
                "\n".join(
                    (
                        f"CALIBRATION_CACHE={shlex.quote(str(cache))}",
                        f"CALIBRATION_CACHE_CANDIDATE={shlex.quote(str(candidate))}",
                        f"HANDSON_CALIBRATION_RECEIPT_PATH={shlex.quote(str(receipt))}",
                        "HANDSON_CALIBRATION_RECEIPT_TOKEN=guided-token",
                        "printf 'Startup calibration ready: source=guided\\n' >\"$LOG_DIR/bridge.log\"",
                        "start_with_cleanup() { :; }",
                        "status() { return 0; }",
                        "stop_components() { :; }",
                        "start_direct_with_receipt",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(cache.read_text(encoding="utf-8"), "old-cache")
            self.assertFalse(candidate.exists())
            self.assertFalse(receipt.exists())
            self.assertIn("no cache candidate was saved", result.stderr)

    def test_manus_timeout_identifies_missing_left_glove(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "HAND=both",
                        "MANUS_HAND_WAIT_TIMEOUT=0",
                        "printf 'Calibration loaded successfully for Right glove (ID: 1234)\\n' >\"$LOG_DIR/publisher.log\"",
                        ": >\"$LOG_DIR/input.log\"",
                        "printf '/hand_input msgs=0 hand=never sent(right=0 left=0 tracker=1)\\n' >\"$LOG_DIR/bridge.log\"",
                        "if wait_for_manus_hand_stream 4242; then exit 31; fi",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("not detected: Left (detected: Right)", result.stderr)
            self.assertNotIn("not detected: Right", result.stderr)

    def test_manus_license_failure_runs_partial_component_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            cleanup_marker = runtime / "manus-license-cleaned"
            result = self.run_sourced(
                "\n".join(
                    (
                        "HAND=both",
                        "IGNORE_MANUS_LICENSE=0",
                        "MANUS_HAND_WAIT_TIMEOUT=0",
                        "printf 'No compatible license found. Please connect a license with the SDK component.\\n' >\"$LOG_DIR/publisher.log\"",
                        ": >\"$LOG_DIR/input.log\"",
                        ": >\"$LOG_DIR/bridge.log\"",
                        "start() { : >\"$STARTUP_COMPONENTS_MARKER\"; wait_for_manus_hand_stream 4242; }",
                        f"stop_components() {{ : >{shlex.quote(str(cleanup_marker))}; }}",
                        "start_with_cleanup",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 1)
            self.assertTrue(cleanup_marker.exists())
            self.assertIn("SDK license is unavailable", result.stderr)
            self.assertIn("cleaning all partially started components", result.stderr)

    def test_failed_start_always_runs_partial_component_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            cleanup_marker = runtime / "cleaned"
            result = self.run_sourced(
                "\n".join(
                    (
                        "start() { : >\"$STARTUP_COMPONENTS_MARKER\"; exit 23; }",
                        f"stop_components() {{ : >{shlex.quote(str(cleanup_marker))}; }}",
                        "start_with_cleanup",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 23)
            self.assertTrue(cleanup_marker.exists())
            self.assertIn("cleaning all partially started components", result.stderr)

    def test_calibration_timeout_also_runs_partial_component_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            cleanup_marker = runtime / "calibration-cleaned"
            result = self.run_sourced(
                "\n".join(
                    (
                        "CALIBRATION_WAIT_TIMEOUT=0",
                        "start() { : >\"$STARTUP_COMPONENTS_MARKER\"; wait_for_startup_calibration 4242; }",
                        f"stop_components() {{ : >{shlex.quote(str(cleanup_marker))}; }}",
                        "start_with_cleanup",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 1)
            self.assertTrue(cleanup_marker.exists())
            self.assertIn("Startup calibration did not complete", result.stderr)

    def test_preflight_failure_preserves_existing_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            cleanup_marker = runtime / "should-not-clean"
            result = self.run_sourced(
                "\n".join(
                    (
                        "start() { return 19; }",
                        f"stop_components() {{ : >{shlex.quote(str(cleanup_marker))}; }}",
                        "start_with_cleanup",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 19)
            self.assertFalse(cleanup_marker.exists())
            self.assertIn("existing device stack was left untouched", result.stderr)

    def test_direct_start_receipt_requires_start_and_final_status_success(self) -> None:
        cases = (
            ("success", 0, 0, 0, True),
            ("start-failed", 23, 0, 23, False),
            ("status-failed", 0, 7, 1, False),
        )
        for name, start_result, status_result, expected_result, expect_receipt in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary_directory:
                runtime = Path(temporary_directory)
                receipt = runtime / "receipt" / "calibration.done"
                token = f"token:{name}:with spaces"
                result = self.run_sourced(
                    "\n".join(
                        (
                            "HANDSON_MODE_A_DISABLE_SYSTEMD=1",
                            f"HANDSON_CALIBRATION_RECEIPT_PATH={shlex.quote(str(receipt))}",
                            f"HANDSON_CALIBRATION_RECEIPT_TOKEN={shlex.quote(token)}",
                            f"start() {{ return {start_result}; }}",
                            f"status() {{ return {status_result}; }}",
                            "run_independent start both tracker relative",
                        )
                    ),
                    runtime,
                )
                self.assertEqual(result.returncode, expected_result, result.stderr)
                self.assertEqual(receipt.exists(), expect_receipt)
                if expect_receipt:
                    self.assertEqual(receipt.read_text(encoding="utf-8"), token)

    def test_failed_stop_refuses_to_launch_duplicate_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            scan_marker = runtime / "should-not-scan"
            result = self.run_sourced(
                "\n".join(
                    (
                        "stop_components() { return 12; }",
                        f"component_process_pids() {{ : >{shlex.quote(str(scan_marker))}; }}",
                        "replace_existing_components",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 1)
            self.assertFalse(scan_marker.exists())
            self.assertIn("refusing to launch duplicates", result.stderr)

    def test_recover_without_hand_uses_saved_stack_hand(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "printf 'HAND=right\\nTRACKER_REQUIRED=1\\n' >\"$STACK_STATE_FILE\"\n"
                "test \"$(saved_stack_hand)\" = right",
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_recover_openvr_does_not_touch_manus_or_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            events = runtime / "events"
            openvr_config = runtime / "openvr.yaml"
            openvr_config.write_text("tracker_serials: {}\n", encoding="utf-8")
            result = self.run_sourced(
                "\n".join(
                    (
                        f"OPENVR_CONFIG={shlex.quote(str(openvr_config))}",
                        "require_vrserver() { return 0; }",
                        f"require_component_running() {{ printf 'require:%s\\n' \"$1\" >>{shlex.quote(str(events))}; }}",
                        "write_env_script() { : >\"$ENV_SCRIPT\"; }",
                        f"stop_component() {{ printf 'stop:%s\\n' \"$1\" >>{shlex.quote(str(events))}; }}",
                        f"start_openvr_with_retries() {{ printf 'start:openvr\\n' >>{shlex.quote(str(events))}; }}",
                        "recover_openvr",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                events.read_text(encoding="utf-8").splitlines(),
                ["require:publisher", "require:input", "require:bridge", "stop:openvr", "start:openvr"],
            )

    def test_openvr_ready_requires_pose_ready_marker_not_just_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "HAND=both",
                        "process_alive() { return 0; }",
                        "pid_matches_component() { return 0; }",
                        "printf 'Mapped LHR-R -> right_wrist\\nMapped LHR-L -> left_wrist\\n' >\"$LOG_DIR/openvr.log\"",
                        "if wait_for_openvr_client 4242 0; then echo mapped-was-accepted; exit 31; fi",
                        "printf '[OpenVR Input] Tracker stream ready: right_wrist,left_wrist\\n' >\"$LOG_DIR/openvr.log\"",
                        "wait_for_openvr_client 4242 1",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("mapped-was-accepted", result.stdout)

    def test_openvr_both_requires_roles_in_same_latest_state_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            result = self.run_sourced(
                "\n".join(
                    (
                        "HAND=both",
                        "printf '[OpenVR Input] Tracker stream state: right_wrist\\n' >\"$LOG_DIR/openvr.log\"",
                        "printf '[OpenVR Input] Tracker stream state: left_wrist\\n' >>\"$LOG_DIR/openvr.log\"",
                        "state_line=\"$(latest_openvr_tracker_state)\"",
                        "if openvr_required_roles_ready \"$state_line\"; then echo historical-roles-were-combined; exit 31; fi",
                        "HAND=left",
                        "openvr_required_roles_ready \"$state_line\"",
                        "HAND=both",
                        "printf '[OpenVR Input] Tracker stream ready: right_wrist,left_wrist\\n' >>\"$LOG_DIR/openvr.log\"",
                        "openvr_required_roles_ready \"$(latest_openvr_tracker_state)\"",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("historical-roles-were-combined", result.stdout)

    def test_timed_out_recovery_keeps_live_self_healing_client(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runtime = Path(temporary_directory)
            stop_marker = runtime / "stopped"
            result = self.run_sourced(
                "\n".join(
                    (
                        "recover_openvr() { return 7; }",
                        "component_running() { return 0; }",
                        f"stop_component() {{ : >{shlex.quote(str(stop_marker))}; }}",
                        "recover_openvr_with_cleanup",
                    )
                ),
                runtime,
            )
            self.assertEqual(result.returncode, 7)
            self.assertFalse(stop_marker.exists())
            self.assertIn("self-healing client remains running", result.stderr)

    def test_transient_launcher_uses_independent_user_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            runtime = temporary / "runtime"
            runtime.mkdir()
            call_log = temporary / "systemd-run.args"
            fake_mode = temporary / "mode-a"
            fake_mode.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"${1:-}\" == status ]]; then exit 0; fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_mode.chmod(0o700)
            fake_systemctl = temporary / "systemctl"
            fake_systemctl.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_systemctl.chmod(0o700)
            fake_systemd_run = temporary / "systemd-run"
            fake_systemd_run.write_text(
                "#!/usr/bin/env bash\n"
                f"printf '%s\\n' \"$*\" >{shlex.quote(str(call_log))}\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_systemd_run.chmod(0o700)
            env = dict(os.environ)
            env.update(
                {
                    "XDG_RUNTIME_DIR": str(runtime),
                    "HANDSON_MODE_A_SCRIPT": str(fake_mode),
                    "HANDSON_SYSTEMCTL": str(fake_systemctl),
                    "HANDSON_SYSTEMD_RUN": str(fake_systemd_run),
                }
            )
            result = subprocess.run(
                [str(TRANSIENT_LAUNCHER), "start", "both", "tracker", "relative"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            arguments = call_log.read_text(encoding="utf-8")
            self.assertIn("--scope", arguments)
            self.assertIn("_start-local both tracker relative", arguments)
            scope_file = runtime / "handson-mode-a" / "stack.scope"
            self.assertRegex(scope_file.read_text(encoding="utf-8"), r"^handson-mode-a-stack-.*\.scope\n$")

    def test_transient_receipt_requires_launch_and_final_status_success(self) -> None:
        cases = (
            ("success", 0, 0, 0, True),
            ("launch-failed", 17, 0, 17, False),
            ("status-failed", 0, 1, 1, False),
        )
        for name, launch_result, status_result, expected_result, expect_receipt in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary_directory:
                temporary = Path(temporary_directory)
                runtime = temporary / "runtime"
                runtime.mkdir()
                receipt = temporary / "receipt" / "calibration.done"
                token = f"scope-token:{name}:with spaces"
                fake_mode = temporary / "mode-a"
                fake_mode.write_text(
                    "#!/usr/bin/env bash\n"
                    f"if [[ \"${{1:-}}\" == status ]]; then exit {status_result}; fi\n"
                    "exit 0\n",
                    encoding="utf-8",
                )
                fake_mode.chmod(0o700)
                fake_systemctl = temporary / "systemctl"
                fake_systemctl.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
                fake_systemctl.chmod(0o700)
                fake_systemd_run = temporary / "systemd-run"
                fake_systemd_run.write_text(
                    "#!/usr/bin/env bash\n"
                    f"exit {launch_result}\n",
                    encoding="utf-8",
                )
                fake_systemd_run.chmod(0o700)
                env = dict(os.environ)
                env.update(
                    {
                        "XDG_RUNTIME_DIR": str(runtime),
                        "HANDSON_MODE_A_SCRIPT": str(fake_mode),
                        "HANDSON_SYSTEMCTL": str(fake_systemctl),
                        "HANDSON_SYSTEMD_RUN": str(fake_systemd_run),
                        "HANDSON_CALIBRATION_RECEIPT_PATH": str(receipt),
                        "HANDSON_CALIBRATION_RECEIPT_TOKEN": token,
                    }
                )
                result = subprocess.run(
                    [str(TRANSIENT_LAUNCHER), "start", "both", "tracker", "relative"],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, expected_result, result.stderr)
                self.assertEqual(receipt.exists(), expect_receipt)
                if expect_receipt:
                    self.assertEqual(receipt.read_text(encoding="utf-8"), token)
                    scope_file = runtime / "handson-mode-a" / "stack.scope"
                    self.assertTrue(scope_file.exists())

    def test_transient_local_fallback_uses_public_start_receipt_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            runtime = temporary / "runtime"
            runtime.mkdir()
            receipt = temporary / "receipt" / "calibration.done"
            token = "fallback-token:with spaces"
            action_log = temporary / "action"
            fake_mode = temporary / "mode-a"
            fake_mode.write_text(
                "#!/usr/bin/env bash\n"
                f"source {shlex.quote(str(MODE_A_SCRIPT))}\n"
                "start() { return 0; }\n"
                "status() { return 0; }\n"
                f"printf '%s\\n' \"${{1:-}}\" >{shlex.quote(str(action_log))}\n"
                "run_independent \"$@\"\n",
                encoding="utf-8",
            )
            fake_mode.chmod(0o700)
            unavailable_systemctl = temporary / "systemctl"
            unavailable_systemctl.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
            unavailable_systemctl.chmod(0o700)
            fake_systemd_run = temporary / "systemd-run"
            fake_systemd_run.write_text("#!/usr/bin/env bash\nexit 31\n", encoding="utf-8")
            fake_systemd_run.chmod(0o700)
            env = dict(os.environ)
            env.update(
                {
                    "XDG_RUNTIME_DIR": str(runtime),
                    "HANDSON_MODE_A_SCRIPT": str(fake_mode),
                    "HANDSON_SYSTEMCTL": str(unavailable_systemctl),
                    "HANDSON_SYSTEMD_RUN": str(fake_systemd_run),
                    "HANDSON_CALIBRATION_RECEIPT_PATH": str(receipt),
                    "HANDSON_CALIBRATION_RECEIPT_TOKEN": token,
                }
            )
            result = subprocess.run(
                [str(TRANSIENT_LAUNCHER), "start", "both", "tracker", "relative"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(action_log.read_text(encoding="utf-8"), "start\n")
            self.assertEqual(receipt.read_text(encoding="utf-8"), token)
            self.assertIn("systemd user manager is unavailable", result.stderr)

    def test_hidden_start_never_stops_its_own_transient_scope(self) -> None:
        source = MODE_A_SCRIPT.read_text(encoding="utf-8")
        start_body = source.split("\nstart() {", 1)[1].split("\nstop() {", 1)[0]
        replacement_body = source.split("\nreplace_existing_components() {", 1)[1].split(
            "\ncleanup_transient_units() {", 1
        )[0]
        self.assertIn("replace_existing_components", start_body)
        self.assertIn("stop_components", replacement_body)
        self.assertNotIn('"$0" stop', start_body)
        self.assertNotIn("stop_managed", start_body)
        self.assertNotIn("cleanup_transient_units", start_body)
        self.assertNotIn("cleanup_transient_units", replacement_body)
        self.assertIn("_start-local) start_direct_with_receipt", source)


if __name__ == "__main__":
    unittest.main()
