from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "mode_a_handson.sh"


class ModeATrackerPreflightTest(unittest.TestCase):
    def run_shell(
        self,
        body: str,
        runtime_dir: str,
        *,
        argv0: str = "bash",
    ) -> subprocess.CompletedProcess[str]:
        command = f"source {shlex.quote(str(SCRIPT))}\n{body}"
        env = dict(os.environ)
        env["XDG_RUNTIME_DIR"] = runtime_dir
        return subprocess.run(
            ["bash", "-c", command, argv0],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_missing_vrserver_fails_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as runtime_dir:
            result = self.run_shell(
                "pgrep() { return 1; }\nrequire_vrserver",
                runtime_dir,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("SteamVR (vrserver) is required", result.stderr)

    def test_dead_openvr_process_prints_log_tail_and_fails(self) -> None:
        with tempfile.TemporaryDirectory() as runtime_dir:
            log_path = Path(runtime_dir) / "openvr.log"
            log_path.write_text("OPENVR_STARTUP_FAILURE_MARKER\n", encoding="utf-8")
            result = self.run_shell(
                (
                    "false & pid=$!\n"
                    "wait \"$pid\" || true\n"
                    f"verify_started_process openvr_input \"$pid\" {shlex.quote(str(log_path))}"
                ),
                runtime_dir,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("openvr_input exited during startup", result.stderr)
        self.assertIn("OPENVR_STARTUP_FAILURE_MARKER", result.stderr)

    def test_live_openvr_process_passes_health_check(self) -> None:
        with tempfile.TemporaryDirectory() as runtime_dir:
            log_path = Path(runtime_dir) / "openvr.log"
            result = self.run_shell(
                (
                    "bash -c 'exec -a openvr_input sleep 10' & pid=$!\n"
                    f"verify_started_process openvr_input \"$pid\" {shlex.quote(str(log_path))}\n"
                    "kill \"$pid\"\n"
                    "wait \"$pid\" || true"
                ),
                runtime_dir,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_tracker_preflight_failures_do_not_call_stop(self) -> None:
        failure_bodies = {
            "missing_vrserver": (
                "HAND=both\n"
                "TRACKER=tracker\n"
                "write_env_script() { :; }\n"
                "pgrep() { return 1; }\n"
                "start"
            ),
            "binding_failure": (
                "HAND=both\n"
                "TRACKER=tracker\n"
                "write_env_script() { :; }\n"
                "prepare_openvr_config() { return 23; }\n"
                "start"
            ),
        }
        for name, body in failure_bodies.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as runtime_dir:
                runtime_path = Path(runtime_dir)
                stop_marker = runtime_path / "stop-called"
                fake_self = runtime_path / "mode-a-self"
                fake_self.write_text(
                    "#!/usr/bin/env bash\n"
                    f"printf '%s\\n' \"$*\" >> {shlex.quote(str(stop_marker))}\n",
                    encoding="utf-8",
                )
                fake_self.chmod(0o700)

                result = self.run_shell(
                    body,
                    runtime_dir,
                    argv0=str(fake_self),
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(
                    stop_marker.exists(),
                    f"preflight failure called stop: {stop_marker.read_text() if stop_marker.exists() else ''}",
                )


if __name__ == "__main__":
    unittest.main()
