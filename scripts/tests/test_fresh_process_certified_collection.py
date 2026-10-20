from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from handson.data.physical_replay_certification import (
    LEGACY_RAW_ACTION_INPUT_MODE,
    LEGACY_RAW_ACTION_TRANSITION_CONTRACT,
    RECORDED_DRIVE_TARGET_INPUT_MODE,
    RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT,
)
from scripts import fresh_process_certified_collection as supervisor


FAKE_RECORDER = r'''#!/usr/bin/env python3
import json
import os
import signal
import sys
import time
from pathlib import Path

from handson.data.physical_replay_certification import (
    CERTIFICATION_MODE,
    FRESH_PROCESS_CERTIFICATION_MODE,
    RECORDED_DRIVE_TARGET_INPUT_MODE,
    RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT,
)

def option(name, default=""):
    prefix = f"--{name}="
    for index, token in enumerate(sys.argv[1:]):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == f"--{name}":
            return sys.argv[index + 2]
    return default

counter_path = Path(os.environ["FAKE_RECORDER_COUNTER"])
try:
    counter = int(counter_path.read_text(encoding="utf-8")) + 1
except FileNotFoundError:
    counter = 1
counter_path.write_text(str(counter), encoding="utf-8")
ready_file = os.environ.get("HANDSON_ISAAC_STARTUP_READY_FILE", "")
if ready_file:
    ready = Path(ready_file)
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
out_dir = Path(option("out_dir"))
out_dir.mkdir(parents=True, exist_ok=True)
if os.environ.get("FAKE_RECORDER_SKIP_FIRST", "") == "1" and counter == 1:
    raise SystemExit(
        os.EX_TEMPFAIL
        if os.environ.get("FAKE_RECORDER_UNSTABLE_FIRST", "") == "1"
        else 9
    )
demo = out_dir / f"b03_fake_demo_{counter}"
(demo / "replay.zarr").mkdir(parents=True)
(demo / "replay.zarr" / "payload.bin").write_bytes(bytes([counter]))
(demo / "recording_schema.json").write_text("{}\n", encoding="utf-8")
(demo / "collection_meta.json").write_text("{}\n", encoding="utf-8")
(demo / "videos").mkdir()
(demo / "videos" / "front.mp4").write_bytes(b"fake-mp4")
seed = int(option("seed", "0"))
manifest = {
    "task": {"name": "t09_insert_peg"},
    "recording": {
        "physical_replay_certification_required": True,
        "physical_transition_contract": RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT,
        "trajectory_input_mode": RECORDED_DRIVE_TARGET_INPUT_MODE,
        "physical_replay_goal": CERTIFICATION_MODE,
        "physical_replay_certification_mode": FRESH_PROCESS_CERTIFICATION_MODE,
        "episode_index": 1,
        "collection_run_seed": seed,
        "episode_reset_seed": seed,
        "stable_recording_origin": {
            "reset_stabilization_attempt": 1,
            "reset_stabilization_attempts_allowed": 1,
        },
    },
    "environment": {"variables": {}},
}
(demo / "replay_manifest.json").write_text(
    json.dumps(manifest), encoding="utf-8"
)
(demo / "fake_recorder_args.json").write_text(
    json.dumps({
        "argv": sys.argv[1:],
        "pid": os.getpid(),
        "stable_origin_reset_attempts": os.environ.get(
            "HANDSON_STABLE_ORIGIN_RESET_ATTEMPTS"
        ),
        "fresh_worker": os.environ.get(
            "HANDSON_FRESH_CERTIFIED_COLLECTION_WORKER"
        ),
        "deferred_worker": os.environ.get(
            "HANDSON_ISAAC_DEFERRED_ARM_WORKER"
        ),
        "e01_background": os.environ.get(
            "HANDSON_E01_POSTPROCESS_BACKGROUND"
        ),
    }),
    encoding="utf-8",
)
'''


FAKE_CERTIFIER = r'''#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import signal
import sys
import time
from pathlib import Path

from handson.data.physical_replay_certification import (
    CERTIFICATION_MODE,
    INITIALIZATION_MODE,
    INITIALIZATION_RESTORE_COUNTS_SOURCE,
    RECORDED_DRIVE_TARGET_INPUT_MODE,
    RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT,
    REPORT_FORMAT,
    REPORT_NAME,
    REPORT_VERSION,
    capture_certified_inputs,
)

parser = argparse.ArgumentParser()
parser.add_argument("--demo-dir", required=True, type=Path)
args = parser.parse_args()
demo = args.demo_dir.resolve()
fail_substring = os.environ.get("FAKE_CERT_FAIL_SUBSTRING", "")
failed = bool(fail_substring and fail_substring in demo.name)
inputs = capture_certified_inputs(demo)
report = {
    "format": REPORT_FORMAT,
    "version": REPORT_VERSION,
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": "failed" if failed else "passed",
    "reason": "intentional_test_divergence" if failed else "matched",
    "portable": not failed,
    "task_success_replay": not failed,
    "trace_match": not failed,
    "transition_outcomes_match": not failed,
    "certification_mode": CERTIFICATION_MODE,
    "initialization_mode": INITIALIZATION_MODE,
    "physical_transition_contract": RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT,
    "trajectory_input_mode": RECORDED_DRIVE_TARGET_INPUT_MODE,
    "initialization_restore_counts": {
        "robot_joints": 2,
        "rigid_bodies": 0,
        "articulation_roots": 0,
        "articulation_dofs": 0,
        "deformable_nodes": 0,
    },
    "initialization_restore_counts_source": INITIALIZATION_RESTORE_COUNTS_SOURCE,
    "initialization_readback_match": not failed,
    "post_stabilization_snapshot_restore_performed": False,
    "physics_steps_between_initialization_readback_and_action_0": 0,
    "physics_steps_between_restore_and_action_0": 0,
    "state_writes_performed": True,
    "initialization_state_writes_performed": True,
    "trajectory_state_writes_performed": False,
    "state_writes_after_action_start": False,
    "raw_commands_installed": True,
    "recorded_drive_targets_applied": True,
    "ik_recomputed": False,
    "raw_environment_step_called": False,
    "recorded_drive_target_substeps_expected": 2,
    "recorded_drive_target_substeps_executed": 2,
    "recorded_drive_target_writes_expected": 2,
    "recorded_drive_target_writes_executed": 2,
    "steps_expected": 2,
    "steps_executed": 0 if failed else 2,
    "certified_inputs": inputs,
    "certifier_pid": os.getpid(),
}
(demo / REPORT_NAME).write_text(json.dumps(report), encoding="utf-8")
if (
    os.environ.get("FAKE_CERT_SIGNAL_PARENT_ON_LAST", "") == "1"
    and len(list((demo / "physical_replay_certification_runs").glob("*.json"))) == 2
):
    os.kill(os.getppid(), signal.SIGTERM)
    time.sleep(1)
raise SystemExit(1 if failed else int(os.environ.get("FAKE_CERT_PASS_EXIT_CODE", "0")))
'''


FAKE_HUNG_RECORDER = r'''#!/usr/bin/env python3
import time
time.sleep(60)
'''


FAKE_DEFERRED_WORKER = r'''#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
Path(os.environ["FAKE_DEFERRED_LOG"]).write_text(
    json.dumps({"argv": sys.argv[1:], "pid": os.getpid()}),
    encoding="utf-8",
)
'''


def _write_executable(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides):
    out_dir = tmp_path / "formal"
    arguments = [
        "--exp_name=t09_insert_peg",
        "--sim_backend=isaacsim",
        "--successes_needed=1",
        f"--out_dir={out_dir}",
        "--seed=100",
        "--fresh-certification-runs=3",
        "--fresh-max-attempts=3",
        "--fresh-termination-grace-seconds=1",
    ]
    for name, value in overrides.items():
        arguments.append(f"--{name}={value}")
    monkeypatch.setenv("HANDSON_COLLECTION_SESSION_ID", f"test-{tmp_path.name}")
    return supervisor.build_config(arguments)


@pytest.fixture(autouse=True)
def _isolate_full_certification_report_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep lifecycle tests independent of the full report/schema fixture."""
    monkeypatch.setattr(
        supervisor,
        "certification_report_blocker",
        lambda _demo_dir, _report: None,
    )
    monkeypatch.setattr(
        supervisor,
        "certification_upload_blocker",
        lambda _demo_dir: None,
    )
    monkeypatch.setattr(
        supervisor,
        "fresh_process_acceptance_upload_blocker",
        lambda _demo_dir: None,
    )


def test_build_config_places_private_roots_outside_formal_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)

    assert config.certification_runs == 3
    assert config.staging_root.parent == config.out_dir.parent
    assert config.quarantine_root.parent == config.out_dir.parent
    assert not supervisor._is_relative_to(config.staging_root, config.out_dir)
    assert not supervisor._is_relative_to(config.quarantine_root, config.out_dir)

    with pytest.raises(supervisor.SupervisorError, match="outside and disjoint"):
        supervisor.build_config(
            [
                "--sim_backend=isaacsim",
                f"--out_dir={config.out_dir}",
                f"--fresh-staging-root={config.out_dir / 'private'}",
            ]
        )
    with pytest.raises(supervisor.SupervisorError, match="at least 3"):
        supervisor.build_config(
            [
                "--sim_backend=isaacsim",
                f"--out_dir={config.out_dir}",
                "--fresh-certification-runs=2",
            ]
        )
    with pytest.raises(supervisor.SupervisorError, match="source repository"):
        supervisor.build_config(
            [
                "--sim_backend=isaacsim",
                f"--out_dir={config.out_dir}",
                f"--fresh-staging-root={supervisor.REPO_ROOT / 'private'}",
            ]
        )


def test_default_repo_output_keeps_private_lifecycle_data_outside_git_tree(
) -> None:
    formal = supervisor.REPO_ROOT / "demos"

    staging = supervisor._default_private_root(
        formal,
        "fresh_process_staging",
    )
    quarantine = supervisor._default_private_root(
        formal,
        "physical_replay_quarantine",
    )

    assert not supervisor._is_relative_to(staging, supervisor.REPO_ROOT)
    assert not supervisor._is_relative_to(quarantine, supervisor.REPO_ROOT)
    assert staging.parent == quarantine.parent


def test_recorder_command_forces_one_episode_ground_truth_and_new_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)

    first = supervisor._recorder_command(config, 1, tmp_path / "attempt-1")
    second = supervisor._recorder_command(config, 2, tmp_path / "attempt-2")

    assert "--successes_needed=1" in first
    assert "--certify_physical_replay=false" in first
    assert "--record_physx_ground_truth=true" in first
    assert "--seed=100" in first
    assert "--seed=101" in second
    assert sum(item.startswith("--successes_needed=") for item in first) == 1


def test_fresh_fallback_drops_inherited_persistent_deferred_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    monkeypatch.setenv("HANDSON_DEFER_PHYSICAL_REPLAY_CERTIFICATION", "1")

    child_env = supervisor._recorder_environment(
        config,
        startup_ready_file=tmp_path / "ready.json",
        startup_status_file=tmp_path / "status.json",
    )

    assert "HANDSON_DEFER_PHYSICAL_REPLAY_CERTIFICATION" not in child_env
    assert child_env["HANDSON_FRESH_CERTIFIED_COLLECTION_WORKER"] == "1"
    assert child_env["HANDSON_ISAAC_WORKER_STATUS_FILE"] == str(
        tmp_path / "status.json"
    )
    assert child_env["HANDSON_ISAAC_DEFERRED_ARM_WORKER"] == "1"
    assert child_env["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] == "1"
    assert "HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY" not in child_env


def _write_manifest(
    tmp_path: Path,
    recording: dict,
    *,
    task_name: str = "t09_insert_peg",
) -> Path:
    demo = tmp_path / "demo"
    demo.mkdir(parents=True)
    recording = dict(recording)
    recording.setdefault(
        "physical_transition_contract",
        RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT,
    )
    recording.setdefault(
        "trajectory_input_mode",
        RECORDED_DRIVE_TARGET_INPUT_MODE,
    )
    (demo / "replay_manifest.json").write_text(
        json.dumps({"task": {"name": task_name}, "recording": recording}),
        encoding="utf-8",
    )
    return demo


def test_manifest_binding_enforces_fresh_process_per_attempt_contract(
    tmp_path: Path,
) -> None:
    demo = _write_manifest(
        tmp_path,
        {
            "physical_replay_certification_required": True,
            "episode_index": 1,
            "collection_run_seed": 117,
            "episode_reset_seed": 117,
            "stable_recording_origin": {
                "reset_stabilization_attempt": 1,
                "reset_stabilization_attempts_allowed": 1,
            },
        },
    )

    supervisor._bind_manifest_to_supervisor_contract(
        demo,
        certification_runs=3,
    )

    recording = json.loads(
        (demo / "replay_manifest.json").read_text(encoding="utf-8")
    )["recording"]
    assert recording["fresh_recorder_process_per_attempt_required"] is True
    assert recording["fresh_recorder_process_per_episode_required"] is True


def test_manifest_binding_allows_legacy_raw_action_only_for_e01(
    tmp_path: Path,
) -> None:
    recording = {
        "physical_replay_certification_required": True,
        "physical_transition_contract": LEGACY_RAW_ACTION_TRANSITION_CONTRACT,
        "trajectory_input_mode": LEGACY_RAW_ACTION_INPUT_MODE,
        "episode_index": 1,
        "collection_run_seed": 117,
        "episode_reset_seed": 117,
        "stable_recording_origin": {
            "reset_stabilization_attempt": 1,
            "reset_stabilization_attempts_allowed": 1,
        },
    }
    e01_demo = _write_manifest(
        tmp_path / "e01",
        recording,
        task_name="e01_fold_garment",
    )
    supervisor._bind_manifest_to_supervisor_contract(
        e01_demo,
        certification_runs=3,
    )

    b03_demo = _write_manifest(
        tmp_path / "b03",
        recording,
        task_name="t09_insert_peg",
    )
    with pytest.raises(supervisor.SupervisorError, match="allowed only"):
        supervisor._bind_manifest_to_supervisor_contract(
            b03_demo,
            certification_runs=3,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("episode_index", 2, "episode_index=1"),
        ("episode_index", True, "episode_index=1"),
        ("episode_reset_seed", 118, "episode_reset_seed equal"),
        ("episode_reset_seed", True, "episode_reset_seed equal"),
        ("reset_stabilization_attempt", 2, "attempt 1/1"),
        ("reset_stabilization_attempt", True, "attempt 1/1"),
        ("reset_stabilization_attempts_allowed", 2, "attempt 1/1"),
        ("reset_stabilization_attempts_allowed", True, "attempt 1/1"),
    ],
)
def test_manifest_binding_rejects_nonfresh_attempt_contract(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    recording = {
        "physical_replay_certification_required": True,
        "episode_index": 1,
        "collection_run_seed": 117,
        "episode_reset_seed": 117,
        "stable_recording_origin": {
            "reset_stabilization_attempt": 1,
            "reset_stabilization_attempts_allowed": 1,
        },
    }
    if field.startswith("reset_stabilization_"):
        recording["stable_recording_origin"][field] = value
    else:
        recording[field] = value
    demo = _write_manifest(tmp_path, recording)

    with pytest.raises(supervisor.SupervisorError, match=message):
        supervisor._bind_manifest_to_supervisor_contract(
            demo,
            certification_runs=3,
        )


def test_episode_discovery_ignores_incomplete_recorder_transaction(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    partial = attempt / ".recording_staging" / "demo.partial-123"
    (partial / "replay.zarr").mkdir(parents=True)

    assert supervisor._episode_directories(attempt) == []


def test_supervisor_quarantines_failure_then_atomically_publishes_passed_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _write_executable(tmp_path / "fake_recorder.py", FAKE_RECORDER)
    certifier = _write_executable(tmp_path / "fake_certifier.py", FAKE_CERTIFIER)
    counter = tmp_path / "recorder_counter.txt"
    monkeypatch.setattr(supervisor, "RECORDER", recorder)
    monkeypatch.setattr(supervisor, "CERTIFIER", certifier)
    monkeypatch.setenv("FAKE_RECORDER_COUNTER", str(counter))
    monkeypatch.setenv("FAKE_CERT_FAIL_SUBSTRING", "demo_1")
    config = _config(tmp_path, monkeypatch)

    assert supervisor.supervise(config) == 0

    formal_demos = [
        path.parent
        for path in config.out_dir.rglob("replay.zarr")
        if path.is_dir()
    ]
    assert [path.name for path in formal_demos] == ["b03_fake_demo_2"]
    accepted = json.loads(
        (formal_demos[0] / supervisor.ACCEPTANCE_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert accepted["status"] == "passed"
    assert accepted["certification_runs_required"] == 3
    assert accepted["certification_runs_passed"] == 3
    certification_pids = {
        run["pid"] for run in accepted["certification_runs"]
    }
    assert len(certification_pids) == 3
    assert accepted["recorder"]["pid"] not in certification_pids
    assert accepted["fresh_recorder_process_per_attempt"] is True
    assert supervisor.certification_upload_blocker(formal_demos[0]) is None

    recorder_args = json.loads(
        (formal_demos[0] / "fake_recorder_args.json").read_text(
            encoding="utf-8"
        )
    )["argv"]
    assert "--successes_needed=1" in recorder_args
    assert "--seed=101" in recorder_args
    assert len(list((formal_demos[0] / supervisor.RUN_REPORT_DIRECTORY).glob("*.json"))) == 3

    quarantined = list(
        (config.quarantine_root / config.session_id).rglob(
            supervisor.REJECTION_NAME
        )
    )
    assert len(quarantined) == 1
    rejection = json.loads(quarantined[0].read_text(encoding="utf-8"))
    assert rejection["status"] == "certification_failed"
    assert not list(config.staging_root.rglob("replay.zarr"))
    assert counter.read_text(encoding="utf-8") == "2"


def test_canonical_camera_worker_starts_only_after_formal_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _write_executable(tmp_path / "fake_recorder.py", FAKE_RECORDER)
    certifier = _write_executable(tmp_path / "fake_certifier.py", FAKE_CERTIFIER)
    deferred = _write_executable(
        tmp_path / "fake_deferred_worker.py",
        FAKE_DEFERRED_WORKER,
    )
    deferred_log = tmp_path / "deferred.json"
    monkeypatch.setattr(supervisor, "RECORDER", recorder)
    monkeypatch.setattr(supervisor, "CERTIFIER", certifier)
    monkeypatch.setattr(supervisor, "DEFERRED_ARM_WORKER", deferred)
    monkeypatch.setenv("FAKE_RECORDER_COUNTER", str(tmp_path / "counter.txt"))
    monkeypatch.setenv("FAKE_DEFERRED_LOG", str(deferred_log))
    monkeypatch.delenv("HANDSON_ISAACLAB_DEFER_ARM_RENDER", raising=False)
    config = _config(tmp_path, monkeypatch)

    assert supervisor.supervise(config) == 0
    deadline = time.monotonic() + 2.0
    while not deferred_log.is_file() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert deferred_log.is_file()

    demo = next(path.parent for path in config.out_dir.rglob("replay.zarr"))
    deferred_args = json.loads(deferred_log.read_text(encoding="utf-8"))["argv"]
    assert str(demo) in deferred_args
    assert not any(str(config.staging_root) in item for item in deferred_args)
    recorder_info = json.loads(
        (demo / "fake_recorder_args.json").read_text(encoding="utf-8")
    )
    assert recorder_info["stable_origin_reset_attempts"] == "1"
    assert recorder_info["fresh_worker"] == "1"
    assert recorder_info["deferred_worker"] == "1"
    assert recorder_info["e01_background"] == "0"


def test_zero_episode_retries_and_passed_report_beats_nonzero_kit_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _write_executable(tmp_path / "fake_recorder.py", FAKE_RECORDER)
    certifier = _write_executable(tmp_path / "fake_certifier.py", FAKE_CERTIFIER)
    counter = tmp_path / "recorder_counter.txt"
    monkeypatch.setattr(supervisor, "RECORDER", recorder)
    monkeypatch.setattr(supervisor, "CERTIFIER", certifier)
    monkeypatch.setenv("FAKE_RECORDER_COUNTER", str(counter))
    monkeypatch.setenv("FAKE_RECORDER_SKIP_FIRST", "1")
    monkeypatch.setenv("FAKE_CERT_PASS_EXIT_CODE", "17")
    config = _config(tmp_path, monkeypatch)

    assert supervisor.supervise(config) == 0

    demos = list(config.out_dir.rglob("replay.zarr"))
    assert len(demos) == 1
    acceptance = json.loads(
        (demos[0].parent / supervisor.ACCEPTANCE_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert [run["returncode"] for run in acceptance["certification_runs"]] == [
        17,
        17,
        17,
    ]
    assert len(
        list(
            (config.quarantine_root / config.session_id).rglob(
                supervisor.REJECTION_NAME
            )
        )
    ) == 1


def test_retryable_attempt_exit_is_quarantined_and_retried_in_new_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _write_executable(tmp_path / "fake_recorder.py", FAKE_RECORDER)
    certifier = _write_executable(tmp_path / "fake_certifier.py", FAKE_CERTIFIER)
    counter = tmp_path / "recorder_counter.txt"
    monkeypatch.setattr(supervisor, "RECORDER", recorder)
    monkeypatch.setattr(supervisor, "CERTIFIER", certifier)
    monkeypatch.setenv("FAKE_RECORDER_COUNTER", str(counter))
    monkeypatch.setenv("FAKE_RECORDER_SKIP_FIRST", "1")
    monkeypatch.setenv("FAKE_RECORDER_UNSTABLE_FIRST", "1")
    config = _config(tmp_path, monkeypatch)

    assert supervisor.supervise(config) == 0

    rejections = list(
        (config.quarantine_root / config.session_id).rglob(
            supervisor.REJECTION_NAME
        )
    )
    assert len(rejections) == 1
    rejection = json.loads(rejections[0].read_text(encoding="utf-8"))
    assert rejection["status"] == "recording_attempt_rejected"
    assert rejection["retryable"] is True
    assert rejection["recorder"]["returncode"] == os.EX_TEMPFAIL
    assert counter.read_text(encoding="utf-8") == "2"
    assert len(list(config.out_dir.rglob("replay.zarr"))) == 1


def test_startup_watchdog_kills_hung_recorder_and_quarantines_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _write_executable(
        tmp_path / "hung_recorder.py",
        FAKE_HUNG_RECORDER,
    )
    monkeypatch.setattr(supervisor, "RECORDER", recorder)
    monkeypatch.setenv("HANDSON_ISAAC_STARTUP_TIMEOUT_SECONDS", "0.2")
    config = _config(tmp_path, monkeypatch)

    with pytest.raises(supervisor.SupervisorError, match="startup failed"):
        supervisor.supervise(config)

    rejections = list(
        (config.quarantine_root / config.session_id).rglob(
            supervisor.REJECTION_NAME
        )
    )
    assert len(rejections) == 1
    rejection = json.loads(rejections[0].read_text(encoding="utf-8"))
    assert rejection["status"] == "recorder_startup_failed"
    assert not list(config.out_dir.rglob("replay.zarr"))


def test_stop_after_last_report_wins_race_and_prevents_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _write_executable(tmp_path / "fake_recorder.py", FAKE_RECORDER)
    certifier = _write_executable(tmp_path / "fake_certifier.py", FAKE_CERTIFIER)
    monkeypatch.setattr(supervisor, "RECORDER", recorder)
    monkeypatch.setattr(supervisor, "CERTIFIER", certifier)
    monkeypatch.setenv("FAKE_RECORDER_COUNTER", str(tmp_path / "counter.txt"))
    monkeypatch.setenv("FAKE_CERT_SIGNAL_PARENT_ON_LAST", "1")
    config = _config(tmp_path, monkeypatch)

    with pytest.raises(supervisor.StopRequested):
        supervisor.supervise(config)

    assert not list(config.out_dir.rglob("replay.zarr"))
    rejections = list(
        (config.quarantine_root / config.session_id).rglob(
            supervisor.REJECTION_NAME
        )
    )
    assert len(rejections) == 1
    rejection = json.loads(rejections[0].read_text(encoding="utf-8"))
    assert rejection["status"] == "interrupted"


def test_process_group_cleanup_terminates_active_child() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )

    supervisor._kill_process_group(
        process,
        initial_signal=signal.SIGTERM,
        grace_seconds=1.0,
    )

    assert process.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(process.pid, 0)
