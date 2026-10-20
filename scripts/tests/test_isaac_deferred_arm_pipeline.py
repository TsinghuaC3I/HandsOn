from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from handson.data.episode_publication_lock import (
    INHERITED_EPISODE_PUBLICATION_LOCK_FD_ENV,
)
from scripts.record_demos_zarr import (
    _likely_poisoned_isaac_backend_error,
    _launch_pending_isaac_deferred_arm_replays,
    _run_isaac_deferred_arm_replay,
    _run_isaac_deferred_collection_supervisor,
    _saved_demos_for_deferred_arm_replay,
    _stop_isaac_deferred_collection_worker,
    _uses_isaac_deferred_arm_pipeline,
    _wait_for_isaac_collection_worker,
)
from scripts.isaac_deferred_arm_replay_worker import (
    DEFERRED_COLLECTION_ONLY_ENV_KEYS,
    _render_env,
    _run_one,
)


class _FinishedProcess:
    returncode = 0
    pid = 9502

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def poll(self) -> int:
        return self.returncode


def test_cuda_failure_classifier_ignores_task_keyerror() -> None:
    assert not _likely_poisoned_isaac_backend_error(KeyError("cup_poses"))


def test_cuda_failure_classifier_follows_nested_causes() -> None:
    outer = RuntimeError("YCB04 startup failed")
    outer.__cause__ = RuntimeError(
        "CUDA error 700: an illegal memory access was encountered"
    )

    assert _likely_poisoned_isaac_backend_error(outer)


def _write_canonical_arm_videos(demo_dir: Path) -> None:
    videos = demo_dir / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(
        ("main.mp4", "wrist_left.mp4", "wrist_right.mp4"),
        start=1,
    ):
        (videos / name).write_bytes(bytes([index]) * (index + 3))


def _write_bc021_garbage_video_contract(demo_dir: Path) -> None:
    task_name = "t01_align_blocks_in_midair"
    frame_count = 3
    fps = 30
    camera_profile = {
        "version": "camera_pose_v6",
        "head": {
            "projection": "pinhole",
            "eye_xyz_m": [-0.675, 0.0, 1.5885],
            "quat_wxyz_opengl": [1.0, 0.0, 0.0, 0.0],
            "vertical_fov_deg": 80.0,
        },
        "wrist": {
            "output_projection": "pinhole",
            "output_resolution": [96, 64],
            "horizontal_fov_deg": 100.0,
            "rendered_frames_are_already_rectified": True,
            "K_new": [[70.0, 0.0, 47.5], [0.0, 70.0, 31.5], [0.0, 0.0, 1.0]],
            "mount_pose_xyz_quat_wxyz_by_side": {
                "left": [0.01, 0.02, 0.03, 1.0, 0.0, 0.0, 0.0],
                "right": [-0.01, 0.02, 0.03, 1.0, 0.0, 0.0, 0.0],
            },
        },
    }
    (demo_dir / "collection_meta.json").write_text(
        json.dumps(
            {
                "task_name": task_name,
                "step_count": frame_count,
                "video_fps": fps,
                "recording_camera_profile": camera_profile,
                "canonical_camera_postrender_required": True,
                "online_camera_capture_scope": "all_canonical_postrender",
            }
        ),
        encoding="utf-8",
    )
    (demo_dir / "replay_manifest.json").write_text(
        json.dumps(
            {
                "task": {"name": task_name},
                "recording": {
                    "step_count": frame_count,
                    "video_fps": fps,
                    "recording_camera_profile": camera_profile,
                    "canonical_camera_postrender_required": True,
                    "online_camera_capture_scope": "all_canonical_postrender",
                },
            }
        ),
        encoding="utf-8",
    )
    action_meta = demo_dir / "replay.zarr" / "data" / "action" / ".zarray"
    action_meta.parent.mkdir(parents=True)
    action_meta.write_text(
        json.dumps({"shape": [frame_count, 54]}), encoding="utf-8"
    )
    videos = demo_dir / "videos"
    videos.mkdir()
    aggregate = {"camera_info": {}}
    for camera_key in ("main", "wrist_left", "wrist_right"):
        sidecar = {
            "schema": "handson_video_camera_info.v1",
            "camera_key": camera_key,
            "video_file": f"{camera_key}.mp4",
            "frame": {"width": 96, "height": 64},
            "recording_camera_profile_version": "camera_pose_v6",
        }
        if camera_key == "main":
            sidecar["sim_pinhole_intrinsics"] = {
                "image_width": 96,
                "image_height": 64,
                "fovy_deg": 80.0,
            }
        else:
            sidecar["saved_projection"] = {
                "model": "pinhole",
                "image_width": 96,
                "image_height": 64,
                "horizontal_fov_deg": 100.0,
                "K": camera_profile["wrist"]["K_new"],
                "D": [0.0, 0.0, 0.0, 0.0],
                "already_rectified": True,
                "must_not_rectify_again": True,
            }
        aggregate["camera_info"][camera_key] = sidecar
        (videos / f"{camera_key}_camera_info.json").write_text(
            json.dumps(sidecar), encoding="utf-8"
        )
        (videos / f"{camera_key}.mp4").write_bytes(
            f"garbage-{camera_key}".encode()
        )
    aggregate["schema"] = "handson_video_camera_info.v1"
    (videos / "camera_info.json").write_text(json.dumps(aggregate), encoding="utf-8")


class _InterruptedProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.wait_count = 0
        self.wait_timeouts: list[float | None] = []
        self.signals: list[int] = []

    def wait(self, timeout: float | None = None) -> int:
        self.wait_count += 1
        self.wait_timeouts.append(timeout)
        if self.wait_count == 1:
            raise KeyboardInterrupt
        self.returncode = -2
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class _HangingProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.pid = 9503
        self.signals: list[int] = []
        self.wait_timeouts: list[float | None] = []
        self.terminate_count = 0
        self.kill_count = 0

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("isaac-worker", timeout)
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        self.returncode = -sig

    def terminate(self) -> None:
        self.terminate_count += 1
        self.returncode = -15

    def kill(self) -> None:
        self.kill_count += 1
        self.returncode = -9


class _TermIgnoringProcess(_HangingProcess):
    def terminate(self) -> None:
        self.terminate_count += 1


def test_deferred_arm_pipeline_includes_ycb04_explicitly() -> None:
    assert _uses_isaac_deferred_arm_pipeline(
        "t08_hammer_nail",
        "isaacsim",
        "1",
    )
    assert _uses_isaac_deferred_arm_pipeline(
        "c07_pitcher_pour_cup",
        "isaacsim",
        "1",
    )
    assert _uses_isaac_deferred_arm_pipeline(
        "t41_sort_nested_cups",
        "isaacsim",
        "1",
    )
    assert not _uses_isaac_deferred_arm_pipeline(
        "t08_hammer_nail",
        "isaacsim",
        "0",
    )
    assert not _uses_isaac_deferred_arm_pipeline(
        "e01_fold_garment",
        "isaacsim",
        "1",
    )
    assert not _uses_isaac_deferred_arm_pipeline(
        "t08_hammer_nail",
        "mujoco",
        "1",
    )
    assert _uses_isaac_deferred_arm_pipeline(
        "t09_insert_peg",
        "isaacsim_newton",
        "1",
    )
    assert not _uses_isaac_deferred_arm_pipeline(
        "t26_place_fruit_in_bucket",
        "isaacsim_newton",
        "1",
    )
    assert not _uses_isaac_deferred_arm_pipeline(
        "t27_place_largest_fitting_ball",
        "isaacsim",
        "1",
    )


def test_deferred_arm_replay_forces_complete_robot_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    completed = subprocess.CompletedProcess([], 0, stdout="published\n")
    for key in DEFERRED_COLLECTION_ONLY_ENV_KEYS:
        monkeypatch.setenv(key, "1")

    with patch(
        "scripts.record_demos_zarr.subprocess.run",
        return_value=completed,
    ) as run:
        result = _run_isaac_deferred_arm_replay(
            demo_dir,
            exp_name="t08_hammer_nail",
            video_fps=30,
        )

    command = run.call_args.args[0]
    child_env = run.call_args.kwargs["env"]
    assert result is completed
    assert command[1].endswith("isaac_render_arm_replay.py")
    assert f"--arm_replay_demo_dir={demo_dir.resolve()}" in command
    assert "--exp_name=t08_hammer_nail" in command
    assert "--arm_replay_fps=30" in command
    assert child_env["HANDSON_ISAACLAB_DEFER_ARM_RENDER"] == "0"
    for key in DEFERRED_COLLECTION_ONLY_ENV_KEYS:
        assert key not in child_env


def test_background_arm_replay_worker_drops_deferred_collection_fast_env(
    monkeypatch,
) -> None:
    for key in DEFERRED_COLLECTION_ONLY_ENV_KEYS:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("HANDSON_ISAAC_DEFERRED_ARM_WORKER", "1")

    child_env = _render_env()

    assert child_env["HANDSON_ISAACLAB_DEFER_ARM_RENDER"] == "0"
    assert "HANDSON_ISAAC_DEFERRED_ARM_WORKER" not in child_env
    assert INHERITED_EPISODE_PUBLICATION_LOCK_FD_ENV not in child_env
    for key in DEFERRED_COLLECTION_ONLY_ENV_KEYS:
        if key in {
            "HANDSON_ISAACLAB_DEFER_CAMERA_RENDER",
            "HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY",
        }:
            assert child_env[key] == "0"
        else:
            assert key not in child_env
    assert child_env["HANDSON_ISAACSIM_ULTRA_RENDERING"] == "1"
    assert child_env["HANDSON_ISAACSIM_PATH_TRACING_SPP"] == "4"
    assert child_env["HANDSON_ISAACSIM_PATH_TRACING_TOTAL_SPP"] == "16"
    assert child_env["HANDSON_ISAACSIM_PATH_TRACING_MAX_BOUNCES"] == "6"
    assert child_env["HANDSON_ISAACSIM_ACCURATE_CAMERA_VISIBILITY"] == "1"


def test_background_arm_worker_only_marks_complete_with_hashed_videos(
    tmp_path: Path,
) -> None:
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_canonical_arm_videos(demo_dir)

    with patch(
        "scripts.isaac_deferred_arm_replay_worker.subprocess.Popen",
        return_value=_FinishedProcess(),
    ) as popen:
        assert (
            _run_one(
                demo_dir,
                exp_name="a01_pick_apple",
                video_fps=30,
                log_path=None,
                force=False,
            )
            == 0
        )

    inherited_fd = int(
        popen.call_args.kwargs["env"][
            INHERITED_EPISODE_PUBLICATION_LOCK_FD_ENV
        ]
    )
    assert inherited_fd >= 0
    assert popen.call_args.kwargs["pass_fds"] == (inherited_fd,)
    status = json.loads(
        (demo_dir / "deferred_arm_replay_status.json").read_text()
    )
    assert status["status"] == "complete"
    assert status["artifact_validation_passed"] is True
    assert {
        row["path"] for row in status["artifact_proof"]["files"]
    } == {
        "videos/main.mp4",
        "videos/wrist_left.mp4",
        "videos/wrist_right.mp4",
    }


def test_background_arm_worker_rerenders_legacy_complete_without_proof(
    tmp_path: Path,
) -> None:
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_canonical_arm_videos(demo_dir)
    (demo_dir / "deferred_arm_replay_status.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )

    with patch(
        "scripts.isaac_deferred_arm_replay_worker.subprocess.Popen",
        return_value=_FinishedProcess(),
    ) as popen:
        assert (
            _run_one(
                demo_dir,
                exp_name="a01_pick_apple",
                video_fps=30,
                log_path=None,
                force=False,
            )
            == 0
        )

    popen.assert_called_once()
    status = json.loads(
        (demo_dir / "deferred_arm_replay_status.json").read_text()
    )
    assert status["artifact_validation_passed"] is True
    assert status["artifact_proof"]["files"]


def test_background_arm_worker_fails_closed_when_video_is_missing(
    tmp_path: Path,
) -> None:
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()

    with patch(
        "scripts.isaac_deferred_arm_replay_worker.subprocess.Popen",
        return_value=_FinishedProcess(),
    ):
        assert (
            _run_one(
                demo_dir,
                exp_name="a01_pick_apple",
                video_fps=30,
                log_path=None,
                force=False,
            )
            == 1
        )

    status = json.loads(
        (demo_dir / "deferred_arm_replay_status.json").read_text()
    )
    assert status["status"] == "failed"
    assert status["artifact_validation_passed"] is False
    assert "main.mp4" in status["error"]


def test_bc021_worker_does_not_mark_garbage_mp4s_complete(
    tmp_path: Path,
) -> None:
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_bc021_garbage_video_contract(demo_dir)

    with patch(
        "scripts.isaac_deferred_arm_replay_worker.subprocess.Popen",
        return_value=_FinishedProcess(),
    ):
        assert (
            _run_one(
                demo_dir,
                exp_name="t01_align_blocks_in_midair",
                video_fps=30,
                log_path=None,
                force=False,
            )
            == 1
        )

    status = json.loads(
        (demo_dir / "deferred_arm_replay_status.json").read_text()
    )
    assert status["status"] == "failed"
    assert status["artifact_validation_passed"] is False
    assert "completely decoded" in status["error"]
    assert "artifact_proof" not in status


def test_simplified_worker_fails_closed_without_validated_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    _write_canonical_arm_videos(demo_dir)
    (demo_dir / "replay_manifest.json").write_text(
        json.dumps(
            {
                "recording": {
                    "arm_control_profile": "rokae_simplified_post_ik"
                }
            }
        ),
        encoding="utf-8",
    )
    planner_python = tmp_path / "mujoco-python"
    planner_python.write_text("#!/bin/sh\n", encoding="utf-8")
    planner_python.chmod(0o755)
    monkeypatch.setenv("HANDSON_PYTHON", str(planner_python))

    with patch(
        "scripts.isaac_deferred_arm_replay_worker.subprocess.Popen",
        return_value=_FinishedProcess(),
    ) as popen:
        assert (
            _run_one(
                demo_dir,
                exp_name="t43_squeeze_scissors",
                video_fps=30,
                log_path=None,
                force=False,
            )
            == 1
        )

    assert popen.call_count == 2
    commands = [call.args[0] for call in popen.call_args_list]
    assert commands[0][0] == str(planner_python)
    assert commands[0][1].endswith("simplified_post_ik_plan.py")
    assert commands[1][1].endswith("isaac_render_arm_replay.py")
    status = json.loads(
        (demo_dir / "deferred_arm_replay_status.json").read_text()
    )
    assert status["status"] == "failed"
    assert "replay.zarr is missing" in status["error"]


def test_supervisor_discovers_only_new_demos_from_its_session(
    tmp_path: Path,
) -> None:
    old_demo = tmp_path / "t08_hammer_nail_demo_old"
    own_demo = tmp_path / "t08_hammer_nail_demo_own"
    other_demo = tmp_path / "t08_hammer_nail_demo_other"
    for path, session_id in (
        (old_demo, "session"),
        (own_demo, "session"),
        (other_demo, "other"),
    ):
        (path / "replay.zarr").mkdir(parents=True)
        (path / "collection_meta.json").write_text(
            f'{{"collection_session_id": "{session_id}"}}',
            encoding="utf-8",
        )

    demos = _saved_demos_for_deferred_arm_replay(
        tmp_path,
        exp_name="t08_hammer_nail",
        session_id="session",
        existing={old_demo.resolve()},
    )

    assert demos == [own_demo]


def test_supervisor_runs_collection_worker_before_arm_replays(
    tmp_path: Path,
) -> None:
    worker = _FinishedProcess()

    with (
        patch("scripts.record_demos_zarr.sys.argv", ["record_demos_zarr.py"]),
        patch(
            "scripts.record_demos_zarr.subprocess.Popen",
            return_value=worker,
        ) as popen,
        patch(
            "scripts.record_demos_zarr._launch_pending_isaac_deferred_arm_replays",
            return_value=1,
        ) as launch,
    ):
        _run_isaac_deferred_collection_supervisor(
            exp_name="t08_hammer_nail",
            out_dir=tmp_path,
            video_fps=30,
        )

    assert popen.call_args.kwargs["env"]["HANDSON_ISAAC_DEFERRED_ARM_WORKER"] == "1"
    assert "CUDA_VISIBLE_DEVICES" not in popen.call_args.kwargs["env"]
    assert popen.call_args.kwargs["start_new_session"] is True
    launch.assert_called_once()
    assert launch.call_args.kwargs["exp_name"] == "t08_hammer_nail"
    assert launch.call_args.kwargs["video_fps"] == 30


def test_supervisor_queues_physical_replay_without_exit_autostart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    worker = _FinishedProcess()
    monkeypatch.setenv("HANDSON_DEFER_PHYSICAL_REPLAY_CERTIFICATION", "1")

    with (
        patch("scripts.record_demos_zarr.sys.argv", ["record_demos_zarr.py"]),
        patch(
            "scripts.record_demos_zarr.subprocess.Popen",
            return_value=worker,
        ),
        patch(
            "scripts.record_demos_zarr._launch_pending_isaac_deferred_arm_replays",
        ) as launch,
        patch(
            "scripts.record_demos_zarr._ensure_deferred_physical_replay_certification",
        ) as ensure,
    ):
        _run_isaac_deferred_collection_supervisor(
            exp_name="t08_hammer_nail",
            out_dir=tmp_path,
            video_fps=30,
        )

    launch.assert_not_called()
    ensure.assert_not_called()
    output = capsys.readouterr().out
    assert "arm replay discovery is deferred" in output
    assert "queued 0 background arm replay" not in output


def test_manual_only_supervisor_queues_without_startup_or_exit_certification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    worker = _FinishedProcess()
    monkeypatch.delenv(
        "HANDSON_DEFER_PHYSICAL_REPLAY_CERTIFICATION",
        raising=False,
    )
    monkeypatch.setenv("HANDSON_POSTPROCESS_MANUAL_ONLY", "1")

    with (
        patch("scripts.record_demos_zarr.sys.argv", ["record_demos_zarr.py"]),
        patch(
            "scripts.record_demos_zarr.subprocess.Popen",
            return_value=worker,
        ) as popen,
        patch(
            "scripts.record_demos_zarr._ensure_deferred_physical_replay_certification",
        ) as ensure,
        patch(
            "scripts.record_demos_zarr._launch_pending_isaac_deferred_arm_replays",
        ) as launch_arm,
    ):
        _run_isaac_deferred_collection_supervisor(
            exp_name="t08_hammer_nail",
            out_dir=tmp_path,
            video_fps=30,
        )

    ensure.assert_not_called()
    launch_arm.assert_not_called()
    assert (
        popen.call_args.kwargs["env"]["HANDSON_POSTPROCESS_MANUAL_ONLY"]
        == "1"
    )
    output = capsys.readouterr().out
    assert "postprocessing is manual-only" in output
    assert "queued until the operator starts" in output


def test_supervisor_skips_pending_recovery_after_inner_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _FinishedProcess()
    worker.returncode = 17
    monkeypatch.setenv("HANDSON_DEFER_PHYSICAL_REPLAY_CERTIFICATION", "1")

    with (
        patch("scripts.record_demos_zarr.sys.argv", ["record_demos_zarr.py"]),
        patch(
            "scripts.record_demos_zarr.subprocess.Popen",
            return_value=worker,
        ),
        patch(
            "scripts.record_demos_zarr._ensure_deferred_physical_replay_certification",
        ) as ensure,
        patch(
            "scripts.record_demos_zarr._launch_pending_isaac_deferred_arm_replays",
        ) as launch_arm,
        pytest.raises(RuntimeError, match="exit code 17"),
    ):
        _run_isaac_deferred_collection_supervisor(
            exp_name="t09_insert_peg",
            out_dir=tmp_path,
            video_fps=30,
        )

    ensure.assert_not_called()
    launch_arm.assert_not_called()


def test_supervisor_waits_for_worker_before_pending_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _FinishedProcess()
    monkeypatch.setenv("HANDSON_DEFER_PHYSICAL_REPLAY_CERTIFICATION", "1")
    monkeypatch.setenv("HANDSON_DEFER_PHYSICAL_REPLAY_AUTOSTART", "1")
    order: list[str] = []

    def wait_for_worker(*_args, **_kwargs) -> tuple[int, bool]:
        order.append("worker_reaped")
        return 0, True

    def recover_pending(*_args, **_kwargs) -> int:
        order.append("pending_recovery")
        return 0

    with (
        patch("scripts.record_demos_zarr.sys.argv", ["record_demos_zarr.py"]),
        patch(
            "scripts.record_demos_zarr.subprocess.Popen",
            return_value=worker,
        ),
        patch(
            "scripts.record_demos_zarr._wait_for_isaac_collection_worker",
            side_effect=wait_for_worker,
        ),
        patch(
            "scripts.record_demos_zarr._ensure_deferred_physical_replay_certification",
            side_effect=recover_pending,
        ),
    ):
        _run_isaac_deferred_collection_supervisor(
            exp_name="t09_insert_peg",
            out_dir=tmp_path,
            video_fps=30,
        )

    assert order == ["worker_reaped", "pending_recovery"]


def test_supervisor_stops_worker_when_isaac_scene_startup_hangs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    worker = _HangingProcess()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HANDSON_ISAAC_STARTUP_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("HANDSON_ISAAC_STARTUP_MAX_ATTEMPTS", "1")

    with (
        patch("scripts.record_demos_zarr.sys.argv", ["record_demos_zarr.py"]),
        patch(
            "scripts.record_demos_zarr.subprocess.Popen",
            return_value=worker,
        ) as popen,
        patch(
            "scripts.record_demos_zarr.time.monotonic",
            side_effect=[0.0, 2.0],
        ),
        patch(
            "scripts.record_demos_zarr._launch_pending_isaac_deferred_arm_replays",
        ) as launch,
        pytest.raises(RuntimeError, match="did not finish PhysX scene startup"),
    ):
        _run_isaac_deferred_collection_supervisor(
            exp_name="t24_place_ball_in_bowl",
            out_dir=tmp_path,
            video_fps=30,
        )

    assert worker.terminate_count == 1
    assert "CUDA_VISIBLE_DEVICES" not in popen.call_args.kwargs["env"]
    launch.assert_not_called()


def test_supervisor_does_not_retry_poisoned_cuda_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fatal = _FinishedProcess()
    fatal.returncode = 86
    monkeypatch.setenv("HANDSON_COLLECTION_SESSION_ID", "session-retry")

    with (
        patch("scripts.record_demos_zarr.sys.argv", ["record_demos_zarr.py"]),
        patch(
            "scripts.record_demos_zarr.subprocess.Popen",
            return_value=fatal,
        ) as popen,
        patch(
            "scripts.record_demos_zarr._launch_pending_isaac_deferred_arm_replays",
            return_value=0,
        ),
        pytest.raises(RuntimeError, match="exit code 86"),
    ):
        _run_isaac_deferred_collection_supervisor(
            exp_name="t24_place_ball_in_bowl",
            out_dir=tmp_path,
            video_fps=30,
        )

    assert popen.call_count == 1
    assert popen.call_args.kwargs["start_new_session"] is True
    assert "launching one clean process retry" not in capsys.readouterr().out


def test_runtime_heartbeat_timeout_force_stops_worker_group(
    tmp_path: Path,
) -> None:
    worker = _HangingProcess()
    ready = tmp_path / "ready.json"
    status = tmp_path / "status.json"
    ready.write_text("{}", encoding="utf-8")
    status.write_text('{"phase":"running"}', encoding="utf-8")

    os.utime(status, (1.0, 1.0))

    with (
        patch("scripts.record_demos_zarr.time.time", return_value=1000.0),
        pytest.raises(TimeoutError, match="heartbeat stopped"),
    ):
        _wait_for_isaac_collection_worker(
            worker,
            startup_marker=ready,
            status_path=status,
            startup_timeout=60.0,
            heartbeat_timeout=10.0,
        )

    assert worker.terminate_count == 1


def test_supervisor_replays_saved_demos_after_collection_stop(
    tmp_path: Path,
) -> None:
    worker = _InterruptedProcess()

    with (
        patch("scripts.record_demos_zarr.sys.argv", ["record_demos_zarr.py"]),
        patch(
            "scripts.record_demos_zarr.subprocess.Popen",
            return_value=worker,
        ),
        patch(
            "scripts.record_demos_zarr._launch_pending_isaac_deferred_arm_replays",
            return_value=1,
        ) as launch,
    ):
        _run_isaac_deferred_collection_supervisor(
            exp_name="t08_hammer_nail",
            out_dir=tmp_path,
            video_fps=30,
        )

    assert worker.wait_count == 2
    assert worker.signals == [signal.SIGINT]
    assert worker.wait_timeouts[-1] == 15.0
    launch.assert_called_once()
    assert launch.call_args.kwargs["exp_name"] == "t08_hammer_nail"
    assert launch.call_args.kwargs["video_fps"] == 30


def test_already_signaled_worker_drains_then_escalates_without_second_sigint(
) -> None:
    worker = _TermIgnoringProcess()

    returncode = _stop_isaac_deferred_collection_worker(
        worker,
        already_signaled=True,
    )

    assert returncode == -9
    assert worker.signals == []
    assert worker.wait_timeouts == [15.0, 5.0, 2.0]
    assert worker.terminate_count == 1
    assert worker.kill_count == 1


def test_deferred_arm_replays_launch_background_batch(tmp_path: Path) -> None:
    demo = tmp_path / "t08_hammer_nail_demo_1"
    (demo / "replay.zarr").mkdir(parents=True)
    (demo / "collection_meta.json").write_text(
        '{"collection_session_id": "session-1"}',
        encoding="utf-8",
    )
    worker = _FinishedProcess()

    with patch(
        "scripts.record_demos_zarr.subprocess.Popen",
        return_value=worker,
    ) as popen:
        count = _launch_pending_isaac_deferred_arm_replays(
            tmp_path,
            exp_name="t08_hammer_nail",
            session_id="session-1",
            existing=set(),
            video_fps=30,
        )

    assert count == 1
    command = popen.call_args.args[0]
    child_env = popen.call_args.kwargs["env"]
    status = json.loads(
        (demo / "deferred_arm_replay_status.json").read_text(encoding="utf-8")
    )
    assert command[1].endswith("isaac_deferred_arm_replay_worker.py")
    assert f"--exp_name=t08_hammer_nail" in command
    assert f"--video_fps=30" in command
    assert str(demo.resolve()) in command
    assert popen.call_args.kwargs["start_new_session"] is True
    assert child_env["HANDSON_ISAACLAB_DEFER_ARM_RENDER"] == "0"
    assert status["status"] == "queued"
    assert status["exp_name"] == "t08_hammer_nail"
