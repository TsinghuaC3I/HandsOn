from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from scripts import e01_replay_ipc_cloth as replay
from scripts import e01_render_ipc_replay as ipc_render
from scripts import record_demos_zarr as recorder


def _make_episode(
    path: Path,
    frame_count: int = 3,
    *,
    include_actual_hand_state: bool = True,
) -> Path:
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    state = np.zeros((frame_count, 102), dtype=np.float64)
    next_state = np.zeros_like(state)
    right = np.asarray(
        [-0.37, -0.24, 1.00, 0.70710678, 0.0, 0.70710678, 0.0]
    )
    left = np.asarray(
        [-0.37, 0.24, 1.00, 0.70710678, 0.0, 0.70710678, 0.0]
    )
    state[:, :7] = right
    state[:, 7:14] = left
    next_state[:, :14] = state[:, :14]
    next_state[:, 0] += np.arange(frame_count) * 0.001
    next_state[:, 7] += np.arange(frame_count) * 0.001

    action = np.zeros((frame_count, 54), dtype=np.float64)
    action[:, :7] = next_state[:, :7]
    action[:, 7:27] = np.arange(20, dtype=np.float64) * 0.01
    action[:, 27:34] = next_state[:, 7:14]
    action[:, 34:54] = np.arange(20, dtype=np.float64) * -0.01
    data.create_dataset("state", data=state, shape=state.shape, dtype=state.dtype)
    data.create_dataset(
        "next_state", data=next_state, shape=next_state.shape, dtype=next_state.dtype
    )
    data.create_dataset("action", data=action, shape=action.shape, dtype=action.dtype)
    if include_actual_hand_state:
        # Deliberately differ from action commands so the replay test proves
        # it consumes the same-transition simulator readback.
        current_hand_qpos = np.tile(
            np.linspace(-0.3, 0.3, 40, dtype=np.float64),
            (frame_count, 1),
        )
        next_hand_qpos = current_hand_qpos + (
            np.arange(frame_count, dtype=np.float64)[:, None] + 1.0
        ) * 0.001
        data.create_dataset(
            "e01_hand_qpos",
            data=current_hand_qpos,
            shape=current_hand_qpos.shape,
            dtype=current_hand_qpos.dtype,
        )
        data.create_dataset(
            "next_e01_hand_qpos",
            data=next_hand_qpos,
            shape=next_hand_qpos.shape,
            dtype=next_hand_qpos.dtype,
        )
    timestamp = np.arange(frame_count, dtype=np.float64) / 30.0
    data.create_dataset(
        "timestamp", data=timestamp, shape=timestamp.shape, dtype=timestamp.dtype
    )
    return path


def _contact_names(side: str) -> tuple[str, ...]:
    return (f"{side}_palm_link",) + tuple(
        f"{side}_finger{finger}_link{link}"
        for finger in range(1, 6)
        for link in range(2, 5)
    )


class _FakeScene:
    def step(self) -> None:
        return None


class _FakeIPCEnv:
    def __init__(self, **_kwargs) -> None:
        self._asset_path = Path("/downloaded/GarmentLab/TCLC_Top578_genesis.obj")
        self._nodes = np.asarray(
            [
                [-0.2, -0.1, 0.90],
                [-0.1, -0.1, 0.90],
                [-0.2, 0.1, 0.90],
                [-0.1, 0.1, 0.90],
            ],
            dtype=np.float64,
        )
        self._keypoint_indices = np.arange(4, dtype=np.int64)
        self._scene = _FakeScene()
        self._robots = {
            side: SimpleNamespace(
                material=SimpleNamespace(coup_links=_contact_names(side))
            )
            for side in ("right", "left")
        }
        self._right_tcp_pose = np.zeros(7, dtype=np.float64)
        self._left_tcp_pose = np.zeros(7, dtype=np.float64)
        self._right_hand = np.zeros(20, dtype=np.float64)
        self._left_hand = np.zeros(20, dtype=np.float64)
        self._success_counter = 0
        self.env_step = 0
        self.hz = 30
        self.closed = False

    def reset(self):
        return {}, {}

    def step(self, action):
        self._right_tcp_pose = np.asarray(action["right"][:7], dtype=np.float64)
        self._left_tcp_pose = np.asarray(action["left"][:7], dtype=np.float64)
        self._right_hand = np.asarray(action["right"][7:27], dtype=np.float64)
        self._left_hand = np.asarray(action["left"][7:27], dtype=np.float64)
        self._nodes[:, 0] += 0.0001
        return {}, 0.0, False, False, {}

    def _refresh_state(self, **_kwargs) -> None:
        return None

    def _configure_tracking(self, _nodes) -> None:
        return None

    def _garment_metrics(self) -> dict:
        return {}

    def close(self) -> None:
        self.closed = True


def _run_fake(replay_path: Path, env_factory=_FakeIPCEnv):
    return replay._run_replay(
        replay_path,
        overwrite=True,
        settle_steps=2,
        physics_dt=1.0 / 60.0,
        newton_max_iterations=8,
        store_full_nodes=True,
        table_penetration_tolerance=0.004,
        max_node_step=0.25,
        max_tcp_position_error=0.003,
        max_tcp_orientation_error=0.03,
        env_factory=env_factory,
    )


def test_ipc_replay_uses_protected_tcp_and_action_aligned_actual_hands(
    tmp_path: Path,
) -> None:
    replay_path = _make_episode(tmp_path / "replay.zarr")

    result = _run_fake(replay_path)

    assert result["status"] == "complete"
    data = zarr.open(str(replay_path), mode="r")["data"]
    np.testing.assert_allclose(
        data["e01_ipc_replay_tcp_target"][:], data["next_state"][:, :14]
    )
    expected_hands = data["next_e01_hand_qpos"][:]
    np.testing.assert_allclose(data["e01_ipc_replay_hand_command"][:], expected_hands)
    np.testing.assert_allclose(data["e01_ipc_replay_hand_qpos"][:], expected_hands)
    assert data["e01_ipc_replay_cloth_nodes"].shape == (3, 4, 3)
    assert data["e01_ipc_replay_cloth_keypoints"].shape == (3, 4, 3)
    np.testing.assert_array_equal(data["e01_ipc_replay_valid_frame"][:], 1)
    assert data.attrs["e01_ipc_replay_newton_max_iterations"] == 8
    assert data.attrs["e01_ipc_replay_status"] == "complete"
    assert data.attrs["e01_ipc_replay_hand_source"] == "next_e01_hand_qpos"
    assert data.attrs["e01_ipc_replay_hand_source_kind"] == "actual"
    assert data.attrs["e01_ipc_replay_hand_source_is_actual"] is True

    sidecar = json.loads(
        (tmp_path / "e01_ipc_replay.json").read_text(encoding="utf-8")
    )
    assert sidecar["trajectory_sources"]["tcp"].startswith("data/next_state")
    assert "actual PhysX joint readback" in sidecar["trajectory_sources"]["hand"]
    assert sidecar["trajectory_sources"]["hand_kind"] == "actual"
    assert sidecar["solver"]["backend"] == "Genesis（旧 PBD 诊断 / IPC 离线）"
    assert sidecar["solver"]["full_hand_collision_required"] is True


def test_ipc_replay_legacy_episode_falls_back_to_command_and_marks_metadata(
    tmp_path: Path,
) -> None:
    replay_path = _make_episode(
        tmp_path / "replay.zarr",
        include_actual_hand_state=False,
    )

    _run_fake(replay_path)

    data = zarr.open(str(replay_path), mode="r")["data"]
    expected_hands = np.concatenate(
        [data["action"][:, 7:27], data["action"][:, 34:54]], axis=1
    )
    np.testing.assert_allclose(data["e01_ipc_replay_hand_command"][:], expected_hands)
    assert data.attrs["e01_ipc_replay_hand_source"] == "action"
    assert data.attrs["e01_ipc_replay_hand_source_kind"] == "command_fallback"
    assert data.attrs["e01_ipc_replay_hand_source_is_actual"] is False
    sidecar = json.loads(
        (tmp_path / "e01_ipc_replay.json").read_text(encoding="utf-8")
    )
    assert sidecar["trajectory_sources"]["hand_kind"] == "command_fallback"
    assert sidecar["trajectory_sources"]["hand_alignment"] == (
        "data/action[t] command (no joint readback available)"
    )
    assert "actual joint state unavailable" in sidecar["trajectory_sources"]["hand"]
    assert "command targets rather than actual" in sidecar[
        "trajectory_sources"
    ]["hand_state_limitation"]


def test_ipc_replay_rejects_malformed_actual_hand_state_instead_of_fallback(
    tmp_path: Path,
) -> None:
    replay_path = _make_episode(tmp_path / "replay.zarr")
    data = zarr.open(str(replay_path), mode="a")["data"]
    del data["next_e01_hand_qpos"]
    malformed = np.zeros((3, 39), dtype=np.float64)
    data.create_dataset(
        "next_e01_hand_qpos",
        data=malformed,
        shape=malformed.shape,
        dtype=malformed.dtype,
    )

    with pytest.raises(ValueError, match="action-aligned shape"):
        replay._load_protected_trajectory(data)


def test_ipc_replay_quality_failure_is_persisted_and_raised(tmp_path: Path) -> None:
    replay_path = _make_episode(tmp_path / "replay.zarr", frame_count=1)

    class PenetratingEnv(_FakeIPCEnv):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self._nodes[:, 2] = 0.80

    with pytest.raises(RuntimeError, match="quality checks failed"):
        _run_fake(replay_path, env_factory=PenetratingEnv)

    data = zarr.open(str(replay_path), mode="r")["data"]
    assert data.attrs["e01_ipc_replay_status"] == "failed"
    assert data.attrs["e01_ipc_replay_quality_valid"] is False
    np.testing.assert_array_equal(data["e01_ipc_replay_valid_frame"][:], 0)
    sidecar = json.loads(
        (tmp_path / "e01_ipc_replay.json").read_text(encoding="utf-8")
    )
    assert sidecar["status"] == "failed"
    assert "quality checks failed" in sidecar["failure_reason"]


def test_ipc_render_frame_draws_nonblank_height_views() -> None:
    nodes = np.asarray(
        [
            [-0.25, -0.08, 0.890],
            [-0.18, -0.08, 0.894],
            [-0.25, 0.08, 0.902],
            [-0.18, 0.08, 0.910],
        ],
        dtype=np.float32,
    )
    bounds = ipc_render._bounds(nodes[None, ...])

    frame = ipc_render._render_frame(
        nodes,
        frame_index=0,
        frame_count=1,
        bounds=bounds,
        valid=True,
        width=640,
        height=360,
    )

    assert frame.shape == (360, 640, 3)
    assert frame.dtype == np.uint8
    assert int(frame.max()) > 24


@pytest.mark.parametrize("sim_backend", ["genesis", "isaacsim"])
def test_recorder_ipc_matches_detached_hand_e01_backends(
    sim_backend: str,
) -> None:
    assert recorder._should_auto_replay_e01_ipc(
        "e01_fold_garment", sim_backend=sim_backend, env_value="1"
    )
    assert recorder._should_auto_replay_e01_ipc(
        "f01_cotton_cloth_mvp", sim_backend=sim_backend, env_value="1"
    )
    assert not recorder._should_auto_compute_e01_arm_plan(
        "f01_cotton_cloth_mvp", sim_backend=sim_backend, env_value="1"
    )
    assert not recorder._should_auto_replay_e01_ipc(
        "e01_fold_garment", sim_backend="mujoco", env_value="1"
    )
    assert not recorder._should_auto_replay_e01_ipc(
        "e02_tie_rope_knot", sim_backend=sim_backend, env_value="1"
    )
    assert not recorder._should_auto_replay_e01_ipc(
        "e01_fold_garment", sim_backend=sim_backend, env_value="off"
    )


@pytest.mark.parametrize("sim_backend", ["genesis", "isaacsim"])
def test_recorder_runs_ipc_before_arm_and_persists_subprocess_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sim_backend: str,
) -> None:
    demo_dir = tmp_path / "e01_demo"
    demo_dir.mkdir()
    (demo_dir / "collection_meta.json").write_text(
        json.dumps({"task_outcome": "success"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        recorder,
        "FLAGS",
        SimpleNamespace(
            c01_arm_max_fk_position_error=0.02,
            c01_arm_max_fk_orientation_error=0.20,
            c01_arm_collision_penetration_tolerance=0.002,
            c01_arm_collision_sweep_joint_step=0.02,
        ),
    )
    monkeypatch.setenv("HANDSON_GENESIS_PYTHON", "/fake/genesis-python")
    monkeypatch.setenv("HANDSON_E01_ARM_PLAN_PYTHON", "/fake/plan-python")
    monkeypatch.setenv("HANDSON_PYTHON", "/fake/live-backend-python")
    monkeypatch.setenv(
        "HANDSON_GENESIS_CUDA12_LIBRARY_PATH",
        os.pathsep.join(["/fake/cuda12/a", "/fake/cuda12/b"]),
    )
    monkeypatch.setenv("LD_LIBRARY_PATH", "/fake/live-backend/lib")
    commands: list[list[str]] = []
    subprocess_envs: list[dict[str, str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        subprocess_envs.append(dict(kwargs["env"]))
        return subprocess.CompletedProcess(
            command,
            7 if "e01_replay_ipc_cloth.py" in command[1] else 0,
            stdout="IPC failed geometrically\n" if not commands[:-1] else "IK valid\n",
        )

    monkeypatch.setattr(recorder.subprocess, "run", fake_run)

    result = recorder._run_e01_postprocess_pipeline(
        demo_dir,
        sim_backend=sim_backend,
        ipc_env_value="1",
        arm_env_value="1",
    )

    assert [Path(command[1]).name for command in commands] == [
        "e01_replay_ipc_cloth.py",
        "e01_compute_arm_plan_qpos.py",
    ]
    assert commands[0][0] == "/fake/genesis-python"
    assert commands[1][0] == "/fake/plan-python"
    assert commands[0][commands[0].index("--max-node-step") + 1] == "0.05"
    assert "--table-delta-from-state-101" in commands[1]
    assert subprocess_envs[0]["LD_LIBRARY_PATH"].split(os.pathsep) == [
        "/fake/cuda12/a",
        "/fake/cuda12/b",
        "/fake/live-backend/lib",
    ]
    assert subprocess_envs[1]["LD_LIBRARY_PATH"] == "/fake/live-backend/lib"
    assert result["status"] == "failed"
    assert result["failed_stages"] == ["ipc_replay"]
    assert result["stages"]["ipc_replay"]["returncode"] == 7
    assert result["stages"]["arm_ik"]["status"] == "complete"
    persisted = json.loads(
        (demo_dir / "e01_postprocess.json").read_text(encoding="utf-8")
    )
    assert persisted["status"] == "failed"
    collection_meta = json.loads(
        (demo_dir / "collection_meta.json").read_text(encoding="utf-8")
    )
    assert collection_meta["task_outcome"] == "success"
    assert collection_meta["e01_postprocess_status"] == "failed"


def test_f01_recorder_ipc_uses_high_quality_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_dir = tmp_path / "f01_demo"
    demo_dir.mkdir()
    (demo_dir / "collection_meta.json").write_text(
        json.dumps({"task_outcome": "success"}), encoding="utf-8"
    )
    for key in (
        "HANDSON_E01_IPC_REPLAY_SETTLE_STEPS",
        "HANDSON_E01_IPC_REPLAY_PHYSICS_DT",
        "HANDSON_E01_IPC_REPLAY_NEWTON_MAX_ITERATIONS",
        "HANDSON_E01_IPC_REPLAY_MAX_NODE_STEP",
        "HANDSON_E01_IPC_REPLAY_MAX_TCP_POSITION_ERROR",
        "HANDSON_E01_IPC_REPLAY_MAX_TCP_ORIENTATION_ERROR",
        "HANDSON_GENESIS_E01_YOUNGS_MODULUS",
        "HANDSON_GENESIS_E01_DENSITY",
        "HANDSON_GENESIS_E01_THICKNESS",
        "HANDSON_GENESIS_E01_BENDING_STIFFNESS",
        "HANDSON_GENESIS_E01_CLOTH_FRICTION",
        "HANDSON_GENESIS_IPC_NEWTON_TOLERANCE",
        "HANDSON_GENESIS_IPC_TRANSLATION_TOLERANCE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HANDSON_GENESIS_PYTHON", "/fake/genesis-python")
    commands: list[list[str]] = []
    subprocess_envs: list[dict[str, str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        subprocess_envs.append(dict(kwargs["env"]))
        return subprocess.CompletedProcess(command, 0, stdout="ok\n")

    monkeypatch.setattr(recorder.subprocess, "run", fake_run)

    result = recorder._run_e01_postprocess_pipeline(
        demo_dir,
        exp_name="f01_cotton_cloth_mvp",
        sim_backend="isaacsim",
        ipc_env_value="1",
        arm_env_value="1",
    )

    assert [Path(command[1]).name for command in commands] == [
        "e01_replay_ipc_cloth.py",
        "e01_render_ipc_replay.py",
    ]
    replay_command = commands[0]
    assert replay_command[0] == "/fake/genesis-python"
    assert replay_command[replay_command.index("--settle-steps") + 1] == "80"
    assert replay_command[replay_command.index("--physics-dt") + 1] == str(
        1.0 / 120.0
    )
    assert replay_command[
        replay_command.index("--newton-max-iterations") + 1
    ] == "16"
    assert replay_command[replay_command.index("--max-node-step") + 1] == "0.090"
    assert (
        replay_command[replay_command.index("--max-tcp-position-error") + 1]
        == "0.300"
    )
    assert (
        replay_command[replay_command.index("--max-tcp-orientation-error") + 1]
        == "3.200"
    )
    ipc_env = subprocess_envs[0]
    assert ipc_env["HANDSON_GENESIS_E01_THICKNESS"] == "0.00020"
    assert ipc_env["HANDSON_GENESIS_E01_DENSITY"] == "375"
    assert ipc_env["HANDSON_GENESIS_E01_YOUNGS_MODULUS"] == "1.2e4"
    assert ipc_env["HANDSON_GENESIS_E01_BENDING_STIFFNESS"] == "12"
    assert ipc_env["HANDSON_GENESIS_E01_CLOTH_FRICTION"] == "0.9"
    assert ipc_env["HANDSON_GENESIS_IPC_NEWTON_TOLERANCE"] == "0.01"
    assert ipc_env["HANDSON_GENESIS_IPC_TRANSLATION_TOLERANCE"] == "0.03"
    assert result["status"] == "complete"
    assert result["quality_profile"] == "f01_high_quality_ipc"


def test_f01_ipc_postprocess_launches_background_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_dir = tmp_path / "f01_cotton_cloth_mvp_demo_1"
    (demo_dir / "replay.zarr").mkdir(parents=True)
    (demo_dir / "collection_meta.json").write_text(
        json.dumps({"collection_session_id": "session-1"}),
        encoding="utf-8",
    )

    class _StartedProcess:
        pid = 9511

    monkeypatch.setenv("HANDSON_COLLECTION_SESSION_ID", "session-1")
    with pytest.MonkeyPatch.context() as context:
        context.setattr(
            recorder.subprocess,
            "Popen",
            lambda *args, **kwargs: _StartedProcess(),
        )
        count = recorder._launch_e01_postprocess_batch(
            [demo_dir],
            exp_name="f01_cotton_cloth_mvp",
            sim_backend="isaacsim",
            ipc_env_value="1",
            arm_env_value="1",
            video_fps=30,
        )

    assert count == 1
    status = json.loads(
        (demo_dir / "e01_postprocess.json").read_text(encoding="utf-8")
    )
    collection_meta = json.loads(
        (demo_dir / "collection_meta.json").read_text(encoding="utf-8")
    )
    command = status["command"]
    assert status["status"] == "queued"
    assert status["background"] is True
    assert status["requested_stages"] == ["ipc_replay", "ipc_render"]
    assert command[1].endswith("e01_ipc_postprocess_worker.py")
    assert "--exp_name=f01_cotton_cloth_mvp" in command
    assert "--sim_backend=isaacsim" in command
    assert str(demo_dir.resolve()) in command
    assert collection_meta["e01_postprocess_status"] == "queued"
