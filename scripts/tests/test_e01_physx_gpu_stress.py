from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import e01_physx_gpu_stress as stress


def _poses() -> dict[str, np.ndarray]:
    return {
        "right": np.asarray([-0.3, -0.2, 1.0, 1.0, 0.0, 0.0, 0.0]),
        "left": np.asarray([-0.3, 0.2, 1.0, 1.0, 0.0, 0.0, 0.0]),
    }


def _hands() -> dict[str, np.ndarray]:
    return {
        "right": np.zeros(20, dtype=np.float64),
        "left": np.zeros(20, dtype=np.float64),
    }


def test_e01_stress_action_keeps_public_54d_layout() -> None:
    poses = _poses()
    hands = _hands()
    hands["right"][:] = np.arange(20)
    hands["left"][:] = np.arange(20, 40)
    action = stress.make_action(
        poses["right"], poses["left"], hands["right"], hands["left"]
    )
    assert action.shape == (54,)
    np.testing.assert_array_equal(action[:7], poses["right"])
    np.testing.assert_array_equal(action[7:14], poses["left"])
    np.testing.assert_array_equal(action[14:34], np.arange(20))
    np.testing.assert_array_equal(action[34:54], np.arange(20, 40))


def test_e01_stress_motion_steps_preserve_authored_poke_speed() -> None:
    start = _poses()
    target = {side: value.copy() for side, value in start.items()}
    target["right"][2] -= 0.030
    hands = _hands()
    assert (
        stress.motion_step_count(
            start,
            target,
            hands,
            hands,
            linear_speed_m_s=0.12,
        )
        == 8
    )
    assert (
        stress.motion_step_count(
            start,
            target,
            hands,
            hands,
            linear_speed_m_s=0.05,
        )
        == 18
    )

    nonzero_start_hands = {
        side: np.full(20, 0.30, dtype=np.float64) for side in ("right", "left")
    }
    target_hands = {
        side: np.full(20, 0.50, dtype=np.float64) for side in ("right", "left")
    }
    assert (
        stress.motion_step_count(
            start,
            start,
            nonzero_start_hands,
            target_hands,
            linear_speed_m_s=0.12,
            hand_speed_rad_s=1.2,
        )
        == 5
    )


def test_e01_stress_drive_uses_environment_translation_speed_cap() -> None:
    poses = _poses()
    hands = _hands()
    fake_env = SimpleNamespace(
        _right_tcp_pose=poses["right"].copy(),
        _left_tcp_pose=poses["left"].copy(),
        _right_hand=hands["right"].copy(),
        _left_hand=hands["left"].copy(),
        _control_dt=stress.CONTROL_DT,
        _contact_max_tcp_translation_step=0.002,
    )
    runner = stress.E01StressRunner(fake_env, stress.Thresholds())
    actions: list[np.ndarray] = []
    runner.step = lambda action: actions.append(np.asarray(action).copy())

    target = {side: pose.copy() for side, pose in poses.items()}
    target["right"][2] -= 0.030
    runner.drive(target, linear_speed_m_s=0.12)

    # The requested 0.12 m/s is capped by the live 2 mm/control-frame Plan1
    # boundary to 0.06 m/s, so the complete 30 mm move requires 15 frames.
    assert len(actions) == 15
    np.testing.assert_allclose(actions[-1][:7], target["right"])
    commanded_positions = np.vstack(
        [poses["right"][:3], *[action[:3] for action in actions]]
    )
    commanded_steps = np.linalg.norm(np.diff(commanded_positions, axis=0), axis=1)
    assert float(np.max(commanded_steps)) <= 0.002 + 1e-12


def test_e01_stress_public_wxyz_matches_physx_link_readback() -> None:
    poses = _poses()
    fake_env = SimpleNamespace(_detached_sim_pose=poses)
    runner = stress.E01StressRunner(fake_env, stress.Thresholds())
    public = np.concatenate([poses["right"], poses["left"]])
    position_error, rotation_error = runner._validate_public_tcp_observation(
        {"state": {"tcp_pose": public}}
    )
    assert position_error == 0.0
    assert rotation_error == 0.0

    # Reinterpreting an unambiguous WXYZ rotation as XYZW must be rejected.
    physical = np.asarray([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
    poses["right"][3:7] = physical
    public[3:7] = physical[[1, 2, 3, 0]]
    with pytest.raises(stress.StressFailure, match="public WXYZ orientation"):
        runner._validate_public_tcp_observation({"state": {"tcp_pose": public}})


def test_e01_stress_gpu_process_parser_is_read_only_and_tolerant() -> None:
    rows = stress.parse_compute_process_rows(
        "896921, /home/user/micromamba/envs/handson/bin/python, 8123\n"
        "malformed\n"
        "17, python, [Not Supported]\n"
    )
    assert rows == [
        {
            "pid": 896921,
            "process_name": "/home/user/micromamba/envs/handson/bin/python",
            "used_memory_mib": 8123,
        },
        {"pid": 17, "process_name": "python", "used_memory_mib": None},
    ]
    pmon = stress.parse_pmon_process_rows(
        "# gpu pid type sm mem enc dec jpg ofa command\n"
        "0 2926 G - - - - - - Xorg\n"
        "0 896921 G 40 20 - - - - python\n"
    )
    assert pmon == [
        {"gpu_index": 0, "pid": 2926, "type": "G", "command": "Xorg"},
        {"gpu_index": 0, "pid": 896921, "type": "G", "command": "python"},
    ]


def test_e01_stress_preflight_refuses_existing_cuda_owner(monkeypatch) -> None:
    def fake_run(command, **_kwargs):
        if any("--query-gpu=" in argument for argument in command):
            return SimpleNamespace(stdout="0, NVIDIA RTX 4090, 24564, 590.1\n")
        return SimpleNamespace(stdout="896921, python, 8123\n")

    monkeypatch.setattr(stress.subprocess, "run", fake_run)
    with pytest.raises(stress.StressFailure, match="refusing to launch Kit"):
        stress.gpu_preflight("cuda:0", allow_busy_gpu=False)
    gpu = stress.gpu_preflight("cuda:0", allow_busy_gpu=True)
    assert gpu["preexisting_compute_processes"][0]["pid"] == 896921


def test_e01_stress_preflight_catches_graphics_only_python_collection(
    monkeypatch,
) -> None:
    def fake_run(command, **_kwargs):
        if any("--query-gpu=" in argument for argument in command):
            return SimpleNamespace(stdout="0, NVIDIA RTX 4090, 24564, 590.1\n")
        if "pmon" in command:
            return SimpleNamespace(
                stdout="0 2926 G - - - - - - Xorg\n"
                "0 896921 G 40 20 - - - - python\n"
            )
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(stress.subprocess, "run", fake_run)
    with pytest.raises(stress.StressFailure, match="graphics workload"):
        stress.gpu_preflight("cuda:0", allow_busy_gpu=False)


def test_e01_stress_plan_mode_never_imports_or_launches_kit(capsys) -> None:
    assert (
        stress.main(
            ["--print-plan", "--idle-seconds", "1", "--reset-cycles", "2"]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["launches_kit"] is False
    assert payload["control_hz"] == 30
    assert payload["physics_hz"] == stress.E01_PHYSICS_HZ
    assert payload["phases"] == [
        "idle_1s_simulated",
        "poke_0.03m_s",
        "poke_0.05m_s",
        "poke_0.06m_s",
        "pinch_lift_release",
        "dual_pull_release",
        "tracker_jump_nan",
        "reset_x2",
    ]


def test_e01_stress_defaults_are_fail_closed_and_have_realtime_headroom() -> None:
    thresholds = stress.Thresholds()
    metadata = stress.selected_garmentlab_preset_metadata()
    assert thresholds.min_simulation_node_count == metadata["sim_vertex_count"]
    assert thresholds.max_simulation_node_count == metadata["sim_vertex_count"]
    assert (
        thresholds.min_simulation_triangle_count
        == metadata["sim_triangle_count"]
    )
    assert (
        thresholds.max_simulation_triangle_count
        == metadata["sim_triangle_count"]
    )
    assert thresholds.expected_visual_vertex_count == metadata["visual_vertex_count"]
    assert (
        thresholds.expected_visual_triangle_count
        == metadata["visual_triangle_count"]
    )
    assert thresholds.max_nodal_speed_m_s == pytest.approx(
        stress.E01_CONTACT_OFFSET * stress.E01_PHYSICS_HZ + 0.06
    )
    assert thresholds.max_control_node_step_m == 0.015
    assert thresholds.max_idle_drift_m == 0.005
    assert thresholds.max_reset_drift_m == 0.005
    assert thresholds.max_public_readback_position_error_m == 1e-6
    assert thresholds.max_public_readback_rotation_error_deg == 0.01
    assert thresholds.max_p95_step_ms < stress.CONTROL_DT * 1000.0
    assert thresholds.max_p99_step_ms <= stress.CONTROL_DT * 1000.0 + 0.01
    assert thresholds.max_realtime_overrun_fraction == 0.01

    source = inspect.getsource(stress.main)
    assert source.index("gpu_preflight") < source.index(
        "from handson.sim.isaaclab.e01_detached_hand_env import"
    )
    assert "headless=True" in source
    assert "image_obs=False" in source
    assert "launch_isaac=True" in source
    assert "hz=0" in source

    reset_source = inspect.getsource(stress.E01StressRunner.reset)
    assert "garment_reset_settle_min_seconds" in reset_source
    assert "garment_reset_settle_stable_seconds" in reset_source
    assert "garment_reset_settle_stable_steps" in reset_source
    assert "garment_reset_settle_window_drift" in reset_source
    assert "stable_steps < stable_steps_required" in reset_source


def test_e01_stress_thresholds_follow_selected_garment_preset(monkeypatch) -> None:
    monkeypatch.setenv("HANDSON_E01_GARMENTLAB_PRESET", "2")

    thresholds = stress.Thresholds()

    assert thresholds.min_simulation_node_count == 4_772
    assert thresholds.max_simulation_node_count == 4_772
    assert thresholds.min_simulation_triangle_count == 9_398
    assert thresholds.max_simulation_triangle_count == 9_398
    assert thresholds.expected_visual_vertex_count == 4_772
    assert thresholds.expected_visual_triangle_count == 9_398
