from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
import os
import sys
from typing import Any

import numpy as np
import pytest

from scripts.hs014_auto_collection import (
    DEFAULT_MAX_STEPS,
    HS014AutoController,
    LEFT_FIXTURE_APPROACH_OFFSET_M,
    LEFT_FIXTURE_GRASP_WORLD_QUAT_WXYZ,
    LEFT_FIXTURE_GRIP,
    LEFT_FIXTURE_LATCH_START_ALPHA,
    LEFT_FIXTURE_PALM_OFFSET_M,
    LEFT_FIXTURE_PREGRIP,
    LEFT_GRASP_MAX_FRAMES,
    LEFT_GRASP_MOTION_FRAMES,
    OBJECT_TRANSLATION_FEEDBACK_MAX_M,
    PRELIFT_CLEARANCE_M,
    REQUIRED_ZARR_FIELDS,
    RIGHT_CAP_POWER_GRIP,
    RIGHT_TOP_DOWN_PALM_HEIGHT_M,
    RIGHT_TOP_DOWN_WORLD_QUAT_WXYZ,
    TWIST_DIRECTION,
    TWIST_PHASE_FRAMES,
    _camera_visibility_errors,
    _close_environment_fast,
    _interpolate_pose,
    _quat_to_matrix_wxyz,
    _require_hs014_physical_recording_origin,
    _reset_and_stabilize_physical_origin,
    _right_cap_grip,
    _write_zarr,
    episode_contract_errors,
)


def test_collect_and_replay_reset_use_production_stable_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from handson.sim.isaaclab import recording_origin

    calls: list[tuple[str, object]] = []
    monkeypatch.delenv("HANDSON_STABLE_RECORDING_ORIGIN", raising=False)

    class FakeEnvironment:
        def reset(self, *, seed: int):
            calls.append(("reset", seed))
            return {"raw": True}, {"succeed": False}

    monkeypatch.setattr(
        recording_origin,
        "stable_recording_origin_kwargs_for_task",
        lambda task_name: calls.append(("profile", task_name)) or {"stable_seconds": 10.0},
    )

    def fake_stabilize(env, observation, info, **kwargs):
        assert os.environ["HANDSON_STABLE_RECORDING_ORIGIN"] == "1"
        calls.append(("stabilize", kwargs["stable_seconds"]))
        return {
            "stable": True
        }, {
            **info,
            "physical_recording_origin": {
                "task_name": "bench320_hs014_bottle_stabilized_cap_opening",
                "stable_steps_required": 1200,
                "stable_steps_observed": 1200,
                "task_recording_origin_acceptance": {"accepted": True},
            },
        }

    monkeypatch.setattr(recording_origin, "stabilize_recording_origin", fake_stabilize)

    observation, info = _reset_and_stabilize_physical_origin(
        FakeEnvironment(),
        seed=23,
    )

    assert calls == [
        ("reset", 23),
        ("profile", "bench320_hs014_bottle_stabilized_cap_opening"),
        ("stabilize", 10.0),
    ]
    assert observation == {"stable": True}
    assert (
        info["physical_recording_origin"]
        ["task_recording_origin_acceptance"]["accepted"]
        is True
    )


@pytest.mark.parametrize(
    ("info", "message"),
    [
        ({}, "missing physical_recording_origin"),
        (
            {
                "physical_recording_origin": {
                    "task_name": "bench320_hs014_bottle_stabilized_cap_opening",
                    "stable_steps_required": 1200,
                    "stable_steps_observed": 1199,
                    "task_recording_origin_acceptance": {"accepted": True},
                }
            },
            "complete 1200-frame",
        ),
        (
            {
                "physical_recording_origin": {
                    "task_name": "bench320_hs014_bottle_stabilized_cap_opening",
                    "stable_steps_required": 1200,
                    "stable_steps_observed": 1200,
                }
            },
            "lacks exact task acceptance",
        ),
    ],
)
def test_hs014_production_origin_report_fails_closed(
    info: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _require_hs014_physical_recording_origin(
            info,
            profile={"stable_seconds": 10.0},
        )


def test_branch_handson_package_precedes_installed_packages() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    package_root = (repository_root / "handson").resolve()
    resolved = [Path(entry).resolve() for entry in sys.path if entry]
    assert package_root in resolved
    assert resolved.index(package_root) < min(
        (
            index
            for index, entry in enumerate(resolved)
            if "site-packages" in entry.parts and (entry / "handson").exists()
        ),
        default=len(resolved),
    )


def test_live_validator_enables_and_requires_the_production_origin_gate() -> None:
    validator = (
        Path(__file__).resolve().parents[1]
        / "validate_hs014_priority_zero_live.py"
    )
    source = validator.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'os.environ[STABLE_RECORDING_ORIGIN_ENV] = "1"' in source
    assert 'os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"' in source
    assert "_require_hs014_physical_recording_origin(" in source
    assert "stable_steps_observed" in source
    assert "task_recording_origin_acceptance" in source
    assert "priority_zero_runtime_contract(" in source
    assert '"verified_isaac51_isaaclab232_cuda_physx_runtime"' in source
    assert '"post_reset_cap_mass_properties_readback"' in source
    assert '"composed_cap_thin_wall_mass_properties"' in source
    assert '"diagonal_inertia_kg_m2"' in source
    assert '"post_reset_torque_neutral_cradle_and_material_readback"' in source
    assert '"three_sphere_cradle_exact_contact_readback"' in source
    assert '"contact_truth_topology"' in source
    assert '"three_sphere_cradle_to_com_sphere"' in source


def test_live_validator_false_result_always_exits_nonzero() -> None:
    validator = (
        Path(__file__).resolve().parents[1]
        / "validate_hs014_priority_zero_live.py"
    )
    source = validator.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_exit_code_for_result"
    )
    isolated = ast.fix_missing_locations(
        ast.Module(body=[helper], type_ignores=[])
    )
    namespace: dict[str, Any] = {"Any": Any, "Mapping": Mapping}
    exec(compile(isolated, str(validator), "exec"), namespace)
    exit_code = namespace["_exit_code_for_result"]

    assert exit_code({"passed": True}) == 0
    assert exit_code({"passed": False}) != 0
    assert exit_code({}) != 0
    assert exit_code({"passed": 1}) != 0
    assert source.count("rc = _exit_code_for_result(result)") >= 2
    assert "raise SystemExit(main())" in source


class _FakeRawEnv:
    def __init__(self) -> None:
        self._right_tcp_pose = np.asarray(
            [-0.10, -0.16, 1.07, 0.0, 1.0, 0.0, 0.0], dtype=np.float64
        )
        self._left_tcp_pose = np.asarray(
            [-0.10, 0.16, 1.07, 0.0, 1.0, 0.0, 0.0], dtype=np.float64
        )
        self._poses = {
            "hs014_water_bottle": np.asarray(
                [-0.24, 0.00, 1.0902, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
            ),
            "hs014_bottle_cap": np.asarray(
                [-0.24, 0.00, 1.0912, np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0],
                dtype=np.float64,
            ),
        }
        self._hs014_events = {
            "left_fixture_contact": False,
            "right_cap_contact": False,
            "continuous_twist_seen": False,
            "cap_axially_separated": False,
        }

    def _native_end_effector_pose_matrix(self, side: str, _fallback: np.ndarray):
        pose = self._left_tcp_pose if side == "left" else self._right_tcp_pose
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = pose[:3]
        from scripts.hs014_auto_collection import _quat_to_matrix_wxyz

        matrix[:3, :3] = _quat_to_matrix_wxyz(pose[3:7])
        return matrix

    def _physics_body_pose(self, name: str):
        return self._poses[name]

    def hs014_palm_pose_for_object_pose(self, _key: str, pose: np.ndarray):
        result = np.asarray(pose, dtype=np.float64).copy()
        result[2] += 0.05
        return result


def _frame() -> np.ndarray:
    grid = np.indices((24, 32)).sum(axis=0) % 2
    return np.repeat((grid[..., None] * 220 + 20).astype(np.uint8), 3, axis=2)


def _transition(index: int, *, succeed: bool) -> dict:
    hidden = np.full(16, index, dtype=np.float64)
    truth = np.zeros(8, dtype=np.float64)
    if succeed:
        truth.fill(1.0)
    passthrough = {
        "hs014_hidden_state": hidden,
        "hs014_success_truth": truth,
        "physx_robot_joint_position": np.full((4, 2, 33), index, dtype=np.float64),
        "physx_robot_joint_target": np.full((4, 2, 33), index + 0.1, dtype=np.float64),
        "physx_task_rigid_body_state": np.full((4, 4, 13), index, dtype=np.float64),
        "physx_contact_signature_pair_metrics": np.full((4, 8, 5), index, dtype=np.float64),
    }
    next_passthrough = {
        key: value + 1 for key, value in passthrough.items()
    }
    return {
        "action": np.zeros(54, dtype=np.float64),
        "observation": {"state": np.full(96, index, dtype=np.float64)},
        "next_observation": {"state": np.full(96, index + 1, dtype=np.float64)},
        "reward": float(succeed),
        "terminal": bool(succeed),
        "succeed": bool(succeed),
        "cameras": {
            "rgb_front": _frame(),
            "rgb_wrist_left": _frame(),
            "rgb_wrist_right": _frame(),
        },
        "passthrough": passthrough,
        "next_passthrough": next_passthrough,
    }


def test_controller_emits_only_finite_native_wuji_actions() -> None:
    raw = _FakeRawEnv()
    controller = HS014AutoController(raw)
    for _ in range(11):
        action = controller.action()
        assert action.shape == (54,)
        assert np.all(np.isfinite(action))
    assert controller.phase == "left_preorient"


def test_left_hand_preorients_without_translating_toward_task() -> None:
    raw = _FakeRawEnv()
    controller = HS014AutoController(raw)
    controller.phase_index = controller.PHASES.index("left_preorient")
    controller.phase_frame = 0
    controller.phase_start_left = controller._current_physical_tcp_pose("left")
    controller._configure_goals()
    np.testing.assert_allclose(
        controller.left_goal[:3], controller.phase_start_left[:3], atol=1.0e-12
    )
    np.testing.assert_allclose(
        controller.left_goal[3:7], [0.0, 1.0, 0.0, 0.0], atol=1.0e-12
    )


def test_left_fixture_approach_passes_through_measured_contact_geometry() -> None:
    raw = _FakeRawEnv()
    controller = HS014AutoController(raw)
    bottle = raw._poses["hs014_water_bottle"]

    controller.phase_index = controller.PHASES.index("left_approach")
    controller._configure_goals()
    np.testing.assert_allclose(
        controller.left_goal[:3],
        bottle[:3] + LEFT_FIXTURE_APPROACH_OFFSET_M,
        atol=1.0e-12,
    )

    controller.phase_index = controller.PHASES.index("left_grasp")
    controller._configure_goals()
    np.testing.assert_allclose(
        controller.left_goal[:3],
        bottle[:3] + LEFT_FIXTURE_PALM_OFFSET_M,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        controller.left_goal[3:7],
        LEFT_FIXTURE_GRASP_WORLD_QUAT_WXYZ,
        atol=1.0e-12,
    )
    assert np.linalg.norm(LEFT_FIXTURE_GRASP_WORLD_QUAT_WXYZ) == pytest.approx(1.0)


def test_left_fixture_closure_dwells_at_the_contact_goal() -> None:
    raw = _FakeRawEnv()
    controller = HS014AutoController(raw)
    controller.phase_index = controller.PHASES.index("left_grasp")
    controller.phase_start_left = controller._current_physical_tcp_pose("left")
    controller._configure_goals()
    controller.phase_frame = LEFT_GRASP_MOTION_FRAMES

    action = controller.action()

    assert controller.DURATIONS["left_grasp"] == LEFT_GRASP_MAX_FRAMES
    assert LEFT_GRASP_MAX_FRAMES > LEFT_GRASP_MOTION_FRAMES
    np.testing.assert_allclose(action[7:14], controller.left_goal, atol=1.0e-12)
    np.testing.assert_allclose(action[34:54], LEFT_FIXTURE_GRIP, atol=1.0e-12)


def test_left_fixture_pregrip_delays_role_latch_until_stable_path_segment() -> None:
    raw = _FakeRawEnv()
    controller = HS014AutoController(raw)
    controller.phase_index = controller.PHASES.index("left_grasp")
    controller.phase_start_left = controller._current_physical_tcp_pose("left")
    controller._configure_goals()
    controller.phase_frame = int(
        LEFT_GRASP_MOTION_FRAMES * LEFT_FIXTURE_LATCH_START_ALPHA
    ) - 1

    pregrip = controller.action()
    np.testing.assert_allclose(pregrip[34:54], LEFT_FIXTURE_PREGRIP, atol=1.0e-12)
    assert LEFT_FIXTURE_PREGRIP < 0.45

    controller.phase_frame = int(
        LEFT_GRASP_MOTION_FRAMES * LEFT_FIXTURE_LATCH_START_ALPHA
    )
    closure = controller.action()
    np.testing.assert_allclose(closure[34:54], LEFT_FIXTURE_GRIP, atol=1.0e-12)


def test_right_hand_preorients_then_uses_reachable_top_down_approach() -> None:
    raw = _FakeRawEnv()
    controller = HS014AutoController(raw)
    controller.phase_index = controller.PHASES.index("right_raise")
    controller.phase_frame = 0
    controller.phase_start_right = controller._current_physical_tcp_pose("right")
    controller._configure_goals()
    np.testing.assert_allclose(
        controller.right_goal[:2], controller.phase_start_right[:2], atol=1.0e-12
    )
    assert controller.right_goal[2] > raw._poses["hs014_bottle_cap"][2] + 0.10
    top_down = controller._planned_top_down_palm_pose(
        raw._poses["hs014_bottle_cap"]
    )
    np.testing.assert_allclose(controller.right_goal[3:7], top_down[3:7], atol=1.0e-12)
    np.testing.assert_allclose(top_down[:2], raw._poses["hs014_bottle_cap"][:2])
    assert top_down[2] == pytest.approx(
        raw._poses["hs014_bottle_cap"][2] + RIGHT_TOP_DOWN_PALM_HEIGHT_M
    )
    np.testing.assert_allclose(top_down[3:7], RIGHT_TOP_DOWN_WORLD_QUAT_WXYZ)
    assert np.linalg.norm(RIGHT_TOP_DOWN_WORLD_QUAT_WXYZ) == pytest.approx(1.0)

    controller.phase_index = controller.PHASES.index("right_approach")
    controller.phase_frame = 0
    controller.phase_start_right = controller._current_physical_tcp_pose("right")
    controller._configure_goals()
    expected = top_down.copy()
    expected[2] += 0.105
    np.testing.assert_allclose(controller.right_goal, expected, atol=1.0e-12)


def test_top_down_palm_azimuth_is_invariant_to_round_cap_yaw() -> None:
    first = np.asarray([-0.42, -0.12, 1.10, 1.0, 0.0, 0.0, 0.0])
    second = np.asarray([-0.42, -0.12, 1.10, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        HS014AutoController._planned_top_down_palm_pose(first),
        HS014AutoController._planned_top_down_palm_pose(second),
    )


def test_right_close_in_tracks_live_cap_translation() -> None:
    raw = _FakeRawEnv()
    controller = HS014AutoController(raw)
    controller.phase_index = controller.PHASES.index("right_close_in")
    controller.phase_frame = 0
    controller.phase_start_right = controller._current_physical_tcp_pose("right")
    controller._configure_goals()
    stale_goal = controller.right_goal.copy()
    raw._poses["hs014_bottle_cap"][:3] += [0.030, -0.020, 0.010]
    live_goal = controller._planned_top_down_palm_pose(
        raw._poses["hs014_bottle_cap"]
    )
    expected = _interpolate_pose(
        controller.phase_start_right,
        live_goal,
        1.0 / controller.DURATIONS["right_close_in"],
    )
    action = controller.action()
    np.testing.assert_allclose(action[:7], expected, atol=1.0e-12)
    stale = _interpolate_pose(
        controller.phase_start_right,
        stale_goal,
        1.0 / controller.DURATIONS["right_close_in"],
    )
    assert not np.allclose(action[:3], stale[:3])


def test_fast_close_uses_existing_runtime_shutdown_contract() -> None:
    class _App:
        call = None

        def close(self, **kwargs):
            self.call = kwargs

    class _Raw:
        def __init__(self):
            self._simulation_app = _App()
            self._fast_shutdown_on_close = False

    class _Env:
        def __init__(self):
            self.unwrapped = _Raw()
            self.closed = False

        def close(self):
            self.closed = True

    env = _Env()
    _close_environment_fast(env)
    assert env.closed is True
    assert env.unwrapped._fast_shutdown_on_close is True
    assert env.unwrapped._simulation_app.call == {
        "wait_for_replicator": False,
        "skip_cleanup": True,
    }


def test_settle_holds_physical_tcp_instead_of_stale_command_pose() -> None:
    raw = _FakeRawEnv()
    physical_right = np.asarray([-0.37, -0.24, 0.998, 1.0, 0.0, 0.0, 0.0])
    physical_left = np.asarray([-0.37, 0.24, 0.998, 1.0, 0.0, 0.0, 0.0])

    def physical_pose(side: str, _fallback: np.ndarray):
        pose = physical_left if side == "left" else physical_right
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = pose[:3]
        matrix[:3, :3] = np.eye(3)
        return matrix

    raw._native_end_effector_pose_matrix = physical_pose
    action = HS014AutoController(raw).action()
    np.testing.assert_allclose(action[:7], physical_right, atol=1.0e-12)
    np.testing.assert_allclose(action[7:14], physical_left, atol=1.0e-12)
    np.testing.assert_array_equal(action[14:], np.zeros(40, dtype=np.float64))


def test_camera_contract_requires_task_targets_in_all_three_frames() -> None:
    visible = {
        "hs014_camera_visibility": {
            name: {"required_target_count": 1, "targets_in_frame": True}
            for name in ("front", "wrist_left", "wrist_right")
        }
    }
    assert _camera_visibility_errors(visible) == []
    visible["hs014_camera_visibility"]["wrist_left"]["targets_in_frame"] = False
    assert _camera_visibility_errors(visible) == [
        "wrist_left required task target is outside the frame"
    ]
    assert _camera_visibility_errors({}) == [
        "HS014 camera target-visibility truth is missing"
    ]


def test_controller_event_gates_roles_before_cap_twist() -> None:
    raw = _FakeRawEnv()
    controller = HS014AutoController(raw)
    controller.phase_index = controller.PHASES.index("left_grasp")
    controller.phase_frame = 1
    raw._hs014_events["left_fixture_contact"] = True
    controller.action()
    assert controller.phase == "right_raise"
    controller.phase_index = controller.PHASES.index("right_close_in")
    controller.phase_frame = 1
    raw._hs014_events["right_cap_contact"] = True
    controller.action()
    assert controller.phase == "right_grasp"
    controller.phase_index = controller.PHASES.index("right_grasp")
    controller.phase_frame = 1
    raw._hs014_events["right_cap_contact"] = True
    controller.action()
    assert controller.phase == "prelift"
    assert controller.right_goal[2] == pytest.approx(
        raw._poses["hs014_bottle_cap"][2] + PRELIFT_CLEARANCE_M + 0.05
    )
    controller.phase_frame = controller.DURATIONS["prelift"]
    controller.action()
    assert controller.phase == "twist_1"


def test_twist_schedule_is_slow_enough_for_physical_arm_tracking() -> None:
    controller = HS014AutoController(_FakeRawEnv())
    assert all(
        controller.DURATIONS[f"twist_{index}"] == TWIST_PHASE_FRAMES
        for index in range(1, 8)
    )
    assert DEFAULT_MAX_STEPS > sum(controller.DURATIONS.values())


def test_twist_targets_remain_anchored_to_bottle_instead_of_drifted_cap() -> None:
    raw = _FakeRawEnv()
    controller = HS014AutoController(raw)
    controller._capture_twist_anchor()
    anchor_position = raw._poses["hs014_bottle_cap"][:3].copy()

    raw._poses["hs014_bottle_cap"][:3] += [0.20, -0.10, 0.08]
    goal = controller._twisted_cap_pose(60.0)
    np.testing.assert_allclose(goal[:3], anchor_position, atol=1.0e-12)

    raw._poses["hs014_water_bottle"][:3] += [0.01, -0.02, 0.0]
    translated_goal = controller._twisted_cap_pose(90.0)
    np.testing.assert_allclose(
        translated_goal[:3], anchor_position + [0.01, -0.02, 0.0], atol=1.0e-12
    )


def test_twist_phase_transition_starts_from_previous_commanded_goal() -> None:
    raw = _FakeRawEnv()
    controller = HS014AutoController(raw)
    controller._capture_twist_anchor()
    controller.phase_index = controller.PHASES.index("twist_1")
    controller.phase_start_right = controller._current_physical_tcp_pose("right")
    controller._configure_goals()
    previous_commanded_goal = controller.right_goal.copy()

    raw._right_tcp_pose = previous_commanded_goal.copy()
    raw._right_tcp_pose[:3] += [0.12, -0.08, 0.04]
    controller._advance()

    assert controller.phase == "twist_2"
    np.testing.assert_allclose(
        controller.phase_start_right, previous_commanded_goal, atol=1.0e-12
    )


def test_twist_action_tracks_live_bottle_translation_each_control_frame() -> None:
    raw = _FakeRawEnv()
    controller = HS014AutoController(raw)
    controller._capture_twist_anchor()
    controller.phase_index = controller.PHASES.index("twist_3")
    controller.phase_frame = 49
    controller._configure_goals()
    first = controller.action()[:7]

    translation = np.asarray([0.012, -0.009, 0.004], dtype=np.float64)
    raw._poses["hs014_water_bottle"][:3] += translation
    raw._poses["hs014_bottle_cap"][:3] += translation
    second = controller.action()[:7]

    np.testing.assert_allclose(second[:3] - first[:3], translation, atol=1.0e-12)


def test_twist_action_does_not_ratchet_from_live_cap_translation_error() -> None:
    raw = _FakeRawEnv()
    controller = HS014AutoController(raw)
    controller._capture_twist_anchor()
    controller.phase_index = controller.PHASES.index("twist_3")
    controller.phase_frame = 49
    controller._configure_goals()
    raw._poses["hs014_bottle_cap"][:3] += [0.050, -0.030, 0.040]

    alpha = 50.0 / controller.DURATIONS["twist_3"]
    cap_goal = controller._twisted_cap_pose(30.0 * (2.0 + alpha))
    expected = raw.hs014_palm_pose_for_object_pose("right_cap", cap_goal)
    action = controller.action()

    np.testing.assert_allclose(action[:7], expected, atol=1.0e-12)


def test_bimanual_hold_keeps_the_left_fixture_goal_absolute() -> None:
    raw = _FakeRawEnv()
    raw._hs014_events.update(
        left_fixture_contact=True,
        right_cap_contact=True,
    )
    controller = HS014AutoController(raw)
    controller.phase_index = controller.PHASES.index("twist_3")
    controller.phase_start_left = controller.left_hold.copy()
    controller.phase_start_left[:3] += [0.08, -0.05, 0.03]
    controller._capture_twist_anchor()
    controller._configure_goals()

    action = controller.action()

    expected = raw.hs014_palm_pose_for_object_pose(
        "left_fixture", controller.fixture_reference_pose
    )
    np.testing.assert_allclose(action[7:14], expected, atol=1.0e-12)


def test_object_translation_feedback_is_bounded() -> None:
    correction = HS014AutoController._bounded_translation_feedback(
        np.asarray([0.10, -0.05, 0.03]), np.zeros(3)
    )
    assert np.linalg.norm(correction) == pytest.approx(
        OBJECT_TRANSLATION_FEEDBACK_MAX_M
    )
    small = HS014AutoController._bounded_translation_feedback(
        np.asarray([0.003, -0.002, 0.001]), np.zeros(3)
    )
    np.testing.assert_allclose(small, [0.003, -0.002, 0.001], atol=1.0e-12)


def test_controller_uses_single_opening_twist_direction() -> None:
    assert TWIST_DIRECTION == 1.0
    raw = _FakeRawEnv()
    controller = HS014AutoController(raw)
    controller._capture_twist_anchor()
    goals = []
    for index in range(1, 8):
        controller.phase_index = controller.PHASES.index(f"twist_{index}")
        controller.phase_start_right = controller._current_physical_tcp_pose("right")
        controller._configure_goals()
        goal = controller.right_goal.copy()
        expected = controller._twisted_cap_pose(30.0 * index)
        expected[2] += 0.05
        np.testing.assert_allclose(goal[:3], expected[:3], atol=1.0e-12)
        np.testing.assert_allclose(
            _quat_to_matrix_wxyz(goal[3:7]),
            _quat_to_matrix_wxyz(expected[3:7]),
            atol=1.0e-12,
        )
        goals.append(goal)
    assert all(np.all(np.isfinite(goal)) for goal in goals)
    assert not any(np.allclose(first, second) for first, second in zip(goals, goals[1:]))


def test_right_approach_closes_at_safe_height_before_descent() -> None:
    raw = _FakeRawEnv()
    controller = HS014AutoController(raw)
    controller.phase_index = controller.PHASES.index("right_approach")
    controller.phase_start_right = controller._current_physical_tcp_pose("right")
    controller._configure_goals()
    action = controller.action()
    np.testing.assert_allclose(action[14:34], _right_cap_grip(0.40), atol=1.0e-12)
    assert np.max(np.abs(action[14:34])) < 0.45
    assert np.all(action[14:34].reshape(5, 4)[:, 1] <= 0.05)
    controller.phase_frame = int(0.90 * controller.DURATIONS["right_approach"])
    action = controller.action()
    np.testing.assert_allclose(action[14:34], _right_cap_grip(0.85), atol=1.0e-12)
    assert np.max(np.abs(action[14:34])) > 0.45
    np.testing.assert_allclose(
        _right_cap_grip(0.92), RIGHT_CAP_POWER_GRIP, atol=1.0e-12
    )


def test_right_descend_keeps_latch_ready_grip() -> None:
    raw = _FakeRawEnv()
    controller = HS014AutoController(raw)
    controller.phase_index = controller.PHASES.index("right_descend")
    controller.phase_start_right = controller._current_physical_tcp_pose("right")
    controller._configure_goals()
    controller.phase_frame = int(0.50 * controller.DURATIONS["right_descend"])
    action = controller.action()
    np.testing.assert_allclose(action[14:34], _right_cap_grip(0.85), atol=1.0e-12)
    controller.phase_frame = int(0.75 * controller.DURATIONS["right_descend"])
    action = controller.action()
    np.testing.assert_allclose(action[14:34], _right_cap_grip(0.85), atol=1.0e-12)
    controller.phase_index = controller.PHASES.index("right_close_in")
    controller.phase_frame = 0
    controller.phase_start_right = controller._current_physical_tcp_pose("right")
    controller._configure_goals()
    np.testing.assert_allclose(
        controller.right_goal,
        controller._planned_top_down_palm_pose(raw._poses["hs014_bottle_cap"]),
        atol=1.0e-12,
    )
    action = controller.action()
    np.testing.assert_allclose(action[14:34], _right_cap_grip(0.85), atol=1.0e-12)


def test_controller_requires_continuous_twist_before_lift() -> None:
    raw = _FakeRawEnv()
    controller = HS014AutoController(raw)
    controller.phase_index = controller.PHASES.index("twist_7")
    controller.phase_frame = controller.DURATIONS["twist_7"]
    controller.action()
    assert controller.failed_reason == "continuous_twist_event_timeout"

    raw = _FakeRawEnv()
    raw._hs014_events["continuous_twist_seen"] = True
    controller = HS014AutoController(raw)
    controller.phase_index = controller.PHASES.index("twist_7")
    controller.phase_frame = controller.DURATIONS["twist_7"]
    controller.action()
    assert controller.phase == "lift"


def test_zarr_episode_contract_contains_actions_truth_physics_and_cameras(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode"
    transitions = [
        _transition(0, succeed=False),
        _transition(1, succeed=True),
    ]
    summary = _write_zarr(
        episode / "replay.zarr",
        transitions,
        seed=19,
        source_commit="1" * 40,
        randomize=False,
    )
    assert summary["steps"] == 2
    assert episode_contract_errors(episode) == []
    assert set(REQUIRED_ZARR_FIELDS) <= set(summary["fields"])


def test_episode_contract_rejects_non_success(tmp_path: Path) -> None:
    episode = tmp_path / "episode"
    _write_zarr(
        episode / "replay.zarr",
        [_transition(0, succeed=False)],
        seed=19,
        source_commit="2" * 40,
        randomize=False,
    )
    assert "recorded trajectory does not end in success" in episode_contract_errors(
        episode
    )


def test_auto_controller_source_has_no_post_reset_state_writer() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "hs014_auto_collection.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        ".set_transforms(",
        ".set_velocities(",
        "write_root_pose_to_sim(",
        "write_root_velocity_to_sim(",
        "_set_task_body_world_pose(",
        "_set_body_rigid_grasp_target(",
        "_set_body_soft_target(",
    )
    assert not any(token in source for token in forbidden)


def test_direct_entry_prioritizes_repo_taskdev_package() -> None:
    source = (Path(__file__).resolve().parents[1] / "hs014_auto_collection.py").read_text(
        encoding="utf-8"
    )
    assert "REPO_ROOT = Path(__file__).resolve().parents[1]" in source
    assert source.index("sys.path.insert(0, str(REPO_ROOT))") < source.index("import zarr")
