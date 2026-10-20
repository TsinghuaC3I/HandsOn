from __future__ import annotations

import time
import types

import numpy as np
import pytest

from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
from handson.sim.isaaclab.bimanual_assembly_env import (
    _B03_NEWTON_HAND_TARGET_SLEW_RATE_ENV,
    _B03_NEWTON_HAND_TARGET_SLEW_RATE_RAD_S,
    BimanualAssemblyIsaacLabEnv,
)


class _Robot:
    def __init__(self, torch, default: list[float]) -> None:
        default_position = torch.tensor([default], dtype=torch.float64)
        joint_count = default_position.shape[1]
        self.data = types.SimpleNamespace(
            default_joint_pos=default_position,
            default_joint_vel=torch.zeros_like(default_position),
            joint_pos=default_position.clone(),
            soft_joint_pos_limits=torch.tensor(
                [[[-1.0, 1.0]] * joint_count],
                dtype=torch.float64,
            ),
        )
        self.position_targets: list[object] = []
        self.position_writes: list[object] = []
        self.velocity_writes: list[object] = []
        self.write_data_calls = 0

    def set_joint_position_target(self, target) -> None:
        self.position_targets.append(target.clone())

    def set_joint_velocity_target(self, target) -> None:
        self.velocity_writes.append(target.clone())

    def write_joint_state_to_sim(self, position, velocity) -> None:
        self.position_writes.append(position.clone())
        self.velocity_writes.append(velocity.clone())

    def write_data_to_sim(self) -> None:
        self.write_data_calls += 1


def _slew_env(torch) -> tuple[BimanualAssemblyIsaacLabEnv, dict[str, _Robot]]:
    env = BimanualAssemblyIsaacLabEnv.__new__(
        BimanualAssemblyIsaacLabEnv
    )
    robots = {
        "right": _Robot(torch, [0.0, 0.10, -0.10, 0.0]),
        "left": _Robot(torch, [0.0, -0.05, 0.05, 0.0]),
    }
    env.task_name = "t09_insert_peg"
    env._physics_backend = "newton"
    env._direct_hand_control = False
    env._newton_hand_target_slew_enabled = True
    env._newton_hand_target_slew_rate_rad_s = 2.0
    env._newton_hand_live_goal_serial = 0
    env._newton_hand_goal_targets = {}
    env._physics_dt = 1.0 / 120.0
    env._torch = torch
    env._scene = {
        "right_robot": robots["right"],
        "left_robot": robots["left"],
    }
    env._launch_isaac = True
    env._joint_targets = {
        side: robot.data.default_joint_pos.clone()
        for side, robot in robots.items()
    }
    env._hand_joint_ids = {"right": [1, 2], "left": [1, 2]}
    env._arm_joint_ids = {"right": [], "left": []}
    env._right_hand = np.asarray([0.50, -0.50], dtype=np.float64)
    env._left_hand = np.asarray([-0.40, 0.40], dtype=np.float64)
    env._right_tcp_pose = np.zeros(7, dtype=np.float64)
    env._left_tcp_pose = np.zeros(7, dtype=np.float64)
    env._right_pose_commanded = False
    env._left_pose_commanded = False
    env._arm_ik_backend = "dls"
    env._last_control_log_time = time.monotonic()
    env._compute_arm_joint_target = types.MethodType(
        lambda _self, _side, _robot, _pose, _commanded: (None, "idle"),
        env,
    )
    return env, robots


def test_b03_articulated_newton_hand_slew_defaults_to_two_rad_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_DIRECT_HAND_CONTROL", "0")
    monkeypatch.delenv(_B03_NEWTON_HAND_TARGET_SLEW_RATE_ENV, raising=False)

    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
        hz=30,
    )

    assert env._newton_hand_target_slew_enabled is True
    assert env._newton_hand_target_slew_rate_rad_s == pytest.approx(
        _B03_NEWTON_HAND_TARGET_SLEW_RATE_RAD_S
    )
    assert (
        env._newton_hand_target_slew_rate_rad_s * env._physics_dt
        == pytest.approx(1.0 / 120.0)
    )
    env._newton_ground_truth_enabled = True
    contract = env.get_physical_replay_reset_contract()["newton"][
        "hand_target_slew"
    ]
    assert contract == {
        "enabled": True,
        "scope": "b03_articulated_newton",
        "rate_rad_s": pytest.approx(2.0),
        "maximum_delta_per_physics_substep_rad": pytest.approx(1.0 / 120.0),
        "reference": "previous_applied_drive_target",
        "raw_hand_command_mutated": False,
        "recording": "actual_applied_per_substep_target",
        "replay": "exact_recorded_pre_substep_targets_bypass_slew",
    }


def test_hand_slew_is_scoped_to_articulated_b03_newton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_B03_NEWTON_HAND_TARGET_SLEW_RATE_ENV, raising=False)
    monkeypatch.setenv("HANDSON_ISAACLAB_DIRECT_HAND_CONTROL", "0")
    physx = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="physx",
        launch_isaac=False,
        image_obs=False,
    )
    a01_newton = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
    )
    assert physx._newton_hand_target_slew_enabled is False
    assert a01_newton._newton_hand_target_slew_enabled is False

    monkeypatch.setenv("HANDSON_ISAACLAB_DIRECT_HAND_CONTROL", "1")
    direct_b03 = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
    )
    assert direct_b03._newton_hand_target_slew_enabled is False

    monkeypatch.setenv("HANDSON_ISAACLAB_DIRECT_HAND_CONTROL", "0")
    monkeypatch.setenv(_B03_NEWTON_HAND_TARGET_SLEW_RATE_ENV, "0")
    disabled = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
    )
    assert disabled._newton_hand_target_slew_enabled is False


def test_hand_goal_stays_raw_while_applied_target_slews_each_substep() -> None:
    torch = pytest.importorskip("torch")
    env, robots = _slew_env(torch)
    raw_right = env._right_hand.copy()
    raw_left = env._left_hand.copy()
    initial = {
        side: target.clone() for side, target in env._joint_targets.items()
    }

    env._apply_joint_targets()

    np.testing.assert_array_equal(env._right_hand, raw_right)
    np.testing.assert_array_equal(env._left_hand, raw_left)
    for side in ("right", "left"):
        torch.testing.assert_close(env._joint_targets[side], initial[side])
    torch.testing.assert_close(
        env._newton_hand_goal_targets["right"],
        torch.tensor([[0.50, -0.50]], dtype=torch.float64),
    )

    maximum_delta = 2.0 / 120.0
    previous = {
        side: target.clone() for side, target in env._joint_targets.items()
    }
    for _ in range(4):
        assert env._advance_newton_hand_target_slew() == 4
        for side in ("right", "left"):
            current = env._joint_targets[side]
            delta = torch.abs(current - previous[side])[:, [1, 2]]
            assert float(torch.max(delta).item()) <= maximum_delta + 1.0e-12
            torch.testing.assert_close(
                robots[side].position_targets[-1],
                current,
            )
            previous[side] = current.clone()

    torch.testing.assert_close(
        env._joint_targets["right"][:, [1, 2]],
        torch.tensor(
            [[0.10 + 4.0 * maximum_delta, -0.10 - 4.0 * maximum_delta]],
            dtype=torch.float64,
        ),
    )
    np.testing.assert_array_equal(env._right_hand, raw_right)
    np.testing.assert_array_equal(env._left_hand, raw_left)


def test_hand_slew_runs_for_native_abc_robot_views_without_scene() -> None:
    """ABC tasks own ``_robots`` directly and intentionally have no scene."""

    torch = pytest.importorskip("torch")
    env, robots = _slew_env(torch)
    env._scene = None
    env._robots = robots
    env._control_robot = types.MethodType(
        lambda self, side: self._robots[side],
        env,
    )

    env._apply_joint_targets()
    previous = env._joint_targets["right"].clone()

    assert env._advance_newton_hand_target_slew() == 4
    assert not torch.equal(env._joint_targets["right"], previous)
    torch.testing.assert_close(
        robots["right"].position_targets[-1],
        env._joint_targets["right"],
    )


def test_exact_recorded_substep_callback_bypasses_live_slew() -> None:
    torch = pytest.importorskip("torch")
    env, robots = _slew_env(torch)
    recorded = torch.tensor(
        [[0.25, 0.333, -0.444, -0.25]],
        dtype=torch.float64,
    )
    advance_calls = 0

    def fail_if_slewed() -> int:
        nonlocal advance_calls
        advance_calls += 1
        return 0

    def apply_recorded_target(index: int, count: int) -> None:
        assert (index, count) == (2, 4)
        exact = recorded.clone()
        env._joint_targets["right"] = exact
        robots["right"].set_joint_position_target(exact)

    env._advance_newton_hand_target_slew = fail_if_slewed
    env._handson_pre_physics_substep_callback = apply_recorded_target

    mode = env._prepare_physics_substep_joint_targets(2, 4)

    assert mode == "exact_pre_substep_callback"
    assert advance_calls == 0
    torch.testing.assert_close(env._joint_targets["right"], recorded)
    torch.testing.assert_close(robots["right"].position_targets[-1], recorded)


def test_live_substep_callback_still_uses_slew_after_installing_new_goal() -> None:
    torch = pytest.importorskip("torch")
    env, _robots = _slew_env(torch)
    advance_calls = 0

    def advance() -> int:
        nonlocal advance_calls
        advance_calls += 1
        return 4

    def install_live_goal(_index: int, _count: int) -> None:
        env._newton_hand_live_goal_serial += 1

    env._advance_newton_hand_target_slew = advance
    env._handson_pre_physics_substep_callback = install_live_goal

    mode = env._prepare_physics_substep_joint_targets(0, 4)

    assert mode == "live_callback_target_slew"
    assert advance_calls == 1


def test_robot_reset_synchronizes_raw_goal_and_applied_slew_state() -> None:
    torch = pytest.importorskip("torch")
    env, robots = _slew_env(torch)
    env._newton_hand_live_goal_serial = 99
    env._newton_hand_goal_targets = {
        "right": torch.full((1, 2), 0.9, dtype=torch.float64),
        "left": torch.full((1, 2), -0.9, dtype=torch.float64),
    }
    env._joint_targets = {
        "right": torch.full((1, 4), 0.8, dtype=torch.float64),
        "left": torch.full((1, 4), -0.8, dtype=torch.float64),
    }
    env._ik_controllers = {}
    env._arm_pose_filters = {"right": None, "left": None}
    env._right_pose_commanded = True
    env._left_pose_commanded = True

    env._restore_default_robot_joint_state(
        reset_controllers=False,
        sync=False,
    )

    assert env._newton_hand_live_goal_serial == 0
    for side in ("right", "left"):
        default = robots[side].data.default_joint_pos
        torch.testing.assert_close(env._joint_targets[side], default)
        torch.testing.assert_close(
            env._newton_hand_goal_targets[side],
            default[:, [1, 2]],
        )
        assert robots[side].write_data_calls == 1
    np.testing.assert_allclose(env._right_hand, [0.10, -0.10])
    np.testing.assert_allclose(env._left_hand, [-0.05, 0.05])
    assert env._right_pose_commanded is False
    assert env._left_pose_commanded is False
