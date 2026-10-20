from __future__ import annotations

import json
from urllib.request import Request, urlopen

import numpy as np
import pytest

from handson.runtime.primitive_control import (
    AdaptiveGraspExecutor,
    HandPrimitiveController,
    PrimitiveCommand,
    PrimitiveCommandBroker,
    PrimitiveCommandError,
    PrimitiveHTTPService,
    adaptive_grasp_strategies,
    apply_primitive_command,
    available_grasp_presets,
    native_hand_preset,
)
from handson.sim.envs.hand_specs import SIMPLIFIED_POST_IK_ARM_TYPE
from handson.sim.isaaclab.abc_collection_env import (
    _A01_CLEAR_RIGHT_DETACHED_PALM_START,
    _RAISED_LEFT_START_TCP,
    HandsOnABCIsaacLabEnv,
)


def _controller(*, bimanual: bool = False) -> HandPrimitiveController:
    controller = HandPrimitiveController(
        hand_type="wuji1",
        bimanual=bimanual,
        workspace_lower=(-1.0, -1.0, 0.0),
        workspace_upper=(1.0, 1.0, 2.0),
        hand_slew_rate_rad_s=1.0,
    )
    controller.sync_side(
        "right",
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        np.zeros(20),
    )
    if bimanual:
        controller.sync_side(
            "left",
            [0.0, 0.2, 1.0, 1.0, 0.0, 0.0, 0.0],
            np.zeros(20),
        )
    return controller


def test_move_clamps_workspace_and_local_frame_uses_tcp_orientation() -> None:
    controller = _controller()
    controller.rotate_axis([0, 0, 1], 90.0)
    controller.move([0.1, 0.0, 0.0], frame="hand")
    assert controller.poses["right"][:3] == pytest.approx([0.0, 0.1, 1.0])

    controller.move([10.0, 10.0, -10.0])
    assert controller.poses["right"][:3] == pytest.approx([1.0, 1.0, 0.0])


def test_rotation_stays_normalized_and_clockwise_uses_local_wrist_axis() -> None:
    controller = _controller()
    command = PrimitiveCommand(
        1,
        "rotate",
        {"clockwise_degrees": 50.0},
    )
    apply_primitive_command(controller, command)
    quaternion = controller.poses["right"][3:7]
    assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    assert quaternion[3] < 0.0


def test_grasp_presets_are_native_and_slew_instead_of_jump() -> None:
    controller = _controller()
    power = native_hand_preset(
        "wuji1", "full", side="right", bimanual=False
    )
    pinch = native_hand_preset(
        "wuji1", "pinch", side="right", bimanual=False
    )
    assert power.shape == (20,)
    assert pinch.shape == (20,)
    assert not np.allclose(power, pinch)

    controller.set_grasp("full")
    controller.advance(0.1)
    assert np.max(np.abs(controller.hands["right"])) <= 0.1000001
    assert not np.allclose(controller.hands["right"], power)


def test_extended_grasp_presets_are_finite_native_wuji_targets() -> None:
    presets = available_grasp_presets()
    assert presets == (
        "open",
        "relax",
        "power",
        "cylindrical",
        "spherical",
        "pinch",
        "tripod",
        "lateral",
        "hook",
        "point",
    )
    targets = {
        preset: native_hand_preset(
            "wuji1", preset, side="right", bimanual=False
        )
        for preset in presets
    }
    assert all(target.shape == (20,) for target in targets.values())
    assert all(np.all(np.isfinite(target)) for target in targets.values())
    assert not np.allclose(targets["spherical"], targets["cylindrical"])
    np.testing.assert_allclose(
        native_hand_preset("wuji1", "cup", side="right", bimanual=False),
        targets["cylindrical"],
    )


def test_independent_finger_curls_and_raw_joint_values() -> None:
    controller = _controller()
    power = native_hand_preset(
        "wuji1", "power", side="right", bimanual=False
    )
    target = controller.set_fingers(
        {"thumb": 0.5, "index": 1.0, "middle": 0.0}
    )
    np.testing.assert_allclose(target[0:4], 0.5 * power[0:4])
    np.testing.assert_allclose(target[4:8], power[4:8])
    np.testing.assert_allclose(target[8:12], 0.0)

    target = controller.set_fingers(
        {"finger5": [0.1, -0.2, 0.3, 0.4]}
    )
    np.testing.assert_allclose(target[16:20], [0.1, -0.2, 0.3, 0.4])
    with pytest.raises(PrimitiveCommandError):
        controller.set_fingers({"index": 1.2})


def test_adaptive_grasp_selects_shape_strategy_and_requires_physical_lift() -> None:
    controller = _controller()
    adaptive = AdaptiveGraspExecutor()
    started = adaptive.start(
        controller,
        {
            "object": "apple",
            "strategy": "auto",
            "lift_distance_m": 0.03,
        },
    )
    assert started["strategy"] == "spherical"
    assert adaptive_grasp_strategies("025_mug")[0] == "hook"

    feedback = {
        "right": {
            "actual_pose_wxyz": [0, 0, 1, 1, 0, 0, 0],
            "force_saturated": False,
            "finger_tracking": {
                name: {"contact_likely": name in {"thumb", "index", "middle"}}
                for name in ("thumb", "index", "middle", "ring", "pinky")
            },
        }
    }
    objects = {
        "apple": {"pose_wxyz": [0, 0, 0.96, 1, 0, 0, 0]},
    }
    for _ in range(3):
        adaptive.update(
            controller,
            dt=0.1,
            physical_hands=feedback,
            objects=objects,
        )
    adaptive.update(
        controller,
        dt=0.3,
        physical_hands=feedback,
        objects=objects,
    )
    assert adaptive.state()["phase"] == "lifting"
    assert controller.poses["right"][2] == pytest.approx(1.03)

    lifted_feedback = {
        "right": {
            **feedback["right"],
            "actual_pose_wxyz": [0, 0, 1.03, 1, 0, 0, 0],
        }
    }
    lifted_objects = {
        "apple": {"pose_wxyz": [0, 0, 0.98, 1, 0, 0, 0]},
    }
    finished = adaptive.update(
        controller,
        dt=0.8,
        physical_hands=lifted_feedback,
        objects=lifted_objects,
    )
    assert finished["active"] is False
    assert finished["result"] == "succeeded"


def test_action_layout_matches_single_and_bimanual_handson_contracts() -> None:
    assert _controller().action().shape == (27,)
    controller = _controller(bimanual=True)
    action = controller.action()
    assert action.shape == (54,)
    assert action[:7] == pytest.approx(controller.poses["right"])
    assert action[7:14] == pytest.approx(controller.poses["left"])


def test_command_validation_and_control_plane_effects() -> None:
    controller = _controller(bimanual=True)
    apply_primitive_command(
        controller,
        PrimitiveCommand(1, "move", {"hand": "left", "x": 0.02}),
    )
    assert controller.poses["left"][0] == pytest.approx(0.02)
    apply_primitive_command(controller, PrimitiveCommand(2, "toggle_hand", {}))
    assert controller.selected_hand == "left"
    assert apply_primitive_command(
        controller, PrimitiveCommand(3, "reset", {})
    )["reset"]
    assert apply_primitive_command(
        controller, PrimitiveCommand(4, "quit", {})
    )["quit"]
    with pytest.raises(PrimitiveCommandError):
        controller.select_hand("middle")


def test_http_service_only_queues_commands_for_simulation_thread() -> None:
    broker = PrimitiveCommandBroker()
    broker.update_state({"ready": True, "task": "t26_place_fruit_in_bucket"})
    service = PrimitiveHTTPService(broker, port=0)
    host, port = service.start()
    try:
        with urlopen(f"http://{host}:{port}/v1/state", timeout=2.0) as response:
            state = json.loads(response.read().decode("utf-8"))
        assert state["ready"] is True

        request = Request(
            f"http://{host}:{port}/v1/move",
            data=json.dumps({"delta_position_m": [0.01, 0.0, 0.0]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2.0) as response:
            queued = json.loads(response.read().decode("utf-8"))
        assert queued["status"] == "queued"
        commands = broker.drain()
        assert len(commands) == 1
        assert commands[0].operation == "move"

        with urlopen(
            f"http://{host}:{port}/v1/primitives", timeout=2.0
        ) as response:
            primitives = json.loads(response.read().decode("utf-8"))
        assert "cylindrical" in primitives["grasp_presets"]

        with urlopen(
            f"http://{host}:{port}/Pinch?hand=right", timeout=2.0
        ) as response:
            queued = json.loads(response.read().decode("utf-8"))
        assert queued["operation"] == "grasp"
        command = broker.drain()[0]
        assert command.payload["type"] == "pinch"

        with urlopen(
            f"http://{host}:{port}/move?x=0.01&y=0&z=0", timeout=2.0
        ) as response:
            queued = json.loads(response.read().decode("utf-8"))
        assert queued["operation"] == "move"
        assert broker.drain()[0].payload["x"] == pytest.approx(0.01)

        with urlopen(
            f"http://{host}:{port}/lift?distance=0.02", timeout=2.0
        ) as response:
            queued = json.loads(response.read().decode("utf-8"))
        assert queued["operation"] == "move"
        lift = broker.drain()[0]
        assert lift.payload["delta_position_m"] == pytest.approx(
            [0.0, 0.0, 0.02]
        )
    finally:
        service.close()


def test_hands_only_a01_uses_the_normal_a01_physical_palm_pose() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t26_place_fruit_in_bucket",
        arm_type=SIMPLIFIED_POST_IK_ARM_TYPE,
        launch_isaac=False,
        image_obs=False,
    )
    env._sample_reset_state(clear_manual_request=True)
    np.testing.assert_allclose(
        env._right_tcp_pose,
        _A01_CLEAR_RIGHT_DETACHED_PALM_START,
    )
    np.testing.assert_allclose(env._left_tcp_pose, _RAISED_LEFT_START_TCP)


def test_physical_servo_bounds_target_around_actual_physx_pose() -> None:
    env = HandsOnABCIsaacLabEnv.__new__(HandsOnABCIsaacLabEnv)
    actual = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    requested = np.asarray(
        [1.0, 0.0, 1.0, np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    )
    bounded = env._bound_direct_servo_pose(actual, requested)

    assert np.linalg.norm(bounded[:3] - actual[:3]) == pytest.approx(0.010)
    angular_step = env._direct_pose_velocity(actual, bounded, 1.0)[3:6]
    assert 0.0 < np.linalg.norm(angular_step) <= np.deg2rad(5.0)


def test_physical_servo_substep_never_uses_live_state_writer() -> None:
    env = HandsOnABCIsaacLabEnv.__new__(HandsOnABCIsaacLabEnv)
    env._direct_hand_control = True
    env._direct_physics_servo_enabled = True
    env._n_substeps = 4
    env._direct_motion_substep = 0
    env._robots = {"right": object()}
    pose = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    hand = np.zeros(20)
    env._direct_motion_start_pose = {"right": pose.copy()}
    env._direct_motion_target_pose = {"right": pose.copy()}
    env._direct_motion_start_hand = {"right": hand.copy()}
    env._direct_motion_target_hand = {"right": hand.copy()}
    env._direct_sim_pose = {"right": pose.copy()}
    env._direct_sim_hand = {"right": hand.copy()}
    calls = []
    env._apply_direct_hand_servo_substep = (
        lambda side, target_pose, target_hand: calls.append(
            (side, target_pose.copy(), target_hand.copy())
        )
    )
    env._write_direct_hand_state = lambda *args, **kwargs: pytest.fail(
        "physical servo must not overwrite root or joint state during a step"
    )

    env._write_direct_hand_substep()

    assert len(calls) == 1
    assert calls[0][0] == "right"
