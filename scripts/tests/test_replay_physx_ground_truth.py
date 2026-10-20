import os
from pathlib import Path
import subprocess
import sys


def test_physx_replay_loader_requires_paired_named_snapshots() -> None:
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "handson"), str(root), env.get("PYTHONPATH", "")]
    )
    code = r'''
import json
import tempfile
from pathlib import Path

import numpy as np
import zarr

from scripts.replay_demos_zarr import (
    _assert_robot_path_callbacks_available,
    _demo_source_succeeded,
    _load_physx_replay_trajectory,
    _physx_replay_readback_field_errors,
    _physx_replay_readback_max_error,
    _physx_replay_verification_report,
    _validate_exact_robot_substep_path,
)

joint_valid = np.asarray([[1, 1, 0], [1, 0, 0]], dtype=np.uint8)
before_robot = {
    "physx_robot_joint_position": np.asarray(
        [[0.0, 0.1, 0.0], [0.2, 0.0, 0.0]]
    ),
    "physx_robot_joint_velocity": np.zeros((2, 3)),
    "physx_robot_joint_valid": joint_valid.copy(),
    "physx_robot_joint_target": np.asarray(
        [[0.2, 0.3, 0.0], [0.4, 0.0, 0.0]]
    ),
    "physx_robot_joint_target_valid": joint_valid.copy(),
}
after_robot = {
    "physx_robot_joint_position": np.asarray(
        [[0.2, 0.3, 0.0], [0.4, 0.0, 0.0]]
    ),
    "physx_robot_joint_velocity": np.asarray(
        [[0.1, 0.1, 0.0], [0.1, 0.0, 0.0]]
    ),
    "physx_robot_joint_valid": joint_valid.copy(),
    "physx_robot_joint_target": np.asarray(
        [[0.2, 0.3, 0.0], [0.4, 0.0, 0.0]]
    ),
    "physx_robot_joint_target_valid": joint_valid.copy(),
}
substep_position = np.asarray(
    [
        [[0.1, 0.2, 0.0], [0.3, 0.0, 0.0]],
        after_robot["physx_robot_joint_position"],
    ]
)
substep_velocity = np.asarray(
    [
        [[0.05, 0.05, 0.0], [0.05, 0.0, 0.0]],
        after_robot["physx_robot_joint_velocity"],
    ]
)
substep_target = np.stack(
    [after_robot["physx_robot_joint_target"]] * 2
)
substep_valid = np.stack([joint_valid] * 2)
substep_target_valid = np.stack([joint_valid] * 2)

active = _validate_exact_robot_substep_path(
    before_robot,
    after_robot,
    substep_position,
    substep_velocity,
    substep_valid,
    substep_target,
    substep_target_valid,
)
np.testing.assert_array_equal(active, joint_valid.astype(bool))

bad_valid = substep_valid.copy()
bad_valid[0, 0, 0] = 0
try:
    _validate_exact_robot_substep_path(
        before_robot,
        after_robot,
        substep_position,
        substep_velocity,
        bad_valid,
        substep_target,
        substep_target_valid,
    )
except RuntimeError as exc:
    assert "coverage is incomplete" in str(exc)
else:
    raise AssertionError("incomplete substep joint coverage was accepted")

bad_position = substep_position.copy()
bad_position[0, 0, 0] = np.nan
try:
    _validate_exact_robot_substep_path(
        before_robot,
        after_robot,
        bad_position,
        substep_velocity,
        substep_valid,
        substep_target,
        substep_target_valid,
    )
except RuntimeError as exc:
    assert "non-finite substep position" in str(exc)
else:
    raise AssertionError("non-finite substep position was accepted")

bad_endpoint = substep_position.copy()
bad_endpoint[-1, 0, 0] += 0.01
try:
    _validate_exact_robot_substep_path(
        before_robot,
        after_robot,
        bad_endpoint,
        substep_velocity,
        substep_valid,
        substep_target,
        substep_target_valid,
    )
except RuntimeError as exc:
    assert "control-frame after-state for position" in str(exc)
else:
    raise AssertionError("inconsistent robot-path endpoint was accepted")

class CallbackEnv:
    _handson_pre_physics_substep_callback = None
    _handson_post_physics_substep_callback = staticmethod(lambda *_: None)

try:
    _assert_robot_path_callbacks_available(CallbackEnv())
except RuntimeError as exc:
    assert "_handson_post_physics_substep_callback" in str(exc)
else:
    raise AssertionError("an existing PhysX post callback was accepted")

safe_callback_env = CallbackEnv()
safe_callback_env.physical_replay_post_substep_observer_is_safe = (
    lambda callback: callback
    is safe_callback_env._handson_post_physics_substep_callback
)
_assert_robot_path_callbacks_available(safe_callback_env)

truthy_callback_env = CallbackEnv()
truthy_callback_env.physical_replay_post_substep_observer_is_safe = (
    lambda _callback: "true"
)
try:
    _assert_robot_path_callbacks_available(truthy_callback_env)
except RuntimeError as exc:
    assert "_handson_post_physics_substep_callback" in str(exc)
else:
    raise AssertionError("a non-boolean callback approval was accepted")

with tempfile.TemporaryDirectory() as temp:
    demo = Path(temp) / "c04_demo"
    demo.mkdir()
    schema = {
        "environment_schema": {
            "physx": {
                "robot_sides": ["right", "left"],
                "robot_joint_names": {
                    "right": ["r0", "r1"],
                    "left": ["l0", "l1"],
                },
                "rigid_body_names": [],
                "articulation_names": ["fixed_object"],
                "articulation_dof_names": {"fixed_object": []},
                "deformable_name": None,
            }
        }
    }
    (demo / "recording_schema.json").write_text(json.dumps(schema))
    (demo / "collection_meta.json").write_text(
        json.dumps({"task_outcome": "success"})
    )
    group = zarr.open_group(str(demo / "replay.zarr"), mode="w")
    data = group.create_group("data")
    shape = (2, 2, 2)
    for name, values in {
        "physx_robot_joint_position": np.arange(8).reshape(shape),
        "next_physx_robot_joint_position": np.arange(8).reshape(shape) + 10,
        "physx_robot_joint_velocity": np.zeros(shape),
        "next_physx_robot_joint_velocity": np.ones(shape),
        "physx_robot_joint_valid": np.ones(shape, dtype=np.uint8),
        "next_physx_robot_joint_valid": np.ones(shape, dtype=np.uint8),
        "physx_task_articulation_root_state": np.zeros((2, 1, 13)),
        "next_physx_task_articulation_root_state": np.ones((2, 1, 13)),
        "physx_task_articulation_root_valid": np.ones(
            (2, 1), dtype=np.uint8
        ),
        "next_physx_task_articulation_root_valid": np.ones(
            (2, 1), dtype=np.uint8
        ),
    }.items():
        data.create_dataset(name, data=values)

    replay = _load_physx_replay_trajectory(
        demo / "replay.zarr",
        demo,
        expected_steps=2,
    )
    assert replay.step_count == 2
    np.testing.assert_array_equal(
        replay.snapshot(1, after_action=False)["physx_robot_joint_position"],
        np.asarray([[4, 5], [6, 7]]),
    )
    np.testing.assert_array_equal(
        replay.snapshot(1, after_action=True)["physx_robot_joint_position"],
        np.asarray([[14, 15], [16, 17]]),
    )
    assert all(
        "articulation_dof" not in name
        for name in replay.snapshot(1, after_action=True)
    )
    assert _demo_source_succeeded(demo)
    expected = replay.snapshot(1, after_action=True)
    actual = {name: value.copy() for name, value in expected.items()}
    assert _physx_replay_readback_max_error(expected, actual) == 0.0
    assert _physx_replay_readback_field_errors(expected, actual) == {
        "physx_robot_joint_position": 0.0,
        "physx_robot_joint_velocity": 0.0,
        "physx_task_articulation_root_state": 0.0,
    }
    actual["physx_robot_joint_position"] = actual[
        "physx_robot_joint_position"
    ].astype(np.float64)
    actual["physx_robot_joint_position"][0, 0] += 0.25
    assert _physx_replay_readback_max_error(expected, actual) == 0.25

    verification = _physx_replay_verification_report(
        [
            {
                "infos": {
                    "physx_ground_truth_replay": True,
                    "source_state_index": 0,
                    "physx_initial_readback_max_abs_error": 1e-7,
                    "physx_readback_max_abs_error": 0.0,
                }
            },
            {
                "infos": {
                    "physx_ground_truth_replay": True,
                    "source_state_index": 1,
                    "physx_initial_readback_max_abs_error": 1e-7,
                    "physx_readback_max_abs_error": 0.25,
                }
            },
        ]
    )
    assert verification["checked_state_count"] == 3
    assert verification["verified_state_count"] == 2
    assert verification["mode"] == "state_restore"
    assert verification["trajectory_max_abs_error"] == 0.25
    assert verification["trajectory_verified"] is False
    assert verification["failed_states"] == [
        {"phase": "post_action", "source_state_index": 1}
    ]

    action_verification = _physx_replay_verification_report(
        [
            {
                "infos": {
                    "physx_action_replay_audit": True,
                    "physx_replay_mode": "action_reexecution",
                    "source_state_index": 0,
                    "physx_initial_readback_max_abs_error": 1e-7,
                    "physx_initial_readback_field_max_abs_error": {
                        "physx_robot_joint_position": 1e-7,
                    },
                    "physx_readback_max_abs_error": 0.02,
                    "physx_readback_field_max_abs_error": {
                        "physx_robot_joint_position": 0.02,
                        "physx_task_articulation_root_state": 0.01,
                    },
                }
            }
        ]
    )
    assert action_verification["mode"] == "action_reexecution"
    assert action_verification["checked_state_count"] == 2
    assert action_verification["verified_state_count"] == 1
    assert action_verification["trajectory_verified"] is False
    assert action_verification["trajectory_field_max_abs_error"] == {
        "physx_robot_joint_position": 0.02,
        "physx_task_articulation_root_state": 0.01,
    }

    target_verification = _physx_replay_verification_report(
        [
            {
                "infos": {
                    "physx_joint_target_replay_audit": True,
                    "physx_replay_mode": "joint_target_reexecution",
                    "source_state_index": 0,
                    "physx_initial_readback_max_abs_error": 0.0,
                    "physx_readback_max_abs_error": 0.01,
                }
            }
        ]
    )
    assert target_verification["mode"] == "joint_target_reexecution"
    assert target_verification["checked_state_count"] == 2
    assert target_verification["verified_state_count"] == 1
    assert target_verification["trajectory_verified"] is False

    del data["next_physx_robot_joint_velocity"]
    try:
        _load_physx_replay_trajectory(
            demo / "replay.zarr",
            demo,
            expected_steps=2,
        )
    except ValueError as exc:
        assert "next_physx_robot_joint_qd" in str(exc)
    else:
        raise AssertionError("unpaired PhysX fields were accepted")
'''

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def _exercise_recorded_robot_path_with_b320_airborne_observer() -> None:
    import numpy as np

    from handson.sim.isaaclab.b320_bimanual_env import (
        HandsOnB320BimanualIsaacLabEnv,
    )
    from scripts.replay_demos_zarr import _step_with_recorded_robot_path

    os.environ.pop("HANDSON_GUIDED_PHYSICS_REPLAY", None)
    joint_valid = np.asarray([[1, 1, 0], [1, 0, 0]], dtype=np.uint8)
    before_robot = {
        "physx_robot_joint_position": np.asarray(
            [[0.0, 0.1, 0.0], [0.2, 0.0, 0.0]]
        ),
        "physx_robot_joint_velocity": np.zeros((2, 3)),
        "physx_robot_joint_valid": joint_valid.copy(),
        "physx_robot_joint_target": np.asarray(
            [[0.2, 0.3, 0.0], [0.4, 0.0, 0.0]]
        ),
        "physx_robot_joint_target_valid": joint_valid.copy(),
    }
    after_robot = {
        "physx_robot_joint_position": np.asarray(
            [[0.2, 0.3, 0.0], [0.4, 0.0, 0.0]]
        ),
        "physx_robot_joint_velocity": np.asarray(
            [[0.1, 0.1, 0.0], [0.1, 0.0, 0.0]]
        ),
        "physx_robot_joint_valid": joint_valid.copy(),
        "physx_robot_joint_target": np.asarray(
            [[0.2, 0.3, 0.0], [0.4, 0.0, 0.0]]
        ),
        "physx_robot_joint_target_valid": joint_valid.copy(),
    }
    substep_position = np.asarray(
        [
            [[0.1, 0.2, 0.0], [0.3, 0.0, 0.0]],
            after_robot["physx_robot_joint_position"],
        ]
    )
    substep_velocity = np.asarray(
        [
            [[0.05, 0.05, 0.0], [0.05, 0.0, 0.0]],
            after_robot["physx_robot_joint_velocity"],
        ]
    )
    after_snapshot = {
        **after_robot,
        "physx_substep_robot_joint_position": substep_position,
        "physx_substep_robot_joint_velocity": substep_velocity,
        "physx_substep_robot_joint_valid": np.stack([joint_valid] * 2),
        "physx_substep_robot_joint_target": np.stack(
            [after_robot["physx_robot_joint_target"]] * 2
        ),
        "physx_substep_robot_joint_target_valid": np.stack(
            [joint_valid] * 2
        ),
        "physx_substep_count": np.asarray([2], dtype=np.int64),
    }

    raw_env = object.__new__(HandsOnB320BimanualIsaacLabEnv)
    raw_env.task_name = "t10_lift_carton"
    raw_env._b320_carton_airborne_substeps = 0
    raw_env._read_contacts = lambda *, update_latches: None
    raw_env._carton_instantaneous_state = lambda: {
        "airborne": True,
        "hold_valid": True,
    }
    raw_env.env_step = 1
    raw_env._control_dt = 1.0 / 30.0
    raw_env._time_limit = 10.0
    raw_env._refresh_object_poses = lambda: None
    raw_env._compute_success = (
        lambda: raw_env._b320_carton_airborne_substeps >= 2
    )
    raw_env._compute_failure = lambda: False
    raw_env._compute_observation = lambda: {"state": np.zeros(1)}
    raw_env._attach_physx_ground_truth = lambda observation: observation

    def step_with_current_joint_targets():
        for substep_index in range(2):
            raw_env._handson_pre_physics_substep_callback(substep_index, 2)
            raw_env._handson_post_physics_substep_callback(substep_index, 2)
        return {"state": np.zeros(1)}, 0.0, False, False, {}

    raw_env.step_with_current_joint_targets = step_with_current_joint_targets

    class Wrapper:
        unwrapped = raw_env

        @staticmethod
        def observation(observation):
            return observation

    class Trace:
        @staticmethod
        def restore_robot_state_only(_snapshot, _schema):
            return None

        @staticmethod
        def apply_robot_joint_targets(_snapshot, _schema):
            return None

        @staticmethod
        def snapshot():
            return after_robot

    _next_obs, reward, done, truncated, info = _step_with_recorded_robot_path(
        Wrapper(),
        Trace(),
        {},
        before_robot,
        after_snapshot,
    )

    assert raw_env._b320_carton_airborne_substeps == 2
    assert reward == 1.0
    assert done is True
    assert truncated is False
    assert info["succeed"] is True
    assert callable(raw_env._handson_post_physics_substep_callback)
    assert not hasattr(raw_env, "_handson_pre_physics_substep_callback")


def test_recorded_robot_path_preserves_the_b320_airborne_observer() -> None:
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "handson"), str(root), env.get("PYTHONPATH", "")]
    )
    code = (
        "from scripts.tests.test_replay_physx_ground_truth import "
        "_exercise_recorded_robot_path_with_b320_airborne_observer; "
        "_exercise_recorded_robot_path_with_b320_airborne_observer()"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
