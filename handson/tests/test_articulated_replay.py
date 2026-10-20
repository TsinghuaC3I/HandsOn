from __future__ import annotations

import numpy as np
import pytest

from handson.sim.isaaclab.articulated_replay import (
    RecordedArticulatedDrivePath,
    run_articulated_drive_control_step,
    validate_recorded_physics_configuration,
)


def _schema() -> dict:
    return {
        "environment_schema": {
            "physx": {
                "robot_sides": ["right", "left"],
                "robot_joint_names": {
                    "right": ["r0", "r1"],
                    "left": ["l0"],
                },
                "physics_config": {
                    "device": "cuda:0",
                    "physics_dt": 1.0 / 120.0,
                    "n_substeps": 4,
                    "solver": {
                        "solver_type": 1,
                        "enable_enhanced_determinism": True,
                    },
                },
            }
        }
    }


def _snapshot() -> dict[str, np.ndarray]:
    targets = np.asarray(
        [
            [[0.1, 0.2, 99.0], [0.3, 99.0, 99.0]],
            [[0.2, 0.4, 99.0], [0.5, 99.0, 99.0]],
            [[0.4, 0.7, 99.0], [0.9, 99.0, 99.0]],
        ],
        dtype=np.float64,
    )
    valid = np.asarray(
        [
            [[1, 1, 0], [1, 0, 0]],
            [[1, 1, 0], [1, 0, 0]],
            [[1, 1, 0], [1, 0, 0]],
        ],
        dtype=np.uint8,
    )
    return {
        "physx_substep_count": np.asarray([3], dtype=np.int64),
        "physx_substep_robot_joint_target": targets,
        "physx_substep_robot_joint_target_valid": valid,
        "physx_robot_joint_target": targets[-1].copy(),
        "physx_robot_joint_target_valid": valid[-1].copy(),
        # These measured fields must be ignored by articulated-drive replay.
        "physx_substep_robot_joint_position": targets + 10.0,
        "physx_substep_robot_joint_velocity": targets + 20.0,
        "physx_substep_robot_joint_effort": targets + 30.0,
    }


def _guard_partial_snapshot() -> dict[str, np.ndarray]:
    full = _snapshot()
    targets = np.zeros((4, 2, 3), dtype=np.float64)
    valid = np.zeros((4, 2, 3), dtype=np.uint8)
    targets[:2] = full["physx_substep_robot_joint_target"][:2]
    valid[:2] = full["physx_substep_robot_joint_target_valid"][:2]
    return {
        "physx_substep_count": np.asarray([2], dtype=np.int64),
        "physx_substep_robot_joint_target": targets,
        "physx_substep_robot_joint_target_valid": valid,
        "physx_robot_joint_target": targets[1].copy(),
        "physx_robot_joint_target_valid": valid[1].copy(),
        # Measured fields are solver output and remain irrelevant here.
        "physx_substep_robot_joint_position": targets + 10.0,
        "physx_substep_robot_joint_velocity": targets + 20.0,
        "physx_substep_robot_joint_effort": targets + 30.0,
    }


def test_drive_path_requires_complete_finite_targets_and_matches_endpoint() -> None:
    path = RecordedArticulatedDrivePath(_schema())
    step = path.control_step(_snapshot())

    assert step.substep_capacity == 3
    assert step.substep_count == 3
    assert step.expected_joint_count == 3
    assert step.max_intra_step_target_delta == pytest.approx(0.4)
    assert set(step.target_snapshot(0)) == {
        "physx_robot_joint_target",
        "physx_robot_joint_target_valid",
    }

    incomplete = _snapshot()
    incomplete["physx_substep_robot_joint_target_valid"][1, 0, 1] = 0
    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        path.control_step(incomplete)

    non_finite = _snapshot()
    non_finite["physx_substep_robot_joint_target"][1, 1, 0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite target"):
        path.control_step(non_finite)

    bad_endpoint = _snapshot()
    bad_endpoint["physx_robot_joint_target"][0, 0] += 0.01
    with pytest.raises(RuntimeError, match="final substep target"):
        path.control_step(bad_endpoint)


def test_drive_path_accepts_strictly_zero_guard_terminal_tail() -> None:
    path = RecordedArticulatedDrivePath(_schema())
    with pytest.raises(RuntimeError, match="explicit authorization"):
        path.control_step(_guard_partial_snapshot())
    step = path.control_step(
        _guard_partial_snapshot(),
        allow_partial_terminal_guard=True,
    )

    assert step.substep_capacity == 4
    assert step.substep_count == 2
    assert step.max_intra_step_target_delta == pytest.approx(0.2)
    np.testing.assert_array_equal(
        step.target_snapshot(1)["physx_robot_joint_target"],
        step.targets[1],
    )
    with pytest.raises(IndexError, match=r"outside \[0, 2\)"):
        step.target_snapshot(2)

    nonzero_valid_tail = _guard_partial_snapshot()
    nonzero_valid_tail["physx_substep_robot_joint_target_valid"][2, 0, 0] = 1
    with pytest.raises(RuntimeError, match="target-valid tail"):
        path.control_step(
            nonzero_valid_tail,
            allow_partial_terminal_guard=True,
        )

    nonzero_target_tail = _guard_partial_snapshot()
    nonzero_target_tail["physx_substep_robot_joint_target"][3, 1, 2] = 1.0e-30
    with pytest.raises(RuntimeError, match="target tail"):
        path.control_step(
            nonzero_target_tail,
            allow_partial_terminal_guard=True,
        )

    count_exceeds_capacity = _guard_partial_snapshot()
    count_exceeds_capacity["physx_substep_count"][0] = 5
    with pytest.raises(RuntimeError, match="exceeds.*capacity"):
        path.control_step(
            count_exceeds_capacity,
            allow_partial_terminal_guard=True,
        )


class _FakeTrace:
    def __init__(self) -> None:
        self.snapshots: list[dict[str, np.ndarray]] = []

    def apply_robot_joint_targets(self, snapshot, recorded_schema) -> int:
        assert recorded_schema == _schema()
        assert set(snapshot) == {
            "physx_robot_joint_target",
            "physx_robot_joint_target_valid",
        }
        self.snapshots.append(
            {name: np.asarray(value).copy() for name, value in snapshot.items()}
        )
        return 3


class _FakeEnv:
    def __init__(
        self,
        substep_capacity: int = 3,
        *,
        break_after_substeps: int | None = None,
    ) -> None:
        self.substep_capacity = substep_capacity
        self.break_after_substeps = break_after_substeps

    def step_with_current_joint_targets(self):
        for index in range(self.substep_capacity):
            self._handson_pre_physics_substep_callback(
                index,
                self.substep_capacity,
            )
            post_substep = getattr(
                self,
                "_handson_post_physics_substep_callback",
                None,
            )
            if callable(post_substep):
                post_substep(index, self.substep_capacity)
            if self.break_after_substeps == index + 1:
                break
        return "obs", 0.0, False, False, {"succeed": False}


def test_control_step_writes_only_targets_at_every_substep() -> None:
    path = RecordedArticulatedDrivePath(_schema())
    drive_step = path.control_step(_snapshot())
    raw_env = _FakeEnv()
    trace = _FakeTrace()

    result, report = run_articulated_drive_control_step(
        raw_env,
        trace,
        _schema(),
        drive_step,
    )

    assert result == ("obs", 0.0, False, False, {"succeed": False})
    assert not hasattr(raw_env, "_handson_pre_physics_substep_callback")
    assert len(trace.snapshots) == 3
    for index, applied in enumerate(trace.snapshots):
        np.testing.assert_array_equal(
            applied["physx_robot_joint_target"],
            drive_step.targets[index],
        )
    assert report == {
        "recorded_substep_capacity": 3,
        "recorded_substep_count": 3,
        "applied_robot_joint_target_count": 9,
        "applied_robot_joint_target_count_per_substep": 3,
        "max_intra_step_target_delta": pytest.approx(0.4),
    }


def test_control_step_preserves_an_explicit_read_only_post_observer() -> None:
    drive_step = RecordedArticulatedDrivePath(_schema()).control_step(
        _snapshot()
    )
    raw_env = _FakeEnv()
    observed: list[int] = []
    raw_env._handson_post_physics_substep_callback = (
        lambda index, _count: observed.append(index)
    )
    raw_env.physical_replay_post_substep_observer_is_safe = (
        lambda callback: callback
        is raw_env._handson_post_physics_substep_callback
    )

    run_articulated_drive_control_step(
        raw_env,
        _FakeTrace(),
        _schema(),
        drive_step,
    )

    assert observed == [0, 1, 2]
    assert callable(raw_env._handson_post_physics_substep_callback)
    assert not hasattr(raw_env, "_handson_pre_physics_substep_callback")


def test_control_step_rejects_a_non_boolean_observer_approval() -> None:
    drive_step = RecordedArticulatedDrivePath(_schema()).control_step(
        _snapshot()
    )
    raw_env = _FakeEnv()
    raw_env._handson_post_physics_substep_callback = lambda *_: None
    raw_env.physical_replay_post_substep_observer_is_safe = lambda _callback: "true"

    with pytest.raises(RuntimeError, match="post_physics_substep_callback"):
        run_articulated_drive_control_step(
            raw_env,
            _FakeTrace(),
            _schema(),
            drive_step,
        )


def test_guard_terminal_control_step_executes_count_with_full_capacity() -> None:
    drive_step = RecordedArticulatedDrivePath(_schema()).control_step(
        _guard_partial_snapshot(),
        allow_partial_terminal_guard=True,
    )
    raw_env = _FakeEnv(
        substep_capacity=4,
        break_after_substeps=2,
    )
    trace = _FakeTrace()

    result, report = run_articulated_drive_control_step(
        raw_env,
        trace,
        _schema(),
        drive_step,
    )

    assert result == ("obs", 0.0, False, False, {"succeed": False})
    assert len(trace.snapshots) == 2
    assert report["recorded_substep_capacity"] == 4
    assert report["recorded_substep_count"] == 2
    assert report["applied_robot_joint_target_count"] == 6

    underrun_env = _FakeEnv(
        substep_capacity=4,
        break_after_substeps=1,
    )
    with pytest.raises(RuntimeError, match="unexpected number of substeps"):
        run_articulated_drive_control_step(
            underrun_env,
            _FakeTrace(),
            _schema(),
            drive_step,
        )

    overrun_env = _FakeEnv(substep_capacity=4)
    with pytest.raises(RuntimeError, match="exceeded.*terminal-guard"):
        run_articulated_drive_control_step(
            overrun_env,
            _FakeTrace(),
            _schema(),
            drive_step,
        )
    assert not hasattr(
        overrun_env,
        "_handson_pre_physics_substep_callback",
    )


def test_control_step_rejects_callback_conflicts_and_substep_mismatch() -> None:
    drive_step = RecordedArticulatedDrivePath(_schema()).control_step(
        _snapshot()
    )
    trace = _FakeTrace()
    conflict_env = _FakeEnv()
    conflict_env._handson_post_physics_substep_callback = lambda *_: None
    with pytest.raises(RuntimeError, match="cannot share"):
        run_articulated_drive_control_step(
            conflict_env,
            trace,
            _schema(),
            drive_step,
        )

    mismatch_env = _FakeEnv(substep_capacity=2)
    with pytest.raises(RuntimeError, match="substep capacity differs"):
        run_articulated_drive_control_step(
            mismatch_env,
            trace,
            _schema(),
            drive_step,
        )
    assert not hasattr(
        mismatch_env,
        "_handson_pre_physics_substep_callback",
    )


def test_physics_configuration_must_match_recording() -> None:
    report = validate_recorded_physics_configuration(_schema(), _schema())
    assert report["matched"] is True
    assert "solver.solver_type" in report["checked_fields"]

    changed = _schema()
    changed["environment_schema"]["physx"]["physics_config"][
        "physics_dt"
    ] = 1.0 / 60.0
    with pytest.raises(RuntimeError, match="physics_dt"):
        validate_recorded_physics_configuration(_schema(), changed)


@pytest.mark.parametrize(
    "device",
    [
        None,
        "",
        "cpu",
        "cuda",
        "CUDA:0",
        " cuda:0",
        "cuda:0 ",
        "cuda:00",
        "cuda:01",
        "cuda:-1",
        "cuda:²",
    ],
)
@pytest.mark.parametrize("side", ["source", "runtime"])
def test_physics_configuration_rejects_noncanonical_cuda_device(
    device: object,
    side: str,
) -> None:
    source = _schema()
    runtime = _schema()
    target = source if side == "source" else runtime
    target["environment_schema"]["physx"]["physics_config"][
        "device"
    ] = device

    with pytest.raises(
        RuntimeError,
        match=rf"{side} requires canonical cuda:N",
    ):
        validate_recorded_physics_configuration(source, runtime)
