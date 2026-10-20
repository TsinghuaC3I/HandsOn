from __future__ import annotations

import sys
import types

import numpy as np

from handson.sim.isaaclab.newton_replay import (
    RecordedNewtonDrivePath,
    destroy_newton_scene,
    run_newton_drive_control_step,
)


def _schema() -> dict:
    return {
        "environment_schema": {
            "newton": {
                "robot_sides": ["right", "left"],
                "robot_joint_names": {
                    "right": ["r0", "r1"],
                    "left": ["l0", "l1"],
                },
            }
        }
    }


def _snapshot() -> dict[str, np.ndarray]:
    targets = np.asarray(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
        ],
        dtype=np.float64,
    )
    return {
        "newton_substep_robot_joint_target": targets,
        "newton_substep_robot_joint_target_valid": np.ones_like(
            targets,
            dtype=np.uint8,
        ),
        "newton_substep_count": np.asarray([2], dtype=np.int64),
        "newton_robot_joint_target": targets[-1],
        "newton_robot_joint_target_valid": np.ones(
            (2, 2),
            dtype=np.uint8,
        ),
    }


def test_newton_drive_path_requires_complete_per_substep_targets() -> None:
    step = RecordedNewtonDrivePath(_schema()).control_step(_snapshot())
    assert step.substep_count == 2
    assert step.expected_joint_count == 4
    assert np.array_equal(
        step.target_snapshot(1)["newton_robot_joint_target"],
        _snapshot()["newton_robot_joint_target"],
    )


def test_newton_drive_replay_writes_targets_only_at_substep_callback() -> None:
    step = RecordedNewtonDrivePath(_schema()).control_step(_snapshot())

    class Trace:
        snapshots: list[dict[str, np.ndarray]] = []

        def apply_robot_joint_targets(self, snapshot, schema):
            assert schema == _schema()
            self.snapshots.append(snapshot)
            return 4

    class Env:
        def step_with_current_joint_targets(self):
            for index in range(2):
                self._handson_pre_physics_substep_callback(index, 2)
            return ({}, 0.0, False, False, {})

    env = Env()
    trace = Trace()
    result = run_newton_drive_control_step(env, trace, _schema(), step)
    assert result[1:] == (0.0, False, False, {})
    assert len(trace.snapshots) == 2
    assert not hasattr(env, "_handson_pre_physics_substep_callback")


def test_destroy_newton_scene_releases_every_runtime_owner(monkeypatch) -> None:
    class Manager:
        _model = object()
        _solver = object()
        _state_0 = object()

        @classmethod
        def clear(cls) -> None:
            cls._model = None
            cls._solver = None
            cls._state_0 = None

    class SimulationContext:
        cleared = False

        @classmethod
        def clear_instance(cls) -> None:
            cls.cleared = True

    newton_manager = types.ModuleType(
        "isaaclab_newton.physics.newton_manager"
    )
    newton_manager.NewtonManager = Manager
    newton_physics = types.ModuleType("isaaclab_newton.physics")
    newton_physics.__path__ = []
    isaaclab_newton = types.ModuleType("isaaclab_newton")
    isaaclab_newton.__path__ = []
    isaaclab_sim = types.ModuleType("isaaclab.sim")
    isaaclab_sim.SimulationContext = SimulationContext
    isaaclab = types.ModuleType("isaaclab")
    isaaclab.__path__ = []
    monkeypatch.setitem(sys.modules, "isaaclab_newton", isaaclab_newton)
    monkeypatch.setitem(sys.modules, "isaaclab_newton.physics", newton_physics)
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_newton.physics.newton_manager",
        newton_manager,
    )
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", isaaclab_sim)

    class Sim:
        callbacks_cleared = False
        stopped = False

        def clear_all_callbacks(self) -> None:
            self.callbacks_cleared = True

        def stop(self) -> None:
            self.stopped = True

    sim = Sim()
    raw = types.SimpleNamespace(
        _sim=sim,
        _stage=object(),
        _scene=object(),
        _physx_ground_truth_trace=object(),
    )
    environment = types.SimpleNamespace(unwrapped=raw)

    report = destroy_newton_scene(environment)

    assert report == {
        "simulation_stopped": True,
        "newton_manager_cleared": True,
        "simulation_context_cleared": True,
        "scene_destroyed": True,
    }
    assert sim.callbacks_cleared is True
    assert sim.stopped is True
    assert SimulationContext.cleared is True
    assert raw._sim is None
    assert raw._stage is None
    assert raw._scene is None
    assert raw._physx_ground_truth_trace is None
