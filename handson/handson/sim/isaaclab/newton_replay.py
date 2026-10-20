"""Reusable fresh-scene NVIDIA Newton drive-target replay primitives.

The live trajectory writes controller targets only.  Measured joint/task state
is always solver output and is compared after stepping; it is never restored.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NewtonDriveControlStep:
    targets: np.ndarray
    valid: np.ndarray
    substep_count: int
    expected_joint_count: int

    def target_snapshot(self, substep_index: int) -> dict[str, np.ndarray]:
        if not 0 <= substep_index < self.substep_count:
            raise IndexError(
                f"Newton replay substep {substep_index} is outside "
                f"[0, {self.substep_count})"
            )
        return {
            "newton_robot_joint_target": self.targets[substep_index],
            "newton_robot_joint_target_valid": self.valid[substep_index],
        }


class RecordedNewtonDrivePath:
    """Validate complete per-substep Newton position-drive targets by name."""

    def __init__(self, recorded_schema: Mapping[str, Any]) -> None:
        environment_schema = recorded_schema.get(
            "environment_schema",
            recorded_schema,
        )
        if not isinstance(environment_schema, Mapping):
            raise RuntimeError("Newton replay has no environment schema")
        newton = environment_schema.get("newton")
        if not isinstance(newton, Mapping):
            raise RuntimeError("Newton replay has no Newton schema")
        sides = newton.get("robot_sides")
        joint_names = newton.get("robot_joint_names")
        if not isinstance(sides, (list, tuple)) or not sides:
            raise RuntimeError("Newton replay schema has no robot sides")
        if not isinstance(joint_names, Mapping):
            raise RuntimeError("Newton replay schema has no robot joint names")
        self.sides = tuple(str(side) for side in sides)
        if len(set(self.sides)) != len(self.sides):
            raise RuntimeError("Newton replay schema has duplicate robot sides")
        self.joint_names: dict[str, tuple[str, ...]] = {}
        for side in self.sides:
            values = joint_names.get(side)
            if not isinstance(values, (list, tuple)) or not values:
                raise RuntimeError(
                    f"Newton replay schema has no joints for {side}"
                )
            names = tuple(str(name) for name in values)
            if len(set(names)) != len(names):
                raise RuntimeError(
                    f"Newton replay schema has duplicate joints for {side}"
                )
            self.joint_names[side] = names
        self.expected_joint_count = sum(
            len(names) for names in self.joint_names.values()
        )

    @staticmethod
    def _array(
        snapshot: Mapping[str, Any],
        name: str,
        *,
        ndim: int,
    ) -> np.ndarray:
        if name not in snapshot:
            raise RuntimeError(f"Newton replay requires recorded field {name}")
        value = np.asarray(snapshot[name])
        if value.ndim != ndim:
            raise RuntimeError(
                f"Newton replay field {name} has shape {value.shape}; "
                f"expected {ndim} dimensions"
            )
        return value

    def control_step(
        self,
        after_snapshot: Mapping[str, Any],
    ) -> NewtonDriveControlStep:
        targets = self._array(
            after_snapshot,
            "newton_substep_robot_joint_target",
            ndim=3,
        )
        valid_raw = self._array(
            after_snapshot,
            "newton_substep_robot_joint_target_valid",
            ndim=3,
        )
        count_value = self._array(
            after_snapshot,
            "newton_substep_count",
            ndim=1,
        )
        if targets.shape != valid_raw.shape:
            raise RuntimeError(
                "Newton replay target/mask shape mismatch: "
                f"target={targets.shape}, valid={valid_raw.shape}"
            )
        if not np.all(np.isin(valid_raw, (0, 1))):
            raise RuntimeError("Newton replay target-valid mask is not binary")
        if count_value.size != 1 or not np.isfinite(count_value[0]):
            raise RuntimeError("Newton replay substep count must be finite")
        substep_count = int(count_value[0])
        if substep_count <= 0 or float(count_value[0]) != float(substep_count):
            raise RuntimeError(
                "Newton replay substep count must be a positive integer"
            )
        if targets.shape[0] != substep_count:
            raise RuntimeError(
                "Newton replay target rows differ from substep count: "
                f"target={targets.shape[0]}, count={substep_count}"
            )
        if targets.shape[1] != len(self.sides):
            raise RuntimeError(
                "Newton replay target side count differs from schema"
            )

        active = np.zeros(targets.shape[1:], dtype=bool)
        for side_index, side in enumerate(self.sides):
            joint_count = len(self.joint_names[side])
            if joint_count > targets.shape[2]:
                raise RuntimeError(
                    f"Newton replay target width cannot hold {side}"
                )
            active[side_index, :joint_count] = True
        valid = valid_raw.astype(bool)
        if not np.all(valid[:, active]):
            raise RuntimeError(
                "Newton replay target coverage is incomplete for real joints"
            )
        if not np.all(np.isfinite(targets[:, active])):
            raise RuntimeError("Newton replay has non-finite joint targets")

        control_targets = self._array(
            after_snapshot,
            "newton_robot_joint_target",
            ndim=2,
        )
        control_valid = self._array(
            after_snapshot,
            "newton_robot_joint_target_valid",
            ndim=2,
        )
        if (
            control_targets.shape != targets.shape[1:]
            or control_valid.shape != targets.shape[1:]
        ):
            raise RuntimeError(
                "Newton replay control target layout differs from substeps"
            )
        if not np.allclose(
            targets[-1][active],
            control_targets[active],
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise RuntimeError(
                "Newton replay final substep target differs from control target"
            )

        target_copy = np.asarray(targets).copy()
        valid_copy = np.asarray(valid_raw).copy()
        target_copy.setflags(write=False)
        valid_copy.setflags(write=False)
        return NewtonDriveControlStep(
            targets=target_copy,
            valid=valid_copy,
            substep_count=substep_count,
            expected_joint_count=self.expected_joint_count,
        )


def run_newton_drive_control_step(
    raw_env: Any,
    trace: Any,
    recorded_schema: Mapping[str, Any],
    drive_step: NewtonDriveControlStep,
) -> tuple[Any, Any, Any, Any, Any]:
    """Advance one control frame, writing only recorded drive targets."""

    callback_name = "_handson_pre_physics_substep_callback"
    if getattr(raw_env, callback_name, None) is not None:
        raise RuntimeError("Newton replay cannot share a pre-substep callback")
    step_targets = getattr(raw_env, "step_with_current_joint_targets", None)
    if not callable(step_targets):
        raise RuntimeError(
            "Newton environment does not support drive-target replay"
        )
    applied_counts: list[int] = []

    def apply_target(substep_index: int, runtime_count: int) -> None:
        if runtime_count != drive_step.substep_count:
            raise RuntimeError(
                "Newton replay substep count differs from source: "
                f"runtime={runtime_count}, source={drive_step.substep_count}"
            )
        applied = int(
            trace.apply_robot_joint_targets(
                drive_step.target_snapshot(substep_index),
                recorded_schema,
            )
        )
        if applied != drive_step.expected_joint_count:
            raise RuntimeError(
                "Newton replay did not apply every robot target: "
                f"applied={applied}, expected={drive_step.expected_joint_count}"
            )
        applied_counts.append(applied)

    setattr(raw_env, callback_name, apply_target)
    try:
        result = step_targets()
    finally:
        try:
            delattr(raw_env, callback_name)
        except AttributeError:
            pass
    if len(applied_counts) != drive_step.substep_count:
        raise RuntimeError(
            "Newton replay did not execute every recorded substep"
        )
    return result


def destroy_newton_scene(env: Any) -> dict[str, bool]:
    """Destroy Newton's singleton scene while leaving Kit closable afterward.

    ``SimulationApp.close()`` may terminate the process inside Kit, so callers
    must first run this function, persist their destruction report, and only
    then call :func:`close_newton_application`.
    """

    raw_env = getattr(env, "unwrapped", env)
    scene_stop_error = False
    try:
        unsubscribe = getattr(raw_env, "_unsubscribe_scene_input_callbacks", None)
        if callable(unsubscribe):
            unsubscribe()
        sim = getattr(raw_env, "_sim", None)
        clear_callbacks = getattr(sim, "clear_all_callbacks", None)
        if callable(clear_callbacks):
            clear_callbacks()
        stop = getattr(sim, "stop", None)
        if callable(stop):
            stop()
    except Exception:
        scene_stop_error = True
    manager_cleared = False
    simulation_context_cleared = False
    try:
        from isaaclab_newton.physics.newton_manager import NewtonManager

        NewtonManager.clear()
        manager_cleared = (
            getattr(NewtonManager, "_model", None) is None
            and getattr(NewtonManager, "_solver", None) is None
            and getattr(NewtonManager, "_state_0", None) is None
        )
    except Exception:
        manager_cleared = False
    try:
        from isaaclab.sim import SimulationContext

        SimulationContext.clear_instance()
        simulation_context_cleared = True
    except Exception:
        simulation_context_cleared = False
    for name in (
        "_stage",
        "_scene",
        "_sim",
        "_physx_ground_truth_trace",
    ):
        try:
            setattr(raw_env, name, None)
        except Exception:
            pass
    if scene_stop_error or not manager_cleared or not simulation_context_cleared:
        raise RuntimeError(
            "Newton replay scene destruction did not release every runtime "
            f"owner: stop_error={scene_stop_error}, manager={manager_cleared}, "
            f"simulation_context={simulation_context_cleared}"
        )
    return {
        "simulation_stopped": True,
        "newton_manager_cleared": True,
        "simulation_context_cleared": True,
        "scene_destroyed": True,
    }


def close_newton_application(env: Any) -> None:
    """Close Kit only after the scene-destruction report is durable."""

    raw_env = getattr(env, "unwrapped", env)
    simulation_app = getattr(raw_env, "_simulation_app", None)
    setattr(raw_env, "_simulation_app", None)
    setattr(raw_env, "_app_launcher", None)
    if simulation_app is None:
        return
    try:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=False)
    except TypeError:
        try:
            simulation_app.close(wait_for_replicator=False)
        except TypeError:
            try:
                simulation_app.close()
            except SystemExit:
                pass
        except SystemExit:
            pass
    except SystemExit:
        # Kit requests interpreter exit after shutdown.  The command-line
        # wrapper must still return its own pass/fail status to automation.
        pass
