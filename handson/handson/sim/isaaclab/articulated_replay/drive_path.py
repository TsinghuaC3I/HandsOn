"""Auditable replay of recorded targets on the original robot articulation.

The distinction in this module is intentional:

* joint position *targets* are controller inputs and are replayed;
* measured joint position, velocity, and effort are solver outputs and are
  never written during the trajectory.

Consequently the source asset keeps its finite mass, inertia, compliant
drives, collision geometry, and contact response.  The resulting state can
diverge and is measured rather than corrected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from handson.runtime.gpu import is_physx_recording_device_allowed


_TARGET_FIELD = "physx_substep_robot_joint_target"
_VALID_FIELD = "physx_substep_robot_joint_target_valid"
_COUNT_FIELD = "physx_substep_count"
_CONTROL_TARGET_FIELD = "physx_robot_joint_target"
_CONTROL_VALID_FIELD = "physx_robot_joint_target_valid"
_ENDPOINT_TOLERANCE = 1.0e-9


def _flatten_configuration(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for raw_name, item in value.items():
        name = f"{prefix}.{raw_name}" if prefix else str(raw_name)
        if isinstance(item, Mapping):
            flattened.update(
                _flatten_configuration(item, prefix=name)
            )
        else:
            flattened[name] = item
    return flattened


def _physx_schema(recorded_schema: Mapping[str, Any]) -> Mapping[str, Any]:
    environment_schema = recorded_schema.get(
        "environment_schema",
        recorded_schema,
    )
    if not isinstance(environment_schema, Mapping):
        raise RuntimeError("articulated replay has no environment schema")
    physx = environment_schema.get("physx")
    if not isinstance(physx, Mapping):
        raise RuntimeError("articulated replay has no PhysX schema")
    return physx


def _required_array(
    snapshot: Mapping[str, Any],
    name: str,
    *,
    ndim: int,
) -> np.ndarray:
    if name not in snapshot:
        raise RuntimeError(
            f"articulated replay requires recorded field {name}"
        )
    value = np.asarray(snapshot[name])
    if value.ndim != ndim:
        raise RuntimeError(
            f"articulated replay field {name} has shape {value.shape}; "
            f"expected {ndim} dimensions"
        )
    return value


def validate_recorded_physics_configuration(
    recorded_schema: Mapping[str, Any],
    runtime_schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject a replay whose recorded public PhysX configuration changed."""

    recorded_physx = _physx_schema(recorded_schema)
    runtime_physx = _physx_schema(runtime_schema)
    recorded_config = recorded_physx.get("physics_config")
    runtime_config = runtime_physx.get("physics_config")
    if not isinstance(recorded_config, Mapping):
        raise RuntimeError(
            "articulated replay source has no recorded physics_config"
        )
    if not isinstance(runtime_config, Mapping):
        raise RuntimeError(
            "articulated replay runtime has no physics_config"
        )
    for label, physx, config in (
        ("source", recorded_physx, recorded_config),
        ("runtime", runtime_physx, runtime_config),
    ):
        device = config.get("device")
        if not is_physx_recording_device_allowed(
            device,
            cpu_exception=physx.get("cpu_backend_exception"),
        ):
            raise RuntimeError(
                "articulated replay "
                f"{label} requires canonical cuda:N or the exact P0-03 CPU "
                f"contract; got {device!r}"
            )
    recorded_flat = _flatten_configuration(recorded_config)
    runtime_flat = _flatten_configuration(runtime_config)
    mismatches: dict[str, dict[str, Any]] = {}
    for name, recorded_value in recorded_flat.items():
        if name not in runtime_flat:
            mismatches[name] = {
                "recorded": recorded_value,
                "runtime": "<missing>",
            }
            continue
        runtime_value = runtime_flat[name]
        if isinstance(recorded_value, bool) or isinstance(runtime_value, bool):
            matches = type(recorded_value) is type(runtime_value) and (
                recorded_value == runtime_value
            )
        elif isinstance(recorded_value, (int, float)) and isinstance(
            runtime_value,
            (int, float),
        ):
            matches = bool(
                np.isclose(
                    float(recorded_value),
                    float(runtime_value),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
        else:
            matches = recorded_value == runtime_value
        if not matches:
            mismatches[name] = {
                "recorded": recorded_value,
                "runtime": runtime_value,
            }
    if mismatches:
        details = ", ".join(
            f"{name}: source={values['recorded']!r}, "
            f"runtime={values['runtime']!r}"
            for name, values in sorted(mismatches.items())
        )
        raise RuntimeError(
            "articulated replay PhysX configuration differs from source: "
            + details
        )
    return {
        "matched": True,
        "checked_field_count": len(recorded_flat),
        "checked_fields": sorted(recorded_flat),
        "unrecorded_static_dynamics_fingerprint": True,
    }


@dataclass(frozen=True)
class ArticulatedDriveControlStep:
    """Validated position-drive inputs for one control interval."""

    targets: np.ndarray
    valid: np.ndarray
    substep_capacity: int
    substep_count: int
    expected_joint_count: int
    max_intra_step_target_delta: float

    def target_snapshot(self, substep_index: int) -> dict[str, np.ndarray]:
        """Return only the two fields accepted by the drive-target writer."""

        if not 0 <= substep_index < self.substep_count:
            raise IndexError(
                f"articulated replay substep {substep_index} is outside "
                f"[0, {self.substep_count})"
            )
        return {
            "physx_robot_joint_target": self.targets[substep_index],
            "physx_robot_joint_target_valid": self.valid[substep_index],
        }


class RecordedArticulatedDrivePath:
    """Validate and expose named, complete per-substep drive targets."""

    def __init__(self, recorded_schema: Mapping[str, Any]) -> None:
        physx = _physx_schema(recorded_schema)
        raw_sides = physx.get("robot_sides")
        raw_names = physx.get("robot_joint_names")
        if not isinstance(raw_sides, (list, tuple)) or not raw_sides:
            raise RuntimeError(
                "articulated replay schema has no recorded robot sides"
            )
        if not isinstance(raw_names, Mapping):
            raise RuntimeError(
                "articulated replay schema has no recorded robot joint names"
            )

        self.sides = tuple(str(side) for side in raw_sides)
        if len(set(self.sides)) != len(self.sides):
            raise RuntimeError(
                "articulated replay schema has duplicate robot sides"
            )
        joint_names: dict[str, tuple[str, ...]] = {}
        for side in self.sides:
            names_value = raw_names.get(side)
            if not isinstance(names_value, (list, tuple)) or not names_value:
                raise RuntimeError(
                    f"articulated replay schema has no joints for {side}"
                )
            names = tuple(str(name) for name in names_value)
            if len(set(names)) != len(names):
                raise RuntimeError(
                    f"articulated replay schema has duplicate joints for {side}"
                )
            joint_names[side] = names
        self.joint_names = joint_names
        self.expected_joint_count = sum(
            len(names) for names in self.joint_names.values()
        )

    def control_step(
        self,
        after_snapshot: Mapping[str, Any],
        *,
        allow_partial_terminal_guard: bool = False,
    ) -> ArticulatedDriveControlStep:
        """Validate one recorded control interval without reading q/qd/effort.

        Partial intervals require per-event authorization from discarded
        session-history metadata; accepted trajectories stay complete-only.
        """

        targets = _required_array(
            after_snapshot,
            _TARGET_FIELD,
            ndim=3,
        )
        valid_raw = _required_array(
            after_snapshot,
            _VALID_FIELD,
            ndim=3,
        )
        if targets.shape != valid_raw.shape:
            raise RuntimeError(
                "articulated replay target/mask shape mismatch: "
                f"target={targets.shape}, valid={valid_raw.shape}"
            )
        if not np.all(np.isin(valid_raw, (0, 1))):
            raise RuntimeError(
                "articulated replay target-valid mask is not binary"
            )

        count_value = _required_array(
            after_snapshot,
            _COUNT_FIELD,
            ndim=1,
        )
        if count_value.size != 1 or not np.isfinite(count_value[0]):
            raise RuntimeError(
                "articulated replay substep count must be one finite scalar"
            )
        substep_count = int(count_value[0])
        if float(count_value[0]) != float(substep_count) or substep_count <= 0:
            raise RuntimeError(
                "articulated replay substep count must be a positive integer"
            )
        substep_capacity = int(targets.shape[0])
        if substep_count > substep_capacity:
            raise RuntimeError(
                "articulated replay substep count exceeds the recorded "
                f"target capacity: capacity={substep_capacity}, "
                f"count={substep_count}"
            )
        if (
            substep_count < substep_capacity
            and allow_partial_terminal_guard is not True
        ):
            raise RuntimeError(
                "articulated replay partial terminal-guard substeps require "
                "explicit authorization: "
                f"capacity={substep_capacity}, count={substep_count}"
            )
        if targets.shape[1] != len(self.sides):
            raise RuntimeError(
                "articulated replay target side count differs from schema: "
                f"target={targets.shape[1]}, schema={len(self.sides)}"
            )

        active = np.zeros(targets.shape[1:], dtype=bool)
        for side_index, side in enumerate(self.sides):
            joint_count = len(self.joint_names[side])
            if joint_count > targets.shape[2]:
                raise RuntimeError(
                    f"articulated replay target width cannot hold {side}: "
                    f"target={targets.shape[2]}, joints={joint_count}"
                )
            active[side_index, :joint_count] = True
        valid = valid_raw.astype(bool)
        if not np.all(valid[:substep_count, active]):
            raise RuntimeError(
                "articulated replay target coverage is incomplete for one "
                "or more real robot joints"
            )
        if not np.all(np.isfinite(targets[:substep_count, active])):
            raise RuntimeError(
                "articulated replay has a non-finite target for a real "
                "robot joint"
            )
        if np.any(valid_raw[substep_count:] != 0):
            raise RuntimeError(
                "articulated replay unused target-valid tail must be "
                "exactly zero"
            )
        if np.any(targets[substep_count:] != 0):
            raise RuntimeError(
                "articulated replay unused target tail must be exactly zero"
            )

        control_target = _required_array(
            after_snapshot,
            _CONTROL_TARGET_FIELD,
            ndim=2,
        )
        control_valid_raw = _required_array(
            after_snapshot,
            _CONTROL_VALID_FIELD,
            ndim=2,
        )
        if (
            control_target.shape != targets.shape[1:]
            or control_valid_raw.shape != targets.shape[1:]
        ):
            raise RuntimeError(
                "articulated replay control-frame target shape differs from "
                "the substep target layout"
            )
        if not np.all(np.isin(control_valid_raw, (0, 1))):
            raise RuntimeError(
                "articulated replay control target-valid mask is not binary"
            )
        control_valid = control_valid_raw.astype(bool)
        if not np.all(control_valid[active]):
            raise RuntimeError(
                "articulated replay control target coverage is incomplete"
            )
        if not np.all(np.isfinite(control_target[active])):
            raise RuntimeError(
                "articulated replay has a non-finite control-frame target"
            )
        if not np.allclose(
            targets[substep_count - 1][active],
            control_target[active],
            rtol=0.0,
            atol=_ENDPOINT_TOLERANCE,
        ):
            raise RuntimeError(
                "articulated replay final substep target does not match the "
                "control-frame target"
            )

        if substep_count <= 1:
            max_delta = 0.0
        else:
            max_delta = float(
                np.max(
                    np.abs(
                        targets[1:substep_count, active]
                        - targets[: substep_count - 1, active]
                    )
                )
            )
        targets_copy = np.asarray(targets).copy()
        valid_copy = np.asarray(valid_raw).copy()
        targets_copy.setflags(write=False)
        valid_copy.setflags(write=False)
        return ArticulatedDriveControlStep(
            targets=targets_copy,
            valid=valid_copy,
            substep_capacity=substep_capacity,
            substep_count=substep_count,
            expected_joint_count=self.expected_joint_count,
            max_intra_step_target_delta=max_delta,
        )


def _assert_callbacks_available(raw_env: Any) -> None:
    conflicts = []
    if getattr(raw_env, "_handson_pre_physics_substep_callback", None) is not None:
        conflicts.append("_handson_pre_physics_substep_callback")
    post_substep = getattr(
        raw_env,
        "_handson_post_physics_substep_callback",
        None,
    )
    validator = getattr(
        raw_env,
        "physical_replay_post_substep_observer_is_safe",
        None,
    )
    try:
        safe_post_substep = (
            post_substep is not None
            and callable(validator)
            and validator(post_substep) is True
        )
    except Exception:
        safe_post_substep = False
    if post_substep is not None and not safe_post_substep:
        conflicts.append("_handson_post_physics_substep_callback")
    if conflicts:
        raise RuntimeError(
            "articulated replay cannot share PhysX substep callbacks: "
            + ", ".join(conflicts)
        )


def run_articulated_drive_control_step(
    raw_env: Any,
    physx_trace: Any,
    recorded_schema: Mapping[str, Any],
    drive_step: ArticulatedDriveControlStep,
) -> tuple[tuple[Any, Any, Any, Any, Any], dict[str, Any]]:
    """Advance PhysX while writing only the recorded drive input each substep."""

    _assert_callbacks_available(raw_env)
    step_targets = getattr(raw_env, "step_with_current_joint_targets", None)
    if not callable(step_targets):
        raise RuntimeError(
            "Isaac environment does not support articulated drive replay"
        )

    applied_counts: list[int] = []

    def apply_substep_target(
        substep_index: int,
        runtime_substep_count: int,
    ) -> None:
        if runtime_substep_count != drive_step.substep_capacity:
            raise RuntimeError(
                "articulated replay PhysX substep capacity differs from "
                "source: "
                f"runtime={runtime_substep_count}, "
                f"recorded={drive_step.substep_capacity}"
            )
        try:
            target_snapshot = drive_step.target_snapshot(substep_index)
        except IndexError as exc:
            raise RuntimeError(
                "articulated replay runtime exceeded the recorded terminal-"
                "guard substep count: "
                f"index={substep_index}, count={drive_step.substep_count}, "
                f"capacity={drive_step.substep_capacity}"
            ) from exc
        applied = int(
            physx_trace.apply_robot_joint_targets(
                target_snapshot,
                recorded_schema,
            )
        )
        if applied != drive_step.expected_joint_count:
            raise RuntimeError(
                "articulated replay did not apply every recorded robot joint "
                f"target: applied={applied}, "
                f"expected={drive_step.expected_joint_count}"
            )
        applied_counts.append(applied)

    raw_env._handson_pre_physics_substep_callback = apply_substep_target
    try:
        result = step_targets()
    finally:
        try:
            delattr(raw_env, "_handson_pre_physics_substep_callback")
        except AttributeError:
            pass

    if len(applied_counts) != drive_step.substep_count:
        raise RuntimeError(
            "articulated replay environment executed an unexpected number "
            f"of substeps: applied={len(applied_counts)}, "
            f"recorded={drive_step.substep_count}"
        )
    return result, {
        "recorded_substep_capacity": int(drive_step.substep_capacity),
        "recorded_substep_count": int(drive_step.substep_count),
        "applied_robot_joint_target_count": int(sum(applied_counts)),
        "applied_robot_joint_target_count_per_substep": int(
            drive_step.expected_joint_count
        ),
        "max_intra_step_target_delta": float(
            drive_step.max_intra_step_target_delta
        ),
    }
