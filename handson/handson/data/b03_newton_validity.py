"""Fail-closed source-trajectory validity checks for B03 Newton.

The validator in this module is intentionally simulator-independent.  It reads
only the action-aligned Newton ground-truth arrays that are already captured at
every physics substep.  It never repairs a trajectory, changes a drive target,
or writes simulator state.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


B03_NEWTON_HAND_MAX_ABS_JOINT_VELOCITY_RAD_S = 4.0
B03_NEWTON_HAND_PEG_MAX_SOLID_PENETRATION_M = 1.0e-3

B03_NEWTON_SOURCE_VALIDITY_GATE_FORMAT = (
    "handson.b03_newton_source_validity_gate"
)
B03_NEWTON_SOURCE_VALIDITY_GATE_VERSION = 1
B03_NEWTON_SOURCE_VALIDITY_REPORT_FORMAT = (
    "handson.b03_newton_source_physical_validity"
)
B03_NEWTON_SOURCE_VALIDITY_REPORT_VERSION = 1

_EXPECTED_ROBOT_SIDES = ("right", "left")
_EXPECTED_HAND_JOINT_COUNT_PER_SIDE = 20
_EXPECTED_GATE_CONTRACT: dict[str, Any] = {
    "format": B03_NEWTON_SOURCE_VALIDITY_GATE_FORMAT,
    "version": B03_NEWTON_SOURCE_VALIDITY_GATE_VERSION,
    "evaluation": "every_recorded_newton_physics_substep",
    "hand_joint_selection": "schema_robot_joint_names_finger_prefix",
    "hand_max_abs_joint_velocity_rad_s": (
        B03_NEWTON_HAND_MAX_ABS_JOINT_VELOCITY_RAD_S
    ),
    "hand_peg_max_solid_penetration_m": (
        B03_NEWTON_HAND_PEG_MAX_SOLID_PENETRATION_M
    ),
    "missing_or_nonfinite_policy": "invalid",
    "latching": "first_violation_invalidates_episode",
    "simulator_state_write": "none_readback_only",
}


def b03_newton_source_validity_gate_contract() -> dict[str, Any]:
    """Return a fresh copy of the immutable source-validity contract."""

    return dict(_EXPECTED_GATE_CONTRACT)


def _base_report() -> dict[str, Any]:
    return {
        "format": B03_NEWTON_SOURCE_VALIDITY_REPORT_FORMAT,
        "version": B03_NEWTON_SOURCE_VALIDITY_REPORT_VERSION,
        "status": "invalid",
        "valid": False,
        "code": "not_evaluated",
        "detail": "B03 Newton source validity was not evaluated",
        "evaluated_control_steps": 0,
        "evaluated_physics_substeps": 0,
        "hand_max_abs_joint_velocity_rad_s": None,
        "hand_max_abs_joint_velocity_limit_rad_s": (
            B03_NEWTON_HAND_MAX_ABS_JOINT_VELOCITY_RAD_S
        ),
        "hand_peg_max_solid_penetration_m": None,
        "hand_peg_max_solid_penetration_limit_m": (
            B03_NEWTON_HAND_PEG_MAX_SOLID_PENETRATION_M
        ),
        "hand_joint_readback_complete": False,
        "hand_peg_contact_readback_complete": False,
        "first_violation": None,
    }


def _invalid(
    report: dict[str, Any],
    code: str,
    detail: str,
    *,
    first_violation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report["status"] = "invalid"
    report["valid"] = False
    report["code"] = str(code)
    report["detail"] = str(detail)
    report["first_violation"] = (
        None if first_violation is None else dict(first_violation)
    )
    return report


def _valid(report: dict[str, Any]) -> dict[str, Any]:
    report["status"] = "valid"
    report["valid"] = True
    report["code"] = "ok"
    report["detail"] = (
        "Every recorded B03 Newton physics substep satisfies the hand-speed "
        "and hand/peg solid-penetration bounds"
    )
    report["first_violation"] = None
    return report


def _schema_layout(
    environment_schema: Mapping[str, Any],
) -> tuple[int, tuple[str, ...], dict[str, tuple[str, ...]], int]:
    if environment_schema.get("name") != "handson.b03_newton_ground_truth":
        raise ValueError("environment schema is not B03 Newton ground truth")
    newton = environment_schema.get("newton")
    if not isinstance(newton, Mapping):
        raise ValueError("environment_schema.newton is missing")

    gate = newton.get("source_validity_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("environment_schema.newton.source_validity_gate is missing")
    for key, expected in _EXPECTED_GATE_CONTRACT.items():
        if gate.get(key) != expected:
            raise ValueError(
                "environment_schema.newton.source_validity_gate."
                f"{key} must be {expected!r}"
            )

    raw_substeps = newton.get("n_substeps")
    if (
        type(raw_substeps) is not int
        or isinstance(raw_substeps, bool)
        or raw_substeps <= 0
    ):
        raise ValueError("environment_schema.newton.n_substeps is invalid")
    substeps = int(raw_substeps)

    raw_sides = newton.get("robot_sides")
    if not isinstance(raw_sides, list):
        raise ValueError("environment_schema.newton.robot_sides is missing")
    sides = tuple(str(side) for side in raw_sides)
    if sides != _EXPECTED_ROBOT_SIDES:
        raise ValueError(
            "environment_schema.newton.robot_sides must be right_then_left"
        )

    raw_joint_names = newton.get("robot_joint_names")
    if not isinstance(raw_joint_names, Mapping):
        raise ValueError("environment_schema.newton.robot_joint_names is missing")
    joint_names: dict[str, tuple[str, ...]] = {}
    for side in sides:
        raw_names = raw_joint_names.get(side)
        if not isinstance(raw_names, list) or not raw_names:
            raise ValueError(f"Newton joint names are missing for {side}")
        names = tuple(str(name) for name in raw_names)
        if len(names) != len(set(names)):
            raise ValueError(f"Newton joint names are not unique for {side}")
        hand_names = tuple(
            name for name in names if name.startswith(f"{side}_finger")
        )
        if len(hand_names) != _EXPECTED_HAND_JOINT_COUNT_PER_SIDE:
            raise ValueError(
                f"Newton {side} hand layout has {len(hand_names)} finger "
                f"joints; expected {_EXPECTED_HAND_JOINT_COUNT_PER_SIDE}"
            )
        joint_names[side] = names
    return substeps, sides, joint_names, max(map(len, joint_names.values()))


def _array(
    arrays: Mapping[str, Any],
    name: str,
    *,
    dtype: Any | None = None,
) -> np.ndarray:
    if name not in arrays:
        raise KeyError(name)
    value = np.asarray(arrays[name], dtype=dtype)
    return value


def evaluate_b03_newton_episode_arrays(
    arrays: Mapping[str, Any],
    environment_schema: Mapping[str, Any],
    *,
    field_prefix: str = "next_",
) -> dict[str, Any]:
    """Evaluate a batch of complete B03 Newton control transitions.

    ``arrays`` uses the recorder/Zarr layout ``(T, ...)``.  Only the post-action
    ``next_*`` trace is authoritative: the first current observation is the
    reset boundary and intentionally has zero captured action substeps.
    """

    report = _base_report()
    try:
        substeps, sides, joint_names, max_joint_count = _schema_layout(
            environment_schema
        )
    except (TypeError, ValueError) as exc:
        return _invalid(report, "invalid_schema_contract", str(exc))

    names = {
        logical: f"{field_prefix}{field}"
        for logical, field in {
            "count": "newton_substep_count",
            "velocity": "newton_substep_robot_joint_velocity",
            "joint_valid": "newton_substep_robot_joint_valid",
            "contact_valid": "newton_substep_hand_peg_contact_valid",
            "penetration": (
                "newton_substep_hand_peg_max_solid_penetration"
            ),
        }.items()
    }
    try:
        counts = _array(arrays, names["count"])
        velocity = _array(arrays, names["velocity"], dtype=np.float64)
        joint_valid = _array(arrays, names["joint_valid"])
        contact_valid = _array(arrays, names["contact_valid"])
        penetration = _array(
            arrays,
            names["penetration"],
            dtype=np.float64,
        )
    except KeyError as exc:
        return _invalid(
            report,
            "missing_ground_truth_field",
            f"required B03 Newton field is missing: {exc.args[0]}",
        )
    except (TypeError, ValueError) as exc:
        return _invalid(
            report,
            "unreadable_ground_truth_field",
            f"B03 Newton ground-truth array cannot be read: {exc}",
        )

    if counts.ndim != 2 or counts.shape[1:] != (1,):
        return _invalid(
            report,
            "invalid_substep_count_shape",
            f"{names['count']} has shape {counts.shape}; expected (T, 1)",
        )
    control_steps = int(counts.shape[0])
    report["evaluated_control_steps"] = control_steps
    if control_steps <= 0:
        return _invalid(report, "empty_episode", "B03 Newton episode is empty")

    expected_velocity_shape = (
        control_steps,
        substeps,
        len(sides),
        max_joint_count,
    )
    expected_contact_shape = (control_steps, substeps, 1)
    for logical, value, expected in (
        ("velocity", velocity, expected_velocity_shape),
        ("joint_valid", joint_valid, expected_velocity_shape),
        ("contact_valid", contact_valid, expected_contact_shape),
        ("penetration", penetration, expected_contact_shape),
    ):
        if value.shape != expected:
            return _invalid(
                report,
                "invalid_ground_truth_shape",
                f"{names[logical]} has shape {value.shape}; expected {expected}",
            )

    if counts.dtype.kind not in "iu" or np.any(counts[:, 0] != substeps):
        first = int(np.flatnonzero(counts[:, 0] != substeps)[0]) if np.any(
            counts[:, 0] != substeps
        ) else 0
        value = counts[first, 0]
        return _invalid(
            report,
            "incomplete_substep_trace",
            f"control step {first} records {value!r} substeps; expected {substeps}",
            first_violation={
                "kind": "substep_count",
                "control_step": first,
                "value": int(value) if np.isscalar(value) else str(value),
                "limit": substeps,
            },
        )
    report["evaluated_physics_substeps"] = control_steps * substeps

    hand_rows: list[tuple[int, int, str, str]] = []
    for side_index, side in enumerate(sides):
        for joint_index, joint_name in enumerate(joint_names[side]):
            if joint_name.startswith(f"{side}_finger"):
                hand_rows.append(
                    (side_index, joint_index, side, joint_name)
                )

    for side_index, joint_index, side, joint_name in hand_rows:
        valid_values = joint_valid[:, :, side_index, joint_index]
        bad = np.argwhere(valid_values != 1)
        if bad.size:
            control_index, substep_index = map(int, bad[0])
            return _invalid(
                report,
                "hand_joint_readback_invalid",
                f"{side} hand joint {joint_name} has no valid readback at "
                f"control step {control_index}, substep {substep_index}",
                first_violation={
                    "kind": "hand_joint_readback",
                    "control_step": control_index,
                    "physics_substep": substep_index,
                    "side": side,
                    "joint_name": joint_name,
                    "value": int(valid_values[control_index, substep_index]),
                    "limit": 1,
                },
            )
    report["hand_joint_readback_complete"] = True

    hand_velocity = np.stack(
        [
            velocity[:, :, side_index, joint_index]
            for side_index, joint_index, _side, _joint_name in hand_rows
        ],
        axis=-1,
    )
    finite = np.isfinite(hand_velocity)
    if not np.all(finite):
        control_index, substep_index, hand_index = map(
            int,
            np.argwhere(~finite)[0],
        )
        _side_index, _joint_index, side, joint_name = hand_rows[hand_index]
        return _invalid(
            report,
            "hand_joint_velocity_nonfinite",
            f"{side} hand joint {joint_name} has a non-finite velocity at "
            f"control step {control_index}, substep {substep_index}",
            first_violation={
                "kind": "hand_joint_velocity_nonfinite",
                "control_step": control_index,
                "physics_substep": substep_index,
                "side": side,
                "joint_name": joint_name,
                "value": str(hand_velocity[control_index, substep_index, hand_index]),
                "limit": B03_NEWTON_HAND_MAX_ABS_JOINT_VELOCITY_RAD_S,
            },
        )
    absolute_hand_velocity = np.abs(hand_velocity)
    maximum_hand_velocity = float(np.max(absolute_hand_velocity))
    report["hand_max_abs_joint_velocity_rad_s"] = maximum_hand_velocity
    velocity_violations = np.argwhere(
        absolute_hand_velocity
        > B03_NEWTON_HAND_MAX_ABS_JOINT_VELOCITY_RAD_S
    )
    if velocity_violations.size:
        control_index, substep_index, hand_index = map(
            int,
            velocity_violations[0],
        )
        _side_index, _joint_index, side, joint_name = hand_rows[hand_index]
        value = float(
            absolute_hand_velocity[control_index, substep_index, hand_index]
        )
        return _invalid(
            report,
            "hand_joint_velocity_exceeded",
            f"{side} hand joint {joint_name} reached {value:.9g} rad/s at "
            f"control step {control_index}, substep {substep_index}; limit is "
            f"{B03_NEWTON_HAND_MAX_ABS_JOINT_VELOCITY_RAD_S:.9g} rad/s",
            first_violation={
                "kind": "hand_joint_velocity",
                "control_step": control_index,
                "physics_substep": substep_index,
                "side": side,
                "joint_name": joint_name,
                "value": value,
                "limit": B03_NEWTON_HAND_MAX_ABS_JOINT_VELOCITY_RAD_S,
            },
        )

    bad_contact = np.argwhere(contact_valid[:, :, 0] != 1)
    if bad_contact.size:
        control_index, substep_index = map(int, bad_contact[0])
        return _invalid(
            report,
            "hand_peg_contact_readback_invalid",
            "hand/peg contact buffers are unreadable at control step "
            f"{control_index}, substep {substep_index}",
            first_violation={
                "kind": "hand_peg_contact_readback",
                "control_step": control_index,
                "physics_substep": substep_index,
                "value": int(contact_valid[control_index, substep_index, 0]),
                "limit": 1,
            },
        )
    report["hand_peg_contact_readback_complete"] = True

    penetration_values = penetration[:, :, 0]
    finite_penetration = np.isfinite(penetration_values)
    if not np.all(finite_penetration):
        control_index, substep_index = map(
            int,
            np.argwhere(~finite_penetration)[0],
        )
        return _invalid(
            report,
            "hand_peg_penetration_nonfinite",
            "hand/peg solid penetration is non-finite at control step "
            f"{control_index}, substep {substep_index}",
            first_violation={
                "kind": "hand_peg_solid_penetration_nonfinite",
                "control_step": control_index,
                "physics_substep": substep_index,
                "value": str(penetration_values[control_index, substep_index]),
                "limit": B03_NEWTON_HAND_PEG_MAX_SOLID_PENETRATION_M,
            },
        )
    negative_penetration = np.argwhere(penetration_values < 0.0)
    if negative_penetration.size:
        control_index, substep_index = map(int, negative_penetration[0])
        value = float(penetration_values[control_index, substep_index])
        return _invalid(
            report,
            "hand_peg_penetration_negative",
            "hand/peg solid penetration is negative at control step "
            f"{control_index}, substep {substep_index}: {value:.9g} m",
            first_violation={
                "kind": "hand_peg_solid_penetration_negative",
                "control_step": control_index,
                "physics_substep": substep_index,
                "value": value,
                "limit": 0.0,
            },
        )
    maximum_penetration = float(np.max(penetration_values))
    report["hand_peg_max_solid_penetration_m"] = maximum_penetration
    penetration_violations = np.argwhere(
        penetration_values > B03_NEWTON_HAND_PEG_MAX_SOLID_PENETRATION_M
    )
    if penetration_violations.size:
        control_index, substep_index = map(int, penetration_violations[0])
        value = float(penetration_values[control_index, substep_index])
        return _invalid(
            report,
            "hand_peg_solid_penetration_exceeded",
            f"hand/peg solid penetration reached {value:.9g} m at control "
            f"step {control_index}, substep {substep_index}; limit is "
            f"{B03_NEWTON_HAND_PEG_MAX_SOLID_PENETRATION_M:.9g} m",
            first_violation={
                "kind": "hand_peg_solid_penetration",
                "control_step": control_index,
                "physics_substep": substep_index,
                "value": value,
                "limit": B03_NEWTON_HAND_PEG_MAX_SOLID_PENETRATION_M,
            },
        )

    return _valid(report)


def evaluate_b03_newton_substep_observation(
    observation: Mapping[str, Any],
    environment_schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the complete substep trace returned by one ``env.step``."""

    batched: dict[str, np.ndarray] = {}
    for name in (
        "newton_substep_count",
        "newton_substep_robot_joint_velocity",
        "newton_substep_robot_joint_valid",
        "newton_substep_hand_peg_contact_valid",
        "newton_substep_hand_peg_max_solid_penetration",
    ):
        if name in observation:
            batched[name] = np.expand_dims(np.asarray(observation[name]), axis=0)
    return evaluate_b03_newton_episode_arrays(
        batched,
        environment_schema,
        field_prefix="",
    )


def b03_newton_source_validity_error(report: Mapping[str, Any]) -> str | None:
    """Return a concise error for an invalid/malformed validity report."""

    if report.get("valid") is True and report.get("status") == "valid":
        return None
    code = str(report.get("code") or "invalid_source_physics")
    detail = str(report.get("detail") or "B03 Newton source physics is invalid")
    return f"{code}: {detail}"


__all__ = [
    "B03_NEWTON_HAND_MAX_ABS_JOINT_VELOCITY_RAD_S",
    "B03_NEWTON_HAND_PEG_MAX_SOLID_PENETRATION_M",
    "B03_NEWTON_SOURCE_VALIDITY_GATE_FORMAT",
    "B03_NEWTON_SOURCE_VALIDITY_GATE_VERSION",
    "B03_NEWTON_SOURCE_VALIDITY_REPORT_FORMAT",
    "B03_NEWTON_SOURCE_VALIDITY_REPORT_VERSION",
    "b03_newton_source_validity_error",
    "b03_newton_source_validity_gate_contract",
    "evaluate_b03_newton_episode_arrays",
    "evaluate_b03_newton_substep_observation",
]
