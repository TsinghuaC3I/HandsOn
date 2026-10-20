from __future__ import annotations

import copy

import numpy as np
import pytest

from handson.data.b03_newton_validity import (
    B03_NEWTON_HAND_MAX_ABS_JOINT_VELOCITY_RAD_S,
    B03_NEWTON_HAND_PEG_MAX_SOLID_PENETRATION_M,
    b03_newton_source_validity_error,
    b03_newton_source_validity_gate_contract,
    evaluate_b03_newton_episode_arrays,
    evaluate_b03_newton_substep_observation,
)


def _joint_names(side: str) -> list[str]:
    return [
        *(f"joint{index}_{side}_1" for index in range(1, 8)),
        *(
            f"{side}_finger{finger}_joint{joint}"
            for finger in range(1, 6)
            for joint in range(1, 5)
        ),
    ]


def _schema(*, substeps: int = 4) -> dict:
    return {
        "name": "handson.b03_newton_ground_truth",
        "version": 3,
        "newton": {
            "n_substeps": substeps,
            "robot_sides": ["right", "left"],
            "robot_joint_names": {
                "right": _joint_names("right"),
                "left": _joint_names("left"),
            },
            "source_validity_gate": b03_newton_source_validity_gate_contract(),
        },
    }


def _episode_arrays(
    *,
    control_steps: int = 3,
    substeps: int = 4,
    prefix: str = "next_",
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}newton_substep_count": np.full(
            (control_steps, 1),
            substeps,
            dtype=np.int64,
        ),
        f"{prefix}newton_substep_robot_joint_velocity": np.zeros(
            (control_steps, substeps, 2, 27),
            dtype=np.float64,
        ),
        f"{prefix}newton_substep_robot_joint_valid": np.ones(
            (control_steps, substeps, 2, 27),
            dtype=np.uint8,
        ),
        f"{prefix}newton_substep_hand_peg_contact_valid": np.ones(
            (control_steps, substeps, 1),
            dtype=np.uint8,
        ),
        f"{prefix}newton_substep_hand_peg_max_solid_penetration": np.zeros(
            (control_steps, substeps, 1),
            dtype=np.float64,
        ),
    }


def test_b03_newton_source_validity_accepts_exact_safety_bounds() -> None:
    arrays = _episode_arrays()
    arrays["next_newton_substep_robot_joint_velocity"][1, 2, 0, 7] = (
        -B03_NEWTON_HAND_MAX_ABS_JOINT_VELOCITY_RAD_S
    )
    arrays[
        "next_newton_substep_hand_peg_max_solid_penetration"
    ][2, 3, 0] = B03_NEWTON_HAND_PEG_MAX_SOLID_PENETRATION_M

    report = evaluate_b03_newton_episode_arrays(arrays, _schema())

    assert report["valid"] is True
    assert report["status"] == "valid"
    assert report["code"] == "ok"
    assert report["evaluated_control_steps"] == 3
    assert report["evaluated_physics_substeps"] == 12
    assert report["hand_max_abs_joint_velocity_rad_s"] == pytest.approx(4.0)
    assert report["hand_peg_max_solid_penetration_m"] == pytest.approx(0.001)
    assert b03_newton_source_validity_error(report) is None


def test_b03_newton_source_validity_rejects_hand_speed_but_ignores_arm_speed() -> None:
    arrays = _episode_arrays()
    velocity = arrays["next_newton_substep_robot_joint_velocity"]
    velocity[0, 0, 0, 0] = 50.0  # Arm speed is outside this hand-only gate.
    velocity[1, 2, 1, 9] = np.nextafter(4.0, np.inf)

    report = evaluate_b03_newton_episode_arrays(arrays, _schema())

    assert report["valid"] is False
    assert report["code"] == "hand_joint_velocity_exceeded"
    assert report["first_violation"] == {
        "kind": "hand_joint_velocity",
        "control_step": 1,
        "physics_substep": 2,
        "side": "left",
        "joint_name": "left_finger1_joint3",
        "value": np.nextafter(4.0, np.inf),
        "limit": 4.0,
    }


def test_b03_newton_source_validity_rejects_one_micron_over_penetration() -> None:
    arrays = _episode_arrays()
    arrays[
        "next_newton_substep_hand_peg_max_solid_penetration"
    ][1, 3, 0] = 0.001001

    report = evaluate_b03_newton_episode_arrays(arrays, _schema())

    assert report["valid"] is False
    assert report["code"] == "hand_peg_solid_penetration_exceeded"
    assert report["first_violation"]["control_step"] == 1
    assert report["first_violation"]["physics_substep"] == 3
    assert report["first_violation"]["value"] == pytest.approx(0.001001)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            lambda arrays: arrays.pop(
                "next_newton_substep_robot_joint_velocity"
            ),
            "missing_ground_truth_field",
        ),
        (
            lambda arrays: arrays["next_newton_substep_count"].__setitem__(
                (1, 0), 3
            ),
            "incomplete_substep_trace",
        ),
        (
            lambda arrays: arrays[
                "next_newton_substep_robot_joint_valid"
            ].__setitem__((0, 1, 0, 7), 0),
            "hand_joint_readback_invalid",
        ),
        (
            lambda arrays: arrays[
                "next_newton_substep_robot_joint_velocity"
            ].__setitem__((0, 1, 0, 7), np.nan),
            "hand_joint_velocity_nonfinite",
        ),
        (
            lambda arrays: arrays[
                "next_newton_substep_hand_peg_contact_valid"
            ].__setitem__((0, 1, 0), 0),
            "hand_peg_contact_readback_invalid",
        ),
        (
            lambda arrays: arrays[
                "next_newton_substep_hand_peg_max_solid_penetration"
            ].__setitem__((0, 1, 0), np.inf),
            "hand_peg_penetration_nonfinite",
        ),
    ],
)
def test_b03_newton_source_validity_is_fail_closed(
    mutation,
    expected_code: str,
) -> None:
    arrays = _episode_arrays()
    mutation(arrays)

    report = evaluate_b03_newton_episode_arrays(arrays, _schema())

    assert report["valid"] is False
    assert report["code"] == expected_code
    assert b03_newton_source_validity_error(report)


def test_b03_newton_source_validity_rejects_modified_gate_contract() -> None:
    schema = _schema()
    schema["newton"]["source_validity_gate"][
        "hand_max_abs_joint_velocity_rad_s"
    ] = 4.1

    report = evaluate_b03_newton_episode_arrays(_episode_arrays(), schema)

    assert report["valid"] is False
    assert report["code"] == "invalid_schema_contract"
    assert "must be 4.0" in report["detail"]


def test_b03_newton_one_step_observation_uses_the_same_validator() -> None:
    arrays = _episode_arrays(control_steps=1, prefix="")
    observation = {name: value[0] for name, value in arrays.items()}
    observation["newton_substep_robot_joint_velocity"][0, 0, 7] = 4.01

    report = evaluate_b03_newton_substep_observation(
        observation,
        copy.deepcopy(_schema()),
    )

    assert report["valid"] is False
    assert report["code"] == "hand_joint_velocity_exceeded"
    assert report["evaluated_control_steps"] == 1
    assert report["first_violation"]["control_step"] == 0
