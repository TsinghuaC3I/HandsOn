from __future__ import annotations

from dataclasses import fields, replace
import math

import numpy as np
import pytest

from handson.sim.isaaclab.b320_bc021_contract import (
    BC021AcceptanceCalibration,
    BC021Frame,
    BC021Geometry,
    BC021Stage,
    BC021State,
    CONTACT_KEYS,
    HIDDEN_EVENT_FRAME_SENTINEL,
    HIDDEN_OBSERVATION_FIELDS,
    HIDDEN_OBSERVATION_SHAPE,
    MAX_REFERENCE_TRANSLATION_M,
    PROVISIONAL_ACCEPTANCE_CALIBRATION,
    PROVISIONAL_GEOMETRY,
    PROVISIONAL_SCAFFOLD_ASSET_DIGEST_SHA256,
    RESET_ARM_BASE_POSITIONS,
    RESET_COMFORT_MAX_RADIUS_M,
    RESET_COMFORT_MIN_RADIUS_M,
    RESET_COMMON_X_JITTER_M,
    RESET_INDEPENDENT_X_JITTER_M,
    RESET_Y_JITTER_M,
    STABLE_HOLD_SECONDS,
    STABLE_HOLD_SUBSTEPS,
    TERMINAL_REASON_CODES,
    WORKSPACE_DEBOUNCE_FRAMES,
    sample_bc021_reset,
)


FINAL_TEST_GEOMETRY = replace(PROVISIONAL_GEOMETRY, provisional=False)
FINAL_TEST_CALIBRATION = replace(
    PROVISIONAL_ACCEPTANCE_CALIBRATION,
    basis="pure-test final asset measurement and contact sweep",
    version="bc021-pure-test-final-v1",
    provisional=False,
    final=True,
    asset_digest_sha256="a" * 64,
    geometry_digest_sha256=FINAL_TEST_GEOMETRY.digest_sha256(),
)


def _pose(x: float, y: float, z: float) -> np.ndarray:
    return np.asarray([x, y, z, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _frame(
    *,
    left: np.ndarray | None = None,
    right: np.ndarray | None = None,
    left_velocity: np.ndarray | None = None,
    right_velocity: np.ndarray | None = None,
    forces: dict[str, float] | None = None,
    readable: dict[str, bool] | None = None,
    points: np.ndarray | None = None,
    point_forces: np.ndarray | None = None,
    point_truth: bool = True,
    pose_truth: bool = True,
    velocity_truth: bool = True,
) -> BC021Frame:
    z = 1.01
    contact_points = (
        np.asarray([[0.0, 0.0, z]], dtype=np.float64)
        if points is None
        else np.asarray(points, dtype=np.float64).reshape(-1, 3)
    )
    return BC021Frame(
        left_pose_wxyz=(
            _pose(-0.0375, 0.0, z) if left is None else left.copy()
        ),
        right_pose_wxyz=(
            _pose(0.0375, 0.0, z) if right is None else right.copy()
        ),
        left_velocity=(
            np.zeros(6, dtype=np.float64)
            if left_velocity is None
            else np.asarray(left_velocity, dtype=np.float64)
        ),
        right_velocity=(
            np.zeros(6, dtype=np.float64)
            if right_velocity is None
            else np.asarray(right_velocity, dtype=np.float64)
        ),
        contact_force_n=(
            {"LA": 0.25, "LB": 0.0, "RA": 0.0, "RB": 0.25, "AB": 0.08}
            if forces is None
            else forces
        ),
        contact_readable=(
            {key: True for key in CONTACT_KEYS}
            if readable is None
            else readable
        ),
        block_contact_points_w=contact_points,
        block_contact_point_forces_n=(
            np.full(len(contact_points), 0.02, dtype=np.float64)
            if point_forces is None
            else np.asarray(point_forces, dtype=np.float64)
        ),
        block_contact_points_readable=point_truth,
        pose_truth_readable=pose_truth,
        velocity_truth_readable=velocity_truth,
    )


def _evaluate_valid_hold(
    state: BC021State,
    *,
    start_frame: int = 1,
    count: int = STABLE_HOLD_SUBSTEPS,
    geometry: BC021Geometry = FINAL_TEST_GEOMETRY,
    calibration: BC021AcceptanceCalibration = FINAL_TEST_CALIBRATION,
) -> tuple[dict[str, object], tuple[bool, ...]]:
    result: tuple[dict[str, object], tuple[bool, ...]] | None = None
    for frame_index in range(start_frame, start_frame + count):
        result = state.evaluate(
            _frame(),
            geometry,
            calibration=calibration,
            frame_index=frame_index,
            update_latches=True,
        )
    assert result is not None
    return result


def test_bc021_clean_room_provisional_calibration_is_explicit() -> None:
    calibration = PROVISIONAL_ACCEPTANCE_CALIBRATION
    calibration_fields = {item.name for item in fields(calibration)}

    assert "clean-room" in calibration.basis
    assert calibration.version == "bc021-clean-room-provisional-v1"
    assert calibration.provisional is True
    assert calibration.final is False
    assert calibration.asset_digest_sha256 == (
        PROVISIONAL_SCAFFOLD_ASSET_DIGEST_SHA256
    )
    assert calibration.geometry_digest_sha256 == (
        PROVISIONAL_GEOMETRY.digest_sha256()
    )
    assert calibration.geometry_is_compatible(PROVISIONAL_GEOMETRY) is True
    assert len(calibration.digest_sha256()) == 64
    assert calibration.minimum_bottom_clearance_m == pytest.approx(0.080)
    assert calibration.max_normal_error_deg == pytest.approx(5.0)
    assert calibration.max_lateral_error_m == pytest.approx(0.005)
    assert calibration.min_axial_gap_m == pytest.approx(-0.002)
    assert calibration.max_axial_gap_m == pytest.approx(0.003)
    assert calibration.stable_hold_seconds == pytest.approx(0.40)
    assert calibration.physics_hz == 120
    assert calibration.stable_hold_substeps == 48
    assert STABLE_HOLD_SECONDS == pytest.approx(0.40)
    assert STABLE_HOLD_SUBSTEPS == 48

    # The legacy threshold remains available for replay diagnostics, but is
    # intentionally outside the current acceptance calibration.
    assert MAX_REFERENCE_TRANSLATION_M == pytest.approx(0.020)
    assert "max_reference_translation_m" not in calibration_fields


def test_bc021_provisional_calibration_allows_task_success_but_not_formal_release() -> None:
    state = BC021State()
    metrics, flags = _evaluate_valid_hold(
        state,
        geometry=FINAL_TEST_GEOMETRY,
        calibration=PROVISIONAL_ACCEPTANCE_CALIBRATION,
    )

    assert metrics["geometry_contract_final"] is True
    assert metrics["acceptance_calibration_geometry_compatible"] is True
    assert metrics["acceptance_calibration_provisional"] is True
    assert metrics["acceptance_calibration_final"] is False
    assert metrics["acceptance_calibration_formal_ready"] is False
    assert all(flags)
    assert len(flags) == 4
    assert metrics["success_now"] is True
    assert state.first_success_frame == STABLE_HOLD_SUBSTEPS


def test_bc021_calibration_digest_mismatch_blocks_formal_not_task_success() -> None:
    mismatch = replace(
        FINAL_TEST_CALIBRATION,
        geometry_digest_sha256="b" * 64,
    )
    state = BC021State()
    metrics, flags = _evaluate_valid_hold(
        state,
        geometry=FINAL_TEST_GEOMETRY,
        calibration=mismatch,
    )

    assert mismatch.final is True
    assert mismatch.geometry_is_compatible(FINAL_TEST_GEOMETRY) is False
    assert mismatch.formal_for_geometry(FINAL_TEST_GEOMETRY) is False
    assert metrics["acceptance_calibration_geometry_compatible"] is False
    assert metrics["acceptance_calibration_formal_ready"] is False
    assert all(flags)
    assert len(flags) == 4
    assert metrics["success_now"] is True


def test_bc021_geometry_digest_binds_semantics_not_evidence_status() -> None:
    assert PROVISIONAL_GEOMETRY.digest_sha256() == (
        FINAL_TEST_GEOMETRY.digest_sha256()
    )
    changed_geometry = replace(
        FINAL_TEST_GEOMETRY,
        left_half_extents_m=(0.038, 0.0375, 0.0375),
    )
    assert changed_geometry.digest_sha256() != (
        FINAL_TEST_GEOMETRY.digest_sha256()
    )
    assert FINAL_TEST_CALIBRATION.formal_for_geometry(FINAL_TEST_GEOMETRY)
    assert not FINAL_TEST_CALIBRATION.formal_for_geometry(changed_geometry)
    changed_calibration = replace(
        FINAL_TEST_CALIBRATION,
        max_normal_error_deg=5.9,
    )
    assert changed_calibration.digest_sha256() != (
        FINAL_TEST_CALIBRATION.digest_sha256()
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("asset_digest_sha256", "not-a-sha256"),
        ("asset_digest_sha256", "A" * 64),
        ("geometry_digest_sha256", "0" * 63),
        ("geometry_digest_sha256", "g" * 64),
    ],
)
def test_bc021_calibration_rejects_malformed_digests(
    field_name: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match="SHA-256 digest"):
        replace(
            PROVISIONAL_ACCEPTANCE_CALIBRATION,
            **{field_name: value},
        )


def test_bc021_final_calibration_cannot_relabel_provisional_scaffold() -> None:
    with pytest.raises(ValueError, match="provisional scaffold"):
        replace(
            PROVISIONAL_ACCEPTANCE_CALIBRATION,
            basis="invalid final relabel",
            version="invalid-final-v1",
            provisional=False,
            final=True,
        )

    with pytest.raises(ValueError, match="exactly one"):
        replace(
            PROVISIONAL_ACCEPTANCE_CALIBRATION,
            provisional=True,
            final=True,
        )


def test_bc021_requires_48_physics_substeps_not_one_sample() -> None:
    state = BC021State()
    metrics, flags = _evaluate_valid_hold(
        state,
        count=STABLE_HOLD_SUBSTEPS - 1,
    )
    assert flags[:3] == (True, True, True)
    assert flags[3] is False
    assert metrics["stable_substeps"] == STABLE_HOLD_SUBSTEPS - 1

    metrics, flags = state.evaluate(
        _frame(),
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=STABLE_HOLD_SUBSTEPS,
        update_latches=True,
    )
    assert all(flags)
    assert metrics["success_now"] is True
    assert metrics["acceptance_calibration_final"] is True
    assert metrics["acceptance_calibration_geometry_compatible"] is True
    assert metrics["acceptance_calibration_formal_ready"] is True
    assert state.first_success_frame == STABLE_HOLD_SUBSTEPS
    assert state.stage == BC021Stage.ALIGNED


def test_bc021_provisional_geometry_can_never_report_formal_success() -> None:
    state = BC021State()
    metrics, flags = _evaluate_valid_hold(
        state,
        geometry=PROVISIONAL_GEOMETRY,
    )
    assert flags[2] is False
    assert flags[3] is False
    assert metrics["geometry_contract_final"] is False
    assert metrics["acceptance_calibration_final"] is True
    assert metrics["acceptance_calibration_formal_ready"] is False
    assert metrics["success_now"] is False


@pytest.mark.parametrize(
    ("kwargs", "failed_flag"),
    [
        (
            {
                "left": _pose(-0.0375, 0.0, 0.96),
                "right": _pose(0.0375, 0.0, 0.96),
            },
            1,
        ),
        ({"points": np.asarray([[0.0, 0.034, 1.01]])}, 3),
        (
            {
                "forces": {
                    "LA": 0.25,
                    "LB": 0.10,
                    "RA": 0.0,
                    "RB": 0.25,
                    "AB": 0.08,
                }
            },
            0,
        ),
        ({"readable": {key: key != "AB" for key in CONTACT_KEYS}}, 0),
    ],
)
def test_bc021_rejects_tabletop_wrong_face_cross_hand_and_missing_truth(
    kwargs: dict[str, object],
    failed_flag: int,
) -> None:
    state = BC021State()
    _metrics, flags = state.evaluate(
        _frame(**kwargs),
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=1,
        update_latches=True,
    )
    assert flags[failed_flag] is False
    assert not all(flags)
    assert state.first_terminal_reason is None


def test_bc021_accepts_solver_manifold_points_on_aligned_face_boundary() -> None:
    state = BC021State()
    boundary_contact = _frame(
        points=np.asarray(
            [[0.0, 0.0, 1.01], [0.0, 0.034, 1.01]],
            dtype=np.float64,
        ),
        point_forces=np.asarray([0.02, 0.02], dtype=np.float64),
    )
    result = None
    for frame_index in range(1, STABLE_HOLD_SUBSTEPS + 1):
        result = state.evaluate(
            boundary_contact,
            FINAL_TEST_GEOMETRY,
            calibration=FINAL_TEST_CALIBRATION,
            frame_index=frame_index,
            update_latches=True,
        )
    assert result is not None
    metrics, flags = result
    assert metrics["contact_point_outside_marked_patch"] is True
    assert metrics["marked_patch_contact_point_current"] is True
    assert metrics["target_face_contact_geometry"] is True
    assert metrics["non_target_block_contact"] is False
    assert metrics["marked_face_contact_current"] is True
    assert all(flags)


def test_bc021_block_contact_cannot_bypass_misaligned_face_geometry() -> None:
    state = BC021State()
    metrics, flags = state.evaluate(
        _frame(
            right=_pose(0.0375, 0.012, 1.01),
            points=np.asarray([[0.0, 0.006, 1.01]], dtype=np.float64),
            point_forces=np.asarray([0.0], dtype=np.float64),
        ),
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=1,
        update_latches=True,
    )
    assert metrics["block_contact_current"] is True
    assert metrics["target_face_contact_geometry"] is False
    assert metrics["marked_face_contact_current"] is False
    assert metrics["non_target_block_contact"] is True
    assert flags[2] is False
    assert flags[3] is False


def test_bc021_reported_marked_contact_has_no_force_threshold() -> None:
    state = BC021State()
    zero_force_contact = _frame(
        forces={"LA": 0.25, "LB": 0.0, "RA": 0.0, "RB": 0.25, "AB": 0.0},
        point_forces=np.asarray([0.0], dtype=np.float64),
    )
    metrics, _flags = state.evaluate(
        zero_force_contact,
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=1,
        update_latches=True,
    )
    assert metrics["block_contact_current"] is True
    assert metrics["marked_face_contact_current"] is True

    no_contact_metrics, _flags = BC021State().evaluate(
        _frame(
            forces={
                "LA": 0.25,
                "LB": 0.0,
                "RA": 0.0,
                "RB": 0.25,
                "AB": 1.0,
            },
            points=np.zeros((0, 3), dtype=np.float64),
            point_forces=np.zeros(0, dtype=np.float64),
        ),
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=1,
        update_latches=True,
    )
    assert no_contact_metrics["block_contact_current"] is False
    assert no_contact_metrics["marked_face_contact_current"] is False


def test_bc021_does_not_latch_tabletop_marked_contact_as_task_progress() -> None:
    state = BC021State()
    tabletop_z = 0.9275
    state.evaluate(
        _frame(
            left=_pose(-0.0375, 0.0, tabletop_z),
            right=_pose(0.0375, 0.0, tabletop_z),
            points=np.asarray([[0.0, 0.0, tabletop_z]]),
        ),
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=1,
        update_latches=True,
    )
    assert state.marked_face_contact_seen is False
    assert state.first_marked_contact_frame is None
    assert state.both_airborne_seen is False

    no_block_contact = {
        "LA": 0.25,
        "LB": 0.0,
        "RA": 0.0,
        "RB": 0.25,
        "AB": 0.0,
    }
    state.evaluate(
        _frame(
            forces=no_block_contact,
            points=np.zeros((0, 3), dtype=np.float64),
        ),
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=2,
        update_latches=True,
    )
    assert state.both_airborne_seen is True
    assert state.marked_face_contact_seen is False

    state.evaluate(
        _frame(),
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=3,
        update_latches=True,
    )
    assert state.marked_face_contact_seen is True
    assert state.first_marked_contact_frame == 3


def test_bc021_contact_hold_does_not_require_velocity_threshold() -> None:
    velocity = np.asarray([0.10, 0.0, 0.0, 0.0, 0.0, 0.0])
    state = BC021State()
    result = None
    for frame_index in range(1, STABLE_HOLD_SUBSTEPS + 1):
        result = state.evaluate(
            _frame(left_velocity=velocity, right_velocity=velocity),
            FINAL_TEST_GEOMETRY,
            calibration=FINAL_TEST_CALIBRATION,
            frame_index=frame_index,
            update_latches=True,
        )
    assert result is not None
    metrics, flags = result
    assert metrics["relative_linear_speed_m_s"] == pytest.approx(0.0)
    assert metrics["motion_stable"] is False
    assert all(flags)


@pytest.mark.parametrize("kind", ["pose_flag", "nan_pose"])
def test_bc021_pose_truth_fails_closed(kind: str) -> None:
    kwargs: dict[str, object] = {}
    if kind == "pose_flag":
        kwargs["pose_truth"] = False
    else:
        pose = _pose(-0.0375, 0.0, 1.01)
        pose[0] = math.nan
        kwargs["left"] = pose

    state = BC021State()
    metrics, flags = state.evaluate(
        _frame(**kwargs),
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=1,
        update_latches=True,
    )
    assert not all(flags)
    assert metrics["pose_truth_readable"] is False
    numeric = [
        metrics["minimum_bottom_clearance_m"],
        metrics["normal_error_deg"],
        metrics["lateral_error_m"],
        metrics["axial_gap_m"],
    ]
    assert np.all(np.isfinite(numeric))


@pytest.mark.parametrize("kind", ["velocity_flag", "nan_velocity"])
def test_bc021_velocity_is_diagnostic_not_a_success_gate(kind: str) -> None:
    kwargs: dict[str, object] = {}
    if kind == "velocity_flag":
        kwargs["velocity_truth"] = False
    else:
        velocity = np.zeros(6, dtype=np.float64)
        velocity[0] = math.nan
        kwargs["left_velocity"] = velocity

    state = BC021State()
    result = None
    for frame_index in range(1, STABLE_HOLD_SUBSTEPS + 1):
        result = state.evaluate(
            _frame(**kwargs),
            FINAL_TEST_GEOMETRY,
            calibration=FINAL_TEST_CALIBRATION,
            frame_index=frame_index,
            update_latches=True,
        )
    assert result is not None
    metrics, flags = result
    assert metrics["velocity_truth_readable"] is False
    assert metrics["motion_stable"] is False
    assert all(flags)


def test_bc021_reference_anchor_starts_when_left_block_becomes_airborne() -> None:
    state = BC021State()
    no_mate = {"LA": 0.25, "LB": 0.0, "RA": 0.0, "RB": 0.0, "AB": 0.0}
    state.evaluate(
        _frame(
            right=_pose(0.0375, 0.0, 0.9275),
            forces=no_mate,
            points=np.zeros((0, 3), dtype=np.float64),
        ),
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=1,
        update_latches=True,
    )
    assert state.reference_pose_wxyz is not None
    assert state.reference_latched_frame == 1
    assert state.assigned_contact_seen is False
    assert state.both_airborne_seen is False

    for frame_index in (2, 3):
        state.evaluate(
            _frame(
                left=_pose(-0.0375, 0.030, 1.01),
                right=_pose(0.0375, 0.030, 1.01),
                points=np.asarray([[0.0, 0.030, 1.01]]),
            ),
            FINAL_TEST_GEOMETRY,
            calibration=FINAL_TEST_CALIBRATION,
            frame_index=frame_index,
            update_latches=True,
        )
    assert state.max_reference_translation_m == pytest.approx(0.030)
    assert state.reference_limit_frames == 2
    assert state.first_terminal_reason is None


def test_bc021_reference_anchor_waits_for_level_left_block() -> None:
    state = BC021State()
    half_angle = math.radians(17.0) / 2.0
    tilted_left = np.asarray(
        [
            -0.0375,
            0.0,
            1.01,
            math.cos(half_angle),
            math.sin(half_angle),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    metrics, _flags = state.evaluate(
        _frame(left=tilted_left),
        FINAL_TEST_GEOMETRY,
        frame_index=1,
        update_latches=True,
    )
    assert metrics["left_flat"] is False
    assert state.reference_pose_wxyz is None

    metrics, _flags = state.evaluate(
        _frame(),
        FINAL_TEST_GEOMETRY,
        frame_index=2,
        update_latches=True,
    )
    assert metrics["left_flat"] is True
    assert metrics["reference_latched"] is True
    assert state.reference_latched_frame == 2


def test_bc021_reference_anchor_does_not_wait_for_low_motion() -> None:
    state = BC021State()
    metrics, _flags = state.evaluate(
        _frame(
            left_velocity=np.asarray([0.08, 0.0, 0.0, 0.0, 0.0, 0.0]),
            right_velocity=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
        FINAL_TEST_GEOMETRY,
        frame_index=1,
        update_latches=True,
    )
    assert metrics["left_flat"] is True
    assert state.reference_latched_frame == 1


def test_bc021_reference_excursion_is_diagnostic_only() -> None:
    state = BC021State()
    state.evaluate(
        _frame(),
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=1,
        update_latches=True,
    )
    excursion_m = MAX_REFERENCE_TRANSLATION_M + 0.010
    moved = _frame(
        left=_pose(-0.0375, excursion_m, 1.01),
        right=_pose(0.0375, excursion_m, 1.01),
        points=np.asarray([[0.0, excursion_m, 1.01]]),
    )
    metrics = None
    flags = None
    for frame_index in range(2, STABLE_HOLD_SUBSTEPS + 1):
        metrics, flags = state.evaluate(
            moved,
            FINAL_TEST_GEOMETRY,
            calibration=FINAL_TEST_CALIBRATION,
            frame_index=frame_index,
            update_latches=True,
        )

    assert metrics is not None and flags is not None
    assert metrics["max_reference_translation_m"] == pytest.approx(excursion_m)
    assert state.reference_limit_frames == STABLE_HOLD_SUBSTEPS - 1
    assert state.first_terminal_reason is None
    assert all(flags)


def test_bc021_soft_error_resets_hold_and_can_recover() -> None:
    state = BC021State()
    _evaluate_valid_hold(state, count=20)
    assert state.stable_substeps == 20
    separated = _frame(
        right=_pose(0.050, 0.0, 1.01),
        forces={"LA": 0.25, "LB": 0.0, "RA": 0.0, "RB": 0.25, "AB": 0.0},
        points=np.zeros((0, 3), dtype=np.float64),
    )
    state.evaluate(
        separated,
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=21,
        update_latches=True,
    )
    assert state.stable_substeps == 0
    assert state.first_terminal_reason is None
    _metrics, flags = _evaluate_valid_hold(
        state,
        start_frame=22,
        count=STABLE_HOLD_SUBSTEPS,
    )
    assert all(flags)


def test_bc021_reference_excursion_does_not_terminal_before_drop() -> None:
    state = BC021State()
    state.evaluate(
        _frame(),
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=1,
        update_latches=True,
    )
    moved = _frame(
        left=_pose(-0.0375, 0.021, 1.01),
        right=_pose(0.0375, 0.021, 1.01),
        points=np.asarray([[0.0, 0.021, 1.01]]),
    )
    for frame_index in (2, 3):
        state.evaluate(
            moved,
            FINAL_TEST_GEOMETRY,
            calibration=FINAL_TEST_CALIBRATION,
            frame_index=frame_index,
            update_latches=True,
        )
    assert state.first_terminal_reason is None

    dropped = _frame(
        left=_pose(-0.0375, 0.0, 0.80),
        right=_pose(0.0375, 0.0, 0.80),
    )
    for frame_index in (4, 5, 6):
        state.evaluate(
            dropped,
            FINAL_TEST_GEOMETRY,
            calibration=FINAL_TEST_CALIBRATION,
            frame_index=frame_index,
            update_latches=True,
        )
    assert state.first_terminal_reason == "block_dropped_after_airborne"
    assert state.first_terminal_frame == 6
    assert state.stage == BC021Stage.FAILED


def test_bc021_workspace_excursion_is_recoverable_until_consecutive_limit() -> None:
    state = BC021State()
    state.evaluate(
        _frame(),
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=1,
        update_latches=True,
    )
    outside = _frame(right=_pose(0.28, 0.0, 1.01))

    metrics, _flags = state.evaluate(
        outside,
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=2,
        update_latches=True,
    )
    assert metrics["outside_collection_workspace"] is True
    assert state.workspace_frames == 1
    assert state.first_terminal_reason is None

    metrics, _flags = state.evaluate(
        _frame(),
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=3,
        update_latches=True,
    )
    assert metrics["outside_collection_workspace"] is False
    assert state.workspace_frames == 0
    assert state.first_terminal_reason is None

    terminal_frame = 3 + WORKSPACE_DEBOUNCE_FRAMES
    for frame_index in range(4, terminal_frame):
        state.evaluate(
            outside,
            FINAL_TEST_GEOMETRY,
            calibration=FINAL_TEST_CALIBRATION,
            frame_index=frame_index,
            update_latches=True,
        )
        assert state.first_terminal_reason is None
    metrics, _flags = state.evaluate(
        outside,
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=terminal_frame,
        update_latches=True,
    )
    assert state.workspace_frames == WORKSPACE_DEBOUNCE_FRAMES
    assert metrics["workspace_debounce_frames_required"] == (
        WORKSPACE_DEBOUNCE_FRAMES
    )
    assert state.first_terminal_reason == "block_left_collection_workspace"
    assert state.first_terminal_frame == terminal_frame


def test_bc021_deep_penetration_requires_real_block_contact_and_debounce() -> None:
    no_contact_state = BC021State()
    no_contact_state.evaluate(
        _frame(),
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=1,
        update_latches=True,
    )
    penetrating_without_contact = _frame(
        right=_pose(0.025, 0.0, 1.01),
        forces={"LA": 0.25, "LB": 0.0, "RA": 0.0, "RB": 0.25, "AB": 0.0},
        points=np.zeros((0, 3), dtype=np.float64),
    )
    for frame_index in (2, 3, 4):
        no_contact_state.evaluate(
            penetrating_without_contact,
            FINAL_TEST_GEOMETRY,
            calibration=FINAL_TEST_CALIBRATION,
            frame_index=frame_index,
            update_latches=True,
        )
    assert no_contact_state.first_terminal_reason is None

    contact_state = BC021State()
    contact_state.evaluate(
        _frame(),
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=1,
        update_latches=True,
    )
    penetrating_contact = _frame(
        right=_pose(0.025, 0.0, 1.01),
        points=np.asarray([[-0.00625, 0.0, 1.01]], dtype=np.float64),
        point_forces=np.asarray([0.0], dtype=np.float64),
    )
    for frame_index in (2, 3):
        contact_state.evaluate(
            penetrating_contact,
            FINAL_TEST_GEOMETRY,
            calibration=FINAL_TEST_CALIBRATION,
            frame_index=frame_index,
            update_latches=True,
        )
        assert contact_state.first_terminal_reason is None
    contact_state.evaluate(
        penetrating_contact,
        FINAL_TEST_GEOMETRY,
        calibration=FINAL_TEST_CALIBRATION,
        frame_index=4,
        update_latches=True,
    )
    assert (
        contact_state.first_terminal_reason
        == "sustained_deep_face_penetration"
    )


def test_bc021_arbitrary_semantic_face_frame_is_not_hardcoded_to_x() -> None:
    geometry = BC021Geometry(
        left_half_extents_m=(0.0375, 0.0375, 0.0375),
        right_half_extents_m=(0.0375, 0.0375, 0.0375),
        left_marked_normal_local=(0.0, 1.0, 0.0),
        right_marked_normal_local=(0.0, -1.0, 0.0),
        left_marked_tangent_u_local=(0.0, 0.0, 1.0),
        left_marked_tangent_v_local=(1.0, 0.0, 0.0),
        right_marked_tangent_u_local=(0.0, 0.0, 1.0),
        right_marked_tangent_v_local=(-1.0, 0.0, 0.0),
        provisional=False,
    )
    calibration = replace(
        FINAL_TEST_CALIBRATION,
        geometry_digest_sha256=geometry.digest_sha256(),
    )
    state = BC021State()
    frame = _frame(
        left=_pose(0.0, -0.0375, 1.01),
        right=_pose(0.0, 0.0375, 1.01),
        points=np.asarray([[0.0, 0.0, 1.01]]),
    )
    result = None
    for frame_index in range(1, STABLE_HOLD_SUBSTEPS + 1):
        result = state.evaluate(
            frame,
            geometry,
            calibration=calibration,
            frame_index=frame_index,
            update_latches=True,
        )
    assert result is not None
    metrics, flags = result
    assert metrics["normal_error_deg"] == pytest.approx(0.0)
    assert metrics["marked_face_contact_current"] is True
    assert all(flags)


def test_bc021_seeded_reset_is_repeatable_supported_and_separated() -> None:
    left_home = _pose(-0.32, 0.20, 0.9275)
    right_home = _pose(-0.32, -0.20, 0.9275)
    left_hand = _pose(-0.32, 0.32, 1.15)
    right_hand = _pose(-0.32, -0.32, 1.15)

    def sample(seed: int):
        return sample_bc021_reset(
            np.random.RandomState(seed),
            left_home_pose_wxyz=left_home,
            right_home_pose_wxyz=right_home,
            geometry=FINAL_TEST_GEOMETRY,
            randomize=True,
            left_hand_home_wxyz=left_hand,
            right_hand_home_wxyz=right_hand,
        )

    first = sample(17)
    second = sample(17)
    other = sample(18)
    np.testing.assert_allclose(first.left_pose_wxyz, second.left_pose_wxyz)
    np.testing.assert_allclose(first.right_pose_wxyz, second.right_pose_wxyz)
    assert not np.array_equal(first.left_pose_wxyz, other.left_pose_wxyz)
    assert first.fallback_used is False
    assert first.left_pose_wxyz[1] > 0.0
    assert first.right_pose_wxyz[1] < 0.0
    assert np.linalg.norm(first.left_pose_wxyz[:3] - first.right_pose_wxyz[:3]) > 0.20
    samples = [sample(seed) for seed in range(64)]
    assert len(
        {
            tuple(np.round(item.left_pose_wxyz[:2], decimals=8))
            for item in samples
        }
    ) >= 2
    assert len(
        {
            tuple(np.round(item.left_pose_wxyz[3:7], decimals=8))
            for item in samples
        }
    ) >= 2
    for item in samples:
        assert item.left_pose_wxyz[1] > 0.0
        assert item.right_pose_wxyz[1] < 0.0
        assert item.fallback_used is False
        assert np.linalg.norm(
            item.left_pose_wxyz[:3] - item.right_pose_wxyz[:3]
        ) > 0.20
        for side, pose, home in (
            ("left", item.left_pose_wxyz, left_home),
            ("right", item.right_pose_wxyz, right_home),
        ):
            max_x_jitter = (
                RESET_COMMON_X_JITTER_M + RESET_INDEPENDENT_X_JITTER_M
            )
            assert abs(float(pose[0] - home[0])) <= max_x_jitter + 1.0e-12
            assert abs(float(pose[1] - home[1])) <= RESET_Y_JITTER_M + 1.0e-12
            arm_radius = float(
                np.linalg.norm(
                    pose[:3]
                    - np.asarray(
                        RESET_ARM_BASE_POSITIONS[side],
                        dtype=np.float64,
                    )
                )
            )
            assert RESET_COMFORT_MIN_RADIUS_M <= arm_radius
            assert arm_radius <= RESET_COMFORT_MAX_RADIUS_M

    # The comfort sampler preserves a deliberate 100 mm margin to the live
    # controller's 0.78 m physical workspace limit.
    assert RESET_COMFORT_MAX_RADIUS_M == pytest.approx(0.78 - 0.10)


def test_bc021_nonrandom_reset_is_validated_instead_of_bypassed() -> None:
    with pytest.raises(RuntimeError, match="non-random reset"):
        sample_bc021_reset(
            np.random.RandomState(1),
            left_home_pose_wxyz=_pose(-0.32, 0.20, 1.20),
            right_home_pose_wxyz=_pose(-0.32, -0.20, 1.20),
            geometry=FINAL_TEST_GEOMETRY,
            randomize=False,
            left_hand_home_wxyz=_pose(-0.32, 0.32, 1.15),
            right_hand_home_wxyz=_pose(-0.32, -0.32, 1.15),
        )


def test_bc021_terminal_and_stage_observations_have_fixed_shapes() -> None:
    state = BC021State()
    state.first_terminal_reason = "block_left_collection_workspace"
    state.first_terminal_frame = 9
    state.stage = BC021Stage.FAILED
    np.testing.assert_allclose(
        state.terminal_observation(),
        np.asarray([1.0, 4.0, 9.0, float(BC021Stage.FAILED)]),
    )
    assert state.stage_observation().shape == (12,)


def test_bc021_hidden_observation_has_complete_fixed_replay_schema() -> None:
    assert HIDDEN_OBSERVATION_SHAPE == (30,)
    assert len(HIDDEN_OBSERVATION_FIELDS) == HIDDEN_OBSERVATION_SHAPE[0]
    assert len(set(HIDDEN_OBSERVATION_FIELDS)) == len(
        HIDDEN_OBSERVATION_FIELDS
    )
    assert {item.name for item in fields(BC021State)} == {
        "stage",
        "assigned_contact_seen",
        "left_airborne_seen",
        "both_airborne_seen",
        "reference_pose_wxyz",
        "max_reference_translation_m",
        "marked_face_contact_seen",
        "stable_substeps",
        "reference_limit_frames",
        "deep_penetration_frames",
        "drop_frames",
        "workspace_frames",
        "first_assigned_contact_frame",
        "first_left_airborne_frame",
        "first_both_airborne_frame",
        "reference_latched_frame",
        "first_marked_contact_frame",
        "stable_started_frame",
        "first_success_frame",
        "first_terminal_reason",
        "first_terminal_frame",
        # Explicitly excluded because it is recomputed from physical truth.
        "last_metrics",
    }

    state = BC021State(
        stage=BC021Stage.FAILED,
        assigned_contact_seen=True,
        left_airborne_seen=True,
        both_airborne_seen=True,
        reference_pose_wxyz=_pose(-0.0375, 0.012, 1.01),
        max_reference_translation_m=0.012,
        marked_face_contact_seen=True,
        stable_substeps=17,
        reference_limit_frames=1,
        deep_penetration_frames=2,
        drop_frames=1,
        workspace_frames=2,
        first_assigned_contact_frame=3,
        first_left_airborne_frame=4,
        first_both_airborne_frame=5,
        reference_latched_frame=4,
        first_marked_contact_frame=6,
        stable_started_frame=7,
        first_terminal_reason="block_left_collection_workspace",
        first_terminal_frame=61,
    )
    observation = state.hidden_observation()
    values = dict(zip(HIDDEN_OBSERVATION_FIELDS, observation, strict=True))

    assert observation.shape == HIDDEN_OBSERVATION_SHAPE
    assert observation.dtype == np.float64
    assert np.all(np.isfinite(observation))
    assert values["stage"] == float(BC021Stage.FAILED)
    assert values["assigned_contact_seen"] == 1.0
    assert values["left_airborne_seen"] == 1.0
    assert values["both_airborne_seen"] == 1.0
    assert values["marked_face_contact_seen"] == 1.0
    assert values["reference_pose_valid"] == 1.0
    np.testing.assert_allclose(
        observation[6:13],
        state.reference_pose_wxyz,
    )
    assert values["max_reference_translation_m"] == pytest.approx(0.012)
    assert values["stable_substeps"] == 17.0
    assert values["reference_limit_frames"] == 1.0
    assert values["deep_penetration_frames"] == 2.0
    assert values["drop_frames"] == 1.0
    assert values["workspace_frames"] == 2.0
    assert values["first_assigned_contact_frame"] == 3.0
    assert values["first_left_airborne_frame"] == 4.0
    assert values["first_both_airborne_frame"] == 5.0
    assert values["reference_latched_frame"] == 4.0
    assert values["first_marked_contact_frame"] == 6.0
    assert values["stable_started_frame"] == 7.0
    assert values["first_success_frame"] == HIDDEN_EVENT_FRAME_SENTINEL
    assert values["first_terminal_frame"] == 61.0
    assert values["success_latched"] == 0.0
    assert values["terminal_latched"] == 1.0
    assert values["terminal_reason_code"] == float(
        TERMINAL_REASON_CODES["block_left_collection_workspace"]
    )


def test_bc021_hidden_observation_uses_unambiguous_sentinels_and_resets() -> None:
    state = BC021State(
        reference_pose_wxyz=_pose(-0.1, 0.2, 1.1),
        stable_substeps=9,
        reference_limit_frames=1,
        deep_penetration_frames=1,
        drop_frames=1,
        workspace_frames=1,
        first_assigned_contact_frame=0,
        first_success_frame=8,
        first_terminal_reason="block_left_collection_workspace",
        first_terminal_frame=9,
    )
    state.reset()
    observation = state.hidden_observation()
    values = dict(zip(HIDDEN_OBSERVATION_FIELDS, observation, strict=True))

    assert values["stage"] == float(BC021Stage.RESET)
    assert values["reference_pose_valid"] == 0.0
    np.testing.assert_allclose(
        observation[6:13],
        np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    for name in (
        "stable_substeps",
        "reference_limit_frames",
        "deep_penetration_frames",
        "drop_frames",
        "workspace_frames",
    ):
        assert values[name] == 0.0
    for name in (
        "first_assigned_contact_frame",
        "first_left_airborne_frame",
        "first_both_airborne_frame",
        "reference_latched_frame",
        "first_marked_contact_frame",
        "stable_started_frame",
        "first_success_frame",
        "first_terminal_frame",
    ):
        assert values[name] == HIDDEN_EVENT_FRAME_SENTINEL
    assert values["success_latched"] == 0.0
    assert values["terminal_latched"] == 0.0
    assert values["terminal_reason_code"] == 0.0


def test_bc021_first_success_frame_is_immutable_in_hidden_state() -> None:
    state = BC021State()
    _evaluate_valid_hold(state)
    first_success_frame = state.first_success_frame
    assert first_success_frame == STABLE_HOLD_SUBSTEPS

    outside = _frame(right=_pose(0.28, 0.0, 1.01))
    flags = None
    for frame_index in range(
        STABLE_HOLD_SUBSTEPS + 1,
        STABLE_HOLD_SUBSTEPS + WORKSPACE_DEBOUNCE_FRAMES + 1,
    ):
        _metrics, flags = state.evaluate(
            outside,
            FINAL_TEST_GEOMETRY,
            calibration=FINAL_TEST_CALIBRATION,
            frame_index=frame_index,
            update_latches=True,
        )
    values = dict(
        zip(
            HIDDEN_OBSERVATION_FIELDS,
            state.hidden_observation(),
            strict=True,
        )
    )
    assert flags is not None and all(flags)
    assert state.first_success_frame == first_success_frame
    assert state.first_terminal_reason is None
    assert state.stage == BC021Stage.ALIGNED
    assert values["success_latched"] == 1.0
    assert values["terminal_latched"] == 0.0
    assert values["first_success_frame"] == float(first_success_frame)
