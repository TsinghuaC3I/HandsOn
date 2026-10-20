from __future__ import annotations

import os

import numpy as np
import pytest

from scripts.c01_compute_arm_plan_qpos import (
    _DEFAULT_ACCELERATION_WEIGHT,
    _DEFAULT_MAX_JOINT_STEP,
    _DEFAULT_VELOCITY_WEIGHT,
    _bidirectional_stitch_candidates,
    _collision_valid_frames,
    _configure_env,
    _fk_valid_frames,
    _global_branch_select,
    _global_smooth_arm_path,
    _quaternion_angular_error,
    _raise_for_invalid_postcheck,
)
from scripts.c01_render_arm_replay import _require_validated_arm_plan
from scripts.record_demos_zarr import (
    _e01_offline_arm_python,
    _should_auto_compute_e01_arm_plan,
    _should_auto_generate_c01_arm_video,
    _should_auto_generate_plan_arm_video,
    _uses_e01_detached_hand_pipeline,
)


def _joint_ranges() -> np.ndarray:
    return np.tile(np.asarray([[-2.0, 2.0]], dtype=np.float64), (14, 1))


@pytest.mark.parametrize("task_name", ["t08_hammer_nail", "plan1_hammer_nail"])
def test_arm_completion_entry_supports_c01_and_plan1(task_name: str) -> None:
    assert _should_auto_generate_c01_arm_video(
        task_name,
        enabled=True,
        env_value="1",
    )


def test_arm_completion_entry_preserves_opt_outs() -> None:
    assert not _should_auto_generate_c01_arm_video(
        "water_plant",
        enabled=True,
        env_value="1",
    )
    assert not _should_auto_generate_c01_arm_video(
        "t08_hammer_nail",
        enabled=False,
        env_value="1",
    )
    assert not _should_auto_generate_c01_arm_video(
        "plan1_hammer_nail",
        enabled=True,
        env_value="false",
    )


@pytest.mark.parametrize(
    "task_name",
    ["plan2_fruit_to_bucket", "plan8_peg_in_socket", "plan10_clear_box_lid"],
)
def test_online_plan_arm_renderer_supports_plan2_plus(task_name: str) -> None:
    assert _should_auto_generate_plan_arm_video(
        task_name,
        enabled=True,
        env_value="1",
    )


def test_online_plan_arm_renderer_preserves_boundaries_and_opt_outs() -> None:
    assert not _should_auto_generate_plan_arm_video(
        "plan1_hammer_nail",
        enabled=True,
        env_value="1",
    )
    assert not _should_auto_generate_plan_arm_video(
        "t26_place_fruit_in_bucket",
        enabled=True,
        env_value="1",
    )
    assert not _should_auto_generate_plan_arm_video(
        "plan2_fruit_to_bucket",
        enabled=False,
        env_value="1",
    )
    assert not _should_auto_generate_plan_arm_video(
        "plan2_fruit_to_bucket",
        enabled=True,
        env_value="off",
    )


@pytest.mark.parametrize("sim_backend", ["genesis", "isaacsim"])
def test_e01_new_backends_complete_arm_ik_after_detached_hand_collection(
    sim_backend: str,
) -> None:
    assert _should_auto_compute_e01_arm_plan(
        "e01_fold_garment",
        sim_backend=sim_backend,
        env_value="1",
    )
    assert _uses_e01_detached_hand_pipeline("e01_fold_garment", sim_backend)


def test_e01_offline_arm_ik_uses_plan_runtime_instead_of_genesis_python() -> None:
    assert _e01_offline_arm_python(
        env_value=" /opt/handson/bin/python ",
        fallback="/opt/genesis/bin/python",
    ) == "/opt/handson/bin/python"
    assert _e01_offline_arm_python(
        env_value="",
        fallback="/opt/genesis/bin/python",
    ) == "/opt/genesis/bin/python"


@pytest.mark.parametrize(
    ("task_name", "sim_backend", "env_value"),
    [
        ("e01_fold_garment", "mujoco", "1"),
        ("t08_hammer_nail", "genesis", "1"),
        ("e02_tie_rope_knot", "isaacsim", "1"),
        ("e01_fold_garment", "genesis", "false"),
        ("e01_fold_garment", "isaacsim", "off"),
    ],
)
def test_e01_offline_arm_ik_preserves_backend_and_opt_out_boundaries(
    task_name: str,
    sim_backend: str,
    env_value: str,
) -> None:
    assert not _should_auto_compute_e01_arm_plan(
        task_name,
        sim_backend=sim_backend,
        env_value=env_value,
    )


def test_offline_planner_keeps_distant_analytic_branch_candidates(
    monkeypatch,
) -> None:
    monkeypatch.delenv("HANDSON_ROKAE_WUJI_IK_FALLBACK_JUMP", raising=False)

    _configure_env()

    assert os.environ["HANDSON_ROKAE_WUJI_IK_FALLBACK_JUMP"] == "0"


def test_bidirectional_stitch_bypasses_opposite_track_jumps() -> None:
    forward = np.zeros((10, 14), dtype=np.float64)
    backward = np.zeros_like(forward)
    forward[:, 0] = np.asarray(
        [0.0, 0.1, 0.2, 0.3, 0.4, 2.0, 2.1, 2.2, 2.3, 2.4]
    )
    backward[:, 0] = np.asarray(
        [-2.0, -1.0, 0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 4.0]
    )

    candidates = _bidirectional_stitch_candidates(
        forward,
        backward,
        max_joint_step=0.4,
    )

    stitched, source, metadata = next(
        candidate
        for candidate in candidates
        if candidate[2]["strategy"] == "forward_backward_forward"
    )
    assert metadata["start"] < metadata["stop"]
    assert source[0] == 0
    assert source[-1] == 0
    assert np.any(source == 1)
    assert np.max(np.abs(np.diff(stitched, axis=0))) <= 0.4 + 1e-12


def test_global_branch_select_optimizes_the_whole_sequence() -> None:
    """The branch choice is not locked to the locally nearest previous frame."""

    forward = np.zeros((6, 14), dtype=np.float64)
    backward = np.zeros_like(forward)
    forward[:, 0] = np.asarray([0.0, 0.1, 0.2, 1.4, 1.5, 1.6])
    backward[:, 0] = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    success = np.ones((6, 1), dtype=np.float64)
    error = np.full((6, 1), 0.001, dtype=np.float64)

    selected, source = _global_branch_select(
        forward,
        backward,
        success,
        success,
        error,
        error,
        max_joint_step=0.4,
        max_ik_error=0.08,
    )

    # The reverse track is globally continuous; a frame-local solver would
    # otherwise commit to the forward branch and carry a 1.2-rad jump.
    np.testing.assert_array_equal(source, np.ones(6, dtype=np.uint8))
    np.testing.assert_allclose(selected[:, 0], backward[:, 0])


def test_global_smoother_prunes_invalid_high_error_and_short_branch_islands() -> None:
    frame_count = 12
    baseline = np.linspace(0.0, 0.11, frame_count, dtype=np.float64)
    raw = np.zeros((frame_count, 14), dtype=np.float64)
    raw[:, 0] = baseline
    errors = np.full(frame_count, 0.01, dtype=np.float64)

    raw[2] = np.nan
    raw[5, 0] = 1.55
    errors[5] = 0.35
    raw[8:10, 0] += 1.20

    smoothed, rejected = _global_smooth_arm_path(
        raw,
        errors,
        _joint_ranges(),
        np.zeros(14, dtype=np.float64),
        velocity_weight=1.5,
        acceleration_weight=24.0,
        max_joint_step=0.05,
    )

    assert rejected[2]
    assert rejected[5]
    assert np.all(rejected[8:10])
    assert np.all(np.isfinite(smoothed))
    assert np.max(np.abs(np.diff(smoothed, axis=0))) <= 0.05 + 1e-10
    assert np.all(smoothed >= -2.0)
    assert np.all(smoothed <= 2.0)


def test_global_smoother_reduces_acceleration_without_erasing_endpoints() -> None:
    frame_count = 30
    time = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    clean = 0.6 * time
    noise = 0.025 * np.sin(np.arange(frame_count, dtype=np.float64) * np.pi)
    noise[1::2] = 0.025
    noise[2::2] = -0.025
    raw = np.zeros((frame_count, 14), dtype=np.float64)
    raw[:, 0] = clean + noise
    raw[0, 0] = clean[0]
    raw[-1, 0] = clean[-1]

    smoothed, rejected = _global_smooth_arm_path(
        raw,
        np.full(frame_count, 0.01, dtype=np.float64),
        _joint_ranges(),
        np.zeros(14, dtype=np.float64),
        velocity_weight=1.5,
        acceleration_weight=24.0,
        max_joint_step=0.05,
    )

    raw_acceleration = np.diff(raw[:, 0], n=2)
    smooth_acceleration = np.diff(smoothed[:, 0], n=2)
    assert not np.any(rejected)
    assert np.linalg.norm(smooth_acceleration) < np.linalg.norm(raw_acceleration)
    assert smoothed[0, 0] == pytest.approx(raw[0, 0], abs=0.02)
    assert smoothed[-1, 0] == pytest.approx(raw[-1, 0], abs=0.02)
    assert np.max(np.abs(np.diff(smoothed[:, 0]))) <= 0.05 + 1e-10


def test_production_defaults_follow_fast_reachable_right_arm_motion() -> None:
    raw = np.zeros((7, 14), dtype=np.float64)
    raw[:, 0] = np.asarray([0.0, 0.0, 0.0, 0.35, 0.70, 0.70, 0.70])

    smoothed, rejected = _global_smooth_arm_path(
        raw,
        np.full(raw.shape[0], 0.001, dtype=np.float64),
        _joint_ranges(),
        np.zeros(14, dtype=np.float64),
        velocity_weight=_DEFAULT_VELOCITY_WEIGHT,
        acceleration_weight=_DEFAULT_ACCELERATION_WEIGHT,
        max_joint_step=_DEFAULT_MAX_JOINT_STEP,
    )

    assert not np.any(rejected)
    # A reachable transition faster than the old 0.05 rad/frame limit must not
    # be flattened into a lagging arm path that later fails its palm FK audit.
    assert np.max(np.abs(np.diff(smoothed[:, 0]))) > 0.20
    assert np.max(np.abs(smoothed[:, 0] - raw[:, 0])) < 0.10
    assert np.max(np.abs(np.diff(smoothed, axis=0))) <= (
        _DEFAULT_MAX_JOINT_STEP + 1e-10
    )


def test_all_unreachable_frames_fall_back_to_a_finite_stationary_plan() -> None:
    fallback = np.linspace(-0.3, 0.3, 14, dtype=np.float64)
    smoothed, rejected = _global_smooth_arm_path(
        np.ones((5, 14), dtype=np.float64),
        np.full(5, np.inf, dtype=np.float64),
        _joint_ranges(),
        fallback,
        velocity_weight=1.5,
        acceleration_weight=24.0,
        max_joint_step=0.05,
    )

    assert np.all(rejected)
    np.testing.assert_allclose(smoothed, np.tile(fallback, (5, 1)), atol=1e-8)


def test_global_smoother_rejects_invalid_error_threshold() -> None:
    with pytest.raises(ValueError, match="max_ik_error"):
        _global_smooth_arm_path(
            np.zeros((2, 14), dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            _joint_ranges(),
            np.zeros(14, dtype=np.float64),
            velocity_weight=1.5,
            acceleration_weight=24.0,
            max_joint_step=0.05,
            max_ik_error=0.0,
        )


def test_global_smoother_rejects_the_environment_failure_boundary() -> None:
    _, rejected = _global_smooth_arm_path(
        np.zeros((3, 14), dtype=np.float64),
        np.asarray([0.01, 0.08, -0.01], dtype=np.float64),
        _joint_ranges(),
        np.zeros(14, dtype=np.float64),
        velocity_weight=1.5,
        acceleration_weight=24.0,
        max_joint_step=0.05,
        max_ik_error=0.08,
    )

    np.testing.assert_array_equal(rejected, [False, True, True])


def test_fk_postcheck_uses_final_pose_errors_not_pruning_history() -> None:
    position_errors = np.asarray(
        [
            [0.010, 0.020],
            [0.049, 0.010],
            [0.051, 0.010],
        ],
        dtype=np.float64,
    )
    orientation_errors = np.asarray(
        [
            [0.10, 0.20],
            [0.49, 0.10],
            [0.10, 0.51],
        ],
        dtype=np.float64,
    )

    valid = _fk_valid_frames(
        position_errors,
        orientation_errors,
        max_position_error=0.05,
        max_orientation_error=0.50,
    )

    # The second row can represent a pruned/interpolated frame: it remains
    # valid because its final smoothed FK pose reaches the recorded palms.
    np.testing.assert_array_equal(valid, [True, True, False])


def test_quaternion_error_is_sign_invariant_and_reports_half_turn() -> None:
    identity = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(
        _quaternion_angular_error(identity, -identity),
        0.0,
        atol=1e-12,
    )
    assert _quaternion_angular_error(
        identity,
        np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
    ) == pytest.approx(np.pi)
    assert np.isinf(_quaternion_angular_error(identity, np.zeros(4)))


def test_collision_postcheck_allows_only_configured_penetration() -> None:
    valid = _collision_valid_frames(
        np.asarray([np.inf, -0.001, -0.003, np.inf]),
        np.asarray([np.inf, np.inf, np.inf, np.nan]),
        np.asarray([np.inf, -0.0015, np.inf, np.inf]),
        penetration_tolerance=0.002,
    )

    np.testing.assert_array_equal(valid, [True, True, False, False])


def test_invalid_postcheck_fails_before_render_with_a_clear_summary(tmp_path) -> None:
    audit = {
        "valid": np.asarray([True, False, True]),
        "position_error": np.asarray(
            [[0.01, 0.01], [0.08, 0.02], [0.01, 0.01]], dtype=np.float64
        ),
        "orientation_error": np.asarray(
            [[0.1, 0.1], [0.2, 0.6], [0.1, 0.1]], dtype=np.float64
        ),
        "arm_table_min_distance": np.asarray([np.inf, -0.003, np.inf]),
        "inter_arm_min_distance": np.full(3, np.inf),
        "self_collision_min_distance": np.full(3, np.inf),
    }

    with pytest.raises(RuntimeError, match="rendering is blocked") as exc_info:
        _raise_for_invalid_postcheck(
            tmp_path / "replay.zarr",
            audit,
            max_fk_position_error=0.05,
            max_fk_orientation_error=0.50,
            collision_penetration_tolerance=0.002,
        )

    message = str(exc_info.value)
    assert "1/3 frame(s)" in message
    assert "arm/table penetration=0.003000 m" in message


def test_renderer_refuses_failed_or_truncated_audit_before_opening_writers() -> None:
    with pytest.raises(RuntimeError, match="failed post-FK/collision"):
        _require_validated_arm_plan(
            {"c01_arm_plan_valid_frame": np.asarray([[1], [0], [1]])},
            3,
        )
    with pytest.raises(RuntimeError, match="2 rows for 3 render frames"):
        _require_validated_arm_plan(
            {"c01_arm_plan_valid_frame": np.asarray([[1], [1]])},
            3,
        )

    _require_validated_arm_plan(
        {"c01_arm_plan_valid_frame": np.asarray([[1], [1], [1]])},
        3,
    )
