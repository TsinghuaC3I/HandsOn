from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from handson.sim.isaaclab.p006_contract import (
    FEMALE_ASSET_SHA256,
    LATCH_DWELL_STEPS,
    MALE_ASSET_SHA256,
    MAX_PENETRATION_M,
    MAX_TRANSVERSE_OFFSET_M,
    MIN_INSERTION_DEPTH_M,
    MIN_ROLE_CONTACT_FORCE_N,
    P006LatchStateMachine,
    P006PhysicalReadback,
    SOURCE_STL_SHA256,
    SUCCESS_HOLD_STEPS,
    TASK_ID,
    TASK_NAME,
    physical_metrics,
    reset_is_valid,
    sample_reset_state,
)


TASK_DIR = Path(__file__).resolve().parent
ASSET_DIR = TASK_DIR / "assets" / "side_release_buckle_cc0"
REPO_ROOT = TASK_DIR.parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_live_runtime_reports_both_buckle_table_support_pairs() -> None:
    source = (
        REPO_ROOT
        / "handson/handson/sim/isaaclab/p006_collection_env.py"
    ).read_text(encoding="utf-8")
    compact = " ".join(source.split())
    assert (
        'self._create_contact_view( physics_view, "female_support", '
        "FEMALE_ROOT_PATH, [str(table.GetPath())] )"
    ) in compact
    assert (
        'self._create_contact_view( physics_view, "male_support", '
        "MALE_ROOT_PATH, [str(table.GetPath())] )"
    ) in compact
    validator = (TASK_DIR / "live_physics_acceptance.py").read_text(
        encoding="utf-8"
    )
    assert 'environment._p006_contact_paths.get("female_support")' in validator
    assert 'environment._p006_contact_paths.get("male_support")' in validator
    assert validator.count('{"sensor_count": 1, "filter_count": 1}') >= 2
    assert "support_pairs_exact" in validator


def test_exact_female_runtime_view_fills_reset_cache_when_name_scan_misses() -> None:
    """The referenced female root must still be resettable before teleop."""

    import torch

    from handson.sim.isaaclab import abc_collection_env
    from handson.sim.isaaclab.p006_collection_env import HandsOnP006IsaacLabEnv

    class _FakeRigidBodyView:
        count = 1

        def get_transforms(self):
            return torch.tensor(
                [[-0.18, 0.12, 0.892, 0.0, 0.0, 0.0, 1.0]],
                dtype=torch.float32,
            )

        def get_velocities(self):
            return torch.zeros((1, 6), dtype=torch.float32)

    environment = object.__new__(HandsOnP006IsaacLabEnv)
    environment._torch = torch
    environment._isaac_device = "cpu"
    environment._task_reset_body_states = []
    environment._p006_body_views = {"female": _FakeRigidBodyView()}
    environment._task_body_views = {}
    warnings: list[str] = []
    environment._add_task_state_warning = warnings.append

    # The generic capture is intentionally stubbed to reproduce the imported
    # USD case where the exact female root is not discovered by name.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        abc_collection_env.HandsOnABCIsaacLabEnv,
        "_capture_task_scene_state",
        lambda self, task_mjcf: None,
    )
    try:
        environment._capture_task_scene_state(Path("unused.xml"))
    finally:
        monkeypatch.undo()

    assert warnings == []
    assert [row[0] for row in environment._task_reset_body_states] == [
        "/World/Task/P006FemaleBuckle"
    ]
    assert environment._task_body_views["p006_female_buckle"] is environment._p006_body_views[
        "female"
    ]


def test_priority_zero_validator_samples_each_substep_before_telemetry_downsample() -> None:
    source = (TASK_DIR / "live_physics_acceptance.py").read_text(
        encoding="utf-8"
    )
    loop = source.split("for step in range(stability_steps):", 1)[1]
    every_frame, _telemetry = loop.split(
        "if step % 12 == 0 or step == stability_steps - 1:",
        1,
    )
    assert "readback = environment._physical_readback()" in every_frame
    assert '"female_support"' in every_frame
    assert '"male_support"' in every_frame
    assert "support_samples += 1" in every_frame
    assert "max_penetration = max(" in every_frame
    assert "max_support_penetration = max(" in every_frame
    assert "nan_count +=" in every_frame
    assert "max_female_translation_drift_m = max(" in every_frame
    assert "max_male_translation_drift_m = max(" in every_frame
    assert "max_female_rotation_drift_rad = max(" in every_frame
    assert "max_male_rotation_drift_rad = max(" in every_frame
    assert "max_female_translation_drift_m <= 0.005" in source
    assert "max_male_translation_drift_m <= 0.005" in source
    assert "max_female_rotation_drift_rad <= math.radians(3.0)" in source
    assert "max_male_rotation_drift_rad <= math.radians(3.0)" in source
    assert "supported_samples == support_samples" in source
    assert "static_pose_covered_steps += int(frame_static_pose)" in every_frame
    assert "PRIORITY_ZERO_LINEAR_SPEED_LIMIT_MPS = 0.001" in source
    assert "PRIORITY_ZERO_ANGULAR_SPEED_LIMIT_RADPS = 0.002" in source
    assert "STATIC_POSE_TRANSLATION_STEP_LIMIT_M = 1.0e-6" in source
    assert "STATIC_POSE_ROTATION_STEP_LIMIT_RAD = 1.0e-5" in source
    assert "velocity_or_static_pose_pass" in source
    assert "PRIORITY_ZERO_MAX_PENETRATION_M = 0.0005" in source
    assert "int(args.reset_trials) >= 20" in source
    assert '"initial_zero_velocity_write_performed": False' in source
    assert '"final_zero_velocity_write_performed": False' in source
    assert "allow_nan=False" in source


def test_priority_zero_reset_checks_both_raw_buckle_velocities_without_step() -> None:
    source = (TASK_DIR / "live_physics_acceptance.py").read_text(
        encoding="utf-8"
    )
    reset_loop = source.split("for seed in range(int(args.reset_trials)):", 1)[1]
    reset_loop = reset_loop.split("reachability: list[dict[str, Any]]", 1)[0]
    reset_gate = source.split("reset_pass = bool(", 1)[1].split(
        "reachability_pass = bool(", 1
    )[0]
    assert "sim.step(" not in reset_loop
    for field in (
        '"female_linear_speed_mps"',
        '"female_angular_speed_radps"',
        '"male_linear_speed_mps"',
        '"male_angular_speed_radps"',
    ):
        assert field in reset_loop
        assert f"row[{field}]" in reset_gate


def _quat_z(degrees: float) -> tuple[float, float, float, float]:
    half = math.radians(degrees) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _readback(**overrides) -> P006PhysicalReadback:
    values = {
        "female_position_m": (0.0, 0.0, 1.0),
        "female_quaternion_wxyz": (1.0, 0.0, 0.0, 0.0),
        "male_position_m": (0.065, 0.0, 1.0),
        "male_quaternion_wxyz": (1.0, 0.0, 0.0, 0.0),
        "female_linear_velocity_mps": (0.0, 0.0, 0.0),
        "female_angular_velocity_radps": (0.0, 0.0, 0.0),
        "male_linear_velocity_mps": (0.0, 0.0, 0.0),
        "male_angular_velocity_radps": (0.0, 0.0, 0.0),
        "right_female_contact_force_n": MIN_ROLE_CONTACT_FORCE_N + 0.1,
        "left_male_contact_force_n": MIN_ROLE_CONTACT_FORCE_N + 0.1,
        "buckle_body_contact_force_n": 0.2,
        "robot_table_contact_force_n": 0.0,
        "max_penetration_m": 0.0,
    }
    values.update(overrides)
    return P006PhysicalReadback(**values)


def _reports_success(readback: P006PhysicalReadback, steps: int = 30) -> bool:
    machine = P006LatchStateMachine()
    return any(bool(machine.update(readback)["success_now"]) for _ in range(steps))


def test_authored_reset_pose_gate_catches_z_and_roll_errors() -> None:
    from handson.sim.isaaclab.p006_collection_env import HandsOnP006IsaacLabEnv

    environment = object.__new__(HandsOnP006IsaacLabEnv)
    environment._p006_reset = sample_reset_state(7)
    reset = environment._p006_reset
    exact = _readback(
        female_position_m=reset.female_position_m,
        female_quaternion_wxyz=_quat_z(reset.female_yaw_deg),
        male_position_m=reset.male_position_m,
        male_quaternion_wxyz=_quat_z(reset.male_yaw_deg),
    )
    assert environment._authored_reset_pose_report(exact)["passed"] is True

    female_position = list(reset.female_position_m)
    female_position[2] += 0.005001
    shifted = P006PhysicalReadback(
        **{
            **exact.__dict__,
            "female_position_m": tuple(female_position),
        }
    )
    assert environment._authored_reset_pose_report(shifted)["passed"] is False

    half_roll = math.radians(3.001) * 0.5
    yaw = np.asarray(_quat_z(reset.male_yaw_deg), dtype=np.float64)
    roll = np.asarray([math.cos(half_roll), math.sin(half_roll), 0.0, 0.0])
    w1, x1, y1, z1 = yaw
    w2, x2, y2, z2 = roll
    yaw_roll = (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
    rolled = P006PhysicalReadback(
        **{**exact.__dict__, "male_quaternion_wxyz": yaw_roll}
    )
    assert environment._authored_reset_pose_report(rolled)["passed"] is False


def test_identity_and_no_pull_stage() -> None:
    assert TASK_ID == "P0-06"
    assert TASK_NAME == "p006_side_release_buckle_align_press"
    assert not hasattr(P006LatchStateMachine(), "pull_dwell_steps")


def test_exact_assets_and_auditable_usda() -> None:
    assert _sha256(ASSET_DIR / "model.stl") == SOURCE_STL_SHA256
    assert _sha256(ASSET_DIR / "female_buckle.usda") == FEMALE_ASSET_SHA256
    assert _sha256(ASSET_DIR / "male_buckle.usda") == MALE_ASSET_SHA256
    female = (ASSET_DIR / "female_buckle.usda").read_text(encoding="utf-8")
    male = (ASSET_DIR / "male_buckle.usda").read_text(encoding="utf-8")
    assert female.count('def Cube "Collision_') == 6
    assert male.count('def Cube "Collision_') == 7
    assert male.count('def Mesh "Collision_LatchBump"') == 2
    assert "PhysicsRigidBodyAPI" in female and "PhysicsRigidBodyAPI" in male
    assert "PhysicsMassAPI" in female and "PhysicsMassAPI" in male
    assert "PhysicsArticulationRootAPI" in male
    assert 'def Xform "OuterTongueNeg"' in male
    assert 'def Xform "OuterTonguePos"' in male
    assert 'def PhysicsPrismaticJoint "OuterTongueNegJoint"' in male
    assert 'def PhysicsPrismaticJoint "OuterTonguePosJoint"' in male
    assert '"!resetXformStack!", "xformOp:translate"' in male
    assert male.count('Collision_LatchBump') == 2


def test_one_thousand_seeded_resets_are_valid_and_reproducible() -> None:
    samples = [sample_reset_state(seed) for seed in range(1000)]
    assert all(reset_is_valid(sample) for sample in samples)
    assert samples == [sample_reset_state(seed) for seed in range(1000)]


def test_buckles_use_strict_every_frame_static_pose_proof() -> None:
    source = (
        REPO_ROOT
        / "handson/handson/sim/isaaclab/recording_origin.py"
    ).read_text(encoding="utf-8")
    assert '"p006_side_release_buckle_align_press": (1.0e-6, 1.0e-5)' in source


def test_positive_contract_requires_latch_and_stable_dwells() -> None:
    machine = P006LatchStateMachine()
    first_success_step = None
    for step in range(LATCH_DWELL_STEPS + SUCCESS_HOLD_STEPS + 3):
        if machine.update(_readback())["success_now"] and first_success_step is None:
            first_success_step = step + 1
    assert first_success_step == LATCH_DWELL_STEPS + SUCCESS_HOLD_STEPS - 1
    assert machine.latched_seen


@pytest.mark.parametrize(
    "overrides",
    [
        {"right_female_contact_force_n": 0.0},
        {"left_male_contact_force_n": 0.0},
        {"buckle_body_contact_force_n": 0.0},
        {"male_position_m": (0.080, 0.0, 1.0)},
        {"male_position_m": (0.065, MAX_TRANSVERSE_OFFSET_M + 0.001, 1.0)},
        {"male_quaternion_wxyz": _quat_z(13.0)},
        {"max_penetration_m": MAX_PENETRATION_M + 0.001},
        {"male_linear_velocity_mps": (0.13, 0.0, 0.0)},
        {"male_angular_velocity_radps": (0.0, 0.0, 1.1)},
    ],
)
def test_negative_contracts_never_report_success(overrides) -> None:
    assert not _reports_success(_readback(**overrides))


def test_metrics_match_frozen_geometry() -> None:
    metrics = physical_metrics(_readback())
    assert metrics["aligned"]
    assert metrics["deep_enough"]
    assert metrics["body_contact"]
    assert metrics["insertion_depth_m"] >= MIN_INSERTION_DEPTH_M


def test_nan_and_excessive_penetration_are_rejected() -> None:
    nan_metrics = physical_metrics(_readback(male_position_m=(math.nan, 0.0, 1.0)))
    assert not nan_metrics["finite"]
    assert not _reports_success(_readback(max_penetration_m=MAX_PENETRATION_M + 1e-6))


def test_normal_table_contact_does_not_block_completion() -> None:
    assert _reports_success(_readback(robot_table_contact_force_n=5.0))
