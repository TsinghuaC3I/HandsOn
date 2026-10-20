from __future__ import annotations

import types

import pytest

from handson.sim.isaaclab.abc_collection_env import (
    _A03_WUJI2_HAND_COMMAND_DEADBAND_RAD,
    _A03_WUJI2_HAND_FILTER_CUTOFF_HZ,
    _A03_WUJI2_HAND_OTHER_EFFORT_LIMIT,
    _A03_WUJI2_HAND_RATE_HZ,
    _A03_WUJI2_HAND_SMOOTHER_DAMPING,
    _A03_WUJI2_HAND_SMOOTHER_HZ,
    _A03_WUJI2_HAND_SMOOTHER_STIFFNESS,
    _A03_WUJI2_HAND_SMOOTHER_W,
    _A03_WUJI2_HAND_SMOOTHER_Z,
    _A03_WUJI2_HAND_STATE_COMMAND_RATE_HZ,
    _A03_WUJI2_HAND_THUMB_GRASP_EFFORT_LIMIT,
    _B02_COLLECTION_SAFE_PHYSICS_DT,
    _B02_DEFAULT_PHYSICS_DT,
    _B02_MUJOCO_PHYSICS_DT,
    _HANDSON_SMOOTH_GRASP_HAND_JOINT_DRIVES,
    _G05_HAND_CONTACT_MARGIN,
    _smooth_grasp_expected_hand_collider_count,
    _smooth_grasp_hand_drive_maps,
    _smooth_grasp_physics_dt,
    _smooth_grasp_profile_enabled,
    HandsOnABCIsaacLabEnv,
    _physx_scene_ccd_enabled,
)


@pytest.mark.parametrize(
    "task_name",
    (
        "t04_cover_pitcher",
        "t09_insert_peg",
        "t03_cover_box",
        "t23_photograph_logo",

    ),
)
def test_physx_bc_tasks_restore_legacy_120hz_timing(
    monkeypatch: pytest.MonkeyPatch,
    task_name: str,
) -> None:
    for name in (
        "HANDSON_DISABLE_SMOOTH_GRASP_PROFILE",
        "HANDSON_SMOOTH_GRASP_PHYSICS_HZ",
        "HANDSON_SMOOTH_GRASP_SMALL_OBJECT_PHYSICS_HZ",
        "HANDSON_B02_PHYSICS_HZ",
    ):
        monkeypatch.delenv(name, raising=False)

    env = HandsOnABCIsaacLabEnv(
        task_name,
        physics_backend="physx",
        launch_isaac=False,
        image_obs=False,
        hz=30,
    )

    assert env.control_dt == pytest.approx(0.02)
    assert env.physics_dt == pytest.approx(1.0 / 120.0)
    assert env._n_substeps == 2
    assert env._smooth_grasp_hand_target_interpolation_enabled is False
    assert env._smooth_grasp_soft_hand_enabled() is False


@pytest.mark.parametrize(
    "task_name",
    (
        "t25_place_cup_on_plate",
        "t42_sort_objects",


    ),
)
def test_wuji1_non_bc_tasks_preserve_legacy_replay_timing(
    task_name: str,
) -> None:
    env = HandsOnABCIsaacLabEnv(
        task_name,
        hand_type="wuji1",
        physics_backend="physx",
        launch_isaac=False,
        image_obs=False,
        hz=30,
    )

    assert env.control_dt == pytest.approx(0.02)
    assert env.physics_dt == pytest.approx(1.0 / 120.0)
    assert env._n_substeps == 2
    assert env._smooth_grasp_hand_target_interpolation_enabled is False
    assert env._smooth_grasp_soft_hand_enabled() is False
    contract = env.get_physical_replay_reset_contract()
    assert contract["control_dt"] == pytest.approx(0.02)
    assert contract["control_substeps"] == 2
    assert contract["physx"]["smooth_grasp_hand_target_interpolation"] == {
        "enabled": False,
        "scope": "disabled_pre_global_profile",
        "mode": "linear_previous_applied_target_to_control_goal",
        "samples_per_control_step": 2,
        "final_substep_reaches_control_goal": True,
        "recording": "actual_applied_per_substep_target",
        "replay": "exact_recorded_pre_substep_targets_bypass_interpolation",
    }


@pytest.mark.parametrize(
    "task_name",
    (
        "t25_place_cup_on_plate",
        "t42_sort_objects",


    ),
)
def test_wuji2_non_bc_tasks_also_restore_legacy_replay_timing(
    task_name: str,
) -> None:
    env = HandsOnABCIsaacLabEnv(
        task_name,
        hand_type="wuji2",
        physics_backend="physx",
        launch_isaac=False,
        image_obs=False,
        hz=30,
    )

    assert env.control_dt == pytest.approx(0.02)
    assert env.physics_dt == pytest.approx(1.0 / 120.0)
    assert env._n_substeps == 2
    assert env._smooth_grasp_hand_target_interpolation_enabled is False
    assert env._smooth_grasp_soft_hand_enabled() is False
    contract = env.get_physical_replay_reset_contract()
    assert contract["physx"]["smooth_grasp_hand_target_interpolation"][
        "scope"
    ] == "disabled_pre_global_profile"


def test_a03_wuji2_maps_driver_effort_to_physx_actuator() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t24_place_ball_in_bowl",
        hand_type="wuji2",
        physics_backend="physx",
        launch_isaac=False,
        image_obs=False,
        hz=30,
    )

    for side in ("left", "right"):
        prefix = "l" if side == "left" else "r"
        actuator = types.SimpleNamespace(
            stiffness=None,
            damping=None,
            effort_limit_sim=None,
            velocity_limit_sim=1.2,
        )
        robot_cfg = types.SimpleNamespace(actuators={"hand": actuator})
        assert env._apply_a03_wuji2_hand_parameters(robot_cfg, side)
        assert len(actuator.stiffness) == len(actuator.damping) == 20
        assert len(actuator.effort_limit_sim) == 20
        assert actuator.stiffness[f"{prefix}_thumb_cmc_flex"] == pytest.approx(
            _A03_WUJI2_HAND_SMOOTHER_STIFFNESS
        )
        assert actuator.damping[f"{prefix}_index_finger_mcp_abd"] == pytest.approx(
            _A03_WUJI2_HAND_SMOOTHER_DAMPING
        )
        assert actuator.effort_limit_sim[f"{prefix}_thumb_cmc_flex"] == pytest.approx(
            _A03_WUJI2_HAND_OTHER_EFFORT_LIMIT
        )
        assert actuator.effort_limit_sim[f"{prefix}_thumb_cmc_abd"] == pytest.approx(
            _A03_WUJI2_HAND_THUMB_GRASP_EFFORT_LIMIT
        )
        assert actuator.effort_limit_sim[f"{prefix}_thumb_mcp"] == pytest.approx(
            _A03_WUJI2_HAND_THUMB_GRASP_EFFORT_LIMIT
        )
        assert actuator.effort_limit_sim[f"{prefix}_thumb_ip"] == pytest.approx(
            _A03_WUJI2_HAND_THUMB_GRASP_EFFORT_LIMIT
        )
        assert actuator.velocity_limit_sim is None
        assert env._a03_wuji2_hand_driver_parameters == {
            "rate_hz": _A03_WUJI2_HAND_RATE_HZ,
            "state_command_rate_hz": _A03_WUJI2_HAND_STATE_COMMAND_RATE_HZ,
            "filter_cutoff_hz": _A03_WUJI2_HAND_FILTER_CUTOFF_HZ,
            "thumb_grasp_effort_limit": _A03_WUJI2_HAND_THUMB_GRASP_EFFORT_LIMIT,
            "other_effort_limit": _A03_WUJI2_HAND_OTHER_EFFORT_LIMIT,
            "command_deadband_rad": _A03_WUJI2_HAND_COMMAND_DEADBAND_RAD,
            "smoother_hz": _A03_WUJI2_HAND_SMOOTHER_HZ,
            "smoother_w": _A03_WUJI2_HAND_SMOOTHER_W,
            "smoother_z": _A03_WUJI2_HAND_SMOOTHER_Z,
        }


@pytest.mark.parametrize(
    ("task_name", "hand_type"),
    (
        ("t24_place_ball_in_bowl", "wuji1"),
        ("t25_place_cup_on_plate", "wuji2"),
    ),
)
def test_a03_wuji2_driver_parameters_do_not_leak_to_other_profiles(
    task_name: str,
    hand_type: str,
) -> None:
    env = HandsOnABCIsaacLabEnv(
        task_name,
        hand_type=hand_type,
        physics_backend="physx",
        launch_isaac=False,
        image_obs=False,
        hz=30,
    )
    actuator = types.SimpleNamespace(
        stiffness=None,
        damping=None,
        effort_limit_sim=None,
        velocity_limit_sim=None,
    )
    robot_cfg = types.SimpleNamespace(actuators={"hand": actuator})

    assert not env._apply_a03_wuji2_hand_parameters(robot_cfg, "right")
    assert actuator.stiffness is None
    assert actuator.damping is None
    assert actuator.effort_limit_sim is None
    assert actuator.velocity_limit_sim is None


def test_b03_newton_retains_qualified_240hz_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HANDSON_ISAACLAB_DIRECT_HAND_CONTROL", "0")

    env = HandsOnABCIsaacLabEnv(
        "t09_insert_peg",
        physics_backend="newton",
        launch_isaac=False,
        image_obs=False,
        hz=30,
    )

    assert env.control_dt == pytest.approx(1.0 / 30.0)
    assert env.physics_dt == pytest.approx(1.0 / 240.0)
    assert env._n_substeps == 8
    assert env._smooth_grasp_hand_target_interpolation_enabled is False
    assert env._newton_hand_target_slew_enabled is True


def test_non_b02_task_preserves_explicit_caller_physics_dt() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t04_cover_pitcher",
        physics_backend="physx",
        physics_dt=1.0 / 1000.0,
        launch_isaac=False,
        image_obs=False,
    )

    assert env.physics_dt == pytest.approx(1.0 / 1000.0)
    assert env._n_substeps == 20


def test_wuji2_smooth_grasp_uses_official_joint_names_and_effort_limits() -> None:
    stiffness, damping, effort = _smooth_grasp_hand_drive_maps(
        "right",
        "wuji2",
    )

    assert len(stiffness) == len(damping) == len(effort) == 20
    assert stiffness["r_thumb_cmc_flex"] == pytest.approx(
        0.40844710645048043
    )
    assert damping["r_index_finger_mcp_abd"] == pytest.approx(
        0.019798167597016643
    )
    assert effort["r_thumb_cmc_flex"] == pytest.approx(0.6)
    assert effort["r_index_finger_mcp_flex"] == pytest.approx(2.0)
    assert effort["r_pinky_dip"] == pytest.approx(0.3)
    assert _smooth_grasp_expected_hand_collider_count("wuji1") == 52
    assert _smooth_grasp_expected_hand_collider_count("wuji2") == 40


def test_wuji1_smooth_grasp_drive_map_is_byte_order_compatible() -> None:
    stiffness, damping, effort = _smooth_grasp_hand_drive_maps(
        "right",
        "wuji1",
    )
    expected_items = tuple(_HANDSON_SMOOTH_GRASP_HAND_JOINT_DRIVES.items())
    expected_names = tuple(f"right_{suffix}" for suffix, _ in expected_items)

    assert tuple(stiffness) == expected_names
    assert tuple(damping) == expected_names
    assert tuple(effort) == expected_names
    assert stiffness == {
        name: values[0]
        for name, (_, values) in zip(expected_names, expected_items, strict=True)
    }
    assert damping == {
        name: values[1]
        for name, (_, values) in zip(expected_names, expected_items, strict=True)
    }
    assert effort == {
        name: values[2]
        for name, (_, values) in zip(expected_names, expected_items, strict=True)
    }


@pytest.mark.parametrize(
    "task_name",
    (
        "t25_place_cup_on_plate",
        "t44_stack_blocks",
        "t23_photograph_logo",
        "t42_sort_objects",


    ),
)
def test_global_smooth_grasp_profile_is_disabled_for_both_hand_generations(
    task_name: str,
) -> None:
    assert _smooth_grasp_profile_enabled(task_name, "physx", "wuji1") is False
    assert _smooth_grasp_profile_enabled(task_name, "physx", "wuji2") is False
    assert _smooth_grasp_profile_enabled(task_name, "newton") is False


def test_only_b02_keeps_the_legacy_soft_hand_profile() -> None:
    assert _smooth_grasp_profile_enabled(
        "b02_nut_bolt_assembly",
        "physx",
        "wuji1",
    ) is True
    assert _smooth_grasp_profile_enabled(
        "b02_nut_bolt_assembly",
        "physx",
        "wuji2",
    ) is True
    assert _smooth_grasp_profile_enabled(
        "e01_fold_garment",
        "physx",
        "wuji1",
    ) is False
    assert _smooth_grasp_profile_enabled(
        "e01_fold_garment",
        "physx",
        "wuji2",
    ) is False
    assert _smooth_grasp_physics_dt(
        "e01_fold_garment",
        "physx",
        1.0 / 60.0,
    ) == pytest.approx(1.0 / 60.0)
    assert _smooth_grasp_physics_dt(
        "e01_fold_garment",
        "physx",
        1.0 / 60.0,
        hand_type="wuji2",
    ) == pytest.approx(1.0 / 60.0)


def test_scene_ccd_restores_pre_global_profile_scope() -> None:
    assert _physx_scene_ccd_enabled("t04_cover_pitcher") is False
    assert _physx_scene_ccd_enabled("t09_insert_peg") is False
    assert _physx_scene_ccd_enabled("t23_photograph_logo") is False
    assert _physx_scene_ccd_enabled("c07_pitcher_pour_cup") is True
    assert _physx_scene_ccd_enabled("g05_folding_fan_bimanual_open") is False
    assert _physx_scene_ccd_enabled("t25_place_cup_on_plate") is False


def test_g05_thin_fan_uses_two_millimeter_predictive_hand_shell() -> None:
    assert _G05_HAND_CONTACT_MARGIN == pytest.approx(0.002)


def test_recorded_physx_substep_target_uses_exact_callback_without_interpolation() -> None:
    env = HandsOnABCIsaacLabEnv(
        "t04_cover_pitcher",
        physics_backend="physx",
        launch_isaac=False,
        image_obs=False,
    )
    interpolation_calls: list[tuple[int, int]] = []

    def recorded_target(_index: int, _count: int) -> None:
        # Exact replay deliberately leaves the live-goal serial unchanged.
        return None

    env._handson_pre_physics_substep_callback = recorded_target
    env._advance_smooth_grasp_hand_target_interpolation = (
        lambda index, count, **_kwargs: interpolation_calls.append((index, count))
    )

    assert env._prepare_physics_substep_joint_targets(3, 8) == (
        "exact_pre_substep_callback"
    )
    assert interpolation_calls == []
