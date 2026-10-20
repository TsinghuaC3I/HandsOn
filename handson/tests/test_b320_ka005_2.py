from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from handson.runtime.collection_support import (
    ISAACSIM_COLLECTION_TASKS,
    ISAACSIM_COLLECTION_UNVERIFIED_TASKS,
    ISAACSIM_FORMAL_COLLECTION_TASKS,
    ISAACSIM_NONSTANDARD_DEFERRED_RENDER_TASKS,
)
from handson.sim.isaaclab import b320_bimanual_env as b320_env_module
from handson.sim.isaaclab.b320_bimanual_env import (
    HandsOnB320BimanualIsaacLabEnv,
)
from handson.sim.isaaclab.b320_bimanual_scene import (
    B320_KA005_2_ASSET_DIRECTORY,
    B320_KA005_2_TASK_NAME,
    B320_TASK_SPECS,
    validate_b320_ka005_2_asset_files,
)
from handson.sim.isaaclab import b320_ka005_2_contract as knob_contract
from handson.sim.isaaclab.b320_ka005_2_contract import (
    CONTACT_OFF_N,
    CONTACT_ON_N,
    DualKnobSyncState,
    KnobStage,
    sample_dual_knob_reset,
)
from handson.sim.isaaclab.recording_origin import (
    RecordingOriginStabilityError,
    _KNOWN_REVOLUTE_TASK_DOF_NAMES,
)
from handson.tasks.d_long_horizon_config import DLongHorizonTaskConfigBase
from handson.tasks.mappings import CONFIG_MAPPING


DT = 1.0 / 120.0
ASSIGNED = {"LL": 0.20, "RR": 0.20, "LR": 0.0, "RL": 0.0}
RELEASED = {key: 0.0 for key in ("LL", "RR", "LR", "RL")}


def _step(
    state: DualKnobSyncState,
    q: np.ndarray,
    qd: np.ndarray,
    forces: dict[str, float],
    count: int = 1,
) -> None:
    for _ in range(count):
        state.update(
            q,
            qd,
            forces,
            dof_available=True,
            contact_available=True,
            dt=DT,
        )


def _successful_state(*, release: bool = True) -> DualKnobSyncState:
    state = DualKnobSyncState()
    target = np.asarray([math.radians(60.0), math.radians(-60.0)])
    state.reset((0.0, 0.0), target)
    q = np.zeros(2, dtype=np.float64)
    _step(state, q, np.zeros(2), RELEASED, 31)
    _step(state, q, np.zeros(2), ASSIGNED, 1)
    steps = 121
    previous = q.copy()
    for index in range(1, steps + 1):
        q = target * (index / steps)
        qd = (q - previous) / DT
        _step(state, q, qd, ASSIGNED)
        previous = q.copy()
    _step(state, target, np.zeros(2), ASSIGNED, 38)
    if release:
        _step(state, target, np.zeros(2), RELEASED, 95)
    return state


def test_identity_registration_and_runtime_gate_are_independent() -> None:
    config = CONFIG_MAPPING[B320_KA005_2_TASK_NAME]
    spec = B320_TASK_SPECS[B320_KA005_2_TASK_NAME]
    assert (config.task_id, config.task_name) == (
        "38",
        B320_KA005_2_TASK_NAME,
    )
    assert spec.label.endswith("Setting-2")
    assert spec.fixture_kind == "dual_knob_box_2"
    assert spec.left_role == "knob_left"
    assert spec.right_role == "knob_right"
    assert spec.dof_names == ("left_knob_joint", "right_knob_joint")
    assert B320_KA005_2_TASK_NAME in ISAACSIM_COLLECTION_TASKS
    assert B320_KA005_2_TASK_NAME not in ISAACSIM_COLLECTION_UNVERIFIED_TASKS
    assert B320_KA005_2_TASK_NAME in ISAACSIM_FORMAL_COLLECTION_TASKS
    # KA005-2 must keep both articulated ROKAE arms live.  Hiding the
    # imported arm Gprims before the first SimulationContext.reset() poisons
    # PhysX GPU narrowphase (CUDA error 700) for this fixed-base two-DOF
    # fixture.  Canonical cameras remain deferred independently.
    assert B320_KA005_2_TASK_NAME not in ISAACSIM_NONSTANDARD_DEFERRED_RENDER_TASKS


def test_teleop_profile_matches_d08_exactly() -> None:
    config = CONFIG_MAPPING[B320_KA005_2_TASK_NAME]
    d08_config = CONFIG_MAPPING["t12_load_and_start_microwave"]

    assert config.teleop is DLongHorizonTaskConfigBase.teleop
    assert config.teleop == d08_config.teleop
    assert config.teleop.filter_alpha == pytest.approx(0.55)
    assert config.teleop == d08_config.teleop
    assert config.teleop.tracker_jump_translation_m == pytest.approx(0.12)
    assert config.teleop.tracker_jump_rotation_deg == pytest.approx(35.0)
    assert config.teleop.fail_closed_rebase_on_tracker_jump is True
    assert config.teleop.require_stable_pair_on_enable is False


def test_recording_origin_classifies_both_knob_dofs_as_radians() -> None:
    assert {"left_knob_joint", "right_knob_joint"} <= set(
        _KNOWN_REVOLUTE_TASK_DOF_NAMES
    )


def test_sampled_knob_reset_is_the_authored_origin_dof_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MetaType:
        dof_indices = {"left_knob_joint": 0, "right_knob_joint": 1}

    class _View:
        def __init__(self) -> None:
            self.root = np.asarray(
                [[-0.14, 0.0, 0.915, 0.0, 0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            self.root_velocity = np.zeros((1, 6), dtype=np.float32)
            self.q = np.zeros((1, 2), dtype=np.float32)
            self.qd = np.ones((1, 2), dtype=np.float32)
            self.targets = np.zeros((1, 2), dtype=np.float32)

        def get_root_transforms(self):
            return self.root.copy()

        def get_root_velocities(self):
            return self.root_velocity.copy()

        def get_dof_positions(self):
            return self.q.copy()

        def get_dof_velocities(self):
            return self.qd.copy()

        def get_metatype(self, _index: int):
            return _MetaType()

        def set_dof_positions(self, values, _indices) -> None:
            self.q = np.asarray(values, dtype=np.float32).copy()

        def set_dof_velocities(self, values, _indices) -> None:
            self.qd = np.asarray(values, dtype=np.float32).copy()

        def set_dof_position_targets(self, values, _indices) -> None:
            self.targets = np.asarray(values, dtype=np.float32).copy()

    monkeypatch.setattr(
        b320_env_module,
        "_array_like",
        lambda _reference, values, _torch: np.asarray(
            values,
            dtype=np.float32,
        ).copy(),
    )
    env = HandsOnB320BimanualIsaacLabEnv(
        B320_KA005_2_TASK_NAME,
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    env.reset(seed=20260823)
    view = _View()
    path = "/World/Task/B320Benchmark/panel_base/world_fixed_joint"
    env._task_reset_articulation_states = [
        (
            path,
            view,
            np.asarray([0], dtype=np.int32),
            view.root.copy(),
            view.root_velocity.copy(),
            view.q.copy(),
            view.qd.copy(),
        )
    ]
    env._recording_origin_reset_write_capture_active = True
    env._recording_origin_authored_task_reset_writes = {}

    env._set_articulation_reset(
        view,
        np.asarray([0], dtype=np.int32),
        fixture=True,
    )

    expected_q = np.asarray(
        env._ka005_2_reset_sample.initial_rad,
        dtype=np.float64,
    )
    np.testing.assert_allclose(view.q.reshape(-1), expected_q, atol=1e-7)
    np.testing.assert_array_equal(view.qd, 0.0)
    np.testing.assert_allclose(view.targets.reshape(-1), expected_q, atol=1e-7)
    journal = env._recording_origin_authored_task_reset_writes[path]
    np.testing.assert_allclose(journal["dof_position"], expected_q, atol=1e-7)
    np.testing.assert_array_equal(journal["dof_velocity"], 0.0)


def test_cleanroom_task_modules_do_not_import_the_prior_task() -> None:
    root = Path(__file__).resolve().parents[2]
    dedicated_files = (
        root
        / "handson/handson/tasks/t38_set_dual_knobs/config.py",
        root / "handson/handson/sim/isaaclab/b320_ka005_2_contract.py",
    )
    forbidden = "synchronized_dual_knob_setting" + ".config"
    for path in dedicated_files:
        assert forbidden not in path.read_text(encoding="utf-8")
    preview_path = root / "handson/handson/sim/isaaclab/task_preview.py"
    preview_tree = ast.parse(preview_path.read_text(encoding="utf-8"))
    top_level_imports = {
        node.module
        for node in preview_tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert "dual_knob_panel" not in top_level_imports
    assert ".dual_knob_panel" not in top_level_imports


def test_downloaded_asset_closure_and_geometry_manifest() -> None:
    required = validate_b320_ka005_2_asset_files()
    assert len(required) == 12
    manifest = json.loads(
        (B320_KA005_2_ASSET_DIRECTORY / "asset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["clean_room"] is True
    rotary = manifest["sources"]["rotary_knob"]
    assert rotary["license"] == "CC-BY-4.0"
    assert rotary["selected_nodes"] == [
        "Knob base_0",
        "Grip_1",
        "Top face_2",
        "Index line_3",
    ]
    assert rotary["geometry"]["runtime_axis"] == "+Z"
    assert rotary["geometry"]["target_grip_diameter_m"] == pytest.approx(0.060)
    moving = rotary["geometry"]["moving_bounds_m"]
    assert moving["max"][2] == pytest.approx(0.048239, abs=1.0e-6)
    assert manifest["sources"]["project_boxy"]["license"] == "MIT"
    assert manifest["sources"]["project_boxy"]["geometry"][
        "runtime_dimensions_m"
    ] == [0.21, 0.49, 0.05]
    assert manifest["physics"]["visual_files_contain_physics_schemas"] is False


def test_seeded_reset_sampler_is_deterministic_and_bounded() -> None:
    first = sample_dual_knob_reset(np.random.RandomState(77), randomize=True)
    second = sample_dual_knob_reset(np.random.RandomState(77), randomize=True)
    assert first == second
    for seed in range(100):
        sample = sample_dual_knob_reset(np.random.RandomState(seed), randomize=True)
        initial = np.degrees(sample.initial_rad)
        target = np.degrees(sample.target_rad)
        travel = np.abs(target - initial)
        assert np.all(travel >= 45.0)
        assert np.all(travel <= 120.0)
        assert np.all(np.abs(target) <= 85.0)
        assert target[0] > initial[0]
        assert target[1] < initial[1]


def test_physical_positive_requires_ordered_release_and_stability() -> None:
    state = _successful_state()
    assert state.success is True
    assert state.stage == KnobStage.COMPLETE
    assert all(state.condition_flags())
    assert knob_contract._time_at_least(
        state.release_s,
        knob_contract.RELEASE_HOLD_S,
    )
    assert knob_contract._time_at_least(
        state.stable_s,
        knob_contract.STABLE_HOLD_S,
    )


def test_no_release_never_succeeds() -> None:
    state = _successful_state(release=False)
    assert state.target_hold_complete is True
    assert state.stage == KnobStage.WAIT_RELEASE
    assert state.success is False


def test_release_and_stability_windows_do_not_double_count_a_substep() -> None:
    state = _successful_state(release=False)
    target = np.asarray([math.radians(60.0), math.radians(-60.0)])
    _step(state, target, np.zeros(2), RELEASED, 29)
    assert state.stage == KnobStage.WAIT_RELEASE
    assert state.release_s == pytest.approx(29 * DT)
    _step(state, target, np.zeros(2), RELEASED, 1)
    assert state.stage == KnobStage.STABILIZE
    assert state.release_s == pytest.approx(0.25)
    assert state.stable_s == pytest.approx(0.0)
    _step(state, target, np.zeros(2), RELEASED, 59)
    assert state.stage == KnobStage.STABILIZE
    assert state.success is False
    assert state.stable_s == pytest.approx(59 * DT)
    _step(state, target, np.zeros(2), RELEASED, 1)
    assert state.stage == KnobStage.COMPLETE
    assert state.success is True
    assert state.stable_s == pytest.approx(0.50)


@pytest.mark.parametrize(
    "threshold_s",
    [
        knob_contract.REARM_RELEASE_S,
        knob_contract.TARGET_HOLD_S,
        knob_contract.RELEASE_HOLD_S,
        knob_contract.STABLE_HOLD_S,
    ],
)
def test_timing_windows_accept_exact_substeps_but_not_one_fewer(
    threshold_s: float,
) -> None:
    required_substeps = round(threshold_s / DT)
    one_short = sum(DT for _ in range(required_substeps - 1))
    exact = sum(DT for _ in range(required_substeps))
    assert knob_contract._time_at_least(one_short, threshold_s) is False
    assert knob_contract._time_at_least(exact, threshold_s) is True
    assert knob_contract._time_exceeds(exact, threshold_s) is False
    assert knob_contract._time_exceeds(exact + DT, threshold_s) is True


def test_rearm_release_window_requires_exactly_thirty_free_substeps() -> None:
    target = (math.radians(60.0), math.radians(-60.0))

    one_short = DualKnobSyncState()
    one_short.reset((0.0, 0.0), target)
    _step(one_short, np.zeros(2), np.zeros(2), RELEASED, 29)
    _step(one_short, np.zeros(2), np.zeros(2), ASSIGNED, 1)
    assert one_short.attempt_id == 0
    assert one_short.stage == KnobStage.WAIT_GRASP

    exact = DualKnobSyncState()
    exact.reset((0.0, 0.0), target)
    _step(exact, np.zeros(2), np.zeros(2), RELEASED, 30)
    _step(exact, np.zeros(2), np.zeros(2), ASSIGNED, 1)
    assert exact.attempt_id == 1
    assert exact.stage == KnobStage.WAIT_MOTION


def test_sequential_motion_can_complete_with_more_than_three_seconds_between_targets() -> None:
    state = DualKnobSyncState()
    target = np.asarray([math.radians(60.0), math.radians(-60.0)])
    state.reset((0.0, 0.0), target)
    q = np.zeros(2)
    _step(state, q, np.zeros(2), RELEASED, 31)
    left_only = {"LL": 0.20, "RR": 0.0, "LR": 0.0, "RL": 0.0}
    right_only = {"LL": 0.0, "RR": 0.20, "LR": 0.0, "RL": 0.0}
    _step(state, q, np.zeros(2), ASSIGNED)
    previous = q.copy()
    for index in range(1, 61):
        q = np.asarray([target[0] * index / 60.0, 0.0])
        qd = (q - previous) / DT
        _step(state, q, qd, left_only)
        previous = q.copy()
    _step(state, q, np.zeros(2), left_only, 362)
    previous = q.copy()
    for index in range(1, 61):
        q = np.asarray([target[0], target[1] * index / 60.0])
        qd = (q - previous) / DT
        _step(state, q, qd, right_only)
        previous = q.copy()
    _step(state, target, np.zeros(2), right_only, 38)
    _step(state, target, np.zeros(2), RELEASED, 95)
    assert state.success is True
    assert state.stage == KnobStage.COMPLETE
    assert state.target_entry_skew_s > 3.0


def test_motion_start_skew_is_allowed_for_sequential_operation() -> None:
    state = DualKnobSyncState()
    target = np.asarray([math.radians(60.0), math.radians(-60.0)])
    state.reset((0.0, 0.0), target)
    q = np.zeros(2, dtype=np.float64)
    _step(state, q, np.zeros(2), RELEASED, 30)
    _step(state, q, np.zeros(2), ASSIGNED)

    # One knob may start well before the other in the sequential workflow.
    for _ in range(126):
        q_next = q.copy()
        q_next[0] += 0.13 * DT
        _step(state, q_next, np.asarray([0.13, 0.0]), ASSIGNED)
        q = q_next
    assert 126 * DT > 1.0
    assert state.stage == KnobStage.TURNING
    assert state.attempt_invalid_reason == ""
    assert state.success is False


def test_assigned_contacts_can_occur_more_than_one_second_apart() -> None:
    state = DualKnobSyncState()
    target = np.asarray([math.radians(60.0), math.radians(-60.0)])
    state.reset((0.0, 0.0), target)
    q = np.zeros(2, dtype=np.float64)
    left_only = {"LL": 0.20, "RR": 0.0, "LR": 0.0, "RL": 0.0}
    right_only = {"LL": 0.0, "RR": 0.20, "LR": 0.0, "RL": 0.0}
    _step(state, q, np.zeros(2), RELEASED, 30)
    _step(state, q, np.zeros(2), left_only)
    assert state.attempt_id == 1
    assert state.stage == KnobStage.WAIT_MOTION
    assert state.dual_grasp_seen is False
    _step(state, q, np.zeros(2), left_only, 122)
    _step(state, q, np.zeros(2), right_only)
    assert state.contact_entry_skew_s > 1.0
    assert state.attempt_invalid_reason == ""
    assert state.dual_grasp_seen is True


def test_small_reverse_motion_does_not_invalidate_dominant_target_trajectory() -> None:
    state = DualKnobSyncState()
    target = np.asarray([math.radians(60.0), math.radians(-60.0)])
    state.reset((0.0, 0.0), target)
    q = np.zeros(2, dtype=np.float64)
    _step(state, q, np.zeros(2), RELEASED, 30)
    _step(state, q, np.zeros(2), ASSIGNED)

    waypoints = (
        np.asarray([math.radians(25.0), math.radians(-25.0)]),
        np.asarray([math.radians(21.0), math.radians(-21.0)]),
        target,
    )
    for waypoint, count in zip(waypoints, (45, 8, 70)):
        start = q.copy()
        for index in range(1, count + 1):
            q_next = start + (waypoint - start) * (index / count)
            _step(state, q_next, (q_next - q) / DT, ASSIGNED)
            q = q_next
    _step(state, target, np.zeros(2), ASSIGNED, 38)
    _step(state, target, np.zeros(2), RELEASED, 95)

    assert np.all(state.wrong_direction_progress > 0.0)
    assert state.attempt_invalid_reason == ""
    assert state.success is True


def test_cross_hand_contact_cannot_substitute_for_assigned_roles() -> None:
    state = DualKnobSyncState()
    state.reset((0.0, 0.0), (math.radians(60.0), math.radians(-60.0)))
    _step(state, np.zeros(2), np.zeros(2), RELEASED, 31)
    crossed = {"LL": 0.20, "RR": 0.20, "LR": 0.20, "RL": 0.0}
    _step(state, np.zeros(2), np.zeros(2), crossed, 5)
    assert state.attempt_id == 0
    assert state.stage == KnobStage.WAIT_GRASP
    assert state.success is False


def test_contact_hysteresis_and_missing_truth_fail_closed() -> None:
    state = DualKnobSyncState()
    state.update(
        (0.0, 0.0),
        (0.0, 0.0),
        {"LL": CONTACT_ON_N, "RR": CONTACT_ON_N, "LR": 0.0, "RL": 0.0},
        dof_available=True,
        contact_available=True,
        dt=DT,
    )
    assert state.contact_active["LL"] is True
    state.update(
        (0.0, 0.0),
        (0.0, 0.0),
        {"LL": CONTACT_OFF_N, "RR": CONTACT_OFF_N, "LR": 0.0, "RL": 0.0},
        dof_available=True,
        contact_available=True,
        dt=DT,
    )
    assert state.contact_active["LL"] is False
    state.update(
        (0.0, 0.0),
        (0.0, 0.0),
        RELEASED,
        dof_available=True,
        contact_available=False,
        dt=DT,
    )
    assert state.failure_reason == "contact_truth_unavailable"
    assert state.success is False


def test_first_terminal_failure_reason_is_preserved_across_substeps() -> None:
    state = DualKnobSyncState()
    state.update(
        (0.0, 0.0),
        (0.0, 0.0),
        RELEASED,
        dof_available=True,
        contact_available=False,
        dt=DT,
    )
    assert state.failure_reason == "contact_truth_unavailable"
    state.update(
        (0.0, 0.0),
        (0.0, 0.0),
        RELEASED,
        dof_available=False,
        contact_available=True,
        dt=DT,
    )
    assert state.failure_reason == "contact_truth_unavailable"


def test_recording_origin_truth_prime_does_not_advance_task_time() -> None:
    state = DualKnobSyncState()
    q = np.asarray([0.01, -0.02], dtype=np.float64)
    target = np.asarray([math.radians(60.0), math.radians(-60.0)])
    state.reset(q, target)

    state.prime_recording_origin_truth(q, np.zeros(2), RELEASED)

    assert state.sim_time_s == 0.0
    assert state.stage == KnobStage.WAIT_GRASP
    assert state.attempt_id == 0
    assert state.attempt_armed is False
    assert state.failure_reason == ""
    assert state.dof_truth_available is True
    assert state.contact_truth_available is True
    np.testing.assert_array_equal(state.initial_q, q)
    np.testing.assert_array_equal(state.last_q, q)
    np.testing.assert_array_equal(state.last_qd, 0.0)
    assert not any(state.contact_active.values())


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("control_dt", 1.0 / 60.0),
        ("physics_dt", 1.0 / 60.0),
    ],
)
def test_runtime_rejects_ka005_2_timing_overrides(
    override: str,
    value: float,
) -> None:
    kwargs = {override: value}
    with pytest.raises(ValueError, match="30 Hz control / 120 Hz physics"):
        HandsOnB320BimanualIsaacLabEnv(
            B320_KA005_2_TASK_NAME,
            launch_isaac=False,
            image_obs=False,
            randomize=False,
            **kwargs,
        )


def test_interface_observation_exposes_fixed_knob_contract() -> None:
    env = HandsOnB320BimanualIsaacLabEnv(
        B320_KA005_2_TASK_NAME,
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    observation, info = env.reset(seed=3)
    assert env._control_dt == pytest.approx(1.0 / 30.0)
    assert env._sim_dt == pytest.approx(1.0 / 120.0)
    assert env._n_substeps == 4
    assert env.observation_space.contains(observation)
    state = observation["state"]
    assert state["knob_joint_state"].shape == (4,)
    assert state["knob_target_state"].shape == (6,)
    assert state["knob_contact_state"].shape == (8,)
    assert state["task_validity"].shape == (4,)
    assert state["sync_task_state"].shape == (20,)
    assert state["task_terminal_state"].shape == (3,)
    assert state["task_terminal_state"] == pytest.approx([0.0, 0.0, 0.0])
    assert "task_terminal_state" in CONFIG_MAPPING[
        B320_KA005_2_TASK_NAME
    ].b320_proprio_keys
    assert info["ka005_2_reset"]["target_rad"] == pytest.approx(
        [math.radians(60.0), math.radians(-60.0)]
    )
    progress, travel, reached = env._ka005_2_progress_values()
    np.testing.assert_allclose(progress, [0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(travel, [60.0, 60.0], atol=1e-8)
    np.testing.assert_array_equal(reached, [False, False])
    env._ka005_2_state.dof_truth_available = True
    env._ka005_2_state.last_q = np.radians([120.0, -120.0])
    progress, travel, reached = env._ka005_2_progress_values()
    np.testing.assert_allclose(progress, [120.0, 120.0], atol=1e-8)
    np.testing.assert_allclose(travel, [60.0, 60.0], atol=1e-8)
    np.testing.assert_array_equal(reached, [False, False])
    replay = env.get_physical_replay_reset_contract()
    assert replay["ka005_2_dof_names"] == [
        "left_knob_joint",
        "right_knob_joint",
    ]
    assert replay["ka005_2_dof_units"] == ["radian", "radian"]


def test_live_origin_truth_rebases_action_zero_and_reset_clears_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = HandsOnB320BimanualIsaacLabEnv(
        B320_KA005_2_TASK_NAME,
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    env.reset(seed=13)
    authored_initial = np.asarray(
        env._ka005_2_reset_sample.initial_rad,
        dtype=np.float64,
    )
    target = np.asarray(
        env._ka005_2_reset_sample.target_rad,
        dtype=np.float64,
    )
    accepted_q = np.asarray([0.01, -0.02], dtype=np.float64)
    accepted_poses = {
        "panel_base": np.asarray(
            [-0.139, 0.0, 0.915, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
        "knob_left": np.asarray(
            [-0.139, 0.135, 0.940, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
        "knob_right": np.asarray(
            [-0.139, -0.135, 0.940, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
    }
    views = {key: object() for key in knob_contract.CONTACT_KEYS}
    readable = {key: True for key in knob_contract.CONTACT_KEYS}
    env._launch_isaac = True
    env._ka005_2_contact_views = views
    monkeypatch.setattr(
        env,
        "_ka005_2_physics_dof_state",
        lambda: (accepted_q.copy(), np.zeros(2), True),
    )
    monkeypatch.setattr(
        env,
        "_read_b320_contact_force_sample",
        lambda view, _dt: (
            0.0,
            readable[
                next(key for key, candidate in views.items() if candidate is view)
            ],
        ),
    )
    monkeypatch.setattr(
        env,
        "_physics_body_pose",
        lambda name: accepted_poses.get(name),
    )

    readable["LR"] = False
    not_ready = env.stable_recording_origin_task_readiness()
    assert not_ready is not None
    assert not_ready["ready"] is False
    assert not_ready["dof_truth_available"] is True
    assert not_ready["contact_readable"]["LR"] is False
    assert env._ka005_2_state.failure_reason == ""
    assert env._ka005_2_state.sim_time_s == 0.0

    with pytest.raises(
        RecordingOriginStabilityError,
        match="continuous two-DOF/four-contact truth proof",
    ):
        env.accept_stable_recording_origin(
            {
                "task_readiness_required": True,
                "final_task_readiness": not_ready,
            }
        )

    readable["LR"] = True
    ready = env.stable_recording_origin_task_readiness()
    assert ready is not None and ready["ready"] is True
    acceptance = env.accept_stable_recording_origin(
        {
            "task_readiness_required": True,
            "final_task_readiness": ready,
        }
    )

    assert acceptance == {
        "accepted": True,
        "kind": "ka005_2_stable_physical_boundary_rebase",
        "physics_state_writes": False,
        "recording_origin_rad": pytest.approx(accepted_q.tolist()),
        "baseline_pose_names": ["panel_base", "knob_left", "knob_right"],
        "dof_truth_available": True,
        "contact_truth_available": True,
    }
    assert env._ka005_2_recording_origin_accepted is True
    np.testing.assert_array_equal(
        env._ka005_2_recording_origin_q_rad,
        accepted_q,
    )
    np.testing.assert_array_equal(
        env._ka005_2_state.initial_q,
        accepted_q,
    )
    np.testing.assert_array_equal(env._ka005_2_state.last_q, accepted_q)
    np.testing.assert_array_equal(env._ka005_2_state.target_q, target)
    np.testing.assert_array_equal(
        np.asarray(env._ka005_2_reset_sample.initial_rad),
        authored_initial,
    )
    assert env._ka005_2_state.sim_time_s == 0.0
    assert env._ka005_2_state.attempt_id == 0
    assert env._ka005_2_state.failure_reason == ""
    for name, pose in accepted_poses.items():
        np.testing.assert_array_equal(env._b320_reset_poses[name], pose)
    assert env._ka005_2_base_stability() == pytest.approx((0.0, 0.0, True))
    task_validity = env._compute_observation()["state"]["task_validity"]
    np.testing.assert_array_equal(task_validity, np.ones(4))

    # The first operator substep starts from the accepted q.  Settling motion
    # before this boundary cannot become attempt motion or a failure latch.
    env._update_ka005_2_physics_substep()
    assert env._ka005_2_state.sim_time_s == pytest.approx(DT)
    assert env._ka005_2_state.attempt_id == 0
    assert env._ka005_2_state.stage == KnobStage.WAIT_GRASP
    assert env._ka005_2_state.failure_reason == ""
    np.testing.assert_array_equal(env._ka005_2_state.last_q, accepted_q)

    env._launch_isaac = False
    env.reset(seed=13)
    assert env._ka005_2_recording_origin_accepted is False
    np.testing.assert_array_equal(env._ka005_2_recording_origin_q_rad, 0.0)
    assert env._ka005_2_recording_origin_poses == {}
    assert env._ka005_2_state.dof_truth_available is False
    assert env._ka005_2_state.contact_truth_available is False
    assert env._ka005_2_state.failure_reason == ""


@pytest.mark.parametrize(
    ("reason", "reason_code"),
    [
        ("dof_truth_unavailable", 1.0),
        ("contact_truth_unavailable", 2.0),
        ("joint_limit_exceeded", 3.0),
        ("unexpected_failure", 5.0),
    ],
)
def test_terminal_observation_has_stable_failure_reason_codes(
    reason: str,
    reason_code: float,
) -> None:
    env = HandsOnB320BimanualIsaacLabEnv(
        B320_KA005_2_TASK_NAME,
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    env.reset(seed=5)
    env._ka005_2_state.failure_reason = reason
    terminal = env._compute_observation()["state"]["task_terminal_state"]
    assert terminal == pytest.approx([0.0, 1.0, reason_code])
    assert env._compute_failure() is True
    env._ka005_2_state.failure_reason = "dof_truth_unavailable"
    terminal = env._compute_observation()["state"]["task_terminal_state"]
    assert terminal == pytest.approx([0.0, 1.0, reason_code])


def test_terminal_observation_exposes_success_and_base_drift_latches() -> None:
    env = HandsOnB320BimanualIsaacLabEnv(
        B320_KA005_2_TASK_NAME,
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    env.reset(seed=6)
    env._ka005_2_state.success = True
    terminal = env._compute_observation()["state"]["task_terminal_state"]
    assert terminal == pytest.approx([1.0, 0.0, 0.0])

    env._ka005_2_state.success = False
    pose = env._poses["panel_base"].copy()
    pose[0] += 0.003
    env._poses["panel_base"] = pose
    terminal = env._compute_observation()["state"]["task_terminal_state"]
    assert terminal == pytest.approx([0.0, 1.0, 4.0])


def test_panel_base_rotation_is_part_of_the_validity_contract() -> None:
    env = HandsOnB320BimanualIsaacLabEnv(
        B320_KA005_2_TASK_NAME,
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    env.reset(seed=4)
    pose = env._poses["panel_base"].copy()
    half_angle = math.radians(2.0) * 0.5
    pose[3:7] = [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]
    env._poses["panel_base"] = pose
    translation_m, rotation_deg, stable = env._ka005_2_base_stability()
    assert translation_m == pytest.approx(0.0)
    assert rotation_deg == pytest.approx(2.0)
    assert stable is False


@pytest.mark.parametrize(
    "live_readback",
    [
        None,
        np.asarray([np.nan, 0.0, 0.915, 1.0, 0.0, 0.0, 0.0]),
    ],
)
def test_live_panel_base_truth_fails_closed_when_unreadable(
    monkeypatch: pytest.MonkeyPatch,
    live_readback: np.ndarray | None,
) -> None:
    env = HandsOnB320BimanualIsaacLabEnv(
        B320_KA005_2_TASK_NAME,
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    env.reset(seed=8)
    env._launch_isaac = True
    monkeypatch.setattr(
        env,
        "_physics_body_pose",
        lambda name: live_readback if name == "panel_base" else None,
    )
    translation_m, rotation_deg, stable = env._ka005_2_base_stability()
    assert translation_m == pytest.approx(1.0)
    assert rotation_deg == pytest.approx(180.0)
    assert stable is False
    assert env._ka005_2_terminal_failure_reason() == "panel_base_drift"
