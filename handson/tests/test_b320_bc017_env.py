from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

import handson.sim.isaaclab.b320_bc017_env as bc017_env_module
from handson.sim.envs.rokae_wuji_ik_replay import rokae as rokae08
from handson.sim.isaaclab.b320_bc017_env import (
    HandsOnB320BC017IsaacLabEnv,
    _BC017_MAIN_RENDER_RESOLUTION_HW,
    _BC017_RESET_ARM_HOME,
    _DOF_NAMES,
    _PHYSICAL_CONTACT_KEYS,
    _TARGET_SELECTION_ALGORITHM,
)
from handson.sim.isaaclab.b320_bc017_scene import (
    BC017_ASSET_DIRECTORY,
    BC017_DOF_NAMES,
    BC017_DRAWER_JOINT_NAMES,
    BC017_DRAWER_ROLES_BY_SIDE,
    BC017AssetGateError,
    BC017_HANDLE_GRIP_LOCAL_BOUNDS_M,
    bc017_asset_point_world,
    bc017_drawer_role,
    bc017_handle_grip_bounds_world,
    bc017_point_on_handle_surface_world,
)


def _env(**kwargs: object) -> HandsOnB320BC017IsaacLabEnv:
    options: dict[str, object] = {
        "launch_isaac": False,
        "image_obs": False,
        "randomize": False,
    }
    options.update(kwargs)
    return HandsOnB320BC017IsaacLabEnv(**options)


class _SixDofView:
    count = 1

    def __init__(self, q: np.ndarray, qd: np.ndarray | None = None) -> None:
        self.positions = np.asarray(q, dtype=np.float32).reshape(1, 6)
        self.velocities = np.asarray(
            qd if qd is not None else np.zeros(6), dtype=np.float32
        ).reshape(1, 6)

    def get_dof_positions(self) -> np.ndarray:
        return self.positions.copy()

    def get_dof_velocities(self) -> np.ndarray:
        return self.velocities.copy()

    def get_metatype(self, _index: int) -> SimpleNamespace:
        return SimpleNamespace(
            dof_indices={name: index for index, name in enumerate(_DOF_NAMES)}
        )

    def set_dof_positions(self, value: object, _indices: object) -> None:
        self.positions = np.asarray(value, dtype=np.float32).copy()

    def set_dof_velocities(self, value: object, _indices: object) -> None:
        self.velocities = np.asarray(value, dtype=np.float32).copy()


def _bind_six_dofs(
    env: HandsOnB320BC017IsaacLabEnv,
    view: _SixDofView,
) -> None:
    env._task_dof_views = {
        name: (view, index, None) for index, name in enumerate(_DOF_NAMES)
    }


def test_bc017_interface_uses_authored_drawer_travel_without_release_gate() -> None:
    env = _env()
    try:
        calibration = env._bc017_state.calibration
        assert calibration.left_travel.closed_q_m == 0.0
        assert calibration.right_travel.closed_q_m == 0.0
        assert calibration.left_travel.open_q_m == pytest.approx(0.248)
        assert calibration.right_travel.open_q_m == pytest.approx(0.248)

        info = env._task_info()
        reset_contract = env.get_physical_replay_reset_contract()
        assert tuple(info["bc017_dof_names"]) == BC017_DOF_NAMES
        assert len(info["bc017_q_m"]) == 6
        assert len(info["bc017_active_q_m"]) == 2
        selection = info["bc017_target_selection"]
        assert selection["algorithm"] == _TARGET_SELECTION_ALGORITHM
        assert selection["slot_names"] == ["top", "middle", "bottom"]
        assert reset_contract["bc017_target_selection"] == selection
        assert info["bc017_handle_contact_geometry"] == (
            "source_u_pull_three_hull_surface"
        )
        assert info["bc017_grasp_constraint_kind"] == (
            "none_native_physx_contact_only"
        )
        assert reset_contract["bc017_handle_contact"] == {
            "trigger": (
                "live_assigned_hand_manifold_on_source_derived_"
                "three_hull_u_pull_surface"
            ),
            "constraint": "none_native_physx_contact_only",
            "whole_drawer_face_contact_allowed": False,
            "wrong_hand_contact_allowed": False,
            "contact_persists_only_while_manifold_exists": True,
        }
        for payload in (info, reset_contract):
            assert "bc017_release_receipt" not in payload
            assert "bc017_asset_snapshot" not in payload
            assert "bc017_live_usd_audit" not in payload
            assert "bc017_manifest_sha256" not in payload
    finally:
        env.close()


def test_bc017_every_reset_uses_two_seeded_draws_even_without_randomization() -> None:
    # These are the first two RandomState.randint(0, 3) results.  The fixed
    # corpus proves all nine independent left/right pairs without relying on a
    # statistical assertion.
    expected_pairs = {
        0: (0, 1),
        1: (1, 0),
        3: (2, 0),
        4: (2, 2),
        5: (2, 1),
        10: (1, 1),
        14: (0, 0),
        16: (1, 2),
        25: (0, 2),
    }
    env = _env(randomize=False)
    try:
        observed: set[tuple[int, int]] = set()
        for seed, expected in expected_pairs.items():
            env.reset(seed=seed)
            assert env._bc017_target_slots == expected
            assert env._bc017_state.target_slots == expected
            observed.add(expected)

        assert observed == {
            (left, right) for left in range(3) for right in range(3)
        }
        env.reset(seed=5)
        first = env._bc017_state.hidden_observation().copy()
        env.reset(seed=25)
        env.reset(seed=5)
        np.testing.assert_array_equal(
            env._bc017_state.hidden_observation(), first
        )
    finally:
        env.close()


def test_bc017_target_draws_precede_shared_reset_unconditionally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _env(randomize=False)
    calls: list[tuple[int, int]] = []

    class _DeterministicRandom:
        def randint(self, low: int, high: int) -> int:
            calls.append((low, high))
            return (2, 1)[len(calls) - 1]

    def _shared_reset(
        _self: object,
        *,
        clear_manual_request: bool,
    ) -> None:
        assert clear_manual_request is True
        assert calls == [(0, 3), (0, 3)]

    try:
        env._random = _DeterministicRandom()  # type: ignore[assignment]
        monkeypatch.setattr(
            HandsOnB320BC017IsaacLabEnv.__mro__[1],
            "_sample_reset_state",
            _shared_reset,
        )
        env._sample_reset_state(clear_manual_request=True)
        assert env._bc017_target_slots == (2, 1)
        assert env._bc017_state.target_slots == (2, 1)
    finally:
        env.close()


def test_bc017_reset_maps_twelve_physical_views_to_selected_semantics() -> None:
    env = _env()
    try:
        sentinels = {key: object() for key in _PHYSICAL_CONTACT_KEYS}
        env._bc017_physical_contact_views = dict(sentinels)
        env._bc017_physical_contact_bindings = {
            key: {
                "drawer_role": key[0],
                "hand_side": key[1],
                "bound": True,
            }
            for key in _PHYSICAL_CONTACT_KEYS
        }
        env._reset_bc017_runtime_state(target_slots=(2, 1))

        left_role = bc017_drawer_role("left", 2)
        right_role = bc017_drawer_role("right", 1)
        expected = {
            "LL": (left_role, "left"),
            "LR": (right_role, "left"),
            "RL": (left_role, "right"),
            "RR": (right_role, "right"),
        }
        assert len(env._bc017_physical_contact_views) == 12
        assert set(env._bc017_contact_views) == set(expected)
        selected_physical_keys = set(expected.values())
        assert all(
            sentinel not in env._bc017_contact_views.values()
            for physical_key, sentinel in sentinels.items()
            if physical_key not in selected_physical_keys
        )
        for semantic_key, physical_key in expected.items():
            assert env._bc017_contact_views[semantic_key] is sentinels[
                physical_key
            ]
            assert env._bc017_contact_bindings[semantic_key][
                "drawer_role"
            ] == physical_key[0]
        assert env._bc017_selected_dof_names() == (
            BC017_DRAWER_JOINT_NAMES[left_role],
            BC017_DRAWER_JOINT_NAMES[right_role],
        )
    finally:
        env.close()


def test_bc017_exact_topology_and_reset_cover_all_six_dofs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _env()
    try:
        view = _SixDofView(
            np.linspace(0.01, 0.06, 6),
            np.linspace(-0.03, 0.03, 6),
        )
        _bind_six_dofs(env, view)
        exact, diagnostics = env._bc017_exact_dof_topology()
        assert exact is True
        assert diagnostics["indices_by_name"] == list(range(6))
        assert diagnostics["tensor_dof_count"] == 6

        env._torch = object()
        env._task_reset_articulation_states = [("/World/DrawerRoot", view)]
        authored: list[tuple[str, np.ndarray, np.ndarray]] = []
        monkeypatch.setattr(
            bc017_env_module,
            "_indices_like",
            lambda _reference, count, _torch: np.arange(count),
        )
        monkeypatch.setattr(
            bc017_env_module,
            "_array_like",
            lambda _reference, values, _torch: np.asarray(values).copy(),
        )
        monkeypatch.setattr(
            env,
            "_record_authored_task_reset_write",
            lambda path, *, dof_position, dof_velocity: authored.append(
                (
                    path,
                    np.asarray(dof_position).copy(),
                    np.asarray(dof_velocity).copy(),
                )
            ),
        )
        env._write_bc017_closed_dofs()
        np.testing.assert_array_equal(view.positions, np.zeros((1, 6)))
        np.testing.assert_array_equal(view.velocities, np.zeros((1, 6)))
        assert len(authored) == 1
        np.testing.assert_array_equal(authored[0][1], np.zeros((1, 6)))
        np.testing.assert_array_equal(authored[0][2], np.zeros((1, 6)))
    finally:
        env.close()


def test_bc017_truth_projects_only_selected_pair_and_ignores_unselected_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _env()
    try:
        env._reset_bc017_runtime_state(target_slots=(2, 1))
        # left-bottom/right-middle are selected and remain closed.  Every other
        # drawer may move without changing the selected-pair success contract.
        view = _SixDofView([0.20, 0.21, 0.0, 0.22, 0.0, 0.23])
        _bind_six_dofs(env, view)
        monkeypatch.setattr(
            env,
            "_read_bc017_handle_contact_force",
            lambda _key, _q: (0.0, True),
        )
        monkeypatch.setattr(
            env,
            "_bc017_fixture_pose",
            lambda: (
                np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                True,
            ),
        )
        frame = env._read_bc017_physics_truth(
            physics_step=0,
            physics_time_s=0.0,
            allow_fixture_candidate=True,
        )
        assert frame.q_m == (0.0, 0.0)
        assert env._bc017_all_q_m.tolist() == pytest.approx(
            [0.20, 0.21, 0.0, 0.22, 0.0, 0.23]
        )
        snapshot = env._bc017_state.prime(frame)
        assert snapshot.failure is False
        assert snapshot.terminated is False
        next_frame = env._read_bc017_physics_truth(
            physics_step=1,
            physics_time_s=1.0 / 120.0,
            allow_fixture_candidate=True,
        )
        snapshot = env._bc017_state.update(next_frame)
        assert snapshot.failure is False
        assert snapshot.success is False
        assert snapshot.terminated is False
    finally:
        env.close()


def test_bc017_origin_reprime_preserves_selected_slots_and_all_six_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _env()
    try:
        env._reset_bc017_runtime_state(target_slots=(2, 1))
        env.env_step = 0
        env._bc017_all_q_m = np.zeros(6, dtype=np.float64)
        frame = bc017_env_module.BC017Frame(
            physics_step=0,
            physics_time_s=0.0,
            left_q_m=0.0,
            right_q_m=0.0,
            left_qd_m_s=0.0,
            right_qd_m_s=0.0,
            left_dof_readable=True,
            right_dof_readable=True,
            ll_force_n=0.0,
            lr_force_n=0.0,
            rl_force_n=0.0,
            rr_force_n=0.0,
            ll_readable=True,
            lr_readable=True,
            rl_readable=True,
            rr_readable=True,
            fixture_translation_drift_m=0.0,
            fixture_rotation_drift_rad=0.0,
            fixture_truth_readable=True,
        )
        monkeypatch.setattr(
            env,
            "stable_recording_origin_task_readiness",
            lambda: {"ready": True},
        )
        monkeypatch.setattr(
            env,
            "_bc017_fixture_pose",
            lambda: (
                np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                True,
            ),
        )
        monkeypatch.setattr(
            env,
            "_read_bc017_physics_truth",
            lambda **_kwargs: frame,
        )

        accepted = env.accept_stable_recording_origin(
            {
                "task_readiness_required": True,
                "final_task_readiness": {"ready": True},
            }
        )
        assert accepted is not None
        assert accepted["origin"]["q_m"] == [0.0] * 6
        assert accepted["target_selection"]["target_slots"] == [2, 1]
        assert env._bc017_state.target_slots == (2, 1)
        assert env._bc017_state.snapshot().target_slots == (2, 1)
        assert env._bc017_state.hidden_observation()[-2:].tolist() == [2.0, 1.0]
    finally:
        env.close()


def test_bc017_selected_roles_drive_generic_role_poses() -> None:
    env = _env()
    try:
        env._reset_bc017_runtime_state(target_slots=(1, 2))
        left_role = bc017_drawer_role("left", 1)
        right_role = bc017_drawer_role("right", 2)
        left_pose = np.asarray([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
        right_pose = np.asarray([4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0])
        env._poses[left_role] = left_pose.copy()
        env._poses[right_role] = right_pose.copy()
        env._refresh_object_poses()
        np.testing.assert_array_equal(env._poses["left_drawer"], left_pose)
        np.testing.assert_array_equal(env._poses["right_drawer"], right_pose)
    finally:
        env.close()


def test_bc017_default_main_camera_frames_scaled_desk_without_moving_pose() -> None:
    env = _env()
    try:
        assert _BC017_MAIN_RENDER_RESOLUTION_HW == (640, 1280)
        assert (env._render_height, env._render_width) == (640, 1280)
        assert env.recording_camera_profile["head"]["output_resolution"] == [
            1280,
            640,
        ]
        override = env.recording_camera_profile["task_override"]
        assert override["task_name"] == "t15_open_drawers"
        assert override["version"] == "bc017-study-camera-v2"
        # Camera draws are deterministic for the task/seed and use the packaged profile.
        other = _env()
        try:
            assert env.recording_camera_profile == other.recording_camera_profile
            assert np.isfinite(env.recording_camera_profile["head"]["eye_xyz_m"]).all()
        finally:
            other.close()
    finally:
        env.close()


def test_bc017_live_path_composes_asset_without_preflight_or_acceptance() -> None:
    source = inspect.getsource(
        HandsOnB320BC017IsaacLabEnv._spawn_task_specific_assets
    )
    assert "spawn_bc017_task_assets" in source
    assert "validate_bc017" not in source
    assert "asset_snapshot" not in source
    assert "release" not in source


def test_bc017_pre_simulation_audit_is_disabled() -> None:
    env = _env()
    try:
        assert env._pre_simulation_context_task_audit() is None
    finally:
        env.close()


@pytest.mark.usefixtures("rokae_assets")
def test_bc017_reset_home_moves_both_palms_back_and_outward() -> None:
    env = _env()
    try:
        for side in ("right", "left"):
            joint_pos = {
                f"joint{index}_{side}(_1)?": -99.0
                for index in range(1, 8)
            }
            robot_cfg = SimpleNamespace(
                init_state=SimpleNamespace(joint_pos=joint_pos),
            )
            configured = env._configure_task_robot_cfg(robot_cfg, side)
            actual_home = np.asarray(
                [
                    configured.init_state.joint_pos[
                        f"joint{index}_{side}(_1)?"
                    ]
                    for index in range(1, 8)
                ],
                dtype=np.float64,
            )
            np.testing.assert_allclose(
                actual_home,
                _BC017_RESET_ARM_HOME[side],
                atol=0.0,
                rtol=0.0,
            )

            base_in_world = rokae08.base_in_world_matrix(side)
            base_in_world[:3, 3] = np.asarray(
                [-0.75, 0.065 if side == "left" else -0.065, 1.34],
                dtype=np.float64,
            )
            palm_in_world = base_in_world @ rokae08.load_model(side).fk(
                actual_home
            )
            expected_y = 0.28 if side == "left" else -0.28
            np.testing.assert_allclose(
                palm_in_world[:3, 3],
                (-0.48, expected_y, 0.9975),
                atol=2.0e-5,
                rtol=0.0,
            )
            assert rokae08.wrist_limit_ratio(actual_home) < 0.07
    finally:
        env.close()


def _left_handle_surface_points(drawer_q_m: float) -> np.ndarray:
    lineage = json.loads(
        (
            BC017_ASSET_DIRECTORY / "evidence/source_lineage.json"
        ).read_text(encoding="utf-8")
    )
    bounds = lineage["derivation"]["authored_closed_collision_audit"][
        "collider_bounds"
    ]
    points: list[tuple[float, float, float]] = []
    for segment in ("negative_leg", "crossbar", "positive_leg"):
        row = next(
            value
            for path, value in bounds.items()
            if path.startswith("bc017_left_drawer_link/")
            and f"handle_{segment}_source_convex" in path
        )
        minimum = np.asarray(row["minimum_m"], dtype=np.float64)
        maximum = np.asarray(row["maximum_m"], dtype=np.float64)
        local = 0.5 * (minimum + maximum)
        # The source-derived minimum local-Y face is a stable exposed plane
        # on each convex hull, away from the two compound seams.
        local[1] = minimum[1]
        local[0] += drawer_q_m
        points.append(bc017_asset_point_world(local))
    return np.asarray(points, dtype=np.float64)


def test_bc017_contact_accepts_three_handle_hulls_and_rejects_drawer_face(
) -> None:
    drawer_q_m = 0.10
    surface_points = _left_handle_surface_points(drawer_q_m)
    assert surface_points.shape == (3, 3)
    assert all(
        bc017_point_on_handle_surface_world(
            "left",
            drawer_q_m,
            point,
            tolerance_m=1.0e-5,
        )
        for point in surface_points
    )

    left_top_role = bc017_drawer_role("left", 0)
    local_minimum, local_maximum = BC017_HANDLE_GRIP_LOCAL_BOUNDS_M[
        left_top_role
    ]
    drawer_face_local = 0.5 * (
        np.asarray(local_minimum, dtype=np.float64)
        + np.asarray(local_maximum, dtype=np.float64)
    )
    drawer_face_local[0] = local_minimum[0] + drawer_q_m
    drawer_face_point = np.asarray(
        bc017_asset_point_world(drawer_face_local), dtype=np.float64
    )
    minimum, maximum = bc017_handle_grip_bounds_world(
        left_top_role, drawer_q_m
    )
    assert np.all(drawer_face_point >= np.asarray(minimum))
    assert np.all(drawer_face_point <= np.asarray(maximum))
    assert not bc017_point_on_handle_surface_world(
        left_top_role,
        drawer_q_m,
        drawer_face_point,
        tolerance_m=1.0e-5,
    )

    class ContactView:
        sensor_count = 1
        filter_count = 1

        @staticmethod
        def get_contact_data(_dt: float):
            points = np.vstack((surface_points, drawer_face_point))
            return (
                np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64),
                points,
                np.zeros_like(points),
                np.zeros(4, dtype=np.float64),
                np.asarray([[4]], dtype=np.int64),
                np.asarray([[0]], dtype=np.int64),
            )

    env = _env()
    try:
        env._bc017_contact_views["LL"] = ContactView()
        env._bc017_contact_bindings["LL"] = {
            "drawer_role": left_top_role
        }
        force_n, readable = env._read_bc017_handle_contact_force(
            "LL", drawer_q_m
        )
        assert readable is True
        assert force_n == pytest.approx(7.0)
    finally:
        env.close()


def test_bc017_nonfinite_drawer_q_fails_contact_closed_before_readback() -> None:
    class UntouchableContactView:
        def get_contact_data(self, _dt: float):
            raise AssertionError("non-finite q must fail before contact readback")

    env = _env()
    try:
        env._bc017_contact_views["LL"] = UntouchableContactView()
        env._bc017_contact_bindings["LL"] = {
            "drawer_role": bc017_drawer_role("left", 0)
        }
        assert env._read_bc017_handle_contact_force("LL", float("nan")) == (
            0.0,
            False,
        )
        with pytest.raises(
            BC017AssetGateError, match="drawer_q_m must be a finite number"
        ):
            bc017_point_on_handle_surface_world(
                bc017_drawer_role("left", 0),
                float("nan"),
                (0.0, 0.0, 0.0),
                tolerance_m=0.0015,
            )
    finally:
        env.close()


def test_bc017_study_composes_after_physics_and_is_reported_render_only() -> None:
    source = inspect.getsource(HandsOnB320BC017IsaacLabEnv.__init__)
    assert source.index("super().__init__") < source.index(
        "_spawn_bc017_study_room"
    )
    assert "if self._launch_isaac" in source
    env = _env()
    try:
        info = env._task_info()
        reset_contract = env.get_physical_replay_reset_contract()
        assert info["bc017_study_room_root"] == (
            "/World/VisualEnvironment/BC017Study"
        )
        assert info["bc017_study_room_render_paths"] == []
        assert reset_contract["bc017_study_room"] == {
            "root": "/World/VisualEnvironment/BC017Study",
            "render_only": True,
            "paths": [],
        }
    finally:
        env.close()
