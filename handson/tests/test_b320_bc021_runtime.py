from __future__ import annotations

from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from handson.runtime.collection_support import (
    ISAACSIM_COLLECTION_UNVERIFIED_TASKS,
    ISAACSIM_FORMAL_COLLECTION_TASKS,
    ISAACSIM_NONSTANDARD_DEFERRED_RENDER_TASKS,
)
from handson.sim.isaaclab.b320_bc021_contract import (
    FINAL_GEOMETRY,
    HIDDEN_OBSERVATION_FIELDS,
    HIDDEN_OBSERVATION_SHAPE,
    PROVISIONAL_GEOMETRY,
    PROVISIONAL_ACCEPTANCE_CALIBRATION,
    STABLE_HOLD_SUBSTEPS,
    TASK_NAME,
    sample_bc021_reset,
)
from handson.sim.isaaclab.b320_bc021_env import (
    _BC021_COLLIDER_ENVELOPE_COVER_RADIUS_M,
    _BC021_RESET_CLEARANCE_PADDING_M,
    _STATE_OBSERVATION_KEYS,
    HandsOnB320BC021IsaacLabEnv,
)
from handson.sim.isaaclab.b320_bc021_scene import (
    BLOCK_HOME_POSES_WXYZ,
    MARKER_SURFACE_OFFSET_M,
    MARKER_THICKNESS_M,
    TASK_REQUIREMENT,
    _marked_face_marker_center_local,
)
from handson.sim.isaaclab.b320_bimanual_scene import B320_TASK_SPECS
from handson.sim.isaaclab.recording_origin import STATIC_POSE_FALLBACK_TASKS
from handson.tasks.t01_align_blocks_in_midair.config import TaskConfig


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_bc021_requirement_is_canonical_and_runtime_is_dedicated() -> None:
    config = TaskConfig()
    yaml_path = (
        _repository_root()
        / "configs"
        / "multi_task"
        / "t01_align_blocks_in_midair.yaml"
    )
    parsed = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert config.task_prompt == TASK_REQUIREMENT
    assert parsed["prompt"] == TASK_REQUIREMENT
    assert B320_TASK_SPECS[TASK_NAME].prompt == TASK_REQUIREMENT
    assert config.b320_runtime == "bc021"
    assert config.time_limit_seconds == 60
    assert parsed["time_limit_seconds"] == 60
    assert B320_TASK_SPECS[TASK_NAME].time_limit_seconds == 60


def test_bc021_interface_observations_are_fixed_finite_and_fail_closed() -> None:
    env = HandsOnB320BC021IsaacLabEnv(
        TASK_NAME,
        launch_isaac=False,
        image_obs=False,
        randomize=False,
        render_mode="rgb_array",
    )
    try:
        observation, info = env.reset(seed=11)
        state = observation["state"]
        expected_shapes = {
            "tcp_pose": (14,),
            "gripper_pose": (32,),
            "table_delta_height": (1,),
            "bc021_block_pose": (14,),
            "bc021_marked_face_state": (20,),
            "bc021_contact_state": (15,),
            "bc021_stage_state": (12,),
            "bc021_terminal_state": (4,),
            "bc021_hidden_state": HIDDEN_OBSERVATION_SHAPE,
        }
        assert tuple(state) == _STATE_OBSERVATION_KEYS
        assert set(state) == set(expected_shapes)
        for key, shape in expected_shapes.items():
            assert state[key].shape == shape
            assert state[key].dtype == np.float64
            assert np.all(np.isfinite(state[key]))
        assert env.observation_space.contains(observation)
        assert not {
            "b320_carton_contact_detail_available",
            "b320_carton_zone_valid",
            "b320_carton_airborne_now",
            "b320_pen_table_contact_available",
            "b320_handover_completed",
        } & set(info)
        metrics = env._metrics(update=False)
        assert metrics["contact_truth_ready"] is False
        assert metrics["pose_truth_readable"] is False
        assert metrics["velocity_truth_readable"] is False
        assert metrics["geometry_contract_final"] is True
        assert metrics["success_now"] is False
        replay = env.get_physical_replay_reset_contract()
        assert replay["bc021_geometry_provisional"] is False
        assert replay["bc021_acceptance_calibration_final"] is False
        assert replay["bc021_acceptance_calibration_sha256"] == (
            PROVISIONAL_ACCEPTANCE_CALIBRATION.digest_sha256()
        )
        assert replay["bc021_formal_collection_ready"] is False
        assert replay["bc021_release_required"] is False
        assert replay["bc021_stable_hold_substeps"] == STABLE_HOLD_SUBSTEPS
        assert "marked_contact_manifold" in replay["bc021_state_machine"]
        assert replay["bc021_hidden_state_shape"] == [30]
        assert replay["bc021_hidden_state_fields"] == list(
            HIDDEN_OBSERVATION_FIELDS
        )
    finally:
        env.close()


@pytest.mark.parametrize("side", ("left", "right"))
def test_bc021_visible_marker_surface_matches_physical_face(side: str) -> None:
    marked_center = FINAL_GEOMETRY.marked_center_local(side)
    marked_normal = FINAL_GEOMETRY.marked_normal_local(side)
    marker_center = _marked_face_marker_center_local(side)
    visible_surface = marker_center + marked_normal * (0.5 * MARKER_THICKNESS_M)
    np.testing.assert_allclose(
        visible_surface,
        marked_center + marked_normal * MARKER_SURFACE_OFFSET_M,
        atol=1.0e-12,
    )
    assert MARKER_SURFACE_OFFSET_M <= 0.00005


def test_bc021_left_hand_lock_captures_level_reference_and_preserves_right_hand() -> None:
    env = HandsOnB320BC021IsaacLabEnv(
        TASK_NAME,
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    try:
        left_tcp = env._left_tcp_pose.copy()
        left_hand = env._left_hand.copy()
        right_tcp = env._right_tcp_pose.copy()
        env._bc021_physics_frame_index = 23
        env._bc021_maybe_latch_left_hand_lock(
            {
                "reference_latched": True,
                "left_bottom_clearance_m": (
                    env._bc021_acceptance_calibration.minimum_bottom_clearance_m
                ),
                "left_flat": True,
                "left_flatness_error_deg": 0.0,
                "left_face_normal_w": np.asarray([0.0, -1.0, 0.0]),
            }
        )
        assert env._bc021_left_hand_lock_active is True
        assert env._bc021_left_hand_lock_frame == 23

        env._left_tcp_pose = left_tcp + 0.10
        env._left_hand = left_hand + 0.10
        env._right_tcp_pose = right_tcp + 0.10
        assert env._bc021_clamp_left_hand_lock() is True
        np.testing.assert_array_equal(env._left_tcp_pose, left_tcp)
        np.testing.assert_array_equal(env._left_hand, left_hand)
        np.testing.assert_array_equal(env._right_tcp_pose, right_tcp + 0.10)
    finally:
        env.close()


def test_bc021_left_hand_lock_requires_height_level_and_blue_face_right() -> None:
    env = HandsOnB320BC021IsaacLabEnv(
        TASK_NAME,
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    try:
        target_height = (
            env._bc021_acceptance_calibration.minimum_bottom_clearance_m
        )
        env._bc021_physics_frame_index = 31
        env._bc021_maybe_latch_left_hand_lock(
            {
                "left_bottom_clearance_m": target_height - 0.001,
                "left_flat": True,
                "left_flatness_error_deg": 3.0,
                "left_face_normal_w": np.asarray([0.0, -1.0, 0.0]),
                "reference_latched": True,
            }
        )
        assert env._bc021_left_hand_lock_active is False

        env._bc021_maybe_latch_left_hand_lock(
            {
                "left_bottom_clearance_m": target_height,
                "left_flat": False,
                "left_flatness_error_deg": 12.0,
                "left_face_normal_w": np.asarray([0.0, -1.0, 0.0]),
                "reference_latched": True,
            }
        )
        assert env._bc021_left_hand_lock_active is False

        env._bc021_maybe_latch_left_hand_lock(
            {
                "left_bottom_clearance_m": target_height,
                "left_flat": True,
                "left_flatness_error_deg": 3.0,
                "left_face_normal_w": np.asarray([1.0, 0.0, 0.0]),
                "reference_latched": True,
            }
        )
        assert env._bc021_left_hand_lock_active is False

        env._bc021_maybe_latch_left_hand_lock(
            {
                "left_bottom_clearance_m": target_height,
                "left_flat": True,
                "left_flatness_error_deg": 3.0,
                "left_face_normal_w": np.asarray([0.0, -1.0, 0.0]),
                "reference_latched": False,
            }
        )
        assert env._bc021_left_hand_lock_active is True
        assert env._bc021_left_hand_lock_frame == 31
    finally:
        env.close()


def test_bc021_left_hand_lock_reapplies_the_live_drive_target_after_latch() -> None:
    env = HandsOnB320BC021IsaacLabEnv(
        TASK_NAME,
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    try:
        calls: list[int] = []
        env._bc021_left_hand_lock_active = True
        env._bc021_left_hand_lock_tcp_pose = env._left_tcp_pose.copy()
        env._bc021_left_hand_lock_hand = env._left_hand.copy()
        env._bc021_left_hand_lock_target_applied = False
        env._apply_joint_targets = lambda: calls.append(1)
        env._prepare_physics_substep_joint_targets(0, 4)
        assert calls == [1]
        assert env._bc021_left_hand_lock_target_applied is True
        env._prepare_physics_substep_joint_targets(1, 4)
        assert calls == [1]
    finally:
        env.close()


def test_bc021_task_config_adapter_includes_every_dedicated_key() -> None:
    config = TaskConfig()
    env = config.get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=False,
        launch_isaac=False,
        image_obs=False,
    )
    try:
        assert env.action_space.shape == (52,)
        assert type(env.unwrapped).__name__ == "HandsOnB320BC021IsaacLabEnv"
        # Live EEF state requires robot tensor views. Inspect native CPU state here.
        observation, _info = env.unwrapped.reset(seed=12)
        assert env.unwrapped.observation_space.contains(observation)
        assert config.proprio_keys == [
            "tcp_pose",
            "gripper_pose",
            "table_delta_height",
            "bc021_block_pose",
            "bc021_marked_face_state",
            "bc021_contact_state",
            "bc021_stage_state",
            "bc021_terminal_state",
            "bc021_hidden_state",
        ]
    finally:
        env.close()


def test_bc021_accepted_origin_poses_are_hash_bound_in_info_and_replay() -> None:
    env = HandsOnB320BC021IsaacLabEnv(
        TASK_NAME,
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    try:
        env.reset(seed=81)
        poses = {
            "block_a": np.asarray(
                BLOCK_HOME_POSES_WXYZ["block_a"], dtype=np.float64
            ),
            "block_b": np.asarray(
                BLOCK_HOME_POSES_WXYZ["block_b"], dtype=np.float64
            ),
        }
        payload, digest = env._bc021_pose_set_payload(poses)
        assert digest is not None and len(digest) == 64
        env._bc021_recording_origin_accepted = True
        env._bc021_recording_origin_poses = {
            name: pose.copy() for name, pose in poses.items()
        }
        env._bc021_recording_origin_pose_sha256 = digest

        info = env._task_info()
        replay = env.get_physical_replay_reset_contract()
        for record in (info, replay):
            assert record["bc021_recording_origin_poses_wxyz"] == payload
            assert record["bc021_recording_origin_pose_sha256"] == digest
            assert record["bc021_recording_origin_source"] == (
                "post_stable_origin_physics_truth"
            )

        env._bc021_recording_origin_poses["block_a"][0] += 0.001
        with pytest.raises(RuntimeError, match="pose hash drifted"):
            env.get_physical_replay_reset_contract()
    finally:
        env.close()


def test_bc021_allows_collection_and_uses_three_camera_post_render_path() -> None:
    assert TASK_NAME not in ISAACSIM_COLLECTION_UNVERIFIED_TASKS
    assert TASK_NAME in ISAACSIM_FORMAL_COLLECTION_TASKS
    assert TASK_NAME in ISAACSIM_NONSTANDARD_DEFERRED_RENDER_TASKS
    assert TASK_NAME in STATIC_POSE_FALLBACK_TASKS


def test_bc021_cameras_use_live_paper_rendering_by_default(
    monkeypatch,
) -> None:
    monkeypatch.delenv("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", raising=False)
    collection_env = HandsOnB320BC021IsaacLabEnv(
        TASK_NAME,
        launch_isaac=False,
        image_obs=True,
    )
    try:
        assert collection_env._deferred_camera_render is False
    finally:
        collection_env.close()

    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "0")
    postrender_env = HandsOnB320BC021IsaacLabEnv(
        TASK_NAME,
        launch_isaac=False,
        image_obs=True,
    )
    try:
        assert postrender_env._deferred_camera_render is False
    finally:
        postrender_env.close()


def test_bc021_legacy_square_block_asset_is_removed() -> None:
    legacy = (
        _repository_root()
        / "handson"
        / "handson"
        / "sim"
        / "envs"
        / "usd"
        / "b320_bimanual_benchmark"
        / "assets"
        / "dexverse_authored_square_block"
    )
    assert not legacy.exists()


class _BC021FakeRobot:
    def __init__(
        self,
        body_names: tuple[str, ...],
        positions: tuple[tuple[float, float, float], ...],
    ) -> None:
        self.body_names = body_names
        self.body_matrices = []
        for position in positions:
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, 3] = np.asarray(position, dtype=np.float64)
            self.body_matrices.append(matrix)
        home = np.asarray([[0.1, -0.2]], dtype=np.float64)
        self.data = SimpleNamespace(
            joint_pos=home.copy(),
            default_joint_pos=home.copy(),
        )


def _bc021_fake_home_geometry() -> tuple[
    dict[str, _BC021FakeRobot],
    tuple[tuple[str, str, np.ndarray, np.ndarray], ...],
]:
    robots = {
        "left": _BC021FakeRobot(
            ("base_left", "left_palm_link", "left_finger1_tip"),
            ((-0.80, 0.24, 1.30), (-0.58, 0.24, 1.08), (-0.51, 0.24, 1.00)),
        ),
        "right": _BC021FakeRobot(
            ("base_right", "right_palm_link", "right_finger1_tip"),
            ((-0.80, -0.24, 1.30), (-0.58, -0.24, 1.08), (-0.51, -0.24, 1.00)),
        ),
    }
    bounds = (
        (
            "left",
            "/World/HandsOnRobots/LeftRobot/left_palm_link/collider",
            np.asarray([-0.62, 0.20, 1.04]),
            np.asarray([-0.54, 0.28, 1.12]),
        ),
        (
            "left",
            "/World/HandsOnRobots/LeftRobot/left_finger1_tip/collider",
            np.asarray([-0.53, 0.22, 0.98]),
            np.asarray([-0.49, 0.26, 1.02]),
        ),
        (
            "right",
            "/World/HandsOnRobots/RightRobot/right_palm_link/collider",
            np.asarray([-0.62, -0.28, 1.04]),
            np.asarray([-0.54, -0.20, 1.12]),
        ),
        (
            "right",
            "/World/HandsOnRobots/RightRobot/right_finger1_tip/collider",
            np.asarray([-0.53, -0.26, 0.98]),
            np.asarray([-0.49, -0.22, 1.02]),
        ),
    )
    return robots, bounds


def _install_bc021_fake_home_envelope(
    env: HandsOnB320BC021IsaacLabEnv,
) -> tuple[tuple[str, str, np.ndarray, np.ndarray], ...]:
    robots, bounds = _bc021_fake_home_geometry()
    env._robots = robots
    env._robot_body_pose_matrix = (
        lambda robot, body_index: robot.body_matrices[int(body_index)].copy()
    )
    env._bc021_robot_collider_bounds_w = lambda: bounds
    env._cache_bc021_reset_home_clearance_envelope()
    return bounds


def test_bc021_reset_home_envelope_covers_palm_and_fingertip_colliders(
) -> None:
    env = HandsOnB320BC021IsaacLabEnv(
        TASK_NAME,
        launch_isaac=False,
        image_obs=False,
        randomize=True,
    )
    try:
        bounds = _install_bc021_fake_home_envelope(env)
        envelope = env._bc021_reset_home_clearance_points_w

        assert env._bc021_reset_home_clearance_cached is True
        assert env._bc021_reset_home_body_count == 6
        assert env._bc021_reset_home_collider_count == 4
        assert envelope.flags.writeable is False
        assert all(
            not pose.flags.writeable
            for pose in env._bc021_reset_home_hand_poses_wxyz.values()
        )
        assert _BC021_COLLIDER_ENVELOPE_COVER_RADIUS_M < (
            _BC021_RESET_CLEARANCE_PADDING_M
        )

        # Every composed palm/fingertip collider contributes all eight actual
        # world-bound corners, not merely its rigid-body origin.
        for _side, path, low, high in bounds:
            assert "palm" in path or "finger" in path
            for corner in product(*zip(low, high)):
                expected = np.asarray(corner, dtype=np.float64)
                assert np.any(np.all(np.isclose(envelope, expected), axis=1))

        with pytest.raises(ValueError):
            envelope[0, 0] = 0.0

        frozen = envelope.copy()
        env._robot_body_pose_matrix = lambda *_args: (_ for _ in ()).throw(
            AssertionError("live episode body pose was reread")
        )
        env._bc021_robot_collider_bounds_w = lambda: (_ for _ in ()).throw(
            AssertionError("live episode collider was reread")
        )
        env._cache_bc021_reset_home_clearance_envelope()
        np.testing.assert_array_equal(
            env._bc021_reset_home_clearance_points_w,
            frozen,
        )
    finally:
        env._robots = {}
        env.close()


def test_bc021_sampler_rejects_collider_interior_missed_by_body_origins() -> None:
    env = HandsOnB320BC021IsaacLabEnv(
        TASK_NAME,
        launch_isaac=False,
        image_obs=False,
        randomize=False,
    )
    try:
        robots, raw_bounds = _bc021_fake_home_geometry()
        block_center = np.asarray(
            BLOCK_HOME_POSES_WXYZ["block_a"][:3],
            dtype=np.float64,
        )
        bounds = list(raw_bounds)
        bounds[0] = (
            bounds[0][0],
            bounds[0][1],
            block_center - 0.010,
            block_center + 0.010,
        )
        env._robots = robots
        env._robot_body_pose_matrix = (
            lambda robot, body_index: robot.body_matrices[
                int(body_index)
            ].copy()
        )
        env._bc021_robot_collider_bounds_w = lambda: tuple(bounds)
        env._cache_bc021_reset_home_clearance_envelope()

        left_hand = env._bc021_reset_home_hand_poses_wxyz["left"]
        right_hand = env._bc021_reset_home_hand_poses_wxyz["right"]
        # All six link origins remain far enough away, so the legacy
        # origin-only check accepts the authored reset.
        sample_bc021_reset(
            np.random.RandomState(4),
            left_home_pose_wxyz=np.asarray(BLOCK_HOME_POSES_WXYZ["block_a"]),
            right_home_pose_wxyz=np.asarray(BLOCK_HOME_POSES_WXYZ["block_b"]),
            geometry=PROVISIONAL_GEOMETRY,
            randomize=False,
            left_hand_home_wxyz=left_hand,
            right_hand_home_wxyz=right_hand,
            hand_clearance_points_w=np.asarray(
                [
                    matrix[:3, 3]
                    for robot in robots.values()
                    for matrix in robot.body_matrices
                ],
                dtype=np.float64,
            ),
        )
        # The complete palm-collider volume contains the block center and must
        # therefore fail closed when the exact same reset reaches the sampler.
        with pytest.raises(
            RuntimeError,
            match="authored non-random reset is not physically valid",
        ):
            sample_bc021_reset(
                np.random.RandomState(4),
                left_home_pose_wxyz=np.asarray(
                    BLOCK_HOME_POSES_WXYZ["block_a"]
                ),
                right_home_pose_wxyz=np.asarray(
                    BLOCK_HOME_POSES_WXYZ["block_b"]
                ),
                geometry=PROVISIONAL_GEOMETRY,
                randomize=False,
                left_hand_home_wxyz=left_hand,
                right_hand_home_wxyz=right_hand,
                hand_clearance_points_w=(
                    env._bc021_reset_home_clearance_points_w
                ),
            )
    finally:
        env._robots = {}
        env.close()


def test_bc021_initialization_resets_and_forwards_robot_before_cache() -> None:
    env = HandsOnB320BC021IsaacLabEnv(
        TASK_NAME,
        launch_isaac=False,
        image_obs=False,
    )
    events: list[str] = []

    class _Robot:
        def __init__(self, side: str) -> None:
            self.side = side

        def write_data_to_sim(self) -> None:
            events.append(f"write:{self.side}")

        def update(self, dt: float) -> None:
            assert dt == 0.0
            events.append(f"update:{self.side}")

    try:
        env._robots = {
            "left": _Robot("left"),
            "right": _Robot("right"),
        }
        env._reset_robot_control_state = lambda: events.append("reset-home")
        env._sim = SimpleNamespace(forward=lambda: events.append("forward"))
        env._scene = SimpleNamespace(
            update=lambda dt: events.append(f"scene:{dt}")
        )
        env._cache_bc021_reset_home_clearance_envelope = (
            lambda: events.append("cache")
        )

        env._initialize_bc021_reset_home_clearance_cache()

        assert events == [
            "reset-home",
            "write:left",
            "write:right",
            "forward",
            "scene:0.0",
            "update:left",
            "update:right",
            "cache",
        ]
    finally:
        env._robots = {}
        env._sim = None
        env._scene = None
        env.close()


def test_bc021_same_seed_reset_is_independent_of_previous_episode_pose() -> None:
    env = HandsOnB320BC021IsaacLabEnv(
        TASK_NAME,
        launch_isaac=False,
        image_obs=False,
        randomize=True,
    )
    try:
        _install_bc021_fake_home_envelope(env)
        env._random.seed(829)
        env._sample_reset_state(clear_manual_request=True)
        first = (
            env._bc021_reset_sample.left_pose_wxyz.copy(),
            env._bc021_reset_sample.right_pose_wxyz.copy(),
            env._bc021_reset_sample.attempts,
        )

        # Simulate an arbitrarily stale prior episode.  Sampling must not touch
        # these moving body tensors or the mutable command TCPs.
        for robot in env._robots.values():
            for matrix in robot.body_matrices:
                matrix[:3, 3] = np.asarray([9.0, 9.0, 9.0])
        env._left_tcp_pose[:] = np.asarray([8.0, 8.0, 8.0, 1.0, 0.0, 0.0, 0.0])
        env._right_tcp_pose[:] = np.asarray([7.0, 7.0, 7.0, 1.0, 0.0, 0.0, 0.0])
        env._robot_body_pose_matrix = lambda *_args: (_ for _ in ()).throw(
            AssertionError("reset sampled stale live robot state")
        )

        env._random.seed(829)
        env._sample_reset_state(clear_manual_request=True)
        second = (
            env._bc021_reset_sample.left_pose_wxyz.copy(),
            env._bc021_reset_sample.right_pose_wxyz.copy(),
            env._bc021_reset_sample.attempts,
        )

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        assert first[2] == second[2]
        assert env._bc021_reset_clearance_point_count == len(
            env._bc021_reset_home_clearance_points_w
        )
    finally:
        env._robots = {}
        env.close()
