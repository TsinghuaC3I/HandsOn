"""Isolated IsaacLab runtime for B320-BC021 two-block face mating."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from gymnasium import spaces
import numpy as np

from .b320_bc021_contract import (
    BC021Frame,
    BC021ResetSample,
    BC021State,
    CONTACT_KEYS,
    HIDDEN_OBSERVATION_FIELDS,
    HIDDEN_OBSERVATION_SHAPE,
    MAX_REFERENCE_TRANSLATION_M,
    FINAL_GEOMETRY,
    PROVISIONAL_ACCEPTANCE_CALIBRATION,
    REFERENCE_FLAT_MAX_TILT_DEG,
    TASK_NAME,
    sample_bc021_reset,
)
from .b320_bc021_scene import (
    ASSET_CANDIDATE_LICENSE,
    ASSET_CANDIDATE_URL,
    ASSET_INTEGRATION_STATUS,
    BLOCK_HOME_POSES_WXYZ,
    MARKER_SURFACE_OFFSET_M,
    TASK_REQUIREMENT,
    spawn_bc021_task_assets,
)
from .b320_bimanual_env import HandsOnB320BimanualIsaacLabEnv


_OBSERVATION_KEYS = (
    "bc021_block_pose",
    "bc021_marked_face_state",
    "bc021_contact_state",
    "bc021_stage_state",
    "bc021_terminal_state",
    "bc021_hidden_state",
)
_BASE_OBSERVATION_KEYS = (
    "tcp_pose",
    "gripper_pose",
    "table_delta_height",
)
_STATE_OBSERVATION_KEYS = (*_BASE_OBSERVATION_KEYS, *_OBSERVATION_KEYS)

# ``sample_bc021_reset`` treats every clearance point as the center of a
# block-bounding-sphere exclusion zone with 55 mm of extra padding.  Sampling
# each collision AABB on a 50 mm Cartesian lattice leaves a worst-case lattice
# covering radius of sqrt(3) * 25 mm = 43.3 mm.  Consequently an intersection
# with any point in the conservative collider AABB is still rejected with at
# least 11.7 mm of the contract padding remaining.  This is deliberately a
# volume envelope (not just eight corners), so large palm/base colliders cannot
# have an undetected interior between sparse surface samples.
_BC021_COLLIDER_ENVELOPE_GRID_STEP_M = 0.050
_BC021_RESET_CLEARANCE_PADDING_M = 0.055
_BC021_COLLIDER_ENVELOPE_COVER_RADIUS_M = (
    math.sqrt(3.0) * 0.5 * _BC021_COLLIDER_ENVELOPE_GRID_STEP_M
)
# The left block's blue/marked face must point toward the operator's right.
# BC021 assigns the left workspace to +Y and the right workspace to -Y.
_BC021_OPERATOR_RIGHT_AXIS_W = (0.0, -1.0, 0.0)
_BC021_LEFT_HAND_LOCK_FACE_RIGHT_MAX_ERROR_DEG = 10.0


def _finite_vector(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(shape)
    except Exception:
        return np.zeros(shape, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        return np.zeros(shape, dtype=np.float64)
    return result.copy()


class HandsOnB320BC021IsaacLabEnv(HandsOnB320BimanualIsaacLabEnv):
    """120 Hz, five-contact-channel, fail-closed BC021 specialization."""

    _canonical_camera_postrender_supported = True

    def _bc021_calibration_is_final(self) -> bool:
        return self._bc021_acceptance_calibration.formal_for_geometry(
            self._bc021_geometry
        )

    def _bc021_formal_collection_ready(self) -> bool:
        return bool(
            ASSET_INTEGRATION_STATUS == "final"
            and self._bc021_calibration_is_final()
        )

    def __init__(self, task_name: str = TASK_NAME, **kwargs: Any):
        if task_name != TASK_NAME:
            raise ValueError(f"BC021 runtime does not support {task_name!r}")
        self._bc021_geometry = FINAL_GEOMETRY
        self._bc021_acceptance_calibration = (
            PROVISIONAL_ACCEPTANCE_CALIBRATION
        )
        self._bc021_state = BC021State()
        self._bc021_contact_views: dict[str, Any] = {}
        self._bc021_contact_bindings: dict[str, dict[str, Any]] = {}
        self._bc021_contact_force_n = {key: 0.0 for key in CONTACT_KEYS}
        self._bc021_contact_readable = {key: False for key in CONTACT_KEYS}
        self._bc021_block_contact_points_w = np.zeros(
            (0, 3), dtype=np.float64
        )
        self._bc021_block_contact_point_forces_n = np.zeros(
            0, dtype=np.float64
        )
        self._bc021_block_contact_points_readable = False
        self._bc021_pose_truth_readable = False
        self._bc021_velocity_truth_readable = False
        self._bc021_last_pose_wxyz = {
            name: np.asarray(pose, dtype=np.float64)
            for name, pose in BLOCK_HOME_POSES_WXYZ.items()
        }
        self._bc021_last_velocity = {
            name: np.zeros(6, dtype=np.float64)
            for name in BLOCK_HOME_POSES_WXYZ
        }
        self._bc021_physics_frame_index = 0
        self._bc021_reset_clearance_point_count = 0
        empty_clearance = np.zeros((0, 3), dtype=np.float64)
        empty_clearance.setflags(write=False)
        self._bc021_reset_home_clearance_points_w = empty_clearance
        self._bc021_reset_home_hand_poses_wxyz: dict[str, np.ndarray] = {}
        self._bc021_reset_home_collider_paths: tuple[str, ...] = ()
        self._bc021_reset_home_body_count = 0
        self._bc021_reset_home_collider_count = 0
        self._bc021_reset_home_clearance_cached = False
        self._bc021_recording_origin_accepted = False
        self._bc021_recording_origin_poses: dict[str, np.ndarray] = {}
        self._bc021_recording_origin_pose_sha256: str | None = None
        self._bc021_reset_sample = BC021ResetSample(
            self._bc021_last_pose_wxyz["block_a"].copy(),
            self._bc021_last_pose_wxyz["block_b"].copy(),
            0,
            False,
        )
        self._bc021_left_hand_lock_active = False
        self._bc021_left_hand_lock_tcp_pose: np.ndarray | None = None
        self._bc021_left_hand_lock_hand: np.ndarray | None = None
        self._bc021_left_hand_lock_frame: int | None = None
        self._bc021_left_hand_lock_target_applied = False
        super().__init__(task_name=task_name, **kwargs)

    def physical_replay_post_substep_observer_is_safe(
        self,
        callback: Any,
    ) -> bool:
        return bool(
            getattr(callback, "__self__", None) is self
            and getattr(callback, "__func__", None)
            is HandsOnB320BC021IsaacLabEnv._handson_post_physics_substep_callback
        )

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        if self._stage is None:
            raise RuntimeError("BC021 stage is unavailable before asset spawn")
        self._b320_spawn_result = spawn_bc021_task_assets(
            self._stage,
            sim_utils,
        )

    def _init_task_specific_runtime(self) -> None:
        super()._init_task_specific_runtime()
        self._bc021_contact_views = {}
        self._bc021_contact_bindings = {}
        if not self._robots:
            return
        self._initialize_bc021_reset_home_clearance_cache()
        physics_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_view is None:
            self._add_task_state_warning(
                "BC021 PhysX contact view is unavailable"
            )
            return

        bindings = (
            (
                "LA",
                self._role_body_path("block_a"),
                self._hand_body_paths("left"),
            ),
            (
                "LB",
                self._role_body_path("block_b"),
                self._hand_body_paths("left"),
            ),
            (
                "RA",
                self._role_body_path("block_a"),
                self._hand_body_paths("right"),
            ),
            (
                "RB",
                self._role_body_path("block_b"),
                self._hand_body_paths("right"),
            ),
            (
                "AB",
                self._role_body_path("block_a"),
                [self._role_body_path("block_b")],
            ),
        )
        for key, sensor, raw_filters in bindings:
            filters = [str(path) for path in raw_filters if path]
            binding = {
                "sensor_path": str(sensor or ""),
                "filter_paths": filters,
                "sensor_count": 0,
                "filter_count": 0,
                "bound": False,
            }
            self._bc021_contact_bindings[key] = binding
            if not sensor or not filters:
                self._add_task_state_warning(
                    f"BC021 {key} contact binding is incomplete"
                )
                continue
            try:
                view = physics_view.create_rigid_contact_view(
                    str(sensor),
                    filter_patterns=filters,
                    max_contact_data_count=max(128, len(filters) * 16),
                )
                sensor_count = int(getattr(view, "sensor_count", 0))
                filter_count = int(getattr(view, "filter_count", 0))
                binding.update(
                    {
                        "sensor_count": sensor_count,
                        "filter_count": filter_count,
                        "bound": bool(sensor_count > 0 and filter_count > 0),
                    }
                )
                if binding["bound"]:
                    self._bc021_contact_views[key] = view
            except Exception as exc:
                self._add_task_state_warning(
                    f"Could not initialize BC021 {key} contact view: {exc}"
                )
        missing = sorted(set(CONTACT_KEYS) - set(self._bc021_contact_views))
        if missing:
            self._add_task_state_warning(
                "BC021 five-channel contact truth is incomplete; success "
                "fails closed: " + ",".join(missing)
            )

    @staticmethod
    def _bc021_numpy(value: Any) -> np.ndarray:
        raw = value
        detach = getattr(raw, "detach", None)
        if callable(detach):
            raw = detach()
        cpu = getattr(raw, "cpu", None)
        if callable(cpu):
            raw = cpu()
        numpy = getattr(raw, "numpy", None)
        if callable(numpy):
            raw = numpy()
        return np.asarray(raw, dtype=np.float64)

    def _initialize_bc021_reset_home_clearance_cache(self) -> None:
        """Reset the complete robots once, then freeze their home envelope.

        ABC calls this hook immediately after its first ``self._sim.reset()``
        and before the first virtual ``_sample_reset_state`` call.  Perform the
        same complete robot reset used at episode boundaries, forward its state
        into PhysX/USD, and only then read collider bounds.  Later episode
        samples never consult moving articulation state.
        """

        if self._bc021_reset_home_clearance_cached or not self._robots:
            return
        self._reset_robot_control_state()
        for robot in self._robots.values():
            write_data = getattr(robot, "write_data_to_sim", None)
            if callable(write_data):
                write_data()
        if self._sim is not None:
            self._sim.forward()
        if self._scene is not None:
            try:
                self._scene.update(0.0)
            except Exception:
                pass
        for robot in self._robots.values():
            update = getattr(robot, "update", None)
            if callable(update):
                update(0.0)
        self._cache_bc021_reset_home_clearance_envelope()

    @staticmethod
    def _bc021_aabb_envelope_points(
        low_w: np.ndarray,
        high_w: np.ndarray,
    ) -> np.ndarray:
        """Return a bounded-spacing volume lattice including all AABB corners."""

        low = np.asarray(low_w, dtype=np.float64).reshape(3)
        high = np.asarray(high_w, dtype=np.float64).reshape(3)
        extent = high - low
        if (
            not np.all(np.isfinite(low))
            or not np.all(np.isfinite(high))
            or np.any(extent < 0.0)
        ):
            raise RuntimeError("BC021 robot collider has invalid world bounds")
        axis_counts = np.maximum(
            1,
            np.ceil(extent / _BC021_COLLIDER_ENVELOPE_GRID_STEP_M).astype(
                np.int64
            )
            + 1,
        )
        point_count = int(np.prod(axis_counts, dtype=np.int64))
        if point_count > 250_000:
            raise RuntimeError(
                "BC021 robot collider envelope exceeds the safe point budget"
            )
        axes = [
            np.linspace(low[index], high[index], int(axis_counts[index]))
            for index in range(3)
        ]
        grid = np.meshgrid(*axes, indexing="ij")
        return np.stack(grid, axis=-1).reshape(-1, 3)

    def _bc021_robot_collider_bounds_w(
        self,
    ) -> tuple[tuple[str, str, np.ndarray, np.ndarray], ...]:
        """Read every enabled composed robot CollisionAPI world AABB."""

        if self._stage is None:
            raise RuntimeError("BC021 stage is unavailable for collider audit")
        from pxr import Usd, UsdGeom, UsdPhysics

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
                UsdGeom.Tokens.guide,
            ],
            useExtentsHint=True,
        )
        predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
        bounds: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        for side in ("left", "right"):
            root = self._stage.GetPrimAtPath(
                f"/World/HandsOnRobots/{side.capitalize()}Robot"
            )
            if root is None or not root.IsValid():
                raise RuntimeError(
                    f"BC021 cannot resolve the {side} reset-home robot root"
                )
            side_bounds = 0
            for prim in Usd.PrimRange(root, predicate):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                collision = UsdPhysics.CollisionAPI(prim)
                enabled_attr = collision.GetCollisionEnabledAttr()
                if (
                    enabled_attr
                    and enabled_attr.IsValid()
                    and enabled_attr.HasAuthoredValueOpinion()
                    and not bool(enabled_attr.Get())
                ):
                    continue
                collider_path = str(prim.GetPath())
                world_range = bbox_cache.ComputeWorldBound(
                    prim
                ).ComputeAlignedRange()
                low = np.asarray(world_range.GetMin(), dtype=np.float64)
                high = np.asarray(world_range.GetMax(), dtype=np.float64)
                if (
                    low.shape != (3,)
                    or high.shape != (3,)
                    or not np.all(np.isfinite(low))
                    or not np.all(np.isfinite(high))
                    or np.any(high < low)
                ):
                    raise RuntimeError(
                        "BC021 cannot bound reset-home robot collider "
                        f"{collider_path}"
                    )
                bounds.append((side, collider_path, low.copy(), high.copy()))
                side_bounds += 1
            if side_bounds == 0:
                raise RuntimeError(
                    f"BC021 {side} reset-home robot exposes no colliders"
                )
        return tuple(bounds)

    def _cache_bc021_reset_home_clearance_envelope(self) -> None:
        """Capture one immutable, complete reset-home body/collider envelope."""

        if self._bc021_reset_home_clearance_cached:
            return
        if _BC021_COLLIDER_ENVELOPE_COVER_RADIUS_M >= (
            _BC021_RESET_CLEARANCE_PADDING_M
        ):
            raise RuntimeError("BC021 collider envelope lattice is not conservative")

        body_points: list[np.ndarray] = []
        hand_poses: dict[str, np.ndarray] = {}
        expected_body_count = 0
        for side in ("left", "right"):
            robot = self._robots.get(side)
            if robot is None:
                raise RuntimeError(
                    f"BC021 cannot cache the missing {side} reset-home robot"
                )
            body_names = tuple(getattr(robot, "body_names", ()))
            if not body_names:
                raise RuntimeError(f"BC021 {side} robot exposes no rigid bodies")
            expected_body_count += len(body_names)

            current_joints = self._bc021_numpy(robot.data.joint_pos)
            default_joints = self._bc021_numpy(robot.data.default_joint_pos)
            if (
                current_joints.shape != default_joints.shape
                or not np.all(np.isfinite(current_joints))
                or not np.all(np.isfinite(default_joints))
                or not np.allclose(
                    current_joints,
                    default_joints,
                    rtol=0.0,
                    atol=1.0e-6,
                )
            ):
                raise RuntimeError(
                    f"BC021 attempted to cache {side} before robot reset-home"
                )

            for body_index in range(len(body_names)):
                try:
                    matrix = np.asarray(
                        self._robot_body_pose_matrix(robot, body_index),
                        dtype=np.float64,
                    ).reshape(4, 4)
                except Exception as exc:
                    raise RuntimeError(
                        "BC021 cannot read reset-home body "
                        f"{side}/{body_names[body_index]}"
                    ) from exc
                if not np.all(np.isfinite(matrix)):
                    raise RuntimeError(
                        "BC021 reset-home body pose is non-finite: "
                        f"{side}/{body_names[body_index]}"
                    )
                body_points.append(matrix[:3, 3].copy())
                body_name = str(body_names[body_index]).split("/")[-1]
                if body_name == f"{side}_palm_link":
                    pose = np.asarray(
                        self._pose_matrix_to_vector(matrix),
                        dtype=np.float64,
                    ).reshape(7)
                    pose.setflags(write=False)
                    hand_poses[side] = pose

        if len(body_points) != expected_body_count:
            raise RuntimeError("BC021 reset-home body envelope is incomplete")
        if set(hand_poses) != {"left", "right"}:
            raise RuntimeError(
                "BC021 reset-home envelope cannot resolve both physical palms"
            )

        collider_bounds = self._bc021_robot_collider_bounds_w()
        paths = tuple(value[1] for value in collider_bounds)
        lowered_paths = tuple(path.lower() for path in paths)
        for side in ("left", "right"):
            if not any(
                side in path and "palm" in path for path in lowered_paths
            ):
                raise RuntimeError(
                    f"BC021 {side} palm collider is absent from the envelope"
                )
            if not any(
                side in path and "finger" in path for path in lowered_paths
            ):
                raise RuntimeError(
                    f"BC021 {side} finger colliders are absent from the envelope"
                )

        collider_points = [
            self._bc021_aabb_envelope_points(low, high)
            for _side, _path, low, high in collider_bounds
        ]
        points = np.concatenate(
            [np.asarray(body_points, dtype=np.float64).reshape(-1, 3)]
            + collider_points,
            axis=0,
        )
        # Duplicate samples from nested/touching collider bounds only add work;
        # nanometre rounding is far below the certified 11.7 mm residual margin.
        points = np.unique(np.round(points, decimals=9), axis=0)
        if not len(points) or not np.all(np.isfinite(points)):
            raise RuntimeError("BC021 reset-home clearance envelope is empty")
        points.setflags(write=False)

        # Publish atomically only after every body, palm, finger, and collider
        # has passed validation.  These arrays are process-lifetime metadata.
        self._bc021_reset_home_clearance_points_w = points
        self._bc021_reset_home_hand_poses_wxyz = hand_poses
        self._bc021_reset_home_collider_paths = paths
        self._bc021_reset_home_body_count = expected_body_count
        self._bc021_reset_home_collider_count = len(collider_bounds)
        self._bc021_reset_home_clearance_cached = True

    def _bc021_robot_clearance_points(self) -> np.ndarray:
        """Return the frozen reset-home envelope, never live episode state."""

        return self._bc021_reset_home_clearance_points_w.copy()

    def _reset_bc021_runtime_state(self) -> None:
        self._bc021_state.reset()
        self._bc021_contact_force_n = {key: 0.0 for key in CONTACT_KEYS}
        self._bc021_contact_readable = {key: False for key in CONTACT_KEYS}
        self._bc021_block_contact_points_w = np.zeros(
            (0, 3), dtype=np.float64
        )
        self._bc021_block_contact_point_forces_n = np.zeros(
            0, dtype=np.float64
        )
        self._bc021_block_contact_points_readable = False
        self._bc021_pose_truth_readable = False
        self._bc021_velocity_truth_readable = False
        self._bc021_physics_frame_index = 0
        self._bc021_recording_origin_accepted = False
        self._bc021_recording_origin_poses = {}
        self._bc021_recording_origin_pose_sha256 = None
        self._bc021_left_hand_lock_active = False
        self._bc021_left_hand_lock_tcp_pose = None
        self._bc021_left_hand_lock_hand = None
        self._bc021_left_hand_lock_frame = None
        self._bc021_left_hand_lock_target_applied = False

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self._reset_bc021_runtime_state()
        if (
            self._launch_isaac
            and self._robots
            and not self._bc021_reset_home_clearance_cached
        ):
            raise RuntimeError(
                "BC021 reset sampling started before the immutable robot-home "
                "clearance envelope was cached"
            )
        clearance_points = self._bc021_robot_clearance_points()
        self._bc021_reset_clearance_point_count = int(len(clearance_points))
        left_hand_home = self._bc021_reset_home_hand_poses_wxyz.get(
            "left",
            self._left_tcp_pose,
        )
        right_hand_home = self._bc021_reset_home_hand_poses_wxyz.get(
            "right",
            self._right_tcp_pose,
        )
        self._bc021_reset_sample = sample_bc021_reset(
            self._random,
            left_home_pose_wxyz=np.asarray(
                BLOCK_HOME_POSES_WXYZ["block_a"],
                dtype=np.float64,
            ),
            right_home_pose_wxyz=np.asarray(
                BLOCK_HOME_POSES_WXYZ["block_b"],
                dtype=np.float64,
            ),
            geometry=self._bc021_geometry,
            randomize=bool(self.randomize),
            left_hand_home_wxyz=left_hand_home,
            right_hand_home_wxyz=right_hand_home,
            hand_clearance_points_w=(
                clearance_points if len(clearance_points) else None
            ),
        )
        self._bc021_last_pose_wxyz = {
            "block_a": self._bc021_reset_sample.left_pose_wxyz.copy(),
            "block_b": self._bc021_reset_sample.right_pose_wxyz.copy(),
        }
        self._bc021_last_velocity = {
            "block_a": np.zeros(6, dtype=np.float64),
            "block_b": np.zeros(6, dtype=np.float64),
        }
        if not self._launch_isaac:
            self._poses["block_a"] = (
                self._bc021_reset_sample.left_pose_wxyz.copy()
            )
            self._poses["block_b"] = (
                self._bc021_reset_sample.right_pose_wxyz.copy()
            )
            self._b320_reset_poses.update(
                {
                    "block_a": self._poses["block_a"].copy(),
                    "block_b": self._poses["block_b"].copy(),
                }
            )

    def _randomized_pose(
        self,
        pose: np.ndarray,
        *,
        fixture: bool,
        object_name: str | None = None,
    ) -> np.ndarray:
        if object_name == "block_a":
            return self._bc021_reset_sample.left_pose_wxyz.copy()
        if object_name == "block_b":
            return self._bc021_reset_sample.right_pose_wxyz.copy()
        return super()._randomized_pose(
            pose,
            fixture=fixture,
            object_name=object_name,
        )

    def _bc021_physical_pose(self, name: str) -> tuple[np.ndarray, bool]:
        value = self._physics_body_pose(name)
        if value is not None:
            raw = np.asarray(value, dtype=np.float64).reshape(-1)
            if (
                raw.shape == (7,)
                and np.all(np.isfinite(raw))
                and float(np.linalg.norm(raw[3:7])) >= 1.0e-8
            ):
                return raw.copy(), True
        return self._pose(name), False

    def _bc021_physical_velocity(
        self,
        name: str,
    ) -> tuple[np.ndarray, bool]:
        value = self._physics_body_velocity(name)
        if value is not None:
            raw = np.asarray(value, dtype=np.float64).reshape(-1)
            if raw.shape == (6,) and np.all(np.isfinite(raw)):
                return raw.copy(), True
        return np.zeros(6, dtype=np.float64), False

    def _read_bc021_physics_truth(
        self,
        *,
        update_latches: bool,
    ) -> tuple[dict[str, Any], tuple[bool, ...]]:
        for key in CONTACT_KEYS:
            view = self._bc021_contact_views.get(key)
            force, readable = self._read_b320_contact_force_sample(
                view,
                self._sim_dt,
            )
            self._bc021_contact_force_n[key] = float(force)
            self._bc021_contact_readable[key] = bool(readable)
        points, point_forces, point_truth = self._read_b320_contact_points(
            self._bc021_contact_views.get("AB"),
            self._sim_dt,
        )
        self._bc021_block_contact_points_w = points.copy()
        self._bc021_block_contact_point_forces_n = point_forces.copy()
        self._bc021_block_contact_points_readable = bool(point_truth)

        left_pose, left_pose_truth = self._bc021_physical_pose("block_a")
        right_pose, right_pose_truth = self._bc021_physical_pose("block_b")
        left_velocity, left_velocity_truth = self._bc021_physical_velocity(
            "block_a"
        )
        right_velocity, right_velocity_truth = self._bc021_physical_velocity(
            "block_b"
        )
        self._bc021_last_pose_wxyz = {
            "block_a": left_pose.copy(),
            "block_b": right_pose.copy(),
        }
        self._bc021_last_velocity = {
            "block_a": left_velocity.copy(),
            "block_b": right_velocity.copy(),
        }
        self._bc021_pose_truth_readable = bool(
            left_pose_truth and right_pose_truth
        )
        self._bc021_velocity_truth_readable = bool(
            left_velocity_truth and right_velocity_truth
        )
        if update_latches:
            self._bc021_physics_frame_index += 1

        frame = BC021Frame(
            left_pose_wxyz=left_pose,
            right_pose_wxyz=right_pose,
            left_velocity=left_velocity,
            right_velocity=right_velocity,
            contact_force_n=dict(self._bc021_contact_force_n),
            contact_readable=dict(self._bc021_contact_readable),
            block_contact_points_w=points,
            block_contact_point_forces_n=point_forces,
            block_contact_points_readable=bool(point_truth),
            pose_truth_readable=self._bc021_pose_truth_readable,
            velocity_truth_readable=self._bc021_velocity_truth_readable,
        )
        metrics, flags = self._bc021_state.evaluate(
            frame,
            self._bc021_geometry,
            frame_index=self._bc021_physics_frame_index,
            update_latches=update_latches,
            calibration=self._bc021_acceptance_calibration,
        )
        for side, key in (("left", "LA"), ("right", "RB")):
            self._b320_contact_available[side] = (
                key in self._bc021_contact_views
            )
            self._b320_contact_readable[side] = bool(
                self._bc021_contact_readable[key]
            )
            self._b320_contact_force[side] = float(
                self._bc021_contact_force_n[key]
            )
            if self._bc021_contact_readable[key]:
                self._b320_contact_peak[side] = max(
                    self._b320_contact_peak[side],
                    float(self._bc021_contact_force_n[key]),
                )
                if (
                    update_latches
                    and self._bc021_contact_force_n[key]
                    >= self._bc021_acceptance_calibration.assigned_hand_contact_min_n
                ):
                    self._b320_contact_seen[side] = True
        return metrics, flags

    def _bc021_clamp_left_hand_lock(self) -> bool:
        """Hold the left TCP/hand command captured after the block is level."""

        if not self._bc021_left_hand_lock_active:
            return False
        if (
            self._bc021_left_hand_lock_tcp_pose is None
            or self._bc021_left_hand_lock_hand is None
        ):
            return False
        self._left_tcp_pose = self._bc021_left_hand_lock_tcp_pose.copy()
        self._left_hand = self._bc021_left_hand_lock_hand.copy()
        self._left_pose_commanded = True
        return True

    def _bc021_maybe_latch_left_hand_lock(
        self,
        metrics: dict[str, Any],
    ) -> None:
        """Latch only at target height, level, with the blue face rightward.

        This is a control-target assist, not a teleport or object freeze.  The
        right hand remains fully operator-controlled and the existing physical
        BC021 face/contact predicates remain authoritative.
        """

        if self._bc021_left_hand_lock_active:
            return
        # This operator assist intentionally does not require a contact sample,
        # but it must never bypass the task's physical pose requirements via a
        # previously latched reference.  All three geometric gates are checked
        # on the current physics frame.
        left_bottom_clearance_m = float(
            metrics.get("left_bottom_clearance_m", -math.inf)
        )
        required_clearance_m = float(
            self._bc021_acceptance_calibration.minimum_bottom_clearance_m
        )
        left_at_target_height = bool(
            left_bottom_clearance_m >= required_clearance_m
        )
        left_level = bool(metrics.get("left_flat", False))
        left_face_normal_w = _finite_vector(
            metrics.get("left_face_normal_w"),
            (3,),
        )
        left_face_normal_norm = float(np.linalg.norm(left_face_normal_w))
        if left_face_normal_norm < 1.0e-8:
            blue_face_right_error_deg = math.inf
        else:
            blue_face_right_error_deg = float(
                math.degrees(
                    math.acos(
                        float(
                            np.clip(
                                np.dot(
                                    left_face_normal_w / left_face_normal_norm,
                                    np.asarray(
                                        _BC021_OPERATOR_RIGHT_AXIS_W,
                                        dtype=np.float64,
                                    ),
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
            )
        blue_face_right = bool(
            blue_face_right_error_deg
            <= _BC021_LEFT_HAND_LOCK_FACE_RIGHT_MAX_ERROR_DEG
        )
        if not (left_at_target_height and left_level and blue_face_right):
            return
        tcp_pose = np.asarray(self._left_tcp_pose, dtype=np.float64).reshape(-1)
        hand = np.asarray(self._left_hand, dtype=np.float64).reshape(-1)
        if tcp_pose.shape != (7,) or hand.ndim != 1:
            return
        if not np.all(np.isfinite(tcp_pose)) or not np.all(np.isfinite(hand)):
            return
        self._bc021_left_hand_lock_tcp_pose = tcp_pose.copy()
        self._bc021_left_hand_lock_hand = hand.copy()
        self._bc021_left_hand_lock_frame = int(
            self._bc021_physics_frame_index
        )
        self._bc021_left_hand_lock_active = True
        # The latch happens after a physics substep, while the arm targets for
        # that substep have already been written.  Mark the target dirty so the
        # next live substep reapplies the captured pose to the actual drive,
        # rather than merely changing the Python-side command fields.
        self._bc021_left_hand_lock_target_applied = False
        print(
            "[HandsOn/IsaacLab] BC021 left-hand lock active: "
            f"frame={self._bc021_left_hand_lock_frame}, "
            f"clearance={left_bottom_clearance_m:.3f}/"
            f"{required_clearance_m:.3f}m, "
            f"flatness={float(metrics.get('left_flatness_error_deg', 180.0)):.2f}deg, "
            f"blue_face_right_error={blue_face_right_error_deg:.2f}deg",
            flush=True,
        )
        self._bc021_clamp_left_hand_lock()

    def _handson_post_physics_substep_callback(
        self,
        _substep_index: int,
        _substep_count: int,
    ) -> None:
        metrics, _flags = self._read_bc021_physics_truth(update_latches=True)
        self._bc021_maybe_latch_left_hand_lock(metrics)

    def _apply_operator_action_extras(self, action) -> None:
        super()._apply_operator_action_extras(action)
        # Exact recorded-drive replay owns every per-substep target and must
        # not be modified by a live operator assist.
        if not callable(
            getattr(self, "_handson_pre_physics_substep_callback", None)
        ):
            if self._bc021_clamp_left_hand_lock():
                # The normal live path calls _apply_joint_targets immediately
                # after this hook.  The flag prevents the substep hook below
                # from redundantly rewriting both arms every frame.
                self._bc021_left_hand_lock_target_applied = True

    def _prepare_physics_substep_joint_targets(
        self,
        substep_index: int,
        substep_count: int,
    ) -> str:
        if not callable(
            getattr(self, "_handson_pre_physics_substep_callback", None)
        ):
            if self._bc021_clamp_left_hand_lock():
                if not self._bc021_left_hand_lock_target_applied:
                    self._apply_joint_targets()
                    self._bc021_left_hand_lock_target_applied = True
        return super()._prepare_physics_substep_joint_targets(
            substep_index,
            substep_count,
        )

    def _metrics(self, *, update: bool) -> dict[str, Any]:
        metrics, flags = self._read_bc021_physics_truth(
            update_latches=False
        )
        self._b320_condition_flags = tuple(bool(value) for value in flags)
        values = dict(metrics)
        values["success_now"] = bool(all(self._b320_condition_flags))
        self._b320_last_metrics = values
        return values

    def _success_steps_required(self) -> int:
        # The dedicated state already enforces the calibrated physics hold.
        return 1

    def _compute_failure(self) -> bool:
        reason = self._bc021_state.first_terminal_reason
        self._b320_failure_reason = reason
        if reason is not None:
            self._success_counter = 0
            return True
        return bool(super()._compute_failure())

    @staticmethod
    def _bc021_pose_set_payload(
        poses: dict[str, np.ndarray],
    ) -> tuple[dict[str, list[float]], str | None]:
        """Return canonical replay metadata for an accepted physical origin."""

        if set(poses) != {"block_a", "block_b"}:
            return {}, None
        payload = {
            name: np.asarray(poses[name], dtype=np.float64).reshape(7).tolist()
            for name in sorted(poses)
        }
        if not all(
            np.all(np.isfinite(np.asarray(value, dtype=np.float64)))
            for value in payload.values()
        ):
            raise RuntimeError("BC021 accepted-origin pose metadata is non-finite")
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return payload, hashlib.sha256(encoded).hexdigest()

    def _bc021_recording_origin_manifest(
        self,
    ) -> tuple[dict[str, list[float]], str | None]:
        payload, digest = self._bc021_pose_set_payload(
            self._bc021_recording_origin_poses
        )
        if digest != self._bc021_recording_origin_pose_sha256:
            raise RuntimeError("BC021 accepted-origin pose hash drifted")
        if bool(self._bc021_recording_origin_accepted) != (digest is not None):
            raise RuntimeError(
                "BC021 accepted-origin flag and complete pose manifest disagree"
            )
        return payload, digest

    def stable_recording_origin_task_readiness(
        self,
    ) -> dict[str, Any] | None:
        metrics, _flags = self._read_bc021_physics_truth(
            update_latches=False
        )
        ready = bool(
            metrics["contact_truth_ready"]
            and metrics["pose_truth_readable"]
            and metrics["velocity_truth_readable"]
            and set(self._bc021_contact_views) == set(CONTACT_KEYS)
        )
        return {
            "kind": "bc021_two_pose_velocity_and_five_contact_truth",
            "ready": ready,
            "pose_truth_available": bool(metrics["pose_truth_readable"]),
            "velocity_truth_available": bool(
                metrics["velocity_truth_readable"]
            ),
            "contact_truth_available": bool(metrics["contact_truth_ready"]),
            "contact_keys": list(CONTACT_KEYS),
            "contact_view_present": {
                key: key in self._bc021_contact_views for key in CONTACT_KEYS
            },
            "contact_readable": dict(self._bc021_contact_readable),
            "geometry_contract_final": bool(
                metrics["geometry_contract_final"]
            ),
            "formal_collection_ready": (
                self._bc021_formal_collection_ready()
            ),
        }

    def accept_stable_recording_origin(
        self,
        report: dict[str, Any],
    ) -> dict[str, Any] | None:
        from .recording_origin import RecordingOriginStabilityError

        final_readiness = report.get("final_task_readiness")
        if (
            report.get("task_readiness_required") is not True
            or not isinstance(final_readiness, dict)
            or final_readiness.get("ready") is not True
        ):
            raise RecordingOriginStabilityError(
                "B320-BC021 reached the origin boundary without continuous "
                "pose/velocity/five-contact truth.",
                diagnostics={
                    "failure_phase": "bc021_origin_acceptance",
                    "task_readiness": final_readiness,
                },
            )
        if int(self.env_step) != 0:
            raise RecordingOriginStabilityError(
                "B320-BC021 origin acceptance ran after action zero.",
                diagnostics={
                    "failure_phase": "bc021_origin_acceptance",
                    "env_step": int(self.env_step),
                },
            )
        readiness = self.stable_recording_origin_task_readiness()
        max_speed = max(
            float(np.linalg.norm(velocity))
            for velocity in self._bc021_last_velocity.values()
        )
        if readiness is None or not readiness["ready"] or max_speed > 1.0e-8:
            raise RecordingOriginStabilityError(
                "B320-BC021 physical truth changed after the exact-zero "
                "recording boundary.",
                diagnostics={
                    "failure_phase": "bc021_origin_acceptance",
                    "readiness": readiness,
                    "max_root_speed": max_speed,
                },
            )
        accepted_poses = {
            name: pose.copy()
            for name, pose in self._bc021_last_pose_wxyz.items()
        }
        accepted_pose_payload, accepted_pose_sha256 = (
            self._bc021_pose_set_payload(accepted_poses)
        )
        if accepted_pose_sha256 is None:
            raise RecordingOriginStabilityError(
                "B320-BC021 accepted origin lacks both block poses.",
                diagnostics={
                    "failure_phase": "bc021_origin_pose_manifest",
                    "pose_names": sorted(accepted_poses),
                },
            )
        self._b320_reset_poses.update(accepted_poses)
        self._poses.update(accepted_poses)
        self._bc021_state.reset()
        self._bc021_physics_frame_index = 0
        self._bc021_recording_origin_accepted = True
        self._bc021_recording_origin_poses = accepted_poses
        self._bc021_recording_origin_pose_sha256 = accepted_pose_sha256
        self._b320_failure_reason = None
        return {
            "accepted": True,
            "kind": "bc021_stable_physical_boundary_rebase",
            "physics_state_writes": False,
            "baseline_pose_names": sorted(accepted_poses),
            "baseline_poses_wxyz": accepted_pose_payload,
            "baseline_pose_sha256": accepted_pose_sha256,
            "baseline_source": "post_stable_origin_physics_truth",
            "pose_truth_available": True,
            "velocity_truth_available": True,
            "contact_truth_available": True,
            "geometry_contract_final": bool(
                not self._bc021_geometry.provisional
            ),
        }

    def _build_observation_space(self) -> spaces.Dict:
        result = super()._build_observation_space()
        state_space = result.spaces["state"]
        if not isinstance(state_space, spaces.Dict):
            raise RuntimeError("BC021 base state observation is not a Dict")
        # The generic rigid-B320 state also contains goal/fixture/articulation
        # and handover/stir history.  None belongs to the clean-room BC021
        # task, so do not merely append task fields to that schema.
        task_state_spaces: dict[str, spaces.Space] = {
            key: state_space.spaces[key] for key in _BASE_OBSERVATION_KEYS
        }
        task_state_spaces.update(
            {
                "bc021_block_pose": spaces.Box(
                    -np.inf, np.inf, shape=(14,), dtype=np.float64
                ),
                "bc021_marked_face_state": spaces.Box(
                    -np.inf, np.inf, shape=(20,), dtype=np.float64
                ),
                "bc021_contact_state": spaces.Box(
                    -np.inf, np.inf, shape=(15,), dtype=np.float64
                ),
                "bc021_stage_state": spaces.Box(
                    -np.inf, np.inf, shape=(12,), dtype=np.float64
                ),
                "bc021_terminal_state": spaces.Box(
                    -np.inf, np.inf, shape=(4,), dtype=np.float64
                ),
                "bc021_hidden_state": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=HIDDEN_OBSERVATION_SHAPE,
                    dtype=np.float64,
                ),
            }
        )
        result.spaces["state"] = spaces.Dict(task_state_spaces)
        return result

    def _bc021_observation_values(
        self,
        metrics: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        block_pose = np.concatenate(
            [
                _finite_vector(self._bc021_last_pose_wxyz["block_a"], (7,)),
                _finite_vector(self._bc021_last_pose_wxyz["block_b"], (7,)),
            ]
        )
        marked_face_state = np.concatenate(
            [
                _finite_vector(metrics.get("left_face_center_w"), (3,)),
                _finite_vector(metrics.get("right_face_center_w"), (3,)),
                _finite_vector(metrics.get("left_face_normal_w"), (3,)),
                _finite_vector(metrics.get("right_face_normal_w"), (3,)),
                np.asarray(
                    [
                        float(metrics.get("normal_error_deg", 0.0)),
                        float(metrics.get("lateral_error_m", 0.0)),
                        float(metrics.get("axial_gap_m", 0.0)),
                        float(metrics.get("minimum_bottom_clearance_m", 0.0)),
                        float(metrics.get("reference_translation_m", 0.0)),
                        float(metrics.get("max_reference_translation_m", 0.0)),
                        float(metrics.get("geometry_contract_final", False)),
                        float(metrics.get("motion_stable", False)),
                    ],
                    dtype=np.float64,
                ),
            ]
        )
        contact_state = np.asarray(
            [
                *[self._bc021_contact_force_n[key] for key in CONTACT_KEYS],
                *[
                    float(self._bc021_contact_readable[key])
                    for key in CONTACT_KEYS
                ],
                float(metrics.get("contact_truth_ready", False)),
                float(metrics.get("assigned_contact_current", False)),
                float(metrics.get("cross_contact_clear", False)),
                float(metrics.get("block_contact_current", False)),
                float(metrics.get("marked_face_contact_current", False)),
            ],
            dtype=np.float64,
        )
        values = {
            "bc021_block_pose": block_pose,
            "bc021_marked_face_state": marked_face_state,
            "bc021_contact_state": contact_state,
            "bc021_stage_state": self._bc021_state.stage_observation(),
            "bc021_terminal_state": self._bc021_state.terminal_observation(),
            "bc021_hidden_state": self._bc021_state.hidden_observation(),
        }
        for key, value in values.items():
            if not np.all(np.isfinite(value)):
                raise RuntimeError(f"BC021 observation {key} is non-finite")
        return values

    def _compute_observation(self) -> dict[str, Any]:
        observation = super()._compute_observation()
        inherited_state = observation.get("state")
        if not isinstance(inherited_state, dict):
            raise RuntimeError("BC021 observation is missing state mapping")
        metrics = dict(self._b320_last_metrics)
        state = {
            key: np.asarray(inherited_state[key], dtype=np.float64).copy()
            for key in _BASE_OBSERVATION_KEYS
        }
        state.update(self._bc021_observation_values(metrics))
        if tuple(state) != _STATE_OBSERVATION_KEYS:
            raise RuntimeError(
                "BC021 clean-room observation order drifted: "
                f"{tuple(state)!r}"
            )
        observation["state"] = state
        return observation

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        self._metrics(update=False)
        return [
            self._condition_row(
                f"b320_bc021_{index + 1}",
                label,
                self._b320_condition_flags[index],
            )
            for index, label in enumerate(self._b320_contract.success_labels)
        ]

    def _task_info(self) -> dict[str, Any]:
        # Bypass the generic rigid-B320 task-info serializer: it publishes
        # carton, pen and handover state which is unrelated to BC021.  Keep
        # the shared Isaac collection lifecycle info, then add only the small
        # B320 compatibility summary and the dedicated BC021 contract.
        base = super(HandsOnB320BimanualIsaacLabEnv, self)._task_info()
        info = dict(base) if isinstance(base, dict) else {}
        metrics = dict(self._metrics(update=False))
        info.update(
            {
                "task_id": self._b320_contract.task_id,
                "task_name": self.task_name,
                "b320_metrics": metrics,
                "b320_failure_reason": self._b320_failure_reason,
                "b320_success_counter": int(self._success_counter),
                "b320_success_frames_required": self._success_steps_required(),
                "b320_physical_replay_certified": False,
            }
        )
        origin_poses, origin_pose_sha256 = (
            self._bc021_recording_origin_manifest()
        )
        info.update(
            {
                "bc021_task_requirement": TASK_REQUIREMENT,
                "bc021_asset_integration_status": ASSET_INTEGRATION_STATUS,
                "bc021_asset_candidate_url": ASSET_CANDIDATE_URL,
                "bc021_asset_candidate_license": ASSET_CANDIDATE_LICENSE,
                "bc021_geometry_contract_final": bool(
                    not self._bc021_geometry.provisional
                ),
                "bc021_geometry_digest_sha256": (
                    self._bc021_geometry.digest_sha256()
                ),
                "bc021_acceptance_calibration": (
                    self._bc021_acceptance_calibration.as_dict()
                ),
                "bc021_acceptance_calibration_sha256": (
                    self._bc021_acceptance_calibration.digest_sha256()
                ),
                "bc021_acceptance_calibration_final": bool(
                    self._bc021_acceptance_calibration.final
                ),
                "bc021_contact_bindings": {
                    key: dict(value)
                    for key, value in self._bc021_contact_bindings.items()
                },
                "bc021_contact_force_n": dict(
                    self._bc021_contact_force_n
                ),
                "bc021_contact_readable": dict(
                    self._bc021_contact_readable
                ),
                "bc021_block_contact_points_readable": bool(
                    self._bc021_block_contact_points_readable
                ),
                "bc021_pose_truth_readable": bool(
                    self._bc021_pose_truth_readable
                ),
                "bc021_velocity_truth_readable": bool(
                    self._bc021_velocity_truth_readable
                ),
                "bc021_reset": self._bc021_reset_sample.as_dict(),
                "bc021_reset_clearance_point_count": int(
                    self._bc021_reset_clearance_point_count
                ),
                "bc021_reset_home_clearance_cached": bool(
                    self._bc021_reset_home_clearance_cached
                ),
                "bc021_reset_home_body_count": int(
                    self._bc021_reset_home_body_count
                ),
                "bc021_reset_home_collider_count": int(
                    self._bc021_reset_home_collider_count
                ),
                "bc021_reset_home_envelope_cover_radius_m": float(
                    _BC021_COLLIDER_ENVELOPE_COVER_RADIUS_M
                ),
                "bc021_physics_frame_index": int(
                    self._bc021_physics_frame_index
                ),
                "bc021_hidden_state_fields": list(
                    HIDDEN_OBSERVATION_FIELDS
                ),
                "bc021_hidden_state": (
                    self._bc021_state.hidden_observation().tolist()
                ),
                "bc021_stable_substeps": int(
                    self._bc021_state.stable_substeps
                ),
                "bc021_stable_substeps_required": int(
                    self._bc021_acceptance_calibration.stable_hold_substeps
                ),
                "bc021_reference_latched_frame": (
                    self._bc021_state.reference_latched_frame
                ),
                "bc021_reference_flat_max_tilt_deg": (
                    REFERENCE_FLAT_MAX_TILT_DEG
                ),
                "bc021_left_hand_lock_active": bool(
                    self._bc021_left_hand_lock_active
                ),
                "bc021_left_hand_lock_frame": (
                    self._bc021_left_hand_lock_frame
                ),
                "bc021_left_hand_lock_tcp_pose_wxyz": (
                    None
                    if self._bc021_left_hand_lock_tcp_pose is None
                    else self._bc021_left_hand_lock_tcp_pose.tolist()
                ),
                "bc021_first_success_frame": (
                    self._bc021_state.first_success_frame
                ),
                "bc021_first_terminal_reason": (
                    self._bc021_state.first_terminal_reason
                ),
                "bc021_first_terminal_frame": (
                    self._bc021_state.first_terminal_frame
                ),
                "bc021_recording_origin_accepted": bool(
                    self._bc021_recording_origin_accepted
                ),
                "bc021_recording_origin_poses_wxyz": origin_poses,
                "bc021_recording_origin_pose_sha256": origin_pose_sha256,
                "bc021_recording_origin_source": (
                    "post_stable_origin_physics_truth"
                    if origin_pose_sha256 is not None
                    else None
                ),
            }
        )
        return info

    def get_physical_replay_reset_contract(self) -> dict[str, Any]:
        contract = dict(super().get_physical_replay_reset_contract())
        origin_poses, origin_pose_sha256 = (
            self._bc021_recording_origin_manifest()
        )
        contract.update(
            {
                "bc021_task_requirement": TASK_REQUIREMENT,
                "bc021_asset_integration_status": ASSET_INTEGRATION_STATUS,
                "bc021_asset_candidate_url": ASSET_CANDIDATE_URL,
                "bc021_asset_candidate_license": ASSET_CANDIDATE_LICENSE,
                "bc021_geometry_provisional": bool(
                    self._bc021_geometry.provisional
                ),
                "bc021_geometry_digest_sha256": (
                    self._bc021_geometry.digest_sha256()
                ),
                "bc021_acceptance_calibration": (
                    self._bc021_acceptance_calibration.as_dict()
                ),
                "bc021_acceptance_calibration_sha256": (
                    self._bc021_acceptance_calibration.digest_sha256()
                ),
                "bc021_acceptance_calibration_final": bool(
                    self._bc021_acceptance_calibration.final
                ),
                "bc021_geometry_half_extents_m": {
                    "left": list(
                        self._bc021_geometry.left_half_extents_m
                    ),
                    "right": list(
                        self._bc021_geometry.right_half_extents_m
                    ),
                },
                "bc021_marked_face_frame": {
                    "left_center_local_m": self._bc021_geometry.marked_center_local(
                        "left"
                    ).tolist(),
                    "right_center_local_m": self._bc021_geometry.marked_center_local(
                        "right"
                    ).tolist(),
                    "left_normal_local": self._bc021_geometry.marked_normal_local(
                        "left"
                    ).tolist(),
                    "right_normal_local": self._bc021_geometry.marked_normal_local(
                        "right"
                    ).tolist(),
                    "left_tangents_local": [
                        value.tolist()
                        for value in self._bc021_geometry.marked_tangents_local(
                            "left"
                        )
                    ],
                    "right_tangents_local": [
                        value.tolist()
                        for value in self._bc021_geometry.marked_tangents_local(
                            "right"
                        )
                    ],
                },
                "bc021_contact_keys": list(CONTACT_KEYS),
                "bc021_contact_bindings": {
                    key: dict(value)
                    for key, value in self._bc021_contact_bindings.items()
                },
                "bc021_reset": self._bc021_reset_sample.as_dict(),
                "bc021_reset_home_clearance_cached": bool(
                    self._bc021_reset_home_clearance_cached
                ),
                "bc021_reset_home_body_count": int(
                    self._bc021_reset_home_body_count
                ),
                "bc021_reset_home_collider_count": int(
                    self._bc021_reset_home_collider_count
                ),
                "bc021_reset_home_envelope_cover_radius_m": float(
                    _BC021_COLLIDER_ENVELOPE_COVER_RADIUS_M
                ),
                "bc021_state_machine": (
                    "assigned_hands_both_airborne_marked_contact_manifold_"
                    f"{self._bc021_acceptance_calibration.stable_hold_substeps}_"
                    "substep_hold_v5_asset_bound_calibration_"
                    "complete_hidden_state"
                ),
                "bc021_hidden_state_fields": list(
                    HIDDEN_OBSERVATION_FIELDS
                ),
                "bc021_hidden_state_shape": list(HIDDEN_OBSERVATION_SHAPE),
                "bc021_reference_max_translation_m": (
                    MAX_REFERENCE_TRANSLATION_M
                ),
                "bc021_reference_flat_max_tilt_deg": (
                    REFERENCE_FLAT_MAX_TILT_DEG
                ),
                "bc021_left_hand_lock": {
                    "active": bool(self._bc021_left_hand_lock_active),
                    "frame": self._bc021_left_hand_lock_frame,
                    "tcp_pose_wxyz": (
                        None
                        if self._bc021_left_hand_lock_tcp_pose is None
                        else self._bc021_left_hand_lock_tcp_pose.tolist()
                    ),
                    "hand_target": (
                        None
                        if self._bc021_left_hand_lock_hand is None
                        else self._bc021_left_hand_lock_hand.tolist()
                    ),
                },
                "bc021_stable_hold_seconds": float(
                    self._bc021_acceptance_calibration.stable_hold_seconds
                ),
                "bc021_stable_hold_physics_hz": int(
                    self._bc021_acceptance_calibration.physics_hz
                ),
                "bc021_stable_hold_substeps": int(
                    self._bc021_acceptance_calibration.stable_hold_substeps
                ),
                "bc021_marked_contact_force_threshold_n": None,
                "bc021_motion_threshold_is_success_condition": False,
                "bc021_marker_surface_offset_m": MARKER_SURFACE_OFFSET_M,
                "bc021_release_required": False,
                "bc021_recording_origin_accepted": bool(
                    self._bc021_recording_origin_accepted
                ),
                "bc021_recording_origin_poses_wxyz": origin_poses,
                "bc021_recording_origin_pose_sha256": origin_pose_sha256,
                "bc021_recording_origin_source": (
                    "post_stable_origin_physics_truth"
                    if origin_pose_sha256 is not None
                    else None
                ),
                "bc021_formal_collection_ready": (
                    self._bc021_formal_collection_ready()
                ),
                "b320_physical_replay_certified": False,
            }
        )
        return contract


__all__ = [
    "HandsOnB320BC021IsaacLabEnv",
    "_BASE_OBSERVATION_KEYS",
    "_OBSERVATION_KEYS",
    "_STATE_OBSERVATION_KEYS",
]
