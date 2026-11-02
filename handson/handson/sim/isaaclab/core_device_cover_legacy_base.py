from __future__ import annotations

from .bimanual_runtime import BimanualRuntimeSupport

from argparse import Namespace
from collections.abc import Mapping
from typing import Any, Literal
import gc
import math
import os
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from handson.runtime.gpu import is_physx_recording_device_allowed

from ..assembly_success import (
    ASSEMBLY_STABLE_ANGULAR_SPEED,
    ASSEMBLY_STABLE_HOLD_SECONDS,
    ASSEMBLY_STABLE_LINEAR_SPEED,
    assembly_geometry,
    assembly_geometry_is_inserted,
)

from ..envs.hand_specs import (
    SIMPLIFIED_POST_IK_ARM_TYPE,
    WUJI_HAND_TYPES,
    hand_action_dim,
    is_simplified_post_ik_arm_type,
    native_hand_joint_names,
    normalize_arm_type,
    normalize_hand_type,
)
from ..isaacsim.runtime import IsaacSimUnavailable
from .compat import (
    app_launcher_is_headless,
    configure_app_launcher,
    initialize_interactive_scene,
    isaaclab_uses_xyzw,
    make_simulation_cfg,
    normalize_physics_backend,
)
from .physics_config import (
    SINGLE_ENV_GPU_MAX_RIGID_CONTACT_COUNT,
    SINGLE_ENV_GPU_MAX_RIGID_PATCH_COUNT,
)
from .physx_ground_truth import (
    PhysxGroundTruthTrace,
    configure_physx_contact_reports_for_recording,
    physx_ground_truth_enabled,
)
from .newton_ground_truth import (
    NewtonGroundTruthTrace,
    newton_execution_config,
    newton_ground_truth_enabled,
)
from .recording_cameras import (
    HEAD_RECORDING_CAMERA_EYE,
    HEAD_RECORDING_CAMERA_QUAT_WXYZ_OPENGL,
    OPERATOR_VIEWPORT_CAMERA_EYE,
    OPERATOR_VIEWPORT_CAMERA_TARGET,
    WRIST_RECORDING_CAMERA_LOCAL_POSES,
    WRIST_RECORDING_CAMERA_MOUNT_POSES,
    WRIST_RECORDING_CAMERA_OUTPUT_SIZE,
    recording_camera_profile,
    set_wrist_obstructions_hidden,
    wrist_obstruction_visibility_records,
)
from .realistic_kitchen_scene import (
    REALISTIC_KITCHEN_PREVIEW_ROOT,
    RealisticKitchenReport,
    compose_realistic_kitchen_visual,
    realistic_kitchen_preview_enabled,
)
from .runtime_display import (
    configure_isaac_renderer_launcher,
    enforce_isaac_ultra_rendering,
    enforce_isaac_viewport_only,
    isaac_app_renderer_resolution,
    isaac_kit_args,
    isaac_rendering_mode,
)


_DEFAULT_RENDER_SIZE = (640, 640)
_RANDOM_SEED_MODULUS = 2**32 - 1
_TRAJECTORY_STATE_WRITE_JOURNAL_LIMIT = 32
_RIGHT_HOME = np.asarray([-0.10, -0.16, 1.07, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
_LEFT_HOME = np.asarray([-0.10, 0.16, 1.07, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
_PEG_POSE = np.asarray([-0.10, -0.16, 0.94, 0.6964, -0.1228, 0.6964, 0.1228], dtype=np.float64)
_SOCKET_POSE = np.asarray([-0.10, 0.16, 0.92, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
_IDENTITY_QUAT_WXYZ = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
_PEG_RANDOMIZED_SAMPLING_BOUNDS = np.asarray(
    [[-0.36, -0.34], [-0.06, -0.08]],
    dtype=np.float64,
)
_SOCKET_RANDOMIZED_SAMPLING_BOUNDS = np.asarray(
    [[-0.36, 0.06], [-0.02, 0.34]],
    dtype=np.float64,
)
_PEG_RANDOMIZED_YAW_BOUNDS_DEG = np.asarray([-35.0, 35.0], dtype=np.float64)
_SOCKET_RANDOMIZED_YAW_BOUNDS_DEG = np.asarray([-50.0, 50.0], dtype=np.float64)
_ASSEMBLY_OBJECT_MIN_DIST = 0.17
_ARM_TARGET_MAX_JOINT_STEP = 0.03
_NEWTON_ARM_TARGET_MAX_JOINT_STEP = 0.08
_B03_NEWTON_HAND_TARGET_SLEW_RATE_RAD_S = 2.0
_B03_NEWTON_HAND_TARGET_SLEW_RATE_ENV = (
    "HANDSON_NEWTON_B03_HAND_TARGET_SLEW_RATE_RAD_S"
)
_SMOOTH_GRASP_PROFILE_DISABLE_ENV = "HANDSON_DISABLE_SMOOTH_GRASP_PROFILE"
_ARM_TARGET_JUMP_LIMIT = 0.45
_ARM_TARGET_MAX_SITE_STEP = 0.035
_ARM_TARGET_JUMP_DELTA = 0.10
_ARM_TARGET_POS_DEADBAND = 0.002
_ARM_TARGET_ROT_JUMP_DEG = 20.0
_ARM_TARGET_ROT_DEADBAND_DEG = 1.0
_ARM_TARGET_ALPHA = 0.25
_ARM_WORKSPACE_MAX_RADIUS = 0.78
def _torch_view(value: Any) -> Any:
    """Return a Torch view for Lab 3 ProxyArray/raw Warp values."""

    try:
        return value.torch
    except AttributeError:
        pass
    try:
        import warp as wp

        return wp.to_torch(value)
    except (ImportError, AttributeError, TypeError):
        return value


def _clone_array_like(value: Any) -> Any:
    """Clone a runtime tensor while keeping lightweight tests NumPy-only."""

    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    copy = getattr(value, "copy", None)
    if callable(copy):
        return copy()
    return np.asarray(value).copy()


def _quat_wxyz_to_isaaclab_array(quat: Any, *, uses_xyzw: bool) -> np.ndarray:
    value = np.asarray(quat).copy()
    if value.shape[-1] != 4:
        raise ValueError(f"Expected quaternion shape (..., 4), got {value.shape}")
    return value[..., (1, 2, 3, 0)] if uses_xyzw else value


def _quat_isaaclab_to_wxyz_array(quat: Any, *, uses_xyzw: bool) -> np.ndarray:
    value = np.asarray(quat).copy()
    if value.shape[-1] != 4:
        raise ValueError(f"Expected quaternion shape (..., 4), got {value.shape}")
    return value[..., (3, 0, 1, 2)] if uses_xyzw else value


def _pose_wxyz_to_isaaclab_array(pose: Any, *, uses_xyzw: bool) -> np.ndarray:
    value = np.asarray(pose).copy()
    if value.shape[-1] != 7:
        raise ValueError(f"Expected pose shape (..., 7), got {value.shape}")
    value[..., 3:7] = _quat_wxyz_to_isaaclab_array(
        value[..., 3:7],
        uses_xyzw=uses_xyzw,
    )
    return value


def _pose_isaaclab_to_wxyz_array(pose: Any, *, uses_xyzw: bool) -> np.ndarray:
    value = np.asarray(pose).copy()
    if value.shape[-1] != 7:
        raise ValueError(f"Expected pose shape (..., 7), got {value.shape}")
    value[..., 3:7] = _quat_isaaclab_to_wxyz_array(
        value[..., 3:7],
        uses_xyzw=uses_xyzw,
    )
    return value


def _end_effector_link_jacobian(
    robot: Any,
    ee_jacobi_idx: int,
    arm_joint_ids: Any,
    *,
    lab3: bool,
) -> Any:
    """Select the end-effector link-origin Jacobian across Lab 2 and 3."""

    joint_ids = [int(joint_id) for joint_id in arm_joint_ids]
    if lab3:
        joint_offset = int(robot.num_base_dofs)
        joint_ids = [joint_id + joint_offset for joint_id in joint_ids]
        full_jacobian = _torch_view(robot.data.body_link_jacobian_w)
    else:
        full_jacobian = _torch_view(robot.root_physx_view.get_jacobians())
    body_jacobian = full_jacobian[:, int(ee_jacobi_idx), :, :]
    return body_jacobian[:, :, joint_ids]


def _bool_env(name: str, default: bool = False) -> bool:
    import os

    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _str_env(name: str, default: str) -> str:
    import os

    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _float_env(name: str, default: float) -> float:
    import os

    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        q = _IDENTITY_QUAT_WXYZ
    else:
        q = q / norm
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [
                0.25 * s,
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / s,
                    0.25 * s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    0.25 * s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                ],
                dtype=np.float64,
            )
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=np.float64,
            )
    return _normalize_quat_wxyz(quat)


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        return _IDENTITY_QUAT_WXYZ.copy()
    return quat / norm


def _quat_mul_wxyz(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    q0 = _normalize_quat_wxyz(q0)
    q1 = _normalize_quat_wxyz(q1)
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return _normalize_quat_wxyz(
        np.asarray(
            [
                w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            ],
            dtype=np.float64,
        )
    )


def _clamp_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if max_norm <= 0.0 or norm <= max_norm or norm < 1e-9:
        return vec.copy()
    return vec * (max_norm / norm)


def _blend_quat_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize_quat_wxyz(q0)
    q1 = _normalize_quat_wxyz(q1)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    return _normalize_quat_wxyz((1.0 - alpha) * q0 + alpha * q1)


def _quat_delta_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = _normalize_quat_wxyz(q0)
    q1 = _normalize_quat_wxyz(q1)
    dot = float(np.clip(abs(np.dot(q0, q1)), -1.0, 1.0))
    return float(np.rad2deg(2.0 * np.arccos(dot)))


def _pose_vector_to_matrix(pose: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_wxyz_to_matrix(pose[3:7])
    matrix[:3, 3] = pose[:3]
    return matrix


def _invert_pose_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    inv_pose = np.eye(4, dtype=np.float64)
    inv_pose[:3, :3] = pose[:3, :3].T
    inv_pose[:3, 3] = -(inv_pose[:3, :3] @ pose[:3, 3])
    return inv_pose


def _time_based_random_seed() -> int:
    """Return a NumPy RandomState-compatible seed for non-replay runs."""

    return int((time.time_ns() ^ (os.getpid() << 16)) % _RANDOM_SEED_MODULUS)


class BimanualAssemblyIsaacLabEnv(BimanualRuntimeSupport, gym.Env):
    """Gym wrapper around the native Isaac Lab bimanual assembly scene."""

    # Articulated-arm tasks can safely use the atomic authored-home restore
    # during recording-origin setup.  Specialized environments with a
    # different root controller (for example E01's detached palm wrench
    # servo) explicitly opt out and keep their own reset contract.
    stable_recording_origin_default_robot_restore_enabled = True
    recorded_drive_target_replay_supported = True

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        seed: int | None = None,
        control_dt: float = 0.02,
        physics_dt: float = 1.0 / 120.0,
        time_limit: float = 120.0,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = True,
        randomize: bool = False,
        randomize_dynamics: bool = False,
        hand_type: str = "wuji1",
        arm_type: str = "rokae",
        launch_isaac: bool | None = None,
        headless: bool | None = None,
        render_resolution: tuple[int, int] = _DEFAULT_RENDER_SIZE,
        hz: int = 30,
        isaac_device: str | None = None,
        physics_backend: str = "physx",
        existing_simulation_app: Any | None = None,
        existing_app_launcher: Any | None = None,
        **_: Any,
    ):
        del randomize_dynamics
        self.hz = max(
            0,
            int(round(_float_env("HANDSON_ISAACLAB_CONTROL_HZ", float(hz)))),
        )
        self.render_mode = render_mode
        self.image_obs = bool(image_obs)
        self.randomize = bool(randomize)
        self.hand_type = normalize_hand_type(hand_type)
        self.requested_arm_type = str(arm_type)
        self.arm_control_profile = (
            SIMPLIFIED_POST_IK_ARM_TYPE
            if is_simplified_post_ik_arm_type(arm_type)
            else "articulated_live_ik"
        )
        self.arm_type = normalize_arm_type(arm_type)
        self._physics_backend = normalize_physics_backend(physics_backend)
        self._isaaclab_uses_xyzw = False
        self.recorded_drive_target_replay_supported = (
            self._physics_backend == "physx"
            or (
                self._physics_backend == "newton"
                and getattr(self, "task_name", None) == "t09_insert_peg"
                and not getattr(self, "_direct_hand_control", False)
            )
        )
        if (
            self.arm_type != "rokae"
            or self.hand_type not in WUJI_HAND_TYPES
            or (
                self._physics_backend == "newton"
                and self.hand_type != "wuji1"
            )
        ):
            raise ValueError(
                "Isaac Lab PhysX supports arm_type=rokae with "
                "hand_type=wuji1 or wuji2; Newton currently requires wuji1."
            )

        self._hand_action_dim = hand_action_dim(self.hand_type)
        self._seed = _time_based_random_seed() if seed is None else int(seed)
        self._random = np.random.RandomState(self._seed)
        self._visual_randomization_seed: int | None = None
        self._control_dt = max(
            1e-6,
            _float_env("HANDSON_ISAACLAB_CONTROL_DT", float(control_dt)),
        )
        self._physics_dt = float(physics_dt)
        self._n_substeps = max(1, int(round(self._control_dt / self._physics_dt)))
        configured_hand_slew_rate = _float_env(
            _B03_NEWTON_HAND_TARGET_SLEW_RATE_ENV,
            _B03_NEWTON_HAND_TARGET_SLEW_RATE_RAD_S,
        )
        if not np.isfinite(configured_hand_slew_rate):
            configured_hand_slew_rate = (
                _B03_NEWTON_HAND_TARGET_SLEW_RATE_RAD_S
            )
        self._newton_hand_target_slew_rate_rad_s = max(
            0.0,
            float(configured_hand_slew_rate),
        )
        self._newton_hand_target_slew_enabled = bool(
            (
                (
                    self._physics_backend == "newton"
                    and getattr(self, "task_name", None)
                    == "t09_insert_peg"
                )
                or (
                    self._physics_backend == "physx"
                    and getattr(self, "task_name", None)
                    == "t21_pass_block"
                )
            )
            and not getattr(self, "_direct_hand_control", False)
            and self._newton_hand_target_slew_rate_rad_s > 0.0
        )
        # The global PhysX soft-grasp profile was removed.  Live articulated
        # hands again install each control-frame command directly, matching the
        # pre-profile recording and replay contract.
        self._smooth_grasp_hand_target_interpolation_enabled = False
        self._time_limit = float(time_limit)
        self._render_height = int(render_resolution[0])
        self._render_width = int(render_resolution[1])
        self._image_shape = (self._render_height, self._render_width, 3)
        self._physx_ground_truth_enabled = (
            self._physics_backend == "physx" and physx_ground_truth_enabled()
        )
        # B03 records a Newton-native drive-target trace.  Keep this separate
        # from A01 and from the PhysX contact-report/stable-origin machinery.
        self._newton_ground_truth_enabled = bool(
            self._physics_backend == "newton"
            and getattr(self, "task_name", None) == "t09_insert_peg"
            and not getattr(self, "_direct_hand_control", False)
            and newton_ground_truth_enabled()
        )
        self._recording_ground_truth_enabled = bool(
            self._physx_ground_truth_enabled
            or self._newton_ground_truth_enabled
        )
        if self._physx_ground_truth_enabled:
            self._isaac_device = (
                os.environ.get("HANDSON_ISAACLAB_DEVICE", "cuda:0")
                if isaac_device is None
                else isaac_device
            )
        else:
            self._isaac_device = isaac_device or _str_env(
                "HANDSON_ISAACLAB_DEVICE",
                "cuda:0",
            )
        if self._physx_ground_truth_enabled and not is_physx_recording_device_allowed(
            self._isaac_device,
            cpu_exception=getattr(
                self,
                "_physx_cpu_ground_truth_exception",
                "",
            ),
        ):
            raise RuntimeError(
                "PhysX ground-truth recording requires an explicit canonical "
                "CUDA device, except for a schema-bound exact-asset CPU "
                "contract; got "
                f"{self._isaac_device!r}."
            )

        if launch_isaac is None:
            launch_isaac = not _bool_env("HANDSON_ISAACSIM_INTERFACE_ONLY")
        self._launch_isaac = bool(launch_isaac)
        self._headless = render_mode != "human" if headless is None else bool(headless)

        self._app_launcher = existing_app_launcher
        self._simulation_app = existing_simulation_app
        self._sim = None
        self._scene = None
        self._stage = None
        self._realistic_kitchen_preview_enabled = (
            realistic_kitchen_preview_enabled()
        )
        self._realistic_kitchen_preview_report: (
            RealisticKitchenReport | None
        ) = None
        self._torch = None
        self._sim_dt = self._physics_dt
        self._joint_targets: dict[str, Any] = {}
        # The raw MANUS/Tracker command remains in ``_{side}_hand``.  This
        # second buffer holds the joint-limit-clamped destination while
        # ``_joint_targets`` remains the exact target applied on the previous
        # physics substep.  Keeping those roles separate is required both for
        # responsive teleoperation and exact drive-target recording/replay.
        self._newton_hand_goal_targets: dict[str, Any] = {}
        self._smooth_grasp_hand_start_targets: dict[str, Any] = {}
        self._newton_hand_live_goal_serial = 0
        self._hand_joint_ids: dict[str, list[int]] = {"right": [], "left": []}
        self._arm_joint_ids: dict[str, Any] = {}
        self._ee_body_indices: dict[str, int] = {}
        self._ee_jacobi_indices: dict[str, int] = {}
        self._ik_controllers: dict[str, Any] = {}
        self._math_utils: dict[str, Any] = {}
        self._randomize_scene_visuals = None
        self._reference_robot_visual_roots: tuple[str, ...] = ()
        self._haorui_rokae = None
        self._haorui_ik_models: dict[str, Any] = {}
        self._haorui_ik_solvers: dict[str, Any] = {}
        self._haorui_position_ik_solvers: dict[str, Any] = {}
        self._haorui_ik_enabled = False
        self._arm_ik_backend = self._normalize_arm_ik_backend(
            _str_env("HANDSON_ISAACLAB_ARM_IK_BACKEND", "dls")
        )
        self._arm_pose_filters: dict[str, dict[str, np.ndarray] | None] = {"right": None, "left": None}
        self._isaac_warnings: list[str] = []
        self._wrist_obstruction_visibility: tuple[tuple[str, Any], ...] | None = None
        self._wrist_obstructions_hidden: bool | None = None
        self._recording_camera_pose_cache: dict[str, np.ndarray] = {}
        self.recording_camera_profile = recording_camera_profile()
        self._hand_target_control = True
        self._timeline = None
        self._timeline_pause_subscription = None
        self._keyboard_input_interface = None
        self._keyboard = None
        self._keyboard_subscription = None
        self._manual_reset_requested = False
        self._manual_tracker_rebase_requested = False
        self._manual_reset_already_applied = False
        self._manual_reset_viewport_camera_state: dict[str, Any] | None = None
        self._keyboard_reset_in_progress = False
        self._last_control_log_time = 0.0
        self._control_perf_window_start = time.monotonic()
        self._control_perf_steps = 0
        self._control_perf_phase_totals: dict[str, float] = {}
        self._right_pose_commanded = False
        self._left_pose_commanded = False
        self._reset_physics_steps = 0
        self._physx_ground_truth_trace: (
            PhysxGroundTruthTrace | NewtonGroundTruthTrace | None
        ) = None
        self._physx_scene_generation = 0
        self._recording_origin_authored_reset_snapshot: (
            dict[str, np.ndarray] | None
        ) = None
        self._recording_origin_authored_reset_serial = 0
        self._recording_origin_reset_write_capture_active = False
        self._recording_origin_authored_task_reset_writes: dict[
            str, dict[str, np.ndarray]
        ] = {}
        self._trajectory_action_phase_active = False
        self._trajectory_action_phase_substep_index = -1
        self._task_physics_substep_abort_requested = False
        self._trajectory_state_write_count = 0
        self._trajectory_state_write_journal: list[dict[str, Any]] = []
        self._trajectory_state_write_journal_dropped = 0
        self._trajectory_state_write_origin: dict[str, Any] | None = None

        self.delta_h = np.zeros(1, dtype=np.float64)
        self._right_tcp_pose = _RIGHT_HOME.copy()
        self._left_tcp_pose = _LEFT_HOME.copy()
        self._right_hand = np.zeros(self._hand_action_dim, dtype=np.float64)
        self._left_hand = np.zeros(self._hand_action_dim, dtype=np.float64)
        self._peg_ori_pose = _PEG_POSE.copy()
        self._socket_ori_pose = _SOCKET_POSE.copy()
        self.env_step = 0
        self._success_counter = 0
        self._last_assembly_success_metrics: dict[str, Any] = {}

        self.action_space = spaces.Box(
            low=np.full(7 + self._hand_action_dim, -1.0, dtype=np.float32),
            high=np.full(7 + self._hand_action_dim, 1.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = self._build_observation_space()

        if self._launch_isaac:
            try:
                self._setup_isaaclab()
                self._spawn_realistic_kitchen_preview()
            except BaseException:
                self.close()
                raise
        self._initialize_physx_ground_truth_recording()

    def _in_process_scene_rebuild_kwargs(self, seed: int) -> dict[str, Any]:
        """Return the constructor recipe for a fresh scene in the current app."""

        return {
            "seed": int(seed),
            "control_dt": float(self._control_dt),
            "physics_dt": float(self._physics_dt),
            "time_limit": float(self._time_limit),
            "render_mode": self.render_mode,
            "image_obs": bool(self.image_obs),
            "randomize": bool(self.randomize),
            "hand_type": self.hand_type,
            "arm_type": self.arm_type,
            "launch_isaac": True,
            "headless": bool(self._headless),
            "render_resolution": (
                int(self._render_height),
                int(self._render_width),
            ),
            "hz": int(self.hz),
            "isaac_device": str(self._isaac_device),
            "physics_backend": str(self._physics_backend),
        }


    @staticmethod
    def _release_camera_for_scene_rebuild(camera: Any) -> bool:
        """Release one Isaac Lab camera before its USD Stage disappears.

        Isaac Lab cameras own RTX render products and annotator attachments in
        their private renderer object.  Ordinarily ``Camera.__del__`` releases
        those resources, but an in-process Stage replacement must not leave
        the order of that destructor relative to ``close_stage()`` up to the
        cyclic garbage collector.  Detach while the old Stage is still valid,
        then clear the renderer reference so the later destructor cannot
        release the same Hydra texture a second time.
        """

        renderer = getattr(camera, "_renderer", None)
        render_data = getattr(camera, "_render_data", None)
        cleanup = getattr(renderer, "cleanup", None)
        released = False
        cleanup_error: Exception | None = None
        if callable(cleanup):
            try:
                cleanup(render_data)
            except Exception as exc:
                cleanup_error = exc
            else:
                released = True
                try:
                    camera._renderer = None
                    camera._render_data = None
                except Exception:
                    pass

        clear_callbacks = getattr(camera, "_clear_callbacks", None)
        callback_error: Exception | None = None
        if callable(clear_callbacks):
            try:
                clear_callbacks()
            except Exception as exc:
                callback_error = exc

        # CameraData owns the GPU output buffers and the view retains the old
        # Stage paths.  They are not useful after renderer cleanup and keeping
        # either alive across ``SimulationContext.clear_instance`` can retain
        # stale Fabric/RTX state.
        for name, value in (
            ("_data", None),
            ("_view", None),
            ("_sensor_prims", []),
        ):
            if hasattr(camera, name):
                try:
                    setattr(camera, name, value)
                except Exception:
                    pass
        if cleanup_error is not None:
            raise cleanup_error
        if callback_error is not None:
            raise callback_error
        return released

    def _release_scene_cameras_for_rebuild(self) -> int:
        """Release every distinct standalone or scene-owned camera."""

        candidates: list[Any] = []
        cameras = getattr(self, "_cameras", None)
        if isinstance(cameras, Mapping):
            candidates.extend(cameras.values())

        scene = getattr(self, "_scene", None)
        try:
            scene_sensors = getattr(scene, "sensors", None)
        except Exception:
            scene_sensors = None
        if isinstance(scene_sensors, Mapping):
            candidates.extend(scene_sensors.values())

        released = 0
        seen: set[int] = set()
        for camera in candidates:
            identity = id(camera)
            if identity in seen:
                continue
            seen.add(identity)
            # Camera initializes ``_renderer`` to None before the first
            # physics-ready event.  Such a camera still owns Stage callbacks
            # that must be detached explicitly, whereas non-camera sensors do
            # not expose this attribute at all.
            if not hasattr(camera, "_renderer"):
                continue
            try:
                released += int(self._release_camera_for_scene_rebuild(camera))
            except Exception as exc:
                print(
                    "[HandsOn/IsaacLab] Camera cleanup before scene rebuild "
                    f"failed for {type(camera).__name__}: {exc}",
                    flush=True,
                )
        return released

    def _release_scene_overlay_for_rebuild(self) -> bool:
        """Detach the task overlay from the viewport before it changes Stage."""

        released = False
        frame = getattr(self, "_success_condition_frame", None)
        if frame is not None:
            released = True
            try:
                frame.visible = False
            except Exception:
                pass
            clear = getattr(frame, "clear", None)
            if callable(clear):
                try:
                    clear()
                except Exception:
                    pass
            self._success_condition_frame = None

        window = getattr(self, "_success_condition_window", None)
        if window is not None:
            released = True
            try:
                window.visible = False
            except Exception:
                pass
            destroy = getattr(window, "destroy", None)
            if callable(destroy):
                try:
                    destroy()
                except Exception:
                    pass
            self._success_condition_window = None

        for name in (
            "_success_condition_labels",
            "_success_condition_text_labels",
        ):
            labels = getattr(self, name, None)
            if hasattr(labels, "clear"):
                try:
                    labels.clear()
                except Exception:
                    pass
        if hasattr(self, "_task_status_label"):
            self._task_status_label = None
        return released

    def _prepare_scene_for_in_process_rebuild(self) -> dict[str, Any]:
        """Quiesce visual resources while the current Stage is still valid."""

        wrist_visibility_restored = False
        if (
            getattr(self, "_wrist_obstruction_visibility", None) is not None
            or getattr(self, "_wrist_obstructions_hidden", None) is True
        ):
            try:
                self._set_wrist_obstructions_hidden(False)
                wrist_visibility_restored = True
            except Exception as exc:
                print(
                    "[HandsOn/IsaacLab] Could not restore wrist visuals before "
                    f"scene rebuild: {exc}",
                    flush=True,
                )

        # Deferred render paths can temporarily hide full arm visual bodies.
        # Restore them too so no invisible authored opinions survive an
        # asynchronous viewport/Stage handoff.
        if getattr(self, "_deferred_robot_body_visibility", None) is not None:
            restore_deferred = getattr(
                self,
                "_set_deferred_robot_body_hidden",
                None,
            )
            if callable(restore_deferred):
                try:
                    restore_deferred(False)
                except Exception as exc:
                    print(
                        "[HandsOn/IsaacLab] Could not restore deferred robot "
                        f"visuals before scene rebuild: {exc}",
                        flush=True,
                    )

        close_preview = getattr(self, "_close_direct_hand_preview", None)
        if callable(close_preview):
            try:
                close_preview()
            except Exception as exc:
                print(
                    "[HandsOn/IsaacLab] Could not close direct-hand preview "
                    f"before scene rebuild: {exc}",
                    flush=True,
                )

        overlay_released = self._release_scene_overlay_for_rebuild()
        camera_count = self._release_scene_cameras_for_rebuild()
        return {
            "wrist_visibility_restored": wrist_visibility_restored,
            "overlay_released": overlay_released,
            "camera_renderers_released": camera_count,
        }

    def hard_reset_physx_scene(self, *, seed: int) -> dict[str, Any]:
        """Replace the USD/PhysX scene without restarting Kit or the process.

        A normal environment reset only writes public actor state.  This path
        creates a new USD stage and ``SimulationContext``, which makes the
        PhysX plugin release the old ``PxScene`` and therefore drops contact
        manifolds, friction anchors, warm-start impulses, islands, and sleep
        history.  The Python object identity is retained so existing Gym and
        teleoperation wrappers remain attached to the rebuilt environment.
        """

        if self._simulation_app is None or self._sim is None:
            raise RuntimeError(
                "An in-process PhysX hard reset requires a live Isaac app and scene."
            )

        environment_type = type(self)
        constructor_kwargs = self._in_process_scene_rebuild_kwargs(int(seed))
        simulation_app = self._simulation_app
        app_launcher = self._app_launcher
        viewport_camera_state = getattr(
            self,
            "_manual_reset_viewport_camera_state",
            None,
        )
        if viewport_camera_state is None:
            viewport_camera_state = self._capture_viewport_camera_state()
        next_generation = int(self._physx_scene_generation) + 1
        preserved_attributes = {
            name: getattr(self, name)
            for name in (
                "task_overlay_title",
                "task_overlay_prompt",
                "task_overlay_prompt_getter",
                "_fast_shutdown_on_close",
            )
            if hasattr(self, name)
        }

        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.sim import SimulationContext

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._unsubscribe_scene_input_callbacks()
        clear_callbacks = getattr(self._sim, "clear_all_callbacks", None)
        if callable(clear_callbacks):
            clear_callbacks()
        del clear_callbacks

        visual_cleanup = self._prepare_scene_for_in_process_rebuild()
        # Renderer detach and UI destruction enqueue Kit-side work.  Service it
        # while both the old Stage and its render products are still alive;
        # otherwise cleanup races ``SimulationContext.clear_instance()``.
        simulation_app.update()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(
            "[HandsOn/IsaacLab] Scene visual resources released before hard "
            f"reset: {visual_cleanup}",
            flush=True,
        )

        # Camera, tensor, articulation and contact views must die before the
        # old stage is detached.  Holding one of these views across a stage
        # replacement can leave it pointing at released PhysX memory.
        self.__dict__.clear()
        gc.collect()

        try:
            SimulationContext.clear_instance()
            sim_utils.create_new_stage()
            simulation_app.update()

            # Re-run the concrete constructor on this same Python object.
            # Constructing a temporary replacement and copying __dict__ would
            # leave Kit subscriptions bound to the temporary object's identity.
            environment_type.__init__(
                self,
                existing_simulation_app=simulation_app,
                existing_app_launcher=app_launcher,
                **constructor_kwargs,
            )
            # Let the replacement visualizer bind to the new Stage and finish
            # its first Fabric/RTX synchronization before any viewport state is
            # restored.  Two Kit turns are required by the asynchronous
            # viewport initialization in Isaac Sim 6 / Isaac Lab 3.
            simulation_app.update()
            simulation_app.update()
        except BaseException:
            # Do not let ordinary shutdown accidentally close a still-running
            # app through stale scene handles after a failed reconstruction.
            self._simulation_app = simulation_app
            self._app_launcher = app_launcher
            raise

        self._physx_scene_generation = next_generation
        self.__dict__.update(preserved_attributes)
        if viewport_camera_state is not None:
            self._restore_viewport_camera_state(viewport_camera_state)
        return {
            "scene_rebuilt": True,
            "scene_generation": next_generation,
            "seed": int(seed),
            "process_restarted": False,
            "simulation_app_reused": True,
        }

    def _capture_viewport_camera_state(self) -> dict[str, Any] | None:
        """Copy the active free-camera state before replacing the USD stage."""
        if getattr(self, "_headless", True) or self._stage is None:
            return None
        try:
            from omni.kit.viewport.utility import get_active_viewport
            from omni.kit.viewport.utility.camera_state import ViewportCameraState

            viewport = get_active_viewport()
            if viewport is None:
                return None
            camera_path = str(viewport.camera_path)
            camera_prim = self._stage.GetPrimAtPath(camera_path)
            if not camera_prim.IsValid():
                return None

            camera_state = ViewportCameraState(
                camera_path=camera_path,
                viewport=viewport,
            )
            eye = tuple(float(value) for value in camera_state.position_world)
            target = tuple(float(value) for value in camera_state.target_world)

            # USD values such as Gf vectors and Vt arrays are independent value
            # objects.  Keeping them does not retain the old Stage or any PhysX
            # view, but it does preserve the viewport's orbit center and lens.
            authored_attributes = {
                str(attribute.GetName()): attribute.Get()
                for attribute in camera_prim.GetAttributes()
                if attribute.HasAuthoredValueOpinion()
                and attribute.Get() is not None
            }
            return {
                "camera_path": camera_path,
                "eye": eye,
                "target": target,
                "attributes": authored_attributes,
            }
        except Exception as exc:
            message = f"Could not preserve viewport camera before scene rebuild: {exc}"
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)
                print(f"[HandsOn/IsaacLab] {message}", flush=True)
            return None

    def _restore_viewport_camera_state(self, state: dict[str, Any]) -> bool:
        """Restore a free-camera state after a fresh USD stage is constructed."""
        if getattr(self, "_headless", True) or self._stage is None:
            return False
        try:
            from omni.kit.viewport.utility import get_active_viewport
            from pxr import Sdf, UsdGeom

            viewport = get_active_viewport()
            if viewport is None:
                return False

            camera_path = str(state.get("camera_path") or "/OmniverseKit_Persp")
            camera_prim = self._stage.GetPrimAtPath(camera_path)
            if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
                camera_path = "/OmniverseKit_Persp"
                camera_prim = self._stage.GetPrimAtPath(camera_path)
            if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
                return self._reset_viewport_camera_to_default(announce=False)

            eye = tuple(state.get("eye") or ())
            target = tuple(state.get("target") or ())
            has_world_view = len(eye) == 3 and len(target) == 3
            restored = 0

            # Isaac Lab 3's SimulationContext.set_camera_view accepts only
            # ``eye`` and ``target``.  Select the captured camera on the
            # viewport first so that API updates the intended prim rather than
            # silently reframing the replacement Stage's default camera.
            viewport.camera_path = Sdf.Path(camera_path)
            if self._simulation_app is not None:
                self._simulation_app.update()

            # Run two passes around a Kit update.  The first update lets the
            # new viewport attach to the replacement Stage; the second pass
            # wins over its asynchronous free-camera initialization.
            for _ in range(2):
                if has_world_view:
                    self._sim.set_camera_view(
                        eye=eye,
                        target=target,
                    )
                for name, value in dict(state.get("attributes") or {}).items():
                    if has_world_view and (
                        name.startswith("xformOp")
                        or name == "omni:kit:centerOfInterest"
                    ):
                        continue
                    attribute = camera_prim.GetAttribute(name)
                    if not attribute.IsValid():
                        continue
                    attribute.Set(value)
                    restored += 1
                viewport.camera_path = Sdf.Path(camera_path)
                if self._simulation_app is not None:
                    self._simulation_app.update()
            if not has_world_view and restored == 0:
                return self._reset_viewport_camera_to_default(announce=False)

            print(
                "[HandsOn/IsaacLab] Viewport camera restored after scene rebuild "
                f"(eye={eye}, target={target})",
                flush=True,
            )
            return True
        except Exception as exc:
            message = f"Could not restore viewport camera after scene rebuild: {exc}"
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)
                print(f"[HandsOn/IsaacLab] {message}", flush=True)
            return self._reset_viewport_camera_to_default(announce=False)

    def _default_viewport_camera_view(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        spec = getattr(self, "_spec", None)
        if spec is not None and hasattr(spec, "head_eye") and hasattr(spec, "head_target"):
            return tuple(spec.head_eye), tuple(spec.head_target)
        return OPERATOR_VIEWPORT_CAMERA_EYE, OPERATOR_VIEWPORT_CAMERA_TARGET

    def _reset_viewport_camera_to_default(self, *, announce: bool = True) -> bool:
        """Return the interactive viewport to this task's configured view."""
        if self._sim is None:
            return False
        try:
            eye, target = self._default_viewport_camera_view()
            self._sim.set_camera_view(eye=eye, target=target)
            if self._simulation_app is not None:
                self._simulation_app.update()
            if announce:
                print(
                    "[HandsOn/IsaacLab] Viewport camera reset to task default",
                    flush=True,
                )
            return True
        except Exception as exc:
            message = f"Could not reset viewport camera: {exc}"
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)
                print(f"[HandsOn/IsaacLab] {message}", flush=True)
            return False


    def _build_observation_space(self) -> spaces.Dict:
        state_space = spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14,)),
                "gripper_pose": spaces.Box(-np.inf, np.inf, shape=(32,)),
                "socket_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "peg_ori_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "table_delta_height": spaces.Box(-np.inf, np.inf, shape=(1,)),
            }
        )
        observation_spaces: dict[str, Any] = {"state": state_space}
        if self.image_obs:
            wrist_width, wrist_height = WRIST_RECORDING_CAMERA_OUTPUT_SIZE
            wrist_shape = (wrist_height, wrist_width, 3)
            observation_spaces["images"] = spaces.Dict(
                {
                    "wrist_left": spaces.Box(0, 255, shape=wrist_shape, dtype=np.uint8),
                    "wrist_right": spaces.Box(0, 255, shape=wrist_shape, dtype=np.uint8),
                    "random_camera" if self.randomize else "ego": spaces.Box(
                        0,
                        255,
                        shape=self._image_shape,
                        dtype=np.uint8,
                    ),
                }
            )
        return spaces.Dict(observation_spaces)

    def _initialize_physx_ground_truth_recording(self) -> None:
        """Install the selected recording-only simulator readback contract."""

        if not self._recording_ground_truth_enabled:
            return
        trace_type = (
            NewtonGroundTruthTrace
            if self._newton_ground_truth_enabled
            else PhysxGroundTruthTrace
        )
        self._physx_ground_truth_trace = trace_type(self)
        trace_prefix = "newton_" if self._newton_ground_truth_enabled else "physx_"
        observation_spaces = {
            name: space
            for name, space in self.observation_space.spaces.items()
            if not name.startswith(trace_prefix)
        }
        for name, (shape, dtype) in (
            self._physx_ground_truth_trace.observation_space_specs.items()
        ):
            if dtype == np.dtype(np.uint8):
                observation_spaces[name] = spaces.Box(
                    0,
                    np.iinfo(dtype).max,
                    shape=shape,
                    dtype=dtype,
                )
            elif np.issubdtype(dtype, np.unsignedinteger):
                observation_spaces[name] = spaces.Box(
                    0,
                    np.iinfo(dtype).max,
                    shape=shape,
                    dtype=dtype,
                )
            elif np.issubdtype(dtype, np.signedinteger):
                observation_spaces[name] = spaces.Box(
                    np.iinfo(dtype).min,
                    np.iinfo(dtype).max,
                    shape=shape,
                    dtype=dtype,
                )
            else:
                observation_spaces[name] = spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=shape,
                    dtype=dtype,
                )
        self.observation_space = spaces.Dict(observation_spaces)

    def _capture_recording_origin_authored_reset_snapshot(
        self,
        *,
        require_authored_task_writes: bool = True,
    ) -> None:
        """Save the authored reset state before its first solver step.

        ``reset()`` advances PhysX once before returning.  Capturing here is
        therefore the only point at which the stability gate can distinguish
        the authored reset from a state that was already displaced while
        resolving its first contact manifold.  A task may explicitly adopt a
        freshly constructed and already-settled scene; that path has no reset
        setters to audit and keeps the live snapshot unchanged.
        """

        if not self._physx_ground_truth_enabled:
            self._recording_origin_reset_write_capture_active = False
            return
        trace = self._physx_ground_truth_trace
        if trace is None:
            self._initialize_physx_ground_truth_recording()
            trace = self._physx_ground_truth_trace
        if trace is None:
            raise RuntimeError(
                "Could not capture the authored reset before its first PhysX "
                "step because the ground-truth trace is unavailable."
            )
        snapshot = trace.snapshot()
        required_task_reset_paths: dict[str, int] = (
            {
                str(state[0]): 0
                for state in getattr(self, "_task_reset_body_states", ())
                if state
            }
            if require_authored_task_writes
            else {}
        )
        required_task_root_writes: dict[str, bool] = {
            path: True for path in required_task_reset_paths
        }
        root_is_writable = getattr(
            self,
            "_task_articulation_root_is_writable",
            None,
        )
        dofs_are_writable = getattr(
            self,
            "_task_articulation_dofs_are_writable",
            None,
        )
        for state in (
            getattr(self, "_task_reset_articulation_states", ())
            if require_authored_task_writes
            else ()
        ):
            if not state:
                continue
            dof_count = 0
            if len(state) > 5 and state[5] is not None:
                try:
                    dof_count = int(state[5].shape[-1])
                except (AttributeError, IndexError, TypeError, ValueError):
                    dof_count = -1
            path = str(state[0])
            root_write_required = (
                True
                if not callable(root_is_writable)
                else bool(root_is_writable(path))
            )
            dof_write_required = (
                True
                if not callable(dofs_are_writable)
                else bool(dofs_are_writable(path))
            )
            if not root_write_required and not dof_write_required:
                continue
            required_task_reset_paths[path] = (
                dof_count if dof_write_required else 0
            )
            required_task_root_writes[path] = root_write_required
        snapshot = self._apply_authored_task_reset_writes(
            trace,
            snapshot,
            self._recording_origin_authored_task_reset_writes,
            required_task_reset_paths=required_task_reset_paths,
            required_task_root_writes=required_task_root_writes,
        )
        self._recording_origin_authored_reset_snapshot = {
            name: np.asarray(value).copy()
            for name, value in snapshot.items()
        }
        self._recording_origin_authored_reset_serial += 1
        self._recording_origin_reset_write_capture_active = False

    def _begin_recording_origin_reset_write_capture(self) -> None:
        """Start one exact-write journal for the pre-solver reset boundary."""

        self._recording_origin_authored_task_reset_writes = {}
        self._recording_origin_reset_write_capture_active = bool(
            self._physx_ground_truth_enabled
        )

    def _record_trajectory_state_write(
        self,
        writer: str,
        *,
        subject: str | None = None,
    ) -> None:
        """Record a successful direct state write during an action interval.

        Stable-origin settling calls ``_step_isaaclab`` directly and therefore
        never enters this action phase.  The bounded journal is diagnostic;
        the two boolean origin fields are the fail-closed publication signal.
        """

        if not self._trajectory_action_phase_active:
            return

        origin = getattr(self, "_physical_recording_origin", None)
        if isinstance(origin, dict) and origin is not self._trajectory_state_write_origin:
            self._trajectory_state_write_origin = origin
            self._trajectory_state_write_count = 0
            self._trajectory_state_write_journal = []
            self._trajectory_state_write_journal_dropped = 0

        self._trajectory_state_write_count += 1
        entry: dict[str, Any] = {
            "write_index": int(self._trajectory_state_write_count),
            "control_step": int(self.env_step),
            "physics_substep": int(
                self._trajectory_action_phase_substep_index
            ),
            "writer": str(writer),
        }
        if subject:
            entry["subject"] = str(subject)
        if len(self._trajectory_state_write_journal) < (
            _TRAJECTORY_STATE_WRITE_JOURNAL_LIMIT
        ):
            self._trajectory_state_write_journal.append(entry)
        else:
            self._trajectory_state_write_journal_dropped += 1

        if not isinstance(origin, dict):
            return
        origin["trajectory_per_frame_state_writes"] = True
        origin["per_frame_state_writes"] = True
        origin["trajectory_state_write_count"] = int(
            self._trajectory_state_write_count
        )
        origin["trajectory_state_write_journal_limit"] = int(
            _TRAJECTORY_STATE_WRITE_JOURNAL_LIMIT
        )
        origin["trajectory_state_write_journal_dropped"] = int(
            self._trajectory_state_write_journal_dropped
        )
        origin["trajectory_state_write_journal"] = [
            dict(value) for value in self._trajectory_state_write_journal
        ]

    @staticmethod
    def _reset_write_array(value: Any, *, width: int) -> np.ndarray:
        """Copy one tensor row without relying on an asynchronous readback."""

        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        array = np.asarray(value, dtype=np.float64)
        if array.size == 0 or array.shape[-1] != width:
            raise ValueError(
                "Authored reset write has an invalid tensor shape: "
                f"expected trailing width {width}, got {array.shape}"
            )
        row = array.reshape(-1, width)[0].copy()
        if not np.all(np.isfinite(row)):
            raise ValueError("Authored reset write contains non-finite values")
        return row

    def _record_authored_task_reset_write(
        self,
        path: str,
        *,
        transform_xyzw: Any | None = None,
        velocity: Any | None = None,
        dof_position: Any | None = None,
        dof_velocity: Any | None = None,
    ) -> None:
        """Journal exact task tensors passed to PhysX before solver step zero.

        GPU tensor setters are asynchronous: querying the same articulation
        view immediately after a root write can return the previous episode's
        cached pose.  The journal records the public setter arguments, while
        the first post-write PhysX sample still proves that the write took
        effect through the normal reset-drift gate.
        """

        if not self._recording_origin_reset_write_capture_active:
            return
        entry = self._recording_origin_authored_task_reset_writes.setdefault(
            str(path),
            {},
        )
        if transform_xyzw is not None:
            entry["transform_xyzw"] = self._reset_write_array(
                transform_xyzw,
                width=7,
            )
        if velocity is not None:
            entry["velocity"] = self._reset_write_array(velocity, width=6)
        if dof_position is not None:
            position = np.asarray(
                dof_position.detach().cpu().numpy()
                if hasattr(dof_position, "detach")
                else dof_position,
                dtype=np.float64,
            ).reshape(-1)
            if not np.all(np.isfinite(position)):
                raise ValueError(
                    "Authored reset DOF position contains non-finite values"
                )
            entry["dof_position"] = position.copy()
        if dof_velocity is not None:
            velocity_array = np.asarray(
                dof_velocity.detach().cpu().numpy()
                if hasattr(dof_velocity, "detach")
                else dof_velocity,
                dtype=np.float64,
            ).reshape(-1)
            if not np.all(np.isfinite(velocity_array)):
                raise ValueError(
                    "Authored reset DOF velocity contains non-finite values"
                )
            entry["dof_velocity"] = velocity_array.copy()

    @staticmethod
    def _apply_authored_task_reset_writes(
        trace: PhysxGroundTruthTrace,
        snapshot: dict[str, np.ndarray],
        writes: Mapping[str, Mapping[str, np.ndarray]],
        *,
        required_task_reset_paths: Mapping[str, int] | None = None,
        required_task_root_writes: Mapping[str, bool] | None = None,
    ) -> dict[str, np.ndarray]:
        """Overlay exact public setter arguments on stale GPU view readback.

        The validity masks remain the live PhysX masks.  Inactive parked
        variants therefore stay invalid/zero, while active task actors get the
        exact pose, velocity and DOF state submitted at the authored boundary.
        The first solver sample is still an independent tensor readback and
        must match these values within the strict reset-drift limits.
        """

        for path, expected_dof_count in (
            required_task_reset_paths or {}
        ).items():
            entry = writes.get(str(path))
            root_write_required = bool(
                (required_task_root_writes or {}).get(str(path), True)
            )
            if root_write_required and (
                entry is None or "transform_xyzw" not in entry
            ):
                raise RuntimeError(
                    "Authored reset write journal is incomplete for task "
                    f"path {path!r}"
                )
            if int(expected_dof_count) < 0:
                raise RuntimeError(
                    "Authored reset articulation DOF count is unavailable "
                    f"for {path!r}"
                )
            if int(expected_dof_count) > 0:
                for field_name in ("dof_position", "dof_velocity"):
                    values = None if entry is None else entry.get(field_name)
                    if values is None or np.asarray(values).size != int(
                        expected_dof_count
                    ):
                        raise RuntimeError(
                            "Authored reset write journal is incomplete for "
                            f"{path!r}: {field_name} expected "
                            f"{expected_dof_count} values"
                        )

        result = {
            name: np.asarray(value).copy()
            for name, value in snapshot.items()
        }

        def state_from_write(
            entry: Mapping[str, np.ndarray],
            current: np.ndarray,
        ) -> np.ndarray:
            state = np.asarray(current, dtype=np.float64).copy()
            transform = entry.get("transform_xyzw")
            if transform is not None:
                pose = np.asarray(transform, dtype=np.float64).reshape(7)
                state[:3] = pose[:3]
                state[3:7] = pose[[6, 3, 4, 5]]
            velocity = entry.get("velocity")
            if velocity is not None:
                state[7:13] = np.asarray(
                    velocity,
                    dtype=np.float64,
                ).reshape(6)
            if not np.all(np.isfinite(state)):
                raise ValueError(
                    "Authored task reset overlay contains non-finite state"
                )
            return state

        rigid_states = result.get("physx_task_rigid_body_state")
        rigid_valid = result.get("physx_task_rigid_body_valid")
        if rigid_states is not None and rigid_valid is not None:
            for index, body in enumerate(trace.rigid_bodies):
                entry = writes.get(str(body.name))
                if entry is None or not bool(np.asarray(rigid_valid)[index]):
                    continue
                rigid_states[index] = state_from_write(
                    entry,
                    rigid_states[index],
                )

        articulation_states = result.get(
            "physx_task_articulation_root_state"
        )
        articulation_valid = result.get(
            "physx_task_articulation_root_valid"
        )
        dof_positions = result.get(
            "physx_task_articulation_dof_position"
        )
        dof_velocities = result.get(
            "physx_task_articulation_dof_velocity"
        )
        dof_valid = result.get("physx_task_articulation_dof_valid")
        for index, articulation in enumerate(trace.articulations):
            entry = writes.get(str(articulation.name))
            if entry is None:
                continue
            if (
                articulation_states is not None
                and articulation_valid is not None
                and bool(np.asarray(articulation_valid)[index])
            ):
                articulation_states[index] = state_from_write(
                    entry,
                    articulation_states[index],
                )
            dof_count = int(articulation.dof_count)
            if dof_count <= 0 or dof_valid is None:
                continue
            valid_row = np.asarray(dof_valid)[index, :dof_count].astype(bool)
            position = entry.get("dof_position")
            if position is not None and dof_positions is not None:
                values = np.asarray(position, dtype=np.float64).reshape(-1)
                if values.size != dof_count:
                    raise ValueError(
                        "Authored reset DOF position count changed: "
                        f"path={articulation.name}, expected={dof_count}, "
                        f"actual={values.size}"
                    )
                dof_positions[index, :dof_count][valid_row] = values[valid_row]
            velocity = entry.get("dof_velocity")
            if velocity is not None and dof_velocities is not None:
                values = np.asarray(velocity, dtype=np.float64).reshape(-1)
                if values.size != dof_count:
                    raise ValueError(
                        "Authored reset DOF velocity count changed: "
                        f"path={articulation.name}, expected={dof_count}, "
                        f"actual={values.size}"
                    )
                dof_velocities[index, :dof_count][valid_row] = values[valid_row]
        return result

    def get_recording_schema(self) -> dict[str, Any] | None:
        """Return the optional action-aligned simulator ground-truth schema."""

        if not self._recording_ground_truth_enabled:
            return None
        if self._physx_ground_truth_trace is None:
            self._initialize_physx_ground_truth_recording()
        if self._physx_ground_truth_trace is None:
            return None
        schema = dict(self._physx_ground_truth_trace.schema())
        schema["reset_replay_contract"] = (
            self.get_physical_replay_reset_contract()
        )
        return schema

    def get_physical_replay_reset_contract(self) -> dict[str, Any]:
        """Describe the deterministic boundary immediately before action[0]."""

        episode_reset_seed = getattr(self, "_seed", None)
        if bool(getattr(self, "_newton_ground_truth_enabled", False)):
            solver_settings = dict(
                getattr(self, "_newton_solver_settings", {})
            )
            return {
                "strategy": (
                    "fresh_newton_process_seeded_environment_reset_then_exact_"
                    "recorded_per_substep_drive_targets"
                ),
                "requires_collection_run_seed": True,
                "requires_episode_reset_seed": True,
                "requires_fresh_process": True,
                "allow_pre_action_state_overwrite": False,
                "allows_single_recorded_t0_restore": False,
                "forbids_state_writes_after_action_start": True,
                "pre_action_observation_index": 0,
                "recorded_idle_actions_before_intervention": True,
                "physics_steps_after_reset_state_write": int(
                    self._reset_physics_steps
                ),
                "physics_dt": float(self._physics_dt),
                "control_dt": float(self._control_dt),
                "control_substeps": int(self._n_substeps),
                "device": str(self._isaac_device),
                "newton": {
                    "solver": str(solver_settings.get("solver", "newton")),
                    "integrator": str(
                        solver_settings.get("integrator", "implicitfast")
                    ),
                    "iterations": int(solver_settings.get("iterations", 100)),
                    "line_search_iterations": int(
                        solver_settings.get("ls_iterations", 50)
                    ),
                    "ccd_iterations": int(
                        solver_settings.get("ccd_iterations", 50)
                    ),
                    "njmax": solver_settings.get("njmax"),
                    "nconmax": solver_settings.get("nconmax"),
                    "use_mujoco_contacts": bool(
                        solver_settings.get("use_mujoco_contacts", False)
                    ),
                    "use_newton_actuators": True,
                    "hand_target_slew": {
                        "enabled": bool(
                            self._newton_hand_target_slew_enabled
                        ),
                        "scope": "b03_articulated_newton",
                        "rate_rad_s": float(
                            self._newton_hand_target_slew_rate_rad_s
                        ),
                        "maximum_delta_per_physics_substep_rad": float(
                            self._newton_hand_target_slew_rate_rad_s
                            * self._physics_dt
                        ),
                        "reference": "previous_applied_drive_target",
                        "raw_hand_command_mutated": False,
                        "recording": "actual_applied_per_substep_target",
                        "replay": (
                            "exact_recorded_pre_substep_targets_bypass_slew"
                        ),
                    },
                    "execution": newton_execution_config(self),
                    "solver_history_reset": dict(
                        getattr(
                            self,
                            "_newton_solver_history_reset_contract",
                            {},
                        )
                    ),
                },
                "episode_reset_seed": (
                    None
                    if episode_reset_seed is None
                    else int(episode_reset_seed)
                ),
            }
        contract = {
            "strategy": (
                "seeded_environment_reset_then_exact_recorded_physx_t0_restore"
            ),
            "requires_collection_run_seed": True,
            "requires_episode_reset_seed": True,
            "allow_pre_action_state_overwrite": True,
            "allows_single_recorded_t0_restore": True,
            "forbids_state_writes_after_action_start": not bool(
                getattr(self, "_simplified_post_ik", False)
            ),
            "deterministic_kinematic_hand_inputs": bool(
                getattr(self, "_simplified_post_ik", False)
            ),
            "pre_action_observation_index": 0,
            "recorded_idle_actions_before_intervention": True,
            "physics_steps_after_exact_recorded_t0_restore": 0,
            "physics_steps_after_reset_state_write": int(
                self._reset_physics_steps
            ),
            "physics_dt": float(self._physics_dt),
            "control_dt": float(self._control_dt),
            "control_substeps": int(self._n_substeps),
            "device": str(self._isaac_device),
            "visual_randomization": {
                "enabled": bool(getattr(self, "randomize", False)),
                "episode_reset_seed": (
                    None
                    if episode_reset_seed is None
                    else int(episode_reset_seed)
                ),
                "visual_seed": getattr(
                    self,
                    "_visual_randomization_seed",
                    None,
                ),
                "seed_derivation": (
                    "episode_reset_seed_xor_0x5eed5eed_mod_2pow32"
                ),
                "scope": "background_table_materials_and_lighting",
                "applied_once_per_reset": True,
            },
            "physx": {
                "solver_type": 1,
                "enable_enhanced_determinism": True,
                "max_position_iteration_count": 64,
                "max_velocity_iteration_count": 4,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": int(
                    SINGLE_ENV_GPU_MAX_RIGID_CONTACT_COUNT
                ),
                "gpu_max_rigid_patch_count": int(
                    SINGLE_ENV_GPU_MAX_RIGID_PATCH_COUNT
                ),
                "smooth_grasp_hand_target_interpolation": {
                    "enabled": bool(
                        getattr(
                            self,
                            "_smooth_grasp_hand_target_interpolation_enabled",
                            False,
                        )
                    ),
                    "scope": "disabled_pre_global_profile",
                    "mode": "linear_previous_applied_target_to_control_goal",
                    "samples_per_control_step": int(self._n_substeps),
                    "final_substep_reaches_control_goal": True,
                    "recording": "actual_applied_per_substep_target",
                    "replay": "exact_recorded_pre_substep_targets_bypass_interpolation",
                },
            },
        }
        stable_origin = getattr(self, "_physical_recording_origin", None)
        if isinstance(stable_origin, dict):
            contract["stable_recording_origin"] = dict(stable_origin)
            contract["strategy"] = (
                "seeded_stable_reset_then_exact_recorded_physx_t0_restore"
            )
        return contract

    def _attach_physx_ground_truth(
        self,
        observation: dict[str, Any],
        *,
        reuse_latest_substep: bool = False,
    ) -> dict[str, Any]:
        if not self._recording_ground_truth_enabled:
            return observation
        if self._physx_ground_truth_trace is None:
            self._initialize_physx_ground_truth_recording()
        if self._physx_ground_truth_trace is None:
            raise RuntimeError("Simulator ground-truth trace failed to initialize.")
        result = dict(observation)
        if reuse_latest_substep:
            result.update(
                self._physx_ground_truth_trace.snapshot_from_latest_substep()
            )
        else:
            result.update(self._physx_ground_truth_trace.snapshot())
        result.update(
            self._physx_ground_truth_trace.substep_snapshot(consume=True)
        )
        return result

    def _spawn_realistic_kitchen_preview(self) -> None:
        """Compose the shared kitchen in the standalone assembly scene."""

        if not self._realistic_kitchen_preview_enabled:
            return
        if self._stage is None:
            raise RuntimeError(
                "Assembly stage is unavailable before kitchen composition"
            )
        self._realistic_kitchen_preview_report = (
            compose_realistic_kitchen_visual(
                self._stage,
                root_path=REALISTIC_KITCHEN_PREVIEW_ROOT,
                table_prim_path="/World/envs/env_0/Table",
                floor_prim_path="/World/Ground",
                transform_op_name="sharedKitchen",
                log_label="bimanual_assembly",
                align_to_table=True,
            )
        )

    def _app_launcher_device(self) -> str:
        """Return Kit's renderer device, normally shared with simulation."""

        return self._isaac_device

    def _setup_isaaclab(self) -> None:
        import os

        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        try:
            from isaaclab.app import AppLauncher
        except Exception as exc:  # pragma: no cover - exercised in Isaac runtime
            raise IsaacSimUnavailable("Isaac Lab AppLauncher is not available in this Python environment.") from exc

        self._isaaclab_uses_xyzw = isaaclab_uses_xyzw()
        launcher_args = Namespace(
            headless=self._headless,
            enable_cameras=self.image_obs,
            width=self._render_width,
            height=self._render_height,
            device=self._app_launcher_device(),
        )
        if self._isaaclab_uses_xyzw:
            launcher_args.visualizer = None
        rendering_mode = isaac_rendering_mode()
        if rendering_mode is not None:
            launcher_args.rendering_mode = rendering_mode
        configure_isaac_renderer_launcher(launcher_args)
        launcher_args = configure_app_launcher(launcher_args)
        launch_headless = app_launcher_is_headless(launcher_args)
        self._headless = launch_headless
        launcher_args.width, launcher_args.height = isaac_app_renderer_resolution(
            self._render_width,
            self._render_height,
            headless=launch_headless,
            isaaclab_v3=self._isaaclab_uses_xyzw,
        )
        kit_args = isaac_kit_args(headless=launch_headless)
        if kit_args:
            launcher_args.kit_args = kit_args
        if self._simulation_app is None:
            from .compat import create_app_launcher
            self._app_launcher = create_app_launcher(AppLauncher, launcher_args)
            self._simulation_app = self._app_launcher.app
        enforce_isaac_ultra_rendering(self._simulation_app)
        enforce_isaac_viewport_only(self._simulation_app, headless=launch_headless)
        self._setup_timeline_pause_guard()
        self._setup_keyboard_reset()

        try:
            try:
                from isaacsim.core.utils.extensions import enable_extension
            except ModuleNotFoundError:
                from isaacsim.core.experimental.utils.impl.app import (
                    enable_extension,
                )

            enable_extension("isaacsim.asset.importer.mjcf")
            self._simulation_app.update()
        except Exception as exc:
            self._isaac_warnings.append(f"Could not enable MJCF importer: {exc}")

        try:
            import torch
            import isaaclab.sim as sim_utils
            from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
            from isaaclab.scene import InteractiveScene
            from isaaclab.sim import SimulationContext
            from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms
        except Exception as exc:  # pragma: no cover - exercised in Isaac runtime
            raise IsaacSimUnavailable("Isaac Lab APIs are not available in this Python environment.") from exc

        self._ik_controller_cls = DifferentialIKController
        self._ik_controller_cfg_cls = DifferentialIKControllerCfg
        self._math_utils = {
            "matrix_from_quat": matrix_from_quat,
            "quat_inv": quat_inv,
            "subtract_frame_transforms": subtract_frame_transforms,
        }

        from .bimanual_assembly_scene import (
            apply_scene_visuals,
            fix_robot_bases,
            make_bimanual_assembly_scene_cfg,
            randomize_scene_visuals,
            validate_asset_files,
        )

        self._torch = torch
        self._randomize_scene_visuals = randomize_scene_visuals
        validate_asset_files()

        sim_cfg = make_simulation_cfg(
            sim_utils,
            physics_backend=self._physics_backend,
            device=self._isaac_device,
            dt=self._physics_dt,
            render_interval=max(1, int(round(self._control_dt / self._physics_dt))),
            gravity=(0.0, 0.0, -9.81),
            physx_kwargs={
                "solver_type": 1,
                "enable_enhanced_determinism": True,
                "max_position_iteration_count": 64,
                "max_velocity_iteration_count": 4,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": SINGLE_ENV_GPU_MAX_RIGID_CONTACT_COUNT,
                "gpu_max_rigid_patch_count": SINGLE_ENV_GPU_MAX_RIGID_PATCH_COUNT,
            },
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
        )
        self._sim = SimulationContext(sim_cfg)
        self._sim.set_camera_view(
            eye=OPERATOR_VIEWPORT_CAMERA_EYE,
            target=OPERATOR_VIEWPORT_CAMERA_TARGET,
        )

        print("[HandsOn/IsaacLab] Constructing native collection scene...", flush=True)
        scene_cfg = make_bimanual_assembly_scene_cfg(
            num_envs=1,
            env_spacing=2.5,
            physics_backend=self._physics_backend,
            hand_type=self.hand_type,
        )
        self._scene = InteractiveScene(scene_cfg)
        initialize_interactive_scene(self._sim, self._scene)
        import omni.usd

        self._stage = omni.usd.get_context().get_stage()
        self._reference_robot_visual_roots = (
            "/World/envs/env_0/RightRobot",
            "/World/envs/env_0/LeftRobot",
            "/World/envs/env_0/CenterRig",
        )
        apply_scene_visuals(1)
        fix_robot_bases(1)
        if self._physx_ground_truth_enabled:
            task_contact_paths = [
                str(prim.GetPath())
                for prim in self._stage.Traverse()
                if str(prim.GetPath()).endswith(("/Peg", "/Socket"))
            ]
            self._physx_contact_sensor_paths_preconfigured = tuple(
                task_contact_paths
            )
            self._physx_contact_report_paths_preconfigured = (
                configure_physx_contact_reports_for_recording(
                    self._stage,
                    task_contact_paths,
                )
            )
        engine_label = "NVIDIA Newton/MuJoCo-Warp" if self._physics_backend == "newton" else "PhysX"
        print(f"[HandsOn/IsaacLab] Starting {engine_label} for collection...", flush=True)
        self._sim.reset()
        self._sim_dt = float(self._sim.get_physics_dt())
        self._cache_joint_targets()
        self._init_haorui_rokae_ik()
        self._reset_scene_state()
        self._refresh_object_poses()
        print("[HandsOn/IsaacLab] Native bimanual assembly collection ready", flush=True)

    def reapply_reference_robot_appearance(self) -> int:
        """Refresh the visual-only photo-matched finish for replay renders."""

        if self._stage is None or not self._reference_robot_visual_roots:
            return 0
        import isaaclab.sim as sim_utils

        from .bimanual_assembly_scene import apply_reference_robot_appearance

        configured = apply_reference_robot_appearance(
            self._stage,
            robot_root_paths=self._reference_robot_visual_roots,
        )
        if configured:
            sim_utils.update_stage()
        return int(configured)


    def _setup_keyboard_reset(self) -> None:
        if self._headless:
            return
        try:
            import carb.input
            import omni.appwindow

            app_window = omni.appwindow.get_default_app_window()
            if app_window is None:
                raise RuntimeError("default app window is unavailable")
            keyboard = app_window.get_keyboard()
            input_interface = carb.input.acquire_input_interface()
            self._keyboard = keyboard
            self._keyboard_input_interface = input_interface
            self._keyboard_subscription = input_interface.subscribe_to_keyboard_events(
                keyboard,
                lambda event, *args, **kwargs: self._on_keyboard_event(carb.input, event),
            )
            print(
                "[HandsOn/IsaacLab] Keyboard controls enabled: R resets; "
                "Z rebases frozen tracking; V restores the task view; "
                "Space is ignored",
                flush=True,
            )
        except Exception as exc:
            message = f"Keyboard reset unavailable: {exc}"
            self._isaac_warnings.append(message)
            print(f"[HandsOn/IsaacLab] {message}", flush=True)

    def _on_keyboard_event(self, carb_input, event) -> bool:
        event_type = getattr(event, "type", None)
        key_input = getattr(event, "input", None)
        press_type = getattr(carb_input.KeyboardEventType, "KEY_PRESS", None)
        event_type_name = str(event_type).lower()
        is_press = event_type == press_type or ("press" in event_type_name and "release" not in event_type_name)
        key_name = str(key_input).lower()
        is_r_key = key_input == getattr(carb_input.KeyboardInput, "R", None) or key_name in {"r", "keyboardinput.r"}
        is_v_key = key_input == getattr(carb_input.KeyboardInput, "V", None) or key_name in {
            "v",
            "keyboardinput.v",
        }
        is_z_key = key_input == getattr(
            carb_input.KeyboardInput,
            "Z",
            None,
        ) or key_name in {
            "z",
            "keyboardinput.z",
        }
        is_space_key = key_input == getattr(carb_input.KeyboardInput, "SPACE", None) or key_name in {
            "space",
            "keyboardinput.space",
        }
        if is_press and is_space_key:
            operator_space_action = getattr(
                self,
                "_handle_operator_space_action",
                None,
            )
            if callable(operator_space_action):
                operator_space_action()
            # Returning False consumes the event before Kit's default Space
            # shortcut can pause the Timeline.  If that shortcut ran first,
            # resume here as well so SimulationContext.step() cannot block.
            self._resume_timeline_if_paused(source="Space pause")
            return False
        if is_press and is_r_key:
            # Freeze the camera before Kit or the recorder gets another event
            # cycle.  Reading it only when the USD Stage is later destroyed is
            # too late: Kit's own R handling may already have reframed it.
            self._manual_reset_viewport_camera_state = (
                self._capture_viewport_camera_state()
            )
            self._manual_reset_requested = True
            # The recorder owns the reset transaction: it discards the current
            # trajectory, recreates the spool, re-randomizes the scene, waits
            # for the stable recording origin, then rebases teleoperation.
            # Resetting here as well used to create a second, concurrent scene
            # reset and let Kit's default R shortcut see the same key event.
            print(
                "[HandsOn/IsaacLab] Keyboard reset requested; queued for recorder "
                f"(camera_frozen={self._manual_reset_viewport_camera_state is not None})",
                flush=True,
            )
            return False
        if is_press and is_z_key:
            # The teleoperation wrapper consumes this one-shot request before
            # sampling its next action.  It asks Mode A for the current stable
            # tracker pose and maps that pose onto the palm's *current* PhysX
            # pose, so an operator can escape a strict reconnect hold without
            # teleporting the simulated hand.
            self._manual_tracker_rebase_requested = True
            print(
                "[HandsOn/IsaacLab] Tracker rebase requested; keep the glove "
                "still briefly while Z remaps it to the current simulated pose",
                flush=True,
            )
            return False
        if is_press and is_v_key:
            self._reset_viewport_camera_to_default()
            return False
        return True

    def consume_manual_tracker_rebase_request(self) -> bool:
        """Consume one operator-authorized, no-jump tracker rebase request."""

        requested = bool(self._manual_tracker_rebase_requested)
        self._manual_tracker_rebase_requested = False
        return requested


    def _init_haorui_rokae_ik(self) -> None:
        position_multistart_enabled = (
            self._enable_position_only_multistart_ik()
        )
        if self._arm_ik_backend != "haorui" and not position_multistart_enabled:
            self._haorui_ik_enabled = False
            print("[HandsOn/IsaacLab] ROKAE arm IK backend=isaaclab_dls", flush=True)
            return
        try:
            from ..envs.rokae_wuji_ik_replay import rokae as rokae08

            self._haorui_rokae = rokae08
            self._haorui_ik_models = {
                "right": rokae08.load_model("right"),
                "left": rokae08.load_model("left"),
            }
            self._haorui_ik_solvers = (
                {
                    side: rokae08.make_solver(side)
                    for side in self._haorui_ik_models
                }
                if self._arm_ik_backend == "haorui"
                else {}
            )
            self._haorui_position_ik_solvers = (
                {
                    side: rokae08.make_solver(side, position_only=True)
                    for side in self._haorui_ik_models
                }
                if position_multistart_enabled
                else {}
            )
            for solver in (
                *self._haorui_ik_solvers.values(),
                *self._haorui_position_ik_solvers.values(),
            ):
                solver.warmup()
            self._haorui_ik_enabled = True
            backend_label = (
                "haorui_new"
                if self._arm_ik_backend == "haorui"
                else "isaaclab_dls+haorui_position_multistart"
            )
            print(
                f"[HandsOn/IsaacLab] ROKAE arm IK backend={backend_label}",
                flush=True,
            )
        except Exception as exc:
            self._haorui_ik_enabled = False
            message = f"Haorui ROKAE IK unavailable; fallback=isaaclab_dls reason={exc}"
            self._isaac_warnings.append(message)
            print(f"[HandsOn/IsaacLab] {message}", flush=True)


    def _differential_ik_command_type(self, side: str) -> str:
        """Return the IsaacLab differential-IK command type for one arm."""

        del side
        return "pose"

    def _use_position_only_differential_ik(self, side: str) -> bool:
        """Return whether this control frame should ignore orientation error."""

        del side
        return False

    def _enable_position_only_multistart_ik(self) -> bool:
        """Return whether this task needs the optional global position solver."""

        return False

    def _use_position_only_multistart_ik(self, side: str) -> bool:
        """Return whether this control frame should try the global solver."""

        del side
        return False

    def _position_only_differential_ik_nullspace_gain(self, side: str) -> float:
        """Return the task-specific gain for biasing redundant joints home."""

        del side
        return 0.0

    def _cache_joint_targets(self) -> None:
        self._joint_targets = {}
        self._newton_hand_goal_targets = {}
        self._smooth_grasp_hand_start_targets = {}
        self._newton_hand_live_goal_serial = 0
        self._hand_joint_ids = {"right": [], "left": []}
        self._arm_joint_ids = {}
        self._ee_body_indices = {}
        self._ee_jacobi_indices = {}
        self._ik_controllers = {}
        self._ik_command_types = {}
        for side in ("right", "left"):
            robot = self._control_robot(side)
            target = _torch_view(robot.data.default_joint_pos).clone()
            self._joint_targets[side] = target

            arm_joint_names = [
                f"joint{index}_{side}(_1)?" for index in range(1, 8)
            ]
            arm_joint_ids = robot.find_joints(arm_joint_names)[0]
            self._arm_joint_ids[side] = arm_joint_ids
            ee_body_idx = robot.find_bodies(f"{side}_palm_link")[0][0]
            self._ee_body_indices[side] = int(ee_body_idx)
            # Isaac Lab omits the root row only for fixed-base Jacobians.  A
            # floating-base articulation retains the root at body index zero.
            self._ee_jacobi_indices[side] = int(ee_body_idx) - int(
                bool(robot.is_fixed_base)
            )
            command_type = self._differential_ik_command_type(side)
            if command_type not in {"pose", "position"}:
                raise ValueError(
                    f"Unsupported differential IK command type for {side}: "
                    f"{command_type!r}"
                )
            self._ik_command_types[side] = command_type
            self._ik_controllers[side] = self._ik_controller_cls(
                self._ik_controller_cfg_cls(
                    command_type=command_type,
                    use_relative_mode=False,
                    ik_method="dls",
                    ik_params={"lambda_val": 0.04},
                ),
                num_envs=1,
                device=robot.device,
            )

            joint_names = list(robot.joint_names)
            for joint_name in native_hand_joint_names(self.hand_type, side):
                if joint_name in joint_names:
                    self._hand_joint_ids[side].append(joint_names.index(joint_name))
            self._sync_newton_hand_target_slew_state(side, target)
            print(
                f"[HandsOn/IsaacLab] {side} control map: "
                f"{len(arm_joint_ids)} arm joints, {len(self._hand_joint_ids[side])} hand joints, "
                f"ee_body={robot.body_names[self._ee_body_indices[side]]}",
                flush=True,
            )

    def _sync_newton_hand_target_slew_state(
        self,
        side: str,
        applied_target: Any,
    ) -> None:
        """Synchronize live hand motion state to one applied drive target."""

        if not (
            self._newton_hand_target_slew_enabled
            or getattr(
                self,
                "_smooth_grasp_hand_target_interpolation_enabled",
                False,
            )
        ):
            self._newton_hand_goal_targets.pop(side, None)
            self._smooth_grasp_hand_start_targets.pop(side, None)
            return
        joint_ids = list(self._hand_joint_ids.get(side, ()))
        if not joint_ids:
            self._newton_hand_goal_targets.pop(side, None)
            self._smooth_grasp_hand_start_targets.pop(side, None)
            return
        target = _torch_view(applied_target)
        hand_target = _clone_array_like(target[:, joint_ids])
        self._newton_hand_goal_targets[side] = hand_target
        self._smooth_grasp_hand_start_targets[side] = _clone_array_like(
            hand_target
        )

    def _reset_robot_control_state(self) -> None:
        for side in ("right", "left"):
            robot = self._control_robot(side)
            root_state = _torch_view(robot.data.default_root_state).clone()
            if self._scene is not None and hasattr(self._scene, "env_origins"):
                root_state[:, :3] += self._scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

        self._restore_default_robot_joint_state(
            reset_controllers=True,
            sync=False,
        )

    def _restore_default_robot_joint_state(
        self,
        *,
        reset_controllers: bool = True,
        sync: bool = False,
    ) -> dict[str, Any]:
        """Atomically restore robot joints and every matching control target.

        A joint-state teleport alone is not a complete reset for an implicit
        actuator.  Its previous position target remains live and accelerates
        the joint again on the next PhysX frame.  Keep the simulator state,
        actuator buffers, and Python-side command state on the same default
        pose so the pre-action recording boundary cannot inherit an old
        teleoperation target.
        """

        targets: dict[str, Any] = {}
        self._newton_hand_live_goal_serial = 0
        for side in ("right", "left"):
            try:
                robot = self._control_robot(side)
            except RuntimeError:
                continue

            joint_pos = _torch_view(robot.data.default_joint_pos).clone()
            if self._torch is not None:
                joint_vel = self._torch.zeros_like(joint_pos)
            else:
                joint_vel = _torch_view(robot.data.default_joint_vel).clone()
                zero_ = getattr(joint_vel, "zero_", None)
                if callable(zero_):
                    zero_()
                else:
                    joint_vel[...] = 0.0

            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.set_joint_position_target(joint_pos)
            set_velocity_target = getattr(
                robot,
                "set_joint_velocity_target",
                None,
            )
            if callable(set_velocity_target):
                set_velocity_target(joint_vel)
            write_data = getattr(robot, "write_data_to_sim", None)
            if callable(write_data):
                write_data()

            target = joint_pos.clone()
            self._joint_targets[side] = target
            targets[side] = target.clone()

            joint_ids = list(self._hand_joint_ids.get(side, ()))
            hand_command = getattr(self, f"_{side}_hand", None)
            if joint_ids and isinstance(hand_command, np.ndarray):
                values = (
                    target[:, joint_ids]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                count = min(hand_command.size, values.size)
                hand_command[:count] = values[:count]

            self._sync_newton_hand_target_slew_state(side, target)

            setattr(self, f"_{side}_pose_commanded", False)

        self._arm_pose_filters = {"right": None, "left": None}
        if reset_controllers:
            for controller in self._ik_controllers.values():
                reset = getattr(controller, "reset", None)
                if callable(reset):
                    reset()

        if sync:
            if self._sim is not None:
                self._sim.forward()
            if self._scene is not None:
                try:
                    self._scene.update(0.0)
                except Exception:
                    pass
            for side in targets:
                try:
                    self._control_robot(side).update(0.0)
                except Exception:
                    pass
        return targets


    def reset(self, seed: int | None = None, **_: Any):
        if seed is not None:
            self._seed = int(seed)
            self._random.seed(self._seed)

        if self._manual_reset_already_applied:
            self._manual_reset_already_applied = False
            self._manual_reset_requested = False
            self._manual_reset_viewport_camera_state = None
            self._refresh_object_poses()
            self._sync_b320_tcp_command_poses_to_physical_state()
            info = {
                "succeed": False,
                "manual_reset_applied": True,
                "isaac_warnings": tuple(self._isaac_warnings),
            }
            info.update(self._task_info())
            return self._attach_physx_ground_truth(self._compute_observation()), info

        self._begin_recording_origin_reset_write_capture()
        self._sample_reset_state(clear_manual_request=True)
        self._reset_scene_state()
        self._apply_reset_visual_randomization()
        self._reset_physics_steps = 0
        self._capture_recording_origin_authored_reset_snapshot()
        self._step_isaaclab()
        self._reset_physics_steps += 1
        self._refresh_object_poses()
        self._sync_b320_tcp_command_poses_to_physical_state()
        info = {
            "succeed": False,
            "isaac_warnings": tuple(self._isaac_warnings),
        }
        info.update(self._task_info())
        return self._attach_physx_ground_truth(self._compute_observation()), info

    def _sync_b320_tcp_command_poses_to_physical_state(self) -> None:
        """Use the reset PhysX palms as B320's Cartesian command zero."""

        if not self._is_b320_collection_task() or not self._launch_isaac:
            return
        if not getattr(self, "_robots", None):
            return
        try:
            right_matrix, left_matrix = self.get_end_effector_pose_matrix()
            for side, value in (
                ("right", right_matrix),
                ("left", left_matrix),
            ):
                matrix = np.asarray(value, dtype=np.float64).reshape(4, 4)
                if not np.all(np.isfinite(matrix)):
                    raise ValueError(f"{side} physical palm pose is non-finite")
                pose = np.concatenate(
                    [
                        matrix[:3, 3],
                        _matrix_to_quat_wxyz(matrix[:3, :3]),
                    ]
                )
                setattr(self, f"_{side}_tcp_pose", pose)
            print(
                "[HandsOn/IsaacLab] B320 TCP command zero synchronized to "
                "the reset PhysX palms.",
                flush=True,
            )
        except Exception as exc:
            message = f"B320 physical TCP reset synchronization failed: {exc}"
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)
                print(f"[HandsOn/IsaacLab] {message}", flush=True)

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        invalidate_trace = getattr(
            getattr(self, "_physx_ground_truth_trace", None),
            "invalidate_latest_batched_state",
            None,
        )
        if callable(invalidate_trace):
            invalidate_trace()
        self.delta_h = np.asarray(
            [self._random.uniform(0.0, 0.05) if self.randomize else 0.0],
            dtype=np.float64,
        )
        self._right_tcp_pose = _RIGHT_HOME.copy()
        self._left_tcp_pose = _LEFT_HOME.copy()
        self._right_hand.fill(0.0)
        self._left_hand.fill(0.0)
        self._newton_hand_goal_targets = {}
        self._smooth_grasp_hand_start_targets = {}
        self._newton_hand_live_goal_serial = 0
        self._peg_ori_pose, self._socket_ori_pose = self._sample_initial_object_poses()
        self.env_step = 0
        self._success_counter = 0
        self._last_assembly_success_metrics = {}
        self._right_pose_commanded = False
        self._left_pose_commanded = False
        if clear_manual_request:
            self._manual_reset_requested = False
            # A normal reset has now completed without needing a future Stage
            # rebuild, so an R-key snapshot must not leak into another episode.
            self._manual_reset_viewport_camera_state = None
        self._arm_pose_filters = {"right": None, "left": None}

    def _apply_reset_visual_randomization(self) -> None:
        if not self.randomize:
            self._visual_randomization_seed = None
            return
        random_state = self._visual_random_state_for_reset()
        if self._launch_isaac:
            try:
                self._randomize_scene_visuals(1, random_state)
            except Exception as exc:
                message = f"Visual randomization unavailable: {exc}"
                if message not in self._isaac_warnings:
                    self._isaac_warnings.append(message)
                    print(f"[HandsOn/IsaacLab] {message}", flush=True)

    def _visual_random_state_for_reset(self) -> np.random.RandomState:
        """Return an RNG derived only from this episode's reset seed.

        Keeping visual draws off ``self._random`` means adding a new texture or
        light parameter cannot perturb the physical object reset sequence.
        """
        self._visual_randomization_seed = (
            int(self._seed) ^ 0x5EED5EED
        ) % (2**32)
        return np.random.RandomState(self._visual_randomization_seed)

    def _recording_camera_pose_changed(
        self,
        name: str,
        pose: np.ndarray,
    ) -> bool:
        """Cache camera transforms so unchanged USD poses are not re-authored."""
        value = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        previous = self._recording_camera_pose_cache.get(name)
        if previous is not None and np.allclose(
            previous,
            value,
            rtol=0.0,
            atol=1e-10,
        ):
            return False
        self._recording_camera_pose_cache[name] = value.copy()
        return True

    def _sample_initial_object_poses(self) -> tuple[np.ndarray, np.ndarray]:
        peg_pose = _PEG_POSE.copy()
        socket_pose = _SOCKET_POSE.copy()
        if self.randomize:
            peg_xy, socket_xy = self._sample_assembly_object_xy()
            peg_pose[:2] = peg_xy
            socket_pose[:2] = socket_xy
            peg_pose[3:7] = self._sample_yawed_quat(
                _PEG_POSE[3:7],
                _PEG_RANDOMIZED_YAW_BOUNDS_DEG,
            )
            socket_pose[3:7] = self._sample_yawed_quat(
                _SOCKET_POSE[3:7],
                _SOCKET_RANDOMIZED_YAW_BOUNDS_DEG,
            )
        peg_pose[2] += float(self.delta_h[0])
        socket_pose[2] += float(self.delta_h[0])
        return peg_pose, socket_pose

    def _sample_assembly_object_xy(self) -> tuple[np.ndarray, np.ndarray]:
        peg_xy = _PEG_POSE[:2].copy()
        socket_xy = _SOCKET_POSE[:2].copy()
        for _ in range(64):
            peg_xy = self._random.uniform(
                _PEG_RANDOMIZED_SAMPLING_BOUNDS[0],
                _PEG_RANDOMIZED_SAMPLING_BOUNDS[1],
            )
            socket_xy = self._random.uniform(
                _SOCKET_RANDOMIZED_SAMPLING_BOUNDS[0],
                _SOCKET_RANDOMIZED_SAMPLING_BOUNDS[1],
            )
            if float(np.linalg.norm(peg_xy - socket_xy)) >= _ASSEMBLY_OBJECT_MIN_DIST:
                return peg_xy.astype(np.float64), socket_xy.astype(np.float64)
        return peg_xy.astype(np.float64), socket_xy.astype(np.float64)

    def _sample_yawed_quat(self, init_quat: np.ndarray, yaw_bounds_deg: np.ndarray) -> np.ndarray:
        yaw = np.deg2rad(self._random.uniform(float(yaw_bounds_deg[0]), float(yaw_bounds_deg[1])))
        yaw_quat = np.asarray([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)
        return _quat_mul_wxyz(yaw_quat, init_quat)

    def step(self, action):
        start_time = time.monotonic()

        # Task environments may receive asynchronous control-plane events
        # (for example a reset request from an external preview window).  Keep
        # the robot stepping implementation shared while allowing those events
        # to be consumed before the manual-reset guard below.
        self._before_control_step()

        # Keyboard events are delivered while Kit is pumping a simulation
        # frame.  If R was handled between recorder iterations, return the
        # reset transition before an action from the old episode can be
        # applied.
        if self._manual_reset_requested:
            return self._manual_reset_step_result()

        self._apply_action(action)
        self._apply_joint_targets()
        return self.step_with_current_joint_targets(start_time=start_time)

    def _before_control_step(self) -> None:
        """Hook for task-only control-plane work before an action is applied."""
        return None


    def _step_with_current_joint_targets_impl(
        self,
        *,
        start_time: float | None = None,
    ):
        """Implement one action interval while the write audit is active."""

        if start_time is None:
            start_time = time.monotonic()
        if self._manual_reset_requested:
            return self._manual_reset_step_result()

        physx_trace = getattr(self, "_physx_ground_truth_trace", None)
        phase_seconds: dict[str, float] = {}
        phase_started = time.monotonic()
        if physx_trace is not None:
            physx_trace.begin_substep_capture()
        phase_seconds["trace_begin"] = time.monotonic() - phase_started
        prepare_seconds = 0.0
        physics_seconds = 0.0
        capture_seconds = 0.0
        post_substep_seconds = 0.0
        self._task_physics_substep_abort_requested = False
        for substep_index in range(self._n_substeps):
            self._trajectory_action_phase_substep_index = int(substep_index)
            phase_started = time.monotonic()
            self._prepare_physics_substep_joint_targets(
                substep_index,
                self._n_substeps,
            )
            prepare_seconds += time.monotonic() - phase_started
            phase_started = time.monotonic()
            self._step_isaaclab()
            physics_seconds += time.monotonic() - phase_started
            phase_started = time.monotonic()
            if physx_trace is not None:
                physx_trace.capture_substep()
            after_capture = getattr(
                self,
                "_after_physics_substep_trace_capture",
                None,
            )
            if callable(after_capture):
                after_capture(physx_trace)
            capture_seconds += time.monotonic() - phase_started
            phase_started = time.monotonic()
            post_substep = getattr(
                self,
                "_handson_post_physics_substep_callback",
                None,
            )
            if callable(post_substep):
                post_substep(substep_index, self._n_substeps)
            post_substep_seconds += time.monotonic() - phase_started
            if self._task_physics_substep_abort_requested:
                break
            # A keyboard request can arrive while Kit pumps this physics
            # frame.  Consume it only after the complete control interval so
            # every recorded transition always contains the configured,
            # replayable number of substeps.  The added reset latency is at
            # most one control frame.
        finish_trace = getattr(
            physx_trace,
            "finish_substep_capture",
            None,
        )
        if callable(finish_trace):
            # B03 Newton batches all eight physics snapshots on the solver's
            # CUDA stream and performs its sole synchronized D2H here.  Do it
            # before task pose/success consumers so they can reuse the owned
            # final-substep CPU state instead of synchronizing independently.
            phase_started = time.monotonic()
            finish_trace()
            phase_seconds["trace_finish"] = time.monotonic() - phase_started
        else:
            phase_seconds["trace_finish"] = 0.0
        phase_seconds["prepare"] = prepare_seconds
        phase_seconds["physics"] = physics_seconds
        phase_seconds["capture"] = capture_seconds
        phase_seconds["post_substep"] = post_substep_seconds
        phase_started = time.monotonic()
        self._refresh_object_poses()
        phase_seconds["refresh"] = time.monotonic() - phase_started

        phase_started = time.monotonic()
        success = self._compute_success()
        failed = self._compute_failure()
        reward = 1.0 if success else 0.0
        self.env_step += 1
        terminated = success or failed or (self.env_step * self._control_dt >= self._time_limit)
        manual_reset = self._consume_manual_reset_request()

        info = {
            "succeed": success,
            "manual_reset": manual_reset,
            "isaac_warnings": tuple(self._isaac_warnings),
        }
        if failed:
            info["failed"] = True
        info.update(self._task_info())
        phase_seconds["task"] = time.monotonic() - phase_started

        observation_started = time.monotonic()
        observation = self._compute_observation()
        phase_seconds["observation"] = time.monotonic() - observation_started
        attach_started = time.monotonic()
        observation = self._attach_physx_ground_truth(
            observation,
            reuse_latest_substep=True,
        )
        phase_seconds["attach"] = time.monotonic() - attach_started
        render_seconds = time.monotonic() - observation_started
        work_seconds = time.monotonic() - start_time
        if self.hz > 0:
            control_period = 1.0 / self.hz
            if self._is_b320_collection_task():
                # B320 live collection renders the operator viewport at a
                # lower cadence than tracker sampling, IK and PhysX. Pace to
                # an absolute deadline so a slightly over-budget render frame
                # can use the following non-render frame's spare time instead
                # of permanently lowering the tracker-to-arm update rate.
                now = time.monotonic()
                deadline = getattr(self, "_control_pacing_deadline", None)
                if (
                    deadline is None
                    or start_time - deadline > 2.0 * control_period
                    or deadline - start_time > 2.0 * control_period
                ):
                    deadline = start_time + control_period
                time.sleep(max(0.0, deadline - now))
                self._control_pacing_deadline = deadline + control_period
            else:
                time.sleep(max(0.0, control_period - work_seconds))
        wall_seconds = max(time.monotonic() - start_time, 1e-9)
        control_timing = {
            "target_hz": int(self.hz),
            "actual_hz": float(1.0 / wall_seconds),
            "work_ms": float(work_seconds * 1000.0),
            "render_ms": float(render_seconds * 1000.0),
            "sleep_ms": float(max(0.0, wall_seconds - work_seconds) * 1000.0),
            "sim_realtime_factor": float(self._control_dt / wall_seconds),
        }
        control_timing["phase_ms"] = {
            name: float(seconds * 1000.0)
            for name, seconds in phase_seconds.items()
        }
        info["control_timing"] = control_timing
        self._log_control_performance(control_timing)
        return (
            observation,
            reward,
            terminated,
            False,
            info,
        )

    def _prepare_physics_substep_joint_targets(
        self,
        substep_index: int,
        substep_count: int,
    ) -> str:
        """Install the one target path that owns this physics substep.

        Fresh physical replay supplies exact recorded targets through the
        existing pre-substep callback.  That callback must bypass the live
        slew path; filtering an already recorded target would change the
        solver input.  Without such a callback, advance the live B03 hand
        destination immediately before ``write_data_to_sim`` runs.
        """

        pre_substep = getattr(
            self,
            "_handson_pre_physics_substep_callback",
            None,
        )
        if callable(pre_substep):
            live_goal_serial = self._newton_hand_live_goal_serial
            pre_substep(substep_index, substep_count)
            # Global live-command trajectories call _apply_joint_targets from
            # their callback and therefore still need this substep's slew.
            # Recorded-drive replay writes _joint_targets directly, leaves the
            # live-goal serial unchanged, and must remain byte-for-byte exact.
            if self._newton_hand_live_goal_serial != live_goal_serial:
                if getattr(
                    self,
                    "_smooth_grasp_hand_target_interpolation_enabled",
                    False,
                ):
                    # Global-trajectory callbacks already provide the command
                    # sampled for this exact physics substep.  Install that
                    # goal directly instead of interpolating an interpolated
                    # sample a second time.
                    self._advance_smooth_grasp_hand_target_interpolation(
                        substep_index,
                        substep_count,
                        fraction=1.0,
                    )
                    return "live_callback_exact_interpolated_target"
                self._advance_newton_hand_target_slew()
                return "live_callback_target_slew"
            return "exact_pre_substep_callback"
        if getattr(
            self,
            "_smooth_grasp_hand_target_interpolation_enabled",
            False,
        ):
            self._advance_smooth_grasp_hand_target_interpolation(
                substep_index,
                substep_count,
            )
            return "live_smooth_grasp_interpolation"
        self._advance_newton_hand_target_slew()
        return "live_target_slew"

    def _log_control_performance(self, timing: dict[str, Any]) -> None:
        self._control_perf_steps += 1
        for name, value in timing.get("phase_ms", {}).items():
            self._control_perf_phase_totals[name] = (
                self._control_perf_phase_totals.get(name, 0.0)
                + float(value)
            )
        now = time.monotonic()
        elapsed = now - self._control_perf_window_start
        if elapsed < 2.0:
            return
        actual_hz = self._control_perf_steps / max(elapsed, 1e-9)
        realtime_factor = self._control_perf_steps * self._control_dt / max(elapsed, 1e-9)
        phase_average = {
            name: total / max(self._control_perf_steps, 1)
            for name, total in self._control_perf_phase_totals.items()
        }
        phase_text = " ".join(
            f"{name}={phase_average.get(name, 0.0):.1f}ms"
            for name in (
                "prepare",
                "physics",
                "capture",
                "trace_finish",
                "refresh",
                "task",
                "observation",
                "attach",
            )
        )
        print(
            "[HandsOn/IsaacLab] control timing: "
            f"target={self.hz}Hz actual={actual_hz:.1f}Hz "
            f"rtf={realtime_factor:.2f} "
            f"work={float(timing['work_ms']):.1f}ms "
            f"render={float(timing['render_ms']):.1f}ms "
            f"phases[{phase_text}]",
            flush=True,
        )
        self._control_perf_window_start = now
        self._control_perf_steps = 0
        self._control_perf_phase_totals = {}


    def _task_info(self) -> dict[str, Any]:
        return dict(self._last_assembly_success_metrics)


    def _pump_startup_ui(self) -> bool:
        """Service the human viewport without advancing an extra PhysX frame."""
        if self._headless or self._sim is None:
            return False
        # SimulationContext.render() temporarily disables playSimulations while
        # it services Kit and the viewport.  That keeps the desktop window
        # responsive during thousands of unrecorded settling substeps without
        # changing the physics-step history that replay reconstructs.
        self._sim.render()
        return True

    def _update_recording_cameras(self) -> None:
        if self._scene is None or self._torch is None:
            return
        try:
            root_pos = _torch_view(self._scene["peg"].data.root_pos_w)
            dtype = root_pos.dtype
            device = root_pos.device
            head_pose = _pose_vector_to_matrix(
                np.asarray(
                    [
                        *HEAD_RECORDING_CAMERA_EYE,
                        *HEAD_RECORDING_CAMERA_QUAT_WXYZ_OPENGL,
                    ],
                    dtype=np.float64,
                )
            )
            head_position = self._torch.tensor(
                [[*HEAD_RECORDING_CAMERA_EYE]],
                dtype=dtype,
                device=device,
            )
            head_orientation = self._torch.tensor(
                _quat_wxyz_to_isaaclab_array(
                    HEAD_RECORDING_CAMERA_QUAT_WXYZ_OPENGL,
                    uses_xyzw=self._isaaclab_uses_xyzw,
                ).reshape(1, 4),
                dtype=dtype,
                device=device,
            )
            if self._recording_camera_pose_changed("head_camera", head_pose):
                self._scene["head_camera"].set_world_poses(
                    head_position,
                    head_orientation,
                    convention="opengl",
                )

            for side, sensor_name in (
                ("left", "wrist_left_camera"),
                ("right", "wrist_right_camera"),
            ):
                camera_pose = self._recording_wrist_camera_pose_matrix(side)
                position = self._torch.tensor(
                    [[*camera_pose[:3, 3]]],
                    dtype=dtype,
                    device=device,
                )
                orientation = self._torch.tensor(
                    _quat_wxyz_to_isaaclab_array(
                        _matrix_to_quat_wxyz(camera_pose[:3, :3]),
                        uses_xyzw=self._isaaclab_uses_xyzw,
                    ).reshape(1, 4),
                    dtype=dtype,
                    device=device,
                )
                if self._recording_camera_pose_changed(sensor_name, camera_pose):
                    self._scene[sensor_name].set_world_poses(
                        position,
                        orientation,
                        convention="opengl",
                    )
        except Exception as exc:
            message = f"Recording camera pose update failed: {exc}"
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)
                print(f"[HandsOn/IsaacLab] {message}", flush=True)

    def _recording_wrist_camera_pose_matrix(self, side: str) -> np.ndarray:
        robot = self._control_robot(side)
        mount_name = f"{side}_wrist_fisheye_mount"
        for index, body_name in enumerate(robot.body_names):
            if mount_name in body_name:
                return (
                    self._robot_body_pose_matrix(robot, index)
                    @ _pose_vector_to_matrix(
                        WRIST_RECORDING_CAMERA_MOUNT_POSES[side]
                    )
                )
        fallback = self._left_tcp_pose if side == "left" else self._right_tcp_pose
        palm_pose = self._body_pose_matrix(
            f"{side}_robot",
            f"{side}_palm_link",
            fallback,
        )
        return (
            palm_pose
            @ _pose_vector_to_matrix(WRIST_RECORDING_CAMERA_LOCAL_POSES[side])
        )

    def _is_b320_collection_task(self) -> bool:
        """Return whether this runtime belongs to either B320 naming family."""

        task_name = str(getattr(self, "task_name", "")).strip().lower()
        return task_name.startswith(("b320_", "bench320_"))

    def _set_wrist_obstructions_hidden(self, hidden: bool) -> None:
        if self._stage is None:
            return
        hidden = bool(hidden)
        if self._wrist_obstructions_hidden is hidden:
            return
        if self._wrist_obstruction_visibility is None:
            self._wrist_obstruction_visibility = (
                wrist_obstruction_visibility_records(self._stage)
            )
            print(
                "[HandsOn/IsaacLab] Wrist-only camera obstruction filter: "
                f"{len(self._wrist_obstruction_visibility)} visual prims",
                flush=True,
            )
        set_wrist_obstructions_hidden(
            self._stage,
            self._wrist_obstruction_visibility,
            hidden,
        )
        self._wrist_obstructions_hidden = hidden

    def _apply_action(self, action) -> None:
        if action is None:
            return
        if isinstance(action, dict):
            right = np.asarray(action.get("right", []), dtype=np.float64)
            left = np.asarray(action.get("left", []), dtype=np.float64)
        else:
            flat = np.asarray(action, dtype=np.float64).reshape(-1)
            if flat.size >= 14 + 2 * self._hand_action_dim:
                right = np.concatenate([flat[:7], flat[14 : 14 + self._hand_action_dim]])
                left = np.concatenate(
                    [
                        flat[7:14],
                        flat[14 + self._hand_action_dim : 14 + 2 * self._hand_action_dim],
                    ]
                )
            elif flat.size >= 7 + self._hand_action_dim:
                right = flat
                left = np.asarray([], dtype=np.float64)
            else:
                return

        if right.size >= 7 and not np.allclose(right[:7], 0.0):
            self._right_tcp_pose = right[:7].astype(np.float64, copy=True)
            self._right_pose_commanded = True
        if left.size >= 7 and not np.allclose(left[:7], 0.0):
            self._left_tcp_pose = left[:7].astype(np.float64, copy=True)
            self._left_pose_commanded = True
        if right.size >= 7 + self._hand_action_dim:
            self._right_hand = right[7 : 7 + self._hand_action_dim].astype(np.float64, copy=True)
        if left.size >= 7 + self._hand_action_dim:
            self._left_hand = left[7 : 7 + self._hand_action_dim].astype(np.float64, copy=True)
        postprocess = getattr(self, "_postprocess_hand_action", None)
        if callable(postprocess):
            postprocess()
        operator_controls = getattr(
            self,
            "_apply_operator_action_extras",
            None,
        )
        if callable(operator_controls):
            operator_controls(action)

    def _apply_joint_targets(self) -> None:
        # Generic ABC/D collection environments own their imported robot
        # articulations in ``_robots`` instead of an ``InteractiveScene``.
        # Treat either representation as a live control scene.  Requiring
        # only ``_scene`` made every native ABC/D reset succeed and then made
        # the first real action fail with "control scene is not initialized".
        has_control_robots = bool(getattr(self, "_robots", None))
        if self._scene is None and not has_control_robots:
            if self._launch_isaac:
                raise RuntimeError("IsaacLab control scene is not initialized")
            # Interface-only mode intentionally has no IsaacLab scene.  Keep
            # the Gym transition functional so callers can validate action,
            # observation, termination, and wrapper contracts without an
            # Isaac Sim installation.
            return

        status: dict[str, str] = {}
        slew_live_hand_targets = bool(
            self._newton_hand_target_slew_enabled
            or getattr(
                self,
                "_smooth_grasp_hand_target_interpolation_enabled",
                False,
            )
        )
        for side, hand in (
            ("right", self._right_hand),
            ("left", self._left_hand),
        ):
            robot = self._control_robot(side)
            target = self._joint_targets.get(side)
            if target is None:
                target = _torch_view(robot.data.default_joint_pos).clone()
            else:
                target = target.clone()

            pose = self._right_tcp_pose if side == "right" else self._left_tcp_pose
            pose_commanded = self._right_pose_commanded if side == "right" else self._left_pose_commanded
            arm_target, arm_status = self._compute_arm_joint_target(side, robot, pose, pose_commanded)
            status[f"{side}_arm"] = arm_status
            if arm_target is not None:
                if self._physics_backend == "newton":
                    arm_target = self._limit_newton_arm_joint_target(
                        side,
                        robot,
                        arm_target,
                    )
                elif (
                    self._arm_ik_backend == "haorui"
                    or self._physx_contact_arm_target_limits_enabled()
                    or self._use_position_only_multistart_ik(side)
                    or _bool_env(
                        "HANDSON_ISAACLAB_FORCE_ARM_LIMITS", False
                    )
                ):
                    arm_target = self._limit_arm_joint_target(side, robot, arm_target)
                target[:, self._arm_joint_ids[side]] = arm_target

            joint_ids = list(self._hand_joint_ids.get(side, ()))
            hand_goal = None
            if slew_live_hand_targets and joint_ids:
                # Do not overwrite the previous applied target at the 30 Hz
                # control boundary.  Store the raw-command destination in a
                # separate tensor; each physics substep below either samples
                # the hand-qualified PhysX smooth-grasp interpolation or
                # advances the Newton/D03 fixed-rate slew.
                hand_goal = _clone_array_like(target[:, joint_ids])
                if getattr(
                    self,
                    "_smooth_grasp_hand_target_interpolation_enabled",
                    False,
                ):
                    if not hasattr(self, "_smooth_grasp_hand_start_targets"):
                        self._smooth_grasp_hand_start_targets = {}
                    self._smooth_grasp_hand_start_targets[side] = (
                        _clone_array_like(target[:, joint_ids])
                    )
                for hand_index in range(min(len(joint_ids), hand.size)):
                    hand_goal[:, hand_index] = float(hand[hand_index])
            else:
                for hand_index, joint_id in enumerate(
                    joint_ids[: hand.size]
                ):
                    target[:, joint_id] = float(hand[hand_index])
            actual_hand_max = 0.0
            if joint_ids:
                if (
                    getattr(self, "task_name", None) == "t09_insert_peg"
                    and self._physics_backend == "newton"
                ):
                    # This is display-only telemetry.  Reuse the prior frame's
                    # owned ring endpoint; two per-frame torch.item() calls
                    # otherwise serialize the 30 Hz command path with CUDA.
                    trace = getattr(self, "_physx_ground_truth_trace", None)
                    latest = getattr(trace, "latest_robot_joint_state", None)
                    cached = latest(side) if callable(latest) else None
                    if cached is not None:
                        values = np.asarray(cached["position"]).reshape(-1)
                        valid_ids = [
                            int(index)
                            for index in joint_ids
                            if 0 <= int(index) < values.size
                        ]
                        if valid_ids:
                            actual_hand_max = float(
                                np.max(np.abs(values[valid_ids]))
                            )
                else:
                    try:
                        actual_hand_max = float(
                            self._torch.max(
                                self._torch.abs(
                                    _torch_view(robot.data.joint_pos)[
                                        :, joint_ids
                                    ]
                                )
                            ).item()
                        )
                    except Exception:
                        actual_hand_max = 0.0
            status[f"{side}_hand"] = (
                f"target={float(np.max(np.abs(hand))) if hand.size else 0.0:.3f}"
                f",actual={actual_hand_max:.3f},joints={len(joint_ids)}"
            )

            limits = getattr(robot.data, "soft_joint_pos_limits", None)
            if limits is not None:
                limits = _torch_view(limits)
                target = self._torch.maximum(target, limits[:, :, 0])
                target = self._torch.minimum(target, limits[:, :, 1])
                if hand_goal is not None:
                    hand_limits = limits[:, joint_ids, :]
                    hand_goal = self._torch.maximum(
                        hand_goal,
                        hand_limits[:, :, 0],
                    )
                    hand_goal = self._torch.minimum(
                        hand_goal,
                        hand_limits[:, :, 1],
                    )

            if hand_goal is not None:
                self._newton_hand_goal_targets[side] = hand_goal

            robot.set_joint_position_target(target)
            self._joint_targets[side] = target

        if slew_live_hand_targets:
            self._newton_hand_live_goal_serial += 1

        now = time.monotonic()
        if now - self._last_control_log_time >= 2.0:
            self._last_control_log_time = now
            for side in ("right", "left"):
                diagnostic = self._arm_control_diagnostic(side)
                if diagnostic:
                    status[f"{side}_arm"] = (
                        f"{status.get(f'{side}_arm')}[{diagnostic}]"
                    )
            print(
                "[HandsOn/IsaacLab] teleop targets: "
                f"right_arm={status.get('right_arm')} left_arm={status.get('left_arm')} "
                f"right_hand={status.get('right_hand')} left_hand={status.get('left_hand')}",
                flush=True,
            )

    def _advance_smooth_grasp_hand_target_interpolation(
        self,
        substep_index: int,
        substep_count: int,
        *,
        fraction: float | None = None,
    ) -> int:
        """Linearly sample a live glove goal at one PhysX substep.

        The start is the drive target that was actually applied at the end of
        the preceding control frame.  The last physics substep reaches the new
        glove goal exactly.  Recorded-drive replay bypasses this method in
        ``_prepare_physics_substep_joint_targets``.
        """

        if (
            not getattr(
                self,
                "_smooth_grasp_hand_target_interpolation_enabled",
                False,
            )
            or (
                self._scene is None
                and not bool(getattr(self, "_robots", None))
            )
        ):
            return 0
        if fraction is None:
            count = max(1, int(substep_count))
            alpha = float(
                np.clip((int(substep_index) + 1) / count, 0.0, 1.0)
            )
        else:
            alpha = float(np.clip(fraction, 0.0, 1.0))

        applied_joint_count = 0
        for side in ("right", "left"):
            joint_ids = list(self._hand_joint_ids.get(side, ()))
            goal = self._newton_hand_goal_targets.get(side)
            start = self._smooth_grasp_hand_start_targets.get(side)
            if not joint_ids or goal is None or start is None:
                continue

            robot = self._control_robot(side)
            target = self._joint_targets.get(side)
            if target is None:
                target = _torch_view(robot.data.default_joint_pos)
            target = _clone_array_like(_torch_view(target))
            goal = _torch_view(goal)
            start = _torch_view(start)
            current_hand_target = target[:, joint_ids]
            if (
                tuple(goal.shape) != tuple(current_hand_target.shape)
                or tuple(start.shape) != tuple(current_hand_target.shape)
            ):
                raise RuntimeError(
                    "Smooth-grasp hand target layout changed: "
                    f"side={side}, start={tuple(start.shape)}, "
                    f"goal={tuple(goal.shape)}, "
                    f"applied={tuple(current_hand_target.shape)}"
                )
            target[:, joint_ids] = start + alpha * (goal - start)

            limits = getattr(robot.data, "soft_joint_pos_limits", None)
            if limits is not None:
                limits = _torch_view(limits)
                target = self._torch.maximum(target, limits[:, :, 0])
                target = self._torch.minimum(target, limits[:, :, 1])

            robot.set_joint_position_target(target)
            self._joint_targets[side] = target
            applied_joint_count += len(joint_ids)
        return applied_joint_count

    def _advance_newton_hand_target_slew(self) -> int:
        """Advance contact-sensitive hand drives by one bounded physics step."""

        if (
            not self._newton_hand_target_slew_enabled
            or (
                self._scene is None
                and not bool(getattr(self, "_robots", None))
            )
        ):
            return 0
        maximum_delta = float(
            self._newton_hand_target_slew_rate_rad_s * self._physics_dt
        )
        if maximum_delta <= 0.0:
            return 0

        applied_joint_count = 0
        for side in ("right", "left"):
            joint_ids = list(self._hand_joint_ids.get(side, ()))
            goal = self._newton_hand_goal_targets.get(side)
            if not joint_ids or goal is None:
                continue

            robot = self._control_robot(side)
            target = self._joint_targets.get(side)
            if target is None:
                target = _torch_view(robot.data.default_joint_pos)
            target = _clone_array_like(_torch_view(target))
            current_hand_target = target[:, joint_ids]
            goal = _torch_view(goal)
            if tuple(goal.shape) != tuple(current_hand_target.shape):
                raise RuntimeError(
                    "Hand target slew layout changed: "
                    f"side={side}, goal={tuple(goal.shape)}, "
                    f"applied={tuple(current_hand_target.shape)}"
                )
            target[:, joint_ids] = current_hand_target + self._torch.clamp(
                goal - current_hand_target,
                -maximum_delta,
                maximum_delta,
            )

            limits = getattr(robot.data, "soft_joint_pos_limits", None)
            if limits is not None:
                limits = _torch_view(limits)
                target = self._torch.maximum(target, limits[:, :, 0])
                target = self._torch.minimum(target, limits[:, :, 1])

            # _step_isaaclab writes this actuator buffer to Newton immediately
            # after this method returns.  Ground-truth capture then copies the
            # robot's CUDA joint_pos_target buffer written from this exact
            # _joint_targets value, so the record is the actual per-substep
            # solver input rather than the raw 30 Hz goal.
            robot.set_joint_position_target(target)
            self._joint_targets[side] = target
            applied_joint_count += len(joint_ids)
        return applied_joint_count

    def _arm_control_diagnostic(self, side: str) -> str:
        """Summarize the physical arm servo without changing its state."""

        is_b320_physx = (
            self._physics_backend == "physx"
            and self._is_b320_collection_task()
        )
        if self._physics_backend != "newton" and not is_b320_physx:
            return ""
        if getattr(self, "task_name", None) == "t09_insert_peg":
            trace = getattr(self, "_physx_ground_truth_trace", None)
            latest = getattr(trace, "latest_robot_joint_state", None)
            cached = latest(side) if callable(latest) else None
            if cached is None:
                return "batched_trace_pending"
            try:
                robot = self._control_robot(side)
                arm_joint_ids = np.asarray(
                    self._arm_joint_ids[side],
                    dtype=np.int64,
                )
                actual = np.asarray(cached["position"], dtype=np.float64)[
                    arm_joint_ids
                ]
                target_arm = np.asarray(
                    cached["target"],
                    dtype=np.float64,
                )[arm_joint_ids]
                velocity = np.asarray(
                    cached["velocity"],
                    dtype=np.float64,
                )[arm_joint_ids]
                error = np.abs(target_arm - actual)
                peak_local = int(np.argmax(error))
                peak_joint_id = int(arm_joint_ids[peak_local])
                peak_name = str(robot.joint_names[peak_joint_id])
                pose = (
                    self._right_tcp_pose
                    if side == "right"
                    else self._left_tcp_pose
                )
                tcp = np.asarray(
                    cached["tcp_state"],
                    dtype=np.float64,
                )
                ee_error = float(
                    np.linalg.norm(tcp[:3] - np.asarray(pose[:3]))
                )
                return (
                    f"q_err={float(np.max(error)):.3f},"
                    f"q_vel={float(np.max(np.abs(velocity))):.3f},"
                    f"peak={peak_name}:{actual[peak_local]:.3f}->"
                    f"{target_arm[peak_local]:.3f}@"
                    f"{velocity[peak_local]:.3f},"
                    f"ee_err={ee_error:.3f}m,source=batched_prev_frame"
                )
            except Exception:
                return "batched_trace_invalid"
        try:
            robot = self._control_robot(side)
            arm_joint_ids = self._arm_joint_ids[side]
            actual = _torch_view(robot.data.joint_pos)[:, arm_joint_ids]
            default = _torch_view(robot.data.default_joint_pos)[:, arm_joint_ids]
            target = self._joint_targets.get(side)
            target_arm = (
                actual
                if target is None
                else _torch_view(target)[:, arm_joint_ids]
            )
            velocity = _torch_view(robot.data.joint_vel)[:, arm_joint_ids]

            actual_home = float(
                self._torch.max(self._torch.abs(actual - default)).item()
            )
            target_home = float(
                self._torch.max(self._torch.abs(target_arm - default)).item()
            )
            drive_error = float(
                self._torch.max(self._torch.abs(target_arm - actual)).item()
            )
            speed = float(self._torch.max(self._torch.abs(velocity)).item())
            peak_local = int(
                self._torch.argmax(self._torch.abs(actual - default)[0]).item()
            )
            peak_joint_id = int(arm_joint_ids[peak_local])
            peak_name = str(robot.joint_names[peak_joint_id])
            peak_actual = float(actual[0, peak_local].item())
            peak_target = float(target_arm[0, peak_local].item())
            peak_velocity = float(velocity[0, peak_local].item())
            sim_target_delta = float("nan")
            try:
                sim_target = _torch_view(robot.data.joint_pos_target)[
                    :, arm_joint_ids
                ]
                sim_target_delta = float(
                    self._torch.max(
                        self._torch.abs(sim_target - target_arm)
                    ).item()
                )
            except Exception:
                pass

            pose = self._right_tcp_pose if side == "right" else self._left_tcp_pose
            actual_ee = self._robot_body_pose_matrix(
                robot,
                self._ee_body_indices[side],
            )
            ee_error = float(
                np.linalg.norm(actual_ee[:3, 3] - np.asarray(pose[:3]))
            )
            ee_rot_error = _quat_delta_deg(
                _matrix_to_quat_wxyz(actual_ee[:3, :3]),
                _normalize_quat_wxyz(np.asarray(pose[3:7], dtype=np.float64)),
            )
            return (
                f"q_home={actual_home:.3f},target_home={target_home:.3f},"
                f"q_err={drive_error:.3f},q_vel={speed:.3f},"
                f"peak={peak_name}:{peak_actual:.3f}->{peak_target:.3f}"
                f"@{peak_velocity:.3f},sim_target_delta={sim_target_delta:.3g},"
                f"ee_err={ee_error:.3f}m,ee_rot_err={ee_rot_error:.1f}deg"
            )
        except Exception as exc:
            message = f"{side} Newton arm diagnostic failed: {exc}"
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)
                print(f"[HandsOn/IsaacLab] {message}", flush=True)
            return "diagnostic_error"

    def _compute_arm_joint_target(self, side: str, robot, pose: np.ndarray, commanded: bool):
        if not commanded:
            return None, "idle"
        if (
            self._arm_ik_backend == "haorui"
            or self._physx_contact_arm_target_limits_enabled()
        ):
            pose = self._limit_arm_pose_target(side, robot, pose)
        if (
            self._haorui_ik_enabled
            and self._use_position_only_multistart_ik(side)
        ):
            arm_target = self._compute_arm_joint_target_haorui(
                side, robot, pose, position_only=True
            )
            if arm_target is not None:
                return arm_target, "haorui_position_multistart"
        if self._arm_ik_backend == "haorui" and self._haorui_ik_enabled:
            arm_target = self._compute_arm_joint_target_haorui(side, robot, pose)
            if arm_target is not None:
                return arm_target, "haorui"
            if not _bool_env("HANDSON_ISAACLAB_ALLOW_DLS_FALLBACK", False):
                return None, "haorui_hold"
        try:
            controller = self._ik_controllers[side]
            arm_joint_ids = self._arm_joint_ids[side]
            ee_body_idx = self._ee_body_indices[side]
            ee_jacobi_idx = self._ee_jacobi_indices[side]

            jacobian = _end_effector_link_jacobian(
                robot,
                ee_jacobi_idx,
                arm_joint_ids,
                lab3=self._isaaclab_uses_xyzw,
            )
            ee_pose_w = _torch_view(robot.data.body_pose_w)[:, ee_body_idx]
            root_pose_w = _torch_view(robot.data.root_pose_w)
            base_rot = root_pose_w[:, 3:7]
            base_rot_matrix = self._math_utils["matrix_from_quat"](self._math_utils["quat_inv"](base_rot))
            jacobian[:, :3, :] = self._torch.bmm(base_rot_matrix, jacobian[:, :3, :])
            jacobian[:, 3:, :] = self._torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

            ee_pos_b, ee_quat_b = self._math_utils["subtract_frame_transforms"](
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3],
                ee_pose_w[:, 3:7],
            )
            target_pose_w = self._torch.tensor(
                _pose_wxyz_to_isaaclab_array(
                    np.asarray(pose, dtype=np.float32).reshape(1, 7),
                    uses_xyzw=self._isaaclab_uses_xyzw,
                ),
                dtype=ee_pose_w.dtype,
                device=ee_pose_w.device,
            )
            target_pos_b, target_quat_b = self._math_utils["subtract_frame_transforms"](
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                target_pose_w[:, 0:3],
                target_pose_w[:, 3:7],
            )
            command_type = self._ik_command_types.get(side, "pose")
            position_only = (
                command_type == "position"
                or self._use_position_only_differential_ik(side)
            )
            if command_type == "position":
                controller.set_command(target_pos_b, ee_quat=ee_quat_b)
            else:
                controller.set_command(
                    self._torch.cat((target_pos_b, target_quat_b), dim=1)
                )
            joint_pos = _torch_view(robot.data.joint_pos)[:, arm_joint_ids]
            status = "haorui_fallback_dls" if self._arm_ik_backend == "haorui" else "isaaclab_dls"
            if position_only and command_type != "position":
                position_error = target_pos_b - ee_pos_b
                jacobian_pos = jacobian[:, :3, :]
                jacobian_transpose = self._torch.transpose(
                    jacobian_pos, dim0=1, dim1=2
                )
                damping = (0.04**2) * self._torch.eye(
                    n=jacobian_pos.shape[1],
                    dtype=jacobian_pos.dtype,
                    device=jacobian_pos.device,
                )
                damped_inverse = self._torch.inverse(
                    jacobian_pos @ jacobian_transpose + damping
                )
                jacobian_pinv = jacobian_transpose @ damped_inverse
                delta_joint_pos = (
                    jacobian_pinv @ position_error.unsqueeze(-1)
                ).squeeze(-1)
                nullspace_gain = float(
                    self._position_only_differential_ik_nullspace_gain(side)
                )
                if nullspace_gain > 0.0:
                    joint_count = jacobian_pos.shape[2]
                    projector = self._torch.eye(
                        n=joint_count,
                        dtype=jacobian_pos.dtype,
                        device=jacobian_pos.device,
                    ) - jacobian_pinv @ jacobian_pos
                    home_joint_pos = _torch_view(robot.data.default_joint_pos)[
                        :, arm_joint_ids
                    ]
                    home_error = home_joint_pos - joint_pos
                    delta_joint_pos = delta_joint_pos + nullspace_gain * (
                        projector @ home_error.unsqueeze(-1)
                    ).squeeze(-1)
                return joint_pos + delta_joint_pos, f"{status}_position_only"
            return controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos), status
        except Exception as exc:
            message = f"{side} arm IK failed: {exc}"
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)
                print(f"[HandsOn/IsaacLab] {message}", flush=True)
            return None, "ik_error"

    def _d03_physx_arm_target_limits_enabled(self) -> bool:
        """Reject tracker discontinuities before D03's DLS controller."""

        return bool(
            self._physics_backend == "physx"
            and getattr(self, "task_name", None)
            == "t21_pass_block"
        )

    def _physx_contact_arm_target_limits_enabled(self) -> bool:
        """Keep the task-local PhysX target limiter on D03 only."""

        return self._d03_physx_arm_target_limits_enabled()

    def _limit_arm_pose_target(self, side: str, robot, pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64).reshape(7)
        current_matrix = self._robot_body_pose_matrix(robot, self._ee_body_indices[side])
        current_pos = current_matrix[:3, 3].copy()
        current_quat = _matrix_to_quat_wxyz(current_matrix[:3, :3])

        target_pos = pose[:3].copy()
        target_quat = _normalize_quat_wxyz(pose[3:7])
        if not np.all(np.isfinite(target_pos)):
            target_pos = current_pos.copy()
        if not np.all(np.isfinite(target_quat)):
            target_quat = current_quat.copy()

        max_radius = _float_env("HANDSON_ISAACLAB_ARM_WORKSPACE_MAX_RADIUS", _ARM_WORKSPACE_MAX_RADIUS)
        if max_radius > 0.0:
            base_pos = self._robot_base_pose_matrix(side, robot)[:3, 3]
            target_pos = base_pos + _clamp_norm(target_pos - base_pos, max_radius)

        previous = self._arm_pose_filters.get(side)
        if previous is None:
            previous = {
                "raw_pos": current_pos.copy(),
                "raw_quat": current_quat.copy(),
                "pos": current_pos.copy(),
                "quat": current_quat.copy(),
            }

        jump_delta = _float_env("HANDSON_ISAACLAB_ARM_TARGET_JUMP_DELTA", _ARM_TARGET_JUMP_DELTA)
        if jump_delta > 0.0:
            target_pos = previous["raw_pos"] + _clamp_norm(target_pos - previous["raw_pos"], jump_delta)

        if float(np.linalg.norm(target_pos - previous["raw_pos"])) < _float_env(
            "HANDSON_ISAACLAB_ARM_TARGET_POS_DEADBAND",
            _ARM_TARGET_POS_DEADBAND,
        ):
            target_pos = previous["raw_pos"].copy()

        rot_delta = _quat_delta_deg(previous["raw_quat"], target_quat)
        rot_jump = _float_env("HANDSON_ISAACLAB_ARM_TARGET_ROT_JUMP_DEG", _ARM_TARGET_ROT_JUMP_DEG)
        if rot_jump > 0.0 and rot_delta > rot_jump:
            target_quat = _blend_quat_wxyz(previous["raw_quat"], target_quat, rot_jump / rot_delta)
        elif rot_delta < _float_env("HANDSON_ISAACLAB_ARM_TARGET_ROT_DEADBAND_DEG", _ARM_TARGET_ROT_DEADBAND_DEG):
            target_quat = previous["raw_quat"].copy()

        alpha = float(np.clip(_float_env("HANDSON_ISAACLAB_ARM_TARGET_ALPHA", _ARM_TARGET_ALPHA), 0.0, 1.0))
        filtered_pos = (1.0 - alpha) * previous["pos"] + alpha * target_pos
        filtered_quat = _blend_quat_wxyz(previous["quat"], target_quat, alpha)

        max_step = _float_env(
            "HANDSON_ISAACLAB_ARM_MAX_SITE_STEP",
            _ARM_TARGET_MAX_SITE_STEP,
        )
        if max_step > 0.0:
            filtered_pos = current_pos + _clamp_norm(filtered_pos - current_pos, max_step)

        self._arm_pose_filters[side] = {
            "raw_pos": target_pos.copy(),
            "raw_quat": target_quat.copy(),
            "pos": filtered_pos.copy(),
            "quat": filtered_quat.copy(),
        }
        return np.concatenate([filtered_pos, filtered_quat]).astype(np.float64, copy=False)

    def _limit_arm_joint_target(self, side: str, robot, arm_target):
        arm_joint_ids = self._arm_joint_ids[side]
        current = self._joint_targets.get(side)
        if current is None:
            current = _torch_view(robot.data.joint_pos)
        current_q = current[:, arm_joint_ids]
        limited = arm_target

        jump_limit = _float_env("HANDSON_ISAACLAB_ARM_TARGET_JUMP_LIMIT", _ARM_TARGET_JUMP_LIMIT)
        if jump_limit > 0.0:
            jump_delta = self._torch.clamp(limited - current_q, -jump_limit, jump_limit)
            limited = current_q + jump_delta

        max_step = _float_env("HANDSON_ISAACLAB_ARM_MAX_JOINT_STEP", _ARM_TARGET_MAX_JOINT_STEP)
        if max_step > 0.0:
            step_delta = self._torch.clamp(limited - current_q, -max_step, max_step)
            limited = current_q + step_delta
        return limited

    def _limit_newton_arm_joint_target(self, side: str, robot, arm_target):
        """Bound one exact Newton servo update without inheriting PhysX knobs."""

        arm_joint_ids = self._arm_joint_ids[side]
        current = self._joint_targets.get(side)
        if current is None:
            current = _torch_view(robot.data.joint_pos)
        current_arm = current[:, arm_joint_ids]
        max_step = _float_env(
            "HANDSON_NEWTON_ARM_MAX_JOINT_STEP",
            _NEWTON_ARM_TARGET_MAX_JOINT_STEP,
        )
        if max_step <= 0.0:
            return arm_target
        return current_arm + self._torch.clamp(
            arm_target - current_arm,
            -max_step,
            max_step,
        )

    def _compute_arm_joint_target_haorui(
        self,
        side: str,
        robot,
        pose: np.ndarray,
        *,
        position_only: bool = False,
    ):
        try:
            q_reference = self._arm_reference_q(side, robot)
            arm_joint_ids = self._arm_joint_ids.get(side)
            if q_reference is None or arm_joint_ids is None:
                return None
            q_measured = (
                _torch_view(robot.data.joint_pos)[:, arm_joint_ids]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=True)
                .reshape(-1)
            )
            if q_measured.size != 7 or not np.all(np.isfinite(q_measured)):
                return None

            base_from_world = _invert_pose_matrix(self._robot_base_pose_matrix(side, robot))
            current_ee_in_base = base_from_world @ self._robot_body_pose_matrix(
                robot,
                self._ee_body_indices[side],
            )
            target_ee_in_base = base_from_world @ _pose_vector_to_matrix(pose)

            model = self._haorui_ik_models[side]
            solvers = (
                self._haorui_position_ik_solvers
                if position_only
                else self._haorui_ik_solvers
            )
            solver = solvers[side]
            current_tcp_in_base = model.fk(q_measured)
            tcp_to_ee = _invert_pose_matrix(current_tcp_in_base) @ current_ee_in_base
            target_tcp_in_base = target_ee_in_base @ _invert_pose_matrix(tcp_to_ee)
            if not np.all(np.isfinite(target_tcp_in_base)):
                return None

            seeds = self._haorui_rokae.make_seeds(side, q_reference)
            sol = solver.solve_multi_start(
                target_tcp_in_base,
                seeds,
                first_success=True,
                reference_q=q_reference,
            )
            if not sol.success:
                return None

            q = np.asarray(sol.q, dtype=np.float64).reshape(1, -1)
            if q.shape[1] != 7 or not np.all(np.isfinite(q)):
                return None
            joint_pos = _torch_view(robot.data.joint_pos)
            return self._torch.tensor(
                q,
                dtype=joint_pos.dtype,
                device=joint_pos.device,
            )
        except Exception as exc:
            fallback = "isaaclab_dls" if _bool_env("HANDSON_ISAACLAB_ALLOW_DLS_FALLBACK", False) else "hold_previous"
            message = f"{side} Haorui IK failed; fallback={fallback} reason={exc}"
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)
                print(f"[HandsOn/IsaacLab] {message}", flush=True)
            return None

    def _arm_reference_q(self, side: str, robot) -> np.ndarray | None:
        arm_joint_ids = self._arm_joint_ids.get(side)
        if arm_joint_ids is None:
            return None
        target = self._joint_targets.get(side)
        source = target if target is not None else _torch_view(robot.data.joint_pos)
        return (
            source[:, arm_joint_ids]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=True)
            .reshape(-1)
        )

    def _robot_base_pose_matrix(self, side: str, robot) -> np.ndarray:
        for index, name in enumerate(robot.body_names):
            if name == f"base_{side}" or name.endswith(f"/base_{side}"):
                return self._robot_body_pose_matrix(robot, index)
        root_pose = (
            _torch_view(robot.data.root_pose_w)[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        root_pose = _pose_isaaclab_to_wxyz_array(
            root_pose,
            uses_xyzw=self._isaaclab_uses_xyzw,
        )
        return _pose_vector_to_matrix(root_pose)

    def _robot_body_pose_matrix(self, robot, body_index: int) -> np.ndarray:
        pose = _torch_view(robot.data.body_pose_w)[0, int(body_index), :7]
        pose_np = pose.detach().cpu().numpy().astype(np.float64)
        pose_wxyz = _pose_isaaclab_to_wxyz_array(
            pose_np,
            uses_xyzw=self._isaaclab_uses_xyzw,
        )
        return _pose_vector_to_matrix(pose_wxyz)


    def _rigid_object_pose(self, name: str, fallback: np.ndarray) -> np.ndarray:
        try:
            root_state = _torch_view(self._scene[name].data.root_state_w)[0, :7]
            pose = root_state.detach().cpu().numpy().astype(np.float64, copy=True)
            return _pose_isaaclab_to_wxyz_array(
                pose,
                uses_xyzw=self._isaaclab_uses_xyzw,
            )
        except Exception:
            return fallback.astype(np.float64, copy=True)

    def _body_pose_matrix(self, scene_name: str, body_substring: str, fallback_pose: np.ndarray) -> np.ndarray:
        try:
            side = "right" if scene_name.startswith("right") else "left"
            robot = self._control_robot(side)
            body_names = list(robot.body_names)
            body_index = next(index for index, name in enumerate(body_names) if body_substring in name)
            body_state = _torch_view(robot.data.body_state_w)[0, body_index, :7]
            pose = body_state.detach().cpu().numpy().astype(np.float64)
            pose_wxyz = _pose_isaaclab_to_wxyz_array(
                pose,
                uses_xyzw=self._isaaclab_uses_xyzw,
            )
            return _pose_vector_to_matrix(pose_wxyz)
        except Exception:
            return _pose_vector_to_matrix(fallback_pose)

    def _compute_success(self) -> bool:
        geometry = assembly_geometry(self._peg_ori_pose, self._socket_ori_pose)
        peg_velocity = self._rigid_object_velocity("peg")
        socket_velocity = self._rigid_object_velocity("socket")

        def stable(velocity: np.ndarray | None) -> bool:
            return bool(
                velocity is not None
                and velocity.shape == (6,)
                and np.all(np.isfinite(velocity))
                and float(np.linalg.norm(velocity[:3]))
                <= ASSEMBLY_STABLE_LINEAR_SPEED
                and float(np.linalg.norm(velocity[3:]))
                <= ASSEMBLY_STABLE_ANGULAR_SPEED
            )

        peg_stable = stable(peg_velocity)
        socket_stable = stable(socket_velocity)
        inserted = assembly_geometry_is_inserted(geometry)
        success_now = bool(inserted and peg_stable and socket_stable)
        self._last_assembly_success_metrics = {
            "assembly_inserted": bool(inserted),
            "assembly_radial_error": float(geometry["radial_error"]),
            "assembly_insertion_depth": float(geometry["insertion_depth"]),
            "assembly_axis_alignment": float(geometry["axis_alignment"]),
            "assembly_tip_end_correct": bool(geometry["tip_end_correct"]),
            "assembly_handle_outside": bool(geometry["handle_outside"]),
            "assembly_peg_velocity_available": peg_velocity is not None,
            "assembly_socket_velocity_available": socket_velocity is not None,
            "assembly_peg_stable": peg_stable,
            "assembly_socket_stable": socket_stable,
            "assembly_success_now": success_now,
        }
        if success_now:
            self._success_counter += 1
        else:
            self._success_counter = 0
        required_steps = max(
            1,
            int(
                np.ceil(
                    ASSEMBLY_STABLE_HOLD_SECONDS
                    / max(float(self._control_dt), 1e-6)
                )
            ),
        )
        self._last_assembly_success_metrics["assembly_success_counter"] = int(
            self._success_counter
        )
        self._last_assembly_success_metrics["assembly_success_steps_required"] = int(
            required_steps
        )
        return self._success_counter >= required_steps

    def _rigid_object_velocity(self, name: str) -> np.ndarray | None:
        if self._scene is None:
            return None
        try:
            data = self._scene[name].data
            if hasattr(data, "root_vel_w"):
                value = _torch_view(data.root_vel_w)[0, :6]
            else:
                value = _torch_view(data.root_state_w)[0, 7:13]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            velocity = np.asarray(value, dtype=np.float64).reshape(6)
            return velocity if np.all(np.isfinite(velocity)) else None
        except Exception:
            return None


    def _camera_rgb_frame(self, sensor_name: str, fallback_key: str) -> np.ndarray:
        is_wrist = sensor_name.startswith("wrist_")
        if is_wrist:
            target_width, target_height = WRIST_RECORDING_CAMERA_OUTPUT_SIZE
        else:
            target_height, target_width = self._render_height, self._render_width
        if self._scene is not None:
            try:
                camera = self._scene[sensor_name]
                frame = _torch_view(camera.data.output["rgb"])[0]
                if hasattr(frame, "detach"):
                    frame = frame.detach().cpu().numpy()
                frame = np.asarray(frame)
                if frame.ndim == 3 and frame.shape[-1] >= 3:
                    frame = frame[..., :3]
                    if frame.dtype != np.uint8:
                        max_value = float(np.nanmax(frame)) if frame.size else 0.0
                        if max_value <= 1.0:
                            frame = frame * 255.0
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8, copy=False)
                    if frame.shape[:2] == (target_height, target_width):
                        return frame.copy()
                    return self._resize_rgb_nearest(frame, target_height, target_width)
            except Exception as exc:
                message = f"{sensor_name} RGB unavailable; using fallback frame: {exc}"
                if message not in self._isaac_warnings:
                    self._isaac_warnings.append(message)
                    print(f"[HandsOn/IsaacLab] {message}", flush=True)
        return self._fallback_camera_frame(
            fallback_key,
            height=target_height,
            width=target_width,
        )
