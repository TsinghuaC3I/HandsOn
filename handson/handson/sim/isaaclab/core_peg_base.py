from __future__ import annotations

from .bimanual_runtime import BimanualRuntimeSupport

from typing import Any, Literal
import gc
import os
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..envs.hand_specs import hand_action_dim, normalize_arm_type, normalize_hand_type
from ..isaacsim.runtime import IsaacSimUnavailable
from .physics_config import (
    SINGLE_ENV_GPU_MAX_RIGID_CONTACT_COUNT,
    SINGLE_ENV_GPU_MAX_RIGID_PATCH_COUNT,
)
from .physx_ground_truth import (
    PhysxGroundTruthTrace,
    configure_physx_contact_reports_for_recording,
    physx_ground_truth_enabled,
)
from .recording_cameras import (
    HEAD_RECORDING_CAMERA_EYE,
    HEAD_RECORDING_CAMERA_QUAT_WXYZ_OPENGL,
    WRIST_RECORDING_CAMERA_LOCAL_POSES,
    WRIST_RECORDING_CAMERA_MOUNT_POSES,
    WRIST_RECORDING_CAMERA_OUTPUT_SIZE,
    recording_camera_profile,
    set_wrist_obstructions_hidden,
    wrist_obstruction_visibility_records,
)
from .runtime_display import enforce_isaac_viewport_only, isaac_kit_args


_DEFAULT_RENDER_SIZE = (640, 640)
_RANDOM_SEED_MODULUS = 2**32 - 1
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
_ARM_TARGET_JUMP_LIMIT = 0.45
_ARM_TARGET_MAX_SITE_STEP = 0.035
_ARM_TARGET_JUMP_DELTA = 0.10
_ARM_TARGET_POS_DEADBAND = 0.002
_ARM_TARGET_ROT_JUMP_DEG = 20.0
_ARM_TARGET_ROT_DEADBAND_DEG = 1.0
_ARM_TARGET_ALPHA = 0.25
_ARM_WORKSPACE_MAX_RADIUS = 0.78
_HAND_JOINT_NAMES = tuple(
    f"{side}_finger{finger}_joint{joint}"
    for side in ("right", "left")
    for finger in range(1, 6)
    for joint in range(1, 5)
)


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
        self.arm_type = normalize_arm_type(arm_type)
        if self.hand_type != "wuji1" or self.arm_type != "rokae":
            raise ValueError("Isaac Lab bimanual assembly currently supports arm_type=rokae and hand_type=wuji1.")

        self._hand_action_dim = hand_action_dim(self.hand_type)
        self._seed = _time_based_random_seed() if seed is None else int(seed)
        self._random = np.random.RandomState(self._seed)
        self._control_dt = max(
            1e-6,
            _float_env("HANDSON_ISAACLAB_CONTROL_DT", float(control_dt)),
        )
        self._physics_backend = "physx"  # This frozen implementation uses PhysX only.
        self._physics_dt = float(physics_dt)
        self._n_substeps = max(1, int(round(self._control_dt / self._physics_dt)))
        self._time_limit = float(time_limit)
        self._render_height = int(render_resolution[0])
        self._render_width = int(render_resolution[1])
        self._image_shape = (self._render_height, self._render_width, 3)
        self._isaac_device = isaac_device or _str_env("HANDSON_ISAACLAB_DEVICE", "cuda:0")

        if launch_isaac is None:
            launch_isaac = not _bool_env("HANDSON_ISAACSIM_INTERFACE_ONLY")
        self._launch_isaac = bool(launch_isaac)
        self._headless = render_mode != "human" if headless is None else bool(headless)

        self._app_launcher = existing_app_launcher
        self._simulation_app = existing_simulation_app
        self._sim = None
        self._scene = None
        self._stage = None
        self._torch = None
        self._sim_dt = self._physics_dt
        self._joint_targets: dict[str, Any] = {}
        self._hand_joint_ids: dict[str, list[int]] = {"right": [], "left": []}
        self._arm_joint_ids: dict[str, Any] = {}
        self._ee_body_indices: dict[str, int] = {}
        self._ee_jacobi_indices: dict[str, int] = {}
        self._ik_controllers: dict[str, Any] = {}
        self._math_utils: dict[str, Any] = {}
        self._randomize_scene_visuals = None
        self._haorui_rokae = None
        self._haorui_ik_models: dict[str, Any] = {}
        self._haorui_ik_solvers: dict[str, Any] = {}
        self._haorui_ik_enabled = False
        self._arm_ik_backend = self._normalize_arm_ik_backend(
            _str_env("HANDSON_ISAACLAB_ARM_IK_BACKEND", "dls")
        )
        self._arm_pose_filters: dict[str, dict[str, np.ndarray] | None] = {"right": None, "left": None}
        self._isaac_warnings: list[str] = []
        self._wrist_obstruction_visibility: tuple[tuple[str, Any], ...] | None = None
        self._wrist_obstructions_hidden: bool | None = None
        self.recording_camera_profile = recording_camera_profile()
        self._hand_target_control = True
        self._timeline = None
        self._timeline_pause_subscription = None
        self._keyboard_input_interface = None
        self._keyboard = None
        self._keyboard_subscription = None
        self._manual_reset_requested = False
        self._manual_reset_already_applied = False
        self._keyboard_reset_in_progress = False
        self._last_control_log_time = 0.0
        self._control_perf_window_start = time.monotonic()
        self._control_perf_steps = 0
        self._right_pose_commanded = False
        self._left_pose_commanded = False
        self._reset_physics_steps = 0
        self._physx_ground_truth_enabled = physx_ground_truth_enabled()
        self._physx_ground_truth_trace: PhysxGroundTruthTrace | None = None
        self._physx_scene_generation = 0

        self.delta_h = np.zeros(1, dtype=np.float64)
        self._right_tcp_pose = _RIGHT_HOME.copy()
        self._left_tcp_pose = _LEFT_HOME.copy()
        self._right_hand = np.zeros(self._hand_action_dim, dtype=np.float64)
        self._left_hand = np.zeros(self._hand_action_dim, dtype=np.float64)
        self._peg_ori_pose = _PEG_POSE.copy()
        self._socket_ori_pose = _SOCKET_POSE.copy()
        self.env_step = 0
        self._success_counter = 0

        self.action_space = spaces.Box(
            low=np.full(7 + self._hand_action_dim, -1.0, dtype=np.float32),
            high=np.full(7 + self._hand_action_dim, 1.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = self._build_observation_space()

        if self._launch_isaac:
            try:
                self._setup_isaaclab()
            except BaseException:
                self.close()
                raise
        self._initialize_physx_ground_truth_recording()


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

        self._unsubscribe_scene_input_callbacks()
        clear_callbacks = getattr(self._sim, "clear_all_callbacks", None)
        if callable(clear_callbacks):
            clear_callbacks()
        del clear_callbacks

        # Camera, tensor, articulation and contact views must die before the
        # old stage is detached.  Holding one of these views across a stage
        # replacement can leave it pointing at released PhysX memory.
        self.__dict__.clear()
        gc.collect()

        try:
            import torch
            import isaaclab.sim as sim_utils
            from isaaclab.sim import SimulationContext

            if torch.cuda.is_available():
                torch.cuda.synchronize()
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
        except BaseException:
            # Do not let ordinary shutdown accidentally close a still-running
            # app through stale scene handles after a failed reconstruction.
            self._simulation_app = simulation_app
            self._app_launcher = app_launcher
            raise

        self._physx_scene_generation = next_generation
        self.__dict__.update(preserved_attributes)
        return {
            "scene_rebuilt": True,
            "scene_generation": next_generation,
            "seed": int(seed),
            "process_restarted": False,
            "simulation_app_reused": True,
        }


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
        """Install the recording-only PhysX readback contract after scene setup."""

        if not self._physx_ground_truth_enabled:
            return
        self._physx_ground_truth_trace = PhysxGroundTruthTrace(self)
        observation_spaces = {
            name: space
            for name, space in self.observation_space.spaces.items()
            if not name.startswith("physx_")
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


    def get_physical_replay_reset_contract(self) -> dict[str, Any]:
        """Describe the deterministic boundary immediately before action[0]."""

        contract = {
            "strategy": (
                "seeded_environment_reset_then_exact_recorded_physx_t0_restore"
            ),
            "requires_collection_run_seed": True,
            "requires_episode_reset_seed": True,
            "allow_pre_action_state_overwrite": True,
            "allows_single_recorded_t0_restore": True,
            "forbids_state_writes_after_action_start": True,
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
            },
        }
        stable_origin = getattr(self, "_physical_recording_origin", None)
        if isinstance(stable_origin, dict):
            contract["stable_recording_origin"] = dict(stable_origin)
            contract["strategy"] = (
                "seeded_stable_reset_then_exact_recorded_physx_t0_restore"
            )
        return contract


    def _setup_isaaclab(self) -> None:
        import os

        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        try:
            from isaaclab.app import AppLauncher
        except Exception as exc:  # pragma: no cover - exercised in Isaac runtime
            raise IsaacSimUnavailable("Isaac Lab AppLauncher is not available in this Python environment.") from exc

        launcher_args = {
            "headless": self._headless,
            "enable_cameras": self.image_obs,
            "width": self._render_width,
            "height": self._render_height,
            "device": self._isaac_device,
        }
        kit_args = isaac_kit_args(headless=self._headless)
        if kit_args:
            launcher_args["kit_args"] = kit_args
        if self._simulation_app is None:
            from .compat import create_app_launcher
            self._app_launcher = create_app_launcher(AppLauncher, launcher_args)
            self._simulation_app = self._app_launcher.app
        enforce_isaac_viewport_only(self._simulation_app, headless=self._headless)
        self._setup_timeline_pause_guard()
        self._setup_keyboard_reset()

        try:
            from isaacsim.core.utils.extensions import enable_extension

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
            BimanualAssemblySceneCfg,
            apply_scene_visuals,
            fix_robot_bases,
            randomize_scene_visuals,
            validate_asset_files,
        )

        self._torch = torch
        self._randomize_scene_visuals = randomize_scene_visuals
        validate_asset_files()

        sim_cfg = sim_utils.SimulationCfg(
            device=self._isaac_device,
            dt=self._physics_dt,
            render_interval=max(1, int(round(self._control_dt / self._physics_dt))),
            gravity=(0.0, 0.0, -9.81),
            physx=sim_utils.PhysxCfg(
                solver_type=1,
                enable_enhanced_determinism=True,
                max_position_iteration_count=64,
                max_velocity_iteration_count=4,
                bounce_threshold_velocity=0.2,
                friction_offset_threshold=0.01,
                friction_correlation_distance=0.00625,
                gpu_max_rigid_contact_count=SINGLE_ENV_GPU_MAX_RIGID_CONTACT_COUNT,
                gpu_max_rigid_patch_count=SINGLE_ENV_GPU_MAX_RIGID_PATCH_COUNT,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
        )
        self._sim = SimulationContext(sim_cfg)
        self._sim.set_camera_view(eye=(-1.45, -0.05, 1.70), target=(-0.05, 0.0, 1.02))

        print("[HandsOn/IsaacLab] Constructing native collection scene...", flush=True)
        self._scene = InteractiveScene(BimanualAssemblySceneCfg(num_envs=1, env_spacing=2.5))
        import omni.usd

        self._stage = omni.usd.get_context().get_stage()
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
        print("[HandsOn/IsaacLab] Starting PhysX for collection...", flush=True)
        self._sim.reset()
        self._sim_dt = float(self._sim.get_physics_dt())
        self._cache_joint_targets()
        self._init_haorui_rokae_ik()
        self._reset_scene_state()
        self._refresh_object_poses()
        print("[HandsOn/IsaacLab] Native bimanual assembly collection ready", flush=True)


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
                "[HandsOn/IsaacLab] Keyboard controls enabled: R resets; Space is ignored",
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
        is_space_key = key_input == getattr(carb_input.KeyboardInput, "SPACE", None) or key_name in {
            "space",
            "keyboardinput.space",
        }
        if is_press and is_space_key:
            # Returning False consumes the event before Kit's default Space
            # shortcut can pause the Timeline.  If that shortcut ran first,
            # resume here as well so SimulationContext.step() cannot block.
            self._resume_timeline_if_paused(source="Space pause")
            return False
        if is_press and is_r_key:
            self._manual_reset_requested = True
            # The recorder owns the reset transaction: it discards the current
            # trajectory, recreates the spool, re-randomizes the scene, waits
            # for the stable recording origin, then rebases teleoperation.
            # Resetting here as well used to create a second, concurrent scene
            # reset and let Kit's default R shortcut see the same key event.
            print("[HandsOn/IsaacLab] Keyboard reset requested; queued for recorder", flush=True)
            return False
        return True


    def _apply_keyboard_reset_immediately(self) -> None:
        if self._keyboard_reset_in_progress:
            return
        if self._sim is None:
            return
        self._keyboard_reset_in_progress = True
        try:
            self._sample_reset_state(clear_manual_request=False)
            self._reset_scene_state()
            self._apply_reset_visual_randomization()
            self._step_isaaclab()
            self._refresh_object_poses()
            self._manual_reset_already_applied = True
        except Exception as exc:
            message = f"Immediate keyboard reset failed; recorder fallback will reset: {exc}"
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)
                print(f"[HandsOn/IsaacLab] {message}", flush=True)
        finally:
            self._keyboard_reset_in_progress = False


    def _cache_joint_targets(self) -> None:
        self._joint_targets = {}
        self._hand_joint_ids = {"right": [], "left": []}
        self._arm_joint_ids = {}
        self._ee_body_indices = {}
        self._ee_jacobi_indices = {}
        self._ik_controllers = {}
        for side in ("right", "left"):
            robot = self._control_robot(side)
            target = robot.data.default_joint_pos.clone()
            self._joint_targets[side] = target

            arm_joint_names = [f"joint{index}_{side}" for index in range(1, 8)]
            arm_joint_ids = robot.find_joints(arm_joint_names)[0]
            self._arm_joint_ids[side] = arm_joint_ids
            ee_body_idx = robot.find_bodies(f"{side}_palm_link")[0][0]
            self._ee_body_indices[side] = int(ee_body_idx)
            # PhysX jacobians omit the fixed base row, matching Isaac Lab's
            # DifferentialIKController examples for fixed-base manipulators.
            self._ee_jacobi_indices[side] = int(ee_body_idx) - 1
            self._ik_controllers[side] = self._ik_controller_cls(
                self._ik_controller_cfg_cls(
                    command_type="pose",
                    use_relative_mode=False,
                    ik_method="dls",
                    ik_params={"lambda_val": 0.04},
                ),
                num_envs=1,
                device=robot.device,
            )

            joint_names = list(robot.joint_names)
            for joint_name in _HAND_JOINT_NAMES:
                if not joint_name.startswith(f"{side}_"):
                    continue
                if joint_name in joint_names:
                    self._hand_joint_ids[side].append(joint_names.index(joint_name))
            print(
                f"[HandsOn/IsaacLab] {side} control map: "
                f"{len(arm_joint_ids)} arm joints, {len(self._hand_joint_ids[side])} hand joints, "
                f"ee_body={robot.body_names[self._ee_body_indices[side]]}",
                flush=True,
            )


    def reset(self, seed: int | None = None, **_: Any):
        if seed is not None:
            self._seed = int(seed)
            self._random.seed(self._seed)

        if self._manual_reset_already_applied:
            self._manual_reset_already_applied = False
            self._manual_reset_requested = False
            self._refresh_object_poses()
            info = {
                "succeed": False,
                "manual_reset_applied": True,
                "isaac_warnings": tuple(self._isaac_warnings),
            }
            info.update(self._task_info())
            return self._attach_physx_ground_truth(self._compute_observation()), info

        self._sample_reset_state(clear_manual_request=True)
        self._reset_scene_state()
        self._apply_reset_visual_randomization()
        self._reset_physics_steps = 0
        self._step_isaaclab()
        self._reset_physics_steps += 1
        self._refresh_object_poses()
        info = {
            "succeed": False,
            "isaac_warnings": tuple(self._isaac_warnings),
        }
        info.update(self._task_info())
        return self._attach_physx_ground_truth(self._compute_observation()), info

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        self.delta_h = np.asarray(
            [self._random.uniform(0.0, 0.05) if self.randomize else 0.0],
            dtype=np.float64,
        )
        self._right_tcp_pose = _RIGHT_HOME.copy()
        self._left_tcp_pose = _LEFT_HOME.copy()
        self._right_hand.fill(0.0)
        self._left_hand.fill(0.0)
        self._peg_ori_pose, self._socket_ori_pose = self._sample_initial_object_poses()
        self.env_step = 0
        self._success_counter = 0
        self._right_pose_commanded = False
        self._left_pose_commanded = False
        if clear_manual_request:
            self._manual_reset_requested = False
        self._arm_pose_filters = {"right": None, "left": None}


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


    def step_with_current_joint_targets(self, *, start_time: float | None = None):
        """Advance one control step using targets already written to the drives."""

        if start_time is None:
            start_time = time.monotonic()
        if self._manual_reset_requested:
            return self._manual_reset_step_result()

        physx_trace = getattr(self, "_physx_ground_truth_trace", None)
        if physx_trace is not None:
            physx_trace.begin_substep_capture()
        for substep_index in range(self._n_substeps):
            pre_substep = getattr(
                self,
                "_handson_pre_physics_substep_callback",
                None,
            )
            if callable(pre_substep):
                pre_substep(substep_index, self._n_substeps)
            self._step_isaaclab()
            if physx_trace is not None:
                physx_trace.capture_substep()
            post_substep = getattr(
                self,
                "_handson_post_physics_substep_callback",
                None,
            )
            if callable(post_substep):
                post_substep(substep_index, self._n_substeps)
            # A keyboard request can arrive while Kit pumps this physics
            # frame.  Consume it only after the complete control interval so
            # every recorded transition always contains the configured,
            # replayable number of substeps.  The added reset latency is at
            # most one control frame.
        self._refresh_object_poses()

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

        render_started = time.monotonic()
        observation = self._attach_physx_ground_truth(self._compute_observation())
        render_seconds = time.monotonic() - render_started
        work_seconds = time.monotonic() - start_time
        if self.hz > 0:
            time.sleep(max(0.0, (1.0 / self.hz) - work_seconds))
        wall_seconds = max(time.monotonic() - start_time, 1e-9)
        control_timing = {
            "target_hz": int(self.hz),
            "actual_hz": float(1.0 / wall_seconds),
            "work_ms": float(work_seconds * 1000.0),
            "render_ms": float(render_seconds * 1000.0),
            "sleep_ms": float(max(0.0, wall_seconds - work_seconds) * 1000.0),
            "sim_realtime_factor": float(self._control_dt / wall_seconds),
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


    def _update_recording_cameras(self) -> None:
        if self._scene is None or self._torch is None:
            return
        try:
            dtype = self._scene["peg"].data.root_pos_w.dtype
            device = self._scene["peg"].data.root_pos_w.device
            head_position = self._torch.tensor(
                [[*HEAD_RECORDING_CAMERA_EYE]],
                dtype=dtype,
                device=device,
            )
            head_orientation = self._torch.tensor(
                [[*HEAD_RECORDING_CAMERA_QUAT_WXYZ_OPENGL]],
                dtype=dtype,
                device=device,
            )
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
                    [[*_matrix_to_quat_wxyz(camera_pose[:3, :3])]],
                    dtype=dtype,
                    device=device,
                )
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


    def _apply_joint_targets(self) -> None:
        status: dict[str, str] = {}
        for side, hand in (
            ("right", self._right_hand),
            ("left", self._left_hand),
        ):
            robot = self._control_robot(side)
            target = self._joint_targets.get(side)
            if target is None:
                target = robot.data.default_joint_pos.clone()
            else:
                target = target.clone()

            pose = self._right_tcp_pose if side == "right" else self._left_tcp_pose
            pose_commanded = self._right_pose_commanded if side == "right" else self._left_pose_commanded
            arm_target, arm_status = self._compute_arm_joint_target(side, robot, pose, pose_commanded)
            status[f"{side}_arm"] = arm_status
            if arm_target is not None:
                if self._arm_ik_backend == "haorui" or _bool_env("HANDSON_ISAACLAB_FORCE_ARM_LIMITS", False):
                    arm_target = self._limit_arm_joint_target(side, robot, arm_target)
                target[:, self._arm_joint_ids[side]] = arm_target

            joint_ids = self._hand_joint_ids.get(side, [])
            for hand_index, joint_id in enumerate(joint_ids[: hand.size]):
                target[:, joint_id] = float(hand[hand_index])
            actual_hand_max = 0.0
            if joint_ids:
                try:
                    actual_hand_max = float(
                        self._torch.max(self._torch.abs(robot.data.joint_pos[:, joint_ids])).item()
                    )
                except Exception:
                    actual_hand_max = 0.0
            status[f"{side}_hand"] = (
                f"target={float(np.max(np.abs(hand))) if hand.size else 0.0:.3f}"
                f",actual={actual_hand_max:.3f},joints={len(joint_ids)}"
            )

            limits = getattr(robot.data, "soft_joint_pos_limits", None)
            if limits is not None:
                target = self._torch.maximum(target, limits[:, :, 0])
                target = self._torch.minimum(target, limits[:, :, 1])

            robot.set_joint_position_target(target)
            self._joint_targets[side] = target

        now = time.monotonic()
        if now - self._last_control_log_time >= 2.0:
            self._last_control_log_time = now
            print(
                "[HandsOn/IsaacLab] teleop targets: "
                f"right_arm={status.get('right_arm')} left_arm={status.get('left_arm')} "
                f"right_hand={status.get('right_hand')} left_hand={status.get('left_hand')}",
                flush=True,
            )

    def _compute_arm_joint_target(self, side: str, robot, pose: np.ndarray, commanded: bool):
        if not commanded:
            return None, "idle"
        if self._arm_ik_backend == "haorui":
            pose = self._limit_arm_pose_target(side, robot, pose)
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

            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
            ee_pose_w = robot.data.body_pose_w[:, ee_body_idx]
            root_pose_w = robot.data.root_pose_w
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
                np.asarray(pose, dtype=np.float32).reshape(1, 7),
                dtype=ee_pose_w.dtype,
                device=ee_pose_w.device,
            )
            target_pos_b, target_quat_b = self._math_utils["subtract_frame_transforms"](
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                target_pose_w[:, 0:3],
                target_pose_w[:, 3:7],
            )
            controller.set_command(self._torch.cat((target_pos_b, target_quat_b), dim=1))
            joint_pos = robot.data.joint_pos[:, arm_joint_ids]
            status = "haorui_fallback_dls" if self._arm_ik_backend == "haorui" else "isaaclab_dls"
            return controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos), status
        except Exception as exc:
            message = f"{side} arm IK failed: {exc}"
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)
                print(f"[HandsOn/IsaacLab] {message}", flush=True)
            return None, "ik_error"

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

        max_step = _float_env("HANDSON_ISAACLAB_ARM_MAX_SITE_STEP", _ARM_TARGET_MAX_SITE_STEP)
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
            current = robot.data.joint_pos
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

    def _compute_arm_joint_target_haorui(self, side: str, robot, pose: np.ndarray):
        try:
            q_now = self._arm_reference_q(side, robot)
            if q_now is None:
                return None

            base_from_world = _invert_pose_matrix(self._robot_base_pose_matrix(side, robot))
            current_ee_in_base = base_from_world @ self._robot_body_pose_matrix(
                robot,
                self._ee_body_indices[side],
            )
            target_ee_in_base = base_from_world @ _pose_vector_to_matrix(pose)

            model = self._haorui_ik_models[side]
            solver = self._haorui_ik_solvers[side]
            current_tcp_in_base = model.fk(q_now)
            tcp_to_ee = _invert_pose_matrix(current_tcp_in_base) @ current_ee_in_base
            target_tcp_in_base = target_ee_in_base @ _invert_pose_matrix(tcp_to_ee)
            if not np.all(np.isfinite(target_tcp_in_base)):
                return None

            seeds = self._haorui_rokae.make_seeds(side, q_now)
            sol = solver.solve_multi_start(
                target_tcp_in_base,
                seeds,
                first_success=True,
                reference_q=q_now,
            )
            if not sol.success:
                return None

            q = np.asarray(sol.q, dtype=np.float64).reshape(1, -1)
            if q.shape[1] != 7 or not np.all(np.isfinite(q)):
                return None
            return self._torch.tensor(
                q,
                dtype=robot.data.joint_pos.dtype,
                device=robot.data.joint_pos.device,
            )
        except Exception as exc:
            fallback = "isaaclab_dls" if _bool_env("HANDSON_ISAACLAB_ALLOW_DLS_FALLBACK", False) else "hold_previous"
            message = f"{side} Haorui IK failed; fallback={fallback} reason={exc}"
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)
                print(f"[HandsOn/IsaacLab] {message}", flush=True)
            return None


    def _robot_base_pose_matrix(self, side: str, robot) -> np.ndarray:
        for index, name in enumerate(robot.body_names):
            if name == f"base_{side}" or name.endswith(f"/base_{side}"):
                return self._robot_body_pose_matrix(robot, index)
        root_pose = robot.data.root_pose_w[0].detach().cpu().numpy().astype(np.float64)
        return _pose_vector_to_matrix(root_pose)

    def _robot_body_pose_matrix(self, robot, body_index: int) -> np.ndarray:
        pose = robot.data.body_pose_w[0, int(body_index), :7]
        return _pose_vector_to_matrix(pose.detach().cpu().numpy().astype(np.float64))


    def _body_pose_matrix(self, scene_name: str, body_substring: str, fallback_pose: np.ndarray) -> np.ndarray:
        try:
            side = "right" if scene_name.startswith("right") else "left"
            robot = self._control_robot(side)
            body_names = list(robot.body_names)
            body_index = next(index for index, name in enumerate(body_names) if body_substring in name)
            body_state = robot.data.body_state_w[0, body_index, :7]
            pose = body_state.detach().cpu().numpy().astype(np.float64)
            return _pose_vector_to_matrix(pose)
        except Exception:
            return _pose_vector_to_matrix(fallback_pose)


    def _camera_rgb_frame(self, sensor_name: str, fallback_key: str) -> np.ndarray:
        is_wrist = sensor_name.startswith("wrist_")
        if is_wrist:
            target_width, target_height = WRIST_RECORDING_CAMERA_OUTPUT_SIZE
        else:
            target_height, target_width = self._render_height, self._render_width
        if self._scene is not None:
            try:
                camera = self._scene[sensor_name]
                frame = camera.data.output["rgb"][0]
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
