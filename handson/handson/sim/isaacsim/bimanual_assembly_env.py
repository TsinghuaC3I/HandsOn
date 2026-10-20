from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Literal

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..envs.hand_specs import hand_action_dim, normalize_arm_type, normalize_hand_type
from .runtime import IsaacSimLaunchConfig, IsaacSimUnavailable, launch_simulation_app
from .urdf_assets import prepare_rokae_urdf_for_isaac, wuji_usd_path


_XML_DIR = Path(__file__).resolve().parents[1] / "envs" / "xmls"
_DEFAULT_SCENE_XML = _XML_DIR / "arena_arm_hand_bimanual_assembly.xml"
_DEFAULT_RENDER_SIZE = (640, 640)
_IDENTITY_QUAT_WXYZ = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
_RIGHT_HOME = np.asarray([-0.10, -0.16, 1.07, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
_LEFT_HOME = np.asarray([-0.10, 0.16, 1.07, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
_PEG_POSE = np.asarray([-0.16, -0.08, 0.94, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
_SOCKET_POSE = np.asarray([-0.13, 0.08, 0.94, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _bool_env(name: str, default: bool = False) -> bool:
    import os

    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _scene_xml_path(arm_type: str, hand_type: str) -> Path:
    arm = normalize_arm_type(arm_type)
    hand = normalize_hand_type(hand_type)
    if hand == "wuji1":
        suffix = f"{arm}_{hand}" if arm != "panda" else hand
        candidate = _DEFAULT_SCENE_XML.with_name(
            f"{_DEFAULT_SCENE_XML.stem}_{suffix}{_DEFAULT_SCENE_XML.suffix}"
        )
        if candidate.exists():
            return candidate
    return _DEFAULT_SCENE_XML


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


def _pose_vector_to_matrix(pose: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_wxyz_to_matrix(pose[3:7])
    matrix[:3, 3] = pose[:3]
    return matrix


def _coerce_rgb(frame: Any, shape: tuple[int, int, int]) -> np.ndarray | None:
    try:
        arr = np.asarray(frame)
    except Exception:
        return None
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
            arr = np.clip(arr * 255.0, 0, 255)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape != shape:
        return None
    return arr


class BimanualAssemblyIsaacSimEnv(gym.Env):
    """Isaac Sim backend for the HandsOn bimanual assembly task.

    This environment preserves HandsOn's bimanual assembly Gym contract while
    moving scene construction into Isaac Sim. It first tries to convert the
    task MJCF/XML with Isaac's MJCF importer; if that is unavailable, it falls
    back to importing the two ROKAE URDF arms and a small procedural task scene.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 10.0,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = True,
        randomize: bool = False,
        randomize_dynamics: bool = False,
        hand_type: str = "wuji1",
        arm_type: str = "rokae",
        launch_isaac: bool | None = None,
        headless: bool | None = None,
        render_resolution: tuple[int, int] = _DEFAULT_RENDER_SIZE,
        isaac_cache_dir: str | Path | None = None,
        hz: int = 30,
        **_: Any,
    ):
        del randomize_dynamics
        self.hz = int(hz)
        self.render_mode = render_mode
        self.image_obs = bool(image_obs)
        self.randomize = bool(randomize)
        self.hand_type = normalize_hand_type(hand_type)
        self.arm_type = normalize_arm_type(arm_type)
        self._hand_action_dim = hand_action_dim(self.hand_type)
        self._random = np.random.RandomState(seed)
        self._control_dt = float(control_dt)
        self._physics_dt = float(physics_dt)
        self._n_substeps = max(1, int(round(self._control_dt / self._physics_dt)))
        self._time_limit = float(time_limit)
        self._render_height = int(render_resolution[0])
        self._render_width = int(render_resolution[1])
        self._image_shape = (self._render_height, self._render_width, 3)
        self._isaac_cache_dir = (
            Path(isaac_cache_dir)
            if isaac_cache_dir is not None
            else Path.cwd() / ".handson_isaacsim"
        )
        self._isaac_cache_dir.mkdir(parents=True, exist_ok=True)

        if launch_isaac is None:
            launch_isaac = not _bool_env("HANDSON_ISAACSIM_INTERFACE_ONLY")
        self._launch_isaac = bool(launch_isaac)
        self._headless = render_mode != "human" if headless is None else bool(headless)
        self._simulation_app = None
        self._world = None
        self._stage = None
        self._timeline = None
        self._camera_sensors: dict[str, Any] = {}
        self._isaac_warnings: list[str] = []

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
            self._setup_isaac()

    @property
    def control_dt(self) -> float:
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        return self._physics_dt

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random

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
            observation_spaces["images"] = spaces.Dict(
                {
                    "wrist_left": spaces.Box(0, 255, shape=self._image_shape, dtype=np.uint8),
                    "wrist_right": spaces.Box(0, 255, shape=self._image_shape, dtype=np.uint8),
                    "random_camera" if self.randomize else "ego": spaces.Box(
                        0,
                        255,
                        shape=self._image_shape,
                        dtype=np.uint8,
                    ),
                }
            )
        return spaces.Dict(observation_spaces)

    def _setup_isaac(self) -> None:
        launch_config = IsaacSimLaunchConfig(
            headless=self._headless,
            width=self._render_width,
            height=self._render_height,
        )
        self._simulation_app = launch_simulation_app(launch_config)

        try:
            from isaacsim.core.api import World
            import omni.timeline
            import omni.usd
        except Exception as exc:  # pragma: no cover - exercised in Isaac runtime
            raise IsaacSimUnavailable("Isaac Sim core APIs are not available.") from exc

        try:
            World.clear_instance()
        except Exception:
            pass
        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=self._physics_dt,
            rendering_dt=self._control_dt,
        )
        self._world.scene.add_default_ground_plane()
        self._stage = omni.usd.get_context().get_stage()
        self._timeline = omni.timeline.get_timeline_interface()
        if self._timeline is not None:
            self._timeline.play()

        imported = self._try_import_mjcf_scene()
        if not imported:
            self._create_procedural_scene()
            self._try_import_rokae_urdfs()
            self._try_reference_wuji_usds()
        self._create_target_markers()
        self._try_create_cameras()
        self._step_isaac(render=True)

    def _try_enable_extension(self, name: str) -> None:
        try:
            from isaacsim.core.utils.extensions import enable_extension

            enable_extension(name)
            if self._simulation_app is not None:
                self._simulation_app.update()
        except Exception as exc:
            self._isaac_warnings.append(f"Could not enable {name}: {exc}")

    def _try_import_mjcf_scene(self) -> bool:
        self._try_enable_extension("isaacsim.asset.importer.mjcf")
        try:
            from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
        except Exception as exc:
            self._isaac_warnings.append(f"MJCF importer unavailable: {exc}")
            return False

        scene_xml = _scene_xml_path(self.arm_type, self.hand_type)
        output_dir = self._isaac_cache_dir / "mjcf"
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            config = MJCFImporterConfig(
                mjcf_path=str(scene_xml),
                usd_path=str(output_dir),
                import_scene=True,
                merge_mesh=False,
                collision_from_visuals=False,
            )
            usd_path = MJCFImporter(config).import_mjcf()
            self._reference_usd(usd_path, "/World/HandsOn/MJCFScene")
            return True
        except Exception as exc:
            self._isaac_warnings.append(f"MJCF scene import failed for {scene_xml}: {exc}")
            return False

    def _try_import_rokae_urdfs(self) -> None:
        self._try_enable_extension("isaacsim.asset.importer.urdf")
        try:
            from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
        except Exception as exc:
            self._isaac_warnings.append(f"URDF importer unavailable: {exc}")
            return

        urdf_dir = self._isaac_cache_dir / "urdf"
        usd_dir = self._isaac_cache_dir / "usd"
        for side, prim_path in (
            ("right", "/World/HandsOn/RokaeRight"),
            ("left", "/World/HandsOn/RokaeLeft"),
        ):
            try:
                urdf_path = prepare_rokae_urdf_for_isaac(side, urdf_dir)
                config = URDFImporterConfig(
                    urdf_path=str(urdf_path),
                    usd_path=str(usd_dir),
                    collision_from_visuals=True,
                    merge_mesh=False,
                )
                usd_path = URDFImporter(config).import_urdf()
                self._reference_usd(usd_path, prim_path)
            except Exception as exc:
                self._isaac_warnings.append(f"ROKAE {side} URDF import failed: {exc}")

    def _try_reference_wuji_usds(self) -> None:
        if self.hand_type != "wuji1":
            return
        for side, prim_path in (
            ("right", "/World/HandsOn/WujiRight"),
            ("left", "/World/HandsOn/WujiLeft"),
        ):
            usd_path = wuji_usd_path(side)
            if not usd_path.exists():
                self._isaac_warnings.append(f"Wuji {side} USD asset not found: {usd_path}")
                continue
            try:
                self._reference_usd(usd_path, prim_path)
            except Exception as exc:
                self._isaac_warnings.append(f"Wuji {side} USD reference failed: {exc}")

    def _reference_usd(self, usd_path: str | Path, prim_path: str) -> None:
        if self._stage is None:
            return
        prim = self._stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(str(usd_path))

    def _create_procedural_scene(self) -> None:
        if self._stage is None:
            return
        try:
            from pxr import Gf, UsdGeom, UsdPhysics
        except Exception as exc:
            self._isaac_warnings.append(f"USD scene creation unavailable: {exc}")
            return

        UsdGeom.Xform.Define(self._stage, "/World/HandsOn")

        def cube(path: str, pos: tuple[float, float, float], scale: tuple[float, float, float]):
            prim = UsdGeom.Cube.Define(self._stage, path)
            prim.CreateSizeAttr(1.0)
            xform = UsdGeom.XformCommonAPI(prim)
            xform.SetTranslate(Gf.Vec3d(*pos))
            xform.SetScale(Gf.Vec3f(*scale))
            UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
            return prim

        cube("/World/HandsOn/Table", (-0.15, 0.0, 0.86), (0.8, 1.6, 0.06))
        cube("/World/HandsOn/Peg", tuple(_PEG_POSE[:3]), (0.025, 0.025, 0.08))
        cube("/World/HandsOn/Socket", tuple(_SOCKET_POSE[:3]), (0.10, 0.10, 0.025))

    def _create_target_markers(self) -> None:
        if self._stage is None:
            return
        try:
            from pxr import Gf, UsdGeom
        except Exception:
            return
        for name, pose in (
            ("RightTarget", self._right_tcp_pose),
            ("LeftTarget", self._left_tcp_pose),
        ):
            sphere = UsdGeom.Sphere.Define(self._stage, f"/World/HandsOn/{name}")
            sphere.CreateRadiusAttr(0.025)
            UsdGeom.XformCommonAPI(sphere).SetTranslate(Gf.Vec3d(*pose[:3]))

    def _set_prim_translation(self, prim_path: str, pos: np.ndarray) -> None:
        if self._stage is None:
            return
        try:
            from pxr import Gf, UsdGeom
        except Exception:
            return
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return
        UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*np.asarray(pos, dtype=float)))

    def _try_create_cameras(self) -> None:
        self._try_enable_extension("isaacsim.sensors.experimental.rtx")
        try:
            from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
        except Exception as exc:
            self._isaac_warnings.append(f"RTX camera APIs unavailable: {exc}")
            return

        specs = {
            "ego": ("/World/HandsOn/Cameras/Ego", (1.2, 0.0, 1.8), (0.707, 0.0, 0.707, 0.0)),
            "wrist_left": (
                "/World/HandsOn/Cameras/WristLeft",
                (-0.05, 0.16, 1.12),
                (0.707, 0.0, 0.707, 0.0),
            ),
            "wrist_right": (
                "/World/HandsOn/Cameras/WristRight",
                (-0.05, -0.16, 1.12),
                (0.707, 0.0, 0.707, 0.0),
            ),
        }
        for key, (path, pos, quat) in specs.items():
            try:
                camera = RtxCamera(
                    path,
                    positions=np.asarray([pos], dtype=np.float32),
                    orientations=np.asarray([quat], dtype=np.float32),
                    tick_rate=float(self.hz),
                )
                sensor = CameraSensor(
                    camera,
                    annotators=["rgb"],
                    resolution=(self._render_width, self._render_height),
                )
                self._camera_sensors[key] = sensor
            except Exception as exc:
                self._isaac_warnings.append(f"Camera {key} creation failed: {exc}")

    def _step_isaac(self, render: bool = False) -> None:
        if self._world is not None:
            self._world.step(render=render)
        elif self._simulation_app is not None:
            self._simulation_app.update()

    def reset(self, seed: int | None = None, **_: Any):
        if seed is not None:
            self._random.seed(seed)

        self.delta_h = np.asarray([self._random.uniform(0.0, 0.05)], dtype=np.float64)
        self._right_tcp_pose = _RIGHT_HOME.copy()
        self._left_tcp_pose = _LEFT_HOME.copy()
        self._right_hand.fill(0.0)
        self._left_hand.fill(0.0)
        self._peg_ori_pose = _PEG_POSE.copy()
        self._socket_ori_pose = _SOCKET_POSE.copy()
        self._peg_ori_pose[2] += float(self.delta_h[0])
        self._socket_ori_pose[2] += float(self.delta_h[0])
        self.env_step = 0
        self._success_counter = 0

        if self._world is not None:
            self._world.reset()
        self._sync_targets_to_stage()
        self._step_isaac(render=self.render_mode == "human")
        return self._compute_observation(), {
            "succeed": False,
            "isaac_warnings": tuple(self._isaac_warnings),
        }

    def step(self, action):
        start_time = time.time()
        self._apply_action(action)
        self._sync_targets_to_stage()
        for _ in range(self._n_substeps):
            self._step_isaac(render=self.render_mode == "human")

        success = self._compute_success()
        reward = 1.0 if success else 0.0
        self.env_step += 1
        terminated = success or (self.env_step * self._control_dt >= self._time_limit)

        dt = time.time() - start_time
        if self.hz > 0:
            time.sleep(max(0.0, (1.0 / self.hz) - dt))

        return self._compute_observation(), reward, terminated, False, {
            "succeed": success,
            "isaac_warnings": tuple(self._isaac_warnings),
        }

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
        if left.size >= 7 and not np.allclose(left[:7], 0.0):
            self._left_tcp_pose = left[:7].astype(np.float64, copy=True)
        if right.size >= 7 + self._hand_action_dim:
            self._right_hand = right[7 : 7 + self._hand_action_dim].astype(np.float64, copy=True)
        if left.size >= 7 + self._hand_action_dim:
            self._left_hand = left[7 : 7 + self._hand_action_dim].astype(np.float64, copy=True)

    def _sync_targets_to_stage(self) -> None:
        self._set_prim_translation("/World/HandsOn/RightTarget", self._right_tcp_pose[:3])
        self._set_prim_translation("/World/HandsOn/LeftTarget", self._left_tcp_pose[:3])
        self._set_prim_translation("/World/HandsOn/WujiRight", self._right_tcp_pose[:3])
        self._set_prim_translation("/World/HandsOn/WujiLeft", self._left_tcp_pose[:3])
        self._set_prim_translation("/World/HandsOn/Peg", self._peg_ori_pose[:3])
        self._set_prim_translation("/World/HandsOn/Socket", self._socket_ori_pose[:3])

    def _compute_success(self) -> bool:
        distance = float(np.linalg.norm(self._peg_ori_pose[:3] - self._socket_ori_pose[:3]))
        if distance < 0.025:
            self._success_counter += 1
        else:
            self._success_counter = 0
        return self._success_counter >= 30

    def render(self):
        frames = []
        for key in ("ego", "wrist_left", "wrist_right"):
            frame = self._render_camera(key)
            frames.append(frame)
        return frames

    def _render_camera(self, key: str) -> np.ndarray:
        sensor = self._camera_sensors.get(key)
        if sensor is not None:
            try:
                data = sensor.get_data("rgb")
                if isinstance(data, tuple):
                    data = data[0]
                frame = _coerce_rgb(data, self._image_shape)
                if frame is not None:
                    return frame
            except Exception:
                pass

        frame = np.zeros(self._image_shape, dtype=np.uint8)
        colors = {
            "ego": (56, 78, 96),
            "wrist_left": (80, 64, 112),
            "wrist_right": (112, 72, 54),
        }
        frame[:, :] = colors.get(key, (48, 48, 48))
        h, w, _ = frame.shape
        frame[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3, :] = (150, 150, 140)
        return frame

    def _compute_observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = {}
        if self.image_obs:
            ego, wrist_left, wrist_right = self.render()
            obs["images"] = {
                "random_camera" if self.randomize else "ego": ego,
                "wrist_left": wrist_left,
                "wrist_right": wrist_right,
            }

        gripper_pose = np.zeros(32, dtype=np.float64)
        right_dim = min(16, self._right_hand.size)
        left_dim = min(16, self._left_hand.size)
        gripper_pose[:right_dim] = self._right_hand[:right_dim]
        gripper_pose[16 : 16 + left_dim] = self._left_hand[:left_dim]

        obs["state"] = {
            "tcp_pose": np.concatenate([self._right_tcp_pose, self._left_tcp_pose]).astype(
                np.float64
            ),
            "gripper_pose": gripper_pose,
            "socket_ori_pose": self._socket_ori_pose.astype(np.float64, copy=True),
            "peg_ori_pose": self._peg_ori_pose.astype(np.float64, copy=True),
            "table_delta_height": self.delta_h.astype(np.float64, copy=True),
        }
        return obs

    def get_end_effector_pose_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            _pose_vector_to_matrix(self._right_tcp_pose),
            _pose_vector_to_matrix(self._left_tcp_pose),
        )

    def close(self) -> None:
        if self._timeline is not None:
            try:
                self._timeline.stop()
            except Exception:
                pass
            self._timeline = None
        if self._world is not None:
            try:
                self._world.stop()
            except Exception:
                pass
            try:
                self._world.clear()
            except Exception:
                pass
            self._world = None
        try:
            from isaacsim.core.api import World

            World.clear_instance()
        except Exception:
            pass
