"""Native Isaac Sim 5.1 / PhysX runtime for P0-02 stage one."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from gymnasium import spaces
import numpy as np

from .abc_collection_env import HandsOnABCIsaacLabEnv
from .p002_contract import (
    ASSET_RELATIVE_PATH,
    ASSET_SCALE,
    ASSET_SHA256,
    CONTROL_HZ,
    DOOR_BODY_PATH,
    DOOR_DOF,
    DOOR_ROOT_PATH,
    EXPLOSION_ANGULAR_SPEED_RADPS,
    EXPECTED_DOF_NAMES,
    FORBIDDEN_CONTACT_FORCE_N,
    FRAME_BODY_PATH,
    FRONT_HANDLE_BODY_PATH,
    FRONT_HANDLE_DOF,
    GRAVITY_DISABLED_BODY_PATHS,
    HANDLE_DAMPING,
    HANDLE_MAX_FORCE,
    HANDLE_PRESS_DEG,
    HANDLE_RELEASE_DEG,
    HANDLE_STIFFNESS,
    LOCKED_DOOR_DAMPING,
    LOCKED_DOOR_MAX_FORCE,
    LOCKED_DOOR_STIFFNESS,
    MAX_PENETRATION_M,
    NOMINAL_ROOT_XYZ_M,
    NOMINAL_ROOT_YAW_DEG,
    PHYSICS_HZ,
    P002LatchStateMachine,
    P002PhysicalReadback,
    P002ResetState,
    P002_ROLE_CONTRACT_VERSION,
    REAR_HANDLE_BODY_PATH,
    REAR_HANDLE_DOF,
    RENDER_RESOLUTION_HW,
    SUCCESS_HOLD_STEPS,
    TASK_ID,
    TASK_NAME,
    TIMEOUT_STEPS,
    UNLOCKED_DOOR_DAMPING,
    UNLOCKED_DOOR_MAX_FORCE,
    UNLOCKED_DOOR_STIFFNESS,
    evaluate_physical_readback,
    physical_failure_reasons,
    sample_reset_state,
)
from .p002_scene_geometry import (
    P002SceneGeometryReport,
    apply_p002_scene_geometry,
)


_LIGHT_ROOT = "/World/Task/P002Lights"
_MATERIAL_PATH = "/World/PhysicsMaterials/P002Door"
# P0-02 uses the shared platform base placement and mirrored arm home. Moving
# both bases 0.35 m forward while solving the palms back to the same TCP home
# produced a visually folded, task-specific initial posture even though the
# palm-only gate passed. Keep the ordinary bimanual posture and move the task
# fixture instead.
_P002_ARM_HOME_RAD = {
    "right": (
        -1.5707963,
        1.5707963,
        1.5707963,
        1.5707963,
        -1.5707963,
        0.0,
        0.0,
    ),
    "left": (
        1.5707963,
        1.5707963,
        -1.5707963,
        1.5707963,
        1.5707963,
        0.0,
        0.0,
    ),
}
_P002_EXPECTED_ROBOT_BASE_POSES_WXYZ = {
    "right": (-0.75, -0.065, 1.34, 0.7071067812, 0.7071067812, 0.0, 0.0),
    "left": (-0.75, 0.065, 1.34, 0.7071067812, -0.7071067812, 0.0, 0.0),
}
_P002_EXPECTED_PALM_POSES_WXYZ = {
    # Physical ``palm_link`` world poses measured by the current Isaac 5.1
    # acceptance run for the standard x=-0.75 m bases and mirrored arm home.
    # These are world readback poses, not controller task-frame TCP targets.
    "right": np.asarray(
        [
            -0.3715,
            -0.2395,
            0.9975,
            0.7071067811865476,
            0.0,
            0.7071067811865476,
            0.0,
        ],
        dtype=np.float64,
    ),
    "left": np.asarray(
        [
            -0.3715,
            0.2395,
            0.9975,
            0.7071067811865476,
            0.0,
            0.7071067811865476,
            0.0,
        ],
        dtype=np.float64,
    ),
}
# Door-panel and door-frame contact are non-terminal.  Table contact has no
# P0-02 semantic role at all; its legacy observation/telemetry slot remains
# zero-filled.  Only the intended handle pair and the rear handle contribute
# to the protected penetration aggregate.
_P002_HARD_FAILURE_PENETRATION_KEYS = (
    "right_front_handle",
    "robot_rear_handle",
)
_P002_SUCCESS_CONDITION_DISPLAY_LABELS = {
    "p002_right_handle": "Right hand contacts and depresses the front lever",
    "p002_unlatched": "Lever reaches 45 deg for 5 steps; latch released",
    "p002_left_door": "Left hand contacts the panel after unlatching",
    "p002_target": "Door angle is at least 45 deg",
    "p002_stable": "Door angular speed is at most 0.5 rad/s",
    "p002_forbidden": "Rear handle clear; penetration <= 5 mm",
    # Cached/legacy row keys should still render the corrected role wording.
    "p002_left_handle": "Right hand contacts and depresses the front lever",
    "p002_right_door": "Left hand contacts the panel after unlatching",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quat_mul_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.asarray(left, dtype=np.float64).reshape(4)
    rx, ry, rz, rw = np.asarray(right, dtype=np.float64).reshape(4)
    result = np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )
    return result / np.linalg.norm(result)


def _pose_matrix_wxyz(pose: np.ndarray | tuple[float, ...]) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64).reshape(7)
    w, x, y, z = values[3:7]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    result[:3, 3] = values[:3]
    return result


def _rotation_error_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left[:3, :3]).T @ np.asarray(right[:3, :3])
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _numpy_tensor(value: Any) -> np.ndarray:
    value = value.torch if hasattr(value, "torch") else value
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _randomized_root_pose_xyzw(
    composed_pose_xyzw: np.ndarray,
    reset: P002ResetState,
) -> np.ndarray:
    """Apply one sampled task-frame delta to the nominal composed root pose."""

    pose = np.asarray(composed_pose_xyzw, dtype=np.float64).reshape(7).copy()
    if not np.isfinite(pose).all():
        raise ValueError("P0-02 composed root pose must be finite")
    pose[:3] += np.asarray(
        [
            float(reset.root_x_m) - float(NOMINAL_ROOT_XYZ_M[0]),
            float(reset.root_y_m) - float(NOMINAL_ROOT_XYZ_M[1]),
            float(reset.root_z_m) - float(NOMINAL_ROOT_XYZ_M[2]),
        ],
        dtype=np.float64,
    )
    delta_yaw = math.radians(
        float(reset.root_yaw_deg) - float(NOMINAL_ROOT_YAW_DEG)
    )
    delta_quat = np.asarray(
        [0.0, 0.0, math.sin(delta_yaw / 2.0), math.cos(delta_yaw / 2.0)],
        dtype=np.float64,
    )
    pose[3:7] = _quat_mul_xyzw(delta_quat, pose[3:7])
    return pose


def _json_safe_number(value: Any) -> float | str | None:
    """Keep terminal diagnostics serializable even for non-finite readback."""

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0.0 else "-inf"
    return number


class HandsOnP002IsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Bimanual lever-unlock and door-opening environment."""

    def __init__(self, task_name: str = TASK_NAME, *args: Any, **kwargs: Any):
        if task_name != TASK_NAME:
            raise ValueError(f"P0-02 runtime requires {TASK_NAME!r}, got {task_name!r}")
        physics_backend = str(kwargs.get("physics_backend", "physx")).lower()
        if physics_backend not in {"physx", "isaacsim", "isaac"}:
            raise ValueError("P0-02 stage one requires Isaac Sim 5.1 PhysX")
        kwargs["physics_backend"] = "physx"
        kwargs.setdefault("physics_dt", 1.0 / PHYSICS_HZ)
        kwargs.setdefault("control_dt", 1.0 / CONTROL_HZ)
        kwargs.setdefault("hz", CONTROL_HZ)
        kwargs.setdefault("time_limit", TIMEOUT_STEPS / CONTROL_HZ)
        kwargs.setdefault("render_resolution", RENDER_RESOLUTION_HW)

        requested_seed = 0 if kwargs.get("seed") is None else int(kwargs["seed"])
        self._p002_reset_serial = -1
        self._p002_reset = sample_reset_state(requested_seed)
        self._p002_articulation_view: Any | None = None
        self._p002_body_views: dict[str, Any] = {}
        self._p002_contact_views: dict[str, Any] = {}
        self._p002_contact_paths: dict[str, tuple[str, tuple[str, ...]]] = {}
        self._p002_dof_indices = {
            "rear_handle": REAR_HANDLE_DOF,
            "front_handle": FRONT_HANDLE_DOF,
            "door": DOOR_DOF,
        }
        self._p002_latch = P002LatchStateMachine()
        self._p002_drive_locked: bool | None = None
        self._p002_last_readback: P002PhysicalReadback | None = None
        self._p002_last_metrics: dict[str, Any] = {}
        self._p002_failure_latched = False
        self._p002_first_failure_event: dict[str, Any] | None = None
        self._p002_terminal_reported = False
        self._p002_contact_contract_ready = False
        self._p002_scene_geometry_report: P002SceneGeometryReport | None = None
        self._p002_initial_robot_state_report: dict[str, Any] | None = None
        self._p002_initial_task_state_report: dict[str, Any] | None = None
        self._p002_composed_frame_pose_wxyz: np.ndarray | None = None
        self._p002_reset_frame_pose_wxyz = np.asarray(
            [*NOMINAL_ROOT_XYZ_M, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        super().__init__(task_name, *args, **kwargs)

    def _validate_task_initial_state_before_ready(self) -> dict[str, Any] | None:
        """Validate actual P0-02 robot and task state before every ready boundary."""
        if not self._robots or self._p002_articulation_view is None:
            return None
        robot_report = self._validate_initial_robot_state_or_raise()
        task_report = self._validate_initial_task_state_or_raise()
        self._p002_initial_robot_state_report = robot_report
        self._p002_initial_task_state_report = task_report
        return {"robots": robot_report, "task": task_report}

    def _validate_initial_robot_state_or_raise(self) -> dict[str, Any]:
        """Fail before collection-ready if the applied dual-arm home is wrong."""

        expected_palms = _P002_EXPECTED_PALM_POSES_WXYZ
        report: dict[str, Any] = {}
        for side in ("right", "left"):
            robot = self._robots[side]
            arm_ids = self._arm_joint_ids[side]
            actual_base = self._robot_base_pose_matrix(side, robot)
            expected_base = _pose_matrix_wxyz(
                _P002_EXPECTED_ROBOT_BASE_POSES_WXYZ[side]
            )
            actual_palm = self._robot_body_pose_matrix(
                robot, self._ee_body_indices[side]
            )
            expected_palm = _pose_matrix_wxyz(expected_palms[side])
            actual_q = _numpy_tensor(robot.data.joint_pos)[0, arm_ids]
            actual_qd = _numpy_tensor(robot.data.joint_vel)[0, arm_ids]
            expected_q = np.asarray(_P002_ARM_HOME_RAD[side], dtype=np.float64)
            checks = {
                "base_position": float(
                    np.linalg.norm(actual_base[:3, 3] - expected_base[:3, 3])
                )
                <= 0.002,
                "base_orientation": _rotation_error_deg(
                    actual_base, expected_base
                )
                <= 0.5,
                "arm_home": float(np.max(np.abs(actual_q - expected_q))) <= 0.01,
                "arm_stationary": float(np.max(np.abs(actual_qd))) <= 0.05,
                "palm_position": float(
                    np.linalg.norm(actual_palm[:3, 3] - expected_palm[:3, 3])
                )
                <= 0.03,
                "palm_orientation": _rotation_error_deg(
                    actual_palm, expected_palm
                )
                <= 5.0,
                "correct_side": bool(
                    actual_palm[1, 3] < -0.05
                    if side == "right"
                    else actual_palm[1, 3] > 0.05
                ),
            }
            report[side] = {
                "passed": all(checks.values()),
                "checks": checks,
                "actual_base_position_m": actual_base[:3, 3].tolist(),
                "actual_arm_joint_position_rad": actual_q.tolist(),
                "actual_palm_position_m": actual_palm[:3, 3].tolist(),
                "palm_position_error_m": float(
                    np.linalg.norm(actual_palm[:3, 3] - expected_palm[:3, 3])
                ),
                "palm_orientation_error_deg": _rotation_error_deg(
                    actual_palm, expected_palm
                ),
            }
        if not all(row["passed"] for row in report.values()):
            raise RuntimeError(
                "P0-02 initial robot base/home/palm gate failed: "
                + json.dumps(report, sort_keys=True)
            )
        print(
            "[HandsOn/P002] initial robot gate ready: bases=2/2, "
            "arm_homes=2/2, palms=2/2",
            flush=True,
        )
        return report

    def _validate_initial_task_state_or_raise(self) -> dict[str, Any]:
        """Fail closed on the applied Door042 pose, joints, and contacts."""

        view = self._p002_articulation_view
        if view is None:
            raise RuntimeError("P0-02 initial task-state gate lacks Door042 view")
        actual_xyzw = self._view_numpy(
            view.get_root_transforms(),
            (int(view.count), 7),
        )[0]
        actual_wxyz = np.concatenate(
            [actual_xyzw[:3], actual_xyzw[6:7], actual_xyzw[3:6]]
        )
        expected_wxyz = np.asarray(
            self._p002_reset_frame_pose_wxyz,
            dtype=np.float64,
        ).reshape(7)
        actual_matrix = _pose_matrix_wxyz(actual_wxyz)
        expected_matrix = _pose_matrix_wxyz(expected_wxyz)
        positions = self._joint_positions()
        velocities = self._joint_velocities()
        readback = self._physical_readback()
        initial_front_handle = self._contact_reading(
            "initial_robot_front_handle"
        )
        initial_door = self._contact_reading("initial_robot_door")
        initial_frame = self._contact_reading("robot_frame")
        initial_rear_handle = self._contact_reading("robot_rear_handle")
        contact_forces = np.asarray(
            [
                readback.right_front_handle_contact_force_n,
                readback.left_door_contact_force_n,
                readback.robot_frame_contact_force_n,
                readback.robot_rear_handle_contact_force_n,
                initial_front_handle[0],
                initial_door[0],
            ],
            dtype=np.float64,
        )
        initial_clearance_penetrations = np.asarray(
            [
                initial_front_handle[1],
                initial_door[1],
                initial_frame[1],
                initial_rear_handle[1],
            ],
            dtype=np.float64,
        )
        numeric_truth = np.concatenate(
            [
                actual_wxyz,
                expected_wxyz,
                positions,
                velocities,
                contact_forces,
                initial_clearance_penetrations,
                np.asarray(
                    [
                        readback.max_penetration_m,
                        readback.hard_failure_max_penetration_m,
                    ],
                    dtype=np.float64,
                ),
            ]
        )
        checks = {
            "finite": bool(np.isfinite(numeric_truth).all()),
            "root_quaternion": bool(
                np.isclose(np.linalg.norm(actual_wxyz[3:7]), 1.0, atol=1.0e-6)
                and np.isclose(
                    np.linalg.norm(expected_wxyz[3:7]),
                    1.0,
                    atol=1.0e-6,
                )
            ),
            "root_position": float(
                np.linalg.norm(actual_matrix[:3, 3] - expected_matrix[:3, 3])
            )
            <= 0.002,
            "root_orientation": _rotation_error_deg(
                actual_matrix,
                expected_matrix,
            )
            <= 0.5,
            "neutral_joints": float(np.max(np.abs(positions)))
            <= math.radians(2.5),
            "stationary_joints": float(np.max(np.abs(velocities))) <= 0.5,
            "initial_contacts_clear": bool(
                np.max(np.abs(contact_forces)) < FORBIDDEN_CONTACT_FORCE_N
            ),
            "initial_robot_door_clearance": bool(
                np.max(initial_clearance_penetrations) <= 0.0
            ),
            "hard_penetration": (
                float(readback.hard_failure_max_penetration_m)
                <= MAX_PENETRATION_M
            ),
        }
        report = {
            "passed": all(checks.values()),
            "checks": checks,
            "expected_root_pose_wxyz": expected_wxyz.tolist(),
            "actual_root_pose_wxyz": actual_wxyz.tolist(),
            "root_position_error_m": float(
                np.linalg.norm(actual_matrix[:3, 3] - expected_matrix[:3, 3])
            ),
            "root_orientation_error_deg": _rotation_error_deg(
                actual_matrix,
                expected_matrix,
            ),
            "joint_position_rad": positions.tolist(),
            "joint_velocity_rad_s": velocities.tolist(),
            "initial_contact_forces_n": contact_forces.tolist(),
            "initial_robot_door_penetrations_m": (
                initial_clearance_penetrations.tolist()
            ),
            "hard_failure_max_penetration_m": float(
                readback.hard_failure_max_penetration_m
            ),
        }
        if not report["passed"]:
            raise RuntimeError(
                "P0-02 initial task pose/joint/contact gate failed: "
                + json.dumps(report, sort_keys=True)
            )
        print(
            "[HandsOn/P002] initial task gate ready: root=1/1, "
            "joints=3/3, role_contacts=4/4, full_robot_clearance=4/4",
            flush=True,
        )
        return report

    def _configure_task_robot_cfg(self, robot_cfg: Any, side: str) -> Any:
        robot_cfg = super()._configure_task_robot_cfg(robot_cfg, side)
        expected_base = np.asarray(
            _P002_EXPECTED_ROBOT_BASE_POSES_WXYZ[side][:3],
            dtype=np.float64,
        )
        configured_base = np.asarray(
            robot_cfg.init_state.pos,
            dtype=np.float64,
        ).reshape(3)
        if not np.allclose(configured_base, expected_base, atol=1.0e-12, rtol=0.0):
            raise RuntimeError(
                f"P0-02 requires the standard {side} robot base, got "
                f"{configured_base.tolist()}"
            )
        joint_pos = robot_cfg.init_state.joint_pos
        if not isinstance(joint_pos, dict):
            raise RuntimeError("P0-02 robot init_state.joint_pos must be a mapping")
        for index, value in enumerate(_P002_ARM_HOME_RAD[side], start=1):
            key = f"joint{index}_{side}(_1)?"
            configured = joint_pos.get(key)
            if configured is None or not math.isclose(
                float(configured), float(value), abs_tol=1.0e-7, rel_tol=0.0
            ):
                raise RuntimeError(
                    f"P0-02 requires the standard {side} arm home at {key}; "
                    f"expected {float(value):.7f}, got {configured!r}"
                )
        # The HP acceptance runner may point at a converter-produced USD set
        # that has already passed the baseline robot smoke tests.  This avoids
        # Isaac 5.1's non-reentrant adjacent ``*_tmp`` mesh conversion when
        # several benchmark jobs share the same source tree.  Normal platform
        # collection keeps the repository's standard MJCF path unless this
        # explicit variable is present.
        prebuilt_root = os.environ.get("HANDSON_P002_PREBUILT_ROBOT_USD_ROOT", "")
        if prebuilt_root:
            from isaaclab import sim as sim_utils

            usd_path = (
                Path(prebuilt_root).expanduser()
                / f"rokae_wuji1_{side}"
                / f"rokae_wuji1_{side}_isaac.usd"
            )
            if not usd_path.is_file():
                raise FileNotFoundError(
                    f"P0-02 prebuilt {side} robot USD is missing: {usd_path}"
                )
            source_spawn = robot_cfg.spawn
            robot_cfg.spawn = sim_utils.UsdFileCfg(
                usd_path=str(usd_path.resolve()),
                rigid_props=source_spawn.rigid_props,
                articulation_props=source_spawn.articulation_props,
                joint_drive_props=source_spawn.joint_drive_props,
            )
        return robot_cfg

    def _build_observation_space(self) -> spaces.Dict:
        observation_space = super()._build_observation_space()
        state_spaces = observation_space.spaces["state"].spaces
        state_spaces.update(
            {
                "front_handle_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "door_panel_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "p002_joint_state": spaces.Box(-np.inf, np.inf, shape=(6,)),
                "p002_role_contacts": spaces.Box(-np.inf, np.inf, shape=(5,)),
                "p002_latch_state": spaces.Box(-np.inf, np.inf, shape=(8,)),
                "p002_role_contract_version": spaces.Box(
                    -np.inf, np.inf, shape=(1,)
                ),
            }
        )
        return observation_space

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        if self._stage is None:
            raise RuntimeError("P0-02 stage is unavailable before asset composition")

        from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics

        package_root = Path(__file__).resolve().parents[2]
        asset_path = package_root / ASSET_RELATIVE_PATH
        if not asset_path.is_file():
            raise FileNotFoundError(f"P0-02 normalized Door042 asset is missing: {asset_path}")
        actual_sha = _sha256(asset_path)
        if actual_sha != ASSET_SHA256:
            raise RuntimeError(
                f"P0-02 asset SHA mismatch: expected {ASSET_SHA256}, got {actual_sha}"
            )

        root = UsdGeom.Xform.Define(self._stage, DOOR_ROOT_PATH).GetPrim()
        root.GetReferences().ClearReferences()
        root.GetReferences().AddReference(str(asset_path.resolve()))
        xform = UsdGeom.Xformable(root)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*[float(value) for value in NOMINAL_ROOT_XYZ_M])
        )
        xform.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat).Set(
            Gf.Vec3f(0.0, 0.0, float(NOMINAL_ROOT_YAW_DEG))
        )
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(
            Gf.Vec3f(ASSET_SCALE, ASSET_SCALE, ASSET_SCALE)
        )

        required_paths = (
            FRAME_BODY_PATH,
            DOOR_BODY_PATH,
            FRONT_HANDLE_BODY_PATH,
            REAR_HANDLE_BODY_PATH,
        )
        required = [self._stage.GetPrimAtPath(path) for path in required_paths]
        if not all(prim.IsValid() for prim in required):
            raise RuntimeError("P0-02 composed frame, door, or handle prim is missing")

        masses = {
            FRAME_BODY_PATH: 4.0,
            DOOR_BODY_PATH: 2.2,
            FRONT_HANDLE_BODY_PATH: 0.15,
            REAR_HANDLE_BODY_PATH: 0.15,
        }
        for path, mass in masses.items():
            prim = self._stage.GetPrimAtPath(path)
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(float(mass))
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
            rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rigid_api.CreateMaxLinearVelocityAttr(3.0)
            rigid_api.CreateMaxAngularVelocityAttr(12.0)
            if path in GRAVITY_DISABLED_BODY_PATHS:
                # The explicit angular drives are the physical return springs.
                # Keeping gravity on these two small links made both levers
                # sag about 0.013 rad before action 0, so every platform start
                # violated the strict recording-origin drift gate and rebuilt
                # the scene three times before exiting.
                rigid_api.CreateDisableGravityAttr(True)

        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.65,
            dynamic_friction=0.55,
            restitution=0.0,
            friction_combine_mode="average",
            restitution_combine_mode="min",
        )
        material_cfg.func(_MATERIAL_PATH, material_cfg)
        sim_utils.bind_physics_material(
            DOOR_ROOT_PATH,
            _MATERIAL_PATH,
            stage=self._stage,
            stronger_than_descendants=True,
        )

        # P0-02 alone has no front table. Keep the shared MJCF prim only for
        # stage/reset compatibility, but make it invisible and collision-free.
        # Replace only Handle002's visual/collision payload with the selected
        # Helindu asset; keep the existing rigid body, joint, and semantic path.
        self._p002_scene_geometry_report = apply_p002_scene_geometry(self._stage)
        geometry = self._p002_scene_geometry_report
        dimensions_mm = tuple(
            1000.0 * value for value in geometry.front_handle_dimensions_m
        )
        print(
            "[HandsOn/P002] scene geometry ready: "
            f"table_hidden={geometry.table_hidden}, "
            f"table_colliders_disabled={geometry.table_colliders_disabled}, "
            "table_collision_instance_deactivated="
            f"{geometry.table_collision_instance_deactivated}, "
            f"handle_asset={geometry.replacement_asset_title!r}, "
            f"handle_author={geometry.replacement_asset_author!r}, "
            "handle_dimensions_mm="
            f"({dimensions_mm[0]:.3f},{dimensions_mm[1]:.3f},"
            f"{dimensions_mm[2]:.3f}), "
            "grasp_clearance_mm="
            f"{1000.0 * geometry.front_handle_grasp_clearance_m:.3f}, "
            f"replacement_colliders={geometry.replacement_colliders}, "
            "replacement_collision_primitive="
            f"{geometry.replacement_collision_primitive}, "
            "replacement_max_convex_vertices="
            f"{geometry.replacement_max_convex_vertices}",
            flush=True,
        )

        dome = UsdLux.DomeLight.Define(self._stage, f"{_LIGHT_ROOT}/Dome")
        dome.CreateIntensityAttr(650.0)
        dome.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
        key = UsdLux.SphereLight.Define(self._stage, f"{_LIGHT_ROOT}/Key")
        key.CreateIntensityAttr(2200.0)
        key.CreateRadiusAttr(0.18)
        key.AddTranslateOp().Set(Gf.Vec3d(-0.20, -0.25, 2.10))

    def _robot_rigid_body_paths(self, side: str, *, hand_only: bool) -> list[str]:
        if self._stage is None:
            return []
        from pxr import Usd, UsdPhysics

        root = self._stage.GetPrimAtPath(
            f"/World/HandsOnRobots/{side.capitalize()}Robot"
        )
        if not root.IsValid():
            return []
        paths: list[str] = []
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            name = str(prim.GetName()).lower()
            if hand_only and not (
                name == f"{side}_palm_link" or name.startswith(f"{side}_finger")
            ):
                continue
            paths.append(str(prim.GetPath()))
        return paths

    def _create_contact_view(
        self,
        physics_view: Any,
        key: str,
        sensor_path: str,
        filter_paths: list[str],
    ) -> None:
        if not filter_paths:
            raise RuntimeError(f"P0-02 contact filter {key} resolved no bodies")
        view = physics_view.create_rigid_contact_view(
            sensor_path,
            filter_patterns=filter_paths,
            max_contact_data_count=max(128, len(filter_paths) * 16),
        )
        try:
            counts_match = bool(
                view is not None
                and int(view.sensor_count) == 1
                and int(view.filter_count) == len(filter_paths)
            )
        except (AttributeError, RuntimeError):
            counts_match = False
        if not counts_match:
            raise RuntimeError(f"P0-02 contact view {key} is empty or incomplete")
        self._p002_contact_views[key] = view
        self._p002_contact_paths[key] = (sensor_path, tuple(filter_paths))

    def _init_task_specific_runtime(self) -> None:
        if not self._robots:
            return
        physics_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_view is None:
            raise RuntimeError("P0-02 PhysX tensor simulation view is unavailable")
        articulation = physics_view.create_articulation_view(DOOR_ROOT_PATH)
        if int(articulation.count) != 1 or int(articulation.max_dofs) != 3:
            raise RuntimeError("P0-02 requires one three-DOF Door042 articulation")
        metatype = articulation.get_metatype(0)
        dof_indices = {str(name): int(index) for name, index in metatype.dof_indices.items()}
        if set(dof_indices) != set(EXPECTED_DOF_NAMES):
            raise RuntimeError(
                f"P0-02 unexpected Door042 DOFs: {sorted(dof_indices)}"
            )
        self._p002_dof_indices = {
            "rear_handle": dof_indices["Handle001_joint"],
            "front_handle": dof_indices["Handle002_joint"],
            "door": dof_indices["Door_joint"],
        }
        self._p002_articulation_view = articulation
        self._p002_body_views = {
            "frame": physics_view.create_rigid_body_view(FRAME_BODY_PATH),
            "door": physics_view.create_rigid_body_view(DOOR_BODY_PATH),
            "front_handle": physics_view.create_rigid_body_view(FRONT_HANDLE_BODY_PATH),
            "rear_handle": physics_view.create_rigid_body_view(REAR_HANDLE_BODY_PATH),
        }
        if any(int(view.count) != 1 for view in self._p002_body_views.values()):
            raise RuntimeError("P0-02 one or more Door042 rigid-body views are empty")

        left_hand = self._robot_rigid_body_paths("left", hand_only=True)
        right_hand = self._robot_rigid_body_paths("right", hand_only=True)
        all_robot = self._robot_rigid_body_paths("left", hand_only=False)
        all_robot.extend(self._robot_rigid_body_paths("right", hand_only=False))
        self._create_contact_view(
            physics_view, "right_front_handle", FRONT_HANDLE_BODY_PATH, right_hand
        )
        self._create_contact_view(
            physics_view, "left_door", DOOR_BODY_PATH, left_hand
        )
        self._create_contact_view(
            physics_view, "robot_frame", FRAME_BODY_PATH, all_robot
        )
        self._create_contact_view(
            physics_view, "robot_rear_handle", REAR_HANDLE_BODY_PATH, all_robot
        )
        # These two views exist only to qualify the authored/reset origin.
        # Together with the all-robot frame and rear-handle views they cover
        # the complete arm/hand envelope against every Door042 rigid body.
        self._create_contact_view(
            physics_view,
            "initial_robot_front_handle",
            FRONT_HANDLE_BODY_PATH,
            all_robot,
        )
        self._create_contact_view(
            physics_view,
            "initial_robot_door",
            DOOR_BODY_PATH,
            all_robot,
        )
        self._p002_contact_contract_ready = True
        self._set_door_drive_locked(True)
        print(
            "[HandsOn/P002] physical contract ready: "
            f"dofs={dof_indices}, left_hand={len(left_hand)}, "
            f"right_hand={len(right_hand)}",
            flush=True,
        )

    @staticmethod
    def _view_numpy(value: Any, shape: tuple[int, ...]) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64).reshape(shape)

    def _set_door_drive_locked(self, locked: bool) -> None:
        view = self._p002_articulation_view
        if view is None or self._torch is None:
            self._p002_drive_locked = bool(locked)
            return
        stiffness = view.get_dof_stiffnesses().clone()
        damping = view.get_dof_dampings().clone()
        max_force = view.get_dof_max_forces().clone()
        targets = view.get_dof_position_targets().clone()
        rear = self._p002_dof_indices["rear_handle"]
        front = self._p002_dof_indices["front_handle"]
        door = self._p002_dof_indices["door"]
        stiffness[:, rear] = HANDLE_STIFFNESS
        stiffness[:, front] = HANDLE_STIFFNESS
        damping[:, rear] = HANDLE_DAMPING
        damping[:, front] = HANDLE_DAMPING
        max_force[:, rear] = HANDLE_MAX_FORCE
        max_force[:, front] = HANDLE_MAX_FORCE
        targets[:, rear] = 0.0
        targets[:, front] = 0.0
        if locked:
            stiffness[:, door] = LOCKED_DOOR_STIFFNESS
            damping[:, door] = LOCKED_DOOR_DAMPING
            max_force[:, door] = LOCKED_DOOR_MAX_FORCE
            targets[:, door] = 0.0
        else:
            stiffness[:, door] = UNLOCKED_DOOR_STIFFNESS
            damping[:, door] = UNLOCKED_DOOR_DAMPING
            max_force[:, door] = UNLOCKED_DOOR_MAX_FORCE
            targets[:, door] = view.get_dof_positions()[:, door]
        def _indices_like(value: Any):
            return self._torch.arange(
                int(view.count), dtype=self._torch.long, device=value.device
            )

        # Isaac/PhysX may expose authored DOF attributes on CPU while keeping
        # live drive targets on the simulation device.  Each setter requires
        # its index tensor to live beside that setter's value tensor.
        view.set_dof_stiffnesses(stiffness, _indices_like(stiffness))
        view.set_dof_dampings(damping, _indices_like(damping))
        view.set_dof_max_forces(max_force, _indices_like(max_force))
        view.set_dof_position_targets(targets, _indices_like(targets))
        self._p002_drive_locked = bool(locked)

    def reset(self, seed: int | None = None, **kwargs: Any):
        if seed is not None:
            self._p002_reset_serial = -1
        result = super().reset(seed=seed, **kwargs)
        if self._p002_articulation_view is not None:
            self._validate_task_initial_state_before_ready()
        return result

    def accept_stable_recording_origin(
        self,
        report: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Recheck actual robot state at the accepted recording boundary."""
        parent_acceptance = super().accept_stable_recording_origin(report)
        payload = (
            dict(parent_acceptance)
            if isinstance(parent_acceptance, Mapping)
            else {}
        )
        initial_state = self._validate_task_initial_state_before_ready()
        if initial_state is None:
            raise RuntimeError(
                "P0-02 stable origin lacks its actual robot-state views"
            )
        payload.update(
            {
                "accepted": True,
                "kind": "p002_actual_robot_and_task_state_boundary",
                "initial_robot_state": initial_state["robots"],
                "initial_task_state": initial_state["task"],
            }
        )
        return payload

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self.delta_h[:] = 0.0
        self._p002_reset_serial += 1
        episode_seed = int(self._seed) + 104729 * self._p002_reset_serial
        self._p002_reset = sample_reset_state(episode_seed)
        self._p002_latch.reset()
        self._p002_failure_latched = False
        self._p002_last_readback = None
        self._p002_last_metrics = {}
        self._p002_drive_locked = None
        self._p002_first_failure_event = None
        self._p002_terminal_reported = False

    def _apply_task_pose_randomization(self) -> None:
        if self._torch is None:
            return
        entry = next(
            (
                value
                for value in self._task_reset_articulation_states
                if value[0] == DOOR_ROOT_PATH
            ),
            None,
        )
        if entry is None:
            self._add_task_state_warning("P0-02 articulation reset state is unavailable")
            return
        path, view, indices, root_pose, root_velocity, joint_pos, joint_vel = entry
        pose_np = self._view_numpy(root_pose, (int(view.count), 7)).copy()
        composed_pose_xyzw = pose_np[0].copy()
        self._p002_composed_frame_pose_wxyz = np.concatenate(
            [
                composed_pose_xyzw[:3],
                composed_pose_xyzw[6:7],
                composed_pose_xyzw[3:6],
            ]
        )
        pose_np[0] = _randomized_root_pose_xyzw(
            composed_pose_xyzw,
            self._p002_reset,
        )
        pose = self._torch.as_tensor(
            pose_np, dtype=root_pose.dtype, device=root_pose.device
        )
        zero_root_velocity = self._torch.zeros_like(root_velocity)
        positions = self._torch.zeros_like(joint_pos)
        velocities = self._torch.zeros_like(joint_vel)
        view.set_root_transforms(pose, indices)
        view.set_root_velocities(zero_root_velocity, indices)
        view.set_dof_positions(positions, indices)
        view.set_dof_velocities(velocities, indices)
        self._record_authored_task_reset_write(
            path,
            transform_xyzw=pose,
            velocity=zero_root_velocity,
            dof_position=positions,
            dof_velocity=velocities,
        )
        self._p002_reset_frame_pose_wxyz = np.concatenate(
            [pose_np[0, :3], pose_np[0, 6:7], pose_np[0, 3:6]]
        )
        self._set_door_drive_locked(True)
        print(
            "[HandsOn/P002] seeded reset: "
            f"seed={self._p002_reset.seed}, "
            f"xy=({self._p002_reset.root_x_m:+.4f},"
            f"{self._p002_reset.root_y_m:+.4f})m, "
            f"yaw={self._p002_reset.root_yaw_deg:+.2f}deg",
            flush=True,
        )

    def _joint_positions(self) -> np.ndarray:
        view = self._p002_articulation_view
        if view is None:
            return np.zeros(3, dtype=np.float64)
        return self._view_numpy(view.get_dof_positions(), (int(view.count), -1))[0]

    def _joint_velocities(self) -> np.ndarray:
        view = self._p002_articulation_view
        if view is None:
            return np.zeros(3, dtype=np.float64)
        return self._view_numpy(view.get_dof_velocities(), (int(view.count), -1))[0]

    def _contact_reading(self, key: str) -> tuple[float, float]:
        view = self._p002_contact_views.get(key)
        if view is None:
            return 0.0, 0.0
        try:
            normal_forces, _points, _normals, separations, counts, starts = (
                view.get_contact_data(self._sim_dt)
            )
            forces = self._view_numpy(normal_forces, (-1,))
            separations = self._view_numpy(separations, (-1,))
            counts = self._view_numpy(
                counts, (int(view.sensor_count), int(view.filter_count))
            ).astype(np.int64)
            starts = self._view_numpy(starts, counts.shape).astype(np.int64)
            selected_force = 0.0
            max_penetration = 0.0
            for count, start in zip(counts.ravel(), starts.ravel()):
                if count <= 0:
                    continue
                stop = int(start + count)
                selected_force += float(np.sum(np.abs(forces[int(start):stop])))
                max_penetration = max(
                    max_penetration,
                    float(np.max(np.maximum(0.0, -separations[int(start):stop]))),
                )
            return selected_force, max_penetration
        except Exception as exc:
            self._add_task_state_warning(f"P0-02 contact readback {key} failed: {exc}")
            return 0.0, math.inf

    def _physical_readback(self) -> P002PhysicalReadback:
        positions = self._joint_positions()
        velocities = self._joint_velocities()
        readings = {
            key: self._contact_reading(key)
            for key in (
                "right_front_handle",
                "left_door",
                "robot_frame",
                "robot_rear_handle",
            )
        }
        # Preserve the historical readback/observation slot without sensing or
        # gating on the table.  This makes table contact incapable of causing a
        # terminal, success veto, penetration failure, or readback error.
        readings["robot_table"] = (0.0, 0.0)
        return P002PhysicalReadback(
            front_handle_angle_rad=float(
                positions[self._p002_dof_indices["front_handle"]]
            ),
            rear_handle_angle_rad=float(
                positions[self._p002_dof_indices["rear_handle"]]
            ),
            door_angle_rad=float(positions[self._p002_dof_indices["door"]]),
            door_angular_speed_radps=float(
                velocities[self._p002_dof_indices["door"]]
            ),
            # These two dataclass fields are legacy storage slots.  Canonical
            # accessors expose the corrected right-handle/left-door names.
            left_front_handle_contact_force_n=readings["right_front_handle"][0],
            right_door_contact_force_n=readings["left_door"][0],
            robot_table_contact_force_n=0.0,
            robot_frame_contact_force_n=readings["robot_frame"][0],
            robot_rear_handle_contact_force_n=readings["robot_rear_handle"][0],
            max_penetration_m=max(value[1] for value in readings.values()),
            hard_failure_max_penetration_m=max(
                readings[key][1] for key in _P002_HARD_FAILURE_PENETRATION_KEYS
            ),
        )

    def _first_failure_snapshot(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Freeze the first hard-failure truth before later substeps overwrite it."""

        readback = getattr(self, "_p002_last_readback", None)
        reasons = tuple(str(value) for value in metrics.get("failure_reasons", ()))
        if not reasons and readback is not None:
            reasons = physical_failure_reasons(readback)
        if not reasons:
            reasons = ("unclassified_physical_failure",)
        action_phase = bool(
            getattr(self, "_trajectory_action_phase_active", False)
        )
        substep_index = int(
            getattr(self, "_trajectory_action_phase_substep_index", -1)
        )
        event = {
            "event": "p002_first_terminal",
            "reason": reasons[0],
            "failure_reasons": list(reasons),
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "failure_observed_env_step": int(getattr(self, "env_step", 0)),
            "failure_observed_during_action": action_phase,
            "physics_substep_index": substep_index if action_phase else None,
            "arming_stage": (
                "action_physics_substep" if action_phase else "pre_action_physics"
            ),
            "finite": bool(metrics.get("finite", False)),
            "latch_phase": str(metrics.get("phase", "unknown")),
            "latch_phase_code": int(metrics.get("phase_code", -1)),
            "latch_locked": bool(metrics.get("locked", True)),
            "latch_press_steps": int(metrics.get("press_steps", 0)),
            "right_handle_contact_seen": bool(
                metrics.get("right_handle_contact_seen", False)
            ),
            "left_door_contact_after_unlock_seen": bool(
                metrics.get("left_door_contact_after_unlock_seen", False)
            ),
            # Compatibility aliases retained for existing terminal parsers.
            "left_handle_contact_seen": bool(
                metrics.get("right_handle_contact_seen", False)
            ),
            "right_door_contact_after_unlock_seen": bool(
                metrics.get("left_door_contact_after_unlock_seen", False)
            ),
            "unlocked_seen": bool(metrics.get("unlocked_seen", False)),
            "door_escape_seen": bool(metrics.get("door_escape_seen", False)),
            "front_handle_angle_deg": _json_safe_number(
                metrics.get("front_handle_angle_deg")
            ),
            "rear_handle_angle_deg": _json_safe_number(
                None
                if readback is None
                else math.degrees(float(readback.rear_handle_angle_rad))
            ),
            "door_angle_deg": _json_safe_number(metrics.get("door_angle_deg")),
            "door_angular_speed_radps": _json_safe_number(
                None if readback is None else readback.door_angular_speed_radps
            ),
            "right_front_handle_contact_force_n": _json_safe_number(
                None
                if readback is None
                else readback.right_front_handle_contact_force_n
            ),
            "left_door_contact_force_n": _json_safe_number(
                None if readback is None else readback.left_door_contact_force_n
            ),
            # Compatibility aliases retain their historical JSON keys while
            # carrying the corrected role-slot values.
            "left_front_handle_contact_force_n": _json_safe_number(
                None
                if readback is None
                else readback.right_front_handle_contact_force_n
            ),
            "right_door_contact_force_n": _json_safe_number(
                None if readback is None else readback.left_door_contact_force_n
            ),
            "robot_table_contact_force_n": _json_safe_number(
                None if readback is None else readback.robot_table_contact_force_n
            ),
            "robot_frame_contact_force_n": _json_safe_number(
                None if readback is None else readback.robot_frame_contact_force_n
            ),
            "robot_rear_handle_contact_force_n": _json_safe_number(
                None
                if readback is None
                else readback.robot_rear_handle_contact_force_n
            ),
            "max_penetration_m": _json_safe_number(
                None if readback is None else readback.max_penetration_m
            ),
            "hard_failure_max_penetration_m": _json_safe_number(
                None
                if readback is None
                else readback.hard_failure_max_penetration_m
            ),
            "forbidden_contact_force_threshold_n": FORBIDDEN_CONTACT_FORCE_N,
            "rear_handle_contact_force_threshold_n": FORBIDDEN_CONTACT_FORCE_N,
            "max_penetration_threshold_m": MAX_PENETRATION_M,
            "door_speed_explosion_threshold_radps": (
                EXPLOSION_ANGULAR_SPEED_RADPS
            ),
        }
        event["nonfinite_fields"] = sorted(
            key
            for key, value in event.items()
            if isinstance(value, str) and value in {"nan", "inf", "-inf"}
        )
        return event

    def _refresh_metrics(self, *, update_latch: bool) -> dict[str, Any]:
        self._p002_last_readback = self._physical_readback()
        previous_locked = self._p002_latch.locked
        if update_latch:
            self._p002_latch.update(self._p002_last_readback)
        if self._p002_latch.locked != previous_locked:
            self._set_door_drive_locked(self._p002_latch.locked)
        self._p002_last_metrics = evaluate_physical_readback(
            self._p002_last_readback, self._p002_latch
        )
        failed_now = bool(self._p002_last_metrics["failed"])
        if failed_now and not self._p002_failure_latched:
            self._p002_first_failure_event = self._first_failure_snapshot(
                self._p002_last_metrics
            )
        self._p002_failure_latched |= failed_now
        return self._p002_last_metrics

    def _after_task_physics_substep(self) -> None:
        # High-rate readback latches safety failures, while the five-step
        # handle dwell remains a control-rate state-machine condition.
        if self._p002_contact_contract_ready:
            self._refresh_metrics(update_latch=False)

    def _compute_success_now(self) -> bool:
        metrics = self._refresh_metrics(update_latch=True)
        return bool(metrics["success_now"] and not self._p002_failure_latched)

    def _compute_failure(self) -> bool:
        return bool(self._p002_failure_latched)

    def step_with_current_joint_targets(self, *, start_time: float | None = None):
        """Expose the first physical failure or time limit before reset."""

        observation, reward, terminated, truncated, info = (
            super().step_with_current_joint_targets(start_time=start_time)
        )
        if info.get("failed", False):
            info = dict(info)
            first_failure = getattr(self, "_p002_first_failure_event", None)
            if first_failure is None:
                first_failure = self._first_failure_snapshot(
                    getattr(self, "_p002_last_metrics", {})
                )
            event = dict(first_failure)
            reset_state = getattr(self, "_p002_reset", None)
            reset_seed = getattr(reset_state, "seed", None)
            event.update(
                {
                    "elapsed_sim_time_seconds": float(
                        self.env_step * self._control_dt
                    ),
                    "env_step": int(self.env_step),
                    "reset_seed": (
                        int(reset_seed) if reset_seed is not None else None
                    ),
                    "succeed": bool(info.get("succeed", False)),
                    "failed": True,
                    "manual_reset": bool(info.get("manual_reset", False)),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            reason = str(event["reason"])
            reasons = list(event["failure_reasons"])
            info["failure_reason"] = reason
            info["p002_failure_reason"] = reason
            info["p002_failure_reasons"] = reasons
            info["p002_terminal_reason"] = reason
            info["p002_first_failure"] = dict(event)
            if not self._p002_terminal_reported:
                print(
                    "[HandsOn/P002 terminal] "
                    + json.dumps(
                        event,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                self._p002_terminal_reported = True
        timed_out = bool(
            self.env_step * self._control_dt >= self._time_limit
            and not info.get("succeed", False)
            and not info.get("failed", False)
            and not info.get("manual_reset", False)
        )
        if timed_out:
            terminated = False
            truncated = True
            info = dict(info)
            info["time_limit_reached"] = True
            info["truncated_reason"] = "time_limit"
            info["p002_terminal_reason"] = "time_limit"
            if not self._p002_terminal_reported:
                metrics = dict(getattr(self, "_p002_last_metrics", {}))
                readback = getattr(self, "_p002_last_readback", None)
                event = {
                    "elapsed_sim_time_seconds": float(
                        self.env_step * self._control_dt
                    ),
                    "env_step": int(self.env_step),
                    "event": "p002_first_terminal",
                    "front_handle_angle_deg": _json_safe_number(
                        metrics.get("front_handle_angle_deg")
                    ),
                    "handle_press_threshold_deg": HANDLE_PRESS_DEG,
                    "handle_release_threshold_deg": HANDLE_RELEASE_DEG,
                    "latch_locked": bool(metrics.get("locked", True)),
                    "latch_phase": str(metrics.get("phase", "unknown")),
                    "latch_press_steps": int(metrics.get("press_steps", 0)),
                    "door_angle_deg": _json_safe_number(
                        metrics.get("door_angle_deg")
                    ),
                    "door_escape_seen": bool(
                        metrics.get("door_escape_seen", False)
                    ),
                    "p002_drive_locked": bool(
                        getattr(self, "_p002_drive_locked", True)
                    ),
                    "reason": "time_limit",
                    "right_front_handle_contact_force_n": _json_safe_number(
                        None
                        if readback is None
                        else readback.right_front_handle_contact_force_n
                    ),
                    "terminated": False,
                    "time_limit_seconds": float(self._time_limit),
                    "timeout_steps": int(
                        round(self._time_limit / self._control_dt)
                    ),
                    "truncated": True,
                    "unlocked_seen": bool(
                        metrics.get("unlocked_seen", False)
                    ),
                }
                print(
                    "[HandsOn/P002 terminal] "
                    + json.dumps(
                        event,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                self._p002_terminal_reported = True
        return observation, reward, terminated, truncated, info

    def _success_steps_required(self) -> int:
        return SUCCESS_HOLD_STEPS

    def _task_info(self) -> dict[str, Any]:
        metrics = dict(self._p002_last_metrics)
        geometry = self._p002_scene_geometry_report
        metrics.update(
            {
                "p002_task_id": "P0-02",
                "p002_success_counter": int(self._success_counter),
                "p002_success_steps_required": SUCCESS_HOLD_STEPS,
                "p002_reset_seed": int(self._p002_reset.seed),
                "p002_drive_locked": bool(self._p002_drive_locked),
                "p002_role_contract_version": P002_ROLE_CONTRACT_VERSION,
                "p002_role_contacts_layout": (
                    "right_front_handle,left_door,legacy_table_zero,"
                    "diagnostic_frame,rear_handle"
                ),
                "p002_latch_state_layout": (
                    "locked,unlocked_seen,door_escape_seen,right_handle_seen,"
                    "left_door_after_unlock_seen,target_ok,door_stable,"
                    "forbidden_clear"
                ),
                "p002_front_table_present": False,
                "p002_front_table_colliders_enabled": False,
                "p002_front_table_collision_instance_deactivated": (
                    None
                    if geometry is None
                    else geometry.table_collision_instance_deactivated
                ),
                "p002_front_handle_replacement_asset": (
                    None if geometry is None else geometry.replacement_asset_title
                ),
                "p002_front_handle_replacement_author": (
                    None if geometry is None else geometry.replacement_asset_author
                ),
                "p002_front_handle_source_archive_sha256": (
                    None
                    if geometry is None
                    else geometry.replacement_source_archive_sha256
                ),
                "p002_front_handle_dimensions_m": (
                    None
                    if geometry is None
                    else list(geometry.front_handle_dimensions_m)
                ),
                "p002_front_handle_grasp_clearance_m": (
                    None
                    if geometry is None
                    else geometry.front_handle_grasp_clearance_m
                ),
                "p002_front_handle_min_clearance_m": (
                    None
                    if geometry is None
                    else geometry.front_handle_min_clearance_m
                ),
                "p002_front_handle_max_convex_vertices": (
                    None
                    if geometry is None
                    else geometry.replacement_max_convex_vertices
                ),
                "p002_front_handle_collision_primitive": (
                    None
                    if geometry is None
                    else geometry.replacement_collision_primitive
                ),
                "p002_initial_robot_state_valid": bool(
                    getattr(self, "_p002_initial_robot_state_report", None)
                    and all(
                        row.get("passed", False)
                        for row in self._p002_initial_robot_state_report.values()
                    )
                ),
                "p002_initial_task_state_valid": bool(
                    getattr(self, "_p002_initial_task_state_report", None)
                    and self._p002_initial_task_state_report.get("passed", False)
                ),
            }
        )
        return metrics

    def _task_status_text(self) -> str:
        metrics = self._p002_last_metrics
        return (
            f"Latch {str(metrics.get('phase', 'locked'))}  |  "
            f"Right lever {float(metrics.get('front_handle_angle_deg', 0.0)):.1f} deg"
            f"  |  Left door {float(metrics.get('door_angle_deg', 0.0)):.1f} deg"
        )

    @staticmethod
    def _success_condition_display_label(row: dict[str, Any]) -> str:
        key = str(row.get("key") or "")
        if key in _P002_SUCCESS_CONDITION_DISPLAY_LABELS:
            return _P002_SUCCESS_CONDITION_DISPLAY_LABELS[key]
        return HandsOnABCIsaacLabEnv._success_condition_display_label(row)

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._p002_last_metrics
        return self._append_success_hold_row(
            [
                self._condition_row(
                    "p002_right_handle",
                    _P002_SUCCESS_CONDITION_DISPLAY_LABELS["p002_right_handle"],
                    bool(metrics.get("right_handle_contact_seen", False)),
                ),
                self._condition_row(
                    "p002_unlatched",
                    _P002_SUCCESS_CONDITION_DISPLAY_LABELS["p002_unlatched"],
                    bool(metrics.get("unlocked_seen", False)),
                ),
                self._condition_row(
                    "p002_left_door",
                    _P002_SUCCESS_CONDITION_DISPLAY_LABELS["p002_left_door"],
                    bool(metrics.get("left_door_contact_after_unlock_seen", False)),
                ),
                self._condition_row(
                    "p002_target",
                    _P002_SUCCESS_CONDITION_DISPLAY_LABELS["p002_target"],
                    bool(metrics.get("target_ok", False)),
                ),
                self._condition_row(
                    "p002_stable",
                    _P002_SUCCESS_CONDITION_DISPLAY_LABELS["p002_stable"],
                    bool(metrics.get("door_stable", False)),
                ),
                self._condition_row(
                    "p002_forbidden",
                    _P002_SUCCESS_CONDITION_DISPLAY_LABELS["p002_forbidden"],
                    bool(metrics.get("forbidden_clear", False))
                    and bool(metrics.get("penetration_ok", False)),
                ),
            ]
        )

    def _body_pose_wxyz(self, key: str) -> np.ndarray:
        view = self._p002_body_views.get(key)
        if view is None:
            return np.zeros(7, dtype=np.float64)
        transform = self._view_numpy(view.get_transforms()[0], (7,))
        return np.concatenate([transform[:3], transform[6:7], transform[3:6]])

    def _compute_observation(self) -> dict[str, Any]:
        observation = super()._compute_observation()
        if self._p002_last_readback is None and self._p002_articulation_view is not None:
            self._refresh_metrics(update_latch=False)
        readback = self._p002_last_readback or P002PhysicalReadback(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        metrics = self._p002_last_metrics
        state = observation["state"]
        state.update(
            {
                "front_handle_pose": self._body_pose_wxyz("front_handle"),
                "door_panel_pose": self._body_pose_wxyz("door"),
                "p002_joint_state": np.asarray(
                    [
                        readback.front_handle_angle_rad,
                        readback.rear_handle_angle_rad,
                        readback.door_angle_rad,
                        readback.door_angular_speed_radps,
                        float(self._p002_latch.press_steps),
                        float(self._p002_latch.phase),
                    ],
                    dtype=np.float64,
                ),
                "p002_role_contacts": np.asarray(
                    [
                        # Keep the historical five-value shape; slots 0/1 now
                        # represent right-handle and left-door respectively.
                        readback.right_front_handle_contact_force_n,
                        readback.left_door_contact_force_n,
                        readback.robot_table_contact_force_n,
                        readback.robot_frame_contact_force_n,
                        readback.robot_rear_handle_contact_force_n,
                    ],
                    dtype=np.float64,
                ),
                "p002_latch_state": np.asarray(
                    [
                        float(self._p002_latch.locked),
                        float(self._p002_latch.unlocked_seen),
                        float(self._p002_latch.door_escape_seen),
                        float(self._p002_latch.right_handle_contact_seen),
                        float(self._p002_latch.left_door_contact_after_unlock_seen),
                        float(metrics.get("target_ok", False)),
                        float(metrics.get("door_stable", False)),
                        float(metrics.get("forbidden_clear", True)),
                    ],
                    dtype=np.float64,
                ),
                "p002_role_contract_version": np.asarray(
                    [P002_ROLE_CONTRACT_VERSION], dtype=np.float64
                ),
            }
        )
        return observation

    def restore_recorded_initial_state(
        self, state_parts: Mapping[str, np.ndarray]
    ) -> dict[str, Any]:
        self._p002_failure_latched = False
        self._p002_first_failure_event = None
        self._p002_terminal_reported = False
        observation = super().restore_recorded_initial_state(state_parts)
        joint_state = np.asarray(
            state_parts.get("p002_joint_state", np.zeros(0)), dtype=np.float64
        ).ravel()
        view = self._p002_articulation_view
        if view is not None and self._torch is not None and joint_state.size >= 3:
            positions = view.get_dof_positions().clone()
            velocities = self._torch.zeros_like(view.get_dof_velocities())
            positions[:, self._p002_dof_indices["front_handle"]] = joint_state[0]
            positions[:, self._p002_dof_indices["rear_handle"]] = joint_state[1]
            positions[:, self._p002_dof_indices["door"]] = joint_state[2]
            if joint_state.size >= 4:
                velocities[:, self._p002_dof_indices["door"]] = joint_state[3]
            indices = self._torch.arange(
                int(view.count), dtype=self._torch.long, device=positions.device
            )
            view.set_dof_positions(positions, indices)
            view.set_dof_velocities(velocities, indices)
            self._p002_latch.reset()
            if joint_state.size >= 6 and int(round(float(joint_state[5]))) >= 2:
                self._p002_latch.unlocked_seen = True
                self._p002_latch.phase = type(self._p002_latch.phase)(
                    int(round(float(joint_state[5])))
                )
            self._set_door_drive_locked(self._p002_latch.locked)
            if self._sim is not None:
                self._sim.forward()
            self._refresh_metrics(update_latch=False)
            observation = self._compute_observation()
        return observation


__all__ = ["HandsOnP002IsaacLabEnv"]
