"""Production Isaac Sim 5.1/PhysX vertical slice for B320-HS014."""

from __future__ import annotations

import importlib.util
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from .abc_collection_env import (
    HandsOnABCIsaacLabEnv,
    _TASK_ROOT,
    _TABLE_TOP_Z,
    _invert_pose_matrix,
    _matrix_to_quat_wxyz,
    _pose_vector_to_matrix,
    _quat_wxyz_to_matrix,
)
from .bimanual_assembly_env import _torch_view
from .priority_zero_robot_audit import initial_robot_state_report, pose_error_wxyz


TASK_NAME = "bench320_hs014_bottle_stabilized_cap_opening"
TASK_ASSET_ROOT = f"{_TASK_ROOT}/HS014"
BOTTLE_NAME = "hs014_water_bottle"
CAP_NAME = "hs014_bottle_cap"
BOTTLE_PRIM_PATH = f"{TASK_ASSET_ROOT}/WaterBottle"
CAP_PRIM_PATH = f"{TASK_ASSET_ROOT}/BottleCap"

BOTTLE_NOMINAL_XY = np.asarray([-0.4225, -0.1225], dtype=np.float64)
CAP_NOMINAL_XY = BOTTLE_NOMINAL_XY.copy()
BOTTLE_ROOT_Z = _TABLE_TOP_Z + 0.2002
CAP_SEATED_HEIGHT_M = -0.0002
CAP_ROOT_Z = BOTTLE_ROOT_Z + CAP_SEATED_HEIGHT_M
BOTTLE_XY_JITTER = 0.015
BOTTLE_YAW_DEG = 8.0
CAP_RELATIVE_YAW_DEG = 12.0

# The hash-bound source USD is Z-up as a stage, but the bottle mesh itself is
# authored along local +Y from base at Y=0 to open mouth at Y=0.2 m.  The task
# keeps its bottle root at the mouth, so rotate +Y to local +Z and translate
# the source down 0.2 m.  This is the same source-axis contract used by KA015.
BOTTLE_SOURCE_LOCAL_AABB_M = np.asarray(
    [[-0.028000, 0.0, -0.028000], [0.028000, 0.200000, 0.028000]],
    dtype=np.float64,
)
BOTTLE_VISUAL_CORRECTION_WXYZ = np.asarray(
    [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0], dtype=np.float64
)
BOTTLE_VISUAL_TRANSLATION_M = np.asarray([0.0, 0.0, -0.2], dtype=np.float64)
BOTTLE_CENTER_OF_MASS_LOCAL_Z_M = -0.10
CAP_MASS_KG = 0.009
CAP_CENTER_OF_MASS_LOCAL_M = np.asarray([0.0, 0.009, 0.0], dtype=np.float64)
CAP_DIAGONAL_INERTIA_KG_M2 = np.asarray(
    [2.1e-6, 2.8e-6, 2.1e-6], dtype=np.float64
)
CAP_PRINCIPAL_AXES_WXYZ = np.asarray(
    [1.0, 0.0, 0.0, 0.0], dtype=np.float64
)
CAP_CRADLE_RADIUS_M = 0.013
CAP_CRADLE_CENTER_Y_M = 0.009
BOTTLE_CRADLE_SUPPORT_RADIUS_M = 0.0015
BOTTLE_CRADLE_SUPPORT_RING_RADIUS_M = 0.0024
BOTTLE_CRADLE_SUPPORT_CENTER_Z_M = -0.0055
BOTTLE_CRADLE_SUPPORT_ANGLES_DEG = (0.0, 120.0, 240.0)
BOTTLE_CRADLE_SUPPORT_CENTERS_M = np.asarray(
    [
        [
            BOTTLE_CRADLE_SUPPORT_RING_RADIUS_M
            * math.cos(math.radians(angle_deg)),
            BOTTLE_CRADLE_SUPPORT_RING_RADIUS_M
            * math.sin(math.radians(angle_deg)),
            BOTTLE_CRADLE_SUPPORT_CENTER_Z_M,
        ]
        for angle_deg in BOTTLE_CRADLE_SUPPORT_ANGLES_DEG
    ],
    dtype=np.float64,
)
BOTTLE_CRADLE_SUPPORT_RADIAL_ENVELOPE_M = (
    BOTTLE_CRADLE_SUPPORT_RING_RADIUS_M + BOTTLE_CRADLE_SUPPORT_RADIUS_M
)
BOTTLE_MOUTH_CONTAINMENT_RADIUS_M = 0.008
CRADLE_HELPER_CONTACT_OFFSET_M = 0.00005
CRADLE_HELPER_REST_OFFSET_M = 0.0
CRADLE_HELPER_STATIC_FRICTION = 0.0
CRADLE_HELPER_DYNAMIC_FRICTION = 0.0
CRADLE_HELPER_RESTITUTION = 0.0
CRADLE_HELPER_FRICTION_COMBINE_MODE = "min"
CRADLE_CENTER_DZ_M = (
    CAP_SEATED_HEIGHT_M
    + CAP_CRADLE_CENTER_Y_M
    - BOTTLE_CRADLE_SUPPORT_CENTER_Z_M
)
CRADLE_CENTER_DISTANCE_M = math.hypot(
    BOTTLE_CRADLE_SUPPORT_RING_RADIUS_M, CRADLE_CENTER_DZ_M
)
CRADLE_RADIUS_SUM_M = CAP_CRADLE_RADIUS_M + BOTTLE_CRADLE_SUPPORT_RADIUS_M
# Native 5.1 seeded-reset tests: 16/8 leaves persistent bottle/cradle jitter.
# More TGS position steps and one velocity iteration resolve this small contact
# island. Keep the shared scene/robot caps unchanged (64/8).
RIGID_SOLVER_POSITION_ITERATIONS = 64
RIGID_SOLVER_VELOCITY_ITERATIONS = 1
RIGID_MAX_DEPENETRATION_VELOCITY_M_S = 1.0
CAP_SOLVER_POSITION_ITERATIONS = 64
CAP_SOLVER_VELOCITY_ITERATIONS = 1
SCENE_MAX_VELOCITY_ITERATIONS = 8
BOTTLE_LINEAR_DAMPING = 2.0
BOTTLE_ANGULAR_DAMPING = 50.0
BOTTLE_SLEEP_THRESHOLD = 0.0
CAP_LINEAR_DAMPING = 2.0
CAP_ANGULAR_DAMPING = 20.0
CAP_SLEEP_THRESHOLD = 0.0
CAP_SEATED_AXIAL_MIN_M = -0.0005
CAP_SEATED_AXIAL_MAX_M = 0.0015
CAP_SEATED_RADIAL_MAX_M = 0.002
CAP_SEATED_AXIS_ERROR_MAX_DEG = 2.0
INITIAL_HAND_OBJECT_FORCE_MAX_N = 0.01
RESET_CONTACT_CLEARANCE_M = 0.030

THREAD_PITCH_M = 0.00318
ROTATION_TARGET_RAD = math.pi
AXIAL_SEPARATION_TARGET_M = 0.040
BOTTLE_DISPLACEMENT_LIMIT_M = 0.020
BOTTLE_ORIENTATION_LIMIT_DEG = 5.0
BOTTLE_YAW_DRIFT_LIMIT_DEG = 8.0
SUCCESS_HOLD_STEPS = 15
CONTACT_FORCE_EPSILON = 1.0e-6
HAND_CLOSE_COMMAND = 0.45
ROLE_CONTACT_HOLD_SUBSTEPS = 4
MAX_TRACKED_TWIST_DELTA_RAD = math.radians(45.0)

_HS014_ASSET_SUPPORT_MODULE_NAME = "_handson_hs014_taskdev_asset_support"
_HS014_ASSET_SUPPORT_PATH = (
    Path(__file__).resolve().parents[4]
    / "taskdev"
    / "tasks"
    / TASK_NAME
    / "core_scene.py"
)


def _load_hs014_asset_support() -> Any:
    """Load the repository-owned asset composer without widening ``sys.path``.

    The production recorder intentionally exposes only ``<repo>/handson`` on
    ``PYTHONPATH``.  HS014's large, hash-bound source assets still live in its
    task-development directory, so resolve that one module by its canonical
    repository path instead of relying on an accidental import of ``taskdev``.
    """

    cached = sys.modules.get(_HS014_ASSET_SUPPORT_MODULE_NAME)
    if cached is not None:
        return cached
    if not _HS014_ASSET_SUPPORT_PATH.is_file():
        raise RuntimeError(
            "HS014 asset support is missing from the canonical repository: "
            f"{_HS014_ASSET_SUPPORT_PATH}"
        )
    spec = importlib.util.spec_from_file_location(
        _HS014_ASSET_SUPPORT_MODULE_NAME,
        _HS014_ASSET_SUPPORT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not create the HS014 asset-support loader: {_HS014_ASSET_SUPPORT_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_HS014_ASSET_SUPPORT_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_HS014_ASSET_SUPPORT_MODULE_NAME, None)
        raise
    required = (
        "BOTTLE_ASSET",
        "compose_assets",
        "decode_authoritative_cap",
        "source_asset_contract",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        sys.modules.pop(_HS014_ASSET_SUPPORT_MODULE_NAME, None)
        raise RuntimeError(f"HS014 asset support is incomplete: {missing}")
    return module


def _yaw_quaternion_wxyz(degrees: float) -> np.ndarray:
    angle = math.radians(float(degrees))
    return np.asarray(
        [math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle)],
        dtype=np.float64,
    )


def _quat_mul_wxyz(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(first, dtype=np.float64).reshape(4)
    bw, bx, by, bz = np.asarray(second, dtype=np.float64).reshape(4)
    value = np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )
    return value / max(float(np.linalg.norm(value)), 1.0e-12)


def _cap_axial_position_m(bottle: np.ndarray, cap: np.ndarray) -> float:
    bottle_pose = np.asarray(bottle, dtype=np.float64).reshape(7)
    cap_pose = np.asarray(cap, dtype=np.float64).reshape(7)
    bottle_axis = _quat_wxyz_to_matrix(bottle_pose[3:7])[:, 2]
    return float(np.dot(cap_pose[:3] - bottle_pose[:3], bottle_axis))


def sample_reset_layout(
    random_state: np.random.RandomState,
    *,
    randomize: bool,
    table_delta_height: float,
) -> dict[str, np.ndarray]:
    """Return a bounded, deterministic and support-safe reset transaction."""

    bottle_xy = BOTTLE_NOMINAL_XY.copy()
    bottle_yaw = 0.0
    cap_relative_yaw = 0.0
    if randomize:
        bottle_xy += random_state.uniform(
            -BOTTLE_XY_JITTER, BOTTLE_XY_JITTER, size=2
        )
        bottle_yaw = float(random_state.uniform(-BOTTLE_YAW_DEG, BOTTLE_YAW_DEG))
        cap_relative_yaw = float(
            random_state.uniform(-CAP_RELATIVE_YAW_DEG, CAP_RELATIVE_YAW_DEG)
        )

    bottle_pose = np.asarray(
        [
            bottle_xy[0],
            bottle_xy[1],
            BOTTLE_ROOT_Z + float(table_delta_height),
            *_yaw_quaternion_wxyz(bottle_yaw),
        ],
        dtype=np.float64,
    )
    cap_upright = np.asarray(
        [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0],
        dtype=np.float64,
    )
    cap_pose = np.asarray(
        [
            bottle_xy[0],
            bottle_xy[1],
            CAP_ROOT_Z + float(table_delta_height),
            *_quat_mul_wxyz(
                _yaw_quaternion_wxyz(bottle_yaw + cap_relative_yaw), cap_upright
            ),
        ],
        dtype=np.float64,
    )
    return {BOTTLE_NAME: bottle_pose, CAP_NAME: cap_pose}


class HS014IsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Bimanual free-cap opening with continuous physical rotation truth."""

    def _init_haorui_rokae_ik(self) -> None:
        """Load bounded multi-start IK for the right arm only.

        The left arm uses continuous Isaac Lab DLS until both hands have made
        physical role contact. It then switches to the repository's absolute,
        limit-aware solver so external twist load cannot make a relative DLS
        target drift. Deferring the switch until the free cap is retained
        prevents the small solver handoff transient from dislodging it. The
        right wrist uses the bounded solver throughout the long axial twist.
        """

        self._arm_ik_backend = "haorui"
        super()._init_haorui_rokae_ik()
        self._arm_ik_backend = "dls"
        if not self._haorui_ik_enabled:
            raise RuntimeError("HS014 requires the repository Haorui ROKAE IK backend")
        print(
            "[HandsOn/IsaacLab] HS014 arm IK: right=haorui_new "
            "left=isaaclab_dls_then_haorui_new",
            flush=True,
        )

    def _compute_arm_joint_target(
        self, side: str, robot: Any, pose: np.ndarray, commanded: bool
    ) -> tuple[Any | None, str]:
        events = getattr(self, "_hs014_events", {})
        bimanual_roles_held = bool(
            events.get("left_fixture_contact", False)
            and events.get("right_cap_contact", False)
        )
        if side == "left" and not bimanual_roles_held:
            return super()._compute_arm_joint_target(side, robot, pose, commanded)
        if not commanded:
            return None, "idle"
        filtered_pose = self._limit_arm_pose_target(side, robot, pose)
        target = self._compute_hs014_arm_joint_target_haorui(
            side, robot, filtered_pose
        )
        if target is None:
            return None, f"hs014_{side}_haorui_hold"
        return (
            self._limit_arm_joint_target(side, robot, target),
            f"hs014_{side}_haorui",
        )

    def _compute_hs014_arm_joint_target_haorui(
        self, side: str, robot: Any, pose: np.ndarray
    ) -> Any | None:
        """Choose the most continuous bounded IK solution across all seeds.

        The solver reference and the TCP-to-end-effector calibration have
        different time semantics.  The former follows the previous commanded
        joint target for branch continuity, while the latter must pair the
        measured joints with the measured end-effector pose.  Mixing a target
        configuration with a measured pose creates a load-dependent transform
        error that feeds back during the long cap twist.
        """

        try:
            q_reference = self._arm_reference_q(side, robot)
            if q_reference is None:
                return None
            joint_pos = _torch_view(robot.data.joint_pos)
            measured_q = (
                joint_pos[:, self._arm_joint_ids[side]]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=True)
                .reshape(-1)
            )
            if measured_q.shape != (7,) or not np.all(np.isfinite(measured_q)):
                return None
            base_from_world = _invert_pose_matrix(
                self._robot_base_pose_matrix(side, robot)
            )
            current_ee_in_base = base_from_world @ self._robot_body_pose_matrix(
                robot, self._ee_body_indices[side]
            )
            target_ee_in_base = base_from_world @ _pose_vector_to_matrix(pose)
            model = self._haorui_ik_models[side]
            current_tcp_in_base = model.fk(measured_q)
            tcp_to_ee = _invert_pose_matrix(current_tcp_in_base) @ current_ee_in_base
            target_tcp_in_base = target_ee_in_base @ _invert_pose_matrix(tcp_to_ee)
            if not np.all(np.isfinite(target_tcp_in_base)):
                return None
            solution = self._haorui_ik_solvers[side].solve_multi_start(
                target_tcp_in_base,
                self._haorui_rokae.make_seeds(side, q_reference),
                first_success=False,
                reference_q=q_reference,
                max_iters=180,
            )
            if not solution.success:
                return None
            q = np.asarray(solution.q, dtype=np.float64).reshape(1, -1)
            if q.shape[1] != 7 or not np.all(np.isfinite(q)):
                return None
            return self._torch.tensor(
                q,
                dtype=joint_pos.dtype,
                device=joint_pos.device,
            )
        except Exception as exc:
            message = f"{side} HS014 continuous Haorui IK failed; hold reason={exc}"
            if message not in self._isaac_warnings:
                self._isaac_warnings.append(message)
                print(f"[HandsOn/IsaacLab] {message}", flush=True)
            return None

    def __init__(self, **kwargs: Any) -> None:
        # ABC stores task_name in its reconstruction kwargs. Consume it before
        # forwarding the canonical positional name during a hard PhysX reset.
        task_name = kwargs.pop("task_name", TASK_NAME)
        if task_name != TASK_NAME:
            raise ValueError(f"HS014 cannot reconstruct task {task_name!r}")
        physics_backend = str(kwargs.get("physics_backend", "physx")).lower()
        if physics_backend != "physx":
            raise ValueError("HS014 is PhysX-only and forbids Newton.")
        # This task's strict constructor gate intentionally raises before the
        # ready marker when source-derived geometry is invalid.  Use the same
        # owned-process fast shutdown as other B320 validators so a rejected
        # seed cannot hang inside SimulationContext.stop().
        self._fast_shutdown_on_close = True
        self._hs014_composition: dict[str, Any] = {}
        self._hs014_reset_layout: dict[str, np.ndarray] = {}
        self._hs014_bottle_baseline = np.zeros(7, dtype=np.float64)
        self._hs014_authored_bottle_baseline = np.zeros(7, dtype=np.float64)
        self._hs014_reset_axial_position_m = CAP_SEATED_HEIGHT_M
        self._hs014_contact_views: dict[str, Any] = {}
        self._hs014_contact_diagnostics: dict[str, dict[str, Any]] = {}
        self._hs014_contact_forces = {
            "left_bottle": 0.0,
            "left_cap": 0.0,
            "right_bottle": 0.0,
            "right_cap": 0.0,
            "cap_bottle": 0.0,
            "bottle_support": 0.0,
        }
        self._hs014_joint_prims: dict[str, Any] = {}
        self._hs014_joint_enabled = {"left_fixture": False, "right_cap": False}
        # Keep the cap seated in an engaged revolute pair until the measured
        # 180-degree opening twist is complete; incidental hand contact cannot
        # translate a free rigid cap off the neck.
        self._hs014_thread_joint: Any | None = None
        self._hs014_thread_joint_enabled = False
        self._hs014_contact_streaks = {"left_fixture": 0, "right_cap": 0}
        self._hs014_events: dict[str, bool] = {}
        self._hs014_failure_latched = False
        self._hs014_failure_reason = ""
        self._hs014_last_metrics: dict[str, Any] = {}
        self._hs014_stage = 0
        self._hs014_joint_object_to_palm: dict[str, np.ndarray] = {}
        self._hs014_source_contract: dict[str, Any] = {}
        self._hs014_previous_twist_reference: np.ndarray | None = None
        self._hs014_accumulated_twist_rad = 0.0
        self._hs014_initial_state_report: dict[str, Any] = {}
        self._hs014_runtime_physics_schema: dict[str, Any] = {}
        self._hs014_reset_contact_refresh_performed = False
        super().__init__(TASK_NAME, **kwargs)

    def _ready_state_settle_physics_steps(self) -> int:
        """Settle the source-derived bottle/cap for one second before ready checks."""

        return max(5, int(round(1.0 / float(self._physics_dt))))

    def _parse_bimanual_hand_angles(self, right: Any, left: Any) -> tuple[Any, Any]:
        """Capability marker for the standard native-Wuji policy wrapper.

        The shared wrapper selects two 20-DOF native hand vectors when this
        migrated bimanual runtime exposes the parser capability.
        """

        return right, left

    def _build_observation_space(self) -> spaces.Dict:
        observation_space = super()._build_observation_space()
        state_spaces = observation_space.spaces["state"].spaces
        state_spaces.update(
            {
                "bottle_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
                "cap_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
                "cap_to_bottle": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
                "opening_metrics": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float64),
                "fixture_role_state": spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float64),
                "task_stage": spaces.Box(0.0, 4.0, shape=(1,), dtype=np.float64),
            }
        )
        observation_space.spaces["hs014_hidden_state"] = spaces.Box(
            -np.inf, np.inf, shape=(16,), dtype=np.float64
        )
        observation_space.spaces["hs014_success_truth"] = spaces.Box(
            0.0, 1.0, shape=(8,), dtype=np.float64
        )
        return observation_space

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        if self._stage is None:
            raise RuntimeError("HS014 task stage is unavailable")
        if self._physics_backend != "physx":
            raise RuntimeError("HS014 may only compose into PhysX")

        from pxr import Usd

        asset_support = _load_hs014_asset_support()
        bottle_asset = Path(asset_support.BOTTLE_ASSET).resolve()

        self._hs014_source_contract = asset_support.source_asset_contract()
        if not self._hs014_source_contract.get("pass", False):
            raise RuntimeError("HS014 authoritative source hashes changed")

        # task_preview removes the complete shared YCB set from HS014's MJCF
        # before USD conversion.  Runtime mutation would be too late: piled
        # dynamic clutter can enter first PhysX before reset ownership exists.
        self._hs014_source_contract["shared_ycb_seed_filter"] = "pre_import_removed_all"

        self._hs014_composition = asset_support.compose_assets(
            self._stage,
            table_top_z=_TABLE_TOP_Z,
            bottle_reference=str(bottle_asset),
            cap_data=asset_support.decode_authoritative_cap(),
            task_root=TASK_ASSET_ROOT,
            task_complete=False,
            scope="full_collection_vertical_slice",
        )
        # The collection scene is generated from the authoritative model. Compose
        # its verified compact visual here; keep runtime material and collision owners.
        compact_scene = _HS014_ASSET_SUPPORT_PATH.with_name("hs014_core_scene.usda")
        if compact_scene.is_file():
            compact_visual = self._stage.GetPrimAtPath(
                f"{CAP_PRIM_PATH}/Visual/AuthoritativeCapMesh"
            )
            for name in (
                "points", "faceVertexIndices", "normals", "primvars:st",
                "primvars:st:indices", "primvars:normals", "primvars:normals:indices",
            ):
                compact_visual.RemoveProperty(name)
            compact_visual.GetReferences().AddReference(
                str(compact_scene),
                "/World/HS014Core/BottleCap/Visual/AuthoritativeCapMesh",
            )
        cap_contract = self._hs014_composition.get("cap", {})
        if not (
            math.isclose(
                float(cap_contract.get("mass_kg", math.nan)),
                CAP_MASS_KG,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and np.allclose(
                cap_contract.get("center_of_mass_local_m", []),
                CAP_CENTER_OF_MASS_LOCAL_M,
                rtol=0.0,
                atol=1.0e-12,
            )
            and np.allclose(
                cap_contract.get("diagonal_inertia_kg_m2", []),
                CAP_DIAGONAL_INERTIA_KG_M2,
                rtol=0.0,
                atol=1.0e-12,
            )
            and np.allclose(
                cap_contract.get("principal_axes_wxyz", []),
                CAP_PRINCIPAL_AXES_WXYZ,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise RuntimeError("HS014 cap mass-property composition changed")
        bottle_contract = self._hs014_composition.get("bottle", {})
        topology_contract = self._hs014_composition.get(
            "cap_bottle_contact_topology", {}
        )
        source_paths = [
            str(path) for path in bottle_contract.get("source_collision_paths", [])
        ]
        support_paths = [
            str(path) for path in bottle_contract.get("helper_collision_paths", [])
        ]
        expected_support_paths = [
            f"{BOTTLE_PRIM_PATH}/CradleSupport{index}" for index in range(3)
        ]
        cap_outer_path = str(cap_contract.get("outer_collision_path", ""))
        cap_helper_path = str(cap_contract.get("cradle_helper_path", ""))
        material_path = str(cap_contract.get("helper_physics_material_path", ""))
        expected_filtered_pairs = {
            *(
                (source_path, cap_path)
                for source_path in source_paths
                for cap_path in (cap_outer_path, cap_helper_path)
            ),
            *((support_path, cap_outer_path) for support_path in support_paths),
        }
        if not (
            source_paths
            and support_paths == expected_support_paths
            and cap_outer_path
            and cap_helper_path == f"{CAP_PRIM_PATH}/CradleSphere"
            and material_path
            and cap_contract.get("helper_collision_paths") == [cap_helper_path]
            and cap_contract.get("collision_paths")
            == [cap_outer_path, cap_helper_path]
            and cap_contract.get("collision_model")
            == "outer_cylinder_and_com_centered_cradle_sphere"
            and math.isclose(
                float(cap_contract.get("cradle_radius_m", math.nan)),
                CAP_CRADLE_RADIUS_M,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and math.isclose(
                float(cap_contract.get("cradle_center_y_m", math.nan)),
                CAP_CRADLE_CENTER_Y_M,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and cap_contract.get("cradle_inside_visible_bounds") is True
            and cap_contract.get("cradle_center_matches_center_of_mass") is True
            and np.allclose(
                bottle_contract.get("cradle_support_centers_m", []),
                BOTTLE_CRADLE_SUPPORT_CENTERS_M,
                rtol=0.0,
                atol=1.0e-12,
            )
            and bottle_contract.get("cradle_support_inside_mouth") is True
            and math.isclose(
                float(cap_contract.get("helper_contact_offset_m", math.nan)),
                CRADLE_HELPER_CONTACT_OFFSET_M,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and cap_contract.get("helper_friction_combine_mode")
            == CRADLE_HELPER_FRICTION_COMBINE_MODE
            and cap_contract.get("outer_has_helper_material_binding") is False
            and topology_contract.get("contact_truth_topology")
            == "three_sphere_cradle_to_com_sphere"
            and topology_contract.get("helper_collision_paths")
            == [*support_paths, cap_helper_path]
            and {
                tuple(str(path) for path in pair)
                for pair in topology_contract.get("filtered_shape_pairs", [])
            }
            == expected_filtered_pairs
            and topology_contract.get("source_collision_paths") == source_paths
            and topology_contract.get("bottle_support_paths") == support_paths
            and topology_contract.get("cap_outer_path") == cap_outer_path
            and topology_contract.get("cap_helper_path") == cap_helper_path
            and topology_contract.get("helper_physics_material_path") == material_path
            and topology_contract.get("exact_initial_tangency") is True
            and topology_contract.get("contact_normals_intersect_cap_com") is True
        ):
            raise RuntimeError("HS014 cap-bottle collision topology changed")
        source_geometry = self._stage.GetPrimAtPath(f"{BOTTLE_PRIM_PATH}/geometry")
        if not source_geometry.IsValid():
            raise RuntimeError("HS014 authoritative bottle geometry is unavailable")
        visual_axis = self._apply_bottle_visual_axis_correction(source_geometry)
        self._hs014_composition["bottle"].update(visual_axis)
        runtime_stability = self._apply_bottle_runtime_stability()
        self._hs014_composition["bottle"].update(runtime_stability)
        self._hs014_source_contract["bottle_source_local_aabb_m"] = (
            BOTTLE_SOURCE_LOCAL_AABB_M.tolist()
        )
        self._hs014_source_contract.update(visual_axis)
        self._hs014_source_contract.update(runtime_stability)
        sim_utils.update_stage()

    @staticmethod
    def _apply_bottle_visual_axis_correction(source_geometry: Any) -> dict[str, Any]:
        from pxr import Gf, Sdf, UsdGeom

        xform = UsdGeom.Xformable(source_geometry)
        translate = next(
            (
                op
                for op in xform.GetOrderedXformOps()
                if op.GetOpName() == "xformOp:translate"
            ),
            None,
        )
        if translate is None:
            translate = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        translate.Set(
            Gf.Vec3d(*[float(value) for value in BOTTLE_VISUAL_TRANSLATION_M])
        )
        orient = next(
            (
                op
                for op in xform.GetOrderedXformOps()
                if op.GetOpName() == "xformOp:orient"
            ),
            None,
        )
        if orient is None:
            orient = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        if orient.GetAttr().GetTypeName() != Sdf.ValueTypeNames.Quatd:
            raise RuntimeError("HS014 bottle source orient is not quatd")
        quaternion = BOTTLE_VISUAL_CORRECTION_WXYZ
        orient.Set(
            Gf.Quatd(
                float(quaternion[0]),
                Gf.Vec3d(*[float(value) for value in quaternion[1:4]]),
            )
        )
        mapped_axis = _quat_wxyz_to_matrix(quaternion) @ np.asarray(
            [0.0, 1.0, 0.0], dtype=np.float64
        )
        if not np.allclose(mapped_axis, [0.0, 0.0, 1.0], atol=1.0e-12):
            raise RuntimeError("HS014 bottle visual axis correction changed")
        return {
            "bottle_source_mesh_axis": "+Y",
            "bottle_source_axis_semantics": "base_to_open_mouth",
            "bottle_composed_axis": "+Z_from_base_to_open_mouth",
            "bottle_visual_correction_wxyz": quaternion.tolist(),
            "bottle_visual_translation_m": BOTTLE_VISUAL_TRANSLATION_M.tolist(),
            "bottle_visual_proxy_axis_aligned": True,
        }

    def _apply_bottle_runtime_stability(self) -> dict[str, Any]:
        """Make the upright reset physically quiet without pinning the bottle."""

        if self._stage is None:
            raise RuntimeError("HS014 task stage is unavailable")
        from pxr import Gf, PhysxSchema, UsdPhysics

        bottle = self._stage.GetPrimAtPath(BOTTLE_PRIM_PATH)
        if not bottle.IsValid():
            raise RuntimeError("HS014 bottle rigid owner is unavailable")
        mass = UsdPhysics.MassAPI.Apply(bottle)
        mass.CreateCenterOfMassAttr().Set(
            Gf.Vec3f(0.0, 0.0, BOTTLE_CENTER_OF_MASS_LOCAL_Z_M)
        )
        physx = PhysxSchema.PhysxRigidBodyAPI.Apply(bottle)
        physx.CreateLinearDampingAttr().Set(BOTTLE_LINEAR_DAMPING)
        physx.CreateAngularDampingAttr().Set(BOTTLE_ANGULAR_DAMPING)
        physx.CreateSleepThresholdAttr().Set(BOTTLE_SLEEP_THRESHOLD)
        physx.CreateSolverPositionIterationCountAttr().Set(
            RIGID_SOLVER_POSITION_ITERATIONS
        )
        physx.CreateSolverVelocityIterationCountAttr().Set(
            RIGID_SOLVER_VELOCITY_ITERATIONS
        )
        physx.CreateMaxDepenetrationVelocityAttr().Set(
            RIGID_MAX_DEPENETRATION_VELOCITY_M_S
        )

        cap = self._stage.GetPrimAtPath(CAP_PRIM_PATH)
        if not cap.IsValid():
            raise RuntimeError("HS014 cap rigid owner is unavailable")
        cap_mass = UsdPhysics.MassAPI.Apply(cap)
        cap_mass.CreateMassAttr().Set(CAP_MASS_KG)
        cap_mass.CreateCenterOfMassAttr().Set(
            Gf.Vec3f(*[float(value) for value in CAP_CENTER_OF_MASS_LOCAL_M])
        )
        cap_mass.CreateDiagonalInertiaAttr().Set(
            Gf.Vec3f(*[float(value) for value in CAP_DIAGONAL_INERTIA_KG_M2])
        )
        cap_mass.CreatePrincipalAxesAttr().Set(
            Gf.Quatf(
                float(CAP_PRINCIPAL_AXES_WXYZ[0]),
                Gf.Vec3f(
                    *[float(value) for value in CAP_PRINCIPAL_AXES_WXYZ[1:]]
                ),
            )
        )
        cap_physx = PhysxSchema.PhysxRigidBodyAPI.Apply(cap)
        cap_physx.CreateLinearDampingAttr().Set(CAP_LINEAR_DAMPING)
        cap_physx.CreateAngularDampingAttr().Set(CAP_ANGULAR_DAMPING)
        cap_physx.CreateSleepThresholdAttr().Set(CAP_SLEEP_THRESHOLD)
        cap_physx.CreateSolverPositionIterationCountAttr().Set(
            CAP_SOLVER_POSITION_ITERATIONS
        )
        cap_physx.CreateSolverVelocityIterationCountAttr().Set(
            CAP_SOLVER_VELOCITY_ITERATIONS
        )
        cap_physx.CreateMaxDepenetrationVelocityAttr().Set(
            RIGID_MAX_DEPENETRATION_VELOCITY_M_S
        )

        return {
            "bottle_reset_support_gap_m": 0.0002,
            "bottle_extra_stability_proxy": None,
            "bottle_source_collider_only": False,
            "bottle_internal_mouth_support": "three_cradle_support_spheres",
            "bottle_internal_mouth_support_count": 3,
            "bottle_runtime_center_of_mass_local_m": [
                0.0,
                0.0,
                BOTTLE_CENTER_OF_MASS_LOCAL_Z_M,
            ],
            "bottle_runtime_linear_damping": BOTTLE_LINEAR_DAMPING,
            "bottle_runtime_angular_damping": BOTTLE_ANGULAR_DAMPING,
            "bottle_runtime_sleep_threshold": BOTTLE_SLEEP_THRESHOLD,
            "rigid_solver_position_iteration_count": (
                RIGID_SOLVER_POSITION_ITERATIONS
            ),
            "cap_solver_position_iteration_count": (
                CAP_SOLVER_POSITION_ITERATIONS
            ),
            "rigid_solver_velocity_iteration_count": (
                RIGID_SOLVER_VELOCITY_ITERATIONS
            ),
            "cap_solver_velocity_iteration_count": (
                CAP_SOLVER_VELOCITY_ITERATIONS
            ),
            "rigid_max_depenetration_velocity_m_s": (
                RIGID_MAX_DEPENETRATION_VELOCITY_M_S
            ),
            "cap_runtime_linear_damping": CAP_LINEAR_DAMPING,
            "cap_runtime_angular_damping": CAP_ANGULAR_DAMPING,
            "cap_runtime_sleep_threshold": CAP_SLEEP_THRESHOLD,
            "cap_thread_pitch_m": THREAD_PITCH_M,
            "cap_runtime_mass_kg": CAP_MASS_KG,
            "cap_runtime_center_of_mass_local_m": (
                CAP_CENTER_OF_MASS_LOCAL_M.tolist()
            ),
            "cap_runtime_diagonal_inertia_kg_m2": (
                CAP_DIAGONAL_INERTIA_KG_M2.tolist()
            ),
            "cap_runtime_principal_axes_wxyz": (
                CAP_PRINCIPAL_AXES_WXYZ.tolist()
            ),
        }

    def _body_awake_readback(self, name: str) -> tuple[bool, bool, str]:
        """Read a task rigid body's PhysX sleep flag without changing it."""

        prim = self._find_rigid_body_prim(name)
        if prim is None:
            return False, False, f"{name} rigid body is unavailable"
        try:
            import omni.physx
            import omni.usd
            from pxr import PhysicsSchemaTools, Sdf

            simulation = omni.physx.get_physx_simulation_interface()
            stage_id = omni.usd.get_context().get_stage_id()
            sleeping = simulation.is_sleeping(
                stage_id,
                PhysicsSchemaTools.sdfPathToInt(Sdf.Path(str(prim.GetPath()))),
            )
            return not bool(sleeping), True, ""
        except Exception as exc:
            return False, False, f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _seated_cap_geometry(
        bottle: np.ndarray | None,
        cap: np.ndarray | None,
    ) -> tuple[float, float, float] | None:
        """Return axial, radial, and axis-error truth for a seated cap."""

        if bottle is None or cap is None:
            return None
        bottle_pose = np.asarray(bottle, dtype=np.float64).reshape(7)
        cap_pose = np.asarray(cap, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite([*bottle_pose, *cap_pose])):
            return None
        bottle_axis = _quat_wxyz_to_matrix(bottle_pose[3:7])[:, 2]
        cap_axis = _quat_wxyz_to_matrix(cap_pose[3:7])[:, 1]
        displacement = cap_pose[:3] - bottle_pose[:3]
        axial_position = float(np.dot(displacement, bottle_axis))
        radial_offset = float(
            np.linalg.norm(displacement - axial_position * bottle_axis)
        )
        axis_alignment = float(
            np.clip(np.dot(bottle_axis, cap_axis), -1.0, 1.0)
        )
        axis_error_deg = math.degrees(math.acos(axis_alignment))
        if not np.all(
            np.isfinite([axial_position, radial_offset, axis_error_deg])
        ):
            return None
        return axial_position, radial_offset, axis_error_deg

    @staticmethod
    def _aim_wrist_optical_pose(
        camera_pose: np.ndarray, target_world: np.ndarray
    ) -> np.ndarray:
        """Keep the physical mount position and aim OpenGL -Z at a task target."""

        corrected = np.asarray(camera_pose, dtype=np.float64).reshape(4, 4).copy()
        target = np.asarray(target_world, dtype=np.float64).reshape(3)
        forward = target - corrected[:3, 3]
        distance = float(np.linalg.norm(forward))
        if not np.isfinite(distance) or distance <= 1.0e-8:
            return corrected
        forward /= distance
        backward = -forward
        up_reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(up_reference, backward))) > 0.94:
            up_reference = corrected[:3, 1]
        up = up_reference - float(np.dot(up_reference, backward)) * backward
        up_norm = float(np.linalg.norm(up))
        if not np.isfinite(up_norm) or up_norm <= 1.0e-8:
            up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            up -= float(np.dot(up, backward)) * backward
            up_norm = float(np.linalg.norm(up))
        up /= max(up_norm, 1.0e-12)
        right = np.cross(up, backward)
        right /= max(float(np.linalg.norm(right)), 1.0e-12)
        up = np.cross(backward, right)
        corrected[:3, :3] = np.column_stack([right, up, backward])
        return corrected

    def _wrist_camera_task_target(self, side: str) -> np.ndarray:
        if side == "left":
            pose = self._physics_body_pose(BOTTLE_NAME)
            if pose is None:
                pose = self._hs014_reset_layout.get(
                    BOTTLE_NAME,
                    np.asarray(
                        [
                            *BOTTLE_NOMINAL_XY,
                            BOTTLE_ROOT_Z + float(self.delta_h[0]),
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        dtype=np.float64,
                    ),
                )
            pose = np.asarray(pose, dtype=np.float64).reshape(7)
            return pose[:3] + _quat_wxyz_to_matrix(pose[3:7]) @ np.asarray(
                [0.0, 0.0, -0.10], dtype=np.float64
            )
        pose = self._physics_body_pose(CAP_NAME)
        if pose is None:
            pose = self._hs014_reset_layout.get(
                CAP_NAME,
                np.asarray(
                    [
                        *CAP_NOMINAL_XY,
                        CAP_ROOT_Z + float(self.delta_h[0]),
                        math.sqrt(0.5),
                        math.sqrt(0.5),
                        0.0,
                        0.0,
                    ],
                    dtype=np.float64,
                ),
            )
        pose = np.asarray(pose, dtype=np.float64).reshape(7)
        return pose[:3] + _quat_wxyz_to_matrix(pose[3:7]) @ np.asarray(
            [0.0, 0.013, 0.0], dtype=np.float64
        )

    def _recording_camera_pose_matrix_for_key(
        self, camera_key: str
    ) -> np.ndarray | None:
        pose = super()._recording_camera_pose_matrix_for_key(camera_key)
        if pose is None:
            return None
        key = str(camera_key)
        if key in {"wrist_left", "wrist_left_camera", "left_wrist"}:
            return self._aim_wrist_optical_pose(
                pose, self._wrist_camera_task_target("left")
            )
        if key in {"wrist_right", "wrist_right_camera", "right_wrist", "wrist"}:
            return self._aim_wrist_optical_pose(
                pose, self._wrist_camera_task_target("right")
            )
        return pose

    def _update_recording_cameras(self) -> None:
        """Apply the HS014 wrist optical correction after shared mount tracking."""

        super()._update_recording_cameras()
        if not self._cameras or self._torch is None:
            return
        for name in ("wrist_left_camera", "wrist_right_camera"):
            if name not in self._cameras:
                continue
            camera_pose = self._recording_camera_pose_matrix_for_key(name)
            if camera_pose is None:
                continue
            quaternion = _matrix_to_quat_wxyz(camera_pose[:3, :3])
            if self._isaaclab_uses_xyzw:
                quaternion = quaternion[[1, 2, 3, 0]]
            position = self._torch.as_tensor(
                camera_pose[:3, 3].reshape(1, 3),
                dtype=self._torch.float32,
                device=self._isaac_device,
            )
            orientation = self._torch.as_tensor(
                quaternion.reshape(1, 4),
                dtype=self._torch.float32,
                device=self._isaac_device,
            )
            self._cameras[name].set_world_poses(
                position, orientation, convention="opengl"
            )

    def _hand_body_paths(self, side: str) -> list[str]:
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
            name = prim.GetName()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and (
                name == f"{side}_palm_link" or name.startswith(f"{side}_finger")
            ):
                paths.append(str(prim.GetPath()))
        return paths

    def _palm_prim(self, side: str) -> Any | None:
        if self._stage is None:
            return None
        from pxr import Usd, UsdPhysics

        root = self._stage.GetPrimAtPath(
            f"/World/HandsOnRobots/{side.capitalize()}Robot"
        )
        if not root.IsValid():
            return None
        for prim in Usd.PrimRange(root):
            if (
                prim.GetName() == f"{side}_palm_link"
                and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ):
                return prim
        return None

    def _create_fixed_joint(self, key: str, side: str, object_path: str) -> None:
        if self._stage is None:
            return
        from pxr import Gf, UsdPhysics

        palm = self._palm_prim(side)
        object_prim = self._stage.GetPrimAtPath(object_path)
        if palm is None or not object_prim.IsValid():
            raise RuntimeError(f"HS014 {key} fixed-joint bodies are unavailable")
        joint = UsdPhysics.FixedJoint.Define(
            self._stage, f"/World/HS014RuntimeJoints/{key}"
        )
        joint.CreateBody0Rel().SetTargets([palm.GetPath()])
        joint.CreateBody1Rel().SetTargets([object_prim.GetPath()])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        joint.CreateJointEnabledAttr(False)
        joint.CreateBreakForceAttr(float("inf"))
        joint.CreateBreakTorqueAttr(float("inf"))
        self._hs014_joint_prims[key] = joint

    def _create_thread_joint(self) -> None:
        """Author the engaged bottle-cap screw gate in the Isaac PhysX stage."""
        if self._stage is None:
            return
        from pxr import Gf, UsdPhysics

        bottle = self._stage.GetPrimAtPath(BOTTLE_PRIM_PATH)
        cap = self._stage.GetPrimAtPath(CAP_PRIM_PATH)
        if not bottle.IsValid() or not cap.IsValid():
            raise RuntimeError("HS014 thread-joint bodies are unavailable")
        joint = UsdPhysics.RevoluteJoint.Define(
            self._stage, "/World/HS014RuntimeJoints/cap_thread"
        )
        joint.CreateBody0Rel().SetTargets([bottle.GetPath()])
        joint.CreateBody1Rel().SetTargets([cap.GetPath()])
        # Bottle root local Z is the neck axis. The cap source is along local
        # +Y, so its joint frame rotates -90 degrees around X.
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(
            Gf.Quatf(
                float(math.sqrt(0.5)),
                Gf.Vec3f(-float(math.sqrt(0.5)), 0.0, 0.0),
            )
        )
        joint.CreateAxisAttr().Set(UsdPhysics.Tokens.z)
        # Keep the authored cradle/contact probes active for reset and runtime
        # diagnostics; PhysX joints otherwise default to suppressing collision
        # between their connected bodies on some Isaac Sim builds.
        joint.CreateCollisionEnabledAttr(True)
        joint.CreateJointEnabledAttr(True)
        joint.CreateBreakForceAttr(float("inf"))
        joint.CreateBreakTorqueAttr(float("inf"))
        self._hs014_thread_joint = joint
        self._hs014_thread_joint_enabled = True

    def _read_thread_joint_enabled(self) -> bool:
        joint = self._hs014_thread_joint
        if joint is None:
            self._hs014_thread_joint_enabled = False
            return False
        self._hs014_thread_joint_enabled = bool(joint.GetJointEnabledAttr().Get())
        return self._hs014_thread_joint_enabled

    def _set_thread_joint_enabled(self, enabled: bool) -> bool:
        joint = self._hs014_thread_joint
        if joint is None:
            self._hs014_thread_joint_enabled = False
            return False
        requested = bool(enabled)
        joint.GetJointEnabledAttr().Set(requested)
        actual = self._read_thread_joint_enabled()
        return actual == requested

    def _create_contact_view(
        self, physics_sim_view: Any, sensor_path: str, filters: list[str]
    ) -> Any:
        view = physics_sim_view.create_rigid_contact_view(
            sensor_path,
            filter_patterns=filters,
            max_contact_data_count=max(64, len(filters) * 8),
        )
        if view is None or int(view.sensor_count) < 1 or int(view.filter_count) < 1:
            raise RuntimeError(
                f"HS014 contact view has no sensor/filter: {sensor_path}"
            )
        return view

    def _init_task_specific_runtime(self) -> None:
        if not self._robots:
            return
        self._hs014_runtime_physics_schema = (
            self._readback_hs014_runtime_physics_schema()
        )
        physics_sim_view = self._robots["right"]._physics_sim_view
        if physics_sim_view is None:
            raise RuntimeError("HS014 PhysX simulation view is unavailable")
        left_paths = self._hand_body_paths("left")
        right_paths = self._hand_body_paths("right")
        if not left_paths or not right_paths:
            raise RuntimeError("HS014 could not resolve both Wuji hand body sets")
        table = self._find_rigid_body_prim("table")
        if table is None:
            raise RuntimeError("HS014 table rigid body is unavailable")
        self._hs014_contact_views = {
            "left_bottle": self._create_contact_view(
                physics_sim_view, BOTTLE_PRIM_PATH, left_paths
            ),
            "left_cap": self._create_contact_view(
                physics_sim_view, CAP_PRIM_PATH, left_paths
            ),
            "right_bottle": self._create_contact_view(
                physics_sim_view, BOTTLE_PRIM_PATH, right_paths
            ),
            "right_cap": self._create_contact_view(
                physics_sim_view, CAP_PRIM_PATH, right_paths
            ),
            "cap_bottle": self._create_contact_view(
                physics_sim_view, CAP_PRIM_PATH, [BOTTLE_PRIM_PATH]
            ),
            "bottle_support": self._create_contact_view(
                physics_sim_view, BOTTLE_PRIM_PATH, [str(table.GetPath())]
            ),
        }
        self._create_fixed_joint("left_fixture", "left", BOTTLE_PRIM_PATH)
        self._create_fixed_joint("right_cap", "right", CAP_PRIM_PATH)
        self._create_thread_joint()
        self._handson_post_physics_substep_callback = (
            self._hs014_post_physics_substep
        )

    def _readback_hs014_collision_topology(self) -> dict[str, Any]:
        """Read the exact torque-neutral sphere-sphere cradle after first reset."""

        if self._stage is None:
            raise RuntimeError("HS014 collision topology stage is unavailable")
        from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade

        bottle = self._hs014_composition.get("bottle", {})
        cap = self._hs014_composition.get("cap", {})
        topology = self._hs014_composition.get("cap_bottle_contact_topology", {})
        source_paths = [str(path) for path in bottle.get("source_collision_paths", [])]
        support_paths = [str(path) for path in bottle.get("helper_collision_paths", [])]
        expected_support_paths = [
            f"{BOTTLE_PRIM_PATH}/CradleSupport{index}" for index in range(3)
        ]
        cap_outer_path = str(cap.get("outer_collision_path", ""))
        cap_helper_path = str(cap.get("cradle_helper_path", ""))
        material_path = str(cap.get("helper_physics_material_path", ""))
        expected_targets = {
            **{
                source_path: {cap_outer_path, cap_helper_path}
                for source_path in source_paths
            },
            **{support_path: {cap_outer_path} for support_path in support_paths},
            cap_outer_path: set(source_paths) | set(support_paths),
            cap_helper_path: set(source_paths),
        }
        checks: dict[str, bool] = {
            "exact_helper_paths": bool(
                source_paths
                and support_paths == expected_support_paths
                and cap_helper_path == f"{CAP_PRIM_PATH}/CradleSphere"
            ),
            "contact_truth_topology": (
                topology.get("contact_truth_topology")
                == "three_sphere_cradle_to_com_sphere"
            ),
            "no_seat_helper": not self._stage.GetPrimAtPath(
                f"{CAP_PRIM_PATH}/Seat"
            ).IsValid(),
        }
        observed_targets: dict[str, list[str]] = {}
        for path, targets in expected_targets.items():
            prim = self._stage.GetPrimAtPath(path)
            valid = bool(prim.IsValid())
            checks[f"shape_present:{path}"] = valid
            if not valid:
                observed_targets[path] = []
                continue
            checks[f"collision_enabled:{path}"] = bool(
                prim.HasAPI(UsdPhysics.CollisionAPI)
                and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                is not False
            )
            checks[f"no_nested_rigid_owner:{path}"] = not prim.HasAPI(
                UsdPhysics.RigidBodyAPI
            )
            relation = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel()
            observed = {
                str(target)
                for target in (
                    [] if not relation or not relation.IsValid() else relation.GetTargets()
                )
            }
            observed_targets[path] = sorted(observed)
            checks[f"exact_filtered_targets:{path}"] = observed == targets

        for root_path in (BOTTLE_PRIM_PATH, CAP_PRIM_PATH):
            root = self._stage.GetPrimAtPath(root_path)
            checks[f"no_root_blanket_filter:{root_path}"] = bool(
                root.IsValid() and not root.HasAPI(UsdPhysics.FilteredPairsAPI)
            )

        def translation(prim: Any) -> list[float] | None:
            if not prim.IsValid():
                return None
            value = next(
                (
                    op.Get()
                    for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
                    if op.GetOpName() == "xformOp:translate"
                ),
                None,
            )
            return None if value is None else [float(component) for component in value]

        def helper_sphere_contract(
            path: str, radius: float, expected_center: list[float]
        ) -> bool:
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsA(UsdGeom.Sphere):
                return False
            physx = PhysxSchema.PhysxCollisionAPI(prim)
            binding_targets = [
                str(target)
                for target in UsdShade.MaterialBindingAPI(prim)
                .GetDirectBindingRel("physics")
                .GetTargets()
            ]
            return bool(
                math.isclose(
                    float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()),
                    radius,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                and np.allclose(
                    translation(prim), expected_center, rtol=0.0, atol=1.0e-12
                )
                and UsdGeom.Imageable(prim).ComputeVisibility()
                == UsdGeom.Tokens.invisible
                and physx.GetContactOffsetAttr().HasAuthoredValueOpinion()
                and math.isclose(
                    float(physx.GetContactOffsetAttr().Get()),
                    # PhysxCollisionAPI stores this attribute as float32.
                    float(np.float32(CRADLE_HELPER_CONTACT_OFFSET_M)),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                and physx.GetRestOffsetAttr().HasAuthoredValueOpinion()
                and math.isclose(
                    float(physx.GetRestOffsetAttr().Get()),
                    CRADLE_HELPER_REST_OFFSET_M,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                and binding_targets == [material_path]
            )

        for index, (path, center) in enumerate(
            zip(support_paths, BOTTLE_CRADLE_SUPPORT_CENTERS_M, strict=True)
        ):
            checks[f"bottle_support_sphere_geometry:{index}"] = (
                helper_sphere_contract(
                    path,
                    BOTTLE_CRADLE_SUPPORT_RADIUS_M,
                    [float(value) for value in center],
                )
            )
        checks["cap_com_sphere_geometry"] = helper_sphere_contract(
            cap_helper_path,
            CAP_CRADLE_RADIUS_M,
            [0.0, CAP_CRADLE_CENTER_Y_M, 0.0],
        )
        checks["cap_helper_inside_visible_bounds"] = (
            cap.get("cradle_inside_visible_bounds") is True
        )
        checks["cap_helper_center_matches_com"] = bool(
            cap.get("cradle_center_matches_center_of_mass") is True
            and np.allclose(
                [0.0, CAP_CRADLE_CENTER_Y_M, 0.0],
                CAP_CENTER_OF_MASS_LOCAL_M,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        checks["bottle_supports_inside_mouth"] = bool(
            bottle.get("cradle_support_inside_mouth") is True
            and math.isclose(
                BOTTLE_CRADLE_SUPPORT_RADIAL_ENVELOPE_M,
                0.0039,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and BOTTLE_CRADLE_SUPPORT_RADIAL_ENVELOPE_M
            <= BOTTLE_MOUTH_CONTAINMENT_RADIUS_M
        )
        helper_targets = set(observed_targets.get(cap_helper_path, []))
        checks["three_support_helper_pairs_unfiltered"] = bool(
            len(support_paths) == 3
            and all(
                cap_helper_path not in set(observed_targets.get(path, []))
                and path not in helper_targets
                for path in support_paths
            )
        )
        checks["exact_initial_sphere_tangency"] = bool(
            math.isclose(CRADLE_CENTER_DZ_M, math.sqrt(CRADLE_RADIUS_SUM_M**2 - BOTTLE_CRADLE_SUPPORT_RING_RADIUS_M**2), rel_tol=0.0, abs_tol=1.0e-12)
            and math.isclose(
                CRADLE_CENTER_DISTANCE_M, CRADLE_RADIUS_SUM_M, rel_tol=0.0, abs_tol=1.0e-12
            )
            and math.isclose(
                CRADLE_RADIUS_SUM_M, 0.0145, rel_tol=0.0, abs_tol=1.0e-12
            )
            and math.isclose(
                CRADLE_CENTER_DISTANCE_M,
                CRADLE_RADIUS_SUM_M,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
        checks["contact_normals_intersect_cap_com"] = bool(
            topology.get("contact_normals_intersect_cap_com") is True
            and cap.get("cradle_center_matches_center_of_mass") is True
        )

        material_prim = self._stage.GetPrimAtPath(material_path)
        material_api = UsdPhysics.MaterialAPI(material_prim)
        physx_material_api = PhysxSchema.PhysxMaterialAPI(material_prim)
        attrs = {
            "static_friction": material_api.GetStaticFrictionAttr(),
            "dynamic_friction": material_api.GetDynamicFrictionAttr(),
            "restitution": material_api.GetRestitutionAttr(),
            "friction_combine_mode": physx_material_api.GetFrictionCombineModeAttr(),
        }
        expected_values = {
            "static_friction": CRADLE_HELPER_STATIC_FRICTION,
            "dynamic_friction": CRADLE_HELPER_DYNAMIC_FRICTION,
            "restitution": CRADLE_HELPER_RESTITUTION,
            "friction_combine_mode": CRADLE_HELPER_FRICTION_COMBINE_MODE,
        }
        observed_values: dict[str, Any] = {}
        authored: dict[str, bool] = {}
        for key, attr in attrs.items():
            valid = bool(attr and attr.IsValid())
            value = attr.Get() if valid else None
            observed_values[key] = value
            authored[key] = bool(valid and attr.HasAuthoredValueOpinion())
            expected = expected_values[key]
            checks[f"helper_material_authored:{key}"] = authored[key]
            checks[f"helper_material_value:{key}"] = bool(
                value == expected
                if isinstance(expected, str)
                else value is not None
                and math.isclose(
                    float(value), expected, rel_tol=0.0, abs_tol=1.0e-12
                )
            )
        outer = self._stage.GetPrimAtPath(cap_outer_path)
        outer_cylinder = UsdGeom.Cylinder(outer)
        outer_translate = next((op.Get() for op in UsdGeom.Xformable(outer).GetOrderedXformOps() if op.GetOpName() == "xformOp:translate"), None)
        checks["cap_outer_cylinder_geometry"] = bool(
            outer.IsA(UsdGeom.Cylinder)
            and math.isclose(float(outer_cylinder.GetRadiusAttr().Get()), float(cap.get("collision_radius_m", math.nan)), rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(float(outer_cylinder.GetHeightAttr().Get()), float(cap.get("collision_height_m", math.nan)), rel_tol=0.0, abs_tol=1e-12)
            and outer_cylinder.GetAxisAttr().Get() == UsdGeom.Tokens.y
            and outer_translate is not None
            and np.allclose([float(v) for v in outer_translate], [0.0, float(cap.get("collision_center_y_m", math.nan)), 0.0], rtol=0.0, atol=1e-12)
        )
        checks["cap_outer_invisible"] = bool(
            UsdGeom.Imageable(outer).ComputeVisibility() == UsdGeom.Tokens.invisible
        )
        outer_helper_material_targets = [
            str(target)
            for target in UsdShade.MaterialBindingAPI(outer)
            .GetDirectBindingRel("physics")
            .GetTargets()
        ]
        checks["helper_material_valid"] = bool(
            material_prim.IsValid()
            and material_prim.HasAPI(UsdPhysics.MaterialAPI)
            and material_prim.HasAPI(PhysxSchema.PhysxMaterialAPI)
        )
        checks["cap_outer_has_no_helper_material"] = (
            material_path not in outer_helper_material_targets
        )
        material_readback = {
            "material_path": material_path,
            "values": observed_values,
            "authored_value_opinions": authored,
            "cap_outer_binding_targets": outer_helper_material_targets,
            "passed": bool(
                checks["helper_material_valid"]
                and checks["cap_outer_has_no_helper_material"]
                and all(authored.values())
                and all(
                    checks[f"helper_material_value:{key}"] for key in expected_values
                )
            ),
        }
        return {
            "phase": "post_first_reset_torque_neutral_cradle_readback",
            "contact_truth_topology": "three_sphere_cradle_to_com_sphere",
            "source_collision_paths": source_paths,
            "bottle_support_paths": support_paths,
            "cap_outer_path": cap_outer_path,
            "cap_helper_path": cap_helper_path,
            "unfiltered_contact_pairs": [
                [path, cap_helper_path] for path in support_paths
            ],
            "observed_filtered_targets": observed_targets,
            "center_delta_z_m": CRADLE_CENTER_DZ_M,
            "center_horizontal_offset_m": BOTTLE_CRADLE_SUPPORT_RING_RADIUS_M,
            "center_distance_m": CRADLE_CENTER_DISTANCE_M,
            "radius_sum_m": CRADLE_RADIUS_SUM_M,
            "helper_physics_material": material_readback,
            "root_contact_view_sensor_path": CAP_PRIM_PATH,
            "root_contact_view_filter_path": BOTTLE_PRIM_PATH,
            "checks": checks,
            "passed": all(checks.values()),
        }


    def _readback_hs014_runtime_physics_schema(self) -> dict[str, Any]:
        """Fail closed on the post-first-reset HS014 PhysX schema contract."""

        if self._stage is None or self._sim is None:
            raise RuntimeError("HS014 runtime physics schema is unavailable")
        from pxr import PhysxSchema, UsdPhysics

        expected = {
            "linear_damping": BOTTLE_LINEAR_DAMPING,
            "angular_damping": BOTTLE_ANGULAR_DAMPING,
            "sleep_threshold": BOTTLE_SLEEP_THRESHOLD,
            "solver_position_iteration_count": (
                RIGID_SOLVER_POSITION_ITERATIONS
            ),
            "solver_velocity_iteration_count": (
                RIGID_SOLVER_VELOCITY_ITERATIONS
            ),
            "max_depenetration_velocity_m_s": (
                RIGID_MAX_DEPENETRATION_VELOCITY_M_S
            ),
        }
        getters = {
            "linear_damping": "GetLinearDampingAttr",
            "angular_damping": "GetAngularDampingAttr",
            "sleep_threshold": "GetSleepThresholdAttr",
            "solver_position_iteration_count": (
                "GetSolverPositionIterationCountAttr"
            ),
            "solver_velocity_iteration_count": (
                "GetSolverVelocityIterationCountAttr"
            ),
            "max_depenetration_velocity_m_s": (
                "GetMaxDepenetrationVelocityAttr"
            ),
        }
        mass_expected = {
            "bottle": {
                "mass_kg": 0.10,
                "center_of_mass_local_m": [
                    0.0,
                    0.0,
                    BOTTLE_CENTER_OF_MASS_LOCAL_Z_M,
                ],
                "diagonal_inertia_kg_m2": [0.00035, 0.00035, 0.000041],
                "principal_axes_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "cap": {
                "mass_kg": CAP_MASS_KG,
                "center_of_mass_local_m": CAP_CENTER_OF_MASS_LOCAL_M.tolist(),
                "diagonal_inertia_kg_m2": (
                    CAP_DIAGONAL_INERTIA_KG_M2.tolist()
                ),
                "principal_axes_wxyz": CAP_PRINCIPAL_AXES_WXYZ.tolist(),
            },
        }
        mass_getters = {
            "mass_kg": "GetMassAttr",
            "center_of_mass_local_m": "GetCenterOfMassAttr",
            "diagonal_inertia_kg_m2": "GetDiagonalInertiaAttr",
            "principal_axes_wxyz": "GetPrincipalAxesAttr",
        }
        # MassAPI vector/scalar attributes are stored as USD float precision.
        # Compare at a field-appropriate tolerance instead of treating their
        # normal float32 quantization as a contract failure.  The much smaller
        # inertia values still retain a tighter absolute bound.
        mass_tolerances = {
            "mass_kg": 1.0e-8,
            "center_of_mass_local_m": 1.0e-8,
            "diagonal_inertia_kg_m2": 1.0e-10,
            "principal_axes_wxyz": 1.0e-8,
        }
        bodies: dict[str, Any] = {}
        checks: dict[str, bool] = {}
        for label, path in (("bottle", BOTTLE_PRIM_PATH), ("cap", CAP_PRIM_PATH)):
            body_expected = dict(expected)
            if label == "cap":
                body_expected.update(
                    linear_damping=CAP_LINEAR_DAMPING,
                    angular_damping=CAP_ANGULAR_DAMPING,
                    sleep_threshold=CAP_SLEEP_THRESHOLD,
                    solver_position_iteration_count=(
                        CAP_SOLVER_POSITION_ITERATIONS
                    ),
                    solver_velocity_iteration_count=(
                        CAP_SOLVER_VELOCITY_ITERATIONS
                    ),
                )
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"HS014 {label} rigid owner is unavailable")
            api = PhysxSchema.PhysxRigidBodyAPI(prim)
            values: dict[str, Any] = {}
            for key, getter_name in getters.items():
                attribute = getattr(api, getter_name)()
                value = None if not attribute or not attribute.IsValid() else attribute.Get()
                if value is None:
                    raise RuntimeError(
                        f"HS014 {label} is missing PhysX attribute {key}"
                    )
                if key.startswith("solver_"):
                    values[key] = int(value)
                    checks[f"{label}_{key}"] = int(value) == int(
                        body_expected[key]
                    )
                else:
                    values[key] = float(value)
                    checks[f"{label}_{key}"] = math.isclose(
                        float(value),
                        float(body_expected[key]),
                        rel_tol=0.0,
                        abs_tol=1.0e-6,
                    )
            mass_api = UsdPhysics.MassAPI(prim)
            for key, getter_name in mass_getters.items():
                attribute = getattr(mass_api, getter_name)()
                if (
                    not attribute
                    or not attribute.IsValid()
                    or not attribute.HasAuthoredValueOpinion()
                ):
                    raise RuntimeError(
                        f"HS014 {label} is missing explicit MassAPI attribute {key}"
                    )
                value = attribute.Get()
                if value is None:
                    raise RuntimeError(
                        f"HS014 {label} MassAPI attribute {key} has no value"
                    )
                if key == "mass_kg":
                    values[key] = float(value)
                    checks[f"{label}_{key}"] = math.isclose(
                        float(value),
                        float(mass_expected[label][key]),
                        rel_tol=0.0,
                        abs_tol=mass_tolerances[key],
                    )
                    continue
                if key == "principal_axes_wxyz":
                    imaginary = value.GetImaginary()
                    observed = np.asarray(
                        [
                            float(value.GetReal()),
                            float(imaginary[0]),
                            float(imaginary[1]),
                            float(imaginary[2]),
                        ],
                        dtype=np.float64,
                    )
                else:
                    observed = np.asarray(
                        [float(component) for component in value],
                        dtype=np.float64,
                    )
                values[key] = observed.tolist()
                checks[f"{label}_{key}"] = bool(
                    np.allclose(
                        observed,
                        mass_expected[label][key],
                        rtol=0.0,
                        atol=mass_tolerances[key],
                    )
                )
            bodies[label] = values

        physx_cfg = getattr(getattr(self._sim, "cfg", None), "physx", None)
        scene_max_velocity_iterations = getattr(
            physx_cfg, "max_velocity_iteration_count", None
        )
        checks["scene_max_velocity_iteration_count"] = (
            type(scene_max_velocity_iterations) is int
            and scene_max_velocity_iterations == SCENE_MAX_VELOCITY_ITERATIONS
        )
        collision_topology = self._readback_hs014_collision_topology()
        for key, passed in collision_topology["checks"].items():
            checks[f"collision_topology:{key}"] = bool(passed)
        report = {
            "phase": "post_first_simulation_context_reset_usd_schema_readback",
            "bodies": bodies,
            "collision_topology": collision_topology,
            "scene_max_velocity_iteration_count": scene_max_velocity_iterations,
            "checks": checks,
            "passed": all(checks.values()),
        }
        if not report["passed"]:
            failed = [key for key, passed in checks.items() if not passed]
            raise RuntimeError(
                "HS014 runtime PhysX schema readback failed: " + ",".join(failed)
            )
        return report

    def _validate_task_initial_state_before_ready(
        self,
        *,
        commit_semantic_baseline: bool = True,
    ) -> Mapping[str, Any] | None:
        base_report = super()._validate_task_initial_state_before_ready()
        errors: list[str] = []
        bottle = self._physics_body_pose(BOTTLE_NAME)
        cap = self._physics_body_pose(CAP_NAME)
        bottle_velocity = self._physics_body_velocity(BOTTLE_NAME)
        cap_velocity = self._physics_body_velocity(CAP_NAME)
        support_force, support_penetration = self._contact_reading(
            "bottle_support"
        )
        cap_bottle_force, cap_bottle_penetration = self._contact_reading(
            "cap_bottle"
        )
        initial_hand_forces = {
            key: float(self._contact_reading(key)[0])
            for key in ("left_bottle", "left_cap", "right_bottle", "right_cap")
        }
        bottle_awake, bottle_awake_available, bottle_awake_error = (
            self._body_awake_readback(BOTTLE_NAME)
        )
        cap_awake, cap_awake_available, cap_awake_error = (
            self._body_awake_readback(CAP_NAME)
        )
        robot_passed, robot_report = initial_robot_state_report(self)
        state_values = [bottle, cap, bottle_velocity, cap_velocity]
        seated_geometry = self._seated_cap_geometry(bottle, cap)
        bottle_authored_pose_ok = False
        bottle_authored_translation_error = math.inf
        bottle_authored_rotation_error = math.inf
        if bottle is not None:
            (
                bottle_authored_pose_ok,
                bottle_authored_translation_error,
                bottle_authored_rotation_error,
            ) = pose_error_wxyz(bottle, self._hs014_authored_bottle_baseline)
        if any(value is None for value in state_values) or not all(
            np.all(np.isfinite(np.asarray(value, dtype=np.float64)))
            for value in state_values
            if value is not None
        ):
            errors.append("actual bottle/cap pose or velocity is unavailable or non-finite")
        if not np.isfinite(support_force) or support_force < 0.01:
            errors.append(
                f"bottle has no measured table support force: {support_force:.6f}N"
            )
        if not np.isfinite(support_penetration) or support_penetration > 0.0005:
            errors.append(
                "bottle/table support penetration is unavailable or exceeds 0.5 mm: "
                f"{support_penetration:.6f}m"
            )
        if not np.isfinite(cap_bottle_force) or cap_bottle_force < 0.01:
            errors.append(
                f"cap has no measured bottle contact force: {cap_bottle_force:.6f}N"
            )
        if not np.isfinite(cap_bottle_penetration) or cap_bottle_penetration > 0.0005:
            errors.append(
                "cap/bottle penetration is unavailable or exceeds 0.5 mm: "
                f"{cap_bottle_penetration:.6f}m"
            )
        if not all(
            np.isfinite(force) and force <= INITIAL_HAND_OBJECT_FORCE_MAX_N
            for force in initial_hand_forces.values()
        ):
            errors.append(
                "a hand already contacts the bottle or cap before ready: "
                f"{initial_hand_forces}"
            )
        for label, velocity in (
            ("bottle", bottle_velocity),
            ("cap", cap_velocity),
        ):
            if velocity is None:
                continue
            velocity_array = np.asarray(velocity, dtype=np.float64).reshape(6)
            linear_speed = float(np.linalg.norm(velocity_array[:3]))
            angular_speed = float(np.linalg.norm(velocity_array[3:]))
            if linear_speed > 0.05:
                errors.append(
                    f"{label} initial linear speed exceeds 0.05 m/s: "
                    f"{linear_speed:.6f}m/s"
                )
            if angular_speed > 0.75:
                errors.append(
                    f"{label} initial angular speed exceeds 0.75 rad/s: "
                    f"{angular_speed:.6f}rad/s"
                )
        if not robot_passed:
            errors.append("actual dual-ROKAE initial state failed the P0 audit")
        if not (
            bottle_authored_pose_ok
            and bottle_authored_translation_error <= 0.005
            and bottle_authored_rotation_error <= math.radians(2.0)
        ):
            errors.append(
                "bottle settled outside its authored 5 mm / 2 deg window: "
                f"translation={bottle_authored_translation_error:.6f}m, "
                f"rotation={math.degrees(bottle_authored_rotation_error):.6f}deg"
            )
        if seated_geometry is None:
            errors.append("actual cap seated geometry is unavailable or non-finite")
        else:
            axial_position, radial_offset, axis_error_deg = seated_geometry
            if not CAP_SEATED_AXIAL_MIN_M <= axial_position <= CAP_SEATED_AXIAL_MAX_M:
                errors.append(
                    "cap axial position is outside the seated window: "
                    f"{axial_position:.6f}m"
                )
            if radial_offset > CAP_SEATED_RADIAL_MAX_M:
                errors.append(
                    "cap radial offset exceeds the seated window: "
                    f"{radial_offset:.6f}m"
                )
            if axis_error_deg > CAP_SEATED_AXIS_ERROR_MAX_DEG:
                errors.append(
                    "cap axis error exceeds the seated window: "
                    f"{axis_error_deg:.6f}deg"
                )
        for label, awake, available, error in (
            ("bottle", bottle_awake, bottle_awake_available, bottle_awake_error),
            ("cap", cap_awake, cap_awake_available, cap_awake_error),
        ):
            if not available:
                errors.append(f"{label} PhysX sleep state is unavailable: {error}")
            elif not awake:
                errors.append(f"{label} is sleeping before ready")
        diagnostic_report = {
            "passed": not errors,
            "errors": list(errors),
            "bottle_pose_wxyz": (
                None
                if bottle is None
                else np.asarray(bottle, dtype=np.float64).reshape(7).tolist()
            ),
            "cap_pose_wxyz": (
                None
                if cap is None
                else np.asarray(cap, dtype=np.float64).reshape(7).tolist()
            ),
            "bottle_velocity": (
                None
                if bottle_velocity is None
                else np.asarray(bottle_velocity, dtype=np.float64).reshape(6).tolist()
            ),
            "cap_velocity": (
                None
                if cap_velocity is None
                else np.asarray(cap_velocity, dtype=np.float64).reshape(6).tolist()
            ),
            "bottle_support_force_n": float(support_force),
            "bottle_support_penetration_m": float(support_penetration),
            "cap_bottle_force_n": float(cap_bottle_force),
            "cap_bottle_penetration_m": float(cap_bottle_penetration),
            "cap_bottle_contact_diagnostics": dict(
                self._hs014_contact_diagnostics.get("cap_bottle", {})
            ),
            "initial_hand_object_forces_n": initial_hand_forces,
            "bottle_authored_translation_error_m": float(
                bottle_authored_translation_error
            ),
            "bottle_authored_rotation_error_deg": math.degrees(
                bottle_authored_rotation_error
            ),
            "bottle_awake": bool(bottle_awake),
            "cap_awake": bool(cap_awake),
            "cap_seated_axial_position_m": (
                None if seated_geometry is None else float(seated_geometry[0])
            ),
            "cap_seated_radial_offset_m": (
                None if seated_geometry is None else float(seated_geometry[1])
            ),
            "cap_seated_axis_error_deg": (
                None if seated_geometry is None else float(seated_geometry[2])
            ),
            "actual_robot_state": robot_report,
        }
        self._hs014_initial_state_report = dict(diagnostic_report)
        if errors:
            raise RuntimeError(
                "HS014 initial physical-state audit failed before ready: "
                + "; ".join(errors)
            )
        assert seated_geometry is not None
        if commit_semantic_baseline:
            self._hs014_reset_axial_position_m = float(seated_geometry[0])
            assert bottle is not None
            self._hs014_bottle_baseline = np.asarray(
                bottle, dtype=np.float64
            ).reshape(7).copy()
            self._hs014_last_metrics = {}
            self._hs014_previous_twist_reference = None
            self._hs014_accumulated_twist_rad = 0.0
        report = {
            **diagnostic_report,
            "semantic_baseline_committed": bool(commit_semantic_baseline),
        }
        if commit_semantic_baseline:
            self._hs014_initial_state_report = report
        if isinstance(base_report, Mapping):
            return {**base_report, **report}
        return report

    def stable_recording_origin_task_readiness(self) -> Mapping[str, Any]:
        bottle = self._physics_body_pose(BOTTLE_NAME)
        cap = self._physics_body_pose(CAP_NAME)
        bottle_velocity = self._physics_body_velocity(BOTTLE_NAME)
        cap_velocity = self._physics_body_velocity(CAP_NAME)
        support_force, support_penetration = self._contact_reading(
            "bottle_support"
        )
        cap_bottle_force, cap_bottle_penetration = self._contact_reading(
            "cap_bottle"
        )
        initial_hand_forces = {
            key: float(self._contact_reading(key)[0])
            for key in ("left_bottle", "left_cap", "right_bottle", "right_cap")
        }
        bottle_awake, bottle_awake_available, bottle_awake_error = (
            self._body_awake_readback(BOTTLE_NAME)
        )
        cap_awake, cap_awake_available, cap_awake_error = (
            self._body_awake_readback(CAP_NAME)
        )
        values = [bottle, cap, bottle_velocity, cap_velocity]
        seated_geometry = self._seated_cap_geometry(bottle, cap)
        finite = bool(
            all(value is not None for value in values)
            and all(
                np.all(np.isfinite(np.asarray(value, dtype=np.float64)))
                for value in values
                if value is not None
            )
        )
        ready = bool(
            finite
            and bottle_awake_available
            and cap_awake_available
            and bottle_awake
            and cap_awake
            and seated_geometry is not None
            and CAP_SEATED_AXIAL_MIN_M <= seated_geometry[0] <= CAP_SEATED_AXIAL_MAX_M
            and seated_geometry[1] <= CAP_SEATED_RADIAL_MAX_M
            and seated_geometry[2] <= CAP_SEATED_AXIS_ERROR_MAX_DEG
            and support_force >= 0.01
            and np.isfinite(support_penetration)
            and support_penetration <= 0.0005
            and cap_bottle_force >= 0.01
            and np.isfinite(cap_bottle_penetration)
            and cap_bottle_penetration <= 0.0005
            and all(
                force <= INITIAL_HAND_OBJECT_FORCE_MAX_N
                for force in initial_hand_forces.values()
            )
            and not self._hs014_failure_latched
            and self._hs014_stage == 0
            and self._success_counter == 0
            and not any(self._read_joint_enabled(key) for key in self._hs014_joint_enabled)
        )
        return {
            "ready": ready,
            "finite": finite,
            "bottle_support_force_n": float(support_force),
            "bottle_support_penetration_m": float(support_penetration),
            "cap_bottle_force_n": float(cap_bottle_force),
            "cap_bottle_penetration_m": float(cap_bottle_penetration),
            "initial_hand_object_forces_n": initial_hand_forces,
            "failure_latched": bool(self._hs014_failure_latched),
            "bottle_awake": bool(bottle_awake),
            "cap_awake": bool(cap_awake),
            "awake_readback_available": bool(
                bottle_awake_available and cap_awake_available
            ),
            "awake_readback_errors": [
                error
                for error in (bottle_awake_error, cap_awake_error)
                if error
            ],
            "cap_seated_axial_position_m": (
                None if seated_geometry is None else float(seated_geometry[0])
            ),
            "cap_seated_radial_offset_m": (
                None if seated_geometry is None else float(seated_geometry[1])
            ),
            "cap_seated_axis_error_deg": (
                None if seated_geometry is None else float(seated_geometry[2])
            ),
            "task_stage": int(self._hs014_stage),
            "fixed_joints_disabled": bool(
                not any(self._hs014_joint_enabled.values())
            ),
        }

    def accept_stable_recording_origin(
        self,
        report: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        parent = super().accept_stable_recording_origin(report)
        payload = dict(parent) if isinstance(parent, Mapping) else {}
        actual = self._validate_task_initial_state_before_ready(
            commit_semantic_baseline=False
        )
        if actual is None:
            raise RuntimeError("HS014 stable origin lacks actual-state evidence")
        readiness = self.stable_recording_origin_task_readiness()
        if not bool(readiness.get("ready", False)):
            raise RuntimeError(
                "HS014 final stable-origin boundary is not task-ready: "
                f"{dict(readiness)}"
            )
        if not self._set_thread_joint_enabled(True):
            raise RuntimeError("HS014 cap thread joint could not be enabled before action 0")
        payload.update(
            {
                "accepted": True,
                "kind": "hs014_awake_bottle_support_and_actual_robot_boundary",
                "initial_state": dict(actual),
                "task_readiness": dict(readiness),
            }
        )
        return payload

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        # Keep the canonical dual-hand home clear of the tabletop.  HS014's
        # object reset already supplies bounded planar/yaw randomization, while
        # the inherited +5 cm table lift can physically push the open hands and
        # arms out of their authored action-0 state.
        self.delta_h[:] = 0.0
        self._hs014_reset_layout = sample_reset_layout(
            self._random,
            randomize=bool(self.randomize),
            table_delta_height=float(self.delta_h[0]),
        )
        self._hs014_bottle_baseline = self._hs014_reset_layout[BOTTLE_NAME].copy()
        self._hs014_authored_bottle_baseline = self._hs014_reset_layout[
            BOTTLE_NAME
        ].copy()
        self._hs014_reset_axial_position_m = _cap_axial_position_m(
            self._hs014_reset_layout[BOTTLE_NAME],
            self._hs014_reset_layout[CAP_NAME],
        )
        self._hs014_events = {
            "left_fixture_contact": False,
            "right_cap_contact": False,
            "continuous_twist_seen": False,
            "cap_axially_separated": False,
            "fixture_retained": False,
            "right_cap_retained": False,
        }
        self._hs014_failure_latched = False
        self._hs014_failure_reason = ""
        self._hs014_stage = 0
        self._hs014_last_metrics = {}
        self._hs014_joint_object_to_palm = {}
        self._hs014_previous_twist_reference = None
        self._hs014_accumulated_twist_rad = 0.0
        for key in self._hs014_contact_forces:
            self._hs014_contact_forces[key] = 0.0
        for key in self._hs014_joint_enabled:
            self._hs014_joint_enabled[key] = False
        self._hs014_thread_joint_enabled = False
        for key in self._hs014_contact_streaks:
            self._hs014_contact_streaks[key] = 0

    def _read_joint_enabled(self, key: str) -> bool:
        joint = self._hs014_joint_prims.get(key)
        if joint is None:
            self._hs014_joint_enabled[key] = False
            return False
        actual = bool(joint.GetJointEnabledAttr().Get())
        self._hs014_joint_enabled[key] = actual
        return actual

    def _set_joint_enabled(self, key: str, enabled: bool) -> bool:
        joint = self._hs014_joint_prims.get(key)
        if joint is None:
            self._hs014_joint_enabled[key] = False
            return False
        attribute = joint.GetJointEnabledAttr()
        requested = bool(enabled)
        attribute.Set(requested)
        actual = self._read_joint_enabled(key)
        return actual == requested

    def _reset_scene_state(self) -> None:
        # Hand ownership joints are reset disabled. Restore the engaged thread
        # after the authored reset pose is written.
        for key in tuple(self._hs014_joint_prims):
            self._set_joint_enabled(key, False)
        self._set_thread_joint_enabled(False)
        self._hs014_reset_contact_refresh_performed = False
        super()._reset_scene_state()
        self._refresh_hs014_reset_support_contacts()
        # Leave the thread gate disabled while the shared stable-origin audit
        # measures the authored cradle contact. It is engaged immediately
        # before action 0 in ``accept_stable_recording_origin``.

    def _refresh_hs014_reset_support_contacts(self) -> None:
        """Invalidate old table contact patches inside the reset transaction only.

        Teleporting a supported bottle laterally/yaw-wise can reuse the previous
        contact patch on the first solve and kick off the cap. Separate the pair
        from the table for one unrecorded setup step, then restore *authored*
        poses/zero velocities once. Never read a possibly stale post-setter pose
        as the reset target; never call this from the action/settling loop.
        Ordinary awake settling and all physical origin gates still follow.
        """
        if not self._task_body_views or self._torch is None or self._sim is None:
            return
        self._write_hs014_reset_poses(
            height_offset=RESET_CONTACT_CLEARANCE_M, journal=False
        )
        self._sim.step(render=False)
        self._write_hs014_reset_poses()
        self._hs014_reset_contact_refresh_performed = True

    def _apply_task_pose_randomization(self) -> None:
        # The shared path restores the table transaction. HS014 then writes its
        # exact bounded absolute poses before any post-reset physics step.
        super()._apply_task_pose_randomization()
        self._write_hs014_reset_poses()

    def _write_hs014_reset_poses(
        self, *, height_offset: float = 0.0, journal: bool = True
    ) -> None:
        if not self._task_body_views or self._torch is None:
            return
        for name, pose_wxyz in self._hs014_reset_layout.items():
            view = self._task_body_views.get(name)
            if view is None:
                raise RuntimeError(f"HS014 reset view is missing for {name}")
            transforms = view.get_transforms().clone()
            velocities = view.get_velocities().clone()
            pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7).copy()
            pose[2] += float(height_offset)
            transforms[:, :3] = self._torch.as_tensor(
                pose[:3], dtype=transforms.dtype, device=transforms.device
            )
            transforms[:, 3:7] = self._torch.as_tensor(
                [pose[4], pose[5], pose[6], pose[3]],
                dtype=transforms.dtype,
                device=transforms.device,
            )
            velocities.zero_()
            indices = self._torch.arange(
                int(view.count), dtype=self._torch.long, device=transforms.device
            )
            view.set_velocities(velocities, indices)
            view.set_transforms(transforms, indices)
            if journal:
                self._record_authored_task_reset_write(
                    str(self._find_rigid_body_prim(name).GetPath()),
                    transform_xyzw=transforms,
                    velocity=velocities,
                )

    def _contact_force(self, key: str) -> float:
        view = self._hs014_contact_views.get(key)
        if view is None:
            return 0.0
        try:
            matrix = view.get_contact_force_matrix(self._sim_dt)
            if hasattr(matrix, "detach"):
                matrix = matrix.detach().cpu().numpy()
            return float(np.linalg.norm(np.asarray(matrix, dtype=np.float64), axis=-1).sum())
        except Exception:
            return 0.0

    def _contact_reading(self, key: str) -> tuple[float, float]:
        """Return exact-pair normal force and worst solved penetration."""

        view = self._hs014_contact_views.get(key)
        if view is None:
            return 0.0, math.inf
        try:
            normal_forces, _points, _normals, separations, counts, starts = (
                view.get_contact_data(self._sim_dt)
            )

            def numpy(value: Any) -> np.ndarray:
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                return np.asarray(value)

            forces = numpy(normal_forces).astype(np.float64, copy=False).reshape(-1)
            distances = numpy(separations).astype(np.float64, copy=False).reshape(-1)
            shape = (int(view.sensor_count), int(view.filter_count))
            pair_counts = numpy(counts).astype(np.int64, copy=False).reshape(shape)
            pair_starts = numpy(starts).astype(np.int64, copy=False).reshape(shape)
            total_force = 0.0
            max_penetration = 0.0
            sampled_separations: list[float] = []
            for count, start in zip(pair_counts.ravel(), pair_starts.ravel()):
                if count <= 0:
                    continue
                stop = int(start + count)
                total_force += float(np.sum(np.abs(forces[int(start):stop])))
                sampled_separations.extend(
                    float(value) for value in distances[int(start):stop]
                )
                max_penetration = max(
                    max_penetration,
                    float(
                        np.max(
                            np.maximum(0.0, -distances[int(start):stop])
                        )
                    ),
                )
            self._hs014_contact_diagnostics[key] = {
                "contact_count": len(sampled_separations),
                "min_signed_separation_m": (
                    None if not sampled_separations else min(sampled_separations)
                ),
                "max_signed_separation_m": (
                    None if not sampled_separations else max(sampled_separations)
                ),
                "total_normal_force_n": float(total_force),
                "max_penetration_m": float(max_penetration),
            }
            if key == "cap_bottle":
                topology = self._hs014_runtime_physics_schema.get(
                    "collision_topology", {}
                )
                self._hs014_contact_diagnostics[key].update(
                    {
                        "contact_truth_topology": topology.get(
                            "contact_truth_topology"
                        ),
                        "unfiltered_contact_pairs": list(
                            topology.get("unfiltered_contact_pairs", [])
                        ),
                        "helper_physics_material_readback_passed": bool(
                            topology.get("helper_physics_material", {}).get(
                                "passed", False
                            )
                        ),
                        "root_contact_view_represents_three_sphere_cradle": bool(
                            topology.get("passed", False)
                        ),
                    }
                )
            return total_force, max_penetration
        except Exception as exc:
            self._hs014_contact_diagnostics[key] = {
                "error": f"{type(exc).__name__}: {exc}"
            }
            self._add_task_state_warning(
                f"HS014 contact readback {key!r} failed: {exc}"
            )
            return 0.0, math.inf

    def _hand_command(self, side: str) -> float:
        hand = self._left_hand if side == "left" else self._right_hand
        return float(np.max(np.abs(hand))) if hand.size else 0.0

    def _attach_joint_at_current_pose(self, key: str, side: str, name: str) -> bool:
        joint = self._hs014_joint_prims.get(key)
        object_pose = self._physics_body_pose(name)
        palm = self._native_robot_body_pose_matrix(
            side, f"{side}_palm_link"
        )
        if joint is None or object_pose is None or palm is None:
            return False
        from pxr import Gf

        relative = _invert_pose_matrix(_pose_vector_to_matrix(object_pose)) @ palm
        self._hs014_joint_object_to_palm[key] = relative.copy()
        quaternion = _matrix_to_quat_wxyz(relative[:3, :3])
        joint.GetLocalPos1Attr().Set(
            Gf.Vec3f(*[float(value) for value in relative[:3, 3]])
        )
        joint.GetLocalRot1Attr().Set(
            Gf.Quatf(
                float(quaternion[0]),
                Gf.Vec3f(*[float(value) for value in quaternion[1:4]]),
            )
        )
        return self._set_joint_enabled(key, True)

    def hs014_palm_pose_for_object_pose(
        self, key: str, object_pose_wxyz: np.ndarray
    ) -> np.ndarray | None:
        """Map a desired object pose to the action TCP using the contacted grasp."""

        relative = self._hs014_joint_object_to_palm.get(key)
        if relative is None:
            return None
        palm = _pose_vector_to_matrix(
            np.asarray(object_pose_wxyz, dtype=np.float64).reshape(7)
        ) @ np.asarray(relative, dtype=np.float64).reshape(4, 4)
        return np.concatenate(
            [palm[:3, 3], _matrix_to_quat_wxyz(palm[:3, :3])]
        ).astype(np.float64)

    def _relative_twist_reference(
        self, bottle: np.ndarray, cap: np.ndarray
    ) -> np.ndarray | None:
        """Project cap local +X into the bottle-local horizontal plane."""

        relative_rotation = (
            _quat_wxyz_to_matrix(bottle[3:7]).T
            @ _quat_wxyz_to_matrix(cap[3:7])
        )
        reference = np.asarray(relative_rotation[:2, 0], dtype=np.float64)
        norm = float(np.linalg.norm(reference))
        if not np.isfinite(norm) or norm <= 1.0e-8:
            return None
        return reference / norm

    def _update_continuous_twist(self, bottle: np.ndarray, cap: np.ndarray) -> None:
        if not (
            self._hs014_events.get("right_cap_contact", False)
            and self._hs014_joint_enabled.get("right_cap", False)
        ):
            self._hs014_previous_twist_reference = None
            return
        # Opening is an ordered milestone. Once the required twist has been
        # physically measured, reorienting the removed cap must not undo it.
        if self._hs014_events.get("continuous_twist_seen", False):
            return
        reference = self._relative_twist_reference(bottle, cap)
        if reference is None:
            self._hs014_previous_twist_reference = None
            return
        if self._hs014_previous_twist_reference is not None:
            previous = self._hs014_previous_twist_reference
            delta = math.atan2(
                float(previous[0] * reference[1] - previous[1] * reference[0]),
                float(np.dot(previous, reference)),
            )
            if abs(delta) <= MAX_TRACKED_TWIST_DELTA_RAD:
                # Signed integration makes reversals cancel instead of rewarding
                # back-and-forth wrist oscillation.
                self._hs014_accumulated_twist_rad += delta
            else:
                self._latch_failure("implausible_twist_discontinuity")
        self._hs014_previous_twist_reference = reference
        if abs(self._hs014_accumulated_twist_rad) + 1.0e-9 >= ROTATION_TARGET_RAD:
            self._hs014_events["continuous_twist_seen"] = True

    def _opening_truth(self) -> dict[str, Any]:
        bottle = self._physics_body_pose(BOTTLE_NAME)
        cap = self._physics_body_pose(CAP_NAME)
        if bottle is None:
            bottle = self._poses.get(BOTTLE_NAME, self._hs014_bottle_baseline)
        if cap is None:
            cap = self._poses.get(CAP_NAME, self._hs014_reset_layout.get(CAP_NAME, np.zeros(7)))
        bottle = np.asarray(bottle, dtype=np.float64).reshape(7)
        cap = np.asarray(cap, dtype=np.float64).reshape(7)
        bottle_rotation = _quat_wxyz_to_matrix(bottle[3:7])
        cap_rotation = _quat_wxyz_to_matrix(cap[3:7])
        bottle_axis = bottle_rotation[:, 2]
        cap_axis = cap_rotation[:, 1]
        axis_dot = float(np.clip(np.dot(bottle_axis, cap_axis), -1.0, 1.0))
        axis_error_deg = math.degrees(math.acos(axis_dot))
        axial_position = _cap_axial_position_m(bottle, cap)
        cap_displacement = cap[:3] - bottle[:3]
        radial_offset = float(
            np.linalg.norm(cap_displacement - axial_position * bottle_axis)
        )
        axial_lift = axial_position - float(self._hs014_reset_axial_position_m)
        baseline = np.asarray(self._hs014_bottle_baseline, dtype=np.float64)
        bottle_displacement = float(np.linalg.norm(bottle[:3] - baseline[:3]))
        baseline_rotation = _quat_wxyz_to_matrix(baseline[3:7])
        bottle_relative_rotation = baseline_rotation.T @ bottle_rotation
        orientation_cosine = float(
            np.clip((np.trace(bottle_relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
        )
        bottle_orientation_drift = math.degrees(math.acos(orientation_cosine))
        bottle_yaw_drift = abs(
            math.degrees(
                math.atan2(
                    float(bottle_relative_rotation[1, 0]),
                    float(bottle_relative_rotation[0, 0]),
                )
            )
        )
        bottle_tilt = math.degrees(
            math.acos(float(np.clip(np.dot(bottle_axis, [0.0, 0.0, 1.0]), -1.0, 1.0)))
        )
        cap_velocity = self._physics_body_velocity(CAP_NAME)
        cap_velocity = np.zeros(6, dtype=np.float64) if cap_velocity is None else cap_velocity
        cap_linear_speed = float(np.linalg.norm(cap_velocity[:3]))
        cap_angular_speed = float(np.linalg.norm(cap_velocity[3:]))
        return {
            "bottle_pose": bottle,
            "cap_pose": cap,
            "axis_error_deg": axis_error_deg,
            "cap_radial_offset_m": radial_offset,
            # Task semantics are displacement from the seeded reset baseline,
            # not the cap root's absolute offset from the bottle root.  The
            # authored support gap must not reduce the required 40 mm lift.
            "axial_separation_m": axial_lift,
            "axial_position_m": axial_position,
            "reset_axial_position_m": float(self._hs014_reset_axial_position_m),
            "accumulated_twist_rad": float(self._hs014_accumulated_twist_rad),
            "rotation_progress_rad": abs(float(self._hs014_accumulated_twist_rad)),
            "bottle_displacement_m": bottle_displacement,
            "bottle_orientation_drift_deg": bottle_orientation_drift,
            "bottle_yaw_drift_deg": bottle_yaw_drift,
            "bottle_tilt_deg": bottle_tilt,
            "cap_linear_speed_mps": cap_linear_speed,
            "cap_angular_speed_radps": cap_angular_speed,
            "cap_bottle_contact": bool(
                self._hs014_contact_forces["cap_bottle"] > CONTACT_FORCE_EPSILON
            ),
        }

    def _latch_failure(self, reason: str) -> None:
        if not self._hs014_failure_latched:
            self._hs014_failure_latched = True
            self._hs014_failure_reason = str(reason)

    def _update_role_contact_gate(
        self,
        *,
        key: str,
        event: str,
        side: str,
        object_name: str,
        candidate: bool,
    ) -> bool:
        """Latch a physical role only after stable contact and joint readback."""

        if self._hs014_events.get(event, False):
            if self._hs014_joint_enabled.get(key, False):
                return True
            self._latch_failure(f"{key}_joint_lost_after_ownership")
            return False
        if not candidate:
            self._hs014_contact_streaks[key] = 0
            return False
        streak = int(self._hs014_contact_streaks.get(key, 0)) + 1
        self._hs014_contact_streaks[key] = streak
        if streak < ROLE_CONTACT_HOLD_SUBSTEPS:
            return False
        if not self._hs014_joint_enabled.get(key, False):
            if not self._attach_joint_at_current_pose(key, side, object_name):
                return False
        if not self._hs014_joint_enabled.get(key, False):
            return False
        self._hs014_events[event] = True
        return True

    def _hs014_post_physics_substep(self, _index: int, _count: int) -> None:
        for key in self._hs014_joint_enabled:
            self._read_joint_enabled(key)
        self._read_thread_joint_enabled()
        for key in self._hs014_contact_forces:
            self._hs014_contact_forces[key] = self._contact_force(key)
        left_close = self._hand_command("left") >= HAND_CLOSE_COMMAND
        right_close = self._hand_command("right") >= HAND_CLOSE_COMMAND
        left_bottle = self._hs014_contact_forces["left_bottle"] > CONTACT_FORCE_EPSILON
        right_cap = self._hs014_contact_forces["right_cap"] > CONTACT_FORCE_EPSILON

        # Incidental touch is not wrong-hand ownership. In particular both
        # hands may arrive in the same physics step: qualify the left fixture
        # first, then allow the right-cap contact streak. Only the designated
        # hand/object pairs can attach; early/wrong-side touches give no credit.
        for key, event, close in (
            ("left_fixture", "left_fixture_contact", left_close),
            ("right_cap", "right_cap_contact", right_close),
        ):
            if not close and self._hs014_joint_enabled.get(key, False):
                if not self._set_joint_enabled(key, False):
                    self._latch_failure(f"{key}_release_failed")
                    continue
                self._hs014_events[event] = False
                self._hs014_joint_object_to_palm.pop(key, None)
                self._hs014_contact_streaks[key] = 0
                self._hs014_previous_twist_reference = None
                if not self._hs014_events.get("continuous_twist_seen", False):
                    self._hs014_accumulated_twist_rad = 0.0

        self._update_role_contact_gate(
            key="left_fixture",
            event="left_fixture_contact",
            side="left",
            object_name=BOTTLE_NAME,
            candidate=bool(left_bottle and left_close),
        )
        self._update_role_contact_gate(
            key="right_cap",
            event="right_cap_contact",
            side="right",
            object_name=CAP_NAME,
            candidate=bool(
                right_cap
                and right_close
                and self._hs014_events.get("left_fixture_contact", False)
            ),
        )

        metrics = self._opening_truth()
        if left_close and right_close and self._hs014_joint_enabled["left_fixture"]:
            self._update_continuous_twist(metrics["bottle_pose"], metrics["cap_pose"])
        else:
            self._hs014_previous_twist_reference = None
        # Release the engaged thread only after the physical signed twist has
        # crossed 180 degrees. Until then PhysX blocks axial separation, so a
        # grazing finger cannot knock the cap off the neck.
        if (
            self._hs014_events.get("continuous_twist_seen", False)
            and self._hs014_thread_joint_enabled
        ):
            if not self._set_thread_joint_enabled(False):
                self._latch_failure("cap_thread_release_failed")
        metrics = self._opening_truth()
        if (
            metrics["axial_separation_m"] >= AXIAL_SEPARATION_TARGET_M
            and self._hs014_events.get("right_cap_contact", False)
            and not self._hs014_events.get("continuous_twist_seen", False)
        ):
            self._latch_failure("cap_lifted_before_continuous_twist")
        if (
            self._hs014_events.get("continuous_twist_seen", False)
            and self._hs014_joint_enabled.get("right_cap", False)
            and metrics["axial_separation_m"] >= AXIAL_SEPARATION_TARGET_M
        ):
            self._hs014_events["cap_axially_separated"] = True
        self._hs014_events["fixture_retained"] = bool(
            self._hs014_events.get("left_fixture_contact", False)
            and self._hs014_joint_enabled["left_fixture"]
            and left_close
        )
        self._hs014_events["right_cap_retained"] = bool(
            self._hs014_events.get("right_cap_contact", False)
            and self._hs014_joint_enabled["right_cap"]
            and right_close
        )

        bottle = metrics["bottle_pose"]
        cap = metrics["cap_pose"]
        table_top = _TABLE_TOP_Z + float(self.delta_h[0])
        if metrics["bottle_tilt_deg"] > 30.0 or bottle[2] < table_top + 0.08:
            self._latch_failure("bottle_tipped_or_dropped")
        if cap[2] < table_top - 0.010:
            self._latch_failure("cap_dropped_below_table")
        if not (-0.58 <= cap[0] <= 0.10 and -0.42 <= cap[1] <= 0.42):
            self._latch_failure("cap_left_workspace")
        self._hs014_stage = sum(
            int(self._hs014_events.get(key, False))
            for key in (
                "left_fixture_contact",
                "right_cap_contact",
                "continuous_twist_seen",
                "cap_axially_separated",
            )
        )
        self._hs014_last_metrics = metrics

    def _success_steps_required(self) -> int:
        return SUCCESS_HOLD_STEPS

    def _success_predicates(self) -> dict[str, bool]:
        metrics = self._hs014_last_metrics or self._opening_truth()
        return {
            "left_fixture_retained": bool(self._hs014_events.get("fixture_retained", False)),
            "right_cap_retained": bool(self._hs014_events.get("right_cap_retained", False)),
            "continuous_twist_over_180_deg": bool(
                self._hs014_events.get("continuous_twist_seen", False)
                and metrics["rotation_progress_rad"] + 1.0e-9 >= ROTATION_TARGET_RAD
            ),
            "cap_axially_separated_40_mm": bool(
                self._hs014_events.get("cap_axially_separated", False)
                and metrics["axial_separation_m"] >= AXIAL_SEPARATION_TARGET_M
            ),
            "bottle_displacement_bounded": bool(
                metrics["bottle_displacement_m"] <= BOTTLE_DISPLACEMENT_LIMIT_M
            ),
            "bottle_orientation_bounded": bool(
                metrics["bottle_orientation_drift_deg"] <= BOTTLE_ORIENTATION_LIMIT_DEG
            ),
            "bottle_yaw_bounded": bool(
                metrics["bottle_yaw_drift_deg"] <= BOTTLE_YAW_DRIFT_LIMIT_DEG
            ),
            "order_and_failure_clear": bool(not self._hs014_failure_latched),
        }

    def _compute_success_now(self) -> bool:
        predicates = self._success_predicates()
        return bool(all(predicates.values()))

    def _compute_failure(self) -> bool:
        return bool(self._hs014_failure_latched or super()._compute_failure())

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        predicates = self._success_predicates()
        labels = {
            "left_fixture_retained": "Left hand physically fixtures the bottle",
            "right_cap_retained": "Right hand physically retains the free cap",
            "continuous_twist_over_180_deg": "Continuous signed cap rotation reaches 180 degrees",
            "cap_axially_separated_40_mm": "Cap is at least 40 mm above the bottle mouth",
            "bottle_displacement_bounded": "Bottle displacement is at most 20 mm",
            "bottle_orientation_bounded": "Bottle orientation drift is at most 5 degrees",
            "bottle_yaw_bounded": "Bottle yaw drift is at most 8 degrees",
            "order_and_failure_clear": "Twist-before-lift order is valid and no failure is latched",
        }
        rows = [
            self._condition_row(key, labels[key], value)
            for key, value in predicates.items()
        ]
        return self._append_success_hold_row(rows)

    def _task_status_text(self) -> str:
        metrics = self._hs014_last_metrics or {}
        blockers = [key for key, passed in self._success_predicates().items() if not passed]
        reason = self._hs014_failure_reason or ", ".join(blockers) or "hold steady"
        return (
            f"stage {self._hs014_stage}/4 | twist "
            f"{math.degrees(float(metrics.get('rotation_progress_rad', 0.0))):.1f} deg | "
            f"separation {1000.0 * float(metrics.get('axial_separation_m', 0.0)):.1f} mm | "
            f"hold {self._success_counter}/{SUCCESS_HOLD_STEPS} | {reason}"
        )

    def _relative_cap_to_bottle(self, bottle: np.ndarray, cap: np.ndarray) -> np.ndarray:
        relative = _invert_pose_matrix(_pose_vector_to_matrix(bottle)) @ _pose_vector_to_matrix(cap)
        return np.concatenate(
            [relative[:3, 3], _matrix_to_quat_wxyz(relative[:3, :3])]
        ).astype(np.float64)

    def _compute_observation(self) -> dict[str, Any]:
        observation = super()._compute_observation()
        metrics = self._hs014_last_metrics or self._opening_truth()
        bottle = np.asarray(metrics["bottle_pose"], dtype=np.float64).copy()
        cap = np.asarray(metrics["cap_pose"], dtype=np.float64).copy()
        predicates = self._success_predicates()
        event_order = (
            "left_fixture_contact",
            "right_cap_contact",
            "continuous_twist_seen",
            "cap_axially_separated",
            "fixture_retained",
            "right_cap_retained",
        )
        role_state = np.asarray(
            [float(bool(self._hs014_events.get(key, False))) for key in event_order],
            dtype=np.float64,
        )
        opening = np.asarray(
            [
                metrics["accumulated_twist_rad"],
                metrics["axial_separation_m"],
                metrics["bottle_displacement_m"],
                metrics["bottle_orientation_drift_deg"],
                metrics["bottle_yaw_drift_deg"],
                metrics["cap_linear_speed_mps"],
                metrics["cap_angular_speed_radps"],
                float(self._success_counter),
            ],
            dtype=np.float64,
        )
        observation["state"].update(
            {
                "bottle_pose": bottle,
                "cap_pose": cap,
                "cap_to_bottle": self._relative_cap_to_bottle(bottle, cap),
                "opening_metrics": opening,
                "fixture_role_state": role_state,
                "task_stage": np.asarray([self._hs014_stage], dtype=np.float64),
            }
        )
        observation["hs014_hidden_state"] = np.asarray(
            [
                *role_state,
                float(self._hs014_failure_latched),
                float(self._hs014_joint_enabled["left_fixture"]),
                float(self._hs014_joint_enabled["right_cap"]),
                self._hs014_contact_forces["left_bottle"],
                self._hs014_contact_forces["right_cap"],
                self._hs014_contact_forces["left_cap"],
                self._hs014_contact_forces["right_bottle"],
                float(self._hs014_stage),
                float(self._success_counter),
                float(getattr(self, "_trajectory_state_write_count", 0)),
            ],
            dtype=np.float64,
        )
        observation["hs014_success_truth"] = np.asarray(
            [float(value) for value in predicates.values()], dtype=np.float64
        )
        return observation

    def _camera_visibility_truth(self) -> dict[str, Any]:
        metrics = self._hs014_last_metrics or self._opening_truth()
        bottle = np.asarray(metrics["bottle_pose"], dtype=np.float64)
        cap = np.asarray(metrics["cap_pose"], dtype=np.float64)
        bottle_axis = _quat_wxyz_to_matrix(bottle[3:7])[:, 2]
        targets = {
            "front": np.stack([bottle[:3] - 0.10 * bottle_axis, cap[:3]]),
            "wrist_left": np.stack([bottle[:3] - 0.10 * bottle_axis]),
            "wrist_right": np.stack([cap[:3]]),
        }
        sizes = {
            "front": (640, 480),
            "wrist_left": (848, 480),
            "wrist_right": (848, 480),
        }
        result: dict[str, Any] = {}
        for key, points in targets.items():
            width, height = sizes[key]
            pose = self._recording_camera_pose_matrix_for_key(key)
            pixels = None
            if pose is not None:
                pixels = self._project_opengl_camera_points(
                    pose,
                    points,
                    image_width=width,
                    image_height=height,
                    camera_key=key,
                )
            inside = bool(
                pixels is not None
                and all(0 <= x < width and 0 <= y < height for x, y in pixels)
            )
            result[key] = {
                "required_target_count": int(len(points)),
                "projected_pixels": (
                    []
                    if pixels is None
                    else [list(item) for item in pixels]
                ),
                "targets_in_frame": inside,
            }
        return result

    def _task_info(self) -> dict[str, Any]:
        metrics = dict(self._hs014_last_metrics or self._opening_truth())
        for key in ("bottle_pose", "cap_pose"):
            value = metrics.get(key)
            if isinstance(value, np.ndarray):
                metrics[key] = value.tolist()
        return {
            "hs014_reset_contact_refresh": {
                "performed": self._hs014_reset_contact_refresh_performed,
                "setup_steps": int(self._hs014_reset_contact_refresh_performed),
                "clearance_m": RESET_CONTACT_CLEARANCE_M,
                "scope": "reset_only_before_authored_origin_and_action_zero",
            },
            "hs014_stage": int(self._hs014_stage),
            "hs014_events": dict(self._hs014_events),
            "hs014_failure_latched": bool(self._hs014_failure_latched),
            "hs014_failure_reason": self._hs014_failure_reason,
            "hs014_joint_enabled": dict(self._hs014_joint_enabled),
            "hs014_cap_thread": {
                "joint_path": "/World/HS014RuntimeJoints/cap_thread",
                "joint_type": "RevoluteJoint",
                "pitch_m": THREAD_PITCH_M,
                "rotation_release_rad": ROTATION_TARGET_RAD,
                "enabled": bool(self._hs014_thread_joint_enabled),
            },
            "hs014_contact_streaks": dict(self._hs014_contact_streaks),
            "hs014_contact_forces": dict(self._hs014_contact_forces),
            "hs014_success_predicates": self._success_predicates(),
            "hs014_success_blockers": [
                key for key, passed in self._success_predicates().items() if not passed
            ],
            "hs014_success_counter": int(self._success_counter),
            "hs014_success_steps_required": SUCCESS_HOLD_STEPS,
            "hs014_metrics": metrics,
            "hs014_source_contract": dict(self._hs014_source_contract),
            "hs014_runtime_physics_schema": dict(
                self._hs014_runtime_physics_schema
            ),
            "hs014_reset_axial_position_m": float(
                self._hs014_reset_axial_position_m
            ),
            "hs014_reset_layout": {
                key: value.tolist() for key, value in self._hs014_reset_layout.items()
            },
            "hs014_initial_state_report": dict(
                self._hs014_initial_state_report
            ),
            "hs014_camera_visibility": self._camera_visibility_truth(),
            "hs014_trajectory_state_write_count": int(
                getattr(self, "_trajectory_state_write_count", 0)
            ),
        }

    def get_physical_replay_reset_contract(self) -> dict[str, Any]:
        contract = dict(super().get_physical_replay_reset_contract())
        contract.update(
            {
                "strategy": "fresh_physx_process_seeded_hs014_reset_then_exact_actions",
                "requires_fresh_process": True,
                "allow_pre_action_state_overwrite": False,
                "allows_single_recorded_t0_restore": False,
                "forbids_state_writes_after_action_start": True,
                "task_hidden_state_field": "hs014_hidden_state",
                "success_truth_field": "hs014_success_truth",
                "axial_lift_reference": "seeded_reset_cap_to_bottle_axis_position",
                "ordered_role_events": [
                    "left_fixture_contact",
                    "right_cap_contact",
                    "continuous_twist_seen",
                    "cap_axially_separated",
                    "fixture_retained",
                    "right_cap_retained",
                ],
            }
        )
        return contract
