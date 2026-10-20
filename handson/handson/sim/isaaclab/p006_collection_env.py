"""Native Isaac Sim 5.1 / PhysX runtime for P0-06."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from gymnasium import spaces
import numpy as np

from .abc_collection_env import HandsOnABCIsaacLabEnv
from .priority_zero_robot_audit import initial_robot_state_report, pose_error_wxyz
from .p006_contract import (
    ASSET_DIR_RELATIVE_PATH,
    CONTROL_HZ,
    FEMALE_ASSET_SHA256,
    FEMALE_ROOT_PATH,
    MALE_ASSET_SHA256,
    MALE_ROOT_PATH,
    MALE_TONGUE_PATHS,
    GRIP_COLLIDER_PATHS,
    GRIP_STATIC_FRICTION,
    GRIP_DYNAMIC_FRICTION,
    MIN_INSERTION_DEPTH_M,
    MIN_BODY_CONTACT_FORCE_N,
    MIN_ROLE_CONTACT_FORCE_N,
    MAX_TRANSVERSE_OFFSET_M,
    MAX_AXIS_ERROR_DEG,
    MAX_PENETRATION_M,
    NOMINAL_FEMALE_Y_M,
    NOMINAL_MALE_Y_M,
    NOMINAL_ROOT_Z_M,
    NOMINAL_YAW_DEG,
    PHYSICS_HZ,
    P006LatchStateMachine,
    P006PhysicalReadback,
    RENDER_RESOLUTION_HW,
    SUCCESS_HOLD_STEPS,
    TASK_NAME,
    TIMEOUT_STEPS,
    physical_metrics,
    sample_reset_state,
)


_LIGHT_ROOT = "/World/Task/P006Lights"
_MATERIAL_PATH = "/World/PhysicsMaterials/P006Buckle"
_GRIP_MATERIAL_PATH = "/World/PhysicsMaterials/P006BuckleGrip"
_AUTHORED_RESET_TRANSLATION_LIMIT_M = 0.005
_AUTHORED_RESET_ROTATION_LIMIT_RAD = math.radians(3.0)
_PENETRATION_FAILURE_DWELL_SECONDS = 0.20
_P006_RUNTIME_CONTACT_KEYS = (
    "right_female",
    "left_male",
    "left_male_tongue_neg",
    "left_male_tongue_pos",
    "buckle_body",
    "robot_table",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class HandsOnP006IsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Role-specific bimanual alignment and press of a side-release buckle."""

    def __init__(self, task_name: str = TASK_NAME, *args: Any, **kwargs: Any):
        if task_name != TASK_NAME:
            raise ValueError(f"P0-06 runtime requires {TASK_NAME!r}, got {task_name!r}")
        physics_backend = str(kwargs.get("physics_backend", "physx")).lower()
        if physics_backend not in {"physx", "isaacsim", "isaac"}:
            raise ValueError("P0-06 requires Isaac Sim 5.1 PhysX")
        kwargs["physics_backend"] = "physx"
        kwargs.setdefault("physics_dt", 1.0 / PHYSICS_HZ)
        kwargs.setdefault("control_dt", 1.0 / CONTROL_HZ)
        kwargs.setdefault("hz", CONTROL_HZ)
        kwargs.setdefault("time_limit", TIMEOUT_STEPS / CONTROL_HZ)
        kwargs.setdefault("render_resolution", RENDER_RESOLUTION_HW)

        requested_seed = 0 if kwargs.get("seed") is None else int(kwargs["seed"])
        self._p006_reset_serial = -1
        self._p006_reset = sample_reset_state(requested_seed)
        self._p006_body_views: dict[str, Any] = {}
        self._p006_contact_views: dict[str, Any] = {}
        self._p006_contact_paths: dict[str, tuple[str, tuple[str, ...]]] = {}
        self._p006_latch = P006LatchStateMachine()
        self._p006_last_readback: P006PhysicalReadback | None = None
        self._p006_last_metrics: dict[str, Any] = {}
        self._p006_failure_latched = False
        self._p006_failure_reason = ""
        self._p006_first_failure_snapshot: dict[str, Any] = {}
        self._p006_last_contact_readings: dict[str, tuple[float, float]] = {}
        self._p006_penetration_peak_m: dict[str, float] = {}
        self._p006_penetration_violation_substeps = 0
        self._p006_penetration_failure_dwell_substeps = max(
            1,
            int(
                math.ceil(
                    _PENETRATION_FAILURE_DWELL_SECONDS
                    / float(kwargs["physics_dt"])
                )
            ),
        )
        self._p006_contact_contract_ready = False
        self._p006_initial_state_report: dict[str, Any] = {}
        super().__init__(task_name, *args, **kwargs)

    def _ready_state_settle_physics_steps(self) -> int:
        """Settle free task bodies for one real second before ready-state checks."""

        return max(5, int(round(1.0 / float(self._physics_dt))))

    def _authored_reset_pose_report(
        self,
        readback: P006PhysicalReadback,
    ) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        for label, position, yaw_deg, actual_position, actual_quaternion in (
            (
                "female",
                self._p006_reset.female_position_m,
                self._p006_reset.female_yaw_deg,
                readback.female_position_m,
                readback.female_quaternion_wxyz,
            ),
            (
                "male",
                self._p006_reset.male_position_m,
                self._p006_reset.male_yaw_deg,
                readback.male_position_m,
                readback.male_quaternion_wxyz,
            ),
        ):
            half_yaw = math.radians(float(yaw_deg)) * 0.5
            target = np.asarray(
                [*position, math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
                dtype=np.float64,
            )
            actual = np.asarray(
                [*actual_position, *actual_quaternion], dtype=np.float64
            )
            finite, translation_error, rotation_error = pose_error_wxyz(
                actual, target
            )
            rows[label] = {
                "passed": bool(
                    finite
                    and translation_error <= _AUTHORED_RESET_TRANSLATION_LIMIT_M
                    and rotation_error <= _AUTHORED_RESET_ROTATION_LIMIT_RAD
                ),
                "target_pose_wxyz": target.tolist(),
                "actual_pose_wxyz": actual.tolist(),
                "translation_error_m": float(translation_error),
                "rotation_error_deg": math.degrees(rotation_error),
            }
        return {
            "passed": all(bool(row["passed"]) for row in rows.values()),
            "bodies": rows,
        }

    def _build_observation_space(self) -> spaces.Dict:
        observation_space = super()._build_observation_space()
        state_spaces = observation_space.spaces["state"].spaces
        state_spaces.update(
            {
                "female_buckle_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "male_buckle_pose": spaces.Box(-np.inf, np.inf, shape=(7,)),
                "p006_alignment_state": spaces.Box(-np.inf, np.inf, shape=(8,)),
                "p006_role_contacts": spaces.Box(-np.inf, np.inf, shape=(4,)),
                "p006_latch_state": spaces.Box(-np.inf, np.inf, shape=(8,)),
            }
        )
        return observation_space

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        if self._stage is None:
            raise RuntimeError("P0-06 stage is unavailable before asset composition")
        from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics

        repository_root = Path(__file__).resolve().parents[4]
        asset_dir = repository_root / ASSET_DIR_RELATIVE_PATH
        asset_specs = (
            (
                "female",
                asset_dir / "female_buckle.usda",
                FEMALE_ASSET_SHA256,
                FEMALE_ROOT_PATH,
                (float(-0.18), float(NOMINAL_FEMALE_Y_M), float(NOMINAL_ROOT_Z_M)),
            ),
            (
                "male",
                asset_dir / "male_buckle.usda",
                MALE_ASSET_SHA256,
                MALE_ROOT_PATH,
                (float(-0.18), float(NOMINAL_MALE_Y_M), float(NOMINAL_ROOT_Z_M)),
            ),
        )
        for label, asset_path, expected_sha, prim_path, position in asset_specs:
            if not asset_path.is_file():
                raise FileNotFoundError(f"P0-06 {label} asset is missing: {asset_path}")
            actual_sha = _sha256(asset_path)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"P0-06 {label} asset SHA mismatch: expected {expected_sha}, got {actual_sha}"
                )
            root = UsdGeom.Xform.Define(self._stage, prim_path).GetPrim()
            root.GetReferences().ClearReferences()
            root.GetReferences().AddReference(str(asset_path.resolve()))
            xform = UsdGeom.Xformable(root)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*position))
            xform.AddRotateZOp(UsdGeom.XformOp.PrecisionFloat).Set(float(NOMINAL_YAW_DEG))
            if not root.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"P0-06 {label} reference did not compose a rigid root")
            PhysxSchema.PhysxContactReportAPI.Apply(root).CreateThresholdAttr(0.0)
            rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
            rigid_api.CreateMaxLinearVelocityAttr(3.0)
            rigid_api.CreateMaxAngularVelocityAttr(15.0)

        table = self._find_rigid_body_prim("table")
        if table is None:
            raise RuntimeError("P0-06 could not find the imported table rigid body")
        PhysxSchema.PhysxContactReportAPI.Apply(table).CreateThresholdAttr(0.0)

        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.75,
            dynamic_friction=0.62,
            restitution=0.0,
            friction_combine_mode="average",
            restitution_combine_mode="min",
        )
        material_cfg.func(_MATERIAL_PATH, material_cfg)
        for prim_path in (FEMALE_ROOT_PATH, MALE_ROOT_PATH):
            sim_utils.bind_physics_material(
                prim_path,
                _MATERIAL_PATH,
                stage=self._stage,
                stronger_than_descendants=False,
            )
        grip_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=GRIP_STATIC_FRICTION,
            dynamic_friction=GRIP_DYNAMIC_FRICTION,
            restitution=0.0,
            friction_combine_mode="average",
            restitution_combine_mode="min",
        )
        grip_cfg.func(_GRIP_MATERIAL_PATH, grip_cfg)
        for path in GRIP_COLLIDER_PATHS:
            if not self._stage.GetPrimAtPath(path).HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError(f"P0-06 grasp collider is missing: {path}")
            sim_utils.bind_physics_material(
                path, _GRIP_MATERIAL_PATH,
                stage=self._stage, stronger_than_descendants=True,
            )

        dome = UsdLux.DomeLight.Define(self._stage, f"{_LIGHT_ROOT}/Dome")
        dome.CreateIntensityAttr(650.0)
        dome.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
        key = UsdLux.SphereLight.Define(self._stage, f"{_LIGHT_ROOT}/Key")
        key.CreateIntensityAttr(1900.0)
        key.CreateRadiusAttr(0.16)
        key.AddTranslateOp().Set(Gf.Vec3d(-0.25, -0.20, 1.85))

    def _robot_rigid_body_paths(self, side: str, *, hand_only: bool) -> list[str]:
        if self._stage is None:
            return []
        from pxr import Usd, UsdPhysics

        root = self._stage.GetPrimAtPath(f"/World/HandsOnRobots/{side.capitalize()}Robot")
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
            raise RuntimeError(f"P0-06 contact filter {key} resolved no bodies")
        view = physics_view.create_rigid_contact_view(
            sensor_path,
            filter_patterns=filter_paths,
            max_contact_data_count=max(128, len(filter_paths) * 16),
        )
        try:
            complete = bool(
                view is not None
                and int(view.sensor_count) == 1
                and int(view.filter_count) == len(filter_paths)
            )
        except (AttributeError, RuntimeError):
            complete = False
        if not complete:
            raise RuntimeError(f"P0-06 contact view {key} is empty or incomplete")
        self._p006_contact_views[key] = view
        self._p006_contact_paths[key] = (sensor_path, tuple(filter_paths))

    def _init_task_specific_runtime(self) -> None:
        if not self._robots:
            return
        physics_view = next(iter(self._robots.values()))._physics_sim_view
        if physics_view is None:
            raise RuntimeError("P0-06 PhysX tensor simulation view is unavailable")
        self._p006_body_views = {
            "female": physics_view.create_rigid_body_view(FEMALE_ROOT_PATH),
            "male": physics_view.create_rigid_body_view(MALE_ROOT_PATH),
        }
        if any(int(view.count) != 1 for view in self._p006_body_views.values()):
            raise RuntimeError("P0-06 female or male rigid-body view is empty")

        left_hand = self._robot_rigid_body_paths("left", hand_only=True)
        right_hand = self._robot_rigid_body_paths("right", hand_only=True)
        all_robot = self._robot_rigid_body_paths("left", hand_only=False)
        all_robot.extend(self._robot_rigid_body_paths("right", hand_only=False))
        table = self._find_rigid_body_prim("table")
        if table is None:
            raise RuntimeError("P0-06 table rigid body is unavailable")
        self._create_contact_view(
            physics_view, "right_female", FEMALE_ROOT_PATH, right_hand
        )
        self._create_contact_view(
            physics_view, "left_male", MALE_ROOT_PATH, left_hand
        )
        for suffix, path in zip(("neg", "pos"), MALE_TONGUE_PATHS):
            view = physics_view.create_rigid_body_view(path)
            if int(view.count) != 1:
                raise RuntimeError(f"P0-06 articulated tongue view is missing: {path}")
            self._p006_body_views[f"tongue_{suffix}"] = view
            self._create_contact_view(
                physics_view, f"left_male_tongue_{suffix}", path, left_hand
            )
        self._create_contact_view(
            physics_view, "buckle_body", FEMALE_ROOT_PATH,
            [MALE_ROOT_PATH, *MALE_TONGUE_PATHS]
        )
        self._create_contact_view(
            physics_view, "robot_table", str(table.GetPath()), all_robot
        )
        self._create_contact_view(
            physics_view, "female_support", FEMALE_ROOT_PATH, [str(table.GetPath())]
        )
        self._create_contact_view(
            physics_view, "male_support", MALE_ROOT_PATH, [str(table.GetPath())]
        )
        self._p006_contact_contract_ready = True
        print(
            "[HandsOn/P006] physical contract ready: "
            f"left_hand={len(left_hand)}, right_hand={len(right_hand)}",
            flush=True,
        )

    def _capture_task_scene_state(self, task_mjcf: Path) -> None:
        """Capture the authored female buckle through its exact runtime view.

        P0-06 composes the female buckle from a referenced USD after the MJCF
        task has been imported. Isaac Sim can therefore expose the root to a
        direct ``RigidBodyView`` while the generic name-based scene scan misses
        it. Without a cached reset state the seeded pose write is skipped and
        the initial-state audit aborts before teleoperation is armed, leaving
        the robots visibly parked at their side-raised home pose.
        """

        super()._capture_task_scene_state(task_mjcf)
        cached_paths = {
            path for path, *_rest in self._task_reset_body_states
        }
        if self._torch is None or FEMALE_ROOT_PATH in cached_paths:
            return

        view = self._p006_body_views.get("female")
        if view is None:
            self._add_task_state_warning(
                f"P0-06 exact female rigid-body view is unavailable: {FEMALE_ROOT_PATH}"
            )
            return
        try:
            count = int(view.count)
            if count < 1:
                raise RuntimeError("view is empty")
            indices = self._torch.arange(
                count,
                dtype=self._torch.long,
                device=self._isaac_device,
            )
            root_pose = view.get_transforms().clone()
            root_velocity = view.get_velocities().clone()
            self._task_reset_body_states.append(
                (
                    FEMALE_ROOT_PATH,
                    view,
                    indices,
                    root_pose,
                    root_velocity,
                )
            )
            # Keep the logical aliases available to generic pose/telemetry
            # helpers even when the USD name scan did not resolve this root.
            self._task_body_views.setdefault("p006_female_buckle", view)
            self._task_body_views.setdefault("female_buckle", view)
            print(
                "[HandsOn/P006] Captured exact female buckle reset state "
                f"through {FEMALE_ROOT_PATH}",
                flush=True,
            )
        except Exception as exc:
            self._add_task_state_warning(
                f"Could not snapshot exact female buckle {FEMALE_ROOT_PATH}: {exc}"
            )

    def _validate_task_initial_state_before_ready(self) -> Mapping[str, Any] | None:
        base_report = super()._validate_task_initial_state_before_ready()
        errors: list[str] = []
        readback = self._physical_readback()
        values = np.concatenate(
            [
                np.asarray(readback.female_position_m),
                np.asarray(readback.female_quaternion_wxyz),
                np.asarray(readback.male_position_m),
                np.asarray(readback.male_quaternion_wxyz),
                np.asarray(readback.female_linear_velocity_mps),
                np.asarray(readback.female_angular_velocity_radps),
                np.asarray(readback.male_linear_velocity_mps),
                np.asarray(readback.male_angular_velocity_radps),
            ]
        )
        female_force, female_penetration = self._contact_reading(
            "female_support"
        )
        male_force, male_penetration = self._contact_reading("male_support")
        robot_pass, robot_report = initial_robot_state_report(self)
        authored_pose_report = self._authored_reset_pose_report(readback)
        if not np.all(np.isfinite(values)):
            errors.append("actual buckle state is unavailable or non-finite")
        for label, force, penetration in (
            ("female", female_force, female_penetration),
            ("male", male_force, male_penetration),
        ):
            if force < 0.01:
                errors.append(
                    f"{label} buckle has no measured table support force: "
                    f"{force:.6f}N"
                )
            if not np.isfinite(penetration) or penetration > 0.0005:
                errors.append(
                    f"{label} buckle support penetration is unavailable or "
                    f"exceeds 0.5 mm: {penetration:.6f}m"
                )
        for label, linear, angular in (
            (
                "female",
                readback.female_linear_velocity_mps,
                readback.female_angular_velocity_radps,
            ),
            (
                "male",
                readback.male_linear_velocity_mps,
                readback.male_angular_velocity_radps,
            ),
        ):
            linear_speed = float(np.linalg.norm(np.asarray(linear, dtype=np.float64)))
            angular_speed = float(np.linalg.norm(np.asarray(angular, dtype=np.float64)))
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
        if not robot_pass:
            errors.append("actual dual-ROKAE initial state failed the P0 audit")
        if not bool(authored_pose_report["passed"]):
            errors.append("actual buckle poses differ from this seed's authored reset")
        if self._p006_failure_latched:
            errors.append("task failure latched during pre-ready settling")
        if self._p006_latch.success_hold_steps >= SUCCESS_HOLD_STEPS:
            errors.append("task reached success during pre-ready settling")
        if errors:
            raise RuntimeError(
                "P0-06 initial physical-state audit failed before ready: "
                + "; ".join(errors)
            )
        report = {
            "female_support_force_n": float(female_force),
            "female_support_penetration_m": float(female_penetration),
            "male_support_force_n": float(male_force),
            "male_support_penetration_m": float(male_penetration),
            "actual_robot_state": robot_report,
            "authored_reset_pose": authored_pose_report,
        }
        self._p006_initial_state_report = report
        if isinstance(base_report, Mapping):
            return {**base_report, **report}
        return report

    def stable_recording_origin_task_readiness(self) -> Mapping[str, Any]:
        readback = self._physical_readback()
        metrics = physical_metrics(readback)
        authored_pose_report = self._authored_reset_pose_report(readback)
        female_force, female_penetration = self._contact_reading(
            "female_support"
        )
        male_force, male_penetration = self._contact_reading("male_support")
        ready = bool(
            metrics["finite"]
            and bool(authored_pose_report["passed"])
            and female_force >= 0.01
            and male_force >= 0.01
            and np.isfinite(female_penetration)
            and np.isfinite(male_penetration)
            and max(female_penetration, male_penetration) <= 0.0005
            and not self._p006_failure_latched
            and self._p006_latch.success_hold_steps < SUCCESS_HOLD_STEPS
        )
        return {
            "ready": ready,
            "finite": bool(metrics["finite"]),
            "female_support_force_n": float(female_force),
            "female_support_penetration_m": float(female_penetration),
            "male_support_force_n": float(male_force),
            "male_support_penetration_m": float(male_penetration),
            "failure_latched": bool(self._p006_failure_latched),
            "authored_reset_pose": authored_pose_report,
        }

    def accept_stable_recording_origin(
        self,
        report: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        parent = super().accept_stable_recording_origin(report)
        payload = dict(parent) if isinstance(parent, Mapping) else {}
        actual = self._validate_task_initial_state_before_ready()
        if actual is None:
            raise RuntimeError("P0-06 stable origin lacks actual-state evidence")
        payload.update(
            {
                "accepted": True,
                "kind": "p006_awake_dual_support_and_actual_robot_boundary",
                "initial_state": dict(actual),
            }
        )
        return payload

    @staticmethod
    def _view_numpy(value: Any, shape: tuple[int, ...]) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64).reshape(shape)

    def reset(self, seed: int | None = None, **kwargs: Any):
        if seed is not None:
            self._p006_reset_serial = -1
        return super().reset(seed=seed, **kwargs)

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self.delta_h[:] = 0.0
        self._p006_reset_serial += 1
        episode_seed = int(self._seed) + 104729 * self._p006_reset_serial
        self._p006_reset = sample_reset_state(episode_seed)
        self._p006_latch.reset()
        self._p006_failure_latched = False
        self._p006_failure_reason = ""
        self._p006_first_failure_snapshot = {}
        self._p006_last_contact_readings = {}
        self._p006_penetration_peak_m = {}
        self._p006_penetration_violation_substeps = 0
        self._p006_last_readback = None
        self._p006_last_metrics = {}

    @staticmethod
    def _pose_xyzw(position: tuple[float, float, float], yaw_deg: float) -> np.ndarray:
        half = math.radians(float(yaw_deg)) * 0.5
        return np.asarray(
            [*position, 0.0, 0.0, math.sin(half), math.cos(half)],
            dtype=np.float64,
        )

    def _apply_task_pose_randomization(self) -> None:
        if self._torch is None:
            return
        desired = {
            FEMALE_ROOT_PATH: self._pose_xyzw(
                self._p006_reset.female_position_m,
                self._p006_reset.female_yaw_deg,
            ),
            MALE_ROOT_PATH: self._pose_xyzw(
                self._p006_reset.male_position_m,
                self._p006_reset.male_yaw_deg,
            ),
        }
        found = set()

        # The male buckle is an articulation after the outer tongues are
        # split into prismatic links.  Its root must be randomized through the
        # reduced-coordinate view; writing a rigid-body view for an
        # articulation root would either be ignored or reset the links to an
        # invalid pose.  Keep the two tongue DOFs at their authored neutral
        # position for every seeded reset.
        male_articulation = next(
            (
                value
                for value in self._task_reset_articulation_states
                if value[0] == MALE_ROOT_PATH
            ),
            None,
        )
        if male_articulation is not None:
            (
                path,
                view,
                indices,
                root_pose,
                root_velocity,
                joint_pos,
                joint_velocity,
            ) = male_articulation
            pose = root_pose.clone()
            pose_np = self._view_numpy(pose, (int(view.count), 7))
            pose_np[0] = self._pose_xyzw(
                self._p006_reset.male_position_m,
                self._p006_reset.male_yaw_deg,
            )
            pose = self._torch.as_tensor(
                pose_np, dtype=root_pose.dtype, device=root_pose.device
            )
            neutral_dofs = self._torch.zeros_like(joint_pos)
            zero_root_velocity = self._torch.zeros_like(root_velocity)
            zero_joint_velocity = self._torch.zeros_like(joint_velocity)
            view.set_root_transforms(pose, indices)
            view.set_root_velocities(zero_root_velocity, indices)
            view.set_dof_positions(neutral_dofs, indices)
            view.set_dof_velocities(zero_joint_velocity, indices)
            self._record_authored_task_reset_write(
                path,
                transform_xyzw=pose,
                velocity=zero_root_velocity,
                dof_position=neutral_dofs,
                dof_velocity=zero_joint_velocity,
            )
            found.add(MALE_ROOT_PATH)

        for path, view, indices, root_pose, root_velocity in self._task_reset_body_states:
            target = desired.get(path)
            if target is None:
                continue
            found.add(path)
            pose = root_pose.clone()
            pose_np = self._view_numpy(pose, (int(view.count), 7))
            pose_np[0] = target
            pose = self._torch.as_tensor(
                pose_np, dtype=root_pose.dtype, device=root_pose.device
            )
            velocity = self._torch.zeros_like(root_velocity)
            view.set_transforms(pose, indices)
            view.set_velocities(velocity, indices)
            self._record_authored_task_reset_write(
                path,
                transform_xyzw=pose,
                velocity=velocity,
            )
        missing = set(desired) - found
        if missing:
            self._add_task_state_warning(
                f"P0-06 rigid reset states are unavailable: {sorted(missing)}"
            )
        print(
            "[HandsOn/P006] seeded reset: "
            f"seed={self._p006_reset.seed}, "
            f"female={self._p006_reset.female_position_m}, "
            f"male={self._p006_reset.male_position_m}",
            flush=True,
        )

    def _contact_reading(self, key: str) -> tuple[float, float]:
        view = self._p006_contact_views.get(key)
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
            self._add_task_state_warning(f"P0-06 contact readback {key} failed: {exc}")
            return 0.0, math.inf

    def _body_state(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        view = self._p006_body_views.get(key)
        if view is None:
            return np.zeros(7, dtype=np.float64), np.zeros(6, dtype=np.float64)
        transform = self._view_numpy(view.get_transforms()[0], (7,))
        velocity = self._view_numpy(view.get_velocities()[0], (6,))
        pose_wxyz = np.concatenate([transform[:3], transform[6:7], transform[3:6]])
        return pose_wxyz, velocity

    def _physical_readback(self) -> P006PhysicalReadback:
        female_pose, female_velocity = self._body_state("female")
        male_pose, male_velocity = self._body_state("male")
        readings = {
            key: self._contact_reading(key) for key in _P006_RUNTIME_CONTACT_KEYS
        }
        self._p006_last_contact_readings = readings
        for key, (_force, penetration) in readings.items():
            if math.isfinite(penetration):
                self._p006_penetration_peak_m[key] = max(
                    self._p006_penetration_peak_m.get(key, 0.0),
                    float(penetration),
                )
        return P006PhysicalReadback(
            female_position_m=tuple(female_pose[:3]),
            female_quaternion_wxyz=tuple(female_pose[3:7]),
            male_position_m=tuple(male_pose[:3]),
            male_quaternion_wxyz=tuple(male_pose[3:7]),
            female_linear_velocity_mps=tuple(female_velocity[:3]),
            female_angular_velocity_radps=tuple(female_velocity[3:6]),
            male_linear_velocity_mps=tuple(male_velocity[:3]),
            male_angular_velocity_radps=tuple(male_velocity[3:6]),
            right_female_contact_force_n=readings["right_female"][0],
            left_male_contact_force_n=sum(
                readings[key][0] for key in
                ("left_male", "left_male_tongue_neg", "left_male_tongue_pos")
            ),
            buckle_body_contact_force_n=readings["buckle_body"][0],
            robot_table_contact_force_n=readings["robot_table"][0],
            # Hand/desk overlap is diagnostic, not an insertion failure. Keep
            # unavailable contact data fail-closed and gate inter-buckle overlap.
            max_penetration_m=(readings["buckle_body"][1] if all(
                math.isfinite(value[1]) for value in readings.values()) else math.inf),
        )

    def _failure_reason_now(
        self,
        readback: P006PhysicalReadback,
        *,
        include_penetration: bool,
    ) -> str:
        metrics = physical_metrics(readback)
        positions = (readback.female_position_m, readback.male_position_m)
        out_of_bounds = any(
            position[2] < 0.84
            or position[2] > 1.40
            or abs(position[0]) > 0.60
            or abs(position[1]) > 0.60
            for position in positions
        )
        if not metrics["finite"]:
            return "non_finite_readback"
        if out_of_bounds:
            return "object_out_of_bounds"
        if include_penetration and readback.max_penetration_m > MAX_PENETRATION_M:
            return "excessive_penetration"
        return ""

    def _failure_now(self, readback: P006PhysicalReadback) -> bool:
        """Return the raw single-sample failure predicate used by audits."""

        return bool(self._failure_reason_now(readback, include_penetration=True))

    def _contact_penetrations_m(self) -> dict[str, float]:
        return {
            key: float(self._p006_last_contact_readings.get(key, (0.0, 0.0))[1])
            for key in _P006_RUNTIME_CONTACT_KEYS
        }

    def _update_failure_monitor(self, readback: P006PhysicalReadback) -> None:
        hard_failure_reason = self._failure_reason_now(
            readback,
            include_penetration=False,
        )
        previous_violation_substeps = self._p006_penetration_violation_substeps
        penetration_exceeded = bool(
            math.isfinite(readback.max_penetration_m)
            and readback.max_penetration_m > MAX_PENETRATION_M
        )
        if penetration_exceeded:
            self._p006_penetration_violation_substeps += 1
            if previous_violation_substeps == 0:
                print(
                    "[HandsOn/P006] penetration grace started: "
                    f"max={1000.0 * readback.max_penetration_m:.2f}mm, "
                    f"contacts_m={self._contact_penetrations_m()}, "
                    f"dwell_substeps={self._p006_penetration_failure_dwell_substeps}",
                    flush=True,
                )
        else:
            self._p006_penetration_violation_substeps = 0
            if previous_violation_substeps > 0 and not hard_failure_reason:
                print(
                    "[HandsOn/P006] transient penetration recovered: "
                    f"substeps={previous_violation_substeps}, "
                    f"peak_m={self._p006_penetration_peak_m}",
                    flush=True,
                )

        failure_reason = hard_failure_reason
        if (
            not failure_reason
            and self._p006_penetration_violation_substeps
            >= self._p006_penetration_failure_dwell_substeps
        ):
            failure_reason = "persistent_excessive_penetration"

        if failure_reason and not self._p006_failure_latched:
            self._p006_failure_reason = failure_reason
            self._p006_first_failure_snapshot = {
                "reason": failure_reason,
                "max_penetration_m": float(readback.max_penetration_m),
                "contact_penetrations_m": self._contact_penetrations_m(),
                "contact_penetration_peaks_m": dict(self._p006_penetration_peak_m),
                "penetration_violation_substeps": int(
                    self._p006_penetration_violation_substeps
                ),
                "penetration_failure_dwell_substeps": int(
                    self._p006_penetration_failure_dwell_substeps
                ),
                "female_position_m": tuple(readback.female_position_m),
                "male_position_m": tuple(readback.male_position_m),
            }
            print(
                "[HandsOn/P006] failure latched: "
                f"{self._p006_first_failure_snapshot}",
                flush=True,
            )
        self._p006_failure_latched |= bool(failure_reason)

    def _refresh_metrics(
        self,
        *,
        update_latch: bool,
        update_failure_monitor: bool = False,
    ) -> dict[str, Any]:
        self._p006_last_readback = self._physical_readback()
        if update_latch:
            self._p006_last_metrics = self._p006_latch.update(self._p006_last_readback)
        else:
            self._p006_last_metrics = {
                **physical_metrics(self._p006_last_readback),
                **self._p006_latch.to_dict(),
                "success_now": bool(self._p006_latch.success_hold_steps >= SUCCESS_HOLD_STEPS
                                    and not self._p006_failure_latched),
            }
        if update_failure_monitor:
            self._update_failure_monitor(self._p006_last_readback)
        if self._p006_failure_latched:
            self._p006_last_metrics["success_now"] = False
        self._p006_last_metrics.update(
            {
                "right_female_contact_force_n": float(self._p006_last_readback.right_female_contact_force_n),
                "left_male_contact_force_n": float(self._p006_last_readback.left_male_contact_force_n),
                "buckle_body_contact_force_n": float(self._p006_last_readback.buckle_body_contact_force_n),
                "max_penetration_m": float(
                    self._p006_last_readback.max_penetration_m
                ),
                "p006_contact_penetrations_m": self._contact_penetrations_m(),
                "p006_contact_penetration_peaks_m": dict(
                    self._p006_penetration_peak_m
                ),
                "p006_penetration_violation_substeps": int(
                    self._p006_penetration_violation_substeps
                ),
                "p006_penetration_failure_dwell_substeps": int(
                    self._p006_penetration_failure_dwell_substeps
                ),
                "p006_failure_reason": self._p006_failure_reason,
                "p006_first_failure_snapshot": dict(
                    self._p006_first_failure_snapshot
                ),
                "failed": self._p006_failure_latched,
            }
        )
        return self._p006_last_metrics

    def _after_task_physics_substep(self) -> None:
        if self._p006_contact_contract_ready:
            self._refresh_metrics(
                update_latch=False,
                update_failure_monitor=True,
            )

    def _compute_success_now(self) -> bool:
        metrics = self._refresh_metrics(update_latch=True)
        return bool(metrics.get("success_now", False) and not self._p006_failure_latched)

    def _compute_failure(self) -> bool:
        return bool(self._p006_failure_latched)

    def _success_steps_required(self) -> int:
        # The task-owned state machine already enforces the 12-frame final hold.
        return 1

    def _task_info(self) -> dict[str, Any]:
        return {
            **self._p006_last_metrics,
            "p006_task_id": "P0-06",
            "p006_reset_seed": int(self._p006_reset.seed),
            "p006_success_steps_required": SUCCESS_HOLD_STEPS,
            "p006_pull_test_required": False,
            "p006_progress_blockers": self._progress_blockers(),
            "p006_initial_state_report": dict(
                self._p006_initial_state_report
            ),
        }

    def _progress_blockers(self) -> list[str]:
        """Read-only hints; no extra latch or success predicate."""
        m = self._p006_last_metrics
        if m.get("failed"):
            return [str(m.get("p006_failure_reason") or "failed - reset required")]
        if m.get("success_now"):
            return []
        blockers = []
        if not m.get("latched_seen"):
            for key, label in (("right_female_contact_force_n", "right hand on blue buckle"),
                               ("left_male_contact_force_n", "left hand on orange buckle")):
                if float(m.get(key, 0)) < MIN_ROLE_CONTACT_FORCE_N:
                    blockers.append(label)
            if float(m.get("buckle_body_contact_force_n", 0)) < MIN_BODY_CONTACT_FORCE_N:
                blockers.append("press buckle surfaces into contact")
        if float(m.get("insertion_depth_m", 0)) < MIN_INSERTION_DEPTH_M:
            blockers.append("insert >= 18 mm")
        if float(m.get("transverse_offset_m", math.inf)) > MAX_TRANSVERSE_OFFSET_M:
            blockers.append("center all three tongues")
        if float(m.get("axis_error_deg", math.inf)) > MAX_AXIS_ERROR_DEG:
            blockers.append("align axes <= 12 deg")
        if float(m.get("max_penetration_m", 0)) > MAX_PENETRATION_M:
            blockers.append("reduce buckle-body overlap")
        return blockers or ["hold press contact" if not m.get("latched_seen") else "hold still"]

    def _task_status_text(self) -> str:
        m = self._p006_last_metrics
        return (
            f"Phase {m.get('phase', 'separated')} | "
            f"Insert {1000 * float(m.get('insertion_depth_m', 0)):.1f}/18 mm | "
            f"Offset {1000 * float(m.get('transverse_offset_m', 0)):.1f}/15 mm | "
            f"Axis {float(m.get('axis_error_deg', 0)):.1f}/12 deg\n"
            f"R blue/L orange/body contact "
            f"{float(m.get('right_female_contact_force_n', 0)):.2f}/"
            f"{float(m.get('left_male_contact_force_n', 0)):.2f}/"
            f"{float(m.get('buckle_body_contact_force_n', 0)):.2f} N | "
            f"Press {int(m.get('latch_dwell_steps', 0))}/5 | "
            f"Stable {int(m.get('success_hold_steps', 0))}/12\n"
            f"Need: {', '.join(self._progress_blockers()) or 'complete'}"
        )

    @staticmethod
    def _success_condition_display_label(row: dict[str, Any]) -> str:
        key = str(row.get("key") or "")
        if key.startswith("p006_") and row.get("label"):
            return str(row["label"])
        return HandsOnABCIsaacLabEnv._success_condition_display_label(row)

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._p006_last_metrics
        return [
            self._condition_row(
                "p006_roles",
                "Right hand holds female; left hand holds male",
                bool(
                    metrics.get("right_female_role_contact", False)
                    and metrics.get("left_male_role_contact", False)
                ),
            ),
            self._condition_row(
                "p006_alignment",
                "Lateral error <= 15 mm; axis error <= 12 deg",
                bool(metrics.get("aligned", False)),
            ),
            self._condition_row(
                "p006_press",
                "Insertion >= 18 mm with buckle-body contact",
                bool(metrics.get("deep_enough", False) and metrics.get("body_contact", False)),
            ),
            self._condition_row(
                "p006_latched",
                "Pressed contact held; buckle latched",
                bool(metrics.get("latched_seen", False)),
            ),
            self._condition_row(
                "p006_stable",
                "Latched pair stable; no pull test",
                bool(metrics.get("success_now", False)),
            ),
        ]

    def _compute_observation(self) -> dict[str, Any]:
        observation = super()._compute_observation()
        if self._p006_last_readback is None and self._p006_body_views:
            self._refresh_metrics(update_latch=False)
        readback = self._p006_last_readback
        if readback is None:
            zero_pose = (0.0,) * 7
            readback = P006PhysicalReadback(
                zero_pose[:3], (1.0, 0.0, 0.0, 0.0),
                zero_pose[:3], (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                0.0, 0.0, 0.0, 0.0, 0.0,
            )
        metrics = self._p006_last_metrics
        state = observation["state"]
        state.update(
            {
                "female_buckle_pose": np.asarray(
                    [*readback.female_position_m, *readback.female_quaternion_wxyz],
                    dtype=np.float64,
                ),
                "male_buckle_pose": np.asarray(
                    [*readback.male_position_m, *readback.male_quaternion_wxyz],
                    dtype=np.float64,
                ),
                "p006_alignment_state": np.asarray(
                    [
                        metrics.get("axial_separation_m", 0.0),
                        metrics.get("insertion_depth_m", 0.0),
                        metrics.get("transverse_offset_m", 0.0),
                        metrics.get("axis_error_deg", 0.0),
                        metrics.get("relative_linear_speed_mps", 0.0),
                        metrics.get("relative_angular_speed_radps", 0.0),
                        float(metrics.get("aligned", False)),
                        float(metrics.get("deep_enough", False)),
                    ],
                    dtype=np.float64,
                ),
                "p006_role_contacts": np.asarray(
                    [
                        readback.right_female_contact_force_n,
                        readback.left_male_contact_force_n,
                        readback.buckle_body_contact_force_n,
                        readback.robot_table_contact_force_n,
                    ],
                    dtype=np.float64,
                ),
                "p006_latch_state": np.asarray(
                    [
                        float(self._p006_latch.phase),
                        float(self._p006_latch.latch_dwell_steps),
                        float(self._p006_latch.success_hold_steps),
                        float(self._p006_latch.right_female_contact_seen),
                        float(self._p006_latch.left_male_contact_seen),
                        float(self._p006_latch.aligned_seen),
                        float(self._p006_latch.latched_seen),
                        float(self._p006_failure_latched),
                    ],
                    dtype=np.float64,
                ),
            }
        )
        return observation

    def restore_recorded_initial_state(
        self, state_parts: Mapping[str, np.ndarray]
    ) -> dict[str, Any]:
        observation = super().restore_recorded_initial_state(state_parts)
        self._p006_latch.reset()
        self._p006_failure_latched = False
        self._p006_failure_reason = ""
        self._p006_first_failure_snapshot = {}
        self._p006_last_contact_readings = {}
        self._p006_penetration_peak_m = {}
        self._p006_penetration_violation_substeps = 0
        self._refresh_metrics(update_latch=False)
        return self._compute_observation() if self._p006_body_views else observation


__all__ = ["HandsOnP006IsaacLabEnv"]
