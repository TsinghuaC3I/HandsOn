"""Isaac Sim 5.1 / IsaacLab 2.3.2 runtime for LW native USD tasks."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from gymnasium import spaces
import numpy as np

from .abc_collection_env import HandsOnABCIsaacLabEnv
from .lw_native_articulation_contract import (
    CONDITION_LABELS,
    G08_LID_OPEN_THRESHOLD_DEG,
    LWNativeArticulationState,
    LWNativeResetSample,
    MAX_DOF_COUNT,
    OBSERVATION_VECTOR_SIZE,
    TASK_SPECS as LW_CONTRACT_SPECS,
    sample_reset,
)
from .lw_native_articulation_scene import (
    G05_TASK_NAME,
    G08_LID_HARD_STOP_DEG,
    G08_TASK_NAME,
    G10_TASK_NAME,
    GEN3_TASK_NAME,
    LW_CONTROL_HZ,
    LW_PHYSICS_HZ,
    LW_SUCCESS_CONDITION_LABELS,
    LW_TABLE_COLLIDER_PATH,
    LW_TASK_NAMES,
    LW_TASK_SPECS,
    spawn_lw_task_assets,
)
from .recording_cameras import WRIST_RECORDING_CAMERA_OUTPUT_SIZE


_TRUTH_STARTUP_GRACE_SUBSTEPS = 60
_ORIGIN_MAX_LINEAR_SPEED_MPS = 0.015
_ORIGIN_MAX_ANGULAR_SPEED_RADPS = 0.08
_ORIGIN_MAX_DOF_SPEED_RADPS = 0.08
_ORIGIN_MAX_INITIAL_DOF_DRIFT_RAD = math.radians(0.5)
_G08_SUPPORT_FORCE_MIN_N = 4.0
_G08_SUPPORT_CLEARANCE_MIN_M = -0.00025
_G08_SUPPORT_CLEARANCE_MAX_M = 0.00050
_G08_HAND_SOLVER_POSITION_ITERATIONS = 64
_G08_HAND_SOLVER_VELOCITY_ITERATIONS = 8
_G08_HAND_MAX_DEPENETRATION_VELOCITY_MPS = 0.40
_G10_HAND_SOLVER_POSITION_ITERATIONS = 64
_G10_HAND_SOLVER_VELOCITY_ITERATIONS = 8
_G10_HAND_MAX_DEPENETRATION_VELOCITY_MPS = 0.40
_G08_BASE_BOTTOM_CORNERS_LOCAL_M = np.asarray(
    [
        (0.4084579646587372, -0.2231619432568550, -0.1364001780748367),
        (-0.4076589643955231, -0.2231620028614998, -0.1364001780748367),
        (0.4084578752517700, 0.2193565890192986, -0.1364005804061890),
        (-0.4076589643955231, 0.2193564996123314, -0.1364005804061890),
    ],
    dtype=np.float64,
) * np.asarray(LW_TASK_SPECS[G08_TASK_NAME].scale, dtype=np.float64)


def _finite_float_or_none(value: Any) -> float | None:
    """Keep fail-closed sentinels out of strict JSON evidence payloads."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _quat_angle_degrees(lhs: np.ndarray, rhs: np.ndarray) -> float:
    left = np.asarray(lhs, dtype=np.float64).reshape(4)
    right = np.asarray(rhs, dtype=np.float64).reshape(4)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1.0e-12 or right_norm <= 1.0e-12:
        return math.inf
    cosine = float(
        np.clip(abs(np.dot(left / left_norm, right / right_norm)), 0.0, 1.0)
    )
    return float(math.degrees(2.0 * math.acos(cosine)))


def _rotate_wxyz(vectors: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12 or not np.isfinite(norm):
        return np.full_like(values, np.nan)
    w = float(quat[0] / norm)
    xyz = quat[1:4] / norm
    first = np.cross(np.broadcast_to(xyz, values.shape), values)
    return values + 2.0 * (w * first + np.cross(xyz, first))


def _quat_mul_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.asarray(left, dtype=np.float64).reshape(4)
    rx, ry, rz, rw = np.asarray(right, dtype=np.float64).reshape(4)
    return np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _transform_composed_root_pose(
    reference_pose_xyzw: np.ndarray,
    reference_reset: LWNativeResetSample,
    target_reset: LWNativeResetSample,
) -> np.ndarray:
    """Move a composed LW root-link pose between seeded task frames."""

    pose = np.asarray(reference_pose_xyzw, dtype=np.float64).reshape(7).copy()
    values = (
        reference_reset.root_x_m,
        reference_reset.root_y_m,
        target_reset.root_x_m,
        target_reset.root_y_m,
    )
    if any(value is None for value in values):
        return pose
    if (
        float(reference_reset.root_x_m) == float(target_reset.root_x_m)
        and float(reference_reset.root_y_m) == float(target_reset.root_y_m)
        and float(reference_reset.root_yaw_deg)
        == float(target_reset.root_yaw_deg)
    ):
        return pose

    reference_yaw = math.radians(float(reference_reset.root_yaw_deg))
    target_yaw = math.radians(float(target_reset.root_yaw_deg))

    def rotation_xy(yaw: float) -> np.ndarray:
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)

    reference_origin = np.asarray(
        [reference_reset.root_x_m, reference_reset.root_y_m], dtype=np.float64
    )
    target_origin = np.asarray(
        [target_reset.root_x_m, target_reset.root_y_m], dtype=np.float64
    )
    body_offset_in_task_frame = rotation_xy(reference_yaw).T @ (
        pose[:2] - reference_origin
    )
    pose[:2] = target_origin + rotation_xy(target_yaw) @ body_offset_in_task_frame

    yaw_delta = target_yaw - reference_yaw
    delta_quat = np.asarray(
        [0.0, 0.0, math.sin(yaw_delta / 2.0), math.cos(yaw_delta / 2.0)],
        dtype=np.float64,
    )
    pose[3:7] = _quat_mul_xyzw(delta_quat, pose[3:7])
    return pose


def _g08_support_clearance_m(root_pose: np.ndarray | None) -> float:
    if root_pose is None:
        return math.inf
    pose = np.asarray(root_pose, dtype=np.float64).reshape(7)
    corners = pose[:3] + _rotate_wxyz(
        _G08_BASE_BOTTOM_CORNERS_LOCAL_M,
        pose[3:7],
    )
    if not np.all(np.isfinite(corners)):
        return math.inf
    return float(np.min(corners[:, 2]) - 0.890)


class HandsOnLWNativeArticulationIsaacLabEnv(HandsOnABCIsaacLabEnv):
    """Thin ABC specialization with native articulation/contact truth only."""

    def _physx_contact_arm_target_limits_enabled(self) -> bool:
        """Bound G08 arm commands without changing its restored control mode.

        G08 keeps the shared tabletop arm home and the ordinary full-pose DLS
        controller.  Its first valid tracker packet can still contain a much
        larger Cartesian correction than one PhysX control frame should apply,
        especially immediately after reset or tracker recovery.  Reuse the
        shared pose and joint target limiters so that correction cannot eject a
        hand or inject a one-frame impulse into the hinged lid.
        """

        if self.task_name in (G08_TASK_NAME, GEN3_TASK_NAME):
            return True
        return super()._physx_contact_arm_target_limits_enabled()

    def __init__(self, task_name: str, **kwargs: Any):
        if task_name not in LW_TASK_NAMES:
            supported = ", ".join(LW_TASK_NAMES)
            raise ValueError(
                f"LW native-articulation runtime does not support {task_name!r}; "
                f"expected one of: {supported}"
            )
        self._lw_scene_spec = LW_TASK_SPECS[task_name]
        if task_name == GEN3_TASK_NAME:
            self.stable_recording_origin_default_robot_restore_enabled = False
        self._lw_contract_spec = LW_CONTRACT_SPECS[task_name]
        self._lw_manifest: dict[str, Any] = {}
        self._lw_contact_views: dict[str, Any] = {}
        self._lw_contact_force_n = {"left": 0.0, "right": 0.0}
        self._lw_contact_readable = {"left": False, "right": False}
        self._lw_support_contact_view: Any | None = None
        self._lw_support_force_n = 0.0
        self._lw_support_readable = False
        self._lw_support_contact_count = 0
        self._lw_support_min_separation_m = math.inf
        self._lw_articulation_view: Any | None = None
        self._lw_articulation_indices: Any | None = None
        self._lw_articulation_path = ""
        self._lw_dof_indices: tuple[int, ...] = ()
        self._lw_reset_serial = -1
        self._lw_composed_reset_sample = sample_reset(
            task_name, seed=0, randomize=False
        )
        self._lw_reset_sample = self._lw_composed_reset_sample
        self._lw_state = LWNativeArticulationState(
            self._lw_contract_spec,
            self._lw_reset_sample,
        )
        self._lw_root_baseline_pose: np.ndarray | None = None
        self._lw_origin_accepted = False
        self._lw_truth_primed = False
        self._lw_truth_wait_substeps = 0
        self._lw_last_terminal_reason_reported = ""
        self._lw_last_q = np.asarray(
            self._lw_reset_sample.initial_q_rad,
            dtype=np.float64,
        )
        self._lw_last_qd = np.zeros_like(self._lw_last_q)
        self._lw_reset_tcp_sync_count = 0
        self._reset_post_accept_substep_audits()
        kwargs.setdefault("physics_backend", "physx")
        kwargs.setdefault("physics_dt", 1.0 / LW_PHYSICS_HZ)
        kwargs.setdefault("hz", LW_CONTROL_HZ)
        kwargs.setdefault("control_dt", 1.0 / LW_CONTROL_HZ)
        kwargs.setdefault("time_limit", 90.0)
        kwargs.setdefault("render_resolution", (480, 640))
        kwargs.setdefault("wrist_camera_output_size", (640, 480))
        super().__init__(task_name=task_name, **kwargs)

    def _spawn_task_specific_assets(self, sim_utils: Any) -> None:
        if self._stage is None:
            raise RuntimeError("LW asset spawn requires an open USD stage")
        self._lw_manifest = spawn_lw_task_assets(
            self._stage,
            sim_utils,
            self.task_name,
        )
        expected_dofs = tuple(self._lw_scene_spec.dof_names)
        if tuple(self._lw_manifest.get("dof_names", ())) != expected_dofs:
            raise RuntimeError("LW spawn manifest changed the ordered DOF contract")
        composed_limits = self._lw_manifest.get("composed_joint_limits_deg", {})
        expected_limits: dict[str, tuple[float, float]] = {}
        if self.task_name in (G08_TASK_NAME, GEN3_TASK_NAME):
            expected_limits = {
                expected_dofs[0]: (-float(G08_LID_HARD_STOP_DEG), 0.0)
            }
        elif self.task_name == G10_TASK_NAME:
            expected_limits = {
                "Prop001_joint": (-180.0, 180.0),
                "Prop002_joint": (-180.0, 180.0),
            }
        for dof_name, expected_limit in expected_limits.items():
            actual_limit = composed_limits.get(dof_name)
            if actual_limit is None or any(
                not math.isclose(
                    float(actual_limit[index]),
                    expected_limit[index],
                    abs_tol=1.0e-6,
                )
                for index in (0, 1)
            ):
                raise RuntimeError(
                    f"{self._lw_contract_spec.task_id} spawn manifest is missing "
                    f"the composed joint-limit override for {dof_name}: "
                    f"expected={expected_limit}, actual={actual_limit!r}"
                )
        print(
            "[HandsOn/LW] Spawned native asset before first PhysX reset: "
            f"task={self.task_name}, bodies={self._lw_manifest['body_count']}, "
            f"colliders={self._lw_manifest['collider_count']}, "
            f"dofs={len(expected_dofs)}, "
            f"fixed_root={bool(self._lw_manifest.get('fixed_root', False))}, "
            f"composed_joint_limits_deg={composed_limits!r}",
            flush=True,
        )

    def _configure_task_collision_safety(
        self, sim_utils: Any
    ) -> tuple[int, int]:
        bodies, colliders = super()._configure_task_collision_safety(sim_utils)
        if (
            self.task_name not in LW_TASK_NAMES
            or self._stage is None
        ):
            return bodies, colliders

        from pxr import PhysxSchema, UsdPhysics

        # G05's ribs, G08's lid, and G10's ruler links are thin collision
        # geometry.  Swept CCD alone can still let a high-speed Wuji finger
        # enter the object between GPU solver substeps; speculative CCD grows a
        # conservative contact shell before that happens.  The lower
        # depenetration caps also prevent a first pinch from injecting a large
        # impulse into a fixed or table-supported body.
        if self.task_name in (G08_TASK_NAME, GEN3_TASK_NAME):
            # Keep G08 on the already validated swept-CCD hand path.  Enabling
            # speculative CCD on every Wuji link triggered CUDA 700 during
            # Isaac Sim 5.1 articulation initialization in an earlier layout.
            tuned_by_side = {"left": 0, "right": 0}
            for side in ("left", "right"):
                for path in self._hand_body_paths(side):
                    prim = self._stage.GetPrimAtPath(path)
                    if not prim.IsValid() or not prim.HasAPI(
                        UsdPhysics.RigidBodyAPI
                    ):
                        continue
                    rigid_api = (
                        PhysxSchema.PhysxRigidBodyAPI(prim)
                        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
                        else PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                    )
                    rigid_api.CreateEnableCCDAttr().Set(True)
                    rigid_api.CreateSolverPositionIterationCountAttr().Set(
                        _G08_HAND_SOLVER_POSITION_ITERATIONS
                    )
                    rigid_api.CreateSolverVelocityIterationCountAttr().Set(
                        _G08_HAND_SOLVER_VELOCITY_ITERATIONS
                    )
                    rigid_api.CreateMaxDepenetrationVelocityAttr().Set(
                        _G08_HAND_MAX_DEPENETRATION_VELOCITY_MPS
                    )
                    tuned_by_side[side] += 1
            if not all(tuned_by_side.values()):
                raise RuntimeError(
                    "G08 hand swept-CCD setup requires physical rigid bodies "
                    f"for both hands; tuned={tuned_by_side}"
                )
            return bodies + sum(tuned_by_side.values()), colliders

        if self.task_name == G05_TASK_NAME:
            max_depenetration_velocity = 0.60
        else:
            max_depenetration_velocity = (
                _G10_HAND_MAX_DEPENETRATION_VELOCITY_MPS
            )

        tuned_by_side = {"left": 0, "right": 0}
        for side in ("left", "right"):
            for path in self._hand_body_paths(side):
                prim = self._stage.GetPrimAtPath(path)
                if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                rigid_api = (
                    PhysxSchema.PhysxRigidBodyAPI(prim)
                    if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
                    else PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                )
                if self.task_name == G10_TASK_NAME:
                    # G10 is a free-root ruler; use swept CCD for thin links
                    # and leave speculative shells disabled.
                    rigid_api.CreateEnableCCDAttr().Set(True)
                else:
                    rigid_api.CreateEnableSpeculativeCCDAttr().Set(True)
                rigid_api.CreateSolverPositionIterationCountAttr().Set(
                    _G10_HAND_SOLVER_POSITION_ITERATIONS
                    if self.task_name == G10_TASK_NAME
                    else 64
                )
                rigid_api.CreateSolverVelocityIterationCountAttr().Set(
                    _G10_HAND_SOLVER_VELOCITY_ITERATIONS
                    if self.task_name == G10_TASK_NAME
                    else 8
                )
                rigid_api.CreateMaxDepenetrationVelocityAttr().Set(
                    max_depenetration_velocity
                )
                tuned_by_side[side] += 1

        if not all(tuned_by_side.values()):
            raise RuntimeError(
                f"{self._lw_contract_spec.task_id} anti-tunnelling setup requires "
                "physical rigid bodies for "
                f"both hands; tuned={tuned_by_side}"
            )
        return bodies + sum(tuned_by_side.values()), colliders

    def _configure_task_simulation_settings(self) -> None:
        """Enable PhysX contact processing before LW tensor views are bound.

        IsaacLab's ``SimulationContext`` defaults
        ``/physics/disableContactProcessing`` to ``True``.  That leaves the
        collision solver active (objects still visibly push one another) but
        suppresses the contact manifolds consumed by the LW assigned-contact
        contract, making the two 0.15 N G08 rows permanently false.  G10's
        completion contract is geometric (both hinge leaves straight) and it
        deliberately keeps the stock setting: enabling the contact pipeline
        for its free-root, thin-link asset trips PhysX GPU narrowphase 700 on
        some Isaac 5.1 drivers.  Author the override only for G08 before the
        first ``sim.reset`` and fail closed if Kit does not retain it.
        """

        import carb

        settings = carb.settings.get_settings()
        # Do not touch the setting for G10.  Calling ``set_bool`` after the
        # PhysX scene has been created is itself unsafe on some Isaac 5.1 GPU
        # builds (even when restoring the default ``True`` value), and G10's
        # geometric contract does not consume contact manifolds.  A fresh
        # G10 process therefore keeps IsaacLab's stock mode.  The collection
        # server launches LW tasks in isolated workers; G08's override is
        # contained to its worker.
        if self.task_name not in (G08_TASK_NAME, GEN3_TASK_NAME):
            return
        settings.set_bool("/physics/disableContactProcessing", False)
        if bool(settings.get("/physics/disableContactProcessing")):
            raise RuntimeError(
                "LW native articulation could not apply its PhysX contact-processing mode"
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
        palm = f"{side}_palm_link"
        prefix = f"{side}_finger"
        return [
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and (prim.GetName() == palm or prim.GetName().startswith(prefix))
        ]

    def _init_task_specific_runtime(self) -> None:
        self._lw_contact_views = {}
        self._lw_support_contact_view = None
        if not self._robots:
            raise RuntimeError(
                "LW assigned-contact truth requires both live robot articulations"
            )
        physics_view = self._task_tensor_sim_view()
        if physics_view is None:
            raise RuntimeError("LW PhysX contact tensor view is unavailable")
        bindings = {
            "left": (
                self._lw_scene_spec.left_body_path,
                self._hand_body_paths("left"),
            ),
            "right": (
                self._lw_scene_spec.right_body_path,
                self._hand_body_paths("right"),
            ),
        }
        errors: list[str] = []
        for side, (sensor_path, filters) in bindings.items():
            if not filters:
                errors.append(f"{side}: no palm/finger rigid bodies")
                continue
            try:
                view = physics_view.create_rigid_contact_view(
                    sensor_path,
                    filter_patterns=filters,
                    max_contact_data_count=max(128, len(filters) * 12),
                )
                if (
                    view is None
                    or int(view.sensor_count) != 1
                    or int(view.filter_count) != len(filters)
                ):
                    errors.append(
                        f"{side}: incomplete assigned contact view "
                        f"(sensors={0 if view is None else int(view.sensor_count)}, "
                        f"filters={0 if view is None else int(view.filter_count)}/"
                        f"{len(filters)})"
                    )
                else:
                    self._lw_contact_views[side] = view
            except Exception as exc:
                errors.append(f"{side}: {exc}")
        if self.task_name == G08_TASK_NAME:
            try:
                support_view = physics_view.create_rigid_contact_view(
                    self._lw_scene_spec.root_body_path,
                    filter_patterns=[LW_TABLE_COLLIDER_PATH],
                    max_contact_data_count=32,
                )
                if (
                    support_view is None
                    or int(support_view.sensor_count) != 1
                    or int(support_view.filter_count) != 1
                ):
                    errors.append(
                        "support: incomplete base-to-table contact view "
                        f"(sensors={0 if support_view is None else int(support_view.sensor_count)}, "
                        f"filters={0 if support_view is None else int(support_view.filter_count)}/1)"
                    )
                else:
                    self._lw_support_contact_view = support_view
            except Exception as exc:
                errors.append(f"support: {exc}")
        if errors:
            raise RuntimeError(
                "LW assigned-contact truth is incomplete: "
                + "; ".join(errors)
            )

    def _capture_task_scene_state(self, task_mjcf: Any) -> None:
        super()._capture_task_scene_state(task_mjcf)
        expected = tuple(self._lw_scene_spec.dof_names)
        matches: list[tuple[Any, ...]] = []
        for entry in self._task_reset_articulation_states:
            _path, view, *_rest = entry
            try:
                mapping = {
                    str(name): int(index)
                    for name, index in view.get_metatype(0).dof_indices.items()
                }
            except Exception:
                continue
            if set(mapping) == set(expected) and len(mapping) == len(expected):
                matches.append(entry)
        if len(matches) != 1:
            raise RuntimeError(
                f"{self.task_name} requires exactly one native articulation "
                f"with DOFs {expected}; found {len(matches)}"
            )
        entry = matches[0]
        self._lw_articulation_path = str(entry[0])
        self._lw_articulation_view = entry[1]
        self._lw_articulation_indices = entry[2]
        metatype = self._lw_articulation_view.get_metatype(0)
        mapping = {
            str(name): int(index)
            for name, index in metatype.dof_indices.items()
        }
        expected_path = (
            f"{self._lw_scene_spec.asset_path}/world_fixed_joint"
            if self._lw_scene_spec.fixed_root
            else self._lw_scene_spec.asset_path
        )
        if self._lw_articulation_path != expected_path:
            raise RuntimeError(
                f"{self.task_name} articulation root changed: "
                f"{self._lw_articulation_path!r} != {expected_path!r}"
            )
        if int(self._lw_articulation_view.count) != 1:
            raise RuntimeError(
                f"{self.task_name} requires one native articulation instance; "
                f"found {int(self._lw_articulation_view.count)}"
            )
        if bool(metatype.fixed_base) != bool(self._lw_scene_spec.fixed_root):
            raise RuntimeError(
                f"{self.task_name} fixed-base classification changed: "
                f"live={bool(metatype.fixed_base)}, "
                f"expected={self._lw_scene_spec.fixed_root}"
            )
        if self.task_name in (G08_TASK_NAME, GEN3_TASK_NAME):
            # Re-read the authored attributes from the composed stage after
            # PhysX has discovered the articulation.  The spawn manifest is
            # checked above, while this second read proves the live composed
            # layer still carries the task-scoped -90/0 degree limits and the
            # articulation was classified as a native fixed base.
            from pxr import UsdPhysics

            joint_path = (
                f"{self._lw_scene_spec.asset_path}/"
                f"{self._lw_scene_spec.right_body_name}/"
                f"{self._lw_scene_spec.dof_names[0]}"
            )
            joint_prim = self._stage.GetPrimAtPath(joint_path)
            if not joint_prim.IsValid() or not joint_prim.IsA(
                UsdPhysics.RevoluteJoint
            ):
                raise RuntimeError(
                    f"G08 composed lid joint disappeared before runtime capture: {joint_path}"
                )
            joint = UsdPhysics.RevoluteJoint(joint_prim)
            composed_lower = float(joint.GetLowerLimitAttr().Get())
            composed_upper = float(joint.GetUpperLimitAttr().Get())
            if not (
                math.isclose(
                    composed_lower,
                    -float(G08_LID_HARD_STOP_DEG),
                    abs_tol=1.0e-6,
                )
                and math.isclose(composed_upper, 0.0, abs_tol=1.0e-6)
            ):
                raise RuntimeError(
                    "G08 live composed lid limit differs from the task hard stop: "
                    f"path={joint_path}, lower={composed_lower}, upper={composed_upper}"
                )
            print(
                "[HandsOn/LW] G08 composed-stage truth: "
                f"articulation={self._lw_articulation_path}, "
                f"metatype.fixed_base={bool(metatype.fixed_base)}, "
                f"joint={joint_path}, "
                f"lower_limit_deg={composed_lower:.6f}, "
                f"upper_limit_deg={composed_upper:.6f}",
                flush=True,
            )
        elif self.task_name == G10_TASK_NAME:
            # Verify the free-root composition retained the task-scoped
            # symmetric ±180-degree limits after PhysX discovered the native
            # articulation.  The readback proves the composed override,
            # rather than a YAML-only claim.
            if getattr(self, "_stage", None) is not None:
                from pxr import UsdPhysics

                expected_limits = {
                    "Prop001_joint": (-180.0, 180.0),
                    "Prop002_joint": (-180.0, 180.0),
                }
                for dof_name, (expected_lower, expected_upper) in expected_limits.items():
                    joint_path = f"{self._lw_scene_spec.asset_path}/"
                    joint_path += (
                        "Ruler035_Prop001/Prop001_joint"
                        if dof_name == "Prop001_joint"
                        else "Ruler035_Prop002/Prop002_joint"
                    )
                    joint_prim = self._stage.GetPrimAtPath(joint_path)
                    if not joint_prim.IsValid() or not joint_prim.IsA(
                        UsdPhysics.RevoluteJoint
                    ):
                        raise RuntimeError(
                            "G10 composed ruler joint disappeared before runtime "
                            f"capture: {joint_path}"
                        )
                    joint = UsdPhysics.RevoluteJoint(joint_prim)
                    lower = float(joint.GetLowerLimitAttr().Get())
                    upper = float(joint.GetUpperLimitAttr().Get())
                    if not (
                        math.isclose(lower, expected_lower, abs_tol=1.0e-6)
                        and math.isclose(upper, expected_upper, abs_tol=1.0e-6)
                    ):
                        raise RuntimeError(
                            "G10 live composed joint limit differs from the signed "
                            f"task envelope: path={joint_path}, lower={lower}, upper={upper}"
                        )
                print(
                    "[HandsOn/LW] G10 composed-stage truth: "
                    f"articulation={self._lw_articulation_path}, "
                    f"metatype.fixed_base={bool(metatype.fixed_base)}, "
                    "joint_limits_deg={Prop001:(-180,+180), Prop002:(-180,+180)}, "
                    "free_root=True",
                    flush=True,
                )
        self._lw_dof_indices = tuple(mapping[name] for name in expected)
        before = len(self._task_reset_body_states)
        asset_prefix = f"{self._lw_scene_spec.asset_path}/"
        self._task_reset_body_states[:] = [
            state
            for state in self._task_reset_body_states
            if str(state[0]) != self._lw_scene_spec.asset_path
            and not str(state[0]).startswith(asset_prefix)
        ]
        remaining = [
            str(state[0])
            for state in self._task_reset_body_states
            if str(state[0]) == self._lw_scene_spec.asset_path
            or str(state[0]).startswith(asset_prefix)
        ]
        if remaining:
            raise RuntimeError(
                "LW reset cache retained articulation link rigid views: "
                f"{remaining}"
            )
        removed = before - len(self._task_reset_body_states)
        if removed:
            print(
                "[HandsOn/LW] Removed duplicate articulation-link reset views: "
                f"{removed}",
                flush=True,
            )
        q, qd, dof_available = self._read_lw_dofs()
        if dof_available:
            self._lw_last_q = q
            self._lw_last_qd = qd

    def _task_articulation_root_is_writable(self, path: str) -> bool:
        if (
            self.task_name in (G08_TASK_NAME, GEN3_TASK_NAME)
            and str(path)
            == f"{self._lw_scene_spec.asset_path}/world_fixed_joint"
        ):
            # Like FU015, the fixed articulation is reset-positioned by a
            # supported tensor write before action zero, then remains fixed.
            return True
        if (
            self._lw_scene_spec.fixed_root
            and self._lw_scene_spec.asset_path in str(path)
        ):
            return False
        parent_method = getattr(
            super(), "_task_articulation_root_is_writable", None
        )
        if callable(parent_method):
            return bool(parent_method(path))
        # Lightweight runtime doubles used by the CPU contract tests may not
        # expose the optional base helper.  Fail closed for fixed-articulation
        # paths while preserving ordinary paths as writable.
        return not (
            self._lw_scene_spec.fixed_root
            and self._lw_scene_spec.asset_path in str(path)
        )

    def _task_articulation_root_velocity_is_writable(self, path: str) -> bool:
        return self._task_articulation_root_is_writable(path)

    def reset(self, seed: int | None = None, **kwargs: Any):
        if seed is not None:
            self._lw_reset_serial = -1
        return super().reset(seed=seed, **kwargs)

    def _sync_lw_reset_tcp_commands_to_physics(self) -> int:
        """Use the post-reset PhysX palms as LW Cartesian command zero.

        The ABC reset homes the articulated ROKAE arms before it samples the
        LW asset, but its nominal TCP command buffers are intentionally owned
        by the broader B320 task family.  Leaving those buffers at the module
        default makes the first valid glove packet look like a large jump and
        can drive both hands away from the authored reset envelope (or into a
        thin collider).  Read the already-settled palm poses and commit both
        command buffers atomically; this is a read-only synchronization of the
        controller state and does not write an object pose, velocity, or task
        joint target.
        """

        if (
            not bool(getattr(self, "_launch_isaac", False))
            or bool(getattr(self, "_direct_hand_control", False))
            or not self._robots
        ):
            return 0

        physical_poses: dict[str, np.ndarray] = {}
        for side in ("right", "left"):
            robot = self._robots.get(side)
            body_index = self._ee_body_indices.get(side)
            if robot is None or body_index is None:
                raise RuntimeError(
                    f"LW reset cannot resolve the {side} physical palm pose."
                )
            try:
                robot.update(0.0)
                matrix = np.asarray(
                    self._robot_body_pose_matrix(robot, int(body_index)),
                    dtype=np.float64,
                ).reshape(4, 4)
                pose = self._pose_matrix_to_vector(matrix)
            except Exception as exc:
                raise RuntimeError(
                    f"LW reset could not read the {side} physical palm pose."
                ) from exc
            if not np.all(np.isfinite(pose)) or not np.isclose(
                np.linalg.norm(pose[3:7]),
                1.0,
                atol=1.0e-6,
            ):
                raise RuntimeError(
                    f"LW reset produced an invalid {side} physical palm pose."
                )
            physical_poses[side] = pose.copy()

        # Commit both sides only after both reads pass validation.  A failed
        # second read must never leave one side rebased to another reset.
        self._right_tcp_pose = physical_poses["right"]
        self._left_tcp_pose = physical_poses["left"]
        self._lw_reset_tcp_sync_count += 1
        print(
            "[HandsOn/LW] TCP command zero synchronized to the reset PhysX palms.",
            flush=True,
        )
        return 2

    def _reset_scene_state(self) -> None:
        """Reset the scene, then align LW TCP commands before action zero."""

        super()._reset_scene_state()
        self._sync_lw_reset_tcp_commands_to_physics()

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        super()._sample_reset_state(clear_manual_request=clear_manual_request)
        self.delta_h[:] = 0.0
        self._lw_reset_serial += 1
        episode_seed = int(self._seed) + 104729 * self._lw_reset_serial
        self._lw_reset_sample = sample_reset(
            self._lw_contract_spec,
            seed=episode_seed,
            randomize=bool(self.randomize),
        )
        self._lw_state.reset(self._lw_reset_sample)
        self._lw_root_baseline_pose = None
        self._lw_origin_accepted = False
        self._lw_truth_primed = False
        self._lw_truth_wait_substeps = 0
        self._lw_last_terminal_reason_reported = ""
        self._lw_last_q = np.asarray(
            self._lw_reset_sample.initial_q_rad,
            dtype=np.float64,
        )
        self._lw_last_qd = np.zeros_like(self._lw_last_q)
        for side in ("left", "right"):
            self._lw_contact_force_n[side] = 0.0
            self._lw_contact_readable[side] = False
        self._lw_support_force_n = 0.0
        self._lw_support_readable = False
        self._lw_support_contact_count = 0
        self._lw_support_min_separation_m = math.inf
        self._reset_post_accept_substep_audits()

    def _reset_post_accept_substep_audits(self) -> None:
        """Clear read-only 120 Hz evidence after reset or origin acceptance."""

        self._lw_post_accept_substep_audit = {
            "sample_count": 0,
            "unreadable_count": 0,
            "max_abs_q_error_rad": 0.0,
            "max_abs_qd_rad_s": 0.0,
            "max_root_translation_error_m": 0.0,
            "max_root_rotation_error_deg": 0.0,
        }
        self._lw_g08_support_substep_audit = {
            "sample_count": 0,
            "unreadable_count": 0,
            "min_support_force_n": math.inf,
            "min_support_contact_count": math.inf,
            "min_support_clearance_m": math.inf,
            "max_support_clearance_m": -math.inf,
            "min_support_separation_m": math.inf,
            "max_support_separation_m": -math.inf,
        }

    def _post_accept_substep_audit_snapshot(self) -> dict[str, Any]:
        audit = self._lw_post_accept_substep_audit
        return {
            "sample_count": int(audit["sample_count"]),
            "unreadable_count": int(audit["unreadable_count"]),
            "truth_unreadable": bool(audit["unreadable_count"]),
            "max_abs_q_error_rad": float(audit["max_abs_q_error_rad"]),
            "max_abs_qd_rad_s": float(audit["max_abs_qd_rad_s"]),
            "max_root_translation_error_m": float(
                audit["max_root_translation_error_m"]
            ),
            "max_root_rotation_error_deg": float(
                audit["max_root_rotation_error_deg"]
            ),
        }

    def _g08_support_substep_audit_snapshot(self) -> dict[str, Any]:
        audit = self._lw_g08_support_substep_audit

        def finite_or_none(value: Any) -> float | None:
            number = float(value)
            return number if np.isfinite(number) else None

        clearance_min = finite_or_none(audit["min_support_clearance_m"])
        clearance_max = finite_or_none(audit["max_support_clearance_m"])
        separation_min = finite_or_none(audit["min_support_separation_m"])
        separation_max = finite_or_none(audit["max_support_separation_m"])
        contact_count = finite_or_none(audit["min_support_contact_count"])
        return {
            "sample_count": int(audit["sample_count"]),
            "unreadable_count": int(audit["unreadable_count"]),
            "support_unreadable": bool(audit["unreadable_count"]),
            "min_support_force_n": finite_or_none(
                audit["min_support_force_n"]
            ),
            "min_support_contact_count": (
                None if contact_count is None else int(contact_count)
            ),
            "support_clearance_range_m": (
                None
                if clearance_min is None or clearance_max is None
                else [clearance_min, clearance_max]
            ),
            "support_separation_range_m": (
                None
                if separation_min is None or separation_max is None
                else [separation_min, separation_max]
            ),
        }

    def _accumulate_post_accept_substep_audits(
        self,
        truth: Mapping[str, Any],
    ) -> None:
        """Latch one post-accept physics frame without changing task state."""

        if not self._lw_origin_accepted:
            return

        audit = self._lw_post_accept_substep_audit
        audit["sample_count"] += 1
        reference_q = np.asarray(
            self._lw_reset_sample.initial_q_rad,
            dtype=np.float64,
        )
        try:
            q = np.asarray(truth["q"], dtype=np.float64).reshape(
                reference_q.shape
            )
            q_readable = bool(np.all(np.isfinite(q)))
        except (KeyError, TypeError, ValueError):
            q = np.full(reference_q.shape, np.nan, dtype=np.float64)
            q_readable = False
        try:
            qd = np.asarray(truth["qd"], dtype=np.float64).reshape(
                reference_q.shape
            )
            qd_readable = bool(np.all(np.isfinite(qd)))
        except (KeyError, TypeError, ValueError):
            qd = np.full(reference_q.shape, np.nan, dtype=np.float64)
            qd_readable = False
        try:
            root_translation = float(truth["root_translation_error_m"])
            root_rotation = float(truth["root_rotation_error_deg"])
            root_readable = bool(
                truth.get("root_pose") is not None
                and np.isfinite(root_translation)
                and np.isfinite(root_rotation)
            )
        except (KeyError, TypeError, ValueError):
            root_translation = math.inf
            root_rotation = math.inf
            root_readable = False
        readable = bool(
            truth.get("dof_available", False)
            and truth.get("contact_available", False)
            and q_readable
            and qd_readable
            and root_readable
        )
        if not readable:
            audit["unreadable_count"] += 1
        if q_readable:
            audit["max_abs_q_error_rad"] = max(
                float(audit["max_abs_q_error_rad"]),
                float(np.max(np.abs(q - reference_q))),
            )
        if qd_readable:
            audit["max_abs_qd_rad_s"] = max(
                float(audit["max_abs_qd_rad_s"]),
                float(np.max(np.abs(qd))),
            )
        if root_readable:
            audit["max_root_translation_error_m"] = max(
                float(audit["max_root_translation_error_m"]),
                root_translation,
            )
            audit["max_root_rotation_error_deg"] = max(
                float(audit["max_root_rotation_error_deg"]),
                root_rotation,
            )

        if self.task_name != G08_TASK_NAME:
            return
        support = self._lw_g08_support_substep_audit
        support["sample_count"] += 1
        try:
            force = float(truth["support_force"])
            count = int(truth["support_contact_count"])
            clearance = float(truth["support_clearance_m"])
            separation = float(truth["support_min_separation_m"])
        except (KeyError, TypeError, ValueError, OverflowError):
            force = math.nan
            count = 0
            clearance = math.nan
            separation = math.nan
        support_readable = bool(
            truth.get("support_available", False)
            and truth.get("support_clearance_available", False)
            and np.isfinite(force)
            and np.isfinite(clearance)
            and np.isfinite(separation)
        )
        if not support_readable:
            support["unreadable_count"] += 1
            force = 0.0
            count = 0
        support["min_support_force_n"] = min(
            float(support["min_support_force_n"]),
            force,
        )
        support["min_support_contact_count"] = min(
            int(support["min_support_contact_count"])
            if np.isfinite(support["min_support_contact_count"])
            else count,
            count,
        )
        if np.isfinite(clearance):
            support["min_support_clearance_m"] = min(
                float(support["min_support_clearance_m"]),
                clearance,
            )
            support["max_support_clearance_m"] = max(
                float(support["max_support_clearance_m"]),
                clearance,
            )
        if np.isfinite(separation):
            support["min_support_separation_m"] = min(
                float(support["min_support_separation_m"]),
                separation,
            )
            support["max_support_separation_m"] = max(
                float(support["max_support_separation_m"]),
                separation,
            )

    def _apply_task_pose_randomization(self) -> None:
        if self._lw_articulation_view is None or self._torch is None:
            return
        entry = next(
            (
                value
                for value in self._task_reset_articulation_states
                if value[1] is self._lw_articulation_view
            ),
            None,
        )
        if entry is None:
            raise RuntimeError("LW articulation reset state is unavailable")
        path, view, indices, root_pose, root_velocity, joint_pos, joint_vel = entry
        positions = joint_pos.clone()
        velocities = self._torch.zeros_like(joint_vel)
        for source_index, view_index in enumerate(self._lw_dof_indices):
            positions[:, view_index] = float(
                self._lw_reset_sample.initial_q_rad[source_index]
            )
        root_writable = self._task_articulation_root_is_writable(path)
        pose_tensor = root_pose
        if root_writable and self.task_name in (G08_TASK_NAME, G10_TASK_NAME, GEN3_TASK_NAME):
            pose_np = root_pose.detach().cpu().numpy().copy()
            pose_np[0] = _transform_composed_root_pose(
                pose_np[0],
                self._lw_composed_reset_sample,
                self._lw_reset_sample,
            )
            pose_tensor = self._torch.as_tensor(
                pose_np,
                dtype=root_pose.dtype,
                device=root_pose.device,
            )
        zero_root_velocity = self._torch.zeros_like(root_velocity)
        if root_writable:
            view.set_root_transforms(pose_tensor, indices)
            view.set_root_velocities(zero_root_velocity, indices)
        view.set_dof_positions(positions, indices)
        view.set_dof_velocities(velocities, indices)
        self._record_authored_task_reset_write(
            path,
            transform_xyzw=pose_tensor if root_writable else None,
            velocity=zero_root_velocity if root_writable else None,
            dof_position=positions,
            dof_velocity=velocities,
        )
        if root_writable and self.task_name in (G08_TASK_NAME, G10_TASK_NAME, GEN3_TASK_NAME):
            print(
                f"[HandsOn/LW] {self._lw_contract_spec.task_id} seeded reset: "
                f"seed={self._lw_reset_sample.seed}, "
                f"xy=({self._lw_reset_sample.root_x_m:+.4f},"
                f"{self._lw_reset_sample.root_y_m:+.4f})m, "
                f"yaw={self._lw_reset_sample.root_yaw_deg:+.2f}deg",
                flush=True,
            )

    def _read_lw_dofs(self) -> tuple[np.ndarray, np.ndarray, bool]:
        count = self._lw_contract_spec.dof_count
        if self._lw_articulation_view is None or not self._lw_dof_indices:
            return (
                np.asarray(self._lw_last_q, dtype=np.float64).reshape(count),
                np.asarray(self._lw_last_qd, dtype=np.float64).reshape(count),
                False,
            )
        try:
            q_raw = self._lw_articulation_view.get_dof_positions()[0]
            qd_raw = self._lw_articulation_view.get_dof_velocities()[0]
            if hasattr(q_raw, "detach"):
                q_raw = q_raw.detach().cpu().numpy()
            if hasattr(qd_raw, "detach"):
                qd_raw = qd_raw.detach().cpu().numpy()
            q_full = np.asarray(q_raw, dtype=np.float64).reshape(-1)
            qd_full = np.asarray(qd_raw, dtype=np.float64).reshape(-1)
            indices = np.asarray(self._lw_dof_indices, dtype=np.int64)
            q = q_full[indices]
            qd = qd_full[indices]
            available = bool(
                q.shape == (count,)
                and qd.shape == (count,)
                and np.all(np.isfinite(q))
                and np.all(np.isfinite(qd))
            )
            if available:
                self._lw_last_q = q.copy()
                self._lw_last_qd = qd.copy()
            return q, qd, available
        except Exception:
            return (
                np.asarray(self._lw_last_q, dtype=np.float64).reshape(count),
                np.asarray(self._lw_last_qd, dtype=np.float64).reshape(count),
                False,
            )

    def _read_assigned_contact(self, side: str) -> tuple[float, bool]:
        view = self._lw_contact_views.get(side)
        if view is None:
            self._lw_contact_force_n[side] = 0.0
            self._lw_contact_readable[side] = False
            return 0.0, False
        try:
            matrix = view.get_contact_force_matrix(self._sim_dt)
            if hasattr(matrix, "detach"):
                matrix = matrix.detach().cpu().numpy()
            values = np.asarray(matrix, dtype=np.float64)
            if values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError("empty or non-finite contact tensor")
            # The matrix is one net force vector per sensor/filter pair.  Do
            # not silently accept a flattened or otherwise malformed tensor:
            # that makes the 0.15 N rows look valid while reading the wrong
            # entries.  A few Isaac Sim builds expose the pair dimensions only
            # on the view, while light-weight test doubles do not, so retain
            # the trailing-vector check and enforce the full shape whenever
            # the metadata is available.
            sensor_count = int(getattr(view, "sensor_count", 0) or 0)
            filter_count = int(getattr(view, "filter_count", 0) or 0)
            expected_shape = (
                sensor_count,
                filter_count,
                3,
            )
            matrix_shape_valid = bool(
                values.ndim >= 1
                and values.shape[-1] == 3
                and (
                    sensor_count <= 0
                    or filter_count <= 0
                    or values.shape == expected_shape
                )
            )
            matrix_force = (
                float(np.linalg.norm(values, axis=-1).sum())
                if matrix_shape_valid
                else 0.0
            )

            # Net force can cancel when several fingers press opposite sides
            # of the lid.  In that case the matrix remains readable but is
            # close to zero even though the detailed contact rows carry real
            # normal forces.  Sum those rows as a conservative physical
            # contact magnitude, using the same counts/starts contract as
            # IsaacLab's other ground-truth readers.
            detail_force = 0.0
            detail_valid = False
            get_contact_data = getattr(view, "get_contact_data", None)
            if callable(get_contact_data):
                try:
                    data = get_contact_data(self._sim_dt)
                    if not isinstance(data, (tuple, list)) or len(data) < 6:
                        raise ValueError("incomplete contact detail tensor")
                    normal_forces = data[0]
                    counts = data[4]
                    starts = data[5]

                    def _numpy(value: Any, dtype: Any) -> np.ndarray:
                        if hasattr(value, "detach"):
                            value = value.detach().cpu().numpy()
                        return np.asarray(value, dtype=dtype)

                    normal_forces_np = _numpy(normal_forces, np.float64).reshape(-1)
                    counts_np = _numpy(counts, np.float64).reshape(-1)
                    starts_np = _numpy(starts, np.float64).reshape(-1)
                    if not (
                        counts_np.shape == starts_np.shape
                        and np.all(np.isfinite(counts_np))
                        and np.all(np.isfinite(starts_np))
                        and np.all(counts_np >= 0.0)
                        and np.all(starts_np >= 0.0)
                        and np.all(counts_np == np.floor(counts_np))
                        and np.all(starts_np == np.floor(starts_np))
                    ):
                        raise ValueError("invalid contact row indexing")
                    pair_count = sensor_count * filter_count
                    if pair_count > 0 and counts_np.size != pair_count:
                        raise ValueError("contact row pair count mismatch")
                    if pair_count <= 0 and counts_np.size == 0:
                        raise ValueError("empty contact row indexing")
                    total = 0.0
                    for count_value, start_value in zip(counts_np, starts_np):
                        start = int(start_value)
                        end = start + int(count_value)
                        if start < 0 or end > normal_forces_np.size:
                            raise ValueError("contact row range is out of bounds")
                        total += float(
                            np.abs(normal_forces_np[start:end]).sum()
                        )
                    if not np.isfinite(total):
                        raise ValueError("non-finite contact row force")
                    detail_force = total
                    detail_valid = True
                except Exception:
                    # Detailed rows are an enhancement/fallback.  A valid
                    # matrix remains sufficient, preserving compatibility
                    # with older PhysX bindings that do not expose rows.
                    detail_valid = False

            if not matrix_shape_valid and not detail_valid:
                raise ValueError("malformed contact force tensor")
            force = max(matrix_force, detail_force if detail_valid else 0.0)
            self._lw_contact_force_n[side] = force
            self._lw_contact_readable[side] = True
            return force, True
        except Exception:
            self._lw_contact_force_n[side] = 0.0
            self._lw_contact_readable[side] = False
            return 0.0, False

    def _read_support_contact(self) -> tuple[float, bool]:
        if self.task_name != G08_TASK_NAME:
            return 0.0, True
        view = self._lw_support_contact_view
        if view is None:
            self._lw_support_force_n = 0.0
            self._lw_support_readable = False
            self._lw_support_contact_count = 0
            self._lw_support_min_separation_m = math.inf
            return 0.0, False
        try:
            matrix = view.get_contact_force_matrix(self._sim_dt)
            if hasattr(matrix, "detach"):
                matrix = matrix.detach().cpu().numpy()
            values = np.asarray(matrix, dtype=np.float64)
            if values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError("empty or non-finite support-contact tensor")
            force_vector = np.sum(values.reshape(-1, 3), axis=0)
            force = max(0.0, float(force_vector[2]))
            contact_data = view.get_contact_data(self._sim_dt)
            if not isinstance(contact_data, tuple) or len(contact_data) < 6:
                raise ValueError("incomplete support-contact detail tensor")
            distances = contact_data[3]
            counts = contact_data[4]
            starts = contact_data[5]
            if hasattr(distances, "detach"):
                distances = distances.detach().cpu().numpy()
            if hasattr(counts, "detach"):
                counts = counts.detach().cpu().numpy()
            if hasattr(starts, "detach"):
                starts = starts.detach().cpu().numpy()
            distances = np.asarray(distances, dtype=np.float64).reshape(-1)
            counts = np.asarray(counts, dtype=np.int64).reshape(-1)
            starts = np.asarray(starts, dtype=np.int64).reshape(-1)
            if counts.size != 1 or starts.size != 1:
                raise ValueError(
                    "support contact detail must contain one sensor/filter pair"
                )
            contact_count = int(counts[0])
            contact_start = int(starts[0])
            contact_end = contact_start + contact_count
            if (
                contact_count <= 0
                or contact_start < 0
                or contact_end > distances.size
            ):
                raise ValueError("support contact has no readable contact points")
            min_separation = float(
                np.min(distances[contact_start:contact_end])
            )
            if not np.isfinite(min_separation):
                raise ValueError("support contact separation is non-finite")
            self._lw_support_force_n = force
            self._lw_support_readable = True
            self._lw_support_contact_count = contact_count
            self._lw_support_min_separation_m = min_separation
            return force, True
        except Exception:
            self._lw_support_force_n = 0.0
            self._lw_support_readable = False
            self._lw_support_contact_count = 0
            self._lw_support_min_separation_m = math.inf
            return 0.0, False

    def _root_pose(self) -> np.ndarray | None:
        pose = self._physics_body_pose(self._lw_scene_spec.root_body_name)
        if pose is None:
            return None
        value = np.asarray(pose, dtype=np.float64).reshape(7)
        return value if np.all(np.isfinite(value)) else None

    def _root_errors(self, pose: np.ndarray | None) -> tuple[float, float]:
        if pose is None:
            return math.inf, math.inf
        if self._lw_root_baseline_pose is None:
            self._lw_root_baseline_pose = pose.copy()
        baseline = self._lw_root_baseline_pose
        return (
            float(np.linalg.norm(pose[:3] - baseline[:3])),
            _quat_angle_degrees(pose[3:7], baseline[3:7]),
        )

    def _read_lw_truth(self) -> dict[str, Any]:
        q, qd, dof_available = self._read_lw_dofs()
        left_force, left_available = self._read_assigned_contact("left")
        right_force, right_available = self._read_assigned_contact("right")
        support_force, support_available = self._read_support_contact()
        root_pose = self._root_pose()
        support_clearance = (
            _g08_support_clearance_m(root_pose)
            if self.task_name == G08_TASK_NAME
            else 0.0
        )
        support_clearance_available = bool(np.isfinite(support_clearance))
        support_clearance_valid = bool(
            support_clearance_available
            and _G08_SUPPORT_CLEARANCE_MIN_M
            <= support_clearance
            <= _G08_SUPPORT_CLEARANCE_MAX_M
        )
        support_separation_valid = bool(
            support_available
            and self._lw_support_contact_count >= 1
            and _G08_SUPPORT_CLEARANCE_MIN_M
            <= self._lw_support_min_separation_m
            <= _G08_SUPPORT_CLEARANCE_MAX_M
        )
        translation_error, rotation_error = self._root_errors(root_pose)
        return {
            "q": q,
            "qd": qd,
            "dof_available": dof_available,
            "left_force": left_force,
            "right_force": right_force,
            "contact_available": bool(left_available and right_available),
            "support_force": support_force,
            "support_available": bool(support_available),
            "support_active": bool(
                support_available
                and self._lw_support_contact_count >= 1
                and support_force >= _G08_SUPPORT_FORCE_MIN_N
            ),
            "support_contact_count": int(self._lw_support_contact_count),
            "support_min_separation_m": float(
                self._lw_support_min_separation_m
            ),
            "support_separation_valid": support_separation_valid,
            "support_clearance_m": support_clearance,
            "support_clearance_available": support_clearance_available,
            "support_clearance_valid": support_clearance_valid,
            "root_pose": root_pose,
            "root_translation_error_m": translation_error,
            "root_rotation_error_deg": rotation_error,
        }

    def _recording_origin_sample(
        self,
        q: Any,
    ) -> tuple[LWNativeResetSample | None, str]:
        """Validate the settled coordinates before rebasing action zero."""

        try:
            values = np.asarray(q, dtype=np.float64).reshape(
                self._lw_contract_spec.dof_count
            )
        except (TypeError, ValueError):
            return None, "invalid_dof_shape"
        if not np.all(np.isfinite(values)):
            return None, "non_finite_dof"
        lower = np.asarray(
            self._lw_contract_spec.lower_limits_rad,
            dtype=np.float64,
        )
        upper = np.asarray(
            self._lw_contract_spec.upper_limits_rad,
            dtype=np.float64,
        )
        if np.any(values < lower - 1.0e-6) or np.any(values > upper + 1.0e-6):
            return None, "joint_limit_exceeded"
        locked_index = self._lw_contract_spec.locked_dof_index
        if (
            locked_index is not None
            and abs(float(values[locked_index]))
            > self._lw_contract_spec.locked_dof_tolerance_rad
        ):
            return None, "locked_joint_moved"
        sample = LWNativeResetSample(
            task_id=self._lw_contract_spec.task_id,
            task_name=self.task_name,
            seed=int(self._lw_reset_sample.seed),
            randomize=bool(self._lw_reset_sample.randomize),
            initial_q_rad=tuple(float(value) for value in values),
            target_q_rad=tuple(self._lw_contract_spec.target_q_rad),
            root_x_m=self._lw_reset_sample.root_x_m,
            root_y_m=self._lw_reset_sample.root_y_m,
            root_yaw_deg=self._lw_reset_sample.root_yaw_deg,
        )
        try:
            LWNativeArticulationState(self._lw_contract_spec, sample)
        except ValueError as exc:
            return None, str(exc)
        return sample, ""

    def stable_recording_origin_task_readiness(self) -> dict[str, Any]:
        truth = self._read_lw_truth()
        support_required = bool(self._lw_contract_spec.requires_support_contact)
        support_available = bool(
            truth.get("support_available", not support_required)
        )
        support_active = bool(
            truth.get("support_active", not support_required)
        )
        support_force = float(truth.get("support_force", 0.0))
        support_clearance = float(truth.get("support_clearance_m", math.inf))
        support_clearance_available = bool(
            truth.get("support_clearance_available", not support_required)
        )
        support_clearance_valid = bool(
            truth.get("support_clearance_valid", not support_required)
        )
        support_contact_count = int(truth.get("support_contact_count", 0))
        support_min_separation = float(
            truth.get("support_min_separation_m", math.inf)
        )
        support_separation_valid = bool(
            truth.get("support_separation_valid", not support_required)
        )
        root_velocity = self._physics_body_velocity(
            self._lw_scene_spec.root_body_name
        )
        velocity_available = root_velocity is not None
        if velocity_available:
            root_velocity = np.asarray(root_velocity, dtype=np.float64).reshape(6)
            velocity_available = bool(np.all(np.isfinite(root_velocity)))
        linear_speed = (
            float(np.linalg.norm(root_velocity[:3])) if velocity_available else math.inf
        )
        angular_speed = (
            float(np.linalg.norm(root_velocity[3:])) if velocity_available else math.inf
        )
        dof_speed = (
            float(np.max(np.abs(truth["qd"])))
            if truth["dof_available"]
            else math.inf
        )
        initial_dof_drift = (
            float(
                np.max(
                    np.abs(
                        np.asarray(truth["q"], dtype=np.float64)
                        - np.asarray(
                            self._lw_reset_sample.initial_q_rad,
                            dtype=np.float64,
                        )
                    )
                )
            )
            if truth["dof_available"]
            else math.inf
        )
        origin_sample, origin_sample_error = self._recording_origin_sample(
            truth["q"]
        )
        root_within_contract = bool(
            not self._lw_contract_spec.root_pose_constraint
            or (
                truth["root_translation_error_m"]
                <= self._lw_contract_spec.root_translation_limit_m
                and truth["root_rotation_error_deg"]
                <= self._lw_contract_spec.root_rotation_limit_deg
            )
        )
        ready = bool(
            truth["dof_available"]
            and truth["contact_available"]
            and truth["root_pose"] is not None
            and velocity_available
            and origin_sample is not None
            and root_within_contract
            and initial_dof_drift <= _ORIGIN_MAX_INITIAL_DOF_DRIFT_RAD
            and linear_speed <= _ORIGIN_MAX_LINEAR_SPEED_MPS
            and angular_speed <= _ORIGIN_MAX_ANGULAR_SPEED_RADPS
            and dof_speed <= _ORIGIN_MAX_DOF_SPEED_RADPS
            and (
                not support_required
                or (
                    support_active
                    and support_clearance_valid
                    and support_separation_valid
                )
            )
        )
        return {
            "kind": "lw_native_articulation_truth_and_velocity",
            "ready": ready,
            "task_name": self.task_name,
            "dof_names": list(self._lw_scene_spec.dof_names),
            "q_rad": [
                _finite_float_or_none(value) for value in truth["q"]
            ],
            "qd_rad_s": [
                _finite_float_or_none(value) for value in truth["qd"]
            ],
            "dof_truth_available": bool(truth["dof_available"]),
            "contact_truth_available": bool(truth["contact_available"]),
            "support_truth_required": support_required,
            "support_truth_available": support_available,
            "support_contact_active": support_active,
            "support_force_n": support_force,
            "support_force_min_n": _G08_SUPPORT_FORCE_MIN_N,
            "support_clearance_available": support_clearance_available,
            "support_clearance_valid": support_clearance_valid,
            "support_clearance_m": _finite_float_or_none(
                support_clearance
            ),
            "support_clearance_min_m": _G08_SUPPORT_CLEARANCE_MIN_M,
            "support_clearance_max_m": _G08_SUPPORT_CLEARANCE_MAX_M,
            "support_contact_count": support_contact_count,
            "support_min_separation_m": _finite_float_or_none(
                support_min_separation
            ),
            "support_separation_valid": support_separation_valid,
            "root_pose_available": bool(truth["root_pose"] is not None),
            "root_velocity_available": bool(velocity_available),
            "initial_coordinates_valid": bool(origin_sample is not None),
            "initial_coordinates_error": origin_sample_error,
            "root_within_task_contract": root_within_contract,
            "root_translation_error_m": _finite_float_or_none(
                truth["root_translation_error_m"]
            ),
            "root_rotation_error_deg": _finite_float_or_none(
                truth["root_rotation_error_deg"]
            ),
            "root_linear_speed_mps": _finite_float_or_none(linear_speed),
            "root_angular_speed_radps": _finite_float_or_none(
                angular_speed
            ),
            "max_dof_speed_radps": _finite_float_or_none(dof_speed),
            "max_initial_dof_drift_rad": _finite_float_or_none(
                initial_dof_drift
            ),
            "max_initial_dof_drift_limit_rad": (
                _ORIGIN_MAX_INITIAL_DOF_DRIFT_RAD
            ),
        }

    def accept_stable_recording_origin(
        self,
        report: Mapping[str, Any],
    ) -> dict[str, Any]:
        from .recording_origin import RecordingOriginStabilityError

        final_readiness = report.get("final_task_readiness")
        if (
            report.get("task_readiness_required") is not True
            or not isinstance(final_readiness, Mapping)
            or final_readiness.get("ready") is not True
            or final_readiness.get("task_name") != self.task_name
        ):
            raise RecordingOriginStabilityError(
                "LW recording-origin acceptance requires the final continuous "
                "native articulation/contact/velocity proof.",
                diagnostics={
                    "failure_phase": "lw_origin_acceptance",
                    "task_name": self.task_name,
                    "task_readiness": final_readiness,
                },
            )
        if int(self.env_step) != 0:
            raise RecordingOriginStabilityError(
                "LW recording-origin acceptance ran after action zero.",
                diagnostics={
                    "failure_phase": "lw_origin_acceptance",
                    "task_name": self.task_name,
                    "env_step": int(self.env_step),
                },
            )
        truth = self._read_lw_truth()
        support_required = bool(self._lw_contract_spec.requires_support_contact)
        support_available = bool(
            truth.get("support_available", not support_required)
        )
        support_active = bool(
            truth.get("support_active", not support_required)
        )
        support_force = float(truth.get("support_force", 0.0))
        support_clearance = float(truth.get("support_clearance_m", math.inf))
        support_clearance_available = bool(
            truth.get("support_clearance_available", not support_required)
        )
        support_clearance_valid = bool(
            truth.get("support_clearance_valid", not support_required)
        )
        support_contact_count = int(truth.get("support_contact_count", 0))
        support_min_separation = float(
            truth.get("support_min_separation_m", math.inf)
        )
        support_separation_valid = bool(
            truth.get("support_separation_valid", not support_required)
        )
        if (
            not truth["dof_available"]
            or not truth["contact_available"]
            or (
                support_required
                and not (
                    support_active
                    and support_clearance_valid
                    and support_separation_valid
                )
            )
            or truth["root_pose"] is None
            or not np.array_equal(
                np.asarray(truth["qd"], dtype=np.float64),
                np.zeros(self._lw_contract_spec.dof_count, dtype=np.float64),
            )
        ):
            raise RecordingOriginStabilityError(
                "LW final truth changed after the exact-zero recording boundary.",
                diagnostics={
                    "failure_phase": "lw_origin_acceptance",
                    "task_name": self.task_name,
                    "dof_truth_available": bool(truth["dof_available"]),
                    "contact_truth_available": bool(
                        truth["contact_available"]
                    ),
                    "root_pose_available": bool(truth["root_pose"] is not None),
                    "qd_rad_s": [
                        _finite_float_or_none(value)
                        for value in truth["qd"]
                    ],
                    "contact_readable": dict(self._lw_contact_readable),
                    "support_truth_available": support_available,
                    "support_contact_active": support_active,
                    "support_force_n": support_force,
                    "support_clearance_available": support_clearance_available,
                    "support_clearance_valid": support_clearance_valid,
                    "support_clearance_m": _finite_float_or_none(
                        support_clearance
                    ),
                    "support_contact_count": support_contact_count,
                    "support_min_separation_m": _finite_float_or_none(
                        support_min_separation
                    ),
                    "support_separation_valid": support_separation_valid,
                },
            )
        accepted_sample, sample_error = self._recording_origin_sample(truth["q"])
        if accepted_sample is None:
            raise RecordingOriginStabilityError(
                "LW final articulation coordinates are not a valid task origin.",
                diagnostics={
                    "failure_phase": "lw_origin_acceptance",
                    "task_name": self.task_name,
                    "initial_coordinates_error": sample_error,
                    "q_rad": [
                        _finite_float_or_none(value)
                        for value in truth["q"]
                    ],
                },
            )
        self._lw_reset_sample = accepted_sample
        self._lw_state.reset(accepted_sample)
        self._lw_state.prime_recording_origin_truth(
            truth["q"],
            truth["qd"],
            truth["left_force"],
            truth["right_force"],
            True,
            True,
            support_available,
            bool(
                not support_required
                or (
                    support_active
                    and support_clearance_valid
                    and support_separation_valid
                )
            ),
        )
        self._lw_root_baseline_pose = truth["root_pose"].copy()
        self._lw_origin_accepted = True
        self._lw_truth_primed = True
        self._lw_truth_wait_substeps = 0
        self._reset_post_accept_substep_audits()
        return {
            "accepted": True,
            "task_name": self.task_name,
            "recording_origin_q_rad": [float(value) for value in truth["q"]],
            "baseline_root_pose_wxyz": self._lw_root_baseline_pose.tolist(),
            "dof_truth_available": True,
            "contact_truth_available": True,
            "support_truth_available": support_available,
            "support_contact_active": support_active,
            "support_force_n": support_force,
            "support_clearance_available": support_clearance_available,
            "support_clearance_valid": support_clearance_valid,
            "support_clearance_m": _finite_float_or_none(
                support_clearance
            ),
            "support_contact_count": support_contact_count,
            "support_min_separation_m": _finite_float_or_none(
                support_min_separation
            ),
            "support_separation_valid": support_separation_valid,
        }

    def _handson_post_physics_substep_callback(
        self,
        _substep_index: int,
        _substep_count: int,
    ) -> None:
        if not self._launch_isaac:
            return
        truth = self._read_lw_truth()
        self._accumulate_post_accept_substep_audits(truth)
        if not self._lw_truth_primed:
            if truth["dof_available"] and truth["contact_available"]:
                self._lw_state.prime_recording_origin_truth(
                    truth["q"],
                    truth["qd"],
                    truth["left_force"],
                    truth["right_force"],
                    True,
                    True,
                    bool(truth["support_available"]),
                    bool(
                        not self._lw_contract_spec.requires_support_contact
                        or (
                            truth["support_active"]
                            and truth["support_clearance_valid"]
                            and truth["support_separation_valid"]
                        )
                    ),
                )
                self._lw_truth_primed = True
                self._lw_truth_wait_substeps = 0
                if truth["root_pose"] is not None:
                    self._lw_root_baseline_pose = truth["root_pose"].copy()
                return
            self._lw_truth_wait_substeps += 1
            if self._lw_truth_wait_substeps < _TRUTH_STARTUP_GRACE_SUBSTEPS:
                return
        # Stable-origin settling has its own continuous truth/readiness audit.
        # Do not arm task progress or terminal predicates until action zero has
        # been accepted; accept_stable_recording_origin() resets and primes the
        # contract from that exact physical boundary.
        if not self._lw_origin_accepted:
            return
        self._lw_state.update(
            truth["q"],
            truth["qd"],
            truth["left_force"],
            truth["right_force"],
            bool(truth["dof_available"]),
            bool(truth["contact_available"]),
            float(truth["root_translation_error_m"]),
            float(truth["root_rotation_error_deg"]),
            float(self._sim_dt),
            bool(truth["support_available"]),
            bool(
                not self._lw_contract_spec.requires_support_contact
                or (
                    truth["support_active"]
                    and truth["support_clearance_valid"]
                    and truth["support_separation_valid"]
                )
            ),
        )

    def _build_observation_space(self) -> spaces.Dict:
        state_spaces = {
            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
            "gripper_pose": spaces.Box(-np.inf, np.inf, shape=(32,), dtype=np.float64),
            "lw_joint_state": spaces.Box(-np.inf, np.inf, shape=(60,), dtype=np.float64),
            "lw_contact_state": spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float64),
            "lw_task_state": spaces.Box(
                -np.inf,
                np.inf,
                shape=(OBSERVATION_VECTOR_SIZE,),
                dtype=np.float64,
            ),
            "lw_root_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "lw_role_poses": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
            "table_delta_height": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
        }
        observation_spaces: dict[str, spaces.Space] = {
            "state": spaces.Dict(state_spaces)
        }
        if self.image_obs:
            head = lambda: spaces.Box(
                0,
                255,
                shape=self._image_shape,
                dtype=np.uint8,
            )
            wrist_width, wrist_height = WRIST_RECORDING_CAMERA_OUTPUT_SIZE
            wrist_shape = (wrist_height, wrist_width, 3)
            wrist = lambda: spaces.Box(0, 255, shape=wrist_shape, dtype=np.uint8)
            observation_spaces["images"] = spaces.Dict(
                {
                    "random_camera" if self.randomize else "front": head(),
                    "wrist_left": wrist(),
                    "wrist_right": wrist(),
                }
            )
        return spaces.Dict(observation_spaces)

    def _role_pose(self, name: str) -> np.ndarray:
        pose = self._physics_body_pose(name)
        if pose is not None:
            value = np.asarray(pose, dtype=np.float64).reshape(7)
            if np.all(np.isfinite(value)):
                return value
        if name == self._lw_scene_spec.root_body_name:
            return np.asarray(
                [
                    *self._lw_scene_spec.translation,
                    *self._lw_scene_spec.orientation_wxyz,
                ],
                dtype=np.float64,
            )
        return np.asarray(
            [
                *self._lw_scene_spec.translation,
                *self._lw_scene_spec.orientation_wxyz,
            ],
            dtype=np.float64,
        )

    def _compute_observation(self) -> dict[str, Any]:
        q, qd, dof_available = self._read_lw_dofs()
        q_padded = np.zeros(MAX_DOF_COUNT, dtype=np.float64)
        qd_padded = np.zeros(MAX_DOF_COUNT, dtype=np.float64)
        target_padded = np.zeros(MAX_DOF_COUNT, dtype=np.float64)
        count = self._lw_contract_spec.dof_count
        q_padded[:count] = q
        qd_padded[:count] = qd
        target_padded[:count] = np.asarray(
            self._lw_contract_spec.target_q_rad,
            dtype=np.float64,
        )
        joint_state = np.concatenate(
            [
                q_padded,
                qd_padded,
                target_padded,
                np.asarray(
                    [float(count), float(dof_available), float(self._lw_origin_accepted)],
                    dtype=np.float64,
                ),
            ]
        )
        root_pose = self._role_pose(self._lw_scene_spec.root_body_name)
        support_clearance = (
            _g08_support_clearance_m(root_pose)
            if self.task_name == G08_TASK_NAME
            else 0.0
        )
        if not np.isfinite(support_clearance):
            support_clearance = 0.0
        support_separation = float(self._lw_support_min_separation_m)
        if not np.isfinite(support_separation):
            support_separation = 0.0
        contact_state = np.asarray(
            [
                float(self._lw_contact_readable["left"]),
                float(self._lw_contact_readable["right"]),
                float(self._lw_state.contact_active[0]),
                float(self._lw_state.contact_active[1]),
                self._lw_contact_force_n["left"],
                self._lw_contact_force_n["right"],
                float(self._lw_state.contact_seen[0]),
                float(self._lw_state.contact_seen[1]),
                float(self.task_name == G08_TASK_NAME),
                float(self._lw_support_readable),
                float(
                    self._lw_support_readable
                    and self._lw_support_contact_count >= 1
                    and self._lw_support_force_n >= _G08_SUPPORT_FORCE_MIN_N
                ),
                self._lw_support_force_n,
                float(
                    self.task_name != G08_TASK_NAME
                    or _G08_SUPPORT_CLEARANCE_MIN_M
                    <= support_clearance
                    <= _G08_SUPPORT_CLEARANCE_MAX_M
                ),
                support_clearance,
                float(self._lw_support_contact_count),
                support_separation,
            ],
            dtype=np.float64,
        )
        left_pose = self._role_pose(self._lw_scene_spec.left_body_name)
        right_pose = self._role_pose(self._lw_scene_spec.right_body_name)
        tcp = np.concatenate([self._right_tcp_pose, self._left_tcp_pose]).astype(
            np.float64
        )
        gripper = np.zeros(32, dtype=np.float64)
        gripper[: min(16, self._right_hand.size)] = self._right_hand[:16]
        gripper[16 : 16 + min(16, self._left_hand.size)] = self._left_hand[:16]
        state = {
            "tcp_pose": tcp,
            "gripper_pose": gripper,
            "lw_joint_state": joint_state,
            "lw_contact_state": contact_state,
            "lw_task_state": np.asarray(
                self._lw_state.observation_vector(),
                dtype=np.float64,
            ),
            "lw_root_pose": root_pose,
            "lw_role_poses": np.concatenate([left_pose, right_pose]),
            "table_delta_height": self.delta_h.astype(np.float64, copy=True),
        }
        observation: dict[str, Any] = {"state": state}
        if self.image_obs:
            front, wrist_left, wrist_right = self.render()
            observation["images"] = {
                "random_camera" if self.randomize else "front": front,
                "wrist_left": wrist_left,
                "wrist_right": wrist_right,
            }
        return observation

    def _compute_success_now(self) -> bool:
        return bool(self._lw_state.success and not self._lw_state.failure_reason)

    def _compute_failure(self) -> bool:
        reason = str(self._lw_state.failure_reason or "")
        if reason and reason != self._lw_last_terminal_reason_reported:
            self._lw_last_terminal_reason_reported = reason
            metrics = self._lw_state.metrics()
            summary = {
                "event": "lw_first_terminal",
                "task_id": self._lw_contract_spec.task_id,
                "task_name": self.task_name,
                "failure_reason": reason,
                "stage": int(self._lw_state.stage),
                "sim_time_s": _finite_float_or_none(
                    metrics.get("sim_time_s")
                ),
                "root_translation_error_m": _finite_float_or_none(
                    metrics.get("root_translation_error_m")
                ),
                "root_rotation_error_deg": _finite_float_or_none(
                    metrics.get("root_rotation_error_deg")
                ),
                "root_goal_translation_limit_m": float(
                    self._lw_contract_spec.root_translation_limit_m
                ),
                "root_goal_rotation_limit_deg": float(
                    self._lw_contract_spec.root_rotation_limit_deg
                ),
                "root_translation_drift_hold_s": float(
                    self._lw_contract_spec.root_translation_drift_hold_s
                ),
                "root_rotation_drift_hold_s": float(
                    self._lw_contract_spec.root_rotation_drift_hold_s
                ),
                "root_translation_drift_s": _finite_float_or_none(
                    metrics.get("root_translation_drift_s")
                ),
                "root_rotation_drift_s": _finite_float_or_none(
                    metrics.get("root_rotation_drift_s")
                ),
                "root_escape_translation_limit_m": _finite_float_or_none(
                    self._lw_contract_spec.root_escape_translation_limit_m
                ),
                "root_escape_rotation_limit_deg": _finite_float_or_none(
                    self._lw_contract_spec.root_escape_rotation_limit_deg
                ),
                "root_escape_hold_s": float(
                    self._lw_contract_spec.root_escape_hold_s
                ),
                "root_translation_escape_s": _finite_float_or_none(
                    metrics.get("root_translation_escape_s")
                ),
                "root_rotation_escape_s": _finite_float_or_none(
                    metrics.get("root_rotation_escape_s")
                ),
                "root_goal_valid": bool(metrics.get("root_goal_valid")),
                "support_goal_valid": bool(
                    metrics.get("support_goal_valid")
                ),
                "arming_stage": "post_recording_origin_action_substeps",
            }
            print(
                "[HandsOn/LW terminal] "
                + json.dumps(
                    summary,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        return bool(reason or super()._compute_failure())

    def _task_status_text(self) -> str:
        if self.task_name != GEN3_TASK_NAME:
            return super()._task_status_text()
        state = self._lw_state
        opening = -math.degrees(float(state.last_q[0]))
        status = "Success" if state.success else "Failed" if state.failure_reason else "Open the lid"
        suffix = f" | {state.failure_reason}" if state.failure_reason else ""
        return f"{status} | Opening {opening:.1f} / 75 degrees{suffix}"

    @staticmethod
    def _success_condition_display_label(row: dict[str, Any]) -> str:
        if str(row.get("key", "")).startswith("gen3_"):
            return str(row["label"])
        return HandsOnABCIsaacLabEnv._success_condition_display_label(row)

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        flags = dict(zip(CONDITION_LABELS, self._lw_state.condition_flags()))
        if self.task_name == GEN3_TASK_NAME:
            return [self._condition_row(
                "gen3_lid_opened", "Lid opening at least 75 degrees",
                bool(flags["success"]),
            )]
        if self.task_name == G08_TASK_NAME:
            return [
                self._condition_row(
                    "g08_lid_opened",
                    f"盒盖开启至少 {G08_LID_OPEN_THRESHOLD_DEG:g}°",
                    bool(flags["success"]),
                )
            ]
        if self._lw_contract_spec.straightness_only:
            # G10 is intentionally a geometric end-state task: the operator
            # only needs to pull the two ruler hinges straight.  Do not show
            # the legacy contact/synchronization/release rows, which would
            # imply requirements that the native contract no longer enforces.
            return [
                self._condition_row(
                    "lw_straight",
                    "The ruler is straight",
                    bool(flags["success"]),
                )
            ]
        labels = LW_SUCCESS_CONDITION_LABELS[self.task_name]
        return [
            self._condition_row(
                "lw_assigned_contacts",
                labels[0],
                bool(flags["left_contact_seen"] and flags["right_contact_seen"]),
            ),
            self._condition_row(
                "lw_signed_motion",
                labels[1],
                bool(flags["left_role_qualified"] and flags["right_role_qualified"]),
            ),
            self._condition_row(
                "lw_target",
                labels[2],
                bool(flags["synchronized_target"] and flags["target_hold_complete"]),
            ),
            self._condition_row(
                "lw_release_stability",
                labels[3],
                bool(flags["release_complete"] and flags["success"]),
            ),
        ]

    def _task_info(self) -> dict[str, Any]:
        info = super()._task_info()
        info.update(
            {
                "task_id": self._lw_contract_spec.task_id,
                "task_name": self.task_name,
                "lw_asset_source": self._lw_scene_spec.relative_usd,
                "lw_source_license_status": (
                    "CC-BY-4.0" if self.task_name == GEN3_TASK_NAME
                    else "unresolved_user_provided"
                ),
                "lw_reset_sample": self._lw_reset_sample.as_dict(),
                "lw_dof_names": tuple(self._lw_scene_spec.dof_names),
                "lw_contact_readable": dict(self._lw_contact_readable),
                "lw_contact_force_n": dict(self._lw_contact_force_n),
                "lw_support_contact_required": (
                    bool(self._lw_contract_spec.requires_support_contact)
                ),
                "lw_support_contact_readable": bool(
                    self._lw_support_readable
                ),
                "lw_support_contact_force_n": float(
                    self._lw_support_force_n
                ),
                "lw_support_contact_count": int(
                    self._lw_support_contact_count
                ),
                "lw_support_min_separation_m": _finite_float_or_none(
                    self._lw_support_min_separation_m
                ),
                "lw_metrics": dict(self._lw_state.metrics()),
                "lw_failed": bool(self._lw_state.failure_reason),
                "lw_failure_reason": self._lw_state.failure_reason or None,
                "failure_reason": self._lw_state.failure_reason or None,
                "lw_origin_accepted": bool(self._lw_origin_accepted),
                "lw_physical_replay_certified": False,
                "lw_reset_tcp_sync_count": int(
                    getattr(self, "_lw_reset_tcp_sync_count", 0)
                ),
            }
        )
        return info

    def get_physical_replay_reset_contract(self) -> dict[str, Any]:
        contract = dict(super().get_physical_replay_reset_contract())
        articulation_root = (
            f"{self._lw_scene_spec.asset_path}/world_fixed_joint"
            if self._lw_scene_spec.fixed_root
            else self._lw_scene_spec.asset_path
        )
        contract.update(
            {
                "lw_articulation_root": articulation_root,
                "lw_ordered_dof_names": tuple(self._lw_scene_spec.dof_names),
                "lw_assigned_body_paths": (
                    self._lw_scene_spec.left_body_path,
                    self._lw_scene_spec.right_body_path,
                ),
                "lw_requires_stable_recording_origin": True,
                "lw_requires_fresh_process_replay": True,
                "lw_physical_replay_certified": False,
            }
        )
        return contract


__all__ = ["HandsOnLWNativeArticulationIsaacLabEnv"]
