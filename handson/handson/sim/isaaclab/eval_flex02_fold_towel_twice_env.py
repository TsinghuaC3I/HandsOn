"""Isaac Sim 5.1 runtime for the ordered bimanual towel double fold."""

from __future__ import annotations

import math
import os
from typing import Any, Mapping

from gymnasium import spaces
import numpy as np

from . import flexible_cotton_towel as cotton
from .open_source_towels import SOURCES, ParticleClothViewAdapter
from .marc_sawyer_towel import SOURCE as MARC_SAWYER_TOWEL
from .abc_collection_env import HandsOnABCIsaacLabEnv
from .e_collection_env import HandsOnEIsaacLabEnv
from .e_collection_scene import E01SurfacePhysicsTuning
from .flexible_towel_double_fold_contract import (
    FOLD_CONTRACT_VERSION,
    MATERIAL_NODE_COUNT,
    RELEASE_STABLE_FRAMES_REQUIRED,
    ROTATION_TARGET_RAD,
    ROTATION_TOLERANCE_RAD,
    TASK_ID,
    TASK_NAME,
    FlexibleTowelFoldSequence,
    FoldSequenceSnapshot,
    FoldStage,
    compute_fold_metrics,
)
from .flexible_towel_double_fold_scene import (
    FLEXIBLE_TOWEL_CONTACT_OFFSET,
    FLEXIBLE_TOWEL_PLACEMENT,
    FLEXIBLE_TOWEL_DYNAMIC_FRICTION,
    FLEXIBLE_TOWEL_REST_OFFSET,
    FLEXIBLE_TOWEL_RESET_X_BOUNDS_M,
    FLEXIBLE_TOWEL_RESET_Y_BOUNDS_M,
    FLEXIBLE_TOWEL_SOLVER_POSITION_ITERATIONS,
    flexible_towel_interface_nodes,
    flexible_towel_surface_metadata,
    spawn_flexible_towel_surface,
)
from .g01_deformable_asset_scene import (
    G01_TOWEL_SURFACE_MESH_PATH,
    G01_TOWEL_SURFACE_ROOT_PATH,
)
from .b320_bimanual_scene import (
    B320_CONTROL_HZ,
    B320_PHYSICS_HZ,
    B320_TABLE_TOP_Z,
)


_HAND_ENGAGED_DISTANCE_M = 0.055
_HAND_RELEASE_DISTANCE_M = 0.065
_INVALID_HAND_DISTANCE_M = 1.0e3
# Finite motion limits for live cloth manipulation.
# At 30 Hz, allow 15 mm of target lead over physical palm readback.
# The hand/table envelope below still constrains motion near the cloth.
FLEXIBLE_TOWEL_ARM_TRANSLATION_SPEED_MPS = 0.45
FLEXIBLE_TOWEL_ARM_ROTATION_SPEED_RAD_S = 2.00
# A 20-degree wrist preload dominated the 11.7 mm position step in DLS:
# downward commands could move the palm upward and never reach the cloth.
# Keep wrist slew, but bound its physical lag to one small rotation step.
FLEXIBLE_TOWEL_WRIST_MAX_LEAD_RAD = math.radians(4.0)


def _slew_wrist_quaternion(
    current: np.ndarray, target: np.ndarray, max_angle: float
) -> np.ndarray:
    """Advance on the shortest quaternion arc, including a full wrist flip."""
    from .bimanual_assembly_env import _normalize_quat_wxyz

    current = _normalize_quat_wxyz(current)
    target = _normalize_quat_wxyz(target)
    dot = float(np.dot(current, target))
    if dot < 0.0:
        target = -target
        dot = -dot
    half_angle = math.acos(float(np.clip(dot, 0.0, 1.0)))
    if 2.0 * half_angle <= max_angle or half_angle < 1e-8:
        return target.copy()
    fraction = max_angle / (2.0 * half_angle)
    return _normalize_quat_wxyz(
        (math.sin((1.0 - fraction) * half_angle) * current
         + math.sin(fraction * half_angle) * target) / math.sin(half_angle)
    )


class HandsOnFlexibleTowelDoubleFoldRotate90IsaacLabEnv(
    HandsOnEIsaacLabEnv
):
    """Two successive half folds in any directions, then a released hold."""

    _TASK_NAME = TASK_NAME
    _TASK_ID = TASK_ID
    _arm_translation_speed_mps = FLEXIBLE_TOWEL_ARM_TRANSLATION_SPEED_MPS
    _SURFACE_LOGICAL_NAME = "flexible_towel"
    _SURFACE_ROOT_PATH = G01_TOWEL_SURFACE_ROOT_PATH
    _SURFACE_ASSET_LABEL = "Marc Sawyer white towel with source-derived PBD collision sheet"

    # This full-articulation runtime can capture cameras live or defer them to
    # the canonical post-render pass selected by the collection frontend.
    _canonical_camera_postrender_supported = True
    # The dedicated cloth surface has no canonical rigid contact pair for the
    # generic origin signature. Requiring that signature makes startup loop
    # forever even though the robot/table pose is already stable.
    stable_recording_origin_gate_enabled = False

    def __init__(
        self,
        task_name: str = TASK_NAME,
        **kwargs: Any,
    ) -> None:
        self._towel_source = (
            MARC_SAWYER_TOWEL if task_name == TASK_NAME else SOURCES.get(task_name)
        )
        if task_name in SOURCES:
            self._arm_translation_speed_mps = 0.35
        if task_name != TASK_NAME and self._towel_source is None:
            raise ValueError(
                f"{TASK_ID} requires task_name={TASK_NAME!r}; got {task_name!r}"
            )
        self._TASK_NAME = task_name
        self._TASK_ID = self._towel_source.task_id if self._towel_source else TASK_ID
        if self._towel_source is not None:
            self._SURFACE_ASSET_LABEL = self._towel_source.label
        requested_hand = str(kwargs.get("hand_type", "wuji1")).strip().lower()
        requested_arm = str(kwargs.get("arm_type", "rokae")).strip().lower()
        if requested_hand not in {"wuji1", "wuji", "wuji_hand1", "wuji-hand1"}:
            raise ValueError(f"{TASK_ID} requires hand_type='wuji1'")
        if requested_arm != "rokae":
            raise ValueError(f"{TASK_ID} requires arm_type='rokae'")

        self._flexible_sequence = FlexibleTowelFoldSequence()
        self._flexible_initial_state = np.zeros(6, dtype=np.float64)
        self._flexible_last_metrics: dict[str, Any] = {}

        requested_hz = int(kwargs.get("hz", B320_CONTROL_HZ))
        if requested_hz != B320_CONTROL_HZ:
            raise ValueError(
                f"{TASK_ID} requires hz={B320_CONTROL_HZ}; got {requested_hz}"
            )
        expected_control_dt = 1.0 / B320_CONTROL_HZ
        expected_physics_dt = 1.0 / B320_PHYSICS_HZ
        if "control_dt" in kwargs and not np.isclose(
            float(kwargs["control_dt"]),
            expected_control_dt,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"{TASK_ID} requires control_dt=1/{B320_CONTROL_HZ}; "
                f"got {kwargs['control_dt']!r}"
            )
        if "physics_dt" in kwargs and not np.isclose(
            float(kwargs["physics_dt"]),
            expected_physics_dt,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"{TASK_ID} requires physics_dt=1/{B320_PHYSICS_HZ}; "
                f"got {kwargs['physics_dt']!r}"
            )
        for env_name, expected in (
            ("HANDSON_ISAACLAB_CONTROL_HZ", float(B320_CONTROL_HZ)),
            ("HANDSON_ISAACLAB_CONTROL_DT", expected_control_dt),
        ):
            raw_value = os.environ.get(env_name)
            if raw_value is None or not raw_value.strip():
                continue
            try:
                configured_value = float(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"{TASK_ID} requires {env_name}={expected!r}; "
                    f"got {raw_value!r}"
                ) from exc
            if not math.isfinite(configured_value) or not math.isclose(
                configured_value,
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"{TASK_ID} requires {env_name}={expected!r}; "
                    f"got {raw_value!r}"
                )

        tuning = E01SurfacePhysicsTuning(
            contact_offset=FLEXIBLE_TOWEL_CONTACT_OFFSET,
            rest_offset=FLEXIBLE_TOWEL_REST_OFFSET,
            surface_thickness=cotton.THICKNESS,
            surface_density=cotton.DENSITY,
            dynamic_friction=FLEXIBLE_TOWEL_DYNAMIC_FRICTION,
            surface_stretch_stiffness=cotton.STRETCH_STIFFNESS,
            surface_shear_stiffness=cotton.SHEAR_STIFFNESS,
            surface_bend_stiffness=cotton.BEND_STIFFNESS,
            elasticity_damping=cotton.ELASTICITY_DAMPING,
            max_linear_velocity=cotton.MAX_SPEED,
            max_depenetration_velocity=cotton.MAX_DEPENETRATION_SPEED,
            solver_position_iterations=(
                FLEXIBLE_TOWEL_SOLVER_POSITION_ITERATIONS
            ),
            collision_pair_update_frequency=cotton.COLLISION_UPDATES,
            collision_iteration_multiplier=cotton.COLLISION_ITERATIONS,
        )
        kwargs.setdefault("surface_physics_tuning", tuning)
        kwargs.setdefault("control_dt", expected_control_dt)
        kwargs.setdefault("physics_dt", expected_physics_dt)
        kwargs.setdefault("hz", B320_CONTROL_HZ)
        kwargs.setdefault("time_limit", 180.0)
        # The 41x41 cloth can legitimately move several millimetres in one
        # control frame when a Wuji finger first makes contact.  The generic
        # E01 readback guard treats that expected impulse as a terminal
        # garment failure, which makes the collection UI immediately reset
        # (appearing as a flash/crash).  Keep topology and finite-value checks,
        # but let this task's ordered contract validate the motion instead.
        kwargs.setdefault("relaxed_garment_readback_guards", True)
        kwargs["physics_backend"] = "physx"
        kwargs["hand_type"] = "wuji1"
        kwargs["arm_type"] = "rokae"
        super().__init__(task_name=task_name, **kwargs)
        if (
            int(self.hz) != B320_CONTROL_HZ
            or not math.isclose(
                float(self._control_dt),
                expected_control_dt,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                float(self._physics_dt),
                expected_physics_dt,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                float(self._sim_dt),
                expected_physics_dt,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or int(self._n_substeps) != 4
        ):
            raise RuntimeError(
                f"{TASK_ID} did not realize its required 30/120 Hz, "
                "four-substep timing contract"
            )
        self._sync_b320_tcp_command_poses_to_physical_state()

    def _sync_b320_tcp_command_poses_to_physical_state(self) -> None:
        """Use the shared post-reset hook to align this task's DLS commands.

        The inherited E01 nominal commands differ from the actual FLEXIBLE
        palm homes. Keeping those commands as the zero point makes the first
        operator action translate and rotate an otherwise stationary hand.
        Only controller buffers are synchronized; physical poses stay intact.
        """
        if not self._launch_isaac or not getattr(self, "_robots", None):
            return
        from .bimanual_assembly_env import _matrix_to_quat_wxyz

        poses = []
        for matrix in self.get_end_effector_pose_matrix():
            matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
            if not np.isfinite(matrix).all():
                raise ValueError("FLEXIBLE-001 cannot synchronize a non-finite palm pose")
            poses.append(np.r_[matrix[:3, 3], _matrix_to_quat_wxyz(matrix[:3, :3])])
        self._right_tcp_pose, self._left_tcp_pose = poses
        self._arm_pose_filters = {"right": None, "left": None}

    def _surface_mesh_metadata(self) -> dict[str, Any]:
        if self._towel_source is MARC_SAWYER_TOWEL:
            return self._towel_source.surface_metadata()
        metadata = flexible_towel_surface_metadata()
        if self._towel_source:
            metadata.update(label=self._towel_source.label,
                            solver_label=self._towel_source.solver,
                            nominal_mass_kg=self._towel_source.mass_kg)
        return metadata

    def _interface_surface_nodes(self) -> np.ndarray:
        if self._towel_source:
            points, _ = self._towel_source.topology()
            nodes = np.asarray(points, dtype=np.float64) + FLEXIBLE_TOWEL_PLACEMENT
            nodes[:, 2] += getattr(self._towel_source, "spawn_lift", 0.0)
            return nodes
        return flexible_towel_interface_nodes()

    def _spawn_surface_deformable(self) -> str:
        if self._stage is None:
            raise RuntimeError(f"{TASK_ID} stage is unavailable before towel spawn")
        import isaaclab.sim as sim_utils

        return spawn_flexible_towel_surface(self._stage, sim_utils, self._towel_source)

    def _create_surface_tensor_view(self, physics_sim_view):
        if self._towel_source and self._towel_source.solver == "physx_particle_cloth":
            return ParticleClothViewAdapter(
                physics_sim_view.create_particle_cloth_view(G01_TOWEL_SURFACE_MESH_PATH)
            )
        return super()._create_surface_tensor_view(physics_sim_view)

    def _configure_randomizable_table_physics(self) -> tuple[int, bool]:
        """The task authors one fixed B320 worktable and never translates it."""

        return 0, False

    def _configure_task_robot_cfg(self, robot_cfg: Any, side: str) -> Any:
        robot_cfg = super()._configure_task_robot_cfg(robot_cfg, side)
        hand = getattr(robot_cfg, "actuators", {}).get("hand")
        if hand is not None:
            # The tracker joint target is the simulated target.  The previous
            # 2 rad/s cap made the Wuji fingers trail it; leave the authored
            # implicit drive's damping/effort in place without an extra
            # simulation-only velocity limiter.
            hand.velocity_limit_sim = None
        return robot_cfg

    def _physx_contact_arm_target_limits_enabled(self) -> bool:
        return True

    def _arm_control_diagnostic(self, side: str) -> str:
        from .flexible_hand_colliders import towel_hand_diagnostic

        return towel_hand_diagnostic(self, side)

    def _limit_arm_pose_target(
        self, side: str, robot: Any, pose: np.ndarray
    ) -> np.ndarray:
        """Pass the live TCP target through with only physical safety guards.

        The collection profile already disables the shared EMA, deadbands and
        joint/Cartesian slew limits.  Reapplying a task-local velocity cap here
        made a tracker displacement arrive several frames late and broke the
        required one-to-one hand/simulation mapping.  Workspace and table
        clearance remain hard guards because they prevent invalid IK/contact
        states rather than smoothing operator input.
        """
        from .bimanual_assembly_env import (
            _ARM_WORKSPACE_MAX_RADIUS,
            _float_env,
            _matrix_to_quat_wxyz,
            _normalize_quat_wxyz,
        )

        raw = np.asarray(pose, dtype=np.float64).reshape(7)
        actual = self._robot_body_pose_matrix(robot, self._ee_body_indices[side])
        current_quat = _matrix_to_quat_wxyz(actual[:3, :3])
        target_pos = raw[:3] if np.all(np.isfinite(raw[:3])) else actual[:3, 3]
        target_quat = (
            _normalize_quat_wxyz(raw[3:7])
            if np.all(np.isfinite(raw[3:7]))
            else current_quat
        )
        max_radius = _float_env(
            "HANDSON_ISAACLAB_ARM_WORKSPACE_MAX_RADIUS",
            _ARM_WORKSPACE_MAX_RADIUS,
        )
        if max_radius > 0.0:
            # Interface tests and pre-spawn diagnostics may pass a sentinel
            # robot object.  The live Isaac path always has a base pose; keep
            # the safety clamp there while allowing the direct-pose contract
            # to be tested without constructing a simulator articulation.
            try:
                base_pos = self._robot_base_pose_matrix(side, robot)[:3, 3]
            except (AttributeError, IndexError, TypeError):
                base_pos = None
            if base_pos is not None:
                distance = float(np.linalg.norm(target_pos - base_pos))
                if distance > max_radius:
                    target_pos = base_pos + (target_pos - base_pos) * (max_radius / distance)
        limited = np.concatenate([target_pos, target_quat]).astype(np.float64, copy=False)
        limited = self._project_hand_table_target(side, robot, actual, limited)
        self._arm_pose_filters[side] = {
            "raw_pos": target_pos.copy(),
            "raw_quat": target_quat.copy(),
            "pos": limited[:3].copy(),
            "quat": limited[3:7].copy(),
        }
        return limited

    def _project_hand_table_target(self, side, robot, actual_palm, target):
        """Keep the requested hand envelope above a single towel on the table.

        Native contact tests showed that a stiff arm target below the table
        squeezed this light surface through its support. This is a Cartesian
        working-clearance constraint, not a cloth force or state correction.
        The shared detached-hand guard uses different XML primitive colliders;
        this full articulation must use its actual CAD-derived hulls instead.
        """
        if self._stage is None or not getattr(self, "_flexible_hand_collider_audit", None):
            return target
        from .bimanual_assembly_env import (
            _matrix_to_quat_wxyz, _quat_wxyz_to_matrix,
        )

        cache = getattr(self, "_flexible_hull_points_by_side", None)
        if cache is None:
            from pxr import UsdGeom

            cache = {"right": [], "left": []}
            for row in self._flexible_hand_collider_audit:
                path = row["collider"]
                owner = "right" if "/RightRobot/" in path else "left"
                prim = self._stage.GetPrimAtPath(path)
                name = prim.GetParent().GetName()
                index = self._robots[owner].body_names.index(name)
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
                cache[owner].append((index, points))
            self._flexible_hull_points_by_side = cache
        world = []
        for index, points in cache[side]:
            body = self._robot_body_pose_matrix(robot, index)
            world.append(points @ body[:3, :3].T + body[:3, 3])
        vertices = np.vstack(world)
        relative = (vertices - actual_palm[:3, 3]) @ actual_palm[:3, :3]
        floor = B320_TABLE_TOP_Z + getattr(
            getattr(self, "_towel_source", None), "thickness", cotton.THICKNESS
        )

        def project(quaternion):
            predicted = relative @ _quat_wxyz_to_matrix(quaternion).T + target[:3]
            projected = target.copy()
            projected[3:7] = quaternion
            projected[2] += max(0.0, floor - float(predicted[:, 2].min()))
            return projected

        # Clearance is the only remaining projection.  Do not use it as a
        # translation or wrist slew limiter: an operator target must reach the
        # simulated TCP in the same control tick.
        return project(target[3:7])

    def _configure_task_collision_safety(
        self, sim_utils: Any
    ) -> tuple[int, int]:
        """Prevent thin-cloth/finger tunnelling at the 120 Hz contact rate."""

        if self._stage is None:
            return 0, 0

        from pxr import PhysxSchema, Usd, UsdPhysics

        # Imported hand collision shapes are instance proxies, so a normal
        # traversal only reaches their rigid links. Make the collision branches
        # editable before physics initialization to apply thin-cloth offsets to
        # the actual shapes as well as the inherited friction material.
        collision_instances = [
            prim for prim in self._stage.Traverse()
            if prim.IsInstance()
            and any(part in str(prim.GetPath()).lower() for part in ("finger", "palm"))
            and "collisions" in str(prim.GetPath()).lower()
        ]
        for prim in collision_instances:
            prim.SetInstanceable(False)
        hand_bodies, hand_colliders = (
            super()._configure_task_collision_safety(sim_utils)
        )

        tuned_bodies = 0
        cloth_root = self._stage.GetPrimAtPath(G01_TOWEL_SURFACE_ROOT_PATH)
        cloth_ccd = cloth_root.GetAttribute(
            "physxDeformableBody:enableSpeculativeCCD"
        )
        if cloth_ccd.IsValid():
            cloth_ccd.Set(True)
            tuned_bodies += 1

        robots_root = self._stage.GetPrimAtPath("/World/HandsOnRobots")
        if robots_root.IsValid():
            for prim in Usd.PrimRange(robots_root):
                path_lower = str(prim.GetPath()).lower()
                is_hand = "palm" in path_lower or "finger" in path_lower
                if is_hand and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                    rigid_api.CreateEnableSpeculativeCCDAttr().Set(True)
                    rigid_api.CreateSolverPositionIterationCountAttr().Set(64)
                    rigid_api.CreateSolverVelocityIterationCountAttr().Set(8)
                    rigid_api.CreateMaxDepenetrationVelocityAttr().Set(0.60)
                    tuned_bodies += 1

        tuned_colliders = 0
        mesh = self._stage.GetPrimAtPath(G01_TOWEL_SURFACE_MESH_PATH)
        if mesh.IsValid() and not (
            self._towel_source and self._towel_source.solver == "physx_particle_cloth"
        ):
            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(mesh)
            collision_api.CreateContactOffsetAttr().Set(
                FLEXIBLE_TOWEL_CONTACT_OFFSET
            )
            collision_api.CreateRestOffsetAttr().Set(FLEXIBLE_TOWEL_REST_OFFSET)
            tuned_colliders = 1
        from .flexible_hand_colliders import author_hand_contact_hulls

        self._flexible_hand_collider_audit = author_hand_contact_hulls(
            self._stage,
            contact_offset=cotton.HAND_CONTACT_OFFSET,
            rest_offset=cotton.HAND_REST_OFFSET,
        )
        print(
            "[HandsOn/IsaacLab] FLEXIBLE hand contact hulls from visible geometry: "
            f"{len(self._flexible_hand_collider_audit)} components",
            flush=True,
        )
        tuned_colliders += len(self._flexible_hand_collider_audit)
        return tuned_bodies + hand_bodies, tuned_colliders + hand_colliders

    def _uses_cloth_hand_contact_material(self) -> bool:
        return True

    def _bind_f01_hand_contact_material(
        self, material_path: str
    ) -> tuple[int, int, int]:
        # The shared runtime binds this material again after initialization.
        # Apply the towel offsets on both calls so the second binding cannot
        # silently replace them with the thin F01 garment defaults.
        counts = super()._bind_f01_hand_contact_material(material_path)
        if self._stage is not None:
            from pxr import PhysxSchema, UsdPhysics, UsdShade

            for prim in self._stage.Traverse():
                path = str(prim.GetPath()).lower()
                if (
                    path.startswith("/world/handsonrobots/")
                    and ("palm" in path or "finger" in path)
                    and prim.HasAPI(UsdPhysics.CollisionAPI)
                ):
                    shape = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    shape.CreateContactOffsetAttr().Set(cotton.HAND_CONTACT_OFFSET)
                    shape.CreateRestOffsetAttr().Set(cotton.HAND_REST_OFFSET)
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                        UsdShade.Material(self._stage.GetPrimAtPath(material_path)),
                        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                        materialPurpose="physics",
                    )
        return counts

    def _sample_reset_state(self, *, clear_manual_request: bool) -> None:
        # Use the dedicated tabletop region, with replayable XY/yaw/wrinkle
        # variation and a fixed physical table.
        HandsOnABCIsaacLabEnv._sample_reset_state(
            self,
            clear_manual_request=clear_manual_request,
        )
        self.delta_h[:] = 0.0
        if self.randomize:
            self._flexible_initial_state = np.asarray(
                [
                    self._random.uniform(*FLEXIBLE_TOWEL_RESET_X_BOUNDS_M),
                    self._random.uniform(*FLEXIBLE_TOWEL_RESET_Y_BOUNDS_M),
                    math.radians(self._random.uniform(-5.0, 5.0)),
                    self._random.uniform(0.0, 0.002),
                    float(self._random.randint(0, 3)),
                    0.0,
                ],
                dtype=np.float64,
            )
        else:
            self._flexible_initial_state = np.zeros(6, dtype=np.float64)
        self._e_initial_state = self._flexible_initial_state.copy()
        self._flexible_sequence.reset()
        self._flexible_last_metrics = {}
        self._e_last_metrics = {}
        self._clear_garment_failure_state()

    def _apply_task_pose_randomization(self) -> None:
        if self._towel_source and self._towel_source.solver == "physx_particle_cloth":
            # Particle cloth has working position/velocity reset tensors. Its
            # actor must not use the FEM-specific deactivate/recook workaround.
            super()._apply_task_pose_randomization()
            return
        if self._e_garment_view is None or self._stage is None or self._sim is None:
            super()._apply_task_pose_randomization()
            return

        # In 107.3 the surface tensor setter reads back the requested position,
        # but the next integration step can resume the previous solver state
        # and launch the cloth sideways. Recreate this surface at reset only;
        # retain the robot views and the ordinary dynamic simulation thereafter.
        from pxr import Usd, UsdGeom, Vt

        nodes = self._garment_reset_nodes()
        issue, diagnostics = self._basic_nodal_state_issue(
            nodes, np.zeros_like(nodes),
            expected_node_count=self._e_expected_node_count,
        )
        if issue is not None:
            raise RuntimeError(f"Invalid towel reset: {issue}, {diagnostics}")
        root = self._stage.GetPrimAtPath(self._SURFACE_ROOT_PATH)
        mesh = UsdGeom.Mesh(self._stage.GetPrimAtPath(G01_TOWEL_SURFACE_MESH_PATH))
        inverse = np.asarray(
            UsdGeom.XformCache(Usd.TimeCode.Default())
            .GetLocalToWorldTransform(mesh.GetPrim()).GetInverse()
        )
        local = (np.c_[nodes, np.ones(len(nodes))] @ inverse)[:, :3]
        local = np.asarray(local, dtype=np.float32)
        root.SetActive(False)
        try:
            self._step_isaaclab()
        finally:
            root.SetActive(True)
        mesh = UsdGeom.Mesh(self._stage.GetPrimAtPath(G01_TOWEL_SURFACE_MESH_PATH))
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(local))
        mesh.CreateVelocitiesAttr(Vt.Vec3fArray.FromNumpy(np.zeros_like(local)))
        self._step_isaaclab()
        self._bind_e01_surface_view(initialize_baseline=False)
        if np.max(np.linalg.norm(self._e_nodes_np[:, :2] - nodes[:, :2], axis=1)) > 1e-4:
            raise RuntimeError("Towel reset did not retain its sampled XY pose")

    @staticmethod
    def _finite_distance(value: float) -> float:
        return float(value) if math.isfinite(value) else _INVALID_HAND_DISTANCE_M

    def _side_hand_distance(self, side: str, nodes: np.ndarray) -> float:
        stride = max(1, int(nodes.shape[0]) // 512)
        samples = nodes[::stride]
        points: list[np.ndarray] = []
        try:
            points.extend(
                np.asarray(point, dtype=np.float64).reshape(3)
                for point in self._a06_hand_points(side)
            )
        except Exception:
            points = []
        if not points:
            pose = self._left_tcp_pose if side == "left" else self._right_tcp_pose
            point = np.asarray(pose[:3], dtype=np.float64)
            if point.shape == (3,) and bool(np.all(np.isfinite(point))):
                points.append(point)
        if not points:
            return _INVALID_HAND_DISTANCE_M
        return self._finite_distance(
            min(
                float(np.min(np.linalg.norm(samples - point, axis=1)))
                for point in points
            )
        )

    def _raw_towel_metrics(self) -> dict[str, Any]:
        self._ensure_metric_state_is_safe()
        fold = compute_fold_metrics(
            self._e_nodes_np,
            self._e_velocities_np,
            self._e_reset_nodes_np,
            B320_TABLE_TOP_Z,
        )
        left_distance = self._side_hand_distance("left", self._e_nodes_np)
        right_distance = self._side_hand_distance("right", self._e_nodes_np)
        both_hands_engaged = bool(
            left_distance <= _HAND_ENGAGED_DISTANCE_M
            and right_distance <= _HAND_ENGAGED_DISTANCE_M
        )
        hands_released = bool(
            left_distance >= _HAND_RELEASE_DISTANCE_M
            and right_distance >= _HAND_RELEASE_DISTANCE_M
        )
        payload = fold.to_dict()
        payload.update(
            {
                "bounds_min": np.asarray(fold.bounds_min, dtype=np.float64),
                "bounds_max": np.asarray(fold.bounds_max, dtype=np.float64),
                "centroid": np.mean(self._e_nodes_np, axis=0),
                "left_hand_distance": left_distance,
                "right_hand_distance": right_distance,
                "both_hands_engaged": both_hands_engaged,
                "hands_released": hands_released,
                "success_now": False,
            }
        )
        return payload

    @staticmethod
    def _snapshot_fields(snapshot: FoldSequenceSnapshot) -> dict[str, Any]:
        return {
            "fold_stage": snapshot.stage,
            "flat_latched": snapshot.flat_latched,
            "first_fold_latched": snapshot.first_fold_latched,
            "rotation_latched": snapshot.rotation_latched,
            "second_fold_latched": snapshot.second_fold_latched,
            "release_latched": snapshot.release_latched,
            "first_fold_yaw_rad": snapshot.first_fold_yaw_rad,
            "last_yaw_rad": snapshot.last_yaw_rad,
            "cumulative_rotation_rad": snapshot.cumulative_rotation_rad,
            "rotation_direction": snapshot.rotation_direction,
            "flat_stable_frames": snapshot.flat_stable_frames,
            "release_stable_frames": snapshot.release_stable_frames,
            "sequence_update_count": snapshot.update_count,
            "sequence_ready": snapshot.ready,
            "sequence_success": snapshot.success,
        }

    def _garment_metrics(self) -> dict[str, Any]:
        metrics = self._raw_towel_metrics()
        snapshot = self._flexible_sequence.snapshot()
        metrics.update(self._snapshot_fields(snapshot))
        metrics["success_now"] = bool(
            snapshot.success and not self._e_readback_failed
        )
        return metrics

    def _refresh_garment_state(self) -> None:
        # The shared method performs strict shape/finite/speed/displacement
        # readback validation before task geometry sees the tensor.
        super()._refresh_garment_state()
        if self._stage is not None and not self._e_readback_failed:
            update_visual = getattr(
                self._towel_source, "update_visual", cotton.update_towel_shell
            )
            update_visual(
                self._stage, self._SURFACE_ROOT_PATH, self._e_nodes_np
            )
        # The parent just computed these metrics from the validated nodal
        # readback. Reuse that snapshot instead of repeating hand readbacks
        # and cloth geometry queries in the same control frame.
        metrics = dict(self._e_last_metrics)
        snapshot = self._flexible_sequence.update(
            metrics,
            both_hands_engaged=metrics["both_hands_engaged"],
            hands_released=metrics["hands_released"],
        )
        metrics.update(self._snapshot_fields(snapshot))
        metrics["success_now"] = bool(
            snapshot.success and not self._e_readback_failed
        )
        self._flexible_last_metrics = metrics
        self._e_last_metrics = dict(metrics)

    @staticmethod
    def _event_observation(snapshot: FoldSequenceSnapshot) -> np.ndarray:
        parts = snapshot.observation()
        return np.asarray(
            [
                *parts["stage"],
                *parts["latches"],
                *parts["rotation"],
                *parts["counters"],
                *parts["terminal"],
            ],
            dtype=np.float64,
        )

    def _build_observation_space(self) -> spaces.Dict:
        state_spaces = {
            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
            "gripper_pose": spaces.Box(
                -np.inf, np.inf, shape=(32,), dtype=np.float64
            ),
            "towel_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            "towel_keypoints": spaces.Box(
                -np.inf, np.inf, shape=(24,), dtype=np.float64
            ),
            "towel_aabb": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64),
            "fold_metrics": spaces.Box(
                -np.inf, np.inf, shape=(20,), dtype=np.float64
            ),
            "rotation_metrics": spaces.Box(
                -np.inf, np.inf, shape=(8,), dtype=np.float64
            ),
            "task_phase": spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float64),
            "fold_event_state": spaces.Box(
                -np.inf, np.inf, shape=(16,), dtype=np.float64
            ),
            "towel_initial_state": spaces.Box(
                -np.inf, np.inf, shape=(6,), dtype=np.float64
            ),
            "table_delta_height": spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float64
            ),
        }
        observation_spaces: dict[str, spaces.Space] = {
            "state": spaces.Dict(state_spaces),
            "e01_hand_qpos": spaces.Box(
                -np.inf, np.inf, shape=(40,), dtype=np.float64
            ),
        }
        if self.image_obs:
            image_space = lambda: spaces.Box(
                0, 255, shape=self._image_shape, dtype=np.uint8
            )
            observation_spaces["images"] = spaces.Dict(
                {
                    "random_camera" if self.randomize else "front": image_space(),
                    "wrist_left": image_space(),
                    "wrist_right": image_space(),
                }
            )
        return spaces.Dict(observation_spaces)

    def _compute_observation(self) -> dict[str, Any]:
        metrics = self._flexible_last_metrics or self._garment_metrics()
        snapshot = self._flexible_sequence.snapshot()
        tcp, gripper = self._robot_state()
        stage_index = tuple(FoldStage).index(FoldStage(snapshot.stage))
        phase = np.zeros(len(FoldStage), dtype=np.float64)
        phase[stage_index] = 1.0
        state = {
            "tcp_pose": tcp,
            "gripper_pose": gripper,
            "towel_pose": self._garment_pose(),
            "towel_keypoints": self._e_nodes_np[
                self._e_keypoint_indices
            ].reshape(-1).astype(np.float64, copy=True),
            "towel_aabb": np.concatenate(
                [metrics["bounds_min"], metrics["bounds_max"]]
            ).astype(np.float64),
            "fold_metrics": np.asarray(
                [
                    metrics["row_mirror_xy_error"],
                    metrics["column_mirror_xy_error"],
                    metrics["oriented_footprint_ratio"],
                    metrics["edge_stretch"],
                    metrics["bounds_max"][2] - metrics["bounds_min"][2],
                    metrics["support_fraction"],
                    metrics["rms_speed"],
                    metrics["max_speed"],
                    metrics["flat"],
                    metrics["first_fold"],
                    metrics["first_fold_maintained"],
                    metrics["final_fold"],
                    metrics["final_fold_maintained"],
                    metrics["supported"],
                    metrics["in_workspace"],
                    metrics["stable"],
                    metrics["finite"],
                    metrics["left_hand_distance"],
                    metrics["right_hand_distance"],
                    metrics["both_hands_engaged"],
                ],
                dtype=np.float64,
            ),
            "rotation_metrics": np.asarray(
                [
                    metrics["material_x_yaw"],
                    snapshot.first_fold_yaw_rad,
                    snapshot.last_yaw_rad,
                    snapshot.cumulative_rotation_rad,
                    snapshot.rotation_direction,
                    abs(snapshot.cumulative_rotation_rad),
                    ROTATION_TARGET_RAD,
                    ROTATION_TOLERANCE_RAD,
                ],
                dtype=np.float64,
            ),
            "task_phase": phase,
            "fold_event_state": self._event_observation(snapshot),
            "towel_initial_state": self._flexible_initial_state.astype(
                np.float64, copy=True
            ),
            "table_delta_height": self.delta_h.astype(np.float64, copy=True),
        }
        observation: dict[str, Any] = {
            "state": state,
            "e01_hand_qpos": self._e01_recording_hand_qpos(),
        }
        if self.image_obs:
            front, wrist_left, wrist_right = self.render()
            observation["images"] = {
                "random_camera" if self.randomize else "front": front,
                "wrist_left": wrist_left,
                "wrist_right": wrist_right,
            }
        return observation

    def _compute_success_now(self) -> bool:
        snapshot = self._flexible_sequence.snapshot()
        return bool(snapshot.success and not self._e_readback_failed)

    def _success_condition_rows(self) -> list[dict[str, Any]]:
        metrics = self._flexible_last_metrics or self._garment_metrics()
        snapshot = self._flexible_sequence.snapshot()
        return [
            self._condition_row(
                "flexible_flat_start",
                "A stable flat reset was observed before manipulation",
                snapshot.flat_latched,
            ),
            self._condition_row(
                "flexible_first_fold",
                "The first half fold was completed in any direction",
                snapshot.first_fold_latched,
            ),
            self._condition_row(
                "flexible_second_fold",
                "The second half fold was completed in the same or another direction",
                snapshot.second_fold_latched,
            ),
            self._condition_row(
                "flexible_release",
                "Both hands are at least 65 mm clear of the final fold",
                metrics["hands_released"] and snapshot.release_latched,
            ),
            self._condition_row(
                "flexible_release_stable",
                "The supported final fold is stable for 15 control frames",
                snapshot.release_stable_frames
                >= RELEASE_STABLE_FRAMES_REQUIRED,
            ),
            self._condition_row(
                "flexible_readback",
                "The 1681-node PhysX surface readback remains valid",
                not self._e_readback_failed,
            ),
        ]

    def _task_info(self) -> dict[str, Any]:
        info = super()._task_info()
        if self._towel_source is MARC_SAWYER_TOWEL:
            source = self._towel_source
            # The inherited fields describe FEM, which is not the active solver.
            # Keep dimensional geometry fields truthful and report particle
            # coefficients with their native names in towel_source below.
            for key in (
                "garment_surface_stretch_stiffness", "garment_surface_shear_stiffness",
                "garment_surface_bend_stiffness", "garment_elasticity_damping",
            ):
                info.pop(key, None)
            info.update(
                garment_solver=source.solver,
                garment_contact_offset=1.4 * source.thickness / 2,
                garment_surface_thickness=source.thickness,
                garment_surface_density=source.mass_kg / (
                    source.metadata()["area_m2"] * source.thickness
                ),
            )
        metrics = self._flexible_last_metrics or self._garment_metrics()
        snapshot = self._flexible_sequence.snapshot()
        info.update(
            {
                "task_id": self._TASK_ID,
                "task_name": self._TASK_NAME,
                "towel_source": self._towel_source.metadata() if self._towel_source else None,
                "towel_surface_root": G01_TOWEL_SURFACE_ROOT_PATH,
                "towel_surface_only": True,
                "towel_node_count": MATERIAL_NODE_COUNT,
                "fold_sequence": snapshot.to_dict(),
                "fold_contract_version": FOLD_CONTRACT_VERSION,
                "fold_metrics": {
                    key: value.tolist()
                    if isinstance(value, np.ndarray)
                    else value
                    for key, value in metrics.items()
                    if key not in {"centroid"}
                },
                "replay_mode": "diagnostic_state_guided",
                "physical_replay_certified": False,
                "collection_verified": False,
            }
        )
        return info

    def get_physical_replay_reset_contract(self) -> dict[str, Any]:
        contract = dict(super().get_physical_replay_reset_contract())
        contract.update(
            {
                "flexible_task_id": self._TASK_ID,
                "towel_source": self._towel_source.metadata() if self._towel_source else None,
                "fold_contract_version": FOLD_CONTRACT_VERSION,
                "fold_order": "two_half_folds_any_direction",
                "reset_center_x_offsets_m": list(FLEXIBLE_TOWEL_RESET_X_BOUNDS_M),
                "reset_center_y_offsets_m": list(FLEXIBLE_TOWEL_RESET_Y_BOUNDS_M),
                "surface_root": G01_TOWEL_SURFACE_ROOT_PATH,
                "surface_only": True,
                "surface_node_count": MATERIAL_NODE_COUNT,
                "physics_hz": B320_PHYSICS_HZ,
                "control_hz": B320_CONTROL_HZ,
                "ordered_hidden_state_field": "state/fold_event_state",
                "replay_mode": "diagnostic_state_guided",
                "physical_replay_certified": False,
            }
        )
        return contract

    @staticmethod
    def _restore_sequence_observation(values: np.ndarray) -> dict[str, Any]:
        raw = np.asarray(values, dtype=np.float64).reshape(-1)
        if raw.shape != (16,) or not bool(np.all(np.isfinite(raw))):
            raise ValueError("fold_event_state must contain 16 finite values")
        stage_index = int(round(float(raw[0])))
        stages = tuple(FoldStage)
        if stage_index < 0 or stage_index >= len(stages):
            raise ValueError("fold_event_state contains an invalid stage")
        return {
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "stage": stages[stage_index].value,
            "flat_latched": bool(raw[1]),
            "first_fold_latched": bool(raw[2]),
            "rotation_latched": bool(raw[3]),
            "second_fold_latched": bool(raw[4]),
            "release_latched": bool(raw[5]),
            "first_fold_yaw_rad": float(raw[6]),
            "last_yaw_rad": float(raw[7]),
            "cumulative_rotation_rad": float(raw[8]),
            "rotation_direction": int(round(float(raw[9]))),
            "update_count": int(round(float(raw[10]))),
            "flat_stable_frames": int(round(float(raw[11]))),
            "release_stable_frames": int(round(float(raw[12]))),
            "last_metrics_finite": bool(raw[13]),
            "ready": bool(raw[14]),
            "success": bool(raw[15]),
        }

    def restore_recorded_initial_state(
        self, state_parts: Mapping[str, np.ndarray]
    ) -> dict[str, Any]:
        initial = np.asarray(
            state_parts.get("towel_initial_state", self._flexible_initial_state),
            dtype=np.float64,
        ).reshape(-1)
        if initial.size < 6 or not bool(np.all(np.isfinite(initial[:6]))):
            raise ValueError("towel_initial_state must contain six finite values")
        self._flexible_initial_state = initial[:6].copy()
        self._flexible_initial_state[5] = 0.0
        self._e_initial_state = self._flexible_initial_state.copy()

        translated = dict(state_parts)
        translated["garment_initial_state"] = self._e_initial_state.copy()
        translated["table_delta_height"] = np.zeros(1, dtype=np.float64)
        super().restore_recorded_initial_state(translated)

        event_state = state_parts.get("fold_event_state")
        if event_state is None:
            self._flexible_sequence.reset()
        else:
            self._flexible_sequence.restore(
                self._restore_sequence_observation(event_state)
            )
        self._flexible_last_metrics = self._garment_metrics()
        self._e_last_metrics = dict(self._flexible_last_metrics)
        return self._compute_observation()


__all__ = ["HandsOnFlexibleTowelDoubleFoldRotate90IsaacLabEnv"]
