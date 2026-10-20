#!/usr/bin/env python3
"""Bounded Isaac Sim 5.1 core-scene validation for B320-FU015.

The runner composes the exact laptop asset with the production dual
ROKAE/Wuji rig, table, task lighting, and robot head camera. It performs a
30-step zero-action reset smoke only; no grasp, path, opening action, or
success motion is executed. The separate task-level completion gate remains
10 seconds and is deliberately not claimed by this core-scene package.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .core_scene_spec import (
    BASE_PATH,
    CONTROL_HZ,
    DISPLAY_PATH,
    EXPECTED_COLLISION_PATH_SUFFIXES,
    FOUNDATION_SCHEMA,
    HEAD_CAMERA_PATH,
    HEAD_SHAPE_HWC,
    HINGE_DRIVE_DAMPING,
    HINGE_DRIVE_MAX_FORCE,
    HINGE_DRIVE_STIFFNESS,
    HINGE_LOWER_DEG,
    HINGE_PATH,
    HINGE_UPPER_DEG,
    INITIAL_HINGE_DEG,
    INITIAL_HINGE_RAD,
    LAPTOP_RELATIVE,
    LAPTOP_ROOT,
    PHYSICS_HZ,
    PLACEMENT_XY_M,
    PLACEMENT_YAW_DEG,
    REPOSITORY_ROOT,
    ROBOT_RELATIVE,
    ROBOT_SHA256,
    RUNTIME_TASK_NAME,
    SCALE_XYZ,
    SCENE_ROOT,
    SOURCE_SHA256,
    SOURCE_TASK_PROPS_TO_DEACTIVATE,
    TABLE_SUPPORT_GAP_M,
    ZERO_ACTION_SECONDS,
    ZERO_ACTION_STEPS,
)


HANDSON_PACKAGE_ROOT = REPOSITORY_ROOT / "handson"
sys.path.insert(0, str(HANDSON_PACKAGE_ROOT))

# AppLauncher must precede every remaining Isaac/Kit/USD import.
from isaaclab.app import AppLauncher  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _range_list(bound: Any) -> list[list[float]]:
    return [
        [float(value) for value in bound.GetMin()],
        [float(value) for value in bound.GetMax()],
    ]


def _main() -> int:
    args = _parser().parse_args()
    args.headless = True
    args.enable_cameras = True
    args.device = "cuda:0"
    args.width = 640
    args.height = 480
    args.kit_args = " ".join(
        filter(
            None,
            (
                str(getattr(args, "kit_args", "") or ""),
                "--/rtx/rendermode=RaytracedLighting",
                "--/rtx/post/tonemap/op=4",
                "--/rtx/post/tonemap/filmIso=200",
            ),
        )
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Post-app imports only.
    import carb  # noqa: E402
    from omni.physx import get_physx_simulation_interface  # noqa: E402
    from pxr import (  # noqa: E402
        Gf,
        PhysicsSchemaTools,
        PhysxSchema,
        Usd,
        UsdGeom,
        UsdLux,
        UsdPhysics,
    )

    from handson.sim.isaaclab.abc_collection_env import (  # noqa: E402
        HandsOnABCIsaacLabEnv,
    )
    from .core_scene_runtime import (  # noqa: E402
        aggregate_checks,
        atomic_write_json,
        decoded_rgb_metrics,
        head_content_pass,
        pose_drift_m,
        quaternion_angle_rad,
        require_recursive_finite,
        sha256_file,
    )

    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "RaytracedLighting")
    settings.set("/rtx/post/tonemap/op", 4)
    settings.set("/rtx/post/tonemap/filmIso", 200)

    started = time.time()
    environment = None
    contact_subscription = None
    summary: dict[str, Any] = {
        "schema": FOUNDATION_SCHEMA,
        "task": RUNTIME_TASK_NAME,
        "foundation_pass": False,
        "task_complete": False,
        "success_motion_executed": False,
        "path_or_action_tuning_executed": False,
        "object_state_writes_during_zero_action": 0,
    }
    rc = 1
    try:
        physical_gpu = os.environ.get("TASKDEV_PHYSICAL_GPU", "")
        logical_gpu = os.environ.get("TASKDEV_LOGICAL_GPU", "")
        run_dir = output_dir.parent
        namespace_path = run_dir / "gpu_namespace.json"
        namespace = json.loads(namespace_path.read_text(encoding="utf-8"))
        namespace_pass = bool(
            physical_gpu.isdigit()
            and int(physical_gpu) == int(namespace.get("physical_gpu", -1))
            and logical_gpu == "0"
            and namespace.get("logical_gpu") == 0
            and os.environ.get("CUDA_VISIBLE_DEVICES") is None
            and os.environ.get("TASKDEV_HOST") == "hp-13"
        )
        summary["scheduler_namespace"] = {
            "host": os.environ.get("TASKDEV_HOST"),
            "observed_hostname": socket.gethostname(),
            "physical_gpu": int(physical_gpu) if physical_gpu.isdigit() else -1,
            "logical_gpu": int(logical_gpu) if logical_gpu.isdigit() else -1,
            "mapping": namespace,
            "path": str(namespace_path),
            "sha256": sha256_file(namespace_path),
            "cuda_visible_devices_absent": os.environ.get("CUDA_VISIBLE_DEVICES") is None,
            "pass": namespace_pass,
        }

        laptop_asset = REPOSITORY_ROOT / LAPTOP_RELATIVE
        robot_asset = REPOSITORY_ROOT / ROBOT_RELATIVE
        asset_hashes = {
            str(LAPTOP_RELATIVE): sha256_file(laptop_asset),
            str(ROBOT_RELATIVE): sha256_file(robot_asset),
        }
        hashes_pass = bool(
            asset_hashes[str(LAPTOP_RELATIVE)] == SOURCE_SHA256
            and asset_hashes[str(ROBOT_RELATIVE)] == ROBOT_SHA256
        )

        environment = HandsOnABCIsaacLabEnv(
            "t02_connect_charger_and_cable",
            seed=0,
            control_dt=1.0 / float(CONTROL_HZ),
            physics_dt=1.0 / float(PHYSICS_HZ),
            time_limit=ZERO_ACTION_SECONDS + 5.0,
            render_mode="rgb_array",
            image_obs=True,
            randomize=False,
            randomize_dynamics=False,
            hand_type="wuji1",
            arm_type="rokae",
            launch_isaac=True,
            headless=True,
            render_resolution=(480, 640),
            hz=CONTROL_HZ,
            isaac_device="cuda:0",
            physics_backend="physx",
            existing_simulation_app=simulation_app,
            existing_app_launcher=app_launcher,
        )
        stage = environment._stage
        sim = environment._sim
        if stage is None or sim is None:
            raise RuntimeError("HandsOn environment did not expose stage/simulation")

        deactivated: dict[str, str] = {}
        for name in SOURCE_TASK_PROPS_TO_DEACTIVATE:
            prim = environment._find_prim(name)
            if prim is None or not prim.IsValid():
                raise RuntimeError(f"source task prop not found: {name}")
            deactivated[name] = str(prim.GetPath())
            prim.SetActive(False)

        root = UsdGeom.Xform.Define(stage, LAPTOP_ROOT).GetPrim()
        root.GetReferences().ClearReferences()
        root.GetReferences().AddReference(str(laptop_asset.resolve()))
        xform = UsdGeom.Xformable(root)
        xform.ClearXformOpOrder()
        table_top = float(environment._current_table_top_z())
        root_z = table_top + TABLE_SUPPORT_GAP_M
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(float(PLACEMENT_XY_M[0]), float(PLACEMENT_XY_M[1]), root_z)
        )
        xform.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set(PLACEMENT_YAW_DEG)

        base = stage.GetPrimAtPath(BASE_PATH)
        display = stage.GetPrimAtPath(DISPLAY_PATH)
        joint_prim = stage.GetPrimAtPath(HINGE_PATH)
        if not all(prim.IsValid() for prim in (base, display, joint_prim)):
            raise RuntimeError("composed laptop rigid bodies or hinge are missing")

        articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(base)
        articulation_api.CreateEnabledSelfCollisionsAttr(False)
        base_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(base)
        base_rigid.CreateDisableGravityAttr(False)
        display_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(display)
        display_rigid.CreateDisableGravityAttr(True)

        joint = UsdPhysics.RevoluteJoint(joint_prim)
        drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
        drive.CreateTargetPositionAttr(INITIAL_HINGE_DEG)
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateStiffnessAttr(HINGE_DRIVE_STIFFNESS)
        drive.CreateDampingAttr(HINGE_DRIVE_DAMPING)
        drive.CreateMaxForceAttr(HINGE_DRIVE_MAX_FORCE)
        if joint_prim.HasAPI(PhysxSchema.JointStateAPI, "angular"):
            joint_state = PhysxSchema.JointStateAPI.Get(joint_prim, "angular")
        else:
            joint_state = PhysxSchema.JointStateAPI.Apply(joint_prim, "angular")
        joint_state.GetPositionAttr().Set(INITIAL_HINGE_DEG)
        joint_state.GetVelocityAttr().Set(0.0)

        for prim in (base, display):
            report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            report.CreateThresholdAttr(0.0)

        dome = UsdLux.DomeLight.Define(stage, f"{SCENE_ROOT}/DomeLight")
        dome.CreateIntensityAttr(300.0)
        dome.CreateColorAttr(Gf.Vec3f(0.86, 0.90, 1.0))
        distant = UsdLux.DistantLight.Define(stage, f"{SCENE_ROOT}/DistantLight")
        distant.CreateIntensityAttr(2500.0)
        distant.CreateAngleAttr(0.53)
        distant.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.86))

        contact_rows: list[dict[str, Any]] = []
        contact_errors: list[str] = []
        physics_step = 0

        def on_contacts(headers: Any, data: Any) -> None:
            nonlocal physics_step
            try:
                for header in headers:
                    actor0 = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
                    actor1 = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
                    if not (actor0.startswith(LAPTOP_ROOT) or actor1.startswith(LAPTOP_ROOT)):
                        continue
                    start = int(header.contact_data_offset)
                    stop = start + int(header.num_contact_data)
                    for point_index in range(start, stop):
                        point = data[point_index]
                        impulse = np.asarray(point.impulse, dtype=np.float64)
                        contact_rows.append(
                            {
                                "physics_step": physics_step,
                                "actor0": actor0,
                                "actor1": actor1,
                                "collider0": str(
                                    PhysicsSchemaTools.intToSdfPath(header.collider0)
                                ),
                                "collider1": str(
                                    PhysicsSchemaTools.intToSdfPath(header.collider1)
                                ),
                                "separation_m": float(point.separation),
                                "impulse_ns": float(np.linalg.norm(impulse)),
                            }
                        )
            except Exception as exc:
                contact_errors.append(f"{type(exc).__name__}: {exc}")

        contact_subscription = (
            get_physx_simulation_interface().subscribe_contact_report_events(on_contacts)
        )
        # IsaacLab's hard reset performs two Hydra warm-up renders whenever
        # cameras are enabled. Keep reset/physics initialization render-free;
        # the original production-camera mode is restored for the one explicit
        # evidence frame after the physical summary has been persisted.
        production_render_mode = sim.render_mode
        sim.render_mode = sim.RenderMode.NO_RENDERING
        sim.reset()
        physics_view = environment._robots["right"]._physics_sim_view
        if physics_view is None:
            raise RuntimeError("post-reset PhysX simulation view is unavailable")
        articulation = physics_view.create_articulation_view(BASE_PATH)
        base_view = physics_view.create_rigid_body_view(BASE_PATH)
        display_view = physics_view.create_rigid_body_view(DISPLAY_PATH)
        if min(int(articulation.count), int(base_view.count), int(display_view.count)) < 1:
            raise RuntimeError("live laptop tensor views are empty")

        metatype = articulation.get_metatype(0)
        dof_indices = {str(name): int(index) for name, index in metatype.dof_indices.items()}
        if len(dof_indices) != 1:
            raise RuntimeError(f"expected one laptop hinge DOF, got {dof_indices}")

        initial_joint = _as_numpy(articulation.get_dof_positions()).astype(np.float64)
        initial_joint_velocity = _as_numpy(
            articulation.get_dof_velocities()
        ).astype(np.float64)
        initial_base_pose = _as_numpy(base_view.get_transforms())[0].astype(np.float64)
        initial_display_pose = _as_numpy(display_view.get_transforms())[0].astype(np.float64)

        peak_base_linear = 0.0
        peak_base_angular = 0.0
        peak_display_linear = 0.0
        peak_display_angular = 0.0
        peak_hinge_error = 0.0
        zero_action_started = time.time()
        for step in range(ZERO_ACTION_STEPS):
            for robot in environment._robots.values():
                robot.write_data_to_sim()
            physics_step = step + 1
            # Keep the stability observation independent of Hydra. The production
            # head camera is rendered once, after all physics evidence is durable.
            sim.step(render=False)
            for robot in environment._robots.values():
                robot.update(environment._sim_dt)
            if step % 12 == 0 or step == ZERO_ACTION_STEPS - 1:
                base_velocity = _as_numpy(base_view.get_velocities())[0].astype(np.float64)
                display_velocity = _as_numpy(display_view.get_velocities())[0].astype(
                    np.float64
                )
                hinge = float(_as_numpy(articulation.get_dof_positions())[0, 0])
                peak_base_linear = max(
                    peak_base_linear, float(np.linalg.norm(base_velocity[:3]))
                )
                peak_base_angular = max(
                    peak_base_angular, float(np.linalg.norm(base_velocity[3:]))
                )
                peak_display_linear = max(
                    peak_display_linear, float(np.linalg.norm(display_velocity[:3]))
                )
                peak_display_angular = max(
                    peak_display_angular, float(np.linalg.norm(display_velocity[3:]))
                )
                peak_hinge_error = max(peak_hinge_error, abs(hinge - INITIAL_HINGE_RAD))
        zero_action_wall_seconds = time.time() - zero_action_started

        final_joint = _as_numpy(articulation.get_dof_positions()).astype(np.float64)
        final_joint_velocity = _as_numpy(
            articulation.get_dof_velocities()
        ).astype(np.float64)
        final_base_pose = _as_numpy(base_view.get_transforms())[0].astype(np.float64)
        final_display_pose = _as_numpy(display_view.get_transforms())[0].astype(np.float64)
        final_base_velocity = _as_numpy(base_view.get_velocities())[0].astype(np.float64)
        final_display_velocity = _as_numpy(display_view.get_velocities())[0].astype(
            np.float64
        )

        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        collision_paths = []
        predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
        for prim in Usd.PrimRange(root, predicate):
            if prim.IsActive() and prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_paths.append(str(prim.GetPath()))
        collision_paths.sort()
        expected_collision_paths = sorted(
            f"{LAPTOP_ROOT}/{suffix}" for suffix in EXPECTED_COLLISION_PATH_SUFFIXES
        )
        world_bbox = _range_list(cache.ComputeWorldBound(root).ComputeAlignedRange())

        self_rows = [
            row
            for row in contact_rows
            if row["actor0"].startswith(LAPTOP_ROOT)
            and row["actor1"].startswith(LAPTOP_ROOT)
        ]
        unexpected_rows = [
            row
            for row in contact_rows
            if not (
                row["actor0"].startswith(LAPTOP_ROOT)
                and row["actor1"].startswith(LAPTOP_ROOT)
            )
            and "table" not in row["actor0"].lower()
            and "table" not in row["actor1"].lower()
        ]
        max_penetration = max(
            (max(0.0, -float(row["separation_m"])) for row in contact_rows),
            default=0.0,
        )
        base_drift = pose_drift_m(initial_base_pose, final_base_pose)
        base_angle_drift = quaternion_angle_rad(initial_base_pose, final_base_pose)
        final_hinge_error = abs(float(final_joint[0, 0]) - INITIAL_HINGE_RAD)

        camera = environment._cameras.get("head_camera")
        camera_prim_path = str(getattr(camera.cfg, "prim_path", "")) if camera else ""
        robot_inventory = {
            "names": sorted(environment._robots),
            "arm_joint_counts": {
                side: len(environment._arm_joint_ids[side]) for side in ("left", "right")
            },
            "hand_joint_counts": {
                side: len(environment._hand_joint_ids[side]) for side in ("left", "right")
            },
        }
        checks = {
            "scheduler_fresh_audit_and_lease": namespace_pass,
            "asset_hashes": hashes_pass,
            "identity_scale": tuple(SCALE_XYZ) == (1.0, 1.0, 1.0),
            "hinge_topology": bool(
                joint.GetAxisAttr().Get() == "X"
                and float(joint.GetLowerLimitAttr().Get()) == HINGE_LOWER_DEG
                and float(joint.GetUpperLimitAttr().Get()) == HINGE_UPPER_DEG
                and abs(float(initial_joint[0, 0]) - INITIAL_HINGE_RAD) <= math.radians(0.5)
            ),
            "colliders": collision_paths == expected_collision_paths,
            "dual_rokae_wuji": bool(
                robot_inventory["names"] == ["left", "right"]
                and all(value == 7 for value in robot_inventory["arm_joint_counts"].values())
                and all(value == 20 for value in robot_inventory["hand_joint_counts"].values())
            ),
            "initial_interpenetration": bool(
                not self_rows and not unexpected_rows and max_penetration <= 0.005
            ),
            "zero_action_stability": bool(
                base_drift <= 0.015
                and base_angle_drift <= math.radians(5.0)
                and final_hinge_error <= math.radians(1.0)
                and float(np.linalg.norm(final_base_velocity[:3])) <= 0.01
                and float(np.linalg.norm(final_display_velocity[:3])) <= 0.01
                and np.isfinite(final_base_pose).all()
                and np.isfinite(final_display_pose).all()
                and np.isfinite(final_joint).all()
            ),
            # Fail closed until a decoded production-camera frame passes below.
            "head_camera": False,
            "lighting": True,
            "zero_action_write_audit": True,
        }
        foundation_pass, failures = aggregate_checks(checks)
        physical_core_pass, physical_failures = aggregate_checks(
            {name: passed for name, passed in checks.items() if name != "head_camera"}
        )
        summary.update(
            {
                "checks": checks,
                "required_check_failures": failures,
                "foundation_pass": foundation_pass,
                "physical_core_pass": physical_core_pass,
                "physical_core_failures": physical_failures,
                "asset_hashes": asset_hashes,
                "source_props_deactivated": deactivated,
                "composition": {
                    "laptop_root": LAPTOP_ROOT,
                    "scale_xyz": list(SCALE_XYZ),
                    "position_xyz_m": [*PLACEMENT_XY_M, root_z],
                    "yaw_deg": PLACEMENT_YAW_DEG,
                    "table_top_z_m": table_top,
                    "table_support_gap_m": TABLE_SUPPORT_GAP_M,
                    "world_aabb_m": world_bbox,
                    "dome_light": {"path": str(dome.GetPath()), "intensity": 300.0},
                    "distant_light": {
                        "path": str(distant.GetPath()),
                        "intensity": 2500.0,
                    },
                    "render_mode": "RayTracedLighting",
                    "aces_tonemap_op": 4,
                    "film_iso": 200,
                },
                "asset_topology": {
                    "base_path": BASE_PATH,
                    "display_path": DISPLAY_PATH,
                    "hinge_path": HINGE_PATH,
                    "joint_axis": str(joint.GetAxisAttr().Get()),
                    "joint_limits_deg": [
                        float(joint.GetLowerLimitAttr().Get()),
                        float(joint.GetUpperLimitAttr().Get()),
                    ],
                    "self_collisions_enabled": False,
                    "collision_paths": collision_paths,
                    "collision_count": len(collision_paths),
                },
                "robot": robot_inventory,
                "zero_action": {
                    "simulated_seconds": ZERO_ACTION_SECONDS,
                    "physics_hz": PHYSICS_HZ,
                    "steps": ZERO_ACTION_STEPS,
                    "wall_seconds": zero_action_wall_seconds,
                    "initial_hinge_position_rad": float(initial_joint[0, 0]),
                    "initial_hinge_velocity_rad_s": float(initial_joint_velocity[0, 0]),
                    "final_hinge_position_rad": float(final_joint[0, 0]),
                    "final_hinge_velocity_rad_s": float(final_joint_velocity[0, 0]),
                    "peak_hinge_error_rad": peak_hinge_error,
                    "final_hinge_error_rad": final_hinge_error,
                    "initial_base_pose_xyzw": initial_base_pose.tolist(),
                    "final_base_pose_xyzw": final_base_pose.tolist(),
                    "initial_display_pose_xyzw": initial_display_pose.tolist(),
                    "final_display_pose_xyzw": final_display_pose.tolist(),
                    "base_translation_drift_m": base_drift,
                    "base_rotation_drift_rad": base_angle_drift,
                    "peak_base_linear_velocity_m_s": peak_base_linear,
                    "peak_base_angular_velocity_rad_s": peak_base_angular,
                    "peak_display_linear_velocity_m_s": peak_display_linear,
                    "peak_display_angular_velocity_rad_s": peak_display_angular,
                    "final_base_velocity": final_base_velocity.tolist(),
                    "final_display_velocity": final_display_velocity.tolist(),
                    "object_state_writes": 0,
                },
                "contacts": {
                    "row_count": len(contact_rows),
                    "self_contact_count": len(self_rows),
                    "unexpected_non_table_count": len(unexpected_rows),
                    "max_penetration_m": max_penetration,
                    "callback_errors": contact_errors,
                },
                "head_camera": {
                    "prim_path": camera_prim_path,
                    "resolution_hwc": list(HEAD_SHAPE_HWC),
                    "status": "pending_production_render",
                },
                "task_complete": False,
                "success_motion_executed": False,
                "path_or_action_tuning_executed": False,
                "object_state_writes_during_zero_action": 0,
            }
        )

        # Persist measured composition/physics before entering the renderer. This
        # is intentionally not a pass marker: the head-camera gate remains false.
        # It preserves auditable zero-action evidence if shared GPU infrastructure
        # stalls or terminates inside a native render call.
        partial_summary_path = output_dir / "core_scene_summary.json"
        require_recursive_finite(summary)
        atomic_write_json(partial_summary_path, summary)
        atomic_write_json(
            output_dir / "CORE_SCENE_PARTIAL.json",
            {
                "schema": FOUNDATION_SCHEMA,
                "foundation_pass": False,
                "physical_core_pass": physical_core_pass,
                "task_complete": False,
                "summary_path": str(partial_summary_path),
                "summary_sha256": sha256_file(partial_summary_path),
                "runner_rc": 1,
            },
        )

        sim.render_mode = production_render_mode
        frames = environment.render()
        if not isinstance(frames, (list, tuple)) or not frames:
            raise RuntimeError("production head camera returned no RGB frame")
        frame = np.asarray(frames[0])
        if frame.shape != HEAD_SHAPE_HWC or frame.dtype != np.uint8:
            raise RuntimeError(
                f"head camera must return uint8 {HEAD_SHAPE_HWC}, got {frame.shape}/{frame.dtype}"
            )
        from PIL import Image

        preview_path = output_dir / "reset_head.png"
        Image.fromarray(frame).save(preview_path, compress_level=0)
        frame_metrics = decoded_rgb_metrics(frame)
        frame_pass = head_content_pass(frame_metrics, preview_path.stat().st_size)
        checks["head_camera"] = bool(camera_prim_path == HEAD_CAMERA_PATH and frame_pass)
        foundation_pass, failures = aggregate_checks(checks)
        summary.update(
            {
                "checks": checks,
                "required_check_failures": failures,
                "foundation_pass": foundation_pass,
                "head_camera": {
                    "prim_path": camera_prim_path,
                    "resolution_hwc": list(frame.shape),
                    "status": "decoded",
                    "preview_path": str(preview_path),
                    "preview_sha256": sha256_file(preview_path),
                    "preview_bytes": preview_path.stat().st_size,
                    "metrics": frame_metrics,
                },
            }
        )
        (output_dir / "CORE_SCENE_PARTIAL.json").unlink(missing_ok=True)
        rc = 0 if foundation_pass else 1
    except Exception as exc:
        summary["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        rc = 1
    finally:
        summary["elapsed_wall_seconds"] = time.time() - started
        summary["runner_rc"] = rc
        try:
            require_recursive_finite(summary)
            summary_path = output_dir / "core_scene_summary.json"
            atomic_write_json(summary_path, summary)
            atomic_write_json(
                output_dir
                / ("CORE_SCENE_PASS.json" if rc == 0 else "CORE_SCENE_FAIL.json"),
                {
                    "schema": FOUNDATION_SCHEMA,
                    "foundation_pass": bool(summary.get("foundation_pass") is True),
                    "task_complete": False,
                    "summary_path": str(summary_path),
                    "summary_sha256": sha256_file(summary_path),
                    "runner_rc": rc,
                },
            )
        except Exception:
            traceback.print_exc()
            rc = 1
        contact_subscription = None
        if environment is not None:
            try:
                environment.close()
            except Exception:
                traceback.print_exc()
                rc = 1
        try:
            simulation_app.close()
        except SystemExit:
            # Preserve the validation result if Kit emits a clean-exit request.
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(_main())
