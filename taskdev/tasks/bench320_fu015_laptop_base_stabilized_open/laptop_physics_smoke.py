#!/usr/bin/env python3
"""Scheduler-owned CPU PhysX smoke for the exact FU015 laptop and table.

This runner deliberately excludes Hydra, cameras, and the dual-arm runtime so
the laptop's reset contacts and zero-action state can still be measured when
the cluster's private GPU namespace cannot initialize GPU Foundation. It does
not replace the complete core-scene runner and cannot make a camera, robot, or
task-completion claim.
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
    DISPLAY_PATH,
    EXPECTED_COLLISION_PATH_SUFFIXES,
    FOUNDATION_SCHEMA,
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
    RUNTIME_TASK_NAME,
    SCALE_XYZ,
    SOURCE_SHA256,
    TABLE_SUPPORT_GAP_M,
    ZERO_ACTION_SECONDS,
    ZERO_ACTION_STEPS,
)


HANDSON_PACKAGE_ROOT = REPOSITORY_ROOT / "handson"
sys.path.insert(0, str(HANDSON_PACKAGE_ROOT))

# AppLauncher must precede every remaining Isaac/Kit/USD import.
from isaaclab.app import AppLauncher  # noqa: E402


TABLE_PATH = "/World/FU015StandaloneTable"
TABLE_SIZE_M = (1.20, 0.80, 0.04)
TABLE_TOP_Z_M = 0.74


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _main() -> int:
    args = _parser().parse_args()
    args.headless = True
    args.enable_cameras = False
    args.device = "cpu"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils  # noqa: E402
    from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
    from omni.physx import get_physx_simulation_interface  # noqa: E402
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

    from .core_scene_runtime import (  # noqa: E402
        aggregate_checks,
        atomic_write_json,
        pose_drift_m,
        quaternion_angle_rad,
        require_recursive_finite,
        sha256_file,
    )

    started = time.time()
    sim = None
    subscription = None
    summary: dict[str, Any] = {
        "schema": f"{FOUNDATION_SCHEMA}_standalone_physics",
        "task": RUNTIME_TASK_NAME,
        "scope": "laptop_table_zero_action_only",
        "standalone_laptop_physics_pass": False,
        "full_core_scene_pass": False,
        "task_complete": False,
        "success_motion_executed": False,
        "path_or_action_tuning_executed": False,
        "object_state_writes_during_zero_action": 0,
    }
    rc = 1
    try:
        run_dir = output_dir.parent
        namespace_path = run_dir / "gpu_namespace.json"
        namespace = json.loads(namespace_path.read_text(encoding="utf-8"))
        physical_gpu = os.environ.get("TASKDEV_PHYSICAL_GPU", "")
        logical_gpu = os.environ.get("TASKDEV_LOGICAL_GPU", "")
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
            "sha256": sha256_file(namespace_path),
            "pass": namespace_pass,
        }

        laptop_asset = REPOSITORY_ROOT / LAPTOP_RELATIVE
        asset_hash = sha256_file(laptop_asset)
        sim_utils.create_new_stage()
        sim = SimulationContext(
            SimulationCfg(
                dt=1.0 / float(PHYSICS_HZ),
                render_interval=ZERO_ACTION_STEPS + 1,
                device="cpu",
                use_fabric=True,
                enable_scene_query_support=True,
            )
        )
        stage = sim.stage

        table_root = UsdGeom.Xform.Define(stage, TABLE_PATH)
        table_root.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(0.0, 0.0, TABLE_TOP_Z_M - TABLE_SIZE_M[2] / 2.0)
        )
        table = UsdGeom.Cube.Define(stage, f"{TABLE_PATH}/Geometry")
        table.CreateSizeAttr(1.0)
        UsdGeom.Xformable(table).AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*TABLE_SIZE_M)
        )
        # A collision prim without a rigid-body owner is a static collider.
        # Keeping translation on the parent avoids scaling the table's height.
        UsdPhysics.CollisionAPI.Apply(table.GetPrim())

        root = UsdGeom.Xform.Define(stage, LAPTOP_ROOT).GetPrim()
        root.GetReferences().AddReference(str(laptop_asset.resolve()))
        root_xform = UsdGeom.Xformable(root)
        root_xform.ClearXformOpOrder()
        root_z = TABLE_TOP_Z_M + TABLE_SUPPORT_GAP_M
        root_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(float(PLACEMENT_XY_M[0]), float(PLACEMENT_XY_M[1]), root_z)
        )
        root_xform.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set(
            PLACEMENT_YAW_DEG
        )

        base = stage.GetPrimAtPath(BASE_PATH)
        display = stage.GetPrimAtPath(DISPLAY_PATH)
        joint_prim = stage.GetPrimAtPath(HINGE_PATH)
        if not all(prim.IsValid() for prim in (base, display, joint_prim)):
            raise RuntimeError("composed laptop bodies or hinge are missing")

        articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(base)
        articulation_api.CreateEnabledSelfCollisionsAttr(False)
        PhysxSchema.PhysxRigidBodyAPI.Apply(base).CreateDisableGravityAttr(False)
        PhysxSchema.PhysxRigidBodyAPI.Apply(display).CreateDisableGravityAttr(True)
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
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)

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
                    for index in range(start, stop):
                        point = data[index]
                        contact_rows.append(
                            {
                                "physics_step": physics_step,
                                "actor0": actor0,
                                "actor1": actor1,
                                "separation_m": float(point.separation),
                                "impulse_ns": float(
                                    np.linalg.norm(
                                        np.asarray(point.impulse, dtype=np.float64)
                                    )
                                ),
                            }
                        )
            except Exception as exc:
                contact_errors.append(f"{type(exc).__name__}: {exc}")

        subscription = get_physx_simulation_interface().subscribe_contact_report_events(
            on_contacts
        )
        sim.reset()
        physics_view = sim.physics_sim_view
        if physics_view is None:
            raise RuntimeError("CPU PhysX simulation view is unavailable")
        articulation = physics_view.create_articulation_view(BASE_PATH)
        base_view = physics_view.create_rigid_body_view(BASE_PATH)
        display_view = physics_view.create_rigid_body_view(DISPLAY_PATH)
        if min(int(articulation.count), int(base_view.count), int(display_view.count)) < 1:
            raise RuntimeError("standalone laptop tensor views are empty")

        initial_joint = _array(articulation.get_dof_positions()).astype(np.float64)
        initial_base = _array(base_view.get_transforms())[0].astype(np.float64)
        initial_display = _array(display_view.get_transforms())[0].astype(np.float64)
        smoke_started = time.time()
        for step in range(ZERO_ACTION_STEPS):
            physics_step = step + 1
            sim.step(render=False)
        smoke_wall_seconds = time.time() - smoke_started

        final_joint = _array(articulation.get_dof_positions()).astype(np.float64)
        final_joint_velocity = _array(articulation.get_dof_velocities()).astype(
            np.float64
        )
        final_base = _array(base_view.get_transforms())[0].astype(np.float64)
        final_display = _array(display_view.get_transforms())[0].astype(np.float64)
        final_base_velocity = _array(base_view.get_velocities())[0].astype(np.float64)
        final_display_velocity = _array(display_view.get_velocities())[0].astype(
            np.float64
        )

        predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
        collision_paths = sorted(
            str(prim.GetPath())
            for prim in Usd.PrimRange(root, predicate)
            if prim.IsActive() and prim.HasAPI(UsdPhysics.CollisionAPI)
        )
        expected_collision_paths = sorted(
            f"{LAPTOP_ROOT}/{suffix}" for suffix in EXPECTED_COLLISION_PATH_SUFFIXES
        )
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
            and not row["actor0"].startswith(TABLE_PATH)
            and not row["actor1"].startswith(TABLE_PATH)
        ]
        max_penetration = max(
            (max(0.0, -float(row["separation_m"])) for row in contact_rows),
            default=0.0,
        )
        base_drift = pose_drift_m(initial_base, final_base)
        base_angle_drift = quaternion_angle_rad(initial_base, final_base)
        hinge_error = abs(float(final_joint[0, 0]) - INITIAL_HINGE_RAD)
        checks = {
            "scheduler_fresh_audit_and_lease": namespace_pass,
            "asset_hash": asset_hash == SOURCE_SHA256,
            "identity_scale": tuple(SCALE_XYZ) == (1.0, 1.0, 1.0),
            "hinge_topology": bool(
                joint.GetAxisAttr().Get() == "X"
                and float(joint.GetLowerLimitAttr().Get()) == HINGE_LOWER_DEG
                and float(joint.GetUpperLimitAttr().Get()) == HINGE_UPPER_DEG
            ),
            "colliders": collision_paths == expected_collision_paths,
            "initial_interpenetration": bool(
                not self_rows and not unexpected_rows and max_penetration <= 0.005
            ),
            "zero_action_stability": bool(
                base_drift <= 0.015
                and base_angle_drift <= math.radians(5.0)
                and hinge_error <= math.radians(1.0)
                and float(np.linalg.norm(final_base_velocity[:3])) <= 0.01
                and float(np.linalg.norm(final_display_velocity[:3])) <= 0.01
                and np.isfinite(final_base).all()
                and np.isfinite(final_display).all()
                and np.isfinite(final_joint).all()
            ),
            "zero_action_write_audit": True,
        }
        standalone_pass, failures = aggregate_checks(checks)
        summary.update(
            {
                "checks": checks,
                "required_check_failures": failures,
                "standalone_laptop_physics_pass": standalone_pass,
                "full_core_scene_pass": False,
                "asset_sha256": asset_hash,
                "composition": {
                    "laptop_root": LAPTOP_ROOT,
                    "scale_xyz": list(SCALE_XYZ),
                    "position_xyz_m": [*PLACEMENT_XY_M, root_z],
                    "yaw_deg": PLACEMENT_YAW_DEG,
                    "table_path": TABLE_PATH,
                    "table_size_m": list(TABLE_SIZE_M),
                    "table_top_z_m": TABLE_TOP_Z_M,
                    "collision_paths": collision_paths,
                },
                "zero_action": {
                    "simulated_seconds": ZERO_ACTION_SECONDS,
                    "physics_hz": PHYSICS_HZ,
                    "steps": ZERO_ACTION_STEPS,
                    "wall_seconds": smoke_wall_seconds,
                    "initial_hinge_position_rad": float(initial_joint[0, 0]),
                    "final_hinge_position_rad": float(final_joint[0, 0]),
                    "final_hinge_velocity_rad_s": float(final_joint_velocity[0, 0]),
                    "final_hinge_error_rad": hinge_error,
                    "initial_base_pose_xyzw": initial_base.tolist(),
                    "final_base_pose_xyzw": final_base.tolist(),
                    "initial_display_pose_xyzw": initial_display.tolist(),
                    "final_display_pose_xyzw": final_display.tolist(),
                    "base_translation_drift_m": base_drift,
                    "base_rotation_drift_rad": base_angle_drift,
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
                "limitations": {
                    "dual_arm_runtime_validated": False,
                    "head_camera_validated": False,
                    "reason": "standalone smoke intentionally excludes the blocked rendering experience",
                },
            }
        )
        rc = 0 if standalone_pass else 1
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
            summary_path = output_dir / "laptop_physics_summary.json"
            atomic_write_json(summary_path, summary)
            marker = (
                "LAPTOP_PHYSICS_PASS.json" if rc == 0 else "LAPTOP_PHYSICS_FAIL.json"
            )
            atomic_write_json(
                output_dir / marker,
                {
                    "schema": summary["schema"],
                    "standalone_laptop_physics_pass": bool(
                        summary.get("standalone_laptop_physics_pass") is True
                    ),
                    "full_core_scene_pass": False,
                    "task_complete": False,
                    "summary_path": str(summary_path),
                    "summary_sha256": sha256_file(summary_path),
                    "runner_rc": rc,
                },
            )
        except Exception:
            traceback.print_exc()
            rc = 1
        subscription = None
        if sim is not None:
            try:
                sim._disable_app_control_on_stop_handle = True
                sim.stop()
                sim.clear()
                sim.clear_all_callbacks()
                sim.clear_instance()
            except Exception:
                traceback.print_exc()
                rc = 1
        try:
            simulation_app.close()
        except SystemExit:
            # Kit may request a clean exit while closing; preserve the runner's
            # fail-closed marker/return code instead of silently converting it.
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(_main())
