#!/usr/bin/env python3
"""Historical immutable-source Ruler035 PhysX canary for Isaac Sim 5.1.

Run one target pose per isolated process.  This source-only diagnostic
references the delivered USD directly, exercises its five authored colliders,
and adds temporary position drives.  It never saves or rewrites the asset, but
it also does not exercise the current analytic collision overlay, passive
hinges, tabletop runtime, or success contract.  Consequently its result is
historical non-runtime evidence, not a current P0-03 acceptance gate.  Current
physics gates are ``handson/tests/test_p003_physics_overlay.py`` and
``runtime_reset_smoke.py`` in this directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPOSITORY_ROOT / "handson"
for entry in (str(REPOSITORY_ROOT), str(PACKAGE_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from handson.sim.isaaclab.p003_contract import (  # noqa: E402
    ASSET_RELATIVE_PATH,
    ASSET_SHA256,
    JOINT_1_PATH,
    JOINT_2_PATH,
    RULER_ROOT_PATH,
    S0_BODY_PATH,
    S1_BODY_PATH,
    S2_BODY_PATH,
    TARGET_SHAPES,
)
from isaaclab.app import AppLauncher  # noqa: E402


PHYSICS_HZ = 120
SIM_SECONDS = 10.0
SIM_STEPS = int(PHYSICS_HZ * SIM_SECONDS)
TABLE_PATH = "/World/P003CanaryTable"
TABLE_TOP_Z_M = 0.80
TABLE_SIZE_M = (1.20, 1.20, 0.04)
ROOT_SUPPORT_OFFSET_M = 0.012


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-key", choices=tuple(TARGET_SHAPES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _json_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _main() -> int:
    args = _parser().parse_args()
    args.headless = True
    args.enable_cameras = False
    # The delivered convex hulls explicitly fall back to CPU cooking in Isaac
    # Sim 5.1.  Exercise their authoritative compatible PhysX path instead of
    # pretending the unsupported GPU-cooking path is a valid collision test.
    args.device = "cpu"
    output_path = args.output.resolve()
    target = TARGET_SHAPES[args.pose_key]

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils  # noqa: E402
    from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
    from omni.physx import get_physx_simulation_interface  # noqa: E402
    from pxr import (  # noqa: E402
        Gf,
        PhysicsSchemaTools,
        PhysxSchema,
        Usd,
        UsdGeom,
        UsdPhysics,
    )

    summary: dict[str, object] = {
        "format": "handson.p003.ruler035_source_physx_canary_v2",
        "evidence_scope": "immutable_source_asset_only",
        "evidence_status": "historical_non_runtime_acceptance",
        "current_tabletop_physics_gate": False,
        "uses_runtime_analytic_collision_overlay": False,
        "uses_passive_hinges": False,
        "current_physics_gate_references": [
            "handson/tests/test_p003_physics_overlay.py",
            (
                "taskdev/tasks/"
                "p003_bimanual_folding_ruler_shape_configuration/"
                "runtime_reset_smoke.py"
            ),
        ],
        "pose_key": args.pose_key,
        "target_joint_angles_deg": list(target.joint_targets_deg),
        "sim_seconds": SIM_SECONDS,
        "physics_hz": PHYSICS_HZ,
        "asset_modified": False,
        "source_asset_canary_passed": False,
        "passed": False,
    }
    sim = None
    contact_subscription = None
    started = time.time()
    rc = 1
    try:
        asset_path = REPOSITORY_ROOT / ASSET_RELATIVE_PATH
        actual_hash = _sha256(asset_path)
        if actual_hash != ASSET_SHA256:
            raise RuntimeError(f"Ruler035 hash mismatch: {actual_hash}")

        sim_utils.create_new_stage()
        sim = SimulationContext(
            SimulationCfg(
                dt=1.0 / PHYSICS_HZ,
                render_interval=SIM_STEPS + 1,
                device="cpu",
                use_fabric=True,
                enable_scene_query_support=True,
            )
        )
        stage = sim.stage

        table_cfg = sim_utils.CuboidCfg(
            size=TABLE_SIZE_M,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
        )
        table_cfg.func(
            TABLE_PATH,
            table_cfg,
            translation=(0.0, 0.0, TABLE_TOP_Z_M - TABLE_SIZE_M[2] / 2.0),
        )

        root = UsdGeom.Xform.Define(stage, RULER_ROOT_PATH).GetPrim()
        root.GetReferences().AddReference(str(asset_path.resolve()))
        root_xform = UsdGeom.Xformable(root)
        root_xform.ClearXformOpOrder()
        root_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(0.0, 0.0, TABLE_TOP_Z_M + ROOT_SUPPORT_OFFSET_M)
        )
        root_xform.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set(90.0)

        body_paths = (S0_BODY_PATH, S1_BODY_PATH, S2_BODY_PATH)
        joint_paths = (JOINT_1_PATH, JOINT_2_PATH)
        bodies = [stage.GetPrimAtPath(path) for path in body_paths]
        joints = [stage.GetPrimAtPath(path) for path in joint_paths]
        if not all(prim.IsValid() for prim in [*bodies, *joints]):
            observed = [
                str(prim.GetPath())
                for prim in stage.Traverse()
                if str(prim.GetPath()).startswith(RULER_ROOT_PATH)
            ]
            raise RuntimeError(
                f"Ruler035 composed topology paths are missing; observed={observed}"
            )

        if not root.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise RuntimeError("Ruler035 source articulation root API is missing")
        PhysxSchema.PhysxArticulationAPI.Apply(root).CreateEnabledSelfCollisionsAttr(False)
        for body in bodies:
            PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateDisableGravityAttr(False)
            PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0.0)

        topology_rows: list[dict[str, object]] = []
        for joint_prim, target_deg in zip(joints, target.joint_targets_deg):
            joint = UsdPhysics.RevoluteJoint(joint_prim)
            body0 = [str(path) for path in joint.GetBody0Rel().GetTargets()]
            body1 = [str(path) for path in joint.GetBody1Rel().GetTargets()]
            topology_rows.append(
                {
                    "path": str(joint_prim.GetPath()),
                    "axis": str(joint.GetAxisAttr().Get()),
                    "lower_limit_deg": float(joint.GetLowerLimitAttr().Get()),
                    "upper_limit_deg": float(joint.GetUpperLimitAttr().Get()),
                    "body0": body0,
                    "body1": body1,
                }
            )
            drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive.CreateTargetPositionAttr(float(target_deg))
            drive.CreateTargetVelocityAttr(0.0)
            drive.CreateStiffnessAttr(2.0)
            drive.CreateDampingAttr(0.20)
            drive.CreateMaxForceAttr(2.0)
            if joint_prim.HasAPI(PhysxSchema.JointStateAPI, "angular"):
                state = PhysxSchema.JointStateAPI.Get(joint_prim, "angular")
            else:
                state = PhysxSchema.JointStateAPI.Apply(joint_prim, "angular")
            state.GetPositionAttr().Set(float(target_deg))
            state.GetVelocityAttr().Set(0.0)

        predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
        table_collision_prims = [
            prim
            for prim in Usd.PrimRange(stage.GetPrimAtPath(TABLE_PATH), predicate)
            if prim.IsActive() and prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        collision_prims = [
            prim
            for prim in Usd.PrimRange(root, predicate)
            if prim.IsActive() and prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        rigid_prims = [
            prim
            for prim in Usd.PrimRange(root, predicate)
            if prim.IsActive() and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        revolute_prims = [
            prim
            for prim in Usd.PrimRange(root, predicate)
            if prim.IsActive() and prim.IsA(UsdPhysics.RevoluteJoint)
        ]
        negative_scale_colliders: list[str] = []
        for prim in collision_prims:
            transform = UsdGeom.Xformable(prim).GetLocalTransformation()
            if float(transform.GetDeterminant()) < 0.0:
                negative_scale_colliders.append(str(prim.GetPath()))

        contact_rows: list[dict[str, object]] = []
        contact_errors: list[str] = []
        physics_step = 0

        def on_contacts(headers: Any, data: Any) -> None:
            nonlocal physics_step
            try:
                for header in headers:
                    actor0 = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
                    actor1 = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
                    if not (
                        actor0.startswith(RULER_ROOT_PATH)
                        or actor1.startswith(RULER_ROOT_PATH)
                    ):
                        continue
                    start = int(header.contact_data_offset)
                    stop = start + int(header.num_contact_data)
                    for index in range(start, stop):
                        point = data[index]
                        contact_rows.append(
                            {
                                "step": physics_step,
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

        contact_subscription = get_physx_simulation_interface().subscribe_contact_report_events(
            on_contacts
        )
        sim.reset()
        physics_view = sim.physics_sim_view
        if physics_view is None:
            raise RuntimeError("PhysX tensor simulation view is unavailable")
        articulation = physics_view.create_articulation_view(RULER_ROOT_PATH)
        body_views = [physics_view.create_rigid_body_view(path) for path in body_paths]
        if int(articulation.count) != 1 or any(int(view.count) != 1 for view in body_views):
            raise RuntimeError(
                "Ruler035 tensor views do not expose exactly one articulation and three bodies"
            )

        initial_joints = _array(articulation.get_dof_positions()).astype(np.float64)
        maximum_abs_position_m = 0.0
        maximum_abs_velocity = 0.0
        all_finite = True
        for step in range(SIM_STEPS):
            physics_step = step + 1
            sim.step(render=False)
            for view in body_views:
                transforms = _array(view.get_transforms()).astype(np.float64)
                velocities = _array(view.get_velocities()).astype(np.float64)
                all_finite &= bool(np.isfinite(transforms).all() and np.isfinite(velocities).all())
                maximum_abs_position_m = max(
                    maximum_abs_position_m,
                    float(np.max(np.abs(transforms[..., :3]))),
                )
                maximum_abs_velocity = max(
                    maximum_abs_velocity,
                    float(np.max(np.abs(velocities))),
                )

        final_joints = _array(articulation.get_dof_positions()).astype(np.float64)
        final_joint_velocities = _array(articulation.get_dof_velocities()).astype(
            np.float64
        )
        final_body_rows: list[dict[str, object]] = []
        for path, view in zip(body_paths, body_views):
            transform = _array(view.get_transforms())[0].astype(np.float64)
            velocity = _array(view.get_velocities())[0].astype(np.float64)
            all_finite &= bool(np.isfinite(transform).all() and np.isfinite(velocity).all())
            final_body_rows.append(
                {
                    "path": path,
                    "transform_xyz_xyzw": transform.tolist(),
                    "velocity_linear_angular": velocity.tolist(),
                    "linear_speed_mps": float(np.linalg.norm(velocity[:3])),
                    "angular_speed_radps": float(np.linalg.norm(velocity[3:])),
                }
            )

        target_rad = np.radians(np.asarray(target.joint_targets_deg, dtype=np.float64))
        final_joint_vector = final_joints.reshape(-1)[:2]
        joint_errors = np.abs(final_joint_vector - target_rad)
        table_contact_rows = [
            row
            for row in contact_rows
            if row["actor0"].startswith(TABLE_PATH) or row["actor1"].startswith(TABLE_PATH)
        ]
        unexpected_contact_rows = [
            row
            for row in contact_rows
            if row not in table_contact_rows
            and not (
                row["actor0"].startswith(RULER_ROOT_PATH)
                and row["actor1"].startswith(RULER_ROOT_PATH)
            )
        ]
        max_table_penetration = max(
            (max(0.0, -float(row["separation_m"])) for row in table_contact_rows),
            default=0.0,
        )
        minimum_joint_limit_margin_rad = min(
            math.radians(target.joint_targets_deg[0] - (-180.0)),
            math.radians(0.0 - target.joint_targets_deg[0]),
            math.radians(target.joint_targets_deg[1] - 0.0),
            math.radians(180.0 - target.joint_targets_deg[1]),
        )
        checks = {
            "asset_hash": actual_hash == ASSET_SHA256,
            "topology": bool(
                len(rigid_prims) == 3
                and len(revolute_prims) == 2
                and len(collision_prims) == 5
                and len(table_collision_prims) >= 1
                and topology_rows[0]["axis"] == "Z"
                and topology_rows[0]["lower_limit_deg"] == -180.0
                and topology_rows[0]["upper_limit_deg"] == 0.0
                and topology_rows[1]["axis"] == "Z"
                and topology_rows[1]["lower_limit_deg"] == 0.0
                and topology_rows[1]["upper_limit_deg"] == 180.0
            ),
            "physx_tensor_views": True,
            "joint_target": bool(np.max(joint_errors) <= math.radians(5.0)),
            "joint_limit_margin": minimum_joint_limit_margin_rad >= 0.05,
            "table_penetration": max_table_penetration <= 0.005,
            "unexpected_contacts": not unexpected_contact_rows,
            "contact_callback": not contact_errors,
            "finite_state": all_finite,
            "no_explosion": bool(
                maximum_abs_position_m < 5.0 and maximum_abs_velocity < 50.0
            ),
            "final_stability": bool(
                all(row["linear_speed_mps"] <= 0.05 for row in final_body_rows)
                and all(row["angular_speed_radps"] <= 0.50 for row in final_body_rows)
            ),
            "supported_final_height": bool(
                all(
                    abs(float(row["transform_xyz_xyzw"][2]) - TABLE_TOP_Z_M)
                    <= 0.03
                    for row in final_body_rows
                )
            ),
            "source_asset_unmodified": _sha256(asset_path) == ASSET_SHA256,
        }
        summary.update(
            {
                "asset_sha256": actual_hash,
                "scheduler_namespace": {
                    "host": os.environ.get("TASKDEV_HOST"),
                    "physical_gpu": os.environ.get("TASKDEV_PHYSICAL_GPU"),
                    "logical_gpu": os.environ.get("TASKDEV_LOGICAL_GPU"),
                    "run_id": os.environ.get("TASKDEV_RUN_ID"),
                },
                "topology": {
                    "rigid_body_paths": [str(prim.GetPath()) for prim in rigid_prims],
                    "revolute_joint_paths": [str(prim.GetPath()) for prim in revolute_prims],
                    "collision_paths": [str(prim.GetPath()) for prim in collision_prims],
                    "table_collision_paths": [
                        str(prim.GetPath()) for prim in table_collision_prims
                    ],
                    "negative_scale_collision_paths": negative_scale_colliders,
                    "joints": topology_rows,
                },
                "initial_joint_positions_rad": initial_joints.tolist(),
                "final_joint_positions_rad": final_joints.tolist(),
                "final_joint_velocities_radps": final_joint_velocities.tolist(),
                "joint_errors_rad": joint_errors.tolist(),
                "minimum_joint_limit_margin_rad": minimum_joint_limit_margin_rad,
                "final_bodies": final_body_rows,
                "maximum_abs_position_m": maximum_abs_position_m,
                "maximum_abs_velocity": maximum_abs_velocity,
                "contacts": {
                    "row_count": len(contact_rows),
                    "table_row_count": len(table_contact_rows),
                    "unexpected_row_count": len(unexpected_contact_rows),
                    "max_table_penetration_m": max_table_penetration,
                    "callback_errors": contact_errors,
                },
                "checks": checks,
                "source_asset_canary_passed": all(checks.values()),
                # Compatibility for archived tooling only.  The explicit
                # evidence fields above prevent this source-only boolean from
                # being mistaken for current tabletop-runtime acceptance.
                "passed": all(checks.values()),
            }
        )
        rc = 0 if summary["passed"] else 1
    except BaseException as exc:
        summary["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        rc = 1
    finally:
        summary["wall_seconds"] = time.time() - started
        summary["runner_rc"] = rc
        try:
            _json_write(output_path, summary)
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        except Exception:
            traceback.print_exc()
            rc = 1
        contact_subscription = None
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
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(_main())
