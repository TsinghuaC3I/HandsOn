#!/usr/bin/env python3
"""Bounded Isaac 5.1 probe for the normalized Door042 task asset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np


TASK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_DIR.parents[2]
HANDSON_PACKAGE_ROOT = REPOSITORY_ROOT / "handson"
sys.path.insert(0, str(HANDSON_PACKAGE_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402


ASSET = TASK_DIR / "assets/zryz_door042_source/Door042_stage1.usd"
ROOT_PATH = "/World/P002DoorProbe"
FRAME_PATH = f"{ROOT_PATH}/Door042"
DOOR_PATH = f"{ROOT_PATH}/Door042_Door"
FRONT_HANDLE_PATH = f"{ROOT_PATH}/Door042_Handle002"
REAR_HANDLE_PATH = f"{ROOT_PATH}/Door042_Handle001"
SCALE = 0.60


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _range(box: Any) -> list[list[float]]:
    aligned = box.ComputeAlignedRange()
    return [
        [float(value) for value in aligned.GetMin()],
        [float(value) for value in aligned.GetMax()],
    ]


def _pose_matrix_xyzw(pose: np.ndarray) -> np.ndarray:
    x, y, z, qx, qy, qz, qw = np.asarray(pose, dtype=np.float64).reshape(7)
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )
    matrix[:3, 3] = (x, y, z)
    return matrix


def _canonical_directions(values: list[np.ndarray]) -> list[np.ndarray]:
    unique: dict[tuple[float, float, float], np.ndarray] = {}
    for value in values:
        direction = np.asarray(value, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            continue
        direction /= norm
        first = next((item for item in direction if abs(float(item)) > 1e-7), 1.0)
        if first < 0.0:
            direction = -direction
        unique[tuple(np.round(direction, 6))] = direction
    return list(unique.values())


def _convex_features(vertices: np.ndarray, convex_hull: Any) -> tuple[list[np.ndarray], list[np.ndarray]]:
    hull = convex_hull(vertices, qhull_options="QJ")
    normals = _canonical_directions([row[:3] for row in hull.equations])
    edges: list[np.ndarray] = []
    for simplex in hull.simplices:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edges.append(vertices[int(simplex[second])] - vertices[int(simplex[first])])
    return normals, _canonical_directions(edges)


def _sat_penetration(
    left: np.ndarray,
    right: np.ndarray,
    left_features: tuple[list[np.ndarray], list[np.ndarray]],
    right_features: tuple[list[np.ndarray], list[np.ndarray]],
) -> float:
    if np.any(np.max(left, axis=0) < np.min(right, axis=0)) or np.any(
        np.max(right, axis=0) < np.min(left, axis=0)
    ):
        return 0.0
    left_normals, left_edges = left_features
    right_normals, right_edges = right_features
    cross_axes = [np.cross(a, b) for a in left_edges for b in right_edges]
    axes = _canonical_directions(left_normals + right_normals + cross_axes)
    minimum_overlap = math.inf
    for axis in axes:
        left_projection = left @ axis
        right_projection = right @ axis
        overlap = min(float(np.max(left_projection)), float(np.max(right_projection))) - max(
            float(np.min(left_projection)), float(np.min(right_projection))
        )
        if overlap <= 0.0:
            return 0.0
        minimum_overlap = min(minimum_overlap, overlap)
    return 0.0 if not math.isfinite(minimum_overlap) else minimum_overlap


def main() -> int:
    args = _parser().parse_args()
    args.headless = True
    args.enable_cameras = False
    args.device = "cuda:0"
    launcher = AppLauncher(args)
    app = launcher.app
    sim = None
    result: dict[str, Any] = {
        "schema": "handson.p002_door042_asset_probe.v1",
        "asset": str(ASSET),
        "scale": SCALE,
        "passed": False,
    }
    rc = 1
    try:
        import isaaclab.sim as sim_utils  # noqa: E402
        from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402
        from scipy.spatial import ConvexHull  # noqa: E402

        if not ASSET.is_file():
            raise FileNotFoundError(ASSET)
        sim_utils.create_new_stage()
        sim = SimulationContext(
            SimulationCfg(
                dt=1.0 / 120.0,
                render_interval=121,
                device="cuda:0",
                use_fabric=True,
                enable_scene_query_support=True,
            )
        )
        stage = sim.stage
        root = UsdGeom.Xform.Define(stage, ROOT_PATH).GetPrim()
        root.GetReferences().AddReference(str(ASSET.resolve()))
        xform = UsdGeom.Xformable(root)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 1.05))
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(
            Gf.Vec3f(SCALE, SCALE, SCALE)
        )
        sim_utils.update_stage()

        required = (FRAME_PATH, DOOR_PATH, FRONT_HANDLE_PATH, REAR_HANDLE_PATH)
        if not all(stage.GetPrimAtPath(path).IsValid() for path in required):
            raise RuntimeError("Door042 required prims are missing after composition")

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        bounds = {
            path: _range(bbox_cache.ComputeWorldBound(stage.GetPrimAtPath(path)))
            for path in required
        }
        collision_paths = sorted(
            str(prim.GetPath())
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
            if prim.IsActive() and prim.HasAPI(UsdPhysics.CollisionAPI)
        )
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        collider_records: list[dict[str, Any]] = []
        for collision_path in collision_paths:
            prim = stage.GetPrimAtPath(collision_path)
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            if points.ndim != 2 or points.shape[0] < 4:
                raise RuntimeError(f"P0-02 collider is not a usable mesh: {collision_path}")
            points = points[ConvexHull(points, qhull_options="QJ").vertices]
            body_path = next(
                (path for path in required if collision_path.startswith(f"{path}/")),
                None,
            )
            if body_path is None:
                raise RuntimeError(f"P0-02 collider has no rigid parent: {collision_path}")
            collider_world = np.asarray(
                xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64
            ).T
            homogeneous = np.concatenate(
                [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
            )
            collider_records.append(
                {
                    "path": collision_path,
                    "body": body_path,
                    "static_world_vertices": (
                        collider_world @ homogeneous.T
                    ).T[:, :3],
                }
            )

        sim.reset()
        physics_view = sim.physics_sim_view
        if physics_view is None:
            raise RuntimeError("PhysX tensor view is unavailable")
        articulation = physics_view.create_articulation_view(ROOT_PATH)
        body_views = {
            path: physics_view.create_rigid_body_view(path) for path in required
        }
        if int(articulation.count) != 1 or int(articulation.max_dofs) != 3:
            raise RuntimeError(
                f"expected one three-DOF articulation, got count={articulation.count}, "
                f"max_dofs={articulation.max_dofs}"
            )
        if any(int(view.count) != 1 for view in body_views.values()):
            raise RuntimeError("one or more Door042 rigid-body tensor views are empty")

        methods = {
            name: bool(hasattr(articulation, name))
            for name in (
                "get_dof_positions",
                "get_dof_velocities",
                "get_dof_position_targets",
                "get_dof_stiffnesses",
                "get_dof_dampings",
                "get_dof_max_forces",
                "set_dof_position_targets",
                "set_dof_stiffnesses",
                "set_dof_dampings",
                "set_dof_max_forces",
                "set_dof_actuation_forces",
                "set_dof_limits",
            )
        }
        initial_positions = _array(articulation.get_dof_positions()).astype(float)
        initial_velocities = _array(articulation.get_dof_velocities()).astype(float)
        initial_stiffnesses = _array(articulation.get_dof_stiffnesses()).astype(float)
        initial_dampings = _array(articulation.get_dof_dampings()).astype(float)
        initial_max_forces = _array(articulation.get_dof_max_forces()).astype(float)
        metatype = getattr(articulation, "shared_metatype", None)
        dof_names = list(getattr(metatype, "dof_names", ()) or ())
        link_names = list(getattr(metatype, "link_names", ()) or ())
        initial_transforms = {
            path: _array(view.get_transforms())[0].astype(float).tolist()
            for path, view in body_views.items()
        }
        for collider in collider_records:
            static_vertices = collider["static_world_vertices"]
            homogeneous = np.concatenate(
                [
                    static_vertices,
                    np.ones((len(static_vertices), 1), dtype=np.float64),
                ],
                axis=1,
            )
            world_from_body = _pose_matrix_xyzw(
                np.asarray(initial_transforms[str(collider["body"])])
            )
            collider["body_vertices"] = (
                np.linalg.inv(world_from_body) @ homogeneous.T
            ).T[:, :3]
        for _ in range(120):
            sim.step(render=False)
        final_positions = _array(articulation.get_dof_positions()).astype(float)
        final_velocities = _array(articulation.get_dof_velocities()).astype(float)
        final_transforms = {
            path: _array(view.get_transforms())[0].astype(float).tolist()
            for path, view in body_views.items()
        }
        metatype = articulation.get_metatype(0)
        dof_indices = {
            str(name): int(index) for name, index in metatype.dof_indices.items()
        }
        expected_dofs = {"Door_joint", "Handle001_joint", "Handle002_joint"}
        if set(dof_indices) != expected_dofs:
            raise RuntimeError(f"unexpected Door042 DOFs: {dof_indices}")
        articulation_indices = None
        multipose: list[dict[str, Any]] = []
        for handle_deg in (0.0, 25.0, 45.0, 70.0, 90.0):
            for door_deg in (0.0, 12.0, 45.0, 75.0):
                positions = articulation.get_dof_positions().clone()
                velocities = articulation.get_dof_velocities().clone()
                positions.zero_()
                velocities.zero_()
                positions[:, dof_indices["Door_joint"]] = math.radians(door_deg)
                positions[:, dof_indices["Handle002_joint"]] = math.radians(handle_deg)
                if articulation_indices is None:
                    import torch

                    articulation_indices = torch.arange(
                        int(articulation.count), dtype=torch.long, device=positions.device
                    )
                articulation.set_dof_positions(positions, articulation_indices)
                articulation.set_dof_velocities(velocities, articulation_indices)
                targets = articulation.get_dof_position_targets().clone()
                targets.copy_(positions.to(device=targets.device))
                target_indices = articulation_indices.to(device=targets.device)
                articulation.set_dof_position_targets(targets, target_indices)
                sim.forward()
                transforms = {
                    path: _array(view.get_transforms())[0].astype(float)
                    for path, view in body_views.items()
                }
                world_vertices: list[
                    tuple[
                        dict[str, Any],
                        np.ndarray,
                        tuple[list[np.ndarray], list[np.ndarray]],
                    ]
                ] = []
                for collider in collider_records:
                    world_from_body = _pose_matrix_xyzw(
                        transforms[str(collider["body"])]
                    )
                    homogeneous = np.concatenate(
                        [
                            collider["body_vertices"],
                            np.ones((len(collider["body_vertices"]), 1)),
                        ],
                        axis=1,
                    )
                    vertices = (world_from_body @ homogeneous.T).T[:, :3]
                    world_vertices.append(
                        (collider, vertices, _convex_features(vertices, ConvexHull))
                    )
                max_penetration = 0.0
                worst_pair: list[str] = []
                worst_pair_aabbs: list[list[list[float]]] = []
                for left_index, (
                    left_record,
                    left_vertices,
                    left_features,
                ) in enumerate(world_vertices):
                    for (
                        right_record,
                        right_vertices,
                        right_features,
                    ) in world_vertices[left_index + 1 :]:
                        if left_record["body"] == right_record["body"]:
                            continue
                        penetration = _sat_penetration(
                            left_vertices,
                            right_vertices,
                            left_features,
                            right_features,
                        )
                        if penetration > max_penetration:
                            max_penetration = penetration
                            worst_pair = [left_record["path"], right_record["path"]]
                            worst_pair_aabbs = [
                                [
                                    np.min(left_vertices, axis=0).tolist(),
                                    np.max(left_vertices, axis=0).tolist(),
                                ],
                                [
                                    np.min(right_vertices, axis=0).tolist(),
                                    np.max(right_vertices, axis=0).tolist(),
                                ],
                            ]
                multipose.append(
                    {
                        "front_handle_deg": handle_deg,
                        "door_deg": door_deg,
                        "finite": bool(
                            all(np.isfinite(value).all() for value in transforms.values())
                        ),
                        "max_convex_penetration_m": max_penetration,
                        "worst_pair": worst_pair,
                        "worst_pair_aabbs_m": worst_pair_aabbs,
                    }
                )
        multipose_pass = bool(
            len(multipose) == 20
            and all(
                row["finite"] and row["max_convex_penetration_m"] <= 0.005
                for row in multipose
            )
        )
        max_joint_drift = float(np.max(np.abs(final_positions - initial_positions)))
        max_speed = float(np.max(np.abs(final_velocities)))
        all_finite = bool(
            np.isfinite(initial_positions).all()
            and np.isfinite(final_positions).all()
            and np.isfinite(final_velocities).all()
        )
        required_methods = (
            "set_dof_position_targets",
            "set_dof_stiffnesses",
            "set_dof_dampings",
            "set_dof_max_forces",
        )
        passed = bool(
            len(collision_paths) == 13
            and all_finite
            and max_joint_drift <= math.radians(2.5)
            and max_speed <= 0.5
            and all(methods[name] for name in required_methods)
            and multipose_pass
        )
        result.update(
            {
                "passed": passed,
                "collision_count": len(collision_paths),
                "collision_paths": collision_paths,
                "world_bounds_m": bounds,
                "articulation": {
                    "count": int(articulation.count),
                    "max_dofs": int(articulation.max_dofs),
                    "dof_names": dof_names,
                    "link_names": link_names,
                    "methods": methods,
                    "initial_positions_rad": initial_positions.tolist(),
                    "initial_velocities_rad_s": initial_velocities.tolist(),
                    "initial_stiffnesses": initial_stiffnesses.tolist(),
                    "initial_dampings": initial_dampings.tolist(),
                    "initial_max_forces": initial_max_forces.tolist(),
                    "final_positions_rad": final_positions.tolist(),
                    "final_velocities_rad_s": final_velocities.tolist(),
                    "max_joint_drift_rad": max_joint_drift,
                    "max_joint_speed_rad_s": max_speed,
                },
                "initial_transforms_xyzw": initial_transforms,
                "final_transforms_xyzw": final_transforms,
                "multipose_geometry": {
                    "method": "convex_hull_separating_axis_theorem",
                    "combinations": multipose,
                    "max_allowed_penetration_m": 0.005,
                    "passed": multipose_pass,
                },
            }
        )
        rc = 0 if passed else 1
    except Exception as exc:
        result["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        rc = 1
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
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
            app.close()
        except SystemExit:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
