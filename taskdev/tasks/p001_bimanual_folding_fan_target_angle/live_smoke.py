"""Bounded live Isaac smoke/trajectory runner for P0-01 evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback
from typing import Any, Mapping

import numpy as np

from handson.sim.isaaclab.bimanual_assembly_env import _matrix_to_quat_wxyz
from handson.sim.isaaclab.p001_contract import (
    CENTER_BODY_PATH,
    LEFT_END_BODY_PATH,
    P001ScriptedPolicy,
    RIGHT_END_BODY_PATH,
    WUJI_FAN_GRIP,
    calibrated_feedback_probe_poses,
)
from handson.tasks.t48_unfold_fan.config import TaskConfig


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _contact_force_breakdown(raw_env: Any, key: str) -> dict[str, float]:
    view = raw_env._p001_contact_views[key]
    _sensor, filters = raw_env._p001_contact_paths[key]
    forces, _points, _normals, _separations, counts, starts = (
        view.get_contact_data(raw_env._sim_dt)
    )
    force_values = np.abs(_numpy(forces).reshape(-1))
    count_values = _numpy(counts).reshape(-1).astype(np.int64)
    start_values = _numpy(starts).reshape(-1).astype(np.int64)
    result: dict[str, float] = {}
    for index, (count, start) in enumerate(zip(count_values, start_values)):
        if count <= 0:
            continue
        path = str(filters[index % len(filters)])
        result[path] = result.get(path, 0.0) + float(
            np.sum(force_values[start : start + count])
        )
    return dict(sorted(result.items(), key=lambda item: (-item[1], item[0])))


def _contact_point_breakdown(raw_env: Any, key: str) -> dict[str, list[float]]:
    view = raw_env._p001_contact_views[key]
    _sensor, filters = raw_env._p001_contact_paths[key]
    forces, points, _normals, _separations, counts, starts = (
        view.get_contact_data(raw_env._sim_dt)
    )
    force_values = np.abs(_numpy(forces).reshape(-1))
    point_values = _numpy(points).reshape(-1, 3)
    count_values = _numpy(counts).reshape(-1).astype(np.int64)
    start_values = _numpy(starts).reshape(-1).astype(np.int64)
    result: dict[str, list[float]] = {}
    for index, (count, start) in enumerate(zip(count_values, start_values)):
        if count <= 0:
            continue
        path = str(filters[index % len(filters)])
        segment = point_values[start : start + count]
        weights = force_values[start : start + count]
        if float(np.sum(weights)) > 0.0:
            point = np.average(segment, axis=0, weights=weights)
        else:
            point = np.mean(segment, axis=0)
        result[path] = np.asarray(point, dtype=np.float64).tolist()
    return result


def _net_contact_force(raw_env: Any, key: str) -> float:
    forces = _numpy(
        raw_env._p001_contact_views[key].get_net_contact_forces(raw_env._sim_dt)
    ).reshape(-1, 3)
    return float(np.sum(np.linalg.norm(forces, axis=1)))


def _robot_body_poses(raw_env: Any, side: str) -> dict[str, np.ndarray]:
    robot = raw_env._control_robot(side)
    poses = _numpy(robot.data.body_pose_w)[0]
    return {
        str(name): np.asarray(poses[index, :7], dtype=np.float64)
        for index, name in enumerate(robot.body_names)
    }


def _actual_tcp_pose_wxyz(raw_env: Any, side: str) -> np.ndarray:
    matrix = raw_env._robot_body_pose_matrix(
        raw_env._control_robot(side), raw_env._ee_body_indices[side]
    )
    return np.concatenate(
        [matrix[:3, 3], _matrix_to_quat_wxyz(matrix[:3, :3])]
    ).astype(np.float64)


def _prim_world_bbox(raw_env: Any, path: str) -> dict[str, list[float]]:
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render],
        False,
        True,
    )
    box = cache.ComputeWorldBound(raw_env._stage.GetPrimAtPath(path)).ComputeAlignedBox()
    return {
        "min": [float(value) for value in box.GetMin()],
        "max": [float(value) for value in box.GetMax()],
    }


def _collision_shape_points(prim: Any) -> list[Any]:
    from pxr import Gf, UsdGeom

    mesh = UsdGeom.Mesh(prim)
    mesh_points = mesh.GetPointsAttr().Get() if mesh else None
    if mesh_points:
        return list(mesh_points)
    sphere = UsdGeom.Sphere(prim)
    if sphere:
        radius = float(sphere.GetRadiusAttr().Get())
        return [
            Gf.Vec3d(x, y, z)
            for x in (-radius, radius)
            for y in (-radius, radius)
            for z in (-radius, radius)
        ]
    capsule = UsdGeom.Capsule(prim)
    if capsule:
        radius = float(capsule.GetRadiusAttr().Get())
        half_axis = float(capsule.GetHeightAttr().Get()) / 2.0 + radius
        extents = [radius, radius, radius]
        axis = str(capsule.GetAxisAttr().Get() or "Z").upper()
        extents[{"X": 0, "Y": 1, "Z": 2}.get(axis, 2)] = half_axis
        return [
            Gf.Vec3d(x, y, z)
            for x in (-extents[0], extents[0])
            for y in (-extents[1], extents[1])
            for z in (-extents[2], extents[2])
        ]
    return []


def _collision_world_bboxes(raw_env: Any, body_path: str) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render],
        False,
        True,
    )
    body = raw_env._stage.GetPrimAtPath(body_path)
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    bboxes = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render],
        False,
        True,
    )
    result: dict[str, Any] = {}

    for prim in Usd.PrimRange(body):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        points = _collision_shape_points(prim)
        if points:
            transform = xforms.GetLocalToWorldTransform(prim)
            vertices = np.asarray(
                [[float(value) for value in transform.Transform(point)] for point in points],
                dtype=np.float64,
            )
            minimum = np.min(vertices, axis=0).tolist()
            maximum = np.max(vertices, axis=0).tolist()
        else:
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            minimum = [float(value) for value in box.GetMin()]
            maximum = [float(value) for value in box.GetMax()]
        result[str(prim.GetPath())] = {
            "type": str(prim.GetTypeName()),
            "point_count": len(points),
            "min": minimum,
            "max": maximum,
        }
    return result


def _runtime_collision_world_bboxes(
    raw_env: Any,
    body_path: str,
    runtime_matrix: np.ndarray,
) -> dict[str, Any]:
    """Transform authored collider vertices through an exact runtime body pose."""

    from pxr import Usd, UsdGeom, UsdPhysics

    body = raw_env._stage.GetPrimAtPath(body_path)
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    bboxes = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render],
        False,
        True,
    )
    authored_body_inverse = xforms.GetLocalToWorldTransform(body).GetInverse()
    runtime_matrix = np.asarray(runtime_matrix, dtype=np.float64).reshape(4, 4)
    result: dict[str, Any] = {}

    def add_vertices(key: str, authored_world_vertices: list[Any]) -> None:
        local_vertices = np.asarray(
            [
                [float(value) for value in authored_body_inverse.Transform(point)]
                for point in authored_world_vertices
            ],
            dtype=np.float64,
        )
        world_vertices = (
            runtime_matrix[:3, :3] @ local_vertices.T
        ).T + runtime_matrix[:3, 3]
        result[key] = {
            "point_count": len(authored_world_vertices),
            "min": np.min(world_vertices, axis=0).tolist(),
            "max": np.max(world_vertices, axis=0).tolist(),
            "vertices": world_vertices.tolist() if len(world_vertices) <= 32 else [],
        }

    for prim in Usd.PrimRange(body):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() if mesh else None
        if points:
            authored_collider = xforms.GetLocalToWorldTransform(prim)
            authored_world_vertices = [
                authored_collider.Transform(point) for point in points
            ]
        else:
            box = bboxes.ComputeWorldBound(prim).ComputeAlignedBox()
            minimum = box.GetMin()
            maximum = box.GetMax()
            authored_world_vertices = [
                type(minimum)(x, y, z)
                for x in (minimum[0], maximum[0])
                for y in (minimum[1], maximum[1])
                for z in (minimum[2], maximum[2])
            ]
        add_vertices(str(prim.GetPath()), authored_world_vertices)
        if mesh:
            result[str(prim.GetPath())].update(
                {
                    "face_count": len(mesh.GetFaceVertexCountsAttr().Get() or []),
                    "face_index_count": len(
                        mesh.GetFaceVertexIndicesAttr().Get() or []
                    ),
                }
            )

    # Robot collision branches are USD instances.  PrimRange intentionally
    # stops at the instance root, so map prototype mesh points through the
    # authored instance transform before expressing them in the rigid body.
    for instance in Usd.PrimRange(body):
        if not instance.IsInstance():
            continue
        prototype = instance.GetPrototype()
        if not prototype.IsValid():
            continue
        instance_world = xforms.GetLocalToWorldTransform(instance)
        prototype_inverse = xforms.GetLocalToWorldTransform(prototype).GetInverse()
        for prim in Usd.PrimRange(prototype):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            points = _collision_shape_points(prim)
            if not points:
                continue
            prototype_mesh_world = xforms.GetLocalToWorldTransform(prim)
            authored_world_vertices = [
                instance_world.Transform(
                    prototype_inverse.Transform(
                        prototype_mesh_world.Transform(point)
                    )
                )
                for point in points
            ]
            key = f"{instance.GetPath()}::{prim.GetPath()}"
            add_vertices(key, authored_world_vertices)
            capsule = UsdGeom.Capsule(prim)
            sphere = UsdGeom.Sphere(prim)
            if capsule:
                result[key].update(
                    {
                        "shape": "Capsule",
                        "axis": str(capsule.GetAxisAttr().Get()),
                        "height": float(capsule.GetHeightAttr().Get()),
                        "radius": float(capsule.GetRadiusAttr().Get()),
                    }
                )
            elif sphere:
                result[key].update(
                    {"shape": "Sphere", "radius": float(sphere.GetRadiusAttr().Get())}
                )
    return result


def _xyzw_pose_matrix(pose: Any) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64).reshape(7)
    x, y, z, w = values[3:7]
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = values[:3]
    return matrix


def _wxyz_pose_matrix(pose: Any) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64).reshape(7)
    return _xyzw_pose_matrix(
        np.concatenate([values[:3], values[4:7], values[3:4]])
    )


def _matrix_to_pose_wxyz(matrix: Any) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    return np.concatenate(
        [values[:3, 3], _matrix_to_quat_wxyz(values[:3, :3])]
    )


def _rotate_matrix_about_world_z(
    matrix: Any, pivot_xyz: Any, angle_deg: float
) -> np.ndarray:
    angle = np.deg2rad(float(angle_deg))
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    pivot = np.asarray(pivot_xyz, dtype=np.float64).reshape(3)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = pivot - rotation @ pivot
    return transform @ np.asarray(matrix, dtype=np.float64).reshape(4, 4)


def _robot_runtime_collision_bboxes(
    raw_env: Any, side: str, body_name: str
) -> dict[str, Any]:
    robot = raw_env._control_robot(side)
    body_index = list(robot.body_names).index(body_name)
    matrix = raw_env._robot_body_pose_matrix(robot, body_index)
    body_path = (
        f"/World/HandsOnRobots/{side.capitalize()}Robot/"
        f"base_{side}/{body_name}"
    )
    return _runtime_collision_world_bboxes(raw_env, body_path, matrix)


def _prim_structure(raw_env: Any, path: str) -> dict[str, Any]:
    from pxr import Usd, UsdPhysics

    prim = raw_env._stage.GetPrimAtPath(path)
    descendants = []
    if prim.IsValid():
        for item in Usd.PrimRange(prim):
            descendants.append(
                {
                    "path": str(item.GetPath()),
                    "type": str(item.GetTypeName()),
                    "collision": bool(item.HasAPI(UsdPhysics.CollisionAPI)),
                    "instance": bool(item.IsInstance()),
                    "instance_proxy": bool(item.IsInstanceProxy()),
                }
            )
        if prim.IsInstance() and prim.GetPrototype().IsValid():
            for item in Usd.PrimRange(prim.GetPrototype()):
                descendants.append(
                    {
                        "path": str(item.GetPath()),
                        "type": str(item.GetTypeName()),
                        "collision": bool(item.HasAPI(UsdPhysics.CollisionAPI)),
                        "instance": bool(item.IsInstance()),
                        "instance_proxy": bool(item.IsInstanceProxy()),
                    }
                )
    return {
        "valid": bool(prim.IsValid()),
        "type": str(prim.GetTypeName()) if prim.IsValid() else "",
        "instance": bool(prim.IsInstance()) if prim.IsValid() else False,
        "instance_proxy": bool(prim.IsInstanceProxy()) if prim.IsValid() else False,
        "prototype": (
            str(prim.GetPrototype().GetPath())
            if prim.IsValid() and prim.IsInstance() and prim.GetPrototype().IsValid()
            else ""
        ),
        "descendants": descendants[:80],
    }


def _physics_prim_properties(raw_env: Any, path: str) -> dict[str, Any]:
    prim = raw_env._stage.GetPrimAtPath(path)
    values: dict[str, Any] = {
        "valid": bool(prim.IsValid()),
        "applied_schemas": list(prim.GetAppliedSchemas()) if prim.IsValid() else [],
    }
    if not prim.IsValid():
        return values
    attributes: dict[str, Any] = {}
    for attribute in prim.GetAttributes():
        name = str(attribute.GetName())
        if any(
            token in name.lower()
            for token in (
                "collision",
                "contact",
                "restoffset",
                "approximation",
                "rigidbody",
                "kinematic",
                "gravity",
                "purpose",
            )
        ):
            value = attribute.Get()
            attributes[name] = _jsonable(value)
    values["attributes"] = attributes
    values["relationships"] = {
        str(relationship.GetName()): [str(target) for target in relationship.GetTargets()]
        for relationship in prim.GetRelationships()
        if any(
            token in str(relationship.GetName()).lower()
            for token in ("collision", "filter", "owner", "body")
        )
    }
    return values


def _collision_groups(raw_env: Any) -> dict[str, Any]:
    from pxr import UsdPhysics

    result: dict[str, Any] = {}
    for prim in raw_env._stage.Traverse():
        if not prim.IsA(UsdPhysics.CollisionGroup):
            continue
        group = UsdPhysics.CollisionGroup(prim)
        result[str(prim.GetPath())] = {
            "filtered_groups": [
                str(target) for target in group.GetFilteredGroupsRel().GetTargets()
            ],
            "includes": [
                str(target)
                for target in group.GetCollidersCollectionAPI()
                .GetIncludesRel()
                .GetTargets()
            ],
            "excludes": [
                str(target)
                for target in group.GetCollidersCollectionAPI()
                .GetExcludesRel()
                .GetTargets()
            ],
        }
    return result


def _filtered_pairs(raw_env: Any) -> dict[str, list[str]]:
    from pxr import UsdPhysics

    result: dict[str, list[str]] = {}
    for prim in raw_env._stage.Traverse():
        if not prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            continue
        targets = [
            str(target)
            for target in UsdPhysics.FilteredPairsAPI(prim)
            .GetFilteredPairsRel()
            .GetTargets()
        ]
        if targets:
            result[str(prim.GetPath())] = targets
    return result


def _collision_state_under(raw_env: Any, path: str) -> dict[str, Any]:
    from pxr import Usd, UsdPhysics

    root = raw_env._stage.GetPrimAtPath(path)
    result: dict[str, Any] = {}
    if not root.IsValid():
        return result
    prims = list(Usd.PrimRange(root))
    if root.IsInstance() and root.GetPrototype().IsValid():
        prims.extend(Usd.PrimRange(root.GetPrototype()))
    for prim in prims:
        collision = UsdPhysics.CollisionAPI(prim)
        if not collision:
            continue
        enabled = collision.GetCollisionEnabledAttr()
        result[str(prim.GetPath())] = {
            "enabled": None if not enabled or not enabled.IsValid() else bool(enabled.Get()),
            "schemas": list(prim.GetAppliedSchemas()),
        }
    return result


def _array_shapes(value: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    if isinstance(value, Mapping):
        result: dict[str, dict[str, Any]] = {}
        for key, item in value.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            result.update(_array_shapes(item, child))
        return result
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value)
    except Exception:
        return {}
    description: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.size and np.issubdtype(array.dtype, np.number):
        finite = array[np.isfinite(array)]
        if finite.size:
            description["min"] = float(np.min(finite))
            description["max"] = float(np.max(finite))
    return {prefix or "value": description}


def _save_rgb_arrays(value: Any, output_dir: Path, prefix: str) -> list[str]:
    from PIL import Image

    saved: list[str] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{path}_{key}" if path else str(key))
            return
        if hasattr(item, "detach"):
            item = item.detach().cpu().numpy()
        try:
            array = np.asarray(item)
        except Exception:
            return
        while array.ndim > 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            return
        rgb = array[..., :3]
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
            if float(np.max(rgb)) <= 1.5:
                rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{prefix}_{path.replace('/', '_')}.png"
        destination = output_dir / filename
        Image.fromarray(rgb).save(destination)
        saved.append(str(destination))

    visit(value, "observation")
    return saved


def main() -> None:
    probe_pose_env = os.environ.get("P001_RIGHT_PROBE_POSE", "").strip()
    probe_pose_default = (
        [float(value) for value in probe_pose_env.split(",")]
        if probe_pose_env
        else None
    )
    if probe_pose_default is not None and len(probe_pose_default) != 7:
        raise ValueError("P001_RIGHT_PROBE_POSE must contain seven comma-separated values")
    probe_end_pose_env = os.environ.get("P001_RIGHT_PROBE_END_POSE", "").strip()
    probe_end_pose_default = (
        [float(value) for value in probe_end_pose_env.split(",")]
        if probe_end_pose_env
        else None
    )
    if probe_end_pose_default is not None and len(probe_end_pose_default) != 7:
        raise ValueError(
            "P001_RIGHT_PROBE_END_POSE must contain seven comma-separated values"
        )
    left_probe_pose_env = os.environ.get("P001_LEFT_PROBE_POSE", "").strip()
    left_probe_pose_default = (
        [float(value) for value in left_probe_pose_env.split(",")]
        if left_probe_pose_env
        else None
    )
    if left_probe_pose_default is not None and len(left_probe_pose_default) != 7:
        raise ValueError("P001_LEFT_PROBE_POSE must contain seven comma-separated values")
    left_probe_end_pose_env = os.environ.get(
        "P001_LEFT_PROBE_END_POSE", ""
    ).strip()
    left_probe_end_pose_default = (
        [float(value) for value in left_probe_end_pose_env.split(",")]
        if left_probe_end_pose_env
        else None
    )
    if left_probe_end_pose_default is not None and len(left_probe_end_pose_default) != 7:
        raise ValueError(
            "P001_LEFT_PROBE_END_POSE must contain seven comma-separated values"
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17001)
    parser.add_argument("--target", type=float, choices=(60.0, 90.0, 120.0), default=90.0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument(
        "--mode",
        choices=(
            "zero",
            "scripted",
            "hand_only",
            "right_pose",
            "left_pose",
            "right_probe",
            "right_staged_probe",
            "left_probe",
            "left_staged_probe",
            "bimanual_staged_probe",
            "bimanual_side_probe",
            "bimanual_delayed_right_probe",
            "bimanual_feedback_open",
        ),
        default="zero",
    )
    parser.add_argument(
        "--right-probe-pose",
        type=float,
        nargs=7,
        default=probe_pose_default,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
    )
    parser.add_argument(
        "--right-probe-grip",
        type=float,
        default=(
            float(os.environ["P001_RIGHT_PROBE_GRIP"])
            if "P001_RIGHT_PROBE_GRIP" in os.environ
            else None
        ),
    )
    parser.add_argument(
        "--right-probe-end-pose",
        type=float,
        nargs=7,
        default=probe_end_pose_default,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
    )
    parser.add_argument(
        "--left-probe-pose",
        type=float,
        nargs=7,
        default=left_probe_pose_default,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
    )
    parser.add_argument(
        "--left-probe-grip",
        type=float,
        default=(
            float(os.environ["P001_LEFT_PROBE_GRIP"])
            if "P001_LEFT_PROBE_GRIP" in os.environ
            else None
        ),
    )
    parser.add_argument(
        "--left-probe-end-pose",
        type=float,
        nargs=7,
        default=left_probe_end_pose_default,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-dir", type=Path)
    args = parser.parse_args()
    if args.right_probe_grip is None:
        args.right_probe_grip = 0.0
    if args.left_probe_grip is None:
        args.left_probe_grip = (
            0.05 if args.mode == "bimanual_feedback_open" else 0.0
        )

    env = TaskConfig().get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=True,
        sim_backend="isaacsim",
        seed=args.seed,
        target_open_deg=args.target,
        image_obs=bool(args.render),
        show_sim_cameras=False,
    )
    result: dict[str, Any] = {
        "schema": "p001_live_smoke_v1",
        "seed": args.seed,
        "target_deg": args.target,
        "mode": args.mode,
        "requested_steps": args.steps,
    }
    try:
        observation, reset_info = env.reset(seed=args.seed, target_open_deg=args.target)
        initial_shapes = _array_shapes(observation)
        saved_images = (
            _save_rgb_arrays(observation, args.image_dir, "reset")
            if args.image_dir is not None
            else []
        )
        raw_env = env.unwrapped
        if args.mode == "bimanual_feedback_open":
            calibrated_right, calibrated_left_start, calibrated_left_end = (
                calibrated_feedback_probe_poses(raw_env._p001_reset)
            )
            if args.right_probe_pose is None:
                args.right_probe_pose = calibrated_right.tolist()
            if args.left_probe_pose is None:
                args.left_probe_pose = calibrated_left_start.tolist()
            if args.left_probe_end_pose is None:
                args.left_probe_end_pose = calibrated_left_end.tolist()
            result["feedback_controller"] = {
                "right_probe_pose_wxyz": args.right_probe_pose,
                "left_probe_pose_wxyz": args.left_probe_pose,
                "left_probe_end_pose_wxyz": args.left_probe_end_pose,
                "right_grip_scale": args.right_probe_grip,
                "left_grip_scale": args.left_probe_grip,
                "root_relative_calibration": True,
                "object_pose_writes_after_reset": 0,
            }
        policy = P001ScriptedPolicy(
            raw_env._p001_reset,
            right_home=_actual_tcp_pose_wxyz(raw_env, "right"),
            left_home=_actual_tcp_pose_wxyz(raw_env, "left"),
        )
        last_info: dict[str, Any] = dict(reset_info)
        trajectory_samples: list[dict[str, Any]] = []
        feedback_state: dict[str, np.ndarray] = {}
        executed = 0
        terminal = False
        for _ in range(args.steps):
            if args.mode == "zero":
                action = np.zeros(env.action_space.shape, dtype=np.float64)
            elif args.mode in ("right_probe", "right_staged_probe"):
                if args.right_probe_pose is None:
                    raise ValueError("right_probe requires --right-probe-pose")
                action = np.zeros(env.action_space.shape, dtype=np.float64)
                right_probe_pose = np.asarray(
                    args.right_probe_pose, dtype=np.float64
                ).copy()
                if args.mode == "right_staged_probe":
                    high_pose = right_probe_pose.copy()
                    high_pose[2] = max(1.020, float(right_probe_pose[2]) + 0.060)
                    probe_step = int(raw_env.env_step)
                    if probe_step < 30:
                        commanded_pose = high_pose
                    elif probe_step < 75:
                        alpha = (probe_step - 30) / 45.0
                        weight = alpha * alpha * (3.0 - 2.0 * alpha)
                        commanded_pose = high_pose.copy()
                        commanded_pose[:3] += weight * (
                            right_probe_pose[:3] - high_pose[:3]
                        )
                    elif probe_step < 135 and args.right_probe_end_pose is not None:
                        alpha = (probe_step - 75) / 60.0
                        weight = alpha * alpha * (3.0 - 2.0 * alpha)
                        end_pose = np.asarray(
                            args.right_probe_end_pose, dtype=np.float64
                        )
                        commanded_pose = right_probe_pose.copy()
                        commanded_pose[:3] += weight * (
                            end_pose[:3] - right_probe_pose[:3]
                        )
                        start_quaternion = right_probe_pose[3:7].copy()
                        end_quaternion = end_pose[3:7].copy()
                        if float(np.dot(start_quaternion, end_quaternion)) < 0.0:
                            end_quaternion *= -1.0
                        commanded_pose[3:7] = (
                            (1.0 - weight) * start_quaternion
                            + weight * end_quaternion
                        )
                        commanded_pose[3:7] /= np.linalg.norm(
                            commanded_pose[3:7]
                        )
                    elif args.right_probe_end_pose is not None:
                        commanded_pose = np.asarray(
                            args.right_probe_end_pose, dtype=np.float64
                        )
                    else:
                        commanded_pose = right_probe_pose
                else:
                    commanded_pose = right_probe_pose
                action[:7] = commanded_pose
                action[14:34] = float(args.right_probe_grip) * WUJI_FAN_GRIP
            elif args.mode == "bimanual_staged_probe":
                if (
                    args.right_probe_pose is None
                    or args.right_probe_end_pose is None
                    or args.left_probe_pose is None
                ):
                    raise ValueError(
                        "bimanual_staged_probe requires right start/end and left pose"
                    )
                action = np.zeros(env.action_space.shape, dtype=np.float64)
                probe_step = int(raw_env.env_step)
                right_start = np.asarray(
                    args.right_probe_pose, dtype=np.float64
                ).copy()
                right_end = np.asarray(
                    args.right_probe_end_pose, dtype=np.float64
                ).copy()
                right_high = right_start.copy()
                right_high[2] = max(1.020, float(right_start[2]) + 0.060)
                left_low = np.asarray(args.left_probe_pose, dtype=np.float64).copy()
                left_high = left_low.copy()
                left_high[2] = max(1.020, float(left_low[2]) + 0.080)
                if probe_step < 30:
                    right_command = right_high
                elif probe_step < 75:
                    alpha = (probe_step - 30) / 45.0
                    weight = alpha * alpha * (3.0 - 2.0 * alpha)
                    right_command = right_high.copy()
                    right_command[:3] += weight * (
                        right_start[:3] - right_high[:3]
                    )
                elif probe_step < 135:
                    alpha = (probe_step - 75) / 60.0
                    weight = alpha * alpha * (3.0 - 2.0 * alpha)
                    right_command = right_start.copy()
                    right_command[:3] += weight * (
                        right_end[:3] - right_start[:3]
                    )
                else:
                    right_command = right_end
                if probe_step < 135:
                    left_command = left_high
                elif probe_step < 195:
                    alpha = (probe_step - 135) / 60.0
                    weight = alpha * alpha * (3.0 - 2.0 * alpha)
                    left_command = left_high.copy()
                    left_command[:3] += weight * (
                        left_low[:3] - left_high[:3]
                    )
                else:
                    left_command = left_low
                action[:7] = right_command
                action[7:14] = left_command
                action[14:34] = float(args.right_probe_grip) * WUJI_FAN_GRIP
                action[34:54] = float(args.left_probe_grip) * WUJI_FAN_GRIP
            elif args.mode == "bimanual_side_probe":
                if (
                    args.right_probe_pose is None
                    or args.left_probe_pose is None
                    or args.left_probe_end_pose is None
                ):
                    raise ValueError(
                        "bimanual_side_probe requires right pose and left start/end"
                    )
                action = np.zeros(env.action_space.shape, dtype=np.float64)
                probe_step = int(raw_env.env_step)
                right_low = np.asarray(
                    args.right_probe_pose, dtype=np.float64
                ).copy()
                right_high = right_low.copy()
                right_high[2] = max(1.020, float(right_low[2]) + 0.060)
                left_start = np.asarray(
                    args.left_probe_pose, dtype=np.float64
                ).copy()
                left_end = np.asarray(
                    args.left_probe_end_pose, dtype=np.float64
                ).copy()
                left_high = left_start.copy()
                left_high[2] = max(1.020, float(left_start[2]) + 0.080)
                if probe_step < 30:
                    right_command = right_high
                    left_command = left_high
                elif probe_step < 75:
                    alpha = (probe_step - 30) / 45.0
                    weight = alpha * alpha * (3.0 - 2.0 * alpha)
                    right_command = right_high.copy()
                    right_command[:3] += weight * (
                        right_low[:3] - right_high[:3]
                    )
                    left_command = left_high.copy()
                    left_command[:3] += weight * (
                        left_start[:3] - left_high[:3]
                    )
                elif probe_step < 255:
                    alpha = (probe_step - 75) / 180.0
                    weight = alpha * alpha * (3.0 - 2.0 * alpha)
                    right_command = right_low.copy()
                    if args.right_probe_end_pose is not None:
                        right_command[:3] += weight * (
                            np.asarray(
                                args.right_probe_end_pose, dtype=np.float64
                            )[:3]
                            - right_low[:3]
                        )
                    left_command = left_start.copy()
                    left_command[:3] += weight * (
                        left_end[:3] - left_start[:3]
                    )
                else:
                    right_command = (
                        np.asarray(args.right_probe_end_pose, dtype=np.float64)
                        if args.right_probe_end_pose is not None
                        else right_low
                    )
                    left_command = left_end
                action[:7] = right_command
                action[7:14] = left_command
                action[14:34] = float(args.right_probe_grip) * WUJI_FAN_GRIP
                action[34:54] = float(args.left_probe_grip) * WUJI_FAN_GRIP
            elif args.mode == "bimanual_delayed_right_probe":
                if (
                    args.right_probe_pose is None
                    or args.left_probe_pose is None
                    or args.left_probe_end_pose is None
                ):
                    raise ValueError(
                        "bimanual_delayed_right_probe requires right pose and "
                        "left start/end"
                    )
                action = np.zeros(env.action_space.shape, dtype=np.float64)
                probe_step = int(raw_env.env_step)
                right_low = np.asarray(
                    args.right_probe_pose, dtype=np.float64
                ).copy()
                right_high = right_low.copy()
                right_high[2] = max(1.020, float(right_low[2]) + 0.060)
                left_start = np.asarray(
                    args.left_probe_pose, dtype=np.float64
                ).copy()
                left_end = np.asarray(
                    args.left_probe_end_pose, dtype=np.float64
                ).copy()
                left_high = left_start.copy()
                left_high[2] = max(1.020, float(left_start[2]) + 0.080)
                if probe_step < 30:
                    left_command = left_high
                elif probe_step < 75:
                    alpha = (probe_step - 30) / 45.0
                    weight = alpha * alpha * (3.0 - 2.0 * alpha)
                    left_command = left_high.copy()
                    left_command[:3] += weight * (
                        left_start[:3] - left_high[:3]
                    )
                elif probe_step < 135:
                    alpha = (probe_step - 75) / 60.0
                    weight = alpha * alpha * (3.0 - 2.0 * alpha)
                    left_command = left_start.copy()
                    left_command[:3] += weight * (
                        left_end[:3] - left_start[:3]
                    )
                else:
                    left_command = left_end
                if probe_step < 105:
                    right_command = right_high
                elif probe_step < 135:
                    alpha = (probe_step - 105) / 30.0
                    weight = alpha * alpha * (3.0 - 2.0 * alpha)
                    right_command = right_high.copy()
                    right_command[:3] += weight * (
                        right_low[:3] - right_high[:3]
                    )
                elif (
                    probe_step < 315 and args.right_probe_end_pose is not None
                ):
                    alpha = (probe_step - 135) / 180.0
                    weight = alpha * alpha * (3.0 - 2.0 * alpha)
                    right_command = right_low.copy()
                    right_command[:3] += weight * (
                        np.asarray(args.right_probe_end_pose, dtype=np.float64)[:3]
                        - right_low[:3]
                    )
                else:
                    right_command = (
                        np.asarray(args.right_probe_end_pose, dtype=np.float64)
                        if args.right_probe_end_pose is not None
                        else right_low
                    )
                action[:7] = right_command
                action[7:14] = left_command
                action[14:34] = float(args.right_probe_grip) * WUJI_FAN_GRIP
                action[34:54] = float(args.left_probe_grip) * WUJI_FAN_GRIP
            elif args.mode == "bimanual_feedback_open":
                if (
                    args.right_probe_pose is None
                    or args.left_probe_pose is None
                    or args.left_probe_end_pose is None
                ):
                    raise ValueError(
                        "bimanual_feedback_open requires right pose and left "
                        "start/end"
                    )
                action = np.zeros(env.action_space.shape, dtype=np.float64)
                feedback_step = int(raw_env.env_step)
                target_uses_four_regrasps = bool(
                    float(args.target) == 120.0
                    and float(args.left_probe_pose[2])
                    >= float(
                        os.environ.get(
                            "P001_FOUR_REGRASP_MIN_LEFT_GRASP_Z_M", "0.984"
                        )
                    )
                )
                regrasp_start_default = (
                    "255,335,415,495"
                    if target_uses_four_regrasps
                    else os.environ.get(
                        "P001_REGRASP_START_STEP", "255,375,495"
                    )
                )
                regrasp_duration_steps = int(
                    os.environ.get(
                        "P001_REGRASP_DURATION_STEPS",
                        "60" if target_uses_four_regrasps else "90",
                    )
                )
                regrasp_release_steps = int(
                    os.environ.get(
                        "P001_REGRASP_RELEASE_STEPS",
                        "10" if target_uses_four_regrasps else "15",
                    )
                )
                regrasp_close_start_steps = int(
                    os.environ.get(
                        "P001_REGRASP_CLOSE_START_STEPS",
                        "45" if target_uses_four_regrasps else "60",
                    )
                )
                if not (
                    0 < regrasp_release_steps < regrasp_close_start_steps
                    < regrasp_duration_steps
                ):
                    raise ValueError("invalid P0-01 regrasp phase schedule")
                right_low = np.asarray(
                    args.right_probe_pose, dtype=np.float64
                ).copy()
                right_high = right_low.copy()
                right_high[2] = max(1.000, float(right_low[2]) + 0.015)
                left_start = np.asarray(
                    args.left_probe_pose, dtype=np.float64
                ).copy()
                left_end = np.asarray(
                    args.left_probe_end_pose, dtype=np.float64
                ).copy()
                left_high = left_start.copy()
                left_high[2] = max(1.020, float(left_start[2]) + 0.080)
                if feedback_step < 30:
                    right_command = right_high
                    left_command = left_high
                elif feedback_step < 75:
                    alpha = (feedback_step - 30) / 45.0
                    weight = alpha * alpha * (3.0 - 2.0 * alpha)
                    right_command = right_high
                    left_command = left_high.copy()
                    left_command[:3] += weight * (
                        left_start[:3] - left_high[:3]
                    )
                elif feedback_step < 135:
                    alpha = (feedback_step - 75) / 60.0
                    weight = alpha * alpha * (3.0 - 2.0 * alpha)
                    right_command = right_high
                    left_command = left_start.copy()
                    left_command[:3] += weight * (
                        left_end[:3] - left_start[:3]
                    )
                elif feedback_step < 150:
                    alpha = (feedback_step - 135) / 15.0
                    right_weight = alpha * alpha * (3.0 - 2.0 * alpha)
                    right_command = right_high.copy()
                    right_command[:3] += right_weight * (
                        right_low[:3] - right_high[:3]
                    )
                    left_command = left_end
                else:
                    transforms = _numpy(
                        raw_env._p001_all_body_view.get_transforms()
                    ).reshape(-1, 7)
                    center_matrix = _xyzw_pose_matrix(transforms[0])
                    left_body_matrix = _xyzw_pose_matrix(transforms[1])
                    right_body_matrix = _xyzw_pose_matrix(transforms[19])
                    if "left_relative" not in feedback_state:
                        feedback_state["left_relative"] = (
                            np.linalg.inv(left_body_matrix)
                            @ _wxyz_pose_matrix(left_end)
                        )
                        feedback_state["right_relative"] = (
                            np.linalg.inv(right_body_matrix)
                            @ _wxyz_pose_matrix(right_low)
                        )
                        # Freeze a planar reference at first bilateral contact.
                        # Tracking the full live body transforms feeds any rib
                        # tilt back into the hand targets and can progressively
                        # lift the fan.  The controller should advance only the
                        # intended world-Z opening DOF.
                        feedback_state["left_body_reference"] = left_body_matrix.copy()
                        feedback_state["right_body_reference"] = right_body_matrix.copy()
                        feedback_state["center_reference"] = center_matrix.copy()
                        feedback_state["left_center_reference"] = center_matrix.copy()
                        feedback_state["left_span_reference_deg"] = np.asarray(
                            [float(last_info.get("p001_left_span_deg", 0.0))]
                        )
                        feedback_state["right_span_reference_deg"] = np.asarray(
                            [float(last_info.get("p001_right_span_deg", 0.0))]
                        )
                        feedback_state["commanded_left_span_deg"] = np.asarray(
                            [float(feedback_state["left_span_reference_deg"][0])]
                        )
                        feedback_state["commanded_right_span_deg"] = np.asarray(
                            [float(feedback_state["right_span_reference_deg"][0])]
                        )
                    left_span_deg = float(
                        last_info.get("p001_left_span_deg", 0.0)
                    )
                    right_span_deg = float(
                        last_info.get("p001_right_span_deg", 0.0)
                    )
                    total_span_deg = left_span_deg + right_span_deg
                    regrasp_start_steps = tuple(
                        int(value.strip())
                        for value in os.environ.get(
                            "P001_REGRASP_START_STEPS",
                            regrasp_start_default,
                        ).split(",")
                        if value.strip()
                    )
                    regrasp_cycle_index = int(
                        feedback_state.get("regrasp_completed_count", 0)
                    )
                    regrasp_start_step = (
                        regrasp_start_steps[regrasp_cycle_index]
                        if regrasp_cycle_index < len(regrasp_start_steps)
                        else -1
                    )
                    if (
                        float(args.target) == 90.0
                        and regrasp_cycle_index == 2
                        and "third_regrasp_start_step" not in feedback_state
                        and feedback_step
                        >= int(
                            os.environ.get(
                                "P001_THIRD_REGRASP_EARLY_START_STEP", "480"
                            )
                        )
                        and total_span_deg
                        < float(args.target)
                        - float(
                            os.environ.get(
                                "P001_THIRD_REGRASP_PROGRESS_MARGIN_DEG", "15"
                            )
                        )
                    ):
                        feedback_state["third_regrasp_start_step"] = int(
                            os.environ.get(
                                "P001_THIRD_REGRASP_EARLY_START_STEP", "480"
                            )
                        )
                    if (
                        regrasp_cycle_index == 2
                        and "third_regrasp_start_step" in feedback_state
                    ):
                        regrasp_start_step = int(
                            feedback_state["third_regrasp_start_step"]
                        )
                    if feedback_state.get("target_hold_active", False):
                        regrasp_start_step = -1
                    regrasp_end_step = (
                        regrasp_start_step + regrasp_duration_steps
                    )
                    if (
                        regrasp_start_step >= 0
                        and feedback_step >= regrasp_end_step
                        and "regrasp_left_hold" in feedback_state
                    ):
                        current_left_pose = _actual_tcp_pose_wxyz(
                            raw_env, "left"
                        )
                        feedback_state["left_relative"] = (
                            np.linalg.inv(left_body_matrix)
                            @ _wxyz_pose_matrix(current_left_pose)
                        )
                        feedback_state["left_body_reference"] = (
                            left_body_matrix.copy()
                        )
                        feedback_state["left_center_reference"] = (
                            center_matrix.copy()
                        )
                        feedback_state["left_span_reference_deg"] = (
                            np.asarray([left_span_deg])
                        )
                        feedback_state["commanded_left_span_deg"] = (
                            np.asarray([left_span_deg])
                        )
                        feedback_state["commanded_right_span_deg"] = (
                            np.asarray([right_span_deg])
                        )
                        contact_points = _contact_point_breakdown(
                            raw_env, "left_end_reverse"
                        )
                        if contact_points:
                            contact_point = np.asarray(
                                next(iter(contact_points.values())),
                                dtype=np.float64,
                            )
                            current_rotation = _wxyz_pose_matrix(
                                current_left_pose
                            )[:3, :3]
                            feedback_state[
                                "left_contact_reference"
                            ] = contact_point
                            feedback_state[
                                "left_contact_center_reference"
                            ] = center_matrix[:3, 3].copy()
                            feedback_state[
                                "left_contact_local"
                            ] = current_rotation.T @ (
                                contact_point - current_left_pose[:3]
                            )
                            feedback_state[
                                "left_contact_quat"
                            ] = current_left_pose[3:].copy()
                        feedback_state["regrasp_completed_count"] = (
                            regrasp_cycle_index + 1
                        )
                        for key in (
                            "regrasp_left_hold",
                            "regrasp_left_target",
                            "regrasp_right_hold",
                            "regrasp_contact_target",
                            "regrasp_contact_local",
                        ):
                            feedback_state.pop(key, None)
                    side_target_deg = 0.5 * float(args.target)
                    pace_span_deg = min(left_span_deg, right_span_deg)
                    fine_phase = pace_span_deg >= 20.0
                    balance_gain = 1.0 if fine_phase else 0.30
                    balance_correction_deg = balance_gain * (
                        right_span_deg - left_span_deg
                    )
                    side_drive_limit_deg = side_target_deg + 10.0
                    if total_span_deg >= 70.0:
                        common_lead_deg = 0.70 * (
                            float(args.target) - total_span_deg
                        )
                        tracking_correction_deg = 2.00 * (
                            right_span_deg - left_span_deg
                        )
                        requested_left_span_deg = left_span_deg + float(
                            np.clip(
                                common_lead_deg + tracking_correction_deg,
                                -12.0,
                                32.0,
                            )
                        )
                        requested_right_span_deg = right_span_deg + float(
                            np.clip(
                                common_lead_deg - tracking_correction_deg,
                                -12.0,
                                32.0,
                            )
                        )
                    else:
                        pace_lead_deg = 12.0 if fine_phase else 20.0
                        requested_left_span_deg = (
                            pace_span_deg
                            + pace_lead_deg
                            + balance_correction_deg
                        )
                        requested_right_span_deg = (
                            pace_span_deg
                            + pace_lead_deg
                            - balance_correction_deg
                        )
                    requested_left_span_deg = min(
                        side_drive_limit_deg, requested_left_span_deg
                    )
                    requested_right_span_deg = min(
                        side_drive_limit_deg, requested_right_span_deg
                    )
                    for side, requested_span_deg in (
                        ("left", requested_left_span_deg),
                        ("right", requested_right_span_deg),
                    ):
                        commanded_state = feedback_state[
                            f"commanded_{side}_span_deg"
                        ]
                        commanded_state[0] += float(
                            np.clip(
                                requested_span_deg - float(commanded_state[0]),
                                -0.35,
                                0.35,
                            )
                        )
                    hold_static_keys = (
                        "p001_finite",
                        "p001_joint_direction_ok",
                        "p001_joint_limits_ok",
                        "p001_safe_zone",
                        "p001_target_ok",
                        "p001_balance_ok",
                        "p001_net_increase_ok",
                        "p001_bilateral_expansion_ok",
                        "p001_left_end_contact",
                        "p001_right_end_contact",
                        "p001_wrong_end_clear",
                        "p001_root_stable",
                        "p001_planarity_ok",
                    )
                    if (
                        not bool(last_info.get("p001_failure", False))
                        and all(
                            bool(last_info.get(key, False))
                            for key in hold_static_keys
                        )
                    ):
                        feedback_state["target_hold_active"] = True
                    commanded_left_deg = float(
                        feedback_state["commanded_left_span_deg"][0]
                    ) - float(
                        feedback_state["left_span_reference_deg"][0]
                    )
                    commanded_right_deg = float(
                        feedback_state["commanded_right_span_deg"][0]
                    ) - float(
                        feedback_state["right_span_reference_deg"][0]
                    )
                    pivot = feedback_state["center_reference"][:3, 3]
                    desired_left_body = _rotate_matrix_about_world_z(
                        feedback_state["left_body_reference"],
                        feedback_state["left_center_reference"][:3, 3],
                        commanded_left_deg,
                    )
                    desired_right_body = _rotate_matrix_about_world_z(
                        feedback_state["right_body_reference"],
                        pivot,
                        -commanded_right_deg,
                    )
                    left_command = _matrix_to_pose_wxyz(
                        desired_left_body @ feedback_state["left_relative"]
                    )
                    right_command = _matrix_to_pose_wxyz(
                        desired_right_body @ feedback_state["right_relative"]
                    )
                    if "left_contact_reference" in feedback_state:
                        contact_pose = np.eye(4, dtype=np.float64)
                        contact_pose[:3, 3] = feedback_state[
                            "left_contact_reference"
                        ]
                        desired_contact = _rotate_matrix_about_world_z(
                            contact_pose,
                            feedback_state[
                                "left_contact_center_reference"
                            ],
                            commanded_left_deg,
                        )[:3, 3]
                        contact_quat = feedback_state[
                            "left_contact_quat"
                        ]
                        contact_rotation = _wxyz_pose_matrix(
                            np.concatenate(
                                [
                                    np.zeros(3, dtype=np.float64),
                                    contact_quat,
                                ]
                            )
                        )[:3, :3]
                        left_command[:3] = desired_contact - (
                            contact_rotation
                            @ feedback_state["left_contact_local"]
                        )
                        contact_z_offsets = tuple(
                            float(value.strip())
                            for value in os.environ.get(
                                "P001_CONTACT_Z_OFFSETS_M",
                                (
                                    "0,0,0.006,0.006"
                                    if target_uses_four_regrasps
                                    else "0,0,0.006"
                                ),
                            ).split(",")
                            if value.strip()
                        )
                        completed_regrasps = int(
                            feedback_state.get("regrasp_completed_count", 0)
                        )
                        contact_z_offset = contact_z_offsets[
                            min(
                                max(completed_regrasps - 1, 0),
                                len(contact_z_offsets) - 1,
                            )
                        ]
                        left_command[2] += contact_z_offset
                        if completed_regrasps >= 3:
                            left_command[2] = min(
                                float(left_command[2]),
                                float(left_end[2]),
                            )
                        left_command[3:] = contact_quat
                    else:
                        reachable_left_body = _rotate_matrix_about_world_z(
                            feedback_state["left_body_reference"],
                            feedback_state["left_center_reference"][:3, 3],
                            min(
                                commanded_left_deg,
                                float(
                                    os.environ.get(
                                        "P001_LEFT_WRIST_CAP_DEG", "30.0"
                                    )
                                ),
                            ),
                        )
                        reachable_left_command = _matrix_to_pose_wxyz(
                            reachable_left_body
                            @ feedback_state["left_relative"]
                        )
                        left_command[3:] = reachable_left_command[3:]
                        left_command[2] = float(left_end[2])
                    right_height_offset_m = (
                        0.003
                        if total_span_deg >= 70.0
                        else 0.0
                    )
                    right_command[2] = (
                        float(right_low[2]) + right_height_offset_m
                    )
                    if (
                        regrasp_start_step >= 0
                        and regrasp_start_step
                        <= feedback_step
                        < regrasp_end_step
                    ):
                        if "regrasp_left_hold" not in feedback_state:
                            feedback_state["regrasp_left_hold"] = (
                                _actual_tcp_pose_wxyz(raw_env, "left")
                            )
                            feedback_state["regrasp_left_target"] = (
                                feedback_state["regrasp_left_hold"].copy()
                            )
                            regrasp_radius_scales = tuple(
                                float(value.strip())
                                for value in os.environ.get(
                                    "P001_REGRASP_RADIUS_SCALES",
                                    os.environ.get(
                                        "P001_REGRASP_RADIUS_SCALE",
                                        (
                                            "0.67,0.67,0.92,0.92"
                                            if target_uses_four_regrasps
                                            else "0.67,0.67,1.0"
                                        ),
                                    ),
                                ).split(",")
                                if value.strip()
                            )
                            regrasp_radius_scale = regrasp_radius_scales[
                                min(
                                    regrasp_cycle_index,
                                    len(regrasp_radius_scales) - 1,
                                )
                            ]
                            if (
                                regrasp_cycle_index == 2
                                and "third_regrasp_start_step" in feedback_state
                            ):
                                regrasp_radius_scale = float(
                                    os.environ.get(
                                        "P001_EARLY_THIRD_REGRASP_RADIUS_SCALE",
                                        "0.92",
                                    )
                                )
                            contact_points = _contact_point_breakdown(
                                raw_env, "left_end_reverse"
                            )
                            if contact_points:
                                contact_point = np.asarray(
                                    next(iter(contact_points.values())),
                                    dtype=np.float64,
                                )
                                hold_rotation = _wxyz_pose_matrix(
                                    feedback_state["regrasp_left_hold"]
                                )[:3, :3]
                                target_contact_point = contact_point.copy()
                                target_contact_point[:2] = (
                                    center_matrix[:2, 3]
                                    + regrasp_radius_scale
                                    * (
                                        contact_point[:2]
                                        - center_matrix[:2, 3]
                                    )
                                )
                                feedback_state[
                                    "regrasp_contact_target"
                                ] = target_contact_point
                                feedback_state[
                                    "regrasp_contact_local"
                                ] = hold_rotation.T @ (
                                    contact_point
                                    - feedback_state["regrasp_left_hold"][:3]
                                )
                            else:
                                center_xy = center_matrix[:2, 3]
                                feedback_state[
                                    "regrasp_left_target"
                                ][:2] = center_xy + regrasp_radius_scale * (
                                    feedback_state["regrasp_left_hold"][:2]
                                    - center_xy
                                )
                            feedback_state["regrasp_right_hold"] = (
                                _actual_tcp_pose_wxyz(raw_env, "right")
                            )
                        left_command = feedback_state[
                            "regrasp_left_hold"
                        ].copy()
                        right_command = feedback_state[
                            "regrasp_right_hold"
                        ].copy()
                        rotate_alpha = float(
                            np.clip(
                                (
                                    feedback_step
                                    - regrasp_start_step
                                    - regrasp_release_steps
                                )
                                / float(
                                    regrasp_close_start_steps
                                    - regrasp_release_steps
                                ),
                                0.0,
                                1.0,
                            )
                        )
                        rotate_weight = rotate_alpha * rotate_alpha * (
                            3.0 - 2.0 * rotate_alpha
                        )
                        start_quat = feedback_state[
                            "regrasp_left_hold"
                        ][3:].copy()
                        regrasp_wrist_degs = tuple(
                            float(value.strip())
                            for value in os.environ.get(
                                "P001_REGRASP_WRIST_DEGS",
                                os.environ.get(
                                    "P001_REGRASP_WRIST_DEG",
                                    (
                                        "10,0,-10,-10"
                                        if target_uses_four_regrasps
                                        else "10,0,-10"
                                    ),
                                ),
                            ).split(",")
                            if value.strip()
                        )
                        regrasp_wrist_deg = regrasp_wrist_degs[
                            min(
                                regrasp_cycle_index,
                                len(regrasp_wrist_degs) - 1,
                            )
                        ]
                        regrasp_end_body = _rotate_matrix_about_world_z(
                            feedback_state["left_body_reference"],
                            feedback_state["left_center_reference"][:3, 3],
                            regrasp_wrist_deg,
                        )
                        regrasp_end_command = _matrix_to_pose_wxyz(
                            regrasp_end_body
                            @ feedback_state["left_relative"]
                        )
                        end_quat = regrasp_end_command[3:].copy()
                        if "regrasp_contact_local" in feedback_state:
                            end_rotation = _wxyz_pose_matrix(
                                np.concatenate(
                                    [np.zeros(3, dtype=np.float64), end_quat]
                                )
                            )[:3, :3]
                            feedback_state["regrasp_left_target"][:3] = (
                                feedback_state["regrasp_contact_target"]
                                - end_rotation
                                @ feedback_state["regrasp_contact_local"]
                            )
                        left_command[:3] += rotate_weight * (
                            feedback_state["regrasp_left_target"][:3]
                            - feedback_state["regrasp_left_hold"][:3]
                        )
                        if float(np.dot(start_quat, end_quat)) < 0.0:
                            end_quat *= -1.0
                        blended_quat = (
                            (1.0 - rotate_weight) * start_quat
                            + rotate_weight * end_quat
                        )
                        left_command[3:] = blended_quat / np.linalg.norm(
                            blended_quat
                        )
                    if feedback_state.get("target_hold_active", False):
                        if "target_hold_left_pose" not in feedback_state:
                            hold_blend = float(
                                np.clip(
                                    float(
                                        os.environ.get(
                                            "P001_TARGET_HOLD_COMMAND_BLEND",
                                            "0.5",
                                        )
                                    ),
                                    0.0,
                                    1.0,
                                )
                            )
                            actual_left_pose = _actual_tcp_pose_wxyz(
                                raw_env, "left"
                            )
                            actual_right_pose = _actual_tcp_pose_wxyz(
                                raw_env, "right"
                            )
                            feedback_state["target_hold_left_pose"] = (
                                actual_left_pose.copy()
                            )
                            feedback_state["target_hold_right_pose"] = (
                                actual_right_pose.copy()
                            )
                            # Preserve a bounded share of the in-plane command
                            # lead at target entry.  Freezing the measured TCP
                            # alone lets the compliant fan relax below the
                            # lower tolerance bound.  Keep measured wrist
                            # orientation and right-hand Z, while never
                            # lowering the left palm below its planned table
                            # clearance.
                            feedback_state["target_hold_left_pose"][:2] += (
                                hold_blend
                                * (left_command[:2] - actual_left_pose[:2])
                            )
                            feedback_state["target_hold_left_pose"][2] = max(
                                float(actual_left_pose[2]),
                                float(left_command[2]),
                                float(left_end[2])
                                + float(
                                    os.environ.get(
                                        "P001_TARGET_HOLD_LEFT_Z_CLEARANCE_M",
                                        "0.0",
                                    )
                                ),
                            )
                            feedback_state["target_hold_right_pose"][:2] += (
                                hold_blend
                                * (right_command[:2] - actual_right_pose[:2])
                            )
                        left_command = feedback_state[
                            "target_hold_left_pose"
                        ].copy()
                        right_command = feedback_state[
                            "target_hold_right_pose"
                        ].copy()
                action[:7] = right_command
                action[7:14] = left_command
                grip_start_step = float(
                    os.environ.get("P001_GRIP_START_STEP", "210")
                )
                grip_alpha = float(
                    np.clip(
                        (feedback_step - grip_start_step) / 30.0,
                        0.0,
                        1.0,
                    )
                )
                grip_weight = grip_alpha * grip_alpha * (
                    3.0 - 2.0 * grip_alpha
                )
                regrasp_start_steps = tuple(
                    int(value.strip())
                    for value in os.environ.get(
                        "P001_REGRASP_START_STEPS",
                        regrasp_start_default,
                    ).split(",")
                    if value.strip()
                )
                regrasp_cycle_index = int(
                    feedback_state.get("regrasp_completed_count", 0)
                )
                regrasp_start_step = (
                    regrasp_start_steps[regrasp_cycle_index]
                    if regrasp_cycle_index < len(regrasp_start_steps)
                    else -1
                )
                if (
                    regrasp_cycle_index == 2
                    and "third_regrasp_start_step" in feedback_state
                ):
                    regrasp_start_step = int(
                        feedback_state["third_regrasp_start_step"]
                    )
                if feedback_state.get("target_hold_active", False):
                    regrasp_start_step = -1
                if (
                    regrasp_start_step >= 0
                    and regrasp_start_step
                    <= feedback_step
                    < regrasp_start_step + regrasp_duration_steps
                ):
                    if (
                        feedback_step
                        < regrasp_start_step + regrasp_release_steps
                    ):
                        release_alpha = (
                            feedback_step - regrasp_start_step
                        ) / float(regrasp_release_steps)
                        grip_weight = 1.0 - (
                            release_alpha
                            * release_alpha
                            * (3.0 - 2.0 * release_alpha)
                        )
                    elif (
                        feedback_step
                        < regrasp_start_step + regrasp_close_start_steps
                    ):
                        grip_weight = 0.0
                    else:
                        close_alpha = (
                            feedback_step
                            - regrasp_start_step
                            - regrasp_close_start_steps
                        ) / float(
                            regrasp_duration_steps
                            - regrasp_close_start_steps
                        )
                        grip_weight = close_alpha * close_alpha * (
                            3.0 - 2.0 * close_alpha
                        )
                action[14:34] = (
                    grip_weight
                    * float(args.right_probe_grip)
                    * WUJI_FAN_GRIP
                )
                left_grip_scale = float(args.left_probe_grip)
                if (
                    float(args.target) == 90.0
                    and bool(last_info.get("p001_target_ok", False))
                    and bool(last_info.get("p001_balance_ok", False))
                ):
                    left_grip_scale = max(
                        left_grip_scale,
                        float(
                            os.environ.get(
                                "P001_FINAL_LEFT_GRIP", "0.10"
                            )
                        ),
                    )
                if feedback_state.get("target_hold_active", False):
                    left_grip_scale = max(
                        left_grip_scale,
                        float(
                            os.environ.get(
                                "P001_TARGET_HOLD_LEFT_GRIP",
                                str(args.left_probe_grip),
                            )
                        ),
                    )
                action[34:54] = (
                    grip_weight
                    * left_grip_scale
                    * WUJI_FAN_GRIP
                )
            elif args.mode in ("left_probe", "left_staged_probe"):
                if args.left_probe_pose is None:
                    raise ValueError("left_probe requires --left-probe-pose")
                action = np.zeros(env.action_space.shape, dtype=np.float64)
                left_probe_pose = np.asarray(
                    args.left_probe_pose, dtype=np.float64
                ).copy()
                high_pose = left_probe_pose.copy()
                high_pose[2] = max(1.020, float(left_probe_pose[2]) + 0.080)
                if int(raw_env.env_step) < 30:
                    commanded_pose = high_pose
                elif int(raw_env.env_step) < 75:
                    alpha = (int(raw_env.env_step) - 30) / 45.0
                    weight = alpha * alpha * (3.0 - 2.0 * alpha)
                    commanded_pose = high_pose.copy()
                    commanded_pose[:3] += weight * (
                        left_probe_pose[:3] - high_pose[:3]
                    )
                elif (
                    args.mode == "left_staged_probe"
                    and int(raw_env.env_step) < 135
                    and args.left_probe_end_pose is not None
                ):
                    alpha = (int(raw_env.env_step) - 75) / 60.0
                    weight = alpha * alpha * (3.0 - 2.0 * alpha)
                    commanded_pose = left_probe_pose.copy()
                    commanded_pose[:3] += weight * (
                        np.asarray(args.left_probe_end_pose, dtype=np.float64)[:3]
                        - left_probe_pose[:3]
                    )
                elif (
                    args.mode == "left_staged_probe"
                    and args.left_probe_end_pose is not None
                ):
                    commanded_pose = np.asarray(
                        args.left_probe_end_pose, dtype=np.float64
                    )
                else:
                    commanded_pose = left_probe_pose
                action[7:14] = commanded_pose
                action[34:54] = float(args.left_probe_grip) * WUJI_FAN_GRIP
            else:
                action = policy.action(int(raw_env.env_step))
                if args.mode == "hand_only":
                    action[:14] = 0.0
                    action[14:] = 0.05
                elif args.mode == "right_pose":
                    action[7:14] = 0.0
                elif args.mode == "left_pose":
                    action[:7] = 0.0
            if executed == 0:
                print(
                    "[P001_LIVE_STEP_BEGIN] "
                    + json.dumps(
                        {
                            "action_shape": list(np.asarray(action).shape),
                            "action_finite": bool(np.all(np.isfinite(action))),
                            "action_head": np.asarray(action)[:14].tolist(),
                            "env_step": int(raw_env.env_step),
                            "raw_right_tcp": np.asarray(
                                raw_env._right_tcp_pose
                            ).tolist(),
                            "raw_left_tcp": np.asarray(
                                raw_env._left_tcp_pose
                            ).tolist(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            observation, _reward, terminated, truncated, info = env.step(action)
            if executed == 0:
                print("[P001_LIVE_STEP_END] first step returned", flush=True)
            last_info = dict(info)
            executed += 1
            terminal = bool(terminated or truncated)
            if (
                executed == 1
                or executed % 15 == 0
                or executed == args.steps
                or terminal
            ):
                trajectory_samples.append(
                    {
                        "step": executed,
                        "right_command_wxyz": np.asarray(raw_env._right_tcp_pose).tolist(),
                        "left_command_wxyz": np.asarray(raw_env._left_tcp_pose).tolist(),
                        "right_actual_wxyz": _actual_tcp_pose_wxyz(raw_env, "right").tolist(),
                        "left_actual_wxyz": _actual_tcp_pose_wxyz(raw_env, "left").tolist(),
                        "robot_table_contact_force_n": float(
                            last_info.get("p001_robot_table_contact_force_n", 0.0)
                        ),
                        "left_end_contact_force_n": float(
                            last_info.get("p001_left_end_contact_force_n", 0.0)
                        ),
                        "left_end_reverse_contact_force_by_body_n": (
                            _contact_force_breakdown(raw_env, "left_end_reverse")
                        ),
                        "left_end_reverse_contact_point_by_body_m": (
                            _contact_point_breakdown(
                                raw_env, "left_end_reverse"
                            )
                        ),
                        "right_end_contact_force_n": float(
                            last_info.get("p001_right_end_contact_force_n", 0.0)
                        ),
                        "right_end_net_contact_force_n": _net_contact_force(
                            raw_env, "right_end_net"
                        ),
                        "span_deg": float(last_info.get("p001_span_deg", 0.0)),
                        "left_span_deg": float(
                            last_info.get("p001_left_span_deg", 0.0)
                        ),
                        "right_span_deg": float(
                            last_info.get("p001_right_span_deg", 0.0)
                        ),
                        "balance_error_deg": float(
                            last_info.get("p001_balance_error_deg", 0.0)
                        ),
                        "max_body_linear_speed_mps": float(
                            last_info.get(
                                "p001_max_body_linear_speed_mps", 0.0
                            )
                        ),
                        "max_joint_speed_radps": float(
                            last_info.get("p001_max_joint_speed_radps", 0.0)
                        ),
                        "success_counter": int(
                            last_info.get("p001_success_counter", 0)
                        ),
                        "right_end_contact_force_by_body_n": (
                            _contact_force_breakdown(raw_env, "right_end")
                        ),
                        "right_end_reverse_contact_force_by_body_n": (
                            _contact_force_breakdown(raw_env, "right_end_reverse")
                        ),
                        "right_wrong_end_contact_force_by_body_n": (
                            _contact_force_breakdown(raw_env, "right_wrong_end")
                        ),
                        "left_wrong_end_contact_force_by_body_n": (
                            _contact_force_breakdown(raw_env, "left_wrong_end")
                        ),
                        "left_wrong_end_contact_force_n": float(
                            last_info.get(
                                "p001_left_wrong_end_contact_force_n", 0.0
                            )
                        ),
                        "right_wrong_end_contact_force_n": float(
                            last_info.get(
                                "p001_right_wrong_end_contact_force_n", 0.0
                            )
                        ),
                        "planarity_error_m": float(
                            last_info.get("p001_planarity_error_m", 0.0)
                        ),
                        "root_displacement_m": float(
                            last_info.get("p001_root_displacement_m", 0.0)
                        ),
                        "root_lift_m": float(
                            last_info.get("p001_root_lift_m", 0.0)
                        ),
                        "robot_table_contact_force_by_body_n": (
                            _contact_force_breakdown(raw_env, "robot_table")
                        ),
                        "fan_table_contact_force_by_body_n": (
                            _contact_force_breakdown(raw_env, "fan_table")
                        ),
                        "fan_any_right_contact_force_by_body_n": (
                            _contact_force_breakdown(raw_env, "fan_any_right")
                        ),
                        "fan_any_right_reverse_contact_force_by_body_n": (
                            _contact_force_breakdown(raw_env, "fan_any_right_reverse")
                        ),
                    }
                )
            if terminal:
                break
        result.update(
            {
                "executed_steps": executed,
                "terminal": terminal,
                "success": bool(last_info.get("succeed", False)),
                "failure": bool(last_info.get("failed", False)),
                "trajectory_samples": trajectory_samples,
                "task_info": {
                    key: value
                    for key, value in last_info.items()
                    if str(key).startswith("p001_")
                },
                "observation_keys": sorted(observation),
                "observation_arrays": _array_shapes(observation),
                "initial_observation_arrays": initial_shapes,
                "runtime_diagnostics": {
                    "arm_ik_backend": str(raw_env._arm_ik_backend),
                    "arm_ik_enabled": bool(raw_env._haorui_ik_enabled),
                    "right_tcp_command_wxyz": np.asarray(
                        raw_env._right_tcp_pose, dtype=np.float64
                    ),
                    "left_tcp_command_wxyz": np.asarray(
                        raw_env._left_tcp_pose, dtype=np.float64
                    ),
                    "right_tcp_actual_matrix": raw_env._robot_body_pose_matrix(
                        raw_env._control_robot("right"),
                        raw_env._ee_body_indices["right"],
                    ),
                    "left_tcp_actual_matrix": raw_env._robot_body_pose_matrix(
                        raw_env._control_robot("left"),
                        raw_env._ee_body_indices["left"],
                    ),
                    "right_robot_joint_positions": _numpy(
                        raw_env._control_robot("right").data.joint_pos
                    ),
                    "left_robot_joint_positions": _numpy(
                        raw_env._control_robot("left").data.joint_pos
                    ),
                    "right_robot_body_poses": _robot_body_poses(raw_env, "right"),
                    "left_robot_body_poses": _robot_body_poses(raw_env, "left"),
                    "robot_table_contact_force_by_body_n": (
                        _contact_force_breakdown(raw_env, "robot_table")
                    ),
                    "fan_table_contact_force_by_body_n": (
                        _contact_force_breakdown(raw_env, "fan_table")
                    ),
                    "left_end_contact_force_by_body_n": (
                        _contact_force_breakdown(raw_env, "left_end")
                    ),
                    "right_end_contact_force_by_body_n": (
                        _contact_force_breakdown(raw_env, "right_end")
                    ),
                    "right_end_reverse_contact_force_by_body_n": (
                        _contact_force_breakdown(raw_env, "right_end_reverse")
                    ),
                    "left_wrong_end_contact_force_by_body_n": (
                        _contact_force_breakdown(raw_env, "left_wrong_end")
                    ),
                    "right_wrong_end_contact_force_by_body_n": (
                        _contact_force_breakdown(raw_env, "right_wrong_end")
                    ),
                    "articulation_root_transforms": _numpy(
                        raw_env._p001_articulation_view.get_root_transforms()
                    ),
                    "center_transforms": _numpy(
                        raw_env._p001_center_view.get_transforms()
                    ),
                    "all_body_transforms": _numpy(
                        raw_env._p001_all_body_view.get_transforms()
                    ),
                    "active_joint_positions_rad": raw_env._active_joint_state()[0],
                    "active_joint_velocities_radps": raw_env._active_joint_state()[1],
                    "authored_world_bboxes": {
                        "center": _prim_world_bbox(raw_env, CENTER_BODY_PATH),
                        "left_end": _prim_world_bbox(raw_env, LEFT_END_BODY_PATH),
                        "right_end": _prim_world_bbox(raw_env, RIGHT_END_BODY_PATH),
                    },
                    "authored_collision_world_bboxes": {
                        "left_end": _collision_world_bboxes(raw_env, LEFT_END_BODY_PATH),
                        "right_end": _collision_world_bboxes(raw_env, RIGHT_END_BODY_PATH),
                    },
                    "runtime_collision_world_bboxes": {
                        "left_end": _runtime_collision_world_bboxes(
                            raw_env,
                            LEFT_END_BODY_PATH,
                            _xyzw_pose_matrix(
                                _numpy(raw_env._p001_left_end_view.get_transforms())[0]
                            ),
                        ),
                        "right_end": _runtime_collision_world_bboxes(
                            raw_env,
                            RIGHT_END_BODY_PATH,
                            _xyzw_pose_matrix(
                                _numpy(raw_env._p001_right_end_view.get_transforms())[0]
                            ),
                        ),
                        "right_finger1_link4": _robot_runtime_collision_bboxes(
                            raw_env, "right", "right_finger1_link4"
                        ),
                        "right_finger2_link4": _robot_runtime_collision_bboxes(
                            raw_env, "right", "right_finger2_link4"
                        ),
                        "right_finger3_link4": _robot_runtime_collision_bboxes(
                            raw_env, "right", "right_finger3_link4"
                        ),
                        "right_finger4_link4": _robot_runtime_collision_bboxes(
                            raw_env, "right", "right_finger4_link4"
                        ),
                        "right_finger5_link4": _robot_runtime_collision_bboxes(
                            raw_env, "right", "right_finger5_link4"
                        ),
                        "left_finger1_link4": _robot_runtime_collision_bboxes(
                            raw_env, "left", "left_finger1_link4"
                        ),
                        "left_finger2_link4": _robot_runtime_collision_bboxes(
                            raw_env, "left", "left_finger2_link4"
                        ),
                    },
                    "robot_fingertip_prim_structure": {
                        "fan_right": _prim_structure(raw_env, RIGHT_END_BODY_PATH),
                        "right_finger1_link4": _prim_structure(
                            raw_env,
                            "/World/HandsOnRobots/RightRobot/base_right/right_finger1_link4/collisions",
                        ),
                        "right_finger2_link4": _prim_structure(
                            raw_env,
                            "/World/HandsOnRobots/RightRobot/base_right/right_finger2_link4/collisions",
                        ),
                    },
                    "collision_filter_diagnostics": {
                        "groups": _collision_groups(raw_env),
                        "filtered_pairs": _filtered_pairs(raw_env),
                        "fan_right_colliders": _collision_state_under(
                            raw_env, RIGHT_END_BODY_PATH
                        ),
                        "right_finger5_colliders": _collision_state_under(
                            raw_env,
                            "/World/HandsOnRobots/RightRobot/base_right/right_finger5_link4/collisions",
                        ),
                        "fan_right_collider": _physics_prim_properties(
                            raw_env,
                            f"{RIGHT_END_BODY_PATH}/Collisions/Foldingfan001_Collider19",
                        ),
                        "fan_right_body": _physics_prim_properties(
                            raw_env, RIGHT_END_BODY_PATH
                        ),
                        "fan_right_joint": _physics_prim_properties(
                            raw_env, f"{RIGHT_END_BODY_PATH}/Prop019_joint"
                        ),
                        "table_body": _physics_prim_properties(
                            raw_env, str(raw_env._p001_contact_paths["robot_table"][0])
                        ),
                    },
                },
            }
        )
        if args.image_dir is not None:
            saved_images.extend(
                _save_rgb_arrays(observation, args.image_dir, f"step_{executed:04d}")
            )
            result["saved_images"] = saved_images
        print("[P001_LIVE_RESULT] " + json.dumps(_jsonable(result), sort_keys=True), flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except BaseException:
        print("[P001_LIVE_EXCEPTION]", flush=True)
        traceback.print_exc()
        raise
    finally:
        print("[P001_LIVE_CLOSE_BEGIN]", flush=True)
        env.close()
        print("[P001_LIVE_CLOSE_END]", flush=True)


if __name__ == "__main__":
    main()
