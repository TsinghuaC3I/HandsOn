#!/usr/bin/env python3
"""Bounded Isaac Sim 5.1 live audit for the BC025 Terry Cloth runtime.

The audit is deliberately narrower than a collection certification.  It
checks the authored scene, the real PhysX surface readback, reset
randomization, collision broad-phase ownership, zero-action stability, and
the three deferred-path RTX cameras.  A complete demonstrated manipulation
episode, a human viewport session, and physical surface-solver replay remain
separate gates and are never inferred from this smoke run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import threading
import traceback
from typing import Any, Iterable, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
HANDSON_ROOT = REPO_ROOT / "handson"
for candidate in (REPO_ROOT, HANDSON_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


TASK_NAME = "b320_towel_corner_synchronized_spread"
TASK_ID = "B320-BC025"
SURFACE_ROOT_PATH = "/World/Task/B320Benchmark/BC025TerryCloth"
SURFACE_MESH_PATH = f"{SURFACE_ROOT_PATH}/mesh"
TABLE_ROOT_PATH = "/World/Task/B320Benchmark/worktable"
TARGET_ROOT_PATH = "/World/Task/B320Benchmark/towel_target_frame"
PROXY_ROOT_PATH = "/World/E01HandContactProxies"
ROBOT_ROOT_PATH = "/World/HandsOnRobots"
EXPECTED_CAMERA_PRIMS = {
    "head_camera",
    "wrist_left_camera",
    "wrist_right_camera",
}
EXPECTED_STATE_KEYS = {
    "tcp_pose",
    "gripper_pose",
    "towel_pose",
    "towel_corners",
    "target_bounds",
    "spread_metrics",
    "grasp_state",
    "garment_initial_state",
    "table_delta_height",
}
OVERLAP_EPSILON_M = 1.0e-9
SUPPORT_OVERLAP_LIMIT_M = 0.010
SAME_HAND_OVERLAP_LIMIT_M = 0.120
PROXY_HAND_OVERLAP_LIMIT_M = 0.100
ZERO_ACTION_SURFACE_DRIFT_LIMIT_M = 0.012
ZERO_ACTION_PALM_DRIFT_LIMIT_M = 0.012


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--reset-count", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument(
        "--disable-ground-truth-recording",
        action="store_true",
        help="diagnostic-only: permit CPU PhysX or isolate the ground-truth recorder; does not certify replay",
    )
    return parser


def _phase(name: str) -> None:
    """Emit a flushed boundary so a Kit/CUDA stall has an auditable site."""

    print(f"BC025_LIVE_PHASE={name}", flush=True)


def _unwrap(environment: Any) -> Any:
    current = environment
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_bc025_surface_translation_sample_m"):
            return current
        next_environment = getattr(current, "env", None)
        if next_environment is None:
            next_environment = getattr(current, "_env", None)
        if next_environment is None:
            break
        current = next_environment
    raise RuntimeError(
        "Could not unwrap HandsOnB320TowelIsaacLabEnv from the policy adapter"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _numpy(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    if hasattr(value, "torch"):
        value = value.torch
    elif hasattr(value, "warp"):
        value = value.warp
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype).copy()


def _runtime_contract() -> dict[str, str]:
    try:
        isaac_sim_version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Isaac Sim distribution metadata is unavailable") from exc
    if isaac_sim_version != "5.1.0.0":
        raise RuntimeError(
            f"{TASK_ID} requires Isaac Sim 5.1.0.0, got {isaac_sim_version!r}"
        )

    import isaaclab

    source = Path(isaaclab.__file__).resolve()
    version_path: Path | None = None
    for parent in (source.parent, *source.parents):
        candidate = parent / "VERSION"
        if candidate.is_file():
            version_path = candidate
            break
    if version_path is None:
        raise RuntimeError(f"Could not find IsaacLab VERSION near {source}")
    isaaclab_version = version_path.read_text(encoding="utf-8").strip()
    if isaaclab_version != "2.3.2":
        raise RuntimeError(
            f"{TASK_ID} requires IsaacLab 2.3.2, got {isaaclab_version!r}"
        )
    return {
        "isaac_sim_version": isaac_sim_version,
        "isaaclab_version": isaaclab_version,
        "isaaclab_source": str(source),
        "isaaclab_version_path": str(version_path),
    }


def _prim_range(root: Any) -> Iterable[Any]:
    from pxr import Usd

    try:
        predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
        return Usd.PrimRange(root, predicate)
    except (AttributeError, TypeError):
        return Usd.PrimRange(root)


def _stage_prim_range(stage: Any) -> Iterable[Any]:
    from pxr import Usd

    try:
        predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
        return Usd.PrimRange.Stage(stage, predicate)
    except (AttributeError, TypeError):
        return stage.Traverse()


def _collision_enabled(prim: Any, usd_physics: Any) -> bool:
    if not prim.HasAPI(usd_physics.CollisionAPI):
        return False
    collision = usd_physics.CollisionAPI(prim)
    enabled = collision.GetCollisionEnabledAttr()
    if enabled and enabled.IsValid() and enabled.HasAuthoredValueOpinion():
        value = enabled.Get()
        if value is not None and not bool(value):
            return False
    return True


def _visibility(prim: Any, usd_geom: Any) -> str:
    if not prim.IsA(usd_geom.Gprim):
        return "non_geometry"
    value = usd_geom.Imageable(prim).ComputeVisibility()
    return str(value)


def _aabb_from_prim(cache: Any, prim: Any) -> tuple[np.ndarray, np.ndarray]:
    from pxr import UsdGeom

    if not prim.IsA(UsdGeom.Boundable):
        raise RuntimeError(f"BC025 collider is not boundable: {prim.GetPath()}")
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    low = np.asarray([float(value) for value in aligned.GetMin()], dtype=np.float64)
    high = np.asarray([float(value) for value in aligned.GetMax()], dtype=np.float64)
    if (
        low.shape != (3,)
        or high.shape != (3,)
        or not np.all(np.isfinite(low))
        or not np.all(np.isfinite(high))
        or np.any(high < low)
    ):
        raise RuntimeError(f"BC025 collider has invalid world bounds: {prim.GetPath()}")
    return low, high


def _surface_aabb(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(nodes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (3,) or not np.all(np.isfinite(values)):
        raise RuntimeError(f"BC025 surface readback cannot form an AABB: {values.shape}")
    return np.min(values, axis=0), np.max(values, axis=0)


def _overlap_depth(low_a: np.ndarray, high_a: np.ndarray, low_b: np.ndarray, high_b: np.ndarray) -> tuple[float, np.ndarray]:
    depths = np.minimum(high_a, high_b) - np.maximum(low_a, low_b)
    if not np.all(depths > OVERLAP_EPSILON_M):
        return 0.0, depths
    return float(np.min(depths)), depths


def _collision_kind(path: str) -> tuple[str, str | None]:
    if path == SURFACE_MESH_PATH:
        return "surface", None
    if path == TABLE_ROOT_PATH or path.startswith(TABLE_ROOT_PATH + "/"):
        return "table", None
    if path.startswith(PROXY_ROOT_PATH + "/"):
        side = "left" if "/left_" in path.lower() else "right" if "/right_" in path.lower() else None
        return "proxy", side
    for side in ("left", "right"):
        prefix = f"{ROBOT_ROOT_PATH}/{side.capitalize()}Robot"
        if path == prefix or path.startswith(prefix + "/"):
            return "robot", side
    return "other", None


@dataclass
class _AuditState:
    checks: dict[str, dict[str, Any]]
    errors: list[str]
    warnings: list[str]

    def check(self, name: str, passed: bool, detail: Any = None) -> None:
        record: dict[str, Any] = {"status": "pass" if passed else "fail"}
        if detail is not None:
            record["detail"] = detail
        self.checks[name] = record
        if not passed:
            self.errors.append(f"{name}: {detail}")


def _collision_inventory(
    stage: Any,
    *,
    surface_nodes: np.ndarray,
) -> dict[str, Any]:
    """Inventory every active task/hand collision Gprim and audit pair policy.

    This is a conservative broad-phase AABB supplement.  It is intentionally
    not described as a triangle-level contact proof; Isaac Sim 5.1's public
    rigid contact tensor view does not expose a deformable contact manifold.
    """

    from pxr import Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
        useExtentsHint=True,
    )
    records: list[dict[str, Any]] = []
    for prim in _stage_prim_range(stage):
        if not prim.IsValid() or not prim.IsActive():
            continue
        path = str(prim.GetPath())
        kind, side = _collision_kind(path)
        if kind == "other":
            continue
        if not prim.IsA(UsdGeom.Gprim) or not _collision_enabled(prim, UsdPhysics):
            continue
        low, high = _aabb_from_prim(cache, prim)
        if path == SURFACE_MESH_PATH:
            low, high = _surface_aabb(surface_nodes)
            bounds_source = "physx_surface_nodal_readback"
        else:
            bounds_source = "usd_bbox_cache_world_bound"
        records.append(
            {
                "path": path,
                "kind": kind,
                "side": side,
                "visibility": _visibility(prim, UsdGeom),
                "minimum_m": low.tolist(),
                "maximum_m": high.tolist(),
                "bounds_source": bounds_source,
                "classification": prim.GetCustomDataByKey("handson:classification"),
                "collision_role": prim.GetCustomDataByKey("handson:collisionRole"),
            }
        )

    by_path = {str(record["path"]): record for record in records}
    required = {
        SURFACE_MESH_PATH,
        f"{PROXY_ROOT_PATH}/left_contact_proxy",
        f"{PROXY_ROOT_PATH}/right_contact_proxy",
    }
    missing = sorted(required - set(by_path))
    if not any(record["kind"] == "table" for record in records):
        # The authored support is normally an Xform root with a collision
        # mesh below it, so requiring TABLE_ROOT_PATH itself would reject a
        # valid scene merely because the root is not a Gprim.
        missing.append(f"{TABLE_ROOT_PATH}/* collision Gprim")
    if missing:
        raise RuntimeError(f"BC025 collision inventory is missing {missing}")

    def policy(first: Mapping[str, Any], second: Mapping[str, Any], depth: float) -> tuple[str, bool, float | None]:
        kinds = {str(first["kind"]), str(second["kind"])}
        first_side, second_side = first.get("side"), second.get("side")
        if kinds == {"surface", "table"}:
            return "intended_towel_table_support", depth <= SUPPORT_OVERLAP_LIMIT_M, SUPPORT_OVERLAP_LIMIT_M
        if kinds == {"proxy", "robot"} and first_side == second_side:
            return "same_hand_proxy_attached_to_robot", depth <= PROXY_HAND_OVERLAP_LIMIT_M, PROXY_HAND_OVERLAP_LIMIT_M
        if kinds == {"robot"} and first_side == second_side:
            return "same_hand_robot_self_articulation", depth <= SAME_HAND_OVERLAP_LIMIT_M, SAME_HAND_OVERLAP_LIMIT_M
        return "forbidden_reset_overlap", False, 0.0

    pairs: list[dict[str, Any]] = []
    unapproved: list[dict[str, Any]] = []
    for index, first in enumerate(records):
        first_low = np.asarray(first["minimum_m"], dtype=np.float64)
        first_high = np.asarray(first["maximum_m"], dtype=np.float64)
        for second in records[index + 1 :]:
            second_low = np.asarray(second["minimum_m"], dtype=np.float64)
            second_high = np.asarray(second["maximum_m"], dtype=np.float64)
            depth, axis_depths = _overlap_depth(
                first_low,
                first_high,
                second_low,
                second_high,
            )
            if depth <= 0.0:
                continue
            reason, allowed, limit = policy(first, second, depth)
            pair = {
                "left": first["path"],
                "left_kind": first["kind"],
                "left_side": first["side"],
                "right": second["path"],
                "right_kind": second["kind"],
                "right_side": second["side"],
                "axis_overlap_m": axis_depths.tolist(),
                "overlap_depth_m": depth,
                "policy": reason,
                "allowlisted": bool(allowed),
                "allowlist_limit_m": limit,
            }
            pairs.append(pair)
            if not allowed:
                unapproved.append(pair)

    robot_records = [record for record in records if record["kind"] == "robot"]
    robot_by_side = {
        side: [record for record in robot_records if record["side"] == side]
        for side in ("left", "right")
    }
    non_hand_robot_colliders = [
        record["path"]
        for record in robot_records
        if "palm" not in str(record["path"]).lower()
        and "finger" not in str(record["path"]).lower()
    ]
    return {
        "method": "active_collision_gprim_world_aabb_broad_phase_with_explicit_reset_allowlist",
        "surface_readback_source": "PhysX nodal positions",
        "records": records,
        "counts": {
            "all_active_collision_gprims": len(records),
            "surface": sum(record["kind"] == "surface" for record in records),
            "table": sum(record["kind"] == "table" for record in records),
            "proxies": sum(record["kind"] == "proxy" for record in records),
            "robot_total": len(robot_records),
            "robot_left": len(robot_by_side["left"]),
            "robot_right": len(robot_by_side["right"]),
        },
        "robot_collision_paths_by_side": {
            side: sorted(record["path"] for record in robot_by_side[side])
            for side in ("left", "right")
        },
        "non_hand_robot_colliders": sorted(non_hand_robot_colliders),
        "positive_overlap_pairs": pairs,
        "unapproved_overlap_pairs": unapproved,
        "formal_unapproved_overlap_gate": not bool(unapproved),
    }


def _root_translation(stage: Any) -> list[float]:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(SURFACE_ROOT_PATH)
    if not prim.IsValid():
        raise RuntimeError(f"BC025 surface root is missing: {SURFACE_ROOT_PATH}")
    ops = [
        op
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    if len(ops) != 1:
        raise RuntimeError(
            f"BC025 surface root must have exactly one translate op, got {len(ops)}"
        )
    value = ops[0].Get()
    if value is None:
        raise RuntimeError("BC025 surface root translate op has no value")
    result = [float(component) for component in value]
    if len(result) != 3 or not np.all(np.isfinite(result)):
        raise RuntimeError(f"BC025 surface root translation is invalid: {result}")
    return result


def _mesh_contract(stage: Any) -> dict[str, Any]:
    from pxr import UsdGeom

    root = stage.GetPrimAtPath(SURFACE_ROOT_PATH)
    mesh_prim = stage.GetPrimAtPath(SURFACE_MESH_PATH)
    if not root.IsValid() or not mesh_prim.IsValid():
        raise RuntimeError("BC025 surface root or mesh is absent")
    if not root.IsA(UsdGeom.Xform) or not mesh_prim.IsA(UsdGeom.Mesh):
        raise RuntimeError("BC025 surface hierarchy is not Xform -> Mesh")
    mesh = UsdGeom.Mesh(mesh_prim)
    points_value = mesh.GetPointsAttr().Get()
    counts_value = mesh.GetFaceVertexCountsAttr().Get()
    indices_value = mesh.GetFaceVertexIndicesAttr().Get()
    points = [] if points_value is None else points_value
    counts = [] if counts_value is None else counts_value
    indices = [] if indices_value is None else indices_value
    if len(points) != 1681 or len(counts) != 3200 or len(indices) != 9600:
        raise RuntimeError(
            "BC025 authored topology drifted: "
            f"points={len(points)}, faces={len(counts)}, indices={len(indices)}"
        )
    if any(int(count) != 3 for count in counts):
        raise RuntimeError("BC025 surface contains a non-triangular face")
    rest_points_value = mesh_prim.GetAttribute("omniphysics:restShapePoints").Get()
    rest_triangles_value = mesh_prim.GetAttribute("omniphysics:restTriVtxIndices").Get()
    rest_points = [] if rest_points_value is None else rest_points_value
    rest_triangles = [] if rest_triangles_value is None else rest_triangles_value
    if len(rest_points) != 1681 or len(rest_triangles) != 3200:
        raise RuntimeError(
            "BC025 PhysX rest topology drifted: "
            f"points={len(rest_points)}, triangles={len(rest_triangles)}"
        )
    return {
        "root_path": SURFACE_ROOT_PATH,
        "mesh_path": SURFACE_MESH_PATH,
        "root_type": root.GetTypeName(),
        "mesh_type": mesh_prim.GetTypeName(),
        "root_active": bool(root.IsActive()),
        "mesh_active": bool(mesh_prim.IsActive()),
        "authored_points": len(points),
        "authored_faces": len(counts),
        "authored_face_indices": len(indices),
        "physx_rest_points": len(rest_points),
        "physx_rest_triangles": len(rest_triangles),
        "root_translate_m": _root_translation(stage),
        "root_applied_schemas": list(root.GetAppliedSchemas()),
        "mesh_applied_schemas": list(mesh_prim.GetAppliedSchemas()),
        "root_custom_data": root.GetCustomData(),
        "mesh_custom_data": mesh_prim.GetCustomData(),
    }


def _scene_classification_audit(stage: Any, raw: Any) -> dict[str, Any]:
    from pxr import UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(SURFACE_ROOT_PATH)
    mesh = stage.GetPrimAtPath(SURFACE_MESH_PATH)
    table = stage.GetPrimAtPath(TABLE_ROOT_PATH)
    target = stage.GetPrimAtPath(TARGET_ROOT_PATH)
    if not all(prim.IsValid() for prim in (root, mesh, table, target)):
        raise RuntimeError("BC025 task-owned root/table/target prim is missing")

    classifications: dict[str, Any] = {
        "task_root": {
            "path": SURFACE_ROOT_PATH,
            "classification": root.GetCustomDataByKey("handson:classification"),
            "visible": _visibility(root, UsdGeom),
            "collision": bool(root.HasAPI(UsdPhysics.CollisionAPI)),
        },
        "surface_mesh": {
            "path": SURFACE_MESH_PATH,
            "classification": mesh.GetCustomDataByKey("handson:classification"),
            "collision_role": mesh.GetCustomDataByKey("handson:collisionRole"),
            "visible": _visibility(mesh, UsdGeom),
            "collision": bool(mesh.HasAPI(UsdPhysics.CollisionAPI)),
            "same_mesh_flag": bool(
                mesh.GetCustomDataByKey("handson:renderSimulationCollisionSameMesh")
            ),
        },
        "table": {
            "path": TABLE_ROOT_PATH,
            "classification": table.GetCustomDataByKey("handson:classification"),
            "collision_role": table.GetCustomDataByKey("handson:collisionRole"),
            "visible": _visibility(table, UsdGeom),
        },
        "target": {
            "path": TARGET_ROOT_PATH,
            "classification": target.GetCustomDataByKey("handson:classification"),
            "visible": _visibility(target, UsdGeom),
            "collision": bool(target.HasAPI(UsdPhysics.CollisionAPI)),
        },
        "proxies": [],
        "robot_collision_visibility": [],
    }
    for prim in _prim_range(stage.GetPrimAtPath(PROXY_ROOT_PATH)):
        if not prim.IsValid() or not prim.IsActive() or not prim.IsA(UsdGeom.Gprim):
            continue
        classifications["proxies"].append(
            {
                "path": str(prim.GetPath()),
                "classification": prim.GetCustomDataByKey("handson:classification"),
                "collision_role": prim.GetCustomDataByKey("handson:collisionRole"),
                "visual_contract": prim.GetCustomDataByKey("handson:visualContract"),
                "visible": _visibility(prim, UsdGeom),
                "collision": bool(prim.HasAPI(UsdPhysics.CollisionAPI)),
            }
        )
    robot_root = stage.GetPrimAtPath(ROBOT_ROOT_PATH)
    if robot_root.IsValid():
        for prim in _prim_range(robot_root):
            if not prim.IsValid() or not prim.IsActive() or not prim.IsA(UsdGeom.Gprim):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            classifications["robot_collision_visibility"].append(
                {
                    "path": str(prim.GetPath()),
                    "visible": _visibility(prim, UsdGeom),
                    "name": prim.GetName(),
                }
            )

    required_root_schemas = {
        "OmniPhysicsDeformableBodyAPI",
        "OmniPhysicsBodyAPI",
        "PhysxSurfaceDeformableBodyAPI",
        "PhysxBaseDeformableBodyAPI",
    }
    required_mesh_schemas = {
        "OmniPhysicsSurfaceDeformableSimAPI",
        "PhysicsCollisionAPI",
        "PhysxCollisionAPI",
    }
    root_schemas = set(str(value) for value in root.GetAppliedSchemas())
    mesh_schemas = set(str(value) for value in mesh.GetAppliedSchemas())
    failures: list[str] = []
    if classifications["task_root"]["classification"] != "task_dynamic_surface_root":
        failures.append("task root classification")
    if classifications["surface_mesh"]["classification"] != "task_visible_deformable_surface":
        failures.append("surface classification")
    if classifications["surface_mesh"]["collision_role"] != "formal_surface":
        failures.append("surface collision role")
    if classifications["surface_mesh"]["visible"] == "invisible":
        failures.append("surface is invisible")
    if not classifications["surface_mesh"]["collision"]:
        failures.append("surface lacks CollisionAPI")
    if not classifications["surface_mesh"]["same_mesh_flag"]:
        failures.append("render/simulation/collision same-mesh flag")
    if classifications["table"]["classification"] != "task_visible_support":
        failures.append("table classification")
    if classifications["target"]["classification"] != "task_visible_marker":
        failures.append("target classification")
    if classifications["target"]["collision"]:
        failures.append("target has CollisionAPI")
    if not required_root_schemas.issubset(root_schemas):
        failures.append(f"root schemas missing {sorted(required_root_schemas - root_schemas)}")
    if not required_mesh_schemas.issubset(mesh_schemas):
        failures.append(f"mesh schemas missing {sorted(required_mesh_schemas - mesh_schemas)}")
    if len(classifications["proxies"]) != 2:
        failures.append("expected exactly two active proxy Gprims")
    for proxy in classifications["proxies"]:
        if proxy["visible"] != "invisible":
            failures.append(f"proxy is visible: {proxy['path']}")
        if proxy["classification"] != "collision_only_grasp_proxy":
            failures.append(f"proxy classification: {proxy['path']}")
        if proxy["visual_contract"] != "computed_invisible":
            failures.append(f"proxy visual contract: {proxy['path']}")
    for collider in classifications["robot_collision_visibility"]:
        if collider["visible"] != "invisible":
            failures.append(f"robot collision is visible: {collider['path']}")
    if not classifications["robot_collision_visibility"]:
        failures.append("no active robot collision Gprims found")

    # A surface task must not quietly compose Pillow024's volume or a second
    # hidden collision mesh under the task-owned root.
    forbidden_names = [
        str(prim.GetPath())
        for prim in _prim_range(root)
        if "pillow" in str(prim.GetPath()).lower()
        or "simulation_mesh" in str(prim.GetPath()).lower()
    ]
    if forbidden_names:
        failures.append(f"Pillow/volume geometry composed under surface root: {forbidden_names}")

    return {
        "classifications": classifications,
        "root_schemas": sorted(root_schemas),
        "mesh_schemas": sorted(mesh_schemas),
        "forbidden_composed_volume_paths": forbidden_names,
        "robot_collision_gprim_count": len(classifications["robot_collision_visibility"]),
        "pass": not failures,
        "failures": failures,
    }


def _material_audit(stage: Any, assets: Any, collision_source: Mapping[str, Any]) -> dict[str, Any]:
    from pxr import UsdShade

    material_prim = stage.GetPrimAtPath("/World/Looks/BC025TerryCloth")
    mesh_prim = stage.GetPrimAtPath(SURFACE_MESH_PATH)
    if not material_prim.IsValid() or not mesh_prim.IsValid():
        raise RuntimeError("BC025 Terry Cloth material or mesh binding is missing")
    custom = material_prim.GetCustomData()
    expected_hash = str(assets.archive_sha256)
    if custom.get("handson:materialArchiveSha256") != expected_hash:
        raise RuntimeError("BC025 live material archive hash does not match download")
    binding_path = None
    try:
        binding = UsdShade.MaterialBindingAPI(mesh_prim).GetDirectBinding()
        material = binding.GetMaterial()
        if material and material.GetPrim().IsValid():
            binding_path = str(material.GetPath())
    except Exception:
        binding_path = None
    if binding_path != "/World/Looks/BC025TerryCloth":
        raise RuntimeError(f"BC025 mesh material binding drifted: {binding_path!r}")
    return {
        "material_path": str(material_prim.GetPath()),
        "binding_path": binding_path,
        "custom_data": custom,
        "texture_paths": {name: str(path) for name, path in assets.textures.items()},
        "archive_sha256": assets.archive_sha256,
        "collision_source_sha256": assets.collision_source_sha256,
        "collision_source_audit": dict(collision_source),
        "material_only_geometry_source": True,
        "pillow_volume_composed": False,
    }


def _reset_record(raw: Any, observation: Mapping[str, Any], info: Mapping[str, Any], seed: int) -> dict[str, Any]:
    sample = np.asarray(raw._bc025_surface_translation_sample_m, dtype=np.float64).reshape(3).copy()
    initial = np.asarray(raw._e_initial_state, dtype=np.float64).reshape(6).copy()
    target_nodes = np.asarray(raw._e_reset_nodes_np, dtype=np.float64).copy()
    current_nodes = np.asarray(raw._e_nodes_np, dtype=np.float64).copy()
    baseline_nodes = np.asarray(raw._e_base_nodes_np, dtype=np.float64).copy()
    for label, values in (
        ("sample", sample),
        ("initial_state", initial),
        ("target_nodes", target_nodes),
        ("current_nodes", current_nodes),
        ("baseline_nodes", baseline_nodes),
    ):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"BC025 seed {seed} produced non-finite {label}")
    root_translation = np.asarray(_root_translation(raw._stage), dtype=np.float64)
    from handson.sim.isaaclab.b320_towel_scene import B320_TOWEL_PLACEMENT

    expected_root = np.asarray(B320_TOWEL_PLACEMENT, dtype=np.float64).copy()
    expected_root[:2] += sample[:2]
    expected_root[2] += initial[5]
    root_error = float(np.max(np.abs(root_translation - expected_root)))
    expected_centroid = np.mean(baseline_nodes, axis=0) + np.asarray(
        [sample[0], sample[1], initial[5]], dtype=np.float64
    )
    target_centroid = np.mean(target_nodes, axis=0)
    current_centroid = np.mean(current_nodes, axis=0)
    target_centroid_error = float(np.linalg.norm(target_centroid - expected_centroid))
    if root_error > 2.0e-6:
        raise RuntimeError(
            f"BC025 seed {seed} root translation is not synchronized: error={root_error:.8f}"
        )
    if target_centroid_error > 2.0e-4:
        raise RuntimeError(
            f"BC025 seed {seed} reset centroid is not synchronized: error={target_centroid_error:.8f}"
        )
    if observation is not None and not isinstance(observation, Mapping):
        raise RuntimeError(f"BC025 seed {seed} returned a non-mapping observation")
    metrics = dict(getattr(raw, "_b320_towel_metrics", {}) or {})
    return {
        "seed": int(seed),
        "sample_m": sample.tolist(),
        "garment_initial_state": initial.tolist(),
        "root_translation_m": root_translation.tolist(),
        "expected_root_translation_m": expected_root.tolist(),
        "root_translation_max_error_m": root_error,
        "baseline_centroid_m": np.mean(baseline_nodes, axis=0).tolist(),
        "target_reset_centroid_m": target_centroid.tolist(),
        "expected_target_centroid_m": expected_centroid.tolist(),
        "target_centroid_error_m": target_centroid_error,
        "settled_current_centroid_m": current_centroid.tolist(),
        "settled_offset_from_target_m": (current_centroid - target_centroid).tolist(),
        "target_nodes_aabb_m": [*map(list, _surface_aabb(target_nodes))],
        "settled_nodes_aabb_m": [*map(list, _surface_aabb(current_nodes))],
        "settle": {
            "converged": bool(getattr(raw, "_e_reset_settle_converged", False)),
            "timed_out": bool(getattr(raw, "_e_reset_settle_timed_out", False)),
            "steps": int(getattr(raw, "_e_reset_settle_steps", 0)),
            "seconds": float(getattr(raw, "_e_reset_settle_seconds", 0.0)),
            "rms_speed_mps": float(getattr(raw, "_e_reset_settle_rms_speed", 0.0)),
            "max_speed_mps": float(getattr(raw, "_e_reset_settle_max_speed", 0.0)),
            "window_drift_m": float(getattr(raw, "_e_reset_settle_window_drift", 0.0)),
            "failure_reason": getattr(raw, "_e_failure_reason", None),
        },
        "reset_info_summary": {
            "succeed": bool(info.get("succeed", False)),
            "physical_randomization": bool(info.get("physical_randomization", False)),
            "sample_from_info": info.get("b320_surface_translation_sample_m"),
        },
        "metrics": metrics,
    }


def _frame_metrics(path: Path, frame: Any, *, expected_shape: tuple[int, int, int]) -> dict[str, Any]:
    array = np.asarray(frame)
    if array.dtype != np.uint8 or array.shape != expected_shape:
        raise RuntimeError(
            f"BC025 camera frame {path.name} has shape/dtype {array.shape}/{array.dtype}, expected {expected_shape}/uint8"
        )
    try:
        import imageio.v3 as iio

        iio.imwrite(path, array)
    except Exception:
        from PIL import Image

        Image.fromarray(array).save(path)
    rgb = array[..., :3]
    luma = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )
    quantized = (rgb // 8).reshape(-1, 3)
    _colors, counts = np.unique(quantized, axis=0, return_counts=True)
    probabilities = counts.astype(np.float64) / max(float(np.sum(counts)), 1.0)
    return {
        "path": str(path),
        "shape": list(array.shape),
        "mean_rgb": float(np.mean(rgb)),
        "black_fraction": float(np.mean(np.all(rgb == 0, axis=-1))),
        "unique_rgb_count": int(np.unique(rgb.reshape(-1, 3), axis=0).shape[0]),
        "coarse_rgb_count": int(len(counts)),
        "coarse_rgb_entropy_bits": float(-np.sum(probabilities * np.log2(np.maximum(probabilities, 1.0e-12)))),
        "dominant_coarse_rgb_fraction": float(np.max(probabilities)),
        "channel_std": [float(value) for value in np.std(rgb, axis=(0, 1))],
        "luma_std": float(np.std(luma)),
        "luma_p95_minus_p05": float(np.percentile(luma, 95) - np.percentile(luma, 5)),
        "horizontal_edge_fraction": float(np.mean(np.abs(np.diff(luma, axis=1)) > 4.0)),
        "vertical_edge_fraction": float(np.mean(np.abs(np.diff(luma, axis=0)) > 4.0)),
    }


def _camera_audit(raw: Any, output_dir: Path, *, save_images: bool) -> dict[str, Any]:
    if set(raw._cameras) != EXPECTED_CAMERA_PRIMS:
        raise RuntimeError(
            f"BC025 camera set drifted: {sorted(raw._cameras)} != {sorted(EXPECTED_CAMERA_PRIMS)}"
        )
    # BC025 intentionally uses the fresh-process deferred camera contract.
    # E01 still creates real RTX products in this mode; forcing the immediate
    # post-render path is an independently known CUDA-700 failure mode.
    if not bool(raw._deferred_camera_render):
        raise RuntimeError("BC025 validator expected the deferred camera path")
    frames = raw.render()
    wrist_width, wrist_height = raw._wrist_recording_camera_output_size
    specs = {
        "ego": (frames[0], "ego", (raw._render_height, raw._render_width, 3)),
        "wrist_left": (frames[1], "wrist_left", (wrist_height, wrist_width, 3)),
        "wrist_right": (frames[2], "wrist_right", (wrist_height, wrist_width, 3)),
    }
    result: dict[str, Any] = {
        "deferred_camera_render": True,
        "canonical_camera_postrender_supported": bool(
            getattr(raw, "_canonical_camera_postrender_supported", False)
        ),
        "camera_keys": sorted(raw._cameras),
        "frames": {},
        "human_viewport": {
            "verified": False,
            "status": "pending_headless_live_audit",
            "reason": "validator uses headless=True; operator viewport requires a separate interactive session",
        },
    }
    for name, (frame, fallback_key, shape) in specs.items():
        array = np.asarray(frame)
        fallback = raw._fallback_camera_frame(
            fallback_key,
            height=shape[0],
            width=shape[1],
        )
        fallback_equal = bool(np.array_equal(array[..., :3], fallback[..., :3]))
        if fallback_equal:
            raise RuntimeError(f"BC025 camera {name} returned its known fallback frame")
        if save_images:
            path = output_dir / f"{name}_reset.png"
            stats = _frame_metrics(path, array, expected_shape=shape)
        else:
            stats = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        stats["fallback_equal"] = fallback_equal
        result["frames"][name] = stats
    camera_warnings = [
        warning
        for warning in raw._isaac_warnings
        if "Recording cameras unavailable" in warning or "RGB unavailable" in warning
    ]
    if camera_warnings:
        raise RuntimeError(f"BC025 camera fallback warnings: {camera_warnings}")
    return result


def _state_schema_audit(environment: Any, raw: Any, task_config: Any, observation: Mapping[str, Any]) -> dict[str, Any]:
    raw_observation = raw._compute_observation()
    raw_state = raw_observation.get("state")
    if not isinstance(raw_state, Mapping):
        raise RuntimeError("BC025 raw observation state is not a mapping")
    actual_keys = set(raw_state)
    if actual_keys != EXPECTED_STATE_KEYS:
        raise RuntimeError(
            f"BC025 raw state keys drifted: {sorted(actual_keys)} != {sorted(EXPECTED_STATE_KEYS)}"
        )
    if not environment.observation_space.contains(observation):
        raise RuntimeError("BC025 adapted reset observation violates its observation space")
    expected_size = sum(
        int(np.prod(raw.observation_space["state"][key].shape))
        for key in task_config.proprio_keys
    )
    adapted_state = np.asarray(observation["state"])
    if adapted_state.shape != (expected_size,):
        raise RuntimeError(
            f"BC025 adapted state shape drifted: {adapted_state.shape} != {(expected_size,)}"
        )
    return {
        "raw_state_keys": sorted(actual_keys),
        "proprio_keys": list(task_config.proprio_keys),
        "adapted_state_shape": list(adapted_state.shape),
        "observation_space_contains": True,
    }


def _finite_observation(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_observation(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_observation(item) for item in value)
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            return bool(np.all(np.isfinite(value)))
        return True
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _zero_action_audit(environment: Any, raw: Any, steps: int) -> dict[str, Any]:
    action = np.zeros(environment.action_space.shape, dtype=np.float32)
    before_nodes = np.asarray(raw._e_nodes_np, dtype=np.float64).copy()
    before_palms = {
        side: np.asarray(raw._detached_sim_pose[side], dtype=np.float64).copy()
        for side in ("left", "right")
    }
    samples: list[dict[str, Any]] = []
    final_info: Mapping[str, Any] = {}
    for step_index in range(steps):
        observation, reward, terminated, truncated, final_info = environment.step(action)
        if not environment.observation_space.contains(observation):
            raise RuntimeError(f"BC025 zero-action observation failed at step {step_index}")
        if not _finite_observation(observation):
            raise RuntimeError(f"BC025 zero-action observation was non-finite at step {step_index}")
        if abs(float(reward)) > 1.0e-9 or bool(final_info.get("succeed", False)):
            raise RuntimeError(f"BC025 zero action produced false success at step {step_index}: {final_info}")
        if terminated or truncated:
            raise RuntimeError(f"BC025 zero action terminated at step {step_index}: {final_info}")
        nodes = np.asarray(raw._e_nodes_np, dtype=np.float64)
        velocities = np.asarray(raw._e_velocities_np, dtype=np.float64)
        if nodes.shape != (1681, 3) or velocities.shape != (1681, 3):
            raise RuntimeError(f"BC025 zero-action topology changed at step {step_index}")
        if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(velocities)):
            raise RuntimeError(f"BC025 zero-action PhysX nodal state was non-finite at step {step_index}")
        samples.append(
            {
                "step": step_index + 1,
                "max_nodal_speed_mps": float(np.max(np.linalg.norm(velocities, axis=1))),
                "max_node_delta_from_previous_control_frame_m": float(
                    np.max(np.linalg.norm(nodes - before_nodes, axis=1))
                ),
                "failure_reason": getattr(raw, "_e_failure_reason", None),
            }
        )
        before_nodes = nodes.copy()
    after_nodes = np.asarray(raw._e_nodes_np, dtype=np.float64)
    surface_drift = float(np.max(np.linalg.norm(after_nodes - np.asarray(raw._e_reset_nodes_np), axis=1)))
    palm_drift = {
        side: float(
            np.linalg.norm(
                np.asarray(raw._detached_sim_pose[side], dtype=np.float64)[:3]
                - before_palms[side][:3]
            )
        )
        for side in ("left", "right")
    }
    max_speed = max((sample["max_nodal_speed_mps"] for sample in samples), default=0.0)
    if surface_drift > ZERO_ACTION_SURFACE_DRIFT_LIMIT_M:
        raise RuntimeError(f"BC025 zero-action surface drift exceeds limit: {surface_drift:.6f} m")
    if any(value > ZERO_ACTION_PALM_DRIFT_LIMIT_M for value in palm_drift.values()):
        raise RuntimeError(f"BC025 zero-action palm drift exceeds limit: {palm_drift}")
    if getattr(raw, "_e_failure_reason", None) is not None:
        raise RuntimeError(f"BC025 zero-action left a latched garment failure: {raw._e_failure_reason}")
    return {
        "steps": int(steps),
        "samples": samples,
        "max_nodal_speed_mps": max_speed,
        "surface_drift_from_reset_target_m": surface_drift,
        "palm_drift_m": palm_drift,
        "final_info": dict(final_info),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.steps < 1 or args.steps > 300:
        raise ValueError("--steps must be in [1, 300]")
    if args.reset_count < 1 or args.reset_count > 32:
        raise ValueError("--reset-count must be in [1, 32]")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # BC025's qualified live path is the deferred-camera E01 path.  In
    # particular, do not opt into the immediate post-render branch here.
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    os.environ["HANDSON_STABLE_RECORDING_ORIGIN"] = "1"
    os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = (
        "0" if args.disable_ground_truth_recording else "1"
    )
    os.environ["HANDSON_ISAACLAB_DEFER_ARM_RENDER"] = "0"
    os.environ["HANDSON_ISAACLAB_DIRECT_HAND_CONTROL"] = "0"
    os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "1"
    os.environ["HANDSON_ISAACSIM_ACCURATE_CAMERA_VISIBILITY"] = "0"
    os.environ.pop("HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY", None)

    audit = _AuditState(checks={}, errors=[], warnings=[])
    environment: Any | None = None
    raw: Any | None = None
    result: dict[str, Any] = {
        "schema": "handson_bc025_live_validation_v1",
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "seed": int(args.seed),
        "reset_count": int(args.reset_count),
        "randomize": bool(args.randomize),
        "steps": int(args.steps),
        "camera_mode": "deferred_qualified_path",
        "formal_collection_certified": False,
        "physical_replay_certified": False,
        "replay_scope": "diagnostic_state_guided",
        "collection_policy_changed": False,
        "physx_ground_truth_recording_requested": not bool(
            args.disable_ground_truth_recording
        ),
    }
    exit_code = 0
    try:
        _phase("import_task_and_asset_helpers")
        from handson.tasks.b320_towel_corner_synchronized_spread.config import TaskConfig
        from handson.sim.isaaclab.b320_bc025_assets import (
            bc025_static_asset_manifest,
            inspect_pillow024_collision_source,
            resolve_bc025_assets,
        )
        from handson.sim.isaaclab.b320_towel_scene import (
            BC025_PILLOW_COLLISION_CALIBRATION,
            BC025_TOWEL_TRANSLATION_BOUNDS_M,
        )

        _phase("runtime_contract")
        runtime = _runtime_contract()
        result["runtime"] = runtime
        _phase("resolve_assets")
        assets = resolve_bc025_assets(required=True)
        if assets is None:
            raise RuntimeError("BC025 selected assets could not be resolved")
        # The standalone Isaac Sim interpreter does not expose the pxr module
        # until the Kit/SimulationContext has been launched.  Keep archive
        # hashing and manifest capture before launch, but defer opening the
        # Pillow USD until after the live environment has initialized.
        source_audit: dict[str, Any] | None = None
        static_manifest = bc025_static_asset_manifest()
        result["asset_manifest"] = assets.manifest()
        result["static_asset_manifest"] = static_manifest
        result["declared_translation_bounds_m"] = [
            [float(value) for value in bounds]
            for bounds in BC025_TOWEL_TRANSLATION_BOUNDS_M
        ]

        _phase("construct_environment")
        task_config = TaskConfig()
        environment = task_config.get_native_environment(
            policy_mode=True,
            sim_backend="isaacsim",
            launch_isaac=True,
            image_obs=bool(args.images),
            randomize=bool(args.randomize),
            randomize_dynamics=False,
            render_mode="rgb_array",
            headless=True,
            isaac_device=args.device,
            hz=30,
            control_dt=1.0 / 30.0,
            physics_dt=1.0 / 120.0,
            time_limit=float(task_config.time_limit_seconds),
        )
        _phase("environment_ready")
        raw = _unwrap(environment)
        raw._fast_shutdown_on_close = True
        if str(raw._physics_backend).lower() != "physx":
            raise RuntimeError(f"BC025 requires PhysX, got {raw._physics_backend!r}")
        if not np.isclose(float(raw._sim_dt), 1.0 / 120.0, rtol=0.0, atol=1.0e-9):
            raise RuntimeError(f"BC025 physics dt drifted: {raw._sim_dt}")
        result["physics_backend"] = str(raw._physics_backend)
        result["physics_dt"] = float(raw._sim_dt)
        result["physics_hz"] = float(1.0 / raw._sim_dt)
        result["physx_ground_truth_enabled"] = bool(
            getattr(raw, "_physx_ground_truth_enabled", False)
        )

        stage = raw._stage
        if stage is None:
            raise RuntimeError("BC025 Isaac stage is unavailable")
        _phase("audit_pillow_usd")
        source_audit = inspect_pillow024_collision_source(assets.collision_source_path)
        result["pillow024_collision_audit"] = source_audit
        audit.check(
            "assets",
            assets.archive_sha256 == static_manifest["material"]["archive_sha256"]
            and assets.collision_source_sha256 == static_manifest["collision_source"]["sha256"],
            {
                "material_archive_sha256": assets.archive_sha256,
                "collision_source_sha256": assets.collision_source_sha256,
                "pillow_calibration": source_audit,
            },
        )
        _phase("audit_scene_schema")
        mesh_contract = _mesh_contract(stage)
        classification = _scene_classification_audit(stage, raw)
        material = _material_audit(stage, assets, source_audit)
        result["mesh_contract"] = mesh_contract
        result["scene_classification"] = classification
        result["material_audit"] = material
        audit.check("scene_classification", bool(classification["pass"]), classification)
        audit.check("topology", mesh_contract["authored_points"] == 1681 and mesh_contract["authored_faces"] == 3200 and mesh_contract["physx_rest_triangles"] == 3200, mesh_contract)

        observation: Mapping[str, Any]
        info: Mapping[str, Any]
        reset_records: list[dict[str, Any]] = []
        _phase("reset_randomization")
        for reset_index in range(args.reset_count):
            seed = int(args.seed + reset_index)
            observation_value, info_value = environment.reset(seed=seed)
            observation = observation_value
            info = info_value
            reset_records.append(_reset_record(raw, observation, info, seed))
            if raw._e_failure_reason is not None:
                raise RuntimeError(
                    f"BC025 seed {seed} reset failed: {raw._e_failure_reason}; {raw._e_failure_diagnostics}"
                )
            if not environment.observation_space.contains(observation):
                raise RuntimeError(f"BC025 seed {seed} observation violates its space")

        first_sample = np.asarray(reset_records[0]["sample_m"], dtype=np.float64)
        repeat_observation, repeat_info = environment.reset(seed=int(args.seed))
        repeat_record = _reset_record(raw, repeat_observation, repeat_info, int(args.seed))
        repeat_sample = np.asarray(repeat_record["sample_m"], dtype=np.float64)
        same_seed_equal = bool(np.array_equal(first_sample, repeat_sample))
        sample_matrix = np.asarray([record["sample_m"] for record in reset_records], dtype=np.float64)
        span = np.ptp(sample_matrix, axis=0)
        if args.randomize and not same_seed_equal:
            raise RuntimeError("BC025 same-seed reset sample is not repeatable")
        if args.randomize and (float(span[0]) <= 1.0e-9 or float(span[1]) <= 1.0e-9):
            raise RuntimeError(f"BC025 different seeds did not span both XY axes: {span.tolist()}")
        reset_result = {
            "records": reset_records,
            "repeat_seed_record": repeat_record,
            "same_seed_sample_equal": same_seed_equal,
            "different_seed_translation_span_m": span.tolist(),
            "sample_count": len(reset_records),
            "randomization_enabled": bool(args.randomize),
        }
        result["reset"] = reset_result
        audit.check("seeded_reset", (not args.randomize) or (same_seed_equal and float(span[0]) > 1.0e-9 and float(span[1]) > 1.0e-9), reset_result)

        _phase("audit_state_and_collision")
        result["state_schema"] = _state_schema_audit(environment, raw, task_config, repeat_observation)
        audit.check("state_schema", True, result["state_schema"])

        current_nodes = np.asarray(raw._e_nodes_np, dtype=np.float64)
        settled_inventory = _collision_inventory(stage, surface_nodes=current_nodes)
        authored_inventory = _collision_inventory(
            stage,
            surface_nodes=np.asarray(raw._e_reset_nodes_np, dtype=np.float64),
        )
        result["collision_audit"] = {
            "authored_first_physx_surface_target": authored_inventory,
            "settled_physx_surface_readback": settled_inventory,
            "report_scope": "surface uses raw PhysX nodal readback; rigid bounds use USD BBoxCache; positive AABB overlap is broad-phase evidence",
            "rigid_deformable_manifold_report": "unavailable_in_isaacsim_5_1_public_view",
        }
        collision_ok = (
            bool(authored_inventory["formal_unapproved_overlap_gate"])
            and bool(settled_inventory["formal_unapproved_overlap_gate"])
            and not authored_inventory["non_hand_robot_colliders"]
            and not settled_inventory["non_hand_robot_colliders"]
        )
        audit.check("collision_overlap", collision_ok, result["collision_audit"])
        robot_counts = settled_inventory["counts"]
        complete_hands = bool(
            robot_counts["robot_left"] >= 20
            and robot_counts["robot_right"] >= 20
        )
        audit.check(
            "palm_finger_collision_coverage",
            complete_hands,
            {
                "counts": robot_counts,
                "requirement": "at least 20 active collision Gprims per complete palm/finger hand; all must be invisible",
            },
        )

        # A static reachability record is useful, but it is not a substitute
        # for an executed hand-to-corner trajectory.  Keep that gate pending.
        assigned_nodes = np.asarray(raw._e_nodes_np, dtype=np.float64)[[1640, 1680]]
        proxy_distances = {
            side: float(
                np.linalg.norm(
                    np.asarray(raw._contact_proxy_last_pose[side], dtype=np.float64)[:3]
                    - assigned_nodes[index]
                )
            )
            for index, side in enumerate(("left", "right"))
        }
        result["reachability"] = {
            "assigned_corner_nodes": {"left": 1640, "right": 1680},
            "assigned_corner_positions_m": {"left": assigned_nodes[0].tolist(), "right": assigned_nodes[1].tolist()},
            "home_proxy_distance_m": proxy_distances,
            "status": "pending_dynamic_hand_trajectory_probe",
        }

        _phase("zero_action")
        result["zero_action"] = _zero_action_audit(environment, raw, args.steps)
        audit.check("zero_action_stability", True, result["zero_action"])

        if args.images:
            _phase("deferred_cameras")
            result["cameras"] = _camera_audit(raw, output_dir, save_images=True)
            audit.check("cameras", True, result["cameras"])
        else:
            result["cameras"] = {
                "status": "not_requested",
                "deferred_camera_render": bool(raw._deferred_camera_render),
                "human_viewport": {"verified": False, "status": "pending"},
            }

        result["contact"] = {
            "status": "pending_physical_contact_probe",
            "evidence_source": "localized_surface_nodal_response_after_proxy_approach",
            "report_scope": "deformable_nodal_readback; no rigid contact manifold in Isaac Sim 5.1 public view",
            "observed_at_reset": dict(raw._bc025_physical_contact_observed),
            "armed_at_reset": dict(raw._bc025_physical_contact_armed),
            "thresholds": dict(raw._task_info().get("b320_contact_response_thresholds", {})),
        }
        result["success"] = {
            "status": "pending_complete_live_episode",
            "success_at_reset": bool(raw._metrics(update_events=False).get("success_now", False)),
            "formal_positive_trials": 0,
            "formal_negative_trials": 0,
        }
        result["stability"] = {
            "status": "pass",
            "zero_action_steps": int(args.steps),
            "reset_settle": [record["settle"] for record in reset_records],
        }
        result["collection"] = {
            "status": "pending_physical_replay_and_recording_gate",
            "collection_policy_changed": False,
            "physical_replay_certified": False,
            "replay_scope": "diagnostic_state_guided",
        }
        result["task_info"] = raw._task_info()
        result["physical_replay_reset_contract"] = raw.get_physical_replay_reset_contract()
        _phase("audit_complete")
    except Exception as exc:
        exit_code = 1
        audit.errors.append(f"exception: {exc}")
        result["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        traceback.print_exc()
    finally:
        result["checks"] = audit.checks
        result["errors"] = audit.errors
        result["warnings"] = audit.warnings
        result["validation_pass"] = not audit.errors
        result["pending_gates"] = [
            "dynamic_hand_reachability_trajectory",
            "physical_assigned_corner_contact_probe",
            "complete_positive_and_negative_success_trials",
            "human_viewport_interactive_visibility",
            "formal_recording_and_physical_surface_replay",
        ]
        result_path = output_dir / "validation.json"
        result_path.write_text(
            json.dumps(_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(f"VALIDATION_JSON={result_path}", flush=True)
        if raw is not None:
            active_exception = sys.exc_info()
            close_exit_code = 0 if exit_code == 0 and active_exception[0] is None else 1
            shutdown_guard = threading.Timer(15.0, lambda: os._exit(close_exit_code))
            shutdown_guard.daemon = True
            shutdown_guard.start()
            try:
                environment.close()
            finally:
                shutdown_guard.cancel()
    if exit_code == 0:
        print(
            "BC025_LIVE_AUDIT_COMPLETED_WITH_PENDING_GATES "
            f"steps={args.steps} randomize={bool(args.randomize)} images={bool(args.images)}",
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
