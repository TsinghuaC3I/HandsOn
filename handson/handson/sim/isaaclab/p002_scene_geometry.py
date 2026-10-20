"""P0-02-only table removal and Helindu front-handle replacement.

The Door042 articulation, door panel, Handle002 rigid body, revolute joint,
mass, and task-facing prim paths remain unchanged. Only the old Handle002
visual/collision payload is disabled and replaced below the same body. The
fixed rosette is referenced below the door panel and has no collision.

This module deliberately imports pxr only inside the stage mutation function
so its constants and evidence helpers remain CPU-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any


P002_TASK_NAME = "t17_open_latched_door"

P002_TABLE_ROOT_PATH = "/World/Task/table"
P002_TABLE_BODY_PATH = f"{P002_TABLE_ROOT_PATH}/table"
P002_TABLE_COLLISIONS_PATH = f"{P002_TABLE_BODY_PATH}/collisions"

P002_DOOR_ROOT_PATH = "/World/Task/P002Door"
P002_DOOR_BODY_PATH = f"{P002_DOOR_ROOT_PATH}/Door042_Door"
P002_FRONT_HANDLE_BODY_PATH = f"{P002_DOOR_ROOT_PATH}/Door042_Handle002"
P002_FRONT_HANDLE_JOINT_PATH = (
    f"{P002_FRONT_HANDLE_BODY_PATH}/Handle002_joint"
)
P002_OLD_FRONT_HANDLE_VISUALS_PATH = f"{P002_FRONT_HANDLE_BODY_PATH}/Visuals"
P002_OLD_FRONT_HANDLE_COLLISIONS_PATH = (
    f"{P002_FRONT_HANDLE_BODY_PATH}/Collisions"
)
P002_ROSETTE_PATH = f"{P002_DOOR_BODY_PATH}/P002HelinduRosette"
P002_LEVER_PATH = f"{P002_FRONT_HANDLE_BODY_PATH}/P002HelinduLever"

P002_FRONT_HANDLE_JOINT_LOCAL_POS0_M = (
    -0.6834181,
    0.06933168,
    0.0491462,
)
P002_FRONT_HANDLE_JOINT_LOCAL_POS1_M = (0.0, 0.0, 0.0)
P002_FRONT_HANDLE_SOURCE_ARCHIVE_SHA256 = (
    "330c5bf51744ae54468c55896280d10d6c5ef0d7b63783112fe8052ecf36affd"
)
P002_FRONT_HANDLE_ROSETTE_SHA256 = (
    "31bbcf68fb22601d3b7006bd62be4ad0d487a2f343c5620f493ff2e8c256af9c"
)
P002_FRONT_HANDLE_LEVER_SHA256 = (
    "53c666d6f38ffc0654c892827e837b4e426041aa25f207e59d204107e8e98037"
)
P002_FRONT_HANDLE_DIMENSIONS_M = (
    0.1500000000000000,
    0.04148298257902923,
    0.04849960621353222,
)
P002_FRONT_HANDLE_MIN_CLEARANCE_M = 0.009719147885820635
P002_FRONT_HANDLE_GRASP_X_RANGE_M = (0.055, 0.105)
P002_FRONT_HANDLE_GRASP_CLEARANCE_M = 0.03002410272964051
P002_FRONT_HANDLE_EXPECTED_OLD_VISUAL_MESHES = 1
P002_FRONT_HANDLE_EXPECTED_OLD_COLLIDERS = 4
P002_FRONT_HANDLE_EXPECTED_REPLACEMENT_VISUAL_MESHES = 2
P002_FRONT_HANDLE_EXPECTED_REPLACEMENT_COLLIDERS = 4
P002_FRONT_HANDLE_MAX_GPU_CONVEX_VERTICES = 8

_RUNTIME_DIR = (
    Path(__file__).resolve().parent
    / "usd/t17_open_latched_door/helindu_door_handle"
)
P002_FRONT_HANDLE_ROSETTE_USDA = _RUNTIME_DIR / "helindu_rosette.usda"
P002_FRONT_HANDLE_LEVER_USDA = _RUNTIME_DIR / "helindu_lever.usda"

_TABLE_INSTANCE_DISABLED_MARKER = "handson:p002CollisionInstanceDisabled"
_TABLE_INSTANCE_DISABLED_COUNT = "handson:p002DisabledColliderCount"
_REPLACEMENT_MARKER = "handson:p002HelinduHandleReplacement"
_REPLACEMENT_VERSION = 1
_TOLERANCE = 2.0e-7


@dataclass(frozen=True)
class P002SceneGeometryReport:
    """Auditable result of the P0-02 task-scoped geometry mutation."""

    table_hidden: bool
    table_colliders_disabled: int
    table_collision_instance_deactivated: bool
    old_handle_visual_meshes_hidden: int
    old_handle_colliders_disabled: int
    replacement_visual_meshes: int
    replacement_colliders: int
    replacement_max_convex_vertices: int
    replacement_collision_primitive: str
    replacement_asset_title: str
    replacement_asset_author: str
    replacement_asset_license: str
    replacement_source_archive_sha256: str
    front_handle_dimensions_m: tuple[float, float, float]
    front_handle_min_clearance_m: float
    front_handle_grasp_x_range_m: tuple[float, float]
    front_handle_grasp_clearance_m: float
    front_handle_body_scale: tuple[float, float, float]
    front_handle_mass_kg: float | None
    front_handle_explicit_inertia_authored: bool
    front_handle_joint_local_pos0_m: tuple[float, float, float]
    front_handle_joint_local_pos1_m: tuple[float, float, float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid(value: Any) -> bool:
    return value is not None and bool(value)


def _tuple3(value: Any, *, label: str) -> tuple[float, float, float]:
    if value is None or len(value) != 3:
        raise RuntimeError(f"P0-02 {label} is missing or not a vec3")
    result = tuple(float(component) for component in value)
    if not all(abs(component) < float("inf") for component in result):
        raise RuntimeError(f"P0-02 {label} is non-finite")
    return result  # type: ignore[return-value]


def _require_vec3(
    actual: tuple[float, float, float],
    expected: tuple[float, float, float],
    *,
    label: str,
) -> None:
    if any(abs(left - right) > _TOLERANCE for left, right in zip(actual, expected)):
        raise RuntimeError(
            f"P0-02 {label} drifted: expected {expected!r}, got {actual!r}"
        )


def _prim_scale(prim: Any, *, label: str) -> tuple[float, float, float]:
    attr = prim.GetAttribute("xformOp:scale")
    value = attr.Get() if _valid(attr) else None
    return (1.0, 1.0, 1.0) if value is None else _tuple3(value, label=label)


def _relationship_targets(relationship: Any) -> tuple[str, ...]:
    return () if relationship is None else tuple(
        str(target) for target in relationship.GetTargets()
    )


def p002_front_handle_dimensions_m() -> tuple[float, float, float]:
    return P002_FRONT_HANDLE_DIMENSIONS_M


def _validate_runtime_assets() -> None:
    expected = (
        (P002_FRONT_HANDLE_ROSETTE_USDA, P002_FRONT_HANDLE_ROSETTE_SHA256),
        (P002_FRONT_HANDLE_LEVER_USDA, P002_FRONT_HANDLE_LEVER_SHA256),
    )
    for path, digest in expected:
        if not path.is_file():
            raise FileNotFoundError(f"P0-02 replacement handle asset is missing: {path}")
        actual = _sha256(path)
        if actual != digest:
            raise RuntimeError(
                f"P0-02 replacement handle SHA mismatch for {path.name}: "
                f"expected {digest}, got {actual}"
            )


def _table_collision_inventory(
    table_collisions: Any,
    *,
    usd: Any,
    usd_physics: Any,
) -> tuple[int, bool, tuple[Any, ...]]:
    if not table_collisions.IsActive():
        marker = table_collisions.GetCustomDataByKey(_TABLE_INSTANCE_DISABLED_MARKER)
        count = table_collisions.GetCustomDataByKey(_TABLE_INSTANCE_DISABLED_COUNT)
        if marker is not True or not isinstance(count, int) or count < 1:
            raise RuntimeError(
                "P0-02 table collision group is inactive without its audit marker"
            )
        return int(count), True, ()
    if table_collisions.IsInstance():
        prototype = table_collisions.GetPrototype()
        if not _valid(prototype):
            raise RuntimeError("P0-02 table collision instance has no prototype")
        colliders = tuple(
            prim
            for prim in usd.PrimRange(prototype)
            if prim.HasAPI(usd_physics.CollisionAPI)
        )
        if not colliders:
            raise RuntimeError("P0-02 table prototype has no collision shapes")
        return len(colliders), True, ()
    colliders = tuple(
        prim
        for prim in usd.PrimRange(table_collisions)
        if prim.HasAPI(usd_physics.CollisionAPI)
    )
    if not colliders:
        raise RuntimeError("P0-02 table has no collision shapes to disable")
    return len(colliders), False, colliders


def _add_asset_reference(
    stage: Any,
    path: str,
    asset_path: Path,
    *,
    usd_geom: Any,
) -> Any:
    root = usd_geom.Xform.Define(stage, path).GetPrim()
    root.GetReferences().ClearReferences()
    root.GetReferences().AddReference(str(asset_path.resolve()))
    root.SetCustomDataByKey(_REPLACEMENT_MARKER, _REPLACEMENT_VERSION)
    return root


def apply_p002_scene_geometry(stage: Any) -> P002SceneGeometryReport:
    """Install the selected handle before the first PhysX reset.

    The operation is fail-closed and idempotent. All source topology, joint,
    path, body-scale, and runtime-asset checks complete before any mutation.
    """

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    _validate_runtime_assets()
    required_paths = (
        P002_TABLE_ROOT_PATH,
        P002_TABLE_BODY_PATH,
        P002_TABLE_COLLISIONS_PATH,
        P002_DOOR_BODY_PATH,
        P002_FRONT_HANDLE_BODY_PATH,
        P002_FRONT_HANDLE_JOINT_PATH,
        P002_OLD_FRONT_HANDLE_VISUALS_PATH,
        P002_OLD_FRONT_HANDLE_COLLISIONS_PATH,
    )
    prims = {path: stage.GetPrimAtPath(path) for path in required_paths}
    missing = tuple(path for path, prim in prims.items() if not prim.IsValid())
    if missing:
        raise RuntimeError(f"P0-02 scene geometry prims are missing: {missing!r}")

    table_root = prims[P002_TABLE_ROOT_PATH]
    table_body = prims[P002_TABLE_BODY_PATH]
    table_collisions = prims[P002_TABLE_COLLISIONS_PATH]
    if not table_body.IsActive():
        raise RuntimeError("P0-02 compatibility table body is inactive")
    table_imageable = UsdGeom.Imageable(table_root)
    if not table_imageable:
        raise RuntimeError("P0-02 compatibility table is not imageable")
    table_count, table_instance, table_collision_prims = (
        _table_collision_inventory(
            table_collisions,
            usd=Usd,
            usd_physics=UsdPhysics,
        )
    )

    handle_body = prims[P002_FRONT_HANDLE_BODY_PATH]
    handle_visuals = prims[P002_OLD_FRONT_HANDLE_VISUALS_PATH]
    handle_collisions = prims[P002_OLD_FRONT_HANDLE_COLLISIONS_PATH]
    joint_prim = prims[P002_FRONT_HANDLE_JOINT_PATH]
    if not joint_prim.IsA(UsdPhysics.RevoluteJoint):
        raise RuntimeError("P0-02 Handle002_joint is not a revolute joint")
    old_visuals = tuple(
        prim for prim in Usd.PrimRange(handle_visuals) if prim.IsA(UsdGeom.Mesh)
    )
    old_colliders = tuple(
        prim
        for prim in Usd.PrimRange(handle_collisions)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    )
    if len(old_visuals) != P002_FRONT_HANDLE_EXPECTED_OLD_VISUAL_MESHES:
        raise RuntimeError(
            "P0-02 old Handle002 visual topology drifted: "
            f"expected 1, got {len(old_visuals)}"
        )
    if len(old_colliders) != P002_FRONT_HANDLE_EXPECTED_OLD_COLLIDERS:
        raise RuntimeError(
            "P0-02 old Handle002 collision topology drifted: "
            f"expected 4, got {len(old_colliders)}"
        )

    body_scale = _prim_scale(handle_body, label="Handle002 body scale")
    _require_vec3(body_scale, (1.0, 1.0, 1.0), label="Handle002 body scale")
    joint = UsdPhysics.RevoluteJoint(joint_prim)
    if _relationship_targets(joint.GetBody0Rel()) != (P002_DOOR_BODY_PATH,):
        raise RuntimeError("P0-02 Handle002_joint body0 drifted")
    if _relationship_targets(joint.GetBody1Rel()) != (P002_FRONT_HANDLE_BODY_PATH,):
        raise RuntimeError("P0-02 Handle002_joint body1 drifted")
    local_pos0 = _tuple3(
        joint.GetLocalPos0Attr().Get(), label="Handle002_joint localPos0"
    )
    local_pos1 = _tuple3(
        joint.GetLocalPos1Attr().Get(), label="Handle002_joint localPos1"
    )
    _require_vec3(
        local_pos0,
        P002_FRONT_HANDLE_JOINT_LOCAL_POS0_M,
        label="Handle002_joint localPos0",
    )
    _require_vec3(
        local_pos1,
        P002_FRONT_HANDLE_JOINT_LOCAL_POS1_M,
        label="Handle002_joint localPos1",
    )
    mass_api = UsdPhysics.MassAPI(handle_body)
    mass_attr = mass_api.GetMassAttr()
    mass_value = mass_attr.Get() if _valid(mass_attr) else None
    mass_kg = None if mass_value is None else float(mass_value)
    explicit_inertia = any(
        _valid(attr) and attr.HasAuthoredValueOpinion()
        for attr in (
            mass_api.GetCenterOfMassAttr(),
            mass_api.GetDiagonalInertiaAttr(),
            mass_api.GetPrincipalAxesAttr(),
        )
    )

    # Mutations begin only after all preflight checks above have passed.
    table_imageable.MakeInvisible()
    if table_instance:
        if table_collisions.IsActive():
            table_collisions.SetCustomDataByKey(_TABLE_INSTANCE_DISABLED_MARKER, True)
            table_collisions.SetCustomDataByKey(_TABLE_INSTANCE_DISABLED_COUNT, table_count)
            if table_collisions.SetActive(False) is False:
                raise RuntimeError("P0-02 could not deactivate table collision instance")
    else:
        for prim in table_collision_prims:
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False).Set(False)

    UsdGeom.Imageable(handle_visuals).MakeInvisible()
    for prim in old_colliders:
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False).Set(False)

    rosette_root = _add_asset_reference(
        stage,
        P002_ROSETTE_PATH,
        P002_FRONT_HANDLE_ROSETTE_USDA,
        usd_geom=UsdGeom,
    )
    rosette_xform = UsdGeom.Xformable(rosette_root)
    rosette_xform.ClearXformOpOrder()
    rosette_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*P002_FRONT_HANDLE_JOINT_LOCAL_POS0_M)
    )
    lever_root = _add_asset_reference(
        stage,
        P002_LEVER_PATH,
        P002_FRONT_HANDLE_LEVER_USDA,
        usd_geom=UsdGeom,
    )
    UsdGeom.Xformable(lever_root).ClearXformOpOrder()

    replacement_roots = (rosette_root, lever_root)
    replacement_visuals = tuple(
        prim
        for root in replacement_roots
        for prim in Usd.PrimRange(root)
        if prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.CollisionAPI)
    )
    replacement_colliders = tuple(
        prim
        for root in replacement_roots
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    )
    nested_bodies = tuple(
        str(prim.GetPath())
        for root in replacement_roots
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    )
    if nested_bodies:
        raise RuntimeError(
            f"P0-02 replacement introduced nested rigid bodies: {nested_bodies!r}"
        )
    if len(replacement_visuals) != P002_FRONT_HANDLE_EXPECTED_REPLACEMENT_VISUAL_MESHES:
        raise RuntimeError(
            "P0-02 replacement visual topology drifted: "
            f"expected 2, got {len(replacement_visuals)}"
        )
    if len(replacement_colliders) != P002_FRONT_HANDLE_EXPECTED_REPLACEMENT_COLLIDERS:
        raise RuntimeError(
            "P0-02 replacement collision topology drifted: "
            f"expected 4, got {len(replacement_colliders)}"
        )
    if UsdGeom.Imageable(handle_visuals).ComputeVisibility() != "invisible":
        raise RuntimeError("P0-02 old Handle002 visual remained visible")
    for prim in old_colliders:
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
            raise RuntimeError("P0-02 old Handle002 collision remained enabled")
    replacement_vertex_counts: list[int] = []
    for prim in replacement_colliders:
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True:
            raise RuntimeError("P0-02 replacement collision is not enabled")
        if not prim.IsA(UsdGeom.Cube):
            raise RuntimeError(
                "P0-02 replacement collision must be a GPU-safe Cube primitive"
            )
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            raise RuntimeError(
                "P0-02 replacement collision unexpectedly requires mesh cooking"
            )
        replacement_vertex_counts.append(8)
    if not replacement_vertex_counts or max(replacement_vertex_counts) > (
        P002_FRONT_HANDLE_MAX_GPU_CONVEX_VERTICES
    ):
        raise RuntimeError(
            "P0-02 replacement exceeds the GPU convex vertex budget: "
            f"{replacement_vertex_counts!r}"
        )
    _require_vec3(
        _prim_scale(handle_body, label="Handle002 body scale after replacement"),
        body_scale,
        label="Handle002 body scale after replacement",
    )

    return P002SceneGeometryReport(
        table_hidden=True,
        table_colliders_disabled=table_count,
        table_collision_instance_deactivated=table_instance,
        old_handle_visual_meshes_hidden=len(old_visuals),
        old_handle_colliders_disabled=len(old_colliders),
        replacement_visual_meshes=len(replacement_visuals),
        replacement_colliders=len(replacement_colliders),
        replacement_max_convex_vertices=max(replacement_vertex_counts),
        replacement_collision_primitive="Cube",
        replacement_asset_title="Door Handle",
        replacement_asset_author="Helindu",
        replacement_asset_license="CC-BY-4.0",
        replacement_source_archive_sha256=P002_FRONT_HANDLE_SOURCE_ARCHIVE_SHA256,
        front_handle_dimensions_m=P002_FRONT_HANDLE_DIMENSIONS_M,
        front_handle_min_clearance_m=P002_FRONT_HANDLE_MIN_CLEARANCE_M,
        front_handle_grasp_x_range_m=P002_FRONT_HANDLE_GRASP_X_RANGE_M,
        front_handle_grasp_clearance_m=P002_FRONT_HANDLE_GRASP_CLEARANCE_M,
        front_handle_body_scale=body_scale,
        front_handle_mass_kg=mass_kg,
        front_handle_explicit_inertia_authored=explicit_inertia,
        front_handle_joint_local_pos0_m=local_pos0,
        front_handle_joint_local_pos1_m=local_pos1,
    )


__all__ = [
    "P002_FRONT_HANDLE_BODY_PATH",
    "P002_FRONT_HANDLE_DIMENSIONS_M",
    "P002_FRONT_HANDLE_EXPECTED_REPLACEMENT_COLLIDERS",
    "P002_FRONT_HANDLE_MAX_GPU_CONVEX_VERTICES",
    "P002_FRONT_HANDLE_EXPECTED_REPLACEMENT_VISUAL_MESHES",
    "P002_FRONT_HANDLE_GRASP_CLEARANCE_M",
    "P002_FRONT_HANDLE_GRASP_X_RANGE_M",
    "P002_FRONT_HANDLE_JOINT_LOCAL_POS0_M",
    "P002_FRONT_HANDLE_JOINT_LOCAL_POS1_M",
    "P002_FRONT_HANDLE_JOINT_PATH",
    "P002_FRONT_HANDLE_LEVER_SHA256",
    "P002_FRONT_HANDLE_LEVER_USDA",
    "P002_FRONT_HANDLE_MIN_CLEARANCE_M",
    "P002_FRONT_HANDLE_ROSETTE_SHA256",
    "P002_FRONT_HANDLE_ROSETTE_USDA",
    "P002_FRONT_HANDLE_SOURCE_ARCHIVE_SHA256",
    "P002_LEVER_PATH",
    "P002_ROSETTE_PATH",
    "P002SceneGeometryReport",
    "P002_TABLE_BODY_PATH",
    "P002_TABLE_COLLISIONS_PATH",
    "P002_TABLE_ROOT_PATH",
    "P002_TASK_NAME",
    "apply_p002_scene_geometry",
    "p002_front_handle_dimensions_m",
]
