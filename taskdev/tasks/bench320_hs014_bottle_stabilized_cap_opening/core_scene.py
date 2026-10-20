"""Isaac Sim 5.1 core-scene composition for B320-HS014.

This module deliberately stops at scene construction and zero-action auditing.
It does not contain IK, grasp, insertion, or success-motion code.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np


TASK_NAME = "bench320_hs014_bottle_stabilized_cap_opening"
TASK_ID = "B320-HS014"
TASK_ROOT = "/World/HS014Core"
HEAD_CAMERA_PATH = "/World/RecordCameras/head_camera"
PHYSICS_HZ = 120
RENDER_HEIGHT = 480
RENDER_WIDTH = 640

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = REPOSITORY_ROOT / "taskdev" / "tasks" / TASK_NAME
BOTTLE_ASSET = TASK_DIR / "assets" / "ganten_570ml" / "ganten_570ml_bottle.usda"
CAP_ASSET = (
    TASK_DIR
    / "assets"
    / "ganten_570ml"
    / "ganten_570ml_cap.glb"
)

EXPECTED_SOURCE_HASHES = {
    "bottle": "86594f66838f9c421fc417699e50ba4cbfb4304e248f1c97c2d2f4f915a85e06",
    "cap": "ccd761ea9a5dbb7fdf08048c39fc463831914a8f082151b1024a54987e5238fa",
}

# The bottle stage is Z-up, but its referenced mesh is authored along local
# +Y from base (Y=0) to open mouth (Y=0.2 m).  Keep the task's rigid-owner root
# at the mouth: rotate source +Y to local +Z and translate the source down by
# 0.2 m.  The cap GLB is also Y-up and its node translation makes its bottom
# Y=0, so both source assets then use the same upright axial convention.
BOTTLE_SOURCE_LOCAL_AABB_M = np.asarray(
    [[-0.028000, 0.0, -0.028000], [0.028000, 0.200000, 0.028000]],
    dtype=np.float64,
)
BOTTLE_LOCAL_AABB_M = np.asarray(
    [[-0.028000, -0.028000, -0.200000], [0.028000, 0.028000, 0.000000]],
    dtype=np.float64,
)
BOTTLE_VISUAL_CORRECTION_WXYZ = np.asarray(
    [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0], dtype=np.float64
)
BOTTLE_VISUAL_TRANSLATION_M = np.asarray([0.0, 0.0, -0.2], dtype=np.float64)
CAP_LOCAL_AABB_M = np.asarray(
    [[-0.015225, 0.000000, -0.015225], [0.015225, 0.018000, 0.015225]],
    dtype=np.float64,
)
# Keep the free cap's mass properties independent of its collision topology.
# These values are the uniform thin-wall surface integral of the hash-bound
# visible cap mesh at 9 g.  In particular, adding or refining an internal
# collider must not silently change the physical body's center of mass or
# inertia in a later scene revision.
CAP_MASS_KG = 0.009
CAP_CENTER_OF_MASS_LOCAL_M = (0.0, 0.009, 0.0)
CAP_DIAGONAL_INERTIA_KG_M2 = (
    2.1e-6,
    2.8e-6,
    2.1e-6,
)
CAP_PRINCIPAL_AXES_WXYZ = (1.0, 0.0, 0.0, 0.0)
CAP_MASS_PROPERTIES_SOURCE = "uniform_surface_integral_of_hash_bound_visible_mesh"
# The source cap rounds inward over the first/last 2 mm of its local +Y
# extent.  Keep the analytic cylinder wholly inside the visible straight-wall
# region while moving its physical seating face 3 mm above the visual bottom.
# The old full-height cylinder touched the bottle's convex decomposition at an
# empirically measured +3.03 mm cap-root offset even though the signed contact
# separation was approximately zero.  This inset removes that hidden bottom
# protrusion without widening the strict seated-pose gate.
CAP_COLLISION_RADIUS_M = 0.0142
CAP_COLLISION_BOTTOM_INSET_M = 0.002
CAP_COLLISION_TOP_INSET_M = 0.002
CAP_COLLISION_HEIGHT_M = (
    float(CAP_LOCAL_AABB_M[1, 1] - CAP_LOCAL_AABB_M[0, 1])
    - CAP_COLLISION_BOTTOM_INSET_M
    - CAP_COLLISION_TOP_INSET_M
)
CAP_COLLISION_CENTER_Y_M = (
    float(CAP_LOCAL_AABB_M[0, 1])
    + CAP_COLLISION_BOTTOM_INSET_M
    + 0.5 * CAP_COLLISION_HEIGHT_M
)
CAP_COLLISION_LOCAL_AABB_M = np.asarray(
    [
        [
            -CAP_COLLISION_RADIUS_M,
            float(CAP_LOCAL_AABB_M[0, 1]) + CAP_COLLISION_BOTTOM_INSET_M,
            -CAP_COLLISION_RADIUS_M,
        ],
        [
            CAP_COLLISION_RADIUS_M,
            float(CAP_LOCAL_AABB_M[1, 1]) - CAP_COLLISION_TOP_INSET_M,
            CAP_COLLISION_RADIUS_M,
        ],
    ],
    dtype=np.float64,
)
CAP_CRADLE_RADIUS_M = 0.013
CAP_CRADLE_CENTER_Y_M = 0.009
CAP_CRADLE_LOCAL_AABB_M = np.asarray(
    [[-0.013, 0.0, -0.013], [0.013, 0.017, 0.013]], dtype=np.float64
)
BOTTLE_CRADLE_SUPPORT_RADIUS_M = 0.0015
BOTTLE_CRADLE_SUPPORT_RING_RADIUS_M = 0.0024
BOTTLE_CRADLE_SUPPORT_CENTER_Z_M = -0.0055
BOTTLE_CRADLE_SUPPORT_ANGLES_DEG = (0.0, 120.0, 240.0)
BOTTLE_CRADLE_SUPPORT_CENTERS_M = np.asarray(
    [
        [
            BOTTLE_CRADLE_SUPPORT_RING_RADIUS_M
            * math.cos(math.radians(angle_deg)),
            BOTTLE_CRADLE_SUPPORT_RING_RADIUS_M
            * math.sin(math.radians(angle_deg)),
            BOTTLE_CRADLE_SUPPORT_CENTER_Z_M,
        ]
        for angle_deg in BOTTLE_CRADLE_SUPPORT_ANGLES_DEG
    ],
    dtype=np.float64,
)
BOTTLE_CRADLE_SUPPORT_RADIAL_ENVELOPE_M = (
    BOTTLE_CRADLE_SUPPORT_RING_RADIUS_M + BOTTLE_CRADLE_SUPPORT_RADIUS_M
)
BOTTLE_MOUTH_CONTAINMENT_RADIUS_M = 0.008
CRADLE_HELPER_CONTACT_OFFSET_M = 0.00005
CRADLE_HELPER_REST_OFFSET_M = 0.0
CRADLE_HELPER_STATIC_FRICTION = 0.0
CRADLE_HELPER_DYNAMIC_FRICTION = 0.0
CRADLE_HELPER_RESTITUTION = 0.0
CRADLE_HELPER_FRICTION_COMBINE_MODE = "min"

BOTTLE_XY_M = np.asarray([-0.4225, -0.1225], dtype=np.float64)
CAP_XY_M = BOTTLE_XY_M.copy()
CAP_SEATED_HEIGHT_M = -0.0002
CRADLE_CENTER_DZ_M = (
    CAP_SEATED_HEIGHT_M
    + CAP_CRADLE_CENTER_Y_M
    - BOTTLE_CRADLE_SUPPORT_CENTER_Z_M
)
CRADLE_CENTER_DISTANCE_M = math.hypot(
    BOTTLE_CRADLE_SUPPORT_RING_RADIUS_M, CRADLE_CENTER_DZ_M
)
CRADLE_RADIUS_SUM_M = CAP_CRADLE_RADIUS_M + BOTTLE_CRADLE_SUPPORT_RADIUS_M
CAP_UPRIGHT_QUAT_XYZW = np.asarray(
    [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], dtype=np.float64
)
COLLISION_CONTACT_OFFSET_M = 0.0005
COLLISION_REST_OFFSET_M = 0.0
RIGID_SOLVER_POSITION_ITERATIONS = 16
RIGID_SOLVER_VELOCITY_ITERATIONS = 8
RIGID_MAX_DEPENETRATION_VELOCITY_M_S = 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if len(raw) < 20:
        raise ValueError("GLB is truncated")
    magic, version, total = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or total != len(raw):
        raise ValueError("GLB header is invalid")
    chunks: dict[int, bytes] = {}
    offset = 12
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunks[kind] = raw[offset : offset + length]
        offset += length
    if offset != len(raw) or 0x4E4F534A not in chunks or 0x004E4942 not in chunks:
        raise ValueError("GLB JSON/BIN chunks are invalid")
    return json.loads(chunks[0x4E4F534A]), chunks[0x004E4942]


_COMPONENT_FORMAT = {5123: "H", 5125: "I", 5126: "f"}
_COMPONENT_BYTES = {5123: 2, 5125: 4, 5126: 4}
_TYPE_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}


def _decode_accessor(document: dict[str, Any], binary: bytes, index: int) -> np.ndarray:
    accessor = document["accessors"][index]
    if "sparse" in accessor:
        raise ValueError("sparse GLB accessors are not supported")
    component_type = int(accessor["componentType"])
    value_type = str(accessor["type"])
    if component_type not in _COMPONENT_FORMAT or value_type not in _TYPE_WIDTH:
        raise ValueError("unsupported GLB accessor encoding")
    width = _TYPE_WIDTH[value_type]
    component_bytes = _COMPONENT_BYTES[component_type]
    view = document["bufferViews"][int(accessor["bufferView"])]
    base = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", width * component_bytes))
    count = int(accessor["count"])
    values = []
    fmt = "<" + _COMPONENT_FORMAT[component_type] * width
    for row in range(count):
        values.append(struct.unpack_from(fmt, binary, base + row * stride))
    array = np.asarray(values)
    if value_type == "SCALAR":
        array = array.reshape(count)
    return array


def decode_authoritative_cap(path: Path = CAP_ASSET) -> dict[str, Any]:
    document, binary = _read_glb(path)
    if len(document.get("meshes", [])) != 1:
        raise ValueError("authoritative cap must contain exactly one mesh")
    primitive = document["meshes"][0]["primitives"][0]
    attributes = primitive["attributes"]
    points = _decode_accessor(document, binary, int(attributes["POSITION"]))
    normals = _decode_accessor(document, binary, int(attributes["NORMAL"]))
    texcoords = _decode_accessor(document, binary, int(attributes["TEXCOORD_0"]))
    indices = _decode_accessor(document, binary, int(primitive["indices"]))
    node = next(item for item in document["nodes"] if item.get("mesh") == 0)
    material = document["materials"][int(primitive["material"])]
    pbr = material["pbrMetallicRoughness"]
    translated_points = points + np.asarray(node.get("translation", [0, 0, 0]))
    observed_bounds = np.asarray(
        [translated_points.min(axis=0), translated_points.max(axis=0)],
        dtype=np.float64,
    )
    if points.ndim != 2 or points.shape[1] != 3 or indices.ndim != 1 or len(indices) % 3:
        raise ValueError("authoritative cap mesh topology is invalid")
    if not np.allclose(observed_bounds, CAP_LOCAL_AABB_M, atol=1.0e-8, rtol=0.0):
        raise ValueError(f"authoritative cap bounds changed: {observed_bounds.tolist()}")
    return {
        "points": points.astype(np.float32),
        "normals": normals.astype(np.float32),
        "texcoords": texcoords.astype(np.float32),
        "indices": indices.astype(np.int64),
        "node_translation": np.asarray(node.get("translation", [0, 0, 0])),
        "base_color": np.asarray(pbr["baseColorFactor"], dtype=np.float32),
        "roughness": float(pbr["roughnessFactor"]),
        "metallic": float(pbr["metallicFactor"]),
        "bounds": observed_bounds,
    }


def cap_collision_containment_contract(cap: dict[str, Any]) -> dict[str, Any]:
    """Prove the inset cylinder lies inside the cap's straight-wall profile."""

    points = np.asarray(cap["points"], dtype=np.float64) + np.asarray(
        cap["node_translation"], dtype=np.float64
    )
    radial = np.linalg.norm(points[:, (0, 2)], axis=1)
    full_radius_levels: list[float] = []
    for level in np.unique(points[:, 1]):
        at_level = np.isclose(points[:, 1], level, rtol=0.0, atol=1.0e-8)
        if float(np.min(radial[at_level])) >= CAP_COLLISION_RADIUS_M - 1.0e-8:
            full_radius_levels.append(float(level))
    if not full_radius_levels:
        raise ValueError("authoritative cap has no full-radius straight wall")
    straight_wall_interval = np.asarray(
        [min(full_radius_levels), max(full_radius_levels)], dtype=np.float64
    )
    collision_interval = CAP_COLLISION_LOCAL_AABB_M[:, 1]
    inside_straight_wall = bool(
        collision_interval[0] >= straight_wall_interval[0]
        and collision_interval[1] <= straight_wall_interval[1]
    )
    inside_bounds = bool(
        np.all(CAP_COLLISION_LOCAL_AABB_M[0] >= cap["bounds"][0])
        and np.all(CAP_COLLISION_LOCAL_AABB_M[1] <= cap["bounds"][1])
    )
    return {
        "collision_local_aabb_m": CAP_COLLISION_LOCAL_AABB_M.tolist(),
        "visible_straight_wall_interval_y_m": straight_wall_interval.tolist(),
        "inside_visible_bounds": inside_bounds,
        "inside_visible_straight_wall": inside_straight_wall,
        "pass": bool(inside_bounds and inside_straight_wall),
    }


def source_asset_contract() -> dict[str, Any]:
    observed = {
        "bottle": sha256_file(BOTTLE_ASSET),
        "cap": sha256_file(CAP_ASSET),
    }
    cap = decode_authoritative_cap()
    cap_collision = cap_collision_containment_contract(cap)
    cap_cradle_inside = bool(
        np.all(CAP_CRADLE_LOCAL_AABB_M[0] >= cap["bounds"][0])
        and np.all(CAP_CRADLE_LOCAL_AABB_M[1] <= cap["bounds"][1])
    )
    bottle_cradle_inside = bool(
        BOTTLE_CRADLE_SUPPORT_RADIAL_ENVELOPE_M
        <= BOTTLE_MOUTH_CONTAINMENT_RADIUS_M
        and BOTTLE_CRADLE_SUPPORT_CENTER_Z_M
        + BOTTLE_CRADLE_SUPPORT_RADIUS_M <= 0.0
    )
    return {
        "observed_sha256": observed,
        "expected_sha256": EXPECTED_SOURCE_HASHES,
        "cap_vertex_count": int(cap["points"].shape[0]),
        "cap_triangle_count": int(cap["indices"].shape[0] // 3),
        "cap_source_local_aabb_m": cap["bounds"].tolist(),
        "cap_collision_local_aabb_m": cap_collision["collision_local_aabb_m"],
        "cap_collision_visible_straight_wall_interval_y_m": cap_collision[
            "visible_straight_wall_interval_y_m"
        ],
        "cap_collision_inside_visible_bounds": cap_collision[
            "inside_visible_bounds"
        ],
        "cap_collision_inside_visible_straight_wall": cap_collision[
            "inside_visible_straight_wall"
        ],
        "cap_cradle_local_aabb_m": CAP_CRADLE_LOCAL_AABB_M.tolist(),
        "cap_cradle_inside_visible_bounds": cap_cradle_inside,
        "bottle_cradle_support_radial_envelope_m": (
            BOTTLE_CRADLE_SUPPORT_RADIAL_ENVELOPE_M
        ),
        "bottle_cradle_support_inside_mouth": bottle_cradle_inside,
        "bottle_source_local_aabb_m": BOTTLE_SOURCE_LOCAL_AABB_M.tolist(),
        "bottle_composed_local_aabb_m": BOTTLE_LOCAL_AABB_M.tolist(),
        "pass": bool(
            all(observed.get(key) == value for key, value in EXPECTED_SOURCE_HASHES.items())
            and cap_collision["pass"]
            and cap_cradle_inside
            and bottle_cradle_inside
        ),
    }


def _quat_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def world_aabb(local_aabb: np.ndarray, transform_xyzw: np.ndarray) -> np.ndarray:
    low, high = np.asarray(local_aabb, dtype=np.float64).reshape(2, 3)
    corners = np.asarray(
        [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]
    )
    transform = np.asarray(transform_xyzw, dtype=np.float64).reshape(7)
    points = corners @ _quat_matrix_xyzw(transform[3:]).T + transform[:3]
    return np.asarray([points.min(axis=0), points.max(axis=0)])


def aabb_overlap_depth(first: np.ndarray, second: np.ndarray) -> float:
    overlap = np.minimum(first[1], second[1]) - np.maximum(first[0], second[0])
    return float(np.min(overlap)) if np.all(overlap > 0.0) else 0.0


def _set_identity_scale_and_pose(
    prim: Any,
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> None:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*[float(value) for value in position])
    )
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(
            float(quaternion[3]),
            Gf.Vec3d(*[float(value) for value in quaternion[:3]]),
        )
    )
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))


def _configure_rigid_owner(
    prim: Any,
    *,
    mass_kg: float,
    center_of_mass: tuple[float, float, float] | None = None,
    diagonal_inertia: tuple[float, float, float] | None = None,
) -> None:
    from pxr import Gf, PhysxSchema, UsdPhysics

    rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(float(mass_kg))
    if center_of_mass is not None:
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*center_of_mass))
    if diagonal_inertia is not None:
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*diagonal_inertia))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    physx = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx.CreateSolverPositionIterationCountAttr(
        RIGID_SOLVER_POSITION_ITERATIONS
    )
    physx.CreateSolverVelocityIterationCountAttr(
        RIGID_SOLVER_VELOCITY_ITERATIONS
    )
    physx.CreateMaxDepenetrationVelocityAttr(
        RIGID_MAX_DEPENETRATION_VELOCITY_M_S
    )
    try:
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
    except Exception:
        # Contact-report schema availability varies across Isaac 5.1 builds;
        # owner/collision checks do not rely on callback delivery.
        pass


def _configure_collider(
    prim: Any,
    *,
    contact_offset_m: float = COLLISION_CONTACT_OFFSET_M,
    rest_offset_m: float = COLLISION_REST_OFFSET_M,
) -> None:
    from pxr import PhysxSchema, UsdPhysics

    collision = UsdPhysics.CollisionAPI.Apply(prim)
    collision.CreateCollisionEnabledAttr(True)
    try:
        physx = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        # A 2 mm offset on both the bottle mouth and cap created a combined
        # 4 mm contact envelope and held the visible cap about 3.4 mm above
        # the visible mouth.  Keep contact generation local to this 26 mm cap
        # without weakening the global stability/velocity gates.
        physx.CreateContactOffsetAttr(float(contact_offset_m))
        physx.CreateRestOffsetAttr(float(rest_offset_m))
    except Exception:
        pass


def _author_cap_visual(stage: Any, cap_root_path: str, cap_data: dict[str, Any]) -> str:
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    visual_path = f"{cap_root_path}/Visual"
    visual = UsdGeom.Xform.Define(stage, visual_path)
    visual_xform = UsdGeom.Xformable(visual.GetPrim())
    visual_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*[float(value) for value in cap_data["node_translation"]])
    )
    mesh = UsdGeom.Mesh.Define(stage, f"{visual_path}/AuthoritativeCapMesh")
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreatePointsAttr(
        [Gf.Vec3f(*[float(value) for value in row]) for row in cap_data["points"]]
    )
    indices = [int(value) for value in cap_data["indices"]]
    mesh.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateNormalsAttr(
        [Gf.Vec3f(*[float(value) for value in row]) for row in cap_data["normals"]]
    )
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    primvar.Set(
        [Gf.Vec2f(*[float(value) for value in row]) for row in cap_data["texcoords"]]
    )

    material = UsdShade.Material.Define(stage, f"{cap_root_path}/Looks/CapBlue")
    shader = UsdShade.Shader.Define(stage, f"{cap_root_path}/Looks/CapBlue/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    color = cap_data["base_color"]
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
    )
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(color[3]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
        float(cap_data["roughness"])
    )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
        float(cap_data["metallic"])
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
    return str(mesh.GetPath())


def _author_cap_collision(stage: Any, cap_root_path: str) -> str:
    from pxr import Gf, UsdGeom

    cylinder = UsdGeom.Cylinder.Define(stage, f"{cap_root_path}/Collision")
    cylinder.CreateAxisAttr(UsdGeom.Tokens.y)
    cylinder.CreateRadiusAttr(CAP_COLLISION_RADIUS_M)
    cylinder.CreateHeightAttr(CAP_COLLISION_HEIGHT_M)
    xform = UsdGeom.Xformable(cylinder.GetPrim())
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(0.0, CAP_COLLISION_CENTER_Y_M, 0.0)
    )
    UsdGeom.Imageable(cylinder.GetPrim()).MakeInvisible()
    _configure_collider(cylinder.GetPrim())
    return str(cylinder.GetPath())


def _author_cradle_helpers(
    stage: Any, bottle_root_path: str, cap_root_path: str
) -> tuple[list[str], str, str]:
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    material_path = f"{cap_root_path}/PhysicsMaterials/CradleFrictionless"
    material = UsdShade.Material.Define(stage, material_path)
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(CRADLE_HELPER_STATIC_FRICTION)
    material_api.CreateDynamicFrictionAttr().Set(CRADLE_HELPER_DYNAMIC_FRICTION)
    material_api.CreateRestitutionAttr().Set(CRADLE_HELPER_RESTITUTION)
    physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material_api.CreateFrictionCombineModeAttr().Set(
        CRADLE_HELPER_FRICTION_COMBINE_MODE
    )

    support_paths: list[str] = []
    for index, center in enumerate(BOTTLE_CRADLE_SUPPORT_CENTERS_M):
        support = UsdGeom.Sphere.Define(
            stage, f"{bottle_root_path}/CradleSupport{index}"
        )
        support.CreateRadiusAttr(BOTTLE_CRADLE_SUPPORT_RADIUS_M)
        UsdGeom.Xformable(support.GetPrim()).AddTranslateOp(
            UsdGeom.XformOp.PrecisionDouble
        ).Set(Gf.Vec3d(*[float(value) for value in center]))
        UsdGeom.Imageable(support.GetPrim()).MakeInvisible()
        _configure_collider(
            support.GetPrim(),
            contact_offset_m=CRADLE_HELPER_CONTACT_OFFSET_M,
            rest_offset_m=CRADLE_HELPER_REST_OFFSET_M,
        )
        UsdShade.MaterialBindingAPI.Apply(support.GetPrim()).Bind(
            material, bindingStrength="fallbackStrength", materialPurpose="physics"
        )
        support_paths.append(str(support.GetPath()))

    cap_helper = UsdGeom.Sphere.Define(stage, f"{cap_root_path}/CradleSphere")
    cap_helper.CreateRadiusAttr(CAP_CRADLE_RADIUS_M)
    UsdGeom.Xformable(cap_helper.GetPrim()).AddTranslateOp(
        UsdGeom.XformOp.PrecisionDouble
    ).Set(Gf.Vec3d(0.0, CAP_CRADLE_CENTER_Y_M, 0.0))
    UsdGeom.Imageable(cap_helper.GetPrim()).MakeInvisible()
    _configure_collider(
        cap_helper.GetPrim(),
        contact_offset_m=CRADLE_HELPER_CONTACT_OFFSET_M,
        rest_offset_m=CRADLE_HELPER_REST_OFFSET_M,
    )
    UsdShade.MaterialBindingAPI.Apply(cap_helper.GetPrim()).Bind(
        material, bindingStrength="fallbackStrength", materialPurpose="physics"
    )
    if len(support_paths) != 3:
        raise RuntimeError("HS014 requires exactly three cradle support spheres")
    return support_paths, str(cap_helper.GetPath()), material_path


def _author_cradle_filters(
    stage: Any,
    source_colliders: list[Any],
    support_paths: list[str],
    cap_outer_path: str,
    cap_helper_path: str,
) -> list[list[str]]:
    from pxr import Sdf, UsdPhysics

    source_paths = [str(prim.GetPath()) for prim in source_colliders]
    expected: dict[str, set[str]] = {
        **{
            source_path: {cap_outer_path, cap_helper_path}
            for source_path in source_paths
        },
        **{support_path: {cap_outer_path} for support_path in support_paths},
        cap_outer_path: set(source_paths) | set(support_paths),
        cap_helper_path: set(source_paths),
    }
    for path, targets in expected.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"HS014 cradle filter prim is unavailable: {path}")
        relation = UsdPhysics.FilteredPairsAPI.Apply(
            prim
        ).CreateFilteredPairsRel()
        for target in targets:
            relation.AddTarget(Sdf.Path(target))
        if {str(target) for target in relation.GetTargets()} != targets:
            raise RuntimeError(f"HS014 cradle filter targets changed: {path}")
    return [
        *[
            [source_path, cap_path]
            for source_path in source_paths
            for cap_path in (cap_outer_path, cap_helper_path)
        ],
        *[[support_path, cap_outer_path] for support_path in support_paths],
    ]


def _collision_prims(root: Any) -> list[Any]:
    from pxr import Usd, UsdPhysics

    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    enabled = []
    for prim in Usd.PrimRange(root, predicate):
        if not prim.IsActive() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        value = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if value is not False:
            enabled.append(prim)
    return enabled


def _apply_bottle_visual_axis_correction(source_geometry: Any) -> dict[str, Any]:
    from pxr import Gf, Sdf, UsdGeom

    xform = UsdGeom.Xformable(source_geometry)
    translate = next(
        (
            op
            for op in xform.GetOrderedXformOps()
            if op.GetOpName() == "xformOp:translate"
        ),
        None,
    )
    if translate is None:
        # Add translation first so a portable unresolved-reference overlay
        # receives the same canonical T*R*S op order as the source USD.
        translate = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate.Set(Gf.Vec3d(*[float(value) for value in BOTTLE_VISUAL_TRANSLATION_M]))
    orient = next(
        (op for op in xform.GetOrderedXformOps() if op.GetOpName() == "xformOp:orient"),
        None,
    )
    if orient is None:
        orient = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    if orient.GetAttr().GetTypeName() != Sdf.ValueTypeNames.Quatd:
        raise RuntimeError("HS014 bottle source orient is not quatd")
    quaternion = BOTTLE_VISUAL_CORRECTION_WXYZ
    orient.Set(
        Gf.Quatd(
            float(quaternion[0]),
            Gf.Vec3d(*[float(value) for value in quaternion[1:4]]),
        )
    )
    mapped_axis = _quat_matrix_xyzw(
        np.asarray([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    ) @ np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    aligned = bool(np.allclose(mapped_axis, [0.0, 0.0, 1.0], atol=1.0e-12))
    if not aligned:
        raise RuntimeError("HS014 bottle visual axis correction changed")
    return {
        "bottle_source_mesh_axis": "+Y",
        "bottle_source_axis_semantics": "base_to_open_mouth",
        "bottle_composed_axis": "+Z_from_base_to_open_mouth",
        "bottle_visual_correction_wxyz": quaternion.tolist(),
        "bottle_visual_translation_m": BOTTLE_VISUAL_TRANSLATION_M.tolist(),
        "bottle_visual_proxy_axis_aligned": aligned,
    }


def compose_assets(
    stage: Any,
    *,
    table_top_z: float,
    bottle_reference: str,
    cap_data: dict[str, Any],
    task_root: str = TASK_ROOT,
    task_complete: bool = False,
    scope: str = "core_scene_only",
) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdPhysics

    cap_collision_containment = cap_collision_containment_contract(cap_data)
    if not cap_collision_containment["pass"]:
        raise RuntimeError(
            "cap collision is not fully contained by the authoritative visual geometry"
        )
    root = UsdGeom.Xform.Define(stage, task_root).GetPrim()
    root.SetCustomDataByKey("task_id", TASK_ID)
    root.SetCustomDataByKey("task_complete", bool(task_complete))
    root.SetCustomDataByKey("scope", scope)

    bottle_path = f"{task_root}/WaterBottle"
    bottle = UsdGeom.Xform.Define(stage, bottle_path).GetPrim()
    bottle.GetReferences().AddReference(bottle_reference)
    source_geometry_path = f"{bottle_path}/geometry"
    source_geometry = stage.GetPrimAtPath(source_geometry_path)
    source_reference_resolved = source_geometry.IsValid()
    if not source_reference_resolved:
        # Portable overlays are first authored beside their evidence report,
        # where the task-relative asset may not resolve until the USDA is
        # copied back into the task directory.  Author the same local opinions
        # as overs so they compose correctly once the reference resolves.
        source_geometry = stage.OverridePrim(source_geometry_path)
    if not source_reference_resolved or source_geometry.IsInstance():
        # The source packages its geometry as an instanceable reference.  A
        # local de-instance opinion is required before collisionEnabled can be
        # overridden on the referenced collider; visuals and authored scale
        # remain byte-for-byte sourced from the authoritative USD.
        source_geometry.SetInstanceable(False)
    visual_axis = _apply_bottle_visual_axis_correction(source_geometry)
    bottle_position = np.asarray(
        [BOTTLE_XY_M[0], BOTTLE_XY_M[1], table_top_z + 0.2002], dtype=np.float64
    )
    _set_identity_scale_and_pose(
        bottle, bottle_position, np.asarray([0.0, 0.0, 0.0, 1.0])
    )
    _configure_rigid_owner(
        bottle,
        mass_kg=0.10,
        center_of_mass=(0.0, 0.0, -0.10),
        diagonal_inertia=(0.00035, 0.00035, 0.000041),
    )

    # Use the source asset's authored convex-decomposition collider under the
    # same corrected transform as its visible mesh.  The historical analytic
    # cylinders and 34 mm stability foot were qualified while the visual was
    # upside down and protruded materially beyond the surface, so they are not
    # part of the corrected production geometry.
    source_colliders = _collision_prims(bottle)
    if not source_colliders:
        # Portable overlays are authored before their relative asset bundle is
        # copied beside them.  Preserve an enabled opinion at the hash-bound
        # source mesh path; it composes onto the authored collider once the
        # bundle is present and does not invent substitute geometry.
        source_collision_path = f"{source_geometry_path}/mesh"
        source_collider = stage.OverridePrim(source_collision_path)
        _configure_collider(source_collider)
        source_colliders = [source_collider]
    for source_collider in source_colliders:
        _configure_collider(source_collider)
    nested_bottle_owners = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(bottle)
        if prim != bottle and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if nested_bottle_owners:
        raise RuntimeError(f"bottle contains nested rigid owners: {nested_bottle_owners}")

    cap_path = f"{task_root}/BottleCap"
    cap = UsdGeom.Xform.Define(stage, cap_path).GetPrim()
    cap_position = np.asarray(
        [
            CAP_XY_M[0],
            CAP_XY_M[1],
            table_top_z + 0.2002 + CAP_SEATED_HEIGHT_M,
        ],
        dtype=np.float64,
    )
    _set_identity_scale_and_pose(cap, cap_position, CAP_UPRIGHT_QUAT_XYZW)
    _configure_rigid_owner(
        cap,
        mass_kg=CAP_MASS_KG,
        center_of_mass=CAP_CENTER_OF_MASS_LOCAL_M,
        diagonal_inertia=CAP_DIAGONAL_INERTIA_KG_M2,
    )
    cap_visual = _author_cap_visual(stage, cap_path, cap_data)
    cap_collision = _author_cap_collision(stage, cap_path)
    support_paths, cap_helper_path, helper_material_path = _author_cradle_helpers(
        stage, bottle_path, cap_path
    )
    filtered_pairs = _author_cradle_filters(
        stage, source_colliders, support_paths, cap_collision, cap_helper_path
    )
    bottle_colliders = _collision_prims(bottle)
    if bottle.HasAPI(UsdPhysics.FilteredPairsAPI) or cap.HasAPI(
        UsdPhysics.FilteredPairsAPI
    ):
        raise RuntimeError("HS014 cradle forbids root-level filtered pairs")
    nested_cap_owners = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(cap)
        if prim != cap and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if nested_cap_owners:
        raise RuntimeError(f"cap contains nested rigid owners: {nested_cap_owners}")

    return {
        "root_path": task_root,
        "cap_bottle_contact_topology": {
            "contact_truth_topology": "three_sphere_cradle_to_com_sphere",
            "helper_collision_paths": [*support_paths, cap_helper_path],
            "filtered_shape_pairs": filtered_pairs,
            "source_collision_paths": [
                str(prim.GetPath()) for prim in source_colliders
            ],
            "bottle_support_paths": support_paths,
            "cap_outer_path": cap_collision,
            "cap_helper_path": cap_helper_path,
            "helper_physics_material_path": helper_material_path,
            "unfiltered_contact_pairs": [
                [support_path, cap_helper_path] for support_path in support_paths
            ],
            "contact_normals_intersect_cap_com": True,
            "center_delta_z_m": CRADLE_CENTER_DZ_M,
            "center_horizontal_offset_m": BOTTLE_CRADLE_SUPPORT_RING_RADIUS_M,
            "center_distance_m": CRADLE_CENTER_DISTANCE_M,
            "radius_sum_m": CRADLE_RADIUS_SUM_M,
            "exact_initial_tangency": math.isclose(
                CRADLE_CENTER_DISTANCE_M,
                CRADLE_RADIUS_SUM_M,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            "axial_lift_separates_immediately": True,
        },
        "bottle": {
            "prim_path": bottle_path,
            "position_m": bottle_position.tolist(),
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
            "collision_paths": [str(prim.GetPath()) for prim in bottle_colliders],
            "collision_model": "source_convex_decomposition_dynamic",
            "source_collision_paths": [
                str(prim.GetPath()) for prim in source_colliders
            ],
            "helper_collision_paths": support_paths,
            "cradle_support_radius_m": BOTTLE_CRADLE_SUPPORT_RADIUS_M,
            "cradle_support_ring_radius_m": BOTTLE_CRADLE_SUPPORT_RING_RADIUS_M,
            "cradle_support_center_z_m": BOTTLE_CRADLE_SUPPORT_CENTER_Z_M,
            "cradle_support_angles_deg": list(BOTTLE_CRADLE_SUPPORT_ANGLES_DEG),
            "cradle_support_centers_m": BOTTLE_CRADLE_SUPPORT_CENTERS_M.tolist(),
            "cradle_support_radial_envelope_m": (
                BOTTLE_CRADLE_SUPPORT_RADIAL_ENVELOPE_M
            ),
            "cradle_support_inside_mouth": bool(
                BOTTLE_CRADLE_SUPPORT_RADIAL_ENVELOPE_M
                <= BOTTLE_MOUTH_CONTAINMENT_RADIUS_M
            ),
            "nested_rigid_owner_paths": nested_bottle_owners,
            "disabled_source_collision_paths": [],
            "deinstanced_source_geometry": bool(
                source_geometry.IsValid() and not source_geometry.IsInstance()
            ),
            "proxy_collision_paths": [],
            "center_of_mass_local_m": [0.0, 0.0, -0.10],
            "diagonal_inertia_kg_m2": [0.00035, 0.00035, 0.000041],
            "contact_offset_m": COLLISION_CONTACT_OFFSET_M,
            "rest_offset_m": COLLISION_REST_OFFSET_M,
            "solver_position_iteration_count": RIGID_SOLVER_POSITION_ITERATIONS,
            "solver_velocity_iteration_count": RIGID_SOLVER_VELOCITY_ITERATIONS,
            "max_depenetration_velocity_m_s": (
                RIGID_MAX_DEPENETRATION_VELOCITY_M_S
            ),
            **visual_axis,
        },
        "cap": {
            "prim_path": cap_path,
            "position_m": cap_position.tolist(),
            "orientation_xyzw": CAP_UPRIGHT_QUAT_XYZW.tolist(),
            "scale": [1.0, 1.0, 1.0],
            "mass_kg": CAP_MASS_KG,
            "center_of_mass_local_m": list(CAP_CENTER_OF_MASS_LOCAL_M),
            "diagonal_inertia_kg_m2": list(CAP_DIAGONAL_INERTIA_KG_M2),
            "principal_axes_wxyz": list(CAP_PRINCIPAL_AXES_WXYZ),
            "mass_properties_source": CAP_MASS_PROPERTIES_SOURCE,
            "visual_path": cap_visual,
            "collision_paths": [cap_collision, cap_helper_path],
            "outer_collision_path": cap_collision,
            "cradle_helper_path": cap_helper_path,
            "collision_model": "outer_cylinder_and_com_centered_cradle_sphere",
            "collision_radius_m": CAP_COLLISION_RADIUS_M,
            "collision_height_m": CAP_COLLISION_HEIGHT_M,
            "collision_center_y_m": CAP_COLLISION_CENTER_Y_M,
            "collision_bottom_inset_m": CAP_COLLISION_BOTTOM_INSET_M,
            "collision_top_inset_m": CAP_COLLISION_TOP_INSET_M,
            "collision_local_aabb_m": CAP_COLLISION_LOCAL_AABB_M.tolist(),
            "collision_inside_visible_bounds": bool(
                cap_collision_containment["inside_visible_bounds"]
            ),
            "collision_inside_visible_straight_wall": bool(
                cap_collision_containment["inside_visible_straight_wall"]
            ),
            "helper_collision_paths": [cap_helper_path],
            "cradle_radius_m": CAP_CRADLE_RADIUS_M,
            "cradle_center_y_m": CAP_CRADLE_CENTER_Y_M,
            "cradle_local_aabb_m": CAP_CRADLE_LOCAL_AABB_M.tolist(),
            "cradle_inside_visible_bounds": bool(
                np.all(CAP_CRADLE_LOCAL_AABB_M[0] >= CAP_LOCAL_AABB_M[0])
                and np.all(CAP_CRADLE_LOCAL_AABB_M[1] <= CAP_LOCAL_AABB_M[1])
            ),
            "cradle_center_matches_center_of_mass": bool(
                np.allclose(
                    [0.0, CAP_CRADLE_CENTER_Y_M, 0.0],
                    CAP_CENTER_OF_MASS_LOCAL_M,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ),
            "helper_physics_material_path": helper_material_path,
            "helper_contact_offset_m": CRADLE_HELPER_CONTACT_OFFSET_M,
            "helper_rest_offset_m": CRADLE_HELPER_REST_OFFSET_M,
            "helper_static_friction": CRADLE_HELPER_STATIC_FRICTION,
            "helper_dynamic_friction": CRADLE_HELPER_DYNAMIC_FRICTION,
            "helper_restitution": CRADLE_HELPER_RESTITUTION,
            "helper_friction_combine_mode": CRADLE_HELPER_FRICTION_COMBINE_MODE,
            "outer_has_helper_material_binding": False,
            "nested_rigid_owner_paths": nested_cap_owners,
            "contact_offset_m": COLLISION_CONTACT_OFFSET_M,
            "rest_offset_m": COLLISION_REST_OFFSET_M,
            "solver_position_iteration_count": RIGID_SOLVER_POSITION_ITERATIONS,
            "solver_velocity_iteration_count": RIGID_SOLVER_VELOCITY_ITERATIONS,
            "max_depenetration_velocity_m_s": (
                RIGID_MAX_DEPENETRATION_VELOCITY_M_S
            ),
        },
    }


def _tensor_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _nearest_rigid_owner_path(prim: Any) -> str | None:
    from pxr import UsdPhysics

    candidate = prim
    while candidate is not None and candidate.IsValid():
        if candidate.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(candidate.GetPath())
        candidate = candidate.GetParent()
    return None


def _quat_angle_rad(first_xyzw: np.ndarray, second_xyzw: np.ndarray) -> float:
    first = np.asarray(first_xyzw, dtype=np.float64)
    second = np.asarray(second_xyzw, dtype=np.float64)
    first /= max(float(np.linalg.norm(first)), 1.0e-12)
    second /= max(float(np.linalg.norm(second)), 1.0e-12)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(first, second)), 0.0, 1.0)))


def _rgb_metrics(frame: np.ndarray) -> dict[str, Any]:
    array = np.asarray(frame)
    rgb = array[..., :3].astype(np.float64)
    luminance = rgb.mean(axis=2)
    pixel_std = float(rgb.std())
    return {
        "shape_hwc": list(array.shape),
        "dtype": str(array.dtype),
        "mean_rgb": float(rgb.mean()),
        "max_rgb": float(rgb.max()),
        "pixel_std": pixel_std,
        "pixel_variance": pixel_std**2,
        "black_fraction": float(np.mean(luminance < 5.0)),
    }


class HS014CoreScene:
    """Non-production asset scene and ten-second no-action object audit.

    This helper deliberately starts from the B01 shared scene and composes the
    HS014 assets into it.  It does not qualify the exact production HS014 child,
    its first ``SimulationContext.reset()``, or either arm's live reset truth.
    """

    def __init__(
        self,
        *,
        simulation_app: Any,
        app_launcher: Any,
        output_dir: Path,
        device: str,
    ) -> None:
        from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv

        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.env = HandsOnABCIsaacLabEnv(
            "t04_cover_pitcher",
            seed=19,
            control_dt=1.0 / 30.0,
            physics_dt=1.0 / float(PHYSICS_HZ),
            time_limit=30.0,
            render_mode="rgb_array",
            image_obs=True,
            randomize=False,
            randomize_dynamics=False,
            hand_type="wuji1",
            arm_type="rokae",
            launch_isaac=True,
            headless=True,
            render_resolution=(RENDER_HEIGHT, RENDER_WIDTH),
            hz=30,
            isaac_device=device,
            physics_backend="physx",
            existing_simulation_app=simulation_app,
            existing_app_launcher=app_launcher,
        )
        self.stage = self.env._stage
        self.sim = self.env._sim
        if self.stage is None or self.sim is None:
            raise RuntimeError("native HandsOn Isaac scene did not initialize")

    def close(self) -> None:
        self.env.close()

    def _hide_semantic_seed_objects(self) -> dict[str, Any]:
        from pxr import Usd, UsdGeom, UsdPhysics

        changed_visuals = 0
        disabled_colliders = 0
        roots = []
        for name in ("019_pitcher_base", "pitcher_lid"):
            prim = self.env._find_prim(name)
            if prim is None or not prim.IsValid():
                continue
            roots.append(str(prim.GetPath()))
            for candidate in Usd.PrimRange(prim):
                if candidate.IsA(UsdGeom.Gprim):
                    UsdGeom.Imageable(candidate).MakeInvisible()
                    changed_visuals += 1
                if candidate.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(candidate).CreateCollisionEnabledAttr(False)
                    disabled_colliders += 1
        return {
            "semantic_seed": "t04_cover_pitcher",
            "hidden_roots": roots,
            "hidden_visual_count": changed_visuals,
            "disabled_collision_count": disabled_colliders,
            "pass": len(roots) == 2 and changed_visuals > 0 and disabled_colliders > 0,
        }

    def _ensure_lighting(self) -> dict[str, Any]:
        from pxr import Gf, UsdGeom, UsdLux

        lights_root = UsdGeom.Scope.Define(self.stage, f"{TASK_ROOT}/Lights")
        dome = UsdLux.DomeLight.Define(self.stage, f"{lights_root.GetPath()}/Dome")
        dome.CreateIntensityAttr(120.0)
        dome.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.0))
        distant = UsdLux.DistantLight.Define(
            self.stage, f"{lights_root.GetPath()}/KeyDistant"
        )
        distant.CreateIntensityAttr(950.0)
        distant.CreateColorAttr(Gf.Vec3f(1.0, 0.93, 0.84))
        distant.CreateAngleAttr(0.53)
        orient = UsdGeom.Xformable(distant.GetPrim())
        orient.AddRotateXYZOp().Set(Gf.Vec3f(-42.0, 18.0, -28.0))
        return {
            "dome": str(dome.GetPath()),
            "dome_intensity": 120.0,
            "distant": str(distant.GetPath()),
            "distant_intensity": 950.0,
        }

    def _physics_view(self) -> Any:
        view = self.env._robots["right"]._physics_sim_view
        if view is None:
            raise RuntimeError("PhysX simulation view is unavailable")
        return view

    def _owner_views(self, composition: dict[str, Any]) -> dict[str, Any]:
        views = {}
        for name in ("bottle", "cap"):
            path = composition[name]["prim_path"]
            view = self._physics_view().create_rigid_body_view(path)
            if view is None or int(view.count) != 1:
                raise RuntimeError(f"expected one live rigid owner for {name}: {path}")
            views[name] = view
        return views

    @staticmethod
    def _owner_state(view: Any) -> tuple[np.ndarray, np.ndarray]:
        transform = _tensor_numpy(view.get_transforms()).reshape(1, 7)[0].astype(np.float64)
        velocity = _tensor_numpy(view.get_velocities()).reshape(1, 6)[0].astype(np.float64)
        return transform, velocity

    def _step_physics(self, steps: int, views: dict[str, Any]) -> dict[str, Any]:
        peak_linear = {name: 0.0 for name in views}
        peak_angular = {name: 0.0 for name in views}
        finite = True
        for _ in range(int(steps)):
            for robot in self.env._robots.values():
                robot.write_data_to_sim()
            self.sim.step(render=False)
            for robot in self.env._robots.values():
                robot.update(self.env._sim_dt)
            for name, view in views.items():
                transform, velocity = self._owner_state(view)
                finite = bool(finite and np.isfinite(transform).all() and np.isfinite(velocity).all())
                peak_linear[name] = max(peak_linear[name], float(np.linalg.norm(velocity[:3])))
                peak_angular[name] = max(peak_angular[name], float(np.linalg.norm(velocity[3:])))
        return {
            "peak_linear_velocity_mps": peak_linear,
            "peak_angular_velocity_radps": peak_angular,
            "finite": finite,
        }

    def _collision_owners(self, composition: dict[str, Any]) -> dict[str, Any]:
        records = {}
        for name in ("bottle", "cap"):
            expected = composition[name]["prim_path"]
            rows = []
            for path in composition[name]["collision_paths"]:
                prim = self.stage.GetPrimAtPath(path)
                rows.append(
                    {
                        "collision_path": path,
                        "nearest_rigid_owner": _nearest_rigid_owner_path(prim),
                    }
                )
            records[name] = {
                "expected_owner": expected,
                "colliders": rows,
                "pass": bool(rows and all(row["nearest_rigid_owner"] == expected for row in rows)),
            }
        return records

    def _robot_inventory(self) -> dict[str, Any]:
        inventory = {}
        for side in ("left", "right"):
            robot = self.env._robots.get(side)
            if robot is None:
                inventory[side] = {"present": False}
                continue
            inventory[side] = {
                "present": True,
                "body_count": len(robot.body_names),
                "joint_count": len(robot.joint_names),
                "palm_present": f"{side}_palm_link" in set(robot.body_names),
            }
        return {
            "model": "rokae",
            "hands": "wuji_gen1",
            "arms": inventory,
            "pass": all(
                inventory.get(side, {}).get("present")
                and inventory[side]["body_count"] > 20
                and inventory[side]["joint_count"] >= 27
                and inventory[side]["palm_present"]
                for side in ("left", "right")
            ),
        }

    def _robot_object_clearance(self, object_aabbs: dict[str, np.ndarray]) -> dict[str, Any]:
        rows = []
        minimum = float("inf")
        inside_count = 0
        for side, robot in self.env._robots.items():
            body_positions = _tensor_numpy(robot.data.body_pose_w)[0, :, :3].astype(np.float64)
            for body_name, point in zip(robot.body_names, body_positions):
                for object_name, aabb in object_aabbs.items():
                    clamped = np.minimum(np.maximum(point, aabb[0]), aabb[1])
                    distance = float(np.linalg.norm(point - clamped))
                    minimum = min(minimum, distance)
                    inside = bool(np.all(point >= aabb[0]) and np.all(point <= aabb[1]))
                    inside_count += int(inside)
                    if inside or distance < 0.03:
                        rows.append(
                            {
                                "side": side,
                                "body": str(body_name),
                                "object": object_name,
                                "distance_to_aabb_m": distance,
                                "origin_inside_aabb": inside,
                            }
                        )
        return {
            "minimum_body_origin_to_object_aabb_m": minimum,
            "nearby_rows": rows,
            "body_origin_inside_count": inside_count,
            "pass": inside_count == 0,
        }

    def _render_preview(self) -> dict[str, Any]:
        from PIL import Image

        frames = None
        for _ in range(5):
            frames = self.env.render()
        if not isinstance(frames, (list, tuple)) or not frames:
            raise RuntimeError("production head camera returned no frames")
        head = np.asarray(frames[0])
        if head.shape != (RENDER_HEIGHT, RENDER_WIDTH, 3) or head.dtype != np.uint8:
            raise RuntimeError(f"unexpected head frame: {head.shape}/{head.dtype}")
        path = self.output_dir / "reset_head.png"
        Image.fromarray(head).save(path, compress_level=0)
        metrics = _rgb_metrics(head)
        camera = self.env._cameras.get("head_camera")
        camera_path = str(getattr(getattr(camera, "cfg", None), "prim_path", ""))
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "camera_prim_path": camera_path,
            "resolution_wh": [RENDER_WIDTH, RENDER_HEIGHT],
            "metrics": metrics,
            "pass": bool(
                camera_path == HEAD_CAMERA_PATH
                and path.stat().st_size >= 150_000
                and metrics["mean_rgb"] > 30.0
                and metrics["pixel_variance"] > 15.0
                and metrics["black_fraction"] < 0.25
            ),
        }

    def _save_portable_overlay(self, table_top_z: float, cap_data: dict[str, Any]) -> dict[str, Any]:
        from pxr import Gf, Usd, UsdGeom, UsdLux

        path = self.output_dir / "hs014_core_scene.usda"
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        stage.SetDefaultPrim(world)
        relative_bottle = "./assets/dexverse_water_bottle/water_bottle_opened.usd"
        overlay = compose_assets(
            stage,
            table_top_z=table_top_z,
            bottle_reference=relative_bottle,
            cap_data=cap_data,
        )
        # The task-local USD is an overlay on the native HandsOn bimanual
        # runtime. Store that requirement in the file rather than fabricating
        # duplicate robot geometry.
        contract = UsdGeom.Scope.Define(stage, f"{TASK_ROOT}/RuntimeContract").GetPrim()
        contract.SetCustomDataByKey("base_task", "t04_cover_pitcher")
        contract.SetCustomDataByKey(
            "robot_asset", "handson/handson/sim/envs/xmls/rokae_wuji1_bimanual.xml"
        )
        contract.SetCustomDataByKey("camera_prim_path", HEAD_CAMERA_PATH)
        contract.SetCustomDataByKey("task_complete", False)
        lights = UsdGeom.Scope.Define(stage, f"{TASK_ROOT}/Lights")
        dome = UsdLux.DomeLight.Define(stage, f"{lights.GetPath()}/Dome")
        dome.CreateIntensityAttr(120.0)
        dome.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.0))
        distant = UsdLux.DistantLight.Define(stage, f"{lights.GetPath()}/KeyDistant")
        distant.CreateIntensityAttr(950.0)
        distant.CreateColorAttr(Gf.Vec3f(1.0, 0.93, 0.84))
        distant.CreateAngleAttr(0.53)
        stage.GetRootLayer().Save()
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "overlay": overlay,
            "pass": path.stat().st_size > 1024,
        }

    def run(self) -> dict[str, Any]:
        started = time.time()
        assets = source_asset_contract()
        if not assets["pass"]:
            raise RuntimeError("authoritative asset SHA-256 mismatch")
        cap_data = decode_authoritative_cap()
        hidden_seed = self._hide_semantic_seed_objects()
        table_top = float(self.env._current_table_top_z())
        composition = compose_assets(
            self.stage,
            table_top_z=table_top,
            bottle_reference=str(BOTTLE_ASSET.resolve()),
            cap_data=cap_data,
        )
        lighting = self._ensure_lighting()
        self.sim.reset()
        views = self._owner_views(composition)
        owner_contract = self._collision_owners(composition)
        robot = self._robot_inventory()

        # Reset authoring above is the only object-state initialization. The
        # following loop performs no actions and no object pose/velocity writes.
        settle = self._step_physics(PHYSICS_HZ // 2, views)
        baseline = {name: self._owner_state(view)[0] for name, view in views.items()}
        no_action = self._step_physics(10 * PHYSICS_HZ, views)
        final = {name: self._owner_state(view) for name, view in views.items()}

        final_aabbs = {
            "bottle": world_aabb(BOTTLE_LOCAL_AABB_M, final["bottle"][0]),
            "cap": world_aabb(CAP_LOCAL_AABB_M, final["cap"][0]),
        }
        overlap = aabb_overlap_depth(final_aabbs["bottle"], final_aabbs["cap"])
        table_penetration = {
            name: max(0.0, table_top - float(bounds[0, 2]))
            for name, bounds in final_aabbs.items()
        }
        drift = {
            name: {
                "translation_m": float(np.linalg.norm(final[name][0][:3] - baseline[name][:3])),
                "rotation_rad": _quat_angle_rad(final[name][0][3:], baseline[name][3:]),
                "final_linear_velocity_mps": float(np.linalg.norm(final[name][1][:3])),
                "final_angular_velocity_radps": float(np.linalg.norm(final[name][1][3:])),
            }
            for name in ("bottle", "cap")
        }
        stability_pass = bool(
            settle["finite"]
            and no_action["finite"]
            and all(item["translation_m"] <= 0.005 for item in drift.values())
            and all(item["rotation_rad"] <= 0.05 for item in drift.values())
            and all(item["final_linear_velocity_mps"] <= 0.02 for item in drift.values())
            and all(item["final_angular_velocity_radps"] <= 0.10 for item in drift.values())
        )
        clear_initial_state = bool(
            overlap == 0.0 and max(table_penetration.values()) <= 0.005
        )
        robot_clearance = self._robot_object_clearance(final_aabbs)
        preview = self._render_preview()
        overlay = self._save_portable_overlay(table_top, cap_data)
        checks = {
            "authoritative_assets": assets["pass"],
            "identity_scale_and_upright_pose": bool(
                composition["bottle"]["scale"] == [1.0, 1.0, 1.0]
                and composition["cap"]["scale"] == [1.0, 1.0, 1.0]
                and composition["bottle"]["bottle_visual_proxy_axis_aligned"] is True
                and np.allclose(
                    composition["cap"]["orientation_xyzw"],
                    CAP_UPRIGHT_QUAT_XYZW,
                    atol=1.0e-12,
                )
            ),
            "dual_rokae_wuji": robot["pass"],
            "semantic_seed_hidden": hidden_seed["pass"],
            "collision_owner": all(item["pass"] for item in owner_contract.values()),
            "initial_clearance": clear_initial_state and robot_clearance["pass"],
            "zero_action_stability": stability_pass,
            "head_camera_preview": preview["pass"],
            "portable_usd_saved": overlay["pass"],
        }
        return {
            "schema": "hs014_core_scene_report_v1",
            "task_id": TASK_ID,
            "scope": "core_scene_only",
            "exact_production_child": False,
            "priority_zero_qualified": False,
            "constructed_runtime_task": "t04_cover_pitcher",
            "qualification_note": (
                "Asset-only B01 composition; run the exact HS014 production child "
                "for first-reset, dual-arm readback, and priority-zero evidence."
            ),
            "task_complete": False,
            "action_command_count": 0,
            "object_state_writes_during_no_action": 0,
            "ik_executed": False,
            "grasp_executed": False,
            "success_motion_executed": False,
            "physics_hz": PHYSICS_HZ,
            "zero_action_sim_seconds": 10.0,
            "table_top_z_m": table_top,
            "assets": assets,
            "composition": composition,
            "semantic_seed_suppression": hidden_seed,
            "lighting": lighting,
            "robot": robot,
            "collision_owner": owner_contract,
            "final_owner_states": {
                name: {
                    "transform_xyzw": final[name][0].tolist(),
                    "velocity": final[name][1].tolist(),
                    "world_aabb_m": final_aabbs[name].tolist(),
                }
                for name in ("bottle", "cap")
            },
            "bottle_cap_aabb_overlap_m": overlap,
            "table_penetration_m": table_penetration,
            "robot_object_clearance": robot_clearance,
            "settle_metrics": settle,
            "zero_action_metrics": no_action,
            "zero_action_drift": drift,
            "preview": preview,
            "portable_usd": overlay,
            "checks": checks,
            "core_scene_pass": all(checks.values()),
            "elapsed_wall_seconds": time.time() - started,
        }
