"""Plaggy Sponge 3 surface and native PhysX volume body with the YCB mug."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .ycb11_semantics import (
    YCB11_MUG_SOURCE_SIZE_M,
    YCB11_SPONGE_SOURCE_SIZE_M,
    YCB11_TASK_NAME,
)


YCB11_SCENE_XML = "arena_arm_hand_eval_flex01_compress_sponge_into_mug_rokae_wuji1.xml"
YCB11_TABLE_TOP_Z = 0.89
YCB11_MUG_BODY_NAME = "ycb11_mug"
YCB11_SPONGE_ROOT_PATH = "/World/Task/YCB11Sponge"
YCB11_SPONGE_MESH_PATH = f"{YCB11_SPONGE_ROOT_PATH}/mesh"
# The current PhysX volume-deformable pipeline keeps the render/cooking mesh
# separate from the generated simulation and collision TetMeshes.  These must
# be immediate children of the body root for
# the explicit current volume-body hierarchy.
YCB11_SPONGE_SIM_MESH_PATH = f"{YCB11_SPONGE_ROOT_PATH}/simulation_mesh"
YCB11_SPONGE_COLLISION_MESH_PATH = f"{YCB11_SPONGE_ROOT_PATH}/collision_mesh"
YCB11_SPONGE_PHYSICS_MATERIAL_PATH = "/World/PhysicsMaterials/YCB11Sponge"
YCB11_SPONGE_VISUAL_MATERIAL_PATH = "/World/Looks/YCB11Sponge3Texture"
# Both physical volumes derive from the selected source surface. The render
# mesh retains the original UVs and source scale (after unit conversion).
YCB11_SPONGE_SELF_COLLISION = False
# Isotropic, freely compressible foam proxy, not a measured material
# identification. A low Young's modulus and near-zero Poisson ratio let a
# contact normal in X, Y, or Z shorten the volume instead of transferring most
# of the load into lateral expansion. The same scalar material is used for all
# tetrahedra; there is no preferred compression axis.
YCB11_SPONGE_YOUNGS_MODULUS_PA = 5_000.0
YCB11_SPONGE_POISSONS_RATIO = 0.05
# Keep the isotropic low-stiffness response. This damping is velocity loss only;
# it does not introduce a preferred compression axis.
YCB11_SPONGE_ELASTICITY_DAMPING = 0.02
# Cap reset-contact correction velocity so the deformable does not inject a
# large first-frame impulse into the formal recording-origin gate.
YCB11_SPONGE_MAX_DEPENETRATION_VELOCITY_MPS = 0.05
# Keep the fixed 120 Hz collection budget while the material's lower stiffness
# and damping provide the all-axis compliance.
YCB11_SPONGE_SOLVER_ITERATIONS = 32

YCB11_MUG_NOMINAL_POSITION = np.asarray(
    [-0.24, 0.12, YCB11_TABLE_TOP_Z + 0.0407645], dtype=np.float64
)
# Keep the source-scale mug wakeable and easy to manipulate while removing
# the long-lived contact jitter caused by PhysX's much wider default mesh
# contact shell.  In particular, the default shell can lift the flat 81.5 mm
# mug more than a centimetre above the table and tip it before origin capture.
YCB11_MUG_RESET_LINEAR_DAMPING = 0.12
YCB11_MUG_RESET_ANGULAR_DAMPING = 0.80
YCB11_MUG_RESET_SLEEP_THRESHOLD = 0.005
YCB11_MUG_CONTACT_OFFSET_M = 0.0010
YCB11_MUG_REST_OFFSET_M = 0.0
# The checked-in Sponge 3 volume spans z=-15 mm..+15 mm in local space. Keep
# its lowest collision nodes exactly at the table's 0.5 mm rest offset so the
# first formal reset does not spend its first solver frame closing a 1.5 mm
# gap and then get frozen by the recording-origin velocity boundary.
YCB11_SPONGE_TABLE_REST_CLEARANCE_M = 0.0005
# Exact bounds of the tracked YCB mug's source Collider9 bottom puck.  The
# runtime MJCF expresses this regular 32-gon as its analytic cylinder limit so
# table support has one smooth contact surface instead of 32 facet seams.
YCB11_MUG_SUPPORT_CENTER_LOCAL_M = np.asarray(
    [0.001814, 0.014081, -0.0356525], dtype=np.float64
)
YCB11_MUG_SUPPORT_RADIUS_M = 0.037510
YCB11_MUG_SUPPORT_HALF_HEIGHT_M = 0.005112
YCB11_SPONGE_NOMINAL_POSITION = np.asarray(
    # Keep the sponge under the right fingertips' nonsingular low-speed
    # workspace.  The former y=-0.13 pose required wrist momentum to bridge
    # the last few millimetres, causing the first contact to crush a cell.
    [
        -0.24,
        -0.05,
        YCB11_TABLE_TOP_Z + 0.015 + YCB11_SPONGE_TABLE_REST_CLEARANCE_M,
    ], dtype=np.float64
)
YCB11_SPONGE_NOMINAL_YAW_RAD = math.radians(0.0)

# Provenance locks for the selected Sponge 3 surface/albedo and shared YCB mug.
YCB11_SPONGE_OBJ_SHA256 = (
    "433a1dcab8c927e109c228c4138ca42b19f742dfe587eb123c8b942479b13dde"
)
YCB11_SPONGE_TEXTURE_SHA256 = (
    "16a4aaa9f7a72861a8c3060e7c56de9a62d9308983a822dfbaa16fbac93b9e3d"
)
YCB11_MUG_OBJ_SHA256 = (
    "94cc099c39064ea5bde3757183146071984636e4d335f52d80f94d1ed97410f2"
)

YCB11_SUCCESS_CONDITION_LABELS = (
    "Left hand physically stabilizes the original-scale YCB mug for 0.25 seconds",
    "Right thumb and opposing fingers physically grasp and lift the Sponge 3",
    "The sponge material-axis span is compressed to at most 82 percent before rim entry",
    "At least 85 percent of sponge nodes are in the mug and 80 percent are below its rim",
    "Both hands release; the upright mug and packed sponge remain stable for 1 second",
)


def _xml_root() -> Path:
    return Path(__file__).resolve().parent.parent / "envs" / "xmls"


def ycb11_asset_paths() -> dict[str, Path]:
    root = _xml_root() / "lightwheel_ycb"
    sponge = _xml_root() / "assets/plaggy_sponge3"
    return {
        "sponge_model": sponge / "source/scene.gltf",
        "sponge_obj": sponge / "sponge.obj",
        "sponge_texture": sponge / "source/textures/Sponge3_baseColor.png",
        "sponge_physics": sponge / "sponge_tets.npz",
        "sponge_license": sponge / "source/license.txt",
        "mug_model": root / "mug/model.xml",
        "mug_obj": root / "mug/visuals/mug.obj",
        "mug_texture": root / "mug/visuals/T_mug_BC001.png",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class YCBObjMesh:
    points: np.ndarray
    face_counts: np.ndarray
    face_indices: np.ndarray
    face_uvs: np.ndarray


def _obj_index(token: str, size: int) -> int:
    value = int(token)
    index = value - 1 if value > 0 else size + value
    if index < 0 or index >= size:
        raise ValueError(f"OBJ index {value} is outside an array of size {size}")
    return index


def read_ycb_obj(path: Path) -> YCBObjMesh:
    """Read the selected sponge OBJ while preserving shared watertight vertices."""

    vertices: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    corner_uvs: list[tuple[float, float]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if fields[0] == "v" and len(fields) >= 4:
                vertices.append(tuple(float(value) for value in fields[1:4]))
            elif fields[0] == "vt" and len(fields) >= 3:
                texcoords.append((float(fields[1]), float(fields[2])))
            elif fields[0] == "f":
                if len(fields) != 4:
                    raise ValueError(
                        f"YCB11 requires triangulated OBJ faces; {path}:{line_number}"
                    )
                triangle: list[int] = []
                triangle_uvs: list[tuple[float, float]] = []
                for corner in fields[1:]:
                    components = corner.split("/")
                    triangle.append(_obj_index(components[0], len(vertices)))
                    if len(components) < 2 or not components[1]:
                        raise ValueError(
                            f"YCB11 Sponge 3 is missing UVs at {path}:{line_number}"
                        )
                    triangle_uvs.append(
                        texcoords[_obj_index(components[1], len(texcoords))]
                    )
                triangles.append(tuple(triangle))
                corner_uvs.extend(triangle_uvs)
    points = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int32)
    uvs = np.asarray(corner_uvs, dtype=np.float32)
    if points.shape != (290, 3) or faces.shape != (576, 3):
        raise ValueError(
            "Plaggy Sponge 3 topology changed: "
            f"vertices={points.shape}, faces={faces.shape}"
        )
    if uvs.shape != (3 * faces.shape[0], 2):
        raise ValueError("Plaggy Sponge 3 face-varying UV topology is invalid")
    return YCBObjMesh(
        points=points,
        face_counts=np.full(faces.shape[0], 3, dtype=np.int32),
        face_indices=faces.reshape(-1),
        face_uvs=uvs,
    )


def _edge_manifold_counts(face_indices: np.ndarray) -> np.ndarray:
    faces = np.asarray(face_indices, dtype=np.int64).reshape(-1, 3)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges.sort(axis=1)
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    return counts


def validate_ycb11_assets(*, require_content: bool = True) -> dict[str, Any]:
    """Check source provenance, physical scale, UVs and watertightness."""

    paths = ycb11_asset_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"YCB11 task assets are missing: {missing}")
    result: dict[str, Any] = {"paths": {key: str(value) for key, value in paths.items()}}
    if not require_content:
        return result

    expected_hashes = {
        "sponge_obj": YCB11_SPONGE_OBJ_SHA256,
        "sponge_texture": YCB11_SPONGE_TEXTURE_SHA256,
        "mug_obj": YCB11_MUG_OBJ_SHA256,
    }
    actual_hashes = {key: _sha256(paths[key]) for key in expected_hashes}
    mismatches = {
        key: (expected_hashes[key], actual_hashes[key])
        for key in expected_hashes
        if actual_hashes[key] != expected_hashes[key]
    }
    if mismatches:
        raise ValueError(f"YCB11 source provenance hash mismatch: {mismatches}")

    sponge = read_ycb_obj(paths["sponge_obj"])
    sponge_size = np.ptp(sponge.points, axis=0)
    if not np.allclose(sponge_size, YCB11_SPONGE_SOURCE_SIZE_M, atol=5e-6):
        raise ValueError(
            "YCB11 sponge must retain source dimensions: "
            f"actual={sponge_size.tolist()}, expected={YCB11_SPONGE_SOURCE_SIZE_M.tolist()}"
        )
    edge_counts = _edge_manifold_counts(sponge.face_indices)
    if not bool(np.all(edge_counts == 2)):
        raise ValueError(
            "YCB11 Sponge 3 render surface is not watertight and cannot be volume cooked"
        )
    result.update(
        {
            "sha256": actual_hashes,
            "sponge_size_m": sponge_size.tolist(),
            "mug_size_m": YCB11_MUG_SOURCE_SIZE_M.tolist(),
            "sponge_vertices": int(sponge.points.shape[0]),
            "sponge_triangles": int(sponge.face_counts.shape[0]),
            "watertight": True,
            "scale": [1.0, 1.0, 1.0],
        }
    )
    return result


def _transform_points(points: np.ndarray, position: np.ndarray, yaw: float) -> np.ndarray:
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return np.asarray(points, dtype=np.float64) @ rotation.T + np.asarray(position)


def _author_ycb_texture_material(stage: Any, mesh_prim: Any, texture_path: Path) -> None:
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, YCB11_SPONGE_VISUAL_MATERIAL_PATH)
    shader = UsdShade.Shader.Define(
        stage, f"{YCB11_SPONGE_VISUAL_MATERIAL_PATH}/PreviewSurface"
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.78)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    texture = UsdShade.Shader.Define(
        stage, f"{YCB11_SPONGE_VISUAL_MATERIAL_PATH}/DiffuseTexture"
    )
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(texture_path.resolve()))
    )
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    # The source UV atlas reaches the texture edge; avoid opposite-edge bleed.
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    st_reader = UsdShade.Shader.Define(
        stage, f"{YCB11_SPONGE_VISUAL_MATERIAL_PATH}/PrimvarReader"
    )
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    )
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    # Preserve the selected glTF's roughness/metalness and tangent-space normal.
    for name, filename in (
        ("MetallicRoughness", "Sponge3_metallicRoughness.png"),
        ("Normal", "Sponge3_normal.png"),
    ):
        sampler = UsdShade.Shader.Define(stage, f"{YCB11_SPONGE_VISUAL_MATERIAL_PATH}/{name}")
        sampler.CreateIdAttr("UsdUVTexture")
        sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(texture_path.parent / filename))
        )
        sampler.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st_reader.GetOutput("result")
        )
        if name == "Normal":
            sampler.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((2.0, 2.0, 2.0, 1.0))
            sampler.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((-1.0, -1.0, -1.0, 0.0))
            shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                sampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            )
        else:
            shader.GetInput("roughness").ConnectToSource(sampler.CreateOutput("g", Sdf.ValueTypeNames.Float))
            shader.GetInput("metallic").ConnectToSource(sampler.CreateOutput("b", Sdf.ValueTypeNames.Float))
    material.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )

    from .ycb11_visual_materials import bind_ycb11_mug_glaze

    bind_ycb11_mug_glaze(stage, ycb11_asset_paths()["mug_texture"])


def spawn_ycb11_volume_deformable_sponge(stage: Any, sim_utils: Any) -> Any:
    """Author the source YCB skin over an explicit current volume body.

    Isaac Sim 5.1 exposes two deformable schema generations.  The old
    ``isaaclab.sim.schemas.define_deformable_body_properties`` helper applies
    ``PhysxDeformableBodyAPI`` to a surface mesh and consequently cannot be
    consumed by the successor ``create_volume_deformable_body_view`` tensor
    API.  YCB11 uses the current explicit tetrahedral hierarchy: an
    Xform body root, a source render mesh, and immediate simulation/collision
    TetMesh children with ``OmniPhysics*`` schemas.
    """

    del sim_utils
    from omni.physx.scripts import deformableUtils, physicsUtils
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    validation = validate_ycb11_assets(require_content=True)
    paths = ycb11_asset_paths()
    source = read_ycb_obj(paths["sponge_obj"])
    points = _transform_points(
        source.points,
        YCB11_SPONGE_NOMINAL_POSITION,
        YCB11_SPONGE_NOMINAL_YAW_RAD,
    )

    root = UsdGeom.Xform.Define(stage, YCB11_SPONGE_ROOT_PATH)
    root_prim = root.GetPrim()
    root_prim.SetCustomDataByKey("handson:taskId", "YCB11")
    root_prim.SetCustomDataByKey("handson:assetType", "source_surface_volume_deformable")
    root_prim.SetCustomDataByKey("handson:ycbSource", "assets/plaggy_sponge3/source/scene.gltf")
    root_prim.SetCustomDataByKey("handson:ycbSourceSha256", YCB11_SPONGE_OBJ_SHA256)
    root_prim.SetCustomDataByKey("handson:sourceScale", "1.0 1.0 1.0")
    root_prim.SetCustomDataByKey("handson:runtimeRepresentation", "physx_cooked_tetrahedral_volume")

    mesh = UsdGeom.Mesh.Define(stage, YCB11_SPONGE_MESH_PATH)
    mesh.CreatePointsAttr().Set(
        [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points]
    )
    mesh.CreateFaceVertexCountsAttr().Set(source.face_counts.tolist())
    mesh.CreateFaceVertexIndicesAttr().Set(source.face_indices.tolist())
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(False)
    mesh.CreateExtentAttr().Set(
        [
            Gf.Vec3f(*np.min(points, axis=0).astype(float).tolist()),
            Gf.Vec3f(*np.max(points, axis=0).astype(float).tolist()),
        ]
    )
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    st.Set([Gf.Vec2f(float(u), float(v)) for u, v in source.face_uvs])
    _author_ycb_texture_material(stage, mesh.GetPrim(), paths["sponge_texture"])

    # Coarse FEM and denser contact fill the same source-outline volume.
    # Every contact node lies in the FEM; contact loads are interpolated,
    # not extrapolated. The original source remains the textured skin.
    with np.load(paths["sponge_physics"], allow_pickle=False) as physical:
        nodal_points = _transform_points(physical["points"], YCB11_SPONGE_NOMINAL_POSITION, YCB11_SPONGE_NOMINAL_YAW_RAD)
        collision_points = _transform_points(physical["collision_points"], YCB11_SPONGE_NOMINAL_POSITION, YCB11_SPONGE_NOMINAL_YAW_RAD)
        tetrahedra = physical["tets"]
        faces = physical["faces"]
        collision_tets = physical["collision_tets"]
        collision_faces = physical["collision_faces"]
    root_prim.ApplyAPI("OmniPhysicsDeformableBodyAPI")
    for path, is_simulation in ((YCB11_SPONGE_SIM_MESH_PATH, True), (YCB11_SPONGE_COLLISION_MESH_PATH, False)):
        vertices = [Gf.Vec3f(*map(float, point)) for point in (nodal_points if is_simulation else collision_points)]
        indices = [Gf.Vec4i(*map(int, tet)) for tet in (tetrahedra if is_simulation else collision_tets)]
        surface_faces = faces if is_simulation else collision_faces
        tet_mesh = UsdGeom.TetMesh.Define(stage, path)
        tet_mesh.CreatePointsAttr(vertices)
        tet_mesh.CreateVelocitiesAttr([Gf.Vec3f(0.0)] * len(vertices))
        tet_mesh.CreateOrientationAttr("rightHanded")
        tet_mesh.CreateTetVertexIndicesAttr(indices)
        tet_mesh.CreateSurfaceFaceVertexIndicesAttr([Gf.Vec3i(*map(int, face)) for face in surface_faces])
        prim = tet_mesh.GetPrim()
        if is_simulation:
            prim.ApplyAPI("OmniPhysicsVolumeDeformableSimAPI")
            prim.GetAttribute("omniphysics:restShapePoints").Set(vertices)
            prim.GetAttribute("omniphysics:restTetVtxIndices").Set(indices)
        else:
            UsdPhysics.CollisionAPI.Apply(prim)
        prim.ApplyAPI("OmniPhysicsDeformablePoseAPI", "default")
        prim.CreateAttribute("deformablePose:default:omniphysics:purposes", Sdf.ValueTypeNames.TokenArray).Set(["bindPose"])
        prim.CreateAttribute("deformablePose:default:omniphysics:points", Sdf.ValueTypeNames.Point3fArray).Set(vertices)
    skin = mesh.GetPrim()
    skin.ApplyAPI("OmniPhysicsDeformablePoseAPI", "default")
    skin.CreateAttribute("deformablePose:default:omniphysics:purposes", Sdf.ValueTypeNames.TokenArray).Set(["bindPose"])
    skin.CreateAttribute("deformablePose:default:omniphysics:points", Sdf.ValueTypeNames.Point3fArray).Set(mesh.GetPointsAttr().Get())

    if not deformableUtils.add_deformable_material(
        stage,
        YCB11_SPONGE_PHYSICS_MATERIAL_PATH,
        density=35.0,
        static_friction=0.90,
        dynamic_friction=0.90,
        youngs_modulus=YCB11_SPONGE_YOUNGS_MODULUS_PA,
        poissons_ratio=YCB11_SPONGE_POISSONS_RATIO,
    ):
        raise RuntimeError("Could not create the YCB11 deformable material")
    root_prim = stage.GetPrimAtPath(YCB11_SPONGE_ROOT_PATH)
    physicsUtils.add_physics_material_to_prim(
        stage,
        root_prim,
        YCB11_SPONGE_PHYSICS_MATERIAL_PATH,
    )
    material_prim = stage.GetPrimAtPath(YCB11_SPONGE_PHYSICS_MATERIAL_PATH)
    material_prim.ApplyAPI("PhysxDeformableMaterialAPI")
    material_prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(
        YCB11_SPONGE_ELASTICITY_DAMPING
    )

    # Body-level limits are current PhysX base attributes.  They keep the
    # source-scale sponge bounded while it is being grasped and also avoid a
    # large first-frame impulse if the mug is touched during reset.
    root_prim.ApplyAPI("PhysxBaseDeformableBodyAPI")
    body_attrs = {
        "physxDeformableBody:selfCollision": YCB11_SPONGE_SELF_COLLISION,
        "physxDeformableBody:enableSpeculativeCCD": True,
        "physxDeformableBody:solverPositionIterationCount": YCB11_SPONGE_SOLVER_ITERATIONS,
        "physxDeformableBody:maxLinearVelocity": 2.0,
        "physxDeformableBody:maxDepenetrationVelocity": (
            YCB11_SPONGE_MAX_DEPENETRATION_VELOCITY_MPS
        ),
        "physxDeformableBody:linearDamping": 0.08,
    }
    for attr_name, value in body_attrs.items():
        attribute = root_prim.GetAttribute(attr_name)
        if not attribute.IsValid():
            raise RuntimeError(f"YCB11 current body attribute is missing: {attr_name}")
        attribute.Set(value)

    collision_prim = stage.GetPrimAtPath(YCB11_SPONGE_COLLISION_MESH_PATH)
    if collision_prim.IsValid():
        # These are harmless on versions where the generated collision mesh
        # does not expose the optional PhysX collision tuning schema.
        collision_api = collision_prim.ApplyAPI("PhysxCollisionAPI")
        if collision_api:
            collision_prim.GetAttribute("physxCollision:contactOffset").Set(0.0020)
            collision_prim.GetAttribute("physxCollision:restOffset").Set(0.0005)

    # Guide purpose alone still leaves debug TetMeshes visible in composed
    # stage audits and some viewports. Only the textured source is rendered.
    for path in (YCB11_SPONGE_SIM_MESH_PATH, YCB11_SPONGE_COLLISION_MESH_PATH):
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
    root_prim.SetCustomDataByKey(
        "handson:sourceValidationJson",
        json.dumps(validation, sort_keys=True, separators=(",", ":")),
    )
    return root.GetPath()


__all__ = [
    "YCB11_MUG_BODY_NAME",
    "YCB11_MUG_CONTACT_OFFSET_M",
    "YCB11_MUG_NOMINAL_POSITION",
    "YCB11_MUG_RESET_ANGULAR_DAMPING",
    "YCB11_MUG_RESET_LINEAR_DAMPING",
    "YCB11_MUG_RESET_SLEEP_THRESHOLD",
    "YCB11_MUG_REST_OFFSET_M",
    "YCB11_MUG_SUPPORT_CENTER_LOCAL_M",
    "YCB11_MUG_SUPPORT_HALF_HEIGHT_M",
    "YCB11_MUG_SUPPORT_RADIUS_M",
    "YCB11_SCENE_XML",
    "YCB11_SPONGE_MESH_PATH",
    "YCB11_SPONGE_ELASTICITY_DAMPING",
    "YCB11_SPONGE_POISSONS_RATIO",
    "YCB11_SPONGE_SELF_COLLISION",
    "YCB11_SPONGE_SIM_MESH_PATH",
    "YCB11_SPONGE_COLLISION_MESH_PATH",
    "YCB11_SPONGE_NOMINAL_POSITION",
    "YCB11_SPONGE_TABLE_REST_CLEARANCE_M",
    "YCB11_SPONGE_OBJ_SHA256",
    "YCB11_SPONGE_PHYSICS_MATERIAL_PATH",
    "YCB11_SPONGE_YOUNGS_MODULUS_PA",
    "YCB11_SPONGE_ROOT_PATH",
    "YCB11_SUCCESS_CONDITION_LABELS",
    "YCB11_TABLE_TOP_Z",
    "YCB11_TASK_NAME",
    "read_ycb_obj",
    "spawn_ycb11_volume_deformable_sponge",
    "validate_ycb11_assets",
    "ycb11_asset_paths",
]
