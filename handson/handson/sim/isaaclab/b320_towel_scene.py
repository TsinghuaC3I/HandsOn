"""BC025 Terry Cloth surface-deformable scene authoring.

The downloaded Terry Cloth archive is used as a visual material source.  The
task owns the rectangular mid-surface so the render, simulation, and collision
topologies remain one exact mesh.  Pillow024 is consulted for collision and
material calibration; its tetrahedral pillow geometry is never composed into
this task.
"""

from __future__ import annotations

import math
from typing import Any

from .b320_bimanual_scene import (
    B320_ASSET_ROOT_PATH,
    B320_PHYSICS_HZ,
    B320_TABLE_TOP_Z,
    B320_TOWEL_TASK_NAME,
    _spawn_goal_marker,
    _spawn_worktable,
)
from .b320_bc025_assets import (
    BC025_ASSET_ID,
    BC025_COLLISION_SOURCE_TITLE,
    BC025_MATERIAL_TITLE,
    BC025_PILLOW_COLLISION_CALIBRATION,
    author_bc025_terry_cloth_material,
    bc025_static_asset_manifest,
    inspect_pillow024_collision_source,
    resolve_bc025_assets,
)


# Keep all BC025 paths task-owned so an accidental legacy reference is easy to
# detect in a composed stage and in static source audits.
BC025_TOWEL_ROOT_PATH = f"{B320_ASSET_ROOT_PATH}/BC025TerryCloth"
# PhysX requires the simulation mesh to be an immediate child of the prim
# carrying OmniPhysicsDeformableBodyAPI.  Keep the task-owned body and the
# rendered/simulated/colliding mesh in one two-prim hierarchy; an extra
# presentation Xform here would make the beta parser reject the body.
BC025_TOWEL_SURFACE_ROOT_PATH = BC025_TOWEL_ROOT_PATH
BC025_TOWEL_SURFACE_MESH_PATH = f"{BC025_TOWEL_ROOT_PATH}/mesh"
BC025_TOWEL_MATERIAL_PATH = "/World/PhysicsMaterials/BC025TerryClothSurface"
BC025_TOWEL_VISUAL_MATERIAL_PATH = "/World/Looks/BC025TerryCloth"

# A deterministic 41x41 grid is small enough for the real-time surface view,
# while retaining enough edge resolution for two independent corner contacts.
BC025_TOWEL_COLUMNS = 41
BC025_TOWEL_ROWS = 41
BC025_TOWEL_NODE_COUNT = BC025_TOWEL_COLUMNS * BC025_TOWEL_ROWS
BC025_TOWEL_TRIANGLE_COUNT = (
    2 * (BC025_TOWEL_COLUMNS - 1) * (BC025_TOWEL_ROWS - 1)
)
BC025_TOWEL_SIZE_X = 0.280
BC025_TOWEL_SIZE_Y = 0.260
BC025_TOWEL_WRINKLE_AMPLITUDE = 0.0035
# The towel is the only free task object in this scene.  Keep the reset
# translation small enough to preserve the two-hand home clearance and the
# camera framing, while making both horizontal axes genuinely seed-dependent.
# Bounds are expressed as offsets from the settled canonical surface baseline.
BC025_TOWEL_TRANSLATION_BOUNDS_M = ((-0.030, 0.030), (-0.030, 0.030))

# The source Pillow024 USD is a volume-deformable asset, so its generic
# Young's modulus/density fields cannot be copied as a sheet density without
# changing units.  Collision offsets, friction, damping, and solver controls
# are copied directly; surface stiffness/density are explicit BC025 sheet
# values and are recorded as an adaptation in the manifest.
BC025_TOWEL_CONTACT_OFFSET = float(
    BC025_PILLOW_COLLISION_CALIBRATION["contact_offset_m"]
)
BC025_TOWEL_REST_OFFSET = float(
    BC025_PILLOW_COLLISION_CALIBRATION["rest_offset_m"]
)
BC025_TOWEL_SURFACE_THICKNESS = BC025_TOWEL_REST_OFFSET
BC025_TOWEL_SURFACE_DENSITY = 720.0
BC025_TOWEL_STATIC_FRICTION = float(
    BC025_PILLOW_COLLISION_CALIBRATION["static_friction"]
)
BC025_TOWEL_DYNAMIC_FRICTION = float(
    BC025_PILLOW_COLLISION_CALIBRATION["dynamic_friction"]
)
BC025_TOWEL_YOUNGS_MODULUS = float(
    BC025_PILLOW_COLLISION_CALIBRATION["youngs_modulus_pa"]
)
BC025_TOWEL_POISSONS_RATIO = float(
    BC025_PILLOW_COLLISION_CALIBRATION["poissons_ratio"]
)
BC025_TOWEL_STRETCH_STIFFNESS = 5.0e3
BC025_TOWEL_SHEAR_STIFFNESS = 50.0
BC025_TOWEL_BEND_STIFFNESS = 0.3
BC025_TOWEL_ELASTICITY_DAMPING = float(
    BC025_PILLOW_COLLISION_CALIBRATION["elasticity_damping"]
)
BC025_TOWEL_SOLVER_POSITION_ITERATIONS = int(
    BC025_PILLOW_COLLISION_CALIBRATION["solver_position_iterations"]
)
BC025_TOWEL_LINEAR_DAMPING = float(
    BC025_PILLOW_COLLISION_CALIBRATION["linear_damping"]
)
BC025_TOWEL_MAX_LINEAR_VELOCITY = BC025_TOWEL_CONTACT_OFFSET * B320_PHYSICS_HZ
BC025_TOWEL_MAX_DEPENETRATION_VELOCITY = 0.25

B320_TOWEL_PLACEMENT = (-0.05, 0.0, B320_TABLE_TOP_Z + BC025_TOWEL_REST_OFFSET)
B320_TOWEL_TARGET_CENTER_XY = (-0.05, 0.0)
B320_TOWEL_TARGET_SIZE_XY = (0.240, 0.235)
# The assigned upper corners span the target frame's X dimension in the task
# frame.  Keep the success target derived from the authored goal geometry
# rather than duplicating a second unexplained distance constant.
BC025_TOWEL_TARGET_END_DISTANCE_M = B320_TOWEL_TARGET_SIZE_XY[0]


def bc025_towel_surface_topology(
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[int, int, int], ...],
]:
    """Return the deterministic, slightly gathered BC025 rest surface."""

    points: list[tuple[float, float, float]] = []
    x_min = -0.5 * BC025_TOWEL_SIZE_X
    y_min = -0.5 * BC025_TOWEL_SIZE_Y
    for row in range(BC025_TOWEL_ROWS):
        v = row / float(BC025_TOWEL_ROWS - 1)
        y = y_min + v * BC025_TOWEL_SIZE_Y
        for column in range(BC025_TOWEL_COLUMNS):
            u = column / float(BC025_TOWEL_COLUMNS - 1)
            x = x_min + u * BC025_TOWEL_SIZE_X
            envelope = math.sin(math.pi * u) * math.sin(math.pi * v)
            wrinkle = 0.5 + 0.5 * math.sin(4.0 * math.pi * u) * math.sin(
                4.0 * math.pi * v
            )
            # A small transverse gather makes the reset state visibly
            # recoverable without changing the exact corner footprint.
            gather_x = 0.0015 * envelope * math.sin(2.0 * math.pi * v)
            gather_y = 0.0010 * envelope * math.sin(2.0 * math.pi * u)
            z = BC025_TOWEL_WRINKLE_AMPLITUDE * envelope * wrinkle
            points.append((x + gather_x, y + gather_y, z))

    triangles: list[tuple[int, int, int]] = []
    for row in range(BC025_TOWEL_ROWS - 1):
        for column in range(BC025_TOWEL_COLUMNS - 1):
            a = row * BC025_TOWEL_COLUMNS + column
            b = a + 1
            c = a + BC025_TOWEL_COLUMNS
            d = c + 1
            triangles.extend(((a, b, d), (a, d, c)))
    return tuple(points), tuple(triangles)


def b320_towel_surface_metadata() -> dict[str, Any]:
    """Return the exact BC025 topology/provenance contract."""

    return {
        "label": "HandsOn/BC025/TerryCloth/41x41Surface",
        "asset_id": BC025_ASSET_ID,
        "visual_vertex_count": BC025_TOWEL_NODE_COUNT,
        "visual_triangle_count": BC025_TOWEL_TRIANGLE_COUNT,
        "sim_vertex_count": BC025_TOWEL_NODE_COUNT,
        "sim_triangle_count": BC025_TOWEL_TRIANGLE_COUNT,
        "surface_size_m": (BC025_TOWEL_SIZE_X, BC025_TOWEL_SIZE_Y),
        "target_end_distance_m": BC025_TOWEL_TARGET_END_DISTANCE_M,
        "corner_indices": {
            "bottom_left": 0,
            "bottom_right": BC025_TOWEL_COLUMNS - 1,
            "top_left": (BC025_TOWEL_ROWS - 1) * BC025_TOWEL_COLUMNS,
            "top_right": BC025_TOWEL_NODE_COUNT - 1,
        },
        "material_source": BC025_MATERIAL_TITLE,
        "collision_source": BC025_COLLISION_SOURCE_TITLE,
        "collision_source_use": "parameters_only; pillow_volume_not_composed",
        "asset_manifest": bc025_static_asset_manifest(),
    }


def _set_extent(mesh: Any, points: tuple[tuple[float, float, float], ...]) -> None:
    from pxr import Gf

    minimum = tuple(min(point[index] for point in points) for index in range(3))
    maximum = tuple(max(point[index] for point in points) for index in range(3))
    mesh.CreateExtentAttr().Set(
        [
            Gf.Vec3f(*map(float, minimum)),
            Gf.Vec3f(*map(float, maximum)),
        ]
    )


def _set_surface_body_attribute(root_prim: Any, name: str, value: Any) -> None:
    attribute = root_prim.GetAttribute(name)
    if not attribute.IsValid():
        raise RuntimeError(f"Missing BC025 surface-deformable attribute {name}")
    attribute.Set(value)


def _author_bc025_surface_mesh(stage: Any) -> Any:
    from pxr import Gf, Sdf, UsdGeom

    points, triangles = bc025_towel_surface_topology()
    mesh = UsdGeom.Mesh.Define(stage, BC025_TOWEL_SURFACE_MESH_PATH)
    mesh_points = [Gf.Vec3f(*map(float, point)) for point in points]
    mesh.GetPointsAttr().Set(mesh_points)
    mesh.GetFaceVertexCountsAttr().Set([3] * len(triangles))
    mesh.GetFaceVertexIndicesAttr().Set(
        [index for triangle in triangles for index in triangle]
    )
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(True)
    _set_extent(mesh, points)

    uv_by_node = [
        Gf.Vec2f(
            (point[0] + 0.5 * BC025_TOWEL_SIZE_X) / BC025_TOWEL_SIZE_X,
            (point[1] + 0.5 * BC025_TOWEL_SIZE_Y) / BC025_TOWEL_SIZE_Y,
        )
        for point in points
    ]
    flat_indices = [index for triangle in triangles for index in triangle]
    uv_primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    uv_primvar.Set([uv_by_node[index] for index in flat_indices])
    return mesh


def _author_bc025_surface_physics(
    stage: Any,
    root_prim: Any,
    mesh_prim: Any,
    collision: dict[str, Any],
) -> None:
    from omni.physx.scripts import deformableUtils, physicsUtils
    from pxr import Gf, PhysxSchema, UsdPhysics

    if not mesh_prim.ApplyAPI("OmniPhysicsSurfaceDeformableSimAPI"):
        raise RuntimeError("Could not apply BC025 surface simulation API")
    if not mesh_prim.ApplyAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("Could not apply BC025 surface collision API")

    mesh_points = mesh_prim.GetAttribute("points").Get()
    if mesh_points is None:
        raise RuntimeError("BC025 surface points were not authored")
    _points, triangles = bc025_towel_surface_topology()
    mesh_prim.GetAttribute("omniphysics:restShapePoints").Set(mesh_points)
    mesh_prim.GetAttribute("omniphysics:restTriVtxIndices").Set(
        [Gf.Vec3i(*triangle) for triangle in triangles]
    )

    if not root_prim.ApplyAPI("OmniPhysicsDeformableBodyAPI"):
        raise RuntimeError("Could not apply BC025 deformable-body API")
    if not root_prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI"):
        raise RuntimeError("Could not apply BC025 PhysX surface API")

    _set_surface_body_attribute(
        root_prim,
        "physxDeformableBody:selfCollision",
        bool(collision["self_collision"]),
    )
    _set_surface_body_attribute(
        root_prim,
        "physxDeformableBody:enableSpeculativeCCD",
        bool(collision["enable_speculative_ccd"]),
    )
    _set_surface_body_attribute(
        root_prim,
        "physxDeformableBody:solverPositionIterationCount",
        int(collision["solver_position_iterations"]),
    )
    _set_surface_body_attribute(
        root_prim,
        "physxDeformableBody:collisionPairUpdateFrequency",
        4,
    )
    _set_surface_body_attribute(
        root_prim,
        "physxDeformableBody:collisionIterationMultiplier",
        4,
    )
    _set_surface_body_attribute(
        root_prim,
        "physxDeformableBody:maxLinearVelocity",
        BC025_TOWEL_MAX_LINEAR_VELOCITY,
    )
    _set_surface_body_attribute(
        root_prim,
        "physxDeformableBody:maxDepenetrationVelocity",
        BC025_TOWEL_MAX_DEPENETRATION_VELOCITY,
    )
    _set_surface_body_attribute(
        root_prim,
        "physxDeformableBody:linearDamping",
        float(collision["linear_damping"]),
    )

    mesh_prim.ApplyAPI(PhysxSchema.PhysxCollisionAPI)
    collision_api = PhysxSchema.PhysxCollisionAPI(mesh_prim)
    collision_api.GetContactOffsetAttr().Set(float(collision["contact_offset_m"]))
    collision_api.GetRestOffsetAttr().Set(float(collision["rest_offset_m"]))

    if not deformableUtils.add_surface_deformable_material(
        stage,
        BC025_TOWEL_MATERIAL_PATH,
        density=BC025_TOWEL_SURFACE_DENSITY,
        static_friction=float(collision["static_friction"]),
        dynamic_friction=float(collision["dynamic_friction"]),
        youngs_modulus=float(collision["youngs_modulus_pa"]),
        poissons_ratio=float(collision["poissons_ratio"]),
        surface_thickness=BC025_TOWEL_SURFACE_THICKNESS,
        surface_stretch_stiffness=BC025_TOWEL_STRETCH_STIFFNESS,
        surface_shear_stiffness=BC025_TOWEL_SHEAR_STIFFNESS,
        surface_bend_stiffness=BC025_TOWEL_BEND_STIFFNESS,
    ):
        raise RuntimeError("Could not create BC025 surface material")
    physicsUtils.add_physics_material_to_prim(
        stage,
        root_prim,
        BC025_TOWEL_MATERIAL_PATH,
    )
    material_prim = stage.GetPrimAtPath(BC025_TOWEL_MATERIAL_PATH)
    material_prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
    material_prim.GetAttribute(
        "physxDeformableMaterial:elasticityDamping"
    ).Set(float(collision["elasticity_damping"]))
    material_prim.GetAttribute("physxDeformableMaterial:bendDamping").Set(
        float(collision["elasticity_damping"])
    )

    # Put the source audit on the live prim so a USD-only reviewer can see
    # exactly which values came from Pillow024 and which values are BC025
    # surface adaptations.
    for name, value in {
        "handson:collisionSource": BC025_COLLISION_SOURCE_TITLE,
        "handson:collisionSourceUse": "parameters_only",
        "handson:pillowContactOffsetM": float(collision["contact_offset_m"]),
        "handson:pillowRestOffsetM": float(collision["rest_offset_m"]),
        "handson:pillowStaticFriction": float(collision["static_friction"]),
        "handson:pillowDynamicFriction": float(collision["dynamic_friction"]),
        "handson:pillowYoungsModulusPa": float(collision["youngs_modulus_pa"]),
        "handson:pillowPoissonsRatio": float(collision["poissons_ratio"]),
        "handson:pillowSolverPositionIterations": int(
            collision["solver_position_iterations"]
        ),
        "handson:pillowSelfCollision": bool(collision["self_collision"]),
        "handson:pillowSpeculativeCCD": bool(
            collision["enable_speculative_ccd"]
        ),
        "handson:pillowLinearDamping": float(collision["linear_damping"]),
        "handson:pillowElasticityDamping": float(
            collision["elasticity_damping"]
        ),
        "handson:surfaceDensityAdaptation": BC025_TOWEL_SURFACE_DENSITY,
        "handson:surfaceStiffnessAdaptation": BC025_TOWEL_STRETCH_STIFFNESS,
    }.items():
        root_prim.SetCustomDataByKey(name, value)


def spawn_b320_towel_surface(stage: Any, sim_utils: Any) -> str:
    """Author the BC025 surface, material, collision, table, and target."""

    import carb
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    if not carb.settings.get_settings().get_as_bool(
        "/persistent/physics/enableDeformableBeta"
    ):
        raise RuntimeError(
            "B320-BC025 requires /persistent/physics/enableDeformableBeta=true"
        )
    if sim_utils is None:
        raise RuntimeError("BC025 scene author requires IsaacLab sim_utils")

    assets = resolve_bc025_assets(required=True)
    assert assets is not None
    collision = inspect_pillow024_collision_source(assets.collision_source_path)

    benchmark_root = UsdGeom.Xform.Define(stage, B320_ASSET_ROOT_PATH).GetPrim()
    benchmark_root.SetCustomDataByKey("handson:task", B320_TOWEL_TASK_NAME)
    benchmark_root.SetCustomDataByKey("handson:taskId", "B320-BC025")

    root = UsdGeom.Xform.Define(stage, BC025_TOWEL_ROOT_PATH)
    root_prim = root.GetPrim()
    root.AddTranslateOp().Set(Gf.Vec3d(*B320_TOWEL_PLACEMENT))
    root_prim.SetCustomDataByKey("handson:task", B320_TOWEL_TASK_NAME)
    root_prim.SetCustomDataByKey("handson:taskId", "B320-BC025")
    root_prim.SetCustomDataByKey("handson:assetId", BC025_ASSET_ID)
    root_prim.SetCustomDataByKey(
        "handson:assetType", "surface_deformable_towel"
    )
    root_prim.SetCustomDataByKey("handson:materialSource", BC025_MATERIAL_TITLE)
    root_prim.SetCustomDataByKey(
        "handson:materialArchiveSha256", assets.archive_sha256
    )
    root_prim.SetCustomDataByKey(
        "handson:collisionSourceSha256", assets.collision_source_sha256
    )
    root_prim.SetCustomDataByKey(
        "handson:renderSimulationCollisionSameMesh", True
    )
    root_prim.SetCustomDataByKey("handson:pillowVolumeComposed", False)
    root_prim.SetCustomDataByKey("handson:nodeCount", BC025_TOWEL_NODE_COUNT)
    root_prim.SetCustomDataByKey(
        "handson:triangleCount", BC025_TOWEL_TRIANGLE_COUNT
    )

    mesh = _author_bc025_surface_mesh(stage)
    visual_material = author_bc025_terry_cloth_material(
        stage,
        BC025_TOWEL_VISUAL_MATERIAL_PATH,
        assets,
    )
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(
        visual_material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )
    _author_bc025_surface_physics(
        stage,
        root_prim,
        mesh.GetPrim(),
        collision,
    )

    table_path = _spawn_worktable(
        stage, sim_utils, task_name=B320_TOWEL_TASK_NAME
    )
    table_prim = stage.GetPrimAtPath(table_path)
    table_prim.SetCustomDataByKey(
        "handson:classification", "task_visible_support"
    )
    table_prim.SetCustomDataByKey("handson:collisionRole", "static_support")
    _spawn_goal_marker(
        stage,
        sim_utils,
        name="towel_target_frame",
        translation=(
            B320_TOWEL_TARGET_CENTER_XY[0],
            B320_TOWEL_TARGET_CENTER_XY[1],
            B320_TABLE_TOP_Z + 0.002,
        ),
        size=(*B320_TOWEL_TARGET_SIZE_XY, 0.003),
        color=(0.10, 0.72, 0.25),
    )
    marker_prim = stage.GetPrimAtPath(
        f"{B320_ASSET_ROOT_PATH}/towel_target_frame"
    )
    marker_prim.SetCustomDataByKey(
        "handson:classification", "task_visible_marker"
    )
    marker_prim.SetCustomDataByKey("handson:collisionEnabled", False)

    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Scene):
            continue
        prim.ApplyAPI(PhysxSchema.PhysxSceneAPI)
        scene_api = PhysxSchema.PhysxSceneAPI(prim)
        scene_api.GetTimeStepsPerSecondAttr().Set(B320_PHYSICS_HZ)
        scene_api.GetGpuMaxDeformableSurfaceContactsAttr().Set(4 * 1_048_576)
        break

    root_prim.SetCustomDataByKey("handson:formalTruth", True)
    root_prim.SetCustomDataByKey("handson:surfaceOnly", True)
    root_prim.SetCustomDataByKey(
        "handson:classification", "task_dynamic_surface_root"
    )
    mesh_prim = stage.GetPrimAtPath(BC025_TOWEL_SURFACE_MESH_PATH)
    mesh_prim.SetCustomDataByKey(
        "handson:classification", "task_visible_deformable_surface"
    )
    mesh_prim.SetCustomDataByKey("handson:collisionRole", "formal_surface")
    mesh_prim.SetCustomDataByKey(
        "handson:renderSimulationCollisionSameMesh", True
    )
    print(
        "[HandsOn/IsaacLab] BC025 assets integrated: "
        f"material={BC025_MATERIAL_TITLE}, "
        f"collision_source={BC025_COLLISION_SOURCE_TITLE}, "
        f"topology={BC025_TOWEL_NODE_COUNT}v/{BC025_TOWEL_TRIANGLE_COUNT}t, "
        f"contact_offset={collision['contact_offset_m']:.4f}, "
        f"rest_offset={collision['rest_offset_m']:.4f}, "
        f"friction={collision['dynamic_friction']:.2f}, "
        f"solver_iterations={collision['solver_position_iterations']}",
        flush=True,
    )
    return BC025_TOWEL_SURFACE_ROOT_PATH


__all__ = [
    "BC025_TOWEL_BEND_STIFFNESS",
    "BC025_TOWEL_COLUMNS",
    "BC025_TOWEL_CONTACT_OFFSET",
    "BC025_TOWEL_DYNAMIC_FRICTION",
    "BC025_TOWEL_ELASTICITY_DAMPING",
    "BC025_TOWEL_MAX_DEPENETRATION_VELOCITY",
    "BC025_TOWEL_MAX_LINEAR_VELOCITY",
    "BC025_TOWEL_NODE_COUNT",
    "BC025_TOWEL_POISSONS_RATIO",
    "BC025_TOWEL_REST_OFFSET",
    "BC025_TOWEL_ROOT_PATH",
    "BC025_TOWEL_ROWS",
    "BC025_TOWEL_SHEAR_STIFFNESS",
    "BC025_TOWEL_STATIC_FRICTION",
    "BC025_TOWEL_STRETCH_STIFFNESS",
    "BC025_TOWEL_SURFACE_DENSITY",
    "BC025_TOWEL_SURFACE_MESH_PATH",
    "BC025_TOWEL_SURFACE_ROOT_PATH",
    "BC025_TOWEL_SURFACE_THICKNESS",
    "BC025_TOWEL_TARGET_END_DISTANCE_M",
    "BC025_TOWEL_TRIANGLE_COUNT",
    "BC025_TOWEL_TRANSLATION_BOUNDS_M",
    "BC025_TOWEL_SIZE_X",
    "BC025_TOWEL_SIZE_Y",
    "B320_TOWEL_PLACEMENT",
    "B320_TOWEL_TARGET_CENTER_XY",
    "B320_TOWEL_TARGET_SIZE_XY",
    "b320_towel_surface_metadata",
    "bc025_towel_surface_topology",
    "spawn_b320_towel_surface",
]
