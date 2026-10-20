"""Native wiping-cloth asset and C01-derived reachable table mask for E05."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


E05_TASK_NAME = "e05_fold_cloth_wipe_table"
E05_SCENE_XML = "arena_arm_hand_e05_fold_cloth_wipe_table_rokae_wuji1.xml"
E05_CLOTH_ROOT_PATH = "/World/Task/E05WipingCloth"
E05_CLOTH_SKIN_MESH_PATH = f"{E05_CLOTH_ROOT_PATH}/mesh"
E05_CLOTH_SIM_MESH_PATH = f"{E05_CLOTH_ROOT_PATH}/simMesh"
E05_CLOTH_MATERIAL_PATH = "/World/PhysicsMaterials/E05WipingCloth"

E05_SUCCESS_CONDITION_LABELS = (
    "The wiping cloth is folded before wiping starts",
    "The folded cloth contacts and sweeps the reachable table cells",
    "At least 90 percent of the C01 arm-reachable table mask is covered",
    "The final cloth remains on the reachable tabletop and stable for 15 frames",
)

E05_TABLE_TOP_Z = 0.89
E05_CLOTH_CENTER = np.asarray([-0.22, 0.0, E05_TABLE_TOP_Z + 0.007])
E05_CLOTH_SIZE_X = 0.32
E05_CLOTH_SIZE_Y = 0.24
E05_CLOTH_COLUMNS = 17
E05_CLOTH_ROWS = 13
E05_CLOTH_NODE_COUNT = E05_CLOTH_COLUMNS * E05_CLOTH_ROWS
E05_CLOTH_TRIANGLE_COUNT = 2 * (E05_CLOTH_COLUMNS - 1) * (E05_CLOTH_ROWS - 1)
E05_CLOTH_SURFACE_THICKNESS = 0.0012
E05_CLOTH_REST_OFFSET = 0.0010
E05_CLOTH_CONTACT_OFFSET = 0.0060
E05_CLOTH_STRETCH_STIFFNESS = 8.0e3
E05_CLOTH_SHEAR_STIFFNESS = 600.0
E05_CLOTH_BEND_STIFFNESS = 35.0
E05_CLOTH_YOUNGS_MODULUS = 2.0e6
E05_CLOTH_POISSONS_RATIO = 0.35
E05_CLOTH_DENSITY = 250.0
E05_CLOTH_STATIC_FRICTION = 1.05
E05_CLOTH_DYNAMIC_FRICTION = 0.88
E05_CLOTH_ELASTICITY_DAMPING = 0.20
E05_CLOTH_BEND_DAMPING = 0.15

E05_WIPE_GRID_CELL_SIZE = 0.05
E05_WIPE_GRID_X_COUNT = 16
E05_WIPE_GRID_Y_COUNT = 32
E05_WIPE_GRID_CELL_COUNT = E05_WIPE_GRID_X_COUNT * E05_WIPE_GRID_Y_COUNT
E05_WIPE_GRID_X_MIN = -0.525
E05_WIPE_GRID_Y_MIN = -0.775
E05_REACH_CACHE_PATH = (
    Path(__file__).resolve().parent.parent
    / "envs"
    / "c01_rokae_wuji_reachability.npz"
)


def e05_initial_cloth_nodes() -> np.ndarray:
    x_values = np.linspace(
        E05_CLOTH_CENTER[0] - 0.5 * E05_CLOTH_SIZE_X,
        E05_CLOTH_CENTER[0] + 0.5 * E05_CLOTH_SIZE_X,
        E05_CLOTH_COLUMNS,
    )
    y_values = np.linspace(
        E05_CLOTH_CENTER[1] - 0.5 * E05_CLOTH_SIZE_Y,
        E05_CLOTH_CENTER[1] + 0.5 * E05_CLOTH_SIZE_Y,
        E05_CLOTH_ROWS,
    )
    return np.asarray(
        [(x, y, E05_CLOTH_CENTER[2]) for y in y_values for x in x_values],
        dtype=np.float64,
    )


def e05_cloth_surface_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = e05_initial_cloth_nodes()
    triangles = np.empty((E05_CLOTH_TRIANGLE_COUNT, 3), dtype=np.int32)
    triangle_index = 0
    for row in range(E05_CLOTH_ROWS - 1):
        for column in range(E05_CLOTH_COLUMNS - 1):
            a = row * E05_CLOTH_COLUMNS + column
            b = a + 1
            c = a + E05_CLOTH_COLUMNS + 1
            d = a + E05_CLOTH_COLUMNS
            triangles[triangle_index] = (a, b, c)
            triangles[triangle_index + 1] = (a, c, d)
            triangle_index += 2
    face_counts = np.full(E05_CLOTH_TRIANGLE_COUNT, 3, dtype=np.int32)
    return points, face_counts, triangles.reshape(-1)


def e05_wipe_grid_centers() -> np.ndarray:
    x_values = E05_WIPE_GRID_X_MIN + E05_WIPE_GRID_CELL_SIZE * np.arange(
        E05_WIPE_GRID_X_COUNT,
        dtype=np.float64,
    )
    y_values = E05_WIPE_GRID_Y_MIN + E05_WIPE_GRID_CELL_SIZE * np.arange(
        E05_WIPE_GRID_Y_COUNT,
        dtype=np.float64,
    )
    return np.asarray([(x, y) for x in x_values for y in y_values], dtype=np.float64)


@lru_cache(maxsize=1)
def e05_reachable_wipe_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Project the generated C01 palm reach cache onto the 5 cm table grid."""
    if not E05_REACH_CACHE_PATH.is_file():
        raise FileNotFoundError(
            "E05 requires the generated C01 ROKAE+Wuji reachability cache: "
            f"{E05_REACH_CACHE_PATH}"
        )
    cache = np.load(E05_REACH_CACHE_PATH, allow_pickle=False)
    grid = e05_wipe_grid_centers()
    side_masks: list[np.ndarray] = []
    for side in ("right", "left"):
        points = np.asarray(cache[f"{side}_points"], dtype=np.float64)
        table_slice = points[(points[:, 2] >= 0.88) & (points[:, 2] <= 0.98)]
        if table_slice.shape[0] < 3:
            raise RuntimeError(f"C01 {side} reach cache has no tabletop slice")
        distances_squared = np.sum(
            np.square(grid[:, None, :] - table_slice[None, :, :2]),
            axis=2,
        )
        side_masks.append(
            np.min(distances_squared, axis=1)
            <= (2.5 * float(cache["resolution"][0])) ** 2
        )
    right_mask, left_mask = side_masks
    union_mask = right_mask | left_mask
    if int(np.count_nonzero(union_mask)) < 100:
        raise RuntimeError("C01 reach projection produced an implausibly small E05 mask")
    metadata = {
        "resolution_m": float(cache["resolution"][0]),
        "samples_per_arm": float(cache["samples"][0]),
        "right_cell_count": float(np.count_nonzero(right_mask)),
        "left_cell_count": float(np.count_nonzero(left_mask)),
        "union_cell_count": float(np.count_nonzero(union_mask)),
        "union_area_m2": float(
            np.count_nonzero(union_mask) * E05_WIPE_GRID_CELL_SIZE**2
        ),
    }
    for array in (right_mask, left_mask, union_mask):
        array.setflags(write=False)
    return right_mask, left_mask, union_mask, metadata


def spawn_e05_surface_deformable_cloth(stage: Any) -> Path:
    """Author a self-colliding rectangular microfiber wiping cloth."""
    from omni.physx.scripts import deformableUtils, physicsUtils
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    root = UsdGeom.Xform.Define(stage, E05_CLOTH_ROOT_PATH)
    root_prim = root.GetPrim()
    root_prim.SetCustomDataByKey("handson:assetType", "surface_deformable_wiping_cloth")
    root_prim.SetCustomDataByKey("handson:sizeXMeters", E05_CLOTH_SIZE_X)
    root_prim.SetCustomDataByKey("handson:sizeYMeters", E05_CLOTH_SIZE_Y)
    root_prim.SetCustomDataByKey("handson:nodeCount", E05_CLOTH_NODE_COUNT)

    points, face_counts, face_indices = e05_cloth_surface_mesh()
    skin_mesh = UsdGeom.Mesh.Define(stage, E05_CLOTH_SKIN_MESH_PATH)
    skin_mesh.CreatePointsAttr().Set(
        [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points]
    )
    skin_mesh.CreateFaceVertexCountsAttr().Set(face_counts.tolist())
    skin_mesh.CreateFaceVertexIndicesAttr().Set(face_indices.tolist())
    skin_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    skin_mesh.CreateDoubleSidedAttr().Set(True)
    skin_mesh.CreateExtentAttr().Set(
        [
            Gf.Vec3f(*np.min(points, axis=0).astype(float).tolist()),
            Gf.Vec3f(*np.max(points, axis=0).astype(float).tolist()),
        ]
    )
    colors = np.tile(np.asarray([[0.18, 0.72, 0.58]], dtype=np.float32), (points.shape[0], 1))
    for index in range(points.shape[0]):
        if (index // E05_CLOTH_COLUMNS + index % E05_CLOTH_COLUMNS) % 2:
            colors[index] += np.asarray([0.025, 0.025, 0.025], dtype=np.float32)
    color_primvar = UsdGeom.PrimvarsAPI(skin_mesh.GetPrim()).CreatePrimvar(
        "displayColor",
        Sdf.ValueTypeNames.Color3fArray,
        UsdGeom.Tokens.vertex,
    )
    color_primvar.Set(
        [Gf.Vec3f(float(r), float(g), float(b)) for r, g, b in colors]
    )

    created = deformableUtils.create_auto_surface_deformable_hierarchy(
        stage,
        root_prim_path=root.GetPath(),
        simulation_mesh_path=E05_CLOTH_SIM_MESH_PATH,
        cooking_src_mesh_path=E05_CLOTH_SKIN_MESH_PATH,
        cooking_src_simplification_enabled=False,
        set_visibility_with_guide_purpose=True,
    )
    if not created:
        raise RuntimeError("PhysX could not cook the E05 wiping cloth")

    root_prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    for name, value in {
        "physxDeformableBody:selfCollision": True,
        "physxDeformableBody:enableSpeculativeCCD": True,
        "physxDeformableBody:solverPositionIterationCount": 32,
        "physxDeformableBody:collisionPairUpdateFrequency": 4,
        "physxDeformableBody:collisionIterationMultiplier": 4,
        "physxDeformableBody:maxLinearVelocity": E05_CLOTH_CONTACT_OFFSET * 120.0,
    }.items():
        attribute = root_prim.GetAttribute(name)
        if not attribute.IsValid():
            raise RuntimeError(f"Missing E05 surface-deformable attribute {name}")
        attribute.Set(value)

    simulation_mesh_prim = stage.GetPrimAtPath(E05_CLOTH_SIM_MESH_PATH)
    simulation_mesh_prim.ApplyAPI(PhysxSchema.PhysxCollisionAPI)
    collision_api = PhysxSchema.PhysxCollisionAPI(simulation_mesh_prim)
    collision_api.GetRestOffsetAttr().Set(E05_CLOTH_REST_OFFSET)
    collision_api.GetContactOffsetAttr().Set(E05_CLOTH_CONTACT_OFFSET)

    material_created = deformableUtils.add_surface_deformable_material(
        stage,
        E05_CLOTH_MATERIAL_PATH,
        density=E05_CLOTH_DENSITY,
        static_friction=E05_CLOTH_STATIC_FRICTION,
        dynamic_friction=E05_CLOTH_DYNAMIC_FRICTION,
        youngs_modulus=E05_CLOTH_YOUNGS_MODULUS,
        poissons_ratio=E05_CLOTH_POISSONS_RATIO,
        surface_thickness=E05_CLOTH_SURFACE_THICKNESS,
        surface_stretch_stiffness=E05_CLOTH_STRETCH_STIFFNESS,
        surface_shear_stiffness=E05_CLOTH_SHEAR_STIFFNESS,
        surface_bend_stiffness=E05_CLOTH_BEND_STIFFNESS,
    )
    if not material_created:
        raise RuntimeError("PhysX could not create the E05 cloth material")
    physicsUtils.add_physics_material_to_prim(
        stage,
        root_prim,
        E05_CLOTH_MATERIAL_PATH,
    )
    material_prim = stage.GetPrimAtPath(E05_CLOTH_MATERIAL_PATH)
    material_prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
    material_prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(
        E05_CLOTH_ELASTICITY_DAMPING
    )
    material_prim.GetAttribute("physxDeformableMaterial:bendDamping").Set(
        E05_CLOTH_BEND_DAMPING
    )

    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Scene):
            continue
        prim.ApplyAPI(PhysxSchema.PhysxSceneAPI)
        scene_api = PhysxSchema.PhysxSceneAPI(prim)
        scene_api.GetTimeStepsPerSecondAttr().Set(120)
        scene_api.GetGpuMaxDeformableSurfaceContactsAttr().Set(4 * 1_048_576)
        break
    return Path(E05_CLOTH_ROOT_PATH)


__all__ = [
    "E05_CLOTH_COLUMNS",
    "E05_CLOTH_CONTACT_OFFSET",
    "E05_CLOTH_NODE_COUNT",
    "E05_CLOTH_ROOT_PATH",
    "E05_CLOTH_ROWS",
    "E05_CLOTH_SIZE_X",
    "E05_CLOTH_SIZE_Y",
    "E05_CLOTH_TRIANGLE_COUNT",
    "E05_REACH_CACHE_PATH",
    "E05_SCENE_XML",
    "E05_SUCCESS_CONDITION_LABELS",
    "E05_TABLE_TOP_Z",
    "E05_TASK_NAME",
    "E05_WIPE_GRID_CELL_COUNT",
    "E05_WIPE_GRID_CELL_SIZE",
    "E05_WIPE_GRID_X_COUNT",
    "E05_WIPE_GRID_Y_COUNT",
    "e05_cloth_surface_mesh",
    "e05_initial_cloth_nodes",
    "e05_reachable_wipe_masks",
    "e05_wipe_grid_centers",
    "spawn_e05_surface_deformable_cloth",
]
