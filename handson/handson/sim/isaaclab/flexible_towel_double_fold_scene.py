"""Shared support scene for the native bimanual double-fold towels.

The environment supplies the selected source author. Historical cotton authoring
and source integrity helpers remain for callers that explicitly use the old asset.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from . import flexible_cotton_towel as cotton

from .b320_bimanual_scene import (
    B320_ASSET_ROOT_PATH,
    B320_CONTROL_HZ,
    B320_PHYSICS_HZ,
    B320_TABLE_TOP_Z,
    _spawn_worktable,
)
from .g01_deformable_asset_scene import (
    G01_ASSET_ROOT_PATH,
    G01_TOWEL_COLUMNS,
    G01_TOWEL_MATERIAL_PATH,
    G01_TOWEL_NODE_COUNT,
    G01_TOWEL_ROWS,
    G01_TOWEL_SIZE_X,
    G01_TOWEL_SIZE_Y,
    G01_TOWEL_SURFACE_MESH_PATH,
    G01_TOWEL_SURFACE_ROOT_PATH,
    G01_TOWEL_TRIANGLE_COUNT,
)

FLEXIBLE_TOWEL_TASK_NAME = "eval_flex02_fold_towel_twice"
FLEXIBLE_TOWEL_TASK_ID = "FLEXIBLE-001"
FLEXIBLE_TOWEL_SCENE_XML = "arena_arm_hand_b320_bimanual_benchmark_rokae_wuji1.xml"
FLEXIBLE_TOWEL_SUCCESS_CONDITION_LABELS = (
    "The initially flat towel is folded in half in any direction",
    "The folded towel is folded in half again, in the same or another direction",
    "Both hands release and the supported towel remains stable for 15 frames",
)
FLEXIBLE_TOWEL_ASSET_ROOT = (
    Path(__file__).resolve().parent.parent
    / "envs"
    / "usd"
    / "eval_flex02_fold_towel_twice"
    / "Towl011"
)
FLEXIBLE_TOWEL_SOURCE_PATH = FLEXIBLE_TOWEL_ASSET_ROOT / "Towl011.usd"
FLEXIBLE_TOWEL_PLACEMENT = (-0.05, 0.0, B320_TABLE_TOP_Z + 0.004)
# Center offsets from placement; bounded away from the table edges.
FLEXIBLE_TOWEL_RESET_X_BOUNDS_M = (-0.080, 0.080)
FLEXIBLE_TOWEL_RESET_Y_BOUNDS_M = (-0.120, 0.120)

# Shared nominal material settings; live contact qualification is separate
# from the historical source-file integrity check below.
FLEXIBLE_TOWEL_CONTACT_OFFSET = cotton.CONTACT_OFFSET
FLEXIBLE_TOWEL_REST_OFFSET = cotton.REST_OFFSET
FLEXIBLE_TOWEL_SOLVER_POSITION_ITERATIONS = cotton.SOLVER_ITERATIONS
FLEXIBLE_TOWEL_STATIC_FRICTION = cotton.STATIC_FRICTION
FLEXIBLE_TOWEL_DYNAMIC_FRICTION = cotton.DYNAMIC_FRICTION

_ASSET_SHA256 = {
    "Towl011.usd": ("e2e84e9f22ce21550bc2ddecd995de7dbe08004a1f6bb001ee6fd2162b26e327"),
    "texture/T_Towl011_BC001.png": (
        "fc43cd8f058c3413f75813b9a0d9d8ee114e9a30b046ac805648cfacf95921e0"
    ),
    "texture/T_Towl011_N001.png": (
        "28b07d6caf7c8289b0d0d8e594bb232b95868499b58ec915e7784dae2326cea9"
    ),
    "texture/T_Towl011_ORM001.png": (
        "2606180f241830b1ca600ad1c7391b3250e106639737f6f68960d55fd49b23a9"
    ),
}


def validate_flexible_towel_asset_files() -> tuple[Path, ...]:
    """Fail closed if the selected debug asset or a texture changes."""

    validated: list[Path] = []
    for relative, expected in _ASSET_SHA256.items():
        path = FLEXIBLE_TOWEL_ASSET_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Flexible towel asset is missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                "Flexible towel asset hash mismatch: "
                f"path={path} expected={expected} actual={actual}"
            )
        validated.append(path)
    return tuple(validated)


def flexible_towel_interface_nodes() -> Any:
    """Return the authored flat rest grid in world coordinates."""

    import numpy as np

    points, _triangles = cotton.towel_topology()
    nodes = np.asarray(points, dtype=np.float64)
    nodes += np.asarray(FLEXIBLE_TOWEL_PLACEMENT, dtype=np.float64)
    return nodes


def flexible_towel_surface_metadata() -> dict[str, Any]:
    return {
        "label": f"rebuilt-{cotton.MATERIAL_REVISION}-{cotton.AREAL_DENSITY * 1000:g}gsm-3mm/41x41-surface",
        "nominal_mass_kg": cotton.MASS,
        "visual_vertex_count": len(cotton.towel_shell_topology()[0]),
        "visual_triangle_count": len(cotton.towel_shell_topology()[1]),
        "sim_vertex_count": G01_TOWEL_NODE_COUNT,
        "sim_triangle_count": G01_TOWEL_TRIANGLE_COUNT,
    }


def spawn_flexible_towel_surface(stage: Any, sim_utils: Any, source=None) -> str:
    """Author the selected cloth source and support table."""

    import carb
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    if not carb.settings.get_settings().get_as_bool(
        "/persistent/physics/enableDeformableBeta"
    ):
        raise RuntimeError(
            "FLEXIBLE-001 requires " "/persistent/physics/enableDeformableBeta=true"
        )
    root = UsdGeom.Xform.Define(stage, G01_ASSET_ROOT_PATH).GetPrim()
    root.SetCustomDataByKey("handson:task", FLEXIBLE_TOWEL_TASK_NAME)
    root.SetCustomDataByKey("handson:taskId", FLEXIBLE_TOWEL_TASK_ID)
    placement = UsdGeom.Xform.Define(stage, f"{G01_ASSET_ROOT_PATH}/Towl011")
    spawn_position = list(FLEXIBLE_TOWEL_PLACEMENT)
    spawn_position[2] += getattr(source, "spawn_lift", 0.0)
    placement.AddTranslateOp().Set(Gf.Vec3d(*spawn_position))
    author = source.author if source is not None else cotton.author_towel
    author(
        stage,
        G01_TOWEL_SURFACE_ROOT_PATH,
        G01_TOWEL_SURFACE_MESH_PATH,
        G01_TOWEL_MATERIAL_PATH,
    )
    if source is not None:
        root.SetCustomDataByKey("handson:task", source.task_name)
        root.SetCustomDataByKey("handson:taskId", source.task_id)

    benchmark_root = UsdGeom.Xform.Define(stage, B320_ASSET_ROOT_PATH).GetPrim()
    benchmark_root.SetCustomDataByKey("handson:task", FLEXIBLE_TOWEL_TASK_NAME)
    benchmark_root.SetCustomDataByKey("handson:taskId", FLEXIBLE_TOWEL_TASK_ID)
    _spawn_worktable(stage, sim_utils, task_name=FLEXIBLE_TOWEL_TASK_NAME)

    active_scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if len(active_scenes) != 1:
        raise RuntimeError(
            "Flexible towel requires exactly one active PhysicsScene after "
            "source sanitization; got "
            f"{[str(prim.GetPath()) for prim in active_scenes]}"
        )
    scene_prim = active_scenes[0]
    scene_prim.ApplyAPI(PhysxSchema.PhysxSceneAPI)
    scene_api = PhysxSchema.PhysxSceneAPI(scene_prim)
    scene_api.GetTimeStepsPerSecondAttr().Set(B320_PHYSICS_HZ)
    scene_api.GetGpuMaxDeformableSurfaceContactsAttr().Set(4 * 1_048_576)
    scene_prim.SetCustomDataByKey("handson:controlHz", B320_CONTROL_HZ)
    scene_prim.SetCustomDataByKey("handson:physicsHz", B320_PHYSICS_HZ)
    return G01_TOWEL_SURFACE_ROOT_PATH


__all__ = [
    "FLEXIBLE_TOWEL_ASSET_ROOT",
    "FLEXIBLE_TOWEL_CONTACT_OFFSET",
    "FLEXIBLE_TOWEL_DYNAMIC_FRICTION",
    "FLEXIBLE_TOWEL_FINAL_TARGET_SIZE_XY",
    "FLEXIBLE_TOWEL_PLACEMENT",
    "FLEXIBLE_TOWEL_REST_OFFSET",
    "FLEXIBLE_TOWEL_SCENE_XML",
    "FLEXIBLE_TOWEL_SOLVER_POSITION_ITERATIONS",
    "FLEXIBLE_TOWEL_SOURCE_PATH",
    "FLEXIBLE_TOWEL_STATIC_FRICTION",
    "FLEXIBLE_TOWEL_SUCCESS_CONDITION_LABELS",
    "FLEXIBLE_TOWEL_TASK_ID",
    "FLEXIBLE_TOWEL_TASK_NAME",
    "flexible_towel_interface_nodes",
    "flexible_towel_surface_metadata",
    "spawn_flexible_towel_surface",
    "validate_flexible_towel_asset_files",
]
