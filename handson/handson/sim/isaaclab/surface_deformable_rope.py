"""Reusable native PhysX authoring for continuous deformable ropes.

The module deliberately imports Isaac Sim only inside the authoring function.
Geometry generators and task-interface tests therefore remain usable from a
normal Python process.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SurfaceDeformableRopeCfg:
    root_path: str
    skin_mesh_path: str
    simulation_mesh_path: str
    material_path: str
    diameter: float
    centerline_length: float
    centerline_segment_count: int
    radial_vertex_count: int
    material_density: float
    surface_thickness: float
    static_friction: float
    dynamic_friction: float
    rest_offset: float
    contact_offset: float
    stretch_stiffness: float
    shear_stiffness: float
    bend_stiffness: float
    youngs_modulus: float
    poissons_ratio: float
    elasticity_damping: float
    bend_damping: float
    solver_position_iterations: int = 64
    collision_pair_update_frequency: int = 4
    collision_iteration_multiplier: int = 4

    @property
    def max_linear_velocity(self) -> float:
        return 120.0 * self.contact_offset


def author_surface_deformable_rope(
    stage: Any,
    cfg: SurfaceDeformableRopeCfg,
    *,
    points: np.ndarray,
    face_vertex_counts: np.ndarray,
    face_vertex_indices: np.ndarray,
    display_colors: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Author one self-colliding tube as a Surface Deformable body."""
    from omni.physx.scripts import deformableUtils, physicsUtils
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    face_vertex_counts = np.asarray(face_vertex_counts, dtype=np.int32).ravel()
    face_vertex_indices = np.asarray(face_vertex_indices, dtype=np.int32).ravel()
    display_colors = np.asarray(display_colors, dtype=np.float32).reshape(-1, 3)
    if display_colors.shape[0] != points.shape[0]:
        raise ValueError(
            "Surface rope colors must match points: "
            f"colors={display_colors.shape[0]}, points={points.shape[0]}"
        )
    if int(np.sum(face_vertex_counts)) != face_vertex_indices.size:
        raise ValueError("Surface rope face counts do not match face indices")

    root = UsdGeom.Xform.Define(stage, cfg.root_path)
    root_prim = root.GetPrim()
    root_prim.SetCustomDataByKey("handson:assetType", "surface_deformable_rope")
    root_prim.SetCustomDataByKey("handson:diameterMeters", cfg.diameter)
    root_prim.SetCustomDataByKey(
        "handson:centerlineLengthMeters", cfg.centerline_length
    )
    root_prim.SetCustomDataByKey(
        "handson:centerlineSegmentCount", cfg.centerline_segment_count
    )
    for key, value in (metadata or {}).items():
        root_prim.SetCustomDataByKey(f"handson:{key}", value)

    skin_mesh = UsdGeom.Mesh.Define(stage, cfg.skin_mesh_path)
    skin_mesh.CreatePointsAttr().Set(
        [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points]
    )
    skin_mesh.CreateFaceVertexCountsAttr().Set(face_vertex_counts.tolist())
    skin_mesh.CreateFaceVertexIndicesAttr().Set(face_vertex_indices.tolist())
    skin_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    skin_mesh.CreateDoubleSidedAttr().Set(True)
    bounds_min = np.min(points, axis=0)
    bounds_max = np.max(points, axis=0)
    skin_mesh.CreateExtentAttr().Set(
        [
            Gf.Vec3f(*bounds_min.astype(float).tolist()),
            Gf.Vec3f(*bounds_max.astype(float).tolist()),
        ]
    )
    color_primvar = UsdGeom.PrimvarsAPI(skin_mesh.GetPrim()).CreatePrimvar(
        "displayColor",
        Sdf.ValueTypeNames.Color3fArray,
        UsdGeom.Tokens.vertex,
    )
    color_primvar.Set(
        [Gf.Vec3f(float(r), float(g), float(b)) for r, g, b in display_colors]
    )

    created = deformableUtils.create_auto_surface_deformable_hierarchy(
        stage,
        root_prim_path=root.GetPath(),
        simulation_mesh_path=cfg.simulation_mesh_path,
        cooking_src_mesh_path=cfg.skin_mesh_path,
        cooking_src_simplification_enabled=False,
        set_visibility_with_guide_purpose=True,
    )
    if not created:
        raise RuntimeError(
            f"PhysX could not cook the surface-deformable rope at {cfg.root_path}"
        )

    root_prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    body_attributes = {
        "physxDeformableBody:selfCollision": True,
        "physxDeformableBody:enableSpeculativeCCD": True,
        "physxDeformableBody:solverPositionIterationCount": (
            cfg.solver_position_iterations
        ),
        "physxDeformableBody:collisionPairUpdateFrequency": (
            cfg.collision_pair_update_frequency
        ),
        "physxDeformableBody:collisionIterationMultiplier": (
            cfg.collision_iteration_multiplier
        ),
        "physxDeformableBody:maxLinearVelocity": cfg.max_linear_velocity,
    }
    for name, value in body_attributes.items():
        attribute = root_prim.GetAttribute(name)
        if not attribute.IsValid():
            raise RuntimeError(f"Missing surface-deformable attribute {name}")
        attribute.Set(value)

    simulation_mesh_prim = stage.GetPrimAtPath(cfg.simulation_mesh_path)
    simulation_mesh_prim.ApplyAPI(PhysxSchema.PhysxCollisionAPI)
    collision_api = PhysxSchema.PhysxCollisionAPI(simulation_mesh_prim)
    collision_api.GetRestOffsetAttr().Set(cfg.rest_offset)
    collision_api.GetContactOffsetAttr().Set(cfg.contact_offset)

    material_created = deformableUtils.add_surface_deformable_material(
        stage,
        cfg.material_path,
        density=cfg.material_density,
        static_friction=cfg.static_friction,
        dynamic_friction=cfg.dynamic_friction,
        youngs_modulus=cfg.youngs_modulus,
        poissons_ratio=cfg.poissons_ratio,
        surface_thickness=cfg.surface_thickness,
        surface_stretch_stiffness=cfg.stretch_stiffness,
        surface_shear_stiffness=cfg.shear_stiffness,
        surface_bend_stiffness=cfg.bend_stiffness,
    )
    if not material_created:
        raise RuntimeError(
            f"PhysX could not create rope material {cfg.material_path}"
        )
    physicsUtils.add_physics_material_to_prim(
        stage,
        root_prim,
        cfg.material_path,
    )
    material_prim = stage.GetPrimAtPath(cfg.material_path)
    material_prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
    material_prim.GetAttribute(
        "physxDeformableMaterial:elasticityDamping"
    ).Set(cfg.elasticity_damping)
    material_prim.GetAttribute("physxDeformableMaterial:bendDamping").Set(
        cfg.bend_damping
    )

    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Scene):
            continue
        prim.ApplyAPI(PhysxSchema.PhysxSceneAPI)
        scene_api = PhysxSchema.PhysxSceneAPI(prim)
        scene_api.GetTimeStepsPerSecondAttr().Set(120)
        scene_api.GetGpuMaxDeformableSurfaceContactsAttr().Set(4 * 1_048_576)
        break

    return Path(cfg.root_path)


__all__ = ["SurfaceDeformableRopeCfg", "author_surface_deformable_rope"]
