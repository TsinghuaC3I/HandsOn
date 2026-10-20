"""Native Surface Deformable rope metadata and scene authoring for E02.

Isaac Sim imports stay inside the authoring function so task discovery and
interface-only tests remain usable from a regular Python runtime.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


E02_TASK_NAME = "e02_tie_rope_knot"
E02_SCENE_XML = "arena_arm_hand_e02_tie_rope_knot_rokae_wuji1.xml"
E02_ROPE_ROOT_PATH = "/World/Task/E02Rope"
E02_ROPE_SKIN_MESH_PATH = f"{E02_ROPE_ROOT_PATH}/skinMesh"
E02_ROPE_SIM_MESH_PATH = f"{E02_ROPE_ROOT_PATH}/simMesh"
E02_ROPE_MATERIAL_PATH = "/World/PhysicsMaterials/E02BraidedRope"

E02_SUCCESS_CONDITION_LABELS = (
    "A stable overhand knot is formed",
    "Both free ends extend from the knot core",
    "The knot is tightened by pulling both ends",
    "The tightened knot remains stable for 15 frames",
)

# A continuous octagonal tube is used instead of the former capsule-and-joint
# chain.  There are still 56 centerline segments so observation, replay, and
# success-condition shapes do not change.
E02_ROPE_DIAMETER = 0.010
E02_ROPE_RADIUS = 0.5 * E02_ROPE_DIAMETER
E02_ROPE_SEGMENT_COUNT = 56
E02_ROPE_SEGMENT_SPACING = 0.014
E02_ROPE_CENTERLINE_LENGTH = E02_ROPE_SEGMENT_COUNT * E02_ROPE_SEGMENT_SPACING
E02_ROPE_RADIAL_VERTEX_COUNT = 8
E02_ROPE_RING_COUNT = E02_ROPE_SEGMENT_COUNT + 1
E02_ROPE_NODE_COUNT = E02_ROPE_RING_COUNT * E02_ROPE_RADIAL_VERTEX_COUNT
E02_ROPE_TRIANGLE_COUNT = (
    2 * E02_ROPE_SEGMENT_COUNT * E02_ROPE_RADIAL_VERTEX_COUNT
)

E02_ROPE_LINEAR_DENSITY = 0.060  # kg / m
E02_ROPE_TOTAL_MASS = E02_ROPE_LINEAR_DENSITY * E02_ROPE_CENTERLINE_LENGTH
# With a shell thickness of radius/2, circumference * thickness has the same
# cross-sectional area as a solid cylinder.  This density therefore preserves
# the requested linear density without adding a rigid-body mass approximation.
E02_ROPE_SURFACE_THICKNESS = 0.5 * E02_ROPE_RADIUS
E02_ROPE_MATERIAL_DENSITY = E02_ROPE_TOTAL_MASS / (
    math.pi * E02_ROPE_RADIUS**2 * E02_ROPE_CENTERLINE_LENGTH
)

E02_ROPE_STATIC_FRICTION = 1.15
E02_ROPE_DYNAMIC_FRICTION = 0.95
E02_ROPE_RESTITUTION = 0.0
E02_ROPE_REST_OFFSET = 0.0005
E02_ROPE_CONTACT_OFFSET = 0.0030
E02_ROPE_STRETCH_STIFFNESS = 5.0e4
E02_ROPE_SHEAR_STIFFNESS = 5.0e3
E02_ROPE_BEND_STIFFNESS = 20.0
E02_ROPE_YOUNGS_MODULUS = 5.0e6
E02_ROPE_POISSONS_RATIO = 0.35
E02_ROPE_ELASTICITY_DAMPING = 0.12
E02_ROPE_BEND_DAMPING = 0.08
E02_ROPE_SOLVER_POSITION_ITERATIONS = 64
E02_ROPE_COLLISION_PAIR_UPDATE_FREQUENCY = 4
E02_ROPE_COLLISION_ITERATION_MULTIPLIER = 4
E02_ROPE_MAX_LINEAR_VELOCITY = E02_ROPE_CONTACT_OFFSET * 120.0

E02_TABLE_TOP_Z = 0.89
E02_ROPE_BASE_CENTER = np.asarray([-0.16, 0.0, 0.897], dtype=np.float64)
E02_ROPE_CURVE_AMPLITUDE = 0.030


def e02_initial_rope_vertices() -> np.ndarray:
    """Return the unknotted tabletop centerline vertices."""
    arc = np.linspace(
        -0.5 * E02_ROPE_CENTERLINE_LENGTH,
        0.5 * E02_ROPE_CENTERLINE_LENGTH,
        E02_ROPE_RING_COUNT,
        dtype=np.float64,
    )
    phase = 2.0 * math.pi * arc / E02_ROPE_CENTERLINE_LENGTH
    vertices = np.zeros((E02_ROPE_RING_COUNT, 3), dtype=np.float64)
    vertices[:, 0] = E02_ROPE_BASE_CENTER[0] + E02_ROPE_CURVE_AMPLITUDE * np.sin(
        phase
    )
    vertices[:, 1] = E02_ROPE_BASE_CENTER[1] + arc
    vertices[:, 2] = E02_ROPE_BASE_CENTER[2]
    return vertices


def e02_initial_rope_centers() -> np.ndarray:
    vertices = e02_initial_rope_vertices()
    return 0.5 * (vertices[:-1] + vertices[1:])


def e02_initial_rope_surface_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the open-ended triangular tube used by the deformable solver.

    Returns points, face vertex counts, and flattened face vertex indices.
    Point order is ring-major and is deliberately stable: the environment can
    average each ring's live PhysX nodes back into the legacy centerline state.
    """
    centerline = e02_initial_rope_vertices()
    points = np.empty((E02_ROPE_NODE_COUNT, 3), dtype=np.float64)
    for ring_index, center in enumerate(centerline):
        if ring_index == 0:
            tangent = centerline[1] - centerline[0]
        elif ring_index == E02_ROPE_RING_COUNT - 1:
            tangent = centerline[-1] - centerline[-2]
        else:
            tangent = centerline[ring_index + 1] - centerline[ring_index - 1]
        tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
        lateral = np.asarray([tangent[1], -tangent[0], 0.0], dtype=np.float64)
        lateral /= max(float(np.linalg.norm(lateral)), 1e-12)
        vertical = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        for radial_index in range(E02_ROPE_RADIAL_VERTEX_COUNT):
            angle = 2.0 * math.pi * radial_index / E02_ROPE_RADIAL_VERTEX_COUNT
            offset = E02_ROPE_RADIUS * (
                math.cos(angle) * lateral + math.sin(angle) * vertical
            )
            point_index = (
                ring_index * E02_ROPE_RADIAL_VERTEX_COUNT + radial_index
            )
            points[point_index] = center + offset

    triangles = np.empty((E02_ROPE_TRIANGLE_COUNT, 3), dtype=np.int32)
    triangle_index = 0
    for segment_index in range(E02_ROPE_SEGMENT_COUNT):
        first_ring = segment_index * E02_ROPE_RADIAL_VERTEX_COUNT
        second_ring = (segment_index + 1) * E02_ROPE_RADIAL_VERTEX_COUNT
        for radial_index in range(E02_ROPE_RADIAL_VERTEX_COUNT):
            next_radial = (radial_index + 1) % E02_ROPE_RADIAL_VERTEX_COUNT
            a = first_ring + radial_index
            b = second_ring + radial_index
            c = second_ring + next_radial
            d = first_ring + next_radial
            triangles[triangle_index] = (a, b, c)
            triangles[triangle_index + 1] = (a, c, d)
            triangle_index += 2

    face_vertex_counts = np.full(E02_ROPE_TRIANGLE_COUNT, 3, dtype=np.int32)
    return points, face_vertex_counts, triangles.reshape(-1)


def _rope_display_colors() -> np.ndarray:
    colors = np.empty((E02_ROPE_NODE_COUNT, 3), dtype=np.float32)
    for ring_index in range(E02_ROPE_RING_COUNT):
        for radial_index in range(E02_ROPE_RADIAL_VERTEX_COUNT):
            point_index = (
                ring_index * E02_ROPE_RADIAL_VERTEX_COUNT + radial_index
            )
            if ring_index < 5:
                color = (0.12, 0.34, 0.88)
            elif ring_index >= E02_ROPE_RING_COUNT - 5:
                color = (0.90, 0.20, 0.12)
            else:
                shade = 0.035 if (ring_index + radial_index) % 2 else 0.0
                color = (0.76 + shade, 0.48 + shade, 0.19)
            colors[point_index] = color
    return colors


def spawn_e02_surface_deformable_rope(stage: Any) -> Path:
    """Author a free, self-colliding Surface Deformable rope for E02."""
    from omni.physx.scripts import deformableUtils, physicsUtils
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    root = UsdGeom.Xform.Define(stage, E02_ROPE_ROOT_PATH)
    root_prim = root.GetPrim()
    root_prim.SetCustomDataByKey("handson:assetType", "surface_deformable_rope")
    root_prim.SetCustomDataByKey("handson:diameterMeters", E02_ROPE_DIAMETER)
    root_prim.SetCustomDataByKey(
        "handson:linearDensityKgPerMeter", E02_ROPE_LINEAR_DENSITY
    )
    root_prim.SetCustomDataByKey(
        "handson:centerlineSampleCount", E02_ROPE_SEGMENT_COUNT
    )

    points, face_counts, face_indices = e02_initial_rope_surface_mesh()
    skin_mesh = UsdGeom.Mesh.Define(stage, E02_ROPE_SKIN_MESH_PATH)
    skin_mesh.CreatePointsAttr().Set(
        [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points]
    )
    skin_mesh.CreateFaceVertexCountsAttr().Set(face_counts.tolist())
    skin_mesh.CreateFaceVertexIndicesAttr().Set(face_indices.tolist())
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
    display_color = UsdGeom.PrimvarsAPI(skin_mesh.GetPrim()).CreatePrimvar(
        "displayColor",
        Sdf.ValueTypeNames.Color3fArray,
        UsdGeom.Tokens.vertex,
    )
    display_color.Set(
        [Gf.Vec3f(float(r), float(g), float(b)) for r, g, b in _rope_display_colors()]
    )

    created = deformableUtils.create_auto_surface_deformable_hierarchy(
        stage,
        root_prim_path=root.GetPath(),
        simulation_mesh_path=E02_ROPE_SIM_MESH_PATH,
        cooking_src_mesh_path=E02_ROPE_SKIN_MESH_PATH,
        cooking_src_simplification_enabled=False,
        set_visibility_with_guide_purpose=True,
    )
    if not created:
        raise RuntimeError("PhysX could not cook the E02 surface-deformable rope")

    root_prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    body_attributes = {
        "physxDeformableBody:selfCollision": True,
        "physxDeformableBody:enableSpeculativeCCD": True,
        "physxDeformableBody:solverPositionIterationCount": (
            E02_ROPE_SOLVER_POSITION_ITERATIONS
        ),
        "physxDeformableBody:collisionPairUpdateFrequency": (
            E02_ROPE_COLLISION_PAIR_UPDATE_FREQUENCY
        ),
        "physxDeformableBody:collisionIterationMultiplier": (
            E02_ROPE_COLLISION_ITERATION_MULTIPLIER
        ),
        "physxDeformableBody:maxLinearVelocity": E02_ROPE_MAX_LINEAR_VELOCITY,
    }
    for name, value in body_attributes.items():
        attribute = root_prim.GetAttribute(name)
        if not attribute.IsValid():
            raise RuntimeError(f"Missing surface-deformable attribute {name}")
        attribute.Set(value)

    sim_mesh_prim = stage.GetPrimAtPath(E02_ROPE_SIM_MESH_PATH)
    sim_mesh_prim.ApplyAPI(PhysxSchema.PhysxCollisionAPI)
    collision_api = PhysxSchema.PhysxCollisionAPI(sim_mesh_prim)
    collision_api.GetRestOffsetAttr().Set(E02_ROPE_REST_OFFSET)
    collision_api.GetContactOffsetAttr().Set(E02_ROPE_CONTACT_OFFSET)

    material_created = deformableUtils.add_surface_deformable_material(
        stage,
        E02_ROPE_MATERIAL_PATH,
        density=E02_ROPE_MATERIAL_DENSITY,
        static_friction=E02_ROPE_STATIC_FRICTION,
        dynamic_friction=E02_ROPE_DYNAMIC_FRICTION,
        youngs_modulus=E02_ROPE_YOUNGS_MODULUS,
        poissons_ratio=E02_ROPE_POISSONS_RATIO,
        surface_thickness=E02_ROPE_SURFACE_THICKNESS,
        surface_stretch_stiffness=E02_ROPE_STRETCH_STIFFNESS,
        surface_shear_stiffness=E02_ROPE_SHEAR_STIFFNESS,
        surface_bend_stiffness=E02_ROPE_BEND_STIFFNESS,
    )
    if not material_created:
        raise RuntimeError("PhysX could not create the E02 rope material")
    physicsUtils.add_physics_material_to_prim(
        stage,
        root_prim,
        E02_ROPE_MATERIAL_PATH,
    )
    material_prim = stage.GetPrimAtPath(E02_ROPE_MATERIAL_PATH)
    material_prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
    material_prim.GetAttribute(
        "physxDeformableMaterial:elasticityDamping"
    ).Set(E02_ROPE_ELASTICITY_DAMPING)
    material_prim.GetAttribute("physxDeformableMaterial:bendDamping").Set(
        E02_ROPE_BEND_DAMPING
    )

    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Scene):
            continue
        prim.ApplyAPI(PhysxSchema.PhysxSceneAPI)
        scene_api = PhysxSchema.PhysxSceneAPI(prim)
        scene_api.GetTimeStepsPerSecondAttr().Set(120)
        scene_api.GetGpuMaxDeformableSurfaceContactsAttr().Set(4 * 1_048_576)
        break

    return Path(E02_ROPE_ROOT_PATH)


__all__ = [
    "E02_ROPE_BEND_STIFFNESS",
    "E02_ROPE_CENTERLINE_LENGTH",
    "E02_ROPE_CONTACT_OFFSET",
    "E02_ROPE_DIAMETER",
    "E02_ROPE_DYNAMIC_FRICTION",
    "E02_ROPE_LINEAR_DENSITY",
    "E02_ROPE_MATERIAL_DENSITY",
    "E02_ROPE_NODE_COUNT",
    "E02_ROPE_RADIAL_VERTEX_COUNT",
    "E02_ROPE_RADIUS",
    "E02_ROPE_REST_OFFSET",
    "E02_ROPE_RING_COUNT",
    "E02_ROPE_ROOT_PATH",
    "E02_ROPE_SEGMENT_COUNT",
    "E02_ROPE_SEGMENT_SPACING",
    "E02_ROPE_SHEAR_STIFFNESS",
    "E02_ROPE_SIM_MESH_PATH",
    "E02_ROPE_SKIN_MESH_PATH",
    "E02_ROPE_SOLVER_POSITION_ITERATIONS",
    "E02_ROPE_STATIC_FRICTION",
    "E02_ROPE_STRETCH_STIFFNESS",
    "E02_ROPE_SURFACE_THICKNESS",
    "E02_ROPE_TOTAL_MASS",
    "E02_ROPE_TRIANGLE_COUNT",
    "E02_SCENE_XML",
    "E02_SUCCESS_CONDITION_LABELS",
    "E02_TABLE_TOP_Z",
    "E02_TASK_NAME",
    "e02_initial_rope_centers",
    "e02_initial_rope_surface_mesh",
    "e02_initial_rope_vertices",
    "spawn_e02_surface_deformable_rope",
]
