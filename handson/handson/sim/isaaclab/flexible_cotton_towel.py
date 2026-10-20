"""Cotton towel rest mesh shared by Blender authoring and native PhysX.

Dimensions are metres, mass is kg. These are nominal 750 gsm terry-cotton
parameters, not measured material identification. The visible shell follows
the actual simulated midsurface in the task runtime; it must never become a second collider.
"""

from __future__ import annotations

ROWS = COLUMNS = 41
SIZE_X, SIZE_Y = 0.27484, 0.27020
THICKNESS = 0.003
AREAL_DENSITY = 0.750
DENSITY = AREAL_DENSITY / THICKNESS
MASS = SIZE_X * SIZE_Y * AREAL_DENSITY
# Detect a falling sheet before it can cross the table within a 120 Hz step.
# Rest offset still sets the physical 3 mm thickness between stacked layers.
CONTACT_OFFSET = 0.012
REST_OFFSET = THICKNESS / 2
HAND_CONTACT_OFFSET = 0.008
# The task derives contact hulls from the visible hand components. No broad
# inflation is needed to compensate for the previous capsule/visual mismatch.
HAND_REST_OFFSET = 0.0005
# Shared by both hands in FLEXIBLE-001/002; retain a finite finger speed
# without making opening/closing lag behind the faster palm translation.
HAND_JOINT_SPEED_RAD_S = 2.0
STATIC_FRICTION, DYNAMIC_FRICTION = 0.85, 0.65
YOUNGS_MODULUS, POISSONS_RATIO = 3000000.0, 0.25
# Stretch/shear overrides are not implemented by Omni PhysX 107.3.
STRETCH_STIFFNESS, SHEAR_STIFFNESS = 0.0, 0.0
# Terry cloth is difficult to stretch but easy to bend. Do not derive bending
# from the bulk plate formula E*h**3: the yarns can slide and rotate.
# B = 2.50e-6 N m is about 25.5 mgf cm, within the cotton-terry cantilever
# range in Sekerden (2012), Figure 2. The previous 41.3 mgf cm fit left a
# pronounced arch at a released fold. This softer nominal fit lets layers
# settle together without reducing ground-weave stretch stiffness or solver
# iterations. It is not identification of the user's towel. PhysX applies h**3.
# Sources and native before/after measurements: docs/flexible_towel_material.md.
BENDING_RIGIDITY = 2.50e-6
BEND_STIFFNESS = BENDING_RIGIDITY / THICKNESS**3
ELASTICITY_DAMPING = 0.25
SOLVER_ITERATIONS = 64
# Use the shared surface-cloth contact cadence at the fixed 120 Hz step.
# Keep 64 material solver iterations; eight contact updates resolve fast thin
# sheet contacts without reducing the ground weave's stretch resistance.
COLLISION_UPDATES = 8
COLLISION_ITERATIONS = 4
MAX_SPEED = 2.0
MAX_DEPENETRATION_SPEED = 0.20
# Passive velocity damping suppresses the undamped flutter after hand release.
# Keep gravity and native collisions; never clamp the live nodal state.
LINEAR_DAMPING = 0.8
COLOR = (0.24, 0.48, 0.53)
MATERIAL_REVISION = "terry-v4-damped-contact"


def towel_topology(size_xy=None):
    """A connected flat sheet; alternating diagonals avoid a preferred bias."""
    size_x, size_y = size_xy or (SIZE_X, SIZE_Y)
    points = [
        (size_x * (c / (COLUMNS - 1) - 0.5), size_y * (r / (ROWS - 1) - 0.5), 0.0)
        for r in range(ROWS)
        for c in range(COLUMNS)
    ]
    triangles = []
    for r in range(ROWS - 1):
        for c in range(COLUMNS - 1):
            a = r * COLUMNS + c
            b = a + 1
            d = a + COLUMNS
            e = d + 1
            triangles.extend(
                ((a, b, e), (a, e, d)) if (r + c) % 2 == 0 else ((a, b, d), (b, e, d))
            )
    return points, triangles


def towel_shell_topology(size_xy=None, mesh_data=None):
    """Closed 3 mm render shell, with node order matching the midsurface."""
    points, triangles = towel_topology(size_xy) if mesh_data is None else mesh_data
    n = len(points)
    shell = [(x, y, z + THICKNESS / 2) for x, y, z in points]
    shell += [(x, y, z - THICKNESS / 2) for x, y, z in points]
    faces = list(triangles) + [(c + n, b + n, a + n) for a, b, c in triangles]
    boundary = list(range(COLUMNS))
    boundary += [r * COLUMNS + COLUMNS - 1 for r in range(1, ROWS)]
    boundary += [(ROWS - 1) * COLUMNS + c for c in range(COLUMNS - 2, -1, -1)]
    boundary += [r * COLUMNS for r in range(ROWS - 2, 0, -1)]
    for a, b in zip(boundary, boundary[1:] + boundary[:1]):
        faces.extend(((a, a + n, b + n), (a, b + n, b)))
    return shell, faces


def visual_mesh_path(root_path):
    """Keep graphics outside the native deformable's automatic skin hierarchy."""
    return root_path + "_render/CottonSurface"


def author_towel(stage, root_path, mesh_path, material_path, *,
                 size_xy=None, mass=None, color_texture=None, color=None, mesh_data=None):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
    from omni.physx.scripts import deformableUtils, physicsUtils

    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    size_x, size_y = size_xy or (SIZE_X, SIZE_Y)
    mass = MASS if mass is None else mass
    use_textures = color is None or color_texture is not None
    color = COLOR if color is None else color
    points, triangles = towel_topology((size_x, size_y)) if mesh_data is None else mesh_data
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr([v for f in triangles for v in f])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateVisibilityAttr("invisible")
    sim = mesh.GetPrim()
    for api in ("OmniPhysicsSurfaceDeformableSimAPI", "PhysicsCollisionAPI"):
        if not sim.ApplyAPI(api):
            raise RuntimeError(f"Cannot apply {api}")
    sim.GetAttribute("omniphysics:restShapePoints").Set(mesh.GetPointsAttr().Get())
    sim.GetAttribute("omniphysics:restTriVtxIndices").Set(
        [Gf.Vec3i(*t) for t in triangles]
    )
    for api in ("OmniPhysicsDeformableBodyAPI", "PhysxSurfaceDeformableBodyAPI"):
        if not root.ApplyAPI(api):
            raise RuntimeError(f"Cannot apply {api}")
    for key, value in {
        "selfCollision": True,
        "enableSpeculativeCCD": True,
        "solverPositionIterationCount": SOLVER_ITERATIONS,
        "collisionPairUpdateFrequency": COLLISION_UPDATES,
        "collisionIterationMultiplier": COLLISION_ITERATIONS,
        "maxLinearVelocity": MAX_SPEED,
        "maxDepenetrationVelocity": MAX_DEPENETRATION_SPEED,
        "linearDamping": LINEAR_DAMPING,
    }.items():
        attr = root.GetAttribute("physxDeformableBody:" + key)
        if not attr.IsValid():
            raise RuntimeError(f"Missing deformable property {key}")
        attr.Set(value)
    collision = PhysxSchema.PhysxCollisionAPI.Apply(sim)
    collision.CreateContactOffsetAttr(CONTACT_OFFSET)
    collision.CreateRestOffsetAttr(REST_OFFSET)
    ok = deformableUtils.add_surface_deformable_material(
        stage,
        material_path,
        density=mass / (size_x * size_y * THICKNESS),
        static_friction=STATIC_FRICTION,
        dynamic_friction=DYNAMIC_FRICTION,
        youngs_modulus=YOUNGS_MODULUS,
        poissons_ratio=POISSONS_RATIO,
        surface_thickness=THICKNESS,
        surface_stretch_stiffness=STRETCH_STIFFNESS,
        surface_shear_stiffness=SHEAR_STIFFNESS,
        surface_bend_stiffness=BEND_STIFFNESS,
    )
    if not ok:
        raise RuntimeError("Cannot author cotton material")
    physicsUtils.add_physics_material_to_prim(stage, root, material_path)
    mat = stage.GetPrimAtPath(material_path)
    mat.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
    mat.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(
        ELASTICITY_DAMPING
    )
    mat.GetAttribute("physxDeformableMaterial:bendDamping").Set(ELASTICITY_DAMPING)

    shell_points, shell_faces = towel_shell_topology((size_x, size_y), (points, triangles))
    # Native PhysX otherwise registers this as an automatic skin and overwrites
    # our live graphics at render time, using an unsuitable default bind pose.
    UsdGeom.Xform.Define(stage, root_path + "_render")
    skin = UsdGeom.Mesh.Define(stage, visual_mesh_path(root_path))
    skin.CreatePointsAttr([Gf.Vec3f(*p) for p in shell_points])
    skin.CreateFaceVertexCountsAttr([3] * len(shell_faces))
    skin.CreateFaceVertexIndicesAttr([v for f in shell_faces for v in f])
    skin.CreateSubdivisionSchemeAttr("none")
    skin.CreateDoubleSidedAttr(False)
    skin.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    skin.CreateExtentAttr(
        [
            Gf.Vec3f(-size_x / 2, -size_y / 2, -THICKNESS / 2),
            Gf.Vec3f(size_x / 2, size_y / 2, THICKNESS / 2),
        ]
    )
    uv = UsdGeom.PrimvarsAPI(skin).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    uv.Set([Gf.Vec2f(x / size_x + 0.5, y / size_y + 0.5) for x, y, z in shell_points])
    visual = UsdShade.Material.Define(stage, material_path + "Visual")
    shader = UsdShade.Shader.Define(stage, material_path + "Visual/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.92)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    # Render detail stays on the live physical sheet. Pile relief is a normal
    # map, so it adds no untracked collider or independent cloth motion.
    from pathlib import Path

    texture_dir = Path(__file__).resolve().parent / "usd" / "flexible_cotton_towel"
    reader = UsdShade.Shader.Define(stage, material_path + "Visual/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    for name, target, filename in (
        ("Color", "diffuseColor", "terry_color.png"),
        ("Normal", "normal", "terry_normal.png"),
    ):
        # Imported comparison materials have their own albedo. SoftGym uses
        # a plain colour and has no terry pile normal map in its source.
        if not use_textures:
            continue
        if name == "Color" and color_texture is not None:
            filename = str(color_texture)
        texture = UsdShade.Shader.Define(stage, material_path + "Visual/" + name)
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(texture_dir / filename))
        )
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
            "raw" if name == "Normal" else "sRGB"
        )
        if name == "Normal":
            texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(2, 2, 2, 1)
            )
            texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(-1, -1, -1, 0)
            )
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput(
            target,
            (
                Sdf.ValueTypeNames.Normal3f
                if name == "Normal"
                else Sdf.ValueTypeNames.Color3f
            ),
        ).ConnectToSource(texture.ConnectableAPI(), "rgb")
    visual.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(skin.GetPrim()).Bind(visual)
    root.SetCustomDataByKey(
        "handson:assetType", "cotton_towel_surface_with_render_shell"
    )
    root.SetCustomDataByKey("handson:nominalMassKg", mass)
    root.SetCustomDataByKey("handson:thicknessM", THICKNESS)
    root.SetCustomDataByKey("handson:materialCalibrated", False)
    root.SetCustomDataByKey("handson:materialRevision", MATERIAL_REVISION)
    root.SetCustomDataByKey("handson:bendingRigidityNm", BENDING_RIGIDITY)
    return root_path


def shell_points_from_midpoints(points):
    """Derive render vertices only; this never changes physical nodes."""
    import numpy as np

    p = np.asarray(points, dtype=np.float64)
    if p.shape != (ROWS * COLUMNS, 3) or not np.isfinite(p).all():
        raise ValueError("Towel shell requires 1681 finite simulation points")
    tri = np.asarray(towel_topology()[1], dtype=np.int64)
    face = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    normal = np.zeros_like(p)
    for corner in range(3):
        for axis in range(3):
            normal[:, axis] += np.bincount(
                tri[:, corner], weights=face[:, axis], minlength=len(p)
            )
    length = np.linalg.norm(normal, axis=1)
    valid = length > 1.0e-14
    normal[valid] /= length[valid, None]
    normal[~valid] = (0.0, 0.0, 1.0)
    offset = normal * (THICKNESS / 2)
    return np.vstack((p + offset, p - offset)).astype(np.float32)


def update_towel_shell(stage, root_path, world_points):
    """Render-only thickness driven by live solver readback, including replay.

    Omni PhysX 107.3 left the separate surface skin static in live validation.
    Updating its graphics vertices explicitly avoids a frozen or detached skin.
    The simulation midsurface remains the only cloth collider and state source.
    """
    import numpy as np
    from pxr import UsdGeom, Vt, Usd

    skin = UsdGeom.Mesh(stage.GetPrimAtPath(visual_mesh_path(root_path)))
    if not skin:
        return
    world = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
        skin.GetPrim()
    )
    inv = np.asarray(world.GetInverse())
    local = np.c_[np.asarray(world_points), np.ones(len(world_points))] @ inv
    points = shell_points_from_midpoints(local[:, :3])
    skin.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    skin.GetExtentAttr().Set(
        Vt.Vec3fArray.FromNumpy(
            np.array([points.min(0), points.max(0)], dtype=np.float32)
        )
    )
