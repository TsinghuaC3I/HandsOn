"""Scene metadata and authored-USD placement for G01 asset inspection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any


G01_TASK_NAME = "g01_deformable_asset_inspection"
G01_2_TASK_NAME = "g01_2_deformable_asset_inspection"
G01_3_TASK_NAME = "g01_3_bimanual_pillow_to_carton"
G01_4_TASK_NAME = "g01_4_soft_pillow_to_carton"
G01_5_TASK_NAME = "eval_flex03_pack_pillow"
G01_COMPARISON_PROFILES = {G01_4_TASK_NAME: "soft", G01_5_TASK_NAME: "cc0"}
G01_CARTON_TASK_NAMES = (G01_3_TASK_NAME, *G01_COMPARISON_PROFILES)
G01_TASK_NAMES = (G01_TASK_NAME, G01_2_TASK_NAME, *G01_CARTON_TASK_NAMES)
G01_COMPARISON_NODE_COUNTS = {G01_4_TASK_NAME: 721, G01_5_TASK_NAME: 538}
G01_COMPARISON_TET_COUNTS = {G01_4_TASK_NAME: 2832, G01_5_TASK_NAME: 2028}
# G01-5 uses a slightly fuller vertical profile while retaining the authored
# comparison mesh footprint.
G01_5_PILLOW_HALF_HEIGHT_M = 0.110
G01_5_PILLOW_HALF_LENGTH_M = 0.300
G01_5_PILLOW_HALF_WIDTH_M = 0.211
G01_TASK_IDS = dict(zip(G01_TASK_NAMES, ("G01", "G01-2", "G01-3", "G01-4", "G01-5")))
G01_SCENE_XML = (
    "arena_arm_hand_g01_deformable_asset_inspection_rokae_wuji1.xml"
)
G01_3_SCENE_XML = "arena_arm_hand_g01_3_pillow_to_carton_rokae_wuji1.xml"
G01_TABLE_TOP_Z = 0.89
G01_ASSET_ROOT_PATH = "/World/Task/G01DeformableAssets"
G01_PHYSICS_HZ = 60

G01_SUCCESS_CONDITION_LABELS = (
    "Pillow024, Powercable001, and Towl011 are loaded on the tabletop",
    "Either physical Wuji hand touches the green lower-right confirmation button",
    "The physical contact remains valid for three consecutive control frames",
)
G01_3_SUCCESS_CONDITION_LABELS = (
    "Both assigned physical hands contact Pillow024-v2 before it leaves the table",
    "The pillow is lifted at least 30 mm clear of the tabletop",
    "At least 98% of the pillow simulation nodes are inside the open BC004 carton",
    "Both complete hand envelopes release at least 90 mm from the pillow",
    "The released pillow remains contained and stable for one continuous second",
)
G01_5_SUCCESS_CONDITION_LABELS = (
    "Both assigned physical hands contact the foam pillow before lifting it",
    G01_3_SUCCESS_CONDITION_LABELS[1],
    "The pillow rests inside the carton with at least 90% below the rim; raised corners are allowed",
    G01_3_SUCCESS_CONDITION_LABELS[3],
    G01_3_SUCCESS_CONDITION_LABELS[4],
)


@dataclass(frozen=True)
class G01AssetSpec:
    name: str
    relative_usd: str
    translation: tuple[float, float, float]
    rotation_z_degrees: float = 0.0
    uniform_scale: float = 1.0
    local_bounds_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_bounds_max: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def source_path(self) -> Path:
        return G01_ASSET_DIRECTORY / self.relative_usd

    @property
    def placement_path(self) -> str:
        return f"{G01_ASSET_ROOT_PATH}/{self.name}"

    @property
    def reference_path(self) -> str:
        return f"{self.placement_path}/Asset"


G01_ASSET_DIRECTORY = (
    Path(__file__).resolve().parent.parent
    / "envs"
    / "usd"
    / "g01_deformable_asset_inspection"
)

# The table spans x=[-0.55, 0.25], y=[-0.80, 0.80].  The supplier-authored
# scale is retained.  Powercable001 occupies the front, directly reachable
# lane; Pillow024 and Towl011 are parked behind it with collision-shell
# clearance, while the lower-right operator lane remains clear for the button.
G01_ASSET_SPECS = (
    G01AssetSpec(
        name="Pillow024",
        relative_usd="Pillow024/Pillow024.usd",
        translation=(0.035, 0.32, G01_TABLE_TOP_Z + 0.09250 + 0.004),
        rotation_z_degrees=90.0,
        local_bounds_min=(-0.4000, -0.2111, -0.09250),
        local_bounds_max=(0.4000, 0.2111, 0.09250),
    ),
    G01AssetSpec(
        name="Powercable001",
        relative_usd="Powercable001/Powercable001.usd",
        translation=(-0.30, 0.00, G01_TABLE_TOP_Z + 0.000636 + 0.003),
        local_bounds_min=(-0.066538, -0.41082, -0.000636),
        local_bounds_max=(0.085899, 0.41005, 0.004834),
    ),
    G01AssetSpec(
        name="Towl011",
        relative_usd="Towl011/Towl011.usd",
        translation=(0.00, -0.45, G01_TABLE_TOP_Z + 0.004),
        local_bounds_min=(-0.13742, -0.13510, 0.0),
        local_bounds_max=(-0.13742 + 0.27484, -0.13510 + 0.27020, 0.002),
    ),
)

# G01-2 intentionally keeps the verified G01 table layout, cable, towel, and
# operator-success semantics.  Its only scene-content difference is the
# supplier's current-schema Pillow024 revision.  Keeping this as a distinct
# task protects existing G01 recordings from an in-place asset substitution.
G01_2_ASSET_SPECS = (
    G01AssetSpec(
        name="Pillow024",
        relative_usd="Pillow024_v2/Pillow024.usd",
        # The repaired simulation TetMesh extends 0.76 mm below the visual
        # surface.  Place the collision truth, not only the render mesh,
        # exactly 4 mm above the fixed tabletop rest plane.
        translation=(
            0.035,
            0.32,
            G01_TABLE_TOP_Z + 0.09325544 + 0.004,
        ),
        rotation_z_degrees=90.0,
        local_bounds_min=(-0.4000, -0.2111, -0.09325544),
        local_bounds_max=(0.40012142, 0.2111, 0.09324887),
    ),
    *G01_ASSET_SPECS[1:],
)

# G01-3 uses the exact G01-2 pillow size.  The carton footprint, not the
# requested source pillow, is adapted below so this remains a placement task.
G01_3_PILLOW_UNIFORM_SCALE = 1.0
G01_3_PILLOW_LINEAR_DAMPING = 5.0
G01_3_PILLOW_SETTLING_DAMPING = 5.0
G01_3_PILLOW_SETTLING_THRESHOLD_MPS = 0.5
# Pillow024-v2's source material is intentionally validated at its authored
# 3.0/3.0 friction for G01-2 inspection.  G01-3 is a load-bearing bimanual
# grasp, so use a task-local cloth/fill contact profile; retaining 3.0/3.0
# makes a 0.8 kg pillow lock against the small Wuji contact pads before the
# arms can generate a stable lift.
G01_3_PILLOW_STATIC_FRICTION = 0.9
G01_3_PILLOW_DYNAMIC_FRICTION = 0.8
G01_3_ASSET_SPECS = (
    G01AssetSpec(
        name="Pillow024",
        relative_usd="Pillow024_v2/Pillow024.usd",
        translation=(
            -0.43,
            0.0,
            G01_TABLE_TOP_Z
            + 0.09325544 * G01_3_PILLOW_UNIFORM_SCALE
            + 0.004,
        ),
        rotation_z_degrees=90.0,
        uniform_scale=G01_3_PILLOW_UNIFORM_SCALE,
        local_bounds_min=(-0.4000, -0.2111, -0.09325544),
        local_bounds_max=(0.40012142, 0.2111, 0.09324887),
    ),
)

G01_3_CARTON_SOURCE_USD = (
    Path(__file__).resolve().parent.parent
    / "envs"
    / "usd"
    / "b320_bimanual_benchmark"
    / "assets"
    / "dv_carton"
    / "model_carton__rigid_collapsed.usd"
)
G01_3_CARTON_PATH = f"{G01_ASSET_ROOT_PATH}/g01_3_carton"
G01_3_CARTON_SOURCE_BODY_PATH = f"{G01_3_CARTON_PATH}/E_carton_1"
G01_3_CARTON_SOURCE_MESH_PATH = (
    f"{G01_3_CARTON_SOURCE_BODY_PATH}/P_126c3c58f54ffd7c"
)
G01_3_CARTON_TRANSLATION = (0.10, 0.0, G01_TABLE_TOP_Z)
G01_3_CARTON_RENDER_SCALE = (1.20, 1.85, 1.0)
G01_3_CARTON_SOURCE_LOCAL_MIN = (-0.1989, -0.2344, 0.0)
G01_3_CARTON_SOURCE_LOCAL_MAX = (0.2028, 0.2360, 0.2854)
# These are measured source-mesh inner surfaces, not guessed wall thicknesses.
G01_3_CARTON_SOURCE_INNER_LOCAL_MIN = (-0.1943, -0.2302, 0.0020)
G01_3_CARTON_SOURCE_INNER_LOCAL_MAX = (0.1981, 0.2318, 0.2854)
G01_3_CARTON_LOCAL_MIN = tuple(
    value * scale
    for value, scale in zip(
        G01_3_CARTON_SOURCE_LOCAL_MIN, G01_3_CARTON_RENDER_SCALE
    )
)
G01_3_CARTON_LOCAL_MAX = tuple(
    value * scale
    for value, scale in zip(
        G01_3_CARTON_SOURCE_LOCAL_MAX, G01_3_CARTON_RENDER_SCALE
    )
)
G01_3_CARTON_INNER_LOCAL_MIN = tuple(
    value * scale
    for value, scale in zip(
        G01_3_CARTON_SOURCE_INNER_LOCAL_MIN, G01_3_CARTON_RENDER_SCALE
    )
)
G01_3_CARTON_INNER_LOCAL_MAX = tuple(
    value * scale
    for value, scale in zip(
        G01_3_CARTON_SOURCE_INNER_LOCAL_MAX, G01_3_CARTON_RENDER_SCALE
    )
)
G01_3_CARTON_TOP_FACE_COUNT = 14
G01_3_CARTON_OPEN_FACE_COUNT = 30
# Closed inward extrusions of the source panels: no hull spans the opening.
G01_3_CARTON_COLLIDER_PATHS = tuple(
    f"{G01_3_CARTON_SOURCE_BODY_PATH}/OpenContact/{name}"
    for name in ("Wall0", "Wall1", "Wall2", "Wall3", "Floor")
)
G01_3_CARTON_WALL_THICKNESS_M = 0.004
G01_3_CARTON_FLOOR_THICKNESS_M = 0.002


def g01_asset_specs(
    task_name: str = G01_TASK_NAME,
) -> tuple[G01AssetSpec, ...]:
    """Return the immutable asset bundle for one registered G01 variant."""

    if task_name == G01_TASK_NAME:
        return G01_ASSET_SPECS
    if task_name == G01_2_TASK_NAME:
        return G01_2_ASSET_SPECS
    if task_name == G01_3_TASK_NAME:
        return G01_3_ASSET_SPECS
    if task_name in G01_COMPARISON_PROFILES:
        return (G01AssetSpec(
            name="Pillow024",
            relative_usd=f"comparison_pillows/{G01_COMPARISON_PROFILES[task_name]}/Pillow.usda",
            translation=(
                -0.43140994,
                -0.02862781,
                G01_TABLE_TOP_Z
                + (
                    G01_5_PILLOW_HALF_HEIGHT_M
                    if task_name == G01_5_TASK_NAME
                    else 0.09325
                )
                + 0.004,
            ),
            rotation_z_degrees=90.0,
            local_bounds_min=(
                -G01_5_PILLOW_HALF_LENGTH_M
                if task_name == G01_5_TASK_NAME else -0.4,
                -G01_5_PILLOW_HALF_WIDTH_M
                if task_name == G01_5_TASK_NAME else -0.2111,
                -(
                    G01_5_PILLOW_HALF_HEIGHT_M
                    if task_name == G01_5_TASK_NAME
                    else 0.09325
                ),
            ),
            local_bounds_max=(
                G01_5_PILLOW_HALF_LENGTH_M
                if task_name == G01_5_TASK_NAME else 0.4,
                G01_5_PILLOW_HALF_WIDTH_M
                if task_name == G01_5_TASK_NAME else 0.2111,
                G01_5_PILLOW_HALF_HEIGHT_M
                if task_name == G01_5_TASK_NAME
                else 0.09325,
            ),
        ),)
    raise ValueError(
        f"Unsupported G01 task {task_name!r}; expected one of {G01_TASK_NAMES!r}"
    )

G01_PILLOW_BODY_PATH = f"{G01_ASSET_ROOT_PATH}/Pillow024/Asset/Pillow024"
G01_PILLOW_SKIN_MESH_PATH = f"{G01_PILLOW_BODY_PATH}/Visuals/Pillow024"
G01_PILLOW_SIM_MESH_PATH = f"{G01_PILLOW_BODY_PATH}/HandsOnSimulationMesh"
G01_PILLOW_COLLISION_MESH_PATH = (
    f"{G01_PILLOW_BODY_PATH}/HandsOnCollisionMesh"
)
G01_PILLOW_MATERIAL_PATH = "/World/PhysicsMaterials/G01Pillow024"
G01_2_PILLOW_SIM_MESH_PATH = f"{G01_PILLOW_BODY_PATH}/simulation_mesh"
G01_2_PILLOW_COLLISION_MESH_PATH = G01_2_PILLOW_SIM_MESH_PATH
G01_2_PILLOW_MATERIAL_PATH = (
    f"{G01_ASSET_ROOT_PATH}/Pillow024/Asset/Physics/PhysicsMaterial"
)
G01_2_PILLOW_VISUAL_POINT_COUNT = 5_212
G01_2_PILLOW_SIM_POINT_COUNT = 11_643
G01_2_PILLOW_TET_COUNT = 36_327
G01_2_PILLOW_SURFACE_FACE_COUNT = 20_822

G01_TOWEL_LEGACY_BODY_PATH = (
    f"{G01_ASSET_ROOT_PATH}/Towl011/Asset/Towl011"
)
G01_TOWEL_LEGACY_MESH_PATH = f"{G01_TOWEL_LEGACY_BODY_PATH}/Visuals/Towl011"
G01_TOWEL_VISUAL_MATERIAL_PATH = (
    f"{G01_ASSET_ROOT_PATH}/Towl011/Asset/Looks/M_Towl011_001"
)
G01_TOWEL_SURFACE_ROOT_PATH = (
    f"{G01_ASSET_ROOT_PATH}/Towl011/SurfaceDeformable"
)
G01_TOWEL_SURFACE_MESH_PATH = f"{G01_TOWEL_SURFACE_ROOT_PATH}/mesh"
G01_TOWEL_MATERIAL_PATH = "/World/PhysicsMaterials/G01Towl011Cloth"

# Keep the E01 single-mesh surface-deformable contract: render, simulation,
# and collision all use one exact triangulated grid.  The supplier Towl011
# mesh is a closed 13.1 mm volume, which behaves like a rubber mouse pad even
# when marked deformable; this mid-surface grid is the actual foldable towel.
G01_TOWEL_COLUMNS = 41
G01_TOWEL_ROWS = 41
G01_TOWEL_NODE_COUNT = G01_TOWEL_COLUMNS * G01_TOWEL_ROWS
G01_TOWEL_TRIANGLE_COUNT = (
    2 * (G01_TOWEL_COLUMNS - 1) * (G01_TOWEL_ROWS - 1)
)
G01_TOWEL_SIZE_X = 0.27484
G01_TOWEL_SIZE_Y = 0.27020
G01_TOWEL_WRINKLE_AMPLITUDE = 0.002
G01_TOWEL_CONTACT_OFFSET = 0.002
G01_TOWEL_REST_OFFSET = 0.0005
G01_TOWEL_SURFACE_THICKNESS = 0.001
G01_TOWEL_SURFACE_DENSITY = 720.0
G01_TOWEL_DYNAMIC_FRICTION = 0.65
G01_TOWEL_STATIC_FRICTION = 0.75
G01_TOWEL_STRETCH_STIFFNESS = 5.0e3
G01_TOWEL_SHEAR_STIFFNESS = 50.0
G01_TOWEL_BEND_STIFFNESS = 0.3
G01_TOWEL_ELASTICITY_DAMPING = 0.55
G01_TOWEL_MAX_DEPENETRATION_VELOCITY = 0.18
G01_TOWEL_SOLVER_POSITION_ITERATIONS = 16

G01_VOLUME_DEFORMABLE_PRIM_PATHS = {
    "Pillow024": G01_PILLOW_BODY_PATH,
}
G01_SURFACE_DEFORMABLE_PRIM_PATHS = {
    "Towl011": G01_TOWEL_SURFACE_ROOT_PATH,
}
G01_DEFORMABLE_VIEW_SPECS = {
    "Pillow024": ("volume", G01_PILLOW_BODY_PATH),
    "Towl011": ("surface", G01_TOWEL_SURFACE_ROOT_PATH),
}


def g01_deformable_view_specs(
    task_name: str = G01_TASK_NAME,
) -> dict[str, tuple[str, str]]:
    """Return the exact deformable bodies present in one G01 variant."""

    if task_name in G01_CARTON_TASK_NAMES:
        return {"Pillow024": ("volume", G01_PILLOW_BODY_PATH)}
    if task_name in (G01_TASK_NAME, G01_2_TASK_NAME):
        return dict(G01_DEFORMABLE_VIEW_SPECS)
    raise ValueError(f"Unsupported G01 task {task_name!r}")

G01_CABLE_RIGID_BODY_PATHS = tuple(
    f"{G01_ASSET_ROOT_PATH}/Powercable001/Asset/segments/segment_{index:03d}"
    for index in range(64)
) + (
    f"{G01_ASSET_ROOT_PATH}/Powercable001/Asset/"
    "cable_plug011/DataCable011_Plug001",
    f"{G01_ASSET_ROOT_PATH}/Powercable001/Asset/"
    "cable_plug011_01/DataCable011_Plug001",
)


def validate_g01_asset_files(
    task_name: str = G01_TASK_NAME,
) -> tuple[Path, ...]:
    """Return source USDs after validating every external dependency."""

    specs = g01_asset_specs(task_name)
    required = [spec.source_path for spec in specs]
    if task_name in G01_COMPARISON_PROFILES:
        required.append(G01_3_CARTON_SOURCE_USD)
        texture_names = (
            ("Pillow1_baseColor.png", "Pillow1_normal.png")
            if G01_COMPARISON_PROFILES[task_name] == "cc0"
            else ("cotton_albedo.png", "cotton_normal.png")
        )
        required.extend(
            specs[0].source_path.parent / name
            for name in texture_names
        )
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        return tuple(required)
    if task_name == G01_3_TASK_NAME:
        required.extend(
            G01_ASSET_DIRECTORY
            / "Pillow024"
            / "texture"
            / f"T_Pillow024_{channel}001.png"
            for channel in ("BC", "N", "ORM")
        )
        required.append(G01_3_CARTON_SOURCE_USD)
        missing = [path for path in required if not path.is_file()]
        if missing:
            detail = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"G01-3 asset bundle is incomplete: {detail}")
        return tuple(spec.source_path for spec in specs) + (
            G01_3_CARTON_SOURCE_USD,
        )
    required.append(
        G01_ASSET_DIRECTORY / "Powercable001" / "cable_plug011.usd"
    )
    for directory, stem in (
        ("Pillow024", "T_Pillow024"),
        ("Powercable001", "T_cable_plug011"),
        ("Towl011", "T_Towl011"),
    ):
        required.extend(
            G01_ASSET_DIRECTORY
            / directory
            / "texture"
            / f"{stem}_{channel}001.png"
            for channel in ("BC", "N", "ORM")
        )
    missing = [path for path in required if not path.is_file()]
    if missing:
        detail = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"G01 asset bundle is incomplete: {detail}")
    return tuple(spec.source_path for spec in specs)


def g01_towel_surface_topology(
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[int, int, int], ...],
]:
    """Return the deterministic E01-style rest grid for the foldable towel."""

    points: list[tuple[float, float, float]] = []
    x_min = -0.5 * G01_TOWEL_SIZE_X
    y_min = -0.5 * G01_TOWEL_SIZE_Y
    for row in range(G01_TOWEL_ROWS):
        v = row / float(G01_TOWEL_ROWS - 1)
        y = y_min + v * G01_TOWEL_SIZE_Y
        for column in range(G01_TOWEL_COLUMNS):
            u = column / float(G01_TOWEL_COLUMNS - 1)
            x = x_min + u * G01_TOWEL_SIZE_X
            envelope = math.sin(math.pi * u) * math.sin(math.pi * v)
            wrinkle = 0.5 + 0.5 * math.sin(4.0 * math.pi * u) * math.sin(
                4.0 * math.pi * v
            )
            z = G01_TOWEL_WRINKLE_AMPLITUDE * envelope * wrinkle
            points.append((x, y, z))

    triangles: list[tuple[int, int, int]] = []
    for row in range(G01_TOWEL_ROWS - 1):
        for column in range(G01_TOWEL_COLUMNS - 1):
            a = row * G01_TOWEL_COLUMNS + column
            b = a + 1
            c = a + G01_TOWEL_COLUMNS
            d = c + 1
            triangles.extend(((a, b, d), (a, d, c)))
    return tuple(points), tuple(triangles)


def _remove_legacy_volume_schema(prim: Any, *, label: str) -> None:
    """Suppress a referenced legacy soft body before enabling the beta pipeline."""

    if prim is None or not prim.IsValid():
        raise RuntimeError(f"G01 {label} legacy deformable prim did not compose")
    if prim.HasAPI("PhysxDeformableBodyAPI"):
        if not prim.RemoveAPI("PhysxDeformableBodyAPI"):
            raise RuntimeError(f"Could not remove G01 {label} legacy deformable API")
    if prim.HasAPI("PhysxCollisionAPI"):
        prim.RemoveAPI("PhysxCollisionAPI")
    physics_binding = prim.GetRelationship("material:binding:physics")
    if physics_binding.IsValid():
        physics_binding.ClearTargets(True)


def _apply_g01_body_tuning(
    root_prim: Any,
    *,
    self_collision: bool,
    max_linear_velocity: float,
    max_depenetration_velocity: float,
    solver_position_iterations: int,
    linear_damping: float,
) -> None:
    if not root_prim.HasAPI("PhysxBaseDeformableBodyAPI") and not root_prim.ApplyAPI(
        "PhysxBaseDeformableBodyAPI"
    ):
        raise RuntimeError(
            f"Could not apply PhysX deformable tuning at {root_prim.GetPath()}"
        )
    attributes = {
        "physxDeformableBody:selfCollision": bool(self_collision),
        "physxDeformableBody:enableSpeculativeCCD": True,
        "physxDeformableBody:solverPositionIterationCount": int(
            solver_position_iterations
        ),
        "physxDeformableBody:maxLinearVelocity": float(max_linear_velocity),
        "physxDeformableBody:maxDepenetrationVelocity": float(
            max_depenetration_velocity
        ),
        "physxDeformableBody:linearDamping": float(linear_damping),
    }
    for name, value in attributes.items():
        attribute = root_prim.GetAttribute(name)
        if not attribute.IsValid():
            raise RuntimeError(f"Missing deformable-body attribute {name}")
        attribute.Set(value)


def _migrate_g01_pillow_volume_deformable(stage: Any) -> None:
    """Reuse the supplier tet meshes in the current PhysX volume pipeline."""

    from omni.physx.scripts import deformableUtils, physicsUtils
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    root_prim = stage.GetPrimAtPath(G01_PILLOW_BODY_PATH)
    skin_prim = stage.GetPrimAtPath(G01_PILLOW_SKIN_MESH_PATH)
    if root_prim is None or not root_prim.IsValid():
        raise RuntimeError("G01 Pillow024 body root did not compose")
    if skin_prim is None or not skin_prim.IsA(UsdGeom.Mesh):
        raise RuntimeError("G01 Pillow024 skin mesh did not compose")

    simulation_points = skin_prim.GetAttribute(
        "physxDeformable:simulationRestPoints"
    ).Get()
    simulation_indices = skin_prim.GetAttribute(
        "physxDeformable:simulationIndices"
    ).Get()
    collision_points = skin_prim.GetAttribute(
        "physxDeformable:collisionRestPoints"
    ).Get()
    collision_indices = skin_prim.GetAttribute(
        "physxDeformable:collisionIndices"
    ).Get()
    if not all(
        value is not None and len(value) > 0
        for value in (
            simulation_points,
            simulation_indices,
            collision_points,
            collision_indices,
        )
    ):
        raise RuntimeError("G01 Pillow024 legacy tetrahedral data is incomplete")
    if len(simulation_indices) % 4 or len(collision_indices) % 4:
        raise RuntimeError("G01 Pillow024 tetrahedral indices are malformed")

    _remove_legacy_volume_schema(skin_prim, label="Pillow024")

    simulation_mesh = UsdGeom.TetMesh.Define(stage, G01_PILLOW_SIM_MESH_PATH)
    simulation_mesh.GetPointsAttr().Set(simulation_points)
    simulation_tets = [
        Gf.Vec4i(*map(int, simulation_indices[offset : offset + 4]))
        for offset in range(0, len(simulation_indices), 4)
    ]
    simulation_mesh.GetTetVertexIndicesAttr().Set(simulation_tets)
    simulation_prim = simulation_mesh.GetPrim()
    if not simulation_prim.ApplyAPI("OmniPhysicsVolumeDeformableSimAPI"):
        raise RuntimeError("Could not apply G01 Pillow024 volume simulation API")
    simulation_prim.GetAttribute("omniphysics:restShapePoints").Set(
        simulation_points
    )
    simulation_prim.GetAttribute("omniphysics:restTetVtxIndices").Set(
        simulation_tets
    )

    collision_mesh = UsdGeom.TetMesh.Define(
        stage,
        G01_PILLOW_COLLISION_MESH_PATH,
    )
    collision_mesh.GetPointsAttr().Set(collision_points)
    collision_tets = [
        Gf.Vec4i(*map(int, collision_indices[offset : offset + 4]))
        for offset in range(0, len(collision_indices), 4)
    ]
    collision_mesh.GetTetVertexIndicesAttr().Set(collision_tets)
    collision_prim = collision_mesh.GetPrim()
    if not collision_prim.ApplyAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("Could not apply G01 Pillow024 volume collision API")
    collision_mesh.GetSurfaceFaceVertexIndicesAttr().Set(
        UsdGeom.TetMesh.ComputeSurfaceFaces(
            collision_mesh,
            Usd.TimeCode.Default(),
        )
    )

    purposes = ["bindPose"]
    skin_prim.ApplyAPI("OmniPhysicsDeformablePoseAPI", "default")
    skin_prim.CreateAttribute(
        "deformablePose:default:omniphysics:purposes",
        Sdf.ValueTypeNames.TokenArray,
    ).Set(purposes)
    skin_prim.CreateAttribute(
        "deformablePose:default:omniphysics:points",
        Sdf.ValueTypeNames.Point3fArray,
    ).Set(UsdGeom.Mesh(skin_prim).GetPointsAttr().Get())
    for prim, bind_pose_points in (
        (simulation_prim, simulation_points),
        (collision_prim, collision_points),
    ):
        prim.ApplyAPI("OmniPhysicsDeformablePoseAPI", "default")
        prim.CreateAttribute(
            "deformablePose:default:omniphysics:purposes",
            Sdf.ValueTypeNames.TokenArray,
        ).Set(purposes)
        prim.CreateAttribute(
            "deformablePose:default:omniphysics:points",
            Sdf.ValueTypeNames.Point3fArray,
        ).Set(bind_pose_points)

    if not root_prim.ApplyAPI("OmniPhysicsDeformableBodyAPI"):
        raise RuntimeError("Could not apply G01 Pillow024 deformable-body API")
    _apply_g01_body_tuning(
        root_prim,
        self_collision=False,
        max_linear_velocity=0.5,
        max_depenetration_velocity=0.25,
        solver_position_iterations=16,
        linear_damping=0.15,
    )

    collision_prim.ApplyAPI(PhysxSchema.PhysxCollisionAPI)
    collision_api = PhysxSchema.PhysxCollisionAPI(collision_prim)
    collision_api.GetContactOffsetAttr().Set(0.008)
    collision_api.GetRestOffsetAttr().Set(0.004)

    if not deformableUtils.add_deformable_material(
        stage,
        G01_PILLOW_MATERIAL_PATH,
        density=27.0,
        static_friction=0.9,
        dynamic_friction=0.8,
        youngs_modulus=2.0e4,
        poissons_ratio=0.35,
    ):
        raise RuntimeError("Could not create G01 Pillow024 deformable material")
    physicsUtils.add_physics_material_to_prim(
        stage,
        root_prim,
        G01_PILLOW_MATERIAL_PATH,
    )
    material_prim = stage.GetPrimAtPath(G01_PILLOW_MATERIAL_PATH)
    material_prim.ApplyAPI("PhysxDeformableMaterialAPI")
    material_prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(
        0.20
    )
    mass_api = UsdPhysics.MassAPI(root_prim)
    if not mass_api:
        mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.CreateMassAttr().Set(1.2)
    root_prim.SetCustomDataByKey("handson:assetType", "volume_deformable_pillow")


def _g01_api_schema_names(prim: Any) -> set[str]:
    """Read built-in and codeless API tokens from a composed prim."""

    names = set(map(str, prim.GetAppliedSchemas()))
    authored = prim.GetMetadata("apiSchemas")
    if authored is not None:
        names.update(map(str, authored.GetAddedOrExplicitItems()))
    return names


def _validate_g01_2_pillow_volume_deformable(stage: Any) -> None:
    """Fail closed unless Pillow024-v2 starts in its normalized rest state."""

    import numpy as np
    from pxr import UsdGeom

    root_prim = stage.GetPrimAtPath(G01_PILLOW_BODY_PATH)
    skin_prim = stage.GetPrimAtPath(G01_PILLOW_SKIN_MESH_PATH)
    simulation_prim = stage.GetPrimAtPath(G01_2_PILLOW_SIM_MESH_PATH)
    material_prim = stage.GetPrimAtPath(G01_2_PILLOW_MATERIAL_PATH)
    if root_prim is None or not root_prim.IsValid():
        raise RuntimeError("G01-2 Pillow024-v2 body root did not compose")
    if skin_prim is None or not skin_prim.IsA(UsdGeom.Mesh):
        raise RuntimeError("G01-2 Pillow024-v2 visual mesh did not compose")
    if simulation_prim is None or not simulation_prim.IsA(UsdGeom.TetMesh):
        raise RuntimeError("G01-2 Pillow024-v2 simulation TetMesh did not compose")
    if material_prim is None or not material_prim.IsValid():
        raise RuntimeError("G01-2 Pillow024-v2 physics material did not compose")

    root_schemas = _g01_api_schema_names(root_prim)
    required_root_schemas = {
        "PhysxAutoDeformableBodyAPI",
        "OmniPhysicsDeformableBodyAPI",
        "PhysxBaseDeformableBodyAPI",
    }
    missing_root_schemas = required_root_schemas - root_schemas
    if missing_root_schemas:
        raise RuntimeError(
            "G01-2 Pillow024-v2 is missing current body schemas: "
            + ", ".join(sorted(missing_root_schemas))
        )
    simulation_schemas = _g01_api_schema_names(simulation_prim)
    required_simulation_schemas = {
        "OmniPhysicsVolumeDeformableSimAPI",
        "PhysicsCollisionAPI",
        "PhysxCollisionAPI",
    }
    missing_simulation_schemas = required_simulation_schemas - simulation_schemas
    if missing_simulation_schemas:
        raise RuntimeError(
            "G01-2 Pillow024-v2 is missing current TetMesh schemas: "
            + ", ".join(sorted(missing_simulation_schemas))
        )
    if "PhysicsMassAPI" in _g01_api_schema_names(skin_prim):
        raise RuntimeError(
            "G01-2 Pillow024-v2 retains PhysicsMassAPI on its visual child"
        )

    skin = UsdGeom.Mesh(skin_prim)
    skin_points = np.asarray(skin.GetPointsAttr().Get(), dtype=np.float64)
    bind_points = np.asarray(
        skin_prim.GetAttribute(
            "deformablePose:default:omniphysics:points"
        ).Get(),
        dtype=np.float64,
    )
    simulation_mesh = UsdGeom.TetMesh(simulation_prim)
    simulation_points = np.asarray(
        simulation_mesh.GetPointsAttr().Get(), dtype=np.float64
    )
    rest_points = np.asarray(
        simulation_prim.GetAttribute("omniphysics:restShapePoints").Get(),
        dtype=np.float64,
    )
    velocities = np.asarray(
        simulation_mesh.GetVelocitiesAttr().Get(), dtype=np.float64
    )
    tets = simulation_mesh.GetTetVertexIndicesAttr().Get()
    surface_faces = simulation_mesh.GetSurfaceFaceVertexIndicesAttr().Get()
    if skin_points.shape != (G01_2_PILLOW_VISUAL_POINT_COUNT, 3):
        raise RuntimeError(
            f"G01-2 Pillow024-v2 visual point shape is {skin_points.shape}"
        )
    if simulation_points.shape != (G01_2_PILLOW_SIM_POINT_COUNT, 3):
        raise RuntimeError(
            "G01-2 Pillow024-v2 simulation point shape is "
            f"{simulation_points.shape}"
        )
    if len(tets) != G01_2_PILLOW_TET_COUNT:
        raise RuntimeError(
            f"G01-2 Pillow024-v2 has {len(tets)} tetrahedra, expected "
            f"{G01_2_PILLOW_TET_COUNT}"
        )
    if len(surface_faces) != G01_2_PILLOW_SURFACE_FACE_COUNT:
        raise RuntimeError(
            f"G01-2 Pillow024-v2 has {len(surface_faces)} surface faces, "
            f"expected {G01_2_PILLOW_SURFACE_FACE_COUNT}"
        )
    if bind_points.shape != skin_points.shape or not np.array_equal(
        skin_points, bind_points
    ):
        raise RuntimeError(
            "G01-2 Pillow024-v2 visual points are not the authored bind pose"
        )
    if rest_points.shape != simulation_points.shape or not np.array_equal(
        simulation_points, rest_points
    ):
        raise RuntimeError(
            "G01-2 Pillow024-v2 TetMesh points are not the rest shape"
        )
    if velocities.shape != simulation_points.shape or np.any(velocities != 0.0):
        raise RuntimeError(
            "G01-2 Pillow024-v2 must have one zero velocity per TetMesh point"
        )

    for label, points, imageable in (
        ("visual", skin_points, skin),
        ("simulation", simulation_points, simulation_mesh),
    ):
        extent = np.asarray(imageable.GetExtentAttr().Get(), dtype=np.float64)
        expected_extent = np.stack(
            (np.min(points, axis=0), np.max(points, axis=0))
        )
        if extent.shape != (2, 3) or not np.allclose(
            extent, expected_extent, rtol=0.0, atol=1.0e-6
        ):
            raise RuntimeError(
                f"G01-2 Pillow024-v2 {label} extent is stale"
            )

    for attribute_name in (
        "physics:mass",
        "physics:density",
        "physics:centerOfMass",
        "physics:diagonalInertia",
        "physics:principalAxes",
    ):
        attribute = skin_prim.GetAttribute(attribute_name)
        if attribute.IsValid() and attribute.HasAuthoredValueOpinion():
            raise RuntimeError(
                "G01-2 Pillow024-v2 retains a conflicting visual-child "
                f"mass property: {attribute_name}"
            )
    if skin_prim.GetRelationship("material:binding:physics").IsValid():
        raise RuntimeError(
            "G01-2 Pillow024-v2 visual child retains a physics material binding"
        )
    visibility = UsdGeom.Imageable(simulation_prim).GetVisibilityAttr().Get()
    if visibility != UsdGeom.Tokens.invisible:
        raise RuntimeError(
            "G01-2 Pillow024-v2 simulation TetMesh must be explicitly invisible"
        )
    root_mass = root_prim.GetAttribute("omniphysics:mass").Get()
    if root_mass is None or abs(float(root_mass) - 0.8) > 1.0e-6:
        raise RuntimeError(
            f"G01-2 Pillow024-v2 root mass is {root_mass!r}, expected 0.8 kg"
        )

    root_parameter_expectations = {
        "physxDeformableBody:enableSpeculativeCCD": True,
        "physxDeformableBody:selfCollision": True,
        "physxDeformableBody:solverPositionIterationCount": 32,
        "physxDeformableBody:linearDamping": 0.1,
    }
    for attribute_name, expected in root_parameter_expectations.items():
        actual = root_prim.GetAttribute(attribute_name).Get()
        if isinstance(expected, bool):
            matches = actual is not None and bool(actual) is expected
        elif isinstance(expected, int):
            matches = actual is not None and int(actual) == expected
        else:
            matches = actual is not None and abs(float(actual) - expected) <= 1.0e-6
        if not matches:
            raise RuntimeError(
                f"G01-2 Pillow024-v2 {attribute_name} is {actual!r}, "
                f"expected {expected!r}"
            )

    simulation_parameter_expectations = {
        "physxCollision:contactOffset": 0.008,
        "physxCollision:restOffset": 0.004,
    }
    for attribute_name, expected in simulation_parameter_expectations.items():
        actual = simulation_prim.GetAttribute(attribute_name).Get()
        if actual is None or abs(float(actual) - expected) > 1.0e-6:
            raise RuntimeError(
                f"G01-2 Pillow024-v2 {attribute_name} is {actual!r}, "
                f"expected {expected}"
            )

    expected_material_target = str(material_prim.GetPath())
    for label, prim in (("root", root_prim), ("simulation", simulation_prim)):
        relationship = prim.GetRelationship("material:binding:physics")
        targets = list(map(str, relationship.GetTargets()))
        if targets != [expected_material_target]:
            raise RuntimeError(
                f"G01-2 Pillow024-v2 {label} physics material targets are "
                f"{targets!r}, expected {[expected_material_target]!r}"
            )

    material_expectations = {
        "omniphysics:youngsModulus": 1.0e6,
        "omniphysics:poissonsRatio": 0.48,
        "omniphysics:staticFriction": 3.0,
        "omniphysics:dynamicFriction": 3.0,
    }
    for attribute_name, expected in material_expectations.items():
        actual = material_prim.GetAttribute(attribute_name).Get()
        if actual is None or abs(float(actual) - expected) > max(
            1.0e-6, abs(expected) * 1.0e-6
        ):
            raise RuntimeError(
                f"G01-2 Pillow024-v2 {attribute_name} is {actual!r}, "
                f"expected {expected}"
            )

    root_prim.SetCustomDataByKey(
        "handson:assetType", "volume_deformable_pillow_v2"
    )
    root_prim.SetCustomDataByKey("handson:normalizedRestState", True)
    root_prim.SetCustomDataByKey(
        "handson:tetBoundaryNonManifoldEdgeCount", 39
    )


def _author_g01_towel_surface_deformable(stage: Any) -> None:
    """Replace the thick supplier soft body with an E01-style cloth sheet."""

    from omni.physx.scripts import deformableUtils, physicsUtils
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    legacy_root = stage.GetPrimAtPath(G01_TOWEL_LEGACY_BODY_PATH)
    legacy_mesh = stage.GetPrimAtPath(G01_TOWEL_LEGACY_MESH_PATH)
    if legacy_root is None or not legacy_root.IsValid():
        raise RuntimeError("G01 Towl011 source root did not compose")
    _remove_legacy_volume_schema(legacy_mesh, label="Towl011")
    UsdGeom.Imageable(legacy_root).MakeInvisible()

    root = UsdGeom.Xform.Define(stage, G01_TOWEL_SURFACE_ROOT_PATH)
    root_prim = root.GetPrim()
    root_prim.SetCustomDataByKey(
        "handson:assetType",
        "surface_deformable_towel",
    )
    root_prim.SetCustomDataByKey("handson:nodeCount", G01_TOWEL_NODE_COUNT)
    root_prim.SetCustomDataByKey(
        "handson:triangleCount",
        G01_TOWEL_TRIANGLE_COUNT,
    )

    points, triangles = g01_towel_surface_topology()
    mesh = UsdGeom.Mesh.Define(stage, G01_TOWEL_SURFACE_MESH_PATH)
    mesh_points = [Gf.Vec3f(*map(float, point)) for point in points]
    mesh.GetPointsAttr().Set(mesh_points)
    mesh.GetFaceVertexCountsAttr().Set([3] * len(triangles))
    flat_indices = [index for triangle in triangles for index in triangle]
    mesh.GetFaceVertexIndicesAttr().Set(flat_indices)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(True)
    mesh.CreateExtentAttr().Set(
        [
            Gf.Vec3f(-0.5 * G01_TOWEL_SIZE_X, -0.5 * G01_TOWEL_SIZE_Y, 0.0),
            Gf.Vec3f(
                0.5 * G01_TOWEL_SIZE_X,
                0.5 * G01_TOWEL_SIZE_Y,
                G01_TOWEL_WRINKLE_AMPLITUDE,
            ),
        ]
    )

    uv_by_node = [
        Gf.Vec2f(
            (point[0] + 0.5 * G01_TOWEL_SIZE_X) / G01_TOWEL_SIZE_X,
            (point[1] + 0.5 * G01_TOWEL_SIZE_Y) / G01_TOWEL_SIZE_Y,
        )
        for point in points
    ]
    uv_primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    uv_primvar.Set([uv_by_node[index] for index in flat_indices])

    source_material_prim = stage.GetPrimAtPath(G01_TOWEL_VISUAL_MATERIAL_PATH)
    if source_material_prim is None or not source_material_prim.IsA(
        UsdShade.Material
    ):
        raise RuntimeError("G01 Towl011 visual material did not compose")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        UsdShade.Material(source_material_prim),
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )

    sim_prim = mesh.GetPrim()
    if not sim_prim.ApplyAPI("OmniPhysicsSurfaceDeformableSimAPI"):
        raise RuntimeError("Could not apply G01 Towl011 surface simulation API")
    if not sim_prim.ApplyAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("Could not apply G01 Towl011 surface collision API")
    sim_prim.GetAttribute("omniphysics:restShapePoints").Set(mesh_points)
    sim_prim.GetAttribute("omniphysics:restTriVtxIndices").Set(
        [Gf.Vec3i(*triangle) for triangle in triangles]
    )
    if not root_prim.ApplyAPI("OmniPhysicsDeformableBodyAPI"):
        raise RuntimeError("Could not apply G01 Towl011 deformable-body API")
    if not root_prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI"):
        raise RuntimeError("Could not apply G01 Towl011 PhysX surface API")
    _apply_g01_body_tuning(
        root_prim,
        self_collision=True,
        max_linear_velocity=G01_TOWEL_CONTACT_OFFSET * G01_PHYSICS_HZ,
        max_depenetration_velocity=G01_TOWEL_MAX_DEPENETRATION_VELOCITY,
        solver_position_iterations=G01_TOWEL_SOLVER_POSITION_ITERATIONS,
        linear_damping=0.0,
    )
    for name, value in {
        "physxDeformableBody:collisionPairUpdateFrequency": 4,
        "physxDeformableBody:collisionIterationMultiplier": 4,
    }.items():
        attribute = root_prim.GetAttribute(name)
        if not attribute.IsValid():
            raise RuntimeError(f"Missing surface-deformable attribute {name}")
        attribute.Set(value)

    sim_prim.ApplyAPI(PhysxSchema.PhysxCollisionAPI)
    collision_api = PhysxSchema.PhysxCollisionAPI(sim_prim)
    collision_api.GetContactOffsetAttr().Set(G01_TOWEL_CONTACT_OFFSET)
    collision_api.GetRestOffsetAttr().Set(G01_TOWEL_REST_OFFSET)

    if not deformableUtils.add_surface_deformable_material(
        stage,
        G01_TOWEL_MATERIAL_PATH,
        density=G01_TOWEL_SURFACE_DENSITY,
        static_friction=G01_TOWEL_STATIC_FRICTION,
        dynamic_friction=G01_TOWEL_DYNAMIC_FRICTION,
        surface_thickness=G01_TOWEL_SURFACE_THICKNESS,
        surface_stretch_stiffness=G01_TOWEL_STRETCH_STIFFNESS,
        surface_shear_stiffness=G01_TOWEL_SHEAR_STIFFNESS,
        surface_bend_stiffness=G01_TOWEL_BEND_STIFFNESS,
    ):
        raise RuntimeError("Could not create G01 Towl011 surface material")
    physicsUtils.add_physics_material_to_prim(
        stage,
        root_prim,
        G01_TOWEL_MATERIAL_PATH,
    )
    material_prim = stage.GetPrimAtPath(G01_TOWEL_MATERIAL_PATH)
    material_prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
    material_prim.GetAttribute(
        "physxDeformableMaterial:elasticityDamping"
    ).Set(G01_TOWEL_ELASTICITY_DAMPING)
    material_prim.GetAttribute("physxDeformableMaterial:bendDamping").Set(
        G01_TOWEL_ELASTICITY_DAMPING
    )


def _g01_3_carton_contact_meshes(points, triangles):
    """Extrude the four outer source panels inward and the bottom upward.

    BC004 is a bevelled closed solid, not a hollow carton: after removing its
    top it has only outward-facing triangles. Double-sided rendering does not
    create inward PhysX contacts. Each separate closed convex panel keeps the
    source outer plane and cannot introduce an invisible lid over the cavity.
    Dimensions here are source-local, before the existing render scale.
    """
    import numpy as np
    from scipy.spatial import ConvexHull

    points = np.asarray(points, dtype=np.float64)
    groups = []
    for triangle in np.asarray(triangles).reshape(-1, 3):
        vertices = points[triangle]
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        area2 = np.linalg.norm(normal)
        if area2 < 0.01:  # Source bevel strips are not full wall panels.
            continue
        normal /= area2
        if normal[2] > 0.01:
            continue
        for direction, members in groups:
            if np.dot(normal, direction) > 0.99999:
                members.extend(triangle.tolist())
                break
        else:
            groups.append((normal, triangle.tolist()))
    walls = [(n, ids) for n, ids in groups if abs(n[2]) < 0.01]
    floors = [(n, ids) for n, ids in groups if n[2] < -0.99]
    if len(walls) != 4 or len(floors) != 1:
        raise RuntimeError("BC004 must have four source panels and one bottom")
    result = []
    for normal, indices in walls + floors:
        outer = np.unique(points[indices], axis=0)
        if normal[2] < -0.99:
            # Almost-collinear bevel vertices make PhysX reject a 2 mm hull
            # for GPU contact. The source bottom's planar rectangle needs
            # only four corners; keep its exact support height and bounds.
            lower, upper = outer.min(0), outer.max(0)
            outer = np.array([[x, y, lower[2]] for x in (lower[0], upper[0])
                              for y in (lower[1], upper[1])])
        thickness = (G01_3_CARTON_FLOOR_THICKNESS_M if normal[2] < -0.99
                     else G01_3_CARTON_WALL_THICKNESS_M)
        vertices = np.vstack((outer, outer - normal * thickness))
        hull = ConvexHull(vertices)
        faces = hull.simplices.copy()
        for i, face in enumerate(faces):
            a, b, c = vertices[face]
            if np.dot(np.cross(b - a, c - a), hull.equations[i, :3]) < 0:
                faces[i, 1], faces[i, 2] = faces[i, 2], faces[i, 1]
        result.append((vertices, faces))
    return result


def _g01_3_carton_interior_floor(points, triangles):
    """Move the source bottom's visible face onto its physical inner surface."""
    import numpy as np

    points = np.asarray(points, dtype=np.float64).copy()
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3).copy()
    floor_faces = np.all(
        np.isclose(points[triangles, 2], G01_3_CARTON_SOURCE_LOCAL_MIN[2], atol=1e-7),
        axis=1,
    )
    if np.count_nonzero(floor_faces) != 2:
        raise RuntimeError("BC004 visible bottom must contain two source triangles")
    floor_ids = np.unique(triangles[floor_faces])
    if np.intersect1d(floor_ids, triangles[~floor_faces]).size:
        raise RuntimeError("BC004 floor must retain separate source face vertices")
    # The original underside is coplanar with the table and z-fights away.
    # Reuse its texture coordinates on the top of the existing 2 mm collider.
    points[floor_ids, 2] = G01_3_CARTON_SOURCE_INNER_LOCAL_MIN[2]
    for index in np.flatnonzero(floor_faces):
        a, b, c = points[triangles[index]]
        if np.cross(b - a, c - a)[2] < 0:
            triangles[index] = triangles[index, ::-1]
    return points, triangles, floor_ids


def _author_g01_3_open_carton(stage: Any, task_name: str = G01_3_TASK_NAME) -> None:
    """Keep the open BC004 visual and derive five closed contact panels."""

    import numpy as np
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Vt

    carton = UsdGeom.Xform.Define(stage, G01_3_CARTON_PATH)
    carton.AddTranslateOp().Set(Gf.Vec3d(*G01_3_CARTON_TRANSLATION))
    carton_prim = carton.GetPrim()
    carton_prim.GetReferences().AddReference(str(G01_3_CARTON_SOURCE_USD.resolve()))
    carton_prim.SetCustomDataByKey("handson:sourceTask", "B320-BC004")
    carton_prim.SetCustomDataByKey(
        "handson:sourceAsset", "dv_carton/model_carton__rigid_collapsed.usd"
    )
    carton_prim.SetCustomDataByKey("handson:derivedUse", "open_top_container")

    source_body_prim = stage.GetPrimAtPath(G01_3_CARTON_SOURCE_BODY_PATH)
    source_mesh_prim = stage.GetPrimAtPath(G01_3_CARTON_SOURCE_MESH_PATH)
    if source_body_prim is None or not source_body_prim.IsValid():
        raise RuntimeError("G01-3 BC004 carton source body did not compose")
    if source_mesh_prim is None or not source_mesh_prim.IsA(UsdGeom.Mesh):
        raise RuntimeError("G01-3 BC004 carton source mesh did not compose")
    source_body_xform = UsdGeom.Xformable(source_body_prim)
    source_body_xform.ClearXformOpOrder()
    source_body_xform.AddScaleOp().Set(Gf.Vec3f(*G01_3_CARTON_RENDER_SCALE))
    source_mesh = UsdGeom.Mesh(source_mesh_prim)
    points = np.asarray(source_mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(
        source_mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64
    )
    indices = np.asarray(
        source_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64
    )
    if points.shape != (132, 3) or counts.shape != (44,):
        raise RuntimeError(
            "G01-3 BC004 carton topology changed: "
            f"points={points.shape}, faces={counts.shape}"
        )
    if not np.allclose(
        np.min(points, axis=0),
        np.asarray(G01_3_CARTON_SOURCE_LOCAL_MIN),
        rtol=0.0,
        atol=5.0e-4,
    ) or not np.allclose(
        np.max(points, axis=0),
        np.asarray(G01_3_CARTON_SOURCE_LOCAL_MAX),
        rtol=0.0,
        atol=5.0e-4,
    ):
        raise RuntimeError("G01-3 BC004 carton bounds changed")

    top_faces: list[int] = []
    offset = 0
    for face_index, count in enumerate(counts.tolist()):
        face_indices = indices[offset : offset + int(count)]
        offset += int(count)
        if float(np.min(points[face_indices, 2])) > 0.27:
            top_faces.append(face_index)
    if len(top_faces) != G01_3_CARTON_TOP_FACE_COUNT:
        raise RuntimeError(
            "G01-3 BC004 carton open-top face selection changed: "
            f"{len(top_faces)} != {G01_3_CARTON_TOP_FACE_COUNT}"
        )

    kept_counts: list[int] = []
    kept_indices: list[int] = []
    offset = 0
    top_face_set = set(top_faces)
    for face_index, count in enumerate(counts.tolist()):
        face_indices = indices[offset : offset + int(count)]
        offset += int(count)
        if face_index in top_face_set:
            continue
        kept_counts.append(int(count))
        kept_indices.extend(map(int, face_indices))
    if len(kept_counts) != G01_3_CARTON_OPEN_FACE_COUNT:
        raise RuntimeError("G01-3 BC004 open render face count changed")
    render_points, render_triangles, floor_ids = _g01_3_carton_interior_floor(
        points, kept_indices
    )
    source_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(render_points.astype(np.float32)))
    source_mesh.GetFaceVertexCountsAttr().Set(kept_counts)
    source_mesh.GetFaceVertexIndicesAttr().Set(render_triangles.flatten().tolist())
    normals = np.asarray(source_mesh.GetNormalsAttr().Get(), dtype=np.float32).copy()
    if source_mesh.GetNormalsInterpolation() != "vertex" or normals.shape != points.shape:
        raise RuntimeError("BC004 source vertex normals changed")
    normals[floor_ids] = (0.0, 0.0, 1.0)
    source_mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
    source_mesh.CreateDoubleSidedAttr().Set(True)
    source_mesh_prim.SetCustomDataByKey(
        "handson:removedSourceTopFaceCount", len(top_faces)
    )
    source_mesh_prim.SetCustomDataByKey(
        "handson:openRenderFaceCount", len(kept_counts)
    )
    for schema in (
        PhysxSchema.PhysxSDFMeshCollisionAPI,
        PhysxSchema.PhysxConvexHullCollisionAPI,
    ):
        if source_mesh_prim.HasAPI(schema):
            source_mesh_prim.RemoveAPI(schema)
    UsdPhysics.CollisionAPI.Apply(source_mesh_prim).CreateCollisionEnabledAttr(False)
    panels = _g01_3_carton_contact_meshes(points, kept_indices)
    for path, (vertices, faces) in zip(G01_3_CARTON_COLLIDER_PATHS, panels):
        if path.endswith("/Floor"):
            # The source floor is only 2 mm thick over an 85 cm span. PhysX
            # rejects even its eight-vertex convex hull for GPU deformables;
            # the equivalent native box has no convex cooking requirement.
            lower, upper = vertices.min(0), vertices.max(0)
            shape = UsdGeom.Cube.Define(stage, path)
            shape.CreateSizeAttr(1.0)
            shape.AddTranslateOp().Set(Gf.Vec3d(*((lower + upper) / 2)))
            shape.AddScaleOp().Set(Gf.Vec3f(*(upper - lower)))
        else:
            shape = UsdGeom.Mesh.Define(stage, path)
            shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
            shape.CreateFaceVertexCountsAttr([3] * len(faces))
            shape.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
            shape.CreateSubdivisionSchemeAttr("none")
            shape.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(
                np.array([vertices.min(0), vertices.max(0)], dtype=np.float32)
            ))
            UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr("convexHull")
        shape.CreateVisibilityAttr("invisible")
        prim = shape.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        tuning = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        tuning.CreateContactOffsetAttr(0.006)
        tuning.CreateRestOffsetAttr(0.0)
        prim.SetCustomDataByKey("handson:collisionSource", "bc004_source_panel_extrusion")
        prim.SetCustomDataByKey("handson:sourceMesh", G01_3_CARTON_SOURCE_MESH_PATH)
    # Merely setting rigidBodyEnabled=false leaves the applied API in the
    # hierarchy.  PhysX still diagnoses that authored child as a nested rigid
    # body below our fixed carton root, and the invalid hierarchy can poison
    # the deformable tensor readback.  Remove the source schema completely;
    # the root rigid body below owns the five closed source-panel colliders.
    if source_body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        source_body_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

    rigid = UsdPhysics.RigidBodyAPI.Apply(carton_prim)
    rigid.CreateRigidBodyEnabledAttr().Set(True)
    rigid.CreateKinematicEnabledAttr().Set(True)
    rigid_tuning = PhysxSchema.PhysxRigidBodyAPI.Apply(carton_prim)
    rigid_tuning.CreateDisableGravityAttr().Set(True)

    physics_material = UsdShade.Material.Define(
        stage, f"{G01_3_CARTON_PATH}/Physics/CartonMaterial"
    )
    material_api = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    # Let the foam slide down a wall during an angled placement instead of
    # storing strain against a high-friction rim and springing back out.
    material_api.CreateStaticFrictionAttr().Set(0.45 if task_name == G01_5_TASK_NAME else 0.9)
    material_api.CreateDynamicFrictionAttr().Set(0.35 if task_name == G01_5_TASK_NAME else 0.8)
    material_api.CreateRestitutionAttr().Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(carton_prim).Bind(
        physics_material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )


def _author_g01_3_pillow_passive_damping(stage: Any) -> None:
    """Add passive damping and a task-local grasp material override.

    The source USD remains untouched and is still validated before this
    overlay.  Only the composed G01-3 material opinion is changed, so G01-2
    keeps its current-schema 3.0/3.0 inspection material contract.
    """

    root_prim = stage.GetPrimAtPath(G01_PILLOW_BODY_PATH)
    if root_prim is None or not root_prim.IsValid():
        raise RuntimeError("G01-3 Pillow024-v2 root is unavailable for damping")
    values = {
        "physxDeformableBody:linearDamping": G01_3_PILLOW_LINEAR_DAMPING,
        "physxDeformableBody:settlingDamping": G01_3_PILLOW_SETTLING_DAMPING,
        "physxDeformableBody:settlingThreshold": (
            G01_3_PILLOW_SETTLING_THRESHOLD_MPS
        ),
    }
    for attribute_name, expected in values.items():
        attribute = root_prim.GetAttribute(attribute_name)
        if not attribute.IsValid():
            raise RuntimeError(
                f"G01-3 Pillow024-v2 damping attribute is missing: "
                f"{attribute_name}"
            )
        attribute.Set(expected)
        if not math.isclose(
            float(attribute.Get()), expected, rel_tol=0.0, abs_tol=1.0e-6
        ):
            raise RuntimeError(
                f"G01-3 Pillow024-v2 damping write failed: {attribute_name}"
            )
    material_prim = stage.GetPrimAtPath(G01_2_PILLOW_MATERIAL_PATH)
    if material_prim is None or not material_prim.IsValid():
        raise RuntimeError(
            "G01-3 Pillow024-v2 physics material is unavailable for grasp tuning"
        )
    material_values = {
        "omniphysics:staticFriction": G01_3_PILLOW_STATIC_FRICTION,
        "omniphysics:dynamicFriction": G01_3_PILLOW_DYNAMIC_FRICTION,
    }
    for attribute_name, expected in material_values.items():
        attribute = material_prim.GetAttribute(attribute_name)
        if not attribute.IsValid():
            raise RuntimeError(
                f"G01-3 Pillow024-v2 material attribute is missing: "
                f"{attribute_name}"
            )
        attribute.Set(expected)
        if not math.isclose(
            float(attribute.Get()), expected, rel_tol=0.0, abs_tol=1.0e-6
        ):
            raise RuntimeError(
                f"G01-3 Pillow024-v2 material write failed: {attribute_name}"
            )
    root_prim.SetCustomDataByKey(
        "handson:g01_3PassiveDamping",
        {
            "linearPerSecond": G01_3_PILLOW_LINEAR_DAMPING,
            "settlingPerSecond": G01_3_PILLOW_SETTLING_DAMPING,
            "settlingThresholdMps": G01_3_PILLOW_SETTLING_THRESHOLD_MPS,
            "staticFriction": G01_3_PILLOW_STATIC_FRICTION,
            "dynamicFriction": G01_3_PILLOW_DYNAMIC_FRICTION,
        },
    )


def _author_g01_5_placement_damping(stage: Any) -> None:
    """Dissipate foam rebound without changing its stiffness or writing state."""
    # Preserve the asset's calibrated elasticity damping (0.18). Raising it
    # to 0.35 increases gravity-driven nodal relaxation from 54.6 to 70.6 mm
    # at 120 Hz / 64 iterations, exceeding the 60 mm recording-origin limit.
    # Body damping and contact recovery still dissipate placement rebound.
    for path, values in (
        (G01_PILLOW_BODY_PATH, {
            "physxDeformableBody:linearDamping": 3.0,
            "physxDeformableBody:maxDepenetrationVelocity": 0.25,
        }),
    ):
        prim = stage.GetPrimAtPath(path)
        for name, value in values.items():
            attribute = prim.GetAttribute(name)
            if not attribute.IsValid():
                raise RuntimeError(f"G01-5 placement material lacks {path}.{name}")
            attribute.Set(value)


def spawn_g01_deformable_assets(
    stage: Any,
    task_name: str = G01_TASK_NAME,
) -> tuple[str, ...]:
    """Reference one G01 asset variant and author current deformables."""

    import carb
    from pxr import Gf, UsdGeom

    if not carb.settings.get_settings().get_as_bool(
        "/persistent/physics/enableDeformableBeta"
    ):
        raise RuntimeError(
            "G01 Pillow024/Towl011 use the current PhysX volume/surface "
            "deformable pipeline and require "
            "/persistent/physics/enableDeformableBeta=true"
        )
    specs = g01_asset_specs(task_name)
    validate_g01_asset_files(task_name)
    root = UsdGeom.Xform.Define(stage, G01_ASSET_ROOT_PATH)
    root.GetPrim().SetCustomDataByKey("handson:task", task_name)

    spawned: list[str] = []
    for spec in specs:
        placement = UsdGeom.Xform.Define(stage, spec.placement_path)
        placement.AddTranslateOp().Set(Gf.Vec3d(*spec.translation))
        if spec.rotation_z_degrees:
            placement.AddRotateZOp().Set(float(spec.rotation_z_degrees))
        if not math.isclose(spec.uniform_scale, 1.0):
            placement.AddScaleOp().Set(
                Gf.Vec3f(
                    spec.uniform_scale,
                    spec.uniform_scale,
                    spec.uniform_scale,
                )
            )
        placement_prim = placement.GetPrim()
        placement_prim.SetCustomDataByKey("handson:sourceAsset", spec.name)

        reference = UsdGeom.Xform.Define(stage, spec.reference_path)
        reference.GetPrim().GetReferences().AddReference(
            str(spec.source_path.resolve())
        )
        spawned.append(spec.reference_path)

    if task_name in G01_COMPARISON_PROFILES:
        _validate_g01_comparison_pillow(stage, task_name)
    elif task_name in (G01_2_TASK_NAME, G01_3_TASK_NAME):
        # Pillow024-v2 already contains the explicit current-schema TetMesh.
        # Re-running the G01 legacy migration would discard that repaired mesh
        # and fail because the four legacy attributes intentionally do not
        # exist in the revision.
        _validate_g01_2_pillow_volume_deformable(stage)
    else:
        _migrate_g01_pillow_volume_deformable(stage)
    if task_name in G01_CARTON_TASK_NAMES:
        if task_name == G01_3_TASK_NAME:
            _author_g01_3_pillow_passive_damping(stage)
        elif task_name == G01_5_TASK_NAME:
            _author_g01_5_placement_damping(stage)
        _author_g01_3_open_carton(stage, task_name)
    else:
        _author_g01_towel_surface_deformable(stage)

    from pxr import PhysxSchema, UsdPhysics

    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Scene):
            continue
        prim.ApplyAPI(PhysxSchema.PhysxSceneAPI)
        scene_api = PhysxSchema.PhysxSceneAPI(prim)
        scene_api.GetTimeStepsPerSecondAttr().Set(
            120 if task_name in G01_COMPARISON_PROFILES else G01_PHYSICS_HZ
        )
        scene_api.GetGpuMaxDeformableSurfaceContactsAttr().Set(4 * 1_048_576)
        break
    return tuple(spawned)


def _pillow_surface_sample_error(points, target_points, target_faces) -> float:
    """Conservative nearest-triangle distance for each supplied sample.

    Searching nearby triangle centroids only can overestimate the distance,
    never accept a sample without an actual triangle within the tolerance.
    Keep scratch arrays bounded for the detailed render surface.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    triangles = np.asarray(target_points)[np.asarray(target_faces)]
    tree = cKDTree(triangles.mean(axis=1))
    maximum = 0.0
    def nearest_subset(query, count):
        _, indices = tree.query(query, k=count)
        near = triangles[np.asarray(indices).reshape(len(query), -1)]
        a, b, c = near[:, :, 0], near[:, :, 1], near[:, :, 2]
        q = query[:, None, :]
        ab, ac, aq = b-a, c-a, q-a
        dot = lambda x, y: np.einsum("...i,...i->...", x, y)
        aa, cc, cross = dot(ab, ab), dot(ac, ac), dot(ab, ac)
        denom = aa*cc-cross*cross
        safe = np.maximum(denom, np.finfo(float).tiny)
        v = (cc*dot(aq, ab)-cross*dot(aq, ac))/safe
        w = (aa*dot(aq, ac)-cross*dot(aq, ab))/safe
        projected = a+v[..., None]*ab+w[..., None]*ac
        distance = np.where((denom > 0) & (v >= 0) & (w >= 0) & (v+w <= 1),
                            dot(q-projected, q-projected), np.inf)
        for x, y in ((a, b), (b, c), (c, a)):
            edge = y-x
            t = np.clip(dot(q-x, edge)/np.maximum(dot(edge, edge), np.finfo(float).tiny), 0, 1)
            delta = q-(x+t[..., None]*edge)
            distance = np.minimum(distance, dot(delta, delta))
        return np.sqrt(distance.min(axis=1))

    for start in range(0, len(points), 128):
        query = np.asarray(points[start:start + 128])
        errors = np.full(len(query), np.inf)
        pending = np.arange(len(query))
        count = min(32, len(triangles))
        while len(pending):
            errors[pending] = np.minimum(errors[pending], nearest_subset(query[pending], count))
            if count == len(triangles):
                break
            # Long, flat triangles can have distant centroids. Widen only
            # samples without a nearby surface witness, up to the full mesh.
            pending = pending[errors[pending] > 0.0015]
            count = min(count*4, len(triangles))
        maximum = max(maximum, float(errors.max()))
    return maximum


def _validate_g01_comparison_pillow(stage: Any, task_name: str) -> None:
    """Check physical topology before PhysX first cooks the new pillow."""
    import numpy as np
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(G01_2_PILLOW_SIM_MESH_PATH)
    mesh = UsdGeom.TetMesh(prim)
    if not mesh:
        raise RuntimeError("Comparison pillow has no simulation TetMesh")
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    tets = np.asarray(mesh.GetTetVertexIndicesAttr().Get(), dtype=np.int64)
    if points.shape != (G01_COMPARISON_NODE_COUNTS[task_name], 3):
        raise RuntimeError(f"Comparison pillow node inventory changed: {points.shape}")
    if tets.shape != (G01_COMPARISON_TET_COUNTS[task_name], 4) or not np.all(np.isfinite(points)):
        raise RuntimeError("Comparison pillow topology/state is invalid")
    volumes = np.linalg.det(points[tets[:, 1:]] - points[tets[:, :1]]) / 6
    if np.min(volumes) <= 1.0e-11:
        raise RuntimeError("Comparison pillow contains inverted/degenerate tets")
    volume_faces = np.concatenate([tets[:, columns] for columns in (
        (1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)
    )])
    unique_faces, incidence = np.unique(np.sort(volume_faces, axis=1), axis=0, return_counts=True)
    boundary = unique_faces[incidence == 1]
    faces = np.asarray(mesh.GetSurfaceFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if np.max(incidence) > 2 or set(map(tuple, boundary)) != set(map(tuple, np.sort(faces, axis=1))):
        raise RuntimeError("Comparison pillow collision surface is not its volume boundary")
    edges = np.sort(np.concatenate([faces[:, columns] for columns in ((0, 1), (1, 2), (2, 0))]), axis=1)
    _, edge_incidence = np.unique(edges, axis=0, return_counts=True)
    if not np.all(edge_incidence == 2):
        raise RuntimeError("Comparison pillow collision boundary is not closed/manifold")
    envelope_volume = np.einsum("ij,ij->i", points[faces[:, 0]], np.cross(
        points[faces[:, 1]], points[faces[:, 2]])).sum() / 6
    if not np.isclose(volumes.sum(), envelope_volume, rtol=1e-5):
        raise RuntimeError("Comparison pillow volume folds/overlaps inside its envelope")
    collision = UsdGeom.TetMesh(stage.GetPrimAtPath(G01_PILLOW_BODY_PATH + "/collision_mesh"))
    if not collision:
        raise RuntimeError("Comparison pillow has no detailed collision TetMesh")
    collision_points = np.asarray(collision.GetPointsAttr().Get(), dtype=np.float64)
    collision_tets = np.asarray(collision.GetTetVertexIndicesAttr().Get(), dtype=np.int64)
    collision_volumes = np.linalg.det(collision_points[collision_tets[:, 1:]] - collision_points[collision_tets[:, :1]]) / 6
    if not np.isfinite(collision_points).all() or collision_volumes.min() <= 1e-11:
        raise RuntimeError("Invalid comparison pillow collision tetrahedra")
    points, faces = collision_points, np.asarray(collision.GetSurfaceFaceVertexIndicesAttr().Get(), dtype=np.int64)
    collision_volume_faces = np.concatenate([collision_tets[:, cols] for cols in ((1,2,3),(0,3,2),(0,1,3),(0,2,1))])
    exposed, counts = np.unique(np.sort(collision_volume_faces,axis=1),axis=0,return_counts=True)
    if counts.max()>2 or set(map(tuple,exposed[counts==1]))!=set(map(tuple,np.sort(faces,axis=1))):
        raise RuntimeError("Detailed collision surface is not its tetrahedral boundary")
    oriented_volume=np.einsum("ij,ij->i",points[faces[:,0]],np.cross(points[faces[:,1]],points[faces[:,2]])).sum()/6
    if not np.isclose(oriented_volume,collision_volumes.sum(),rtol=1e-5):
        raise RuntimeError("Detailed collision tetrahedra overlap")
    if UsdGeom.Imageable(collision).ComputeVisibility() != "invisible":
        raise RuntimeError("Detailed collision mesh must be invisible")
    skin = UsdGeom.Mesh(stage.GetPrimAtPath(G01_PILLOW_SKIN_MESH_PATH))
    skin_points = np.asarray(skin.GetPointsAttr().Get())
    skin_faces = np.asarray(skin.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    # Render triangles are a 1-to-4 subdivision of each physical triangle.
    triangles = points[faces]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
    expected = np.stack([np.stack(f, axis=1) for f in (
        (a,ab,ca), (ab,b,bc), (ca,bc,c), (ab,bc,ca))], axis=1).reshape(-1,3,3)
    exact_surface = skin_points[skin_faces].shape == expected.shape and np.allclose(
        skin_points[skin_faces], expected, atol=1e-7, rtol=0
    )
    if not exact_surface:
        if task_name != G01_5_TASK_NAME:
            raise RuntimeError("Comparison pillow visual/collision surfaces disagree")
        # G01-5 uses a lighter collision surface with the detailed original
        # render mesh. Check actual composed geometry in both directions;
        # never trust a manifest's error estimate in place of the mesh.
        visual_samples = np.concatenate((skin_points, skin_points[skin_faces].mean(axis=1)))
        collision_samples = np.concatenate((points[np.unique(faces)], points[faces].mean(axis=1)))
        error = max(
            _pillow_surface_sample_error(visual_samples, points, faces),
            _pillow_surface_sample_error(collision_samples, skin_points, skin_faces),
        )
        if not np.isfinite(error) or error > 0.0015:
            raise RuntimeError(f"Pillow visual/collision surface error exceeds 1.5 mm: {error}")
    if UsdGeom.Imageable(prim).ComputeVisibility() != "invisible":
        raise RuntimeError("Comparison pillow collision mesh must be invisible")
    rest = np.asarray(prim.GetAttribute("omniphysics:restShapePoints").Get())
    if not np.array_equal(np.asarray(mesh.GetPointsAttr().Get()), rest):
        raise RuntimeError("Comparison pillow was not authored at rest")


__all__ = [
    "G01_2_ASSET_SPECS",
    "G01_2_PILLOW_COLLISION_MESH_PATH",
    "G01_2_PILLOW_MATERIAL_PATH",
    "G01_2_PILLOW_SIM_MESH_PATH",
    "G01_2_TASK_NAME",
    "G01_3_ASSET_SPECS",
    "G01_3_CARTON_COLLIDER_PATHS",
    "G01_3_CARTON_INNER_LOCAL_MAX",
    "G01_3_CARTON_INNER_LOCAL_MIN",
    "G01_3_CARTON_LOCAL_MAX",
    "G01_3_CARTON_LOCAL_MIN",
    "G01_3_CARTON_OPEN_FACE_COUNT",
    "G01_3_CARTON_PATH",
    "G01_3_CARTON_RENDER_SCALE",
    "G01_3_CARTON_SOURCE_USD",
    "G01_3_CARTON_SOURCE_INNER_LOCAL_MAX",
    "G01_3_CARTON_SOURCE_INNER_LOCAL_MIN",
    "G01_3_CARTON_SOURCE_BODY_PATH",
    "G01_3_CARTON_SOURCE_LOCAL_MAX",
    "G01_3_CARTON_SOURCE_LOCAL_MIN",
    "G01_3_CARTON_SOURCE_MESH_PATH",
    "G01_3_CARTON_TOP_FACE_COUNT",
    "G01_3_CARTON_TRANSLATION",
    "G01_3_PILLOW_UNIFORM_SCALE",
    "G01_5_PILLOW_HALF_HEIGHT_M",
    "G01_3_PILLOW_LINEAR_DAMPING",
    "G01_3_PILLOW_STATIC_FRICTION",
    "G01_3_PILLOW_DYNAMIC_FRICTION",
    "G01_3_PILLOW_SETTLING_DAMPING",
    "G01_3_PILLOW_SETTLING_THRESHOLD_MPS",
    "G01_3_SCENE_XML",
    "G01_3_SUCCESS_CONDITION_LABELS",
    "G01_3_TASK_NAME",
    "G01_ASSET_DIRECTORY",
    "G01_ASSET_ROOT_PATH",
    "G01_ASSET_SPECS",
    "G01_CABLE_RIGID_BODY_PATHS",
    "G01_DEFORMABLE_VIEW_SPECS",
    "G01_PHYSICS_HZ",
    "G01_PILLOW_BODY_PATH",
    "G01_PILLOW_COLLISION_MESH_PATH",
    "G01_PILLOW_SIM_MESH_PATH",
    "G01_SCENE_XML",
    "G01_SURFACE_DEFORMABLE_PRIM_PATHS",
    "G01_SUCCESS_CONDITION_LABELS",
    "G01_TABLE_TOP_Z",
    "G01_TASK_NAME",
    "G01_TASK_NAMES",
    "G01_TOWEL_BEND_STIFFNESS",
    "G01_TOWEL_COLUMNS",
    "G01_TOWEL_CONTACT_OFFSET",
    "G01_TOWEL_ELASTICITY_DAMPING",
    "G01_TOWEL_MAX_DEPENETRATION_VELOCITY",
    "G01_TOWEL_NODE_COUNT",
    "G01_TOWEL_ROWS",
    "G01_TOWEL_SHEAR_STIFFNESS",
    "G01_TOWEL_STRETCH_STIFFNESS",
    "G01_TOWEL_SURFACE_MESH_PATH",
    "G01_TOWEL_SURFACE_ROOT_PATH",
    "G01_TOWEL_SURFACE_THICKNESS",
    "G01_TOWEL_TRIANGLE_COUNT",
    "G01_VOLUME_DEFORMABLE_PRIM_PATHS",
    "g01_asset_specs",
    "g01_deformable_view_specs",
    "g01_towel_surface_topology",
    "spawn_g01_deformable_assets",
    "validate_g01_asset_files",
]
