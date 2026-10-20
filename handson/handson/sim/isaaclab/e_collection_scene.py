"""GarmentLab asset metadata and native PhysX setup for E01.

The module stays importable without Isaac Sim so task catalogs and interface
tests can inspect E01.  Isaac/PhysX imports are intentionally local to the
scene-authoring function.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from handson.runtime.project_paths import resolve_project_path
from typing import Any, Mapping


E01_TASK_NAME = "e01_fold_garment"
E_TASK_SCENES = {
    E01_TASK_NAME: "arena_arm_hand_e01_fold_garment_rokae_wuji1.xml",
}
E_TASK_NAMES = frozenset(E_TASK_SCENES)

E_SUCCESS_CONDITION_LABELS = {
    E01_TASK_NAME: (
        "Garment folded into a compact two-layer footprint",
        "Entire garment remains on the tabletop",
        "Both hands released from the garment",
        "Fold remains stable for 15 frames",
    ),
}

GARMENTLAB_ASSET_RELATIVE_PATH = Path(
    "Garment/Tops/NoCollar_Lsleeve_FrontClose/TNLC_shirt2/"
    "TNLC_shirt2_obj.usd"
)
E01_GARMENT_DIFFUSE_TEXTURE_NAME = "texture.png"
E01_GARMENT_NORMAL_TEXTURE_NAME = None
E01_GARMENT_ROOT_PATH = "/World/E01Garment"
E01_GARMENT_SKIN_MESH_PATH = f"{E01_GARMENT_ROOT_PATH}/mesh"
# TNLC_shirt2 is an official GarmentLab asset whose native topology is already
# small enough for real-time PhysX.  Use that exact mesh for rendering,
# collision, and simulation; there is no proxy mesh and therefore no skinning
# map that can separate or explode under contact.
E01_GARMENT_SIM_MESH_PATH = E01_GARMENT_SKIN_MESH_PATH

_TOP578_GARMENTLAB_ASSET_RELATIVE_PATH = Path(
    "Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top578/"
    "TCLC_Top578_obj.usd"
)
_TOP578_GARMENTLAB_DIFFUSE_TEXTURE_NAME = "Top578_diffuse_1001.png"
_TOP578_GARMENTLAB_NORMAL_TEXTURE_NAME = "Top578_normal_1001.png"

_E01_GARMENTLAB_PRESET_ID_ENV = "HANDSON_E01_GARMENTLAB_PRESET"
_E01_GARMENTLAB_PRESET_METADATA = {
    "1": {
        "relative_path": GARMENTLAB_ASSET_RELATIVE_PATH,
        "label": "GarmentLab/TNLC_shirt2",
        "visual_vertex_count": 1_468,
        "visual_triangle_count": 2_891,
        "sim_vertex_count": 1_468,
        "sim_triangle_count": 2_891,
        "scale": 0.006,
        "orientation": (0.70710678, 0.0, 0.0, 0.70710678),
        "translation": (0.4988, 0.04835, 0.949619),
        "diffuse_texture": E01_GARMENT_DIFFUSE_TEXTURE_NAME,
        "normal_texture": E01_GARMENT_NORMAL_TEXTURE_NAME,
    },
    "2": {
        "relative_path": _TOP578_GARMENTLAB_ASSET_RELATIVE_PATH,
        "label": "GarmentLab/TCLC_Top578",
        "visual_vertex_count": 4_772,
        "visual_triangle_count": 9_398,
        "sim_vertex_count": 4_772,
        "sim_triangle_count": 9_398,
        "scale": 0.0087,
        "orientation": (0.70710678, 0.0, 0.0, 0.70710678),
        "translation": (0.918486, 0.000608, 1.005735),
        "diffuse_texture": _TOP578_GARMENTLAB_DIFFUSE_TEXTURE_NAME,
        "normal_texture": _TOP578_GARMENTLAB_NORMAL_TEXTURE_NAME,
    },
}
_E01_GARMENTLAB_PRESET_ALIASES = {
    "1": "1",
    "t1": "1",
    "tnlc": "1",
    "tnlc_shirt2": "1",
    "2": "2",
    "t2": "2",
    "tclc": "2",
    "top578": "2",
}
E01_REST_SHAPE_MODE_ENV = "HANDSON_E01_REST_SHAPE_MODE"
E01_REST_SHAPE_MODE_DEFAULT = "tabletop_wrinkled"
_E01_REST_SHAPE_MODE_ALIASES = {
    "asset": "asset",
    "native": "asset",
    "original": "asset",
    "flat": "tabletop_flat",
    "table_flat": "tabletop_flat",
    "tabletop_flat": "tabletop_flat",
    "table": "tabletop_wrinkled",
    "tabletop": "tabletop_wrinkled",
    "wrinkled": "tabletop_wrinkled",
    "table_wrinkled": "tabletop_wrinkled",
    "tabletop_wrinkled": "tabletop_wrinkled",
}
E01_GARMENT_MATERIAL_PATH = "/World/PhysicsMaterials/E01GarmentLabCloth"
E01_GARMENT_VISUAL_MATERIAL_PATH = "/World/Looks/E01GarmentLabOpaque"
# White keeps the authored blue cotton albedo unchanged when OmniPBR
# multiplies its constant color with the diffuse texture.
E01_GARMENT_VISUAL_COLOR = (1.0, 1.0, 1.0)
E01_VISUAL_MESH_VERTEX_COUNT = 1_468
E01_VISUAL_MESH_TRIANGLE_COUNT = 2_891
E01_SIM_MESH_VERTEX_COUNT = E01_VISUAL_MESH_VERTEX_COUNT
E01_SIM_MESH_TARGET_TRIANGLE_COUNT = E01_VISUAL_MESH_TRIANGLE_COUNT
# The runtime tensor contract is exact because the downloaded native mesh is
# authored explicitly as the rest/simulation topology.  No auto-cooking or
# runtime simplification is allowed on launch.
E01_SIM_MESH_MIN_TRIANGLE_COUNT = E01_SIM_MESH_TARGET_TRIANGLE_COUNT
E01_SIM_MESH_MAX_TRIANGLE_COUNT = E01_SIM_MESH_TARGET_TRIANGLE_COUNT
E01_SIM_MESH_MIN_NODE_COUNT = E01_SIM_MESH_VERTEX_COUNT
E01_SIM_MESH_MAX_NODE_COUNT = E01_SIM_MESH_VERTEX_COUNT

# TNLC_shirt2 is the official no-collar long-sleeve shirt distributed in the
# GarmentLab/DexGarmentLab archive.  Its USD is Y-up and authored in source
# garment units.  This scale gives a roughly 69 x 47 cm shirt on the table.
E01_GARMENT_SCALE = 0.006
E01_GARMENT_ORIENTATION = (0.70710678, 0.0, 0.0, 0.70710678)
# These are PhysX Surface Deformable coefficients, not the legacy particle-
# cloth spring coefficients used by GarmentLab.  In particular, carrying the
# particle bend value (100) across made the shirt behave like a thin plastic
# shell and stored far too much contact energy.  Keep the in-plane weave firm
# while allowing a cotton-like low-energy fold.
E01_STRETCH_STIFFNESS = 5.0e3
E01_SHEAR_STIFFNESS = 50.0
E01_BEND_STIFFNESS = 0.3
E01_ELASTICITY_DAMPING = 0.55
E01_DYNAMIC_FRICTION = 0.65
# The roughly 0.446 m^2 shell weighs about 67 g at these explicit values.
# A 0.3 mm shell also fits the downloaded mesh's closest non-neighbour gaps;
# the previous 1.5 mm shell began in self-penetration before either hand moved.
# Leaving density at the schema sentinel (-1) makes both appearance and
# contact response runtime-dependent.
E01_SURFACE_DENSITY = 500.0
E01_REST_OFFSET = 0.0005
E01_CONTACT_OFFSET = 0.002
E01_SURFACE_THICKNESS = 0.0003
E01_PHYSICS_HZ = 60
E01_SOLVER_POSITION_ITERATIONS = 16
# Keep recovery from a bad hand/cloth overlap bounded.  PhysX otherwise leaves
# this at infinity for surface deformables, so one late contact can turn a few
# millimetres of penetration into an unbounded separation impulse.
E01_MAX_DEPENETRATION_VELOCITY = 0.18
E01_TABLE_TOP_Z = 0.89
E01_TABLETOP_REST_CLEARANCE = 0.004
E01_TABLETOP_REST_WRINKLE_AMPLITUDE = 0.004


@dataclass(frozen=True)
class E01SurfacePhysicsTuning:
    contact_offset: float = E01_CONTACT_OFFSET
    rest_offset: float = E01_REST_OFFSET
    surface_thickness: float = E01_SURFACE_THICKNESS
    surface_density: float = E01_SURFACE_DENSITY
    dynamic_friction: float = E01_DYNAMIC_FRICTION
    surface_stretch_stiffness: float = E01_STRETCH_STIFFNESS
    surface_shear_stiffness: float = E01_SHEAR_STIFFNESS
    surface_bend_stiffness: float = E01_BEND_STIFFNESS
    elasticity_damping: float = E01_ELASTICITY_DAMPING
    max_linear_velocity: float | None = None
    max_depenetration_velocity: float = E01_MAX_DEPENETRATION_VELOCITY
    solver_position_iterations: int = E01_SOLVER_POSITION_ITERATIONS
    collision_pair_update_frequency: int = 4
    collision_iteration_multiplier: int = 4

    @property
    def resolved_max_linear_velocity(self) -> float:
        if self.max_linear_velocity is not None:
            return float(self.max_linear_velocity)
        return float(self.contact_offset) * float(E01_PHYSICS_HZ)

# Native bounds are x=[-65.6456,49.5291], y=[83.7086,162.5450], and
# z=[-9.1031,24.0153].  After the rotation and scale below, the lowest node is
# 5 mm above the 0.89 m tabletop and the nominal hands begin outside the cloth
# contact envelope, avoiding a reset-time depenetration kick.
E01_GARMENT_TRANSLATION = (0.4988, 0.04835, 0.949619)


def selected_garmentlab_preset_id() -> str:
    """Return the canonical GarmentLab preset id selected in the current process."""

    raw = os.environ.get(_E01_GARMENTLAB_PRESET_ID_ENV, "1").strip().lower()
    preset_id = _E01_GARMENTLAB_PRESET_ALIASES.get(raw, raw)
    if preset_id not in _E01_GARMENTLAB_PRESET_METADATA:
        raise ValueError(
            f"Unsupported GarmentLab E01 preset {raw!r}; "
            f"supported={sorted(_E01_GARMENTLAB_PRESET_METADATA)}"
        )
    return preset_id


def selected_garmentlab_preset_metadata() -> dict[str, Any]:
    """Return the canonical metadata dictionary for the active preset."""

    return dict(_E01_GARMENTLAB_PRESET_METADATA[selected_garmentlab_preset_id()])


def _normalise_e01_rest_shape_mode(raw_mode: str) -> str:
    raw = str(raw_mode).strip().lower().replace("-", "_")
    mode = _E01_REST_SHAPE_MODE_ALIASES.get(raw, raw)
    if mode not in {"asset", "tabletop_flat", "tabletop_wrinkled"}:
        raise ValueError(
            f"Unsupported E01 rest-shape mode {raw_mode!r}; "
            "supported=['asset', 'tabletop_flat', 'tabletop_wrinkled']"
        )
    return mode


def selected_e01_rest_shape_mode() -> str:
    """Return how E01 should author the garment's PhysX rest shape."""

    return _normalise_e01_rest_shape_mode(
        os.environ.get(E01_REST_SHAPE_MODE_ENV, E01_REST_SHAPE_MODE_DEFAULT)
    )


def _e01_quat_wxyz_to_matrix(orientation: Any) -> Any:
    import numpy as np

    quat = np.asarray(tuple(float(value) for value in orientation), dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"E01 orientation must be wxyz quaternion; got {quat!r}")
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError("E01 orientation quaternion must be non-zero")
    w, qx, qy, qz = quat / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * w),
                2.0 * (qx * qz + qy * w),
            ],
            [
                2.0 * (qx * qy + qz * w),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * w),
            ],
            [
                2.0 * (qx * qz - qy * w),
                2.0 * (qy * qz + qx * w),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def _e01_points_array(points: Any) -> Any:
    import numpy as np

    array = np.asarray(
        [[float(point[0]), float(point[1]), float(point[2])] for point in points],
        dtype=np.float64,
    )
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError(f"E01 mesh points must have shape (N, 3); got {array.shape}")
    return array


def _e01_points_to_world(points: Any, metadata: Mapping[str, Any]) -> Any:
    import numpy as np

    local_points = _e01_points_array(points)
    rotation = _e01_quat_wxyz_to_matrix(metadata["orientation"])
    translation = np.asarray(metadata["translation"], dtype=np.float64)
    scale = float(metadata["scale"])
    return local_points @ rotation.T * scale + translation


def _e01_points_from_world(points: Any, metadata: Mapping[str, Any]) -> Any:
    import numpy as np

    world_points = _e01_points_array(points)
    rotation = _e01_quat_wxyz_to_matrix(metadata["orientation"])
    translation = np.asarray(metadata["translation"], dtype=np.float64)
    scale = float(metadata["scale"])
    if scale <= 0.0:
        raise ValueError(f"E01 garment scale must be positive; got {scale!r}")
    return ((world_points - translation) / scale) @ rotation


def _e01_tabletop_rest_shape_points(
    points: Any,
    metadata: Mapping[str, Any],
    *,
    wrinkle_amplitude: float = E01_TABLETOP_REST_WRINKLE_AMPLITUDE,
) -> Any:
    """Return local mesh points whose world pose is a quiet tabletop garment."""

    import numpy as np

    world_points = _e01_points_to_world(points, metadata)
    xy = world_points[:, :2]
    xy_min = np.min(xy, axis=0)
    xy_span = np.ptp(xy, axis=0)
    safe_span = np.where(xy_span > 1e-9, xy_span, 1.0)
    normalized_xy = (xy - xy_min) / safe_span
    x = normalized_xy[:, 0]
    y = normalized_xy[:, 1]
    wrinkle = (
        0.45 * np.sin(2.5 * np.pi * x + 0.35) * np.cos(1.7 * np.pi * y - 0.2)
        + 0.35 * np.sin(4.0 * np.pi * y + 0.8)
        + 0.20 * np.cos(3.0 * np.pi * (x - y))
    )
    wrinkle -= float(np.min(wrinkle))
    wrinkle_span = float(np.ptp(wrinkle))
    if wrinkle_span > 1e-9:
        wrinkle /= wrinkle_span
    else:
        wrinkle.fill(0.0)

    tabletop_world = world_points.copy()
    tabletop_world[:, 2] = (
        E01_TABLE_TOP_Z
        + E01_TABLETOP_REST_CLEARANCE
        + max(0.0, float(wrinkle_amplitude)) * wrinkle
    )
    return _e01_points_from_world(tabletop_world, metadata)


def _e01_rest_shape_points(
    points: Any,
    metadata: Mapping[str, Any],
    *,
    mode: str | None = None,
) -> Any:
    selected_mode = (
        selected_e01_rest_shape_mode()
        if mode is None
        else _normalise_e01_rest_shape_mode(mode)
    )
    if selected_mode == "asset":
        return _e01_points_array(points).copy()
    wrinkle_amplitude = (
        0.0
        if selected_mode == "tabletop_flat"
        else E01_TABLETOP_REST_WRINKLE_AMPLITUDE
    )
    return _e01_tabletop_rest_shape_points(
        points,
        metadata,
        wrinkle_amplitude=wrinkle_amplitude,
    )


def _resolve_garment_surface_mesh_prim_path(
    stage: Any,
    root_prim: Any,
) -> Path:
    """Return the authored mesh prim path with the largest point cloud."""

    from pxr import Usd, UsdGeom

    selected: tuple[Path | None, int] = (None, -1)
    for candidate in Usd.PrimRange(root_prim):
        if not candidate.IsA(UsdGeom.Mesh):
            continue
        points = UsdGeom.Mesh(candidate).GetPointsAttr().Get()
        if points is None:
            continue
        count = len(points)
        current_path = Path(str(candidate.GetPath()))
        if count > selected[1]:
            selected = (current_path, count)
    if selected[0] is None:
        raise RuntimeError(
            "GarmentLab asset did not expose a mesh under the referenced root"
        )
    return selected[0]


def _repair_garmentlab_rendering(
    stage: Any,
    asset_path: Path,
    *,
    skin_mesh_path: str,
    diffuse_texture_name: str,
) -> tuple[int, int]:
    """Make the referenced GarmentLab cloth render as an opaque two-sided mesh.

    TNLC_shirt2's skin mesh is single-sided and its legacy preview material
    authors zero opacity.  With current RTX/Fabric rendering, the cloth can
    therefore disappear while PhysX still casts its shadow.  Authored normals
    also become stale as the surface deforms, so they are blocked and
    regenerated by the renderer.
    """
    import numpy as np
    from isaacsim.core.api.materials.omni_pbr import OmniPBR
    from pxr import Usd, UsdGeom, UsdShade

    skin_prim = stage.GetPrimAtPath(skin_mesh_path)
    if not skin_prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(
            "GarmentLab skin mesh is not a USD mesh: "
            f"{skin_mesh_path}"
        )

    skin_mesh = UsdGeom.Mesh(skin_prim)
    skin_mesh.CreateDoubleSidedAttr().Set(True)

    normals_attr = skin_mesh.GetNormalsAttr()
    if normals_attr.IsValid():
        normals_attr.Block()

    repaired_materials = 0
    root_prim = stage.GetPrimAtPath(E01_GARMENT_ROOT_PATH)
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() != "UsdPreviewSurface":
            continue

        opacity_attr = shader.GetInput("opacity").GetAttr()
        if opacity_attr.IsValid():
            opacity_attr.ClearConnections()
            opacity_attr.Set(1.0)

        threshold_attr = shader.GetInput("opacityThreshold").GetAttr()
        if threshold_attr.IsValid():
            threshold_attr.ClearConnections()
            threshold_attr.Set(0.0)
        repaired_materials += 1

    if repaired_materials == 0:
        raise RuntimeError(
            "GarmentLab skin mesh did not provide a UsdPreviewSurface "
            "material"
        )

    diffuse_path = _resolve_garmentlab_texture_path(
        asset_path,
        diffuse_texture_name,
    )

    # Isaac Sim 5.1 renders the asset's legacy PreviewSurface nearly black.
    # Rebind a current opaque OmniPBR shader while keeping the downloaded
    # garment's authored UVs and official fabric texture.
    visual_material = OmniPBR(
        prim_path=E01_GARMENT_VISUAL_MATERIAL_PATH,
        texture_path=str(diffuse_path),
        color=np.asarray(E01_GARMENT_VISUAL_COLOR, dtype=np.float32),
    )
    visual_material.set_project_uvw(False)
    visual_material.set_reflection_roughness(0.92)
    visual_material.set_metallic_constant(0.0)

    rebound_prims = 0
    for prim in Usd.PrimRange(skin_prim):
        if prim == skin_prim or prim.IsA(UsdGeom.Subset):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                visual_material.material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
            rebound_prims += 1
    return repaired_materials, rebound_prims


def _resolve_garmentlab_texture_path(asset_path: Path, texture_name: str) -> Path:
    asset_dir = asset_path.parent
    raw_texture_path = Path(texture_name)
    if raw_texture_path.is_absolute() and raw_texture_path.is_file():
        return raw_texture_path

    candidates = (
        asset_dir / raw_texture_path,
        asset_dir / "textures" / raw_texture_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "GarmentLab asset is incomplete; missing its official fabric "
        f"texture {texture_name!r}. Searched: {searched}"
    )


def garmentlab_asset_root_candidates() -> tuple[Path, ...]:
    """Return explicit overrides and the portable in-project GarmentLab root."""
    candidates: list[Path] = []
    for env_name in ("HANDSON_GARMENTLAB_ASSET_ROOT", "GARMENTLAB_ASSET_ROOT"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(resolve_project_path(value))

    repository_root = Path(__file__).resolve().parents[4]
    candidates.extend(
        (
            repository_root / "third_party" / "GarmentLab" / "Assets",
        )
    )
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def resolve_garmentlab_asset_path(
    *,
    preset: str | None = None,
    required: bool = True,
) -> Path | None:
    """Resolve the selected GarmentLab USD portably."""

    preset_id = (
        _E01_GARMENTLAB_PRESET_ALIASES.get(
            str(preset).strip().lower(),
            str(preset).strip().lower(),
        )
        if preset is not None
        else selected_garmentlab_preset_id()
    )
    requested = (
        str(preset).strip().lower()
        if preset is not None
        else os.environ.get(_E01_GARMENTLAB_PRESET_ID_ENV, "1").strip().lower()
    )
    if preset_id is None or preset_id not in _E01_GARMENTLAB_PRESET_METADATA:
        raise ValueError(
            f"Unsupported GarmentLab E01 preset {requested!r}; "
            f"supported={sorted(_E01_GARMENTLAB_PRESET_METADATA)}"
        )
    relative_path = _E01_GARMENTLAB_PRESET_METADATA[preset_id]["relative_path"]

    for root in garmentlab_asset_root_candidates():
        asset_path = root / relative_path
        if asset_path.is_file():
            return asset_path
    if not required:
        return None
    searched = "\n  - ".join(
        str(root / relative_path)
        for root in garmentlab_asset_root_candidates()
    )
    label = _E01_GARMENTLAB_PRESET_METADATA[preset_id]["label"]
    raise FileNotFoundError(
        f"E01 requires the GarmentLab {label} garment USD. Set "
        "HANDSON_GARMENTLAB_ASSET_ROOT to GarmentLab/Assets. Searched:\n  - "
        f"{searched}"
    )


def _author_native_surface_hierarchy(
    stage: Any,
    root_prim: Any,
    *,
    skin_mesh_path: str,
    metadata: Mapping[str, Any],
    sim_vertex_count: int,
    sim_triangle_count: int,
) -> None:
    """Apply simulation topology from the authored native cloth mesh."""

    from pxr import Gf, UsdGeom, UsdPhysics

    sim_prim = stage.GetPrimAtPath(skin_mesh_path)
    if not sim_prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(
            "E01 native simulation prim is not a mesh: "
            f"{skin_mesh_path}"
        )
    sim_mesh = UsdGeom.Mesh(sim_prim)
    points = sim_mesh.GetPointsAttr().Get()
    face_counts = sim_mesh.GetFaceVertexCountsAttr().Get()
    face_indices = sim_mesh.GetFaceVertexIndicesAttr().Get()
    if points is None or face_counts is None or face_indices is None:
        raise RuntimeError("E01 native garment mesh has incomplete topology")

    node_count = len(points)
    triangle_count = len(face_counts)
    if node_count != sim_vertex_count:
        raise RuntimeError(
            "E01 native garment vertex count changed: "
            f"vertices={node_count}, expected={sim_vertex_count}"
        )
    if triangle_count != sim_triangle_count:
        raise RuntimeError(
            "E01 native garment triangle count changed: "
            f"triangles={triangle_count}, expected={sim_triangle_count}"
        )
    if any(int(count) != 3 for count in face_counts):
        raise RuntimeError("E01 native garment mesh must be triangulated")
    if len(face_indices) != triangle_count * 3:
        raise RuntimeError("E01 native garment mesh has invalid face indices")

    rest_shape_mode = selected_e01_rest_shape_mode()
    rest_points_np = _e01_rest_shape_points(
        points,
        metadata,
        mode=rest_shape_mode,
    )
    rest_points = [
        Gf.Vec3f(float(point[0]), float(point[1]), float(point[2]))
        for point in rest_points_np
    ]

    triangles: list[Gf.Vec3i] = []
    for offset in range(0, len(face_indices), 3):
        indices = tuple(int(index) for index in face_indices[offset : offset + 3])
        if any(index < 0 or index >= node_count for index in indices):
            raise RuntimeError("E01 native garment mesh contains an invalid index")
        if len(set(indices)) != 3:
            raise RuntimeError("E01 native garment mesh contains a degenerate face")
        triangles.append(Gf.Vec3i(*indices))

    sim_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    sim_mesh.GetPointsAttr().Set(rest_points)
    if not sim_prim.ApplyAPI("OmniPhysicsSurfaceDeformableSimAPI"):
        raise RuntimeError("Could not apply E01 surface simulation API")
    if not sim_prim.ApplyAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("Could not apply E01 surface collision API")
    sim_prim.GetAttribute("omniphysics:restShapePoints").Set(rest_points)
    sim_prim.GetAttribute("omniphysics:restTriVtxIndices").Set(triangles)
    print(
        "[HandsOn/IsaacLab] E01 PhysX rest shape authored: "
        f"mode={rest_shape_mode}, table_top_z={E01_TABLE_TOP_Z:.3f}, "
        f"clearance={E01_TABLETOP_REST_CLEARANCE:.3f}, "
        f"wrinkle_amplitude={E01_TABLETOP_REST_WRINKLE_AMPLITUDE:.3f}",
        flush=True,
    )

    if not root_prim.ApplyAPI("OmniPhysicsDeformableBodyAPI"):
        raise RuntimeError("Could not apply E01 deformable-body API")


def spawn_e01_garmentlab_surface_deformable(
    stage: Any,
    *,
    physics_tuning: E01SurfacePhysicsTuning | None = None,
) -> Path:
    """Reference the selected GarmentLab asset and configure its native surface mesh."""
    from omni.physx.scripts import deformableUtils, physicsUtils
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    tuning = physics_tuning or E01SurfacePhysicsTuning()
    metadata = selected_garmentlab_preset_metadata()
    asset_path = resolve_garmentlab_asset_path(required=True)
    assert asset_path is not None

    root = UsdGeom.Xform.Define(stage, E01_GARMENT_ROOT_PATH)
    root_prim = root.GetPrim()
    root_prim.GetReferences().AddReference(str(asset_path))
    skin_mesh_path = _resolve_garment_surface_mesh_prim_path(
        stage,
        root_prim,
    )
    print(
        "[HandsOn/IsaacLab] E01 selected garment mesh path: "
        f"{skin_mesh_path}"
    )

    repaired_materials, rebound_prims = _repair_garmentlab_rendering(
        stage,
        asset_path,
        skin_mesh_path=str(skin_mesh_path),
        diffuse_texture_name=metadata["diffuse_texture"],
    )
    print(
        "[HandsOn/IsaacLab] E01 GarmentLab rendering repaired: "
        f"double_sided=True, opaque_materials={repaired_materials}, "
        f"native_material_bindings={rebound_prims}, dynamic_normals=True, "
        "official_fabric_texture=True"
    )

    physicsUtils.setup_transform_as_scale_orient_translate(root)
    physicsUtils.set_or_add_translate_op(
        root,
        Gf.Vec3f(*metadata["translation"]),
    )
    physicsUtils.set_or_add_orient_op(
        root,
        Gf.Quatf(
            metadata["orientation"][0],
            Gf.Vec3f(*metadata["orientation"][1:]),
        ),
    )
    physicsUtils.set_or_add_scale_op(root, Gf.Vec3f(metadata["scale"]))

    _author_native_surface_hierarchy(
        stage,
        root_prim,
        skin_mesh_path=str(skin_mesh_path),
        metadata=metadata,
        sim_vertex_count=metadata["sim_vertex_count"],
        sim_triangle_count=metadata["sim_triangle_count"],
    )
    print(
        "[HandsOn/IsaacLab] E01 native single-mesh PhysX topology authored: "
        f"{metadata['sim_vertex_count']}v/"
        f"{metadata['sim_triangle_count']}t, "
        "render=simulation=collision, skinning=False, runtime_cooking=False",
        flush=True,
    )

    root_prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    body_attributes = {
        "physxDeformableBody:selfCollision": True,
        "physxDeformableBody:enableSpeculativeCCD": True,
        "physxDeformableBody:solverPositionIterationCount": (
            int(tuning.solver_position_iterations)
        ),
        "physxDeformableBody:collisionPairUpdateFrequency": int(
            tuning.collision_pair_update_frequency
        ),
        "physxDeformableBody:collisionIterationMultiplier": int(
            tuning.collision_iteration_multiplier
        ),
        "physxDeformableBody:maxLinearVelocity": (
            tuning.resolved_max_linear_velocity
        ),
        "physxDeformableBody:maxDepenetrationVelocity": (
            float(tuning.max_depenetration_velocity)
        ),
    }
    for name, value in body_attributes.items():
        attribute = root_prim.GetAttribute(name)
        if not attribute.IsValid():
            raise RuntimeError(f"Missing surface-deformable attribute {name}")
        attribute.Set(value)

    sim_mesh_prim = stage.GetPrimAtPath(str(skin_mesh_path))
    sim_mesh_prim.ApplyAPI(PhysxSchema.PhysxCollisionAPI)
    collision_api = PhysxSchema.PhysxCollisionAPI(sim_mesh_prim)
    collision_api.GetRestOffsetAttr().Set(float(tuning.rest_offset))
    collision_api.GetContactOffsetAttr().Set(float(tuning.contact_offset))

    deformableUtils.add_surface_deformable_material(
        stage,
        E01_GARMENT_MATERIAL_PATH,
        density=float(tuning.surface_density),
        dynamic_friction=float(tuning.dynamic_friction),
        surface_thickness=float(tuning.surface_thickness),
        surface_stretch_stiffness=float(tuning.surface_stretch_stiffness),
        surface_shear_stiffness=float(tuning.surface_shear_stiffness),
        surface_bend_stiffness=float(tuning.surface_bend_stiffness),
    )
    physicsUtils.add_physics_material_to_prim(
        stage,
        root_prim,
        E01_GARMENT_MATERIAL_PATH,
    )
    material_prim = stage.GetPrimAtPath(E01_GARMENT_MATERIAL_PATH)
    material_prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
    material_prim.GetAttribute(
        "physxDeformableMaterial:elasticityDamping"
    ).Set(float(tuning.elasticity_damping))
    material_prim.GetAttribute("physxDeformableMaterial:bendDamping").Set(
        float(tuning.elasticity_damping)
    )
    print(
        "[HandsOn/IsaacLab] E01 surface physics tuning: "
        f"contact_offset={float(tuning.contact_offset):.4f}, "
        f"rest_offset={float(tuning.rest_offset):.4f}, "
        f"thickness={float(tuning.surface_thickness):.4f}, "
        f"density={float(tuning.surface_density):.1f}, "
        f"friction={float(tuning.dynamic_friction):.2f}, "
        f"stretch={float(tuning.surface_stretch_stiffness):.1f}, "
        f"shear={float(tuning.surface_shear_stiffness):.1f}, "
        f"bend={float(tuning.surface_bend_stiffness):.2f}, "
        f"damping={float(tuning.elasticity_damping):.2f}, "
        f"max_linear_velocity={float(tuning.resolved_max_linear_velocity):.2f}, "
        f"max_depenetration_velocity={float(tuning.max_depenetration_velocity):.2f}",
        flush=True,
    )

    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Scene):
            continue
        prim.ApplyAPI(PhysxSchema.PhysxSceneAPI)
        scene_api = PhysxSchema.PhysxSceneAPI(prim)
        scene_api.GetTimeStepsPerSecondAttr().Set(E01_PHYSICS_HZ)
        scene_api.GetGpuMaxDeformableSurfaceContactsAttr().Set(4 * 1_048_576)
        break

    return asset_path


__all__ = [
    "E01_BEND_STIFFNESS",
    "E01_CONTACT_OFFSET",
    "E01SurfacePhysicsTuning",
    "E01_GARMENT_ROOT_PATH",
    "E01_GARMENT_DIFFUSE_TEXTURE_NAME",
    "E01_GARMENT_NORMAL_TEXTURE_NAME",
    "E01_GARMENT_ORIENTATION",
    "E01_GARMENT_SCALE",
    "E01_GARMENT_SIM_MESH_PATH",
    "E01_GARMENT_SKIN_MESH_PATH",
    "E01_GARMENT_TRANSLATION",
    "E01_MAX_DEPENETRATION_VELOCITY",
    "E01_PHYSICS_HZ",
    "E01_REST_OFFSET",
    "E01_REST_SHAPE_MODE_DEFAULT",
    "E01_REST_SHAPE_MODE_ENV",
    "E01_SHEAR_STIFFNESS",
    "E01_SIM_MESH_MAX_NODE_COUNT",
    "E01_SIM_MESH_MAX_TRIANGLE_COUNT",
    "E01_SIM_MESH_MIN_NODE_COUNT",
    "E01_SIM_MESH_MIN_TRIANGLE_COUNT",
    "E01_SIM_MESH_TARGET_TRIANGLE_COUNT",
    "E01_SIM_MESH_VERTEX_COUNT",
    "E01_STRETCH_STIFFNESS",
    "E01_SURFACE_DENSITY",
    "E01_SURFACE_THICKNESS",
    "E01_TABLE_TOP_Z",
    "E01_TABLETOP_REST_CLEARANCE",
    "E01_TABLETOP_REST_WRINKLE_AMPLITUDE",
    "E01_TASK_NAME",
    "E01_VISUAL_MESH_TRIANGLE_COUNT",
    "E01_VISUAL_MESH_VERTEX_COUNT",
    "E_SUCCESS_CONDITION_LABELS",
    "E_TASK_NAMES",
    "E_TASK_SCENES",
    "GARMENTLAB_ASSET_RELATIVE_PATH",
    "resolve_garmentlab_asset_path",
    "selected_e01_rest_shape_mode",
    "spawn_e01_garmentlab_surface_deformable",
]
