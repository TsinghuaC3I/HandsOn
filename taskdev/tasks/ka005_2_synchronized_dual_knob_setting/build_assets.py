#!/usr/bin/env python3
"""Build the clean-room KA005-2 visual USD assets.

Run with a build-time Python that exposes ``pxr`` (for example an isolated
``usd-core`` environment).  The collection runtime itself remains pinned to
Isaac Sim 5.1.  The script reads only the task-local copies of the two
user-supplied source archives, selects the first metal Rotary Knob variant,
normalizes its axis and scale, and authors an auditable rounded Project Boxy
derivative.  Physics is intentionally authored by the runtime so these files
remain visual-only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
from typing import Any, Iterable

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


TASK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_DIR.parents[2]
SOURCE_DIR = TASK_DIR / "source"
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "handson"
    / "handson"
    / "sim"
    / "envs"
    / "usd"
    / "b320_bimanual_benchmark"
    / "assets"
    / "ka005_2_dual_knob"
)

KNOB_SOURCE = SOURCE_DIR / "rotary_knob" / "scene.gltf"
KNOB_BINARY = SOURCE_DIR / "rotary_knob" / "scene.bin"
PROJECT_BOXY_SOURCE = (
    SOURCE_DIR / "project_boxy" / "project-boxy-master" / "project_box.scad"
)
ROTARY_KNOB_LICENSE = SOURCE_DIR / "rotary_knob" / "license.txt"
PROJECT_BOXY_LICENSE = (
    SOURCE_DIR / "project_boxy" / "project-boxy-master" / "LICENSE"
)

SELECTED_KNOB_NODES = {
    "fixed": ("Knob base_0",),
    "moving": ("Grip_1", "Top face_2", "Index line_3"),
}
TARGET_GRIP_DIAMETER_M = 0.060

_COMPONENT_DTYPES = {
    5120: np.dtype("<i1"),
    5121: np.dtype("<u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_TYPE_WIDTHS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _node_matrix(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4, order="F")
    translation = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    x, y, z, w = np.asarray(
        node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64
    )
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


class GltfDocument:
    def __init__(self, source: Path, binary: Path):
        self.source = source
        self.data = json.loads(source.read_text(encoding="utf-8"))
        self.binary = binary.read_bytes()
        self.parents: dict[int, int] = {}
        for parent_index, node in enumerate(self.data["nodes"]):
            for child_index in node.get("children", []):
                self.parents[int(child_index)] = int(parent_index)

    def world_matrix(self, node_index: int) -> np.ndarray:
        chain: list[int] = []
        current: int | None = int(node_index)
        while current is not None:
            chain.append(current)
            current = self.parents.get(current)
        matrix = np.eye(4, dtype=np.float64)
        for index in reversed(chain):
            matrix = matrix @ _node_matrix(self.data["nodes"][index])
        return matrix

    def find_node(self, name: str) -> int:
        matches = [
            index
            for index, node in enumerate(self.data["nodes"])
            if node.get("name") == name
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one glTF node named {name!r}; got {matches}")
        return matches[0]

    def mesh_node(self, selected_parent: int) -> int:
        candidates = [
            index
            for index in [selected_parent, *self.data["nodes"][selected_parent].get("children", [])]
            if "mesh" in self.data["nodes"][index]
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Selected node {selected_parent} must resolve to one mesh; got {candidates}"
            )
        return candidates[0]

    def accessor(self, index: int) -> np.ndarray:
        accessor = self.data["accessors"][int(index)]
        if "sparse" in accessor:
            raise ValueError("Sparse glTF accessors are not supported by this asset build")
        component_type = int(accessor["componentType"])
        dtype = _COMPONENT_DTYPES[component_type]
        width = _TYPE_WIDTHS[str(accessor["type"])]
        count = int(accessor["count"])
        view = self.data["bufferViews"][int(accessor["bufferView"])]
        if int(view.get("buffer", 0)) != 0:
            raise ValueError("Rotary Knob unexpectedly references multiple glTF buffers")
        start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        item_bytes = dtype.itemsize * width
        stride = int(view.get("byteStride", item_bytes))
        if stride == item_bytes:
            result = np.frombuffer(
                self.binary,
                dtype=dtype,
                count=count * width,
                offset=start,
            ).reshape(count, width)
        else:
            result = np.empty((count, width), dtype=dtype)
            unpack = struct.Struct("<" + {
                5120: "b",
                5121: "B",
                5122: "h",
                5123: "H",
                5125: "I",
                5126: "f",
            }[component_type] * width)
            for row in range(count):
                result[row] = unpack.unpack_from(self.binary, start + row * stride)
        if bool(accessor.get("normalized", False)) and not np.issubdtype(dtype, np.floating):
            info = np.iinfo(dtype)
            if np.issubdtype(dtype, np.signedinteger):
                result = np.maximum(result.astype(np.float64) / info.max, -1.0)
            else:
                result = result.astype(np.float64) / info.max
        return np.asarray(result).copy()

    def primitive_for(self, selected_name: str) -> tuple[dict[str, Any], np.ndarray]:
        selected = self.find_node(selected_name)
        mesh_node = self.mesh_node(selected)
        mesh_index = int(self.data["nodes"][mesh_node]["mesh"])
        primitives = self.data["meshes"][mesh_index].get("primitives", [])
        if len(primitives) != 1 or int(primitives[0].get("mode", 4)) != 4:
            raise ValueError(f"{selected_name} must be a single triangle primitive")
        return primitives[0], self.world_matrix(mesh_node)


@dataclass(frozen=True)
class KnobPart:
    name: str
    points_source: np.ndarray
    normals_source: np.ndarray
    texcoords: np.ndarray
    indices: np.ndarray
    material_index: int


def _load_knob_parts(document: GltfDocument) -> dict[str, KnobPart]:
    parts: dict[str, KnobPart] = {}
    for names in SELECTED_KNOB_NODES.values():
        for name in names:
            primitive, world = document.primitive_for(name)
            attributes = primitive["attributes"]
            local_points = document.accessor(attributes["POSITION"]).astype(np.float64)
            local_normals = document.accessor(attributes["NORMAL"]).astype(np.float64)
            points_h = np.concatenate(
                [local_points, np.ones((len(local_points), 1), dtype=np.float64)], axis=1
            )
            points = (world @ points_h.T).T[:, :3]
            normal_matrix = np.linalg.inv(world[:3, :3]).T
            normals = (normal_matrix @ local_normals.T).T
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
            texcoords = document.accessor(attributes["TEXCOORD_0"]).astype(np.float64)
            indices = document.accessor(primitive["indices"]).reshape(-1).astype(np.int64)
            if len(indices) % 3:
                raise ValueError(f"{name} has a non-triangular index count")
            parts[name] = KnobPart(
                name=name,
                points_source=points,
                normals_source=normals,
                texcoords=texcoords,
                indices=indices,
                material_index=int(primitive.get("material", -1)),
            )
    return parts


def _normalize_knob_parts(parts: dict[str, KnobPart]) -> tuple[dict[str, KnobPart], dict[str, Any]]:
    grip = parts["Grip_1"].points_source
    center_x = 0.5 * (float(grip[:, 0].min()) + float(grip[:, 0].max()))
    center_z = 0.5 * (float(grip[:, 2].min()) + float(grip[:, 2].max()))
    all_points = np.concatenate([part.points_source for part in parts.values()], axis=0)
    bottom_y = float(all_points[:, 1].min())
    grip_diameter = max(float(np.ptp(grip[:, 0])), float(np.ptp(grip[:, 2])))
    # Sketchfab's scene wrapper cancels two display rotations before the
    # component matrix is applied.  The resulting selected Grip bounds are
    # about 0.403 m across; keep a broad but fail-closed source check so an
    # unrelated archive cannot silently pass this deterministic conversion.
    if not 0.39 < grip_diameter < 0.42:
        raise ValueError(f"Unexpected Rotary Knob grip diameter: {grip_diameter}")
    scale = TARGET_GRIP_DIAMETER_M / grip_diameter
    normalized: dict[str, KnobPart] = {}
    # glTF is right-handed Y-up.  Rx(+90 deg) maps (x, y, z) to (x, -z, y).
    axis_conversion = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    source_origin = np.asarray([center_x, bottom_y, center_z], dtype=np.float64)
    for name, part in parts.items():
        points = ((part.points_source - source_origin) @ axis_conversion.T) * scale
        normals = part.normals_source @ axis_conversion.T
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
        normalized[name] = KnobPart(
            name=name,
            points_source=points,
            normals_source=normals,
            texcoords=part.texcoords,
            indices=part.indices,
            material_index=part.material_index,
        )

    # The downloaded showcase rotates the first metal knob decoratively.
    # Normalize the moving index to +X so q=0, positive joint rotation, and
    # the authored target ticks share one unambiguous task convention.
    index_center = normalized["Index line_3"].points_source[:, :2].mean(axis=0)
    index_angle = math.atan2(float(index_center[1]), float(index_center[0]))
    cosine = math.cos(-index_angle)
    sine = math.sin(-index_angle)
    zero_rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    normalized = {
        name: KnobPart(
            name=part.name,
            points_source=part.points_source @ zero_rotation.T,
            normals_source=part.normals_source @ zero_rotation.T,
            texcoords=part.texcoords,
            indices=part.indices,
            material_index=part.material_index,
        )
        for name, part in normalized.items()
    }

    bounds: dict[str, dict[str, list[float]]] = {}
    for name, part in normalized.items():
        points = part.points_source
        bounds[name] = {
            "min_m": [float(value) for value in points.min(axis=0)],
            "max_m": [float(value) for value in points.max(axis=0)],
        }
    moving_points = np.concatenate(
        [normalized[name].points_source for name in SELECTED_KNOB_NODES["moving"]],
        axis=0,
    )
    fixed_points = normalized["Knob base_0"].points_source
    metadata = {
        "source_axis": "+Y",
        "runtime_axis": "+Z",
        "axis_conversion": "Rx(+90deg): (x,y,z)->(x,-z,y)",
        "zero_index_rotation_deg": math.degrees(-index_angle),
        "source_axis_origin_m": [center_x, bottom_y, center_z],
        "source_grip_diameter_m": grip_diameter,
        "uniform_scale": scale,
        "target_grip_diameter_m": TARGET_GRIP_DIAMETER_M,
        "component_bounds": bounds,
        "fixed_bounds_m": {
            "min": [float(value) for value in fixed_points.min(axis=0)],
            "max": [float(value) for value in fixed_points.max(axis=0)],
        },
        "moving_bounds_m": {
            "min": [float(value) for value in moving_points.min(axis=0)],
            "max": [float(value) for value in moving_points.max(axis=0)],
        },
    }
    return normalized, metadata


def _new_stage(path: Path, default_prim: str) -> tuple[Usd.Stage, UsdGeom.Xform]:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, f"/{default_prim}")
    stage.SetDefaultPrim(root.GetPrim())
    return stage, root


def _make_preview_material(
    stage: Usd.Stage,
    path: str,
    *,
    color: tuple[float, float, float],
    metallic: float,
    roughness: float,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _make_textured_material(
    stage: Usd.Stage,
    path: str,
    *,
    base_color_asset: str,
    metallic_roughness_asset: str,
    fallback_color: tuple[float, float, float],
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*fallback_color)
    )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.65)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.38)

    reader = UsdShade.Shader.Define(stage, f"{path}/PrimvarReader_st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    base = UsdShade.Shader.Define(stage, f"{path}/BaseColorTexture")
    base.CreateIdAttr("UsdUVTexture")
    base.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(base_color_asset))
    base.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    base.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    base.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.GetInput("diffuseColor").ConnectToSource(base.ConnectableAPI(), "rgb")

    mr = UsdShade.Shader.Define(stage, f"{path}/MetallicRoughnessTexture")
    mr.CreateIdAttr("UsdUVTexture")
    mr.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(metallic_roughness_asset)
    )
    mr.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    mr.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    mr.CreateOutput("g", Sdf.ValueTypeNames.Float)
    mr.CreateOutput("b", Sdf.ValueTypeNames.Float)
    shader.GetInput("roughness").ConnectToSource(mr.ConnectableAPI(), "g")
    shader.GetInput("metallic").ConnectToSource(mr.ConnectableAPI(), "b")

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _author_mesh(
    stage: Usd.Stage,
    path: str,
    part: KnobPart,
    material: UsdShade.Material,
) -> None:
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = [Gf.Vec3f(*map(float, row)) for row in part.points_source]
    normals = [Gf.Vec3f(*map(float, row)) for row in part.normals_source]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * (len(part.indices) // 3))
    mesh.CreateFaceVertexIndicesAttr([int(value) for value in part.indices])
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    extent = UsdGeom.PointBased.ComputeExtent(points)
    mesh.CreateExtentAttr(extent)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set([Gf.Vec2f(float(row[0]), 1.0 - float(row[1])) for row in part.texcoords])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    mesh.GetPrim().SetCustomDataByKey("handson:sourceNode", part.name)
    mesh.GetPrim().SetCustomDataByKey("handson:visualOnly", True)


def _author_knob_assets(
    output_dir: Path,
    parts: dict[str, KnobPart],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    fixed_path = output_dir / "rotary_knob_fixed_visual.usda"
    moving_path = output_dir / "rotary_knob_moving_visual.usda"

    fixed_stage, fixed_root = _new_stage(fixed_path, "RotaryKnobFixedVisual")
    fixed_root.GetPrim().SetCustomDataByKey("handson:license", "CC-BY-4.0")
    fixed_root.GetPrim().SetCustomDataByKey("handson:author", "0x.anas")
    fixed_root.GetPrim().SetCustomDataByKey(
        "handson:source",
        "https://sketchfab.com/3d-models/rotary-knob-d3dd68a3c5ae4b3dbe861f2bcffa376e",
    )
    gray = _make_textured_material(
        fixed_stage,
        "/RotaryKnobFixedVisual/Looks/GreyPaintedMetal",
        base_color_asset="textures/Grey_painted_damaged_metal_baseColor.jpeg",
        metallic_roughness_asset="textures/Grey_painted_damaged_metal_metallicRoughness.png",
        fallback_color=(0.34, 0.36, 0.38),
    )
    _author_mesh(
        fixed_stage,
        "/RotaryKnobFixedVisual/KnobBase",
        parts["Knob base_0"],
        gray,
    )
    fixed_stage.GetRootLayer().Save()

    moving_stage, moving_root = _new_stage(moving_path, "RotaryKnobMovingVisual")
    moving_root.GetPrim().SetCustomDataByKey("handson:license", "CC-BY-4.0")
    moving_root.GetPrim().SetCustomDataByKey("handson:author", "0x.anas")
    moving_root.GetPrim().SetCustomDataByKey("handson:rotationAxis", "+Z")
    nickel = _make_textured_material(
        moving_stage,
        "/RotaryKnobMovingVisual/Looks/BlackNickel",
        base_color_asset="textures/Black_Nickel_baseColor.jpeg",
        metallic_roughness_asset="textures/Black_Nickel_metallicRoughness.png",
        fallback_color=(0.06, 0.07, 0.08),
    )
    index = _make_preview_material(
        moving_stage,
        "/RotaryKnobMovingVisual/Looks/Index",
        color=(0.95, 0.62, 0.04),
        metallic=0.0,
        roughness=0.30,
    )
    for name, prim_name in (
        ("Grip_1", "Grip"),
        ("Top face_2", "TopFace"),
        ("Index line_3", "IndexLine"),
    ):
        _author_mesh(
            moving_stage,
            f"/RotaryKnobMovingVisual/{prim_name}",
            parts[name],
            index if name == "Index line_3" else nickel,
        )
    moving_stage.GetRootLayer().Save()
    return fixed_path, moving_path


def _rounded_prism(
    width_x: float,
    width_y: float,
    z_min: float,
    z_max: float,
    radius: float,
    segments_per_corner: int,
) -> tuple[list[Gf.Vec3f], list[int], list[int]]:
    if radius <= 0.0 or radius * 2.0 >= min(width_x, width_y):
        raise ValueError("Rounded-prism radius is outside the valid envelope")
    centers = (
        (width_x * 0.5 - radius, width_y * 0.5 - radius, 0.0),
        (-width_x * 0.5 + radius, width_y * 0.5 - radius, 90.0),
        (-width_x * 0.5 + radius, -width_y * 0.5 + radius, 180.0),
        (width_x * 0.5 - radius, -width_y * 0.5 + radius, 270.0),
    )
    loop: list[tuple[float, float]] = []
    for cx, cy, start_deg in centers:
        for step in range(segments_per_corner + 1):
            angle = math.radians(start_deg + step * 90.0 / segments_per_corner)
            loop.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    points = [Gf.Vec3f(x, y, z_min) for x, y in loop] + [
        Gf.Vec3f(x, y, z_max) for x, y in loop
    ]
    count = len(loop)
    face_counts = [count, count] + [4] * count
    face_indices: list[int] = list(reversed(range(count))) + list(range(count, 2 * count))
    for index in range(count):
        next_index = (index + 1) % count
        face_indices.extend([index, next_index, count + next_index, count + index])
    return points, face_counts, face_indices


def _author_enclosure_asset(output_dir: Path) -> tuple[Path, dict[str, Any]]:
    output_path = output_dir / "project_boxy_enclosure_visual.usda"
    stage, root = _new_stage(output_path, "ProjectBoxyEnclosureVisual")
    root.GetPrim().SetCustomDataByKey("handson:license", "MIT")
    root.GetPrim().SetCustomDataByKey("handson:author", "abusetech")
    root.GetPrim().SetCustomDataByKey(
        "handson:source", "https://github.com/abusetech/project-boxy"
    )
    root.GetPrim().SetCustomDataByKey(
        "handson:derivedDimensionsM", "0.21,0.49,0.05"
    )
    root.GetPrim().SetCustomDataByKey("handson:visualOnly", True)

    body_material = _make_preview_material(
        stage,
        "/ProjectBoxyEnclosureVisual/Looks/Body",
        color=(0.085, 0.095, 0.11),
        metallic=0.28,
        roughness=0.48,
    )
    lid_material = _make_preview_material(
        stage,
        "/ProjectBoxyEnclosureVisual/Looks/Lid",
        color=(0.20, 0.23, 0.27),
        metallic=0.38,
        roughness=0.36,
    )
    seam_material = _make_preview_material(
        stage,
        "/ProjectBoxyEnclosureVisual/Looks/Seam",
        color=(0.015, 0.018, 0.022),
        metallic=0.05,
        roughness=0.72,
    )
    screw_material = _make_preview_material(
        stage,
        "/ProjectBoxyEnclosureVisual/Looks/Screw",
        color=(0.42, 0.45, 0.48),
        metallic=0.90,
        roughness=0.24,
    )

    for name, dimensions, z_range, radius, material in (
        ("BoxBody", (0.210, 0.490), (-0.025, 0.019), 0.012, body_material),
        ("Lid", (0.206, 0.486), (0.0205, 0.025), 0.010, lid_material),
        ("LidSeam", (0.208, 0.488), (0.0188, 0.0208), 0.011, seam_material),
    ):
        points, counts, indices = _rounded_prism(
            dimensions[0], dimensions[1], z_range[0], z_range[1], radius, 8
        )
        mesh = UsdGeom.Mesh.Define(stage, f"/ProjectBoxyEnclosureVisual/{name}")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
        mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        mesh.GetPrim().SetCustomDataByKey("handson:visualOnly", True)

    for x in (-0.082, 0.082):
        for y in (-0.222, 0.222):
            screw = UsdGeom.Cylinder.Define(
                stage,
                f"/ProjectBoxyEnclosureVisual/Screw_{'p' if x > 0 else 'n'}x_"
                f"{'p' if y > 0 else 'n'}y",
            )
            screw.CreateAxisAttr(UsdGeom.Tokens.z)
            screw.CreateRadiusAttr(0.0042)
            screw.CreateHeightAttr(0.0018)
            screw.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0254))
            UsdShade.MaterialBindingAPI.Apply(screw.GetPrim()).Bind(screw_material)
            screw.GetPrim().SetCustomDataByKey("handson:visualOnly", True)

    stage.GetRootLayer().Save()
    metadata = {
        "source_design": "Project Boxy parameterized body/lid/chamfer design",
        "source_dimensions_unit": "millimetres",
        "runtime_dimensions_m": [0.21, 0.49, 0.05],
        "rounding_radius_m": 0.012,
        "lid_thickness_m": 0.0045,
        "moving_visual": False,
        "collision_authored_in_visual_asset": False,
    }
    return output_path, metadata


def _write_manifest(
    output_dir: Path,
    generated: Iterable[Path],
    knob_metadata: dict[str, Any],
    enclosure_metadata: dict[str, Any],
    runtime_dependencies: Iterable[Path],
) -> Path:
    manifest_path = output_dir / "asset_manifest.json"
    payload = {
        "schema": "handson_ka005_2_asset_manifest_v1",
        "task_id": "B320-KA005-2",
        "task_name": "t38_set_dual_knobs",
        "clean_room": True,
        "sources": {
            "rotary_knob": {
                "title": "Rotary Knob",
                "author": "0x.anas",
                "source_url": "https://sketchfab.com/3d-models/rotary-knob-d3dd68a3c5ae4b3dbe861f2bcffa376e",
                "license": "CC-BY-4.0",
                "original_archive_sha256": "9f65d6d81959812851dc12205d3412c0fbe3f4958e0fa8cb4c17bb4451e9b87b",
                "selected_nodes": [
                    "Knob base_0",
                    "Grip_1",
                    "Top face_2",
                    "Index line_3",
                ],
                "geometry": knob_metadata,
            },
            "project_boxy": {
                "title": "Project Boxy",
                "author": "abusetech",
                "source_url": "https://github.com/abusetech/project-boxy",
                "license": "MIT",
                "original_archive_sha256": "0726041e66f680b6b5dc61b0dcb5c76219bcf42f83b5825bd7913681635e6299",
                "geometry": enclosure_metadata,
            },
        },
        "generated_files": {
            path.name: _sha256(path) for path in sorted(generated)
        },
        "runtime_dependencies": {
            path.relative_to(output_dir).as_posix(): _sha256(path)
            for path in sorted(runtime_dependencies)
        },
        "physics": {
            "visual_files_contain_physics_schemas": False,
            "runtime_base_collider": "analytic box derived from enclosure dimensions",
            "runtime_knob_colliders": "analytic cylinders derived from normalized moving bounds",
        },
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _copy_runtime_dependencies(output_dir: Path) -> tuple[Path, ...]:
    mappings = {
        SOURCE_DIR / "rotary_knob" / "textures" / "Grey_painted_damaged_metal_baseColor.jpeg": output_dir / "textures" / "Grey_painted_damaged_metal_baseColor.jpeg",
        SOURCE_DIR / "rotary_knob" / "textures" / "Grey_painted_damaged_metal_metallicRoughness.png": output_dir / "textures" / "Grey_painted_damaged_metal_metallicRoughness.png",
        SOURCE_DIR / "rotary_knob" / "textures" / "Grey_painted_damaged_metal_normal.jpeg": output_dir / "textures" / "Grey_painted_damaged_metal_normal.jpeg",
        SOURCE_DIR / "rotary_knob" / "textures" / "Black_Nickel_baseColor.jpeg": output_dir / "textures" / "Black_Nickel_baseColor.jpeg",
        SOURCE_DIR / "rotary_knob" / "textures" / "Black_Nickel_metallicRoughness.png": output_dir / "textures" / "Black_Nickel_metallicRoughness.png",
        SOURCE_DIR / "rotary_knob" / "textures" / "Black_Nickel_normal.jpeg": output_dir / "textures" / "Black_Nickel_normal.jpeg",
        ROTARY_KNOB_LICENSE: output_dir / "licenses" / "Rotary_Knob_CC-BY-4.0.txt",
        PROJECT_BOXY_LICENSE: output_dir / "licenses" / "Project_Boxy_MIT.txt",
    }
    copied: list[Path] = []
    for source, destination in mappings.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append(destination)
    return tuple(copied)


def build(output_dir: Path) -> dict[str, Any]:
    for required in (KNOB_SOURCE, KNOB_BINARY, PROJECT_BOXY_SOURCE):
        if not required.is_file():
            raise FileNotFoundError(required)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dependencies = _copy_runtime_dependencies(output_dir)
    document = GltfDocument(KNOB_SOURCE, KNOB_BINARY)
    parts, knob_metadata = _normalize_knob_parts(_load_knob_parts(document))
    fixed_path, moving_path = _author_knob_assets(output_dir, parts, knob_metadata)
    enclosure_path, enclosure_metadata = _author_enclosure_asset(output_dir)
    manifest_path = _write_manifest(
        output_dir,
        (fixed_path, moving_path, enclosure_path),
        knob_metadata,
        enclosure_metadata,
        runtime_dependencies,
    )
    return {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "generated": [str(fixed_path), str(moving_path), str(enclosure_path)],
        "knob": knob_metadata,
        "enclosure": enclosure_metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    result = build(arguments.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
