#!/usr/bin/env python3
"""Build the selected Helindu handle into P0-02 fixed/moving USD parts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


TASK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_DIR.parents[2]
SOURCE_DIR = TASK_DIR / "assets/helindu_door_handle/source"
ARCHIVE_PATH = SOURCE_DIR / "door_handle.zip"
RUNTIME_DIR = (
    REPOSITORY_ROOT
    / "handson/handson/sim/isaaclab/usd"
    / "t17_open_latched_door/helindu_door_handle"
)
EVIDENCE_PATH = TASK_DIR / "evidence/helindu_handle_build.json"

EXPECTED_ARCHIVE_SHA256 = (
    "330c5bf51744ae54468c55896280d10d6c5ef0d7b63783112fe8052ecf36affd"
)
EXPECTED_SOURCE_ENTRIES = {"license.txt", "scene.bin", "scene.gltf"}
DOOR_ROOT_SCALE = 0.60
TARGET_OVERALL_LENGTH_M = 0.150
GRASPABLE_LEVER_X_RANGE_M = (0.055, 0.105)
MINIMUM_GRASP_CLEARANCE_M = 0.028
DOOR_PANEL_SURFACE_Y_M = 0.04000015401006031
HANDLE_JOINT_POS0_Y_M = 0.06933168
MOUNT_OFFSET_Y_M = DOOR_PANEL_SURFACE_Y_M - HANDLE_JOINT_POS0_Y_M
HANDLE_JOINT_LOCAL_POS0_M = (-0.6834181, 0.06933168, 0.0491462)

# Collision hulls overlap slightly along the lever length.  Every hull is
# derived from selected visual vertices; no old Door042 handle collider is
# reused after integration.
LEVER_COLLISION_X_ZONES_M = (
    (-0.020, 0.025),
    (0.015, 0.065),
    (0.055, 0.105),
    (0.095, 0.130),
)
GPU_COLLIDER_PRIMITIVE = "Cube"
GPU_COLLIDER_LOGICAL_VERTICES = 8
GPU_COLLIDER_CROSS_SECTION_SCALE = np.asarray(
    [1.0, 0.75, 0.75], dtype=np.float64
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_accessor(document: dict, blob: bytes, index: int) -> np.ndarray:
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    component_types = {5125: np.dtype("<u4"), 5126: np.dtype("<f4")}
    cardinalities = {"SCALAR": 1, "VEC3": 3}
    dtype = component_types[accessor["componentType"]]
    width = cardinalities[accessor["type"]]
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    values = np.frombuffer(
        blob,
        dtype=dtype,
        count=int(accessor["count"]) * width,
        offset=offset,
    )
    return values.reshape(int(accessor["count"]), width).copy()


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(
            (4, 4), order="F"
        )
    matrix = np.eye(4, dtype=np.float64)
    if "scale" in node:
        matrix[:3, :3] = np.diag(np.asarray(node["scale"], dtype=np.float64))
    if "translation" in node:
        matrix[:3, 3] = np.asarray(node["translation"], dtype=np.float64)
    if "rotation" in node:
        x, y, z, w = np.asarray(node["rotation"], dtype=np.float64)
        matrix[:3, :3] = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
    return matrix


def _mesh_world_matrix(document: dict) -> np.ndarray:
    roots = document["scenes"][document.get("scene", 0)]["nodes"]
    matches: list[np.ndarray] = []

    def visit(index: int, parent: np.ndarray) -> None:
        node = document["nodes"][index]
        current = parent @ _node_matrix(node)
        if "mesh" in node:
            matches.append(current)
        for child in node.get("children", []):
            visit(int(child), current)

    for root in roots:
        visit(int(root), np.eye(4, dtype=np.float64))
    if len(matches) != 1:
        raise RuntimeError(f"expected one mesh node, found {len(matches)}")
    return matches[0]


def _connected_components(vertex_count: int, faces: np.ndarray) -> list[np.ndarray]:
    parent = np.arange(vertex_count, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for triangle in faces:
        union(int(triangle[0]), int(triangle[1]))
        union(int(triangle[1]), int(triangle[2]))
    grouped: dict[int, list[int]] = {}
    for vertex in range(vertex_count):
        grouped.setdefault(find(vertex), []).append(vertex)
    return [
        np.asarray(values, dtype=np.int64)
        for values in sorted(grouped.values(), key=len, reverse=True)
    ]


def _extract_component(
    points: np.ndarray,
    faces: np.ndarray,
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    membership = np.zeros(len(points), dtype=bool)
    membership[vertices] = True
    selected_faces = faces[np.all(membership[faces], axis=1)]
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[vertices] = np.arange(len(vertices), dtype=np.int64)
    return points[vertices].copy(), remap[selected_faces]


def _vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(points, dtype=np.float64)
    triangles = points[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 1.0e-12):
        raise RuntimeError("visual mesh produced invalid vertex normals")
    return normals / lengths[:, None]


def _fmt_float(value: float) -> str:
    if abs(value) < 5.0e-12:
        value = 0.0
    return f"{value:.10g}"


def _fmt_vec3(values: np.ndarray) -> str:
    return "(" + ", ".join(_fmt_float(float(value)) for value in values) + ")"


def _fmt_vec3_array(values: np.ndarray) -> str:
    return "[" + ", ".join(_fmt_vec3(row) for row in values) + "]"


def _fmt_int_array(values: np.ndarray) -> str:
    return "[" + ", ".join(str(int(value)) for value in values.reshape(-1)) + "]"


def _material_block(root: str) -> str:
    return f'''    def Scope "Looks"
    {{
        def Material "Metal"
        {{
            token outputs:surface.connect = <{root}/Looks/Metal/PBR.outputs:surface>
            def Shader "PBR"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.601372, 0.601372, 0.601372)
                float inputs:metallic = 0.78
                float inputs:roughness = 0.2726753
                token outputs:surface
            }}
        }}
    }}
'''


def _visual_mesh_block(
    *,
    name: str,
    root: str,
    points: np.ndarray,
    faces: np.ndarray,
) -> str:
    normals = _vertex_normals(points, faces)
    extent = np.asarray([points.min(axis=0), points.max(axis=0)])
    return f'''    def Xform "Visuals"
    {{
        def Mesh "{name}" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            float3[] extent = {_fmt_vec3_array(extent)}
            int[] faceVertexCounts = [{", ".join("3" for _ in faces)}]
            int[] faceVertexIndices = {_fmt_int_array(faces)}
            normal3f[] normals = {_fmt_vec3_array(normals)}
            token normals:interpolation = "vertex"
            point3f[] points = {_fmt_vec3_array(points)}
            rel material:binding = <{root}/Looks/Metal>
            uniform token subdivisionScheme = "none"
        }}
    }}
'''


def _collision_block(points: np.ndarray) -> tuple[str, list[dict]]:
    sections: list[str] = ['    def Xform "Collisions"\n    {']
    evidence: list[dict] = []
    effective_world_x = points[:, 0] * DOOR_ROOT_SCALE
    covered = np.zeros(len(points), dtype=bool)
    for index, (minimum, maximum) in enumerate(LEVER_COLLISION_X_ZONES_M, start=1):
        selected = np.flatnonzero(
            (effective_world_x >= minimum) & (effective_world_x <= maximum)
        )
        covered[selected] = True
        if len(selected) < 4:
            raise RuntimeError(f"collision zone {index} has too few vertices")
        source = points[selected]
        # GPU convex-mesh cooking was not repeatably safe even after reducing
        # the visual hulls to 19-24 vertices.  Use native box primitives so the
        # first PhysX state has no task-specific GPU convex mesh at all.  The
        # cross-section is inset by 25 percent; the X spans still overlap, so
        # contact remains continuous without placing collision at the visual
        # AABB corners.
        visual_minimum = source.min(axis=0)
        visual_maximum = source.max(axis=0)
        collider_size = (
            visual_maximum - visual_minimum
        ) * GPU_COLLIDER_CROSS_SECTION_SCALE
        collider_center = 0.5 * (visual_minimum + visual_maximum)
        if np.any(collider_size <= 1.0e-12):
            raise RuntimeError(f"collision zone {index} is degenerate")
        collider_extent = np.asarray(
            [
                collider_center - 0.5 * collider_size,
                collider_center + 0.5 * collider_size,
            ]
        )
        sections.append(
            f'''        def Cube "LeverBox{index}" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            bool physics:collisionEnabled = true
            double size = 1
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            uniform token purpose = "guide"
            token visibility = "invisible"
            double3 xformOp:translate = {_fmt_vec3(collider_center)}
            double3 xformOp:scale = {_fmt_vec3(collider_size)}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}'''
        )
        evidence.append(
            {
                "name": f"LeverBox{index}",
                "primitive_type": GPU_COLLIDER_PRIMITIVE,
                "source_vertex_count": int(len(selected)),
                "hull_vertex_count": GPU_COLLIDER_LOGICAL_VERTICES,
                "face_count": 6,
                "effective_world_bounds_m": (
                    collider_extent * DOOR_ROOT_SCALE
                ).tolist(),
            }
        )
    if not bool(np.all(covered)):
        missing = effective_world_x[~covered]
        raise RuntimeError(
            "collision X zones do not cover every lever vertex: "
            f"{float(missing.min())}..{float(missing.max())}"
        )
    sections.append("    }")
    return "\n".join(sections) + "\n", evidence


def _write_component(
    path: Path,
    *,
    root_name: str,
    mesh_name: str,
    points: np.ndarray,
    faces: np.ndarray,
    collision: bool,
) -> list[dict]:
    root = f"/{root_name}"
    collision_text, collision_evidence = (
        _collision_block(points) if collision else ("", [])
    )
    text = f'''#usda 1.0
(
    defaultPrim = "{root_name}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root_name}"
{{
{_material_block(root)}{_visual_mesh_block(name=mesh_name, root=root, points=points, faces=faces)}{collision_text}}}
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return collision_evidence


def main() -> int:
    if _sha256(ARCHIVE_PATH) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("selected Helindu source archive hash drifted")
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        if set(archive.namelist()) != EXPECTED_SOURCE_ENTRIES:
            raise RuntimeError("selected Helindu archive contents drifted")
        document = json.loads(archive.read("scene.gltf"))
        blob = archive.read("scene.bin")
        license_text = archive.read("license.txt").decode("utf-8")
    if (
        document.get("asset", {}).get("extras", {}).get("source")
        != "https://sketchfab.com/3d-models/door-handle-b17802a127954b868a1250f55597142d"
        or "CC-BY-4.0" not in license_text
    ):
        raise RuntimeError("selected Helindu provenance or license drifted")

    positions = _read_accessor(document, blob, 0).astype(np.float64)
    faces = _read_accessor(document, blob, 2).reshape(-1, 3).astype(np.int64)
    matrix = _mesh_world_matrix(document)
    transformed = (matrix @ np.column_stack([positions, np.ones(len(positions))]).T).T
    transformed = transformed[:, :3]
    expected_bounds = np.asarray(
        [
            [-25.5416443, -24.2861589, 1.1411015],
            [79.8619594, 24.2134473, 42.6240841],
        ]
    )
    if not np.allclose(
        np.asarray([transformed.min(axis=0), transformed.max(axis=0)]),
        expected_bounds,
        rtol=0.0,
        atol=2.0e-5,
    ):
        raise RuntimeError("selected Helindu transformed bounds drifted")

    components = _connected_components(len(transformed), faces)
    if [len(component) for component in components] != [1321, 937]:
        raise RuntimeError("selected Helindu connected-component topology drifted")
    rosette_points, rosette_faces = _extract_component(
        transformed, faces, components[0]
    )
    lever_points, lever_faces = _extract_component(
        transformed, faces, components[1]
    )
    if not (
        np.ptp(rosette_points[:, 0]) > 48.0
        and np.ptp(rosette_points[:, 1]) > 48.0
        and lever_points[:, 0].max() > 79.0
    ):
        raise RuntimeError("selected Helindu rosette/lever classification drifted")

    rosette_min = rosette_points.min(axis=0)
    rosette_max = rosette_points.max(axis=0)
    axis_x = float((rosette_min[0] + rosette_max[0]) / 2.0)
    axis_y = float((rosette_min[1] + rosette_max[1]) / 2.0)
    rosette_back_z = float(rosette_min[2])
    rosette_left_m = (axis_x - float(rosette_min[0])) * 0.001
    lever_tip_m = (float(lever_points[:, 0].max()) - axis_x) * 0.001
    lever_x_scale = (TARGET_OVERALL_LENGTH_M - rosette_left_m) / lever_tip_m
    if not 1.5 < lever_x_scale < 1.6:
        raise RuntimeError(f"unexpected Helindu lever X scale {lever_x_scale}")

    def normalize(points: np.ndarray, *, moving: bool) -> np.ndarray:
        x_scale = lever_x_scale if moving else 1.0
        result = np.empty_like(points, dtype=np.float64)
        # Author pre-root coordinates.  Door042's composed 0.60 scale produces
        # the intended world dimensions without scaling the rigid-body prim.
        result[:, 0] = (points[:, 0] - axis_x) * 0.001 * x_scale / DOOR_ROOT_SCALE
        result[:, 1] = (
            (points[:, 2] - rosette_back_z) * 0.001 / DOOR_ROOT_SCALE
            + MOUNT_OFFSET_Y_M
        )
        result[:, 2] = -(points[:, 1] - axis_y) * 0.001 / DOOR_ROOT_SCALE
        return result

    rosette_runtime = normalize(rosette_points, moving=False)
    lever_runtime = normalize(lever_points, moving=True)
    rosette_path = RUNTIME_DIR / "helindu_rosette.usda"
    lever_path = RUNTIME_DIR / "helindu_lever.usda"
    _write_component(
        rosette_path,
        root_name="HelinduRosette",
        mesh_name="RosetteVisual",
        points=rosette_runtime,
        faces=rosette_faces,
        collision=False,
    )
    collision_evidence = _write_component(
        lever_path,
        root_name="HelinduLever",
        mesh_name="LeverVisual",
        points=lever_runtime,
        faces=lever_faces,
        collision=True,
    )

    effective_rosette_bounds = (
        np.asarray([rosette_runtime.min(axis=0), rosette_runtime.max(axis=0)])
        * DOOR_ROOT_SCALE
    )
    effective_lever_bounds = (
        np.asarray([lever_runtime.min(axis=0), lever_runtime.max(axis=0)])
        * DOOR_ROOT_SCALE
    )
    overall_min = np.minimum(effective_rosette_bounds[0], effective_lever_bounds[0])
    overall_max = np.maximum(effective_rosette_bounds[1], effective_lever_bounds[1])
    if not np.isclose(overall_max[0] - overall_min[0], TARGET_OVERALL_LENGTH_M, atol=1e-8):
        raise RuntimeError("built Helindu handle length missed the task target")
    effective_lever_points = lever_runtime * DOOR_ROOT_SCALE
    grasp_mask = (
        (effective_lever_points[:, 0] >= GRASPABLE_LEVER_X_RANGE_M[0])
        & (effective_lever_points[:, 0] <= GRASPABLE_LEVER_X_RANGE_M[1])
    )
    if int(np.count_nonzero(grasp_mask)) < 50:
        raise RuntimeError("built Helindu handle has too few grasp-zone vertices")
    panel_surface_relative_to_joint_m = MOUNT_OFFSET_Y_M * DOOR_ROOT_SCALE
    minimum_grasp_clearance_m = float(
        effective_lever_points[grasp_mask, 1].min()
        - panel_surface_relative_to_joint_m
    )
    if minimum_grasp_clearance_m < MINIMUM_GRASP_CLEARANCE_M:
        raise RuntimeError(
            "built Helindu handle grip clearance is too small: "
            f"{minimum_grasp_clearance_m:.6f} m"
        )
    evidence = {
        "schema": "handson.p002.helindu_handle_build.v2",
        "source": {
            "title": "Door Handle",
            "author": "Helindu",
            "url": "https://sketchfab.com/3d-models/door-handle-b17802a127954b868a1250f55597142d",
            "license": "CC-BY-4.0",
            "archive": str(ARCHIVE_PATH.relative_to(REPOSITORY_ROOT)),
            "archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "vertices": int(len(transformed)),
            "faces": int(len(faces)),
            "connected_components": [int(len(component)) for component in components],
        },
        "normalization": {
            "source_units": "millimetres after authored glTF node transforms",
            "source_up_axis": "Y before Sketchfab glTF transforms; audited transformed frame used",
            "task_axis_mapping": "asset X -> Handle002 X; asset Z -> Handle002 Y; -asset Y -> Handle002 Z",
            "door_root_scale": DOOR_ROOT_SCALE,
            "lever_x_scale": float(lever_x_scale),
            "target_overall_length_m": TARGET_OVERALL_LENGTH_M,
            "mount_offset_y_pre_root_m": MOUNT_OFFSET_Y_M,
            "joint_local_pos0_m": list(HANDLE_JOINT_LOCAL_POS0_M),
        },
        "effective_world_bounds_m": {
            "rosette": effective_rosette_bounds.tolist(),
            "lever": effective_lever_bounds.tolist(),
            "overall": [overall_min.tolist(), overall_max.tolist()],
            "overall_dimensions": (overall_max - overall_min).tolist(),
            "lever_to_panel_min_clearance_m": float(
                effective_lever_bounds[0, 1]
                - panel_surface_relative_to_joint_m
            ),
            "graspable_lever_x_range_m": list(GRASPABLE_LEVER_X_RANGE_M),
            "minimum_grasp_clearance_m": minimum_grasp_clearance_m,
        },
        "runtime": {
            "rosette_usda": str(rosette_path.relative_to(REPOSITORY_ROOT)),
            "rosette_sha256": _sha256(rosette_path),
            "lever_usda": str(lever_path.relative_to(REPOSITORY_ROOT)),
            "lever_sha256": _sha256(lever_path),
            "collision_hulls": collision_evidence,
            "fixed_rosette_collision": False,
        },
    }
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
