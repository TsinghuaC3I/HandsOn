#!/usr/bin/env python3
"""Build the reviewed Poly Haven flathead screwdriver for AS019.

The source archive is immutable input.  This builder validates its byte-exact
allowlist, decodes the single glTF primitive without a lossy interchange step,
bakes the audited source-to-task transform into the visual and collision
geometry, and atomically writes one Isaac Sim 5.1 USD plus its provenance
manifest.  Collision geometry is a five-piece, source-derived convex
decomposition under the same rigid root as the one visible PBR mesh.

The module deliberately keeps all USD imports inside ``author_usd`` so its
archive, transform, collider, and mass-model contracts can be tested with an
ordinary Python interpreter.  The command-line entry point launches the
Isaac Sim 5.1 application before authoring USD schemas.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
from scipy.spatial import ConvexHull


ASSET_ID = "polyhaven-flathead-screwdriver-as019"
ASSET_REPO_RELATIVE_DIR = (
    "taskdev/tasks/bench320_as019_screwdriver_safe_orientation_handover/"
    "assets/polyhaven_flathead_screwdriver"
)
TITLE = "Flathead Screwdriver"
AUTHOR = "Dylan Detwiller"
AUTHOR_URL = "https://polyhaven.com/all?a=Dylan%20Detwiller"
SOURCE_URL = "https://polyhaven.com/a/flathead_screwdriver"
LICENSE_NAME = "CC0-1.0"
LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
POLY_HAVEN_LICENSE_URL = "https://polyhaven.com/license"

ARCHIVE_FILENAME = "flathead_screwdriver_4k.gltf.zip"
ARCHIVE_SIZE_BYTES = 27_238_057
ARCHIVE_SHA256 = "0162ab02ff2357585ebf702fc3844e4bf6313cf043239eaf763180ab12c26dbb"


@dataclass(frozen=True)
class MemberSpec:
    path: str
    size_bytes: int
    sha256: str


MEMBER_SPECS = (
    MemberSpec(
        "flathead_screwdriver_4k.gltf",
        2_848,
        "657eb5aab940361252ecd071ed3b8e5a1f159fd27b74417829792c43419ff0c1",
    ),
    MemberSpec(
        "flathead_screwdriver.bin",
        63_656,
        "d6cf744aa352c7ed92532c2aaa98e84baf2936361aa07025a6665a8d9ca0b9b0",
    ),
    MemberSpec(
        "textures/flathead_screwdriver_arm_4k.jpg",
        7_707_968,
        "97043e2b562a0acb7835d9da26259ad17abbbe820649b97c6458af1bb5998439",
    ),
    MemberSpec(
        "textures/flathead_screwdriver_diff_4k.jpg",
        7_975_278,
        "d3df707481cd6a14bbe0a075fb06abc9684316d0933a816682f7e935b586f4b1",
    ),
    MemberSpec(
        "textures/flathead_screwdriver_nor_gl_4k.jpg",
        11_487_473,
        "b85b2ca09f710358b5a8b037654b7b676db58b8e4f4d4538b30466583676c730",
    ),
)
MEMBER_PATHS = tuple(spec.path for spec in MEMBER_SPECS)

GLTF_MEMBER = "flathead_screwdriver_4k.gltf"
BUFFER_MEMBER = "flathead_screwdriver.bin"
TEXTURE_MEMBERS = (
    "textures/flathead_screwdriver_arm_4k.jpg",
    "textures/flathead_screwdriver_diff_4k.jpg",
    "textures/flathead_screwdriver_nor_gl_4k.jpg",
)

# glTF uses metres and a right-handed Y-up frame.  The model's source +Y end
# is the sharp tip and source -Y end is the wooden tail.  The rotation maps
# +Y to task-local +X and preserves handedness.
UNIFORM_SCALE = 0.9950483266936156
ROTATION_SOURCE_TO_TASK = np.asarray(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
TRANSLATION_M = np.asarray(
    [-0.0778213224721204, 0.025, 0.0], dtype=np.float64
)
ROTATION_QUATERNION_WXYZ = (
    math.sqrt(0.5),
    0.0,
    0.0,
    -math.sqrt(0.5),
)
NORMALIZATION_MATRIX_ROW_MAJOR = np.asarray(
    [
        [0.0, UNIFORM_SCALE, 0.0, TRANSLATION_M[0]],
        [-UNIFORM_SCALE, 0.0, 0.0, TRANSLATION_M[1]],
        [0.0, 0.0, UNIFORM_SCALE, TRANSLATION_M[2]],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

EXPECTED_SOURCE_BOUNDS_M = np.asarray(
    [
        [-0.01501486822962761, -0.047413453459739685, -0.015014867298305035],
        [0.01501486822962761, 0.21388038992881775, 0.015014867298305035],
    ],
    dtype=np.float64,
)
EXPECTED_TASK_BOUNDS_M = np.asarray(
    [
        [-0.125, 0.010059480492583916, -0.014940518580705114],
        [0.13500000163912776, 0.039940519507416086, 0.014940518580705114],
    ],
    dtype=np.float64,
)

# These are measured source-coordinate rings, not guessed task-space boxes.
# Including the split ring in both adjacent hulls gives a closed, gap-free
# compound shape; the pieces only share their boundary plane.
HANDLE_SPLIT_SOURCE_Y_M = 0.02714129351079464
BLADE_START_SOURCE_Y_M = 0.18961845338344574

ROOT_PATH = "/FlatheadScrewdriver"
VISUAL_PATH = ROOT_PATH + "/Visual/SourceMesh"
MATERIAL_PATH = ROOT_PATH + "/Looks/PolyHavenPBR"
PHYSICS_MATERIAL_PATH = ROOT_PATH + "/Looks/PhysicsMaterial"
COLLIDER_PATHS = (
    ROOT_PATH + "/Colliders/TailMain",
    ROOT_PATH + "/Colliders/HandleNeck",
    ROOT_PATH + "/Colliders/Ferrule",
    ROOT_PATH + "/Colliders/Stem",
    ROOT_PATH + "/Colliders/FlatBlade",
)

NOMINAL_MASS_KG = 0.18
WOOD_DENSITY_KG_M3 = 700.0
STEEL_DENSITY_KG_M3 = 7_850.0
CONTACT_REPORT_THRESHOLD_N = 0.0
DEFAULT_STATIC_FRICTION = 0.68
DEFAULT_DYNAMIC_FRICTION = 0.68
DEFAULT_RESTITUTION = 0.0
# The largest audited gap is the grooved handle neck at 2.044 mm.  A 2.1 mm
# gate retains only 56 micrometres of headroom and prevents a future broad
# proxy from reintroducing the no-contact "pickup" failure this asset replaces.
PROTRUSION_SAMPLE_GAP_LIMIT_M = 0.0021


@dataclass(frozen=True)
class ArchiveData:
    archive_path: Path
    members: dict[str, bytes]
    member_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SourceMesh:
    points: np.ndarray
    normals: np.ndarray
    uv: np.ndarray
    faces: np.ndarray
    gltf: dict[str, Any]


@dataclass(frozen=True)
class SourceComponent:
    name: str
    points: np.ndarray
    visual_faces: np.ndarray


@dataclass(frozen=True)
class ColliderData:
    name: str
    path: str
    material: str
    source_component: str
    source_points: np.ndarray
    task_points: np.ndarray
    faces: np.ndarray
    source_y_range_m: tuple[float, float]
    internal_cut_source_y_m: float | None
    volume_m3: float
    centroid_m: np.ndarray
    protrusion_audit: dict[str, Any]


@dataclass(frozen=True)
class BuildPlan:
    archive: ArchiveData
    mesh: SourceMesh
    task_points: np.ndarray
    task_normals: np.ndarray
    components: tuple[SourceComponent, ...]
    colliders: tuple[ColliderData, ...]
    mass_model: dict[str, Any]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounds(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("bounds require a non-empty Nx3 point array")
    if not np.all(np.isfinite(points)):
        raise ValueError("bounds received non-finite points")
    return np.asarray([points.min(axis=0), points.max(axis=0)], dtype=np.float64)


def _bounds_manifest(points: np.ndarray) -> dict[str, list[float]]:
    bounds = _bounds(points)
    return {
        "min_m": [float(value) for value in bounds[0]],
        "max_m": [float(value) for value in bounds[1]],
        "extent_m": [float(value) for value in bounds[1] - bounds[0]],
        "center_m": [float(value) for value in bounds.mean(axis=0)],
    }


def transform_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return UNIFORM_SCALE * points @ ROTATION_SOURCE_TO_TASK.T + TRANSLATION_M


def inverse_transform_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return (points - TRANSLATION_M) @ ROTATION_SOURCE_TO_TASK / UNIFORM_SCALE


def transform_normals(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float64)
    transformed = normals @ ROTATION_SOURCE_TO_TASK.T
    lengths = np.linalg.norm(transformed, axis=1)
    if np.any(lengths <= 0.0) or not np.all(np.isfinite(lengths)):
        raise ValueError("source mesh contains invalid normals")
    return transformed / lengths[:, None]


def validate_archive(archive_path: Path) -> ArchiveData:
    """Validate the immutable archive before returning any member bytes."""

    archive_path = Path(archive_path).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"source archive not found: {archive_path}")
    if archive_path.name != ARCHIVE_FILENAME:
        raise ValueError(
            f"unexpected source archive name: {archive_path.name!r}; "
            f"expected {ARCHIVE_FILENAME!r}"
        )
    actual_size = archive_path.stat().st_size
    if actual_size != ARCHIVE_SIZE_BYTES:
        raise ValueError(
            f"source archive size mismatch: {actual_size} != {ARCHIVE_SIZE_BYTES}"
        )
    actual_sha = sha256_file(archive_path)
    if actual_sha != ARCHIVE_SHA256:
        raise ValueError(f"source archive SHA-256 mismatch: {actual_sha}")

    expected_by_path = {spec.path: spec for spec in MEMBER_SPECS}
    with zipfile.ZipFile(archive_path, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("source archive contains duplicate member paths")
        if tuple(names) != MEMBER_PATHS:
            raise ValueError(
                "source archive member order/allowlist mismatch: "
                f"{names!r} != {list(MEMBER_PATHS)!r}"
            )
        members: dict[str, bytes] = {}
        records: list[dict[str, Any]] = []
        for info in infos:
            pure_path = PurePosixPath(info.filename)
            if (
                pure_path.is_absolute()
                or ".." in pure_path.parts
                or "\\" in info.filename
                or info.is_dir()
            ):
                raise ValueError(f"unsafe or unexpected archive member: {info.filename!r}")
            if info.flag_bits & 0x1:
                raise ValueError(f"encrypted archive member is not allowed: {info.filename}")
            spec = expected_by_path[info.filename]
            if info.file_size != spec.size_bytes:
                raise ValueError(
                    f"member size mismatch for {info.filename}: "
                    f"{info.file_size} != {spec.size_bytes}"
                )
            data = archive.read(info)
            actual_member_sha = sha256_bytes(data)
            if actual_member_sha != spec.sha256:
                raise ValueError(
                    f"member SHA-256 mismatch for {info.filename}: {actual_member_sha}"
                )
            members[info.filename] = data
            records.append(
                {
                    "path": info.filename,
                    "size_bytes": int(info.file_size),
                    "compressed_size_bytes": int(info.compress_size),
                    "compression_method": int(info.compress_type),
                    "crc32": f"{int(info.CRC):08x}",
                    "sha256": actual_member_sha,
                }
            )
    return ArchiveData(archive_path, members, tuple(records))


_COMPONENT_DTYPES = {
    5120: np.dtype("<i1"),
    5121: np.dtype("<u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


def _read_accessor(
    gltf: dict[str, Any], buffer_bytes: bytes, accessor_index: int
) -> np.ndarray:
    accessor = gltf["accessors"][accessor_index]
    if "sparse" in accessor:
        raise ValueError("sparse glTF accessors are not allowed")
    view = gltf["bufferViews"][accessor["bufferView"]]
    if view.get("buffer") != 0:
        raise ValueError("the reviewed glTF must use exactly buffer 0")
    component_type = int(accessor["componentType"])
    if component_type not in _COMPONENT_DTYPES:
        raise ValueError(f"unsupported glTF component type: {component_type}")
    accessor_type = accessor["type"]
    if accessor_type not in _TYPE_COMPONENTS:
        raise ValueError(f"unsupported glTF accessor type: {accessor_type}")
    dtype = _COMPONENT_DTYPES[component_type]
    component_count = _TYPE_COMPONENTS[accessor_type]
    packed_stride = dtype.itemsize * component_count
    stride = int(view.get("byteStride", packed_stride))
    if stride != packed_stride:
        raise ValueError("interleaved glTF accessors are outside the reviewed source contract")
    count = int(accessor["count"])
    byte_offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    byte_length = count * packed_stride
    if byte_offset < 0 or byte_offset + byte_length > len(buffer_bytes):
        raise ValueError("glTF accessor exceeds the reviewed binary buffer")
    result = np.frombuffer(
        buffer_bytes, dtype=dtype, count=count * component_count, offset=byte_offset
    ).reshape(count, component_count)
    return np.asarray(result)


def decode_source_mesh(archive: ArchiveData) -> SourceMesh:
    gltf = json.loads(archive.members[GLTF_MEMBER].decode("utf-8"))
    expected_inventory = {
        "asset_version": gltf.get("asset", {}).get("version"),
        "scene": gltf.get("scene"),
        "scene_count": len(gltf.get("scenes", [])),
        "node_count": len(gltf.get("nodes", [])),
        "mesh_count": len(gltf.get("meshes", [])),
        "material_count": len(gltf.get("materials", [])),
        "buffer_count": len(gltf.get("buffers", [])),
    }
    if expected_inventory != {
        "asset_version": "2.0",
        "scene": 0,
        "scene_count": 1,
        "node_count": 1,
        "mesh_count": 1,
        "material_count": 1,
        "buffer_count": 1,
    }:
        raise ValueError(f"unexpected reviewed glTF inventory: {expected_inventory}")
    node = gltf["nodes"][0]
    if node != {"mesh": 0, "name": "flathead_screwdriver"}:
        raise ValueError(f"reviewed glTF node is no longer identity: {node!r}")
    mesh_record = gltf["meshes"][0]
    if mesh_record.get("name") != "Cylinder.005" or len(mesh_record.get("primitives", [])) != 1:
        raise ValueError("reviewed glTF must contain the one Cylinder.005 primitive")
    primitive = mesh_record["primitives"][0]
    if primitive.get("mode", 4) != 4 or primitive.get("material") != 0:
        raise ValueError("reviewed glTF primitive must be indexed triangles with material 0")
    if primitive.get("attributes") != {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}:
        raise ValueError("reviewed glTF primitive attribute layout changed")
    if primitive.get("indices") != 3:
        raise ValueError("reviewed glTF primitive index accessor changed")
    buffer_record = gltf["buffers"][0]
    if buffer_record != {"byteLength": 63_656, "uri": BUFFER_MEMBER}:
        raise ValueError(f"reviewed glTF buffer record changed: {buffer_record!r}")
    material = gltf["materials"][0]
    if material.get("name") != "flathead_screwdriver" or not material.get("doubleSided"):
        raise ValueError("reviewed glTF PBR material identity changed")
    if material.get("normalTexture", {}).get("index") != 0:
        raise ValueError("reviewed glTF normal texture binding changed")
    pbr = material.get("pbrMetallicRoughness", {})
    if pbr.get("baseColorTexture", {}).get("index") != 1:
        raise ValueError("reviewed glTF base-color texture binding changed")
    if pbr.get("metallicRoughnessTexture", {}).get("index") != 2:
        raise ValueError("reviewed glTF ARM texture binding changed")

    buffer_bytes = archive.members[BUFFER_MEMBER]
    if len(buffer_bytes) != int(buffer_record["byteLength"]):
        raise ValueError("reviewed glTF binary buffer length changed")
    points = _read_accessor(gltf, buffer_bytes, 0).astype(np.float64)
    normals = _read_accessor(gltf, buffer_bytes, 1).astype(np.float64)
    uv = _read_accessor(gltf, buffer_bytes, 2).astype(np.float64)
    indices = _read_accessor(gltf, buffer_bytes, 3).reshape(-1).astype(np.int64)
    if points.shape != (1558, 3) or normals.shape != (1558, 3) or uv.shape != (1558, 2):
        raise ValueError("reviewed glTF vertex/accessor counts changed")
    if indices.shape != (6900,) or len(indices) % 3:
        raise ValueError("reviewed glTF triangle index count changed")
    faces = indices.reshape(-1, 3)
    if np.any(faces < 0) or np.any(faces >= len(points)):
        raise ValueError("reviewed glTF contains out-of-range indices")
    if not all(np.all(np.isfinite(values)) for values in (points, normals, uv)):
        raise ValueError("reviewed glTF contains non-finite vertex data")
    if not np.allclose(_bounds(points), EXPECTED_SOURCE_BOUNDS_M, atol=0.0, rtol=0.0):
        raise ValueError(f"reviewed glTF source bounds changed: {_bounds(points)!r}")
    normal_lengths = np.linalg.norm(normals, axis=1)
    if not np.allclose(normal_lengths, 1.0, atol=2.0e-6, rtol=0.0):
        raise ValueError("reviewed glTF normals are not unit length")
    triangles = points[faces]
    twice_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if np.any(twice_areas <= 1.0e-14):
        raise ValueError("reviewed glTF contains a degenerate triangle")
    return SourceMesh(points, normals, uv, faces, gltf)


def _union_find_components(vertex_count: int, faces: np.ndarray) -> np.ndarray:
    parent = np.arange(vertex_count, dtype=np.int64)

    def find(value: int) -> int:
        value = int(value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            lower, upper = sorted((first_root, second_root))
            parent[upper] = lower

    for first, second, third in np.asarray(faces, dtype=np.int64):
        union(int(first), int(second))
        union(int(second), int(third))
        union(int(third), int(first))
    return np.asarray([find(index) for index in range(vertex_count)], dtype=np.int64)


def source_components(mesh: SourceMesh) -> tuple[SourceComponent, ...]:
    # Exact float32 coordinates are welded to recover connectivity hidden by
    # UV/normal seams.  No geometric tolerance or position edit is involved.
    welded_points, original_to_welded = np.unique(
        mesh.points, axis=0, return_inverse=True
    )
    welded_faces = original_to_welded[mesh.faces]
    roots = _union_find_components(len(welded_points), welded_faces)
    face_roots = roots[welded_faces[:, 0]]
    if not np.all(roots[welded_faces] == face_roots[:, None]):
        raise ValueError("a reviewed source face crosses welded components")
    root_values = sorted(set(int(value) for value in roots))
    raw_components: list[tuple[np.ndarray, np.ndarray]] = []
    for root in root_values:
        points = welded_points[roots == root]
        faces = mesh.faces[face_roots == root]
        raw_components.append((points, faces))
    raw_components.sort(key=lambda item: float(item[0][:, 1].min()))
    if tuple(len(points) for points, _ in raw_components) != (942, 160, 77):
        raise ValueError(
            "reviewed source component topology changed: "
            f"{tuple(len(points) for points, _ in raw_components)!r}"
        )
    names = ("wood_handle", "steel_ferrule", "steel_shaft_blade")
    components = tuple(
        SourceComponent(name, points, faces)
        for name, (points, faces) in zip(names, raw_components, strict=True)
    )
    expected_ranges = (
        (-0.047413453459739685, 0.061649926006793976),
        (0.061093542724847794, 0.08428783714771271),
        (0.0639905035495758, 0.21388038992881775),
    )
    actual_ranges = tuple(
        (float(component.points[:, 1].min()), float(component.points[:, 1].max()))
        for component in components
    )
    if actual_ranges != expected_ranges:
        raise ValueError(f"reviewed source component longitudinal ranges changed: {actual_ranges!r}")
    return components


def _canonical_convex_hull(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable, outward-wound triangulated hull of source points."""

    points = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(points) < 4 or np.linalg.matrix_rank(points - points.mean(axis=0)) != 3:
        raise ValueError("collider source points do not span a finite 3-D hull")
    hull = ConvexHull(points)
    hull_source_indices = np.asarray(hull.vertices, dtype=np.int64)
    hull_points = points[hull_source_indices]
    order = np.lexsort((hull_points[:, 2], hull_points[:, 1], hull_points[:, 0]))
    hull_source_indices = hull_source_indices[order]
    hull_points = points[hull_source_indices]
    source_to_output = {
        int(source_index): int(output_index)
        for output_index, source_index in enumerate(hull_source_indices)
    }
    center = hull_points.mean(axis=0)
    canonical_faces: set[tuple[int, int, int]] = set()
    for source_face in np.asarray(hull.simplices, dtype=np.int64):
        face = [source_to_output[int(index)] for index in source_face]
        first, second, third = hull_points[face]
        normal = np.cross(second - first, third - first)
        face_center = (first + second + third) / 3.0
        if float(np.dot(normal, face_center - center)) < 0.0:
            face[1], face[2] = face[2], face[1]
        minimum_position = face.index(min(face))
        face = face[minimum_position:] + face[:minimum_position]
        canonical_faces.add(tuple(int(value) for value in face))
    faces = np.asarray(sorted(canonical_faces), dtype=np.int64)
    if len(faces) < 4:
        raise ValueError("convex hull produced too few faces")
    edge_counts: dict[tuple[int, int], int] = {}
    for face in faces:
        for first, second in zip(face, np.roll(face, -1), strict=True):
            edge = tuple(sorted((int(first), int(second))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    if not edge_counts or set(edge_counts.values()) != {2}:
        raise ValueError("convex hull is not a closed two-manifold")
    return hull_points, faces


def _hull_volume_centroid(points: np.ndarray, faces: np.ndarray) -> tuple[float, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    interior = points.mean(axis=0)
    tetra_volumes: list[float] = []
    tetra_centroids: list[np.ndarray] = []
    for face in faces:
        first, second, third = points[face]
        volume = abs(
            float(
                np.dot(
                    first - interior,
                    np.cross(second - interior, third - interior),
                )
            )
        ) / 6.0
        if volume <= 0.0 or not np.isfinite(volume):
            raise ValueError("convex hull contains an invalid tetrahedral volume")
        tetra_volumes.append(volume)
        tetra_centroids.append((interior + first + second + third) / 4.0)
    volume_array = np.asarray(tetra_volumes, dtype=np.float64)
    volume = float(volume_array.sum())
    centroid = np.average(np.asarray(tetra_centroids), axis=0, weights=volume_array)
    if volume <= 0.0 or not np.isfinite(volume) or not np.all(np.isfinite(centroid)):
        raise ValueError("convex hull mass properties are not finite")
    return volume, centroid


def _point_to_triangles_min_distance(point: np.ndarray, triangles: np.ndarray) -> float:
    """Exact point-to-triangle distance, vectorized over candidate triangles."""

    point = np.asarray(point, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    first = triangles[:, 0]
    second = triangles[:, 1]
    third = triangles[:, 2]
    edge_ab = second - first
    edge_ac = third - first
    normals = np.cross(edge_ab, edge_ac)
    normal_sq = np.einsum("ij,ij->i", normals, normals)
    if np.any(normal_sq <= 1.0e-24):
        raise ValueError("protrusion audit received a degenerate visual triangle")
    signed_scale = np.einsum("ij,ij->i", point - first, normals) / normal_sq
    projected = point - signed_scale[:, None] * normals
    vector_ap = projected - first
    dot00 = np.einsum("ij,ij->i", edge_ab, edge_ab)
    dot01 = np.einsum("ij,ij->i", edge_ab, edge_ac)
    dot11 = np.einsum("ij,ij->i", edge_ac, edge_ac)
    dot20 = np.einsum("ij,ij->i", vector_ap, edge_ab)
    dot21 = np.einsum("ij,ij->i", vector_ap, edge_ac)
    denominator = dot00 * dot11 - dot01 * dot01
    if np.any(np.abs(denominator) <= 1.0e-24):
        raise ValueError("protrusion audit received a singular visual triangle")
    bary_b = (dot11 * dot20 - dot01 * dot21) / denominator
    bary_c = (dot00 * dot21 - dot01 * dot20) / denominator
    inside = (bary_b >= -1.0e-12) & (bary_c >= -1.0e-12) & (
        bary_b + bary_c <= 1.0 + 1.0e-12
    )
    plane_distance_sq = signed_scale * signed_scale * normal_sq

    def segment_distance_sq(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        edge = end - start
        edge_sq = np.einsum("ij,ij->i", edge, edge)
        parameter = np.einsum("ij,ij->i", point - start, edge) / edge_sq
        parameter = np.clip(parameter, 0.0, 1.0)
        closest = start + parameter[:, None] * edge
        delta = point - closest
        return np.einsum("ij,ij->i", delta, delta)

    edge_distance_sq = np.minimum.reduce(
        (
            segment_distance_sq(first, second),
            segment_distance_sq(second, third),
            segment_distance_sq(third, first),
        )
    )
    distances_sq = np.where(inside, plane_distance_sq, edge_distance_sq)
    return float(math.sqrt(max(0.0, float(distances_sq.min()))))


def _protrusion_audit(
    *,
    hull_points: np.ndarray,
    hull_faces: np.ndarray,
    component_visual_points: np.ndarray,
    component_visual_faces: np.ndarray,
    excluded_cap_source_y_m: Iterable[float],
) -> dict[str, Any]:
    visual_triangles = component_visual_points[component_visual_faces]
    hull_source_points = inverse_transform_points(hull_points)
    audit_samples: list[np.ndarray] = [point for point in hull_points]
    excluded_cap_values = tuple(float(value) for value in excluded_cap_source_y_m)
    excluded_cut_faces = 0
    external_face_count = 0
    for face in hull_faces:
        source_face_points = hull_source_points[face]
        if any(
            np.allclose(
                source_face_points[:, 1],
                excluded_cap,
                atol=1.0e-9,
                rtol=0.0,
            )
            for excluded_cap in excluded_cap_values
        ):
            excluded_cut_faces += 1
            continue
        external_face_count += 1
        first, second, third = hull_points[face]
        audit_samples.extend(
            (
                (first + second) * 0.5,
                (second + third) * 0.5,
                (third + first) * 0.5,
                (first + second + third) / 3.0,
            )
        )
    sample_array = np.unique(np.round(np.asarray(audit_samples), decimals=14), axis=0)
    distances = np.asarray(
        [
            _point_to_triangles_min_distance(point, visual_triangles)
            for point in sample_array
        ],
        dtype=np.float64,
    )
    worst_index = int(np.argmax(distances))
    max_gap = float(distances[worst_index])
    vertex_source_distances = np.asarray(
        [
            np.min(np.linalg.norm(component_visual_points - point, axis=1))
            for point in hull_points
        ],
        dtype=np.float64,
    )
    vertex_source_max = float(vertex_source_distances.max())
    passed = bool(
        np.all(np.isfinite(distances))
        and vertex_source_max <= 2.0e-8
        and max_gap <= PROTRUSION_SAMPLE_GAP_LIMIT_M
    )
    result = {
        "method": "hull_vertices_plus_external_face_edge_midpoints_and_centroids_to_exact_source_triangles",
        "source_derived_hull_vertices": bool(vertex_source_max <= 2.0e-8),
        "maximum_hull_vertex_to_source_vertex_distance_m": vertex_source_max,
        "sample_count": int(len(sample_array)),
        "external_hull_face_count": int(external_face_count),
        "excluded_internal_partition_cap_face_count": int(excluded_cut_faces),
        "excluded_partition_or_open_shell_cap_source_y_m": [
            float(value) for value in excluded_cap_values
        ],
        "maximum_sampled_surface_gap_m": max_gap,
        "maximum_gap_sample_m": [float(value) for value in sample_array[worst_index]],
        "acceptance_limit_m": float(PROTRUSION_SAMPLE_GAP_LIMIT_M),
        "pass": passed,
    }
    if not passed:
        raise ValueError(f"source-derived collider protrusion audit failed: {result}")
    return result


def build_colliders(
    mesh: SourceMesh, components: tuple[SourceComponent, ...]
) -> tuple[ColliderData, ...]:
    component_by_name = {component.name: component for component in components}
    handle = component_by_name["wood_handle"]
    ferrule = component_by_name["steel_ferrule"]
    shaft = component_by_name["steel_shaft_blade"]
    specifications = (
        (
            "TailMain",
            COLLIDER_PATHS[0],
            "wood",
            handle,
            handle.points[handle.points[:, 1] <= HANDLE_SPLIT_SOURCE_Y_M],
            HANDLE_SPLIT_SOURCE_Y_M,
            (HANDLE_SPLIT_SOURCE_Y_M,),
        ),
        (
            "HandleNeck",
            COLLIDER_PATHS[1],
            "wood",
            handle,
            handle.points[handle.points[:, 1] >= HANDLE_SPLIT_SOURCE_Y_M],
            HANDLE_SPLIT_SOURCE_Y_M,
            (HANDLE_SPLIT_SOURCE_Y_M,),
        ),
        (
            "Ferrule",
            COLLIDER_PATHS[2],
            "steel",
            ferrule,
            ferrule.points,
            None,
            (
                float(ferrule.points[:, 1].min()),
                float(ferrule.points[:, 1].max()),
            ),
        ),
        (
            "Stem",
            COLLIDER_PATHS[3],
            "steel",
            shaft,
            shaft.points[shaft.points[:, 1] <= BLADE_START_SOURCE_Y_M],
            BLADE_START_SOURCE_Y_M,
            (float(shaft.points[:, 1].min()), BLADE_START_SOURCE_Y_M),
        ),
        (
            "FlatBlade",
            COLLIDER_PATHS[4],
            "steel",
            shaft,
            shaft.points[shaft.points[:, 1] >= BLADE_START_SOURCE_Y_M],
            BLADE_START_SOURCE_Y_M,
            (BLADE_START_SOURCE_Y_M,),
        ),
    )
    colliders: list[ColliderData] = []
    for (
        name,
        path,
        material,
        component,
        selected_points,
        internal_cut,
        excluded_caps,
    ) in specifications:
        source_hull_points, faces = _canonical_convex_hull(selected_points)
        task_hull_points = transform_points(source_hull_points)
        volume, centroid = _hull_volume_centroid(task_hull_points, faces)
        audit = _protrusion_audit(
            hull_points=task_hull_points,
            hull_faces=faces,
            component_visual_points=transform_points(mesh.points),
            component_visual_faces=component.visual_faces,
            excluded_cap_source_y_m=excluded_caps,
        )
        colliders.append(
            ColliderData(
                name=name,
                path=path,
                material=material,
                source_component=component.name,
                source_points=source_hull_points,
                task_points=task_hull_points,
                faces=faces,
                source_y_range_m=(
                    float(selected_points[:, 1].min()),
                    float(selected_points[:, 1].max()),
                ),
                internal_cut_source_y_m=internal_cut,
                volume_m3=volume,
                centroid_m=centroid,
                protrusion_audit=audit,
            )
        )
    if tuple(collider.path for collider in colliders) != COLLIDER_PATHS:
        raise AssertionError("internal collider path order changed")
    return tuple(colliders)


def build_mass_model(colliders: tuple[ColliderData, ...]) -> dict[str, Any]:
    raw_weights = np.asarray(
        [
            collider.volume_m3
            * (
                WOOD_DENSITY_KG_M3
                if collider.material == "wood"
                else STEEL_DENSITY_KG_M3
            )
            for collider in colliders
        ],
        dtype=np.float64,
    )
    raw_total = float(raw_weights.sum())
    if raw_total <= 0.0 or not np.isfinite(raw_total):
        raise ValueError("mass proxy model has an invalid raw total")
    normalization_factor = NOMINAL_MASS_KG / raw_total
    masses = raw_weights * normalization_factor
    centroids = np.asarray([collider.centroid_m for collider in colliders])
    center_of_mass = np.average(centroids, axis=0, weights=masses)
    # The reviewed mesh is mirror-symmetric about task Z.  Qhull's coplanar
    # triangle order can leave a femtometre-scale residual; canonicalize that
    # mathematical zero so scipy patch releases cannot perturb the contract.
    center_of_mass[np.abs(center_of_mass) < 1.0e-12] = 0.0
    if not np.all(np.isfinite(center_of_mass)):
        raise ValueError("mass proxy model produced a non-finite center of mass")

    # A deterministic diagonal approximation is derived from each proxy's
    # AABB and combined with the parallel-axis theorem.  The COM itself uses
    # the exact convex-hull volumes above; only the inertia tensor drops the
    # small off-diagonal terms.
    diagonal_inertia = np.zeros(3, dtype=np.float64)
    piece_records: list[dict[str, Any]] = []
    for collider, raw_mass, mass in zip(
        colliders, raw_weights, masses, strict=True
    ):
        dimensions = np.ptp(collider.task_points, axis=0)
        local_box_inertia = mass / 12.0 * np.asarray(
            [
                dimensions[1] ** 2 + dimensions[2] ** 2,
                dimensions[0] ** 2 + dimensions[2] ** 2,
                dimensions[0] ** 2 + dimensions[1] ** 2,
            ],
            dtype=np.float64,
        )
        offset = collider.centroid_m - center_of_mass
        parallel_axis = mass * np.asarray(
            [
                offset[1] ** 2 + offset[2] ** 2,
                offset[0] ** 2 + offset[2] ** 2,
                offset[0] ** 2 + offset[1] ** 2,
            ],
            dtype=np.float64,
        )
        diagonal_inertia += local_box_inertia + parallel_axis
        density = (
            WOOD_DENSITY_KG_M3
            if collider.material == "wood"
            else STEEL_DENSITY_KG_M3
        )
        piece_records.append(
            {
                "name": collider.name,
                "path": collider.path,
                "material": collider.material,
                "density_kg_m3": float(density),
                "proxy_convex_hull_volume_m3": float(collider.volume_m3),
                "raw_density_mass_kg": float(raw_mass),
                "normalized_mass_kg": float(mass),
                "proxy_centroid_m": [float(value) for value in collider.centroid_m],
                "proxy_aabb_dimensions_m": [float(value) for value in dimensions],
            }
        )

    handle_points = np.vstack(
        [
            collider.task_points
            for collider in colliders
            if collider.material == "wood"
        ]
    )
    handle_bounds = _bounds(handle_points)
    support_contains_com = bool(
        np.all(center_of_mass >= handle_bounds[0] - 1.0e-12)
        and np.all(center_of_mass <= handle_bounds[1] + 1.0e-12)
    )
    if not support_contains_com:
        raise ValueError(
            "density-weighted center of mass is outside the wooden support region: "
            f"com={center_of_mass.tolist()}, bounds={handle_bounds.tolist()}"
        )
    if np.any(diagonal_inertia <= 0.0) or not np.all(np.isfinite(diagonal_inertia)):
        raise ValueError("mass proxy model produced invalid diagonal inertia")
    return {
        "method": "five_source_derived_convex_hull_density_proxies_normalized_to_nominal_mass",
        "overlap_policy": (
            "component proxy overlaps are retained explicitly; the resulting density "
            "weights are uniformly normalized to the measured nominal mass"
        ),
        "wood_density_kg_m3": float(WOOD_DENSITY_KG_M3),
        "steel_density_kg_m3": float(STEEL_DENSITY_KG_M3),
        "raw_density_mass_kg": raw_total,
        "normalization_factor": float(normalization_factor),
        "nominal_mass_kg": float(NOMINAL_MASS_KG),
        "center_of_mass_m": [float(value) for value in center_of_mass],
        "center_of_mass_finite": True,
        "support_region": {
            "definition": "union_aabb_of_wooden_tail_main_and_handle_neck_proxies",
            **_bounds_manifest(handle_points),
            "contains_center_of_mass": support_contains_com,
        },
        "inertia_model": "sum_of_piece_uniform_aabb_diagonals_plus_parallel_axis_terms",
        "diagonal_inertia_kg_m2": [float(value) for value in diagonal_inertia],
        "principal_axes_wxyz": [1.0, 0.0, 0.0, 0.0],
        "pieces": piece_records,
    }


def create_build_plan(archive_path: Path) -> BuildPlan:
    archive = validate_archive(archive_path)
    mesh = decode_source_mesh(archive)
    task_points = transform_points(mesh.points)
    if not np.allclose(_bounds(task_points), EXPECTED_TASK_BOUNDS_M, atol=2.0e-15, rtol=0.0):
        raise ValueError(f"normalized task bounds changed: {_bounds(task_points)!r}")
    task_normals = transform_normals(mesh.normals)
    components = source_components(mesh)
    colliders = build_colliders(mesh, components)
    mass_model = build_mass_model(colliders)
    return BuildPlan(
        archive=archive,
        mesh=mesh,
        task_points=task_points,
        task_normals=task_normals,
        components=components,
        colliders=colliders,
        mass_model=mass_model,
    )


def _vec3_array(values: np.ndarray, Gf: Any, Vt: Any) -> Any:
    return Vt.Vec3fArray(
        [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in values]
    )


def _vec2_array(values: np.ndarray, Gf: Any, Vt: Any) -> Any:
    return Vt.Vec2fArray([Gf.Vec2f(float(x), float(y)) for x, y in values])


def _define_preview_material(stage: Any, runtime_dir: Path, modules: dict[str, Any]) -> Any:
    Gf = modules["Gf"]
    Sdf = modules["Sdf"]
    UsdGeom = modules["UsdGeom"]
    UsdShade = modules["UsdShade"]
    del Gf, runtime_dir
    UsdGeom.Scope.Define(stage, ROOT_PATH + "/Looks")
    material = UsdShade.Material.Define(stage, MATERIAL_PATH)
    surface = UsdShade.Shader.Define(stage, MATERIAL_PATH + "/PreviewSurface")
    surface.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    st_reader = UsdShade.Shader.Define(stage, MATERIAL_PATH + "/PrimvarST")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    def texture_shader(
        shader_name: str, filename: str, *, color_space: str
    ) -> Any:
        shader = UsdShade.Shader.Define(stage, MATERIAL_PATH + "/" + shader_name)
        shader.CreateIdAttr("UsdUVTexture")
        shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("textures/" + filename)
        )
        shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st_reader.GetOutput("result")
        )
        shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
        shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateOutput("r", Sdf.ValueTypeNames.Float)
        shader.CreateOutput("g", Sdf.ValueTypeNames.Float)
        shader.CreateOutput("b", Sdf.ValueTypeNames.Float)
        shader.CreateOutput("a", Sdf.ValueTypeNames.Float)
        return shader

    diffuse = texture_shader(
        "DiffuseTexture", "flathead_screwdriver_diff_4k.jpg", color_space="sRGB"
    )
    arm = texture_shader(
        "ArmTexture", "flathead_screwdriver_arm_4k.jpg", color_space="raw"
    )
    normal = texture_shader(
        "NormalTexture", "flathead_screwdriver_nor_gl_4k.jpg", color_space="raw"
    )
    normal.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((2.0, 2.0, 2.0, 1.0))
    normal.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((-1.0, -1.0, -1.0, 0.0))
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        diffuse.GetOutput("rgb")
    )
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        arm.GetOutput("g")
    )
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(
        arm.GetOutput("b")
    )
    surface.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        normal.GetOutput("rgb")
    )
    surface.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    return material


def _define_mesh(
    stage: Any,
    path: str,
    points: np.ndarray,
    faces: np.ndarray,
    modules: dict[str, Any],
) -> Any:
    Gf = modules["Gf"]
    UsdGeom = modules["UsdGeom"]
    Vt = modules["Vt"]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr().Set(_vec3_array(points, Gf, Vt))
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr().Set(
        [int(index) for index in np.asarray(faces).reshape(-1)]
    )
    mesh.CreateExtentAttr().Set(_vec3_array(_bounds(points), Gf, Vt))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr().Set(UsdGeom.Tokens.rightHanded)
    return mesh


def author_usd(plan: BuildPlan, output_path: Path) -> dict[str, Any]:
    """Author the exact rigid/visual/collision topology after Isaac launches."""

    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    modules = {
        "Gf": Gf,
        "PhysxSchema": PhysxSchema,
        "Sdf": Sdf,
        "Usd": Usd,
        "UsdGeom": UsdGeom,
        "UsdPhysics": UsdPhysics,
        "UsdShade": UsdShade,
        "Vt": Vt,
    }
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    if stage is None:
        raise RuntimeError(f"failed to create USD stage: {output_path}")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, ROOT_PATH)
    root_prim = root.GetPrim()
    stage.SetDefaultPrim(root_prim)
    root_prim.SetCustomDataByKey("handson:assetId", ASSET_ID)
    root_prim.SetCustomDataByKey("handson:sourceArchiveSha256", ARCHIVE_SHA256)
    root_prim.SetCustomDataByKey("handson:sourceUrl", SOURCE_URL)
    root_prim.SetCustomDataByKey("handson:license", LICENSE_NAME)
    root_prim.SetCustomDataByKey("handson:tipAxis", "+X")
    root_prim.SetCustomDataByKey("handson:tailAxis", "-X")

    rigid_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rigid_api.CreateRigidBodyEnabledAttr().Set(True)
    rigid_api.CreateKinematicEnabledAttr().Set(False)
    rigid_api.CreateStartsAsleepAttr().Set(True)
    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.CreateMassAttr().Set(float(NOMINAL_MASS_KG))
    center_of_mass = plan.mass_model["center_of_mass_m"]
    mass_api.CreateCenterOfMassAttr().Set(
        Gf.Vec3f(*[float(value) for value in center_of_mass])
    )
    mass_api.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(
            *[
                float(value)
                for value in plan.mass_model["diagonal_inertia_kg_m2"]
            ]
        )
    )
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    contact_api = PhysxSchema.PhysxContactReportAPI.Apply(root_prim)
    contact_api.CreateThresholdAttr().Set(float(CONTACT_REPORT_THRESHOLD_N))

    UsdGeom.Scope.Define(stage, ROOT_PATH + "/Visual")
    material = _define_preview_material(stage, output_path.parent, modules)
    physics_material = UsdShade.Material.Define(stage, PHYSICS_MATERIAL_PATH)
    physics_material_api = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    physics_material_api.CreateStaticFrictionAttr().Set(float(DEFAULT_STATIC_FRICTION))
    physics_material_api.CreateDynamicFrictionAttr().Set(float(DEFAULT_DYNAMIC_FRICTION))
    physics_material_api.CreateRestitutionAttr().Set(float(DEFAULT_RESTITUTION))
    visual = _define_mesh(
        stage, VISUAL_PATH, plan.task_points, plan.mesh.faces, modules
    )
    visual.CreateNormalsAttr().Set(_vec3_array(plan.task_normals, Gf, Vt))
    visual.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    visual.CreateDoubleSidedAttr().Set(True)
    st_primvar = UsdGeom.PrimvarsAPI(visual).CreatePrimvar(
        "st", Sdf.ValueTypeNames.Float2Array, UsdGeom.Tokens.vertex
    )
    st_primvar.Set(_vec2_array(plan.mesh.uv, Gf, Vt))
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)
    visual.GetPrim().SetCustomDataByKey("handson:sourceDerived", True)
    visual.GetPrim().SetCustomDataByKey("handson:semanticRole", "visible_pbr_source_mesh")

    UsdGeom.Scope.Define(stage, ROOT_PATH + "/Colliders")
    for collider in plan.colliders:
        mesh = _define_mesh(
            stage, collider.path, collider.task_points, collider.faces, modules
        )
        prim = mesh.GetPrim()
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
            "convexHull"
        )
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            physics_material,
            UsdShade.Tokens.strongerThanDescendants,
            "physics",
        )
        prim.SetCustomDataByKey("handson:collisionOnly", True)
        prim.SetCustomDataByKey("handson:sourceDerived", True)
        prim.SetCustomDataByKey("handson:semanticRole", collider.name)
        prim.SetCustomDataByKey(
            "handson:sampledSurfaceGapM",
            float(collider.protrusion_audit["maximum_sampled_surface_gap_m"]),
        )

    stage.GetRootLayer().Save()
    expected_paths = {ROOT_PATH, VISUAL_PATH, *COLLIDER_PATHS}
    if not all(stage.GetPrimAtPath(path).IsValid() for path in expected_paths):
        raise RuntimeError("generated USD is missing a required prim path")
    if UsdPhysics.CollisionAPI(visual.GetPrim()):
        raise RuntimeError("visible source mesh unexpectedly owns CollisionAPI")
    rigid_owners = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    mass_owners = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.MassAPI)
    ]
    contact_owners = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
    ]
    collision_paths = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if rigid_owners != [ROOT_PATH] or mass_owners != [ROOT_PATH] or contact_owners != [ROOT_PATH]:
        raise RuntimeError(
            "generated USD must have exactly one rigid/mass/contact owner at root: "
            f"rigid={rigid_owners}, mass={mass_owners}, contact={contact_owners}"
        )
    if tuple(collision_paths) != COLLIDER_PATHS:
        raise RuntimeError(
            f"generated USD collision path set changed: {collision_paths!r}"
        )
    del stage
    return {
        "rigid_owner": ROOT_PATH,
        "visual_mesh_paths": [VISUAL_PATH],
        "collider_paths": list(COLLIDER_PATHS),
        "collider_count": len(COLLIDER_PATHS),
        "collision_approximation": "convexHull",
        "mass_kg": float(NOMINAL_MASS_KG),
        "center_of_mass_m": [float(value) for value in center_of_mass],
        "starts_asleep": True,
        "physics_material_path": PHYSICS_MATERIAL_PATH,
        "default_static_friction": float(DEFAULT_STATIC_FRICTION),
        "default_dynamic_friction": float(DEFAULT_DYNAMIC_FRICTION),
        "default_restitution": float(DEFAULT_RESTITUTION),
    }


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".building")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _collider_manifest(collider: ColliderData) -> dict[str, Any]:
    return {
        "name": collider.name,
        "path": collider.path,
        "rigid_owner": ROOT_PATH,
        "material_mass_role": collider.material,
        "source_component": collider.source_component,
        "source_y_range_m": [float(value) for value in collider.source_y_range_m],
        "internal_partition_cut_source_y_m": (
            float(collider.internal_cut_source_y_m)
            if collider.internal_cut_source_y_m is not None
            else None
        ),
        "source_hull_vertex_count": int(len(collider.source_points)),
        "triangle_count": int(len(collider.faces)),
        "task_local_bounds_m": _bounds_manifest(collider.task_points),
        "approximation": "convexHull",
        "visibility": "invisible",
        "collision_only": True,
        "source_derived": True,
        "convex_hull_volume_m3": float(collider.volume_m3),
        "convex_hull_centroid_m": [float(value) for value in collider.centroid_m],
        "protrusion_audit": collider.protrusion_audit,
    }


def build_manifest(
    plan: BuildPlan,
    runtime_path: Path,
    runtime_contract: dict[str, Any],
    isaac_version: str,
) -> dict[str, Any]:
    runtime_path = Path(runtime_path)
    runtime_sha = sha256_file(runtime_path)
    runtime_size = runtime_path.stat().st_size
    components = {
        component.name: {
            "unique_welded_vertex_count": int(len(component.points)),
            "source_bounds_m": _bounds_manifest(component.points),
            "task_local_bounds_m": _bounds_manifest(transform_points(component.points)),
            "source_triangle_count": int(len(component.visual_faces)),
        }
        for component in plan.components
    }
    collider_records = [_collider_manifest(collider) for collider in plan.colliders]
    task_bounds = _bounds(plan.task_points)
    handle_bounds = _bounds(
        next(
            component.points
            for component in plan.components
            if component.name == "wood_handle"
        )
    )
    handle_task_bounds = _bounds(transform_points(handle_bounds))
    # Transforming only the two AABB corners is valid here because source Y
    # maps to task X and source X maps to task -Y without mixing the extrema.
    semantic_ranges = {
        "task_local_tip_axis": "+X",
        "task_local_tail_axis": "-X",
        "wood_handle_local_x_m": [
            float(handle_task_bounds[:, 0].min()),
            float(handle_task_bounds[:, 0].max()),
        ],
        "tail_main_local_x_m": [
            float(_bounds(plan.colliders[0].task_points)[0, 0]),
            float(_bounds(plan.colliders[0].task_points)[1, 0]),
        ],
        "handle_neck_local_x_m": [
            float(_bounds(plan.colliders[1].task_points)[0, 0]),
            float(_bounds(plan.colliders[1].task_points)[1, 0]),
        ],
        "right_tail_grasp_recommended_local_x_m": [-0.12, -0.08],
        "left_middle_grasp_recommended_local_x_m": [-0.075, -0.025],
        "blade_start_local_x_m": float(
            UNIFORM_SCALE * BLADE_START_SOURCE_Y_M + TRANSLATION_M[0]
        ),
        "tip_point_local_m": [float(task_bounds[1, 0]), 0.025, 0.0],
        "tail_point_local_m": [float(task_bounds[0, 0]), 0.025, 0.0],
    }
    all_audits_pass = all(
        collider.protrusion_audit["pass"] for collider in plan.colliders
    )
    manifest = {
        "schema_version": "as019.polyhaven_asset.v1",
        "asset_id": ASSET_ID,
        "title": TITLE,
        "author": AUTHOR,
        "author_url": AUTHOR_URL,
        "source_url": SOURCE_URL,
        "license": LICENSE_NAME,
        "license_url": LICENSE_URL,
        "commercial_use": True,
        "provenance": {
            "archive_path": f"{ASSET_REPO_RELATIVE_DIR}/source/{ARCHIVE_FILENAME}",
            "archive_sha256": ARCHIVE_SHA256,
            "archive_size_bytes": int(ARCHIVE_SIZE_BYTES),
            "source_page": SOURCE_URL,
            "author_page": AUTHOR_URL,
            "poly_haven_license_page": POLY_HAVEN_LICENSE_URL,
            "license_deed": LICENSE_URL,
            "archive_retained_unchanged": True,
        },
        "source": {
            "archive_path": f"{ASSET_REPO_RELATIVE_DIR}/source/{ARCHIVE_FILENAME}",
            "archive_sha256": ARCHIVE_SHA256,
            "member_allowlist": list(MEMBER_PATHS),
            "members": list(plan.archive.member_records),
            "coordinate_system": "glTF 2.0 right-handed Y-up metres",
            "longitudinal_axis": "+Y tip / -Y wooden tail",
            "scene_count": 1,
            "node_count": 1,
            "mesh_count": 1,
            "primitive_count": 1,
            "material_count": 1,
            "vertex_count": int(len(plan.mesh.points)),
            "triangle_count": int(len(plan.mesh.faces)),
            "mesh_name": "Cylinder.005",
            "node_name": "flathead_screwdriver",
            "material_name": "flathead_screwdriver",
            "bounds_m": _bounds_manifest(plan.mesh.points),
            "components": components,
        },
        "normalization": {
            "formula": "p_task = uniform_scale * Rz(-90deg) * p_source + translation",
            "uniform_scale": float(UNIFORM_SCALE),
            "rotation_axis": "Z",
            "rotation_degrees": -90.0,
            "rotation_quaternion_wxyz": [
                float(value) for value in ROTATION_QUATERNION_WXYZ
            ],
            "rotation_matrix_row_major": [
                [float(value) for value in row]
                for row in ROTATION_SOURCE_TO_TASK
            ],
            "translation_m": [float(value) for value in TRANSLATION_M],
            "matrix_row_major": [
                [float(value) for value in row]
                for row in NORMALIZATION_MATRIX_ROW_MAJOR
            ],
            "orientation_preserving": True,
            "linear_determinant": float(UNIFORM_SCALE**3),
            "baked_into_authored_vertices": True,
        },
        "bounds_m": {
            "source": _bounds_manifest(plan.mesh.points),
            "task_local": _bounds_manifest(plan.task_points),
        },
        "semantic_ranges": semantic_ranges,
        "visual": {
            "mesh_paths": [VISUAL_PATH],
            "mesh_count": 1,
            "visible_mesh_count": 1,
            "material_path": MATERIAL_PATH,
            "pbr_workflow": "UsdPreviewSurface diffuse + glTF ARM(G roughness/B metallic) + OpenGL normal",
            "texture_paths": [
                f"{ASSET_REPO_RELATIVE_DIR}/runtime/textures/flathead_screwdriver_arm_4k.jpg",
                f"{ASSET_REPO_RELATIVE_DIR}/runtime/textures/flathead_screwdriver_diff_4k.jpg",
                f"{ASSET_REPO_RELATIVE_DIR}/runtime/textures/flathead_screwdriver_nor_gl_4k.jpg",
            ],
            "collision_api_authored": False,
            "source_derived": True,
        },
        "colliders": collider_records,
        "collision_summary": {
            "rigid_owner": ROOT_PATH,
            "paths": list(COLLIDER_PATHS),
            "count": int(len(COLLIDER_PATHS)),
            "approximation": "convexHull",
            "all_source_derived": True,
            "all_invisible": True,
            "all_collision_only": True,
            "visual_has_no_collision_api": True,
            "physics_material_path": PHYSICS_MATERIAL_PATH,
            "physics_material_binding_purpose": "physics",
            "physics_material_binding_strength": "strongerThanDescendants",
        },
        "physics_material": {
            "path": PHYSICS_MATERIAL_PATH,
            "material_api_count": 1,
            "bound_collider_paths": list(COLLIDER_PATHS),
            "binding_purpose": "physics",
            "binding_strength": "strongerThanDescendants",
            "static_friction": float(DEFAULT_STATIC_FRICTION),
            "dynamic_friction": float(DEFAULT_DYNAMIC_FRICTION),
            "restitution": float(DEFAULT_RESTITUTION),
            "episode_randomization_owner": "AS019 collection environment",
        },
        "rigid_body": {
            "owner": ROOT_PATH,
            "rigid_body_api_count": 1,
            "mass_api_count": 1,
            "contact_report_api_count": 1,
            "contact_report_threshold_n": float(CONTACT_REPORT_THRESHOLD_N),
            "starts_asleep": True,
            "mass_kg": float(NOMINAL_MASS_KG),
            "center_of_mass_m": list(plan.mass_model["center_of_mass_m"]),
            "mass_model": plan.mass_model,
        },
        "protrusion_audit": {
            "method": "per-collider exact source-triangle distance samples; artificial internal partition caps excluded",
            "acceptance_limit_m": float(PROTRUSION_SAMPLE_GAP_LIMIT_M),
            "all_pass": bool(all_audits_pass),
            "maximum_sampled_surface_gap_m": float(
                max(
                    collider.protrusion_audit["maximum_sampled_surface_gap_m"]
                    for collider in plan.colliders
                )
            ),
            "per_collider": {
                collider.name: collider.protrusion_audit
                for collider in plan.colliders
            },
        },
        "runtime": {
            "path": f"{ASSET_REPO_RELATIVE_DIR}/runtime/flathead_screwdriver.usd",
            "sha256": runtime_sha,
            "size_bytes": int(runtime_size),
            **runtime_contract,
            "isaac_sim_version": isaac_version,
            "stage_meters_per_unit": 1.0,
            "stage_up_axis": "Z",
            "default_prim": ROOT_PATH,
        },
    }
    return manifest


def build_asset(
    archive_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    isaac_version: str,
) -> dict[str, Any]:
    plan = create_build_plan(archive_path)
    output_path = Path(output_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    expected_runtime_name = "flathead_screwdriver.usd"
    if output_path.name != expected_runtime_name:
        raise ValueError(f"runtime output must be named {expected_runtime_name!r}")
    expected_manifest_name = "asset_manifest.json"
    if manifest_path.name != expected_manifest_name:
        raise ValueError(f"manifest output must be named {expected_manifest_name!r}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    building_output = output_path.with_name("." + output_path.name + ".building.usd")
    building_output.unlink(missing_ok=True)
    try:
        runtime_contract = author_usd(plan, building_output)
        # The texture bytes are copied only after every archive and geometry
        # gate passes.  Exact filenames keep the USD package self-contained.
        for member_path in TEXTURE_MEMBERS:
            destination = output_path.parent / member_path
            _atomic_write_bytes(destination, plan.archive.members[member_path])
        os.replace(building_output, output_path)
    finally:
        building_output.unlink(missing_ok=True)

    manifest = build_manifest(plan, output_path, runtime_contract, isaac_version)
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(manifest_path, manifest_bytes)
    return manifest


def parse_args() -> argparse.Namespace:
    task_dir = Path(__file__).resolve().parent
    asset_dir = task_dir / "assets/polyhaven_flathead_screwdriver"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=asset_dir / "source" / ARCHIVE_FILENAME,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=asset_dir / "runtime/flathead_screwdriver.usd",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=asset_dir / "asset_manifest.json",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="run all byte/geometry/mass audits without launching USD authoring",
    )
    return parser.parse_args()


def _require_isaac_sim_51() -> str:
    try:
        version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "asset USD must be built with the repository's Isaac Sim 5.1 Python"
        ) from exc
    if version != "5.1.0.0":
        raise RuntimeError(f"Isaac Sim 5.1.0.0 is required, found {version!r}")
    return version


def main() -> None:
    args = parse_args()
    if args.plan_only:
        plan = create_build_plan(args.archive)
        print(
            json.dumps(
                {
                    "archive_sha256": ARCHIVE_SHA256,
                    "collider_paths": list(COLLIDER_PATHS),
                    "mass_kg": NOMINAL_MASS_KG,
                    "center_of_mass_m": plan.mass_model["center_of_mass_m"],
                    "protrusion_audit_pass": all(
                        collider.protrusion_audit["pass"]
                        for collider in plan.colliders
                    ),
                },
                sort_keys=True,
            )
        )
        return
    isaac_version = _require_isaac_sim_51()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        manifest = build_asset(
            args.archive,
            args.output,
            args.manifest,
            isaac_version=isaac_version,
        )
        print(
            json.dumps(
                {
                    "built": str(Path(args.output).resolve()),
                    "manifest": str(Path(args.manifest).resolve()),
                    "runtime_sha256": manifest["runtime"]["sha256"],
                    "center_of_mass_m": manifest["runtime"]["center_of_mass_m"],
                    "collider_paths": manifest["runtime"]["collider_paths"],
                },
                sort_keys=True,
            )
        )
    finally:
        app.close()


if __name__ == "__main__":
    main()
