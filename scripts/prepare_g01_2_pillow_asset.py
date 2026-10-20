#!/usr/bin/env python3
"""Normalize the reviewed Pillow024 v2 source for G01 under Isaac Sim 5.1.

The supplier source is immutable and selected by an exact SHA-256.  This
builder restores the authored bind/rest state, removes editor/runtime residue,
re-authors the supplier face-varying bind normals byte-for-byte, and keeps the
supplier tetrahedral and material parameters unchanged.  The two in-process
builds must be byte-identical before the repository output is replaced.

Run only with ``HANDSON_ISAACSIM_PYTHON`` from the single Isaac Sim 5.1 /
IsaacLab v2.3.2 collection toolchain.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
import traceback
from typing import Any, Iterable, Sequence


SOURCE_USD = Path("/home/user/下载/Pillow024/Pillow024.usd")
EXPECTED_SOURCE_SHA256 = (
    "8b08e24ba0e8bc359f8802fce393fddb6234b31ca60a8e51a85404638cc80305"
)
EXPECTED_SOURCE_SIZE_BYTES = 19_832_135
INHERITED_SUPPLIER_ARCHIVE_NAME = "ZRYZ_LW_Assets_20260807.zip"
INHERITED_SUPPLIER_ARCHIVE_SHA256 = (
    "90798bdad84d8b1de38eb625984f7fefa4ae3b56c7297fb4769bd35151189306"
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ASSET_PARENT = (
    REPOSITORY_ROOT
    / "handson"
    / "handson"
    / "sim"
    / "envs"
    / "usd"
    / "g01_deformable_asset_inspection"
)
ASSET_DIRECTORY = ASSET_PARENT / "Pillow024_v2"
OUTPUT_USD_NAME = "Pillow024.usd"
MANIFEST_NAME = "asset_manifest.json"
README_NAME = "README.md"

ROOT_PATH = "/root/Pillow024"
VISUAL_PATH = "/root/Pillow024/Visuals/Pillow024"
SIMULATION_PATH = "/root/Pillow024/simulation_mesh"
SHADER_PATH = "/root/Looks/M_Pillow024_001/Shader"
PHYSICS_MATERIAL_PATH = "/root/Physics/PhysicsMaterial"

VISUAL_BIND_POINTS = "deformablePose:default:omniphysics:points"
SIMULATION_REST_POINTS = "omniphysics:restShapePoints"
SIMULATION_REST_TETS = "omniphysics:restTetVtxIndices"
UNUSED_COLOR_PRIMVAR = "primvars:______"

EXPECTED_VISUAL_POINT_COUNT = 5_212
EXPECTED_VISUAL_FACE_COUNT = 10_418
EXPECTED_VISUAL_CORNER_COUNT = 31_256
EXPECTED_SIMULATION_POINT_COUNT = 11_643
EXPECTED_TET_COUNT = 36_327
EXPECTED_SURFACE_FACE_COUNT = 20_822
EXPECTED_NONMANIFOLD_SURFACE_EDGE_COUNT = 39

EXPECTED_ROOT_MASS_KG = 0.8
EXPECTED_ROOT_PARAMETERS = {
    "physxDeformableBody:enableSpeculativeCCD": True,
    "physxDeformableBody:linearDamping": 0.1,
    "physxDeformableBody:selfCollision": True,
    "physxDeformableBody:solverPositionIterationCount": 32,
}
EXPECTED_SIMULATION_PARAMETERS = {
    "physxCollision:contactOffset": 0.008,
    "physxCollision:restOffset": 0.004,
}
EXPECTED_MATERIAL_PARAMETERS = {
    "omniphysics:density": 0.0,
    "omniphysics:dynamicFriction": 3.0,
    "omniphysics:poissonsRatio": 0.48,
    "omniphysics:staticFriction": 3.0,
    "omniphysics:youngsModulus": 1_000_000.0,
    "physxDeformableMaterial:elasticityDamping": 0.0,
}

SHARED_TEXTURES = {
    "inputs:diffuse_texture": (
        "../Pillow024/texture/T_Pillow024_BC001.png",
        "027587a43d19b53f00450757c5e5b9fd548570dd54baffa916c700448f3c3894",
    ),
    "inputs:normalmap_texture": (
        "../Pillow024/texture/T_Pillow024_N001.png",
        "9e6deb699524675b1dd455d72ea124b407f1515765808c09a4deebd35ad8edf6",
    ),
    "inputs:ORM_texture": (
        "../Pillow024/texture/T_Pillow024_ORM001.png",
        "b2ddc49cdd2c8dc1836194ca5389ad97e1911a50214d1e7c08dcaa5a75b59258",
    ),
}

EXPECTED_ROOT_APIS = {
    "PhysxAutoDeformableBodyAPI",
    "PhysxAutoDeformableMeshSimplificationAPI",
    "OmniPhysicsDeformableBodyAPI",
    "PhysxBaseDeformableBodyAPI",
    "MaterialBindingAPI",
}
EXPECTED_SIMULATION_APIS = {
    "OmniPhysicsVolumeDeformableSimAPI",
    "PhysicsCollisionAPI",
    "OmniPhysicsDeformablePoseAPI:default",
    "PhysxCollisionAPI",
    "MaterialBindingAPI",
}

README_TEXT = """# Pillow024 v2 normalized asset

This directory is a deterministic derivative of the reviewed supplier file
`/home/user/下载/Pillow024/Pillow024.usd` with SHA-256
`8b08e24ba0e8bc359f8802fce393fddb6234b31ca60a8e51a85404638cc80305`.
The download source is never modified or copied into this repository.
It did not include independent license metadata.  The source inherits the
repository's existing supplier-archive provenance record for
`ZRYZ_LW_Assets_20260807.zip` (SHA-256
`90798bdad84d8b1de38eb625984f7fefa4ae3b56c7297fb4769bd35151189306`),
but that provenance record is not a new license grant or authorization.

Build with the single collection runtime:

```bash
"$HANDSON_ISAACSIM_PYTHON" scripts/prepare_g01_2_pillow_asset.py
```

The builder requires Isaac Sim 5.1.0.0 and IsaacLab v2.3.2, builds twice,
requires byte-identical USD output, and validates the committed result again.
It performs only these asset-local normalizations:

- restores visual `points` from the deformable bind points and recomputes the
  visual extent;
- preserves and re-authors the supplier face-varying normals byte-for-byte,
  while checking them against a smooth bind-surface reference;
- restores simulation `points` from `restShapePoints`, zeroes velocities, and
  recomputes the simulation extent;
- removes the visual child `PhysicsMassAPI` / `physics:mass`, retaining the
  deformable root mass of 0.8 kg;
- removes the visual child physics-material binding while retaining the
  physics-material bindings on the deformable root and simulation TetMesh;
- explicitly hides the simulation TetMesh;
- removes `/Render` editor prims and the unused `primvars:______` data;
- reuses the byte-identical textures in `../Pillow024/texture/` rather than
  copying them.

The supplier TetMesh has 39 boundary-surface edges with incidence four.  This
known non-manifold risk is detected and recorded, but the builder deliberately
does not remesh or change the tetrahedral topology.  The preserved normals are
validated against the initial bind surface; live deformable rendering must
still be checked in Isaac Sim 5.1 to prove runtime normal updates after
deformation.

Companion repository changes wire this asset into the G01-2 scene,
environment, and task registration.  This builder itself remains asset-local
and does not edit those integration files.  No formal physical-replay claim is
made.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _verify_runtime() -> dict[str, str]:
    isaacsim_version = importlib.metadata.version("isaacsim")
    if isaacsim_version != "5.1.0.0":
        raise RuntimeError(
            "Pillow024 v2 preparation requires isaacsim 5.1.0.0; got "
            f"{isaacsim_version!r}"
        )

    import isaaclab

    module_path = Path(isaaclab.__file__).resolve()
    distribution_version = importlib.metadata.version("isaaclab")
    version_path = next(
        (
            parent / "VERSION"
            for parent in (module_path.parent, *module_path.parents)
            if (parent / "VERSION").is_file()
        ),
        None,
    )
    source_version = (
        version_path.read_text(encoding="utf-8").strip()
        if version_path is not None
        else None
    )
    if (
        distribution_version not in {"2.3.2", "2.3.2.post1"}
        and source_version != "2.3.2"
    ):
        raise RuntimeError(
            "Pillow024 v2 preparation requires IsaacLab v2.3.2; got "
            f"distribution={distribution_version!r}, "
            f"source={source_version!r}, module={str(module_path)!r}"
        )
    return {
        "isaacsim": isaacsim_version,
        "isaaclab": "2.3.2",
        "isaaclab_distribution": distribution_version,
        "isaaclab_module": str(module_path),
    }


def _verify_source() -> None:
    if not SOURCE_USD.is_file():
        raise FileNotFoundError(SOURCE_USD)
    size = SOURCE_USD.stat().st_size
    if size != EXPECTED_SOURCE_SIZE_BYTES:
        raise RuntimeError(
            f"Pillow024 source size mismatch: expected "
            f"{EXPECTED_SOURCE_SIZE_BYTES}, got {size}"
        )
    actual = sha256(SOURCE_USD)
    if actual != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "Pillow024 source SHA-256 differs from the reviewed v2 source: "
            f"{actual}"
        )


def _verify_shared_textures(asset_directory: Path) -> list[dict[str, object]]:
    dependencies: list[dict[str, object]] = []
    for attribute, (authored_path, expected_sha) in SHARED_TEXTURES.items():
        resolved = (asset_directory / authored_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"shared Pillow024 texture for {attribute} is missing: {resolved}"
            )
        actual_sha = sha256(resolved)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"shared Pillow024 texture hash mismatch for {resolved}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        dependencies.append(
            {
                "shader_attribute": attribute,
                "authored_path": authored_path,
                "resolved_repository_path": str(resolved.relative_to(REPOSITORY_ROOT)),
                "sha256": actual_sha,
                "size_bytes": resolved.stat().st_size,
                "copied_into_v2": False,
            }
        )
    return dependencies


def _extent(points: Sequence[Any]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not points:
        raise RuntimeError("cannot compute extent of an empty point array")
    minimum = [float(points[0][axis]) for axis in range(3)]
    maximum = minimum.copy()
    for point in points[1:]:
        for axis in range(3):
            value = float(point[axis])
            minimum[axis] = min(minimum[axis], value)
            maximum[axis] = max(maximum[axis], value)
    return tuple(minimum), tuple(maximum)


def _vec3_array_sha256(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(
            struct.pack(
                "<fff",
                float(value[0]),
                float(value[1]),
                float(value[2]),
            )
        )
    return digest.hexdigest()


def _newell_normal(points: Sequence[Any], vertex_ids: Sequence[int]) -> tuple[float, float, float]:
    nx = ny = nz = 0.0
    for offset, vertex_id in enumerate(vertex_ids):
        next_id = vertex_ids[(offset + 1) % len(vertex_ids)]
        current = points[vertex_id]
        following = points[next_id]
        x0, y0, z0 = map(float, current)
        x1, y1, z1 = map(float, following)
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    return nx, ny, nz


def _normalized(vector: Sequence[float], *, label: str) -> tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in vector))
    if not math.isfinite(length) or length <= 1.0e-20:
        raise RuntimeError(f"cannot normalize {label}: {tuple(vector)}")
    return tuple(component / length for component in vector)


def _recompute_face_varying_normals(
    points: Sequence[Any],
    face_counts: Sequence[int],
    face_indices: Sequence[int],
    *,
    Gf: Any,
    Vt: Any,
) -> Any:
    if sum(int(count) for count in face_counts) != len(face_indices):
        raise RuntimeError("visual face counts do not match face indices")
    accumulated = [[0.0, 0.0, 0.0] for _ in points]
    cursor = 0
    for face_index, count_raw in enumerate(face_counts):
        count = int(count_raw)
        if count < 3:
            raise RuntimeError(f"visual face {face_index} has fewer than 3 vertices")
        ids = [int(value) for value in face_indices[cursor : cursor + count]]
        cursor += count
        if len(set(ids)) != count:
            raise RuntimeError(f"visual face {face_index} repeats a vertex")
        normal = _newell_normal(points, ids)
        if sum(component * component for component in normal) <= 1.0e-30:
            raise RuntimeError(f"visual face {face_index} is degenerate")
        for vertex_id in ids:
            if not 0 <= vertex_id < len(points):
                raise RuntimeError(
                    f"visual face {face_index} index {vertex_id} is out of range"
                )
            for axis in range(3):
                accumulated[vertex_id][axis] += normal[axis]
    vertex_normals = [
        _normalized(value, label=f"visual vertex normal {index}")
        for index, value in enumerate(accumulated)
    ]
    return Vt.Vec3fArray(
        [Gf.Vec3f(*vertex_normals[int(index)]) for index in face_indices]
    )


def _float_equal(actual: Any, expected: Any, *, name: str) -> float:
    value = float(actual)
    target = float(expected)
    if not math.isclose(value, target, rel_tol=0.0, abs_tol=1.0e-6):
        raise RuntimeError(f"{name} changed: expected {target}, got {value}")
    return value


def _attribute_value(prim: Any, name: str) -> Any:
    attribute = prim.GetAttribute(name)
    if not attribute.IsValid() or not attribute.HasAuthoredValueOpinion():
        raise RuntimeError(f"required authored attribute is missing: {prim.GetPath()}.{name}")
    value = attribute.Get()
    if value is None:
        raise RuntimeError(f"required attribute has no value: {prim.GetPath()}.{name}")
    return value


def _api_names(prim: Any) -> set[str]:
    operation = prim.GetMetadata("apiSchemas")
    if operation is None:
        return set()
    if operation.isExplicit:
        return {str(value) for value in operation.explicitItems}
    return {str(value) for value in prim.GetAppliedSchemas()}


def _tet_signed_volume(points: Sequence[Any], tet: Sequence[int]) -> float:
    a, b, c, d = (points[int(index)] for index in tet)
    bax, bay, baz = (float(b[axis] - a[axis]) for axis in range(3))
    cax, cay, caz = (float(c[axis] - a[axis]) for axis in range(3))
    dax, day, daz = (float(d[axis] - a[axis]) for axis in range(3))
    cross_x = cay * daz - caz * day
    cross_y = caz * dax - cax * daz
    cross_z = cax * day - cay * dax
    return (bax * cross_x + bay * cross_y + baz * cross_z) / 6.0


def _normal_preservation(source: Sequence[Any], rebuilt: Sequence[Any]) -> dict[str, object]:
    if len(source) != len(rebuilt):
        raise RuntimeError("source and normalized normal counts differ")
    angles: list[float] = []
    maximum_absolute_component_delta = 0.0
    changed_vectors = 0
    for old, new in zip(source, rebuilt):
        old_tuple = tuple(float(old[axis]) for axis in range(3))
        new_tuple = tuple(float(new[axis]) for axis in range(3))
        if old_tuple != new_tuple:
            changed_vectors += 1
        maximum_absolute_component_delta = max(
            maximum_absolute_component_delta,
            *(abs(old_tuple[axis] - new_tuple[axis]) for axis in range(3)),
        )
        old_unit = _normalized(old_tuple, label="source normal")
        new_unit = _normalized(new_tuple, label="rebuilt normal")
        dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(old_unit, new_unit))))
        angles.append(math.degrees(math.acos(dot)))
    return {
        "source_array_sha256": _vec3_array_sha256(source),
        "normalized_array_sha256": _vec3_array_sha256(rebuilt),
        "preserved_byte_exact": changed_vectors == 0,
        "changed_vector_count": changed_vectors,
        "maximum_absolute_component_delta": maximum_absolute_component_delta,
        "mean_angular_delta_degrees": sum(angles) / len(angles),
        "maximum_angular_delta_degrees": max(angles),
    }


def _bind_smooth_normal_alignment(
    authored: Sequence[Any],
    smooth_reference: Sequence[Any],
) -> dict[str, float]:
    if len(authored) != len(smooth_reference):
        raise RuntimeError("authored and smooth-reference normal counts differ")
    dots: list[float] = []
    for authored_normal, reference_normal in zip(authored, smooth_reference):
        authored_unit = _normalized(
            tuple(float(authored_normal[axis]) for axis in range(3)),
            label="authored bind normal",
        )
        reference_unit = _normalized(
            tuple(float(reference_normal[axis]) for axis in range(3)),
            label="smooth bind reference normal",
        )
        dots.append(sum(a * b for a, b in zip(authored_unit, reference_unit)))
    if min(dots) <= 0.999999:
        raise RuntimeError(
            "preserved normals do not match the smooth bind-surface reference: "
            f"minimum dot={min(dots)}"
        )
    return {
        "minimum_corner_dot": min(dots),
        "mean_corner_dot": sum(dots) / len(dots),
        "maximum_corner_dot": max(dots),
    }


def _validate_tet_topology(
    points: Sequence[Any],
    tets: Sequence[Any],
    surface_faces: Sequence[Any],
) -> dict[str, object]:
    if len(points) != EXPECTED_SIMULATION_POINT_COUNT:
        raise RuntimeError(f"unexpected simulation point count: {len(points)}")
    if len(tets) != EXPECTED_TET_COUNT:
        raise RuntimeError(f"unexpected tetrahedron count: {len(tets)}")
    if len(surface_faces) != EXPECTED_SURFACE_FACE_COUNT:
        raise RuntimeError(f"unexpected simulation surface count: {len(surface_faces)}")

    minimum_volume = math.inf
    maximum_volume = 0.0
    total_volume = 0.0
    volume_face_counts: Counter[tuple[int, int, int]] = Counter()
    seen_tets: set[tuple[int, int, int, int]] = set()
    for tet_index, tet_raw in enumerate(tets):
        tet = tuple(int(value) for value in tet_raw)
        if len(set(tet)) != 4:
            raise RuntimeError(f"tet {tet_index} repeats a vertex")
        if min(tet) < 0 or max(tet) >= len(points):
            raise RuntimeError(f"tet {tet_index} index is out of range: {tet}")
        canonical = tuple(sorted(tet))
        if canonical in seen_tets:
            raise RuntimeError(f"duplicate tet topology: {canonical}")
        seen_tets.add(canonical)
        volume = _tet_signed_volume(points, tet)
        if not math.isfinite(volume) or volume <= 0.0:
            raise RuntimeError(
                f"rest tet {tet_index} is degenerate or inverted: volume={volume}"
            )
        minimum_volume = min(minimum_volume, volume)
        maximum_volume = max(maximum_volume, volume)
        total_volume += volume
        a, b, c, d = tet
        for face in ((b, c, d), (a, d, c), (a, b, d), (a, c, b)):
            volume_face_counts[tuple(sorted(face))] += 1

    if set(volume_face_counts.values()) != {1, 2}:
        raise RuntimeError(
            "tet face incidence is not a closed volume complex: "
            f"{dict(Counter(volume_face_counts.values()))}"
        )
    derived_boundary = {
        face for face, incidence in volume_face_counts.items() if incidence == 1
    }
    authored_boundary = {
        tuple(sorted(int(value) for value in face)) for face in surface_faces
    }
    if len(authored_boundary) != len(surface_faces):
        raise RuntimeError("simulation surface faces contain duplicates")
    if authored_boundary != derived_boundary:
        raise RuntimeError(
            "simulation surface faces do not match the tetrahedral boundary: "
            f"missing={len(derived_boundary - authored_boundary)}, "
            f"extra={len(authored_boundary - derived_boundary)}"
        )

    surface_edges: Counter[tuple[int, int]] = Counter()
    for face in surface_faces:
        a, b, c = (int(value) for value in face)
        for edge in ((a, b), (b, c), (c, a)):
            surface_edges[tuple(sorted(edge))] += 1
    edge_incidence = Counter(surface_edges.values())
    nonmanifold_edges = sum(
        count for incidence, count in edge_incidence.items() if incidence > 2
    )
    if edge_incidence != Counter({2: 31_155, 4: 39}):
        raise RuntimeError(
            "unexpected simulation boundary-edge incidence: "
            f"{dict(sorted(edge_incidence.items()))}"
        )
    if nonmanifold_edges != EXPECTED_NONMANIFOLD_SURFACE_EDGE_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_NONMANIFOLD_SURFACE_EDGE_COUNT} known "
            f"non-manifold surface edges, got {nonmanifold_edges}"
        )
    return {
        "rest_tets_all_positive": True,
        "minimum_signed_tet_volume_m3": minimum_volume,
        "maximum_signed_tet_volume_m3": maximum_volume,
        "total_tet_volume_m3": total_volume,
        "tet_face_incidence": {
            str(key): value for key, value in sorted(Counter(volume_face_counts.values()).items())
        },
        "surface_faces_match_derived_boundary": True,
        "surface_edge_incidence": {
            str(key): value for key, value in sorted(edge_incidence.items())
        },
        "nonmanifold_surface_edge_count": nonmanifold_edges,
        "topology_remeshed": False,
    }


def _validate_visual_topology(
    points: Sequence[Any], face_counts: Sequence[int], face_indices: Sequence[int]
) -> dict[str, object]:
    if len(points) != EXPECTED_VISUAL_POINT_COUNT:
        raise RuntimeError(f"unexpected visual point count: {len(points)}")
    if len(face_counts) != EXPECTED_VISUAL_FACE_COUNT:
        raise RuntimeError(f"unexpected visual face count: {len(face_counts)}")
    if len(face_indices) != EXPECTED_VISUAL_CORNER_COUNT:
        raise RuntimeError(f"unexpected visual corner count: {len(face_indices)}")
    distribution = Counter(int(value) for value in face_counts)
    if distribution != Counter({3: 10_416, 4: 2}):
        raise RuntimeError(f"unexpected visual face distribution: {distribution}")
    if min(face_indices) < 0 or max(face_indices) >= len(points):
        raise RuntimeError("visual face index is out of range")
    edges: Counter[tuple[int, int]] = Counter()
    cursor = 0
    seen_faces: set[tuple[int, ...]] = set()
    for face_index, count_raw in enumerate(face_counts):
        count = int(count_raw)
        face = tuple(int(value) for value in face_indices[cursor : cursor + count])
        cursor += count
        if len(set(face)) != count:
            raise RuntimeError(f"visual face {face_index} repeats a vertex")
        canonical = tuple(sorted(face))
        if canonical in seen_faces:
            raise RuntimeError(f"duplicate visual face: {canonical}")
        seen_faces.add(canonical)
        for start, end in zip(face, (*face[1:], face[0])):
            edges[tuple(sorted((start, end)))] += 1
    incidence = Counter(edges.values())
    if incidence != Counter({2: 15_628}):
        raise RuntimeError(f"visual mesh is not a closed 2-manifold: {incidence}")
    return {
        "face_vertex_count_distribution": {
            str(key): value for key, value in sorted(distribution.items())
        },
        "edge_incidence": {
            str(key): value for key, value in sorted(incidence.items())
        },
        "closed_two_manifold": True,
    }


def _normalize_stage(stage: Any, *, Gf: Any, Sdf: Any, UsdGeom: Any, UsdPhysics: Any, Vt: Any) -> None:
    if not stage.GetDefaultPrim().IsValid() or str(stage.GetDefaultPrim().GetPath()) != "/root":
        raise RuntimeError("Pillow024 source default prim must be /root")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise RuntimeError("Pillow024 source must be Z-up")
    if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise RuntimeError("Pillow024 source must use meters")

    root = stage.GetPrimAtPath(ROOT_PATH)
    visual = stage.GetPrimAtPath(VISUAL_PATH)
    simulation = stage.GetPrimAtPath(SIMULATION_PATH)
    shader = stage.GetPrimAtPath(SHADER_PATH)
    if not root.IsValid() or not visual.IsA(UsdGeom.Mesh):
        raise RuntimeError("Pillow024 source body or visual Mesh is missing")
    if not simulation.IsA(UsdGeom.TetMesh):
        raise RuntimeError("Pillow024 source simulation TetMesh is missing")
    if not shader.IsValid():
        raise RuntimeError("Pillow024 source OmniPBR shader is missing")

    bind_points = _attribute_value(visual, VISUAL_BIND_POINTS)
    face_counts = _attribute_value(visual, "faceVertexCounts")
    face_indices = _attribute_value(visual, "faceVertexIndices")
    _validate_visual_topology(bind_points, face_counts, face_indices)
    visual.GetAttribute("points").Set(bind_points)
    minimum, maximum = _extent(bind_points)
    visual.GetAttribute("extent").Set(
        Vt.Vec3fArray([Gf.Vec3f(*minimum), Gf.Vec3f(*maximum)])
    )
    # The supplier normals are already the smooth bind-pose normals.  Re-set
    # the exact array after restoring bind points so USD has an explicit,
    # self-consistent initial surface without changing seam lighting.
    source_normals = _attribute_value(visual, "normals")
    visual.GetAttribute("normals").Set(source_normals)
    UsdGeom.Mesh(visual).SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)

    mass_attribute = visual.GetAttribute("physics:mass")
    if mass_attribute.IsValid() and mass_attribute.HasAuthoredValueOpinion():
        mass_attribute.Clear()
    visual.RemoveProperty("physics:mass")
    if visual.HasAPI(UsdPhysics.MassAPI) and not visual.RemoveAPI(UsdPhysics.MassAPI):
        raise RuntimeError("could not remove visual child PhysicsMassAPI")
    if "PhysicsMassAPI" in _api_names(visual):
        raise RuntimeError("visual child still carries PhysicsMassAPI")
    visual.RemoveProperty("material:binding:physics")
    visual.RemoveProperty(UNUSED_COLOR_PRIMVAR)

    rest_points = _attribute_value(simulation, SIMULATION_REST_POINTS)
    rest_tets = _attribute_value(simulation, SIMULATION_REST_TETS)
    authored_tets = _attribute_value(simulation, "tetVertexIndices")
    if list(rest_tets) != list(authored_tets):
        raise RuntimeError("simulation rest tet indices differ from tetVertexIndices")
    _validate_tet_topology(
        rest_points,
        rest_tets,
        _attribute_value(simulation, "surfaceFaceVertexIndices"),
    )
    simulation.GetAttribute("points").Set(rest_points)
    simulation.GetAttribute("velocities").Set(
        Vt.Vec3fArray(
            [Gf.Vec3f(0.0, 0.0, 0.0) for _ in range(len(rest_points))]
        )
    )
    minimum, maximum = _extent(rest_points)
    simulation.GetAttribute("extent").Set(
        Vt.Vec3fArray([Gf.Vec3f(*minimum), Gf.Vec3f(*maximum)])
    )
    simulation.GetAttribute("visibility").Set(UsdGeom.Tokens.invisible)

    if stage.GetPrimAtPath("/Render").IsValid() and not stage.RemovePrim("/Render"):
        raise RuntimeError("could not remove /Render editor prims")
    for attribute_name, (authored_path, _expected_sha) in SHARED_TEXTURES.items():
        attribute = shader.GetAttribute(attribute_name)
        if not attribute.IsValid():
            raise RuntimeError(f"shader attribute is missing: {attribute_name}")
        attribute.Set(Sdf.AssetPath(authored_path))


def _validate_stage(
    path: Path,
    *,
    Gf: Any,
    Sdf: Any,
    Usd: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    Vt: Any,
) -> dict[str, object]:
    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"could not reopen normalized Pillow024: {path}")
    root = stage.GetPrimAtPath(ROOT_PATH)
    visual = stage.GetPrimAtPath(VISUAL_PATH)
    simulation = stage.GetPrimAtPath(SIMULATION_PATH)
    shader = stage.GetPrimAtPath(SHADER_PATH)
    material = stage.GetPrimAtPath(PHYSICS_MATERIAL_PATH)
    for prim, label in (
        (root, "root deformable"),
        (visual, "visual mesh"),
        (simulation, "simulation mesh"),
        (shader, "shader"),
        (material, "physics material"),
    ):
        if not prim.IsValid():
            raise RuntimeError(f"normalized Pillow024 is missing {label}")

    root_apis = _api_names(root)
    simulation_apis = _api_names(simulation)
    if root_apis != EXPECTED_ROOT_APIS:
        raise RuntimeError(f"root deformable APIs changed: {sorted(root_apis)}")
    if simulation_apis != EXPECTED_SIMULATION_APIS:
        raise RuntimeError(f"simulation APIs changed: {sorted(simulation_apis)}")

    root_mass = _float_equal(
        _attribute_value(root, "omniphysics:mass"),
        EXPECTED_ROOT_MASS_KG,
        name="root deformable mass",
    )
    root_parameters: dict[str, object] = {}
    for name, expected in EXPECTED_ROOT_PARAMETERS.items():
        actual = _attribute_value(root, name)
        if isinstance(expected, bool):
            if bool(actual) is not expected:
                raise RuntimeError(f"{name} changed: expected {expected}, got {actual}")
            root_parameters[name] = bool(actual)
        elif isinstance(expected, int):
            if int(actual) != expected:
                raise RuntimeError(f"{name} changed: expected {expected}, got {actual}")
            root_parameters[name] = int(actual)
        else:
            root_parameters[name] = _float_equal(actual, expected, name=name)

    if visual.HasAPI(UsdPhysics.MassAPI) or "PhysicsMassAPI" in _api_names(visual):
        raise RuntimeError("normalized visual child carries PhysicsMassAPI")
    visual_mass = visual.GetAttribute("physics:mass")
    if visual_mass.IsValid() and visual_mass.HasAuthoredValueOpinion():
        raise RuntimeError("normalized visual child carries authored physics:mass")
    visual_physics_binding = visual.GetRelationship("material:binding:physics")
    if visual_physics_binding.IsValid():
        raise RuntimeError(
            "normalized visual child retains material:binding:physics"
        )
    if visual.GetProperty(UNUSED_COLOR_PRIMVAR).IsValid():
        raise RuntimeError(f"normalized visual retains {UNUSED_COLOR_PRIMVAR}")
    if stage.GetPrimAtPath("/Render").IsValid():
        raise RuntimeError("normalized asset retains /Render editor prims")

    bind_points = _attribute_value(visual, VISUAL_BIND_POINTS)
    visual_points = _attribute_value(visual, "points")
    face_counts = _attribute_value(visual, "faceVertexCounts")
    face_indices = _attribute_value(visual, "faceVertexIndices")
    visual_topology = _validate_visual_topology(
        visual_points, face_counts, face_indices
    )
    if list(visual_points) != list(bind_points):
        raise RuntimeError("normalized visual points differ from bind points")
    expected_visual_extent = _extent(bind_points)
    actual_visual_extent = _attribute_value(visual, "extent")
    if tuple(map(float, actual_visual_extent[0])) != expected_visual_extent[0] or tuple(
        map(float, actual_visual_extent[1])
    ) != expected_visual_extent[1]:
        raise RuntimeError("normalized visual extent does not match bind points")

    source_stage = Usd.Stage.Open(str(SOURCE_USD), Usd.Stage.LoadAll)
    if source_stage is None:
        raise RuntimeError("could not reopen immutable Pillow024 source")
    source_normals = _attribute_value(
        source_stage.GetPrimAtPath(VISUAL_PATH), "normals"
    )
    rebuilt_normals = _attribute_value(visual, "normals")
    smooth_reference_normals = _recompute_face_varying_normals(
        bind_points,
        face_counts,
        face_indices,
        Gf=Gf,
        Vt=Vt,
    )
    if list(rebuilt_normals) != list(source_normals):
        raise RuntimeError("visual normals differ from the immutable source array")
    if visual.GetAttribute("normals").GetMetadata("interpolation") != "faceVarying":
        raise RuntimeError("visual normals are not authored faceVarying")
    normal_preservation = _normal_preservation(source_normals, rebuilt_normals)
    if not normal_preservation["preserved_byte_exact"]:
        raise RuntimeError("source face-varying normals were not preserved byte-for-byte")
    normal_alignment = _bind_smooth_normal_alignment(
        rebuilt_normals,
        smooth_reference_normals,
    )

    rest_points = _attribute_value(simulation, SIMULATION_REST_POINTS)
    simulation_points = _attribute_value(simulation, "points")
    rest_tets = _attribute_value(simulation, SIMULATION_REST_TETS)
    authored_tets = _attribute_value(simulation, "tetVertexIndices")
    surface_faces = _attribute_value(simulation, "surfaceFaceVertexIndices")
    if list(simulation_points) != list(rest_points):
        raise RuntimeError("normalized simulation points differ from restShapePoints")
    if list(authored_tets) != list(rest_tets):
        raise RuntimeError("normalized tet indices differ from restTetVtxIndices")
    velocities = _attribute_value(simulation, "velocities")
    if len(velocities) != len(rest_points) or any(
        float(value[axis]) != 0.0 for value in velocities for axis in range(3)
    ):
        raise RuntimeError("normalized simulation velocities are not all zero")
    expected_simulation_extent = _extent(rest_points)
    actual_simulation_extent = _attribute_value(simulation, "extent")
    if tuple(map(float, actual_simulation_extent[0])) != expected_simulation_extent[0] or tuple(
        map(float, actual_simulation_extent[1])
    ) != expected_simulation_extent[1]:
        raise RuntimeError("normalized simulation extent does not match rest points")
    if _attribute_value(simulation, "visibility") != UsdGeom.Tokens.invisible:
        raise RuntimeError("simulation TetMesh is not explicitly invisible")
    tet_topology = _validate_tet_topology(rest_points, rest_tets, surface_faces)

    simulation_parameters = {
        name: _float_equal(_attribute_value(simulation, name), expected, name=name)
        for name, expected in EXPECTED_SIMULATION_PARAMETERS.items()
    }
    material_parameters = {
        name: _float_equal(_attribute_value(material, name), expected, name=name)
        for name, expected in EXPECTED_MATERIAL_PARAMETERS.items()
    }
    expected_physics_material_target = "/root/Physics/PhysicsMaterial"
    physics_bindings: dict[str, str] = {}
    for binding_prim, label in ((root, "root"), (simulation, "simulation")):
        relationship = binding_prim.GetRelationship("material:binding:physics")
        targets = [str(value) for value in relationship.GetTargets()]
        if targets != [expected_physics_material_target]:
            raise RuntimeError(
                f"{label} physics-material binding changed: {targets}"
            )
        physics_bindings[label] = targets[0]

    texture_paths: dict[str, str] = {}
    for attribute_name, (authored_path, _expected_sha) in SHARED_TEXTURES.items():
        asset_path = _attribute_value(shader, attribute_name)
        if asset_path.path != authored_path:
            raise RuntimeError(
                f"{attribute_name} path changed: expected {authored_path!r}, "
                f"got {asset_path.path!r}"
            )
        expected_resolved = (path.parent / authored_path).resolve()
        if Path(asset_path.resolvedPath).resolve() != expected_resolved:
            raise RuntimeError(
                f"{attribute_name} resolved path mismatch: "
                f"expected {expected_resolved}, got {asset_path.resolvedPath!r}"
            )
        texture_paths[attribute_name] = asset_path.path

    return {
        "stage": {
            "default_prim": str(stage.GetDefaultPrim().GetPath()),
            "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        },
        "visual": {
            "point_count": len(visual_points),
            "face_count": len(face_counts),
            "corner_count": len(face_indices),
            "extent_m": [list(expected_visual_extent[0]), list(expected_visual_extent[1])],
            "points_equal_bind_points": True,
            "child_mass_api_removed": True,
            "child_physics_material_binding_removed": True,
            "unused_color_primvar_removed": True,
            "normal_interpolation": "faceVarying",
            "normal_preservation": normal_preservation,
            "bind_smooth_reference_alignment": normal_alignment,
            **visual_topology,
        },
        "simulation": {
            "point_count": len(simulation_points),
            "tet_count": len(rest_tets),
            "surface_face_count": len(surface_faces),
            "extent_m": [
                list(expected_simulation_extent[0]),
                list(expected_simulation_extent[1]),
            ],
            "points_equal_rest_shape_points": True,
            "velocities_all_zero": True,
            "explicit_visibility": "invisible",
            "parameters": simulation_parameters,
            **tet_topology,
        },
        "physics": {
            "root_mass_kg": root_mass,
            "root_parameters": root_parameters,
            "material_parameters": material_parameters,
            "physics_material_bindings": physics_bindings,
            "root_apis": sorted(root_apis),
            "simulation_apis": sorted(simulation_apis),
        },
        "material": {
            "shader_path": SHADER_PATH,
            "texture_paths": texture_paths,
        },
        "editor_cleanup": {
            "render_prim_removed": True,
        },
    }


def _build_one(
    destination: Path,
    *,
    Gf: Any,
    Sdf: Any,
    Usd: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    Vt: Any,
) -> dict[str, object]:
    shutil.copyfile(SOURCE_USD, destination)
    stage = Usd.Stage.Open(str(destination), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"could not open copied Pillow024 source: {destination}")
    _normalize_stage(
        stage,
        Gf=Gf,
        Sdf=Sdf,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
        Vt=Vt,
    )
    if not stage.GetRootLayer().Save():
        raise RuntimeError(f"could not save normalized Pillow024: {destination}")
    stage = None
    return _validate_stage(
        destination,
        Gf=Gf,
        Sdf=Sdf,
        Usd=Usd,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
        Vt=Vt,
    )


def _manifest(
    *,
    runtime: dict[str, str],
    output: Path,
    report: dict[str, object],
    dependencies: list[dict[str, object]],
    usd_version: Sequence[int],
) -> dict[str, object]:
    output_sha = sha256(output)
    readme_bytes = README_TEXT.encode("utf-8")
    return {
        "schema": "handson.g01.pillow024-v2.asset-manifest",
        "schema_version": 1,
        "source": {
            "path": str(SOURCE_USD),
            "sha256": EXPECTED_SOURCE_SHA256,
            "size_bytes": EXPECTED_SOURCE_SIZE_BYTES,
            "immutable": True,
            "copied_into_repository": False,
        },
        "provenance_and_license": {
            "inherited_supplier_archive_record": {
                "filename": INHERITED_SUPPLIER_ARCHIVE_NAME,
                "sha256": INHERITED_SUPPLIER_ARCHIVE_SHA256,
                "relationship": (
                    "Inherited from the repository's existing Pillow024 "
                    "supplier provenance record."
                ),
            },
            "independent_license_metadata_in_download": False,
            "license_name": None,
            "new_license_grant_or_authorization": False,
            "boundary": (
                "The inherited provenance record identifies source lineage; "
                "it does not create a new license grant or authorization."
            ),
        },
        "runtime": {
            **runtime,
            "pxr_usd_version": ".".join(str(value) for value in usd_version),
            "collection_runtime_family": "isaac_sim_5_1",
        },
        "output": {
            "path": OUTPUT_USD_NAME,
            "sha256": output_sha,
            "size_bytes": output.stat().st_size,
            "deterministic_double_build_byte_identical": True,
            "source_texture_files_copied": False,
        },
        "normalization": {
            "operations": [
                "visual_points_from_deformable_bind_points",
                "visual_extent_recomputed",
                "source_face_varying_bind_normals_reauthored_byte_exact",
                "simulation_points_from_rest_shape_points",
                "simulation_velocities_zeroed",
                "simulation_extent_recomputed",
                "visual_child_physics_mass_api_removed",
                "visual_child_physics_material_binding_removed",
                "simulation_tetmesh_explicitly_hidden",
                "render_editor_prims_removed",
                "unused_face_varying_color_primvar_removed",
                "textures_retargeted_to_shared_pillow024_directory",
            ],
            "tet_topology_changed": False,
            "material_parameters_changed": False,
            "root_mass_changed": False,
        },
        "validation": report,
        "dependencies": dependencies,
        "known_risks": [
            {
                "id": "simulation_boundary_nonmanifold_edges",
                "count": EXPECTED_NONMANIFOLD_SURFACE_EDGE_COUNT,
                "detail": (
                    "The supplier tetrahedral boundary has 39 surface edges "
                    "with incidence four; detected and retained without remeshing."
                ),
                "remeshed": False,
            },
            {
                "id": "live_deformation_normal_update_not_certified",
                "detail": (
                    "Initial supplier normals are preserved byte-for-byte and "
                    "verified against a smooth bind-surface reference, but "
                    "live runtime normal updates still require an Isaac Sim 5.1 smoke test."
                ),
            },
        ],
        "files": [
            {
                "path": OUTPUT_USD_NAME,
                "role": "normalized_runtime_usd",
                "sha256": output_sha,
                "size_bytes": output.stat().st_size,
            },
            {
                "path": README_NAME,
                "role": "asset_readme",
                "sha256": hashlib.sha256(readme_bytes).hexdigest(),
                "size_bytes": len(readme_bytes),
            },
        ],
        "repository_integration": {
            "g01_2_scene_wired": True,
            "g01_2_environment_wired": True,
            "g01_2_task_registration_wired": True,
            "authored_by_this_asset_builder": False,
            "formal_physical_replay_claimed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the committed normalized asset without rebuilding it",
    )
    args = parser.parse_args()

    runtime = _verify_runtime()
    _verify_source()
    dependencies = _verify_shared_textures(ASSET_DIRECTORY)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "enable_cameras": False})
    build_directories: list[Path] = []
    try:
        app.update()
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

        output = ASSET_DIRECTORY / OUTPUT_USD_NAME
        if args.check:
            if not output.is_file():
                raise FileNotFoundError(output)
            report = _validate_stage(
                output,
                Gf=Gf,
                Sdf=Sdf,
                Usd=Usd,
                UsdGeom=UsdGeom,
                UsdPhysics=UsdPhysics,
                Vt=Vt,
            )
            print(
                json.dumps(
                    {
                        "asset": str(output),
                        "sha256": sha256(output),
                        "validation": report,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0

        ASSET_PARENT.mkdir(parents=True, exist_ok=True)
        for label in ("a", "b"):
            directory = Path(
                tempfile.mkdtemp(
                    prefix=f".pillow024-v2-build-{label}-",
                    dir=ASSET_PARENT,
                )
            )
            build_directories.append(directory)
        first = build_directories[0] / OUTPUT_USD_NAME
        second = build_directories[1] / OUTPUT_USD_NAME
        first_report = _build_one(
            first,
            Gf=Gf,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            Vt=Vt,
        )
        second_report = _build_one(
            second,
            Gf=Gf,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            Vt=Vt,
        )
        first_sha = sha256(first)
        second_sha = sha256(second)
        if first_sha != second_sha or first.read_bytes() != second.read_bytes():
            raise RuntimeError(
                "Pillow024 normalization is not byte deterministic: "
                f"first={first_sha}, second={second_sha}"
            )
        if first_report != second_report:
            raise RuntimeError("Pillow024 deterministic validation reports differ")

        ASSET_DIRECTORY.mkdir(parents=True, exist_ok=True)
        first.replace(output)
        _write_atomic(ASSET_DIRECTORY / README_NAME, README_TEXT.encode("utf-8"))
        committed_report = _validate_stage(
            output,
            Gf=Gf,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            Vt=Vt,
        )
        if committed_report != first_report:
            raise RuntimeError("committed Pillow024 validation differs from staged build")
        manifest = _manifest(
            runtime=runtime,
            output=output,
            report=committed_report,
            dependencies=dependencies,
            usd_version=Usd.GetVersion(),
        )
        _write_atomic(ASSET_DIRECTORY / MANIFEST_NAME, _json_bytes(manifest))
        print(
            json.dumps(
                {
                    "asset": str(output),
                    "manifest": str(ASSET_DIRECTORY / MANIFEST_NAME),
                    "sha256": sha256(output),
                    "deterministic_double_build": True,
                    "nonmanifold_surface_edge_count": (
                        committed_report["simulation"][
                            "nonmanifold_surface_edge_count"
                        ]
                    ),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        for directory in build_directories:
            if directory.is_dir():
                shutil.rmtree(directory)
        app.close(wait_for_replicator=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
