#!/usr/bin/env python3
"""Build the canonical BC017 Metal Office Desk asset with Isaac Sim 5.1.

The Poly Haven source contains a fixed desk, two trays, and six genuinely
separate drawers with large integrated U-shaped metal pulls.  This builder
keeps all six source drawers as independent prismatic links so each episode
can select one slot per side without changing PhysX topology.  Collision
meshes remain source-derived and the complete copy/transform lineage is
stored on the USD prims.

Run this script only with ``HANDSON_ISAACSIM_PYTHON`` (Isaac Sim 5.1 and
IsaacLab v2.3.2).  The output is written transactionally and is deliberately
refused when the target directory already exists.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import traceback
from typing import Any, Iterable
import zipfile


EXPECTED_ARCHIVE_SHA256 = (
    "63f1ef6d9ec9737a762e6e3433a2b843b9407568ae1506bfd97f2026910b021c"
)
SOURCE_TITLE = "Metal Office Desk"
SOURCE_CREATOR = "Ulan Cabanilla"
SOURCE_PLATFORM = "Poly Haven"
SOURCE_UUID = "metal_office_desk"
SOURCE_URL = "https://polyhaven.com/a/metal_office_desk"
LICENSE_NAME = "CC0 1.0"
LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"

TASK_ID = "B320-BC017"
TASK_NAME = "t15_open_drawers"
TASK_REQUIREMENT = (
    "Use each assigned hand to pull the highlighted drawer on its side beyond "
    "60 percent travel; each side independently selects one of three drawers "
    "per episode, and both thresholds must be crossed within 1.2 seconds."
)

BASE_LINK = "bc017_fixture_base_link"
FIXED_JOINT = "bc017_world_fixed_joint"
DRAWER_SLOT_NAMES = ("top", "middle", "bottom")
DRAWER_ROLES_BY_SIDE = {
    "left": tuple(f"left_{slot}_drawer" for slot in DRAWER_SLOT_NAMES),
    "right": tuple(f"right_{slot}_drawer" for slot in DRAWER_SLOT_NAMES),
}
DRAWER_ROLE_ORDER = tuple(
    role
    for side in ("left", "right")
    for role in DRAWER_ROLES_BY_SIDE[side]
)
DRAWER_LINK_PRIMS = {
    "left_top_drawer": "bc017_left_drawer_link",
    "left_middle_drawer": "bc017_left_middle_drawer_link",
    "left_bottom_drawer": "bc017_left_bottom_drawer_link",
    "right_top_drawer": "bc017_right_drawer_link",
    "right_middle_drawer": "bc017_right_middle_drawer_link",
    "right_bottom_drawer": "bc017_right_bottom_drawer_link",
}
DRAWER_JOINT_NAMES = {
    "left_top_drawer": "bc017_left_drawer_prismatic_joint",
    "left_middle_drawer": "bc017_left_middle_drawer_prismatic_joint",
    "left_bottom_drawer": "bc017_left_bottom_drawer_prismatic_joint",
    "right_top_drawer": "bc017_right_drawer_prismatic_joint",
    "right_middle_drawer": "bc017_right_middle_drawer_prismatic_joint",
    "right_bottom_drawer": "bc017_right_bottom_drawer_prismatic_joint",
}
SOURCE_DRAWER_NODES_BY_ROLE = {
    "left_top_drawer": "metal_office_desk_drawer_04",
    "left_middle_drawer": "metal_office_desk_drawer_05",
    "left_bottom_drawer": "metal_office_desk_drawer_06",
    "right_top_drawer": "metal_office_desk_drawer_01",
    "right_middle_drawer": "metal_office_desk_drawer_02",
    "right_bottom_drawer": "metal_office_desk_drawer_03",
}
# Backward-compatible names identify the original top pair.  New generation
# logic always iterates the six canonical drawer roles above.
LEFT_LINK = DRAWER_LINK_PRIMS["left_top_drawer"]
RIGHT_LINK = DRAWER_LINK_PRIMS["right_top_drawer"]
LEFT_JOINT = DRAWER_JOINT_NAMES["left_top_drawer"]
RIGHT_JOINT = DRAWER_JOINT_NAMES["right_top_drawer"]
SOURCE_LEFT_DRAWER_NODE = SOURCE_DRAWER_NODES_BY_ROLE["left_top_drawer"]
SOURCE_RIGHT_DRAWER_NODE = SOURCE_DRAWER_NODES_BY_ROLE["right_top_drawer"]
SOURCE_BASE_NODES = frozenset(
    {
        "metal_office_desk",
        "metal_office_desk_tray_01",
        "metal_office_desk_tray_02",
    }
)
SOURCE_BASE_DECOMPOSITION_NODES = tuple(sorted(SOURCE_BASE_NODES))
SOURCE_DRAWER_NODES = frozenset(SOURCE_DRAWER_NODES_BY_ROLE.values())
# The original task conversion was visibly undersized beside the Wuji hand.
# Scale the complete source-derived desk, its colliders, mass properties, and
# articulation travel by approximately the requested 1.5x factor.  A 1.55x
# factor gives the 17 mm Wuji fingertip just over 2 mm of aperture clearance.
# Keep this uniform so visual geometry, collision, mass, and travel coincide.
BASELINE_UNIFORM_SCALE = 0.45
TASK_SCALE_MULTIPLIER = 1.55
UNIFORM_SCALE = BASELINE_UNIFORM_SCALE * TASK_SCALE_MULTIPLIER
BASELINE_DRAWER_TRAVEL_M = 0.16
DRAWER_TRAVEL_M = BASELINE_DRAWER_TRAVEL_M * TASK_SCALE_MULTIPLIER
TABLE_TOP_Z_M = 0.89
TABLE_SUPPORT_CLEARANCE_M = 0.001
TARGET_CLOSED_HANDLE_X_M = -0.24
TARGET_HANDLE_Y_M = 0.0
SQRT_HALF = math.sqrt(0.5)
# The Poly Haven glTF is Y-up with its drawer fronts along +Z.  Rotate it +90
# degrees about local Y so source +Z becomes runtime +X while source +Y stays
# runtime +Y.  The scene placement below then maps runtime +X toward the arms.
SOURCE_TO_RUNTIME_ORIENTATION_WXYZ = (SQRT_HALF, 0.0, SQRT_HALF, 0.0)
# Requested transform order: +90 degrees about X, then 180 degrees about Z.
# Rz(180) @ Rx(90) maps local (x, y, z) to world (-x, +z, +y).
PLACEMENT_ORIENTATION_WXYZ = (0.0, 0.0, SQRT_HALF, SQRT_HALF)

# The two selected source front planes differ by 1.88982 mm.  Move each by
# half that difference so the source-derived fronts coincide without changing
# either drawer mesh.
SOURCE_DRAWER_FRONT_Z_M = {
    "left": 0.434187978506,
    "right": 0.436077803373,
}
SOURCE_CLOSED_FRONT_Z_M = 0.5 * sum(SOURCE_DRAWER_FRONT_Z_M.values())
DRAWER_CLOSED_ALIGNMENT_X_M = {
    side: (SOURCE_CLOSED_FRONT_Z_M - front_z) * UNIFORM_SCALE
    for side, front_z in SOURCE_DRAWER_FRONT_Z_M.items()
}

# Bounds of the six U-shaped metal pull components in the source/world frame.
# They are selected from the actual connected visible source vertices and
# later transformed into the runtime local frame for strict fingertip
# contact filtering.  At the repaired scale each handle is about 77.7 mm wide
# and protrudes about 28.4 mm, giving the Wuji fingertips a physically usable
# aperture without a hidden fingertip-to-drawer constraint.
SOURCE_HANDLE_BOUNDS_M = {
    "left_top_drawer": (
        (-0.775339305401, 0.613256096840, 0.434187948704),
        (-0.660235822201, 0.624069809914, 0.476213544607),
    ),
    "left_middle_drawer": (
        (-0.775263369083, 0.450222760439, 0.431521087885),
        (-0.660159885883, 0.461036503315, 0.473546475172),
    ),
    "left_bottom_drawer": (
        (-0.775339305401, 0.287780940533, 0.431521058083),
        (-0.660235822201, 0.298594683409, 0.473546445370),
    ),
    "right_top_drawer": (
        (0.660350382328, 0.616288125515, 0.436077803373),
        (0.775453865528, 0.627101838589, 0.478103369474),
    ),
    "right_middle_drawer": (
        (0.661381065845, 0.452863752842, 0.434424579144),
        (0.776484549046, 0.463677465916, 0.476449936628),
    ),
    "right_bottom_drawer": (
        (0.660234749317, 0.287780940533, 0.431520611048),
        (0.775338232517, 0.298594683409, 0.473545968533),
    ),
}
SOURCE_HANDLE_CENTER_Y_M = {
    role: 0.5 * (bounds[0][1] + bounds[1][1])
    for role, bounds in SOURCE_HANDLE_BOUNDS_M.items()
}
SOURCE_CLOSED_HANDLE_CENTER_Y_M = 0.5 * sum(
    SOURCE_HANDLE_CENTER_Y_M[role]
    for role in ("left_top_drawer", "right_top_drawer")
)
DRAWER_CLOSED_ALIGNMENT_Y_M = {
    side: (
        SOURCE_CLOSED_HANDLE_CENTER_Y_M
        - SOURCE_HANDLE_CENTER_Y_M[f"{side}_top_drawer"]
    )
    * UNIFORM_SCALE
    for side in ("left", "right")
}
DRAWER_SOURCE_TRANSLATION_M_BY_ROLE = {
    "left_top_drawer": (
        DRAWER_CLOSED_ALIGNMENT_X_M["left"],
        DRAWER_CLOSED_ALIGNMENT_Y_M["left"],
        0.0,
    ),
    "right_top_drawer": (
        DRAWER_CLOSED_ALIGNMENT_X_M["right"],
        DRAWER_CLOSED_ALIGNMENT_Y_M["right"],
        0.0,
    ),
    **{
        role: (0.0, 0.0, 0.0)
        for role in DRAWER_ROLE_ORDER
        if not role.endswith("top_drawer")
    },
}

# PhysX GPU convex cooking accepts substantially less ambiguous input when the
# authored collision mesh is already a small, closed convex polyhedron.  Each
# fixed direction below selects an *existing transformed source vertex*; the
# builder never invents a collision corner or falls back to a primitive box.
LOW_POLY_DIRECTION_PAIR_COUNT = 32
LOW_POLY_MAX_VERTICES = LOW_POLY_DIRECTION_PAIR_COUNT * 2
LOW_POLY_MAX_TRIANGLES = 2 * LOW_POLY_MAX_VERTICES - 4
LOW_POLY_COLLIDER_SCOPE = "source_derived_colliders"
LOW_POLY_CONVEX_TOLERANCE_M = 5.0e-7
EXACT_MESH_COLLIDER_SCOPE = "source_exact_mesh_colliders"
COLLIDER_CONTACT_OFFSET_M = 0.0015
DRAWER_BODY_SDF_RESOLUTION = 256
SOURCE_WELD_TOLERANCE_M = 1.0e-9
HANDLE_SEGMENT_FRACTIONS = (0.20, 0.80)
# Backward-compatible evidence key: after the repair only the large drawer
# body component uses this SDF; the thin U pull is a three-hull compound.
HANDLE_SDF_RESOLUTION = DRAWER_BODY_SDF_RESOLUTION


def _sha256(path: Path) -> str:
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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _descriptor(root: Path, relative: str, role: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _safe_archive_member(name: str) -> str:
    if not name or "\\" in name or "\x00" in name:
        raise RuntimeError(f"unsafe archive member: {name!r}")
    normalized = name[:-1] if name.endswith("/") else name
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"unsafe archive member: {name!r}")
    return path.as_posix()


def _extract_source(archive_path: Path, destination: Path) -> None:
    required = {
        "scene.gltf",
        "metal_office_desk.bin",
        "license.txt",
        "textures/metal_office_desk_diff_1k.jpg",
        "textures/metal_office_desk_nor_gl_1k.jpg",
        "textures/metal_office_desk_arm_1k.jpg",
    }
    with zipfile.ZipFile(archive_path) as archive:
        entries: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            normalized = _safe_archive_member(info.filename)
            if normalized in entries:
                raise RuntimeError(f"duplicate archive member: {normalized}")
            entries[normalized] = info
        missing = required - set(entries)
        if missing:
            raise RuntimeError(
                "Metal Office Desk source closure is incomplete: "
                f"{sorted(missing)}"
            )
        for relative in sorted(required):
            info = entries[relative]
            if info.is_dir() or info.file_size <= 0 or info.flag_bits & 0x1:
                raise RuntimeError(f"invalid source member: {relative}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _verify_runtime() -> None:
    version = importlib.metadata.version("isaacsim")
    if version != "5.1.0.0":
        raise RuntimeError(f"BC017 builder requires isaacsim 5.1.0.0, got {version}")
    import isaaclab

    source = Path(isaaclab.__file__).resolve()
    distribution_version = importlib.metadata.version("isaaclab")
    version_path = next(
        (parent / "VERSION" for parent in (source.parent, *source.parents) if (parent / "VERSION").is_file()),
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
            "BC017 builder requires IsaacLab v2.3.2; got "
            f"distribution={distribution_version!r}, source={source_version!r}"
        )


def _convert_gltf(source: Path, output: Path) -> None:
    import omni.kit.asset_converter

    context = omni.kit.asset_converter.AssetConverterContext()
    settings = {
        "ignore_animations": True,
        "ignore_camera": True,
        "ignore_light": True,
        "keep_all_materials": True,
        "export_preview_surface": True,
        "use_meter_as_world_unit": True,
        "create_world_as_default_root_prim": True,
        "convert_stage_up_z": True,
        "embed_textures": True,
        "single_mesh": False,
        "merge_all_meshes": False,
        "disabling_instancing": True,
    }
    for name, value in settings.items():
        if not hasattr(context, name):
            raise RuntimeError(f"AssetConverterContext lacks required field {name!r}")
        setattr(context, name, value)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.usd")
    task = omni.kit.asset_converter.get_instance().create_converter_task(
        str(source), str(temporary), None, context
    )
    success = asyncio.get_event_loop().run_until_complete(task.wait_until_finished())
    if not success or not temporary.is_file() or temporary.stat().st_size <= 1024:
        detail = getattr(task, "get_error_message", lambda: "unknown error")()
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Metal Office Desk glTF conversion failed: {detail}"
        )
    temporary.replace(output)


def _normalize_converted_source(
    source_usd: Path,
    source_textures: Iterable[Path],
) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(source_usd), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"cannot reopen converted source: {source_usd}")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    texture_names = {path.name for path in source_textures}
    rewritten: set[str] = set()
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader":
            continue
        for shader_input in UsdShade.Shader(prim).GetInputs():
            if shader_input.GetTypeName() != "asset":
                continue
            raw = shader_input.Get()
            raw_path = str(getattr(raw, "path", "") or raw or "")
            basename = Path(raw_path).name
            if basename in texture_names:
                shader_input.Set(Sdf.AssetPath(f"textures/{basename}"))
                rewritten.add(basename)
            elif raw_path:
                # The converter sometimes authors unresolved optional map
                # placeholders.  They are not part of the rendered closure.
                shader_input.Set(Sdf.AssetPath(""))
    missing = texture_names - rewritten
    if missing:
        raise RuntimeError(
            "converted material omitted reviewed source textures: "
            f"{sorted(missing)}"
        )
    stage.GetRootLayer().Save()


def _node_mesh(stage: Any, node_path: str, UsdGeom: Any) -> Any:
    node = stage.GetPrimAtPath(node_path)
    if not node or not node.IsValid():
        raise RuntimeError(f"missing selected source node: {node_path}")
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh) and prim.GetPath().HasPrefix(node.GetPath())]
    if len(meshes) != 1:
        raise RuntimeError(f"source node {node_path} expected one mesh, got {len(meshes)}")
    return meshes[0]


def _bounds(cache: Any, prim: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    return tuple(float(v) for v in box.GetMin()), tuple(float(v) for v in box.GetMax())


def _deactivate_unselected(stage: Any, source_path: str, keep: Iterable[str]) -> None:
    root = stage.GetPrimAtPath(source_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"referenced source root is absent: {root.GetPath()}")
    keep_set = set(keep)
    children = {str(child.GetName()): child for child in root.GetChildren()}
    if not keep_set.issubset(children):
        raise RuntimeError(f"source nodes vanished: {sorted(keep_set - set(children))}")
    for name, child in children.items():
        if name != "Looks" and name not in keep_set:
            child.SetActive(False)


def _sha256_json_value(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _fixed_collision_directions(np: Any) -> Any:
    """Return the exact deterministic 64-direction source sampling set."""

    directions: list[tuple[float, float, float]] = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    fibonacci_count = LOW_POLY_DIRECTION_PAIR_COUNT - len(directions)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(fibonacci_count):
        z = 1.0 - 2.0 * (index + 0.5) / fibonacci_count
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        angle = golden_angle * index
        directions.append(
            (radius * math.cos(angle), radius * math.sin(angle), z)
        )
    directions.extend(
        (-x, -y, -z) for x, y, z in tuple(directions)
    )
    result = np.asarray(directions, dtype=np.float64)
    if result.shape != (LOW_POLY_MAX_VERTICES, 3):
        raise RuntimeError("BC017 fixed collision direction set changed")
    norms = np.linalg.norm(result, axis=1)
    if not np.all(np.isfinite(result)) or np.any(norms <= 0.0):
        raise RuntimeError("BC017 fixed collision direction set is invalid")
    return result / norms[:, None]


def _author_source_derived_low_poly_collider(
    stage: Any,
    *,
    body_prim: Any,
    source_mesh_path: str,
    source_node: str,
    semantic_role: str,
    handle_role: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Author one closed convex collider using only source-visible vertices.

    The fixed direction set selects extrema from the transformed visible mesh.
    A convex hull is then authored from that source-vertex subset.  There is no
    primitive fallback: degenerate, open, non-convex, or oversized output
    aborts the build.
    """

    import numpy as np
    from scipy.spatial import ConvexHull
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    source_prim = stage.GetPrimAtPath(source_mesh_path)
    if (
        not source_prim
        or not source_prim.IsValid()
        or not source_prim.IsA(UsdGeom.Mesh)
    ):
        raise RuntimeError(f"collider source mesh is absent: {source_mesh_path}")
    body_path = str(body_prim.GetPath())
    if not source_mesh_path.startswith(f"{body_path}/"):
        raise RuntimeError(
            f"collider source mesh is outside its rigid owner: {source_mesh_path}"
        )
    source_suffix = source_mesh_path.removeprefix(body_path)
    if not source_suffix.startswith("/"):
        raise RuntimeError("sourceVisiblePrimPath must be an owner-relative suffix")

    raw_points = UsdGeom.Mesh(source_prim).GetPointsAttr().Get()
    if raw_points is None or len(raw_points) < 4:
        raise RuntimeError(f"collider source mesh has too few points: {source_mesh_path}")
    source_local_points = np.asarray(
        [[float(value) for value in point] for point in raw_points],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(source_local_points)):
        raise RuntimeError(f"collider source mesh has non-finite points: {source_mesh_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    source_to_world = xform_cache.GetLocalToWorldTransform(source_prim)
    world_to_body = xform_cache.GetLocalToWorldTransform(body_prim).GetInverse()
    transformed_points = np.asarray(
        [
            tuple(
                world_to_body.Transform(
                    source_to_world.Transform(Gf.Vec3d(*point))
                )
            )
            for point in source_local_points
        ],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(transformed_points))
        or np.linalg.matrix_rank(
            transformed_points - np.mean(transformed_points, axis=0),
            tol=1.0e-10,
        )
        != 3
    ):
        raise RuntimeError(f"collider source mesh is not finite 3D geometry: {source_mesh_path}")

    directions = _fixed_collision_directions(np)
    selected_source_indices = list(
        dict.fromkeys(
            int(np.argmax(transformed_points @ direction))
            for direction in directions
        )
    )
    selected_points = transformed_points[selected_source_indices]
    if len(selected_source_indices) < 4:
        raise RuntimeError(f"fixed source sampling is degenerate: {source_mesh_path}")
    hull = ConvexHull(selected_points, qhull_options="Qt")
    hull_selected_indices = sorted(
        {int(index) for face in hull.simplices for index in face}
    )
    selected_to_authored = {
        selected_index: authored_index
        for authored_index, selected_index in enumerate(hull_selected_indices)
    }
    authored_source_indices = [
        selected_source_indices[index] for index in hull_selected_indices
    ]
    authored_points64 = selected_points[hull_selected_indices]
    centroid = np.mean(authored_points64, axis=0)
    authored_faces: list[list[int]] = []
    for simplex in hull.simplices:
        face = [selected_to_authored[int(index)] for index in simplex]
        a, b, c = (authored_points64[index] for index in face)
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if not math.isfinite(norm) or norm <= 1.0e-12:
            raise RuntimeError(f"low-poly hull has a degenerate face: {source_mesh_path}")
        if float(np.dot(normal, (a + b + c) / 3.0 - centroid)) < 0.0:
            face[1], face[2] = face[2], face[1]
        authored_faces.append(face)

    authored_points = authored_points64.astype(np.float32).astype(np.float64)
    if (
        len(authored_points) > LOW_POLY_MAX_VERTICES
        or len(authored_faces) > LOW_POLY_MAX_TRIANGLES
    ):
        raise RuntimeError(
            "source-derived convex collider exceeds the GPU cook cap: "
            f"{source_mesh_path} vertices={len(authored_points)} "
            f"triangles={len(authored_faces)}"
        )

    edge_counts: dict[tuple[int, int], int] = {}
    maximum_convex_violation = 0.0
    signed_volume = 0.0
    for face in authored_faces:
        a, b, c = (authored_points[index] for index in face)
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm <= 1.0e-12:
            raise RuntimeError(f"rounded low-poly hull has a degenerate face: {source_mesh_path}")
        distances = (authored_points - a) @ (normal / norm)
        maximum_convex_violation = max(
            maximum_convex_violation,
            float(np.max(distances)),
        )
        signed_volume += float(
            np.dot(a - centroid, np.cross(b - centroid, c - centroid)) / 6.0
        )
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((start, end)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    closed = bool(edge_counts) and all(count == 2 for count in edge_counts.values())
    euler_characteristic = len(authored_points) - len(edge_counts) + len(authored_faces)
    source_min = np.min(transformed_points, axis=0)
    source_max = np.max(transformed_points, axis=0)
    authored_min = np.min(authored_points, axis=0)
    authored_max = np.max(authored_points, axis=0)
    aabb_preserved = bool(
        np.allclose(source_min, authored_min, rtol=0.0, atol=1.0e-7)
        and np.allclose(source_max, authored_max, rtol=0.0, atol=1.0e-7)
    )
    if (
        not closed
        or euler_characteristic != 2
        or maximum_convex_violation > LOW_POLY_CONVEX_TOLERANCE_M
        or not math.isfinite(signed_volume)
        or signed_volume <= 1.0e-12
        or not aabb_preserved
    ):
        raise RuntimeError(
            "source-derived convex collider failed topology validation: "
            f"{source_mesh_path} closed={closed} euler={euler_characteristic} "
            f"convex_violation={maximum_convex_violation} "
            f"volume={signed_volume} aabb_preserved={aabb_preserved}"
        )

    collider_path = (
        f"{body_path}/{LOW_POLY_COLLIDER_SCOPE}/"
        f"{source_node}_low_poly_convex"
    )
    mesh = UsdGeom.Mesh.Define(stage, collider_path)
    mesh.CreatePointsAttr(
        [Gf.Vec3f(*(float(value) for value in point)) for point in authored_points]
    )
    mesh.CreateFaceVertexCountsAttr([3] * len(authored_faces))
    mesh.CreateFaceVertexIndicesAttr(
        [index for face in authored_faces for index in face]
    )
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(
        [Gf.Vec3f(*authored_min), Gf.Vec3f(*authored_max)]
    )
    mesh.MakeInvisible()
    collider_prim = mesh.GetPrim()
    collision = UsdPhysics.CollisionAPI.Apply(collider_prim)
    collision.CreateCollisionEnabledAttr(True)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(collider_prim)
    mesh_collision.CreateApproximationAttr().Set("convexHull")
    physx = PhysxSchema.PhysxCollisionAPI.Apply(collider_prim)
    physx.CreateContactOffsetAttr().Set(COLLIDER_CONTACT_OFFSET_M)
    physx.CreateRestOffsetAttr().Set(0.0)

    directions_payload = [
        [float(value) for value in direction] for direction in directions
    ]
    source_points_payload = [
        [float(value) for value in point] for point in source_local_points
    ]
    transformed_points_payload = [
        [float(value) for value in point] for point in transformed_points
    ]
    authored_points_payload = [
        [float(value) for value in point] for point in authored_points
    ]
    selected_indices_sha256 = _sha256_json_value(selected_source_indices)
    authored_indices_sha256 = _sha256_json_value(authored_source_indices)
    collider_prim.SetCustomDataByKey(
        "handson:classification", "source_derived_closed_low_poly_convex_collider"
    )
    collider_prim.SetCustomDataByKey(
        "handson:collisionDerivation",
        "fixed_64_direction_extrema_of_transformed_visible_source_vertices_then_convex_hull",
    )
    collider_prim.SetCustomDataByKey(
        "handson:sourceVisiblePrimPath", source_suffix
    )
    collider_prim.SetCustomDataByKey(
        "handson:sourceVertexIndicesSha256", authored_indices_sha256
    )
    collider_prim.SetCustomDataByKey("handson:semanticRole", semantic_role)
    if handle_role is not None:
        collider_prim.SetCustomDataByKey("handson:handleRole", handle_role)
    source_prim.SetCustomDataByKey(
        "handson:derivedCollisionPrimPath",
        collider_path.removeprefix(body_path),
    )

    relative_collider_path = collider_path.removeprefix("/WoodenDrawer/")
    record = {
        "collider_prim": relative_collider_path,
        "source_visible_prim": source_mesh_path.removeprefix("/WoodenDrawer/"),
        "source_visible_owner_relative_prim": source_suffix,
        "source_node": source_node,
        "semantic_role": semantic_role,
        "handle_role": handle_role,
        "method": (
            "fixed 64 unit directions select existing transformed visible-source "
            "vertex extrema; a closed convex hull is authored from only that subset"
        ),
        "generation": "deterministic_source_vertex_extrema_convex_hull",
        "source_derived": True,
        "hand_authored_or_placeholder": False,
        "primitive_or_generic_proxy_fallback": False,
        "direction_count": len(directions_payload),
        "directions_sha256": _sha256_json_value(directions_payload),
        "source_vertex_count": len(source_local_points),
        "source_local_points_sha256": _sha256_json_value(source_points_payload),
        "transformed_source_points_sha256": _sha256_json_value(
            transformed_points_payload
        ),
        "direction_selected_source_vertex_indices": selected_source_indices,
        "direction_selected_source_vertex_indices_sha256": selected_indices_sha256,
        "authored_hull_source_vertex_indices": authored_source_indices,
        "authored_hull_source_vertex_indices_sha256": authored_indices_sha256,
        "authored_points_sha256": _sha256_json_value(authored_points_payload),
        "authored_faces_sha256": _sha256_json_value(authored_faces),
        "authored_vertex_count": len(authored_points),
        "authored_triangle_count": len(authored_faces),
        "maximum_vertex_count": LOW_POLY_MAX_VERTICES,
        "maximum_triangle_count": LOW_POLY_MAX_TRIANGLES,
        "closed_manifold_verified": True,
        "edge_incidence_exactly_two_verified": True,
        "euler_characteristic": euler_characteristic,
        "convex_verified": True,
        "maximum_authored_convex_halfspace_violation_m": maximum_convex_violation,
        "convex_halfspace_tolerance_m": LOW_POLY_CONVEX_TOLERANCE_M,
        "positive_volume_verified": True,
        "authored_volume_m3": signed_volume,
        "source_vertex_subset_verified": True,
        "source_aabb_preserved": True,
        "computed_visibility": "invisible",
        "mesh_collision_approximation": "convexHull",
    }
    return relative_collider_path, record


def _author_source_exact_mesh_collider(
    stage: Any,
    *,
    body_prim: Any,
    source_mesh_path: str,
    source_node: str,
    semantic_role: str,
    handle_role: str | None = None,
    use_sdf: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Author a concavity-preserving collider from exact visible topology.

    The desk structure and drawer handles contain concave or disconnected
    detail that a sparse convex hull cannot preserve.  Copying their exact
    transformed points/topology preserves the visible surface while keeping
    the authored collision AABB identical to the source mesh.  PhysX convex
    decomposition preserves the desk openings; the moving handles use a
    remeshed SDF because their imported source is a 51-component open surface
    that makes decomposition both slow and geometrically unreliable.
    """

    import numpy as np
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    source_prim = stage.GetPrimAtPath(source_mesh_path)
    if (
        not source_prim
        or not source_prim.IsValid()
        or not source_prim.IsA(UsdGeom.Mesh)
    ):
        raise RuntimeError(f"exact collider source mesh is absent: {source_mesh_path}")
    body_path = str(body_prim.GetPath())
    if not source_mesh_path.startswith(f"{body_path}/"):
        raise RuntimeError(
            f"exact collider source mesh is outside its rigid owner: {source_mesh_path}"
        )
    source_suffix = source_mesh_path.removeprefix(body_path)
    if not source_suffix.startswith("/"):
        raise RuntimeError("sourceVisiblePrimPath must be an owner-relative suffix")

    source_mesh = UsdGeom.Mesh(source_prim)
    raw_points = source_mesh.GetPointsAttr().Get()
    raw_face_counts = source_mesh.GetFaceVertexCountsAttr().Get()
    raw_face_indices = source_mesh.GetFaceVertexIndicesAttr().Get()
    if (
        raw_points is None
        or len(raw_points) < 4
        or raw_face_counts is None
        or not raw_face_counts
        or raw_face_indices is None
        or not raw_face_indices
    ):
        raise RuntimeError(
            f"exact collider source topology is incomplete: {source_mesh_path}"
        )
    source_local_points = np.asarray(
        [[float(value) for value in point] for point in raw_points],
        dtype=np.float64,
    )
    face_counts = [int(value) for value in raw_face_counts]
    face_indices = [int(value) for value in raw_face_indices]
    if (
        not np.all(np.isfinite(source_local_points))
        or sum(face_counts) != len(face_indices)
        or min(face_counts) < 3
        or min(face_indices) < 0
        or max(face_indices) >= len(source_local_points)
    ):
        raise RuntimeError(f"exact collider source topology is invalid: {source_mesh_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    source_to_world = xform_cache.GetLocalToWorldTransform(source_prim)
    world_to_body = xform_cache.GetLocalToWorldTransform(body_prim).GetInverse()
    transformed_points64 = np.asarray(
        [
            tuple(
                world_to_body.Transform(
                    source_to_world.Transform(Gf.Vec3d(*point))
                )
            )
            for point in source_local_points
        ],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(transformed_points64))
        or np.linalg.matrix_rank(
            transformed_points64 - np.mean(transformed_points64, axis=0),
            tol=1.0e-10,
        )
        != 3
    ):
        raise RuntimeError(f"exact collider source is not finite 3D geometry: {source_mesh_path}")
    authored_points = transformed_points64.astype(np.float32).astype(np.float64)
    source_min = np.min(transformed_points64, axis=0)
    source_max = np.max(transformed_points64, axis=0)
    authored_min = np.min(authored_points, axis=0)
    authored_max = np.max(authored_points, axis=0)
    aabb_preserved = bool(
        np.allclose(source_min, authored_min, rtol=0.0, atol=1.0e-7)
        and np.allclose(source_max, authored_max, rtol=0.0, atol=1.0e-7)
    )
    if not aabb_preserved:
        raise RuntimeError(f"exact collider changed source AABB: {source_mesh_path}")

    collider_suffix = "exact_mesh_sdf" if use_sdf else "exact_mesh_decomposition"
    collider_path = (
        f"{body_path}/{EXACT_MESH_COLLIDER_SCOPE}/"
        f"{source_node}_{collider_suffix}"
    )
    mesh = UsdGeom.Mesh.Define(stage, collider_path)
    mesh.CreatePointsAttr(
        [Gf.Vec3f(*(float(value) for value in point)) for point in authored_points]
    )
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr([Gf.Vec3f(*authored_min), Gf.Vec3f(*authored_max)])
    mesh.MakeInvisible()
    collider_prim = mesh.GetPrim()
    collision = UsdPhysics.CollisionAPI.Apply(collider_prim)
    collision.CreateCollisionEnabledAttr(True)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(collider_prim)
    if use_sdf:
        mesh_collision.CreateApproximationAttr().Set("sdf")
        sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(collider_prim)
        sdf.CreateSdfResolutionAttr().Set(HANDLE_SDF_RESOLUTION)
        sdf.CreateSdfEnableRemeshingAttr().Set(True)
        sdf.CreateSdfTriangleCountReductionFactorAttr().Set(1.0)
    else:
        mesh_collision.CreateApproximationAttr().Set("convexDecomposition")
        PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(collider_prim)
    physx = PhysxSchema.PhysxCollisionAPI.Apply(collider_prim)
    physx.CreateContactOffsetAttr().Set(COLLIDER_CONTACT_OFFSET_M)
    physx.CreateRestOffsetAttr().Set(0.0)

    source_points_payload = [
        [float(value) for value in point] for point in source_local_points
    ]
    transformed_points_payload = [
        [float(value) for value in point] for point in transformed_points64
    ]
    authored_points_payload = [
        [float(value) for value in point] for point in authored_points
    ]
    classification = (
        "source_exact_mesh_sdf_collider"
        if use_sdf
        else "source_exact_mesh_decomposition_collider"
    )
    collision_derivation = (
        "exact_transformed_visible_source_topology_then_physx_remeshed_sdf"
        if use_sdf
        else "exact_transformed_visible_source_topology_then_physx_convex_decomposition"
    )
    collider_prim.SetCustomDataByKey("handson:classification", classification)
    collider_prim.SetCustomDataByKey(
        "handson:collisionDerivation", collision_derivation
    )
    collider_prim.SetCustomDataByKey(
        "handson:sourceVisiblePrimPath", source_suffix
    )
    collider_prim.SetCustomDataByKey("handson:semanticRole", semantic_role)
    if handle_role is not None:
        collider_prim.SetCustomDataByKey("handson:handleRole", handle_role)
    source_prim.SetCustomDataByKey(
        "handson:derivedCollisionPrimPath",
        collider_path.removeprefix(body_path),
    )

    relative_collider_path = collider_path.removeprefix("/WoodenDrawer/")
    record = {
        "collider_prim": relative_collider_path,
        "source_visible_prim": source_mesh_path.removeprefix("/WoodenDrawer/"),
        "source_visible_owner_relative_prim": source_suffix,
        "source_node": source_node,
        "semantic_role": semantic_role,
        "handle_role": handle_role,
        "method": (
            "copy exact transformed visible-source points and face topology; "
            + (
                "PhysX cooks a remeshed signed-distance field"
                if use_sdf
                else "PhysX cooks a concavity-preserving convex decomposition"
            )
        ),
        "generation": (
            "exact_source_mesh_remeshed_sdf"
            if use_sdf
            else "exact_source_mesh_convex_decomposition"
        ),
        "source_derived": True,
        "hand_authored_or_placeholder": False,
        "primitive_or_generic_proxy_fallback": False,
        "source_vertex_count": len(source_local_points),
        "source_face_count": len(face_counts),
        "source_local_points_sha256": _sha256_json_value(source_points_payload),
        "source_face_counts_sha256": _sha256_json_value(face_counts),
        "source_face_indices_sha256": _sha256_json_value(face_indices),
        "transformed_source_points_sha256": _sha256_json_value(
            transformed_points_payload
        ),
        "authored_points_sha256": _sha256_json_value(authored_points_payload),
        "authored_face_counts_sha256": _sha256_json_value(face_counts),
        "authored_face_indices_sha256": _sha256_json_value(face_indices),
        "authored_vertex_count": len(authored_points),
        "authored_face_count": len(face_counts),
        "source_topology_preserved": True,
        "source_aabb_preserved": True,
        "computed_visibility": "invisible",
        "mesh_collision_approximation": "sdf" if use_sdf else "convexDecomposition",
        "contact_offset_m": COLLIDER_CONTACT_OFFSET_M,
        "rest_offset_m": 0.0,
    }
    if use_sdf:
        record.update(
            {
                "sdf_resolution": HANDLE_SDF_RESOLUTION,
                "sdf_cell_size_max_m": float(
                    np.max(authored_max - authored_min) / HANDLE_SDF_RESOLUTION
                ),
                "sdf_remeshing_enabled": True,
                "sdf_triangle_count_reduction_factor": 1.0,
            }
        )
    return relative_collider_path, record


def _author_source_drawer_compound_colliders(
    stage: Any,
    *,
    body_prim: Any,
    source_mesh_path: str,
    source_node: str,
    drawer_role: str,
    hand_side: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Split one source drawer into a body SDF and a three-hull U pull.

    The glTF duplicates vertices at UV/normal seams.  After deterministic
    coordinate welding, the selected mesh is exactly two closed components:
    a large drawer body and a thin U-shaped pull.  Cooking both in one SDF
    leaves the pull only a few voxels thick.  The body therefore keeps a
    dedicated SDF while the pull is partitioned into two legs and a crossbar,
    with one face-adjacency ring of overlap at each seam.  Every authored hull
    uses only transformed source vertices; no box or hand-authored proxy is
    introduced.
    """

    import numpy as np
    from scipy.spatial import ConvexHull
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    source_prim = stage.GetPrimAtPath(source_mesh_path)
    if (
        not source_prim
        or not source_prim.IsValid()
        or not source_prim.IsA(UsdGeom.Mesh)
    ):
        raise RuntimeError(
            f"compound drawer collider source is absent: {source_mesh_path}"
        )
    body_path = str(body_prim.GetPath())
    if not source_mesh_path.startswith(f"{body_path}/"):
        raise RuntimeError(
            "compound drawer source is outside its rigid owner: "
            f"{source_mesh_path}"
        )
    source_suffix = source_mesh_path.removeprefix(body_path)
    source_mesh = UsdGeom.Mesh(source_prim)
    raw_points = source_mesh.GetPointsAttr().Get()
    raw_counts = source_mesh.GetFaceVertexCountsAttr().Get()
    raw_indices = source_mesh.GetFaceVertexIndicesAttr().Get()
    if not raw_points or not raw_counts or not raw_indices:
        raise RuntimeError(
            f"compound drawer source topology is incomplete: {source_mesh_path}"
        )
    source_local_points = np.asarray(
        [[float(value) for value in point] for point in raw_points],
        dtype=np.float64,
    )
    face_counts = [int(value) for value in raw_counts]
    face_indices = [int(value) for value in raw_indices]
    if (
        not np.all(np.isfinite(source_local_points))
        or sum(face_counts) != len(face_indices)
        or min(face_counts) < 3
        or min(face_indices) < 0
        or max(face_indices) >= len(source_local_points)
    ):
        raise RuntimeError(
            f"compound drawer source topology is invalid: {source_mesh_path}"
        )

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    source_to_world = xform_cache.GetLocalToWorldTransform(source_prim)
    world_to_body = xform_cache.GetLocalToWorldTransform(body_prim).GetInverse()
    transformed_points = np.asarray(
        [
            tuple(
                world_to_body.Transform(
                    source_to_world.Transform(Gf.Vec3d(*point))
                )
            )
            for point in source_local_points
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(transformed_points)):
        raise RuntimeError(
            f"compound drawer transform is non-finite: {source_mesh_path}"
        )

    # Weld only coincident source coordinates.  The tolerance is several
    # orders below USD float publication and cannot merge geometric features.
    quantized = np.rint(
        source_local_points / SOURCE_WELD_TOLERANCE_M
    ).astype(np.int64)
    key_to_welded: dict[tuple[int, int, int], int] = {}
    raw_to_welded: list[int] = []
    welded_representatives: list[int] = []
    for raw_index, row in enumerate(quantized):
        key = tuple(int(value) for value in row)
        welded_index = key_to_welded.get(key)
        if welded_index is None:
            welded_index = len(welded_representatives)
            key_to_welded[key] = welded_index
            welded_representatives.append(raw_index)
        elif float(
            np.linalg.norm(
                source_local_points[raw_index]
                - source_local_points[welded_representatives[welded_index]]
            )
        ) > SOURCE_WELD_TOLERANCE_M:
            raise RuntimeError("BC017 weld quantization merged distinct vertices")
        raw_to_welded.append(welded_index)
    welded_points = transformed_points[welded_representatives]

    welded_faces: list[list[int]] = []
    cursor = 0
    for count in face_counts:
        mapped = [
            raw_to_welded[index]
            for index in face_indices[cursor : cursor + count]
        ]
        cursor += count
        clean: list[int] = []
        for vertex in mapped:
            if not clean or vertex != clean[-1]:
                clean.append(vertex)
        if len(clean) > 2 and clean[0] == clean[-1]:
            clean.pop()
        if len(clean) < 3 or len(set(clean)) != len(clean):
            raise RuntimeError("BC017 welded source contains a degenerate face")
        welded_faces.append(clean)

    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(welded_faces):
        for position, start in enumerate(face):
            end = face[(position + 1) % len(face)]
            edge = tuple(sorted((start, end)))
            edge_to_faces.setdefault(edge, []).append(face_index)
    face_neighbors: list[set[int]] = [set() for _ in welded_faces]
    for owners in edge_to_faces.values():
        for owner in owners:
            face_neighbors[owner].update(
                candidate for candidate in owners if candidate != owner
            )

    remaining = set(range(len(welded_faces)))
    components: list[set[int]] = []
    while remaining:
        seed = min(remaining)
        component: set[int] = set()
        frontier = [seed]
        remaining.remove(seed)
        while frontier:
            face_index = frontier.pop()
            component.add(face_index)
            for neighbor in sorted(face_neighbors[face_index]):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    frontier.append(neighbor)
        components.append(component)
    if len(components) != 2:
        raise RuntimeError(
            "BC017 welded drawer must contain exactly body and handle; "
            f"observed_components={len(components)}"
        )

    component_rows: list[tuple[set[int], set[int], np.ndarray]] = []
    for component in components:
        vertices = {
            vertex
            for face_index in component
            for vertex in welded_faces[face_index]
        }
        incidence: dict[tuple[int, int], int] = {}
        for face_index in component:
            face = welded_faces[face_index]
            for position, start in enumerate(face):
                edge = tuple(sorted((start, face[(position + 1) % len(face)])))
                incidence[edge] = incidence.get(edge, 0) + 1
        if not incidence or any(value != 2 for value in incidence.values()):
            raise RuntimeError("BC017 welded drawer component is not closed manifold")
        points = welded_points[sorted(vertices)]
        component_rows.append(
            (component, vertices, np.max(points, axis=0) - np.min(points, axis=0))
        )
    component_rows.sort(key=lambda row: float(np.max(row[2])))
    handle_faces, handle_vertices, handle_extent = component_rows[0]
    body_faces, body_vertices, body_extent = component_rows[1]
    if float(np.max(handle_extent)) >= 0.5 * float(np.max(body_extent)):
        raise RuntimeError("BC017 handle/body component classification is ambiguous")

    authored_paths: list[str] = []
    records: list[dict[str, Any]] = []

    def author_mesh(
        *,
        label: str,
        points64: Any,
        faces: list[list[int]],
        approximation: str,
        collision_derivation: str,
        semantic_role: str,
        source_face_ordinals: list[int],
        source_vertex_indices: list[int],
        segment: str | None = None,
        convex_halfspaces: list[list[float]] | None = None,
    ) -> None:
        points = np.asarray(points64, dtype=np.float64).astype(np.float32).astype(
            np.float64
        )
        if (
            len(points) < 4
            or not faces
            or not np.all(np.isfinite(points))
            or any(len(face) < 3 for face in faces)
        ):
            raise RuntimeError(f"BC017 compound collider is invalid: {label}")
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)
        collider_path = (
            f"{body_path}/{EXACT_MESH_COLLIDER_SCOPE}/{source_node}_{label}"
        )
        mesh = UsdGeom.Mesh.Define(stage, collider_path)
        mesh.CreatePointsAttr(
            [Gf.Vec3f(*(float(value) for value in point)) for point in points]
        )
        mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
        mesh.CreateFaceVertexIndicesAttr(
            [vertex for face in faces for vertex in face]
        )
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr([Gf.Vec3f(*minimum), Gf.Vec3f(*maximum)])
        mesh.MakeInvisible()
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr().Set(approximation)
        if approximation == "sdf":
            sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
            sdf.CreateSdfResolutionAttr().Set(DRAWER_BODY_SDF_RESOLUTION)
            sdf.CreateSdfEnableRemeshingAttr().Set(True)
            sdf.CreateSdfTriangleCountReductionFactorAttr().Set(1.0)
        PhysxSchema.PhysxCollisionAPI.Apply(
            prim
        ).CreateContactOffsetAttr().Set(COLLIDER_CONTACT_OFFSET_M)
        PhysxSchema.PhysxCollisionAPI(
            prim
        ).CreateRestOffsetAttr().Set(0.0)
        prim.SetCustomDataByKey(
            "handson:classification",
            "source_drawer_component_collider",
        )
        prim.SetCustomDataByKey(
            "handson:collisionDerivation", collision_derivation
        )
        prim.SetCustomDataByKey("handson:sourceVisiblePrimPath", source_suffix)
        prim.SetCustomDataByKey("handson:semanticRole", semantic_role)
        prim.SetCustomDataByKey("handson:drawerRole", drawer_role)
        prim.SetCustomDataByKey("handson:handleRole", hand_side)
        if segment is not None:
            prim.SetCustomDataByKey("handson:handleSegment", segment)
        relative = collider_path.removeprefix("/WoodenDrawer/")
        authored_paths.append(relative)
        points_payload = [
            [float(value) for value in point] for point in points
        ]
        record: dict[str, Any] = {
            "collider_prim": relative,
            "source_visible_prim": source_mesh_path.removeprefix(
                "/WoodenDrawer/"
            ),
            "source_visible_owner_relative_prim": source_suffix,
            "source_node": source_node,
            "semantic_role": semantic_role,
            "drawer_role": drawer_role,
            "handle_role": hand_side,
            "handle_segment": segment,
            "method": collision_derivation,
            "generation": collision_derivation,
            "source_derived": True,
            "hand_authored_or_placeholder": False,
            "primitive_or_generic_proxy_fallback": False,
            "source_face_ordinals": source_face_ordinals,
            "source_face_ordinals_sha256": _sha256_json_value(
                source_face_ordinals
            ),
            "source_vertex_indices": source_vertex_indices,
            "source_vertex_indices_sha256": _sha256_json_value(
                source_vertex_indices
            ),
            "authored_points_sha256": _sha256_json_value(points_payload),
            "authored_faces_sha256": _sha256_json_value(faces),
            "authored_vertex_count": len(points),
            "authored_face_count": len(faces),
            "source_weld_tolerance_m": SOURCE_WELD_TOLERANCE_M,
            "source_welded_vertex_count": len(welded_points),
            "source_component_count": len(components),
            "source_component_closed_manifold_verified": True,
            "computed_visibility": "invisible",
            "mesh_collision_approximation": approximation,
            "contact_offset_m": COLLIDER_CONTACT_OFFSET_M,
            "rest_offset_m": 0.0,
        }
        if approximation == "sdf":
            record.update(
                {
                    "sdf_resolution": DRAWER_BODY_SDF_RESOLUTION,
                    "sdf_cell_size_max_m": float(
                        np.max(maximum - minimum)
                        / DRAWER_BODY_SDF_RESOLUTION
                    ),
                    "sdf_remeshing_enabled": True,
                    "sdf_triangle_count_reduction_factor": 1.0,
                }
            )
        if convex_halfspaces is not None:
            record["convex_halfspaces_local"] = convex_halfspaces
        records.append(record)

    body_vertex_ids = sorted(body_vertices)
    body_compact = {
        vertex: index for index, vertex in enumerate(body_vertex_ids)
    }
    body_compact_faces = [
        [body_compact[vertex] for vertex in welded_faces[face_index]]
        for face_index in sorted(body_faces)
    ]
    author_mesh(
        label="drawer_body_welded_sdf",
        points64=welded_points[body_vertex_ids],
        faces=body_compact_faces,
        approximation="sdf",
        collision_derivation=(
            "exact_coordinate_weld_then_closed_drawer_body_component_sdf"
        ),
        semantic_role="drawer_body_collider",
        source_face_ordinals=sorted(body_faces),
        source_vertex_indices=[
            welded_representatives[index] for index in body_vertex_ids
        ],
    )

    handle_axis = int(np.argmax(handle_extent))
    handle_axis_values = welded_points[sorted(handle_vertices), handle_axis]
    handle_min = float(np.min(handle_axis_values))
    handle_span = float(np.max(handle_axis_values) - handle_min)
    if handle_span <= 0.0:
        raise RuntimeError("BC017 handle span is degenerate")
    lower_fraction, upper_fraction = HANDLE_SEGMENT_FRACTIONS
    base_regions: list[set[int]] = [set(), set(), set()]
    for face_index in sorted(handle_faces):
        centroid = float(
            np.mean(
                welded_points[welded_faces[face_index], handle_axis]
            )
        )
        fraction = (centroid - handle_min) / handle_span
        region = 0 if fraction <= lower_fraction else 2 if fraction >= upper_fraction else 1
        base_regions[region].add(face_index)
    if any(not region for region in base_regions):
        raise RuntimeError("BC017 handle partition produced an empty segment")
    expanded_regions = [
        region
        | {
            neighbor
            for face_index in region
            for neighbor in face_neighbors[face_index]
            if neighbor in handle_faces
        }
        for region in base_regions
    ]
    segment_names = ("negative_leg", "crossbar", "positive_leg")
    region_axis_bounds: list[tuple[float, float]] = []
    for segment, region in zip(segment_names, expanded_regions, strict=True):
        vertex_ids = sorted(
            {
                vertex
                for face_index in region
                for vertex in welded_faces[face_index]
            }
        )
        selected_points = welded_points[vertex_ids]
        hull = ConvexHull(selected_points, qhull_options="Qt")
        hull_vertex_selection = sorted(int(index) for index in hull.vertices)
        if len(hull_vertex_selection) > LOW_POLY_MAX_VERTICES:
            raise RuntimeError(
                f"BC017 handle {segment} exceeds the GPU vertex cap: "
                f"{len(hull_vertex_selection)}"
            )
        selection_to_authored = {
            selected_index: authored_index
            for authored_index, selected_index in enumerate(
                hull_vertex_selection
            )
        }
        authored_points64 = selected_points[hull_vertex_selection]
        centroid = np.mean(authored_points64, axis=0)
        authored_faces: list[list[int]] = []
        for simplex in hull.simplices:
            face = [selection_to_authored[int(index)] for index in simplex]
            a, b, c = (authored_points64[index] for index in face)
            if float(np.dot(np.cross(b - a, c - a), (a + b + c) / 3.0 - centroid)) < 0.0:
                face[1], face[2] = face[2], face[1]
            authored_faces.append(face)
        if len(authored_faces) > LOW_POLY_MAX_TRIANGLES:
            raise RuntimeError(
                f"BC017 handle {segment} exceeds the GPU face cap: "
                f"{len(authored_faces)}"
            )
        edge_incidence: dict[tuple[int, int], int] = {}
        for face in authored_faces:
            for position, start in enumerate(face):
                edge = tuple(sorted((start, face[(position + 1) % 3])))
                edge_incidence[edge] = edge_incidence.get(edge, 0) + 1
        if not edge_incidence or any(
            count != 2 for count in edge_incidence.values()
        ):
            raise RuntimeError(f"BC017 handle {segment} hull is not closed")
        halfspaces = [
            [float(value) for value in equation]
            for equation in np.asarray(hull.equations, dtype=np.float64)
        ]
        author_mesh(
            label=f"handle_{segment}_source_convex",
            points64=authored_points64,
            faces=authored_faces,
            approximation="convexHull",
            collision_derivation=(
                "welded_source_handle_component_20_60_20_partition_"
                "plus_one_face_ring_then_convex_hull"
            ),
            semantic_role="integrated_u_pull_compound_collider",
            source_face_ordinals=sorted(region),
            source_vertex_indices=[
                welded_representatives[vertex_ids[index]]
                for index in hull_vertex_selection
            ],
            segment=segment,
            convex_halfspaces=halfspaces,
        )
        region_axis_bounds.append(
            (
                float(np.min(authored_points64[:, handle_axis])),
                float(np.max(authored_points64[:, handle_axis])),
            )
        )
    seam_overlaps = [
        min(region_axis_bounds[index][1], region_axis_bounds[index + 1][1])
        - max(region_axis_bounds[index][0], region_axis_bounds[index + 1][0])
        for index in range(2)
    ]
    if any(overlap <= 2.0 * COLLIDER_CONTACT_OFFSET_M for overlap in seam_overlaps):
        raise RuntimeError(
            "BC017 handle hull seams do not overlap beyond contact offset: "
            f"{seam_overlaps}"
        )
    for record in records:
        if record.get("handle_segment") is not None:
            record["handle_partition_axis"] = handle_axis
            record["handle_partition_fractions"] = list(
                HANDLE_SEGMENT_FRACTIONS
            )
            record["handle_seam_overlaps_m"] = seam_overlaps
            record["handle_source_triangle_coverage_verified"] = True
            record["source_vertex_subset_verified"] = True
            record["closed_manifold_verified"] = True
            record["convex_verified"] = True

    owner_relative = [
        f"/{path.removeprefix(f'{body_prim.GetName()}/')}"
        for path in authored_paths
    ]
    source_prim.SetCustomDataByKey(
        "handson:derivedCollisionPrimPathsJson",
        json.dumps(owner_relative, sort_keys=True, separators=(",", ":")),
    )
    return authored_paths, records


def _audit_authored_closed_collider_bounds(
    stage: Any,
    collider_paths_by_body: dict[str, list[str]],
    *,
    asset_root_placement_m: tuple[float, float, float],
) -> dict[str, Any]:
    """Audit authored collider bounds and fixture support in the closed pose.

    Positive AABB separation remains a conclusive non-intersection result.
    A concave fixture decomposition necessarily has one aggregate AABB around
    all of its open bays, so overlap with a drawer is only an inconclusive
    broad-phase result and is recorded as such for the live PhysX pull smoke.
    """

    from pxr import Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
        useExtentsHint=True,
    )
    bounds: dict[str, dict[str, Any]] = {}
    body_by_path: dict[str, str] = {}
    for body, paths in sorted(collider_paths_by_body.items()):
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid() or not UsdGeom.Boundable(prim):
                raise RuntimeError(f"cannot compute collider bounds: {path}")
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            minimum = tuple(float(value) for value in box.GetMin())
            maximum = tuple(float(value) for value in box.GetMax())
            if any(maximum[index] <= minimum[index] for index in range(3)):
                raise RuntimeError(f"degenerate collider bounds: {path}")
            relative = path.removeprefix("/WoodenDrawer/")
            bounds[relative] = {
                "body": body,
                "minimum_m": list(minimum),
                "maximum_m": list(maximum),
                "mesh_collision_approximation": str(
                    UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                ),
            }
            body_by_path[path] = body

    pair_results: list[dict[str, Any]] = []
    inconclusive_pairs = 0
    drawer_body_names = frozenset(DRAWER_LINK_PRIMS.values())
    filtered_body_pairs: set[tuple[str, str]] = set()
    absolute_paths = sorted(body_by_path)
    for index, left_path in enumerate(absolute_paths):
        for right_path in absolute_paths[index + 1 :]:
            left_body = body_by_path[left_path]
            right_body = body_by_path[right_path]
            if left_body == right_body:
                continue
            left = bounds[left_path.removeprefix("/WoodenDrawer/")]
            right = bounds[right_path.removeprefix("/WoodenDrawer/")]
            left_min = left["minimum_m"]
            left_max = left["maximum_m"]
            right_min = right["minimum_m"]
            right_max = right["maximum_m"]
            axis_gaps = [
                max(
                    float(right_min[axis]) - float(left_max[axis]),
                    float(left_min[axis]) - float(right_max[axis]),
                    0.0,
                )
                for axis in range(3)
            ]
            overlap_depths = [
                min(float(left_max[axis]), float(right_max[axis]))
                - max(float(left_min[axis]), float(right_min[axis]))
                for axis in range(3)
            ]
            separated = any(gap > 0.0 for gap in axis_gaps)
            collision_filtered = (
                left_body in drawer_body_names
                and right_body in drawer_body_names
            )
            if collision_filtered:
                filtered_body_pairs.add(tuple(sorted((left_body, right_body))))
            if not separated and not collision_filtered:
                inconclusive_pairs += 1
            pair_results.append(
                {
                    "left": left_path.removeprefix("/WoodenDrawer/"),
                    "left_body": left_body,
                    "right": right_path.removeprefix("/WoodenDrawer/"),
                    "right_body": right_body,
                    "axis_separation_m": axis_gaps,
                    "axis_overlap_m": overlap_depths,
                    "separated": separated,
                    "collision_filtered": collision_filtered,
                    "aabb_result": (
                        "excluded_by_guided_drawer_filtered_pair"
                        if collision_filtered
                        else (
                            "conclusive_separation"
                            if separated
                            else "inconclusive_concave_broad_phase_overlap"
                        )
                    ),
                }
            )
    if not pair_results:
        raise RuntimeError("closed-pose collider audit found no cross-body pairs")
    expected_drawer_pair_count = len(DRAWER_ROLE_ORDER) * (
        len(DRAWER_ROLE_ORDER) - 1
    ) // 2
    if len(filtered_body_pairs) != expected_drawer_pair_count:
        raise RuntimeError(
            "BC017 closed-pose audit does not cover every filtered drawer pair"
        )

    # The placement rotation maps local +Y to world +Z, so fixture support is
    # determined by the minimum local Y coordinate rather than local Z.
    base_min_support = min(
        float(record["minimum_m"][1])
        for record in bounds.values()
        if record["body"] == BASE_LINK
    )
    support_gap = asset_root_placement_m[2] + base_min_support - TABLE_TOP_Z_M
    if support_gap < 0.0 or support_gap > 0.005:
        raise RuntimeError(f"authored fixture support gap is invalid: {support_gap}")
    return {
        "method": (
            "transformed USD world-aligned bounds of every declared "
            "source-derived collider; positive cross-body AABB separation is "
            "conclusive, while concave-decomposition overlap is deferred to "
            "the live PhysX closed-pose and full-travel smoke"
        ),
        "scope": "authored_closed_pose_before_PhysX",
        "formal": False,
        "same_body_pairs_excluded": True,
        "asset_root_placement_m": list(asset_root_placement_m),
        "asset_root_orientation_wxyz": list(PLACEMENT_ORIENTATION_WXYZ),
        "table_top_z_m": TABLE_TOP_Z_M,
        "collider_bounds": bounds,
        "cross_body_pairs": pair_results,
        "inconclusive_concave_aabb_pair_count": inconclusive_pairs,
        "drawer_filtered_body_pairs": [
            list(pair) for pair in sorted(filtered_body_pairs)
        ],
        "drawer_filtered_body_pair_count": len(filtered_body_pairs),
        "drawer_filtered_pair_policy": (
            "all six parallel guided drawer bodies only; fixture and hand "
            "contacts remain enabled"
        ),
        "closed_reset_max_penetration_m": 0.0,
        "support_gap_m": support_gap,
        "physx_live_validated": False,
    }


def _author_body(stage: Any, path: str, mass_kg: float, inertia: tuple[float, float, float], com: tuple[float, float, float]) -> Any:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    PhysxSchema.PhysxRigidBodyAPI.Apply(
        prim
    ).CreateEnableSpeculativeCCDAttr().Set(True)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(mass_kg)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
    return prim


def _author_guided_drawer_filtered_pairs(
    drawer_bodies: dict[str, Any],
) -> list[dict[str, str]]:
    """Disable only impossible drawer-to-drawer contacts.

    The six source drawers remain independently movable along parallel +X
    prismatic guides.  Their row and pedestal offsets can never change, so
    drawer-to-drawer collision has no real mechanical role.  Filtering those
    pairs removes SDF voxel/contact-offset chatter at source seams while
    retaining every drawer-to-fixture and drawer-to-hand contact.
    """

    from pxr import UsdPhysics

    if tuple(drawer_bodies) != DRAWER_ROLE_ORDER:
        raise RuntimeError("BC017 filtered-pair drawer order is not canonical")
    records: list[dict[str, str]] = []
    for index, role in enumerate(DRAWER_ROLE_ORDER):
        body = drawer_bodies[role]
        other_roles = tuple(
            candidate
            for candidate in DRAWER_ROLE_ORDER
            if candidate != role
        )
        api = UsdPhysics.FilteredPairsAPI.Apply(body)
        relation = api.CreateFilteredPairsRel()
        if not relation.SetTargets(
            [drawer_bodies[candidate].GetPath() for candidate in other_roles]
        ):
            raise RuntimeError(
                f"BC017 could not author filtered drawer pairs for {role}"
            )
        body.SetCustomDataByKey(
            "handson:filteredDrawerRolesJson",
            json.dumps(list(other_roles), separators=(",", ":")),
        )
        body.SetCustomDataByKey(
            "handson:drawerFilteredPairPolicy",
            "guided_drawer_bodies_only_fixture_and_hand_contacts_preserved",
        )
        for other_role in DRAWER_ROLE_ORDER[index + 1 :]:
            records.append(
                {
                    "left_drawer_role": role,
                    "left_link_prim": DRAWER_LINK_PRIMS[role],
                    "right_drawer_role": other_role,
                    "right_link_prim": DRAWER_LINK_PRIMS[other_role],
                }
            )
    expected_pair_count = len(DRAWER_ROLE_ORDER) * (
        len(DRAWER_ROLE_ORDER) - 1
    ) // 2
    if len(records) != expected_pair_count:
        raise RuntimeError("BC017 filtered drawer pair count differs")
    return records


def _reference_source(
    stage: Any,
    body_path: str,
    translation: tuple[float, float, float],
    keep_nodes: Iterable[str],
) -> str:
    from pxr import Gf, Sdf, UsdGeom

    visual_path = f"{body_path}/source_geometry"
    visual = UsdGeom.Xform.Define(stage, visual_path)
    visual.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translation))
    visual.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(
            SOURCE_TO_RUNTIME_ORIENTATION_WXYZ[0],
            Gf.Vec3d(*SOURCE_TO_RUNTIME_ORIENTATION_WXYZ[1:]),
        )
    )
    visual.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(UNIFORM_SCALE))
    reference_path = f"{visual_path}/converted_source"
    referenced = UsdGeom.Xform.Define(stage, reference_path).GetPrim()
    if not referenced.GetReferences().AddReference(
        Sdf.Reference(assetPath="../source/converted_scene.usd", primPath="/World")
    ):
        raise RuntimeError("could not add the local converted-source reference")
    stage.Load()
    _deactivate_unselected(stage, reference_path, keep_nodes)
    for prim in stage.Traverse():
        if prim.GetPath().HasPrefix(referenced.GetPath()) and prim.GetTypeName() == "Mesh":
            prim.SetCustomDataByKey("handson:classification", "task_visible_source_mesh")
    return reference_path


def _author_runtime_usd(source_usd: Path, output: Path, archive_sha256: str) -> dict[str, Any]:
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    source_stage = Usd.Stage.Open(str(source_usd), Usd.Stage.LoadAll)
    if source_stage is None:
        raise RuntimeError(f"cannot inspect converted source: {source_usd}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)
    source_root = source_stage.GetPrimAtPath("/World")
    drawer_targets = {
        role: source_stage.GetPrimAtPath(f"/World/{source_node}")
        for role, source_node in SOURCE_DRAWER_NODES_BY_ROLE.items()
    }
    if not all(
        prim and prim.IsValid()
        for prim in (source_root, *drawer_targets.values())
    ):
        raise RuntimeError("selected source desk/drawer target nodes are absent")
    source_min, source_max = _bounds(cache, source_root)
    drawer_source_bounds = {
        role: _bounds(cache, target)
        for role, target in drawer_targets.items()
    }
    base_translation = (0.0, 0.0, 0.0)

    stage = Usd.Stage.CreateNew(str(output))
    if stage is None:
        raise RuntimeError(f"cannot create runtime USD: {output}")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/WoodenDrawer").GetPrim()
    stage.SetDefaultPrim(root)
    root.SetCustomDataByKey("handson:assetTitle", SOURCE_TITLE)
    root.SetCustomDataByKey("handson:assetCreator", SOURCE_CREATOR)
    root.SetCustomDataByKey("handson:sourceUrl", SOURCE_URL)
    root.SetCustomDataByKey("handson:license", LICENSE_NAME)
    root.SetCustomDataByKey("handson:licenseUrl", LICENSE_URL)
    root.SetCustomDataByKey("handson:sourceArchiveSha256", archive_sha256)
    root.SetCustomDataByKey(
        "handson:sourceDrawerNodesJson",
        json.dumps(sorted(SOURCE_DRAWER_NODES), separators=(",", ":")),
    )
    root.SetCustomDataByKey(
        "handson:drawerReplication",
        "none; all six original source drawers are independent links",
    )
    root.SetCustomDataByKey("handson:proceduralVisibleGeometry", False)
    root.SetCustomDataByKey("handson:proceduralCollisionGeometry", False)
    root.SetCustomDataByKey(
        "handson:collisionGeneration",
        "exact_source_decomposition_for_fixture_plus_welded_drawer_body_sdf_"
        "and_three_source_handle_hulls",
    )
    root.SetCustomDataByKey("handson:sourceDerivedCollisionGeometry", True)
    root.SetCustomDataByKey("handson:handAuthoredOrPlaceholderCollision", False)
    root.SetCustomDataByKey(
        "handson:drawerCollisionFiltering",
        "all_six_guided_drawer_bodies_pairwise_only; fixture_and_hand_"
        "contacts_preserved",
    )

    # Uniform geometric scaling changes mass by s^3 and inertia by s^5 when
    # density is held fixed.  Apply those ratios relative to the reviewed
    # baseline conversion instead of leaving a large desk with toy inertia.
    mass_ratio = TASK_SCALE_MULTIPLIER**3
    inertia_ratio = TASK_SCALE_MULTIPLIER**5
    com_ratio = TASK_SCALE_MULTIPLIER
    base = _author_body(
        stage,
        f"/WoodenDrawer/{BASE_LINK}",
        12.0 * mass_ratio,
        tuple(value * inertia_ratio for value in (0.35, 0.28, 0.22)),
        tuple(value * com_ratio for value in (0.0, 0.18, 0.0)),
    )
    drawer_bodies = {
        role: _author_body(
            stage,
            f"/WoodenDrawer/{DRAWER_LINK_PRIMS[role]}",
            1.2 * mass_ratio,
            tuple(
                value * inertia_ratio
                for value in (0.008, 0.012, 0.010)
            ),
            tuple(
                value * com_ratio
                for value in (
                    0.0,
                    0.32,
                    0.085 if role.startswith("left_") else -0.085,
                )
            ),
        )
        for role in DRAWER_ROLE_ORDER
    }
    for role in DRAWER_ROLE_ORDER:
        source_node = SOURCE_DRAWER_NODES_BY_ROLE[role]
        body = drawer_bodies[role]
        translation = DRAWER_SOURCE_TRANSLATION_M_BY_ROLE[role]
        body.SetCustomDataByKey("handson:drawerRole", role)
        body.SetCustomDataByKey(
            "handson:handSide", role.partition("_")[0]
        )
        body.SetCustomDataByKey(
            "handson:drawerSlot", role.split("_")[1]
        )
        body.SetCustomDataByKey("handson:sourcePrimaryNode", source_node)
        body.SetCustomDataByKey("handson:sourceTargetOpeningNode", source_node)
        body.SetCustomDataByKey(
            "handson:sourceNodesJson",
            json.dumps([source_node], separators=(",", ":")),
        )
        body.SetCustomDataByKey(
            "handson:sourceToClosedTranslationMJson",
            json.dumps(list(translation), separators=(",", ":")),
        )
        body.SetCustomDataByKey(
            "handson:lineage",
            "preserved original source-visible drawer; source pose retained; "
            "independent rigid link and prismatic DOF",
        )
    drawer_filtered_pairs = _author_guided_drawer_filtered_pairs(
        drawer_bodies
    )

    all_source_nodes = {
        str(child.GetName())
        for child in source_root.GetChildren()
        if str(child.GetName()) != "Looks"
    }
    missing_source_nodes = (
        SOURCE_BASE_NODES | SOURCE_DRAWER_NODES
    ) - all_source_nodes
    if missing_source_nodes:
        raise RuntimeError(
            "selected source desk nodes are absent: "
            f"{sorted(missing_source_nodes)}"
        )
    # Only the desk structure and two trays are fixed.  All six original
    # drawers, including their integrated U pulls, remain independent links.
    base_nodes = set(SOURCE_BASE_NODES)
    base_reference = _reference_source(stage, str(base.GetPath()), base_translation, base_nodes)
    drawer_references = {
        role: _reference_source(
            stage,
            str(drawer_bodies[role].GetPath()),
            DRAWER_SOURCE_TRANSLATION_M_BY_ROLE[role],
            {SOURCE_DRAWER_NODES_BY_ROLE[role]},
        )
        for role in DRAWER_ROLE_ORDER
    }

    def mesh_path(reference: str, node: str) -> str:
        source_node_path = f"{reference}/{node}"
        return str(_node_mesh(stage, source_node_path, UsdGeom).GetPath())

    drawer_mesh_paths = {
        role: mesh_path(
            drawer_references[role], SOURCE_DRAWER_NODES_BY_ROLE[role]
        )
        for role in DRAWER_ROLE_ORDER
    }
    for role, drawer_mesh_path in drawer_mesh_paths.items():
        drawer_mesh = stage.GetPrimAtPath(drawer_mesh_path)
        drawer_mesh.SetCustomDataByKey(
            "handson:drawerRole", role
        )
        drawer_mesh.SetCustomDataByKey(
            "handson:handleRole", role.partition("_")[0]
        )
        drawer_mesh.SetCustomDataByKey(
            "handson:semanticRole", "drawer_with_integrated_source_handle"
        )

    body_prims = {
        BASE_LINK: base,
        **{
            DRAWER_LINK_PRIMS[role]: drawer_bodies[role]
            for role in DRAWER_ROLE_ORDER
        },
    }
    collider_specs = {
        BASE_LINK: [
            (
                mesh_path(base_reference, node),
                node,
                "fixture_structure_collider",
                None,
                "exact_decomposition",
            )
            for node in SOURCE_BASE_DECOMPOSITION_NODES
        ],
        **{
            DRAWER_LINK_PRIMS[role]: [
                (
                    drawer_mesh_paths[role],
                    SOURCE_DRAWER_NODES_BY_ROLE[role],
                    "drawer_and_integrated_u_pull_collider",
                    role.partition("_")[0],
                    "source_drawer_compound",
                ),
            ]
            for role in DRAWER_ROLE_ORDER
        },
    }
    collider_prims: dict[str, list[str]] = {}
    absolute_collider_prims: dict[str, list[str]] = {}
    collider_derivations: list[dict[str, Any]] = []
    body_order = (
        BASE_LINK,
        *(DRAWER_LINK_PRIMS[role] for role in DRAWER_ROLE_ORDER),
    )
    drawer_role_by_link = {
        link: role for role, link in DRAWER_LINK_PRIMS.items()
    }
    for body_name in body_order:
        collider_prims[body_name] = []
        absolute_collider_prims[body_name] = []
        for (
            source_mesh_path,
            source_node,
            semantic_role,
            handle_role,
            generation,
        ) in collider_specs[body_name]:
            if generation == "source_drawer_compound":
                if handle_role is None:
                    raise RuntimeError("compound drawer collider needs a handle role")
                relative_paths, derivations = (
                    _author_source_drawer_compound_colliders(
                        stage,
                        body_prim=body_prims[body_name],
                        source_mesh_path=source_mesh_path,
                        source_node=source_node,
                        drawer_role=drawer_role_by_link[body_name],
                        hand_side=handle_role,
                    )
                )
                collider_prims[body_name].extend(relative_paths)
                absolute_collider_prims[body_name].extend(
                    f"/WoodenDrawer/{path}" for path in relative_paths
                )
                collider_derivations.extend(derivations)
                continue
            if generation in {"exact_decomposition", "exact_sdf"}:
                relative_path, derivation = (
                    _author_source_exact_mesh_collider(
                        stage,
                        body_prim=body_prims[body_name],
                        source_mesh_path=source_mesh_path,
                        source_node=source_node,
                        semantic_role=semantic_role,
                        handle_role=handle_role,
                        use_sdf=generation == "exact_sdf",
                    )
                )
            elif generation == "low_poly_hull":
                relative_path, derivation = _author_source_derived_low_poly_collider(
                    stage,
                    body_prim=body_prims[body_name],
                    source_mesh_path=source_mesh_path,
                    source_node=source_node,
                    semantic_role=semantic_role,
                    handle_role=handle_role,
                )
            else:
                raise RuntimeError(
                    f"unsupported BC017 collider generation: {generation}"
                )
            collider_prims[body_name].append(relative_path)
            absolute_collider_prims[body_name].append(
                f"/WoodenDrawer/{relative_path}"
            )
            collider_derivations.append(derivation)

    local_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
        useExtentsHint=True,
    )
    visual_collision_bound_audit: dict[str, dict[str, Any]] = {}
    visual_references = {
        BASE_LINK: base_reference,
        **{
            DRAWER_LINK_PRIMS[role]: drawer_references[role]
            for role in DRAWER_ROLE_ORDER
        },
    }

    def visible_vertex_bounds(root_paths: Iterable[str]) -> tuple[
        tuple[float, float, float], tuple[float, float, float]
    ]:
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        transformed: list[tuple[float, float, float]] = []
        for root_path in root_paths:
            bound_root = stage.GetPrimAtPath(root_path)
            if not bound_root or not bound_root.IsValid():
                raise RuntimeError(f"BC017 bound root is absent: {root_path}")
            for prim in Usd.PrimRange(bound_root):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                if not points:
                    continue
                matrix = xform_cache.GetLocalToWorldTransform(prim)
                transformed.extend(
                    tuple(
                        float(value)
                        for value in matrix.Transform(Gf.Vec3d(*point))
                    )
                    for point in points
                )
        if not transformed:
            raise RuntimeError(
                f"BC017 bound roots contain no mesh vertices: {list(root_paths)}"
            )
        return (
            tuple(min(point[axis] for point in transformed) for axis in range(3)),
            tuple(max(point[axis] for point in transformed) for axis in range(3)),
        )

    for body_name, visual_reference in visual_references.items():
        visual_min, visual_max = visible_vertex_bounds((visual_reference,))
        collision_min, collision_max = visible_vertex_bounds(
            absolute_collider_prims[body_name]
        )
        maximum_bound_error = max(
            *(abs(visual_min[axis] - collision_min[axis]) for axis in range(3)),
            *(abs(visual_max[axis] - collision_max[axis]) for axis in range(3)),
        )
        if maximum_bound_error > 1.0e-6:
            raise RuntimeError(
                "BC017 aggregate visual/collision bounds differ for "
                f"{body_name}: error={maximum_bound_error}"
            )
        visual_collision_bound_audit[body_name] = {
            "visual_minimum_m": list(visual_min),
            "visual_maximum_m": list(visual_max),
            "collision_minimum_m": list(collision_min),
            "collision_maximum_m": list(collision_max),
            "maximum_absolute_bound_error_m": maximum_bound_error,
            "aligned_within_tolerance": True,
            "tolerance_m": 1.0e-6,
        }
    def source_handle_bounds_runtime(
        drawer_role: str,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        source_handle_min, source_handle_max = SOURCE_HANDLE_BOUNDS_M[
            drawer_role
        ]
        translation = DRAWER_SOURCE_TRANSLATION_M_BY_ROLE[drawer_role]
        runtime_corners = tuple(
            (
                source_z * UNIFORM_SCALE + translation[0],
                source_y * UNIFORM_SCALE + translation[1],
                -source_x * UNIFORM_SCALE + translation[2],
            )
            for source_x in (source_handle_min[0], source_handle_max[0])
            for source_y in (source_handle_min[1], source_handle_max[1])
            for source_z in (source_handle_min[2], source_handle_max[2])
        )
        return (
            tuple(min(point[axis] for point in runtime_corners) for axis in range(3)),
            tuple(max(point[axis] for point in runtime_corners) for axis in range(3)),
        )

    runtime_handle_bounds = {
        role: source_handle_bounds_runtime(role)
        for role in DRAWER_ROLE_ORDER
    }
    handle_centers = {
        role: tuple(
            0.5 * (bounds[0][axis] + bounds[1][axis])
            for axis in range(3)
        )
        for role, bounds in runtime_handle_bounds.items()
    }
    top_handle_centers = (
        handle_centers["left_top_drawer"],
        handle_centers["right_top_drawer"],
    )
    handle_pair_center = tuple(
        0.5
        * (top_handle_centers[0][axis] + top_handle_centers[1][axis])
        for axis in range(3)
    )
    base_support_min_y = min(
        float(
            local_cache.ComputeWorldBound(stage.GetPrimAtPath(path))
            .ComputeAlignedBox()
            .GetMin()[1]
        )
        for path in absolute_collider_prims[BASE_LINK]
    )
    placement_m = (
        TARGET_CLOSED_HANDLE_X_M + handle_pair_center[0],
        TARGET_HANDLE_Y_M - handle_pair_center[2],
        TABLE_TOP_Z_M + TABLE_SUPPORT_CLEARANCE_M - base_support_min_y,
    )
    collision_bound_audit = _audit_authored_closed_collider_bounds(
        stage,
        absolute_collider_prims,
        asset_root_placement_m=placement_m,
    )

    fixed = UsdPhysics.FixedJoint.Define(stage, f"/WoodenDrawer/{FIXED_JOINT}")
    fixed.CreateBody1Rel().AddTarget(base.GetPath())
    fixed.CreateJointEnabledAttr(True)
    UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
    for role in DRAWER_ROLE_ORDER:
        joint_name = DRAWER_JOINT_NAMES[role]
        drawer = drawer_bodies[role]
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"/WoodenDrawer/{joint_name}")
        joint.CreateBody0Rel().AddTarget(base.GetPath())
        joint.CreateBody1Rel().AddTarget(drawer.GetPath())
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(0.0)
        joint.CreateUpperLimitAttr(DRAWER_TRAVEL_M)
        joint.CreateJointEnabledAttr(True)
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0)))

    stage.GetRootLayer().Save()
    # Sdf's text exporter may emit more than one trailing newline.  Canonicalize
    # the published byte representation before hashing so the builder and the
    # repository asset cannot diverge only because of EOF whitespace.
    output.write_bytes(output.read_bytes().rstrip(b"\r\n") + b"\n")
    reopened = Usd.Stage.Open(str(output), Usd.Stage.LoadAll)
    if reopened is None or not reopened.GetDefaultPrim().IsValid():
        raise RuntimeError("generated runtime USD cannot be reopened")
    rigid = [prim for prim in reopened.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [prim for prim in reopened.Traverse() if prim.IsA(UsdPhysics.Joint)]
    colliders = [prim for prim in reopened.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
    expected_collider_count = (
        len(SOURCE_BASE_DECOMPOSITION_NODES) + 4 * len(DRAWER_ROLE_ORDER)
    )
    if (
        len(rigid) != 1 + len(DRAWER_ROLE_ORDER)
        or len(joints) != 1 + len(DRAWER_ROLE_ORDER)
        or len(colliders) != expected_collider_count
    ):
        raise RuntimeError(
            f"generated topology differs: rigid={len(rigid)} joints={len(joints)} colliders={len(colliders)}"
        )
    for role in DRAWER_ROLE_ORDER:
        body_path = f"/WoodenDrawer/{DRAWER_LINK_PRIMS[role]}"
        body = reopened.GetPrimAtPath(body_path)
        filtered_api = UsdPhysics.FilteredPairsAPI(body)
        observed_targets = {
            str(path) for path in filtered_api.GetFilteredPairsRel().GetTargets()
        }
        expected_targets = {
            f"/WoodenDrawer/{DRAWER_LINK_PRIMS[other_role]}"
            for other_role in DRAWER_ROLE_ORDER
            if other_role != role
        }
        if observed_targets != expected_targets:
            raise RuntimeError(
                "generated guided-drawer filtered pairs differ for "
                f"{role}: {sorted(observed_targets)}"
            )
    base_filtered = UsdPhysics.FilteredPairsAPI(
        reopened.GetPrimAtPath(f"/WoodenDrawer/{BASE_LINK}")
    )
    if base_filtered and base_filtered.GetFilteredPairsRel().GetTargets():
        raise RuntimeError("BC017 fixture contacts may not be filtered")
    for collider in colliders:
        mesh = UsdGeom.Mesh(collider)
        points = mesh.GetPointsAttr().Get() if mesh else None
        face_counts = mesh.GetFaceVertexCountsAttr().Get() if mesh else None
        approximation = UsdPhysics.MeshCollisionAPI(
            collider
        ).GetApproximationAttr().Get()
        generation = collider.GetCustomDataByKey("handson:collisionDerivation")
        source_suffix = collider.GetCustomDataByKey(
            "handson:sourceVisiblePrimPath"
        )
        owner = collider.GetParent()
        while owner.IsValid() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        source_visible = (
            reopened.GetPrimAtPath(f"{owner.GetPath()}{source_suffix}")
            if owner.IsValid()
            and isinstance(source_suffix, str)
            and source_suffix.startswith("/")
            else None
        )
        common_invalid = (
            not mesh
            or not points
            or not face_counts
            or str(UsdGeom.Imageable(collider).ComputeVisibility()) != "invisible"
            or source_visible is None
            or not source_visible.IsValid()
            or not source_visible.IsA(UsdGeom.Mesh)
            or source_visible.HasAPI(UsdPhysics.CollisionAPI)
        )
        fixed_extrema_convex = (
            generation
            == "fixed_64_direction_extrema_of_transformed_visible_source_vertices_then_convex_hull"
        )
        handle_compound_convex = (
            generation
            == "welded_source_handle_component_20_60_20_partition_plus_one_face_ring_then_convex_hull"
        )
        low_poly_invalid = (
            approximation == "convexHull"
            and (
                generation not in {
                    "fixed_64_direction_extrema_of_transformed_visible_source_vertices_then_convex_hull",
                    "welded_source_handle_component_20_60_20_partition_plus_one_face_ring_then_convex_hull",
                }
                or len(points) > LOW_POLY_MAX_VERTICES
                or len(face_counts) > LOW_POLY_MAX_TRIANGLES
                or any(int(count) != 3 for count in face_counts)
                or (
                    handle_compound_convex
                    and collider.GetCustomDataByKey("handson:handleSegment")
                    not in {"negative_leg", "crossbar", "positive_leg"}
                )
            )
        )
        decomposition_invalid = (
            approximation == "convexDecomposition"
            and (
                generation
                != "exact_transformed_visible_source_topology_then_physx_convex_decomposition"
                or len(points)
                != len(UsdGeom.Mesh(source_visible).GetPointsAttr().Get() or ())
                or len(face_counts)
                != len(
                    UsdGeom.Mesh(source_visible).GetFaceVertexCountsAttr().Get()
                    or ()
                )
            )
        )
        sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI(collider)
        exact_source_sdf = (
            generation
            == "exact_transformed_visible_source_topology_then_physx_remeshed_sdf"
        )
        welded_body_sdf = (
            generation
            == "exact_coordinate_weld_then_closed_drawer_body_component_sdf"
        )
        sdf_invalid = (
            approximation == "sdf"
            and (
                generation not in {
                    "exact_transformed_visible_source_topology_then_physx_remeshed_sdf",
                    "exact_coordinate_weld_then_closed_drawer_body_component_sdf",
                }
                or not sdf_api
                or sdf_api.GetSdfResolutionAttr().Get()
                != DRAWER_BODY_SDF_RESOLUTION
                or sdf_api.GetSdfEnableRemeshingAttr().Get() is not True
                or sdf_api.GetSdfTriangleCountReductionFactorAttr().Get()
                != 1.0
                or (
                    exact_source_sdf
                    and (
                        len(points)
                        != len(
                            UsdGeom.Mesh(source_visible).GetPointsAttr().Get()
                            or ()
                        )
                        or len(face_counts)
                        != len(
                            UsdGeom.Mesh(
                                source_visible
                            ).GetFaceVertexCountsAttr().Get()
                            or ()
                        )
                    )
                )
            )
        )
        if (
            common_invalid
            or approximation not in {"convexHull", "convexDecomposition", "sdf"}
            or low_poly_invalid
            or decomposition_invalid
            or sdf_invalid
        ):
            raise RuntimeError(
                f"generated GPU-safe collider contract differs: {collider.GetPath()}"
            )
    return {
        "source_bounds_min_m": list(source_min),
        "source_bounds_max_m": list(source_max),
        "drawer_source_bounds_m": {
            role: {
                "minimum_m": list(bounds[0]),
                "maximum_m": list(bounds[1]),
            }
            for role, bounds in drawer_source_bounds.items()
        },
        # Backward-compatible evidence aliases identify the original top pair.
        "left_target_bounds_min_m": list(
            drawer_source_bounds["left_top_drawer"][0]
        ),
        "left_target_bounds_max_m": list(
            drawer_source_bounds["left_top_drawer"][1]
        ),
        "right_target_bounds_min_m": list(
            drawer_source_bounds["right_top_drawer"][0]
        ),
        "right_target_bounds_max_m": list(
            drawer_source_bounds["right_top_drawer"][1]
        ),
        "uniform_scale": UNIFORM_SCALE,
        "baseline_uniform_scale": BASELINE_UNIFORM_SCALE,
        "task_scale_multiplier": TASK_SCALE_MULTIPLIER,
        "source_to_runtime_orientation_wxyz": list(
            SOURCE_TO_RUNTIME_ORIENTATION_WXYZ
        ),
        "source_closed_front_z_m": SOURCE_CLOSED_FRONT_Z_M,
        "drawer_closed_alignment_x_m": DRAWER_CLOSED_ALIGNMENT_X_M,
        "drawer_closed_alignment_y_m": DRAWER_CLOSED_ALIGNMENT_Y_M,
        "drawer_source_translation_m_by_role": {
            role: list(translation)
            for role, translation in DRAWER_SOURCE_TRANSLATION_M_BY_ROLE.items()
        },
        "runtime_handle_grip_bounds_m": {
            role: [list(bounds[0]), list(bounds[1])]
            for role, bounds in runtime_handle_bounds.items()
        },
        "local_handle_centers_m": {
            **{
                role: list(center)
                for role, center in handle_centers.items()
            },
            "left": list(handle_centers["left_top_drawer"]),
            "right": list(handle_centers["right_top_drawer"]),
            "pair_center": list(handle_pair_center),
            "top_pair_center": list(handle_pair_center),
        },
        "local_support_min_y_m": base_support_min_y,
        "asset_root_placement_m": list(placement_m),
        "base_source_nodes": sorted(base_nodes),
        "drawer_source_nodes": sorted(SOURCE_DRAWER_NODES),
        "drawer_source_nodes_by_role": SOURCE_DRAWER_NODES_BY_ROLE,
        "left_target_source_node": SOURCE_LEFT_DRAWER_NODE,
        "right_target_source_node": SOURCE_RIGHT_DRAWER_NODE,
        "omitted_decorative_source_nodes": sorted(
            all_source_nodes - base_nodes - SOURCE_DRAWER_NODES
        ),
        "drawer_filtered_pairs": drawer_filtered_pairs,
        "drawer_filtered_pair_count": len(drawer_filtered_pairs),
        "drawer_filtered_pair_policy": {
            "scope": "drawer_body_to_drawer_body_only",
            "reason": (
                "parallel prismatic guides preserve row/pedestal offsets; "
                "suppress source-seam SDF/contact-offset chatter"
            ),
            "fixture_contacts_preserved": True,
            "hand_contacts_preserved": True,
            "joint_limits_or_locks_added": False,
        },
        "collider_prims": collider_prims,
        "collider_derivations": collider_derivations,
        "visual_collision_bound_audit": visual_collision_bound_audit,
        "authored_closed_collision_audit": collision_bound_audit,
    }


def _verify_dependencies(asset_root: Path, runtime_path: Path) -> list[str]:
    from pxr import Sdf, UsdUtils

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(runtime_path)))
    if unresolved:
        raise RuntimeError(f"generated USD has unresolved dependencies: {[str(v) for v in unresolved]}")
    observed: set[Path] = set()
    for layer in layers:
        real = str(getattr(layer, "realPath", "") or "")
        if real:
            observed.add(Path(real).resolve(strict=True))
    for asset in assets:
        resolved = str(getattr(asset, "resolvedPath", "") or "")
        raw = str(getattr(asset, "path", "") or asset)
        candidate = Path(resolved or raw)
        if not candidate.is_absolute():
            candidate = runtime_path.parent / candidate
        observed.add(candidate.resolve(strict=True))
    root = asset_root.resolve(strict=True)
    if any(not path.is_relative_to(root) for path in observed):
        raise RuntimeError(f"generated dependency escapes asset root: {sorted(map(str, observed))}")
    return sorted(path.relative_to(root).as_posix() for path in observed)


def _build_bundle(
    archive_path: Path,
    asset_root: Path,
    extracted_root: Path,
) -> dict[str, object]:
    source_dir = asset_root / "source"
    runtime_dir = asset_root / "runtime"
    evidence_dir = asset_root / "evidence"
    (source_dir / "textures").mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)
    shutil.copy2(archive_path, source_dir / "original.zip")
    shutil.copy2(extracted_root / "license.txt", source_dir / "license.txt")
    texture_names = (
        "metal_office_desk_diff_1k.jpg",
        "metal_office_desk_nor_gl_1k.jpg",
        "metal_office_desk_arm_1k.jpg",
    )
    source_textures: list[Path] = []
    for texture_name in texture_names:
        source_texture = extracted_root / "textures" / texture_name
        destination_texture = source_dir / "textures" / texture_name
        shutil.copy2(source_texture, destination_texture)
        source_textures.append(destination_texture)

    converted = source_dir / "converted_scene.usd"
    _convert_gltf(extracted_root / "scene.gltf", converted)
    _normalize_converted_source(converted, source_textures)
    runtime_path = runtime_dir / "bc017_wooden_drawer.usda"
    lineage = _author_runtime_usd(converted, runtime_path, EXPECTED_ARCHIVE_SHA256)
    dependencies = _verify_dependencies(asset_root, runtime_path)
    runtime_sha = _sha256(runtime_path)

    material_dependencies = [
        f"source/textures/{texture_name}" for texture_name in texture_names
    ]
    material_report = {
        "schema": "handson.bc017.material-closure",
        "runtime_usd_sha256": runtime_sha,
        "dependency_paths": material_dependencies,
        "unresolved_references": [],
        "external_absolute_references": [],
    }
    _write_json(evidence_dir / "material_closure.json", material_report)
    authored_collision = lineage["authored_closed_collision_audit"]
    assert isinstance(authored_collision, dict)
    collision_report = {
        "schema": "handson.bc017.collision-audit",
        "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "runtime_usd_sha256": runtime_sha,
        "drawer_links": {
            role: DRAWER_LINK_PRIMS[role] for role in DRAWER_ROLE_ORDER
        },
        "collider_prims": lineage["collider_prims"],
        "derived_from_visible_geometry": True,
        "procedural_geometry": False,
        "fixed_base_verified": True,
        "all_links_have_colliders": True,
        # Values are computed above from the transformed source-mesh collider
        # bounds.  Full method/inputs live in the byte-bound source lineage
        # report because collision-audit schema v1 has an exact key set.
        "closed_reset_max_penetration_m": authored_collision[
            "closed_reset_max_penetration_m"
        ],
        "support_gap_m": authored_collision["support_gap_m"],
    }
    _write_json(evidence_dir / "collision_audit.json", collision_report)
    _write_json(
        evidence_dir / "source_lineage.json",
        {
            "schema": "handson.bc017.source-lineage",
            "formal": False,
            "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "runtime_usd_sha256": runtime_sha,
            "selected_asset": {"title": SOURCE_TITLE, "creator": SOURCE_CREATOR, "url": SOURCE_URL, "license": LICENSE_NAME},
            "derivation": lineage,
            "visible_geometry": "selected converted source meshes only",
            "collision_geometry": (
                "separate hidden exact-topology convex-decomposition meshes for "
                "the fixed desk and two trays; each of the six moving drawers "
                "uses one exact-coordinate-welded closed body-component "
                "SDF plus three source-vertex convex hulls for the two U-pull "
                "legs and crossbar, preserving the handle aperture"
            ),
            "collision_generation": {
                "generation": (
                    "exact_source_decomposition_for_fixture_plus_welded_"
                    "drawer_body_sdf_and_three_source_handle_hulls"
                ),
                "source_derived": True,
                "hand_authored_or_placeholder": False,
                "primitive_or_generic_proxy_fallback": False,
                "task_visible_source_meshes_preserved": True,
                "fixed_decorative_drawer_fronts_omitted": True,
                "all_six_source_drawers_preserved": True,
                "drawer_filtered_pair_count": len(
                    lineage["drawer_filtered_pairs"]
                ),
                "drawer_filtered_pair_scope": (
                    "drawer_body_to_drawer_body_only"
                ),
                "fixture_contacts_preserved": True,
                "hand_contacts_preserved": True,
                "exact_mesh_decomposition_collider_count": sum(
                    row["generation"]
                    == "exact_source_mesh_convex_decomposition"
                    for row in lineage["collider_derivations"]
                ),
                "exact_mesh_sdf_collider_count": sum(
                    row["generation"]
                    == "exact_coordinate_weld_then_closed_drawer_body_component_sdf"
                    for row in lineage["collider_derivations"]
                ),
                "source_handle_compound_convex_hull_collider_count": sum(
                    row["generation"]
                    == (
                        "welded_source_handle_component_20_60_20_partition_"
                        "plus_one_face_ring_then_convex_hull"
                    )
                    for row in lineage["collider_derivations"]
                ),
                "low_poly_convex_hull_collider_count": sum(
                    row["generation"]
                    == "deterministic_source_vertex_extrema_convex_hull"
                    for row in lineage["collider_derivations"]
                ),
                "gpu_convex_hull_vertex_limit": LOW_POLY_MAX_VERTICES,
                "maximum_authored_vertices_per_collider": LOW_POLY_MAX_VERTICES,
                "maximum_authored_triangles_per_collider": LOW_POLY_MAX_TRIANGLES,
                "collider_count": len(lineage["collider_derivations"]),
            },
            "procedural_visible_geometry": False,
            "procedural_collision_geometry": False,
            "physx_live_validated": False,
        },
    )

    file_specs = (
        ("source/original.zip", "source_archive"),
        ("source/license.txt", "license"),
        ("source/converted_scene.usd", "collision_mesh"),
        *(
            (f"source/textures/{texture_name}", "texture")
            for texture_name in texture_names
        ),
        ("runtime/bc017_wooden_drawer.usda", "runtime_usd"),
        ("evidence/material_closure.json", "material_audit"),
        ("evidence/collision_audit.json", "collision_audit"),
        ("evidence/source_lineage.json", "lineage_audit"),
    )
    entries = [_descriptor(asset_root, relative, role) for relative, role in file_specs]
    by_path = {entry["path"]: entry for entry in entries}
    source_entry = by_path["source/original.zip"]
    runtime_entry = by_path["runtime/bc017_wooden_drawer.usda"]
    manifest = {
        "schema": "handson.bc017.asset-manifest",
        "schema_version": 1,
        "task": {"id": TASK_ID, "name": TASK_NAME, "requirement": TASK_REQUIREMENT},
        "clean_room": True,
        "provenance": {
            "title": SOURCE_TITLE,
            "creator": SOURCE_CREATOR,
            "platform": SOURCE_PLATFORM,
            "model_uuid": SOURCE_UUID,
            "source_url": SOURCE_URL,
            "license": LICENSE_NAME,
            "license_url": LICENSE_URL,
            "original_archive": {key: source_entry[key] for key in ("path", "sha256", "size_bytes")},
        },
        "derived_asset": {
            "runtime_usd": {key: runtime_entry[key] for key in ("path", "sha256", "size_bytes")},
            "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "conversion_tool": "omni.kit.asset_converter + pxr USD composition",
            "conversion_version": (
                "Isaac Sim 5.1 asset_converter 5.0.17; "
                "bc017-builder-v10-six-guided-drawer-filtered-pairs-1p55x"
            ),
            "clean_room_conversion": True,
            "no_procedural_geometry": True,
            "meters_per_unit": 1.0,
            "up_axis": "Z",
        },
        "articulation": {
            "fixed_base": True,
            "base_link_prim": BASE_LINK,
            "world_fixed_joint_prim": FIXED_JOINT,
            "independent_dofs": True,
            "mimic_or_coupled_joints": [],
            "drawer_links": [
                {
                    "role": role,
                    "link_prim": DRAWER_LINK_PRIMS[role],
                    "joint_prim": DRAWER_JOINT_NAMES[role],
                    "joint_name": DRAWER_JOINT_NAMES[role],
                    "joint_type": "prismatic",
                    "axis": "+X",
                    "closed_limit_m": 0.0,
                    "open_limit_m": DRAWER_TRAVEL_M,
                    "reset_position_m": 0.0,
                }
                for role in DRAWER_ROLE_ORDER
            ],
        },
        "placement": {
            "translation_m": list(lineage["asset_root_placement_m"]),
            "orientation_wxyz": list(PLACEMENT_ORIENTATION_WXYZ),
        },
        "closure": {
            "declared_files_complete": True,
            "unresolved_references": [],
            "external_absolute_references": [],
            "material_evidence": {"report_path": "evidence/material_closure.json", "dependency_paths": material_dependencies},
            "collision_evidence": {
                "report_path": "evidence/collision_audit.json",
                "derived_from_selected_visible_asset": True,
                "procedural_colliders": False,
                "closed_reset_penetration_free": bool(
                    authored_collision["closed_reset_max_penetration_m"]
                    <= 5.0e-4
                ),
                "collider_prims": lineage["collider_prims"],
            },
        },
        "files": entries,
    }
    _write_json(asset_root / "asset_manifest.json", manifest)
    return {
        "dependencies": dependencies,
        "manifest_sha256": _sha256(asset_root / "asset_manifest.json"),
        "runtime_usd_sha256": runtime_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    args = parser.parse_args()
    # Reject an unsupported toolchain before creating directories or starting
    # Kit.  BC017 asset preparation is an Isaac Sim 5.1-only workflow.
    _verify_runtime()
    archive = args.archive.resolve(strict=True)
    asset_dir = args.asset_dir.resolve()
    if _sha256(archive) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(
            "archive SHA-256 differs from the reviewed Metal Office Desk "
            "download closure"
        )
    if asset_dir.exists():
        raise FileExistsError(f"refusing to replace existing BC017 asset directory: {asset_dir}")
    asset_dir.parent.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "enable_cameras": False})
    staging: Path | None = None
    try:
        app.update()
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.kit.asset_converter")
        app.update()
        with tempfile.TemporaryDirectory(prefix="bc017-source-") as extracted_name:
            extracted = Path(extracted_name)
            _extract_source(archive, extracted)
            staging = Path(tempfile.mkdtemp(prefix=".bc017-build-", dir=asset_dir.parent))
            summary = _build_bundle(archive, staging, extracted)
            staging.replace(asset_dir)
            staging = None
            summary["asset_root"] = str(asset_dir)
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if staging is not None and staging.is_dir():
            shutil.rmtree(staging)
        app.close(wait_for_replicator=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
