"""Static audit for the original-size YCB toy-airplane asset used by YCB03."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

TASK_NAME = "t35_rotate_propeller"
ASSET_RELATIVE_PATH = Path("handson/sim/envs/xmls/lightwheel_ycb/a_toy_airplane")
PROPELLER_JOINT = "a_toy_airplane_Propeller_joint"
PROPELLER_COLLIDERS = (
    "a_toy_airplane_Propeller_Collider01_m.obj",
    "a_toy_airplane_Propeller_Collider02_m.obj",
    "a_toy_airplane_Propeller_Collider03_m.obj",
    "a_toy_airplane_Propeller_Collider04_m.obj",
)
FUSELAGE_COLLIDERS = tuple(
    f"a_toy_airplane_Collider{index:02d}_m.obj" for index in range(1, 10)
)
WHEEL_COLLIDERS = (
    "a_toy_airplane_FL_Wheel_Collider01_m.obj",
    "a_toy_airplane_FR_Wheel_Collider01_m.obj",
    "a_toy_airplane_RL_Wheel_Collider01_m.obj",
    "a_toy_airplane_RR_Wheel_Collider01_m.obj",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_materialized(path: Path) -> None:
    head = path.read_bytes()[:80]
    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(f"Git LFS asset is not materialized: {path}")


def _load_obj(path: Path) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    _assert_materialized(path)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        fields = raw_line.split()
        if not fields:
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append(tuple(float(value) for value in fields[1:4]))
        elif fields[0] == "f" and len(fields) >= 4:
            indices = tuple(int(value.split("/", 1)[0]) - 1 for value in fields[1:])
            faces.append(indices)
    if not vertices or not faces:
        raise ValueError(f"OBJ has no usable mesh data: {path}")
    return np.asarray(vertices, dtype=np.float64), tuple(faces)


def _rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    x, y, z = unit
    cosine = math.cos(angle)
    sine = math.sin(angle)
    complement = 1.0 - cosine
    return np.asarray(
        [
            [
                cosine + x * x * complement,
                x * y * complement - z * sine,
                x * z * complement + y * sine,
            ],
            [
                y * x * complement + z * sine,
                cosine + y * y * complement,
                y * z * complement - x * sine,
            ],
            [
                z * x * complement - y * sine,
                z * y * complement + x * sine,
                cosine + z * z * complement,
            ],
        ],
        dtype=np.float64,
    )


def _rotate_vertices(
    vertices: np.ndarray,
    pivot: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    rotation = _rotation_about_axis(axis, angle)
    return (vertices - pivot) @ rotation.T + pivot


def _mesh_axes(
    vertices: np.ndarray,
    faces: tuple[tuple[int, ...], ...],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    normals: list[np.ndarray] = []
    edges: list[np.ndarray] = []
    seen_edges: set[tuple[int, int]] = set()
    for face in faces:
        anchor = vertices[face[0]]
        for index in range(1, len(face) - 1):
            normal = np.cross(
                vertices[face[index]] - anchor,
                vertices[face[index + 1]] - anchor,
            )
            norm = float(np.linalg.norm(normal))
            if norm > 1e-10:
                normals.append(normal / norm)
        for index, start in enumerate(face):
            end = face[(index + 1) % len(face)]
            key = (min(start, end), max(start, end))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edge = vertices[end] - vertices[start]
            norm = float(np.linalg.norm(edge))
            if norm > 1e-10:
                edges.append(edge / norm)
    return normals, edges


def _convex_meshes_overlap(
    vertices_a: np.ndarray,
    faces_a: tuple[tuple[int, ...], ...],
    vertices_b: np.ndarray,
    faces_b: tuple[tuple[int, ...], ...],
    *,
    clearance: float = 1e-5,
) -> bool:
    normals_a, edges_a = _mesh_axes(vertices_a, faces_a)
    normals_b, edges_b = _mesh_axes(vertices_b, faces_b)
    candidate_axes = [*normals_a, *normals_b]
    for edge_a in edges_a:
        for edge_b in edges_b:
            axis = np.cross(edge_a, edge_b)
            norm = float(np.linalg.norm(axis))
            if norm > 1e-8:
                candidate_axes.append(axis / norm)
    for axis in candidate_axes:
        projection_a = vertices_a @ axis
        projection_b = vertices_b @ axis
        if (
            float(np.max(projection_a)) <= float(np.min(projection_b)) + clearance
            or float(np.max(projection_b)) <= float(np.min(projection_a)) + clearance
        ):
            return False
    return True


def audit_asset(asset_root: Path) -> dict[str, object]:
    asset_root = asset_root.resolve()
    model_path = asset_root / "model.xml"
    _assert_materialized(model_path)
    xml_root = ET.parse(model_path).getroot()

    mesh_elements = tuple(xml_root.findall("./asset/mesh"))
    scaled_meshes = [mesh.get("name") for mesh in mesh_elements if mesh.get("scale")]
    if scaled_meshes:
        raise AssertionError(f"YCB03 forbids scaled source meshes: {scaled_meshes}")

    propeller_joint = xml_root.find(f".//joint[@name='{PROPELLER_JOINT}']")
    if propeller_joint is None:
        raise AssertionError(f"Missing passive propeller joint {PROPELLER_JOINT}")
    if propeller_joint.get("limited", "false").lower() != "false":
        raise AssertionError("Propeller joint must be unlimited")
    axis = np.fromstring(propeller_joint.attrib["axis"], sep=" ", dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if not np.isclose(axis_norm, 1.0, atol=2e-4):
        raise AssertionError(f"Propeller axis is not unit length: {axis_norm}")
    pivot = np.fromstring(propeller_joint.attrib["pos"], sep=" ", dtype=np.float64)

    propeller_actuator = xml_root.find(
        f"./actuator/general[@joint='{PROPELLER_JOINT}']"
    )
    if propeller_actuator is None:
        raise AssertionError("Missing source propeller actuator record")
    gear = np.fromstring(propeller_actuator.get("gear", ""), sep=" ", dtype=np.float64)
    if gear.size == 0 or not np.allclose(gear, 0.0):
        raise AssertionError("Source propeller actuator must have zero gear")

    collision_root = asset_root / "collision"
    visual_root = asset_root / "visuals"
    propeller_meshes = {
        name: _load_obj(collision_root / name) for name in PROPELLER_COLLIDERS
    }
    fuselage_meshes = {
        name: _load_obj(collision_root / name) for name in FUSELAGE_COLLIDERS
    }
    wheel_meshes = {name: _load_obj(collision_root / name) for name in WHEEL_COLLIDERS}
    visual_meshes = {
        path.name: _load_obj(path)[0] for path in sorted(visual_root.glob("*.obj"))
    }

    all_visual_vertices = np.concatenate(tuple(visual_meshes.values()), axis=0)
    visual_min = np.min(all_visual_vertices, axis=0)
    visual_max = np.max(all_visual_vertices, axis=0)

    all_propeller_vertices = np.concatenate(
        tuple(vertices for vertices, _faces in propeller_meshes.values()), axis=0
    )
    relative_propeller = all_propeller_vertices - pivot
    axial_component = relative_propeller @ (axis / axis_norm)
    radial_component = relative_propeller - np.outer(axial_component, axis / axis_norm)
    propeller_radius = float(np.max(np.linalg.norm(radial_component, axis=1)))
    propeller_diameter = 2.0 * propeller_radius

    wheel_minima = {
        name: float(np.min(vertices[:, 2]))
        for name, (vertices, _faces) in wheel_meshes.items()
    }
    support_root_height = -min(wheel_minima.values())
    wheel_support_gaps = {
        name: float(minimum + support_root_height)
        for name, minimum in wheel_minima.items()
    }
    if max(wheel_support_gaps.values()) > 0.005:
        raise AssertionError(
            f"Landing wheels cannot share table support: {wheel_support_gaps}"
        )

    blade_names = PROPELLER_COLLIDERS[:3]
    sweep_collisions: list[dict[str, object]] = []
    sweep_table_clearances: list[float] = []
    for sample_index in range(72):
        angle = 2.0 * math.pi * sample_index / 72.0
        for propeller_name in blade_names:
            propeller_vertices, propeller_faces = propeller_meshes[propeller_name]
            rotated = _rotate_vertices(propeller_vertices, pivot, axis, angle)
            sweep_table_clearances.append(
                float(np.min(rotated[:, 2]) + support_root_height)
            )
            for fuselage_name, (
                fuselage_vertices,
                fuselage_faces,
            ) in fuselage_meshes.items():
                if _convex_meshes_overlap(
                    rotated,
                    propeller_faces,
                    fuselage_vertices,
                    fuselage_faces,
                ):
                    sweep_collisions.append(
                        {
                            "angle_deg": float(math.degrees(angle)),
                            "propeller": propeller_name,
                            "fuselage": fuselage_name,
                        }
                    )
    if sweep_collisions:
        raise AssertionError(
            "Propeller blade collision meshes intersect the fuselage during the "
            f"sampled full sweep: {sweep_collisions[:8]}"
        )
    minimum_table_clearance = min(sweep_table_clearances)
    if minimum_table_clearance <= 0.0:
        raise AssertionError(
            f"Propeller sweep reaches through the support plane: {minimum_table_clearance}"
        )

    return {
        "task_name": TASK_NAME,
        "asset_root": str(asset_root),
        "model_sha256": _sha256(model_path),
        "mesh_count": len(mesh_elements),
        "scaled_meshes": scaled_meshes,
        "visual_bounds_m": [visual_min.tolist(), visual_max.tolist()],
        "visual_size_m": (visual_max - visual_min).tolist(),
        "propeller_joint": PROPELLER_JOINT,
        "propeller_axis_norm": axis_norm,
        "propeller_passive_zero_gear": True,
        "propeller_diameter_m": propeller_diameter,
        "support_root_height_m": support_root_height,
        "wheel_support_gaps_m": wheel_support_gaps,
        "propeller_sweep_samples": 72,
        "propeller_blade_fuselage_collisions": 0,
        "minimum_propeller_table_clearance_m": minimum_table_clearance,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_asset(args.asset_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
