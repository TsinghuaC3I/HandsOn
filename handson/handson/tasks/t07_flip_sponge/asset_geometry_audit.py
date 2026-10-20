"""Offline geometry gate for the original-scale YCB08 assets.

This module intentionally has no Isaac, MuJoCo, or third-party geometry
dependency.  It reads the tracked LFS pointers from the task checkout and a
hydrated checkout of the same objects, verifies their hashes, and evaluates
the original convex collision pieces with a separating-axis test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ASSET_NAMES = ("skillet", "spatula", "sponge")
PAN_CENTER_XY_M = (0.0010605, 0.0872140)
MIN_PAN_INNER_RADIUS_M = 0.118
MIN_BLADE_WIDTH_M = 0.075
MAX_BLADE_TIP_THICKNESS_M = 0.010
MIN_SPONGE_PAN_RADIAL_CLEARANCE_M = 0.040


@dataclass(frozen=True)
class Mesh:
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]

    @property
    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return (
            tuple(min(vertex[axis] for vertex in self.vertices) for axis in range(3)),
            tuple(max(vertex[axis] for vertex in self.vertices) for axis in range(3)),
        )


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _unit_axis(vector: Sequence[float]) -> tuple[float, float, float] | None:
    length = _norm(vector)
    if length <= 1.0e-10:
        return None
    axis = tuple(value / length for value in vector)
    for value in axis:
        if abs(value) > 1.0e-9:
            if value < 0.0:
                axis = tuple(-component for component in axis)
            break
    return axis


def load_obj(path: Path) -> Mesh:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append(tuple(float(value) for value in fields[1:4]))
        elif fields[0] == "f" and len(fields) >= 4:
            polygon = [int(value.split("/", 1)[0]) - 1 for value in fields[1:]]
            for index in range(1, len(polygon) - 1):
                faces.append((polygon[0], polygon[index], polygon[index + 1]))
    if not vertices or not faces:
        raise ValueError(f"not a hydrated OBJ mesh: {path}")
    return Mesh(tuple(vertices), tuple(faces))


def mesh_volume(mesh: Mesh) -> float:
    """Return closed-mesh volume using an interior vertex centroid."""
    centre = tuple(
        sum(vertex[axis] for vertex in mesh.vertices) / len(mesh.vertices)
        for axis in range(3)
    )
    volume = 0.0
    for i0, i1, i2 in mesh.faces:
        a = _sub(mesh.vertices[i0], centre)
        b = _sub(mesh.vertices[i1], centre)
        c = _sub(mesh.vertices[i2], centre)
        volume += abs(_dot(a, _cross(b, c))) / 6.0
    return volume


def _mesh_axes(mesh: Mesh) -> tuple[tuple[float, float, float], ...]:
    axes: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    for i0, i1, i2 in mesh.faces:
        a, b, c = mesh.vertices[i0], mesh.vertices[i1], mesh.vertices[i2]
        normal = _unit_axis(_cross(_sub(b, a), _sub(c, a)))
        if normal is not None:
            axes[tuple(round(value * 1.0e6) for value in normal)] = normal
    return tuple(axes.values())


def _mesh_edges(mesh: Mesh) -> tuple[tuple[float, float, float], ...]:
    directions: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    for face in mesh.faces:
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            direction = _unit_axis(_sub(mesh.vertices[end], mesh.vertices[start]))
            if direction is not None:
                directions[tuple(round(value * 1.0e6) for value in direction)] = direction
    return tuple(directions.values())


def _aabb_separated(a: Mesh, b: Mesh, tolerance: float) -> bool:
    a_min, a_max = a.bounds
    b_min, b_max = b.bounds
    return any(
        a_max[axis] < b_min[axis] - tolerance
        or b_max[axis] < a_min[axis] - tolerance
        for axis in range(3)
    )


def convex_penetration_bound(a: Mesh, b: Mesh, tolerance: float = 1.0e-7) -> float | None:
    """Return the minimum SAT overlap, or ``None`` when separated.

    Collision OBJ pieces in the source asset are convex.  Face normals and
    edge cross-products therefore form the complete separating-axis set.
    """
    if _aabb_separated(a, b, tolerance):
        return None
    a_edges = _mesh_edges(a)
    b_edges = _mesh_edges(b)
    axes = list(_mesh_axes(a)) + list(_mesh_axes(b))
    for a_edge in a_edges:
        for b_edge in b_edges:
            axis = _unit_axis(_cross(a_edge, b_edge))
            if axis is not None:
                axes.append(axis)
    minimum_overlap = math.inf
    seen: set[tuple[int, int, int]] = set()
    for axis in axes:
        key = tuple(round(value * 1.0e6) for value in axis)
        if key in seen:
            continue
        seen.add(key)
        a_projection = [_dot(vertex, axis) for vertex in a.vertices]
        b_projection = [_dot(vertex, axis) for vertex in b.vertices]
        overlap = min(max(a_projection), max(b_projection)) - max(
            min(a_projection), min(b_projection)
        )
        if overlap < -tolerance:
            return None
        minimum_overlap = min(minimum_overlap, overlap)
    return minimum_overlap


def _matmul(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(sum(row[index] * vector[index] for index in range(3)) for row in matrix)


def matrix_multiply(
    a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(
            sum(a[row][k] * b[k][column] for k in range(3))
            for column in range(3)
        )
        for row in range(3)
    )


def rotation_x(angle: float) -> tuple[tuple[float, float, float], ...]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine))


def rotation_z(angle: float) -> tuple[tuple[float, float, float], ...]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))


def transform_mesh(
    mesh: Mesh,
    rotation: Sequence[Sequence[float]],
    translation: Sequence[float],
) -> Mesh:
    return Mesh(
        tuple(
            tuple(value + translation[axis] for axis, value in enumerate(_matmul(rotation, vertex)))
            for vertex in mesh.vertices
        ),
        mesh.faces,
    )


def _parse_lfs_oid(path: Path) -> str | None:
    try:
        prefix = path.read_bytes()[:180].decode("ascii")
    except (UnicodeDecodeError, OSError):
        return None
    if not prefix.startswith("version https://git-lfs.github.com/spec/v1\n"):
        return None
    for line in prefix.splitlines():
        if line.startswith("oid sha256:"):
            return line.removeprefix("oid sha256:").strip()
    return None


def verify_lfs_assets(pointer_root: Path, hydrated_root: Path) -> dict[str, object]:
    verified: list[str] = []
    for asset_name in ASSET_NAMES:
        pointer_dir = pointer_root / asset_name
        hydrated_dir = hydrated_root / asset_name
        for pointer_path in sorted(pointer_dir.rglob("*")):
            if not pointer_path.is_file():
                continue
            expected = _parse_lfs_oid(pointer_path)
            if expected is None:
                continue
            relative = pointer_path.relative_to(pointer_root)
            hydrated_path = hydrated_root / relative
            actual = hashlib.sha256(hydrated_path.read_bytes()).hexdigest()
            if actual != expected:
                raise AssertionError(
                    f"hydrated asset hash mismatch for {relative}: {actual} != {expected}"
                )
            verified.append(str(relative))
    if not verified:
        raise AssertionError("no tracked LFS asset payloads were verified")
    return {"verified_lfs_files": verified, "verified_count": len(verified)}


def _load_collision_pieces(asset_root: Path, asset_name: str) -> tuple[Mesh, ...]:
    paths = sorted(
        (asset_root / asset_name / "collision").glob("*.obj"),
        key=lambda path: [
            int(token) if token.isdigit() else token
            for token in re.split(r"(\d+)", path.name)
        ],
    )
    return tuple(load_obj(path) for path in paths)


def _penetration_rows(
    first: Sequence[Mesh], second: Sequence[Mesh]
) -> list[tuple[int, int, float]]:
    rows: list[tuple[int, int, float]] = []
    for first_index, first_piece in enumerate(first, start=1):
        for second_index, second_piece in enumerate(second, start=1):
            penetration = convex_penetration_bound(first_piece, second_piece)
            if penetration is not None and penetration > 2.0e-4:
                rows.append((first_index, second_index, penetration))
    return rows


def run_geometry_audit(pointer_root: Path, hydrated_root: Path) -> dict[str, object]:
    hashes = verify_lfs_assets(pointer_root, hydrated_root)
    visuals = {
        name: load_obj(hydrated_root / name / "visuals" / f"{name}.obj")
        for name in ASSET_NAMES
    }
    pieces = {name: _load_collision_pieces(hydrated_root, name) for name in ASSET_NAMES}
    bounds = {name: visuals[name].bounds for name in ASSET_NAMES}
    dimensions = {
        name: tuple(bounds[name][1][axis] - bounds[name][0][axis] for axis in range(3))
        for name in ASSET_NAMES
    }

    pan_bottom = pieces["skillet"][0]
    pan_floor_top = pan_bottom.bounds[1][2]
    pan_wall_vertices = [
        vertex for piece in pieces["skillet"][1:19] for vertex in piece.vertices
    ]
    pan_inner_radius = min(
        math.hypot(vertex[0] - PAN_CENTER_XY_M[0], vertex[1] - PAN_CENTER_XY_M[1])
        for vertex in pan_wall_vertices
    )
    pan_rim_top = max(vertex[2] for vertex in pan_wall_vertices)

    spatula_vertices = [vertex for piece in pieces["spatula"] for vertex in piece.vertices]
    blade_vertices = [vertex for vertex in spatula_vertices if vertex[1] >= 0.060]
    tip_vertices = [vertex for vertex in spatula_vertices if vertex[1] >= 0.150]
    blade_width = max(vertex[0] for vertex in blade_vertices) - min(
        vertex[0] for vertex in blade_vertices
    )
    tip_thickness = max(vertex[2] for vertex in tip_vertices) - min(
        vertex[2] for vertex in tip_vertices
    )

    sponge_half_x = 0.5 * dimensions["sponge"][0]
    sponge_half_y = 0.5 * dimensions["sponge"][1]
    sponge_half_z = 0.5 * dimensions["sponge"][2]
    sponge_corner_radius = math.hypot(sponge_half_x, sponge_half_y)
    radial_clearance = pan_inner_radius - sponge_corner_radius

    reset_sponge_rows: list[dict[str, object]] = []
    for offset_x in (-0.010, 0.0, 0.010):
        for offset_y in (-0.008, 0.0, 0.008):
            for yaw_deg in (-8.0, 0.0, 8.0):
                sponge_rotation = rotation_z(math.radians(yaw_deg))
                sponge_translation = (
                    PAN_CENTER_XY_M[0] + offset_x,
                    PAN_CENTER_XY_M[1] + offset_y,
                    pan_floor_top + sponge_half_z + 0.0005,
                )
                transformed_sponge = [
                    transform_mesh(piece, sponge_rotation, sponge_translation)
                    for piece in pieces["sponge"]
                ]
                penetrations = _penetration_rows(
                    pieces["skillet"], transformed_sponge
                )
                reset_sponge_rows.append(
                    {
                        "offset_x_m": offset_x,
                        "offset_y_m": offset_y,
                        "yaw_deg": yaw_deg,
                        "penetration_count": len(penetrations),
                        "max_penetration_m": max(
                            (row[2] for row in penetrations), default=0.0
                        ),
                        "collision_clear": not penetrations,
                    }
                )

    # Fit the broad blade's lower envelope, then pitch it to approximately
    # level.  This is a geometric candidate only; it never becomes an assist.
    y_min = min(vertex[1] for vertex in blade_vertices)
    y_max = max(vertex[1] for vertex in blade_vertices)
    envelope: list[tuple[float, float]] = []
    for bin_index in range(8):
        lower = y_min + (y_max - y_min) * bin_index / 8.0
        upper = y_min + (y_max - y_min) * (bin_index + 1) / 8.0
        bin_vertices = [vertex for vertex in blade_vertices if lower <= vertex[1] <= upper]
        if bin_vertices:
            envelope.append(
                (
                    sum(vertex[1] for vertex in bin_vertices) / len(bin_vertices),
                    min(vertex[2] for vertex in bin_vertices),
                )
            )
    y_mean = sum(point[0] for point in envelope) / len(envelope)
    z_mean = sum(point[1] for point in envelope) / len(envelope)
    slope = sum((y - y_mean) * (z - z_mean) for y, z in envelope) / sum(
        (y - y_mean) ** 2 for y, _ in envelope
    )
    pitch = -math.atan(slope)
    candidate_rows: list[dict[str, object]] = []
    pan_penetration_cache: dict[
        tuple[float, float, float, float], list[tuple[int, int, float]]
    ] = {}
    lifted_sponge_cache: dict[tuple[float, float], list[Mesh]] = {}
    candidate_pitches = (
        pitch,
        math.radians(-22.0),
        math.radians(-28.0),
        math.radians(-34.0),
        math.radians(-40.0),
    )
    for candidate_pitch in candidate_pitches:
        rotation = matrix_multiply(rotation_z(math.pi), rotation_x(candidate_pitch))
        rotated_blade_z = [_matmul(rotation, vertex)[2] for vertex in blade_vertices]
        blade_top_local_z = max(rotated_blade_z)
        for blade_floor_clearance in (0.001, 0.004, 0.007, 0.010, 0.015):
            blade_translation_z = (
                pan_floor_top + blade_floor_clearance - min(rotated_blade_z)
            )
            transformed_spatula_pieces_base = [
                transform_mesh(piece, rotation, (0.0, 0.0, blade_translation_z))
                for piece in pieces["spatula"]
            ]
            for sponge_offset_x in (-0.010, 0.0, 0.010):
                for sponge_lift_m in (0.012, 0.020, 0.030):
                    sponge_centre = (
                        PAN_CENTER_XY_M[0] + sponge_offset_x,
                        PAN_CENTER_XY_M[1],
                        pan_floor_top
                        + sponge_half_z
                        + 0.0005
                        + sponge_lift_m,
                    )
                    sponge_cache_key = (sponge_offset_x, sponge_lift_m)
                    if sponge_cache_key not in lifted_sponge_cache:
                        lifted_sponge_cache[sponge_cache_key] = [
                            transform_mesh(piece, rotation_z(0.0), sponge_centre)
                            for piece in pieces["sponge"]
                        ]
                    transformed_sponge = lifted_sponge_cache[sponge_cache_key]
                    for overlap_fraction in (
                        0.20,
                        0.35,
                        0.50,
                        0.65,
                        0.80,
                        0.95,
                    ):
                        insertion = overlap_fraction * dimensions["sponge"][1]
                        spatula_origin_y = (
                            sponge_centre[1]
                            + sponge_half_y
                            - insertion
                            + y_max
                        )
                        spatula_origin_x = sponge_centre[0]
                        translated_spatula = [
                            transform_mesh(
                                piece,
                                (
                                    (1.0, 0.0, 0.0),
                                    (0.0, 1.0, 0.0),
                                    (0.0, 0.0, 1.0),
                                ),
                                (spatula_origin_x, spatula_origin_y, 0.0),
                            )
                            for piece in transformed_spatula_pieces_base
                        ]
                        pan_cache_key = (
                            candidate_pitch,
                            blade_floor_clearance,
                            sponge_offset_x,
                            overlap_fraction,
                        )
                        if pan_cache_key not in pan_penetration_cache:
                            pan_penetration_cache[pan_cache_key] = (
                                _penetration_rows(
                                    pieces["skillet"], translated_spatula
                                )
                            )
                        pan_penetration_rows = pan_penetration_cache[
                            pan_cache_key
                        ]
                        sponge_penetration_rows = _penetration_rows(
                            transformed_sponge, translated_spatula
                        )
                        blade_sponge_vertical_clearance = (
                            sponge_centre[2]
                            - sponge_half_z
                            - blade_translation_z
                            - blade_top_local_z
                        )
                        candidate_rows.append(
                            {
                                "spatula_pitch_deg": math.degrees(candidate_pitch),
                                "blade_floor_clearance_m": blade_floor_clearance,
                                "sponge_offset_x_m": sponge_offset_x,
                                "sponge_lift_m": sponge_lift_m,
                                "blade_overlap_fraction": overlap_fraction,
                                "blade_sponge_vertical_clearance_m": (
                                    blade_sponge_vertical_clearance
                                ),
                                "spatula_pan_penetration_count": len(
                                    pan_penetration_rows
                                ),
                                "max_spatula_pan_penetration_m": max(
                                    (row[2] for row in pan_penetration_rows),
                                    default=0.0,
                                ),
                                "spatula_sponge_penetration_count": len(
                                    sponge_penetration_rows
                                ),
                                "max_spatula_sponge_penetration_m": max(
                                    (row[2] for row in sponge_penetration_rows),
                                    default=0.0,
                                ),
                                "collision_clear": bool(
                                    not pan_penetration_rows
                                    and not sponge_penetration_rows
                                    and blade_sponge_vertical_clearance >= -2.0e-4
                                ),
                            }
                        )

    clear_candidates = [row for row in candidate_rows if row["collision_clear"]]
    checks = {
        "asset_hashes_match": hashes["verified_count"] == 69,
        "pan_is_hollow": pan_inner_radius >= MIN_PAN_INNER_RADIUS_M,
        "sponge_fits_pan": radial_clearance >= MIN_SPONGE_PAN_RADIAL_CLEARANCE_M,
        "reset_sponge_sweep_is_collision_clear": all(
            row["collision_clear"] for row in reset_sponge_rows
        ),
        "blade_is_wide_enough": blade_width >= MIN_BLADE_WIDTH_M,
        "blade_tip_is_thin_enough": tip_thickness <= MAX_BLADE_TIP_THICKNESS_M,
        "under_sponge_sweep_has_clear_pose": len(clear_candidates) >= 3,
        "under_sponge_sweep_covers_offsets": len(
            {row["sponge_offset_x_m"] for row in clear_candidates}
        )
        == 3,
        "under_sponge_sweep_covers_lift_at_or_above_threshold": any(
            row["sponge_lift_m"] >= 0.020 for row in clear_candidates
        ),
    }
    densities = {"skillet": 1001.55, "spatula": 554.977, "sponge": 1000.0}
    collision_volumes = {
        name: sum(mesh_volume(piece) for piece in pieces[name]) for name in ASSET_NAMES
    }
    return {
        **hashes,
        "passed": all(checks.values()),
        "checks": checks,
        "visual_dimensions_m": dimensions,
        "collision_piece_counts": {name: len(pieces[name]) for name in ASSET_NAMES},
        "collision_volumes_m3": collision_volumes,
        "source_density_mass_estimates_kg": {
            name: collision_volumes[name] * densities[name] for name in ASSET_NAMES
        },
        "pan_floor_top_local_z_m": pan_floor_top,
        "pan_rim_top_local_z_m": pan_rim_top,
        "pan_inner_radius_lower_bound_m": pan_inner_radius,
        "sponge_pan_radial_clearance_m": radial_clearance,
        "blade_width_m": blade_width,
        "blade_tip_thickness_m": tip_thickness,
        "blade_level_pitch_deg": math.degrees(pitch),
        "reset_sponge_sweep": reset_sponge_rows,
        "candidate_sweep": candidate_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointer-root", type=Path, required=True)
    parser.add_argument("--hydrated-root", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_geometry_audit(arguments.pointer_root, arguments.hydrated_root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
