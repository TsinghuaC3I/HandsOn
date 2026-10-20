#!/usr/bin/env python3
"""Audit whether the original YCB cup meshes can support YCB04.

This is a deliberately simulator-independent first gate.  It treats every
shipped collision component at scale 1.0, projects the upright geometry into
an axisymmetric radial section, and searches for a collision-clear shallow
nesting pose plus a vertical extraction path.  PhysX remains the authoritative
second gate because the real meshes have discrete facets and contact shells.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.spatial import ConvexHull


_CUP_ORDER = ("a_cups", "b_cups", "c_cups")
_COLLIDER_INDEX = re.compile(r"Collider(\d+)_m\.obj$")


@dataclass(frozen=True)
class SectionComponent:
    name: str
    polygon: np.ndarray

    @property
    def z_bounds(self) -> tuple[float, float]:
        return float(self.polygon[:, 1].min()), float(self.polygon[:, 1].max())

    def interval_at(self, z: float, tolerance: float = 1.0e-10) -> tuple[float, float] | None:
        z_min, z_max = self.z_bounds
        if z < z_min - tolerance or z > z_max + tolerance:
            return None

        radii: list[float] = []
        for start, end in zip(self.polygon, np.roll(self.polygon, -1, axis=0)):
            z0, z1 = float(start[1]), float(end[1])
            if abs(z1 - z0) <= tolerance:
                if abs(z - z0) <= tolerance:
                    radii.extend((float(start[0]), float(end[0])))
                continue
            fraction = (z - z0) / (z1 - z0)
            if -tolerance <= fraction <= 1.0 + tolerance:
                radii.append(float(start[0] + fraction * (end[0] - start[0])))
        if len(radii) < 2:
            return None
        return min(radii), max(radii)


@dataclass(frozen=True)
class CupSection:
    name: str
    components: tuple[SectionComponent, ...]
    xyz_min: np.ndarray
    xyz_max: np.ndarray

    def intervals_at(self, z: float) -> tuple[tuple[float, float], ...]:
        intervals = [
            interval
            for component in self.components
            if (interval := component.interval_at(z)) is not None
        ]
        if not intervals:
            return ()
        intervals.sort()
        merged: list[list[float]] = []
        for lower, upper in intervals:
            if not merged or lower > merged[-1][1] + 1.0e-9:
                merged.append([lower, upper])
            else:
                merged[-1][1] = max(merged[-1][1], upper)
        return tuple((float(lower), float(upper)) for lower, upper in merged)


@dataclass(frozen=True)
class SectionRaster:
    """Fixed-phase local-z section table used for fast translated overlap tests."""

    index_offset: int
    lower: np.ndarray
    upper: np.ndarray
    z_step: float

    @property
    def index_stop(self) -> int:
        return self.index_offset + int(self.lower.shape[0])


def _load_vertices(path: Path) -> np.ndarray:
    if path.read_bytes()[:64].startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(f"Git LFS object is not materialized: {path}")
    vertices = [
        [float(value) for value in line.split()[1:4]]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("v ")
    ]
    if len(vertices) < 4:
        raise RuntimeError(f"Collision mesh has fewer than four vertices: {path}")
    return np.asarray(vertices, dtype=np.float64)


def _component_polygon(path: Path, vertices: np.ndarray) -> np.ndarray:
    radial_z = np.column_stack((np.linalg.norm(vertices[:, :2], axis=1), vertices[:, 2]))
    match = _COLLIDER_INDEX.search(path.name)
    if match is None:
        raise RuntimeError(f"Unexpected collision mesh name: {path.name}")

    # Collider13 and Collider26 are the two full circular base plates.  Their
    # OBJ boundary vertices lie on the circumference, so explicitly include
    # the axis to represent the solid plate.  Collider27 is the rolled rim and
    # must retain its central opening.
    if int(match.group(1)) in {13, 26}:
        z_min = float(vertices[:, 2].min())
        z_max = float(vertices[:, 2].max())
        radial_z = np.vstack((radial_z, (0.0, z_min), (0.0, z_max)))

    hull = ConvexHull(radial_z)
    return radial_z[hull.vertices]


def load_cup(asset_root: Path, name: str) -> CupSection:
    collision_dir = asset_root / name / "collision"
    paths = sorted(
        collision_dir.glob(f"{name}_Collider*_m.obj"),
        key=lambda path: int(_COLLIDER_INDEX.search(path.name).group(1)),
    )
    if len(paths) != 27:
        raise RuntimeError(f"{name}: expected 27 original colliders, found {len(paths)}")

    components: list[SectionComponent] = []
    all_vertices: list[np.ndarray] = []
    for path in paths:
        vertices = _load_vertices(path)
        all_vertices.append(vertices)
        components.append(
            SectionComponent(path.name, _component_polygon(path, vertices))
        )
    combined = np.vstack(all_vertices)
    return CupSection(
        name=name,
        components=tuple(components),
        xyz_min=combined.min(axis=0),
        xyz_max=combined.max(axis=0),
    )


def rasterize(cup: CupSection, z_step: float) -> SectionRaster:
    index_start = int(np.floor(float(cup.xyz_min[2]) / z_step))
    index_stop = int(np.ceil(float(cup.xyz_max[2]) / z_step)) + 1
    rows = [cup.intervals_at(index * z_step) for index in range(index_start, index_stop)]
    width = max(len(row) for row in rows)
    lower = np.full((len(rows), width), np.nan, dtype=np.float64)
    upper = np.full((len(rows), width), np.nan, dtype=np.float64)
    for row_index, intervals in enumerate(rows):
        for column_index, (interval_lower, interval_upper) in enumerate(intervals):
            lower[row_index, column_index] = interval_lower
            upper[row_index, column_index] = interval_upper
    return SectionRaster(index_start, lower, upper, z_step)


def maximum_radial_overlap(
    outer: SectionRaster,
    inner: SectionRaster,
    inner_z_index: int,
) -> float:
    inner_world_start = inner.index_offset + inner_z_index
    inner_world_stop = inner.index_stop + inner_z_index
    common_start = max(outer.index_offset, inner_world_start)
    common_stop = min(outer.index_stop, inner_world_stop)
    if common_start >= common_stop:
        return 0.0

    outer_start = common_start - outer.index_offset
    outer_stop = common_stop - outer.index_offset
    inner_start = common_start - inner_world_start
    inner_stop = common_stop - inner_world_start
    outer_lower = outer.lower[outer_start:outer_stop, :, None]
    outer_upper = outer.upper[outer_start:outer_stop, :, None]
    inner_lower = inner.lower[inner_start:inner_stop, None, :]
    inner_upper = inner.upper[inner_start:inner_stop, None, :]
    overlap = np.minimum(outer_upper, inner_upper) - np.maximum(outer_lower, inner_lower)
    finite = overlap[np.isfinite(overlap)]
    return max(0.0, float(finite.max())) if finite.size else 0.0


def find_shallow_nest(
    outer: CupSection,
    inner: CupSection,
    *,
    search_step: float,
    z_step: float,
    permitted_overlap: float,
) -> dict[str, float]:
    search_stride = search_step / z_step
    if abs(search_stride - round(search_stride)) > 1.0e-9:
        raise ValueError("search step must be an integer multiple of section step")
    search_stride_index = int(round(search_stride))
    outer_raster = rasterize(outer, z_step)
    inner_raster = rasterize(inner, z_step)
    outer_top = float(outer.xyz_max[2])
    inner_bottom = float(inner.xyz_min[2])
    separated_z = outer_top - inner_bottom + 0.005
    separated_index = int(np.ceil(separated_z / z_step))
    minimum_index = int(np.floor(-0.001 / z_step))
    candidates = range(separated_index, minimum_index - 1, -search_stride_index)

    deepest_clear: float | None = None
    deepest_overlap = 0.0
    first_blocked_z: float | None = None
    for candidate_index in candidates:
        candidate = candidate_index * z_step
        overlap = maximum_radial_overlap(outer_raster, inner_raster, candidate_index)
        if overlap <= permitted_overlap:
            deepest_clear = float(candidate)
            deepest_overlap = overlap
            continue
        first_blocked_z = float(candidate)
        break

    if deepest_clear is None:
        raise RuntimeError(f"{inner.name} cannot approach {outer.name} without overlap")

    insertion_depth = outer_top - (inner_bottom + deepest_clear)
    if insertion_depth <= 0.004:
        raise RuntimeError(
            f"{inner.name} only inserts {insertion_depth * 1e3:.2f} mm into "
            f"{outer.name}; this is not a usable nested reset"
        )

    # Every pose along a straight upward extraction path must remain no worse
    # than the accepted reset pose.
    extraction_samples = np.linspace(deepest_clear, separated_index * z_step, 33)
    path_peak = max(
        maximum_radial_overlap(
            outer_raster,
            inner_raster,
            int(round(float(z) / z_step)),
        )
        for z in extraction_samples
    )
    if path_peak > permitted_overlap:
        raise RuntimeError(
            f"{inner.name} has no collision-clear vertical extraction path from {outer.name}"
        )

    return {
        "relative_z_m": deepest_clear,
        "insertion_depth_m": insertion_depth,
        "reset_overlap_m": deepest_overlap,
        "first_blocked_z_m": float("nan") if first_blocked_z is None else first_blocked_z,
        "path_peak_overlap_m": path_peak,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "handson/handson/sim/envs/xmls/lightwheel_ycb"
        ),
        help="Directory containing a_cups, b_cups, and c_cups.",
    )
    parser.add_argument("--search-step-mm", type=float, default=0.05)
    parser.add_argument("--section-step-mm", type=float, default=0.05)
    parser.add_argument("--permitted-overlap-mm", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cups = {name: load_cup(args.asset_root, name) for name in _CUP_ORDER}
    kwargs = {
        "search_step": args.search_step_mm * 1.0e-3,
        "z_step": args.section_step_mm * 1.0e-3,
        "permitted_overlap": args.permitted_overlap_mm * 1.0e-3,
    }
    results = {
        "b_in_c": find_shallow_nest(cups["c_cups"], cups["b_cups"], **kwargs),
        "a_in_b": find_shallow_nest(cups["b_cups"], cups["a_cups"], **kwargs),
    }

    for name in _CUP_ORDER:
        cup = cups[name]
        print(
            f"{name}: bounds_m={cup.xyz_min.tolist()}..{cup.xyz_max.tolist()} "
            f"colliders={len(cup.components)}"
        )
    for relation, result in results.items():
        print(
            f"{relation}: relative_z={result['relative_z_m']:.6f} m, "
            f"insertion={result['insertion_depth_m'] * 1e3:.2f} mm, "
            f"reset_overlap={result['reset_overlap_m'] * 1e3:.3f} mm, "
            f"path_peak_overlap={result['path_peak_overlap_m'] * 1e3:.3f} mm"
        )
    print("YCB04_STATIC_GEOMETRY_GATE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
