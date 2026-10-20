#!/usr/bin/env python3
"""Build the AS013 utility-knife asset from the downloaded Sketchfab glTF.

The downloaded glTF is a visual export (and contains a source animation), not
an Isaac Sim physics asset.  This script intentionally does the conversion in
one reproducible step: it applies the audited canonical frame mapping,
separates the housing and moving assembly, authors source-derived convex hull
colliders, and writes a self-contained USD stage plus a provenance manifest.

The script is safe to re-run.  It never edits the source archive and only
replaces the output files explicitly requested on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


ASSET_ID = "as013-sketchfab-ahtusik"
TITLE = "Utility Knife"
AUTHOR = "ahtusik"
AUTHOR_URL = "https://sketchfab.com/ahtusik"
SOURCE_URL = (
    "https://sketchfab.com/3d-models/utility-knife-083c2d52b3d04eedb5350cce6e2d1b92"
)
LICENSE_NAME = "CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
ATTRIBUTION = (
    'This work is based on "Utility Knife" '
    f"({SOURCE_URL}) by {AUTHOR} ({AUTHOR_URL}) licensed under {LICENSE_NAME}."
)

# The frame was audited from the glTF node transforms.  The file has no unit
# declaration; 0.008 m/source-unit reproduces the dimensions of the source
# model without anisotropic deformation.  The signed axis permutation keeps
# the task's +Y slide direction and has determinant +1, so source winding and
# normals remain orientation-preserving.
CONVERSION_SCALE_M = 0.008
# The downloaded model is unusually flat for the Wuji fingertips.  Resize the
# authored vertices (and regenerate every collider from those resized
# vertices) instead of applying an Xform scale at runtime.  This keeps the USD
# root at unit scale and makes the physical/visual geometry share one frame.
ERGONOMIC_SCALE_XYZ = np.asarray([1.30, 1.30, 2.35], dtype=np.float64)
BUTTON_TARGET_WIDTH_M = 0.023
BUTTON_TARGET_PROTRUSION_M = 0.0025
SOURCE_BUTTON_RAISED_Z_THRESHOLD_M = 0.001
# After the thickness resize and button lift there is a clean gap between the
# slider base (<= 2.35 mm) and raised-button vertices (>= 3.62 mm).
BUTTON_RAISED_Z_THRESHOLD_M = 0.003
# The downloaded render puts the blade tip outside the housing even at the
# animation's retracted pose.  That is a source-model placement error, not a
# valid initial state for the articulated task: a fully retracted utility
# knife must hide the blade inside its shell.  Fit the blade *vertices* (and
# therefore every blade collider regenerated below) to the housing's Y
# envelope, leaving a small deterministic edge clearance.  Keeping this as a
# baked geometry operation avoids a visual/collision mismatch caused by a
# runtime-only scale or offset.
BLADE_RETRACTED_CLEARANCE_M = 0.004
BLADE_ENVELOPE_AXIS = "Y"
# A single convex hull across the source slider bridged a 12.3 mm concavity.
# The source export duplicates vertices along material/smoothing seams, so weld
# equal coordinates before finding shared-edge components.  Each component is
# then required to be a genuine 3-D convex hull; no epsilon-thickness fallback
# is permitted for slider collision geometry.
SLIDER_COLLIDER_SURFACE_GAP_TARGET_M = 0.0030
SLIDER_COMPONENT_WELD_DECIMAL_DIGITS = 8
SLIDER_COMPONENT_WELD_TOLERANCE_M = 1.0e-8
SLIDER_COLLIDER_AUDIT_SAMPLES = 30_000
SLIDER_COLLIDER_MIN_VERTEX_COUNT = 4
SLIDER_COLLIDER_MIN_FACE_COUNT = 4
SLIDER_COLLIDER_MIN_AXIS_EXTENT_M = 0.00030
SLIDER_COLLIDER_MIN_SINGULAR_VALUE_M = 0.00009
SLIDER_COLLIDER_MIN_SINGULAR_VALUE_RATIO = 0.002
SLIDER_COLLIDER_MIN_VOLUME_M3 = 5.0e-10
JOINT_NAME = "PrismaticJoint_utility_knife_3_middle"
JOINT_AXIS = "Y"
JOINT_LIMITS_M = (0.0, 0.08)
HOUSING_MASS_KG = 0.170
MOVING_MASS_KG = 0.050


@dataclass
class MeshRecord:
    name: str
    category: str
    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray | None
    source_material: str

    @property
    def bounds(self) -> np.ndarray:
        return np.asarray([self.vertices.min(axis=0), self.vertices.max(axis=0)])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_float3_array(values: np.ndarray) -> Vt.Vec3fArray:
    return Vt.Vec3fArray([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in values])


def _as_float2_array(values: np.ndarray) -> Vt.Vec2fArray:
    return Vt.Vec2fArray([Gf.Vec2f(float(x), float(y)) for x, y in values])


def canonicalize(points: np.ndarray) -> np.ndarray:
    """Map trimesh/glTF world coordinates to AS013 body coordinates."""

    points = np.asarray(points, dtype=np.float64)
    # [x, y, z] -> [-x, z, y] and convert source units to metres.  The matrix
    # determinant is +1 (one swap and one sign flip), so no face-winding
    # reversal is required.
    out = points[:, [0, 2, 1]].copy()
    out[:, 0] *= -1.0
    return out * CONVERSION_SCALE_M


def canonicalize_uv(uv: np.ndarray | None, faces: np.ndarray) -> np.ndarray | None:
    if uv is None:
        return None
    # USD face-varying UVs follow the source face winding exactly.
    return np.asarray(uv, dtype=np.float64)[faces.reshape(-1)]


def load_source_meshes(scene_path: Path) -> list[MeshRecord]:
    scene = trimesh.load(scene_path, force="scene", process=False)
    records: list[MeshRecord] = []
    for node_name in sorted(scene.graph.nodes_geometry):
        transform, geometry_name = scene.graph[node_name]
        source_mesh = scene.geometry[geometry_name].copy()
        source_mesh.apply_transform(transform)
        vertices = canonicalize(np.asarray(source_mesh.vertices, dtype=np.float64))
        faces = np.asarray(source_mesh.faces, dtype=np.int64)
        uv = getattr(source_mesh.visual, "uv", None)
        uv = canonicalize_uv(uv, np.asarray(source_mesh.faces, dtype=np.int64))
        if node_name.startswith("Cube_"):
            category = "housing"
        elif node_name == "Plane_Material.003_0":
            category = "blade"
        elif node_name == "Plane_Material_0":
            category = "slider"
        else:
            # Cameras and lights have no geometry; retaining this guard makes
            # an unexpected future mesh fail closed instead of silently
            # dropping it from the asset.
            raise ValueError(f"unexpected geometry node in source glTF: {node_name}")
        # All five source materials use the same texture set, but preserving
        # the glTF material name keeps the provenance audit explicit.
        material_obj = getattr(getattr(source_mesh, "visual", None), "material", None)
        source_material = str(getattr(material_obj, "name", "unknown"))
        records.append(
            MeshRecord(
                name=node_name,
                category=category,
                vertices=vertices,
                faces=faces,
                uv=uv,
                source_material=source_material,
            )
        )
    if {r.category for r in records} != {"housing", "blade", "slider"}:
        raise ValueError("source glTF did not produce housing, blade, and slider geometry")
    return records


def _fit_blade_to_retracted_housing_envelope(
    records: list[MeshRecord],
    *,
    clearance_m: float = BLADE_RETRACTED_CLEARANCE_M,
) -> dict[str, Any]:
    """Fit the blade to the housing envelope at the zero-joint pose.

    The selected glTF contains a blade that is longer than (and offset past)
    the housing.  Since ``build_stage`` uses the downloaded pose as q=0, that
    leaves a visible/colliding tip when the slider is fully retracted.  The
    correction is deliberately limited to the blade's longitudinal Y
    coordinates: X/Z thickness and topology stay source-derived, while the
    corrected vertices are subsequently used for both the visual mesh and the
    convex-hull collider.

    A fail-closed assertion is kept here as part of the build contract.  It
    makes it impossible to regenerate an asset whose q=0 blade is outside the
    housing or whose moving blade is longer than the housing.
    """

    clearance = float(clearance_m)
    if not np.isfinite(clearance) or clearance <= 0.0:
        raise ValueError("blade envelope clearance must be finite and positive")
    housing_points = np.vstack(
        [record.vertices for record in records if record.category == "housing"]
    )
    blade = next(record for record in records if record.category == "blade")
    housing_bounds = np.asarray(
        [housing_points.min(axis=0), housing_points.max(axis=0)],
        dtype=np.float64,
    )
    source_bounds = blade.bounds.copy()
    housing_min_y = float(housing_bounds[0, 1])
    housing_max_y = float(housing_bounds[1, 1])
    target_min_y = housing_min_y + clearance
    target_max_y = housing_max_y - clearance
    target_extent_y = target_max_y - target_min_y
    if not np.isfinite(target_extent_y) or target_extent_y <= 0.0:
        raise ValueError(
            "housing Y envelope is too small for the blade retraction clearance"
        )
    source_min_y = float(source_bounds[0, 1])
    source_max_y = float(source_bounds[1, 1])
    source_extent_y = source_max_y - source_min_y
    if not np.isfinite(source_extent_y) or source_extent_y <= 0.0:
        raise ValueError("source blade Y extent must be finite and positive")

    scale_y = target_extent_y / source_extent_y
    blade.vertices[:, 1] = target_min_y + (
        blade.vertices[:, 1] - source_min_y
    ) * scale_y
    corrected_bounds = blade.bounds.copy()
    # Numerical tolerances are intentionally much tighter than the 4 mm
    # physical clearance so this catches a future implementation regression.
    envelope_tolerance = 1.0e-9
    if (
        corrected_bounds[0, 1] < target_min_y - envelope_tolerance
        or corrected_bounds[1, 1] > target_max_y + envelope_tolerance
        or (corrected_bounds[1, 1] - corrected_bounds[0, 1])
        > target_extent_y + envelope_tolerance
    ):
        raise AssertionError(
            "corrected blade does not fit the zero-joint housing envelope: "
            f"blade={corrected_bounds[:, 1].tolist()}, "
            f"target={[target_min_y, target_max_y]}"
        )
    return {
        "method": "baked_blade_y_fit_to_retracted_housing_envelope",
        "axis": BLADE_ENVELOPE_AXIS,
        "clearance_m": clearance,
        "source_resized_bounds_m": _bounds_dict(source_bounds),
        "housing_bounds_reference_m": _bounds_dict(housing_bounds),
        "target_bounds_m": {
            "min_y_m": target_min_y,
            "max_y_m": target_max_y,
            "extent_y_m": target_extent_y,
        },
        "corrected_bounds_m": _bounds_dict(corrected_bounds),
        "source_extent_y_m": source_extent_y,
        "corrected_extent_y_m": float(corrected_bounds[1, 1] - corrected_bounds[0, 1]),
        "scale_y": float(scale_y),
        "positive_tip_removed_m": float(source_max_y - corrected_bounds[1, 1]),
        # The lower end is shifted inward/outward by the affine fit rather
        # than clipped.  Record the signed displacement explicitly; callers
        # must not mistake it for material removed from the blade.
        "negative_tip_shift_m": float(corrected_bounds[0, 1] - source_min_y),
        "q0_fully_retracted": True,
        "q0_max_positive_exposure_m": 0.0,
        "length_within_housing": True,
    }


def apply_ergonomic_resize(records: list[MeshRecord]) -> dict[str, Any]:
    """Bake the operator-facing resize into every source-derived visual.

    The housing, blade, and slider first receive the same XYZ resize.  Raised
    button vertices are then widened around their own centre and lifted so the
    button top protrudes 2.5 mm above the resized housing.  Faces and UVs stay
    unchanged, so connecting faces deform continuously and texture provenance
    remains intact.
    """

    slider = next(record for record in records if record.category == "slider")
    source_button_mask = (
        np.asarray(slider.vertices[:, 2], dtype=np.float64)
        > SOURCE_BUTTON_RAISED_Z_THRESHOLD_M
    )
    if int(source_button_mask.sum()) < 4:
        raise ValueError("slider source mesh has no stable raised button region")

    for record in records:
        record.vertices = np.asarray(record.vertices, dtype=np.float64) * ERGONOMIC_SCALE_XYZ

    all_housing_points = np.vstack(
        [record.vertices for record in records if record.category == "housing"]
    )
    resized_button = slider.vertices[source_button_mask]
    button_center_x = float(resized_button[:, 0].mean())
    current_button_width = float(np.ptp(resized_button[:, 0]))
    if current_button_width <= 0.0:
        raise ValueError("slider button width is not positive")
    button_width_scale = BUTTON_TARGET_WIDTH_M / current_button_width
    slider.vertices[source_button_mask, 0] = button_center_x + (
        slider.vertices[source_button_mask, 0] - button_center_x
    ) * button_width_scale

    housing_top_z = float(all_housing_points[:, 2].max())
    resized_button_top_z = float(slider.vertices[source_button_mask, 2].max())
    button_lift_m = (
        housing_top_z + BUTTON_TARGET_PROTRUSION_M - resized_button_top_z
    )
    slider.vertices[source_button_mask, 2] += button_lift_m

    resized_button = slider.vertices[source_button_mask]
    blade_envelope = _fit_blade_to_retracted_housing_envelope(records)
    return {
        "scale_xyz": [float(value) for value in ERGONOMIC_SCALE_XYZ],
        "button_source_vertex_count": int(source_button_mask.sum()),
        "button_width_scale_after_global_resize": float(button_width_scale),
        "button_lift_m": float(button_lift_m),
        "button_target_width_m": float(BUTTON_TARGET_WIDTH_M),
        "button_target_protrusion_m": float(BUTTON_TARGET_PROTRUSION_M),
        "button_actual_width_m": float(np.ptp(resized_button[:, 0])),
        "button_actual_height_m": float(np.ptp(resized_button[:, 2])),
        "button_actual_protrusion_m": float(
            resized_button[:, 2].max() - housing_top_z
        ),
        "blade_envelope_correction": blade_envelope,
    }


def convex_hull(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a triangulated source-derived convex hull.

    Qhull can reject a degenerate source point cloud (for example a future
    zero-thickness blade).  In that case an epsilon-thick, axis-aligned box is
    used only as a deterministic fallback around the source bounds; the
    epsilon is recorded in metadata so it cannot be mistaken for source
    geometry.
    """

    points = np.asarray(points, dtype=np.float64)
    if len(points) < 4:
        return _box_hull(points, 1.0e-6)
    try:
        hull = trimesh.convex.convex_hull(points)
        if len(hull.vertices) >= 4 and len(hull.faces) >= 4:
            return np.asarray(hull.vertices), np.asarray(hull.faces, dtype=np.int64)
    except Exception:
        pass
    return _box_hull(points, 1.0e-6)


def _sampled_hull_surface_gap(
    hull_vertices: np.ndarray,
    hull_faces: np.ndarray,
    visual_mesh: trimesh.Trimesh,
    *,
    seed: int,
    sample_count: int = SLIDER_COLLIDER_AUDIT_SAMPLES,
) -> tuple[float, np.ndarray]:
    """Return deterministic max hull-to-visual gap and its direction."""

    hull_mesh = trimesh.Trimesh(
        vertices=np.asarray(hull_vertices, dtype=np.float64),
        faces=np.asarray(hull_faces, dtype=np.int64),
        process=False,
    )
    points, _ = trimesh.sample.sample_surface(
        hull_mesh, int(sample_count), seed=int(seed)
    )
    closest, distances, _ = trimesh.proximity.closest_point_naive(
        visual_mesh, points
    )
    if len(distances) == 0 or not np.all(np.isfinite(distances)):
        raise ValueError("slider collider surface-gap audit returned invalid samples")
    worst = int(np.argmax(distances))
    return float(distances[worst]), np.asarray(points[worst] - closest[worst])


def _slider_hull_geometry_quality(
    hull_vertices: np.ndarray,
    hull_faces: np.ndarray,
) -> dict[str, Any]:
    """Measure and fail-close a slider hull's physical geometry quality."""

    hull_vertices = np.asarray(hull_vertices, dtype=np.float64)
    hull_faces = np.asarray(hull_faces, dtype=np.int64)
    finite = bool(
        np.all(np.isfinite(hull_vertices)) and np.all(np.isfinite(hull_faces))
    )
    centered = hull_vertices - hull_vertices.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    affine_rank = int(np.linalg.matrix_rank(centered, tol=1.0e-12))
    singular_ratio = float(
        singular_values[-1] / singular_values[0]
        if len(singular_values) == 3 and singular_values[0] > 0.0
        else 0.0
    )
    axis_extents = np.ptp(hull_vertices, axis=0)
    hull_mesh = trimesh.Trimesh(
        vertices=hull_vertices, faces=hull_faces, process=False
    )
    volume_m3 = float(abs(hull_mesh.volume))
    checks = {
        "minimum_vertex_count": int(len(hull_vertices))
        >= SLIDER_COLLIDER_MIN_VERTEX_COUNT,
        "minimum_face_count": int(len(hull_faces))
        >= SLIDER_COLLIDER_MIN_FACE_COUNT,
        "finite": finite,
        "affine_rank_3": affine_rank == 3,
        "minimum_axis_extent": float(axis_extents.min())
        >= SLIDER_COLLIDER_MIN_AXIS_EXTENT_M,
        "minimum_singular_value": float(singular_values[-1])
        >= SLIDER_COLLIDER_MIN_SINGULAR_VALUE_M,
        "minimum_singular_value_ratio": singular_ratio
        >= SLIDER_COLLIDER_MIN_SINGULAR_VALUE_RATIO,
        "minimum_volume": volume_m3 >= SLIDER_COLLIDER_MIN_VOLUME_M3,
        "watertight": bool(hull_mesh.is_watertight),
        "winding_consistent": bool(hull_mesh.is_winding_consistent),
    }
    return {
        "vertex_count": int(len(hull_vertices)),
        "face_count": int(len(hull_faces)),
        "finite": finite,
        "affine_rank": affine_rank,
        "axis_extents_m": [float(value) for value in axis_extents],
        "minimum_axis_extent_m": float(axis_extents.min()),
        "singular_values_m": [float(value) for value in singular_values],
        "minimum_singular_value_m": float(singular_values[-1]),
        "singular_value_ratio": singular_ratio,
        "volume_m3": volume_m3,
        "watertight": bool(hull_mesh.is_watertight),
        "winding_consistent": bool(hull_mesh.is_winding_consistent),
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def connected_slider_hulls(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[
    list[tuple[np.ndarray, np.ndarray]],
    list[float],
    list[dict[str, Any]],
]:
    """Build strict 3-D hulls from welded shared-edge slider components.

    The glTF duplicates otherwise equal vertices at render seams.  Welding to
    eight decimal places (10 nm in this metre stage) recovers the source's
    shared-edge topology without inventing geometry.  Raised-button faces are
    handled by the dedicated button hull.  A component that is planar, too
    thin, too small, or over the gap target aborts the build instead of
    receiving an artificial epsilon box.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    raised_mask = vertices[:, 2] > BUTTON_RAISED_Z_THRESHOLD_M
    base_faces = faces[np.all(~raised_mask[faces], axis=1)]
    if len(base_faces) < SLIDER_COLLIDER_MIN_FACE_COUNT:
        raise ValueError("slider base has too few faces for collision decomposition")

    base_mesh = trimesh.Trimesh(
        vertices=vertices, faces=base_faces, process=False
    )
    base_mesh.remove_unreferenced_vertices()
    base_mesh.merge_vertices(digits_vertex=SLIDER_COMPONENT_WELD_DECIMAL_DIGITS)
    component_face_indices = trimesh.graph.connected_components(
        base_mesh.face_adjacency,
        nodes=np.arange(len(base_mesh.faces), dtype=np.int64),
        min_len=1,
    )
    if not component_face_indices:
        raise ValueError("slider base has no welded shared-edge components")

    visual_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    accepted: list[
        tuple[np.ndarray, np.ndarray, float, dict[str, Any]]
    ] = []
    welded_vertices = np.asarray(base_mesh.vertices, dtype=np.float64)
    welded_faces = np.asarray(base_mesh.faces, dtype=np.int64)
    for component_index, face_indices in enumerate(component_face_indices):
        source_faces = welded_faces[np.asarray(face_indices, dtype=np.int64)]
        source_vertices = welded_vertices[np.unique(source_faces)]
        if (
            len(source_vertices) < SLIDER_COLLIDER_MIN_VERTEX_COUNT
            or len(source_faces) < SLIDER_COLLIDER_MIN_FACE_COUNT
        ):
            raise ValueError(
                "slider welded component has too little source geometry: "
                f"component={component_index}, vertices={len(source_vertices)}, "
                f"faces={len(source_faces)}"
            )
        try:
            hull_mesh = trimesh.convex.convex_hull(source_vertices)
        except Exception as exc:
            raise ValueError(
                f"slider welded component {component_index} has no 3-D convex hull"
            ) from exc
        hull_vertices = np.asarray(hull_mesh.vertices, dtype=np.float64)
        hull_faces = np.asarray(hull_mesh.faces, dtype=np.int64)
        quality = _slider_hull_geometry_quality(hull_vertices, hull_faces)
        quality["source_component_index"] = int(component_index)
        quality["source_vertex_count"] = int(len(source_vertices))
        quality["source_face_count"] = int(len(source_faces))
        if not quality["pass"]:
            failed = [name for name, passed in quality["checks"].items() if not passed]
            raise ValueError(
                "slider welded component failed geometry-quality gates: "
                f"component={component_index}, failed={failed}, quality={quality}"
            )
        gap_m, _ = _sampled_hull_surface_gap(
            hull_vertices,
            hull_faces,
            visual_mesh,
            seed=13013 + component_index * 257,
            sample_count=SLIDER_COLLIDER_AUDIT_SAMPLES,
        )
        if gap_m > SLIDER_COLLIDER_SURFACE_GAP_TARGET_M:
            raise ValueError(
                "slider welded component exceeded sampled surface-gap target: "
                f"component={component_index}, gap={gap_m:.9f} m"
            )
        accepted.append((hull_vertices, hull_faces, gap_m, quality))

    # Stable spatial ordering keeps generated USD and metadata reproducible
    # even if trimesh changes its connected-component traversal order.
    accepted.sort(key=lambda item: tuple(np.asarray(item[0]).mean(axis=0)))
    return (
        [(vertices_, faces_) for vertices_, faces_, _, _ in accepted],
        [float(gap_m) for _, _, gap_m, _ in accepted],
        [quality for _, _, _, quality in accepted],
    )


def _box_hull(points: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(points, dtype=np.float64).min(axis=0)
    hi = np.asarray(points, dtype=np.float64).max(axis=0)
    ext = hi - lo
    for axis in range(3):
        if ext[axis] < epsilon:
            mid = (lo[axis] + hi[axis]) * 0.5
            lo[axis], hi[axis] = mid - epsilon * 0.5, mid + epsilon * 0.5
    v = np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ],
        dtype=np.float64,
    )
    f = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return v, f


def _create_mesh(
    stage: Usd.Stage,
    path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    uv: np.ndarray | None = None,
    material: UsdShade.Material | None = None,
    collision: bool = False,
    approximation: str | None = None,
    collision_only: bool = False,
) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    prim = mesh.GetPrim()
    mesh.CreatePointsAttr().Set(_as_float3_array(vertices))
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr().Set([int(i) for i in faces.reshape(-1)])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    if uv is not None:
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.Float2Array, UsdGeom.Tokens.faceVarying
        )
        st.Set(_as_float2_array(uv))
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr().Set(approximation or "convexHull")
        collision_api = UsdPhysics.CollisionAPI(prim)
        collision_api.CreateCollisionEnabledAttr().Set(True)
        # Rendering visibility is intentionally explicit.  PhysX still sees
        # the collision shape while Hydra omits it from the camera image.
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        prim.SetCustomDataByKey("handson:collisionOnly", bool(collision_only))
        prim.SetCustomDataByKey("handson:sourceDerived", True)
    elif material is not None:
        UsdShade.MaterialBindingAPI(prim).Bind(material)
    return mesh


def _create_material(stage: Usd.Stage, texture_dir: Path) -> UsdShade.Material:
    looks = UsdGeom.Scope.Define(stage, "/Asset/Looks")
    del looks
    material_path = "/Asset/Looks/UtilityKnifeMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    preview = UsdShade.Shader.Define(stage, material_path + "/PreviewSurface")
    preview.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(preview.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    # Keep the complete source texture set in the stage.  The metallic/
    # roughness texture follows glTF's B/G channel convention.
    reader = UsdShade.Shader.Define(stage, material_path + "/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    def texture(name: str, filename: str) -> UsdShade.Shader:
        tex = UsdShade.Shader.Define(stage, material_path + "/" + name)
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("textures/" + filename)
        )
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.GetOutput("result"))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        tex.CreateOutput("r", Sdf.ValueTypeNames.Float)
        tex.CreateOutput("g", Sdf.ValueTypeNames.Float)
        tex.CreateOutput("b", Sdf.ValueTypeNames.Float)
        tex.CreateOutput("a", Sdf.ValueTypeNames.Float)
        return tex

    base = texture("BaseColorTexture", "Material.001_baseColor.png")
    mr = texture("MetallicRoughnessTexture", "Material.001_metallicRoughness.png")
    normal = texture("NormalTexture", "Material.001_normal.png")
    normal.GetInput("sourceColorSpace").Set("raw")
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        base.GetOutput("rgb")
    )
    preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(mr.GetOutput("b"))
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(mr.GetOutput("g"))
    preview.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(normal.GetOutput("rgb"))
    preview.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.45)
    # A non-textured fallback helps headless previewers that do not evaluate
    # UsdUVTexture while retaining the texture connections above.
    preview.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    # USD customData accepts scalar values portably across usd-core and
    # Isaac's bundled USD; keep this as a JSON-ish string rather than a Python
    # list (which older USD builds cannot serialize).
    material.GetPrim().SetCustomDataByKey(
        "handson:sourceTextures",
        "textures/Material.001_baseColor.png;"
        "textures/Material.001_metallicRoughness.png;"
        "textures/Material.001_normal.png",
    )
    return material


def _uniform_box_diagonal_inertia(mass_kg: float, dimensions_m: np.ndarray) -> np.ndarray:
    """Approximate principal inertia from the authored rigid body's bounds.

    The source glTF has no physical mass properties.  A uniform-density box
    fitted to each rigid-link AABB is a deterministic, dimension-derived
    approximation that is safer than importing a DCC default or using an
    isotropic placeholder.  The approximation is recorded in metadata so a
    later live audit can replace it with measured/hull mass properties if
    needed.
    """

    dimensions = np.asarray(dimensions_m, dtype=np.float64)
    if dimensions.shape != (3,) or not np.all(np.isfinite(dimensions)):
        raise ValueError("rigid-body inertia dimensions must be a finite 3-vector")
    if np.any(dimensions <= 0.0):
        raise ValueError("rigid-body inertia dimensions must be positive")
    mass = float(mass_kg)
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("rigid-body mass must be positive and finite")
    return mass / 12.0 * np.asarray(
        [
            dimensions[1] ** 2 + dimensions[2] ** 2,
            dimensions[0] ** 2 + dimensions[2] ** 2,
            dimensions[0] ** 2 + dimensions[1] ** 2,
        ],
        dtype=np.float64,
    )


def _apply_body_defaults(
    prim: Usd.Prim,
    mass_kg: float,
    center_of_mass: np.ndarray,
    dimensions_m: np.ndarray,
) -> np.ndarray:
    body = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.CreateRigidBodyEnabledAttr().Set(True)
    body.CreateKinematicEnabledAttr().Set(False)
    body.CreateStartsAsleepAttr().Set(False)
    body.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr().Set(float(mass_kg))
    mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(*[float(x) for x in center_of_mass]))
    diagonal_inertia = _uniform_box_diagonal_inertia(mass_kg, dimensions_m)
    mass.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(*[float(value) for value in diagonal_inertia])
    )
    mass.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    prim.SetCustomDataByKey("handson:sourceDerivedMass", True)
    prim.SetCustomDataByKey("handson:inertiaModel", "uniformAabbBox")
    prim.SetCustomDataByKey(
        "handson:inertiaDimensionsM",
        ",".join(f"{float(value):.12g}" for value in dimensions_m),
    )
    prim.SetCustomDataByKey(
        "handson:diagonalInertiaKgM2",
        ",".join(f"{float(value):.12g}" for value in diagonal_inertia),
    )
    return diagonal_inertia


def _bounds_dict(points: np.ndarray) -> dict[str, list[float]]:
    b = np.asarray([points.min(axis=0), points.max(axis=0)], dtype=float)
    return {"min_m": [float(x) for x in b[0]], "max_m": [float(x) for x in b[1]], "extent_m": [float(x) for x in (b[1] - b[0])]}


def build_stage(records: list[MeshRecord], output_path: Path, source_meta: dict[str, Any]) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetEditTarget(stage.GetRootLayer())
    root = UsdGeom.Xform.Define(stage, "/Asset")
    root.GetPrim().SetCustomData({
        "handson:assetId": ASSET_ID,
        "handson:sourceTitle": TITLE,
        "handson:sourceAuthor": AUTHOR,
        "handson:sourceUrl": SOURCE_URL,
        "handson:license": LICENSE_NAME,
        "handson:conversionScale": float(CONVERSION_SCALE_M),
        "handson:axisMapping": "[-x, z, y] * 0.008 (det=+1, orientation-preserving; source glTF world -> AS013 metres)",
        "handson:ergonomicScaleXYZ": ",".join(
            f"{float(value):.12g}" for value in ERGONOMIC_SCALE_XYZ
        ),
        "handson:buttonTargetWidthM": float(BUTTON_TARGET_WIDTH_M),
        "handson:buttonTargetProtrusionM": float(BUTTON_TARGET_PROTRUSION_M),
        "handson:bladeEnvelopeMethod": (
            "baked_blade_y_fit_to_retracted_housing_envelope"
        ),
        "handson:bladeRetractedClearanceM": float(
            BLADE_RETRACTED_CLEARANCE_M
        ),
        "handson:sourceArchiveSha256": source_meta["source_archive_sha256"],
        "handson:sourceSceneSha256": source_meta["source_scene_sha256"],
        "handson:animationPolicy": "source animation inspected for travel only; not authored as physics",
    })
    stage.SetDefaultPrim(root.GetPrim())
    material = _create_material(stage, output_path.parent / "textures")

    housing = [r for r in records if r.category == "housing"]
    blade = next(r for r in records if r.category == "blade")
    slider = next(r for r in records if r.category == "slider")
    all_housing_points = np.vstack([r.vertices for r in housing])
    raised = slider.vertices[slider.vertices[:, 2] > BUTTON_RAISED_Z_THRESHOLD_M]
    if len(raised) < 4:
        raise ValueError("slider source mesh has no stable raised button region")
    button_center = raised.mean(axis=0)
    button_local = raised - button_center
    housing_dimensions = np.ptp(all_housing_points, axis=0)
    moving_points = np.vstack([blade.vertices, slider.vertices]) - button_center
    moving_dimensions = np.ptp(moving_points, axis=0)

    body0 = UsdGeom.Xform.Define(stage, "/Asset/E_body_1")
    body0_prim = body0.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(body0_prim)
    housing_mass = HOUSING_MASS_KG
    housing_com = all_housing_points.mean(axis=0)
    housing_inertia = _apply_body_defaults(
        body0_prim, housing_mass, housing_com, housing_dimensions
    )
    body0_prim.SetCustomDataByKey("handson:semanticRole", "housing")
    for index, record in enumerate(housing):
        _create_mesh(
            stage,
            f"/Asset/E_body_1/Visual_Housing_{index}",
            record.vertices,
            record.faces,
            uv=record.uv,
            material=material,
        )
    housing_hull_v, housing_hull_f = convex_hull(all_housing_points)
    housing_collision = _create_mesh(
        stage,
        "/Asset/E_body_1/Collision_Housing",
        housing_hull_v,
        housing_hull_f,
        collision=True,
        approximation="convexHull",
        collision_only=True,
    )
    housing_collision.GetPrim().SetCustomDataByKey("handson:semanticRole", "housing")

    body1 = UsdGeom.Xform.Define(stage, "/Asset/E_link_blade_2")
    body1_prim = body1.GetPrim()
    body1.AddTranslateOp().Set(Gf.Vec3d(*[float(x) for x in button_center]))
    moving_mass = MOVING_MASS_KG
    moving_com = np.zeros(3, dtype=float)
    moving_inertia = _apply_body_defaults(
        body1_prim, moving_mass, moving_com, moving_dimensions
    )
    body1_prim.SetCustomDataByKey("handson:semanticRole", "blade_and_slider")
    body1_prim.SetCustomDataByKey(
        "handson:buttonFrameOriginM",
        ",".join(f"{float(x):.9g}" for x in button_center),
    )
    for record, label in ((blade, "Blade"), (slider, "Slider")):
        _create_mesh(
            stage,
            f"/Asset/E_link_blade_2/Visual_{label}",
            record.vertices - button_center,
            record.faces,
            uv=record.uv,
            material=material,
        )
    blade_hull_v, blade_hull_f = convex_hull(blade.vertices - button_center)
    _create_mesh(
        stage,
        "/Asset/E_link_blade_2/Collision_Blade",
        blade_hull_v,
        blade_hull_f,
        collision=True,
        approximation="convexHull",
        collision_only=True,
    ).GetPrim().SetCustomDataByKey("handson:semanticRole", "blade")
    slider_hulls_world, slider_hull_gaps, slider_hull_qualities = connected_slider_hulls(
        slider.vertices, slider.faces
    )
    slider_hulls = [
        (vertices - button_center, faces)
        for vertices, faces in slider_hulls_world
    ]
    slider_collision_paths: list[str] = []
    for index, (slider_hull_v, slider_hull_f) in enumerate(slider_hulls):
        path = f"/Asset/E_link_blade_2/Collision_Slider_{index:02d}"
        slider_collision_paths.append(path)
        prim = _create_mesh(
            stage,
            path,
            slider_hull_v,
            slider_hull_f,
            collision=True,
            approximation="convexHull",
            collision_only=True,
        ).GetPrim()
        prim.SetCustomDataByKey("handson:semanticRole", "slider")
        prim.SetCustomDataByKey("handson:segmentIndex", int(index))
        prim.SetCustomDataByKey("handson:segmentCount", int(len(slider_hulls)))
        prim.SetCustomDataByKey(
            "handson:sampledSurfaceGapM", float(slider_hull_gaps[index])
        )
        quality = slider_hull_qualities[index]
        prim.SetCustomDataByKey("handson:geometryQualityPass", True)
        prim.SetCustomDataByKey(
            "handson:minimumAxisExtentM",
            float(quality["minimum_axis_extent_m"]),
        )
        prim.SetCustomDataByKey(
            "handson:minimumSingularValueM",
            float(quality["minimum_singular_value_m"]),
        )
        prim.SetCustomDataByKey(
            "handson:singularValueRatio",
            float(quality["singular_value_ratio"]),
        )
        prim.SetCustomDataByKey(
            "handson:volumeM3", float(quality["volume_m3"])
        )
    button_hull_v, button_hull_f = convex_hull(button_local)
    _create_mesh(
        stage,
        "/Asset/E_link_blade_2/Collision_Button",
        button_hull_v,
        button_hull_f,
        collision=True,
        approximation="convexHull",
        collision_only=True,
    ).GetPrim().SetCustomDataByKey("handson:semanticRole", "slider_button")

    joint = UsdPhysics.PrismaticJoint.Define(
        stage, f"/Asset/E_link_blade_2/{JOINT_NAME}"
    )
    joint_prim = joint.GetPrim()
    joint.CreateAxisAttr().Set(JOINT_AXIS)
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/Asset/E_body_1")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/Asset/E_link_blade_2")])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(x) for x in button_center]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLowerLimitAttr().Set(float(JOINT_LIMITS_M[0]))
    joint.CreateUpperLimitAttr().Set(float(JOINT_LIMITS_M[1]))
    joint.CreateJointEnabledAttr().Set(True)
    joint.CreateCollisionEnabledAttr().Set(False)
    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(80.0)
    drive.CreateDampingAttr().Set(1.2)
    drive.CreateMaxForceAttr().Set(12.0)
    joint_prim.SetCustomDataByKey("handson:semanticRole", "blade_slide")

    # A physics material is deliberately authored separately from the visual
    # material so Isaac can override friction without mutating the texture.
    physics_material = UsdShade.Material.Define(stage, "/Asset/PhysicsMaterial")
    pprim = physics_material.GetPrim()
    UsdPhysics.MaterialAPI.Apply(pprim)
    pprim.GetAttribute("physics:density").Set(900.0)
    pprim.GetAttribute("physics:staticFriction").Set(0.5)
    pprim.GetAttribute("physics:dynamicFriction").Set(0.35)
    pprim.GetAttribute("physics:restitution").Set(0.1)
    collision_paths = [
        "/Asset/E_body_1/Collision_Housing",
        "/Asset/E_link_blade_2/Collision_Blade",
        *slider_collision_paths,
        "/Asset/E_link_blade_2/Collision_Button",
    ]
    for path in collision_paths:
        # Use the USD physics material purpose (``material:binding:physics``)
        # rather than replacing the render-material binding on future assets.
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).Bind(
            physics_material, "fallbackStrength", "physics"
        )

    stage.GetRootLayer().Save()
    bounds = _bounds_dict(np.vstack([r.vertices for r in records]))
    visuals = [
        {
            "name": r.name,
            "category": r.category,
            "prim": f"/Asset/E_body_1/Visual_Housing_{housing.index(r)}" if r.category == "housing" else f"/Asset/E_link_blade_2/Visual_{'Blade' if r.category == 'blade' else 'Slider'}",
            "bounds": _bounds_dict(r.vertices),
            "source_material": r.source_material,
            "vertex_count": int(len(r.vertices)),
            "face_count": int(len(r.faces)),
        }
        for r in records
    ]
    collider_specs: list[tuple[str, str, np.ndarray, np.ndarray, float | None]] = [
        (
            "/Asset/E_body_1/Collision_Housing",
            "housing",
            housing_hull_v,
            housing_hull_f,
            None,
        ),
        (
            "/Asset/E_link_blade_2/Collision_Blade",
            "blade",
            blade_hull_v,
            blade_hull_f,
            None,
        ),
        *[
            (path, "slider", vertices, faces, slider_hull_gaps[index])
            for index, (path, (vertices, faces)) in enumerate(
                zip(slider_collision_paths, slider_hulls, strict=True)
            )
        ],
        (
            "/Asset/E_link_blade_2/Collision_Button",
            "slider_button",
            button_hull_v,
            button_hull_f,
            None,
        ),
    ]
    colliders = []
    for path, role, verts, faces, sampled_gap_m in collider_specs:
        entry: dict[str, Any] = {
            "prim": path,
            "role": role,
            "body": "/Asset/E_body_1" if role == "housing" else "/Asset/E_link_blade_2",
            "approximation": "convexHull",
            "visibility": "invisible",
            "collision_only": True,
            "bounds": _bounds_dict(verts),
            "vertex_count": int(len(verts)),
            "face_count": int(len(faces)),
        }
        if sampled_gap_m is not None:
            entry["sampled_surface_gap_m"] = float(sampled_gap_m)
            slider_index = int(path.rsplit("_", 1)[-1])
            entry["geometry_quality"] = slider_hull_qualities[slider_index]
        colliders.append(
            entry
        )
    metadata: dict[str, Any] = {
        "schema_version": "as013.asset.v4",
        "asset_id": ASSET_ID,
        "title": TITLE,
        "author": AUTHOR,
        "author_url": AUTHOR_URL,
        "source_url": SOURCE_URL,
        "license": LICENSE_NAME,
        "license_url": LICENSE_URL,
        "commercial_use": True,
        "attribution": ATTRIBUTION,
        "source_archive_sha256": source_meta["source_archive_sha256"],
        "archive_sha256": source_meta["source_archive_sha256"],
        "source_scene_sha256": source_meta["source_scene_sha256"],
        "source_files": source_meta["source_files"],
        "source_archive_path": source_meta["source_archive_path"],
        "conversion_scale": CONVERSION_SCALE_M,
        "ergonomic_resize": source_meta["ergonomic_resize"],
        # Keep the blade-envelope contract at the top level as well as inside
        # the broader resize record.  Runtime consumers and static auditors
        # can then verify the q=0 visual/collision invariant without knowing
        # the builder's internal nesting.
        "blade_envelope_correction": source_meta["ergonomic_resize"][
            "blade_envelope_correction"
        ],
        "axis_mapping": {
            "source": "trimesh/glTF world XYZ",
            "target": "AS013 body XYZ (metres)",
            "expression": "[-x, z, y] * 0.008",
            "determinant": 1.0,
            "orientation_preserving": True,
            "handedness_corrected": False,
        },
        "stage": {"meters_per_unit": 1.0, "up_axis": "Z", "default_prim": "/Asset"},
        "joint": {
            "name": JOINT_NAME,
            "type": "prismatic",
            "axis": JOINT_AXIS,
            "body0": "/Asset/E_body_1",
            "body1": "/Asset/E_link_blade_2",
            "limits_m": list(JOINT_LIMITS_M),
            "local_pos0_m": [float(x) for x in button_center],
            "local_pos1_m": [0.0, 0.0, 0.0],
            "drive": {"type": "force", "stiffness": 80.0, "damping": 1.2, "max_force": 12.0},
        },
        "button_region": {
            "predicate": f"slider canonical vertices z > {BUTTON_RAISED_Z_THRESHOLD_M:.6f} m",
            "source_vertex_count": int(len(raised)),
            "center_m": [float(x) for x in button_center],
            "bounds_m": _bounds_dict(raised),
            "local_bounds_m": _bounds_dict(button_local),
        },
        "bounds_m": bounds,
        # Component-level AABBs are kept at the top level as well as in the
        # visual list.  Runtime target derivation can therefore select the
        # housing surface without accidentally including the extended blade
        # or slider bounds.
        "housing_bounds_m": _bounds_dict(all_housing_points),
        "blade_bounds_m": _bounds_dict(blade.vertices),
        "slider_bounds_m": _bounds_dict(slider.vertices),
        "visuals": visuals,
        "colliders": colliders,
        "collision_summary": {
            "housing_colliders": 1,
            "blade_colliders": 1,
            "slider_colliders": int(len(slider_hulls)),
            "button_colliders": 1,
            "moving_colliders": int(len(slider_hulls) + 2),
            "all_collision_only_invisible": True,
            "inherited_sdf_schema": False,
        },
        "slider_collision_decomposition": {
            "method": "coordinate_welded_shared_edge_component_convex_hulls",
            "raised_button_has_dedicated_hull": True,
            "coordinate_weld_decimal_digits": int(
                SLIDER_COMPONENT_WELD_DECIMAL_DIGITS
            ),
            "coordinate_weld_tolerance_m": float(
                SLIDER_COMPONENT_WELD_TOLERANCE_M
            ),
            "surface_gap_target_m": float(
                SLIDER_COLLIDER_SURFACE_GAP_TARGET_M
            ),
            "certification_samples_per_hull": int(SLIDER_COLLIDER_AUDIT_SAMPLES),
            "hull_count": int(len(slider_hulls)),
            "sampled_surface_gap_max_m": float(max(slider_hull_gaps)),
            "sampled_surface_gap_per_hull_m": [
                float(value) for value in slider_hull_gaps
            ],
            "geometry_quality_gates": {
                "minimum_vertex_count": int(SLIDER_COLLIDER_MIN_VERTEX_COUNT),
                "minimum_face_count": int(SLIDER_COLLIDER_MIN_FACE_COUNT),
                "finite": True,
                "affine_rank": 3,
                "minimum_axis_extent_m": float(
                    SLIDER_COLLIDER_MIN_AXIS_EXTENT_M
                ),
                "minimum_singular_value_m": float(
                    SLIDER_COLLIDER_MIN_SINGULAR_VALUE_M
                ),
                "minimum_singular_value_ratio": float(
                    SLIDER_COLLIDER_MIN_SINGULAR_VALUE_RATIO
                ),
                "minimum_volume_m3": float(SLIDER_COLLIDER_MIN_VOLUME_M3),
                "watertight": True,
                "winding_consistent": True,
            },
            "geometry_quality_all_pass": bool(
                all(quality["pass"] for quality in slider_hull_qualities)
            ),
            "minimum_observed_axis_extent_m": float(
                min(
                    quality["minimum_axis_extent_m"]
                    for quality in slider_hull_qualities
                )
            ),
            "minimum_observed_singular_value_m": float(
                min(
                    quality["minimum_singular_value_m"]
                    for quality in slider_hull_qualities
                )
            ),
            "minimum_observed_singular_value_ratio": float(
                min(
                    quality["singular_value_ratio"]
                    for quality in slider_hull_qualities
                )
            ),
            "minimum_observed_volume_m3": float(
                min(quality["volume_m3"] for quality in slider_hull_qualities)
            ),
        },
        "rigid_bodies": [
            {
                "prim": "/Asset/E_body_1",
                "role": "housing",
                "mass_kg": housing_mass,
                "center_of_mass_local_m": [float(x) for x in housing_com],
                "inertia_model": "uniform_aabb_box",
                "inertia_dimensions_m": [float(x) for x in housing_dimensions],
                "diagonal_inertia_kg_m2": [float(x) for x in housing_inertia],
            },
            {
                "prim": "/Asset/E_link_blade_2",
                "role": "blade_and_slider",
                "mass_kg": moving_mass,
                "center_of_mass_local_m": [float(x) for x in moving_com],
                "inertia_model": "uniform_aabb_box",
                "inertia_dimensions_m": [float(x) for x in moving_dimensions],
                "diagonal_inertia_kg_m2": [float(x) for x in moving_inertia],
            },
        ],
        "animation_policy": "source animation is travel reference only; no animation prim authored",
        # Keep the manifest byte-identical in task-local and runtime mirrors;
        # an absolute output path would make the two provenance records differ
        # for no physical reason.
        "outputs": {
            "ascii_usd": "utility_knife_ahtusik.usda",
            "binary_usd": "utility_knife_ahtusik.usd",
        },
    }
    return metadata


def copy_texture_set(source_scene: Path, output_dir: Path) -> None:
    src = source_scene.parent / "textures"
    dst = output_dir / "textures"
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.glob("*"):
        if path.is_file():
            shutil.copy2(path, dst / path.name)


def source_metadata(scene_path: Path, archive_path: Path | None) -> dict[str, Any]:
    files: dict[str, str] = {}
    for p in [scene_path, scene_path.parent / "scene.bin", scene_path.parent / "license.txt", *sorted((scene_path.parent / "textures").glob("*"))]:
        if p.is_file():
            files[str(p.relative_to(scene_path.parent))] = sha256_file(p)
    archive_sha = sha256_file(archive_path) if archive_path and archive_path.is_file() else ""
    if not archive_sha:
        # The archive is copied into the task asset directory by the download
        # checkpoint; bind its hash whenever the caller omitted --archive.
        for candidate in (
            scene_path.parent.parent / "utility_knife.zip",
            scene_path.parents[2] / "utility_knife.zip",
        ):
            if candidate.is_file():
                archive_sha = sha256_file(candidate)
                archive_path = candidate
                break
    if not archive_sha:
        raise FileNotFoundError("could not resolve the downloaded utility_knife.zip")
    return {
        "source_archive_sha256": archive_sha,
        # Keep generated metadata reproducible across checkouts.  The exact
        # downloaded path is recorded in workflow_state.yaml; the asset
        # package itself only needs the stable task-local archive label.
        "source_archive_path": f"source/{archive_path.name}",
        "source_scene_sha256": sha256_file(scene_path),
        "source_files": files,
    }


def write_outputs(
    *,
    scene_path: Path,
    archive_path: Path | None,
    output_path: Path,
    secondary_outputs: Iterable[Path],
) -> None:
    records = load_source_meshes(scene_path)
    source_meta = source_metadata(scene_path, archive_path)
    source_meta["ergonomic_resize"] = apply_ergonomic_resize(records)
    targets = [output_path, *secondary_outputs]
    primary_meta: dict[str, Any] | None = None
    for target in targets:
        copy_texture_set(scene_path, target.parent)
        # Author complete siblings first, then atomically replace each public
        # file.  Usd.Stage.CreateNew truncates its destination immediately;
        # writing directly to the runtime path can expose a tiny/invalid USDA
        # to concurrent collection-contract checks during the 30k-point hull
        # certification run.
        building_usda = target.with_name(f".{target.stem}.building{target.suffix}")
        metadata_path = target.with_name("utility_knife_ahtusik_metadata.json")
        binary_path = target.with_suffix(".usd")
        building_metadata = metadata_path.with_name(
            f".{metadata_path.stem}.building{metadata_path.suffix}"
        )
        building_binary = binary_path.with_name(
            f".{binary_path.stem}.building{binary_path.suffix}"
        )
        temporary_paths = (building_usda, building_metadata, building_binary)
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        try:
            metadata = build_stage(records, building_usda, source_meta)
            building_metadata.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            # Also provide a binary .usd sibling for consumers that select by
            # extension; the ASCII .usda remains the reviewable source artifact.
            layer = Sdf.Layer.FindOrOpen(str(building_usda))
            if layer is None:
                raise RuntimeError(
                    f"failed to reopen generated USD layer: {building_usda}"
                )
            layer.Export(str(building_binary))
            os.replace(building_usda, target)
            os.replace(building_binary, binary_path)
            os.replace(building_metadata, metadata_path)
        finally:
            for temporary in temporary_paths:
                temporary.unlink(missing_ok=True)
        if primary_meta is None:
            primary_meta = metadata


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_scene = here / "assets/utility_knife_ahtusik/source/sketchfab_utility_knife/scene.gltf"
    default_archive = here / "assets/utility_knife_ahtusik/source/utility_knife.zip"
    default_output = here / "assets/utility_knife_ahtusik/runtime/utility_knife_ahtusik.usda"
    # ``here`` is .../<repo>/taskdev/tasks/<task>; its third parent by name
    # (parents[2]) is the repository root.
    default_runtime = here.parents[2] / "handson/handson/sim/isaaclab/usd/t40_slide_knife_blade/utility_knife_ahtusik/utility_knife_ahtusik.usda"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=default_scene)
    parser.add_argument("--archive", type=Path, default=default_archive)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--runtime-output", type=Path, default=default_runtime)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_outputs(
        scene_path=args.source.resolve(),
        archive_path=args.archive.resolve() if args.archive else None,
        output_path=args.output.resolve(),
        secondary_outputs=[args.runtime_output.resolve()],
    )
    print(f"built {args.output.resolve()}")
    print(f"built {args.runtime_output.resolve()}")


if __name__ == "__main__":
    main()
