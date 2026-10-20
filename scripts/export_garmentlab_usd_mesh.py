#!/usr/bin/env python3
"""Export a triangulated mesh from a GarmentLab USD as an OBJ.

Genesis' FEM cloth importer consumes triangle meshes such as OBJ, while the
downloaded GarmentLab garments are distributed as USD.  This script performs
only a format conversion: vertex positions and triangle indices are copied
from the authored USD mesh without remeshing or procedural geometry changes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _primvar_element_index(
    interpolation: str,
    *,
    face_index: int,
    face_vertex_index: int,
    vertex_index: int,
) -> int:
    """Map one USD face corner to its authored primvar element."""
    if interpolation == "faceVarying":
        return face_vertex_index
    if interpolation in {"vertex", "varying"}:
        return vertex_index
    if interpolation == "uniform":
        return face_index
    if interpolation == "constant":
        return 0
    raise RuntimeError(f"Unsupported USD texture-coordinate interpolation: {interpolation!r}")


def export_mesh(
    input_path: Path,
    output_path: Path,
    prim_path: str,
    *,
    texture_path: Path | None = None,
    visual_with_uv_seams: bool = False,
) -> tuple[int, int]:
    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover - depends on an installed USD runtime
        raise RuntimeError(
            "USD Python bindings are required. Run this script with the Isaac Sim "
            "Python runtime (or another Python environment that provides pxr)."
        ) from exc

    stage = Usd.Stage.Open(str(input_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {input_path}")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        candidates = [
            str(candidate.GetPath())
            for candidate in stage.Traverse()
            if candidate.IsA(UsdGeom.Mesh)
        ]
        raise RuntimeError(
            f"USD prim {prim_path!r} is not a mesh in {input_path}; "
            f"available meshes: {candidates}"
        )

    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
    counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
    indices = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
    if not points or not counts or not indices:
        raise RuntimeError(f"USD mesh {prim_path} has no authored topology")

    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    vertices = [transform.Transform(Gf.Vec3d(point)) for point in points]
    st_primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    st_values = st_primvar.Get(Usd.TimeCode.Default()) if st_primvar else None
    st_indices = st_primvar.GetIndices(Usd.TimeCode.Default()) if st_primvar else None
    st_interpolation = st_primvar.GetInterpolation() if st_primvar else ""
    # Physics exports intentionally omit UVs. OBJ loaders commonly duplicate
    # vertices at UV seams, which disconnects a cloth mesh. A separate,
    # explicitly visual-only export may retain those seams for skinning.
    has_texcoords = bool(st_values) and texture_path is not None

    triangles: list[tuple[tuple[int, int | None], ...]] = []
    cursor = 0
    for face_index, count in enumerate(counts):
        count = int(count)
        face = [int(index) for index in indices[cursor : cursor + count]]
        if count < 3:
            cursor += count
            continue
        for offset in range(1, count - 1):
            triangle: list[tuple[int, int | None]] = []
            for local_corner in (0, offset, offset + 1):
                vertex_index = face[local_corner]
                texcoord_index: int | None = None
                if has_texcoords:
                    element_index = _primvar_element_index(
                        st_interpolation,
                        face_index=face_index,
                        face_vertex_index=cursor + local_corner,
                        vertex_index=vertex_index,
                    )
                    if st_indices:
                        element_index = int(st_indices[element_index])
                    if element_index < 0 or element_index >= len(st_values):
                        raise RuntimeError(
                            "Malformed USD st primvar: texture-coordinate index "
                            f"{element_index} is outside [0, {len(st_values)})"
                        )
                    texcoord_index = element_index
                triangle.append((vertex_index, texcoord_index))
            triangles.append(tuple(triangle))
        cursor += count
    if cursor != len(indices):
        raise RuntimeError(
            f"Malformed USD topology: consumed {cursor} of {len(indices)} indices"
        )

    # OBJ/trimesh represents independently indexed UV seams by duplicating
    # geometry vertices. That is fine for a render mesh but disconnects a
    # cloth simulation mesh into separate triangles. GarmentLab's authored
    # `st` data for the supported assets is face-varying only in storage: all
    # corners sharing a geometry vertex carry the same coordinate. Compact it
    # back to one UV per geometry vertex and refuse a genuinely seamed asset
    # instead of silently changing its physical topology.
    compact_texcoords: list[tuple[float, float] | None] = [None] * len(vertices)
    texcoords_to_write: list[tuple[float, float]] = []
    if has_texcoords:
        seam_detected = False
        for triangle in triangles:
            for vertex_index, texcoord_index in triangle:
                assert texcoord_index is not None
                uv = (
                    float(st_values[texcoord_index][0]),
                    float(st_values[texcoord_index][1]),
                )
                existing = compact_texcoords[vertex_index]
                if existing is not None and existing != uv:
                    seam_detected = True
                    if not visual_with_uv_seams:
                        raise RuntimeError(
                        "The USD mesh has a genuine UV seam at geometry vertex "
                        f"{vertex_index}. Exporting that seam to OBJ would split "
                        "the cloth topology; use a simulator with separate visual "
                        "and collision meshes instead."
                        )
                compact_texcoords[vertex_index] = uv
        if seam_detected:
            uv_indices: dict[tuple[float, float], int] = {}
            rewritten_triangles: list[tuple[tuple[int, int], ...]] = []
            for triangle in triangles:
                rewritten: list[tuple[int, int]] = []
                for vertex_index, texcoord_index in triangle:
                    assert texcoord_index is not None
                    uv = (
                        float(st_values[texcoord_index][0]),
                        float(st_values[texcoord_index][1]),
                    )
                    compact_index = uv_indices.get(uv)
                    if compact_index is None:
                        compact_index = len(texcoords_to_write)
                        uv_indices[uv] = compact_index
                        texcoords_to_write.append(uv)
                    rewritten.append((vertex_index, compact_index))
                rewritten_triangles.append(tuple(rewritten))
            triangles = rewritten_triangles
        else:
            texcoords_to_write = [
                uv if uv is not None else (0.0, 0.0)
                for uv in compact_texcoords
            ]
            triangles = [
                tuple((vertex_index, vertex_index) for vertex_index, _ in triangle)
                for triangle in triangles
            ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    material_path = output_path.with_suffix(".mtl") if texture_path else None
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "# Format-converted from the downloaded GarmentLab asset; "
            "geometry is unchanged.\n"
        )
        if material_path is not None:
            handle.write(f"mtllib {material_path.name}\n")
        handle.write(f"o {input_path.stem}\n")
        for point in vertices:
            handle.write(f"v {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n")
        if has_texcoords:
            for uv in texcoords_to_write:
                handle.write(f"vt {uv[0]:.9g} {uv[1]:.9g}\n")
        if material_path is not None:
            handle.write("usemtl garment_material\n")
        for triangle in triangles:
            # OBJ indices are one-based. Keep USD's independently indexed,
            # face-varying UVs intact instead of welding texture seams.
            corners = []
            for vertex_index, texcoord_index in triangle:
                if texcoord_index is None:
                    corners.append(str(vertex_index + 1))
                else:
                    corners.append(f"{vertex_index + 1}/{texcoord_index + 1}")
            handle.write("f " + " ".join(corners) + "\n")

    if material_path is not None:
        assert texture_path is not None
        texture_path = texture_path.expanduser().resolve()
        if not texture_path.is_file():
            raise FileNotFoundError(texture_path)
        texture_reference = os.path.relpath(texture_path, material_path.parent)
        material_path.write_text(
            "# Material reference preserved from the downloaded GarmentLab asset.\n"
            "newmtl garment_material\n"
            "Ka 0.2 0.2 0.2\n"
            "Kd 1.0 1.0 1.0\n"
            "Ks 0.03 0.03 0.03\n"
            "Ns 2.0\n"
            "d 1.0\n"
            f"map_Kd {texture_reference}\n",
            encoding="utf-8",
        )

    return len(vertices), len(triangles)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one authored GarmentLab USD mesh to triangulated OBJ."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prim", default="/World/mesh")
    parser.add_argument(
        "--texture",
        type=Path,
        default=None,
        help=(
            "Optional downloaded diffuse texture. When provided, preserve the "
            "USD st primvar in OBJ and emit a sibling MTL that references it."
        ),
    )
    parser.add_argument(
        "--visual-with-uv-seams",
        action="store_true",
        help=(
            "Allow UV seam duplication for a visual-only OBJ. Never use this "
            "output directly as the cloth physics mesh."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    texture_path = None
    if args.texture is not None:
        texture_path = args.texture.expanduser()
        if not texture_path.is_absolute():
            texture_path = (input_path.parent / texture_path).resolve()
    vertices, triangles = export_mesh(
        input_path,
        output_path,
        args.prim,
        texture_path=texture_path,
        visual_with_uv_seams=bool(args.visual_with_uv_seams),
    )
    print(
        f"Exported {vertices} vertices and {triangles} triangles: {output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
