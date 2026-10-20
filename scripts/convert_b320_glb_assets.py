#!/usr/bin/env python3
"""Convert the remaining B320 project-authored GLBs to deterministic USD meshes.

Run this with the repository's Isaac runtime Python.  The source GLB remains
the provenance artifact; each derived stage records its SHA-256 and preserves
the fully transformed source topology, metre scale, and material base color.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom


RELATIVE_ASSETS = (
    (
        "dexverse_authored_spoon",
        "dexverse_authored_spoon.glb",
        "dexverse_authored_spoon.usd",
    ),
    (
        "dexverse_authored_sponge",
        "dexverse_authored_sponge.glb",
        "dexverse_authored_sponge.usd",
    ),
)


def _safe_prim_name(value: str, fallback: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9_]", "_", str(value or ""))
    if candidate and candidate[0].isdigit():
        candidate = f"_{candidate}"
    return candidate if candidate and Sdf.Path.IsValidIdentifier(candidate) else fallback


def _base_color(mesh: trimesh.Trimesh) -> tuple[float, float, float, float]:
    material = getattr(mesh.visual, "material", None)
    raw = getattr(material, "main_color", None)
    if raw is None:
        raw = np.asarray([160, 160, 160, 255], dtype=np.float64)
    color = np.asarray(raw, dtype=np.float64).reshape(-1)
    if color.size == 3:
        color = np.concatenate([color, [255.0]])
    color = np.clip(color[:4] / 255.0, 0.0, 1.0)
    return tuple(float(value) for value in color)


def convert(source: Path, output: Path) -> None:
    scene = trimesh.load(str(source), force="scene", process=False)
    if not isinstance(scene, trimesh.Scene) or not scene.geometry:
        raise ValueError(f"GLB did not contain a mesh scene: {source}")
    if getattr(scene, "units", None) not in {None, "meters"}:
        raise ValueError(f"B320 GLB must use metre units: {source} ({scene.units!r})")

    output.parent.mkdir(parents=True, exist_ok=True)
    # This path is a deterministic build product of the adjacent source GLB.
    output.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    root.SetCustomDataByKey("handson:sourceFormat", "glb")
    root.SetCustomDataByKey("handson:sourceFile", source.name)
    root.SetCustomDataByKey("handson:sourceSha256", source_hash)
    root.SetCustomDataByKey("handson:units", "meters")

    used_names: set[str] = set()
    mesh_count = 0
    for node_index, node_name in enumerate(sorted(scene.graph.nodes_geometry)):
        transform, geometry_name = scene.graph[node_name]
        source_mesh = scene.geometry[geometry_name].copy()
        source_mesh.apply_transform(np.asarray(transform, dtype=np.float64))
        if not isinstance(source_mesh, trimesh.Trimesh):
            raise TypeError(f"Unsupported GLB geometry {geometry_name!r} in {source}")
        if source_mesh.faces.ndim != 2 or source_mesh.faces.shape[1] != 3:
            raise ValueError(f"B320 derived USD requires triangles: {source}:{geometry_name}")

        prim_name = _safe_prim_name(node_name, f"mesh_{node_index}")
        if prim_name in used_names:
            prim_name = f"{prim_name}_{node_index}"
        used_names.add(prim_name)
        usd_mesh = UsdGeom.Mesh.Define(stage, f"/Asset/{prim_name}")
        vertices = np.asarray(source_mesh.vertices, dtype=np.float32)
        faces = np.asarray(source_mesh.faces, dtype=np.int32)
        usd_mesh.GetPointsAttr().Set(
            [Gf.Vec3f(*(float(value) for value in row)) for row in vertices]
        )
        usd_mesh.GetFaceVertexCountsAttr().Set([3] * int(faces.shape[0]))
        usd_mesh.GetFaceVertexIndicesAttr().Set(faces.reshape(-1).tolist())
        normals = np.asarray(source_mesh.vertex_normals, dtype=np.float32)
        if normals.shape == vertices.shape:
            usd_mesh.GetNormalsAttr().Set(
                [Gf.Vec3f(*(float(value) for value in row)) for row in normals]
            )
            usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        usd_mesh.GetDoubleSidedAttr().Set(True)
        red, green, blue, alpha = _base_color(source_mesh)
        usd_mesh.GetDisplayColorAttr().Set([Gf.Vec3f(red, green, blue)])
        usd_mesh.GetDisplayOpacityAttr().Set([float(alpha)])
        usd_mesh.GetPrim().SetCustomDataByKey(
            "handson:sourceGeometry", str(geometry_name)
        )
        mesh_count += 1

    root.SetCustomDataByKey("handson:meshCount", mesh_count)
    stage.GetRootLayer().Save()

    check = Usd.Stage.Open(str(output), Usd.Stage.LoadAll)
    if check is None or not check.GetDefaultPrim().IsValid():
        raise RuntimeError(f"Derived USD failed to reopen: {output}")
    reopened_meshes = sum(prim.IsA(UsdGeom.Mesh) for prim in check.Traverse())
    if reopened_meshes != mesh_count:
        raise RuntimeError(
            f"Derived USD mesh count mismatch: {output} "
            f"expected={mesh_count} actual={reopened_meshes}"
        )
    print(f"converted {source} -> {output} meshes={mesh_count} sha256={source_hash}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    asset_root = (
        args.repo.resolve()
        / "handson"
        / "handson"
        / "sim"
        / "envs"
        / "usd"
        / "b320_bimanual_benchmark"
        / "assets"
    )
    for directory, source_name, output_name in RELATIVE_ASSETS:
        convert(asset_root / directory / source_name, asset_root / directory / output_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
