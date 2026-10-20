#!/usr/bin/env python3
"""Author the two normalized BC021 wooden-block USD wrappers.

The downloaded Sketchfab scene contains four separate meshes.  This script
keeps the complete converted scene as provenance, selects two source nodes,
normalizes each with one uniform scale, and adds a convex-hull collider to the
selected visible mesh.  No primitive visual or collision proxy is generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback


SOURCE_WORLD_ROOT = "/World/BlocksSketchfab_obj_cleaner_materialmerger_gles"
SOURCE_ROOT = "/BlocksSketchfab_obj_cleaner_materialmerger_gles"
SELECTED = {
    "block_a": "Object_2",
    # Use the same broad, square source block for the second rigid body.  The
    # downloaded scene's other selected variant has a narrow rotated support
    # footprint and rocks under the strict BC021 reset-origin gate.
    "block_b": "Object_2",
}
TARGET_SIDE_M = 0.075
SOURCE_TITLE = "Old Fashioned Wooden Blocks"
SOURCE_AUTHOR = "CurlyFryWhy"
SOURCE_URL = (
    "https://sketchfab.com/3d-models/"
    "old-fashioned-wooden-blocks-8331752cf38243638b0681f5ea230c28"
)
LICENSE_NAME = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _author_wrapper(
    *,
    source_usd: Path,
    output: Path,
    source_node: str,
    source_mesh: str,
    source_bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    archive_sha256: str,
    source_sha256: str,
) -> dict[str, object]:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema

    source_min, source_max = source_bounds
    extents = tuple(float(source_max[i] - source_min[i]) for i in range(3))
    source_max_extent = max(extents)
    scale = TARGET_SIDE_M / source_max_extent
    center = tuple((source_min[i] + source_max[i]) * 0.5 for i in range(3))
    # Put the rigid-body origin at the measured visual/collision center.  The
    # BC021 reset contract stores body poses at their centers (rather than at
    # the source mesh's lowest point), so this keeps the authored half-extents
    # and the PhysX body transform in the same frame on reset.
    translation = tuple(-center[i] * scale for i in range(3))

    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    asset = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(asset)
    for key, value in {
        "assetRole": "render-and-collision rigid object",
        "sourceTitle": SOURCE_TITLE,
        "sourceAuthor": SOURCE_AUTHOR,
        "sourceUrl": SOURCE_URL,
        "license": LICENSE_NAME,
        "licenseUrl": LICENSE_URL,
        "archiveSha256": archive_sha256,
        "sourceSha256": source_sha256,
        "selectedSourceNode": source_node,
        "selectedSourceMesh": source_mesh,
        "targetSideM": TARGET_SIDE_M,
        "normalizationScale": scale,
        "sourceUnits": "meters",
        "sourceUpAxis": "converted Z-up world bounds",
        "collisionApproximation": "convexHull",
        "contentMeshCollisionEnabled": True,
        "oldPlaceholderCollisionReused": False,
    }.items():
        asset.SetCustomDataByKey(f"handson:{key}", value)

    normalized = UsdGeom.Xform.Define(stage, "/Asset/Normalized")
    matrix = Gf.Matrix4d(
        (scale, 0.0, 0.0, 0.0),
        (0.0, scale, 0.0, 0.0),
        (0.0, 0.0, scale, 0.0),
        (translation[0], translation[1], translation[2], 1.0),
    )
    normalized.AddTransformOp().Set(matrix)

    source_reference = UsdGeom.Xform.Define(stage, "/Asset/Normalized/Source").GetPrim()
    source_reference.GetReferences().AddReference(
        Sdf.Reference(assetPath="source/converted_scene.usd", primPath="/World")
    )
    source_root = f"/Asset/Normalized/Source{SOURCE_ROOT}"

    collision_reference = UsdGeom.Xform.Define(
        stage, "/Asset/Normalized/CollisionSource"
    ).GetPrim()
    collision_reference.GetReferences().AddReference(
        Sdf.Reference(assetPath="source/converted_scene.usd", primPath="/World")
    )
    UsdGeom.Imageable(collision_reference).MakeInvisible()
    collision_root = f"/Asset/Normalized/CollisionSource{SOURCE_ROOT}"

    for root_path in (source_root, collision_root):
        for sibling in ("Object_2", "Object_3", "Object_4", "Object_5"):
            if sibling != source_node:
                sibling_prim = stage.GetPrimAtPath(f"{root_path}/{sibling}")
                if not sibling_prim.IsValid():
                    raise RuntimeError(f"missing source sibling: {root_path}/{sibling}")
                sibling_prim.SetActive(False)

    selected_prim = stage.GetPrimAtPath(f"{source_root}/{source_node}/{source_mesh}")
    if not selected_prim.IsValid():
        raise RuntimeError(f"missing selected source mesh: {source_root}/{source_node}/{source_mesh}")
    selected_prim.SetCustomDataByKey("handson:classification", "task_visible")

    collision_prim = stage.GetPrimAtPath(
        f"{collision_root}/{source_node}/{source_mesh}"
    )
    if not collision_prim.IsValid():
        raise RuntimeError(
            f"missing collision source mesh: {collision_root}/{source_node}/{source_mesh}"
        )
    UsdPhysics.CollisionAPI.Apply(collision_prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(collision_prim)
    mesh_collision.CreateApproximationAttr().Set("convexHull")
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(collision_prim)
    physx_collision.CreateContactOffsetAttr().Set(0.0015)
    physx_collision.CreateRestOffsetAttr().Set(0.0)
    collision_prim.SetCustomDataByKey("handson:classification", "formal_collider")
    collision_prim.SetCustomDataByKey(
        "handson:collisionRole", "source-derived visual convex hull"
    )
    collision_prim.SetCustomDataByKey("handson:renderAndCollisionSameMesh", True)
    stage.GetRootLayer().Save()

    reopened = Usd.Stage.Open(str(output), Usd.Stage.LoadAll)
    if reopened is None or not reopened.GetDefaultPrim().IsValid():
        raise RuntimeError(f"could not reopen generated wrapper: {output}")
    active_meshes = [
        prim
        for prim in reopened.Traverse()
        if prim.IsA(UsdGeom.Mesh) and prim.IsActive()
    ]
    if len(active_meshes) != 2:
        raise RuntimeError(
            f"{output.name} expected one visible and one collision source mesh, got "
            f"{len(active_meshes)}: {[str(p.GetPath()) for p in active_meshes]}"
        )
    colliders = [
        prim for prim in active_meshes if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if len(colliders) != 1:
        raise RuntimeError(f"{output.name} collider contract failed: {len(colliders)}")
    visible_meshes = [
        prim
        for prim in active_meshes
        if UsdGeom.Imageable(prim).ComputeVisibility()
        != UsdGeom.Tokens.invisible
    ]
    if len(visible_meshes) != 1 or visible_meshes[0].HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError(
            f"{output.name} visible/collision closure failed: "
            f"visible={[str(p.GetPath()) for p in visible_meshes]}"
        )
    return {
        "source_node": source_node,
        "source_mesh": source_mesh,
        "source_bounds_min_m": list(source_min),
        "source_bounds_max_m": list(source_max),
        "source_extents_m": list(extents),
        "normalization_scale": scale,
        "normalized_translation_m": list(translation),
        "normalized_local_bounds_min_m": [
            float(-0.5 * extent * scale) for extent in extents
        ],
        "normalized_local_bounds_max_m": [
            float(0.5 * extent * scale) for extent in extents
        ],
        "normalized_target_max_extent_m": TARGET_SIDE_M,
        "active_mesh_count": len(active_meshes),
        "visible_mesh_count": len(visible_meshes),
        "collider_count": len(colliders),
        "wrapper_sha256": sha256(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    asset_dir = args.asset_dir.resolve()
    source_usd = asset_dir / "source" / "converted_scene.usd"
    if not source_usd.is_file():
        raise FileNotFoundError(source_usd)
    archive = args.archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "enable_cameras": False})
    try:
        app.update()
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(source_usd), Usd.Stage.LoadAll)
        if stage is None:
            raise RuntimeError(f"could not open converted source: {source_usd}")
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_],
            useExtentsHint=True,
        )
        source_sha = sha256(source_usd)
        archive_sha = sha256(archive)
        profiles: dict[str, object] = {
            "schema": "bc021_old_fashioned_wooden_blocks_v1",
            "source": {
                "archive_filename": archive.name,
                "archive_sha256": archive_sha,
                "title": SOURCE_TITLE,
                "author": SOURCE_AUTHOR,
                "source_url": SOURCE_URL,
                "license": LICENSE_NAME,
                "license_url": LICENSE_URL,
                "source_scene_sha256": source_sha,
            },
            "conversion": {
                "runtime": "Isaac Sim 5.1",
                "converter": "omni.kit.asset_converter 5.0.17",
                "output": "source/converted_scene.usd",
                "source_root": SOURCE_ROOT,
                "selected_nodes": SELECTED,
            },
            "blocks": {},
        }
        source_meshes = {
            "block_a": "Object_0",
            "block_b": "Object_0",
        }
        for name, source_node in SELECTED.items():
            prim = stage.GetPrimAtPath(f"{SOURCE_WORLD_ROOT}/{source_node}")
            if not prim.IsValid():
                raise RuntimeError(f"missing selected source node: {prim.GetPath()}")
            box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            minimum = tuple(float(v) for v in box.GetMin())
            maximum = tuple(float(v) for v in box.GetMax())
            wrapper = asset_dir / f"{name}.usda"
            profiles["blocks"][name] = _author_wrapper(
                source_usd=source_usd,
                output=wrapper,
                source_node=source_node,
                source_mesh=source_meshes[name],
                source_bounds=(minimum, maximum),
                archive_sha256=archive_sha,
                source_sha256=source_sha,
            )
        profile_path = asset_dir / "asset_profile.json"
        profile_path.write_text(
            json.dumps(profiles, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(profiles, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close(wait_for_replicator=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
