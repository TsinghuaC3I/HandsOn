#!/usr/bin/env python3
"""Prepare the HS005 kitchen/faucet and Ani 3D mug derivatives for KA021.

KA021 deliberately reuses the reviewed realistic kitchen composition from
B320-HS005. The source faucet is one connected mesh, so this builder makes a
deterministic face partition around its authored side lever. Both generated
USDs retain the source vertices and UVs: one is fixed and one is attached to a
real runtime revolute joint. The HS005 kitchen and open-basin collision source
remain shared in place rather than being copied into a second bundle. KA021
uses the scene-owned analytic basin and open countertop collision directly.

Run with Isaac Sim Python, for example::

    /home/user/isaacsim/python.sh scripts/prepare_ka021_public_faucet_assets.py \
        --hs005-scene taskdev/tasks/bench320_hs005_sink_bottle_stabilized_scrub/scene/hs005_assets.usda \
        --mug-zip /path/to/ceramic_white_mug.zip \
        --asset-dir /path/to/cc_ka021_hs005_kitchen_anilai_white_mug
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import traceback
import zipfile


HS005_SCENE_SHA256 = (
    "ea706ecab4785bab4b7f7aa96a6c83b488602b1b47a8866674063f03e3679cba"
)
KITCHEN_SOURCE_SHA256 = (
    "12ac915b9d442399f10af5b302d9c6ef887e32c8aa1d0f6addd3a1dbc5f58524"
)
SINK_COLLISION_SHA256 = (
    "46839123013a8785fcc70b9744fae0745972561e8c399c14c06b394b4f774082"
)
MUG_ARCHIVE_SHA256 = (
    "f9142fbf5bf300af9444a264ebf6de8c0738d93e8758b289a43415024be778e7"
)
MUG_SOURCE_FILES_SHA256 = {
    "license.txt": "390f6d90c27d3f576ca9e0148840a3caab8619ede8a3f5552ac9063ac03b48b8",
    "scene.bin": "e85d79b4a66c75ecc2c5fbfabf0c084e85f3a1c2d9e163953bb20d78628306c0",
    "scene.gltf": "a0c68b86012fcb1cda80b5c22fd138a3b5adbb247d86ee06b893b2da35fbe36d",
    "textures/Mug_White_baseColor.png": (
        "039fc544b907b3e4a7ef5e92d582b3910c003a6cbd2287d44f5e2ca317ca25b9"
    ),
    "textures/Mug_White_emissive.jpeg": (
        "d1a77e8d38602635e6ec0850819f15919df3fffd613f625f5b40f48daafad9b4"
    ),
    "textures/Mug_White_metallicRoughness.png": (
        "59de154f99948fee53dfda9407f88077983ad9eb0ff4c7b82a9de6bec9cebca6"
    ),
    "textures/Mug_White_normal.png": (
        "be748b70544c54f70210127a02cf48c3d13cf9f2f6c723e183db7e80a79fbecf"
    ),
}
KITCHEN_SOURCE_URL = (
    "https://sketchfab.com/3d-models/"
    "modern-apartment-1fbb649cd6624f2bb7b7d6e30c6533a5"
)
MUG_SOURCE_URL = (
    "https://sketchfab.com/3d-models/"
    "ceramic-white-mug-fdf735d1f8f6469d892a989b509cd245"
)
MUG_DOWNLOAD_URL = f"{MUG_SOURCE_URL}#download"
KITCHEN_LICENSE_NAME = "CC BY 4.0"
KITCHEN_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
MUG_LICENSE_NAME = "CC BY 4.0"
MUG_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
MUG_RUNTIME_SCALE = 0.006
MUG_RIM_HEIGHT_M = 0.09387
FAUCET_PRIM_PATH = "/World/Sink_Faucet3_001_0/Sink_Faucet3_001_0"
KITCHEN_TRANSLATION_XYZ = (-1.135, -2.8, 0.18151906)
HANDLE_PIVOT_W = (0.024, -0.275, 1.120)
OUTLET_POSITION_W = (-0.118627, -0.149335, 1.256958)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


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
        "embed_textures": False,
        "single_mesh": False,
        "merge_all_meshes": False,
        "disabling_instancing": True,
    }
    for name, value in settings.items():
        if not hasattr(context, name):
            raise RuntimeError(f"AssetConverterContext lacks {name!r}")
        setattr(context, name, value)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.usd")
    task = omni.kit.asset_converter.get_instance().create_converter_task(
        str(source), str(temporary), None, context
    )
    success = asyncio.get_event_loop().run_until_complete(task.wait_until_finished())
    if not success or not temporary.is_file() or temporary.stat().st_size <= 1024:
        detail = getattr(task, "get_error_message", lambda: "unknown error")()
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"KA021 glTF conversion failed for {source.name}: {detail}")
    temporary.replace(output)


def _extract_reviewed_mug_archive(archive: Path, destination: Path) -> None:
    """Extract only the reviewed Sketchfab glTF closure and verify every byte."""

    expected_names = set(MUG_SOURCE_FILES_SHA256)
    with zipfile.ZipFile(archive) as bundle:
        file_names = {info.filename for info in bundle.infolist() if not info.is_dir()}
        if file_names != expected_names:
            raise RuntimeError(
                "reviewed KA021 mug archive closure changed: "
                f"expected={sorted(expected_names)} actual={sorted(file_names)}"
            )
        for info in bundle.infolist():
            if info.is_dir():
                continue
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe path in KA021 mug archive: {info.filename}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    for relative, digest in MUG_SOURCE_FILES_SHA256.items():
        path = destination / relative
        if _sha256(path) != digest:
            raise RuntimeError(f"reviewed KA021 mug source hash changed: {relative}")


def _task_world_points(mesh_prim: object) -> object:
    """Return the source faucet points in the reviewed HS005 task frame."""

    import numpy as np
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(mesh_prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    transform = np.asarray(
        UsdGeom.XformCache().GetLocalToWorldTransform(mesh_prim),
        dtype=np.float64,
    )
    source_world = (
        np.column_stack((points, np.ones(len(points), dtype=np.float64)))
        @ transform
    )[:, :3]
    tx, ty, tz = KITCHEN_TRANSLATION_XYZ
    # HS005's reviewed kitchen transform maps source (x, y, z) to
    # task (-z, -x, y), followed by this translation.
    return np.column_stack(
        (
            -source_world[:, 2] + tx,
            -source_world[:, 0] + ty,
            source_world[:, 1] + tz,
        )
    )


def _write_faucet_subset(
    output: Path,
    *,
    source_mesh: object,
    task_points: object,
    face_mask: object,
    component: str,
) -> dict[str, object]:
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    source_indices = np.asarray(
        source_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64
    ).reshape(-1, 3)
    selected = source_indices[np.asarray(face_mask, dtype=bool)]
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, f"/World/Faucet{component.title()}")
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(True)
    mesh.CreatePointsAttr().Set(
        Vt.Vec3fArray.FromNumpy(np.asarray(task_points, dtype=np.float32))
    )
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(selected))
    mesh.CreateFaceVertexIndicesAttr().Set(selected.reshape(-1).tolist())
    used_points = np.asarray(task_points)[np.unique(selected)]
    mesh.CreateExtentAttr().Set(
        [
            Gf.Vec3f(*map(float, used_points.min(axis=0))),
            Gf.Vec3f(*map(float, used_points.max(axis=0))),
        ]
    )

    source_st = UsdGeom.PrimvarsAPI(source_mesh.GetPrim()).GetPrimvar("st")
    if not source_st.IsDefined() or source_st.GetInterpolation() != "faceVarying":
        raise RuntimeError("reviewed HS005 faucet no longer has face-varying UVs")
    if source_st.IsIndexed():
        raise RuntimeError("reviewed HS005 faucet unexpectedly uses indexed UVs")
    source_uv = np.asarray(source_st.Get(), dtype=np.float32).reshape(-1, 3, 2)
    selected_uv = source_uv[np.asarray(face_mask, dtype=bool)].reshape(-1, 2)
    output_st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    output_st.Set(Vt.Vec2fArray.FromNumpy(selected_uv))

    mesh.GetPrim().SetCustomDataByKey("handson:sourceGeometry", FAUCET_PRIM_PATH)
    mesh.GetPrim().SetCustomDataByKey("handson:sourceLicense", KITCHEN_LICENSE_NAME)
    mesh.GetPrim().SetCustomDataByKey("handson:sourceCreator", "Visthétique")
    mesh.GetPrim().SetCustomDataByKey("handson:faucetComponent", component)
    mesh.GetPrim().SetCustomDataByKey("handson:visualOnlySource", True)
    stage.GetRootLayer().Save()
    return {
        "path": output.name,
        "sha256": _sha256(output),
        "size_bytes": output.stat().st_size,
        "mesh_count": 1,
        "face_count": int(len(selected)),
        "bounds_task_xyz_m": {
            "min": used_points.min(axis=0).tolist(),
            "max": used_points.max(axis=0).tolist(),
        },
    }


def _extract_hs005_faucet(
    kitchen_usd: Path,
    *,
    static_output: Path,
    handle_output: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    import numpy as np
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(kitchen_usd), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"cannot open HS005 kitchen: {kitchen_usd}")
    prim = stage.GetPrimAtPath(FAUCET_PRIM_PATH)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"reviewed HS005 faucet prim is missing: {FAUCET_PRIM_PATH}")
    mesh = UsdGeom.Mesh(prim)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    if len(counts) != 10568 or not np.all(counts == 3):
        raise RuntimeError("reviewed HS005 faucet topology changed")
    task_points = _task_world_points(prim)
    indices = np.asarray(
        mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64
    ).reshape(-1, 3)
    centroids = task_points[indices].mean(axis=1)
    # Reviewed on the original 10,568-face mesh. This box cleanly separates
    # the side lever and cartridge from the high-arc body while retaining the
    # exact source vertices and UVs. The expected face counts make upstream
    # topology or transform drift fail closed.
    handle_mask = (
        (centroids[:, 0] < 0.012)
        & (centroids[:, 1] < -0.277)
        & (centroids[:, 2] > 1.095)
        & (centroids[:, 2] < 1.215)
    )
    handle_faces = int(handle_mask.sum())
    if handle_faces != 1904:
        raise RuntimeError(
            "reviewed HS005 faucet face partition changed: "
            f"expected 1904 handle faces, found {handle_faces}"
        )
    components = [
        _write_faucet_subset(
            static_output,
            source_mesh=mesh,
            task_points=task_points,
            face_mask=~handle_mask,
            component="static",
        ),
        _write_faucet_subset(
            handle_output,
            source_mesh=mesh,
            task_points=task_points,
            face_mask=handle_mask,
            component="handle",
        ),
    ]
    split = {
        "method": "reviewed_task_frame_face_centroid_partition",
        "reviewed_rule": (
            "centroid x < 0.012 m, y < -0.277 m, "
            "1.095 m < z < 1.215 m"
        ),
        "source_face_count": int(len(indices)),
        "static_face_count": int((~handle_mask).sum()),
        "handle_face_count": handle_faces,
        "preserves_source_vertices": True,
        "preserves_source_uvs": True,
    }
    return components, split


def _review_mug(path: Path) -> dict[str, object]:
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise RuntimeError(f"cannot open converted KA021 mug: {path}")
    default = stage.GetDefaultPrim()
    meshes = [prim for prim in Usd.PrimRange(default) if prim.IsA(UsdGeom.Mesh)]
    if not meshes:
        raise RuntimeError("reviewed mug conversion expected at least one mesh")
    default.SetCustomDataByKey("handson:sourceLicense", MUG_LICENSE_NAME)
    default.SetCustomDataByKey("handson:sourceUpAxis", "Y")
    default.SetCustomDataByKey("handson:runtimeUpAxis", "Z")
    default.SetCustomDataByKey("handson:runtimeOrientation", "identity")
    # Asset Converter preserves the source Y-up correction as an orient op on
    # the default prim.  IsaacLab authors the runtime pose on that same prim,
    # replacing the correction and leaving visual and collision frames out of
    # sync.  Bake the public asset contract at the default-prim boundary:
    # identity root, converted Z-up visual child, and Z-up analytic colliders.
    orient_ops = [
        op
        for op in UsdGeom.Xformable(default).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient
    ]
    if len(orient_ops) != 1:
        raise RuntimeError(
            "reviewed mug conversion expected exactly one default-root orient op"
        )
    orient_ops[0].Set(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
    for visual_mesh in meshes:
        for api_schema in (
            UsdPhysics.CollisionAPI,
            UsdPhysics.MeshCollisionAPI,
            PhysxSchema.PhysxCollisionAPI,
        ):
            if visual_mesh.HasAPI(api_schema):
                visual_mesh.RemoveAPI(api_schema)

    collider = UsdGeom.Cylinder.Define(stage, "/World/KA021MugCollision")
    collider.CreateAxisAttr().Set(UsdGeom.Tokens.z)
    collider.CreateRadiusAttr().Set(6.75)
    collider.CreateHeightAttr().Set(14.40)
    collider.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 7.45))
    collider.MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(
        collider.GetPrim()
    ).CreateCollisionEnabledAttr().Set(True)
    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collider.GetPrim())
    collision_api.CreateContactOffsetAttr().Set(0.0015)
    collision_api.CreateRestOffsetAttr().Set(0.0)

    foot = UsdGeom.Cube.Define(stage, "/World/KA021MugCollisionFoot")
    foot.CreateSizeAttr().Set(1.0)
    foot_xform = UsdGeom.XformCommonAPI(foot)
    foot_xform.SetTranslate(Gf.Vec3d(0.0, 0.0, 0.50))
    foot_xform.SetScale(Gf.Vec3f(9.0, 9.0, 1.0))
    foot.MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(foot.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    foot_api = PhysxSchema.PhysxCollisionAPI.Apply(foot.GetPrim())
    foot_api.CreateContactOffsetAttr().Set(0.0015)
    foot_api.CreateRestOffsetAttr().Set(0.0)

    default.SetCustomDataByKey("handson:runtimeColor", "source_white_pbr")
    default.SetCustomDataByKey("handson:sourcePbrTexturesPreserved", True)
    default.SetCustomDataByKey(
        "handson:collisionApproximation", "analyticCylinderAndCube"
    )
    default.SetCustomDataByKey("handson:collisionShape", "recessedFlatFoot")
    bounds_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    visual_ranges = [
        bounds_cache.ComputeWorldBound(mesh).ComputeAlignedRange()
        for mesh in meshes
    ]
    visual_min = [
        min(float(bounds.GetMin()[axis]) for bounds in visual_ranges)
        for axis in range(3)
    ]
    visual_max = [
        max(float(bounds.GetMax()[axis]) for bounds in visual_ranges)
        for axis in range(3)
    ]
    if visual_min[2] <= 0.0 or not math.isclose(
        visual_max[2] * MUG_RUNTIME_SCALE,
        MUG_RIM_HEIGHT_M,
        rel_tol=0.0,
        abs_tol=5.0e-5,
    ):
        raise RuntimeError(
            "normalized mug visual is not supported Z-up geometry: "
            f"min={visual_min}, max={visual_max}"
        )
    stage.GetRootLayer().Export(str(path))
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "mesh_count": len(meshes),
        "coordinate_contract": {
            "source_up_axis": "Y",
            "runtime_up_axis": "Z",
            "runtime_home_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "visual_bounds_source_units": {
                "min": visual_min,
                "max": visual_max,
            },
            "runtime_visual_bottom_m": visual_min[2] * MUG_RUNTIME_SCALE,
            "runtime_visual_rim_z_m": visual_max[2] * MUG_RUNTIME_SCALE,
        },
        "collision": {
            "type": "cylinder_with_recessed_cube_foot",
            "source_axis": "Z",
            "runtime_axis": "Z",
            "source_radius_units": 6.75,
            "source_height_units": 14.40,
            "runtime_radius_m": 0.0405,
            "runtime_height_m": 0.0864,
            "runtime_foot_xyz_m": [0.054, 0.054, 0.006],
            "runtime_support_min_z_m": 0.0,
        },
    }


def _write_manifest(
    asset_dir: Path,
    *,
    repo_root: Path,
    hs005_scene: Path,
    kitchen_usd: Path,
    sink_collision: Path,
    components: list[dict[str, object]],
    split: dict[str, object],
) -> None:
    kitchen_relative = _repo_relative(kitchen_usd, repo_root)
    sink_relative = _repo_relative(sink_collision, repo_root)
    scene_relative = _repo_relative(hs005_scene, repo_root)
    manifest = {
        "schema": "handson.ka021.hs005-kitchen-faucet-cup-assets",
        "schema_version": 8,
        "task_id": "B320-KA021",
        "task_name": "t05_fill_cup",
        "license_summary": [KITCHEN_LICENSE_NAME, MUG_LICENSE_NAME],
        "sources": [
            {
                "role": "kitchen_sink_and_faucet_visual",
                "title": "Modern Apartment",
                "creator": "Visthétique",
                "platform": "Sketchfab",
                "source_url": KITCHEN_SOURCE_URL,
                "source_file_repo_relative": kitchen_relative,
                "source_sha256": KITCHEN_SOURCE_SHA256,
                "shared_from_task": "B320-HS005",
                "license": {
                    "name": KITCHEN_LICENSE_NAME,
                    "url": KITCHEN_LICENSE_URL,
                    "attribution_required": True,
                },
            },
            {
                "role": "source_open_sink_collision_derivative",
                "title": "HS005 reviewed sink collision derivative",
                "creator": "HandsOn HS005 task",
                "source_file_repo_relative": sink_relative,
                "source_sha256": SINK_COLLISION_SHA256,
                "license_inherited_from": "Modern Apartment",
            },
            {
                "role": "manipulated_cup",
                "title": "Ceramic White Mug",
                "creator": "Ani 3D (anilai)",
                "platform": "Sketchfab",
                "source_url": MUG_SOURCE_URL,
                "download_url": MUG_DOWNLOAD_URL,
                "source_file": "ceramic_white_mug.zip",
                "source_sha256": MUG_ARCHIVE_SHA256,
                "source_closure_sha256": MUG_SOURCE_FILES_SHA256,
                "license": {
                    "name": MUG_LICENSE_NAME,
                    "url": MUG_LICENSE_URL,
                    "attribution_required": True,
                },
            },
        ],
        "hs005_scene": {
            "source_file_repo_relative": scene_relative,
            "sha256": HS005_SCENE_SHA256,
            "referenced_prim": "/HS005Assets",
            "deactivated_props": ["Bottle", "Brush", "DishSoap", "Sponge"],
            "runtime_disabled_collision_prims": [],
            "collision_policy": (
                "full_height_basin_and_open_countertop_ring_v1"
            ),
            "collision_root": "/HS005Assets/SceneCollision",
        },
        "authorship_boundary": {
            "downloaded_geometry": [
                "Visthétique Modern Apartment kitchen, sink, drain, and faucet",
                "Ani 3D Ceramic White Mug visual mesh and PBR textures",
            ],
            "reused_hs005_work": [
                "reviewed kitchen placement and material overrides",
                "open-basin visual and collision derivative provenance",
            ],
            "handson_authored_runtime": [
                "deterministic face partition of the exact HS005 source faucet",
                "fixed-base articulation and revolute joint for the source side lever",
                "physical side-lever collider",
                "verification and reuse of the shared HS005 scene collision",
                "analytic mug body collider and recessed flat cube foot",
                "outlet metric and render-only water stream",
                "preservation of the source white ceramic PBR material closure",
            ],
            "source_faucet_articulated": False,
            "runtime_handle_visual_uses_source_geometry": True,
            "reuses_hs005_realistic_kitchen": True,
            "source_open_sink_collision_runtime_enabled": True,
            "runtime_safe_open_basin_floor": False,
            "uses_shared_scene_collision": True,
        },
        "runtime": {
            "faucet_handle_pivot_w_m": list(HANDLE_PIVOT_W),
            "outlet_position_w_m": list(OUTLET_POSITION_W),
            "mug_uniform_scale": MUG_RUNTIME_SCALE,
            "mug_rim_height_m": MUG_RIM_HEIGHT_M,
            "mug_up_axis_local": [0.0, 0.0, 1.0],
            "mug_home_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "safe_open_basin_floor": {
                "path": (
                    "/World/Task/B320Benchmark/faucet_fixture/"
                    "hs005_realistic_kitchen/Sink/Collider"
                ),
                "center_w_m": [-0.15, 0.0, 0.885],
                "size_m": [0.42, 0.72, 0.02],
                "top_z_m": 0.895,
                "purpose": "shared HS005 scene-owned analytic basin floor",
            },
            "source_mesh_split": split,
            "components": components,
        },
    }
    (asset_dir / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (asset_dir / "ATTRIBUTION.txt").write_text(
        '"Modern Apartment" by Visthétique is licensed under CC BY 4.0.\n'
        f"Source: {KITCHEN_SOURCE_URL}\n"
        f"License: {KITCHEN_LICENSE_URL}\n\n"
        '"Ceramic White Mug" by Ani 3D (anilai) is licensed under CC BY 4.0.\n'
        f"Source: {MUG_SOURCE_URL}\n"
        f"License: {MUG_LICENSE_URL}\n\n"
        "KA021 reuses B320-HS005's reviewed kitchen placement and open-sink visual. "
        "The shared HS005 scene owns the analytic basin floor, full-height walls, "
        "and open countertop collision used directly by KA021. It partitions the "
        "source faucet faces only to attach the exact "
        "side-lever geometry to runtime physics; it does not claim authorship of "
        "either downloaded visual model.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hs005-scene", type=Path, required=True)
    parser.add_argument("--mug-zip", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    args = parser.parse_args()
    hs005_scene = args.hs005_scene.resolve(strict=True)
    mug_archive = args.mug_zip.resolve(strict=True)
    asset_dir = args.asset_dir.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    kitchen_dir = hs005_scene.parent.parent / "assets" / "modern_apartment_kitchen"
    kitchen_usd = (kitchen_dir / "modern_apartment_kitchen.usd").resolve(strict=True)
    sink_collision = (kitchen_dir / "sink_collision_source.ply").resolve(strict=True)
    expected = {
        hs005_scene: HS005_SCENE_SHA256,
        kitchen_usd: KITCHEN_SOURCE_SHA256,
        sink_collision: SINK_COLLISION_SHA256,
        mug_archive: MUG_ARCHIVE_SHA256,
    }
    for source, digest in expected.items():
        if _sha256(source) != digest:
            raise RuntimeError(f"reviewed KA021 source hash changed: {source}")
    if asset_dir.exists():
        raise FileExistsError(f"refusing to replace existing asset: {asset_dir}")
    asset_dir.parent.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "enable_cameras": False})
    staging: Path | None = None
    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.kit.asset_converter")
        app.update()
        staging = Path(
            tempfile.mkdtemp(prefix=".ka021-hs005-build-", dir=asset_dir.parent)
        )
        shutil.copy2(mug_archive, staging / "ceramic_white_mug.zip")
        mug_source = staging / "ceramic_white_mug_source"
        _extract_reviewed_mug_archive(mug_archive, mug_source)
        faucet_components, split = _extract_hs005_faucet(
            kitchen_usd,
            static_output=staging / "faucet_static.usd",
            handle_output=staging / "faucet_handle.usd",
        )
        _convert_gltf(mug_source / "scene.gltf", staging / "ceramic_white_mug.usd")
        mug_component = _review_mug(staging / "ceramic_white_mug.usd")
        components = [*faucet_components, mug_component]
        _write_manifest(
            staging,
            repo_root=repo_root,
            hs005_scene=hs005_scene,
            kitchen_usd=kitchen_usd,
            sink_collision=sink_collision,
            components=components,
            split=split,
        )
        staging.replace(asset_dir)
        staging = None
        print(
            json.dumps(
                {"asset_dir": str(asset_dir), "components": components},
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
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
