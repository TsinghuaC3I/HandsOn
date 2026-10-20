#!/usr/bin/env python3
"""Prepare the reviewed Poly Haven cabinet for FU001 (Isaac Sim 5.1).

The downloaded archive is kept byte-for-byte under ``source/``.  The builder
converts the glTF with the Isaac asset converter, then emits three USD
components: a fixed cabinet body and the two independent door links.  Every
source mesh is also the collision mesh (PhysX convex decomposition is
requested on the source topology); no primitive placeholder is introduced.

Run with the Isaac Sim Python environment.  The script is intentionally
deterministic and refuses to replace an existing runtime directory.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

EXPECTED_ARCHIVE_SHA256 = "dec8c52b933705c3332619faf6ec8618f5e8ca4ff57c24f25016e9876eba0ddf"
SOURCE_MEMBERS = (
    "painted_wooden_cabinet_4k.gltf",
    "painted_wooden_cabinet.bin",
    "textures/painted_wooden_cabinet_arm_4k.jpg",
    "textures/painted_wooden_cabinet_diff_4k.jpg",
    "textures/painted_wooden_cabinet_nor_gl_4k.jpg",
)
SOURCE_NODES = {
    "painted_cabinet_body.usd": "/World/painted_wooden_cabinet",
    "painted_cabinet_door_right.usd": "/World/painted_wooden_cabinet_door_02",
    "painted_cabinet_door_left.usd": "/World/painted_wooden_cabinet_door_01",
}
DOOR_NODES = {
    "/World/painted_wooden_cabinet",
    "/World/painted_wooden_cabinet_drawer_01",
    "/World/painted_wooden_cabinet_drawer_02",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_member(name: str) -> str:
    if not name or "\\" in name or "\x00" in name:
        raise RuntimeError(f"unsafe archive member: {name!r}")
    p = PurePosixPath(name[:-1] if name.endswith("/") else name)
    if p.is_absolute() or any(part in {"", ".", ".."} for part in p.parts):
        raise RuntimeError(f"unsafe archive member: {name!r}")
    return p.as_posix()


def extract_source(archive: Path, destination: Path) -> None:
    if sha256(archive) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("downloaded FU001 archive hash is not the reviewed archive")
    with zipfile.ZipFile(archive) as zf:
        entries = {safe_member(i.filename): i for i in zf.infolist()}
        missing = sorted(set(SOURCE_MEMBERS) - set(entries))
        if missing:
            raise RuntimeError(f"source closure is incomplete: {missing}")
        for rel in SOURCE_MEMBERS:
            info = entries[rel]
            if info.is_dir() or info.file_size <= 0 or info.flag_bits & 1:
                raise RuntimeError(f"invalid source member: {rel}")
            out = destination / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def convert_gltf(source: Path, output: Path) -> None:
    import omni.kit.asset_converter

    ctx = omni.kit.asset_converter.AssetConverterContext()
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
    for key, value in settings.items():
        if not hasattr(ctx, key):
            raise RuntimeError(f"AssetConverterContext lacks {key!r}")
        setattr(ctx, key, value)
    tmp = output.with_suffix(".tmp.usd")
    task = omni.kit.asset_converter.get_instance().create_converter_task(
        str(source), str(tmp), None, ctx
    )
    ok = asyncio.get_event_loop().run_until_complete(task.wait_until_finished())
    if not ok or not tmp.is_file() or tmp.stat().st_size <= 1024:
        detail = getattr(task, "get_error_message", lambda: "unknown error")()
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"glTF conversion failed: {detail}")
    tmp.replace(output)


def author_component(converted: Path, output: Path, keep: str, remove: list[str]) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(converted), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"cannot open converted stage: {converted}")
    for path in remove:
        stage.RemovePrim(path)
    component = stage.GetPrimAtPath(keep)
    if not component.IsValid():
        raise RuntimeError(f"converted component missing: {keep}")
    # The body node has two sibling drawer nodes in the reviewed glTF.  They
    # are fixed geometry of the cabinet and must retain source-derived
    # collision too; collect every surviving mesh after the role-specific
    # removals rather than only descendants of the body node.
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    if not meshes:
        raise RuntimeError(f"component has no source meshes: {keep}")
    # Apply collision metadata directly to the source mesh.  The visual and
    # collision topology therefore share one lineage and one transform.
    for mesh in meshes:
        UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr().Set(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr().Set(
            "convexDecomposition"
        )
        mesh.SetCustomDataByKey(
            "handson:collisionDerivation",
            "source_exact_mesh_topology_then_physx_convex_decomposition",
        )
        mesh.SetCustomDataByKey("handson:collisionVisibleGeometry", True)
    root = stage.GetDefaultPrim()
    root.SetCustomDataByKey("handson:sourceAsset", "Poly Haven Painted Wooden Cabinet")
    root.SetCustomDataByKey("handson:sourceUrl", "https://polyhaven.com/a/painted_wooden_cabinet")
    root.SetCustomDataByKey("handson:license", "CC0 1.0")
    root.SetCustomDataByKey("handson:licenseUrl", "https://polyhaven.com/license")
    root.SetCustomDataByKey("handson:sourceUpAxis", "Y")
    root.SetCustomDataByKey("handson:runtimeUpAxis", "Z")
    root.SetCustomDataByKey("handson:runtimeOrientation", "+90 degrees about X")
    root.SetCustomDataByKey("handson:sourceNode", keep)
    root.SetCustomDataByKey("handson:collisionSource", "visible source meshes")
    root.SetCustomDataByKey("handson:units", "meters")
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().Export(str(output))
    return {
        "path": output.name,
        "source_node": keep,
        "mesh_count": len(meshes),
        "sha256": sha256(output),
        "size_bytes": output.stat().st_size,
        "collision": "source_exact_mesh_topology_then_physx_convex_decomposition",
    }


def build(archive: Path, asset_dir: Path) -> dict:
    runtime = asset_dir / "runtime"
    if runtime.exists():
        raise FileExistsError(f"refusing to replace existing runtime directory: {runtime}")
    asset_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".fu001-cabinet-build-", dir=asset_dir))
    try:
        source = staging / "source"
        extract_source(archive, source)
        converted = staging / "converted_scene.usd"
        convert_gltf(source / SOURCE_MEMBERS[0], converted)
        out_runtime = staging / "runtime"
        out_runtime.mkdir()
        # The converter writes material paths relative to each component USD.
        # Keep a byte-identical texture copy beside those components so the
        # runtime closure is self-contained while ``source/`` remains the
        # reviewed archive extraction.
        shutil.copytree(source / "textures", out_runtime / "textures")
        outputs = []
        for filename, node in SOURCE_NODES.items():
            remove = [p for p in SOURCE_NODES.values() if p != node]
            if node in {"/World/painted_wooden_cabinet_door_01", "/World/painted_wooden_cabinet_door_02"}:
                remove.extend(sorted(DOOR_NODES - {node}))
            outputs.append(author_component(converted, out_runtime / filename, node, remove))
        # The source closure remains byte-for-byte exactly as downloaded.
        manifest = {
            "schema": "handson.fu001.polyhaven-painted-cabinet-asset",
            "schema_version": 1,
            "task_id": "B320-FU001",
            "task_name": "b320_synchronized_double_cabinet_open",
            "provenance": {
                "title": "Painted Wooden Cabinet",
                "platform": "Poly Haven",
                "author": "Kirill Sannikov / Poly Haven",
                "source_url": "https://polyhaven.com/a/painted_wooden_cabinet",
                "license": "CC0 1.0",
                "license_url": "https://polyhaven.com/license",
                "archive_name": archive.name,
                "archive_sha256": EXPECTED_ARCHIVE_SHA256,
            },
            "source": {
                "members": [
                    {"path": rel, "sha256": sha256(source / rel), "size_bytes": (source / rel).stat().st_size}
                    for rel in SOURCE_MEMBERS
                ],
                "source_format": "glTF 2.0",
                "source_units": "meters",
                "source_up_axis": "Y",
                "independent_nodes": list(SOURCE_NODES.values()),
                "hinge_axis_source": "+Y",
                "hinge_axis_runtime": "+Z after +90 degrees about X",
                "measured_bounds_source_m": {
                    "width_x": 1.1947,
                    "height_y": 1.1819,
                    "depth_z": 0.5465,
                },
                "orientation_mapping_source_to_runtime": "(x,y,z) -> (x,-z,y)",
            },
            # The converter's +90-degree X stage map is followed by the
            # canonical B320 scene-frame rotation (Rz(-90 degrees)).  Keep
            # both maps explicit so a future importer cannot mistake the
            # intermediate Z-up coordinates for world placement.
            "scene_frame": {
                "converter_source_to_runtime": "(x,y,z) -> (x,-z,y)",
                "extra_rotation": "Rz(-90 degrees)",
                "orientation_wxyz": [0.5, 0.5, -0.5, -0.5],
                "source_to_world": "(x,y,z) -> (-z,-x,y)",
            },
            "conversion": {
                "isaac_sim": "5.1",
                "convert_stage_up_z": True,
                "use_meter_as_world_unit": True,
                "single_mesh": False,
                "merge_all_meshes": False,
                "embed_textures": False,
            },
            "runtime_components": outputs,
            "collision": {
                "policy": "source meshes carry collision APIs",
                "approximation": "convexDecomposition",
                "derivation": "source exact mesh topology then PhysX convex decomposition",
                "placeholder_primitives": False,
            },
        }
        (staging / "asset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "LICENSE.txt").write_text(
            "Poly Haven Painted Wooden Cabinet\n"
            "License: CC0 1.0 (commercial use, modification and redistribution permitted).\n"
            "Source: https://polyhaven.com/a/painted_wooden_cabinet\n"
            "License reference: https://polyhaven.com/license\n"
            "The downloaded archive did not include a license file; this record binds\n"
            "the reviewed archive hash to the upstream license page above.\n",
            encoding="utf-8",
        )
        (staging / "source").replace(asset_dir / "source") if not (asset_dir / "source").exists() else shutil.rmtree(staging / "source")
        (staging / "converted_scene.usd").replace(asset_dir / "converted_scene.usd")
        out_runtime.replace(runtime)
        (staging / "asset_manifest.json").replace(asset_dir / "asset_manifest.json")
        (staging / "LICENSE.txt").replace(asset_dir / "LICENSE.txt")
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    args = parser.parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "enable_cameras": False})
    try:
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("omni.kit.asset_converter")
        app.update()
        result = build(args.archive.resolve(), args.asset_dir.resolve())
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        app.close(wait_for_replicator=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
