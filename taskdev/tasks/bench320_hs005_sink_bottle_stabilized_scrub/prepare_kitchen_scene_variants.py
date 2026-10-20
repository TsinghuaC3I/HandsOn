#!/usr/bin/env python3
"""Prepare the three remaining user-downloaded kitchen backgrounds in Isaac 5.1.

The original archive closure is retained below each task-local asset directory.
Converted USDs are render-only; HS005 continues to use its separately audited
sink triangle mesh and analytic dynamic-object colliders for physics.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import zipfile


TASK_DIR = Path(__file__).resolve().parent
DOWNLOADS = Path("/home/user/下载")
LICENSE_NAME = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

SCENES: tuple[dict[str, str], ...] = (
    {
        "id": "modern_kitchen",
        "archive": "modern_kitchen.zip",
        "archive_sha256": "369e98dd793cccfae7cbf998a2cc8d839006ba38d3791aa7690cf5248a29e24d",
        "directory": "cc_by_dylanheyes_modern_kitchen",
        "output": "modern_kitchen_visual.usd",
        "title": "Modern Kitchen",
        "author": "dylanheyes",
        "author_url": "https://sketchfab.com/dylanheyes",
        "source_url": "https://sketchfab.com/3d-models/modern-kitchen-9843a830b96142a9a53f45f25304d93c",
    },
    {
        "id": "realistic_kitchen_scene",
        "archive": "realistic_kitchen_scene.zip",
        "archive_sha256": "a7713fa10a6cfd0839c7a6d961ad4c71bd9ccde0235db590deac0ceb12e489fc",
        "directory": "cc_by_katydid_realistic_kitchen_scene",
        "output": "realistic_kitchen_scene_visual.usd",
        "title": "Realistic kitchen scene",
        "author": "Katydid",
        "author_url": "https://sketchfab.com/Katydid.",
        "source_url": "https://sketchfab.com/3d-models/realistic-kitchen-scene-2880a251b89e4f0dba4e4c5ae161c0c4",
    },
    {
        "id": "appartement",
        "archive": "appartement.zip",
        "archive_sha256": "c8b68631b82cd2d2472beb898b81088cf8dcfdd07b631fa9c6f642299db6f6e5",
        "directory": "cc_by_srmonteiro_appartement",
        "output": "appartement_visual.usd",
        "title": "Appartement",
        "author": "SrMonteiro",
        "author_url": "https://sketchfab.com/crispimrafael",
        "source_url": "https://sketchfab.com/3d-models/appartement-6a7a5fe208344b2e8123a88923dbd5b3",
    },
)

CONVERTER_CONTEXT = {
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def extract_archive(archive: Path, source_dir: Path) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe kitchen archive member: {member.filename}")
        bundle.extractall(source_dir)


def reduce_textures(textures_dir: Path, max_dimension: int = 1024) -> dict[str, int]:
    from PIL import Image

    resized = 0
    files = 0
    for path in sorted(item for item in textures_dir.rglob("*") if item.is_file()):
        if path.suffix.lower() not in {".jpeg", ".jpg", ".png"}:
            continue
        files += 1
        with Image.open(path) as image:
            if max(image.size) <= max_dimension:
                continue
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            save_args: dict[str, object] = {"optimize": True}
            if path.suffix.lower() in {".jpeg", ".jpg"}:
                save_args.update({"quality": 90, "subsampling": 0})
                image = image.convert("RGB")
            image.save(path, **save_args)
            resized += 1
    return {"file_count": files, "resized_count": resized, "max_dimension": max_dimension}


def source_profile(source: Path) -> dict[str, object]:
    import numpy as np
    import trimesh

    scene = trimesh.load(str(source), force="scene", process=False)
    triangle_count = sum(
        len(mesh.faces)
        for mesh in scene.geometry.values()
        if isinstance(mesh, trimesh.Trimesh)
    )
    return {
        "bounds_source": np.asarray(scene.bounds, dtype=float).tolist(),
        "extents_source": np.asarray(scene.extents, dtype=float).tolist(),
        "geometry_count": len(scene.geometry),
        "mesh_node_count": len(scene.graph.nodes_geometry),
        "triangle_count": int(triangle_count),
    }


async def convert_one(source: Path, output: Path) -> None:
    import omni.kit.asset_converter

    context = omni.kit.asset_converter.AssetConverterContext()
    for name, value in CONVERTER_CONTEXT.items():
        if not hasattr(context, name):
            raise RuntimeError(f"AssetConverterContext lacks required field {name!r}")
        setattr(context, name, value)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.usd")
    temporary.unlink(missing_ok=True)
    task = omni.kit.asset_converter.get_instance().create_converter_task(
        str(source), str(temporary), None, context
    )
    if not await task.wait_until_finished():
        detail = getattr(task, "get_error_message", lambda: "unknown error")()
        raise RuntimeError(f"kitchen conversion failed for {source}: {detail}")
    if not temporary.is_file() or temporary.stat().st_size <= 1024:
        raise RuntimeError(f"kitchen conversion produced no usable USD: {source}")
    os.replace(temporary, output)


def audit_usd(output: Path, spec: dict[str, str]) -> dict[str, object]:
    from pxr import Usd, UsdGeom, UsdLux, UsdPhysics

    stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"could not reopen converted kitchen: {output}")
    root = stage.GetDefaultPrim()
    if not root.IsValid():
        raise RuntimeError(f"converted kitchen has no default prim: {output}")
    meshes: list[str] = []
    forbidden: list[str] = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(str(prim.GetPath()))
        if prim.IsA(UsdGeom.Camera) or prim.IsA(UsdLux.BoundableLightBase):
            forbidden.append(str(prim.GetPath()))
        if (
            prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.HasAPI(UsdPhysics.CollisionAPI)
            or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ):
            forbidden.append(str(prim.GetPath()))
    if not meshes or forbidden:
        raise RuntimeError(
            f"invalid render-only kitchen {spec['id']}: "
            f"meshes={len(meshes)}, forbidden={forbidden}"
        )
    root.SetCustomDataByKey("handson:assetRole", "render-only kitchen background")
    root.SetCustomDataByKey("handson:sourceTitle", spec["title"])
    root.SetCustomDataByKey("handson:sourceAuthor", spec["author"])
    root.SetCustomDataByKey("handson:sourceUrl", spec["source_url"])
    root.SetCustomDataByKey("handson:license", LICENSE_NAME)
    root.SetCustomDataByKey("handson:licenseUrl", LICENSE_URL)
    stage.GetRootLayer().Save()
    return {
        "default_prim": str(root.GetPath()),
        "mesh_count": len(meshes),
        "forbidden_schema_count": 0,
    }


def source_markdown(spec: dict[str, str], archive: Path, profile: dict[str, object]) -> str:
    return f"""# {spec['title']}

Task-local render-only kitchen background prepared from the user's download.

- Author: {spec['author']} ({spec['author_url']})
- Source: {spec['source_url']}
- License: {LICENSE_NAME} ({LICENSE_URL})
- Original archive: `{archive}`
- Archive SHA256: `{spec['archive_sha256']}`
- Source mesh nodes: {profile['mesh_node_count']}
- Source triangles: {profile['triangle_count']}

Isaac Sim 5.1 converted the archive closure to Z-up, meter-stage USD with
cameras, lights, animation and all physics schemas excluded. Derived textures
are capped at 1024 pixels; the hash-bound original ZIP remains the rebuild
authority in the user's Downloads directory. The asset remains visual-only;
HS005 physics comes from its separate exact sink collision mesh and task-owned
analytic dynamic-object colliders.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, default=TASK_DIR)
    parser.add_argument("--downloads", type=Path, default=DOWNLOADS)
    args = parser.parse_args()
    task_dir = args.task_dir.resolve()
    downloads = args.downloads.resolve()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "enable_cameras": False})
    try:
        app.update()
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.kit.asset_converter")
        app.update()
        loop = asyncio.get_event_loop()
        report: dict[str, Any] = {}
        for spec in SCENES:
            archive = downloads / spec["archive"]
            if not archive.is_file():
                raise FileNotFoundError(archive)
            actual_archive_sha = sha256_file(archive)
            if actual_archive_sha != spec["archive_sha256"]:
                raise ValueError(
                    f"archive SHA mismatch for {archive}: {actual_archive_sha}"
                )
            asset_dir = task_dir / "assets" / spec["directory"]
            asset_dir.mkdir(parents=True, exist_ok=True)
            stale_source = asset_dir / "source"
            if stale_source.exists():
                shutil.rmtree(stale_source)
            textures_dir = asset_dir / "textures"
            if textures_dir.exists():
                shutil.rmtree(textures_dir)
            with tempfile.TemporaryDirectory(prefix=f"hs005-{spec['id']}-") as temp:
                source_dir = Path(temp)
                extract_archive(archive, source_dir)
                source = source_dir / "scene.gltf"
                profile = source_profile(source)
                license_text = (source_dir / "license.txt").read_text(
                    encoding="utf-8", errors="replace"
                )
                output = asset_dir / spec["output"]
                loop.run_until_complete(convert_one(source, output))
            texture_report = reduce_textures(textures_dir)
            audit = audit_usd(output, spec)
            metadata = {
                "schema": "hs005_kitchen_background_conversion_v1",
                "source": {
                    **spec,
                    "license": LICENSE_NAME,
                    "license_url": LICENSE_URL,
                    "profile": profile,
                },
                "converter": {
                    "runtime": "Isaac Sim 5.1.0.0",
                    "context": CONVERTER_CONTEXT,
                },
                "output": {
                    "filename": output.name,
                    "bytes": output.stat().st_size,
                    "sha256": sha256_file(output),
                    "textures": texture_report,
                    **audit,
                },
            }
            atomic_text(
                asset_dir / "conversion.json",
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
            )
            atomic_text(asset_dir / "SOURCE.md", source_markdown(spec, archive, profile))
            atomic_text(asset_dir / "LICENSE.txt", license_text)
            report[spec["id"]] = metadata["output"]
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
