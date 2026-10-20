#!/usr/bin/env python3
"""Convert the four user-selected HS005 glTF assets with Isaac Sim 5.1.

The downloaded source closures stay checked in beside the generated visual
USDs.  Physics is deliberately authored by ``generate_asset_layer.py`` so the
high-resolution render meshes never become dynamic triangle-mesh colliders.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any


ASSETS: tuple[dict[str, object], ...] = (
    {
        "id": "water_bottle",
        "directory": "cc_by_lukemodels75_water_bottle",
        "output": "water_bottle_visual.usd",
        "title": "Water Bottle 3D Model",
        "author": "LukeModels75",
        "author_url": "https://sketchfab.com/lucss_3d",
        "source_url": "https://sketchfab.com/3d-models/water-bottle-3d-model-c30a04cb09734471a75147b62877e7af",
        "archive": "water_bottle_3d_model.zip",
        "archive_sha256": "b85aefd87746787b4cf321797dc880f267aa7c9b8f5e8ee72c7b83e34077dde5",
    },
    {
        "id": "cleaning_brush",
        "directory": "cc_by_amhyde_cleaning_brush",
        "output": "cleaning_brush_visual.usd",
        "title": "Cleaning Brush",
        "author": "amhyde",
        "author_url": "https://sketchfab.com/amhyde",
        "source_url": "https://sketchfab.com/3d-models/cleaning-brush-277196dd4c634d44bdc0c1afc05d62f7",
        "archive": "cleaning_brush.zip",
        "archive_sha256": "4d566c7c98ce8eca8a6981bcb096c5b8cd6ab81cc5d01c03d76b6a7e997b845c",
    },
    {
        "id": "dishwashing_dispenser",
        "directory": "cc_by_mike_nov_dishwashing_dispenser",
        "output": "dishwashing_dispenser_visual.usd",
        "title": "DIsh washing liquid dispenser",
        "author": "Mike Nov",
        "author_url": "https://sketchfab.com/novmikvis",
        "source_url": "https://sketchfab.com/3d-models/dish-washing-liquid-dispenser-94907db381e5497ea2db0d5b1e8412c0",
        "archive": "dish_washing_liquid_dispenser.zip",
        "archive_sha256": "f11f776e355be49bb933c3ea2b43f52b9ee323f9a380fb6e3a63a42bf9407b01",
    },
    {
        "id": "dish_sponge",
        "directory": "cc_by_requital_dish_sponge",
        "output": "dish_sponge_visual.usd",
        "title": "Sponge For Washing Dishes",
        "author": "Requital (Azamkhon)",
        "author_url": "https://sketchfab.com/requital_0111",
        "source_url": "https://sketchfab.com/3d-models/sponge-for-washing-dishes-7ec3822721dd416aacc6308a7ed46c2c",
        "archive": "sponge_for_washing_dishes.zip",
        "archive_sha256": "2003f17f0913bb7383d48a369fa345fb5ea18f405ea4a6ecb9f51646e852a914",
    },
)

LICENSE_NAME = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
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


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def source_profile(source: Path) -> dict[str, object]:
    import numpy as np
    import trimesh

    scene = trimesh.load(str(source), force="scene", process=False)
    return {
        "bounds_source": np.asarray(scene.bounds, dtype=float).tolist(),
        "extents_source": np.asarray(scene.extents, dtype=float).tolist(),
        "geometry_count": len(scene.geometry),
        "mesh_node_count": len(scene.graph.nodes_geometry),
        "triangle_count": int(sum(len(mesh.faces) for mesh in scene.geometry.values())),
    }


async def convert_one(source: Path, output: Path) -> None:
    import omni.kit.asset_converter

    context = omni.kit.asset_converter.AssetConverterContext()
    for name, value in CONVERTER_CONTEXT.items():
        if not hasattr(context, name):
            raise RuntimeError(f"AssetConverterContext lacks required field {name!r}")
        setattr(context, name, value)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.usd")
    temporary.unlink(missing_ok=True)
    task = omni.kit.asset_converter.get_instance().create_converter_task(
        str(source), str(temporary), None, context
    )
    success = await task.wait_until_finished()
    if not success or not temporary.is_file() or temporary.stat().st_size <= 1024:
        detail = getattr(task, "get_error_message", lambda: "unknown converter error")()
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"HS005 glTF conversion failed for {source}: {detail}")
    os.replace(temporary, output)


def audit_and_stamp(output: Path, spec: dict[str, object]) -> dict[str, object]:
    from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"could not reopen converted HS005 asset: {output}")
    default = stage.GetDefaultPrim()
    if not default.IsValid():
        raise RuntimeError(f"converted HS005 asset lacks a default prim: {output}")

    forbidden: list[str] = []
    meshes: list[str] = []
    materials: list[str] = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(str(prim.GetPath()))
        if prim.IsA(UsdShade.Material):
            materials.append(str(prim.GetPath()))
        if prim.IsA(UsdGeom.Camera) or prim.IsA(UsdLux.BoundableLightBase):
            forbidden.append(str(prim.GetPath()))
        if (
            prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.HasAPI(UsdPhysics.CollisionAPI)
            or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ):
            forbidden.append(str(prim.GetPath()))
    if forbidden:
        raise RuntimeError(
            "downloaded visual asset must remain render-only: "
            + ", ".join(sorted(set(forbidden)))
        )
    if not meshes:
        raise RuntimeError(f"converted HS005 asset contains no mesh: {output}")

    default.SetCustomDataByKey("handson:assetRole", "render-only source geometry")
    default.SetCustomDataByKey("handson:sourceTitle", str(spec["title"]))
    default.SetCustomDataByKey("handson:sourceAuthor", str(spec["author"]))
    default.SetCustomDataByKey("handson:sourceUrl", str(spec["source_url"]))
    default.SetCustomDataByKey("handson:license", LICENSE_NAME)
    default.SetCustomDataByKey("handson:licenseUrl", LICENSE_URL)
    stage.GetRootLayer().Save()
    return {
        "default_prim": str(default.GetPath()),
        "mesh_count": len(meshes),
        "material_count": len(materials),
        "forbidden_schema_count": 0,
    }


def metadata_for(
    asset_dir: Path,
    output: Path,
    spec: dict[str, object],
    source: Path,
) -> dict[str, object]:
    source_files = sorted(path for path in (asset_dir / "source").rglob("*") if path.is_file())
    audit = audit_and_stamp(output, spec)
    return {
        "schema": "hs005_downloaded_visual_conversion_v1",
        "asset_id": spec["id"],
        "source": {
            "title": spec["title"],
            "author": spec["author"],
            "author_url": spec["author_url"],
            "source_url": spec["source_url"],
            "license": LICENSE_NAME,
            "license_url": LICENSE_URL,
            "archive_filename": spec["archive"],
            "archive_sha256": spec["archive_sha256"],
            "files": {
                str(path.relative_to(asset_dir)): sha256_file(path)
                for path in source_files
            },
            "profile": source_profile(source),
        },
        "converter": {
            "runtime": "Isaac Sim 5.1.0.0",
            "extension": "omni.kit.asset_converter 5.0.17",
            "context": CONVERTER_CONTEXT,
        },
        "output": {
            "filename": output.name,
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            **audit,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    task_dir = args.task_dir.resolve()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "enable_cameras": False})
    try:
        app.update()
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.kit.asset_converter")
        app.update()
        loop = asyncio.get_event_loop()
        reports: dict[str, Any] = {}
        for spec in ASSETS:
            asset_dir = task_dir / "assets" / str(spec["directory"])
            source = asset_dir / "source" / "scene.gltf"
            output = asset_dir / str(spec["output"])
            if not source.is_file():
                raise FileNotFoundError(source)
            loop.run_until_complete(convert_one(source, output))
            report = metadata_for(asset_dir, output, spec, source)
            atomic_json(asset_dir / "conversion.json", report)
            reports[str(spec["id"])] = report["output"]
        print(json.dumps(reports, indent=2, sort_keys=True), flush=True)
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
