#!/usr/bin/env python3
"""Prepare the removable-lid Poly Haven pot used by B320-KA012.

The reviewed glTF contains exactly two independent source nodes: the pot base
and the lid.  This builder preserves that split in two runtime USD files,
rebases each part onto its support plane, and authors a concavity-preserving
collision approximation for the open pot so a spoon can enter the interior.

Run with the Isaac Sim Python environment, for example::

    /home/user/isaacsim/python.sh scripts/prepare_ka012_polyhaven_pot_asset.py \
        --archive /path/to/pot_enamel_01_4k.gltf.zip \
        --asset-dir handson/handson/sim/envs/usd/b320_bimanual_benchmark/assets/polyhaven_pot_enamel_01
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import traceback
import zipfile


EXPECTED_ARCHIVE_SHA256 = (
    "bf50e5d0f73a720502447cbc7090ccf6fc8ddefb7fbe2dc1ded13b5053efd2ee"
)
SOURCE_URL = "https://polyhaven.com/a/pot_enamel_01"
SOURCE_CREATOR_URL = (
    "https://sketchfab.com/3d-models/"
    "retro-enamel-pot-with-flame-pattern-a30e9b9ae2bd4ad0be1f5e604a42489e"
)
LICENSE_NAME = "CC0 1.0"
LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"

SOURCE_MEMBERS = (
    "pot_enamel_01_4k.gltf",
    "pot_enamel_01.bin",
    "textures/pot_enamel_01_arm_4k.jpg",
    "textures/pot_enamel_01_diff_4k.jpg",
    "textures/pot_enamel_01_nor_gl_4k.jpg",
)
COMPONENTS = {
    "pot_base.usd": {
        "keep": "/World/pot_enamel_01_base",
        "remove": "/World/pot_enamel_01_lid",
        "support_y_m": 0.00014360249042510986,
        "collision": "convexDecomposition",
    },
    "pot_lid.usd": {
        "keep": "/World/pot_enamel_01_lid",
        "remove": "/World/pot_enamel_01_base",
        "support_y_m": 0.12897279858589172,
        "collision": "convexHull",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member(name: str) -> str:
    if not name or "\\" in name or "\x00" in name:
        raise RuntimeError(f"unsafe archive member: {name!r}")
    path = PurePosixPath(name[:-1] if name.endswith("/") else name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"unsafe archive member: {name!r}")
    return path.as_posix()


def _extract(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        entries = {_safe_member(info.filename): info for info in archive.infolist()}
        missing = set(SOURCE_MEMBERS) - set(entries)
        if missing:
            raise RuntimeError(f"pot source closure is incomplete: {sorted(missing)}")
        for relative in SOURCE_MEMBERS:
            info = entries[relative]
            if info.is_dir() or info.file_size <= 0 or info.flag_bits & 0x1:
                raise RuntimeError(f"invalid pot source member: {relative}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


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
        raise RuntimeError(f"Poly Haven pot conversion failed: {detail}")
    temporary.replace(output)


def _author_component(converted: Path, output: Path, spec: dict[str, object]) -> dict:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(converted), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"cannot open converted pot stage: {converted}")
    stage.RemovePrim(str(spec["remove"]))
    component = stage.GetPrimAtPath(str(spec["keep"]))
    if not component.IsValid():
        raise RuntimeError(f"converted pot component is missing: {spec['keep']}")

    translate = component.GetAttribute("xformOp:translate")
    if not translate.IsValid():
        raise RuntimeError(f"pot component lacks converter translation op: {spec['keep']}")
    translate.Set(Gf.Vec3d(0.0, -float(spec["support_y_m"]), 0.0))

    meshes = [prim for prim in Usd.PrimRange(component) if prim.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise RuntimeError(
            f"pot component must contain exactly one source mesh: {spec['keep']}"
        )
    collision = UsdPhysics.CollisionAPI.Apply(meshes[0])
    collision.CreateCollisionEnabledAttr().Set(True)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(meshes[0])
    mesh_collision.CreateApproximationAttr().Set(str(spec["collision"]))

    world = stage.GetDefaultPrim()
    world.SetCustomDataByKey("handson:sourceAsset", "Poly Haven Pot Enamel 01")
    world.SetCustomDataByKey("handson:sourceUrl", SOURCE_URL)
    world.SetCustomDataByKey("handson:license", LICENSE_NAME)
    world.SetCustomDataByKey("handson:component", component.GetName())
    world.SetCustomDataByKey("handson:sourceUpAxis", "Y")
    world.SetCustomDataByKey("handson:runtimeOrientation", "+90 degrees about X")
    world.SetCustomDataByKey("handson:supportRebased", True)
    stage.GetRootLayer().Export(str(output))

    reopened = Usd.Stage.Open(str(output), Usd.Stage.LoadAll)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )
    bounds = cache.ComputeWorldBound(reopened.GetDefaultPrim()).ComputeAlignedRange()
    return {
        "path": output.name,
        "sha256": _sha256(output),
        "size_bytes": output.stat().st_size,
        "source_node": component.GetName(),
        "collision_approximation": str(spec["collision"]),
        "bounds_source_xyz_m": {
            "min": list(map(float, bounds.GetMin())),
            "max": list(map(float, bounds.GetMax())),
        },
    }


def _write_manifest(asset_dir: Path, archive: Path, runtime: list[dict]) -> None:
    source_files = [
        {
            "path": relative,
            "sha256": _sha256(asset_dir / relative),
            "size_bytes": (asset_dir / relative).stat().st_size,
        }
        for relative in SOURCE_MEMBERS
    ]
    manifest = {
        "schema": "handson.ka012.polyhaven-pot-asset",
        "schema_version": 1,
        "task_id": "B320-KA012",
        "task_name": "t45_stir_pot",
        "provenance": {
            "title": "Pot Enamel 01",
            "platform": "Poly Haven",
            "source_url": SOURCE_URL,
            "creator_source_url": SOURCE_CREATOR_URL,
            "license": LICENSE_NAME,
            "license_url": LICENSE_URL,
            "download_archive": archive.name,
            "download_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        },
        "source_structure": {
            "independent_nodes": [
                "pot_enamel_01_base",
                "pot_enamel_01_lid",
            ],
            "lid_removable": True,
            "fully_modeled_interior": True,
            "source_up_axis": "Y",
        },
        "runtime": {
            "required_spawn_orientation": "+90 degrees about X",
            "uniform_scale": 0.78,
            "components": runtime,
        },
        "files": source_files,
    }
    (asset_dir / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve(strict=True)
    asset_dir = args.asset_dir.resolve()
    if _sha256(archive) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("downloaded Pot Enamel 01 archive hash is not reviewed")
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
        with tempfile.TemporaryDirectory(prefix="ka012-pot-source-") as temp_name:
            conversion_dir = Path(temp_name)
            _extract(archive, conversion_dir)
            converted = conversion_dir / "converted" / "pot_enamel_01.usd"
            converted.parent.mkdir()
            _convert_gltf(conversion_dir / SOURCE_MEMBERS[0], converted)

            staging = Path(
                tempfile.mkdtemp(prefix=".ka012-pot-build-", dir=asset_dir.parent)
            )
            runtime = [
                _author_component(converted, staging / name, spec)
                for name, spec in COMPONENTS.items()
            ]
            # The converter may rewrite temporary textures.  Re-extract the
            # reviewed source closure after conversion so committed bytes stay exact.
            _extract(archive, staging)
            _write_manifest(staging, archive, runtime)
            staging.replace(asset_dir)
            staging = None
            print(
                json.dumps(
                    {"asset_dir": str(asset_dir), "runtime": runtime},
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
