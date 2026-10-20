#!/usr/bin/env python3
"""Sanitize the selected CC BY Modern Kitchen USD for FU015 preview use.

Run this with the repository's pinned Isaac Sim 5.1 Python after converting the
downloaded Sketchfab glTF with ``omni.kit.asset_converter``.  The script edits
only the supplied USD: branded label meshes are deactivated, provenance is
embedded as custom data, and any physics, camera, or light schema fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from isaacsim import SimulationApp


SOURCE_URL = (
    "https://sketchfab.com/3d-models/"
    "modern-kitchen-82c35500aec347d2938ff7943900486f"
)
AUTHOR = "Visthétique"
LICENSE_NAME = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
CONVERTED_ROOT = "/World/node_aca6747011741fc8bb8d24172fc7ec1_fbx/RootNode"
DEACTIVATED_RELATIVE_PATHS = (
    "Wine_bottles",
    "RLF_74925/RLF_74925_Logo_0",
    "TRB_84300/TRB_84300_TRB_843_00_TCS_01_0",
    "DVN_74030_TTC_BK/DVN_74030_TTC_BK_Front_0",
    "HLB_840/HLB_840_Front_001_0",
    "MS_622_BIS/MS_622_BIS_Front_002_0",
    "DSI_46750/DSI_46750_DSI_46750_Screen_0",
    "DSI_46750/DSI_46750_DSI_46750_Door_0",
)
SOURCE_MESH_COUNT = 155
SANITIZED_ACTIVE_MESH_COUNT = 144


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", type=Path)
    args = parser.parse_args()
    asset = args.asset.resolve()
    if not asset.is_file():
        raise FileNotFoundError(asset)

    app = SimulationApp({"headless": True})
    try:
        from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, Vt

        app.update()
        stage = Usd.Stage.Open(asset.as_posix())
        if stage is None:
            raise RuntimeError(f"could not open converted kitchen USD: {asset}")
        default = stage.GetDefaultPrim()
        if not default or str(default.GetPath()) != "/World":
            raise RuntimeError("converted kitchen USD must have /World as default prim")

        forbidden: list[str] = []
        input_active_mesh_count = 0
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                input_active_mesh_count += 1
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
                "Modern Kitchen visual USD contains forbidden scene schemas: "
                + ", ".join(sorted(set(forbidden)))
            )
        if input_active_mesh_count not in {
            SOURCE_MESH_COUNT,
            SANITIZED_ACTIVE_MESH_COUNT,
        }:
            raise RuntimeError(
                "unexpected active kitchen mesh count: expected source or "
                f"sanitized count, got {input_active_mesh_count}"
            )

        deactivated: list[str] = []
        for relative in DEACTIVATED_RELATIVE_PATHS:
            path = f"{CONVERTED_ROOT}/{relative}"
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"brand-cleanup prim is missing: {path}")
            prim.SetActive(False)
            deactivated.append(path)

        default.SetCustomDataByKey("handson:assetRole", "render-only background")
        default.SetCustomDataByKey("handson:sourceUrl", SOURCE_URL)
        default.SetCustomDataByKey("handson:author", AUTHOR)
        default.SetCustomDataByKey("handson:license", LICENSE_NAME)
        default.SetCustomDataByKey("handson:licenseUrl", LICENSE_URL)
        default.SetCustomDataByKey(
            "handson:deactivatedBrandMeshes",
            Vt.StringArray(deactivated),
        )
        stage.GetRootLayer().Save()

        # Reopen the crate before reporting success.  Isaac's shutdown logging
        # can obscure exceptions raised during USD value conversion, so the
        # persisted file itself is the source of truth for this sanitizer.
        reopened = Usd.Stage.Open(asset.as_posix())
        if reopened is None:
            raise RuntimeError(f"could not reopen sanitized kitchen USD: {asset}")
        reopened_default = reopened.GetDefaultPrim()
        persisted_paths = list(
            reopened_default.GetCustomDataByKey(
                "handson:deactivatedBrandMeshes"
            )
            or ()
        )
        if persisted_paths != deactivated:
            raise RuntimeError(
                "brand-cleanup provenance did not persist: "
                f"expected={deactivated!r}, got={persisted_paths!r}"
            )
        still_active = [
            path
            for path in deactivated
            if reopened.GetPrimAtPath(path).IsActive()
        ]
        if still_active:
            raise RuntimeError(
                "brand-cleanup prims remained active after reopen: "
                + ", ".join(still_active)
            )
        active_mesh_count = sum(
            prim.IsA(UsdGeom.Mesh) for prim in reopened.Traverse()
        )
        if active_mesh_count != SANITIZED_ACTIVE_MESH_COUNT:
            raise RuntimeError(
                "brand cleanup produced an unexpected active mesh set: "
                f"expected={SANITIZED_ACTIVE_MESH_COUNT}, "
                f"got={active_mesh_count}"
            )

        print(
            json.dumps(
                {
                    "asset": asset.as_posix(),
                    "sha256": sha256_file(asset),
                    "mesh_count_before_pruning": SOURCE_MESH_COUNT,
                    "input_active_mesh_count": input_active_mesh_count,
                    "active_mesh_count_after_pruning": active_mesh_count,
                    "deactivated_paths": deactivated,
                    "forbidden_schema_count": 0,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    except BaseException:
        # SimulationApp's fast close terminates the process with status zero,
        # which can mask a pending sanitizer exception.  Emit the traceback and
        # exit non-zero directly; the failed process owns no persistent stage.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        app.close(wait_for_replicator=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
