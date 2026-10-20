"""Shared composition contract for the vendored render-only kitchen scene."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .simulation_scene_catalog import (
    FU015_MODERN_KITCHEN_SCENE,
    SIMULATION_SCENE_ENV,
    selected_simulation_scene,
)

REALISTIC_KITCHEN_SOURCE_NODE = "node_aca6747011741fc8bb8d24172fc7ec1_fbx"
REALISTIC_KITCHEN_ASSET_RELATIVE_PATH = Path(
    "sim/envs/usd/modern_kitchen/modern_kitchen_visual.usd"
)
REALISTIC_KITCHEN_ASSET_SHA256 = (
    "babf31ddcaa34b81dd669df0b99fe1cdbddccb15bb29f7c10f0a266d3d8a5cb3"
)
REALISTIC_KITCHEN_ASSET_BYTES = 69_996_564
REALISTIC_KITCHEN_TEXTURE_COUNT = 59
REALISTIC_KITCHEN_TEXTURE_BYTES = 98_093_376
REALISTIC_KITCHEN_MESH_COUNT = 144
REALISTIC_KITCHEN_SOURCE_URL = (
    "https://sketchfab.com/3d-models/"
    "modern-kitchen-82c35500aec347d2938ff7943900486f"
)
REALISTIC_KITCHEN_LICENSE = "CC BY 4.0"
REALISTIC_KITCHEN_PREVIEW_ENV = "HANDSON_FU015_KITCHEN_PREVIEW"
REALISTIC_KITCHEN_PREVIEW_ROOT = "/World/VisualEnvironment/FU015Kitchen"
# The source glTF is meter-scale Y-up.  This audited transform aligns the
# island counter with the canonical B320 table at x=-0.15 and z=0.89 m.
REALISTIC_KITCHEN_SCALE = 0.9801319543941297
REALISTIC_KITCHEN_TRANSLATE_M = (
    -1.3987282953082514,
    -0.13492790523775905,
    -0.17594446726524463,
)
REALISTIC_KITCHEN_ISLAND_MIN_M = (
    -0.5876612619914738,
    -0.8797517402848549,
    -0.026023,
)
REALISTIC_KITCHEN_ISLAND_MAX_M = (
    0.287661261991474,
    0.8797517402848549,
    0.89,
)


def realistic_kitchen_preview_enabled(
    environ: Mapping[str, str] | None = None,
    *,
    default: bool = False,
) -> bool:
    """Read the backward-compatible opt-in flag used by every Isaac task."""

    source = os.environ if environ is None else environ
    if SIMULATION_SCENE_ENV in source:
        return (
            selected_simulation_scene(source)
            == FU015_MODERN_KITCHEN_SCENE
        )
    raw = source.get(REALISTIC_KITCHEN_PREVIEW_ENV)
    if raw is None:
        return bool(default)
    if raw == "1":
        return True
    if raw == "0":
        return False
    raise ValueError(
        f"{REALISTIC_KITCHEN_PREVIEW_ENV} must be exactly '1' or '0'; "
        f"got {raw!r}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_realistic_kitchen_asset(
    repository_root: Path | None = None,
) -> Path:
    """Validate the complete CC BY render closure before Kit starts."""

    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    asset_path = root / REALISTIC_KITCHEN_ASSET_RELATIVE_PATH
    if not asset_path.is_file():
        raise FileNotFoundError(
            "Realistic kitchen asset is missing: " + str(asset_path)
        )
    if asset_path.stat().st_size != REALISTIC_KITCHEN_ASSET_BYTES:
        raise ValueError(
            "Realistic kitchen asset size mismatch: expected "
            f"{REALISTIC_KITCHEN_ASSET_BYTES}, got {asset_path.stat().st_size}"
        )
    actual_sha = _sha256(asset_path)
    if actual_sha != REALISTIC_KITCHEN_ASSET_SHA256:
        raise ValueError(
            "Realistic kitchen SHA mismatch: expected "
            f"{REALISTIC_KITCHEN_ASSET_SHA256}, got {actual_sha}"
        )

    asset_dir = asset_path.parent
    metadata_path = asset_dir / "modern_kitchen_visual.conversion.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Realistic kitchen metadata is invalid: {metadata_path}"
        ) from exc
    output = metadata.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("Realistic kitchen metadata is missing output")
    expected_output = {
        "filename": asset_path.name,
        "sha256": REALISTIC_KITCHEN_ASSET_SHA256,
        "bytes": REALISTIC_KITCHEN_ASSET_BYTES,
        "external_texture_count": REALISTIC_KITCHEN_TEXTURE_COUNT,
        "external_texture_bytes": REALISTIC_KITCHEN_TEXTURE_BYTES,
    }
    for name, expected in expected_output.items():
        if output.get(name) != expected:
            raise ValueError(
                "Realistic kitchen metadata mismatch for "
                f"{name}: expected {expected!r}, got {output.get(name)!r}"
            )

    textures_dir = asset_dir / "textures"
    if not textures_dir.is_dir():
        raise FileNotFoundError(
            "Realistic kitchen texture directory is missing: "
            + str(textures_dir)
        )
    textures = sorted(path for path in textures_dir.iterdir() if path.is_file())
    if len(textures) != REALISTIC_KITCHEN_TEXTURE_COUNT:
        raise ValueError(
            "Realistic kitchen texture count mismatch: expected "
            f"{REALISTIC_KITCHEN_TEXTURE_COUNT}, got {len(textures)}"
        )
    texture_bytes = sum(path.stat().st_size for path in textures)
    if texture_bytes != REALISTIC_KITCHEN_TEXTURE_BYTES:
        raise ValueError(
            "Realistic kitchen texture bytes mismatch: expected "
            f"{REALISTIC_KITCHEN_TEXTURE_BYTES}, got {texture_bytes}"
        )
    unsupported = [
        path.name
        for path in textures
        if path.suffix.lower() not in {".jpeg", ".png"}
    ]
    if unsupported:
        raise ValueError(
            "Realistic kitchen has unsupported texture files: "
            + ", ".join(unsupported)
        )
    return asset_path


def realistic_kitchen_metadata(*, root_path: str) -> dict[str, Any]:
    """Return a stable provenance schema before or after live composition."""

    return {
        "asset_relative_path": str(REALISTIC_KITCHEN_ASSET_RELATIVE_PATH),
        "asset_sha256": REALISTIC_KITCHEN_ASSET_SHA256,
        "asset_bytes": REALISTIC_KITCHEN_ASSET_BYTES,
        "texture_count": REALISTIC_KITCHEN_TEXTURE_COUNT,
        "texture_bytes": REALISTIC_KITCHEN_TEXTURE_BYTES,
        "source_url": REALISTIC_KITCHEN_SOURCE_URL,
        "license": REALISTIC_KITCHEN_LICENSE,
        "root_path": str(root_path),
        "mesh_count": REALISTIC_KITCHEN_MESH_COUNT,
        "island_min_m": list(REALISTIC_KITCHEN_ISLAND_MIN_M),
        "island_max_m": list(REALISTIC_KITCHEN_ISLAND_MAX_M),
        "physical_table_min_m": None,
        "physical_table_max_m": None,
        "hidden_support_paths": [],
        "render_only": True,
        "collision_mode": "canonical_table_and_floor_only",
    }


@dataclass(frozen=True)
class RealisticKitchenReport:
    """Auditable result of composing one render-only kitchen layer."""

    asset_path: Path
    root_path: str
    mesh_count: int
    island_min_m: tuple[float, float, float]
    island_max_m: tuple[float, float, float]
    physical_table_min_m: tuple[float, float, float]
    physical_table_max_m: tuple[float, float, float]
    hidden_support_paths: tuple[str, ...]

    def replay_metadata(self) -> dict[str, Any]:
        metadata = realistic_kitchen_metadata(root_path=self.root_path)
        metadata.update(
            {
                "mesh_count": self.mesh_count,
                "island_min_m": list(self.island_min_m),
                "island_max_m": list(self.island_max_m),
                "physical_table_min_m": list(self.physical_table_min_m),
                "physical_table_max_m": list(self.physical_table_max_m),
                "hidden_support_paths": list(self.hidden_support_paths),
            }
        )
        return metadata


def compose_realistic_kitchen_visual(
    stage: Any,
    *,
    root_path: str,
    table_prim_path: str,
    floor_prim_path: str,
    transform_op_name: str,
    log_label: str,
    align_to_table: bool = False,
) -> RealisticKitchenReport:
    """Compose the existing CC BY kitchen without adding physical schemas."""

    if not str(root_path).startswith("/World/VisualEnvironment/"):
        raise ValueError(
            "Realistic kitchen must live below /World/VisualEnvironment/: "
            f"{root_path!r}"
        )

    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

    table = stage.GetPrimAtPath(table_prim_path)
    if not table.IsValid():
        raise RuntimeError(
            f"Realistic kitchen cannot find canonical table: {table_prim_path}"
        )
    support_bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    table_range = support_bbox_cache.ComputeWorldBound(table).ComputeAlignedRange()
    table_min = np.asarray(table_range.GetMin(), dtype=np.float64)
    table_max = np.asarray(table_range.GetMax(), dtype=np.float64)

    asset_path = validate_realistic_kitchen_asset()
    asset_root_path = f"{root_path}/Asset"

    root = UsdGeom.Xform.Define(stage, root_path)
    root_prim = root.GetPrim()
    root_prim.SetCustomDataByKey(
        "handson:sourceAsset",
        str(REALISTIC_KITCHEN_ASSET_RELATIVE_PATH),
    )
    root_prim.SetCustomDataByKey(
        "handson:sourceSha256",
        REALISTIC_KITCHEN_ASSET_SHA256,
    )
    root_prim.SetCustomDataByKey("handson:sourceUrl", REALISTIC_KITCHEN_SOURCE_URL)
    root_prim.SetCustomDataByKey("handson:license", REALISTIC_KITCHEN_LICENSE)
    root_prim.SetCustomDataByKey("handson:renderOnly", True)

    alignment_offset = np.zeros(3, dtype=np.float64)
    if align_to_table:
        expected_min = np.asarray(
            REALISTIC_KITCHEN_ISLAND_MIN_M,
            dtype=np.float64,
        )
        expected_max = np.asarray(
            REALISTIC_KITCHEN_ISLAND_MAX_M,
            dtype=np.float64,
        )
        alignment_offset[:2] = (
            0.5 * (table_min[:2] + table_max[:2])
            - 0.5 * (expected_min[:2] + expected_max[:2])
        )
        alignment_offset[2] = table_max[2] - expected_max[2]
    tx, ty, tz = (
        np.asarray(REALISTIC_KITCHEN_TRANSLATE_M, dtype=np.float64)
        + alignment_offset
    )
    scale = REALISTIC_KITCHEN_SCALE
    transform = Gf.Matrix4d(
        0.0,
        -scale,
        0.0,
        0.0,
        0.0,
        0.0,
        scale,
        0.0,
        -scale,
        0.0,
        0.0,
        0.0,
        tx,
        ty,
        tz,
        1.0,
    )
    root_xform = UsdGeom.Xformable(root)
    root_xform.ClearXformOpOrder()
    root_xform.AddTransformOp(
        UsdGeom.XformOp.PrecisionDouble,
        transform_op_name,
    ).Set(transform)

    asset_root = UsdGeom.Xform.Define(stage, asset_root_path)
    asset_root.GetPrim().GetReferences().ClearReferences()
    asset_root.GetPrim().GetReferences().AddReference(asset_path.as_posix())

    forbidden_schemas: list[str] = []
    active_mesh_count = 0
    for prim in Usd.PrimRange(asset_root.GetPrim()):
        if prim.IsA(UsdGeom.Mesh):
            active_mesh_count += 1
        if prim.IsA(UsdGeom.Camera) or prim.IsA(UsdLux.BoundableLightBase):
            forbidden_schemas.append(str(prim.GetPath()))
        if (
            prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.HasAPI(UsdPhysics.CollisionAPI)
            or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ):
            forbidden_schemas.append(str(prim.GetPath()))
    if forbidden_schemas:
        raise RuntimeError(
            "Realistic kitchen must remain render-only; forbidden schemas: "
            + ", ".join(sorted(set(forbidden_schemas)))
        )
    if active_mesh_count < 100:
        raise RuntimeError(
            "Realistic kitchen composed an incomplete visual scene: "
            f"active_meshes={active_mesh_count}"
        )

    island_path = (
        f"{asset_root_path}/{REALISTIC_KITCHEN_SOURCE_NODE}/RootNode/Island"
    )
    island = stage.GetPrimAtPath(island_path)
    if not island.IsValid():
        raise RuntimeError(f"Realistic kitchen island is missing: {island_path}")
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    island_range = bbox_cache.ComputeWorldBound(island).ComputeAlignedRange()
    island_min = np.asarray(island_range.GetMin(), dtype=np.float64)
    island_max = np.asarray(island_range.GetMax(), dtype=np.float64)
    expected_min = (
        np.asarray(REALISTIC_KITCHEN_ISLAND_MIN_M, dtype=np.float64)
        + alignment_offset
    )
    expected_max = (
        np.asarray(REALISTIC_KITCHEN_ISLAND_MAX_M, dtype=np.float64)
        + alignment_offset
    )
    if not (
        np.allclose(island_min, expected_min, atol=0.002, rtol=0.0)
        and np.allclose(island_max, expected_max, atol=0.002, rtol=0.0)
    ):
        raise RuntimeError(
            "Realistic kitchen island no longer matches the canonical table: "
            f"min={island_min.tolist()}, max={island_max.tolist()}"
        )

    island_covers_table_xy = bool(
        np.all(island_min[:2] <= table_min[:2] + 0.002)
        and np.all(island_max[:2] >= table_max[:2] - 0.002)
    )
    if not island_covers_table_xy or not np.isclose(
        island_max[2], table_max[2], atol=0.002, rtol=0.0
    ):
        raise RuntimeError(
            "Realistic kitchen island does not cover the canonical physical "
            f"table: island=({island_min.tolist()}, {island_max.tolist()}), "
            f"table=({table_min.tolist()}, {table_max.tolist()})"
        )

    hidden_supports: list[str] = []
    for path in dict.fromkeys((str(table_prim_path), str(floor_prim_path))):
        support = stage.GetPrimAtPath(path)
        if not support.IsValid():
            raise RuntimeError(
                f"Realistic kitchen cannot find canonical support: {path}"
            )
        UsdGeom.Imageable(support).MakeInvisible()
        hidden_supports.append(path)

    report = RealisticKitchenReport(
        asset_path=asset_path,
        root_path=root_path,
        mesh_count=active_mesh_count,
        island_min_m=tuple(float(value) for value in island_min),
        island_max_m=tuple(float(value) for value in island_max),
        physical_table_min_m=tuple(float(value) for value in table_min),
        physical_table_max_m=tuple(float(value) for value in table_max),
        hidden_support_paths=tuple(hidden_supports),
    )
    print(
        f"[HandsOn/{log_label}] Modern Kitchen visual ready: "
        f"meshes={report.mesh_count}, island_min={list(report.island_min_m)}, "
        f"island_max={list(report.island_max_m)}, "
        "collisions=canonical_table_floor",
        flush=True,
    )
    return report


__all__ = [
    "REALISTIC_KITCHEN_ASSET_BYTES",
    "REALISTIC_KITCHEN_ASSET_RELATIVE_PATH",
    "REALISTIC_KITCHEN_ASSET_SHA256",
    "REALISTIC_KITCHEN_ISLAND_MAX_M",
    "REALISTIC_KITCHEN_ISLAND_MIN_M",
    "REALISTIC_KITCHEN_LICENSE",
    "REALISTIC_KITCHEN_MESH_COUNT",
    "REALISTIC_KITCHEN_PREVIEW_ENV",
    "REALISTIC_KITCHEN_PREVIEW_ROOT",
    "REALISTIC_KITCHEN_SCALE",
    "REALISTIC_KITCHEN_SOURCE_URL",
    "REALISTIC_KITCHEN_TEXTURE_BYTES",
    "REALISTIC_KITCHEN_TEXTURE_COUNT",
    "REALISTIC_KITCHEN_TRANSLATE_M",
    "RealisticKitchenReport",
    "compose_realistic_kitchen_visual",
    "realistic_kitchen_metadata",
    "realistic_kitchen_preview_enabled",
    "validate_realistic_kitchen_asset",
]
