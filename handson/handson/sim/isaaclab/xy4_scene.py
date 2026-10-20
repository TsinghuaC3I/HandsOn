"""XY4-only composition of unchanged HS005 and KA021 assets.

The XY clearance screen is conservative layout sampling, not a live collision
or reachability certificate. Those require the final robot/USD/PhysX scene.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .xy4_contract import TASK_NAME

MUG_HOME_XY = (-0.20, 0.0)
# Keep the original outside-sink pad near its authored home, in the main view.
PARK_CENTER_BOUNDS_XY = ((-0.444, -0.140), (-0.440, -0.100))
PAIR_CLEARANCE_M = 0.010
RESET_MAX_ATTEMPTS = 128
# Exact source brush extent and a conservative cylinder around its visual.
BRUSH_EXTENT_X = (-0.0936167649, 0.1907145379)
BRUSH_RADIUS_M = 0.0140212401
BOTTLE_VISUAL_RADIUS_M = 0.035057
MUG_VISUAL_RADIUS_M = 0.074  # Source KA021 includes the complete handle.


def _point_segment_distance(point, start, end) -> float:
    delta = end - start
    t = np.clip(np.dot(point - start, delta) / np.dot(delta, delta), 0.0, 1.0)
    return float(np.linalg.norm(point - start - t * delta))


def layout_clearances(poses: dict[str, np.ndarray]) -> dict[str, float]:
    bottle = np.asarray(poses["hs005_bottle"], dtype=float)
    brush = np.asarray(poses["hs005_brush"], dtype=float)
    cup = np.asarray(poses["ceramic_mug"], dtype=float)
    w, x, y, z = brush[3:7]
    axis = np.asarray([1 - 2 * (y*y + z*z), 2 * (x*y + w*z)])
    start = brush[:2] + BRUSH_EXTENT_X[0] * axis
    end = brush[:2] + BRUSH_EXTENT_X[1] * axis
    return {
        "bottle_mug": float(np.linalg.norm(bottle[:2] - cup[:2]))
            - BOTTLE_VISUAL_RADIUS_M - MUG_VISUAL_RADIUS_M,
        "brush_mug": _point_segment_distance(cup[:2], start, end)
            - BRUSH_RADIUS_M - MUG_VISUAL_RADIUS_M,
        "bottle_brush": _point_segment_distance(bottle[:2], start, end)
            - BOTTLE_VISUAL_RADIUS_M - BRUSH_RADIUS_M,
        # The selected open basin's inner walls, not its wider hidden floor.
        "brush_basin": float(min(
            *(np.minimum(start, end) - np.asarray([-0.340, -0.200])),
            *(np.asarray([0.040, 0.200]) - np.maximum(start, end)),
        )) - BRUSH_RADIUS_M,
    }


def validate_xy4_asset_files() -> tuple[Path, ...]:
    """Check only the source closures actually composed by XY4."""
    from .b320_bimanual_scene import (
        B320_ASSET_DIRECTORY, B320_SOURCE_BUNDLE_SHA256,
        KA021_ASSET_BUNDLE_KEY, KA021_HS005_SCENE_PATH,
        _source_bundle_sha256, validate_b320_ka021_faucet_asset_files,
    )

    files = list(validate_b320_ka021_faucet_asset_files())
    if _source_bundle_sha256(KA021_ASSET_BUNDLE_KEY) != B320_SOURCE_BUNDLE_SHA256[KA021_ASSET_BUNDLE_KEY]:
        raise ValueError("XY4's unchanged KA021 mug/source bundle hash differs")
    files.extend((B320_ASSET_DIRECTORY / KA021_ASSET_BUNDLE_KEY).rglob("*"))
    task_dir = KA021_HS005_SCENE_PATH.parent.parent
    manifest = json.loads((task_dir / "scene/scene_variants.json").read_text())
    entries = [manifest["base"], *manifest["source_assets"].values()]
    for entry in entries:
        path = task_dir / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"XY4 HS005 source asset hash differs: {path}")
        files.append(path)
    return tuple(sorted({p for p in files if p.is_file()}))


def configure_scrub_props(stage, kitchen_path: str) -> None:
    """Preserve HS005 bodies and bind its original 0.75 friction material."""
    from pxr import Usd, UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, f"{kitchen_path}/Looks/HS005PhysicsMaterial")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr().Set(0.75)
    api.CreateDynamicFrictionAttr().Set(0.75)
    api.CreateRestitutionAttr().Set(0.0)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(kitchen_path)):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics"
            )
    for name in ("Bottle", "Brush"):
        prim = stage.GetPrimAtPath(f"{kitchen_path}/{name}")
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"XY4 is missing the original dynamic HS005 {name}")
        prim.SetCustomDataByKey("handson:composition", TASK_NAME)
