"""GEN4 source-asset composition and seeded layouts, before first PhysX reset."""

from __future__ import annotations

import json
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np

from ...tasks.eval_gen04_pass_glue_stick.contract import (
    DISTRACTORS, GLUE_BODY, GLUE_XY, GLUE_XY_HALF_RANGE, GLUE_YAW_DEG,
    TABLE_TOP_M, TASK_NAME, TRAY_BODY, TRAY_XY, TRAY_YAW_DEG,
)

ASSET_DIR = Path(__file__).resolve().parents[1] / "envs" / "usd" / TASK_NAME
TRAY_ASSET = ASSET_DIR.parent / "loaded_tray_gate_assets" / "bc005_selected_wooden_tray.usd"
# compose_gen4_assets replaces the legacy block with the selected glue USD.
# Both construction and seeded reset must support that composed collider.


def load_geometry(*, required=True):
    path = ASSET_DIR / "asset_geometry.json"
    if not path.is_file() and not required:
        return None
    if not path.is_file():
        raise FileNotFoundError(
            f"GEN4 selected glue-stick asset has not been integrated: {path}"
        )
    geometry = json.loads(path.read_text())
    points = np.asarray(geometry["glue"]["occupied_vertices_m"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError("GEN4 asset geometry must contain finite source-derived vertices")
    return geometry


def sample_layout(rng, *, randomize, geometry):
    """Nonzero XY jitter for every free body; the tray is a fixed fixture.

    Keep D03's background identities and its separated partitions. Move only
    the spoon and lower small-object slots clear of the closer receiving tray.
    This sampler is a proposed layout, not a substitute for collision queries.
    """
    glue_xy = np.asarray(GLUE_XY, dtype=float)
    if randomize:
        glue_xy += rng.uniform(-np.asarray(GLUE_XY_HALF_RANGE), GLUE_XY_HALF_RANGE)
    positions = {
        "d03_flat_screwdriver": (-0.42, 0.45),
        "d03_phillips_screwdriver": (0.08, 0.50),
        "d03_spoon": (-0.43, -0.65),
        "d03_plastic_bolt": (-0.43, 0.08),
        "d03_a_marbles": (0.08, 0.72),
        "d03_b_marble_1": (0.08, 0.08),
        "d03_b_marble_2": (0.08, 0.32),
        "d03_hammer": (-0.18, 0.60),
        "d03_nails": (-0.18, -0.65),
        "d03_bowl": (0.12, -0.12),
    }
    support_z = {
        "d03_flat_screwdriver": 0.896, "d03_phillips_screwdriver": 0.896,
        "d03_spoon": 0.894, "d03_plastic_bolt": 0.895,
        "d03_a_marbles": 0.927, "d03_b_marble_1": 0.9075,
        "d03_b_marble_2": 0.9075, "d03_hammer": 0.904,
        "d03_nails": 0.8922, "d03_bowl": 0.916,
    }
    layout = {}
    for name in DISTRACTORS:
        xy = np.asarray(positions[name], dtype=float)
        if randomize:
            xy += rng.uniform(-0.004, 0.004, size=2)
        yaw = float(rng.uniform(-15, 15)) if randomize else 0.0
        measured = geometry.get("distractors", {}).get(name)
        z = TABLE_TOP_M - float(measured["bounds_m"][0][2]) if measured else support_z[name]
        layout[name] = {"xy": xy.tolist(), "z": z, "yaw_degrees": yaw}
    layout[GLUE_BODY] = {
        "xy": glue_xy.tolist(),
        "z": TABLE_TOP_M - float(geometry["glue"]["bounds_m"][0][2]) + 0.0005,
        "yaw_degrees": float(rng.uniform(-GLUE_YAW_DEG, GLUE_YAW_DEG)) if randomize else 0.0,
    }
    layout[TRAY_BODY] = {
        "xy": list(TRAY_XY),
        "z": TABLE_TOP_M - float(geometry["tray"]["bounds_m"][0][2]),
        "yaw_degrees": TRAY_YAW_DEG,
    }
    return layout


def prepare_gen4_mjcf(root):
    """Replace reference object bodies using measured asset frames.

    The existing source geoms only anchor the MJCF body import. Both visual and
    collision scopes are removed and replaced with final source USD geometry
    by compose_gen4_assets before any simulation reset or preview physics.
    """
    geometry = load_geometry()
    layout = sample_layout(np.random.RandomState(0), randomize=False, geometry=geometry)
    rename = {"d03_target_root": GLUE_BODY, "d03_box": TRAY_BODY}
    for body in root.iter("body"):
        old_name = body.get("name")
        name = rename.get(old_name, old_name)
        if old_name in rename:
            body.set("name", name)
        if name not in layout:
            continue
        pose = layout[name]
        body.set("pos", " ".join(str(v) for v in (*pose["xy"], pose["z"])))
        angle = math.radians(pose["yaw_degrees"]) / 2
        body.set("quat", f"{math.cos(angle)} 0 0 {math.sin(angle)}")
        if name == GLUE_BODY or name in geometry.get("distractors", {}):
            measured = geometry["glue"] if name == GLUE_BODY else geometry["distractors"][name]
            if name == GLUE_BODY:
                body.find("freejoint").set("name", "gen4_glue_free_joint")
            inertial = body.find("inertial")
            if inertial is None:
                inertial = ET.SubElement(body, "inertial")
            inertial.set("mass", str(measured["mass_kg"]))
            inertial.set("pos", " ".join(map(str, measured["center_of_mass_m"])))
            inertial.set("quat", " ".join(map(str, measured["principal_axes_wxyz"])))
            inertial.set("diaginertia", " ".join(map(str, measured["diagonal_inertia_kg_m2"])))


def compose_gen4_assets(stage, task_root_path):
    """Mount the selected assets on imported owners before first physics."""
    from pxr import Usd, UsdGeom, UsdPhysics

    geometry = load_geometry()
    assets = [(GLUE_BODY, ASSET_DIR / "glue_stick.usd", None), (TRAY_BODY, TRAY_ASSET, None)]
    assets.extend((name, ASSET_DIR / "clutter_collisions.usd", f"/Clutter/{name}") for name in DISTRACTORS)
    for name, asset, source_prim in assets:
        if not asset.is_file():
            raise FileNotFoundError(f"GEN4 source-derived asset is missing: {asset}")
        root = stage.GetPrimAtPath(task_root_path)
        owners = [p for p in Usd.PrimRange(root) if p.GetName() == name and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(owners) != 1:
            raise RuntimeError(f"GEN4 requires one physical owner for {name}, found {len(owners)}")
        owner = owners[0]
        if name in geometry["distractors"]:
            # Match the source-derived collider's support-frame normalization
            # on existing imported visuals, without touching upstream assets.
            rotation = geometry["distractors"][name].get("source_normalization_rotation")
            visual = stage.GetPrimAtPath(str(owner.GetPath()) + "/visuals")
            if rotation is not None and visual.IsValid():
                from pxr import Gf
                op = UsdGeom.Xformable(visual).MakeMatrixXform()
                op.Set(Gf.Matrix4d(Gf.Matrix3d(*np.asarray(rotation).T.reshape(-1)), Gf.Vec3d(0)))
        # The converter is configured make_instanceable=False. Never edit a
        # shared asset layer or leave source-block/box collision beneath it.
        replaced = {"visuals", "collisions"} if source_prim is None else {"collisions"}
        scopes = [p for p in Usd.PrimRange(owner) if p != owner and p.GetName() in replaced]
        for prim in scopes:
            prim.SetActive(False)
        authored = UsdGeom.Xform.Define(stage, str(owner.GetPath()) + "/SelectedAsset")
        if source_prim is None:
            authored.GetPrim().GetReferences().AddReference(str(asset))
        else:
            authored.GetPrim().GetReferences().AddReference(str(asset), source_prim)
        collisions = [p for p in Usd.PrimRange(authored.GetPrim()) if p.HasAPI(UsdPhysics.CollisionAPI)]
        if not collisions:
            raise RuntimeError(f"GEN4 selected asset has no source-derived collision: {asset}")
        for collider in collisions:
            UsdGeom.Imageable(collider).MakeInvisible()
        print(f"[GEN4] Mounted {name}: {asset.name}; source colliders={len(collisions)}", flush=True)
