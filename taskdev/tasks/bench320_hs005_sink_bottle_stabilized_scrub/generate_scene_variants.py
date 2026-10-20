#!/usr/bin/env python3
"""Generate portable HS005 scene layers for all four downloaded kitchens."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from scene_collision import (
    COLLISION_GEOMETRY_BY_SCENE,
    MODERN_APARTMENT_SCENE,
    SCENE_COLLISION_CONTACT_OFFSET,
    SCENE_COLLISION_POLICY,
    SCENE_COLLISION_ROOT_NAME,
    complete_scene_collision_boxes,
)


TASK_DIR = Path(__file__).resolve().parent
BASE_LAYER = TASK_DIR / "scene/hs005_assets.usda"
OUTPUT_DIR = TASK_DIR / "scene/variants"

KEEP_BASE_CHILDREN = {
    "Looks",
}

BASE_VISUAL_CHILDREN = (
    "Looks",
    "Coffee_Maker_Black_0",
    "Coffee_Maker_Chrome_0",
    "Coffee_Maker_Coffee_Maker_Material5_0",
    "Coffee_Maker_Glass_0",
    "Coffee_Maker_White1_0",
    "Dishwasher_001_DSI_46750_Door_001_0",
    "Dishwasher_001_DSI_46750_Screen_001_0",
    "Dishwasher_001_White1_0",
    "Dishwasher_001_Wood049_0",
    "Hobs_001_Base_001_0",
    "Hobs_001_NightRiverBlack_001_0",
    "Hood_Black_0",
    "Hood_Chrome_0",
    "Hood_Glass_0",
    "Hood_Light_0",
    "Hood_White1_0",
    "Oven_001_Chrome_0",
    "Oven_001_FrontDown_001_0",
    "Oven_001_FrontPanel_001_0",
    "Oven_001_FrontTop_001_0",
    "Oven_001_Logo_001_0",
    "Plane_002_1500_0",
    "Plane_002_Black_001_0",
    "Plane_002_Gold_0",
    "Plane_002_Marble_015_0",
    "Plane_002_Marble_014_0",
    "Plane_002_PavingStones128_0",
    "Plane_002_SprayedWallTexture_0",
    "Plane_002_White1_0",
    "Plane_002_Wood_Floor_039_0",
    "Plane_002_Wood0035_0",
    "Plane_007_Wood049_0",
    "Refrigator_001_Aluminum_001_0",
    "Refrigator_001_Black_0",
    "Refrigator_001_BrushedMetalSeamless_001_0",
    "Refrigator_001_FrontBlack_001_0",
    "Sink_Faucet3_001_0",
    "Sink_Hole_001_0",
    "Sink_Out_001_0",
    "Sink_Sink2Metal_001_0",
    "Sink_Sink4Metal_001_0",
    "kitchenblind_Black_0",
)

VARIANTS: tuple[dict[str, object], ...] = (
    {
        "scene_id": "hs005_modern_kitchen_sink",
        "filename": "hs005_modern_kitchen.usda",
        "asset": "../../assets/cc_by_dylanheyes_modern_kitchen/modern_kitchen_visual.usd",
        "asset_id": "modern_kitchen",
        "scale": 0.47,
        # Put this model's own Sink node at the HS005 interaction origin.
        # This kitchen opens toward decreasing source X, so keep source X
        # positive in task X to place the robot on the open side.
        "basis": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
        "axis_mapping": "source(x,y,z)->task(x,-z,y)",
        "translation": (-2.96718, 1.4100, 0.0867),
        # Measured from the transformed source mesh.  The old universal
        # 0.895 m proxy top left the task props visibly suspended in this
        # substantially deeper basin.
        "basin_floor_z": 0.71930,
        "rim_top_z": 0.99375,
        "countertop_z": 0.97580,
        # Keep tall context props behind the sink and outside both raised hand
        # collision envelopes; the manipulated bottle/brush stay in the basin.
        "dish_soap_xy": (0.16, 0.24),
        "sponge_xy": (0.16, -0.18),
        "hide_paths": (
            "Modern_Kitchen_fbx/RootNode/Structure",
            "Modern_Kitchen_fbx/RootNode/Ceiling",
        ),
    },
    {
        "scene_id": "hs005_realistic_kitchen_scene_sink",
        "filename": "hs005_realistic_kitchen_scene.usda",
        "asset": "../../assets/cc_by_katydid_realistic_kitchen_scene/realistic_kitchen_scene_visual.usd",
        "asset_id": "realistic_kitchen_scene",
        "scale": 0.01,
        # The room-facing side of the downloaded double sink is toward lower
        # source X; source Z runs laterally along the counter.
        "basis": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
        "axis_mapping": "source(x,y,z)->task(x,-z,y)",
        # The actual double-basin cabinet is Material35, centered near source
        # X=54 cm, Z=-150 cm (not the nearby food boards/toaster meshes).
        "translation": (-0.69, -1.50, 0.0),
        "basin_floor_z": 0.64815,
        "rim_top_z": 0.80429,
        "countertop_z": 0.80429,
        # This double-sink camera occludes the rear strip. Keep the props on
        # the visible lateral countertop, but move them 20 cm behind the old
        # x=-0.25 hand-conflicting locations.
        "dish_soap_xy": (-0.05, 0.34),
        "sponge_xy": (-0.05, -0.34),
        "hide_paths": (),
    },
    {
        "scene_id": "hs005_appartement_kitchen_sink",
        "filename": "hs005_appartement.usda",
        "asset": "../../assets/cc_by_srmonteiro_appartement/appartement_visual.usd",
        "asset_id": "appartement",
        "scale": 1.0,
        # Turn the back-wall sink toward the robot: source -Z becomes task +X.
        "basis": ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
        "axis_mapping": "source(x,y,z)->task(-z,-x,y)",
        # Sala_Cozinha carries a (-3.2822, 0, -1.2900) source transform;
        # the visible sink center is therefore near world (-1.31, 0.24).
        "translation": (0.09, -1.31, 0.0),
        "basin_floor_z": 0.75207,
        "rim_top_z": 0.90000,
        "countertop_z": 0.90000,
        "dish_soap_xy": (0.16, 0.24),
        "sponge_xy": (0.16, -0.18),
        "hide_paths": (),
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hide_overs(paths: tuple[str, ...], indent: str) -> list[str]:
    tree: dict[str, dict[str, object]] = {}
    for path in paths:
        branch = tree
        for part in path.split("/"):
            branch = branch.setdefault(part, {})  # type: ignore[assignment]

    lines: list[str] = []

    def emit(branch: dict[str, object], prefix: str) -> None:
        for name, children_value in branch.items():
            children = children_value
            lines.extend([f'{prefix}over "{name}"\n', f"{prefix}{{\n"])
            if children:
                emit(children, prefix + "    ")  # type: ignore[arg-type]
            else:
                lines.append(prefix + '    token visibility = "invisible"\n')
            lines.append(prefix + "}\n")

    emit(tree, indent)
    return lines


def fmt_vec(values: tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{float(value):.9g}" for value in values) + ")"


def collision_proxy_lines(
    proxies: tuple[tuple[str, tuple[float, ...], tuple[float, ...]], ...],
) -> list[str]:
    lines = [
        f'    def Xform "{SCENE_COLLISION_ROOT_NAME}" (\n',
        "        customData = {\n",
        f'            string physics_policy = "{SCENE_COLLISION_POLICY}"\n',
        "        }\n",
        "    )\n",
        "    {\n",
    ]
    for name, center, size in proxies:
        lines.extend(
            [
                f'        def Cube "{name}" (\n',
                '            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]\n',
                "        )\n",
                "        {\n",
                "            bool physics:collisionEnabled = true\n",
                "            float physxCollision:contactOffset = "
                f"{SCENE_COLLISION_CONTACT_OFFSET:.9g}\n",
                "            float physxCollision:restOffset = 0.0\n",
                "            double size = 1.0\n",
                '            token visibility = "invisible"\n',
                f"            double3 xformOp:translate = {fmt_vec(center)}\n",
                f"            float3 xformOp:scale = {fmt_vec(size)}\n",
                '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]\n',
                "        }\n",
            ]
        )
    lines.append("    }\n")
    return lines


def variant_layer(spec: dict[str, object]) -> str:
    scale = float(spec["scale"])
    tx, ty, tz = (float(value) for value in spec["translation"])
    basis = tuple(
        tuple(scale * float(value) for value in row)
        for row in spec["basis"]
    )
    basin_floor_z = float(spec["basin_floor_z"])
    countertop_z = float(spec["countertop_z"])
    dish_soap_xy = tuple(float(value) for value in spec["dish_soap_xy"])
    sponge_xy = tuple(float(value) for value in spec["sponge_xy"])
    bottle_origin = (-0.1, 0.11, basin_floor_z)
    brush_origin = (-0.191, -0.125, basin_floor_z + 0.01402124)
    # The soap collider reaches 21.25 mm below its root.  The sponge collider
    # starts at its root.  In both cases retain the task's 3 mm support gap.
    dish_soap_origin = (*dish_soap_xy, countertop_z - 0.01825)
    sponge_origin = (*sponge_xy, countertop_z + 0.003)
    lines = [
        "#usda 1.0\n",
        "(\n",
        '    defaultPrim = "HS005Assets"\n',
        "    metersPerUnit = 1\n",
        '    upAxis = "Z"\n',
        ")\n\n",
        'def Xform "HS005Assets" (\n',
        '    prepend references = @../hs005_assets.usda@\n',
        "    customData = {\n",
        f'        string simulation_scene = "{spec["scene_id"]}"\n',
        '        string variant_policy = "source_kitchen_sink_visual"\n',
        f'        string physics_policy = "{SCENE_COLLISION_POLICY}"\n',
        "    }\n",
        ")\n",
        "{\n",
        '    over "KitchenBackground"\n',
        "    {\n",
        '        over "Visual"\n',
        "        {\n",
    ]
    for child in BASE_VISUAL_CHILDREN:
        if child in KEEP_BASE_CHILDREN:
            continue
        lines.extend(
            [
                f'            over "{child}"\n',
                "            {\n",
                '                token visibility = "invisible"\n',
                "            }\n",
            ]
        )
    lines.extend(
        [
            "        }\n",
        "    }\n\n",
        '    over "Sink"\n',
        "    {\n",
        '        over "Collider"\n',
        "        {\n",
        "            bool physics:collisionEnabled = false\n",
        "        }\n",
        "    }\n\n",
        '    over "Bottle"\n',
        "    {\n",
        f"        double3 xformOp:translate = {fmt_vec(bottle_origin)}\n",
        "    }\n\n",
        '    over "Brush"\n',
        "    {\n",
        f"        double3 xformOp:translate = {fmt_vec(brush_origin)}\n",
        "    }\n\n",
        '    over "DishSoap"\n',
        "    {\n",
        f"        double3 xformOp:translate = {fmt_vec(dish_soap_origin)}\n",
        "    }\n\n",
        '    over "Sponge"\n',
        "    {\n",
        f"        double3 xformOp:translate = {fmt_vec(sponge_origin)}\n",
        "    }\n\n",
            '    def Xform "SceneVariant"\n',
            "    {\n",
            "        matrix4d xformOp:transform = (\n",
            f"            ({basis[0][0]}, {basis[0][1]}, {basis[0][2]}, 0),\n",
            f"            ({basis[1][0]}, {basis[1][1]}, {basis[1][2]}, 0),\n",
            f"            ({basis[2][0]}, {basis[2][1]}, {basis[2][2]}, 0),\n",
            f"            ({tx}, {ty}, {tz}, 1)\n",
            "        )\n",
            '        uniform token[] xformOpOrder = ["xformOp:transform"]\n\n',
            '        def Xform "Asset" (\n',
            f"            prepend references = @{spec['asset']}@\n",
            "        )\n",
            "        {\n",
        ]
    )
    lines.extend(
        hide_overs(tuple(str(path) for path in spec["hide_paths"]), "            ")
    )
    lines.extend(["        }\n", "    }\n\n"])
    lines.extend(
        collision_proxy_lines(
            complete_scene_collision_boxes(str(spec["scene_id"]))
        )
    )
    lines.append("}\n")
    return "".join(lines)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    base_manifest = json.loads((TASK_DIR / "scene_manifest.json").read_text())
    manifest: dict[str, object] = {
        "schema": "hs005_simulation_scene_variants_v3",
        "base": {
            "scene_id": MODERN_APARTMENT_SCENE,
            "path": "scene/hs005_assets.usda",
            "sha256": sha256_file(BASE_LAYER),
            "bytes": BASE_LAYER.stat().st_size,
            "basin_floor_z": COLLISION_GEOMETRY_BY_SCENE[
                MODERN_APARTMENT_SCENE
            ]["floor_top_z"],
            "rim_top_z": COLLISION_GEOMETRY_BY_SCENE[
                MODERN_APARTMENT_SCENE
            ]["rim_top_z"],
            "countertop_z": COLLISION_GEOMETRY_BY_SCENE[
                MODERN_APARTMENT_SCENE
            ]["countertop_z"],
            "scene_collision_root": f"/HS005Assets/{SCENE_COLLISION_ROOT_NAME}",
            "sink_collision_proxy": [
                {"name": name, "center": center, "size": size}
                for name, center, size in complete_scene_collision_boxes(
                    MODERN_APARTMENT_SCENE
                )
            ],
        },
        "variants": {},
        "source_assets": {},
        "physics_policy": SCENE_COLLISION_POLICY,
    }
    for asset_name, asset in base_manifest["assets"].items():
        manifest["source_assets"][asset_name] = {
            "path": asset["path"],
            "sha256": asset.get("visual_sha256") or asset.get("source_sha256"),
        }
    for spec in VARIANTS:
        output = OUTPUT_DIR / str(spec["filename"])
        atomic_write(output, variant_layer(spec))
        asset_path = (OUTPUT_DIR / str(spec["asset"])).resolve()
        conversion_path = asset_path.parent / "conversion.json"
        conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
        manifest["variants"][spec["scene_id"]] = {
            "path": str(output.relative_to(TASK_DIR)),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "background_asset": str(asset_path.relative_to(TASK_DIR)),
            "background_sha256": sha256_file(asset_path),
            "background_archive_sha256": conversion["source"]["archive_sha256"],
            "background_license": conversion["source"]["license"],
            "scale": spec["scale"],
            "translation": spec["translation"],
            "axis_mapping": spec["axis_mapping"],
            "render_only": False,
            "legacy_sink_collider_disabled": True,
            "basin_floor_z": spec["basin_floor_z"],
            "rim_top_z": spec["rim_top_z"],
            "collision_wall_top_z": spec["rim_top_z"],
            "countertop_z": spec["countertop_z"],
            "scene_collision_root": f"/HS005Assets/{SCENE_COLLISION_ROOT_NAME}",
            "dish_soap_xy": spec["dish_soap_xy"],
            "sponge_xy": spec["sponge_xy"],
            "object_reset_z": {
                "bottle": float(spec["basin_floor_z"]),
                "brush": float(spec["basin_floor_z"]) + 0.01402124,
                "dish_soap": float(spec["countertop_z"]) - 0.01825,
                "sponge": float(spec["countertop_z"]) + 0.003,
            },
            "sink_collision_proxy": [
                {"name": name, "center": center, "size": size}
                for name, center, size in complete_scene_collision_boxes(
                    str(spec["scene_id"])
                )
            ],
        }
    atomic_write(
        TASK_DIR / "scene/scene_variants.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
