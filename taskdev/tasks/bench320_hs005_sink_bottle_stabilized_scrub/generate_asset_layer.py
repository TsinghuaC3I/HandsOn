#!/usr/bin/env python3
"""Generate the portable HS005 USD asset layer without starting Kit.

The pruned Modern Apartment kitchen and four user-selected CC BY visuals remain
relative USD references.  The downloaded sink remains the visible geometry,
while the scene owns an analytic floor, four full-height basin walls, and an
open countertop ring.  The source sink's full triangle mesh is deliberately
not used for collision: its front-right rim overlaps the ROKAE/Wuji home pose
and can eject the arm on the first PhysX frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from scene_contract import (
    BOTTLE_VISUAL_SCALE,
    DISH_SOAP_VISUAL_SCALE,
    SPONGE_VISUAL_SCALE,
    placement_contract,
)
from scene_collision import (
    COLLISION_GEOMETRY_BY_SCENE,
    MODERN_APARTMENT_SCENE,
    SCENE_COLLISION_CONTACT_OFFSET,
    SCENE_COLLISION_POLICY,
    SCENE_COLLISION_ROOT_NAME,
    collision_repair_boxes,
)


def fmt_number(value: float) -> str:
    result = f"{float(value):.9g}"
    return result if "." in result or "e" in result else result + ".0"


def fmt_vec(values: Iterable[float]) -> str:
    return "(" + ", ".join(fmt_number(value) for value in values) + ")"


def cube_block(
    name: str,
    center: Iterable[float],
    size: Iterable[float],
    *,
    contact_offset: float = 0.004,
) -> str:
    return f'''        def Cube "{name}" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]
        )
        {{
            bool physics:collisionEnabled = true
            float physxCollision:contactOffset = {fmt_number(contact_offset)}
            float physxCollision:restOffset = 0.0
            double size = 1.0
            token visibility = "invisible"
            double3 xformOp:translate = {fmt_vec(center)}
            float3 xformOp:scale = {fmt_vec(size)}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
'''


def cylinder_block(
    name: str,
    center: Iterable[float],
    *,
    axis: str,
    height: float,
    radius: float,
    contact_offset: float = 0.003,
) -> str:
    if axis not in {"X", "Y", "Z"}:
        raise ValueError(f"unsupported cylinder axis: {axis!r}")
    return f'''        def Cylinder "{name}" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]
        )
        {{
            uniform token axis = "{axis}"
            double height = {fmt_number(height)}
            double radius = {fmt_number(radius)}
            bool physics:collisionEnabled = true
            float physxCollision:contactOffset = {fmt_number(contact_offset)}
            float physxCollision:restOffset = 0.0
            token visibility = "invisible"
            double3 xformOp:translate = {fmt_vec(center)}
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
'''


def build_layer(task_dir: Path, output_path: Path, table_top_z: float) -> dict[str, object]:
    contract = placement_contract(table_top_z)
    sink_origin = tuple(contract["sink_origin"])
    kitchen_translation = tuple(contract["kitchen_translation"])
    kitchen_orient = tuple(contract["kitchen_orient_quat_real_xyz"])
    bottle_origin = tuple(contract["bottle_origin"])
    brush_origin = tuple(contract["brush_origin"])
    dish_soap_origin = tuple(contract["dish_soap_origin"])
    sponge_origin = tuple(contract["sponge_origin"])

    lines = [
        "#usda 1.0\n",
        "(\n",
        '    defaultPrim = "HS005Assets"\n',
        "    metersPerUnit = 1\n",
        '    upAxis = "Z"\n',
        ")\n\n",
        'def Xform "HS005Assets" (\n',
        "    customData = {\n",
        f'        string physics_policy = "{SCENE_COLLISION_POLICY}"\n',
        f'        string simulation_scene = "{MODERN_APARTMENT_SCENE}"\n',
        '        string scope = "core_scene_only"\n',
        "        bool task_complete = false\n",
        '        string task_id = "B320-HS005"\n',
        "    }\n",
        ")\n{\n",
        '    def Scope "Looks"\n    {\n',
        '        def Material "LightCountertop"\n        {\n',
        '            token outputs:surface.connect = </HS005Assets/Looks/LightCountertop/PBR.outputs:surface>\n\n',
        '            def Shader "PBR"\n            {\n',
        '                uniform token info:id = "UsdPreviewSurface"\n',
        '                color3f inputs:diffuseColor = (0.68, 0.62, 0.52)\n',
        '                float inputs:metallic = 0.02\n',
        '                float inputs:roughness = 0.48\n',
        '                token outputs:surface\n',
        "            }\n",
        "        }\n\n",
        '        def Material "StainlessSink"\n        {\n',
        '            token outputs:surface.connect = </HS005Assets/Looks/StainlessSink/PBR.outputs:surface>\n\n',
        '            def Shader "PBR"\n            {\n',
        '                uniform token info:id = "UsdPreviewSurface"\n',
        '                color3f inputs:diffuseColor = (0.48, 0.52, 0.56)\n',
        '                float inputs:metallic = 0.82\n',
        '                float inputs:roughness = 0.28\n',
        '                token outputs:surface\n',
        "            }\n",
        "        }\n",
        "    }\n\n",
        '    def Xform "KitchenBackground"\n    {\n',
        f"        double3 xformOp:translate = {fmt_vec(kitchen_translation)}\n",
        f"        quatf xformOp:orient = {fmt_vec(kitchen_orient)}\n",
        '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]\n\n',
        '        def Xform "Visual" (\n',
        "            prepend references = @../assets/modern_apartment_kitchen/modern_apartment_kitchen.usd@\n",
        "        )\n        {\n",
        '            over "Plane_007_Wood049_0"\n            {\n',
        '                over "Plane_007_Wood049_0"\n                {\n',
        '                    rel material:binding = </HS005Assets/Looks/LightCountertop>\n',
        "                }\n",
        "            }\n\n",
        '            over "Sink_Sink2Metal_001_0"\n            {\n',
        '                over "Sink_Sink2Metal_001_0"\n                {\n',
        '                    rel material:binding = </HS005Assets/Looks/StainlessSink>\n',
        "                }\n",
        "            }\n\n",
        '            over "Sink_Sink4Metal_001_0"\n            {\n',
        '                over "Sink_Sink4Metal_001_0"\n                {\n',
        '                    rel material:binding = </HS005Assets/Looks/StainlessSink>\n',
        "                }\n",
        "            }\n",
        "        }\n",
        "    }\n\n",
        '    def Xform "Sink"\n    {\n',
        f"        double3 xformOp:translate = {fmt_vec(sink_origin)}\n",
        '        uniform token[] xformOpOrder = ["xformOp:translate"]\n\n',
    ]
    lines.extend(
        [
            '        def Cube "Collider" (\n',
            '            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]\n',
            "        )\n        {\n",
            "            bool physics:collisionEnabled = true\n",
            "            float physxCollision:contactOffset = 0.003\n",
            "            float physxCollision:restOffset = 0.0\n",
            "            double size = 1.0\n",
            '            token visibility = "invisible"\n',
            "            double3 xformOp:translate = (0.0, 0.0, -0.01)\n",
            "            float3 xformOp:scale = (0.42, 0.72, 0.02)\n",
            '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]\n',
            "        }\n",
            "    }\n\n",
        ]
    )
    sink_collision_paths = ["/HS005Assets/Sink/Collider"]

    lines.extend(
        [
            f'    def Xform "{SCENE_COLLISION_ROOT_NAME}" (\n',
            "        customData = {\n",
            f'            string physics_policy = "{SCENE_COLLISION_POLICY}"\n',
            "        }\n",
            "    )\n    {\n",
        ]
    )
    for name, center, size in collision_repair_boxes(MODERN_APARTMENT_SCENE):
        lines.append(
            cube_block(
                name,
                center,
                size,
                contact_offset=SCENE_COLLISION_CONTACT_OFFSET,
            )
        )
        sink_collision_paths.append(
            f"/HS005Assets/{SCENE_COLLISION_ROOT_NAME}/{name}"
        )
    lines.append("    }\n\n")

    lines.extend(
        [
            '    def Xform "Bottle" (\n',
            '        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]\n',
            "    )\n    {\n",
            "        bool physics:rigidBodyEnabled = true\n",
            "        bool physics:kinematicEnabled = false\n",
            "        float physics:mass = 0.60\n",
            "        int physxRigidBody:solverPositionIterationCount = 16\n",
            "        int physxRigidBody:solverVelocityIterationCount = 4\n",
            "        float physxRigidBody:linearDamping = 0.40\n",
            "        float physxRigidBody:angularDamping = 2.00\n",
            f"        double3 xformOp:translate = {fmt_vec(bottle_origin)}\n",
            '        uniform token[] xformOpOrder = ["xformOp:translate"]\n\n',
            '        def Xform "Visual"\n        {\n',
            "            float xformOp:rotateX = 90.0\n",
            f"            float3 xformOp:scale = ({BOTTLE_VISUAL_SCALE}, {BOTTLE_VISUAL_SCALE}, {BOTTLE_VISUAL_SCALE})\n",
            '            uniform token[] xformOpOrder = ["xformOp:rotateX", "xformOp:scale"]\n\n',
            '            def Xform "Asset" (\n',
            "                prepend references = @../assets/cc_by_lukemodels75_water_bottle/water_bottle_visual.usd@\n",
            "            )\n            {\n",
            "            }\n",
            "        }\n\n",
            '        def Xform "Colliders"\n        {\n',
        ]
    )
    bottle_colliders = (
        ("body", (0.0, 0.0, 0.1025), "Z", 0.2050, 0.0347),
        ("shoulder", (0.0, 0.0, 0.22065), "Z", 0.0313, 0.0270),
        ("cap", (0.0, 0.0, 0.2431), "Z", 0.0136, 0.0128),
    )
    bottle_collision_paths = []
    for name, center, axis, height, radius in bottle_colliders:
        lines.append(
            cylinder_block(
                name,
                center,
                axis=axis,
                height=height,
                radius=radius,
            )
        )
        bottle_collision_paths.append(f"/HS005Assets/Bottle/Colliders/{name}")
    lines.extend(["        }\n", "    }\n\n"])

    lines.extend(
        [
            '    def Xform "Brush" (\n',
            '        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]\n',
            "    )\n    {\n",
            "        bool physics:rigidBodyEnabled = true\n",
            "        bool physics:kinematicEnabled = false\n",
            "        float physics:mass = 0.11\n",
            "        point3f physics:centerOfMass = (0.05, 0.0, 0.0)\n",
            "        float3 physics:diagonalInertia = (0.000012, 0.00074, 0.00074)\n",
            "        quatf physics:principalAxes = (1.0, 0.0, 0.0, 0.0)\n",
            "        int physxRigidBody:solverPositionIterationCount = 16\n",
            "        int physxRigidBody:solverVelocityIterationCount = 4\n",
            "        float physxRigidBody:linearDamping = 0.08\n",
            "        float physxRigidBody:angularDamping = 0.12\n",
            f"        double3 xformOp:translate = {fmt_vec(brush_origin)}\n",
            '        uniform token[] xformOpOrder = ["xformOp:translate"]\n\n',
            '        def Xform "Visual"\n        {\n',
            '            def Xform "Asset" (\n',
            "                prepend references = @../assets/cc_by_amhyde_cleaning_brush/cleaning_brush_visual.usd@\n",
            "            )\n            {\n",
            "            }\n",
            "        }\n\n",
            '        def Xform "Colliders"\n        {\n',
        ]
    )
    brush_colliders = (
        ("bristles", (-0.0555, 0.0, 0.0), "X", 0.0765, 0.0135),
        ("ferrule", (-0.0025, 0.0, 0.0), "X", 0.0310, 0.0140),
        ("handle", (0.1013, 0.0, 0.0), "X", 0.1788, 0.0132),
    )
    brush_collision_paths = []
    for name, center, axis, height, radius in brush_colliders:
        lines.append(
            cylinder_block(
                name,
                center,
                axis=axis,
                height=height,
                radius=radius,
            )
        )
        brush_collision_paths.append(f"/HS005Assets/Brush/Colliders/{name}")
    # A horizontal primitive cylinder has no resting roll equilibrium. This
    # 31 x 4 x 1 mm flat, invisible ferrule chord is coplanar with the 14 mm
    # cylinder bottom and stays within 0.15 mm of the source round profile. It
    # provides a real support patch instead of freezing velocity every frame.
    lines.append(
        cube_block(
            "stability_foot",
            (-0.0025, 0.0, -0.0135),
            (0.0310, 0.0040, 0.0010),
        )
    )
    brush_collision_paths.append(
        "/HS005Assets/Brush/Colliders/stability_foot"
    )
    lines.extend(["        }\n", "    }\n\n"])

    lines.extend(
        [
            '    def Xform "DishSoap"\n    {\n',
            f"        double3 xformOp:translate = {fmt_vec(dish_soap_origin)}\n",
            '        uniform token[] xformOpOrder = ["xformOp:translate"]\n\n',
            '        def Xform "Visual"\n        {\n',
            "            float xformOp:rotateX = 90.0\n",
            f"            float3 xformOp:scale = ({DISH_SOAP_VISUAL_SCALE}, {DISH_SOAP_VISUAL_SCALE}, {DISH_SOAP_VISUAL_SCALE})\n",
            '            uniform token[] xformOpOrder = ["xformOp:rotateX", "xformOp:scale"]\n\n',
            '            def Xform "Asset" (\n',
            "                prepend references = @../assets/cc_by_mike_nov_dishwashing_dispenser/dishwashing_dispenser_visual.usd@\n",
            "            )\n            {\n",
            "            }\n",
            "        }\n\n",
            '        def Xform "Colliders"\n        {\n',
            cube_block("body", (0.0, 0.0, 0.1010), (0.0850, 0.0750, 0.1595)),
            cylinder_block("neck", (0.0, 0.0, 0.2090), axis="Z", height=0.0630, radius=0.0208),
            cube_block("pump", (0.0, -0.0343, 0.2434), (0.0356, 0.1002, 0.0200)),
            "        }\n",
            "    }\n\n",
            '    def Xform "Sponge"\n    {\n',
            f"        double3 xformOp:translate = {fmt_vec(sponge_origin)}\n",
            '        uniform token[] xformOpOrder = ["xformOp:translate"]\n\n',
            '        def Xform "Visual"\n        {\n',
            "            float xformOp:rotateX = 90.0\n",
            f"            float3 xformOp:scale = ({SPONGE_VISUAL_SCALE}, {SPONGE_VISUAL_SCALE}, {SPONGE_VISUAL_SCALE})\n",
            '            uniform token[] xformOpOrder = ["xformOp:rotateX", "xformOp:scale"]\n\n',
            '            def Xform "Asset" (\n',
            "                prepend references = @../assets/cc_by_requital_dish_sponge/dish_sponge_visual.usd@\n",
            "            )\n            {\n",
            "            }\n",
            "        }\n\n",
            '        def Xform "Colliders"\n        {\n',
            cube_block("body", (0.0, 0.0, 0.008835), (0.044196, 0.099814, 0.017670)),
            "        }\n",
            "    }\n",
            "}\n",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")
    metadata = {
        "schema": "hs005_asset_layer_v5",
        "scene_path": "scene/hs005_assets.usda",
        "table_top_z": table_top_z,
        "poses": {
            "kitchen_translation": kitchen_translation,
            "sink": sink_origin,
            "bottle": bottle_origin,
            "brush": brush_origin,
            "dish_soap": dish_soap_origin,
            "sponge": sponge_origin,
        },
        "source_scale_xyz": {
            "kitchen": [1.0, 1.0, 1.0],
            "sink": [1.0, 1.0, 1.0],
            "bottle": [BOTTLE_VISUAL_SCALE] * 3,
            "brush": [1.0, 1.0, 1.0],
            "dish_soap": [DISH_SOAP_VISUAL_SCALE] * 3,
            "sponge": [SPONGE_VISUAL_SCALE] * 3,
        },
        "visual_axis_corrections_deg": {
            "kitchen_rotate_xyz": [90.0, 0.0, -90.0],
            "bottle_rotate_x": 90.0,
            "brush_rotate_x": 0.0,
            "dish_soap_rotate_x": 90.0,
            "sponge_rotate_x": 90.0,
        },
        "visual_paths": {
            "kitchen": "../assets/modern_apartment_kitchen/modern_apartment_kitchen.usd",
            "bottle": "../assets/cc_by_lukemodels75_water_bottle/water_bottle_visual.usd",
            "brush": "../assets/cc_by_amhyde_cleaning_brush/cleaning_brush_visual.usd",
            "dish_soap": "../assets/cc_by_mike_nov_dishwashing_dispenser/dishwashing_dispenser_visual.usd",
            "sponge": "../assets/cc_by_requital_dish_sponge/dish_sponge_visual.usd",
        },
        "sink_collision_source": (
            "../assets/modern_apartment_kitchen/sink_collision_source.ply"
        ),
        "sink_collision_policy": SCENE_COLLISION_POLICY,
        "sink_floor_top_z": sink_origin[2],
        "sink_basin_bounds": list(
            COLLISION_GEOMETRY_BY_SCENE[MODERN_APARTMENT_SCENE][
                "basin_bounds"
            ]
        ),
        "sink_rim_top_z": COLLISION_GEOMETRY_BY_SCENE[
            MODERN_APARTMENT_SCENE
        ]["rim_top_z"],
        "countertop_z": COLLISION_GEOMETRY_BY_SCENE[
            MODERN_APARTMENT_SCENE
        ]["countertop_z"],
        "sink_collision_reason": (
            "analytic scene collision seals the basin and countertop while "
            "preserving the measured opening and reset-hand clearance"
        ),
        "collision_paths": {
            "sink": sink_collision_paths,
            "bottle": bottle_collision_paths,
            "brush": brush_collision_paths,
            "dish_soap": [
                "/HS005Assets/DishSoap/Colliders/body",
                "/HS005Assets/DishSoap/Colliders/neck",
                "/HS005Assets/DishSoap/Colliders/pump",
            ],
            "sponge": ["/HS005Assets/Sponge/Colliders/body"],
        },
        "context_prop_policy": {
            "dish_soap": "static physical context; not a success entity",
            "sponge": "static physical context; not a success entity",
        },
        "joint_policy": {
            "sink": "static collision owner; no joint required",
            "bottle": "free rigid body; no articulation joint",
            "brush": "free rigid body; no articulation joint",
            "authored_joint_count": 0,
        },
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--table-top-z", type=float, default=0.89)
    args = parser.parse_args()
    task_dir = args.task_dir.resolve()
    output = (args.output or task_dir / "scene/hs005_assets.usda").resolve()
    metadata = build_layer(task_dir, output, args.table_top_z)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
