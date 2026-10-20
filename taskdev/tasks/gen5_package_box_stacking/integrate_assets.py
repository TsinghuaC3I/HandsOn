"""Reproduce GEN5 assets from the approved, unmodified Sketchfab glTF archive.

Only extract an existing closed component, center, uniformly scale and export.
The invisible collider is the source component with UV/normal seams welded.
"""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
ASSET = ROOT / "handson/handson/sim/envs/xmls/eval_gen07_stack_boxes"
SOURCE = ASSET / "source/cardboard_boxes"
COMPONENT = "Box_big__CardboardBoxes__0"
ARCHIVE_SHA256 = "422418b7bd9396b28a63723d73f4731bf9d87dc7dc6c87c13b7077c1893541a1"


def numbers(values):
    return " ".join(f"{float(x):.12g}" for x in values)


def main():
    archive = ASSET / "source/cardboard_boxes.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_SHA256
    scene = trimesh.load_scene(SOURCE / "scene.gltf", process=False)
    # This component's original local axes are already Z-up. The glTF scene
    # rotates it to Y-up for display; omit that display transform and center it.
    source = scene.geometry[COMPONENT]
    center = source.bounds.mean(axis=0)
    assert len(source.faces) == 44
    objects = {}
    (ASSET / "visuals").mkdir(exist_ok=True)
    (ASSET / "collision").mkdir(exist_ok=True)
    root = ET.Element("mujoco", model="gen5_package_boxes")
    asset = ET.SubElement(root, "asset")
    prefix = "eval_gen07_stack_boxes/"
    ET.SubElement(asset, "texture", name="gen5_cardboard_albedo", type="2d",
                  file=prefix + "source/cardboard_boxes/textures/CardboardBoxes_baseColor.png")
    ET.SubElement(asset, "material", name="gen5_cardboard_material",
                  texture="gen5_cardboard_albedo", rgba="0.743902439 0.743902439 0.743902439 1",
                  specular="0", shininess="0")
    world = ET.SubElement(root, "worldbody")
    placements = {
        "gen5_package_base": (-0.29, 0.10),
        "gen5_package_small_1": (-0.38, -0.20),
        "gen5_package_small_2": (-0.28, -0.22),
        "gen5_package_small_3": (-0.18, -0.16),
    }
    for name, xy in placements.items():
        base = name == "gen5_package_base"
        variant = "base" if base else "small"
        scale = 0.001875 if base else 0.00075
        mass = 0.18 if base else 0.03
        mesh = source.copy()
        mesh.apply_translation(-center)
        mesh.apply_scale(scale)
        collider = mesh.copy()
        collider.visual = trimesh.visual.ColorVisuals(mesh=collider)
        collider.merge_vertices(merge_tex=True, merge_norm=True)
        assert collider.is_watertight and collider.is_convex and collider.is_winding_consistent
        # Preserve original vertex/face data, UVs and normals in the visible OBJ.
        visual_path = f"visuals/{variant}.obj"
        collision_path = f"collision/{variant}.obj"
        visual_obj = trimesh.exchange.obj.export_obj(
            mesh, include_normals=True, include_texture=True,
            include_color=False, write_texture=False, digits=12,
            header="GEN5: extracted from the credited Cardboard Boxes source")
        # MJCF binds the original texture itself; no generated MTL or texture.
        (ASSET / visual_path).write_text("\n".join(
            line for line in visual_obj.splitlines()
            if not line.startswith(("mtllib ", "usemtl "))
        ) + "\n")
        (ASSET / collision_path).write_text(trimesh.exchange.obj.export_obj(
            collider, include_normals=False, include_texture=False,
            include_color=False, digits=12,
            header="Invisible collision mesh: source geometry, welded seams only"))
        inertia = collider.moment_inertia * mass / collider.volume
        com = collider.center_mass
        assert np.all(np.linalg.eigvalsh(inertia) > 0)
        if base or name.endswith("_1"):
            ET.SubElement(asset, "mesh", name=f"gen5_{variant}_visual_mesh", file=prefix + visual_path)
            ET.SubElement(asset, "mesh", name=f"gen5_{variant}_collision_mesh", file=prefix + collision_path)
        body = ET.SubElement(world, "body", name=name,
                             pos=numbers((*xy, 0.89 + mesh.extents[2] / 2)))
        ET.SubElement(body, "freejoint", name=name + "_joint")
        ET.SubElement(body, "inertial", pos=numbers(com), mass=str(mass),
                      fullinertia=numbers([inertia[0, 0], inertia[1, 1], inertia[2, 2],
                                           inertia[0, 1], inertia[0, 2], inertia[1, 2]]))
        ET.SubElement(body, "geom", name=name + "_visual", type="mesh",
                      mesh=f"gen5_{variant}_visual_mesh", material="gen5_cardboard_material",
                      contype="0", conaffinity="0", group="1", mass="0")
        ET.SubElement(body, "geom", name=name + "_collision", type="mesh",
                      mesh=f"gen5_{variant}_collision_mesh", group="3", rgba="0 0 0 0",
                      friction=("1.4 0.04 0.004" if base else "4 0.04 0.004"),
                      condim="4", solref="0.003 1", solimp="0.95 0.99 0.001")
        objects[name] = {
            "source_component": COMPONENT, "source_center": center.tolist(),
            "uniform_scale_m_per_source_unit": scale, "dimensions_m": mesh.extents.tolist(),
            "mass_kg": mass, "center_of_mass_m": com.tolist(), "inertia_kg_m2": inertia.tolist(),
            "inertia_model": "uniform effective density integrated over the closed source mesh",
            "visual_mesh": visual_path, "collision_mesh": collision_path,
            "collision_derivation": "source mesh with UV/normal seams welded; no geometric inflation",
            "visual_triangle_count": len(mesh.faces), "collision_triangle_count": len(collider.faces),
            "support_face_dimensions_m": [47.6 * scale, 47.6 * scale],
            "bevel_m": 0.2 * scale,
            "physx": {"static_friction": 1.4 if base else 4.0,
                      "dynamic_friction": 1.4 if base else 4.0,
                      "restitution": 0, "friction_combine_mode": "max",
                      "contact_offset_m": 0.001, "rest_offset_m": 0,
                      "linear_damping": 0.25, "angular_damping": 1.0,
                      "sleep_threshold": 0.010},
        }
    ET.indent(root)
    ET.ElementTree(root).write(ASSET.parent / "eval_gen07_stack_boxes_objects.xml", encoding="utf-8", xml_declaration=True)
    manifest = {
        "task_id": "GEN5", "source_url": "https://sketchfab.com/3d-models/cardboard-boxes-b0e935d3008b4c2f866e1a8fffe2bc1f",
        "source_archive_sha256": ARCHIVE_SHA256, "license": "CC-BY-4.0",
        "archive_credit": "Isaac Rivera (https://sketchfab.com/silverpirate)",
        "current_page_credit": "JesterArts_ (@JesterArtist_)",
        "credit_note": "Archive and current page use different author names; preserve both without inferring why.",
        "selection": "Use the closed near-square component at two uniform scales. Exclude the open box and the elongated box to retain easy support and grasping.",
        "textures": "Original base-color PNG is referenced directly. Original normal and metallic/roughness maps remain archived; MJCF material uses albedo with the source color factor.",
        "objects": objects,
    }
    (ASSET / "geometry.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({name: {k: obj[k] for k in ("dimensions_m", "mass_kg", "bevel_m")}
                      for name, obj in objects.items()}, indent=2))


if __name__ == "__main__":
    main()
