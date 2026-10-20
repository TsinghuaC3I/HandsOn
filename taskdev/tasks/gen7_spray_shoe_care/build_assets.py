"""Convert the selected CC BY source meshes without modeling visible geometry.

Run from the repository root with a CPU Python containing numpy, scipy, trimesh,
Pillow and python-fcl. Generated OBJ geometry and textures remain beside the
other task assets. This audit is supplementary to the mandatory live PhysX gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "handson/handson/sim/envs/usd/eval_gen06_spray_shoe"
XMLS = REPO / "handson/handson/sim/envs/xmls"
OUTPUT = ASSETS / "meshes"
TABLE_Z = 0.89


def world_mesh(scene, node):
    transform, key = scene.graph[node]
    mesh = scene.geometry[key].copy()
    mesh.apply_transform(transform)
    return mesh


def component_face_groups(mesh, digits=5):
    # Weld a geometry-only copy for connectivity, retaining the original face
    # indices and original textured source for visual extraction.
    geometry = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    geometry.merge_vertices(digits_vertex=digits)
    return sorted(
        trimesh.graph.connected_components(
            geometry.face_adjacency,
            nodes=np.arange(len(geometry.faces)),
            engine="scipy",
        ),
        key=lambda group: int(np.min(group)),
    )


def export_visual(mesh, name):
    """Explicit per-mesh material names prevent source texture overwrites."""
    obj_path = OUTPUT / f"{name}.obj"
    # MJCF binds the material explicitly; preserve UVs in OBJ and export the
    # original base-color image separately. No synthesized texture is used.
    obj = trimesh.exchange.obj.export_obj(
        mesh, include_normals=True, include_texture=False, include_color=False
    )
    if getattr(mesh.visual, "uv", None) is not None:
        uv = np.asarray(mesh.visual.uv)
        lines = ["# Source-derived GEN7 visual mesh"]
        lines.extend("v " + " ".join(f"{v:.10g}" for v in p) for p in mesh.vertices)
        lines.extend("vt " + " ".join(f"{v:.10g}" for v in p) for p in uv)
        lines.extend("vn " + " ".join(f"{v:.10g}" for v in p) for p in mesh.vertex_normals)
        lines.extend("f " + " ".join(f"{i+1}/{i+1}/{i+1}" for i in face) for face in mesh.faces)
        obj = "\n".join(lines) + "\n"
    obj_path.write_text(obj)
    material = getattr(mesh.visual, "material", None)
    rgba = getattr(material, "baseColorFactor", None)
    rgba = np.ones(4) if rgba is None else np.asarray(rgba, dtype=float) / 255.0
    image = getattr(material, "baseColorTexture", None)
    texture = None
    if image is not None:
        texture = f"{name}.png"
        image.save(OUTPUT / texture)
    return {"name": name, "file": obj_path.name, "rgba": rgba.tolist(), "texture": texture}


def clipped_hulls(mesh, cuts, axis=2):
    """Convex pieces cut from source geometry, never a replacement visual."""
    geometry = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    pieces = []
    for lower, upper in zip(cuts[:-1], cuts[1:]):
        sliced = geometry
        for bound, sign in ((lower, 1.0), (upper, -1.0)):
            normal = np.zeros(3); normal[axis] = sign
            origin = np.zeros(3); origin[axis] = bound
            sliced = trimesh.intersections.slice_mesh_plane(sliced, normal, origin, cap=False)
            if not len(sliced.vertices):
                break
        if len(sliced.vertices) >= 4:
            hull = trimesh.convex.convex_hull(sliced.vertices)
            if hull.volume > 1e-12:
                pieces.append(hull)
    return pieces


def clearance_hulls(mesh, swept_trigger, depth=0, source=None):
    """Split source surfaces where a convex hull would seal the trigger slot."""
    hull = trimesh.convex.convex_hull(mesh.vertices)
    source = mesh if source is None else source
    points = np.vstack([hull.vertices, hull.triangles_center])
    closest, distances, faces = trimesh.proximity.closest_point(source, points)
    signed = np.einsum("ij,ij->i", points-closest, source.face_normals[faces])
    protrusion = np.any((signed > 1e-6) & (distances > .0015))
    if not swept_trigger.in_collision_single(hull) and not protrusion:
        return [hull]
    if depth >= 16:
        raise RuntimeError("Source-derived head collision still seals the trigger slot; asset audit failed.")
    axis = int(np.argmax(mesh.extents))
    middle = float(mesh.bounds[:, axis].mean())
    output = []
    for sign in (1.0, -1.0):
        normal = np.zeros(3); normal[axis] = sign
        origin = np.zeros(3); origin[axis] = middle
        cut = trimesh.intersections.slice_mesh_plane(mesh, normal, origin, cap=False)
        if len(cut.vertices) < 4:
            continue
        candidate = trimesh.convex.convex_hull(cut.vertices)
        if candidate.volume > 1e-12:
            output.extend(clearance_hulls(cut, swept_trigger, depth+1, source))
    return output


def normalize_bottle():
    scene = trimesh.load(ASSETS / "sources/spray_bottle.glb", force="scene")
    parts = {node: world_mesh(scene, node) for node in scene.graph.nodes_geometry}
    inside = parts["Cylinder_Spray Inside_0"]
    groups = component_face_groups(inside, digits=3)
    neck_group = next(g for g in groups if len(g) == 256)
    neck = inside.submesh([neck_group], append=True, repair=False)
    fixed = [parts["Cube_Spray Body_0"], parts["Plane_Spray Top_0"], neck]
    trigger = parts["Plane.001_Spray Top_0"]
    source_min_y = float(scene.bounds[0, 1])
    scale = 0.22 / float(np.ptp(scene.bounds[:, 1]))
    # Source: Y up, +Z nozzle. Task: Z up, +Y nozzle, right-handed.
    transform = np.eye(4)
    transform[:3, :3] = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]) * scale
    transform[2, 3] = -source_min_y * scale
    for mesh in fixed + [trigger]:
        mesh.apply_transform(transform)
    pivot = trimesh.transform_points([[0.087, 910.0, 145.0]], transform)[0]
    # Nozzle lies at the frontmost source head surface, outside collision.
    head = fixed[1]
    front = head.vertices[head.vertices[:, 1] > head.bounds[1, 1] - 0.001]
    nozzle = front.mean(axis=0)
    nozzle[0] = 0.0
    nozzle[1] = head.bounds[1, 1] + 0.001
    trigger_collisions = clipped_hulls(
        trigger,
        (trigger.bounds[0, 2] - 1e-7,
         (857.03-source_min_y)*scale,
         (887.91-source_min_y)*scale,
         trigger.bounds[1, 2] + 1e-7),
    )
    swept_trigger = trimesh.collision.CollisionManager()
    for step, angle in enumerate(np.linspace(0, .45, 46)):
        pose = trimesh.transformations.rotation_matrix(angle, [-1, 0, 0], pivot)
        for index, collider in enumerate(trigger_collisions):
            swept_trigger.add_object(f"trigger_{step}_{index}", collider, transform=pose)
    body_collisions = []
    for index, mesh in enumerate(fixed):
        if index == 1:
            # A whole-head convex hull fills the real opening around the
            # lever. Preserve it by splitting the actual head surface.
            geometry = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
            body_collisions.extend(clearance_hulls(geometry, swept_trigger))
        else:
            cuts = np.linspace(mesh.bounds[0, 2] - 1e-7, mesh.bounds[1, 2] + 1e-7, (9 if index == 0 else 2) + 1)
            body_collisions.extend(clipped_hulls(mesh, cuts))
    return fixed, trigger, body_collisions, trigger_collisions, pivot, nozzle, transform


def normalize_shoe():
    scene = trimesh.load(ASSETS / "sources/sneakers.glb", force="scene")
    selected_root = "base.019_50"
    meshes = []
    names = []
    for node in scene.graph.nodes_geometry:
        ancestor = node
        while ancestor in scene.graph.transforms.parents and ancestor != selected_root:
            ancestor = scene.graph.transforms.parents[ancestor]
        if ancestor == selected_root:
            meshes.append(world_mesh(scene, node))
            names.append(node)
    assert len(meshes) == 10, names
    all_vertices = np.vstack([m.vertices for m in meshes])
    minimum, maximum = all_vertices.min(axis=0), all_vertices.max(axis=0)
    scale = 0.27 / float(maximum[2] - minimum[2])
    transform = np.eye(4)
    transform[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) * scale
    # Center XY; support from the actual sole. No supporting box is created.
    center = (minimum + maximum) / 2
    transform[:3, 3] = -transform[:3, :3] @ np.array([center[0], minimum[1], center[2]])
    for mesh in meshes:
        mesh.apply_transform(transform)
    upper = meshes[names.index("Object_81")]
    sole = meshes[names.index("Object_97")]
    inner = meshes[names.index("Object_89")]
    # Top-facing source upper faces above the sidewall are the broad target.
    # The shoe cavity, sole, laces and included foot are never target proxies.
    mask = (upper.face_normals[:, 2] > 0.15) & (upper.triangles_center[:, 2] > 0.025)
    assert int(mask.sum()) > 100
    return meshes, names, (upper, sole, inner), upper.triangles[mask], transform


def numbers(values):
    return " ".join(f"{float(v):.10g}" for v in values)


def build():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Remove only this generator's old collider output when decomposition changes.
    for old in OUTPUT.glob("gen7_*collision_*.obj"):
        old.unlink()
    fixed, trigger, fixed_hulls, trigger_hulls, pivot, nozzle, bottle_transform = normalize_bottle()
    shoes, shoe_names, shoe_collision, target_triangles, shoe_transform = normalize_shoe()
    root = ET.Element("mujoco", model="GEN7_Spray_Shoe_Care_Objects")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    source_rel = "../usd/eval_gen06_spray_shoe/meshes/"

    def visual(body, mesh, name):
        desc = export_visual(mesh, name)
        ET.SubElement(asset, "mesh", name=name, file=source_rel+desc["file"])
        attributes = {"name": name+"_material", "rgba": numbers(desc["rgba"]), "specular": "0.15", "shininess": "0.1"}
        if desc["texture"]:
            ET.SubElement(asset, "texture", name=name+"_texture", type="2d", file=source_rel+desc["texture"])
            attributes["texture"] = name+"_texture"
        ET.SubElement(asset, "material", **attributes)
        ET.SubElement(body, "geom", name=name+"_visual", type="mesh", mesh=name,
                      material=name+"_material", contype="0", conaffinity="0", group="2", density="0")

    def collision(body, mesh, name):
        path = OUTPUT / (name+".obj")
        path.write_text(trimesh.exchange.obj.export_obj(mesh, include_texture=False, include_color=False))
        ET.SubElement(asset, "mesh", name=name, file=source_rel+path.name)
        ET.SubElement(body, "geom", name=name, type="mesh", mesh=name,
                      group="3", rgba="0 0 0 0", density="0", friction="0.9 0.005 0.0001")

    body = ET.SubElement(world, "body", name="link_2", pos=numbers([-.085, -.22, TABLE_Z+.0005]))
    ET.SubElement(body, "freejoint", name="spray_root")
    union = trimesh.util.concatenate(fixed_hulls)
    mass = 0.142
    mass_props = union.mass_properties
    inertia = np.diag(mass_props.inertia) * mass / mass_props.mass
    ET.SubElement(body, "inertial", mass=str(mass), pos=numbers(mass_props.center_mass), diaginertia=numbers(inertia))
    for index, mesh in enumerate(fixed):
        visual(body, mesh, f"gen7_bottle_{index}")
    for index, hull in enumerate(fixed_hulls):
        collision(body, hull, f"gen7_bottle_collision_{index}")

    lever = ET.SubElement(body, "body", name="link_0")
    ET.SubElement(lever, "joint", name="joint_0", type="hinge", pos=numbers(pivot), axis="-1 0 0",
                  range="0 0.45", stiffness="0.2", damping="0.1", frictionloss="0.015")
    lever_union = trimesh.util.concatenate(trigger_hulls)
    props = lever_union.mass_properties
    ET.SubElement(lever, "inertial", mass="0.008", pos=numbers(props.center_mass),
                  diaginertia=numbers(np.diag(props.inertia) * .008 / props.mass))
    visual(lever, trigger, "gen7_trigger")
    for index, hull in enumerate(trigger_hulls):
        collision(lever, hull, f"gen7_trigger_collision_{index}")

    shoe = ET.SubElement(world, "body", name="gen7_shoe", pos=numbers([-.26, .12, TABLE_Z]))
    for name, mesh in zip(shoe_names, shoes):
        visual(shoe, mesh, "gen7_shoe_"+name.lower())
    for index, mesh in enumerate(shoe_collision):
        collision(shoe, mesh, f"gen7_shoe_collision_{index}")
    # The existing C04 cone is an effect, never a collision or success proxy.
    ET.SubElement(asset, "mesh", name="gen7_spray_cone", file="cone/cone.obj", scale="0.025 0.025 0.1")
    ET.SubElement(body, "geom", name="gen7_cone_visual", pos=numbers(nozzle), quat="0.7071067812 -0.7071067812 0 0",
                  type="mesh", mesh="gen7_spray_cone", rgba="0.2 0.4 1 0.45",
                  contype="0", conaffinity="0", group="2", density="0")
    ET.indent(root)
    ET.ElementTree(root).write(XMLS / "eval_gen06_spray_shoe_objects.xml", encoding="utf-8", xml_declaration=True)

    scene = ET.parse(XMLS / "arena_arm_hand_plant_rokae_wuji1.xml")
    scene.getroot().set("model", "GEN7_Spray_Shoe_Care")
    for include in list(scene.getroot().findall("include")):
        if include.get("file") in {"plant.xml", "spray.xml"}:
            scene.getroot().remove(include)
    scene.getroot().insert(1, ET.Element("include", file="eval_gen06_spray_shoe_objects.xml"))
    ET.indent(scene.getroot())
    scene.write(XMLS / "arena_arm_hand_eval_gen06_spray_shoe_rokae_wuji1.xml", encoding="utf-8", xml_declaration=True)
    np.savez_compressed(ASSETS / "shoe_target.npz", triangles=target_triangles)
    manifest = {
        "task_id": "GEN7", "task_name": "eval_gen06_spray_shoe",
        "spray_source_uid": "7441b159f92e469d96401bbe1756f8b9",
        "shoe_source_uid": "4315ca048b954be7abaca9fa1dfee785",
        "bottle_source_transform": bottle_transform.tolist(),
        "shoe_source_transform": shoe_transform.tolist(),
        "bottle_bounds": trimesh.util.concatenate(fixed+[trigger]).bounds.tolist(),
        "shoe_bounds": trimesh.util.concatenate(shoes).bounds.tolist(),
        "bottle_support_min_z": float(min(m.bounds[0,2] for m in fixed_hulls)),
        "shoe_support_min_z": float(min(m.bounds[0,2] for m in shoe_collision)),
        "hinge_pivot": pivot.tolist(), "hinge_axis": [-1,0,0], "hinge_range_rad": [0,.45],
        "nozzle_local": nozzle.tolist(), "nozzle_forward_local": [0,1,0],
        "bottle_mass_kg": mass, "trigger_mass_kg": .008,
        "bottle_collider_count": len(fixed_hulls), "trigger_collider_count": len(trigger_hulls),
        "shoe_collider_approximation": "none (kinematic source triangle mesh)",
        "shoe_target_triangles": len(target_triangles),
        "bottle_visual_faces": sum(len(m.faces) for m in fixed)+len(trigger.faces),
        "shoe_visual_faces": sum(len(m.faces) for m in shoes),
        "removed_source_geometry": ["Static overlapping internal decorative component: 130 faces", "Four unused shoes", "Included foot", "Source floor"],
        "new_visible_geometry": False,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ASSETS / "sources").glob("*.glb")},
    }
    (ASSETS / "asset_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    build()
