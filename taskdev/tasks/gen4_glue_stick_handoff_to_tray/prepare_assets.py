#!/usr/bin/env python3
"""Convert the selected glue-stick glTF/GLB on CPU, preserving its appearance.

This imports an existing licensed model; it does not synthesize a visible asset.
Inspect the source first, then explicitly choose its longitudinal up axis.
Run with --help for the asset-file checkpoint. No GPU or Isaac app is started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

ROOT = Path(__file__).resolve().parents[3]
NAME = "eval_gen04_pass_glue_stick"
OUT = ROOT / "handson/handson/sim/envs/usd" / NAME
TRAY_DIR = OUT.parent / "loaded_tray_gate_assets"
SOURCE_URL = "https://sketchfab.com/3d-models/customisable-glue-stick-d861aa03580340b19e491eea3727a0c8"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def usd_mesh(stage, path, mesh):
    obj = UsdGeom.Mesh.Define(stage, path)
    obj.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32)))
    obj.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    obj.CreateFaceVertexIndicesAttr(np.asarray(mesh.faces).reshape(-1).tolist())
    obj.CreateSubdivisionSchemeAttr("none")
    obj.CreateDoubleSidedAttr(True)
    obj.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertex_normals, dtype=np.float32)))
    obj.SetNormalsInterpolation("vertex")
    return obj


def bind_visual(stage, mesh_usd, mesh, number):
    """Preserve the selected glTF PBR maps, including tangent-space normals."""
    material = UsdShade.Material.Define(stage, f"/GlueStick/Materials/Visual{number}")
    shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    source = mesh.visual.material
    factor = source.baseColorFactor
    color = np.asarray(factor if factor is not None else [1, 1, 1, 1], dtype=float)
    if color.max() > 1:
        color /= 255.0
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color[:3]))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    roughness = source.roughnessFactor
    metallic = source.metallicFactor
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness) if roughness is not None else 1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic) if metallic is not None else 1.0)
    uv = np.asarray(mesh.visual.uv, dtype=np.float32).copy()
    # Trimesh already converts glTF's upper-left UV origin to lower-left.
    # UsdUVTexture uses the same lower-left convention; do not flip twice.
    primvar = UsdGeom.PrimvarsAPI(mesh_usd).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex")
    primvar.Set(Vt.Vec2fArray.FromNumpy(uv))
    reader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    def texture_node(image, name, color_space):
        texture_path = OUT / "textures" / f"{name}_{number}.png"
        texture_path.parent.mkdir(exist_ok=True)
        image.save(texture_path)
        tex = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/" + name)
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("textures/" + texture_path.name))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        for channel in ("r", "g", "b", "a"):
            tex.CreateOutput(channel, Sdf.ValueTypeNames.Float)
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        return tex

    if source.baseColorTexture is not None:
        tex = texture_node(source.baseColorTexture, "BaseColor", "sRGB")
        tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*color))
        shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
    if source.metallicRoughnessTexture is not None:
        tex = texture_node(source.metallicRoughnessTexture, "MetallicRoughness", "raw")
        rough = float(roughness) if roughness is not None else 1.0
        metal = float(metallic) if metallic is not None else 1.0
        tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1, rough, metal, 1))
        shader.GetInput("roughness").ConnectToSource(tex.ConnectableAPI(), "g")
        shader.GetInput("metallic").ConnectToSource(tex.ConnectableAPI(), "b")
    if source.occlusionTexture is not None:
        tex = texture_node(source.occlusionTexture, "Occlusion", "raw")
        shader.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(), "r")
    if source.normalTexture is not None:
        tex = texture_node(source.normalTexture, "Normal", "raw")
        tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
        tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh_usd.GetPrim()).Bind(material)


def split_source_convex(mesh, depth=0):
    """Partition an existing convex hull exactly into small convex pieces.

    Keep each hull within PhysX GPU vertex/polygon limits, without changing
    the external source boundary.
    """
    hull = mesh.convex_hull
    planes = np.c_[hull.face_normals, -np.einsum("ij,ij->i", hull.face_normals, hull.triangles[:, 0])]
    _, groups = np.unique(np.round(planes, 7), axis=0, return_inverse=True)
    polygons = int(groups.max()) + 1
    largest_face = max(len(np.unique(hull.faces[groups == group])) for group in range(polygons))
    if len(hull.vertices) <= 64 and polygons <= 64 and largest_face <= 32:
        return [hull]
    if depth >= 12:
        raise ValueError("Source convex partition did not reach the cooking limit")
    axis = int(np.argsort(hull.extents)[::-1][depth % 3])
    values = hull.vertices[:, axis]
    plane = float((values.min() + values.max()) / 2)
    a, b = hull.vertices[hull.edges_unique[:, 0]], hull.vertices[hull.edges_unique[:, 1]]
    crossing = (a[:, axis] - plane) * (b[:, axis] - plane) < 0
    a, b = a[crossing], b[crossing]
    intersections = a + ((plane - a[:, axis]) / (b[:, axis] - a[:, axis]))[:, None] * (b - a)
    pieces = []
    for positive in (False, True):
        inside = values >= plane if positive else values <= plane
        points = np.concatenate((hull.vertices[inside], intersections))
        pieces.extend(split_source_convex(trimesh.convex.convex_hull(points), depth + 1))
    return pieces


def prepare_clutter():
    """Reuse the same D03 visuals' existing decomposed collision assets."""
    xml_dir = ROOT / "handson/handson/sim/envs/xmls"
    root = ET.parse(xml_dir / "d03_handoff_objects.xml").getroot()
    definitions = {m.get("name"): m for m in root.findall("asset/mesh")}
    stage = Usd.Stage.CreateNew(str(OUT / "clutter_collisions.usd"))
    owner = UsdGeom.Xform.Define(stage, "/Clutter")
    stage.SetDefaultPrim(owner.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    result = {}
    for body in root.findall("worldbody/body"):
        name = body.get("name")
        if name in {"d03_target_root", "d03_box"}: continue
        files = set(); visuals = []; visual_paths = []; original_mass = 0.0
        for geom in body.findall("geom"):
            if geom.get("mesh") in definitions:
                path = xml_dir / definitions[geom.get("mesh")].get("file")
                visuals.append(trimesh.load(path, force="mesh", process=False))
                visual_paths.append(path)
                files.update(path.parent.parent.glob("collision/*.obj"))
            if geom.get("class") == "d03_collision":
                size = np.fromstring(geom.get("size"), sep=" ")
                volume = float(8*np.prod(size)) if geom.get("type")=="box" else float(4*np.pi*size[0]**3/3) if geom.get("type")=="sphere" else float(np.pi*size[0]**2*2*size[1])
                original_mass += volume * float(geom.get("density"))
        if not files: raise ValueError(f"{name}: source-fitted colliders are missing")
        # Keep the material inside each referenced body subtree so its binding
        # survives composing /Clutter/<body> into the imported rigid owner.
        contact = UsdShade.Material.Define(stage, f"/Clutter/{name}/Materials/Contact")
        contact_api = UsdPhysics.MaterialAPI.Apply(contact.GetPrim())
        contact_api.CreateStaticFrictionAttr(1.05)
        contact_api.CreateDynamicFrictionAttr(1.0)
        contact_api.CreateRestitutionAttr(0.0)
        normalization = np.eye(3)
        support_pose = None
        if name in {"d03_flat_screwdriver", "d03_phillips_screwdriver", "d03_spoon", "d03_plastic_bolt", "d03_nails", "d03_b_marble_1", "d03_b_marble_2"}:
            source_hull = trimesh.util.concatenate([
                trimesh.load(path, force="mesh", process=False) for path in sorted(files)
            ]).convex_hull
            stable, probabilities = source_hull.compute_stable_poses(sigma=0, n_samples=1, threshold=0)
            # Select the support face closest to the source orientation. Use
            # the minimum rotation taking its outward-up frame to world Z;
            # leave task-level seeded yaw unchanged.
            index = int(np.argmax(stable[:, 2, 2]))
            up = stable[index, 2, :3]
            angle = float(np.arccos(np.clip(up[2], -1, 1)))
            axis = np.cross(up, [0., 0., 1.])
            if np.linalg.norm(axis) > 1e-12:
                normalization = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()
            support_pose = {"method": "nearest stable convex-support face from unchanged source collision vertices",
                "tilt_degrees":float(np.degrees(angle)), "face_probability":float(probabilities[index])}
        transform = np.eye(4); transform[:3,:3] = normalization
        for visual in visuals: visual.apply_transform(transform)
        meshes = []; sources = []; partition_checks = []
        # D03 represents the multi-marble asset as one rigid package. Its
        # upstream 64 bead hulls omit the visible package bottom by 2.4 mm.
        # Fit that one-body exterior directly to the existing visual mesh.
        selected = visual_paths if name == "d03_a_marbles" else sorted(files)
        for i,path in enumerate(selected):
            mesh = trimesh.load(path, force="mesh", process=False)
            if name == "d03_a_marbles": mesh = mesh.convex_hull
            if mesh.bounds is None: raise ValueError(f"{path}: hydrate the repository LFS asset first")
            parts = split_source_convex(mesh) if name in {"d03_nails", "d03_b_marble_1", "d03_b_marble_2"} else [mesh]
            if len(parts) > 1:
                source_hull = mesh.convex_hull
                volume_error = abs(sum(part.volume for part in parts) - source_hull.volume) / source_hull.volume
                vertices = np.concatenate([part.vertices for part in parts])
                normal = source_hull.face_normals
                offset = np.einsum("ij,ij->i", normal, source_hull.triangles[:, 0])
                exterior_error = float(np.max(vertices @ normal.T - offset))
                bounds_error = float(np.max(np.abs(np.asarray([vertices.min(0), vertices.max(0)]) - source_hull.bounds)))
                assert volume_error < 1e-8 and exterior_error < 1e-9 and bounds_error < 1e-9
                partition_checks.append({"source":str(path.relative_to(ROOT)), "pieces":len(parts),
                    "relative_volume_error":float(volume_error), "maximum_exterior_error_m":exterior_error,
                    "bounds_error_m":bounds_error, "piece_vertices":[len(part.vertices) for part in parts]})
            for piece in parts:
                piece.apply_transform(transform)
                collider = usd_mesh(stage, f"/Clutter/{name}/Part{len(meshes)}", piece)
                UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr("convexHull")
                UsdGeom.Imageable(collider.GetPrim()).MakeInvisible()
                UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(contact, UsdShade.Tokens.weakerThanDescendants, "physics")
                meshes.append(piece)
            sources.append({"path":str(path.relative_to(ROOT)), "sha256":sha256(path)})
        merged = trimesh.util.concatenate(meshes)
        hull = merged.convex_hull
        tensor = hull.moment_inertia * original_mass / hull.volume
        eigen, axes = np.linalg.eigh(tensor)
        if np.linalg.det(axes)<0: axes[:,0] *= -1
        quat = Rotation.from_matrix(axes).as_quat()[[3,0,1,2]]
        result[name] = {"bounds_m":merged.bounds.tolist(), "visual_bounds_m":trimesh.util.concatenate(visuals).bounds.tolist(),
            "mass_kg":original_mass, "center_of_mass_m":hull.center_mass.tolist(),
            "diagonal_inertia_kg_m2":eigen.tolist(), "principal_axes_wxyz":quat.tolist(),
            "physics_material":{"static_friction":1.05,"dynamic_friction":1.0,"restitution":0.0},
            "collision_count":len(meshes), "sources":sources,
            "partition_checks":partition_checks,
            "partition_method":"exact source convex plane cuts to GPU-compatible vertex/polygon limits" if name in {"d03_nails", "d03_b_marble_1", "d03_b_marble_2"} else None,
            "source_normalization_rotation":normalization.tolist(), "stable_support_pose":support_pose}
    stage.GetRootLayer().Save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Selected source .gltf/.glb file; inspect it before conversion")
    parser.add_argument("--up-axis", choices=["x", "y", "z"], required=True)
    parser.add_argument("--diameter-m", type=float, default=0.036)
    parser.add_argument("--mass-kg", type=float, default=0.040)
    args = parser.parse_args()
    if args.source.suffix.lower() not in {".gltf", ".glb"}:
        parser.error("Provide the selected model's glTF or GLB source; no visible replacement is generated")
    if not 0.030 <= args.diameter_m <= 0.040 or not 0.030 <= args.mass_kg <= 0.060:
        parser.error("Requested scale/mass is outside the approved initial GEN4 range")
    scene = trimesh.load(args.source, force="scene", process=False)
    rotations = {
        "z": np.eye(3),
        "y": np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        "x": np.asarray([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    }
    meshes = []
    axis = np.eye(4)
    axis[:3, :3] = rotations[args.up_axis]
    for node in sorted(scene.graph.nodes_geometry):
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            continue
        mesh.apply_transform(axis @ transform)
        meshes.append(mesh)
    if not meshes:
        raise ValueError("The selected source has no triangle mesh")
    vertices = np.concatenate([mesh.vertices for mesh in meshes])
    if not np.isfinite(vertices).all():
        raise ValueError("Selected asset has nonfinite geometry")
    source_bounds = np.asarray([vertices.min(axis=0), vertices.max(axis=0)])
    source_size = np.ptp(source_bounds, axis=0)
    scale = args.diameter_m / max(source_size[:2])
    center = source_bounds.mean(axis=0)
    for mesh in meshes:
        mesh.apply_translation(-center)
        mesh.apply_scale(scale)
    merged = trimesh.util.concatenate(meshes)
    occupied = merged.convex_hull
    if occupied.volume <= 0:
        raise ValueError("The selected closed model does not enclose a positive volume")
    inertia = occupied.moment_inertia * args.mass_kg / occupied.volume
    diagonal, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    principal_xyzw = Rotation.from_matrix(axes).as_quat()
    principal_wxyz = principal_xyzw[[3, 0, 1, 2]].tolist()
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        raise ValueError("Source-derived inertia is not positive definite")
    mass_center = occupied.center_mass.tolist()
    OUT.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(OUT / "glue_stick.usd"))
    root = UsdGeom.Xform.Define(stage, "/GlueStick")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    physics = UsdShade.Material.Define(stage, "/GlueStick/Materials/Contact")
    material_api = UsdPhysics.MaterialAPI.Apply(physics.GetPrim())
    material_api.CreateStaticFrictionAttr(1.0)
    material_api.CreateDynamicFrictionAttr(0.8)
    material_api.CreateRestitutionAttr(0.0)
    collision_paths = []
    collision_vertices = []
    for i, mesh in enumerate(meshes):
        visual = usd_mesh(stage, f"/GlueStick/Visual/Part{i}", mesh)
        bind_visual(stage, visual, mesh, i)
        collision_source = mesh.copy()
        collision_source.merge_vertices(merge_tex=True, merge_norm=True)
        # Materials group disconnected cap/base meshes. A hull of the whole
        # material would bridge their gap and enlarge the handoff grip surface.
        components = trimesh.graph.connected_components(
            collision_source.face_adjacency,
            nodes=np.arange(len(collision_source.faces)), engine="scipy",
        )
        for part, face_ids in enumerate(components):
            vertices = collision_source.vertices[np.unique(collision_source.faces[face_ids])]
            if np.linalg.matrix_rank(vertices - vertices.mean(axis=0)) < 3:
                continue
            hull = trimesh.convex.convex_hull(vertices)
            if hull.volume <= 1e-12:
                continue
            if i == 1 and vertices[:, 2].max() < 0:
                # Preserve the source base's recessed bottom. A single hull
                # seals a 9.7 mm cavity; also bound the cooker's vertex count.
                import coacd
                vertex_ids, inverse = np.unique(collision_source.faces[face_ids], return_inverse=True)
                component = coacd.Mesh(collision_source.vertices[vertex_ids], inverse.reshape(-1, 3))
                parts = coacd.run_coacd(component, threshold=0.0007, real_metric=True,
                    max_convex_hull=32, decimate=True, max_ch_vertex=128,
                    seed=4, preprocess_mode="auto")
                hulls = [trimesh.Trimesh(v, f, process=False) for v, f in parts]
            else:
                if len(hull.vertices) > 128:
                    # Source support vertices give an inscribed convex hull;
                    # avoid leaving vertex reduction to the PhysX cooker.
                    n = np.arange(128)
                    z = 1.0 - 2.0 * (n + 0.5) / 128
                    phi = n * np.pi * (3.0 - np.sqrt(5.0))
                    directions = np.column_stack([np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z])
                    selected = np.unique(np.argmax(hull.vertices @ directions.T, axis=0))
                    hull = trimesh.convex.convex_hull(hull.vertices[selected])
                hulls = [hull]
            for subpart, hull in enumerate(hulls):
                collision = usd_mesh(stage, f"/GlueStick/Collision/Part{i}_{part}_{subpart}", hull)
                UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr("convexHull")
                UsdGeom.Imageable(collision.GetPrim()).MakeInvisible()
                UsdShade.MaterialBindingAPI.Apply(collision.GetPrim()).Bind(physics, UsdShade.Tokens.weakerThanDescendants, "physics")
                collision_paths.append(str(collision.GetPath()))
                collision_vertices.append(hull.vertices)
    if not collision_paths:
        raise ValueError("Selected model has no volumetric collision component")
    stage.GetRootLayer().Save()
    occupied = trimesh.convex.convex_hull(np.concatenate([merged.vertices, *collision_vertices]))
    source_dir = OUT / "source"
    source_dir.mkdir(exist_ok=True)
    shutil.copy2(args.source, source_dir / args.source.name)
    (OUT / "GLUE_LICENSE.md").write_text(
        "# Customisable Glue Stick\n\n"
        "Author: NateSquared (https://sketchfab.com/NateSquared).\n"
        f"Source: {SOURCE_URL}\n\n"
        "License: Creative Commons Attribution 4.0 International (CC BY 4.0).\n"
        "https://creativecommons.org/licenses/by/4.0/\n\n"
        "Downloaded with the user's explicit authorization on 2026-09-07.\n"
        "The original GLB embeds matching author, source and license metadata.\n\n"
        "Changes: axis conversion, uniform scale, rigid-body mass/inertia and\n"
        "invisible source-derived connected-component convex colliders.\n"
        "Original visual topology, UVs, normals and 2K PBR maps are retained.\n"
        "The cap and base are fixed parts of one closed rigid body.\n"
    )
    tray_manifest = json.loads((TRAY_DIR / "asset_manifest.json").read_text())["assets"]["two_handle_tray"]
    tray_stage = Usd.Stage.Open(str(TRAY_DIR / "bc005_selected_wooden_tray.usd"))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"], False, True)
    tray_bound = cache.ComputeLocalBound(tray_stage.GetDefaultPrim()).ComputeAlignedRange()
    raw_interior = tray_manifest["usable_deck_bounds_xy_m"]
    report = {
        "task_id": "GEN4", "source_url": SOURCE_URL, "author": "NateSquared",
        "license": "CC BY 4.0", "source_file": args.source.name,
        "source_sha256": sha256(args.source), "source_up_axis": args.up_axis,
        "uniform_scale": scale, "source_bounds": source_bounds.tolist(),
        "glue": {
            "bounds_m": occupied.bounds.tolist(), "visual_bounds_m": merged.bounds.tolist(), "size_m": merged.extents.tolist(),
            "occupied_vertices_m": occupied.vertices.tolist(), "mass_kg": args.mass_kg,
            "center_of_mass_m": mass_center,
            "inertia_method": "uniform source visual envelope at the specified mass",
            "diagonal_inertia_kg_m2": diagonal.tolist(), "principal_axes_wxyz": principal_wxyz,
            "inertia_tensor_kg_m2": inertia.tolist(), "collision_paths": collision_paths,
            "visual_mesh_count": len(meshes), "visual_triangles": sum(len(m.faces) for m in meshes),
            "rigid_body_policy": "closed cap and base fixed to one body",
            "physics_material": {"static_friction": 1.0, "dynamic_friction": 0.8, "restitution": 0},
        },
        "distractors": prepare_clutter(),
        "tray": {
            "bounds_m": [list(tray_bound.GetMin()), list(tray_bound.GetMax())],
            "interior_xy_m": [raw_interior["min"], raw_interior["max"]],
            "floor_z_m": tray_manifest["deck_top_local_z_m"],
            "asset_sha256": sha256(TRAY_DIR / "bc005_selected_wooden_tray.usd"),
            "collision_count": tray_manifest["collision_api_count"], "role": "fixed fixture",
        },
        "pending_validation": ["source appearance and texture agreement", "compound-hull protrusion", "two-hand grasp space", "full collision-shape overlap audit", "live PhysX and collection"],
    }
    (OUT / "asset_geometry.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(TRAY_DIR / "TRAY_LICENSE.txt", OUT / "TRAY_LICENSE.txt")
    print(json.dumps({"output": str(OUT), "size_m": merged.extents.tolist(), "collision_parts": len(collision_paths), "live_validation": "pending"}, indent=2))


if __name__ == "__main__":
    main()
