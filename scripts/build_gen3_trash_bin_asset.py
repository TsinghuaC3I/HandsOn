#!/usr/bin/env python3
"""Convert the licensed GEN3 GLB into a passive USD articulation on CPU.

Only existing source triangles are used for visible geometry. Collision hulls
are derived from each source connected component; no lid, handle, wall or
other functional surface is synthesized. Run with the GEN3 build environment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import struct

import numpy as np
import trimesh
import vhacdx
import manifold3d
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "handson"))
from handson.sim.isaaclab.gen3_hinged_trash_bin_scene import (
    GEN3_ASSET_DIRECTORY, GEN3_SOURCE_UID, GEN3_SOURCE_URL,
    GEN3_ASSET_HEIGHT_M, GEN3_LID_MASS_KG, GEN3_BASE_MASS_KG,
)


def mesh_prim(stage, path, mesh):
    usd = UsdGeom.Mesh.Define(stage, path)
    usd.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32)))
    usd.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mesh.faces)))
    usd.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(mesh.faces.reshape(-1), dtype=np.int32)))
    usd.CreateSubdivisionSchemeAttr().Set("none")
    usd.CreateDoubleSidedAttr().Set(True)
    return usd


def texture_material(stage, path, mesh, texture_dir, stem):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    src = mesh.visual.material
    uv_reader = UsdShade.Shader.Define(stage, path + "/UV")
    uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
    uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    for image_attr, output, channel, colorspace in (
        ("baseColorTexture", "diffuseColor", "rgb", "sRGB"),
        ("metallicRoughnessTexture", "roughness", "g", "raw"),
        ("normalTexture", "normal", "rgb", "raw"),
    ):
        image = getattr(src, image_attr, None)
        if image is None:
            continue
        filename = f"{stem}_{image_attr}.png"
        image.save(texture_dir / filename)
        tex = UsdShade.Shader.Define(stage, path + "/" + image_attr)
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("textures/" + filename))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(colorspace)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_reader.ConnectableAPI(), "result")
        if output == "normal":
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
            tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        typ = Sdf.ValueTypeNames.Color3f if output == "diffuseColor" else Sdf.ValueTypeNames.Normal3f if output == "normal" else Sdf.ValueTypeNames.Float
        tex.CreateOutput(channel, Sdf.ValueTypeNames.Float3 if channel == "rgb" else Sdf.ValueTypeNames.Float)
        shader.CreateInput(output, typ).ConnectToSource(tex.ConnectableAPI(), channel)
    if getattr(src, "baseColorTexture", None) is None:
        color = np.asarray(src.baseColorFactor if src.baseColorFactor is not None else [255,255,255,255], dtype=float)[:3]/255
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(src.roughnessFactor if src.roughnessFactor is not None else 0.5))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(src.metallicFactor or 0.0))
    return mat


def rigid_body(stage, path, meshes, mass):
    body = UsdGeom.Xform.Define(stage, path)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    combined = trimesh.util.concatenate([trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False) for m in meshes])
    combined.merge_vertices()
    if combined.is_watertight and combined.volume > 0:
        props = combined.mass_properties
        center = props.center_mass
        inertia = props.inertia * mass / props.mass
        mass_method = "source_solid_volume"
    else:
        # The fixed bin shell has an open underside seated on a separate
        # existing source bottom. Use its measured surface distribution;
        # do not synthesize triangles to close that source seam.
        triangles = combined.triangles
        weights = combined.area_faces
        center = np.average(triangles.mean(axis=1), axis=0, weights=weights)
        q = triangles - center
        # Uniform triangular lamina: E[xx^T] = (sum(vi vi^T)+ss^T)/12.
        second = (np.einsum("nki,nkj->nij", q, q) + np.einsum("ni,nj->nij", q.sum(1), q.sum(1))) / 12
        covariance = np.average(second, axis=0, weights=weights)
        inertia = mass * (np.trace(covariance)*np.eye(3) - covariance)
        mass_method = "source_triangular_shell_area"
    eig, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    xyzw = Rotation.from_matrix(axes).as_quat()
    api = UsdPhysics.MassAPI.Apply(body.GetPrim())
    api.CreateMassAttr(mass)
    api.CreateDensityAttr(1000.0)
    api.CreateCenterOfMassAttr(Gf.Vec3f(*center))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*eig))
    api.CreatePrincipalAxesAttr(Gf.Quatf(float(xyzw[3]), Gf.Vec3f(*xyzw[:3])))
    if not np.all(np.isfinite(inertia)) or not np.all(eig > 0):
        raise ValueError(f"Invalid source-derived inertia: {path}")
    return body, {"mass_kg": mass, "method": mass_method, "center_of_mass_m": center.tolist(), "inertia_kg_m2": inertia.tolist(), "principal_inertia_kg_m2": eig.tolist()}


def components(mesh):
    plain = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    plain.merge_vertices()
    labels = trimesh.graph.connected_component_labels(plain.face_adjacency, node_count=len(plain.faces))
    for label in np.unique(labels):
        part = trimesh.Trimesh(vertices=plain.vertices.copy(), faces=plain.faces[labels == label], process=False)
        part.remove_unreferenced_vertices()
        yield part


def convexes(mesh, max_hulls):
    for part_index, part in enumerate(sorted(components(mesh), key=lambda item: -len(item.faces))):
        if part.is_convex and len(part.convex_hull.vertices) <= 32:
            yield part.convex_hull
            continue
        arrays = vhacdx.compute_vhacd(
            np.asarray(part.vertices, dtype=np.float64),
            np.column_stack((np.full(len(part.faces), 3), part.faces)).ravel().astype(np.uint32),
            maxConvexHulls=(max_hulls if part_index == 0 else 8), resolution=400000,
            minimumVolumePercentErrorAllowed=0.5, maxNumVerticesPerCH=32,
            shrinkWrap=True, asyncACD=False,
        )
        for vertices, faces in arrays:
            hull = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).convex_hull
            if len(hull.vertices) < 4 or hull.volume <= 0:
                raise ValueError("Source-derived collision contains a degenerate hull")
            yield hull


def _clip_polygon(polygon, normal, offset):
    """Clip existing source triangles; intersections stay on source surfaces."""
    output = []
    for previous, current in zip(np.roll(polygon, 1, axis=0), polygon):
        before = float(previous @ normal - offset)
        after = float(current @ normal - offset)
        if (before >= -1e-10) != (after >= -1e-10):
            output.append(previous + before / (before - after) * (current - previous))
        if after >= -1e-10:
            output.append(current)
    return np.asarray(output).reshape(-1, 3)


def source_plane_convexes(mesh):
    """Partition the existing solid hinge housing along its authored faces.

    Its narrow recesses are poorly represented by voxel hulls. Boolean plane
    splits preserve those recesses. Numerical fragments below 0.01 mm^3 are
    omitted from collision only; all original visual triangles remain.
    """
    solid = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    solid.merge_vertices()
    if not solid.is_watertight:
        raise ValueError("Source-plane splitting requires the source solid")
    manifold = manifold3d.Manifold(manifold3d.Mesh(
        np.asarray(solid.vertices, np.float32), np.asarray(solid.faces, np.uint32)))
    planes = np.unique(np.round(np.column_stack([
        solid.face_normals,
        np.einsum("ij,ij->i", solid.face_normals, solid.triangles[:, 0]),
    ]), 8), axis=0)
    stack = [manifold]
    splits = 0
    while stack:
        part = stack.pop()
        volume = part.volume()
        if volume < 1e-11:
            continue
        vertices = np.asarray(part.to_mesh().vert_properties)[:, :3]
        hull = trimesh.convex.convex_hull(vertices)
        if (hull.volume-volume <= max(1e-12, volume*1e-4)
                and len(hull.vertices) <= 32):
            yield hull
            continue
        candidates = []
        for normal, offset in ([(p[:3], p[3]) for p in planes]
                + [(np.eye(3)[a], float(vertices[:, a].mean())) for a in range(3)]):
            distance = vertices @ normal-offset
            if distance.min() > -1e-6 or distance.max() < 1e-6:
                continue
            positive, negative = part.split_by_plane(normal, float(offset))
            pieces = [p for half in (positive, negative) for p in half.decompose()
                      if p.volume() > 1e-13]
            if len(pieces) >= 2:
                candidates.append((sum(p.hull().volume() for p in pieces),
                    abs(positive.volume()-negative.volume()), pieces))
        if not candidates or splits >= 100:
            raise ValueError("Source hinge housing could not be partitioned")
        stack.extend(min(candidates, key=lambda item: item[:2])[2])
        splits += 1


def radial_surface_convexes(mesh):
    """Preserve the downloaded barrel/lid cavities with source-surface panels.

    Voxel decomposition filled portions of the thin source shells. Partition
    their existing triangles into 32 angular wedges and authored height bands,
    then hull the clipped surface points. No visible triangles are changed.
    """
    center = mesh.bounds.mean(axis=0)
    heights = np.sort(np.unique(mesh.vertices[:, 2]))
    groups = []
    for height in heights:
        if not groups or height - groups[-1][-1] > 5e-5:
            groups.append([float(height)])
        else:
            groups[-1].append(float(height))
    levels = [min(groups[0])] + [sum(g)/len(g) for g in groups[1:-1]] + [max(groups[-1])]
    for sector in range(32):
        start, stop = np.radians([sector*11.25, (sector+1)*11.25])
        normals = (np.array([-np.sin(start), np.cos(start), 0.]),
                   np.array([np.sin(stop), -np.cos(stop), 0.]))
        polygons = []
        for triangle in mesh.triangles:
            polygon = triangle
            for normal in normals:
                polygon = _clip_polygon(polygon, normal, float(center @ normal))
                if len(polygon) < 3:
                    break
            if len(polygon) >= 3:
                polygons.append(polygon)
        sector_hulls = []
        for lower, upper in zip(levels[:-1], levels[1:]):
            vertices = []
            for polygon in polygons:
                if polygon[:, 2].max() < lower-1e-10 or polygon[:, 2].min() > upper+1e-10:
                    continue
                # A floor face touching only the boundary of the wall band
                # contributes no volume to this band. Including its center
                # in a hull would fill the cavity above that floor.
                if (np.all(np.abs(polygon[:, 2]-lower) < 1e-7)
                        or np.all(np.abs(polygon[:, 2]-upper) < 1e-7)):
                    continue
                clipped = _clip_polygon(polygon, np.array([0.,0.,1.]), lower)
                if len(clipped) < 3:
                    continue
                clipped = _clip_polygon(clipped, np.array([0.,0.,-1.]), -upper)
                if len(clipped) >= 3:
                    vertices.extend(clipped)
            points = np.unique(np.round(vertices, 10), axis=0) if vertices else np.empty((0,3))
            if len(points) < 4 or np.linalg.matrix_rank(points-points.mean(0), tol=1e-9) < 3:
                continue
            hull = trimesh.convex.convex_hull(points)
            if hull.volume < 1e-13:
                continue
            if len(hull.vertices) > 32 or len(hull.faces) > 64:
                raise ValueError("Source panel exceeds the GPU convex topology budget")
            if sector_hulls:
                previous = sector_hulls[-1]
                merged = trimesh.convex.convex_hull(np.vstack([previous.vertices, hull.vertices]))
                volume = previous.volume + hull.volume
                # Merge adjacent bands only when their union is convex;
                # no cavity or source-surface detail may be filled in.
                if (len(merged.vertices) <= 32 and len(merged.faces) <= 64
                        and abs(merged.volume-volume) <= max(1e-13, volume*1e-5)):
                    sector_hulls[-1] = merged
                    continue
            sector_hulls.append(hull)
        yield from sector_hulls


def radial_shell_convexes(mesh):
    """Cut the actual source solid into cavity-preserving convex sectors.

    Boolean clipping retains the existing floor and flat lid centers. Open
    decorative seams use the source-surface partition above, without caps.
    """
    center = mesh.bounds.mean(0)
    for component in components(mesh):
        if not component.is_watertight:
            yield from radial_surface_convexes(component)
            continue
        solid = manifold3d.Manifold(manifold3d.Mesh(
            np.asarray(component.vertices, np.float32), np.asarray(component.faces, np.uint32)))
        levels = []
        for height in np.sort(np.unique(component.vertices[:, 2])):
            if not levels or height-levels[-1] > 5e-5:
                levels.append(float(height))
        levels[-1] = float(component.bounds[1, 2])
        for sector in range(32):
            start, stop = np.radians([sector*11.25, (sector+1)*11.25])
            wedge = solid
            for normal in (np.array([-np.sin(start), np.cos(start), 0.]),
                           np.array([np.sin(stop), -np.cos(stop), 0.])):
                wedge = wedge.split_by_plane(normal, float(center @ normal))[0]
            sector_hulls = []
            for lower, upper in zip(levels[:-1], levels[1:]):
                # Exclude zero-volume floor faces coincident with the cut:
                # Manifold can retain those faces on the wall band, where
                # taking their hull would incorrectly fill the cavity.
                # The 0.1 micrometre inset is covered by the surface audit.
                band = wedge.split_by_plane([0., 0., 1.], lower+1e-7)[0]
                band = band.split_by_plane([0., 0., -1.], -upper+1e-7)[0]
                for piece in band.decompose():
                    if piece.volume() <= 1e-13:
                        continue
                    hull = trimesh.convex.convex_hull(np.asarray(piece.to_mesh().vert_properties)[:, :3])
                    if len(hull.vertices) > 32 or len(hull.faces) > 64:
                        raise ValueError("Source solid sector exceeds GPU convex topology budget")
                    if sector_hulls:
                        previous = sector_hulls[-1]
                        merged = trimesh.convex.convex_hull(np.vstack([previous.vertices, hull.vertices]))
                        volume = previous.volume+hull.volume
                        gap = max(0., hull.bounds[0, 2]-previous.bounds[1, 2])
                        # Rejoin the numerical insets only. Their cross-
                        # section bound cannot admit a macroscopic cavity.
                        inset_volume = (min(np.prod(previous.extents[:2]), np.prod(hull.extents[:2])) * gap
                                        if gap < 1e-6 else 0.)
                        if (len(merged.vertices) <= 32 and len(merged.faces) <= 64
                                and abs(merged.volume-volume) <= max(1e-13, volume*1e-5)+inset_volume):
                            sector_hulls[-1] = merged
                            continue
                    sector_hulls.append(hull)
            yield from sector_hulls


def build():
    source = GEN3_ASSET_DIRECTORY / "source" / f"{GEN3_SOURCE_UID}.glb"
    runtime = GEN3_ASSET_DIRECTORY / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    texture_dir = runtime / "textures"
    texture_dir.mkdir(exist_ok=True)
    scene = trimesh.load(source, force="scene", process=False)
    with source.open("rb") as stream:
        stream.read(12)
        length, _ = struct.unpack("<II", stream.read(8))
        gltf = json.loads(stream.read(length))
    # Source animation moves node 6 (lid and its existing connector). The
    # foot pedal's animation is intentionally unused; that hardware is fixed.
    if len(gltf["meshes"]) != 8 or gltf["animations"][0]["channels"][0]["target"]["node"] != 6:
        raise ValueError("Source structure changed; inspect the moving links")
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    entries = []
    for index, source_mesh in enumerate(gltf["meshes"]):
        name = source_mesh["name"]
        nodes = [n for n in scene.graph.nodes_geometry if scene.graph[n][1] == name]
        if len(nodes) != 1:
            raise ValueError(f"Ambiguous source mesh {index}")
        transform, geometry = scene.graph[nodes[0]]
        mesh = scene.geometry[geometry].copy()
        mesh.apply_transform(transform)
        mesh.vertices = mesh.vertices @ rotation.T
        entries.append((f"SourceMesh{index:02d}", mesh))
    vertices = np.concatenate([m.vertices for _, m in entries])
    lower, upper = vertices.min(0), vertices.max(0)
    # Center the actual bin, not its protruding floor pedal.
    body_bounds = entries[0][1].bounds
    origin = np.array([(body_bounds[0,0]+body_bounds[1,0])/2, (body_bounds[0,1]+body_bounds[1,1])/2, lower[2]])
    scale = GEN3_ASSET_HEIGHT_M / (upper[2]-lower[2])
    hinge_node = gltf["nodes"][6]["name"]
    hinge = ((rotation @ scene.graph[hinge_node][0][:3, 3]) - origin) * scale
    lid_names = {"SourceMesh01", "SourceMesh02"}
    for _, mesh in entries:
        mesh.vertices = (mesh.vertices - origin) * scale
    lid_meshes = [m for n, m in entries if n in lid_names]
    base_meshes = [m for n, m in entries if n not in lid_names]
    for mesh in lid_meshes:
        mesh.vertices -= hinge
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/TrashBin")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("handson:source", GEN3_SOURCE_URL)
    root.GetPrim().SetCustomDataByKey("handson:license", "CC-BY-4.0")
    root.GetPrim().SetCustomDataByKey("handson:selfAuthoredVisualGeometry", False)
    base, base_mass = rigid_body(stage, "/TrashBin/trash_bin_base", base_meshes, GEN3_BASE_MASS_KG)
    lid, lid_mass = rigid_body(stage, "/TrashBin/trash_bin_lid", lid_meshes, GEN3_LID_MASS_KG)
    lid.AddTranslateOp().Set(Gf.Vec3d(*hinge))
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/TrashBin/trash_bin_lid/lid_hinge")
    joint.CreateBody0Rel().SetTargets([base.GetPath()])
    joint.CreateBody1Rel().SetTargets([lid.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*hinge))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0,0,0))
    joint.CreateAxisAttr("X")
    joint.CreateLowerLimitAttr(-90.0)
    joint.CreateUpperLimitAttr(0.0)
    joint.CreateCollisionEnabledAttr(True)
    # Raw schema attributes are portable through usd-core; Isaac adds the
    # PhysxSchema APIs at composition before the first physics step.
    physics_material = UsdShade.Material.Define(stage, "/TrashBin/Materials/Contact")
    physics_api = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    physics_api.CreateStaticFrictionAttr(0.8)
    physics_api.CreateDynamicFrictionAttr(0.6)
    physics_api.CreateRestitutionAttr(0.0)
    report = {"source": GEN3_SOURCE_URL, "source_uid": GEN3_SOURCE_UID,
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "license": "CC-BY-4.0", "uniform_scale": float(scale),
              "source_bounds_z_up_source_units": [lower.tolist(), upper.tolist()],
              "source_origin_z_up_source_units": origin.tolist(), "hinge_in_base_m": hinge.tolist(),
              "closed_height_m": GEN3_ASSET_HEIGHT_M, "joint_axis": "X", "opening_sign": -1,
              "base_mass_properties": base_mass, "lid_mass_properties": lid_mass,
              "visual_meshes": [], "colliders": [], "self_authored_visible_geometry": False,
              "live_PhysX_verified": False}
    for index, (node, mesh) in enumerate(entries):
        parent = lid.GetPath() if node in lid_names else base.GetPath()
        visual_path = str(parent) + "/Visuals/" + node
        visual = mesh_prim(stage, visual_path, mesh)
        visual.GetPrim().SetCustomDataByKey("handson:renderClassification", "source_visual")
        visual.GetPrim().SetCustomDataByKey("handson:sourceNode", node)
        # Trimesh uses lower-left UVs; preserve source texture coordinates.
        uv = UsdGeom.PrimvarsAPI(visual).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        uv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(mesh.visual.uv, dtype=np.float32)))
        material = texture_material(stage, f"/TrashBin/Materials/Source{index}", mesh, texture_dir, f"source{index}")
        UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)
        report["visual_meshes"].append({"source_node": node, "path": visual_path, "triangles": len(mesh.faces), "bounds_m": mesh.bounds.tolist()})
        print(f"Deriving collision from {node}: {len(mesh.faces)} source triangles", flush=True)
        hulls = (radial_shell_convexes(mesh) if node in {"SourceMesh00", "SourceMesh01", "SourceMesh07"}
                 else source_plane_convexes(mesh) if node == "SourceMesh06"
                 else convexes(mesh, 32))
        for h, hull in enumerate(hulls):
            path = str(parent) + f"/Collisions/source{index}_{h:03d}"
            collision = mesh_prim(stage, path, hull)
            UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr("convexHull")
            collision.MakeInvisible()
            collision.GetPrim().SetCustomDataByKey("handson:renderClassification", "collision_only_source_derived")
            collision.GetPrim().SetCustomDataByKey("handson:sourceNode", node)
            UsdShade.MaterialBindingAPI.Apply(collision.GetPrim()).Bind(physics_material, materialPurpose="physics")
            report["colliders"].append({"path": path, "source_node": node, "vertex_count": len(hull.vertices), "bounds_m": hull.bounds.tolist()})
    # The downloaded animation models the hinge as an overlapping connector
    # and housing. Filter only that source housing against the attached lid;
    # the lid retains collisions with the bin wall/rim and with both robots.
    housing_paths = [entry["path"] for entry in report["colliders"] if entry["source_node"] == "SourceMesh06"]
    for path in housing_paths:
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(path)).CreateFilteredPairsRel().SetTargets([lid.GetPath()])
    report["intended_hinge_assembly"] = {
        "base_source_mesh": "SourceMesh06", "attached_lid_source_meshes": ["SourceMesh01", "SourceMesh02"],
        "reason": "Existing animated connector mates inside its fixed hinge housing; the revolute joint represents this assembly.",
        "filtered_housing_colliders": housing_paths,
        "filtered_other_body": str(lid.GetPath()),
        "bin_wall_rim_and_robot_collisions_filtered": False,
    }
    stage.GetRootLayer().Export(str(runtime / "trash_bin.usda"))
    (GEN3_ASSET_DIRECTORY / "geometry_report.json").write_text(json.dumps(report, indent=2)+"\n")
    manifest = {"task_id": "GEN3", "source": GEN3_SOURCE_URL, "license": "CC-BY-4.0", "files": []}
    for path in sorted(GEN3_ASSET_DIRECTORY.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"].append({"path": path.relative_to(GEN3_ASSET_DIRECTORY).as_posix(), "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (GEN3_ASSET_DIRECTORY / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"visuals": len(report["visual_meshes"]), "colliders": len(report["colliders"]), "hinge_m": hinge.tolist(), "scale": scale}), flush=True)

if __name__ == "__main__":
    build()
