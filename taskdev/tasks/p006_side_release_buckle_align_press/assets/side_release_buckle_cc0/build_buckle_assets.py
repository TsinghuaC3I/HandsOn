#!/usr/bin/env python3
"""Convert the audited CC0 side-release-buckle STL into two USD assets.

The source STL is a two-part print plate made from disconnected closed boxes.
This converter preserves the source solids except for a disclosed entry chamfer
on each male latch tooth. The chamfer has matching visual and convex collision
geometry; other solids retain their source appearance and analytic boxes.  The two outer male tongues
are emitted as separate prismatic links, so contact from a finger can push
them inward against a short, damped travel limit instead of leaving all three
tongues welded into one rigid body.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
from pathlib import Path
import struct


EXPECTED_SOURCE_SHA256 = "f68abfbfac56f746fef1f411ec68d511d4cb567cfb0f10472cb8e26168d57b23"
MM_TO_M = 0.002  # 2x the printable source size for robust Wuji grasping.
FEMALE_X_CENTER_MM = 17.0
MALE_BASE_CENTER_X_MM = 51.2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_binary_stl(path: Path):
    payload = path.read_bytes()
    if len(payload) < 84:
        raise ValueError("source STL is too small")
    count = struct.unpack_from("<I", payload, 80)[0]
    if len(payload) != 84 + 50 * count:
        raise ValueError("source STL is not the audited binary STL")
    triangles = []
    for index in range(count):
        values = struct.unpack_from("<12fH", payload, 84 + 50 * index)
        vertices = tuple(
            tuple(round(float(value), 6) for value in values[3 + 3 * corner : 6 + 3 * corner])
            for corner in range(3)
        )
        triangles.append(vertices)
    return triangles


def connected_components(triangles):
    vertex_to_faces = defaultdict(list)
    for face_index, triangle in enumerate(triangles):
        for vertex in triangle:
            vertex_to_faces[vertex].append(face_index)
    adjacency = [set() for _ in triangles]
    for face_indices in vertex_to_faces.values():
        for face_index in face_indices:
            adjacency[face_index].update(face_indices)
    seen = set()
    components = []
    for start in range(len(triangles)):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        component = []
        while queue:
            face_index = queue.popleft()
            component.append(face_index)
            for neighbor in adjacency[face_index]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def transform(vertex, part):
    x_mm, y_mm, z_mm = vertex
    if part == "female":
        # Rotate the printed female half 180 degrees around +Z so its open
        # mouth (source x=0) points along local +X toward the male reset.
        return (
            -(x_mm - FEMALE_X_CENTER_MM) * MM_TO_M,
            -y_mm * MM_TO_M,
            z_mm * MM_TO_M,
        )
    # Rotate the printed male half 180 degrees around +Z.  Its prongs then
    # point toward local -X while its strap/body remains at the grasp end.
    return (
        -(x_mm - MALE_BASE_CENTER_X_MM) * MM_TO_M,
        -y_mm * MM_TO_M,
        z_mm * MM_TO_M,
    )


def fmt_vec(values):
    return "(" + ", ".join(f"{float(value):.9g}" for value in values) + ")"


def component_bounds(triangles, component, part):
    vertices = [transform(vertex, part) for face in component for vertex in triangles[face]]
    lower = tuple(min(vertex[axis] for vertex in vertices) for axis in range(3))
    upper = tuple(max(vertex[axis] for vertex in vertices) for axis in range(3))
    center = tuple((lower[axis] + upper[axis]) * 0.5 for axis in range(3))
    extent = tuple(upper[axis] - lower[axis] for axis in range(3))
    return lower, upper, center, extent


def _source_component_bounds(triangles, component):
    vertices = [vertex for face in component for vertex in triangles[face]]
    lower = tuple(min(vertex[axis] for vertex in vertices) for axis in range(3))
    upper = tuple(max(vertex[axis] for vertex in vertices) for axis in range(3))
    center = tuple((lower[axis] + upper[axis]) * 0.5 for axis in range(3))
    extent = tuple(upper[axis] - lower[axis] for axis in range(3))
    return lower, upper, center, extent


def _mesh_lines(
    triangles,
    components,
    part,
    color,
    *,
    name,
    origin=(0.0, 0.0, 0.0),
):
    points = []
    indices = []
    for component in components:
        for face in component:
            for vertex in triangles[face]:
                transformed = transform(vertex, part)
                point = tuple(
                    transformed[axis] - float(origin[axis]) for axis in range(3)
                )
                indices.append(len(points))
                points.append(point)
    return [
        "",
        f'    def Mesh "{name}"',
        "    {",
        f"        point3f[] points = [{', '.join(fmt_vec(point) for point in points)}]",
        f"        int[] faceVertexIndices = [{', '.join(str(index) for index in indices)}]",
        f"        int[] faceVertexCounts = [{', '.join('3' for _ in range(len(indices) // 3))}]",
        '        token subdivisionScheme = "none"',
        "        bool doubleSided = 1",
        f"        color3f[] primvars:displayColor = [{fmt_vec(color)}]",
        "    }",
    ]


def _collision_lines(
    triangles,
    component,
    part,
    *,
    name,
    origin=(0.0, 0.0, 0.0),
    inward_center=True,
):
    _, _, center, extent = component_bounds(triangles, component, part)
    source_extent = _source_component_bounds(triangles, component)[3]
    # Long tongues are about 22 mm in source X; the 6 mm latch bumps must not
    # inherit the tongue's reduced-height proxy dimensions.
    is_male_prong = part == "male" and source_extent[0] >= 20.0
    if is_male_prong:
        if inward_center:
            center = (center[0], center[1] * 0.90, center[2])
        extent = (extent[0], extent[1] * 0.90, extent[2] * 0.70)
    center = tuple(center[axis] - float(origin[axis]) for axis in range(3))
    return [
        "",
        f'    def Cube "{name}" (',
        '        prepend apiSchemas = ["PhysicsCollisionAPI"]',
        "    )",
        "    {",
        "        bool physics:collisionEnabled = 1",
        "        double size = 1",
        f"        double3 xformOp:scale = {fmt_vec(extent)}",
        f"        double3 xformOp:translate = {fmt_vec(center)}",
        '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
        "        color3f[] primvars:displayColor = [(0.02, 0.02, 0.02)]",
        "        float[] primvars:displayOpacity = [0]",
        '        uniform token purpose = "guide"',
        '        token visibility = "invisible"',
        "    }",
    ]


def latch_tooth_geometry(triangles, component, origin):
    """Bevel the leading 4 mm within the original tooth's source bounds.

    The source rectangular shoulder cannot turn an axial push into inward
    prong compression. Leave a 0.1 mm nose and retain the original trailing
    tooth, so the female side rail supplies a real inward contact normal.
    """
    lower, upper, center, _ = component_bounds(triangles, component, "male")
    sign = 1.0 if center[1] > 0 else -1.0
    inner = min(abs(lower[1]), abs(upper[1]))
    outer = max(abs(lower[1]), abs(upper[1]))
    x0, x1 = lower[0], upper[0]
    outline = [(x0, inner), (x1, inner), (x1, outer),
               (x0 + .004, outer), (x0, inner + .0001)]
    points = [(x - origin[0], sign * y - origin[1], z - origin[2])
              for z in (lower[2], upper[2]) for x, y in outline]
    faces = [[4, 3, 2, 1, 0], [5, 6, 7, 8, 9]]
    faces += [[i, (i + 1) % 5, (i + 1) % 5 + 5, i + 5] for i in range(5)]
    if sign < 0:
        faces = [list(reversed(face)) for face in faces]
    return points, faces


def _latch_tooth_lines(triangles, component, origin, *, name, collision, color):
    points, faces = latch_tooth_geometry(triangles, component, origin)
    indices = [i for face in faces for i in face]
    schema = (' (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"])'
              if collision else '')
    lines = ["", f'    def Mesh "{name}"{schema}', "    {",
             f"        point3f[] points = [{', '.join(fmt_vec(p) for p in points)}]",
             f"        int[] faceVertexIndices = [{', '.join(map(str, indices))}]",
             f"        int[] faceVertexCounts = [{', '.join(str(len(f)) for f in faces)}]",
             '        token subdivisionScheme = "none"',
             '        bool doubleSided = 1']
    if collision:
        lines += ['        bool physics:collisionEnabled = 1',
                  '        uniform token physics:approximation = "convexHull"',
                  '        token visibility = "invisible"']
    else:
        lines += [f"        color3f[] primvars:displayColor = [{fmt_vec(color)}]"]
    return lines + ["    }"]


def write_articulated_male_part(triangles, selected, output_path):
    """Write the male half with two independently compressible outer tongues."""

    groups = {"root": [], "neg": [], "pos": []}
    for component in selected:
        _lower, _upper, source_center, _extent = _source_component_bounds(
            triangles, component
        )
        # The source's three long tongues start at x=55.9 mm.  The centre
        # tongue stays on the housing; the +/-Y tongues (and their tiny latch
        # bumps) become the two moving links.
        if source_center[0] >= 66.0 and source_center[1] < -5.0:
            groups["neg"].append(component)
        elif source_center[0] >= 66.0 and source_center[1] > 5.0:
            groups["pos"].append(component)
        else:
            groups["root"].append(component)
    if len(groups["root"]) != 5 or len(groups["neg"]) != 2 or len(groups["pos"]) != 2:
        raise ValueError(
            "male articulation split changed: "
            f"root={len(groups['root'])}, neg={len(groups['neg'])}, pos={len(groups['pos'])}"
        )

    color = (0.95, 0.42, 0.12)
    tongue_origins = {}
    for key in ("neg", "pos"):
        prong = min(
            groups[key],
            key=lambda component: -_source_component_bounds(triangles, component)[3][1],
        )
        _lower, _upper, tongue_origins[key], _extent = component_bounds(
            triangles, prong, "male"
        )

    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "MaleBuckle"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "MaleBuckle" (',
        '    prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxContactReportAPI"]',
        ")",
        "{",
        "    float physics:mass = 0.075",
        "    float physxContactReport:threshold = 0",
        "    bool physics:rigidBodyEnabled = 1",
    ]
    lines.extend(_mesh_lines(triangles, groups["root"], "male", color, name="Visual"))
    for index, component in enumerate(groups["root"]):
        lines.extend(
            _collision_lines(
                triangles,
                component,
                "male",
                name=f"Collision_{index:02d}",
            )
        )

    for key, link_name, mass in (
        ("neg", "OuterTongueNeg", 0.0075),
        ("pos", "OuterTonguePos", 0.0075),
    ):
        origin = tongue_origins[key]
        lines.extend(
            [
                "",
                f'def Xform "{link_name}" (',
                '    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxContactReportAPI"]',
                "    )",
                "    {",
                f"        float physics:mass = {mass}",
                "        float physxContactReport:threshold = 0",
                "        bool physics:rigidBodyEnabled = 1",
                f"        double3 xformOp:translate = {fmt_vec(origin)}",
                '        uniform token[] xformOpOrder = ["!resetXformStack!", "xformOp:translate"]',
            ]
        )
        prong = min(
            groups[key],
            key=lambda component: -_source_component_bounds(triangles, component)[3][1],
        )
        lines.extend(_mesh_lines(triangles, [prong], "male", color,
                                 name="Visual", origin=origin))
        for component in groups[key]:
            if component is prong:
                lines.extend(_collision_lines(triangles, component, "male",
                                              name="Collision_Prong", origin=origin,
                                              inward_center=False))
            else:
                for name, collision in (("Visual_LatchBump", False),
                                        ("Collision_LatchBump", True)):
                    lines.extend(_latch_tooth_lines(triangles, component, origin,
                                                   name=name, collision=collision,
                                                   color=color))
        lines.extend(["    }"])

    for key, link_name in (("neg", "OuterTongueNeg"), ("pos", "OuterTonguePos")):
        origin = tongue_origins[key]
        # The normalization flips source Y: use transformed local positions,
        # not the source-space group name, to choose inward travel.
        lower, upper = (-0.004, 0.0) if origin[1] > 0.0 else (0.0, 0.004)
        lines.extend(
            [
                "",
                f'def PhysicsPrismaticJoint "{link_name}Joint" (',
                '    prepend apiSchemas = ["PhysicsDriveAPI:linear"]',
                "    )",
                "    {",
                "        float drive:linear:physics:damping = 0.45",
                "        float drive:linear:physics:maxForce = 6",
                "        float drive:linear:physics:stiffness = 35",
                '        uniform token drive:linear:physics:type = "force"',
                '        uniform token physics:axis = "Y"',
                f"        rel physics:body0 = </MaleBuckle>",
                f"        rel physics:body1 = </MaleBuckle/{link_name}>",
                "        bool physics:collisionEnabled = 0",
                "        bool physics:jointEnabled = 1",
                f"        point3f physics:localPos0 = {fmt_vec(origin)}",
                "        point3f physics:localPos1 = (0, 0, 0)",
                "        quatf physics:localRot0 = (1, 0, 0, 0)",
                "        quatf physics:localRot1 = (1, 0, 0, 0)",
                f"        float physics:lowerLimit = {lower}",
                f"        float physics:upperLimit = {upper}",
                "    }",
            ]
        )
    lines.extend(["}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_part(triangles, components, part, output_path):
    selected = []
    for component in components:
        xs = [vertex[0] for face in component for vertex in triangles[face]]
        belongs = max(xs) < 40.0 if part == "female" else min(xs) > 40.0
        if belongs:
            selected.append(component)
    expected_count = 6 if part == "female" else 9
    if len(selected) != expected_count:
        raise ValueError(f"{part} component count changed: {len(selected)} != {expected_count}")

    if part == "male":
        write_articulated_male_part(triangles, selected, output_path)
        return

    points = []
    indices = []
    for component in selected:
        for face in component:
            for vertex in triangles[face]:
                indices.append(len(points))
                points.append(transform(vertex, part))

    root_name = "FemaleBuckle" if part == "female" else "MaleBuckle"
    color = (0.12, 0.36, 0.80) if part == "female" else (0.95, 0.42, 0.12)
    lines = [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{root_name}"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        f'def Xform "{root_name}" (',
        '    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxContactReportAPI"]',
        ")",
        "{",
        f"    float physics:mass = {0.12 if part == 'female' else 0.09}",
        "    float physxContactReport:threshold = 0",
        "    bool physics:rigidBodyEnabled = 1",
        "",
        '    def Mesh "Visual"',
        "    {",
        f"        point3f[] points = [{', '.join(fmt_vec(point) for point in points)}]",
        f"        int[] faceVertexIndices = [{', '.join(str(index) for index in indices)}]",
        f"        int[] faceVertexCounts = [{', '.join('3' for _ in range(len(indices) // 3))}]",
        '        token subdivisionScheme = "none"',
        "        bool doubleSided = 1",
        f"        color3f[] primvars:displayColor = [{fmt_vec(color)}]",
        "    }",
    ]

    for index, component in enumerate(selected):
        _, _, center, extent = component_bounds(triangles, component, part)
        lines.extend(
            [
                "",
                f'    def Cube "Collision_{index:02d}" (',
                '        prepend apiSchemas = ["PhysicsCollisionAPI"]',
                "    )",
                "    {",
                "        bool physics:collisionEnabled = 1",
                "        double size = 1",
                f"        double3 xformOp:scale = {fmt_vec(extent)}",
                f"        double3 xformOp:translate = {fmt_vec(center)}",
                '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
                "        color3f[] primvars:displayColor = [(0.02, 0.02, 0.02)]",
                "        float[] primvars:displayOpacity = [0]",
                "        uniform token purpose = \"guide\"",
                '        token visibility = "invisible"',
                "    }",
            ]
        )
    lines.extend(["}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_stl", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if sha256(args.source_stl) != EXPECTED_SOURCE_SHA256:
        raise SystemExit("source STL SHA256 does not match the audited CC0 file")
    triangles = load_binary_stl(args.source_stl)
    if len(triangles) != 204:
        raise SystemExit(f"unexpected triangle count: {len(triangles)}")
    components = connected_components(triangles)
    if len(components) != 15:
        raise SystemExit(f"unexpected connected component count: {len(components)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_part(triangles, components, "female", args.output_dir / "female_buckle.usda")
    write_part(triangles, components, "male", args.output_dir / "male_buckle.usda")
    for path in sorted(args.output_dir.glob("*.usda")):
        print(f"{path.name} {sha256(path)}")


if __name__ == "__main__":
    main()
