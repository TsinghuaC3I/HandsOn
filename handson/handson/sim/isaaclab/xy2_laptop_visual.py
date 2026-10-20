"""Rebuild the existing XY2 laptop visuals from the user's IMG_1017 reference.

The photograph guides finish and layout, not model identity or calibration.
Every point remains in the unchanged source base/display collision envelope.
No source asset, rigid body, collider, joint, or task predicate is modified.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


BASE_VISUAL_PATH = "/World/Task/FU015Laptop/E_body_1/MacBookVisual"
DISPLAY_VISUAL_PATH = "/World/Task/FU015Laptop/E_displayer_5/MacBookVisual"
# Both existing source variants have these same measured rigid-local bounds.
BASE_LIMITS = ((-.1641000062, -.0020000001, -.0063279001),
               (.1641000062, .2278999984, .0041999999))
DISPLAY_LIMITS = ((-.0377000012, -.0031999999, -.0024000001),
                  (.2904999852, .0103000002, .2275000066))


def _outline(width, depth, radius, *, segments=12, notch_width=0., notch_depth=0.):
    hx, hy = width / 2, depth / 2
    radius = min(radius, hx, hy)
    points = []
    for corner, center in enumerate(((hx-radius, hy-radius), (-hx+radius, hy-radius),
                                     (-hx+radius, -hy+radius), (hx-radius, -hy+radius))):
        angles = np.linspace(corner * math.pi/2, (corner+1) * math.pi/2, segments+1)
        points.extend((center[0]+radius*math.cos(a), center[1]+radius*math.sin(a)) for a in angles)
        if corner == 2 and notch_width:
            for x in np.linspace(-notch_width/2, notch_width/2, 19):
                inset = notch_depth * math.cos(math.pi*x/notch_width)**2
                points.append((x, -hy+inset))
    return np.asarray(points, float)


def _rounded_prism(width, depth, height, radius, bevel, *, notch=0., segments=12, top_cap=True):
    """A smooth multi-ring bevel, with a shallow front-edge finger recess."""
    bevel = min(bevel, height * .45, radius * .7)
    angles = np.linspace(-math.pi/2, 0, 4).tolist() + np.linspace(0, math.pi/2, 4).tolist()
    points, normals, counts, indices = [], [], [], []
    n = None
    for ring_index, angle in enumerate(angles):
        z = (-height/2+bevel if ring_index < 4 else height/2-bevel) + bevel*math.sin(angle)
        inset = bevel * (1-math.cos(angle))
        contour = _outline(
            width-2*inset, depth-2*inset, max(.00005, radius-inset),
            segments=segments, notch_width=.060 if notch else 0.,
            notch_depth=notch*((z+height/2)/height)**4,
        )
        tangent = np.roll(contour, -1, axis=0) - np.roll(contour, 1, axis=0)
        outward = np.column_stack((tangent[:, 1], -tangent[:, 0]))
        outward /= np.linalg.norm(outward, axis=1)[:, None]
        points.extend(np.column_stack((contour, np.full(len(contour), z))))
        normals.extend(np.column_stack((outward*math.cos(angle), np.full(len(contour), math.sin(angle)))))
        n = len(contour)
        if ring_index:
            for i in range(n):
                counts.append(4)
                indices.extend(((ring_index-1)*n+i, (ring_index-1)*n+(i+1)%n,
                                ring_index*n+(i+1)%n, ring_index*n+i))
    # Separate cap vertices retain truly flat top/bottom normals.
    side_points = np.asarray(points)
    for top, offset in ((False, 0), (True, (len(angles)-1)*n)):
        if top and not top_cap:
            continue
        contour = side_points[offset:offset+n]
        start = len(points)
        points.extend(contour)
        normals.extend([(0, 0, 1 if top else -1)]*n)
        center = len(points)
        points.append((0., 0., contour[0, 2]))
        normals.append((0, 0, 1 if top else -1))
        for i in range(n):
            counts.append(3)
            edge = (start+i, start+(i+1)%n)
            indices.extend((center, *edge) if top else (center, *edge[::-1]))
    return np.asarray(points), counts, indices, np.asarray(normals)


def _plane_ring(outer, inner, z):
    assert len(outer) == len(inner)
    n = len(outer)
    points = np.column_stack((np.vstack((outer, inner)), np.full(2*n, z)))
    indices = [j for i in range(n) for j in (i, (i+1)%n, n+(i+1)%n, n+i)]
    return points, [4]*n, indices, np.tile((0., 0., 1.), (2*n, 1))


def _plane_fill(contour, z):
    """A flat convex face, without rounded side normals along its perimeter."""
    n = len(contour)
    points = np.column_stack((np.vstack((contour, np.mean(contour, axis=0))),
                              np.full(n+1, z)))
    indices = [j for i in range(n) for j in (n, i, (i+1)%n)]
    return points, [3]*n, indices, np.tile((0., 0., 1.), (n+1, 1))


def _map_geometry(geometry, center, display=False):
    points, counts, indices, normals = geometry
    points, normals = np.asarray(points).copy(), np.asarray(normals).copy()
    if display:
        points = points[:, (0, 2, 1)] * (1., -1., 1.)
        normals = normals[:, (0, 2, 1)] * (1., -1., 1.)
    return points + np.asarray(center), counts, indices, normals


def _mesh(stage, path, geometry, material):
    from pxr import UsdGeom, UsdShade, Vt

    points, counts, indices, normals = geometry
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(
        np.vstack((points.min(axis=0), points.max(axis=0))).astype(np.float32),
    ))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _material(stage, root, name, color, roughness, *, metallic=0., clearcoat=0., ior=1.46):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, root + "/Materials/" + name)
    shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    for key, value in (("roughness", roughness), ("metallic", metallic),
                       ("clearcoat", clearcoat), ("clearcoatRoughness", .09),
                       ("ior", ior), ("opacity", 1.)):
        shader.CreateInput(key, Sdf.ValueTypeNames.Float).Set(value)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


# Compact single-line glyph outlines, rendered as thin vector strokes.
# These remain legible on keycaps without raster texture blur or new assets.
_GLYPHS = {
    "A": [[(0,0),(.5,1),(1,0)],[(.2,.4),(.8,.4)]],
    "B": [[(0,0),(0,1),(.7,1),(1,.8),(.7,.55),(0,.55)],
          [(.7,.55),(1,.25),(.7,0),(0,0)]],
    "C": [[(1,.85),(.75,1),(.2,1),(0,.75),(0,.2),(.2,0),(.8,0),(1,.15)]],
    "D": [[(0,0),(0,1),(.6,1),(1,.7),(1,.3),(.6,0),(0,0)]],
    "E": [[(1,1),(0,1),(0,0),(1,0)],[(0,.5),(.8,.5)]],
    "F": [[(0,0),(0,1),(1,1)],[(0,.5),(.8,.5)]],
    "G": [[(1,.85),(.75,1),(.2,1),(0,.75),(0,.2),(.2,0),(1,0),(1,.45),(.55,.45)]],
    "H": [[(0,0),(0,1)],[(1,0),(1,1)],[(0,.5),(1,.5)]],
    "I": [[(.1,1),(.9,1)],[(.5,1),(.5,0)],[(.1,0),(.9,0)]],
    "J": [[(1,1),(1,.2),(.8,0),(.2,0),(0,.2)]],
    "K": [[(0,0),(0,1)],[(1,1),(0,.45),(1,0)]],
    "L": [[(0,1),(0,0),(1,0)]],
    "M": [[(0,0),(0,1),(.5,.45),(1,1),(1,0)]],
    "N": [[(0,0),(0,1),(1,0),(1,1)]],
    "O": [[(.2,0),(0,.2),(0,.8),(.2,1),(.8,1),(1,.8),(1,.2),(.8,0),(.2,0)]],
    "P": [[(0,0),(0,1),(.8,1),(1,.8),(1,.6),(.8,.5),(0,.5)]],
    "Q": [[(.2,0),(0,.2),(0,.8),(.2,1),(.8,1),(1,.8),(1,.2),(.8,0),(.2,0)],
          [(.6,.3),(1,-.05)]],
    "R": [[(0,0),(0,1),(.8,1),(1,.8),(1,.6),(.8,.5),(0,.5)],[(.5,.5),(1,0)]],
    "S": [[(1,.85),(.8,1),(.2,1),(0,.8),(.1,.6),(.9,.4),(1,.2),(.8,0),(.2,0),(0,.15)]],
    "T": [[(0,1),(1,1)],[(.5,1),(.5,0)]],
    "U": [[(0,1),(0,.2),(.2,0),(.8,0),(1,.2),(1,1)]],
    "V": [[(0,1),(.5,0),(1,1)]],
    "W": [[(0,1),(.2,0),(.5,.55),(.8,0),(1,1)]],
    "X": [[(0,1),(1,0)],[(0,0),(1,1)]],
    "Y": [[(0,1),(.5,.5),(1,1)],[(.5,.5),(.5,0)]],
    "Z": [[(0,1),(1,1),(0,0),(1,0)]],
    "0": [[(.2,0),(0,.2),(0,.8),(.2,1),(.8,1),(1,.8),(1,.2),(.8,0),(.2,0)]],
    "1": [[(.2,.75),(.5,1),(.5,0)],[(.2,0),(.8,0)]],
    "2": [[(0,.8),(.2,1),(.8,1),(1,.8),(1,.65),(0,0),(1,0)]],
    "3": [[(0,1),(.8,1),(1,.75),(.6,.5),(1,.25),(.8,0),(0,0)],[(.2,.5),(.6,.5)]],
    "4": [[(.8,0),(.8,1),(0,.3),(1,.3)]],
    "5": [[(1,1),(0,1),(0,.55),(.75,.55),(1,.35),(1,.2),(.8,0),(0,0)]],
    "6": [[(1,1),(.3,1),(0,.65),(0,.2),(.2,0),(.8,0),(1,.2),(1,.45),(.8,.6),(0,.6)]],
    "7": [[(0,1),(1,1),(.25,0)]],
    "8": [[(.2,.5),(0,.7),(0,.85),(.2,1),(.8,1),(1,.85),(1,.7),(.8,.5),(.2,.5),
          (0,.3),(0,.15),(.2,0),(.8,0),(1,.15),(1,.3),(.8,.5)]],
    "9": [[(1,.4),(.2,.4),(0,.6),(0,.8),(.2,1),(.8,1),(1,.8),(1,.3),(.7,0),(0,0)]],
    "-": [[(.1,.5),(.9,.5)]], "=": [[(.1,.65),(.9,.65)],[(.1,.35),(.9,.35)]],
    "/": [[(0,0),(1,1)]], "\\": [[(0,1),(1,0)]],
    "[": [[(.8,1),(.2,1),(.2,0),(.8,0)]], "]": [[(.2,1),(.8,1),(.8,0),(.2,0)]],
    ";": [[(.5,.75),(.5,.72)],[(.5,.15),(.3,-.05)]],
    "'": [[(.5,1),(.4,.8)]], ",": [[(.5,.15),(.3,-.05)]],
    ".": [[(.45,.05),(.55,.05)]], "~": [[(0,.45),(.3,.6),(.7,.4),(1,.55)]],
    "<": [[(.8,.85),(.2,.5),(.8,.15)]], ">": [[(.2,.85),(.8,.5),(.2,.15)]],
    "^": [[(.15,.3),(.5,.75),(.85,.3)]],
}


def _legend_geometry(keys):
    points, counts, indices = [], [], []
    for label, cx, cy, height in keys:
        glyph_width = height*.58
        spacing = height*.80
        start = cx - ((len(label)-1)*spacing+glyph_width)/2
        for letter_index, char in enumerate(label):
            for line in _GLYPHS.get(char, []):
                line = np.asarray(line)*[glyph_width, height] + [start+letter_index*spacing, cy-height/2]
                for a, b in zip(line, line[1:]):
                    angle = math.atan2(*(b-a)[::-1])
                    radius = height*.047
                    polygon = np.vstack((
                        a + radius*np.column_stack((np.cos(np.linspace(angle+math.pi/2, angle+3*math.pi/2, 5)),
                                                     np.sin(np.linspace(angle+math.pi/2, angle+3*math.pi/2, 5)))),
                        b + radius*np.column_stack((np.cos(np.linspace(angle-math.pi/2, angle+math.pi/2, 5)),
                                                     np.sin(np.linspace(angle-math.pi/2, angle+math.pi/2, 5)))),
                    ))
                    offset = len(points)
                    points.extend(np.column_stack((polygon, np.full(len(polygon), .00410))))
                    counts.append(len(polygon))
                    indices.extend(range(offset, offset+len(polygon)))
    return np.asarray(points), counts, indices, np.tile((0., 0., 1.), (len(points), 1))


def _add_base(stage, root):
    silver = _material(stage, root, "SatinAluminum", (.46, .48, .51), .29, metallic=.82)
    edge = _material(stage, root, "MachinedEdge", (.52, .54, .57), .21, metallic=.88)
    dark = _material(stage, root, "KeyboardRecess", (.009, .011, .014), .46)
    keys_material = _material(stage, root, "Keycaps", (.014, .017, .021), .37)
    legends_material = _material(stage, root, "KeyLegends", (.49, .51, .54), .50)
    seam = _material(stage, root, "TrackpadSeam", (.045, .050, .058), .38, metallic=.20)

    def box(name, center, size, radius, bevel, material, **kwargs):
        return _mesh(stage, root+"/"+name, _map_geometry(
            _rounded_prism(*size, radius, bevel, **kwargs), center), material)

    # Thin rounded unibody, with the upper front contour scooped for a finger.
    # The deck annulus supplies the top face, leaving a real visual keyboard
    # recess rather than drawing a black rectangle over a closed silver box.
    box("LowerUnibody", (0., .11295, -.00149), (.3276, .2290, .00902),
        .009, .00065, silver, notch=.0019, top_cap=False)
    for index, (x, y) in enumerate(((-.141, .018), (.141, .018), (-.141, .207), (.141, .207))):
        box(f"Feet/Foot_{index}", (x, y, -.00608), (.012, .012, .00035),
            .006, .000075, dark)
    outer = _outline(.3263, .2277, .00835, notch_width=.060, notch_depth=.0019)
    inner = _outline(.3050, .1148, .0042, notch_width=.060)
    outer += (0., .11295)
    inner += (0., .1568)
    _mesh(stage, root+"/TopDeck", _plane_ring(outer, inner, .00302), silver)
    box("KeyboardWell", (0., .1568, .00278), (.3052, .1150, .00012), .0043, .000025, dark)
    # The reference trackpad shares the palm rest's finish and plane. Let the
    # uninterrupted deck supply its surface; only a 0.18 mm seam is overlaid.
    # This avoids a raised panel with a different metal/roughness response.
    trackpad_outer = _outline(.1468, .0760, .0042, segments=16) + (0., .0518)
    trackpad_inner = _outline(.14644, .07564, .00402, segments=16) + (0., .0518)
    _mesh(stage, root+"/TrackpadSeam", _plane_ring(
        trackpad_outer, trackpad_inner, .003028), seam)
    box("HingeTrim", (0., .2203, .0021), (.278, .0048, .0014), .0020, .00025, dark)
    # A thin front seam and bevel are visible without extending the body.
    box("FrontEdgeHighlight", (0., .0018, -.00020), (.305, .00035, .00032), .00015, .00008, edge)

    rows = [
        ([("ESC",1.3), *[(f"F{i}",1.) for i in range(1,13)], ("",1.1)], .2039, .0107),
        ([(c,1.) for c in "~1234567890-="]+[("<",1.5)], .1870, .0162),
        ([("TAB",1.5)]+[(c,1.) for c in "QWERTYUIOP[]"]+[("\\",1.)], .1680, .0162),
        ([("CAPS",1.75)]+[(c,1.) for c in "ASDFGHJKL;'"]+[("RETURN",1.75)], .1490, .0162),
        ([("SHIFT",2.25)]+[(c,1.) for c in "ZXCVBNM,./"]+[("SHIFT",2.25)], .1300, .0162),
        ([("FN",1.),("CTRL",1.),("OPT",1.),("CMD",1.25),("",4.75),
          ("CMD",1.25),("OPT",1.),("<",1.),("^",1.),(">",1.)], .1110, .0162),
    ]
    legends = []
    key_count = 0
    for row_index, (row, y, key_depth) in enumerate(rows):
        unit = .300 / sum(units for _, units in row)
        x = -.150
        for key_index, (label, units) in enumerate(row):
            width = unit*units
            center_x = x+width/2
            key_parts = [(label, y, key_depth)]
            if row_index == 5 and label in ("<", "^", ">"):
                key_parts = [(label, .1069, .0076)] if label != "^" else [
                    ("^", .1154, .0065), ("V", .1069, .0076),
                ]
            for part_index, (part_label, part_y, part_depth) in enumerate(key_parts):
                box(f"Keyboard/Key_{row_index}_{key_index}_{part_index}", (center_x, part_y, .00345),
                    (width-.00155, part_depth, .00120), .00155, .00024, keys_material,
                    segments=8)
                if part_label:
                    legend_height = .0029 if len(part_label) == 1 else .00165
                    legends.append((part_label, center_x, part_y, legend_height))
                key_count += 1
            if row_index == 0 and not label:
                box("Keyboard/FingerprintSensor", (center_x, y, .00407),
                    (.0075, .0075, .00003), .00375, .000007, dark, segments=16)
            x += width
    _mesh(stage, root+"/Keyboard/Legends", _legend_geometry(legends), legends_material)
    return key_count


def _add_display(stage, root):
    silver = _material(stage, root, "LidAluminum", (.46, .48, .51), .28, metallic=.82)
    # One flat cover-glass response across the border and powered-off panel.
    # Different lobes on two stacked boxes made the inner edge look recessed.
    black = _material(stage, root, "BlackGlassBorder", (.0040, .0050, .0065), .14,
                      clearcoat=.25, ior=1.50)
    glass = _material(stage, root, "PoweredOffGlass", (.0025, .0035, .0050), .14,
                      clearcoat=.25, ior=1.50)
    camera = _material(stage, root, "CameraGlass", (.014, .026, .035), .10, clearcoat=.8, ior=1.52)
    lens = _material(stage, root, "LensAperture", (.001, .0015, .002), .16)

    def box(name, center, size, radius, bevel, material, segments=16):
        return _mesh(stage, root+"/"+name, _map_geometry(
            _rounded_prism(*size, radius, bevel, segments=segments), center, True), material)

    # Display local X/Z are its plane; the user-facing glass points toward -Y.
    box("RoundedSilverLid", (.1264, .00825, .11255), (.3274, .2286, .0037),
        .0092, .00065, silver)
    # Retain only the outside glass edge's tiny bevel. The central panel and
    # border are adjacent faces at exactly the same Y, with no inner sidewall.
    glass_center = (.1264, .00628, .11255)
    _mesh(stage, root+"/GlassOuterEdge", _map_geometry(
        _rounded_prism(.3258, .2270, .00024, .0084, .00005,
                       segments=16, top_cap=False), glass_center, True), black)
    outer = _outline(.3257, .2269, .00835, segments=16)
    panel = _outline(.3015, .1920, .0030, segments=16) + (0., .00545)
    _mesh(stage, root+"/BlackBezelGlass", _map_geometry(
        _plane_ring(outer, panel, .00012), glass_center, True), black)
    _mesh(stage, root+"/PoweredOffScreen", _map_geometry(
        _plane_fill(panel, .00012), glass_center, True), glass)
    box("CameraSurround", (.1264, .00609, .2188), (.0034, .0034, .00012),
        .0017, .000025, black)
    box("CameraLens", (.1264, .00599, .2188), (.00165, .00165, .00007),
        .000825, .000015, camera)
    box("CameraPupil", (.1264, .005935, .2188), (.00072, .00072, .000035),
        .00036, .000008, lens)


def _visual_bounds(stage, root):
    from pxr import Usd, UsdGeom

    arrays = [np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
              for prim in Usd.PrimRange(stage.GetPrimAtPath(root)) if prim.IsA(UsdGeom.Mesh)]
    points = np.vstack(arrays)
    if not np.isfinite(points).all():
        raise RuntimeError("Nonfinite laptop visual points")
    return [points.min(axis=0).tolist(), points.max(axis=0).tolist()], len(points), len(arrays)


def rebuild_xy2_laptop_visual(
    stage: Any, *, base_visual_path: str = BASE_VISUAL_PATH,
    display_visual_path: str = DISPLAY_VISUAL_PATH,
) -> dict[str, Any]:
    """Replace only the contents of the two existing MacBookVisual roots."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    from .blender_visual_layer import physics_signature

    roots = [stage.GetPrimAtPath(path) for path in (base_visual_path, display_visual_path)]
    for prim in roots:
        if not prim or not prim.GetParent().HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("An existing laptop visual root under a rigid link is required")
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if any("Physics" in schema or "Physx" in schema for schema in child.GetAppliedSchemas()):
                raise ValueError("Refusing to replace a visual subtree containing physics")
    before = physics_signature(stage)
    for prim in roots:
        prim.SetInstanceable(False)
        prim.GetReferences().SetReferences([])
        for child in list(prim.GetChildren()):
            stage.RemovePrim(child.GetPath())
        UsdShade.MaterialBindingAPI(prim).UnbindAllBindings()
        UsdGeom.Xformable(prim).ClearXformOpOrder()
        UsdGeom.Imageable(prim).MakeVisible()
    key_count = _add_base(stage, base_visual_path)
    _add_display(stage, display_visual_path)
    report = {"reference": "IMG_1017", "physical_model_preserved": True,
              "keyboard_keys": key_count, "screen": "powered_off", "bodies": {}}
    for label, path, limits in (("base", base_visual_path, BASE_LIMITS),
                                ("display", display_visual_path, DISPLAY_LIMITS)):
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if prim.IsA(UsdGeom.Mesh):
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                if not material or not str(material.GetPath()).startswith(path+"/Materials/"):
                    raise RuntimeError("Laptop visual material was overridden: " + str(prim.GetPath()))
        bounds, vertices, meshes = _visual_bounds(stage, path)
        if (np.any(np.asarray(bounds[0]) < np.asarray(limits[0])-1e-7)
                or np.any(np.asarray(bounds[1]) > np.asarray(limits[1])+1e-7)):
            raise RuntimeError(f"New {label} visuals exceed the unchanged source collision envelope")
        report["bodies"][label] = {"bounds_local_m": bounds, "vertices": vertices, "meshes": meshes}
    after = physics_signature(stage)
    if after != before:
        raise RuntimeError("Laptop visual rebuild changed the physical model")
    report.update(physics_before=before, physics_after=after)
    return report
