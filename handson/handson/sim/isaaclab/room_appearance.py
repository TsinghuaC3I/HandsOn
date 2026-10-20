"""Native, deterministic PBR laboratory appearance; no physical schemas."""
import hashlib
import itertools
import json
from pathlib import Path
import random

import numpy as np

ROOT = "/World/HandsOnAppearance/Room"
VERSION = "handson_pbr_lab"


def state_signature(stage, bindings):
    """Authored appearance state only; excludes all task and robot transforms."""
    from pxr import Usd
    rows = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROOT)):
        attrs = [(a.GetName(), str(a.Get())) for a in prim.GetAuthoredAttributes()]
        rels = [(r.GetName(), [str(t) for t in r.GetTargets()]) for r in prim.GetAuthoredRelationships()]
        rows.append((str(prim.GetPath()), prim.GetTypeName(), attrs, rels))
    return hashlib.sha256(json.dumps([rows, bindings], sort_keys=True).encode()).hexdigest()


def verify(stage, receipt):
    from pxr import UsdShade
    import carb.settings
    for target, expected in receipt["bindings"].items():
        actual = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(target)).ComputeBoundMaterial()[0]
        assert actual and str(actual.GetPath()) == expected, target
    signature = state_signature(stage, receipt["bindings"])
    assert signature == receipt["state_signature"], "Appearance changed within the trajectory"
    for path in receipt["inherited_lights_disabled"]:
        assert stage.GetPrimAtPath(path).GetAttribute("inputs:intensity").Get() == 0.
    settings = carb.settings.get_settings()
    assert settings.get("/rtx/rendermode") == "PathTracing"
    assert settings.get("/rtx/pathtracing/spp") == receipt["profile"]["spp"]
    assert settings.get("/rtx/pathtracing/totalSpp") == receipt["profile"]["spp"]
    return {"state_signature": signature, "bindings_unchanged": True, "pathtracing_spp": receipt["profile"]["spp"]}


def resolve(profile):
    """One coherent appearance per episode; stable across all render frames."""
    seed = int(profile.get("seed", 0))
    task = str(profile.get("task", ""))
    digest = hashlib.sha256(f"{VERSION}|{task}|{seed}".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    randomized = profile.get("mode", "fixed") == "randomized"
    material = profile.get("table", rng.choice(["Plastic010", "Wood095", "Wood049", "Wood051"])
                           if randomized else "Wood095")
    palette = rng.choice(["neutral", "slate", "blue"]) if randomized else "neutral"
    result = dict(
        version=VERSION, mode=profile.get("mode", "fixed"), seed=seed,
        task=task, seed_sha256=digest, table=material,
        table_tile_m=(rng.uniform(.72, 1.12) if material.startswith("Wood") else rng.uniform(.22, .32))
                     if randomized else (.8 if material.startswith("Wood") else .26),
        table_rotate_deg=rng.choice([0., 90.]) if randomized else 90.,
        table_offset=[rng.random(), rng.random()] if randomized else [0.17, .31],
        table_roughness=rng.uniform(.36, .55) if randomized else .44,
        table_normal_strength=.30 if material.startswith("Wood") else .16,
        palette=palette,
        cabinet_color={"neutral": [.36, .43, .46], "slate": [.21, .25, .29],
                       "blue": [.15, .27, .36]}[palette],
        wall_color={"neutral": [.985, .985, .975], "slate": [.955, .975, 1.],
                    "blue": [1., .975, .94]}[palette],
        floor_tint={"neutral": [.80, .84, .86], "slate": [.70, .75, .81],
                    "blue": [.85, .83, .78]}[palette],
        floor_tile_m=rng.uniform(2.0, 2.8) if randomized else 2.4,
        floor_roughness=rng.uniform(.56, .76) if randomized else .68,
        wall_tile_m=rng.uniform(1.2, 1.8) if randomized else 1.5,
        cabinet_x=rng.uniform(1.30, 1.54) if randomized else 1.40,
        context_variant=rng.randrange(3) if randomized else 0,
        key_position=[rng.uniform(.1, .5), rng.choice([-1, 1])*rng.uniform(1.3, 1.8),
                      rng.uniform(2.3, 2.7)] if randomized else [.3, 1.6, 2.5],
        key_intensity=rng.uniform(1500., 2050.) if randomized else 1800.,
        key_temperature=rng.uniform(5200., 6200.) if randomized else 5700.,
        key_size=[1.2, .85], fill_intensity=700., rim_intensity=550.,
        dome_intensity=rng.uniform(85., 120.) if randomized else 100.,
        spp=int(profile.get("spp", 128)), exposure_multiplier=3.0,
        context=True,
        ceiling_height=rng.uniform(2.95, 3.20) if randomized else 3.05,
        ceiling_variant=rng.randrange(3) if randomized else 1,
        ceiling_color=rng.choice([[1., 1., .985], [1., .98, .955], [.97, .985, 1.]])
                      if randomized else [1., 1., .985],
        ceiling_light_intensity=rng.uniform(180., 260.) if randomized else 220.,
        ceiling_light_temperature=rng.uniform(4800., 6200.) if randomized else 5600.,
        wall_roughness=rng.uniform(.76, .88) if randomized else .82,
        wall_normal_strength=rng.uniform(.04, .09) if randomized else .06,
        ceiling_roughness=rng.uniform(.80, .91) if randomized else .86,
        ceiling_normal_strength=rng.uniform(.02, .055) if randomized else .035,
        architectural_palette="predominantly_white",
    )
    for key in result:
        if key in profile:
            result[key] = profile[key]
    assert result["table"] in ("Plastic010", "Wood095", "Wood049", "Wood051")
    assert 0.1 <= result["table_tile_m"] <= 3
    assert 0.2 <= result["table_roughness"] <= .85
    assert 16 <= result["spp"] <= 1024
    return result


def apply(stage, cameras, profile, assets, *, table_required=True):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdLux, UsdPhysics
    import carb.settings
    from scipy.spatial import ConvexHull
    cfg = resolve(profile)
    assets = Path(assets).resolve()
    manifest = json.loads((assets/"ASSETS.json").read_text())
    asset_rows = {r["asset_id"]: r for r in manifest["materials"]}
    # This namespace is created only by this module; it has no physical APIs.
    if stage.GetPrimAtPath(ROOT):
        stage.RemovePrim(ROOT)
    UsdGeom.Xform.Define(stage, ROOT)
    bindings = {}
    texture_receipts = []

    def material(name, color, roughness=.5, metallic=0., texture=None,
                 tile_m=1., normal=.25, rotate=0., offset=(0., 0.), desaturate=0.):
        mat = UsdShade.Material.Define(stage, ROOT+"/Materials/"+name)
        sh = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Shader"))
        sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        values = {
            "diffuse_color_constant": (Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*color)),
            "reflection_roughness_constant": (Sdf.ValueTypeNames.Float, float(roughness)),
            "metallic_constant": (Sdf.ValueTypeNames.Float, float(metallic)),
            "specular_level": (Sdf.ValueTypeNames.Float, .5),
        }
        if texture:
            row = asset_rows[texture]
            # Rectangular maps retain their texel aspect. Wood095 is 2048x1024.
            from PIL import Image
            with Image.open(assets/row["maps"]["Color"]["file"]) as image:
                ratio = image.width/image.height
            for channel, slot in [("Color", "diffuse_texture"), ("NormalGL", "normalmap_texture"),
                                  ("Roughness", "reflectionroughness_texture")]:
                path = assets/row["maps"][channel]["file"]
                assert path.is_file()
                assert hashlib.sha256(path.read_bytes()).hexdigest() == row["maps"][channel]["sha256"]
                values[slot] = (Sdf.ValueTypeNames.Asset, Sdf.AssetPath(str(path)))
            values.update({
                "project_uvw": (Sdf.ValueTypeNames.Bool, True),
                "world_or_object": (Sdf.ValueTypeNames.Bool, True),
                "texture_scale": (Sdf.ValueTypeNames.Float2, Gf.Vec2f(1./tile_m, ratio/tile_m)),
                "texture_rotate": (Sdf.ValueTypeNames.Float, float(rotate)),
                "texture_translate": (Sdf.ValueTypeNames.Float2, Gf.Vec2f(*offset)),
                "diffuse_tint": (Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*color)),
                "albedo_desaturation": (Sdf.ValueTypeNames.Float, float(desaturate)),
                "albedo_brightness": (Sdf.ValueTypeNames.Float, 1.5 if texture == "Plastic010" else 1.0),
                "reflection_roughness_texture_influence": (Sdf.ValueTypeNames.Float, .55),
                "bump_factor": (Sdf.ValueTypeNames.Float, float(normal)),
                "flip_tangent_v": (Sdf.ValueTypeNames.Bool, False),
            })
            texture_receipts.append(dict(material=str(mat.GetPath()), asset=texture,
                tile_m=[tile_m, tile_m/ratio], world_projection=True, normal_convention="OpenGL",
                normal_strength=normal, rotate=rotate, offset=list(offset), maps=row["maps"]))
        for key, (kind, value) in values.items():
            sh.CreateInput(key, kind).Set(value)
        sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        return mat

    def bind(prim, mat):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        for _ in range(64):
            actual, relation = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if actual and actual.GetPath() == mat.GetPath():
                bindings[str(prim.GetPath())] = str(mat.GetPath())
                return
            if relation and relation.GetPrim() == prim:
                # Recorded source releases can bind this same surface in the
                # stronger session layer. Override only its visual relationship.
                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                        mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
                actual = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
                assert actual and actual.GetPath() == mat.GetPath(), (
                    str(prim.GetPath()), str(relation.GetPath()), "visual binding unresolved")
                bindings[str(prim.GetPath())] = str(mat.GetPath())
                return
            assert relation, ("No effective visual binding", str(prim.GetPath()))
            UsdShade.MaterialBindingAPI.SetMaterialBindingStrength(relation, UsdShade.Tokens.weakerThanDescendants)
        raise RuntimeError("Could not bind material to "+str(prim.GetPath()))

    table_mat = material("Table", [.90, .93, .94] if cfg["table"] == "Plastic010" else [1., 1., 1.],
        cfg["table_roughness"], texture=cfg["table"], tile_m=cfg["table_tile_m"],
        normal=cfg["table_normal_strength"], rotate=cfg["table_rotate_deg"],
        offset=cfg["table_offset"])
    floor_mat = material("Floor", cfg["floor_tint"], cfg["floor_roughness"], texture="Concrete048",
                         tile_m=cfg["floor_tile_m"], normal=.25, desaturate=.9)
    wall_mat = material("Wall", cfg["wall_color"], cfg["wall_roughness"], texture="Plaster004WhitePaint",
                        tile_m=cfg["wall_tile_m"], normal=cfg["wall_normal_strength"])
    paint = material("CabinetPaint", cfg["cabinet_color"], .38, .15)
    counter = material("Countertop", [.31, .36, .40], .48, texture="Concrete048",
                       tile_m=1.1, normal=.12, desaturate=.9)
    steel = material("BrushedSteel", [.42, .46, .50], .27, .92)
    charcoal = material("Charcoal", [.045, .055, .066], .43)
    white = material("InstrumentWhite", [.68, .70, .70], .32)
    screen = material("Screen", [.012, .033, .045], .18, .35)
    accent = material("Accent", [.12, .26, .32], .44)
    ceiling_mat = material("Ceiling", cfg["ceiling_color"], cfg["ceiling_roughness"], texture="Plaster004WhitePaint",
                           tile_m=1.6, normal=cfg["ceiling_normal_strength"])
    diffuser = material("CeilingDiffuser", [.83, .85, .84], .5)
    diffuser_shader = UsdShade.Shader.Get(stage, diffuser.GetPath().AppendChild("Shader"))
    diffuser_shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
    diffuser_shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.90, .94, 1.))
    diffuser_shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(8.)

    table_targets = []
    floor_targets = []
    for prim in list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())):
        path = str(prim.GetPath())
        if path.startswith(ROOT) or prim.IsInstanceProxy():
            continue
        gprim = prim.IsA(UsdGeom.Gprim)
        visible = (bool(UsdGeom.Imageable(prim))
                   and UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible)
        standard_table = gprim and (
            prim.GetName() == "table_visual"
            or (path.startswith("/World/envs/")
                and (prim.GetName() == "Table" or "/Table/" in path)
                and "collision" not in path.lower()))
        b320_table = path == "/World/Task/B320Benchmark/worktable/geometry/mesh"
        g02_table = path == "/World/Task/G02RepairCloseDevice/workbench/geometry/mesh"
        lw_table = path == '/World/Task/LWNativeArticulation/table/geometry/mesh'
        hs005_counter = (
            path.startswith(("/World/Task/HS005Assets/",
                             "/World/Task/B320Benchmark/faucet_fixture/hs005_realistic_kitchen/"))
            and path.endswith("/KitchenBackground/Visual/Plane_007_Wood049_0/Plane_007_Wood049_0"))
        kitchen_island = (
            path.startswith("/World/VisualEnvironment/BC014Kitchen/Asset/")
            and path.endswith("/RootNode/Island/Island_Countertop_0/Island_Countertop_0"))
        if gprim and visible and (standard_table or b320_table or g02_table or lw_table
                                   or hs005_counter or kitchen_island):
            # Bind the rendered leaf. A source's stronger session-layer child
            # binding can override an apparently successful parent assignment.
            bind(prim, table_mat)
            table_targets.append(path)
        if prim.GetName().lower() in ("floor", "ground", "background") and (
                gprim or prim.IsInstance() or path == "/World/Ground"):
            bind(prim, floor_mat)
            floor_targets.append(path)
    if table_required:
        assert table_targets, "No known visible native tabletop mesh; do not skip PBR"

    def box(name, position, dimensions, mat, bevel=.003):
        """Small render-only chamfers catch light without changing collisions."""
        path = ROOT+"/Environment/"+name
        dims = np.array(dimensions, dtype=float)
        b = min(float(bevel), float(dims.min())*.2)
        vertices = []
        for axis in range(3):
            other = [i for i in range(3) if i != axis]
            for sign in (-1., 1.):
                for signs in itertools.product((-1., 1.), repeat=2):
                    point = np.zeros(3)
                    point[axis] = sign*dims[axis]*.5
                    for i, side in zip(other, signs):
                        point[i] = side*(dims[i]*.5-b)
                    vertices.append(point)
        vertices = np.array(vertices)
        hull = ConvexHull(vertices)
        faces = hull.simplices.copy()
        for i, face in enumerate(faces):
            cross = np.cross(vertices[face[1]]-vertices[face[0]], vertices[face[2]]-vertices[face[0]])
            if np.dot(cross, hull.equations[i, :3]) < 0:
                faces[i] = face[::-1]
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(vertices.tolist())
        mesh.CreateFaceVertexCountsAttr([3]*len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(*position))
        bind(mesh.GetPrim(), mat)
        return mesh.GetPrim()

    if cfg["context"]:
        # Cabinet front is beyond x=1.0 m, outside the supported task bench.
        x = float(cfg["cabinet_x"])
        height = float(cfg["ceiling_height"])
        # Enclose every native wrist-camera direction, including behind the
        # robot and overhead. Geometry is visual-only and outside the task.
        box("RearWall", [2.65, 0., height*.5], [.10, 6.20, height+.1], wall_mat)
        box("RobotBackWall", [-2.95, 0., height*.5], [.10, 6.20, height+.1], wall_mat)
        box("LeftWall", [-.15, 3.05, height*.5], [5.70, .1, height+.1], wall_mat)
        box("RightWall", [-.15, -3.05, height*.5], [5.70, .1, height+.1], wall_mat)
        box("Ceiling", [-.15, 0., height+.055], [5.70, 6.20, .11], ceiling_mat)
        if cfg["ceiling_variant"] == 1:
            for i, x_panel in enumerate([-2.35, -1.25, -.15, .95, 2.05]):
                box(f"CeilingRailX_{i}", [x_panel, 0., height-.006], [.014, 6., .016], white, .001)
            for i, y_panel in enumerate([-2.4, -1.2, 0., 1.2, 2.4]):
                box(f"CeilingRailY_{i}", [-.15, y_panel, height-.006], [5.5, .014, .016], white, .001)
        elif cfg["ceiling_variant"] == 2:
            for i, y_panel in enumerate(np.linspace(-2.8, 2.8, 15)):
                box(f"CeilingSeam_{i}", [-.15, float(y_panel), height-.006],
                    [5.5, .006, .01], paint, .001)
        for i, (lamp_x, lamp_y) in enumerate([(-1.25, -.72), (.95, .72)]):
            box(f"CeilingLampFrame_{i}", [lamp_x, lamp_y, height-.045], [.68, 1.02, .08], white)
            box(f"CeilingLampDiffuser_{i}", [lamp_x, lamp_y, height-.089],
                [.62, .96, .012], diffuser, .001)
        box("BackDoor", [-2.891, -1.35, 1.06], [.018, .94, 2.12], paint)
        box("BackDoorHandle", [-2.87, -1.01, 1.02], [.035, .13, .02], steel, .002)
        box("BackSkirting", [-2.89, 0., .055], [.018, 6., .11], charcoal, .001)
        box("RearSkirting", [2.58, 0., .055], [.018, 6., .11], charcoal, .001)
        box("BenchCounter", [x-.015, 0., .573], [.60, 2.70, .036], counter, .005)
        if cfg["context_variant"] == 1:
            for i, (dx, y) in enumerate(itertools.product([-.22, .22], [-1.22, 1.22])):
                box(f"BenchLeg_{i}", [x+dx, y, .28], [.045, .045, .56], steel, .002)
            box("LowShelf", [x, 0., .14], [.5, 2.5, .03], steel)
            for i, y in enumerate([-.85, 0., .85]):
                box(f"LowerBin_{i}", [x, y, .255], [.38, .58, .20], paint)
                box(f"BinInset_{i}", [x-.194, y, .275], [.008, .22, .022], charcoal, .001)
        else:
            box("BenchPlinth", [x, 0., .07], [.49, 2.54, .14], charcoal)
            box("BenchCarcass", [x, 0., .315], [.55, 2.64, .48], paint)
            for i, y in enumerate([-.99, -.33, .33, .99]):
                levels = [(.30, .38)] if cfg["context_variant"] == 0 else [(.20, .18), (.40, .18)]
                for j, (z, h) in enumerate(levels):
                    box(f"Door_{i}_{j}", [x-.286, y, z], [.018, .642, h], paint, .002)
                    box(f"Handle_{i}_{j}", [x-.315, y, z+h*.5-.045], [.032, .22, .012], steel, .002)
        box("WallCableRail", [2.58, 0., 1.05], [.055, 2.7, .075], white, .002)
        for i, y in enumerate([-1.06, -.42, .42, 1.06]):
            box(f"Outlet_{i}", [2.545, y, 1.05], [.014, .06, .055], charcoal, .002)
        side = [-1., 1., -1.][cfg["context_variant"]]
        # Equipment rests on the background workbench, never the task tabletop.
        for i in range(2):
            y = side*(.65+.26*i)
            box(f"Instrument_{i}", [x-.025, y, .65], [.25, .22, .12], white, .005)
            box(f"InstrumentDisplay_{i}", [x-.154, y-.035, .66], [.004, .105, .045], screen, .001)
            box(f"InstrumentButton_{i}", [x-.158, y+.07, .645], [.009, .022, .022], charcoal, .002)
        box("Tray", [x-.03, -side*.8, .603], [.30, .38, .025], steel, .003)
        for i in range(3):
            box(f"StorageCase_{i}", [x, -side*(.66+i*.13), .643],
                [.22, .105, .06], accent if i == 1 else charcoal, .004)
        box("FloorJointX", [1.15, 0., .0004], [.003, 6., .0006], charcoal, .0001)
        for i, y in enumerate([-2., 0., 2.]):
            box(f"FloorJointY_{i}", [1.65, y, .0004], [2.7, .003, .0006], charcoal, .0001)

    # Remove the old over-bright, nearly uniform contribution in this stage.
    inherited_lights = []
    for prim in Usd.PrimRange.Stage(stage):
        if prim.HasAPI(UsdLux.LightAPI) and not str(prim.GetPath()).startswith(ROOT):
            inherited_lights.append(str(prim.GetPath()))
            UsdLux.LightAPI(prim).GetIntensityAttr().Set(0.)
    lights = []
    dome = UsdLux.DomeLight.Define(stage, ROOT+"/Lights/Ambient")
    dome.CreateIntensityAttr(float(cfg["dome_intensity"])*cfg["exposure_multiplier"])
    dome.CreateColorAttr(Gf.Vec3f(.86, .91, 1.))

    def area(name, position, target, intensity, size, kelvin):
        light = UsdLux.RectLight.Define(stage, ROOT+"/Lights/"+name)
        light.CreateIntensityAttr(float(intensity)*cfg["exposure_multiplier"])
        light.CreateWidthAttr(float(size[0]))
        light.CreateHeightAttr(float(size[1]))
        light.CreateNormalizeAttr(False)
        light.CreateEnableColorTemperatureAttr(True)
        light.CreateColorTemperatureAttr(float(kelvin))
        direction = Gf.Vec3d(*[float(v) for v in np.array(target)-position])
        orient = Gf.Rotation(Gf.Vec3d(0, 0, -1), direction.GetNormalized()).GetQuat()
        xf = UsdGeom.Xformable(light)
        xf.AddTranslateOp().Set(Gf.Vec3d(*position))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(orient)
        lights.append(dict(path=str(light.GetPath()), position=position, target=target,
                           intensity=intensity, size=size, kelvin=kelvin))
    area("WindowKey", cfg["key_position"], [-.1, 0., .89], cfg["key_intensity"],
         cfg["key_size"], cfg["key_temperature"])
    sign = 1 if cfg["key_position"][1] > 0 else -1
    area("SoftFill", [-.7, -sign*1.1, 2.2], [-.1, 0., .9], cfg["fill_intensity"], [1.1, .9], 6500.)
    area("BenchRim", [1.1, -.2, 2.25], [.0, 0., .89], cfg["rim_intensity"], [1.4, .28], 5700.)
    if cfg["context"]:
        for i, (lamp_x, lamp_y) in enumerate([(-1.25, -.72), (.95, .72)]):
            area(f"CeilingPanel_{i}", [lamp_x, lamp_y, cfg["ceiling_height"]-.104],
                 [lamp_x, lamp_y, .0], cfg["ceiling_light_intensity"],
                 [.62, .96], cfg["ceiling_light_temperature"])
    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "PathTracing")
    for key in ("spp", "totalSpp", "clampSpp"):
        settings.set("/rtx/pathtracing/"+key, cfg["spp"])
    settings.set("/rtx/pathtracing/maxBounces", 8)
    settings.set("/rtx/pathtracing/optixDenoiser/temporalMode/enabled", False)
    settings.set("/rtx/post/histogram/enabled", False)
    settings.set("/rtx/post/tonemap/autoExposure/enabled", False)
    assert all(not any("Physics" in api or "Physx" in api for api in p.GetAppliedSchemas())
               for p in Usd.PrimRange(stage.GetPrimAtPath(ROOT)))
    for target, expected in bindings.items():
        actual = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(target)).ComputeBoundMaterial()[0]
        assert actual and str(actual.GetPath()) == expected
    return dict(profile=cfg, table_required=table_required, table_targets=table_targets, floor_targets=floor_targets,
        bindings=bindings, textures=texture_receipts, lights=lights,
        state_signature=state_signature(stage, bindings),
        inherited_lights_disabled=inherited_lights,
        assets_manifest_sha256=hashlib.sha256((assets/"ASSETS.json").read_bytes()).hexdigest(),
        room_enclosure={"ceiling": True, "four_full_walls": True, "visual_only": True,
                        "ceiling_height_m": cfg["ceiling_height"],
                        "ceiling_variant": cfg["ceiling_variant"]},
        visual_only=True, original_robot_and_task_object_materials_preserved=True)
