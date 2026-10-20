"""Runtime scene loader for the user-requested pizza collection task.

The downloaded GLB/glTF files remain the source of truth. On first launch this
module creates cached USD derivatives and references them into the Isaac stage;
it does not author replacement food geometry.
"""

from __future__ import annotations

import gc
import hashlib
import re
from pathlib import Path
from typing import Any

TASK_NAME = "user_make_pizza_long_horizon"
TASK_ROOT = "/World/Task/PizzaLongHorizon"
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SOURCE_ROOT = _REPO_ROOT / "taskdev" / "tasks" / TASK_NAME
_CACHE_ROOT = Path.home() / ".cache" / "handson" / TASK_NAME

ASSETS: tuple[tuple[str, str, tuple[float, float, float], float], ...] = (
    # The source mesh has an offset origin; this pose centers its flat dough
    # component over the realistic bowl so the white starting state is visible
    # and does not interpenetrate the board at reset.
    ("assets/sketchfab_pizza_dough/pizza_dough.glb", "pizza_raw_dough", (-0.18, 0.15, 1.07), 0.008),
    ("assets/polyhaven_wooden_bowl_01/wooden_bowl_01.gltf", "pizza_kneading_bowl", (-0.18, 0.22, 1.00), 1.0),
    ("assets/kenney_food_kit/rollingPin.glb", "pizza_rolling_pin", (-0.02, 0.20, 1.00), 0.12),
    ("assets/polyhaven_wooden_cutting_board/wooden_cutting_board.gltf", "pizza_cutting_board", (-0.02, 0.20, 0.94), 1.0),
    ("assets/kenney_food_kit/bottle-ketchup.glb", "pizza_ketchup_sauce", (-0.15, 0.12, 0.98), 0.12),
    # Keep the ingredients on separate staging spots until the operator
    # performs the ordered placement.  The old layout put all three dynamic
    # topping bodies inside the assembled pizza at reset; GPU PhysX then had
    # to resolve a deep multi-body interpenetration while Kit was still
    # booting, which could leave the collection button looking unresponsive.
    ("assets/kenney_food_kit/cheese.glb", "pizza_cheese_topping", (-0.34, 0.10, 0.99), 0.08),
    ("assets/kenney_food_kit/tomato-slice.glb", "pizza_tomato_topping", (-0.34, 0.00, 0.99), 0.08),
    ("assets/kenney_food_kit/pepper.glb", "pizza_pepper_topping", (-0.24, 0.00, 0.99), 0.08),
    ("assets/polyhaven_brass_pan_01/brass_pan_01.gltf", "pizza_frying_pan", (-0.08, -0.18, 0.98), 1.0),
    ("assets/kenney_food_kit/pizza.glb", "pizza_assembled_slices", (0.02, 0.05, 0.99), 0.08),
    ("assets/kenney_food_kit/pizza-cutter.glb", "pizza_cutter", (0.10, 0.10, 0.99), 0.12),
    ("assets/polyhaven_carved_wooden_plate/carved_wooden_plate.gltf", "pizza_serving_plate", (0.17, 0.15, 0.95), 1.0),
)

# The downloaded Sketchfab dough mesh keeps its original source material for
# provenance, but the operator task requires a plain white dough appearance.
# This is a material-only runtime override; no geometry is authored here.
MATERIAL_MODES: dict[str, str] = {"pizza_raw_dough": "solid_white"}
CONVERTER_VERSION = "v5_source_slice_rigid_bodies"

# The pizza task reuses the bimanual robot/table bootstrap MJCF so the
# operator gets the canonical ROKAE + Wuji rig. That MJCF also contains
# generic assembly props (pegs and mocap targets), which are unrelated to the
# recipe and appear as floating/transparent objects in the viewport. Keep the
# table, world body, robots, and lighting; deactivate only these known props.
DEFAULT_ASSEMBLY_CLUTTER_ROOTS: tuple[str, ...] = (
    "/World/Task/industreal_round_peg_8mm",
    "/World/Task/industreal_tray_insert_round_peg_8mm",
    "/World/Task/target_right",
    "/World/Task/target_left",
)
DEFAULT_TRANSPARENT_TABLE_DECORATION_PREFIXES: tuple[str, ...] = (
    "table_edge_clutter_",
    "tablecloth_",
)
DEFAULT_VIEWPORT_HELPER_NAMES: tuple[str, ...] = (
    "head_main_fov_helper_frame",
)


def hide_default_assembly_clutter(stage) -> tuple[str, ...]:
    """Deactivate non-pizza props and alpha-zero table decoration."""

    hidden: list[str] = []
    for path in DEFAULT_ASSEMBLY_CLUTTER_ROOTS:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and prim.IsActive():
            prim.SetActive(False)
            hidden.append(path)
    table_root = stage.GetPrimAtPath("/World/Task/table")
    if table_root.IsValid():
        from pxr import Usd

        for prim in Usd.PrimRange(table_root):
            name = prim.GetName()
            if not name.startswith(DEFAULT_TRANSPARENT_TABLE_DECORATION_PREFIXES):
                continue
            if prim.IsActive():
                prim.SetActive(False)
                hidden.append(str(prim.GetPath()))
    # The shared ROKAE bootstrap includes a debug-only FOV wireframe under the
    # head camera.  It is not a task asset and is particularly confusing in
    # the operator view because it looks like a floating transparent object.
    task_root = stage.GetPrimAtPath("/World/Task")
    if task_root.IsValid():
        from pxr import Usd

        for prim in Usd.PrimRange(task_root):
            if prim.GetName() not in DEFAULT_VIEWPORT_HELPER_NAMES:
                continue
            if prim.IsActive():
                prim.SetActive(False)
                hidden.append(str(prim.GetPath()))
    return tuple(hidden)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str, fallback: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", value)
    return name if name and not name[0].isdigit() else f"_{name or fallback}"


def _rgba(mesh) -> tuple[float, float, float, float]:
    import numpy as np

    material = getattr(mesh.visual, "material", None)
    raw = getattr(material, "main_color", None)
    color = np.asarray(raw if raw is not None else [180, 180, 180, 255], dtype=np.float64).reshape(-1)
    if color.size == 3:
        color = np.concatenate([color, [255.0]])
    color = np.clip(color[:4] / 255.0, 0.0, 1.0)
    return tuple(float(value) for value in color)


def _convert_source(
    source: Path,
    output: Path,
    *,
    material_mode: str | None = None,
    split_nodes: bool = False,
) -> dict[str, Any]:
    import numpy as np
    import trimesh
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    scene = trimesh.load(str(source), force="scene", process=False)
    if not isinstance(scene, trimesh.Scene) or not scene.geometry:
        raise RuntimeError(f"pizza source did not contain a mesh scene: {source}")
    if scene.units not in (None, "meters"):
        raise RuntimeError(f"pizza source units are not metres: {source} ({scene.units!r})")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    root.SetCustomDataByKey("handson:sourceFile", source.name)
    root.SetCustomDataByKey("handson:sourceSha256", _sha256(source))
    root.SetCustomDataByKey("handson:sourceUnits", "meters")
    root.SetCustomDataByKey("handson:coordinateMap", "gltf_y_up_to_usd_z_up_x_z_neg_y")
    root.SetCustomDataByKey("handson:converterVersion", CONVERTER_VERSION)
    if material_mode:
        root.SetCustomDataByKey("handson:materialOverride", material_mode)
    texture_image = None
    for candidate in scene.geometry.values():
        material = getattr(candidate.visual, "material", None)
        texture = getattr(material, "baseColorTexture", None)
        if texture is not None:
            texture_image = texture.copy()
            break
    texture_path = output.with_name(f"{output.stem}_basecolor.png")
    if material_mode == "solid_white":
        texture_image = None
    pizza_material = UsdShade.Material.Define(
        stage,
        f"/Asset/Materials/{_safe_name(output.stem, 'pizza_material')}",
    )
    surface = UsdShade.Shader.Define(
        stage,
        f"{pizza_material.GetPath()}/Surface",
    )
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
        0.82 if material_mode == "solid_white" else 0.62
    )
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    if texture_image is not None:
        texture_image.convert("RGBA").save(texture_path, format="PNG")
        primvar_reader = UsdShade.Shader.Define(
            stage,
            f"{pizza_material.GetPath()}/PrimvarReader",
        )
        primvar_reader.CreateIdAttr("UsdPrimvarReader_float2")
        primvar_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        primvar_output = primvar_reader.CreateOutput(
            "result", Sdf.ValueTypeNames.Float2
        )
        texture_shader = UsdShade.Shader.Define(
            stage,
            f"{pizza_material.GetPath()}/Texture",
        )
        texture_shader.CreateIdAttr("UsdUVTexture")
        texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(texture_path.resolve()))
        )
        texture_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            primvar_output
        )
        texture_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        texture_shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
            "sRGB"
        )
        texture_rgb = texture_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture_rgb
        )
        pizza_material.CreateSurfaceOutput().ConnectToSource(
            surface.ConnectableAPI(), "surface"
        )
    else:
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(1.0, 1.0, 1.0)
        )
        pizza_material.CreateSurfaceOutput().ConnectToSource(
            surface.ConnectableAPI(), "surface"
        )
    mesh_count = 0
    for index, node_name in enumerate(sorted(scene.graph.nodes_geometry)):
        transform, geometry_name = scene.graph[node_name]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(np.asarray(transform, dtype=np.float64))
        points = np.asarray(mesh.vertices, dtype=np.float32)
        points = np.column_stack((points[:, 0], points[:, 2], -points[:, 1]))
        faces = np.asarray(mesh.faces, dtype=np.int32)
        safe_node_name = _safe_name(str(node_name), f"mesh_{index}")
        if split_nodes:
            # Keep every source pizza slice as its own Xform so the runtime can
            # attach a real rigid body to the source node after the two cut
            # paths are observed.  The mesh points still come directly from
            # the downloaded GLB; this wrapper only supplies a physics owner.
            UsdGeom.Xform.Define(stage, f"/Asset/{safe_node_name}")
            mesh_path = f"/Asset/{safe_node_name}/mesh"
        else:
            mesh_path = f"/Asset/{safe_node_name}"
        usd_mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        usd_mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])
        usd_mesh.CreateFaceVertexCountsAttr([3] * int(len(faces)))
        usd_mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
        usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        usd_mesh.CreateDoubleSidedAttr(True)
        uv = getattr(mesh.visual, "uv", None)
        if uv is not None and len(uv) == len(points):
            uv_primvar = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
                "st",
                Sdf.ValueTypeNames.TexCoord2fArray,
                UsdGeom.Tokens.vertex,
            )
            uv_primvar.Set(
                [Gf.Vec2f(float(value[0]), float(value[1])) for value in uv]
            )
        if pizza_material is not None:
            UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(
                pizza_material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
        red, green, blue, alpha = (
            (1.0, 1.0, 1.0, 1.0)
            if material_mode == "solid_white"
            else _rgba(mesh)
        )
        usd_mesh.CreateDisplayColorAttr([Gf.Vec3f(red, green, blue)])
        usd_mesh.CreateDisplayOpacityAttr([alpha])
        mesh_count += 1
    stage.GetRootLayer().Save()
    return {"source": str(source), "source_sha256": _sha256(source), "derived_usd": str(output), "derived_usd_sha256": _sha256(output), "mesh_count": mesh_count}


def ensure_cached_usd() -> dict[str, Path]:
    derived_root = _CACHE_ROOT / "derived_usd"
    derived_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for source_rel, logical_name, _position, _scale in ASSETS:
        source = _SOURCE_ROOT / source_rel
        if not source.is_file():
            raise FileNotFoundError(f"pizza source asset is missing: {source}")
        output = derived_root / f"{logical_name}.usd"
        source_hash = _sha256(source)
        needs_build = not output.is_file()
        if not needs_build:
            from pxr import Usd
            stage = Usd.Stage.Open(str(output), Usd.Stage.LoadAll)
            root = stage.GetDefaultPrim() if stage is not None else None
            needs_build = (
                root is None
                or not root.IsValid()
                or root.GetCustomDataByKey("handson:sourceSha256") != source_hash
                or root.GetCustomDataByKey("handson:converterVersion")
                != CONVERTER_VERSION
            )
            # Release the open SdfLayer before replacing the same USD file.
            # USD rejects CreateNew while the old layer identifier is still
            # held by this process.
            del root
            del stage
            gc.collect()
        if needs_build:
            _convert_source(
                source,
                output,
                material_mode=MATERIAL_MODES.get(logical_name),
                split_nodes=logical_name == "pizza_assembled_slices",
            )
        paths[logical_name] = output
    return paths


def spawn_assets(stage, sim_utils: Any) -> dict[str, Any]:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    paths = ensure_cached_usd()
    report: list[dict[str, Any]] = []
    for _source_rel, logical_name, position, scale in ASSETS:
        prim_path = f"{TASK_ROOT}/{logical_name}"
        root = UsdGeom.Xform.Define(stage, prim_path)
        root.AddTranslateOp().Set(Gf.Vec3d(*position))
        root.AddScaleOp().Set(Gf.Vec3d(scale, scale, scale))

        if logical_name == "pizza_assembled_slices":
            # The source pizza.glb exposes eight independent nodes. Reference
            # the nodes themselves and give each one a dynamic rigid body;
            # the parent is only a transform namespace and never receives a
            # pose/velocity write during an action.
            root.GetPrim().GetReferences().AddReference(str(paths[logical_name]))
            slice_names = tuple(f"slice{index}" for index in range(1, 9))
            slice_paths: list[str] = []
            for slice_name in slice_names:
                slice_prim = stage.GetPrimAtPath(f"{prim_path}/{slice_name}")
                if not slice_prim.IsValid():
                    raise RuntimeError(
                        f"pizza source node is missing after reference: {slice_name}"
                    )
                rigid = UsdPhysics.RigidBodyAPI.Apply(slice_prim)
                rigid.CreateKinematicEnabledAttr().Set(False)
                mass = UsdPhysics.MassAPI.Apply(slice_prim)
                mass.CreateMassAttr().Set(0.04)
                for prim in Usd.PrimRange(slice_prim):
                    if prim.IsA(UsdGeom.Gprim):
                        collision = UsdPhysics.CollisionAPI.Apply(prim)
                        collision.CreateCollisionEnabledAttr().Set(True)
                        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                        mesh_collision.CreateApproximationAttr().Set("convexHull")
                slice_paths.append(f"{prim_path}/{slice_name}")

            # Keep the eight source bodies assembled until the runtime has
            # observed two physically valid cutter paths.  The runtime only
            # disables these authored joints after that event; it never snaps
            # slice poses or velocities.
            joint_root = f"{prim_path}/source_slice_joints"
            joint_paths: list[str] = []
            for index in range(len(slice_paths) - 1):
                joint_path = f"{joint_root}/slice_joint_{index + 1}"
                joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
                joint.CreateBody0Rel().SetTargets(
                    [Sdf.Path(slice_paths[index])]
                )
                joint.CreateBody1Rel().SetTargets(
                    [Sdf.Path(slice_paths[index + 1])]
                )
                joint_paths.append(joint_path)
            report.append(
                {
                    "name": logical_name,
                    "prim_path": prim_path,
                    "source_usd": str(paths[logical_name]),
                    "source_nodes": slice_names,
                    "slice_paths": tuple(slice_paths),
                    "assembly_joint_paths": tuple(joint_paths),
                }
            )
            continue

        root.GetPrim().GetReferences().AddReference(str(paths[logical_name]))
        rigid = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        rigid.CreateKinematicEnabledAttr(False)
        mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
        mass.CreateMassAttr(0.15 if logical_name not in {"pizza_kneading_bowl", "pizza_cutting_board", "pizza_frying_pan", "pizza_serving_plate"} else 0.5)
        for prim in Usd.PrimRange(root.GetPrim()):
            if prim.IsA(UsdGeom.Gprim):
                collision = UsdPhysics.CollisionAPI.Apply(prim)
                collision.CreateCollisionEnabledAttr(True)
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_collision.CreateApproximationAttr().Set("convexHull")
        report.append({"name": logical_name, "prim_path": prim_path, "source_usd": str(paths[logical_name])})
    sim_utils.update_stage()
    return {"assets": report, "cache_root": str(_CACHE_ROOT)}
