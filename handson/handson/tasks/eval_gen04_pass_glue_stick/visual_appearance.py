"""Keep GEN4's source plastic, printed label, wood, and clutter appearance.

The shared visual layer supplies the adopted robot, workbench and lighting.
This task has already preserved its source PBR maps, so this hook verifies the
composed bindings and reports them without modifying any USD attribute.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .contract import DISTRACTORS, GLUE_BODY, TRAY_BODY


def _material_report(material):
    from pxr import Sdf, Usd, UsdShade

    textures = []
    shaders = []
    for prim in Usd.PrimRange(material.GetPrim()):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        shader_id = str(shader.GetIdAttr().Get())
        if shader_id == "UsdUVTexture":
            asset = shader.GetInput("file").Get()
            if not isinstance(asset, Sdf.AssetPath) or not asset.resolvedPath:
                raise ValueError(f"Unresolved GEN4 texture: {prim.GetPath()}")
            path = Path(asset.resolvedPath)
            textures.append({
                "node": str(prim.GetPath()), "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "color_space": str(shader.GetInput("sourceColorSpace").Get()),
                "scale": list(shader.GetInput("scale").Get()) if shader.GetInput("scale") else None,
                "bias": list(shader.GetInput("bias").Get()) if shader.GetInput("bias") else None,
            })
        if shader_id == "UsdPreviewSurface":
            inputs = {}
            for name in ("diffuseColor", "roughness", "metallic", "normal", "occlusion"):
                input_ = shader.GetInput(name)
                if not input_:
                    continue
                source = input_.GetConnectedSource()
                value = input_.Get()
                inputs[name] = {
                    "connection": str(source[0].GetPath()) + "." + str(source[1]) if source else None,
                    "value": float(value) if isinstance(value, (int, float)) else list(value) if value is not None else None,
                }
            shaders.append({"path": str(prim.GetPath()), "inputs": inputs})
    if not shaders:
        raise ValueError(f"GEN4 source surface shader missing: {material.GetPath()}")
    return {"path": str(material.GetPath()), "surfaces": shaders, "textures": textures}


def apply_visual_appearance(stage, task_root_path="/World/Task") -> dict:
    """Verify source bindings; intentionally author no scene changes."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    root = stage.GetPrimAtPath(task_root_path)
    if not root:
        raise ValueError(f"Missing GEN4 task root: {task_root_path}")
    names = (GLUE_BODY, TRAY_BODY, *DISTRACTORS)
    owners = {name: [] for name in names}
    for prim in Usd.PrimRange(root):
        if prim.GetName() in owners and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            owners[prim.GetName()].append(prim)
    # Asset composition mounts the selected glue/tray/clutter references after
    # the shared appearance hook runs.  Keep startup fail-safe and let the
    # scene composer perform the strict owner/collision validation at mount
    # time instead of terminating the collection process here.
    if any(len(values) != 1 for values in owners.values()):
        return {
            "status": "source_pbr_pending_composition",
            "changed_attributes": 0,
            "objects": {},
            "distractors_preserved": list(DISTRACTORS),
            "scope": "Source appearance verification deferred until GEN4 asset composition.",
            "geometry_physics_camera_control_changed": False,
        }
    objects = {}
    for name in (GLUE_BODY, TRAY_BODY):
        owner = owners[name][0]
        selected = owner.GetChild("SelectedAsset")
        legacy_fallback = False
        if not selected:
            # Formal GEN4 collection intentionally keeps D03/B05's original
            # visual and collision scopes. Those legacy imports predate the
            # newer SelectedAsset wrapper, so inspect their existing visual
            # scope instead of treating the old appearance as missing.
            selected = owner.GetChild("visuals")
            legacy_fallback = bool(selected)
        if not selected:
            raise ValueError(f"Missing GEN4 selected visual asset: {name}")
        meshes = []
        materials = {}
        for prim in Usd.PrimRange(selected, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if not material or (
                not legacy_fallback
                and not material.GetPath().HasPrefix(owner.GetPath())
            ):
                raise ValueError(f"GEN4 source visual material overridden: {prim.GetPath()}")
            meshes.append({"path": str(prim.GetPath()), "material": str(material.GetPath())})
            materials[str(material.GetPath())] = material
        if not meshes:
            raise ValueError(f"GEN4 has no selected visible meshes: {name}")
        objects[name] = {"owner": str(owner.GetPath()), "meshes": meshes,
                         "materials": [_material_report(m) for m in materials.values()],
                         "legacy_fallback": legacy_fallback}
    return {
        "status": "source_pbr_preserved", "changed_attributes": 0,
        "objects": objects, "distractors_preserved": list(DISTRACTORS),
        "scope": "Source label/plastic PBR and wood retained; no v12 bowl/ball material transfer.",
        "geometry_physics_camera_control_changed": False,
    }
