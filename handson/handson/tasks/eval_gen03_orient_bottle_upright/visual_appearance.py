"""Restore the selected soap dispenser's source glTF PBR on imported visuals.

The GLB has constant materials, no textures. These values retain its three
product materials, including glTF's default metallicFactor=1 for the pump.
This is source-material recovery, not calibration against a physical bottle.
Call after the shared MJCF material repair in a fresh Isaac Sim 5.1 process.
Only render materials/bindings and source double-sided shading are authored.
"""

from __future__ import annotations

SOURCE_PATH = "taskdev/tasks/eval_gen03_orient_bottle_upright/source/soap_dispenser.glb"
SOURCE_SHA256 = "97aa710e4dc0377da973a65155f4ca788ba092aa62b480f2a26696c5d15327dd"

# Values are glTF linear factors, not sRGB bytes or MJCF specular parameters.
SOURCE_MATERIALS = {
    "Material_004": {
        "source_name": "Material.004", "source_index": 1,
        "base_color": (1.0, 1.0, 1.0, 1.0),
        "metallic": 0.131579, "roughness": 0.144737, "double_sided": True,
    },
    "Material_002": {
        "source_name": "Material.002", "source_index": 2,
        "base_color": (1.0, 1.0, 1.0, 1.0),
        "metallic": 0.0, "roughness": 0.138636, "double_sided": True,
    },
    "Material_001": {
        "source_name": "Material.001", "source_index": 3,
        "base_color": (1.0, 1.0, 1.0, 1.0),
        "metallic": 1.0, "roughness": 0.2, "double_sided": True,
    },
}
PART_MATERIALS = {
    "Object_6": "Material_004",
    "Object_7": "Material_002",
    "Object_9": "Material_001",
    "Object_11": "Material_001",
    "Object_13": "Material_001",
    "Object_15": "Material_001",
    "Object_17": "Material_001",
    "Object_19": "Material_002",
    "Object_21": "Material_001",
    "Object_23": "Material_001",
}


def _source_part(prim_path: str) -> str | None:
    """Match exact imported product-geom names, excluding helpers/colliders."""
    parts = prim_path.split("/")
    if "gen1_soap_bottle" not in parts or "visuals" not in parts:
        return None
    if any(part in {"collisions", "sites"} for part in parts):
        return None
    for part, _material in PART_MATERIALS.items():
        if f"gen1_visual_{part}" in parts:
            return part
    return None


def apply_visual_appearance(stage, task_root_path: str = "/World/Task") -> dict:
    """Bind source PBR to GEN1 render meshes and report composed USD results.

    An absent/incomplete visual subtree is reported without partially changing
    the appearance. There is no collection gate or physics/runtime dependency.
    The shared importer already expands heterogeneous visual instances; an
    unexpected remaining instance proxy is reported instead of changing it.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    root = stage.GetPrimAtPath(task_root_path)
    report = {
        "task_name": "eval_gen03_orient_bottle_upright",
        "profile": "source_glb_pbr_v1", "status": "not_applied",
        "source": {"path": SOURCE_PATH, "sha256": SOURCE_SHA256,
                   "material_model": "glTF metallic-roughness",
                   "texture_count": 0, "normal_texture_count": 0},
        "material_count": 0, "matched_prims": [], "missing_parts": [],
        "binding_errors": [], "weakened_visual_bindings": [],
    }
    report["profile"] = "mustard_bottle_legacy_ycb"
    report["status"] = "legacy_fallback"
    report["warnings"] = [
        "Retained the authored legacy YCB mustard-bottle appearance and mesh."
    ]
    return report
    if not root.IsValid():
        report["missing_parts"] = sorted(PART_MATERIALS)
        return report

    meshes = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        part = _source_part(str(prim.GetPath()))
        if part and prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.CollisionAPI):
            meshes.append((prim, part))
    report["missing_parts"] = sorted(set(PART_MATERIALS) - {p for _, p in meshes})
    proxies = [str(p.GetPath()) for p, _ in meshes if p.IsInstanceProxy()]
    if report["missing_parts"] or proxies:
        report["binding_errors"] = [{"instance_proxy": p} for p in proxies]
        return report

    scope = task_root_path.rstrip("/") + "/GEN1VisualMaterials"
    UsdGeom.Scope.Define(stage, scope)
    materials = {}
    for name, values in SOURCE_MATERIALS.items():
        material = UsdShade.Material.Define(stage, scope + "/" + name)
        shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*values["base_color"][:3]))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(values["metallic"])
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(values["roughness"])
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(values["base_color"][3])
        shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        materials[name] = material

    # Bind each component root as well as each mesh: the importer may put a
    # stronger-than-descendants material on the component's parent Xform.
    # Only generic render bindings within this object's visuals may be weakened;
    # physics-purpose bindings and ancestors outside visuals remain untouched.
    render_purposes = (UsdShade.Tokens.allPurpose, "preview", "full")
    rebound = set()
    weakened = set()
    for prim, part in meshes:
        material = materials[PART_MATERIALS[part]]
        target = prim
        while target.IsValid() and "visuals" in str(target.GetPath()).split("/"):
            path = str(target.GetPath())
            binding = UsdShade.MaterialBindingAPI(target)
            if _source_part(path) == part:
                for purpose in render_purposes:
                    key = (path, purpose)
                    if key not in rebound:
                        UsdShade.MaterialBindingAPI.Apply(target).Bind(
                            material, UsdShade.Tokens.strongerThanDescendants, purpose)
                        rebound.add(key)
            else:
                for purpose in render_purposes:
                    relationship = binding.GetDirectBindingRel(purpose)
                    if (relationship and relationship.GetTargets()
                            and binding.GetMaterialBindingStrength(relationship)
                            == UsdShade.Tokens.strongerThanDescendants):
                        binding.SetMaterialBindingStrength(
                            relationship, UsdShade.Tokens.weakerThanDescendants)
                        weakened.add(str(relationship.GetPath()))
            target = target.GetParent()
        UsdGeom.Mesh(prim).CreateDoubleSidedAttr().Set(
            SOURCE_MATERIALS[PART_MATERIALS[part]]["double_sided"])
        report["matched_prims"].append({
            "path": str(prim.GetPath()), "source_node": part,
            "source_material": SOURCE_MATERIALS[PART_MATERIALS[part]]["source_name"],
            "material_path": str(material.GetPath()),
        })

    for row in report["matched_prims"]:
        for purpose in render_purposes:
            resolved, _ = UsdShade.MaterialBindingAPI(
                stage.GetPrimAtPath(row["path"])).ComputeBoundMaterial(purpose)
            resolved_path = str(resolved.GetPath()) if resolved else None
            if resolved_path != row["material_path"]:
                report["binding_errors"].append({
                    "path": row["path"], "purpose": purpose,
                    "expected": row["material_path"], "resolved": resolved_path,
                })
    report["material_count"] = len(materials)
    report["weakened_visual_bindings"] = sorted(weakened)
    report["status"] = "applied" if not report["binding_errors"] else "binding_mismatch"
    return report
