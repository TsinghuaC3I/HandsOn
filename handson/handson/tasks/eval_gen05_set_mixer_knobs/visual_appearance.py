"""Source-preserving GEN6 display calibration, applied to runtime USD only.

The mixer already imports its source roughness, metallic and panel textures.
Its LCD is decorative: live CURRENT / TARGET angles remain the existing HUD.
Only the LCD texture's emission multiplier changes here. Lighting and robot
appearance belong to the shared visual layer, not this task module.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


PROFILE_ID = "gen6_source_materials_lcd_v1"
LCD_EMISSION_MULTIPLIER = 0.5
_ASSET_DIR = Path(__file__).resolve().parents[2] / "sim/envs/usd/eval_gen05_set_mixer_knobs"
_VISUALS = {
    "panel_base": "mixer_fixed_visual.usda",
    "knob_left": "knob_visual.usda",
    "knob_right": "knob_visual.usda",
}
_LCD_MATERIAL = "wire_153228153"


def apply_visual_appearance(stage: Any, task_root_path: str = "/World/Task") -> dict[str, Any]:
    """Apply an idempotent LCD adjustment and report actual composed materials.

    A mismatched scene is reported without writes; this function does not alter
    collection availability. It neither edits the referenced assets on disk nor
    authors geometry, bindings, physics, cameras, lights or task state.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    root = Sdf.Path(task_root_path)
    report: dict[str, Any] = {
        "profile_id": PROFILE_ID,
        "status": "skipped",
        "task_root_path": str(root),
        "materials": [],
        "source_assets": [],
        "modified_attributes": [],
        "issues": [],
        "lcd_is_task_readout": False,
    }
    if not root.IsAbsolutePath() or not root.IsPrimPath() or root == Sdf.Path.absoluteRootPath:
        report["issues"].append("task_root_path must be an absolute prim path")
        return report

    material_paths: set[str] = set()
    for role, asset_name in _VISUALS.items():
        visual_path = root.AppendPath(f"B320Benchmark/{role}/source_visual")
        visual = stage.GetPrimAtPath(visual_path)
        source = _ASSET_DIR / asset_name
        if not visual or not source.is_file():
            report["issues"].append(f"Missing GEN6 source visual: {visual_path}")
            continue
        source_path = source.resolve()
        if not any(Path(spec.layer.realPath).resolve() == source_path
                   for spec in visual.GetPrimStack() if spec.layer.realPath):
            report["issues"].append(f"Unexpected source layer: {visual_path}")
            continue
        report["source_assets"].append({
            "role": role, "visual_prim": str(visual_path),
            "file": str(source_path),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        })
        for prim in Usd.PrimRange(visual):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if material:
                material_paths.add(str(material.GetPath()))

    for path in sorted(material_paths):
        surface = UsdShade.Shader.Get(stage, path + "/Surface")
        if not surface or surface.GetIdAttr().Get() != "UsdPreviewSurface":
            report["issues"].append(f"Unexpected bound shader: {path}")
            continue
        inputs = {}
        for name in ("diffuseColor", "roughness", "metallic"):
            value = surface.GetInput(name).Get()
            inputs[name] = list(value) if isinstance(value, (Gf.Vec3f, Gf.Vec3d)) else value
        report["materials"].append({"prim": path, "preserved_inputs": inputs})

    lcd_path = root.AppendPath(f"B320Benchmark/panel_base/source_visual/Materials/{_LCD_MATERIAL}")
    texture_path = lcd_path.AppendChild("emissiveColor")
    texture = UsdShade.Shader.Get(stage, texture_path)
    surface = UsdShade.Shader.Get(stage, lcd_path.AppendChild("Surface"))
    expected_connection = texture_path.AppendProperty("outputs:rgb")
    if str(lcd_path) not in material_paths or not texture or not surface:
        report["issues"].append("The source LCD material is not bound to the mixer")
    elif (texture.GetIdAttr().Get() != "UsdUVTexture" or
          surface.GetInput("emissiveColor").GetAttr().GetConnections() != [expected_connection]):
        report["issues"].append("Unexpected source LCD emission graph")
    if report["issues"]:
        return report

    # Set an absolute multiplier, never multiply the previous runtime override.
    # The source texture/file, UV connection, color space and alpha stay intact.
    multiplier = Gf.Vec4f(*(LCD_EMISSION_MULTIPLIER,) * 3, 1.0)
    attribute = texture.GetInput("scale")
    attribute.Set(multiplier)
    report["status"] = "applied"
    report["modified_attributes"].append({
        "prim": str(texture_path), "attribute": "inputs:scale",
        "value": list(multiplier), "source_value": [1.0, 1.0, 1.0, 1.0],
        "source_texture": str(texture.GetInput("file").Get()),
        "reason": "Reduce decorative LCD dominance while retaining its original image",
    })
    return report
