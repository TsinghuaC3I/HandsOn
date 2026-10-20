"""Check the GEN1 adapter against an existing native USD cache, without Kit.

Run with the common Isaac Sim 5.1 interpreter and its bundled omni.usd.libs
Python/library paths. No cache layer is saved; all changes are in memory.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-stage", required=True, type=Path)
    args = parser.parse_args()
    version = importlib.metadata.version("isaacsim")
    assert version.startswith("5.1."), version
    from pxr import Usd
    from handson.tasks.eval_gen03_orient_bottle_upright import visual_appearance

    root = Path(__file__).resolve().parents[3]
    scene_file = root / "handson/handson/sim/isaaclab/bimanual_assembly_scene.py"
    tree = ast.parse(scene_file.read_text())
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_repair_imported_visual_materials")
    namespace = {"os": os, "_is_robot_collision_render_path": lambda _path: False}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(scene_file), "exec"), namespace)
    stage = Usd.Stage.CreateInMemory()
    task = stage.DefinePrim("/World/Task", "Xform")
    task.GetReferences().AddReference(str(args.cached_stage.resolve()))
    repair_counts = namespace["_repair_imported_visual_materials"](stage, "/World/Task")

    def invariant_snapshot():
        result = {}
        for prim in Usd.PrimRange(task, Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if "/GEN1VisualMaterials" in path:
                continue
            visual = "/visuals" in path
            properties = {}
            for prop in prim.GetProperties():
                if visual and (prop.GetName().startswith("material:binding") or prop.GetName() == "doubleSided"):
                    continue
                value = prop.GetTargets() if hasattr(prop, "GetTargets") else prop.Get()
                properties[prop.GetName()] = repr(value)
            result[path] = {
                "type": prim.GetTypeName(), "properties": properties,
                "apis": [str(v) for v in prim.GetAppliedSchemas() if v != "MaterialBindingAPI"],
                "instanceable": prim.IsInstanceable(),
            }
        return result

    before = invariant_snapshot()
    report = visual_appearance.apply_visual_appearance(stage)
    assert report["status"] == "applied", report
    assert report["material_count"] == 3 and len(report["matched_prims"]) == 10, report
    assert invariant_snapshot() == before, "Unexpected nonappearance USD change"
    first = stage.GetRootLayer().ExportToString()
    assert visual_appearance.apply_visual_appearance(stage)["status"] == "applied"
    assert stage.GetRootLayer().ExportToString() == first, "Adapter is not idempotent"
    print(json.dumps({
        "scope": "CPU native-cache composition; no Kit, GPU, physics steps or rendered-image acceptance",
        "isaacsim": version, "usd_version": list(Usd.GetVersion()),
        "cached_stage": str(args.cached_stage.resolve()),
        "adapter_sha256": hashlib.sha256(Path(visual_appearance.__file__).read_bytes()).hexdigest(),
        "nonappearance_prim_count": len(before), "nonappearance_unchanged": True,
        "invariant_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
        "idempotent": True, "old_material_repair_counts": list(repair_counts),
        "appearance": report,
    }, indent=2))


if __name__ == "__main__":
    main()
