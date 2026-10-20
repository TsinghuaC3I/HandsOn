#!/usr/bin/env python3
"""Check GEN4 source appearance on real assets in a CPU composition fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

from handson.sim.isaaclab.gen4_scene import ASSET_DIR, TRAY_ASSET
from handson.tasks.eval_gen04_pass_glue_stick import contract as c
from handson.tasks.eval_gen04_pass_glue_stick.visual_appearance import apply_visual_appearance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Task")
    for name in (c.GLUE_BODY, c.TRAY_BODY, *c.DISTRACTORS):
        owner = UsdGeom.Xform.Define(stage, "/World/Task/" + name)
        UsdPhysics.RigidBodyAPI.Apply(owner.GetPrim())
    dependencies = {}
    for name, asset in ((c.GLUE_BODY, ASSET_DIR / "glue_stick.usd"), (c.TRAY_BODY, TRAY_ASSET)):
        prim = UsdGeom.Xform.Define(stage, "/World/Task/" + name + "/SelectedAsset")
        prim.GetPrim().GetReferences().AddReference(str(asset))
        layers, files, unresolved = UsdUtils.ComputeAllDependencies(str(asset))
        assert not unresolved, unresolved
        paths = {Path(p.realPath) for p in layers} | {Path(p) for p in files}
        dependencies[name] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    before = stage.GetRootLayer().ExportToString()
    report = apply_visual_appearance(stage)
    assert before == stage.GetRootLayer().ExportToString()
    assert apply_visual_appearance(stage) == report
    assert before == stage.GetRootLayer().ExportToString()

    source = ASSET_DIR / "source/customisable_glue_stick.glb"
    scene = trimesh.load(source, force="scene", process=False)
    glue_stage = Usd.Stage.Open(str(ASSET_DIR / "glue_stick.usd"))
    source_checks = []
    for i, node in enumerate(sorted(scene.graph.nodes_geometry)):
        _, name = scene.graph[node]
        mesh = scene.geometry[name]
        uv = UsdGeom.PrimvarsAPI(glue_stage.GetPrimAtPath(f"/GlueStick/Visual/Part{i}")).GetPrimvar("st").Get()
        assert np.array_equal(np.asarray(mesh.visual.uv, dtype=np.float32), np.asarray(uv))
        for attr, label in (("baseColorTexture", "BaseColor"), ("metallicRoughnessTexture", "MetallicRoughness"),
                            ("normalTexture", "Normal"), ("occlusionTexture", "Occlusion")):
            original = np.asarray(getattr(mesh.visual.material, attr))
            converted = np.asarray(Image.open(ASSET_DIR / "textures" / f"{label}_{i}.png"))
            assert np.array_equal(original, converted)
            row = {"part": i, "map": label, "pixels_exact": True, "shape": list(converted.shape)}
            if label == "MetallicRoughness":
                row.update(metallic_blue_min=int(converted[..., 2].min()),
                           metallic_blue_max=int(converted[..., 2].max()),
                           roughness_green_min=int(converted[..., 1].min()),
                           roughness_green_max=int(converted[..., 1].max()))
            source_checks.append(row)
    raw = source.read_bytes()
    size, _ = struct.unpack_from("<II", raw, 12)
    materials = json.loads(raw[20:20 + size])["materials"]
    assert all(m.get("normalTexture", {}).get("scale", 1.) == 1. for m in materials)
    result = {"completed": True, "fixture": "CPU composed original glue/tray, synthetic owner tree; not native scene acceptance",
              "task_layer_authored_zero_changes": True, "idempotent": True,
              "source_uvs_exact": True, "source_texture_checks": source_checks,
              "dependencies": dependencies, "appearance": report,
              "native_three_camera_validation": "pending fresh Isaac 5.1 processes"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed": True, "maps_exact": len(source_checks), "changed_attributes": 0}))


if __name__ == "__main__":
    main()
