#!/usr/bin/env python3
"""CPU asset/preparation checks. This is NOT an Isaac/PhysX acceptance test.

Only Isaac configuration declarations are replaced during module import;
the actual repository XML preprocessing functions, files and MuJoCo compiler
run unchanged. Nothing initializes Kit, CUDA, cameras or a collection service.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import MagicMock
import xml.etree.ElementTree as ET

import numpy as np


def prepare_without_kit(output: Path):
    class ConfigDeclaration:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    os.environ["HANDSON_ISAACLAB_CACHE"] = str(output / "isaac-cache")
    for name in (
        "isaaclab", "isaaclab.sim", "isaaclab.actuators", "isaaclab.assets",
        "isaaclab.scene", "isaaclab.sensors", "isaaclab.utils", "isaaclab.utils.configclass",
    ):
        sys.modules[name] = MagicMock()
    sys.modules["isaaclab"].sim = sys.modules["isaaclab.sim"]
    sys.modules["isaaclab.sim"].MjcfFileCfg = ConfigDeclaration
    sys.modules["isaaclab.utils.configclass"].configclass = lambda cls: cls
    sys.modules["isaaclab.scene"].InteractiveSceneCfg = object
    from handson.sim.isaaclab.task_preview import prepare_task_mjcf
    return prepare_task_mjcf("eval_comb02_open_laptop_and_click_mouse", first_physx_safe=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/xy2"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "handson"))
    from handson.sim.isaaclab.fu015_contract import ASSET_RELATIVE_PATH, ASSET_SHA256
    from handson.sim.isaaclab.xy2_contract import sample_layout, layout_clearances

    digest = hashlib.sha256((root / ASSET_RELATIVE_PATH).read_bytes()).hexdigest()
    assert digest == ASSET_SHA256
    laptop_assets = {}
    for variant in ("macbook_pro_16_2021", "macbook_pro_m5_max_16_2026"):
        directory = root / "taskdev/tasks/t16_open_laptop/assets" / variant
        for name in ("macbook_base_visual.usdc", "macbook_display_visual.usdc",
                     "macbook_base_collision.usdc", "macbook_display_collision.usdc"):
            path = directory / name
            assert path.is_file(), path
        for path in directory.rglob("*"):
            if path.is_file():
                laptop_assets[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    prepared = prepare_without_kit(output)
    tree = ET.parse(prepared).getroot()
    body_names = [b.get("name") for b in tree.iter("body")]
    assert "link_3" in body_names and "mousepad" in body_names
    assert "display_link_1" not in body_names
    assert not any(n.get("objname") == "display_geom" for n in tree.iter())
    assert {"mouse_joint_0", "mouse_joint_1"} <= {j.get("name") for j in tree.iter("joint")}
    assets = {}
    for node in tree.iter():
        if node.tag not in {"mesh", "texture", "hfield"} or not node.get("file"):
            continue
        path = (prepared.parent / node.get("file")).resolve()
        assert path.is_file(), str(path)
        assets[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()

    # The diagnostic compiler detects dangling sensors, invalid inertias,
    # source-unit mistakes and missing meshes; it does not prove PhysX cooking.
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(prepared))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Native robots are imported separately from the task-only MJCF. Check
    # their actual assets too: a present Git LFS pointer is not a usable mesh.
    from handson.sim.isaaclab.bimanual_assembly_scene import _prepare_robot_mjcf, _prepare_center_mjcf
    native_robots = []
    for native in (_prepare_robot_mjcf("right"), _prepare_robot_mjcf("left"), _prepare_center_mjcf()):
        hashes = {}
        for node in ET.parse(native).getroot().iter():
            if node.tag in {"mesh", "texture"} and node.get("file"):
                path = (native.parent / node.get("file")).resolve()
                content = path.read_bytes()
                assert not content.startswith(b"version https://git-lfs.github.com/spec/v1"), f"Run git lfs pull: {path}"
                hashes[str(path)] = hashlib.sha256(content).hexdigest()
        native_model = mujoco.MjModel.from_xml_path(str(native))
        native_robots.append({"mjcf": str(native), "body_count": native_model.nbody,
                              "joint_count": native_model.njnt, "asset_sha256": hashes})
    dynamics = {}
    for name in ("link_3", "link_0", "link_1", "mousepad"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert body_id >= 0, name
        assert model.body_mass[body_id] > 0, name
        assert np.all(model.body_inertia[body_id] > 0), name
        dynamics[name] = {"mass": float(model.body_mass[body_id]),
                          "inertia": model.body_inertia[body_id].tolist()}

    samples = []
    for angle in (30, 60, 90, 120):
        for seed in (0, 1, 7, 41, 42):
            layout = sample_layout(seed, randomize=True, target_angle=angle)
            assert layout == sample_layout(seed, randomize=True, target_angle=angle)
            samples.append({**layout.to_dict(), "planar_clearances": layout_clearances(layout)})
    report = {
        "scope": "CPU source assets, XML preparation, diagnostic MuJoCo compile and seeded layout only",
        "pass": True, "isaac_physx_verified": False, "gpu_initialized": False,
        "prepared_mjcf": str(prepared.relative_to(root)),
        "laptop_sha256": digest, "prepared_asset_hashes": assets,
        "source_macbook_asset_hashes": laptop_assets,
        "native_robot_models": native_robots,
        "mouse_dynamics_diagnostic": dynamics, "layout_samples": samples,
        "position_spans": {
            "mouse_xy": np.ptp([x["mouse_xy"] for x in samples], axis=0).tolist(),
            "pad_xy": np.ptp([x["pad_xy"] for x in samples], axis=0).tolist(),
            "laptop_xy": np.ptp([[x["laptop"]["root_x_m"], x["laptop"]["root_y_m"]] for x in samples], axis=0).tolist(),
        },
        "pending": ["actual composed USD shape intersections and support", "live PhysX reset and reach",
                    "4-angle complete physical trajectories", "platform collection", "human teleoperation",
                    "three real post-rendered cameras", "fresh full-trajectory physical replay"],
    }
    path = output / "cpu-assets-layout.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": True, "report": str(path), "samples": len(samples), "asset_files": len(assets)}))


if __name__ == "__main__":
    main()
