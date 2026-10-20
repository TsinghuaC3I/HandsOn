#!/usr/bin/env python3
"""Offline source-asset geometry audit; never launches Isaac or claims PhysX QA."""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "handson"))


def prepare_source_scene(cache: Path):
    """Execute real MJCF preparation helpers without GPU-only scene configs.

    Only Isaac imports and the declarative GPU scene configuration class are
    omitted from the assembly module. Mesh/XML preparation code is unchanged.
    No simulator, rendering or physics implementation is replaced or claimed.
    """
    os.environ["HANDSON_ISAACLAB_CACHE"] = str(cache)
    name = "handson.sim.isaaclab.bimanual_assembly_scene"
    source = ROOT / "handson/handson/sim/isaaclab/bimanual_assembly_scene.py"
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("isaaclab")
        or isinstance(node, ast.Import) and any(a.name.startswith("isaaclab") for a in node.names)
        or isinstance(node, ast.ClassDef) and node.name == "BimanualAssemblySceneCfg"
    )]
    module = types.ModuleType(name)
    module.__file__ = str(source)
    module.__package__ = "handson.sim.isaaclab"
    sys.modules[name] = module
    exec(compile(tree, str(source), "exec"), module.__dict__)
    lab = types.ModuleType("isaaclab")
    lab.sim = types.ModuleType("isaaclab.sim")
    sys.modules["isaaclab"] = lab
    sys.modules["isaaclab.sim"] = lab.sim
    preview = importlib.import_module("handson.sim.isaaclab.task_preview")
    from handson.tasks.eval_comb01_cover_pitcher_and_box.layout import TASK_NAME
    return preview.prepare_task_mjcf(TASK_NAME, first_physx_safe=True), preview


def audit_layouts():
    """12 seeds plus eight named boundaries, shared with the live probe."""
    import numpy as np
    from handson.tasks.eval_comb01_cover_pitcher_and_box import layout
    from handson.sim.isaaclab.abc_collection_env import _TABLE_TOP_Z

    rows = []
    for seed in range(12):
        height = _TABLE_TOP_Z + .010 * (seed / 11)
        rows.append((f"seed_{seed}", height, layout.sample_layout(
            np.random.RandomState(seed), randomize=True, table_top_z=height,
        )))
    labels = ("closest_x", "closest_box_y", "farthest_y", "near_table_edge",
              "far_reach", "yaw_negative", "yaw_positive", "closest_high_table")
    for index, label in enumerate(labels):
        height = _TABLE_TOP_Z + (.010 if index in (4, 7) else 0.)
        poses = layout.sample_layout(np.random.RandomState(0), randomize=False, table_top_z=height)
        for name in layout.BODY_NAMES:
            box = name in {"Box_lid", "Clear_Box"}
            front = name in {"pitcher_lid", "Box_lid"}
            low, high = layout.XY_BOUNDS[name]
            if index in (0, 7):
                poses[name][0] = high[0] if box else low[0]
            elif index in (1, 2):
                toward = front if index == 1 else not front
                poses[name][1] = high[1] if toward else low[1]
            elif index == 3:
                poses[name][0] = low[0]
            elif index == 4:
                poses[name][:2] = high
            else:
                poses[name][:2] = low if index == 5 else high
            yaw = np.deg2rad((-1 if index in (0, 3, 5) else 1) * (5. if box else 180.))
            poses[name][3:] = [np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]
        if layout.layout_margin(poses) < 0:
            raise ValueError(f"boundary outside supported layout: {label}")
        rows.append((label, height, poses))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/xy5/cpu_geometry.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mjcf, preview = prepare_source_scene(args.output.parent / "scene_cache")
    import mujoco
    import numpy as np
    from handson.tasks.eval_comb01_cover_pitcher_and_box import layout
    from handson.sim.isaaclab.abc_collection_env import _TABLE_TOP_Z
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    homes = preview._task_arm_home(layout.TASK_NAME)
    for side in ("right", "left"):
        for index, angle in enumerate(homes[side], 1):
            joint = model.joint(f"joint{index}_{side}")
            data.qpos[joint.qposadr[0]] = angle
        data.qpos[model.joint(f"{side}_finger1_joint1").qposadr[0]] = 0.10
    joint_home = data.qpos.copy()
    from handson.sim.isaaclab.eval_comb01_cover_pitcher_and_box_env import XY5PitcherThenBoxLidEnv
    interface_env = XY5PitcherThenBoxLidEnv(launch_isaac=False, image_obs=False)
    rows = []
    layouts = audit_layouts()
    table_id = model.body("table").id
    table_start = model.body_pos[table_id].copy()
    for label, table_height, poses in layouts:
        model.body_pos[table_id] = table_start + [0., 0., table_height - _TABLE_TOP_Z]
        data.qpos[:] = joint_home
        for name, pose in poses.items():
            body = model.body(name)
            joint_id = model.body_jntadr[body.id]
            adr = model.jnt_qposadr[joint_id]
            data.qpos[adr:adr + 7] = pose
        mujoco.mj_forward(model, data)
        interface_env._xy5_layout = poses
        interface_env.delta_h[0] = table_height - _TABLE_TOP_Z
        points = {side: tuple(data.xpos[i].copy() for i in range(model.nbody)
                             if side in model.body(i).name and model.body(i).name != f"base_{side}")
                  for side in ("right", "left")}
        hand_margin = interface_env._reset_hand_layout_margin(points["right"], points["left"])
        contacts = []
        for contact in data.contact:
            if contact.dist < -1e-5:
                geom_a, geom_b = (int(g) for g in contact.geom)
                contacts.append({
                    "bodies": [model.body(int(model.geom_bodyid[g])).name for g in (geom_a, geom_b)],
                    "geoms": [model.geom(g).name or f"geom_{g}" for g in (geom_a, geom_b)],
                    "depth_m": float(-contact.dist),
                })
        rows.append({"label": label, "table_top_z": table_height, "source_hand_clearance_margin_m": hand_margin, "footprint_margin_m": layout.layout_margin(poses), "poses": {n: p.tolist() for n, p in poses.items()}, "penetrations": contacts})
    interface_env.close()
    result = {"scope": "CPU MuJoCo source-geometry screening only; PhysX/reach/recording pending", "scene": str(mjcf), "body_count": model.nbody, "geom_count": model.ngeom, "layouts": rows}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(args.output), "layouts": len(rows), "penetration_counts": [len(r["penetrations"]) for r in rows], "min_source_hand_margin_m": min(r["source_hand_clearance_margin_m"] for r in rows)}))


if __name__ == "__main__":
    main()
