#!/usr/bin/env python3
"""Bounded Isaac Sim 5.1 validation for B320-KA020.

The validator exercises the real collection environment, selected downloaded
visuals, collision/joint topology, contact truth, seeded resets, camera output,
and a ten-second zero-action rollout.  It does not synthesize task success.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import threading
import traceback
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
HANDSON_ROOT = REPO_ROOT / "handson"
for candidate in (REPO_ROOT, HANDSON_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

TASK_NAME = "t32_press_soap_pump"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--reset-count", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--images", action="store_true")
    return parser


def _unwrap(environment: Any) -> Any:
    current = environment
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if (
            getattr(current, "task_name", None) == TASK_NAME
            and hasattr(current, "_ka020_reset_poses")
        ):
            return current
        next_environment = getattr(current, "env", None)
        if next_environment is None:
            next_environment = getattr(current, "_env", None)
        if next_environment is None:
            break
        current = next_environment
    raise RuntimeError("Could not unwrap the B320-KA020 collection environment")


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _runtime_contract() -> dict[str, str]:
    isaac_sim_version = importlib.metadata.version("isaacsim")
    if isaac_sim_version != "5.1.0.0":
        raise RuntimeError(
            f"B320-KA020 requires Isaac Sim 5.1.0.0, got {isaac_sim_version}"
        )
    import isaaclab

    source = Path(isaaclab.__file__).resolve()
    version_path = next(
        (parent / "VERSION" for parent in source.parents if (parent / "VERSION").is_file()),
        None,
    )
    if version_path is None:
        raise RuntimeError(f"Could not locate IsaacLab VERSION from {source}")
    isaaclab_version = version_path.read_text(encoding="utf-8").strip()
    if isaaclab_version != "2.3.2":
        raise RuntimeError(
            f"B320-KA020 requires IsaacLab 2.3.2, got {isaaclab_version}"
        )
    return {
        "isaac_sim_version": isaac_sim_version,
        "isaaclab_version": isaaclab_version,
        "isaaclab_source": str(source),
    }


def _save_frame(path: Path, frame: Any) -> dict[str, Any]:
    from PIL import Image

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] < 3:
        raise RuntimeError(f"Unexpected camera frame {path.name}: {array.shape}")
    array = np.clip(array[..., :3], 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)
    metrics = {
        "path": str(path),
        "shape": list(array.shape),
        "mean_rgb": float(np.mean(array)),
        "black_fraction": float(np.mean(np.all(array == 0, axis=2))),
        "unique_rgb_count": int(
            np.unique(array.reshape(-1, 3), axis=0).shape[0]
        ),
        "channel_std": [
            float(value) for value in np.std(array, axis=(0, 1))
        ],
    }
    if (
        metrics["mean_rgb"] <= 10.0
        or metrics["black_fraction"] >= 0.98
        or metrics["unique_rgb_count"] < 128
        or max(metrics["channel_std"]) < 5.0
    ):
        raise RuntimeError(f"Camera {path.name} failed RGB checks: {metrics}")
    return metrics


def _visibility(prim: Any) -> str:
    from pxr import UsdGeom

    return str(UsdGeom.Imageable(prim).ComputeVisibility())


def _scene_contract(raw: Any) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdPhysics

    from handson.sim.isaaclab.b320_bimanual_scene import (
        B320_ASSET_ROOT_PATH,
        B320_TABLE_TOP_Z,
        KA020_FLOW_PROXY_PATH,
        KA020_MOVING_VISUAL_COMPONENT,
        KA020_PUMP_TRAVEL_M,
        KA020_SPONGE_REMOVED_HOLDER_TRIANGLE_COUNT,
        KA020_SPONGE_RENDER_COMPONENT_COUNT,
        KA020_SPONGE_RENDER_TRIANGLE_COUNT,
    )

    stage = raw._stage
    result = raw._b320_spawn_result
    expected_paths = {
        "sponge": f"{B320_ASSET_ROOT_PATH}/sponge",
        "soap_bottle": f"{B320_ASSET_ROOT_PATH}/soap_bottle",
        "soap_dispense_pad": f"{B320_ASSET_ROOT_PATH}/soap_dispense_pad",
    }
    if result.body_paths != expected_paths:
        raise RuntimeError(
            f"KA020 body-path contract changed: {result.body_paths!r}"
        )
    if result.collider_counts != {
        "sponge": 1,
        "soap_bottle": 2,
        "soap_dispense_pad": 1,
    }:
        raise RuntimeError(
            f"KA020 collider contract changed: {result.collider_counts!r}"
        )
    if result.articulation_root_paths:
        raise RuntimeError(
            "KA020 pump must use the Isaac Sim 5.1-safe maximal-coordinate "
            f"joint, got articulation roots {result.articulation_root_paths!r}"
        )
    expected_sponge_evidence = {
        "ka020_sponge_render_component_count": (
            KA020_SPONGE_RENDER_COMPONENT_COUNT
        ),
        "ka020_sponge_render_triangle_count": KA020_SPONGE_RENDER_TRIANGLE_COUNT,
        "ka020_sponge_removed_holder_triangle_count": (
            KA020_SPONGE_REMOVED_HOLDER_TRIANGLE_COUNT
        ),
        "ka020_sponge_holder_geometry_present": False,
    }
    for key, expected in expected_sponge_evidence.items():
        if result.evidence.get(key) != expected:
            raise RuntimeError(
                f"KA020 sponge-only evidence changed: {key}="
                f"{result.evidence.get(key)!r}, expected={expected!r}"
            )

    prim_contract: dict[str, Any] = {}
    for name, path in expected_paths.items():
        root = stage.GetPrimAtPath(path)
        if root is None or not root.IsValid():
            raise RuntimeError(f"KA020 body is missing: {path}")
        rigid_bodies = []
        colliders = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and not prim.IsInstanceProxy():
                rigid_bodies.append(str(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsInstanceProxy():
                colliders.append(str(prim.GetPath()))
        prim_contract[name] = {
            "rigid_bodies": rigid_bodies,
            "colliders": colliders,
        }
        if len(colliders) != result.collider_counts[name]:
            raise RuntimeError(
                f"KA020 {name} has {len(colliders)} colliders, expected "
                f"{result.collider_counts[name]}: {colliders}"
            )

    sponge_render_meshes = [
        UsdGeom.Mesh(prim)
        for prim in Usd.PrimRange(
            stage.GetPrimAtPath(expected_paths["sponge"]),
            Usd.TraverseInstanceProxies(),
        )
        if prim.IsA(UsdGeom.Mesh)
        and not prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    sponge_render_triangles = sum(
        len(mesh.GetFaceVertexCountsAttr().Get() or [])
        for mesh in sponge_render_meshes
    )
    if len(sponge_render_meshes) != 1 or (
        sponge_render_triangles != KA020_SPONGE_RENDER_TRIANGLE_COUNT
    ):
        raise RuntimeError(
            "KA020 sponge render must be the isolated one-mesh derivative: "
            f"meshes={len(sponge_render_meshes)} "
            f"triangles={sponge_render_triangles}"
        )
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=False,
    )

    def world_bounds(prim: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        low = np.asarray(tuple(aligned.GetMin()), dtype=np.float64)
        high = np.asarray(tuple(aligned.GetMax()), dtype=np.float64)
        return low, high, high - low

    render_low, render_high, render_extent = world_bounds(
        sponge_render_meshes[0].GetPrim()
    )
    collider_prim = stage.GetPrimAtPath(
        prim_contract["sponge"]["colliders"][0]
    )
    collider_low, collider_high, collider_extent = world_bounds(collider_prim)
    for label, extent in (
        ("render", render_extent),
        ("collider", collider_extent),
    ):
        planar = np.sort(extent[:2])
        if not (
            0.014 <= extent[2] <= 0.020
            and planar[0] >= 0.045
            and planar[1] >= 0.090
        ):
            raise RuntimeError(
                f"KA020 sponge-only {label} is not lying flat at the measured "
                f"94.9 x 53.4 x 16.5 mm scale: extent={extent.tolist()}"
            )
    if float(
        np.linalg.norm(
            (render_low + render_high) * 0.5
            - (collider_low + collider_high) * 0.5
        )
    ) > 0.004:
        raise RuntimeError(
            "KA020 sponge-only render/collider centers are misaligned: "
            f"render={[render_low.tolist(), render_high.tolist()]} "
            f"collider={[collider_low.tolist(), collider_high.tolist()]}"
        )

    joint_path = f"{B320_ASSET_ROOT_PATH}/soap_dispense_joint"
    joint_prim = stage.GetPrimAtPath(joint_path)
    if joint_prim is None or not joint_prim.IsA(UsdPhysics.PrismaticJoint):
        raise RuntimeError(f"KA020 physical pump joint is missing: {joint_path}")
    joint = UsdPhysics.PrismaticJoint(joint_prim)
    lower = float(joint.GetLowerLimitAttr().Get())
    upper = float(joint.GetUpperLimitAttr().Get())
    axis = str(joint.GetAxisAttr().Get())
    if not math.isclose(lower, -KA020_PUMP_TRAVEL_M, abs_tol=1.0e-6):
        raise RuntimeError(f"KA020 pump lower limit changed: {lower}")
    if not math.isclose(upper, 0.0, abs_tol=1.0e-6):
        raise RuntimeError(f"KA020 pump upper limit changed: {upper}")
    if axis != "Z":
        raise RuntimeError(f"KA020 pump must press vertically on axis Z, got {axis}")

    visual_truth: dict[str, Any] = {}
    visual_bounds: dict[str, Any] = {}
    for role in ("soap_bottle", "soap_dispense_pad"):
        root_path = f"{expected_paths[role]}/source_visual/asset"
        if role == "soap_bottle":
            root_path += "/root/GLTF_SceneRootNode"
        visual_root = stage.GetPrimAtPath(root_path)
        if visual_root is None or not visual_root.IsValid():
            raise RuntimeError(f"KA020 selected visual root is missing: {root_path}")
        if role == "soap_dispense_pad":
            component_visibility = {
                KA020_MOVING_VISUAL_COMPONENT: _visibility(visual_root)
            }
            visible = [KA020_MOVING_VISUAL_COMPONENT]
            visible_prims = [visual_root]
        else:
            component_visibility = {
                child.GetName(): _visibility(child)
                for child in visual_root.GetChildren()
            }
            visible = sorted(
                name
                for name, value in component_visibility.items()
                if value != "invisible"
            )
            expected_visible = sorted(
                name
                for name in component_visibility
                if name != KA020_MOVING_VISUAL_COMPONENT
            )
            visible_prims = [visual_root.GetChild(name) for name in visible]
            if visible != expected_visible:
                raise RuntimeError(
                    f"KA020 {role} selected-visual split changed: visible={visible}"
                )
        visual_truth[role] = component_visibility
        visible_component_bounds = [world_bounds(prim) for prim in visible_prims]
        visible_low = np.min(
            np.stack([bound[0] for bound in visible_component_bounds]), axis=0
        )
        visible_high = np.max(
            np.stack([bound[1] for bound in visible_component_bounds]), axis=0
        )
        visual_bounds[role] = {
            "low_m": visible_low,
            "high_m": visible_high,
            "extent_m": visible_high - visible_low,
        }

    def collider_union(role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bounds = [
            world_bounds(stage.GetPrimAtPath(path))
            for path in prim_contract[role]["colliders"]
        ]
        low = np.min(np.stack([bound[0] for bound in bounds]), axis=0)
        high = np.max(np.stack([bound[1] for bound in bounds]), axis=0)
        return low, high, high - low

    bottle_collider_low, bottle_collider_high, bottle_collider_extent = (
        collider_union("soap_bottle")
    )
    pump_collider_low, pump_collider_high, pump_collider_extent = collider_union(
        "soap_dispense_pad"
    )
    bottle_visual_low = visual_bounds["soap_bottle"]["low_m"]
    bottle_visual_high = visual_bounds["soap_bottle"]["high_m"]
    bottle_visual_extent = visual_bounds["soap_bottle"]["extent_m"]
    pump_visual_low = visual_bounds["soap_dispense_pad"]["low_m"]
    pump_visual_high = visual_bounds["soap_dispense_pad"]["high_m"]
    pump_visual_extent = visual_bounds["soap_dispense_pad"]["extent_m"]
    expected_support_z = B320_TABLE_TOP_Z + 0.002
    if not (
        bottle_visual_extent[2] >= 0.20
        and max(bottle_visual_extent[:2]) <= 0.14
        and abs(float(bottle_visual_low[2]) - expected_support_z) <= 0.003
        and abs(float(bottle_collider_low[2]) - expected_support_z) <= 0.003
    ):
        raise RuntimeError(
            "KA020 dispenser is not upright and supported on the tabletop: "
            f"visual={[bottle_visual_low.tolist(), bottle_visual_high.tolist()]} "
            f"collider={[bottle_collider_low.tolist(), bottle_collider_high.tolist()]} "
            f"table_top_z={B320_TABLE_TOP_Z}"
        )
    if not (
        np.max(np.abs(pump_visual_low - pump_collider_low)) <= 0.004
        and np.max(np.abs(pump_visual_high - pump_collider_high)) <= 0.004
        and 0.016 <= pump_collider_extent[2] <= 0.024
        and pump_collider_low[2] >= bottle_collider_high[2]
        and pump_collider_high[2] >= B320_TABLE_TOP_Z + 0.22
    ):
        raise RuntimeError(
            "KA020 pump head is not aligned with an accessible top collider: "
            f"visual={[pump_visual_low.tolist(), pump_visual_high.tolist()]} "
            f"collider={[pump_collider_low.tolist(), pump_collider_high.tolist()]} "
            f"bottle_collider_high={bottle_collider_high.tolist()}"
        )

    flow = stage.GetPrimAtPath(KA020_FLOW_PROXY_PATH)
    if flow is None or not flow.IsValid():
        raise RuntimeError("KA020 render-only soap-flow proxy is missing")
    if flow.HasAPI(UsdPhysics.RigidBodyAPI) or flow.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("KA020 soap-flow proxy must remain render-only")
    if _visibility(flow) != "invisible":
        raise RuntimeError("KA020 soap-flow proxy must start hidden")
    if not any(
        prim.IsA(UsdGeom.Gprim)
        for prim in Usd.PrimRange(flow, Usd.TraverseInstanceProxies())
    ):
        raise RuntimeError("KA020 soap-flow proxy has no renderable geometry")

    if set(raw._b320_contact_views) != {"left", "right"}:
        raise RuntimeError(
            f"KA020 hand-object contact views are incomplete: "
            f"{sorted(raw._b320_contact_views)}"
        )
    if raw._ka020_pump_contact_view is None:
        raise RuntimeError("KA020 right-hand/pump-head contact truth is unavailable")
    pump_fraction, pump_fraction_readable = raw._dof_fraction(
        "soap_dispense_joint"
    )
    if not pump_fraction_readable or not math.isfinite(pump_fraction):
        raise RuntimeError("KA020 pump relative-pose travel readback is unavailable")

    return {
        "body_paths": result.body_paths,
        "collider_counts": result.collider_counts,
        "prim_contract": prim_contract,
        "sponge_render_mesh_count": len(sponge_render_meshes),
        "sponge_render_triangle_count": sponge_render_triangles,
        "sponge_render_world_bounds_m": [render_low, render_high],
        "sponge_render_world_extent_m": render_extent,
        "sponge_collider_world_bounds_m": [collider_low, collider_high],
        "sponge_collider_world_extent_m": collider_extent,
        "joint": {
            "path": joint_path,
            "lower_m": lower,
            "upper_m": upper,
            "axis": axis,
            "body0": [str(path) for path in joint.GetBody0Rel().GetTargets()],
            "body1": [str(path) for path in joint.GetBody1Rel().GetTargets()],
        },
        "visual_component_visibility": visual_truth,
        "dispenser_geometry": {
            "bottle_visual_bounds_m": [bottle_visual_low, bottle_visual_high],
            "bottle_visual_extent_m": bottle_visual_extent,
            "bottle_collider_bounds_m": [
                bottle_collider_low,
                bottle_collider_high,
            ],
            "bottle_collider_extent_m": bottle_collider_extent,
            "pump_visual_bounds_m": [pump_visual_low, pump_visual_high],
            "pump_visual_extent_m": pump_visual_extent,
            "pump_collider_bounds_m": [pump_collider_low, pump_collider_high],
            "pump_collider_extent_m": pump_collider_extent,
            "table_top_z_m": B320_TABLE_TOP_Z,
        },
        "flow_proxy_path": KA020_FLOW_PROXY_PATH,
        "spawn_evidence": result.evidence,
        "hand_contact_view_count": len(raw._b320_contact_views),
        "pump_contact_view": True,
        "pump_fraction_readback": float(pump_fraction),
    }


def _reset_record(raw: Any, *, seed: int) -> dict[str, Any]:
    authored = {
        spec.name: np.asarray(
            [*spec.home_translation, *spec.home_orientation_wxyz],
            dtype=np.float64,
        )
        for spec in raw._b320_contract.object_specs
    }
    sampled = {
        name: np.asarray(pose, dtype=np.float64).copy()
        for name, pose in raw._ka020_reset_poses.items()
    }
    if not raw._ka020_reset_layout_is_valid(sampled, authored):
        raise RuntimeError(f"KA020 seed {seed} failed reset layout validation")
    physical = {
        name: np.asarray(raw._physics_body_pose(name), dtype=np.float64)
        for name in ("sponge", "soap_bottle", "soap_dispense_pad")
    }
    if not all(np.all(np.isfinite(pose)) for pose in physical.values()):
        raise RuntimeError(f"KA020 seed {seed} produced a non-finite physical pose")
    for name in ("sponge", "soap_bottle"):
        error = float(np.linalg.norm(physical[name][:3] - sampled[name][:3]))
        if error > 0.010:
            raise RuntimeError(
                f"KA020 seed {seed} reset mismatch for {name}: {error:.6f} m"
            )
    return {
        "seed": int(seed),
        "attempts": int(raw._ka020_reset_attempts),
        "sampled_pose_wxyz": sampled,
        "physical_pose_wxyz": physical,
        "pair_separation_m": float(
            np.linalg.norm(sampled["sponge"][:2] - sampled["soap_bottle"][:2])
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.steps < 1 or args.steps > 900:
        raise ValueError("--steps must be in [1, 900]")
    if args.reset_count < 1 or args.reset_count > 16:
        raise ValueError("--reset-count must be in [1, 16]")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    os.environ["HANDSON_ENABLE_ISAACSIM_NEWTON"] = "0"
    os.environ["HANDSON_STABLE_RECORDING_ORIGIN"] = "1"
    os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
    os.environ["HANDSON_ISAACLAB_DEFER_ARM_RENDER"] = "0"
    os.environ["HANDSON_ISAACLAB_DIRECT_HAND_CONTROL"] = "0"
    os.environ.pop("HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY", None)
    os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = (
        "0" if args.images else "1"
    )
    if args.images:
        os.environ["HANDSON_ISAACSIM_ACCURATE_CAMERA_VISIBILITY"] = "1"

    runtime = _runtime_contract()
    from handson.tasks.mappings import CONFIG_MAPPING

    config = CONFIG_MAPPING[TASK_NAME]()
    environment = config.get_native_environment(
        policy_mode=True,
        sim_backend="isaacsim",
        launch_isaac=True,
        image_obs=bool(args.images),
        randomize=True,
        randomize_dynamics=False,
        render_mode="rgb_array",
        headless=True,
        isaac_device=args.device,
        hz=30,
        control_dt=1.0 / 30.0,
        physics_dt=1.0 / 120.0,
        time_limit=30.0,
    )
    raw = _unwrap(environment)
    raw._fast_shutdown_on_close = True
    try:
        resets = []
        reset_info: dict[str, Any] = {}
        for offset in range(args.reset_count):
            _observation, reset_info = environment.reset(seed=args.seed + offset)
            resets.append(_reset_record(raw, seed=args.seed + offset))

        scene = _scene_contract(raw)
        initial = {
            name: np.asarray(raw._physics_body_pose(name), dtype=np.float64)
            for name in ("sponge", "soap_bottle", "soap_dispense_pad")
        }
        initial_dof = np.asarray(
            raw._dof_fraction("soap_dispense_joint"),
            dtype=np.float64,
        )
        zero_action = np.zeros(environment.action_space.shape, dtype=np.float32)
        final_info: dict[str, Any] = {}
        for step_index in range(args.steps):
            observation, reward, terminated, truncated, final_info = environment.step(
                zero_action
            )
            flat_observation = np.concatenate(
                [np.asarray(value).reshape(-1) for value in observation.values()]
                if isinstance(observation, dict)
                else [np.asarray(observation).reshape(-1)]
            )
            if not np.all(np.isfinite(flat_observation)):
                raise RuntimeError(
                    f"KA020 observation became non-finite at step {step_index}"
                )
            if reward != 0.0 or final_info.get("succeed"):
                raise RuntimeError(
                    f"KA020 zero action produced false success at step {step_index}"
                )
            if terminated or truncated:
                raise RuntimeError(
                    f"KA020 zero action terminated at step {step_index}: {final_info}"
                )

        final = {
            name: np.asarray(raw._physics_body_pose(name), dtype=np.float64)
            for name in initial
        }
        final_dof = np.asarray(
            raw._dof_fraction("soap_dispense_joint"),
            dtype=np.float64,
        )
        if not all(np.all(np.isfinite(pose)) for pose in final.values()):
            raise RuntimeError("KA020 zero-action pose state is non-finite")
        if not np.all(np.isfinite(np.concatenate([initial_dof, final_dof]))):
            raise RuntimeError("KA020 pump state is non-finite")
        drift = {
            name: float(np.linalg.norm(final[name][:3] - initial[name][:3]))
            for name in initial
        }
        print(
            "[B320-KA020] zero-action poses "
            f"initial={_json_safe(initial)} final={_json_safe(final)} "
            f"drift={drift}",
            flush=True,
        )
        for name in ("sponge", "soap_bottle"):
            if drift[name] > 0.020:
                raise RuntimeError(
                    f"KA020 {name} zero-action drift exceeded 20 mm: "
                    f"{drift[name]:.6f} m"
                )
        pump_drift = float(
            abs(final_dof[0] - initial_dof[0])
            * abs(float(scene["joint"]["lower_m"]))
        )
        if pump_drift > 0.002:
            raise RuntimeError(
                f"KA020 pump zero-action drift exceeded 2 mm: {pump_drift:.6f} m"
            )

        pump_view = raw._task_body_views.get("soap_dispense_pad")
        if pump_view is None or raw._torch is None:
            raise RuntimeError("KA020 pump rigid-body view is unavailable")
        # Install the wrench through the environment's pre-substep hook so the
        # same fingertip load reaches all four 120 Hz PhysX substeps.  Applying
        # it once before ``environment.step`` only affects the first substep and
        # is not a valid sustained-contact test.
        pump_press_force_n = 6.0
        previous_pre_substep = getattr(
            raw,
            "_handson_pre_physics_substep_callback",
            None,
        )

        def _apply_pump_press(
            _substep_index: int,
            _substep_count: int,
        ) -> None:
            if callable(previous_pre_substep):
                previous_pre_substep(_substep_index, _substep_count)
            velocity_template = pump_view.get_velocities()
            forces = raw._torch.zeros(
                (int(pump_view.count), 3),
                dtype=velocity_template.dtype,
                device=velocity_template.device,
            )
            forces[:, 2] = -pump_press_force_n
            torques = raw._torch.zeros_like(forces)
            indices = raw._torch.arange(
                int(pump_view.count),
                dtype=raw._torch.long,
                device=forces.device,
            )
            pump_view.apply_forces_and_torques_at_position(
                forces,
                torques,
                None,
                indices,
                True,
            )

        pump_fractions: list[float] = []
        raw._handson_pre_physics_substep_callback = _apply_pump_press
        try:
            for _ in range(18):
                environment.step(zero_action)
                fraction, readable = raw._dof_fraction("soap_dispense_joint")
                if not readable:
                    raise RuntimeError(
                        "KA020 pump became unreadable during force test"
                    )
                pump_fractions.append(float(fraction))
        finally:
            raw._handson_pre_physics_substep_callback = previous_pre_substep
        peak_pump_fraction = max(pump_fractions, default=0.0)
        if peak_pump_fraction < 0.75:
            raise RuntimeError(
                "KA020 pump did not reach 75% travel under a 6 N sustained "
                "top press: "
                f"peak_fraction={peak_pump_fraction:.6f}"
            )
        release_fractions: list[float] = []
        for _ in range(30):
            environment.step(zero_action)
            fraction, readable = raw._dof_fraction("soap_dispense_joint")
            if not readable:
                raise RuntimeError("KA020 pump became unreadable during release test")
            release_fractions.append(float(fraction))
        released_pump_fraction = release_fractions[-1]
        if released_pump_fraction > 0.15:
            raise RuntimeError(
                "KA020 pump did not return after release: "
                f"fraction={released_pump_fraction:.6f}"
            )

        frames: dict[str, Any] = {}
        if args.images:
            rendered = tuple(raw.render())
            if len(rendered) != 3:
                raise RuntimeError(
                    "KA020 render did not return three camera frames: "
                    f"count={len(rendered)}"
                )
            frames = {
                name: _save_frame(output_dir / f"{name}.png", frame)
                for name, frame in zip(
                    ("head", "wrist_left", "wrist_right"),
                    rendered,
                    strict=True,
                )
            }

        metrics, flags = raw._soap_metrics()
        result = {
            "schema": "b320_ka020_live_validation_v1",
            "task_name": TASK_NAME,
            "runtime": runtime,
            "seed": int(args.seed),
            "reset_count": int(args.reset_count),
            "resets": resets,
            "scene": scene,
            "zero_action": {
                "steps": int(args.steps),
                "seconds": float(args.steps / 30.0),
                "initial_pose_wxyz": initial,
                "final_pose_wxyz": final,
                "translation_drift_m": drift,
                "initial_pump_fraction": float(initial_dof[0]),
                "initial_pump_readable": bool(initial_dof[1]),
                "final_pump_fraction": float(final_dof[0]),
                "final_pump_readable": bool(final_dof[1]),
                "pump_drift_m": pump_drift,
                "false_success": False,
                "terminated": False,
                "truncated": False,
            },
            "pump_press": {
                "sustained_force_n": pump_press_force_n,
                "press_steps": 18,
                "peak_fraction": peak_pump_fraction,
                "release_steps": 30,
                "released_fraction": released_pump_fraction,
            },
            "current_metrics": metrics,
            "current_success_flags": flags,
            "reset_info": reset_info,
            "final_info": final_info,
            "frames": frames,
            "pass": True,
        }
        path = output_dir / "validation.json"
        path.write_text(
            json.dumps(_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(f"VALIDATION_JSON={path}", flush=True)
        print(
            "B320_KA020_LIVE_VALIDATION_PASSED "
            f"steps={args.steps} resets={args.reset_count} "
            f"drift_m={drift} pump_drift_m={pump_drift:.8f}",
            flush=True,
        )
        return 0
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # The fast Kit close path exits with status zero before Python can
        # propagate an exception.  Preserve fail-closed validator semantics.
        os._exit(1)
    finally:
        close_exit_code = 0 if sys.exc_info()[0] is None else 1
        sys.stdout.flush()
        sys.stderr.flush()
        shutdown_guard = threading.Timer(
            10.0,
            lambda: os._exit(close_exit_code),
        )
        shutdown_guard.daemon = True
        shutdown_guard.start()
        try:
            environment.close()
        finally:
            shutdown_guard.cancel()


if __name__ == "__main__":
    raise SystemExit(main())
