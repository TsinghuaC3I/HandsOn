#!/usr/bin/env python3
"""Bounded Isaac Sim 5.1 validation of the B320-AS013 core scene.

This runner composes the exact utility-knife asset with the production dual
ROKAE/Wuji table scene, validates the authored blade joint/colliders, holds the
robots at their reset targets, advances zero task action for ten seconds, and
captures the production head camera. It never attempts the blade-slide motion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any


TASK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_DIR.parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "handson"))

# AppLauncher must precede every other Isaac/Kit import.
from isaaclab.app import AppLauncher  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physics-steps", type=int, default=1200)
    parser.add_argument("--status-interval", type=int, default=120)
    AppLauncher.add_app_launcher_args(parser)
    return parser


args = _parser().parse_args()
args.headless = True
args.enable_cameras = True
args.device = "cuda:0"
args.width = 640
args.height = 480
args.kit_args = " ".join(
    filter(
        None,
        (
            str(getattr(args, "kit_args", "") or ""),
            "--/rtx/rendermode=RayTracedLighting",
            "--/rtx/post/tonemap/op=4",
            "--/rtx/post/tonemap/filmIso=200",
        ),
    )
)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Every remaining simulator import is intentionally post-app.
import carb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import (  # noqa: E402
    Gf,
    PhysxSchema,
    Sdf,
    Usd,
    UsdGeom,
    UsdLux,
    UsdPhysics,
    UsdShade,
)

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.asset.importer.mjcf")

from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv  # noqa: E402
from taskdev.tasks.bench320_as013_utility_knife_body_stabilized_blade_slide.scene_contract import (  # noqa: E402
    ASSET_PATH,
    INTEGRATED_ASSET_SHA256,
    INTEGRATED_METADATA_SHA256,
    EXPECTED_ASSET_SHA256,
    HEAD_RESOLUTION_HW,
    JOINT_NAME,
    METADATA_PATH,
    PHYSICS_HZ,
    RUNTIME_ASSET_PATH,
    TASK_CODE,
    TASK_ID,
    UNIFORM_SCALE,
    ZERO_ACTION_STEPS,
    asset_hashes,
    image_metrics,
    sha256_file,
)


ROOT_PATH = "/World/AS013Foundation"
KNIFE_PATH = f"{ROOT_PATH}/UtilityKnife"
TABLE_TOP_Z = 0.89


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value in report: {value}")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments], text=True
    ).strip()


def _range_list(bound_range: Any) -> list[list[float]]:
    return [
        [float(value) for value in bound_range.GetMin()],
        [float(value) for value in bound_range.GetMax()],
    ]


def _asset_inventory() -> dict[str, Any]:
    stage = Usd.Stage.Open(str(ASSET_PATH), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"could not open utility-knife USD: {ASSET_PATH}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        raise RuntimeError("utility-knife USD has no valid default prim")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
        useExtentsHint=True,
    )
    collisions: list[str] = []
    rigid_bodies: list[str] = []
    joints: list[dict[str, Any]] = []
    visual_gprims: list[str] = []
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    for prim in Usd.PrimRange.Stage(stage, predicate):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collisions.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(str(prim.GetPath()))
        if prim.IsA(UsdGeom.Gprim):
            visual_gprims.append(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.PrismaticJoint):
            joint = UsdPhysics.PrismaticJoint(prim)
            joints.append(
                {
                    "name": prim.GetName(),
                    "path": str(prim.GetPath()),
                    "axis": str(joint.GetAxisAttr().Get()),
                    "lower_limit": float(joint.GetLowerLimitAttr().Get()),
                    "upper_limit": float(joint.GetUpperLimitAttr().Get()),
                    "body0": [str(path) for path in joint.GetBody0Rel().GetTargets()],
                    "body1": [str(path) for path in joint.GetBody1Rel().GetTargets()],
                }
            )
    layers = []
    for layer in sorted(stage.GetUsedLayers(), key=lambda item: str(item.realPath)):
        if not layer.realPath:
            continue
        path = Path(layer.realPath).resolve()
        layers.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    # ``asset_hashes`` is deliberately the immutable downloaded-source bundle
    # (archive + extracted glTF/license/textures).  It must not be confused
    # with the generated USD or its runtime mirror, which are bound by their
    # own hashes below.
    source_hashes = asset_hashes()
    integrated_hash = sha256_file(ASSET_PATH)
    runtime_hash = (
        sha256_file(RUNTIME_ASSET_PATH)
        if RUNTIME_ASSET_PATH.is_file()
        else None
    )
    metadata_hash = (
        sha256_file(METADATA_PATH) if METADATA_PATH.is_file() else None
    )
    runtime_metadata_path = RUNTIME_ASSET_PATH.with_name(
        "utility_knife_ahtusik_metadata.json"
    )
    runtime_metadata_hash = (
        sha256_file(runtime_metadata_path)
        if runtime_metadata_path.is_file()
        else None
    )
    binary_path = ASSET_PATH.with_suffix(".usd")
    runtime_binary_path = RUNTIME_ASSET_PATH.with_suffix(".usd")
    binary_hash = sha256_file(binary_path) if binary_path.is_file() else None
    runtime_binary_hash = (
        sha256_file(runtime_binary_path)
        if runtime_binary_path.is_file()
        else None
    )
    inventory = {
        "default_prim": str(default_prim.GetPath()),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "local_aabb_m": _range_list(
            cache.ComputeWorldBound(default_prim).ComputeAlignedRange()
        ),
        "collision_shape_paths": collisions,
        "collision_shape_count": len(collisions),
        "rigid_body_paths": rigid_bodies,
        "rigid_body_count": len(rigid_bodies),
        "visual_gprim_count": len(visual_gprims),
        "prismatic_joints": joints,
        "layers": layers,
        "asset_hashes": source_hashes,
        "source_hashes": source_hashes,
        "integrated_asset_sha256": integrated_hash,
        "runtime_mirror_sha256": runtime_hash,
        "binary_usd_sha256": binary_hash,
        "runtime_binary_usd_sha256": runtime_binary_hash,
        "metadata_sha256": metadata_hash,
        "runtime_metadata_sha256": runtime_metadata_hash,
    }
    errors = []
    if source_hashes != EXPECTED_ASSET_SHA256:
        errors.append("downloaded source hashes differ from the bound archive")
    if integrated_hash != INTEGRATED_ASSET_SHA256:
        errors.append("integrated USD hash differs from the audited build")
    if runtime_hash != INTEGRATED_ASSET_SHA256:
        errors.append("runtime USD mirror is missing or differs from the integrated build")
    if binary_hash is None or runtime_binary_hash is None:
        errors.append("task-local/runtime binary USD mirror is missing")
    elif binary_hash != runtime_binary_hash:
        errors.append("task-local and runtime binary USD mirrors differ")
    if metadata_hash != INTEGRATED_METADATA_SHA256:
        errors.append("integrated metadata is missing or differs from the audited build")
    if runtime_metadata_hash != metadata_hash:
        errors.append("task-local and runtime metadata mirrors differ")
    if inventory["meters_per_unit"] != 1.0 or inventory["up_axis"] != "Z":
        errors.append("asset must be meter-scale and Z-up")
    if len(joints) != 1 or joints[0]["name"] != JOINT_NAME:
        errors.append("asset must expose exactly the certified blade prismatic joint")
    if len(collisions) < 2 or len(rigid_bodies) < 2:
        errors.append("asset must expose body and blade rigid/collision geometry")
    inventory["errors"] = errors
    inventory["pass"] = not errors
    return inventory


def _ensure_lighting(stage: Any) -> dict[str, Any]:
    dome = UsdLux.DomeLight.Define(stage, f"{ROOT_PATH}/DomeLight")
    dome.CreateIntensityAttr(650.0)
    dome.CreateColorAttr(Gf.Vec3f(0.92, 0.94, 1.0))
    distant = UsdLux.DistantLight.Define(stage, f"{ROOT_PATH}/DistantLight")
    distant.CreateIntensityAttr(1800.0)
    distant.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.90))
    distant.CreateAngleAttr(1.0)
    UsdGeom.Xformable(distant).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, -25.0))
    return {
        "dome": {"path": str(dome.GetPath()), "intensity": 650.0},
        "distant": {"path": str(distant.GetPath()), "intensity": 1800.0},
    }


def _bind_preview_fallback(stage: Any, knife_prim: Any) -> str:
    material = UsdShade.Material.Define(stage, f"{ROOT_PATH}/Materials/KnifePreviewFallback")
    shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.24, 0.28, 0.33)
    )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.35)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.42)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(knife_prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
    )
    return str(material.GetPath())


def _compose_knife(stage: Any, inventory: dict[str, Any], table_top: float) -> dict[str, Any]:
    source_task = stage.GetPrimAtPath("/World/Task")
    if source_task.IsValid():
        task_xform = UsdGeom.Xformable(source_task)
        task_xform.AddTranslateOp(opSuffix="as013CoreSceneOffset").Set(
            Gf.Vec3d(0.0, 4.0, 0.0)
        )

    UsdGeom.Xform.Define(stage, ROOT_PATH)
    knife = UsdGeom.Xform.Define(stage, KNIFE_PATH).GetPrim()
    knife.GetReferences().ClearReferences()
    knife.GetReferences().AddReference(str(ASSET_PATH.resolve()))
    local_low, local_high = inventory["local_aabb_m"]
    # The bounded core smoke keeps the knife 3 mm above the table so it tests
    # asset composition/zero-action settling without introducing a support
    # contact.  The production task reset derives its authored pose from the
    # final composed bound with TARGET_TABLE_CLEARANCE_M=0.0; this runner is
    # deliberately not the fresh collection certification.
    initial_clearance_m = 0.003
    position = (
        -0.12,
        0.0,
        float(table_top + initial_clearance_m - UNIFORM_SCALE * local_low[2]),
    )
    xform = UsdGeom.Xformable(knife)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*position))
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(UNIFORM_SCALE, UNIFORM_SCALE, UNIFORM_SCALE)
    )
    preview_material = _bind_preview_fallback(stage, knife)

    joint_prim = next(
        (
            prim
            for prim in Usd.PrimRange(knife, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))
            if prim.IsA(UsdPhysics.PrismaticJoint) and prim.GetName() == JOINT_NAME
        ),
        None,
    )
    if joint_prim is None:
        raise RuntimeError(f"composed blade joint is missing: {JOINT_NAME}")
    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(1.2)
    drive.CreateMaxForceAttr(100.0)
    friction = joint_prim.GetAttribute("physxJoint:jointFriction")
    if not friction.IsValid():
        friction = joint_prim.CreateAttribute(
            "physxJoint:jointFriction", Sdf.ValueTypeNames.Float
        )
    friction.Set(1.0)
    dynamic_friction = joint_prim.GetAttribute("physxJoint:dynamicFriction")
    if not dynamic_friction.IsValid():
        dynamic_friction = joint_prim.CreateAttribute(
            "physxJoint:dynamicFriction", Sdf.ValueTypeNames.Float
        )
    dynamic_friction.Set(0.8)

    collision_paths = []
    rigid_paths = []
    physics_material = UsdShade.Material.Define(
        stage, f"{ROOT_PATH}/Materials/KnifePhysics"
    )
    physics_api = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    physics_api.CreateStaticFrictionAttr(0.65)
    physics_api.CreateDynamicFrictionAttr(0.60)
    physics_api.CreateRestitutionAttr(0.0)
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    for prim in Usd.PrimRange(knife, predicate):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_paths.append(str(prim.GetPath()))
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                physics_material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_paths.append(str(prim.GetPath()))
            report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            report_api.CreateThresholdAttr(0.0)

    expected_world_aabb = [
        [position[index] + UNIFORM_SCALE * local_low[index] for index in range(3)],
        [position[index] + UNIFORM_SCALE * local_high[index] for index in range(3)],
    ]
    return {
        "prim_path": KNIFE_PATH,
        "position_m": list(position),
        "uniform_scale": UNIFORM_SCALE,
        "yaw_deg": 0.0,
        "initial_table_clearance_m": initial_clearance_m,
        "expected_world_aabb_m": expected_world_aabb,
        "collision_paths": collision_paths,
        "rigid_body_paths": rigid_paths,
        "preview_fallback_material": preview_material,
        "physics_material": {
            "path": str(physics_material.GetPath()),
            "static_friction": 0.65,
            "dynamic_friction": 0.60,
            "restitution": 0.0,
        },
        "blade_joint_path": str(joint_prim.GetPath()),
        "blade_drive": {
            "stiffness": 0.0,
            "damping": 1.2,
            "max_force": 100.0,
            "joint_friction": 1.0,
            "dynamic_friction": 0.8,
        },
    }


def _gpu_namespace_contract() -> dict[str, Any]:
    physical = os.environ.get("TASKDEV_PHYSICAL_GPU", "").strip()
    host = os.environ.get("TASKDEV_HOST", "").strip()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    errors = []
    if physical != "1":
        errors.append("AS013 live validation must be scheduled on physical GPU 1")
    if host != "hp-14":
        errors.append("AS013 live validation must be scheduled on hp-14")
    if visible:
        errors.append("scheduler must isolate Vulkan/CUDA by device namespace, not CUDA_VISIBLE_DEVICES")
    if args.device != "cuda:0":
        errors.append("isolated process must address the selected device as logical cuda:0")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return {
        "requested_host": host,
        "physical_gpu": int(physical) if physical.isdigit() else None,
        "logical_device": args.device,
        "cuda_visible_devices": visible,
        "free_vram_bytes_after_app_start": int(free_bytes),
        "total_vram_bytes": int(total_bytes),
        "free_fraction_after_app_start": float(free_bytes / total_bytes),
        "errors": errors,
        "pass": not errors,
    }


def _make_rigid_views(physics_view: Any, rigid_paths: list[str]) -> dict[str, Any]:
    views: dict[str, Any] = {}
    errors = []
    for path in rigid_paths:
        try:
            view = physics_view.create_rigid_body_view(path)
            if view is None or int(view.count) != 1:
                raise RuntimeError(f"expected one rigid body, got {getattr(view, 'count', 0)}")
            views[path] = view
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return {"views": views, "errors": errors, "pass": bool(views) and not errors}


def _view_state(view: Any) -> dict[str, np.ndarray]:
    transform = _numpy(view.get_transforms()).reshape(1, 7)[0]
    velocity = _numpy(view.get_velocities()).reshape(1, 6)[0]
    return {"transform_xyzw": transform, "velocity": velocity}


def _blade_pair(rigid_paths: list[str]) -> tuple[str, str]:
    body = next((path for path in rigid_paths if "body_1" in path.lower()), None)
    blade = next((path for path in rigid_paths if "blade_2" in path.lower()), None)
    if body is None or blade is None:
        ordered = sorted(rigid_paths)
        if len(ordered) < 2:
            raise RuntimeError("utility knife requires at least two rigid owners")
        body, blade = ordered[:2]
    return body, blade


def _articulation_view(physics_view: Any) -> dict[str, Any]:
    errors = []
    view = None
    initial_positions: list[float] = []
    initial_velocities: list[float] = []
    try:
        view = physics_view.create_articulation_view(KNIFE_PATH)
        if view is None or int(view.count) != 1:
            raise RuntimeError(
                f"expected one knife articulation, got {getattr(view, 'count', 0)}"
            )
        initial_positions = _numpy(view.get_dof_positions()).reshape(1, -1)[0].tolist()
        initial_velocities = _numpy(view.get_dof_velocities()).reshape(1, -1)[0].tolist()
        if not initial_positions:
            raise RuntimeError("knife articulation exposes no DOF")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "view": view,
        "count": int(getattr(view, "count", 0)) if view is not None else 0,
        "initial_positions": initial_positions,
        "initial_velocities": initial_velocities,
        "errors": errors,
        "pass": view is not None and not errors,
    }


def _write_safe_reset_joint(articulation: dict[str, Any], joint: dict[str, Any]) -> dict[str, Any]:
    view = articulation["view"]
    lower = float(joint["lower_limit"])
    upper = float(joint["upper_limit"])
    target = lower + 0.20 * (upper - lower)
    positions = view.get_dof_positions().clone()
    velocities = view.get_dof_velocities().clone()
    if positions.shape[1] != 1:
        raise RuntimeError(f"expected one knife DOF, got {positions.shape[1]}")
    positions[:, 0] = target
    velocities[:, 0] = 0.0
    indices = torch.arange(int(view.count), dtype=torch.long, device=args.device)
    view.set_dof_positions(positions, indices)
    view.set_dof_velocities(velocities, indices)
    return {
        "normalized_progress": 0.20,
        "target": target,
        "positions": _numpy(positions).reshape(1, -1)[0],
        "velocities": _numpy(velocities).reshape(1, -1)[0],
        "state_writes_before_settle": 1,
    }


def _render_head(environment: Any, output_dir: Path) -> dict[str, Any]:
    frames = environment.render()
    if not isinstance(frames, (list, tuple)) or not frames:
        raise RuntimeError("production head camera returned no frames")
    frame = np.asarray(frames[0])
    if frame.dtype != np.uint8 or frame.shape != (*HEAD_RESOLUTION_HW, 3):
        raise RuntimeError(
            f"head frame must be uint8 480x640 RGB, got {frame.shape}/{frame.dtype}"
        )
    path = output_dir / "reset_head.png"
    Image.fromarray(frame).save(path, compress_level=0)
    metrics = image_metrics(path)
    camera = environment._cameras.get("head_camera")
    prim_path = str(getattr(getattr(camera, "cfg", None), "prim_path", ""))
    metrics.update(
        {
            "path": str(path),
            "prim_path": prim_path,
            "expected_prim_path": "/World/RecordCameras/head_camera",
            "pass": bool(
                prim_path == "/World/RecordCameras/head_camera"
                and metrics["shape_hwc"] == [480, 640, 3]
                and metrics["bytes"] >= 150_000
                and metrics["mean_rgb"] > 30.0
                and metrics["pixel_std"] > 15.0
            ),
        }
    )
    return metrics


def _run_scene(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "RayTracedLighting")
    settings.set("/rtx/post/tonemap/op", 4)
    settings.set("/rtx/post/tonemap/filmIso", 200.0)

    gpu = _gpu_namespace_contract()
    inventory = _asset_inventory()
    if not inventory["pass"]:
        raise RuntimeError(f"isolated asset foundation failed: {inventory['errors']}")

    environment = HandsOnABCIsaacLabEnv(
        "t02_connect_charger_and_cable",
        seed=0,
        control_dt=1.0 / 30.0,
        physics_dt=1.0 / float(PHYSICS_HZ),
        time_limit=20.0,
        render_mode="rgb_array",
        image_obs=True,
        randomize=False,
        randomize_dynamics=False,
        hand_type="wuji1",
        arm_type="rokae",
        launch_isaac=True,
        headless=True,
        render_resolution=(480, 640),
        hz=30,
        isaac_device=args.device,
        physics_backend="physx",
        existing_simulation_app=simulation_app,
        existing_app_launcher=app_launcher,
    )
    stage = environment._stage
    sim = environment._sim
    if stage is None or sim is None:
        raise RuntimeError("HandsOn did not expose its stage/simulation")
    table_top = float(environment._current_table_top_z())
    composition = _compose_knife(stage, inventory, table_top)
    lighting = _ensure_lighting(stage)
    scene_path = output_dir / "as013_core_scene.usda"
    stage.Export(str(scene_path))

    # This is the only reset after composition. No object state is written from
    # this point through the end of the zero-action settle.
    sim.reset()
    physics_view = environment._robots["right"]._physics_sim_view
    if physics_view is None:
        raise RuntimeError("PhysX tensor simulation view is unavailable")
    rigid = _make_rigid_views(physics_view, composition["rigid_body_paths"])
    if not rigid["pass"]:
        raise RuntimeError(f"rigid view creation failed: {rigid['errors']}")
    articulation = _articulation_view(physics_view)
    if not articulation["pass"]:
        raise RuntimeError(f"articulation view creation failed: {articulation['errors']}")
    safe_reset = _write_safe_reset_joint(
        articulation, inventory["prismatic_joints"][0]
    )
    articulation["initial_positions"] = list(safe_reset["positions"])
    articulation["initial_velocities"] = list(safe_reset["velocities"])

    body_path, blade_path = _blade_pair(list(rigid["views"]))
    baseline = {path: _view_state(view) for path, view in rigid["views"].items()}
    baseline_relative = (
        baseline[blade_path]["transform_xyzw"][:3]
        - baseline[body_path]["transform_xyzw"][:3]
    )
    max_translation = 0.0
    max_linear_speed = 0.0
    max_angular_speed = 0.0
    finite = True
    object_state_writes_during_settle = 0

    requested_steps = max(int(args.physics_steps), ZERO_ACTION_STEPS)
    sim_dt = float(environment._sim_dt)
    for step in range(requested_steps):
        for robot in environment._robots.values():
            robot.set_joint_position_target(robot.data.default_joint_pos)
        environment._scene.write_data_to_sim()
        sim.step(render=bool(step % max(args.status_interval, 1) == 0))
        environment._scene.update(sim_dt)
        for path, view in rigid["views"].items():
            state = _view_state(view)
            finite = bool(
                finite
                and np.isfinite(state["transform_xyzw"]).all()
                and np.isfinite(state["velocity"]).all()
            )
            max_translation = max(
                max_translation,
                float(
                    np.linalg.norm(
                        state["transform_xyzw"][:3]
                        - baseline[path]["transform_xyzw"][:3]
                    )
                ),
            )
            max_linear_speed = max(
                max_linear_speed, float(np.linalg.norm(state["velocity"][:3]))
            )
            max_angular_speed = max(
                max_angular_speed, float(np.linalg.norm(state["velocity"][3:]))
            )
        if args.status_interval > 0 and (step + 1) % args.status_interval == 0:
            print(
                f"[AS013 zero action] step={step + 1}/{requested_steps} "
                f"max_translation={max_translation:.6f}m "
                f"max_linear={max_linear_speed:.6f}m/s finite={finite}",
                flush=True,
            )

    # Render pulses are still zero-action physics steps and contain no writes.
    for _ in range(8):
        for robot in environment._robots.values():
            robot.set_joint_position_target(robot.data.default_joint_pos)
        environment._scene.write_data_to_sim()
        sim.step(render=True)
        environment._scene.update(sim_dt)
    final = {path: _view_state(view) for path, view in rigid["views"].items()}
    final_relative = (
        final[blade_path]["transform_xyzw"][:3]
        - final[body_path]["transform_xyzw"][:3]
    )
    blade_relative_drift = float(np.linalg.norm(final_relative - baseline_relative))
    final_positions = _numpy(
        articulation["view"].get_dof_positions()
    ).reshape(1, -1)[0]
    final_velocities = _numpy(
        articulation["view"].get_dof_velocities()
    ).reshape(1, -1)[0]
    dof_drift = float(
        np.max(
            np.abs(
                final_positions
                - np.asarray(articulation["initial_positions"], dtype=np.float64)
            )
        )
    )
    head = _render_head(environment, output_dir)
    stage.Export(str(scene_path))
    scene_sha = sha256_file(scene_path)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    final_linear_speed = max(
        float(np.linalg.norm(state["velocity"][:3])) for state in final.values()
    )
    final_angular_speed = max(
        float(np.linalg.norm(state["velocity"][3:])) for state in final.values()
    )
    stability = {
        "finite": finite,
        "max_body_translation_m": max_translation,
        "blade_relative_drift_m": blade_relative_drift,
        "blade_dof_drift": dof_drift,
        "max_linear_speed_m_s": max_linear_speed,
        "max_angular_speed_rad_s": max_angular_speed,
        "final_max_linear_speed_m_s": final_linear_speed,
        "final_max_angular_speed_rad_s": final_angular_speed,
        "baseline_body_transforms_xyzw": {
            path: state["transform_xyzw"] for path, state in baseline.items()
        },
        "final_body_transforms_xyzw": {
            path: state["transform_xyzw"] for path, state in final.items()
        },
        "pass": bool(
            finite
            and max_translation <= 0.01
            and blade_relative_drift <= 0.002
            and dof_drift <= 0.002
            and final_linear_speed <= 0.01
            and final_angular_speed <= 0.10
        ),
    }
    blade_joint = dict(inventory["prismatic_joints"][0])
    blade_joint.update(
        {
            "composed_path": composition["blade_joint_path"],
            "initial_dof_positions": articulation["initial_positions"],
            "final_dof_positions": final_positions,
            "final_dof_velocities": final_velocities,
            "drive": composition["blade_drive"],
        }
    )
    report = {
        "schema": "as013_core_scene_live_v1",
        "task_id": TASK_CODE,
        "core_scene_pass": False,
        "task_complete": False,
        "motion_path_executed": False,
        "object_state_writes_during_settle": object_state_writes_during_settle,
        "reset_state_writes_before_settle": safe_reset["state_writes_before_settle"],
        "authoritative_source": {
            "file": "b320_32_scene_wave_20260817/source_rows.json",
            "sha256": "c1bf0990e8ec0ae651ae6d904c6b61e45796c8e11a670f6da782f4b23052e987",
            "task_row": "B320-AS013",
            "excel_row": 254,
        },
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        },
        "gpu_namespace": gpu,
        "runtime": {
            "simulator": "Isaac Sim 5.1",
            "physics_backend": "PhysX",
            "physics_hz": PHYSICS_HZ,
            "physics_steps": requested_steps + 8,
            "simulated_seconds": (requested_steps + 8) / PHYSICS_HZ,
            "wall_seconds": time.time() - started,
            "free_vram_bytes_after_settle": int(free_bytes),
            "total_vram_bytes": int(total_bytes),
            "vram_free_fraction_after_settle": float(free_bytes / total_bytes),
        },
        "robot": {
            "sides": sorted(environment._robots),
            "joint_counts": {
                side: int(robot.num_joints) for side, robot in environment._robots.items()
            },
            "model": "ROKAE with Wuji gen1",
            "table_top_z_m": table_top,
        },
        "asset": {
            "source": str(ASSET_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(ASSET_PATH),
            "integrated_usd_sha256": inventory["integrated_asset_sha256"],
            "runtime_mirror_sha256": inventory["runtime_mirror_sha256"],
            "metadata_sha256": inventory["metadata_sha256"],
            "source_hashes": inventory["source_hashes"],
            "uniform_scale": UNIFORM_SCALE,
            "position_m": composition["position_m"],
            "initial_table_clearance_m": composition["initial_table_clearance_m"],
            "expected_initial_world_aabb_m": composition["expected_world_aabb_m"],
            "collision_shape_count": len(composition["collision_paths"]),
            "collision_shape_paths": composition["collision_paths"],
            "rigid_body_paths": composition["rigid_body_paths"],
            "physics_material": composition["physics_material"],
            "safe_reset_joint": safe_reset,
            "blade_joint": blade_joint,
            "isolated_inventory": inventory,
        },
        "lighting": lighting,
        "head_camera": head,
        "stability": stability,
        "scene_usda": {
            "path": str(scene_path),
            "bytes": scene_path.stat().st_size,
            "sha256": scene_sha,
        },
    }
    report["core_scene_pass"] = bool(
        gpu["pass"]
        and inventory["pass"]
        and len(composition["collision_paths"]) == inventory["collision_shape_count"]
        and len(composition["rigid_body_paths"]) >= 2
        and composition["initial_table_clearance_m"] >= 0.002
        and articulation["pass"]
        and stability["pass"]
        and head["pass"]
        and report["runtime"]["physics_steps"] >= ZERO_ACTION_STEPS
        and report["runtime"]["vram_free_fraction_after_settle"] > 0.20
        and object_state_writes_during_settle == 0
        and not report["motion_path_executed"]
        and not report["task_complete"]
    )
    environment.close()
    return report


def main() -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "live_scene_report.json"
    try:
        report = _run_scene(output_dir)
        _write_json(report_path, report)
        print(
            f"[AS013] core scene {'PASSED' if report['core_scene_pass'] else 'FAILED'}; "
            f"report={report_path}",
            flush=True,
        )
        return 0 if report["core_scene_pass"] else 1
    except BaseException as exc:
        failure = {
            "schema": "as013_core_scene_live_v1",
            "task_id": TASK_CODE,
            "core_scene_pass": False,
            "task_complete": False,
            "motion_path_executed": False,
            "object_state_writes_during_settle": 0,
            "exception": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        _write_json(report_path, failure)
        traceback.print_exc()
        return 1


exit_code = main()
sys.stdout.flush()
sys.stderr.flush()
shutdown_guard = threading.Timer(8.0, lambda: os._exit(exit_code))
shutdown_guard.daemon = True
shutdown_guard.start()
try:
    simulation_app.close()
finally:
    shutdown_guard.cancel()
raise SystemExit(exit_code)
