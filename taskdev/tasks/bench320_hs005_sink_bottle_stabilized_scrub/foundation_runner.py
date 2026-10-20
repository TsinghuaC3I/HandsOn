#!/usr/bin/env python3
"""Build and validate the B320-HS005 core scene in Isaac Sim 5.1.

This runner performs asset composition, reset layout, zero-action settling and
head-camera capture only.  It never drives a robot, grasps an object or authors
a scrub trajectory, and it always writes ``task_complete: false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "handson"))

# AppLauncher is intentionally the first Isaac import.
from isaaclab.app import AppLauncher  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--scene-layer",
        type=Path,
        help="optional HS005 base/variant layer to compose for visual audit",
    )
    result.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="validate the current working tree while recording its dirty-state digest",
    )
    AppLauncher.add_app_launcher_args(result)
    return result


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), *args], text=True
    ).strip()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def tensor_array(value: Any) -> Any:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def vector_norm(value: Any) -> float:
    import numpy as np

    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def create_core_asset_layer(scene_path: Path, table_top_z: float) -> dict[str, Any]:
    """Regenerate the portable task-local layer from its CPU-safe builder."""

    from taskdev.tasks.bench320_hs005_sink_bottle_stabilized_scrub.generate_asset_layer import (
        build_layer,
    )

    return build_layer(TASK_DIR, scene_path, table_top_z)


def main() -> int:
    args = parser().parse_args()
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
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    launcher = AppLauncher(args)
    app = launcher.app

    # All project and Kit imports below occur after the app exists.
    import carb
    import isaaclab.sim as sim_utils
    import numpy as np
    from PIL import Image
    from pxr import Usd, UsdGeom, UsdLux, UsdPhysics

    from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
    from taskdev.tasks.bench320_hs005_sink_bottle_stabilized_scrub.scene_contract import (
        HEAD_HEIGHT,
        HEAD_WIDTH,
        PHYSICS_HZ,
        SIM_SECONDS,
        TASK_ID,
        WARMUP_STEPS,
        ZERO_ACTION_STEPS,
        initial_layout_pass,
        placement_contract,
        release_scope,
        sha256_file,
        zero_action_stability_pass,
    )

    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "RayTracedLighting")
    settings.set("/rtx/post/tonemap/op", 4)
    settings.set("/rtx/post/tonemap/filmIso", 200)
    # Collection defaults defer canonical RGB sensors.  This fresh-process
    # foundation audit explicitly needs the real head-camera render product.
    os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "0"
    HandsOnABCIsaacLabEnv._canonical_camera_postrender_supported = True
    started = time.time()
    environment = None
    summary: dict[str, Any] = {
        "schema": "hs005_core_scene_foundation_v1",
        "task_id": TASK_ID,
        "foundation_pass": False,
        **release_scope(),
    }
    summary["core_scene_ready"] = False
    rc = 1
    try:
        prelaunch_status = git("status", "--porcelain")
        prelaunch_clean = prelaunch_status == ""
        summary["runtime"] = {
            "host": socket.gethostname(),
            "requested_host": os.environ.get("TASKDEV_HOST", "hp-13"),
            "physical_gpu": int(os.environ.get("TASKDEV_PHYSICAL_GPU", "4")),
            "logical_device": args.device,
            "git_head": git("rev-parse", "HEAD"),
            "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "prelaunch_worktree_clean": prelaunch_clean,
            "dirty_source_allowed": bool(args.allow_dirty_source),
            "worktree_status_sha256": hashlib.sha256(
                prelaunch_status.encode("utf-8")
            ).hexdigest(),
            "physics_hz": PHYSICS_HZ,
            "warmup_steps": WARMUP_STEPS,
            "zero_action_steps": ZERO_ACTION_STEPS,
            "simulated_seconds": ZERO_ACTION_STEPS / PHYSICS_HZ,
            "total_physics_steps": WARMUP_STEPS + ZERO_ACTION_STEPS,
        }
        if not prelaunch_clean and not args.allow_dirty_source:
            raise RuntimeError(
                "foundation runner requires a clean launch worktree unless "
                "--allow-dirty-source is supplied"
            )

        environment = HandsOnABCIsaacLabEnv(
            "t02_connect_charger_and_cable",
            seed=0,
            control_dt=1.0 / 30.0,
            physics_dt=1.0 / PHYSICS_HZ,
            time_limit=45.0,
            render_mode="rgb_array",
            image_obs=True,
            randomize=False,
            randomize_dynamics=False,
            hand_type="wuji1",
            arm_type="rokae",
            launch_isaac=True,
            headless=True,
            render_resolution=(HEAD_HEIGHT, HEAD_WIDTH),
            hz=30,
            isaac_device=args.device,
            physics_backend="physx",
            existing_simulation_app=app,
            existing_app_launcher=launcher,
        )
        stage = environment._stage
        sim = environment._sim
        if stage is None or sim is None:
            raise RuntimeError("HandsOn runtime did not expose stage and simulation")

        hidden_native = []
        for name in ("c10_power_strip", "c10_charger_plug", "c10_charging_cable"):
            prim = environment._find_prim(name)
            if prim is not None and prim.IsValid():
                hidden_native.append(str(prim.GetPath()))
                # Visibility is non-structural.  Never deactivate a prim after
                # HandsOn has created PhysX tensor views: doing so invalidates
                # the simulation view for the entire stage.
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()

        table_top_z = float(environment._current_table_top_z())
        table_prim = environment._find_prim("table")
        disabled_table_colliders = []
        if table_prim is not None and table_prim.IsValid():
            table_imageable = UsdGeom.Imageable(table_prim)
            if table_imageable:
                table_imageable.MakeInvisible()
            for prim in Usd.PrimRange(table_prim):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
                disabled_table_colliders.append(str(prim.GetPath()))
        asset_scene = (
            args.scene_layer.resolve()
            if args.scene_layer is not None
            else TASK_DIR / "scene/hs005_assets.usda"
        )
        variant_metadata: dict[str, Any] | None = None
        variants_path = TASK_DIR / "scene/scene_variants.json"
        if args.scene_layer is not None and variants_path.is_file():
            variants_manifest = json.loads(variants_path.read_text(encoding="utf-8"))
            for scene_id, candidate in variants_manifest.get("variants", {}).items():
                candidate_path = (TASK_DIR / candidate["path"]).resolve()
                if candidate_path == asset_scene:
                    variant_metadata = {"scene_id": scene_id, **candidate}
                    break
        asset_layer_metadata = TASK_DIR / "scene/hs005_assets.json"
        if not asset_scene.is_file() or not asset_layer_metadata.is_file():
            raise RuntimeError("pre-generated asset scene or metadata is missing")
        layer = json.loads(asset_layer_metadata.read_text(encoding="utf-8"))
        if abs(float(layer["table_top_z"]) - table_top_z) > 1.0e-4:
            raise RuntimeError(
                f"asset-layer table top {layer['table_top_z']} does not match live table {table_top_z}"
            )
        print(f"[HS005] Composing pre-generated asset layer: {asset_scene}", flush=True)
        live_root = UsdGeom.Xform.Define(stage, "/World/HS005").GetPrim()
        live_root.GetReferences().ClearReferences()
        live_root.GetReferences().AddReference(str(asset_scene.resolve()))
        print("[HS005] Asset layer composed", flush=True)

        lights_root = UsdGeom.Scope.Define(stage, "/World/HS005Lights")
        lights_root.GetPrim().SetActive(True)
        dome = UsdLux.DomeLight.Define(stage, "/World/HS005Lights/Dome")
        dome.CreateIntensityAttr(600.0)
        dome.CreateColorAttr((0.88, 0.92, 1.0))
        distant = UsdLux.DistantLight.Define(stage, "/World/HS005Lights/Key")
        distant.CreateIntensityAttr(2500.0)
        distant.CreateAngleAttr(0.53)
        distant.CreateColorAttr((1.0, 0.95, 0.88))
        # The HS005 layer is composed after the bootstrap environment creates
        # its RTX/Fabric scene.  Synchronize the live stage before the first
        # reset so both PhysX and the camera render products see the new prims.
        sim_utils.update_stage()

        sim.reset()
        physics_view = environment._robots["right"]._physics_sim_view
        if physics_view is None:
            raise RuntimeError("post-reset physics simulation view is unavailable")
        parked_native = []
        for index, name in enumerate(
            ("c10_power_strip", "c10_charger_plug", "c10_charging_cable")
        ):
            prim = environment._find_rigid_body_prim(name)
            if prim is None or not prim.IsValid():
                continue
            view = physics_view.create_rigid_body_view(str(prim.GetPath()))
            if int(view.count) < 1:
                raise RuntimeError(f"native bootstrap body view is empty: {name}")
            transforms = view.get_transforms().clone()
            transforms[:, 0] = 5.0 + float(index)
            transforms[:, 1] = 0.0
            transforms[:, 2] = table_top_z + 0.5
            velocities = view.get_velocities().clone()
            velocities.zero_()
            indices = environment._torch.arange(
                int(view.count),
                dtype=environment._torch.int32,
                device=transforms.device,
            )
            view.set_transforms(transforms, indices)
            view.set_velocities(velocities, indices)
            parked_native.append(
                {
                    "name": name,
                    "path": str(prim.GetPath()),
                    "body_count": int(view.count),
                    "reset_only_pose_writes": int(view.count),
                    "reset_only_velocity_writes": int(view.count),
                }
            )
        parked_generic_table = []
        table_body = environment._find_rigid_body_prim("table")
        if table_body is not None and table_body.IsValid():
            table_view = physics_view.create_rigid_body_view(str(table_body.GetPath()))
            if int(table_view.count) != 1:
                raise RuntimeError(
                    f"expected one generic table body, got {table_view.count}"
                )
            transforms = table_view.get_transforms().clone()
            transforms[:, 0] = 5.0
            transforms[:, 1] = 5.0
            transforms[:, 2] = -5.0
            velocities = table_view.get_velocities().clone()
            velocities.zero_()
            indices = environment._torch.arange(
                int(table_view.count),
                dtype=environment._torch.int32,
                device=transforms.device,
            )
            table_view.set_transforms(transforms, indices)
            table_view.set_velocities(velocities, indices)
            parked_generic_table.append(str(table_body.GetPath()))
        bottle_view = physics_view.create_rigid_body_view("/World/HS005/Bottle")
        brush_view = physics_view.create_rigid_body_view("/World/HS005/Brush")
        if int(bottle_view.count) != 1 or int(brush_view.count) != 1:
            raise RuntimeError(
                f"expected one bottle and brush body, got {bottle_view.count}/{brush_view.count}"
            )

        # The bootstrap C10 table exists during the first reset.  After its
        # body is parked, restore the two newly composed HS005 actors to their
        # authored reset positions and clear the one reset-frame impulse.  No
        # pose or velocity writes occur during the measured action interval.
        expected_z = (
            variant_metadata["object_reset_z"]
            if variant_metadata is not None
            else {"bottle": 0.898, "brush": 0.91202124}
        )
        for view, xyz in (
            (bottle_view, (-0.10, 0.11, float(expected_z["bottle"]))),
            (brush_view, (-0.191, -0.125, float(expected_z["brush"]))),
        ):
            transforms = view.get_transforms().clone()
            transforms[:, :3] = environment._torch.as_tensor(
                xyz,
                dtype=transforms.dtype,
                device=transforms.device,
            )
            transforms[:, 3:7] = environment._torch.as_tensor(
                [0.0, 0.0, 0.0, 1.0],
                dtype=transforms.dtype,
                device=transforms.device,
            )
            velocities = view.get_velocities().clone()
            velocities.zero_()
            indices = environment._torch.arange(
                int(view.count),
                dtype=environment._torch.int32,
                device=transforms.device,
            )
            view.set_transforms(transforms, indices)
            view.set_velocities(velocities, indices)

        for _step in range(WARMUP_STEPS):
            sim.step(render=False)
        initial_bottle = tensor_array(bottle_view.get_transforms()).reshape(1, 7)[0]
        initial_brush = tensor_array(brush_view.get_transforms()).reshape(1, 7)[0]
        for _step in range(ZERO_ACTION_STEPS):
            sim.step(render=False)
        final_bottle = tensor_array(bottle_view.get_transforms()).reshape(1, 7)[0]
        final_brush = tensor_array(brush_view.get_transforms()).reshape(1, 7)[0]
        bottle_velocity = tensor_array(bottle_view.get_velocities()).reshape(1, 6)[0]
        brush_velocity = tensor_array(brush_view.get_velocities()).reshape(1, 6)[0]
        bottle_displacement = vector_norm(final_bottle[:3] - initial_bottle[:3])
        brush_displacement = vector_norm(final_brush[:3] - initial_brush[:3])
        stability = {
            "warmup_steps": WARMUP_STEPS,
            "zero_action_steps": ZERO_ACTION_STEPS,
            "total_physics_steps": WARMUP_STEPS + ZERO_ACTION_STEPS,
            "simulated_seconds": ZERO_ACTION_STEPS / PHYSICS_HZ,
            "object_pose_or_velocity_writes_during_action": 0,
            "bottle": {
                "initial_transform_xyzw": initial_bottle.tolist(),
                "final_transform_xyzw": final_bottle.tolist(),
                "final_velocity": bottle_velocity.tolist(),
                "displacement_m": bottle_displacement,
                "linear_speed_m_s": vector_norm(bottle_velocity[:3]),
                "angular_speed_rad_s": vector_norm(bottle_velocity[3:]),
            },
            "brush": {
                "initial_transform_xyzw": initial_brush.tolist(),
                "final_transform_xyzw": final_brush.tolist(),
                "final_velocity": brush_velocity.tolist(),
                "displacement_m": brush_displacement,
                "linear_speed_m_s": vector_norm(brush_velocity[:3]),
                "angular_speed_rad_s": vector_norm(brush_velocity[3:]),
            },
        }
        stability["pass"] = zero_action_stability_pass(
            bottle_displacement_m=bottle_displacement,
            brush_displacement_m=brush_displacement,
            bottle_linear_speed_m_s=stability["bottle"]["linear_speed_m_s"],
            brush_linear_speed_m_s=stability["brush"]["linear_speed_m_s"],
            bottle_angular_speed_rad_s=stability["bottle"]["angular_speed_rad_s"],
            brush_angular_speed_rad_s=stability["brush"]["angular_speed_rad_s"],
            simulated_seconds=stability["simulated_seconds"],
        )
        visible_support_alignment: dict[str, Any] = {
            "pass": True,
            "expected_root_z": {},
            "settled_root_z": {
                "bottle": float(initial_bottle[2]),
                "brush": float(initial_brush[2]),
            },
        }
        if variant_metadata is not None:
            expected_root_z = variant_metadata.get("object_reset_z", {})
            visible_support_alignment["expected_root_z"] = expected_root_z
            visible_support_alignment["pass"] = bool(
                abs(float(initial_bottle[2]) - float(expected_root_z["bottle"]))
                <= 0.010
                and abs(float(initial_brush[2]) - float(expected_root_z["brush"]))
                <= 0.010
            )

        frame = None
        for _ in range(5):
            frames = environment.render()
            if isinstance(frames, (list, tuple)) and frames:
                frame = np.asarray(frames[0])
        if frame is None or frame.shape != (HEAD_HEIGHT, HEAD_WIDTH, 3):
            raise RuntimeError(f"bad head-camera frame: {None if frame is None else frame.shape}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        preview_path = output_dir / "reset_head.png"
        Image.fromarray(frame).save(preview_path, compress_level=0)
        rgb = frame.astype(np.float64)
        expected_camera_path = "/World/RecordCameras/head_camera"
        camera_prim = stage.GetPrimAtPath(expected_camera_path)
        camera_prim_path = (
            expected_camera_path
            if camera_prim.IsValid() and camera_prim.IsA(UsdGeom.Camera)
            else ""
        )
        unique_rgb_count = int(np.unique(frame.reshape(-1, 3), axis=0).shape[0])
        camera = {
            "prim_path": camera_prim_path,
            "resolution_hwc": list(frame.shape),
            "dtype": str(frame.dtype),
            "mean_rgb": float(rgb.mean()),
            "max_rgb": float(rgb.max()),
            "pixel_stddev": float(rgb.std()),
            "unique_rgb_count": unique_rgb_count,
            "black_fraction": float(np.mean(np.max(rgb, axis=2) < 10.0)),
            "preview_path": str(preview_path),
            "preview_bytes": preview_path.stat().st_size,
            "preview_sha256": sha256_file(preview_path),
        }
        camera["pass"] = bool(
            camera_prim_path == "/World/RecordCameras/head_camera"
            and camera["mean_rgb"] > 30.0
            and camera["pixel_stddev"] > 5.0
            and camera["max_rgb"] > 150.0
            and camera["black_fraction"] <= 0.35
            and camera["unique_rgb_count"] >= 256
            and camera["preview_bytes"] >= 150_000
        )

        source_paths = {
            "kitchen": TASK_DIR
            / "assets/modern_apartment_kitchen/modern_apartment_kitchen.usd",
            "sink": TASK_DIR
            / "assets/modern_apartment_kitchen/sink_collision_source.ply",
            "bottle_visual": TASK_DIR
            / "assets/cc_by_lukemodels75_water_bottle/water_bottle_visual.usd",
            "bottle_source": TASK_DIR
            / "assets/cc_by_lukemodels75_water_bottle/source/scene.gltf",
            "brush_visual": TASK_DIR
            / "assets/cc_by_amhyde_cleaning_brush/cleaning_brush_visual.usd",
            "brush_source": TASK_DIR
            / "assets/cc_by_amhyde_cleaning_brush/source/scene.gltf",
            "dish_soap_visual": TASK_DIR
            / "assets/cc_by_mike_nov_dishwashing_dispenser/dishwashing_dispenser_visual.usd",
            "dish_soap_source": TASK_DIR
            / "assets/cc_by_mike_nov_dishwashing_dispenser/source/scene.gltf",
            "sponge_visual": TASK_DIR
            / "assets/cc_by_requital_dish_sponge/dish_sponge_visual.usd",
            "sponge_source": TASK_DIR
            / "assets/cc_by_requital_dish_sponge/source/scene.gltf",
            "robot": REPOSITORY_ROOT
            / "handson/handson/sim/envs/xmls/rokae_wuji1_bimanual.xml",
        }
        if variant_metadata is not None:
            source_paths["kitchen_variant"] = (
                TASK_DIR / variant_metadata["background_asset"]
            )
        layout = placement_contract(table_top_z)
        robot = {
            "names": sorted(environment._robots),
            "arm_joint_counts": {
                side: len(environment._arm_joint_ids[side])
                for side in ("left", "right")
            },
            "hand_joint_counts": {
                side: len(environment._hand_joint_ids[side])
                for side in ("left", "right")
            },
        }
        robot["pass"] = bool(
            robot["names"] == ["left", "right"]
            and all(value == 7 for value in robot["arm_joint_counts"].values())
            and all(value == 20 for value in robot["hand_joint_counts"].values())
        )
        lighting = {
            "dome": {"path": str(dome.GetPath()), "intensity": 600.0},
            "distant": {"path": str(distant.GetPath()), "intensity": 2500.0},
            "render_mode": "RayTracedLighting",
            "tonemap": "ACES",
            "film_iso": 200,
            "pass": True,
        }
        checks = {
            "source_assets": all(path.is_file() for path in source_paths.values()),
            "asset_layer": asset_scene.is_file() and asset_scene.stat().st_size > 1024,
            "initial_layout": initial_layout_pass(layout),
            "physical_views": int(bottle_view.count) == 1 and int(brush_view.count) == 1,
            "dual_rokae_wuji": robot["pass"],
            "lighting": lighting["pass"],
            "head_camera": camera["pass"],
            "zero_action_stability": stability["pass"],
            "visible_support_alignment": visible_support_alignment["pass"],
            "generic_table_collision_disabled": bool(
                disabled_table_colliders or parked_generic_table
            ),
            "no_action_object_writes": True,
            "scope_guard": all(
                summary[key] is False
                for key in (
                    "scrub_path_validated",
                    "grasp_validated",
                    "action_tuning_performed",
                    "task_complete",
                )
            ),
        }
        foundation_pass = all(checks.values())
        summary.update(
            {
                "foundation_pass": foundation_pass,
                "core_scene_ready": foundation_pass,
                "checks": checks,
                "hidden_native_c10_prims": hidden_native,
                "disabled_generic_table_colliders": disabled_table_colliders,
                "parked_generic_table_bodies": parked_generic_table,
                "parked_native_c10_bodies": parked_native,
                "asset_layer": {
                    **layer,
                    "bytes": asset_scene.stat().st_size,
                    "sha256": sha256_file(asset_scene),
                },
                "simulation_scene_variant": variant_metadata,
                "sources": {
                    name: {
                        "path": str(path.relative_to(REPOSITORY_ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for name, path in source_paths.items()
                },
                "initial_layout": {
                    "support_gaps_m": layout["support_gaps_m"],
                    "forbidden_overlap_m": layout["forbidden_overlap_m"],
                    "pass": initial_layout_pass(layout),
                },
                "robot": robot,
                "lighting": lighting,
                "head_camera": camera,
                "zero_action_stability": stability,
                "visible_support_alignment": visible_support_alignment,
                "success_motion_executed": False,
                "grasp_motion_executed": False,
                "scrub_motion_executed": False,
            }
        )
        rc = 0 if foundation_pass else 1
    except Exception as exc:
        summary["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        rc = 1
    finally:
        summary["elapsed_wall_seconds"] = time.time() - started
        summary["runner_rc"] = rc
        summary["task_complete"] = False
        summary["cleanup_policy"] = {
            "evidence_written_before_teardown": True,
            "kit_cleanup_watchdog_seconds": 20.0,
        }
        summary_path = output_dir / "foundation_summary.json"
        atomic_json(summary_path, summary)
        marker_name = (
            "RUNNER_FOUNDATION_PASS.json" if rc == 0 else "RUNNER_FOUNDATION_FAIL.json"
        )
        opposite_marker = output_dir / (
            "RUNNER_FOUNDATION_FAIL.json"
            if rc == 0
            else "RUNNER_FOUNDATION_PASS.json"
        )
        opposite_marker.unlink(missing_ok=True)
        marker = output_dir / marker_name
        atomic_json(
            marker,
            {
                "schema": "hs005_core_scene_foundation_marker_v1",
                "foundation_pass": summary.get("foundation_pass") is True,
                "task_complete": False,
                "summary_path": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "runner_rc": rc,
            },
        )
        print(f"[HS005] Evidence sealed: {marker}", flush=True)

        # Kit teardown can block indefinitely after complex USD composition on
        # headless cluster nodes. Evidence is already atomically durable here;
        # a short watchdog provides deterministic process cleanup while still
        # giving both the environment and SimulationApp a normal close path.
        import threading

        shutdown_watchdog = threading.Timer(20.0, lambda: os._exit(rc))
        shutdown_watchdog.daemon = True
        shutdown_watchdog.start()
        try:
            if environment is not None:
                environment.close()
            app.close()
        except Exception as exc:
            shutdown_watchdog.cancel()
            summary["environment_close_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            summary["foundation_pass"] = False
            summary["core_scene_ready"] = False
            summary["runner_rc"] = 1
            rc = 1
            atomic_json(summary_path, summary)
            marker.unlink(missing_ok=True)
            marker = output_dir / "RUNNER_FOUNDATION_FAIL.json"
            atomic_json(
                marker,
                {
                    "schema": "hs005_core_scene_foundation_marker_v1",
                    "foundation_pass": False,
                    "task_complete": False,
                    "summary_path": str(summary_path),
                    "summary_sha256": sha256_file(summary_path),
                    "runner_rc": rc,
                },
            )
        else:
            shutdown_watchdog.cancel()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
