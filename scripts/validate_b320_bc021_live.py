#!/usr/bin/env python3
"""Bounded Isaac Sim 5.1 live smoke for the isolated BC021 runtime.

This validator exercises the isolated BC021 runtime and checks the selected
source asset, measured geometry, collision derivation, and manifest before
the task is allowed to open.
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--reset-count", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--require-final-asset", action="store_true")
    return parser


def _unwrap(environment: Any) -> Any:
    current = environment
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_bc021_state"):
            return current
        next_environment = getattr(current, "env", None)
        if next_environment is None:
            next_environment = getattr(current, "_env", None)
        if next_environment is None:
            break
        current = next_environment
    raise RuntimeError("Could not unwrap HandsOnB320BC021IsaacLabEnv")


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
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _runtime_contract() -> dict[str, str]:
    """Read and enforce the one supported collection runtime from disk."""

    try:
        isaac_sim_version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("The Isaac Sim distribution metadata is missing") from exc
    if isaac_sim_version != "5.1.0.0":
        raise RuntimeError(
            "BC021 requires Isaac Sim 5.1.0.0; got "
            f"{isaac_sim_version!r}"
        )

    import isaaclab

    isaaclab_source = Path(isaaclab.__file__).resolve()
    version_path: Path | None = None
    for parent in (isaaclab_source.parent, *isaaclab_source.parents):
        candidate = parent / "VERSION"
        if candidate.is_file():
            version_path = candidate
            break
    if version_path is None:
        raise RuntimeError(
            f"Could not resolve the IsaacLab VERSION from {isaaclab_source}"
        )
    isaaclab_version = version_path.read_text(encoding="utf-8").strip()
    if isaaclab_version != "2.3.2":
        raise RuntimeError(
            "BC021 requires IsaacLab 2.3.2; got "
            f"{isaaclab_version!r} from {version_path}"
        )
    return {
        "isaac_sim_version": isaac_sim_version,
        "isaaclab_version": isaaclab_version,
        "isaaclab_source": str(isaaclab_source),
        "isaaclab_version_path": str(version_path),
    }


def _reset_sample_record(raw: Any, *, seed: int) -> dict[str, Any]:
    """Fail closed on support, side assignment, spacing, and finite poses."""

    from handson.sim.isaaclab.b320_bc021_contract import rotation_matrix

    sample = raw._bc021_reset_sample
    geometry = raw._bc021_geometry
    poses = {
        "block_a": np.asarray(sample.left_pose_wxyz, dtype=np.float64).reshape(7),
        "block_b": np.asarray(sample.right_pose_wxyz, dtype=np.float64).reshape(7),
    }
    if not all(np.all(np.isfinite(pose)) for pose in poses.values()):
        raise RuntimeError(f"BC021 seed {seed} produced a non-finite reset pose")
    if poses["block_a"][1] <= 0.06 or poses["block_b"][1] >= -0.06:
        raise RuntimeError(
            f"BC021 seed {seed} violated the assigned-side reset lanes"
        )

    bottom_errors: dict[str, float] = {}
    for name, side in (("block_a", "left"), ("block_b", "right")):
        pose = poses[name]
        projected_half_height = float(
            np.abs(rotation_matrix(pose[3:7])[2, :])
            @ geometry.half_extents(side)
        )
        bottom_z = float(pose[2] - projected_half_height)
        bottom_errors[name] = float(bottom_z - geometry.table_top_z_m)
        if abs(bottom_errors[name]) > 0.003:
            raise RuntimeError(
                f"BC021 seed {seed} reset support error for {name}: "
                f"{bottom_errors[name]:.6f} m"
            )

    center_separation = float(
        np.linalg.norm(poses["block_a"][:3] - poses["block_b"][:3])
    )
    required_separation = float(
        np.linalg.norm(geometry.half_extents("left"))
        + np.linalg.norm(geometry.half_extents("right"))
        + 0.10
    )
    if center_separation < required_separation:
        raise RuntimeError(
            f"BC021 seed {seed} reset pair overlaps its conservative envelope"
        )

    return {
        "seed": int(seed),
        "sample": sample.as_dict(),
        "bottom_error_m": bottom_errors,
        "center_separation_m": center_separation,
        "required_center_separation_m": required_separation,
        "reset_clearance_point_count": int(
            raw._bc021_reset_clearance_point_count
        ),
        "accepted_origin_poses": {
            name: np.asarray(pose, dtype=np.float64).reshape(7).tolist()
            for name, pose in raw._bc021_recording_origin_poses.items()
        },
    }


def _reset_projection_record(raw: Any) -> dict[str, Any]:
    """Project both rigid centers and both semantic face centers per camera."""

    from handson.sim.isaaclab.b320_bc021_contract import rotation_matrix

    geometry = raw._bc021_geometry
    left = np.asarray(raw._bc021_last_pose_wxyz["block_a"], dtype=np.float64)
    right = np.asarray(raw._bc021_last_pose_wxyz["block_b"], dtype=np.float64)
    world_points = np.stack(
        [
            left[:3],
            right[:3],
            left[:3]
            + rotation_matrix(left[3:7])
            @ geometry.marked_center_local("left"),
            right[:3]
            + rotation_matrix(right[3:7])
            @ geometry.marked_center_local("right"),
        ],
        axis=0,
    )
    wrist_width, wrist_height = raw._wrist_recording_camera_output_size
    camera_specs = {
        "head_camera": (raw._render_width, raw._render_height),
        "wrist_left_camera": (wrist_width, wrist_height),
        "wrist_right_camera": (wrist_width, wrist_height),
    }
    result: dict[str, Any] = {}
    for camera_key, (width, height) in camera_specs.items():
        camera_pose = raw._recording_camera_pose_matrix_for_key(camera_key)
        if camera_pose is None:
            raise RuntimeError(f"BC021 {camera_key} pose is unavailable")
        pixels = raw._project_opengl_camera_points(
            camera_pose,
            world_points,
            image_width=int(width),
            image_height=int(height),
            camera_key=camera_key,
        )
        inside = (
            [False] * len(world_points)
            if pixels is None
            else [
                bool(4 <= x < int(width) - 4 and 4 <= y < int(height) - 4)
                for x, y in pixels
            ]
        )
        result[camera_key] = {
            "pixels": pixels,
            "inside_with_4px_margin": inside,
            "image_size": [int(width), int(height)],
        }
    if not all(result["head_camera"]["inside_with_4px_margin"]):
        raise RuntimeError(
            "BC021 reset block/marker centers leave the head-camera frustum: "
            f"{result['head_camera']}"
        )
    return result


def _save_frame(
    path: Path,
    frame: np.ndarray,
    *,
    expected_shape: tuple[int, int, int],
) -> dict[str, Any]:
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        raise RuntimeError(
            f"Unexpected frame dtype for {path.name}: {array.dtype}"
        )
    if array.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected frame shape for {path.name}: {array.shape}; "
            f"expected {expected_shape}"
        )
    try:
        import imageio.v3 as iio

        iio.imwrite(path, array)
    except Exception:
        from PIL import Image

        Image.fromarray(array).save(path)
    rgb = array[..., :3]
    unique_rgb_count = int(np.unique(rgb.reshape(-1, 3), axis=0).shape[0])
    channel_std = [float(value) for value in np.std(rgb, axis=(0, 1))]
    quantized = (rgb // 8).reshape(-1, 3)
    _colors, counts = np.unique(quantized, axis=0, return_counts=True)
    probabilities = counts.astype(np.float64) / float(np.sum(counts))
    quantized_entropy = float(
        -np.sum(probabilities * np.log2(probabilities))
    )
    luma = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )
    luma_range = float(np.percentile(luma, 95) - np.percentile(luma, 5))
    return {
        "path": str(path),
        "shape": list(array.shape),
        "mean_rgb": float(np.mean(rgb)),
        "black_fraction": float(np.mean(np.all(rgb == 0, axis=-1))),
        "unique_rgb_count": unique_rgb_count,
        "coarse_rgb_count": int(len(counts)),
        "coarse_rgb_entropy_bits": quantized_entropy,
        "dominant_coarse_rgb_fraction": float(np.max(probabilities)),
        "channel_std": channel_std,
        "luma_std": float(np.std(luma)),
        "luma_p95_minus_p05": luma_range,
        "horizontal_edge_fraction": float(
            np.mean(np.abs(np.diff(luma, axis=1)) > 4.0)
        ),
        "vertical_edge_fraction": float(
            np.mean(np.abs(np.diff(luma, axis=0)) > 4.0)
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.steps < 1 or args.steps > 300:
        raise ValueError("--steps must be in [1, 300]")
    if args.reset_count < 1 or args.reset_count > 32:
        raise ValueError("--reset-count must be in [1, 32]")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    os.environ["HANDSON_STABLE_RECORDING_ORIGIN"] = "1"
    os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
    os.environ["HANDSON_ISAACLAB_DEFER_ARM_RENDER"] = "0"
    os.environ["HANDSON_ISAACLAB_DIRECT_HAND_CONTROL"] = "0"
    os.environ.pop("HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY", None)
    os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = (
        "0" if args.images else "1"
    )
    if args.images:
        # Mirror the canonical fresh-process post-render contract.  Live
        # collection leaves this enabled and records state/actions only.
        os.environ["HANDSON_ISAACSIM_ACCURATE_CAMERA_VISIBILITY"] = "1"

    from handson.sim.isaaclab.b320_bc021_contract import CONTACT_KEYS
    from handson.sim.isaaclab.b320_bc021_env import _STATE_OBSERVATION_KEYS
    from handson.sim.isaaclab.b320_bimanual_scene import B320_ASSET_ROOT_PATH
    from handson.tasks.t01_align_blocks_in_midair.config import TaskConfig

    task_config = TaskConfig()
    environment = task_config.get_native_environment(
        policy_mode=True,
        sim_backend="isaacsim",
        launch_isaac=True,
        image_obs=args.images,
        randomize=bool(args.randomize),
        randomize_dynamics=False,
        render_mode="rgb_array",
        headless=True,
        isaac_device=args.device,
        hz=30,
        control_dt=1.0 / 30.0,
        physics_dt=1.0 / 120.0,
        time_limit=float(task_config.time_limit_seconds),
    )
    raw = _unwrap(environment)
    raw._fast_shutdown_on_close = True
    try:
        runtime_contract = _runtime_contract()
        if str(raw._physics_backend).lower() != "physx":
            raise RuntimeError(
                "BC021 collection must use PhysX, not "
                f"{raw._physics_backend!r}"
            )
        if args.require_final_asset and raw._bc021_geometry.provisional:
            raise RuntimeError(
                "BC021 final asset was required but geometry is provisional"
            )

        from handson.sim.isaaclab.recording_origin import (
            stabilize_recording_origin,
            stable_recording_origin_kwargs_for_task,
        )

        reset_records: list[dict[str, Any]] = []
        observation: dict[str, Any]
        reset_info: dict[str, Any]
        authored_reset_info: dict[str, Any]

        def _run_reset(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
            reset_observation, initial_info = environment.reset(seed=seed)
            nonlocal authored_reset_info
            authored_reset_info = dict(initial_info)
            if not environment.observation_space.contains(reset_observation):
                raise RuntimeError(
                    f"BC021 seed {seed} reset observation violates its space"
                )
            reset_observation, accepted_info = stabilize_recording_origin(
                environment,
                reset_observation,
                initial_info,
                **stable_recording_origin_kwargs_for_task(raw.task_name),
            )
            if not raw._bc021_recording_origin_accepted:
                raise RuntimeError(
                    f"BC021 seed {seed} stable recording origin was not accepted"
                )
            if not environment.observation_space.contains(reset_observation):
                raise RuntimeError(
                    f"BC021 seed {seed} accepted-origin observation violates "
                    "its space"
                )
            record = _reset_sample_record(raw, seed=seed)
            if args.randomize and record["sample"]["fallback_used"]:
                raise RuntimeError(
                    f"BC021 randomized seed {seed} fell back to the authored pose"
                )
            if args.images:
                record["camera_projection"] = _reset_projection_record(raw)
            reset_records.append(record)
            return reset_observation, dict(accepted_info)

        authored_reset_info = {}
        for reset_index in range(args.reset_count):
            observation, reset_info = _run_reset(args.seed + reset_index)

        if args.randomize and args.reset_count > 1:
            unique_samples = {
                tuple(
                    np.asarray(
                        record["sample"]["left_pose_wxyz"]
                        + record["sample"]["right_pose_wxyz"],
                        dtype=np.float64,
                    ).round(12)
                )
                for record in reset_records
            }
            if len(unique_samples) < 2:
                raise RuntimeError(
                    "BC021 different randomized seeds produced no layout variation"
                )

        repeat_seed_record: dict[str, Any] | None = None
        if args.reset_count > 1:
            first_sample = dict(reset_records[0]["sample"])
            observation, reset_info = _run_reset(args.seed)
            repeat_seed_record = reset_records.pop()
            for pose_key in ("left_pose_wxyz", "right_pose_wxyz"):
                if not np.array_equal(
                    np.asarray(first_sample[pose_key], dtype=np.float64),
                    np.asarray(
                        repeat_seed_record["sample"][pose_key],
                        dtype=np.float64,
                    ),
                ):
                    raise RuntimeError(
                        "BC021 reset is history-dependent: repeating seed "
                        f"{args.seed} changed {pose_key}"
                    )
            for scalar_key in ("attempts", "fallback_used"):
                if first_sample[scalar_key] != repeat_seed_record["sample"][
                    scalar_key
                ]:
                    raise RuntimeError(
                        "BC021 reset is history-dependent: repeating seed "
                        f"{args.seed} changed {scalar_key}"
                    )

        raw_observation = raw._compute_observation()
        raw_state = raw_observation.get("state")
        if not isinstance(raw_state, dict) or tuple(raw_state) != (
            _STATE_OBSERVATION_KEYS
        ):
            raise RuntimeError(
                "BC021 clean-room raw state schema drifted: "
                f"{tuple(raw_state or {})!r}"
            )
        expected_flat_state_size = sum(
            int(np.prod(raw.observation_space["state"][key].shape))
            for key in task_config.proprio_keys
        )
        adapted_state = np.asarray(observation.get("state"))
        if adapted_state.shape != (expected_flat_state_size,):
            raise RuntimeError(
                "BC021 adapted state shape drifted: "
                f"{adapted_state.shape} != {(expected_flat_state_size,)}"
            )

        from pxr import Usd, UsdGeom, UsdPhysics

        task_root = raw._stage.GetPrimAtPath(B320_ASSET_ROOT_PATH)
        if not task_root.IsValid():
            raise RuntimeError(f"BC021 root is missing: {B320_ASSET_ROOT_PATH}")
        classifications: dict[str, list[str]] = {
            "rigid": [],
            "collider": [],
            "formal_collider": [],
            "visible": [],
            "invisible": [],
            "marker": [],
            "semantic": [],
        }
        allowed_classifications = {
            "task_visible",
            "task_visible_marker",
            "formal_collider",
            "semantic_frame",
        }
        for prim in Usd.PrimRange(task_root):
            path = str(prim.GetPath())
            classification = prim.GetCustomDataByKey("handson:classification")
            if classification and classification not in allowed_classifications:
                raise RuntimeError(
                    f"BC021 prim has an unknown render/collision class: "
                    f"{path}={classification!r}"
                )
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                classifications["rigid"].append(path)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                classifications["collider"].append(path)
            if classification == "formal_collider":
                classifications["formal_collider"].append(path)
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    raise RuntimeError(
                        f"BC021 declared collider lacks CollisionAPI: {path}"
                    )
                if (
                    not prim.IsA(UsdGeom.Gprim)
                    or UsdGeom.Imageable(prim).ComputeVisibility()
                    != UsdGeom.Tokens.invisible
                ):
                    raise RuntimeError(
                        f"BC021 formal collider is not render-invisible: {path}"
                    )
            if classification == "task_visible_marker":
                classifications["marker"].append(path)
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    raise RuntimeError(f"BC021 marker has collision: {path}")
            if classification == "semantic_frame":
                classifications["semantic"].append(path)
                if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdGeom.Gprim):
                    raise RuntimeError(
                        f"BC021 semantic frame has render/collision geometry: {path}"
                    )
            if classification == "task_visible" and prim.HasAPI(
                UsdPhysics.CollisionAPI
            ):
                raise RuntimeError(
                    f"BC021 visible source also carries collision: {path}"
                )
            if prim.IsA(UsdGeom.Gprim):
                visibility = UsdGeom.Imageable(prim).ComputeVisibility()
                target = (
                    "invisible"
                    if visibility == UsdGeom.Tokens.invisible
                    else "visible"
                )
                classifications[target].append(path)
        expected_body_roots = set(raw._b320_spawn_result.body_paths.values())
        if set(classifications["rigid"]) != expected_body_roots:
            raise RuntimeError(
                "BC021 rigid-root closure differs from the spawn manifest: "
                f"expected={sorted(expected_body_roots)}, "
                f"actual={classifications['rigid']}"
            )
        object_collider_paths: set[str] = set()
        for name, body_path in raw._b320_spawn_result.body_paths.items():
            body_colliders = {
                path
                for path in classifications["collider"]
                if path == body_path or path.startswith(f"{body_path}/")
            }
            expected_count = int(
                raw._b320_spawn_result.collider_counts.get(name, 0)
            )
            if len(body_colliders) != expected_count:
                raise RuntimeError(
                    f"BC021 {name} collider closure differs from its spawn "
                    f"manifest: expected={expected_count}, "
                    f"actual={sorted(body_colliders)}"
                )
            if not body_colliders.issubset(
                set(classifications["formal_collider"])
            ):
                raise RuntimeError(
                    f"BC021 {name} has an undeclared or visible collider: "
                    f"{sorted(body_colliders)}"
                )
            object_collider_paths.update(body_colliders)
        static_roots = tuple(raw._b320_spawn_result.static_paths)
        static_collider_paths = {
            path
            for path in classifications["collider"]
            if any(
                path == root or path.startswith(f"{root}/")
                for root in static_roots
            )
        }
        unowned_colliders = (
            set(classifications["collider"])
            - object_collider_paths
            - static_collider_paths
        )
        if unowned_colliders or not static_collider_paths:
            raise RuntimeError(
                "BC021 collider ownership is incomplete: "
                f"unowned={sorted(unowned_colliders)}, "
                f"static={sorted(static_collider_paths)}"
            )
        if len(classifications["marker"]) != 2:
            raise RuntimeError(
                f"Expected two marked-face visuals, got {classifications['marker']}"
            )
        if len(classifications["semantic"]) != 2:
            raise RuntimeError(
                f"Expected two semantic face frames, got {classifications['semantic']}"
            )
        if set(raw._bc021_contact_views) != set(CONTACT_KEYS):
            raise RuntimeError(
                "BC021 contact-view matrix is incomplete: "
                f"{sorted(raw._bc021_contact_views)}"
            )

        poses_start = {
            name: raw._physics_body_pose(name)
            for name in ("block_a", "block_b")
        }
        zero_action = np.zeros(environment.action_space.shape, dtype=np.float32)
        final_info: dict[str, Any] = {}
        for step_index in range(args.steps):
            observation, reward, terminated, truncated, final_info = environment.step(
                zero_action
            )
            if not environment.observation_space.contains(observation):
                raise RuntimeError(
                    f"BC021 observation violated its space at step {step_index}"
                )
            if reward != 0.0 or final_info.get("succeed"):
                raise RuntimeError(
                    f"BC021 zero action produced false success at step {step_index}"
                )
            if terminated or truncated:
                raise RuntimeError(
                    f"BC021 zero action terminated at step {step_index}: {final_info}"
                )

        poses_end = {
            name: raw._physics_body_pose(name)
            for name in ("block_a", "block_b")
        }
        for name in poses_start:
            start = np.asarray(poses_start[name], dtype=np.float64).reshape(7)
            end = np.asarray(poses_end[name], dtype=np.float64).reshape(7)
            if not np.all(np.isfinite(np.concatenate([start, end]))):
                raise RuntimeError(f"BC021 {name} pose truth is non-finite")
            drift = float(np.linalg.norm(end[:3] - start[:3]))
            if drift > 0.008:
                raise RuntimeError(
                    f"BC021 {name} zero-action drift exceeds 8 mm: {drift:.6f}"
                )

        frames: dict[str, Any] = {}
        if args.images:
            if raw._deferred_camera_render:
                raise RuntimeError(
                    "BC021 post-render requested images but cameras stayed deferred"
                )
            if set(raw._cameras) != {
                "head_camera",
                "wrist_left_camera",
                "wrist_right_camera",
            }:
                raise RuntimeError(
                    "BC021 canonical camera set is incomplete: "
                    f"{sorted(raw._cameras)}"
                )
            if raw._deferred_arm_render or raw._direct_hand_control:
                raise RuntimeError(
                    "BC021 camera smoke must render both complete articulated arms"
                )
            front, wrist_left, wrist_right = raw.render()
            wrist_width, wrist_height = raw._wrist_recording_camera_output_size
            frame_specs = {
                "front": (
                    front,
                    "ego",
                    (raw._render_height, raw._render_width, 3),
                ),
                "wrist_left": (
                    wrist_left,
                    "wrist_left",
                    (wrist_height, wrist_width, 3),
                ),
                "wrist_right": (
                    wrist_right,
                    "wrist_right",
                    (wrist_height, wrist_width, 3),
                ),
            }
            for name, (frame, fallback_key, expected_shape) in frame_specs.items():
                array = np.asarray(frame)
                fallback = raw._fallback_camera_frame(
                    fallback_key,
                    height=expected_shape[0],
                    width=expected_shape[1],
                )
                if np.array_equal(array[..., :3], fallback[..., :3]):
                    raise RuntimeError(
                        f"BC021 camera {name} returned the known fallback frame"
                    )
                frames[name] = _save_frame(
                    output_dir / f"{name}.png",
                    array,
                    expected_shape=expected_shape,
                )
            for name, metrics in frames.items():
                if (
                    metrics["mean_rgb"] <= 25.0
                    or metrics["black_fraction"] >= 0.95
                    or metrics["unique_rgb_count"] < 64
                    or metrics["coarse_rgb_count"] < 10
                    or metrics["coarse_rgb_entropy_bits"] < 1.4
                    or metrics["dominant_coarse_rgb_fraction"] >= 0.90
                    or max(metrics["channel_std"]) < 5.0
                    or metrics["luma_std"] < 15.0
                    or metrics["luma_p95_minus_p05"] < 40.0
                    or metrics["horizontal_edge_fraction"] < 0.004
                    or metrics["vertical_edge_fraction"] < 0.004
                ):
                    raise RuntimeError(f"BC021 camera {name} is unreadable: {metrics}")
            camera_warnings = [
                warning
                for warning in raw._isaac_warnings
                if "Recording cameras unavailable" in warning
                or "RGB unavailable" in warning
            ]
            if camera_warnings:
                raise RuntimeError(
                    f"BC021 camera runtime reported fallback warnings: {camera_warnings}"
                )

        result = {
            "task_name": raw.task_name,
            "seed": int(args.seed),
            "reset_count": int(args.reset_count),
            "randomize": bool(args.randomize),
            "steps": int(args.steps),
            "runtime": runtime_contract,
            "physics_backend": str(raw._physics_backend),
            "geometry_provisional": bool(raw._bc021_geometry.provisional),
            "formal_collection_certified": False,
            "deferred_camera_render": bool(raw._deferred_camera_render),
            "recording_origin_accepted": bool(
                raw._bc021_recording_origin_accepted
            ),
            "recording_origin_acceptance": (
                reset_info.get("physical_recording_origin", {}).get(
                    "task_recording_origin_acceptance"
                )
            ),
            "physical_replay_reset_contract": (
                raw.get_physical_replay_reset_contract()
            ),
            "raw_state_keys": list(raw_state),
            "adapted_state_shape": list(adapted_state.shape),
            "isaac_warnings": list(raw._isaac_warnings),
            "contact_keys": list(CONTACT_KEYS),
            "contact_bindings": raw._bc021_contact_bindings,
            "classifications": classifications,
            "reset_records": reset_records,
            "repeat_seed_record": repeat_seed_record,
            "poses_start": poses_start,
            "poses_end": poses_end,
            "authored_reset_info": authored_reset_info,
            "reset_info": reset_info,
            "final_info": final_info,
            "metrics": raw._b320_last_metrics,
            "frames": frames,
        }
        result_path = output_dir / "validation.json"
        result_path.write_text(
            json.dumps(_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(f"VALIDATION_JSON={result_path}", flush=True)
        print(
            "BC021_LIVE_SCAFFOLD_SMOKE_PASSED "
            f"steps={args.steps} provisional={raw._bc021_geometry.provisional}",
            flush=True,
        )
        return 0
    finally:
        active_exception = sys.exc_info()
        close_exit_code = 0 if active_exception[0] is None else 1
        if active_exception[0] is not None:
            # SimulationApp's skip-cleanup path terminates the process before
            # Python can print a pending exception.  Emit it now and use the
            # ordinary close path so a failed validator can never look green.
            traceback.print_exception(*active_exception)
            raw._fast_shutdown_on_close = False
        sys.stdout.flush()
        sys.stderr.flush()
        shutdown_guard = threading.Timer(
            12.0,
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
