#!/usr/bin/env python3
"""Bounded live PhysX validation for Synchronized Dual-Knob Setting."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import threading
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
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-images", action="store_true")
    return parser


def _unwrap(environment: Any) -> Any:
    current = environment
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_dual_knob_tracker"):
            return current
        next_environment = getattr(current, "env", None)
        if next_environment is None:
            next_environment = getattr(current, "_env", None)
        if next_environment is None:
            break
        current = next_environment
    raise RuntimeError("Could not unwrap SynchronizedDualKnobIsaacLabEnv")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _save_frame(path: Path, frame: np.ndarray) -> dict[str, Any]:
    array = np.asarray(frame, dtype=np.uint8)
    if array.ndim != 3 or array.shape[-1] not in {3, 4}:
        raise RuntimeError(f"Unexpected frame shape for {path.name}: {array.shape}")
    try:
        import imageio.v3 as iio

        iio.imwrite(path, array)
    except Exception:
        from PIL import Image

        Image.fromarray(array).save(path)
    return {
        "path": str(path),
        "shape": list(array.shape),
        "mean_rgb": float(np.mean(array[..., :3])),
        "minimum": int(np.min(array)),
        "maximum": int(np.max(array)),
        "black_fraction": float(np.mean(np.all(array[..., :3] == 0, axis=-1))),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.steps < 1 or args.steps > 600:
        raise ValueError("--steps must be in [1, 600]")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")

    from handson.sim.isaaclab.dual_knob_panel import (
        DUAL_KNOB_ROOT_PATH,
        export_dual_knob_panel_usda,
    )
    from handson.tasks.mappings import CONFIG_MAPPING

    task_name = "synchronized_dual_knob_setting"
    config = CONFIG_MAPPING[task_name]()
    environment = config.get_native_environment(
        policy_mode=True,
        sim_backend="isaacsim",
        launch_isaac=True,
        image_obs=not args.no_images,
        randomize=False,
        randomize_dynamics=False,
        render_mode="rgb_array",
        headless=True,
        isaac_device=args.device,
        hz=30,
        control_dt=1.0 / 30.0,
        physics_dt=1.0 / 120.0,
        time_limit=32.0,
    )
    raw = _unwrap(environment)
    # RTX camera teardown may otherwise block indefinitely in SimulationContext.stop().
    # The ABC runtime already supports the same bounded skip-cleanup path used by the
    # repository's IsaacLab preview scripts.
    raw._fast_shutdown_on_close = True
    try:
        observation, reset_info = environment.reset(seed=args.seed)
        del observation
        from pxr import Usd, UsdPhysics

        root = raw._stage.GetPrimAtPath(DUAL_KNOB_ROOT_PATH)
        if not root.IsValid():
            raise RuntimeError(f"Dual-knob root is missing: {DUAL_KNOB_ROOT_PATH}")
        rigid_bodies = []
        colliders = []
        revolute_joints = []
        for prim in Usd.PrimRange(root):
            path = str(prim.GetPath())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_bodies.append(path)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                colliders.append(path)
            if prim.IsA(UsdPhysics.RevoluteJoint):
                revolute_joints.append(path)
        expected_joint_names = {"left_knob_joint", "right_knob_joint"}
        actual_joint_names = {Path(path).name for path in revolute_joints}
        if len(rigid_bodies) != 3:
            raise RuntimeError(f"Expected 3 panel rigid bodies, got {rigid_bodies}")
        if len(colliders) != 3:
            raise RuntimeError(f"Expected 3 panel colliders, got {colliders}")
        if actual_joint_names != expected_joint_names:
            raise RuntimeError(
                f"Expected knob joints {sorted(expected_joint_names)}, got {sorted(actual_joint_names)}"
            )
        if len(raw._dual_knob_contact_views) != 4:
            raise RuntimeError(
                f"Expected four hand-knob contact views, got {len(raw._dual_knob_contact_views)}"
            )

        q_start = np.asarray(raw._joint_reading(), dtype=np.float64)
        root_start = raw._dual_knob_articulation_view.get_root_transforms()[0]
        if hasattr(root_start, "detach"):
            root_start = root_start.detach().cpu().numpy()
        root_start = np.asarray(root_start, dtype=np.float64)
        terminated = False
        truncated = False
        final_info: dict[str, Any] = {}
        zero_action = np.zeros(environment.action_space.shape, dtype=np.float32)
        for step_index in range(args.steps):
            _observation, reward, terminated, truncated, final_info = environment.step(
                zero_action
            )
            if reward != 0.0 or final_info.get("succeed"):
                raise RuntimeError(
                    f"Zero action produced a false success at step {step_index}"
                )
            if terminated or truncated:
                raise RuntimeError(
                    f"Zero action terminated early at step {step_index}: {final_info}"
                )

        q_end = np.asarray(raw._joint_reading(), dtype=np.float64)
        root_end = raw._dual_knob_articulation_view.get_root_transforms()[0]
        if hasattr(root_end, "detach"):
            root_end = root_end.detach().cpu().numpy()
        root_end = np.asarray(root_end, dtype=np.float64)
        if not np.all(np.isfinite(np.concatenate((q_start, q_end, root_start, root_end)))):
            raise RuntimeError("Live panel state contains non-finite values")
        knob_drift_deg = np.degrees(np.abs(q_end[:2] - q_start[:2]))
        if float(np.max(knob_drift_deg)) > 1.0:
            raise RuntimeError(
                f"Zero-action knob drift exceeds 1 degree: {knob_drift_deg.tolist()}"
            )
        root_drift_m = float(np.linalg.norm(root_end[:3] - root_start[:3]))
        if root_drift_m > 0.002:
            raise RuntimeError(f"Fixed panel root drifted {root_drift_m:.6f}m")

        frames: dict[str, Any] = {}
        if not args.no_images:
            main_frame, wrist_left, wrist_right = raw.render()
            frames = {
                "main": _save_frame(output_dir / "main.png", main_frame),
                "wrist_left": _save_frame(output_dir / "wrist_left.png", wrist_left),
                "wrist_right": _save_frame(output_dir / "wrist_right.png", wrist_right),
            }
            for name, metrics in frames.items():
                if metrics["mean_rgb"] <= 30.0 or metrics["black_fraction"] >= 0.95:
                    raise RuntimeError(f"Camera {name} failed RGB energy checks: {metrics}")

        exported_asset = export_dual_knob_panel_usda(
            output_dir / "dual_knob_calibration_panel.usda"
        )
        result = {
            "task_name": task_name,
            "seed": int(args.seed),
            "steps": int(args.steps),
            "action_shape": list(environment.action_space.shape),
            "rigid_bodies": rigid_bodies,
            "colliders": colliders,
            "revolute_joints": revolute_joints,
            "dof_indices": dict(raw._dual_knob_dof_indices),
            "contact_view_count": len(raw._dual_knob_contact_views),
            "q_start_rad_rad_s": q_start.tolist(),
            "q_end_rad_rad_s": q_end.tolist(),
            "knob_drift_deg": knob_drift_deg.tolist(),
            "root_drift_m": root_drift_m,
            "reset_info": reset_info,
            "final_info": final_info,
            "task_metrics": raw._dual_knob_last_metrics,
            "frames": frames,
            "exported_asset": str(exported_asset),
        }
        result_path = output_dir / "validation.json"
        result_path.write_text(
            json.dumps(_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(f"VALIDATION_JSON={result_path}", flush=True)
        print(
            "DUAL_KNOB_LIVE_VALIDATION_PASSED "
            f"steps={args.steps} dofs={sorted(actual_joint_names)} "
            f"knob_drift_deg={knob_drift_deg.tolist()} root_drift_m={root_drift_m:.8f}",
            flush=True,
        )
        return 0
    finally:
        close_exit_code = 0 if sys.exc_info()[0] is None else 1
        sys.stdout.flush()
        sys.stderr.flush()
        shutdown_guard = threading.Timer(
            10.0, lambda: os._exit(close_exit_code)
        )
        shutdown_guard.daemon = True
        shutdown_guard.start()
        try:
            environment.close()
        finally:
            shutdown_guard.cancel()


if __name__ == "__main__":
    raise SystemExit(main())
