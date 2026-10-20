#!/usr/bin/env python3
"""D20 live Isaac Sim 5.1 probe: construction, seeded reset, zero-action hold.

Runs the real ``HandsOnDLongHorizonIsaacLabEnv`` headlessly with PhysX and
writes a JSON report plus the three canonical camera frames.  It performs no
simulator state write after reset and never claims task success.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("HANDSON_ENABLE_ISAACSIM_NEWTON", "0")
os.environ.setdefault("HANDSON_RECORD_PHYSX_GROUND_TRUTH", "1")
# This is a preview-style probe: render the three cameras live so the frames
# can be inspected.  Formal collection keeps the shared deferred post-render.
os.environ.setdefault("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "0")

TASK = "d20_microwave_hot_dog_plating"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _save_images(observation: dict[str, Any], output: Path, tag: str) -> dict[str, Any]:
    from PIL import Image

    saved: dict[str, Any] = {}
    for key, frame in observation.get("images", {}).items():
        array = np.asarray(frame)
        if array.ndim != 3:
            continue
        path = output / f"{tag}_{key}.png"
        Image.fromarray(array[..., :3].astype(np.uint8)).save(path)
        saved[key] = {
            "path": str(path),
            "shape": list(array.shape),
            "mean_rgb": float(array[..., :3].mean()),
            "std_rgb": float(array[..., :3].std()),
        }
    return saved


def _free_body_poses(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    state = observation["state"]
    poses = {
        "hot_dog": np.asarray(state["hot_dog_pose"], dtype=np.float64),
        "tongs": np.asarray(state["tongs_pose"], dtype=np.float64),
    }
    for index, name in enumerate(("plate", "fork", "mustard", "mug")):
        poses[name] = np.asarray(state["tableware_poses"][index], dtype=np.float64)
    return poses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2001, 2002, 2003])
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--output", type=Path, default=Path("d20_live_probe"))
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    isaacsim_version = importlib.metadata.version("isaacsim")
    if not isaacsim_version.startswith("5.1"):
        raise RuntimeError(f"D20 requires Isaac Sim 5.1, got {isaacsim_version}")

    from handson.sim.isaaclab import d20_microwave_hot_dog as d20
    from handson.sim.isaaclab.d_long_horizon_env import HandsOnDLongHorizonIsaacLabEnv

    report: dict[str, Any] = {
        "task": TASK,
        "isaacsim_version": isaacsim_version,
        "physics_backend": "physx",
        "steps_per_seed": args.steps,
        "seeds": {},
    }
    started = time.time()
    env = HandsOnDLongHorizonIsaacLabEnv(
        TASK,
        seed=int(args.seeds[0]),
        randomize=True,
        image_obs=not args.no_images,
        render_mode="rgb_array",
        render_resolution=(480, 640),
        hz=30,
        physics_dt=1.0 / 120.0,
        time_limit=240.0,
        physics_backend="physx",
        headless=True,
    )
    report["construction_seconds"] = time.time() - started
    try:
        report["contact_views"] = sorted(getattr(env, "_d20_hot_dog_contact_views", {}))
        report["contact_view_errors"] = sorted(getattr(env, "_d20_contact_view_errors", set()))
        report["task_state_warnings"] = _jsonable(
            list(getattr(env, "_task_state_warnings", []) or [])
        )
        report["camera_diagnostics"] = {
            "cameras": sorted(getattr(env, "_cameras", {}) or {}),
            "camera_aliases": _jsonable(dict(getattr(env, "_camera_aliases", {}) or {})),
            "deferred_camera_render": bool(getattr(env, "_deferred_camera_render", None)),
            "live_main_camera_only": bool(getattr(env, "_live_main_camera_only", None)),
            "headless": bool(getattr(env, "_headless", None)),
            "image_obs": bool(getattr(env, "image_obs", None)),
        }
        previous_first: dict[str, np.ndarray] | None = None
        for seed in args.seeds:
            seed_report: dict[str, Any] = {}
            observation, reset_info = env.reset(seed=int(seed))
            first = _free_body_poses(observation)
            state = observation["state"]
            seed_report["reset_poses"] = _jsonable({k: v[:3] for k, v in first.items()})
            seed_report["microwave_state"] = _jsonable(state["microwave_state"])
            seed_report["tongs_state"] = _jsonable(state["tongs_state"])
            seed_report["workflow_progress"] = _jsonable(state["workflow_progress"])
            seed_report["reset_info_d20"] = _jsonable(
                {k: v for k, v in reset_info.items() if k.startswith("d20_")}
            )
            seed_report["isaac_warnings"] = _jsonable(reset_info.get("isaac_warnings", []))
            seed_report["finite_reset"] = bool(
                all(np.all(np.isfinite(np.asarray(v))) for v in state.values())
            )
            if seed == args.seeds[0]:
                # Deep camera diagnostics: what does the sensor actually hold?
                deep: dict[str, Any] = {
                    "accurate_camera_visibility": bool(getattr(env, "_accurate_camera_visibility", None)),
                    "deferred_arm_render": bool(getattr(env, "_deferred_arm_render", None)),
                    "render_accumulation_passes": getattr(env, "_render_accumulation_passes", None),
                    "sensors": {},
                }
                try:
                    for _ in range(3):
                        env._sim.render()
                    for name, camera in (getattr(env, "_cameras", {}) or {}).items():
                        try:
                            camera.update(float(env._control_dt))
                            output = camera.data.output
                            entry: dict[str, Any] = {"keys": sorted(output.keys())}
                            rgb = output.get("rgb")
                            if rgb is not None:
                                array = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
                                entry["rgb_shape"] = list(array.shape)
                                entry["rgb_dtype"] = str(array.dtype)
                                entry["rgb_mean"] = float(array.astype(np.float64).mean()) if array.size else None
                                entry["rgb_max"] = float(array.max()) if array.size else None
                            direct = env._camera_rgb_frame(name, "ego")
                            entry["direct_frame_mean"] = float(np.asarray(direct).mean())
                            deep["sensors"][name] = entry
                        except Exception as exc:  # noqa: BLE001
                            deep["sensors"][name] = {"error": repr(exc)}
                except Exception as exc:  # noqa: BLE001
                    deep["error"] = repr(exc)
                report["camera_deep_diagnostics"] = deep
                print("CAMERA_DEEP " + json.dumps(deep, default=str), flush=True)
            # Same-seed repeatability on the second reset of the first seed.
            if seed == args.seeds[0]:
                observation2, _ = env.reset(seed=int(seed))
                second = _free_body_poses(observation2)
                seed_report["same_seed_max_pose_delta_m"] = float(
                    max(np.linalg.norm(first[k][:3] - second[k][:3]) for k in first)
                )
                observation = observation2
                first = second
            if previous_first is not None:
                seed_report["cross_seed_translation_span_m"] = _jsonable(
                    {k: float(np.linalg.norm(first[k][:2] - previous_first[k][:2])) for k in first}
                )
            previous_first = first
            if not args.no_images:
                seed_report["images_after_reset"] = _save_images(
                    observation, args.output, f"seed{seed}_reset"
                )
            drift = {k: 0.0 for k in first}
            min_z = {k: float(first[k][2]) for k in first}
            info: dict[str, Any] = {}
            terminated_at = None
            for step in range(args.steps):
                observation, _reward, terminated, truncated, info = env.step(
                    np.zeros(54, dtype=np.float32)
                )
                poses = _free_body_poses(observation)
                for k in poses:
                    drift[k] = max(drift[k], float(np.linalg.norm(poses[k][:3] - first[k][:3])))
                    min_z[k] = min(min_z[k], float(poses[k][2]))
                if terminated or truncated:
                    terminated_at = step
                    break
            seed_report["zero_action_steps"] = int(args.steps if terminated_at is None else terminated_at + 1)
            seed_report["terminated_early_at"] = terminated_at
            seed_report["max_drift_m"] = drift
            seed_report["min_z_m"] = min_z
            seed_report["final_microwave_state"] = _jsonable(observation["state"]["microwave_state"])
            seed_report["final_tongs_state"] = _jsonable(observation["state"]["tongs_state"])
            seed_report["final_info_d20"] = _jsonable(
                {k: v for k, v in info.items() if k.startswith("d20_") or k in ("env_succeed", "failed", "succeed")}
            )
            if not args.no_images:
                seed_report["images_after_hold"] = _save_images(
                    observation, args.output, f"seed{seed}_hold"
                )
            report["seeds"][str(seed)] = seed_report
            print(json.dumps({"seed": seed, **{k: seed_report[k] for k in ("zero_action_steps", "max_drift_m", "min_z_m", "terminated_early_at")}}, sort_keys=True), flush=True)
        report["isaac_warnings_after"] = _jsonable(
            list(getattr(env, "_isaac_warnings", []) or [])
        )
        report["task_dof_views"] = sorted(getattr(env, "_task_dof_views", {}) or {})
        report["elapsed_seconds"] = time.time() - started
        # Flush the evidence before Kit teardown: SimulationApp.close() can
        # linger for minutes in headless multi-GPU pods.
        (args.output / "probe_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(
            "D20_PROBE_DONE",
            json.dumps({"seeds": list(report["seeds"]), "contact_views": report["contact_views"]}),
            flush=True,
        )
    finally:
        import threading

        threading.Timer(90.0, lambda: os._exit(0)).start()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
