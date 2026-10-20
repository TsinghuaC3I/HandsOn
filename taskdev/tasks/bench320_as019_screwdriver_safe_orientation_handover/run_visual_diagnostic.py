from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np


RUN_DIR = Path(os.environ["AS019_VALIDATION_RUN_DIR"])
SUMMARY_PATH = RUN_DIR / "as019_visual_diagnostic_summary.json"
CAPTURE_STEPS = {
    1,
    135,
    160,
    175,
    185,
    200,
    220,
    250,
    300,
    365,
    420,
    500,
    600,
}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _persist(summary):
    payload = _jsonable(summary)
    SUMMARY_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("AS019_VALIDATION_JSON=" + json.dumps(payload, sort_keys=True), flush=True)
    print(f"AS019_VALIDATION_SUMMARY={SUMMARY_PATH}", flush=True)


def _sample(raw, *, reward=0.0):
    metrics = raw._as019_last_metrics or raw._metrics()
    tool_inverse = np.linalg.inv(raw._tool_matrix())
    right_local = tool_inverse @ raw._palm_matrix("right")
    left_local = tool_inverse @ raw._palm_matrix("left")
    return {
        "step": int(raw.env_step),
        "phase": int(raw._as019_automatic_phase),
        "stage": int(raw._as019_stage),
        "stage_name": str(raw._as019_stage_name),
        "reward": float(reward),
        "right_force": float(metrics.get("right_force", 0.0)),
        "left_force": float(metrics.get("left_force", 0.0)),
        "left_palm_force": float(metrics.get("left_palm_force", 0.0)),
        "right_local_x": float(metrics.get("right_local_x", 0.0)),
        "left_local_x": float(metrics.get("left_local_x", 0.0)),
        "right_palm_local_xyz": right_local[:3, 3].tolist(),
        "left_palm_local_xyz": left_local[:3, 3].tolist(),
        "tool_pose_wxyz": np.asarray(raw._peg_ori_pose).tolist(),
        "outward_alignment": float(metrics.get("outward_alignment", 0.0)),
        "zone_inside": bool(metrics.get("zone_inside", False)),
        "right_clear": bool(metrics.get("right_clear", False)),
        "left_clear": bool(metrics.get("left_clear", False)),
        "linear_speed": float(metrics.get("linear_speed", 0.0)),
        "angular_speed": float(metrics.get("angular_speed", 0.0)),
        "owners": sorted(raw._as019_assist_anchor),
        "event_steps": [int(value) for value in raw._as019_event_steps.tolist()],
        "failure_reason": str(raw._as019_failure_reason),
        "trajectory_state_write_count": int(raw._trajectory_state_write_count),
    }


def _arrays(value, prefix="observation"):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _arrays(item, f"{prefix}.{key}")
        return
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value)
    except Exception:
        return
    if array.ndim >= 3 and array.shape[-1] >= 3:
        yield prefix, array


def _save_images(observation, step):
    from PIL import Image

    saved = []
    for key, array in _arrays(observation):
        if key not in (
            "observation.random_camera",
            "observation.wrist_left",
            "observation.wrist_right",
        ):
            continue
        frame = np.asarray(array)
        while frame.ndim > 3:
            frame = frame[0]
        if frame.ndim != 3 or frame.shape[-1] < 3:
            continue
        rgb = np.asarray(frame[..., :3])
        if np.issubdtype(rgb.dtype, np.floating):
            finite_max = float(np.nanmax(rgb)) if rgb.size else 0.0
            if finite_max <= 1.5:
                rgb = rgb * 255.0
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        safe_key = key.replace(".", "_").replace("/", "_")
        path = RUN_DIR / f"as019_step_{step:04d}_{safe_key}.png"
        Image.fromarray(rgb).save(path)
        saved.append(
            {
                "key": key,
                "path": str(path),
                "shape": list(rgb.shape),
                "mean_rgb": float(np.mean(rgb)),
                "black": bool(np.max(rgb) == 0),
            }
        )
    return saved


def main() -> int:
    started = time.time()
    env = None
    raw = None
    summary = {
        "schema_version": 1,
        "task": "t22_pass_screwdriver",
        "checkpoint": os.environ.get("AS019_EXPECTED_CHECKPOINT", "unknown"),
        "purpose": "visual diagnosis of receiver approach and grasp geometry",
        "formal_zarr_or_replay": False,
        "runtime": {
            "isaac_sim": "5.1",
            "physics_engine": "PhysX",
            "scheduler_run_id": os.environ.get("TASKDEV_RUN_ID"),
            "physical_gpu": os.environ.get("TASKDEV_PHYSICAL_GPU"),
            "physical_gpu_uuid": os.environ.get("TASKDEV_PHYSICAL_GPU_UUID"),
            "logical_gpu": os.environ.get("TASKDEV_LOGICAL_GPU"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    exit_code = 1
    try:
        from handson.tasks.t22_pass_screwdriver.config import TaskConfig
        from handson.sim.isaaclab.as019_screwdriver_handover_env import (
            AS019AutomaticPolicyWrapper,
            _OUTWARD_COS,
            _RELEASE_FORCE_N,
        )

        env = TaskConfig().get_environment(
            policy_mode=True,
            render_mode="rgb_array",
            randomize=True,
            seed=103,
            launch_isaac=True,
            headless=True,
            # AS019 owns three mandatory camera sensors.  Keep image_obs on
            # through construction/reset so AppLauncher selects the rendering
            # experience and every Camera object initializes completely.
            image_obs=True,
            render_resolution=(480, 640),
            isaac_device="cuda:0",
        )
        raw = env.unwrapped
        raw._fast_shutdown_on_close = True
        _observation, reset_info = env.reset(seed=103)
        # Cameras remain initialized, but rendering is enabled only for the
        # selected diagnostic frames so the physical trajectory timing stays
        # close to the normal collection route.
        raw.image_obs = False
        policy = AS019AutomaticPolicyWrapper(env.env, raw)
        summary["initial"] = {
            "action_shape": list(env.action_space.shape),
            "stage": int(raw._as019_stage),
            "contact_owner_evidence_available": bool(raw._as019_contact_available),
            "trajectory_state_write_count": int(raw._trajectory_state_write_count),
            "reset_failure_reason": str(reset_info.get("failure_reason", "")),
        }

        trace = []
        captures = []
        last_stage = -1
        last_info = {}
        terminated = False
        truncated = False
        success = False
        reward = 0.0
        for index in range(max(CAPTURE_STEPS)):
            action = policy.automatic_action()
            if action.shape != (54,) or not np.all(np.isfinite(action)):
                raise RuntimeError(
                    f"invalid automatic action at index {index}: shape={action.shape}"
                )
            next_step = int(raw.env_step) + 1
            raw.image_obs = next_step in CAPTURE_STEPS
            observation, reward, terminated, truncated, last_info = env.step(action)
            if raw.image_obs:
                captures.append(
                    {
                        "sample": _sample(raw, reward=reward),
                        "command_right_pose_wxyz": action[:7].tolist(),
                        "command_left_pose_wxyz": action[7:14].tolist(),
                        "command_right_hand_max": float(np.max(np.abs(action[14:34]))),
                        "command_left_hand_max": float(np.max(np.abs(action[34:54]))),
                        "images": _save_images(observation, int(raw.env_step)),
                    }
                )
                raw.image_obs = False
            stage = int(raw._as019_stage)
            if index % 10 == 0 or stage != last_stage or terminated or truncated:
                event = _sample(raw, reward=reward)
                trace.append(event)
                print("AS019_CORRECTED_PROGRESS=" + json.dumps(event, sort_keys=True), flush=True)
            last_stage = stage
            success = bool(last_info.get("succeed", False))
            if terminated or truncated or success:
                break

        final = _sample(raw, reward=reward)
        event_steps = final["event_steps"]
        ordered_events = bool(
            len(event_steps) == 5
            and all(value >= 0 for value in event_steps)
            and all(left < right for left, right in zip(event_steps, event_steps[1:]))
        )
        checks = {
            "action_54d": summary["initial"]["action_shape"] == [54],
            "contact_owner_evidence_available": bool(raw._as019_contact_available),
            "automatic_success": success,
            "stage_complete": int(raw._as019_stage) == 5,
            "ordered_events": ordered_events,
            "dual_before_right_release": bool(
                event_steps[1] >= 0
                and event_steps[2] > event_steps[1]
            ),
            "right_release_before_zone": bool(
                event_steps[2] >= 0
                and event_steps[3] > event_steps[2]
            ),
            "zone_before_stable_release": bool(
                event_steps[3] >= 0
                and event_steps[4] > event_steps[3]
            ),
            "both_hands_clear": bool(
                final["right_clear"]
                and final["left_clear"]
                and final["right_force"] <= _RELEASE_FORCE_N
                and final["left_force"] <= _RELEASE_FORCE_N
            ),
            "zone_inside": bool(final["zone_inside"]),
            "tip_outward": bool(final["outward_alignment"] >= _OUTWARD_COS),
            "stable_velocity": bool(
                final["linear_speed"] <= 0.020
                and final["angular_speed"] <= 0.20
            ),
            "trajectory_state_write_count_zero": int(raw._trajectory_state_write_count) == 0,
            "no_failure_reason": not bool(raw._as019_failure_reason),
        }
        summary["trajectory"] = {
            "steps": int(raw.env_step),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": success,
            "event_steps": event_steps,
            "final": final,
            "trace": trace,
            "last_info": last_info,
        }
        summary["captures"] = captures
        summary["checks"] = checks
        captured_steps = {item["sample"]["step"] for item in captures}
        summary["diagnostic_pass"] = bool(
            captured_steps == CAPTURE_STEPS
            and all(item["images"] for item in captures)
            and int(raw._trajectory_state_write_count) == 0
        )
        exit_code = 0 if summary["diagnostic_pass"] else 3
    except BaseException as exc:
        summary["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 2
    finally:
        # Persist the physical result before Kit shutdown. Isaac 5.1 can linger
        # in SimulationApp.close(); scheduler cleanup must not erase evidence.
        summary["elapsed_before_close_seconds"] = float(time.time() - started)
        summary["environment_closed"] = False
        _persist(summary)
        if env is not None:
            try:
                env.close()
                summary["environment_closed"] = True
            except BaseException as exc:
                summary["close_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                summary["elapsed_seconds"] = float(time.time() - started)
                _persist(summary)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
