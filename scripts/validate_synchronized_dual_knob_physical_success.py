#!/usr/bin/env python3
"""Prove dual-knob success using only the public 46-D policy action path."""

from __future__ import annotations

import argparse
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
for candidate in (REPO_ROOT, REPO_ROOT / "handson"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

TASK_NAME = "synchronized_dual_knob_setting"
SEED = 20260819
CONTROL_HZ = 30
TARGET_X_DEG = 25.0
TARGET_Y_DEG = 35.0

# Deterministic, randomize=False reset poses measured at the public palm-control
# frames.  They are controller constants, not simulator state writes.
RIGHT_START_POSE = np.asarray(
    (-0.37149766, -0.23950160, 0.99751306, 0.707126, 0.0, 0.707088, 0.0),
    dtype=np.float64,
)
LEFT_START_POSE = np.asarray(
    (-0.37149757, 0.23950188, 0.99751270, 0.707126, 0.0, 0.707088, 0.0),
    dtype=np.float64,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capture-every", type=int, default=2)
    parser.add_argument("--max-feedback-steps", type=int, default=120)
    return parser


def _unwrap(environment: Any) -> Any:
    current = environment
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_dual_knob_tracker"):
            return current
        current = getattr(current, "env", getattr(current, "_env", None))
        if current is None:
            break
    raise RuntimeError("Could not unwrap the dual-knob environment")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _normalized_lerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    first = np.asarray(q0, dtype=np.float64)
    second = np.asarray(q1, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    if float(np.dot(first, second)) < 0.0:
        second = -second
    result = (1.0 - fraction) * first + fraction * second
    return result / np.linalg.norm(result)


def _pack(
    right_pose: np.ndarray,
    left_pose: np.ndarray,
    right_hand: np.ndarray,
    left_hand: np.ndarray,
) -> np.ndarray:
    return np.concatenate((right_pose, left_pose, right_hand, left_hand)).astype(
        np.float32
    )


def _resize_nearest(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    array = np.asarray(frame, dtype=np.uint8)[..., :3]
    if array.shape[:2] == (height, width):
        return array
    ys = np.clip(
        (np.arange(height) * array.shape[0] / height).astype(np.int64),
        0,
        array.shape[0] - 1,
    )
    xs = np.clip(
        (np.arange(width) * array.shape[1] / width).astype(np.int64),
        0,
        array.shape[1] - 1,
    )
    return array[ys][:, xs]


def _render_composite(observation: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    # HandsOnObsAdapter intentionally promotes the camera dict into top-level
    # policy-observation keys.
    main = observation.get("ego", observation.get("random_camera"))
    wrist_left = observation.get("wrist_left")
    wrist_right = observation.get("wrist_right")
    if main is None or wrist_left is None or wrist_right is None:
        raise RuntimeError(
            f"Public observation has incomplete cameras: {sorted(observation)}"
        )
    frames = [np.asarray(value, dtype=np.uint8)[..., :3] for value in (main, wrist_left, wrist_right)]
    height = min(frame.shape[0] for frame in frames)
    width = min(frame.shape[1] for frame in frames)
    resized = [_resize_nearest(frame, height, width) for frame in frames]
    composite = np.concatenate(resized, axis=1)
    metrics = {
        "shape": list(composite.shape),
        "main_mean_rgb": float(np.mean(resized[0])),
        "wrist_left_mean_rgb": float(np.mean(resized[1])),
        "wrist_right_mean_rgb": float(np.mean(resized[2])),
        "black_fraction": float(np.mean(np.all(composite == 0, axis=-1))),
    }
    return composite, metrics


def _save_png(path: Path, frame: np.ndarray) -> None:
    try:
        import imageio.v3 as iio

        iio.imwrite(path, np.asarray(frame, dtype=np.uint8))
    except Exception:
        from PIL import Image

        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path)


def _public_state_layout(environment: Any) -> dict[str, tuple[int, int]]:
    """Return slices for fields in the public flattened policy observation."""

    keys = tuple(environment.proprio_keys)
    spaces = environment.proprio_space.spaces
    layout: dict[str, tuple[int, int]] = {}
    offset = 0
    for key in keys:
        size = int(np.prod(spaces[key].shape, dtype=np.int64))
        layout[str(key)] = (offset, offset + size)
        offset += size
    expected = {
        "knob_joint_state",
        "relation_residuals",
        "knob_contact_state",
        "synchrony_state",
        "dual_knob_hold",
    }
    missing = expected.difference(layout)
    if missing:
        raise RuntimeError(f"Public policy state is missing fields: {sorted(missing)}")
    if tuple(environment.observation_space["state"].shape) != (offset,):
        raise RuntimeError(
            "Public policy observation layout does not match its declared space: "
            f"layout={offset}, space={environment.observation_space['state'].shape}"
        )
    return layout


def _state_row(
    observation: dict[str, Any], layout: dict[str, tuple[int, int]]
) -> dict[str, Any]:
    flat_state = np.asarray(observation["state"], dtype=np.float64).reshape(-1)

    def field(name: str) -> np.ndarray:
        start, stop = layout[name]
        return flat_state[start:stop]

    joint = field("knob_joint_state")
    contact = field("knob_contact_state")
    synchrony = field("synchrony_state")
    hold = field("dual_knob_hold")
    residuals = field("relation_residuals")
    return {
        "q_left_deg": float(joint[0]),
        "q_right_deg": float(joint[1]),
        "qd_left_deg_s": float(joint[2]),
        "qd_right_deg_s": float(joint[3]),
        "left_assigned_force_n": float(contact[0]),
        "right_assigned_force_n": float(contact[1]),
        "left_cross_force_n": float(contact[2]),
        "right_cross_force_n": float(contact[3]),
        "start_delta_s": float(synchrony[0]),
        "entry_delta_s": float(synchrony[1]),
        "start_sync_ok": bool(synchrony[2]),
        "entry_sync_ok": bool(synchrony[3]),
        "hold_frames": int(round(float(hold[0]))),
        "hold_frames_required": int(round(float(hold[1]))),
        "relation_primary_residual": float(residuals[0]),
        "relation_secondary_residual": float(residuals[1]),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.capture_every < 1:
        raise ValueError("--capture-every must be positive")
    if not 30 <= args.max_feedback_steps <= 300:
        raise ValueError("--max-feedback-steps must be in [30, 300]")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")

    from handson.tasks.mappings import CONFIG_MAPPING

    environment = CONFIG_MAPPING[TASK_NAME]().get_native_environment(
        policy_mode=True,
        sim_backend="isaacsim",
        launch_isaac=True,
        image_obs=True,
        randomize=False,
        randomize_dynamics=False,
        render_mode="rgb_array",
        headless=True,
        isaac_device=args.device,
        hz=CONTROL_HZ,
        control_dt=1.0 / CONTROL_HZ,
        physics_dt=1.0 / 120.0,
        time_limit=32.0,
        render_resolution=(320, 240),
    )
    raw = _unwrap(environment)
    # Keep close() from calling os._exit(0), which would mask controller or
    # evidence-validation exceptions.  The bounded shutdown guard below still
    # prevents Isaac Sim teardown from hanging indefinitely and preserves the
    # intended process exit code.
    raw._fast_shutdown_on_close = False
    intended_exit_code = 1
    video_writer = None
    try:
        observation, reset_info = environment.reset(seed=SEED)
        if tuple(environment.action_space.shape) != (46,):
            raise RuntimeError(
                f"Expected public action shape (46,), got {environment.action_space.shape}"
            )
        state_layout = _public_state_layout(environment)
        write_count_start = int(raw._trajectory_state_write_count)
        write_journal_start = list(raw._trajectory_state_write_journal)
        trace: list[dict[str, Any]] = []
        actions: list[np.ndarray] = []
        hand = np.zeros(16, dtype=np.float64)
        frame_count = 0
        last_video_frame = -1

        import imageio.v2 as iio

        video_path = output_dir / "dual_knob_three_view.mp4"
        video_writer = iio.get_writer(
            str(video_path),
            fps=CONTROL_HZ / args.capture_every,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        first_frame, first_camera_metrics = _render_composite(observation)
        _save_png(output_dir / "frame_000_reset.png", first_frame)
        video_writer.append_data(first_frame)
        last_video_frame = 0

        terminated = False
        truncated = False
        success = False
        reward = 0.0
        final_info: dict[str, Any] = {}
        phase_name = "reset"
        left_fraction = 0.0
        right_fraction = 0.0

        def step(
            right_pose: np.ndarray,
            left_pose: np.ndarray,
            phase: str,
        ) -> bool:
            nonlocal observation, terminated, truncated, success, reward
            nonlocal final_info, frame_count, last_video_frame, phase_name
            phase_name = phase
            action = _pack(right_pose, left_pose, hand, hand)
            observation, reward, terminated, truncated, final_info = environment.step(
                action
            )
            frame_count += 1
            metrics = _state_row(observation, state_layout)
            row = {
                "frame": frame_count,
                "phase": phase,
                "left_path_fraction": float(left_fraction),
                "right_path_fraction": float(right_fraction),
                "reward": float(reward),
                "succeed": bool(final_info.get("succeed")),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                **metrics,
            }
            trace.append(row)
            actions.append(action.copy())
            if frame_count % args.capture_every == 0 or final_info.get("succeed"):
                camera_frame, _camera_metrics = _render_composite(observation)
                video_writer.append_data(camera_frame)
                last_video_frame = frame_count
            if frame_count <= 3 or frame_count % 10 == 0 or final_info.get("succeed"):
                print(json.dumps(row, sort_keys=True), flush=True)
            success = success or bool(final_info.get("succeed"))
            return not (terminated or truncated or success)

        def drive(
            starts: dict[str, np.ndarray],
            targets: dict[str, np.ndarray],
            steps: int,
            phase: str,
        ) -> bool:
            for index in range(1, steps + 1):
                fraction = index / steps
                poses: dict[str, np.ndarray] = {}
                for side in ("right", "left"):
                    pose = (1.0 - fraction) * starts[side] + fraction * targets[side]
                    pose[3:7] = _normalized_lerp(
                        starts[side][3:7], targets[side][3:7], fraction
                    )
                    poses[side] = pose
                if not step(poses["right"], poses["left"], phase):
                    return False
            return True

        starts = {
            "right": RIGHT_START_POSE.copy(),
            "left": LEFT_START_POSE.copy(),
        }
        for _ in range(8):
            if not step(starts["right"], starts["left"], "settle"):
                break

        above = {side: starts[side].copy() for side in ("right", "left")}
        for pose in above.values():
            pose[0] += 0.010
            pose[2] -= 0.040
        if not (terminated or truncated or success):
            drive(starts, above, 32, "align_rear")

        contact = {side: above[side].copy() for side in ("right", "left")}
        contact["right"][1] += 0.064
        contact["left"][1] -= 0.064
        if not (terminated or truncated or success):
            drive(above, contact, 32, "engage_rear")

        left_fraction = 0.10
        right_fraction = 0.0
        for _ in range(args.max_feedback_steps):
            if terminated or truncated or success:
                break
            public_metrics = _state_row(observation, state_layout)
            x_deg = -public_metrics["q_left_deg"]
            y_deg = public_metrics["q_right_deg"]
            left_fraction = float(
                np.clip(
                    left_fraction
                    + np.clip(0.004 * (TARGET_X_DEG - x_deg), -0.012, 0.012),
                    0.0,
                    1.0,
                )
            )
            right_fraction = float(
                np.clip(
                    right_fraction
                    + np.clip(0.004 * (TARGET_Y_DEG - y_deg), -0.012, 0.012),
                    0.0,
                    1.0,
                )
            )
            pushed = {side: contact[side].copy() for side in ("right", "left")}
            pushed["left"][0] += left_fraction * 0.030
            pushed["right"][0] += right_fraction * 0.045
            pushed["left"][1] += left_fraction * 0.015
            pushed["right"][1] -= right_fraction * 0.015
            step(pushed["right"], pushed["left"], "feedback_tangent")

        if last_video_frame != frame_count:
            last_frame, _last_metrics = _render_composite(observation)
            video_writer.append_data(last_frame)
        final_frame, final_camera_metrics = _render_composite(observation)
        reset_to_success_mean_abs_delta = float(
            np.mean(
                np.abs(
                    final_frame.astype(np.int16) - first_frame.astype(np.int16)
                )
            )
        )
        _save_png(output_dir / f"frame_{frame_count:03d}_success.png", final_frame)
        video_writer.close()
        video_writer = None

        final_metrics = _state_row(observation, state_layout)
        write_count_end = int(raw._trajectory_state_write_count)
        write_journal_end = list(raw._trajectory_state_write_journal)
        if not success or not terminated or truncated:
            raise RuntimeError(
                "Physical controller did not produce terminal task success: "
                f"success={success} terminated={terminated} truncated={truncated} "
                f"metrics={final_metrics}"
            )
        if final_metrics["hold_frames"] < final_metrics["hold_frames_required"]:
            raise RuntimeError(f"Success hold is incomplete: {final_metrics}")
        if write_count_end != write_count_start or write_journal_end != write_journal_start:
            raise RuntimeError(
                "Direct simulator state writes occurred after action start: "
                f"count={write_count_start}->{write_count_end}, journal={write_journal_end}"
            )
        if max(
            final_camera_metrics["black_fraction"],
            0.0,
        ) >= 0.95 or min(
            final_camera_metrics["main_mean_rgb"],
            final_camera_metrics["wrist_left_mean_rgb"],
            final_camera_metrics["wrist_right_mean_rgb"],
        ) <= 30.0:
            raise RuntimeError(f"Final camera energy check failed: {final_camera_metrics}")
        if reset_to_success_mean_abs_delta <= 1.0:
            raise RuntimeError(
                "Reset and success camera frames did not change materially: "
                f"mean_abs_delta={reset_to_success_mean_abs_delta:.6f}"
            )

        actions_path = output_dir / "policy_actions.npy"
        np.save(actions_path, np.asarray(actions, dtype=np.float32))
        result = {
            "task_name": TASK_NAME,
            "seed": SEED,
            "controller": "public_46d_pose_and_policy_hand_actions",
            "controller_reads": [
                "observation.state.knob_joint_state",
                "observation.state.knob_contact_state",
                "observation.state.synchrony_state",
                "observation.state.dual_knob_hold",
                "info.succeed",
            ],
            "action_shape": list(environment.action_space.shape),
            "public_state_layout": state_layout,
            "action_count": len(actions),
            "actions_path": str(actions_path),
            "video_path": str(video_path),
            "first_camera_metrics": first_camera_metrics,
            "final_camera_metrics": final_camera_metrics,
            "reset_to_success_mean_abs_delta": reset_to_success_mean_abs_delta,
            "reset_info": reset_info,
            "final_info": final_info,
            "final_metrics": final_metrics,
            "trajectory_state_write_count_start": write_count_start,
            "trajectory_state_write_count_end": write_count_end,
            "trajectory_state_write_journal": write_journal_end,
            "trace": trace,
        }
        result_path = output_dir / "physical_success.json"
        result_path.write_text(
            json.dumps(_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(f"PHYSICAL_SUCCESS_JSON={result_path}", flush=True)
        print(f"PHYSICAL_SUCCESS_VIDEO={video_path}", flush=True)
        print(
            "DUAL_KNOB_PHYSICAL_SUCCESS_PASSED "
            f"frames={frame_count} hold={final_metrics['hold_frames']} "
            f"q=({final_metrics['q_left_deg']:.3f},{final_metrics['q_right_deg']:.3f}) "
            f"state_writes={write_count_end - write_count_start}",
            flush=True,
        )
        intended_exit_code = 0
        return 0
    except BaseException as exc:
        failure_path = output_dir / "physical_success_failure.json"
        failure_path.write_text(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        traceback.print_exc()
        raise
    finally:
        if video_writer is not None:
            video_writer.close()
        close_code = intended_exit_code if sys.exc_info()[0] is None else 1
        sys.stdout.flush()
        sys.stderr.flush()
        shutdown_guard = threading.Timer(10.0, lambda: os._exit(close_code))
        shutdown_guard.daemon = True
        shutdown_guard.start()
        try:
            environment.close()
        finally:
            shutdown_guard.cancel()


if __name__ == "__main__":
    raise SystemExit(main())
