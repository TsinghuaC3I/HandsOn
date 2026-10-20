#!/usr/bin/env python3
"""Validate a saved Synchronized Dual-Knob Setting collection episode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zarr


TASK_NAME = "synchronized_dual_knob_setting"
EXPECTED_ACTION_WIDTH = 46


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_report(path: Path, frames_dir: Path | None) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"video contains no decodable frames: {path}")

    selected = {
        "first": frames[0],
        "mid": frames[len(frames) // 2],
        "final": frames[-1],
    }
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for label, frame in selected.items():
            output = frames_dir / f"{path.stem}_{label}.png"
            if not cv2.imwrite(str(output), frame):
                raise RuntimeError(f"failed to write validation frame: {output}")

    stacked = np.stack([selected["first"], selected["mid"], selected["final"]])
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "codec": "H.264" if path.suffix.lower() == ".mp4" else path.suffix,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": len(frames),
        "selected_mean": [float(frame.mean()) for frame in stacked],
        "selected_black_fraction": [
            float(np.mean(np.all(frame < 5, axis=-1))) for frame in stacked
        ],
        "first_final_mean_abs_delta": float(
            np.mean(np.abs(stacked[0].astype(np.float32) - stacked[-1].astype(np.float32)))
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--policy-actions-npy", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--frames-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    episode_dir = args.episode_dir.resolve()
    actions_path = args.policy_actions_npy.resolve()
    required_files = [
        episode_dir / "collection_meta.json",
        episode_dir / "DATASET_PHYSX_V3.json",
        episode_dir / "recording_schema.json",
        episode_dir / "replay_manifest.json",
        episode_dir / "replay.zarr" / ".zgroup",
        episode_dir / "videos" / "main.mp4",
        episode_dir / "videos" / "wrist_left.mp4",
        episode_dir / "videos" / "wrist_right.mp4",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing episode artifacts: {missing}")

    collection_meta = json.loads((episode_dir / "collection_meta.json").read_text())
    manifest = json.loads((episode_dir / "replay_manifest.json").read_text())
    recording = manifest["recording"]
    source_actions = np.load(actions_path)
    root = zarr.open(str(episode_dir / "replay.zarr"), mode="r")
    data = root["data"]
    actions = np.asarray(data["action"])
    succeed = np.asarray(data["succeed"]).astype(bool)
    terminal = np.asarray(data["terminal"]).astype(bool)
    episode_ends = np.asarray(root["meta/episode_ends"])
    task_q = np.asarray(data["next_physx_task_articulation_dof_position"])
    task_qd = np.asarray(data["next_physx_task_articulation_dof_velocity"])

    steps = int(actions.shape[0])
    final_q_deg = np.rad2deg(task_q[-1, 0])
    left_magnitude_deg = float(-final_q_deg[0])
    right_angle_deg = float(final_q_deg[1])
    sum_residual_deg = left_magnitude_deg + right_angle_deg - 60.0
    difference_residual_deg = left_magnitude_deg - right_angle_deg + 10.0
    action_sha256 = _sha256(actions_path)

    stable_origin = recording["stable_recording_origin"]
    reset_contract = recording["reset_replay_contract"]
    checks = {
        "task_name": collection_meta.get("task_name") == TASK_NAME
        and manifest.get("task", {}).get("name") == TASK_NAME,
        "physx_v3": collection_meta.get("is_physx_v3_data") is True
        and recording.get("is_physx_v3_data") is True,
        "source_success": collection_meta.get("source_outcome") == "success"
        and collection_meta.get("task_outcome") == "success",
        "manual_only": collection_meta.get("postprocess_manual_only") is True
        and recording.get("postprocess_manual_only") is True,
        "policy_validation": recording.get("policy_action_validation") is True,
        "public_action_semantics": recording.get("action_semantics")
        == "public_policy_action_adapted_by_task_wrapper",
        "deterministic": recording.get("randomize") is False
        and recording.get("randomize_dynamics") is False,
        "policy_sha256": recording.get("policy_actions_sha256") == action_sha256,
        "action_shape": actions.shape == source_actions.shape
        and actions.ndim == 2
        and actions.shape[1] == EXPECTED_ACTION_WIDTH,
        "actions_exact": bool(np.array_equal(actions, source_actions)),
        "finite_arrays": bool(
            np.isfinite(actions).all()
            and np.isfinite(task_q).all()
            and np.isfinite(task_qd).all()
        ),
        "step_counts": collection_meta.get("step_count") == steps
        and recording.get("step_count") == steps
        and episode_ends.tolist() == [steps],
        "success_only_at_final": np.flatnonzero(succeed).tolist() == [steps - 1],
        "terminal_only_at_final": np.flatnonzero(terminal).tolist() == [steps - 1],
        "task_physx_shapes": task_q.shape == (steps, 1, 2)
        and task_qd.shape == (steps, 1, 2),
        "opposite_directions": left_magnitude_deg > 0.0 and right_angle_deg > 0.0,
        "final_relation": abs(sum_residual_deg) <= 4.0
        and abs(difference_residual_deg) <= 4.0,
        "no_trajectory_state_writes": stable_origin.get(
            "trajectory_per_frame_state_writes"
        )
        is False
        and stable_origin.get("per_frame_state_writes") is False
        and reset_contract.get("forbids_state_writes_after_action_start") is True,
        "recorded_physx_ground_truth": recording.get("record_physx_ground_truth") is True,
    }

    video_reports = {
        name: _video_report(episode_dir / "videos" / f"{name}.mp4", args.frames_dir)
        for name in ("main", "wrist_left", "wrist_right")
    }
    checks["videos_decodable"] = all(
        report["frame_count"] == steps
        and report["fps"] > 0.0
        and report["width"] > 0
        and report["height"] > 0
        for report in video_reports.values()
    )
    checks["videos_non_black"] = all(
        max(report["selected_black_fraction"]) < 0.98
        and min(report["selected_mean"]) > 3.0
        for report in video_reports.values()
    )
    checks["videos_change_over_episode"] = all(
        report["first_final_mean_abs_delta"] > 1.0
        for report in video_reports.values()
    )

    report = {
        "format": "handson.synchronized_dual_knob.collection_validation",
        "version": 1,
        "passed": all(checks.values()),
        "episode_dir": str(episode_dir),
        "episode_sha256": {
            name: _sha256(episode_dir / name)
            for name in (
                "collection_meta.json",
                "DATASET_PHYSX_V3.json",
                "recording_schema.json",
                "replay_manifest.json",
            )
        },
        "policy_actions": {
            "path": str(actions_path),
            "sha256": action_sha256,
            "shape": list(source_actions.shape),
            "zarr_dtype": str(actions.dtype),
            "source_dtype": str(source_actions.dtype),
            "max_abs_diff": float(np.max(np.abs(actions - source_actions))),
        },
        "final_task_state": {
            "q_deg": final_q_deg.tolist(),
            "qd_deg_s": np.rad2deg(task_qd[-1, 0]).tolist(),
            "sum_residual_deg": sum_residual_deg,
            "difference_residual_deg": difference_residual_deg,
        },
        "success_indices": np.flatnonzero(succeed).tolist(),
        "terminal_indices": np.flatnonzero(terminal).tolist(),
        "checks": checks,
        "videos": video_reports,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
