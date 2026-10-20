#!/usr/bin/env python3
"""Record one automated, action-aligned B320-HS005 PhysX episode."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "handson"))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ["HANDSON_ENABLE_ISAACSIM_NEWTON"] = "0"
os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
os.environ["HANDSON_STABLE_RECORDING_ORIGIN"] = "1"

from isaaclab.app import AppLauncher  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--seed", type=int, default=1705)
    AppLauncher.add_app_launcher_args(result)
    return result


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), *args], text=True
    ).strip()


def main() -> int:
    args = parser().parse_args()
    args.headless = True
    args.enable_cameras = True
    args.device = "cuda:0"
    args.width = 640
    args.height = 480
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    launcher = AppLauncher(args)
    app = launcher.app

    import numpy as np
    from PIL import Image

    from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
    from handson.sim.isaaclab.recording_origin import stabilize_recording_origin
    from taskdev.tasks.bench320_hs005_sink_bottle_stabilized_scrub.automated_policy import (
        HS005AutomatedPolicy,
    )
    from taskdev.tasks.bench320_hs005_sink_bottle_stabilized_scrub.scene_contract import (
        sha256_file,
    )
    from taskdev.tasks.bench320_hs005_sink_bottle_stabilized_scrub.vertical_slice_io import (
        HS005EpisodeWriter,
        atomic_json,
        jsonable,
    )

    environment = None
    started = time.time()
    summary = {
        "schema": "handson.b320_hs005.collection_run",
        "version": 1,
        "benchmark_task_id": "B320-HS005",
        "task_name": "t37_scrub_bottle",
        "collection_pass": False,
        "task_complete": False,
    }
    rc = 1
    try:
        if git("status", "--porcelain"):
            raise RuntimeError("HS005 collection requires a clean checkpoint worktree")
        commit = git("rev-parse", "HEAD")
        metadata = {
            "git_commit": commit,
            "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "seed": int(args.seed),
            "runtime": "Isaac Sim 5.1 + IsaacLab 2.3.2 + PhysX",
            "physics_backend": "physx",
            "physics_hz": 240,
            "control_hz": 30,
            "raw_action_layout": "right_tcp7,left_tcp7,right_wuji20,left_wuji20",
            "camera_streams": ["main", "wrist_left", "wrist_right"],
        }
        environment = HandsOnABCIsaacLabEnv(
            "t37_scrub_bottle",
            seed=int(args.seed),
            control_dt=1.0 / 30.0,
            physics_dt=1.0 / 240.0,
            time_limit=45.0,
            render_mode="rgb_array",
            image_obs=True,
            randomize=True,
            randomize_dynamics=True,
            hand_type="wuji1",
            arm_type="rokae",
            launch_isaac=True,
            headless=True,
            render_resolution=(480, 640),
            hz=30,
            isaac_device=args.device,
            physics_backend="physx",
            existing_simulation_app=app,
            existing_app_launcher=launcher,
        )
        observation, reset_info = environment.reset(seed=int(args.seed))
        origin_kwargs = {
            "initial_settle_seconds": 0.50,
            "stable_seconds": 0.25,
            "stability_sample_seconds": 0.25,
            "timeout_seconds": 12.0,
            "linear_velocity_limit": 0.03,
            "angular_velocity_limit": 0.40,
            "pose_translation_limit": 0.001,
            "pose_rotation_limit": 0.02,
        }
        observation, reset_info = stabilize_recording_origin(
            environment,
            observation,
            reset_info,
            **origin_kwargs,
        )
        state = observation["state"]
        policy = HS005AutomatedPolicy(
            state["hs005_bottle_pose"],
            state["hs005_brush_pose"],
            state["tcp_pose"],
            observation["physx_tcp_state"],
            observation["physx_tcp_valid"],
            state["hs005_reset_parameters"],
        )
        episode_dir = output_dir / "b320_hs005_demo_000__physx_v3"
        writer = HS005EpisodeWriter(episode_dir, metadata)
        recording_schema = environment.get_recording_schema()
        if not isinstance(recording_schema, dict):
            raise RuntimeError("HS005 PhysX recording schema is unavailable")
        atomic_json(episode_dir / "recording_schema.json", recording_schema)
        atomic_json(
            episode_dir / "stable_recording_origin.json",
            reset_info["physical_recording_origin"],
        )
        task_tcp_contract = policy.reachability_report
        atomic_json(episode_dir / "task_tcp_contract.json", task_tcp_contract)

        final_info = dict(reset_info)
        for _frame in range(policy.total_frames):
            action = policy.action(
                observation["physx_tcp_state"],
                observation["physx_tcp_valid"],
            )
            next_observation, reward, terminated, truncated, info = environment.step(action)
            writer.append(
                observation,
                action,
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            )
            observation = next_observation
            final_info = dict(info)
            if bool(info.get("succeed")) or terminated or truncated:
                break

        main_frame = np.asarray(
            observation["images"].get(
                "random_camera",
                observation["images"].get("ego"),
            )
        )
        preview_path = episode_dir / "final_main.png"
        Image.fromarray(main_frame).save(preview_path, compress_level=0)
        write_count = int(environment._trajectory_state_write_count)
        success = bool(final_info.get("succeed", False))
        contact_view_count = len(environment._hs005_contact_views)
        collection_pass = bool(
            success
            and write_count == 0
            and 0 < writer.frame_count <= policy.total_frames
            and contact_view_count == 3
            and bool(final_info.get("hs005_runtime_valid", False))
        )
        final_task_info = jsonable(
            {
                key: value
                for key, value in final_info.items()
                if str(key).startswith("hs005_")
            }
        )
        manifest = writer.finalize(
            {
                # Fresh replay is intentionally fail-closed on the exact
                # producer checkpoint and seed.  Bind the complete producer
                # metadata into the episode manifest itself, not only the
                # sibling run summary, so every replay preflight input is
                # hash-bound to the immutable episode contract.
                **metadata,
                "collection_pass": collection_pass,
                "success": success,
                "trajectory_state_write_count": write_count,
                "planned_frame_count": int(policy.total_frames),
                "contact_view_count": int(contact_view_count),
                "final_task_info": final_task_info,
                "recording_schema_sha256": sha256_file(
                    episode_dir / "recording_schema.json"
                ),
                "stable_origin_sha256": sha256_file(
                    episode_dir / "stable_recording_origin.json"
                ),
                "task_tcp_contract_sha256": sha256_file(
                    episode_dir / "task_tcp_contract.json"
                ),
                "task_tcp_contract": task_tcp_contract,
                "preview_sha256": sha256_file(preview_path),
            }
        )
        summary.update(
            {
                **metadata,
                **manifest,
                "collection_pass": collection_pass,
                "success": success,
                "trajectory_state_write_count": write_count,
                "final_task_info": final_task_info,
                "stable_recording_origin": reset_info["physical_recording_origin"],
                "task_tcp_contract": task_tcp_contract,
                "preview_path": str(preview_path),
            }
        )
        rc = 0 if collection_pass else 2
    except Exception as exc:
        summary["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        diagnostics = getattr(exc, "diagnostics", None)
        if isinstance(diagnostics, dict):
            summary["exception"]["diagnostics"] = jsonable(diagnostics)
        rc = 1
    finally:
        summary["elapsed_wall_seconds"] = time.time() - started
        summary["exit_code"] = rc
        summary["cleanup_policy"] = {
            "evidence_written_before_teardown": True,
            "kit_cleanup_watchdog_seconds": 20.0,
        }
        summary_path = output_dir / "collection_summary.json"
        marker_path = output_dir / (
            "COLLECTION_PASS.json" if rc == 0 else "COLLECTION_NON_PASS.json"
        )
        atomic_json(summary_path, summary)
        atomic_json(
            marker_path,
            {
                "collection_pass": rc == 0,
                "task_complete": False,
                "summary": str(summary_path),
            },
        )

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
            rc = 1
            summary["collection_pass"] = False
            summary["exit_code"] = rc
            atomic_json(summary_path, summary)
            marker_path.unlink(missing_ok=True)
            atomic_json(
                output_dir / "COLLECTION_NON_PASS.json",
                {
                    "collection_pass": False,
                    "task_complete": False,
                    "summary": str(summary_path),
                },
            )
        else:
            shutdown_watchdog.cancel()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
