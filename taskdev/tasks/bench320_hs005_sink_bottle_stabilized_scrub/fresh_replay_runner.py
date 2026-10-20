#!/usr/bin/env python3
"""Fresh-process, full-trajectory PhysX replay audit for B320-HS005."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
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
    result.add_argument("episode_dir", type=Path)
    result.add_argument("--report", type=Path, required=True)
    result.add_argument("--global-tolerance", type=float, default=1.0e-4)
    AppLauncher.add_app_launcher_args(result)
    return result


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), *args], text=True
    ).strip()


def _audit_schema(schema: dict) -> dict:
    """Adapt the environment schema to the reusable trajectory auditor."""

    result = dict(schema)
    fields = schema.get("fields", {})
    result["fields"] = {
        str(name): {**dict(spec), "semantics": dict(spec)}
        for name, spec in fields.items()
        if isinstance(spec, dict)
    }
    return result


def _hidden_comparison(expected: dict, actual: dict, step: int) -> dict:
    first = None
    maximum = 0.0
    compared = 0
    for name, lhs in expected.items():
        if name not in actual:
            if first is None:
                first = {"step_zero_based": step, "field": name, "reason": "missing"}
            continue
        import numpy as np

        left = np.asarray(lhs)
        right = np.asarray(actual[name])
        compared += 1
        exact_discrete = left.dtype.kind in "biu" or right.dtype.kind in "biu"
        if left.shape != right.shape:
            if first is None:
                first = {"step_zero_based": step, "field": name, "reason": "shape"}
            continue
        error = float(np.max(np.abs(left.astype(float) - right.astype(float)))) if left.size else 0.0
        maximum = max(maximum, error)
        passed = bool(np.array_equal(left, right)) if exact_discrete else error <= 1.0e-8
        if not passed and first is None:
            first = {
                "step_zero_based": step,
                "field": name,
                "reason": "value",
                "max_abs_error": error,
            }
    return {"compared_field_count": compared, "max_abs_error": maximum, "first_divergence": first}


def main() -> int:
    args = parser().parse_args()
    args.headless = True
    args.enable_cameras = True
    args.device = "cuda:0"
    args.width = 640
    args.height = 480
    episode_dir = args.episode_dir.resolve()
    report_path = args.report.resolve()
    launcher = AppLauncher(args)
    app = launcher.app

    import numpy as np
    import zarr

    from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
    from handson.sim.isaaclab.articulated_replay.drive_path import (
        RecordedArticulatedDrivePath,
        run_articulated_drive_control_step,
        validate_recorded_physics_configuration,
    )
    from handson.sim.isaaclab.articulated_replay.trajectory_audit import TrajectoryAudit
    from handson.sim.isaaclab.recording_origin import (
        stable_recording_origin_kwargs_from_report,
        stabilize_recording_origin,
    )
    from taskdev.tasks.bench320_hs005_sink_bottle_stabilized_scrub.vertical_slice_io import (
        atomic_json,
        jsonable,
        load_zarr_frame,
        sha256_file,
        sha256_tree,
    )

    environment = None
    started = time.time()
    report = {
        "schema": "handson.b320_hs005.fresh_replay",
        "version": 1,
        "benchmark_task_id": "B320-HS005",
        "task_name": "t37_scrub_bottle",
        "fresh_process": True,
        "full_trajectory_executed": False,
        "state_restore_after_action_zero": False,
        "replay_pass": False,
        "task_complete": False,
    }
    rc = 1
    try:
        if git("status", "--porcelain"):
            raise RuntimeError("HS005 replay requires a clean checkpoint worktree")
        manifest_path = episode_dir / "episode_manifest.json"
        schema_path = episode_dir / "recording_schema.json"
        stable_path = episode_dir / "stable_recording_origin.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        stable_report = json.loads(stable_path.read_text(encoding="utf-8"))
        if manifest.get("collection_pass") is not True:
            raise RuntimeError("Fresh replay source episode is not a collection PASS")
        if manifest.get("git_commit") != git("rev-parse", "HEAD"):
            raise RuntimeError("Fresh replay commit differs from collection commit")
        if manifest.get("zarr_sha256") != sha256_tree(episode_dir / "replay.zarr"):
            raise RuntimeError("Fresh replay source Zarr tree hash mismatch")
        if manifest.get("recording_schema_sha256") != sha256_file(schema_path):
            raise RuntimeError("Fresh replay source schema hash mismatch")
        if manifest.get("stable_origin_sha256") != sha256_file(stable_path):
            raise RuntimeError("Fresh replay source stable-origin hash mismatch")

        seed = int(manifest["seed"])
        environment = HandsOnABCIsaacLabEnv(
            "t37_scrub_bottle",
            seed=seed,
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
        observation, reset_info = environment.reset(seed=seed)
        observation, reset_info = stabilize_recording_origin(
            environment,
            observation,
            reset_info,
            **stable_recording_origin_kwargs_from_report(stable_report),
        )
        runtime_schema = environment.get_recording_schema()
        if not isinstance(runtime_schema, dict):
            raise RuntimeError("Fresh replay runtime PhysX schema is unavailable")
        configuration = validate_recorded_physics_configuration(
            recorded_schema, runtime_schema
        )
        source_schema = _audit_schema(recorded_schema)
        live_schema = _audit_schema(runtime_schema)
        audit = TrajectoryAudit(
            source_schema,
            global_tolerance=float(args.global_tolerance),
            runtime_schema=live_schema,
            include_nonreplayable_contact_outputs=False,
        )
        origin_audit = TrajectoryAudit(
            source_schema,
            global_tolerance=float(args.global_tolerance),
            runtime_schema=live_schema,
            include_nonreplayable_contact_outputs=False,
        )
        drive_path = RecordedArticulatedDrivePath(recorded_schema)
        root = zarr.open_group(str(episode_dir / "replay.zarr"), mode="r")
        data = root["data"]
        actions = data["raw_action"]
        expected_frames = int(manifest["frame_count"])
        if len(actions) != expected_frames:
            raise RuntimeError("Fresh replay action count differs from manifest")
        expected_current = load_zarr_frame(data["physical/current"], 0)
        origin_result = origin_audit.compare_step(expected_current, observation, 0)
        if not origin_result["within_schema_tolerance"]:
            raise RuntimeError("Fresh seeded stable origin differs from source; snapshot restore is forbidden")

        hidden_first = None
        hidden_max = 0.0
        applied_target_count = 0
        final_info = dict(reset_info)
        for step in range(expected_frames):
            raw_action = np.asarray(actions[step], dtype=np.float32)
            expected_next = load_zarr_frame(data["physical/next"], step)
            expected_hidden = load_zarr_frame(data["task_hidden/next"], step)
            environment._apply_action(raw_action)
            drive_step = drive_path.control_step(expected_next)
            result, drive_report = run_articulated_drive_control_step(
                environment,
                environment._physx_ground_truth_trace,
                recorded_schema,
                drive_step,
            )
            observation, _reward, terminated, truncated, info = result
            audit.compare_step(expected_next, observation, step)
            hidden = _hidden_comparison(expected_hidden, observation["state"], step)
            hidden_max = max(hidden_max, float(hidden["max_abs_error"]))
            if hidden_first is None and hidden["first_divergence"] is not None:
                hidden_first = hidden["first_divergence"]
            applied_target_count += int(drive_report["applied_robot_joint_target_count"])
            final_info = dict(info)
            source_terminated = bool(data["terminated"][step])
            source_truncated = bool(data["truncated"][step])
            if bool(terminated) != source_terminated or bool(truncated) != source_truncated:
                raise RuntimeError(f"Fresh replay terminal flags diverged at step {step}")

        audit_report = audit.report()
        origin_report = origin_audit.report()
        source_success = bool(data["succeed"][-1])
        replay_success = bool(final_info.get("succeed", False))
        state_write_count = int(environment._trajectory_state_write_count)
        replay_pass = bool(
            audit_report["strict_schema_pass"]
            and origin_report["strict_schema_pass"]
            and hidden_first is None
            and source_success
            and replay_success
            and state_write_count == 0
        )
        report.update(
            {
                "git_commit": git("rev-parse", "HEAD"),
                "source_manifest_sha256": sha256_file(manifest_path),
                "source_zarr_sha256": manifest["zarr_sha256"],
                "seed": seed,
                "physics_configuration": configuration,
                "full_trajectory_executed": True,
                "compared_control_steps": expected_frames,
                "applied_robot_joint_target_count": applied_target_count,
                "trajectory_state_write_count": state_write_count,
                "source_success": source_success,
                "replay_success": replay_success,
                "origin_audit": origin_report,
                "trajectory_audit": audit_report,
                "hidden_state_audit": {
                    "first_divergence": hidden_first,
                    "max_abs_error": hidden_max,
                },
                "final_task_info": jsonable(
                    {key: value for key, value in final_info.items() if str(key).startswith("hs005_")}
                ),
                "replay_pass": replay_pass,
                "task_complete": replay_pass,
            }
        )
        rc = 0 if replay_pass else 2
    except Exception as exc:
        report["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        rc = 1
    finally:
        report["elapsed_wall_seconds"] = time.time() - started
        report["exit_code"] = rc
        report["cleanup_policy"] = {
            "evidence_written_before_teardown": True,
            "kit_cleanup_watchdog_seconds": 20.0,
        }
        marker_path = report_path.with_name(
            "FRESH_REPLAY_PASS.json" if rc == 0 else "FRESH_REPLAY_NON_PASS.json"
        )
        atomic_json(report_path, report)
        atomic_json(
            marker_path,
            {
                "replay_pass": rc == 0,
                "task_complete": rc == 0,
                "report": str(report_path),
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
            report["environment_close_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            rc = 1
            report["replay_pass"] = False
            report["task_complete"] = False
            report["exit_code"] = rc
            atomic_json(report_path, report)
            marker_path.unlink(missing_ok=True)
            atomic_json(
                report_path.with_name("FRESH_REPLAY_NON_PASS.json"),
                {
                    "replay_pass": False,
                    "task_complete": False,
                    "report": str(report_path),
                },
            )
        else:
            shutdown_watchdog.cancel()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
