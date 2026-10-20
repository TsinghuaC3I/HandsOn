#!/usr/bin/env python3
"""Run the bounded B320-HS014 core-scene and zero-action audit."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import traceback
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "handson"))

# AppLauncher must be the first Isaac/Kit import.
from isaaclab.app import AppLauncher  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _namespace_contract() -> dict[str, Any]:
    physical = os.environ.get("TASKDEV_PHYSICAL_GPU", "").strip()
    physical_uuid = os.environ.get("TASKDEV_PHYSICAL_GPU_UUID", "").strip()
    requested_host = os.environ.get("TASKDEV_HOST", "").strip()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    isolation_mode = os.environ.get("TASKDEV_GPU_ISOLATION_MODE", "").strip()
    observed = socket.gethostname()
    private_namespace = isolation_mode == "private_mount_namespace" and visible == ""
    authorized_no_mask = bool(
        isolation_mode == "authorized_lease_backed_gpu0_no_device_mask"
        and physical_uuid.startswith("GPU-")
        and visible == physical_uuid
    )
    return {
        "requested_host": requested_host,
        "observed_hostname": observed,
        "physical_gpu": int(physical) if physical.isdigit() else None,
        "physical_gpu_uuid": physical_uuid,
        "logical_gpu": 0,
        "cuda_visible_devices": visible,
        "isolation_mode": isolation_mode,
        "pass": bool(
            requested_host in {"hp-13", "hp-14"}
            and physical.isdigit()
            and (private_namespace or authorized_no_mask)
            and (observed.startswith("aibox-") or observed == requested_host)
        ),
    }


def main() -> int:
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
                "--/rtx/rendermode=RaytracedLighting",
                "--/rtx/post/tonemap/op=4",
                "--/rtx/post/tonemap/filmIso=200",
            ),
        )
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import carb  # noqa: E402

    from taskdev.tasks.bench320_hs014_bottle_stabilized_cap_opening.core_scene import (  # noqa: E402
        HS014CoreScene,
        TASK_ID,
        sha256_file,
    )

    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "RaytracedLighting")
    settings.set("/rtx/post/tonemap/op", 4)
    settings.set("/rtx/post/tonemap/filmIso", 200.0)
    environment = None
    result: dict[str, Any] = {
        "schema": "hs014_core_scene_report_v1",
        "task_id": TASK_ID,
        "scope": "core_scene_only",
        "task_complete": False,
        "ik_executed": False,
        "grasp_executed": False,
        "success_motion_executed": False,
        "core_scene_pass": False,
    }
    rc = 1
    try:
        namespace = _namespace_contract()
        if not namespace["pass"]:
            raise RuntimeError(f"GPU namespace contract failed: {namespace}")
        environment = HS014CoreScene(
            simulation_app=simulation_app,
            app_launcher=app_launcher,
            output_dir=output_dir,
            device=args.device,
        )
        result = environment.run()
        result["gpu_namespace"] = namespace
        result["checks"]["taskdev_hp_gpu_execution_contract"] = namespace["pass"]
        result["core_scene_pass"] = all(result["checks"].values())
        rc = 0 if result["core_scene_pass"] else 1
    except Exception as exc:
        result["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        rc = 1
    finally:
        report_path = output_dir / "core_scene_report.json"
        _atomic_json(report_path, result)
        seal = {
            "task_id": result.get("task_id", ""),
            "core_scene_pass": bool(result.get("core_scene_pass") is True),
            "task_complete": False,
            "report": report_path.name,
            "report_sha256": sha256_file(report_path),
            "runner_rc": rc,
        }
        _atomic_json(
            output_dir
            / ("RUNNER_CORE_SCENE_PASS.json" if rc == 0 else "RUNNER_CORE_SCENE_FAIL.json"),
            seal,
        )
        if environment is not None:
            try:
                # RTX camera capture can leave SimulationContext.stop() blocked
                # indefinitely on the HP collection image.  The app-level
                # immediate-close path below owns process cleanup, so release
                # the environment without re-entering that stop path.
                environment.env._fast_shutdown_on_close = True
                environment.close()
            except Exception:
                traceback.print_exc()
                rc = 1
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
