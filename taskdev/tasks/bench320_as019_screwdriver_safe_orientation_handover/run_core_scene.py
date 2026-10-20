#!/usr/bin/env python3
"""Run the bounded B320-AS019 Isaac 5.1 core-scene audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() not in {"0", ""}:
        raise RuntimeError("AS019 is assigned exclusively to physical GPU0")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")

    # AppLauncher is deliberately the first Isaac/Isaac Lab import.
    from isaaclab.app import AppLauncher

    launcher_args = argparse.Namespace(
        headless=True,
        enable_cameras=True,
        width=640,
        height=480,
        device="cuda:0",
        kit_args=(
            "--/rtx/rendermode=RayTracedLighting "
            "--/rtx/post/tonemap/op=4 "
            "--/rtx/post/tonemap/filmIso=200"
        ),
    )
    launcher = AppLauncher(launcher_args)
    simulation_app = launcher.app
    rc = 1
    try:
        import carb
        import importlib

        settings = carb.settings.get_settings()
        settings.set("/rtx/rendermode", "RayTracedLighting")
        settings.set("/rtx/post/tonemap/op", 4)
        settings.set("/rtx/post/tonemap/filmIso", 200)
        enable_extension = importlib.import_module(
            "isaacsim.core.utils.extensions"
        ).enable_extension
        enable_extension("isaacsim.asset.importer.mjcf")
        for _ in range(3):
            simulation_app.update()

        from taskdev.tasks.bench320_as019_screwdriver_safe_orientation_handover.scene_runtime import (
            run_core_scene,
        )

        summary = run_core_scene(
            simulation_app=simulation_app,
            output_dir=args.output_dir,
        )
        print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
        rc = 0 if summary.get("core_scene_pass") is True else 2
    except BaseException:
        traceback.print_exc()
        rc = 1
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
