#!/usr/bin/env python3
"""CPU-only audit of one HandsOn task's registry, UI, and YAML contract."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml


def audit_task(repo: Path, task_name: str) -> dict[str, Any]:
    sys.path.insert(0, str(repo / "handson"))
    from handson.runtime.collection_support import ISAACSIM_COLLECTION_TASKS
    from handson.sim.isaaclab.abc_collection_env import (
        supported_native_collection_tasks,
    )
    from handson.tasks.mappings import CONFIG_MAPPING

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    checks["config_mapping"] = task_name in CONFIG_MAPPING
    if checks["config_mapping"]:
        config = CONFIG_MAPPING[task_name]()
        details["task_id"] = str(config.task_id)
        details["prompt"] = str(config.task_prompt)
        checks["canonical_config_name"] = config.task_name == task_name
    else:
        checks["canonical_config_name"] = False

    checks["collection_whitelist"] = task_name in ISAACSIM_COLLECTION_TASKS
    checks["native_runtime"] = task_name in supported_native_collection_tasks()
    # LW native-articulation tasks intentionally do not appear in the generic
    # MuJoCo/XML ``task_preview.TASK_SPECS`` mapping: their preview/recorder
    # path composes the user-provided USD through ``record_demos_zarr.py``.
    # Check that native asset directly before falling back to the generic
    # registry, otherwise this CPU audit reports every G05/G08/G10 preview as
    # missing even though the UI can launch its real scene.
    try:
        from handson.sim.isaaclab.lw_native_articulation_scene import (
            LW_TASK_SPECS,
        )

        native_spec = LW_TASK_SPECS.get(task_name)
    except (ImportError, OSError, AttributeError):
        native_spec = None
    if native_spec is not None:
        checks["preview_scene"] = native_spec.source_path.is_file()
    else:
        # B320 and several newer task families register their preview scene
        # through a shared mapping imported into ``task_preview.py``.  Validate
        # the public registry and the resolved XML path instead of relying on a
        # brittle source string (which incorrectly reported every dynamically
        # registered task as missing its preview scene).
        try:
            from handson.sim.isaaclab.task_preview import TASK_SPECS

            preview_spec = TASK_SPECS.get(task_name)
            # ``TASK_SPECS.source_path`` may point at an Isaac converter cache
            # selected by HANDSON_ISAACLAB_XML_DIR.  A clean checkout does not
            # have that cache yet, while the packaged source XML is already part
            # of the repository.  Accept either location, but never accept a
            # missing/unknown scene name.
            preview_paths = []
            if preview_spec is not None:
                preview_paths.append(preview_spec.source_path)
                try:
                    from handson.sim.isaaclab import task_preview as _task_preview

                    packaged_dir = getattr(_task_preview, "_PACKAGED_XML_DIR", None)
                    if packaged_dir is not None:
                        preview_paths.append(
                            Path(packaged_dir) / preview_spec.scene_xml
                        )
                except (ImportError, OSError, AttributeError):
                    pass
            checks["preview_scene"] = bool(
                preview_spec is not None
                and any(path.is_file() for path in preview_paths)
            )
        except (ImportError, OSError, AttributeError):
            # A CPU-only audit may run outside the Isaac/Kit interpreter, where
            # importing ``task_preview`` is intentionally unavailable.  Verify
            # the generated B320 registry declaration and packaged XML directly
            # in that case instead of turning a dependency limitation into a
            # false registration failure.
            preview_source_path = (
                repo
                / "handson"
                / "handson"
                / "sim"
                / "isaaclab"
                / "task_preview.py"
            )
            preview_source = preview_source_path.read_text(encoding="utf-8")
            # Parse the public task-to-scene declaration without importing
            # IsaacLab.  This also covers task families (such as AS013) whose
            # scene is supplied by a shared XML rather than B320_TASK_SCENES.
            scene_match = re.search(
                rf"[\"']{re.escape(task_name)}[\"']\s*:\s*[\"']([^\"']+\.xml)[\"']",
                preview_source,
            )
            scene_xml_name = scene_match.group(1) if scene_match is not None else None
            if scene_xml_name is None:
                try:
                    from handson.sim.isaaclab.g01_deformable_asset_scene import (
                        PILLOW_CARTON_SCENE_XML,
                        G01_CARTON_TASK_NAMES,
                        G01_SCENE_XML,
                        G01_TASK_NAMES,
                    )

                    if task_name in G01_TASK_NAMES:
                        scene_xml_name = (
                            PILLOW_CARTON_SCENE_XML
                            if task_name in G01_CARTON_TASK_NAMES
                            else G01_SCENE_XML
                        )
                except (ImportError, OSError, AttributeError):
                    pass
            if scene_xml_name is None:
                try:
                    from handson.sim.isaaclab.t06_fit_device_cover_scene import (
                        G02_SCENE_XML,
                        G02_TASK_NAME,
                    )

                    if task_name == G02_TASK_NAME:
                        scene_xml_name = G02_SCENE_XML
                except (ImportError, OSError, AttributeError):
                    pass
            if scene_xml_name is None:
                try:
                    from handson.sim.isaaclab.loaded_tray_gate_scene import (
                        SCENE_XML as LOADED_TRAY_SCENE_XML,
                        TASK_NAME as LOADED_TRAY_TASK_NAME,
                    )

                    if task_name == LOADED_TRAY_TASK_NAME:
                        scene_xml_name = LOADED_TRAY_SCENE_XML
                except (ImportError, OSError, AttributeError):
                    pass
            if scene_xml_name is None:
                try:
                    from handson.sim.isaaclab.b320_bimanual_scene import (
                        B320_TASK_SCENES,
                    )

                    scene_xml_name = B320_TASK_SCENES.get(task_name)
                except (ImportError, OSError, AttributeError):
                    pass
            if scene_xml_name is None:
                # D01-D05 and their XY1 composition use this Kit-free registry.
                from handson.sim.isaaclab.d_collection_scene import D_TASK_SCENES
                scene_xml_name = D_TASK_SCENES.get(task_name)
            if scene_xml_name is None:
                # D06-D20 long-horizon tasks register their preview scene
                # through ``D_LONG_TASK_SCENES`` (a Kit-free module), which
                # ``task_preview.py`` splats into ``TASK_SPECS``.  Without this
                # lookup the CPU-only audit reported every D-long preview as
                # missing even though the scene XML is packaged.
                try:
                    from handson.sim.isaaclab.d_long_horizon_scene import (
                        D_LONG_TASK_SCENES,
                    )

                    scene_xml_name = D_LONG_TASK_SCENES.get(task_name)
                except (ImportError, OSError, AttributeError):
                    pass
            if scene_xml_name is None:
                # YCB11 uses named constants for its MJCF mug scene and adds
                # the original deformable sponge in the native environment.
                try:
                    from handson.sim.isaaclab.ycb11_scene import (
                        YCB11_SCENE_XML,
                        YCB11_TASK_NAME,
                    )

                    if task_name == YCB11_TASK_NAME:
                        scene_xml_name = YCB11_SCENE_XML
                except (ImportError, OSError, AttributeError):
                    pass
            scene_xml = (
                repo
                / "handson"
                / "handson"
                / "sim"
                / "envs"
                / "xmls"
                / str(scene_xml_name)
            )
            checks["preview_scene"] = bool(
                scene_xml_name is not None and scene_xml.is_file()
            )
    yaml_path = repo / "configs" / "multi_task" / f"{task_name}.yaml"
    checks["task_yaml"] = yaml_path.is_file()
    if yaml_path.is_file():
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        checks["yaml_identity"] = payload.get("env_name") == task_name
        camera_mapping = payload.get("camera_mapping")
        checks["three_cameras"] = (
            isinstance(camera_mapping, dict)
            and set(camera_mapping) == {"base", "wrist_left", "wrist_right"}
            and camera_mapping.get("base") in {"ego", "front"}
            and camera_mapping.get("wrist_left") == "wrist_left"
            and camera_mapping.get("wrist_right") == "wrist_right"
        )
    else:
        checks["yaml_identity"] = False
        checks["three_cameras"] = False

    ui_source = (repo / "collection_ui" / "server.py").read_text(encoding="utf-8")
    checks["ui_identity"] = ui_source.count(f'"{task_name}"') >= 3
    checks["ui_isaac_only"] = (
        "ISAACSIM_ONLY_TASKS" in ui_source
        and f'"{task_name}"' in ui_source
    )
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": "handson_task_registration_audit_v1",
        "task_name": task_name,
        "checks": checks,
        "details": details,
        "failures": failures,
        "pass": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--task", required=True)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()
    result = audit_task(args.repo.resolve(), args.task)
    if args.format == "markdown":
        print(f"# Task registration audit: `{result['task_name']}`")
        print()
        for name, passed in result["checks"].items():
            print(f"- {'PASS' if passed else 'FAIL'} `{name}`")
        print()
        print(f"Overall: **{'PASS' if result['pass'] else 'FAIL'}**")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
