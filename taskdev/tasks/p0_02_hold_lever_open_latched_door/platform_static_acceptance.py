#!/usr/bin/env python3
"""Fail-closed static collection-platform acceptance for P0-02."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


TASK_NAME = "t17_open_latched_door"
TASK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_DIR.parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "handson"))
sys.path.insert(0, str(REPOSITORY_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_assets() -> tuple[bool, list[dict[str, Any]]]:
    manifest = json.loads((TASK_DIR / "asset_manifest.json").read_text())
    rows: list[dict[str, Any]] = []
    for item in manifest["files"]:
        repository_path = item.get("repository_path")
        path = (
            REPOSITORY_ROOT / str(repository_path)
            if repository_path
            else TASK_DIR / item["path"]
        )
        exists = path.is_file()
        size = path.stat().st_size if exists else None
        digest = _sha256(path) if exists else None
        row = {
            "path": str(path.relative_to(REPOSITORY_ROOT)),
            "exists": exists,
            "bytes": size,
            "bytes_match": size == item["bytes"],
            "sha256": digest,
            "sha256_match": digest == item["sha256"],
            "role": str(item.get("role") or ""),
        }
        row["passed"] = bool(
            row["exists"] and row["bytes_match"] and row["sha256_match"]
        )
        rows.append(row)
    return all(row["passed"] for row in rows), rows


def _has_scripted_policy_task_guard(source: str, task_name: str) -> bool:
    """Find the recorder's supported-task allowlist without pinning formatting."""

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.left, ast.Name) or node.left.id != "task_id":
            continue
        if not isinstance(node.ops[0], ast.NotIn) or len(node.comparators) != 1:
            continue
        allowlist = node.comparators[0]
        if not isinstance(allowlist, (ast.Set, ast.Tuple, ast.List)):
            continue
        values = {
            element.value
            for element in allowlist.elts
            if isinstance(element, ast.Constant)
            and isinstance(element.value, str)
        }
        if task_name in values:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from handson.runtime.collection_support import ISAACSIM_COLLECTION_TASKS
    from handson.tasks import CONFIG_MAPPING
    import collection_ui.server as collection_server

    task_config = CONFIG_MAPPING[TASK_NAME]()
    recorder_source = (
        REPOSITORY_ROOT / "scripts" / "record_demos_zarr.py"
    ).read_text(encoding="utf-8")
    task_yaml_source = (TASK_DIR / "task.yaml").read_text(encoding="utf-8")
    assets_passed, asset_rows = _check_assets()
    exact_asset_roles = {
        str(row["role"])
        for row in asset_rows
        if bool(row["passed"])
    }

    checks = {
        "config_mapping_discovery": TASK_NAME in CONFIG_MAPPING,
        "config_instantiation": (
            task_config.task_name == TASK_NAME and task_config.task_id == "P0-02"
        ),
        "isaac_collection_capability": TASK_NAME in ISAACSIM_COLLECTION_TASKS,
        "ui_task_order_once": collection_server.TASK_ORDER.count(TASK_NAME) == 1,
        "ui_label": collection_server.TASK_LABELS.get(TASK_NAME)
        == "P0-02 Hold Lever and Open Latched Door",
        "recorder_scripted_policy_flag": '"scripted_policy"' in recorder_source,
        "recorder_task_guard": _has_scripted_policy_task_guard(
            recorder_source,
            TASK_NAME,
        ),
        "recorder_physx_fresh_process_gate": (
            "_enforce_fresh_process_certified_collection_entrypoint" in recorder_source
            and "scripts/fresh_process_certified_collection.py" in recorder_source
        ),
        "task_yaml_recorder_binding": (
            "recorder: scripts/record_demos_zarr.py" in task_yaml_source
            and "fresh_replay: scripts/certify_physx_replay.py" in task_yaml_source
            and "min_successful_episodes: 3" in task_yaml_source
        ),
        "all_manifest_assets_exact": assets_passed,
        "helindu_source_and_license_exact": {
            "helindu_immutable_source_archive",
            "helindu_license",
            "helindu_source_scene",
            "helindu_source_payload",
        }.issubset(exact_asset_roles),
        "helindu_runtime_usdas_exact": {
            "helindu_runtime_rosette",
            "helindu_runtime_lever",
        }.issubset(exact_asset_roles),
    }
    result = {
        "schema": "handson.p002.platform_static_acceptance.v1",
        "task": TASK_NAME,
        "checks": checks,
        "asset_files": asset_rows,
        "passed": all(checks.values()),
        "scope": "static discovery and launch-contract validation only",
        "not_claimed": [
            "rendered camera frames",
            "saved successful episodes",
            "fresh-process episode replay",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
