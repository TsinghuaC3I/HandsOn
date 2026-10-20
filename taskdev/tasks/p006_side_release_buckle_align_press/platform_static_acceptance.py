#!/usr/bin/env python3
"""CPU-only registration and collection-platform wiring gate for P0-06."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TASK_NAME = "p006_side_release_buckle_align_press"
TASK_LABEL = "P0-06 Side-Release Buckle Align and Press"
ROOT = Path(__file__).resolve().parents[3]


def _contains(relative: str, *needles: str) -> dict[str, object]:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in text]
    return {"path": relative, "passed": not missing, "missing": missing}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = [
        _contains("collection_ui/server.py", TASK_NAME, TASK_LABEL, "no pull test is required"),
        _contains("handson/handson/runtime/collection_support.py", TASK_NAME),
        _contains("handson/handson/tasks/mappings.py", TASK_NAME),
        _contains("handson/handson/sim/isaaclab/task_preview.py", TASK_NAME, "no pull test"),
        _contains("handson/handson/sim/isaaclab/abc_collection_env.py", TASK_NAME, "P006FemaleBuckle", "P006MaleBuckle"),
        _contains("handson/handson/sim/isaaclab/p006_collection_env.py", "p006_pull_test_required", "False"),
        _contains("configs/multi_task/p006_side_release_buckle_align_press.yaml", "physics_backend: physx", "no pull test is required"),
        _contains(
            "taskdev/tasks/p006_side_release_buckle_align_press/assets/side_release_buckle_cc0/male_buckle.usda",
            "PhysicsArticulationRootAPI",
            "OuterTongueNeg",
            "OuterTonguePos",
            "PhysicsPrismaticJoint",
        ),
    ]
    asset_dir = ROOT / "taskdev/tasks" / TASK_NAME / "assets/side_release_buckle_cc0"
    hashes = {
        "model.stl": "f68abfbfac56f746fef1f411ec68d511d4cb567cfb0f10472cb8e26168d57b23",
        "female_buckle.usda": "16207f70c61770d23c7d1b2e3593904ec160c962e3ed924dbc1d39bf8f41a809",
        "male_buckle.usda": "12079d21a94beb3c77e0aade3f5552f4098fc51a546d4cd588a59f41ece8b667",
    }
    asset_results = {
        name: {"expected": expected, "actual": _sha(asset_dir / name)}
        for name, expected in hashes.items()
    }
    passed = all(check["passed"] for check in checks) and all(
        item["expected"] == item["actual"] for item in asset_results.values()
    )
    result = {
        "schema": "handson.p006.platform_static_acceptance.v1",
        "passed": passed,
        "task": TASK_NAME,
        "checks": checks,
        "assets": asset_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
