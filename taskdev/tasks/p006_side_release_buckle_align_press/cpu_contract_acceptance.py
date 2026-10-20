#!/usr/bin/env python3
"""Dependency-light CPU gate for the P0-06 frozen semantic contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


TASK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_DIR.parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "handson"))

from handson.sim.isaaclab.p006_contract import (  # noqa: E402
    FEMALE_ASSET_SHA256,
    MALE_ASSET_SHA256,
    MAX_PENETRATION_M,
    MAX_TRANSVERSE_OFFSET_M,
    MIN_ROLE_CONTACT_FORCE_N,
    P006LatchStateMachine,
    P006PhysicalReadback,
    SOURCE_STL_SHA256,
    reset_is_valid,
    sample_reset_state,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quat_z(degrees: float) -> tuple[float, float, float, float]:
    half = math.radians(degrees) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _readback(**overrides) -> P006PhysicalReadback:
    values = {
        "female_position_m": (0.0, 0.0, 1.0),
        "female_quaternion_wxyz": (1.0, 0.0, 0.0, 0.0),
        "male_position_m": (0.065, 0.0, 1.0),
        "male_quaternion_wxyz": (1.0, 0.0, 0.0, 0.0),
        "female_linear_velocity_mps": (0.0, 0.0, 0.0),
        "female_angular_velocity_radps": (0.0, 0.0, 0.0),
        "male_linear_velocity_mps": (0.0, 0.0, 0.0),
        "male_angular_velocity_radps": (0.0, 0.0, 0.0),
        "right_female_contact_force_n": MIN_ROLE_CONTACT_FORCE_N + 0.1,
        "left_male_contact_force_n": MIN_ROLE_CONTACT_FORCE_N + 0.1,
        "buckle_body_contact_force_n": 0.2,
        "robot_table_contact_force_n": 0.0,
        "max_penetration_m": 0.0,
    }
    values.update(overrides)
    return P006PhysicalReadback(**values)


def _success(readback: P006PhysicalReadback, steps: int = 30) -> bool:
    machine = P006LatchStateMachine()
    return any(bool(machine.update(readback)["success_now"]) for _ in range(steps))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asset_dir = TASK_DIR / "assets" / "side_release_buckle_cc0"
    assets = {
        "model.stl": (_sha(asset_dir / "model.stl"), SOURCE_STL_SHA256),
        "female_buckle.usda": (
            _sha(asset_dir / "female_buckle.usda"), FEMALE_ASSET_SHA256
        ),
        "male_buckle.usda": (
            _sha(asset_dir / "male_buckle.usda"), MALE_ASSET_SHA256
        ),
    }
    resets = [sample_reset_state(seed) for seed in range(1000)]
    deterministic_resets = resets == [sample_reset_state(seed) for seed in range(1000)]
    positive_trials = [_success(_readback()) for _ in range(10)]
    negative_cases = {
        "missing_right_female": {"right_female_contact_force_n": 0.0},
        "missing_left_male": {"left_male_contact_force_n": 0.0},
        "missing_body_contact": {"buckle_body_contact_force_n": 0.0},
        "shallow_insertion": {"male_position_m": (0.080, 0.0, 1.0)},
        "transverse_miss": {
            "male_position_m": (0.065, MAX_TRANSVERSE_OFFSET_M + 0.001, 1.0)
        },
        "axis_miss": {"male_quaternion_wxyz": _quat_z(13.0)},
        "excess_penetration": {"max_penetration_m": MAX_PENETRATION_M + 0.001},
        "invalid_contact": {"robot_table_contact_force_n": math.nan},
        "unstable_translation": {"male_linear_velocity_mps": (0.13, 0.0, 0.0)},
        "unstable_rotation": {"male_angular_velocity_radps": (0.0, 0.0, 1.1)},
    }
    negative_results = {
        name: _success(_readback(**overrides))
        for name, overrides in negative_cases.items()
    }
    checks = {
        "exact_asset_hashes": all(actual == expected for actual, expected in assets.values()),
        "one_thousand_valid_resets": all(reset_is_valid(reset) for reset in resets),
        "deterministic_resets": deterministic_resets,
        "ten_positive_trials": all(positive_trials),
        "normal_table_contact_allowed": _success(_readback(robot_table_contact_force_n=5.0)),
        "ten_negative_trials": not any(negative_results.values()),
        "no_pull_state": not hasattr(P006LatchStateMachine(), "pull_dwell_steps"),
    }
    result = {
        "schema": "handson.p006.cpu_contract_acceptance.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "assets": {
            name: {"actual": actual, "expected": expected}
            for name, (actual, expected) in assets.items()
        },
        "reset_trials": len(resets),
        "positive_trials": positive_trials,
        "negative_trials": negative_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
