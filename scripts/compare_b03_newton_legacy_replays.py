#!/usr/bin/env python3
"""Directly compare non-certifying B03 legacy replay traces."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


REPORT_NAME = "diagnostic_report.json"
TRACE_NAME = "diagnostic_trace.npz"
SCHEMA_NAME = "runtime_schema.json"
FORMAT = "handson.b03_newton_legacy_direct_trace_comparison"
VERSION = 1
_GPU_ENVIRONMENT_KEYS = {
    "platform",
    "device",
    "gpu_name",
    "gpu_arch",
    "gpu_uuid",
    "gpu_pci_bus_id",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _software_identity(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise RuntimeError("diagnostic report has no runtime source identity")
    return {
        key: value
        for key, value in identity.items()
        if key not in _GPU_ENVIRONMENT_KEYS
    }


def _maximum_vector_norm(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    if not array.size:
        return 0.0
    return float(
        np.max(np.linalg.norm(array.reshape(-1, array.shape[-1]), axis=1))
    )


def _rigid_errors(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    if actual.shape[-1] != 13 or expected.shape[-1] != 13:
        raise RuntimeError("rigid_state_v1 requires a final dimension of 13")
    actual_rows = np.asarray(actual, dtype=np.float64).reshape(-1, 13)
    expected_rows = np.asarray(expected, dtype=np.float64).reshape(-1, 13)
    actual_quat = actual_rows[:, 3:7]
    expected_quat = expected_rows[:, 3:7]
    actual_norm = np.linalg.norm(actual_quat, axis=1)
    expected_norm = np.linalg.norm(expected_quat, axis=1)
    actual_valid = actual_norm > 1.0e-12
    expected_valid = expected_norm > 1.0e-12
    if np.any(actual_valid != expected_valid):
        quaternion_angle = float("inf")
    elif not np.any(actual_valid):
        quaternion_angle = 0.0
    else:
        dot = np.sum(
            actual_quat[actual_valid] / actual_norm[actual_valid, None]
            * expected_quat[expected_valid] / expected_norm[expected_valid, None],
            axis=1,
        )
        quaternion_angle = float(
            np.max(2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0)))
        )
    return {
        "position_m": _maximum_vector_norm(
            actual_rows[:, :3] - expected_rows[:, :3]
        ),
        "quaternion_angle_rad": quaternion_angle,
        "linear_velocity_m_s": _maximum_vector_norm(
            actual_rows[:, 7:10] - expected_rows[:, 7:10]
        ),
        "angular_velocity_rad_s": _maximum_vector_norm(
            actual_rows[:, 10:13] - expected_rows[:, 10:13]
        ),
    }


def _compare_pair(
    left_dir: Path,
    right_dir: Path,
    specs: dict[str, Any],
) -> dict[str, Any]:
    maxima: dict[str, float] = {}
    tolerances: dict[str, float] = {}
    failures: list[str] = []
    left_path = left_dir / TRACE_NAME
    right_path = right_dir / TRACE_NAME
    with np.load(left_path, allow_pickle=False) as left, np.load(
        right_path,
        allow_pickle=False,
    ) as right:
        left_fields = set(left.files)
        right_fields = set(right.files)
        if left_fields != right_fields:
            failures.append(
                "trace fields differ: "
                f"left_only={sorted(left_fields - right_fields)}, "
                f"right_only={sorted(right_fields - left_fields)}"
            )
        for name in sorted(left_fields & right_fields):
            actual = np.asarray(right[name])
            expected = np.asarray(left[name])
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                failures.append(
                    f"{name}: layout {actual.dtype}{actual.shape} != "
                    f"{expected.dtype}{expected.shape}"
                )
                continue
            spec = specs.get(name)
            if not isinstance(spec, dict) or spec.get("physical_replay") is not True:
                failures.append(f"{name}: missing current physical-replay spec")
                continue
            if spec.get("replay_metric") == "rigid_state_v1":
                raw_tolerances = spec.get("replay_tolerance")
                if not isinstance(raw_tolerances, dict):
                    failures.append(f"{name}: rigid tolerance is missing")
                    continue
                if np.array_equal(actual, expected):
                    errors = {
                        str(metric): 0.0 for metric in raw_tolerances
                    }
                elif not np.all(np.isfinite(actual)) or not np.all(
                    np.isfinite(expected)
                ):
                    errors = {
                        str(metric): float("inf") for metric in raw_tolerances
                    }
                else:
                    errors = _rigid_errors(actual, expected)
                for metric, error in errors.items():
                    key = f"{name}.{metric}"
                    tolerance = float(raw_tolerances[metric])
                    maxima[key] = error
                    tolerances[key] = tolerance
                    if error > tolerance:
                        failures.append(
                            f"{key}: {error:.9g} > {tolerance:.9g}"
                        )
                continue
            tolerance = float(spec.get("replay_atol", 0.0))
            if actual.dtype.kind in "biu":
                error = 0.0 if np.array_equal(actual, expected) else float("inf")
            elif not np.all(np.isfinite(actual)) or not np.all(
                np.isfinite(expected)
            ):
                error = float("inf")
            elif actual.size:
                error = float(np.max(np.abs(actual - expected)))
            else:
                error = 0.0
            maxima[name] = error
            tolerances[name] = tolerance
            if error > tolerance:
                failures.append(f"{name}: {error:.9g} > {tolerance:.9g}")
    finite_ratios = [
        error / tolerance
        for key, error in maxima.items()
        if math.isfinite(error) and (tolerance := tolerances[key]) > 0.0
    ]
    return {
        "left": str(left_dir),
        "right": str(right_dir),
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "max_error_by_metric": maxima,
        "tolerance_by_metric": tolerances,
        "worst_finite_error_to_tolerance_ratio": (
            max(finite_ratios) if finite_ratios else 0.0
        ),
    }


def compare(run_dirs: list[Path]) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise RuntimeError("at least two diagnostic runs are required")
    reports = [_json(path / REPORT_NAME) for path in run_dirs]
    schemas = [_json(path / SCHEMA_NAME) for path in run_dirs]
    errors: list[str] = []
    reference_report = reports[0]
    reference_schema = schemas[0]
    reference_software = _software_identity(
        reference_report.get("runtime_source_identity")
    )
    reference_physics = reference_report.get("runtime_physics_config")
    reference_source_stats = reference_report.get("source_stats")
    for path, report, schema in zip(run_dirs, reports, schemas, strict=True):
        if report.get("passed_execution") is not True:
            errors.append(f"{path}: execution did not pass")
        if report.get("trace_complete") is not True:
            errors.append(f"{path}: trace is incomplete")
        if report.get("state_writes_after_action_start") != 0:
            errors.append(f"{path}: state-write audit is non-zero")
        if report.get("nonfinite_trace_samples") != 0:
            errors.append(f"{path}: trace contains non-finite samples")
        if report.get("source_stats") != reference_source_stats:
            errors.append(f"{path}: legacy drive input identity differs")
        if report.get("runtime_physics_config") != reference_physics:
            errors.append(f"{path}: current physics configuration differs")
        if _software_identity(report.get("runtime_source_identity")) != reference_software:
            errors.append(f"{path}: software/asset identity differs")
        if schema.get("fields") != reference_schema.get("fields"):
            errors.append(f"{path}: current field schema differs")

    specs = reference_schema.get("fields")
    if not isinstance(specs, dict):
        raise RuntimeError("reference runtime schema has no fields")
    pairs = [
        _compare_pair(left, right, specs)
        for left, right in combinations(run_dirs, 2)
    ]
    failed_pairs = [pair for pair in pairs if not pair["passed"]]
    return {
        "format": FORMAT,
        "version": VERSION,
        "non_certifying": True,
        "run_count": len(run_dirs),
        "pair_count": len(pairs),
        "run_contract_errors": errors,
        "failed_pair_count": len(failed_pairs),
        "passed": not errors and not failed_pairs,
        "runs": [
            {
                "path": str(path),
                "run_id": report.get("run_id"),
                "gpu_name": report.get("runtime_source_identity", {}).get(
                    "gpu_name"
                ),
                "gpu_arch": report.get("runtime_source_identity", {}).get(
                    "gpu_arch"
                ),
                "trace_array_sha256": report.get("trace_array_sha256"),
                "maximum_penetration_m": report.get(
                    "runtime_hand_peg_max_solid_penetration_m"
                ),
                "maximum_hand_velocity_rad_s": report.get(
                    "runtime_hand_max_abs_joint_velocity_rad_s"
                ),
                "final_success": report.get("final_success"),
            }
            for path, report in zip(run_dirs, reports, strict=True)
        ],
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()
    run_dirs = [path.expanduser().resolve() for path in args.run_dirs]
    for path in run_dirs:
        for name in (REPORT_NAME, TRACE_NAME, SCHEMA_NAME):
            if not (path / name).is_file():
                parser.error(f"diagnostic run is missing {name}: {path}")
    result = compare(run_dirs)
    report_path = args.report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, report_path)
    print(
        "B03_LEGACY_COMPARISON="
        + json.dumps(
            {
                "passed": result["passed"],
                "run_count": result["run_count"],
                "pair_count": result["pair_count"],
                "run_contract_error_count": len(result["run_contract_errors"]),
                "failed_pair_count": result["failed_pair_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
