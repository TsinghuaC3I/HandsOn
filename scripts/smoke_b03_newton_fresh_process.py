#!/usr/bin/env python3
"""Measure B03 Newton fresh-process repeatability on one local GPU.

This is a diagnostic smoke test, not cross-GPU certification.  It launches two
independent Isaac/Newton processes from the same seed, applies twelve idle
control frames, and reports semantic state errors using the same components as
the formal replay gate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "handson"
for path in (_REPOSITORY_ROOT, _PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np


_FIELDS = (
    "newton_robot_joint_position",
    "newton_robot_joint_velocity",
    "newton_robot_joint_target",
    "newton_task_rigid_body_state",
    "newton_tcp_state",
    "newton_substep_robot_joint_position",
    "newton_substep_robot_joint_velocity",
    "newton_substep_robot_joint_target",
    "newton_substep_task_rigid_body_state",
    "newton_substep_tcp_state",
)
_SCALAR_TOLERANCES = {
    "newton_robot_joint_position": 2.0e-4,
    "newton_robot_joint_velocity": 2.0e-3,
    "newton_robot_joint_target": 1.0e-7,
    "newton_substep_robot_joint_position": 2.0e-4,
    "newton_substep_robot_joint_velocity": 2.0e-3,
    "newton_substep_robot_joint_target": 1.0e-7,
}
_RIGID_FIELDS = (
    "newton_task_rigid_body_state",
    "newton_tcp_state",
    "newton_substep_task_rigid_body_state",
    "newton_substep_tcp_state",
)
_RIGID_TOLERANCES = {
    "position_m": 2.0e-4,
    "quaternion_angle_rad": float(np.deg2rad(0.2)),
    "linear_velocity_m_s": 2.0e-3,
    "angular_velocity_rad_s": 5.0e-3,
}


def _worker(output: Path) -> int:
    from handson.sim.isaaclab.newton_replay import destroy_newton_scene
    from handson.tasks.t09_insert_peg.config import TaskConfig

    environment = None
    payload: dict[str, Any] = {}
    try:
        environment = TaskConfig().get_native_environment(
            sim_backend="isaacsim_newton",
            policy_mode=True,
            render_mode="rgb_array",
            randomize=True,
            randomize_dynamics=False,
            image_obs=False,
            seed=20260807,
            newton_replay_profile="cross_gpu_v1",
        )
        observation, _info = environment.reset(seed=20260810)
        raw = environment.unwrapped
        rows = {
            name: [np.asarray(observation[name]).copy()] for name in _FIELDS
        }
        action = np.zeros(environment.action_space.shape, dtype=np.float32)
        for _ in range(12):
            observation, _, terminated, truncated, _info = environment.step(
                action
            )
            for name in _FIELDS:
                rows[name].append(np.asarray(observation[name]).copy())
            if terminated or truncated:
                break
        np.savez_compressed(
            output,
            **{name: np.stack(values) for name, values in rows.items()},
        )
        schema = raw.get_recording_schema()["newton"]
        determinism = schema["determinism"]
        payload = {
            "steps": int(raw.env_step),
            "state_writes": int(raw._trajectory_state_write_count),
            "source_sha256": schema["source_runtime_identity"][
                "b03_physics_source_sha256"
            ],
            "warp_deterministic_mode": determinism.get(
                "warp_deterministic_mode"
            ),
            "solver_deterministic_mode": determinism.get(
                "solver_deterministic_mode"
            ),
            "collision_pipeline_deterministic": determinism.get(
                "collision_pipeline_deterministic"
            ),
            "collision_broad_phase": determinism.get(
                "collision_broad_phase"
            ),
            "solver_deterministic_max_records": determinism.get(
                "solver_deterministic_max_records"
            ),
            "solver_constraint_capacity": determinism.get(
                "solver_constraint_capacity"
            ),
            "solver_disabled_read_only_sensors": determinism.get(
                "solver_disabled_read_only_sensors"
            ),
        }
    finally:
        if environment is not None:
            payload["lifecycle"] = destroy_newton_scene(environment)
    print("B03_NEWTON_WORKER=" + json.dumps(payload, sort_keys=True), flush=True)
    return 0


def _maximum_norm(value: np.ndarray) -> float:
    rows = np.asarray(value, dtype=np.float64).reshape(-1, value.shape[-1])
    return float(np.max(np.linalg.norm(rows, axis=1))) if rows.size else 0.0


def _rigid_errors(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first_rows = np.asarray(first, dtype=np.float64).reshape(-1, 13)
    second_rows = np.asarray(second, dtype=np.float64).reshape(-1, 13)
    first_quaternion = first_rows[:, 3:7]
    second_quaternion = second_rows[:, 3:7]
    first_norm = np.linalg.norm(first_quaternion, axis=1)
    second_norm = np.linalg.norm(second_quaternion, axis=1)
    first_valid = first_norm > 1.0e-12
    second_valid = second_norm > 1.0e-12
    if np.any(first_valid != second_valid):
        quaternion_angle = float("inf")
    elif not np.any(first_valid):
        quaternion_angle = 0.0
    else:
        quaternion_dot = np.sum(
            first_quaternion[first_valid] / first_norm[first_valid, None]
            * second_quaternion[second_valid] / second_norm[second_valid, None],
            axis=1,
        )
        quaternion_angle = float(
            np.max(
                2.0
                * np.arccos(np.clip(np.abs(quaternion_dot), 0.0, 1.0))
            )
        )
    return {
        "position_m": _maximum_norm(first_rows[:, :3] - second_rows[:, :3]),
        "quaternion_angle_rad": quaternion_angle,
        "linear_velocity_m_s": _maximum_norm(
            first_rows[:, 7:10] - second_rows[:, 7:10]
        ),
        "angular_velocity_rad_s": _maximum_norm(
            first_rows[:, 10:13] - second_rows[:, 10:13]
        ),
    }


def _last_marker(text: str, prefix: str) -> dict[str, Any]:
    rows = [line for line in text.splitlines() if line.startswith(prefix)]
    if not rows:
        raise RuntimeError(f"fresh worker emitted no {prefix} marker")
    return json.loads(rows[-1][len(prefix) :])


def _coordinator() -> int:
    child_environment = os.environ.copy()
    inherited_pythonpath = [
        entry
        for entry in child_environment.get("PYTHONPATH", "").split(
            os.pathsep
        )
        if entry
    ]
    child_environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                dict.fromkeys(
                    (
                        str(_REPOSITORY_ROOT),
                        str(_PACKAGE_ROOT),
                        *inherited_pythonpath,
                    )
                )
            ),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "HANDSON_RECORD_NEWTON_GROUND_TRUTH": "1",
            "HANDSON_NEWTON_REPLAY_PROFILE": "cross_gpu_v1",
            "HANDSON_ISAACLAB_DIRECT_HAND_CONTROL": "0",
            "HANDSON_ISAACLAB_DEFER_CAMERA_RENDER": "0",
        }
    )
    with tempfile.TemporaryDirectory(prefix="b03-newton-smoke-") as raw_dir:
        directory = Path(raw_dir)
        outputs = [directory / "run_1.npz", directory / "run_2.npz"]
        metadata = []
        for output in outputs:
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(output),
                ],
                cwd=str(_REPOSITORY_ROOT),
                env=child_environment,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                sys.stderr.write(result.stdout)
                sys.stderr.write(result.stderr)
                raise RuntimeError(
                    f"fresh B03 worker failed with exit code {result.returncode}"
                )
            metadata.append(_last_marker(result.stdout, "B03_NEWTON_WORKER="))

        with np.load(outputs[0]) as first, np.load(outputs[1]) as second:
            metrics: dict[str, Any] = {
                name: float(np.max(np.abs(first[name] - second[name])))
                for name in _SCALAR_TOLERANCES
            }
            metrics.update(
                {
                    name: _rigid_errors(first[name], second[name])
                    for name in _RIGID_FIELDS
                }
            )
    scalar_pass = all(
        float(metrics[name]) <= tolerance
        for name, tolerance in _SCALAR_TOLERANCES.items()
    )
    rigid_pass = all(
        float(metrics[field][name]) <= tolerance
        for field in _RIGID_FIELDS
        for name, tolerance in _RIGID_TOLERANCES.items()
    )
    lifecycle_pass = all(
        worker.get("steps") == 12
        and worker.get("state_writes") == 0
        and worker.get("warp_deterministic_mode") == "gpu_to_gpu"
        and worker.get("solver_deterministic_mode") == "gpu_to_gpu"
        and worker.get("collision_pipeline_deterministic") is True
        and worker.get("collision_broad_phase") == "explicit"
        and type(worker.get("solver_deterministic_max_records")) is int
        and worker["solver_deterministic_max_records"] > 0
        and type(worker.get("solver_constraint_capacity")) is int
        and worker["solver_constraint_capacity"] > 0
        and type(worker.get("solver_disabled_read_only_sensors")) is int
        and worker["solver_disabled_read_only_sensors"] >= 0
        and isinstance(worker.get("source_sha256"), str)
        and len(worker["source_sha256"]) == 64
        and isinstance(worker.get("lifecycle"), dict)
        and worker["lifecycle"].get("scene_destroyed") is True
        for worker in metadata
    )
    source_hashes = {worker.get("source_sha256") for worker in metadata}
    source_identity_pass = len(source_hashes) == 1 and None not in source_hashes
    passed = bool(
        scalar_pass
        and rigid_pass
        and lifecycle_pass
        and source_identity_pass
    )
    report = {
        "format": "handson.b03_newton_same_gpu_smoke",
        "version": 1,
        "certification_eligible": False,
        "passed": passed,
        "fresh_processes": 2,
        "workers": metadata,
        "tolerances": {
            **_SCALAR_TOLERANCES,
            "rigid_state_v1": _RIGID_TOLERANCES,
        },
        "max_error_by_metric": metrics,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    arguments = parser.parse_args()
    if arguments.worker is not None:
        return _worker(arguments.worker.expanduser().resolve())
    return _coordinator()


if __name__ == "__main__":
    raise SystemExit(main())
