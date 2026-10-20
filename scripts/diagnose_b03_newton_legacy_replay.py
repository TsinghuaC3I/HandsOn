#!/usr/bin/env python3
"""Replay legacy B03 drive targets on the current Newton physics stack.

This command is deliberately non-certifying.  It never mutates the source
episode and never relaxes the production replay gate.  A legacy 120 Hz drive
trace is adapted to the current 240 Hz runtime with zero-order hold, then the
current solver readback is saved for an empirical cross-GPU comparison.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any
import uuid

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "handson"
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HANDSON_NEWTON_REPLAY_PROFILE"] = "cross_gpu_v1"
os.environ["HANDSON_RECORD_NEWTON_GROUND_TRUTH"] = "1"
os.environ["HANDSON_ISAACLAB_DIRECT_HAND_CONTROL"] = "0"
os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "0"
os.environ["HANDSON_NEWTON_USE_CUDA_GRAPH"] = "1"
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import numpy as np

try:
    import zarr
except ModuleNotFoundError as exc:
    if exc.name != "zarr":
        raise
    vendor_raw = os.environ.get("HANDSON_LEGACY_ZARR_VENDOR", "").strip()
    if not vendor_raw:
        raise RuntimeError(
            "The qualified Newton Python has no zarr reader. Set "
            "HANDSON_LEGACY_ZARR_VENDOR to a Python-3.12 site-packages "
            "directory containing zarr, numcodecs, asciitree, and fasteners."
        ) from exc
    vendor = str(Path(vendor_raw).expanduser().resolve())
    sys.path.insert(0, vendor)
    try:
        import zarr
    finally:
        sys.path.remove(vendor)

import scipy  # noqa: F401 -- load BLAS before Kit starts worker threads

from handson.data.b03_newton_validity import (
    B03_NEWTON_HAND_MAX_ABS_JOINT_VELOCITY_RAD_S,
    B03_NEWTON_HAND_PEG_MAX_SOLID_PENETRATION_M,
)
from handson.sim.isaaclab.newton_replay import (
    NewtonDriveControlStep,
    RecordedNewtonDrivePath,
    destroy_newton_scene,
    run_newton_drive_control_step,
)
from handson.tasks.t09_insert_peg.config import TaskConfig


FORMAT = "handson.b03_newton_legacy_current_stack_diagnostic"
VERSION = 1
CURRENT_PHYSICS_HZ = 240
CURRENT_CONTROL_HZ = 30
REPORT_NAME = "diagnostic_report.json"
TRACE_NAME = "diagnostic_trace.npz"
SCHEMA_NAME = "runtime_schema.json"

_TARGET_FIELDS = (
    "next_newton_substep_robot_joint_target",
    "next_newton_substep_robot_joint_target_valid",
    "next_newton_substep_count",
    "next_newton_robot_joint_target",
    "next_newton_robot_joint_target_valid",
)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _framed_array_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        for payload in (
            name.encode("utf-8"),
            value.dtype.str.encode("ascii"),
            json.dumps(value.shape).encode("ascii"),
            value.tobytes(order="C"),
        ):
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def _source_inputs(
    demo_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    manifest = _json_object(demo_dir / "replay_manifest.json")
    recording_schema = _json_object(demo_dir / "recording_schema.json")
    recording = manifest.get("recording")
    if not isinstance(recording, dict):
        raise RuntimeError("replay_manifest.json has no recording object")
    if manifest.get("task", {}).get("name") != "t09_insert_peg":
        raise RuntimeError("legacy diagnostic accepts only t09_insert_peg")
    if recording.get("sim_backend") != "isaacsim_newton":
        raise RuntimeError("legacy diagnostic requires a Newton source")
    environment_schema = recording_schema.get("environment_schema")
    if not isinstance(environment_schema, dict):
        raise RuntimeError("recording schema has no environment_schema")
    if environment_schema.get("name") != "handson.b03_newton_ground_truth":
        raise RuntimeError("source has no B03 Newton ground-truth trace")

    data = zarr.open(str(demo_dir / "replay.zarr"), mode="r")["data"]
    missing = [name for name in _TARGET_FIELDS if name not in data]
    if missing:
        raise RuntimeError(f"legacy source is missing drive fields: {missing}")
    targets = {name: np.asarray(data[name]) for name in _TARGET_FIELDS}
    step_count = int(targets[_TARGET_FIELDS[0]].shape[0])
    if step_count <= 0 or any(value.shape[0] != step_count for value in targets.values()):
        raise RuntimeError("legacy source drive arrays have inconsistent lengths")

    newton = environment_schema.get("newton")
    if not isinstance(newton, dict):
        raise RuntimeError("source environment schema has no Newton contract")
    joint_names = newton.get("robot_joint_names")
    sides = newton.get("robot_sides")
    if not isinstance(joint_names, dict) or not isinstance(sides, list):
        raise RuntimeError("source Newton robot layout is incomplete")
    hand_rows: list[tuple[int, int]] = []
    for side_index, side_value in enumerate(sides):
        side = str(side_value)
        names = joint_names.get(side)
        if not isinstance(names, list):
            raise RuntimeError(f"source Newton joint names are missing for {side}")
        hand_rows.extend(
            (side_index, joint_index)
            for joint_index, name in enumerate(names)
            if str(name).startswith(f"{side}_finger")
        )

    source_stats: dict[str, Any] = {
        "step_count": step_count,
        "target_sha256": _framed_array_digest(targets),
        "recorded_physics_hz": (
            1.0 / float(newton["physics_dt"])
            if float(newton.get("physics_dt", 0.0)) > 0.0
            else None
        ),
        "recorded_substeps_per_control": int(newton.get("n_substeps", 0)),
    }
    velocity_name = "next_newton_substep_robot_joint_velocity"
    penetration_name = "next_newton_substep_hand_peg_max_solid_penetration"
    if velocity_name in data and hand_rows:
        velocity = np.asarray(data[velocity_name], dtype=np.float64)
        hand_velocity = np.stack(
            [velocity[:, :, side, joint] for side, joint in hand_rows],
            axis=-1,
        )
        source_stats["recorded_hand_max_abs_joint_velocity_rad_s"] = float(
            np.max(np.abs(hand_velocity))
        )
        source_stats["recorded_hand_velocity_violation_samples"] = int(
            np.count_nonzero(
                np.abs(hand_velocity)
                > B03_NEWTON_HAND_MAX_ABS_JOINT_VELOCITY_RAD_S
            )
        )
    if penetration_name in data:
        penetration = np.asarray(data[penetration_name], dtype=np.float64)
        source_stats["recorded_hand_peg_max_solid_penetration_m"] = float(
            np.max(penetration)
        )
        source_stats["recorded_penetration_violation_substeps"] = int(
            np.count_nonzero(
                penetration > B03_NEWTON_HAND_PEG_MAX_SOLID_PENETRATION_M
            )
        )
    return recording, recording_schema, targets, source_stats


def _source_drive_snapshot(
    targets: dict[str, np.ndarray],
    index: int,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for stored_name, value in targets.items():
        result[stored_name.removeprefix("next_")] = value[index]
    return result


def _zero_order_hold(
    source: NewtonDriveControlStep,
    runtime_substeps: int,
) -> tuple[NewtonDriveControlStep, int]:
    if runtime_substeps % source.substep_count != 0:
        raise RuntimeError(
            "current/runtime substeps must be an integer multiple of the "
            f"legacy source: runtime={runtime_substeps}, source={source.substep_count}"
        )
    factor = runtime_substeps // source.substep_count
    if factor < 1:
        raise RuntimeError("legacy replay cannot discard recorded drive targets")
    targets = np.repeat(source.targets, factor, axis=0)
    valid = np.repeat(source.valid, factor, axis=0)
    targets.setflags(write=False)
    valid.setflags(write=False)
    return (
        NewtonDriveControlStep(
            targets=targets,
            valid=valid,
            substep_count=runtime_substeps,
            expected_joint_count=source.expected_joint_count,
        ),
        factor,
    )


def _trace_fields(
    observation: dict[str, Any],
    runtime_schema: dict[str, Any],
) -> tuple[str, ...]:
    specs = runtime_schema.get("fields")
    if not isinstance(specs, dict):
        raise RuntimeError("current runtime schema has no field specs")
    fields: list[str] = []
    for raw_name, raw_spec in specs.items():
        name = str(raw_name)
        if (
            isinstance(raw_spec, dict)
            and raw_spec.get("physical_replay") is True
            and name in observation
            and name.startswith("newton_")
        ):
            value = np.asarray(observation[name])
            if value.dtype.kind not in "biuf":
                continue
            fields.append(name)
    if not fields:
        raise RuntimeError("current runtime exposed no physical replay fields")
    return tuple(sorted(fields))


def _hand_rows(runtime_schema: dict[str, Any]) -> list[tuple[int, int]]:
    newton = runtime_schema.get("newton")
    if not isinstance(newton, dict):
        raise RuntimeError("current runtime schema has no Newton layout")
    sides = newton.get("robot_sides")
    joint_names = newton.get("robot_joint_names")
    if not isinstance(sides, list) or not isinstance(joint_names, dict):
        raise RuntimeError("current runtime robot layout is incomplete")
    rows: list[tuple[int, int]] = []
    for side_index, side_value in enumerate(sides):
        side = str(side_value)
        names = joint_names.get(side)
        if not isinstance(names, list):
            raise RuntimeError(f"current joint names are missing for {side}")
        rows.extend(
            (side_index, joint_index)
            for joint_index, name in enumerate(names)
            if str(name).startswith(f"{side}_finger")
        )
    return rows


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(demo_dir: Path, output_dir: Path, max_steps: int | None) -> dict[str, Any]:
    recording, source_schema, source_targets, source_stats = _source_inputs(
        demo_dir
    )
    source_path = RecordedNewtonDrivePath(source_schema)
    total_steps = int(source_stats["step_count"])
    step_count = total_steps if max_steps is None else min(total_steps, max_steps)
    if step_count <= 0:
        raise RuntimeError("diagnostic step count must be positive")

    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / REPORT_NAME
    trace_path = output_dir / TRACE_NAME
    schema_path = output_dir / SCHEMA_NAME
    started_at = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(
        timespec="microseconds"
    )
    report: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "non_certifying": True,
        "certification_eligible": False,
        "reason": (
            "legacy source fingerprint/validity differs from the current "
            "production physics contract"
        ),
        "run_id": uuid.uuid4().hex,
        "process_id": os.getpid(),
        "source_demo": str(demo_dir),
        "source_stats": source_stats,
        "steps_requested": step_count,
        "source_steps_available": total_steps,
        "drive_adaptation": (
            "zero_order_hold_each_recorded_120hz_target_for_two_240hz_substeps"
        ),
        "current_physics_hz": CURRENT_PHYSICS_HZ,
        "current_control_hz": CURRENT_CONTROL_HZ,
        "started_at": started_at,
        "completed": False,
        "trace_complete": False,
        "state_writes_after_action_start": None,
    }
    environment = None
    trace_lists: dict[str, list[np.ndarray]] = {}
    error: BaseException | None = None
    try:
        environment = TaskConfig().get_native_environment(
            sim_backend="isaacsim_newton",
            policy_mode=True,
            render_mode="rgb_array",
            randomize=bool(recording.get("randomize", False)),
            randomize_dynamics=False,
            image_obs=False,
            seed=int(recording["collection_run_seed"]),
            arm_type=str(recording.get("arm_type", "rokae")),
            hand_type=str(recording.get("hand_type", "wuji1")),
            newton_replay_profile="cross_gpu_v1",
            newton_use_cuda_graph=True,
            control_dt=1.0 / CURRENT_CONTROL_HZ,
            physics_dt=1.0 / CURRENT_PHYSICS_HZ,
        )
        observation, _info = environment.reset(
            seed=int(recording["episode_reset_seed"])
        )
        raw = environment.unwrapped
        trace = getattr(raw, "_physx_ground_truth_trace", None)
        if trace is None or not getattr(raw, "_newton_ground_truth_enabled", False):
            raise RuntimeError("current B03 scene did not enable Newton trace")
        runtime_schema = raw.get_recording_schema()
        if not isinstance(runtime_schema, dict):
            raise RuntimeError("current B03 runtime schema is invalid")
        runtime_newton = runtime_schema.get("newton")
        if not isinstance(runtime_newton, dict):
            raise RuntimeError("current B03 runtime has no Newton contract")
        runtime_substeps = int(runtime_newton.get("n_substeps", 0))
        if runtime_substeps != CURRENT_PHYSICS_HZ // CURRENT_CONTROL_HZ:
            raise RuntimeError(
                f"current B03 runtime used {runtime_substeps} substeps; expected 8"
            )
        fields = _trace_fields(dict(observation), runtime_schema)
        for name in fields:
            trace_lists[name] = [np.asarray(observation[name]).copy()]
        hand_rows = _hand_rows(runtime_schema)
        if len(hand_rows) != 40:
            raise RuntimeError(
                f"current B03 runtime exposed {len(hand_rows)} hand joints; expected 40"
            )
        schema_path.write_text(
            json.dumps(runtime_schema, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        report.update(
            {
                "runtime_source_identity": runtime_newton.get(
                    "source_runtime_identity"
                ),
                "runtime_physics_config": runtime_newton.get("physics_config"),
                "runtime_substeps_per_control": runtime_substeps,
                "trace_fields": list(fields),
                "runtime_schema_sha256": hashlib.sha256(
                    schema_path.read_bytes()
                ).hexdigest(),
            }
        )

        maximum_hand_velocity = 0.0
        maximum_penetration = 0.0
        velocity_violation_substeps = 0
        penetration_violation_substeps = 0
        nonfinite_samples = 0
        success_steps = 0
        hold_factor: int | None = None
        for index in range(step_count):
            source_step = source_path.control_step(
                _source_drive_snapshot(source_targets, index)
            )
            drive_step, factor = _zero_order_hold(
                source_step,
                runtime_substeps,
            )
            if hold_factor is None:
                hold_factor = factor
            elif hold_factor != factor:
                raise RuntimeError("legacy drive hold factor changed during replay")
            next_observation, _reward, _terminated, _truncated, step_info = (
                run_newton_drive_control_step(
                    raw,
                    trace,
                    source_schema,
                    drive_step,
                )
            )
            snapshot = dict(next_observation)
            for name in fields:
                value = np.asarray(snapshot[name])
                if value.shape != trace_lists[name][0].shape:
                    raise RuntimeError(
                        f"current trace shape changed for {name}: "
                        f"{value.shape} != {trace_lists[name][0].shape}"
                    )
                nonfinite_samples += int(
                    value.dtype.kind == "f" and np.count_nonzero(~np.isfinite(value))
                )
                trace_lists[name].append(value.copy())

            velocity = np.asarray(
                snapshot["newton_substep_robot_joint_velocity"],
                dtype=np.float64,
            )
            hand_velocity = np.stack(
                [velocity[:, side, joint] for side, joint in hand_rows],
                axis=-1,
            )
            maximum_hand_velocity = max(
                maximum_hand_velocity,
                float(np.max(np.abs(hand_velocity))),
            )
            velocity_violation_substeps += int(
                np.count_nonzero(
                    np.any(
                        np.abs(hand_velocity)
                        > B03_NEWTON_HAND_MAX_ABS_JOINT_VELOCITY_RAD_S,
                        axis=-1,
                    )
                )
            )
            penetration = np.asarray(
                snapshot[
                    "newton_substep_hand_peg_max_solid_penetration"
                ],
                dtype=np.float64,
            )
            maximum_penetration = max(
                maximum_penetration,
                float(np.max(penetration)),
            )
            penetration_violation_substeps += int(
                np.count_nonzero(
                    np.any(
                        penetration
                        > B03_NEWTON_HAND_PEG_MAX_SOLID_PENETRATION_M,
                        axis=-1,
                    )
                )
            )
            success_steps += int(bool(step_info.get("succeed", False)))
            if (index + 1) % 100 == 0 or index + 1 == step_count:
                print(
                    "[B03 legacy diagnostic] "
                    f"step={index + 1}/{step_count} "
                    f"max_qd={maximum_hand_velocity:.6g}rad/s "
                    f"max_penetration={maximum_penetration * 1000.0:.6g}mm",
                    flush=True,
                )

        arrays = {
            name: np.stack(values, axis=0)
            for name, values in trace_lists.items()
        }
        np.savez_compressed(trace_path, **arrays)
        report.update(
            {
                "completed": True,
                "trace_complete": True,
                "steps_replayed": step_count,
                "trace_samples_including_reset": step_count + 1,
                "zero_order_hold_factor": hold_factor,
                "runtime_hand_max_abs_joint_velocity_rad_s": (
                    maximum_hand_velocity
                ),
                "runtime_hand_velocity_limit_rad_s": (
                    B03_NEWTON_HAND_MAX_ABS_JOINT_VELOCITY_RAD_S
                ),
                "runtime_velocity_violation_substeps": (
                    velocity_violation_substeps
                ),
                "runtime_hand_peg_max_solid_penetration_m": (
                    maximum_penetration
                ),
                "runtime_penetration_limit_m": (
                    B03_NEWTON_HAND_PEG_MAX_SOLID_PENETRATION_M
                ),
                "runtime_penetration_violation_substeps": (
                    penetration_violation_substeps
                ),
                "nonfinite_trace_samples": nonfinite_samples,
                "success_steps": success_steps,
                "final_success": bool(success_steps and bool(step_info.get("succeed", False))),
                "state_writes_after_action_start": int(
                    getattr(raw, "_trajectory_state_write_count", 0)
                ),
                "trace_file": TRACE_NAME,
                "trace_array_sha256": _framed_array_digest(arrays),
                "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                "trace_bytes": trace_path.stat().st_size,
            }
        )
    except BaseException as exc:
        error = exc
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        if environment is not None:
            try:
                report["lifecycle"] = destroy_newton_scene(environment)
            except BaseException as exc:
                report["lifecycle_error"] = f"{type(exc).__name__}: {exc}"
                if error is None:
                    error = exc
        report["completed_at"] = (
            dt.datetime.now(dt.timezone.utc)
            .astimezone()
            .isoformat(timespec="microseconds")
        )
        report["passed_execution"] = bool(
            error is None
            and report.get("completed") is True
            and report.get("trace_complete") is True
            and report.get("nonfinite_trace_samples") == 0
            and report.get("state_writes_after_action_start") == 0
            and report.get("lifecycle", {}).get("scene_destroyed") is True
        )
        _atomic_json(report_path, report)
        summary_keys = (
            "passed_execution",
            "steps_replayed",
            "runtime_hand_max_abs_joint_velocity_rad_s",
            "runtime_velocity_violation_substeps",
            "runtime_hand_peg_max_solid_penetration_m",
            "runtime_penetration_violation_substeps",
            "nonfinite_trace_samples",
            "success_steps",
            "final_success",
            "state_writes_after_action_start",
            "trace_array_sha256",
            "trace_sha256",
            "error",
        )
        summary = {key: report[key] for key in summary_keys if key in report}
        print(
            "B03_LEGACY_DIAGNOSTIC="
            + json.dumps(summary, sort_keys=True),
            flush=True,
        )
    if error is not None:
        raise error
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Diagnostic-only prefix length; omit to replay the full episode.",
    )
    args = parser.parse_args()
    demo_dir = args.demo_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not demo_dir.is_dir():
        parser.error(f"demo directory does not exist: {demo_dir}")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    try:
        run(demo_dir, output_dir, args.max_steps)
    except BaseException as exc:
        print(
            f"[B03 legacy diagnostic] FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        os._exit(1)
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
