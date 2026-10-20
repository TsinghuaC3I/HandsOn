#!/usr/bin/env python3
"""Collect or fresh-process replay the complete B320-BC012 PhysX slice."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

import numpy as np
import zarr


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "handson"))

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("HANDSON_ENABLE_ISAACSIM_NEWTON", "0")

from handson.sim.isaaclab.bc012_env import BC012IsaacLabEnv  # noqa: E402
from taskdev.tasks.bench320_bc012_dual_can_synchronized_bin_load.episode import (  # noqa: E402
    TASK_ID,
    TASK_NAME,
    validate_episode,
    write_episode,
)


STATE_POSE_FIELDS = ("left_can_pose", "right_can_pose", "open_bin_pose")
BC012_TASK_STATE_FIELDS = (
    "left_physical_grasp_seen",
    "right_physical_grasp_seen",
    "left_grasp_contact_streak",
    "right_grasp_contact_streak",
    "left_release_contact_streak",
    "right_release_contact_streak",
    "left_release_frame",
    "right_release_frame",
    "success_counter",
    "failed",
    "left_supported",
    "right_supported",
    "left_hand_contact_force",
    "right_hand_contact_force",
)
PHYSX_T0_RESTORE_FIELDS = (
    "physx_command_tcp_pose",
    "physx_command_hand",
    "physx_command_hand_valid",
    "physx_pose_commanded",
    "physx_pose_filter_state",
    "physx_pose_filter_valid",
    "physx_rng_mt19937_keys",
    "physx_rng_mt19937_position",
    "physx_rng_mt19937_has_gauss",
    "physx_rng_mt19937_cached_gaussian",
    "physx_rng_seed",
    "physx_robot_joint_position",
    "physx_robot_joint_velocity",
    "physx_robot_joint_valid",
    "physx_robot_joint_target",
    "physx_robot_joint_target_valid",
    "physx_task_rigid_body_state",
    "physx_task_rigid_body_valid",
    "physx_control_step",
)
PHYSX_REPLAY_COMPARE_FIELDS = {
    "physx_robot_joint_position": 0.05,
    "physx_robot_root_state": 0.05,
    "physx_tcp_state": 0.05,
    "physx_task_rigid_body_state": 0.05,
    "physx_task_articulation_root_state": 0.05,
    "physx_task_articulation_dof_position": 0.05,
    "physx_substep_robot_joint_position": 0.05,
    "physx_substep_robot_root_state": 0.05,
    "physx_substep_tcp_state": 0.05,
    "physx_substep_task_rigid_body_state": 0.05,
    "physx_substep_task_articulation_root_state": 0.05,
    "physx_substep_task_articulation_dof_position": 0.05,
}
_INNER_PROCESS_ENV = "HANDSON_BC012_VERTICAL_SLICE_INNER"
_STDOUT_LOG_ENV = "HANDSON_BC012_STDOUT_LOG"
_MAX_ACCEPTANCE_LOG_BYTES = 8 * 1024 * 1024
_DISJOINTED_JOINT_WARNING = (
    "CreateJoint - found a joint with disjointed body transforms"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("collect", "replay"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--zero-action-steps", type=int, default=1)
    return parser


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _namespace_evidence() -> dict[str, Any]:
    return {
        "taskdev_host": os.environ.get("TASKDEV_HOST", ""),
        "observed_hostname": socket.gethostname(),
        "physical_gpu": os.environ.get("TASKDEV_PHYSICAL_GPU", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "isaac_device": os.environ.get("HANDSON_ISAACLAB_DEVICE", "cuda:0"),
    }


def _state(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    value = observation.get("state")
    if not isinstance(value, Mapping):
        raise TypeError("BC012 runner requires the unflattened state observation")
    return value


def _recorded_t0_snapshot(data: Any) -> dict[str, np.ndarray]:
    """Load the one schema-authorized physical snapshot before action zero."""

    missing = [name for name in PHYSX_T0_RESTORE_FIELDS if name not in data]
    if missing:
        raise RuntimeError(
            "BC012 replay episode is missing t0 restore fields: "
            + ", ".join(missing)
        )
    return {
        name: np.asarray(data[name][0]).copy()
        for name in PHYSX_T0_RESTORE_FIELDS
    }


def _compare_public_t0_origin(
    observation: Mapping[str, Any],
    recorded: Mapping[str, np.ndarray],
    recording_schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare a rebuilt fresh-scene origin using recorded field tolerances."""

    field_specs = recording_schema.get("fields")
    if not isinstance(field_specs, Mapping):
        raise RuntimeError("BC012 recording schema has no field specifications")
    errors: dict[str, float] = {}
    tolerances: dict[str, float] = {}
    for name in PHYSX_T0_RESTORE_FIELDS:
        if name not in observation or name not in recorded:
            raise RuntimeError(f"BC012 public t0 origin is missing {name}")
        spec = field_specs.get(name)
        if not isinstance(spec, Mapping):
            raise RuntimeError(f"BC012 recording schema has no spec for {name}")
        tolerance = float(spec.get("replay_atol", -1.0))
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise RuntimeError(f"BC012 recording has invalid t0 tolerance for {name}")
        actual = np.asarray(observation[name])
        expected = np.asarray(recorded[name])
        if actual.shape != expected.shape:
            raise RuntimeError(
                f"BC012 public t0 shape changed for {name}: "
                f"actual={actual.shape}, recorded={expected.shape}"
            )
        difference = np.abs(
            actual.astype(np.float64) - expected.astype(np.float64)
        )
        error = (
            float(np.max(difference))
            if difference.size and np.isfinite(difference).all()
            else (0.0 if not difference.size else float("inf"))
        )
        errors[name] = error
        tolerances[name] = tolerance
    return {
        "matched": all(errors[name] <= tolerances[name] for name in errors),
        "maximum_errors": errors,
        "tolerances": tolerances,
    }


def _camera_metrics(observation: Mapping[str, Any]) -> dict[str, Any]:
    images = observation.get("images")
    if not isinstance(images, Mapping):
        raise TypeError("BC012 collection requires three live RGB cameras")
    result = {}
    for canonical, aliases in {
        "ego": ("ego", "random_camera"),
        "wrist_left": ("wrist_left",),
        "wrist_right": ("wrist_right",),
    }.items():
        frame = next((images[name] for name in aliases if name in images), None)
        if frame is None:
            raise KeyError(f"missing camera {canonical}")
        array = np.asarray(frame)
        result[canonical] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "nonzero_fraction": float(np.count_nonzero(array) / array.size),
            "pass": bool(
                array.shape == (480, 640, 3)
                and array.dtype == np.uint8
                and float(np.mean(array)) >= 2.0
                and float(np.std(array)) >= 1.0
            ),
        }
    return result


def _save_preview(observation: Mapping[str, Any], path: Path) -> None:
    import imageio.v3 as iio

    images = observation["images"]
    frame = images.get("ego", images.get("random_camera"))
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.asarray(frame, dtype=np.uint8))


def _make_env(*, seed: int, device: str, image_obs: bool) -> BC012IsaacLabEnv:
    env = BC012IsaacLabEnv(
        seed=seed,
        physics_dt=1.0 / 120.0,
        time_limit=30.0,
        render_mode="rgb_array",
        image_obs=image_obs,
        randomize=True,
        randomize_dynamics=False,
        hand_type="wuji1",
        arm_type="rokae",
        launch_isaac=True,
        headless=True,
        render_resolution=(480, 640),
        hz=30,
        isaac_device=device,
        physics_backend="physx",
    )
    env._fast_shutdown_on_close = True
    return env


def _collect(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
    os.environ["HANDSON_ISAACLAB_DEVICE"] = str(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / "episode" / "replay.zarr"
    summary_path = output_dir / "collection_summary.json"
    env: BC012IsaacLabEnv | None = None
    started = time.time()
    summary: dict[str, Any] = {
        "phase": "collect",
        "task_id": TASK_ID,
        "runtime_task_name": TASK_NAME,
        "seed": int(args.seed),
        "source_pid": os.getpid(),
        "namespace": _namespace_evidence(),
        "pass": False,
    }
    try:
        env = _make_env(seed=args.seed, device=args.device, image_obs=True)
        observation, reset_info = env.reset(seed=args.seed)
        first_reset = {
            name: np.asarray(_state(observation)[name]).copy()
            for name in STATE_POSE_FIELDS
        }
        camera_metrics = _camera_metrics(observation)
        _save_preview(observation, output_dir / "preview.png")

        preflight_zero_action_steps = max(1, int(args.zero_action_steps))
        zero_action = np.zeros(54, dtype=np.float32)
        zero_info: Mapping[str, Any] = {}
        for _ in range(preflight_zero_action_steps):
            zero_observation, _, zero_terminal, zero_truncated, zero_info = env.step(
                zero_action
            )
            if zero_terminal or zero_truncated:
                raise RuntimeError("zero-action smoke unexpectedly terminated")
            if not all(
                np.isfinite(np.asarray(_state(zero_observation)[name])).all()
                for name in STATE_POSE_FIELDS
            ):
                raise RuntimeError("zero-action smoke produced non-finite truth")

        observation, second_reset_info = env.reset(seed=args.seed)
        reset_pose_error = max(
            float(
                np.max(
                    np.abs(
                        first_reset[name]
                        - np.asarray(_state(observation)[name], dtype=np.float64)
                    )
                )
            )
            for name in STATE_POSE_FIELDS
        )
        if reset_pose_error > 1.0e-9:
            raise RuntimeError(
                f"same-seed reset is not exact: max pose error {reset_pose_error}"
            )

        actions = env.automatic_success_actions(observation)
        recording_schema = env.get_recording_schema()
        if not isinstance(recording_schema, Mapping):
            raise RuntimeError("BC012 PhysX recording schema is unavailable")
        transitions: list[dict[str, Any]] = []
        final_info: Mapping[str, Any] = second_reset_info
        terminal = False
        truncated = False
        for action in actions:
            next_observation, _, terminal, truncated, final_info = env.step(action)
            transitions.append(
                {
                    "observation": observation,
                    "action": np.asarray(action, dtype=np.float32),
                    "next_observation": next_observation,
                    "terminal": bool(terminal),
                    "truncated": bool(truncated),
                    "succeed": bool(final_info.get("succeed", False)),
                }
            )
            observation = next_observation
            if terminal or truncated:
                break

        episode_summary = write_episode(
            episode_path,
            transitions,
            reset_seed=args.seed,
            reset_options={
                "randomize": True,
                "randomize_dynamics": False,
                "physics_dt": 1.0 / 120.0,
                "control_hz": 30,
                "image_shape_hwc": [480, 640, 3],
                "preflight_zero_action_steps": preflight_zero_action_steps,
            },
            recording_schema=recording_schema,
        )
        semantic_success = bool(final_info.get("succeed", False))
        state_write_count = int(getattr(env, "_trajectory_state_write_count", 0))
        summary.update(
            {
                "reset_info": dict(reset_info),
                "same_seed_reset_max_pose_error_m": reset_pose_error,
                "zero_action": {
                    "steps": preflight_zero_action_steps,
                    "succeed": bool(zero_info.get("succeed", False)),
                    "failed": bool(zero_info.get("failed", False)),
                    "pass": True,
                },
                "camera_metrics": camera_metrics,
                "automatic_action_count": len(actions),
                "recorded_steps": len(transitions),
                "terminal": bool(terminal),
                "truncated": bool(truncated),
                "semantic_success": semantic_success,
                "final_info": dict(final_info),
                "trajectory_state_write_count": state_write_count,
                "episode": episode_summary,
                "preview": str(output_dir / "preview.png"),
                "elapsed_seconds": time.time() - started,
                "pass": bool(
                    semantic_success
                    and terminal
                    and not truncated
                    and state_write_count == 0
                    and all(value["pass"] for value in camera_metrics.values())
                    and episode_summary["pass"]
                ),
            }
        )
        _write_json(summary_path, summary)
        if not summary["pass"]:
            raise RuntimeError("BC012 collection vertical slice did not pass")
        validate_episode(
            episode_path,
            require_success=True,
            require_physical_replay=True,
        )
        return summary
    except Exception as exc:
        summary.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.time() - started,
                "pass": False,
            }
        )
        _write_json(summary_path, summary)
        raise
    finally:
        if env is not None:
            env.close()


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
    os.environ["HANDSON_ISAACLAB_DEVICE"] = str(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "fresh_process_replay.json"
    episode_path = (args.episode or output_dir / "episode" / "replay.zarr").resolve()
    validation = validate_episode(
        episode_path,
        require_success=True,
        require_physical_replay=True,
    )
    root = zarr.open_group(str(episode_path), mode="r")
    data = root["data"]
    seed = int(root.attrs["reset_seed"])
    source_pid = int(root.attrs["source_pid"])
    recording_schema = json.loads(root.attrs["recording_schema_json"])
    reset_options = json.loads(root.attrs["reset_options_json"])
    preflight_zero_action_steps = int(
        reset_options.get("preflight_zero_action_steps", 0)
    )
    if preflight_zero_action_steps < 1:
        raise RuntimeError(
            "BC012 episode does not record its collection preflight steps"
        )
    if source_pid == os.getpid():
        raise RuntimeError("fresh-process replay cannot reuse the collection PID")
    env: BC012IsaacLabEnv | None = None
    started = time.time()
    report: dict[str, Any] = {
        "phase": "replay",
        "task_id": TASK_ID,
        "runtime_task_name": TASK_NAME,
        "source_pid": source_pid,
        "replay_pid": os.getpid(),
        "seed": seed,
        "episode": str(episode_path),
        "namespace": _namespace_evidence(),
        "pass": False,
    }
    try:
        # Recreate the same camera-enabled USD scene as collection.  Omitting
        # the three sensors changes prim creation/topology order before first
        # PhysX and invalidates contact-history reconstruction even though the
        # scalar solver configuration still matches.
        env = _make_env(seed=seed, device=args.device, image_obs=True)
        # Collection warms the live scene with one recorded number of idle
        # actions, then resets the same scene again before action[0].  PhysX
        # does not expose all contact/friction caches for trace.restore(), so
        # fresh replay must mirror this preflight before applying the exact t0
        # snapshot rather than starting from a newly constructed cold scene.
        observation, _ = env.reset(seed=seed)
        zero_action = np.zeros(54, dtype=np.float32)
        for _ in range(preflight_zero_action_steps):
            _, _, preflight_terminal, preflight_truncated, _ = env.step(
                zero_action
            )
            if preflight_terminal or preflight_truncated:
                raise RuntimeError(
                    "BC012 replay preflight unexpectedly terminated"
                )
        observation, _ = env.reset(seed=seed)
        from handson.sim.isaaclab.articulated_replay import (
            RecordedArticulatedDrivePath,
            run_articulated_drive_control_step,
            validate_recorded_physics_configuration,
        )

        runtime_schema = env.get_recording_schema()
        if not isinstance(runtime_schema, Mapping):
            raise RuntimeError("fresh BC012 PhysX schema is unavailable")
        physics_configuration = validate_recorded_physics_configuration(
            recording_schema,
            runtime_schema,
        )
        drive_path = RecordedArticulatedDrivePath(recording_schema)
        trace = env._physx_ground_truth_trace
        if trace is None:
            raise RuntimeError("fresh BC012 PhysX trace is unavailable")
        reset_contract = recording_schema.get("reset_replay_contract")
        if not isinstance(reset_contract, Mapping):
            raise RuntimeError("BC012 recording has no reset replay contract")
        if (
            reset_contract.get("allows_single_recorded_t0_restore") is not True
            or reset_contract.get("allow_pre_action_state_overwrite") is not True
            or reset_contract.get("forbids_state_writes_after_action_start") is not True
            or int(
                reset_contract.get(
                    "physics_steps_after_exact_recorded_t0_restore",
                    -1,
                )
            )
            != 0
        ):
            raise RuntimeError(
                "BC012 recording does not authorize one no-step pre-action t0 restore"
            )
        t0_snapshot = _recorded_t0_snapshot(data)
        observation = env._attach_physx_ground_truth(env._compute_observation())
        t0_origin = _compare_public_t0_origin(
            observation,
            t0_snapshot,
            recording_schema,
        )
        if t0_origin["matched"]:
            # Match the generic certifier: a reconstructed fresh-scene origin
            # must be preserved, because another state write would discard the
            # private contact/warm-start history we just rebuilt.
            trace.begin_substep_capture()
            t0_restore: dict[str, int] = {}
            t0_restore_applied = False
        else:
            t0_restore = trace.restore(t0_snapshot, recording_schema)
            # Synchronize public tensor caches without advancing PhysX.  This
            # optional fallback is the sole measured-state write and remains
            # before action[0], as authorized by the recorded reset contract.
            observation = env._attach_physx_ground_truth(
                env._compute_observation()
            )
            trace.begin_substep_capture()
            t0_restore_applied = True
        initial_errors = {
            name: float(
                np.max(
                    np.abs(
                        np.asarray(_state(observation)[name], dtype=np.float64)
                        - np.asarray(data[f"state_{name}"][0], dtype=np.float64)
                    )
                )
            )
            for name in STATE_POSE_FIELDS
        }
        maximum_errors = {name: 0.0 for name in STATE_POSE_FIELDS}
        maximum_error_steps = {name: -1 for name in STATE_POSE_FIELDS}
        first_pose_tolerance_exceedance: dict[str, Any] | None = None
        physical_errors: dict[str, float] = {}
        physical_error_steps: dict[str, int] = {}
        first_physical_tolerance_exceedance: dict[str, Any] | None = None
        compared_physical_fields = {
            name: tolerance
            for name, tolerance in PHYSX_REPLAY_COMPARE_FIELDS.items()
            if name in observation and f"next_{name}" in data
        }
        required_physical_fields = {
            "physx_robot_joint_position",
            "physx_task_rigid_body_state",
            "physx_substep_robot_joint_position",
            "physx_substep_task_rigid_body_state",
        }
        missing_physical_fields = sorted(
            required_physical_fields - set(compared_physical_fields)
        )
        if missing_physical_fields:
            raise RuntimeError(
                "fresh BC012 replay is missing required physical comparison "
                f"fields: {missing_physical_fields}"
            )
        maximum_hidden_state_error = 0.0
        maximum_hidden_state_errors = np.zeros(
            len(BC012_TASK_STATE_FIELDS), dtype=np.float64
        )
        terminal_step = -1
        final_info: Mapping[str, Any] = {}
        applied_substeps = 0
        applied_joint_targets = 0
        maximum_intra_step_target_delta = 0.0
        for index, action in enumerate(data["action"]):
            raw_action = np.asarray(action, dtype=np.float32)
            # Install the raw command without recomputing IK; recorded
            # finite-mass articulation targets are then applied per substep.
            # Grasp/release events remain live PhysX contact outcomes.
            env._apply_action(raw_action)
            expected_next_hidden = np.asarray(
                data["next_state_bc012_task_state"][index],
                dtype=np.float64,
            )
            recorded_next = {
                name: np.asarray(data[f"next_{name}"][index])
                for name in (
                    "physx_substep_count",
                    "physx_substep_robot_joint_target",
                    "physx_substep_robot_joint_target_valid",
                    "physx_robot_joint_target",
                    "physx_robot_joint_target_valid",
                )
            }
            drive_step = drive_path.control_step(recorded_next)
            result, drive_report = run_articulated_drive_control_step(
                env,
                trace,
                recording_schema,
                drive_step,
            )
            observation, _, terminal, truncated, final_info = result
            applied_substeps += int(drive_report["recorded_substep_count"])
            applied_joint_targets += int(
                drive_report["applied_robot_joint_target_count"]
            )
            maximum_intra_step_target_delta = max(
                maximum_intra_step_target_delta,
                float(drive_report["max_intra_step_target_delta"]),
            )
            for name in STATE_POSE_FIELDS:
                error = float(
                    np.max(
                        np.abs(
                            np.asarray(_state(observation)[name], dtype=np.float64)
                            - np.asarray(
                                data[f"next_state_{name}"][index],
                                dtype=np.float64,
                            )
                        )
                    )
                )
                if error > maximum_errors[name]:
                    maximum_errors[name] = error
                    maximum_error_steps[name] = index
                if error > 0.03 and first_pose_tolerance_exceedance is None:
                    first_pose_tolerance_exceedance = {
                        "field": name,
                        "step": index,
                        "error": error,
                    }
            hidden_errors = np.abs(
                np.asarray(
                    _state(observation)["bc012_task_state"],
                    dtype=np.float64,
                )
                - expected_next_hidden
            )
            maximum_hidden_state_errors = np.maximum(
                maximum_hidden_state_errors,
                hidden_errors,
            )
            hidden_error = float(np.max(hidden_errors))
            maximum_hidden_state_error = max(
                maximum_hidden_state_error,
                hidden_error,
            )
            for name in compared_physical_fields:
                actual = np.asarray(observation[name])
                expected = np.asarray(data[f"next_{name}"][index])
                if actual.shape != expected.shape:
                    raise RuntimeError(
                        f"fresh BC012 replay field {name} shape changed: "
                        f"actual={actual.shape}, recorded={expected.shape}"
                    )
                field_error = float(
                    np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))
                )
                if field_error > physical_errors.get(name, -1.0):
                    physical_errors[name] = field_error
                    physical_error_steps[name] = index
                if (
                    field_error > compared_physical_fields[name]
                    and first_physical_tolerance_exceedance is None
                ):
                    flat_index = int(
                        np.argmax(
                            np.abs(
                                actual.astype(np.float64)
                                - expected.astype(np.float64)
                            )
                        )
                    )
                    first_physical_tolerance_exceedance = {
                        "field": name,
                        "step": index,
                        "error": field_error,
                        "flat_index": flat_index,
                        "actual": float(actual.reshape(-1)[flat_index]),
                        "expected": float(expected.reshape(-1)[flat_index]),
                    }
            if terminal or truncated:
                terminal_step = index
                break
        expected_terminal_step = int(np.flatnonzero(data["terminal"][:])[-1])
        state_write_count = int(getattr(env, "_trajectory_state_write_count", 0))
        report.update(
            {
                "episode_validation": validation,
                "initial_pose_errors": initial_errors,
                "maximum_pose_errors": maximum_errors,
                "maximum_pose_error_steps": maximum_error_steps,
                "first_pose_tolerance_exceedance": first_pose_tolerance_exceedance,
                "pose_tolerance_m_or_quaternion_component": 0.03,
                "maximum_hidden_state_error": maximum_hidden_state_error,
                "maximum_hidden_state_errors": {
                    name: float(error)
                    for name, error in zip(
                        BC012_TASK_STATE_FIELDS,
                        maximum_hidden_state_errors,
                    )
                },
                "hidden_state_tolerance": 0.0,
                "physical_field_maximum_errors": physical_errors,
                "physical_field_maximum_error_steps": physical_error_steps,
                "first_physical_tolerance_exceedance": (
                    first_physical_tolerance_exceedance
                ),
                "physical_field_tolerances": compared_physical_fields,
                "compared_physical_field_count": len(compared_physical_fields),
                "compared_physical_substep_field_count": sum(
                    name.startswith("physx_substep_")
                    for name in compared_physical_fields
                ),
                "terminal_step": terminal_step,
                "expected_terminal_step": expected_terminal_step,
                "semantic_success": bool(final_info.get("succeed", False)),
                "final_task_info": {
                    name: final_info.get(name)
                    for name in (
                        "bc012_event_history",
                        "bc012_grasp_errors",
                        "bc012_left_owned",
                        "bc012_right_owned",
                        "bc012_left_ready",
                        "bc012_right_ready",
                        "bc012_left_supported",
                        "bc012_right_supported",
                        "bc012_release_delta_frames",
                        "bc012_success_counter",
                        "bc012_failed",
                        "succeed",
                    )
                },
                "physics_configuration": physics_configuration,
                "pre_action_t0_restore": {
                    "applied_once": t0_restore_applied,
                    "physics_steps_after_restore": 0,
                    "restored": t0_restore,
                },
                "pre_action_public_t0_origin": t0_origin,
                "pre_action_reset_preflight": {
                    "same_seed_reset_count": 2,
                    "zero_action_steps": preflight_zero_action_steps,
                },
                "collection_scene_configuration_recreated": {
                    "image_observations": True,
                    "camera_count": 3,
                },
                "replay_input_mode": "recorded_articulated_drive_targets",
                "ik_recomputed": False,
                "recorded_drive_target_substeps_applied": applied_substeps,
                "recorded_drive_target_joint_writes_applied": applied_joint_targets,
                "grasp_assist_enabled": False,
                "maximum_intra_step_target_delta": maximum_intra_step_target_delta,
                "trajectory_state_write_count": state_write_count,
                "elapsed_seconds": time.time() - started,
                "pass": bool(
                    max(initial_errors.values()) <= 1.0e-9
                    and max(maximum_errors.values()) <= 0.03
                    and maximum_hidden_state_error <= 0.0
                    and all(
                        physical_errors.get(name, float("inf")) <= tolerance
                        for name, tolerance in compared_physical_fields.items()
                    )
                    and terminal_step == expected_terminal_step
                    and bool(final_info.get("succeed", False))
                    and applied_substeps > 0
                    and applied_joint_targets > 0
                    and state_write_count == 0
                ),
            }
        )
        _write_json(report_path, report)
        if not report["pass"]:
            raise RuntimeError("BC012 fresh-process replay did not pass")
        return report
    except Exception as exc:
        report.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.time() - started,
                "pass": False,
            }
        )
        _write_json(report_path, report)
        raise
    finally:
        if env is not None:
            env.close()


def _inner_main() -> int:
    args = _parser().parse_args()
    if args.phase == "collect":
        summary = _collect(args)
    else:
        if args.episode is None:
            raise SystemExit("--episode is required for fresh-process replay")
        summary = _replay(args)
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    return 0


def _bounded_stdout_audit(path: Path | None) -> dict[str, Any]:
    """Read one bounded child log and count forbidden joint warnings."""

    result: dict[str, Any] = {
        "path": str(path) if path is not None else "",
        "available": False,
        "bytes_scanned": 0,
        "maximum_bytes": _MAX_ACCEPTANCE_LOG_BYTES,
        "truncated": False,
        "disjointed_joint_warning_count": 0,
    }
    if path is None:
        return result
    try:
        size = int(path.stat().st_size)
        with path.open("rb") as stream:
            payload = stream.read(_MAX_ACCEPTANCE_LOG_BYTES + 1)
    except OSError:
        return result
    result["available"] = True
    result["bytes_scanned"] = min(len(payload), _MAX_ACCEPTANCE_LOG_BYTES)
    result["truncated"] = bool(
        size > _MAX_ACCEPTANCE_LOG_BYTES
        or len(payload) > _MAX_ACCEPTANCE_LOG_BYTES
    )
    bounded = payload[:_MAX_ACCEPTANCE_LOG_BYTES]
    result["disjointed_joint_warning_count"] = bounded.count(
        _DISJOINTED_JOINT_WARNING.encode("utf-8")
    )
    return result


def _stdout_log_path() -> Path | None:
    explicit = os.environ.get(_STDOUT_LOG_ENV)
    if explicit:
        return Path(explicit).resolve()
    try:
        target = os.readlink("/proc/self/fd/1")
    except OSError:
        return None
    path = Path(target)
    return path.resolve() if path.is_absolute() else None


def _collection_acceptance(
    report: Mapping[str, Any],
    stdout_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the bounded corrected-frame physical collection contract."""

    errors: list[str] = []

    def require(condition: bool, detail: str) -> None:
        if not condition:
            errors.append(detail)

    def integer(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value

    require(report.get("pass") is True, "child collection report is not PASS")
    require(
        report.get("semantic_success") is True,
        "semantic_success is not true",
    )
    require(report.get("terminal") is True, "collection did not terminate")
    require(report.get("truncated") is False, "collection was truncated")
    automatic_action_count = integer(report.get("automatic_action_count"))
    require(
        automatic_action_count == 732,
        f"automatic action contract is {automatic_action_count}, not 732 controls",
    )
    state_write_count = integer(report.get("trajectory_state_write_count"))
    require(
        state_write_count == 0,
        f"trajectory performed {state_write_count} forbidden state writes",
    )

    cameras = report.get("camera_metrics")
    require(
        isinstance(cameras, Mapping)
        and set(cameras) == {"ego", "wrist_left", "wrist_right"}
        and all(
            isinstance(value, Mapping) and value.get("pass") is True
            for value in cameras.values()
        ),
        "three-camera acceptance failed",
    )
    episode = report.get("episode")
    require(
        isinstance(episode, Mapping) and episode.get("pass") is True,
        "episode schema validation failed",
    )
    success_count = (
        integer(episode.get("success_count"))
        if isinstance(episode, Mapping)
        else None
    )
    require(
        success_count is not None and success_count >= 1,
        "episode contains no successful transition",
    )

    require(stdout_audit.get("available") is True, "stdout log is unavailable")
    require(stdout_audit.get("truncated") is False, "stdout audit exceeded bound")
    warning_count = integer(stdout_audit.get("disjointed_joint_warning_count"))
    require(warning_count == 0, f"disjointed joint warning count is {warning_count}")

    final_info = report.get("final_info")
    require(isinstance(final_info, Mapping), "final_info is unavailable")
    if not isinstance(final_info, Mapping):
        final_info = {}
    require(not final_info.get("bc012_grasp_errors"), "grasp errors are present")
    require(
        final_info.get("bc012_grasp_mode") == "physical_contacts",
        "grasp mode is not physical_contacts",
    )
    require(
        final_info.get("bc012_grasp_assist_enabled") is False,
        "grasp assist is enabled",
    )
    require(final_info.get("bc012_left_owned") is False, "left can remains owned")
    require(final_info.get("bc012_right_owned") is False, "right can remains owned")
    require(final_info.get("bc012_left_ready") is True, "left can is not ready")
    require(final_info.get("bc012_right_ready") is True, "right can is not ready")
    require(
        final_info.get("bc012_left_supported") is True,
        "left can lacks bin support",
    )
    require(
        final_info.get("bc012_right_supported") is True,
        "right can lacks bin support",
    )
    require(
        final_info.get("bc012_success_now") is True,
        "final physical predicate is false",
    )
    required_steps = integer(final_info.get("bc012_success_steps_required"))
    success_steps = integer(final_info.get("bc012_success_counter"))
    require(required_steps == 15, "success stability contract is not 15 frames")
    require(
        success_steps is not None
        and required_steps is not None
        and success_steps >= required_steps,
        f"stable success held for {success_steps}/{required_steps} frames",
    )
    require(
        final_info.get("bc012_synchronized") is True,
        "release synchronization predicate is false",
    )
    release_delta = integer(final_info.get("bc012_release_delta_frames"))
    sync_window = integer(final_info.get("bc012_sync_window_frames"))
    require(
        release_delta is not None
        and sync_window is not None
        and 0 <= release_delta <= sync_window,
        f"release delta {release_delta} exceeds window {sync_window}",
    )

    events = final_info.get("bc012_event_history")
    event_rows = list(events) if isinstance(events, (list, tuple)) else []
    for side in ("left", "right"):
        grasp_events = [
            event
            for event in event_rows
            if isinstance(event, Mapping)
            and event.get("event") == "grasp"
            and event.get("side") == side
        ]
        release_events = [
            event
            for event in event_rows
            if isinstance(event, Mapping)
            and event.get("event") == "release"
            and event.get("side") == side
        ]
        grasp_frames = [integer(event.get("frame")) for event in grasp_events]
        release_frames = [integer(event.get("frame")) for event in release_events]
        require(len(grasp_events) == 1, f"{side} grasp event count is {len(grasp_events)}")
        require(
            len(release_events) == 1,
            f"{side} release event count is {len(release_events)}",
        )
        grasp_frame = grasp_frames[0] if len(grasp_frames) == 1 else None
        release_frame = release_frames[0] if len(release_frames) == 1 else None
        require(
            grasp_frame is not None
            and release_frame is not None
            and grasp_frame < release_frame,
            f"{side} physical grasp/release ordering is invalid",
        )
        require(
            bool(grasp_events)
            and grasp_events[0].get("source") == "assigned_hand_contact"
            and grasp_events[0].get("lifted") is True,
            f"{side} grasp is not proven by lifted assigned-hand contact",
        )
        require(
            bool(release_events)
            and release_events[0].get("source")
            == "assigned_hand_contact_loss",
            f"{side} release is not proven by assigned-hand contact loss",
        )

    return {
        "schema_version": "handson.bc012.collection_acceptance.v1",
        "pass": not errors,
        "errors": errors,
        "stdout_audit": dict(stdout_audit),
    }


def _discard_failed_collection_episode(output_dir: Path) -> bool:
    output = output_dir.resolve()
    episode_path = output / "episode" / "replay.zarr"
    if episode_path.parent != output / "episode" or episode_path.name != "replay.zarr":
        raise RuntimeError("refusing to remove an unexpected BC012 episode path")
    if episode_path.is_symlink() or episode_path.is_file():
        episode_path.unlink()
        return True
    if episode_path.is_dir():
        shutil.rmtree(episode_path)
        return True
    return False


def _apply_collection_acceptance(
    output_dir: Path,
    report: dict[str, Any],
    stdout_audit: Mapping[str, Any],
) -> bool:
    acceptance = _collection_acceptance(report, stdout_audit)
    episode_path = output_dir.resolve() / "episode" / "replay.zarr"
    episode_exists = episode_path.is_dir() and not episode_path.is_symlink()
    if acceptance["pass"] and not episode_exists:
        acceptance["errors"].append("successful Zarr episode is absent")
        acceptance["pass"] = False
    removed = False
    if not acceptance["pass"]:
        removed = _discard_failed_collection_episode(output_dir)
    report["collection_acceptance"] = acceptance
    report["episode_retained"] = bool(acceptance["pass"] and episode_exists)
    report["failed_episode_removed"] = bool(removed)
    report["pass"] = bool(report.get("pass") is True and acceptance["pass"])
    _write_json(output_dir.resolve() / "collection_summary.json", report)
    return bool(report["pass"])


def _run_checked_child() -> int:
    args = _parser().parse_args()
    environment = os.environ.copy()
    environment[_INNER_PROCESS_ENV] = "1"
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        env=environment,
        check=False,
    )
    report_name = (
        "collection_summary.json"
        if args.phase == "collect"
        else "fresh_process_replay.json"
    )
    report_path = args.output_dir.resolve() / report_name
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        if args.phase == "collect":
            _discard_failed_collection_episode(args.output_dir)
        print(
            f"BC012 child returned zero without a readable {report_name}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return int(completed.returncode) if completed.returncode != 0 else 86
    accepted = True
    if args.phase == "collect":
        accepted = _apply_collection_acceptance(
            args.output_dir,
            report,
            _bounded_stdout_audit(_stdout_log_path()),
        )
    if completed.returncode != 0:
        return int(completed.returncode)
    if report.get("pass") is not True or not accepted:
        print(
            f"BC012 child returned zero without semantic PASS in {report_path}",
            file=sys.stderr,
            flush=True,
        )
        return 86
    return 0


def main() -> int:
    if os.environ.get(_INNER_PROCESS_ENV) == "1":
        return _inner_main()
    return _run_checked_child()


if __name__ == "__main__":
    raise SystemExit(main())
