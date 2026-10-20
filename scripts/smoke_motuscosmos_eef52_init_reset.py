#!/usr/bin/env python3
"""One-process, one-task HandsOn init/reset/readback smoke for MotusCosmos.

This research smoke deliberately does not load a policy, create a network
client, construct an action, or call ``env.step``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OWNED_PACKAGE_ROOT = (REPO_ROOT / "handson").resolve()
EXPECTED_CONTROL_DT = 1.0 / 30.0
EXPECTED_PHYSICS_DT = 1.0 / 120.0
EXPECTED_SUBSTEPS = 4
ENVIRONMENT_ENTRY_KWARGS = {
    "policy_mode": True,
    "render_mode": "rgb_array",
    "randomize": True,
    "randomize_dynamics": False,
    "sim_backend": "isaacsim",
    "arm_type": "rokae",
    "hand_type": "wuji1",
    "control_dt": EXPECTED_CONTROL_DT,
    "physics_dt": EXPECTED_PHYSICS_DT,
}
OWNER_RUNTIME_PROFILE_KEYS = (
    "HANDSON_STABLE_RECORDING_ORIGIN",
    "HANDSON_ISAAC_IN_PROCESS_HARD_RESET",
    "HANDSON_ISAACLAB_ARM_IK_BACKEND",
    "HANDSON_ISAACLAB_ARM_TARGET_ALPHA",
    "HANDSON_ISAACLAB_ARM_MAX_SITE_STEP",
    "HANDSON_ISAACLAB_ARM_TARGET_JUMP_DELTA",
    "HANDSON_ISAACLAB_ARM_TARGET_POS_DEADBAND",
    "HANDSON_ISAACLAB_ARM_TARGET_ROT_JUMP_DEG",
    "HANDSON_ISAACLAB_ARM_TARGET_ROT_DEADBAND_DEG",
    "HANDSON_ISAACLAB_ARM_TARGET_JUMP_LIMIT",
    "HANDSON_ISAACLAB_ARM_MAX_JOINT_STEP",
    "HANDSON_ISAACLAB_DEFER_ARM_RENDER",
)


def _exact_float(value: object, expected: float, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not math.isclose(
        number, expected, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(f"{label} must be {expected!r}, got {number!r}")
    return number


def _module_path(module_name: str) -> str:
    module = sys.modules[module_name]
    path = Path(module.__file__).resolve()
    if OWNED_PACKAGE_ROOT not in path.parents:
        raise RuntimeError(f"{module_name} imported outside the owned repository: {path}")
    return str(path)


def _camera_report(cameras: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        role: {
            "shape_hwc": list(image.shape),
            "dtype": str(image.dtype),
            "color_space": "RGB",
            "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "min": int(np.min(image)),
            "max": int(np.max(image)),
            "std": float(np.std(image)),
            "nonconstant": bool(np.min(image) != np.max(image)),
        }
        for role, image in cameras.items()
    }


def _scalar_info(info: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in info.items():
        if isinstance(value, (bool, str, int)):
            result[str(key)] = value
        elif isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
            result[str(key)] = float(value)
    return result


def run(task: str, seed: int) -> dict[str, object]:
    from handson.motuscosmos_integration.actual_eef52 import (
        TASK_SPECS,
        camera_bundle_from_observation,
        measured_actual_eef52,
        validate_camera_bundle,
    )
    from handson.motuscosmos_integration.evaluation_reset import (
        reset_with_fresh_physx_and_stable_origin,
    )
    from handson.remote_env.contract import load_contract
    from handson.tasks.mappings import CONFIG_MAPPING

    if task not in TASK_SPECS:
        raise ValueError(f"unsupported task: {task!r}")
    spec = TASK_SPECS[task]
    config = CONFIG_MAPPING[task]()
    config_task_name = config._task_name()
    if config_task_name != spec.source_task:
        raise RuntimeError(
            f"config task mismatch: expected {spec.source_task!r}, "
            f"got {config_task_name!r}"
        )
    config_module = _module_path(type(config).__module__)
    env = None
    try:
        # Required by the owner evaluator before scene construction so the
        # collection-aligned stable-origin contact reporters are available.
        os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
        env = config.get_native_environment(
            **ENVIRONMENT_ENTRY_KWARGS,
            seed=seed,
        )
        raw = env.unwrapped
        if raw.task_name != task:
            raise RuntimeError(f"runtime task mismatch: {raw.task_name!r}")
        if type(raw).__name__ != spec.environment_class:
            raise RuntimeError(
                f"runtime class mismatch: expected {spec.environment_class}, "
                f"got {type(raw).__name__}"
            )
        environment_module = _module_path(type(raw).__module__)
        timing = {
            "control_dt_seconds": _exact_float(
                raw._control_dt, EXPECTED_CONTROL_DT, label="control_dt"
            ),
            "physics_dt_seconds": _exact_float(
                raw._physics_dt, EXPECTED_PHYSICS_DT, label="physics_dt"
            ),
            "physics_substeps_per_control": int(raw._n_substeps),
            "native_time_limit_seconds": _exact_float(
                raw._time_limit,
                spec.native_time_limit_seconds,
                label="native_time_limit",
            ),
        }
        if timing["physics_substeps_per_control"] != EXPECTED_SUBSTEPS:
            raise RuntimeError(
                f"n_substeps must be {EXPECTED_SUBSTEPS}, got {raw._n_substeps}"
            )

        target_label = spec.evaluation_target_labels[seed % len(spec.evaluation_target_labels)]
        env_step_before = int(raw.env_step)
        observation, info, initialization = (
            reset_with_fresh_physx_and_stable_origin(
                env,
                task=task,
                target_label=target_label,
                base_seed=seed,
            )
        )
        print(
            "MOTUSCOSMOS_PHASE4C_RESET_ACCEPTED "
            + json.dumps(
                {
                    "task": task,
                    "base_seed": seed,
                    "episode_reset_seed": initialization["episode_reset_seed"],
                    "fresh_scene_rebuilt": initialization["fresh_scene"].get(
                        "scene_rebuilt"
                    ),
                    "stable_exact_zero": initialization["stable_origin"].get(
                        "final_boundary_task_and_joint_velocity_exact_zero"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        actual_target_object_name = getattr(raw, "_target_object_name", None)
        env_step_after = int(raw.env_step)
        if env_step_before != 0 or env_step_after != 0:
            raise RuntimeError(
                f"reset-only smoke changed env_step: {env_step_before} -> {env_step_after}"
            )
        if not isinstance(info, Mapping) or "succeed" not in info:
            raise RuntimeError("reset info does not expose readable 'succeed'")
        success = bool(info["succeed"])
        done = bool(success or env_step_after * raw._control_dt >= raw._time_limit)

        contract = load_contract(REPO_ROOT / "contracts/handson_actual_eef52_v1.json")
        state = measured_actual_eef52(raw, contract)
        if state.dtype != np.float32 or state.shape != (52,) or not np.isfinite(state).all():
            raise RuntimeError("live measured state is not finite float32[52]")
        cameras = camera_bundle_from_observation(observation)
        validate_camera_bundle(cameras, contract)
        if any(int(np.max(image)) == 0 or float(np.std(image)) == 0.0 for image in cameras.values()):
            raise RuntimeError("live reset camera is all-black or constant")

        articulations = {}
        for side in ("right", "left"):
            robot = raw._control_robot(side)
            count = len(robot.joint_names)
            soft_limits = robot.data.soft_joint_pos_limits.detach().cpu().numpy()
            hard_limits = robot.root_physx_view.get_dof_limits()
            if hasattr(hard_limits, "detach"):
                hard_limits = hard_limits.detach().cpu().numpy()
            soft_limits = np.asarray(soft_limits, dtype=np.float64)
            hard_limits = np.asarray(hard_limits, dtype=np.float64)
            if soft_limits.shape == (1, count, 2):
                soft_limits = soft_limits[0]
            if hard_limits.shape == (1, count, 2):
                hard_limits = hard_limits[0]
            if soft_limits.shape != (count, 2) or hard_limits.shape != (count, 2):
                raise RuntimeError(
                    f"{side} joint limits have unexpected shapes: "
                    f"soft={soft_limits.shape}, hard={hard_limits.shape}"
                )
            for limit_kind, limits in (
                ("soft", soft_limits),
                ("hard", hard_limits),
            ):
                nonfinite = np.argwhere(~np.isfinite(limits))
                if nonfinite.size:
                    joint_index, bound_index = map(int, nonfinite[0])
                    raise RuntimeError(
                        f"{side} {limit_kind} joint limit is non-finite: "
                        f"joint_index={joint_index}, bound_index={bound_index}, "
                        f"value={limits[joint_index, bound_index]!r}"
                    )
                inverted = np.flatnonzero(limits[:, 0] > limits[:, 1])
                if inverted.size:
                    joint_index = int(inverted[0])
                    raise RuntimeError(
                        f"{side} {limit_kind} joint limit is inverted: "
                        f"joint_index={joint_index}, "
                        f"lower={limits[joint_index, 0]!r}, "
                        f"upper={limits[joint_index, 1]!r}"
                    )
            hand_ids = [int(index) for index in raw._hand_joint_ids[side]]
            policy_ids = [*map(int, raw._arm_joint_ids[side]), *hand_ids]
            policy_soft_limits = soft_limits[policy_ids]
            policy_hard_limits = hard_limits[policy_ids]
            articulations[side] = {
                "arm_joint_ids": [int(index) for index in raw._arm_joint_ids[side]],
                "hand_joint_ids_in_policy_order": hand_ids,
                "hand_joint_names_in_policy_order": [
                    str(robot.joint_names[index]) for index in hand_ids
                ],
                "eef_body_name": str(robot.body_names[int(raw._ee_body_indices[side])]),
                "policy_arm7_hand20_soft_limits_rad": policy_soft_limits.tolist(),
                "policy_arm7_hand20_hard_limits_rad": policy_hard_limits.tolist(),
                "soft_limits_sha256": hashlib.sha256(soft_limits.tobytes()).hexdigest(),
                "hard_limits_sha256": hashlib.sha256(hard_limits.tobytes()).hexdigest(),
            }

        report = {
            "format": "motuscosmos.handson.eef52.init_reset_smoke.v1",
            "task": task,
            "task_id": spec.task_id,
            "seed": seed,
            "evaluation_target_label": target_label,
            "actual_target_object_name_after_reset": (
                None
                if actual_target_object_name is None
                else str(actual_target_object_name)
            ),
            "evaluation_target_labels": list(spec.evaluation_target_labels),
            "evaluation_target_labels_are_not_complete_variations": True,
            "active_sides": list(spec.active_sides),
            "runtime": {
                "class": type(raw).__name__,
                "module_path": environment_module,
                "task_name": raw.task_name,
                "config": {
                    "class": type(config).__name__,
                    "module_path": config_module,
                    "resolved_task_name": config_task_name,
                },
                "timing": timing,
                "isaac_device": str(raw._isaac_device),
                "resolved_launch_isaac": bool(raw._launch_isaac),
                "resolved_headless": bool(raw._headless),
                "resolved_image_obs": bool(raw.image_obs),
                "environment_entry_kwargs": dict(ENVIRONMENT_ENTRY_KWARGS),
                "owner_runtime_profile": {
                    key: os.environ.get(key) for key in OWNER_RUNTIME_PROFILE_KEYS
                },
            },
            "reset": {
                "env_step_before": env_step_before,
                "env_step_after": env_step_after,
                "success": success,
                "done": done,
                "info_scalars": _scalar_info(info),
                "initialization": initialization,
            },
            "measured_actual_eef52": {
                "shape": list(state.shape),
                "dtype": str(state.dtype),
                "right_first": True,
                "finite": True,
                "sha256": hashlib.sha256(state.tobytes()).hexdigest(),
            },
            "articulations": articulations,
            "cameras": _camera_report(cameras),
            "negative_attestations": {
                "policy_loaded": False,
                "network_created": False,
                "action_constructed": False,
                "env_step_called": False,
            },
        }
        print(
            "MOTUSCOSMOS_PHASE4C_VALIDATION_COMPLETE "
            + json.dumps({"task": task, "seed": seed}, sort_keys=True),
            flush=True,
        )
        return report
    finally:
        # This smoke owns its Isaac process.  The owner evaluator/probe uses a
        # process boundary because SimulationApp.close() can block indefinitely.
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.task, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": "PASS", "task": args.task, "output": str(args.output)}))
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    os._exit(exit_code)
