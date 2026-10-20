#!/usr/bin/env python3
"""Live G08 right-hand/lid contact and penetration audit."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from validate_lw_native_articulation_live import (
    _git_metadata,
    _json_ready,
    _reset_and_accept_origin,
)


TASK_NAME = "t18_open_micropipette_box"
TASK_MODULE = "handson.tasks.t18_open_micropipette_box.config"
EXPECTED_GPU_BOX_COLLIDERS = {
    "MicropipetteBox001_Collider1",
    "MicropipetteBox001_Collider14",
    "MicropipetteBox001_Collider22",
    "MicropipetteBox001_Collider23",
    "MicropipetteBox001_Lid001_Collider4",
    "MicropipetteBox001_Lid001_Collider6",
    "MicropipetteBox001_Lid001_Collider7",
    "MicropipetteBox001_Lid001_Collider9",
}
CONTACT_FORCE_MIN_N = 0.15
CONTACT_FORCE_MAX_N = 40.0
PENETRATION_LIMIT_M = 0.005
HOLD_CONTROL_STEPS = 30


def _numpy(value: Any, *, dtype: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _right_contact_sample(raw: Any) -> dict[str, Any]:
    view = raw._lw_contact_views["right"]
    matrix = _numpy(
        view.get_contact_force_matrix(raw._sim_dt),
        dtype=np.float64,
    )
    if matrix.size == 0 or not np.all(np.isfinite(matrix)):
        raise RuntimeError("G08 right-hand contact force tensor is unreadable")
    force_n = float(np.linalg.norm(matrix, axis=-1).sum())
    data = view.get_contact_data(raw._sim_dt)
    if not isinstance(data, tuple) or len(data) < 6:
        raise RuntimeError("G08 right-hand contact detail tensor is incomplete")
    separations = _numpy(data[3], dtype=np.float64).reshape(-1)
    counts = _numpy(data[4], dtype=np.int64).reshape(-1)
    starts = _numpy(data[5], dtype=np.int64).reshape(-1)
    if counts.shape != starts.shape:
        raise RuntimeError("G08 contact count/start tensor shapes changed")
    active: list[np.ndarray] = []
    for count, start in zip(counts, starts, strict=True):
        count = int(count)
        start = int(start)
        end = start + count
        if count < 0 or start < 0 or end > separations.size:
            raise RuntimeError("G08 contact detail tensor contains invalid ranges")
        if count:
            active.append(separations[start:end])
    contact_count = int(sum(values.size for values in active))
    min_separation_m = (
        float(min(float(np.min(values)) for values in active))
        if active
        else None
    )
    if min_separation_m is not None and not math.isfinite(min_separation_m):
        raise RuntimeError("G08 contact separation is non-finite")
    return {
        "force_n": force_n,
        "contact_count": contact_count,
        "min_separation_m": min_separation_m,
        "penetration_m": max(0.0, -(min_separation_m or 0.0)),
    }


def _step_pose(raw: Any, pose_wxyz: np.ndarray) -> dict[str, Any]:
    right = np.concatenate(
        (
            np.asarray(pose_wxyz, dtype=np.float64).reshape(7),
            np.asarray(raw._right_hand, dtype=np.float64).reshape(-1),
        )
    )
    result = raw.step({"right": right})
    if not isinstance(result, tuple) or len(result) < 5:
        raise RuntimeError(f"G08 contact probe returned invalid step: {result!r}")
    if bool(result[2]) or bool(result[3]):
        raise RuntimeError(
            "G08 contact probe terminated before the hold completed: "
            f"terminated={bool(result[2])}, truncated={bool(result[3])}, "
            f"info={result[4]!r}"
        )
    if raw._lw_state.failure_reason:
        raise RuntimeError(
            "G08 contact probe latched task failure: "
            f"{raw._lw_state.failure_reason}"
        )
    return _right_contact_sample(raw)


def _actual_right_palm_pose_wxyz(raw: Any) -> np.ndarray:
    from handson.sim.isaaclab.bimanual_assembly_env import (
        _matrix_to_quat_wxyz,
    )

    matrix = np.asarray(
        raw.get_end_effector_pose_matrix()[0],
        dtype=np.float64,
    ).reshape(4, 4)
    return np.concatenate(
        (matrix[:3, 3], _matrix_to_quat_wxyz(matrix[:3, :3]))
    ).astype(np.float64)


def _actual_right_palm_xyz(raw: Any) -> list[float]:
    return [
        float(value) for value in _actual_right_palm_pose_wxyz(raw)[:3]
    ]


def _gpu_collider_audit(raw: Any) -> dict[str, str]:
    from pxr import Usd, UsdPhysics

    found: dict[str, str] = {}
    root = raw._stage.GetPrimAtPath(raw._lw_scene_spec.asset_path)
    for prim in Usd.PrimRange(root):
        name = prim.GetName()
        if name not in EXPECTED_GPU_BOX_COLLIDERS:
            continue
        api = UsdPhysics.MeshCollisionAPI(prim)
        found[name] = str(api.GetApproximationAttr().Get()) if api else ""
    missing = sorted(EXPECTED_GPU_BOX_COLLIDERS - found.keys())
    wrong = {
        name: value
        for name, value in found.items()
        if value != str(UsdPhysics.Tokens.boundingCube)
    }
    if missing or wrong:
        raise RuntimeError(
            "G08 composed GPU collider audit failed: "
            f"missing={missing}, wrong={wrong}"
        )
    return dict(sorted(found.items()))


def _run_contact_probe(raw: Any) -> dict[str, Any]:
    sample = raw._lw_reset_sample
    production_position_only = raw._use_position_only_differential_ik
    raw._use_position_only_differential_ik = lambda side: bool(
        side == "right" or production_position_only(side)
    )
    # Preserve the physical reset orientation.  Reusing the generic command
    # buffer's nominal quaternion made DLS spend its authority rotating the
    # wrist and prevented the palm from reaching the lid position.
    initial_palm_pose = _actual_right_palm_pose_wxyz(raw)
    orientation_wxyz = initial_palm_pose[3:7]
    # Put the palm over the right half of the broad top panel.  The production
    # grip target is farther out at local y=-0.16 m so the fingers can curl
    # around the free edge; the contact audit intentionally stays 60 mm
    # inboard to strike the panel itself rather than relying on a fingertip.
    center_xy = np.asarray(
        [sample.root_x_m, sample.root_y_m - 0.05],
        dtype=np.float64,
    )
    approach = np.concatenate((center_xy, [1.085], orientation_wxyz))

    # Let the shared IK/pose filter reach the broad center of the closed lid
    # from the normal right-hand home pose without a discontinuous arm target.
    approach_samples: list[dict[str, Any]] = []
    approach_reached = False
    for _ in range(360):
        contact = _step_pose(raw, approach)
        actual_xyz = np.asarray(_actual_right_palm_xyz(raw), dtype=np.float64)
        contact["actual_palm_xyz_m"] = actual_xyz.tolist()
        contact["position_error_m"] = float(
            np.linalg.norm(actual_xyz - approach[:3])
        )
        approach_samples.append(contact)
        if contact["position_error_m"] <= 0.025:
            approach_reached = True
            break
    # The DLS controller can retain several centimetres of tangential error
    # at this workspace edge while still descending normally.  Keep the
    # measured convergence result as evidence, but let the contact tensor --
    # not an IK-position heuristic -- decide whether the physical probe lands.

    samples: list[dict[str, Any]] = []
    contact_pose = approach.copy()
    contact_found = False
    for target_z_m in np.arange(1.080, 0.979, -0.002):
        contact_pose = approach.copy()
        contact_pose[2] = float(target_z_m)
        for _ in range(6):
            contact = _step_pose(raw, contact_pose)
            contact["command_z_m"] = float(target_z_m)
            contact["actual_palm_xyz_m"] = _actual_right_palm_xyz(raw)
            samples.append(contact)
            if (
                contact["contact_count"] > 0
                and contact["force_n"] >= CONTACT_FORCE_MIN_N
            ):
                contact_found = True
                break
        if contact_found:
            break
    if not contact_found:
        raise RuntimeError(
            "G08 right palm reached the audited descent limit without a "
            "physical lid contact; "
            f"first_sample={samples[0] if samples else None!r}, "
            f"last_sample={samples[-1] if samples else None!r}"
        )

    actual_z_m = float(_actual_right_palm_xyz(raw)[2])
    hold_pose = contact_pose.copy()
    hold_pose[2] = actual_z_m - 0.0005
    consecutive_contact_steps = 0
    hold_samples: list[dict[str, Any]] = []
    for _ in range(120):
        contact = _step_pose(raw, hold_pose)
        if contact["force_n"] > CONTACT_FORCE_MAX_N:
            hold_pose[2] += 0.0005
        elif contact["force_n"] < CONTACT_FORCE_MIN_N:
            hold_pose[2] -= 0.00025
        contact["command_z_m"] = float(hold_pose[2])
        contact["actual_palm_xyz_m"] = _actual_right_palm_xyz(raw)
        hold_samples.append(contact)
        if (
            contact["contact_count"] > 0
            and contact["force_n"] >= CONTACT_FORCE_MIN_N
        ):
            consecutive_contact_steps += 1
        else:
            consecutive_contact_steps = 0
        if consecutive_contact_steps >= HOLD_CONTROL_STEPS:
            break
    if consecutive_contact_steps < HOLD_CONTROL_STEPS:
        raise RuntimeError(
            "G08 could not maintain one continuous second of physical "
            f"right-hand/lid contact; consecutive={consecutive_contact_steps}"
        )

    all_contacts = [
        item
        for item in samples + hold_samples
        if item["contact_count"] > 0
    ]
    maximum_penetration_m = max(
        float(item["penetration_m"]) for item in all_contacts
    )
    if maximum_penetration_m > PENETRATION_LIMIT_M:
        raise RuntimeError(
            "G08 right-hand/lid penetration exceeded the task contract: "
            f"{maximum_penetration_m:.9g} m > {PENETRATION_LIMIT_M:.9g} m"
        )

    retract = approach.copy()
    for _ in range(45):
        _step_pose(raw, retract)
    return {
        "approach_samples": approach_samples,
        "approach_control_steps": len(approach_samples),
        "approach_converged": approach_reached,
        "probe_arm_control": "right_position_only_differential_ik",
        "initial_palm_pose_wxyz": initial_palm_pose,
        "descent_samples": samples,
        "hold_samples": hold_samples,
        "continuous_contact_control_steps": consecutive_contact_steps,
        "continuous_contact_seconds": (
            consecutive_contact_steps * float(raw._control_dt)
        ),
        "maximum_penetration_m": maximum_penetration_m,
        "penetration_limit_m": PENETRATION_LIMIT_M,
        "maximum_force_n": max(float(item["force_n"]) for item in all_contacts),
        "final_lid_truth": {
            "q_rad": raw._read_lw_truth()["q"],
            "qd_rad_s": raw._read_lw_truth()["qd"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
    os.environ["HANDSON_STABLE_RECORDING_ORIGIN"] = "1"

    repository_root = Path(__file__).resolve().parents[1]
    evidence: dict[str, Any] = {
        "schema": "handson.g08_lid_contact_live.v1",
        "task_name": TASK_NAME,
        "seed": int(args.seed),
        "status": "running",
        "git": _git_metadata(repository_root),
    }
    env = None
    try:
        import importlib

        module = importlib.import_module(TASK_MODULE)
        env = module.TaskConfig().get_native_environment(
            policy_mode=True,
            render_mode="rgb_array",
            randomize=True,
            sim_backend="isaacsim",
            physics_backend="physx",
            launch_isaac=True,
            headless=True,
            image_obs=False,
            seed=args.seed,
            isaac_device="cuda:0",
        )
        raw = env.unwrapped
        evidence["gpu_collider_approximations"] = _gpu_collider_audit(raw)
        evidence["recording_origin"] = _reset_and_accept_origin(
            env, raw, args.seed
        )
        evidence["contact_probe"] = _run_contact_probe(raw)
        evidence["status"] = "pass"
        print(f"G08_LID_CONTACT_PASS {args.output}", flush=True)
        return 0
    except Exception as exc:
        evidence["status"] = "fail"
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        evidence["traceback"] = traceback.format_exc()
        print(f"G08_LID_CONTACT_FAIL {args.output}: {exc}", flush=True)
        return 1
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                _json_ready(evidence),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
