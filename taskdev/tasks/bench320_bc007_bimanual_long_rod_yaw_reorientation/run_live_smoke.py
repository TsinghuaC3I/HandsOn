#!/usr/bin/env python3
"""Run the B320-BC007 collection-entry and multi-seed live PhysX smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from typing import Any, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "handson"))

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("HANDSON_ENABLE_ISAACSIM_NEWTON", "0")

from handson.sim.isaaclab.bc007_scene import (  # noqa: E402
    ROD_LENGTH_M,
    ROD_NAME,
    ROD_RADIUS_M,
    ROD_USD,
    ROD_USD_SHA256,
    TASK_ID,
    TASK_NAME,
    validate_asset_contract,
)
from handson.sim.isaaclab.priority_zero_robot_audit import (  # noqa: E402
    RobotStabilityAudit,
    initial_robot_state_report,
    priority_zero_runtime_contract,
    robot_snapshot,
)
from handson.tasks.mappings import CONFIG_MAPPING  # noqa: E402


DEFAULT_SEEDS = (7, 23, 61, 101, 149, 211, 307, 401)
DEFAULT_CONTROL_STEPS = 300
EXPECTED_PHYSICS_STEPS_PER_CONTROL = 4
TABLE_SUPPORT_FORCE_MIN_N = 0.01


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_CONTROL_STEPS)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _unwrap_core(env: Any) -> Any:
    current = env
    for _ in range(12):
        if getattr(current, "task_name", None) == TASK_NAME:
            return current
        child = getattr(current, "env", None)
        if child is None or child is current:
            break
        current = child
    raise RuntimeError("Could not unwrap the BC007 native environment")


def _neutral_hold_action(
    env: Any,
    core: Any,
    initial_robot_state: Mapping[str, Mapping[str, np.ndarray]],
) -> np.ndarray:
    """Build the no-motion action for the authored Wuji reset posture.

    The Cartesian zero fields mean "keep the synchronized physical TCP" in
    this runtime, but the hand fields are absolute joint targets.  An all-zero
    vector is therefore not neutral: Wuji1's canonical open posture has the
    thumb-base joint at 0.10 rad and an all-zero command drives it toward its
    lower limit.  Preserve the actual, already validated reset hand posture so
    the 10-second probe measures stability instead of commanding a transition.
    """

    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    flat = action.reshape(-1)
    hand_dim = int(core._hand_action_dim)
    expected_size = 14 + 2 * hand_dim
    if flat.size != expected_size:
        raise RuntimeError(
            "BC007 neutral hold requires the exact bimanual action layout "
            f"[{expected_size}], got [{flat.size}]"
        )
    for side, offset in (
        ("right", 14),
        ("left", 14 + hand_dim),
    ):
        hand = np.asarray(
            initial_robot_state[side]["hand_joint_position"],
            dtype=np.float64,
        ).reshape(-1)
        if hand.shape != (hand_dim,) or not np.all(np.isfinite(hand)):
            raise RuntimeError(
                f"BC007 {side} reset hand readback is not finite [{hand_dim}]"
            )
        flat[offset : offset + hand_dim] = hand.astype(flat.dtype, copy=False)
    return action


def _state(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    value = observation.get("state")
    if not isinstance(value, Mapping):
        raise TypeError("BC007 native observation has no state mapping")
    return value


def _camera_metrics(observation: Mapping[str, Any]) -> dict[str, Any]:
    images = observation.get("images")
    if not isinstance(images, Mapping):
        raise TypeError("BC007 collection requires live RGB images")
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


def _max_penetration(view: Any | None, dt: float) -> dict[str, Any]:
    if view is None:
        return {"available": False, "contact_count": 0, "max_penetration_m": None}
    _, _, _, separations, counts, _ = view.get_contact_data(float(dt))
    separations = np.asarray(separations.detach().cpu(), dtype=np.float64).reshape(-1)
    counts = np.asarray(counts.detach().cpu(), dtype=np.int64).reshape(-1)
    count = int(np.sum(counts))
    active = separations[:count] if count else np.zeros(0, dtype=np.float64)
    penetration = max(0.0, -float(np.min(active))) if active.size else 0.0
    return {
        "available": True,
        "contact_count": count,
        "max_penetration_m": penetration,
        "separations_m": active.tolist(),
    }


def _world_extent(core: Any) -> dict[str, Any]:
    from pxr import Usd, UsdGeom, UsdPhysics

    prim = core._find_prim(ROD_NAME)
    if prim is None or not prim.IsValid():
        raise RuntimeError("BC007 rod prim is missing after scene composition")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    world_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    low = np.asarray(world_range.GetMin(), dtype=np.float64)
    high = np.asarray(world_range.GetMax(), dtype=np.float64)
    collision_paths = [
        str(descendant.GetPath())
        for descendant in Usd.PrimRange(prim)
        if descendant.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(descendant).GetCollisionEnabledAttr().Get()
    ]
    rigid_paths = [
        str(descendant.GetPath())
        for descendant in Usd.PrimRange(prim)
        if descendant.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    extent = high - low
    dimensions = np.sort(extent)
    return {
        "aabb_m": [low.tolist(), high.tolist()],
        "extent_m": extent.tolist(),
        "collision_paths": collision_paths,
        "rigid_body_paths": rigid_paths,
        "pass": bool(
            len(collision_paths) == 1
            and len(rigid_paths) == 1
            and abs(float(dimensions[2]) - ROD_LENGTH_M) <= 0.004
            and abs(float(dimensions[0]) - 2.0 * ROD_RADIUS_M) <= 0.004
            and abs(float(dimensions[1]) - 2.0 * ROD_RADIUS_M) <= 0.004
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "live_smoke_summary.json"
    started = time.time()
    summary: dict[str, Any] = {
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "host": os.environ.get("TASKDEV_HOST", ""),
        "observed_hostname": socket.gethostname(),
        "physical_gpu": os.environ.get("TASKDEV_PHYSICAL_GPU", ""),
        "device": args.device,
        "seeds": [int(seed) for seed in args.seeds],
        "pass": False,
    }
    env = None
    try:
        asset_errors = validate_asset_contract()
        if asset_errors:
            raise RuntimeError("; ".join(asset_errors))
        config_class = CONFIG_MAPPING[TASK_NAME]
        config = config_class()
        env = config.get_environment(
            policy_mode=True,
            render_mode="rgb_array",
            randomize=True,
            sim_backend="isaacsim",
            seed=int(args.seeds[0]),
            image_obs=True,
            randomize_dynamics=False,
            hand_type="wuji1",
            arm_type="rokae",
            launch_isaac=True,
            headless=True,
            isaac_device=args.device,
            physics_backend="physx",
        )
        core = _unwrap_core(env)
        core._fast_shutdown_on_close = True
        runtime_contract = priority_zero_runtime_contract(
            core,
            requested_device=args.device,
            physics_hz=120,
            control_hz=30,
        )
        summary["runtime"] = runtime_contract
        contact_views = {
            "left_hand": core._bc007_hand_contact_views.get("left"),
            "right_hand": core._bc007_hand_contact_views.get("right"),
            "table": core._bc007_table_contact_view,
        }
        if not all(view is not None for view in contact_views.values()):
            raise RuntimeError(
                "BC007 exact contact views are unavailable: "
                + ",".join(name for name, view in contact_views.items() if view is None)
            )
        physics_steps_per_control = int(
            round(float(core._control_dt) / float(core._physics_dt))
        )
        if physics_steps_per_control != EXPECTED_PHYSICS_STEPS_PER_CONTROL:
            raise RuntimeError(
                "BC007 requires exactly four 120 Hz physics steps per 30 Hz "
                f"control step, got {physics_steps_per_control}"
            )

        reset_records = []
        first_seed_pose = None
        camera = None
        extent = None
        for index, seed in enumerate(args.seeds):
            _, reset_info = env.reset(seed=int(seed))
            if not core._wake_rod_after_reset():
                raise RuntimeError(f"BC007 seed {seed} could not wake the rod")
            observation = core._compute_observation()
            state = _state(observation)
            initial_pose = np.asarray(state["rod_pose"], dtype=np.float64).copy()
            initial_robot_pass, initial_robot_report = (
                initial_robot_state_report(core)
            )
            robot_audit = RobotStabilityAudit.start(core)
            initial_robot_state = robot_audit.reference
            if index == 0:
                first_seed_pose = initial_pose.copy()
                camera = _camera_metrics(observation)
                extent = _world_extent(core)

            maximum_table_penetration = 0.0
            maximum_hand_penetration = 0.0
            maximum_table_support_force = 0.0
            action = _neutral_hold_action(env, core, initial_robot_state)
            terminal = False
            truncated = False
            final_info: Mapping[str, Any] = reset_info
            executed_control_steps = 0
            if core._bc007_per_physics_robot_audit is not None:
                raise RuntimeError(
                    "BC007 per-physics robot audit is already installed"
                )
            core._bc007_per_physics_robot_audit = robot_audit
            try:
                for _ in range(max(1, int(args.settle_steps))):
                    _, _, terminal, truncated, final_info = env.step(action)
                    executed_control_steps += 1
                    support_force, support_available = core._contact_force(
                        contact_views["table"]
                    )
                    if not support_available:
                        raise RuntimeError(
                            f"BC007 seed {seed} lost rod/table support-force readback"
                        )
                    maximum_table_support_force = max(
                        maximum_table_support_force, float(support_force)
                    )
                    table_contact = _max_penetration(
                        contact_views["table"], core._physics_dt
                    )
                    maximum_table_penetration = max(
                        maximum_table_penetration,
                        float(table_contact["max_penetration_m"] or 0.0),
                    )
                    for name in ("left_hand", "right_hand"):
                        hand_contact = _max_penetration(
                            contact_views[name], core._physics_dt
                        )
                        maximum_hand_penetration = max(
                            maximum_hand_penetration,
                            float(hand_contact["max_penetration_m"] or 0.0),
                        )
                    if terminal or truncated:
                        break
            finally:
                core._bc007_per_physics_robot_audit = None

            final_observation = core._compute_observation()
            final_state = _state(final_observation)
            final_pose = np.asarray(final_state["rod_pose"], dtype=np.float64)
            final_robot_state = robot_snapshot(core)
            metrics = core._bc007_last_metrics or core._bc007_metrics(update=False)
            endpoints = np.asarray(
                final_state["rod_endpoints"], dtype=np.float64
            ).reshape(2, 3)
            physics_steps = executed_control_steps * physics_steps_per_control
            sampled_physics_steps = int(robot_audit.samples)
            robot_stability = robot_audit.to_dict()
            robot_stability_pass = robot_audit.passed(
                physics_steps=physics_steps,
                physics_hz=120,
            )
            record = {
                "seed": int(seed),
                "initial_pose": initial_pose.tolist(),
                "final_pose": final_pose.tolist(),
                "center_drift_m": float(
                    np.linalg.norm(final_pose[:3] - initial_pose[:3])
                ),
                "control_steps": int(executed_control_steps),
                "physics_steps": int(physics_steps),
                "sampled_physics_steps": sampled_physics_steps,
                "per_physics_sample_count_matches": bool(
                    sampled_physics_steps == physics_steps
                ),
                "maximum_table_penetration_m": maximum_table_penetration,
                "maximum_hand_penetration_m": maximum_hand_penetration,
                "maximum_table_support_force_n": maximum_table_support_force,
                "robot_initial_state_report": _json_safe(initial_robot_report),
                "robot_initial_state_pass": bool(initial_robot_pass),
                "robot_initial_actual_state": _json_safe(initial_robot_state),
                "neutral_hold_action": _json_safe(action),
                "robot_final_actual_state": _json_safe(final_robot_state),
                "robot_stability": _json_safe(robot_stability),
                "robot_stability_pass": bool(robot_stability_pass),
                "initial_success": bool(reset_info.get("succeed", False)),
                "final_success": bool(final_info.get("succeed", False)),
                "terminal": bool(terminal),
                "truncated": bool(truncated),
                "failed": bool(metrics["failed"]),
                "failure_reason": str(metrics["failure_reason"]),
                "endpoints": endpoints.tolist(),
                "finite": bool(
                    np.isfinite(final_pose).all()
                    and np.isfinite(endpoints).all()
                    and all(
                        np.isfinite(value).all()
                        for side_state in final_robot_state.values()
                        for value in side_state.values()
                    )
                ),
                "inside_workspace": bool(metrics["finite_and_in_workspace"]),
                "required_contact_evidence_available": bool(
                    metrics["required_contact_evidence_available"]
                ),
            }
            record["pass"] = bool(
                record["finite"]
                and record["inside_workspace"]
                and not record["initial_success"]
                and not record["final_success"]
                and not record["terminal"]
                and not record["truncated"]
                and not record["failed"]
                and record["required_contact_evidence_available"]
                and record["control_steps"] >= DEFAULT_CONTROL_STEPS
                and record["physics_steps"] >= 1200
                and record["per_physics_sample_count_matches"]
                and record["robot_initial_state_pass"]
                and record["robot_stability_pass"]
                and maximum_table_support_force >= TABLE_SUPPORT_FORCE_MIN_N
                and record["center_drift_m"] <= 0.005
                and maximum_table_penetration <= 0.005
                and maximum_hand_penetration <= 0.005
            )
            reset_records.append(record)

        _, same_seed_info = env.reset(seed=int(args.seeds[0]))
        same_seed_pose = np.asarray(
            _state(core._compute_observation())["rod_pose"], dtype=np.float64
        )
        deterministic_error = float(np.max(np.abs(same_seed_pose - first_seed_pose)))
        position_samples = np.asarray([record["initial_pose"][:3] for record in reset_records])
        randomized_span = np.ptp(position_samples[:, :2], axis=0)
        summary.update(
            {
                "asset": {
                    "path": str(ROD_USD.relative_to(REPOSITORY_ROOT)),
                    "sha256": _sha256(ROD_USD),
                    "expected_sha256": ROD_USD_SHA256,
                    "pass": _sha256(ROD_USD) == ROD_USD_SHA256,
                },
                "config_load": {
                    "config_class": f"{config_class.__module__}.{config_class.__name__}",
                    "action_shape": list(env.action_space.shape),
                    "same_seed_reset_info": _json_safe(dict(same_seed_info)),
                    "pass": True,
                },
                "rod_geometry": extent,
                "contact_views": {
                    name: {
                        "sensor_count": int(view.sensor_count),
                        "filter_count": int(view.filter_count),
                        "max_contact_data_count": int(view.max_contact_data_count),
                    }
                    for name, view in contact_views.items()
                },
                "neutral_hold_contract": {
                    "requested_control_steps": int(args.settle_steps),
                    "physics_steps_per_control": physics_steps_per_control,
                    "required_minimum_control_steps": DEFAULT_CONTROL_STEPS,
                    "required_minimum_physics_steps": 1200,
                    "physics_dt_s": float(core._physics_dt),
                    "control_dt_s": float(core._control_dt),
                    "robot_audit": "priority_zero_robot_audit.RobotStabilityAudit",
                    "tcp_fields": "zero_keep_synchronized_physical_tcp",
                    "hand_fields": "absolute_actual_reset_joint_positions",
                    "robot_tolerances": reset_records[0]["robot_stability"][
                        "thresholds"
                    ],
                    "table_support_force_min_n": TABLE_SUPPORT_FORCE_MIN_N,
                },
                "camera": camera,
                "reset_records": reset_records,
                "same_seed_max_pose_error": deterministic_error,
                "randomized_xy_span_m": randomized_span.tolist(),
                "elapsed_seconds": time.time() - started,
            }
        )
        summary["pass"] = bool(
            summary["asset"]["pass"]
            and extent["pass"]
            and all(metric["pass"] for metric in camera.values())
            and len(reset_records) >= 8
            and all(record["pass"] for record in reset_records)
            and deterministic_error <= 1.0e-9
            and bool(np.all(randomized_span > 0.010))
        )
    except Exception as exc:
        summary.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.time() - started,
            }
        )
    finally:
        _write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)
        # RTX cleanup can hang after camera capture in this production stack.
        # This executable owns a scheduler-isolated process, so persist and
        # flush all evidence before terminating with the truthful gate code.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if summary.get("pass") else 1)
    return 0 if summary.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
