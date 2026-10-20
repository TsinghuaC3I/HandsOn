#!/usr/bin/env python3
"""Replay Isaac recordings with a MuJoCo-style deterministic control contract.

The runtime remains Isaac/PhysX.  This script restores the complete recorded
state once at frame zero, then moves the robot along its recorded joint path
at every physics substep. Dynamic task objects are never written back after
reset, so contacts and task motion are solved again by PhysX.

This deliberately does not fall back to per-frame Cartesian DLS.  Such a
fallback can look plausible while silently selecting a different IK branch
and is not a valid physical replay certification.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr


MODE_NAME = "isaac_mujoco_style_replay"
ISAAC_PYTHON = Path(
    "/home/user/下载/handson/.handson_isaacsim/toolchain/env/bin/python"
)
REPLAY_SCRIPT = Path(__file__).with_name("replay_demos_zarr.py")

_REQUIRED_CONTROL_FIELDS = (
    "physx_sim_time",
    "next_physx_sim_time",
    "physx_substep_sim_time",
    "next_physx_substep_sim_time",
    "physx_robot_joint_position",
    "next_physx_robot_joint_position",
    "physx_robot_joint_velocity",
    "next_physx_robot_joint_velocity",
    "physx_robot_joint_valid",
    "next_physx_robot_joint_valid",
    "physx_robot_joint_target",
    "next_physx_robot_joint_target",
    "physx_robot_joint_target_valid",
    "next_physx_robot_joint_target_valid",
    "physx_substep_count",
    "next_physx_substep_count",
    "physx_substep_robot_joint_position",
    "next_physx_substep_robot_joint_position",
    "physx_substep_robot_joint_velocity",
    "next_physx_substep_robot_joint_velocity",
    "physx_substep_robot_joint_valid",
    "next_physx_substep_robot_joint_valid",
    "physx_substep_robot_joint_target",
    "next_physx_substep_robot_joint_target",
    "physx_substep_robot_joint_target_valid",
    "next_physx_substep_robot_joint_target_valid",
)


def _task_name(episode: Path) -> str:
    match = re.match(r"(.+)_demo_\d+_", episode.name)
    if not match:
        raise ValueError(f"cannot infer task name from episode: {episode.name}")
    return match.group(1)


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _validate_contract(episode: Path) -> dict[str, Any]:
    replay_path = episode / "replay.zarr"
    schema_path = episode / "recording_schema.json"
    if not replay_path.is_dir():
        raise FileNotFoundError(replay_path)
    if not schema_path.is_file():
        raise RuntimeError(
            "recording_schema.json is required for named Isaac state restore"
        )

    root = zarr.open_group(str(replay_path), mode="r")
    if "data" not in root:
        raise RuntimeError(f"{replay_path} has no data group")
    data = root["data"]
    missing = [name for name in _REQUIRED_CONTROL_FIELDS if name not in data]
    if missing:
        raise RuntimeError(
            "episode cannot use deterministic joint-target replay; missing "
            + ", ".join(missing)
        )

    target = data["next_physx_robot_joint_target"]
    target_valid = data["next_physx_robot_joint_target_valid"]
    if target.ndim != 3 or target_valid.shape != target.shape:
        raise RuntimeError(
            "invalid PhysX joint-target layout: "
            f"target={target.shape}, valid={target_valid.shape}"
        )
    if target.shape[0] == 0:
        raise RuntimeError("recorded joint-target trajectory is empty")

    valid = np.asarray(target_valid[:], dtype=bool)
    finite = np.isfinite(np.asarray(target[:]))
    usable = valid & finite
    per_frame = np.any(usable, axis=(1, 2))
    if not bool(np.all(per_frame)):
        invalid_frames = np.flatnonzero(~per_frame)
        preview = ", ".join(str(int(index)) for index in invalid_frames[:12])
        raise RuntimeError(
            "joint-controller targets are missing at recorded frames: "
            f"{preview}"
        )

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    environment_schema = (
        schema.get("environment_schema", {}) if isinstance(schema, dict) else {}
    )
    if int(environment_schema.get("version", 0)) < 4:
        raise RuntimeError(
            "recording schema predates exact robot substep capture "
            "(required version >= 4)"
        )
    physx_schema = (
        environment_schema.get("physx", {})
    )
    robot_sides = tuple(str(side) for side in physx_schema.get("robot_sides", ()))
    if not robot_sides:
        raise RuntimeError("recording schema does not declare PhysX robot sides")

    substep_position = np.asarray(
        data["next_physx_substep_robot_joint_position"][:]
    )
    substep_velocity = np.asarray(
        data["next_physx_substep_robot_joint_velocity"][:]
    )
    substep_valid = np.asarray(
        data["next_physx_substep_robot_joint_valid"][:],
        dtype=bool,
    )
    substep_target = np.asarray(
        data["next_physx_substep_robot_joint_target"][:]
    )
    substep_target_valid = np.asarray(
        data["next_physx_substep_robot_joint_target_valid"][:],
        dtype=bool,
    )
    substep_counts = np.asarray(
        data["next_physx_substep_count"][:],
        dtype=np.int64,
    ).reshape(-1)
    control_time = np.asarray(data["physx_sim_time"][:], dtype=np.float64)
    next_control_time = np.asarray(
        data["next_physx_sim_time"][:],
        dtype=np.float64,
    )
    substep_time = np.asarray(
        data["next_physx_substep_sim_time"][:],
        dtype=np.float64,
    )
    expected_substeps = int(
        physx_schema.get("physics_config", {}).get("n_substeps", 0)
    )
    physics_dt = float(
        physx_schema.get("physics_config", {}).get("physics_dt", 0.0)
    )
    expected_shape = (
        int(target.shape[0]),
        expected_substeps,
        int(target.shape[1]),
        int(target.shape[2]),
    )
    if expected_substeps <= 0 or substep_position.shape != expected_shape:
        raise RuntimeError(
            "invalid exact robot substep position layout: "
            f"expected={expected_shape}, actual={substep_position.shape}"
        )
    for name, values in (
        ("velocity", substep_velocity),
        ("valid", substep_valid),
        ("target", substep_target),
        ("target_valid", substep_target_valid),
    ):
        if values.shape != expected_shape:
            raise RuntimeError(
                f"invalid exact robot substep {name} layout: "
                f"expected={expected_shape}, actual={values.shape}"
            )
    if substep_counts.shape != (target.shape[0],) or not bool(
        np.all(substep_counts == expected_substeps)
    ):
        raise RuntimeError(
            "recorded PhysX substep counts are incomplete or differ from "
            f"schema n_substeps={expected_substeps}"
        )
    expected_time_shape = (target.shape[0], expected_substeps, 1)
    if (
        physics_dt <= 0.0
        or control_time.shape != (target.shape[0], 1)
        or next_control_time.shape != (target.shape[0], 1)
        or substep_time.shape != expected_time_shape
        or not bool(np.all(np.isfinite(substep_time)))
    ):
        raise RuntimeError(
            "invalid PhysX timing contract: "
            f"physics_dt={physics_dt}, control={control_time.shape}, "
            f"next={next_control_time.shape}, "
            f"substeps={substep_time.shape}"
        )
    timing_tolerance = max(1e-9, physics_dt * 1e-5)
    if not bool(
        np.allclose(
            substep_time[:, 0, 0],
            control_time[:, 0] + physics_dt,
            atol=timing_tolerance,
            rtol=0.0,
        )
    ):
        raise RuntimeError(
            "first recorded PhysX substep is not one physics_dt after the "
            "control-frame pre-state"
        )
    if expected_substeps > 1 and not bool(
        np.allclose(
            np.diff(substep_time[:, :, 0], axis=1),
            physics_dt,
            atol=timing_tolerance,
            rtol=0.0,
        )
    ):
        raise RuntimeError(
            "recorded PhysX substep timestamps are not uniformly spaced"
        )
    if not bool(
        np.allclose(
            substep_time[:, -1, 0],
            next_control_time[:, 0],
            atol=timing_tolerance,
            rtol=0.0,
        )
    ):
        raise RuntimeError(
            "last PhysX substep timestamp does not match the control-frame "
            "post-state"
        )
    next_position = np.asarray(
        data["next_physx_robot_joint_position"][:],
        dtype=np.float64,
    )
    next_velocity = np.asarray(
        data["next_physx_robot_joint_velocity"][:],
        dtype=np.float64,
    )
    if not bool(
        np.allclose(
            substep_position[:, -1],
            next_position,
            atol=1e-8,
            rtol=0.0,
        )
        and np.allclose(
            substep_velocity[:, -1],
            next_velocity,
            atol=1e-8,
            rtol=0.0,
        )
    ):
        raise RuntimeError(
            "last PhysX robot substep does not match the control-frame "
            "post-state"
        )
    required_joint_mask = valid[:, None, :, :]
    usable_substep_state = (
        substep_valid
        & np.isfinite(substep_position)
        & np.isfinite(substep_velocity)
    )
    usable_substep_target = (
        substep_target_valid & np.isfinite(substep_target)
    )
    if not bool(np.all(usable_substep_state | ~required_joint_mask)):
        raise RuntimeError(
            "one or more recorded robot joints lack a finite state at a "
            "PhysX substep"
        )
    if not bool(np.all(usable_substep_target | ~required_joint_mask)):
        raise RuntimeError(
            "one or more recorded robot joints lack a finite target at a "
            "PhysX substep"
        )

    action_steps = int(data["action"].shape[0]) if "action" in data else None
    if action_steps is not None and action_steps != int(target.shape[0]):
        raise RuntimeError(
            "action and joint-target frame counts differ: "
            f"{action_steps} != {target.shape[0]}"
        )

    return {
        "frame_count": int(target.shape[0]),
        "robot_sides": list(robot_sides),
        "joint_slots": int(target.shape[2]),
        "valid_target_values": int(np.count_nonzero(usable)),
        "physics_substeps_per_control_step": expected_substeps,
        "exact_substep_state_values": int(
            np.count_nonzero(usable_substep_state)
        ),
        "initial_state_fields": [
            "physx_robot_joint_position",
            "physx_robot_joint_velocity",
        ],
        "control_fields": [
            "next_physx_substep_robot_joint_position",
            "next_physx_substep_robot_joint_velocity",
            "next_physx_substep_robot_joint_target",
        ],
    }


def _make_isolated_input(episode: Path) -> tuple[Path, Path]:
    work_root = Path(
        tempfile.mkdtemp(
            prefix=f".{MODE_NAME}-",
            dir=str(episode.parent),
        )
    )
    work_episode = work_root / episode.name
    shutil.copytree(episode, work_episode, copy_function=_link_or_copy)
    return work_root, work_episode


def _command(
    *,
    work_root: Path,
    output_dir: Path,
    task_name: str,
) -> list[str]:
    return [
        str(ISAAC_PYTHON),
        str(REPLAY_SCRIPT),
        f"--input_dir={work_root}",
        f"--out_dir={output_dir}",
        f"--exp_name={task_name}",
        "--sim_backend=isaacsim",
        "--randomize=False",
        "--restore_state=True",
        "--save_failed=True",
        "--state_guided=False",
        "--kinematic_visual_replay=False",
        "--physx_ground_truth_replay=False",
        "--physx_action_replay_audit=False",
        "--physx_joint_target_replay_audit=False",
        "--physx_robot_path_replay_audit=True",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Restore one Isaac state, then physically re-execute recorded "
            "joint targets without per-frame object writeback."
        )
    )
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    episode = args.episode.expanduser().resolve()
    if not episode.is_dir():
        raise FileNotFoundError(episode)
    if not ISAAC_PYTHON.is_file():
        raise FileNotFoundError(ISAAC_PYTHON)
    if not REPLAY_SCRIPT.is_file():
        raise FileNotFoundError(REPLAY_SCRIPT)

    contract = _validate_contract(episode)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else episode.parent / f"{MODE_NAME}_output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root, _ = _make_isolated_input(episode)

    command = _command(
        work_root=work_root,
        output_dir=output_dir,
        task_name=_task_name(episode),
    )
    child_env = dict(os.environ)
    child_env.update(
        {
            "HANDSON_PHYSX_RESET_PREROLL_STEPS": "0",
            "HANDSON_STABLE_RECORDING_ORIGIN": "0",
            "HANDSON_GLOBAL_IK_PLAN": "0",
            "PYTHONHASHSEED": "0",
        }
    )

    print(
        f"[{MODE_NAME}] frame0 state restore + "
        f"{contract['frame_count']} exact recorded robot-path steps; "
        "contacts solved by PhysX",
        flush=True,
    )
    started = time.time()
    completed = subprocess.run(command, env=child_env, check=False)
    report = {
        "mode": MODE_NAME,
        "collection_engine": "isaac",
        "replay_engine": "isaac_physx",
        "control_contract": (
            "frame0_state_plus_exact_recorded_physx_substep_robot_path"
        ),
        "robot_state_written_each_substep": True,
        "per_frame_object_state_writeback": False,
        "online_cartesian_dls": False,
        "contact_solver": "physx",
        "episode": str(episode),
        "output_dir": str(output_dir),
        "return_code": int(completed.returncode),
        "elapsed_seconds": time.time() - started,
        "contract": contract,
    }
    report_path = episode / f"{MODE_NAME}.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    if completed.returncode == 0:
        shutil.rmtree(work_root, ignore_errors=True)
    else:
        report["retained_input"] = str(work_root)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    print(f"[{MODE_NAME}] report={report_path}", flush=True)
    return int(completed.returncode)


if __name__ == "__main__":
    sys.exit(main())
