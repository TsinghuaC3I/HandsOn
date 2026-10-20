#!/usr/bin/env python3
"""Plan and certify smooth full ROKAE arms for simplified A-D episodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.c01_compute_arm_plan_qpos import (
    _DEFAULT_ACCELERATION_WEIGHT,
    _DEFAULT_COLLISION_PENETRATION_TOLERANCE,
    _DEFAULT_COLLISION_SWEEP_JOINT_STEP,
    _DEFAULT_MAX_FK_ORIENTATION_ERROR,
    _DEFAULT_MAX_FK_POSITION_ERROR,
    _DEFAULT_MAX_IK_ERROR,
    _DEFAULT_MAX_JOINT_STEP,
    _DEFAULT_VELOCITY_WEIGHT,
    _compute_for_replay,
    _resolve_replay_paths,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Demo dir, replay.zarr, or directory containing replay.zarr demos.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hz", type=int, default=1000)
    parser.add_argument("--velocity-weight", type=float, default=_DEFAULT_VELOCITY_WEIGHT)
    parser.add_argument(
        "--acceleration-weight",
        type=float,
        default=_DEFAULT_ACCELERATION_WEIGHT,
    )
    parser.add_argument("--max-joint-step", type=float, default=_DEFAULT_MAX_JOINT_STEP)
    parser.add_argument("--max-ik-error", type=float, default=_DEFAULT_MAX_IK_ERROR)
    parser.add_argument(
        "--max-fk-position-error",
        type=float,
        default=_DEFAULT_MAX_FK_POSITION_ERROR,
    )
    parser.add_argument(
        "--max-fk-orientation-error",
        type=float,
        default=_DEFAULT_MAX_FK_ORIENTATION_ERROR,
    )
    parser.add_argument(
        "--collision-penetration-tolerance",
        type=float,
        default=_DEFAULT_COLLISION_PENETRATION_TOLERANCE,
    )
    parser.add_argument(
        "--collision-sweep-joint-step",
        type=float,
        default=_DEFAULT_COLLISION_SWEEP_JOINT_STEP,
    )
    args = parser.parse_args()
    replay_paths = _resolve_replay_paths(args.paths)
    if not replay_paths:
        raise FileNotFoundError("No replay.zarr paths found")
    for replay_path in replay_paths:
        _compute_for_replay(
            replay_path,
            overwrite=bool(args.overwrite),
            hz=max(1, int(args.hz)),
            velocity_weight=float(args.velocity_weight),
            acceleration_weight=float(args.acceleration_weight),
            max_joint_step=float(args.max_joint_step),
            max_ik_error=float(args.max_ik_error),
            max_fk_position_error=float(args.max_fk_position_error),
            max_fk_orientation_error=float(args.max_fk_orientation_error),
            collision_penetration_tolerance=float(
                args.collision_penetration_tolerance
            ),
            collision_sweep_joint_step=float(args.collision_sweep_joint_step),
            output_prefix="simplified_arm_plan",
            log_label="Simplified post-IK arm plan",
            # Ordinary A-D states do not share C01's private table-delta slot.
            table_delta_index=None,
            state_dataset="state",
            tcp_pose_dataset="physx_tcp_state",
            tcp_valid_dataset="physx_tcp_valid",
            external_plan_path=(
                replay_path.parent / "simplified_arm_plan.npz"
            ),
        )


if __name__ == "__main__":
    main()
