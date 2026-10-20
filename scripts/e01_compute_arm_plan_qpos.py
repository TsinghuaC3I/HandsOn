#!/usr/bin/env python3
"""Compute the post-success E01 dual-arm plan from a hands-only recording.

The actual IK, global smoothing, FK validation, and swept collision audit are
shared with C01/Plan1.  E01 only supplies its recording-layout rules and writes
an independent ``e01_arm_plan_*`` result set.
"""

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

_E01_TABLE_DELTA_INDEX = 101


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a validated E01 ROKAE arm trajectory after hands-only "
            "Genesis collection succeeds."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Demo dir, replay.zarr, or directory containing replay.zarr demos.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hz", type=int, default=1000)
    parser.add_argument(
        "--velocity-weight",
        type=float,
        default=_DEFAULT_VELOCITY_WEIGHT,
        help="First-difference smoothing weight for the offline arm path.",
    )
    parser.add_argument(
        "--acceleration-weight",
        type=float,
        default=_DEFAULT_ACCELERATION_WEIGHT,
        help="Second-difference smoothing weight for the offline arm path.",
    )
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=_DEFAULT_MAX_JOINT_STEP,
        help="Maximum per-recorded-frame arm joint change in radians.",
    )
    parser.add_argument(
        "--max-ik-error",
        type=float,
        default=_DEFAULT_MAX_IK_ERROR,
        help="Reject IK frames at or above this position/orientation error budget.",
    )
    parser.add_argument(
        "--max-fk-position-error",
        type=float,
        default=_DEFAULT_MAX_FK_POSITION_ERROR,
        help="Fail if either final palm exceeds this FK position error (m).",
    )
    parser.add_argument(
        "--max-fk-orientation-error",
        type=float,
        default=_DEFAULT_MAX_FK_ORIENTATION_ERROR,
        help="Fail if either final palm exceeds this FK orientation error (rad).",
    )
    parser.add_argument(
        "--collision-penetration-tolerance",
        type=float,
        default=_DEFAULT_COLLISION_PENETRATION_TOLERANCE,
        help="Allowed audited arm collision penetration before failure (m).",
    )
    parser.add_argument(
        "--collision-sweep-joint-step",
        type=float,
        default=_DEFAULT_COLLISION_SWEEP_JOINT_STEP,
        help="Maximum interpolation step for between-frame collision audit (rad).",
    )
    parser.add_argument(
        "--table-delta-from-state-101",
        action="store_true",
        help=(
            "Read E01 table_delta_height from its explicit state index 101. "
            "Without this flag, use a fixed zero for every frame."
        ),
    )
    args = parser.parse_args()

    replay_paths = _resolve_replay_paths(args.paths)
    if not replay_paths:
        raise SystemExit("No replay.zarr paths found.")
    table_delta_index = (
        _E01_TABLE_DELTA_INDEX if args.table_delta_from_state_101 else None
    )
    for replay_path in replay_paths:
        _compute_for_replay(
            replay_path,
            overwrite=args.overwrite,
            hz=args.hz,
            velocity_weight=args.velocity_weight,
            acceleration_weight=args.acceleration_weight,
            max_joint_step=args.max_joint_step,
            max_ik_error=args.max_ik_error,
            max_fk_position_error=args.max_fk_position_error,
            max_fk_orientation_error=args.max_fk_orientation_error,
            collision_penetration_tolerance=(
                args.collision_penetration_tolerance
            ),
            collision_sweep_joint_step=args.collision_sweep_joint_step,
            output_prefix="e01_arm_plan",
            log_label="E01 arm plan",
            table_delta_index=table_delta_index,
            # Each saved transition stores the protected palm pose reached by
            # that action in next_state.  Using state here would shift the arm
            # plan back one frame and omit the final contact pose.
            state_dataset="next_state",
        )


if __name__ == "__main__":
    main()
