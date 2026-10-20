from __future__ import annotations

import json

import numpy as np
import zarr

from scripts.compare_physx_ground_truth import summarize_physx_episode


def test_physx_comparison_uses_action_aligned_next_state(tmp_path) -> None:
    replay = tmp_path / "demo" / "replay.zarr"
    root = zarr.open(str(replay), mode="w")
    data = root.create_group("data")
    steps = 2

    current_joint = np.zeros((steps, 2, 2), dtype=np.float64)
    next_joint = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [4.0, 5.0]],
        ]
    )
    next_target = next_joint + 0.5
    valid_joint = np.ones((steps, 2, 2), dtype=np.uint8)
    current_tcp = np.zeros((steps, 2, 13), dtype=np.float64)
    next_tcp = np.zeros((steps, 2, 13), dtype=np.float64)
    next_tcp[0, :, 0] = 1.0
    next_tcp[1, :, 0] = 3.0
    valid_tcp = np.ones((steps, 2), dtype=np.uint8)

    arrays = {
        "physx_robot_joint_position": current_joint,
        "next_physx_robot_joint_position": next_joint,
        "next_physx_robot_joint_target": next_target,
        "next_physx_robot_joint_target_valid": valid_joint,
        "physx_robot_joint_valid": valid_joint,
        "next_physx_robot_joint_valid": valid_joint,
        "physx_tcp_state": current_tcp,
        "next_physx_tcp_state": next_tcp,
        "physx_tcp_valid": valid_tcp,
        "next_physx_tcp_valid": valid_tcp,
        "physx_sim_time": np.asarray([[0.0], [0.1]]),
        "next_physx_sim_time": np.asarray([[0.1], [0.2]]),
    }
    for name, value in arrays.items():
        data.array(name, data=value)
    root.attrs["recording_schema"] = {
        "environment_schema": {
            "physx": {
                "robot_joint_names": {
                    "right": ["r0", "r1"],
                    "left": ["l0", "l1"],
                }
            }
        }
    }

    result = summarize_physx_episode(replay.parent)

    assert result["control_transitions"] == 2
    assert result["duration_s"] == 0.2
    assert result["robot"]["right"]["joint_target_actual_rmse"] == 0.5
    assert result["robot"]["left"]["joint_target_actual_max_abs"] == 0.5
    assert result["robot"]["right"]["tcp"]["path_length_m"] == 3.0
    assert result["robot"]["right"]["joint_names"] == ["r0", "r1"]
    json.dumps(result, allow_nan=False)
