#!/usr/bin/env python3
"""B320-FU006 scheduled live probe, automatic collection, and fresh replay.

The automatic policy emits only the declared 54-D environment action.  It
never restores or writes robot/object simulator state after action 0.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

import numpy as np


TASK_NAME = "t46_store_charger"
STATE_KEYS = (
    "tcp_pose",
    "gripper_pose",
    "charger_pose",
    "drawer_pose",
    "drawer_joint_state",
    "fu006_task_state",
    "table_delta_height",
)
CAMERA_KEYS = ("ego", "wrist_left", "wrist_right")
_SHARED_ROBOT_IMPORT_MESHES = (
    "wrist_fisheye/meshes/wrist_fisheye_flange_assembly.stl",
    "wuji_attachment/direct_adapter/meshes/Direct-Adapter-Mount-binary.stl",
)
_AUTOMATIC_PHASE_SCHEDULE = (
    ("right_approach", 60),
    ("right_descend", 35),
    ("right_close", 60),
    ("right_initial_lift", 90),
    ("right_flatten", 120),
    ("right_stage_outside_side", 180),
    ("right_lift_outside_side", 300),
    ("left_stage", 90),
    ("left_approach", 90),
    ("left_contact_correction", 90),
    ("left_contact_refine", 90),
    ("left_close", 45),
    ("left_pull", 45),
    ("cross_side_above", 300),
    ("align_above_drawer", 120),
    ("lower_into_drawer", 180),
    ("seat_in_drawer", 120),
    ("right_release", 100),
    ("right_withdraw", 120),
    ("settle", 100),
)
_ACTION_GATE_TELEMETRY_FIELDS = (
    "commanded_right_palm_pose",
    "commanded_left_palm_pose",
)
_NEXT_GATE_TELEMETRY_FIELDS = (
    "measured_right_palm_pose",
    "measured_left_palm_pose",
    "right_palm_command_error_m",
    "left_palm_command_error_m",
    "nominal_right_fingertip_to_charger_m",
    "nominal_left_fingertip_to_handle_m",
    "charger_pose",
    "drawer_pose",
    "drawer_handle_contact_world",
    "fu006_drawer_joint_position",
    "fu006_drawer_joint_velocity",
    "fu006_drawer_open",
    "fu006_left_contact_available",
    "fu006_right_contact_available",
    "fu006_left_contact_force",
    "fu006_right_contact_force",
    "fu006_left_hand_command_max",
    "fu006_left_hand_actual_max",
    "fu006_left_hand_closed_actual",
    "fu006_left_hold_now",
    "fu006_left_hold_seen",
    "fu006_left_released_now",
    "fu006_containment_margin",
    "fu006_containment_margin_x",
    "fu006_containment_margin_y",
    "fu006_containment_margin_z",
    "fu006_charger_contained",
    "fu006_contained_seen",
    "fu006_right_hand_command_max",
    "fu006_right_hand_actual_max",
    "fu006_right_hand_open_actual",
    "fu006_right_hand_clearance",
    "fu006_right_released_now",
    "fu006_release_seen",
    "fu006_stable_now",
    "fu006_stability_steps",
    "fu006_stability_steps_required",
    "fu006_success_now",
)
_PHYSICAL_GATE_TELEMETRY_FIELDS = (
    *_ACTION_GATE_TELEMETRY_FIELDS,
    *_NEXT_GATE_TELEMETRY_FIELDS,
)
_FIRST_PHYSICAL_GATE = "scaled_drawer_q_reset_readback"
_EXPECTED_DRAWER_RESET_Q_M = 0.35 * 0.28
_DRAWER_RESET_Q_TOLERANCE_M = 0.005

# Exact aligned modern-drawer geometry and canonical Wuji fingertip FK.  Handle
# contact is transformed from link_1's live rigid-body pose rather than
# reconstructed from the root Xform and joint q; the latter is unsafe because
# PhysX prismatic q is already expressed in stage metres below a scaled root.
_DRAWER_SCALE = 0.35
_DRAWER_HANDLE_LOCAL_MIN = _DRAWER_SCALE * np.asarray(
    [-0.089235253632, -0.082089945674, 1.073254466057],
    dtype=np.float64,
)
_DRAWER_HANDLE_LOCAL_MAX = _DRAWER_SCALE * np.asarray(
    [0.177743837237, -0.067091614008, 1.108387708664],
    dtype=np.float64,
)
_RIGHT_POWER_MAIN_FINGERTIP_LOCAL = np.asarray(
    [0.068072232235673, -0.004110890672540, 0.112131915101676],
    dtype=np.float64,
)
_LEFT_STABILIZE_MAIN_FINGERTIP_LOCAL = np.asarray(
    [0.050383601506508, 0.006308643366351, 0.162349251052619],
    dtype=np.float64,
)
_PALM_DOWN_WXYZ = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
_VERTICAL_FINGERS_WXYZ = np.asarray(
    [1.0, 0.0, 0.0, 0.0], dtype=np.float64
)
_POWER_GRIP = np.asarray(
    [
        0.60,
        0.10,
        0.84,
        0.70,
        0.68,
        0.00,
        0.90,
        0.78,
        0.70,
        0.00,
        0.92,
        0.80,
        0.68,
        0.00,
        0.90,
        0.78,
        0.64,
        0.00,
        0.86,
        0.74,
    ],
    dtype=np.float64,
)
_STABILIZE_GRIP = 0.13 * _POWER_GRIP
# The full carry grip leaves about 18.6 N on the charger while seated and
# ejects it as the fingers open.  Relax inside the exact cavity before release
# so the dynamic object can transfer onto the drawer floor without stored
# squeeze energy.
_RIGHT_SEAT_GRIP = 0.40 * _POWER_GRIP
# Live trials bounded the usable handle contact between the open pose and a
# 0.30-rad maximum closure.  Keep a shallow 0.12-rad closure above the
# unchanged 0.08-rad physical closed-hand gate; deeper closure curls the
# fingertips away from this thin rail after opening it.
_LEFT_HANDLE_CONTACT_FRACTION = 0.85
_LEFT_HANDLE_PULL_M = 0.025
# Preserve the validated handle-relative pose when the fixed drawer layout is
# translated.  Interpolating the XYZ from the unchanged robot home would move
# the hand by only 85% of a scene translation and weakened the live hold from
# about 51 N to 1 N.
_LEFT_HANDLE_HOLD_OFFSET_WORLD = np.asarray(
    [0.0261683, 0.0536312, -0.0332302], dtype=np.float64
)
# The charger has 44.0 mm of vertical room in the aligned modern cavity.  The
# automatic validator therefore measures and removes roll/pitch after the
# first physical lift, then derives all placement contacts from the observed
# object-to-fingertip offset.  This remains a deterministic 54-D action policy;
# it never writes state after action 0 and does not modify success thresholds.
_RIGHT_INITIAL_LIFT_M = 0.150
# Successful zero-yaw trials settle the closed power grasp at about -24
# degrees object yaw.  Normalize every randomized start to that proven world
# yaw while leveling; otherwise the same high-lift path enters different IK
# branches.  The later cavity turn remains deferred until full clearance.
_FLAT_CARRY_WORLD_YAW_RAD = np.deg2rad(-24.0)
# The charger footprint is far smaller than the exact cavity in both
# horizontal axes, so use only the turn needed to keep the low DLS branch and
# physical grasp viable.  Live 0 and 30 degrees lost the charger during
# descent, while 90 degrees escaped during the cross.  The 60-degree path is
# the only tested orientation that preserves the grasp through full lowering;
# scene placement, rather than additional wrist rotation, resolves reach.
_CAVITY_CARRY_YAW_OFFSET_RAD = np.pi / 3.0
# Bound only the final integral update.  The earlier full descent must cover
# the real cavity height, while this last correction exists solely to remove a
# small tracking residual and must not wind up through the drawer when IK lags.
_FINAL_RESIDUAL_CORRECTION_LIMIT_M = 0.035
# In the drawer link frame, +X points toward negative world Y and +Z points
# toward negative world X.  First stage the closed grasp beyond the cabinet's
# left-front corner at low height, then lift vertically.  A single diagonal
# lift crossed that corner while the fingers were still inside its footprint
# and repeatedly ejected the charger near world z=1.22 m.  The high cross into
# the drawer starts only after this exterior lane clears the cabinet top.
_DRAWER_SAFE_LIFT_CENTER_LOCAL_X = 0.200
_DRAWER_SAFE_LIFT_LOCAL_Z_OFFSET_M = 0.060
_LEFT_CONTACT_CORRECTION_LIMIT_M = 0.180
# Clear the cabinet outer top before crossing its front edge.
# At local Y=0.085 the flat charger's lower support retains about 24 mm of
# nominal clearance; placement then descends to the aligned live cavity.
# Local X=0.020 keeps the rotated charger about 44 mm from the nearer side,
# instead of releasing against the former 4 mm side margin.
_DRAWER_OUTSIDE_ABOVE_HEIGHT_LOCAL_M = 0.085
_DRAWER_ABOVE_CENTER_LOCAL = np.asarray([0.020, 0.085, 0.270], dtype=np.float64)
_DRAWER_DESIRED_CENTER_LOCAL = np.asarray(
    [0.020, -0.02920268, 0.270], dtype=np.float64
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _configure_task_private_isaac_runtime(output: Path) -> dict[str, str]:
    """Isolate FU006 MJCF conversion without changing the canonical robot.

    Isaac Sim converts referenced STL meshes through adjacent ``*_tmp`` USD
    layers.  The left and right canonical MJCFs share the wrist fixture and
    hand adapter meshes, so a cold dual-arm conversion can race on those temp
    layers.  A stale global conversion cache can preserve the resulting broken
    reference.  This task-private overlay changes only the two mesh *paths* in
    private copies of the robot XMLs; the mesh bytes, robot kinematics, drives,
    controller, and initial state remain canonical.
    """
    source_value = os.environ.get("HANDSON_ISAACLAB_XML_DIR")
    source_root = (
        Path(source_value).expanduser()
        if source_value
        else _repo_root() / "handson/handson/sim/envs/xmls"
    ).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"FU006 Isaac XML root is missing: {source_root}")

    workspace_value = os.environ.get("WORKSPACE_DIR")
    workspace = (
        Path(workspace_value).expanduser().resolve()
        if workspace_value
        else output.expanduser().resolve().parent
    )
    private_root = workspace / "fu006_runtime"
    overlay = private_root / "xml_overlay"
    cache = private_root / "isaaclab_cache"
    overlay.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    robot_xml_names = {
        "left": "rokae_wuji1_left.xml",
        "right": "rokae_wuji1_right.xml",
    }
    for entry in sorted(source_root.iterdir(), key=lambda path: path.name):
        if entry.name in robot_xml_names.values():
            continue
        destination = overlay / entry.name
        if destination.is_symlink():
            if destination.resolve() != entry.resolve():
                raise RuntimeError(
                    f"FU006 private overlay has an unexpected link: {destination}"
                )
            continue
        if destination.exists():
            raise RuntimeError(
                f"FU006 private overlay path already exists: {destination}"
            )
        destination.symlink_to(entry, target_is_directory=entry.is_dir())

    for side, xml_name in robot_xml_names.items():
        source_xml = source_root / xml_name
        if not source_xml.is_file():
            raise FileNotFoundError(f"FU006 canonical robot XML is missing: {source_xml}")
        content = source_xml.read_text(encoding="utf-8")
        for relative_mesh in _SHARED_ROBOT_IMPORT_MESHES:
            if relative_mesh not in content:
                raise RuntimeError(
                    f"FU006 canonical {xml_name} lacks mesh {relative_mesh}"
                )
            source_mesh = (source_root / relative_mesh).resolve()
            if not source_mesh.is_file():
                raise FileNotFoundError(
                    f"FU006 canonical robot mesh is missing: {source_mesh}"
                )
            alias_relative = (
                Path("fu006_robot_mesh_aliases") / side / source_mesh.name
            )
            alias_path = overlay / alias_relative
            alias_path.parent.mkdir(parents=True, exist_ok=True)
            if (
                not alias_path.is_file()
                or alias_path.stat().st_size != source_mesh.stat().st_size
                or _sha256(alias_path) != _sha256(source_mesh)
            ):
                shutil.copy2(source_mesh, alias_path)
            content = content.replace(relative_mesh, alias_relative.as_posix())

        destination_xml = overlay / xml_name
        encoded = content.encode("utf-8")
        if not destination_xml.is_file() or destination_xml.read_bytes() != encoded:
            destination_xml.write_bytes(encoded)

    os.environ["HANDSON_ISAACLAB_XML_DIR"] = str(overlay)
    os.environ["HANDSON_ISAACLAB_CACHE"] = str(cache)
    return {
        "source_xml_root": str(source_root),
        "private_xml_overlay": str(overlay),
        "private_cache_root": str(cache),
    }


def _runtime_contract() -> dict[str, Any]:
    isaacsim = importlib.metadata.version("isaacsim")
    if not isaacsim.startswith("5.1"):
        raise RuntimeError(f"FU006 requires Isaac Sim 5.1, got {isaacsim}")
    if os.environ.get("HANDSON_ENABLE_ISAACSIM_NEWTON", "0").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }:
        raise RuntimeError("FU006 forbids Newton")
    root = _repo_root()
    files = {
        "abc_collection_env.py": root
        / "handson/handson/sim/isaaclab/abc_collection_env.py",
        "task_preview.py": root
        / "handson/handson/sim/isaaclab/task_preview.py",
        "task_config.py": root
        / "handson/handson/tasks/t46_store_charger/config.py",
        "task_yaml": root
        / "configs/multi_task/t46_store_charger.yaml",
        "drawer_usd": root
        / "handson/handson/sim/envs/xmls/assets/bench320_fu006/dv_drawer_45575/45575.usd",
        "charger_usd": root
        / "handson/handson/sim/envs/xmls/assets/bench320_fu006/dexverse_charger/model_charger.usd",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"FU006 runtime inputs missing: {missing}")
    return {
        "task_name": TASK_NAME,
        "runtime_family": "isaac_sim_5_1",
        "isaacsim_version": isaacsim,
        "physics_backend": "physx",
        "newton_enabled": False,
        "hashes": {name: _sha256(path) for name, path in files.items()},
    }


def _new_env(seed: int, *, image_obs: bool = True):
    os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
    os.environ["HANDSON_ENABLE_ISAACSIM_NEWTON"] = "0"
    from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv

    try:
        return HandsOnABCIsaacLabEnv(
            TASK_NAME,
            seed=seed,
            randomize=True,
            image_obs=image_obs,
            render_mode="rgb_array",
            render_resolution=(480, 640),
            hz=30,
            physics_dt=1.0 / 120.0,
            time_limit=90.0,
            physics_backend="physx",
            headless=True,
        )
    except SystemExit as exc:
        # SimulationApp.close() may raise SystemExit(0) while cleaning up a
        # constructor failure.  A scheduled probe must never report that
        # incomplete scene as a successful process exit.
        raise RuntimeError(
            "FU006 environment construction exited before returning a live scene"
        ) from exc


def _state_vector(observation: dict[str, Any]) -> np.ndarray:
    state = observation["state"]
    return np.concatenate(
        [np.asarray(state[key], dtype=np.float64).reshape(-1) for key in STATE_KEYS]
    )


def _camera_frame(observation: dict[str, Any], key: str) -> np.ndarray:
    images = observation.get("images", {})
    source = "random_camera" if key == "ego" and "random_camera" in images else key
    frame = np.asarray(images[source])
    if frame.ndim != 3 or frame.shape[-1] < 3:
        raise RuntimeError(f"FU006 camera {key} has invalid shape {frame.shape}")
    frame = np.asarray(frame[..., :3], dtype=np.uint8)
    if not np.any(frame) or float(np.std(frame)) < 1.0:
        raise RuntimeError(f"FU006 camera {key} is black/constant")
    return frame


def _truth_fields(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    for name, value in observation.items():
        if name in {"state", "images"}:
            continue
        array = np.asarray(value)
        if array.dtype == object or array.ndim == 0:
            continue
        fields[name] = np.ascontiguousarray(array)
    return fields


def _action(
    right_pose: np.ndarray,
    left_pose: np.ndarray,
    right_hand: np.ndarray,
    left_hand: np.ndarray,
) -> np.ndarray:
    return np.concatenate([right_pose, left_pose, right_hand, left_hand]).astype(
        np.float32
    )


def _lerp_pose(start: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    result = np.asarray(start, dtype=np.float64).copy()
    result[:3] = (1.0 - alpha) * start[:3] + alpha * target[:3]
    start_quat = np.asarray(start[3:7], dtype=np.float64)
    target_quat = np.asarray(target[3:7], dtype=np.float64)
    start_quat /= max(float(np.linalg.norm(start_quat)), 1e-9)
    target_quat /= max(float(np.linalg.norm(target_quat)), 1e-9)
    if float(np.dot(start_quat, target_quat)) < 0.0:
        target_quat = -target_quat
    result[3:7] = (1.0 - alpha) * start_quat + alpha * target_quat
    result[3:7] /= max(float(np.linalg.norm(result[3:7])), 1e-9)
    return result


def _phase_actions(
    start_right: np.ndarray,
    start_left: np.ndarray,
    target_right: np.ndarray,
    target_left: np.ndarray,
    start_right_hand: np.ndarray,
    start_left_hand: np.ndarray,
    target_right_hand: np.ndarray,
    target_left_hand: np.ndarray,
    frames: int,
) -> Iterable[np.ndarray]:
    for index in range(1, frames + 1):
        alpha = float(index / max(1, frames))
        yield _action(
            _lerp_pose(start_right, target_right, alpha),
            _lerp_pose(start_left, target_left, alpha),
            (1.0 - alpha) * start_right_hand + alpha * target_right_hand,
            (1.0 - alpha) * start_left_hand + alpha * target_left_hand,
        )


def _rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-9)
    scalar = float(quaternion[0])
    imaginary = quaternion[1:4]
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    return (
        (scalar * scalar - float(np.dot(imaginary, imaginary))) * vector
        + 2.0 * imaginary * float(np.dot(imaginary, vector))
        + 2.0 * scalar * np.cross(imaginary, vector)
    )


def _multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the normalized Hamilton product of two WXYZ quaternions."""
    w1, x1, y1, z1 = np.asarray(left, dtype=np.float64).reshape(4)
    w2, x2, y2, z2 = np.asarray(right, dtype=np.float64).reshape(4)
    result = np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
    result /= max(float(np.linalg.norm(result)), 1e-9)
    return result


def _inverse_wxyz(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float64).reshape(4).copy()
    result /= max(float(np.linalg.norm(result)), 1e-9)
    result[1:4] *= -1.0
    return result


def _yaw_only_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Remove roll/pitch while preserving the live object's world yaw."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = max(float(np.linalg.norm([w, x, y, z])), 1e-9)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return np.asarray(
        [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)],
        dtype=np.float64,
    )


def _flat_carry_orientation_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Return the level, arm-friendly charger carry orientation."""
    _yaw_only_wxyz(quaternion)
    half_yaw = 0.5 * _FLAT_CARRY_WORLD_YAW_RAD
    return np.asarray(
        [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
        dtype=np.float64,
    )


def _cavity_carry_orientation_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Return the level, cavity-facing charger orientation."""
    half_offset = 0.5 * _CAVITY_CARRY_YAW_OFFSET_RAD
    yaw_offset = np.asarray(
        [np.cos(half_offset), 0.0, 0.0, np.sin(half_offset)],
        dtype=np.float64,
    )
    return _multiply_wxyz(yaw_offset, _yaw_only_wxyz(quaternion))


def _palm_pose_for_contact(
    contact_world: np.ndarray,
    fingertip_local: np.ndarray,
    palm_orientation_wxyz: np.ndarray = _PALM_DOWN_WXYZ,
) -> np.ndarray:
    orientation = np.asarray(
        palm_orientation_wxyz, dtype=np.float64
    ).reshape(4).copy()
    orientation /= max(float(np.linalg.norm(orientation)), 1e-9)
    fingertip_world = _rotate_wxyz(
        orientation,
        np.asarray(fingertip_local, dtype=np.float64),
    )
    return np.concatenate(
        [
            np.asarray(contact_world, dtype=np.float64) - fingertip_world,
            orientation,
        ]
    )


def _drawer_handle_contact_local() -> np.ndarray:
    center = 0.5 * (_DRAWER_HANDLE_LOCAL_MIN + _DRAWER_HANDLE_LOCAL_MAX)
    # Keep the left hand near the positive-X end so it does not obstruct the
    # right-hand charger transfer through the drawer center.
    center[0] = _DRAWER_HANDLE_LOCAL_MAX[0] - 0.015
    return center


def _drawer_handle_contact_world(drawer_pose: np.ndarray) -> np.ndarray:
    drawer = np.asarray(drawer_pose, dtype=np.float64).reshape(7)
    return drawer[:3] + _rotate_wxyz(
        drawer[3:7], _drawer_handle_contact_local()
    )


def _measured_palm_pose(env: Any, side: str) -> np.ndarray:
    """Read one real PhysX palm pose without changing robot/controller state."""
    robot = env._control_robot(side)
    matrix = env._robot_body_pose_matrix(robot, env._ee_body_indices[side])
    pose = np.asarray(
        env._pose_matrix_to_vector(matrix), dtype=np.float64
    ).reshape(7)
    if not np.all(np.isfinite(pose)):
        raise RuntimeError(f"FU006 measured {side} palm pose is non-finite")
    return pose


def _measured_fingertips(env: Any, side: str) -> dict[str, np.ndarray]:
    """Read all five visible PhysX fingertip centers for diagnostics."""
    positions = {
        str(name): np.asarray(value, dtype=np.float64).reshape(3).copy()
        for name, value in env._wuji_fingertip_positions(side).items()
    }
    if len(positions) != 5 or any(
        not np.all(np.isfinite(value)) for value in positions.values()
    ):
        raise RuntimeError(
            f"FU006 measured {side} fingertips are incomplete or non-finite"
        )
    return positions


def _scaled_drawer_reset_gate(
    env: Any, observation: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed unless reset exposes the scaled live PhysX drawer q."""
    reading = env._physics_dof_position("joint_1")
    velocity = env._physics_dof_velocity("joint_1")
    if reading is None or velocity is None:
        raise RuntimeError(
            "FU006 scaled drawer reset gate lacks a live PhysX DOF readback"
        )
    physical_q, physical_initial_q = map(float, reading)
    observed = np.asarray(
        observation["state"]["drawer_joint_state"], dtype=np.float64
    ).reshape(-1)
    values = np.asarray(
        [physical_q, physical_initial_q, float(velocity), *observed[:2]],
        dtype=np.float64,
    )
    if observed.size < 2 or not np.all(np.isfinite(values)):
        raise RuntimeError(
            "FU006 scaled drawer reset gate readback is incomplete or non-finite"
        )
    error_m = abs(physical_q - _EXPECTED_DRAWER_RESET_Q_M)
    observation_error_m = abs(float(observed[0]) - physical_q)
    passed = bool(
        error_m <= _DRAWER_RESET_Q_TOLERANCE_M
        and observation_error_m <= 1.0e-6
    )
    gate = {
        "name": _FIRST_PHYSICAL_GATE,
        "passed": passed,
        "expected_q_m": _EXPECTED_DRAWER_RESET_Q_M,
        "tolerance_m": _DRAWER_RESET_Q_TOLERANCE_M,
        "physical_q_m": physical_q,
        "physical_initial_q_m": physical_initial_q,
        "physical_qd_m_s": float(velocity),
        "observation_q_m": float(observed[0]),
        "observation_qd_m_s": float(observed[1]),
        "error_m": error_m,
        "observation_error_m": observation_error_m,
    }
    if not passed:
        raise RuntimeError(
            "FU006 scaled drawer reset gate failed: "
            + json.dumps(gate, sort_keys=True)
        )
    return gate


def _nominal_fingertip_world(
    palm_pose: np.ndarray, fingertip_local: np.ndarray
) -> np.ndarray:
    palm = np.asarray(palm_pose, dtype=np.float64).reshape(7)
    return palm[:3] + _rotate_wxyz(palm[3:7], fingertip_local)


def _asset_frame_waypoints(
    observation: dict[str, Any],
    *,
    measured_right_home: np.ndarray | None = None,
    measured_left_home: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Derive palm waypoints from action-0 asset and physical robot frames."""
    state = observation["state"]
    tcp = np.asarray(state["tcp_pose"], dtype=np.float64).reshape(14)
    right_home = np.asarray(
        tcp[:7] if measured_right_home is None else measured_right_home,
        dtype=np.float64,
    ).reshape(7).copy()
    left_home = np.asarray(
        tcp[7:14] if measured_left_home is None else measured_left_home,
        dtype=np.float64,
    ).reshape(7).copy()
    for name, home in (("right", right_home), ("left", left_home)):
        if not np.all(np.isfinite(home)):
            raise ValueError(f"FU006 requires a finite measured {name} palm pose")
        home[3:7] /= max(float(np.linalg.norm(home[3:7])), 1e-9)
    charger = np.asarray(state["charger_pose"], dtype=np.float64).reshape(7)
    drawer = np.asarray(state["drawer_pose"], dtype=np.float64).reshape(7)
    drawer_joint = np.asarray(
        state["drawer_joint_state"], dtype=np.float64
    ).reshape(-1)
    if drawer_joint.size < 1 or not np.all(np.isfinite(drawer_joint)):
        raise ValueError("FU006 requires finite drawer_joint_state truth")

    left_contact = _palm_pose_for_contact(
        _drawer_handle_contact_world(drawer),
        _LEFT_STABILIZE_MAIN_FINGERTIP_LOCAL,
        _VERTICAL_FINGERS_WXYZ,
    )
    left_hold = _lerp_pose(
        left_home, left_contact, _LEFT_HANDLE_CONTACT_FRACTION
    )
    left_hold[:3] = left_contact[:3] + _LEFT_HANDLE_HOLD_OFFSET_WORLD
    # First move behind the cabinet's front plane while remaining outside its
    # positive-Y side, then approach the handle laterally.  A single diagonal
    # segment crosses the fixed cabinet shell after the reachability shift.
    left_stage = left_hold.copy()
    left_stage[1] = left_home[1]
    left_pull = left_hold.copy()
    left_pull[0] -= _LEFT_HANDLE_PULL_M
    # Align the power-grasp frame with the live charger yaw.  FU006 preserves
    # the production ±30 degree object-yaw randomization; keeping the palm at
    # its fixed home yaw pushed rotated chargers sideways during closure and
    # lost contact before the first lift.
    right_grasp_orientation = _multiply_wxyz(
        _yaw_only_wxyz(charger[3:7]), right_home[3:7]
    )
    right_grasp = _palm_pose_for_contact(
        charger[:3],
        _RIGHT_POWER_MAIN_FINGERTIP_LOCAL,
        right_grasp_orientation,
    )
    right_approach = right_grasp.copy()
    right_approach[2] += 0.10
    right_initial_lift = right_grasp.copy()
    right_initial_lift[2] += _RIGHT_INITIAL_LIFT_M
    return {
        "right_home": right_home,
        "left_home": left_home,
        "left_contact": left_contact,
        "left_stage": left_stage,
        "left_hold": left_hold,
        "left_pull": left_pull,
        "right_approach": right_approach,
        "right_grasp": right_grasp,
        "right_initial_lift": right_initial_lift,
    }


def _automatic_phase_for_step(step: int) -> str:
    if step < 1:
        raise ValueError("FU006 action steps are one-based")
    boundary = 0
    for name, frames in _AUTOMATIC_PHASE_SCHEDULE:
        boundary += frames
        if step <= boundary:
            return name
    raise ValueError(f"FU006 action step {step} exceeds the phase schedule")


def _phase_frames(name: str) -> int:
    for phase_name, frames in _AUTOMATIC_PHASE_SCHEDULE:
        if phase_name == name:
            return frames
    raise KeyError(f"FU006 automatic schedule lacks phase {name}")


def automatic_prefix_actions(
    observation: dict[str, Any],
    *,
    measured_right_home: np.ndarray | None = None,
    measured_left_home: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Build the action-0-derived prefix through the initial grasp lift."""
    waypoints = _asset_frame_waypoints(
        observation,
        measured_right_home=measured_right_home,
        measured_left_home=measured_left_home,
    )
    right_home = waypoints["right_home"]
    left_home = waypoints["left_home"]
    open_hand = np.zeros(20, dtype=np.float64)

    actions: list[np.ndarray] = []
    current_right = right_home
    current_left = left_home
    current_right_hand = open_hand
    current_left_hand = open_hand
    phases = (
        (
            waypoints["right_approach"],
            left_home,
            open_hand,
            open_hand,
        ),
        (
            waypoints["right_grasp"],
            left_home,
            open_hand,
            open_hand,
        ),
        (
            waypoints["right_grasp"],
            left_home,
            _POWER_GRIP,
            open_hand,
        ),
        (
            waypoints["right_initial_lift"],
            left_home,
            _POWER_GRIP,
            open_hand,
        ),
    )
    for (name, frames), (
        target_right,
        target_left,
        right_hand,
        left_hand,
    ) in zip(_AUTOMATIC_PHASE_SCHEDULE[: len(phases)], phases, strict=True):
        del name
        actions.extend(
            _phase_actions(
                current_right,
                current_left,
                target_right,
                target_left,
                current_right_hand,
                current_left_hand,
                right_hand,
                left_hand,
                frames,
            )
        )
        current_right = target_right
        current_left = target_left
        current_right_hand = right_hand
        current_left_hand = left_hand
    return actions


def _left_approach_actions(
    start_action: np.ndarray, waypoints: dict[str, np.ndarray]
) -> list[np.ndarray]:
    """Move the open left hand from home to the nominal handle approach."""
    current = np.asarray(start_action, dtype=np.float64).reshape(54)
    current_right = current[:7].copy()
    current_left = current[7:14].copy()
    current_right_hand = current[14:34].copy()
    current_left_hand = current[34:54].copy()
    open_hand = np.zeros(20, dtype=np.float64)
    phases = (
        ("left_stage", waypoints["left_stage"], open_hand),
        ("left_approach", waypoints["left_hold"], open_hand),
    )
    actions: list[np.ndarray] = []
    for name, target_left, target_left_hand in phases:
        actions.extend(
            _phase_actions(
                current_right,
                current_left,
                current_right,
                target_left,
                current_right_hand,
                current_left_hand,
                current_right_hand,
                target_left_hand,
                _phase_frames(name),
            )
        )
        current_left = target_left
        current_left_hand = target_left_hand
    return actions


def _left_contact_correction_waypoint(
    observation: dict[str, Any],
    left_palm_command: np.ndarray,
    measured_left_palm: np.ndarray,
    measured_left_fingertips: dict[str, np.ndarray],
) -> np.ndarray:
    """Rebase the palm command on the live nearest-tip handle error."""
    drawer = np.asarray(
        observation["state"]["drawer_pose"], dtype=np.float64
    ).reshape(7)
    command = np.asarray(left_palm_command, dtype=np.float64).reshape(7)
    measured_palm = np.asarray(measured_left_palm, dtype=np.float64).reshape(7)
    handle = _drawer_handle_contact_world(drawer)
    fingertips = [
        np.asarray(value, dtype=np.float64).reshape(3)
        for value in measured_left_fingertips.values()
    ]
    if not fingertips:
        raise ValueError("FU006 left contact correction requires fingertips")
    values = np.concatenate(
        [drawer, command, measured_palm, handle, *fingertips]
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("FU006 left contact correction requires finite poses")
    nearest = min(fingertips, key=lambda value: np.linalg.norm(value - handle))
    correction = np.clip(
        handle - nearest,
        -_LEFT_CONTACT_CORRECTION_LIMIT_M,
        _LEFT_CONTACT_CORRECTION_LIMIT_M,
    )
    target = command.copy()
    # The DLS controller can lag the Cartesian command substantially near the
    # cabinet.  Adding a measured fingertip residual to that unachieved command
    # compounds the tracking error on every refinement.  Rebase translation on
    # the measured palm while retaining the intended grasp orientation.
    target[:3] = measured_palm[:3] + correction
    target[3:7] /= max(float(np.linalg.norm(target[3:7])), 1e-9)
    return target


def _left_contact_actions(
    start_action: np.ndarray, left_hold: np.ndarray
) -> list[np.ndarray]:
    """Apply the live handle correction, close, and add pull preload."""
    current = np.asarray(start_action, dtype=np.float64).reshape(54)
    current_right = current[:7].copy()
    current_left = current[7:14].copy()
    current_right_hand = current[14:34].copy()
    current_left_hand = current[34:54].copy()
    open_hand = np.zeros(20, dtype=np.float64)
    target_hold = np.asarray(left_hold, dtype=np.float64).reshape(7)
    target_pull = target_hold.copy()
    target_pull[0] -= _LEFT_HANDLE_PULL_M
    phases = (
        ("left_contact_refine", target_hold, open_hand),
        ("left_close", target_hold, _STABILIZE_GRIP),
        ("left_pull", target_pull, _STABILIZE_GRIP),
    )
    actions: list[np.ndarray] = []
    for name, target_left, target_left_hand in phases:
        actions.extend(
            _phase_actions(
                current_right,
                current_left,
                current_right,
                target_left,
                current_right_hand,
                current_left_hand,
                current_right_hand,
                target_left_hand,
                _phase_frames(name),
            )
        )
        current_left = target_left
        current_left_hand = target_left_hand
    return actions


def _grasp_orientation_waypoint(
    observation: dict[str, Any],
    measured_right_palm: np.ndarray,
    desired_charger_orientation: np.ndarray,
) -> np.ndarray:
    """Rotate a live rigid grasp about the observed charger center."""
    charger = np.asarray(
        observation["state"]["charger_pose"], dtype=np.float64
    ).reshape(7)
    palm = np.asarray(measured_right_palm, dtype=np.float64).reshape(7)
    desired = np.asarray(
        desired_charger_orientation, dtype=np.float64
    ).reshape(4)
    values = np.concatenate([charger, palm, desired])
    if not np.all(np.isfinite(values)):
        raise ValueError("FU006 grasp orientation waypoint requires finite poses")
    desired /= max(float(np.linalg.norm(desired)), 1e-9)
    world_adjustment = _multiply_wxyz(
        desired, _inverse_wxyz(charger[3:7])
    )
    desired_palm_orientation = _multiply_wxyz(
        world_adjustment, palm[3:7]
    )
    # Preserve the live rigid grasp transform while changing orientation.
    # The nominal straight-finger FK is not the contact point of the closed
    # power grip; rotating around it pulled the physical charger out of the
    # flexed fingers.  Rotating the measured palm-to-object vector about the
    # actual object center avoids that artificial translation.
    desired_palm_position = charger[:3] + _rotate_wxyz(
        world_adjustment, palm[:3] - charger[:3]
    )
    return np.concatenate(
        [desired_palm_position, desired_palm_orientation]
    )


def _flatten_grasp_waypoint(
    observation: dict[str, Any], measured_right_palm: np.ndarray
) -> np.ndarray:
    """Rotate the live grasp about the object center to remove roll/pitch."""
    charger = np.asarray(
        observation["state"]["charger_pose"], dtype=np.float64
    ).reshape(7)
    return _grasp_orientation_waypoint(
        observation,
        measured_right_palm,
        _flat_carry_orientation_wxyz(charger[3:7]),
    )


def _cavity_turn_grasp_waypoint(
    observation: dict[str, Any], measured_right_palm: np.ndarray
) -> np.ndarray:
    """Return the virtual high-clearance grasp after the cavity yaw turn."""
    charger = np.asarray(
        observation["state"]["charger_pose"], dtype=np.float64
    ).reshape(7)
    return _grasp_orientation_waypoint(
        observation,
        measured_right_palm,
        _cavity_carry_orientation_wxyz(charger[3:7]),
    )


def _adaptive_carry_waypoints(
    observation: dict[str, Any], measured_right_palm: np.ndarray
) -> dict[str, np.ndarray]:
    """Use the post-flatten live grasp offset to target the exact cavity."""
    state = observation["state"]
    charger = np.asarray(state["charger_pose"], dtype=np.float64).reshape(7)
    drawer = np.asarray(state["drawer_pose"], dtype=np.float64).reshape(7)
    palm = np.asarray(measured_right_palm, dtype=np.float64).reshape(7)
    values = np.concatenate([charger, drawer, palm])
    if not np.all(np.isfinite(values)):
        raise ValueError("FU006 adaptive carry requires finite live poses")
    drawer_inverse_orientation = _inverse_wxyz(drawer[3:7])
    contact_world = _nominal_fingertip_world(
        palm, _RIGHT_POWER_MAIN_FINGERTIP_LOCAL
    )
    object_contact_offset_local = _rotate_wxyz(
        drawer_inverse_orientation, charger[:3] - contact_world
    )
    charger_center_local = _rotate_wxyz(
        drawer_inverse_orientation, charger[:3] - drawer[:3]
    )
    stage_center_local = charger_center_local.copy()
    stage_center_local[0] = _DRAWER_SAFE_LIFT_CENTER_LOCAL_X
    stage_center_local[2] += _DRAWER_SAFE_LIFT_LOCAL_Z_OFFSET_M
    outside_center_local = stage_center_local.copy()
    outside_center_local[1] = _DRAWER_OUTSIDE_ABOVE_HEIGHT_LOCAL_M

    def target(center_local: np.ndarray) -> np.ndarray:
        contact_local = (
            np.asarray(center_local, dtype=np.float64)
            - object_contact_offset_local
        )
        contact = drawer[:3] + _rotate_wxyz(
            drawer[3:7], contact_local
        )
        return _palm_pose_for_contact(
            contact,
            _RIGHT_POWER_MAIN_FINGERTIP_LOCAL,
            palm[3:7],
        )

    right_stage = target(stage_center_local)
    right_lift = target(outside_center_local)
    drawer_above = target(_DRAWER_ABOVE_CENTER_LOCAL)
    drawer_inside = target(_DRAWER_DESIRED_CENTER_LOCAL)
    return {
        "right_stage": right_stage,
        "right_lift": right_lift,
        "drawer_above": drawer_above,
        "drawer_inside": drawer_inside,
        "right_withdraw": drawer_above.copy(),
    }


def _desired_charger_center_world(observation: dict[str, Any]) -> np.ndarray:
    """Transform the aligned exact-cavity target through the live drawer."""
    drawer = np.asarray(
        observation["state"]["drawer_pose"], dtype=np.float64
    ).reshape(7)
    if not np.all(np.isfinite(drawer)):
        raise ValueError("FU006 desired charger center requires a finite drawer pose")
    return drawer[:3] + _rotate_wxyz(
        drawer[3:7], _DRAWER_DESIRED_CENTER_LOCAL
    )


def _closed_loop_grasp_translation_waypoint(
    observation: dict[str, Any],
    right_palm_command: np.ndarray,
    *,
    horizontal_only: bool,
    translation_limit_m: float | None = None,
) -> np.ndarray:
    """Translate the live rigid grasp toward the exact cavity center.

    Horizontal alignment is performed while the charger is above the cabinet;
    the subsequent full correction lowers it only after the footprint is over
    the live drawer.  This changes only commanded 54-D actions and never writes
    robot, object, or drawer state.
    """
    charger = np.asarray(
        observation["state"]["charger_pose"], dtype=np.float64
    ).reshape(7)
    command = np.asarray(right_palm_command, dtype=np.float64).reshape(7)
    desired_center = _desired_charger_center_world(observation)
    values = np.concatenate([charger, command, desired_center])
    if not np.all(np.isfinite(values)):
        raise ValueError("FU006 grasp translation waypoint requires finite poses")
    translation = desired_center - charger[:3]
    if horizontal_only:
        translation[2] = 0.0
    if translation_limit_m is not None:
        if not np.isfinite(translation_limit_m) or translation_limit_m <= 0.0:
            raise ValueError("FU006 translation limit must be finite and positive")
        translation = np.clip(
            translation, -translation_limit_m, translation_limit_m
        )
    # Accumulate the measured object error on the previous command.  Rebasing
    # on the measured palm would discard the DLS controller's repeatable
    # tracking offset at every correction and leave the charger outside/high.
    target = command.copy()
    target[:3] += translation
    target[3:7] /= max(float(np.linalg.norm(target[3:7])), 1e-9)
    return target


def _adaptive_suffix_actions(
    start_action: np.ndarray,
    waypoints: dict[str, np.ndarray],
    *,
    include_lift: bool = True,
) -> list[np.ndarray]:
    """Build carry, release, withdraw, and settle from live waypoints."""
    current = np.asarray(start_action, dtype=np.float64).reshape(54)
    current_right = current[:7].copy()
    current_left = current[7:14].copy()
    current_right_hand = current[14:34].copy()
    current_left_hand = current[34:54].copy()
    open_hand = np.zeros(20, dtype=np.float64)
    phases = (
        ("right_lift_outside_side", waypoints["right_lift"], _POWER_GRIP),
        ("cross_side_above", waypoints["drawer_above"], _POWER_GRIP),
        ("align_above_drawer", waypoints["drawer_above"], _POWER_GRIP),
        ("lower_into_drawer", waypoints["drawer_inside"], _POWER_GRIP),
        ("seat_in_drawer", waypoints["drawer_inside"], _RIGHT_SEAT_GRIP),
        ("right_release", waypoints["drawer_inside"], open_hand),
        ("right_withdraw", waypoints["right_withdraw"], open_hand),
        ("settle", waypoints["right_withdraw"], open_hand),
    )
    if not include_lift:
        phases = phases[1:]
    actions: list[np.ndarray] = []
    for name, target_right, target_right_hand in phases:
        target_left_hand = open_hand if name == "settle" else current_left_hand
        actions.extend(
            _phase_actions(
                current_right,
                current_left,
                target_right,
                current_left,
                current_right_hand,
                current_left_hand,
                target_right_hand,
                target_left_hand,
                _phase_frames(name),
            )
        )
        current_right = target_right
        current_right_hand = target_right_hand
        current_left_hand = target_left_hand
    return actions


def _release_suffix_actions(
    start_action: np.ndarray, right_withdraw: np.ndarray
) -> list[np.ndarray]:
    """Open only after the two live placement corrections, then withdraw."""
    current = np.asarray(start_action, dtype=np.float64).reshape(54)
    current_right = current[:7].copy()
    current_left = current[7:14].copy()
    current_right_hand = current[14:34].copy()
    current_left_hand = current[34:54].copy()
    open_hand = np.zeros(20, dtype=np.float64)
    actions: list[np.ndarray] = []
    for name, target_right in (
        ("right_release", current_right),
        ("right_withdraw", np.asarray(right_withdraw, dtype=np.float64)),
        ("settle", np.asarray(right_withdraw, dtype=np.float64)),
    ):
        target_left_hand = open_hand if name == "settle" else current_left_hand
        actions.extend(
            _phase_actions(
                current_right,
                current_left,
                target_right,
                current_left,
                current_right_hand,
                current_left_hand,
                open_hand,
                target_left_hand,
                _phase_frames(name),
            )
        )
        current_right = target_right
        current_right_hand = open_hand
        current_left_hand = target_left_hand
    return actions


def _seat_actions(
    start_action: np.ndarray, right_seat: np.ndarray
) -> list[np.ndarray]:
    """Hold the live seat pose while unloading the right-hand grasp."""
    current = np.asarray(start_action, dtype=np.float64).reshape(54)
    return list(
        _phase_actions(
            current[:7],
            current[7:14],
            np.asarray(right_seat, dtype=np.float64).reshape(7),
            current[7:14],
            current[14:34],
            current[34:54],
            _RIGHT_SEAT_GRIP,
            current[34:54],
            _phase_frames("seat_in_drawer"),
        )
    )


def _stack_numeric(
    transitions: list[dict[str, Any]], key: str, *, next_value: bool = False
) -> np.ndarray:
    obs_key = "next" if next_value else "current"
    return np.stack(
        [np.asarray(item[obs_key][key]) for item in transitions], axis=0
    )


def _physical_gate_telemetry_arrays(
    task_trace: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Persist action commands and post-action gates on explicit timebases."""
    if not task_trace:
        raise ValueError("FU006 physical-gate telemetry is empty")
    phase_ids = {
        name: index
        for index, (name, _frames) in enumerate(_AUTOMATIC_PHASE_SCHEDULE)
    }
    arrays: dict[str, np.ndarray] = {
        "fu006_gate_step": np.asarray(
            [int(item["step"]) for item in task_trace], dtype=np.int32
        ),
        "fu006_gate_phase_id": np.asarray(
            [phase_ids[str(item["phase"])] for item in task_trace],
            dtype=np.int16,
        ),
    }
    for field in _PHYSICAL_GATE_TELEMETRY_FIELDS:
        if any(field not in item for item in task_trace):
            raise RuntimeError(f"FU006 physical-gate telemetry lacks {field}")
        prefix = (
            "fu006_gate_"
            if field in _ACTION_GATE_TELEMETRY_FIELDS
            else "next_fu006_gate_"
        )
        arrays[f"{prefix}{field}"] = np.stack(
            [np.asarray(item[field]) for item in task_trace]
        )
    return arrays


def _success_artifact_alignment(
    transitions: list[dict[str, Any]],
    task_trace: list[dict[str, Any]],
    reset_gate: dict[str, Any],
) -> dict[str, Any]:
    """Prove reset, action rows, post-action gates, and success share an index."""
    if not transitions:
        raise RuntimeError("FU006 success artifact has no transitions")
    count = len(transitions)
    if len(task_trace) != count:
        raise RuntimeError(
            "FU006 physical-gate telemetry length does not match transitions"
        )
    expected_steps = np.arange(1, count + 1, dtype=np.int64)
    actual_steps = np.asarray(
        [int(item["step"]) for item in task_trace], dtype=np.int64
    )
    if not np.array_equal(actual_steps, expected_steps):
        raise RuntimeError("FU006 gate steps do not align with action rows")
    if any(
        str(item["phase"]) != _automatic_phase_for_step(int(item["step"]))
        for item in task_trace
    ):
        raise RuntimeError("FU006 gate phases do not align with action steps")
    if any(np.asarray(item["action"]).shape != (54,) for item in transitions):
        raise RuntimeError("FU006 success artifact requires one 54-D action per step")

    if (
        reset_gate.get("name") != _FIRST_PHYSICAL_GATE
        or reset_gate.get("passed") is not True
    ):
        raise RuntimeError("FU006 success artifact lacks its passed first gate")
    initial_joint = np.asarray(
        transitions[0]["current"]["state"]["drawer_joint_state"],
        dtype=np.float64,
    ).reshape(-1)
    if initial_joint.size < 1 or not np.isclose(
        initial_joint[0], float(reset_gate["physical_q_m"]), atol=1.0e-6
    ):
        raise RuntimeError("FU006 first gate does not align with row-0 state")

    succeeded = np.asarray(
        [bool(item["succeed"]) for item in transitions], dtype=np.bool_
    )
    terminated = np.asarray(
        [bool(item["terminated"]) for item in transitions], dtype=np.bool_
    )
    truncated = np.asarray(
        [bool(item["truncated"]) for item in transitions], dtype=np.bool_
    )
    gate_success = np.asarray(
        [bool(item["fu006_success_now"]) for item in task_trace],
        dtype=np.bool_,
    )
    if not np.array_equal(gate_success, succeeded):
        raise RuntimeError("FU006 post-action success gate is misaligned")
    if (
        np.any(succeeded[:-1])
        or not bool(succeeded[-1])
        or np.any(terminated[:-1])
        or not bool(terminated[-1])
        or np.any(truncated)
    ):
        raise RuntimeError("FU006 success must be the sole terminal final row")
    return {
        "action_count": count,
        "action_step_origin": 1,
        "first_physical_gate": _FIRST_PHYSICAL_GATE,
        "reset_physical_q_m": float(reset_gate["physical_q_m"]),
        "telemetry_timebase": "post_action_next_state",
        "terminal_success_step": count,
    }


def _physical_gate_progress(
    task_trace: list[dict[str, Any]], reset_gate: dict[str, Any]
) -> dict[str, Any]:
    """Summarize the first unmet ordered physical gate from live telemetry."""
    predicates = (
        (
            _FIRST_PHYSICAL_GATE,
            lambda _item: bool(reset_gate.get("passed", False)),
            0,
        ),
        (
            "left_handle_contact",
            lambda item: bool(item["fu006_left_hold_seen"]),
            None,
        ),
        (
            "drawer_open_while_held",
            lambda item: bool(
                item["fu006_left_hold_now"] and item["fu006_drawer_open"]
            ),
            None,
        ),
        (
            "charger_contained_after_hold",
            lambda item: bool(item["fu006_contained_seen"]),
            None,
        ),
        (
            "right_release_after_containment",
            lambda item: bool(item["fu006_release_seen"]),
            None,
        ),
        (
            "left_release_before_stability",
            lambda item: bool(
                item["fu006_release_seen"]
                and item["fu006_left_released_now"]
            ),
            None,
        ),
        (
            "terminal_success",
            lambda item: bool(item["fu006_success_now"]),
            None,
        ),
    )
    gates: dict[str, dict[str, Any]] = {}
    blocked = False
    first_unmet: str | None = None
    last_step = -1
    for name, predicate, fixed_step in predicates:
        first_step = fixed_step
        if first_step is None:
            first_step = next(
                (
                    int(item["step"])
                    for item in task_trace
                    if predicate(item)
                ),
                None,
            )
        passed = first_step is not None
        if blocked and passed:
            raise RuntimeError(
                f"FU006 physical gate {name} passed before an earlier gate"
            )
        if passed and int(first_step) < last_step:
            raise RuntimeError("FU006 physical gate steps are out of order")
        if passed:
            last_step = int(first_step)
        else:
            blocked = True
            if first_unmet is None:
                first_unmet = name
        gates[name] = {"first_step": first_step, "passed": passed}
    return {
        "first_unmet_gate": first_unmet,
        "ordered_gates": [name for name, _predicate, _step in predicates],
        "passed_gate_count": sum(
            int(record["passed"]) for record in gates.values()
        ),
        "gates": gates,
    }


def _write_episode(
    output: Path,
    transitions: list[dict[str, Any]],
    runtime: dict[str, Any],
    seed: int,
    *,
    reset_gate: dict[str, Any],
    task_trace: list[dict[str, Any]],
) -> Path:
    alignment = _success_artifact_alignment(
        transitions, task_trace, reset_gate
    )
    gate_progress = _physical_gate_progress(task_trace, reset_gate)
    import zarr
    from handson.data.episode_store import ZarrEpisodeStore

    output.mkdir(parents=True, exist_ok=False)
    current_truth = _truth_fields(transitions[0]["current"])
    next_truth = _truth_fields(transitions[0]["next"])
    common_truth = sorted(set(current_truth) & set(next_truth))
    if "physx_robot_joint_target" not in common_truth:
        raise RuntimeError("FU006 collection lacks low-level PhysX drive targets")
    physical_gate_telemetry = _physical_gate_telemetry_arrays(task_trace)
    episode_data: dict[str, np.ndarray] = {
        "action": np.stack([item["action"] for item in transitions]),
        "state": np.stack(
            [_state_vector(item["current"]) for item in transitions]
        ),
        "next_state": np.stack(
            [_state_vector(item["next"]) for item in transitions]
        ),
        "reward": np.asarray([item["reward"] for item in transitions], dtype=np.float32),
        "terminal": np.asarray([item["terminated"] for item in transitions], dtype=np.uint8),
        "truncated": np.asarray([item["truncated"] for item in transitions], dtype=np.uint8),
        "succeed": np.asarray([item["succeed"] for item in transitions], dtype=np.uint8),
        "timestamp": np.arange(len(transitions), dtype=np.float64) / 30.0,
        "fu006_task_state": np.stack(
            [item["current"]["state"]["fu006_task_state"] for item in transitions]
        ),
        "next_fu006_task_state": np.stack(
            [item["next"]["state"]["fu006_task_state"] for item in transitions]
        ),
        **physical_gate_telemetry,
    }
    for key in common_truth:
        try:
            episode_data[key] = _stack_numeric(transitions, key)
            episode_data[f"next_{key}"] = _stack_numeric(
                transitions, key, next_value=True
            )
        except (TypeError, ValueError):
            continue
    for camera in CAMERA_KEYS:
        episode_data[f"image_{camera}"] = np.stack(
            [_camera_frame(item["current"], camera) for item in transitions]
        )
        episode_data[f"next_image_{camera}"] = np.stack(
            [_camera_frame(item["next"], camera) for item in transitions]
        )

    store = zarr.DirectoryStore(str(output / "replay.zarr"))
    episode = ZarrEpisodeStore.create_empty(storage=store)
    episode.root.attrs.update(
        {
            "format": "handson.fu006.physx_vertical_slice",
            "version": 1,
            "task_name": TASK_NAME,
            "task_complete": False,
            "episode_reset_seed": int(seed),
            "runtime_contract_json": json.dumps(runtime, sort_keys=True),
            "action_semantics": "exact_raw_environment_action_executed",
            "action_zero_is_validated_boundary": True,
            "state_writes_after_action_zero": False,
            "fresh_process_full_trajectory_replay_required": True,
            "scaled_drawer_reset_gate_json": json.dumps(
                reset_gate, sort_keys=True
            ),
            "physical_gate_alignment_json": json.dumps(
                alignment, sort_keys=True
            ),
            "physical_gate_progress_json": json.dumps(
                gate_progress, sort_keys=True
            ),
            "physical_gate_telemetry_json": json.dumps(
                {
                    "action_aligned_fields": list(
                        _ACTION_GATE_TELEMETRY_FIELDS
                    ),
                    "first_physical_gate": _FIRST_PHYSICAL_GATE,
                    "next_state_aligned_fields": list(
                        _NEXT_GATE_TELEMETRY_FIELDS
                    ),
                    "phase_ids": {
                        name: index
                        for index, (name, _frames) in enumerate(
                            _AUTOMATIC_PHASE_SCHEDULE
                        )
                    },
                    "fields": sorted(physical_gate_telemetry),
                },
                sort_keys=True,
            ),
        }
    )
    episode.append_episode(episode_data, compressors="disk")
    manifest = {
        **runtime,
        "format": "handson.fu006.physx_vertical_slice",
        "version": 1,
        "task_complete": False,
        "episode_reset_seed": int(seed),
        "step_count": len(transitions),
        "action_shape": list(episode_data["action"].shape),
        "fields": sorted(episode_data),
        "success": bool(transitions[-1]["succeed"]),
        "state_writes_after_action_zero": False,
        "fresh_replay_required": True,
        "physical_gate_alignment": alignment,
        "physical_gate_progress": gate_progress,
        "scaled_drawer_reset_gate": reset_gate,
        "physical_gate_telemetry": {
            "action_aligned_fields": list(_ACTION_GATE_TELEMETRY_FIELDS),
            "first_physical_gate": _FIRST_PHYSICAL_GATE,
            "next_state_aligned_fields": list(
                _NEXT_GATE_TELEMETRY_FIELDS
            ),
            "phase_ids": {
                name: index
                for index, (name, _frames) in enumerate(
                    _AUTOMATIC_PHASE_SCHEDULE
                )
            },
            "fields": sorted(physical_gate_telemetry),
        },
    }
    (output / "episode_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output / "replay.zarr"


def _preview(output: Path, observation: dict[str, Any]) -> dict[str, str]:
    import cv2

    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for camera in CAMERA_KEYS:
        path = output / f"preview_{camera}.png"
        rgb = _camera_frame(observation, camera)
        if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"could not save {camera} preview")
        paths[camera] = str(path)
    return paths


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def _write_collection_failure_report(
    output: Path,
    *,
    runtime: dict[str, Any],
    seed: int,
    reset_gate: dict[str, Any],
    actions_generated: int,
    transitions: list[dict[str, Any]],
    initial_observation: dict[str, Any],
    waypoints: dict[str, np.ndarray],
    observation: dict[str, Any],
    final_info: dict[str, Any],
    task_trace: list[dict[str, Any]],
) -> Path:
    """Persist physical failure truth before Isaac teardown can block stderr."""
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_name(f"{output.name}_failure_report.json")
    payload = {
        **runtime,
        "format": "handson.fu006.collection_failure",
        "version": 1,
        "task_complete": False,
        "success": False,
        "episode_reset_seed": int(seed),
        "physical_gate_progress": _physical_gate_progress(
            task_trace, reset_gate
        ),
        "actions_generated": int(actions_generated),
        "steps_executed": len(transitions),
        "phase_schedule": [
            {"name": name, "frames": frames}
            for name, frames in _AUTOMATIC_PHASE_SCHEDULE
        ],
        "initial_state": {
            key: _jsonable_value(initial_observation["state"][key])
            for key in STATE_KEYS
        },
        "policy_waypoints": {
            key: _jsonable_value(value) for key, value in waypoints.items()
        },
        "final_task_info": {
            key: _jsonable_value(value)
            for key, value in final_info.items()
            if key.startswith("fu006_")
        },
        "final_state": {
            key: _jsonable_value(observation["state"][key])
            for key in STATE_KEYS
        },
        "task_trace": task_trace,
        "state_writes_after_action_zero": False,
        "replay_eligible": False,
    }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True), flush=True)
    return report_path


def run_probe(args: argparse.Namespace) -> int:
    runtime = _runtime_contract()
    env = _new_env(args.seed)
    try:
        observation, reset_info = env.reset(seed=args.seed)
        reset_gate = _scaled_drawer_reset_gate(env, observation)
        initial_state = np.asarray(
            observation["state"]["fu006_task_state"], dtype=np.float64
        )
        if bool(initial_state[12]) or bool(initial_state[13]):
            raise RuntimeError("FU006 reset is already successful or failed")
        max_pose_drift = 0.0
        start_charger = np.asarray(observation["state"]["charger_pose"][:3]).copy()
        info: dict[str, Any] = {}
        for _ in range(args.steps):
            observation, _reward, terminated, truncated, info = env.step(
                np.zeros(54, dtype=np.float32)
            )
            charger = np.asarray(observation["state"]["charger_pose"][:3])
            max_pose_drift = max(
                max_pose_drift, float(np.linalg.norm(charger - start_charger))
            )
            if terminated or truncated:
                raise RuntimeError(f"FU006 zero action terminated early: {info}")
        previews = _preview(args.output, observation)
        report = {
            **runtime,
            "mode": "probe",
            "seed": args.seed,
            "steps": args.steps,
            "tensor_views_valid": True,
            "scaled_drawer_reset_gate": reset_gate,
            "max_charger_translation_drift_m": max_pose_drift,
            "task_info": {
                key: value
                for key, value in info.items()
                if key.startswith("fu006_")
            },
            "previews": previews,
            "reset_info_keys": sorted(reset_info),
        }
        (args.output / "probe_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, sort_keys=True, default=str), flush=True)
        return 0
    finally:
        env.close()


def _live_tensor_contract(
    raw: Any,
    observation: dict[str, Any],
    info: dict[str, Any],
) -> dict[str, Any]:
    """Read every FU006 task-owned PhysX view without mutating scene state."""

    contact_views: dict[str, dict[str, Any]] = {}
    for label, attribute in (
        ("left_hand_drawer", "_fu006_left_drawer_contact_view"),
        ("right_hand_charger", "_fu006_right_charger_contact_view"),
    ):
        view = getattr(raw, attribute, None)
        if view is None:
            raise RuntimeError(f"FU006 {label} contact view is missing")
        sensor_count = int(view.sensor_count)
        filter_count = int(view.filter_count)
        if sensor_count < 1 or filter_count < 1:
            raise RuntimeError(
                f"FU006 {label} contact view is empty: "
                f"sensor_count={sensor_count}, filter_count={filter_count}"
            )
        contact_views[label] = {
            "sensor_count": sensor_count,
            "filter_count": filter_count,
        }
    print("[FU006 hard-reset canary] Contact view metadata is valid.", flush=True)

    simulation_view = getattr(raw, "_task_physics_sim_view", None)
    if simulation_view is None:
        raise RuntimeError("FU006 task PhysX simulation view is missing")
    print("[FU006 hard-reset canary] Task simulation view is present.", flush=True)
    charger_pose = raw._physics_body_pose("fu006_charger")
    if charger_pose is None or not np.all(np.isfinite(charger_pose)):
        raise RuntimeError("FU006 charger rigid-body view is unreadable")
    print("[FU006 hard-reset canary] Charger body view is readable.", flush=True)
    drawer_joint_state = np.asarray(
        observation["state"]["drawer_joint_state"],
        dtype=np.float64,
    ).reshape(2)
    if not np.all(np.isfinite(drawer_joint_state)):
        raise RuntimeError("FU006 production drawer observation is unreadable")
    print("[FU006 hard-reset canary] Drawer observation is readable.", flush=True)
    required_metrics = (
        "fu006_drawer_joint_position",
        "fu006_drawer_joint_velocity",
        "fu006_left_contact_force",
        "fu006_right_contact_force",
    )
    nonfinite_metrics = [
        name
        for name in required_metrics
        if not np.isfinite(float(info.get(name, np.nan)))
    ]
    if nonfinite_metrics:
        raise RuntimeError(
            f"FU006 success-gate metrics are non-finite: {nonfinite_metrics}"
        )
    print("[FU006 hard-reset canary] FU006 gate metrics are readable.", flush=True)
    return {
        "scene_generation": int(raw._physx_scene_generation),
        "contact_views": contact_views,
        "task_body_view_count": len(raw._task_body_views),
        "task_articulation_view_count": len(raw._task_articulation_views),
        "task_dof_view_count": len(raw._task_dof_views),
        "charger_pose_wxyz": np.asarray(charger_pose).tolist(),
        "drawer_joint_position": float(drawer_joint_state[0]),
        "drawer_joint_velocity": float(drawer_joint_state[1]),
        "success_gate_metrics": {
            name: _jsonable_value(info[name])
            for name in required_metrics
        },
    }


def run_hard_reset_canary(args: argparse.Namespace) -> int:
    """Prove FU006 tensor resources survive repeated in-process Scene rebuilds."""

    runtime = _runtime_contract()
    env = _new_env(args.seed, image_obs=False)
    raw = getattr(env, "unwrapped", env)
    resets: list[dict[str, Any]] = []
    try:
        observation, _reset_info = env.reset(seed=args.seed)
        for _ in range(max(1, args.steps)):
            observation, _reward, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            if terminated or truncated:
                raise RuntimeError(
                    f"FU006 canary terminated before the first rebuild: {info}"
                )
        print(
            "[FU006 hard-reset canary] Reading initial tensor contract...",
            flush=True,
        )
        initial_contract = _live_tensor_contract(raw, observation, info)
        print(
            "[FU006 hard-reset canary] Initial tensor contract is readable.",
            flush=True,
        )
        resets.append(
            {
                "seed": int(args.seed),
                "hard_reset_result": None,
                "tensor_contract": initial_contract,
            }
        )

        for ordinal in range(1, args.reset_trials + 1):
            seed = int(args.seed + ordinal)
            print(
                f"[FU006 hard-reset canary] Starting rebuild {ordinal}...",
                flush=True,
            )
            hard_reset_result = raw.hard_reset_physx_scene(seed=seed)
            if hard_reset_result.get("scene_rebuilt") is not True:
                raise RuntimeError(
                    f"FU006 hard reset {ordinal} did not confirm a rebuilt Scene"
                )
            task_cleanup = dict(
                hard_reset_result.get("pre_rebuild_cleanup", {})
            ).get("task_physics_cleanup")
            if not isinstance(task_cleanup, dict):
                raise RuntimeError(
                    f"FU006 hard reset {ordinal} lacks task-physics cleanup evidence"
                )
            if int(task_cleanup.get("tensor_view_handles_dropped", 0)) < 1:
                raise RuntimeError(
                    f"FU006 hard reset {ordinal} did not drop tensor views"
                )
            if task_cleanup.get("simulation_view_dropped") is not True:
                raise RuntimeError(
                    f"FU006 hard reset {ordinal} did not drop its simulation view"
                )

            observation, _reset_info = env.reset(seed=seed)
            for _ in range(max(1, args.steps)):
                observation, _reward, terminated, truncated, info = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )
                if terminated or truncated:
                    raise RuntimeError(
                        f"FU006 canary terminated after rebuild {ordinal}: {info}"
                    )
            contract = _live_tensor_contract(raw, observation, info)
            if contract["scene_generation"] != ordinal:
                raise RuntimeError(
                    "FU006 Scene generation mismatch after rebuild "
                    f"{ordinal}: {contract['scene_generation']}"
                )
            resets.append(
                {
                    "seed": seed,
                    "hard_reset_result": hard_reset_result,
                    "tensor_contract": contract,
                }
            )

        report = {
            **runtime,
            "mode": "hard-reset-canary",
            "status": "pass",
            "reset_trials": int(args.reset_trials),
            "zero_action_steps_per_scene": max(1, int(args.steps)),
            "scenes": resets,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        report_path = args.output / "hard_reset_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, sort_keys=True, default=str), flush=True)
        return 0
    finally:
        raw._fast_shutdown_on_close = True
        env.close()


def run_collect(args: argparse.Namespace) -> int:
    runtime = _runtime_contract()
    env = _new_env(args.seed)
    try:
        observation, _reset_info = env.reset(seed=args.seed)
        reset_gate = _scaled_drawer_reset_gate(env, observation)
        initial_observation = {
            "state": {
                key: np.asarray(observation["state"][key]).copy()
                for key in STATE_KEYS
            }
        }
        measured_right_home = _measured_palm_pose(env, "right")
        measured_left_home = _measured_palm_pose(env, "left")
        waypoints = _asset_frame_waypoints(
            observation,
            measured_right_home=measured_right_home,
            measured_left_home=measured_left_home,
        )
        actions = automatic_prefix_actions(
            observation,
            measured_right_home=measured_right_home,
            measured_left_home=measured_left_home,
        )
        prefix_steps = sum(
            frames for _name, frames in _AUTOMATIC_PHASE_SCHEDULE[:4]
        )
        flatten_end_step = prefix_steps + _phase_frames("right_flatten")
        stage_end_step = flatten_end_step + _phase_frames(
            "right_stage_outside_side"
        )
        lift_end_step = stage_end_step + _phase_frames(
            "right_lift_outside_side"
        )
        left_approach_end_step = lift_end_step + sum(
            _phase_frames(name)
            for name in ("left_stage", "left_approach")
        )
        left_correction_end_step = left_approach_end_step + _phase_frames(
            "left_contact_correction"
        )
        left_hold_end_step = left_correction_end_step + sum(
            _phase_frames(name)
            for name in ("left_contact_refine", "left_close", "left_pull")
        )
        cross_end_step = left_hold_end_step + _phase_frames("cross_side_above")
        align_end_step = cross_end_step + _phase_frames("align_above_drawer")
        lower_end_step = align_end_step + _phase_frames("lower_into_drawer")
        seat_end_step = lower_end_step + _phase_frames("seat_in_drawer")
        transitions: list[dict[str, Any]] = []
        task_trace: list[dict[str, Any]] = []
        info: dict[str, Any] = {}
        step_index = 0
        while step_index < len(actions):
            action = actions[step_index]
            step_index += 1
            current = observation
            observation, reward, terminated, truncated, info = env.step(action)
            measured_right = _measured_palm_pose(env, "right")
            measured_left = _measured_palm_pose(env, "left")
            measured_right_fingertips = _measured_fingertips(env, "right")
            measured_left_fingertips = _measured_fingertips(env, "left")
            state = observation["state"]
            charger_pose = np.asarray(state["charger_pose"], dtype=np.float64)
            drawer_pose = np.asarray(state["drawer_pose"], dtype=np.float64)
            handle_contact = _drawer_handle_contact_world(drawer_pose)
            right_fingertip = _nominal_fingertip_world(
                measured_right, _RIGHT_POWER_MAIN_FINGERTIP_LOCAL
            )
            left_fingertip = _nominal_fingertip_world(
                measured_left, _LEFT_STABILIZE_MAIN_FINGERTIP_LOCAL
            )
            task_trace.append(
                {
                    "step": step_index,
                    "phase": _automatic_phase_for_step(step_index),
                    "commanded_right_palm_pose": _jsonable_value(action[:7]),
                    "commanded_left_palm_pose": _jsonable_value(action[7:14]),
                    "measured_right_palm_pose": _jsonable_value(measured_right),
                    "measured_left_palm_pose": _jsonable_value(measured_left),
                    "right_palm_command_error_m": float(
                        np.linalg.norm(measured_right[:3] - action[:3])
                    ),
                    "left_palm_command_error_m": float(
                        np.linalg.norm(measured_left[:3] - action[7:10])
                    ),
                    "nominal_right_fingertip_to_charger_m": float(
                        np.linalg.norm(right_fingertip - charger_pose[:3])
                    ),
                    "nominal_left_fingertip_to_handle_m": float(
                        np.linalg.norm(left_fingertip - handle_contact)
                    ),
                    "actual_right_fingertip_to_charger_m": float(
                        min(
                            np.linalg.norm(value - charger_pose[:3])
                            for value in measured_right_fingertips.values()
                        )
                    ),
                    "actual_left_fingertip_to_handle_m": float(
                        min(
                            np.linalg.norm(value - handle_contact)
                            for value in measured_left_fingertips.values()
                        )
                    ),
                    "measured_right_fingertips": _jsonable_value(
                        measured_right_fingertips
                    ),
                    "measured_left_fingertips": _jsonable_value(
                        measured_left_fingertips
                    ),
                    "charger_pose": _jsonable_value(charger_pose),
                    "drawer_pose": _jsonable_value(drawer_pose),
                    "drawer_handle_contact_world": _jsonable_value(
                        handle_contact
                    ),
                    **{
                        key: _jsonable_value(value)
                        for key, value in info.items()
                        if key.startswith("fu006_")
                    },
                }
            )
            transitions.append(
                {
                    "current": current,
                    "next": observation,
                    "action": np.asarray(action, dtype=np.float32),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "succeed": bool(info.get("succeed", False)),
                }
            )
            if terminated or truncated:
                break
            if step_index == prefix_steps:
                flatten = _flatten_grasp_waypoint(
                    observation, measured_right
                )
                waypoints["right_flatten"] = flatten
                last = np.asarray(action, dtype=np.float64).reshape(54)
                actions.extend(
                    _phase_actions(
                        last[:7],
                        last[7:14],
                        flatten,
                        last[7:14],
                        last[14:34],
                        last[34:54],
                        last[14:34],
                        last[34:54],
                        _phase_frames("right_flatten"),
                    )
                )
            elif step_index == flatten_end_step:
                adaptive = _adaptive_carry_waypoints(
                    observation, measured_right
                )
                waypoints.update(adaptive)
                last = np.asarray(action, dtype=np.float64).reshape(54)
                actions.extend(
                    _phase_actions(
                        last[:7],
                        last[7:14],
                        adaptive["right_stage"],
                        last[7:14],
                        last[14:34],
                        last[34:54],
                        last[14:34],
                        last[34:54],
                        _phase_frames("right_stage_outside_side"),
                    )
                )
            elif step_index == stage_end_step:
                adaptive = _adaptive_carry_waypoints(
                    observation, measured_right
                )
                waypoints.update(adaptive)
                lift_frames = _phase_frames("right_lift_outside_side")
                actions.extend(
                    _adaptive_suffix_actions(action, adaptive)[:lift_frames]
                )
            elif step_index == lift_end_step:
                actions.extend(_left_approach_actions(action, waypoints))
            elif step_index == left_approach_end_step:
                left_hold = _left_contact_correction_waypoint(
                    observation,
                    action[7:14],
                    measured_left,
                    measured_left_fingertips,
                )
                waypoints["left_hold_live"] = left_hold
                last = np.asarray(action, dtype=np.float64).reshape(54)
                actions.extend(
                    _phase_actions(
                        last[:7],
                        last[7:14],
                        last[:7],
                        left_hold,
                        last[14:34],
                        last[34:54],
                        last[14:34],
                        last[34:54],
                        _phase_frames("left_contact_correction"),
                    )
                )
            elif step_index == left_correction_end_step:
                refined_left_hold = _left_contact_correction_waypoint(
                    observation,
                    action[7:14],
                    measured_left,
                    measured_left_fingertips,
                )
                waypoints["left_hold_refined_live"] = refined_left_hold
                left_pull = refined_left_hold.copy()
                left_pull[0] -= _LEFT_HANDLE_PULL_M
                waypoints["left_pull_live"] = left_pull
                actions.extend(
                    _left_contact_actions(action, refined_left_hold)
                )
            elif step_index == left_hold_end_step:
                cavity_turn = _cavity_turn_grasp_waypoint(
                    observation, measured_right
                )
                adaptive = _adaptive_carry_waypoints(
                    observation, cavity_turn
                )
                waypoints["right_cavity_turn_virtual"] = cavity_turn
                for name in ("drawer_above", "drawer_inside", "right_withdraw"):
                    waypoints[name] = adaptive[name]
                cross_frames = _phase_frames("cross_side_above")
                actions.extend(
                    _adaptive_suffix_actions(
                        action, adaptive, include_lift=False
                    )[:cross_frames]
                )
            elif step_index == cross_end_step:
                align = _closed_loop_grasp_translation_waypoint(
                    observation, action[:7], horizontal_only=True
                )
                waypoints["right_align_above_live"] = align
                last = np.asarray(action, dtype=np.float64).reshape(54)
                actions.extend(
                    _phase_actions(
                        last[:7],
                        last[7:14],
                        align,
                        last[7:14],
                        last[14:34],
                        last[34:54],
                        last[14:34],
                        last[34:54],
                        _phase_frames("align_above_drawer"),
                    )
                )
            elif step_index == align_end_step:
                lower = _closed_loop_grasp_translation_waypoint(
                    observation, action[:7], horizontal_only=False
                )
                waypoints["right_lower_live"] = lower
                last = np.asarray(action, dtype=np.float64).reshape(54)
                actions.extend(
                    _phase_actions(
                        last[:7],
                        last[7:14],
                        lower,
                        last[7:14],
                        last[14:34],
                        last[34:54],
                        last[14:34],
                        last[34:54],
                        _phase_frames("lower_into_drawer"),
                    )
                )
            elif step_index == lower_end_step:
                seat = _closed_loop_grasp_translation_waypoint(
                    observation,
                    action[:7],
                    horizontal_only=False,
                    translation_limit_m=_FINAL_RESIDUAL_CORRECTION_LIMIT_M,
                )
                waypoints["right_seat_live"] = seat
                actions.extend(_seat_actions(action, seat))
            elif step_index == seat_end_step:
                actions.extend(
                    _release_suffix_actions(action, waypoints["right_withdraw"])
                )
        if not transitions or not transitions[-1]["succeed"]:
            detail = {
                key: value
                for key, value in info.items()
                if key.startswith("fu006_")
            }
            report_path = _write_collection_failure_report(
                args.output,
                runtime=runtime,
                seed=args.seed,
                reset_gate=reset_gate,
                actions_generated=len(actions),
                transitions=transitions,
                initial_observation=initial_observation,
                waypoints=waypoints,
                observation=observation,
                final_info=info,
                task_trace=task_trace,
            )
            raise RuntimeError(
                "FU006 automatic physical trajectory did not reach success: "
                + json.dumps(detail, sort_keys=True, default=str)
                + f"; diagnostic={report_path}"
            )
        replay = _write_episode(
            args.output,
            transitions,
            runtime,
            args.seed,
            reset_gate=reset_gate,
            task_trace=task_trace,
        )
        _preview(args.output, observation)
        print(json.dumps({"episode": str(replay), "steps": len(transitions)}))
        return 0
    finally:
        env.close()


def run_replay(args: argparse.Namespace) -> int:
    import zarr

    runtime = _runtime_contract()
    episode_dir = args.episode.resolve()
    manifest = json.loads((episode_dir / "episode_manifest.json").read_text())
    if manifest["runtime_family"] != "isaac_sim_5_1":
        raise RuntimeError("FU006 replay manifest is not Isaac Sim 5.1")
    if manifest["hashes"] != runtime["hashes"]:
        raise RuntimeError("FU006 replay code/config/asset hashes do not match")
    root = zarr.open_group(str(episode_dir / "replay.zarr"), mode="r")
    actions = np.asarray(root["data/action"])
    if actions.ndim != 2 or actions.shape[1] != 54:
        raise RuntimeError(f"FU006 replay action shape is invalid: {actions.shape}")
    seed = int(manifest["episode_reset_seed"])
    env = _new_env(seed)
    try:
        observation, _reset_info = env.reset(seed=seed)
        final_info: dict[str, Any] = {}
        for action in actions:
            observation, _reward, terminated, truncated, final_info = env.step(
                np.asarray(action, dtype=np.float32)
            )
            if terminated or truncated:
                break
        success = bool(final_info.get("succeed", False))
        report = {
            **runtime,
            "format": "handson.fu006.fresh_replay",
            "version": 1,
            "fresh_process": True,
            "full_trajectory": True,
            "state_restores_after_action_zero": 0,
            "step_count_recorded": int(actions.shape[0]),
            "step_count_replayed": int(env.env_step),
            "success": success,
            "final_task_info": {
                key: value
                for key, value in final_info.items()
                if key.startswith("fu006_")
            },
        }
        report_path = episode_dir / "fresh_replay_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, sort_keys=True, default=str), flush=True)
        return 0 if success and env.env_step == actions.shape[0] else 2
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("probe", "hard-reset-canary", "collect", "replay"),
    )
    parser.add_argument("--seed", type=int, default=320006)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--reset-trials", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("fu006_live"))
    parser.add_argument("--episode", type=Path)
    args = parser.parse_args()
    _configure_task_private_isaac_runtime(args.output)
    if args.mode == "probe":
        return run_probe(args)
    if args.mode == "hard-reset-canary":
        return run_hard_reset_canary(args)
    if args.mode == "collect":
        return run_collect(args)
    if args.episode is None:
        parser.error("replay requires --episode")
    return run_replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
