#!/usr/bin/env python3
"""
Replay previously recorded Zarr demos under the policy interface and save the
result as a fresh Zarr episode plus MP4 videos.

For each input demo directory containing a `replay.zarr`, the script writes:
- `<exp_name>_demo_<index>_replay_<timestamp>/replay.zarr/`
- exactly three canonical videos: `main.mp4`, `wrist_left.mp4`, `wrist_right.mp4`
- matching `camera_info.json` sidecars with MuJoCo view pose and wrist fisheye
  intrinsics scaled to each saved MP4 frame size

The script plays the recorded action sequence on a freshly reset environment
constructed via `TaskConfig.get_native_environment(policy_mode=True, ...)`. With
`--randomize=True`, the underlying task draws a new preset camera from
`handson/sim/envs/replay_cameras.npy` plus randomized lighting/texture, which
makes this useful for visual augmentation of an existing dataset.

When `--restore_state=True` (default), `state[0]` from the input zarr is
sliced by the task's `proprio_keys` and the env's initial object poses +
table height are patched back to match the recording. The per-task restorers
live in `handson.tasks.state_restorers` for MuJoCo and on the native task
environment for IsaacLab.
"""

import copy
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "handson"
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import numpy as np
import zarr
from absl import app, flags
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from handson.data.camera_info import (
    DEFAULT_FISHEYE_INTRINSICS_PATH,
    describe_camera_views,
    write_video_camera_info,
)
from handson.data.canonical_videos import (
    CANONICAL_CAMERA_KEYS,
    observation_image,
    postprocess_canonical_image,
    select_canonical_camera_sources,
    select_canonical_camera_sources_for_trajectory,
)
from handson.data.depth_capture import collect_depth_frames, write_depth_outputs
from handson.data.episode_store import ZarrEpisodeStore
from handson.data.recording_fields import (
    declared_recording_field_names,
    resolve_environment_recording_schema,
    stack_numeric_observation_fields,
    stack_transition_numeric_observation_fields,
    write_recording_schema,
)
from handson.data.video_writer import Mp4VideoWriter
from handson.data.simplified_post_ik import (
    declares_simplified_post_ik,
    load_validated_simplified_arm_plan,
    simplified_arm_plan_errors,
)
from handson.runtime.collection_support import (
    BC011_COLLECTION_ANGLE_GUIDANCE_ENV,
)
from handson.sim.isaaclab.replay_visuals import (
    reapply_reference_robot_appearance_for_replay,
)
from handson.tasks.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "exp_name",
    "water_plant",
    "Task name, such as water_plant.",
)
flags.DEFINE_string(
    "input_dir",
    "./",
    "Directory containing recorded demo folders (each holds a replay.zarr).",
)
flags.DEFINE_string(
    "out_dir",
    "./replay_output",
    "Output base directory for the new zarr and videos.",
)
flags.DEFINE_integer("video_fps", 30, "FPS for saved MP4 videos")
flags.DEFINE_integer(
    "video_crf",
    21,
    "H.264 constant-quality value; lower is higher quality.",
)
flags.DEFINE_integer(
    "render_width",
    0,
    "Override Isaac main-camera render width; 0 keeps the task default.",
)
flags.DEFINE_integer(
    "render_height",
    0,
    "Override Isaac main-camera render height; 0 keeps the task default.",
)
flags.DEFINE_integer(
    "wrist_render_width",
    0,
    "Override Isaac wrist-camera render width; 0 keeps the task default.",
)
flags.DEFINE_integer(
    "wrist_render_height",
    0,
    "Override Isaac wrist-camera render height; 0 keeps the task default.",
)
flags.DEFINE_bool(
    "stream_videos",
    False,
    "Encode canonical videos during replay and discard in-memory RGB frames. "
    "Use this for high-resolution replay renders.",
)
flags.DEFINE_bool(
    "force_complete_arm_render",
    False,
    "Show complete robot arm/body visuals even when the source episode used "
    "deferred-arm collection. This also renders the head and wrist camera "
    "visibility groups separately so the main view keeps complete wrists "
    "without blocking the wrist sensors.",
)
flags.DEFINE_integer(
    "max_replay_steps",
    0,
    "Limit replayed control frames for render smoke tests; 0 runs all frames.",
)
flags.DEFINE_float(
    "data_fps",
    30,
    "Sampling frequency of recorded low-dim data in Hz (used to write timestamps)",
)
flags.DEFINE_bool(
    "randomize",
    True,
    "Enable environment randomization (random preset camera, lighting, texture) at reset",
)
flags.DEFINE_integer(
    "seed",
    0,
    "Base seed for the replay environment; the demo index is added per demo",
)
flags.DEFINE_integer(
    "extend_steps",
    0,
    "Extra steps to repeat the last action after the recorded trajectory ends. "
    "Default 0 is fine for demos recorded by record_demos_zarr.py (the saved "
    "trajectory ends on the succeed step). Set >0 only when replaying older "
    "data that was clipped before success.",
)
flags.DEFINE_bool(
    "save_failed",
    False,
    "Save the replayed demo even when the environment does not report success",
)
flags.DEFINE_bool(
    "restore_state",
    True,
    "Restore the recorded initial scene state (table height + object poses) from state[0]",
)
flags.DEFINE_bool(
    "state_guided",
    False,
    "For Isaac video rerendering, restore each recorded task-object state before "
    "capturing the next frame so PhysX contact divergence cannot move objects away "
    "from the source trajectory.",
)
flags.DEFINE_bool(
    "kinematic_visual_replay",
    False,
    "For Isaac video rerendering, render directly from each recorded state instead "
    "of stepping the action through PhysX. This is a visual reconstruction mode: "
    "task objects are restored exactly and robot visuals are driven from recorded "
    "tcp_pose/gripper_pose when the task exposes restore_recorded_visual_state().",
)
flags.DEFINE_bool(
    "physx_ground_truth_replay",
    False,
    "For Isaac recordings with physx_* fields, restore the recorded pre/post "
    "PhysX state for every action instead of resimulating the action. Entity "
    "and joint names are validated against recording_schema.json before replay.",
)
flags.DEFINE_bool(
    "physx_action_replay_audit",
    False,
    "For Isaac recordings with physx_* fields, restore only the source initial "
    "state, re-execute every recorded action through PhysX, and compare each "
    "resulting state with next_physx_*[t]. This measures action replay drift "
    "without correcting it.",
)
flags.DEFINE_bool(
    "physx_joint_target_replay_audit",
    False,
    "For Isaac recordings with physx_* fields, restore only the source initial "
    "state, apply each recorded next_physx_robot_joint_target directly to the "
    "position drives, advance PhysX, and compare the resulting state with the "
    "recording. This is RoboTwin-style low-level path replay without state "
    "teleportation; V8 recordings contain position targets, not velocity-drive "
    "targets.",
)
flags.DEFINE_bool(
    "physx_robot_path_replay_audit",
    False,
    "For Isaac recordings with physx_* fields, restore the source initial "
    "state, move only the recorded robot joint state along an interpolated "
    "path at every PhysX substep, and leave all task objects dynamic. This is "
    "the Isaac equivalent of a MuJoCo mocap-driven physical object replay.",
)
flags.DEFINE_bool(
    "recapture_physx_v4",
    False,
    "Save a fresh schema-v4 current/next PhysX trace while running "
    "--physx_action_replay_audit. Task objects remain dynamic after the "
    "single initial-state restore.",
)
flags.DEFINE_integer(
    "visual_replay_ik_iterations",
    8,
    "Number of IK/write iterations used by --kinematic_visual_replay to snap "
    "robot visuals toward recorded tcp_pose before rendering each frame.",
)
flags.DEFINE_bool(
    "save_depth",
    False,
    "Also render a depth_array per RGB camera and save <cam>_depth.npz + .mp4 alongside each video",
)
flags.DEFINE_bool(
    "save_camera_info",
    True,
    "Write JSON camera metadata sidecars next to saved MP4 videos.",
)
flags.DEFINE_bool(
    "physics_only",
    False,
    "Disable camera creation and video output. The replay still executes every "
    "recorded action through the selected physics backend and saves replay.zarr; "
    "use this for trajectory audits on headless workers without RTX rendering.",
)
flags.DEFINE_string(
    "fisheye_intrinsics_yaml",
    str(DEFAULT_FISHEYE_INTRINSICS_PATH),
    "OpenCV YAML intrinsics for the wrist fisheye camera.",
)
flags.DEFINE_bool(
    "camera_screen_effect",
    False,
    "(bimanual_photograph only) Enable the camera live-screen effect "
    "(live preview + shutter flash + frozen photo). When False, the screen "
    "geom is hidden entirely.",
)
flags.DEFINE_bool(
    "ipad_screen_effect",
    False,
    "(bimanual_unlock_ipad only) Enable the iPad unlock screen fade-in "
    "transition. When False, the screen flips to unlocked instantly on the "
    "final correct press.",
)
flags.DEFINE_enum(
    "sim_backend",
    "auto",
    ["auto", "mujoco", "isaacsim", "isaacsim_newton"],
    "Simulation backend to use. Auto reads collection_meta.json and falls back "
    "to MuJoCo only for legacy episodes without backend metadata.",
)


_PHYSX_REQUIRED_ROBOT_FIELDS = (
    "physx_robot_joint_q",
    "physx_robot_joint_qd",
)
_PHYSX_REPLAY_VERIFY_TOLERANCE = 1e-4
_PHYSX_REPLAY_VERIFY_FIELDS = {
    "physx_robot_joint_position": None,
    "physx_robot_joint_velocity": None,
    "physx_robot_joint_target": None,
    "physx_robot_root_state": None,
    "physx_tcp_state": None,
    "physx_task_rigid_body_state": None,
    "physx_task_articulation_root_state": None,
    "physx_task_articulation_dof_position": None,
    "physx_task_articulation_dof_velocity": None,
    "physx_deformable_position": None,
    "physx_deformable_velocity": None,
    "physx_command_tcp_pose": None,
    "physx_command_hand": None,
    "physx_pose_commanded": None,
    "physx_pose_filter_state": None,
    "physx_rng_seed": None,
    "physx_rng_mt19937_keys": None,
    "physx_rng_mt19937_position": None,
    "physx_rng_mt19937_has_gauss": None,
    "physx_rng_mt19937_cached_gaussian": None,
}


@dataclass(frozen=True)
class PhysxReplayTrajectory:
    """Named pre/post PhysX snapshots loaded from one recorded episode."""

    fields: dict[str, np.ndarray]
    schema: dict
    step_count: int

    def snapshot(
        self,
        step_index: int,
        *,
        after_action: bool,
    ) -> dict[str, np.ndarray]:
        if not 0 <= int(step_index) < self.step_count:
            raise IndexError(
                f"PhysX replay step {step_index} is outside [0, {self.step_count})"
            )
        snapshot = {}
        for name, values in self.fields.items():
            if after_action:
                if not name.startswith("next_physx_"):
                    continue
                output_name = name[len("next_") :]
            else:
                if not name.startswith("physx_"):
                    continue
                output_name = name
            snapshot[output_name] = np.asarray(values[int(step_index)])
        if not snapshot:
            phase = "post-action" if after_action else "pre-action"
            raise ValueError(f"PhysX replay has no {phase} fields")
        return snapshot


def _physx_replay_readback_field_errors(
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
) -> dict[str, float]:
    """Return the valid absolute write/readback error for every state field."""

    errors = {}
    for field, valid_field in _PHYSX_REPLAY_VERIFY_FIELDS.items():
        if field not in expected:
            continue
        if field not in actual or (
            valid_field is not None and valid_field not in expected
        ):
            errors[field] = float("inf")
            continue
        expected_values = np.asarray(expected[field], dtype=np.float64)
        actual_values = np.asarray(actual[field], dtype=np.float64)
        if expected_values.shape != actual_values.shape:
            errors[field] = float("inf")
            continue
        valid = (
            np.ones(expected_values.shape, dtype=bool)
            if valid_field is None
            else np.asarray(expected[valid_field], dtype=bool)
        )
        while valid.ndim < expected_values.ndim:
            valid = np.expand_dims(valid, axis=-1)
        try:
            valid = np.broadcast_to(valid, expected_values.shape)
        except ValueError:
            errors[field] = float("inf")
            continue
        if not np.any(valid):
            continue
        difference = np.abs(expected_values - actual_values)
        if not np.all(np.isfinite(difference[valid])):
            errors[field] = float("inf")
            continue
        errors[field] = float(np.max(difference[valid]))
    return errors


def _physx_replay_readback_max_error(
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
) -> float:
    """Return the largest valid absolute write/readback error."""

    errors = _physx_replay_readback_field_errors(expected, actual)
    return max(errors.values()) if errors else float("inf")


def _safe_squeeze_image(img: np.ndarray) -> np.ndarray:
    """Squeeze common single-batch dimensions and ensure HWC uint8 RGB."""
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=2)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if np.nanmax(arr) <= 1.0:
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            else:
                arr = np.clip(arr, 0.0, 255.0)
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def convert_action_quat_to_rotvec(action: np.ndarray) -> np.ndarray:
    """Convert action quaternions from wxyz to rotation vectors.

    Supported layouts:
      - single-arm 23-d Allegro: [pos3, quat4, 16 joints] -> 22-d
      - single-arm 27-d Wuji: [pos3, quat4, 20 joints] -> 26-d
      - C08 28-d Wuji: [pos3, quat4, 20 joints, legacy no-op] -> 27-d
      - bimanual 46-d Allegro: [right23, left23] -> 44-d
      - bimanual 54-d Wuji: [right27, left27] -> 52-d
    """
    action = np.asarray(action)
    if action.ndim != 1:
        return None

    def _convert_single_arm(single_arm_action: np.ndarray) -> np.ndarray:
        pos = single_arm_action[0:3]
        quat = single_arm_action[3:7].astype(np.float64)
        hand_joints = single_arm_action[7:]

        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            rotvec = np.zeros(3, dtype=np.float64)
        else:
            quat = quat / norm
            quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]
            rotvec = Rotation.from_quat(quat_xyzw).as_rotvec()

        return np.concatenate([pos, rotvec, hand_joints])

    if action.shape[0] in (23, 27):
        return _convert_single_arm(action)
    if action.shape[0] == 28:
        return np.concatenate(
            [_convert_single_arm(action[:27]), action[27:28]],
            axis=0,
        )
    if action.shape[0] in (46, 54):
        arm_dim = action.shape[0] // 2
        return np.concatenate(
            [
                _convert_single_arm(action[:arm_dim]),
                _convert_single_arm(action[arm_dim:]),
            ],
            axis=0,
        )
    return None


def _load_episode_actions(zarr_path: Path) -> np.ndarray:
    """Read the recorded action array for a single-episode demo zarr."""
    root = zarr.open(str(zarr_path), mode="r")
    return np.asarray(root["data"]["action"])


def _load_episode_initial_state(zarr_path: Path):
    """Read state[0] for a single-episode demo zarr, or None if not stored."""
    root = zarr.open(str(zarr_path), mode="r")
    if "state" not in root["data"]:
        return None
    return np.asarray(root["data"]["state"][0]).ravel()


def _load_episode_states(zarr_path: Path) -> np.ndarray | None:
    root = zarr.open(str(zarr_path), mode="r")
    if "state" not in root["data"]:
        return None
    return np.asarray(root["data"]["state"])


_YCB02_RECORDING_ONLY_FIELDS = (
    "ycb02_all_cup_poses",
    "ycb02_all_ball_poses",
    "ycb02_hidden_state",
)

# The recording-only fields are time-major arrays.  Keep the shape contract in
# the replay reader instead of relying on ``reshape`` in the Isaac task: a
# truncated or flattened Zarr field would otherwise be silently treated as a
# fresh/random scene for the missing frames.  The four-ball topology uses a
# ten-value identity prefix and a complete twenty-two-value vector.  Older
# three- and five-ball recordings are intentionally rejected because their
# candidate identity blocks are not shape-compatible with this task.
_YCB02_RECORDING_ONLY_TAIL_SHAPES = {
    "ycb02_all_cup_poses": ((3, 7),),
    "ycb02_all_ball_poses": ((4, 7),),
    "ycb02_hidden_state": ((10,), (22,)),
}


def _load_ycb02_recording_state_extras(
    zarr_path: Path,
    *,
    expected_steps: int | None = None,
    field_shapes: dict | None = None,
    task_label: str = "YCB02",
) -> dict[str, np.ndarray]:
    """Load YCB02's non-policy state fields from a recorded Zarr episode.

    YCB02 keeps the complete seven-body layout and hidden variant/event state
    out of the flattened policy ``state``.  Isaac state restorers consume
    those values when available, but the replay loader historically only read
    ``state[0]`` and therefore silently rebuilt a fresh/random YCB02 scene.
    Return only the recording-only arrays, preserving their time axis; absent
    fields remain a valid empty mapping for legacy episodes and other tasks.
    """

    shapes = _YCB02_RECORDING_ONLY_TAIL_SHAPES if field_shapes is None else field_shapes
    root = zarr.open(str(zarr_path), mode="r")
    data = root.get("data")
    if data is None:
        return {}
    extras: dict[str, np.ndarray] = {}
    if expected_steps is not None:
        try:
            numeric_steps = float(expected_steps)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{task_label} recording expected_steps must be an integer, got {expected_steps!r}"
            ) from exc
        if not np.isfinite(numeric_steps) or numeric_steps != round(numeric_steps):
            raise ValueError(
                f"{task_label} recording expected_steps must be an integer, got {expected_steps!r}"
            )
        expected_steps = int(round(numeric_steps))
        if expected_steps < 0:
            raise ValueError(
                f"{task_label} recording expected_steps must be non-negative, got {expected_steps}"
            )
    observed_steps: int | None = None
    for name in shapes:
        if name not in data:
            continue
        try:
            values = np.asarray(data[name], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{task_label} recording field {name!r} cannot be converted to float64"
            ) from exc
        allowed_tails = shapes[name]
        if values.ndim < 1 or tuple(values.shape[1:]) not in allowed_tails:
            expected = " or ".join(
                f"(T, {tail[0]}, {tail[1]})" if len(tail) == 2
                else f"(T, {tail[0]})"
                for tail in allowed_tails
            )
            raise ValueError(
                f"{task_label} recording field {name!r} must have shape {expected} "
                f"(got {values.shape})"
            )
        steps = int(values.shape[0])
        if expected_steps is not None and steps != expected_steps:
            raise ValueError(
                f"{task_label} recording field {name!r} has {steps} frames; "
                f"expected exactly {expected_steps}"
            )
        if observed_steps is None:
            observed_steps = steps
        elif steps != observed_steps:
            raise ValueError(
                f"{task_label} recording-only fields have inconsistent frame counts: "
                f"{name!r} has {steps}, expected {observed_steps}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{task_label} recording field {name!r} must be finite")
        if name.endswith("_poses"):
            quaternion_norms = np.linalg.norm(values[..., 3:7], axis=-1)
            if np.any(
                ~np.isfinite(quaternion_norms) | (quaternion_norms <= 1.0e-8)
            ):
                raise ValueError(
                    f"{task_label} recording field {name!r} contains a zero/non-finite quaternion"
                )
        extras[name] = np.ascontiguousarray(values)
    return extras


def _load_xy6_recording_state_extras(zarr_path: Path, *, expected_steps: int) -> dict[str, np.ndarray]:
    """Load XY6's five-body layout and history for initial/visual restoration.

    Physical certification still compares these fields without restoring them
    after action 0. The policy vector does not contain the lid and spoon poses.
    """
    from handson.sim.isaaclab.eval_comb06_season_and_stir_env import XY6_HIDDEN_STATE_FIELDS

    data = zarr.open(str(zarr_path), mode="r")["data"]
    tails = {"xy6_object_poses": (5, 7), "xy6_hidden_state": (len(XY6_HIDDEN_STATE_FIELDS),)}
    result = {}
    for name, tail in tails.items():
        if name not in data:
            raise ValueError(f"XY6 camera/initial replay requires {name}")
        values = np.asarray(data[name], dtype=np.float64)
        if values.shape != (expected_steps, *tail) or not np.isfinite(values).all():
            raise ValueError(f"XY6 recording field {name} must be finite shape {(expected_steps, *tail)}")
        if name == "xy6_object_poses" and np.any(np.linalg.norm(values[..., 3:7], axis=-1) <= 1e-8):
            raise ValueError("XY6 object poses contain a zero quaternion")
        result[name] = values
    return result


def _ycb02_recording_extras_at(
    recording_extras: dict[str, np.ndarray] | None,
    frame_index: int,
) -> dict[str, np.ndarray]:
    """Return one frame of YCB02 recording-only telemetry, if available."""

    if not recording_extras:
        return {}
    index = int(frame_index)
    result: dict[str, np.ndarray] = {}
    for name, values in recording_extras.items():
        array = np.asarray(values)
        if array.ndim == 0 or index < 0 or index >= array.shape[0]:
            continue
        result[name] = np.asarray(array[index], dtype=np.float64).copy()
    return result


def _load_recording_schema(demo_dir: Path) -> dict:
    path = demo_dir / "recording_schema.json"
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"{demo_dir.name} has PhysX fields but no recording_schema.json"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc
    if not isinstance(schema, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return schema


def _load_replay_manifest(demo_dir: Path) -> dict:
    path = demo_dir / "replay_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return manifest


def _manifest_recording_parameters(manifest: dict) -> dict:
    recording = manifest.get("recording")
    return recording if isinstance(recording, dict) else {}


def _optional_manifest_seed(recording: dict, name: str) -> int | None:
    value = recording.get(name)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Replay manifest recording.{name} must be an integer")
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Replay manifest recording.{name} must be an integer"
        ) from exc
    if seed < 0:
        raise ValueError(
            f"Replay manifest recording.{name} must be non-negative"
        )
    return seed


def _apply_recorded_handson_environment(manifest: dict) -> None:
    # This live operator readout is collection guidance, not replay content.
    os.environ.pop(BC011_COLLECTION_ANGLE_GUIDANCE_ENV, None)
    environment = manifest.get("environment")
    environment = environment if isinstance(environment, dict) else {}
    variables = environment.get("variables")
    if not isinstance(variables, dict):
        return
    for name, value in variables.items():
        normalized_name = str(name)
        if (
            normalized_name.startswith("HANDSON_")
            and normalized_name != BC011_COLLECTION_ANGLE_GUIDANCE_ENV
            and value is not None
        ):
            os.environ[normalized_name] = str(value)


def _apply_isaac_render_overrides(
    *,
    wrist_width: int,
    wrist_height: int,
    force_complete_arm_render: bool,
) -> None:
    if wrist_width > 0:
        os.environ["HANDSON_ISAACLAB_WRIST_CAMERA_OUTPUT_SIZE"] = (
            f"{int(wrist_width)}x{int(wrist_height)}"
        )
    if force_complete_arm_render:
        os.environ["HANDSON_ISAACLAB_DEFER_ARM_RENDER"] = "0"
        os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "0"
        os.environ.pop("HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY", None)
        os.environ["HANDSON_ISAACSIM_ACCURATE_CAMERA_VISIBILITY"] = "1"


def _load_physx_replay_trajectory(
    zarr_path: Path,
    demo_dir: Path,
    *,
    expected_steps: int,
) -> PhysxReplayTrajectory:
    schema = _load_recording_schema(demo_dir)
    environment_schema = schema.get("environment_schema")
    if not isinstance(environment_schema, dict):
        raise ValueError(
            f"{demo_dir.name} recording schema has no environment_schema"
        )
    physx_schema = environment_schema.get("physx")
    if not isinstance(physx_schema, dict):
        raise ValueError(
            f"{demo_dir.name} recording schema has no PhysX entity layout"
        )

    required = list(_PHYSX_REQUIRED_ROBOT_FIELDS)
    if physx_schema.get("rigid_body_names"):
        required.extend(
            (
                "physx_task_rigid_body_state",
                "physx_task_rigid_body_valid",
            )
        )
    articulation_names = tuple(physx_schema.get("articulation_names") or ())
    articulation_dof_names = physx_schema.get("articulation_dof_names") or {}
    has_articulation_dofs = any(
        articulation_dof_names.get(name) for name in articulation_names
    )
    if articulation_names:
        required.extend(
            (
                "physx_task_articulation_root_state",
                "physx_task_articulation_root_valid",
            )
        )
    if has_articulation_dofs:
        required.extend(
            (
                "physx_task_articulation_dof_position",
                "physx_task_articulation_dof_velocity",
                "physx_task_articulation_dof_valid",
            )
        )
    if physx_schema.get("deformable_name"):
        required.extend(
            (
                "physx_deformable_position",
                "physx_deformable_velocity",
                "physx_deformable_valid",
            )
        )

    root = zarr.open(str(zarr_path), mode="r")
    data = root["data"]
    available = set(data.array_keys())
    field_aliases = {
        "physx_robot_joint_q": ("physx_robot_joint_position",),
        "physx_robot_joint_qd": ("physx_robot_joint_velocity",),
    }

    def has_field(base_name: str, *, after_action: bool) -> bool:
        prefix = "next_" if after_action else ""
        candidates = (base_name, *field_aliases.get(base_name, ()))
        return any(f"{prefix}{name}" in available for name in candidates)

    missing = sorted(
        f"{'next_' if after_action else ''}{base_name}"
        for base_name in required
        for after_action in (False, True)
        if not has_field(base_name, after_action=after_action)
    )
    if missing:
        raise ValueError(
            f"{demo_dir.name} is missing PhysX replay fields: {missing}"
        )

    fields = {}
    for name in sorted(available):
        if not (
            name.startswith("physx_")
            or name.startswith("next_physx_")
        ):
            continue
        values = np.asarray(data[name])
        if values.shape[0] != expected_steps:
            raise ValueError(
                f"{demo_dir.name} field {name} has {values.shape[0]} steps; "
                f"action has {expected_steps}"
            )
        fields[name] = values
    canonical_aliases = {
        "physx_robot_joint_q": "physx_robot_joint_position",
        "physx_robot_joint_qd": "physx_robot_joint_velocity",
        "physx_robot_tcp_pose": "physx_tcp_state",
    }
    for prefix in ("", "next_"):
        for legacy_name, canonical_name in canonical_aliases.items():
            legacy_key = f"{prefix}{legacy_name}"
            canonical_key = f"{prefix}{canonical_name}"
            if canonical_key not in fields and legacy_key in fields:
                fields[canonical_key] = fields[legacy_key]
    return PhysxReplayTrajectory(
        fields=fields,
        schema=schema,
        step_count=int(expected_steps),
    )


def _demo_source_succeeded(demo_dir: Path) -> bool:
    try:
        meta = json.loads(
            (demo_dir / "collection_meta.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    if not isinstance(meta, dict):
        return False
    outcomes = (
        str(meta.get("task_outcome") or "").strip().lower(),
        str(meta.get("source_outcome") or "").strip().lower(),
    )
    return "success" in outcomes


def _sim_backend_for_demo(demo_dir: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    def _json_object(path: Path) -> dict:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    manifest = _json_object(demo_dir / "replay_manifest.json")
    recording = manifest.get("recording")
    recording = recording if isinstance(recording, dict) else {}
    manifest_backend = str(
        recording.get("sim_backend") or manifest.get("sim_backend") or ""
    ).strip().lower()
    if manifest_backend in {"mujoco", "isaacsim", "isaacsim_newton"}:
        return manifest_backend
    if manifest_backend == "genesis":
        raise RuntimeError(
            f"{demo_dir.name} was recorded with Genesis, which this replay "
            "script does not support"
        )

    meta = _json_object(demo_dir / "collection_meta.json")
    recorded_backend = str(meta.get("sim_backend") or "").strip().lower()
    if recorded_backend in {"mujoco", "isaacsim", "isaacsim_newton"}:
        return recorded_backend
    engine_code = str(meta.get("engine_code") or "").strip().upper()
    if engine_code == "E01":
        return "mujoco"
    if engine_code == "E02":
        return "isaacsim"
    if engine_code == "E04":
        return "isaacsim_newton"
    if engine_code == "E03" or recorded_backend == "genesis":
        raise RuntimeError(
            f"{demo_dir.name} was recorded with Genesis, which this replay "
            "script does not support"
        )
    print(
        f"[Warning] {demo_dir.name} has no simulation backend metadata; "
        "falling back to MuJoCo for legacy compatibility."
    )
    return "mujoco"


def _initial_state_layout(raw_env, config) -> tuple[tuple[str, int], ...]:
    ref_state = raw_env._compute_observation()["state"]
    return tuple(
        (key, np.asarray(ref_state[key]).ravel().size)
        for key in config.proprio_keys
    )


_STATE_LAYOUT_COMPAT_WARNED: set[tuple[str, int, int, tuple[str, ...], int]] = set()


def _state_layout_skip_cost(key: str, size: int) -> int:
    """Cost used when matching legacy flat-state vectors to current layouts.

    Older cloud zarr episodes did not always save bookkeeping fields that were
    later added to task proprio layouts.  Those fields are not required for the
    kinematic visual replay itself, so they are preferred skip candidates when a
    recorded flat state is shorter than the current config layout.
    """
    if key in {"target_selector", "target_id", "target_name"}:
        return 0
    if key.endswith("_metrics") or key in {
        "assembly_metrics",
        "alignment_error",
        "connection_state",
        "nail_depth",
        "particle_count",
        "sequence_progress",
        "sequence_target",
        "spill_count",
        "stapler_hinge",
        "tongs_open_angle",
        "wrong_hit_count",
    }:
        return 5
    if size <= 2 and not key.endswith("_pose") and not key.endswith("_poses"):
        return 20
    return 1000


def _select_state_layout_for_recorded_size(
    layout: tuple[tuple[str, int], ...],
    recorded_size: int,
) -> tuple[tuple[tuple[str, int], ...], tuple[str, ...], int]:
    expected_size = sum(size for _key, size in layout)
    if recorded_size == expected_size:
        return layout, (), 0

    if recorded_size > expected_size:
        return layout, (), recorded_size - expected_size

    missing = expected_size - recorded_size
    # Subset-sum over layout fields: choose skipped fields whose sizes exactly
    # explain the legacy length gap, preferring bookkeeping fields.
    choices: dict[int, tuple[int, tuple[str, ...]]] = {0: (0, ())}
    for key, size in layout:
        cost = _state_layout_skip_cost(key, size)
        for total, (total_cost, skipped) in list(choices.items()):
            new_total = total + size
            if new_total > missing:
                continue
            candidate = (total_cost + cost, skipped + (key,))
            current = choices.get(new_total)
            if current is None or candidate[0] < current[0]:
                choices[new_total] = candidate
    if missing in choices:
        skipped = choices[missing][1]
        skipped_set = set(skipped)
        return tuple((key, size) for key, size in layout if key not in skipped_set), skipped, 0

    # Last-resort greedy compatibility: do not partially consume a field that
    # does not fit; keep later smaller fields aligned when possible.
    kept = []
    skipped = []
    remaining = recorded_size
    for key, size in layout:
        if size <= remaining:
            kept.append((key, size))
            remaining -= size
        else:
            skipped.append(key)
    return tuple(kept), tuple(skipped), remaining


def _warn_state_layout_compat(
    config,
    recorded_size: int,
    expected_size: int,
    skipped: tuple[str, ...],
    trailing_values: int,
) -> None:
    task_name = str(getattr(config, "task_name", "") or getattr(config, "task_id", ""))
    key = (task_name, recorded_size, expected_size, skipped, trailing_values)
    if key in _STATE_LAYOUT_COMPAT_WARNED:
        return
    _STATE_LAYOUT_COMPAT_WARNED.add(key)
    details = []
    if skipped:
        details.append(f"skipping missing fields: {', '.join(skipped)}")
    if trailing_values:
        details.append(f"ignoring {trailing_values} trailing values")
    if not details:
        details.append("using best-effort layout compatibility")
    print(
        "[Warning] Recorded state length differs from current layout for "
        f"{task_name}: recorded={recorded_size}, current={expected_size}; "
        + "; ".join(details)
    )


def _split_initial_state(
    config,
    state_vec: np.ndarray,
    layout: tuple[tuple[str, int], ...],
) -> dict[str, np.ndarray]:
    """Undo HandsOnObsAdapter's proprio concatenation without MuJoCo imports."""
    state_arr = np.asarray(state_vec)
    recorded_size = state_arr.size
    expected_size = sum(size for _key, size in layout)
    active_layout, skipped, trailing_values = _select_state_layout_for_recorded_size(
        layout,
        recorded_size,
    )
    if recorded_size != expected_size:
        _warn_state_layout_compat(
            config,
            recorded_size,
            expected_size,
            skipped,
            trailing_values,
        )

    parts = {}
    offset = 0
    for key, size in active_layout:
        parts[key] = np.asarray(state_arr[offset : offset + size], dtype=np.float64)
        offset += size
    if offset > recorded_size:
        raise ValueError(
            f"Recorded state has {recorded_size} values, "
            f"but {config.task_name} consumes {offset}"
        )
    return parts


def _recorded_action_storage_layout(recording: dict, *, task_id: str) -> str | None:
    """Read the recorder's layout, including the original XY6 camera probe."""
    layout = recording.get("action_storage_layout")
    if layout is None:
        # Before the layout fix, this specific diagnostic wrote native Box
        # actions directly. Formal teleop recordings use per-side actions.
        if (
            task_id == "eval_comb06_season_and_stir"
            and recording.get("recording_purpose") == "xy6_camera_diagnostic"
            and recording.get("recorder") == "validate_xy6_live.py"
        ):
            return "right_pose_left_pose_right_hand_left_hand"
        return None
    if layout not in {
        "single_pose_hand",
        "right_pose_hand_then_left_pose_hand",
        "right_pose_left_pose_right_hand_left_hand",
        "flat_raw_env_action",
    }:
        raise ValueError(f"Unsupported recorded action storage layout: {layout!r}")
    return layout


def _step_with_recorded_action(
    env, action_flat: np.ndarray, *, action_storage_layout: str | None = None,
):
    """Step the raw env with a recorded action and rewrap the observation.

    Teleop stores ``[right(23/27), left(23/27)]``; native Box actions store
    ``[r_pose, l_pose, r_hand, l_hand]``. Use the manifest layout before
    bypassing the policy wrapper. Unlabeled legacy recordings retain their
    historical per-side interpretation.
    """
    raw_env = env.unwrapped
    action_flat = np.asarray(action_flat, dtype=np.float64)
    bimanual = action_flat.ndim == 1 and action_flat.shape[0] in (46, 54)
    if action_storage_layout is None:
        action_storage_layout = (
            "right_pose_hand_then_left_pose_hand" if bimanual else "flat_raw_env_action"
        )
    if action_storage_layout in {
        "right_pose_hand_then_left_pose_hand", "right_pose_left_pose_right_hand_left_hand",
    } and not bimanual:
        raise ValueError(f"Bimanual recorded action has invalid shape {action_flat.shape}")
    if action_storage_layout == "right_pose_hand_then_left_pose_hand":
        side_dim = action_flat.shape[0] // 2
        raw_action = {
            "right": action_flat[:side_dim],
            "left": action_flat[side_dim:],
        }
    elif action_storage_layout in {
        "single_pose_hand", "right_pose_left_pose_right_hand_left_hand", "flat_raw_env_action",
    }:
        raw_action = action_flat
    else:
        raise ValueError(f"Unsupported recorded action storage layout: {action_storage_layout!r}")
    raw_obs, rew, done, trunc, info = raw_env.step(raw_action)
    return env.observation(raw_obs), rew, done, trunc, info


def _step_with_current_joint_targets(env):
    """Advance PhysX once after recorded targets were written to the drives."""

    raw_env = env.unwrapped
    step_targets = getattr(raw_env, "step_with_current_joint_targets", None)
    if not callable(step_targets):
        raise RuntimeError(
            "Isaac environment does not support low-level joint-target replay"
        )
    raw_obs, rew, done, trunc, info = step_targets()
    return env.observation(raw_obs), rew, done, trunc, info


_ROBOT_PATH_STATE_FIELDS = (
    "physx_robot_joint_position",
    "physx_robot_joint_velocity",
    "physx_robot_joint_valid",
    "physx_robot_joint_target",
    "physx_robot_joint_target_valid",
)


def _robot_path_snapshot(snapshot: dict) -> dict:
    missing = [
        name for name in _ROBOT_PATH_STATE_FIELDS if name not in snapshot
    ]
    if missing:
        raise RuntimeError(
            "MuJoCo-style robot path replay is missing fields: "
            + ", ".join(missing)
        )
    return {
        name: np.asarray(snapshot[name]).copy()
        for name in _ROBOT_PATH_STATE_FIELDS
    }


def _validate_exact_robot_substep_path(
    before_robot: dict[str, np.ndarray],
    after_robot: dict[str, np.ndarray],
    substep_position: np.ndarray,
    substep_velocity: np.ndarray,
    substep_valid: np.ndarray,
    substep_target: np.ndarray,
    substep_target_valid: np.ndarray,
) -> np.ndarray:
    """Validate that every real robot joint has one finite exact substep path."""

    joint_shape = tuple(substep_position.shape[1:])
    for phase, snapshot in (
        ("before", before_robot),
        ("after", after_robot),
    ):
        for name in _ROBOT_PATH_STATE_FIELDS:
            value = np.asarray(snapshot[name])
            if value.shape != joint_shape:
                raise RuntimeError(
                    "exact PhysX robot path control-frame shape mismatch: "
                    f"{phase}.{name}={value.shape}, expected={joint_shape}"
                )

    before_valid_raw = np.asarray(before_robot["physx_robot_joint_valid"])
    after_valid_raw = np.asarray(after_robot["physx_robot_joint_valid"])
    before_target_valid_raw = np.asarray(
        before_robot["physx_robot_joint_target_valid"]
    )
    after_target_valid_raw = np.asarray(
        after_robot["physx_robot_joint_target_valid"]
    )
    mask_fields = {
        "before joint valid": before_valid_raw,
        "after joint valid": after_valid_raw,
        "before target valid": before_target_valid_raw,
        "after target valid": after_target_valid_raw,
        "substep joint valid": substep_valid,
        "substep target valid": substep_target_valid,
    }
    for label, values in mask_fields.items():
        if not np.all(np.isin(values, (0, 1))):
            raise RuntimeError(
                f"exact PhysX robot path has non-binary {label} mask"
            )

    active = before_valid_raw.astype(bool)
    if not np.any(active):
        raise RuntimeError(
            "exact PhysX robot path has no valid recorded robot joints"
        )
    if not np.array_equal(active, after_valid_raw.astype(bool)):
        raise RuntimeError(
            "exact PhysX robot path joint-valid coverage changes between "
            "the control-frame endpoints"
        )
    if not np.all(before_target_valid_raw.astype(bool)[active]):
        raise RuntimeError(
            "exact PhysX robot path target-valid coverage is incomplete "
            "at the control-frame start"
        )
    if not np.all(after_target_valid_raw.astype(bool)[active]):
        raise RuntimeError(
            "exact PhysX robot path target-valid coverage is incomplete "
            "at the control-frame endpoint"
        )
    if not np.all(substep_valid.astype(bool)[:, active]):
        raise RuntimeError(
            "exact PhysX robot path joint-valid coverage is incomplete "
            "at one or more physics substeps"
        )
    if not np.all(substep_target_valid.astype(bool)[:, active]):
        raise RuntimeError(
            "exact PhysX robot path target-valid coverage is incomplete "
            "at one or more physics substeps"
        )

    finite_fields = {
        "before position": before_robot["physx_robot_joint_position"],
        "before velocity": before_robot["physx_robot_joint_velocity"],
        "before target": before_robot["physx_robot_joint_target"],
        "after position": after_robot["physx_robot_joint_position"],
        "after velocity": after_robot["physx_robot_joint_velocity"],
        "after target": after_robot["physx_robot_joint_target"],
    }
    for label, values in finite_fields.items():
        if not np.all(np.isfinite(np.asarray(values)[active])):
            raise RuntimeError(
                f"exact PhysX robot path has non-finite {label} values"
            )
    for label, values in (
        ("substep position", substep_position),
        ("substep velocity", substep_velocity),
        ("substep target", substep_target),
    ):
        if not np.all(np.isfinite(np.asarray(values)[:, active])):
            raise RuntimeError(
                f"exact PhysX robot path has non-finite {label} values"
            )

    for label, endpoint, control_after in (
        (
            "position",
            substep_position[-1],
            after_robot["physx_robot_joint_position"],
        ),
        (
            "velocity",
            substep_velocity[-1],
            after_robot["physx_robot_joint_velocity"],
        ),
        (
            "target",
            substep_target[-1],
            after_robot["physx_robot_joint_target"],
        ),
    ):
        if not np.allclose(
            np.asarray(endpoint)[active],
            np.asarray(control_after)[active],
            rtol=0.0,
            atol=_PHYSX_REPLAY_VERIFY_TOLERANCE,
        ):
            raise RuntimeError(
                "exact PhysX robot path final substep does not match the "
                f"control-frame after-state for {label}"
            )
    return active


def _assert_robot_path_callbacks_available(raw_env) -> None:
    conflicts = []
    if getattr(raw_env, "_handson_pre_physics_substep_callback", None) is not None:
        conflicts.append("_handson_pre_physics_substep_callback")
    post_substep = getattr(
        raw_env,
        "_handson_post_physics_substep_callback",
        None,
    )
    validator = getattr(
        raw_env,
        "physical_replay_post_substep_observer_is_safe",
        None,
    )
    try:
        safe_post_substep = (
            post_substep is not None
            and callable(validator)
            and validator(post_substep) is True
        )
    except Exception:
        safe_post_substep = False
    if post_substep is not None and not safe_post_substep:
        conflicts.append("_handson_post_physics_substep_callback")
    if conflicts:
        raise RuntimeError(
            "robot path replay cannot share PhysX substep callbacks: "
            + ", ".join(conflicts)
        )


def _step_with_recorded_robot_path(
    env,
    physx_trace,
    recorded_schema: dict,
    before_snapshot: dict,
    after_snapshot: dict,
):
    """Drive the exact recorded robot substep path while PhysX solves contacts."""

    raw_env = env.unwrapped
    before_robot = _robot_path_snapshot(before_snapshot)
    after_robot = _robot_path_snapshot(after_snapshot)
    required_substep_fields = (
        "physx_substep_robot_joint_position",
        "physx_substep_robot_joint_velocity",
        "physx_substep_robot_joint_valid",
        "physx_substep_robot_joint_target",
        "physx_substep_robot_joint_target_valid",
        "physx_substep_count",
    )
    missing_substep_fields = [
        name for name in required_substep_fields if name not in after_snapshot
    ]
    if missing_substep_fields:
        raise RuntimeError(
            "exact PhysX robot-path replay requires recorded robot state at "
            "every physics substep; missing "
            + ", ".join(missing_substep_fields)
        )
    substep_position = np.asarray(
        after_snapshot["physx_substep_robot_joint_position"]
    )
    substep_velocity = np.asarray(
        after_snapshot["physx_substep_robot_joint_velocity"]
    )
    substep_valid = np.asarray(
        after_snapshot["physx_substep_robot_joint_valid"]
    )
    substep_target = np.asarray(
        after_snapshot["physx_substep_robot_joint_target"]
    )
    substep_target_valid = np.asarray(
        after_snapshot["physx_substep_robot_joint_target_valid"]
    )
    recorded_substep_count = int(
        np.asarray(after_snapshot["physx_substep_count"]).reshape(-1)[0]
    )
    expected_shape = substep_position.shape
    if (
        substep_position.ndim != 3
        or recorded_substep_count <= 0
        or recorded_substep_count != expected_shape[0]
        or substep_velocity.shape != expected_shape
        or substep_valid.shape != expected_shape
        or substep_target.shape != expected_shape
        or substep_target_valid.shape != expected_shape
    ):
        raise RuntimeError(
            "invalid exact PhysX robot substep layout: "
            f"count={recorded_substep_count}, "
            f"position={substep_position.shape}, "
            f"velocity={substep_velocity.shape}, "
            f"valid={substep_valid.shape}, "
            f"target={substep_target.shape}, "
            f"target_valid={substep_target_valid.shape}"
        )
    _validate_exact_robot_substep_path(
        before_robot,
        after_robot,
        substep_position,
        substep_velocity,
        substep_valid,
        substep_target,
        substep_target_valid,
    )
    _assert_robot_path_callbacks_available(raw_env)

    guided_physics_replay = os.environ.get(
        "HANDSON_GUIDED_PHYSICS_REPLAY",
        "0",
    ).strip().lower() in {"1", "true", "yes", "on"}
    guided_reports = []
    if guided_physics_replay:
        required_guided_fields = (
            "physx_substep_task_rigid_body_state",
            "physx_substep_task_rigid_body_valid",
        )
        missing_guided_fields = [
            name for name in required_guided_fields if name not in after_snapshot
        ]
        if missing_guided_fields:
            raise RuntimeError(
                "guided physical replay requires recorded rigid-body state "
                "at every physics substep; missing "
                + ", ".join(missing_guided_fields)
            )
        guided_states = np.asarray(
            after_snapshot["physx_substep_task_rigid_body_state"]
        )
        guided_valid = np.asarray(
            after_snapshot["physx_substep_task_rigid_body_valid"]
        )
        if (
            guided_states.shape[0] != recorded_substep_count
            or guided_valid.shape[0] != recorded_substep_count
        ):
            raise RuntimeError(
                "guided physical replay rigid-body substep count differs "
                "from the recorded robot path"
            )

    def write_recorded_robot(substep_index: int, substep_count: int) -> None:
        if substep_count != recorded_substep_count:
            raise RuntimeError(
                "PhysX substep count differs from the recording: "
                f"runtime={substep_count}, recorded={recorded_substep_count}"
            )
        if substep_index == 0:
            position = before_robot["physx_robot_joint_position"]
            velocity = before_robot["physx_robot_joint_velocity"]
            valid = before_robot["physx_robot_joint_valid"]
        else:
            source_index = substep_index - 1
            position = substep_position[source_index]
            velocity = substep_velocity[source_index]
            valid = substep_valid[source_index]
        substep_snapshot = {
            "physx_robot_joint_position": position,
            "physx_robot_joint_velocity": velocity,
            "physx_robot_joint_valid": valid,
            "physx_robot_joint_target": substep_target[substep_index],
            "physx_robot_joint_target_valid": substep_target_valid[
                substep_index
            ],
        }
        physx_trace.restore_robot_state_only(
            substep_snapshot,
            recorded_schema,
        )
        if guided_physics_replay:
            guided_reports.append(
                physx_trace.apply_task_tracking_wrenches(
                    {
                        "physx_task_rigid_body_state": guided_states[
                            substep_index
                        ],
                        "physx_task_rigid_body_valid": guided_valid[
                            substep_index
                        ],
                    },
                    recorded_schema,
                    linear_position_gain=float(
                        os.environ.get(
                            "HANDSON_GUIDED_LINEAR_POSITION_GAIN",
                            "3600.0",
                        )
                    ),
                    linear_velocity_gain=float(
                        os.environ.get(
                            "HANDSON_GUIDED_LINEAR_VELOCITY_GAIN",
                            "120.0",
                        )
                    ),
                    angular_position_gain=float(
                        os.environ.get(
                            "HANDSON_GUIDED_ANGULAR_POSITION_GAIN",
                            "1600.0",
                        )
                    ),
                    angular_velocity_gain=float(
                        os.environ.get(
                            "HANDSON_GUIDED_ANGULAR_VELOCITY_GAIN",
                            "80.0",
                        )
                    ),
                    max_linear_acceleration=float(
                        os.environ.get(
                            "HANDSON_GUIDED_MAX_LINEAR_ACCELERATION",
                            "500.0",
                        )
                    ),
                    max_angular_acceleration=float(
                        os.environ.get(
                            "HANDSON_GUIDED_MAX_ANGULAR_ACCELERATION",
                            "2000.0",
                        )
                    ),
                    physics_dt=float(
                        getattr(raw_env, "_physics_dt", 1.0 / 120.0)
                    ),
                )
            )

    raw_env._handson_pre_physics_substep_callback = write_recorded_robot
    try:
        physx_trace.apply_robot_joint_targets(
            after_robot,
            recorded_schema,
        )
        next_obs, rew, done, trunc, info = _step_with_current_joint_targets(env)
    finally:
        try:
            delattr(raw_env, "_handson_pre_physics_substep_callback")
        except AttributeError:
            pass

    pre_correction_robot_field_errors = (
        _physx_replay_readback_field_errors(
            after_robot,
            physx_trace.snapshot(),
        )
    )
    pre_correction_robot_error = (
        max(pre_correction_robot_field_errors.values())
        if pre_correction_robot_field_errors
        else float("inf")
    )

    # Keep the kinematic driver exactly on the recorded endpoint before the
    # next interval. Task objects are deliberately not present in this write.
    endpoint_robot = {
        "physx_robot_joint_position": substep_position[-1],
        "physx_robot_joint_velocity": substep_velocity[-1],
        "physx_robot_joint_valid": substep_valid[-1],
        "physx_robot_joint_target": substep_target[-1],
        "physx_robot_joint_target_valid": substep_target_valid[-1],
    }
    physx_trace.restore_robot_state_only(endpoint_robot, recorded_schema)
    refresh_objects = getattr(raw_env, "_refresh_object_poses", None)
    if callable(refresh_objects):
        refresh_objects()
    success = bool(raw_env._compute_success())
    failed = bool(raw_env._compute_failure())
    rew = 1.0 if success else 0.0
    done = bool(
        success
        or failed
        or (
            raw_env.env_step * raw_env._control_dt
            >= raw_env._time_limit
        )
    )
    info = dict(info) if isinstance(info, dict) else {}
    info.update(
        {
            "physx_robot_pre_correction_readback_max_abs_error": float(
                pre_correction_robot_error
            ),
            "physx_robot_pre_correction_readback_field_max_abs_error": (
                pre_correction_robot_field_errors
            ),
            "guided_physics_replay": bool(guided_physics_replay),
            "guided_physics_body_count": int(
                max(
                    (
                        int(report.get("body_count", 0))
                        for report in guided_reports
                    ),
                    default=0,
                )
            ),
            "guided_physics_max_position_error": float(
                max(
                    (
                        float(report.get("max_position_error", 0.0))
                        for report in guided_reports
                    ),
                    default=0.0,
                )
            ),
            "guided_physics_max_angle_error": float(
                max(
                    (
                        float(report.get("max_angle_error", 0.0))
                        for report in guided_reports
                    ),
                    default=0.0,
                )
            ),
        }
    )
    info["succeed"] = success
    if failed:
        info["failed"] = True
    else:
        info.pop("failed", None)
    raw_obs = raw_env._attach_physx_ground_truth(
        raw_env._compute_observation()
    )
    next_obs = env.observation(raw_obs)
    return next_obs, rew, done, trunc, info


def _restore_isaac_recorded_state(
    env,
    config,
    state_vec: np.ndarray,
    state_layout: tuple[tuple[str, int], ...],
    *,
    visual: bool,
    recording_extras: dict[str, np.ndarray] | None = None,
):
    raw_env = env.unwrapped
    state_parts = _split_initial_state(config, state_vec, state_layout)
    # Recording-only telemetry is deliberately merged after splitting the
    # public policy state.  It never changes the policy interface or flat
    # state layout; it only gives Isaac's task-owned restorer the complete
    # YCB02 scene/identity needed for faithful visual reconstruction.
    if recording_extras:
        state_parts.update(
            {
                name: np.asarray(value, dtype=np.float64).copy()
                for name, value in recording_extras.items()
            }
        )
    if visual:
        restore_visual = getattr(raw_env, "restore_recorded_visual_state", None)
        if restore_visual is not None:
            return env.observation(
                restore_visual(
                    state_parts,
                    robot_iterations=FLAGS.visual_replay_ik_iterations,
                )
            )
        print(
            "[Warning] Isaac task does not support visual-state restoration; "
            "falling back to object-only state restoration."
        )
    restore = getattr(raw_env, "restore_recorded_initial_state", None)
    if restore is None:
        raise RuntimeError("Isaac task does not support recorded-state restoration")
    return env.observation(restore(state_parts))


def _collect_camera_keys(observation: dict) -> list:
    keys = []
    for k, v in observation.items():
        if isinstance(v, np.ndarray) and v.ndim >= 3 and v.shape[-1] == 3:
            keys.append(k)
    return keys


def _physx_replay_mode_from_info(info: dict) -> str | None:
    mode = info.get("physx_replay_mode")
    if mode in (
        "state_restore",
        "action_reexecution",
        "joint_target_reexecution",
        "kinematic_robot_path_reexecution",
        "guided_physical_robot_path_reexecution",
    ):
        return str(mode)
    if info.get("physx_ground_truth_replay"):
        return "state_restore"
    if info.get("physx_action_replay_audit"):
        return "action_reexecution"
    if info.get("physx_joint_target_replay_audit"):
        return "joint_target_reexecution"
    if info.get("physx_robot_path_replay_audit"):
        return "kinematic_robot_path_reexecution"
    return None


def _episode_data_from_trajectory(
    trajectory: list[dict],
    *,
    data_fps: float,
    environment_schema: dict | None = None,
) -> dict[str, np.ndarray]:
    """Build replay arrays while retaining small Plan/telemetry observations."""
    actions = np.stack([transition["actions"] for transition in trajectory], axis=0)
    episode_data = {
        "action": actions,
        "timestamp": np.arange(actions.shape[0]) / float(data_fps),
        "terminal": np.asarray(
            [
                int(index == actions.shape[0] - 1)
                for index in range(actions.shape[0])
            ],
            dtype=np.uint8,
        ),
        "succeed": np.asarray(
            [
                int(bool(transition.get("infos", {}).get("succeed", False)))
                for transition in trajectory
            ],
            dtype=np.uint8,
        ),
    }

    states = [transition["observations"].get("state") for transition in trajectory]
    if all(isinstance(state, np.ndarray) for state in states):
        try:
            episode_data["state"] = np.stack(states, axis=0)
        except ValueError:
            pass

    # Preserve fixed-shape recording telemetry produced by Plan environments
    # (and any future schema-backed task) when a replay is written as a new
    # episode. Camera tensors and the already handled flattened state are
    # filtered by the shared recording-field boundary.
    episode_data.update(stack_numeric_observation_fields(trajectory))
    if environment_schema is not None:
        episode_data.update(
            stack_transition_numeric_observation_fields(
                trajectory,
                required_fields=declared_recording_field_names(
                    environment_schema
                ),
                environment_schema=environment_schema,
            )
        )

    physx_infos = [
        transition.get("infos", {})
        for transition in trajectory
        if _physx_replay_mode_from_info(transition.get("infos", {}))
        is not None
    ]
    if len(physx_infos) == len(trajectory) and physx_infos:
        errors = np.asarray(
            [
                info["physx_readback_max_abs_error"]
                for info in physx_infos
            ],
            dtype=np.float64,
        )
        episode_data["physx_replay_readback_max_abs_error"] = errors[
            :, None
        ]
        episode_data["physx_replay_state_verified"] = np.asarray(
            [
                np.isfinite(error)
                and error <= _PHYSX_REPLAY_VERIFY_TOLERANCE
                for error in errors
            ],
            dtype=np.uint8,
        )[:, None]

    converted = [convert_action_quat_to_rotvec(action) for action in actions]
    if all(action is not None for action in converted):
        episode_data["action_rotvec"] = np.stack(converted, axis=0)
    return episode_data


def _write_physx_recapture_manifest(
    demo_dir: Path,
    *,
    source_demo_dir: Path,
    environment_schema: dict,
    episode_data: dict[str, np.ndarray],
    succeeded: bool,
    data_fps: float,
) -> Path:
    """Write explicit provenance for a fresh Isaac action recapture."""
    source_manifest_path = source_demo_dir / "replay_manifest.json"
    if source_manifest_path.is_file():
        source_manifest = json.loads(
            source_manifest_path.read_text(encoding="utf-8")
        )
        manifest = (
            copy.deepcopy(source_manifest)
            if isinstance(source_manifest, dict)
            else {}
        )
    else:
        manifest = {}

    recording = manifest.setdefault("recording", {})
    if not isinstance(recording, dict):
        recording = {}
        manifest["recording"] = recording
    recording.update(
        {
            "step_count": int(episode_data["action"].shape[0]),
            "data_fps": float(data_fps),
            "action_semantics": "exact_raw_environment_action_executed",
            "physical_transition_contract": (
                "physx_state_t_plus_exact_robot_substep_path_t_"
                "produces_next_physx_state_t"
            ),
            "physical_replay_state_recorded": True,
            "physical_replay_certification_required": True,
            "physical_replay_action_source": (
                "recorded_physx_substep_robot_joint_state_and_targets"
            ),
            "physical_replay_ik": (
                "source_action_reexecuted_in_isaac_then_actual_robot_"
                "substep_path_recorded"
            ),
        }
    )
    manifest["recapture"] = {
        "format": "handson.isaac_physx_v4_action_recapture",
        "version": 1,
        "source_demo_dir": str(source_demo_dir.resolve()),
        "created_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(timespec="seconds"),
        "engine": "isaac_physx",
        "mode": "action_reexecution",
        "environment_schema_version": int(
            environment_schema.get("version", 0)
        ),
        "source_initial_state_restored_once": True,
        "task_object_state_written_during_action_reexecution": False,
        "task_success": bool(succeeded),
    }
    manifest_path = demo_dir / "replay_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (demo_dir / "collection_meta.json").write_text(
        json.dumps(
            {
                "task_outcome": "success" if succeeded else "failure",
                "source_outcome": "success" if succeeded else "failure",
                "recapture_mode": "isaac_physx_v4_action_reexecution",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _physx_replay_verification_report(
    trajectory: list[dict],
) -> dict | None:
    """Build a state-by-state audit for exact, action, or target replay."""

    infos = [transition.get("infos", {}) for transition in trajectory]
    modes = [_physx_replay_mode_from_info(info) for info in infos]
    if not modes or modes[0] is None or any(mode != modes[0] for mode in modes):
        return None

    def state_result(
        *,
        phase: str,
        source_state_index: int,
        error_value: object,
        field_errors: object = None,
    ) -> dict:
        error = float(error_value)
        verified = bool(
            np.isfinite(error)
            and error <= _PHYSX_REPLAY_VERIFY_TOLERANCE
        )
        safe_field_errors = {}
        if isinstance(field_errors, dict):
            safe_field_errors = {
                str(field): (
                    float(value) if np.isfinite(float(value)) else None
                )
                for field, value in field_errors.items()
            }
        return {
            "phase": phase,
            "source_state_index": int(source_state_index),
            "max_abs_error": error if np.isfinite(error) else None,
            "field_max_abs_error": safe_field_errors,
            "verified": verified,
        }

    per_state = [
        state_result(
            phase="pre_action_initial",
            source_state_index=0,
            error_value=infos[0]["physx_initial_readback_max_abs_error"],
            field_errors=infos[0].get(
                "physx_initial_readback_field_max_abs_error"
            ),
        )
    ]
    per_state.extend(
        state_result(
            phase="post_action",
            source_state_index=info["source_state_index"],
            error_value=info["physx_readback_max_abs_error"],
            field_errors=info.get("physx_readback_field_max_abs_error"),
        )
        for info in infos
    )
    finite_errors = [
        state["max_abs_error"]
        for state in per_state
        if state["max_abs_error"] is not None
    ]
    failed_states = [
        {
            "phase": state["phase"],
            "source_state_index": state["source_state_index"],
        }
        for state in per_state
        if not state["verified"]
    ]
    field_names = sorted(
        {
            field
            for state in per_state
            for field in state["field_max_abs_error"]
        }
    )
    trajectory_field_errors = {}
    for field in field_names:
        values = [
            state["field_max_abs_error"].get(field)
            for state in per_state
            if field in state["field_max_abs_error"]
        ]
        trajectory_field_errors[field] = (
            max(values)
            if values and all(value is not None for value in values)
            else None
        )
    guided_infos = [
        info for info in infos if bool(info.get("guided_physics_replay"))
    ]
    guided_summary = None
    if guided_infos:
        guided_summary = {
            "task_object_state_written_during_replay": False,
            "task_object_external_tracking_wrenches": True,
            "max_body_count": max(
                int(info.get("guided_physics_body_count", 0))
                for info in guided_infos
            ),
            "max_position_error_before_wrench_m": max(
                float(info.get("guided_physics_max_position_error", 0.0))
                for info in guided_infos
            ),
            "max_angle_error_before_wrench_rad": max(
                float(info.get("guided_physics_max_angle_error", 0.0))
                for info in guided_infos
            ),
        }
    return {
        "format": "handson.physx_replay_verification",
        "version": 1,
        "mode": modes[0],
        "tolerance": float(_PHYSX_REPLAY_VERIFY_TOLERANCE),
        "checked_state_count": len(per_state),
        "verified_state_count": sum(
            int(state["verified"]) for state in per_state
        ),
        "trajectory_max_abs_error": (
            max(finite_errors) if finite_errors else None
        ),
        "trajectory_field_max_abs_error": trajectory_field_errors,
        "trajectory_verified": not failed_states,
        "guided_physics": guided_summary,
        "failed_states": failed_states,
        "per_state": per_state,
    }


class _StreamingCanonicalVideoWriter:
    """Encode replay cameras incrementally so 4K frames do not fill RAM."""

    def __init__(self, base_out: Path, *, fps: int, crf: int):
        self.directory = Path(
            tempfile.mkdtemp(prefix=".streamed-videos-", dir=str(base_out))
        )
        self._fps = max(1, int(fps))
        self._crf = max(0, int(crf))
        self._writers: dict[str, Mp4VideoWriter] = {}
        self._camera_sources: dict[str, str] | None = None
        self._first_frames: dict[str, np.ndarray] = {}
        self.frame_count = 0

    def write_transition(
        self,
        transition: dict,
        camera_sources: dict[str, str],
    ) -> None:
        if self._camera_sources is None:
            missing = [
                key for key in CANONICAL_CAMERA_KEYS if key not in camera_sources
            ]
            if missing:
                raise RuntimeError(
                    f"Streaming replay is missing canonical cameras: {missing}"
                )
            self._camera_sources = {
                key: camera_sources[key] for key in CANONICAL_CAMERA_KEYS
            }
        elif any(
            camera_sources.get(key) != source
            for key, source in self._camera_sources.items()
        ):
            raise RuntimeError("Canonical camera sources changed during replay")

        observations = transition.get("observations", {})
        for camera_key in CANONICAL_CAMERA_KEYS:
            source_key = self._camera_sources[camera_key]
            image = observation_image(observations, source_key)
            if image is None:
                raise RuntimeError(
                    f"Streaming replay frame is missing camera {source_key!r}"
                )
            frame = postprocess_canonical_image(
                _safe_squeeze_image(image),
                camera_key,
                source_key,
            )
            self._first_frames.setdefault(camera_key, frame.copy())
            writer = self._writers.get(camera_key)
            if writer is None:
                writer = Mp4VideoWriter.create_h264(
                    fps=self._fps,
                    codec="h264",
                    input_pix_fmt="rgb24",
                    crf=self._crf,
                    thread_type="FRAME",
                    thread_count=2,
                )
                writer.start(str(self.directory / f"{camera_key}.mp4"))
                self._writers[camera_key] = writer
            writer.write_frame(frame)
        self.frame_count += 1

    def finish(self, camera_view_metadata: dict | None = None) -> Path:
        if self.frame_count <= 0:
            raise RuntimeError("Streaming replay produced no video frames")
        for writer in self._writers.values():
            writer.stop()
        write_video_camera_info(
            videos_dir=self.directory,
            camera_keys=CANONICAL_CAMERA_KEYS,
            first_frames=self._first_frames,
            intrinsics_yaml=FLAGS.fisheye_intrinsics_yaml,
            camera_views=camera_view_metadata or {},
        )
        return self.directory

    def abort(self) -> None:
        for writer in self._writers.values():
            try:
                writer.abort()
            except Exception:
                pass
        shutil.rmtree(self.directory, ignore_errors=True)


def _discard_transition_images(
    transition: dict,
    camera_sources: dict[str, str],
) -> None:
    observations = transition.get("observations")
    if not isinstance(observations, dict):
        return
    observations.pop("images", None)
    for source_key in camera_sources.values():
        observations.pop(source_key, None)


def _write_demo_zarr_and_videos(
    trajectory,
    exp_name: str,
    success_index: int,
    base_out: Path,
    video_fps: int,
    camera_view_metadata: dict | None = None,
    camera_sources: dict[str, str] | None = None,
    recapture_environment_schema: dict | None = None,
    source_demo_dir: Path | None = None,
    succeeded: bool = False,
    streamed_videos_dir: Path | None = None,
):
    """Save one replayed trajectory to a zarr group plus MP4 camera captures."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    demo_dir = base_out / f"{exp_name}_demo_{success_index}_replay_{timestamp}"
    demo_dir.mkdir(parents=True, exist_ok=True)

    episode_data = _episode_data_from_trajectory(
        trajectory,
        data_fps=FLAGS.data_fps,
        environment_schema=recapture_environment_schema,
    )
    T = episode_data["action"].shape[0]

    zarr_path = demo_dir / "replay.zarr"
    store = zarr.DirectoryStore(str(zarr_path))
    episode_store = ZarrEpisodeStore.create_empty(storage=store)
    episode_store.append_episode(episode_data, compressors="disk")
    print(f"[Saved replay.zarr] {zarr_path} (steps: {T})")
    if recapture_environment_schema is not None:
        if source_demo_dir is None:
            raise RuntimeError(
                "A PhysX v4 recapture requires its source demo directory."
            )
        schema_path = write_recording_schema(
            demo_dir,
            episode_store.root,
            episode_data,
            environment_schema=recapture_environment_schema,
        )
        manifest_path = _write_physx_recapture_manifest(
            demo_dir,
            source_demo_dir=source_demo_dir,
            environment_schema=recapture_environment_schema,
            episode_data=episode_data,
            succeeded=succeeded,
            data_fps=FLAGS.data_fps,
        )
        print(f"[Saved recording schema] {schema_path}")
        print(f"[Saved recapture manifest] {manifest_path}")

    verification_report = _physx_replay_verification_report(trajectory)
    if verification_report is not None:
        verification_path = demo_dir / "physx_replay_verification.json"
        verification_path.write_text(
            json.dumps(
                verification_report,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        trajectory_max_error = verification_report[
            "trajectory_max_abs_error"
        ]
        trajectory_max_error_label = (
            f"{trajectory_max_error:.3e}"
            if trajectory_max_error is not None
            else "non-finite"
        )
        print(
            "[Saved PhysX verification] "
            f"{verification_path} "
            f"(mode={verification_report['mode']}, "
            f"{verification_report['verified_state_count']}/"
            f"{verification_report['checked_state_count']} states, "
            "trajectory_max_abs_error="
            f"{trajectory_max_error_label})"
        )

    if FLAGS.physics_only:
        print("[Physics-only replay] Skipped camera videos and metadata.")
        return str(demo_dir)

    videos_dir = demo_dir / "videos"
    if streamed_videos_dir is not None:
        if not streamed_videos_dir.is_dir():
            raise RuntimeError(
                f"Streaming video directory is missing: {streamed_videos_dir}"
            )
        missing = [
            key
            for key in CANONICAL_CAMERA_KEYS
            if not (streamed_videos_dir / f"{key}.mp4").is_file()
        ]
        if missing:
            raise RuntimeError(
                f"Streaming video output is incomplete: {missing}"
            )
        os.replace(streamed_videos_dir, videos_dir)
        print(f"[Saved streamed videos] {videos_dir}")
        return str(demo_dir)

    videos_dir.mkdir(parents=True, exist_ok=True)

    if camera_sources is None:
        camera_sources = select_canonical_camera_sources_for_trajectory(trajectory)
    missing_cameras = [
        key for key in CANONICAL_CAMERA_KEYS if key not in camera_sources
    ]
    if missing_cameras:
        raise RuntimeError(
            "Missing required canonical camera image(s): "
            f"{missing_cameras}. Expected saved videos are exactly "
            f"{list(CANONICAL_CAMERA_KEYS)}."
        )
    camera_keys = list(CANONICAL_CAMERA_KEYS)

    first_frames = {}
    for cam_key in camera_keys:
        source_key = camera_sources[cam_key]
        video_writer = Mp4VideoWriter.create_h264(
            fps=video_fps,
            codec="h264",
            input_pix_fmt="rgb24",
            crf=max(0, int(FLAGS.video_crf)),
            thread_type="FRAME",
            thread_count=2,
        )
        out_path = str(videos_dir / f"{cam_key}.mp4")
        video_writer.start(out_path)

        for step in trajectory:
            img = observation_image(step["observations"], source_key)
            if img is None:
                continue
            frame = _safe_squeeze_image(img)
            frame = postprocess_canonical_image(frame, cam_key, source_key)
            if cam_key not in first_frames:
                first_frames[cam_key] = frame
            video_writer.write_frame(frame)

        video_writer.stop()
        print(f"[Saved video] {out_path} (source: {source_key})")

    if FLAGS.save_camera_info:
        try:
            for path in write_video_camera_info(
                videos_dir=videos_dir,
                camera_keys=camera_keys,
                first_frames=first_frames,
                intrinsics_yaml=FLAGS.fisheye_intrinsics_yaml,
                camera_views=camera_view_metadata or {},
            ):
                print(f"[Saved camera info] {path}")
        except Exception as exc:
            print(f"[Warning] camera info skipped: {exc}")

    if FLAGS.save_depth:
        depth_per_camera = {k: [] for k in camera_keys}
        for step in trajectory:
            for k, frame in step.get("depth", {}).items():
                if k in depth_per_camera and frame is not None:
                    depth_per_camera[k].append(frame)
        for path in write_depth_outputs(depth_per_camera, videos_dir, video_fps):
            print(f"[Saved depth]  {path}")

    return str(demo_dir)


def _enable_isaac_fast_shutdown(env) -> bool:
    """Opt native Isaac environments into their non-blocking close path.

    A headless replay can finish all actions and still hang in
    ``SimulationContext.stop`` while RTX/PhysX owns a stage pointer.  The
    task environments expose this opt-in guard; walk through the Gym wrappers
    so replay teardown cannot leave the UI looking frozen after 100% progress.
    """

    current = getattr(env, "unwrapped", env)
    for _ in range(16):
        if current is None:
            break
        if hasattr(current, "_simulation_app") or hasattr(current, "_sim"):
            try:
                setattr(current, "_fast_shutdown_on_close", True)
            except Exception:
                return False
            return True
        current = getattr(current, "env", None)
    return False


def _replay_single_demo(
    actions: np.ndarray,
    initial_state: np.ndarray,
    recorded_states: np.ndarray | None,
    task_id: str,
    config,
    env_seed: int,
    reset_seed: int | None,
    desc: str,
    *,
    sim_backend: str,
    physx_replay: PhysxReplayTrajectory | None = None,
    physx_action_replay_audit: bool = False,
    physx_joint_target_replay_audit: bool = False,
    physx_robot_path_replay_audit: bool = False,
    physx_seeded_reset: bool = False,
    stable_origin_report: dict | None = None,
    source_succeeded: bool = False,
    close_env: bool = True,
    transition_callback: Callable[
        [dict, dict[str, str]],
        None,
    ]
    | None = None,
    retain_images: bool = True,
    visual_arm_plan_qpos: np.ndarray | None = None,
    arm_type_override: str | None = None,
    hand_type_override: str | None = None,
    recording_state_extras: dict[str, np.ndarray] | None = None,
    action_storage_layout: str | None = None,
):
    if visual_arm_plan_qpos is not None:
        visual_arm_plan_qpos = np.asarray(
            visual_arm_plan_qpos,
            dtype=np.float64,
        )
        if (
            visual_arm_plan_qpos.ndim != 2
            or visual_arm_plan_qpos.shape[1] != 14
            or not np.all(np.isfinite(visual_arm_plan_qpos))
        ):
            raise ValueError(
                "visual_arm_plan_qpos must be a finite (frames, 14) array"
            )
        if recorded_states is None or len(visual_arm_plan_qpos) != len(recorded_states):
            raise ValueError(
                "Offline visual arm plan must have exactly one row per "
                "recorded state frame"
            )
    robot_path_init_mode = os.environ.get(
        "HANDSON_ROBOT_PATH_INIT_MODE",
        "snapshot",
    ).strip().lower()
    if robot_path_init_mode not in {
        "snapshot",
        "stable_then_snapshot",
        "stable_seeded",
    }:
        raise ValueError(
            "HANDSON_ROBOT_PATH_INIT_MODE must be snapshot, "
            "stable_then_snapshot, or stable_seeded"
        )
    robot_path_contact_prime_steps = int(
        os.environ.get("HANDSON_ROBOT_PATH_CONTACT_PRIME_STEPS", "0")
    )
    if robot_path_contact_prime_steps < 0:
        raise ValueError(
            "HANDSON_ROBOT_PATH_CONTACT_PRIME_STEPS must be non-negative"
        )
    env_extra_kwargs = {}
    if str(getattr(config, "motion_mode", "")).strip().lower() == "plan":
        # Plan configs intentionally describe a ROKAE/Wuji topology.  Unlike
        # the recorder CLI, this replay CLI has no arm/hand flags, so make the
        # template's required hardware explicit instead of falling back to the
        # canonical A/B Allegro defaults.
        os.environ["HANDSON_ARM_TYPE"] = "rokae"
        env_extra_kwargs["hand_type"] = "wuji1"
    if task_id == "bimanual_photograph":
        env_extra_kwargs["camera_screen_effect"] = FLAGS.camera_screen_effect
    if task_id == "bimanual_unlock_ipad":
        env_extra_kwargs["ipad_screen_effect"] = FLAGS.ipad_screen_effect
    if FLAGS.physics_only:
        env_extra_kwargs["image_obs"] = False
    if sim_backend != "mujoco":
        env_extra_kwargs["sim_backend"] = sim_backend
        if FLAGS.render_width > 0 and FLAGS.render_height > 0:
            env_extra_kwargs["render_resolution"] = (
                int(FLAGS.render_height),
                int(FLAGS.render_width),
            )
        if task_id != "bimanual_assembly":
            env_extra_kwargs["arm_type"] = arm_type_override or "rokae"
            env_extra_kwargs["hand_type"] = hand_type_override or "wuji1"

    env = config.get_native_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=FLAGS.randomize,
        randomize_dynamics=False,
        seed=env_seed,
        **env_extra_kwargs,
    )
    runtime_recording_schema = None
    runtime_recording_fields: set[str] = set()
    seeded_origin_ready = False
    try:
        if reset_seed is None:
            obs, _info = env.reset()
        else:
            obs, _info = env.reset(seed=int(reset_seed))
            if physx_seeded_reset:
                from handson.sim.isaaclab.recording_origin import (
                    RecordingOriginStabilityError,
                    stable_recording_origin_kwargs_from_report,
                    stabilize_recording_origin,
                )

                stable_kwargs = stable_recording_origin_kwargs_from_report(
                    stable_origin_report
                )
                if (
                    physx_robot_path_replay_audit
                    and robot_path_init_mode == "stable_then_snapshot"
                ):
                    stable_kwargs["timeout_seconds"] = float(
                        stable_kwargs.get(
                            "stable_seconds",
                            0.5,
                        )
                    )
                try:
                    obs, _info = stabilize_recording_origin(
                        env,
                        obs,
                        _info,
                        **stable_kwargs,
                    )
                    seeded_origin_ready = True
                except RecordingOriginStabilityError as exc:
                    if not physx_robot_path_replay_audit:
                        raise
                    print(
                        "[PhysX robot path] recorded seeded origin did not "
                        "reproduce on this runtime; retaining its contact "
                        f"history before snapshot restore: {exc}",
                        flush=True,
                    )
                    raw_preroll_env = env.unwrapped
                    obs = env.observation(
                        raw_preroll_env._compute_observation()
                    )
                preroll_steps = max(
                    0,
                    int(os.environ.get("HANDSON_PHYSX_RESET_PREROLL_STEPS", "0")),
                )
                if preroll_steps:
                    preroll_env = env.unwrapped
                    for _ in range(preroll_steps):
                        preroll_env._apply_joint_targets()
                        preroll_env._step_isaaclab()
                    obs = env.observation(preroll_env._compute_observation())
        if FLAGS.recapture_physx_v4:
            runtime_recording_schema = resolve_environment_recording_schema(
                env
            )
            if not isinstance(runtime_recording_schema, dict) or int(
                runtime_recording_schema.get("version", 0)
            ) < 4:
                raise RuntimeError(
                    "PhysX v4 recapture requires a runtime recording schema "
                    "with version >= 4."
                )
            runtime_recording_fields = declared_recording_field_names(
                runtime_recording_schema
            )
        raw_env = env.unwrapped
        if sim_backend in {"isaacsim", "isaacsim_newton"}:
            reapply_reference_robot_appearance_for_replay(env)
        seeded_action_replay = bool(
            physx_replay is not None
            and physx_action_replay_audit
            and physx_seeded_reset
        )
        seeded_robot_path_replay = bool(
            physx_replay is not None
            and physx_robot_path_replay_audit
            and physx_seeded_reset
            and seeded_origin_ready
            and robot_path_init_mode == "stable_seeded"
        )
        skip_snapshot_initialization = bool(
            seeded_action_replay or seeded_robot_path_replay
        )
        state_layout = (
            _initial_state_layout(raw_env, config)
            if (FLAGS.restore_state and initial_state is not None)
            or recorded_states is not None
            else ()
        )

        if (
            FLAGS.restore_state
            and initial_state is not None
            and not skip_snapshot_initialization
        ):
            if sim_backend == "mujoco":
                from handson.tasks.state_restorers import (
                    has_restorer,
                    restore_initial_state,
                )

                if has_restorer(task_id):
                    obs = restore_initial_state(env, task_id, config, initial_state)
            else:
                restore = getattr(raw_env, "restore_recorded_initial_state", None)
                if restore is None:
                    print(
                        "[Warning] Isaac task does not support initial-state restoration; "
                        "replay uses the task reset pose."
                    )
                else:
                    obs = _restore_isaac_recorded_state(
                        env,
                        config,
                        initial_state,
                        state_layout,
                        visual=bool(
                            FLAGS.kinematic_visual_replay
                            and sim_backend == "isaacsim"
                            and recorded_states is not None
                        ),
                        recording_extras=_ycb02_recording_extras_at(
                            recording_state_extras,
                            0,
                        ),
                    )
                    if visual_arm_plan_qpos is not None:
                        apply_arm_qpos = getattr(
                            raw_env,
                            "apply_recorded_visual_arm_qpos",
                            None,
                        )
                        if not callable(apply_arm_qpos):
                            raise RuntimeError(
                                "Isaac task cannot apply the offline arm plan"
                            )
                        obs = env.observation(
                            apply_arm_qpos(visual_arm_plan_qpos[0])
                        )

        physx_trace = None
        physx_trajectory_verified = True
        physx_exact_replay = bool(
            physx_replay is not None
            and not physx_action_replay_audit
            and not physx_joint_target_replay_audit
            and not physx_robot_path_replay_audit
        )
        if physx_replay is not None:
            if sim_backend != "isaacsim":
                raise ValueError(
                    "PhysX ground-truth replay requires sim_backend=isaacsim"
                )
            from handson.sim.isaaclab.physx_ground_truth import (
                PhysxGroundTruthTrace,
            )

            physx_trace = PhysxGroundTruthTrace(raw_env)
            physx_trace.validate_restore_schema(physx_replay.schema)
            initial_physx_snapshot = physx_replay.snapshot(
                0,
                after_action=False,
            )
            restored = {}
            if not skip_snapshot_initialization:
                restored = physx_trace.restore(
                    initial_physx_snapshot,
                    physx_replay.schema,
                )
                if (
                    physx_robot_path_replay_audit
                    and robot_path_contact_prime_steps
                ):
                    # Build contact manifolds from the exact recorded public
                    # state before the official trajectory begins.  Restore
                    # the same snapshot afterwards so priming cannot alter
                    # any recorded robot or task-object initial condition.
                    physx_trace.apply_robot_joint_targets(
                        initial_physx_snapshot,
                        physx_replay.schema,
                    )
                    for _ in range(robot_path_contact_prime_steps):
                        physx_trace.restore_robot_state_only(
                            initial_physx_snapshot,
                            physx_replay.schema,
                        )
                        raw_env._step_isaaclab()
                    restored = physx_trace.restore(
                        initial_physx_snapshot,
                        physx_replay.schema,
                    )
                    print(
                        "[PhysX robot path] Primed contacts from the exact "
                        "recorded initial state for "
                        f"{robot_path_contact_prime_steps} physics steps, "
                        "then restored that state.",
                        flush=True,
                    )
            initial_readback_field_errors = (
                _physx_replay_readback_field_errors(
                    initial_physx_snapshot,
                    physx_trace.snapshot(),
                )
            )
            initial_readback_error = (
                max(initial_readback_field_errors.values())
                if initial_readback_field_errors
                else float("inf")
            )
            physx_trajectory_verified = bool(
                np.isfinite(initial_readback_error)
                and initial_readback_error <= _PHYSX_REPLAY_VERIFY_TOLERANCE
            )
            if not skip_snapshot_initialization:
                obs = env.observation(raw_env._compute_observation())
            replay_label = (
                "state replay"
                if physx_exact_replay
                else (
                    "kinematic robot-path replay audit"
                    if physx_robot_path_replay_audit
                    else (
                        "joint-target replay audit"
                        if physx_joint_target_replay_audit
                        else "action replay audit"
                    )
                )
            )
            if skip_snapshot_initialization:
                print(
                    f"[PhysX {replay_label}] Rebuilt source pre-action state "
                    "with recorded collection/episode seeds and no state "
                    f"writes: max_abs_error={initial_readback_error:.3e}"
                )
            else:
                print(
                    f"[PhysX {replay_label}] Restored source pre-action state: "
                    + ", ".join(
                        f"{name}={count}" for name, count in restored.items()
                    )
                    + f", max_abs_error={initial_readback_error:.3e}"
                )
        global_ik_plan = None
        global_ik_requested = (
            os.environ.get("HANDSON_GLOBAL_IK_PLAN", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if (
            not physx_joint_target_replay_audit
            and not physx_robot_path_replay_audit
            and global_ik_requested
        ):
            if physx_trace is None or physx_replay is None:
                # An inherited planner flag must not make ordinary action
                # replay fail.  Without a v2 PhysX trajectory there is no
                # valid plan to apply, so continue with the recorded actions.
                print(
                    "[Global IK plan] disabled: no v2 PhysX replay trajectory "
                    "was loaded for this replay mode.",
                    flush=True,
                )
            else:
                from handson.sim.isaaclab.global_ik_trajectory import (
                    build_global_ik_plan,
                )

                global_ik_plan = build_global_ik_plan(
                    raw_env,
                    actions,
                    physx_trace,
                    physx_replay,
                )
                print(
                    "[Global IK plan] "
                    + json.dumps(global_ik_plan.report, sort_keys=True),
                    flush=True,
                )

        if FLAGS.physics_only:
            camera_sources = {}
            camera_view_metadata = {}
        else:
            camera_sources = select_canonical_camera_sources(obs, env=env)
            camera_view_metadata = describe_camera_views(
                env,
                list(camera_sources.keys()),
            )

        trajectory = []
        succeed = False
        num_steps = actions.shape[0]
        total_steps = (
            num_steps
            if physx_replay is not None
            else num_steps + max(0, FLAGS.extend_steps)
        )
        if FLAGS.max_replay_steps > 0:
            total_steps = min(total_steps, int(FLAGS.max_replay_steps))
        if physx_replay is not None and FLAGS.extend_steps:
            print(
                "[Warning] --extend_steps is ignored during PhysX replay/audit "
                "because no source state exists after the final action."
            )
        kinematic_visual = bool(
            FLAGS.kinematic_visual_replay
            and sim_backend == "isaacsim"
            and recorded_states is not None
        )
        if FLAGS.kinematic_visual_replay and not kinematic_visual:
            print(
                "[Warning] --kinematic_visual_replay requires Isaac and recorded states; "
                "falling back to action replay."
            )

        for step_idx in tqdm(range(total_steps), desc=desc):
            action = actions[step_idx if step_idx < num_steps else -1]
            # Capture depth BEFORE step so it lines up with `obs` (which
            # is what gets stored alongside this transition's action).
            depth_now = (
                collect_depth_frames(env, list(camera_sources.keys()))
                if FLAGS.save_depth and not FLAGS.physics_only else None
            )
            if (
                physx_exact_replay
                and physx_trace is not None
                and physx_replay is not None
            ):
                expected_physx_snapshot = physx_replay.snapshot(
                    step_idx,
                    after_action=True,
                )
                restore_counts = physx_trace.restore(
                    expected_physx_snapshot,
                    physx_replay.schema,
                )
                physx_readback_field_errors = (
                    _physx_replay_readback_field_errors(
                        expected_physx_snapshot,
                        physx_trace.snapshot(),
                    )
                )
                physx_readback_error = (
                    max(physx_readback_field_errors.values())
                    if physx_readback_field_errors
                    else float("inf")
                )
                physx_trajectory_verified = bool(
                    physx_trajectory_verified
                    and np.isfinite(physx_readback_error)
                    and physx_readback_error
                    <= _PHYSX_REPLAY_VERIFY_TOLERANCE
                )
                evaluate = getattr(
                    raw_env,
                    "evaluate_restored_physx_replay_state",
                    None,
                )
                if callable(evaluate):
                    replay_info = evaluate()
                    if not isinstance(replay_info, dict):
                        replay_info = {
                            "env_succeed": bool(replay_info),
                        }
                else:
                    try:
                        env_succeed = bool(raw_env._compute_success())
                    except Exception:
                        env_succeed = False
                    replay_info = {"env_succeed": env_succeed}
                env_succeed = bool(replay_info.get("env_succeed", False))
                source_final_success = bool(
                    source_succeeded
                    and physx_trajectory_verified
                    and step_idx == total_steps - 1
                )
                step_succeed = bool(env_succeed or source_final_success)
                next_obs = env.observation(raw_env._compute_observation())
                done = bool(step_idx == total_steps - 1)
                info = {
                    **replay_info,
                    "succeed": step_succeed,
                    "physx_ground_truth_replay": True,
                    "physx_replay_mode": "state_restore",
                    "source_state_index": int(step_idx),
                    "source_outcome_success": bool(source_succeeded),
                    "source_outcome_used": bool(
                        source_final_success and not env_succeed
                    ),
                    "restore_counts": restore_counts,
                    "physx_initial_readback_max_abs_error": float(
                        initial_readback_error
                    ),
                    "physx_initial_readback_field_max_abs_error": (
                        initial_readback_field_errors
                    ),
                    "physx_readback_max_abs_error": float(
                        physx_readback_error
                    ),
                    "physx_readback_field_max_abs_error": (
                        physx_readback_field_errors
                    ),
                    "physx_trajectory_verified": bool(
                        physx_trajectory_verified
                    ),
                    "isaac_warnings": tuple(
                        getattr(raw_env, "_isaac_warnings", ())
                    ),
                }
            elif (
                physx_action_replay_audit
                and physx_trace is not None
                and physx_replay is not None
            ):
                if global_ik_plan is None:
                    next_obs, _rew, done, _trunc, step_info = (
                        _step_with_recorded_action(
                            env, action, action_storage_layout=action_storage_layout,
                        )
                    )
                    applied_global_ik_target_count = 0
                else:
                    from handson.sim.isaaclab.global_ik_trajectory import (
                        apply_global_ik_plan_step,
                    )

                    applied_global_ik_target_count = apply_global_ik_plan_step(
                        raw_env,
                        action,
                        global_ik_plan,
                        step_idx,
                    )
                    next_obs, _rew, done, _trunc, step_info = (
                        _step_with_current_joint_targets(env)
                    )
                if not isinstance(step_info, dict):
                    step_info = {}
                expected_physx_snapshot = physx_replay.snapshot(
                    step_idx,
                    after_action=True,
                )
                physx_readback_field_errors = (
                    _physx_replay_readback_field_errors(
                        expected_physx_snapshot,
                        physx_trace.snapshot(),
                    )
                )
                physx_readback_error = (
                    max(physx_readback_field_errors.values())
                    if physx_readback_field_errors
                    else float("inf")
                )
                state_verified = bool(
                    np.isfinite(physx_readback_error)
                    and physx_readback_error
                    <= _PHYSX_REPLAY_VERIFY_TOLERANCE
                )
                physx_trajectory_verified = bool(
                    physx_trajectory_verified and state_verified
                )
                info = {
                    **step_info,
                    "physx_action_replay_audit": True,
                    "physx_replay_mode": (
                        "global_smooth_ik_reexecution"
                        if global_ik_plan is not None
                        else "action_reexecution"
                    ),
                    "applied_global_ik_target_count": int(
                        applied_global_ik_target_count
                    ),
                    "global_ik_plan_ready_for_physical_replay": bool(
                        global_ik_plan is not None
                        and global_ik_plan.report.get(
                            "plan_ready_for_physical_replay",
                            False,
                        )
                    ),
                    "global_ik_target_physical_replay_success_rate": float(
                        global_ik_plan.report.get(
                            "target_physical_replay_success_rate",
                            0.90,
                        )
                        if global_ik_plan is not None
                        else 0.0
                    ),
                    "physx_initialization_mode": (
                        "seeded_reset"
                        if seeded_action_replay
                        else "snapshot_restore_legacy"
                    ),
                    "source_state_index": int(step_idx),
                    "source_outcome_success": bool(source_succeeded),
                    "source_outcome_used": False,
                    "physx_initial_readback_max_abs_error": float(
                        initial_readback_error
                    ),
                    "physx_initial_readback_field_max_abs_error": (
                        initial_readback_field_errors
                    ),
                    "physx_readback_max_abs_error": float(
                        physx_readback_error
                    ),
                    "physx_readback_field_max_abs_error": (
                        physx_readback_field_errors
                    ),
                    "physx_state_verified": state_verified,
                    "physx_trajectory_verified": bool(
                        physx_trajectory_verified
                    ),
                    "isaac_warnings": tuple(
                        getattr(raw_env, "_isaac_warnings", ())
                    ),
                }
            elif (
                physx_robot_path_replay_audit
                and physx_trace is not None
                and physx_replay is not None
            ):
                before_physx_snapshot = physx_replay.snapshot(
                    step_idx,
                    after_action=False,
                )
                expected_physx_snapshot = physx_replay.snapshot(
                    step_idx,
                    after_action=True,
                )
                next_obs, _rew, done, _trunc, step_info = (
                    _step_with_recorded_robot_path(
                        env,
                        physx_trace,
                        physx_replay.schema,
                        before_physx_snapshot,
                        expected_physx_snapshot,
                    )
                )
                if not isinstance(step_info, dict):
                    step_info = {}
                physx_readback_field_errors = (
                    _physx_replay_readback_field_errors(
                        expected_physx_snapshot,
                        physx_trace.snapshot(),
                    )
                )
                pre_correction_robot_errors = step_info.get(
                    "physx_robot_pre_correction_readback_field_max_abs_error",
                    {},
                )
                if isinstance(pre_correction_robot_errors, dict):
                    for field_name, field_error in (
                        pre_correction_robot_errors.items()
                    ):
                        physx_readback_field_errors[field_name] = max(
                            float(
                                physx_readback_field_errors.get(
                                    field_name,
                                    0.0,
                                )
                            ),
                            float(field_error),
                        )
                physx_readback_error = (
                    max(physx_readback_field_errors.values())
                    if physx_readback_field_errors
                    else float("inf")
                )
                state_verified = bool(
                    np.isfinite(physx_readback_error)
                    and physx_readback_error
                    <= _PHYSX_REPLAY_VERIFY_TOLERANCE
                )
                physx_trajectory_verified = bool(
                    physx_trajectory_verified and state_verified
                )
                info = {
                    **step_info,
                    "physx_robot_path_replay_audit": True,
                    "physx_replay_mode": (
                        "guided_physical_robot_path_reexecution"
                        if bool(step_info.get("guided_physics_replay"))
                        else "kinematic_robot_path_reexecution"
                    ),
                    "source_state_index": int(step_idx),
                    "source_outcome_success": bool(source_succeeded),
                    "source_outcome_used": False,
                    "robot_state_written_each_substep": True,
                    "task_object_state_written_during_replay": False,
                    "task_object_external_tracking_wrenches": bool(
                        step_info.get("guided_physics_replay")
                    ),
                    "contact_solver": "physx",
                    "physx_initial_readback_max_abs_error": float(
                        initial_readback_error
                    ),
                    "physx_initial_readback_field_max_abs_error": (
                        initial_readback_field_errors
                    ),
                    "physx_readback_max_abs_error": float(
                        physx_readback_error
                    ),
                    "physx_readback_field_max_abs_error": (
                        physx_readback_field_errors
                    ),
                    "physx_state_verified": state_verified,
                    "physx_trajectory_verified": bool(
                        physx_trajectory_verified
                    ),
                    "isaac_warnings": tuple(
                        getattr(raw_env, "_isaac_warnings", ())
                    ),
                }
            elif (
                physx_joint_target_replay_audit
                and physx_trace is not None
                and physx_replay is not None
            ):
                expected_physx_snapshot = physx_replay.snapshot(
                    step_idx,
                    after_action=True,
                )
                applied_target_count = (
                    physx_trace.apply_robot_joint_targets(
                        expected_physx_snapshot,
                        physx_replay.schema,
                    )
                )
                next_obs, _rew, done, _trunc, step_info = (
                    _step_with_current_joint_targets(env)
                )
                if not isinstance(step_info, dict):
                    step_info = {}
                physx_readback_field_errors = (
                    _physx_replay_readback_field_errors(
                        expected_physx_snapshot,
                        physx_trace.snapshot(),
                    )
                )
                physx_readback_error = (
                    max(physx_readback_field_errors.values())
                    if physx_readback_field_errors
                    else float("inf")
                )
                state_verified = bool(
                    np.isfinite(physx_readback_error)
                    and physx_readback_error
                    <= _PHYSX_REPLAY_VERIFY_TOLERANCE
                )
                physx_trajectory_verified = bool(
                    physx_trajectory_verified and state_verified
                )
                info = {
                    **step_info,
                    "physx_joint_target_replay_audit": True,
                    "physx_replay_mode": "joint_target_reexecution",
                    "source_state_index": int(step_idx),
                    "source_outcome_success": bool(source_succeeded),
                    "source_outcome_used": False,
                    "applied_robot_joint_target_count": int(
                        applied_target_count
                    ),
                    "recorded_velocity_drive_target_available": False,
                    "physx_initial_readback_max_abs_error": float(
                        initial_readback_error
                    ),
                    "physx_initial_readback_field_max_abs_error": (
                        initial_readback_field_errors
                    ),
                    "physx_readback_max_abs_error": float(
                        physx_readback_error
                    ),
                    "physx_readback_field_max_abs_error": (
                        physx_readback_field_errors
                    ),
                    "physx_state_verified": state_verified,
                    "physx_trajectory_verified": bool(
                        physx_trajectory_verified
                    ),
                    "isaac_warnings": tuple(
                        getattr(raw_env, "_isaac_warnings", ())
                    ),
                }
            elif kinematic_visual:
                source_index = min(step_idx + 1, len(recorded_states) - 1)
                raw_env = env.unwrapped
                next_obs = _restore_isaac_recorded_state(
                    env,
                    config,
                    recorded_states[source_index],
                    state_layout,
                    visual=True,
                    recording_extras=_ycb02_recording_extras_at(
                        recording_state_extras,
                        source_index,
                    ),
                )
                if visual_arm_plan_qpos is not None:
                    apply_arm_qpos = getattr(
                        raw_env,
                        "apply_recorded_visual_arm_qpos",
                        None,
                    )
                    if not callable(apply_arm_qpos):
                        raise RuntimeError(
                            "Isaac task cannot apply the offline arm plan"
                        )
                    next_obs = env.observation(
                        apply_arm_qpos(visual_arm_plan_qpos[source_index])
                    )
                success_now = False
                try:
                    success_now = bool(raw_env._compute_success())
                except Exception:
                    success_now = False
                done = bool(source_index >= len(recorded_states) - 1)
                info = {
                    "succeed": success_now,
                    "kinematic_visual_replay": True,
                    "source_state_index": int(source_index),
                    "isaac_warnings": tuple(getattr(raw_env, "_isaac_warnings", ())),
                }
            else:
                next_obs, _rew, done, _trunc, info = _step_with_recorded_action(
                    env, action, action_storage_layout=action_storage_layout,
                )
                if (
                    FLAGS.state_guided
                    and sim_backend == "isaacsim"
                    and recorded_states is not None
                ):
                    source_index = min(step_idx + 1, len(recorded_states) - 1)
                    raw_env = env.unwrapped
                    restore = getattr(raw_env, "restore_recorded_initial_state", None)
                    if restore is not None:
                        next_obs = _restore_isaac_recorded_state(
                            env,
                            config,
                            recorded_states[source_index],
                            state_layout,
                            visual=False,
                            recording_extras=_ycb02_recording_extras_at(
                                recording_state_extras,
                                source_index,
                            ),
                        )
            transition = dict(
                observations=obs,
                actions=action,
                dones=done,
                infos=info,
            )
            if FLAGS.recapture_physx_v4:
                transition["next_observations"] = {
                    name: next_obs[name]
                    for name in runtime_recording_fields
                    if name in next_obs
                }
            if depth_now is not None:
                transition["depth"] = depth_now
            frozen_transition = copy.deepcopy(transition)
            if transition_callback is not None:
                transition_callback(frozen_transition, camera_sources)
            if not retain_images:
                _discard_transition_images(
                    frozen_transition,
                    camera_sources,
                )
            trajectory.append(frozen_transition)
            obs = next_obs
            if info.get("succeed", False):
                succeed = True
    finally:
        if sim_backend == "isaacsim_newton":
            # A Newton world owns process-wide singleton state.  Process exit
            # alone is not an acceptable replay teardown contract: a later
            # replay must be able to build a genuinely fresh scene even when
            # this function fails part-way through the trajectory.
            from handson.sim.isaaclab.newton_replay import (
                destroy_newton_scene,
            )

            scene_lifecycle = destroy_newton_scene(env)
            if not scene_lifecycle.get("scene_destroyed", False):
                raise RuntimeError("Newton replay did not destroy its scene")
            print("[Newton replay] scene destroyed", flush=True)
        elif close_env:
            try:
                if sim_backend == "isaacsim":
                    _enable_isaac_fast_shutdown(env)
                env.close()
            except Exception:
                pass

    return (
        succeed,
        trajectory,
        camera_view_metadata,
        camera_sources,
        runtime_recording_schema,
    )


def main(_argv):
    task_id = FLAGS.exp_name

    for label, width, height in (
        ("main", FLAGS.render_width, FLAGS.render_height),
        ("wrist", FLAGS.wrist_render_width, FLAGS.wrist_render_height),
    ):
        if (width > 0) != (height > 0):
            raise app.UsageError(
                f"{label} render width and height must be set together"
            )
        if width < 0 or height < 0:
            raise app.UsageError(
                f"{label} render dimensions must be nonnegative"
            )
    if FLAGS.video_crf < 0:
        raise app.UsageError("--video_crf must be nonnegative")
    if FLAGS.max_replay_steps < 0:
        raise app.UsageError("--max_replay_steps must be nonnegative")
    if FLAGS.stream_videos and FLAGS.physics_only:
        raise app.UsageError(
            "--stream_videos cannot be combined with --physics_only"
        )
    if FLAGS.wrist_render_width > 0:
        os.environ["HANDSON_ISAACLAB_WRIST_CAMERA_OUTPUT_SIZE"] = (
            f"{int(FLAGS.wrist_render_width)}x"
            f"{int(FLAGS.wrist_render_height)}"
        )

    base_out = Path(FLAGS.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    input_root = Path(FLAGS.input_dir)
    demo_dirs = sorted(p.parent for p in input_root.glob("*/replay.zarr"))
    if not demo_dirs:
        print(f"No replay.zarr found under {input_root}")
        return

    saved_demo_dirs = []
    physx_modes = (
        FLAGS.physx_ground_truth_replay,
        FLAGS.physx_action_replay_audit,
        FLAGS.physx_joint_target_replay_audit,
        FLAGS.physx_robot_path_replay_audit,
    )
    if sum(bool(enabled) for enabled in physx_modes) > 1:
        raise app.UsageError(
            "--physx_ground_truth_replay, --physx_action_replay_audit, and "
            "--physx_joint_target_replay_audit/--physx_robot_path_replay_audit "
            "are mutually exclusive"
        )
    physx_mode_enabled = any(physx_modes)
    if FLAGS.recapture_physx_v4 and not FLAGS.physx_action_replay_audit:
        raise app.UsageError(
            "--recapture_physx_v4 requires "
            "--physx_action_replay_audit=True."
        )
    if physx_mode_enabled and (
        FLAGS.state_guided or FLAGS.kinematic_visual_replay
    ):
        raise app.UsageError(
            "PhysX replay/audit modes cannot be combined with --state_guided "
            "or --kinematic_visual_replay"
        )
    if physx_mode_enabled and len(demo_dirs) != 1:
        raise app.UsageError(
            "PhysX replay/audit currently requires input_dir to contain "
            "exactly one demo. Isaac RTX replay uses process-level shutdown "
            "after publishing the output."
        )
    newton_mode_used = False
    for index, demo_dir in enumerate(demo_dirs, start=1):
        print(f"\n[{index}/{len(demo_dirs)}] {demo_dir.name}")
        zarr_path = demo_dir / "replay.zarr"
        sim_backend = _sim_backend_for_demo(demo_dir, FLAGS.sim_backend)
        if sim_backend == "isaacsim_newton":
            if len(demo_dirs) != 1:
                raise app.UsageError(
                    "NVIDIA Newton replay currently requires input_dir to "
                    "contain exactly one demo. Run each demo in a fresh "
                    "process so the Newton world is rebuilt cleanly."
                )
            newton_mode_used = True
        print(f"Replay backend: {sim_backend}")
        has_physx_substep_tcp_trace = any(
            (
                zarr_path
                / prefix
                / "next_physx_substep_tcp_state"
            ).is_dir()
            for prefix in ("", "data")
        )
        if (
            sim_backend == "isaacsim"
            and has_physx_substep_tcp_trace
            and FLAGS.physx_action_replay_audit
        ):
            # Only the explicit action-audit mode loads a v2 PhysX trajectory
            # and can consume a global IK plan.  Ordinary action replay may
            # contain the same substep TCP fields, but has no trajectory to
            # plan against; auto-enabling the planner there caused a late
            # startup exception that looked like a hung replay.
            os.environ.setdefault("HANDSON_GLOBAL_IK_PLAN", "1")
            os.environ.setdefault(
                "HANDSON_GLOBAL_IK_STRICT_TRACKING",
                "1",
            )
            if (
                os.environ.get("HANDSON_GLOBAL_IK_PLAN", "1")
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            ):
                print(
                    "[Global IK plan] auto-enabled from recorded PhysX "
                    "substep TCP trace.",
                    flush=True,
                )
            else:
                print(
                    "[Global IK plan] explicitly disabled; replaying the "
                    "recorded TCP/hand actions in Isaac.",
                    flush=True,
                )
        replay_manifest = _load_replay_manifest(demo_dir)
        recording_parameters = _manifest_recording_parameters(
            replay_manifest
        )
        action_storage_layout = _recorded_action_storage_layout(
            recording_parameters, task_id=task_id,
        )
        simplified_post_ik = bool(
            declares_simplified_post_ik(recording_parameters)
        )
        if simplified_post_ik and not physx_mode_enabled:
            if sim_backend != "isaacsim":
                raise RuntimeError(
                    "Simplified post-IK replay requires the Isaac Sim visual "
                    "backend."
                )
            plan_errors = simplified_arm_plan_errors(zarr_path)
            if plan_errors:
                configured_planner = str(
                    os.environ.get("HANDSON_E01_ARM_PLAN_PYTHON")
                    or os.environ.get("HANDSON_PYTHON")
                    or ""
                ).strip()
                planner_python = sys.executable
                if configured_planner:
                    candidate = Path(configured_planner).expanduser().resolve()
                    if not candidate.is_file() or not os.access(
                        candidate,
                        os.X_OK,
                    ):
                        raise RuntimeError(
                            "Configured offline IK Python is not executable: "
                            f"{candidate}"
                        )
                    planner_python = str(candidate)
                plan_command = [
                    planner_python,
                    str(
                        Path(__file__).resolve().parent
                        / "simplified_post_ik_plan.py"
                    ),
                    str(demo_dir),
                    "--overwrite",
                ]
                print(
                    "[Simplified post-IK] computing validated smooth arm plan: "
                    + "; ".join(plan_errors),
                    flush=True,
                )
                subprocess.run(plan_command, check=True)
            FLAGS.restore_state = True
            FLAGS.state_guided = False
            FLAGS.kinematic_visual_replay = True
            os.environ["HANDSON_ISAACLAB_DEFER_ARM_RENDER"] = "0"
            os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "0"
            os.environ["HANDSON_ISAACLAB_DIRECT_HAND_CONTROL"] = "0"
        collection_run_seed = _optional_manifest_seed(
            recording_parameters,
            "collection_run_seed",
        )
        episode_reset_seed = _optional_manifest_seed(
            recording_parameters,
            "episode_reset_seed",
        )
        if sim_backend in {"isaacsim", "isaacsim_newton"}:
            _apply_recorded_handson_environment(replay_manifest)
            _apply_isaac_render_overrides(
                wrist_width=int(FLAGS.wrist_render_width),
                wrist_height=int(FLAGS.wrist_render_height),
                force_complete_arm_render=bool(
                    FLAGS.force_complete_arm_render
                ),
            )
            if simplified_post_ik and not physx_mode_enabled:
                os.environ["HANDSON_ISAACLAB_DEFER_ARM_RENDER"] = "0"
                os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "0"
                os.environ["HANDSON_ISAACLAB_DIRECT_HAND_CONTROL"] = "0"
                os.environ.pop("HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY", None)
            if physx_mode_enabled:
                os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
        config = CONFIG_MAPPING[task_id]()
        actions = _load_episode_actions(zarr_path)
        initial_state = _load_episode_initial_state(zarr_path)
        recording_state_extras = _load_ycb02_recording_state_extras(
            zarr_path,
            expected_steps=(
                int(actions.shape[0])
                if task_id in {"t27_place_largest_fitting_ball", "eval_comb01_cover_pitcher_and_box"}
                else None
            ),
            field_shapes=(
                {"xy5_hidden_state": ((21,),), "xy5_reset_poses": ((4, 7),)}
                if task_id == "eval_comb01_cover_pitcher_and_box" else None
            ),
            task_label="XY5" if task_id == "eval_comb01_cover_pitcher_and_box" else "YCB02",
        )
        if task_id == "eval_comb06_season_and_stir":
            recording_state_extras.update(_load_xy6_recording_state_extras(
                zarr_path, expected_steps=int(actions.shape[0]),
            ))
        physx_replay = (
            _load_physx_replay_trajectory(
                zarr_path,
                demo_dir,
                expected_steps=actions.shape[0],
            )
            if (
                FLAGS.physx_ground_truth_replay
                or FLAGS.physx_action_replay_audit
                or FLAGS.physx_joint_target_replay_audit
                or FLAGS.physx_robot_path_replay_audit
            )
            else None
        )
        recorded_states = (
            _load_episode_states(zarr_path)
            if (FLAGS.state_guided or FLAGS.kinematic_visual_replay)
            else None
        )
        visual_arm_plan_qpos = (
            load_validated_simplified_arm_plan(
                zarr_path,
                expected_frames=(
                    len(recorded_states)
                    if recorded_states is not None
                    else None
                ),
            )
            if simplified_post_ik and not physx_mode_enabled
            else None
        )
        if FLAGS.restore_state and initial_state is None:
            print("[Warning] state[0] not found in input zarr; replay will use the reset scene as-is.")

        fallback_seed = FLAGS.seed + index
        environment_seed = (
            collection_run_seed
            if collection_run_seed is not None
            else fallback_seed
        )
        replay_reset_seed = (
            episode_reset_seed
            if episode_reset_seed is not None
            else environment_seed
        )
        physx_seeded_reset = bool(
            (
                FLAGS.physx_action_replay_audit
                or (
                    FLAGS.physx_robot_path_replay_audit
                    and os.environ.get(
                        "HANDSON_ROBOT_PATH_INIT_MODE",
                        "snapshot",
                    ).strip().lower()
                    in {"stable_then_snapshot", "stable_seeded"}
                )
            )
            and episode_reset_seed is not None
            and os.environ.get(
                "HANDSON_FORCE_PHYSX_SNAPSHOT_INIT",
                "",
            ).strip().lower()
            not in {"1", "true", "yes", "on"}
        )
        if FLAGS.physx_action_replay_audit:
            if physx_seeded_reset:
                print(
                    "PhysX action replay initialization: recorded seeded reset "
                    f"(collection={environment_seed}, episode={replay_reset_seed})"
                )
            else:
                print(
                    "[Warning] Replay manifest has no episode_reset_seed; "
                    "using legacy pre-action snapshot restoration."
                )
        elif FLAGS.physx_robot_path_replay_audit:
            robot_path_init_mode = os.environ.get(
                "HANDSON_ROBOT_PATH_INIT_MODE",
                "snapshot",
            ).strip().lower()
            if physx_seeded_reset:
                print(
                    "PhysX robot-path initialization: "
                    f"{robot_path_init_mode} with recorded seeded reset "
                    f"(collection={environment_seed}, "
                    f"episode={replay_reset_seed})"
                )
            elif robot_path_init_mode != "snapshot":
                print(
                    "[Warning] Replay manifest has no episode_reset_seed; "
                    "falling back to snapshot robot-path initialization."
                )

        streaming_writer = (
            _StreamingCanonicalVideoWriter(
                base_out,
                fps=FLAGS.video_fps,
                crf=FLAGS.video_crf,
            )
            if FLAGS.stream_videos
            else None
        )
        streamed_videos_dir = None
        try:
            (
                succeed,
                trajectory,
                camera_view_metadata,
                camera_sources,
                runtime_recording_schema,
            ) = _replay_single_demo(
                actions,
                initial_state,
                recorded_states,
                task_id,
                config,
                env_seed=environment_seed,
                reset_seed=replay_reset_seed,
                desc=demo_dir.name,
                sim_backend=sim_backend,
                physx_replay=physx_replay,
                physx_action_replay_audit=FLAGS.physx_action_replay_audit,
                physx_joint_target_replay_audit=(
                    FLAGS.physx_joint_target_replay_audit
                ),
                physx_robot_path_replay_audit=(
                    FLAGS.physx_robot_path_replay_audit
                ),
                physx_seeded_reset=physx_seeded_reset,
                stable_origin_report=recording_parameters.get(
                    "stable_recording_origin"
                ),
                source_succeeded=_demo_source_succeeded(demo_dir),
                close_env=not (physx_mode_enabled or newton_mode_used),
                transition_callback=(
                    streaming_writer.write_transition
                    if streaming_writer is not None
                    else None
                ),
                retain_images=streaming_writer is None,
                visual_arm_plan_qpos=visual_arm_plan_qpos,
                arm_type_override=(
                    "rokae_simplified_post_ik"
                    if simplified_post_ik and physx_mode_enabled
                    else "rokae"
                ),
                hand_type_override="wuji1",
                recording_state_extras=recording_state_extras,
                action_storage_layout=action_storage_layout,
            )
            if streaming_writer is not None:
                streamed_videos_dir = streaming_writer.finish(
                    camera_view_metadata
                )
                print(
                    "[Streaming videos] "
                    f"encoded {streaming_writer.frame_count} frames in "
                    f"{streamed_videos_dir}"
                )
        except BaseException:
            if streaming_writer is not None:
                streaming_writer.abort()
            raise
        print(f"Replay finished: succeed={succeed}, steps={len(trajectory)}")

        if not trajectory:
            if streamed_videos_dir is not None:
                shutil.rmtree(streamed_videos_dir, ignore_errors=True)
            continue
        if not (succeed or FLAGS.save_failed):
            print("[Skipped] env did not report success; pass --save_failed to keep.")
            if streamed_videos_dir is not None:
                shutil.rmtree(streamed_videos_dir, ignore_errors=True)
            continue

        saved = _write_demo_zarr_and_videos(
            trajectory,
            task_id,
            index,
            base_out,
            FLAGS.video_fps,
            camera_view_metadata=camera_view_metadata,
            camera_sources=camera_sources,
            recapture_environment_schema=(
                runtime_recording_schema
                if FLAGS.recapture_physx_v4
                else None
            ),
            source_demo_dir=demo_dir if FLAGS.recapture_physx_v4 else None,
            succeeded=succeed,
            streamed_videos_dir=streamed_videos_dir,
        )
        saved_demo_dirs.append(saved)

    print(f"\nReplayed {len(saved_demo_dirs)}/{len(demo_dirs)} demos:")
    for p in saved_demo_dirs:
        print("  ", p)
    if physx_mode_enabled or newton_mode_used:
        # SimulationContext.stop() can block indefinitely after RTX capture.
        # All zarr/video outputs are durable at this point, so use the same
        # process-level shutdown contract as isaac_render_arm_replay.py.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    app.run(main)
