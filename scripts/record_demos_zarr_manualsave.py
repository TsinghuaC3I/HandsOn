#!/usr/bin/env python3
"""
Record successful simulated teleoperation demos as Zarr episodes plus MP4 videos.

For each successful episode, the script writes:
- `replay.zarr/` with low-dimensional observations and actions
- exactly three canonical videos: `main.mp4`, `wrist_left.mp4`, `wrist_right.mp4`
- `videos/camera_info.json` and `videos/<camera_key>_camera_info.json` sidecars
  with the MuJoCo view pose plus scaled fisheye intrinsics for wrist cameras
"""

import copy
import datetime
import os
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "handson"
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import cv2
import numpy as np
import zarr
from absl import app, flags

# Project-specific utilities for replay storage and video writing.
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
    select_current_observation_for_recording,
    select_next_observation_for_recording,
    stack_transition_numeric_observation_fields,
    transition_timestamps,
    write_recording_schema,
    write_replay_assets,
)
from handson.data.video_writer import Mp4VideoWriter
from handson.tasks.mappings import CONFIG_MAPPING
from scipy.spatial.transform import Rotation
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "exp_name",
    "water_plant",
    "Task name, such as water_plant.",
)
flags.DEFINE_integer("successes_needed", 2, "Number of successful demos to collect.")
flags.DEFINE_string(
    "save_key",
    "s",
    "Single key that force-saves the current trajectory regardless of task success.",
)
flags.DEFINE_bool("show_sim_cameras", True, "Show simulation cameras")
flags.DEFINE_integer(
    "max_steps",
    0,
    "Stop after this many environment steps; 0 means run until successes_needed",
)
flags.DEFINE_string("out_dir", "./", "Output base directory for zarr and videos")
flags.DEFINE_integer("video_fps", 30, "FPS for saved MP4 videos")
flags.DEFINE_float(
    "data_fps",
    30,
    "Sampling frequency of recorded low-dim data in Hz (used to write timestamps)",
)
flags.DEFINE_enum(
    "render_mode",
    "human",
    ["human", "rgb_array"],
    "MuJoCo render mode for the task environment",
)
flags.DEFINE_bool(
    "randomize",
    False,
    "Enable environment randomization when the selected task supports it",
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
flags.DEFINE_string(
    "target_label",
    "",
    "Optional task target label, for tasks that support target-specific reset.",
)
flags.DEFINE_enum(
    "arm_type",
    "panda",
    ["panda", "rokae"],
    "Arm XML family to load. Use rokae with --hand_type=wuji1 for ROKAE+Wuji.",
)
flags.DEFINE_enum(
    "hand_type",
    "allegro",
    ["allegro", "wuji1"],
    "Hand XML family to load.",
)
flags.DEFINE_enum(
    "sim_backend",
    "mujoco",
    ["mujoco", "isaacsim"],
    "Simulation backend to use for task construction.",
)
flags.DEFINE_bool(
    "record_physx_ground_truth",
    True,
    "For Isaac Sim, record action-aligned robot/task PhysX readback in physx_* fields.",
)


def _ensure_base_outdir(base: str) -> Path:
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_squeeze_image(img: np.ndarray) -> np.ndarray:
    """Squeeze common single-batch dimensions and ensure HWC uint8 RGB.

    Accepts image formats: (1,H,W,3), (H,W,3), (H,W) (grayscale).
    Returns (H,W,3) uint8 RGB.
    """
    if img is None:
        return None
    arr = np.asarray(img)
    # squeeze batch dim if present
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    # grayscale to 3-channel
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    # if single-channel last dim, expand
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=2)
    # ensure dtype is uint8
    if arr.dtype != np.uint8:
        # try to scale floats in [0,1] to [0,255]
        if np.issubdtype(arr.dtype, np.floating):
            if np.nanmax(arr) <= 1.0:
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            else:
                arr = np.clip(arr, 0.0, 255.0)
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    # At this point arr is uint8 but might be BGR or RGB depending on source.
    # We assume observations are RGB (common in sim). If you know they are
    # BGR, flip channels where needed before writing with Mp4VideoWriter.
    return arr


def _to_bgr_image(image: np.ndarray) -> np.ndarray:
    """Convert a top-level observation image into HWC uint8 BGR for OpenCV."""
    if image is None:
        return None

    frame = np.asarray(image)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    if frame.ndim != 3 or frame.shape[2] != 3:
        return None

    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            if np.nanmax(frame) <= 1.0:
                frame = np.clip(frame, 0.0, 1.0) * 255.0
            else:
                frame = np.clip(frame, 0.0, 255.0)
            frame = frame.astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


class WristCameraViewer:
    """Display only wrist cameras during teleoperation data collection."""

    WINDOW_TITLES = {
        "wrist": "Wrist",
        "wrist_left": "Wrist Left",
        "wrist_right": "Wrist Right",
    }

    def __init__(self, display_scale=0.75):
        self.display_scale = display_scale
        self.active_windows = {}

    def _get_visible_keys(self, observation):
        if not isinstance(observation, dict):
            return []

        if _to_bgr_image(observation.get("wrist")) is not None:
            return ["wrist"]

        visible_keys = []
        for camera_key in ("wrist_left", "wrist_right"):
            if _to_bgr_image(observation.get(camera_key)) is not None:
                visible_keys.append(camera_key)
        return visible_keys

    def _ensure_windows(self, visible_keys):
        active_keys = set(self.active_windows)
        visible_key_set = set(visible_keys)

        for camera_key in active_keys - visible_key_set:
            try:
                cv2.destroyWindow(self.active_windows[camera_key])
            except cv2.error:
                pass
            del self.active_windows[camera_key]

        for camera_key in visible_keys:
            if camera_key not in self.active_windows:
                window_name = self.WINDOW_TITLES[camera_key]
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                self.active_windows[camera_key] = window_name

    def update_and_show(self, observation):
        """Show wrist-only windows and return the last pressed key."""
        visible_keys = self._get_visible_keys(observation)
        self._ensure_windows(visible_keys)

        if not visible_keys:
            return cv2.waitKey(1) & 0xFF

        for camera_key in visible_keys:
            frame = _to_bgr_image(observation.get(camera_key))
            if frame is None:
                continue
            if self.display_scale != 1.0:
                new_width = max(1, int(frame.shape[1] * self.display_scale))
                new_height = max(1, int(frame.shape[0] * self.display_scale))
                frame = cv2.resize(
                    frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
                )
            window_name = self.active_windows[camera_key]
            cv2.imshow(window_name, frame)
            cv2.resizeWindow(window_name, frame.shape[1], frame.shape[0])

        return cv2.waitKey(1) & 0xFF

    def close(self):
        for window_name in list(self.active_windows.values()):
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
        self.active_windows.clear()


def convert_action_quat_to_rotvec(action: np.ndarray) -> np.ndarray:
    """
    Convert action quaternions from wxyz to rotation vectors.

    Supported layouts:
      - single-arm 23-d Allegro: [pos3, quat4, 16 joints] -> 22-d
      - single-arm 27-d Wuji: [pos3, quat4, 20 joints] -> 26-d
      - bimanual 46-d Allegro: [right23, left23] -> 44-d
      - bimanual 54-d Wuji: [right27, left27] -> 52-d

    If quaternion has zero norm, treat it as identity rotation.
    """
    action = np.asarray(action)
    if action.ndim != 1:
        return None

    def _convert_single_arm(single_arm_action: np.ndarray) -> np.ndarray:
        pos = single_arm_action[0:3]
        quat = single_arm_action[3:7]
        hand_joints = single_arm_action[7:]

        quat = quat.astype(np.float64)
        norm = np.linalg.norm(quat)

        if norm < 1e-8:
            rotvec = np.zeros(3, dtype=np.float64)
        else:
            quat = quat / norm
            quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]
            rot = Rotation.from_quat(quat_xyzw)
            rotvec = rot.as_rotvec()

        return np.concatenate([pos, rotvec, hand_joints])

    if action.shape[0] in (23, 27):
        return _convert_single_arm(action)

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


def _flatten_action_for_storage(action) -> np.ndarray:
    if isinstance(action, dict):
        if "right" not in action or "left" not in action:
            raise ValueError(
                f"Dict action must contain right and left keys, got {sorted(action)}."
            )
        return np.concatenate(
            [
                np.asarray(action["right"], dtype=np.float64).reshape(-1),
                np.asarray(action["left"], dtype=np.float64).reshape(-1),
            ],
            axis=0,
        )
    return np.asarray(action, dtype=np.float64).reshape(-1)


def _write_demo_zarr_and_videos(
    trajectory, exp_name: str, success_index: int, base_out: Path, video_fps: int,
    tag: str = "demo", env=None, environment_schema=None,
    required_recording_fields=None,
):
    """Save a single demo trajectory to a zarr group and MP4 camera captures.

    Structure created:
      <base_out>/<exp_name>_<tag>_<index>_<timestamp>/
         replay.zarr/
         videos/<cam_key>.mp4

    ``tag`` is "demo" for success-triggered saves and "manual" for the
    keyboard `s` force-save (this manual-save build).

    The episode store will contain arrays for:
      - action: (T, N)
      - action_rotvec: (T,22/26) for single-arm or (T,44/52) for bimanual if convertible
      - state: (T, ... ) if present under observations['state']
      - timestamp: (T,) in seconds
      - any small consistent fixed-shape numeric observation fields
      - recording_schema.json plus matching Zarr attrs describing every field
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    demo_dir = base_out / f"{exp_name}_{tag}_{success_index}_{timestamp}"
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Prepare episode-level low-dim arrays.
    actions = []
    states = []
    next_states = []

    if required_recording_fields is None:
        if environment_schema is None:
            environment_schema = resolve_environment_recording_schema(env)
        required_recording_fields = declared_recording_field_names(
            environment_schema
        )
    else:
        required_recording_fields = set(required_recording_fields)
    has_next_observations = all(
        isinstance(transition.get("next_observations"), dict)
        for transition in trajectory
    )

    for transition in trajectory:
        obs = transition.get("observations", {})
        act = transition.get("actions")
        actions.append(np.asarray(act))

        if isinstance(obs, dict) and ("state" in obs or "low_dim" in obs):
            s = obs.get("state", obs.get("low_dim"))
            states.append(np.asarray(s))
        if has_next_observations:
            next_obs = transition["next_observations"]
            if "state" in next_obs or "low_dim" in next_obs:
                next_s = next_obs.get("state", next_obs.get("low_dim"))
                next_states.append(np.asarray(next_s))

    # Build episode dict for the Zarr episode store.
    actions_arr = np.stack(actions, axis=0)
    T = actions_arr.shape[0]
    timestamps_arr = transition_timestamps(
        trajectory,
        fallback_hz=float(FLAGS.data_fps),
    )

    episode_data = {
        "action": actions_arr,
        "timestamp": timestamps_arr,
    }

    if len(states) > 0:
        try:
            episode_data["state"] = np.stack(states, axis=0)
        except Exception:
            episode_data["state"] = np.array(states, dtype=object)
    if len(next_states) == T:
        episode_data["next_state"] = np.stack(next_states, axis=0)

    episode_data.update(
        stack_transition_numeric_observation_fields(
            trajectory,
            required_fields=required_recording_fields,
            environment_schema=environment_schema,
        )
    )

    # Prefer a synchronized simulator clock over an assumed recorder rate.
    sim_time = episode_data.get("physx_sim_time")
    if sim_time is None:
        sim_time = episode_data.get("c01_force_sim_time")
    if sim_time is not None and np.asarray(sim_time).shape == (T, 1):
        episode_data["timestamp"] = np.asarray(sim_time, dtype=np.float64)[:, 0]
    next_sim_time = episode_data.get("next_physx_sim_time")
    if next_sim_time is None:
        next_sim_time = episode_data.get("next_c01_force_sim_time")
    if next_sim_time is not None and np.asarray(next_sim_time).shape == (T, 1):
        episode_data["next_timestamp"] = np.asarray(
            next_sim_time,
            dtype=np.float64,
        )[:, 0]

    # action_rotvec when convertible
    converted = [convert_action_quat_to_rotvec(a) for a in episode_data["action"]]
    if all(c is not None for c in converted):
        episode_data["action_rotvec"] = np.stack(converted, axis=0)

    # Create the Zarr episode store.
    zarr_path = demo_dir / "replay.zarr"
    store = zarr.DirectoryStore(str(zarr_path))
    episode_store = ZarrEpisodeStore.create_empty(storage=store)

    # Append the current episode to disk.
    episode_store.append_episode(episode_data, compressors="disk")

    schema_path = write_recording_schema(
        demo_dir,
        episode_store.root,
        episode_data,
        environment_schema=environment_schema,
    )
    replay_manifest_path = write_replay_assets(
        demo_dir,
        env,
        task_name=exp_name,
        recording_parameters={
            "recorder": Path(__file__).name,
            "episode_index": int(success_index),
            "episode_tag": str(tag),
            "step_count": int(T),
            "action_shape_per_step": list(actions_arr.shape[1:]),
            "video_fps": int(video_fps),
            "data_fps": float(FLAGS.data_fps),
            "render_mode": FLAGS.render_mode,
            "randomize": bool(FLAGS.randomize),
            "save_depth": bool(FLAGS.save_depth),
            "save_camera_info": bool(FLAGS.save_camera_info),
            "fisheye_intrinsics_yaml": FLAGS.fisheye_intrinsics_yaml,
            "target_label": FLAGS.target_label,
            "arm_type": FLAGS.arm_type,
            "hand_type": FLAGS.hand_type,
            "sim_backend": FLAGS.sim_backend,
            "record_physx_ground_truth": bool(
                FLAGS.record_physx_ground_truth
                and FLAGS.sim_backend == "isaacsim"
            ),
            "camera_screen_effect": bool(FLAGS.camera_screen_effect),
            "ipad_screen_effect": bool(FLAGS.ipad_screen_effect),
            "manual_save_key": FLAGS.save_key,
        },
        environment_schema=environment_schema,
        repo_root=Path(__file__).resolve().parents[1],
    )

    print(f"[Saved replay.zarr] {zarr_path} (steps: {episode_data['action'].shape[0]})")
    print(f"[Saved recording schema] {schema_path}")
    print(f"[Saved replay manifest/model] {replay_manifest_path}")

    # --- write exactly the canonical saved video views ---
    videos_dir = demo_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    camera_sources = select_canonical_camera_sources_for_trajectory(trajectory, env=env)
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
            crf=21,
            thread_type="FRAME",
            thread_count=2,
        )

        out_path = str(videos_dir / f"{cam_key}.mp4")
        video_writer.start(out_path)

        for step in trajectory:
            obs = step.get("observations", {})
            img = observation_image(obs, source_key)
            if img is None:
                continue
            frame = _safe_squeeze_image(img)
            frame = postprocess_canonical_image(frame, cam_key, source_key)
            frame_filter = getattr(
                getattr(env, "unwrapped", env),
                "postprocess_recording_frame",
                None,
            )
            if callable(frame_filter):
                frame = frame_filter(
                    frame,
                    canonical_key=cam_key,
                    source_key=source_key,
                )
            # _safe_squeeze_image returns RGB uint8. Mp4VideoWriter expects RGB.
            if cam_key not in first_frames:
                first_frames[cam_key] = frame
            video_writer.write_frame(frame)

        video_writer.stop()
        print(f"[Saved video] {out_path} (source: {source_key})")

    if FLAGS.save_camera_info:
        camera_views = describe_camera_views(env, camera_keys) if env is not None else {}
        try:
            for path in write_video_camera_info(
                videos_dir=videos_dir,
                camera_keys=camera_keys,
                first_frames=first_frames,
                intrinsics_yaml=FLAGS.fisheye_intrinsics_yaml,
                camera_views=camera_views,
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


def _has_displayable_wrist_images(obs) -> bool:
    if not isinstance(obs, dict):
        return False
    for key in ("wrist", "wrist_left", "wrist_right"):
        value = obs.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] in (1, 3):
            return True
    return False


def _install_manual_save_hotkey(env, save_key: str = "s"):
    """Patch the teleop wrapper's GLFW key callback to set ``manual_save_trigger``
    when ``save_key`` is pressed in the MuJoCo viewer window.

    Returns the teleop wrapper instance (with a ``manual_save_trigger`` attribute),
    or ``None`` if no GLFW-based teleop wrapper is found (e.g. headless / policy mode).
    """
    # Walk the gym wrapper chain to find the object exposing glfw_on_key.
    wrapper = env
    while wrapper is not None and not hasattr(wrapper, "glfw_on_key"):
        wrapper = getattr(wrapper, "env", None)
    if wrapper is None:
        print("[manual-save] No GLFW teleop wrapper found; `s` will only work in the "
              "OpenCV wrist-camera window.")
        return None

    try:
        import glfw
    except Exception:
        return wrapper

    wrapper.manual_save_trigger = False
    try:
        key_code = getattr(glfw, f"KEY_{save_key.upper()}")
    except AttributeError:
        key_code = glfw.KEY_S

    original_on_key = wrapper.glfw_on_key

    def patched_on_key(window, key, scancode, action, mods,
                       _orig=original_on_key, _w=wrapper, _kc=key_code):
        _orig(window, key, scancode, action, mods)  # keep `;` and `r` behavior
        if action == glfw.PRESS and key == _kc:
            _w.manual_save_trigger = True

    wrapper.glfw_on_key = patched_on_key
    # Force the wrapper to (re)bind the callback to the current window.
    if hasattr(wrapper, "_key_callback_window"):
        wrapper._key_callback_window = None
    if hasattr(wrapper, "_ensure_key_callback"):
        try:
            wrapper._ensure_key_callback()
        except Exception:
            pass
    return wrapper


def main(_argv):
    task_id = FLAGS.exp_name
    if FLAGS.record_physx_ground_truth and FLAGS.sim_backend == "isaacsim":
        os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
    else:
        os.environ.pop("HANDSON_RECORD_PHYSX_GROUND_TRUTH", None)

    config = CONFIG_MAPPING[task_id]()

    env_extra_kwargs = {}
    if task_id == "bimanual_photograph":
        env_extra_kwargs["camera_screen_effect"] = FLAGS.camera_screen_effect
    if task_id == "bimanual_unlock_ipad":
        env_extra_kwargs["ipad_screen_effect"] = FLAGS.ipad_screen_effect
    os.environ["HANDSON_ARM_TYPE"] = FLAGS.arm_type
    env_extra_kwargs["hand_type"] = FLAGS.hand_type
    if FLAGS.sim_backend != "mujoco":
        if task_id != "bimanual_assembly":
            raise ValueError(
                "--sim_backend=isaacsim is currently supported only for "
                "exp_name=bimanual_assembly."
            )
        env_extra_kwargs["sim_backend"] = FLAGS.sim_backend

    env = config.get_native_environment(
        render_mode=FLAGS.render_mode,
        randomize=FLAGS.randomize,
        **env_extra_kwargs,
    )

    reset_kwargs = {}
    if FLAGS.target_label:
        reset_kwargs["target_label"] = FLAGS.target_label

    # Install the `s` (manual force-save) hotkey on the teleop wrapper's GLFW
    # callback, so pressing `s` in the MuJoCo viewer window force-saves the
    # current trajectory (in addition to the OpenCV wrist window, handled below).
    teleop_wrapper = _install_manual_save_hotkey(env, FLAGS.save_key)

    environment_recording_schema = None
    required_recording_fields: set[str] = set()

    def _refresh_episode_recording_contract() -> None:
        nonlocal environment_recording_schema, required_recording_fields
        environment_recording_schema = resolve_environment_recording_schema(env)
        required_recording_fields = declared_recording_field_names(
            environment_recording_schema
        )

    obs, info = env.reset(**reset_kwargs)
    _refresh_episode_recording_contract()
    print(f"Environment reset complete for {task_id}.")
    if FLAGS.target_label:
        print(f"Target label fixed to {FLAGS.target_label}.")
    print(
        f"[manual-save build] Press '{FLAGS.save_key}' (MuJoCo window or wrist-camera window) "
        f"to force-save the current trajectory to {FLAGS.out_dir}; "
        f"';' toggle teleop, 'r' discard & reset."
    )

    base_out = _ensure_base_outdir(FLAGS.out_dir)
    saved_demo_dirs = []
    success_count = 0
    manual_count = 0
    success_needed = FLAGS.successes_needed

    pbar = tqdm(total=success_needed)

    trajectory = []  # temporary storage for current episode transitions
    returns = 0.0
    total_steps = 0

    def _force_save_current(traj):
        """Force-save the current trajectory regardless of task success."""
        nonlocal manual_count
        if not traj:
            print("[Manual Save] current trajectory is empty; nothing to save.")
            return False
        manual_count += 1
        demo_dir = _write_demo_zarr_and_videos(
            traj,
            task_id,
            manual_count,
            base_out,
            FLAGS.video_fps,
            tag="manual",
            env=env,
            environment_schema=environment_recording_schema,
            required_recording_fields=required_recording_fields,
        )
        saved_demo_dirs.append(demo_dir)
        print(f"[Manual Saved] #{manual_count} ({len(traj)} steps) -> {demo_dir}")
        return True

    # Wrist-camera viewer initialization (optional)
    wrist_cam_viewer = None
    if FLAGS.show_sim_cameras and _has_displayable_wrist_images(obs):
        wrist_cam_viewer = WristCameraViewer()

    # Main collection loop
    while success_count < success_needed:
        actions = np.zeros(env.action_space.sample().shape)

        # Capture depth now (env state currently matches `obs`); it'll be
        # stored alongside `obs` below for parity with the RGB cameras.
        depth_now = None
        if FLAGS.save_depth and isinstance(obs, dict):
            camera_sources = select_canonical_camera_sources(obs, env=env)
            depth_now = collect_depth_frames(env, list(camera_sources.keys()))

        if (
            FLAGS.show_sim_cameras
            and wrist_cam_viewer is None
            and _has_displayable_wrist_images(obs)
        ):
            wrist_cam_viewer = WristCameraViewer()

        # `s` pressed inside the MuJoCo viewer window (via GLFW callback) ->
        # force-save the current trajectory, then reset to start a fresh one.
        if teleop_wrapper is not None and getattr(teleop_wrapper, "manual_save_trigger", False):
            teleop_wrapper.manual_save_trigger = False
            _force_save_current(trajectory)
            trajectory = []
            returns = 0.0
            obs, info = env.reset(**reset_kwargs)
            _refresh_episode_recording_contract()
            continue

        if wrist_cam_viewer is not None:
            key = wrist_cam_viewer.update_and_show(obs)
            if key == ord("r"):
                print("[Keyboard] Reset triggered")
                trajectory = []
                returns = 0.0
                obs, info = env.reset(**reset_kwargs)
                _refresh_episode_recording_contract()
                continue
            if key == ord(FLAGS.save_key):
                # `s` pressed inside the OpenCV wrist-camera window.
                print("[Keyboard] Manual save triggered")
                _force_save_current(trajectory)
                trajectory = []
                returns = 0.0
                obs, info = env.reset(**reset_kwargs)
                _refresh_episode_recording_contract()
                continue

        next_obs, rew, done, truncated, info = env.step(actions)
        total_steps += 1
        returns += rew

        if "intervene_action" in info:
            actions = info["intervene_action"]

        actions = _flatten_action_for_storage(actions)
        transition = copy.deepcopy(
            dict(
                observations=select_current_observation_for_recording(
                    obs,
                    required_fields=required_recording_fields,
                    include_required_fields=not trajectory,
                ),
                next_observations=select_next_observation_for_recording(
                    next_obs,
                    required_fields=required_recording_fields,
                ),
                actions=actions,
                dones=done,
                infos=info,
            )
        )
        if depth_now is not None:
            transition["depth"] = depth_now
        trajectory.append(transition)

        pbar.set_description(f"Return: {returns}")
        obs = next_obs

        if info.get("manual_reset", False):
            print("[Teleop] Manual reset triggered; dropping current trajectory.")
            trajectory = []
            returns = 0.0
            obs, info = env.reset(**reset_kwargs)
            _refresh_episode_recording_contract()
            continue

        if done:
            if info.get("succeed", False):
                success_count += 1
                demo_dir = _write_demo_zarr_and_videos(
                    trajectory,
                    task_id,
                    success_count,
                    base_out,
                    FLAGS.video_fps,
                    env=env,
                    environment_schema=environment_recording_schema,
                    required_recording_fields=required_recording_fields,
                )
                saved_demo_dirs.append(demo_dir)
                print(f"[Saved] success #{success_count} -> {demo_dir}")
                pbar.update(1)

                # release trajectory
                trajectory = []
                returns = 0.0
            else:
                trajectory = []
                returns = 0.0

            obs, info = env.reset(**reset_kwargs)
            _refresh_episode_recording_contract()

        if FLAGS.max_steps > 0 and total_steps >= FLAGS.max_steps:
            print(f"Reached max_steps={FLAGS.max_steps}; stopping run.")
            break

    print(
        f"Collected {success_count} success demos + {manual_count} manual-saved demos. "
        f"Individual demo directories:"
    )
    for p in saved_demo_dirs:
        print("  ", p)

    # Attempt to close viewer and env
    if wrist_cam_viewer is not None:
        try:
            wrist_cam_viewer.close()
        except Exception:
            pass

    try:
        env.close()
    except Exception:
        pass

    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(main)
