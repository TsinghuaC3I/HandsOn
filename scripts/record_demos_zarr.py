#!/usr/bin/env python3
"""
Record successful simulated teleoperation demos as Zarr episodes plus MP4 videos.

For each successful episode, the script writes:
- a directory ending in `__physx_v3` when the fresh-scene PhysX contract passed
- `DATASET_PHYSX_V3.json` for an explicit current-generation boundary
- `replay.zarr/` with low-dimensional observations and actions
- exactly three canonical videos: `main.mp4`, `wrist_left.mp4`, `wrist_right.mp4`
- `videos/camera_info.json` and `videos/<camera_key>_camera_info.json` sidecars
  with the MuJoCo view pose plus scaled fisheye intrinsics for wrist cameras
"""

import copy
import datetime
import errno
import fcntl
import hashlib
import json
import os
import secrets
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "handson"
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

# This must precede project imports that transitively freeze camera constants.
from handson.runtime.task_camera_profile import configure_collection_task_camera_profile

configure_collection_task_camera_profile(sys.argv[1:], os.environ)

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
from handson.data.b03_newton_validity import (
    b03_newton_source_validity_error,
    evaluate_b03_newton_episode_arrays,
    evaluate_b03_newton_substep_observation,
)
from handson.data.b320_bc021_final_asset import (
    bc021_collection_meta_binding,
)
from handson.data.canonical_videos import (
    CANONICAL_CAMERA_KEYS,
    ONLINE_CAMERA_POSTRENDER_SCOPES,
    ONLINE_CAMERA_SCOPE_ALL_LIVE,
    ONLINE_CAMERA_SCOPE_ALL_POSTRENDER,
    ONLINE_CAMERA_SCOPE_MAIN_LIVE_POSTRENDER,
    observation_image,
    postprocess_canonical_depth,
    postprocess_canonical_image,
    select_canonical_camera_sources,
)
from handson.data.depth_capture import collect_depth_frames
from handson.data.episode_store import ZarrEpisodeStore
from handson.data.manus_raw_recording import (
    MANUS_RAW_DEFAULT_PORT,
    MANUS_RAW_FORMAT,
    MANUS_RAW_SCHEMA_FILE,
    MANUS_RAW_VERSION,
    ManusRawUdpRecorder,
    write_manus_raw_schema,
    write_manus_raw_zarr,
)
from handson.data.newton_cross_gpu_certification import (
    CERTIFICATION_MODE as NEWTON_CROSS_GPU_CERTIFICATION_MODE,
    DATASET_DIRECTORY_SUFFIX as NEWTON_CROSS_GPU_DATASET_SUFFIX,
    DATASET_GENERATION as NEWTON_CROSS_GPU_DATASET_GENERATION,
    DATASET_MARKER_NAME as NEWTON_CROSS_GPU_DATASET_MARKER,
    DATASET_TRAINING_STATUS as NEWTON_CROSS_GPU_TRAINING_STATUS,
    REQUIRED_RUNS as NEWTON_CROSS_GPU_REQUIRED_RUNS,
)
from handson.data.physical_replay_certification import (
    CERTIFICATION_MODE as PHYSICAL_REPLAY_CERTIFICATION_MODE,
    DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS,
    DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_FORMAT,
    DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_VERSION,
    DETACHED_HAND_ACTION_INPUT_MODE,
    DETACHED_HAND_ACTION_TRANSITION_CONTRACT,
    FRESH_PROCESS_ACCEPTANCE_NAME,
    FRESH_PROCESS_CERTIFICATION_MODE,
    LEGACY_RAW_ACTION_INPUT_MODE,
    LEGACY_RAW_ACTION_TRANSITION_CONTRACT,
    PERSISTENT_RECORDER_LIFECYCLE,
    PHYSICAL_REPLAY_INVALID_OUTCOME_CODE,
    PHYSICAL_REPLAY_OUTCOME_VERSION,
    PHYSICAL_REPLAY_REVIEW_INVALID,
    PHYSICAL_REPLAY_REVIEW_REQUIRED,
    PHYSICAL_REPLAY_REVIEW_REQUIRED_OUTCOME_CODE,
    PHYSICAL_REPLAY_REVIEW_VALID,
    PHYSICAL_REPLAY_VALID_OUTCOME_CODE,
    RECORDED_DRIVE_TARGET_INPUT_MODE,
    RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT,
    REPORT_NAME as PHYSICAL_REPLAY_REPORT_NAME,
    certification_upload_blocker as physical_replay_certification_upload_blocker,
)
from handson.data.physx_session_history import (
    PhysxSessionHistoryJournal,
    load_session_history,
)
from handson.data.recording_fields import (
    RECORDER_MONOTONIC_TIME_KEY,
    declared_recording_field_names,
    effective_sampling_hz,
    resolve_environment_recording_schema,
    select_current_observation_for_recording,
    select_next_observation_for_recording,
    stack_transition_numeric_observation_fields,
    transition_monotonic_timestamps,
    transition_timestamps,
    wall_sampling_contract,
    write_recording_schema,
    write_replay_assets,
)
from handson.data.video_writer import Mp4VideoWriter
from handson.data.simplified_post_ik import (
    SIMPLIFIED_POST_IK_PROFILE,
    SIMPLIFIED_POST_IK_REPLAY_SCOPE,
    SIMPLIFIED_POST_IK_TCP_SOURCE,
)
from handson.data.task_guidance import (
    compile_task_guidance_profile,
    default_guidance_root,
    mark_guidance_planner_result,
    register_guidance_route_demo,
    register_successful_guidance_demo,
    task_guidance_dir,
)
from handson.runtime.collection_support import (
    BC011_COLLECTION_ANGLE_GUIDANCE_ENV,
    ISAACSIM_COLLECTION_TASKS,
    ISAACSIM_COLLECTION_UNVERIFIED_TASKS,
    ISAACSIM_NONSTANDARD_DEFERRED_RENDER_TASKS,
    supports_isaacsim_deferred_camera_render,
)
from handson.runtime.gpu import (
    configure_gpu_environment,
    configure_isaac_gpu_environment,
)
from handson.sim.isaaclab.recording_origin import (
    RecordingOriginStabilityError,
    LW_TASK_NAMES,
    STATIC_POSE_FALLBACK_TASKS,
    STABLE_ORIGIN_ENV,
    VELOCITY_CLAMP_SCOPE_NONE,
    is_abc_style_collection_task,
    stable_recording_origin_kwargs_for_task,
    stable_recording_origin_reset_attempts_for_task,
    stabilize_recording_origin,
)
from handson.sim.isaaclab.b320_bimanual_scene import (
    validate_b320_visual_dependencies,
)
from handson.sim.isaaclab.newton_determinism import (
    NEWTON_CROSS_GPU_PROFILE,
    NEWTON_LOCAL_DEBUG_PROFILE,
    NEWTON_REPLAY_PROFILE_ENV,
)
from handson.sim.isaaclab.runtime_display import (
    configure_isaac_offline_render_environment,
)
from handson.tasks.mappings import CONFIG_MAPPING
from scipy.spatial.transform import Rotation
from tqdm import tqdm

FLAGS = flags.FLAGS
# Native task previews must prove the same physical initial state that a
# recording run would accept.  Keep this list local to the recorder so a
# production-backed native route cannot silently take a permissive preview
# shortcut.
NATIVE_STABLE_PREVIEW_TASKS = frozenset(
    {
        "z01_pipette",
        "z02_bottle_cap",
        "t40_slide_knife_blade",
        "t22_pass_screwdriver",
        "t17_open_latched_door",
        "bench320_hl015_bimanual_cable_route_dual_plug",
        *LW_TASK_NAMES,
    }
)
GENESIS_COLLECTION_TASKS = frozenset({"e01_fold_garment"})
NEWTON_COLLECTION_TASKS = frozenset(
    {"t26_place_fruit_in_bucket", "t09_insert_peg"}
)
ISAAC_RUNTIME_BACKENDS = frozenset({"isaacsim", "isaacsim_newton"})
# Preserve the three established recorder choices as one compatibility group;
# Newton is an additive fourth backend with a separate dataset identity.
LEGACY_RECORDER_SIM_BACKENDS = ["mujoco", "isaacsim", "genesis"]
ENGINE_LABELS = {
    "E01": "旧引擎（MuJoCo）",
    "E02": "新引擎（Isaac Sim / IsaacLab）",
    "E03": "Genesis（旧 PBD 诊断 / IPC 离线）",
    "E04": "NVIDIA Newton（Isaac Sim / MuJoCo-Warp）",
}
STABLE_V2_DATASET_GENERATION = "stable_origin_v2"
PHYSX_V3_DATASET_GENERATION = "physx_fresh_scene_v3"
NEWTON_V1_DATASET_GENERATION = "nvidia_newton_v1"
NEWTON_CROSS_GPU_V2_DATASET_GENERATION = NEWTON_CROSS_GPU_DATASET_GENERATION
LEGACY_DATASET_GENERATION = "legacy_pre_stable_v2"
STABLE_V2_DIRECTORY_SUFFIX = "__stable_v2"
PHYSX_V3_DIRECTORY_SUFFIX = "__physx_v3"
NEWTON_V1_DIRECTORY_SUFFIX = "__newton_v1"
NEWTON_CROSS_GPU_V2_DIRECTORY_SUFFIX = NEWTON_CROSS_GPU_DATASET_SUFFIX
DATASET_GENERATION_DIRECTORY_SUFFIXES = (
    STABLE_V2_DIRECTORY_SUFFIX,
    PHYSX_V3_DIRECTORY_SUFFIX,
    NEWTON_V1_DIRECTORY_SUFFIX,
    NEWTON_CROSS_GPU_V2_DIRECTORY_SUFFIX,
)
STABLE_V2_MARKER_FILE = "DATASET_STABLE_V2.json"
PHYSX_V3_MARKER_FILE = "DATASET_PHYSX_V3.json"
NEWTON_V1_MARKER_FILE = "DATASET_NEWTON_V1.json"
NEWTON_CROSS_GPU_V2_MARKER_FILE = NEWTON_CROSS_GPU_DATASET_MARKER
STABLE_V2_ORIGIN_STRATEGY = "continuous_current_target_all_velocity_settle"
DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_ENV = (
    "HANDSON_DEFER_PHYSICAL_REPLAY_CERTIFICATION"
)
DEFERRED_PHYSICAL_REPLAY_AUTOSTART_ENV = (
    "HANDSON_DEFER_PHYSICAL_REPLAY_AUTOSTART"
)
ISAAC_IN_PROCESS_HARD_RESET_ENV = "HANDSON_ISAAC_IN_PROCESS_HARD_RESET"
POSTPROCESS_MANUAL_ONLY_ENV = "HANDSON_POSTPROCESS_MANUAL_ONLY"
B320_TERMINAL_RESET_HOLD_SECONDS_ENV = (
    "HANDSON_B320_TERMINAL_RESET_HOLD_SECONDS"
)
B320_TERMINAL_RESET_HOLD_SECONDS_DEFAULT = 1.5
B320_TERMINAL_RESET_HOLD_SECONDS_MAX = 5.0
VERIFIED_PUBLICATION_CONTRACT = "handson.verified_episode.v3"
MANUAL_POSTPROCESS_POLICY = "manual_review_render_upload_v1"
DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE = "deferred_post_collection"
ORDERED_SESSION_PHYSICAL_REPLAY_CERTIFICATION_SCOPE = (
    "ordered_collection_session"
)
DEFERRED_SOURCE_RECORDER_PID_ENV = "HANDSON_DEFERRED_SOURCE_RECORDER_PID"
DEFERRED_PHYSICAL_REPLAY_WORKER_SCRIPT_ENV = (
    "HANDSON_DEFERRED_PHYSICAL_REPLAY_WORKER_SCRIPT"
)
FRESH_RECORDER_LIFECYCLE = "fresh_process_per_attempt"
EXPLICIT_COLLECTION_SEED_SOURCE = "explicit_cli_seed"
RANDOM_COLLECTION_SEED_SOURCE = "os_cryptographic_random_31bit"
EPISODE_SEED_DERIVATION = (
    "collection_run_seed_plus_reset_attempt_index_mod_2pow31"
)
PHYSICAL_REPLAY_PENDING_DIRECTORY = ".physical_replay_pending"
PHYSICAL_REPLAY_SESSION_HISTORY_DIRECTORY = ".physx_session_history"
DEFERRED_PHYSICAL_REPLAY_WORKER_STATE_DIRECTORY = (
    ".physical_replay_certification_state"
)
DEFERRED_PHYSICAL_REPLAY_LAUNCH_LOCK = (
    ".physical_replay_certification_launch.lock"
)
MAX_DEFERRED_PHYSICAL_REPLAY_LAUNCH_ATTEMPTS = 2
PHYSICAL_REPLAY_REVIEW_HISTORY_DIRECTORY = ".physical_replay_review_history"
PHYSICAL_REPLAY_REVIEW_HISTORY_FORMAT = (
    "handson.physical_replay_review_attempt_history"
)
PHYSICAL_REPLAY_RUN_DIRECTORY = "physical_replay_certification_runs"


def _environment_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _optional_flag_value(name: str, default=None):
    """Read an optional absl flag before parse and from lightweight test fakes."""

    try:
        flag = FLAGS[name]
    except (AttributeError, KeyError, TypeError):
        try:
            return getattr(FLAGS, name)
        except Exception:
            return default
    return getattr(flag, "value", flag)


def _flag_was_explicitly_supplied(name: str, argv: list[str] | None = None) -> bool:
    """Return whether an absl boolean flag was present on the command line.

    ``FLAGS.<name>`` only exposes the parsed value, so a task-specific default
    cannot distinguish an omitted ``--randomize`` from an explicit
    ``--randomize=False`` after parsing.  Inspect the original argv while
    accepting all absl spellings (``--flag``, ``--flag=true/false``,
    ``--noflag`` and the uncommon separated value form).
    """

    arguments = list(sys.argv[1:] if argv is None else argv)
    positive = f"--{name}"
    negative = f"--no{name}"
    for argument in arguments:
        token = str(argument).strip()
        if token == negative or token.startswith(f"{negative}="):
            return True
        if token == positive or token.startswith(f"{positive}="):
            return True
    return False


def _apply_task_randomization_default(task_id: str, argv: list[str] | None = None) -> None:
    """Default these YCB collection tasks to seeded position randomization.

    An explicit opt-out remains available for deterministic diagnostics/replay.
    """
    if task_id not in {
        "t34_roll_ball_through_gate", "t35_rotate_propeller",
        "t41_sort_nested_cups", "t47_trigger_drill_on_block",
        "t29_place_wine_glass", "t33_reposition_marker_grasp",
        "eval_flex01_compress_sponge_into_mug",
    }:
        return
    if _flag_was_explicitly_supplied("randomize", argv):
        return
    FLAGS.randomize = True


def _postprocess_manual_only_enabled() -> bool:
    """Return whether certification/render/upload must await an operator."""

    return _environment_flag_enabled(POSTPROCESS_MANUAL_ONLY_ENV)


def _isaac_in_process_hard_reset_enabled() -> bool:
    return _environment_flag_enabled(ISAAC_IN_PROCESS_HARD_RESET_ENV)


# Keep a task-scoped fail-closed switch for any future teardown regression.
# FU006 was removed after its task tensor/contact views were explicitly
# released before detach and the rebuilt scene passed five consecutive live
# GPU reset trials.
_ISAAC_IN_PROCESS_HARD_RESET_UNSAFE_TASKS = frozenset()


def _isaac_in_process_hard_reset_enabled_for_task(task_name: object) -> bool:
    """Return whether this task may use the audited in-process Scene reset."""

    normalized = str(task_name).strip().lower()
    return bool(
        _isaac_in_process_hard_reset_enabled()
        and normalized not in _ISAAC_IN_PROCESS_HARD_RESET_UNSAFE_TASKS
    )


def _manual_postprocess_publication_declaration(
    *,
    enabled: bool | None = None,
) -> dict[str, object]:
    """Return the immutable capture-time declaration for verified-v3 data."""

    manual_only = (
        _postprocess_manual_only_enabled()
        if enabled is None
        else bool(enabled)
    )
    if not manual_only:
        return {}
    return {
        "publication_contract": VERIFIED_PUBLICATION_CONTRACT,
        "postprocess_policy": MANUAL_POSTPROCESS_POLICY,
        "postprocess_manual_only": True,
    }


def _deferred_physical_replay_certification_enabled() -> bool:
    # Manual-only postprocessing is a stricter form of deferred publication.
    # Treat it as sufficient authorization for formal Isaac recording so a
    # launcher only needs to set the one policy flag.  The explicit launch
    # guards below ensure that it queues sealed/bound pending data without
    # starting any certification worker.
    return bool(
        _environment_flag_enabled(
            DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_ENV
        )
        or _postprocess_manual_only_enabled()
    )


def _physical_replay_session_history_required(
    *,
    preview_only: bool,
    sim_backend: str,
    record_physx_ground_truth: bool,
    deferred_certification: bool,
    fresh_certified_worker: bool,
) -> bool:
    """Require a replay journal for every formal certified Isaac worker."""

    return bool(
        not preview_only
        and sim_backend == "isaacsim"
        and record_physx_ground_truth
        and (deferred_certification or fresh_certified_worker)
    )


def _deferred_physical_replay_autostart_enabled() -> bool:
    """Return whether this recorder may launch post-collection workers.

    Scheduled collection queues certification by default.  The collection UI
    owns automatic launches inside its 02:00-09:00 window through the recovery
    CLI; standalone callers must opt in explicitly if they need the legacy
    immediate-start behavior.
    """

    return bool(
        _deferred_physical_replay_certification_enabled()
        and not _postprocess_manual_only_enabled()
        and _environment_flag_enabled(
            DEFERRED_PHYSICAL_REPLAY_AUTOSTART_ENV
        )
    )


@dataclass(frozen=True)
class DatasetGeneration:
    name: str
    training_status: str
    directory_suffix: str
    marker_file: str
    stable_origin: dict[str, object] | None

    @property
    def is_stable_v2(self) -> bool:
        return self.name in {
            STABLE_V2_DATASET_GENERATION,
            PHYSX_V3_DATASET_GENERATION,
        }

    @property
    def is_physx_v3(self) -> bool:
        return self.name == PHYSX_V3_DATASET_GENERATION

    @property
    def is_newton_v1(self) -> bool:
        return self.name == NEWTON_V1_DATASET_GENERATION

    @property
    def is_newton_cross_gpu_v2(self) -> bool:
        return self.name == NEWTON_CROSS_GPU_V2_DATASET_GENERATION


flags.DEFINE_string(
    "exp_name",
    "water_plant",
    "Task name, such as water_plant.",
)
flags.DEFINE_integer("successes_needed", 2, "Number of successful demos to collect.")
flags.DEFINE_bool("show_sim_cameras", True, "Show simulation cameras")
flags.DEFINE_integer(
    "max_steps",
    0,
    "Stop after this many environment steps; 0 means run until successes_needed",
)
flags.DEFINE_string("out_dir", "./", "Output base directory for zarr and videos")
flags.DEFINE_bool(
    "guidance_mode",
    False,
    "Record exactly one Space-delimited hands-only route for the task-local "
    "IK guidance library; task completion is not required.",
)
flags.DEFINE_string(
    "guidance_root",
    "",
    "Machine-local task-guidance root. Empty uses the repository-adjacent "
    "handson-local-state directory and never writes guidance into Git.",
)
flags.DEFINE_string(
    "policy_actions_npy",
    "",
    "Task-development validation only: replay a finite T x A array through "
    "the public policy action wrapper instead of reading teleoperation devices.",
)
flags.DEFINE_integer("video_fps", 30, "FPS for saved MP4 videos")
flags.DEFINE_enum("a03_controller_mode", "original", ["original", "real_aligned"],
    "A03 manual execution: native original or current real inference command processing.")
flags.DEFINE_integer("a03_playback_slowdown", 1,
    "Aligned A03 source timing: 3 matches the current real launcher; 1 requires matching inference settings.")
flags.DEFINE_integer("a03_policy_chunk_size", 30,
    "Model chunk length used for the real retimer's measured-reference boundary.")
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
    "Enable seeded scene randomization (for example object poses and visuals) "
    "when the selected task supports it.",
)
flags.DEFINE_bool(
    "randomize_dynamics",
    False,
    "Randomize physical parameters such as mass or friction. Certified PhysX "
    "recordings require this to remain False.",
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
    os.environ.get(
        "HANDSON_FISHEYE_INTRINSICS_YAML",
        str(DEFAULT_FISHEYE_INTRINSICS_PATH),
    ),
    "OpenCV YAML intrinsics for the wrist fisheye camera.",
)
flags.DEFINE_bool(
    "preview_only",
    False,
    "Open the task environment for visual preview only; do not write demos.",
)
flags.DEFINE_bool(
    "scripted_policy",
    False,
    "Run the task-owned deterministic policy instead of live teleoperation. "
    "Currently supported by FU015, P0-02, and P0-03.",
)
flags.DEFINE_bool(
    "scripted_as013_success",
    False,
    "Drive the B320-AS013 physical success policy through the ordinary 54-value "
    "dual-arm action path. Valid only for the AS013 IsaacSim task.",
)
flags.DEFINE_bool(
    "camera_screen_effect",
    False,
    "(bimanual_photograph only; C06 is always enabled) Enable the camera "
    "live-screen effect "
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
    ["panda", "rokae", SIMPLIFIED_POST_IK_PROFILE],
    "Arm XML/control profile. rokae_simplified_post_ik collects detached "
    "Wuji hands and reconstructs smooth ROKAE arms offline.",
)
flags.DEFINE_enum(
    "hand_type",
    "allegro",
    ["allegro", "wuji1", "wuji2"],
    "Hand XML family to load.",
)
flags.DEFINE_enum(
    "sim_backend",
    "mujoco",
    [*LEGACY_RECORDER_SIM_BACKENDS, "isaacsim_newton"],
    "Simulation backend to use for task construction.",
)
flags.DEFINE_bool(
    "record_physx_ground_truth",
    True,
    "For Isaac Sim, record action-aligned robot/task PhysX readback in physx_* fields.",
)
flags.DEFINE_integer(
    "seed",
    -1,
    "Collection-process seed. A negative value generates an explicit random "
    "seed that is still recorded; each episode receives a derived reset seed.",
)
flags.DEFINE_bool(
    "certify_physical_replay",
    True,
    "After Isaac collection closes, replay each saved action sequence in a "
    "fresh process and require its recorded PhysX state/success contract.",
)
flags.DEFINE_integer(
    "physical_replay_certification_timeout_seconds",
    1800,
    "Maximum wall time for one fresh-process PhysX replay certification.",
)
flags.DEFINE_bool(
    "auto_c01_arm_video",
    True,
    "For t08_hammer_nail and plan1_hammer_nail, globally plan and render "
    "videos_arm/ with ROKAE arms visible after each saved demo.",
)
flags.DEFINE_bool(
    "auto_plan_arm_video",
    True,
    "For single-arm Plan2+ tasks, render videos_arm/ from the recorded "
    "per-frame Plan arm completion after each saved demo.",
)
flags.DEFINE_float(
    "c01_arm_max_fk_position_error",
    0.02,
    "Maximum post-smoothing palm FK position error in meters before the "
    "C01/Plan1 automatic arm pipeline fails and skips rendering.",
)
flags.DEFINE_float(
    "c01_arm_max_fk_orientation_error",
    0.20,
    "Maximum post-smoothing palm FK orientation error in radians before the "
    "C01/Plan1 automatic arm pipeline fails and skips rendering.",
)
flags.DEFINE_float(
    "c01_arm_collision_penetration_tolerance",
    0.002,
    "Allowed arm/table, inter-arm, or self-collision penetration in meters during the "
    "C01/Plan1 automatic arm audit.",
)
flags.DEFINE_float(
    "c01_arm_collision_sweep_joint_step",
    0.02,
    "Maximum joint interpolation step in radians for between-frame collision "
    "checks in the C01/Plan1 automatic arm audit.",
)

_C01_ARM_COMPLETION_TASKS = frozenset(
    {
        "t08_hammer_nail",
        "plan1_hammer_nail",
    }
)
_E01_DEFERRED_ARM_COMPLETION_TASKS = frozenset({"e01_fold_garment"})
_E01_IPC_REPLAY_TASKS = frozenset(
    {
        "e01_fold_garment",
        "f01_cotton_cloth_mvp",
    }
)
_E01_DEFERRED_POSTPROCESS_BACKENDS = frozenset({"genesis", "isaacsim"})
_F01_COTTON_CLOTH_MVP_TASK = "f01_cotton_cloth_mvp"
_ISAAC_DEFERRED_ARM_WORKER_ENV = "HANDSON_ISAAC_DEFERRED_ARM_WORKER"
_ISAAC_DEFERRED_ARM_STATUS_FILE = "deferred_arm_replay_status.json"
_FRESH_CERTIFIED_COLLECTION_WORKER_ENV = (
    "HANDSON_FRESH_CERTIFIED_COLLECTION_WORKER"
)
# ``EX_TEMPFAIL`` tells the supervisor that this whole episode attempt was
# rejected and must be retried in a new process.  A fresh worker may never
# reset and continue in the same PhysX scene after an unstable origin, an
# operator reset, or an unsuccessful terminal transition.
_FRESH_WORKER_RETRYABLE_ATTEMPT_EXIT_CODE = getattr(os, "EX_TEMPFAIL", 75)
_FRESH_WORKER_RETRYABLE_ORIGIN_EXIT_CODE = (
    _FRESH_WORKER_RETRYABLE_ATTEMPT_EXIT_CODE
)
_E01_IPC_POSTPROCESS_WORKER_ENV = "HANDSON_E01_IPC_POSTPROCESS_WORKER"
_E01_POSTPROCESS_STATUS_FILE = "e01_postprocess.json"
_F01_IPC_REPLAY_DEFAULTS = {
    "HANDSON_E01_IPC_REPLAY_SETTLE_STEPS": "80",
    "HANDSON_E01_IPC_REPLAY_PHYSICS_DT": str(1.0 / 120.0),
    "HANDSON_E01_IPC_REPLAY_NEWTON_MAX_ITERATIONS": "16",
    "HANDSON_E01_IPC_REPLAY_MAX_NODE_STEP": "0.090",
    "HANDSON_E01_IPC_REPLAY_MAX_TCP_POSITION_ERROR": "0.300",
    "HANDSON_E01_IPC_REPLAY_MAX_TCP_ORIENTATION_ERROR": "3.200",
}
_F01_IPC_ENV_DEFAULTS = {
    "HANDSON_GENESIS_E01_YOUNGS_MODULUS": "1.2e4",
    "HANDSON_GENESIS_E01_POISSON_RATIO": "0.30",
    "HANDSON_GENESIS_E01_DENSITY": "375",
    "HANDSON_GENESIS_E01_THICKNESS": "0.00020",
    "HANDSON_GENESIS_E01_BENDING_STIFFNESS": "12",
    "HANDSON_GENESIS_E01_CLOTH_FRICTION": "0.9",
    "HANDSON_GENESIS_IPC_NEWTON_TOLERANCE": "0.01",
    "HANDSON_GENESIS_IPC_TRANSLATION_TOLERANCE": "0.03",
    "HANDSON_GENESIS_IPC_D_HAT": "0.001",
}
_ISAAC_DEFERRED_COLLECTION_ONLY_ENV_KEYS = (
    BC011_COLLECTION_ANGLE_GUIDANCE_ENV,
    "HANDSON_ISAACLAB_DIRECT_HAND_CONTROL",
    "HANDSON_ISAACLAB_DEFER_CAMERA_RENDER",
    "HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY",
    "HANDSON_ISAACLAB_DIRECT_VIEWPORT_DIVISOR",
    "HANDSON_ISAACLAB_ARM_TARGET_ALPHA",
    "HANDSON_ISAACLAB_ARM_MAX_SITE_STEP",
    "HANDSON_ISAACLAB_ARM_TARGET_JUMP_DELTA",
    "HANDSON_ISAACLAB_ARM_TARGET_POS_DEADBAND",
    "HANDSON_ISAACLAB_ARM_TARGET_ROT_JUMP_DEG",
    "HANDSON_ISAACLAB_ARM_TARGET_ROT_DEADBAND_DEG",
    "HANDSON_ISAACLAB_ARM_TARGET_JUMP_LIMIT",
    "HANDSON_ISAACLAB_ARM_MAX_JOINT_STEP",
    "HANDSON_ISAACSIM_VIEWPORT_WIDTH",
    "HANDSON_ISAACSIM_VIEWPORT_HEIGHT",
    "HANDSON_ISAACSIM_DLSS_EXEC_MODE",
)
_ISAAC_STARTUP_READY_FILE_ENV = "HANDSON_ISAAC_STARTUP_READY_FILE"
_ISAAC_WORKER_STATUS_FILE_ENV = "HANDSON_ISAAC_WORKER_STATUS_FILE"
_COLLECTION_RUNTIME_STATUS_FILE_ENV = "HANDSON_COLLECTION_RUNTIME_STATUS_FILE"
_ISAAC_STARTUP_TIMEOUT_ENV = "HANDSON_ISAAC_STARTUP_TIMEOUT_SECONDS"
_ISAAC_STARTUP_TIMEOUT_SECONDS = 60.0
_ISAAC_COLLECTION_DRAIN_TIMEOUT_SECONDS = 15.0
_ISAAC_RUNTIME_HEARTBEAT_TIMEOUT_ENV = (
    "HANDSON_ISAAC_RUNTIME_HEARTBEAT_TIMEOUT_SECONDS"
)
_ISAAC_RUNTIME_HEARTBEAT_TIMEOUT_SECONDS = 60.0
_ISAAC_EPISODE_SAVE_TIMEOUT_SECONDS = 300.0
_ISAAC_STARTUP_MAX_ATTEMPTS_ENV = "HANDSON_ISAAC_STARTUP_MAX_ATTEMPTS"
_ISAAC_STARTUP_MAX_ATTEMPTS = 2
_ISAAC_FATAL_BACKEND_EXIT_CODE = 86


def _uses_isaac_deferred_arm_pipeline(
    exp_name: str,
    sim_backend: str,
    env_value: str,
) -> bool:
    """Return whether an A-D Isaac task records hands before rendering arms."""
    backend = str(sim_backend).strip().lower()
    task = str(exp_name).strip().lower()
    if backend == "isaacsim" and task == "t27_place_largest_fitting_ball":
        # YCB02's operator scene always contains the complete articulated
        # robot. Ignore stale inherited defer flags in direct CLI launches too.
        return False
    ordinary_ad_physx = bool(
        backend == "isaacsim"
        and len(task) >= 4
        and task[0] in {"a", "b", "c", "d"}
        and task[1:3].isdigit()
        and task[3] == "_"
    )
    supported = bool(
        ordinary_ad_physx
        or (backend == "isaacsim" and task.startswith("ycb"))
        or (
            backend == "isaacsim"
            and task in ISAACSIM_NONSTANDARD_DEFERRED_RENDER_TASKS
        )
        or (backend == "isaacsim_newton" and task == "t09_insert_peg")
    )
    return bool(
        supported
        and env_value.strip().lower() not in {"0", "false", "no", "off"}
    )


def _online_camera_capture_scope(
    exp_name: str,
    sim_backend: str,
    *,
    deferred_arm_env_value: str,
    deferred_camera_env_value: str = "0",
    live_main_env_value: str,
) -> str:
    """Describe online cameras and reject unsafe main-only combinations."""

    main_only = str(live_main_env_value).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    all_deferred = str(deferred_camera_env_value).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if main_only and all_deferred:
        raise ValueError(
            "Live-main-only capture cannot defer all recording cameras."
        )
    live_arm_b03_newton = bool(
        str(exp_name).strip().lower() == "t09_insert_peg"
        and str(sim_backend).strip().lower() == "isaacsim_newton"
        and not _uses_isaac_deferred_arm_pipeline(
            exp_name,
            sim_backend,
            deferred_arm_env_value,
        )
    )
    formal_physx_postrender = supports_isaacsim_deferred_camera_render(
        sim_backend,
        exp_name,
    )
    if all_deferred:
        if not (formal_physx_postrender or live_arm_b03_newton):
            raise ValueError(
                "All-camera post-render capture requires a registered formal "
                "Isaac PhysX task or live-arm B03 Newton."
            )
        return ONLINE_CAMERA_SCOPE_ALL_POSTRENDER
    if not main_only:
        return ONLINE_CAMERA_SCOPE_ALL_LIVE
    if not live_arm_b03_newton:
        raise ValueError(
            "Live-main-only capture requires live-arm B03 Newton so the "
            "canonical three-camera post-render can replace its online "
            "placeholders."
        )
    return ONLINE_CAMERA_SCOPE_MAIN_LIVE_POSTRENDER


def _runtime_online_camera_capture_scope(
    exp_name: str,
    sim_backend: str,
) -> str:
    """Resolve the camera contract from the active collection environment."""

    return _validated_online_camera_capture_scope(
        _online_camera_capture_scope(
            exp_name,
            sim_backend,
            deferred_arm_env_value=os.environ.get(
                "HANDSON_ISAACLAB_DEFER_ARM_RENDER",
                "0",
            ),
            deferred_camera_env_value=os.environ.get(
                "HANDSON_ISAACLAB_DEFER_CAMERA_RENDER",
                "0",
            ),
            live_main_env_value=os.environ.get(
                "HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY",
                "0",
            ),
        )
    )


_WRITABLE_ONLINE_CAMERA_CAPTURE_SCOPES = frozenset(
    {
        ONLINE_CAMERA_SCOPE_ALL_LIVE,
        ONLINE_CAMERA_SCOPE_ALL_POSTRENDER,
        ONLINE_CAMERA_SCOPE_MAIN_LIVE_POSTRENDER,
    }
)


def _validated_online_camera_capture_scope(scope: object) -> str:
    """Reject unknown and legacy camera contracts before writing an episode."""

    normalized = str(scope).strip()
    if normalized not in _WRITABLE_ONLINE_CAMERA_CAPTURE_SCOPES:
        raise ValueError(
            "Unsupported writable online camera capture scope "
            f"{normalized!r}; expected one of "
            f"{sorted(_WRITABLE_ONLINE_CAMERA_CAPTURE_SCOPES)}."
        )
    return normalized


def _online_camera_spool_contract(
    scope: object,
    video_spool: object,
    *,
    save_depth: bool,
) -> tuple[str, bool, bool]:
    """Validate the frozen camera scope against its episode spool mode."""

    normalized = _validated_online_camera_capture_scope(scope)
    postrender_required = normalized in ONLINE_CAMERA_POSTRENDER_SCOPES
    encoding_enabled = bool(getattr(video_spool, "encode_videos", True))
    if postrender_required == encoding_enabled:
        raise RuntimeError(
            "Online video spool mode does not match the canonical camera "
            f"contract: scope={normalized!r}, "
            f"encode_videos={encoding_enabled}."
        )
    if postrender_required and save_depth:
        raise RuntimeError(
            "Depth capture cannot be enabled when all canonical cameras are "
            "deferred to postrender."
        )
    return normalized, postrender_required, encoding_enabled


def _uses_b03_newton_direct_hand_pipeline(
    exp_name: str,
    sim_backend: str,
) -> bool:
    return bool(
        str(sim_backend).strip().lower() == "isaacsim_newton"
        and str(exp_name).strip().lower() == "t09_insert_peg"
        and os.environ.get(
            "HANDSON_ISAACLAB_DIRECT_HAND_CONTROL",
            "0",
        ).strip().lower()
        not in {"0", "false", "no", "off", ""}
    )


def _configure_current_isaac_process_environment() -> None:
    """Apply Isaac's GPU selection without a CUDA ordinal remapping mask."""

    configured = configure_isaac_gpu_environment(
        os.environ,
        render_mode="human",
    )
    if "CUDA_VISIBLE_DEVICES" not in configured:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.update(configured)


def _isaac_startup_timeout_seconds(env: Mapping[str, str]) -> float:
    raw = env.get(_ISAAC_STARTUP_TIMEOUT_ENV, "")
    try:
        timeout = float(raw) if str(raw).strip() else _ISAAC_STARTUP_TIMEOUT_SECONDS
    except (TypeError, ValueError):
        timeout = _ISAAC_STARTUP_TIMEOUT_SECONDS
    return max(1.0, timeout)


def _isaac_runtime_heartbeat_timeout_seconds(env: Mapping[str, str]) -> float:
    raw = env.get(_ISAAC_RUNTIME_HEARTBEAT_TIMEOUT_ENV, "")
    try:
        timeout = (
            float(raw)
            if str(raw).strip()
            else _ISAAC_RUNTIME_HEARTBEAT_TIMEOUT_SECONDS
        )
    except (TypeError, ValueError):
        timeout = _ISAAC_RUNTIME_HEARTBEAT_TIMEOUT_SECONDS
    return max(10.0, timeout)


def _isaac_startup_max_attempts(env: Mapping[str, str]) -> int:
    raw = env.get(_ISAAC_STARTUP_MAX_ATTEMPTS_ENV, "")
    try:
        attempts = int(raw) if str(raw).strip() else _ISAAC_STARTUP_MAX_ATTEMPTS
    except (TypeError, ValueError):
        attempts = _ISAAC_STARTUP_MAX_ATTEMPTS
    return max(1, min(attempts, 2))


def _isaac_worker_status(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _isaac_startup_progress_detail(status: Mapping[str, object]) -> str:
    phase = str(status.get("phase") or "")
    progress = status.get("stability_progress")
    if phase == "building_scene":
        heartbeat = int(status.get("startup_heartbeat") or 0)
        suffix = f"（进度心跳 {heartbeat}）" if heartbeat else ""
        return (
            "正在构建 Isaac 场景；首次转换资源时可能需要较长时间"
            f"{suffix}。"
        )
    if not isinstance(progress, Mapping):
        return "Isaac 场景已启动，正在等待并验证稳定初态。"
    physics_steps = int(progress.get("physics_steps") or 0)
    physics_budget = int(progress.get("physics_step_budget") or 0)
    stable_steps = int(progress.get("stable_steps_observed") or 0)
    stable_required = int(progress.get("stable_steps_required") or 0)
    return (
        "正在验证稳定初态：PhysX "
        f"{physics_steps}/{physics_budget} 帧，连续稳定 "
        f"{stable_steps}/{stable_required} 帧。"
    )


_COLLECTION_RUNTIME_STICKY_FIELDS: dict[str, object] = {}


def _write_collection_runtime_status(
    phase: str,
    detail: str,
    **fields: object,
) -> None:
    """Publish supervisor state for the website without parsing noisy logs."""

    raw_path = os.environ.get(_COLLECTION_RUNTIME_STATUS_FILE_ENV, "").strip()
    if not raw_path:
        return
    for key in ("task_runtime",):
        if key in fields:
            value = fields[key]
            if value is None:
                _COLLECTION_RUNTIME_STICKY_FIELDS.pop(key, None)
            else:
                _COLLECTION_RUNTIME_STICKY_FIELDS[key] = value
    _write_json_atomic(
        Path(raw_path),
        {
            "format": "handson.collection_runtime_status",
            "version": 1,
            "phase": str(phase),
            "detail": str(detail),
            "supervisor_pid": os.getpid(),
            "updated_at": _now_iso(),
            **_COLLECTION_RUNTIME_STICKY_FIELDS,
            **fields,
        },
    )


def _quarantine_failed_isaac_startup_journal(
    output_root: Path,
    *,
    session_id: str,
    reason: str,
) -> Path | None:
    """Move a zero-episode startup journal aside before a safe retry."""

    if not session_id.strip():
        return None
    source = _physical_replay_session_history_path(output_root, session_id)
    if not source.is_dir():
        return None
    quarantine_root = source.parent / ".failed_startup"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    destination = quarantine_root / (
        f"{source.name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        f"-{secrets.token_hex(4)}"
    )
    os.replace(source, destination)
    _write_json_atomic(
        destination / "STARTUP_FAILURE.json",
        {
            "format": "handson.isaac_startup_failure",
            "version": 1,
            "session_id": session_id,
            "reason": reason,
            "quarantined_at": _now_iso(),
        },
    )
    return destination


def _mark_isaac_runtime_ready(task_id: str) -> None:
    marker = os.environ.get(_ISAAC_STARTUP_READY_FILE_ENV, "").strip()
    if not marker:
        return
    _write_json_atomic(
        Path(marker),
        {
            "task_id": str(task_id),
            "pid": os.getpid(),
            "ready_at": _now_iso(),
        },
    )


def _write_isaac_worker_status(
    phase: str,
    *,
    task_id: str,
    **details: object,
) -> None:
    """Publish an atomic, machine-readable lifecycle/heartbeat record."""

    path = os.environ.get(_ISAAC_WORKER_STATUS_FILE_ENV, "").strip()
    if not path:
        return
    _write_json_atomic(
        Path(path),
        {
            "phase": str(phase),
            "task_id": str(task_id),
            "pid": os.getpid(),
            "updated_at": _now_iso(),
            **details,
        },
    )


def _touch_isaac_worker_heartbeat() -> None:
    path = os.environ.get(_ISAAC_WORKER_STATUS_FILE_ENV, "").strip()
    if not path:
        return
    try:
        Path(path).touch(exist_ok=True)
    except OSError:
        # A status path is diagnostic only; collection data must not fail just
        # because a supervisor is concurrently cleaning its temporary file.
        pass


def _start_isaac_startup_heartbeat(
    phase: str,
    *,
    task_id: str,
    interval_seconds: float = 2.0,
) -> tuple[threading.Event, threading.Thread] | None:
    """Keep a supervised worker visibly alive inside long Kit/C++ startup calls."""

    if not os.environ.get(_ISAAC_WORKER_STATUS_FILE_ENV, "").strip():
        return None
    stop = threading.Event()

    def _publish() -> None:
        heartbeat_index = 0
        while not stop.wait(max(0.05, float(interval_seconds))):
            heartbeat_index += 1
            try:
                _write_isaac_worker_status(
                    phase,
                    task_id=task_id,
                    startup_heartbeat=heartbeat_index,
                )
            except OSError:
                # The supervisor can remove its temporary status path while
                # stopping the process.  Heartbeats never own collection data.
                return

    thread = threading.Thread(
        target=_publish,
        name="handson-isaac-startup-heartbeat",
        daemon=True,
    )
    thread.start()
    return stop, thread


def _stop_isaac_startup_heartbeat(
    heartbeat: tuple[threading.Event, threading.Thread] | None,
) -> None:
    if heartbeat is None:
        return
    stop, thread = heartbeat
    stop.set()
    thread.join(timeout=1.0)


def _supervised_isaac_worker() -> bool:
    return bool(os.environ.get(_ISAAC_STARTUP_READY_FILE_ENV, "").strip())


def _likely_poisoned_isaac_backend_error(exc: BaseException) -> bool:
    markers = (
        "illegal memory access",
        "cudaerrorillegaladdress",
        "cuda_error_illegal_address",
        "cuda error",
        "error 700",
        "fetching gpu narrowphase failed",
        "synchronizing gpu narrowphase failed",
        "cuda/physx backend is poisoned",
        "failed to get dof velocities from backend",
        "failed to get dof positions from backend",
        "failed to get root velocities from backend",
        "failed to get root transforms from backend",
    )
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending and len(seen) < 64:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        message = f"{type(current).__name__}: {current}".lower()
        if any(marker in message for marker in markers):
            return True
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if isinstance(cause, BaseException):
            pending.append(cause)
        if isinstance(context, BaseException) and context is not cause:
            pending.append(context)
        nested = getattr(current, "exceptions", ())
        if isinstance(nested, tuple):
            pending.extend(
                item for item in nested if isinstance(item, BaseException)
            )
    return False


def _fail_fast_supervised_isaac_startup(
    exc: BaseException,
    *,
    task_id: str,
    phase: str,
) -> None:
    """Exit without Kit teardown only when CUDA/PhysX is known unusable.

    No episode can exist before the ready marker.  Normal Python unwinding is
    actively unsafe after CUDA error 700 because Kit/PhysX destructors can
    wait forever on the poisoned context.  Ordinary task/configuration errors
    keep their original exception and exit code so the website cannot present
    them as a GPU failure.
    """

    if not _supervised_isaac_worker():
        return
    backend_poisoned = _likely_poisoned_isaac_backend_error(exc)
    if not backend_poisoned:
        _write_isaac_worker_status(
            "startup_error",
            task_id=task_id,
            failure_phase=phase,
            error_type=type(exc).__name__,
            error=str(exc),
            backend_poisoned=False,
        )
        print(
            "[Isaac watchdog] startup failed with an ordinary task/runtime "
            "exception; preserving the original exception and exit code.",
            flush=True,
        )
        return
    _write_isaac_worker_status(
        "fatal_startup",
        task_id=task_id,
        failure_phase=phase,
        error_type=type(exc).__name__,
        error=str(exc),
        backend_poisoned=True,
    )
    print(
        "[Isaac watchdog] startup failed before the stable physical origin; "
        "bypassing unsafe Kit/PhysX teardown so the supervisor can recover.",
        flush=True,
    )
    traceback.print_exc()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(_ISAAC_FATAL_BACKEND_EXIT_CODE)


def _uses_e01_detached_hand_pipeline(exp_name: str, sim_backend: str) -> bool:
    """Return whether E01 records hands live and reconstructs cloth/arms later."""
    return bool(
        exp_name in _E01_DEFERRED_ARM_COMPLETION_TASKS
        and str(sim_backend).strip().lower() in _E01_DEFERRED_POSTPROCESS_BACKENDS
    )


def _saved_demos_for_deferred_arm_replay(
    out_dir: str | Path,
    *,
    exp_name: str,
    session_id: str,
    existing: set[Path],
) -> list[Path]:
    output_root = Path(out_dir).expanduser().resolve()
    candidates = sorted(
        path
        for path in output_root.glob(f"{exp_name}_demo_*")
        if path.is_dir()
        and (path / "replay.zarr").is_dir()
        and path.resolve() not in existing
    )
    if not session_id:
        return candidates

    matching: list[Path] = []
    for path in candidates:
        try:
            metadata = json.loads(
                (path / "collection_meta.json").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        if str(metadata.get("collection_session_id") or "") == session_id:
            matching.append(path)
    return matching


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(
        timespec="seconds"
    )


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json_object_best_effort(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _legacy_dataset_generation() -> DatasetGeneration:
    return DatasetGeneration(
        name=LEGACY_DATASET_GENERATION,
        training_status="legacy_invalid_do_not_train",
        directory_suffix="",
        marker_file="",
        stable_origin=None,
    )


def _dataset_generation_for_env(env) -> DatasetGeneration:
    raw_env = getattr(env, "unwrapped", env)
    # Task guides are machine-local planning inputs, never benchmark
    # episodes.  Fail closed even if a reused test/runtime object happens to
    # retain a complete stable-origin mapping from an earlier formal run.
    if bool(getattr(raw_env, "_guidance_mode", False)) or (
        os.environ.get("HANDSON_TASK_GUIDANCE_RECORDING", "")
        .strip()
        .lower()
        not in {"", "0", "false", "no", "off"}
    ):
        return _legacy_dataset_generation()
    if str(getattr(raw_env, "_physics_backend", "")).strip().lower() == "newton":
        if str(getattr(raw_env, "task_name", "")).strip().lower() == (
            "t09_insert_peg"
        ):
            return DatasetGeneration(
                name=NEWTON_CROSS_GPU_V2_DATASET_GENERATION,
                training_status=NEWTON_CROSS_GPU_TRAINING_STATUS,
                directory_suffix=NEWTON_CROSS_GPU_V2_DIRECTORY_SUFFIX,
                marker_file=NEWTON_CROSS_GPU_V2_MARKER_FILE,
                stable_origin=None,
            )
        return DatasetGeneration(
            name=NEWTON_V1_DATASET_GENERATION,
            training_status="newton_backend_requires_separate_validation",
            directory_suffix=NEWTON_V1_DIRECTORY_SUFFIX,
            marker_file=NEWTON_V1_MARKER_FILE,
            stable_origin=None,
        )
    origin = getattr(raw_env, "_physical_recording_origin", None)
    if not isinstance(origin, Mapping):
        return _legacy_dataset_generation()
    try:
        stable_steps_required = int(origin.get("stable_steps_required", 0))
        stable_steps_observed = int(origin.get("stable_steps_observed", 0))
    except (TypeError, ValueError):
        return _legacy_dataset_generation()
    physics_steps = origin.get("physics_steps")
    velocity_clamp_steps = origin.get(
        "stabilization_velocity_clamp_steps"
    )
    origin_task_name = str(
        origin.get("task_name") or getattr(raw_env, "task_name", "")
    ).strip().lower()
    is_abc_collection_task = is_abc_style_collection_task(origin_task_name)
    stable_origin_profile = stable_recording_origin_kwargs_for_task(
        origin_task_name
    )
    expected_sleeping = bool(
        stable_origin_profile.get(
            "post_settle_sleeping",
            not is_abc_collection_task,
        )
    )
    sleep_contract_matches = (
        (
            origin.get("post_settle_sleeping") is True
            and origin.get("contact_warm_start_preserved") is False
        )
        if expected_sleeping
        else (
            origin.get("post_settle_sleeping") is False
            and origin.get("post_settle_sleep") is None
            and origin.get("final_boundary_sleep") is None
            and origin.get("contact_warm_start_preserved") is True
        )
    )
    expected_velocity_clamp = bool(
        stable_origin_profile.get("stabilization_velocity_clamp", False)
    )
    expected_velocity_clamp_scope = str(
        stable_origin_profile.get(
            "stabilization_velocity_clamp_scope",
            VELOCITY_CLAMP_SCOPE_NONE,
        )
    )
    initial_settle_steps = origin.get("initial_settle_steps")
    velocity_clamp_contract_matches = (
        origin.get("stabilization_velocity_clamp") is True
        and origin.get("initialization_per_physics_step_velocity_writes")
        is True
        and origin.get("stabilization_velocity_clamp_scope")
        == expected_velocity_clamp_scope
        and type(physics_steps) is int
        and type(initial_settle_steps) is int
        and type(velocity_clamp_steps) is int
        and velocity_clamp_steps == physics_steps - initial_settle_steps
        and velocity_clamp_steps > 0
        and isinstance(
            origin.get("stabilization_velocity_clamp_write"),
            Mapping,
        )
        if expected_velocity_clamp
        else (
            origin.get("stabilization_velocity_clamp") is False
            and origin.get("initialization_per_physics_step_velocity_writes")
            is False
            and origin.get("stabilization_velocity_clamp_scope")
            == VELOCITY_CLAMP_SCOPE_NONE
            and type(velocity_clamp_steps) is int
            and velocity_clamp_steps == 0
            and origin.get("stabilization_velocity_clamp_write") is None
        )
    )
    has_natural_settle_then_zero = bool(
        velocity_clamp_contract_matches
        and type(physics_steps) is int
        and physics_steps > 0
        and type(initial_settle_steps) is int
        and initial_settle_steps > 0
        and initial_settle_steps <= physics_steps
        and origin.get("post_settle_velocity_zeroing") is True
        and origin.get("boundary_state_write_mode") == "velocity_only"
        and sleep_contract_matches
        and (
            not is_abc_collection_task
            or origin.get("stability_sample_steps") == 1
        )
        and origin.get("final_boundary_task_and_joint_velocity_exact_zero")
        is True
    )
    stability_metric = origin.get("stability_metric")
    metric_is_certifiable = stability_metric == "velocity" or (
        stability_metric == "velocity_or_static_pose"
        and origin_task_name in STATIC_POSE_FALLBACK_TASKS
        and origin.get("full_frame_pose_gate") is True
        and origin.get("stability_sample_steps") == 1
    )
    if not (
        origin.get("strategy") == STABLE_V2_ORIGIN_STRATEGY
        and metric_is_certifiable
        and has_natural_settle_then_zero
        and origin.get("state_writes") is True
        and origin.get("per_frame_state_writes") is False
        and stable_steps_required > 0
        and stable_steps_observed >= stable_steps_required
    ):
        return _legacy_dataset_generation()
    deferred_certification = _deferred_physical_replay_certification_enabled()
    fresh_process_worker = _environment_flag_enabled(
        _FRESH_CERTIFIED_COLLECTION_WORKER_ENV
    )
    # V3 requires a genuinely new PhysX Scene for every candidate.  The
    # persistent recorder proves that with an audited in-process hard reset;
    # the fresh-process supervisor provides the stronger equivalent by
    # allowing exactly one recording attempt in a brand-new Isaac process.
    # Keep the two modes disjoint so a stale deferred-mode environment cannot
    # accidentally turn a persistent FU006 worker into V3 data.
    fresh_scene_source = bool(
        (
            deferred_certification
            and _isaac_in_process_hard_reset_enabled_for_task(
                getattr(raw_env, "task_name", "")
            )
            and callable(getattr(raw_env, "hard_reset_physx_scene", None))
        )
        or (fresh_process_worker and not deferred_certification)
    )
    fresh_scene_v3 = bool(
        fresh_scene_source
        and _environment_flag_enabled("HANDSON_RECORD_PHYSX_GROUND_TRUTH")
    )
    return DatasetGeneration(
        name=(
            PHYSX_V3_DATASET_GENERATION
            if fresh_scene_v3
            else STABLE_V2_DATASET_GENERATION
        ),
        training_status="new_requires_physical_certification",
        directory_suffix=(
            PHYSX_V3_DIRECTORY_SUFFIX
            if fresh_scene_v3
            else STABLE_V2_DIRECTORY_SUFFIX
        ),
        marker_file=(
            PHYSX_V3_MARKER_FILE
            if fresh_scene_v3
            else STABLE_V2_MARKER_FILE
        ),
        stable_origin=copy.deepcopy(dict(origin)),
    )


def _dataset_generation_metadata(
    generation: DatasetGeneration,
    *,
    episode_id: str,
) -> dict[str, object]:
    return {
        "episode_id": str(episode_id),
        "dataset_generation": generation.name,
        "training_generation_status": generation.training_status,
        "dataset_directory_suffix": generation.directory_suffix,
        "dataset_marker_file": generation.marker_file,
        "is_stable_v2_data": generation.is_stable_v2,
        "is_physx_v3_data": generation.is_physx_v3,
        "is_newton_v1_data": generation.is_newton_v1,
        "is_newton_cross_gpu_v2_data": generation.is_newton_cross_gpu_v2,
    }


def _dataset_generation_suffix_for_episode(episode_id: str) -> str:
    return next(
        (
            suffix
            for suffix in DATASET_GENERATION_DIRECTORY_SUFFIXES
            if str(episode_id).endswith(suffix)
        ),
        "",
    )


def _write_dataset_generation_marker(
    demo_dir: Path,
    generation: DatasetGeneration,
    *,
    episode_id: str,
) -> Path | None:
    if not (
        generation.is_stable_v2
        or generation.is_newton_v1
        or generation.is_newton_cross_gpu_v2
    ):
        return None
    marker_path = Path(demo_dir) / generation.marker_file
    _write_json_atomic(
        marker_path,
        {
            "format": "handson.dataset_generation",
            "version": 1,
            **_dataset_generation_metadata(
                generation,
                episode_id=episode_id,
            ),
            "old_data_policy": (
                "episodes_without_DATASET_PHYSX_V3_are_pre_fresh_scene_data_"
                "do_not_mix"
                if generation.is_physx_v3
                else (
                    "episodes_without_DATASET_NEWTON_CROSS_GPU_V2_are_"
                    "pre_certification_newton_data_do_not_train"
                )
                if generation.is_newton_cross_gpu_v2
                else "episodes_without_DATASET_NEWTON_V1_are_not_newton_data_do_not_mix"
                if generation.is_newton_v1
                else "episodes_without_this_marker_are_legacy_invalid_do_not_train"
            ),
            "stable_recording_origin": copy.deepcopy(
                generation.stable_origin
            ),
        },
    )
    return marker_path


def _write_deferred_arm_replay_status(
    demo_dir: Path,
    payload: Mapping[str, object],
) -> None:
    _write_json_atomic(demo_dir / _ISAAC_DEFERRED_ARM_STATUS_FILE, payload)


def _wait_for_isaac_collection_worker(
    worker: subprocess.Popen,
    *,
    startup_marker: Path,
    status_path: Path,
    startup_timeout: float,
    heartbeat_timeout: float,
) -> tuple[int, bool]:
    """Wait with hard startup/runtime liveness bounds."""

    startup_started_at = time.monotonic()
    startup_stall_deadline = startup_started_at + startup_timeout
    startup_absolute_timeout = max(
        startup_timeout * 10.0,
        startup_timeout + 60.0,
    )
    startup_absolute_deadline = (
        startup_started_at + startup_absolute_timeout
    )
    startup_status_mtime_ns: int | None = None
    startup_ready = False
    while True:
        if not startup_ready and startup_marker.is_file():
            startup_ready = True
            _write_collection_runtime_status(
                "running",
                "稳定初态验证通过，采集运行中。",
                worker_pid=int(getattr(worker, "pid", 0) or 0),
            )
            print(
                "[Isaac deferred arm supervisor] stable physical origin is "
                "ready; collection is live",
                flush=True,
            )

        if not startup_ready:
            try:
                status_mtime_ns = status_path.stat().st_mtime_ns
            except (FileNotFoundError, OSError):
                status_mtime_ns = None
            if (
                status_mtime_ns is not None
                and status_mtime_ns != startup_status_mtime_ns
            ):
                startup_status_mtime_ns = status_mtime_ns
                startup_stall_deadline = time.monotonic() + startup_timeout
                status = _isaac_worker_status(status_path)
                _write_collection_runtime_status(
                    "stabilizing_initial_state",
                    _isaac_startup_progress_detail(status),
                    worker_pid=int(getattr(worker, "pid", 0) or 0),
                    worker_phase=str(status.get("phase") or "unknown"),
                    stability_progress=status.get("stability_progress"),
                )
            now_monotonic = time.monotonic()
            absolute_remaining = startup_absolute_deadline - now_monotonic
            if absolute_remaining <= 0.0:
                _write_collection_runtime_status(
                    "recovering",
                    "Isaac 总启动时间超限，正在回收进程组。",
                    worker_pid=int(getattr(worker, "pid", 0) or 0),
                )
                stopped_returncode = _stop_isaac_deferred_collection_worker(
                    worker,
                    force=True,
                )
                raise TimeoutError(
                    "Isaac collection did not finish PhysX scene startup "
                    "within its absolute bounded startup window of "
                    f"{startup_absolute_timeout:.0f}s; the complete worker "
                    "process group was stopped. "
                    f"worker_returncode={stopped_returncode}"
                )
            remaining = min(
                startup_stall_deadline - now_monotonic,
                absolute_remaining,
            )
            if remaining <= 0.0:
                _write_collection_runtime_status(
                    "recovering",
                    "Isaac 启动心跳超时，正在回收无响应的进程组。",
                    worker_pid=int(getattr(worker, "pid", 0) or 0),
                )
                stopped_returncode = _stop_isaac_deferred_collection_worker(
                    worker,
                    force=True,
                )
                raise TimeoutError(
                    "Isaac collection did not finish PhysX scene startup because "
                    "it published no scene/stabilization progress for "
                    f"{startup_timeout:.0f}s; the complete worker process "
                    "group was stopped instead of remaining unresponsive. "
                    f"worker_returncode={stopped_returncode}"
                )
            wait_timeout = min(0.5, remaining)
        else:
            wait_timeout = 0.5
            if status_path.is_file():
                heartbeat_age = max(
                    0.0,
                    time.time() - status_path.stat().st_mtime,
                )
                allowed_heartbeat_age = heartbeat_timeout
                status: dict[str, object] = {}
                phase = "unknown"
                if heartbeat_age > heartbeat_timeout:
                    status = _isaac_worker_status(status_path)
                    phase = str(status.get("phase") or "unknown")
                    if phase in {
                        "stabilizing_initial_state",
                        "stabilizing_reset_state",
                    }:
                        allowed_heartbeat_age = max(
                            allowed_heartbeat_age,
                            startup_timeout,
                        )
                    elif phase == "saving_episode":
                        allowed_heartbeat_age = max(
                            allowed_heartbeat_age,
                            _ISAAC_EPISODE_SAVE_TIMEOUT_SECONDS,
                        )
                if heartbeat_age > allowed_heartbeat_age:
                    _write_collection_runtime_status(
                        "recovering",
                        "Isaac 运行心跳超时，正在回收卡住的进程组。",
                        worker_pid=int(getattr(worker, "pid", 0) or 0),
                        worker_phase=phase,
                    )
                    stopped_returncode = (
                        _stop_isaac_deferred_collection_worker(
                            worker,
                            force=True,
                        )
                    )
                    raise TimeoutError(
                        "Isaac collection heartbeat stopped for "
                        f"{heartbeat_age:.0f}s during phase={phase}; the "
                        "complete worker process group was stopped instead "
                        "of leaving the website in a false running state. "
                        f"worker_returncode={stopped_returncode}"
                    )

        try:
            return worker.wait(timeout=wait_timeout), startup_ready
        except subprocess.TimeoutExpired:
            continue


def _run_isaac_deferred_collection_supervisor(
    *,
    exp_name: str,
    out_dir: str | Path,
    video_fps: int,
    certification_timeout_seconds: int = 1800,
) -> None:
    """Run arm-free collection in a child, then render complete robots."""
    output_root = Path(out_dir).expanduser().resolve()
    deferred_physical_publication = (
        _deferred_physical_replay_certification_enabled()
    )
    postprocess_manual_only = _postprocess_manual_only_enabled()
    deferred_physical_autostart = (
        _deferred_physical_replay_autostart_enabled()
    )
    existing = {
        path.resolve()
        for path in output_root.glob(f"{exp_name}_demo_*")
        if path.is_dir()
    }
    base_worker_env = configure_isaac_gpu_environment(
        os.environ,
        render_mode="human",
    )
    base_worker_env[_ISAAC_DEFERRED_ARM_WORKER_ENV] = "1"
    startup_timeout = _isaac_startup_timeout_seconds(base_worker_env)
    heartbeat_timeout = _isaac_runtime_heartbeat_timeout_seconds(
        base_worker_env
    )
    max_attempts = _isaac_startup_max_attempts(base_worker_env)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    interrupted = False
    worker_returncode: int | None = None
    worker: subprocess.Popen | None = None
    terminal_worker_status: dict[str, object] = {}
    session_id = os.environ.get("HANDSON_COLLECTION_SESSION_ID", "").strip()
    try:
        for attempt in range(1, max_attempts + 1):
            startup_marker = Path(tempfile.gettempdir()) / (
                f"handson-isaac-ready-{os.getpid()}-"
                f"{secrets.token_hex(8)}.json"
            )
            status_path = Path(tempfile.gettempdir()) / (
                f"handson-isaac-status-{os.getpid()}-"
                f"{secrets.token_hex(8)}.json"
            )
            worker_env = dict(base_worker_env)
            worker_env[_ISAAC_STARTUP_READY_FILE_ENV] = str(startup_marker)
            worker_env[_ISAAC_WORKER_STATUS_FILE_ENV] = str(status_path)
            _write_collection_runtime_status(
                "stabilizing_initial_state",
                "正在构建 Isaac 场景并等待、验证稳定初态。",
                startup_attempt=attempt,
                startup_attempts_allowed=max_attempts,
            )
            print(
                "[Isaac deferred arm supervisor] starting detached-hand "
                f"collection (startup attempt {attempt}/{max_attempts})",
                flush=True,
            )
            worker = subprocess.Popen(
                command,
                env=worker_env,
                start_new_session=True,
            )
            startup_ready = False
            watchdog_error: TimeoutError | None = None
            try:
                try:
                    worker_returncode, startup_ready = (
                        _wait_for_isaac_collection_worker(
                            worker,
                            startup_marker=startup_marker,
                            status_path=status_path,
                            startup_timeout=startup_timeout,
                            heartbeat_timeout=heartbeat_timeout,
                        )
                    )
                except TimeoutError as exc:
                    watchdog_error = exc
                    worker_returncode = worker.poll()
            except KeyboardInterrupt:
                interrupted = True
                _write_collection_runtime_status(
                    "stopping",
                    "收到停止请求，正在封存有效数据并回收仿真进程。",
                    worker_pid=int(getattr(worker, "pid", 0) or 0),
                )
                print(
                    "[Isaac deferred arm supervisor] collection stop "
                    "requested; waiting for saved-demo post-collection work",
                    flush=True,
                )
                worker_returncode = _stop_isaac_deferred_collection_worker(
                    worker,
                )
            except Exception:
                # Recovery/setup failures must never orphan the independent
                # inner process group.
                if worker.poll() is None:
                    _stop_isaac_deferred_collection_worker(
                        worker,
                        force=True,
                    )
                raise
            finally:
                terminal_worker_status = _isaac_worker_status(status_path)
                startup_marker.unlink(missing_ok=True)
                status_path.unlink(missing_ok=True)

            if interrupted:
                break

            published_during_attempt = any(
                path.is_dir() and path.resolve() not in existing
                for path in output_root.glob(f"{exp_name}_demo_*")
            )
            fatal_backend = bool(
                worker_returncode == _ISAAC_FATAL_BACKEND_EXIT_CODE
            )
            # CUDA error 700 poisons the process and often the driver's next
            # Isaac context as well.  Never launch an automatic second worker
            # after exit 86; let the UI tell the operator to restart first.
            # A watchdog timeout without a CUDA signal may still get the one
            # bounded clean-process retry.
            retryable_startup_failure = bool(
                not startup_ready
                and not fatal_backend
                and watchdog_error is not None
            )
            if (
                retryable_startup_failure
                and not published_during_attempt
                and attempt < max_attempts
            ):
                reason = (
                    str(watchdog_error)
                    if watchdog_error is not None
                    else (
                        "the worker reported a fatal CUDA/PhysX backend "
                        f"failure (exit code {worker_returncode})"
                    )
                )
                quarantined = _quarantine_failed_isaac_startup_journal(
                    output_root,
                    session_id=session_id,
                    reason=reason,
                )
                evidence = (
                    f"; evidence preserved at {quarantined}"
                    if quarantined is not None
                    else ""
                )
                print(
                    "[Isaac watchdog] startup failed safely before any "
                    "episode was accepted; launching one clean process retry"
                    f"{evidence}",
                    flush=True,
                )
                _write_collection_runtime_status(
                    "recovering",
                    "PhysX/GPU 启动失败已安全回收，正在启动一次干净重试。",
                    failed_startup_attempt=attempt,
                    next_startup_attempt=attempt + 1,
                )
                continue

            if watchdog_error is not None:
                _write_collection_runtime_status(
                    "failed",
                    str(watchdog_error),
                    returncode=worker_returncode,
                )
                raise RuntimeError(str(watchdog_error)) from watchdog_error
            break
        else:  # pragma: no cover - the bounded loop always breaks or raises
            raise RuntimeError("Isaac startup attempts were exhausted.")
    finally:
        # Never let unrelated replay recovery delay reaping a failed startup.
        # The previous startup-time scan could block on a large old Zarr chunk
        # while the newly failed worker remained a zombie and the website kept
        # reporting collection as running.  Successful/stopped sessions still
        # trigger the normal post-collection recovery below.
        if deferred_physical_autostart and (
            worker_returncode == 0 or interrupted
        ):
            ensured = _ensure_deferred_physical_replay_certification(
                output_root,
                exp_name=exp_name,
                certification_runs=(
                    DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS
                ),
                certification_timeout_seconds=(
                    certification_timeout_seconds
                ),
                session_id=os.environ.get(
                    "HANDSON_COLLECTION_SESSION_ID",
                    "",
                ),
                source_recorder_pid=os.getpid(),
            )
            if ensured:
                print(
                    "[PhysX post-collection certification] ensured "
                    "background certification for "
                    f"{ensured} pending candidate(s)",
                    flush=True,
                )
    replay_count: int | None = None
    if not deferred_physical_publication:
        replay_count = _launch_pending_isaac_deferred_arm_replays(
            output_root,
            exp_name=exp_name,
            session_id=os.environ.get("HANDSON_COLLECTION_SESSION_ID", ""),
            existing=existing,
            video_fps=video_fps,
        )
    if worker_returncode != 0 and not interrupted:
        restart_target = (
            "AS019 任务"
            if exp_name == "t22_pass_screwdriver"
            else "当前 Isaac 任务"
        )
        fatal_backend = bool(
            worker_returncode == _ISAAC_FATAL_BACKEND_EXIT_CODE
        )
        worker_phase = str(terminal_worker_status.get("phase") or "")
        worker_error_type = str(
            terminal_worker_status.get("error_type") or ""
        )
        worker_error = str(terminal_worker_status.get("error") or "")
        ordinary_startup_error = bool(
            worker_phase == "startup_error" and worker_error
        )
        ordinary_detail = (
            f"Isaac 任务启动失败：{worker_error_type}: {worker_error}"
            if ordinary_startup_error
            else f"采集进程异常退出（returncode={worker_returncode}）。"
        )
        failure_hint = (
            " (fatal CUDA/PhysX backend failure; the process group was "
            f"recovered; the website will ask the operator to restart {restart_target})"
            if fatal_backend
            else (
                f" ({worker_error_type}: {worker_error})"
                if ordinary_startup_error
                else ""
            )
        )
        _write_collection_runtime_status(
            "failed",
            (
                "PhysX/GPU 后端失败；损坏的进程组已回收，请重启"
                f"{restart_target}后再试。"
                if fatal_backend
                else ordinary_detail
            ),
            returncode=worker_returncode,
            code=(
                "physx_cuda_700_restart_required"
                if fatal_backend
                else "isaac_startup_error"
                if ordinary_startup_error
                else ""
            ),
            restart_required=fatal_backend,
            worker_phase=worker_phase,
            error_type=worker_error_type,
            error=worker_error,
        )
        raise RuntimeError(
            "Detached-hand Isaac collection failed with exit code "
            f"{worker_returncode}{failure_hint}"
        )
    if interrupted:
        _write_collection_runtime_status(
            "stopped",
            "采集已停止，有效数据和会话证据已封存。",
            returncode=worker_returncode,
        )
    if replay_count is None:
        if postprocess_manual_only:
            print(
                "[Isaac deferred arm supervisor] postprocessing is manual-only; "
                "sealed pending candidates are queued until the operator starts "
                "certification, rendering, and upload",
                flush=True,
            )
        else:
            print(
                "[Isaac deferred arm supervisor] physical certification is "
                "publishing candidates asynchronously; arm replay discovery is "
                "deferred until certified demos become visible",
                flush=True,
            )
    else:
        print(
            "[Isaac deferred arm supervisor] queued "
            f"{replay_count} background arm replay(s)",
            flush=True,
        )


def _stop_isaac_deferred_collection_worker(
    worker: subprocess.Popen,
    *,
    already_signaled: bool = False,
    force: bool = False,
) -> int | None:
    """Drain an independent worker process group with bounded escalation."""

    def _signal_group(sig: int) -> int | None:
        if worker.poll() is not None:
            return worker.returncode
        worker_pid = getattr(worker, "pid", None)
        if worker_pid is not None:
            try:
                os.killpg(int(worker_pid), sig)
                return None
            except ProcessLookupError:
                # Unit-test fakes and a child which has not completed setsid()
                # can legitimately have no matching process group.  Fall back
                # to the direct process API without weakening real cleanup.
                pass
            except OSError:
                pass
        try:
            if sig == signal.SIGTERM:
                worker.terminate()
            elif sig == signal.SIGKILL:
                worker.kill()
            else:
                worker.send_signal(sig)
        except ProcessLookupError:
            return worker.poll()
        return None

    if worker.poll() is not None:
        return worker.returncode

    if force:
        _signal_group(signal.SIGTERM)
        try:
            return worker.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _signal_group(signal.SIGKILL)
        try:
            return worker.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            return worker.poll()

    if not already_signaled:
        result = _signal_group(signal.SIGINT)
        if result is not None:
            return result
    try:
        return worker.wait(
            timeout=(
                _ISAAC_COLLECTION_DRAIN_TIMEOUT_SECONDS
                if already_signaled
                else 15.0
            )
        )
    except subprocess.TimeoutExpired:
        pass
    result = _signal_group(signal.SIGTERM)
    if result is not None:
        return result
    try:
        return worker.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass
    result = _signal_group(signal.SIGKILL)
    if result is not None:
        return result
    try:
        return worker.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        return worker.poll()


def _launch_pending_isaac_deferred_arm_replays(
    output_root: Path,
    *,
    exp_name: str,
    session_id: str,
    existing: set[Path],
    video_fps: int,
) -> int:
    demos = _saved_demos_for_deferred_arm_replay(
        output_root,
        exp_name=exp_name,
        session_id=session_id,
        existing=existing,
    )
    if not demos:
        return 0

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = output_root / ".deferred_arm_replay_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_session = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in (session_id or exp_name)
    ).strip("_") or exp_name
    log_path = log_dir / f"{safe_session}-{timestamp}.log"
    worker_script = Path(__file__).resolve().parent / "isaac_deferred_arm_replay_worker.py"
    command = [
        sys.executable,
        str(worker_script),
        f"--exp_name={exp_name}",
        f"--video_fps={max(1, int(video_fps))}",
        f"--log_path={log_path}",
        *[str(demo.resolve()) for demo in demos],
    ]
    env_vars = os.environ.copy()
    env_vars.pop(_ISAAC_DEFERRED_ARM_WORKER_ENV, None)
    env_vars["HANDSON_ISAACLAB_DEFER_ARM_RENDER"] = "0"
    env_vars["HANDSON_ISAACLAB_DIRECT_HAND_CONTROL"] = "0"
    env_vars["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "0"
    pythonpath = env_vars.get("PYTHONPATH", "")
    pythonpath_entries = [str(_PACKAGE_ROOT), str(_PACKAGE_ROOT.parent)]
    pythonpath_entries.extend(part for part in pythonpath.split(os.pathsep) if part)
    env_vars["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath_entries))

    queued_at = _now_iso()
    for demo_dir in demos:
        existing.add(demo_dir.resolve())
        _write_deferred_arm_replay_status(
            demo_dir,
            {
                "status": "queued",
                "queued_at": queued_at,
                "updated_at": queued_at,
                "exp_name": exp_name,
                "video_fps": max(1, int(video_fps)),
                "command": command,
                "log_path": str(log_path),
            },
        )

    try:
        log_handle = log_path.open("ab")
        try:
            process = subprocess.Popen(
                command,
                cwd=str(_PACKAGE_ROOT.parent),
                env=env_vars,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_handle.close()
    except Exception as exc:  # noqa: BLE001
        failed_at = _now_iso()
        for demo_dir in demos:
            _write_deferred_arm_replay_status(
                demo_dir,
                {
                    "status": "failed",
                    "queued_at": queued_at,
                    "updated_at": failed_at,
                    "completed_at": failed_at,
                    "exp_name": exp_name,
                    "video_fps": max(1, int(video_fps)),
                    "command": command,
                    "log_path": str(log_path),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        raise

    print(
        "[Isaac deferred arm supervisor] background replay worker "
        f"pid={process.pid} demos={len(demos)} log={log_path}",
        flush=True,
    )
    return len(demos)


def _run_pending_isaac_deferred_arm_replays(
    output_root: Path,
    *,
    exp_name: str,
    session_id: str,
    existing: set[Path],
    video_fps: int,
) -> int:
    demos = _saved_demos_for_deferred_arm_replay(
        output_root,
        exp_name=exp_name,
        session_id=session_id,
        existing=existing,
    )
    failures: list[Path] = []
    for demo_dir in demos:
        existing.add(demo_dir.resolve())
        result = _run_isaac_deferred_arm_replay(
            demo_dir,
            exp_name=exp_name,
            video_fps=video_fps,
        )
        if result.returncode != 0:
            failures.append(demo_dir)
    if failures:
        failed = ", ".join(str(path) for path in failures)
        raise RuntimeError(f"Deferred Isaac arm replay failed for: {failed}")
    return len(demos)


def _launch_e01_postprocess_batch(
    demo_dirs: list[str] | list[Path],
    *,
    exp_name: str,
    sim_backend: str,
    ipc_env_value: str,
    arm_env_value: str,
    video_fps: int,
) -> int:
    demos = [
        Path(demo).expanduser().resolve()
        for demo in demo_dirs
        if (Path(demo).expanduser().resolve() / "replay.zarr").is_dir()
    ]
    if not demos:
        return 0

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = demos[0].parent / ".e01_ipc_postprocess_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_session = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in (os.environ.get("HANDSON_COLLECTION_SESSION_ID", "") or exp_name)
    ).strip("_") or exp_name
    log_path = log_dir / f"{safe_session}-{timestamp}.log"
    worker_script = Path(__file__).resolve().parent / "e01_ipc_postprocess_worker.py"
    command = [
        sys.executable,
        str(worker_script),
        f"--exp_name={exp_name}",
        f"--sim_backend={sim_backend}",
        f"--ipc_env_value={ipc_env_value}",
        f"--arm_env_value={arm_env_value}",
        f"--video_fps={max(1, int(video_fps))}",
        f"--log_path={log_path}",
        *[str(demo) for demo in demos],
    ]

    env_vars = os.environ.copy()
    env_vars[_E01_IPC_POSTPROCESS_WORKER_ENV] = "1"
    pythonpath = env_vars.get("PYTHONPATH", "")
    pythonpath_entries = [str(_PACKAGE_ROOT), str(_PACKAGE_ROOT.parent)]
    pythonpath_entries.extend(part for part in pythonpath.split(os.pathsep) if part)
    env_vars["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath_entries))

    queued_at = _now_iso()
    requested_stages = _e01_postprocess_requested_stages(
        exp_name,
        sim_backend=sim_backend,
        ipc_env_value=ipc_env_value,
        arm_env_value=arm_env_value,
    )

    for demo_dir in demos:
        _write_e01_postprocess_document(
            demo_dir,
            {
                "format": "handson.e01_postprocess",
                "version": 1,
                "status": "queued",
                "queued_at": queued_at,
                "started_at": None,
                "completed_at": None,
                "background": True,
                "exp_name": exp_name,
                "sim_backend": sim_backend,
                "video_fps": max(1, int(video_fps)),
                "requested_stages": requested_stages,
                "command": command,
                "log_path": str(log_path),
                "stages": {},
            },
        )

    try:
        log_handle = log_path.open("ab")
        try:
            process = subprocess.Popen(
                command,
                cwd=str(_PACKAGE_ROOT.parent),
                env=env_vars,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_handle.close()
    except Exception as exc:  # noqa: BLE001
        failed_at = _now_iso()
        for demo_dir in demos:
            _write_e01_postprocess_document(
                demo_dir,
                {
                    "format": "handson.e01_postprocess",
                    "version": 1,
                    "status": "failed",
                    "queued_at": queued_at,
                    "started_at": None,
                    "completed_at": failed_at,
                    "background": True,
                    "exp_name": exp_name,
                    "sim_backend": sim_backend,
                    "video_fps": max(1, int(video_fps)),
                    "requested_stages": requested_stages,
                    "command": command,
                    "log_path": str(log_path),
                    "stages": {},
                    "failed_stages": requested_stages,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        raise

    print(
        "[E01 postprocess] background IPC worker "
        f"pid={process.pid} demos={len(demos)} log={log_path}",
        flush=True,
    )
    return len(demos)


def _deferred_physical_worker_state_path(demo_dir: Path) -> Path:
    return demo_dir.parent / DEFERRED_PHYSICAL_REPLAY_WORKER_STATE_DIRECTORY / (
        f"{demo_dir.name}.json"
    )


def _demo_collection_session_id(demo_dir: Path) -> str:
    metadata = _read_json_object_best_effort(
        demo_dir / "collection_meta.json"
    )
    return str(metadata.get("collection_session_id") or "")


def _is_sha256_hex(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _invalid_outcome_evidence_report_is_bound(
    demo_dir: Path,
    acceptance: Mapping[str, object],
    evidence: Mapping[str, object],
) -> bool:
    relative = evidence.get("report")
    expected_sha256 = evidence.get("report_sha256")
    evidence_run = evidence.get("run")
    if (
        not isinstance(relative, str)
        or type(evidence_run) is not int
        or evidence_run <= 0
        or relative
        != f"{PHYSICAL_REPLAY_RUN_DIRECTORY}/run_{evidence_run:03d}.json"
        or not _is_sha256_hex(expected_sha256)
    ):
        return False
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or relative_path.parts[:1] != (PHYSICAL_REPLAY_RUN_DIRECTORY,)
        or len(relative_path.parts) != 2
        or relative_path.name in {"", ".", ".."}
        or not relative_path.name.startswith("run_")
        or relative_path.suffix != ".json"
        or len(relative_path.stem) != len("run_000")
        or not relative_path.stem.removeprefix("run_").isdigit()
    ):
        return False
    cursor = demo_dir
    for part in relative_path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            return False
    try:
        report_path = (demo_dir / relative_path).resolve(strict=True)
        report_path.relative_to(demo_dir.resolve(strict=True))
    except (FileNotFoundError, OSError, ValueError):
        return False
    if not report_path.is_file() or report_path.is_symlink():
        return False
    try:
        if _sha256_regular_file(report_path) != expected_sha256:
            return False
    except OSError:
        return False

    rows = acceptance.get("certification_runs")
    if not isinstance(rows, list):
        return False
    matching_rows = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("run") == evidence_run
        and row.get("report") == relative
        and row.get("report_sha256") == expected_sha256
    ]
    return len(matching_rows) == 1


def _has_final_failed_deferred_physical_proof(demo_dir: Path) -> bool:
    """Return true only for the current evidence-bearing invalid outcome.

    Historical ``failed``/``invalid`` acceptance documents predate the
    versioned outcome contract and are deliberately not terminal: recovery
    must submit them to the current reviewer once more.
    """

    acceptance = _read_json_object_best_effort(
        demo_dir / FRESH_PROCESS_ACCEPTANCE_NAME
    )
    evidence = acceptance.get("outcome_evidence")
    return bool(
        acceptance.get("format")
        == DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_FORMAT
        and acceptance.get("version")
        == DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_VERSION
        and acceptance.get("mode")
        == DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE
        and acceptance.get("status") == "failed"
        and acceptance.get("physical_replay_outcome_version")
        == PHYSICAL_REPLAY_OUTCOME_VERSION
        and acceptance.get("physical_replay_outcome_code")
        == PHYSICAL_REPLAY_INVALID_OUTCOME_CODE
        and acceptance.get("review_status")
        == PHYSICAL_REPLAY_REVIEW_INVALID
        and isinstance(evidence, Mapping)
        and evidence.get("format")
        == "handson.physical_replay_invalid_outcome_evidence"
        and evidence.get("version") == PHYSICAL_REPLAY_OUTCOME_VERSION
        and _is_sha256_hex(evidence.get("report_sha256"))
        and _invalid_outcome_evidence_report_is_bound(
            demo_dir,
            acceptance,
            evidence,
        )
    )


def _ordered_session_history_binding_ready(demo_dir: Path) -> bool:
    """Hide a newly written demo until its sealed session journal is bound.

    Legacy deferred candidates did not declare an ordered-session scope, so
    recovery keeps handling them as before.  Once either metadata document
    declares the new scope, both documents must contain the same complete,
    sealed binding before a worker may discover the candidate.
    """

    manifest = _read_json_object_best_effort(
        demo_dir / "replay_manifest.json"
    )
    recording = manifest.get("recording")
    if not isinstance(recording, Mapping):
        recording = {}
    metadata = _read_json_object_best_effort(
        demo_dir / "collection_meta.json"
    )
    scope_key = "physical_replay_certification_scope"
    ordered_scope = ORDERED_SESSION_PHYSICAL_REPLAY_CERTIFICATION_SCOPE
    if (
        recording.get(scope_key) != ordered_scope
        and metadata.get(scope_key) != ordered_scope
    ):
        return True
    if (
        recording.get(scope_key) != ordered_scope
        or metadata.get(scope_key) != ordered_scope
    ):
        return False
    pending_key = "physical_replay_session_history_pending"
    if recording.get(pending_key) is not False:
        return False
    if metadata.get(pending_key) is not False:
        return False

    shared_keys = (
        "physical_replay_session_history_id",
        "physical_replay_session_history_root_relative_path",
        "physical_replay_session_history_sha256",
        "physical_replay_session_history_event_count",
        "physical_replay_session_history_accepted_event_ordinal",
        "physical_replay_session_history_certified_through_event_ordinal",
    )
    if any(recording.get(key) != metadata.get(key) for key in shared_keys):
        return False
    history_id = recording.get(shared_keys[0])
    relative_path = recording.get(shared_keys[1])
    history_sha256 = recording.get(shared_keys[2])
    event_count = recording.get(shared_keys[3])
    accepted_ordinal = recording.get(shared_keys[4])
    certified_through = recording.get(shared_keys[5])
    if not isinstance(history_id, str) or not history_id.strip():
        return False
    if history_id != str(metadata.get("collection_session_id") or ""):
        return False
    if not isinstance(relative_path, str) or not relative_path.strip():
        return False
    parsed_relative_path = Path(relative_path)
    if parsed_relative_path.is_absolute() or ".." in parsed_relative_path.parts:
        return False
    expected_prefix = (
        f"{PHYSICAL_REPLAY_PENDING_DIRECTORY}/"
        f"{PHYSICAL_REPLAY_SESSION_HISTORY_DIRECTORY}/"
    )
    if not relative_path.startswith(expected_prefix):
        return False
    if not isinstance(history_sha256, str) or len(history_sha256) != 64:
        return False
    try:
        int(history_sha256, 16)
    except ValueError:
        return False
    if type(event_count) is not int or event_count <= 0:
        return False
    if (
        type(accepted_ordinal) is not int
        or accepted_ordinal < 0
        or accepted_ordinal >= event_count
    ):
        return False
    if (
        type(certified_through) is not int
        or certified_through < accepted_ordinal
        or certified_through >= event_count
    ):
        return False
    return True


def _pending_deferred_physical_replay_candidates(
    output_root: str | Path,
    *,
    exp_name: str,
    session_id: str | None = None,
    physx_v3_only: bool = False,
) -> list[Path]:
    pending_parent = _physical_replay_pending_parent(
        Path(output_root).expanduser().resolve()
    )
    if not pending_parent.is_dir():
        return []
    individually_ready: list[Path] = []
    for demo_dir in sorted(pending_parent.glob(f"{exp_name}_demo_*")):
        if not demo_dir.is_dir() or not (demo_dir / "replay.zarr").is_dir():
            continue
        if physx_v3_only and not (
            demo_dir.name.endswith(PHYSX_V3_DIRECTORY_SUFFIX)
            and (demo_dir / PHYSX_V3_MARKER_FILE).is_file()
        ):
            continue
        if not _ordered_session_history_binding_ready(demo_dir):
            continue
        if session_id is not None and (
            _demo_collection_session_id(demo_dir) != session_id
        ):
            continue
        individually_ready.append(demo_dir.resolve())

    candidates: list[Path] = []
    ordered_groups: dict[tuple[str, str, str], list[Path]] = {}
    for demo_dir in individually_ready:
        manifest = _read_json_object_best_effort(
            demo_dir / "replay_manifest.json"
        )
        recording = manifest.get("recording")
        recording = recording if isinstance(recording, Mapping) else {}
        if recording.get("physical_replay_certification_scope") != (
            ORDERED_SESSION_PHYSICAL_REPLAY_CERTIFICATION_SCOPE
        ):
            if not _has_final_failed_deferred_physical_proof(demo_dir):
                candidates.append(demo_dir)
            continue
        key = (
            str(recording.get("physical_replay_session_history_id") or ""),
            str(
                recording.get(
                    "physical_replay_session_history_root_relative_path"
                )
                or ""
            ),
            str(
                recording.get("physical_replay_session_history_sha256")
                or ""
            ),
        )
        ordered_groups.setdefault(key, []).append(demo_dir)

    output_path = Path(output_root).expanduser().resolve()
    for (_history_id, relative_path, expected_sha), demos in ordered_groups.items():
        # One current, evidence-bearing invalid member is terminal for the
        # same shared-physics proof.  Legacy failed markers remain reviewable.
        if any(_has_final_failed_deferred_physical_proof(demo) for demo in demos):
            continue
        try:
            unresolved_history = output_path / relative_path
            cursor = output_path
            for part in Path(relative_path).parts:
                cursor = cursor / part
                if cursor.is_symlink():
                    raise ValueError(
                        "ordered-session history path traverses a symlink"
                    )
            history_path = unresolved_history.resolve(strict=True)
            history_path.relative_to(output_path)
            if not history_path.is_dir():
                raise ValueError("invalid ordered-session history path")
            loaded = load_session_history(history_path, require_finalized=True)
            finalization = loaded.get("finalization")
            if not isinstance(finalization, Mapping):
                raise ValueError("ordered-session history is not finalized")
            identity = finalization.get("history_identity")
            if (
                not isinstance(identity, Mapping)
                or identity.get("sha256") != expected_sha
            ):
                raise ValueError("ordered-session history identity changed")
            accepted = finalization.get("accepted_demos")
            if not isinstance(accepted, list) or any(
                not isinstance(row, Mapping)
                or not isinstance(row.get("demo_id"), str)
                for row in accepted
            ):
                raise ValueError("ordered-session accepted list is invalid")
            expected_names = [str(row["demo_id"]) for row in accepted]
            actual_names = {demo.name for demo in demos}
            if len(expected_names) != len(set(expected_names)) or (
                set(expected_names) != actual_names
            ):
                # This is the critical all-or-none discovery condition.  A
                # crash halfway through manifest binding leaves the entire
                # session hidden until every accepted member is ready.
                continue
        except (OSError, TypeError, ValueError):
            continue
        candidates.extend(demos)
    return sorted(candidates)


def _process_command(pid: int) -> tuple[str, ...]:
    if type(pid) is not int or pid <= 1:
        return ()
    try:
        os.kill(pid, 0)
        raw_command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (ProcessLookupError, PermissionError, FileNotFoundError, OSError):
        return ()
    command = tuple(
        part.decode(errors="replace")
        for part in raw_command.split(b"\0")
        if part
    )
    return command


def _deferred_physical_worker_command(pid: int) -> tuple[str, ...]:
    command = _process_command(pid)
    worker_name = "deferred_physical_replay_certification_worker.py"
    if not any(Path(part).name == worker_name for part in command):
        return ()
    return command


def _recorded_source_recorder_pid(demo_dir: Path) -> int | None:
    manifest = _read_json_object_best_effort(
        demo_dir / "replay_manifest.json"
    )
    recording = manifest.get("recording")
    if not isinstance(recording, Mapping):
        return None
    pid = recording.get("recorder_pid")
    return pid if type(pid) is int and pid > 1 else None


def _recorded_source_recorder_is_live(demo_dir: Path) -> bool:
    pid = _recorded_source_recorder_pid(demo_dir)
    if pid is None:
        return False
    command = _process_command(pid)
    return any(
        Path(part).name == "record_demos_zarr.py" for part in command
    )


def _recover_physx_session_history_bindings(
    output_root: str | Path,
    *,
    exp_name: str,
    session_id: str | None,
) -> int:
    """Seal dead journals and restore all accepted manifest bindings.

    A recorder may be killed after its accepted event directories are durable
    but before graceful finalization, or halfway through binding N manifests.
    Recovery is all-or-none: it binds only when every demo referenced by the
    verified journal is present in the hidden pending directory.
    """

    root = Path(output_root).expanduser().resolve()
    pending_parent = _physical_replay_pending_parent(root)
    history_parent = pending_parent / PHYSICAL_REPLAY_SESSION_HISTORY_DIRECTORY
    if not history_parent.is_dir() or history_parent.is_symlink():
        return 0
    repaired = 0
    for history_path in sorted(history_parent.iterdir()):
        if history_path.is_symlink() or not history_path.is_dir():
            continue
        try:
            loaded = load_session_history(
                history_path,
                require_finalized=False,
            )
            header = loaded.get("session")
            if not isinstance(header, Mapping):
                continue
            if header.get("task_name") != exp_name:
                continue
            history_session_id = str(header.get("session_id") or "")
            if session_id is not None and history_session_id != session_id:
                continue
            finalization = loaded.get("finalization")
            journal = PhysxSessionHistoryJournal.open(history_path)
            if not isinstance(finalization, Mapping):
                metadata = header.get("metadata")
                metadata = metadata if isinstance(metadata, Mapping) else {}
                recorder_pid = metadata.get("recorder_pid")
                if type(recorder_pid) is int and recorder_pid > 1:
                    command = _process_command(recorder_pid)
                    if any(
                        Path(part).name == "record_demos_zarr.py"
                        for part in command
                    ):
                        # Never seal a journal that its collector can still
                        # append to after the scan.
                        continue
                finalization = journal.finalize()

            accepted_rows = finalization.get("accepted_demos")
            if not isinstance(accepted_rows, list) or not accepted_rows:
                continue
            names: list[str] = []
            valid_rows = True
            for row in accepted_rows:
                name = row.get("demo_id") if isinstance(row, Mapping) else None
                if (
                    not isinstance(name, str)
                    or not name
                    or Path(name).name != name
                    or name in {".", ".."}
                    or name in names
                ):
                    valid_rows = False
                    break
                names.append(name)
            if not valid_rows:
                continue
            demos = [(pending_parent / name).resolve() for name in names]
            if any(
                demo.parent != pending_parent
                or not demo.is_dir()
                or not (demo / "replay.zarr").is_dir()
                or _demo_collection_session_id(demo) != history_session_id
                for demo in demos
            ):
                continue
            not_ready = sum(
                not _ordered_session_history_binding_ready(demo)
                for demo in demos
            )
            if not_ready:
                _bind_physx_session_history_to_demos(
                    demos,
                    output_root=root,
                    journal=journal,
                    finalization=finalization,
                )
                repaired += not_ready
        except (OSError, TypeError, ValueError):
            # Corrupt/incomplete histories remain hidden for diagnosis; one
            # bad orphan must not prevent recovery of another session.
            continue
    return repaired


def _legacy_deferred_physical_state_requires_review(
    state: Mapping[str, object],
) -> bool:
    version = state.get("physical_replay_outcome_version")
    code = state.get("physical_replay_outcome_code")
    review_status = state.get("review_status")
    current_outcome = bool(
        version == PHYSICAL_REPLAY_OUTCOME_VERSION
        and (
            (
                code == PHYSICAL_REPLAY_INVALID_OUTCOME_CODE
                and review_status == PHYSICAL_REPLAY_REVIEW_INVALID
                and isinstance(state.get("outcome_evidence"), Mapping)
            )
            or (
                code == PHYSICAL_REPLAY_REVIEW_REQUIRED_OUTCOME_CODE
                and review_status == PHYSICAL_REPLAY_REVIEW_REQUIRED
            )
            or (
                code == PHYSICAL_REPLAY_VALID_OUTCOME_CODE
                and review_status == PHYSICAL_REPLAY_REVIEW_VALID
            )
        )
    )
    return bool(
        not current_outcome
        and (
            state.get("terminal") is True
            or state.get("status")
            in {
                "failed",
                "invalid",
                "launch_failed",
                "worker_exited",
                "recovery_exhausted",
                "stopped",
            }
        )
    )


def _deferred_physical_launch_attempt_count(
    demo_dir: Path,
    *,
    allow_review_required_rerun: bool = False,
    manual_review_run_id: str | None = None,
) -> int:
    state = _read_json_object_best_effort(
        _deferred_physical_worker_state_path(demo_dir)
    )
    count = state.get("launch_attempt_count")
    if count is None and state.get("status") in {
        "launching",
        "running",
        "launch_failed",
        "worker_exited",
    }:
        count = 1
    attempt_count = count if type(count) is int and count >= 0 else 0
    if (
        allow_review_required_rerun
        and manual_review_run_id
        and state.get("manual_review_run_id") == manual_review_run_id
        and attempt_count >= 1
    ):
        # One explicit automatic review pass launches each recorded session at
        # most once. Infrastructure failures stay retryable in a later pass,
        # but must not relaunch the same session forever in this pass.
        return MAX_DEFERRED_PHYSICAL_REPLAY_LAUNCH_ATTEMPTS
    if allow_review_required_rerun and (
        state.get("physical_replay_outcome_version")
        == PHYSICAL_REPLAY_OUTCOME_VERSION
        and state.get("physical_replay_outcome_code")
        == PHYSICAL_REPLAY_REVIEW_REQUIRED_OUTCOME_CODE
        and state.get("review_status") == PHYSICAL_REPLAY_REVIEW_REQUIRED
    ):
        return 0
    if (
        state.get("physical_replay_outcome_version")
        == PHYSICAL_REPLAY_OUTCOME_VERSION
        and state.get("physical_replay_outcome_code")
        == PHYSICAL_REPLAY_INVALID_OUTCOME_CODE
        and not _has_final_failed_deferred_physical_proof(demo_dir)
    ):
        # A code without its hash-bound archived report is not a terminal
        # physical fact. Treat it like a legacy marker and review it again.
        return 0
    legacy_failure = _legacy_deferred_physical_state_requires_review(state)
    if legacy_failure:
        # Compatibility migration: retain the previous counter in the attempt
        # archive, but give every pre-versioned terminal marker one base review
        # under the current outcome contract.  The launch writes attempt 1;
        # MAX=2 deliberately leaves exactly one infrastructure-recovery retry.
        return 0
    return attempt_count


def _mark_deferred_physical_recovery_exhausted(demo_dir: Path) -> None:
    state_path = _deferred_physical_worker_state_path(demo_dir)
    state = _read_json_object_best_effort(state_path)
    if (
        state.get("status") == "recovery_exhausted"
        and state.get("review_status") == PHYSICAL_REPLAY_REVIEW_REQUIRED
    ):
        return
    state.update(
        {
            "format": "handson.deferred_physical_replay_worker_state",
            "version": 1,
            "status": "recovery_exhausted",
            "pid": None,
            "terminal": True,
            "physical_replay_outcome_version": (
                PHYSICAL_REPLAY_OUTCOME_VERSION
            ),
            "physical_replay_outcome_code": (
                PHYSICAL_REPLAY_REVIEW_REQUIRED_OUTCOME_CODE
            ),
            "review_status": PHYSICAL_REPLAY_REVIEW_REQUIRED,
            "updated_at": _now_iso(),
            "recovery_exhausted_at": _now_iso(),
            "recovery_reason": (
                "automatic recovery limit reached; manual diagnosis required"
            ),
        }
    )
    _write_json_atomic(state_path, state)


def _live_deferred_physical_worker_coverage(
    demo_dirs: list[Path],
) -> set[Path]:
    demos = {demo.resolve() for demo in demo_dirs}
    if not demos:
        return set()
    candidate_pids: set[int] = set()
    for demo_dir in demos:
        state = _read_json_object_best_effort(
            _deferred_physical_worker_state_path(demo_dir)
        )
        pid = state.get("pid")
        if type(pid) is int and pid > 1:
            candidate_pids.add(pid)
    try:
        candidate_pids.update(
            int(path.name)
            for path in Path("/proc").iterdir()
            if path.name.isdigit()
        )
    except OSError:
        pass

    covered: set[Path] = set()
    demo_by_argument = {str(demo): demo for demo in demos}
    for pid in candidate_pids:
        command = _deferred_physical_worker_command(pid)
        if not command:
            continue
        command_arguments = set(command)
        covered.update(
            demo
            for argument, demo in demo_by_argument.items()
            if argument in command_arguments
        )
    return covered


def _ensure_deferred_physical_replay_certification(
    output_root: str | Path,
    *,
    exp_name: str,
    certification_runs: int,
    certification_timeout_seconds: int,
    session_id: str | None = None,
    source_recorder_pid: int | None = None,
    skip_live_source_recorders: bool = False,
    recovery_summary: dict[str, object] | None = None,
    allow_review_required_rerun: bool = False,
    manual_review_run_id: str | None = None,
    physx_v3_only: bool = False,
) -> int:
    """Recover orphan candidates while never duplicating a live worker."""

    pending_parent = _physical_replay_pending_parent(
        Path(output_root).expanduser().resolve()
    )
    pending_parent.mkdir(parents=True, exist_ok=True)
    lock_path = pending_parent / DEFERRED_PHYSICAL_REPLAY_LAUNCH_LOCK
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            recovered_bindings = _recover_physx_session_history_bindings(
                output_root,
                exp_name=exp_name,
                # Recovery is deliberately global for this task.  A new UI
                # launch has a new collection_session_id and must still seal
                # journals orphaned by an earlier workbench/supervisor crash.
                session_id=None,
            )
            candidates = _pending_deferred_physical_replay_candidates(
                output_root,
                exp_name=exp_name,
                session_id=None,
                physx_v3_only=physx_v3_only,
            )
            covered = _live_deferred_physical_worker_coverage(candidates)
            effective_source_pid = (
                os.getpid()
                if source_recorder_pid is None
                else int(source_recorder_pid)
            )
            source_live = {
                demo_dir
                for demo_dir in candidates
                if demo_dir not in covered
                if _recorded_source_recorder_is_live(demo_dir)
                and (
                    skip_live_source_recorders
                    or _recorded_source_recorder_pid(demo_dir)
                    != effective_source_pid
                )
            }
            uncovered = [
                demo_dir
                for demo_dir in candidates
                if demo_dir not in covered and demo_dir not in source_live
            ]
            legacy_review_required = [
                demo_dir
                for demo_dir in uncovered
                if _legacy_deferred_physical_state_requires_review(
                    _read_json_object_best_effort(
                        _deferred_physical_worker_state_path(demo_dir)
                    )
                )
            ]
            exhausted = [
                demo_dir
                for demo_dir in uncovered
                if _deferred_physical_launch_attempt_count(
                    demo_dir,
                    allow_review_required_rerun=allow_review_required_rerun,
                    manual_review_run_id=manual_review_run_id,
                )
                >= MAX_DEFERRED_PHYSICAL_REPLAY_LAUNCH_ATTEMPTS
            ]
            for demo_dir in exhausted:
                _mark_deferred_physical_recovery_exhausted(demo_dir)
            exhausted_set = set(exhausted)
            launchable = [
                demo_dir
                for demo_dir in uncovered
                if demo_dir not in exhausted_set
            ]
            launched = 0
            if launchable:
                by_session: dict[str, list[Path]] = {}
                for demo_dir in launchable:
                    session_key = _demo_collection_session_id(demo_dir)
                    # An empty legacy id cannot safely share hidden PhysX
                    # history with another candidate.  Keep it isolated so a
                    # recovery sweep never manufactures one mixed session.
                    if not session_key:
                        session_key = f"legacy:{demo_dir}"
                    by_session.setdefault(session_key, []).append(demo_dir)
                # PhysX/Isaac is a single heavy GPU session.  Launch one
                # collection session per recovery sweep; the caller polls
                # until it exits before asking for the next session.
                session_demos = next(iter(by_session.values()), [])
                if session_demos:
                    launch_kwargs = {
                        "exp_name": exp_name,
                        "certification_runs": certification_runs,
                        "certification_timeout_seconds": (
                            certification_timeout_seconds
                        ),
                        "source_recorder_pid": source_recorder_pid,
                    }
                    if allow_review_required_rerun:
                        launch_kwargs["allow_review_required_rerun"] = True
                    if manual_review_run_id:
                        launch_kwargs["manual_review_run_id"] = (
                            manual_review_run_id
                        )
                    launched += _launch_deferred_physical_replay_certification_batch(
                        session_demos,
                        **launch_kwargs,
                    )
            if recovery_summary is not None:
                recovery_summary.clear()
                recovery_summary.update(
                    {
                        "task": exp_name,
                        "candidate_count": len(candidates),
                        "launched_count": int(launched),
                        "live_worker_skipped_count": len(covered),
                        "live_source_skipped_count": len(source_live),
                        "recovery_exhausted_count": len(exhausted),
                        "legacy_review_migrated_count": len(
                            legacy_review_required
                        ),
                        "session_history_bindings_recovered_count": int(
                            recovered_bindings
                        ),
                    }
                )
            return int(launched)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _sha256_regular_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_review_history_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _archive_deferred_physical_review_attempt(
    demo_dir: Path,
    *,
    previous_state: Mapping[str, object],
) -> dict[str, object] | None:
    """Atomically preserve review JSON before a new attempt overwrites it."""

    state_path = _deferred_physical_worker_state_path(demo_dir)
    sources: list[tuple[Path, Path]] = [
        (state_path, Path("worker_state.json")),
        (
            demo_dir / FRESH_PROCESS_ACCEPTANCE_NAME,
            Path(FRESH_PROCESS_ACCEPTANCE_NAME),
        ),
        (
            demo_dir / PHYSICAL_REPLAY_REPORT_NAME,
            Path(PHYSICAL_REPLAY_REPORT_NAME),
        ),
    ]
    run_dir = demo_dir / PHYSICAL_REPLAY_RUN_DIRECTORY
    if run_dir.is_dir() and not run_dir.is_symlink():
        sources.extend(
            (run_path, Path(PHYSICAL_REPLAY_RUN_DIRECTORY) / run_path.name)
            for run_path in sorted(run_dir.glob("run_*.json"))
        )
    sources = [
        (source, relative)
        for source, relative in sources
        if source.is_file() and not source.is_symlink()
    ]
    if not sources:
        return None

    history_parent = (
        demo_dir.parent
        / PHYSICAL_REPLAY_REVIEW_HISTORY_DIRECTORY
        / demo_dir.name
    )
    history_parent.mkdir(parents=True, exist_ok=True)
    attempt_index = 1
    while (history_parent / f"attempt_{attempt_index:06d}").exists():
        attempt_index += 1
    target = history_parent / f"attempt_{attempt_index:06d}"
    temporary = Path(
        tempfile.mkdtemp(prefix=".attempt-tmp-", dir=str(history_parent))
    )
    archived_at = _now_iso()
    documents: list[dict[str, object]] = []
    try:
        for source, relative in sources:
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination, follow_symlinks=False)
            with destination.open("rb") as archived_stream:
                os.fsync(archived_stream.fileno())
            documents.append(
                {
                    "source": str(source),
                    "archive_path": relative.as_posix(),
                    "sha256": _sha256_regular_file(destination),
                    "size_bytes": destination.stat().st_size,
                }
            )
        manifest_path = temporary / "history_manifest.json"
        _write_json_atomic(
            manifest_path,
            {
                "format": PHYSICAL_REPLAY_REVIEW_HISTORY_FORMAT,
                "version": 1,
                "demo_name": demo_dir.name,
                "archived_at": archived_at,
                "reason": "review_requeue_before_overwrite",
                "legacy_review_migration": (
                    _legacy_deferred_physical_state_requires_review(
                        previous_state
                    )
                ),
                "previous_status": previous_state.get("status"),
                "previous_terminal": previous_state.get("terminal"),
                "previous_launch_attempt_count": previous_state.get(
                    "launch_attempt_count"
                ),
                "previous_physical_replay_outcome_code": previous_state.get(
                    "physical_replay_outcome_code"
                ),
                "documents": documents,
            },
        )
        with manifest_path.open("rb") as manifest_stream:
            os.fsync(manifest_stream.fileno())
        manifest_sha256 = _sha256_regular_file(manifest_path)
        for directory in sorted(
            {path.parent for path in temporary.rglob("*") if path.is_file()},
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _fsync_review_history_directory(directory)
        _fsync_review_history_directory(temporary)
        os.replace(temporary, target)
        _fsync_review_history_directory(history_parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "format": PHYSICAL_REPLAY_REVIEW_HISTORY_FORMAT,
        "version": 1,
        "archive_path": str(target),
        "history_manifest_sha256": manifest_sha256,
        "archived_at": archived_at,
        "document_count": len(documents),
    }


def _deferred_physical_replay_worker_script() -> Path:
    worker_name = "deferred_physical_replay_certification_worker.py"
    configured = str(
        os.environ.get(DEFERRED_PHYSICAL_REPLAY_WORKER_SCRIPT_ENV) or ""
    ).strip()
    if not configured:
        return Path(__file__).resolve().parent / worker_name

    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        raise RuntimeError(
            f"{DEFERRED_PHYSICAL_REPLAY_WORKER_SCRIPT_ENV} must be absolute"
        )
    if candidate.name != worker_name:
        raise RuntimeError(
            f"{DEFERRED_PHYSICAL_REPLAY_WORKER_SCRIPT_ENV} basename must be "
            f"{worker_name}"
        )
    cursor = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(
                f"{DEFERRED_PHYSICAL_REPLAY_WORKER_SCRIPT_ENV} must not "
                "traverse a symlink"
            )
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(
            f"{DEFERRED_PHYSICAL_REPLAY_WORKER_SCRIPT_ENV} is unavailable: "
            f"{candidate}"
        ) from exc
    if resolved.is_symlink() or not resolved.is_file() or resolved.name != worker_name:
        raise RuntimeError(
            f"{DEFERRED_PHYSICAL_REPLAY_WORKER_SCRIPT_ENV} must resolve to "
            f"a regular {worker_name} file"
        )
    return resolved


def _launch_deferred_physical_replay_certification_batch(
    demo_dirs: list[str] | list[Path],
    *,
    exp_name: str,
    certification_runs: int,
    certification_timeout_seconds: int,
    source_recorder_pid: int | None = None,
    allow_review_required_rerun: bool = False,
    manual_review_run_id: str | None = None,
) -> int:
    demos = sorted(
        {
            Path(demo).expanduser().resolve()
            for demo in demo_dirs
            if (Path(demo).expanduser().resolve() / "replay.zarr").is_dir()
        }
    )
    if not demos:
        return 0

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = demos[0].parent / ".physical_replay_certification_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    recorded_session_id = _demo_collection_session_id(demos[0])
    safe_session = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in (
            recorded_session_id
            or os.environ.get("HANDSON_COLLECTION_SESSION_ID", "")
            or exp_name
        )
    ).strip("_") or exp_name
    log_path = log_dir / f"{safe_session}-{timestamp}.log"
    worker_script = _deferred_physical_replay_worker_script()
    command = [
        sys.executable,
        str(worker_script),
        f"--exp_name={exp_name}",
        f"--certification-runs={max(1, int(certification_runs))}",
        (
            "--certification-timeout-seconds="
            f"{max(1, int(certification_timeout_seconds))}"
        ),
        f"--log_path={log_path}",
        *[str(demo) for demo in demos],
    ]

    env_vars = os.environ.copy()
    env_vars.pop(DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_ENV, None)
    resolved_source_pid = (
        os.getpid()
        if source_recorder_pid is None
        else int(source_recorder_pid)
    )
    if resolved_source_pid > 1:
        env_vars[DEFERRED_SOURCE_RECORDER_PID_ENV] = str(
            resolved_source_pid
        )
    else:
        env_vars.pop(DEFERRED_SOURCE_RECORDER_PID_ENV, None)
    pythonpath = env_vars.get("PYTHONPATH", "")
    pythonpath_entries = [str(_PACKAGE_ROOT), str(_PACKAGE_ROOT.parent)]
    pythonpath_entries.extend(part for part in pythonpath.split(os.pathsep) if part)
    env_vars["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath_entries))

    queued_at = _now_iso()
    previous_states = {
        demo_dir: _read_json_object_best_effort(
            _deferred_physical_worker_state_path(demo_dir)
        )
        for demo_dir in demos
    }
    launch_attempt_counts = {
        demo_dir: _deferred_physical_launch_attempt_count(
            demo_dir,
            allow_review_required_rerun=allow_review_required_rerun,
            manual_review_run_id=manual_review_run_id,
        )
        + 1
        for demo_dir in demos
    }
    review_archives = {
        demo_dir: _archive_deferred_physical_review_attempt(
            demo_dir,
            previous_state=previous_states[demo_dir],
        )
        for demo_dir in demos
    }

    def write_worker_state(
        status: str,
        *,
        pid: int | None,
        error: str = "",
    ) -> None:
        for demo_dir in demos:
            launch_attempt_count = launch_attempt_counts[demo_dir]
            previous_state = previous_states[demo_dir]
            legacy_review_migration = (
                _legacy_deferred_physical_state_requires_review(
                    previous_state
                )
            )
            payload: dict[str, object] = {
                "format": "handson.deferred_physical_replay_worker_state",
                "version": 1,
                "status": status,
                "pid": pid,
                "queued_at": queued_at,
                "last_launch_at": queued_at,
                "updated_at": _now_iso(),
                "exp_name": exp_name,
                "collection_session_id": recorded_session_id,
                "demo_dirs": [str(demo) for demo in demos],
                "command": command,
                "log_path": str(log_path),
                "launch_attempt_count": launch_attempt_count,
                "recovery_attempt_count": max(
                    0,
                    launch_attempt_count - 1,
                ),
                "terminal": False,
                "physical_replay_outcome_version": (
                    PHYSICAL_REPLAY_OUTCOME_VERSION
                ),
                "physical_replay_outcome_code": (
                    PHYSICAL_REPLAY_REVIEW_REQUIRED_OUTCOME_CODE
                ),
                "review_status": PHYSICAL_REPLAY_REVIEW_REQUIRED,
                "legacy_review_migration": legacy_review_migration,
                "previous_launch_attempt_count": previous_state.get(
                    "launch_attempt_count"
                ),
            }
            if manual_review_run_id:
                payload["manual_review_run_id"] = manual_review_run_id
            review_archive = review_archives[demo_dir]
            if review_archive is not None:
                payload["previous_review_archive"] = review_archive
            if error:
                payload["error"] = error
            _write_json_atomic(
                _deferred_physical_worker_state_path(demo_dir),
                payload,
            )

    write_worker_state("launching", pid=None)
    try:
        log_handle = log_path.open("ab")
        try:
            process = subprocess.Popen(
                command,
                cwd=str(_PACKAGE_ROOT.parent),
                env=env_vars,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_handle.close()
    except Exception as exc:
        write_worker_state(
            "launch_failed",
            pid=None,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    write_worker_state("running", pid=int(process.pid))

    print(
        "[PhysX post-collection certification] background worker "
        f"pid={process.pid} demos={len(demos)} log={log_path}",
        flush=True,
    )
    return len(demos)


def _uses_e01_ipc_replay_pipeline(exp_name: str, sim_backend: str) -> bool:
    """Return whether a saved garment trajectory can run IPC cloth replay."""
    return bool(
        exp_name in _E01_IPC_REPLAY_TASKS
        and str(sim_backend).strip().lower() in _E01_DEFERRED_POSTPROCESS_BACKENDS
    )


_B320_TERMINAL_REASON_LABELS_ZH = {
    "carton_contact_view_unavailable": "纸箱接触传感器不可用",
    "carton_contact_force_unreadable": "纸箱接触力无法读取",
    "carton_contact_points_unreadable": "纸箱接触点无法读取",
    "carton_side_force_exceeded": "纸箱侧面受力超限",
    "carton_tilt_exceeded": "纸箱倾角超限",
    "carton_dropped": "纸箱掉落",
    "carton_transport_support_lost": "搬运中双手支撑丢失",
    "time_limit": "达到任务时限",
    "unsuccessful_terminal": "任务未成功即进入终止状态",
}


def _terminal_flag(value: object) -> bool:
    """Read one scalar Gym flag without treating vectors as truthy."""

    try:
        array = np.asarray(value)
        return bool(array.item()) if array.size == 1 else False
    except (TypeError, ValueError):
        return False


def _terminal_json_value(value: object) -> object:
    """Convert a small terminal diagnostic value to strict JSON data."""

    if isinstance(value, Mapping):
        return {
            str(key): _terminal_json_value(member)
            for key, member in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_terminal_json_value(member) for member in value]
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _terminal_json_value(value.reshape(-1)[0])
        return [_terminal_json_value(member) for member in value.tolist()]
    if isinstance(value, np.generic):
        return _terminal_json_value(value.item())
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    return str(value)


def _b320_terminal_summary(
    task_id: str,
    *,
    done: object,
    truncated: object,
    info: object,
    episode_steps: int | None = None,
    episode_sim_time_seconds: float | None = None,
) -> dict[str, object] | None:
    """Build the exact B320 outcome shown before an automatic reset.

    This is diagnostic only: it consumes the environment's existing
    fail-closed flags and never promotes a failed or truncated trajectory to
    success.
    """

    normalized_task_id = str(task_id).strip().lower()
    # Most B320 runtime keys carry the family prefix, but BC005 keeps its
    # canonical public task name.  Include it explicitly so its physical
    # ``failure_reason`` is not degraded to a generic unsuccessful reset in
    # the collection journal.
    if (
        not normalized_task_id.startswith("b320_")
        and normalized_task_id not in {"loaded_tray_through_narrow_gate", "eval_comb06_season_and_stir"}
    ):
        return None
    episode_info = info if isinstance(info, Mapping) else {}
    failure_reason = str(
        episode_info.get("b320_failure_reason")
        or episode_info.get("failure_reason")
        or ""
    ).strip()
    truncated_reason = str(
        episode_info.get("truncated_reason") or ""
    ).strip()
    success = _terminal_flag(episode_info.get("succeed", False))
    failed = bool(
        _terminal_flag(episode_info.get("failed", False))
        or failure_reason
    )
    was_truncated = _terminal_flag(truncated)
    if success:
        outcome = "success"
        terminal_reason = "success"
    elif failed:
        outcome = "failure"
        terminal_reason = failure_reason or "unsuccessful_terminal"
    elif was_truncated:
        outcome = "truncated"
        terminal_reason = truncated_reason or "time_limit"
    else:
        outcome = "terminated"
        terminal_reason = "unsuccessful_terminal"

    summary: dict[str, object] = {
        "task_id": normalized_task_id,
        "outcome": outcome,
        "terminal_reason": terminal_reason,
        "failure_reason": failure_reason or None,
        "truncated_reason": truncated_reason or None,
        "terminated": _terminal_flag(done),
        "truncated": was_truncated,
        "succeed": success,
        "failed": failed,
        "time_limit_reached": _terminal_flag(
            episode_info.get("time_limit_reached", False)
        ),
    }
    if episode_steps is not None:
        summary["episode_steps"] = max(0, int(episode_steps))
    if episode_sim_time_seconds is not None:
        try:
            sim_seconds = float(episode_sim_time_seconds)
        except (TypeError, ValueError):
            sim_seconds = float("nan")
        if np.isfinite(sim_seconds):
            summary["episode_sim_time_seconds"] = max(0.0, sim_seconds)

    diagnostic_fields = (
        "b320_contact_available",
        "b320_contact_readable",
        "b320_contact_force_n",
        "b320_contact_peak_n",
        "b320_carton_contact_detail_available",
        "b320_carton_zone_valid",
        "b320_carton_zone_force_n",
        "b320_carton_transport_started",
        "b320_carton_transport_support_lost",
        "b320_pen_table_contact_available",
        "b320_pen_table_contact_readable",
        "b320_pen_table_contact_force_n",
        "b320_handover_completed",
        "bc011_pen_variant",
        "bc011_pen_tip_axis_local",
        "b320_success_counter",
        "b320_success_frames_required",
    )
    summary.update(
        (field, _terminal_json_value(episode_info[field]))
        for field in diagnostic_fields
        if field in episode_info
    )
    metrics = episode_info.get("b320_metrics")
    if isinstance(metrics, Mapping):
        summary["b320_metrics"] = _terminal_json_value(metrics)
    return summary


def _print_b320_terminal_summary(
    task_id: str,
    *,
    done: object,
    truncated: object,
    info: object,
    episode_steps: int | None = None,
    episode_sim_time_seconds: float | None = None,
) -> dict[str, object] | None:
    """Print and return one machine-readable B320 terminal diagnosis."""

    summary = _b320_terminal_summary(
        task_id,
        done=done,
        truncated=truncated,
        info=info,
        episode_steps=episode_steps,
        episode_sim_time_seconds=episode_sim_time_seconds,
    )
    if summary is not None:
        print(
            "[B320 episode] "
            + json.dumps(
                summary,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ),
            flush=True,
        )
    return summary


def _b320_terminal_reset_hold_seconds(
    environment: Mapping[str, str] | None = None,
) -> float:
    """Return a bounded wall-only terminal-frame hold for human operators."""

    source = os.environ if environment is None else environment
    raw_value = str(
        source.get(B320_TERMINAL_RESET_HOLD_SECONDS_ENV, "")
    ).strip()
    try:
        seconds = (
            float(raw_value)
            if raw_value
            else B320_TERMINAL_RESET_HOLD_SECONDS_DEFAULT
        )
    except (TypeError, ValueError):
        seconds = B320_TERMINAL_RESET_HOLD_SECONDS_DEFAULT
    if not np.isfinite(seconds):
        seconds = B320_TERMINAL_RESET_HOLD_SECONDS_DEFAULT
    return float(
        np.clip(seconds, 0.0, B320_TERMINAL_RESET_HOLD_SECONDS_MAX)
    )


def _b320_terminal_status_detail(
    summary: Mapping[str, object],
    *,
    reset_complete: bool,
    hold_seconds: float = 0.0,
) -> str:
    """Render a concise Chinese UI status without hiding the exact reason."""

    reason = str(summary.get("terminal_reason") or "unsuccessful_terminal")
    reason_label = _B320_TERMINAL_REASON_LABELS_ZH.get(reason, reason)
    outcome = str(summary.get("outcome") or "terminated")
    if outcome == "failure":
        prefix = (
            f"上一条轨迹已判定为物理失败（{reason_label}，非任务成功）"
        )
    elif outcome == "truncated":
        prefix = f"上一条轨迹已截断（{reason_label}，非任务成功）"
    else:
        prefix = f"上一条轨迹未成功即终止（{reason_label}）"

    facts: list[str] = []
    peak_force = summary.get("b320_contact_peak_n")
    if isinstance(peak_force, Mapping):
        left = peak_force.get("left")
        right = peak_force.get("right")
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            facts.append(
                f"峰值力 左 {float(left):.2f} N / 右 {float(right):.2f} N"
            )
    if isinstance(summary.get("episode_steps"), int):
        facts.append(f"step={summary['episode_steps']}")
    sim_seconds = summary.get("episode_sim_time_seconds")
    if isinstance(sim_seconds, (int, float)):
        facts.append(f"sim_time={float(sim_seconds):.3f} s")
    if facts:
        prefix += "；" + "，".join(facts)
    if reset_complete:
        return (
            prefix
            + "。该轨迹已丢弃，场景已完成复位，可继续采集。"
        )
    if hold_seconds > 0.0:
        return prefix + f"。终态画面保留 {hold_seconds:.1f} 秒后复位。"
    return prefix + "。正在复位。"


def _hold_b320_unsuccessful_terminal_frame(
    task_id: str,
    summary: Mapping[str, object] | None,
    *,
    seconds: float | None = None,
) -> float:
    """Keep the final image visible briefly without advancing simulation."""

    if (
        (not str(task_id).strip().lower().startswith("b320_")
         and str(task_id).strip().lower() != "eval_comb06_season_and_stir")
        or not isinstance(summary, Mapping)
        or summary.get("outcome") == "success"
    ):
        return 0.0
    try:
        duration = (
            _b320_terminal_reset_hold_seconds()
            if seconds is None
            else float(seconds)
        )
    except (TypeError, ValueError):
        duration = B320_TERMINAL_RESET_HOLD_SECONDS_DEFAULT
    if not np.isfinite(duration):
        duration = B320_TERMINAL_RESET_HOLD_SECONDS_DEFAULT
    duration = float(
        np.clip(duration, 0.0, B320_TERMINAL_RESET_HOLD_SECONDS_MAX)
    )
    if duration > 0.0:
        # This is deliberately wall time only.  No simulator step occurs, so
        # the recorded terminal state and replay determinism are unchanged.
        time.sleep(duration)
    return duration


def _print_p004_terminal_summary(
    task_id: str,
    *,
    done: object,
    truncated: object,
    info: object,
    episode_steps: int | None = None,
) -> None:
    """Print P0-04's physical failure reason before an automatic reset."""

    if str(task_id).strip().lower() != "p004_hold_spring_clamp_open_insert_card":
        return
    episode_info = info if isinstance(info, Mapping) else {}
    succeed = _terminal_flag(episode_info.get("succeed", False))
    failed = _terminal_flag(episode_info.get("failed", False))
    failure_reason = str(
        episode_info.get("p004_failure_reason")
        or episode_info.get("failure_reason")
        or ""
    ).strip()
    stage = str(
        episode_info.get("p004_stage_name")
        or episode_info.get("stage_name")
        or "unknown"
    )
    if succeed:
        outcome = "success"
    elif failed or failure_reason:
        outcome = "failure"
    elif _terminal_flag(truncated):
        outcome = "time_limit"
    else:
        outcome = "terminated"
    summary: dict[str, object] = {
        "outcome": outcome,
        "terminated": _terminal_flag(done),
        "truncated": _terminal_flag(truncated),
        "succeed": succeed,
        "failed": failed,
        "failure_reason": failure_reason or None,
        "stage": stage,
    }
    if episode_steps is not None:
        summary["episode_steps"] = max(0, int(episode_steps))
    for field in (
        "p004_clamp_held",
        "p004_clamp_laid_down",
        "p004_left_clamp_motion_lock_active",
        "p004_insertion_depth_m",
        "p004_card_between_pads",
        "p004_card_fixed_jaw_contact",
        "p004_card_moving_jaw_contact",
        "p004_right_card_contact",
        "p004_success_hold",
    ):
        if field in episode_info:
            summary[field] = _terminal_json_value(episode_info[field])
    print(
        "[P0-04 episode] "
        + json.dumps(summary, ensure_ascii=False, allow_nan=False, sort_keys=True),
        flush=True,
    )


def _print_e01_terminal_summary(
    task_id: str,
    *,
    done: object,
    truncated: object,
    info: object,
) -> None:
    """Print one best-effort E01 terminal summary before automatic reset."""
    if task_id != "e01_fold_garment":
        return

    episode_info = info if isinstance(info, Mapping) else {}

    def _flag(value: object) -> bool:
        try:
            array = np.asarray(value)
            return bool(array.item()) if array.size == 1 else False
        except (TypeError, ValueError):
            return False

    failure_reason = episode_info.get("failure_reason")
    garment_failure_reason = episode_info.get("garment_failure_reason")
    success = _flag(episode_info.get("succeed", False))
    failed = _flag(episode_info.get("failed", False)) or any(
        reason is not None and str(reason) != ""
        for reason in (failure_reason, garment_failure_reason)
    )
    was_truncated = _flag(truncated)
    if success:
        outcome = "success"
    elif failed:
        outcome = "failure"
    elif was_truncated:
        outcome = "truncated"
    else:
        outcome = "terminated"

    summary: dict[str, object] = {
        "outcome": outcome,
        "terminated": _flag(done),
        "truncated": was_truncated,
        "succeed": success,
        "failed": failed,
        "failure_reason": failure_reason,
        "garment_failure_reason": garment_failure_reason,
    }
    optional_fields = (
        "garment_readback_max_nodal_speed",
        "garment_max_control_node_displacement",
        "palm_force_saturated_right",
        "palm_force_saturated_left",
        "palm_tracking_translation_error_m_right",
        "palm_tracking_translation_error_m_left",
        "palm_tracking_rotation_error_deg_right",
        "palm_tracking_rotation_error_deg_left",
    )
    summary.update(
        (field, episode_info[field])
        for field in optional_fields
        if field in episode_info
    )
    print(
        "[E01 episode] "
        + json.dumps(summary, ensure_ascii=False, default=str, sort_keys=True),
        flush=True,
    )


def _should_auto_generate_c01_arm_video(
    exp_name: str,
    *,
    enabled: bool,
    env_value: str,
) -> bool:
    """Return whether a saved hands-only demo needs the arm completion pass."""
    return bool(
        exp_name in _C01_ARM_COMPLETION_TASKS
        and enabled
        and env_value.strip().lower() not in {"0", "false", "no", "off"}
    )


def _should_auto_generate_plan_arm_video(
    exp_name: str,
    *,
    enabled: bool,
    env_value: str,
) -> bool:
    """Return whether an online single-arm Plan recording needs arm rendering."""
    return bool(
        exp_name.startswith("plan")
        and exp_name not in _C01_ARM_COMPLETION_TASKS
        and enabled
        and env_value.strip().lower() not in {"0", "false", "no", "off"}
    )


def _offline_arm_render_declaration(
    exp_name: str,
    *,
    sim_backend: str,
    manual_only: bool,
    c01_enabled: bool,
    plan_enabled: bool,
    c01_env_value: str,
    plan_env_value: str,
) -> dict[str, object]:
    """Describe C01/Plan offline arm work before certification hashes it."""

    if not manual_only:
        return {}
    deferred = bool(
        _should_auto_generate_c01_arm_video(
            exp_name,
            enabled=(
                c01_enabled
                and str(sim_backend).strip().lower() == "mujoco"
            ),
            env_value=c01_env_value,
        )
        or _should_auto_generate_plan_arm_video(
            exp_name,
            enabled=plan_enabled,
            env_value=plan_env_value,
        )
    )
    return {
        # Capture-time intent is immutable.  Render completion belongs in a
        # dedicated operational receipt; never rewrite replay_manifest.json.
        "offline_arm_render_deferred": deferred,
        "offline_arm_render_status": (
            "pending" if deferred else "not_required"
        ),
    }


def _should_auto_compute_e01_arm_plan(
    exp_name: str,
    *,
    sim_backend: str,
    env_value: str,
) -> bool:
    """Complete detached-hand E01 recordings only after task success."""
    return bool(
        _uses_e01_detached_hand_pipeline(exp_name, sim_backend)
        and env_value.strip().lower() not in {"0", "false", "no", "off"}
    )


def _should_auto_replay_e01_ipc(
    exp_name: str,
    *,
    sim_backend: str,
    env_value: str,
) -> bool:
    """Run quality-first cloth replay for saved E/F garment episodes."""
    return bool(
        _uses_e01_ipc_replay_pipeline(exp_name, sim_backend)
        and env_value.strip().lower() not in {"0", "false", "no", "off"}
    )


def _e01_postprocess_requested(
    exp_name: str,
    *,
    sim_backend: str,
    ipc_env_value: str,
    arm_env_value: str,
) -> bool:
    return bool(
        _should_auto_replay_e01_ipc(
            exp_name,
            sim_backend=sim_backend,
            env_value=ipc_env_value,
        )
        or _should_auto_compute_e01_arm_plan(
            exp_name,
            sim_backend=sim_backend,
            env_value=arm_env_value,
        )
    )


def _e01_postprocess_requested_stages(
    exp_name: str,
    *,
    sim_backend: str,
    ipc_env_value: str,
    arm_env_value: str,
) -> list[str]:
    stages: list[str] = []
    if _should_auto_replay_e01_ipc(
        exp_name,
        sim_backend=sim_backend,
        env_value=ipc_env_value,
    ):
        stages.extend(["ipc_replay", "ipc_render"])
    if _should_auto_compute_e01_arm_plan(
        exp_name,
        sim_backend=sim_backend,
        env_value=arm_env_value,
    ):
        stages.append("arm_ik")
    return stages


def _should_autostart_e01_postprocess(
    exp_name: str,
    *,
    sim_backend: str,
    ipc_env_value: str,
    arm_env_value: str,
) -> bool:
    """Keep E/F postprocessing discoverable but operator-triggered."""

    return bool(
        not _postprocess_manual_only_enabled()
        and _e01_postprocess_requested(
            exp_name,
            sim_backend=sim_backend,
            ipc_env_value=ipc_env_value,
            arm_env_value=arm_env_value,
        )
    )


def _e01_postprocess_background_default(exp_name: str) -> str:
    """Keep F01 collection responsive; keep established E01 behavior unchanged."""
    return "1" if exp_name == _F01_COTTON_CLOTH_MVP_TASK else "0"


def _should_background_e01_postprocess(
    exp_name: str,
    *,
    sim_backend: str,
    env_value: str,
    ipc_env_value: str,
    arm_env_value: str,
) -> bool:
    return bool(
        _e01_postprocess_requested(
            exp_name,
            sim_backend=sim_backend,
            ipc_env_value=ipc_env_value,
            arm_env_value=arm_env_value,
        )
        and env_value.strip().lower() not in {"0", "false", "no", "off"}
    )


def _e01_ipc_replay_default(exp_name: str, env_key: str, fallback: str) -> str:
    configured = os.environ.get(env_key, "")
    if str(configured).strip():
        return str(configured)
    if exp_name == _F01_COTTON_CLOTH_MVP_TASK:
        return _F01_IPC_REPLAY_DEFAULTS.get(env_key, fallback)
    return fallback


def _e01_offline_arm_python(*, env_value: str, fallback: str) -> str:
    """Use the MuJoCo/Plan runtime, not Genesis' isolated interpreter."""
    configured = str(env_value).strip()
    return configured or str(fallback)


def _e01_offline_genesis_python(*, env_value: str, fallback: str) -> str:
    """Resolve the isolated Genesis/libuipc interpreter for cloth replay."""
    configured = str(env_value).strip()
    return configured or str(fallback)


def _e01_ipc_subprocess_environment(
    base_env: Mapping[str, str],
    *,
    exp_name: str = "e01_fold_garment",
) -> dict[str, str]:
    """Give IPC its CUDA-12 libraries without polluting the live Isaac runtime."""
    env = dict(base_env)
    if exp_name == _F01_COTTON_CLOTH_MVP_TASK:
        for key, value in _F01_IPC_ENV_DEFAULTS.items():
            env.setdefault(key, value)
    cuda12_paths = [
        entry
        for entry in env.get("HANDSON_GENESIS_CUDA12_LIBRARY_PATH", "").split(
            os.pathsep
        )
        if entry
    ]
    inherited_paths = [
        entry for entry in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if entry
    ]
    if cuda12_paths:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            dict.fromkeys([*cuda12_paths, *inherited_paths])
        )
    return env


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


def _recording_spool_video_crf() -> int:
    """Return the configured H.264 CRF used by the episode video spool."""
    raw = os.environ.get("HANDSON_RECORDING_SPOOL_VIDEO_CRF", "21").strip()
    try:
        crf = int(raw)
    except ValueError as exc:
        raise ValueError(
            "HANDSON_RECORDING_SPOOL_VIDEO_CRF must be an integer from 0 "
            f"through 51, got {raw!r}."
        ) from exc
    if not 0 <= crf <= 51:
        raise ValueError(
            "HANDSON_RECORDING_SPOOL_VIDEO_CRF must be from 0 through 51, "
            f"got {crf}."
        )
    return crf


def _recording_spool_queue_steps() -> int:
    """Bound asynchronous RGB work so encoding can never recreate the OOM."""
    raw = os.environ.get("HANDSON_RECORDING_SPOOL_QUEUE_STEPS", "2").strip()
    try:
        steps = int(raw)
    except ValueError as exc:
        raise ValueError(
            "HANDSON_RECORDING_SPOOL_QUEUE_STEPS must be an integer from 1 "
            f"through 8, got {raw!r}."
        ) from exc
    if not 1 <= steps <= 8:
        raise ValueError(
            "HANDSON_RECORDING_SPOOL_QUEUE_STEPS must be from 1 through 8, "
            f"got {steps}."
        )
    return steps


def _recording_spool_parent(base_out: Path) -> Path:
    """Choose a spool parent on the configured output disk, never /tmp by default."""
    configured = os.environ.get("HANDSON_RECORDING_SPOOL_DIR", "").strip()
    if configured:
        parent = Path(configured).expanduser()
        if not parent.is_absolute():
            parent = Path(base_out) / parent
        return parent
    return Path(base_out) / ".recording_spool"


def _recording_staging_parent(base_out: Path) -> Path:
    """Keep incomplete demos on the output filesystem in a UI-ignored folder."""
    output_root = Path(base_out).expanduser().resolve()
    return output_root / ".recording_staging"


def _physical_replay_pending_parent(base_out: Path) -> Path:
    """Return the hidden same-filesystem queue for uncertified candidates."""

    return Path(base_out).expanduser().resolve() / (
        PHYSICAL_REPLAY_PENDING_DIRECTORY
    )


def _physical_replay_session_history_path(
    base_out: Path,
    session_id: str,
) -> Path:
    """Return a collision-resistant hidden journal path for one UI session."""

    raw_session_id = str(session_id).strip()
    if not raw_session_id:
        raise ValueError("PhysX session history requires a collection session id.")
    readable = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in raw_session_id
    ).strip("_")[:64] or "session"
    suffix = hashlib.sha256(raw_session_id.encode("utf-8")).hexdigest()[:12]
    return (
        _physical_replay_pending_parent(base_out)
        / PHYSICAL_REPLAY_SESSION_HISTORY_DIRECTORY
        / f"{readable}-{suffix}"
    )


def _bind_physx_session_history_to_demos(
    demo_dirs: list[Path],
    *,
    output_root: Path,
    journal: PhysxSessionHistoryJournal,
    finalization: Mapping[str, object],
) -> None:
    """Bind every accepted demo manifest to the sealed external journal."""

    history_identity = finalization.get("history_identity")
    accepted_rows = finalization.get("accepted_demos")
    if not isinstance(history_identity, Mapping) or not isinstance(
        accepted_rows,
        list,
    ):
        raise ValueError("PhysX session finalization is missing identity metadata.")
    accepted_by_demo = {
        str(row.get("demo_id")): row
        for row in accepted_rows
        if isinstance(row, Mapping) and isinstance(row.get("demo_id"), str)
    }
    root = Path(output_root).expanduser().resolve()
    history_relative_path = journal.path.relative_to(root).as_posix()
    history_sha256 = history_identity.get("sha256")
    event_count = finalization.get("event_count")
    if not isinstance(history_sha256, str) or len(history_sha256) != 64:
        raise ValueError("PhysX session history has no SHA-256 identity.")
    if type(event_count) is not int or event_count < 0:
        raise ValueError("PhysX session history has an invalid event count.")

    for raw_demo_dir in demo_dirs:
        demo_dir = Path(raw_demo_dir).expanduser().resolve()
        accepted = accepted_by_demo.get(demo_dir.name)
        if accepted is None:
            raise ValueError(
                f"Accepted demo is absent from PhysX session history: {demo_dir.name}"
            )
        binding = {
            "physical_replay_certification_scope": (
                ORDERED_SESSION_PHYSICAL_REPLAY_CERTIFICATION_SCOPE
            ),
            "physical_replay_session_history_pending": False,
            "physical_replay_session_history_id": journal.session_id,
            "physical_replay_session_history_root_relative_path": (
                history_relative_path
            ),
            "physical_replay_session_history_sha256": history_sha256,
            "physical_replay_session_history_event_count": event_count,
            "physical_replay_session_history_accepted_event_ordinal": (
                accepted.get("event_ordinal")
            ),
            "physical_replay_session_history_certified_through_event_ordinal": (
                finalization.get("certified_through_event_ordinal")
            ),
        }
        manifest_path = demo_dir / "replay_manifest.json"
        manifest = _read_json_object_best_effort(manifest_path)
        recording = manifest.get("recording")
        if not isinstance(recording, Mapping):
            raise ValueError(f"Replay manifest has no recording object: {demo_dir}")
        manifest["recording"] = {**dict(recording), **binding}
        _write_json_atomic(manifest_path, manifest)

        metadata_path = demo_dir / "collection_meta.json"
        metadata = _read_json_object_best_effort(metadata_path)
        if not metadata:
            raise ValueError(f"Collection metadata is unavailable: {demo_dir}")
        metadata.update(binding)
        _write_json_atomic(metadata_path, metadata)


_RGB_OBSERVATION_KEY_HINTS = frozenset(
    {
        "main",
        "head_main",
        "front",
        "random_camera",
        "ego",
        "ego_left",
        "ego_right",
        "wrist",
        "wrist_left",
        "wrist_right",
        "left_wrist",
        "right_wrist",
        "handcam_rgb",
        "handcam_rgb_left",
        "handcam_rgb_right",
        "wrist_fisheye_left",
        "wrist_fisheye_right",
    }
)


def _looks_like_rgb_frame(value, *, min_side: int = 32) -> bool:
    """Identify camera-shaped RGB(A) arrays without catching force telemetry."""
    if not isinstance(value, np.ndarray):
        return False
    array = np.asarray(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    return bool(
        array.ndim == 3
        and array.shape[-1] in (3, 4)
        and array.shape[0] >= min_side
        and array.shape[1] >= min_side
    )


@dataclass(frozen=True)
class SpooledFrameRef:
    """Small lazy reference to one depth frame outside the trajectory heap."""

    path: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    offset: int = 0

    def load(self) -> np.ndarray:
        if self.kind == "depth_raw":
            dtype = np.dtype(self.dtype)
            byte_count = int(np.prod(self.shape, dtype=np.int64)) * dtype.itemsize
            with Path(self.path).open("rb") as stream:
                stream.seek(self.offset)
                payload = stream.read(byte_count)
            if len(payload) != byte_count:
                raise ValueError(
                    f"Spooled depth frame {self.path} is truncated at byte "
                    f"offset {self.offset}."
                )
            array = np.frombuffer(payload, dtype=dtype).reshape(self.shape)
        else:
            raise ValueError(f"Unknown spooled frame kind {self.kind!r}.")

        if tuple(array.shape) != tuple(self.shape):
            raise ValueError(
                f"Spooled frame {self.path} decoded as {array.shape}, expected "
                f"{self.shape}."
            )
        expected_dtype = np.dtype(self.dtype)
        if array.dtype != expected_dtype:
            array = array.astype(expected_dtype, copy=False)
        return array

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        array = self.load()
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        if copy:
            array = array.copy()
        return array

    def __deepcopy__(self, memo):
        # The reference is immutable; copying it must not decode or duplicate a frame.
        return self


class RecordingSpool:
    """Episode-scoped recorder spool with optional online H.264 capture.

    The default parent lives under ``--out_dir`` so a machine where ``/tmp``
    is tmpfs cannot silently move camera RAM pressure into another memory-backed
    filesystem. ``HANDSON_RECORDING_SPOOL_DIR`` can point at a different data
    disk. When ``encode_videos`` is true, RGB is encoded exactly once into the
    three canonical MP4 streams; ``HANDSON_RECORDING_SPOOL_VIDEO_CRF`` controls
    their quality. Camera-postrender episodes set it false and record only the
    synchronized low-dimensional/physics trajectory online.
    """

    def __init__(
        self,
        base_out: Path,
        *,
        video_fps: int,
        encode_videos: bool = True,
    ):
        self.video_crf = _recording_spool_video_crf()
        self.queue_steps = _recording_spool_queue_steps()
        self.video_fps = int(video_fps)
        self.encode_videos = bool(encode_videos)
        if self.video_fps <= 0:
            raise ValueError(f"video_fps must be positive, got {video_fps}.")
        self._shared_parent = bool(
            os.environ.get("HANDSON_RECORDING_SPOOL_DIR", "").strip()
        )
        self.parent = _recording_spool_parent(Path(base_out))
        self.parent.mkdir(parents=True, exist_ok=True)
        self.root = Path(
            tempfile.mkdtemp(
                prefix=f"episode-{os.getpid()}-",
                dir=str(self.parent),
            )
        )
        self._writers = {}
        self._writer_executors: dict[str, ThreadPoolExecutor] = {}
        self._pending_video_steps: deque[dict[str, Future]] = deque()
        self._encoder_backends = {}
        self._camera_sources = {}
        self._first_frames = {}
        self._frame_counts = {key: 0 for key in CANONICAL_CAMERA_KEYS}
        self._depth_paths = {}
        self._depth_offsets = {}
        self._observation_count = 0
        self._depth_step_count = 0
        self._published = False
        self._closed = False

    @property
    def frame_count(self) -> int:
        return self._observation_count

    @property
    def depth_frame_count(self) -> int:
        return self._depth_step_count

    def _start_video_writers(
        self,
        frames: Mapping[str, np.ndarray],
        sources: Mapping[str, str],
    ) -> None:
        for canonical_key in CANONICAL_CAMERA_KEYS:
            writer = self._new_video_writer()
            requested_codec = str(getattr(writer, "codec", "")).lower()
            output_path = self.root / f"{canonical_key}.mp4"
            try:
                writer.start(str(output_path))
                # NVENC commonly passes codec discovery/add_stream and only
                # reports a driver or session error at the first encode. Do a
                # synchronous first-frame preflight before accepting the
                # writer into the bounded asynchronous queue.
                writer.write_frame(frames[canonical_key])
            except Exception as gpu_exc:
                self._discard_video_writer(writer, output_path)
                can_retry_cpu = (
                    requested_codec != "h264"
                    and str(getattr(writer, "fallback_codec", "")).lower()
                    == "h264"
                )
                if not can_retry_cpu:
                    raise
                print(
                    f"[Recorder] {canonical_key} encoder {requested_codec} "
                    f"failed first-frame preflight ({gpu_exc}); retrying "
                    "with CPU H.264.",
                    flush=True,
                )
                writer = self._new_video_writer(force_cpu=True)
                try:
                    writer.start(str(output_path))
                    writer.write_frame(frames[canonical_key])
                except Exception:
                    self._discard_video_writer(writer, output_path)
                    raise

            self._writers[canonical_key] = writer
            self._writer_executors[canonical_key] = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"handson-{canonical_key}",
            )
            self._camera_sources[canonical_key] = str(sources[canonical_key])
            self._first_frames[canonical_key] = frames[canonical_key].copy()
            stream = getattr(writer, "_stream", None)
            codec_context = getattr(stream, "codec_context", None)
            self._encoder_backends[canonical_key] = str(
                getattr(codec_context, "name", None)
                or getattr(writer, "codec", "unknown")
            )
        print(
            "[Recorder] Video spool encoders: "
            + ", ".join(
                f"{key}={self._encoder_backends[key]}"
                for key in CANONICAL_CAMERA_KEYS
            )
            + f"; bounded_queue={self.queue_steps} step(s)",
            flush=True,
        )

    def _new_video_writer(self, *, force_cpu: bool = False):
        writer = Mp4VideoWriter.create_h264(
            fps=self.video_fps,
            codec="h264",
            input_pix_fmt="rgb24",
            crf=self.video_crf,
            thread_type="FRAME",
            thread_count=2,
            encoder="cpu" if force_cpu else None,
        )
        # The collection machine may lack NVENC even when it has an NVIDIA
        # GPU. Keep CPU encoding in its lowest-latency mode. Apply the same
        # options to the runtime NVENC fallback configured by the writer.
        cpu_options = {
            "preset": os.environ.get(
                "HANDSON_RECORDING_SPOOL_CPU_PRESET",
                "ultrafast",
            ),
            "complexity": "low",
            "usage": "camera_video_realtime",
        }
        if str(getattr(writer, "codec", "")).lower() == "h264":
            for option, value in cpu_options.items():
                writer.codec_options.setdefault(option, value)
        if str(getattr(writer, "fallback_codec", "")).lower() == "h264":
            for option, value in cpu_options.items():
                writer.fallback_codec_options.setdefault(option, value)
        return writer

    @staticmethod
    def _discard_video_writer(writer, output_path: Path) -> None:
        try:
            writer.stop()
        except Exception:
            # The same encoder failure can be raised again while flushing.
            # Best-effort close the PyAV container before replacing its file.
            container = getattr(writer, "_container", None)
            try:
                if container is not None:
                    container.close()
            except Exception:
                pass
            writer._container = None
            writer._stream = None
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass

    def _drain_oldest_video_step(self) -> None:
        futures = self._pending_video_steps.popleft()
        errors = []
        for canonical_key, future in futures.items():
            try:
                future.result()
            except Exception as exc:
                errors.append((canonical_key, exc))
        if errors:
            canonical_key, exc = errors[0]
            raise RuntimeError(
                f"Failed to encode spooled {canonical_key} frame: {exc}"
            ) from exc

    def _drain_ready_video_steps(self) -> None:
        while self._pending_video_steps and all(
            future.done()
            for future in self._pending_video_steps[0].values()
        ):
            self._drain_oldest_video_step()

    def append_observation(self, observation: Mapping, *, env=None) -> None:
        """Validate and enqueue one complete canonical RGB frame bundle."""
        if self._closed:
            raise RuntimeError("Recording spool is already closed.")
        if self._published:
            raise RuntimeError("Recording spool videos were already published.")
        if not self.encode_videos:
            self._observation_count += 1
            return

        step_sources = select_canonical_camera_sources(observation, env=env)
        sources: dict[str, str] = {}
        frames: dict[str, np.ndarray] = {}
        missing = []
        for canonical_key in CANONICAL_CAMERA_KEYS:
            source_key = self._camera_sources.get(canonical_key)
            if source_key is None:
                source_key = step_sources.get(canonical_key)
            if source_key is None:
                missing.append(f"{canonical_key}: no source")
                continue

            image = observation_image(observation, source_key)
            if image is None:
                missing.append(f"{canonical_key}: source {source_key!r} missing")
                continue
            frame = _safe_squeeze_image(image)
            if frame.ndim == 3 and frame.shape[-1] == 4:
                frame = frame[..., :3]
            frame = postprocess_canonical_image(
                frame,
                canonical_key,
                source_key,
            )
            frame_filter = getattr(
                getattr(env, "unwrapped", env),
                "postprocess_recording_frame",
                None,
            )
            if callable(frame_filter):
                frame = frame_filter(
                    frame,
                    canonical_key=canonical_key,
                    source_key=source_key,
                )
            frame = _safe_squeeze_image(frame)
            if frame.ndim == 3 and frame.shape[-1] == 4:
                frame = frame[..., :3]
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(
                    f"Canonical camera {canonical_key!r} produced invalid RGB "
                    f"shape {frame.shape}."
                )
            frame = np.ascontiguousarray(frame)
            sources[canonical_key] = source_key
            frames[canonical_key] = frame

        if missing:
            raise RuntimeError(
                "Incomplete canonical camera bundle at trajectory step "
                f"{self._observation_count}: " + "; ".join(missing)
            )
        started_now = not self._writers
        if started_now:
            self._start_video_writers(frames, sources)

        if not started_now:
            self._drain_ready_video_steps()
            while len(self._pending_video_steps) >= self.queue_steps:
                self._drain_oldest_video_step()

            pending: dict[str, Future] = {}
            for canonical_key in CANONICAL_CAMERA_KEYS:
                # Observations may reuse their backing render buffers on the
                # next environment step, so each bounded asynchronous slot
                # owns a copy.
                owned_frame = frames[canonical_key].copy()
                pending[canonical_key] = self._writer_executors[
                    canonical_key
                ].submit(self._writers[canonical_key].write_frame, owned_frame)
            self._pending_video_steps.append(pending)
        for canonical_key in CANONICAL_CAMERA_KEYS:
            self._frame_counts[canonical_key] += 1
        self._observation_count += 1

    def spool_depth(self, frame, *, stream_key: str = "depth") -> SpooledFrameRef:
        depth = np.ascontiguousarray(frame, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"Depth spool input must be HxW, got {depth.shape}.")
        safe_key = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in str(stream_key)
        ) or "depth"
        path = self._depth_paths.setdefault(
            str(stream_key),
            self.root / f"{safe_key}_depth.raw",
        )
        offset = self._depth_offsets.get(str(stream_key), 0)
        payload = depth.tobytes(order="C")
        with path.open("ab") as stream:
            stream.write(payload)
        self._depth_offsets[str(stream_key)] = offset + len(payload)
        return SpooledFrameRef(
            path=str(path),
            kind="depth_raw",
            shape=tuple(depth.shape),
            dtype=str(depth.dtype),
            offset=offset,
        )

    def compact_observation(
        self,
        observation: Mapping,
        *,
        protected_keys=(),
    ) -> dict:
        """Return low-dimensional data with all ordinary RGB arrays removed."""
        protected = {str(key) for key in protected_keys} | {"state", "low_dim"}
        compact = {}
        for key, value in observation.items():
            if key == "images" and isinstance(value, Mapping):
                non_images = {
                    image_key: image_value
                    for image_key, image_value in value.items()
                    if not _looks_like_rgb_frame(image_value, min_side=1)
                }
                if non_images:
                    compact[key] = non_images
            elif (
                str(key) not in protected
                and _looks_like_rgb_frame(
                    value,
                    min_side=1
                    if str(key) in _RGB_OBSERVATION_KEY_HINTS
                    else 8,
                )
            ):
                continue
            else:
                compact[key] = value
        return compact

    def spool_depth_mapping(self, frames: Mapping) -> dict:
        missing = [
            key
            for key in CANONICAL_CAMERA_KEYS
            if key not in frames or frames[key] is None
        ]
        if missing:
            raise RuntimeError(
                "Incomplete canonical depth bundle at trajectory step "
                f"{self._depth_step_count}: missing {missing}"
            )
        result = {
            key: self.spool_depth(frames[key], stream_key=str(key))
            for key in CANONICAL_CAMERA_KEYS
        }
        self._depth_step_count += 1
        return result

    def _stop_video_writers(self, *, strict: bool) -> None:
        errors = []
        while self._pending_video_steps:
            try:
                self._drain_oldest_video_step()
            except Exception as exc:
                errors.append(("queued frame", exc))
        executors = self._writer_executors
        self._writer_executors = {}
        for canonical_key, executor in executors.items():
            try:
                executor.shutdown(wait=True, cancel_futures=False)
            except Exception as exc:
                errors.append((f"{canonical_key} executor", exc))

        for canonical_key, writer in self._writers.items():
            try:
                writer.stop()
            except Exception as exc:
                errors.append((canonical_key, exc))
                self._discard_video_writer(
                    writer,
                    self.root / f"{canonical_key}.mp4",
                )
        self._writers = {}
        if strict and errors:
            canonical_key, exc = errors[0]
            raise RuntimeError(
                f"Failed to finalize spooled {canonical_key} video: {exc}"
            ) from exc

    def _abort_video_writers(self) -> None:
        # A reset discards this episode, so flushing delayed encoder frames is
        # wasted work and can stall NVENC for several seconds after rapid R
        # presses.  Cancel queued work, wait only for the writes already in
        # progress, then close each unfinished container without an encoder
        # flush.
        while self._pending_video_steps:
            futures = self._pending_video_steps.popleft()
            for future in futures.values():
                future.cancel()

        executors = self._writer_executors
        self._writer_executors = {}
        for executor in executors.values():
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass

        for writer in self._writers.values():
            try:
                abort = getattr(writer, "abort", None)
                if callable(abort):
                    abort()
                else:
                    writer.stop()
            except Exception:
                pass
        self._writers = {}

    def publish_videos(self, videos_dir: Path):
        """Finalize and atomically publish the three canonical MP4s."""
        if self._closed:
            raise RuntimeError("Recording spool is already closed.")
        if self._published:
            raise RuntimeError("Recording spool videos were already published.")
        if not self.encode_videos:
            self._published = True
            return {}, {}
        self._stop_video_writers(strict=True)

        missing = [
            key
            for key in CANONICAL_CAMERA_KEYS
            if key not in self._camera_sources
            or self._frame_counts[key] != self._observation_count
            or self._observation_count <= 0
            or not (self.root / f"{key}.mp4").is_file()
            or (self.root / f"{key}.mp4").stat().st_size <= 0
        ]
        if missing:
            raise RuntimeError(
                "Missing required canonical camera image(s): "
                f"{missing}. Expected saved videos are exactly "
                f"{list(CANONICAL_CAMERA_KEYS)}."
            )

        videos_dir = Path(videos_dir)
        videos_dir.parent.mkdir(parents=True, exist_ok=True)
        if videos_dir.exists():
            raise FileExistsError(
                f"Refusing to replace existing video directory: {videos_dir}"
            )
        staging_dir = videos_dir.parent / (
            f".{videos_dir.name}.recording-spool-{os.getpid()}-{self.root.name}"
        )
        staging_dir.mkdir()
        try:
            for canonical_key in CANONICAL_CAMERA_KEYS:
                source_path = self.root / f"{canonical_key}.mp4"
                staged_path = staging_dir / f"{canonical_key}.mp4"
                try:
                    os.replace(source_path, staged_path)
                except OSError as exc:
                    if exc.errno != errno.EXDEV:
                        raise
                    shutil.copyfile(source_path, staged_path)
                    source_path.unlink()
            # The staging directory lives beside the destination. Publishing
            # the directory is therefore one same-filesystem atomic rename,
            # so consumers never observe only one or two canonical MP4s.
            os.replace(staging_dir, videos_dir)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

        for canonical_key in CANONICAL_CAMERA_KEYS:
            destination = videos_dir / f"{canonical_key}.mp4"
            print(
                f"[Saved video] {destination} "
                f"(source: {self._camera_sources[canonical_key]})"
            )

        self._published = True
        return dict(self._camera_sources), dict(self._first_frames)

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._abort_video_writers()
        shutil.rmtree(self.root, ignore_errors=True)
        if not self._shared_parent:
            try:
                self.parent.rmdir()
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.cleanup()
        return False


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
    """Display canonical simulator cameras during teleoperation collection."""

    WINDOW_TITLES = {
        "front": "Main",
        "random_camera": "Main",
        "ego": "Main",
        "wrist": "Wrist",
        "wrist_left": "Wrist Left",
        "wrist_right": "Wrist Right",
    }

    def __init__(self, display_scale=0.75, *, include_main=False):
        self.display_scale = display_scale
        self.include_main = bool(include_main)
        self.active_windows = {}

    @staticmethod
    def _image_for_key(observation, camera_key):
        if not isinstance(observation, dict):
            return None
        direct = observation.get(camera_key)
        if direct is not None:
            return direct
        images = observation.get("images")
        if isinstance(images, dict):
            return images.get(camera_key)
        return None

    def _get_visible_keys(self, observation):
        if not isinstance(observation, dict):
            return []

        visible_keys = []
        if self.include_main:
            for camera_key in ("front", "random_camera", "ego"):
                if _to_bgr_image(self._image_for_key(observation, camera_key)) is not None:
                    visible_keys.append(camera_key)
                    break

        if _to_bgr_image(self._image_for_key(observation, "wrist")) is not None:
            visible_keys.append("wrist")
        else:
            for camera_key in ("wrist_left", "wrist_right"):
                if _to_bgr_image(self._image_for_key(observation, camera_key)) is not None:
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
                try:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                except cv2.error as exc:
                    raise RuntimeError(
                        "OpenCV HighGUI is not available; wrist camera windows are disabled."
                    ) from exc
                self.active_windows[camera_key] = window_name

    @staticmethod
    def _draw_preview_overlay(frame: np.ndarray, camera_key: str, env) -> np.ndarray:
        guide_fn = None
        current = env
        for _ in range(16):
            if current is None:
                break
            guide_fn = getattr(current, "c01_hammer_guide_pixels", None)
            if callable(guide_fn):
                break
            unwrapped = getattr(current, "unwrapped", None)
            if unwrapped is not None and unwrapped is not current:
                current = unwrapped
            else:
                current = getattr(current, "env", None)
        if not callable(guide_fn):
            return frame
        try:
            guide = guide_fn(camera_key, frame.shape)
        except Exception:
            return frame
        if not isinstance(guide, dict):
            return frame

        line_start = guide.get("line_start")
        line_end = guide.get("line_end")
        impact = guide.get("impact")
        nail_target = guide.get("nail_target")
        if line_start is None or line_end is None or impact is None:
            return frame

        result = frame.copy()
        color = (80, 230, 80) if guide.get("aligned") else (0, 180, 255)
        cv2.line(result, tuple(line_start), tuple(line_end), color, 2, cv2.LINE_AA)
        cv2.circle(result, tuple(impact), 7, color, 2, cv2.LINE_AA)
        cv2.circle(result, tuple(impact), 2, color, -1, cv2.LINE_AA)
        if nail_target is not None:
            cv2.circle(
                result,
                tuple(nail_target),
                6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return result

    def update_and_show(self, observation, *, env=None):
        """Show simulator camera windows and return the last pressed key."""
        visible_keys = self._get_visible_keys(observation)
        self._ensure_windows(visible_keys)

        if not visible_keys:
            try:
                return cv2.waitKey(1) & 0xFF
            except cv2.error as exc:
                raise RuntimeError(
                    "OpenCV HighGUI is not available; wrist camera windows are disabled."
                ) from exc

        for camera_key in visible_keys:
            frame = _to_bgr_image(self._image_for_key(observation, camera_key))
            if frame is None:
                continue
            if env is not None:
                frame = self._draw_preview_overlay(frame, camera_key, env)
            if self.display_scale != 1.0:
                new_width = max(1, int(frame.shape[1] * self.display_scale))
                new_height = max(1, int(frame.shape[0] * self.display_scale))
                frame = cv2.resize(
                    frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
                )
            window_name = self.active_windows[camera_key]
            try:
                cv2.imshow(window_name, frame)
            except cv2.error as exc:
                raise RuntimeError(
                    "OpenCV HighGUI is not available; wrist camera windows are disabled."
                ) from exc

        try:
            return cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            raise RuntimeError(
                "OpenCV HighGUI is not available; wrist camera windows are disabled."
            ) from exc

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
      - C08 28-d Wuji: [pos3, quat4, 20 joints, legacy no-op] -> 27-d
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


def _group_scripted_bimanual_action(
    action: np.ndarray,
    *,
    hand_action_dim: int,
    task_label: str,
) -> dict[str, np.ndarray]:
    """Convert a flat scripted-policy layout to the recorder's side layout."""

    flat = np.asarray(action, dtype=np.float64).reshape(-1)
    hand_dim = int(hand_action_dim)
    expected_dim = 14 + 2 * hand_dim
    if hand_dim <= 0 or flat.shape != (expected_dim,):
        raise ValueError(
            f"{task_label} scripted action must contain right/left TCP poses followed "
            f"by two hand commands; expected {(expected_dim,)}, got {flat.shape}."
        )
    return {
        "right": np.concatenate([flat[:7], flat[14 : 14 + hand_dim]]),
        "left": np.concatenate([flat[7:14], flat[14 + hand_dim :]]),
    }


def _group_fu015_scripted_action(
    action: np.ndarray,
    *,
    hand_action_dim: int,
) -> dict[str, np.ndarray]:
    return _group_scripted_bimanual_action(
        action,
        hand_action_dim=hand_action_dim,
        task_label="FU015",
    )


def _group_p002_scripted_action(
    action: np.ndarray,
    *,
    hand_action_dim: int,
) -> dict[str, np.ndarray]:
    return _group_scripted_bimanual_action(
        action,
        hand_action_dim=hand_action_dim,
        task_label="P0-02",
    )


def _group_p003_scripted_action(
    action: np.ndarray,
    *,
    hand_action_dim: int,
) -> dict[str, np.ndarray]:
    return _group_scripted_bimanual_action(
        action,
        hand_action_dim=hand_action_dim,
        task_label="P0-03",
    )


def _action_storage_layout(action_shape_per_step) -> str:
    shape = tuple(int(value) for value in action_shape_per_step)
    if shape in ((23,), (27,)):
        return "single_pose_hand"
    if shape in ((46,), (54,)):
        return "right_pose_hand_then_left_pose_hand"
    return "flat_raw_env_action"


def _requires_b03_newton_source_validity(
    exp_name: str,
    sim_backend: str,
) -> bool:
    """Return whether an episode must pass the strict B03 Newton source gate."""

    return bool(
        str(exp_name) == "t09_insert_peg"
        and str(sim_backend).strip().lower() == "isaacsim_newton"
    )


def _evaluate_b03_newton_saved_source_validity(
    episode_data: Mapping[str, object],
    environment_schema: Mapping[str, object] | None,
) -> dict[str, object]:
    """Evaluate the immutable post-action substeps before publishing a demo."""

    schema = environment_schema if isinstance(environment_schema, Mapping) else {}
    return evaluate_b03_newton_episode_arrays(
        episode_data,
        schema,
        field_prefix="next_",
    )


def _action_storage_layout_for_action(action) -> str:
    """Describe the layout before a possibly structured action is flattened."""

    if isinstance(action, Mapping):
        if "right" not in action or "left" not in action:
            raise ValueError(
                f"Dict action must contain right and left keys, got {sorted(action)}."
            )
        return "right_pose_hand_then_left_pose_hand"
    shape = tuple(np.asarray(action).reshape(-1).shape)
    if shape in ((46,), (54,)):
        return "right_pose_left_pose_right_hand_left_hand"
    return _action_storage_layout(shape)


def _physical_replay_recording_contract(
    *,
    recorded_drive_targets: bool = True,
    physics_backend: str = "physx",
    newton_cross_gpu: bool = False,
    deterministic_detached_hand_path: bool = False,
) -> dict[str, object]:
    """Describe the exact trajectory inputs the certifier must execute."""

    if deterministic_detached_hand_path:
        return {
            "physical_transition_contract": (
                DETACHED_HAND_ACTION_TRANSITION_CONTRACT
            ),
            "trajectory_input_mode": DETACHED_HAND_ACTION_INPUT_MODE,
            "physical_replay_goal": PHYSICAL_REPLAY_CERTIFICATION_MODE,
            "physical_replay_action_source": (
                "exact_raw_environment_action_executed"
            ),
            "physical_replay_ik": "not_present_in_live_runtime",
            "deterministic_kinematic_hand_inputs": True,
            "recorded_drive_target_replay_required": False,
        }
    if recorded_drive_targets:
        is_newton = str(physics_backend).strip().lower() == "newton"
        return {
            "physical_transition_contract": (
                (
                    "seeded_fresh_newton_reset_plus_per_substep_drive_targets_"
                    "cross_gpu_v2"
                )
                if is_newton and newton_cross_gpu
                else "seeded_fresh_newton_reset_plus_per_substep_drive_targets_v1"
                if is_newton
                else RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT
            ),
            "trajectory_input_mode": (
                "recorded_newton_substep_robot_joint_position_targets"
                if is_newton
                else RECORDED_DRIVE_TARGET_INPUT_MODE
            ),
            "physical_replay_goal": PHYSICAL_REPLAY_CERTIFICATION_MODE,
            "physical_replay_action_source": (
                "exact_raw_environment_action_executed"
            ),
            "physical_replay_drive_target_source": (
                "exact_recorded_per_newton_substep_robot_position_targets"
                if is_newton
                else "exact_recorded_per_physx_substep_robot_position_targets"
            ),
            "physical_replay_ik": (
                "computed_once_in_source_and_not_recomputed_in_certifier"
            ),
            "recorded_drive_target_replay_required": True,
        }
    return {
        "physical_transition_contract": LEGACY_RAW_ACTION_TRANSITION_CONTRACT,
        "trajectory_input_mode": LEGACY_RAW_ACTION_INPUT_MODE,
        "physical_replay_goal": PHYSICAL_REPLAY_CERTIFICATION_MODE,
        "physical_replay_action_source": (
            "exact_raw_environment_action_executed"
        ),
        "physical_replay_ik": "environment_controller_reexecuted",
        "recorded_drive_target_replay_required": False,
    }


def _write_spooled_depth_outputs(
    depth_frames_per_camera,
    videos_dir: Path,
    video_fps: int,
    *,
    expected_frame_count: int,
):
    """Write depth artifacts with at most one decoded frame resident in RAM."""
    videos_dir = Path(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for cam_key, frames in depth_frames_per_camera.items():
        if len(frames) != int(expected_frame_count):
            raise RuntimeError(
                f"Depth stream {cam_key!r} has {len(frames)} frames; "
                f"expected {expected_frame_count}."
            )

        first = np.asarray(frames[0], dtype=np.float32)
        if first.ndim != 2:
            raise ValueError(
                f"Depth frame for {cam_key!r} must be HxW, got {first.shape}."
            )
        frame_shape = tuple(first.shape)
        stack_path = videos_dir / f".{cam_key}_depth_stack.npy"
        npz_path = videos_dir / f"{cam_key}_depth.npz"
        temporary_npz_path = videos_dir / f".{cam_key}_depth.npz.tmp"

        try:
            depth_stack = np.lib.format.open_memmap(
                stack_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(frames), *frame_shape),
            )
            for frame_index, frame in enumerate(frames):
                array = np.asarray(frame, dtype=np.float32)
                if tuple(array.shape) != frame_shape:
                    raise ValueError(
                        f"Depth frame {frame_index} for {cam_key!r} has shape "
                        f"{array.shape}; expected {frame_shape}."
                    )
                depth_stack[frame_index] = array
            depth_stack.flush()
            del depth_stack

            # np.savez_compressed is convenient but first stacks every frame in
            # RAM. A .npz is just a ZIP containing depth.npy, so archive the
            # disk-backed stack directly to retain the exact public format.
            with zipfile.ZipFile(
                temporary_npz_path,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
                allowZip64=True,
            ) as archive:
                archive.write(stack_path, arcname="depth.npy")
            os.replace(temporary_npz_path, npz_path)
            saved.append(str(npz_path))

            lo = np.inf
            hi = -np.inf
            for frame in frames:
                array = np.asarray(frame, dtype=np.float32)
                finite = array[np.isfinite(array)]
                if finite.size:
                    lo = min(lo, float(finite.min()))
                    hi = max(hi, float(finite.max()))
            if not np.isfinite(lo) or not np.isfinite(hi):
                lo, hi = 0.0, 1.0
            elif hi <= lo:
                hi = lo + 1e-6

            mp4_path = videos_dir / f"{cam_key}_depth.mp4"
            writer = Mp4VideoWriter.create_h264(
                fps=video_fps,
                codec="h264",
                input_pix_fmt="rgb24",
                crf=21,
                thread_type="FRAME",
                thread_count=2,
            )
            writer.start(str(mp4_path))
            try:
                for frame in frames:
                    array = np.asarray(frame, dtype=np.float32)
                    clipped = np.clip(array, lo, hi)
                    normalized = (clipped - lo) / (hi - lo)
                    normalized[~np.isfinite(array)] = 0.0
                    gray = (normalized * 255.0).astype(np.uint8)
                    preview = np.repeat(gray[..., None], 3, axis=-1)
                    writer.write_frame(preview)
            finally:
                writer.stop()
            saved.append(
                f"{mp4_path} (depth range [{lo:.3f}, {hi:.3f}] m)"
            )
        finally:
            for temporary_path in (stack_path, temporary_npz_path):
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass

    return saved


def _canonical_camera_views(env, camera_sources: Mapping[str, str]) -> dict:
    """Describe saved canonical videos using their actual observation sources."""
    if env is None:
        return {}
    source_keys = list(dict.fromkeys(camera_sources.values()))
    source_views = describe_camera_views(env, source_keys)
    canonical_views = {}
    for canonical_key, source_key in camera_sources.items():
        if source_key not in source_views:
            continue
        view = dict(source_views[source_key])
        view["observation_source_key"] = source_key
        canonical_views[canonical_key] = view
    return canonical_views


def _write_demo_contents(
    trajectory,
    exp_name: str,
    success_index: int,
    demo_dir: Path,
    video_fps: int,
    env=None,
    environment_schema=None,
    required_recording_fields=None,
    video_spool: RecordingSpool | None = None,
    collection_run_seed: int | None = None,
    episode_reset_seed: int | None = None,
    collection_seed_source: str | None = None,
    episode_reset_ordinal: int | None = None,
    dataset_generation: DatasetGeneration | None = None,
    published_demo_name: str | None = None,
    manus_raw_events: list[dict[str, object]] | None = None,
    manus_raw_capture_summary: Mapping[str, object] | None = None,
    online_camera_capture_scope: str | None = None,
    guidance_route_only: bool = False,
):
    """Save a single demo trajectory to a zarr group and MP4 camera captures.

    Structure created:
      <base_out>/<exp_name>_demo_<index>_<timestamp>[__stable_v2|__physx_v3]/
         replay.zarr/
         videos/<cam_key>.mp4

    The episode store will contain arrays for:
      - action: (T, N)
      - action_rotvec: (T,22/26) for single-arm or (T,44/52) for bimanual if convertible
      - state: (T, ... ) if present under observations['state']
      - timestamp: (T,) in seconds
      - any small consistent fixed-shape numeric observation fields
      - recording_schema.json plus matching Zarr attrs describing every field
    """
    policy_actions_flag = str(
        _optional_flag_value("policy_actions_npy", "") or ""
    )
    demo_dir = Path(demo_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)
    generation = dataset_generation or _dataset_generation_for_env(env)
    episode_id = str(published_demo_name or demo_dir.name)
    named_generation_suffix = _dataset_generation_suffix_for_episode(
        episode_id
    )
    if named_generation_suffix != generation.directory_suffix:
        raise ValueError(
            "Dataset generation and episode directory name disagree: "
            f"generation={generation.name!r}, episode_id={episode_id!r}."
        )
    generation_metadata = _dataset_generation_metadata(
        generation,
        episode_id=episode_id,
    )

    # Prepare episode-level low-dim arrays.
    actions = []
    action_storage_layouts: set[str] = set()
    states = []
    next_states = []
    terminals = []
    successes = []

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
        transition_layout = transition.get("action_storage_layout")
        if transition_layout is not None:
            if not isinstance(transition_layout, str) or not transition_layout:
                raise ValueError("Transition action_storage_layout must be text.")
            action_storage_layouts.add(transition_layout)
        terminals.append(bool(transition.get("dones", False)))
        transition_info = transition.get("infos", {})
        successes.append(
            bool(
                isinstance(transition_info, Mapping)
                and transition_info.get("succeed", False)
            )
        )

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
    if len(action_storage_layouts) > 1:
        raise ValueError(
            "Trajectory mixes incompatible action storage layouts: "
            f"{sorted(action_storage_layouts)}."
        )
    action_storage_layout = (
        next(iter(action_storage_layouts))
        if action_storage_layouts
        else _action_storage_layout(actions_arr.shape[1:])
    )
    T = actions_arr.shape[0]
    if video_spool is None:
        raise RuntimeError("A finalized episode video spool is required.")
    if video_spool.frame_count != T:
        raise RuntimeError(
            f"Video spool has {video_spool.frame_count} frame bundles, "
            f"but trajectory has {T} transitions."
        )
    if FLAGS.save_depth and video_spool.depth_frame_count != T:
        raise RuntimeError(
            f"Depth spool has {video_spool.depth_frame_count} frame bundles, "
            f"but trajectory has {T} transitions."
        )
    timestamps_arr = transition_timestamps(
        trajectory,
        fallback_hz=float(FLAGS.data_fps),
    )
    timestamp_source = "environment_info_or_fixed_rate_fallback"
    wall_timestamps_arr = transition_monotonic_timestamps(trajectory)
    sampling_timing = (
        wall_sampling_contract(
            wall_timestamps_arr,
            target_hz=float(video_fps),
        )
        if wall_timestamps_arr is not None
        else None
    )
    if len(timestamps_arr) >= 2:
        duration_seconds = max(
            0.0,
            float(timestamps_arr[-1]) - float(timestamps_arr[0]),
        )
    else:
        duration_seconds = float(T) / max(1.0, float(FLAGS.data_fps))

    episode_data = {
        "action": actions_arr,
        "timestamp": timestamps_arr,
        "terminal": np.asarray(terminals, dtype=np.uint8),
        "succeed": np.asarray(successes, dtype=np.uint8),
    }
    if wall_timestamps_arr is not None:
        episode_data["wall_timestamp"] = wall_timestamps_arr

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
    from handson.data.a03_collection import pack_collection_episode, write_training_export
    a03_fields, a03_contract = pack_collection_episode(trajectory)
    episode_data.update(a03_fields)
    if a03_contract is not None:
        # Raw 30 Hz drive evidence and sparse upstream training labels have
        # different roles; never promote interpolated frames into model knots.
        generation_metadata["a03_controller_contract"] = a03_contract
        write_training_export(demo_dir, a03_fields, a03_contract)

    source_physical_validity: dict[str, object] | None = None
    if _requires_b03_newton_source_validity(exp_name, str(FLAGS.sim_backend)):
        source_physical_validity = _evaluate_b03_newton_saved_source_validity(
            episode_data,
            environment_schema,
        )
        source_validity_error = b03_newton_source_validity_error(
            source_physical_validity
        )
        if source_validity_error is not None:
            raise RuntimeError(
                "Refusing to save physically invalid B03 Newton episode: "
                + source_validity_error
            )

    # Prefer a synchronized simulator clock over an assumed recorder rate.
    sim_time = episode_data.get("physx_sim_time")
    if sim_time is None:
        sim_time = episode_data.get("newton_sim_time")
    if sim_time is None:
        sim_time = episode_data.get("c01_force_sim_time")
    if sim_time is not None and np.asarray(sim_time).shape == (T, 1):
        episode_data["timestamp"] = np.asarray(sim_time, dtype=np.float64)[:, 0]
        timestamp_source = (
            "physx_sim_time"
            if episode_data.get("physx_sim_time") is not None
            else "newton_sim_time"
            if episode_data.get("newton_sim_time") is not None
            else "c01_force_sim_time"
        )
    next_sim_time = episode_data.get("next_physx_sim_time")
    if next_sim_time is None:
        next_sim_time = episode_data.get("next_newton_sim_time")
    if next_sim_time is None:
        next_sim_time = episode_data.get("next_c01_force_sim_time")
    if next_sim_time is not None and np.asarray(next_sim_time).shape == (T, 1):
        episode_data["next_timestamp"] = np.asarray(
            next_sim_time,
            dtype=np.float64,
        )[:, 0]
    if T > 0 and "next_timestamp" in episode_data:
        duration_seconds = max(
            0.0,
            float(episode_data["next_timestamp"][-1])
            - float(episode_data["timestamp"][0]),
        )
    recorded_data_fps = round(
        effective_sampling_hz(
            np.asarray(episode_data["timestamp"], dtype=np.float64),
            fallback_hz=float(FLAGS.data_fps),
        ),
        9,
    )

    # action_rotvec when convertible
    converted = [convert_action_quat_to_rotvec(a) for a in episode_data["action"]]
    if all(c is not None for c in converted):
        episode_data["action_rotvec"] = np.stack(converted, axis=0)

    # Create the Zarr episode store.
    zarr_path = demo_dir / "replay.zarr"
    store = zarr.DirectoryStore(str(zarr_path))
    episode_store = ZarrEpisodeStore.create_empty(storage=store)
    for name, value in generation_metadata.items():
        episode_store.root.attrs[name] = value

    # Append the current episode to disk.
    episode_store.append_episode(episode_data, compressors="disk")

    manus_raw_summary: dict[str, object] | None = None
    if manus_raw_events is not None:
        manus_raw_summary = write_manus_raw_zarr(
            episode_store.root,
            manus_raw_events,
            summary=manus_raw_capture_summary,
        )
        write_manus_raw_schema(demo_dir, manus_raw_summary)

    schema_path = write_recording_schema(
        demo_dir,
        episode_store.root,
        episode_data,
        environment_schema=environment_schema,
    )
    raw_env = getattr(env, "unwrapped", env)
    task_guidance_recording = bool(
        getattr(raw_env, "_guidance_mode", False)
        or os.environ.get("HANDSON_TASK_GUIDANCE_RECORDING", "")
        .strip()
        .lower()
        not in {"", "0", "false", "no", "off"}
    )
    deferred_physical_replay_certification = (
        _deferred_physical_replay_certification_enabled()
    )
    postprocess_manual_only = _postprocess_manual_only_enabled()
    manual_publication_declaration = (
        _manual_postprocess_publication_declaration(
            enabled=postprocess_manual_only,
        )
    )
    sim_backend = str(FLAGS.sim_backend)
    offline_arm_render_declaration = _offline_arm_render_declaration(
        exp_name,
        sim_backend=sim_backend,
        manual_only=postprocess_manual_only,
        c01_enabled=bool(FLAGS.auto_c01_arm_video),
        plan_enabled=bool(FLAGS.auto_plan_arm_video),
        c01_env_value=os.environ.get(
            "HANDSON_C01_AUTO_ARM_VIDEO",
            "1",
        ),
        plan_env_value=os.environ.get(
            "HANDSON_PLAN_AUTO_ARM_VIDEO",
            "1",
        ),
    )
    e01_ipc_env_value = os.environ.get(
        "HANDSON_E01_AUTO_IPC_REPLAY",
        "1",
    )
    e01_arm_env_value = os.environ.get(
        "HANDSON_E01_AUTO_ARM_IK",
        "1",
    )
    e01_requested_stages = _e01_postprocess_requested_stages(
        exp_name,
        sim_backend=sim_backend,
        ipc_env_value=e01_ipc_env_value,
        arm_env_value=e01_arm_env_value,
    )
    fresh_certified_worker = (
        os.environ.get(_FRESH_CERTIFIED_COLLECTION_WORKER_ENV, "") == "1"
    )
    recorded_drive_target_replay = bool(
        getattr(raw_env, "recorded_drive_target_replay_supported", False)
        and (
            (
                FLAGS.sim_backend == "isaacsim"
                and FLAGS.record_physx_ground_truth
                and (
                    fresh_certified_worker
                    or deferred_physical_replay_certification
                )
            )
            or bool(getattr(raw_env, "_newton_ground_truth_enabled", False))
        )
    )
    b03_newton_cross_gpu = bool(
        sim_backend == "isaacsim_newton"
        and exp_name == "t09_insert_peg"
    )
    if online_camera_capture_scope is None:
        online_camera_capture_scope = _runtime_online_camera_capture_scope(
            exp_name,
            sim_backend,
        )
    (
        online_camera_capture_scope,
        canonical_camera_postrender_required,
        online_video_encoding_enabled,
    ) = _online_camera_spool_contract(
        online_camera_capture_scope,
        video_spool,
        save_depth=bool(FLAGS.save_depth),
    )
    if b03_newton_cross_gpu:
        replay_profile = str(
            getattr(raw_env, "_newton_replay_profile", "")
        )
        if replay_profile != NEWTON_CROSS_GPU_PROFILE:
            raise RuntimeError(
                "B03 Newton training data must be collected with cross_gpu_v1"
            )
        if getattr(raw_env, "_direct_hand_control", False):
            raise RuntimeError(
                "B03 Newton training data cannot contain direct hand state writes"
            )
        if not (
            getattr(raw_env, "_newton_ground_truth_enabled", False)
            and recorded_drive_target_replay
        ):
            raise RuntimeError(
                "B03 Newton training data requires the complete per-substep "
                "drive-target and solver-readback trace"
            )
    actual_arm_type = str(getattr(raw_env, "arm_type", FLAGS.arm_type))
    requested_arm_type = str(
        getattr(raw_env, "requested_arm_type", FLAGS.arm_type)
    )
    simplified_post_ik = bool(
        requested_arm_type == SIMPLIFIED_POST_IK_PROFILE
    )
    arm_control_profile = str(
        getattr(
            raw_env,
            "arm_control_profile",
            SIMPLIFIED_POST_IK_PROFILE
            if simplified_post_ik
            else "articulated_live_ik",
        )
    )
    simplified_arm_contract = {
        "requested_arm_type": requested_arm_type,
        "arm_control_profile": arm_control_profile,
        "simplified_post_ik": simplified_post_ik,
        "simplified_post_ik_replay_scope": (
            SIMPLIFIED_POST_IK_REPLAY_SCOPE if simplified_post_ik else None
        ),
        "simplified_post_ik_tcp_source": (
            SIMPLIFIED_POST_IK_TCP_SOURCE if simplified_post_ik else None
        ),
    }
    actual_hand_type = str(getattr(raw_env, "hand_type", FLAGS.hand_type))
    camera_profile = getattr(raw_env, "recording_camera_profile", {})
    if not isinstance(camera_profile, dict):
        camera_profile = {}
    reset_contract_provider = getattr(
        raw_env,
        "get_physical_replay_reset_contract",
        None,
    )
    reset_replay_contract = (
        reset_contract_provider()
        if callable(reset_contract_provider)
        else None
    )
    # Only the environment-authored reset contract may grant BC021 final-asset
    # readiness.  The helper returns an empty mapping for every other task and
    # for provisional/incomplete BC021 declarations, so CLI/UI flags cannot
    # manufacture an uploadable asset identity.
    bc021_final_asset_metadata = bc021_collection_meta_binding(
        exp_name,
        reset_replay_contract,
    )
    replay_manifest_path = write_replay_assets(
        demo_dir,
        env,
        task_name=exp_name,
        recording_parameters={
            **({
                "d20_state_layout": getattr(raw_env, "_d20_state_layout", {}),
                "d20_stage_resume": getattr(raw_env, "_d20_stage_resume", {}),
            } if exp_name in {"d20_1_load_and_start", "d20_1_5_set_controls", "d20_2_retrieve_and_plate", "d20_3_finish_table_setting"} else {}),
            "recorder": Path(__file__).name,
            "episode_index": int(success_index),
            "episode_tag": "demo",
            "recording_purpose": (
                "task_guidance" if task_guidance_recording else "benchmark"
            ),
            "guidance_route_only": bool(guidance_route_only),
            **generation_metadata,
            **manual_publication_declaration,
            **offline_arm_render_declaration,
            "step_count": int(T),
            "action_shape_per_step": list(actions_arr.shape[1:]),
            "action_storage_layout": action_storage_layout,
            "action_semantics": (
                "a03_dense_execution_pose_diagnostic_use_policy_knots_for_training"
                if a03_contract is not None
                else "public_policy_action_adapted_by_task_wrapper"
                if policy_actions_flag
                else "exact_raw_environment_action_executed"
            ),
            "a03_controller_contract": a03_contract,
            "policy_action_validation": bool(policy_actions_flag),
            "policy_actions_npy": (
                str(Path(policy_actions_flag).expanduser().resolve())
                if policy_actions_flag
                else ""
            ),
            "policy_actions_sha256": (
                hashlib.sha256(
                    Path(policy_actions_flag)
                    .expanduser()
                    .resolve()
                    .read_bytes()
                ).hexdigest()
                if policy_actions_flag
                else ""
            ),
            **_physical_replay_recording_contract(
                recorded_drive_targets=recorded_drive_target_replay,
                deterministic_detached_hand_path=simplified_post_ik,
                physics_backend=(
                    "newton"
                    if getattr(raw_env, "_newton_ground_truth_enabled", False)
                    else "physx"
                ),
                newton_cross_gpu=b03_newton_cross_gpu,
            ),
            "stable_recording_origin": copy.deepcopy(
                generation.stable_origin
            ),
            "video_fps": int(video_fps),
            "video_spool_crf": int(
                video_spool.video_crf
                if video_spool is not None
                else _recording_spool_video_crf()
            ),
            "data_fps": recorded_data_fps,
            "requested_data_fps": float(FLAGS.data_fps),
            "timestamp_source": timestamp_source,
            **(
                {"wall_clock_sampling": sampling_timing}
                if sampling_timing is not None
                else {}
            ),
            "render_mode": FLAGS.render_mode,
            "image_obs": bool(getattr(raw_env, "image_obs", True)),
            "headless": bool(
                getattr(
                    raw_env,
                    "_headless",
                    FLAGS.render_mode != "human",
                )
            ),
            "randomize": bool(FLAGS.randomize),
            "randomize_dynamics": bool(FLAGS.randomize_dynamics),
            "scripted_policy": bool(getattr(FLAGS, "scripted_policy", False)),
            "save_depth": bool(FLAGS.save_depth),
            "save_camera_info": bool(FLAGS.save_camera_info),
            "fisheye_intrinsics_yaml": FLAGS.fisheye_intrinsics_yaml,
            "apply_fisheye_distortion": os.environ.get(
                "HANDSON_APPLY_FISHEYE_DISTORTION",
                "0",
            ).strip().lower()
            not in {"0", "false", "no", "off", ""},
            "target_label": FLAGS.target_label,
            "arm_type": requested_arm_type,
            "physical_arm_asset_type": actual_arm_type,
            **simplified_arm_contract,
            "hand_type": actual_hand_type,
            "manus_raw_recording": (
                {
                    **manus_raw_summary,
                    "zarr_group": "manus_raw",
                    "schema_file": MANUS_RAW_SCHEMA_FILE,
                    "event_timing": "independent_source_callbacks_not_resampled",
                }
                if manus_raw_summary is not None
                else {
                    "format": MANUS_RAW_FORMAT,
                    "version": MANUS_RAW_VERSION,
                    "complete": False,
                    "reason": "not_a_wuji_manus_collection",
                }
            ),
            "sim_backend": sim_backend,
            "physical_replay_state_recorded": bool(
                isinstance(environment_schema, Mapping)
                and (
                    isinstance(environment_schema.get("physx"), Mapping)
                    or isinstance(environment_schema.get("newton"), Mapping)
                )
            ),
            "newton_physical_replay_available": bool(
                getattr(raw_env, "_newton_ground_truth_enabled", False)
                and recorded_drive_target_replay
            ),
            **(
                {
                    "source_physical_validity": copy.deepcopy(
                        source_physical_validity
                    )
                }
                if source_physical_validity is not None
                else {}
            ),
            "newton_replay_profile": (
                NEWTON_CROSS_GPU_PROFILE if b03_newton_cross_gpu else None
            ),
            "newton_cross_gpu_certification_required": b03_newton_cross_gpu,
            "newton_cross_gpu_certification_runs_required": (
                NEWTON_CROSS_GPU_REQUIRED_RUNS
                if b03_newton_cross_gpu
                else 0
            ),
            "newton_cross_gpu_certification_mode": (
                NEWTON_CROSS_GPU_CERTIFICATION_MODE
                if b03_newton_cross_gpu
                else None
            ),
            "publish_only_after_newton_cross_gpu_certification": (
                b03_newton_cross_gpu
            ),
            "physical_replay_certification_required": bool(
                FLAGS.sim_backend == "isaacsim"
                and FLAGS.record_physx_ground_truth
            ),
            "physical_replay_certification_runs_required": (
                3
                if FLAGS.sim_backend == "isaacsim"
                and FLAGS.record_physx_ground_truth
                else 0
            ),
            "physical_replay_certification_mode": (
                DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE
                if deferred_physical_replay_certification
                else FRESH_PROCESS_CERTIFICATION_MODE
            ),
            **(
                {
                    "physical_replay_certification_scope": (
                        ORDERED_SESSION_PHYSICAL_REPLAY_CERTIFICATION_SCOPE
                    ),
                    "physical_replay_session_history_pending": True,
                }
                if deferred_physical_replay_certification
                else {}
            ),
            "recorder_lifecycle": (
                PERSISTENT_RECORDER_LIFECYCLE
                if deferred_physical_replay_certification
                else FRESH_RECORDER_LIFECYCLE
            ),
            "recorder_pid": int(os.getpid()),
            "source_process_reuse_enabled": bool(
                deferred_physical_replay_certification
            ),
            "source_scene_rebuild_per_reset": bool(
                deferred_physical_replay_certification
                and _isaac_in_process_hard_reset_enabled_for_task(
                    getattr(raw_env, "task_name", "")
                )
            ),
            "source_scene_generation": int(
                (generation.stable_origin or {}).get("scene_generation", 0)
            ),
            "fresh_recorder_process_per_episode_required": bool(
                FLAGS.sim_backend == "isaacsim"
                and FLAGS.record_physx_ground_truth
                and not deferred_physical_replay_certification
            ),
            "fresh_recorder_process_per_attempt_required": bool(
                FLAGS.sim_backend == "isaacsim"
                and FLAGS.record_physx_ground_truth
                and fresh_certified_worker
                and not deferred_physical_replay_certification
            ),
            "publish_only_after_fresh_certification": bool(
                (
                    FLAGS.sim_backend == "isaacsim"
                    and FLAGS.record_physx_ground_truth
                )
                or b03_newton_cross_gpu
            ),
            "precertification_storage": (
                "hidden_same_filesystem_pending_directory"
                if deferred_physical_replay_certification
                else "fresh_supervisor_managed"
            ),
            "publication_destination_name": episode_id,
            "fresh_scene_replay_required": bool(
                (
                    FLAGS.sim_backend == "isaacsim"
                    and FLAGS.record_physx_ground_truth
                )
                or b03_newton_cross_gpu
            ),
            "fresh_certifier_process_per_run_required": bool(
                (
                    FLAGS.sim_backend == "isaacsim"
                    and FLAGS.record_physx_ground_truth
                )
                or b03_newton_cross_gpu
            ),
            "collection_run_seed": (
                None
                if collection_run_seed is None
                else int(collection_run_seed)
            ),
            "episode_reset_seed": (
                None
                if episode_reset_seed is None
                else int(episode_reset_seed)
            ),
            "collection_seed_source": collection_seed_source,
            "episode_reset_ordinal": (
                None
                if episode_reset_ordinal is None
                else int(episode_reset_ordinal)
            ),
            "episode_seed_derivation": (
                EPISODE_SEED_DERIVATION
                if episode_reset_seed is not None
                else None
            ),
            "reset_replay_contract": reset_replay_contract,
            "record_physx_ground_truth": bool(
                FLAGS.record_physx_ground_truth
                and FLAGS.sim_backend == "isaacsim"
            ),
            "recording_camera_profile": camera_profile,
            "online_camera_capture_scope": online_camera_capture_scope,
            "canonical_camera_postrender_required": (
                canonical_camera_postrender_required
            ),
            "online_video_encoding_enabled": online_video_encoding_enabled,
            "live_direct_hand_control": bool(
                _uses_b03_newton_direct_hand_pipeline(
                    exp_name,
                    sim_backend,
                )
            ),
            # This declaration is part of replay_manifest.json and therefore
            # part of the certified-input hash.  collection_meta repeats it
            # for UI discovery, but cannot weaken the upload gate by itself.
            "isaac_arm_render_deferred": bool(
                _uses_isaac_deferred_arm_pipeline(
                    exp_name,
                    str(FLAGS.sim_backend),
                    os.environ.get(
                        "HANDSON_ISAACLAB_DEFER_ARM_RENDER",
                        "0",
                    ),
                )
            ),
            "arm_ik_deferred": bool(simplified_post_ik),
            "camera_screen_effect": bool(
                FLAGS.camera_screen_effect
                or exp_name == "t23_photograph_logo"
            ),
            "ipad_screen_effect": bool(FLAGS.ipad_screen_effect),
            "auto_c01_arm_video": bool(FLAGS.auto_c01_arm_video),
            "auto_plan_arm_video": bool(FLAGS.auto_plan_arm_video),
            "auto_e01_offline_arm_ik": _should_auto_compute_e01_arm_plan(
                exp_name,
                sim_backend=sim_backend,
                env_value=e01_arm_env_value,
            ),
            "auto_e01_offline_ipc_replay": _should_auto_replay_e01_ipc(
                exp_name,
                sim_backend=sim_backend,
                env_value=e01_ipc_env_value,
            ),
            "c01_arm_max_fk_position_error": float(
                FLAGS.c01_arm_max_fk_position_error
            ),
            "c01_arm_max_fk_orientation_error": float(
                FLAGS.c01_arm_max_fk_orientation_error
            ),
            "c01_arm_collision_penetration_tolerance": float(
                FLAGS.c01_arm_collision_penetration_tolerance
            ),
            "c01_arm_collision_sweep_joint_step": float(
                FLAGS.c01_arm_collision_sweep_joint_step
            ),
        },
        environment_schema=environment_schema,
        repo_root=Path(__file__).resolve().parents[1],
    )

    print(f"[Saved replay.zarr] {zarr_path} (steps: {episode_data['action'].shape[0]})")
    print(f"[Saved recording schema] {schema_path}")
    print(f"[Saved replay manifest/model] {replay_manifest_path}")

    # --- write exactly the canonical saved video views ---
    videos_dir = demo_dir / "videos"
    camera_sources, first_frames = video_spool.publish_videos(videos_dir)
    camera_keys = list(CANONICAL_CAMERA_KEYS)

    if canonical_camera_postrender_required:
        print(
            "[Saved camera contract] Canonical RGB videos and sidecars are "
            "pending postrender; no placeholder MP4s were encoded online.",
            flush=True,
        )
    elif FLAGS.save_camera_info:
        camera_views = _canonical_camera_views(env, camera_sources)
        for path in write_video_camera_info(
            videos_dir=videos_dir,
            camera_keys=camera_keys,
            first_frames=first_frames,
            intrinsics_yaml=FLAGS.fisheye_intrinsics_yaml,
            camera_views=camera_views,
            camera_profile=camera_profile,
        ):
            print(f"[Saved camera info] {path}")

    if FLAGS.save_depth and not canonical_camera_postrender_required:
        depth_per_camera = {k: [] for k in camera_keys}
        for step in trajectory:
            step_depth = step.get("depth", {})
            missing_depth = [
                key
                for key in camera_keys
                if key not in step_depth or step_depth[key] is None
            ]
            if missing_depth:
                raise RuntimeError(
                    f"Trajectory depth bundle is missing {missing_depth}."
                )
            for key in camera_keys:
                depth_per_camera[key].append(step_depth[key])
        for path in _write_spooled_depth_outputs(
            depth_per_camera,
            videos_dir,
            video_fps,
            expected_frame_count=T,
        ):
            print(f"[Saved depth]  {path}")

    engine_code = {
        "mujoco": "E01",
        "isaacsim": "E02",
        "isaacsim_newton": "E04",
        "genesis": "E03",
    }[sim_backend]
    collection_meta = {
        "schema_version": 2,
        "data_code": episode_id,
        "recording_purpose": (
            "task_guidance" if task_guidance_recording else "benchmark"
        ),
        **generation_metadata,
        **manual_publication_declaration,
        **offline_arm_render_declaration,
        "task_name": exp_name,
        "task_outcome": "guidance_route" if guidance_route_only else "success",
        "source_outcome": "guidance_route" if guidance_route_only else "success",
        "guidance_route_only": bool(guidance_route_only),
        **bc021_final_asset_metadata,
        "collection_collector_id": os.environ.get(
            "HANDSON_COLLECTOR_ID", "unknown"
        ),
        "collector_id": os.environ.get("HANDSON_COLLECTOR_ID", "unknown"),
        "collection_session_id": os.environ.get(
            "HANDSON_COLLECTION_SESSION_ID", ""
        ),
        "collection_started_at": os.environ.get(
            "HANDSON_COLLECTION_STARTED_AT", ""
        ),
        "collected_at": datetime.datetime.now(
            datetime.timezone.utc
        ).astimezone().isoformat(timespec="seconds"),
        "duration_seconds": round(duration_seconds, 6),
        "step_count": int(T),
        "data_fps": recorded_data_fps,
        "requested_data_fps": float(FLAGS.data_fps),
        "timestamp_source": timestamp_source,
        **(
            {"wall_clock_sampling": sampling_timing}
            if sampling_timing is not None
            else {}
        ),
        "video_fps": int(video_fps),
        "target_label": str(FLAGS.target_label or ""),
        "arm_type": requested_arm_type,
        "physical_arm_asset_type": actual_arm_type,
        **simplified_arm_contract,
        "hand_type": str(FLAGS.hand_type),
        "manus_raw_recording": (
            dict(manus_raw_summary)
            if manus_raw_summary is not None
            else {
                "format": MANUS_RAW_FORMAT,
                "version": MANUS_RAW_VERSION,
                "complete": False,
                "reason": "not_a_wuji_manus_collection",
            }
        ),
        "sim_backend": sim_backend,
        "engine_code": engine_code,
        "engine_generation": "new" if engine_code in {"E02", "E03", "E04"} else "old",
        "engine_label": ENGINE_LABELS[engine_code],
        **(
            {
                "source_physical_validity": copy.deepcopy(
                    source_physical_validity
                )
            }
            if source_physical_validity is not None
            else {}
        ),
        # Duplicate the upload-critical physical replay declaration from the
        # immutable replay manifest.  The collection UI requires exact
        # agreement before granting the no-review A-D Isaac upload policy.
        "physical_replay_certification_required": bool(
            FLAGS.sim_backend == "isaacsim"
            and FLAGS.record_physx_ground_truth
        ),
        "newton_replay_profile": (
            NEWTON_CROSS_GPU_PROFILE if b03_newton_cross_gpu else None
        ),
        "newton_cross_gpu_certification_required": b03_newton_cross_gpu,
        "newton_cross_gpu_certification_runs_required": (
            NEWTON_CROSS_GPU_REQUIRED_RUNS if b03_newton_cross_gpu else 0
        ),
        "newton_cross_gpu_certification_mode": (
            NEWTON_CROSS_GPU_CERTIFICATION_MODE
            if b03_newton_cross_gpu
            else None
        ),
        "publish_only_after_newton_cross_gpu_certification": (
            b03_newton_cross_gpu
        ),
        "physical_replay_certification_mode": (
            DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE
            if deferred_physical_replay_certification
            else FRESH_PROCESS_CERTIFICATION_MODE
        ),
        **(
            {
                "physical_replay_certification_scope": (
                    ORDERED_SESSION_PHYSICAL_REPLAY_CERTIFICATION_SCOPE
                ),
                "physical_replay_session_history_pending": True,
            }
            if deferred_physical_replay_certification
            else {}
        ),
        "recording_camera_profile": camera_profile,
        "online_camera_capture_scope": online_camera_capture_scope,
        "canonical_camera_postrender_required": (
            canonical_camera_postrender_required
        ),
        "online_video_encoding_enabled": online_video_encoding_enabled,
        "live_direct_hand_control": bool(
            simplified_post_ik
            or _uses_b03_newton_direct_hand_pipeline(
                exp_name,
                sim_backend,
            )
        ),
        "live_control_mode": (
            "physx_detached_wuji_hands_validated_offline_rokae_ik"
            if simplified_post_ik
            else "detached_hands_offline_arm_ik"
            if _uses_e01_detached_hand_pipeline(exp_name, sim_backend)
            else "newton_articulated_rokae_wuji_position_drives"
            if b03_newton_cross_gpu
            else (
                (
                    "newton_direct_wuji_hand_deferred_full_robot_render"
                    if _uses_b03_newton_direct_hand_pipeline(
                        exp_name,
                        sim_backend,
                    )
                    else "detached_hands_deferred_isaac_arm_replay"
                )
                if _uses_isaac_deferred_arm_pipeline(
                    exp_name,
                    sim_backend,
                    os.environ.get(
                        "HANDSON_ISAACLAB_DEFER_ARM_RENDER",
                        "0",
                    ),
                )
                else "robot_in_scene"
            )
        ),
        "isaac_arm_render_deferred": _uses_isaac_deferred_arm_pipeline(
            exp_name,
            sim_backend,
            os.environ.get("HANDSON_ISAACLAB_DEFER_ARM_RENDER", "0"),
        ),
        "arm_ik_deferred": bool(
            simplified_post_ik
            or _should_auto_compute_e01_arm_plan(
                exp_name,
                sim_backend=sim_backend,
                env_value=e01_arm_env_value,
            )
        ),
        "ipc_replay_deferred": bool(
            _should_auto_replay_e01_ipc(
                exp_name,
                sim_backend=sim_backend,
                env_value=e01_ipc_env_value,
            )
        ),
        "e01_postprocess_background": bool(
            not postprocess_manual_only
            and _should_background_e01_postprocess(
                exp_name,
                sim_backend=sim_backend,
                env_value=os.environ.get(
                    "HANDSON_E01_POSTPROCESS_BACKGROUND",
                    _e01_postprocess_background_default(exp_name),
                ),
                ipc_env_value=e01_ipc_env_value,
                arm_env_value=e01_arm_env_value,
            )
        ),
        "e01_postprocess_status": (
            "pending"
            if e01_requested_stages
            else "not_requested"
        ),
    }
    (demo_dir / "collection_meta.json").write_text(
        json.dumps(collection_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    marker_path = _write_dataset_generation_marker(
        demo_dir,
        generation,
        episode_id=episode_id,
    )
    if marker_path is not None:
        print(f"[Saved dataset generation] {marker_path}")

    if postprocess_manual_only and e01_requested_stages:
        _write_e01_postprocess_document(
            demo_dir,
            {
                "format": "handson.e01_postprocess",
                "version": 1,
                "status": "pending",
                "created_at": _now_iso(),
                "queued_at": None,
                "started_at": None,
                "completed_at": None,
                "background": False,
                "operator_deferred": True,
                "postprocess_policy": MANUAL_POSTPROCESS_POLICY,
                "exp_name": exp_name,
                "sim_backend": sim_backend,
                "video_fps": max(1, int(video_fps)),
                "requested_stages": e01_requested_stages,
                "stages": {},
            },
        )

    if not postprocess_manual_only and _should_auto_generate_c01_arm_video(
        exp_name,
        enabled=FLAGS.auto_c01_arm_video and sim_backend == "mujoco",
        env_value=os.environ.get("HANDSON_C01_AUTO_ARM_VIDEO", "1"),
    ):
        scripts_dir = Path(__file__).resolve().parent
        compute_script = scripts_dir / "c01_compute_arm_plan_qpos.py"
        render_script = scripts_dir / "c01_render_arm_replay.py"
        env_vars = os.environ.copy()
        env_vars.setdefault("PYTHONPATH", str(_PACKAGE_ROOT.parent))
        env_vars.update(configure_gpu_environment(env_vars, render_mode="offscreen"))
        commands = [
            (
                "plan",
                [
                    sys.executable,
                    str(compute_script),
                    str(demo_dir),
                    "--overwrite",
                    "--max-fk-position-error",
                    str(FLAGS.c01_arm_max_fk_position_error),
                    "--max-fk-orientation-error",
                    str(FLAGS.c01_arm_max_fk_orientation_error),
                    "--collision-penetration-tolerance",
                    str(FLAGS.c01_arm_collision_penetration_tolerance),
                    "--collision-sweep-joint-step",
                    str(FLAGS.c01_arm_collision_sweep_joint_step),
                ],
            ),
            (
                "render",
                [
                    sys.executable,
                    str(render_script),
                    str(demo_dir),
                    "--fps",
                    str(video_fps),
                ],
            ),
        ]
        try:
            for stage, cmd in commands:
                print(f"[C01 arm {stage}] starting: {' '.join(cmd)}", flush=True)
                result = subprocess.run(
                    cmd,
                    env=env_vars,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                if result.stdout:
                    print(
                        result.stdout,
                        end="" if result.stdout.endswith("\n") else "\n",
                    )
                if result.returncode != 0:
                    print(
                        f"[C01 arm pipeline FAILED] {stage} exited with code "
                        f"{result.returncode}; remaining stages, including "
                        "arm rendering, were skipped.",
                        flush=True,
                    )
                    break
        except Exception as exc:
            print(f"[Warning] C01 arm pipeline skipped: {exc}", flush=True)

    if not postprocess_manual_only and _should_auto_generate_plan_arm_video(
        exp_name,
        enabled=FLAGS.auto_plan_arm_video,
        env_value=os.environ.get("HANDSON_PLAN_AUTO_ARM_VIDEO", "1"),
    ):
        render_script = Path(__file__).resolve().parent / "plan_render_arm_replay.py"
        env_vars = os.environ.copy()
        env_vars.setdefault("PYTHONPATH", str(_PACKAGE_ROOT.parent))
        env_vars.update(configure_gpu_environment(env_vars, render_mode="offscreen"))
        cmd = [
            sys.executable,
            str(render_script),
            str(demo_dir),
            "--task-name",
            exp_name,
            "--fps",
            str(video_fps),
        ]
        try:
            print(
                f"[Plan arm render] starting: {' '.join(cmd)}",
                flush=True,
            )
            result = subprocess.run(
                cmd,
                env=env_vars,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if result.stdout:
                print(
                    result.stdout,
                    end="" if result.stdout.endswith("\n") else "\n",
                )
            if result.returncode != 0:
                print(
                    "[Plan arm render FAILED] renderer exited with code "
                    f"{result.returncode}; videos_arm was not published.",
                    flush=True,
                )
        except Exception as exc:
            print(f"[Warning] Plan arm render skipped: {exc}", flush=True)

    return str(demo_dir)


def _write_e01_postprocess_document(
    demo_dir: Path,
    document: Mapping[str, object],
) -> None:
    """Atomically publish pipeline state without changing task outcome."""
    demo_dir = Path(demo_dir)
    path = demo_dir / _E01_POSTPROCESS_STATUS_FILE
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)

    collection_meta_path = demo_dir / "collection_meta.json"
    try:
        collection_meta = json.loads(collection_meta_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return
    collection_meta["e01_postprocess_status"] = document.get("status", "unknown")
    collection_meta["e01_postprocess"] = document
    temporary_meta = collection_meta_path.with_name(
        f".{collection_meta_path.name}.tmp"
    )
    temporary_meta.write_text(
        json.dumps(
            collection_meta,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_meta, collection_meta_path)


def _run_e01_postprocess_pipeline(
    demo_dir: str | Path,
    *,
    exp_name: str = "e01_fold_garment",
    sim_backend: str,
    ipc_env_value: str,
    arm_env_value: str,
    video_fps: int = 30,
    background: bool = False,
    log_path: str = "",
) -> dict[str, object]:
    """Run IPC replay, then independent arm IK, after live collection stops.

    Both stages write explicit validity fields.  A failed IPC stage does not
    prevent the independent arm audit from running, but the aggregate pipeline
    remains failed and the recorder exits unsuccessfully after all queued
    episodes have been checked.
    """
    demo_path = Path(demo_dir).expanduser().resolve()
    run_ipc = _should_auto_replay_e01_ipc(
        exp_name,
        sim_backend=sim_backend,
        env_value=ipc_env_value,
    )
    run_arm = _should_auto_compute_e01_arm_plan(
        exp_name,
        sim_backend=sim_backend,
        env_value=arm_env_value,
    )
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )
    document: dict[str, object] = {
        "format": "handson.e01_postprocess",
        "version": 1,
        "status": "running" if run_ipc or run_arm else "not_requested",
        "started_at": started_at,
        "completed_at": None,
        "background": bool(background),
        "exp_name": exp_name,
        "sim_backend": sim_backend,
        "video_fps": max(1, int(video_fps)),
        "log_path": str(log_path or ""),
        "quality_profile": (
            "f01_high_quality_ipc"
            if exp_name == _F01_COTTON_CLOTH_MVP_TASK and run_ipc
            else "default"
        ),
        "stages": {},
    }
    _write_e01_postprocess_document(demo_path, document)
    if not run_ipc and not run_arm:
        document["completed_at"] = started_at
        _write_e01_postprocess_document(demo_path, document)
        return document

    scripts_dir = Path(__file__).resolve().parent
    env_vars = os.environ.copy()
    env_vars.setdefault("PYTHONPATH", str(_PACKAGE_ROOT.parent))
    env_vars.update(configure_gpu_environment(env_vars, render_mode="offscreen"))
    commands: list[tuple[str, list[str]]] = []
    if run_ipc:
        ipc_python = _e01_offline_genesis_python(
            env_value=os.environ.get("HANDSON_GENESIS_PYTHON", ""),
            fallback=sys.executable,
        )
        ipc_command = [
            ipc_python,
            str(scripts_dir / "e01_replay_ipc_cloth.py"),
            str(demo_path),
            "--overwrite",
            "--settle-steps",
            _e01_ipc_replay_default(
                exp_name,
                "HANDSON_E01_IPC_REPLAY_SETTLE_STEPS",
                "30",
            ),
            "--physics-dt",
            _e01_ipc_replay_default(
                exp_name,
                "HANDSON_E01_IPC_REPLAY_PHYSICS_DT",
                str(1.0 / 60.0),
            ),
            "--newton-max-iterations",
            _e01_ipc_replay_default(
                exp_name,
                "HANDSON_E01_IPC_REPLAY_NEWTON_MAX_ITERATIONS",
                "8",
            ),
            "--max-node-step",
            _e01_ipc_replay_default(
                exp_name,
                "HANDSON_E01_IPC_REPLAY_MAX_NODE_STEP",
                "0.05",
            ),
            "--max-tcp-position-error",
            _e01_ipc_replay_default(
                exp_name,
                "HANDSON_E01_IPC_REPLAY_MAX_TCP_POSITION_ERROR",
                "0.003",
            ),
            "--max-tcp-orientation-error",
            _e01_ipc_replay_default(
                exp_name,
                "HANDSON_E01_IPC_REPLAY_MAX_TCP_ORIENTATION_ERROR",
                "0.03",
            ),
        ]
        if os.environ.get(
            "HANDSON_E01_IPC_REPLAY_KEYPOINTS_ONLY", ""
        ).strip().lower() in {"1", "true", "yes", "on"}:
            ipc_command.append("--keypoints-only")
        commands.append(("ipc_replay", ipc_command))
        if os.environ.get(
            "HANDSON_E01_AUTO_IPC_RENDER", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}:
            commands.append(
                (
                    "ipc_render",
                    [
                        sys.executable,
                        str(scripts_dir / "e01_render_ipc_replay.py"),
                        str(demo_path),
                        "--fps",
                        str(max(1, int(video_fps))),
                    ],
                )
            )
    if run_arm:
        planner_python = _e01_offline_arm_python(
            env_value=os.environ.get(
                "HANDSON_E01_ARM_PLAN_PYTHON",
                os.environ.get("HANDSON_PYTHON", ""),
            ),
            fallback=sys.executable,
        )
        commands.append(
            (
                "arm_ik",
                [
                    planner_python,
                    str(scripts_dir / "e01_compute_arm_plan_qpos.py"),
                    str(demo_path),
                    "--overwrite",
                    "--max-fk-position-error",
                    str(FLAGS.c01_arm_max_fk_position_error),
                    "--max-fk-orientation-error",
                    str(FLAGS.c01_arm_max_fk_orientation_error),
                    "--collision-penetration-tolerance",
                    str(FLAGS.c01_arm_collision_penetration_tolerance),
                    "--collision-sweep-joint-step",
                    str(FLAGS.c01_arm_collision_sweep_joint_step),
                    "--table-delta-from-state-101",
                ],
            )
        )

    stages = document["stages"]
    assert isinstance(stages, dict)
    for stage_name, command in commands:
        stage_started_at = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(timespec="seconds")
        if (
            stage_name == "ipc_render"
            and isinstance(stages.get("ipc_replay"), Mapping)
            and stages["ipc_replay"].get("status") != "complete"
        ):
            stages[stage_name] = {
                "status": "skipped",
                "started_at": stage_started_at,
                "completed_at": stage_started_at,
                "returncode": None,
                "command": command,
                "output_tail": "Skipped because IPC replay did not complete.",
            }
            _write_e01_postprocess_document(demo_path, document)
            continue
        stages[stage_name] = {
            "status": "running",
            "started_at": stage_started_at,
            "completed_at": None,
            "returncode": None,
            "command": command,
            "output_tail": "",
        }
        _write_e01_postprocess_document(demo_path, document)
        print(
            f"[E01 postprocess/{stage_name}] starting: {' '.join(command)}",
            flush=True,
        )
        try:
            stage_env_vars = (
                _e01_ipc_subprocess_environment(env_vars, exp_name=exp_name)
                if stage_name == "ipc_replay"
                else env_vars
            )
            result = subprocess.run(
                command,
                env=stage_env_vars,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            output = result.stdout or ""
            if output:
                print(output, end="" if output.endswith("\n") else "\n")
            success = result.returncode == 0
            stages[stage_name].update(
                {
                    "status": "complete" if success else "failed",
                    "completed_at": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(timespec="seconds"),
                    "returncode": int(result.returncode),
                    "output_tail": output[-8000:],
                }
            )
        except Exception as exc:
            stages[stage_name].update(
                {
                    "status": "failed",
                    "completed_at": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(timespec="seconds"),
                    "returncode": None,
                    "output_tail": f"{type(exc).__name__}: {exc}",
                }
            )
        _write_e01_postprocess_document(demo_path, document)

    failed_stages = sorted(
        name
        for name, result in stages.items()
        if isinstance(result, Mapping)
        and result.get("status") not in {"complete", "skipped"}
    )
    document["status"] = "failed" if failed_stages else "complete"
    document["failed_stages"] = failed_stages
    document["completed_at"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat(timespec="seconds")
    _write_e01_postprocess_document(demo_path, document)
    return document


def _run_isaac_deferred_arm_replay(
    demo_dir: str | Path,
    *,
    exp_name: str,
    video_fps: int,
) -> subprocess.CompletedProcess[str]:
    """Render complete robots after the live detached-hand Isaac app closes."""
    demo_path = Path(demo_dir).expanduser().resolve()
    renderer = Path(__file__).resolve().parent / "isaac_render_arm_replay.py"
    command = [
        sys.executable,
        str(renderer),
        f"--arm_replay_demo_dir={demo_path}",
        f"--exp_name={exp_name}",
        f"--arm_replay_fps={max(1, int(video_fps))}",
    ]
    env_vars = os.environ.copy()
    env_vars["HANDSON_ISAACLAB_DEFER_ARM_RENDER"] = "0"
    for key in _ISAAC_DEFERRED_COLLECTION_ONLY_ENV_KEYS:
        env_vars.pop(key, None)
    env_vars = configure_isaac_offline_render_environment(env_vars)
    env_vars.setdefault("PYTHONPATH", str(_PACKAGE_ROOT.parent))
    configured_env = configure_isaac_gpu_environment(
        env_vars,
        render_mode="offscreen",
    )
    if "CUDA_VISIBLE_DEVICES" not in configured_env:
        env_vars.pop("CUDA_VISIBLE_DEVICES", None)
    env_vars.update(configured_env)
    print(
        f"[Isaac deferred arm replay] starting: {' '.join(command)}",
        flush=True,
    )
    result = subprocess.run(
        command,
        env=env_vars,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(
            result.stdout,
            end="" if result.stdout.endswith("\n") else "\n",
        )
    return result


def _write_demo_zarr_and_videos(
    trajectory,
    exp_name: str,
    success_index: int,
    base_out: Path,
    video_fps: int,
    env=None,
    environment_schema=None,
    required_recording_fields=None,
    video_spool: RecordingSpool | None = None,
    collection_run_seed: int | None = None,
    episode_reset_seed: int | None = None,
    collection_seed_source: str | None = None,
    episode_reset_ordinal: int | None = None,
    manus_raw_events: list[dict[str, object]] | None = None,
    manus_raw_capture_summary: Mapping[str, object] | None = None,
    online_camera_capture_scope: str | None = None,
    guidance_route_only: bool = False,
):
    """Build and publish one demo or Space-delimited guidance route."""
    try:
        sim_backend = str(FLAGS.sim_backend)
    except (AttributeError, flags.UnparsedFlagAccessError):
        sim_backend = "mujoco"
    if online_camera_capture_scope is None:
        online_camera_capture_scope = _runtime_online_camera_capture_scope(
            exp_name,
            sim_backend,
        )
    else:
        online_camera_capture_scope = _validated_online_camera_capture_scope(
            online_camera_capture_scope
        )
    camera_postrender_required = bool(
        online_camera_capture_scope in ONLINE_CAMERA_POSTRENDER_SCOPES
    )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output_root = Path(base_out).expanduser().resolve()
    dataset_generation = _dataset_generation_for_env(env)
    final_demo_dir = output_root / (
        f"{exp_name}_demo_{success_index}_{timestamp}"
        f"{dataset_generation.directory_suffix}"
    )
    deferred_publication = bool(
        _deferred_physical_replay_certification_enabled()
        and sim_backend == "isaacsim"
        and bool(getattr(FLAGS, "record_physx_ground_truth", False))
    )
    destination_dir = final_demo_dir
    if deferred_publication:
        pending_parent = _physical_replay_pending_parent(output_root)
        pending_parent.mkdir(parents=True, exist_ok=True)
        destination_dir = pending_parent / final_demo_dir.name
    # A mount point's parent can be a different filesystem, so the transaction
    # stays inside out_dir for a guaranteed atomic rename. CollectionState
    # explicitly excludes this directory from review and upload scans.
    staging_parent = _recording_staging_parent(output_root)
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{final_demo_dir.name}.partial-",
            dir=str(staging_parent),
        )
    )
    published = False
    try:
        _write_demo_contents(
            trajectory,
            exp_name,
            success_index,
            staging_dir,
            video_fps,
            env=env,
            environment_schema=environment_schema,
            required_recording_fields=required_recording_fields,
            video_spool=video_spool,
            collection_run_seed=collection_run_seed,
            episode_reset_seed=episode_reset_seed,
            collection_seed_source=collection_seed_source,
            episode_reset_ordinal=episode_reset_ordinal,
            dataset_generation=dataset_generation,
            published_demo_name=final_demo_dir.name,
            manus_raw_events=manus_raw_events,
            manus_raw_capture_summary=manus_raw_capture_summary,
            online_camera_capture_scope=online_camera_capture_scope,
            guidance_route_only=guidance_route_only,
        )

        required_paths = [staging_dir / "replay.zarr"]
        if not camera_postrender_required:
            required_paths.extend(
                staging_dir / "videos" / f"{key}.mp4"
                for key in CANONICAL_CAMERA_KEYS
            )
        if manus_raw_events is not None:
            required_paths.append(staging_dir / MANUS_RAW_SCHEMA_FILE)
        if FLAGS.save_camera_info and not camera_postrender_required:
            required_paths.extend(
                [
                    staging_dir / "videos" / "camera_info.json",
                    *(
                        staging_dir / "videos" / f"{key}_camera_info.json"
                        for key in CANONICAL_CAMERA_KEYS
                    ),
                ]
            )
        if FLAGS.save_depth and not camera_postrender_required:
            required_paths.extend(
                path
                for key in CANONICAL_CAMERA_KEYS
                for path in (
                    staging_dir / "videos" / f"{key}_depth.npz",
                    staging_dir / "videos" / f"{key}_depth.mp4",
                )
            )
        if (
            dataset_generation.is_stable_v2
            or dataset_generation.is_newton_v1
            or dataset_generation.is_newton_cross_gpu_v2
        ):
            required_paths.append(staging_dir / dataset_generation.marker_file)
        invalid = []
        for path in required_paths:
            if not path.exists():
                invalid.append(f"{path} (missing)")
            elif path.is_file() and path.stat().st_size <= 0:
                invalid.append(f"{path} (empty)")
        if invalid:
            raise RuntimeError(
                "Refusing to publish incomplete demo; invalid artifacts: "
                + ", ".join(invalid)
            )
        if final_demo_dir.exists():
            raise FileExistsError(
                f"Refusing to replace existing demo: {final_demo_dir}"
            )
        if destination_dir.exists():
            raise FileExistsError(
                f"Refusing to replace existing candidate: {destination_dir}"
            )
        os.replace(staging_dir, destination_dir)
        published = True
        return str(destination_dir)
    finally:
        if not published:
            shutil.rmtree(staging_dir, ignore_errors=True)
        try:
            staging_parent.rmdir()
        except OSError:
            pass


def _has_displayable_wrist_images(obs) -> bool:
    if not isinstance(obs, dict):
        return False
    images = obs.get("images")
    for key in ("wrist", "wrist_left", "wrist_right"):
        value = obs.get(key)
        if value is None and isinstance(images, dict):
            value = images.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] in (1, 3):
            return True
    return False


def _has_displayable_sim_images(obs, *, include_main: bool) -> bool:
    if include_main and isinstance(obs, dict):
        images = obs.get("images")
        for key in ("front", "random_camera", "ego"):
            value = obs.get(key)
            if value is None and isinstance(images, dict):
                value = images.get(key)
            if value is not None and _normalize_frame(value) is not None:
                return True
    return _has_displayable_wrist_images(obs)


def _live_preview_path() -> Path | None:
    if os.environ.get(
        "HANDSON_ISAACLAB_DEFER_CAMERA_RENDER",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}:
        # Offline camera mode intentionally has no current camera frame.
        # Avoid copying/PNG-encoding the synthetic recorder fallback every
        # control step and do not leave a stale image in the web monitor.
        return None
    raw = os.environ.get("HANDSON_LIVE_PREVIEW_PATH", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _live_preview_stride(sim_backend: str, task_id: str) -> int:
    """Return the wall-display publish stride without changing recording FPS.

    B03 Newton's RTX viewport cannot show its dynamic bodies, so the
    file-backed NewtonWarp monitor is the operator's only valid scene view.
    Publishing it every third control step turned an already-slow 7-8 Hz
    simulation into a roughly 2.5 FPS monitor.  Keep other backends on the
    existing low-I/O cadence unless explicitly overridden.
    """

    if _async_live_preview_enabled(sim_backend, task_id):
        # This is the only correct operator view for B03 Newton, so a stale
        # shell override must not silently restore the old 2.5 FPS monitor.
        return 1
    default = 3
    raw = os.environ.get("HANDSON_LIVE_PREVIEW_STRIDE", "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if 1 <= value <= 120 else default


def _async_live_preview_enabled(sim_backend: str, task_id: str) -> bool:
    """Keep Newton's live PNG encoder off the collection control thread."""

    return bool(
        str(sim_backend) == "isaacsim_newton"
        and str(task_id) == "t09_insert_peg"
    )


def _sim_camera_viewer_includes_main(
    *,
    sim_backend: str,
    task_id: str,
    preview_only: bool,
) -> bool:
    """Return whether the operator viewer must include the main camera."""

    return bool(
        str(sim_backend) == "genesis"
        or str(task_id) == "t08_hammer_nail"
        or (
            str(sim_backend) == "isaacsim_newton"
            and str(task_id) == "t09_insert_peg"
        )
    )


def _launch_external_live_preview(
    preview_path: Path | None,
    *,
    title: str = "HandsOn 实时仿真相机",
):
    """Open the file-backed live camera when OpenCV has no GUI backend.

    Isaac Sim installations commonly bundle a headless OpenCV build.  The
    recorder still writes an atomically replaced PNG in that case, so a small
    always-on-top viewer can refresh it without involving the Isaac/RTX
    viewport (which cannot display NewtonWarp bodies).
    """

    if preview_path is None or not preview_path.is_file():
        return None
    if not os.environ.get("DISPLAY"):
        return None
    viewer_script = Path(__file__).with_name("live_camera_viewer.py")
    system_python = Path("/usr/bin/python3")
    command = None
    if system_python.is_file() and viewer_script.is_file():
        command = [
            str(system_python),
            str(viewer_script),
            "--image",
            str(preview_path),
            "--parent-pid",
            str(os.getpid()),
            "--title",
            title,
        ]
    else:
        executable = shutil.which("eog")
        if executable is not None:
            command = [executable, "--new-instance", str(preview_path)]
    if command is None:
        return None
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    except OSError as exc:
        print(
            "[Recorder] Failed to open the external live camera viewer: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return None
    print(
        "[Recorder] OpenCV HighGUI is unavailable; opened the always-on-top "
        "NewtonWarp "
        f"live camera in {preview_path.name} (pid {process.pid}).",
        flush=True,
    )
    return process


class ManualResetSignalBridge:
    """Turn SIGUSR1 into a reset request consumed by the recorder loop.

    Signal handlers must not touch the simulator or recording buffers.  The
    handler only sets a thread-safe flag; the main loop performs the complete
    episode rejection/reset transaction at its next safe boundary.
    """

    def __init__(self) -> None:
        self.requested = threading.Event()
        self.signum = getattr(signal, "SIGUSR1", None)
        self.previous_handler = (
            signal.getsignal(self.signum) if self.signum is not None else None
        )

    def _handle(self, _signum, _frame) -> None:
        self.requested.set()

    def install(self) -> None:
        if self.signum is not None:
            signal.signal(self.signum, self._handle)

    def consume(self) -> bool:
        if not self.requested.is_set():
            return False
        self.requested.clear()
        return True

    def discard_pending(self) -> None:
        self.requested.clear()

    def restore(self) -> None:
        if self.signum is not None:
            signal.signal(self.signum, self.previous_handler)


def _close_external_live_preview(process) -> None:
    if process is None:
        return
    try:
        if process.poll() is not None:
            return
        process.terminate()
        process.wait(timeout=2.0)
    except (OSError, subprocess.SubprocessError):
        try:
            process.kill()
        except OSError:
            pass


def _normalize_frame(image) -> np.ndarray | None:
    if image is None:
        return None

    arr = np.asarray(image)
    if arr.ndim == 4:
        # Observation images are usually unbatched or single-batched. If a
        # stack arrives, use the first frame for the live preview.
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]

    frame = _safe_squeeze_image(arr)
    if frame is None:
        return None
    if frame.ndim != 3:
        return None
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.shape[-1] != 3:
        return None
    return np.ascontiguousarray(frame)


def _preview_frame_and_key_from_observation(
    obs,
) -> tuple[str, np.ndarray] | None:
    if not isinstance(obs, dict):
        return None

    candidates = []
    images = obs.get("images")
    if isinstance(images, dict):
        candidates.extend(
            (key, images.get(key))
            for key in (
                "main",
                "head_main",
                "front",
                "random_camera",
                "ego",
                "wrist",
                "ego_left",
                "ego_right",
            )
        )
    candidates.extend(
        (key, obs.get(key))
        for key in (
            "main",
            "head_main",
            "front",
            "random_camera",
            "ego",
            "wrist",
            "wrist_left",
            "wrist_right",
        )
    )

    for key, candidate in candidates:
        if candidate is None:
            continue
        frame = _normalize_frame(candidate)
        if frame is not None:
            return key, frame
    return None


def _preview_frame_from_observation(obs) -> np.ndarray | None:
    preview = _preview_frame_and_key_from_observation(obs)
    if preview is None:
        return None
    return preview[1]


def _write_live_preview(obs, preview_path: Path | None, *, env=None) -> None:
    if preview_path is None:
        return
    preview = _preview_frame_and_key_from_observation(obs)
    if preview is None:
        return
    camera_key, frame = preview
    _write_live_preview_frame(camera_key, frame, preview_path, env=env)


class LatestFramePreviewPublisher:
    """Encode the newest submitted live frame without blocking collection.

    The slot intentionally has capacity one: if Newton produces another frame
    while PNG encoding is still running, the stale pending frame is replaced.
    The recorder therefore never builds an I/O backlog and the monitor catches
    up to the newest simulator state as soon as the encoder is available.
    """

    def __init__(self, preview_path: Path) -> None:
        self.preview_path = Path(preview_path)
        self.preview_path.parent.mkdir(parents=True, exist_ok=True)
        stale_tmp_path = self.preview_path.with_name(
            f"{self.preview_path.stem}.tmp{self.preview_path.suffix}"
        )
        try:
            stale_tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        self._condition = threading.Condition()
        self._latest: tuple[str, np.ndarray] | None = None
        self._closed = False
        self._worker_error_reported = False
        self._thread = threading.Thread(
            target=self._worker,
            name="handson-live-preview",
            daemon=True,
        )
        self._thread.start()

    def submit(self, obs) -> bool:
        preview = _preview_frame_and_key_from_observation(obs)
        if preview is None:
            return False
        camera_key, frame = preview
        # Camera observations may reuse their backing storage on the next
        # simulator step.  Copy before returning control to the recorder.
        owned_frame = np.array(frame, copy=True, order="C")
        with self._condition:
            if self._closed:
                return False
            self._latest = (camera_key, owned_frame)
            self._condition.notify()
        return True

    def _worker(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._latest is not None or self._closed
                )
                if self._latest is None:
                    return
                camera_key, frame = self._latest
                self._latest = None
            try:
                _write_live_preview_frame(
                    camera_key,
                    frame,
                    self.preview_path,
                )
            except Exception as exc:  # noqa: BLE001
                if not self._worker_error_reported:
                    print(
                        "[Recorder][WARN] Live preview publisher failed; "
                        f"continuing collection: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    self._worker_error_reported = True

    def close(self, *, timeout: float = 2.0) -> None:
        with self._condition:
            if not self._closed:
                self._closed = True
                self._condition.notify()
        self._thread.join(timeout=max(0.0, float(timeout)))


def _write_live_preview_frame(
    camera_key: str,
    frame: np.ndarray,
    preview_path: Path,
    *,
    env=None,
) -> None:
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = preview_path.with_name(f"{preview_path.stem}.tmp{preview_path.suffix}")
    try:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if env is not None:
            bgr = WristCameraViewer._draw_preview_overlay(bgr, camera_key, env)
        ok = cv2.imwrite(
            str(tmp_path),
            bgr,
            [cv2.IMWRITE_PNG_COMPRESSION, 1],
        )
        if ok:
            os.replace(tmp_path, preview_path)
    except (cv2.error, OSError):
        return
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _auto_start_teleop_enabled() -> bool:
    return os.environ.get("HANDSON_AUTO_START_TELEOP", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _find_teleop_wrapper(env):
    current = env
    for _ in range(16):
        if current is None:
            break
        if callable(getattr(current, "set_intervened", None)):
            return current
        current = getattr(current, "env", None)
    return None


def _enable_isaac_fast_shutdown(env) -> bool:
    """Mark the native Isaac environment for non-blocking Kit teardown.

    RTX camera capture can leave ``SimulationContext.stop()`` waiting forever
    after a recorder preview or an interrupted collection.  The native envs
    already implement the guarded ``_fast_shutdown_on_close`` path; the
    recorder must opt into it for every Isaac collection, not only the
    deferred-arm profiles that set it during construction.  Walk through Gym
    wrappers so this also works for the teleop/observation adapters.
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


def _request_c08_keyboard_grasp(env) -> bool:
    """Call the legacy C08 Space hook, which is now intentionally a no-op."""
    current = env
    for _ in range(16):
        if current is None:
            break
        callback = getattr(current, "request_c08_keyboard_grasp", None)
        if callable(callback):
            return bool(callback())
        current = getattr(current, "env", None)
    return False


def _request_guidance_route_completion(env) -> bool:
    """Queue a Space route delimiter from either the sim or camera window."""

    current = env
    for _ in range(16):
        if current is None:
            break
        if hasattr(current, "_guidance_route_complete_requested"):
            try:
                setattr(current, "_guidance_route_complete_requested", True)
                return True
            except Exception:
                return False
        current = getattr(current, "env", None)
    return False


def _maybe_auto_start_teleop(env, *, immediate: bool = False) -> None:
    if not _auto_start_teleop_enabled():
        return
    wrapper = _find_teleop_wrapper(env)
    if wrapper is None or bool(getattr(wrapper, "intervened", False)):
        return
    try:
        wrapper.set_intervened(True, start_delay=not immediate)
    except Exception as exc:
        print(f"Warning: auto-start teleop failed: {exc}", flush=True)


def _reset_recording_episode(env, reset_kwargs: Mapping[str, object]):
    """Repeat an immediate keyboard reset with the same recorded seed."""

    kwargs = dict(reset_kwargs)
    result = env.reset(**kwargs)
    reset_info = result[1] if isinstance(result, tuple) and len(result) == 2 else {}
    if isinstance(reset_info, Mapping) and reset_info.get(
        "manual_reset_applied",
        False,
    ):
        _fast_exit_fresh_worker_for_rejected_attempt(
            "manual_reset_applied_during_initial_reset",
            "the scene cannot be reset a second time in a fresh attempt",
        )
        result = env.reset(**kwargs)
        repeated_info = (
            result[1]
            if isinstance(result, tuple) and len(result) == 2
            else {}
        )
        if isinstance(repeated_info, Mapping) and repeated_info.get(
            "manual_reset_applied",
            False,
        ):
            raise RuntimeError(
                "Environment retained manual_reset_applied after a "
                "same-seed full reset."
            )
        if isinstance(result, tuple) and len(result) == 2:
            repeated_info = dict(result[1] or {})
            repeated_info["_recording_reset_call_count"] = 2
            repeated_info["_manual_reset_applied_recovery"] = True
            result = (result[0], repeated_info)
    return result


def _recording_sim_time(raw_env, *, sim_backend: str) -> float | None:
    """Read the backend-specific simulator time from a recording trace."""

    field_name = {
        "isaacsim": "physx_sim_time",
        "isaacsim_newton": "newton_sim_time",
    }.get(str(sim_backend))
    if field_name is None:
        return None
    trace = getattr(raw_env, "_physx_ground_truth_trace", None)
    snapshot = getattr(trace, "snapshot", None)
    if not callable(snapshot):
        return None
    try:
        payload = snapshot()
        if not isinstance(payload, Mapping):
            return None
        raw_value = payload.get(field_name)
        if raw_value is None:
            return None
        value = np.asarray(raw_value, dtype=np.float64).reshape(-1)
        if value.size != 1 or not np.isfinite(value[0]):
            return None
        return float(value[0])
    except Exception:
        return None


def _reset_recording_episode_with_stable_origin(
    env,
    reset_kwargs: Mapping[str, object],
    *,
    sim_backend: str,
    task_id: str,
    collection_run_seed: int,
    reset_attempt_index: int,
    seed_resets: bool,
    collection_seed_source: str = "recorded_collection_seed",
    reset_attempt_observer=None,
    stabilization_progress_observer=None,
):
    """Reset, naturally settle, and resample a bad physical origin."""

    is_isaac = sim_backend == "isaacsim"
    raw_env = getattr(env, "unwrapped", env)
    reset_task_name = getattr(raw_env, "task_name", task_id)
    attempts_allowed = (
        stable_recording_origin_reset_attempts_for_task(reset_task_name)
        if is_isaac
        else 1
    )

    def current_recording_sim_time() -> float | None:
        if reset_attempt_observer is None:
            return None
        return _recording_sim_time(raw_env, sim_backend=sim_backend)

    for attempt in range(1, attempts_allowed + 1):
        kwargs = dict(reset_kwargs)
        episode_reset_seed = None
        episode_reset_ordinal = int(reset_attempt_index)
        reset_requested_at = _now_iso()
        stable_kwargs = (
            stable_recording_origin_kwargs_for_task(reset_task_name)
            if is_isaac
            else {}
        )
        if seed_resets:
            episode_reset_seed = int(
                (collection_run_seed + reset_attempt_index) % (2**31)
            )
            kwargs["seed"] = episode_reset_seed
        reset_attempt_index += 1
        scene_rebuilt_before_reset = False
        # ``None`` is the canonical evidence for an ordinary reset that did
        # not rebuild the scene.  An empty mapping falsely claims that a
        # rebuild report exists and conflicts with the publication gate's
        # no-rebuild contract.
        scene_rebuild_report: dict[str, object] | None = None
        if (
            is_isaac
            and episode_reset_ordinal > 0
            and _isaac_in_process_hard_reset_enabled_for_task(reset_task_name)
        ):
            hard_reset = getattr(raw_env, "hard_reset_physx_scene", None)
            if not callable(hard_reset):
                raise RuntimeError(
                    "In-process Isaac hard reset was requested, but the "
                    f"environment {type(raw_env).__name__} does not expose "
                    "hard_reset_physx_scene()."
                )
            if episode_reset_seed is None:
                raise RuntimeError(
                    "In-process Isaac hard reset requires an explicit episode seed."
                )
            print(
                "[Recorder] Hard reset: releasing the previous PhysX Scene "
                "and rebuilding every task object/configuration in the same "
                f"Isaac process (reset={episode_reset_ordinal}, "
                f"seed={episode_reset_seed}).",
                flush=True,
            )
            raw_report = hard_reset(seed=int(episode_reset_seed))
            if not isinstance(raw_report, Mapping) or (
                raw_report.get("scene_rebuilt") is not True
            ):
                raise RuntimeError(
                    "In-process Isaac hard reset did not confirm a fresh scene."
                )
            scene_rebuild_report = dict(raw_report)
            scene_rebuilt_before_reset = True
            print(
                "[Recorder] Hard reset complete: fresh PhysX Scene generation="
                f"{scene_rebuild_report.get('scene_generation')}; waiting for "
                "the stable recording origin.",
                flush=True,
            )
        scene_generation = int(
            (scene_rebuild_report or {}).get(
                "scene_generation",
                getattr(raw_env, "_physx_scene_generation", 0),
            )
        )
        attempt_start_sim_time = current_recording_sim_time()
        observation, reset_info = _reset_recording_episode(env, kwargs)
        if (
            reset_attempt_observer is not None
            and isinstance(reset_info, Mapping)
            and int(reset_info.get("_recording_reset_call_count", 1)) != 1
        ):
            raise RuntimeError(
                "A legacy manual_reset_applied path performed more than one "
                "env.reset call inside one reset attempt. Exact persistent "
                "PhysX session history cannot represent that hidden prefix; "
                "collection stopped before another demo could be accepted."
            )
        if not is_isaac:
            return (
                observation,
                reset_info,
                reset_attempt_index,
                episode_reset_seed,
            )

        try:
            update_panel = getattr(raw_env, "set_recording_origin_progress", None)

            def report_stabilization_progress(progress):
                payload = {
                    **progress,
                    "attempt": attempt,
                    "attempts_allowed": attempts_allowed,
                }
                if callable(update_panel):
                    update_panel(payload)
                if stabilization_progress_observer is not None:
                    stabilization_progress_observer(payload)

            observation, reset_info = stabilize_recording_origin(
                env,
                observation,
                reset_info,
                progress_callback=(
                    report_stabilization_progress
                    if (
                        stabilization_progress_observer is not None
                        or callable(update_panel)
                    )
                    else None
                ),
                **stable_kwargs,
            )
            if callable(update_panel):
                update_panel(None)
        except RecordingOriginStabilityError as exc:
            attempt_end_sim_time = current_recording_sim_time()
            failure_diagnostics = getattr(exc, "diagnostics", {})
            if not isinstance(failure_diagnostics, Mapping):
                failure_diagnostics = {}
            if reset_attempt_observer is not None:
                reset_attempt_observer(
                    {
                        "outcome": "rejected_unstable_origin",
                        "attempt": int(attempt),
                        "attempts_allowed": int(attempts_allowed),
                        "episode_reset_ordinal": episode_reset_ordinal,
                        "episode_reset_seed": episode_reset_seed,
                        "episode_reset_requested_at": reset_requested_at,
                        "scene_rebuilt_before_reset": (
                            scene_rebuilt_before_reset
                        ),
                        "scene_generation": scene_generation,
                        "scene_rebuild_report": copy.deepcopy(
                            scene_rebuild_report
                        ),
                        "stable_origin_kwargs": copy.deepcopy(stable_kwargs),
                        "start_sim_time": attempt_start_sim_time,
                        "end_sim_time": attempt_end_sim_time,
                        "stability_diagnostics": copy.deepcopy(
                            dict(failure_diagnostics)
                        ),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            if attempt >= attempts_allowed:
                _fast_exit_fresh_worker_after_unstable_origin(exc)
                raise
            print(
                "[Recorder] Physical origin did not settle; resampling "
                f"the reset pose ({attempt}/{attempts_allowed}): {exc}; "
                "diagnostics="
                f"{{'failure_phase': {failure_diagnostics.get('failure_phase')!r}, "
                "'gate_final_pose_delta': "
                f"{failure_diagnostics.get('gate_final_pose_delta')!r}, "
                "'gate_final_rotation_readback_floor': "
                f"{failure_diagnostics.get('gate_final_rotation_readback_floor')!r}, "
                "'gate_final_drifts': "
                f"{failure_diagnostics.get('gate_final_drifts')!r}}}",
                flush=True,
            )
            continue

        reset_info = dict(reset_info or {})
        origin = dict(reset_info.get("physical_recording_origin", {}))
        origin.update(
            {
                "reset_stabilization_attempt": attempt,
                "reset_stabilization_attempts_allowed": attempts_allowed,
                "collection_run_seed": int(collection_run_seed),
                "collection_seed_source": str(collection_seed_source),
                "episode_reset_ordinal": episode_reset_ordinal,
                "episode_reset_seed": episode_reset_seed,
                "episode_seed_derivation": (
                    EPISODE_SEED_DERIVATION if seed_resets else None
                ),
                "episode_reset_requested_at": reset_requested_at,
                "scene_rebuilt_before_reset": scene_rebuilt_before_reset,
                "scene_generation": scene_generation,
                "scene_rebuild_report": copy.deepcopy(scene_rebuild_report),
            }
        )
        reset_info["physical_recording_origin"] = origin
        raw_env._physical_recording_origin = dict(origin)
        attempt_end_sim_time = current_recording_sim_time()
        if reset_attempt_observer is not None:
            reset_attempt_observer(
                {
                    "outcome": "accepted",
                    "attempt": int(attempt),
                    "attempts_allowed": int(attempts_allowed),
                    "episode_reset_ordinal": episode_reset_ordinal,
                    "episode_reset_seed": episode_reset_seed,
                    "episode_reset_requested_at": reset_requested_at,
                    "scene_rebuilt_before_reset": (
                        scene_rebuilt_before_reset
                    ),
                    "scene_generation": scene_generation,
                    "scene_rebuild_report": copy.deepcopy(
                        scene_rebuild_report
                    ),
                    "stable_origin_kwargs": copy.deepcopy(stable_kwargs),
                    "stable_recording_origin": copy.deepcopy(origin),
                    "start_sim_time": attempt_start_sim_time,
                    "end_sim_time": attempt_end_sim_time,
                }
            )
        return (
            observation,
            reset_info,
            reset_attempt_index,
            episode_reset_seed,
        )

    raise AssertionError("recording-origin retry loop exhausted unexpectedly")


def _fast_exit_fresh_worker_after_unstable_origin(
    error: RecordingOriginStabilityError,
) -> None:
    """Immediately return an unstable fresh attempt to its supervisor.

    Isaac/Kit teardown is intentionally unsafe here.  The process has not
    published an episode, owns no artifacts that need flushing, and the
    supervisor runs every recorder attempt in its own process group.  A hard
    process exit therefore both releases the GPU resources reliably and lets
    the supervisor quarantine this attempt and start a new seed.  Ordinary
    recorder and preview invocations retain their normal exception semantics.
    """

    _fast_exit_fresh_worker_for_rejected_attempt(
        "unstable_physical_origin",
        str(error),
    )


def _fast_exit_fresh_worker_for_rejected_attempt(
    reason: str,
    detail: str = "",
) -> None:
    """Reject one fresh-scene attempt without resetting that PhysX scene."""

    if os.environ.get(_FRESH_CERTIFIED_COLLECTION_WORKER_ENV, "") != "1":
        return
    reason_text = str(reason).strip() or "unspecified_rejection"
    detail_text = str(detail).strip()
    suffix = f": {detail_text}" if detail_text else ""
    message = (
        f"[Recorder] Fresh worker rejected episode attempt "
        f"({reason_text}){suffix}. Exiting immediately without Isaac/Kit "
        "teardown; the supervisor will quarantine this attempt and retry "
        "in a new process/PhysX scene.\n"
    )
    try:
        # Bypass Python's buffered streams as well as its shutdown handlers.
        # This is best-effort diagnostics only; the exit must never depend on
        # a logging object that Kit may already have partially torn down.
        os.write(2, message.encode("utf-8", errors="replace"))
    except OSError:
        pass
    os._exit(_FRESH_WORKER_RETRYABLE_ATTEMPT_EXIT_CODE)


def _certify_saved_physx_replays(demo_dirs) -> None:
    """Run every required action replay in a fresh process and fail closed."""

    certifier = Path(__file__).resolve().parent / "certify_physx_replay.py"
    failures: list[str] = []
    deterministic_names = {
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "GARMENTLAB_ASSET_ROOT",
        "HAND",
        "MKL_NUM_THREADS",
        "NVIDIA_TF32_OVERRIDE",
        "OMP_NUM_THREADS",
        "PYTHONHASHSEED",
    }
    for demo_dir in demo_dirs:
        demo_path = Path(demo_dir).expanduser().resolve()
        report_path = demo_path / PHYSICAL_REPLAY_REPORT_NAME
        try:
            report_path.unlink(missing_ok=True)
        except OSError as exc:
            failures.append(str(demo_path))
            print(
                "[PhysX certification] could not remove stale report for "
                f"{demo_path}: {exc}",
                flush=True,
            )
            continue
        manifest = json.loads(
            (demo_path / "replay_manifest.json").read_text(encoding="utf-8")
        )
        recorded_variables = manifest.get("environment", {}).get(
            "variables",
            {},
        )
        if not isinstance(recorded_variables, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in recorded_variables.items()
        ):
            raise RuntimeError(
                f"Invalid recorded environment variables for {demo_path}."
            )
        child_env = os.environ.copy()
        for name in tuple(child_env):
            if name.startswith("HANDSON_") or name in deterministic_names:
                child_env.pop(name, None)
        child_env.update(recorded_variables)
        pythonpath = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (
                str(_PACKAGE_ROOT),
                str(_PACKAGE_ROOT.parent),
                pythonpath,
            )
            if part
        )
        command = [
            sys.executable,
            str(certifier),
            "--demo-dir",
            str(demo_path),
        ]
        print(
            f"[PhysX certification] starting: {' '.join(command)}",
            flush=True,
        )
        timeout_seconds = max(
            1,
            int(
                FLAGS[
                    "physical_replay_certification_timeout_seconds"
                ].value
            ),
        )
        try:
            result = subprocess.run(
                command,
                env=child_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            if exc.stdout:
                output = (
                    exc.stdout.decode(errors="replace")
                    if isinstance(exc.stdout, bytes)
                    else exc.stdout
                )
                print(
                    output,
                    end="" if output.endswith("\n") else "\n",
                )
            print(
                "[PhysX certification] timed out after "
                f"{timeout_seconds}s: {demo_path}",
                flush=True,
            )
            failures.append(str(demo_path))
            continue
        if result.stdout:
            print(
                result.stdout,
                end="" if result.stdout.endswith("\n") else "\n",
            )
        report_error = _physical_replay_certification_report_error(demo_path)
        if report_error is not None:
            print(
                f"[PhysX certification] invalid report for {demo_path}: "
                f"{report_error}",
                flush=True,
            )
            failures.append(str(demo_path))
        elif result.returncode != 0:
            # Isaac/Kit shutdown can end the child with a non-zero status after
            # the atomic report is already fsynced.  A fresh, fully validated
            # report is authoritative; the return code alone is not.
            print(
                "[PhysX certification] child exited with status "
                f"{result.returncode} after writing a valid passed report: "
                f"{demo_path}",
                flush=True,
            )
    if failures:
        raise RuntimeError(
            "Fresh-process PhysX replay certification failed for: "
            + ", ".join(failures)
        )


def _physical_replay_certification_report_error(
    demo_path: Path,
) -> str | None:
    """Validate the freshly generated report, including its input binding."""

    report_path = demo_path / PHYSICAL_REPLAY_REPORT_NAME
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        return f"{PHYSICAL_REPLAY_REPORT_NAME} is missing or unreadable: {exc}"
    if not isinstance(report, Mapping):
        return f"{PHYSICAL_REPLAY_REPORT_NAME} must contain a JSON object"
    if report.get("status") != "passed":
        reason = str(report.get("reason") or "").strip()
        return "certification status is not passed" + (
            f": {reason}" if reason else ""
        )
    blocker = physical_replay_certification_upload_blocker(demo_path)
    if blocker is not None:
        return str(blocker.get("detail") or blocker.get("code") or blocker)
    return None


def _enforce_fresh_process_certified_collection_entrypoint(
    *,
    preview_only: bool,
    sim_backend: str,
    record_physx_ground_truth: bool,
    certify_physical_replay: bool,
    fresh_worker_authorized: bool,
    guidance_mode: bool = False,
) -> None:
    """Restrict formal Isaac collection to approved launcher contracts."""

    formal_isaac_collection = (
        not preview_only
        and not guidance_mode
        and sim_backend == "isaacsim"
    )
    deferred_postprocess = (
        _deferred_physical_replay_certification_enabled()
    )
    if formal_isaac_collection and not record_physx_ground_truth:
        raise RuntimeError(
            "Formal Isaac collection requires "
            "--record_physx_ground_truth=true and fresh-process physical "
            "certification. Disabling the trace is permitted only for "
            "preview/debug runs that cannot enter the formal dataset."
        )
    if formal_isaac_collection and certify_physical_replay:
        raise RuntimeError(
            "Certified Isaac collection must be launched through "
            "scripts/fresh_process_certified_collection.py. The recorder "
            "worker requires HANDSON_FRESH_CERTIFIED_COLLECTION_WORKER=1 "
            "and --certify_physical_replay=false; continuing here would "
            "bypass fail-closed certification."
        )
    if formal_isaac_collection and (
        not fresh_worker_authorized and not deferred_postprocess
    ):
        raise RuntimeError(
            "Formal Isaac collection must be launched either through "
            "scripts/fresh_process_certified_collection.py or with "
            f"{DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_ENV}=1 so certification "
            "can be deferred until after collection stops; use "
            f"{POSTPROCESS_MANUAL_ONLY_ENV}=1 to queue sealed data without "
            "starting postprocessing automatically."
        )


def _as013_unsuccessful_terminal_reason(
    task_id: str,
    info: object,
) -> str:
    """Bind task-local first-terminal truth into the compact journal reason."""

    base = "unsuccessful_terminal_transition"
    if not isinstance(info, Mapping):
        return base
    if task_id == "t17_open_latched_door":
        first_terminal = str(info.get("p002_terminal_reason") or "").strip()
        return (
            f"{base}:p002_first_terminal={first_terminal}"
            if first_terminal
            else base
        )
    if task_id == "loaded_tray_through_narrow_gate":
        first_failure = str(
            info.get("failure_reason") or info.get("termination_reason") or ""
        ).strip()
        return (
            f"{base}:bc005_first_failure={first_failure}"
            if first_failure
            else base
        )
    if task_id != "t40_slide_knife_blade":
        return base
    first_failure = str(info.get("as013_failure_reason") or "").strip()
    return (
        f"{base}:as013_first_failure={first_failure}"
        if first_failure
        else base
    )


def _stable_recording_origin_required_for_run(
    task_id: str,
    sim_backend: str,
    preview_only: bool,
) -> bool:
    """Keep physical Isaac previews on the production recording-origin gate.

    Formal Isaac collection always requires a stable physical origin.  Native
    previews are the same first-reset acceptance surface: accepting
    ``physics_steps=0`` would make the preview materially less strict than the
    collection path it is meant to validate, and can expose a task as ready
    while its objects are still settling.  Keep the generic diagnostic helper
    permissive for unrelated tasks, but make every production-backed native
    preview use the strict gate.
    """

    return bool(
        str(sim_backend) == "isaacsim"
        and (
            not bool(preview_only)
            or str(task_id) in NATIVE_STABLE_PREVIEW_TASKS
        )
    )


def _publish_d20_source(task_id, source, phase):
    path = os.environ.get("HANDSON_D20_SOURCE_STATUS_FILE", "")
    if not path:
        return
    from handson.data.d20_stage_saves import write_source_status
    write_source_status(
        path, session_id=os.environ.get("HANDSON_COLLECTION_SESSION_ID", ""),
        task_name=task_id, source_mode=os.environ.get("HANDSON_D20_SOURCE_MODE", "normal"),
        source=source, phase=phase, worker_pid=os.getpid(),
    )


def _report_d20_sources_exhausted(task_id, error):
    _publish_d20_source(task_id, None, "exhausted")
    detail = f"D20 续接已停止：{error}"
    print(f"[D20] {detail}", flush=True)
    _write_isaac_worker_status("stopped", task_id=task_id, code="d20_sources_exhausted", detail=detail)
    _write_collection_runtime_status("stopped", detail, task_id=task_id, code="d20_sources_exhausted")


def main(_argv):
    task_id = FLAGS.exp_name
    from handson.sim.controllers.a03.collection_timing import CollectionOptions
    a03_options = CollectionOptions(mode=FLAGS.a03_controller_mode,
        playback_slowdown=FLAGS.a03_playback_slowdown, chunk_size=FLAGS.a03_policy_chunk_size)
    if a03_options.mode == "real_aligned":
        if (task_id, FLAGS.sim_backend, FLAGS.arm_type, FLAGS.hand_type) != (
                "t24_place_ball_in_bowl", "isaacsim", "rokae", "wuji1"):
            raise ValueError("Aligned manual collection requires A03 / IsaacSim / Rokae / Wuji1")
        if FLAGS.guidance_mode or FLAGS.policy_actions_npy or FLAGS.scripted_policy:
            raise ValueError("Aligned manual mode cannot be combined with a different action owner")
        if not FLAGS.preview_only and not FLAGS.record_physx_ground_truth:
            raise ValueError("Aligned collection requires the original per-substep PhysX replay evidence")
        if FLAGS.data_fps != 30 or FLAGS.video_fps != 30:
            raise ValueError("Aligned dense recording stays at 30 Hz; the export selects policy knots")
        if not FLAGS.preview_only:
            os.environ[DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_ENV] = "1"
    from handson.data.d20_stage_saves import (
        NoUsableSource, mark_stitched, select_source, resume_saved_stage,
        state_layout, with_source_retries,
    )
    from handson.sim.isaaclab.d20_stages import STAGE_TASKS
    d20_source_mode = os.environ.get("HANDSON_D20_SOURCE_MODE", "normal")
    try:
        d20_source = select_source(
            FLAGS.out_dir, task_id, os.environ.get("HANDSON_D20_SOURCE_EPISODE", ""),
            source_mode=d20_source_mode,
        )
    except NoUsableSource as exc:
        _report_d20_sources_exhausted(task_id, exc)
        return
    d20_speed = float(os.environ.get("HANDSON_D20_REPLAY_SPEED", "8"))
    # YCB01 needs varied ball starts for useful policy coverage.  Preserve the
    # recorder's historical global default for every other task, while making
    # an omitted flag select the task's seeded random reset.  An explicit
    # ``--randomize=False`` remains the deterministic/replay escape hatch.
    _apply_task_randomization_default(task_id, sys.argv[1:])
    guidance_mode = bool(_optional_flag_value("guidance_mode", False))
    guidance_library_root: Path | None = None
    if guidance_mode:
        if FLAGS.preview_only:
            raise ValueError("Guidance route recording cannot be combined with preview_only")
        if FLAGS.sim_backend != "isaacsim":
            raise ValueError("Guidance route recording currently requires Isaac Sim PhysX")
        if FLAGS.successes_needed != 1:
            raise ValueError("Guidance route recording accepts exactly one route")
        if FLAGS.arm_type not in {"rokae", SIMPLIFIED_POST_IK_PROFILE} or FLAGS.hand_type != "wuji1":
            raise ValueError(
                "Guidance mode requires the ROKAE profile with Wuji1 hands"
            )
        os.environ["HANDSON_TASK_GUIDANCE_RECORDING"] = "1"
        os.environ["HANDSON_TASK_GUIDANCE_ROUTE_ONLY"] = "1"
        # Guidance records only one action-aligned physical TCP readback per
        # side.  The full per-physics-substep V3 robot/body/contact trace is
        # intentionally disabled: it is not consumed by the path planner and
        # was the largest source of live hand latency.
        FLAGS.record_physx_ground_truth = False
        FLAGS.certify_physical_replay = False
        os.environ["HANDSON_POSTPROCESS_MANUAL_ONLY"] = "1"
        os.environ["HANDSON_DEFER_PHYSICAL_REPLAY_AUTOSTART"] = "0"
        # Guidance needs task physics and the native operator view, but not
        # three online video encoders.  Compact low-dimensional replay is the
        # source for the global planner.
        os.environ["HANDSON_ISAACLAB_DEFER_ARM_RENDER"] = "1"
        os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "1"
        # A hands-only guide still needs the exact task's native Isaac
        # viewport. The detached external preview is B03-specific and would
        # force every other task into a headless Kit process.
        os.environ["HANDSON_ISAACLAB_DIRECT_EXTERNAL_PREVIEW"] = "0"
        os.environ.pop("HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY", None)
        guidance_library_root = (
            Path(FLAGS.guidance_root).expanduser().resolve()
            if str(FLAGS.guidance_root).strip()
            else default_guidance_root()
        )
        os.environ["HANDSON_GUIDANCE_ROOT"] = str(guidance_library_root)
        # A direct CLI invocation follows the same machine-local layout as the
        # UI; callers cannot accidentally put raw guides in demos/ or Git.
        FLAGS.out_dir = str(task_guidance_dir(task_id, guidance_library_root))
    else:
        os.environ.pop("HANDSON_TASK_GUIDANCE_RECORDING", None)
        os.environ.pop("HANDSON_TASK_GUIDANCE_ROUTE_ONLY", None)
    scripted_as013_success = bool(
        _optional_flag_value("scripted_as013_success", False)
    )
    if (
        FLAGS.sim_backend == "isaacsim"
        and task_id == "t20_pass_and_orient_pen"
        and not FLAGS.preview_only
    ):
        os.environ[BC011_COLLECTION_ANGLE_GUIDANCE_ENV] = "1"
    else:
        os.environ.pop(BC011_COLLECTION_ANGLE_GUIDANCE_ENV, None)
    if (
        FLAGS.sim_backend == "isaacsim"
        and not FLAGS.preview_only
        and task_id in ISAACSIM_COLLECTION_UNVERIFIED_TASKS
    ):
        raise RuntimeError(
            f"{task_id} is registered for discovery but is not collection-ready; "
            "complete the Isaac 5.1 reset/teleop/camera/episode/fresh-replay/"
            "multi-seed acceptance ladder before formal recording."
        )
    if (
        not FLAGS.preview_only
        and supports_isaacsim_deferred_camera_render(
            str(FLAGS.sim_backend),
            task_id,
        )
    ):
        # Formal Isaac collection is state/action only.  Force the invariant
        # here as well as in the UI so direct CLI and supervised workers cannot
        # regress to synchronous three-camera RTX capture through a stale shell
        # override.
        os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "1"
        os.environ.pop("HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY", None)
    if (
        FLAGS.sim_backend == "isaacsim"
        and task_id == "b320_synchronized_double_cabinet_open"
    ):
        # Run before SimulationApp/Kit imports so missing USD material files
        # produce a deterministic CLI error instead of a Hydra child crash.
        validate_b320_visual_dependencies(task_id)
    if (
        FLAGS.arm_type == SIMPLIFIED_POST_IK_PROFILE
        and (
            FLAGS.sim_backend != "isaacsim"
            or task_id not in ISAACSIM_COLLECTION_TASKS
            or (
                not guidance_mode
                and task_id[:1] not in {"a", "b", "c", "d"}
            )
        )
    ):
        raise ValueError(
            "--arm_type=rokae_simplified_post_ik currently requires "
            "an ordinary A-D task, or --guidance_mode, with "
            "--sim_backend=isaacsim."
        )
    if (
        FLAGS.sim_backend == "isaacsim"
        and FLAGS.arm_type == SIMPLIFIED_POST_IK_PROFILE
    ):
        # This profile is inseparable from offline IK and full-camera
        # postrender.  Make direct CLI use as safe as the collection UI.
        os.environ["HANDSON_ISAACLAB_DEFER_ARM_RENDER"] = "1"
        os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "1"
        os.environ.pop("HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY", None)
    policy_actions = None
    policy_actions_path: Path | None = None
    policy_actions_flag = _optional_flag_value("policy_actions_npy", "")
    if policy_actions_flag:
        if task_id != "synchronized_dual_knob_setting":
            raise ValueError(
                "--policy_actions_npy is currently restricted to the "
                "synchronized_dual_knob_setting task-development validator."
            )
        if FLAGS.sim_backend != "isaacsim":
            raise ValueError("--policy_actions_npy requires --sim_backend=isaacsim")
        if FLAGS.randomize or FLAGS.randomize_dynamics:
            raise ValueError(
                "Policy-action validation requires deterministic scene and "
                "dynamics settings."
            )
        if FLAGS.successes_needed != 1:
            raise ValueError("Policy-action validation requires --successes_needed=1")
        policy_actions_path = Path(policy_actions_flag).expanduser().resolve()
        if not policy_actions_path.is_file():
            raise FileNotFoundError(policy_actions_path)
        policy_actions = np.asarray(
            np.load(policy_actions_path, allow_pickle=False), dtype=np.float32
        )
        if policy_actions.ndim != 2 or policy_actions.shape[0] < 1:
            raise ValueError(
                "Policy-action validation input must be a non-empty T x A array; "
                f"got {policy_actions.shape}."
            )
        if not np.all(np.isfinite(policy_actions)):
            raise ValueError("Policy-action validation input contains NaN/Inf.")
    if scripted_as013_success and (
        task_id
        != "t40_slide_knife_blade"
        or FLAGS.sim_backend != "isaacsim"
    ):
        raise ValueError(
            "--scripted_as013_success is valid only for the B320-AS013 "
            "IsaacSim PhysX collection task"
        )
    graceful_stop_requested = threading.Event()
    manual_reset_bridge = ManualResetSignalBridge()
    previous_stop_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def _request_graceful_stop(signum, _frame) -> None:
        del signum
        graceful_stop_requested.set()

    def _install_graceful_stop_handlers() -> None:
        for signum in previous_stop_handlers:
            signal.signal(signum, _request_graceful_stop)

    # Install once before Kit starts, then again after scene construction:
    # SimulationApp may replace process signal handlers while loading.
    _install_graceful_stop_handlers()
    manual_reset_bridge.install()
    b03_newton = bool(
        FLAGS.sim_backend == "isaacsim_newton"
        and task_id == "t09_insert_peg"
    )
    if b03_newton:
        if FLAGS.preview_only:
            os.environ[NEWTON_REPLAY_PROFILE_ENV] = NEWTON_LOCAL_DEBUG_PROFILE
        else:
            if _uses_b03_newton_direct_hand_pipeline(
                task_id,
                FLAGS.sim_backend,
            ):
                raise RuntimeError(
                    "Formal B03 Newton collection forbids direct hand/root "
                    "state writes; collect with the articulated ROKAE+Wuji "
                    "drive-target runtime."
                )
            if FLAGS.randomize_dynamics:
                raise ValueError(
                    "B03 Newton cross-GPU collection requires "
                    "--randomize_dynamics=False until randomized physical "
                    "parameters are serialized into the replay contract."
                )
            os.environ[NEWTON_REPLAY_PROFILE_ENV] = NEWTON_CROSS_GPU_PROFILE
    if FLAGS.sim_backend in ISAAC_RUNTIME_BACKENDS:
        _configure_current_isaac_process_environment()
    if FLAGS.preview_only:
        # Preview never saves an episode.  Capturing the full per-substep V3
        # trace here adds GPU readbacks both in capture and observations,
        # slowing the operator's hands despite there being no recording.
        FLAGS.record_physx_ground_truth = False
    if d20_source is not None:
        # Saved-stage previews also need the named PhysX restoration interface.
        FLAGS.record_physx_ground_truth = True
    if FLAGS.record_physx_ground_truth and FLAGS.sim_backend == "isaacsim":
        os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
    else:
        os.environ.pop("HANDSON_RECORD_PHYSX_GROUND_TRUTH", None)
    if (
        not FLAGS.preview_only
        and FLAGS.sim_backend == "isaacsim_newton"
        and task_id == "t09_insert_peg"
        and not _uses_b03_newton_direct_hand_pipeline(task_id, FLAGS.sim_backend)
    ):
        os.environ["HANDSON_RECORD_NEWTON_GROUND_TRUTH"] = "1"
    else:
        os.environ.pop("HANDSON_RECORD_NEWTON_GROUND_TRUTH", None)
    if _stable_recording_origin_required_for_run(
        task_id,
        str(FLAGS.sim_backend),
        bool(FLAGS.preview_only),
    ):
        # A formal Isaac collection must never inherit an operator/debug
        # ``HANDSON_STABLE_RECORDING_ORIGIN=0`` override.  The origin gate is
        # the reset/contact/stability proof written into the episode manifest;
        # silently accepting physics_steps=0 creates a false "Stable
        # physical frame" and lets an unverified episode reach the spool.  The
        # same rule applies to P0-02's production-backed visible preview,
        # because that preview is the task's first-reset acceptance surface.
        os.environ[STABLE_ORIGIN_ENV] = "1"
    _enforce_fresh_process_certified_collection_entrypoint(
        preview_only=bool(FLAGS.preview_only),
        guidance_mode=guidance_mode,
        sim_backend=str(FLAGS.sim_backend),
        record_physx_ground_truth=bool(FLAGS.record_physx_ground_truth),
        certify_physical_replay=bool(FLAGS.certify_physical_replay),
        fresh_worker_authorized=(
            os.environ.get(_FRESH_CERTIFIED_COLLECTION_WORKER_ENV, "") == "1"
        ),
    )
    runtime_camera_scope = _runtime_online_camera_capture_scope(
        task_id,
        str(FLAGS.sim_backend),
    )
    camera_postrender_required = bool(
        runtime_camera_scope in ONLINE_CAMERA_POSTRENDER_SCOPES
    )
    online_video_encoding_enabled = not camera_postrender_required
    if camera_postrender_required and FLAGS.save_depth:
        raise ValueError(
            "--save_depth cannot be combined with deferred canonical camera "
            "postrender."
        )
    if (
        not FLAGS.preview_only
        and camera_postrender_required
        and os.environ.get(_ISAAC_DEFERRED_ARM_WORKER_ENV, "") != "1"
    ):
        _run_isaac_deferred_collection_supervisor(
            exp_name=task_id,
            out_dir=FLAGS.out_dir,
            video_fps=FLAGS.video_fps,
            certification_timeout_seconds=int(
                FLAGS[
                    "physical_replay_certification_timeout_seconds"
                ].value
            ),
        )
        return
    if FLAGS.sim_backend == "genesis":
        if task_id not in GENESIS_COLLECTION_TASKS:
            supported = ", ".join(sorted(GENESIS_COLLECTION_TASKS))
            raise ValueError(
                "--sim_backend=genesis is currently supported only for "
                f"migrated collection tasks: {supported}."
            )
        if FLAGS.arm_type != "rokae" or FLAGS.hand_type != "wuji1":
            raise ValueError(
                "--sim_backend=genesis currently requires "
                "--arm_type=rokae --hand_type=wuji1."
            )
        if FLAGS.save_depth:
            raise ValueError(
                "--save_depth is not supported by the first Genesis E01 backend."
            )
    # One authoritative path drives render sizing, RGB/depth postprocessing,
    # replay metadata, and camera sidecars.
    os.environ["HANDSON_FISHEYE_INTRINSICS_YAML"] = str(
        Path(FLAGS.fisheye_intrinsics_yaml).expanduser()
    )

    config = CONFIG_MAPPING[task_id]()
    if FLAGS.scripted_policy:
        if task_id not in {
            "t16_open_laptop",
            "t17_open_latched_door",
            "p003_bimanual_folding_ruler_shape_configuration",
        }:
            raise ValueError(
                "--scripted_policy is currently supported only for FU015, "
                "P0-02, and P0-03."
            )
        if (
            FLAGS.sim_backend != "isaacsim"
            or FLAGS.arm_type != "rokae"
            or FLAGS.hand_type != "wuji1"
        ):
            raise ValueError(
                "Scripted collection requires --sim_backend=isaacsim "
                "--arm_type=rokae --hand_type=wuji1."
            )
    collection_run_seed = (
        int(FLAGS.seed)
        if int(FLAGS.seed) >= 0
        else int(secrets.randbits(31))
    )
    collection_seed_source = (
        EXPLICIT_COLLECTION_SEED_SOURCE
        if int(FLAGS.seed) >= 0
        else RANDOM_COLLECTION_SEED_SOURCE
    )

    env_extra_kwargs = {}
    if task_id == "bimanual_photograph":
        env_extra_kwargs["camera_screen_effect"] = FLAGS.camera_screen_effect
    elif task_id == "t23_photograph_logo":
        # C06 uses the rear screen as the operator's viewfinder and shutter
        # feedback, so it must stay live during normal collection.
        env_extra_kwargs["camera_screen_effect"] = True
    if task_id == "bimanual_unlock_ipad":
        env_extra_kwargs["ipad_screen_effect"] = FLAGS.ipad_screen_effect
    os.environ["HANDSON_ARM_TYPE"] = FLAGS.arm_type
    env_extra_kwargs["hand_type"] = FLAGS.hand_type
    if FLAGS.sim_backend == "isaacsim":
        if task_id not in ISAACSIM_COLLECTION_TASKS:
            supported = ", ".join(sorted(ISAACSIM_COLLECTION_TASKS))
            raise ValueError(
                "--sim_backend=isaacsim is currently supported only for "
                f"migrated collection tasks: {supported}."
            )
        env_extra_kwargs["sim_backend"] = FLAGS.sim_backend
        env_extra_kwargs["arm_type"] = FLAGS.arm_type
        env_extra_kwargs["seed"] = collection_run_seed
        env_extra_kwargs["randomize_dynamics"] = bool(FLAGS.randomize_dynamics)
    elif FLAGS.sim_backend == "isaacsim_newton":
        if task_id not in NEWTON_COLLECTION_TASKS:
            supported = ", ".join(sorted(NEWTON_COLLECTION_TASKS))
            raise ValueError(
                "--sim_backend=isaacsim_newton is currently supported only for "
                f"migrated collection tasks: {supported}."
            )
        if FLAGS.arm_type != "rokae" or FLAGS.hand_type != "wuji1":
            raise ValueError(
                "--sim_backend=isaacsim_newton requires "
                "--arm_type=rokae --hand_type=wuji1."
            )
        env_extra_kwargs["sim_backend"] = FLAGS.sim_backend
        env_extra_kwargs["arm_type"] = FLAGS.arm_type
        env_extra_kwargs["seed"] = collection_run_seed
        env_extra_kwargs["randomize_dynamics"] = bool(FLAGS.randomize_dynamics)
    elif FLAGS.sim_backend == "genesis":
        env_extra_kwargs["sim_backend"] = FLAGS.sim_backend
        if int(FLAGS.seed) >= 0:
            env_extra_kwargs["seed"] = collection_run_seed
    elif int(FLAGS.seed) >= 0:
        env_extra_kwargs["seed"] = collection_run_seed

    if (
        FLAGS.sim_backend == "isaacsim"
        and FLAGS.record_physx_ground_truth
        and FLAGS.certify_physical_replay
        and FLAGS.randomize_dynamics
    ):
        raise ValueError(
            "Certified PhysX collection requires --randomize_dynamics=False "
            "because randomized physical parameters are not part of the "
            "replay contract. Scene randomization (--randomize=True) is "
            "supported and is reproduced from the recorded reset seed and "
            "initial PhysX state."
        )

    startup_heartbeat = None
    if FLAGS.sim_backend in ISAAC_RUNTIME_BACKENDS:
        _write_isaac_worker_status(
            "building_scene",
            task_id=task_id,
        )
        startup_heartbeat = _start_isaac_startup_heartbeat(
            "building_scene",
            task_id=task_id,
        )
    try:
        env = config.get_native_environment(
            policy_mode=bool(
                FLAGS.scripted_policy
                or scripted_as013_success
                or policy_actions is not None
            ),
            render_mode=FLAGS.render_mode,
            randomize=FLAGS.randomize,
            **env_extra_kwargs,
        )
    except Exception as exc:
        # Stop the background writer before publishing a terminal startup
        # failure.  Both use the same atomic status path, and the fatal path
        # intentionally exits without executing Python cleanup handlers.
        _stop_isaac_startup_heartbeat(startup_heartbeat)
        startup_heartbeat = None
        if FLAGS.sim_backend in ISAAC_RUNTIME_BACKENDS:
            _fail_fast_supervised_isaac_startup(
                exc,
                task_id=task_id,
                phase="building_scene",
            )
        raise
    finally:
        _stop_isaac_startup_heartbeat(startup_heartbeat)

    _install_graceful_stop_handlers()
    manual_reset_bridge.install()

    reset_kwargs = {}
    if FLAGS.target_label:
        reset_kwargs["target_label"] = FLAGS.target_label
    deferred_physical_replay_certification = (
        _deferred_physical_replay_certification_enabled()
    )
    base_out = None if FLAGS.preview_only else _ensure_base_outdir(FLAGS.out_dir)
    physx_session_journal: PhysxSessionHistoryJournal | None = None
    if _physical_replay_session_history_required(
        preview_only=bool(FLAGS.preview_only),
        sim_backend=str(FLAGS.sim_backend),
        record_physx_ground_truth=bool(FLAGS.record_physx_ground_truth),
        deferred_certification=deferred_physical_replay_certification,
        fresh_certified_worker=(
            os.environ.get(_FRESH_CERTIFIED_COLLECTION_WORKER_ENV, "") == "1"
        ),
    ):
        session_id = os.environ.get(
            "HANDSON_COLLECTION_SESSION_ID",
            "",
        ).strip()
        if not session_id:
            session_id = (
                f"{task_id}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
                f"-{os.getpid()}-{secrets.token_hex(4)}"
            )
            os.environ["HANDSON_COLLECTION_SESSION_ID"] = session_id
        assert base_out is not None
        physx_session_journal = PhysxSessionHistoryJournal.create(
            _physical_replay_session_history_path(base_out, session_id),
            session_id=session_id,
            task_name=task_id,
            collection_run_seed=collection_run_seed,
            metadata={
                "recorder_pid": os.getpid(),
                "collection_seed_source": collection_seed_source,
                "render_mode": FLAGS.render_mode,
                "randomize": bool(FLAGS.randomize),
                "randomize_dynamics": bool(FLAGS.randomize_dynamics),
                "arm_type": FLAGS.arm_type,
                "hand_type": FLAGS.hand_type,
                "scripted_policy": bool(FLAGS.scripted_policy),
                "in_process_physx_scene_rebuild_per_reset": bool(
                    _isaac_in_process_hard_reset_enabled_for_task(task_id)
                ),
            },
        )

    def _record_physx_reset_attempt(event: Mapping[str, object]) -> None:
        if physx_session_journal is None:
            return
        seed = event.get("episode_reset_seed")
        ordinal = event.get("episode_reset_ordinal")
        if type(seed) is not int or type(ordinal) is not int:
            raise ValueError("Journaled Isaac reset attempt has no seed/ordinal.")
        start_time = event.get("start_sim_time")
        end_time = event.get("end_sim_time")
        time_kwargs = {}
        if start_time is not None and end_time is not None:
            time_kwargs = {
                "start_sim_time": float(start_time),
                "end_sim_time": float(end_time),
            }
        physx_session_journal.record_reset_attempt(
            seed=seed,
            ordinal=ordinal,
            outcome=str(event.get("outcome") or ""),
            stable_kwargs=dict(event.get("stable_origin_kwargs") or {}),
            scene_rebuilt_before_reset=bool(
                event.get("scene_rebuilt_before_reset", False)
            ),
            scene_generation=int(event.get("scene_generation", 0)),
            scene_rebuild_report=(
                dict(event["scene_rebuild_report"])
                if isinstance(event.get("scene_rebuild_report"), Mapping)
                else None
            ),
            stable_origin=(
                dict(event["stable_recording_origin"])
                if isinstance(event.get("stable_recording_origin"), Mapping)
                else None
            ),
            physics_steps=(
                int(event["stability_diagnostics"]["physics_steps"])
                if isinstance(
                    event.get("stability_diagnostics"), Mapping
                )
                and type(
                    event["stability_diagnostics"].get("physics_steps")
                )
                is int
                else None
            ),
            stability_diagnostics=(
                dict(event["stability_diagnostics"])
                if isinstance(
                    event.get("stability_diagnostics"), Mapping
                )
                else None
            ),
            error=(str(event.get("error") or "") or None),
            **time_kwargs,
        )

    a03_collection = None
    if a03_options.mode == "real_aligned":
        from handson.sim.controllers.a03.collection import install_collection
        a03_collection = install_collection(env.unwrapped, a03_options)
    environment_recording_schema = None
    required_recording_fields: set[str] = set()
    reset_attempt_index = 0
    episode_reset_seed: int | None = None
    episode_reset_ordinal: int | None = None
    poisoned_isaac_backend = False

    def _reset_environment_once():
        nonlocal reset_attempt_index, episode_reset_seed, episode_reset_ordinal
        nonlocal poisoned_isaac_backend
        initial_reset = episode_reset_ordinal is None
        if FLAGS.sim_backend in ISAAC_RUNTIME_BACKENDS:
            worker_phase = (
                "stabilizing_initial_state"
                if initial_reset
                else "stabilizing_reset_state"
            )
            _write_isaac_worker_status(
                worker_phase,
                task_id=task_id,
                reset_attempt_index=reset_attempt_index,
            )

            def _publish_stabilization_progress(
                progress: Mapping[str, object],
            ) -> None:
                raw_environment = getattr(env, "unwrapped", env)
                pump_ui = getattr(raw_environment, "_pump_startup_ui", None)
                ui_pumped = False
                ui_error = ""
                if callable(pump_ui):
                    try:
                        ui_pumped = bool(pump_ui())
                    except Exception as exc:
                        # Viewport servicing is diagnostic only.  The strict
                        # physical gate and its replay history must not depend
                        # on whether a desktop renderer is available.
                        ui_error = f"{type(exc).__name__}: {exc}"
                _write_isaac_worker_status(
                    worker_phase,
                    task_id=task_id,
                    reset_attempt_index=reset_attempt_index,
                    stability_progress=dict(progress),
                    startup_ui_pumped=ui_pumped,
                    startup_ui_error=ui_error,
                )
        else:
            _publish_stabilization_progress = None
        try:
            (
                observation,
                reset_info,
                reset_attempt_index,
                episode_reset_seed,
            ) = _reset_recording_episode_with_stable_origin(
                env,
                reset_kwargs,
                sim_backend=FLAGS.sim_backend,
                task_id=task_id,
                collection_run_seed=collection_run_seed,
                reset_attempt_index=reset_attempt_index,
                seed_resets=(
                    FLAGS.sim_backend in ISAAC_RUNTIME_BACKENDS or int(FLAGS.seed) >= 0
                ),
                collection_seed_source=collection_seed_source,
                reset_attempt_observer=_record_physx_reset_attempt,
                stabilization_progress_observer=(
                    _publish_stabilization_progress
                ),
            )
        except Exception as exc:
            if FLAGS.sim_backend in ISAAC_RUNTIME_BACKENDS and initial_reset:
                _fail_fast_supervised_isaac_startup(
                    exc,
                    task_id=task_id,
                    phase="stabilizing_initial_state",
                )
            if (
                FLAGS.sim_backend in ISAAC_RUNTIME_BACKENDS
                and _likely_poisoned_isaac_backend_error(exc)
            ):
                poisoned_isaac_backend = True
                _write_isaac_worker_status(
                    "fatal_runtime",
                    task_id=task_id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            raise
        if d20_source is not None:
            last_prelude_status = [0.0]
            def publish_d20_prelude(message):
                now = time.monotonic()
                if now - last_prelude_status[0] >= 1.0:
                    last_prelude_status[0] = now
                    _write_isaac_worker_status("stabilizing_initial_state", task_id=task_id, d20_prelude=message)
                    _write_collection_runtime_status("running", message)
            observation, resume_metadata = resume_saved_stage(
                env, d20_source, FLAGS.out_dir, speed=d20_speed, progress=publish_d20_prelude)
            reset_info["d20_stage_resume"] = resume_metadata
            # Preserve the initial scene stability evidence and explicitly identify
            # the later saved-state boundary used as the new action origin.
            reset_info.setdefault("physical_recording_origin", {})["d20_saved_stage_boundary"] = resume_metadata
            env.unwrapped._physical_recording_origin = dict(reset_info.get("physical_recording_origin", {}))
        if task_id in STAGE_TASKS:
            env.unwrapped._d20_state_layout = state_layout(env, config)
        episode_reset_ordinal = reset_attempt_index - 1
        if task_id == "synchronized_dual_knob_setting":
            left_target_deg = float(
                reset_info.get("dual_knob_x_target_deg") or 0.0
            )
            right_target_deg = float(
                reset_info.get("dual_knob_y_target_deg") or 0.0
            )
            target_label = str(
                reset_info.get("dual_knob_target_label") or ""
            )
            relation_text = target_label.replace("deg", "°").replace(
                ", ", "，"
            )
            operator_text = (
                f"本轮目标：蓝色左旋钮逆时针 {left_target_deg:g}°；"
                f"红色右旋钮顺时针 {right_target_deg:g}°；"
                f"关系：{relation_text}"
            )
            task_runtime = {
                "kind": "dual_knob_target",
                "task_name": task_id,
                "episode_reset_ordinal": int(episode_reset_ordinal),
                "target_index": int(
                    reset_info.get("dual_knob_target_index") or 0
                ),
                "target_family": str(
                    reset_info.get("dual_knob_target_family") or ""
                ),
                "left_ccw_deg": left_target_deg,
                "right_cw_deg": right_target_deg,
                "relation": relation_text,
                "operator_text": operator_text,
            }
            _write_collection_runtime_status(
                "running",
                operator_text,
                task_runtime=task_runtime,
            )
            print(f"[Recorder][KA005] {operator_text}", flush=True)
        if FLAGS.sim_backend == "isaacsim" and guidance_mode:
            settle_steps = int(
                reset_info.get("task_guidance_settle_physics_steps", 0)
                or 0
            )
            settle_seconds = float(
                reset_info.get("task_guidance_settle_seconds", 0.0)
                or 0.0
            )
            print(
                "[Task guidance] Hands-only route frame 0 ready: "
                f"task_settle={settle_steps} steps/{settle_seconds:.3f}s, "
                "task_success_gate=disabled, formal_v3_robot_gate=not_applicable.",
                flush=True,
            )
            _write_isaac_worker_status(
                "running",
                task_id=task_id,
                reset_attempt_index=reset_attempt_index,
            )
            _write_collection_runtime_status(
                "running",
                "指导初态已就绪，采集运行中。",
                task_id=task_id,
                reset_attempt_index=reset_attempt_index,
            )
        elif FLAGS.sim_backend == "isaacsim":
            origin = reset_info.get("physical_recording_origin", {})
            print(
                "[Recorder] Stable physical frame 0: "
                f"steps={origin.get('physics_steps', 0)}, "
                "linear="
                f"{float(origin.get('final_max_linear_velocity', 0.0)):.6f}m/s, "
                "angular="
                f"{float(origin.get('final_max_angular_velocity', 0.0)):.6f}rad/s, "
                "gate_joint="
                f"{float(origin.get('gate_max_robot_joint_velocity', 0.0)):.6f}rad/s, "
                "target_error="
                f"{float(origin.get('final_robot_position_target_error', 0.0)):.9f}rad, "
                "robot_origin="
                f"{origin.get('robot_hold_target_source', 'legacy')}, "
                "attempt="
                f"{origin.get('reset_stabilization_attempt', 1)}/"
                f"{origin.get('reset_stabilization_attempts_allowed', 1)}.",
                flush=True,
            )
            _write_isaac_worker_status(
                "running",
                task_id=task_id,
                reset_attempt_index=reset_attempt_index,
            )
            _write_collection_runtime_status(
                "running",
                "稳定初态验证通过，采集运行中。",
                task_id=task_id,
                reset_attempt_index=reset_attempt_index,
            )
        elif FLAGS.sim_backend == "isaacsim_newton":
            _write_isaac_worker_status(
                "running",
                task_id=task_id,
                reset_attempt_index=reset_attempt_index,
            )
        if a03_collection is not None:
            a03_collection.start()
            print("[A03 collection] " + json.dumps(a03_collection.summary()), flush=True)
        return observation, reset_info

    def _reset_environment():
        def attempt(source):
            nonlocal d20_source
            # The successful source must own the new episode's lineage/claim.
            d20_source = source
            _publish_d20_source(task_id, source, "attempting")
            result = _reset_environment_once()
            _publish_d20_source(task_id, source, "ready")
            return result

        def report_skip(message):
            print(message, flush=True)
            _write_isaac_worker_status("stabilizing_initial_state", task_id=task_id, d20_prelude=message)
            _write_collection_runtime_status("recovering", message, task_id=task_id)

        if d20_source is None:
            return _reset_environment_once()
        return with_source_retries(
            FLAGS.out_dir, task_id, d20_source, attempt, source_mode=d20_source_mode, progress=report_skip,
        )

    def _refresh_episode_recording_contract() -> None:
        nonlocal environment_recording_schema, required_recording_fields
        environment_recording_schema = resolve_environment_recording_schema(env)
        required_recording_fields = declared_recording_field_names(
            environment_recording_schema
        )

    try:
        obs, info = _reset_environment()
    except NoUsableSource as exc:
        # There are no accepted actions/demos yet. Preserve startup evidence
        # and close the simulator normally instead of escaping into Kit's
        # uncaught-exception shutdown hook.
        _report_d20_sources_exhausted(task_id, exc)
        try:
            if physx_session_journal is not None:
                try:
                    physx_session_journal.finalize([])
                except Exception as journal_error:
                    print(f"[D20] Startup history retained unsealed: {journal_error}", flush=True)
        finally:
            if FLAGS.sim_backend == "isaacsim":
                _enable_isaac_fast_shutdown(env)
            env.close()
        return
    _refresh_episode_recording_contract()
    if FLAGS.sim_backend in ISAAC_RUNTIME_BACKENDS:
        # "Ready" means considerably more than scene construction: the first
        # reset, natural settle and stable-origin gates all passed and an
        # observation can be stepped.  This closes the old false-ready race.
        _mark_isaac_runtime_ready(task_id)
        _write_isaac_worker_status("running", task_id=task_id)
    print(f"Environment reset complete for {task_id}.", flush=True)
    if FLAGS.target_label:
        print(f"Target label fixed to {FLAGS.target_label}.")
    _maybe_auto_start_teleop(env)
    if policy_actions is not None:
        expected_shape = tuple(env.action_space.shape)
        if tuple(policy_actions.shape[1:]) != expected_shape:
            raise ValueError(
                "Policy-action validation shape does not match the public "
                f"environment action space: actions={policy_actions.shape}, "
                f"space={expected_shape}."
            )
        print(
            "[Recorder] Public policy-action validation enabled: "
            f"path={policy_actions_path}, shape={policy_actions.shape}.",
            flush=True,
        )

    live_preview_path = _live_preview_path()
    live_preview_stride = _live_preview_stride(
        str(FLAGS.sim_backend),
        task_id,
    )
    _write_live_preview(obs, live_preview_path, env=env)
    live_preview_publisher = None
    if (
        live_preview_path is not None
        and _async_live_preview_enabled(str(FLAGS.sim_backend), task_id)
    ):
        live_preview_publisher = LatestFramePreviewPublisher(live_preview_path)

    def _publish_live_preview(observation) -> None:
        if live_preview_publisher is not None:
            live_preview_publisher.submit(observation)
            return
        _write_live_preview(observation, live_preview_path, env=env)

    if FLAGS.preview_only:
        print("Preview mode enabled; no demos will be written.")

    saved_demo_dirs = []
    success_count = 0
    success_needed = FLAGS.successes_needed

    pbar = None if FLAGS.preview_only else tqdm(total=success_needed)

    trajectory = []  # temporary storage for current episode transitions
    returns = 0.0
    total_steps = 0

    def _journal_episode_history(
        reason: str,
        *,
        accepted_demo_dir: Path | None = None,
        allow_partial_terminal_guard: bool = False,
    ) -> dict[str, object] | None:
        if physx_session_journal is None:
            return None
        if episode_reset_ordinal is None:
            raise RuntimeError(
                "PhysX session trajectory has no accepted reset ordinal."
            )
        time_kwargs: dict[str, float] = {}
        if not trajectory:
            boundary_time = _observation_physx_time(obs)
            if boundary_time is None:
                raise RuntimeError(
                    "An empty PhysX trajectory has no simulation-time "
                    "boundary; complete session history cannot be proven."
                )
            time_kwargs = {
                "start_sim_time": boundary_time,
                "end_sim_time": boundary_time,
            }
        return physx_session_journal.record_trajectory(
            trajectory,
            reason=reason,
            reset_ordinal=int(episode_reset_ordinal),
            accepted_demo_dir=accepted_demo_dir,
            environment_schema=environment_recording_schema,
            required_fields=required_recording_fields,
            allow_partial_terminal_guard=allow_partial_terminal_guard,
            **time_kwargs,
        )

    def _observation_physx_time(observation) -> float | None:
        if not isinstance(observation, Mapping):
            return None
        value = observation.get("physx_sim_time")
        if value is None:
            return None
        array = np.asarray(value).reshape(-1)
        if array.size != 1 or not np.isfinite(array[0]):
            return None
        return float(array[0])

    episode_steps = 0
    episode_start_sim_time = _observation_physx_time(obs)

    def _append_manual_step_to_session_history(
        current_observation,
        next_observation,
        executed_action,
        *,
        terminal: bool,
        transition_info: Mapping[str, object],
    ) -> bool:
        """Retain a reset-triggering step only when it advanced PhysX."""

        if physx_session_journal is None:
            return False
        current_time = _observation_physx_time(current_observation)
        next_time = _observation_physx_time(next_observation)
        advanced = bool(
            current_time is not None
            and next_time is not None
            and next_time > current_time + 1.0e-12
        )
        if (
            (current_time is None or next_time is None)
            and isinstance(next_observation, Mapping)
        ):
            substeps = next_observation.get("physx_substep_count")
            if substeps is not None:
                values = np.asarray(substeps).reshape(-1)
                advanced = bool(values.size == 1 and int(values[0]) > 0)
        if not advanced:
            return False
        if not isinstance(current_observation, Mapping) or not isinstance(
            next_observation,
            Mapping,
        ):
            raise TypeError("Journaled PhysX transition observations must be mappings.")
        missing_current = sorted(
            name
            for name in required_recording_fields
            if name not in current_observation
        )
        missing_next = sorted(
            name
            for name in required_recording_fields
            if name not in next_observation
        )
        if missing_current or missing_next:
            raise ValueError(
                "Manual-reset PhysX transition is missing recording fields: "
                f"current={missing_current}, next={missing_next}."
            )
        trajectory.append(
            copy.deepcopy(
                {
                    "observations": {
                        name: current_observation[name]
                        for name in required_recording_fields
                    },
                    "next_observations": {
                        name: next_observation[name]
                        for name in required_recording_fields
                    },
                    "actions": _flatten_action_for_storage(executed_action),
                    "action_storage_layout": (
                        _action_storage_layout_for_action(executed_action)
                    ),
                    "dones": bool(terminal),
                    "infos": dict(transition_info),
                }
            )
        )
        return True

    # Simulator-camera viewer initialization (optional)
    wrist_cam_viewer = None
    sim_camera_viewer_include_main = _sim_camera_viewer_includes_main(
        sim_backend=str(FLAGS.sim_backend),
        task_id=task_id,
        preview_only=bool(FLAGS.preview_only),
    )
    if FLAGS.show_sim_cameras and _has_displayable_sim_images(
        obs,
        include_main=sim_camera_viewer_include_main,
    ):
        wrist_cam_viewer = WristCameraViewer(
            display_scale=1.0 if FLAGS.sim_backend == "genesis" else 0.75,
            include_main=sim_camera_viewer_include_main,
        )
    wrist_viewer_disabled = False
    external_live_preview_process = None

    trajectory_spool = None
    manus_raw_recorder: ManusRawUdpRecorder | None = None
    manus_raw_startup_summary: dict[str, object] | None = None
    collection_error: Exception | None = None
    physx_session_finalization: Mapping[str, object] | None = None
    last_isaac_heartbeat = 0.0

    def _reject_current_episode_and_reset(
        reason: str,
        message: str,
        *,
        allow_partial_terminal_guard: bool = False,
    ) -> None:
        nonlocal trajectory, trajectory_spool, returns, obs, info
        nonlocal episode_steps, episode_start_sim_time
        print(message, flush=True)
        _journal_episode_history(
            reason,
            allow_partial_terminal_guard=allow_partial_terminal_guard,
        )
        trajectory = []
        if trajectory_spool is not None:
            trajectory_spool.cleanup()
            trajectory_spool = None
        _fast_exit_fresh_worker_for_rejected_attempt(reason)
        if not FLAGS.preview_only:
            trajectory_spool = RecordingSpool(
                base_out,
                video_fps=FLAGS.video_fps,
                encode_videos=online_video_encoding_enabled,
            )
        returns = 0.0
        obs, info = _reset_environment()
        episode_steps = 0
        episode_start_sim_time = _observation_physx_time(obs)
        if manus_raw_recorder is not None:
            manus_raw_recorder.begin_episode()
        _refresh_episode_recording_contract()
        _maybe_auto_start_teleop(env, immediate=True)
        _publish_live_preview(obs)

    def _save_guidance_route_if_present(*, route_complete: bool = False) -> str | None:
        """Publish a hands-only route only after an explicit Space delimiter.

        Guidance is a route demonstration, not a benchmark episode.  The
        current trajectory therefore has no successful terminal transition;
        it is still a valid guide once both physical palm readbacks are
        present for at least two control frames.  Stop signals, R resets and
        max-step timeouts deliberately discard the in-progress route.
        """

        nonlocal trajectory, trajectory_spool, success_count
        if (
            not guidance_mode
            or FLAGS.preview_only
            or not route_complete
            or success_count >= success_needed
            or len(trajectory) < 2
            or trajectory_spool is None
            or base_out is None
        ):
            if guidance_mode and not FLAGS.preview_only and len(trajectory) < 2:
                print(
                    "[Task guidance] Route discarded: press Space after "
                    "at least two hand frames to seal a guide.",
                    flush=True,
                )
            return None
        if manus_raw_recorder is not None:
            manus_raw_events, manus_raw_summary = manus_raw_recorder.snapshot_report()
        elif manus_raw_startup_summary is not None:
            manus_raw_events = []
            manus_raw_summary = manus_raw_startup_summary
        else:
            manus_raw_events = None
            manus_raw_summary = None
        route_index = success_count + 1
        demo_dir = _write_demo_zarr_and_videos(
            trajectory,
            task_id,
            route_index,
            base_out,
            FLAGS.video_fps,
            env=env,
            environment_schema=environment_recording_schema,
            required_recording_fields=required_recording_fields,
            video_spool=trajectory_spool,
            collection_run_seed=collection_run_seed,
            episode_reset_seed=episode_reset_seed,
            collection_seed_source=collection_seed_source,
            episode_reset_ordinal=episode_reset_ordinal,
            manus_raw_events=manus_raw_events,
            manus_raw_capture_summary=manus_raw_summary,
            online_camera_capture_scope=runtime_camera_scope,
            guidance_route_only=True,
        )
        register_guidance_route_demo(
            Path(demo_dir),
            task_name=task_id,
            collector_id=os.environ.get("HANDSON_COLLECTOR_ID", "collector_001"),
            session_id=os.environ.get("HANDSON_COLLECTION_SESSION_ID", ""),
        )
        _journal_episode_history(
            "guidance_route_recorded",
            accepted_demo_dir=demo_dir,
        )
        success_count = route_index
        saved_demo_dirs.append(demo_dir)
        if pbar is not None:
            pbar.update(1)
        print(
            f"[Task guidance] Saved hand route #{route_index} -> {demo_dir}",
            flush=True,
        )
        trajectory = []
        trajectory_spool.cleanup()
        trajectory_spool = None
        return str(demo_dir)

    def _reset_after_guidance_route() -> None:
        """Start a fresh uncounted route after Space sealed the previous one."""

        nonlocal obs, info, trajectory_spool, episode_steps
        nonlocal episode_start_sim_time, returns
        obs, info = _reset_environment()
        episode_steps = 0
        episode_start_sim_time = _observation_physx_time(obs)
        returns = 0.0
        if manus_raw_recorder is not None:
            manus_raw_recorder.begin_episode()
        _refresh_episode_recording_contract()
        _maybe_auto_start_teleop(env, immediate=True)
        _publish_live_preview(obs)
        if not FLAGS.preview_only:
            trajectory_spool = RecordingSpool(
                base_out,
                video_fps=FLAGS.video_fps,
                encode_videos=online_video_encoding_enabled,
            )

    try:
        if (
            not FLAGS.preview_only
            and str(FLAGS.hand_type) in {"wuji1", "wuji2"}
        ):
            raw_port = int(
                os.environ.get(
                    "HANDSON_MANUS_RAW_RECORDING_PORT",
                    MANUS_RAW_DEFAULT_PORT,
                )
            )
            try:
                manus_raw_recorder = ManusRawUdpRecorder(port=raw_port)
                manus_raw_recorder.begin_episode()
                print(
                    "[Recorder] Raw MANUS sidecar enabled on "
                    f"UDP 127.0.0.1:{manus_raw_recorder.port}; failures "
                    "will be recorded as warnings and will not block collection.",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                manus_raw_startup_summary = {
                    "format": MANUS_RAW_FORMAT,
                    "version": MANUS_RAW_VERSION,
                    "event_count": 0,
                    "topics": [],
                    "sides": [],
                    "sequence_gap_count": 0,
                    "sequence_gaps": [],
                    "receiver_error_count": 1,
                    "receiver_errors": [f"{type(exc).__name__}: {exc}"],
                    "complete": False,
                    "status": "recorder_unavailable",
                }
                print(
                    "[Recorder][WARN] Raw MANUS sidecar is unavailable: "
                    f"{type(exc).__name__}: {exc}. Continuing collection.",
                    flush=True,
                )
        if not FLAGS.preview_only:
            trajectory_spool = RecordingSpool(
                base_out,
                video_fps=FLAGS.video_fps,
                encode_videos=online_video_encoding_enabled,
            )
            if online_video_encoding_enabled:
                print(
                    "[Recorder] Canonical cameras encode directly to "
                    f"{trajectory_spool.root} (H.264 CRF "
                    f"{trajectory_spool.video_crf}).",
                    flush=True,
                )
            else:
                if guidance_mode:
                    print(
                        "[Task guidance] Recording hands-only route: one "
                        "physical two-hand palm pose per control frame; task "
                        "success/failure and full V3 substep trace are disabled.",
                        flush=True,
                    )
                else:
                    print(
                        "[Recorder] Canonical camera postrender is enabled; "
                        "recording action/state/physics trace online without "
                        "placeholder RGB encoding.",
                        flush=True,
                    )

        # Keep FU015's contact-aware scripted policy alive for the whole
        # episode.  Recreating it every frame discards the first-contact hold
        # pose and turns its bounded force relief into an open-loop descent.
        fu015_policy = None
        fu015_policy_seed = None

        # Keep the readback-following policy alive for the full episode.  Its
        # captured body-to-TCP grasp transforms are physical episode state.
        p003_policy = None
        p003_policy_seed = None

        # Main collection loop
        p002_policy = None
        p002_policy_seed = None
        while FLAGS.preview_only or success_count < success_needed:
            if graceful_stop_requested.is_set():
                raise KeyboardInterrupt
            if manual_reset_bridge.consume():
                _reject_current_episode_and_reset(
                    "operator_keyboard_reset",
                    "[Keyboard] R reset triggered from the live camera",
                )
                # Drop auto-repeat requests received while reset stabilization
                # was running; one key press represents one rejected attempt.
                manual_reset_bridge.discard_pending()
                continue
            if FLAGS.sim_backend in ISAAC_RUNTIME_BACKENDS:
                heartbeat_now = time.monotonic()
                if heartbeat_now - last_isaac_heartbeat >= 1.0:
                    _touch_isaac_worker_heartbeat()
                    last_isaac_heartbeat = heartbeat_now
            actions = np.zeros(env.action_space.sample().shape)
            if FLAGS.scripted_policy:
                raw_env = getattr(env, "unwrapped", env)
                if task_id == "t16_open_laptop":
                    from handson.sim.isaaclab.fu015_contract import (
                        FU015ScriptedPolicy,
                        HAND_ACTION_DIM,
                    )

                    reset_state = getattr(raw_env, "_fu015_reset", None)
                    if reset_state is None:
                        raise RuntimeError(
                            "FU015 scripted collection could not read the seeded reset state."
                        )
                    reset_seed = int(reset_state.seed)
                    if fu015_policy is None or fu015_policy_seed != reset_seed:
                        fu015_policy = FU015ScriptedPolicy(reset_state)
                        fu015_policy_seed = reset_seed
                    actions = fu015_policy.action(
                        int(getattr(raw_env, "env_step", 0)),
                        readback=getattr(raw_env, "_fu015_last_readback", None),
                    )
                elif task_id == "t17_open_latched_door":
                    from handson.sim.isaaclab.p002_contract import (
                        HAND_ACTION_DIM,
                        P002ScriptedPolicy,
                    )

                    reset_state = getattr(raw_env, "_p002_reset", None)
                    if reset_state is None:
                        raise RuntimeError(
                            "P0-02 scripted collection could not read the seeded reset state."
                        )
                    reset_seed = int(reset_state.seed)
                    if p002_policy is None or p002_policy_seed != reset_seed:
                        p002_policy = P002ScriptedPolicy(reset_state)
                        p002_policy_seed = reset_seed
                    actions = p002_policy.action(
                        int(getattr(raw_env, "env_step", 0)),
                        readback=getattr(raw_env, "_p002_last_readback", None),
                        latch=getattr(raw_env, "_p002_latch", None),
                    )
                elif task_id == "p003_bimanual_folding_ruler_shape_configuration":
                    from handson.sim.isaaclab.p003_contract import (
                        HAND_ACTION_DIM,
                        P003ScriptedPolicy,
                    )

                    reset_state = getattr(raw_env, "_p003_reset", None)
                    if reset_state is None:
                        raise RuntimeError(
                            "P0-03 scripted collection could not read the seeded reset state."
                        )
                    reset_seed = int(reset_state.seed)
                    if p003_policy is None or p003_policy_seed != reset_seed:
                        p003_policy = P003ScriptedPolicy(
                            reset_state,
                            right_home_pose=np.asarray(
                                raw_env._right_tcp_pose, dtype=np.float64
                            ).copy(),
                            left_home_pose=np.asarray(
                                raw_env._left_tcp_pose, dtype=np.float64
                            ).copy(),
                            initial_body_poses=np.asarray(
                                raw_env._body_poses(), dtype=np.float64
                            ).copy(),
                        )
                        p003_policy_seed = reset_seed
                    actions = p003_policy.action(
                        int(getattr(raw_env, "env_step", 0)),
                        readback=getattr(raw_env, "_p003_last_readback", None),
                        stage=str(raw_env._p003_tracker.stage),
                        body_poses=raw_env._body_poses(),
                        right_tcp_pose=raw_env._actual_tcp_pose("right"),
                        left_tcp_pose=raw_env._actual_tcp_pose("left"),
                    )
                    raw_env.set_p003_control_state(
                        p003_policy.phase,
                        configure_preparing=p003_policy.configure_preparing,
                        workspace_preparing=p003_policy.workspace_preparing,
                    )
                else:
                    raise RuntimeError(
                        f"No scripted policy is registered for {task_id}."
                    )
            elif scripted_as013_success:
                raw_environment = getattr(env, "unwrapped", env)
                policy = getattr(raw_environment, "scripted_success_action", None)
                if not callable(policy):
                    raise RuntimeError(
                        "B320-AS013 runtime does not expose scripted_success_action"
                    )
                actions = np.asarray(
                    policy(int(getattr(raw_environment, "env_step", total_steps))),
                    dtype=np.float32,
                )
                if actions.shape != env.action_space.shape:
                    raise RuntimeError(
                        "B320-AS013 scripted action shape differs from the "
                        f"recording action space: {actions.shape} != "
                        f"{env.action_space.shape}"
                    )
            elif policy_actions is not None:
                if total_steps >= policy_actions.shape[0]:
                    raise RuntimeError(
                        "Policy-action validation exhausted its finite action "
                        f"array after {total_steps} steps without task success."
                    )
                actions = policy_actions[total_steps].copy()

            # Capture depth now (env state currently matches `obs`); it'll be
            # stored alongside `obs` below for parity with the RGB cameras.
            depth_now = None
            if (
                FLAGS.save_depth
                and not FLAGS.preview_only
                and isinstance(obs, dict)
            ):
                camera_sources = select_canonical_camera_sources(obs, env=env)
                source_keys = list(dict.fromkeys(camera_sources.values()))
                source_depth = collect_depth_frames(
                    env,
                    source_keys,
                )
                depth_now = {
                    canonical_key: postprocess_canonical_depth(
                        source_depth[source_key],
                        canonical_key,
                        source_key,
                    )
                    for canonical_key, source_key in camera_sources.items()
                    if source_key in source_depth
                }

            if (
                FLAGS.show_sim_cameras
                and wrist_cam_viewer is None
                and not wrist_viewer_disabled
                and _has_displayable_sim_images(
                    obs,
                    include_main=sim_camera_viewer_include_main,
                )
            ):
                wrist_cam_viewer = WristCameraViewer(
                    display_scale=1.0 if FLAGS.sim_backend == "genesis" else 0.75,
                    include_main=sim_camera_viewer_include_main,
                )

            if wrist_cam_viewer is not None:
                try:
                    key = wrist_cam_viewer.update_and_show(obs, env=env)
                except RuntimeError as exc:
                    print(f"[Recorder] {exc}", flush=True)
                    wrist_cam_viewer.close()
                    wrist_cam_viewer = None
                    wrist_viewer_disabled = True
                    external_live_preview_process = _launch_external_live_preview(
                        live_preview_path,
                        title=(
                            "B03 Newton 实时相机（GPU-to-GPU）"
                            if FLAGS.sim_backend == "isaacsim_newton"
                            and task_id == "t09_insert_peg"
                            else "HandsOn 实时仿真相机"
                        ),
                    )
                    key = -1
                if key == ord("r"):
                    _reject_current_episode_and_reset(
                        "operator_keyboard_reset",
                        "[Keyboard] R reset triggered from the live camera",
                    )
                    manual_reset_bridge.discard_pending()
                    continue
                if key == ord(" "):
                    if guidance_mode:
                        _request_guidance_route_completion(env)
                    else:
                        _request_c08_keyboard_grasp(env)

            if total_steps % live_preview_stride == 0:
                _publish_live_preview(obs)

            step_sampled_monotonic = time.monotonic()
            next_obs, rew, done, truncated, info = env.step(actions)
            episode_steps += 1
            terminal = bool(done or truncated)

            if "intervene_action" in info:
                actions = info["intervene_action"]
            if "a03_collection_action" in info:
                actions = info["a03_collection_action"]

            # A manual reset is not a transition from the old episode.  Handle
            # it before RGB encoding/depth copies so control can resume on the
            # reset scene with minimum latency.
            if info.get("manual_reset", False):
                numerical_guard_abort = isinstance(
                    info.get("numerical_guard_event"),
                    Mapping,
                )
                _append_manual_step_to_session_history(
                    obs,
                    next_obs,
                    actions,
                    terminal=terminal,
                    transition_info=info,
                )
                _reject_current_episode_and_reset(
                    (
                        "numerical_guard_abort"
                        if numerical_guard_abort
                        else "teleoperation_manual_reset"
                    ),
                    (
                        "[Teleop] Numerical safety guard triggered; preserving "
                        "the partial physical trace, dropping the trajectory, "
                        "and rebuilding the scene."
                        if numerical_guard_abort
                        else "[Teleop] Manual reset triggered; dropping current "
                        "trajectory."
                    ),
                    allow_partial_terminal_guard=numerical_guard_abort,
                )
                continue

            if (
                not FLAGS.preview_only
                and _requires_b03_newton_source_validity(
                    task_id,
                    str(FLAGS.sim_backend),
                )
            ):
                source_validity = evaluate_b03_newton_substep_observation(
                    next_obs if isinstance(next_obs, Mapping) else {},
                    (
                        environment_recording_schema
                        if isinstance(environment_recording_schema, Mapping)
                        else {}
                    ),
                )
                source_validity_error = b03_newton_source_validity_error(
                    source_validity
                )
                if source_validity_error is not None:
                    if FLAGS.sim_backend in ISAAC_RUNTIME_BACKENDS:
                        _write_isaac_worker_status(
                            "running",
                            task_id=task_id,
                            last_episode_rejection=(
                                "b03_newton_source_physics_invalid"
                            ),
                            source_physical_validity=source_validity,
                        )
                    _write_collection_runtime_status(
                        "running",
                        "B03 Newton 本条物理证据超限，已丢弃并重置。",
                        source_physical_validity=source_validity,
                    )
                    _reject_current_episode_and_reset(
                        "b03_newton_source_physics_invalid",
                        "[Recorder][B03 Newton][INVALID] "
                        + source_validity_error
                        + "; dropping this episode without saving or repairing "
                        "simulator state.",
                    )
                    continue

            total_steps += 1
            returns += rew

            if not FLAGS.preview_only:
                if trajectory_spool is None:
                    raise RuntimeError("Recorder spool is unavailable.")
                if FLAGS.scripted_policy:
                    if task_id == "t16_open_laptop":
                        actions = _group_fu015_scripted_action(
                            actions,
                            hand_action_dim=HAND_ACTION_DIM,
                        )
                    elif task_id == "t17_open_latched_door":
                        actions = _group_p002_scripted_action(
                            actions,
                            hand_action_dim=HAND_ACTION_DIM,
                        )
                    else:
                        actions = _group_p003_scripted_action(
                            actions,
                            hand_action_dim=HAND_ACTION_DIM,
                        )
                action_storage_layout = _action_storage_layout_for_action(actions)
                actions = _flatten_action_for_storage(actions)
                trajectory_spool.append_observation(obs, env=env)
                selected_observation = select_current_observation_for_recording(
                    obs,
                    required_fields=required_recording_fields,
                    include_required_fields=not trajectory,
                )
                selected_observation = trajectory_spool.compact_observation(
                    selected_observation,
                    protected_keys=required_recording_fields,
                )
                transition = copy.deepcopy(
                    dict(
                        observations=selected_observation,
                        next_observations=select_next_observation_for_recording(
                            next_obs,
                            required_fields=required_recording_fields,
                        ),
                        actions=actions,
                        action_storage_layout=action_storage_layout,
                        dones=terminal,
                        infos=info,
                        **{
                            RECORDER_MONOTONIC_TIME_KEY: (
                                step_sampled_monotonic
                            )
                        },
                    )
                )
                if depth_now is not None:
                    transition["depth"] = trajectory_spool.spool_depth_mapping(
                        depth_now
                    )
                trajectory.append(transition)

            if pbar is not None:
                pbar.set_description(f"Return: {returns}")
            obs = next_obs

            # Space is the only route delimiter in hands-only guidance.  The
            # just-sampled frame is retained, then the next route starts from
            # a clean reset.  R, Stop/SIGINT, and max_steps never count.
            if guidance_mode and bool(info.get("guidance_route_complete", False)):
                _save_guidance_route_if_present(route_complete=True)
                if success_count >= success_needed:
                    break
                _reset_after_guidance_route()
                continue

            if terminal:
                terminal_sim_time = _observation_physx_time(next_obs)
                episode_sim_time_seconds = None
                if (
                    episode_start_sim_time is not None
                    and terminal_sim_time is not None
                ):
                    episode_sim_time_seconds = max(
                        0.0,
                        terminal_sim_time - episode_start_sim_time,
                    )
                b320_terminal_summary = _print_b320_terminal_summary(
                    task_id,
                    done=done,
                    truncated=truncated,
                    info=info,
                    episode_steps=episode_steps,
                    episode_sim_time_seconds=episode_sim_time_seconds,
                )
                b320_reset_hold_seconds = 0.0
                if (
                    isinstance(b320_terminal_summary, Mapping)
                    and b320_terminal_summary.get("outcome") != "success"
                ):
                    b320_reset_hold_seconds = (
                        _b320_terminal_reset_hold_seconds()
                    )
                    _publish_live_preview(obs)
                    terminal_detail = _b320_terminal_status_detail(
                        b320_terminal_summary,
                        reset_complete=False,
                        hold_seconds=b320_reset_hold_seconds,
                    )
                    print(f"[Recorder][B320] {terminal_detail}", flush=True)
                    if FLAGS.sim_backend in ISAAC_RUNTIME_BACKENDS:
                        _write_isaac_worker_status(
                            "running",
                            task_id=task_id,
                            last_episode_rejection=str(
                                b320_terminal_summary.get("terminal_reason")
                                or "unsuccessful_terminal"
                            ),
                            last_terminal_summary=b320_terminal_summary,
                            terminal_reset_hold_seconds=(
                                b320_reset_hold_seconds
                            ),
                        )
                    _write_collection_runtime_status(
                        "running",
                        terminal_detail,
                        last_episode_rejection=str(
                            b320_terminal_summary.get("terminal_reason")
                            or "unsuccessful_terminal"
                        ),
                        last_terminal_summary=b320_terminal_summary,
                        terminal_reset_hold_seconds=b320_reset_hold_seconds,
                    )
                unsuccessful_reason = _as013_unsuccessful_terminal_reason(
                    task_id,
                    info,
                )
                _print_e01_terminal_summary(
                    task_id,
                    done=done,
                    truncated=truncated,
                    info=info,
                )
                _print_p004_terminal_summary(
                    task_id,
                    done=done,
                    truncated=truncated,
                    info=info,
                    episode_steps=episode_steps,
                )
                try:
                    if info.get("succeed", False) and not FLAGS.preview_only:
                        if FLAGS.sim_backend in ISAAC_RUNTIME_BACKENDS:
                            _write_isaac_worker_status(
                                "saving_episode",
                                task_id=task_id,
                                success_index=success_count + 1,
                            )
                        if manus_raw_recorder is not None:
                            manus_raw_events, manus_raw_summary = (
                                manus_raw_recorder.snapshot_report()
                            )
                            if not manus_raw_summary.get("complete", False):
                                print(
                                    "[Recorder][WARN] Raw MANUS sidecar is "
                                    "incomplete for this episode "
                                    f"({manus_raw_summary.get('status')}); "
                                    "saving the episode and diagnostics.",
                                    flush=True,
                                )
                        elif manus_raw_startup_summary is not None:
                            manus_raw_events = []
                            manus_raw_summary = manus_raw_startup_summary
                        else:
                            manus_raw_events = None
                            manus_raw_summary = None
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
                            video_spool=trajectory_spool,
                            collection_run_seed=collection_run_seed,
                            episode_reset_seed=episode_reset_seed,
                            collection_seed_source=collection_seed_source,
                            episode_reset_ordinal=episode_reset_ordinal,
                            manus_raw_events=manus_raw_events,
                            manus_raw_capture_summary=manus_raw_summary,
                            online_camera_capture_scope=runtime_camera_scope,
                        )
                        if d20_source is not None:
                            mark_stitched(
                                d20_source,
                                task_id,
                                consumer_episode=demo_dir,
                            )
                            _publish_d20_source(task_id, d20_source, "consumed")
                        if guidance_mode:
                            register_successful_guidance_demo(
                                Path(demo_dir),
                                task_name=task_id,
                                collector_id=os.environ.get(
                                    "HANDSON_COLLECTOR_ID", "collector_001"
                                ),
                                session_id=os.environ.get(
                                    "HANDSON_COLLECTION_SESSION_ID", ""
                                ),
                            )
                        _journal_episode_history(
                            "successful_terminal",
                            accepted_demo_dir=demo_dir,
                        )
                        saved_demo_dirs.append(demo_dir)
                        print(
                            f"[Saved] success #{success_count} "
                            f"-> {demo_dir}"
                        )
                        if pbar is not None:
                            pbar.update(1)
                        if FLAGS.sim_backend in ISAAC_RUNTIME_BACKENDS:
                            _write_isaac_worker_status(
                                "running",
                                task_id=task_id,
                                success_count=success_count,
                            )
                    elif not FLAGS.preview_only:
                        _journal_episode_history(unsuccessful_reason)
                finally:
                    trajectory = []
                    returns = 0.0
                    if trajectory_spool is not None:
                        trajectory_spool.cleanup()
                        trajectory_spool = None

                if not info.get("succeed", False):
                    _hold_b320_unsuccessful_terminal_frame(
                        task_id,
                        b320_terminal_summary,
                        seconds=b320_reset_hold_seconds,
                    )
                    rejection_detail = (
                        "the episode ended without info.succeed=true"
                    )
                    if isinstance(b320_terminal_summary, Mapping):
                        rejection_detail += "; terminal_reason=" + str(
                            b320_terminal_summary.get("terminal_reason")
                            or "unsuccessful_terminal"
                        )
                    _fast_exit_fresh_worker_for_rejected_attempt(
                        unsuccessful_reason,
                        rejection_detail,
                    )

                if not FLAGS.preview_only and success_count >= success_needed:
                    break

                if d20_source is not None and info.get("succeed", False):
                    try:
                        d20_source = select_source(FLAGS.out_dir, task_id, "", source_mode=d20_source_mode)
                    except NoUsableSource as exc:
                        _report_d20_sources_exhausted(task_id, exc)
                        print(
                            "[D20] No unused successful predecessor remains; "
                            f"stopping after {success_count} successful demo(s).",
                            flush=True,
                        )
                        break

                obs, info = _reset_environment()
                episode_steps = 0
                episode_start_sim_time = _observation_physx_time(obs)
                if manus_raw_recorder is not None:
                    manus_raw_recorder.begin_episode()
                _refresh_episode_recording_contract()
                _maybe_auto_start_teleop(env, immediate=True)
                _publish_live_preview(obs)
                if (
                    isinstance(b320_terminal_summary, Mapping)
                    and b320_terminal_summary.get("outcome") != "success"
                ):
                    reset_detail = _b320_terminal_status_detail(
                        b320_terminal_summary,
                        reset_complete=True,
                    )
                    if FLAGS.sim_backend in ISAAC_RUNTIME_BACKENDS:
                        _write_isaac_worker_status(
                            "running",
                            task_id=task_id,
                            last_episode_rejection=str(
                                b320_terminal_summary.get("terminal_reason")
                                or "unsuccessful_terminal"
                            ),
                            last_terminal_summary=b320_terminal_summary,
                            terminal_reset_complete=True,
                        )
                    _write_collection_runtime_status(
                        "running",
                        reset_detail,
                        last_episode_rejection=str(
                            b320_terminal_summary.get("terminal_reason")
                            or "unsuccessful_terminal"
                        ),
                        last_terminal_summary=b320_terminal_summary,
                        terminal_reset_complete=True,
                    )
                if not FLAGS.preview_only and success_count < success_needed:
                    trajectory_spool = RecordingSpool(
                        base_out,
                        video_fps=FLAGS.video_fps,
                        encode_videos=online_video_encoding_enabled,
                    )

            if FLAGS.max_steps > 0 and total_steps >= FLAGS.max_steps:
                if guidance_mode:
                    print(
                        f"Reached max_steps={FLAGS.max_steps}; the unfinished "
                        "guidance route was discarded (press Space to count it).",
                        flush=True,
                    )
                else:
                    print(f"Reached max_steps={FLAGS.max_steps}; stopping run.")
                break

        if not FLAGS.preview_only and guidance_mode and trajectory:
            print(
                "[Task guidance] Run ended without Space; the unfinished "
                "hand route was discarded.",
                flush=True,
            )
        if FLAGS.preview_only:
            print(f"Preview ended after {total_steps} steps.")
        else:
            summary_prefix = (
                f"Recorded {success_count} hand route(s). "
                if guidance_mode
                else f"Collected {success_count} successful demos. "
            )
            print(
                summary_prefix + "Individual demo directories:"
            )
            for path in saved_demo_dirs:
                print("  ", path)
    except NoUsableSource as exc:
        _report_d20_sources_exhausted(task_id, exc)
    except KeyboardInterrupt:
        print(
            (
                "[Recorder] Stop requested; the current hand route was "
                "discarded (press Space to seal a route)."
                if guidance_mode
                else "[Recorder] Stop requested; closing the current incomplete "
                f"episode and queuing post-collection work for {success_count} "
                "saved demo(s)."
            ),
            flush=True,
        )
    except Exception as exc:
        collection_error = exc
        if (
            FLAGS.sim_backend in ISAAC_RUNTIME_BACKENDS
            and _likely_poisoned_isaac_backend_error(exc)
        ):
            poisoned_isaac_backend = True
            _write_isaac_worker_status(
                "fatal_runtime",
                task_id=task_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            traceback.print_exc()
        print(
            "[Recorder] Collection stopped after an error; sealing the valid "
            f"session prefix before reporting {type(exc).__name__}: {exc}",
            flush=True,
        )
    finally:
        if manus_raw_recorder is not None:
            manus_raw_recorder.close()
        if trajectory_spool is not None:
            trajectory_spool.cleanup()
        if pbar is not None:
            pbar.close()
        if wrist_cam_viewer is not None:
            try:
                wrist_cam_viewer.close()
            except Exception:
                pass
        if live_preview_publisher is not None:
            live_preview_publisher.close()
        _close_external_live_preview(external_live_preview_process)
        if physx_session_journal is not None:
            assert base_out is not None
            try:
                # Stop/Ctrl-C can leave a valid, nonterminal trajectory in
                # memory. Retain it through the existing rejected-episode
                # journal before sealing the session and closing PhysX.
                if trajectory and not guidance_mode:
                    _journal_episode_history(
                        "collection_error" if collection_error is not None
                        else "collection_stopped"
                    )
                    print(
                        "[PhysX session history] retained incomplete "
                        f"trajectory steps={len(trajectory)}",
                        flush=True,
                    )
                    trajectory = []
                physx_session_finalization = (
                    physx_session_journal.finalize(saved_demo_dirs)
                )
                _bind_physx_session_history_to_demos(
                    saved_demo_dirs,
                    output_root=base_out,
                    journal=physx_session_journal,
                    finalization=physx_session_finalization,
                )
                print(
                    "[PhysX session history] sealed before simulator close "
                    f"events={physx_session_finalization.get('event_count', 0)} "
                    "accepted="
                    f"{len(physx_session_finalization.get('accepted_demos', []))} "
                    "sha256="
                    f"{physx_session_finalization.get('history_identity', {}).get('sha256', '')}",
                    flush=True,
                )
            except Exception as exc:
                if collection_error is None:
                    collection_error = exc
                print(
                    "[PhysX session history] failed to seal before simulator "
                    f"close: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        if (
            not poisoned_isaac_backend
            and FLAGS.sim_backend == "isaacsim_newton"
            and task_id == "t09_insert_peg"
        ):
            try:
                from handson.sim.isaaclab.newton_replay import (
                    destroy_newton_scene,
                )

                destruction = destroy_newton_scene(env)
                print(
                    "[Newton scene] B03 scene destroyed before Kit close: "
                    f"{destruction}",
                    flush=True,
                )
            except Exception as exc:
                if collection_error is None:
                    collection_error = RuntimeError(
                        "B03 Newton scene destruction failed before Kit close: "
                        f"{type(exc).__name__}: {exc}"
                    )
                print(
                    "[Newton scene][ERROR] B03 scene destruction failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
        if not poisoned_isaac_backend:
            try:
                if FLAGS.sim_backend == "isaacsim":
                    _enable_isaac_fast_shutdown(env)
                env.close()
            except Exception:
                pass
        else:
            print(
                "[Isaac watchdog] CUDA/PhysX backend is poisoned; skipped "
                "unsafe simulator teardown after sealing disk evidence.",
                flush=True,
            )
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        for signum, previous_handler in previous_stop_handlers.items():
            signal.signal(signum, previous_handler)
        manual_reset_bridge.restore()

    if guidance_mode and saved_demo_dirs:
        _write_collection_runtime_status(
            "planning_guidance",
            "手部路线已保存；正在做全局 IK、FK 和碰撞检查。",
            guide_count=len(saved_demo_dirs),
        )
        planner = Path(__file__).resolve().parent / "simplified_post_ik_plan.py"
        planning_failures: list[str] = []
        for demo_value in saved_demo_dirs:
            demo_path = Path(demo_value).expanduser().resolve()
            result = subprocess.run(
                [
                    sys.executable,
                    str(planner),
                    "--overwrite",
                    str(demo_path),
                ],
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout:
                print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
            if result.stderr:
                print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
            if result.returncode == 0:
                mark_guidance_planner_result(demo_path, success=True)
            else:
                detail = (
                    result.stderr.strip()
                    or result.stdout.strip()
                    or f"planner exited with code {result.returncode}"
                )
                mark_guidance_planner_result(
                    demo_path,
                    success=False,
                    detail=detail[-4000:],
                )
                planning_failures.append(f"{demo_path.name}: {detail[-500:]}")
        assert guidance_library_root is not None
        profile_result = compile_task_guidance_profile(
            task_id,
            root=guidance_library_root,
        )
        print(f"[Task guidance] compiled profile: {profile_result}", flush=True)
        if planning_failures and collection_error is None:
            collection_error = RuntimeError(
                "The hand route was recorded, but its arm plan was "
                "not safe enough to activate: " + "; ".join(planning_failures)
            )

    # Quality-first IPC replay is intentionally outside the live collection
    # loop and after env.close(): it cannot delay episode publication or steal
    # CPU from hand tracking. Synchronous E01 keeps fail-fast semantics; F01
    # defaults to a background worker and lets review/upload gates enforce the
    # postprocess status.
    ipc_env_value = os.environ.get("HANDSON_E01_AUTO_IPC_REPLAY", "1")
    arm_env_value = os.environ.get("HANDSON_E01_AUTO_ARM_IK", "1")
    if (
        not FLAGS.preview_only
        and _should_autostart_e01_postprocess(
            task_id,
            sim_backend=FLAGS.sim_backend,
            ipc_env_value=ipc_env_value,
            arm_env_value=arm_env_value,
        )
    ):
        if _should_background_e01_postprocess(
            task_id,
            sim_backend=FLAGS.sim_backend,
            env_value=os.environ.get(
                "HANDSON_E01_POSTPROCESS_BACKGROUND",
                _e01_postprocess_background_default(task_id),
            ),
            ipc_env_value=ipc_env_value,
            arm_env_value=arm_env_value,
        ):
            queued = _launch_e01_postprocess_batch(
                saved_demo_dirs,
                exp_name=task_id,
                sim_backend=FLAGS.sim_backend,
                ipc_env_value=ipc_env_value,
                arm_env_value=arm_env_value,
                video_fps=video_fps,
            )
            print(
                f"[E01 postprocess] queued {queued} background replay/render job(s)",
                flush=True,
            )
        else:
            failed: list[tuple[str, list[str]]] = []
            for demo_dir in saved_demo_dirs:
                result = _run_e01_postprocess_pipeline(
                    demo_dir,
                    exp_name=task_id,
                    sim_backend=FLAGS.sim_backend,
                    ipc_env_value=ipc_env_value,
                    arm_env_value=arm_env_value,
                    video_fps=video_fps,
                )
                failed_stages = [
                    str(name) for name in result.get("failed_stages", [])
                ]
                if result.get("status") == "failed":
                    failed.append((str(demo_dir), failed_stages))
            if failed:
                details = "; ".join(
                    f"{path}: {','.join(stages) or 'unknown'}"
                    for path, stages in failed
                )
                raise RuntimeError(
                    "E01 offline postprocessing failed; affected stages remain "
                    f"explicitly invalid: {details}"
                )

    if (
        not FLAGS.preview_only
        and FLAGS.sim_backend == "isaacsim"
        and _deferred_physical_replay_autostart_enabled()
    ):
        queued = _ensure_deferred_physical_replay_certification(
            base_out,
            exp_name=task_id,
            certification_runs=DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS,
            certification_timeout_seconds=int(
                FLAGS[
                    "physical_replay_certification_timeout_seconds"
                ].value
            ),
            session_id=os.environ.get(
                "HANDSON_COLLECTION_SESSION_ID",
                "",
            ),
            source_recorder_pid=os.getpid(),
        )
        print(
            "[PhysX post-collection certification] ensured "
            f"{queued} background certification job(s)",
            flush=True,
        )

    if poisoned_isaac_backend and _supervised_isaac_worker():
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(_ISAAC_FATAL_BACKEND_EXIT_CODE)

    if collection_error is not None:
        raise collection_error

if __name__ == "__main__":
    app.run(main)
