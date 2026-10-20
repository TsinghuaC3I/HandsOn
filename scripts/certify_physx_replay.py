#!/usr/bin/env python3
"""Certify task success from recorded hand motion in fresh Isaac physics."""

from __future__ import annotations

import argparse
import copy
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPO_ROOT / "handson"
for _path in (_PACKAGE_ROOT, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import zarr

from handson.data.physical_replay_certification import (
    CERTIFICATION_MODE,
    CLEAN_COMMIT_REVIEWER_UPGRADE_SOURCE_IDENTITY,
    CLEAN_COMMIT_SOURCE_IDENTITY,
    CONTACT_SIGNATURE_OVERLAY_ALGORITHM,
    CONTACT_SIGNATURE_OVERLAY_FORMAT,
    CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD,
    CONTACT_SIGNATURE_OVERLAY_VERSION,
    DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE,
    DETACHED_HAND_ACTION_INPUT_MODE,
    DETACHED_HAND_ACTION_TRANSITION_CONTRACT,
    DIRTY_CONTENT_SOURCE_IDENTITY,
    DIRTY_SNAPSHOT_REVIEWER_UPGRADE_SOURCE_IDENTITY,
    FRESH_PROCESS_CERTIFICATION_MODE,
    FRESH_SCENE_INITIALIZATION_MODE,
    INITIALIZATION_MODE,
    INITIALIZATION_RESTORE_COUNT_FIELDS,
    INITIALIZATION_RESTORE_COUNTS_SOURCE,
    LEGACY_RAW_ACTION_INPUT_MODE,
    LEGACY_RAW_ACTION_TRANSITION_CONTRACT,
    PERSISTENT_RECORDER_LIFECYCLE,
    RECORDED_DRIVE_TARGET_INPUT_MODE,
    RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT,
    REPORT_FORMAT,
    REPORT_NAME,
    REPORT_VERSION,
    SLEEPING_ABC_ORIGIN_TASKS,
    SOURCE_REVIEWER_UPGRADE_ALLOWED_PATHS,
    _DEFERRED_CANONICAL_INITIALIZATION_HISTORY_FIELDS,
    _initialization_history_field_matches,
    _is_natural_settle_then_zero_history,
    _natural_settle_adaptive_history_errors,
    _physical_origin_geometry_history_errors,
    capture_certified_inputs,
)
from handson.data.recording_fields import capture_git_replay_state
from handson.data.recording_fields import capture_runtime_replay_identity
from handson.data.recording_fields import resolve_environment_recording_schema
from handson.data.physx_session_history import (
    DISCARDED_REPLAY_NAME,
    EVENTS_DIRECTORY_NAME,
    EVENT_DOCUMENT_NAME,
    FINALIZATION_NAME,
    SESSION_HEADER_NAME,
    load_discarded_trajectory,
    load_session_history,
)
from handson.runtime.collection_support import (
    BC011_COLLECTION_ANGLE_GUIDANCE_ENV,
)
from handson.sim.isaaclab.recording_origin import (
    DEFAULT_ANGULAR_VELOCITY_LIMIT,
    DEFAULT_JOINT_VELOCITY_LIMIT,
    DEFAULT_LINEAR_VELOCITY_LIMIT,
    DEFAULT_STABLE_SECONDS,
    RecordingOriginStabilityError,
    VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS,
    VELOCITY_CLAMP_SCOPE_GATE_ONLY,
    VELOCITY_CLAMP_SCOPE_NONE,
    is_abc_style_collection_task,
    stable_recording_origin_kwargs_for_task,
    stable_recording_origin_kwargs_from_report,
    stabilize_recording_origin,
)
from handson.sim.isaaclab.b320_bc017_contract import (
    BC017_PROPRIO_FLATTEN_ORDER,
    HIDDEN_OBSERVATION_FIELDS,
    PUBLIC_OBSERVATION_GROUPS,
    TASK_NAME as BC017_TASK_NAME,
)
from handson.sim.isaaclab.as019_screwdriver_handover_env import (
    AS019_HIDDEN_STATE_FIELDS,
    TASK_NAME as AS019_TASK_NAME,
)
from handson.sim.isaaclab.eval_comb06_season_and_stir_env import (
    XY6_HIDDEN_STATE_FIELDS,
    TASK_NAME as XY6_TASK_NAME,
)


_EXACT_ENVIRONMENT_NAMES = {
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "GARMENTLAB_ASSET_ROOT",
    "HAND",
    "MKL_NUM_THREADS",
    "NVIDIA_TF32_OVERRIDE",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
}

_OBSOLETE_EXACT_ROBOT_SUBSTEP_PHYSICAL_TRANSITION_CONTRACT = (
    "physx_state_t_plus_exact_robot_substep_path_t_produces_next_physx_state_t"
)
_EXACT_ROBOT_SUBSTEP_FIELDS = frozenset(
    {
        "physx_sim_time",
        "physx_substep_count",
        "physx_substep_sim_time",
        "physx_substep_robot_joint_position",
        "physx_substep_robot_joint_velocity",
        "physx_substep_robot_joint_valid",
        "physx_substep_robot_joint_target",
        "physx_substep_robot_joint_target_valid",
    }
)
_PHYSX_GROUND_TRUTH_SCHEMA_NAME = "handson.physx_ground_truth"
_PHYSX_GROUND_TRUTH_SCHEMA_VERSION = 6
_BC017_TASK_STATE_FIELD_NAMES = tuple(BC017_PROPRIO_FLATTEN_ORDER)
_BC017_TASK_STATE_LAYOUTS = {
    **{name: tuple(layout) for name, layout in PUBLIC_OBSERVATION_GROUPS},
    "bc017_hidden_state": tuple(HIDDEN_OBSERVATION_FIELDS),
}
_BC017_TASK_STATE_REPLAY_ATOLS = {
    name: (0.0 if name == "bc017_hidden_state" else 1.0e-6)
    for name in _BC017_TASK_STATE_FIELD_NAMES
}
_AS019_TASK_STATE_FIELD_NAMES = ("as019_hidden_state",)
_AS019_TASK_STATE_LAYOUTS = {
    "as019_hidden_state": tuple(AS019_HIDDEN_STATE_FIELDS),
}
_AS019_TASK_STATE_REPLAY_ATOLS = {"as019_hidden_state": 1.0e-6}
from handson.tasks.eval_comb01_cover_pitcher_and_box.recording import TASK_STATE_CONTRACT as _XY5_TASK_STATE_CONTRACT

_TASK_STATE_CONTRACTS = {
    XY6_TASK_NAME: {
        "field_names": ("xy6_hidden_state",),
        "layouts": {"xy6_hidden_state": tuple(XY6_HIDDEN_STATE_FIELDS)},
        "replay_atols": {"xy6_hidden_state": 1.0e-6},
        "hidden_field_names": ("xy6_hidden_state",),
        "source": "xy6_action_aligned_task_state",
    },
    BC017_TASK_NAME: {
        "field_names": _BC017_TASK_STATE_FIELD_NAMES,
        "layouts": _BC017_TASK_STATE_LAYOUTS,
        "replay_atols": _BC017_TASK_STATE_REPLAY_ATOLS,
        "hidden_field_names": ("bc017_hidden_state",),
        "source": "bc017_action_aligned_task_state",
    },
    AS019_TASK_NAME: {
        "field_names": _AS019_TASK_STATE_FIELD_NAMES,
        "layouts": _AS019_TASK_STATE_LAYOUTS,
        "replay_atols": _AS019_TASK_STATE_REPLAY_ATOLS,
        "hidden_field_names": ("as019_hidden_state",),
        "source": "as019_action_aligned_task_state",
    },
}
_P003_TASK_NAME = "p003_bimanual_folding_ruler_shape_configuration"
_TASK_CONTROLLER_STATE_FIELD = "physx_task_controller_state"
_CONTACT_SIGNATURE_FIELDS = frozenset(
    {
        "physx_contact_signature_valid",
        "physx_contact_signature_pair_count",
        "physx_contact_signature_pair_path_indices",
        "physx_contact_signature_pair_counts",
        "physx_contact_signature_pair_metrics",
    }
)
_SUBSTEP_CONTACT_SIGNATURE_FIELDS = frozenset(
    name.replace("physx_", "physx_substep_", 1) for name in _CONTACT_SIGNATURE_FIELDS
)
_REQUIRED_CONTACT_SIGNATURE_FIELDS = (
    _CONTACT_SIGNATURE_FIELDS
    | _SUBSTEP_CONTACT_SIGNATURE_FIELDS
    | {"physx_substep_count"}
)
_RAW_CONTACT_FIELDS = (
    "physx_contact_report_valid",
    "physx_contact_capture_source",
    "physx_contact_header_count",
    "physx_contact_header_ids",
    "physx_contact_header_meta",
    "physx_contact_point_count",
    "physx_contact_point_data",
    "physx_contact_point_face_indices",
    "physx_contact_point_material_ids",
    "physx_friction_anchor_count",
    "physx_friction_anchor_data",
)
PHYSICS_ONLY_EXECUTION_MODE = "physics_only_source_scene_no_step_render_unthrottled"
SESSION_REPLAY_EXECUTION_MODE = "single_process_ordered_episode_replay"
PLATFORM_REVIEW_SOURCE_IDENTITY = "platform_physical_review_no_source_gate"
_PHYSICS_ONLY_APP_WINDOW_STATE_ATTRIBUTE = "_handson_physics_only_app_window_state"
_P001_TASK_NAME = "t48_unfold_fan"
_P001_TARGET_OPEN_DEG_BY_LABEL = {
    "60deg": 60.0,
    "90deg": 90.0,
    "120deg": 120.0,
}


class CertificationError(RuntimeError):
    """Raised when an episode cannot satisfy the replay contract."""


@dataclass(frozen=True)
class SourceIdentityVerification:
    """Auditable result of binding replay to its exact recorded source."""

    source_identity: str
    portable: bool
    source_snapshot_commit: str | None = None
    reviewer_upgrade_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhysicalFieldSpec:
    name: str
    next_name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    replay_atol: float
    replay_restore_supported: bool = True
    replay_domain: str = "physx"


@dataclass(frozen=True)
class LoadedDemo:
    demo_dir: Path
    task_name: str
    collection_run_seed: int
    episode_reset_seed: int
    randomize: bool
    randomize_dynamics: bool
    arm_type: str
    hand_type: str
    render_mode: str
    image_obs: bool
    headless: bool
    target_label: str
    camera_screen_effect: bool
    ipad_screen_effect: bool
    environment_variables: dict[str, str]
    recorded_git: dict[str, Any]
    recorded_runtime_identity: dict[str, Any]
    recorded_physx_schema: dict[str, Any]
    recorded_physx_field_schema: dict[str, Any]
    stable_recording_origin: dict[str, Any]
    action_storage_layout: str
    actions: np.ndarray
    terminals: np.ndarray
    succeeds: np.ndarray
    field_specs: tuple[PhysicalFieldSpec, ...]
    current_fields: dict[str, Any]
    next_fields: dict[str, Any]
    physical_transition_contract: str = LEGACY_RAW_ACTION_TRANSITION_CONTRACT
    trajectory_input_mode: str = LEGACY_RAW_ACTION_INPUT_MODE
    fresh_recorder_process_per_attempt_required: bool = False
    fresh_scene_replay_required: bool = False
    recorder_lifecycle: str = ""
    source_process_reuse_enabled: bool = False
    source_scene_rebuild_per_reset: bool = False
    source_scene_generation: int = 0
    collection_seed_source: str = ""
    episode_reset_ordinal: int = -1
    episode_seed_derivation: str = ""
    source_recorder_pid: int | None = None
    session_history_root_relative_path: str = ""
    session_history_sha256: str = ""
    session_history_event_count: int = -1
    session_history_accepted_event_ordinal: int = -1
    contact_signature_overlay_segment: dict[str, Any] | None = None
    recorded_task_state_field_schema: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class LoadedSessionReplay:
    """One fully verified, immutable collection-session replay prefix."""

    path: Path
    session: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    finalization: dict[str, Any]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CertificationError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CertificationError(f"{path} must contain a JSON object.")
    return value


def _required_mapping(
    parent: Mapping[str, Any],
    key: str,
    source: Path,
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise CertificationError(f"{source} is missing object {key!r}.")
    return value


def _required_int(
    parent: Mapping[str, Any],
    key: str,
    source: Path,
) -> int:
    value = parent.get(key)
    if type(value) is not int:
        raise CertificationError(f"{source} is missing integer {key!r}.")
    return value


def _required_bool(
    parent: Mapping[str, Any],
    key: str,
    source: Path,
) -> bool:
    value = parent.get(key)
    if type(value) is not bool:
        raise CertificationError(f"{source} is missing boolean {key!r}.")
    return value


def _required_text(
    parent: Mapping[str, Any],
    key: str,
    source: Path,
) -> str:
    value = parent.get(key)
    if not isinstance(value, str):
        raise CertificationError(f"{source} is missing string {key!r}.")
    return value


_SESSION_HISTORY_MANIFEST_FIELDS = (
    "physical_replay_session_history_root_relative_path",
    "physical_replay_session_history_sha256",
    "physical_replay_session_history_event_count",
    "physical_replay_session_history_accepted_event_ordinal",
)


def _optional_session_history_manifest_contract(
    recording: Mapping[str, Any],
    source: Path,
) -> tuple[str, str, int, int]:
    """Parse an all-or-none session binding from an accepted demo manifest."""

    present = [name in recording for name in _SESSION_HISTORY_MANIFEST_FIELDS]
    if not any(present):
        # Direct, historical ``certify_demo`` callers remain readable.  The
        # session entry point requires these fields before constructing Isaac.
        return "", "", -1, -1
    if not all(present):
        missing = [
            name
            for name, is_present in zip(
                _SESSION_HISTORY_MANIFEST_FIELDS,
                present,
                strict=True,
            )
            if not is_present
        ]
        raise CertificationError(
            f"{source} has an incomplete PhysX session-history binding; "
            "missing: " + ", ".join(missing)
        )

    relative_path = recording[_SESSION_HISTORY_MANIFEST_FIELDS[0]]
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise CertificationError(
            "PhysX session-history root-relative path must be nonempty text."
        )
    relative_path = relative_path.strip()
    parsed_path = Path(relative_path)
    if parsed_path.is_absolute() or ".." in parsed_path.parts:
        raise CertificationError(
            "PhysX session-history path must stay relative to the output root."
        )

    sha256 = recording[_SESSION_HISTORY_MANIFEST_FIELDS[1]]
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in sha256)
    ):
        raise CertificationError("PhysX session-history manifest SHA-256 is invalid.")

    event_count = recording[_SESSION_HISTORY_MANIFEST_FIELDS[2]]
    accepted_event_ordinal = recording[_SESSION_HISTORY_MANIFEST_FIELDS[3]]
    if type(event_count) is not int or event_count <= 0:
        raise CertificationError(
            "PhysX session-history event count must be a positive integer."
        )
    if (
        type(accepted_event_ordinal) is not int
        or accepted_event_ordinal < 0
        or accepted_event_ordinal >= event_count
    ):
        raise CertificationError(
            "PhysX session accepted-event ordinal is outside its history."
        )
    return (
        relative_path,
        sha256.lower(),
        int(event_count),
        int(accepted_event_ordinal),
    )


def _validated_restore_counts(
    value: Any,
    *,
    label: str,
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise CertificationError(f"{label} must be an object.")
    if set(value) != INITIALIZATION_RESTORE_COUNT_FIELDS:
        raise CertificationError(
            f"{label} must contain exactly: "
            + ", ".join(sorted(INITIALIZATION_RESTORE_COUNT_FIELDS))
            + "."
        )
    if any(type(count) is not int or count < 0 for count in value.values()):
        raise CertificationError(f"{label} counts must be non-negative integers.")
    if not any(count > 0 for count in value.values()):
        raise CertificationError(f"{label} did not restore any physical entity.")
    return {str(name): int(count) for name, count in value.items()}


def _validated_velocity_clamp_contract(
    stable_origin: Mapping[str, Any],
) -> tuple[bool, str]:
    """Validate and normalize old and current initialization clamp reports."""

    raw_enabled = stable_origin.get("stabilization_velocity_clamp")
    raw_per_step_writes = stable_origin.get(
        "initialization_per_physics_step_velocity_writes"
    )
    if raw_enabled is None and raw_per_step_writes is None:
        enabled = False
    elif raw_enabled is None or raw_per_step_writes is None:
        present = raw_per_step_writes if raw_enabled is None else raw_enabled
        if type(present) is not bool:
            raise CertificationError(
                "Stable-origin velocity clamp flags must be booleans."
            )
        enabled = present
    elif type(raw_enabled) is not bool or (type(raw_per_step_writes) is not bool):
        raise CertificationError("Stable-origin velocity clamp flags must be booleans.")
    elif raw_enabled != raw_per_step_writes:
        raise CertificationError("Stable-origin velocity clamp flags must match.")
    else:
        enabled = raw_enabled

    raw_scope = stable_origin.get("stabilization_velocity_clamp_scope")
    if raw_scope is None:
        # Historical true/true reports wrote after strict-gate steps only.
        scope = VELOCITY_CLAMP_SCOPE_GATE_ONLY if enabled else VELOCITY_CLAMP_SCOPE_NONE
    elif not isinstance(raw_scope, str):
        raise CertificationError("Stable-origin velocity clamp scope must be a string.")
    else:
        scope = raw_scope
    approved_scopes = {
        VELOCITY_CLAMP_SCOPE_NONE,
        VELOCITY_CLAMP_SCOPE_GATE_ONLY,
        VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS,
    }
    if scope not in approved_scopes:
        raise CertificationError(
            f"Unsupported stable-origin velocity clamp scope: {scope!r}."
        )
    if enabled == (scope == VELOCITY_CLAMP_SCOPE_NONE):
        raise CertificationError(
            "Stable-origin velocity clamp flag and scope contradict each other."
        )

    raw_clamp_steps = stable_origin.get("stabilization_velocity_clamp_steps")
    raw_clamp_write = stable_origin.get("stabilization_velocity_clamp_write")
    if not enabled:
        disabled_steps_are_valid = raw_clamp_steps is None or (
            type(raw_clamp_steps) is int and raw_clamp_steps == 0
        )
        if not disabled_steps_are_valid or raw_clamp_write is not None:
            raise CertificationError(
                "Disabled stable-origin velocity clamp records unexpected writes."
            )
        return False, scope

    physics_steps = stable_origin.get("physics_steps")
    initial_settle_steps = stable_origin.get("initial_settle_steps")
    if (
        type(physics_steps) is not int
        or physics_steps <= 0
        or type(initial_settle_steps) is not int
        or initial_settle_steps < 0
        or initial_settle_steps > physics_steps
        or type(raw_clamp_steps) is not int
    ):
        raise CertificationError(
            "Enabled stable-origin velocity clamp is missing valid step counts."
        )
    expected_steps = (
        physics_steps
        if scope == VELOCITY_CLAMP_SCOPE_ALL_INITIALIZATION_STEPS
        else physics_steps - initial_settle_steps
    )
    if raw_clamp_steps != expected_steps or expected_steps <= 0:
        raise CertificationError(
            "Stable-origin velocity clamp write count contradicts its scope."
        )
    if not isinstance(raw_clamp_write, Mapping):
        raise CertificationError(
            "Enabled stable-origin velocity clamp is missing write counts."
        )
    legacy_fields = {
        "rigid_bodies",
        "articulation_roots",
        "articulation_dofs",
    }
    current_fields = legacy_fields | {"deformable_nodes"}
    fields = set(raw_clamp_write)
    if fields != current_fields and not (
        scope == VELOCITY_CLAMP_SCOPE_GATE_ONLY and fields == legacy_fields
    ):
        raise CertificationError(
            "Stable-origin velocity clamp write counts have invalid fields."
        )
    if any(
        type(count) is not int or count < 0 for count in raw_clamp_write.values()
    ) or not any(int(count) > 0 for count in raw_clamp_write.values()):
        raise CertificationError(
            "Stable-origin velocity clamp write counts must be non-negative "
            "integers and cover a task entity."
        )
    return True, scope


def _validated_replayed_stable_origin_history(
    recorded: Mapping[str, Any],
    reset_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the certifier to execute the recorded initialization history."""

    actual = reset_info.get("physical_recording_origin")
    if not isinstance(actual, Mapping):
        raise CertificationError(
            "Fresh-scene replay did not report its stable-origin history."
        )
    recorded_enabled, recorded_scope = _validated_velocity_clamp_contract(recorded)
    actual_enabled, actual_scope = _validated_velocity_clamp_contract(actual)
    if (actual_enabled, actual_scope) != (
        recorded_enabled,
        recorded_scope,
    ):
        raise CertificationError(
            "Fresh-scene replay changed the stable-origin velocity clamp flag or scope."
        )
    recorded_natural_settle = _is_natural_settle_then_zero_history(recorded)
    actual_natural_settle = _is_natural_settle_then_zero_history(actual)
    if actual_natural_settle != recorded_natural_settle:
        raise CertificationError(
            "Fresh-scene replay changed the natural-settle/one-time-zero "
            "initialization strategy."
        )
    geometry_errors = [
        *_physical_origin_geometry_history_errors(
            recorded,
            label="recorded stable origin",
        ),
        *_physical_origin_geometry_history_errors(
            actual,
            label="replayed stable origin",
        ),
    ]
    if geometry_errors:
        raise CertificationError("; ".join(geometry_errors))
    if recorded_natural_settle:
        adaptive_errors = [
            *_natural_settle_adaptive_history_errors(
                recorded,
                label="recorded stable origin",
            ),
            *_natural_settle_adaptive_history_errors(
                actual,
                label="replayed stable origin",
            ),
        ]
        if adaptive_errors:
            raise CertificationError("; ".join(adaptive_errors))

    discrete_fields = (
        "initial_settle_steps",
        "stabilization_velocity_clamp_steps",
    )
    if not recorded_natural_settle:
        discrete_fields = ("physics_steps", *discrete_fields)
    matched: dict[str, Any] = {
        "stabilization_velocity_clamp": recorded_enabled,
        "stabilization_velocity_clamp_scope": recorded_scope,
    }
    for name in discrete_fields:
        recorded_value = recorded.get(name)
        actual_value = actual.get(name)
        if name == "stabilization_velocity_clamp_steps" and not (recorded_enabled):
            recorded_value = 0 if recorded_value is None else recorded_value
            actual_value = 0 if actual_value is None else actual_value
        if (
            type(recorded_value) is not int
            or type(actual_value) is not int
            or actual_value != recorded_value
        ):
            raise CertificationError(
                "Fresh-scene replay changed stable-origin initialization "
                f"history at {name}: recorded={recorded_value!r}, "
                f"actual={actual_value!r}."
            )
        matched[name] = int(actual_value)

    recorded_write = recorded.get("stabilization_velocity_clamp_write")
    actual_write = actual.get("stabilization_velocity_clamp_write")
    if recorded_enabled:
        assert isinstance(recorded_write, Mapping)
        assert isinstance(actual_write, Mapping)
        write_fields = (
            "rigid_bodies",
            "articulation_roots",
            "articulation_dofs",
            "deformable_nodes",
        )
        normalized_recorded_write = {
            name: int(recorded_write.get(name, 0)) for name in write_fields
        }
        normalized_actual_write = {
            name: int(actual_write.get(name, 0)) for name in write_fields
        }
        if normalized_actual_write != normalized_recorded_write:
            raise CertificationError(
                "Fresh-scene replay changed stable-origin velocity write "
                "counts: "
                f"recorded={normalized_recorded_write!r}, "
                f"actual={normalized_actual_write!r}."
            )
        matched["stabilization_velocity_clamp_write"] = normalized_actual_write
    elif recorded_write is not None or actual_write is not None:
        raise CertificationError(
            "Disabled fresh-scene velocity clamp reported an unexpected write."
        )
    else:
        matched["stabilization_velocity_clamp_write"] = None

    # Keep this report projection identical to the upload gate's canonical
    # initialization contract.  Historical manifests can still omit fields,
    # while current deferred manifests contain every field and therefore
    # produce a complete gate-ready replay history.
    for name in _DEFERRED_CANONICAL_INITIALIZATION_HISTORY_FIELDS:
        if name not in recorded:
            continue
        recorded_value = _json_safe(recorded[name])
        actual_value = _json_safe(actual.get(name))
        if name not in actual or not _initialization_history_field_matches(
            name,
            recorded_value,
            actual_value,
            natural_settle=recorded_natural_settle,
        ):
            raise CertificationError(
                "Fresh-scene replay changed stable-origin call history at "
                f"{name}: recorded={recorded_value!r}, "
                f"actual={actual_value!r}."
            )
        matched[name] = actual_value
    return matched


def _validated_exact_session_stable_origin_history(
    recorded: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> dict[str, Any]:
    """Require identical initialization step/call history inside a session.

    A standalone diagnostic may accept an adaptive natural-settle duration.
    Session replay cannot: one omitted or added PhysX step changes the hidden
    solver/cache prefix inherited by every later accepted demo.
    """

    if type(recorded.get("physics_steps")) is not int:
        raise CertificationError(
            "Journaled accepted reset has no exact physics_steps count."
        )
    geometry_errors = [
        *_physical_origin_geometry_history_errors(
            recorded,
            label="recorded session stable origin",
        ),
        *_physical_origin_geometry_history_errors(
            actual,
            label="replayed session stable origin",
        ),
    ]
    if geometry_errors:
        raise CertificationError("; ".join(geometry_errors))
    matched: dict[str, Any] = {}
    for name in _DEFERRED_CANONICAL_INITIALIZATION_HISTORY_FIELDS:
        if name not in recorded:
            continue
        if name not in actual or _json_safe(actual[name]) != _json_safe(recorded[name]):
            raise CertificationError(
                "Session replay changed exact stable-origin call history at "
                f"{name}: recorded={recorded.get(name)!r}, "
                f"actual={actual.get(name)!r}."
            )
        matched[name] = _json_safe(actual[name])
    if matched.get("physics_steps") != recorded.get("physics_steps"):
        raise CertificationError(
            "Session replay changed the accepted reset physics-step count."
        )
    return matched


def _field_semantics(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    value = entry.get("semantics")
    return value if isinstance(value, Mapping) else {}


def _canonical_physx_schema(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove process-local identifiers while retaining path construction."""

    canonical = copy.deepcopy(dict(value))
    contact_report = canonical.get("contact_report")
    if isinstance(contact_report, dict):
        path_table = contact_report.get("path_table")
        if isinstance(path_table, list):
            paths = sorted(
                str(entry.get("path"))
                for entry in path_table
                if isinstance(entry, Mapping) and isinstance(entry.get("path"), str)
            )
            contact_report["path_table"] = paths
    return canonical


def _flatten_physics_configuration_values(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Return the exact leaf values covered by a physics-config match."""

    flattened: dict[str, Any] = {}
    for raw_name, item in value.items():
        name = f"{prefix}.{raw_name}" if prefix else str(raw_name)
        if isinstance(item, Mapping):
            flattened.update(_flatten_physics_configuration_values(item, prefix=name))
        else:
            flattened[name] = _json_safe(item)
    return {name: flattened[name] for name in sorted(flattened)}


def _complete_physics_configuration_match_evidence(
    match: Mapping[str, Any],
    recorded_schema: Mapping[str, Any],
    runtime_schema: Mapping[str, Any],
) -> dict[str, Any]:
    recorded_physx = recorded_schema.get("physx")
    runtime_physx = runtime_schema.get("physx")
    recorded_config = (
        recorded_physx.get("physics_config")
        if isinstance(recorded_physx, Mapping)
        else None
    )
    runtime_config = (
        runtime_physx.get("physics_config")
        if isinstance(runtime_physx, Mapping)
        else None
    )
    if not isinstance(recorded_config, Mapping) or not isinstance(
        runtime_config,
        Mapping,
    ):
        raise ValueError("PhysX physics_config evidence is unavailable")
    result = dict(match)
    result["recorded_values"] = _flatten_physics_configuration_values(recorded_config)
    result["runtime_values"] = _flatten_physics_configuration_values(runtime_config)
    return result


def _validated_contact_signature_path_count(
    physx_schema: Mapping[str, Any],
) -> int:
    """Validate stable path identity and non-empty contact sensor coverage."""

    contact = physx_schema.get("contact_report")
    if not isinstance(contact, Mapping):
        raise CertificationError(
            "PhysX schema is missing canonical contact-report construction."
        )
    path_table = contact.get("path_table")
    if not isinstance(path_table, list) or not path_table:
        raise CertificationError(
            "Canonical contact replay requires a non-empty path table."
        )
    paths = []
    process_ids = []
    for entry in path_table:
        if (
            not isinstance(entry, Mapping)
            or type(entry.get("id")) is not int
            or not isinstance(entry.get("path"), str)
            or not entry["path"]
        ):
            raise CertificationError(
                "Canonical contact replay path mapping is incomplete."
            )
        process_ids.append(entry["id"] & 0xFFFFFFFFFFFFFFFF)
        paths.append(entry["path"])
    if len(set(paths)) != len(paths) or len(set(process_ids)) != len(process_ids):
        raise CertificationError(
            "Canonical contact replay path mapping is not one-to-one."
        )
    report_paths = contact.get("report_enabled_task_body_paths")
    sensor_paths = contact.get("tensor_sensor_paths")
    filter_paths = contact.get("tensor_filter_paths")
    report_coverage = isinstance(report_paths, list) and bool(report_paths)
    tensor_coverage = (
        isinstance(sensor_paths, list)
        and bool(sensor_paths)
        and isinstance(filter_paths, list)
        and bool(filter_paths)
    )
    if not report_coverage and not tensor_coverage:
        raise CertificationError(
            "Canonical contact replay has empty reporter/sensor coverage."
        )
    covered_paths = (list(report_paths) if report_coverage else []) + (
        [*sensor_paths, *filter_paths] if tensor_coverage else []
    )
    if any(path not in paths for path in covered_paths):
        raise CertificationError(
            "Canonical contact replay coverage is missing from the path table."
        )
    return len(paths)


def _validate_contact_signature_rows(
    *,
    valid: np.ndarray,
    pair_count: np.ndarray,
    pair_paths: np.ndarray,
    pair_counts: np.ndarray,
    pair_metrics: np.ndarray,
    path_count: int,
    label: str,
    require_valid: bool,
) -> None:
    if valid.shape != (1,) or pair_count.shape != (1,):
        raise CertificationError(f"{label} contact signature shape is invalid.")
    if require_valid and int(valid[0]) != 1:
        raise CertificationError(
            f"{label} has no certifiable canonical contact signature."
        )
    if not require_valid and int(valid[0]) != 0:
        raise CertificationError(
            f"{label} padded contact signature must be invalid/zero."
        )
    count = int(pair_count[0])
    capacity = pair_paths.shape[0]
    if (
        pair_paths.ndim != 2
        or pair_paths.shape[1:] != (4,)
        or pair_counts.shape != (capacity, 2)
        or pair_metrics.shape != (capacity, 3)
        or count < 0
        or count > capacity
    ):
        raise CertificationError(f"{label} contact signature layout is invalid.")
    if not np.all(np.isfinite(pair_metrics)) or np.any(pair_metrics < 0.0):
        raise CertificationError(f"{label} contact signature metrics are invalid.")
    if count:
        active_paths = pair_paths[:count]
        if np.any(active_paths == 0) or np.any(active_paths > path_count):
            raise CertificationError(
                f"{label} contact signature path identity is unresolved."
            )
        rows = [tuple(int(value) for value in row) for row in active_paths]
        if any((row[0], row[1]) > (row[2], row[3]) for row in rows) or rows != sorted(
            set(rows)
        ):
            raise CertificationError(
                f"{label} contact signature topology is not canonical."
            )
        if np.any(np.sum(pair_counts[:count], axis=1, dtype=np.uint64) == 0):
            raise CertificationError(
                f"{label} contains an inactive canonical contact pair."
            )
    if (
        np.any(pair_paths[count:] != 0)
        or np.any(pair_counts[count:] != 0)
        or np.any(pair_metrics[count:] != 0.0)
    ):
        raise CertificationError(f"{label} contact signature has non-zero padding.")


def _validate_recorded_contact_signatures(
    data: Any,
    *,
    steps: int,
    path_count: int,
) -> None:
    """Reject invalid control/substep contact evidence before replay starts."""

    for next_prefix in ("", "next_"):
        control = {
            name: np.asarray(data[f"{next_prefix}{name}"])
            for name in _CONTACT_SIGNATURE_FIELDS
        }
        for step in range(steps):
            _validate_contact_signature_rows(
                valid=control["physx_contact_signature_valid"][step],
                pair_count=control["physx_contact_signature_pair_count"][step],
                pair_paths=control["physx_contact_signature_pair_path_indices"][step],
                pair_counts=control["physx_contact_signature_pair_counts"][step],
                pair_metrics=control["physx_contact_signature_pair_metrics"][step],
                path_count=path_count,
                label=f"{next_prefix or 'current_'}control frame {step}",
                require_valid=True,
            )

        substep = {
            name: np.asarray(data[f"{next_prefix}{name}"])
            for name in _SUBSTEP_CONTACT_SIGNATURE_FIELDS
        }
        substep_counts = np.asarray(data[f"{next_prefix}physx_substep_count"])
        substep_capacity = substep["physx_substep_contact_signature_valid"].shape[1]
        for step in range(steps):
            count = int(substep_counts[step, 0])
            if count < 0 or count > substep_capacity:
                raise CertificationError(
                    f"{next_prefix or 'current_'}substep count is invalid."
                )
            for substep_index in range(substep_capacity):
                _validate_contact_signature_rows(
                    valid=substep["physx_substep_contact_signature_valid"][
                        step, substep_index
                    ],
                    pair_count=substep["physx_substep_contact_signature_pair_count"][
                        step, substep_index
                    ],
                    pair_paths=substep[
                        "physx_substep_contact_signature_pair_path_indices"
                    ][step, substep_index],
                    pair_counts=substep["physx_substep_contact_signature_pair_counts"][
                        step, substep_index
                    ],
                    pair_metrics=substep[
                        "physx_substep_contact_signature_pair_metrics"
                    ][step, substep_index],
                    path_count=path_count,
                    label=(
                        f"{next_prefix or 'current_'}frame {step} "
                        f"substep {substep_index}"
                    ),
                    require_valid=substep_index < count,
                )


class _ReadOnlyArrayOverlay:
    """Project copied replacement arrays without mutating source storage."""

    def __init__(self, source: Any, overrides: Mapping[str, np.ndarray]):
        self._source = source
        self._overrides = dict(overrides)

    def __getitem__(self, name: str) -> Any:
        return self._overrides.get(name, self._source[name])


def _contact_storage_name(
    base_name: str,
    *,
    next_prefix: str,
    substep: bool,
) -> str:
    name = base_name.replace("physx_", "physx_substep_", 1) if substep else base_name
    return f"{next_prefix}{name}"


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _digest_token(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _digest_array(digest: Any, label: str, value: Any) -> None:
    array = np.asarray(value)
    target_dtype = array.dtype.newbyteorder("<")
    normalized = np.ascontiguousarray(array.astype(target_dtype, copy=False))
    metadata = json.dumps(
        {
            "label": label,
            "dtype": target_dtype.str,
            "shape": list(normalized.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    _digest_token(digest, metadata)
    _digest_token(digest, normalized.tobytes(order="C"))


def _validate_overlay_storage_array(
    data: Any,
    field_schema: Mapping[str, Any],
    storage_name: str,
    *,
    steps: int,
) -> None:
    field_name = storage_name.removeprefix("next_")
    entry = field_schema.get(field_name)
    if not isinstance(entry, Mapping):
        raise CertificationError(
            "Recorded data has no certifiable canonical contact signature; "
            f"legacy overlay schema field {field_name} is unavailable."
        )
    try:
        expected_shape = (steps, *(int(value) for value in entry["shape"]))
        expected_dtype = np.dtype(entry["dtype"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CertificationError(
            f"Legacy contact overlay schema for {field_name} is invalid."
        ) from exc
    if storage_name not in data:
        raise CertificationError(
            "Recorded data has no certifiable canonical contact signature; "
            f"legacy overlay field {storage_name} is unavailable."
        )
    array = data[storage_name]
    if tuple(array.shape) != expected_shape or np.dtype(array.dtype) != expected_dtype:
        raise CertificationError(
            f"Legacy contact overlay field {storage_name} conflicts with its "
            "recorded schema."
        )


def _strict_legacy_tensor_contact_frame(
    fields: Mapping[str, np.ndarray],
    *,
    physx_schema: Mapping[str, Any],
    label: str,
) -> int:
    """Validate one legacy tensor frame and return exact zero-row count."""

    contact = physx_schema.get("contact_report")
    if not isinstance(contact, Mapping):
        raise CertificationError(
            f"{label} has no recorded contact-report construction."
        )
    required_contract = {
        "available_in_this_runtime": True,
        "tensor_fallback_available_in_this_runtime": True,
        "tensor_event_type_sentinel": int(np.uint32(0xFFFFFFFE)),
        "tensor_source_actor_and_collider_ids_are_body_ids": True,
        "tensor_source_face_and_material_ids_available": False,
        "tensor_source_friction_rows_are_points_not_cache_anchors": True,
        "replay_restore_supported": False,
    }
    if any(
        contact.get(name) != expected for name, expected in required_contract.items()
    ):
        raise CertificationError(
            f"{label} has no certifiable canonical contact signature; tensor "
            "contact construction is not the sealed legacy contract."
        )
    capture_encoding = contact.get("capture_source_encoding")
    if not isinstance(capture_encoding, Mapping) or (
        capture_encoding.get("2") != "rigid_contact_tensor"
    ):
        raise CertificationError(
            f"{label} does not bind capture source 2 to rigid_contact_tensor."
        )

    path_table = contact.get("path_table")
    sensor_paths = contact.get("tensor_sensor_paths")
    filter_paths = contact.get("tensor_filter_paths")
    if (
        not isinstance(path_table, list)
        or not path_table
        or not isinstance(sensor_paths, list)
        or not sensor_paths
        or not isinstance(filter_paths, list)
        or not filter_paths
        or any(not isinstance(path, str) or not path for path in sensor_paths)
        or any(not isinstance(path, str) or not path for path in filter_paths)
        or sensor_paths != list(dict.fromkeys(sensor_paths))
        or filter_paths != list(dict.fromkeys(filter_paths))
    ):
        raise CertificationError(
            f"{label} tensor contact coverage is incomplete or ambiguous."
        )
    id_to_path: dict[int, str] = {}
    path_to_id: dict[str, int] = {}
    for entry in path_table:
        if (
            not isinstance(entry, Mapping)
            or type(entry.get("id")) is not int
            or not isinstance(entry.get("path"), str)
            or not entry.get("path")
        ):
            raise CertificationError(f"{label} contact path table is malformed.")
        process_id = int(entry["id"]) & 0xFFFFFFFFFFFFFFFF
        path = str(entry["path"])
        if process_id in id_to_path or path in path_to_id:
            raise CertificationError(f"{label} contact path table is not one-to-one.")
        id_to_path[process_id] = path
        path_to_id[path] = process_id
    if any(path not in path_to_id for path in (*sensor_paths, *filter_paths)):
        raise CertificationError(
            f"{label} tensor contact coverage is absent from the path table."
        )

    report_valid = np.asarray(fields["physx_contact_report_valid"])
    capture_source = np.asarray(fields["physx_contact_capture_source"])
    header_count_value = np.asarray(fields["physx_contact_header_count"])
    point_count_value = np.asarray(fields["physx_contact_point_count"])
    anchor_count_value = np.asarray(fields["physx_friction_anchor_count"])
    if any(
        value.shape != (1,)
        for value in (
            report_valid,
            capture_source,
            header_count_value,
            point_count_value,
            anchor_count_value,
        )
    ):
        raise CertificationError(f"{label} contact scalar layout is invalid.")
    if int(report_valid[0]) != 1 or int(capture_source[0]) != 2:
        raise CertificationError(
            f"{label} has no certifiable canonical contact signature; it is "
            "not a valid rigid-contact-tensor capture."
        )

    header_ids = np.asarray(fields["physx_contact_header_ids"])
    header_meta = np.asarray(fields["physx_contact_header_meta"])
    point_data = np.asarray(fields["physx_contact_point_data"])
    face_indices = np.asarray(fields["physx_contact_point_face_indices"])
    material_ids = np.asarray(fields["physx_contact_point_material_ids"])
    anchor_data = np.asarray(fields["physx_friction_anchor_data"])
    if (
        header_ids.ndim != 2
        or header_ids.shape[1] != 5
        or header_meta.shape != (header_ids.shape[0], 7)
        or point_data.ndim != 2
        or point_data.shape[1] != 10
        or face_indices.shape != (point_data.shape[0], 2)
        or material_ids.shape != (point_data.shape[0], 2)
        or anchor_data.ndim != 2
        or anchor_data.shape[1] != 6
        or contact.get("max_headers") != header_ids.shape[0]
        or contact.get("max_contact_points") != point_data.shape[0]
        or contact.get("max_friction_anchors") != anchor_data.shape[0]
    ):
        raise CertificationError(f"{label} raw contact layout is invalid.")
    header_count = int(header_count_value[0])
    point_count = int(point_count_value[0])
    anchor_count = int(anchor_count_value[0])
    if (
        header_count < 0
        or header_count > header_ids.shape[0]
        or point_count < 0
        or point_count > point_data.shape[0]
        or anchor_count < 0
        or anchor_count > anchor_data.shape[0]
    ):
        raise CertificationError(f"{label} raw contact count is out of range.")
    if (
        np.any(header_ids[header_count:] != 0)
        or np.any(header_meta[header_count:] != 0)
        or np.any(point_data[point_count:] != 0.0)
        or np.any(face_indices[point_count:] != 0)
        or np.any(material_ids != 0)
        or np.any(anchor_data[anchor_count:] != 0.0)
    ):
        raise CertificationError(f"{label} raw contact padding is non-zero.")
    if not np.all(np.isfinite(point_data[:point_count])) or not np.all(
        np.isfinite(anchor_data[:anchor_count])
    ):
        raise CertificationError(f"{label} raw contact payload is non-finite.")
    if point_count and np.any(face_indices[:point_count] != np.iinfo(np.uint32).max):
        raise CertificationError(f"{label} tensor contact face sentinels are invalid.")

    point_coverage = np.zeros(point_count, dtype=np.uint32)
    anchor_coverage = np.zeros(anchor_count, dtype=np.uint32)
    seen_pairs: set[tuple[str, str]] = set()
    stage_ids: set[int] = set()
    event_sentinel = int(contact["tensor_event_type_sentinel"])
    uint32_sentinel = int(np.iinfo(np.uint32).max)
    for header_index in range(header_count):
        ids = header_ids[header_index]
        meta = header_meta[header_index]
        if (
            int(ids[0]) != int(ids[2])
            or int(ids[1]) != int(ids[3])
            or int(meta[0]) != event_sentinel
            or int(meta[5]) != uint32_sentinel
            or int(meta[6]) != uint32_sentinel
        ):
            raise CertificationError(
                f"{label} tensor contact header {header_index} is invalid."
            )
        try:
            sensor_path = id_to_path[int(ids[0])]
            filter_path = id_to_path[int(ids[1])]
        except KeyError as exc:
            raise CertificationError(
                f"{label} tensor contact header {header_index} has an "
                "unresolved path id."
            ) from exc
        if sensor_path not in sensor_paths or filter_path not in filter_paths:
            raise CertificationError(
                f"{label} tensor contact header {header_index} is outside "
                "recorded sensor/filter coverage."
            )
        pair = tuple(sorted((sensor_path, filter_path)))
        if pair in seen_pairs:
            raise CertificationError(
                f"{label} tensor contact pair {pair!r} is duplicated."
            )
        seen_pairs.add(pair)
        stage_ids.add(int(ids[4]))
        contact_offset = int(meta[1])
        contact_rows = int(meta[2])
        friction_offset = int(meta[3])
        friction_rows = int(meta[4])
        if (
            (contact_rows <= 0 and friction_rows <= 0)
            or contact_offset < 0
            or contact_rows < 0
            or friction_offset < 0
            or friction_rows < 0
            or contact_offset + contact_rows > point_count
            or friction_offset + friction_rows > anchor_count
        ):
            raise CertificationError(
                f"{label} tensor contact header {header_index} has an invalid "
                "active slice."
            )
        point_coverage[contact_offset : contact_offset + contact_rows] += 1
        anchor_coverage[friction_offset : friction_offset + friction_rows] += 1
    if len(stage_ids) > 1 or (stage_ids and next(iter(stage_ids)) == 0):
        raise CertificationError(f"{label} tensor contact stage id is invalid.")
    if (point_count and not np.all(point_coverage == 1)) or (
        anchor_count and not np.all(anchor_coverage == 1)
    ):
        raise CertificationError(f"{label} raw contact header coverage is not exact.")

    active_points = point_data[:point_count]
    zero_padding = np.all(active_points == 0.0, axis=1)
    zero_count = int(np.count_nonzero(zero_padding))
    if zero_count <= 0:
        raise CertificationError(
            f"{label} has no exact ten-field zero tensor padding row."
        )
    non_padding = active_points[~zero_padding]
    if non_padding.size:
        normal_norms = np.linalg.norm(
            np.asarray(non_padding[:, 3:6], dtype=np.float64),
            axis=1,
        )
        if np.any(normal_norms <= 1.0e-12):
            raise CertificationError(
                f"{label} contains a non-padding contact row with an invalid normal."
            )

    for name in _CONTACT_SIGNATURE_FIELDS:
        value = np.asarray(fields[name])
        if np.any(value != 0):
            raise CertificationError(
                f"{label} legacy stored signature is not canonical zero/invalid."
            )
    return zero_count


def _contact_signature_overlay_envelope(
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not segments:
        return None
    ordered = sorted(
        (_json_safe(dict(segment)) for segment in segments),
        key=lambda segment: str(segment["segment_id"]),
    )
    payload = {
        "format": CONTACT_SIGNATURE_OVERLAY_FORMAT,
        "version": CONTACT_SIGNATURE_OVERLAY_VERSION,
        "algorithm": CONTACT_SIGNATURE_OVERLAY_ALGORITHM,
        "source_open_mode": "read_only",
        "source_files_mutated": False,
        "segments": ordered,
        "segment_count": len(ordered),
        "signature_frames_reconstructed": sum(
            int(segment["signature_frames_reconstructed"]) for segment in ordered
        ),
        "exact_zero_padding_rows_removed": sum(
            int(segment["exact_zero_padding_rows_removed"]) for segment in ordered
        ),
    }
    return {
        **payload,
        "digest_sha256": _canonical_json_sha256(payload),
    }


def _apply_legacy_contact_signature_overlay(
    data: Any,
    *,
    steps: int,
    physx_schema: Mapping[str, Any],
    field_schema: Mapping[str, Any],
    segment_id: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
    """Rebuild only legacy zero-padding signatures into copied arrays."""

    locations: list[tuple[str, bool, int, int | None]] = []
    frames_checked = 0
    for next_prefix in ("", "next_"):
        control_name = f"{next_prefix}physx_contact_signature_valid"
        control_valid = np.asarray(data[control_name])
        if control_valid.shape != (steps, 1):
            raise CertificationError(
                f"{control_name} has an invalid legacy-overlay shape."
            )
        for step in range(steps):
            value = int(control_valid[step, 0])
            if value not in (0, 1):
                raise CertificationError(f"{control_name} must be binary.")
            frames_checked += 1
            if value == 0:
                locations.append((next_prefix, False, step, None))

        count_name = f"{next_prefix}physx_substep_count"
        substep_counts = np.asarray(data[count_name])
        valid_name = f"{next_prefix}physx_substep_contact_signature_valid"
        substep_valid = np.asarray(data[valid_name])
        if (
            substep_counts.shape != (steps, 1)
            or substep_valid.ndim != 3
            or substep_valid.shape[0] != steps
            or substep_valid.shape[2:] != (1,)
        ):
            raise CertificationError(
                f"{next_prefix or 'current_'}substep signature layout is invalid."
            )
        capacity = int(substep_valid.shape[1])
        for step in range(steps):
            count = int(substep_counts[step, 0])
            if count < 0 or count > capacity:
                raise CertificationError(
                    f"{next_prefix or 'current_'}substep count is invalid."
                )
            for substep_index in range(count):
                value = int(substep_valid[step, substep_index, 0])
                if value not in (0, 1):
                    raise CertificationError(f"{valid_name} must be binary.")
                frames_checked += 1
                if value == 0:
                    locations.append((next_prefix, True, step, substep_index))
    if not locations:
        return {}, None

    required_base_fields = (*_RAW_CONTACT_FIELDS, *_CONTACT_SIGNATURE_FIELDS)
    required_sampling_modes = {substep for _prefix, substep, _step, _sub in locations}
    for next_prefix in ("", "next_"):
        for substep in required_sampling_modes:
            for base_name in required_base_fields:
                _validate_overlay_storage_array(
                    data,
                    field_schema,
                    _contact_storage_name(
                        base_name,
                        next_prefix=next_prefix,
                        substep=substep,
                    ),
                    steps=steps,
                )

    signature_storage_names = tuple(
        _contact_storage_name(
            base_name,
            next_prefix=next_prefix,
            substep=substep,
        )
        for next_prefix in ("", "next_")
        for substep in (False, True)
        for base_name in _CONTACT_SIGNATURE_FIELDS
    )
    overrides = {
        name: np.asarray(data[name]).copy() for name in signature_storage_names
    }

    from handson.sim.isaaclab.physx_ground_truth import PhysxGroundTruthTrace

    contact = physx_schema.get("contact_report")
    assert isinstance(contact, Mapping)
    trace = object.__new__(PhysxGroundTruthTrace)
    trace.max_contact_headers = int(contact.get("max_headers", 0))
    trace.max_contact_points = int(contact.get("max_contact_points", 0))
    trace.max_friction_anchors = int(contact.get("max_friction_anchors", 0))
    trace._contact_report_paths = tuple(
        contact.get("report_enabled_task_body_paths", ())
    )
    trace._tensor_contact_sensor_paths = tuple(contact.get("tensor_sensor_paths", ()))
    trace._tensor_contact_filter_paths = tuple(contact.get("tensor_filter_paths", ()))
    trace._contact_report_path_table = tuple(contact.get("path_table", ()))

    contract_seed = {
        "format": CONTACT_SIGNATURE_OVERLAY_FORMAT,
        "version": CONTACT_SIGNATURE_OVERLAY_VERSION,
        "algorithm": CONTACT_SIGNATURE_OVERLAY_ALGORITHM,
        "segment_id": segment_id,
        "contact_report": dict(contact),
    }
    seed = json.dumps(
        _json_safe(contract_seed),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    coordinate_digest = hashlib.sha256()
    for digest in (raw_digest, output_digest, coordinate_digest):
        _digest_token(digest, seed)

    zero_padding_rows = 0
    control_frames = 0
    substep_frames = 0
    for next_prefix, substep, step, substep_index in locations:
        index: int | tuple[int, int] = (step, int(substep_index)) if substep else step
        location_label = (
            f"{next_prefix or 'current_'}"
            f"{'substep' if substep else 'control'}:{step}:"
            f"{substep_index if substep else '-'}"
        )
        _digest_token(coordinate_digest, location_label.encode("utf-8"))
        fields: dict[str, np.ndarray] = {}
        for base_name in required_base_fields:
            storage_name = _contact_storage_name(
                base_name,
                next_prefix=next_prefix,
                substep=substep,
            )
            source = overrides.get(storage_name, data[storage_name])
            fields[base_name] = np.asarray(source[index]).copy()
            _digest_array(
                raw_digest,
                f"{location_label}:{base_name}",
                fields[base_name],
            )
        zero_padding_rows += _strict_legacy_tensor_contact_frame(
            fields,
            physx_schema=physx_schema,
            label=location_label,
        )
        rebuilt = trace._canonicalize_contact_fields(fields)
        if int(rebuilt["physx_contact_signature_valid"][0]) != 1:
            raise CertificationError(
                f"{location_label} could not be deterministically rebuilt."
            )
        for base_name in _CONTACT_SIGNATURE_FIELDS:
            storage_name = _contact_storage_name(
                base_name,
                next_prefix=next_prefix,
                substep=substep,
            )
            overrides[storage_name][index] = rebuilt[base_name]
            _digest_array(
                output_digest,
                f"{location_label}:{base_name}",
                rebuilt[base_name],
            )
        if substep:
            substep_frames += 1
        else:
            control_frames += 1

    return overrides, {
        "segment_id": segment_id,
        "signature_frames_checked": int(frames_checked),
        "signature_frames_reconstructed": len(locations),
        "control_frames_reconstructed": int(control_frames),
        "substep_frames_reconstructed": int(substep_frames),
        "exact_zero_padding_rows_removed": int(zero_padding_rows),
        "coordinates_sha256": coordinate_digest.hexdigest(),
        "raw_input_sha256": raw_digest.hexdigest(),
        "canonical_output_sha256": output_digest.hexdigest(),
    }


def _shared_session_field_value(demo: LoadedDemo, field: str) -> Any:
    """Canonicalize process-local schema details before session comparison."""

    value = getattr(demo, field)
    if field == "recorded_physx_schema":
        value = _canonical_physx_schema(value)
    return _json_safe(value)


def _validate_runtime_trace_schema(
    demo: LoadedDemo,
    runtime_schema: Mapping[str, Any],
) -> None:
    if (
        runtime_schema.get("name") != _PHYSX_GROUND_TRUTH_SCHEMA_NAME
        or runtime_schema.get("version") != _PHYSX_GROUND_TRUTH_SCHEMA_VERSION
    ):
        raise CertificationError("Runtime PhysX trace schema name/version changed.")
    runtime_physx_schema = runtime_schema.get("physx")
    runtime_physx_fields = runtime_schema.get("fields")
    if not isinstance(runtime_physx_schema, Mapping) or (
        _canonical_physx_schema(runtime_physx_schema)
        != _canonical_physx_schema(demo.recorded_physx_schema)
    ):
        raise CertificationError(
            "Runtime PhysX construction/configuration schema changed."
        )
    if (
        not isinstance(runtime_physx_fields, Mapping)
        or _json_safe(runtime_physx_fields) != demo.recorded_physx_field_schema
    ):
        raise CertificationError(
            "Runtime PhysX field set, shapes, dtypes, replay flags, or "
            "tolerances differ from the recorded environment schema."
        )


def _validate_transition_continuity(
    name: str,
    current: Any,
    following: Any,
    *,
    steps: int,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> None:
    """Require stored state[t] to be the exact preceding next_state[t-1]."""

    if steps <= 1:
        return
    values_per_step = max(1, int(np.prod(shape, dtype=np.int64)))
    bytes_per_pair = max(1, values_per_step * int(dtype.itemsize) * 2)
    chunk_steps = max(1, min(steps - 1, (16 * 1024 * 1024) // bytes_per_pair))
    for start in range(1, steps, chunk_steps):
        stop = min(steps, start + chunk_steps)
        current_chunk = np.asarray(current[start:stop])
        previous_next_chunk = np.asarray(following[start - 1 : stop - 1])
        if not np.array_equal(
            current_chunk,
            previous_next_chunk,
            equal_nan=True,
        ):
            raise CertificationError(
                f"Stored transition continuity is broken for {name}: "
                "recorded state[t] must exactly equal next state[t-1]."
            )


def _validated_task_state_schema(
    task_name: str,
    environment_schema: Mapping[str, Any],
    environment_fields: Mapping[str, Any],
    stored_fields: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return one supported exact read-only task-state replay contract."""

    declared = {
        str(name): entry
        for name, entry in environment_fields.items()
        if isinstance(entry, Mapping)
        and entry.get("replay_domain") == "task_state"
    }
    contract = _TASK_STATE_CONTRACTS.get(task_name)
    if contract is None:
        if declared:
            raise CertificationError(
                "This task may not declare the task_state replay domain."
            )
        return {}

    field_names = tuple(contract["field_names"])
    layouts = dict(contract["layouts"])
    replay_atols = dict(contract["replay_atols"])
    hidden_field_names = tuple(contract["hidden_field_names"])
    source = str(contract["source"])
    expected_names = set(field_names)
    if set(declared) != expected_names:
        raise CertificationError(
            f"{task_name} task_state schema must contain exactly "
            f"{sorted(expected_names)}."
        )
    task_state = environment_schema.get("task_state")
    expected_task_state = {
        "task_name": task_name,
        "field_names": list(field_names),
        "hidden_field_names": list(hidden_field_names),
        "capture_mode": "action_aligned_current_and_next_observation",
        "replay_policy": "compare_every_step_never_restore",
    }
    if not isinstance(task_state, Mapping) or dict(task_state) != expected_task_state:
        raise CertificationError(
            f"{task_name} task_state schema envelope is invalid."
        )

    result: dict[str, dict[str, Any]] = {}
    for name in field_names:
        entry = declared[name]
        expected_shape = [len(layouts[name])]
        expected_visibility = (
            "hidden_task_logic"
            if name in hidden_field_names
            else "public_task_state"
        )
        if (
            entry.get("shape") != expected_shape
            or entry.get("dtype") != "float64"
            or entry.get("source") != source
            or entry.get("layout") != list(layouts[name])
            or entry.get("visibility") != expected_visibility
            or entry.get("replay_domain") != "task_state"
            or entry.get("physical_replay") is not True
            or entry.get("replay_restore_supported") is not False
            or entry.get("replay_atol") != replay_atols[name]
        ):
            raise CertificationError(
                f"{task_name} task_state field {name} has an invalid "
                "read-only replay contract."
            )
        for stored_name in (name, f"next_{name}"):
            stored_entry = stored_fields.get(stored_name)
            if (
                not isinstance(stored_entry, Mapping)
                or stored_entry.get("shape_per_step") != expected_shape
                or stored_entry.get("dtype") != "float64"
                or dict(_field_semantics(stored_entry)) != dict(entry)
            ):
                raise CertificationError(
                    f"{task_name} task_state field {stored_name} is not "
                    "paired with its environment schema."
                )
        result[name] = copy.deepcopy(dict(entry))
    return result


def _validate_runtime_task_state_schema(demo: LoadedDemo, env: Any) -> None:
    """Bind fresh replay to the recorded read-only task-state field contract."""

    if not demo.recorded_task_state_field_schema:
        return
    runtime_schema = resolve_environment_recording_schema(env)
    if not isinstance(runtime_schema, Mapping):
        raise CertificationError("Runtime task_state recording schema is unavailable.")
    runtime_fields = runtime_schema.get("fields")
    if not isinstance(runtime_fields, Mapping):
        raise CertificationError("Runtime task_state field schema is unavailable.")
    actual = {
        str(name): copy.deepcopy(dict(entry))
        for name, entry in runtime_fields.items()
        if isinstance(entry, Mapping)
        and entry.get("replay_domain") == "task_state"
    }
    if _json_safe(actual) != demo.recorded_task_state_field_schema:
        raise CertificationError(
            "Runtime task_state field set, shapes, layouts, replay flags, or "
            "tolerances differ from the recording."
        )


def load_demo(demo_dir: str | Path) -> LoadedDemo:
    demo_path = Path(demo_dir).expanduser().resolve()
    manifest_path = demo_path / "replay_manifest.json"
    schema_path = demo_path / "recording_schema.json"
    manifest = _read_json_object(manifest_path)
    schema = _read_json_object(schema_path)
    task = _required_mapping(manifest, "task", manifest_path)
    task_name = _required_text(task, "name", manifest_path)
    recording = _required_mapping(manifest, "recording", manifest_path)
    (
        session_history_root_relative_path,
        session_history_sha256,
        session_history_event_count,
        session_history_accepted_event_ordinal,
    ) = _optional_session_history_manifest_contract(recording, manifest_path)
    environment = _required_mapping(manifest, "environment", manifest_path)
    software = _required_mapping(manifest, "software", manifest_path)
    recorded_git = dict(_required_mapping(software, "git", manifest_path))
    recorded_runtime_identity = dict(
        _required_mapping(
            software,
            "runtime_replay_identity",
            manifest_path,
        )
    )

    if recording.get("physical_replay_state_recorded") is not True:
        raise CertificationError("Manifest does not declare a physical trace.")
    if recording.get("physical_replay_certification_required") is not True:
        raise CertificationError("Manifest does not require certification.")
    stable_origin = dict(
        _required_mapping(
            recording,
            "stable_recording_origin",
            manifest_path,
        )
    )
    stability_metric = stable_origin.get("stability_metric")
    if stability_metric not in {"velocity", "velocity_or_static_pose"}:
        raise CertificationError(
            "Physical replay certification requires the strict velocity gate "
            "or its bounded static-pose fallback."
        )
    velocity_clamp_enabled, velocity_clamp_scope = _validated_velocity_clamp_contract(
        stable_origin
    )
    manifest_task_name = str(task.get("name") or "").strip().lower()
    if is_abc_style_collection_task(manifest_task_name):
        if stable_origin.get("task_name") != manifest_task_name:
            raise CertificationError(
                "ABC stable-origin task identity does not match the manifest."
            )
        if stable_origin.get("stability_sample_steps") != 1:
            raise CertificationError(
                "ABC stable-origin certification must inspect every physics frame."
            )
    if stability_metric == "velocity_or_static_pose":
        task_name = str(task.get("name") or "").strip().lower()
        approved = stable_recording_origin_kwargs_for_task(task_name)
        if (
            approved.get("stability_metric") != "velocity_or_static_pose"
            or stable_origin.get("task_name") != task_name
        ):
            raise CertificationError(
                "The static-pose velocity fallback is restricted to approved "
                "task identities."
            )
        if task_name == "t26_place_fruit_in_bucket" and velocity_clamp_enabled:
            raise CertificationError(
                "Approved static-pose certification requires unclamped physics."
            )
        raw_fallback_values = (
            stable_origin.get("pose_translation_limit"),
            stable_origin.get("pose_rotation_limit"),
            stable_origin.get("linear_velocity_limit"),
            stable_origin.get("angular_velocity_limit"),
            stable_origin.get("joint_velocity_limit"),
            stable_origin.get("physics_dt"),
            stable_origin.get("stability_sample_interval_seconds"),
            stable_origin.get("stable_seconds_required"),
        )
        if any(type(value) is bool for value in raw_fallback_values):
            raise CertificationError(
                "Static-pose fallback values must be numeric, not boolean."
            )
        try:
            pose_translation_limit = float(stable_origin["pose_translation_limit"])
            pose_rotation_limit = float(stable_origin["pose_rotation_limit"])
            linear_velocity_limit = float(stable_origin["linear_velocity_limit"])
            angular_velocity_limit = float(stable_origin["angular_velocity_limit"])
            joint_velocity_limit = float(stable_origin["joint_velocity_limit"])
            physics_dt = float(stable_origin["physics_dt"])
            sample_interval_seconds = float(
                stable_origin["stability_sample_interval_seconds"]
            )
            stable_seconds_required = float(stable_origin["stable_seconds_required"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CertificationError(
                "Static-pose fallback limits are missing or invalid."
            ) from exc
        fallback_limits = (
            pose_translation_limit,
            pose_rotation_limit,
            linear_velocity_limit,
            angular_velocity_limit,
            joint_velocity_limit,
            physics_dt,
            sample_interval_seconds,
            stable_seconds_required,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in fallback_limits):
            raise CertificationError(
                "Static-pose fallback limits must be finite and positive."
            )
        if (
            pose_translation_limit > float(approved["pose_translation_limit"])
            or pose_rotation_limit > float(approved["pose_rotation_limit"])
            or linear_velocity_limit > DEFAULT_LINEAR_VELOCITY_LIMIT
            or angular_velocity_limit > DEFAULT_ANGULAR_VELOCITY_LIMIT
            or joint_velocity_limit > DEFAULT_JOINT_VELOCITY_LIMIT
            or not np.isclose(
                physics_dt,
                1.0 / 120.0,
                rtol=1.0e-12,
                atol=0.0,
            )
            or stable_seconds_required < DEFAULT_STABLE_SECONDS
        ):
            raise CertificationError(
                "Static-pose fallback exceeds its approved absolute contract."
            )
        if (
            stable_origin.get("full_frame_pose_gate") is not True
            or type(stable_origin.get("stability_sample_steps")) is not int
            or stable_origin.get("stability_sample_steps") != 1
            or not np.isclose(
                sample_interval_seconds,
                physics_dt,
                rtol=1.0e-12,
                atol=0.0,
            )
        ):
            raise CertificationError(
                "Static-pose fallback must inspect every physics frame."
            )
        if (
            pose_translation_limit > linear_velocity_limit * sample_interval_seconds
            or pose_rotation_limit > angular_velocity_limit * sample_interval_seconds
        ):
            raise CertificationError(
                "Static-pose fallback is looser than the velocity gate."
            )
        if stable_origin.get("gate_robot_velocity_stable") is not True:
            raise CertificationError(
                "Static-pose fallback requires strict robot velocity stability."
            )
        raw_robot_gate_values = (
            stable_origin.get("gate_max_robot_linear_velocity"),
            stable_origin.get("gate_max_robot_angular_velocity"),
            stable_origin.get("gate_max_robot_joint_velocity"),
        )
        if any(type(value) is bool for value in raw_robot_gate_values):
            raise CertificationError(
                "Static-pose robot gate values must be numeric, not boolean."
            )
        try:
            robot_gate_values = tuple(float(value) for value in raw_robot_gate_values)
        except (TypeError, ValueError) as exc:
            raise CertificationError(
                "Static-pose robot gate values are missing or invalid."
            ) from exc
        if (
            any(not np.isfinite(value) or value < 0.0 for value in robot_gate_values)
            or robot_gate_values[0] > linear_velocity_limit
            or robot_gate_values[1] > angular_velocity_limit
            or robot_gate_values[2] > joint_velocity_limit
        ):
            raise CertificationError(
                "Static-pose robot gate flag contradicts its recorded velocities."
            )
        task_velocity_stable = stable_origin.get("gate_task_velocity_stable") is True
        task_pose_stable = stable_origin.get("gate_task_static_pose_stable") is True
        fallback_samples = stable_origin.get("gate_static_pose_fallback_samples")
        pose_samples = stable_origin.get("gate_pose_samples")
        stable_steps_required = stable_origin.get("stable_steps_required")
        stable_steps_observed = stable_origin.get("stable_steps_observed")
        if (
            type(stable_steps_required) is not int
            or type(stable_steps_observed) is not int
            or stable_steps_required <= 0
            or stable_steps_observed < stable_steps_required
            or stable_steps_required
            != int(np.ceil(stable_seconds_required / physics_dt))
        ):
            raise CertificationError(
                "Static-pose fallback is missing its full gate window."
            )
        if not task_velocity_stable and not task_pose_stable:
            raise CertificationError(
                "Origin satisfied neither task velocity nor static pose."
            )
        if velocity_clamp_enabled and (task_velocity_stable or not task_pose_stable):
            raise CertificationError(
                "Clamped A06 initialization must be certified by the full-frame "
                "static-pose path, not by synthetic zero task velocity."
            )
        if task_velocity_stable:
            raw_task_gate_values = (
                stable_origin.get("gate_max_task_linear_velocity"),
                stable_origin.get("gate_max_task_angular_velocity"),
            )
            if any(type(value) is bool for value in raw_task_gate_values):
                raise CertificationError(
                    "Static-pose task gate values must be numeric, not boolean."
                )
            try:
                task_gate_values = tuple(float(value) for value in raw_task_gate_values)
            except (TypeError, ValueError) as exc:
                raise CertificationError(
                    "Static-pose task gate values are missing or invalid."
                ) from exc
            if (
                any(not np.isfinite(value) or value < 0.0 for value in task_gate_values)
                or task_gate_values[0] > linear_velocity_limit
                or task_gate_values[1] > angular_velocity_limit
            ):
                raise CertificationError(
                    "Task velocity flag contradicts its recorded values."
                )
        if task_pose_stable:
            if (
                type(pose_samples) is not int
                or pose_samples < stable_steps_required
                or type(fallback_samples) is not int
                or fallback_samples < 0
                or fallback_samples > pose_samples
            ):
                raise CertificationError(
                    "Static-pose fallback did not cover its full gate window."
                )
            raw_pose_gate_values = (
                stable_origin.get("gate_max_task_translation_delta"),
                stable_origin.get("gate_max_task_rotation_delta"),
            )
            if any(type(value) is bool for value in raw_pose_gate_values):
                raise CertificationError(
                    "Static-pose gate values must be numeric, not boolean."
                )
            try:
                gate_translation_delta = float(raw_pose_gate_values[0])
                gate_rotation_delta = float(raw_pose_gate_values[1])
            except (KeyError, TypeError, ValueError) as exc:
                raise CertificationError(
                    "Accepted static-pose deltas are missing or invalid."
                ) from exc
            if (
                not np.isfinite(gate_translation_delta)
                or not np.isfinite(gate_rotation_delta)
                or gate_translation_delta < 0.0
                or gate_rotation_delta < 0.0
                or gate_translation_delta > pose_translation_limit
                or gate_rotation_delta > pose_rotation_limit
            ):
                raise CertificationError(
                    "Accepted static-pose deltas exceed their limits."
                )
            if velocity_clamp_enabled:
                raw_fallback_gate_values = (
                    stable_origin.get("gate_max_fallback_translation_delta"),
                    stable_origin.get("gate_max_fallback_rotation_delta"),
                )
                if any(type(value) is bool for value in raw_fallback_gate_values):
                    raise CertificationError("Fallback pose deltas must be numeric.")
                try:
                    fallback_gate_values = tuple(
                        float(value) for value in raw_fallback_gate_values
                    )
                except (TypeError, ValueError) as exc:
                    raise CertificationError(
                        "Fallback pose deltas are missing or invalid."
                    ) from exc
                if (
                    any(
                        not np.isfinite(value) or value < 0.0
                        for value in fallback_gate_values
                    )
                    or fallback_gate_values[0] > pose_translation_limit
                    or fallback_gate_values[1] > pose_rotation_limit
                ):
                    raise CertificationError(
                        "Fallback pose deltas exceed their limits."
                    )
        if not task_velocity_stable and (
            type(fallback_samples) is not int or fallback_samples <= 0
        ):
            raise CertificationError("Static-pose path contains no fallback frames.")
        if velocity_clamp_enabled and (
            type(fallback_samples) is not int
            or type(pose_samples) is not int
            or fallback_samples != pose_samples
        ):
            raise CertificationError(
                "Clamped A06 pose evidence must cover every accepted frame."
            )
    if stable_origin.get("per_frame_state_writes") is not False:
        raise CertificationError(
            "Stable-origin contract must forbid per-frame state writes."
        )
    if stable_origin.get("final_boundary_restore_sync") is not True:
        raise CertificationError(
            "Stable-origin contract must synchronously restore its final "
            "pre-action boundary."
        )
    _validated_restore_counts(
        stable_origin.get("final_boundary_state_write"),
        label="stable_recording_origin.final_boundary_state_write",
    )
    if stable_origin.get("final_boundary_physics_steps_after_write") != 0 or (
        type(stable_origin.get("final_boundary_physics_steps_after_write")) is not int
    ):
        raise CertificationError(
            "Stable-origin contract must execute zero physics steps after "
            "its final boundary write."
        )
    if (
        stable_origin.get("final_boundary_task_and_joint_velocity_exact_zero")
        is not True
    ):
        raise CertificationError(
            "Stable-origin final boundary must have exact zero task and "
            "robot-joint velocity."
        )
    post_settle_sleeping = stable_origin.get("post_settle_sleeping")
    if type(post_settle_sleeping) is not bool:
        raise CertificationError(
            "stable_recording_origin.post_settle_sleeping must be boolean."
        )
    if stable_origin.get("boundary_state_write_mode") != "velocity_only":
        raise CertificationError("Stable-origin boundary writes must be velocity_only.")
    stable_task_name = str(stable_origin.get("task_name") or "").strip().lower()
    is_abc_collection_task = is_abc_style_collection_task(stable_task_name)
    uses_awake_abc_contract = (
        is_abc_collection_task
        and stable_task_name not in SLEEPING_ABC_ORIGIN_TASKS
    )
    if uses_awake_abc_contract:
        if post_settle_sleeping is not False:
            raise CertificationError(
                "ABC stable origins must preserve the awake contact manifold."
            )
        if stable_origin.get("contact_warm_start_preserved") is not True:
            raise CertificationError(
                "ABC stable origins must attest contact_warm_start_preserved=true."
            )
        if stable_origin.get("post_settle_sleep") is not None:
            raise CertificationError(
                "Awake ABC stable origins must record post_settle_sleep=null."
            )
    else:
        if post_settle_sleeping is not True:
            raise CertificationError(
                "Sleeping task origins must retain the task-body sleep contract."
            )
        if stable_origin.get("contact_warm_start_preserved") is not False:
            raise CertificationError(
                "Sleeping task origins cannot attest preserved warm-start state."
            )
    final_boundary_sleep = stable_origin.get("final_boundary_sleep")
    if post_settle_sleeping:
        if not isinstance(final_boundary_sleep, Mapping):
            raise CertificationError(
                "Sleeping stable origin is missing final_boundary_sleep."
            )
        body_count = final_boundary_sleep.get("rigid_bodies")
        body_paths = final_boundary_sleep.get("body_paths")
        skipped_paths = final_boundary_sleep.get(
            "skipped_non_dynamic_or_non_path_names"
        )
        if (
            type(body_count) is not int
            or body_count < 0
            or not isinstance(body_paths, list)
            or not all(isinstance(path, str) for path in body_paths)
            or body_count != len(body_paths)
            or not isinstance(skipped_paths, list)
            or not all(isinstance(path, str) for path in skipped_paths)
        ):
            raise CertificationError(
                "stable_recording_origin.final_boundary_sleep is invalid."
            )
    elif final_boundary_sleep is not None:
        raise CertificationError(
            "Awake stable origin must record final_boundary_sleep=null."
        )
    if recording.get("action_semantics") != "exact_raw_environment_action_executed":
        raise CertificationError("Manifest does not contain exact raw actions.")
    transition_contract = recording.get("physical_transition_contract")
    if transition_contract not in {
        LEGACY_RAW_ACTION_TRANSITION_CONTRACT,
        RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT,
        DETACHED_HAND_ACTION_TRANSITION_CONTRACT,
    }:
        if (
            transition_contract
            == _OBSOLETE_EXACT_ROBOT_SUBSTEP_PHYSICAL_TRANSITION_CONTRACT
        ):
            raise CertificationError(
                "The obsolete exact robot-substep path contract is not a "
                "certifiable controller-input contract. Re-record with the "
                "raw-command plus recorded drive-target contract."
            )
        raise CertificationError("Manifest transition alignment is unsupported.")
    trajectory_input_mode = recording.get("trajectory_input_mode")
    expected_input_mode = (
        RECORDED_DRIVE_TARGET_INPUT_MODE
        if transition_contract == RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT
        else DETACHED_HAND_ACTION_INPUT_MODE
        if transition_contract == DETACHED_HAND_ACTION_TRANSITION_CONTRACT
        else LEGACY_RAW_ACTION_INPUT_MODE
    )
    if trajectory_input_mode is None and (
        transition_contract == LEGACY_RAW_ACTION_TRANSITION_CONTRACT
    ):
        # Old raw-action recordings predate the explicit input-mode field.
        # They remain loadable for diagnostic replay only.
        trajectory_input_mode = LEGACY_RAW_ACTION_INPUT_MODE
    if trajectory_input_mode != expected_input_mode:
        raise CertificationError(
            "Manifest trajectory_input_mode does not match its physical "
            f"transition contract: expected {expected_input_mode!r}, got "
            f"{trajectory_input_mode!r}."
        )
    collection_run_seed = _required_int(
        recording,
        "collection_run_seed",
        manifest_path,
    )
    episode_reset_seed = _required_int(
        recording,
        "episode_reset_seed",
        manifest_path,
    )
    fresh_per_attempt = recording.get(
        "fresh_recorder_process_per_attempt_required",
        False,
    )
    if type(fresh_per_attempt) is not bool:
        raise CertificationError(
            "Manifest fresh_recorder_process_per_attempt_required must be boolean."
        )
    fresh_scene_replay = recording.get(
        "fresh_scene_replay_required",
        fresh_per_attempt,
    )
    if type(fresh_scene_replay) is not bool:
        raise CertificationError(
            "Manifest fresh_scene_replay_required must be boolean."
        )
    recorder_lifecycle = recording.get("recorder_lifecycle")
    if recorder_lifecycle is None:
        recorder_lifecycle = "fresh_process_per_attempt" if fresh_per_attempt else ""
    if not isinstance(recorder_lifecycle, str):
        raise CertificationError("Manifest recorder_lifecycle must be text.")
    source_process_reuse_enabled = recording.get(
        "source_process_reuse_enabled",
        recorder_lifecycle == "persistent_process",
    )
    if type(source_process_reuse_enabled) is not bool:
        raise CertificationError(
            "Manifest source_process_reuse_enabled must be boolean."
        )
    source_scene_rebuild_per_reset = recording.get(
        "source_scene_rebuild_per_reset",
        False,
    )
    if type(source_scene_rebuild_per_reset) is not bool:
        raise CertificationError(
            "Manifest source_scene_rebuild_per_reset must be boolean."
        )
    source_scene_generation = recording.get("source_scene_generation", 0)
    if type(source_scene_generation) is not int or source_scene_generation < 0:
        raise CertificationError(
            "Manifest source_scene_generation must be a nonnegative integer."
        )
    certification_process_mode = recording.get("physical_replay_certification_mode")
    collection_seed_source = ""
    episode_reset_ordinal = -1
    episode_seed_derivation = ""
    source_recorder_pid = recording.get("recorder_pid")
    if source_recorder_pid is not None and (
        type(source_recorder_pid) is not int or source_recorder_pid <= 0
    ):
        raise CertificationError("Manifest recorder_pid must be positive.")
    if certification_process_mode == DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE:
        if fresh_per_attempt:
            raise CertificationError(
                "Persistent deferred recording cannot claim a fresh source "
                "process per attempt."
            )
        if recorder_lifecycle != PERSISTENT_RECORDER_LIFECYCLE:
            raise CertificationError(
                "Deferred certification requires recorder_lifecycle="
                "'persistent_process'."
            )
        if not source_process_reuse_enabled:
            raise CertificationError(
                "Persistent recording must declare source process reuse."
            )
        if not fresh_scene_replay:
            raise CertificationError(
                "Persistent recording requires fresh-scene replay."
            )
        for field in (
            "fresh_recorder_process_per_episode_required",
            "fresh_recorder_process_per_attempt_required",
        ):
            if recording.get(field) is not False:
                raise CertificationError(
                    f"Persistent recording must declare {field}=false."
                )
        for field in (
            "fresh_certifier_process_per_run_required",
            "publish_only_after_fresh_certification",
        ):
            if recording.get(field) is not True:
                raise CertificationError(
                    f"Persistent recording must declare {field}=true."
                )
        if source_recorder_pid is None:
            raise CertificationError(
                "Persistent recording is missing its source recorder_pid."
            )
        collection_seed_source = _required_text(
            recording,
            "collection_seed_source",
            manifest_path,
        )
        if collection_seed_source not in {
            "explicit_cli_seed",
            "os_cryptographic_random_31bit",
        }:
            raise CertificationError(
                "Persistent recording collection_seed_source is unsupported."
            )
        episode_reset_ordinal = _required_int(
            recording,
            "episode_reset_ordinal",
            manifest_path,
        )
        if episode_reset_ordinal < 0:
            raise CertificationError(
                "Persistent recording episode_reset_ordinal must be nonnegative."
            )
        episode_seed_derivation = _required_text(
            recording,
            "episode_seed_derivation",
            manifest_path,
        )
        expected_derivation = "collection_run_seed_plus_reset_attempt_index_mod_2pow31"
        if episode_seed_derivation != expected_derivation:
            raise CertificationError(
                "Persistent recording episode_seed_derivation is unsupported."
            )
        expected_episode_seed = (collection_run_seed + episode_reset_ordinal) % (2**31)
        if episode_reset_seed != expected_episode_seed:
            raise CertificationError(
                "Persistent recording episode_reset_seed does not match its "
                "recorded collection seed and reset ordinal."
            )
    elif certification_process_mode == FRESH_PROCESS_CERTIFICATION_MODE:
        collection_seed_source = str(recording.get("collection_seed_source") or "")
        raw_ordinal = recording.get("episode_reset_ordinal", -1)
        if type(raw_ordinal) is int:
            episode_reset_ordinal = raw_ordinal
        episode_seed_derivation = str(recording.get("episode_seed_derivation") or "")

    stable_scene_rebuilt = stable_origin.get(
        "scene_rebuilt_before_reset",
        False,
    )
    stable_scene_generation = stable_origin.get("scene_generation", 0)
    if type(stable_scene_rebuilt) is not bool or (
        type(stable_scene_generation) is not int
    ):
        raise CertificationError(
            "Stable-origin scene rebuild evidence has invalid types."
        )
    expected_scene_rebuilt = bool(
        source_scene_rebuild_per_reset and episode_reset_ordinal > 0
    )
    if stable_scene_rebuilt is not expected_scene_rebuilt or (
        stable_scene_generation != source_scene_generation
    ):
        raise CertificationError(
            "Stable-origin scene rebuild evidence differs from the manifest."
        )

    action_storage_layout = _required_text(
        recording,
        "action_storage_layout",
        manifest_path,
    )
    if action_storage_layout not in {
        "single_pose_hand",
        "right_pose_hand_then_left_pose_hand",
        "right_pose_left_pose_right_hand_left_hand",
        "flat_raw_env_action",
    }:
        raise CertificationError(
            f"Unsupported action storage layout {action_storage_layout!r}."
        )

    variables = _required_mapping(environment, "variables", manifest_path)
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in variables.items()
    ):
        raise CertificationError("Recorded environment variables are invalid.")

    environment_schema = _required_mapping(
        schema,
        "environment_schema",
        schema_path,
    )
    if (
        environment_schema.get("name") != _PHYSX_GROUND_TRUTH_SCHEMA_NAME
        or environment_schema.get("version") != _PHYSX_GROUND_TRUTH_SCHEMA_VERSION
    ):
        raise CertificationError(
            "Exact recorded t0 restore requires "
            f"{_PHYSX_GROUND_TRUTH_SCHEMA_NAME} schema version "
            f"{_PHYSX_GROUND_TRUTH_SCHEMA_VERSION}."
        )
    reset_replay_contract = _required_mapping(
        environment_schema,
        "reset_replay_contract",
        schema_path,
    )
    detached_hand_contract = bool(
        transition_contract == DETACHED_HAND_ACTION_TRANSITION_CONTRACT
    )
    for name in (
        "allow_pre_action_state_overwrite",
        "allows_single_recorded_t0_restore",
    ):
        if reset_replay_contract.get(name) is not True:
            raise CertificationError(f"Reset replay contract is missing {name}=true.")
    if reset_replay_contract.get(
        "forbids_state_writes_after_action_start"
    ) is not (not detached_hand_contract):
        raise CertificationError(
            "Reset replay state-write declaration does not match the "
            "physical transition contract."
        )
    if detached_hand_contract and reset_replay_contract.get(
        "deterministic_kinematic_hand_inputs"
    ) is not True:
        raise CertificationError(
            "Detached-hand replay requires deterministic kinematic inputs."
        )
    physics_steps_after_restore = reset_replay_contract.get(
        "physics_steps_after_exact_recorded_t0_restore"
    )
    if type(physics_steps_after_restore) is not int or (
        physics_steps_after_restore != 0
    ):
        raise CertificationError(
            "Reset replay contract must execute zero physics steps after the "
            "exact recorded t0 restore and before action 0."
        )
    exact_robot_substep_contract = (
        transition_contract == RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT
    )
    physx_schema = dict(_required_mapping(environment_schema, "physx", schema_path))
    if physx_schema.get("bitwise_checkpoint") is not False:
        raise CertificationError(
            "PhysX schema incorrectly claims a bitwise checkpoint."
        )
    if physx_schema.get("requires_fresh_process_certification") is not True:
        raise CertificationError(
            "PhysX schema is missing requires_fresh_process_certification=true."
        )
    if physx_schema.get("certification_forbids_state_writes") is not False:
        raise CertificationError(
            "PhysX schema must allow the single exact recorded t0 restore."
        )
    if physx_schema.get("allows_single_recorded_t0_restore") is not True:
        raise CertificationError(
            "PhysX schema is missing allows_single_recorded_t0_restore=true."
        )
    if physx_schema.get(
        "certification_forbids_state_writes_after_action_start"
    ) is not (not detached_hand_contract):
        raise CertificationError(
            "PhysX schema state-write declaration does not match the "
            "physical transition contract."
        )
    if detached_hand_contract and physx_schema.get(
        "deterministic_kinematic_hand_inputs"
    ) is not True:
        raise CertificationError(
            "PhysX schema is missing deterministic kinematic hand inputs."
        )
    contact_signature_path_count = _validated_contact_signature_path_count(physx_schema)

    fields = _required_mapping(schema, "fields", schema_path)
    environment_fields = _required_mapping(
        environment_schema,
        "fields",
        schema_path,
    )
    recorded_task_state_field_schema = _validated_task_state_schema(
        task_name,
        environment_schema,
        environment_fields,
        fields,
    )
    if exact_robot_substep_contract:
        required_substep_fields = _EXACT_ROBOT_SUBSTEP_FIELDS | {
            f"next_{name}" for name in _EXACT_ROBOT_SUBSTEP_FIELDS
        }
        missing_substep_fields = sorted(
            required_substep_fields - {str(name) for name in fields}
        )
        if missing_substep_fields:
            raise CertificationError(
                "The exact robot substep transition contract is missing "
                "required schema fields: " + ", ".join(missing_substep_fields)
            )
    all_current_names = sorted(
        str(name) for name in fields if str(name).startswith("physx_")
    )
    if not all_current_names:
        raise CertificationError("Recording schema contains no PhysX fields.")
    environment_physx_names = {
        str(name) for name in environment_fields if str(name).startswith("physx_")
    }
    if task_name == _P003_TASK_NAME:
        controller_entry = environment_fields.get(_TASK_CONTROLLER_STATE_FIELD)
        if not isinstance(controller_entry, Mapping):
            raise CertificationError(
                "P0-03 physical replay requires action-aligned task "
                "controller state."
            )
        if (
            controller_entry.get("shape") != [3]
            or controller_entry.get("dtype") != "uint8"
            or controller_entry.get("physical_replay") is not True
            or controller_entry.get("replay_restore_supported") is not False
        ):
            raise CertificationError(
                "P0-03 task controller state has an invalid replay schema."
            )
    missing_contact_signature_fields = sorted(
        _REQUIRED_CONTACT_SIGNATURE_FIELDS - environment_physx_names
    )
    if missing_contact_signature_fields:
        raise CertificationError(
            "Recording schema is missing canonical contact replay fields: "
            + ", ".join(missing_contact_signature_fields)
        )
    if set(all_current_names) != environment_physx_names:
        raise CertificationError(
            "Recorded PhysX field set differs from the environment schema."
        )
    if set(f"next_{name}" for name in all_current_names) != {
        str(name) for name in fields if str(name).startswith("next_physx_")
    }:
        raise CertificationError("PhysX current/next field sets are not paired.")
    current_names = []
    for name in all_current_names:
        entry = fields[name]
        next_entry = fields[f"next_{name}"]
        if not isinstance(entry, Mapping) or not isinstance(next_entry, Mapping):
            raise CertificationError(f"Invalid schema entry for {name}.")
        environment_entry = environment_fields.get(name)
        if not isinstance(environment_entry, Mapping):
            raise CertificationError(
                f"Environment schema is missing PhysX field {name}."
            )
        if dict(_field_semantics(entry)) != dict(environment_entry) or dict(
            _field_semantics(next_entry)
        ) != dict(environment_entry):
            raise CertificationError(
                f"Recorded semantics for {name} are not bound to the "
                "environment field schema."
            )
        physical_replay = environment_entry.get("physical_replay")
        if physical_replay is True:
            current_names.append(name)
        if name in (_CONTACT_SIGNATURE_FIELDS | _SUBSTEP_CONTACT_SIGNATURE_FIELDS) and (
            physical_replay is not True
            or environment_entry.get("replay_restore_supported") is not False
        ):
            raise CertificationError(
                f"Canonical contact field {name} must participate in replay "
                "comparison and must not be restored."
            )
    if not current_names:
        raise CertificationError(
            "Recording schema contains no replay-certifiable PhysX fields."
        )
    current_names.extend(recorded_task_state_field_schema)

    root = zarr.open_group(str(demo_path / "replay.zarr"), mode="r")
    if "data" not in root:
        raise CertificationError("replay.zarr is missing data.")
    data = root["data"]
    for name in ("action", "terminal", "succeed"):
        if name not in data:
            raise CertificationError(f"replay.zarr is missing {name}.")
    actions = np.asarray(data["action"])
    if actions.ndim != 2 or actions.shape[0] <= 0:
        raise CertificationError(
            f"action must have non-empty shape (T, N), got {actions.shape}."
        )
    if actions.dtype.kind not in "iuf" or not np.all(np.isfinite(actions)):
        raise CertificationError("action must contain finite numeric values.")
    steps = int(actions.shape[0])

    def load_flag(name: str) -> np.ndarray:
        values = np.asarray(data[name])
        if values.shape != (steps,) or values.dtype != np.dtype(np.uint8):
            raise CertificationError(
                f"{name} must have uint8 shape ({steps},), got "
                f"{values.dtype} {values.shape}."
            )
        if not np.all((values == 0) | (values == 1)):
            raise CertificationError(f"{name} must be binary.")
        return values

    terminals = load_flag("terminal")
    succeeds = load_flag("succeed")
    final_only = np.zeros((steps,), dtype=np.uint8)
    final_only[-1] = 1
    if not np.array_equal(terminals, final_only):
        raise CertificationError(
            "A certified demo must terminate exactly at its final action."
        )
    if not np.array_equal(succeeds, final_only):
        raise CertificationError(
            "A certified demo must succeed exactly at its final action."
        )

    field_specs = []
    current_fields = {}
    next_fields = {}
    for name in current_names:
        next_name = f"next_{name}"
        entry = fields[name]
        next_entry = fields[next_name]
        if not isinstance(entry, Mapping) or not isinstance(next_entry, Mapping):
            raise CertificationError(f"Invalid schema entry for {name}.")
        try:
            environment_entry = environment_fields[name]
            shape = tuple(int(value) for value in environment_entry["shape"])
            dtype = np.dtype(environment_entry["dtype"])
            tolerance = float(environment_entry["replay_atol"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CertificationError(
                f"Invalid physical field schema for {name}: {exc}"
            ) from exc
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise CertificationError(f"Invalid replay tolerance for {name}.")
        restore_supported = environment_entry.get(
            "replay_restore_supported",
            True,
        )
        if type(restore_supported) is not bool:
            raise CertificationError(
                f"Invalid replay_restore_supported flag for {name}."
            )
        if tuple(entry.get("shape_per_step", ())) != shape or (
            np.dtype(entry.get("dtype")) != dtype
        ):
            raise CertificationError(
                f"{name} differs from its environment field schema."
            )
        if tuple(next_entry.get("shape_per_step", ())) != shape or (
            np.dtype(next_entry.get("dtype")) != dtype
        ):
            raise CertificationError(f"{next_name} differs from {name}.")
        for storage_name in (name, next_name):
            if storage_name not in data:
                raise CertificationError(f"replay.zarr is missing {storage_name}.")
            array = data[storage_name]
            if tuple(array.shape) != (steps, *shape) or (
                np.dtype(array.dtype) != dtype
            ):
                raise CertificationError(
                    f"{storage_name} conflicts with its schema: "
                    f"{array.dtype} {array.shape}."
                )
        field_specs.append(
            PhysicalFieldSpec(
                name,
                next_name,
                shape,
                dtype,
                tolerance,
                restore_supported,
                str(environment_entry.get("replay_domain") or "physx"),
            )
        )
        current_fields[name] = data[name]
        next_fields[name] = data[next_name]
        _validate_transition_continuity(
            name,
            data[name],
            data[next_name],
            steps=steps,
            shape=shape,
            dtype=dtype,
        )

    signature_overrides, contact_signature_overlay_segment = (
        _apply_legacy_contact_signature_overlay(
            data,
            steps=steps,
            physx_schema=physx_schema,
            field_schema=environment_fields,
            segment_id=f"accepted_demo:{demo_path.name}",
        )
    )
    if signature_overrides:
        for spec in field_specs:
            if spec.name not in (
                _CONTACT_SIGNATURE_FIELDS | _SUBSTEP_CONTACT_SIGNATURE_FIELDS
            ):
                continue
            current_fields[spec.name] = signature_overrides[spec.name]
            next_fields[spec.name] = signature_overrides[spec.next_name]
            _validate_transition_continuity(
                spec.name,
                current_fields[spec.name],
                next_fields[spec.name],
                steps=steps,
                shape=spec.shape,
                dtype=spec.dtype,
            )
    overlaid_data = _ReadOnlyArrayOverlay(data, signature_overrides)
    _validate_recorded_contact_signatures(
        overlaid_data,
        steps=steps,
        path_count=contact_signature_path_count,
    )
    action_storage_layout = _resolve_action_storage_layout(
        action_storage_layout,
        actions,
        next_fields,
    )

    return LoadedDemo(
        demo_dir=demo_path,
        task_name=task_name,
        collection_run_seed=collection_run_seed,
        episode_reset_seed=episode_reset_seed,
        randomize=_required_bool(recording, "randomize", manifest_path),
        randomize_dynamics=bool(recording.get("randomize_dynamics", False)),
        arm_type=_required_text(recording, "arm_type", manifest_path),
        hand_type=_required_text(recording, "hand_type", manifest_path),
        render_mode=_required_text(recording, "render_mode", manifest_path),
        image_obs=_required_bool(recording, "image_obs", manifest_path),
        headless=_required_bool(recording, "headless", manifest_path),
        target_label=_required_text(recording, "target_label", manifest_path),
        camera_screen_effect=_required_bool(
            recording,
            "camera_screen_effect",
            manifest_path,
        ),
        ipad_screen_effect=_required_bool(
            recording,
            "ipad_screen_effect",
            manifest_path,
        ),
        environment_variables=dict(variables),
        recorded_git=recorded_git,
        recorded_runtime_identity=recorded_runtime_identity,
        recorded_physx_schema=physx_schema,
        recorded_physx_field_schema={
            str(name): copy.deepcopy(dict(value))
            for name, value in environment_fields.items()
            if str(name).startswith("physx_") and isinstance(value, Mapping)
        },
        stable_recording_origin=stable_origin,
        action_storage_layout=action_storage_layout,
        actions=actions,
        terminals=terminals,
        succeeds=succeeds,
        field_specs=tuple(field_specs),
        current_fields=current_fields,
        next_fields=next_fields,
        physical_transition_contract=str(transition_contract),
        trajectory_input_mode=str(trajectory_input_mode),
        fresh_recorder_process_per_attempt_required=bool(fresh_per_attempt),
        fresh_scene_replay_required=bool(fresh_scene_replay),
        recorder_lifecycle=str(recorder_lifecycle),
        source_process_reuse_enabled=bool(source_process_reuse_enabled),
        source_scene_rebuild_per_reset=bool(source_scene_rebuild_per_reset),
        source_scene_generation=int(source_scene_generation),
        collection_seed_source=collection_seed_source,
        episode_reset_ordinal=episode_reset_ordinal,
        episode_seed_derivation=episode_seed_derivation,
        source_recorder_pid=(
            int(source_recorder_pid) if source_recorder_pid is not None else None
        ),
        session_history_root_relative_path=(session_history_root_relative_path),
        session_history_sha256=session_history_sha256,
        session_history_event_count=session_history_event_count,
        session_history_accepted_event_ordinal=(session_history_accepted_event_ordinal),
        contact_signature_overlay_segment=(contact_signature_overlay_segment),
        recorded_task_state_field_schema=recorded_task_state_field_schema,
    )


def compare_snapshot(
    recorded: Mapping[str, Any],
    actual: Mapping[str, Any],
    field_specs: Sequence[PhysicalFieldSpec],
    *,
    phase: str,
    action_index: int | None,
    recorded_time_origin: float | None = None,
    actual_time_origin: float | None = None,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    """Compare a complete public-state snapshot using declared tolerances."""

    errors: dict[str, float] = {}
    first_divergence = None
    for spec in field_specs:
        expected = np.asarray(recorded[spec.name])
        observed = np.asarray(actual.get(spec.name))
        if expected.shape != spec.shape or observed.shape != spec.shape:
            raise CertificationError(
                f"Snapshot shape changed for {spec.name}: "
                f"recorded={expected.shape}, actual={observed.shape}."
            )
        if observed.dtype != spec.dtype:
            raise CertificationError(
                f"Snapshot dtype changed for {spec.name}: "
                f"recorded={spec.dtype}, actual={observed.dtype}."
            )
        expected_comparable = expected.astype(np.float64)
        observed_comparable = observed.astype(np.float64)
        if spec.name in {"physx_sim_time", "physx_substep_sim_time"}:
            if recorded_time_origin is None or actual_time_origin is None:
                raise CertificationError(
                    f"Relative time origins are missing for {spec.name}."
                )
            if spec.name == "physx_substep_sim_time":
                # The pre-action snapshot uses zero as an unavailable-history
                # sentinel.  Preserve it; only normalize actual timestamps.
                expected_comparable = np.where(
                    expected_comparable == 0.0,
                    0.0,
                    expected_comparable - recorded_time_origin,
                )
                observed_comparable = np.where(
                    observed_comparable == 0.0,
                    0.0,
                    observed_comparable - actual_time_origin,
                )
            else:
                expected_comparable = expected_comparable - recorded_time_origin
                observed_comparable = observed_comparable - actual_time_origin
        difference = np.abs(observed_comparable - expected_comparable)
        if not np.all(np.isfinite(difference)):
            maximum = float("inf")
            flat_index = int(np.flatnonzero(~np.isfinite(difference))[0])
        else:
            flat_index = int(np.argmax(difference))
            maximum = float(difference.reshape(-1)[flat_index])
        errors[spec.name] = maximum
        if first_divergence is None and maximum > spec.replay_atol:
            index = tuple(
                int(value) for value in np.unravel_index(flat_index, spec.shape)
            )
            first_divergence = {
                "phase": phase,
                "action_index": action_index,
                "field": spec.name,
                "index": list(index),
                "max_abs_error": maximum,
                "replay_atol": spec.replay_atol,
                "recorded": float(expected_comparable[index]),
                "actual": float(observed_comparable[index]),
            }
    return errors, first_divergence


def _initialization_field_partitions(
    field_specs: Sequence[PhysicalFieldSpec],
) -> tuple[
    tuple[PhysicalFieldSpec, ...],
    tuple[PhysicalFieldSpec, ...],
    tuple[str, ...],
]:
    """Separate writable PhysX t0 from read-only task-state comparison."""

    task_state = tuple(
        spec for spec in field_specs if spec.replay_domain == "task_state"
    )
    if any(spec.replay_restore_supported for spec in task_state):
        raise CertificationError("task_state fields must never be restored.")
    restorable = tuple(
        spec for spec in field_specs if spec.replay_restore_supported
    )
    compared = tuple(
        spec
        for spec in field_specs
        if spec.replay_restore_supported or spec.replay_domain == "task_state"
    )
    diagnostic_only = tuple(
        spec.name
        for spec in field_specs
        if not spec.replay_restore_supported and spec.replay_domain != "task_state"
    )
    return restorable, compared, diagnostic_only


def _restore_recorded_reset_snapshot(
    env: Any,
    trace: Any,
    recorded_snapshot: Mapping[str, Any],
    recorded_physx_schema: Mapping[str, Any],
    stable_recording_origin: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any] | None]:
    """Restore and read back the exact recorded boundary before action 0.

    A seeded reset reconstructs scene randomization, but it cannot guarantee
    that GPU PhysX settles contacting bodies to the exact same floating-point
    pose in a fresh process.  Writing the recorded snapshot once here is an
    initial-condition restore, not trajectory puppeteering: after this helper
    returns, certification only advances PhysX with recorded actions and never
    writes state again.

    The recording-origin gate may have put dynamic task bodies to sleep after
    its final zero-velocity write.  Reproduce that public initial condition as
    well, then rebuild every live field from simulator readback.  Substep
    history is deliberately cleared because no physics step is allowed between
    this restore and action 0.
    """

    restore_counts = trace.restore(
        recorded_snapshot,
        {"physx": dict(recorded_physx_schema)},
        sync=True,
    )
    restore_counts = _validated_restore_counts(
        restore_counts,
        label="Exact initial PhysX restore counts",
    )

    sleep_result = None
    recorded_sleep = stable_recording_origin.get("final_boundary_sleep")
    if recorded_sleep is None and stable_recording_origin.get("post_settle_sleeping"):
        recorded_sleep = stable_recording_origin.get("post_settle_sleep")
    if recorded_sleep is not None:
        sleep_result = trace.put_task_bodies_to_sleep()
        if not isinstance(sleep_result, Mapping):
            raise CertificationError(
                "Exact initial PhysX sleep restore did not return a report."
            )

    expected_restore = stable_recording_origin.get("final_boundary_state_write")
    if expected_restore is None:
        expected_restore = stable_recording_origin.get("post_settle_state_write")
    if isinstance(expected_restore, Mapping):
        for name, expected in expected_restore.items():
            if type(expected) is not int:
                continue
            actual = restore_counts.get(name)
            if type(actual) is not int or actual != expected:
                raise CertificationError(
                    "Exact initial PhysX restore count changed for "
                    f"{name}: recorded={expected}, actual={actual}."
                )
    if isinstance(recorded_sleep, Mapping):
        assert isinstance(sleep_result, Mapping)
        for name in (
            "rigid_bodies",
            "body_paths",
            "skipped_non_dynamic_or_non_path_names",
        ):
            if name not in recorded_sleep:
                continue
            expected = recorded_sleep[name]
            actual = sleep_result.get(name)
            if isinstance(expected, (list, tuple)):
                matches = tuple(str(value) for value in actual or ()) == tuple(
                    str(value) for value in expected
                )
            else:
                matches = actual == expected
            if not matches:
                raise CertificationError(
                    "Exact initial PhysX sleep state changed for "
                    f"{name}: recorded={expected!r}, actual={actual!r}."
                )

    raw_env = getattr(env, "unwrapped", env)
    refresh = getattr(raw_env, "_refresh_object_poses", None)
    if callable(refresh):
        refresh()
    compute_observation = getattr(raw_env, "_compute_observation", None)
    attach_physx = getattr(raw_env, "_attach_physx_ground_truth", None)
    if not callable(compute_observation) or not callable(attach_physx):
        raise CertificationError(
            "Environment cannot rebuild a live PhysX observation after "
            "the exact initial restore."
        )
    # Discard stabilization history before rebuilding the action-0 boundary.
    trace.begin_substep_capture()
    restored_observation = _apply_observation_wrappers(
        env,
        attach_physx(compute_observation()),
    )
    if not isinstance(restored_observation, Mapping):
        raise CertificationError(
            "Restored environment did not rebuild a mapping observation."
        )
    # Leave the trace ready for action 0 even when callers reuse this instance.
    trace.begin_substep_capture()
    return (
        dict(restored_observation),
        restore_counts,
        dict(sleep_result) if sleep_result is not None else None,
    )


def _apply_observation_wrappers(
    env: Any,
    raw_observation: Any,
) -> Any:
    """Apply every observation wrapper from the raw env outwards.

    Calling only the outermost wrapper skips intermediate transforms.  That is
    especially visible for Plan tasks, whose outer adapter consumes fields
    created by an inner plan-mode observation wrapper.
    """

    observation_wrappers: list[Any] = []
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if any(base.__name__ == "ObservationWrapper" for base in type(current).__mro__):
            observation_wrappers.append(current)
        current = getattr(current, "env", None)

    observation = raw_observation
    for wrapper in reversed(observation_wrappers):
        observation = wrapper.observation(observation)
    return observation


def _is_git_object_id(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) in {40, 64}
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _run_git(repo_root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CertificationError(
            f"Cannot verify Git source migration: git {' '.join(arguments)} failed."
        ) from exc
    return result.stdout


def _git_diff_paths(
    repo_root: Path,
    older_commit: str,
    newer_commit: str,
) -> tuple[str, ...]:
    payload = _run_git(
        repo_root,
        "diff",
        "--name-only",
        "--no-renames",
        "-z",
        older_commit,
        newer_commit,
        "--",
    )
    try:
        paths = tuple(
            item.decode("utf-8", errors="strict")
            for item in payload.split(b"\0")
            if item
        )
    except UnicodeDecodeError as exc:
        raise CertificationError(
            "Git source migration contains a non-UTF-8 path."
        ) from exc
    if len(paths) != len(set(paths)):
        raise CertificationError(
            "Git source migration contains duplicate changed paths."
        )
    return tuple(sorted(paths))


def _validate_recorded_changed_files(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise CertificationError(
            "Dirty recording lacks a non-empty software.git.changed_files."
        )
    if any(not isinstance(path, str) or not path for path in value):
        raise CertificationError(
            "Dirty recording software.git.changed_files is malformed."
        )
    paths = tuple(value)
    if paths != tuple(sorted(set(paths))):
        raise CertificationError(
            "Dirty recording software.git.changed_files is not canonical."
        )
    for path in paths:
        pure_path = PurePosixPath(path)
        if (
            pure_path.is_absolute()
            or str(pure_path) != path
            or ".." in pure_path.parts
            or " -> " in path
            or any(character in path for character in ("\0", "\n", "\r"))
        ):
            raise CertificationError(
                "Dirty recording contains a path that cannot be migrated "
                f"safely: {path!r}."
            )
    return paths


def _snapshot_worktree_content_sha256(
    repo_root: Path,
    snapshot_commit: str,
    changed_files: tuple[str, ...],
) -> str:
    """Recreate the historical changed-path content digest from a Git tree."""

    content_identity: list[dict[str, str | None]] = []
    for relative_path in changed_files:
        entry = _run_git(
            repo_root,
            "ls-tree",
            "-z",
            snapshot_commit,
            "--",
            f":(literal){relative_path}",
        )
        if not entry:
            kind = "deleted"
            content_sha256 = None
        else:
            rows = [row for row in entry.split(b"\0") if row]
            if len(rows) != 1 or b"\t" not in rows[0]:
                raise CertificationError(
                    f"Snapshot tree returned an ambiguous entry for {relative_path!r}."
                )
            metadata, encoded_path = rows[0].split(b"\t", 1)
            try:
                listed_path = encoded_path.decode("utf-8", errors="strict")
                mode, object_type, object_id = metadata.decode("ascii").split()
            except (UnicodeDecodeError, ValueError) as exc:
                raise CertificationError(
                    "Snapshot tree contains an unsupported Git entry."
                ) from exc
            if listed_path != relative_path or object_type != "blob":
                raise CertificationError(
                    "Snapshot tree contains an unsupported entry for "
                    f"{relative_path!r}."
                )
            if mode == "120000":
                kind = "symlink"
            elif mode in {"100644", "100755"}:
                kind = "file"
            else:
                raise CertificationError(
                    "Snapshot tree contains an unsupported mode for "
                    f"{relative_path!r}: {mode}."
                )
            payload = _run_git(repo_root, "cat-file", "blob", object_id)
            content_sha256 = hashlib.sha256(payload).hexdigest()
        content_identity.append(
            {
                "path": relative_path,
                "kind": kind,
                "sha256": content_sha256,
            }
        )
    serialized = json.dumps(
        content_identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _verify_dirty_snapshot_reviewer_upgrade(
    recorded: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    repo_root: Path,
) -> SourceIdentityVerification:
    recorded_commit = str(recorded["commit"])
    current_commit = current.get("commit")
    if not _is_git_object_id(current_commit):
        raise CertificationError("Current Git HEAD is unavailable or malformed.")
    current_commit = str(current_commit)
    live_commit = (
        _run_git(repo_root, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    )
    if live_commit != current_commit:
        raise CertificationError(
            "Current Git HEAD changed while source identity was verified."
        )
    if _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise CertificationError(
            "Dirty-recording migration requires a completely clean worktree."
        )

    try:
        ancestor_check = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "merge-base",
                "--is-ancestor",
                recorded_commit,
                current_commit,
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CertificationError(
            "Cannot verify that the recorded commit is an ancestor of HEAD."
        ) from exc
    if ancestor_check.returncode != 0:
        raise CertificationError(
            "Recorded Git commit is not an ancestor of current HEAD."
        )

    recorded_paths = _validate_recorded_changed_files(recorded.get("changed_files"))
    recorded_content_sha256 = recorded.get("worktree_content_sha256")
    if not (
        isinstance(recorded_content_sha256, str)
        and len(recorded_content_sha256) == 64
        and recorded_content_sha256 == recorded_content_sha256.lower()
        and all(
            character in "0123456789abcdef" for character in recorded_content_sha256
        )
    ):
        raise CertificationError(
            "Dirty recording lacks a valid software.git.worktree_content_sha256."
        )

    ancestry = (
        _run_git(
            repo_root,
            "rev-list",
            "--reverse",
            "--ancestry-path",
            f"{recorded_commit}..{current_commit}",
        )
        .decode("ascii")
        .splitlines()
    )
    snapshot_commit: str | None = None
    for candidate in ancestry:
        if _git_diff_paths(repo_root, recorded_commit, candidate) != recorded_paths:
            continue
        if (
            _snapshot_worktree_content_sha256(
                repo_root,
                candidate,
                recorded_paths,
            )
            == recorded_content_sha256
        ):
            snapshot_commit = candidate
            break
    if snapshot_commit is None:
        raise CertificationError(
            "No ancestor commit exactly snapshots the recorded dirty path set "
            "and worktree content identity."
        )

    reviewer_paths = _git_diff_paths(
        repo_root,
        snapshot_commit,
        current_commit,
    )
    if not reviewer_paths:
        raise CertificationError(
            "Exact dirty snapshot has no committed reviewer upgrade."
        )
    unexpected_paths = sorted(
        set(reviewer_paths) - SOURCE_REVIEWER_UPGRADE_ALLOWED_PATHS
    )
    if unexpected_paths:
        raise CertificationError(
            "Reviewer upgrade changes paths outside the certification/publication "
            f"allowlist: {unexpected_paths}."
        )
    final_live_commit = (
        _run_git(repo_root, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    )
    if final_live_commit != current_commit:
        raise CertificationError(
            "Current Git HEAD changed while source identity was verified."
        )
    if _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise CertificationError("Worktree changed while source identity was verified.")
    return SourceIdentityVerification(
        source_identity=DIRTY_SNAPSHOT_REVIEWER_UPGRADE_SOURCE_IDENTITY,
        portable=True,
        source_snapshot_commit=snapshot_commit,
        reviewer_upgrade_paths=reviewer_paths,
    )


def _verify_clean_commit_reviewer_upgrade(
    recorded: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    repo_root: Path,
) -> SourceIdentityVerification:
    """Allow committed verifier-only fixes after a clean recording."""

    recorded_commit = str(recorded["commit"])
    current_commit = current.get("commit")
    if not _is_git_object_id(current_commit):
        raise CertificationError("Current Git HEAD is unavailable or malformed.")
    current_commit = str(current_commit)
    live_commit = (
        _run_git(repo_root, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    )
    if live_commit != current_commit:
        raise CertificationError(
            "Current Git HEAD changed while source identity was verified."
        )
    if _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise CertificationError(
            "Clean-recording reviewer upgrade requires a completely clean worktree."
        )
    try:
        ancestor_check = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "merge-base",
                "--is-ancestor",
                recorded_commit,
                current_commit,
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CertificationError(
            "Cannot verify that the recorded commit is an ancestor of HEAD."
        ) from exc
    if ancestor_check.returncode != 0:
        raise CertificationError(
            "Recorded Git commit is not an ancestor of current HEAD."
        )
    reviewer_paths = _git_diff_paths(
        repo_root,
        recorded_commit,
        current_commit,
    )
    if not reviewer_paths:
        raise CertificationError("Clean recording has no committed reviewer upgrade.")
    unexpected_paths = sorted(
        set(reviewer_paths) - SOURCE_REVIEWER_UPGRADE_ALLOWED_PATHS
    )
    if unexpected_paths:
        raise CertificationError(
            "Reviewer upgrade changes paths outside the certification/publication "
            f"allowlist: {unexpected_paths}."
        )
    return SourceIdentityVerification(
        source_identity=CLEAN_COMMIT_REVIEWER_UPGRADE_SOURCE_IDENTITY,
        portable=True,
        source_snapshot_commit=recorded_commit,
        reviewer_upgrade_paths=reviewer_paths,
    )


def verify_source_identity(
    recorded: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> SourceIdentityVerification:
    recorded_commit = recorded.get("commit")
    if not _is_git_object_id(recorded_commit):
        raise CertificationError("Manifest lacks a valid software.git.commit.")
    recorded_dirty = recorded.get("dirty")
    if not isinstance(recorded_dirty, bool):
        raise CertificationError("Manifest lacks software.git.dirty.")
    current_dirty = current.get("dirty")
    if not isinstance(current_dirty, bool):
        raise CertificationError("Current Git dirty state is unavailable.")

    if not recorded_dirty:
        if current_dirty:
            raise CertificationError(
                "Recorded source was clean but the current worktree is dirty."
            )
        if current.get("commit") == recorded_commit:
            return SourceIdentityVerification(
                CLEAN_COMMIT_SOURCE_IDENTITY,
                True,
            )
        if repo_root is None:
            raise CertificationError(
                f"Git HEAD mismatch: current={current.get('commit')} "
                f"recorded={recorded_commit}."
            )
        return _verify_clean_commit_reviewer_upgrade(
            recorded,
            current,
            repo_root=repo_root,
        )

    if current_dirty:
        if current.get("commit") != recorded_commit:
            raise CertificationError(
                f"Dirty Git HEAD mismatch: current={current.get('commit')} "
                f"recorded={recorded_commit}."
            )
        for key in ("changed_files", "worktree_content_sha256"):
            if current.get(key) != recorded.get(key):
                raise CertificationError(
                    f"Dirty source worktree mismatch in software.git.{key}."
                )
        return SourceIdentityVerification(
            DIRTY_CONTENT_SOURCE_IDENTITY,
            False,
        )

    if repo_root is None:
        raise CertificationError(
            "Dirty-recording migration requires a repository root."
        )
    return _verify_dirty_snapshot_reviewer_upgrade(
        recorded,
        current,
        repo_root=repo_root,
    )


@contextlib.contextmanager
def _recorded_environment(variables: Mapping[str, str]):
    managed = {
        name
        for name in os.environ
        if name.startswith("HANDSON_") or name in _EXACT_ENVIRONMENT_NAMES
    } | set(variables)
    previous = {name: os.environ.get(name) for name in managed}
    try:
        for name in managed:
            os.environ.pop(name, None)
        os.environ.update(
            {
                name: value
                for name, value in variables.items()
                if name != BC011_COLLECTION_ANGLE_GUIDANCE_ENV
            }
        )
        os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
        yield
    finally:
        for name in managed | {"HANDSON_RECORD_PHYSX_GROUND_TRUTH"}:
            os.environ.pop(name, None)
        for name, value in previous.items():
            if value is not None:
                os.environ[name] = value


def _construct_environment(config: Any, demo: LoadedDemo):
    if demo.randomize_dynamics:
        raise CertificationError(
            "Certified PhysX replay does not support randomized physical "
            "parameters; expected recording.randomize_dynamics=false."
        )
    kwargs: dict[str, Any] = {
        "policy_mode": True,
        "sim_backend": "isaacsim",
        "randomize": demo.randomize,
        "randomize_dynamics": False,
        "seed": demo.collection_run_seed,
        "arm_type": demo.arm_type,
        "hand_type": demo.hand_type,
        "render_mode": demo.render_mode,
        "image_obs": demo.image_obs,
        "headless": demo.headless,
    }
    if demo.task_name in {"bimanual_photograph", "t23_photograph_logo"}:
        kwargs["camera_screen_effect"] = demo.camera_screen_effect
    if demo.task_name == "bimanual_unlock_ipad":
        kwargs["ipad_screen_effect"] = demo.ipad_screen_effect
    if demo.task_name == _P001_TASK_NAME:
        try:
            kwargs["target_open_deg"] = _P001_TARGET_OPEN_DEG_BY_LABEL[
                demo.target_label
            ]
        except KeyError as exc:
            allowed = ", ".join(_P001_TARGET_OPEN_DEG_BY_LABEL)
            raise CertificationError(
                "P0-01 replay requires an exact recorded target_label; "
                f"expected one of {allowed}, got {demo.target_label!r}."
            ) from exc
    env = config.get_native_environment(**kwargs)
    raw_env = getattr(env, "unwrapped", env)
    # Build Kit, the USD stage, cameras, and every PhysX object from the exact
    # source display configuration above.  Only after construction do we stop
    # producing image observations.  IsaacLab's task environments gate their
    # per-control-step render() call on ``image_obs``, so this removes RTX work
    # without changing scene construction or the PhysX initialization history.
    if not hasattr(raw_env, "image_obs"):
        raise CertificationError(
            "Physics-only replay requires an environment image_obs switch."
        )
    raw_env.image_obs = False
    if bool(getattr(raw_env, "image_obs", True)):
        raise CertificationError(
            "Physics-only replay could not disable per-step image rendering."
        )
    if not hasattr(raw_env, "hz"):
        raise CertificationError(
            "Physics-only replay requires an environment wall-clock hz switch."
        )
    raw_env.hz = 0
    if int(getattr(raw_env, "hz", -1)) != 0:
        raise CertificationError(
            "Physics-only replay could not disable wall-clock throttling."
        )
    window_state = _hide_physics_only_app_window(source_headless=bool(demo.headless))
    setattr(
        raw_env,
        _PHYSICS_ONLY_APP_WINDOW_STATE_ATTRIBUTE,
        dict(window_state),
    )
    return env


def _hide_physics_only_app_window(
    *,
    source_headless: bool,
    appwindow_module: Any | None = None,
    windowing_module: Any | None = None,
) -> dict[str, Any]:
    """Hide the already-constructed native Kit window during fast replay.

    The environment must first be built from the recorded ``headless`` and
    render settings so USD, camera, renderer, and PhysX construction remain
    identical to the source process.  Hiding the native OS window afterwards
    changes neither the stage nor simulation configuration; it only prevents
    the intentionally non-rendering certification loop from looking like an
    unresponsive interactive Isaac session.

    ``appwindow_module`` and ``windowing_module`` are dependency-injection
    seams for the non-Isaac unit test process.  Production callers always use
    Kit's loaded ``omni.appwindow`` and ``carb.windowing`` modules.
    """

    if source_headless:
        return {
            "policy": "hide_after_source_scene_construction",
            "status": "source_headless_no_window",
            "visible": False,
            "hide_call_performed": False,
        }

    if appwindow_module is None or windowing_module is None:
        try:
            import carb.windowing as loaded_windowing
            import omni.appwindow as loaded_appwindow
        except Exception as exc:
            # Window visibility is a presentation concern.  It must never
            # invalidate or leak an already-created source-matched physics
            # environment, so preserve certification and expose the failure
            # in its report instead of raising after construction.
            return {
                "policy": "hide_after_source_scene_construction",
                "status": "window_api_unavailable",
                "visible": None,
                "hide_call_performed": False,
                "error_type": type(exc).__name__,
            }
        appwindow_module = loaded_appwindow
        windowing_module = loaded_windowing

    native_window = None
    windowing = None
    try:
        app_window = appwindow_module.get_default_app_window()
        native_window = app_window.get_window() if app_window is not None else None
        if native_window is None:
            return {
                "policy": "hide_after_source_scene_construction",
                "status": "native_window_already_absent",
                "visible": False,
                "hide_call_performed": False,
            }
        windowing = windowing_module.acquire_windowing_interface()
        if windowing is None:
            raise RuntimeError("carb.windowing interface is unavailable")
        windowing.hide_window(native_window)
        # Flush only the native window request.  This does not update the Kit
        # timeline or advance PhysX and occurs before session-history replay.
        windowing.poll_events()
    except Exception as exc:
        # Minimize is a conservative fallback for older Kit builds whose
        # windowing interface exposes a native handle but rejects hide().
        try:
            if native_window is None or windowing is None:
                raise RuntimeError("native window interface is unavailable")
            windowing.minimize_window(native_window)
            windowing.poll_events()
        except Exception as fallback_exc:
            return {
                "policy": "hide_after_source_scene_construction",
                "status": "hide_and_minimize_failed",
                "visible": None,
                "hide_call_performed": False,
                "error_type": type(exc).__name__,
                "fallback_error_type": type(fallback_exc).__name__,
            }
        return {
            "policy": "hide_after_source_scene_construction",
            "status": "minimized_fallback",
            "visible": False,
            "hide_call_performed": False,
            "minimize_call_performed": True,
            "hide_error_type": type(exc).__name__,
        }

    return {
        "policy": "hide_after_source_scene_construction",
        "status": "hidden",
        "visible": False,
        "hide_call_performed": True,
    }


def _physics_only_app_window_state(
    env: Any,
    demo: LoadedDemo,
) -> dict[str, Any]:
    raw_env = getattr(env, "unwrapped", env)
    state = getattr(
        raw_env,
        _PHYSICS_ONLY_APP_WINDOW_STATE_ATTRIBUTE,
        None,
    )
    if isinstance(state, Mapping):
        return dict(state)
    return {
        "policy": "hide_after_source_scene_construction",
        "status": (
            "source_headless_no_window" if demo.headless else "state_unavailable"
        ),
        "visible": False if demo.headless else None,
        "hide_call_performed": False,
    }


def _source_display_config(demo: LoadedDemo) -> dict[str, Any]:
    return {
        "render_mode": demo.render_mode,
        "image_obs": bool(demo.image_obs),
        "headless": bool(demo.headless),
    }


def _effective_display_config(demo: LoadedDemo) -> dict[str, Any]:
    # render_mode/headless describe the already-constructed source scene and
    # AppLauncher.  Only image production changes for accelerated execution.
    return {
        "render_mode": demo.render_mode,
        "image_obs": False,
        "headless": bool(demo.headless),
    }


def _decode_recorded_action(
    action: np.ndarray,
    *,
    storage_layout: str,
) -> Any:
    """Return the exact raw-environment action represented on disk."""

    action_flat = np.asarray(action, dtype=np.float64)
    if storage_layout == "right_pose_hand_then_left_pose_hand":
        if action_flat.ndim != 1 or action_flat.shape[0] not in (46, 54):
            raise CertificationError(
                f"Bimanual action has invalid shape {action_flat.shape}."
            )
        side_dim = action_flat.shape[0] // 2
        raw_action: Any = {
            "right": action_flat[:side_dim].copy(),
            "left": action_flat[side_dim:].copy(),
        }
    else:
        raw_action = action_flat.copy()
    return raw_action


def _resolve_action_storage_layout(
    declared_layout: str,
    actions: np.ndarray,
    next_fields: Mapping[str, Any],
) -> str:
    """Repair legacy flat bimanual recordings mislabeled from their shape.

    Older recorder builds inferred every 46/54-value array to be two
    concatenated per-side actions.  Isaac's native flat action is instead
    ``right_pose, left_pose, right_hand, left_hand``.  The recorded post-action
    command fields make the distinction exact without task-specific guesses.
    """

    grouped_layout = "right_pose_left_pose_right_hand_left_hand"
    if declared_layout != "right_pose_hand_then_left_pose_hand":
        return declared_layout
    action_rows = np.asarray(actions, dtype=np.float64)
    if (
        action_rows.ndim != 2
        or action_rows.shape[1] not in (46, 54)
        or "physx_command_tcp_pose" not in next_fields
        or "physx_command_hand" not in next_fields
    ):
        return declared_layout

    row_count = min(8, action_rows.shape[0])
    hand_dim = (action_rows.shape[1] - 14) // 2
    sample = action_rows[:row_count]
    recorded_tcp = np.asarray(
        next_fields["physx_command_tcp_pose"][:row_count],
        dtype=np.float64,
    )
    recorded_hand = np.asarray(
        next_fields["physx_command_hand"][:row_count],
        dtype=np.float64,
    )
    if recorded_tcp.shape != (row_count, 2, 7) or recorded_hand.shape != (
        row_count,
        2,
        hand_dim,
    ):
        return declared_layout

    side_dim = action_rows.shape[1] // 2
    declared_tcp = np.stack(
        [sample[:, :7], sample[:, side_dim : side_dim + 7]],
        axis=1,
    )
    declared_hand = np.stack(
        [sample[:, 7:side_dim], sample[:, side_dim + 7 :]],
        axis=1,
    )
    grouped_tcp = np.stack([sample[:, :7], sample[:, 7:14]], axis=1)
    grouped_hand = np.stack(
        [
            sample[:, 14 : 14 + hand_dim],
            sample[:, 14 + hand_dim : 14 + 2 * hand_dim],
        ],
        axis=1,
    )

    declared_error = max(
        float(np.max(np.abs(declared_tcp - recorded_tcp))),
        float(np.max(np.abs(declared_hand - recorded_hand))),
    )
    grouped_error = max(
        float(np.max(np.abs(grouped_tcp - recorded_tcp))),
        float(np.max(np.abs(grouped_hand - recorded_hand))),
    )
    if grouped_error <= 1.0e-9 and declared_error > 1.0e-6:
        return grouped_layout
    return declared_layout


def _install_recorded_raw_command_without_ik(
    env: Any,
    action: np.ndarray,
    *,
    storage_layout: str,
) -> None:
    """Install TCP/hand command memory without computing a joint target.

    The source process already ran IK once and recorded the resulting drive
    target at every physics substep.  Certification must preserve raw command
    and task-postprocessing state, but must not run IK a second time.
    """

    raw_env = getattr(env, "unwrapped", env)
    apply_action = getattr(raw_env, "_apply_action", None)
    if not callable(apply_action):
        raise CertificationError(
            "Environment cannot install a recorded raw command without "
            "executing online IK."
        )
    if bool(getattr(raw_env, "_manual_reset_requested", False)):
        raise CertificationError(
            "Manual reset was pending while installing a recorded command."
        )
    apply_action(
        _decode_recorded_action(
            action,
            storage_layout=storage_layout,
        )
    )


def _install_recorded_task_controller_state(
    env: Any,
    recorded_next: Mapping[str, Any],
) -> bool:
    """Install task-side state that governed the recorded action.

    The recorder samples ``next_*`` after action[t], while P0-03 publishes its
    phase immediately before executing action[t].  Consequently the matching
    controller state is the value stored in ``next_physx_task_controller_state``
    for that action, and it must be installed before the raw command.
    """

    if _TASK_CONTROLLER_STATE_FIELD not in recorded_next:
        return False
    raw_env = getattr(env, "unwrapped", env)
    setter = getattr(
        raw_env,
        "set_physical_replay_task_controller_state",
        None,
    )
    if not callable(setter):
        raise CertificationError(
            "Recording contains task controller state but the runtime cannot "
            "install it."
        )
    value = np.asarray(recorded_next[_TASK_CONTROLLER_STATE_FIELD])
    if value.shape != (3,) or value.dtype != np.dtype(np.uint8):
        raise CertificationError(
            "Recorded task controller state must have uint8 shape (3,), got "
            f"{value.dtype} {value.shape}."
        )
    try:
        setter(value.copy())
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CertificationError(
            f"Could not install recorded task controller state: {exc}"
        ) from exc
    return True


def _step_with_recorded_action(
    env: Any,
    action: np.ndarray,
    *,
    storage_layout: str,
):
    """Legacy diagnostic path that recomputes online controller/IK output."""

    raw_env = getattr(env, "unwrapped", env)
    raw_action = _decode_recorded_action(
        action,
        storage_layout=storage_layout,
    )
    return raw_env.step(raw_action)


def run_certification(
    demo: LoadedDemo,
    *,
    on_environment_created: Callable[[Any], None] | None = None,
    environment: Any | None = None,
    platform_review: bool = False,
) -> dict[str, Any]:
    """Rebuild the stable origin, then physically execute recorded inputs.

    ``certify_demo`` owns the environment lifecycle through
    ``on_environment_created`` so it can durably write the certification report
    before Isaac/Kit shutdown.  A session certifier may instead supply one
    already-created ``environment`` for ordered replay of multiple episodes;
    that shared environment is never closed here.  Direct callers retain the
    historical behavior and have their newly-created environment closed before
    this function returns.
    """

    if environment is not None and on_environment_created is not None:
        raise ValueError(
            "run_certification accepts either environment or "
            "on_environment_created, not both."
        )

    from handson.sim.isaaclab.articulated_replay import (
        RecordedArticulatedDrivePath,
        run_articulated_drive_control_step,
        validate_recorded_physics_configuration,
    )
    from handson.sim.isaaclab.physx_ground_truth import PhysxGroundTruthTrace
    from handson.tasks.mappings import CONFIG_MAPPING

    if demo.task_name not in CONFIG_MAPPING:
        raise CertificationError(f"Unknown task {demo.task_name!r}.")
    current_git: dict[str, Any] = {}
    source_verification: SourceIdentityVerification | None = None
    if platform_review:
        source_identity = PLATFORM_REVIEW_SOURCE_IDENTITY
        portable = False
    else:
        current_git = capture_git_replay_state(
            _REPO_ROOT,
            exclude_path=demo.demo_dir,
        )
        source_verification = verify_source_identity(
            demo.recorded_git,
            current_git,
            repo_root=_REPO_ROOT,
        )
        source_identity = source_verification.source_identity
        portable = source_verification.portable
    current_runtime_identity = capture_runtime_replay_identity()
    if (
        not platform_review
        and current_runtime_identity != demo.recorded_runtime_identity
    ):
        raise CertificationError(
            "Replay host/runtime identity differs from the recording process."
        )
    max_errors = {spec.name: 0.0 for spec in demo.field_specs}
    (
        initialization_field_specs,
        initialization_comparison_field_specs,
        initialization_diagnostic_only_fields,
    ) = _initialization_field_partitions(demo.field_specs)
    if not initialization_field_specs:
        raise CertificationError(
            "Recording schema contains no restorable PhysX fields for the "
            "action-0 initial-state readback gate."
        )
    first_divergence = None
    first_outcome_mismatch = None
    steps_executed = 0
    final_succeed = False
    raw_command_install_count = 0
    task_controller_state_install_count = 0
    recorded_drive_target_substeps_expected = 0
    recorded_drive_target_substeps_executed = 0
    recorded_drive_target_writes_expected = 0
    recorded_drive_target_writes_executed = 0
    recorded_drive_target_write_count_per_substep = 0
    max_intra_step_target_delta = 0.0
    recorded_physics_configuration_match: dict[str, Any] | None = None
    recorded_drive_path: Any | None = None
    recorded_restore_schema = {"physx": demo.recorded_physx_schema}
    uses_recorded_drive_targets = (
        demo.physical_transition_contract == RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT
    )
    uses_detached_hand_inputs = bool(
        demo.physical_transition_contract
        == DETACHED_HAND_ACTION_TRANSITION_CONTRACT
    )
    uses_fresh_scene_origin = bool(
        demo.fresh_scene_replay_required
        or demo.fresh_recorder_process_per_attempt_required
    )
    legacy_fresh_e01 = bool(
        demo.physical_transition_contract == LEGACY_RAW_ACTION_TRANSITION_CONTRACT
        and demo.task_name == "e01_fold_garment"
        and uses_fresh_scene_origin
    )
    certifiable_input_contract = bool(
        (
            (uses_recorded_drive_targets and uses_fresh_scene_origin)
            or (uses_detached_hand_inputs and uses_fresh_scene_origin)
            or legacy_fresh_e01
        )
        and not _validated_velocity_clamp_contract(demo.stable_recording_origin)[0]
    )
    replayed_stable_origin_history: dict[str, Any] | None = None
    env = environment
    close_environment_here = environment is None and on_environment_created is None

    with _recorded_environment(demo.environment_variables):
        try:
            if env is None:
                env = _construct_environment(
                    CONFIG_MAPPING[demo.task_name](),
                    demo,
                )
                if on_environment_created is not None:
                    on_environment_created(env)
            reset_kwargs: dict[str, Any] = {"seed": demo.episode_reset_seed}
            if demo.target_label:
                reset_kwargs["target_label"] = demo.target_label
            reset_observation, reset_info = env.reset(**reset_kwargs)
            reset_observation, reset_info = stabilize_recording_origin(
                env,
                reset_observation,
                reset_info,
                **stable_recording_origin_kwargs_from_report(
                    demo.stable_recording_origin
                ),
            )
            if uses_fresh_scene_origin:
                replayed_stable_origin_history = (
                    _validated_replayed_stable_origin_history(
                        demo.stable_recording_origin,
                        reset_info,
                    )
                )
            raw_env = getattr(env, "unwrapped", env)
            trace = getattr(raw_env, "_physx_ground_truth_trace", None)
            if trace is None:
                trace = PhysxGroundTruthTrace(raw_env)
                # The environment step path must sample the same trace whose
                # exact action-0 state was restored and whose substep history
                # was cleared below.
                setattr(raw_env, "_physx_ground_truth_trace", trace)
            runtime_schema = trace.schema()
            _validate_runtime_trace_schema(demo, runtime_schema)
            _validate_runtime_task_state_schema(demo, env)
            if uses_recorded_drive_targets:
                try:
                    recorded_drive_path = RecordedArticulatedDrivePath(
                        recorded_restore_schema
                    )
                except (RuntimeError, TypeError, ValueError) as exc:
                    raise CertificationError(
                        f"Recorded drive-target replay contract is invalid: {exc}"
                    ) from exc
            if uses_recorded_drive_targets or uses_detached_hand_inputs:
                try:
                    physics_configuration_match = (
                        validate_recorded_physics_configuration(
                            recorded_restore_schema,
                            runtime_schema,
                        )
                    )
                    recorded_physics_configuration_match = (
                        _complete_physics_configuration_match_evidence(
                            physics_configuration_match,
                            recorded_restore_schema,
                            runtime_schema,
                        )
                    )
                except (RuntimeError, TypeError, ValueError) as exc:
                    raise CertificationError(
                        "Recorded physical input contract has incompatible "
                        f"physics configuration: {exc}"
                    ) from exc

            reset_recorded = {
                spec.name: demo.current_fields[spec.name][0]
                for spec in demo.field_specs
            }
            reset_restore_recorded = {
                spec.name: reset_recorded[spec.name]
                for spec in initialization_field_specs
            }
            if uses_fresh_scene_origin:
                if not isinstance(reset_observation, Mapping):
                    raise CertificationError(
                        "Stable fresh-scene initialization did not return a "
                        "mapping observation."
                    )
                # The source and certifier both stop at the boundary returned
                # by stabilize_recording_origin.  Restoring recorded t0 here
                # would create a different PhysX call/cache history and defeat
                # fresh-scene determinism.
                reset_observation = dict(reset_observation)
                actual_stable_origin = reset_info.get("physical_recording_origin")
                assert isinstance(actual_stable_origin, Mapping)
                initialization_restore_counts = _validated_restore_counts(
                    actual_stable_origin.get("final_boundary_state_write"),
                    label=("replayed stable_origin.final_boundary_state_write"),
                )
                replayed_sleep = actual_stable_origin.get("final_boundary_sleep")
                initialization_sleep = (
                    dict(replayed_sleep)
                    if isinstance(replayed_sleep, Mapping)
                    else None
                )
                # The returned observation is the t0 readback.  Discard any
                # capture buffer only after retaining it for comparison, then
                # leave the trace empty and ready for action 0.
                trace.begin_substep_capture()
            else:
                (
                    reset_observation,
                    initialization_restore_counts,
                    initialization_sleep,
                ) = _restore_recorded_reset_snapshot(
                    env,
                    trace,
                    reset_restore_recorded,
                    demo.recorded_physx_schema,
                    demo.stable_recording_origin,
                )
            recorded_time_origin = actual_time_origin = None
            if any(
                spec.name in {"physx_sim_time", "physx_substep_sim_time"}
                for spec in demo.field_specs
            ):
                try:
                    recorded_time_origin = float(
                        np.asarray(reset_recorded["physx_sim_time"]).reshape(-1)[0]
                    )
                    actual_time_origin = float(
                        np.asarray(reset_observation["physx_sim_time"]).reshape(-1)[0]
                    )
                except (KeyError, TypeError, ValueError, IndexError) as exc:
                    raise CertificationError(
                        "Physical replay requires a scalar physx_sim_time origin."
                    ) from exc
            errors, divergence = compare_snapshot(
                reset_recorded,
                reset_observation,
                initialization_comparison_field_specs,
                phase="reset",
                action_index=None,
                recorded_time_origin=recorded_time_origin,
                actual_time_origin=actual_time_origin,
            )
            for name, value in errors.items():
                max_errors[name] = max(max_errors[name], value)
            first_divergence = divergence
            initialization_errors = dict(errors)
            initialization_first_divergence = divergence

            final_info: Mapping[str, Any] = {}
            # A mismatched restored origin is not the recorded experiment.
            # Fail before action 0 instead of wasting time executing a known-
            # invalid trajectory or accidentally reporting its outcome.
            if initialization_first_divergence is None:
                for action_index, action in enumerate(demo.actions):
                    recorded_next = {
                        spec.name: demo.next_fields[spec.name][action_index]
                        for spec in demo.field_specs
                    }
                    if uses_recorded_drive_targets:
                        assert recorded_drive_path is not None
                        if _install_recorded_task_controller_state(
                            env,
                            recorded_next,
                        ):
                            task_controller_state_install_count += 1
                        _install_recorded_raw_command_without_ik(
                            env,
                            action,
                            storage_layout=demo.action_storage_layout,
                        )
                        raw_command_install_count += 1
                        try:
                            drive_step = recorded_drive_path.control_step(recorded_next)
                            recorded_drive_target_substeps_expected += int(
                                drive_step.substep_count
                            )
                            recorded_drive_target_writes_expected += int(
                                drive_step.substep_count
                                * drive_step.expected_joint_count
                            )
                            result, drive_report = run_articulated_drive_control_step(
                                raw_env,
                                trace,
                                recorded_restore_schema,
                                drive_step,
                            )
                        except (RuntimeError, TypeError, ValueError) as exc:
                            raise CertificationError(
                                "Recorded drive-target replay failed at "
                                f"action {action_index}: {exc}"
                            ) from exc
                        recorded_drive_target_substeps_executed += int(
                            drive_report["recorded_substep_count"]
                        )
                        recorded_drive_target_writes_executed += int(
                            drive_report["applied_robot_joint_target_count"]
                        )
                        per_substep_count = int(
                            drive_report["applied_robot_joint_target_count_per_substep"]
                        )
                        if recorded_drive_target_write_count_per_substep not in (
                            0,
                            per_substep_count,
                        ):
                            raise CertificationError(
                                "Recorded drive-target joint coverage changed "
                                "between control steps."
                            )
                        recorded_drive_target_write_count_per_substep = (
                            per_substep_count
                        )
                        max_intra_step_target_delta = max(
                            max_intra_step_target_delta,
                            float(drive_report["max_intra_step_target_delta"]),
                        )
                    else:
                        result = _step_with_recorded_action(
                            env,
                            action,
                            storage_layout=demo.action_storage_layout,
                        )
                    if not isinstance(result, tuple) or len(result) != 5:
                        raise CertificationError(
                            "Physical control step must return five values."
                        )
                    final_info = result[4]
                    if not isinstance(final_info, Mapping):
                        raise CertificationError("env.step info must be a mapping.")
                    steps_executed += 1
                    errors, divergence = compare_snapshot(
                        recorded_next,
                        result[0],
                        demo.field_specs,
                        phase="post_action",
                        action_index=action_index,
                        recorded_time_origin=recorded_time_origin,
                        actual_time_origin=actual_time_origin,
                    )
                    for name, value in errors.items():
                        max_errors[name] = max(max_errors[name], value)
                    if first_divergence is None:
                        first_divergence = divergence

                    actual_terminal = bool(result[2] or result[3])
                    actual_succeed = bool(final_info.get("succeed", False))
                    expected_terminal = bool(demo.terminals[action_index])
                    expected_succeed = bool(demo.succeeds[action_index])
                    if first_outcome_mismatch is None and (
                        actual_terminal != expected_terminal
                        or actual_succeed != expected_succeed
                    ):
                        first_outcome_mismatch = {
                            "action_index": action_index,
                            "expected_terminal": expected_terminal,
                            "actual_terminal": actual_terminal,
                            "expected_succeed": expected_succeed,
                            "actual_succeed": actual_succeed,
                        }
                    if actual_terminal and not expected_terminal:
                        break
            final_succeed = bool(final_info.get("succeed", False))
        finally:
            if close_environment_here and env is not None:
                env.close()

    all_actions_executed = steps_executed == len(demo.actions)
    diagnostic_trace_match = first_divergence is None
    diagnostic_transition_outcomes_match = first_outcome_mismatch is None
    task_success_replay = all_actions_executed and final_succeed
    trace_match = diagnostic_trace_match
    transition_outcomes_match = diagnostic_transition_outcomes_match
    passed = (
        (certifiable_input_contract or platform_review)
        and task_success_replay
        and trace_match
        and transition_outcomes_match
    )
    if passed:
        reason = "continuous_physx_replay_matched"
    elif first_divergence is not None:
        reason = "physical_state_diverged"
    elif first_outcome_mismatch is not None:
        reason = "transition_outcome_diverged"
    elif not task_success_replay:
        reason = "final_info_succeed_is_false"
    elif not certifiable_input_contract:
        reason = "replay_contract_is_diagnostic_only"
    else:
        reason = "certification_contract_not_satisfied"
    contact_signature_overlay = _contact_signature_overlay_envelope(
        (
            [demo.contact_signature_overlay_segment]
            if demo.contact_signature_overlay_segment is not None
            else []
        )
    )
    replay_origin = getattr(raw_env, "_physical_recording_origin", None)
    detached_state_writes_performed = bool(
        uses_detached_hand_inputs
        and isinstance(replay_origin, Mapping)
        and replay_origin.get("trajectory_per_frame_state_writes") is True
        and int(replay_origin.get("trajectory_state_write_count") or 0) > 0
    )
    return {
        "status": "passed" if passed else "failed",
        "reason": reason,
        "task": demo.task_name,
        "execution_mode": PHYSICS_ONLY_EXECUTION_MODE,
        "source_display_config": _source_display_config(demo),
        "effective_display_config": _effective_display_config(demo),
        "app_window_visibility": _physics_only_app_window_state(env, demo),
        "wall_clock_throttle_hz": 0,
        "source_scene_constructed_before_display_override": True,
        "steps_expected": int(len(demo.actions)),
        "steps_executed": steps_executed,
        "final_succeed": final_succeed,
        "certification_mode": CERTIFICATION_MODE,
        "initialization_mode": (
            FRESH_SCENE_INITIALIZATION_MODE
            if uses_fresh_scene_origin
            else INITIALIZATION_MODE
        ),
        "physical_transition_contract": (demo.physical_transition_contract),
        "trajectory_input_mode": demo.trajectory_input_mode,
        "source_recorder_lifecycle": demo.recorder_lifecycle,
        "source_process_reuse_enabled": bool(demo.source_process_reuse_enabled),
        "fresh_source_process_per_attempt": bool(
            demo.fresh_recorder_process_per_attempt_required
        ),
        "fresh_scene_replay_required": bool(uses_fresh_scene_origin),
        "collection_seed_source": demo.collection_seed_source,
        "episode_reset_ordinal": int(demo.episode_reset_ordinal),
        "episode_seed_derivation": demo.episode_seed_derivation,
        "source_recorder_pid": demo.source_recorder_pid,
        "legacy_raw_action_diagnostic_only": (
            demo.physical_transition_contract == LEGACY_RAW_ACTION_TRANSITION_CONTRACT
            and not legacy_fresh_e01
        ),
        "raw_commands_installed": bool(
            uses_recorded_drive_targets
            and raw_command_install_count == len(demo.actions)
        ),
        "raw_command_install_count": int(raw_command_install_count),
        "task_controller_state_installed": bool(
            task_controller_state_install_count == len(demo.actions)
            if demo.task_name == _P003_TASK_NAME
            else task_controller_state_install_count > 0
        ),
        "task_controller_state_install_count": int(
            task_controller_state_install_count
        ),
        "ik_recomputed": bool(
            not uses_recorded_drive_targets and not uses_detached_hand_inputs
        ),
        "raw_environment_step_called": bool(
            not uses_recorded_drive_targets and steps_executed > 0
        ),
        "recorded_drive_targets_applied": bool(
            uses_recorded_drive_targets
            and recorded_drive_target_substeps_expected > 0
            and recorded_drive_target_substeps_executed
            == recorded_drive_target_substeps_expected
            and recorded_drive_target_writes_executed
            == recorded_drive_target_writes_expected
        ),
        "recorded_drive_target_substeps_expected": int(
            recorded_drive_target_substeps_expected
        ),
        "recorded_drive_target_substeps_executed": int(
            recorded_drive_target_substeps_executed
        ),
        "recorded_drive_target_writes_expected": int(
            recorded_drive_target_writes_expected
        ),
        "recorded_drive_target_writes_executed": int(
            recorded_drive_target_writes_executed
        ),
        "recorded_drive_target_write_count_per_substep": int(
            recorded_drive_target_write_count_per_substep
        ),
        "recorded_drive_target_max_intra_step_delta": float(
            max_intra_step_target_delta
        ),
        "recorded_physics_configuration_match": (recorded_physics_configuration_match),
        "initialization_restore_counts": initialization_restore_counts,
        "initialization_restore_counts_source": (
            INITIALIZATION_RESTORE_COUNTS_SOURCE
            if uses_fresh_scene_origin
            else "recorded_t0_snapshot_restore"
        ),
        "initialization_sleep": initialization_sleep,
        "replayed_stable_origin_history": (replayed_stable_origin_history),
        "initialization_history_match": bool(
            replayed_stable_origin_history is not None
            if uses_fresh_scene_origin
            else True
        ),
        "initialization_readback_match": (initialization_first_divergence is None),
        "initialization_field_max_errors": initialization_errors,
        "initialization_readback_fields": [
            spec.name for spec in initialization_comparison_field_specs
        ],
        "initialization_diagnostic_only_fields": (
            initialization_diagnostic_only_fields
        ),
        "initialization_first_divergence": (initialization_first_divergence),
        "post_stabilization_snapshot_restore_performed": bool(
            not uses_fresh_scene_origin
        ),
        "physics_steps_between_initialization_readback_and_action_0": 0,
        **(
            {"physics_steps_between_restore_and_action_0": 0}
            if not uses_fresh_scene_origin
            else {}
        ),
        "diagnostic_trace_match": diagnostic_trace_match,
        "diagnostic_transition_outcomes_match": (diagnostic_transition_outcomes_match),
        "task_success_replay": task_success_replay,
        "trace_match": trace_match,
        "transition_outcomes_match": transition_outcomes_match,
        "source_identity": source_identity,
        "portable": portable,
        **(
            {CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD: contact_signature_overlay}
            if contact_signature_overlay is not None
            else {}
        ),
        **(
            {
                "source_snapshot_commit": (source_verification.source_snapshot_commit),
                "source_reviewer_upgrade_paths": list(
                    source_verification.reviewer_upgrade_paths
                ),
            }
            if (
                source_verification is not None
                and source_verification.source_snapshot_commit is not None
            )
            else {}
        ),
        **(
            {
                "recorded_git": demo.recorded_git,
                "current_git": current_git,
            }
            if not platform_review
            else {}
        ),
        "runtime_replay_identity": current_runtime_identity,
        "first_divergence": first_divergence,
        "first_outcome_mismatch": first_outcome_mismatch,
        "max_errors": max_errors,
        "field_atols": {spec.name: spec.replay_atol for spec in demo.field_specs},
        "state_writes_performed": True,
        "initialization_state_writes_performed": True,
        "trajectory_state_writes_performed": (
            detached_state_writes_performed
        ),
        "state_writes_after_action_start": (
            detached_state_writes_performed
        ),
        "drive_target_writes_after_action_start": bool(
            uses_recorded_drive_targets and recorded_drive_target_substeps_executed > 0
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Certification value is not JSON-safe: {type(value)}")


def write_report_atomic(
    demo_dir: str | Path,
    report: Mapping[str, Any],
) -> Path:
    demo_path = Path(demo_dir).expanduser().resolve()
    destination = demo_path / REPORT_NAME
    temporary = demo_path / f".{REPORT_NAME}.tmp"
    document = {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "demo_dir": str(demo_path),
        **dict(report),
    }
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                _json_safe(document),
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def _failed_certification_report(
    exc: BaseException,
    *,
    demo: LoadedDemo | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "failed",
        "reason": str(exc),
        "error_type": type(exc).__name__,
        "execution_mode": PHYSICS_ONLY_EXECUTION_MODE,
        "task_success_replay": False,
        "trace_match": False,
        "transition_outcomes_match": False,
        "portable": False,
        "first_divergence": None,
        "max_errors": {},
    }
    if demo is not None:
        report["source_display_config"] = _source_display_config(demo)
        report["effective_display_config"] = _effective_display_config(demo)
        report["wall_clock_throttle_hz"] = 0
        report["source_scene_constructed_before_display_override"] = True
        contact_signature_overlay = _contact_signature_overlay_envelope(
            (
                [demo.contact_signature_overlay_segment]
                if demo.contact_signature_overlay_segment is not None
                else []
            )
        )
        if contact_signature_overlay is not None:
            report[CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD] = contact_signature_overlay
    return report


def _ordered_session_demos(
    demos: Sequence[LoadedDemo],
    *,
    platform_review: bool = False,
) -> list[LoadedDemo]:
    """Validate one shared-runtime episode chain and return ordinal order."""

    if not demos:
        raise CertificationError("A replay session requires at least one demo.")

    for demo in demos:
        if demo.episode_reset_ordinal < 0:
            raise CertificationError(
                "Session replay requires a nonnegative "
                "episode_reset_ordinal for every demo."
            )
    ordered = sorted(
        demos,
        key=lambda demo: (demo.episode_reset_ordinal, str(demo.demo_dir)),
    )
    ordinals = [demo.episode_reset_ordinal for demo in ordered]
    if len(set(ordinals)) != len(ordinals):
        raise CertificationError(
            "Multi-episode replay has duplicate episode_reset_ordinal values."
        )

    shared_fields = [
        "task_name",
        "collection_run_seed",
        "randomize",
        "randomize_dynamics",
        "arm_type",
        "hand_type",
        "render_mode",
        "image_obs",
        "headless",
        "target_label",
        "camera_screen_effect",
        "ipad_screen_effect",
        "environment_variables",
        "recorded_physx_schema",
        "recorded_physx_field_schema",
        "recorded_task_state_field_schema",
        "physical_transition_contract",
        "trajectory_input_mode",
        "recorder_lifecycle",
        "source_process_reuse_enabled",
        "source_scene_rebuild_per_reset",
        "collection_seed_source",
        "episode_seed_derivation",
        "source_recorder_pid",
        "session_history_root_relative_path",
        "session_history_event_count",
    ]
    if not platform_review:
        shared_fields.extend(
            (
                "recorded_git",
                "recorded_runtime_identity",
                "session_history_sha256",
            )
        )
    reference = ordered[0]
    for demo in ordered[1:]:
        for field in shared_fields:
            reference_value = _shared_session_field_value(reference, field)
            actual_value = _shared_session_field_value(demo, field)
            if actual_value != reference_value:
                raise CertificationError(
                    "Multi-episode replay cannot share one environment because "
                    f"{field} differs between ordinal "
                    f"{reference.episode_reset_ordinal} and "
                    f"{demo.episode_reset_ordinal}."
                )
    return ordered


def _resolved_session_history_path(demo: LoadedDemo) -> Path:
    """Resolve a manifest path before or after pending-demo publication."""

    raw_relative = demo.session_history_root_relative_path
    relative = Path(raw_relative)
    if not raw_relative or relative.is_absolute() or ".." in relative.parts:
        raise CertificationError(
            f"Demo {demo.demo_dir.name} has an invalid root-relative PhysX "
            "session-history path."
        )

    # Pending demos live at OUTPUT/.physical_replay_pending/DEMO, while a
    # published demo lives directly at OUTPUT/DEMO.  The manifest path is
    # always relative to OUTPUT, so exactly one of these bases should resolve.
    bases = (demo.demo_dir.parent.parent, demo.demo_dir.parent)
    candidates: dict[Path, Path] = {}
    for raw_base in bases:
        try:
            base = raw_base.resolve(strict=True)
        except OSError:
            continue
        unresolved = base / relative
        cursor = base
        has_symlink = False
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                has_symlink = True
                break
            if not cursor.exists():
                break
        if has_symlink:
            raise CertificationError(
                f"PhysX session-history path may not traverse a symlink: {cursor}"
            )
        if not unresolved.exists():
            continue
        try:
            resolved = unresolved.resolve(strict=True)
            resolved.relative_to(base)
        except (OSError, ValueError) as exc:
            raise CertificationError(
                "PhysX session-history path escaped its output-root candidate: "
                f"{unresolved}"
            ) from exc
        if not resolved.is_dir():
            raise CertificationError(
                f"PhysX session history is not a directory: {resolved}"
            )
        candidates[resolved] = base

    if not candidates:
        raise CertificationError(
            "Could not locate the manifest-bound PhysX session history for "
            f"{demo.demo_dir}."
        )
    if len(candidates) != 1:
        raise CertificationError(
            "PhysX session-history path is ambiguous across pending and "
            f"published output roots for {demo.demo_dir}."
        )
    return next(iter(candidates))


def _load_platform_session_history(path: Path) -> dict[str, Any]:
    """Load ordered replay events without any hash or publication proof."""

    session = _read_json_object(path / SESSION_HEADER_NAME)
    finalization = _read_json_object(path / FINALIZATION_NAME)
    events_root = path / EVENTS_DIRECTORY_NAME
    try:
        event_dirs = sorted(
            (entry for entry in events_root.iterdir() if entry.is_dir()),
            key=lambda entry: int(entry.name),
        )
    except (OSError, ValueError) as exc:
        raise CertificationError(
            f"Could not read ordered PhysX session events from {events_root}: {exc}"
        ) from exc
    events = []
    for event_dir in event_dirs:
        event = _read_json_object(event_dir / EVENT_DOCUMENT_NAME)
        event["event_directory"] = event_dir.relative_to(path).as_posix()
        events.append(event)
    return {
        "session": session,
        "events": events,
        "finalization": finalization,
    }


def _validated_session_history(
    demos: Sequence[LoadedDemo],
    *,
    platform_review: bool = False,
) -> LoadedSessionReplay:
    """Load a sealed journal and bind every accepted event to one demo."""

    if not demos:
        raise CertificationError("A replay session requires at least one demo.")
    for demo in demos:
        if (
            not demo.session_history_root_relative_path
            or (not platform_review and not demo.session_history_sha256)
            or demo.session_history_event_count <= 0
            or demo.session_history_accepted_event_ordinal < 0
        ):
            raise CertificationError(
                f"Demo {demo.demo_dir.name} is missing its complete PhysX "
                "session-history manifest binding."
            )

    paths = {_resolved_session_history_path(demo) for demo in demos}
    if len(paths) != 1:
        raise CertificationError(
            "Accepted demos resolve to different PhysX session histories."
        )
    history_path = next(iter(paths))
    try:
        loaded = (
            _load_platform_session_history(history_path)
            if platform_review
            else load_session_history(history_path, require_finalized=True)
        )
    except Exception as exc:
        raise CertificationError(
            f"Could not verify PhysX session history {history_path}: {exc}"
        ) from exc
    if not isinstance(loaded, Mapping):
        raise CertificationError("PhysX session-history loader returned no object.")
    session = loaded.get("session")
    events_value = loaded.get("events")
    finalization = loaded.get("finalization")
    if not isinstance(session, Mapping):
        raise CertificationError("PhysX session history has no valid header.")
    if not isinstance(events_value, (list, tuple)):
        raise CertificationError("PhysX session history has no ordered events.")
    if not isinstance(finalization, Mapping):
        raise CertificationError("PhysX session history is not finalized.")
    if any(not isinstance(event, Mapping) for event in events_value):
        raise CertificationError("PhysX session history contains an invalid event.")
    events = tuple(dict(event) for event in events_value)

    reference = demos[0]
    if session.get("task_name") != reference.task_name or (
        session.get("collection_run_seed") != reference.collection_run_seed
    ):
        raise CertificationError(
            "PhysX session header does not match the accepted demos' task "
            "and collection seed."
        )
    session_metadata = session.get("metadata")
    recorded_scene_rebuild = (
        session_metadata.get(
            "in_process_physx_scene_rebuild_per_reset",
            False,
        )
        if isinstance(session_metadata, Mapping)
        else False
    )
    if type(recorded_scene_rebuild) is not bool or (
        recorded_scene_rebuild is not reference.source_scene_rebuild_per_reset
    ):
        raise CertificationError(
            "PhysX session header scene-rebuild policy does not match the "
            "accepted demos."
        )
    if finalization.get("task_name") != reference.task_name or (
        finalization.get("collection_run_seed") != reference.collection_run_seed
    ):
        raise CertificationError(
            "PhysX session finalization does not match the accepted demos."
        )

    expected_count = reference.session_history_event_count
    if finalization.get("event_count") != expected_count or (
        len(events) != expected_count
    ):
        raise CertificationError(
            "PhysX session-history event count differs from the manifest."
        )
    recorded_hash_chain = finalization.get("event_sha256_chain")
    if not platform_review:
        history_identity = finalization.get("history_identity")
        if not isinstance(history_identity, Mapping) or (
            str(history_identity.get("sha256") or "").lower()
            != reference.session_history_sha256.lower()
        ):
            raise CertificationError(
                "PhysX session-history SHA-256 differs from the manifest."
            )
        if not isinstance(recorded_hash_chain, list) or (
            len(recorded_hash_chain) != len(events)
        ):
            raise CertificationError(
                "PhysX session finalization has an invalid event hash chain."
            )
    previous_hash: str | None = None
    for ordinal, event in enumerate(events):
        event_hash = event.get("event_sha256")
        if (
            event.get("ordinal") != ordinal
            or (
                not platform_review
                and (
                    event.get("previous_event_sha256") != previous_hash
                    or not isinstance(event_hash, str)
                    or len(event_hash) != 64
                    or event_hash != recorded_hash_chain[ordinal]
                )
            )
        ):
            raise CertificationError(
                f"PhysX session event order/hash mismatch at ordinal {ordinal}."
            )
        previous_hash = event_hash if isinstance(event_hash, str) else None

    accepted_rows = finalization.get("accepted_demos")
    if not isinstance(accepted_rows, list) or len(accepted_rows) != len(demos):
        raise CertificationError(
            "PhysX session accepted-demo list does not exactly match the "
            "certification batch."
        )
    previous_accepted_ordinal = -1
    for index, (demo, accepted) in enumerate(zip(demos, accepted_rows, strict=True)):
        if not isinstance(accepted, Mapping):
            raise CertificationError(
                f"PhysX accepted-demo row {index} is not an object."
            )
        accepted_ordinal = demo.session_history_accepted_event_ordinal
        expected_row = {
            "event_ordinal": accepted_ordinal,
            "reset_ordinal": demo.episode_reset_ordinal,
            "demo_id": demo.demo_dir.name,
            "step_count": len(demo.actions),
        }
        if dict(accepted) != expected_row:
            raise CertificationError(
                "PhysX session accepted-demo order/id/length differs from "
                f"manifest demo {demo.demo_dir.name}."
            )
        if accepted_ordinal <= previous_accepted_ordinal or not (
            0 < accepted_ordinal < len(events)
        ):
            raise CertificationError(
                "PhysX accepted-event ordinals are not strictly ordered."
            )
        previous_accepted_ordinal = accepted_ordinal

        trajectory_event = events[accepted_ordinal]
        trajectory_payload = trajectory_event.get("payload")
        reset_event = events[accepted_ordinal - 1]
        reset_payload = reset_event.get("payload")
        if (
            trajectory_event.get("kind") != "trajectory"
            or not isinstance(trajectory_payload, Mapping)
            or trajectory_payload.get("accepted") is not True
            or trajectory_payload.get("storage") != "accepted_demo_reference"
            or trajectory_payload.get("demo_id") != demo.demo_dir.name
            or trajectory_payload.get("reset_ordinal") != demo.episode_reset_ordinal
            or trajectory_payload.get("step_count") != len(demo.actions)
        ):
            raise CertificationError(
                f"Accepted trajectory event does not bind demo {demo.demo_dir.name}."
            )
        if (
            reset_event.get("kind") != "reset_attempt"
            or not isinstance(reset_payload, Mapping)
            or reset_payload.get("accepted") is not True
            or reset_payload.get("outcome") != "accepted"
            or reset_payload.get("reset_ordinal") != demo.episode_reset_ordinal
            or reset_payload.get("seed") != demo.episode_reset_seed
        ):
            raise CertificationError(
                f"Accepted reset event does not bind demo {demo.demo_dir.name}."
            )
        reset_stable_origin = reset_payload.get("stable_origin")
        if not isinstance(reset_stable_origin, Mapping) or reset_payload.get(
            "physics_steps"
        ) != reset_stable_origin.get("physics_steps"):
            raise CertificationError(
                f"Accepted reset event has no exact physics-step history for "
                f"demo {demo.demo_dir.name}."
            )
        if not platform_review:
            _validated_exact_session_stable_origin_history(
                reset_stable_origin,
                demo.stable_recording_origin,
            )

    if finalization.get("certified_through_event_ordinal") != (
        demos[-1].session_history_accepted_event_ordinal
    ):
        raise CertificationError(
            "PhysX session finalization does not certify through the last "
            "accepted demo event."
        )

    return LoadedSessionReplay(
        path=history_path,
        session=dict(session),
        events=events,
        finalization=dict(finalization),
    )


def _rebuild_session_scene_if_recorded(
    environment: Any,
    payload: Mapping[str, Any],
    *,
    event_ordinal: int,
) -> Mapping[str, Any] | None:
    scene_rebuilt = payload.get("scene_rebuilt_before_reset", False)
    scene_generation = payload.get("scene_generation", 0)
    if type(scene_rebuilt) is not bool or type(scene_generation) is not int:
        raise CertificationError(
            f"Session reset event {event_ordinal} has invalid scene-rebuild evidence."
        )
    if not scene_rebuilt:
        if scene_generation != 0:
            raise CertificationError(
                f"Session reset event {event_ordinal} retained a nonzero scene "
                "generation without rebuilding the scene."
            )
        return None

    seed = payload.get("seed")
    if type(seed) is not int or seed < 0:
        raise CertificationError(
            f"Session reset event {event_ordinal} cannot rebuild without a seed."
        )
    raw_env = getattr(environment, "unwrapped", environment)
    hard_reset = getattr(raw_env, "hard_reset_physx_scene", None)
    if not callable(hard_reset):
        raise CertificationError(
            "Recorded session requires an in-process fresh PhysX Scene, but "
            f"{type(raw_env).__name__} has no hard_reset_physx_scene()."
        )
    result = hard_reset(seed=int(seed))
    if not isinstance(result, Mapping) or result.get("scene_rebuilt") is not True:
        raise CertificationError(
            f"Session reset event {event_ordinal} did not rebuild its PhysX Scene."
        )
    if result.get("scene_generation") != scene_generation:
        raise CertificationError(
            f"Session reset event {event_ordinal} rebuilt scene generation "
            f"{result.get('scene_generation')!r}; expected {scene_generation}."
        )
    return dict(result)


def _session_reset_call(
    environment: Any,
    payload: Mapping[str, Any],
    *,
    demo: LoadedDemo,
    event_ordinal: int,
) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    reset_ordinal = payload.get("reset_ordinal")
    seed = payload.get("seed")
    stable_kwargs = payload.get("stable_kwargs")
    if type(reset_ordinal) is not int or reset_ordinal < 0:
        raise CertificationError(
            f"Session reset event {event_ordinal} has an invalid reset ordinal."
        )
    expected_seed = (demo.collection_run_seed + reset_ordinal) % (2**31)
    if type(seed) is not int or seed != expected_seed:
        raise CertificationError(
            f"Session reset event {event_ordinal} seed is not derived from "
            "the collection seed and reset ordinal."
        )
    if not isinstance(stable_kwargs, Mapping):
        raise CertificationError(
            f"Session reset event {event_ordinal} has no stable kwargs."
        )
    reset_kwargs: dict[str, Any] = {"seed": int(seed)}
    if demo.target_label:
        reset_kwargs["target_label"] = demo.target_label
    _rebuild_session_scene_if_recorded(
        environment,
        payload,
        event_ordinal=event_ordinal,
    )
    reset_result = environment.reset(**reset_kwargs)
    if not isinstance(reset_result, tuple) or len(reset_result) != 2:
        raise CertificationError(
            f"Session reset event {event_ordinal} did not return two values."
        )
    return reset_result[0], reset_result[1], dict(stable_kwargs)


def _validated_rejected_reset_diagnostics(
    payload: Mapping[str, Any],
    *,
    event_ordinal: int,
) -> Mapping[str, Any]:
    diagnostics = payload.get("stability_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise CertificationError(
            f"Rejected reset event {event_ordinal} has no diagnostics object."
        )
    phase = diagnostics.get("failure_phase")
    common_counter_fields = {
        "physics_steps",
        "stability_samples",
        "hold_target_sync_passes",
    }
    if phase in {"stability_timeout", "final_boundary_readback"}:
        expected_fields = {"failure_phase", *common_counter_fields}
        if set(diagnostics) != expected_fields or any(
            type(diagnostics[name]) is not int or diagnostics[name] < 0
            for name in common_counter_fields
        ):
            raise CertificationError(
                f"Rejected reset event {event_ordinal} has invalid {phase} diagnostics."
            )
    elif phase == "authored_reset_snapshot":
        if set(diagnostics) != {"failure_phase", "geometry_failures"} or (
            diagnostics.get("geometry_failures") != ["authored_reset_snapshot_missing"]
        ):
            raise CertificationError(
                f"Rejected reset event {event_ordinal} has invalid authored "
                "reset diagnostics."
            )
    elif phase == "initial_geometry_gate":
        expected_fields = {
            "failure_phase",
            "geometry_failures",
            "reset_max_translation_drift_m",
            "reset_max_rotation_drift_rad",
            "reset_max_coordinate_drift",
            "reset_state_coverage",
            "reset_drift_sample_count",
            "reset_drift_expected_sample_count",
            "reset_drift_all_physics_frames_sampled",
            "reset_pose_drift_peaks",
            "contact_summary",
            "gate_contact_summary",
            *common_counter_fields,
        }
        allowed_failures = {
            "reset_pose_layout_not_comparable",
            "reset_coordinate_drift_exceeded",
            "reset_translation_drift_exceeded",
            "reset_rotation_drift_exceeded",
            "physx_contact_report_invalid",
            "contact_penetration_exceeded",
        }
        failures = diagnostics.get("geometry_failures")
        if (
            set(diagnostics) != expected_fields
            or not isinstance(failures, list)
            or not failures
            or len(set(failures)) != len(failures)
            or any(failure not in allowed_failures for failure in failures)
            or any(
                type(diagnostics[name]) is not int or diagnostics[name] < 0
                for name in common_counter_fields
            )
        ):
            raise CertificationError(
                f"Rejected reset event {event_ordinal} has invalid geometry "
                "diagnostics."
            )
        for name in (
            "reset_max_translation_drift_m",
            "reset_max_rotation_drift_rad",
            "reset_max_coordinate_drift",
        ):
            value = diagnostics.get(name)
            if value is not None and (
                type(value) not in {int, float}
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise CertificationError(
                    f"Rejected reset event {event_ordinal} has invalid {name}."
                )
        coverage = diagnostics.get("reset_state_coverage")
        coverage_fields = {
            "rigid_body",
            "articulation_root",
            "robot_root",
            "tcp",
            "articulation_dof",
            "robot_joint",
            "hand_command",
            "deformable_node",
        }
        if coverage is not None and (
            not isinstance(coverage, Mapping)
            or set(coverage) != coverage_fields
            or any(type(value) is not int or value < 0 for value in coverage.values())
        ):
            raise CertificationError(
                f"Rejected reset event {event_ordinal} has invalid drift coverage."
            )
        peaks = diagnostics.get("reset_pose_drift_peaks")
        if peaks is not None and (
            not isinstance(peaks, list)
            or any(not isinstance(peak, Mapping) for peak in peaks)
        ):
            raise CertificationError(
                f"Rejected reset event {event_ordinal} has invalid drift peaks."
            )
        contact = diagnostics.get("contact_summary")
        gate_contact = diagnostics.get("gate_contact_summary")
        contact_fields = {
            "valid",
            "sample_count",
            "invalid_sample_count",
            "capture_sources",
            "maximum_contact_point_count",
            "maximum_penetration_depth_m",
            "deepest_sample_index",
            "deepest_contact_point_index",
            "error",
            "expected_sample_count",
            "all_physics_frames_sampled",
        }
        gate_contact_fields = {
            "valid",
            "capture_source",
            "contact_point_count",
            "maximum_penetration_depth_m",
            "deepest_contact_point_index",
            "error",
        }
        physics_steps = diagnostics["physics_steps"]
        if (
            not isinstance(contact, Mapping)
            or set(contact) != contact_fields
            or type(contact.get("sample_count")) is not int
            or contact.get("sample_count") != physics_steps + 1
            or contact.get("expected_sample_count") != physics_steps + 1
            or type(contact.get("invalid_sample_count")) is not int
            or contact.get("invalid_sample_count") < 0
            or not isinstance(contact.get("capture_sources"), list)
            or any(
                type(source) is not int or source not in {1, 2}
                for source in contact.get("capture_sources", [])
            )
            or type(contact.get("maximum_contact_point_count")) is not int
            or contact.get("maximum_contact_point_count") < 0
            or type(contact.get("maximum_penetration_depth_m")) not in {int, float}
            or isinstance(contact.get("maximum_penetration_depth_m"), bool)
            or not math.isfinite(float(contact.get("maximum_penetration_depth_m")))
            or float(contact.get("maximum_penetration_depth_m")) < 0.0
            or not isinstance(gate_contact, Mapping)
            or set(gate_contact) != gate_contact_fields
            or diagnostics.get("reset_drift_sample_count") != physics_steps + 1
            or diagnostics.get("reset_drift_expected_sample_count") != physics_steps + 1
            or diagnostics.get("reset_drift_all_physics_frames_sampled") is not True
        ):
            raise CertificationError(
                f"Rejected reset event {event_ordinal} has invalid contact diagnostics."
            )
        gate_penetration = gate_contact.get("maximum_penetration_depth_m")
        gate_valid = gate_contact.get("valid")
        gate_source = gate_contact.get("capture_source")
        if (
            type(contact.get("valid")) is not bool
            or type(contact.get("all_physics_frames_sampled")) is not bool
            or (
                contact.get("valid") is True
                and contact.get("invalid_sample_count") != 0
            )
            or (
                contact.get("valid") is False
                and contact.get("invalid_sample_count") <= 0
            )
            or type(gate_valid) is not bool
            or type(gate_source) is not int
            or (gate_valid is True and gate_source not in {1, 2})
            or (gate_valid is False and gate_source not in {0, 1, 2})
            or type(gate_contact.get("contact_point_count")) is not int
            or gate_contact.get("contact_point_count") < 0
            or (
                gate_valid is True
                and (
                    type(gate_penetration) not in {int, float}
                    or isinstance(gate_penetration, bool)
                    or not math.isfinite(float(gate_penetration))
                    or float(gate_penetration) < 0.0
                )
            )
            or (
                gate_valid is False
                and gate_penetration is not None
                and (
                    type(gate_penetration) not in {int, float}
                    or isinstance(gate_penetration, bool)
                    or not math.isfinite(float(gate_penetration))
                    or float(gate_penetration) < 0.0
                )
            )
        ):
            raise CertificationError(
                f"Rejected reset event {event_ordinal} has malformed contact evidence."
            )
        stable_kwargs = payload.get("stable_kwargs")
        try:
            translation_limit = float(stable_kwargs["reset_translation_drift_limit"])
            rotation_limit = float(stable_kwargs["reset_rotation_drift_limit"])
            penetration_limit = float(stable_kwargs["contact_penetration_limit"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CertificationError(
                f"Rejected reset event {event_ordinal} has no geometry limits."
            ) from exc
        if (
            not all(
                math.isfinite(value) and value > 0.0
                for value in (
                    translation_limit,
                    rotation_limit,
                    penetration_limit,
                )
            )
            or stable_kwargs.get("require_valid_contact_report") is not True
        ):
            raise CertificationError(
                f"Rejected reset event {event_ordinal} has invalid geometry limits."
            )
        coordinate_limit = min(translation_limit, rotation_limit)
        translation = diagnostics.get("reset_max_translation_drift_m")
        rotation = diagnostics.get("reset_max_rotation_drift_rad")
        coordinate = diagnostics.get("reset_max_coordinate_drift")
        drift_frames_valid = diagnostics.get("reset_drift_all_physics_frames_sampled")
        evidence_relations = {
            "reset_pose_layout_not_comparable": drift_frames_valid is not True,
            "reset_translation_drift_exceeded": (
                translation is not None and float(translation) > translation_limit
            ),
            "reset_rotation_drift_exceeded": (
                rotation is not None and float(rotation) > rotation_limit
            ),
            "reset_coordinate_drift_exceeded": (
                coordinate is not None and float(coordinate) > coordinate_limit
            ),
            "physx_contact_report_invalid": (
                contact.get("valid") is not True
                or contact.get("all_physics_frames_sampled") is not True
            ),
            "contact_penetration_exceeded": (
                float(contact["maximum_penetration_depth_m"]) > penetration_limit
            ),
        }
        if any(
            (name in failures) is not relation
            for name, relation in evidence_relations.items()
        ):
            raise CertificationError(
                f"Rejected reset event {event_ordinal} geometry failures do "
                "not match their measured evidence."
            )
    else:
        raise CertificationError(
            f"Rejected reset event {event_ordinal} has unsupported failure "
            f"phase {phase!r}."
        )

    payload_steps = payload.get("physics_steps")
    diagnostics_steps = diagnostics.get("physics_steps")
    if phase == "authored_reset_snapshot":
        if payload_steps is not None:
            raise CertificationError(
                f"Rejected reset event {event_ordinal} advanced physics before "
                "its missing authored snapshot failure."
            )
    elif (
        type(payload_steps) is not int
        or payload_steps <= 0
        or diagnostics_steps != payload_steps
    ):
        raise CertificationError(
            f"Rejected reset event {event_ordinal} changed its physics-step count."
        )
    return diagnostics


def _replay_rejected_session_reset(
    environment: Any,
    payload: Mapping[str, Any],
    *,
    demo: LoadedDemo,
    event_ordinal: int,
    platform_review: bool = False,
) -> None:
    recorded_diagnostics = payload.get("stability_diagnostics")
    if platform_review:
        if not isinstance(recorded_diagnostics, Mapping):
            recorded_diagnostics = {}
    else:
        recorded_diagnostics = _validated_rejected_reset_diagnostics(
            payload,
            event_ordinal=event_ordinal,
        )
    if (
        payload.get("outcome") != "rejected_unstable_origin"
        or payload.get("stable_origin") is not None
        or not isinstance(payload.get("error"), str)
        or not payload.get("error")
    ):
        raise CertificationError(
            f"Rejected reset event {event_ordinal} has an invalid failure contract."
        )
    observation, reset_info, stable_kwargs = _session_reset_call(
        environment,
        payload,
        demo=demo,
        event_ordinal=event_ordinal,
    )
    try:
        stabilize_recording_origin(
            environment,
            observation,
            reset_info,
            **stable_kwargs,
        )
    except RecordingOriginStabilityError as exc:
        actual_diagnostics = getattr(exc, "diagnostics", None)
        if not platform_review and (
            not isinstance(actual_diagnostics, Mapping)
            or _json_safe(actual_diagnostics) != _json_safe(recorded_diagnostics)
        ):
            raise CertificationError(
                f"Rejected reset event {event_ordinal} changed its exact "
                "stability diagnostics: "
                f"recorded={dict(recorded_diagnostics)!r}, "
                f"actual={actual_diagnostics!r}."
            )
        return
    except Exception as exc:
        raise CertificationError(
            f"Rejected reset event {event_ordinal} failed with the wrong "
            f"error type: {type(exc).__name__}: {exc}"
        ) from exc
    raise CertificationError(
        f"Rejected reset event {event_ordinal} unexpectedly became stable."
    )


def _replay_accepted_session_reset(
    environment: Any,
    payload: Mapping[str, Any],
    *,
    demo: LoadedDemo,
    event_ordinal: int,
    platform_review: bool = False,
) -> tuple[Any, Mapping[str, Any]]:
    recorded_origin = payload.get("stable_origin")
    if (
        payload.get("outcome") != "accepted"
        or payload.get("error") is not None
        or not isinstance(recorded_origin, Mapping)
        or payload.get("physics_steps") != recorded_origin.get("physics_steps")
    ):
        raise CertificationError(
            f"Accepted reset event {event_ordinal} has an invalid stable "
            "origin contract."
        )
    observation, reset_info, stable_kwargs = _session_reset_call(
        environment,
        payload,
        demo=demo,
        event_ordinal=event_ordinal,
    )
    try:
        stable_observation, stable_info = stabilize_recording_origin(
            environment,
            observation,
            reset_info,
            **stable_kwargs,
        )
    except RecordingOriginStabilityError as exc:
        raise CertificationError(
            f"Accepted reset event {event_ordinal} became unstable: {exc}"
        ) from exc
    if not isinstance(stable_info, Mapping):
        raise CertificationError(
            f"Accepted reset event {event_ordinal} returned invalid info."
        )
    if not platform_review:
        _validated_replayed_stable_origin_history(recorded_origin, stable_info)
        actual_origin = stable_info.get("physical_recording_origin")
        assert isinstance(actual_origin, Mapping)
        _validated_exact_session_stable_origin_history(
            recorded_origin,
            actual_origin,
        )
    return stable_observation, stable_info


def _validated_discarded_trajectory_arrays(
    arrays: Mapping[str, Any],
    payload: Mapping[str, Any],
    demo: LoadedDemo,
    *,
    event_ordinal: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
    values = {str(name): np.asarray(value) for name, value in arrays.items()}
    step_count = payload.get("step_count")
    if type(step_count) is not int or step_count <= 0:
        raise CertificationError(
            f"Discarded event {event_ordinal} has no nonempty step count."
        )
    recorded_fields = payload.get("fields")
    if not isinstance(recorded_fields, list) or sorted(recorded_fields) != sorted(
        values
    ):
        raise CertificationError(
            f"Discarded event {event_ordinal} field list differs from storage."
        )
    for name in ("action", "terminal", "succeed"):
        if name not in values or values[name].shape[0] != step_count:
            raise CertificationError(
                f"Discarded event {event_ordinal} has invalid {name} data."
            )
    actions = values["action"]
    if (
        actions.ndim != 2
        or actions.dtype.kind not in "iuf"
        or not np.all(np.isfinite(actions))
    ):
        raise CertificationError(
            f"Discarded event {event_ordinal} actions are invalid."
        )
    for name in ("terminal", "succeed"):
        flags = values[name]
        if flags.shape != (step_count,) or not np.all((flags == 0) | (flags == 1)):
            raise CertificationError(
                f"Discarded event {event_ordinal} {name} flags are invalid."
            )
    signature_overrides, overlay_segment = _apply_legacy_contact_signature_overlay(
        values,
        steps=step_count,
        physx_schema=demo.recorded_physx_schema,
        field_schema=demo.recorded_physx_field_schema,
        segment_id=f"session_event:{event_ordinal:08d}",
    )
    values.update(signature_overrides)
    for spec in demo.field_specs:
        current = values.get(spec.name)
        following = values.get(spec.next_name)
        expected_shape = (step_count, *spec.shape)
        if (
            current is None
            or following is None
            or current.shape != expected_shape
            or following.shape != expected_shape
            or current.dtype != spec.dtype
            or following.dtype != spec.dtype
        ):
            raise CertificationError(
                f"Discarded event {event_ordinal} field {spec.name} does not "
                "match the accepted-demo PhysX schema."
            )
        _validate_transition_continuity(
            spec.name,
            current,
            following,
            steps=step_count,
            shape=spec.shape,
            dtype=spec.dtype,
        )
    path_count = _validated_contact_signature_path_count(demo.recorded_physx_schema)
    _validate_recorded_contact_signatures(
        values,
        steps=step_count,
        path_count=path_count,
    )
    return values, overlay_segment


def _snapshot_time_origin(snapshot: Mapping[str, Any]) -> float:
    try:
        values = np.asarray(snapshot["physx_sim_time"]).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise CertificationError("Session replay has no scalar PhysX time.") from exc
    if values.size != 1 or not np.isfinite(values[0]):
        raise CertificationError("Session replay has no finite scalar PhysX time.")
    return float(values[0])


def _replay_discarded_session_trajectory(
    environment: Any,
    *,
    session: LoadedSessionReplay | None,
    event_ordinal: int,
    payload: Mapping[str, Any],
    demo: LoadedDemo,
    stable_observation: Any,
    platform_review: bool = False,
) -> dict[str, int]:
    if (
        payload.get("storage") != "compact_replay_zarr"
        or payload.get("replay_input_mode") != "recorded_articulated_drive_targets"
    ):
        raise CertificationError(
            f"Discarded event {event_ordinal} lacks exact recorded drive "
            "targets; raw-action replay is not session-certifiable."
        )
    try:
        if platform_review:
            event = session.events[event_ordinal]
            relative = Path(str(event.get("event_directory") or ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("discarded trajectory path is invalid")
            replay_path = session.path / relative / DISCARDED_REPLAY_NAME
            replay = zarr.open_group(
                store=zarr.DirectoryStore(str(replay_path)),
                mode="r",
            )
            data = replay["data"]
            raw_arrays = {
                str(name): np.asarray(data[name])
                for name in sorted(data.array_keys())
            }
        else:
            raw_arrays = load_discarded_trajectory(
                session.path,
                event_ordinal,
                require_finalized=True,
            )
    except Exception as exc:
        raise CertificationError(
            f"Could not load discarded trajectory event {event_ordinal}: {exc}"
        ) from exc
    arrays, contact_signature_overlay_segment = _validated_discarded_trajectory_arrays(
        raw_arrays,
        payload,
        demo,
        event_ordinal=event_ordinal,
    )
    if not isinstance(stable_observation, Mapping):
        raise CertificationError(
            "Accepted discarded reset returned no physical observation."
        )

    from handson.sim.isaaclab.articulated_replay import (
        RecordedArticulatedDrivePath,
        run_articulated_drive_control_step,
        validate_recorded_physics_configuration,
    )
    from handson.sim.isaaclab.physx_ground_truth import PhysxGroundTruthTrace

    raw_env = getattr(environment, "unwrapped", environment)
    trace = getattr(raw_env, "_physx_ground_truth_trace", None)
    if trace is None:
        trace = PhysxGroundTruthTrace(raw_env)
        setattr(raw_env, "_physx_ground_truth_trace", trace)
    runtime_schema = trace.schema()
    _validate_runtime_trace_schema(demo, runtime_schema)
    _validate_runtime_task_state_schema(demo, environment)
    recorded_schema = {"physx": demo.recorded_physx_schema}
    try:
        validate_recorded_physics_configuration(
            recorded_schema,
            runtime_schema,
        )
        drive_path = RecordedArticulatedDrivePath(recorded_schema)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CertificationError(
            f"Discarded event {event_ordinal} drive schema is invalid: {exc}"
        ) from exc

    initial_recorded = {spec.name: arrays[spec.name][0] for spec in demo.field_specs}
    has_time = any(
        spec.name in {"physx_sim_time", "physx_substep_sim_time"}
        for spec in demo.field_specs
    )
    recorded_time_origin = _snapshot_time_origin(initial_recorded) if has_time else None
    actual_time_origin = _snapshot_time_origin(stable_observation) if has_time else None
    _errors, divergence = compare_snapshot(
        initial_recorded,
        stable_observation,
        demo.field_specs,
        phase="discarded_reset",
        action_index=None,
        recorded_time_origin=recorded_time_origin,
        actual_time_origin=actual_time_origin,
    )
    if divergence is not None:
        raise CertificationError(
            f"Discarded event {event_ordinal} initial PhysX state diverged: "
            f"{divergence}."
        )

    drive_substeps = 0
    step_count = int(payload["step_count"])
    partial_terminal_guard = payload.get("partial_terminal_guard", False)
    if type(partial_terminal_guard) is not bool:
        raise CertificationError(
            f"Discarded event {event_ordinal} has an invalid partial-terminal-"
            "guard marker."
        )
    for action_index in range(step_count):
        recorded_next = {
            spec.name: arrays[spec.next_name][action_index] for spec in demo.field_specs
        }
        _install_recorded_task_controller_state(
            environment,
            recorded_next,
        )
        _install_recorded_raw_command_without_ik(
            environment,
            arrays["action"][action_index],
            storage_layout=demo.action_storage_layout,
        )
        try:
            allow_partial_guard_step = bool(
                partial_terminal_guard
                and action_index == step_count - 1
            )
            drive_step = (
                drive_path.control_step(
                    recorded_next,
                    allow_partial_terminal_guard=True,
                )
                if allow_partial_guard_step
                else drive_path.control_step(recorded_next)
            )
            result, drive_report = run_articulated_drive_control_step(
                raw_env,
                trace,
                recorded_schema,
                drive_step,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise CertificationError(
                f"Discarded event {event_ordinal} drive replay failed at "
                f"action {action_index}: {exc}"
            ) from exc
        if not isinstance(result, tuple) or len(result) != 5:
            raise CertificationError(
                "Discarded physical control step must return five values."
            )
        drive_substeps += int(drive_report["recorded_substep_count"])
        _errors, divergence = compare_snapshot(
            recorded_next,
            result[0],
            demo.field_specs,
            phase="discarded_post_action",
            action_index=action_index,
            recorded_time_origin=recorded_time_origin,
            actual_time_origin=actual_time_origin,
        )
        if divergence is not None:
            raise CertificationError(
                f"Discarded event {event_ordinal} diverged at action "
                f"{action_index}: {divergence}."
            )
        info = result[4]
        if not isinstance(info, Mapping):
            raise CertificationError("Discarded env.step info must be a mapping.")
        if allow_partial_guard_step and (
            info.get("manual_reset") is not True
            or not isinstance(info.get("numerical_guard_event"), Mapping)
        ):
            raise CertificationError(
                f"Discarded event {event_ordinal} did not reproduce its "
                "terminal numerical-guard reset."
            )
        actual_terminal = bool(result[2] or result[3])
        actual_succeed = bool(info.get("succeed", False))
        if actual_terminal != bool(arrays["terminal"][action_index]) or (
            actual_succeed != bool(arrays["succeed"][action_index])
        ):
            raise CertificationError(
                f"Discarded event {event_ordinal} outcome diverged at action "
                f"{action_index}."
            )
    return {
        "actions_replayed": step_count,
        "drive_substeps_replayed": drive_substeps,
        **(
            {"contact_signature_overlay_segment": (contact_signature_overlay_segment)}
            if contact_signature_overlay_segment is not None
            else {}
        ),
    }


def replay_session_events(
    environment: Any,
    *,
    previous_demo: LoadedDemo | None,
    next_demo: LoadedDemo,
    session_events: LoadedSessionReplay | None = None,
    platform_review: bool = False,
) -> dict[str, Any]:
    """Rebuild every hidden reset/discard prefix before one accepted demo."""

    if not isinstance(session_events, LoadedSessionReplay):
        raise CertificationError(
            "Session replay requires one verified PhysX session history."
        )
    events = session_events.events
    target_accepted_ordinal = next_demo.session_history_accepted_event_ordinal
    target_reset_event_ordinal = target_accepted_ordinal - 1
    start_event_ordinal = (
        0
        if previous_demo is None
        else previous_demo.session_history_accepted_event_ordinal + 1
    )
    if not (0 <= start_event_ordinal <= target_reset_event_ordinal < len(events)):
        raise CertificationError(
            "Session replay event boundary is outside the verified history."
        )

    target_reset = events[target_reset_event_ordinal]
    target_payload = target_reset.get("payload")
    if (
        target_reset.get("kind") != "reset_attempt"
        or not isinstance(target_payload, Mapping)
        or target_payload.get("accepted") is not True
        or target_payload.get("reset_ordinal") != next_demo.episode_reset_ordinal
    ):
        raise CertificationError(
            "The target accepted demo is not preceded by its accepted reset."
        )

    event_ordinal = start_event_ordinal
    reset_attempts = 0
    rejected_reset_attempts = 0
    discarded_trajectories = 0
    discarded_actions = 0
    discarded_drive_substeps = 0
    contact_signature_overlay_segments: list[dict[str, Any]] = []
    while event_ordinal < target_reset_event_ordinal:
        event = events[event_ordinal]
        payload = event.get("payload")
        if event.get("kind") != "reset_attempt" or not isinstance(
            payload,
            Mapping,
        ):
            raise CertificationError(
                f"Session replay expected a reset event at ordinal {event_ordinal}."
            )
        reset_attempts += 1
        accepted = payload.get("accepted")
        if accepted is False:
            _replay_rejected_session_reset(
                environment,
                payload,
                demo=next_demo,
                event_ordinal=event_ordinal,
                platform_review=platform_review,
            )
            rejected_reset_attempts += 1
            event_ordinal += 1
            continue
        if accepted is not True:
            raise CertificationError(
                f"Session reset event {event_ordinal} has no accepted flag."
            )

        stable_observation, _stable_info = _replay_accepted_session_reset(
            environment,
            payload,
            demo=next_demo,
            event_ordinal=event_ordinal,
            platform_review=platform_review,
        )
        trajectory_ordinal = event_ordinal + 1
        if trajectory_ordinal >= target_reset_event_ordinal:
            raise CertificationError(
                "A discarded accepted reset has no trajectory before the "
                "next accepted demo."
            )
        trajectory_event = events[trajectory_ordinal]
        trajectory_payload = trajectory_event.get("payload")
        if (
            trajectory_event.get("kind") != "trajectory"
            or not isinstance(trajectory_payload, Mapping)
            or trajectory_payload.get("accepted") is not False
            or trajectory_payload.get("reset_ordinal") != payload.get("reset_ordinal")
        ):
            raise CertificationError(
                "Discarded trajectory does not immediately follow its "
                f"accepted reset at event {event_ordinal}."
            )
        discarded_trajectories += 1
        step_count = trajectory_payload.get("step_count")
        if type(step_count) is not int or step_count < 0:
            raise CertificationError(
                f"Discarded trajectory event {trajectory_ordinal} has an "
                "invalid step count."
            )
        if step_count == 0:
            if (
                trajectory_payload.get("storage") != "none"
                or trajectory_payload.get("replay_input_mode") != "none"
            ):
                raise CertificationError(
                    "Empty discarded trajectory has an invalid storage/input contract."
                )
        else:
            replay_report = _replay_discarded_session_trajectory(
                environment,
                session=session_events,
                event_ordinal=trajectory_ordinal,
                payload=trajectory_payload,
                demo=next_demo,
                stable_observation=stable_observation,
                platform_review=platform_review,
            )
            discarded_actions += int(replay_report["actions_replayed"])
            discarded_drive_substeps += int(replay_report["drive_substeps_replayed"])
            overlay_segment = replay_report.get("contact_signature_overlay_segment")
            if isinstance(overlay_segment, Mapping):
                contact_signature_overlay_segments.append(dict(overlay_segment))
        event_ordinal += 2

    if event_ordinal != target_reset_event_ordinal:
        raise CertificationError(
            "Session event replay did not stop at the next accepted reset."
        )
    return {
        "status": "replayed_verified_session_prefix",
        "events_replayed": int(target_reset_event_ordinal - start_event_ordinal),
        "first_event_ordinal": (
            int(start_event_ordinal)
            if start_event_ordinal < target_reset_event_ordinal
            else None
        ),
        "last_event_ordinal": (
            int(target_reset_event_ordinal - 1)
            if start_event_ordinal < target_reset_event_ordinal
            else None
        ),
        "target_accepted_reset_event_ordinal": int(target_reset_event_ordinal),
        "target_accepted_trajectory_event_ordinal": int(target_accepted_ordinal),
        "reset_attempts_replayed": int(reset_attempts),
        "rejected_reset_attempts_replayed": int(rejected_reset_attempts),
        "discarded_trajectories_replayed": int(discarded_trajectories),
        "discarded_actions_replayed": int(discarded_actions),
        "discarded_drive_substeps_replayed": int(discarded_drive_substeps),
        "previous_episode_reset_ordinal": (
            None if previous_demo is None else int(previous_demo.episode_reset_ordinal)
        ),
        "next_episode_reset_ordinal": int(next_demo.episode_reset_ordinal),
        "_contact_signature_overlay_segments": (contact_signature_overlay_segments),
    }


def _validate_accepted_demo_session_reset_report(
    demo: LoadedDemo,
    session: LoadedSessionReplay,
    report: Mapping[str, Any],
) -> None:
    reset_event_ordinal = demo.session_history_accepted_event_ordinal - 1
    reset_payload = session.events[reset_event_ordinal].get("payload")
    if not isinstance(reset_payload, Mapping) or not isinstance(
        reset_payload.get("stable_origin"),
        Mapping,
    ):
        raise CertificationError(
            "Accepted demo reset event has no stable-origin history."
        )
    replayed = report.get("replayed_stable_origin_history")
    if not isinstance(replayed, Mapping):
        raise CertificationError(
            "Accepted demo report has no replayed stable-origin history."
        )
    _validated_exact_session_stable_origin_history(
        reset_payload["stable_origin"],
        replayed,
    )


def _session_report_metadata(
    demos: Sequence[LoadedDemo],
    *,
    index: int,
    status: str,
    event_replay: Mapping[str, Any],
    session: LoadedSessionReplay | None,
    blocked_by: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    demo = demos[index]
    accepted_event_ordinal = demo.session_history_accepted_event_ordinal
    metadata: dict[str, Any] = {
        "mode": SESSION_REPLAY_EXECUTION_MODE,
        "status": status,
        "shared_environment": True,
        "episode_index": int(index),
        "episode_count": len(demos),
        "ordered_episode_reset_ordinals": [
            int(demo.episode_reset_ordinal) for demo in demos
        ],
        "accepted_reset_event_ordinal": int(accepted_event_ordinal - 1),
        "accepted_event_ordinal": int(accepted_event_ordinal),
        "replayed_through_event_ordinal": (
            int(accepted_event_ordinal) if status == "passed" else None
        ),
        "event_replay": dict(event_replay),
    }
    if session is not None:
        metadata.update(
            {
                "session_history_id": str(session.session["session_id"]),
                "session_history_sha256": demo.session_history_sha256,
                "session_history_event_count": int(demo.session_history_event_count),
                "history_certified_through_event_ordinal": int(
                    session.finalization["certified_through_event_ordinal"]
                ),
            }
        )
    if blocked_by is not None:
        metadata["blocked_by"] = dict(blocked_by)
    return metadata


def _skipped_session_report(
    demo: LoadedDemo,
    *,
    demos: Sequence[LoadedDemo],
    index: int,
    blocked_by: Mapping[str, Any],
    certified_inputs: Mapping[str, Any] | None,
    session: LoadedSessionReplay | None = None,
) -> dict[str, Any]:
    report = _failed_certification_report(
        CertificationError(
            "Skipped because an earlier episode in the shared-physics "
            "session failed certification."
        ),
        demo=demo,
    )
    report["error_type"] = "SessionReplaySkipped"
    report["session_replay"] = _session_report_metadata(
        demos,
        index=index,
        status="skipped_due_to_prior_failure",
        event_replay={
            "status": "skipped_due_to_prior_failure",
            "events_replayed": 0,
        },
        session=session,
        blocked_by=blocked_by,
    )
    if certified_inputs is not None:
        report["certified_inputs"] = dict(certified_inputs)
    return report


def certify_demo_session(
    demo_dirs: Sequence[str | Path],
    *,
    session_events: LoadedSessionReplay | None = None,
    event_replayer: Callable[..., Mapping[str, Any]] | None = None,
    platform_review: bool = False,
    on_episode_completed: Callable[[Path, Mapping[str, Any]], None] | None = None,
) -> dict[Path, dict[str, Any]]:
    """Replay accepted demos in ordinal order inside one Isaac environment."""

    demo_paths = [Path(path).expanduser().resolve() for path in demo_dirs]
    if not demo_paths:
        raise CertificationError("A replay session requires at least one demo.")
    if len(set(demo_paths)) != len(demo_paths):
        raise CertificationError("A replay session contains a duplicate demo path.")
    if session_events is not None:
        raise CertificationError(
            "Callers may not inject an unverified session history; every "
            "session is loaded from its manifest binding."
        )

    inputs_by_path: dict[Path, Mapping[str, Any] | None] = {}
    loaded_by_path: dict[Path, LoadedDemo] = {}
    preflight_failures: dict[Path, dict[str, Any]] = {}
    for path in demo_paths:
        certified_inputs: Mapping[str, Any] | None = None
        demo: LoadedDemo | None = None
        try:
            if not platform_review:
                certified_inputs = capture_certified_inputs(path)
            demo = load_demo(path)
            inputs_by_path[path] = certified_inputs
            loaded_by_path[path] = demo
        except Exception as exc:
            report = _failed_certification_report(exc, demo=demo)
            if certified_inputs is not None:
                report["certified_inputs"] = dict(certified_inputs)
            write_report_atomic(path, report)
            if on_episode_completed is not None:
                on_episode_completed(path, report)
            preflight_failures[path] = report

    if preflight_failures:
        results: dict[Path, dict[str, Any]] = dict(preflight_failures)
        blocked_path = next(iter(preflight_failures))
        blocked_by = {
            "demo_dir": str(blocked_path),
            "episode_reset_ordinal": None,
        }
        valid_demos = sorted(
            loaded_by_path.values(),
            key=lambda demo: (demo.episode_reset_ordinal, str(demo.demo_dir)),
        )
        for index, demo in enumerate(valid_demos):
            report = _skipped_session_report(
                demo,
                demos=valid_demos,
                index=index,
                blocked_by=blocked_by,
                certified_inputs=inputs_by_path.get(demo.demo_dir),
            )
            write_report_atomic(demo.demo_dir, report)
            if on_episode_completed is not None:
                on_episode_completed(demo.demo_dir, report)
            results[demo.demo_dir] = report
        return results

    replay_session: LoadedSessionReplay | None = None
    try:
        if platform_review:
            demos = _ordered_session_demos(
                list(loaded_by_path.values()),
                platform_review=True,
            )
            replay_session = _validated_session_history(
                demos,
                platform_review=True,
            )
        else:
            demos = _ordered_session_demos(list(loaded_by_path.values()))
            replay_session = _validated_session_history(demos)
    except Exception as exc:
        results = {}
        for demo in sorted(
            loaded_by_path.values(),
            key=lambda item: (item.episode_reset_ordinal, str(item.demo_dir)),
        ):
            report = _failed_certification_report(exc, demo=demo)
            report["session_replay"] = {
                "mode": SESSION_REPLAY_EXECUTION_MODE,
                "status": "preflight_failed",
                "shared_environment": False,
            }
            certified_inputs = inputs_by_path.get(demo.demo_dir)
            if certified_inputs is not None:
                report["certified_inputs"] = dict(certified_inputs)
            write_report_atomic(demo.demo_dir, report)
            if on_episode_completed is not None:
                on_episode_completed(demo.demo_dir, report)
            results[demo.demo_dir] = report
        return results

    assert replay_session is not None

    from handson.tasks.mappings import CONFIG_MAPPING

    if event_replayer is None:
        if platform_review:
            def event_replayer(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
                return replay_session_events(
                    *args,
                    **kwargs,
                    platform_review=True,
                )
        else:
            event_replayer = replay_session_events
    results: dict[Path, dict[str, Any]] = {}
    environment = None
    prior_failure: dict[str, Any] | None = None
    previous_demo: LoadedDemo | None = None
    try:
        with _recorded_environment(demos[0].environment_variables):
            try:
                environment = _construct_environment(
                    CONFIG_MAPPING[demos[0].task_name](),
                    demos[0],
                )
            except Exception as exc:
                first = demos[0]
                report = _failed_certification_report(exc, demo=first)
                certified_inputs = inputs_by_path.get(first.demo_dir)
                if certified_inputs is not None:
                    report["certified_inputs"] = dict(certified_inputs)
                report["session_replay"] = _session_report_metadata(
                    demos,
                    index=0,
                    status="environment_construction_failed",
                    event_replay={
                        "status": "not_started",
                        "events_replayed": 0,
                    },
                    session=replay_session,
                )
                write_report_atomic(first.demo_dir, report)
                if on_episode_completed is not None:
                    on_episode_completed(first.demo_dir, report)
                results[first.demo_dir] = report
                prior_failure = {
                    "demo_dir": str(first.demo_dir),
                    "episode_reset_ordinal": int(first.episode_reset_ordinal),
                }

            for index, demo in enumerate(demos):
                if index == 0 and demo.demo_dir in results:
                    continue
                if prior_failure is not None:
                    report = _skipped_session_report(
                        demo,
                        demos=demos,
                        index=index,
                        blocked_by=prior_failure,
                        certified_inputs=inputs_by_path.get(demo.demo_dir),
                        session=replay_session,
                    )
                    write_report_atomic(demo.demo_dir, report)
                    if on_episode_completed is not None:
                        on_episode_completed(demo.demo_dir, report)
                    results[demo.demo_dir] = report
                    continue

                try:
                    event_report = event_replayer(
                        environment,
                        previous_demo=previous_demo,
                        next_demo=demo,
                        session_events=replay_session,
                    )
                    if not isinstance(event_report, Mapping):
                        raise CertificationError(
                            "Session event replayer must return a mapping."
                        )
                    event_report = dict(event_report)
                    raw_event_overlay_segments = event_report.pop(
                        "_contact_signature_overlay_segments",
                        [],
                    )
                    if not isinstance(raw_event_overlay_segments, list) or any(
                        not isinstance(segment, Mapping)
                        for segment in raw_event_overlay_segments
                    ):
                        raise CertificationError(
                            "Session event replay returned invalid contact "
                            "signature overlay provenance."
                        )
                    event_overlay_segments = [
                        dict(segment) for segment in raw_event_overlay_segments
                    ]
                    if demo.source_scene_rebuild_per_reset:
                        target_reset_event_ordinal = (
                            demo.session_history_accepted_event_ordinal - 1
                        )
                        target_reset_payload = replay_session.events[
                            target_reset_event_ordinal
                        ].get("payload")
                        if not isinstance(target_reset_payload, Mapping):
                            raise CertificationError(
                                "Accepted demo has no valid target reset payload."
                            )
                        target_scene_rebuild = _rebuild_session_scene_if_recorded(
                            environment,
                            target_reset_payload,
                            event_ordinal=target_reset_event_ordinal,
                        )
                        event_report["target_scene_rebuild"] = (
                            None
                            if target_scene_rebuild is None
                            else dict(target_scene_rebuild)
                        )
                    if platform_review:
                        report = run_certification(
                            demo,
                            environment=environment,
                            platform_review=True,
                        )
                    else:
                        report = run_certification(
                            demo,
                            environment=environment,
                        )
                    combined_overlay = _contact_signature_overlay_envelope(
                        [
                            *event_overlay_segments,
                            *(
                                [demo.contact_signature_overlay_segment]
                                if demo.contact_signature_overlay_segment is not None
                                else []
                            ),
                        ]
                    )
                    if combined_overlay is None:
                        report.pop(
                            CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD,
                            None,
                        )
                    else:
                        report[CONTACT_SIGNATURE_OVERLAY_REPORT_FIELD] = (
                            combined_overlay
                        )
                    if not platform_review:
                        _validate_accepted_demo_session_reset_report(
                            demo,
                            replay_session,
                            report,
                        )
                        inputs_after = capture_certified_inputs(demo.demo_dir)
                        if inputs_after != inputs_by_path[demo.demo_dir]:
                            raise CertificationError(
                                "Physical replay inputs changed during "
                                "session certification."
                            )
                        report["certified_inputs"] = inputs_after
                    report["session_replay"] = _session_report_metadata(
                        demos,
                        index=index,
                        status=(
                            "passed" if report.get("status") == "passed" else "failed"
                        ),
                        event_replay=event_report,
                        session=replay_session,
                    )
                except Exception as exc:
                    report = _failed_certification_report(exc, demo=demo)
                    certified_inputs = inputs_by_path.get(demo.demo_dir)
                    if certified_inputs is not None:
                        report["certified_inputs"] = dict(certified_inputs)
                    report["session_replay"] = _session_report_metadata(
                        demos,
                        index=index,
                        status="failed",
                        event_replay={
                            "status": "failed",
                            "events_replayed": 0,
                        },
                        session=replay_session,
                    )

                write_report_atomic(demo.demo_dir, report)
                if on_episode_completed is not None:
                    on_episode_completed(demo.demo_dir, report)
                results[demo.demo_dir] = report
                if report.get("status") != "passed" and not (
                    platform_review and not report.get("error_type")
                ):
                    prior_failure = {
                        "demo_dir": str(demo.demo_dir),
                        "episode_reset_ordinal": int(demo.episode_reset_ordinal),
                    }
                else:
                    previous_demo = demo
    finally:
        if environment is not None:
            with contextlib.suppress(Exception):
                environment.close()
    return results


def certify_demo(demo_dir: str | Path) -> dict[str, Any]:
    demo_path = Path(demo_dir).expanduser().resolve()
    certified_inputs = None
    demo: LoadedDemo | None = None
    environments: list[Any] = []
    try:
        certified_inputs = capture_certified_inputs(demo_path)
        demo = load_demo(demo_path)
        report = run_certification(
            demo,
            on_environment_created=environments.append,
        )
        inputs_after = capture_certified_inputs(demo_path)
        if inputs_after != certified_inputs:
            raise CertificationError(
                "Physical replay inputs changed during certification."
            )
        report["certified_inputs"] = inputs_after
    except Exception as exc:
        report = _failed_certification_report(exc, demo=demo)
        if certified_inputs is not None:
            try:
                if capture_certified_inputs(demo_path) == certified_inputs:
                    report["certified_inputs"] = certified_inputs
            except Exception:
                pass
    # Isaac/Kit shutdown may terminate the interpreter instead of returning
    # from env.close().  The atomic, fsynced report must therefore exist before
    # any environment is asked to close.
    write_report_atomic(demo_path, report)
    for environment in reversed(environments):
        with contextlib.suppress(Exception):
            environment.close()
    return report


def _manifest_certification_mode(demo_dir: str | Path) -> str:
    """Read the capture-time process contract used by the CLI dispatcher."""

    demo_path = Path(demo_dir).expanduser().resolve()
    manifest_path = demo_path / "replay_manifest.json"
    manifest = _read_json_object(manifest_path)
    recording = _required_mapping(manifest, "recording", manifest_path)
    mode = _required_text(
        recording,
        "physical_replay_certification_mode",
        manifest_path,
    ).strip()
    if mode not in {
        FRESH_PROCESS_CERTIFICATION_MODE,
        DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE,
    }:
        raise CertificationError(
            f"Unsupported physical replay certification mode {mode!r}."
        )
    return mode


def certify_cli_demo_dirs(
    demo_dirs: Sequence[str | Path],
) -> dict[Path, dict[str, Any]]:
    """Dispatch fresh candidates singly and deferred candidates by session."""

    demo_paths = [Path(path).expanduser().resolve() for path in demo_dirs]
    if not demo_paths:
        raise CertificationError("At least one replay demo is required.")
    modes = {_manifest_certification_mode(path) for path in demo_paths}
    if len(modes) != 1:
        raise CertificationError(
            "One certifier invocation may not mix fresh-process and deferred "
            "session replay candidates."
        )
    mode = next(iter(modes))
    if mode == FRESH_PROCESS_CERTIFICATION_MODE:
        if len(demo_paths) != 1:
            raise CertificationError(
                "Fresh-process certification requires exactly one episode per "
                "certifier process."
            )
        demo_path = demo_paths[0]
        return {demo_path: certify_demo(demo_path)}
    return certify_demo_session(demo_paths)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-dir",
        action="append",
        required=True,
        type=Path,
    )
    arguments = parser.parse_args(argv)
    reports = certify_cli_demo_dirs(arguments.demo_dir)
    failed = 0
    for demo_dir, report in reports.items():
        print(
            f"[{report['status']}] {demo_dir}: {report.get('reason', '')}",
            flush=True,
        )
        failed += report["status"] != "passed"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
