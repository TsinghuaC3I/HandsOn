"""Compact, path-safe publication envelope for verified v3 episodes.

This module deliberately does not mutate any recorder or certification input.
``replay_manifest.json`` is part of the certified input identity, so a compact
cloud representation must be derived beside it rather than rewriting it after
certification.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from handson.data.deferred_arm_replay_artifacts import (
    ARTIFACT_PROOF_FORMAT,
    ARTIFACT_PROOF_VERSION,
    CANONICAL_ARM_VIDEO_PATHS,
    deferred_arm_replay_artifact_errors,
)
from handson.data.physical_replay_certification import (
    DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS,
    DIRTY_SNAPSHOT_REVIEWER_UPGRADE_SOURCE_IDENTITY,
    INPUT_IDENTITY_ALGORITHM,
    PORTABLE_SOURCE_IDENTITIES,
    SOURCE_REVIEWER_UPGRADE_ALLOWED_PATHS,
    capture_certified_inputs,
)
from handson.data.recording_fields import wall_sampling_contract_errors
from handson.data.simplified_post_ik import (
    declares_simplified_post_ik,
    simplified_arm_plan_artifact_errors,
    simplified_arm_plan_errors,
    simplified_post_ik_contract_errors,
)
from handson.runtime.replay_archive import replay_tree_identity


VERIFIED_PUBLICATION_SUFFIX = "__verified_v3"
VERIFIED_EPISODE_FILENAME = "episode_verified_v3.json"
PUBLICATION_COMMIT_FILENAME = "PUBLICATION_COMMIT.v3.json"

VERIFIED_EPISODE_CONTRACT = "handson.verified_episode.v3"
PUBLICATION_COMMIT_CONTRACT = "handson.publication_commit.v3"
VERIFIED_PUBLICATION_SCHEMA_VERSION = 3

# Public integration names use one consistent VERIFIED_PUBLICATION_* prefix.
VERIFIED_PUBLICATION_CONTRACT = VERIFIED_EPISODE_CONTRACT
VERIFIED_PUBLICATION_ENVELOPE_FILENAME = VERIFIED_EPISODE_FILENAME
VERIFIED_PUBLICATION_COMMIT_CONTRACT = PUBLICATION_COMMIT_CONTRACT
VERIFIED_PUBLICATION_COMMIT_FILENAME = PUBLICATION_COMMIT_FILENAME

COLLECTION_META_FILENAME = "collection_meta.json"
REPLAY_MANIFEST_FILENAME = "replay_manifest.json"
RECORDING_SCHEMA_FILENAME = "recording_schema.json"
PHYSICAL_REPLAY_REPORT_FILENAME = "physical_replay_certification.json"
FRESH_PROCESS_ACCEPTANCE_FILENAME = "fresh_process_acceptance.json"
DEFERRED_ARM_REPLAY_STATUS_FILENAME = "deferred_arm_replay_status.json"
WRIST_RERENDER_META_FILENAME = "wrist_rerender_meta.json"
CAMERA_REPAIR_V6_REVIEW_FILENAME = "camera_repair_v6_review.json"
HIGH_SUSPECT_CAMERA_REPAIR_TAG = "高疑似异常"
QUALITY_REVIEW_VIDEO_HASH_FIELD = "canonical_video_sha256"
OFFLINE_ARM_RENDER_STATUS_FILENAME = "offline_arm_render_status.json"
OFFLINE_ARM_ARTIFACT_PROOF_FORMAT = "handson.offline_arm_render_artifacts"
OFFLINE_ARM_ARTIFACT_PROOF_VERSION = 1
OFFLINE_ARM_VIDEO_PATHS = tuple(
    f"videos_arm/{Path(path).name}" for path in CANONICAL_ARM_VIDEO_PATHS
)

_ALLOWED_EVIDENCE_PATHS: frozenset[str] = frozenset()
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {"command", "log_path", "endpoint", "environment"}
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "publication_id",
        "source_episode_id",
        "task_name",
        "collector_id",
        "collection_session_id",
        "collected_at",
        "duration_seconds",
        "step_count",
        "data_fps",
        "video_fps",
        "target_label",
        "arm_type",
        "hand_type",
        "sim_backend",
        "engine_code",
        "task_outcome",
        "source_identity",
        "physical_replay",
        "render",
        "evidence",
    }
)
_SOURCE_IDENTITY_KEYS = frozenset(
    {
        "algorithm",
        "replay_zarr_sha256",
        "replay_zarr_file_count",
        "replay_zarr_bytes",
    }
)
_PHYSICAL_REPLAY_PASSED_KEYS = frozenset(
    {
        "status",
        "portable",
        "source_identity",
        "certification_mode",
        "certification_scope",
        "runs_required",
        "runs_passed",
        "accepted_at",
    }
)
_PHYSICAL_REPLAY_MIGRATED_KEYS = _PHYSICAL_REPLAY_PASSED_KEYS | frozenset(
    {"source_snapshot_commit"}
)
_PHYSICAL_REPLAY_NOT_REQUIRED_KEYS = frozenset({"status"})
_RENDER_VERIFIED_KEYS = frozenset(
    {
        "status",
        "completed_at",
        "video_fps",
        "artifact_proof_format",
        "artifact_proof_version",
        "files",
        "v6",
    }
)
_RENDER_SOURCE_KEYS = frozenset(
    {"status", "completed_at", "video_fps", "files", "v6"}
)
_V6_SOURCE_KEYS = frozenset({"status", "camera_profile"})
_V6_NOT_REQUIRED_KEYS = frozenset({"status"})
_V6_RERENDER_KEYS = frozenset(
    {
        "status",
        "version",
        "camera_profile",
        "method",
        "camera_scope",
        "continuous_physics",
        "per_frame_state_restore",
        "success_in_tail",
        "video_validation",
        "frame_count",
        "visual_helpers_hidden",
    }
)
_COMMIT_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "episode_contract",
        "publication_id",
        "source_episode_id",
        "committed_at",
        "files",
    }
)


class VerifiedPublicationV3Error(ValueError):
    """Raised when local evidence cannot form a verified v3 publication."""


def publication_id_for_episode(demo_dir: str | Path) -> str:
    """Return ``<source basename>__verified_v3`` without double suffixing."""

    name = Path(demo_dir).name
    if not name or name in {".", ".."}:
        raise VerifiedPublicationV3Error("demo_dir has no safe basename")
    if "/" in name or "\\" in name:
        raise VerifiedPublicationV3Error("demo basename must be one path component")
    return name if name.endswith(VERIFIED_PUBLICATION_SUFFIX) else (
        name + VERIFIED_PUBLICATION_SUFFIX
    )


# Backward-compatible internal spelling retained for callers developed while
# this module was being introduced.
publication_id_for_demo = publication_id_for_episode


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize one v3 document in its canonical hash representation."""

    try:
        text = json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise VerifiedPublicationV3Error(
            f"payload is not canonical JSON: {exc}"
        ) from exc
    return (text + "\n").encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _normalize_quality_review_outcome(value: Any) -> str:
    raw = str(value if value is not None else "").strip().lower()
    if raw in {"success", "valid", "vs", "true", "1", "通过", "有效"}:
        return "valid"
    if raw in {
        "failure",
        "invalid",
        "vf",
        "it",
        "ip",
        "false",
        "0",
        "不通过",
        "无效",
    }:
        return "invalid"
    if raw in {"discard", "suspect", "na", "存疑", "待定"}:
        return "suspect"
    if raw in {"bug", "qb", "ex", "漏洞"}:
        return "bug"
    return "unknown"


def quality_review_publication_errors(metadata: Mapping[str, Any]) -> list[str]:
    """Reject an explicit non-success review even if task_outcome says success."""

    quality = metadata.get("quality_review")
    if not isinstance(quality, Mapping) or not quality:
        legacy_value: Any = None
        legacy_declared = False
        for key in ("review_outcome", "final_outcome", "is_valid", "valid"):
            if key in metadata:
                legacy_value = metadata.get(key)
                legacy_declared = True
                break
        if not legacy_declared and metadata.get("labeled_at") and "outcome" in metadata:
            legacy_value = metadata.get("outcome")
            legacy_declared = True
        if not legacy_declared:
            return []
        outcome = _normalize_quality_review_outcome(legacy_value)
        errors = []
        if outcome != "valid":
            errors.append(f"legacy quality review outcome is not valid: {outcome}")
        result_code = str(
            metadata.get("result_code") or metadata.get("reason_code") or ""
        ).strip().upper()
        if result_code.startswith("F"):
            errors.append(
                f"legacy quality review result_code is not successful: {result_code}"
            )
        return errors
    errors: list[str] = []
    raw_outcome = quality.get("final_outcome")
    outcome_declared = raw_outcome not in {None, ""}
    outcome = _normalize_quality_review_outcome(raw_outcome)
    success_declared = "is_success" in quality
    if outcome_declared and outcome != "valid":
        errors.append(f"quality review outcome is not valid: {outcome}")
    if success_declared and quality.get("is_success") is not True:
        errors.append("quality review is_success must be true")
    result_code = str(quality.get("result_code") or "").strip().upper()
    if result_code.startswith("F"):
        errors.append(f"quality review result_code is not successful: {result_code}")
    if not outcome_declared and not success_declared:
        errors.append("quality review is incomplete")
    return errors


def _parse_aware_iso_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _final_quality_review_time(quality: Mapping[str, Any]) -> Any:
    direct = quality.get("reviewed_at")
    if direct:
        return direct
    reviewer_id = str(quality.get("reviewer_id") or "").strip()
    history = quality.get("history")
    rows = history if isinstance(history, list) else []
    for row in reversed(rows):
        if not isinstance(row, Mapping):
            continue
        row_reviewer = str(row.get("reviewer_id") or "").strip()
        if reviewer_id and row_reviewer and row_reviewer != reviewer_id:
            continue
        if row.get("reviewed_at"):
            return row.get("reviewed_at")
    return None


def _human_reviewer_id(value: Any) -> bool:
    reviewer = str(value or "").strip().lower()
    return bool(
        reviewer
        and not reviewer.startswith("auto")
        and reviewer not in {"system", "remote-system", "pipeline", "robot"}
    )


def camera_repair_human_review_errors(
    demo_dir: str | Path,
    metadata: Mapping[str, Any],
    repair_review: Mapping[str, Any],
) -> list[str]:
    """Require a hash-bound human review made after a suspect camera repair."""

    raw_flags = metadata.get("risk_flags")
    risk_flags = {
        str(item).strip()
        for item in raw_flags
        if str(item).strip()
    } if isinstance(raw_flags, list) else set()
    review_status = str(repair_review.get("status") or "").strip().lower()
    repair_tag = str(repair_review.get("tag") or "").strip()
    required = bool(
        HIGH_SUSPECT_CAMERA_REPAIR_TAG in risk_flags
        or repair_tag == HIGH_SUSPECT_CAMERA_REPAIR_TAG
        or review_status in {"replaced_locally", "kept_original"}
    )
    if not required:
        return []

    errors: list[str] = []
    if review_status not in {"replaced_locally", "kept_original"}:
        errors.append("high-suspect camera repair has no resolved repair decision")

    quality = metadata.get("quality_review")
    if not isinstance(quality, Mapping) or not quality:
        return [*errors, "human quality review after camera repair is required"]
    if _normalize_quality_review_outcome(quality.get("final_outcome")) != "valid":
        errors.append("post-repair human quality review must be valid")
    if quality.get("is_success") is not True:
        errors.append("post-repair human quality review must confirm success")
    reviewer_id = str(quality.get("reviewer_id") or "").strip()
    if not _human_reviewer_id(reviewer_id):
        errors.append("post-repair review must identify a human reviewer")

    repair_time = _parse_aware_iso_datetime(repair_review.get("updated_at"))
    review_time = _parse_aware_iso_datetime(_final_quality_review_time(quality))
    if repair_time is None:
        errors.append("camera repair updated_at is missing or invalid")
    if review_time is None:
        errors.append("human quality review reviewed_at is missing or invalid")
    if repair_time is not None and review_time is not None and review_time < repair_time:
        errors.append("human quality review predates the camera repair")

    declared_hashes = quality.get(QUALITY_REVIEW_VIDEO_HASH_FIELD)
    declared_hashes = (
        declared_hashes if isinstance(declared_hashes, Mapping) else {}
    )
    root = Path(demo_dir).expanduser().resolve()
    for relative_path in CANONICAL_ARM_VIDEO_PATHS:
        expected = str(declared_hashes.get(relative_path) or "").strip()
        if not _is_sha256(expected):
            errors.append(
                f"post-repair review is missing canonical video hash: {relative_path}"
            )
            continue
        try:
            actual = _sha256_bytes(_read_regular_file(root, relative_path))
        except (OSError, VerifiedPublicationV3Error) as exc:
            errors.append(f"cannot verify reviewed video {relative_path}: {exc}")
            continue
        if actual != expected:
            errors.append(
                f"post-repair review hash is stale for {relative_path}"
            )
    return errors


def _is_git_object_id(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value
    ) is not None


def _safe_relative_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith("/") or ".." in path.parts:
        return None
    if any(part in {"", "."} for part in path.parts):
        return None
    return path.as_posix()


def _looks_absolute_path(value: str) -> bool:
    return bool(
        value.startswith(("/", "~/", "file://"))
        or _WINDOWS_ABSOLUTE_RE.match(value)
    )


def _public_leak_errors(value: Any, *, location: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_location = f"{location}.{key}"
            if key.lower() in _FORBIDDEN_PUBLIC_KEYS:
                errors.append(f"{child_location} is forbidden in v3 publication")
            errors.extend(_public_leak_errors(child, location=child_location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(
                _public_leak_errors(child, location=f"{location}[{index}]")
            )
    elif isinstance(value, str) and _looks_absolute_path(value):
        errors.append(f"{location} contains an absolute path")
    return errors


def _read_regular_file(root: Path, relative_path: str) -> bytes:
    safe_relative = _safe_relative_path(relative_path)
    if safe_relative != relative_path:
        raise VerifiedPublicationV3Error(
            f"unsafe evidence path: {relative_path!r}"
        )
    path = root / safe_relative
    if path.is_symlink() or not path.is_file():
        raise VerifiedPublicationV3Error(
            f"required evidence is missing or not regular: {relative_path}"
        )
    size_before = path.stat().st_size
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise VerifiedPublicationV3Error(
            f"cannot read evidence {relative_path}: {exc}"
        ) from exc
    if len(data) != size_before or path.stat().st_size != size_before:
        raise VerifiedPublicationV3Error(
            f"evidence changed while reading: {relative_path}"
        )
    return data


def _file_identity_from_bytes(relative_path: str, data: bytes) -> dict[str, Any]:
    return {
        "path": relative_path,
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
    }


def _read_json_evidence(
    root: Path,
    relative_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = _read_regular_file(root, relative_path)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VerifiedPublicationV3Error(
            f"invalid JSON evidence {relative_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise VerifiedPublicationV3Error(
            f"JSON evidence must be an object: {relative_path}"
        )
    return payload, _file_identity_from_bytes(relative_path, data)


def _optional_json_evidence(
    root: Path,
    relative_path: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    path = root / relative_path
    if path.is_symlink():
        raise VerifiedPublicationV3Error(
            f"optional evidence cannot be a symlink: {relative_path}"
        )
    if not path.exists():
        return {}, None
    payload, identity = _read_json_evidence(root, relative_path)
    return payload, identity


def _explicit_bool_declaration(
    metadata: Mapping[str, Any],
    recording: Mapping[str, Any],
    field: str,
) -> bool:
    """Resolve one optional mirrored contract without accepting disagreement."""

    declared = [owner[field] for owner in (metadata, recording) if field in owner]
    if not declared:
        return False
    if any(type(value) is not bool for value in declared):
        raise VerifiedPublicationV3Error(f"{field} declaration must be boolean")
    if len(set(declared)) != 1:
        raise VerifiedPublicationV3Error(
            f"{field} declaration differs between metadata and manifest"
        )
    return declared[0]


def _capture_source_recording_videos(root: Path) -> list[dict[str, Any]]:
    """Hash canonical source MP4s without importing private launch state."""

    videos_dir = root / "videos"
    if videos_dir.is_symlink() or not videos_dir.is_dir():
        return []
    canonical = [root / path for path in CANONICAL_ARM_VIDEO_PATHS]
    if all(path.is_file() and not path.is_symlink() for path in canonical):
        candidates = canonical
    else:
        # Depth streams are sensor payloads, not human-review render views.
        # They may still be included by the publication transport allowlist,
        # but must not become official render files in the envelope.
        candidates = sorted(
            path
            for path in videos_dir.glob("*.mp4")
            if not path.name.endswith("_depth.mp4")
        )
    rows: list[dict[str, Any]] = []
    for path in candidates:
        if path.is_symlink() or not path.is_file():
            raise VerifiedPublicationV3Error(
                f"source video is not a regular file: {path.name}"
            )
        relative_path = path.relative_to(root).as_posix()
        data = _read_regular_file(root, relative_path)
        if not data:
            raise VerifiedPublicationV3Error(
                f"source video is empty: {relative_path}"
            )
        rows.append(_file_identity_from_bytes(relative_path, data))
    return sorted(rows, key=lambda row: row["path"])


def _verified_offline_arm_render(
    root: Path,
    status: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if status.get("status") != "complete":
        raise VerifiedPublicationV3Error("offline arm render is not complete")
    raw_rows = status.get("files")
    raw_rows = raw_rows if isinstance(raw_rows, list) else []
    by_path = {
        str(row.get("path") or ""): row
        for row in raw_rows
        if isinstance(row, Mapping)
    }
    if set(by_path) != set(OFFLINE_ARM_VIDEO_PATHS) or len(raw_rows) != len(
        OFFLINE_ARM_VIDEO_PATHS
    ):
        raise VerifiedPublicationV3Error(
            "offline arm render proof must cover exactly three videos"
        )
    files: list[dict[str, Any]] = []
    for relative_path in OFFLINE_ARM_VIDEO_PATHS:
        data = _read_regular_file(root, relative_path)
        if not data:
            raise VerifiedPublicationV3Error(
                f"offline arm video is empty: {relative_path}"
            )
        actual = _file_identity_from_bytes(relative_path, data)
        row = by_path[relative_path]
        if row.get("bytes") != actual["bytes"] or row.get("sha256") != actual[
            "sha256"
        ]:
            raise VerifiedPublicationV3Error(
                f"offline arm render proof is stale: {relative_path}"
            )
        files.append(actual)
    return sorted(files, key=lambda row: row["path"])


def _legacy_task_name(source_episode_id: str) -> str:
    if "_demo_" in source_episode_id:
        return source_episode_id.split("_demo_", 1)[0]
    return source_episode_id


def _legacy_collected_at(source_episode_id: str) -> str:
    match = re.search(
        r"_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})_\d+(?:__.*)?$",
        source_episode_id,
    )
    if match is None:
        return ""
    return f"{match.group(1)}T{match.group(2)}:{match.group(3)}:{match.group(4)}+08:00"


def _replay_step_count(root: Path) -> int:
    descriptor = root / "replay.zarr" / "data" / "action" / ".zarray"
    if descriptor.is_symlink() or not descriptor.is_file():
        return 0
    try:
        payload = json.loads(descriptor.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return 0
    shape = payload.get("shape") if isinstance(payload, Mapping) else None
    if not isinstance(shape, list) or not shape:
        return 0
    return shape[0] if type(shape[0]) is int and shape[0] >= 0 else 0


def _required_text(payload: Mapping[str, Any], key: str, *, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VerifiedPublicationV3Error(f"{label}.{key} is required")
    return value.strip()


def _optional_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise VerifiedPublicationV3Error(f"{label} must be a positive integer")
    return value


def _verified_publication_payload_path_allowed(value: str) -> bool:
    """Allow only compact metadata and canonical sensor/render artifacts."""

    if value in {
        VERIFIED_EPISODE_FILENAME,
        "replay.zarr.zip",
        RECORDING_SCHEMA_FILENAME,
        "model.mjb",
        "simplified_arm_plan.npz",
    }:
        return True
    path = PurePosixPath(value)
    if len(path.parts) != 2 or path.parts[0] not in {
        "videos",
        "videos_arm",
    }:
        return False
    name = path.name
    return bool(
        name.endswith(".mp4")
        or name.endswith("_depth.npz")
        or name == "camera_info.json"
        or name.endswith("_camera_info.json")
    )


def _nonnegative_number(value: Any, *, label: str) -> int | float:
    if type(value) not in {int, float} or value < 0:
        raise VerifiedPublicationV3Error(
            f"{label} must be a non-negative number"
        )
    return value


def _camera_profile_version(
    metadata: Mapping[str, Any],
    recording: Mapping[str, Any],
) -> str:
    for owner in (recording, metadata):
        profile = owner.get("recording_camera_profile")
        if isinstance(profile, Mapping):
            version = profile.get("version")
            if isinstance(version, str) and version.strip():
                return version.strip()
    return ""


def _compact_v6_summary(
    *,
    metadata: Mapping[str, Any],
    recording: Mapping[str, Any],
    replacement: Mapping[str, Any],
    render_files: list[dict[str, Any]],
    step_count: int,
    v6_required: bool,
) -> dict[str, Any]:
    camera_profile = _camera_profile_version(metadata, recording)
    if not replacement:
        if v6_required:
            raise VerifiedPublicationV3Error(
                "local contract requires v6 rerender proof"
            )
        return {
            "status": "source_recording",
            "camera_profile": camera_profile,
        }

    if replacement.get("version") != 6:
        raise VerifiedPublicationV3Error("wrist rerender metadata is not v6")
    if replacement.get("status") not in {"replaced_locally", "uploaded"}:
        raise VerifiedPublicationV3Error("v6 rerender status is incomplete")
    method = _required_text(replacement, "method", label="wrist_rerender_meta")
    camera_scope = _required_text(
        replacement, "camera_scope", label="wrist_rerender_meta"
    )
    if camera_scope != "main_and_wrists":
        raise VerifiedPublicationV3Error("v6 camera scope must cover main and wrists")

    quality = replacement.get("quality_gate")
    if not isinstance(quality, Mapping):
        raise VerifiedPublicationV3Error("v6 quality_gate is required")
    if quality.get("continuous_physics") is not True:
        raise VerifiedPublicationV3Error("v6 continuous_physics must be true")
    if quality.get("per_frame_state_restore") is not False:
        raise VerifiedPublicationV3Error(
            "v6 per_frame_state_restore must be false"
        )
    if quality.get("success_in_tail") is not True:
        raise VerifiedPublicationV3Error("v6 success_in_tail must be true")
    if quality.get("video_validation") != "passed":
        raise VerifiedPublicationV3Error("v6 video_validation must be passed")
    frame_count = _positive_int(
        quality.get("frame_count"), label="v6 quality_gate.frame_count"
    )
    if frame_count != step_count:
        raise VerifiedPublicationV3Error(
            "v6 frame_count must equal the episode step_count"
        )

    helpers_hidden = all(
        replacement.get(field) is True
        for field in (
            "task_sites_hidden",
            "mocap_target_helpers_hidden",
            "wrist_camera_models_hidden",
        )
    )
    if not helpers_hidden:
        raise VerifiedPublicationV3Error("v6 visual helpers must be hidden")

    proof_by_path = {row["path"]: row for row in render_files}
    replacement_rows = replacement.get("videos")
    if not isinstance(replacement_rows, list):
        raise VerifiedPublicationV3Error("v6 video proof list is required")
    replacement_by_path = {
        str(row.get("path") or ""): row
        for row in replacement_rows
        if isinstance(row, Mapping)
    }
    for relative_path in CANONICAL_ARM_VIDEO_PATHS:
        row = replacement_by_path.get(relative_path)
        if not isinstance(row, Mapping):
            raise VerifiedPublicationV3Error(
                f"v6 metadata is missing {relative_path}"
            )
        if row.get("new_sha256") != proof_by_path[relative_path]["sha256"]:
            raise VerifiedPublicationV3Error(
                f"v6 metadata hash is stale for {relative_path}"
            )
        if row.get("bytes") != proof_by_path[relative_path]["bytes"]:
            raise VerifiedPublicationV3Error(
                f"v6 metadata size is stale for {relative_path}"
            )
        if row.get("frame_count") != frame_count:
            raise VerifiedPublicationV3Error(
                f"v6 frame count is stale for {relative_path}"
            )

    return {
        "status": "rerendered",
        "version": 6,
        "camera_profile": camera_profile or "camera_pose_v6",
        "method": method,
        "camera_scope": camera_scope,
        "continuous_physics": True,
        "per_frame_state_restore": False,
        "success_in_tail": True,
        "video_validation": "passed",
        "frame_count": frame_count,
        "visual_helpers_hidden": True,
    }


def build_episode_verified_v3(
    demo_dir: str | Path,
    *,
    publication_id: str | None = None,
) -> dict[str, Any]:
    """Build the compact public envelope from already-final local evidence.

    The source manifest, collection metadata, reports, and render status are
    read-only.  All output fields are selected from an explicit allowlist.
    """

    raw_root = Path(demo_dir).expanduser()
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise VerifiedPublicationV3Error("demo_dir must be a regular directory")
    root = raw_root.resolve()
    source_episode_id = raw_root.name

    metadata, _metadata_identity = _read_json_evidence(
        root, COLLECTION_META_FILENAME
    )
    manifest, manifest_identity = _optional_json_evidence(
        root, REPLAY_MANIFEST_FILENAME
    )
    replacement, _replacement_identity = _optional_json_evidence(
        root, WRIST_RERENDER_META_FILENAME
    )
    repair_review, _repair_review_identity = _optional_json_evidence(
        root, CAMERA_REPAIR_V6_REVIEW_FILENAME
    )

    task = manifest.get("task")
    task = task if isinstance(task, Mapping) else {}
    recording = manifest.get("recording")
    recording = recording if isinstance(recording, Mapping) else {}

    physical_required = _explicit_bool_declaration(
        metadata, recording, "physical_replay_certification_required"
    )
    arm_render_required = _explicit_bool_declaration(
        metadata, recording, "isaac_arm_render_deferred"
    )
    camera_postrender_required = _explicit_bool_declaration(
        metadata,
        recording,
        "canonical_camera_postrender_required",
    )
    offline_arm_render_required = _explicit_bool_declaration(
        metadata, recording, "offline_arm_render_deferred"
    )
    canonical_isaac_render_required = bool(
        arm_render_required or camera_postrender_required
    )
    if canonical_isaac_render_required and offline_arm_render_required:
        raise VerifiedPublicationV3Error(
            "episode cannot require both Isaac canonical-camera and offline "
            "arm render paths"
        )
    v6_declarations = [
        _explicit_bool_declaration(metadata, recording, field)
        for field in ("camera_repair_v6_required", "v6_rerender_required")
    ]
    v6_required = any(v6_declarations)
    review_status = _optional_text(repair_review, "status")
    if review_status in {"pending_review", "repair_failed"}:
        raise VerifiedPublicationV3Error(
            f"v6 camera repair is not publishable: {review_status}"
        )
    if review_status and review_status not in {
        "replaced_locally",
        "kept_original",
    }:
        raise VerifiedPublicationV3Error(
            f"v6 camera repair status is invalid: {review_status}"
        )
    if review_status == "replaced_locally":
        v6_required = True
    if replacement:
        v6_required = True

    task_name = _optional_text(metadata, "task_name") or str(
        task.get("name") or ""
    ).strip() or _legacy_task_name(source_episode_id)
    if task.get("name") not in {None, "", task_name}:
        raise VerifiedPublicationV3Error(
            "task_name differs between collection metadata and manifest"
        )
    collector_id = _optional_text(metadata, "collector_id") or _optional_text(
        metadata, "collection_collector_id"
    )
    collection_session_id = _optional_text(metadata, "collection_session_id")
    collected_at = _optional_text(metadata, "collected_at") or _legacy_collected_at(
        source_episode_id
    )
    raw_step_count = metadata.get("step_count")
    step_count = (
        raw_step_count
        if type(raw_step_count) is int and raw_step_count >= 0
        else _replay_step_count(root)
    )
    if "step_count" in recording and recording.get("step_count") != step_count:
        raise VerifiedPublicationV3Error(
            "step_count differs between collection metadata and manifest"
        )
    simplified_contract_errors = simplified_post_ik_contract_errors(
        recording,
        metadata,
    )
    if simplified_contract_errors:
        raise VerifiedPublicationV3Error(
            "; ".join(simplified_contract_errors)
        )
    if declares_simplified_post_ik(recording):
        plan_errors = simplified_arm_plan_errors(
            root / "replay.zarr",
            expected_frames=step_count,
        )
        if plan_errors:
            raise VerifiedPublicationV3Error(
                "simplified post-IK arm plan is not publishable: "
                + "; ".join(plan_errors)
            )
    raw_data_fps = metadata.get("data_fps", recording.get("data_fps", 30.0))
    data_fps = _nonnegative_number(raw_data_fps, label="collection_meta.data_fps")
    raw_video_fps = metadata.get("video_fps", recording.get("video_fps", 30))
    video_fps = _positive_int(raw_video_fps, label="collection_meta.video_fps")
    wall_sampling_field = "wall_clock_sampling"
    metadata_has_wall_sampling = wall_sampling_field in metadata
    recording_has_wall_sampling = wall_sampling_field in recording
    if metadata_has_wall_sampling or recording_has_wall_sampling:
        if not metadata_has_wall_sampling or not recording_has_wall_sampling:
            raise VerifiedPublicationV3Error(
                "wall sampling contract must be mirrored in metadata and manifest"
            )
        metadata_wall_sampling = metadata.get(wall_sampling_field)
        recording_wall_sampling = recording.get(wall_sampling_field)
        if metadata_wall_sampling != recording_wall_sampling:
            raise VerifiedPublicationV3Error(
                "wall sampling contract differs between metadata and manifest"
            )
        timing_errors = wall_sampling_contract_errors(recording_wall_sampling)
        if isinstance(recording_wall_sampling, Mapping):
            if recording_wall_sampling.get("target_hz") != video_fps:
                timing_errors.append(
                    "wall sampling target_hz differs from video_fps"
                )
            if recording_wall_sampling.get("sample_count") != step_count:
                timing_errors.append(
                    "wall sampling sample_count differs from step_count"
                )
        if timing_errors:
            raise VerifiedPublicationV3Error("; ".join(timing_errors))
    raw_duration = metadata.get("duration_seconds")
    duration_seconds = (
        _nonnegative_number(raw_duration, label="collection_meta.duration_seconds")
        if raw_duration is not None
        else round(step_count / data_fps, 6)
        if data_fps > 0
        else 0
    )
    quality_review = metadata.get("quality_review")
    quality_review = quality_review if isinstance(quality_review, Mapping) else {}
    review_errors = quality_review_publication_errors(metadata)
    if review_errors:
        raise VerifiedPublicationV3Error("; ".join(review_errors))
    camera_review_errors = camera_repair_human_review_errors(
        root,
        metadata,
        repair_review,
    )
    if camera_review_errors:
        raise VerifiedPublicationV3Error("; ".join(camera_review_errors))
    task_outcome = (
        _optional_text(metadata, "task_outcome")
        or _optional_text(metadata, "source_outcome")
        or ("success" if quality_review.get("is_success") is True else "")
    )
    if task_outcome != "success":
        raise VerifiedPublicationV3Error(
            "verified v3 can only publish successful episodes"
        )

    try:
        replay_identity = replay_tree_identity(root / "replay.zarr")
    except (OSError, ValueError) as exc:
        raise VerifiedPublicationV3Error(
            f"cannot capture replay tree identity: {exc}"
        ) from exc

    # collection_meta is a mutable local/review document. Its selected,
    # immutable capture fields are projected into this envelope, but its raw
    # bytes (especially quality_review) must not affect publication identity.
    evidence: list[dict[str, Any]] = []
    schema_identity: dict[str, Any] | None = None
    schema_path = root / RECORDING_SCHEMA_FILENAME
    if schema_path.exists():
        schema_bytes = _read_regular_file(root, RECORDING_SCHEMA_FILENAME)
        schema_identity = _file_identity_from_bytes(
            RECORDING_SCHEMA_FILENAME, schema_bytes
        )

    if physical_required:
        if manifest_identity is None or schema_identity is None:
            raise VerifiedPublicationV3Error(
                "declared physical replay requires manifest and recording schema"
            )
        report, _report_identity = _read_json_evidence(
            root, PHYSICAL_REPLAY_REPORT_FILENAME
        )
        acceptance, _acceptance_identity = _read_json_evidence(
            root, FRESH_PROCESS_ACCEPTANCE_FILENAME
        )
        try:
            actual_inputs = capture_certified_inputs(root)
        except (OSError, UnicodeError, OverflowError) as exc:
            raise VerifiedPublicationV3Error(
                f"cannot capture current certified inputs: {exc}"
            ) from exc
        if report.get("certified_inputs") != actual_inputs:
            raise VerifiedPublicationV3Error(
                "physical replay report is stale for current inputs"
            )
        if acceptance.get("certified_inputs") != actual_inputs:
            raise VerifiedPublicationV3Error(
                "fresh process acceptance is stale for current inputs"
            )
        if manifest_identity["sha256"] != actual_inputs["replay_manifest_sha256"]:
            raise VerifiedPublicationV3Error(
                "manifest identity changed while building"
            )
        if schema_identity["sha256"] != actual_inputs["recording_schema_sha256"]:
            raise VerifiedPublicationV3Error(
                "recording schema identity changed while building"
            )
        if report.get("status") != "passed":
            raise VerifiedPublicationV3Error("physical replay report is not passed")
        if report.get("portable") is not True:
            raise VerifiedPublicationV3Error(
                "physical replay report is not portable"
            )
        report_source_identity = report.get("source_identity")
        if report_source_identity not in PORTABLE_SOURCE_IDENTITIES:
            raise VerifiedPublicationV3Error(
                "portable replay has an unsupported source identity"
            )
        source_snapshot_commit: str | None = None
        if (
            report_source_identity
            == DIRTY_SNAPSHOT_REVIEWER_UPGRADE_SOURCE_IDENTITY
        ):
            source_snapshot_commit = report.get("source_snapshot_commit")
            reviewer_paths = report.get("source_reviewer_upgrade_paths")
            recorded_git = report.get("recorded_git")
            current_git = report.get("current_git")
            if not _is_git_object_id(source_snapshot_commit):
                raise VerifiedPublicationV3Error(
                    "migrated replay lacks a valid source snapshot commit"
                )
            if (
                not isinstance(reviewer_paths, list)
                or not reviewer_paths
                or any(not isinstance(path, str) for path in reviewer_paths)
                or reviewer_paths != sorted(set(reviewer_paths))
                or not set(reviewer_paths).issubset(
                    SOURCE_REVIEWER_UPGRADE_ALLOWED_PATHS
                )
            ):
                raise VerifiedPublicationV3Error(
                    "migrated replay has invalid reviewer upgrade paths"
                )
            if (
                not isinstance(recorded_git, Mapping)
                or recorded_git.get("dirty") is not True
                or not _is_git_object_id(recorded_git.get("commit"))
                or not isinstance(recorded_git.get("changed_files"), list)
                or not recorded_git.get("changed_files")
                or any(
                    not isinstance(path, str) or not path
                    for path in recorded_git.get("changed_files", [])
                )
                or recorded_git.get("changed_files")
                != sorted(set(recorded_git.get("changed_files", [])))
                or not _is_sha256(
                    recorded_git.get("worktree_content_sha256")
                )
                or recorded_git.get("commit") == source_snapshot_commit
            ):
                raise VerifiedPublicationV3Error(
                    "migrated replay has invalid recorded Git evidence"
                )
            if (
                not isinstance(current_git, Mapping)
                or current_git.get("dirty") is not False
                or not _is_git_object_id(current_git.get("commit"))
                or current_git.get("changed_files") != []
                or current_git.get("commit") == source_snapshot_commit
                or (
                    isinstance(recorded_git, Mapping)
                    and current_git.get("commit") == recorded_git.get("commit")
                )
            ):
                raise VerifiedPublicationV3Error(
                    "migrated replay has invalid current Git evidence"
                )
        elif (
            "source_snapshot_commit" in report
            or "source_reviewer_upgrade_paths" in report
        ):
            raise VerifiedPublicationV3Error(
                "clean replay cannot carry dirty snapshot migration evidence"
            )
        for field in (
            "task_success_replay",
            "trace_match",
            "transition_outcomes_match",
        ):
            if report.get(field) is not True:
                raise VerifiedPublicationV3Error(
                    f"physical replay {field} is not true"
                )
        if acceptance.get("status") != "passed":
            raise VerifiedPublicationV3Error(
                "fresh process acceptance is not passed"
            )
        if (
            acceptance.get("published_after_post_collection_certification")
            is not True
        ):
            raise VerifiedPublicationV3Error(
                "episode was not atomically published after certification"
            )
        runs_required = _positive_int(
            acceptance.get("certification_runs_required"),
            label="acceptance.certification_runs_required",
        )
        runs_passed = _positive_int(
            acceptance.get("certification_runs_passed"),
            label="acceptance.certification_runs_passed",
        )
        if (
            runs_required != DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS
            or runs_passed != runs_required
        ):
            raise VerifiedPublicationV3Error(
                "verified v3 requires all three certification runs to pass"
            )
        physical_summary: dict[str, Any] = {
            "status": "passed",
            "portable": True,
            "source_identity": report_source_identity,
            "certification_mode": _required_text(
                report, "certification_mode", label="physical_replay_report"
            ),
            "certification_scope": _required_text(
                acceptance, "certification_scope", label="acceptance"
            ),
            "runs_required": runs_required,
            "runs_passed": runs_passed,
            "accepted_at": _required_text(
                acceptance, "accepted_at", label="acceptance"
            ),
            **(
                {"source_snapshot_commit": source_snapshot_commit}
                if source_snapshot_commit is not None
                else {}
            ),
        }
    else:
        physical_summary = {"status": "not_required"}

    if canonical_isaac_render_required:
        render_status, _render_status_identity = _read_json_evidence(
            root, DEFERRED_ARM_REPLAY_STATUS_FILENAME
        )
        if render_status.get("status") != "complete":
            raise VerifiedPublicationV3Error(
                "Isaac canonical-camera post-render is not complete"
            )
        if render_status.get("artifact_validation_passed") is not True:
            raise VerifiedPublicationV3Error(
                "render artifact validation did not pass"
            )
        artifact_proof = render_status.get("artifact_proof")
        artifact_errors = deferred_arm_replay_artifact_errors(root, artifact_proof)
        if artifact_errors:
            raise VerifiedPublicationV3Error("; ".join(artifact_errors))
        if declares_simplified_post_ik(recording):
            plan_artifact_errors = simplified_arm_plan_artifact_errors(
                root,
                render_status.get("simplified_arm_plan_artifact"),
            )
            if plan_artifact_errors:
                raise VerifiedPublicationV3Error(
                    "; ".join(plan_artifact_errors)
                )
        assert isinstance(artifact_proof, Mapping)
        raw_render_files = artifact_proof.get("files")
        assert isinstance(raw_render_files, list)
        render_files = sorted(
            (
                {
                    "path": str(row["path"]),
                    "bytes": int(row["bytes"]),
                    "sha256": str(row["sha256"]),
                }
                for row in raw_render_files
                if isinstance(row, Mapping)
            ),
            key=lambda row: row["path"],
        )
        render_video_fps = _positive_int(
            render_status.get("video_fps", video_fps),
            label="render_status.video_fps",
        )
        completed_at = _optional_text(
            render_status, "completed_at"
        ) or _required_text(render_status, "updated_at", label="render_status")
        v6_summary = _compact_v6_summary(
            metadata=metadata,
            recording=recording,
            replacement=replacement,
            render_files=render_files,
            step_count=step_count,
            v6_required=v6_required,
        )
        render_summary: dict[str, Any] = {
            "status": "complete",
            "completed_at": completed_at,
            "video_fps": render_video_fps,
            "artifact_proof_format": ARTIFACT_PROOF_FORMAT,
            "artifact_proof_version": ARTIFACT_PROOF_VERSION,
            "files": render_files,
            "v6": v6_summary,
        }
    elif offline_arm_render_required:
        if replacement or v6_required:
            raise VerifiedPublicationV3Error(
                "offline C01/Plan arm render cannot carry Isaac v6 replacement"
            )
        offline_status, _offline_status_identity = _read_json_evidence(
            root, OFFLINE_ARM_RENDER_STATUS_FILENAME
        )
        render_files = _verified_offline_arm_render(root, offline_status)
        completed_at = _optional_text(
            offline_status, "completed_at"
        ) or _required_text(
            offline_status, "updated_at", label="offline_arm_render_status"
        )
        render_summary = {
            "status": "complete",
            "completed_at": completed_at,
            "video_fps": _positive_int(
                offline_status.get("video_fps", video_fps),
                label="offline_arm_render_status.video_fps",
            ),
            "artifact_proof_format": OFFLINE_ARM_ARTIFACT_PROOF_FORMAT,
            "artifact_proof_version": OFFLINE_ARM_ARTIFACT_PROOF_VERSION,
            "files": render_files,
            "v6": {
                "status": "source_recording",
                "camera_profile": _camera_profile_version(
                    metadata, recording
                ),
            },
        }
    else:
        render_files = _capture_source_recording_videos(root)
        if render_files:
            v6_summary = _compact_v6_summary(
                metadata=metadata,
                recording=recording,
                replacement=replacement,
                render_files=render_files,
                step_count=step_count,
                v6_required=v6_required,
            )
            render_summary = {
                "status": "source_recording",
                "completed_at": collected_at,
                "video_fps": video_fps,
                "files": render_files,
                "v6": v6_summary,
            }
        elif v6_required:
            raise VerifiedPublicationV3Error(
                "v6 rerender was declared but no canonical videos exist"
            )
        else:
            render_summary = {
                "status": "not_required",
                "completed_at": collected_at,
                "video_fps": video_fps,
                "files": [],
                "v6": {"status": "not_required"},
            }

    expected_publication_id = publication_id_for_episode(raw_root)
    if publication_id is not None and publication_id != expected_publication_id:
        raise VerifiedPublicationV3Error(
            "publication_id must equal the idempotent verified-v3 episode id"
        )
    envelope: dict[str, Any] = {
        "contract": VERIFIED_EPISODE_CONTRACT,
        "schema_version": VERIFIED_PUBLICATION_SCHEMA_VERSION,
        "publication_id": expected_publication_id,
        "source_episode_id": source_episode_id,
        "task_name": task_name,
        "collector_id": collector_id,
        "collection_session_id": collection_session_id,
        "collected_at": collected_at,
        "duration_seconds": duration_seconds,
        "step_count": step_count,
        "data_fps": data_fps,
        "video_fps": video_fps,
        "target_label": _optional_text(metadata, "target_label"),
        "arm_type": _optional_text(metadata, "arm_type"),
        "hand_type": _optional_text(metadata, "hand_type"),
        "sim_backend": _required_text(
            metadata, "sim_backend", label="collection_meta"
        ),
        "engine_code": _required_text(
            metadata, "engine_code", label="collection_meta"
        ),
        "task_outcome": task_outcome,
        "source_identity": {
            "algorithm": replay_identity["algorithm"],
            "replay_zarr_sha256": replay_identity["sha256"],
            "replay_zarr_file_count": replay_identity["file_count"],
            "replay_zarr_bytes": replay_identity["bytes"],
        },
        "physical_replay": physical_summary,
        "render": render_summary,
        "evidence": evidence,
    }
    validate_verified_episode_v3(envelope)
    return envelope


build_verified_episode_v3 = build_episode_verified_v3


def _file_row_errors(row: Any, *, label: str) -> list[str]:
    if not isinstance(row, Mapping):
        return [f"{label} must be an object"]
    errors: list[str] = []
    if set(row) != {"path", "bytes", "sha256"}:
        errors.append(f"{label} must contain only path/bytes/sha256")
    if _safe_relative_path(row.get("path")) is None:
        errors.append(f"{label}.path must be a safe relative POSIX path")
    if type(row.get("bytes")) is not int or row.get("bytes") < 0:
        errors.append(f"{label}.bytes must be a non-negative integer")
    if not _is_sha256(row.get("sha256")):
        errors.append(f"{label}.sha256 must be lowercase SHA-256")
    return errors


def verified_episode_v3_errors(payload: Any) -> list[str]:
    """Return fail-closed schema and public-data leakage errors."""

    if not isinstance(payload, Mapping):
        return ["verified episode envelope must be an object"]
    errors: list[str] = []
    if set(payload) != _TOP_LEVEL_KEYS:
        errors.append("verified episode top-level fields do not match v3 schema")
    if payload.get("contract") != VERIFIED_EPISODE_CONTRACT:
        errors.append("verified episode contract is invalid")
    if payload.get("schema_version") != VERIFIED_PUBLICATION_SCHEMA_VERSION:
        errors.append("verified episode schema_version is invalid")
    source_episode_id = payload.get("source_episode_id")
    if (
        not isinstance(source_episode_id, str)
        or not source_episode_id
        or Path(source_episode_id).name != source_episode_id
        or source_episode_id in {".", ".."}
    ):
        errors.append("source_episode_id must be one safe basename")
    else:
        try:
            expected_publication_id = publication_id_for_episode(source_episode_id)
        except VerifiedPublicationV3Error as exc:
            errors.append(str(exc))
        else:
            if payload.get("publication_id") != expected_publication_id:
                errors.append("publication_id does not match the source basename")

    for field in (
        "task_name",
        "sim_backend",
        "engine_code",
    ):
        if not isinstance(payload.get(field), str) or not payload.get(field):
            errors.append(f"{field} must be a non-empty string")
    for field in (
        "collector_id",
        "collection_session_id",
        "collected_at",
        "target_label",
        "arm_type",
        "hand_type",
    ):
        if not isinstance(payload.get(field), str):
            errors.append(f"{field} must be a string")
    if payload.get("task_outcome") != "success":
        errors.append("task_outcome must be success")
    if type(payload.get("step_count")) is not int or payload.get("step_count") < 0:
        errors.append("step_count must be a non-negative integer")
    if type(payload.get("video_fps")) is not int or payload.get("video_fps") <= 0:
        errors.append("video_fps must be a positive integer")
    for field in ("duration_seconds", "data_fps"):
        value = payload.get(field)
        if type(value) not in {int, float} or value < 0:
            errors.append(f"{field} must be a non-negative number")

    source = payload.get("source_identity")
    if not isinstance(source, Mapping) or set(source) != _SOURCE_IDENTITY_KEYS:
        errors.append("source_identity does not match v3 schema")
    else:
        if source.get("algorithm") != INPUT_IDENTITY_ALGORITHM:
            errors.append("source_identity algorithm is invalid")
        if not _is_sha256(source.get("replay_zarr_sha256")):
            errors.append("source replay_zarr_sha256 is invalid")
        for field in ("replay_zarr_file_count", "replay_zarr_bytes"):
            if type(source.get(field)) is not int or source.get(field) < 0:
                errors.append(f"source_identity.{field} is invalid")

    physical = payload.get("physical_replay")
    if not isinstance(physical, Mapping):
        errors.append("physical_replay does not match v3 schema")
    elif physical.get("status") == "not_required":
        if set(physical) != _PHYSICAL_REPLAY_NOT_REQUIRED_KEYS:
            errors.append("not-required physical replay has unexpected fields")
    elif physical.get("status") == "passed":
        physical_source_identity = physical.get("source_identity")
        expected_physical_keys = (
            _PHYSICAL_REPLAY_MIGRATED_KEYS
            if physical_source_identity
            == DIRTY_SNAPSHOT_REVIEWER_UPGRADE_SOURCE_IDENTITY
            else _PHYSICAL_REPLAY_PASSED_KEYS
        )
        if set(physical) != expected_physical_keys:
            errors.append("passed physical_replay does not match v3 schema")
        if physical.get("portable") is not True:
            errors.append("physical_replay portable must be true")
        if physical_source_identity not in PORTABLE_SOURCE_IDENTITIES:
            errors.append("physical_replay source identity is unsupported")
        if (
            physical_source_identity
            == DIRTY_SNAPSHOT_REVIEWER_UPGRADE_SOURCE_IDENTITY
            and not _is_git_object_id(physical.get("source_snapshot_commit"))
        ):
            errors.append("physical_replay source snapshot commit is invalid")
        if physical.get("runs_required") != DEFAULT_PHYSICAL_REPLAY_CERTIFICATION_RUNS:
            errors.append("physical_replay must require three runs")
        if physical.get("runs_passed") != physical.get("runs_required"):
            errors.append("all physical replay runs must pass")
        for field in ("certification_mode", "certification_scope", "accepted_at"):
            if not isinstance(physical.get(field), str) or not physical.get(field):
                errors.append(f"physical_replay.{field} must be non-empty")
    else:
        errors.append("physical_replay status is invalid")

    render = payload.get("render")
    proof_format: Any = None
    if not isinstance(render, Mapping):
        errors.append("render does not match v3 schema")
    else:
        render_status = render.get("status")
        if render_status == "complete":
            if set(render) != _RENDER_VERIFIED_KEYS:
                errors.append("complete render does not match v3 schema")
        elif render_status in {"source_recording", "not_required"}:
            if set(render) != _RENDER_SOURCE_KEYS:
                errors.append("source render does not match v3 schema")
        else:
            errors.append("render status is invalid")
        completed_at = render.get("completed_at")
        if (
            not isinstance(completed_at, str)
            or (not completed_at and render_status != "not_required")
        ):
            errors.append("render completed_at must be non-empty")
        if type(render.get("video_fps")) is not int or render.get("video_fps") <= 0:
            errors.append("render video_fps must be positive")
        if render_status == "complete":
            proof_format = render.get("artifact_proof_format")
            proof_version = render.get("artifact_proof_version")
            if proof_format == ARTIFACT_PROOF_FORMAT:
                expected_proof_version = ARTIFACT_PROOF_VERSION
                expected_render_paths = set(CANONICAL_ARM_VIDEO_PATHS)
            elif proof_format == OFFLINE_ARM_ARTIFACT_PROOF_FORMAT:
                expected_proof_version = OFFLINE_ARM_ARTIFACT_PROOF_VERSION
                expected_render_paths = set(OFFLINE_ARM_VIDEO_PATHS)
            else:
                expected_proof_version = None
                expected_render_paths = set()
                errors.append("render artifact proof format is invalid")
            if proof_version != expected_proof_version:
                errors.append("render artifact proof version is invalid")
        else:
            expected_render_paths = set()
        rows = render.get("files")
        rows = rows if isinstance(rows, list) else []
        for index, row in enumerate(rows):
            errors.extend(_file_row_errors(row, label=f"render.files[{index}]"))
        paths = [row.get("path") for row in rows if isinstance(row, Mapping)]
        if len(paths) != len(set(paths)):
            errors.append("render file paths must be unique")
        if render_status == "complete" and (
            len(paths) != 3 or set(paths) != expected_render_paths
        ):
            errors.append("render files must cover exactly three canonical videos")
        if render_status == "source_recording" and not paths:
            errors.append("source recording must contain at least one video")
        if render_status == "not_required" and paths:
            errors.append("not-required render cannot contain videos")
        v6 = render.get("v6")
        if not isinstance(v6, Mapping):
            errors.append("render.v6 must be an object")
        elif v6.get("status") == "source_recording":
            if set(v6) != _V6_SOURCE_KEYS:
                errors.append("source-recording v6 summary has unexpected fields")
            if not isinstance(v6.get("camera_profile"), str):
                errors.append("source-recording camera_profile must be a string")
        elif v6.get("status") == "not_required":
            if set(v6) != _V6_NOT_REQUIRED_KEYS:
                errors.append("not-required v6 summary has unexpected fields")
        elif v6.get("status") == "rerendered":
            if set(v6) != _V6_RERENDER_KEYS:
                errors.append("rerendered v6 summary has unexpected fields")
            if v6.get("version") != 6:
                errors.append("rerendered v6 version is invalid")
            for field in ("camera_profile", "method"):
                if not isinstance(v6.get(field), str) or not v6.get(field):
                    errors.append(f"rerendered v6 {field} is invalid")
            if v6.get("camera_scope") != "main_and_wrists":
                errors.append("rerendered v6 camera scope is invalid")
            if v6.get("continuous_physics") is not True:
                errors.append("rerendered v6 continuous physics is invalid")
            if v6.get("per_frame_state_restore") is not False:
                errors.append("rerendered v6 state restore policy is invalid")
            if v6.get("success_in_tail") is not True:
                errors.append("rerendered v6 success gate is invalid")
            if v6.get("video_validation") != "passed":
                errors.append("rerendered v6 video validation is invalid")
            if type(v6.get("frame_count")) is not int or v6.get("frame_count") <= 0:
                errors.append("rerendered v6 frame_count is invalid")
            if v6.get("visual_helpers_hidden") is not True:
                errors.append("rerendered v6 visual helper gate is invalid")
        else:
            errors.append("render.v6 status is invalid")

    evidence = payload.get("evidence")
    evidence = evidence if isinstance(evidence, list) else []
    evidence_paths: list[str] = []
    for index, row in enumerate(evidence):
        errors.extend(_file_row_errors(row, label=f"evidence[{index}]"))
        if isinstance(row, Mapping) and isinstance(row.get("path"), str):
            evidence_paths.append(row["path"])
    if len(evidence_paths) != len(set(evidence_paths)):
        errors.append("evidence paths must be unique")
    if not set(evidence_paths).issubset(_ALLOWED_EVIDENCE_PATHS):
        errors.append("verified v3 evidence contains a non-whitelisted file")

    errors.extend(_public_leak_errors(payload))
    return errors


def validate_verified_episode_v3(payload: Any) -> None:
    """Raise :class:`VerifiedPublicationV3Error` unless the envelope is valid."""

    errors = verified_episode_v3_errors(payload)
    if errors:
        raise VerifiedPublicationV3Error("; ".join(errors))


def verified_episode_v3_bytes(payload: Mapping[str, Any]) -> bytes:
    """Validate and return the exact bytes whose hash is used by the commit."""

    validate_verified_episode_v3(payload)
    return canonical_json_bytes(payload)


def _compact_file_rows(files: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    compact: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(files):
        if not isinstance(raw_row, Mapping):
            raise VerifiedPublicationV3Error(
                f"files[{index}] must be an object"
            )
        row = {
            "path": raw_row.get("path"),
            "bytes": raw_row.get("bytes"),
            "sha256": raw_row.get("sha256"),
        }
        row_errors = _file_row_errors(row, label=f"files[{index}]")
        if row_errors:
            raise VerifiedPublicationV3Error("; ".join(row_errors))
        path = str(row["path"])
        if path == PUBLICATION_COMMIT_FILENAME:
            raise VerifiedPublicationV3Error(
                "publication commit cannot include its own identity"
            )
        if path in compact and compact[path] != row:
            raise VerifiedPublicationV3Error(
                f"conflicting duplicate publication file: {path}"
            )
        compact[path] = row
    return [compact[path] for path in sorted(compact)]


def build_publication_commit_v3(
    envelope: Mapping[str, Any],
    files: Iterable[Mapping[str, Any]],
    *,
    committed_at: str,
) -> dict[str, Any]:
    """Purely construct the final cloud commit marker.

    Input rows may contain uploader-specific fields, but the returned ``files``
    list contains only ``path``, ``bytes``, and ``sha256``.  The canonical
    envelope identity is always inserted (or replaced) automatically.
    """

    validate_verified_episode_v3(envelope)
    if not isinstance(committed_at, str) or not committed_at.strip():
        raise VerifiedPublicationV3Error("committed_at is required")
    if _looks_absolute_path(committed_at):
        raise VerifiedPublicationV3Error("committed_at cannot be an absolute path")

    rows = _compact_file_rows(files)
    envelope_bytes = verified_episode_v3_bytes(envelope)
    envelope_row = {
        "path": VERIFIED_EPISODE_FILENAME,
        "bytes": len(envelope_bytes),
        "sha256": _sha256_bytes(envelope_bytes),
    }
    by_path = {row["path"]: row for row in rows}
    by_path[VERIFIED_EPISODE_FILENAME] = envelope_row
    compact_rows = [by_path[path] for path in sorted(by_path)]

    commit = {
        "contract": PUBLICATION_COMMIT_CONTRACT,
        "schema_version": VERIFIED_PUBLICATION_SCHEMA_VERSION,
        "episode_contract": VERIFIED_EPISODE_CONTRACT,
        "publication_id": envelope["publication_id"],
        "source_episode_id": envelope["source_episode_id"],
        "committed_at": committed_at.strip(),
        "files": compact_rows,
    }
    validate_publication_commit_v3(commit, envelope=envelope)
    return commit


def publication_commit_v3_errors(
    payload: Any,
    *,
    envelope: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return schema errors for a final verified-v3 commit marker."""

    if not isinstance(payload, Mapping):
        return ["publication commit must be an object"]
    errors: list[str] = []
    if set(payload) != _COMMIT_KEYS:
        errors.append("publication commit top-level fields do not match v3 schema")
    if payload.get("contract") != PUBLICATION_COMMIT_CONTRACT:
        errors.append("publication commit contract is invalid")
    if payload.get("schema_version") != VERIFIED_PUBLICATION_SCHEMA_VERSION:
        errors.append("publication commit schema_version is invalid")
    if payload.get("episode_contract") != VERIFIED_EPISODE_CONTRACT:
        errors.append("publication commit episode_contract is invalid")
    if not isinstance(payload.get("committed_at"), str) or not payload.get(
        "committed_at"
    ):
        errors.append("publication commit committed_at is required")

    rows = payload.get("files")
    rows = rows if isinstance(rows, list) else []
    paths: list[str] = []
    by_path: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        errors.extend(_file_row_errors(row, label=f"files[{index}]"))
        if isinstance(row, Mapping) and isinstance(row.get("path"), str):
            paths.append(row["path"])
            by_path[row["path"]] = row
    if len(paths) != len(set(paths)):
        errors.append("publication commit file paths must be unique")
    if paths != sorted(paths):
        errors.append("publication commit files must be sorted by path")
    if PUBLICATION_COMMIT_FILENAME in paths:
        errors.append("publication commit cannot include itself")
    if COLLECTION_META_FILENAME in paths:
        errors.append(
            "mutable collection_meta review sidecar cannot enter the publication commit"
        )
    if "replay.zarr.zip" not in paths:
        errors.append("publication commit is missing replay.zarr.zip")
    unexpected_paths = sorted(
        path
        for path in paths
        if not _verified_publication_payload_path_allowed(path)
    )
    if unexpected_paths:
        errors.append(
            "publication commit contains non-compact/private files: "
            + ", ".join(unexpected_paths)
        )
    envelope_row = by_path.get(VERIFIED_EPISODE_FILENAME)
    if not isinstance(envelope_row, Mapping):
        errors.append("publication commit is missing the verified envelope hash")

    source_episode_id = payload.get("source_episode_id")
    if isinstance(source_episode_id, str):
        try:
            expected_id = publication_id_for_episode(source_episode_id)
        except VerifiedPublicationV3Error as exc:
            errors.append(str(exc))
        else:
            if payload.get("publication_id") != expected_id:
                errors.append("publication commit publication_id is invalid")
    else:
        errors.append("publication commit source_episode_id is invalid")

    if envelope is not None:
        envelope_errors = verified_episode_v3_errors(envelope)
        if envelope_errors:
            errors.append("referenced verified episode envelope is invalid")
        else:
            if payload.get("publication_id") != envelope.get("publication_id"):
                errors.append("commit and envelope publication_id differ")
            if payload.get("source_episode_id") != envelope.get(
                "source_episode_id"
            ):
                errors.append("commit and envelope source_episode_id differ")
            expected_bytes = canonical_json_bytes(envelope)
            expected_row = {
                "path": VERIFIED_EPISODE_FILENAME,
                "bytes": len(expected_bytes),
                "sha256": _sha256_bytes(expected_bytes),
            }
            if envelope_row != expected_row:
                errors.append("commit envelope hash does not match the envelope")
            render = envelope.get("render")
            render = render if isinstance(render, Mapping) else {}
            render_rows = render.get("files")
            render_rows = render_rows if isinstance(render_rows, list) else []
            required_render_paths = {
                str(row.get("path") or "")
                for row in render_rows
                if isinstance(row, Mapping) and row.get("path")
            }
            missing_render = sorted(required_render_paths - set(paths))
            if missing_render:
                errors.append(
                    "publication commit is missing render artifacts: "
                    + ", ".join(missing_render)
                )
            for row in render_rows:
                if (
                    isinstance(row, Mapping)
                    and row.get("path")
                    and by_path.get(str(row["path"])) != dict(row)
                ):
                    errors.append(
                        "publication commit render identity differs from envelope: "
                        + str(row["path"])
                    )
            unexpected_videos = sorted(
                path
                for path in paths
                if path.endswith(".mp4")
                and not path.endswith("_depth.mp4")
                and path not in required_render_paths
            )
            if unexpected_videos:
                errors.append(
                    "publication commit contains non-official videos: "
                    + ", ".join(unexpected_videos)
                )
            evidence = envelope.get("evidence")
            evidence = evidence if isinstance(evidence, list) else []
            # collection_meta is projected to a mutable compact review index;
            # every other declared source evidence object stays immutable and
            # must be retrievable from the committed prefix.
            required_evidence_paths = {
                str(row.get("path") or "")
                for row in evidence
                if isinstance(row, Mapping)
                and row.get("path")
                and row.get("path") != COLLECTION_META_FILENAME
            }
            missing_evidence = sorted(required_evidence_paths - set(paths))
            if missing_evidence:
                errors.append(
                    "publication commit is missing evidence: "
                    + ", ".join(missing_evidence)
                )
            for row in evidence:
                if (
                    isinstance(row, Mapping)
                    and row.get("path")
                    and row.get("path") != COLLECTION_META_FILENAME
                    and by_path.get(str(row["path"])) != dict(row)
                ):
                    errors.append(
                        "publication commit evidence identity differs from envelope: "
                        + str(row["path"])
                    )
    errors.extend(_public_leak_errors(payload))
    return errors


def validate_publication_commit_v3(
    payload: Any,
    *,
    envelope: Mapping[str, Any] | None = None,
) -> None:
    """Raise unless one publication commit is structurally valid."""

    errors = publication_commit_v3_errors(payload, envelope=envelope)
    if errors:
        raise VerifiedPublicationV3Error("; ".join(errors))


__all__ = [
    "CAMERA_REPAIR_V6_REVIEW_FILENAME",
    "HIGH_SUSPECT_CAMERA_REPAIR_TAG",
    "OFFLINE_ARM_ARTIFACT_PROOF_FORMAT",
    "OFFLINE_ARM_ARTIFACT_PROOF_VERSION",
    "OFFLINE_ARM_RENDER_STATUS_FILENAME",
    "OFFLINE_ARM_VIDEO_PATHS",
    "PUBLICATION_COMMIT_CONTRACT",
    "PUBLICATION_COMMIT_FILENAME",
    "QUALITY_REVIEW_VIDEO_HASH_FIELD",
    "VERIFIED_PUBLICATION_COMMIT_CONTRACT",
    "VERIFIED_PUBLICATION_COMMIT_FILENAME",
    "VERIFIED_PUBLICATION_CONTRACT",
    "VERIFIED_PUBLICATION_ENVELOPE_FILENAME",
    "VERIFIED_EPISODE_CONTRACT",
    "VERIFIED_EPISODE_FILENAME",
    "VERIFIED_PUBLICATION_SCHEMA_VERSION",
    "VERIFIED_PUBLICATION_SUFFIX",
    "VerifiedPublicationV3Error",
    "build_publication_commit_v3",
    "build_episode_verified_v3",
    "build_verified_episode_v3",
    "camera_repair_human_review_errors",
    "canonical_json_bytes",
    "publication_commit_v3_errors",
    "publication_id_for_demo",
    "publication_id_for_episode",
    "quality_review_publication_errors",
    "validate_publication_commit_v3",
    "validate_verified_episode_v3",
    "verified_episode_v3_bytes",
    "verified_episode_v3_errors",
]
