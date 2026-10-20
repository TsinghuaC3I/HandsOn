"""Fail-closed publication contract for B320-BC021 final assets.

The collection environment is the sole source of this contract through its
``reset_replay_contract``.  The recorder may mirror a valid final declaration
into ``collection_meta.json``; upload code only verifies the two immutable
copies and never invents or upgrades asset readiness itself.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse


BC021_TASK_NAME = "t01_align_blocks_in_midair"
BC021_FINAL_ASSET_CONTRACT_FORMAT = (
    "handson.b320_bc021_final_asset_contract"
)
BC021_FINAL_ASSET_CONTRACT_VERSION = 1
BC021_FINAL_ASSET_STATUS = "final"
BC021_FINAL_ASSET_CONTRACT_KEY = "bc021_final_asset_contract"
BC021_FINAL_ASSET_CONTRACT_SHA256_KEY = (
    "bc021_final_asset_contract_sha256"
)

_FINAL_FLAG_FIELDS = (
    "bc021_asset_integration_status",
    "bc021_geometry_provisional",
    "bc021_formal_collection_ready",
)
_DIGEST_FIELDS = (
    # ``asset_sha256`` identifies the operator-selected source file;
    # ``runtime_asset_sha256`` identifies the converted asset composed by USD.
    "asset_sha256",
    "runtime_asset_sha256",
    "geometry_sha256",
    "collider_sha256",
    "marked_face_frame_sha256",
    "calibration_sha256",
)
_PLACEHOLDER_TEXT = frozenset(
    {
        "",
        "n/a",
        "none",
        "pending",
        "procedural_scaffold",
        "provisional",
        "unknown",
        "waiting_for_user_download",
    }
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_lower_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def bc021_final_asset_contract_sha256(
    contract: Mapping[str, Any],
) -> str:
    """Return the canonical identity of one complete final-asset contract."""

    encoded = json.dumps(
        dict(contract),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _task_name_from_manifest(manifest: Mapping[str, Any]) -> str:
    return str(_mapping(manifest.get("task")).get("name") or "").strip()


def _contract_from_owner(owner: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(owner.get(BC021_FINAL_ASSET_CONTRACT_KEY))


def bc021_episode_declared(
    collection_meta: Mapping[str, Any],
    replay_manifest: Mapping[str, Any],
    *,
    episode_name: str = "",
) -> bool:
    """Detect BC021 even when one task-identity copy was removed or changed."""

    metadata = _mapping(collection_meta)
    manifest = _mapping(replay_manifest)
    recording = _mapping(manifest.get("recording"))
    reset_contract = _mapping(recording.get("reset_replay_contract"))
    task_hints = {
        str(metadata.get("task_name") or "").strip(),
        _task_name_from_manifest(manifest),
        str(recording.get("task_name") or "").strip(),
        str(_contract_from_owner(metadata).get("task_name") or "").strip(),
        str(
            _contract_from_owner(reset_contract).get("task_name") or ""
        ).strip(),
    }
    if BC021_TASK_NAME in task_hints:
        return True
    identity_values = (
        episode_name,
        metadata.get("data_code"),
        metadata.get("episode_id"),
        recording.get("episode_id"),
    )
    if any(
        str(value or "").startswith(f"{BC021_TASK_NAME}_")
        for value in identity_values
    ):
        return True
    return bool(
        any(str(key).startswith("bc021_") for key in metadata)
        or any(str(key).startswith("bc021_") for key in recording)
        or any(str(key).startswith("bc021_") for key in reset_contract)
    )


def _final_asset_contract_errors(
    contract: Mapping[str, Any],
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if not contract:
        return [f"{label}.{BC021_FINAL_ASSET_CONTRACT_KEY} is missing"]
    if contract.get("format") != BC021_FINAL_ASSET_CONTRACT_FORMAT:
        errors.append(f"{label} final-asset contract format is invalid")
    if type(contract.get("version")) is not int or contract.get(
        "version"
    ) != BC021_FINAL_ASSET_CONTRACT_VERSION:
        errors.append(f"{label} final-asset contract version is invalid")
    if contract.get("task_name") != BC021_TASK_NAME:
        errors.append(f"{label} final-asset task identity is invalid")
    if contract.get("asset_integration_status") != BC021_FINAL_ASSET_STATUS:
        errors.append(f"{label} asset integration status is not final")
    if contract.get("geometry_provisional") is not False:
        errors.append(f"{label} geometry is provisional or undeclared")
    if contract.get("formal_collection_ready") is not True:
        errors.append(f"{label} formal collection readiness is not true")

    source_url = contract.get("asset_source_url")
    parsed_source = urlparse(source_url) if isinstance(source_url, str) else None
    if (
        parsed_source is None
        or parsed_source.scheme not in {"http", "https"}
        or not parsed_source.netloc
        or source_url.strip().lower() in _PLACEHOLDER_TEXT
    ):
        errors.append(f"{label} final asset source URL is invalid")
    license_name = contract.get("asset_license")
    if (
        not isinstance(license_name, str)
        or license_name.strip().lower() in _PLACEHOLDER_TEXT
    ):
        errors.append(f"{label} final asset license is invalid")
    if contract.get("asset_commercial_use_allowed") is not True:
        errors.append(
            f"{label} final asset commercial-use approval is not true"
        )
    runtime_asset_path = contract.get("runtime_asset_relative_path")
    if not isinstance(runtime_asset_path, str):
        errors.append(f"{label} runtime asset relative path is invalid")
    else:
        normalized_path = runtime_asset_path.strip().replace("\\", "/")
        pure_path = PurePosixPath(normalized_path)
        if (
            not normalized_path
            or pure_path.is_absolute()
            or ".." in pure_path.parts
            or pure_path.suffix.lower() not in {".usd", ".usda", ".usdc"}
        ):
            errors.append(f"{label} runtime asset relative path is invalid")
    meters_per_unit = contract.get("source_meters_per_unit")
    if (
        isinstance(meters_per_unit, bool)
        or not isinstance(meters_per_unit, (int, float))
        or not math.isfinite(float(meters_per_unit))
        or float(meters_per_unit) <= 0.0
    ):
        errors.append(f"{label} source meters-per-unit is invalid")
    scale_xyz = contract.get("runtime_scale_xyz")
    if (
        not isinstance(scale_xyz, (list, tuple))
        or len(scale_xyz) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in scale_xyz
        )
    ):
        errors.append(f"{label} runtime scale xyz is invalid")
    for field in _DIGEST_FIELDS:
        if not _is_lower_sha256(contract.get(field)):
            errors.append(f"{label}.{field} is not a canonical SHA-256")
    return errors


def _owner_binding_errors(
    owner: Mapping[str, Any],
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if owner.get("bc021_asset_integration_status") != BC021_FINAL_ASSET_STATUS:
        errors.append(f"{label}.bc021_asset_integration_status is not final")
    if owner.get("bc021_geometry_provisional") is not False:
        errors.append(f"{label}.bc021_geometry_provisional must be false")
    if owner.get("bc021_formal_collection_ready") is not True:
        errors.append(f"{label}.bc021_formal_collection_ready must be true")

    contract = _contract_from_owner(owner)
    errors.extend(_final_asset_contract_errors(contract, label=label))
    declared_sha256 = owner.get(BC021_FINAL_ASSET_CONTRACT_SHA256_KEY)
    if not _is_lower_sha256(declared_sha256):
        errors.append(
            f"{label}.{BC021_FINAL_ASSET_CONTRACT_SHA256_KEY} is invalid"
        )
    elif contract:
        try:
            actual_sha256 = bc021_final_asset_contract_sha256(contract)
        except (TypeError, ValueError):
            errors.append(f"{label} final-asset contract is not canonical JSON")
        else:
            if declared_sha256 != actual_sha256:
                errors.append(f"{label} final-asset contract SHA-256 mismatches")
    return errors


def bc021_final_asset_episode_errors(
    collection_meta: Mapping[str, Any],
    replay_manifest: Mapping[str, Any],
    *,
    episode_name: str = "",
) -> list[str]:
    """Validate the mirrored, environment-authored BC021 publication proof."""

    metadata = _mapping(collection_meta)
    manifest = _mapping(replay_manifest)
    if not bc021_episode_declared(
        metadata,
        manifest,
        episode_name=episode_name,
    ):
        return []

    errors: list[str] = []
    metadata_task_name = str(metadata.get("task_name") or "").strip()
    manifest_task_name = _task_name_from_manifest(manifest)
    if metadata_task_name != BC021_TASK_NAME:
        errors.append("collection_meta task identity is not B320-BC021")
    if manifest_task_name != BC021_TASK_NAME:
        errors.append("replay_manifest task identity is not B320-BC021")
    if metadata_task_name != manifest_task_name:
        errors.append("collection_meta and replay_manifest task identities differ")

    recording = _mapping(manifest.get("recording"))
    reset_contract = _mapping(recording.get("reset_replay_contract"))
    if not reset_contract:
        errors.append("replay_manifest.recording.reset_replay_contract is missing")
    errors.extend(_owner_binding_errors(metadata, label="collection_meta"))
    errors.extend(
        _owner_binding_errors(
            reset_contract,
            label="replay_manifest.recording.reset_replay_contract",
        )
    )

    metadata_contract = _contract_from_owner(metadata)
    replay_contract = _contract_from_owner(reset_contract)
    if metadata_contract != replay_contract:
        errors.append("collection_meta and replay final-asset contracts differ")
    metadata_sha256 = metadata.get(BC021_FINAL_ASSET_CONTRACT_SHA256_KEY)
    replay_sha256 = reset_contract.get(
        BC021_FINAL_ASSET_CONTRACT_SHA256_KEY
    )
    if metadata_sha256 != replay_sha256:
        errors.append("collection_meta and replay final-asset SHA-256 differ")
    return errors


def bc021_collection_meta_binding(
    task_name: str,
    reset_replay_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Mirror only an environment-authored, complete final BC021 contract."""

    if str(task_name).strip() != BC021_TASK_NAME:
        return {}
    reset_contract = _mapping(reset_replay_contract)
    if _owner_binding_errors(
        reset_contract,
        label="reset_replay_contract",
    ):
        return {}
    return {
        field: copy.deepcopy(reset_contract[field])
        for field in (
            *_FINAL_FLAG_FIELDS,
            BC021_FINAL_ASSET_CONTRACT_KEY,
            BC021_FINAL_ASSET_CONTRACT_SHA256_KEY,
        )
    }


__all__ = [
    "BC021_FINAL_ASSET_CONTRACT_FORMAT",
    "BC021_FINAL_ASSET_CONTRACT_KEY",
    "BC021_FINAL_ASSET_CONTRACT_SHA256_KEY",
    "BC021_FINAL_ASSET_CONTRACT_VERSION",
    "BC021_FINAL_ASSET_STATUS",
    "BC021_TASK_NAME",
    "bc021_collection_meta_binding",
    "bc021_episode_declared",
    "bc021_final_asset_contract_sha256",
    "bc021_final_asset_episode_errors",
]
