"""Fail-closed NVIDIA Newton replay-profile and runtime identity helpers.

Every HandsOn Newton run uses Warp's ``GPU_TO_GPU`` deterministic atomic mode.
Formal cross-GPU portability remains an empirical physical-replay claim rather
than a blanket bitwise claim: the mode covers supported atomic operations, and
the distinct-GPU replay gate checks the complete task transition.  Newton 1.4
is required because its MuJoCo-Warp solver computes the generated-kernel record
bound from the finalized model instead of relying on the old fixed bound.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import hashlib
import os
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any


NEWTON_REPLAY_PROFILE_ENV = "HANDSON_NEWTON_REPLAY_PROFILE"
NEWTON_DETERMINISTIC_MAX_RECORDS_ENV = (
    "HANDSON_WARP_DETERMINISTIC_MAX_RECORDS"
)
NEWTON_DETERMINISTIC_MAX_RECORDS_DEFAULT = 8
NEWTON_LOCAL_DEBUG_PROFILE = "local_debug_v1"
NEWTON_CROSS_GPU_PROFILE = "cross_gpu_v1"
NEWTON_CROSS_GPU_METHOD = (
    "warp_gpu_to_gpu_plus_semantic_three_fresh_runs_distinct_gpu_model_v2"
)
NEWTON_REPLAY_PROFILES = frozenset(
    {NEWTON_LOCAL_DEBUG_PROFILE, NEWTON_CROSS_GPU_PROFILE}
)

_CROSS_GPU_MINIMUM_VERSIONS = {
    "isaaclab-newton": (0, 13, 6),
    "newton": (1, 4, 0),
    "newton-actuators": (0, 1, 0),
    "warp-lang": (1, 15, 0),
    "mujoco": (3, 10, 0),
    "mujoco-warp": (3, 10, 0, 2),
}
_RUNTIME_PACKAGES = (
    "isaacsim",
    "isaaclab",
    "isaaclab-newton",
    "newton",
    "newton-actuators",
    "warp-lang",
    "mujoco",
    "mujoco-warp",
    "numpy",
    "torch",
)

# These values describe the host/card that executed a run.  The projection is
# recorded for audit and compared exactly for same-model smoke checks.  Across
# GPU models only card identity and OS/platform differences are non-blocking;
# both CUDA/NVIDIA driver identities remain part of ``software_identity``.
_RUNTIME_ENVIRONMENT_IDENTITY_FIELDS = (
    "platform",
    "device",
    "gpu_name",
    "gpu_arch",
    "gpu_uuid",
    "gpu_pci_bus_id",
    "cuda_driver_version",
    "nvidia_driver_version",
)

# Cross-model replay is only meaningful when both hosts execute the exact same
# task physics, recorder and replay implementation.  Package versions alone do
# not cover a dirty checkout or locally regenerated MJCF assets, so bind the
# formal runtime identity to the bytes that define B03's physical transition.
_B03_PHYSICS_SOURCE_PATHS = (
    "handson/handson/data/b03_newton_validity.py",
    "handson/handson/data/newton_cross_gpu_certification.py",
    "handson/handson/sim/assembly_success.py",
    "handson/handson/sim/isaaclab/abc_collection_env.py",
    "handson/handson/sim/isaaclab/bimanual_assembly_env.py",
    "handson/handson/sim/isaaclab/bimanual_assembly_scene.py",
    "handson/handson/sim/isaaclab/compat.py",
    "handson/handson/sim/isaaclab/newton_determinism.py",
    "handson/handson/sim/isaaclab/newton_ground_truth.py",
    "handson/handson/sim/isaaclab/newton_replay.py",
    "handson/handson/sim/isaaclab/physics_config.py",
    "handson/handson/sim/isaaclab/physx_ground_truth.py",
    "handson/handson/sim/isaaclab/task_preview.py",
    "handson/handson/tasks/t09_insert_peg/config.py",
    "handson/handson/tasks/sim_teleop.py",
    "handson/handson/sim/envs/xmls/arena_arm_hand_t09_insert_peg_rokae_wuji1.xml",
    "handson/handson/sim/envs/xmls/t09_insert_peg_scene.xml",
    "handson/handson/sim/envs/xmls/industreal_round_peg_8mm.xml",
    "handson/handson/sim/envs/xmls/industreal_tray_insert_round_peg_8mm.xml",
    "handson/handson/sim/envs/xmls/rokae_wuji1_left.xml",
    "handson/handson/sim/envs/xmls/rokae_wuji1_right.xml",
    "handson/handson/sim/envs/xmls/wuji_hand1/mjcf/left_primitive.xml",
    "handson/handson/sim/envs/xmls/wuji_hand1/mjcf/right_primitive.xml",
    "scripts/record_demos_zarr.py",
    "scripts/replay_b03_newton_physics.py",
    "scripts/install_newton_gpu_to_gpu_overlay.sh",
    "scripts/patches/newton_gpu_to_gpu_v1/b03_qd_projection.patch",
)

_NEWTON_QD_PROJECTION_MARKER = "HANDSON_B03_NEWTON_QD_PROJECTION_V1"
_NEWTON_QD_PROJECTION_MODULES = (
    "mujoco_warp._src.forward",
    "newton._src.solvers.mujoco.solver_mujoco",
)
_B03_MJCF_CLOSURE_ROOTS = (
    "handson/handson/sim/envs/xmls/"
    "arena_arm_hand_t09_insert_peg_rokae_wuji1.xml",
)


class NewtonCrossGpuUnavailable(RuntimeError):
    """Raised when a runtime cannot satisfy the cross-GPU profile."""


def _newton_module_source_path(module_name: str) -> Path | None:
    """Resolve an overlay module without importing Newton or USD packages."""

    loaded = sys.modules.get(module_name)
    if loaded is not None:
        raw_path = getattr(loaded, "__file__", None)
        if not raw_path:
            raise NewtonCrossGpuUnavailable(
                "Loaded Newton velocity projection module has no source path: "
                f"{module_name}"
            )
        candidate = Path(raw_path)
        if candidate.suffix in {".pyc", ".pyo"}:
            try:
                candidate = Path(importlib.util.source_from_cache(str(candidate)))
            except (ValueError, NotImplementedError):
                pass
        if not candidate.is_file():
            raise NewtonCrossGpuUnavailable(
                "Loaded Newton velocity projection source is unavailable: "
                f"{module_name}: {candidate}"
            )
        return candidate

    relative = Path(*module_name.split(".")).with_suffix(".py")
    for entry in sys.path:
        candidate = Path(entry or os.curdir) / relative
        if candidate.is_file():
            return candidate
    return None


def qualify_newton_velocity_projection_runtime() -> dict[str, Any]:
    """Verify and fingerprint B03's solver-internal hard velocity projection."""

    digest = hashlib.sha256()
    module_hashes: dict[str, str] = {}
    for module_name in _NEWTON_QD_PROJECTION_MODULES:
        path = _newton_module_source_path(module_name)
        if path is None:
            raise NewtonCrossGpuUnavailable(
                f"Newton velocity projection source is unavailable: {module_name}"
            )
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise NewtonCrossGpuUnavailable(
                "Newton velocity projection source cannot be read: "
                f"{module_name}: {exc}"
            ) from exc
        if _NEWTON_QD_PROJECTION_MARKER.encode("utf-8") not in payload:
            raise NewtonCrossGpuUnavailable(
                "B03 Newton requires the solver-internal qd projection patch; "
                f"the marker is missing from {module_name}. Reinstall the "
                "GPU_TO_GPU overlay with install_newton_gpu_to_gpu_overlay.sh."
            )
        label = module_name.encode("utf-8")
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        module_hashes[module_name] = hashlib.sha256(payload).hexdigest()
    return {
        "format": "handson.newton_qd_projection",
        "version": 1,
        "algorithm": "candidate_qd_clamp_before_semi_implicit_qpos",
        "scope": "model_joint_velocity_limit_all_mapped_dofs",
        "integrator": "implicitfast",
        "trajectory_state_install": False,
        "cuda_graph_safe": True,
        "module_sha256": module_hashes,
        "combined_sha256": digest.hexdigest(),
    }


@lru_cache(maxsize=1)
def b03_physics_source_fingerprint() -> str:
    """Hash every repository file that defines B03's Newton transition.

    Paths and byte lengths are framed into the digest, so concatenation or
    checkout-location differences cannot create an ambiguous identity.
    """

    repository_root = Path(__file__).resolve().parents[4]
    digest = hashlib.sha256()
    for relative in _B03_PHYSICS_SOURCE_PATHS:
        path = repository_root / relative
        payload = path.read_bytes()
        label = relative.encode("utf-8")
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    pending = [repository_root / relative for relative in _B03_MJCF_CLOSURE_ROOTS]
    visited_documents: set[Path] = set()
    referenced_assets: set[Path] = set()
    while pending:
        document = pending.pop().resolve()
        if document in visited_documents:
            continue
        visited_documents.add(document)
        root = ET.parse(document).getroot()
        compiler = root.find("compiler")
        asset_dir = compiler.attrib.get("assetdir", "") if compiler is not None else ""
        mesh_dir = (
            compiler.attrib.get("meshdir", asset_dir)
            if compiler is not None
            else asset_dir
        )
        texture_dir = (
            compiler.attrib.get("texturedir", asset_dir)
            if compiler is not None
            else asset_dir
        )
        for element in root.iter():
            raw_reference = element.attrib.get("file")
            if not raw_reference:
                continue
            if element.tag == "include":
                relative_dir = ""
            elif element.tag in {"mesh", "skin"}:
                relative_dir = mesh_dir
            elif element.tag in {"texture", "hfield"}:
                relative_dir = texture_dir
            else:
                relative_dir = asset_dir
            candidate = Path(raw_reference)
            asset = (
                candidate
                if candidate.is_absolute()
                else document.parent / relative_dir / candidate
            ).resolve()
            if not asset.is_file():
                raise FileNotFoundError(
                    f"B03 MJCF closure references a missing file: {asset}"
                )
            referenced_assets.add(asset)
            if element.tag == "include":
                pending.append(asset)
    for path in sorted(referenced_assets):
        try:
            relative = path.relative_to(repository_root).as_posix()
        except ValueError as exc:
            raise OSError(
                f"B03 MJCF closure escapes the repository: {path}"
            ) from exc
        payload = path.read_bytes()
        label = f"mjcf_closure:{relative}".encode("utf-8")
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def normalize_newton_replay_profile(value: str | None = None) -> str:
    """Return one canonical replay profile name."""

    raw = (
        os.environ.get(NEWTON_REPLAY_PROFILE_ENV, NEWTON_LOCAL_DEBUG_PROFILE)
        if value is None
        else value
    )
    profile = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "": NEWTON_LOCAL_DEBUG_PROFILE,
        "local": NEWTON_LOCAL_DEBUG_PROFILE,
        "debug": NEWTON_LOCAL_DEBUG_PROFILE,
        "local_v1": NEWTON_LOCAL_DEBUG_PROFILE,
        "cross_gpu": NEWTON_CROSS_GPU_PROFILE,
    }
    profile = aliases.get(profile, profile)
    if profile not in NEWTON_REPLAY_PROFILES:
        supported = ", ".join(sorted(NEWTON_REPLAY_PROFILES))
        raise ValueError(
            f"Unsupported Newton replay profile {value!r}; expected {supported}."
        )
    return profile


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _numeric_version(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value)[:4])


def _version_at_least(value: str, minimum: tuple[int, ...]) -> bool:
    numeric = _numeric_version(value)
    if not numeric:
        return False
    padded = numeric + (0,) * max(0, len(minimum) - len(numeric))
    return padded[: len(minimum)] >= minimum


def _deterministic_mode_name(warp: Any) -> str:
    value = getattr(getattr(warp, "config", None), "deterministic", None)
    enum_type = getattr(warp, "DeterministicMode", None)
    for name in ("GPU_TO_GPU", "RUN_TO_RUN", "NOT_GUARANTEED"):
        candidate = getattr(enum_type, name, object())
        if value == candidate:
            return name.lower()
    return "not_guaranteed" if value in {None, 0, False} else str(value)


@lru_cache(maxsize=1)
def _nvidia_driver_version() -> str:
    """Return the exact NVIDIA driver release, not only its CUDA ABI level."""

    try:
        value = Path("/proc/driver/nvidia/version").read_text(
            encoding="utf-8"
        )
        match = re.search(
            r"NVRM version:.*?\s(\d+\.\d+(?:\.\d+)?)\s+Release Build",
            value,
        )
        if match is not None:
            return match.group(1)
    except (OSError, UnicodeError):
        pass
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        versions = {
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        }
        if result.returncode == 0 and len(versions) == 1:
            return versions.pop()
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


def configure_newton_determinism(
    profile: str | None = None,
) -> dict[str, Any]:
    """Configure Warp before Newton kernels load and return the live contract.

    ``local_debug_v1`` deliberately makes no portability claim, but it still
    executes the exact same ``GPU_TO_GPU`` solver mode as production so a
    preview cannot silently exercise different physics.  Cross-model
    portability is granted only by the separate three-run replay acceptance
    gate.
    """

    normalized = normalize_newton_replay_profile(profile)
    versions = {
        name: _package_version(name) for name in _RUNTIME_PACKAGES
    }
    contract: dict[str, Any] = {
        "profile": normalized,
        "cross_gpu_requested": normalized == NEWTON_CROSS_GPU_PROFILE,
        "cross_gpu_ready": False,
        "warp_deterministic_mode": "unconfigured",
        "solver_deterministic_mode": "unconfigured",
        "warp_deterministic_max_records": 0,
        "bitwise_cross_gpu_claim": False,
        "cross_gpu_method": NEWTON_CROSS_GPU_METHOD,
        "collision_pipeline_deterministic_required": True,
        "versions": versions,
    }
    version_errors = [
        f"{name}>={'.'.join(str(part) for part in minimum)} "
        f"(found {versions[name]})"
        for name, minimum in _CROSS_GPU_MINIMUM_VERSIONS.items()
        if not _version_at_least(versions[name], minimum)
    ]
    if version_errors:
        raise NewtonCrossGpuUnavailable(
            "Newton GPU_TO_GPU replay requires the qualified minimum stack: "
            + "; ".join(version_errors)
            + ". Upgrade the Isaac Sim 6 Newton toolchain; falling back to "
            "non-deterministic GPU atomics is forbidden."
        )

    warp = importlib.import_module("warp")
    enum_type = getattr(warp, "DeterministicMode", None)
    gpu_to_gpu = getattr(enum_type, "GPU_TO_GPU", None)
    if gpu_to_gpu is None:
        raise NewtonCrossGpuUnavailable(
            "The selected Warp runtime does not expose "
            "DeterministicMode.GPU_TO_GPU; warp-lang>=1.15.0 is required."
        )
    warp.config.deterministic = gpu_to_gpu
    raw_max_records = os.environ.get(
        NEWTON_DETERMINISTIC_MAX_RECORDS_ENV,
        str(NEWTON_DETERMINISTIC_MAX_RECORDS_DEFAULT),
    )
    try:
        max_records = int(raw_max_records)
    except ValueError as exc:
        raise NewtonCrossGpuUnavailable(
            f"{NEWTON_DETERMINISTIC_MAX_RECORDS_ENV} must be an integer; "
            f"got {raw_max_records!r}."
        ) from exc
    if not 1 <= max_records <= 1024:
        raise NewtonCrossGpuUnavailable(
            f"{NEWTON_DETERMINISTIC_MAX_RECORDS_ENV} must be in [1, 1024]; "
            f"got {max_records}."
        )
    # Newton's convex manifold kernel emits up to four dynamic contact-counter
    # records from one thread.  Warp cannot prove that loop bound statically,
    # so the generated lower bound alone overflows on A01/B03.  Eight leaves a
    # small safety margin without the massive buffers caused by old 8K tests.
    warp.config.deterministic_max_records = max_records
    mode_name = _deterministic_mode_name(warp)
    if mode_name != "gpu_to_gpu":
        raise NewtonCrossGpuUnavailable(
            "Failed to enable Warp DeterministicMode.GPU_TO_GPU; "
            f"the live mode is {mode_name!r}."
        )
    contract["warp_deterministic_mode"] = mode_name
    contract["warp_deterministic_max_records"] = max_records
    return contract


def capture_newton_runtime_identity(
    device: str = "cuda:0",
    *,
    determinism_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture software and physical GPU identity without shell parsing."""

    versions = {
        name: _package_version(name) for name in _RUNTIME_PACKAGES
    }
    identity: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "software_versions": versions,
    }
    try:
        identity["b03_physics_source_sha256"] = (
            b03_physics_source_fingerprint()
        )
    except (OSError, UnicodeError) as exc:
        identity["b03_physics_source_fingerprint_error"] = (
            f"{type(exc).__name__}: {exc}"
        )
    try:
        identity["newton_qd_projection"] = (
            qualify_newton_velocity_projection_runtime()
        )
    except (OSError, ImportError, NewtonCrossGpuUnavailable) as exc:
        identity["newton_qd_projection_error"] = (
            f"{type(exc).__name__}: {exc}"
        )
    try:
        warp = importlib.import_module("warp")
        warp.init()
        warp_device = warp.get_device(str(device))
        arch = getattr(warp_device, "arch", None)
        driver_version = warp.get_cuda_driver_version()
        toolkit_version = warp.get_cuda_toolkit_version()
        identity.update(
            {
                "gpu_name": str(getattr(warp_device, "name", "")),
                "gpu_arch": (
                    f"sm_{int(arch)}" if arch is not None else ""
                ),
                "gpu_uuid": str(getattr(warp_device, "uuid", "")),
                "gpu_pci_bus_id": str(
                    getattr(warp_device, "pci_bus_id", "")
                ),
                "cuda_driver_version": (
                    [int(part) for part in driver_version]
                    if driver_version is not None
                    else None
                ),
                "cuda_toolkit_version": (
                    [int(part) for part in toolkit_version]
                    if toolkit_version is not None
                    else None
                ),
                "nvidia_driver_version": _nvidia_driver_version(),
                "warp_deterministic_mode": _deterministic_mode_name(warp),
                "warp_deterministic_max_records": int(
                    getattr(
                        warp.config,
                        "deterministic_max_records",
                        0,
                    )
                ),
            }
        )
    except Exception as exc:
        identity["runtime_identity_error"] = f"{type(exc).__name__}: {exc}"
    if determinism_contract is not None:
        identity["determinism_profile"] = str(
            determinism_contract.get("profile", "")
        )
        identity["cross_gpu_ready"] = bool(
            determinism_contract.get("cross_gpu_ready", False)
        )
        identity["solver_deterministic_mode"] = str(
            determinism_contract.get("solver_deterministic_mode", "")
        )
        identity["collision_pipeline_deterministic"] = bool(
            determinism_contract.get(
                "collision_pipeline_deterministic",
                False,
            )
        )
        identity["collision_broad_phase"] = str(
            determinism_contract.get("collision_broad_phase", "")
        )
        identity["cross_gpu_method"] = str(
            determinism_contract.get("cross_gpu_method", "")
        )
    return identity


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and re.fullmatch(r"[0-9a-f]{64}", value)
    )


def _asset_tree_identity_errors(value: Any, *, label: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return [f"{label} is missing"]
    if not _is_sha256(value.get("combined_sha256")):
        errors.append(f"{label}.combined_sha256 is invalid")
    file_count = value.get("file_count")
    files = value.get("files")
    if type(file_count) is not int or file_count <= 0:
        errors.append(f"{label}.file_count is invalid")
    if not isinstance(files, list) or len(files) != file_count:
        errors.append(f"{label}.files is incomplete")
    elif any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("path"), str)
        or not row.get("path")
        or not _is_sha256(row.get("sha256"))
        for row in files
    ):
        errors.append(f"{label}.files contains an invalid entry")
    return errors


def newton_runtime_identity_errors(
    identity: Mapping[str, Any],
    *,
    label: str = "Newton runtime identity",
) -> list[str]:
    """Validate the complete identity required by B03 certification."""

    required_fields = (
        "python",
        "platform",
        "software_versions",
        "b03_physics_source_sha256",
        "newton_qd_projection",
        "b03_generated_robot_assets",
        "gpu_name",
        "gpu_arch",
        "cuda_driver_version",
        "cuda_toolkit_version",
        "nvidia_driver_version",
    )
    errors = [
        f"{label}.{field} is missing"
        for field in required_fields
        if not identity.get(field)
    ]
    for field in (
        "runtime_identity_error",
        "b03_physics_source_fingerprint_error",
        "newton_qd_projection_error",
    ):
        if identity.get(field):
            errors.append(f"{label}.{field}: {identity[field]}")
    if not _is_sha256(identity.get("b03_physics_source_sha256")):
        errors.append(f"{label}.b03_physics_source_sha256 is invalid")

    projection = identity.get("newton_qd_projection")
    if not isinstance(projection, Mapping):
        errors.append(f"{label}.newton_qd_projection is invalid")
    else:
        if not _is_sha256(projection.get("combined_sha256")):
            errors.append(
                f"{label}.newton_qd_projection.combined_sha256 is invalid"
            )
        module_hashes = projection.get("module_sha256")
        if not isinstance(module_hashes, Mapping) or set(module_hashes) != set(
            _NEWTON_QD_PROJECTION_MODULES
        ):
            errors.append(
                f"{label}.newton_qd_projection.module_sha256 is incomplete"
            )
        elif any(not _is_sha256(value) for value in module_hashes.values()):
            errors.append(
                f"{label}.newton_qd_projection.module_sha256 is invalid"
            )

    assets = identity.get("b03_generated_robot_assets")
    if not isinstance(assets, Mapping) or set(assets) != {
        "left",
        "right",
        "task",
    }:
        errors.append(f"{label}.b03_generated_robot_assets is incomplete")
    else:
        for side in ("left", "right"):
            asset = assets.get(side)
            asset_label = f"{label}.b03_generated_robot_assets.{side}"
            if not isinstance(asset, Mapping):
                errors.append(f"{asset_label} is invalid")
                continue
            if asset.get("format") != "handson.b03_generated_robot_asset":
                errors.append(f"{asset_label}.format is invalid")
            if asset.get("version") != 3:
                errors.append(f"{asset_label}.version must be 3")
            if not _is_sha256(asset.get("generated_mjcf_sha256")):
                errors.append(f"{asset_label}.generated_mjcf_sha256 is invalid")
            errors.extend(
                _asset_tree_identity_errors(
                    asset.get("referenced_assets"),
                    label=f"{asset_label}.referenced_assets",
                )
            )
            errors.extend(
                _asset_tree_identity_errors(
                    asset.get("loaded_usd_cache"),
                    label=f"{asset_label}.loaded_usd_cache",
                )
            )
        task_asset = assets.get("task")
        task_label = f"{label}.b03_generated_robot_assets.task"
        if not isinstance(task_asset, Mapping):
            errors.append(f"{task_label} is invalid")
        else:
            if task_asset.get("format") != "handson.b03_loaded_task_asset":
                errors.append(f"{task_label}.format is invalid")
            if task_asset.get("version") != 2:
                errors.append(f"{task_label}.version must be 2")
            if not _is_sha256(task_asset.get("generated_mjcf_sha256")):
                errors.append(
                    f"{task_label}.generated_mjcf_sha256 is invalid"
                )
            if not _is_sha256(
                task_asset.get("canonical_generated_mjcf_sha256")
            ):
                errors.append(
                    f"{task_label}.canonical_generated_mjcf_sha256 is invalid"
                )
            errors.extend(
                _asset_tree_identity_errors(
                    task_asset.get("referenced_assets"),
                    label=f"{task_label}.referenced_assets",
                )
            )
            errors.extend(
                _asset_tree_identity_errors(
                    task_asset.get("loaded_usd_cache"),
                    label=f"{task_label}.loaded_usd_cache",
                )
            )
    return errors


def gpu_model_key(identity: Mapping[str, Any]) -> tuple[str, str]:
    """Return a model-level key; UUID intentionally does not distinguish cards."""

    return (
        str(identity.get("gpu_name") or "").strip().casefold(),
        str(identity.get("gpu_arch") or "").strip().casefold(),
    )


def is_distinct_gpu_model(
    source: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> bool:
    """Return whether source and replay use genuinely different GPU models."""

    source_key = gpu_model_key(source)
    replay_key = gpu_model_key(replay)
    return bool(
        all(source_key)
        and all(replay_key)
        and source_key != replay_key
    )


def software_identity(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Project semantic physics inputs that must match across GPU models.

    GPU/card and OS platform metadata is deliberately excluded.  Those
    differences are recorded separately and accepted only after the full
    trajectory stays within its fixed physical tolerances.  NVIDIA/CUDA
    driver identities remain strict inputs because the cross-host run is
    qualified with matching drivers.
    """

    return {
        "python": identity.get("python"),
        "software_versions": identity.get("software_versions"),
        "cuda_driver_version": identity.get("cuda_driver_version"),
        "cuda_toolkit_version": identity.get("cuda_toolkit_version"),
        "nvidia_driver_version": identity.get("nvidia_driver_version"),
        "b03_physics_source_sha256": identity.get(
            "b03_physics_source_sha256"
        ),
        "newton_qd_projection": identity.get("newton_qd_projection"),
        "b03_generated_robot_assets": identity.get(
            "b03_generated_robot_assets"
        ),
        "determinism_profile": identity.get("determinism_profile"),
        "warp_deterministic_mode": identity.get("warp_deterministic_mode"),
        "warp_deterministic_max_records": identity.get(
            "warp_deterministic_max_records"
        ),
        "solver_deterministic_mode": identity.get(
            "solver_deterministic_mode"
        ),
        "collision_pipeline_deterministic": identity.get(
            "collision_pipeline_deterministic"
        ),
        "collision_broad_phase": identity.get("collision_broad_phase"),
        "cross_gpu_method": identity.get("cross_gpu_method"),
        "cross_gpu_ready": identity.get("cross_gpu_ready"),
    }


def runtime_environment_identity(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Project recorded host/card metadata for audit or same-host checks."""

    return {
        field: identity.get(field)
        for field in _RUNTIME_ENVIRONMENT_IDENTITY_FIELDS
    }


def runtime_environment_differences(
    source: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return non-blocking cross-GPU environment differences."""

    source_environment = runtime_environment_identity(source)
    replay_environment = runtime_environment_identity(replay)
    return {
        field: {
            "source": source_environment[field],
            "replay": replay_environment[field],
        }
        for field in _RUNTIME_ENVIRONMENT_IDENTITY_FIELDS
        if source_environment[field] != replay_environment[field]
    }


__all__ = [
    "NEWTON_CROSS_GPU_PROFILE",
    "NEWTON_DETERMINISTIC_MAX_RECORDS_DEFAULT",
    "NEWTON_DETERMINISTIC_MAX_RECORDS_ENV",
    "NEWTON_LOCAL_DEBUG_PROFILE",
    "NEWTON_REPLAY_PROFILE_ENV",
    "NewtonCrossGpuUnavailable",
    "b03_physics_source_fingerprint",
    "capture_newton_runtime_identity",
    "configure_newton_determinism",
    "gpu_model_key",
    "is_distinct_gpu_model",
    "newton_runtime_identity_errors",
    "normalize_newton_replay_profile",
    "qualify_newton_velocity_projection_runtime",
    "runtime_environment_differences",
    "runtime_environment_identity",
    "software_identity",
]
