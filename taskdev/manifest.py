"""Canonical task-manifest validation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import REQUIRED_CHECKS, TASK_ID_PATTERN
from .io import TaskDevError, load_yaml, resolve_inside


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class ValidationResult:
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "error_count": len(self.issues),
            "errors": [issue.as_dict() for issue in self.issues],
        }


_MISSING = object()
_TYPE_NAMES = {
    "mapping": dict,
    "sequence": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


def _get_path(document: Any, dotted_path: str) -> Any:
    value = document
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _is_expected_type(value: Any, expected: str) -> bool:
    expected_type = _TYPE_NAMES[expected]
    if expected in {"integer", "number"} and isinstance(value, bool):
        return False
    return isinstance(value, expected_type)


class ManifestValidator:
    """Validate structure, assets, hardware contract, and acceptance thresholds."""

    def __init__(self, repository_root: Path, schema_path: Path | None = None) -> None:
        self.repository_root = repository_root.resolve()
        self.schema_path = schema_path or (
            Path(__file__).resolve().parent / "schema" / "task_manifest_v1.schema.yaml"
        )
        schema = load_yaml(self.schema_path)
        if not isinstance(schema, dict):
            raise TaskDevError(f"schema root must be a mapping: {self.schema_path}")
        self.schema = schema

    def validate_path(self, manifest_path: Path) -> ValidationResult:
        document = load_yaml(manifest_path)
        return self.validate(document, manifest_path=manifest_path)

    def validate(self, document: Any, *, manifest_path: Path | None = None) -> ValidationResult:
        issues: list[ValidationIssue] = []
        if not isinstance(document, dict):
            return ValidationResult(
                (ValidationIssue("$", "type", "manifest root must be a mapping"),)
            )
        self._validate_schema_rules(document, issues)
        self._validate_task(document, issues)
        self._validate_hardware(document, issues)
        self._validate_workspace(document, issues)
        self._validate_assets(document, issues)
        self._validate_reset_and_contact(document, issues)
        self._validate_success(document, issues)
        self._validate_camera_stability_collection(document, issues)
        self._validate_acceptance(document, issues)
        return ValidationResult(tuple(issues))

    @staticmethod
    def _issue(issues: list[ValidationIssue], path: str, code: str, message: str) -> None:
        issues.append(ValidationIssue(path, code, message))

    def _validate_schema_rules(self, document: dict[str, Any], issues: list[ValidationIssue]) -> None:
        if self.schema.get("strict_top_level"):
            allowed = set(self.schema.get("allowed_top_level", []))
            for key in sorted(set(document) - allowed):
                self._issue(issues, key, "unknown_field", "unknown top-level field")

        for path, expected in self.schema.get("required_paths", {}).items():
            value = _get_path(document, path)
            if value is _MISSING:
                self._issue(issues, path, "required", "required field is missing")
            elif expected not in _TYPE_NAMES or not _is_expected_type(value, expected):
                self._issue(issues, path, "type", f"expected {expected}")

        if document.get("schema_version") != self.schema.get("schema_version"):
            self._issue(issues, "schema_version", "unsupported", "only schema_version 1 is supported")

        for path, allowed in self.schema.get("enums", {}).items():
            value = _get_path(document, path)
            if value is not _MISSING and value not in allowed:
                self._issue(issues, path, "enum", f"must be one of {allowed}")

        for path, bounds in self.schema.get("numeric_ranges", {}).items():
            value = _get_path(document, path)
            if value is _MISSING or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if "min" in bounds and value < bounds["min"]:
                self._issue(issues, path, "range", f"must be >= {bounds['min']}")
            if "max" in bounds and value > bounds["max"]:
                self._issue(issues, path, "range", f"must be <= {bounds['max']}")
            if "exclusive_min" in bounds and value <= bounds["exclusive_min"]:
                self._issue(issues, path, "range", f"must be > {bounds['exclusive_min']}")

    def _validate_task(self, document: dict[str, Any], issues: list[ValidationIssue]) -> None:
        task = document.get("task")
        if not isinstance(task, dict):
            return
        task_id = task.get("id")
        if isinstance(task_id, str) and not re.fullmatch(TASK_ID_PATTERN, task_id):
            self._issue(issues, "task.id", "format", "must be lower snake_case")
        interactions = _get_path(document, "task.taxonomy.interactions")
        if isinstance(interactions, list):
            all_strings = bool(interactions) and all(isinstance(item, str) and item for item in interactions)
            if not all_strings:
                self._issue(issues, "task.taxonomy.interactions", "content", "requires non-empty strings")
            if all_strings and len(set(interactions)) != len(interactions):
                self._issue(issues, "task.taxonomy.interactions", "duplicate", "interactions must be unique")
        source = document.get("source")
        if isinstance(source, dict):
            origin = source.get("origin")
            if isinstance(origin, str) and origin in {"adapted", "handson"} and not source.get("reference"):
                self._issue(issues, "source.reference", "required", "adapted/handson tasks require a source reference")
            if origin == "handson" and not source.get("license"):
                self._issue(issues, "source.license", "required", "DexJoco-origin tasks require license provenance")

    def _validate_hardware(self, document: dict[str, Any], issues: list[ValidationIssue]) -> None:
        arms = _get_path(document, "robot.arms")
        hands = _get_path(document, "robot.hands")
        for path, block in (("robot.arms", arms), ("robot.hands", hands)):
            if not isinstance(block, dict):
                continue
            asset = block.get("asset")
            if isinstance(asset, str):
                self._check_file(asset, f"{path}.asset", issues)
        if isinstance(arms, dict):
            count = arms.get("count")
            poses = arms.get("base_poses")
            if isinstance(count, int) and isinstance(poses, list) and len(poses) != count:
                self._issue(issues, "robot.arms.base_poses", "length", "must contain one pose per arm")
            if isinstance(poses, list):
                for index, pose in enumerate(poses):
                    self._validate_pose(pose, f"robot.arms.base_poses[{index}]", issues)
        if isinstance(hands, dict) and hands.get("generation") != 1:
            self._issue(issues, "robot.hands.generation", "hardware", "only Wuji generation 1 is accepted")
        handedness = _get_path(document, "task.taxonomy.handedness")
        if handedness == "bimanual":
            if not isinstance(arms, dict) or arms.get("count") != 2:
                self._issue(issues, "robot.arms.count", "hardware", "bimanual tasks require two ROKAE arms")
            if not isinstance(hands, dict) or hands.get("count") != 2:
                self._issue(issues, "robot.hands.count", "hardware", "bimanual tasks require two Wuji gen1 hands")

    def _validate_workspace(self, document: dict[str, Any], issues: list[ValidationIssue]) -> None:
        bounds = _get_path(document, "workspace.bounds_m")
        if isinstance(bounds, dict):
            minimum, maximum = bounds.get("min"), bounds.get("max")
            if self._vector(minimum, 3) and self._vector(maximum, 3):
                if any(low >= high for low, high in zip(minimum, maximum)):
                    self._issue(issues, "workspace.bounds_m", "range", "every min component must be below max")
            else:
                self._issue(issues, "workspace.bounds_m", "shape", "min/max must be numeric XYZ vectors")
        targets = _get_path(document, "workspace.targets")
        if isinstance(targets, list):
            if not targets:
                self._issue(issues, "workspace.targets", "empty", "at least one reachability target is required")
            names: list[str] = []
            for index, target in enumerate(targets):
                path = f"workspace.targets[{index}]"
                if not isinstance(target, dict):
                    self._issue(issues, path, "type", "target must be a mapping")
                    continue
                for key in ("id", "arm", "frame", "position_m", "orientation_xyzw", "clearance_m"):
                    if key not in target:
                        self._issue(issues, f"{path}.{key}", "required", "required field is missing")
                if isinstance(target.get("id"), str):
                    names.append(target["id"])
                arm = target.get("arm")
                if not isinstance(arm, str) or arm not in {"left", "right", "either"}:
                    self._issue(issues, f"{path}.arm", "enum", "must be left, right, or either")
                if not self._vector(target.get("position_m"), 3):
                    self._issue(issues, f"{path}.position_m", "shape", "must be a numeric XYZ vector")
                elif (
                    isinstance(bounds, dict)
                    and self._vector(bounds.get("min"), 3)
                    and self._vector(bounds.get("max"), 3)
                    and target.get("frame") == _get_path(document, "workspace.frame")
                    and any(
                        coordinate < low or coordinate > high
                        for coordinate, low, high in zip(
                            target["position_m"], bounds["min"], bounds["max"]
                        )
                    )
                ):
                    self._issue(issues, f"{path}.position_m", "workspace", "target is outside workspace bounds")
                if not self._vector(target.get("orientation_xyzw"), 4):
                    self._issue(issues, f"{path}.orientation_xyzw", "shape", "must be a numeric quaternion")
                elif not self._unit_quaternion(target["orientation_xyzw"]):
                    self._issue(issues, f"{path}.orientation_xyzw", "quaternion", "must be normalized")
                clearance = target.get("clearance_m")
                if isinstance(clearance, bool) or not isinstance(clearance, (int, float)) or clearance < 0:
                    self._issue(issues, f"{path}.clearance_m", "range", "must be >= 0")
            if len(set(names)) != len(names):
                self._issue(issues, "workspace.targets", "duplicate", "target ids must be unique")

    def _validate_assets(self, document: dict[str, Any], issues: list[ValidationIssue]) -> None:
        assets = document.get("assets")
        if not isinstance(assets, list):
            return
        if not assets:
            self._issue(issues, "assets", "empty", "at least one task asset is required")
        elif not any(isinstance(asset, dict) and asset.get("required") is True for asset in assets):
            self._issue(issues, "assets", "required", "at least one asset must be required")
        ids: list[str] = []
        for index, asset in enumerate(assets):
            path = f"assets[{index}]"
            if not isinstance(asset, dict):
                self._issue(issues, path, "type", "asset must be a mapping")
                continue
            for key in ("id", "path", "kind", "required", "scale_m", "collision"):
                if key not in asset:
                    self._issue(issues, f"{path}.{key}", "required", "required field is missing")
            if isinstance(asset.get("id"), str):
                ids.append(asset["id"])
            asset_path = asset.get("path")
            if isinstance(asset_path, str) and asset.get("required") is True:
                self._check_file(asset_path, f"{path}.path", issues)
            kind = asset.get("kind")
            if not isinstance(kind, str) or kind not in {"mjcf", "urdf", "usd", "mesh", "texture", "config", "other"}:
                self._issue(issues, f"{path}.kind", "enum", "unsupported asset kind")
            if not isinstance(asset.get("required"), bool):
                self._issue(issues, f"{path}.required", "type", "must be boolean")
            if not self._vector(asset.get("scale_m"), 3, positive=True):
                self._issue(issues, f"{path}.scale_m", "shape", "must be a positive XYZ scale")
            collision = asset.get("collision")
            if not isinstance(collision, dict) or not isinstance(collision.get("enabled"), bool):
                self._issue(issues, f"{path}.collision.enabled", "required", "must be boolean")
        if len(set(ids)) != len(ids):
            self._issue(issues, "assets", "duplicate", "asset ids must be unique")

    def _validate_reset_and_contact(self, document: dict[str, Any], issues: list[ValidationIssue]) -> None:
        reset = document.get("reset")
        if isinstance(reset, dict):
            if reset.get("seedable") is not True:
                self._issue(issues, "reset.seedable", "value", "reset must be seedable")
            invariants = reset.get("invariants")
            expected = {
                "all_required_assets_present",
                "finite_actor_state",
                "no_initial_interpenetration_above_threshold",
                "targets_inside_workspace",
            }
            if not isinstance(invariants, list) or not all(isinstance(item, str) for item in invariants) or not expected.issubset(set(invariants)):
                self._issue(issues, "reset.invariants", "content", f"must include {sorted(expected)}")
        contact = document.get("contact")
        if not isinstance(contact, dict):
            return
        monitored = contact.get("monitored_pairs")
        if isinstance(monitored, list):
            if not monitored:
                self._issue(issues, "contact.monitored_pairs", "empty", "at least one contact pair is required")
            for index, pair in enumerate(monitored):
                self._validate_contact_pair(pair, f"contact.monitored_pairs[{index}]", issues, required=True)
        forbidden = contact.get("forbidden_pairs")
        if isinstance(forbidden, list):
            for index, pair in enumerate(forbidden):
                self._validate_contact_pair(pair, f"contact.forbidden_pairs[{index}]", issues, required=False)

    def _validate_success(self, document: dict[str, Any], issues: list[ValidationIssue]) -> None:
        success = document.get("success")
        if not isinstance(success, dict):
            return
        predicates = success.get("predicates")
        if isinstance(predicates, list):
            if not predicates:
                self._issue(issues, "success.predicates", "empty", "at least one predicate is required")
            names: list[str] = []
            for index, predicate in enumerate(predicates):
                path = f"success.predicates[{index}]"
                if not isinstance(predicate, dict):
                    self._issue(issues, path, "type", "predicate must be a mapping")
                    continue
                for key in ("id", "type", "parameters"):
                    if key not in predicate:
                        self._issue(issues, f"{path}.{key}", "required", "required field is missing")
                if isinstance(predicate.get("id"), str):
                    names.append(predicate["id"])
                predicate_type = predicate.get("type")
                if not isinstance(predicate_type, str) or not predicate_type.strip():
                    self._issue(issues, f"{path}.type", "content", "must be a non-empty string")
                if not isinstance(predicate.get("parameters"), dict) or not predicate.get("parameters"):
                    self._issue(issues, f"{path}.parameters", "empty", "parameters must be non-empty")
            if len(set(names)) != len(names):
                self._issue(issues, "success.predicates", "duplicate", "predicate ids must be unique")
        hold, timeout = success.get("hold_steps"), success.get("timeout_steps")
        if isinstance(hold, int) and isinstance(timeout, int) and hold >= timeout:
            self._issue(issues, "success.hold_steps", "range", "must be below timeout_steps")

    def _validate_camera_stability_collection(self, document: dict[str, Any], issues: list[ValidationIssue]) -> None:
        camera = document.get("camera")
        if isinstance(camera, dict):
            if camera.get("primary") != "head":
                self._issue(issues, "camera.primary", "camera", "primary camera must be the robot head camera")
            if camera.get("require_rgb") is not True:
                self._issue(issues, "camera.require_rgb", "camera", "head RGB is required")
            resolution = camera.get("resolution")
            if not self._integer_vector(resolution, 2) or resolution[0] < 640 or resolution[1] < 480:
                self._issue(issues, "camera.resolution", "range", "minimum resolution is 640x480")
            prim_path = camera.get("prim_path")
            if not isinstance(prim_path, str) or not prim_path.startswith("/World/") or "head" not in prim_path.lower():
                self._issue(issues, "camera.prim_path", "camera", "must be an absolute /World head-camera prim")
        stability = document.get("stability")
        if isinstance(stability, dict):
            physics_hz, control_hz = stability.get("physics_hz"), stability.get("control_hz")
            if isinstance(physics_hz, int) and isinstance(control_hz, int) and physics_hz % control_hz:
                self._issue(issues, "stability.control_hz", "ratio", "physics_hz must be divisible by control_hz")
        collection = document.get("collection")
        if isinstance(collection, dict):
            output_variable = collection.get("output_env_var")
            if not isinstance(output_variable, str) or not re.fullmatch(r"^[A-Z][A-Z0-9_]*$", output_variable):
                self._issue(issues, "collection.output_env_var", "format", "must be an uppercase environment variable")
            streams = collection.get("required_streams")
            minimum = {"action", "proprioception", "rgb_head", "contacts", "success"}
            if isinstance(streams, list) and (
                not all(isinstance(item, str) for item in streams)
                or not minimum.issubset(set(streams))
            ):
                self._issue(issues, "collection.required_streams", "content", f"must include {sorted(minimum)}")

    def _validate_acceptance(self, document: dict[str, Any], issues: list[ValidationIssue]) -> None:
        required = _get_path(document, "validation.required_checks")
        if isinstance(required, list):
            if not all(isinstance(item, str) for item in required):
                self._issue(issues, "validation.required_checks", "type", "check names must be strings")
                normalized: set[str] = set()
            else:
                normalized = set(required)
            if len(required) != len(normalized):
                self._issue(issues, "validation.required_checks", "duplicate", "checks must be unique")
            if normalized != set(REQUIRED_CHECKS):
                self._issue(issues, "validation.required_checks", "content", f"must contain exactly {list(REQUIRED_CHECKS)}")
        report = _get_path(document, "validation.report")
        if isinstance(report, str):
            try:
                resolve_inside(self.repository_root, report, label="validation.report")
            except TaskDevError as exc:
                self._issue(issues, "validation.report", "path", str(exc))
        thresholds = _get_path(document, "validation.thresholds")
        if not isinstance(thresholds, dict):
            return
        for check in REQUIRED_CHECKS:
            if not isinstance(thresholds.get(check), dict):
                self._issue(issues, f"validation.thresholds.{check}", "required", "threshold mapping is required")
        required_metrics = {
            "schema": ("max_errors",),
            "assets": ("max_missing",),
            "reachability": (
                "min_target_fraction",
                "min_joint_limit_margin_rad",
                "max_position_error_m",
                "max_orientation_error_rad",
            ),
            "contact": ("max_penetration_m", "max_forbidden_events"),
            "reset": ("min_trials", "max_failure_fraction"),
            "success": ("min_positive_trials", "min_negative_trials", "max_false_positive_fraction", "max_false_negative_fraction"),
            "camera": ("min_frames", "max_black_fraction", "min_mean_rgb", "min_width", "min_height"),
            "stability": ("min_sim_seconds", "max_nan_count", "max_exploding_body_count"),
            "collection": ("min_episodes", "max_schema_errors"),
        }
        for check, names in required_metrics.items():
            block = thresholds.get(check)
            if not isinstance(block, dict):
                continue
            for name in names:
                value = block.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    self._issue(issues, f"validation.thresholds.{check}.{name}", "required", "numeric threshold is required")
                elif not math.isfinite(float(value)) or value < 0:
                    self._issue(issues, f"validation.thresholds.{check}.{name}", "range", "threshold must be finite and >= 0")
                elif "fraction" in name and value > 1:
                    self._issue(issues, f"validation.thresholds.{check}.{name}", "range", "fraction threshold must be <= 1")

    def _check_file(self, candidate: str, path: str, issues: list[ValidationIssue]) -> None:
        try:
            resolved = resolve_inside(self.repository_root, candidate, label=path)
        except TaskDevError as exc:
            self._issue(issues, path, "path", str(exc))
            return
        if not resolved.is_file():
            self._issue(issues, path, "missing_asset", f"file does not exist: {candidate}")

    @staticmethod
    def _validate_pose(value: Any, path: str, issues: list[ValidationIssue]) -> None:
        if not isinstance(value, dict):
            issues.append(ValidationIssue(path, "type", "pose must be a mapping"))
            return
        if not ManifestValidator._vector(value.get("position_m"), 3):
            issues.append(ValidationIssue(f"{path}.position_m", "shape", "must be a numeric XYZ vector"))
        if not ManifestValidator._vector(value.get("orientation_xyzw"), 4):
            issues.append(ValidationIssue(f"{path}.orientation_xyzw", "shape", "must be a numeric quaternion"))
        elif not ManifestValidator._unit_quaternion(value["orientation_xyzw"]):
            issues.append(ValidationIssue(f"{path}.orientation_xyzw", "quaternion", "must be normalized"))

    @staticmethod
    def _validate_contact_pair(value: Any, path: str, issues: list[ValidationIssue], *, required: bool) -> None:
        if not isinstance(value, dict):
            issues.append(ValidationIssue(path, "type", "contact pair must be a mapping"))
            return
        bodies = value.get("bodies")
        if not isinstance(bodies, list) or len(bodies) != 2 or any(not isinstance(item, str) or not item for item in bodies):
            issues.append(ValidationIssue(f"{path}.bodies", "shape", "must contain two body selectors"))
        if required:
            penetration = value.get("max_penetration_m")
            if isinstance(penetration, bool) or not isinstance(penetration, (int, float)) or penetration < 0:
                issues.append(ValidationIssue(f"{path}.max_penetration_m", "range", "must be >= 0"))

    @staticmethod
    def _vector(value: Any, length: int, *, positive: bool = False) -> bool:
        if not isinstance(value, list) or len(value) != length:
            return False
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return False
            if not math.isfinite(float(item)):
                return False
            if positive and item <= 0:
                return False
        return True

    @staticmethod
    def _integer_vector(value: Any, length: int) -> bool:
        return isinstance(value, list) and len(value) == length and all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        )

    @staticmethod
    def _unit_quaternion(value: list[float]) -> bool:
        squared_norm = sum(float(item) ** 2 for item in value)
        return math.isfinite(squared_norm) and abs(squared_norm - 1.0) <= 1e-3
