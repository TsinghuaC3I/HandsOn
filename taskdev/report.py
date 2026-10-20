"""Fail-closed validation-report gate."""

from __future__ import annotations

import datetime as dt
import math
import re
from pathlib import Path
from typing import Any

from .constants import REQUIRED_CHECKS, SHA256_PATTERN
from .io import TaskDevError, load_yaml, resolve_inside, sha256_file
from .manifest import ManifestValidator, ValidationIssue, ValidationResult


class ReportGate:
    """Recompute task acceptance from evidence and category metrics."""

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root.resolve()
        self.manifest_validator = ManifestValidator(self.repository_root)

    def evaluate_paths(self, manifest_path: Path, report_path: Path) -> ValidationResult:
        manifest = load_yaml(manifest_path)
        report = load_yaml(report_path)
        return self.evaluate(
            manifest,
            report,
            manifest_path=manifest_path,
        )

    def evaluate(
        self,
        manifest: Any,
        report: Any,
        *,
        manifest_path: Path,
    ) -> ValidationResult:
        issues = list(self.manifest_validator.validate(manifest, manifest_path=manifest_path).issues)
        if not isinstance(manifest, dict):
            return ValidationResult(tuple(issues))
        if not isinstance(report, dict):
            issues.append(ValidationIssue("report", "type", "report root must be a mapping"))
            return ValidationResult(tuple(issues))

        self._validate_header(manifest, report, manifest_path, issues)
        checks = report.get("checks")
        if not isinstance(checks, dict):
            issues.append(ValidationIssue("checks", "required", "checks must be a mapping"))
            return ValidationResult(tuple(issues))
        unknown = sorted(set(checks) - set(REQUIRED_CHECKS))
        if unknown:
            issues.append(ValidationIssue("checks", "unknown", f"unknown checks: {unknown}"))
        validation = manifest.get("validation")
        thresholds = validation.get("thresholds", {}) if isinstance(validation, dict) else {}
        for name in REQUIRED_CHECKS:
            check = checks.get(name)
            path = f"checks.{name}"
            if not isinstance(check, dict):
                issues.append(ValidationIssue(path, "required", "check result is missing"))
                continue
            if check.get("status") != "pass":
                issues.append(ValidationIssue(f"{path}.status", "gate", "status must be pass"))
            summary = check.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                issues.append(ValidationIssue(f"{path}.summary", "required", "non-empty summary is required"))
            self._validate_evidence(check.get("evidence"), f"{path}.evidence", issues)
            metrics = check.get("metrics")
            if not isinstance(metrics, dict):
                issues.append(ValidationIssue(f"{path}.metrics", "required", "metrics mapping is required"))
                continue
            threshold = thresholds.get(name) if isinstance(thresholds, dict) else None
            if not isinstance(threshold, dict):
                issues.append(ValidationIssue(f"validation.thresholds.{name}", "required", "threshold mapping is required"))
                continue
            self._evaluate_metrics(name, manifest, metrics, threshold, issues)
        return ValidationResult(tuple(issues))

    def _validate_header(
        self,
        manifest: dict[str, Any],
        report: dict[str, Any],
        manifest_path: Path,
        issues: list[ValidationIssue],
    ) -> None:
        if report.get("schema_version") != 1:
            issues.append(ValidationIssue("report.schema_version", "unsupported", "must be 1"))
        task = manifest.get("task")
        task_id = task.get("id") if isinstance(task, dict) else None
        if report.get("task_id") != task_id:
            issues.append(ValidationIssue("report.task_id", "mismatch", "must match manifest task.id"))
        expected_hash = sha256_file(manifest_path)
        report_hash = report.get("manifest_sha256")
        if not isinstance(report_hash, str) or not re.fullmatch(SHA256_PATTERN, report_hash):
            issues.append(ValidationIssue("report.manifest_sha256", "format", "must be lowercase SHA-256"))
        elif report_hash != expected_hash:
            issues.append(ValidationIssue("report.manifest_sha256", "mismatch", "manifest changed after validation"))
        simulator = report.get("simulator")
        if not isinstance(simulator, dict):
            issues.append(ValidationIssue("report.simulator", "required", "simulator mapping is required"))
        else:
            if simulator.get("name") != "isaac_sim":
                issues.append(ValidationIssue("report.simulator.name", "value", "must be isaac_sim"))
            if simulator.get("version") != "5.1":
                issues.append(ValidationIssue("report.simulator.version", "value", "this pipeline accepts Isaac Sim 5.1 only"))
        generated_at = report.get("generated_at")
        if not isinstance(generated_at, str):
            issues.append(ValidationIssue("report.generated_at", "required", "timezone-aware timestamp is required"))
        else:
            try:
                parsed = dt.datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    raise ValueError("missing timezone")
            except ValueError:
                issues.append(ValidationIssue("report.generated_at", "format", "must be an ISO-8601 timestamp with timezone"))

    def _validate_evidence(self, value: Any, path: str, issues: list[ValidationIssue]) -> None:
        if not isinstance(value, list) or not value:
            issues.append(ValidationIssue(path, "required", "at least one immutable evidence file is required"))
            return
        for index, evidence in enumerate(value):
            item_path = f"{path}[{index}]"
            if not isinstance(evidence, dict):
                issues.append(ValidationIssue(item_path, "type", "evidence must be a mapping"))
                continue
            candidate, expected_hash = evidence.get("path"), evidence.get("sha256")
            try:
                resolved = resolve_inside(self.repository_root, candidate, label=f"{item_path}.path")
            except TaskDevError as exc:
                issues.append(ValidationIssue(f"{item_path}.path", "path", str(exc)))
                continue
            if not resolved.is_file() or resolved.is_symlink():
                issues.append(ValidationIssue(f"{item_path}.path", "missing", "must be a regular, non-symlink file"))
                continue
            if not isinstance(expected_hash, str) or not re.fullmatch(SHA256_PATTERN, expected_hash):
                issues.append(ValidationIssue(f"{item_path}.sha256", "format", "must be lowercase SHA-256"))
            elif sha256_file(resolved) != expected_hash:
                issues.append(ValidationIssue(f"{item_path}.sha256", "mismatch", "evidence file hash mismatch"))

    def _evaluate_metrics(
        self,
        name: str,
        manifest: dict[str, Any],
        metrics: dict[str, Any],
        threshold: dict[str, Any],
        issues: list[ValidationIssue],
    ) -> None:
        path = f"checks.{name}.metrics"
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)) or value < 0:
                issues.append(ValidationIssue(f"{path}.{key}", "range", "numeric metrics must be finite and >= 0"))
            if isinstance(key, str) and "fraction" in key and value > 1:
                issues.append(ValidationIssue(f"{path}.{key}", "range", "fraction metrics must be <= 1"))
        if name == "schema":
            self._at_most(metrics, "error_count", threshold, "max_errors", path, issues)
        elif name == "assets":
            self._at_most(metrics, "missing_count", threshold, "max_missing", path, issues)
            self._boolean(metrics, "all_scales_verified", True, path, issues)
            self._boolean(metrics, "all_collision_shapes_verified", True, path, issues)
        elif name == "reachability":
            self._at_least(metrics, "target_fraction", threshold, "min_target_fraction", path, issues)
            self._at_least(metrics, "min_joint_limit_margin_rad", threshold, "min_joint_limit_margin_rad", path, issues)
            self._at_most(metrics, "max_position_error_m", threshold, "max_position_error_m", path, issues)
            self._at_most(metrics, "max_orientation_error_rad", threshold, "max_orientation_error_rad", path, issues)
            self._boolean(metrics, "rokae_model_verified", True, path, issues)
            self._boolean(metrics, "wuji_gen1_verified", True, path, issues)
        elif name == "contact":
            self._at_most(metrics, "max_penetration_m", threshold, "max_penetration_m", path, issues)
            self._at_most(metrics, "forbidden_events", threshold, "max_forbidden_events", path, issues)
            self._boolean(metrics, "required_pairs_observed", True, path, issues)
        elif name == "reset":
            self._at_least(metrics, "trials", threshold, "min_trials", path, issues)
            self._at_most(metrics, "failure_fraction", threshold, "max_failure_fraction", path, issues)
            self._boolean(metrics, "deterministic", True, path, issues)
        elif name == "success":
            self._at_least(metrics, "positive_trials", threshold, "min_positive_trials", path, issues)
            self._at_least(metrics, "negative_trials", threshold, "min_negative_trials", path, issues)
            self._at_most(metrics, "false_positive_fraction", threshold, "max_false_positive_fraction", path, issues)
            self._at_most(metrics, "false_negative_fraction", threshold, "max_false_negative_fraction", path, issues)
            success = manifest.get("success")
            expected_hold = success.get("hold_steps") if isinstance(success, dict) else None
            self._numeric_at_least_value(metrics, "hold_steps_observed", expected_hold, path, issues)
        elif name == "camera":
            self._at_least(metrics, "frames", threshold, "min_frames", path, issues)
            self._at_most(metrics, "black_fraction", threshold, "max_black_fraction", path, issues)
            self._at_least(metrics, "mean_rgb", threshold, "min_mean_rgb", path, issues)
            self._at_least(metrics, "width", threshold, "min_width", path, issues)
            self._at_least(metrics, "height", threshold, "min_height", path, issues)
            if metrics.get("primary") != "head":
                issues.append(ValidationIssue(f"{path}.primary", "threshold", "must be head"))
        elif name == "stability":
            self._at_least(metrics, "sim_seconds", threshold, "min_sim_seconds", path, issues)
            self._at_most(metrics, "nan_count", threshold, "max_nan_count", path, issues)
            self._at_most(metrics, "exploding_body_count", threshold, "max_exploding_body_count", path, issues)
        elif name == "collection":
            self._at_least(metrics, "episodes", threshold, "min_episodes", path, issues)
            self._at_most(metrics, "schema_errors", threshold, "max_schema_errors", path, issues)
            collection = manifest.get("collection")
            configured_streams = collection.get("required_streams", []) if isinstance(collection, dict) else []
            expected = set(item for item in configured_streams if isinstance(item, str))
            observed = metrics.get("streams")
            if (
                not isinstance(observed, list)
                or not all(isinstance(item, str) for item in observed)
                or not expected.issubset(set(observed))
            ):
                issues.append(ValidationIssue(f"{path}.streams", "threshold", f"must include {sorted(expected)}"))

    @staticmethod
    def _number(mapping: dict[str, Any], key: str, path: str, issues: list[ValidationIssue]) -> float | None:
        value = mapping.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            issues.append(ValidationIssue(f"{path}.{key}", "required", "numeric metric is required"))
            return None
        if not math.isfinite(float(value)):
            issues.append(ValidationIssue(f"{path}.{key}", "range", "numeric metric must be finite"))
            return None
        return float(value)

    def _at_most(
        self,
        metrics: dict[str, Any], metric_key: str,
        thresholds: dict[str, Any], threshold_key: str,
        path: str, issues: list[ValidationIssue],
    ) -> None:
        value = self._number(metrics, metric_key, path, issues)
        limit = self._number(thresholds, threshold_key, f"validation.thresholds.{path.split('.')[1]}", issues)
        if value is not None and limit is not None and value > limit:
            issues.append(ValidationIssue(f"{path}.{metric_key}", "threshold", f"{value} exceeds {limit}"))

    def _at_least(
        self,
        metrics: dict[str, Any], metric_key: str,
        thresholds: dict[str, Any], threshold_key: str,
        path: str, issues: list[ValidationIssue],
    ) -> None:
        value = self._number(metrics, metric_key, path, issues)
        limit = self._number(thresholds, threshold_key, f"validation.thresholds.{path.split('.')[1]}", issues)
        if value is not None and limit is not None and value < limit:
            issues.append(ValidationIssue(f"{path}.{metric_key}", "threshold", f"{value} is below {limit}"))

    def _numeric_at_least_value(
        self, metrics: dict[str, Any], key: str, limit: Any,
        path: str, issues: list[ValidationIssue],
    ) -> None:
        value = self._number(metrics, key, path, issues)
        if isinstance(limit, bool) or not isinstance(limit, (int, float)):
            issues.append(ValidationIssue(f"{path}.{key}", "manifest", "manifest limit is missing"))
        elif value is not None and value < float(limit):
            issues.append(ValidationIssue(f"{path}.{key}", "threshold", f"{value} is below {limit}"))

    @staticmethod
    def _boolean(
        mapping: dict[str, Any], key: str, expected: bool,
        path: str, issues: list[ValidationIssue],
    ) -> None:
        if mapping.get(key) is not expected:
            issues.append(ValidationIssue(f"{path}.{key}", "threshold", f"must be {expected}"))
