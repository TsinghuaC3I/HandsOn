"""Streaming, schema-driven audit for articulated PhysX trajectory replay.

The audit deliberately separates public solver readback from the public
contact report.  Contact points and solved impulses are useful observations,
but they are not PhysX's private persistent manifold or warm-start cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


_GROUP_CONTROL = "control_frame"
_GROUP_SUBSTEP = "substep"
_GROUP_TIMELINE = "timeline_and_counters"
_GROUP_CONTACT_CONTROL = "contact_output_control_frame"
_GROUP_CONTACT_SUBSTEP = "contact_output_substep"
_GROUPS = (
    _GROUP_CONTROL,
    _GROUP_SUBSTEP,
    _GROUP_TIMELINE,
    _GROUP_CONTACT_CONTROL,
    _GROUP_CONTACT_SUBSTEP,
)
_PHYSICAL_GROUPS = (_GROUP_CONTROL, _GROUP_SUBSTEP)
_CONTACT_GROUPS = (_GROUP_CONTACT_CONTROL, _GROUP_CONTACT_SUBSTEP)
_TIMELINE_FIELDS = {
    "physx_control_step",
    "physx_sim_time",
    "physx_substep_count",
    "physx_substep_sim_time",
}

# These fields describe commands, filters, or RNG caches rather than solver
# output.  The drive-target fields are intentionally retained: their readback
# proves that the intended input reached every control/substep boundary.
_NON_READBACK_FRAGMENTS = (
    "_command_hand",
    "_command_tcp_pose",
    "_pose_commanded",
    "_pose_filter_",
    "_rng_",
)


@dataclass(frozen=True)
class AuditField:
    name: str
    group: str
    schema_atol: float


@dataclass(frozen=True)
class _ContactPathIdentity:
    """Resolve process-local PhysX path IDs to stable USD path identities."""

    expected_id_to_canonical: Mapping[int, int]
    actual_id_to_canonical: Mapping[int, int]
    expected_path_count: int
    actual_path_count: int
    shared_path_count: int

    def report(self) -> dict[str, Any]:
        return {
            "semantic_comparison_enabled": True,
            "identity": "USD prim path from contact_report.path_table",
            "process_local_numeric_path_ids_excluded_from_semantic_pass": True,
            "process_local_stage_id_excluded_from_semantic_pass": True,
            "expected_path_table_count": self.expected_path_count,
            "actual_path_table_count": self.actual_path_count,
            "shared_path_count": self.shared_path_count,
        }


def _is_contact_field(name: str) -> bool:
    return "_contact_" in name or "_friction_anchor_" in name


def select_audit_fields(schema: Mapping[str, Any]) -> tuple[AuditField, ...]:
    """Select public physical readback and contact-output fields from schema."""

    raw_fields = schema.get("fields")
    if not isinstance(raw_fields, Mapping):
        raise ValueError("recording schema has no fields object")
    selected: list[AuditField] = []
    for raw_name, raw_spec in raw_fields.items():
        name = str(raw_name)
        if not name.startswith("physx_") or not isinstance(raw_spec, Mapping):
            continue
        semantics = raw_spec.get("semantics")
        if not isinstance(semantics, Mapping):
            continue
        atol_value = semantics.get("replay_atol", 0.0)
        try:
            atol = float(atol_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} has invalid replay_atol={atol_value!r}") from exc
        if not np.isfinite(atol) or atol < 0.0:
            raise ValueError(f"{name} has invalid replay_atol={atol!r}")

        if _is_contact_field(name):
            group = (
                _GROUP_CONTACT_SUBSTEP
                if name.startswith("physx_substep_")
                else _GROUP_CONTACT_CONTROL
            )
        else:
            if semantics.get("physical_replay") is not True:
                continue
            if any(fragment in name for fragment in _NON_READBACK_FRAGMENTS):
                continue
            if name in _TIMELINE_FIELDS:
                group = _GROUP_TIMELINE
            else:
                group = (
                    _GROUP_SUBSTEP
                    if name.startswith("physx_substep_")
                    else _GROUP_CONTROL
                )
        selected.append(AuditField(name=name, group=group, schema_atol=atol))
    if not selected:
        raise ValueError("recording schema selected no PhysX audit fields")
    return tuple(sorted(selected, key=lambda item: (item.group, item.name)))


def audit_field_names(schema: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(field.name for field in select_audit_fields(schema))


def _contact_path_table(schema: Mapping[str, Any], label: str) -> dict[int, str]:
    environment_schema = schema.get("environment_schema")
    schema_root = (
        environment_schema
        if isinstance(environment_schema, Mapping)
        else schema
    )
    physx = schema_root.get("physx")
    contact_report = physx.get("contact_report") if isinstance(physx, Mapping) else None
    raw_table = (
        contact_report.get("path_table")
        if isinstance(contact_report, Mapping)
        else None
    )
    if not isinstance(raw_table, list):
        raise ValueError(
            f"{label} schema has no environment_schema.physx.contact_report.path_table"
        )

    id_to_path: dict[int, str] = {}
    path_to_id: dict[str, int] = {}
    for index, raw_entry in enumerate(raw_table):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"{label} contact path_table[{index}] is not an object")
        raw_id = raw_entry.get("id")
        raw_path = raw_entry.get("path")
        if isinstance(raw_id, bool) or not isinstance(raw_id, (int, np.integer)):
            raise ValueError(
                f"{label} contact path_table[{index}] has invalid id={raw_id!r}"
            )
        path_id = int(raw_id)
        path = str(raw_path) if isinstance(raw_path, str) else ""
        if path_id < 0 or not path:
            raise ValueError(
                f"{label} contact path_table[{index}] has invalid entry"
            )
        if path_id in id_to_path and id_to_path[path_id] != path:
            raise ValueError(
                f"{label} contact path id {path_id} maps to multiple USD paths"
            )
        if path in path_to_id and path_to_id[path] != path_id:
            raise ValueError(
                f"{label} USD path {path!r} maps to multiple contact path ids"
            )
        id_to_path[path_id] = path
        path_to_id[path] = path_id
    return id_to_path


def _build_contact_path_identity(
    expected_schema: Mapping[str, Any],
    actual_schema: Mapping[str, Any],
) -> _ContactPathIdentity:
    expected_id_to_path = _contact_path_table(expected_schema, "recorded")
    actual_id_to_path = _contact_path_table(actual_schema, "runtime")
    all_paths = sorted(set(expected_id_to_path.values()) | set(actual_id_to_path.values()))
    canonical = {path: index + 1 for index, path in enumerate(all_paths)}
    return _ContactPathIdentity(
        expected_id_to_canonical={
            path_id: canonical[path] for path_id, path in expected_id_to_path.items()
        },
        actual_id_to_canonical={
            path_id: canonical[path] for path_id, path in actual_id_to_path.items()
        },
        expected_path_count=len(expected_id_to_path),
        actual_path_count=len(actual_id_to_path),
        shared_path_count=len(
            set(expected_id_to_path.values()) & set(actual_id_to_path.values())
        ),
    )


def _canonical_contact_path_ids(
    values: np.ndarray,
    id_to_canonical: Mapping[int, int],
) -> tuple[np.ndarray, list[int]]:
    result = np.zeros(values.shape, dtype=np.uint64)
    unknown: set[int] = set()
    for raw_value in np.unique(values):
        path_id = int(raw_value)
        canonical = id_to_canonical.get(path_id)
        if canonical is None:
            unknown.add(path_id)
            continue
        result[values == raw_value] = np.uint64(canonical)
    return result, sorted(unknown)


def _equal_with_nan(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    try:
        return np.equal(expected, actual) | (np.isnan(expected) & np.isnan(actual))
    except TypeError:
        return np.equal(expected, actual)


def _plain_comparison(expected: Any, actual: Any) -> dict[str, Any]:
    lhs = np.asarray(expected)
    rhs = np.asarray(actual)
    result: dict[str, Any] = {
        "expected_shape": list(lhs.shape),
        "actual_shape": list(rhs.shape),
        "expected_dtype": str(lhs.dtype),
        "actual_dtype": str(rhs.dtype),
        "dtype_exact": lhs.dtype == rhs.dtype,
        "sum_squared_error": 0.0,
        "rms_element_count": 0,
        "unquantifiable_mismatch": False,
    }
    if lhs.shape != rhs.shape:
        result.update(
            {
                "exact": False,
                "max_abs_error": None,
                "max_finite_abs_error": None,
                "reason": "shape",
                "unquantifiable_mismatch": True,
            }
        )
        return result

    equal = _equal_with_nan(lhs, rhs)
    result["exact"] = bool(result["dtype_exact"] and np.all(equal))
    if not (
        np.issubdtype(lhs.dtype, np.number)
        and np.issubdtype(rhs.dtype, np.number)
    ):
        result.update(
            {
                "max_abs_error": None,
                "max_finite_abs_error": None,
                "unquantifiable_mismatch": not bool(np.all(equal)),
            }
        )
        return result

    discrete = bool(
        np.issubdtype(lhs.dtype, np.integer)
        or np.issubdtype(lhs.dtype, np.bool_)
        or np.issubdtype(rhs.dtype, np.integer)
        or np.issubdtype(rhs.dtype, np.bool_)
    )
    # longdouble preserves uint64 identity on the supported x86 Isaac hosts;
    # exact/tolerance pass for discrete fields still uses np.equal below and
    # therefore never relies on this diagnostic conversion.
    conversion_dtype = np.longdouble if discrete else np.float64
    lhs64 = lhs.astype(conversion_dtype, copy=False)
    rhs64 = rhs.astype(conversion_dtype, copy=False)
    finite = np.isfinite(lhs64) & np.isfinite(rhs64)
    finite_diff = np.abs(lhs64[finite] - rhs64[finite])
    if finite_diff.size:
        max_finite = float(np.max(finite_diff))
        squared = np.square(finite_diff)
        sum_squared = float(np.sum(squared, dtype=np.float64))
    else:
        max_finite = 0.0
        sum_squared = 0.0
    nonfinite_mismatch = bool(np.any(~equal & ~finite))
    overflow = not np.isfinite(sum_squared)
    unquantifiable = bool(nonfinite_mismatch or overflow)
    result.update(
        {
            "max_abs_error": None if unquantifiable else max_finite,
            "max_finite_abs_error": max_finite,
            "sum_squared_error": sum_squared if not overflow else 0.0,
            "rms_element_count": int(finite_diff.size),
            "unquantifiable_mismatch": unquantifiable,
            "discrete": discrete,
        }
    )
    if nonfinite_mismatch:
        result["nonfinite_mismatch"] = True
    if overflow:
        result["squared_error_overflow"] = True
    return result


def _contact_count_name(name: str) -> str | None:
    prefix = "physx_substep_" if name.startswith("physx_substep_") else "physx_"
    suffix = name[len(prefix) :]
    if suffix in {"contact_header_ids", "contact_header_meta"}:
        return prefix + "contact_header_count"
    if suffix in {
        "contact_point_data",
        "contact_point_face_indices",
        "contact_point_material_ids",
    }:
        return prefix + "contact_point_count"
    if suffix == "friction_anchor_data":
        return prefix + "friction_anchor_count"
    return None


def _valid_row_batches(data: Any, counts: Any) -> list[np.ndarray]:
    array = np.asarray(data)
    count_array = np.asarray(counts)
    flat_counts = count_array.reshape(-1)
    if flat_counts.size == 1:
        batches = [array]
    elif array.ndim >= 2 and array.shape[0] == flat_counts.size:
        batches = [array[index] for index in range(flat_counts.size)]
    else:
        raise ValueError(
            f"contact data shape {array.shape} does not match count shape "
            f"{count_array.shape}"
        )
    result: list[np.ndarray] = []
    for batch, raw_count in zip(batches, flat_counts, strict=True):
        count = int(raw_count)
        if float(raw_count) != float(count) or count < 0 or count > batch.shape[0]:
            raise ValueError(
                f"invalid contact valid-row count {raw_count!r} for capacity "
                f"{batch.shape[0]}"
            )
        result.append(np.asarray(batch[:count]))
    return result


def _contact_valid_row_comparison(
    field_name: str,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    contact_path_identity: _ContactPathIdentity | None = None,
) -> dict[str, Any]:
    count_name = _contact_count_name(field_name)
    if count_name is None:
        return _plain_comparison(expected[field_name], actual[field_name])
    if count_name not in expected or count_name not in actual:
        return {
            "exact": False,
            "max_abs_error": None,
            "max_finite_abs_error": None,
            "sum_squared_error": 0.0,
            "rms_element_count": 0,
            "unquantifiable_mismatch": True,
            "reason": f"missing_valid_row_count:{count_name}",
        }
    try:
        expected_batches = _valid_row_batches(expected[field_name], expected[count_name])
        actual_batches = _valid_row_batches(actual[field_name], actual[count_name])
    except (TypeError, ValueError, IndexError) as exc:
        return {
            "exact": False,
            "max_abs_error": None,
            "max_finite_abs_error": None,
            "sum_squared_error": 0.0,
            "rms_element_count": 0,
            "unquantifiable_mismatch": True,
            "reason": "invalid_valid_row_layout",
            "detail": str(exc),
        }
    if len(expected_batches) != len(actual_batches):
        return {
            "exact": False,
            "max_abs_error": None,
            "max_finite_abs_error": None,
            "sum_squared_error": 0.0,
            "rms_element_count": 0,
            "unquantifiable_mismatch": True,
            "reason": "valid_row_batch_count",
            "expected_batch_count": len(expected_batches),
            "actual_batch_count": len(actual_batches),
        }

    expected_counts = [int(batch.shape[0]) for batch in expected_batches]
    actual_counts = [int(batch.shape[0]) for batch in actual_batches]
    overlaps = [min(lhs, rhs) for lhs, rhs in zip(expected_counts, actual_counts, strict=True)]
    expected_rows = sum(expected_counts)
    actual_rows = sum(actual_counts)
    compared_rows = sum(overlaps)
    expected_tail = np.asarray(expected[field_name]).shape[-1:]
    actual_tail = np.asarray(actual[field_name]).shape[-1:]
    if expected_tail != actual_tail:
        comparison = _plain_comparison(expected[field_name], actual[field_name])
    else:
        empty_shape = (0,) + tuple(expected_tail)
        lhs = (
            np.concatenate(
                [batch[:count] for batch, count in zip(expected_batches, overlaps, strict=True)],
                axis=0,
            )
            if compared_rows
            else np.empty(empty_shape, dtype=np.asarray(expected[field_name]).dtype)
        )
        rhs = (
            np.concatenate(
                [batch[:count] for batch, count in zip(actual_batches, overlaps, strict=True)],
                axis=0,
            )
            if compared_rows
            else np.empty(empty_shape, dtype=np.asarray(actual[field_name]).dtype)
        )
        comparison = _plain_comparison(lhs, rhs)
        # All five values are process-local encodings.  Resolve the first four
        # through each process's recorded path table; the fifth is only an
        # Omni stage handle and has no contact-pair semantics.
        if (
            field_name.endswith("contact_header_ids")
            and lhs.ndim == 2
            and rhs.ndim == 2
            and lhs.shape[1:] == (5,)
            and rhs.shape[1:] == (5,)
        ):
            raw_comparison = comparison
            raw_path_ids = _plain_comparison(lhs[:, :4], rhs[:, :4])
            raw_stage_ids = _plain_comparison(lhs[:, 4:], rhs[:, 4:])
            expected_unknown: list[int] = []
            actual_unknown: list[int] = []
            if contact_path_identity is None:
                comparison = raw_path_ids
                semantics = (
                    "raw actor/collider path IDs because no runtime path_table "
                    "was supplied; process-local stage_id reported separately"
                )
                mapping_complete: bool | None = None
            else:
                expected_canonical, expected_unknown = _canonical_contact_path_ids(
                    lhs[:, :4],
                    contact_path_identity.expected_id_to_canonical,
                )
                actual_canonical, actual_unknown = _canonical_contact_path_ids(
                    rhs[:, :4],
                    contact_path_identity.actual_id_to_canonical,
                )
                comparison = _plain_comparison(
                    expected_canonical,
                    actual_canonical,
                )
                mapping_complete = not expected_unknown and not actual_unknown
                semantics = (
                    "actor/collider USD prim paths resolved through each "
                    "process's path_table; raw numeric IDs and stage_id "
                    "reported separately"
                )
                if not mapping_complete:
                    comparison["exact"] = False
                    comparison["unquantifiable_mismatch"] = True
                    comparison["max_abs_error"] = None
                    comparison["reason"] = "unmapped_process_local_contact_path_id"
            comparison.update(
                {
                    "comparison_semantics": semantics,
                    "contact_path_identity_mapping_used": (
                        contact_path_identity is not None
                    ),
                    "contact_path_identity_mapping_complete": mapping_complete,
                    "expected_unmapped_contact_path_ids": expected_unknown,
                    "actual_unmapped_contact_path_ids": actual_unknown,
                    "process_local_numeric_path_ids_excluded_from_semantic_pass": (
                        contact_path_identity is not None
                    ),
                    "process_local_stage_id_excluded_from_semantic_pass": True,
                    "raw_exact_including_process_local_ids": bool(
                        raw_comparison["exact"]
                    ),
                    "raw_max_abs_error_including_process_local_ids": (
                        raw_comparison.get("max_abs_error")
                    ),
                    "raw_process_local_path_ids_exact": bool(
                        raw_path_ids["exact"]
                    ),
                    "raw_process_local_path_ids_max_abs_error": raw_path_ids.get(
                        "max_abs_error"
                    ),
                    "process_local_stage_id_exact": bool(raw_stage_ids["exact"]),
                    "process_local_stage_id_max_abs_error": raw_stage_ids.get(
                        "max_abs_error"
                    ),
                }
            )
    row_count_exact = expected_counts == actual_counts
    comparison.update(
        {
            "valid_row_count_exact": row_count_exact,
            "expected_valid_rows": expected_rows,
            "actual_valid_rows": actual_rows,
            "compared_valid_rows": compared_rows,
            "expected_valid_rows_per_batch": expected_counts,
            "actual_valid_rows_per_batch": actual_counts,
        }
    )
    comparison["exact"] = bool(comparison["exact"] and row_count_exact)
    if "raw_exact_including_process_local_ids" in comparison:
        comparison["raw_exact_including_process_local_ids"] = bool(
            comparison["raw_exact_including_process_local_ids"] and row_count_exact
        )
        comparison["raw_process_local_path_ids_exact"] = bool(
            comparison["raw_process_local_path_ids_exact"] and row_count_exact
        )
        comparison["process_local_stage_id_exact"] = bool(
            comparison["process_local_stage_id_exact"] and row_count_exact
        )
    if not row_count_exact:
        comparison["valid_row_count_mismatch"] = True
    return comparison


def compare_audit_field(
    field: AuditField,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    global_tolerance: float,
    contact_path_identity: _ContactPathIdentity | None = None,
) -> dict[str, Any]:
    if field.name not in expected:
        comparison = {
            "exact": False,
            "max_abs_error": None,
            "max_finite_abs_error": None,
            "sum_squared_error": 0.0,
            "rms_element_count": 0,
            "unquantifiable_mismatch": True,
            "reason": "missing_source_field",
        }
    elif field.name not in actual:
        comparison = {
            "exact": False,
            "max_abs_error": None,
            "max_finite_abs_error": None,
            "sum_squared_error": 0.0,
            "rms_element_count": 0,
            "unquantifiable_mismatch": True,
            "reason": "missing_runtime_field",
        }
    elif _is_contact_field(field.name):
        comparison = _contact_valid_row_comparison(
            field.name,
            expected,
            actual,
            contact_path_identity,
        )
    else:
        comparison = _plain_comparison(expected[field.name], actual[field.name])

    error = comparison.get("max_abs_error")
    unquantifiable = bool(comparison.get("unquantifiable_mismatch", False))
    discrete = bool(comparison.get("discrete", False))
    dtype_exact = bool(comparison.get("dtype_exact", False))
    structural_exact = not bool(comparison.get("valid_row_count_mismatch", False))
    comparison["schema_atol"] = float(field.schema_atol)
    comparison["global_tolerance"] = float(global_tolerance)
    comparison["within_schema_tolerance"] = bool(
        not unquantifiable
        and dtype_exact
        and structural_exact
        and error is not None
        and (
            bool(comparison["exact"])
            if discrete
            else float(error) <= field.schema_atol
        )
    )
    comparison["within_global_tolerance"] = bool(
        not unquantifiable
        and dtype_exact
        and structural_exact
        and error is not None
        and (
            bool(comparison["exact"])
            if discrete
            else float(error) <= global_tolerance
        )
    )
    return comparison


class _FieldStats:
    def __init__(self, field: AuditField, global_tolerance: float) -> None:
        self.field = field
        self.global_tolerance = float(global_tolerance)
        self.compared_step_count = 0
        self.nonexact_step_count = 0
        self.above_schema_tolerance_step_count = 0
        self.above_global_tolerance_step_count = 0
        self.first_nonexact_step: int | None = None
        self.first_schema_divergence_step: int | None = None
        self.first_global_divergence_step: int | None = None
        self.max_finite_abs_error = 0.0
        self.sum_squared_error = 0.0
        self.rms_element_count = 0
        self.unquantifiable_mismatch_step_count = 0
        self.valid_row_count_mismatch_step_count = 0
        self.expected_valid_rows = 0
        self.actual_valid_rows = 0
        self.compared_valid_rows = 0
        self.contact_path_identity_mapping_used = False
        self.contact_path_identity_mapping_incomplete_step_count = 0
        self.first_contact_path_identity_mapping_incomplete_step: int | None = None
        self.expected_unmapped_contact_path_ids: set[int] = set()
        self.actual_unmapped_contact_path_ids: set[int] = set()
        self.raw_all_id_nonexact_step_count = 0
        self.first_raw_all_id_nonexact_step: int | None = None
        self.raw_all_id_max_abs_error = 0.0
        self.raw_path_id_nonexact_step_count = 0
        self.first_raw_path_id_nonexact_step: int | None = None
        self.raw_path_id_max_abs_error = 0.0
        self.raw_stage_id_nonexact_step_count = 0
        self.first_raw_stage_id_nonexact_step: int | None = None
        self.raw_stage_id_max_abs_error = 0.0

    def update(self, step_index: int, comparison: Mapping[str, Any]) -> None:
        self.compared_step_count += 1
        if not bool(comparison["exact"]):
            self.nonexact_step_count += 1
            if self.first_nonexact_step is None:
                self.first_nonexact_step = int(step_index)
        if not bool(comparison["within_schema_tolerance"]):
            self.above_schema_tolerance_step_count += 1
            if self.first_schema_divergence_step is None:
                self.first_schema_divergence_step = int(step_index)
        if not bool(comparison["within_global_tolerance"]):
            self.above_global_tolerance_step_count += 1
            if self.first_global_divergence_step is None:
                self.first_global_divergence_step = int(step_index)
        finite_max = comparison.get("max_finite_abs_error")
        if finite_max is not None:
            self.max_finite_abs_error = max(self.max_finite_abs_error, float(finite_max))
        self.sum_squared_error += float(comparison.get("sum_squared_error", 0.0))
        self.rms_element_count += int(comparison.get("rms_element_count", 0))
        if comparison.get("unquantifiable_mismatch", False):
            self.unquantifiable_mismatch_step_count += 1
        if comparison.get("valid_row_count_mismatch", False):
            self.valid_row_count_mismatch_step_count += 1
        self.expected_valid_rows += int(comparison.get("expected_valid_rows", 0))
        self.actual_valid_rows += int(comparison.get("actual_valid_rows", 0))
        self.compared_valid_rows += int(comparison.get("compared_valid_rows", 0))
        if comparison.get("contact_path_identity_mapping_used") is True:
            self.contact_path_identity_mapping_used = True
        if comparison.get("contact_path_identity_mapping_complete") is False:
            self.contact_path_identity_mapping_incomplete_step_count += 1
            if self.first_contact_path_identity_mapping_incomplete_step is None:
                self.first_contact_path_identity_mapping_incomplete_step = int(
                    step_index
                )
        self.expected_unmapped_contact_path_ids.update(
            int(value)
            for value in comparison.get("expected_unmapped_contact_path_ids", [])
        )
        self.actual_unmapped_contact_path_ids.update(
            int(value)
            for value in comparison.get("actual_unmapped_contact_path_ids", [])
        )

        raw_all_exact = comparison.get("raw_exact_including_process_local_ids")
        if raw_all_exact is False:
            self.raw_all_id_nonexact_step_count += 1
            if self.first_raw_all_id_nonexact_step is None:
                self.first_raw_all_id_nonexact_step = int(step_index)
        raw_all_error = comparison.get(
            "raw_max_abs_error_including_process_local_ids"
        )
        if isinstance(raw_all_error, (int, float)):
            self.raw_all_id_max_abs_error = max(
                self.raw_all_id_max_abs_error,
                float(raw_all_error),
            )

        raw_path_exact = comparison.get("raw_process_local_path_ids_exact")
        if raw_path_exact is False:
            self.raw_path_id_nonexact_step_count += 1
            if self.first_raw_path_id_nonexact_step is None:
                self.first_raw_path_id_nonexact_step = int(step_index)
        raw_path_error = comparison.get(
            "raw_process_local_path_ids_max_abs_error"
        )
        if isinstance(raw_path_error, (int, float)):
            self.raw_path_id_max_abs_error = max(
                self.raw_path_id_max_abs_error,
                float(raw_path_error),
            )

        raw_stage_exact = comparison.get("process_local_stage_id_exact")
        if raw_stage_exact is False:
            self.raw_stage_id_nonexact_step_count += 1
            if self.first_raw_stage_id_nonexact_step is None:
                self.first_raw_stage_id_nonexact_step = int(step_index)
        raw_stage_error = comparison.get("process_local_stage_id_max_abs_error")
        if isinstance(raw_stage_error, (int, float)):
            self.raw_stage_id_max_abs_error = max(
                self.raw_stage_id_max_abs_error,
                float(raw_stage_error),
            )

    def report(self) -> dict[str, Any]:
        rms = (
            float(np.sqrt(self.sum_squared_error / self.rms_element_count))
            if self.rms_element_count
            else None
        )
        unquantifiable = self.unquantifiable_mismatch_step_count > 0
        result: dict[str, Any] = {
            "schema_atol": float(self.field.schema_atol),
            "global_tolerance": self.global_tolerance,
            "compared_step_count": self.compared_step_count,
            "nonexact_step_count": self.nonexact_step_count,
            "above_schema_tolerance_step_count": self.above_schema_tolerance_step_count,
            "above_global_tolerance_step_count": self.above_global_tolerance_step_count,
            "first_nonexact_step_zero_based": self.first_nonexact_step,
            "first_schema_divergence_step_zero_based": self.first_schema_divergence_step,
            "first_global_divergence_step_zero_based": self.first_global_divergence_step,
            "max_abs_error": None if unquantifiable else self.max_finite_abs_error,
            "max_finite_abs_error": self.max_finite_abs_error,
            "rms_abs_error": rms,
            "rms_element_count": self.rms_element_count,
            "rms_covers_all_mismatches": not unquantifiable,
            "unquantifiable_mismatch_step_count": self.unquantifiable_mismatch_step_count,
            "all_steps_exact": self.nonexact_step_count == 0,
            "within_schema_tolerance": self.above_schema_tolerance_step_count == 0,
            "within_global_tolerance": self.above_global_tolerance_step_count == 0,
        }
        if _is_contact_field(self.field.name):
            result.update(
                {
                    "valid_row_count_mismatch_step_count": self.valid_row_count_mismatch_step_count,
                    "expected_valid_rows_across_trajectory": self.expected_valid_rows,
                    "actual_valid_rows_across_trajectory": self.actual_valid_rows,
                    "compared_valid_rows_across_trajectory": self.compared_valid_rows,
                }
            )
        if self.field.name.endswith("contact_header_ids"):
            result.update(
                {
                    "comparison_semantics": (
                        "actor/collider USD prim paths resolved through each "
                        "process's path_table; raw numeric IDs and stage_id "
                        "reported separately"
                        if self.contact_path_identity_mapping_used
                        else "raw actor/collider path IDs; stage_id reported separately"
                    ),
                    "contact_path_identity_mapping_used": (
                        self.contact_path_identity_mapping_used
                    ),
                    "contact_path_identity_mapping_incomplete_step_count": (
                        self.contact_path_identity_mapping_incomplete_step_count
                    ),
                    "first_contact_path_identity_mapping_incomplete_step_zero_based": (
                        self.first_contact_path_identity_mapping_incomplete_step
                    ),
                    "expected_unmapped_contact_path_ids": sorted(
                        self.expected_unmapped_contact_path_ids
                    ),
                    "actual_unmapped_contact_path_ids": sorted(
                        self.actual_unmapped_contact_path_ids
                    ),
                    "process_local_numeric_path_ids_excluded_from_semantic_pass": (
                        self.contact_path_identity_mapping_used
                    ),
                    "process_local_stage_id_excluded_from_semantic_pass": True,
                    "raw_all_process_local_ids_nonexact_step_count": (
                        self.raw_all_id_nonexact_step_count
                    ),
                    "first_raw_all_process_local_ids_nonexact_step_zero_based": (
                        self.first_raw_all_id_nonexact_step
                    ),
                    "raw_all_process_local_ids_max_abs_error": (
                        self.raw_all_id_max_abs_error
                    ),
                    "raw_process_local_path_ids_nonexact_step_count": (
                        self.raw_path_id_nonexact_step_count
                    ),
                    "first_raw_process_local_path_ids_nonexact_step_zero_based": (
                        self.first_raw_path_id_nonexact_step
                    ),
                    "raw_process_local_path_ids_max_abs_error": (
                        self.raw_path_id_max_abs_error
                    ),
                    "raw_stage_id_nonexact_step_count": (
                        self.raw_stage_id_nonexact_step_count
                    ),
                    "first_raw_stage_id_nonexact_step_zero_based": (
                        self.first_raw_stage_id_nonexact_step
                    ),
                    "raw_stage_id_max_abs_error": self.raw_stage_id_max_abs_error,
                }
            )
        return result


class TrajectoryAudit:
    """Accumulate full-trajectory errors without changing replay execution."""

    def __init__(
        self,
        schema: Mapping[str, Any],
        global_tolerance: float,
        *,
        runtime_schema: Mapping[str, Any] | None = None,
    ) -> None:
        if not np.isfinite(global_tolerance) or global_tolerance < 0.0:
            raise ValueError("global tolerance must be finite and non-negative")
        self.global_tolerance = float(global_tolerance)
        self._contact_path_identity = (
            _build_contact_path_identity(schema, runtime_schema)
            if runtime_schema is not None
            else None
        )
        self.fields = select_audit_fields(schema)
        self._stats = {
            field.name: _FieldStats(field, self.global_tolerance) for field in self.fields
        }
        self._group_fields = {
            group: tuple(field for field in self.fields if field.group == group)
            for group in _GROUPS
        }
        self.step_count = 0
        self._first_events: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    def compare_step(
        self,
        expected: Mapping[str, Any],
        actual: Mapping[str, Any],
        step_index: int,
    ) -> dict[str, Any]:
        comparisons: dict[str, dict[str, Any]] = {}
        for field in self.fields:
            comparison = compare_audit_field(
                field,
                expected,
                actual,
                global_tolerance=self.global_tolerance,
                contact_path_identity=self._contact_path_identity,
            )
            comparisons[field.name] = comparison
            self._stats[field.name].update(step_index, comparison)
        self.step_count += 1
        return {
            "all_exact": all(item["exact"] for item in comparisons.values()),
            "within_schema_tolerance": all(
                item["within_schema_tolerance"] for item in comparisons.values()
            ),
            "within_global_tolerance": all(
                item["within_global_tolerance"] for item in comparisons.values()
            ),
            "comparisons": comparisons,
        }

    def _group_report(self, group: str) -> dict[str, Any]:
        fields = self._group_fields[group]
        field_reports = {field.name: self._stats[field.name].report() for field in fields}
        return _summarize_field_reports(field_reports, self.step_count)

    def report(self) -> dict[str, Any]:
        groups = {group: self._group_report(group) for group in _GROUPS}
        physical_reports = {
            name: item
            for group in _PHYSICAL_GROUPS
            for name, item in groups[group]["per_field"].items()
        }
        contact_reports = {
            name: item
            for group in _CONTACT_GROUPS
            for name, item in groups[group]["per_field"].items()
        }
        physical = _summarize_field_reports(physical_reports, self.step_count)
        contact = _summarize_field_reports(contact_reports, self.step_count)
        timeline = groups[_GROUP_TIMELINE]
        return {
            "compared_control_steps": self.step_count,
            "contact_path_identity": (
                self._contact_path_identity.report()
                if self._contact_path_identity is not None
                else {
                    "semantic_comparison_enabled": False,
                    "reason": "runtime schema was not supplied",
                }
            ),
            "strict_schema_pass": bool(
                physical["within_schema_tolerance"]
                and timeline["within_schema_tolerance"]
                and contact["within_schema_tolerance"]
            ),
            "all_audited_public_outputs_pass": bool(
                physical["within_schema_tolerance"]
                and timeline["within_schema_tolerance"]
                and contact["within_schema_tolerance"]
            ),
            "solver_state_schema_pass": physical["within_schema_tolerance"],
            "timeline_schema_pass": timeline["within_schema_tolerance"],
            "global_tolerance_pass": bool(
                physical["within_global_tolerance"]
                and timeline["within_global_tolerance"]
                and contact["within_global_tolerance"]
            ),
            "physical_state_schema_pass": physical["within_schema_tolerance"],
            "contact_output_schema_pass": contact["within_schema_tolerance"],
            "first_nonexact": physical["first_nonexact"],
            "first_divergence": physical["first_schema_divergence"],
            "first_global_tolerance_divergence": physical[
                "first_global_divergence"
            ],
            "control_frame": groups[_GROUP_CONTROL],
            "substep": groups[_GROUP_SUBSTEP],
            "timeline_and_counters": timeline,
            "contact_output": {
                "is_public_readback_not_private_solver_cache": True,
                "strict_schema_pass": contact["within_schema_tolerance"],
                "global_tolerance_pass": contact["within_global_tolerance"],
                "first_mismatch": contact["first_nonexact"],
                "first_divergence": contact["first_schema_divergence"],
                "control_frame": groups[_GROUP_CONTACT_CONTROL],
                "substep": groups[_GROUP_CONTACT_SUBSTEP],
            },
        }


def _first_field_event(
    field_reports: Mapping[str, Mapping[str, Any]],
    key: str,
) -> dict[str, Any] | None:
    candidates = [
        (int(item[key]), name)
        for name, item in field_reports.items()
        if item.get(key) is not None
    ]
    if not candidates:
        return None
    first_step = min(step for step, _name in candidates)
    fields = sorted(name for step, name in candidates if step == first_step)
    return {"step_zero_based": first_step, "fields": fields}


def _summarize_field_reports(
    field_reports: Mapping[str, Mapping[str, Any]],
    compared_step_count: int,
) -> dict[str, Any]:
    values = list(field_reports.values())
    finite_maxima = [float(item["max_finite_abs_error"]) for item in values]
    rms_sum = 0.0
    rms_count = 0
    for item in values:
        count = int(item["rms_element_count"])
        rms = item.get("rms_abs_error")
        if count and rms is not None:
            rms_sum += float(rms) ** 2 * count
            rms_count += count
    all_rms_covered = all(item["rms_covers_all_mismatches"] for item in values)
    return {
        "field_count": len(field_reports),
        "compared_control_steps": int(compared_step_count),
        # Optional field families (for example deformables or contacts) are
        # vacuously passing when the schema records none of them.  field_count
        # keeps that distinct from a populated group that actually matched.
        "all_steps_exact": all(item["all_steps_exact"] for item in values),
        "within_schema_tolerance": all(
            item["within_schema_tolerance"] for item in values
        ),
        "within_global_tolerance": all(
            item["within_global_tolerance"] for item in values
        ),
        "first_nonexact": _first_field_event(
            field_reports, "first_nonexact_step_zero_based"
        ),
        "first_schema_divergence": _first_field_event(
            field_reports, "first_schema_divergence_step_zero_based"
        ),
        "first_global_divergence": _first_field_event(
            field_reports, "first_global_divergence_step_zero_based"
        ),
        "max_abs_error": (
            max(finite_maxima, default=0.0) if all_rms_covered else None
        ),
        "max_finite_abs_error": max(finite_maxima, default=0.0),
        "rms_abs_error": float(np.sqrt(rms_sum / rms_count)) if rms_count else None,
        "rms_element_count": rms_count,
        "rms_covers_all_mismatches": all_rms_covered,
        "per_field": dict(field_reports),
    }


__all__ = (
    "AuditField",
    "TrajectoryAudit",
    "audit_field_names",
    "compare_audit_field",
    "select_audit_fields",
)
