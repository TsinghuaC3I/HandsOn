from __future__ import annotations

import numpy as np

from handson.sim.isaaclab.articulated_replay.trajectory_audit import (
    TrajectoryAudit,
)


def _schema(path_table: list[tuple[int, str]]) -> dict:
    contact_semantics = {"replay_atol": 0.0}
    return {
        "fields": {
            "physx_contact_header_count": {
                "semantics": contact_semantics,
            },
            "physx_contact_header_ids": {
                "semantics": contact_semantics,
            },
        },
        "environment_schema": {
            "physx": {
                "contact_report": {
                    "path_table": [
                        {"id": path_id, "path": path}
                        for path_id, path in path_table
                    ]
                }
            }
        },
    }


def _snapshot(header: list[int]) -> dict[str, np.ndarray]:
    ids = np.zeros((2, 5), dtype=np.uint64)
    ids[0] = np.asarray(header, dtype=np.uint64)
    return {
        "physx_contact_header_count": np.asarray([1], dtype=np.uint32),
        "physx_contact_header_ids": ids,
    }


def _id_report(audit: TrajectoryAudit) -> dict:
    return audit.report()["contact_output"]["control_frame"]["per_field"][
        "physx_contact_header_ids"
    ]


def test_contact_ids_compare_usd_paths_not_process_local_numbers() -> None:
    source = _schema([(101, "/World/A"), (102, "/World/B")])
    runtime = _schema([(501, "/World/A"), (502, "/World/B")])
    audit = TrajectoryAudit(
        source,
        0.0,
        runtime_schema=runtime["environment_schema"],
    )

    step = audit.compare_step(
        _snapshot([101, 102, 101, 102, 11]),
        _snapshot([501, 502, 501, 502, 99]),
        0,
    )
    report = _id_report(audit)

    assert step["all_exact"] is True
    assert audit.report()["contact_output_schema_pass"] is True
    assert report["all_steps_exact"] is True
    assert report["contact_path_identity_mapping_used"] is True
    assert report["contact_path_identity_mapping_incomplete_step_count"] == 0
    assert report["raw_process_local_path_ids_nonexact_step_count"] == 1
    assert report["raw_stage_id_nonexact_step_count"] == 1


def test_contact_ids_fail_when_resolved_usd_paths_differ() -> None:
    source = _schema([(101, "/World/A"), (102, "/World/B")])
    runtime = _schema([(501, "/World/A"), (502, "/World/C")])
    audit = TrajectoryAudit(
        source,
        0.0,
        runtime_schema=runtime["environment_schema"],
    )

    step = audit.compare_step(
        _snapshot([101, 102, 101, 102, 11]),
        _snapshot([501, 502, 501, 502, 99]),
        0,
    )

    assert step["all_exact"] is False
    assert audit.report()["contact_output_schema_pass"] is False
    assert _id_report(audit)["nonexact_step_count"] == 1


def test_contact_ids_fail_closed_when_a_path_id_is_not_in_table() -> None:
    source = _schema([(101, "/World/A"), (102, "/World/B")])
    runtime = _schema([(501, "/World/A"), (502, "/World/B")])
    audit = TrajectoryAudit(
        source,
        0.0,
        runtime_schema=runtime["environment_schema"],
    )

    step = audit.compare_step(
        _snapshot([101, 102, 101, 102, 11]),
        _snapshot([501, 999, 501, 502, 99]),
        0,
    )
    report = _id_report(audit)

    assert step["all_exact"] is False
    assert audit.report()["contact_output_schema_pass"] is False
    assert report["contact_path_identity_mapping_incomplete_step_count"] == 1
    assert report["actual_unmapped_contact_path_ids"] == [999]
