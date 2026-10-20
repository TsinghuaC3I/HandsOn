from __future__ import annotations

import argparse
from pathlib import Path

from .orchestrate import _trial_command
from .reporting import build_aggregate_report


def _trial(index: int, *, exit_code: int = 0, divergence_step: int | None = None):
    divergence = (
        {
            "step_zero_based": divergence_step,
            "diverged_fields": ["physx_task_rigid_body_state"],
        }
        if divergence_step is not None
        else None
    )
    return {
        "trial_index": index,
        "exit_code": exit_code,
        "worker_report": {
            "completed": True,
            "all_demo_trajectories_compared": True,
            "strict_full_trajectory_pass": divergence is None,
            "demos": [
                {
                    "demo_index": 1,
                    "trajectory": {
                        "source_control_steps": 5,
                        "compared_control_steps": 5,
                        "trajectory_exact": divergence is None,
                        "trajectory_within_tolerance": divergence is None,
                        "first_nonexact": divergence,
                        "first_divergence": divergence,
                        "field_max_abs_error": {
                            "physx_task_rigid_body_state": float(index),
                        },
                    },
                    "outcome": {"runtime_succeeded_ever": index == 1},
                }
            ],
        },
    }


def test_aggregate_keeps_each_trial_and_cross_trial_maximum() -> None:
    report = build_aggregate_report(
        demo_dirs=["demo1"],
        requested_trial_count=3,
        trials=[_trial(1), _trial(2, divergence_step=4), _trial(3, divergence_step=2)],
        completed=True,
    )
    demo = report["per_demo"][0]
    assert report["finished_trial_count"] == 3
    assert report["process_summary"]["runner_completed_trials"] == 3
    assert report["process_summary"]["all_demos_executed_trials"] == 3
    assert report["process_summary"]["strict_full_trajectory_pass_trials"] == 1
    assert demo["full_trajectory_trials"] == 3
    assert demo["runtime_success_trials"] == 1
    assert demo["cross_trial_field_max_abs_error"] == {
        "physx_task_rigid_body_state": 3.0,
    }
    assert demo["earliest_first_divergence"]["trial_index"] == 3
    assert demo["earliest_first_divergence"]["step_zero_based"] == 2


def test_aggregate_marks_crash_without_fabricating_demo_metrics() -> None:
    crashed = {
        "trial_index": 2,
        "exit_code": -11,
        "worker_report": None,
        "log_path": "/tmp/trial_002/isaac.log",
    }
    report = build_aggregate_report(
        demo_dirs=["demo1", "demo2", "demo3"],
        requested_trial_count=3,
        trials=[_trial(1), crashed],
        completed=False,
    )
    assert report["process_summary"]["signal_exit_trials"] == 1
    assert report["process_summary"]["hard_crash_or_early_exit_trials"] == 1
    assert report["process_summary"]["missing_worker_report_trials"] == 1
    assert report["per_demo"][0]["trials"][1]["available"] is False
    assert report["per_demo"][1]["full_trajectory_trials"] == 0


def test_aggregate_accepts_full_chain_v2_schema_without_certifying_history() -> None:
    trial = {
        "trial_index": 1,
        "exit_code": 0,
        "worker_report": {
            "completed": True,
            "execution": {"all_demos_executed": True},
            "trajectory_strict_pass": True,
            "strict_pass": False,
            "exact_history_certificate_valid": False,
            "demos": [
                {
                    "index": 1,
                    "strict_schema_pass": True,
                    "global_tolerance_pass": True,
                    "pre_action": {
                        "core_public_snapshot": {
                            "core_physical_fields_exact": True,
                        },
                        "strict_schema_pass": True,
                    },
                    "trajectory": {
                        "source_control_steps": 7,
                        "executed_control_steps": 7,
                        "first_divergence": None,
                        "audit": {
                            "control_frame": {
                                "per_field": {
                                    "q": {"max_abs_error": 0.25},
                                }
                            },
                            "substep": {"compared": True},
                            "contact_output": {"compared": True},
                        },
                    },
                    "task_outcome": {
                        "source": {"any_succeed": True},
                        "actual": {"any_succeed": True},
                        "stepwise_match_for_executed_prefix": True,
                        "final_outcome_match": True,
                    },
                }
            ],
        },
    }
    report = build_aggregate_report(
        demo_dirs=["demo1"],
        requested_trial_count=3,
        trials=[trial],
        completed=False,
    )
    summary = report["process_summary"]
    assert summary["all_demos_executed_trials"] == 1
    assert summary["strict_full_trajectory_pass_trials"] == 1
    assert summary["runner_strict_pass_trials"] == 0
    assert summary["exact_history_certificate_valid_trials"] == 0
    assert report["per_demo"][0]["runtime_success_trials"] == 1
    assert report["per_demo"][0]["cross_trial_field_max_abs_error"] == {
        "audit.control_frame.q": 0.25,
    }


def test_trial_command_passes_three_demos_to_full_chain_runner() -> None:
    args = argparse.Namespace(
        isaac_python=Path("/opt/isaac-python"),
        runner=Path("/repo/history_rebuild_b03.py"),
        demo_dirs=[Path("/d/one"), Path("/d/two"), Path("/d/three")],
        progress_every=17,
        divergence_tolerance=2e-5,
        no_wall_clock_throttle=True,
    )
    command = _trial_command(args, Path("/out/report.json"))
    assert command[:5] == [
        "/opt/isaac-python",
        "/repo/history_rebuild_b03.py",
        "/d/one",
        "/d/two",
        "/d/three",
    ]
    assert command[-1] == "--no-wall-clock-throttle"
