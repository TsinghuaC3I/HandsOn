"""Pure report aggregation for repeated B03 history-rebuild trials."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _demo_report(worker_report: Mapping[str, Any], demo_index: int) -> Mapping[str, Any]:
    demos = worker_report.get("demos")
    if not isinstance(demos, Sequence) or isinstance(demos, (str, bytes)):
        return {}
    for candidate in demos:
        item = _as_mapping(candidate)
        if item.get("demo_index", item.get("index")) == demo_index:
            return item
    return {}


def _trajectory_steps(trajectory: Mapping[str, Any]) -> tuple[Any, Any]:
    control_frame = _as_mapping(trajectory.get("control_frame"))
    source = trajectory.get(
        "source_control_steps", control_frame.get("source_control_steps")
    )
    executed = trajectory.get(
        "compared_control_steps",
        trajectory.get(
            "executed_control_steps", control_frame.get("executed_control_steps")
        ),
    )
    return source, executed


def _field_maxima(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    direct = trajectory.get("field_max_abs_error")
    maxima: dict[str, Any] = {}
    if isinstance(direct, Mapping):
        maxima.update({str(name): value for name, value in direct.items()})

    def visit_stat(node: Mapping[str, Any], prefix: str) -> None:
        max_keys = [
            str(name)
            for name in node
            if str(name).startswith("max_")
            and "error" in str(name)
            and "step" not in str(name)
        ]
        if "trajectory_max_abs_error" in node:
            max_keys.append("trajectory_max_abs_error")
        if max_keys:
            preferred = (
                "max_abs_error"
                if "max_abs_error" in node
                else "trajectory_max_abs_error"
                if "trajectory_max_abs_error" in node
                else None
            )
            if preferred is not None:
                maxima[prefix] = node.get(preferred)
            for name in sorted(set(max_keys)):
                if name != preferred:
                    maxima[f"{prefix}.{name}"] = node.get(name)
        if "max_abs_error" in node or "trajectory_max_abs_error" in node:
            return
        for name, child in node.items():
            if isinstance(child, Mapping):
                child_prefix = f"{prefix}.{name}" if prefix else str(name)
                visit_stat(child, child_prefix)

    def scan(node: Mapping[str, Any], scope: str = "") -> None:
        for stats_name in ("field_stats", "per_field"):
            stats = node.get(stats_name)
            if isinstance(stats, Mapping):
                for field, raw_stat in stats.items():
                    stat = _as_mapping(raw_stat)
                    if stat:
                        prefix = f"{scope}.{field}" if scope else str(field)
                        visit_stat(stat, prefix)
        for section in (
            "control_frame",
            "substep",
            "timeline_and_counters",
            "contact_output",
        ):
            child = node.get(section)
            if isinstance(child, Mapping):
                child_scope = f"{scope}.{section}" if scope else section
                scan(child, child_scope)

    scan(trajectory)
    audit = trajectory.get("audit")
    if isinstance(audit, Mapping):
        scan(audit, "audit")
    return maxima


def _first_divergence_key(item: Mapping[str, Any]) -> tuple[int, int]:
    divergence = _as_mapping(item.get("first_divergence"))
    step = divergence.get(
        "step_zero_based",
        divergence.get("control_step_zero_based", divergence.get("step")),
    )
    return (
        int(step) if isinstance(step, int) else 2**63 - 1,
        int(item.get("trial_index", 2**31 - 1)),
    )


def _outcome_matches(outcome: Mapping[str, Any]) -> bool:
    legacy = outcome.get("matches_source_success")
    if isinstance(legacy, bool):
        return legacy
    return bool(
        outcome.get("stepwise_match_for_executed_prefix") is True
        and outcome.get("final_outcome_match") is True
    )


def _runtime_succeeded(outcome: Mapping[str, Any]) -> bool:
    legacy = outcome.get("runtime_succeeded_ever")
    if isinstance(legacy, bool):
        return legacy
    return _as_mapping(outcome.get("actual")).get("any_succeed") is True


def _aggregate_demo(
    demo_index: int,
    demo_dir: str,
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    trial_metrics: list[dict[str, Any]] = []
    cross_trial_field_max: dict[str, float | None] = {}
    field_non_numeric_error_trials: dict[str, int] = {}
    divergence_candidates: list[Mapping[str, Any]] = []
    contact_divergence_candidates: list[Mapping[str, Any]] = []

    for trial in trials:
        worker = _as_mapping(trial.get("worker_report"))
        demo = _demo_report(worker, demo_index)
        trajectory = _as_mapping(demo.get("trajectory"))
        audit = _as_mapping(trajectory.get("audit"))
        outcome = _as_mapping(
            demo.get("outcome", demo.get("task_outcome", trajectory.get("task_outcome")))
        )
        source_steps, compared_steps = _trajectory_steps(trajectory)
        full_trajectory = bool(
            isinstance(source_steps, int)
            and isinstance(compared_steps, int)
            and source_steps == compared_steps
        )
        first_divergence = trajectory.get(
            "first_divergence", audit.get("first_divergence")
        )
        contact_output = _as_mapping(
            trajectory.get("contact_output", audit.get("contact_output"))
        )
        contact_first_divergence = contact_output.get("first_divergence")
        field_maxima = _field_maxima(trajectory)
        pre_action = _as_mapping(
            demo.get("pre_action_comparison", demo.get("pre_action"))
        )
        strict_schema_pass = demo.get(
            "strict_schema_pass", trajectory.get("strict_schema_pass")
        )
        global_tolerance_pass = demo.get(
            "global_tolerance_pass", trajectory.get("global_tolerance_pass")
        )
        strict_demo_pass = bool(
            full_trajectory
            and (
                strict_schema_pass is True
                and _outcome_matches(outcome)
                or pre_action.get("core_physical_fields_exact") is True
                and trajectory.get("trajectory_exact") is True
                and _outcome_matches(outcome)
            )
        )
        metric = {
            "trial_index": trial.get("trial_index"),
            "available": bool(demo),
            "process_exit_code": trial.get("exit_code"),
            "worker_completed": worker.get("completed") is True,
            "source_control_steps": source_steps,
            "compared_control_steps": compared_steps,
            "full_trajectory_compared": full_trajectory,
            "strict_demo_pass": strict_demo_pass,
            "global_tolerance_demo_pass": bool(
                full_trajectory
                and global_tolerance_pass is True
                and _outcome_matches(outcome)
            ),
            "trajectory_exact": trajectory.get("trajectory_exact"),
            "trajectory_within_tolerance": trajectory.get(
                "trajectory_within_tolerance",
                demo.get(
                    "global_tolerance_pass",
                    trajectory.get("global_tolerance_pass"),
                ),
            ),
            "first_nonexact": trajectory.get(
                "first_nonexact", audit.get("first_nonexact")
            ),
            "first_divergence": first_divergence,
            "contact_first_divergence": contact_first_divergence,
            "first_global_tolerance_divergence": audit.get(
                "first_global_tolerance_divergence"
            ),
            "field_max_abs_error": dict(field_maxima),
            "field_stats": trajectory.get("field_stats"),
            "control_frame": trajectory.get(
                "control_frame", audit.get("control_frame")
            ),
            "substep": trajectory.get("substep", audit.get("substep")),
            "contact_output": dict(contact_output),
            "timeline_and_counters": trajectory.get(
                "timeline_and_counters", audit.get("timeline_and_counters")
            ),
            "outcome": dict(outcome),
        }
        trial_metrics.append(metric)
        if isinstance(first_divergence, Mapping):
            divergence_candidates.append(metric)
        if isinstance(contact_first_divergence, Mapping):
            contact_divergence_candidates.append(
                {
                    **metric,
                    "first_divergence": contact_first_divergence,
                }
            )
        for field, raw_error in field_maxima.items():
            if not isinstance(raw_error, (int, float)):
                cross_trial_field_max.setdefault(str(field), None)
                field_non_numeric_error_trials[str(field)] = (
                    field_non_numeric_error_trials.get(str(field), 0) + 1
                )
                continue
            error = float(raw_error)
            previous = cross_trial_field_max.get(str(field))
            if previous is None or error > previous:
                cross_trial_field_max[str(field)] = error

    earliest = (
        min(divergence_candidates, key=_first_divergence_key)
        if divergence_candidates
        else None
    )
    earliest_contact = (
        min(contact_divergence_candidates, key=_first_divergence_key)
        if contact_divergence_candidates
        else None
    )
    return {
        "demo_index": demo_index,
        "demo_dir": demo_dir,
        "trials_with_demo_report": sum(item["available"] for item in trial_metrics),
        "full_trajectory_trials": sum(
            item["full_trajectory_compared"] for item in trial_metrics
        ),
        "trajectory_exact_trials": sum(
            item["trajectory_exact"] is True for item in trial_metrics
        ),
        "strict_demo_pass_trials": sum(
            item["strict_demo_pass"] for item in trial_metrics
        ),
        "global_tolerance_demo_pass_trials": sum(
            item["global_tolerance_demo_pass"] for item in trial_metrics
        ),
        "trajectory_within_tolerance_trials": sum(
            item["trajectory_within_tolerance"] is True for item in trial_metrics
        ),
        "runtime_success_trials": sum(
            _runtime_succeeded(_as_mapping(item["outcome"]))
            for item in trial_metrics
        ),
        "earliest_first_divergence": (
            {
                "trial_index": earliest["trial_index"],
                **dict(_as_mapping(earliest["first_divergence"])),
            }
            if earliest is not None
            else None
        ),
        "earliest_contact_first_divergence": (
            {
                "trial_index": earliest_contact["trial_index"],
                **dict(_as_mapping(earliest_contact["first_divergence"])),
            }
            if earliest_contact is not None
            else None
        ),
        "cross_trial_field_max_abs_error": cross_trial_field_max,
        "field_non_numeric_error_trial_count": field_non_numeric_error_trials,
        "trials": trial_metrics,
    }


def build_aggregate_report(
    *,
    demo_dirs: Sequence[str],
    requested_trial_count: int,
    trials: Sequence[Mapping[str, Any]],
    completed: bool,
) -> dict[str, Any]:
    """Build the crash-tolerant aggregate without importing Isaac Sim."""

    normalized_trials = [dict(_as_mapping(trial)) for trial in trials]
    for trial in normalized_trials:
        worker = _as_mapping(trial.get("worker_report"))
        execution = _as_mapping(worker.get("execution"))
        exit_code = trial.get("exit_code")
        trial["runner_completed"] = worker.get("completed") is True
        trial["all_demos_executed"] = (
            execution.get("all_demos_executed") is True
            or worker.get("all_demo_trajectories_compared") is True
        )
        trial["strict_full_trajectory_pass"] = (
            exit_code == 0
            and worker.get("completed") is True
            and (
                worker.get("trajectory_strict_pass") is True
                or worker.get("strict_full_trajectory_pass") is True
            )
        )
        trial["runner_strict_pass"] = worker.get("strict_pass") is True
        trial["exact_history_certificate_valid"] = (
            worker.get("exact_history_certificate_valid") is True
        )
        # Native crashes frequently leave no decodable worker JSON.  Keep this
        # separate from a normal Python failure report and from physics fail.
        trial["hard_crash_or_early_exit"] = bool(
            isinstance(exit_code, int)
            and exit_code != 0
            and not isinstance(trial.get("worker_report"), Mapping)
        )
    signal_exits = sum(
        isinstance(trial.get("exit_code"), int) and int(trial["exit_code"]) < 0
        for trial in normalized_trials
    )
    return {
        "format": "handson.b03_full_history_multi_trial",
        "version": 1,
        "completed": bool(completed),
        "fresh_process_per_trial": True,
        "experiment_scope": (
            "repeated fresh-process runs of the runner's declared "
            "published/recoverable-history reconstruction; original actions "
            "in one pre-demo episode are unavailable"
        ),
        "exact_source_m_t_lambda_t_recovery_claimed": False,
        "requested_trial_count": int(requested_trial_count),
        "finished_trial_count": len(normalized_trials),
        "demo_dirs": list(demo_dirs),
        "process_summary": {
            "zero_exit_trials": sum(
                trial.get("exit_code") == 0 for trial in normalized_trials
            ),
            "nonzero_exit_trials": sum(
                isinstance(trial.get("exit_code"), int)
                and int(trial["exit_code"]) != 0
                for trial in normalized_trials
            ),
            "signal_exit_trials": signal_exits,
            "launch_error_trials": sum(
                trial.get("kind") == "launch_error" for trial in normalized_trials
            ),
            "missing_worker_report_trials": sum(
                not isinstance(trial.get("worker_report"), Mapping)
                for trial in normalized_trials
            ),
            "runner_completed_trials": sum(
                trial["runner_completed"] for trial in normalized_trials
            ),
            "hard_crash_or_early_exit_trials": sum(
                trial["hard_crash_or_early_exit"] for trial in normalized_trials
            ),
            "all_demos_executed_trials": sum(
                trial["all_demos_executed"] for trial in normalized_trials
            ),
            "strict_full_trajectory_pass_trials": sum(
                trial["strict_full_trajectory_pass"] for trial in normalized_trials
            ),
            "runner_strict_pass_trials": sum(
                trial["runner_strict_pass"] for trial in normalized_trials
            ),
            "exact_history_certificate_valid_trials": sum(
                trial["exact_history_certificate_valid"]
                for trial in normalized_trials
            ),
        },
        "per_demo": [
            _aggregate_demo(index, str(path), normalized_trials)
            for index, path in enumerate(demo_dirs, start=1)
        ],
        "trials": normalized_trials,
    }
