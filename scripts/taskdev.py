#!/usr/bin/env python3
"""CLI for isolated task scaffolding and fail-closed validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
while str(REPOSITORY_ROOT) in sys.path:
    sys.path.remove(str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from taskdev.io import TaskDevError, load_yaml, resolve_inside  # noqa: E402
from taskdev.manifest import ManifestValidator  # noqa: E402
from taskdev.report import ReportGate  # noqa: E402
from taskdev.scaffold import TaskScaffolder  # noqa: E402


def _path_in_repository(root: Path, value: str, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TaskDevError(f"{label} must be a non-empty path")
    raw = Path(value)
    if raw.is_absolute():
        resolved = raw.resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise TaskDevError(f"{label} must be inside repository root") from exc
        return resolved
    return resolve_inside(root, value, label=label)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Git worktree root (default: repository containing this script)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("validate-manifest", help="validate a canonical task YAML")
    manifest.add_argument("manifest")

    report = subparsers.add_parser("gate-report", help="gate a validation report and its evidence")
    report.add_argument("manifest")
    report.add_argument("--report", help="report path; defaults to manifest validation.report")

    scaffold = subparsers.add_parser("scaffold", help="create one task branch and worktree")
    scaffold.add_argument("--task-id", required=True)
    scaffold.add_argument("--title", required=True)
    scaffold.add_argument("--base-ref", default="remote-task-dev-latest")
    scaffold.add_argument("--branch-prefix", default="taskdev")
    scaffold.add_argument(
        "--worktree-root",
        type=Path,
        required=True,
        help="parent directory for isolated task worktrees",
    )
    scaffold.add_argument("--dry-run", action="store_true")
    scaffold.add_argument(
        "--resume-existing",
        action="store_true",
        help="resume only a registered, clean half-created worktree at the exact base commit",
    )
    return parser


def _respond(payload: dict[str, object], returncode: int) -> int:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return returncode


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.repository_root.expanduser().resolve()
    try:
        if arguments.command == "validate-manifest":
            manifest_path = _path_in_repository(root, arguments.manifest, "manifest")
            result = ManifestValidator(root).validate_path(manifest_path)
            return _respond(
                {"command": "validate-manifest", "manifest": str(manifest_path), **result.as_dict()},
                0 if result.valid else 1,
            )

        if arguments.command == "gate-report":
            manifest_path = _path_in_repository(root, arguments.manifest, "manifest")
            if arguments.report:
                report_path = _path_in_repository(root, arguments.report, "report")
            else:
                manifest = load_yaml(manifest_path)
                if not isinstance(manifest, dict):
                    raise TaskDevError("manifest root must be a mapping")
                configured = manifest.get("validation", {}).get("report")
                report_path = _path_in_repository(root, configured, "validation.report")
            result = ReportGate(root).evaluate_paths(manifest_path, report_path)
            return _respond(
                {
                    "command": "gate-report",
                    "manifest": str(manifest_path),
                    "report": str(report_path),
                    **result.as_dict(),
                },
                0 if result.valid else 1,
            )

        if arguments.command == "scaffold":
            scaffolder = TaskScaffolder(root)
            plan = scaffolder.plan(
                task_id=arguments.task_id,
                worktree_root=arguments.worktree_root,
                base_ref=arguments.base_ref,
                branch_prefix=arguments.branch_prefix,
                resume_existing=arguments.resume_existing,
            )
            if not arguments.dry_run:
                plan = scaffolder.create(plan, title=arguments.title)
            return _respond(
                {
                    "command": "scaffold",
                    "created": not arguments.dry_run,
                    "valid": True,
                    **plan.as_dict(),
                },
                0,
            )
        raise TaskDevError(f"unknown command: {arguments.command}")
    except TaskDevError as exc:
        return _respond(
            {"command": arguments.command, "valid": False, "error_count": 1, "errors": [{"message": str(exc)}]},
            2,
        )
    except Exception as exc:
        return _respond(
            {
                "command": arguments.command,
                "valid": False,
                "error_count": 1,
                "errors": [{"message": f"internal fail-closed error: {type(exc).__name__}: {exc}"}],
            },
            2,
        )


if __name__ == "__main__":
    raise SystemExit(main())
