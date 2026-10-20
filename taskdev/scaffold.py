"""Safe one-task-per-branch/worktree scaffolding."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from string import Template
from typing import Sequence

from .constants import TASK_ID_PATTERN
from .io import TaskDevError, atomic_write_text


REQUIRED_TRACKED_TEMPLATES = (
    "taskdev/templates/task.yaml",
    "taskdev/templates/validation_report.yaml",
    "taskdev/templates/WORKLOG.md",
)


@dataclass(frozen=True)
class ScaffoldPlan:
    task_id: str
    branch: str
    worktree: Path
    base_commit: str
    existing_worktree: bool = False
    reconciled_git_error: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "branch": self.branch,
            "worktree": str(self.worktree),
            "base_commit": self.base_commit,
            "existing_worktree": self.existing_worktree,
            "reconciled_git_error": self.reconciled_git_error,
        }


class TaskScaffolder:
    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root.resolve()
        self._git(("rev-parse", "--show-toplevel"))

    def plan(
        self,
        *,
        task_id: str,
        worktree_root: Path,
        base_ref: str,
        branch_prefix: str = "taskdev",
        resume_existing: bool = False,
    ) -> ScaffoldPlan:
        if not re.fullmatch(TASK_ID_PATTERN, task_id):
            raise TaskDevError("task id must be lower snake_case")
        branch = f"{branch_prefix}/{task_id}"
        valid_ref = self._git(("check-ref-format", f"refs/heads/{branch}"), check=False)
        if valid_ref.returncode:
            raise TaskDevError(f"invalid task branch: {branch}")
        branch_exists = (
            self._git(
                ("show-ref", "--verify", "--quiet", f"refs/heads/{branch}"),
                check=False,
            ).returncode
            == 0
        )
        base = self._git(("rev-parse", "--verify", f"{base_ref}^{{commit}}"), check=False)
        if base.returncode:
            raise TaskDevError(f"base ref does not resolve to a commit: {base_ref}")
        root = worktree_root.expanduser().resolve()
        if root == Path(root.anchor):
            raise TaskDevError("worktree root cannot be a filesystem root")
        if "\n" in str(root) or "\r" in str(root):
            raise TaskDevError("worktree root cannot contain newline characters")
        worktree = root / task_id
        if resume_existing:
            if not branch_exists:
                raise TaskDevError(f"cannot resume because task branch does not exist: {branch}")
            if not worktree.is_dir() or worktree.is_symlink():
                raise TaskDevError(f"cannot resume because worktree path is not a regular directory: {worktree}")
        else:
            if branch_exists:
                raise TaskDevError(f"task branch already exists: {branch}")
            if worktree.exists() or worktree.is_symlink():
                raise TaskDevError(f"worktree path already exists: {worktree}")
        return ScaffoldPlan(
            task_id,
            branch,
            worktree,
            base.stdout.strip(),
            existing_worktree=resume_existing,
        )

    def create(self, plan: ScaffoldPlan, *, title: str) -> ScaffoldPlan:
        if not title.strip():
            raise TaskDevError("task title must be non-empty")
        if plan.existing_worktree:
            self._verify_completed_worktree(plan)
        else:
            plan.worktree.parent.mkdir(parents=True, exist_ok=True)
            result = self._git(
                ("worktree", "add", "-b", plan.branch, str(plan.worktree), plan.base_commit),
                check=False,
            )
            if result.returncode:
                try:
                    self._verify_completed_worktree(plan)
                except TaskDevError as verification_error:
                    detail = (result.stderr or result.stdout).strip()[:2000]
                    raise TaskDevError(
                        "git worktree add returned non-zero and the resulting worktree "
                        f"failed read-only verification: {verification_error}; git said: {detail}"
                    ) from verification_error
                plan = replace(plan, reconciled_git_error=True)
            else:
                self._verify_completed_worktree(plan)

        task_root = plan.worktree / "taskdev" / "tasks" / plan.task_id
        replacements = {"TASK_ID": plan.task_id, "TASK_TITLE": title.strip()}
        if task_root.exists() or task_root.is_symlink():
            raise TaskDevError(f"task scaffold destination already exists: {task_root}")
        rendered_templates: list[tuple[str, str]] = []
        for tracked_path in REQUIRED_TRACKED_TEMPLATES:
            source = plan.worktree / tracked_path
            destination_name = Path(tracked_path).name
            rendered_templates.append(
                (
                    destination_name,
                    Template(source.read_text(encoding="utf-8")).substitute(replacements),
                )
            )
        try:
            task_root.mkdir(parents=True, exist_ok=False)
            (task_root / "assets").mkdir()
            (task_root / "evidence").mkdir()
            for destination_name, rendered in rendered_templates:
                atomic_write_text(task_root / destination_name, rendered)
        except Exception as exc:
            raise TaskDevError(
                f"worktree was created but task files are incomplete at {task_root}: {exc}; "
                "inspect it manually (the tool never deletes worktrees)"
            ) from exc
        return plan

    def _verify_completed_worktree(self, plan: ScaffoldPlan) -> None:
        """Read-only proof that a non-zero checkout nevertheless completed safely."""

        if not plan.worktree.is_dir() or plan.worktree.is_symlink():
            raise TaskDevError(f"worktree is not a regular directory: {plan.worktree}")

        expected_branch = f"refs/heads/{plan.branch}"
        registered = None
        for record in self._worktree_records():
            candidate = record.get("worktree")
            if candidate and Path(candidate).resolve() == plan.worktree.resolve():
                registered = record
                break
        if registered is None:
            raise TaskDevError("worktree path is not registered in git worktree list")
        if registered.get("HEAD") != plan.base_commit:
            raise TaskDevError(
                f"registered HEAD is {registered.get('HEAD')}, expected {plan.base_commit}"
            )
        if registered.get("branch") != expected_branch:
            raise TaskDevError(
                f"registered branch is {registered.get('branch')}, expected {expected_branch}"
            )

        top_level = self._git_at(plan.worktree, ("rev-parse", "--show-toplevel"), check=False)
        if top_level.returncode or Path(top_level.stdout.strip()).resolve() != plan.worktree.resolve():
            raise TaskDevError("worktree top-level path does not match the requested path")
        head = self._git_at(plan.worktree, ("rev-parse", "--verify", "HEAD^{commit}"), check=False)
        if head.returncode or head.stdout.strip() != plan.base_commit:
            raise TaskDevError("worktree HEAD does not match the requested base commit")
        branch = self._git_at(
            plan.worktree,
            ("symbolic-ref", "--quiet", "HEAD"),
            check=False,
        )
        if branch.returncode or branch.stdout.strip() != expected_branch:
            raise TaskDevError("worktree symbolic branch does not match the requested task branch")
        status = self._git_at(
            plan.worktree,
            ("status", "--porcelain=v1", "--untracked-files=all"),
            check=False,
        )
        if status.returncode:
            raise TaskDevError(f"cannot verify worktree cleanliness: {self._detail(status)}")
        if status.stdout:
            raise TaskDevError("worktree is not clean after checkout")

        for tracked_path in REQUIRED_TRACKED_TEMPLATES:
            candidate = plan.worktree / tracked_path
            if not candidate.is_file() or candidate.is_symlink():
                raise TaskDevError(f"required tracked file is missing or unsafe: {tracked_path}")
            tracked = self._git_at(
                plan.worktree,
                ("ls-files", "--error-unmatch", "--", tracked_path),
                check=False,
            )
            if tracked.returncode:
                raise TaskDevError(f"required file is not tracked: {tracked_path}")
            expected = self._git(
                ("cat-file", "blob", f"{plan.base_commit}:{tracked_path}"),
                check=False,
            )
            if expected.returncode:
                raise TaskDevError(f"required file is absent from base commit: {tracked_path}")
            try:
                observed = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise TaskDevError(f"cannot read required tracked file {tracked_path}: {exc}") from exc
            if observed != expected.stdout:
                raise TaskDevError(f"required tracked file is incomplete or modified: {tracked_path}")

    def _worktree_records(self) -> list[dict[str, str]]:
        result = self._git(("worktree", "list", "--porcelain"))
        records: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if not line:
                if current:
                    records.append(current)
                    current = {}
                continue
            key, separator, value = line.partition(" ")
            current[key] = value if separator else "true"
        if current:
            records.append(current)
        return records

    def _git(self, args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return self._git_at(self.repository_root, args, check=check)

    def _git_at(
        self,
        root: Path,
        args: Sequence[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["GIT_TERMINAL_PROMPT"] = "0"
        completed = subprocess.run(
            ["git", "-c", "submodule.recurse=false", "-C", str(root), *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
        if check and completed.returncode:
            raise TaskDevError(f"git {' '.join(args)} failed: {self._detail(completed)}")
        return completed

    @staticmethod
    def _detail(completed: subprocess.CompletedProcess[str]) -> str:
        detail = (completed.stderr or completed.stdout).strip()
        return (detail or f"exit code {completed.returncode}")[:2000]
