from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from string import Template

import yaml

from taskdev.constants import REQUIRED_CHECKS
from taskdev.io import TaskDevError, sha256_file
from taskdev.manifest import ManifestValidator
from taskdev.report import ReportGate
from taskdev.scaffold import TaskScaffolder


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_ROOT = REPOSITORY_ROOT / "taskdev" / "templates"
CLI = REPOSITORY_ROOT / "scripts" / "taskdev.py"


class TaskDevFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.task_id = "good_task"
        self.task_root = self.root / "taskdev" / "tasks" / self.task_id
        self.asset_root = self.task_root / "assets"
        self.evidence_root = self.task_root / "evidence"
        self.asset_root.mkdir(parents=True)
        self.evidence_root.mkdir()
        for name in ("REPLACE_WITH_ROKAE_ASSET", "REPLACE_WITH_WUJI_GEN1_ASSET", "REPLACE_WITH_TASK_ASSET"):
            (self.asset_root / name).write_text(f"fixture {name}\n", encoding="utf-8")
        template = Template((TEMPLATE_ROOT / "task.yaml").read_text(encoding="utf-8"))
        rendered = template.substitute(TASK_ID=self.task_id, TASK_TITLE="Good task")
        self.manifest_path = self.task_root / "task.yaml"
        self.manifest_path.write_text(rendered, encoding="utf-8")
        self.manifest = yaml.safe_load(rendered)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_manifest(self) -> None:
        self.manifest_path.write_text(yaml.safe_dump(self.manifest, sort_keys=False), encoding="utf-8")

    def valid_report(self) -> dict[str, object]:
        evidence: dict[str, list[dict[str, str]]] = {}
        for name in REQUIRED_CHECKS:
            path = self.evidence_root / f"{name}.json"
            path.write_text(json.dumps({"check": name, "measured": True}) + "\n", encoding="utf-8")
            evidence[name] = [
                {
                    "path": str(path.relative_to(self.root)),
                    "sha256": sha256_file(path),
                }
            ]
        metrics: dict[str, dict[str, object]] = {
            "schema": {"error_count": 0},
            "assets": {
                "missing_count": 0,
                "all_scales_verified": True,
                "all_collision_shapes_verified": True,
            },
            "reachability": {
                "target_fraction": 1.0,
                "min_joint_limit_margin_rad": 0.1,
                "max_position_error_m": 0.005,
                "max_orientation_error_rad": 0.1,
                "rokae_model_verified": True,
                "wuji_gen1_verified": True,
            },
            "contact": {
                "max_penetration_m": 0.002,
                "forbidden_events": 0,
                "required_pairs_observed": True,
            },
            "reset": {"trials": 20, "failure_fraction": 0.0, "deterministic": True},
            "success": {
                "positive_trials": 10,
                "negative_trials": 10,
                "false_positive_fraction": 0.0,
                "false_negative_fraction": 0.0,
                "hold_steps_observed": 30,
            },
            "camera": {
                "frames": 30,
                "black_fraction": 0.0,
                "mean_rgb": 80.0,
                "width": 640,
                "height": 480,
                "primary": "head",
            },
            "stability": {"sim_seconds": 10.0, "nan_count": 0, "exploding_body_count": 0},
            "collection": {
                "episodes": 3,
                "schema_errors": 0,
                "streams": ["action", "proprioception", "rgb_head", "contacts", "success"],
            },
        }
        return {
            "schema_version": 1,
            "task_id": self.task_id,
            "manifest_sha256": sha256_file(self.manifest_path),
            "generated_at": "2026-08-17T12:00:00+08:00",
            "simulator": {"name": "isaac_sim", "version": "5.1"},
            "checks": {
                name: {
                    "status": "pass",
                    "summary": f"{name} measured",
                    "evidence": evidence[name],
                    "metrics": metrics[name],
                }
                for name in REQUIRED_CHECKS
            },
        }


class ManifestValidatorTest(TaskDevFixture):
    def test_canonical_template_is_valid_after_real_assets_are_supplied(self) -> None:
        result = ManifestValidator(self.root).validate_path(self.manifest_path)
        self.assertTrue(result.valid, result.as_dict())

    def test_rejects_non_head_camera_and_missing_asset(self) -> None:
        self.manifest["camera"]["primary"] = "wrist"
        self.manifest["robot"]["arms"]["asset"] = "missing/rokae.xml"
        self.write_manifest()
        result = ManifestValidator(self.root).validate_path(self.manifest_path)
        self.assertFalse(result.valid)
        paths = {issue.path for issue in result.issues}
        self.assertIn("camera.primary", paths)
        self.assertIn("robot.arms.asset", paths)

    def test_rejects_incomplete_acceptance_contract(self) -> None:
        self.manifest["validation"]["required_checks"].remove("contact")
        del self.manifest["validation"]["thresholds"]["success"]["min_negative_trials"]
        self.write_manifest()
        result = ManifestValidator(self.root).validate_path(self.manifest_path)
        self.assertFalse(result.valid)
        paths = {issue.path for issue in result.issues}
        self.assertIn("validation.required_checks", paths)
        self.assertIn("validation.thresholds.success.min_negative_trials", paths)

    def test_malformed_sequence_members_fail_without_crashing(self) -> None:
        self.manifest["task"]["taxonomy"]["interactions"] = [{"not": "a string"}]
        self.manifest["validation"]["required_checks"] = [{"not": "a check"}]
        self.manifest["collection"]["required_streams"] = [{"not": "a stream"}]
        self.write_manifest()
        result = ManifestValidator(self.root).validate_path(self.manifest_path)
        self.assertFalse(result.valid)
        paths = {issue.path for issue in result.issues}
        self.assertIn("task.taxonomy.interactions", paths)
        self.assertIn("validation.required_checks", paths)
        self.assertIn("collection.required_streams", paths)


class ReportGateTest(TaskDevFixture):
    def test_accepts_all_measured_categories(self) -> None:
        result = ReportGate(self.root).evaluate(
            self.manifest,
            self.valid_report(),
            manifest_path=self.manifest_path,
        )
        self.assertTrue(result.valid, result.as_dict())

    def test_pending_category_fails_closed(self) -> None:
        report = self.valid_report()
        report["checks"]["contact"]["status"] = "pending"
        result = ReportGate(self.root).evaluate(self.manifest, report, manifest_path=self.manifest_path)
        self.assertFalse(result.valid)
        self.assertIn("checks.contact.status", {issue.path for issue in result.issues})

    def test_pass_label_cannot_override_metric_threshold(self) -> None:
        report = self.valid_report()
        report["checks"]["contact"]["metrics"]["max_penetration_m"] = 0.02
        result = ReportGate(self.root).evaluate(self.manifest, report, manifest_path=self.manifest_path)
        self.assertFalse(result.valid)
        self.assertIn("checks.contact.metrics.max_penetration_m", {issue.path for issue in result.issues})

    def test_changed_manifest_and_evidence_are_rejected(self) -> None:
        report = self.valid_report()
        evidence_path = self.root / report["checks"]["camera"]["evidence"][0]["path"]
        evidence_path.write_text("changed\n", encoding="utf-8")
        self.manifest_path.write_text(self.manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        result = ReportGate(self.root).evaluate(self.manifest, report, manifest_path=self.manifest_path)
        self.assertFalse(result.valid)
        paths = {issue.path for issue in result.issues}
        self.assertIn("report.manifest_sha256", paths)
        self.assertIn("checks.camera.evidence[0].sha256", paths)

    def test_isaac_sim_version_is_exactly_5_1(self) -> None:
        report = self.valid_report()
        report["simulator"]["version"] = "6.0"
        result = ReportGate(self.root).evaluate(self.manifest, report, manifest_path=self.manifest_path)
        self.assertFalse(result.valid)
        self.assertIn("report.simulator.version", {issue.path for issue in result.issues})

    def test_negative_or_nonfinite_metrics_are_rejected(self) -> None:
        report = self.valid_report()
        report["checks"]["camera"]["metrics"]["black_fraction"] = -1.0
        report["checks"]["stability"]["metrics"]["sim_seconds"] = float("nan")
        result = ReportGate(self.root).evaluate(self.manifest, report, manifest_path=self.manifest_path)
        self.assertFalse(result.valid)
        paths = {issue.path for issue in result.issues}
        self.assertIn("checks.camera.metrics.black_fraction", paths)
        self.assertIn("checks.stability.metrics.sim_seconds", paths)


class ScaffoldTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repository = self.root / "repository"
        self.worktrees = self.root / "worktrees"
        self.git(None, "init", str(self.repository))
        self.git(self.repository, "config", "user.name", "TaskDev Test")
        self.git(self.repository, "config", "user.email", "taskdev@example.invalid")
        (self.repository / "README.md").write_text("fixture\n", encoding="utf-8")
        template_destination = self.repository / "taskdev" / "templates"
        template_destination.mkdir(parents=True)
        for name in ("task.yaml", "validation_report.yaml", "WORKLOG.md"):
            (template_destination / name).write_text(
                (TEMPLATE_ROOT / name).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        self.git(self.repository, "add", ".")
        self.git(self.repository, "commit", "-m", "base")
        self.git(self.repository, "branch", "-M", "main")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def git(
        self,
        root: Path | None,
        *args: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = ["git"]
        if root is not None:
            command.extend(["-C", str(root)])
        command.extend(args)
        return subprocess.run(command, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def install_post_checkout_hook(self, body: str) -> None:
        common_dir = Path(
            self.git(self.repository, "rev-parse", "--git-common-dir").stdout.strip()
        )
        if not common_dir.is_absolute():
            common_dir = self.repository / common_dir
        hook = common_dir / "hooks" / "post-checkout"
        hook.write_text(f"#!/bin/sh\nset -eu\n{body}\n", encoding="utf-8")
        hook.chmod(0o755)

    def test_creates_one_branch_and_one_task_worktree(self) -> None:
        scaffolder = TaskScaffolder(self.repository)
        plan = scaffolder.plan(
            task_id="fixture_task",
            worktree_root=self.worktrees,
            base_ref="main",
        )
        scaffolder.create(plan, title="Fixture task")
        self.assertEqual(plan.branch, "taskdev/fixture_task")
        self.assertTrue((plan.worktree / "taskdev/tasks/fixture_task/task.yaml").is_file())
        self.assertTrue((plan.worktree / "taskdev/tasks/fixture_task/validation_report.yaml").is_file())
        report = yaml.safe_load(
            (plan.worktree / "taskdev/tasks/fixture_task/validation_report.yaml").read_text(encoding="utf-8")
        )
        self.assertTrue(all(report["checks"][name]["status"] == "pending" for name in REQUIRED_CHECKS))

    def test_refuses_existing_worktree_destination(self) -> None:
        destination = self.worktrees / "fixture_task"
        destination.mkdir(parents=True)
        with self.assertRaisesRegex(Exception, "already exists"):
            TaskScaffolder(self.repository).plan(
                task_id="fixture_task",
                worktree_root=self.worktrees,
                base_ref="main",
            )

    def test_reconciles_nonzero_hook_when_checkout_is_complete(self) -> None:
        self.install_post_checkout_hook("exit 23")
        scaffolder = TaskScaffolder(self.repository)
        plan = scaffolder.plan(
            task_id="hook_warning_task",
            worktree_root=self.worktrees,
            base_ref="main",
        )
        completed = scaffolder.create(plan, title="Hook warning task")
        self.assertTrue(completed.reconciled_git_error)
        self.assertTrue(
            (completed.worktree / "taskdev/tasks/hook_warning_task/task.yaml").is_file()
        )

    def test_rejects_nonzero_hook_when_worktree_is_dirty(self) -> None:
        self.install_post_checkout_hook(
            'root=$(git rev-parse --show-toplevel)\n: > "$root/hook_checkout_incomplete"\nexit 29'
        )
        scaffolder = TaskScaffolder(self.repository)
        plan = scaffolder.plan(
            task_id="hook_failure_task",
            worktree_root=self.worktrees,
            base_ref="main",
        )
        with self.assertRaisesRegex(TaskDevError, "failed read-only verification"):
            scaffolder.create(plan, title="Hook failure task")
        self.assertFalse(
            (plan.worktree / "taskdev/tasks/hook_failure_task").exists(),
            "templates must not be written after failed verification",
        )

    def test_resume_existing_verifies_half_created_worktree(self) -> None:
        self.install_post_checkout_hook("exit 31")
        worktree = self.worktrees / "resume_task"
        add = self.git(
            self.repository,
            "worktree",
            "add",
            "-b",
            "taskdev/resume_task",
            str(worktree),
            "main",
            check=False,
        )
        self.assertNotEqual(add.returncode, 0)
        scaffolder = TaskScaffolder(self.repository)
        plan = scaffolder.plan(
            task_id="resume_task",
            worktree_root=self.worktrees,
            base_ref="main",
            resume_existing=True,
        )
        completed = scaffolder.create(plan, title="Resume task")
        self.assertTrue(completed.existing_worktree)
        self.assertTrue((worktree / "taskdev/tasks/resume_task/task.yaml").is_file())


class CliTest(TaskDevFixture):
    def test_cli_emits_machine_readable_rejection(self) -> None:
        self.manifest["camera"]["primary"] = "wrist"
        self.write_manifest()
        completed = subprocess.run(
            [
                "python3",
                str(CLI),
                "--repository-root",
                str(self.root),
                "validate-manifest",
                str(self.manifest_path),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stderr, "")
        payload = json.loads(completed.stdout)
        self.assertFalse(payload["valid"])
        self.assertGreater(payload["error_count"], 0)


if __name__ == "__main__":
    unittest.main()
