from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "collection_release.py"
RELEASE_BRANCH = "release/collection"
HARDWARE_PATH = Path(
    "teleoperation/manus/vendor/wuji-hand-teleop/src/input_devices/"
    "openvr_input/config/openvr_input.yaml"
)


class CollectionReleaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)
        self.remote = self.workspace / "remote.git"
        self.source = self.workspace / "source"
        self.git(None, "init", "--bare", str(self.remote))
        self.git(None, "init", str(self.source))
        self.configure_repository(self.source)
        self.git(self.source, "remote", "add", "origin", str(self.remote))

        (self.source / HARDWARE_PATH).parent.mkdir(parents=True)
        (self.source / HARDWARE_PATH).write_text("station: source\n", encoding="utf-8")
        (self.source / ".gitignore").write_text(
            ".env\n.handson_collection.env\n/demos/\n",
            encoding="utf-8",
        )
        (self.source / "code.txt").write_text("version 1\n", encoding="utf-8")
        self.git(self.source, "add", ".")
        self.git(self.source, "commit", "-m", "initial")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def git(self, root: Path | None, *args: str) -> subprocess.CompletedProcess[str]:
        command = ["git"]
        if root is not None:
            command.extend(["-C", str(root)])
        command.extend(args)
        return subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def configure_repository(self, root: Path) -> None:
        self.git(root, "config", "user.name", "Collection Release Test")
        self.git(root, "config", "user.email", "collection-release@example.invalid")

    def run_cli(
        self,
        root: Path,
        command: str,
        *extra: str,
    ) -> tuple[int, dict[str, object]]:
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--root",
                str(root),
                "--remote",
                "origin",
                "--branch",
                RELEASE_BRANCH,
                "--fetch",
                command,
                *extra,
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(completed.stderr, "")
        payload = json.loads(completed.stdout)
        for key in (
            "success",
            "updated",
            "local_sha",
            "remote_sha",
            "update_available",
            "diverged",
            "can_publish",
            "can_update",
            "code",
            "detail",
        ):
            self.assertIn(key, payload)
        return completed.returncode, payload

    def publish_initial(self) -> dict[str, object]:
        returncode, payload = self.run_cli(self.source, "publish")
        self.assertEqual(returncode, 0)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["code"], "published")
        return payload

    def clone_client(self, name: str = "client") -> Path:
        client = self.workspace / name
        self.git(
            None,
            "clone",
            "--branch",
            RELEASE_BRANCH,
            "--single-branch",
            str(self.remote),
            str(client),
        )
        self.configure_repository(client)
        return client

    def commit_source_code(self, content: str, message: str = "release update") -> str:
        (self.source / "code.txt").write_text(content, encoding="utf-8")
        self.git(self.source, "add", "code.txt")
        self.git(self.source, "commit", "-m", message)
        return self.git(self.source, "rev-parse", "HEAD").stdout.strip()

    def test_initial_publish_and_status_schema(self) -> None:
        returncode, checked = self.run_cli(self.source, "check")
        self.assertEqual(returncode, 0)
        self.assertEqual(checked["code"], "ready")
        self.assertEqual(checked["relation"], "remote_missing")
        self.assertTrue(checked["can_publish"])

        published = self.publish_initial()
        self.assertEqual(published["local_sha"], published["remote_sha"])

        returncode, status = self.run_cli(self.source, "status")
        self.assertEqual(returncode, 0)
        self.assertEqual(status["relation"], "up_to_date")
        self.assertFalse(status["update_available"])
        self.assertTrue(status["can_update"])

    def test_status_keeps_local_version_available_when_remote_is_offline(self) -> None:
        self.publish_initial()
        client = self.clone_client()

        expected_remote = self.commit_source_code("version 2\n")
        self.run_cli(self.source, "publish")
        returncode, online = self.run_cli(client, "status")
        self.assertEqual(returncode, 0)
        self.assertEqual(online["remote_sha"], expected_remote)
        self.assertTrue(online["remote_available"])

        unavailable_remote = self.workspace / "remote-offline.git"
        self.remote.rename(unavailable_remote)
        try:
            returncode, offline = self.run_cli(client, "status")
        finally:
            unavailable_remote.rename(self.remote)

        self.assertEqual(returncode, 0)
        self.assertTrue(offline["success"])
        self.assertTrue(offline["available"])
        self.assertFalse(offline["remote_available"])
        self.assertEqual(offline["code"], "remote_status_unavailable")
        self.assertEqual(
            offline["local_sha"],
            self.git(client, "rev-parse", "HEAD").stdout.strip(),
        )
        self.assertEqual(offline["remote_sha"], expected_remote)
        self.assertFalse(offline["can_publish"])
        self.assertFalse(offline["can_update"])

    def test_publish_allows_hardware_config_but_blocks_other_dirty_paths(self) -> None:
        self.publish_initial()
        (self.source / HARDWARE_PATH).write_text("station: locally-calibrated\n", encoding="utf-8")
        new_head = self.commit_source_code("version 2\n")

        returncode, published = self.run_cli(self.source, "publish")
        self.assertEqual(returncode, 0)
        self.assertEqual(published["code"], "published")
        self.assertEqual(published["remote_sha"], new_head)
        hardware_entry = next(
            entry for entry in published["worktree"] if entry["path"] == HARDWARE_PATH.as_posix()
        )
        self.assertTrue(hardware_entry["allowed_for_publish"])

        (self.source / "untracked.txt").write_text("not released\n", encoding="utf-8")
        returncode, blocked = self.run_cli(self.source, "publish")
        self.assertEqual(returncode, 1)
        self.assertEqual(blocked["code"], "dirty_worktree")
        self.assertIn("untracked.txt", blocked["blocking_publish_paths"])

    def test_publish_rejects_diverged_remote(self) -> None:
        self.publish_initial()
        other = self.clone_client("other-publisher")
        (other / "remote-only.txt").write_text("remote\n", encoding="utf-8")
        self.git(other, "add", "remote-only.txt")
        self.git(other, "commit", "-m", "remote only")
        self.git(other, "push", "origin", f"HEAD:{RELEASE_BRANCH}")
        self.commit_source_code("source divergence\n", "source only")

        status_returncode, status = self.run_cli(self.source, "status")
        self.assertEqual(status_returncode, 0)
        self.assertTrue(status["success"])
        self.assertTrue(status["diverged"])

        returncode, blocked = self.run_cli(self.source, "publish")
        self.assertEqual(returncode, 1)
        self.assertEqual(blocked["code"], "diverged")
        self.assertTrue(blocked["diverged"])

    def test_publish_respects_pre_push_hook(self) -> None:
        published = self.publish_initial()
        remote_before = published["remote_sha"]
        self.commit_source_code("version blocked by hook\n")
        hook = self.source / ".git" / "hooks" / "pre-push"
        hook.write_text("#!/bin/sh\nexit 23\n", encoding="utf-8")
        hook.chmod(0o755)

        returncode, blocked = self.run_cli(self.source, "publish")
        self.assertEqual(returncode, 1)
        self.assertEqual(blocked["code"], "push_rejected")
        remote_after = self.git(
            None,
            "--git-dir",
            str(self.remote),
            "rev-parse",
            f"refs/heads/{RELEASE_BRANCH}",
        ).stdout.strip()
        self.assertEqual(remote_after, remote_before)

    def test_update_fast_forwards_and_preserves_local_machine_data(self) -> None:
        self.publish_initial()
        client = self.clone_client()
        (client / HARDWARE_PATH).write_text("station: client\n", encoding="utf-8")
        (client / ".env").write_text("TOKEN=local\n", encoding="utf-8")
        (client / "demos").mkdir()
        (client / "demos" / "capture.bin").write_bytes(b"local demo")

        expected_head = self.commit_source_code("version 2\n")
        returncode, _ = self.run_cli(self.source, "publish")
        self.assertEqual(returncode, 0)

        returncode, updated = self.run_cli(client, "update")
        self.assertEqual(returncode, 0)
        self.assertEqual(updated["code"], "updated")
        self.assertTrue(updated["updated"])
        self.assertEqual(updated["local_sha"], expected_head)
        self.assertEqual(updated["remote_sha"], expected_head)
        self.assertEqual(updated["applied_paths"], ["code.txt"])
        self.assertFalse(updated["update_available"])
        self.assertEqual((client / HARDWARE_PATH).read_text(encoding="utf-8"), "station: client\n")
        self.assertEqual((client / ".env").read_text(encoding="utf-8"), "TOKEN=local\n")
        self.assertEqual((client / "demos" / "capture.bin").read_bytes(), b"local demo")

    def test_update_overwrites_overlapping_local_change(self) -> None:
        self.publish_initial()
        client = self.clone_client()
        (client / "code.txt").write_text("client edit\n", encoding="utf-8")
        (client / HARDWARE_PATH).write_text("station: client\n", encoding="utf-8")
        expected_head = self.commit_source_code("version 2\n")
        self.run_cli(self.source, "publish")

        returncode, updated = self.run_cli(client, "update")
        self.assertEqual(returncode, 0)
        self.assertEqual(updated["code"], "updated")
        self.assertEqual(updated["conflicting_paths"], ["code.txt"])
        self.assertEqual(updated["discarded_local_paths"], ["code.txt"])
        self.assertEqual(self.git(client, "rev-parse", "HEAD").stdout.strip(), expected_head)
        self.assertEqual((client / "code.txt").read_text(encoding="utf-8"), "version 2\n")
        self.assertEqual((client / HARDWARE_PATH).read_text(encoding="utf-8"), "station: client\n")

    def test_update_removes_untracked_file_that_conflicts_with_release(self) -> None:
        self.publish_initial()
        client = self.clone_client()
        conflict = client / "new_release_code.py"
        conflict.write_text("local untracked code\n", encoding="utf-8")

        (self.source / "new_release_code.py").write_text(
            "released code\n",
            encoding="utf-8",
        )
        self.git(self.source, "add", "new_release_code.py")
        self.git(self.source, "commit", "-m", "add release code")
        expected_head = self.git(self.source, "rev-parse", "HEAD").stdout.strip()
        self.run_cli(self.source, "publish")

        returncode, updated = self.run_cli(client, "update")
        self.assertEqual(returncode, 0)
        self.assertEqual(updated["code"], "updated")
        self.assertEqual(updated["conflicting_paths"], ["new_release_code.py"])
        self.assertEqual(updated["discarded_local_paths"], ["new_release_code.py"])
        self.assertEqual(self.git(client, "rev-parse", "HEAD").stdout.strip(), expected_head)
        self.assertEqual(conflict.read_text(encoding="utf-8"), "released code\n")

    def test_update_discards_unrelated_local_code_but_preserves_hardware(self) -> None:
        self.publish_initial()
        client = self.clone_client()
        (client / "code.txt").write_text("uncommitted client edit\n", encoding="utf-8")
        (client / "local_only.py").write_text("untracked code\n", encoding="utf-8")
        (client / HARDWARE_PATH).write_text("station: client\n", encoding="utf-8")

        (self.source / "remote_only.py").write_text("released code\n", encoding="utf-8")
        self.git(self.source, "add", "remote_only.py")
        self.git(self.source, "commit", "-m", "add unrelated release code")
        self.run_cli(self.source, "publish")

        returncode, updated = self.run_cli(client, "update")
        self.assertEqual(returncode, 0)
        self.assertEqual(updated["code"], "updated")
        self.assertEqual(
            updated["discarded_local_paths"],
            ["code.txt", "local_only.py"],
        )
        self.assertEqual((client / "code.txt").read_text(encoding="utf-8"), "version 1\n")
        self.assertFalse((client / "local_only.py").exists())
        self.assertEqual((client / HARDWARE_PATH).read_text(encoding="utf-8"), "station: client\n")

    def test_update_overlays_local_hardware_when_release_changes_same_file(self) -> None:
        self.publish_initial()
        client = self.clone_client()
        (client / HARDWARE_PATH).write_text("station: client\n", encoding="utf-8")

        (self.source / HARDWARE_PATH).write_text("station: released-default\n", encoding="utf-8")
        self.git(self.source, "add", HARDWARE_PATH.as_posix())
        self.git(self.source, "commit", "-m", "update released hardware defaults")
        expected_head = self.git(self.source, "rev-parse", "HEAD").stdout.strip()
        self.run_cli(self.source, "publish")

        returncode, updated = self.run_cli(client, "update")
        self.assertEqual(returncode, 0)
        self.assertEqual(updated["code"], "updated")
        self.assertEqual(updated["local_sha"], expected_head)
        self.assertEqual(updated["preserved_local_paths"], [HARDWARE_PATH.as_posix()])
        self.assertEqual((client / HARDWARE_PATH).read_text(encoding="utf-8"), "station: client\n")

    def test_update_restores_local_code_when_release_head_is_unchanged(self) -> None:
        self.publish_initial()
        client = self.clone_client()
        head_before = self.git(client, "rev-parse", "HEAD").stdout.strip()
        (client / "code.txt").write_text("uncommitted client edit\n", encoding="utf-8")

        returncode, updated = self.run_cli(client, "update")
        self.assertEqual(returncode, 0)
        self.assertEqual(updated["code"], "updated")
        self.assertTrue(updated["updated"])
        self.assertEqual(updated["discarded_local_paths"], ["code.txt"])
        self.assertEqual(updated["applied_paths"], ["code.txt"])
        self.assertEqual(self.git(client, "rev-parse", "HEAD").stdout.strip(), head_before)
        self.assertEqual((client / "code.txt").read_text(encoding="utf-8"), "version 1\n")

    def test_update_rejects_incoming_ignored_data(self) -> None:
        self.publish_initial()
        client = self.clone_client()
        (self.source / "demos").mkdir()
        (self.source / "demos" / "published.bin").write_bytes(b"must not deploy")
        self.git(self.source, "add", "-f", "demos/published.bin")
        self.git(self.source, "commit", "-m", "accidentally track demo")
        self.run_cli(self.source, "publish")

        returncode, blocked = self.run_cli(client, "update")
        self.assertEqual(returncode, 1)
        self.assertEqual(blocked["code"], "protected_path")
        self.assertEqual(blocked["protected_incoming_paths"], ["demos/published.bin"])
        self.assertFalse((client / "demos" / "published.bin").exists())

    def test_repository_lock_rejects_concurrent_operation(self) -> None:
        self.publish_initial()
        lock_path = self.source / ".git" / "collection-release.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            returncode, blocked = self.run_cli(self.source, "status")
        self.assertEqual(returncode, 1)
        self.assertEqual(blocked["code"], "busy")


if __name__ == "__main__":
    unittest.main()
