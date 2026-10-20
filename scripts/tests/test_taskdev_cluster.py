#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import signal
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "taskdev_cluster.py"
SPEC = importlib.util.spec_from_file_location("taskdev_cluster", SCRIPT)
assert SPEC and SPEC.loader
cluster = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cluster
SPEC.loader.exec_module(cluster)


class HostPolicyTests(unittest.TestCase):
    def test_only_hp13_and_hp14_are_allowed(self) -> None:
        self.assertEqual(cluster.canonical_host("jinshan-h200-ningxia-13"), "hp-13")
        self.assertEqual(cluster.canonical_host("hp-14"), "hp-14")
        with self.assertRaisesRegex(cluster.SchedulerError, "permanently forbidden"):
            cluster.canonical_host("hp-17")
        with self.assertRaisesRegex(cluster.SchedulerError, "allowlist"):
            cluster.canonical_host("hp-12")

    def test_target_hostname_must_match(self) -> None:
        self.assertEqual(
            cluster.assert_running_on("hp-13", "jinshan-h200-ningxia-13"),
            "jinshan-h200-ningxia-13",
        )
        with self.assertRaisesRegex(cluster.SchedulerError, "identity mismatch"):
            cluster.assert_running_on("hp-13", "jinshan-h200-ningxia-14")
        self.assertEqual(
            cluster.assert_running_on("hp-13", "aibox-r28e5d209492-pod"),
            "aibox-r28e5d209492-pod",
        )


class AuditParsingTests(unittest.TestCase):
    def test_parse_gpu_compute_and_meminfo(self) -> None:
        gpu_rows = cluster.parse_gpu_rows(
            "0, GPU-a, 1, 0\n1, GPU-b, 4096, 77\n"
        )
        self.assertEqual(gpu_rows[1], (1, "GPU-b", 4096, 77))
        processes = cluster.parse_compute_rows("GPU-b, 123, python, 4000\n")
        self.assertEqual(processes[0].pid, 123)
        self.assertAlmostEqual(
            cluster.parse_mem_available_gib("MemTotal: 9 kB\nMemAvailable: 1048576 kB\n"),
            1.0,
        )

    def test_driver_only_reservation_is_idle(self) -> None:
        gpu = cluster.classify_gpu(
            index=0,
            uuid="GPU-a",
            memory_used_mib=1,
            utilization_percent=0,
            compute_processes=[],
            driver_reservation_mib=256,
            max_idle_utilization=5,
        )
        self.assertTrue(gpu.available)
        self.assertEqual(gpu.state, "idle")

    def test_unexplained_utilization_fails_closed_as_contradictory(self) -> None:
        gpu = cluster.classify_gpu(
            index=2,
            uuid="GPU-c",
            memory_used_mib=1,
            utilization_percent=100,
            compute_processes=[],
            driver_reservation_mib=256,
            max_idle_utilization=5,
        )
        self.assertFalse(gpu.available)
        self.assertEqual(gpu.state, "contradictory")

    def test_compute_process_is_busy(self) -> None:
        process = cluster.ComputeProcess("GPU-a", 123, "python", 200)
        gpu = cluster.classify_gpu(
            index=0,
            uuid="GPU-a",
            memory_used_mib=200,
            utilization_percent=0,
            compute_processes=[process],
            driver_reservation_mib=256,
            max_idle_utilization=5,
        )
        self.assertEqual(gpu.state, "busy")

    def test_process_on_unknown_gpu_uuid_is_not_silently_ignored(self) -> None:
        gpu = mock.Mock(stdout="0, GPU-a, 1, 0\n")
        compute = mock.Mock(stdout="GPU-unknown, 123, python, 200\n")
        with mock.patch.object(
            cluster, "assert_running_on", return_value="jinshan-h200-ningxia-13"
        ), mock.patch.object(
            cluster, "run_checked", side_effect=[gpu, compute]
        ), mock.patch.object(
            cluster.Path, "read_text", return_value="MemAvailable: 104857600 kB\n"
        ):
            with self.assertRaisesRegex(cluster.SchedulerError, "absent from the GPU query"):
                cluster.audit_local("hp-13")


class LeaseAndCommandTests(unittest.TestCase):
    def test_atomic_per_gpu_lease_and_token_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lease = cluster.acquire_lease(
                root,
                host="hp-13",
                gpu=3,
                run_id="run-a",
                task_id="task-a",
                token="token-a",
            )
            with self.assertRaisesRegex(cluster.SchedulerError, "already exists"):
                cluster.acquire_lease(
                    root,
                    host="hp-13",
                    gpu=3,
                    run_id="run-b",
                    task_id="task-b",
                    token="token-b",
                )
            with self.assertRaisesRegex(cluster.SchedulerError, "another token"):
                cluster.release_lease(lease, "wrong-token")
            self.assertTrue(cluster.release_lease(lease, "token-a"))
            self.assertFalse(lease.exists())

    def test_physical_gpu_is_mapped_to_logical_zero(self) -> None:
        command = cluster.expand_command(
            [
                "python",
                "preview.py",
                "--device",
                "cuda:{logical_gpu}",
                "--physical-gpu",
                "{physical_gpu}",
            ],
            gpu=6,
            task_id="b05",
            worktree=Path("/tmp/worktree"),
            run_dir=Path("/tmp/run"),
        )
        self.assertEqual(command[-3:], ["cuda:0", "--physical-gpu", "6"])
        environment = cluster.clean_job_environment(
            {"CUDA_VISIBLE_DEVICES": "1", "KEEP": "yes"},
            host="hp-14",
            gpu=6,
            run_id="run",
        )
        self.assertNotIn("CUDA_VISIBLE_DEVICES", environment)
        self.assertEqual(environment["TASKDEV_PHYSICAL_GPU"], "6")
        self.assertEqual(environment["TASKDEV_LOGICAL_GPU"], "0")
        self.assertEqual(environment["HANDSON_GPU_DEVICE"], "0")
        self.assertEqual(environment["HANDSON_ISAACLAB_DEVICE"], "cuda:0")
        with self.assertRaisesRegex(cluster.SchedulerError, "must not set"):
            cluster.normalize_command(["CUDA_VISIBLE_DEVICES=0", "python", "x.py"])

    def test_namespace_wrapper_records_physical_gpu(self) -> None:
        with mock.patch.object(cluster.shutil, "which", return_value="/usr/bin/unshare"):
            command = cluster.gpu_namespace_command(
                ["python", "task.py"], physical_gpu=4, run_dir=Path("/runs/a")
            )
        self.assertEqual(
            command[:4],
            [
                "/usr/bin/unshare",
                "--mount",
                "--propagation",
                "private",
            ],
        )
        self.assertIn("_gpu-namespace-exec", command)
        self.assertEqual(command[-2:], ["python", "task.py"])

    def test_task_command_has_a_hard_wall_clock_deadline(self) -> None:
        with mock.patch.object(cluster.shutil, "which", return_value="/usr/bin/timeout"):
            command = cluster.bounded_command(
                ["python", "task.py"], max_runtime_seconds=123.5
            )
        self.assertEqual(
            command[:5],
            [
                "/usr/bin/timeout",
                "--foreground",
                "--signal=TERM",
                "--kill-after=30.0s",
                "123.5s",
            ],
        )
        with self.assertRaisesRegex(cluster.SchedulerError, "must be positive"):
            cluster.bounded_command(["python", "task.py"], max_runtime_seconds=0.0)

    def test_isolation_hides_every_non_selected_gpu(self) -> None:
        fake_stat = types.SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=2002)
        selected = mock.Mock()
        selected.stat.return_value = fake_stat
        nodes = {0: mock.Mock(), 1: mock.Mock(), 2: selected, 3: mock.Mock()}
        with mock.patch.object(cluster, "assert_private_mount_namespace", return_value=777), mock.patch.object(
            cluster, "nvidia_gpu_device_nodes", return_value=nodes
        ), mock.patch.object(cluster, "bind_null_over_device") as hide:
            for index, path in nodes.items():
                if index != 2:
                    path.stat.return_value = types.SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=13)
            with mock.patch.object(
                cluster.os, "major", side_effect=lambda value: {2002: 195, 13: 1}[value]
            ), mock.patch.object(
                cluster.os, "minor", side_effect=lambda value: {2002: 2, 13: 3}[value]
            ):
                mapping = cluster.isolate_physical_gpu(2)
        self.assertEqual(hide.call_count, 3)
        self.assertNotIn(mock.call(selected), hide.call_args_list)
        self.assertEqual(mapping["physical_gpu"], 2)
        self.assertEqual(mapping["logical_gpu"], 0)

    def test_ssh_disables_forwarding_and_quotes_remote_arguments(self) -> None:
        argv = cluster.ssh_command("hp-13", ["python3", "/p/a b.py", "x;y"])
        self.assertIn("ClearAllForwardings=yes", argv)
        self.assertEqual(argv[-2], "hp-13")
        self.assertIn("'/p/a b.py'", argv[-1])
        self.assertIn("'x;y'", argv[-1])


class DryRunAndStopTests(unittest.TestCase):
    def test_zombie_worker_is_not_treated_as_live(self) -> None:
        with mock.patch.object(cluster, "read_process_stat", return_value=("Z", 99)):
            self.assertFalse(cluster.process_exists_with_identity(4321, 99))

    def test_worker_cleanup_terminates_nested_group_members_before_release(self) -> None:
        snapshots = [{5001: 101, 5002: 102}, {5002: 102}, {}]
        with mock.patch.object(cluster.os, "getpgrp", return_value=5000), mock.patch.object(
            cluster.os, "getpid", return_value=5000
        ), mock.patch.object(
            cluster, "live_process_group_members", side_effect=snapshots
        ), mock.patch.object(
            cluster, "signal_verified_group_members"
        ) as signal_members, mock.patch.object(cluster.time, "sleep"):
            cluster.cleanup_owned_process_group(
                5000,
                owner_pid=5000,
                term_seconds=1.0,
                kill_seconds=1.0,
            )
        self.assertEqual(
            signal_members.call_args_list,
            [
                mock.call(5000, {5001: 101, 5002: 102}, signal.SIGTERM),
                mock.call(5000, {5002: 102}, signal.SIGTERM),
            ],
        )

    def test_worker_cleanup_fails_closed_when_members_survive_kill(self) -> None:
        with mock.patch.object(cluster.os, "getpgrp", return_value=5000), mock.patch.object(
            cluster.os, "getpid", return_value=5000
        ), mock.patch.object(
            cluster, "live_process_group_members", return_value={5002: 102}
        ), mock.patch.object(
            cluster, "signal_verified_group_members"
        ), mock.patch.object(
            cluster.time, "monotonic", side_effect=[0.0, 2.0, 2.0, 4.0]
        ):
            with self.assertRaisesRegex(cluster.SchedulerError, "still has live members"):
                cluster.cleanup_owned_process_group(
                    5000,
                    owner_pid=5000,
                    term_seconds=1.0,
                    kill_seconds=1.0,
                )

    def test_launch_is_dry_run_without_explicit_flag(self) -> None:
        parser = cluster.build_parser()
        args = parser.parse_args(
            [
                "launch",
                "--host",
                "hp-13",
                "--gpu",
                "0",
                "--task-id",
                "b05",
                "--repo",
                "/mnt/pfs/repo",
                "--",
                "python",
                "preview.py",
                "--device",
                "cuda:{gpu}",
            ]
        )
        self.assertFalse(args.launch)
        snapshot = {
            "host_memory_ok": True,
            "gpus": [{"index": 0, "state": "idle", "reasons": []}],
        }
        with mock.patch.object(cluster, "audit_remote", return_value=snapshot), mock.patch.object(
            cluster, "remote_json"
        ) as remote_launch:
            plan = cluster.controller_launch(args)
        remote_launch.assert_not_called()
        self.assertEqual(plan["mode"], "dry-run")
        self.assertEqual(plan["command"][-1], "cuda:0")
        self.assertEqual(plan["max_runtime_seconds"], 1800.0)

    def test_stop_signals_only_recorded_process_group(self) -> None:
        metadata = {"status": "running"}
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            run_json = run_dir / "run.json"
            cluster.atomic_write_json(run_json, metadata)
            with mock.patch.object(
                cluster,
                "validate_stop_ownership",
                return_value=(metadata, run_dir / "lease", "token", 4321, 4321, 99),
            ), mock.patch.object(
                cluster,
                "process_exists_with_identity",
                side_effect=[True, False, False, False],
            ), mock.patch.object(cluster.os, "killpg") as kill_group, mock.patch.object(
                cluster, "release_lease"
            ), mock.patch.object(
                cluster, "drain_verified_process_group"
            ) as drain_group:
                cluster.stop_local(run_dir, 0.0)
            kill_group.assert_called_once_with(4321, signal.SIGTERM)
            drain_group.assert_called_once_with(
                4321,
                term_seconds=0.0,
                kill_seconds=5.0,
            )

    def test_stop_drains_recorded_group_before_releasing_orphaned_lease(self) -> None:
        metadata = {"status": "cleanup_failed"}
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            cluster.atomic_write_json(run_dir / "run.json", metadata)
            with mock.patch.object(
                cluster,
                "validate_stop_ownership",
                return_value=(metadata, run_dir / "lease", "token", 4321, 4321, 99),
            ), mock.patch.object(
                cluster, "process_exists_with_identity", return_value=False
            ), mock.patch.object(
                cluster, "drain_verified_process_group"
            ) as drain_group, mock.patch.object(
                cluster, "release_lease"
            ) as release:
                cluster.stop_local(run_dir, 7.0)
            drain_group.assert_called_once_with(
                4321,
                term_seconds=7.0,
                kill_seconds=5.0,
            )
            release.assert_called_once_with(run_dir / "lease", "token")

    def test_stop_rejects_missing_safe_pgid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            lease = run_dir / "lease"
            lease.mkdir()
            metadata = {
                "host": "hp-13",
                "physical_gpu": 0,
                "run_id": "run",
                "lease_token": "token",
                "lease_dir": str(lease),
                "worker_pid": 1,
                "worker_pgid": 1,
                "worker_start_ticks": 0,
            }
            cluster.atomic_write_json(run_dir / "run.json", metadata)
            cluster.atomic_write_json(
                lease / "lease.json",
                {"host": "hp-13", "gpu": 0, "run_id": "run", "token": "token"},
            )
            with mock.patch.object(cluster, "assert_running_on"):
                with self.assertRaisesRegex(cluster.SchedulerError, "safe isolated"):
                    cluster.validate_stop_ownership(run_dir)


if __name__ == "__main__":
    unittest.main()
