#!/usr/bin/env python3
"""Fail-closed one-task/one-physical-GPU scheduler for hp-13 and hp-14.

The public commands are a small SSH controller.  Target-side internal commands
perform the lease, fresh resource audit, worktree creation, process-group
launch, heartbeat, status, and exact process-group stop.  No daemon or network
listener is created.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import dataclasses
import datetime as dt
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ALLOWED_HOSTS = {
    "hp-13": {"hp-13", "jinshan-h200-ningxia-13"},
    "hp-14": {"hp-14", "jinshan-h200-ningxia-14"},
}
DENIED_HOSTS = {"hp-17", "jinshan-h200-ningxia-17"}
KUBERNETES_POD_HOSTNAME = re.compile(r"^aibox-[a-z0-9-]+$")
DEFAULT_DRIVER_RESERVATION_MIB = 256
DEFAULT_MAX_IDLE_UTILIZATION = 5
DEFAULT_MIN_HOST_AVAILABLE_GIB = 64.0
DEFAULT_HEARTBEAT_SECONDS = 5.0
DEFAULT_MAX_RUNTIME_SECONDS = 1800.0
TIMEOUT_KILL_AFTER_SECONDS = 30.0
LOGICAL_GPU_INDEX = 0
NVIDIA_GPU_DEVICE_MAJOR = 195
MS_BIND = 4096
SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ClearAllForwardings=yes",
    "-o",
    "ConnectTimeout=10",
)


class SchedulerError(RuntimeError):
    """Expected safety or runtime failure."""


@dataclasses.dataclass(frozen=True)
class ComputeProcess:
    gpu_uuid: str
    pid: int
    name: str
    used_memory_mib: int | None


@dataclasses.dataclass(frozen=True)
class GpuSnapshot:
    index: int
    uuid: str
    memory_used_mib: int
    utilization_percent: int
    compute_processes: tuple[ComputeProcess, ...]
    state: str
    reasons: tuple[str, ...]

    @property
    def available(self) -> bool:
        return self.state == "idle"


@dataclasses.dataclass(frozen=True)
class HostSnapshot:
    host: str
    observed_hostname: str
    observed_at: str
    mem_available_gib: float
    min_mem_available_gib: float
    host_memory_ok: bool
    gpus: tuple[GpuSnapshot, ...]

    def gpu(self, index: int) -> GpuSnapshot:
        for gpu in self.gpus:
            if gpu.index == index:
                return gpu
        raise SchedulerError(f"GPU {index} was not reported by nvidia-smi on {self.host}")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        os.chmod(temporary, 0o600)
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise SchedulerError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SchedulerError(f"expected a JSON object in {path}")
    return value


def update_json(path: Path, **changes: Any) -> dict[str, Any]:
    payload = load_json(path)
    payload.update(changes)
    atomic_write_json(path, payload)
    return payload


def canonical_host(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in DENIED_HOSTS or normalized.endswith("-17"):
        raise SchedulerError("hp-17 is permanently forbidden for new workloads")
    for canonical, aliases in ALLOWED_HOSTS.items():
        if normalized in aliases:
            return canonical
    raise SchedulerError(
        f"host {value!r} is outside this scheduler's strict hp-13/hp-14 allowlist"
    )


def assert_running_on(requested_host: str, observed_hostname: str | None = None) -> str:
    requested = canonical_host(requested_host)
    observed = (observed_hostname or socket.gethostname().split(".", 1)[0]).lower()
    # The Jinshan SSH aliases currently land in per-node Kubernetes pods whose
    # hostname is an ephemeral ``aibox-*`` pod name, not the hp-* node alias.
    # For those targets the SSH allowlisted alias is the host trust boundary;
    # PID/PGID/start-time checks independently protect stop operations.
    if observed not in ALLOWED_HOSTS[requested] and not KUBERNETES_POD_HOSTNAME.fullmatch(observed):
        raise SchedulerError(
            f"target identity mismatch: requested {requested}, observed {observed}"
        )
    return observed


def run_checked(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(argv),
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or "").strip()
        else:
            detail = str(exc)
        raise SchedulerError(f"command failed: {shlex.join(argv)}: {detail}") from exc


def parse_int(value: str, field: str) -> int:
    cleaned = value.strip()
    if cleaned in {"", "N/A", "[N/A]", "Not Supported"}:
        raise SchedulerError(f"nvidia-smi returned no numeric {field}: {value!r}")
    try:
        return int(cleaned)
    except ValueError as exc:
        raise SchedulerError(f"invalid numeric {field}: {value!r}") from exc


def parse_optional_int(value: str) -> int | None:
    cleaned = value.strip()
    if cleaned in {"", "N/A", "[N/A]", "Not Supported"}:
        return None
    try:
        return int(cleaned)
    except ValueError as exc:
        raise SchedulerError(f"invalid optional integer: {value!r}") from exc


def parse_gpu_rows(output: str) -> list[tuple[int, str, int, int]]:
    rows: list[tuple[int, str, int, int]] = []
    for values in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(values) != 4:
            raise SchedulerError(f"unexpected nvidia-smi GPU row: {values!r}")
        rows.append(
            (
                parse_int(values[0], "GPU index"),
                values[1].strip(),
                parse_int(values[2], "memory.used"),
                parse_int(values[3], "utilization.gpu"),
            )
        )
    if not rows:
        raise SchedulerError("nvidia-smi returned no GPUs")
    indices = [row[0] for row in rows]
    if len(indices) != len(set(indices)):
        raise SchedulerError("nvidia-smi returned duplicate physical GPU indices")
    return rows


def parse_compute_rows(output: str) -> list[ComputeProcess]:
    processes: list[ComputeProcess] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or "No running processes found" in stripped:
            continue
        values = next(csv.reader([line]))
        if len(values) != 4:
            raise SchedulerError(f"unexpected nvidia-smi compute row: {values!r}")
        processes.append(
            ComputeProcess(
                gpu_uuid=values[0].strip(),
                pid=parse_int(values[1], "compute PID"),
                name=values[2].strip(),
                used_memory_mib=parse_optional_int(values[3]),
            )
        )
    return processes


def parse_mem_available_gib(meminfo: str) -> float:
    match = re.search(r"^MemAvailable:\s+(\d+)\s+kB\s*$", meminfo, re.MULTILINE)
    if not match:
        raise SchedulerError("/proc/meminfo does not contain a valid MemAvailable row")
    return int(match.group(1)) / (1024.0 * 1024.0)


def classify_gpu(
    *,
    index: int,
    uuid: str,
    memory_used_mib: int,
    utilization_percent: int,
    compute_processes: Iterable[ComputeProcess],
    driver_reservation_mib: int,
    max_idle_utilization: int,
) -> GpuSnapshot:
    processes = tuple(process for process in compute_processes if process.gpu_uuid == uuid)
    reasons: list[str] = []
    contradictions: list[str] = []
    if processes:
        reasons.append(f"{len(processes)} compute process(es) present")
    if memory_used_mib > driver_reservation_mib:
        reasons.append(
            f"memory.used={memory_used_mib} MiB exceeds idle allowance "
            f"{driver_reservation_mib} MiB"
        )
        if not processes:
            contradictions.append("high GPU memory without a reported compute process")
    if utilization_percent > max_idle_utilization:
        reasons.append(
            f"utilization={utilization_percent}% exceeds idle allowance "
            f"{max_idle_utilization}%"
        )
        if not processes:
            contradictions.append("high GPU utilization without a reported compute process")
    if contradictions:
        return GpuSnapshot(
            index,
            uuid,
            memory_used_mib,
            utilization_percent,
            processes,
            "contradictory",
            tuple(reasons + contradictions),
        )
    if reasons:
        return GpuSnapshot(
            index,
            uuid,
            memory_used_mib,
            utilization_percent,
            processes,
            "busy",
            tuple(reasons),
        )
    return GpuSnapshot(
        index,
        uuid,
        memory_used_mib,
        utilization_percent,
        processes,
        "idle",
        (f"driver-only reservation {memory_used_mib} MiB",),
    )


def audit_local(
    host: str,
    *,
    driver_reservation_mib: int = DEFAULT_DRIVER_RESERVATION_MIB,
    max_idle_utilization: int = DEFAULT_MAX_IDLE_UTILIZATION,
    min_host_available_gib: float = DEFAULT_MIN_HOST_AVAILABLE_GIB,
    observed_hostname: str | None = None,
) -> HostSnapshot:
    canonical = canonical_host(host)
    observed = assert_running_on(canonical, observed_hostname)
    gpu_result = run_checked(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        )
    )
    compute_result = run_checked(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        )
    )
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError as exc:
        raise SchedulerError(f"cannot read /proc/meminfo: {exc}") from exc
    memory_gib = parse_mem_available_gib(meminfo)
    compute_processes = parse_compute_rows(compute_result.stdout)
    gpu_rows = parse_gpu_rows(gpu_result.stdout)
    reported_uuids = {row[1] for row in gpu_rows}
    unknown_compute_uuids = sorted(
        {process.gpu_uuid for process in compute_processes} - reported_uuids
    )
    if unknown_compute_uuids:
        raise SchedulerError(
            "compute-process query referenced GPUs absent from the GPU query: "
            + ", ".join(unknown_compute_uuids)
        )
    gpus = tuple(
        classify_gpu(
            index=index,
            uuid=uuid,
            memory_used_mib=memory,
            utilization_percent=utilization,
            compute_processes=compute_processes,
            driver_reservation_mib=driver_reservation_mib,
            max_idle_utilization=max_idle_utilization,
        )
        for index, uuid, memory, utilization in gpu_rows
    )
    return HostSnapshot(
        host=canonical,
        observed_hostname=observed,
        observed_at=utc_now(),
        mem_available_gib=memory_gib,
        min_mem_available_gib=min_host_available_gib,
        host_memory_ok=memory_gib >= min_host_available_gib,
        gpus=gpus,
    )


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    return value


def ssh_command(host: str, remote_argv: Sequence[str]) -> list[str]:
    canonical = canonical_host(host)
    return ["ssh", *SSH_OPTIONS, canonical, shlex.join(remote_argv)]


def remote_json(host: str, remote_argv: Sequence[str], *, timeout: float = 60.0) -> dict[str, Any]:
    result = run_checked(ssh_command(host, remote_argv), timeout=timeout)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SchedulerError(
            f"target returned invalid JSON: {result.stdout[-500:]!r}"
        ) from exc
    if not isinstance(value, dict):
        raise SchedulerError("target response is not a JSON object")
    return value


def audit_remote(
    host: str,
    remote_script: Path,
    *,
    driver_reservation_mib: int,
    max_idle_utilization: int,
    min_host_available_gib: float,
) -> dict[str, Any]:
    return remote_json(
        host,
        (
            "python3",
            str(remote_script),
            "_audit-local",
            "--host",
            canonical_host(host),
            "--driver-reservation-mib",
            str(driver_reservation_mib),
            "--max-idle-utilization",
            str(max_idle_utilization),
            "--min-host-available-gib",
            str(min_host_available_gib),
        ),
    )


def ensure_snapshot_available(snapshot: Mapping[str, Any], gpu_index: int) -> Mapping[str, Any]:
    if not snapshot.get("host_memory_ok"):
        raise SchedulerError(
            f"host has only {snapshot.get('mem_available_gib')} GiB MemAvailable"
        )
    matches = [gpu for gpu in snapshot.get("gpus", []) if gpu.get("index") == gpu_index]
    if len(matches) != 1:
        raise SchedulerError(f"physical GPU {gpu_index} is absent or ambiguously reported")
    gpu = matches[0]
    if gpu.get("state") != "idle":
        reasons = "; ".join(str(item) for item in gpu.get("reasons", []))
        raise SchedulerError(
            f"physical GPU {gpu_index} is {gpu.get('state')}: {reasons or 'no reason'}"
        )
    return gpu


def task_slug(task_id: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", task_id.strip()).strip("-.")
    if not value or len(value) > 80:
        raise SchedulerError("task id must yield a non-empty slug no longer than 80 characters")
    return value


def make_run_id(task_id: str) -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{task_slug(task_id)}-{timestamp}-{secrets.token_hex(4)}"


def normalize_command(command: Sequence[str]) -> list[str]:
    normalized = list(command)
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    if not normalized:
        raise SchedulerError("a command is required after --")
    if any("CUDA_VISIBLE_DEVICES" in token for token in normalized):
        raise SchedulerError(
            "command must not set CUDA_VISIBLE_DEVICES; use {logical_gpu} for runtime devices"
        )
    return normalized


def nvidia_gpu_device_nodes(dev_root: Path = Path("/dev")) -> dict[int, Path]:
    """Return physical NVIDIA GPU character devices, excluding control nodes."""

    nodes: dict[int, Path] = {}
    try:
        candidates = list(dev_root.glob("nvidia[0-9]*"))
    except OSError as exc:
        raise SchedulerError(f"cannot enumerate NVIDIA device nodes in {dev_root}: {exc}") from exc
    for path in candidates:
        match = re.fullmatch(r"nvidia([0-9]+)", path.name)
        if not match:
            continue
        try:
            metadata = path.stat()
        except OSError as exc:
            raise SchedulerError(f"cannot stat NVIDIA device node {path}: {exc}") from exc
        if not stat.S_ISCHR(metadata.st_mode) or os.major(metadata.st_rdev) != NVIDIA_GPU_DEVICE_MAJOR:
            raise SchedulerError(f"unexpected NVIDIA GPU device node identity: {path}")
        index = int(match.group(1))
        if os.minor(metadata.st_rdev) != index:
            raise SchedulerError(
                f"NVIDIA GPU device node minor mismatch: {path} has {os.minor(metadata.st_rdev)}"
            )
        nodes[index] = path
    if not nodes:
        raise SchedulerError(f"no NVIDIA GPU character devices found in {dev_root}")
    return nodes


def assert_private_mount_namespace() -> int:
    """Fail closed unless the current process has its own mount namespace."""

    try:
        current = os.stat("/proc/self/ns/mnt").st_ino
        initial = os.stat("/proc/1/ns/mnt").st_ino
    except OSError as exc:
        raise SchedulerError(f"cannot verify mount namespace identity: {exc}") from exc
    if current == initial:
        raise SchedulerError("GPU isolation must run inside a private mount namespace")
    return current


def bind_null_over_device(path: Path) -> None:
    """Hide one GPU node in the current private mount namespace."""

    libc = ctypes.CDLL(None, use_errno=True)
    mount = libc.mount
    mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_void_p]
    mount.restype = ctypes.c_int
    result = mount(b"/dev/null", os.fsencode(path), None, MS_BIND, None)
    if result != 0:
        error_number = ctypes.get_errno()
        raise SchedulerError(
            f"cannot hide non-selected GPU node {path}: {os.strerror(error_number)}"
        )


def isolate_physical_gpu(physical_gpu: int, dev_root: Path = Path("/dev")) -> dict[str, Any]:
    """Expose one physical GPU so CUDA and Vulkan both enumerate it as device zero."""

    namespace_inode = assert_private_mount_namespace()
    nodes = nvidia_gpu_device_nodes(dev_root)
    if physical_gpu not in nodes:
        raise SchedulerError(f"physical GPU {physical_gpu} has no validated device node")
    hidden: list[str] = []
    for index, path in sorted(nodes.items()):
        if index == physical_gpu:
            continue
        bind_null_over_device(path)
        try:
            metadata = path.stat()
        except OSError as exc:
            raise SchedulerError(f"cannot verify hidden GPU node {path}: {exc}") from exc
        if not stat.S_ISCHR(metadata.st_mode) or (
            os.major(metadata.st_rdev), os.minor(metadata.st_rdev)
        ) != (1, 3):
            raise SchedulerError(f"GPU node isolation verification failed for {path}")
        hidden.append(str(path))
    selected = nodes[physical_gpu]
    selected_metadata = selected.stat()
    return {
        "schema_version": 1,
        "physical_gpu": physical_gpu,
        "logical_gpu": LOGICAL_GPU_INDEX,
        "mount_namespace_inode": namespace_inode,
        "selected_device": str(selected),
        "selected_rdev": [os.major(selected_metadata.st_rdev), os.minor(selected_metadata.st_rdev)],
        "hidden_devices": hidden,
        "isolated_at": utc_now(),
    }


def gpu_namespace_command(
    command: Sequence[str], *, physical_gpu: int, run_dir: Path
) -> list[str]:
    unshare = shutil.which("unshare")
    if not unshare:
        raise SchedulerError("unshare is required for fail-closed physical GPU isolation")
    return [
        unshare,
        "--mount",
        "--propagation",
        "private",
        sys.executable,
        str(Path(__file__).resolve()),
        "_gpu-namespace-exec",
        "--physical-gpu",
        str(physical_gpu),
        "--run-dir",
        str(run_dir),
        "--",
        *command,
    ]


def bounded_command(command: Sequence[str], *, max_runtime_seconds: float) -> list[str]:
    """Wrap a task in a wall-clock deadline that terminates its child tree."""

    if not (max_runtime_seconds > 0.0):
        raise SchedulerError("max runtime seconds must be positive")
    timeout = shutil.which("timeout")
    if not timeout:
        raise SchedulerError("GNU timeout is required for bounded task execution")
    return [
        timeout,
        # GNU timeout otherwise creates a child process group.  That would
        # let the scheduler's exact worker-PGID stop miss the real task.
        "--foreground",
        "--signal=TERM",
        f"--kill-after={TIMEOUT_KILL_AFTER_SECONDS}s",
        f"{max_runtime_seconds}s",
        *command,
    ]


def expand_command(
    command: Sequence[str],
    *,
    gpu: int,
    task_id: str,
    worktree: Path,
    run_dir: Path,
) -> list[str]:
    substitutions = {
        "gpu": str(gpu),
        "physical_gpu": str(gpu),
        "logical_gpu": str(LOGICAL_GPU_INDEX),
        "task_id": task_id,
        "worktree": str(worktree),
        "run_dir": str(run_dir),
    }
    expanded: list[str] = []
    for token in normalize_command(command):
        try:
            expanded.append(token.format_map(substitutions))
        except (KeyError, ValueError) as exc:
            raise SchedulerError(f"invalid command template token {token!r}: {exc}") from exc
    return expanded


def lease_path(state_root: Path, host: str, gpu: int) -> Path:
    return state_root / "leases" / canonical_host(host) / f"gpu-{gpu}"


def acquire_lease(
    state_root: Path,
    *,
    host: str,
    gpu: int,
    run_id: str,
    task_id: str,
    token: str,
) -> Path:
    path = lease_path(state_root, host, gpu)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as exc:
        owner = path / "lease.json"
        detail = load_json(owner) if owner.exists() else {"state": "incomplete lease"}
        raise SchedulerError(
            f"GPU lease already exists at {path}: {json.dumps(detail, sort_keys=True)}"
        ) from exc
    try:
        atomic_write_json(
            path / "lease.json",
            {
                "schema_version": 1,
                "host": canonical_host(host),
                "gpu": gpu,
                "run_id": run_id,
                "task_id": task_id,
                "token": token,
                "acquired_at": utc_now(),
                "controller_pid": os.getpid(),
            },
        )
    except BaseException:
        try:
            path.rmdir()
        except OSError:
            pass
        raise
    return path


def release_lease(path: Path, token: str) -> bool:
    metadata_path = path / "lease.json"
    if not path.exists():
        return False
    metadata = load_json(metadata_path)
    if not secrets.compare_digest(str(metadata.get("token", "")), token):
        raise SchedulerError(f"refusing to release a lease owned by another token: {path}")
    metadata_path.unlink()
    path.rmdir()
    return True


def read_process_stat(pid: int) -> tuple[str, int]:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError as exc:
        raise SchedulerError(f"cannot read process identity for PID {pid}: {exc}") from exc
    closing = stat.rfind(")")
    fields_after_comm = stat[closing + 2 :].split()
    if closing < 0 or len(fields_after_comm) < 20:
        raise SchedulerError(f"unexpected /proc/{pid}/stat format")
    return fields_after_comm[0], int(fields_after_comm[19])


def read_process_group_identity(pid: int) -> tuple[str, int, int]:
    """Return process state, process-group id, and immutable start ticks."""

    try:
        process_stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError as exc:
        raise SchedulerError(f"cannot read process identity for PID {pid}: {exc}") from exc
    closing = process_stat.rfind(")")
    fields_after_comm = process_stat[closing + 2 :].split()
    if closing < 0 or len(fields_after_comm) < 20:
        raise SchedulerError(f"unexpected /proc/{pid}/stat format")
    return (
        fields_after_comm[0],
        int(fields_after_comm[2]),
        int(fields_after_comm[19]),
    )


def read_process_start_ticks(pid: int) -> int:
    return read_process_stat(pid)[1]


def process_exists_with_identity(pid: int, start_ticks: int) -> bool:
    try:
        state, observed_start_ticks = read_process_stat(pid)
        return state != "Z" and observed_start_ticks == start_ticks
    except SchedulerError:
        return False


def live_process_group_members(pgid: int, *, exclude_pid: int | None = None) -> dict[int, int]:
    """Snapshot non-zombie members of one scheduler-owned process group."""

    if pgid <= 1:
        raise SchedulerError(f"refusing to inspect unsafe process group {pgid}")
    members: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == exclude_pid:
            continue
        try:
            state, observed_pgid, start_ticks = read_process_group_identity(pid)
        except SchedulerError:
            continue
        if observed_pgid == pgid and state != "Z":
            members[pid] = start_ticks
    return members


def signal_verified_group_members(pgid: int, members: Mapping[int, int], signum: int) -> None:
    """Signal only PIDs whose start time and PGID still match the snapshot."""

    for pid, start_ticks in sorted(members.items()):
        if not process_exists_with_identity(pid, start_ticks):
            continue
        try:
            observed_pgid = os.getpgid(pid)
        except ProcessLookupError:
            continue
        except OSError as exc:
            raise SchedulerError(f"cannot revalidate PID {pid} before cleanup: {exc}") from exc
        if observed_pgid != pgid:
            raise SchedulerError(
                f"refusing to signal PID {pid}: process group changed from {pgid} to {observed_pgid}"
            )
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            continue


def drain_verified_process_group(
    pgid: int,
    *,
    exclude_pid: int | None = None,
    term_seconds: float = 5.0,
    kill_seconds: float = 5.0,
) -> None:
    """Drain a recorded PGID using PID/start-time revalidation for every signal."""

    term_deadline = time.monotonic() + max(0.0, term_seconds)
    members = live_process_group_members(pgid, exclude_pid=exclude_pid)
    while members and time.monotonic() < term_deadline:
        signal_verified_group_members(pgid, members, signal.SIGTERM)
        time.sleep(0.05)
        members = live_process_group_members(pgid, exclude_pid=exclude_pid)
    kill_deadline = time.monotonic() + max(0.0, kill_seconds)
    while members and time.monotonic() < kill_deadline:
        signal_verified_group_members(pgid, members, signal.SIGKILL)
        time.sleep(0.05)
        members = live_process_group_members(pgid, exclude_pid=exclude_pid)
    if members:
        raise SchedulerError(
            f"owned process group {pgid} still has live members after cleanup: {sorted(members)}"
        )


def cleanup_owned_process_group(
    pgid: int,
    *,
    owner_pid: int,
    term_seconds: float = 5.0,
    kill_seconds: float = 5.0,
) -> None:
    """Drain every live task process before its GPU lease can be released."""

    if pgid != os.getpgrp() or owner_pid != os.getpid() or owner_pid != pgid:
        raise SchedulerError("worker cannot prove ownership of its cleanup process group")
    drain_verified_process_group(
        pgid,
        exclude_pid=owner_pid,
        term_seconds=term_seconds,
        kill_seconds=kill_seconds,
    )


def create_worktree(repo: Path, base_ref: str, branch: str, worktree: Path) -> str:
    if not repo.is_absolute() or not worktree.is_absolute():
        raise SchedulerError("repo and worktree paths must be absolute")
    if worktree.exists():
        raise SchedulerError(f"worktree path already exists: {worktree}")
    commit = run_checked(
        ("git", "-C", str(repo), "rev-parse", "--verify", f"{base_ref}^{{commit}}")
    ).stdout.strip()
    run_checked(("git", "check-ref-format", "--branch", branch))
    worktree.parent.mkdir(parents=True, exist_ok=True)
    run_checked(
        (
            "git",
            "-C",
            str(repo),
            "-c",
            "core.hooksPath=/dev/null",
            "worktree",
            "add",
            "-b",
            branch,
            str(worktree),
            commit,
        ),
        timeout=300.0,
    )
    return commit


def clean_job_environment(base: Mapping[str, str], *, host: str, gpu: int, run_id: str) -> dict[str, str]:
    environment = dict(base)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update(
        {
            "TASKDEV_HOST": canonical_host(host),
            "TASKDEV_PHYSICAL_GPU": str(gpu),
            "TASKDEV_LOGICAL_GPU": str(LOGICAL_GPU_INDEX),
            "TASKDEV_RUN_ID": run_id,
            "HANDSON_GPU_DEVICE": str(LOGICAL_GPU_INDEX),
            "HANDSON_ISAACLAB_DEVICE": f"cuda:{LOGICAL_GPU_INDEX}",
            "MUJOCO_EGL_DEVICE_ID": str(LOGICAL_GPU_INDEX),
            "HANDSON_VIDEO_GPU": str(LOGICAL_GPU_INDEX),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def launch_local(args: argparse.Namespace) -> dict[str, Any]:
    host = canonical_host(args.host)
    observed = assert_running_on(host)
    gpu = int(args.gpu)
    if gpu < 0:
        raise SchedulerError("GPU index must be non-negative")
    repo = Path(args.repo).resolve()
    state_root = Path(args.state_root).resolve()
    run_root = Path(args.run_root).resolve()
    worktree_root = Path(args.worktree_root).resolve()
    run_id = args.run_id
    token = args.token
    run_dir = run_root / run_id
    worktree = worktree_root / run_id
    lease = acquire_lease(
        state_root,
        host=host,
        gpu=gpu,
        run_id=run_id,
        task_id=args.task_id,
        token=token,
    )
    worktree_created = False
    try:
        # This is intentionally inside the atomic per-(host, GPU) lease.  No
        # worktree or child process is created from an older controller audit.
        snapshot = audit_local(
            host,
            driver_reservation_mib=args.driver_reservation_mib,
            max_idle_utilization=args.max_idle_utilization,
            min_host_available_gib=args.min_host_available_gib,
            observed_hostname=observed,
        )
        snapshot_payload = jsonable(snapshot)
        ensure_snapshot_available(snapshot_payload, gpu)
        branch = args.branch
        commit = create_worktree(repo, args.base_ref, branch, worktree)
        worktree_created = True
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise SchedulerError(f"run directory already exists: {run_dir}") from exc
        command = expand_command(
            args.command,
            gpu=gpu,
            task_id=args.task_id,
            worktree=worktree,
            run_dir=run_dir,
        )
        run_json = run_dir / "run.json"
        gate = run_dir / ".start-gate"
        payload: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "task_id": args.task_id,
            "host": host,
            "observed_hostname": observed,
            "physical_gpu": gpu,
            "logical_gpu": LOGICAL_GPU_INDEX,
            "gpu_isolation": "private-mount-namespace",
            "max_runtime_seconds": float(args.max_runtime_seconds),
            "repo": str(repo),
            "base_ref": args.base_ref,
            "commit": commit,
            "branch": branch,
            "worktree": str(worktree),
            "run_dir": str(run_dir),
            "lease_dir": str(lease),
            "lease_token": token,
            "command": command,
            "audit": snapshot_payload,
            "status": "starting",
            "created_at": utc_now(),
            "heartbeat_at": None,
            "worker_pid": None,
            "worker_pgid": None,
            "worker_start_ticks": None,
            "actual_pid": None,
            "exit_code": None,
        }
        atomic_write_json(run_json, payload)
        worker_log = (run_dir / "worker.log").open("ab", buffering=0)
        try:
            worker = subprocess.Popen(
                (
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "_worker",
                    "--run-dir",
                    str(run_dir),
                    "--lease-dir",
                    str(lease),
                    "--token",
                    token,
                    "--heartbeat-seconds",
                    str(args.heartbeat_seconds),
                ),
                cwd=str(worktree),
                stdin=subprocess.DEVNULL,
                stdout=worker_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            worker_log.close()
        pgid = os.getpgid(worker.pid)
        if pgid != worker.pid:
            raise SchedulerError(
                f"worker did not receive an isolated process group: pid={worker.pid}, pgid={pgid}"
            )
        start_ticks = read_process_start_ticks(worker.pid)
        payload.update(
            {
                "worker_pid": worker.pid,
                "worker_pgid": pgid,
                "worker_start_ticks": start_ticks,
                "status": "launching",
            }
        )
        atomic_write_json(run_json, payload)
        gate.touch(exist_ok=False)
        return payload
    except BaseException:
        # A successfully started worker owns cleanup after the start gate.  All
        # failures before that point are owned by this transaction.
        if not (run_dir / ".start-gate").exists():
            try:
                release_lease(lease, token)
            except Exception:
                pass
        raise


def heartbeat(run_dir: Path, status: str, **extra: Any) -> None:
    now = utc_now()
    atomic_write_json(
        run_dir / "heartbeat.json",
        {"observed_at": now, "status": status, **extra},
    )
    update_json(run_dir / "run.json", heartbeat_at=now, status=status, **extra)


def worker_main(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    lease = Path(args.lease_dir).resolve()
    token = args.token
    metadata = load_json(run_dir / "run.json")
    lease_metadata = load_json(lease / "lease.json")
    if not secrets.compare_digest(str(metadata.get("lease_token", "")), token):
        raise SchedulerError("worker token does not match run.json")
    if not secrets.compare_digest(str(lease_metadata.get("token", "")), token):
        raise SchedulerError("worker token does not match the GPU lease")
    gate = run_dir / ".start-gate"
    deadline = time.monotonic() + 30.0
    while not gate.exists():
        if time.monotonic() >= deadline:
            raise SchedulerError("launch metadata gate was not opened within 30 seconds")
        time.sleep(0.05)
    metadata = load_json(run_dir / "run.json")
    if metadata.get("worker_pid") != os.getpid() or metadata.get("worker_pgid") != os.getpgrp():
        raise SchedulerError("worker PID/PGID does not match launch metadata")
    environment = clean_job_environment(
        os.environ,
        host=str(metadata["host"]),
        gpu=int(metadata["physical_gpu"]),
        run_id=str(metadata["run_id"]),
    )
    command = [str(item) for item in metadata["command"]]
    isolated_command = gpu_namespace_command(
        command,
        physical_gpu=int(metadata["physical_gpu"]),
        run_dir=run_dir,
    )
    task_command = bounded_command(
        isolated_command,
        max_runtime_seconds=float(metadata["max_runtime_seconds"]),
    )
    output = (run_dir / "stdout.log").open("ab", buffering=0)
    stopping = False
    cleanup_error: Exception | None = None

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        process = subprocess.Popen(
            task_command,
            cwd=str(metadata["worktree"]),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=False,
            close_fds=True,
        )
        heartbeat(run_dir, "running", actual_pid=process.pid)
        while process.poll() is None:
            if stopping:
                # The stop command signals this exact process group, so the
                # child has already received SIGTERM alongside the worker.
                heartbeat(run_dir, "stopping", actual_pid=process.pid)
            else:
                heartbeat(run_dir, "running", actual_pid=process.pid)
            time.sleep(max(0.25, float(args.heartbeat_seconds)))
        exit_code = int(process.returncode)
        final_status = (
            "stopped"
            if stopping
            else (
                "succeeded"
                if exit_code == 0
                else ("timed_out" if exit_code == 124 else "failed")
            )
        )
        heartbeat(run_dir, final_status, actual_pid=process.pid, exit_code=exit_code)
        return exit_code
    except BaseException as exc:
        heartbeat(run_dir, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        output.close()
        try:
            cleanup_owned_process_group(
                os.getpgrp(),
                owner_pid=os.getpid(),
            )
            update_json(run_dir / "run.json", descendants_cleaned_at=utc_now())
        except Exception as exc:
            cleanup_error = exc
            try:
                update_json(
                    run_dir / "run.json",
                    status="cleanup_failed",
                    cleanup_error=f"{type(exc).__name__}: {exc}",
                    lease_retained=True,
                )
            except Exception:
                pass
        if cleanup_error is None:
            try:
                release_lease(lease, token)
            except Exception as release_exc:
                try:
                    update_json(run_dir / "run.json", lease_release_error=str(release_exc))
                except Exception:
                    pass
        else:
            raise cleanup_error


def gpu_namespace_exec(args: argparse.Namespace) -> int:
    """Isolate one physical GPU, record the mapping, then replace this process."""

    physical_gpu = int(args.physical_gpu)
    if physical_gpu < 0:
        raise SchedulerError("physical GPU index must be non-negative")
    command = normalize_command(args.command)
    run_dir = Path(args.run_dir).resolve()
    mapping = isolate_physical_gpu(physical_gpu)
    atomic_write_json(run_dir / "gpu_namespace.json", mapping)
    environment = clean_job_environment(
        os.environ,
        host=str(os.environ.get("TASKDEV_HOST", "")),
        gpu=physical_gpu,
        run_id=str(os.environ.get("TASKDEV_RUN_ID", "")),
    )
    try:
        os.execvpe(command[0], command, environment)
    except OSError as exc:
        raise SchedulerError(f"cannot exec isolated task command {command[0]!r}: {exc}") from exc


def validate_stop_ownership(run_dir: Path) -> tuple[dict[str, Any], Path, str, int, int, int]:
    metadata = load_json(run_dir / "run.json")
    assert_running_on(str(metadata.get("host", "")))
    lease = Path(str(metadata.get("lease_dir", ""))).resolve()
    token = str(metadata.get("lease_token", ""))
    lease_metadata = load_json(lease / "lease.json")
    for key in ("host", "gpu", "run_id", "token"):
        expected = metadata.get("physical_gpu") if key == "gpu" else metadata.get(
            "lease_token" if key == "token" else key
        )
        if lease_metadata.get(key) != expected:
            raise SchedulerError(f"lease/run ownership mismatch for {key}")
    pid = int(metadata.get("worker_pid") or 0)
    pgid = int(metadata.get("worker_pgid") or 0)
    start_ticks = int(metadata.get("worker_start_ticks") or 0)
    if pid <= 1 or pgid <= 1 or start_ticks <= 0 or pid != pgid:
        raise SchedulerError("run.json does not contain a safe isolated worker PID/PGID")
    if process_exists_with_identity(pid, start_ticks):
        try:
            actual_pgid = os.getpgid(pid)
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        except OSError as exc:
            raise SchedulerError(f"cannot validate worker identity: {exc}") from exc
        if actual_pgid != pgid:
            raise SchedulerError("worker process group no longer matches run.json")
        if b"_worker" not in cmdline or str(run_dir).encode() not in cmdline:
            raise SchedulerError("PID identity does not match this scheduler worker")
    return metadata, lease, token, pid, pgid, start_ticks


def stop_local(run_dir: Path, timeout_seconds: float) -> dict[str, Any]:
    metadata, lease, token, pid, pgid, start_ticks = validate_stop_ownership(run_dir)
    if not process_exists_with_identity(pid, start_ticks):
        drain_verified_process_group(
            pgid,
            term_seconds=max(0.0, timeout_seconds),
            kill_seconds=5.0,
        )
        release_lease(lease, token)
        return update_json(
            run_dir / "run.json",
            status="not-running",
            stopped_at=utc_now(),
        )
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while process_exists_with_identity(pid, start_ticks) and time.monotonic() < deadline:
        time.sleep(0.1)
    if process_exists_with_identity(pid, start_ticks):
        os.killpg(pgid, signal.SIGKILL)
        kill_deadline = time.monotonic() + 5.0
        while process_exists_with_identity(pid, start_ticks) and time.monotonic() < kill_deadline:
            time.sleep(0.05)
    if process_exists_with_identity(pid, start_ticks):
        raise SchedulerError(f"exact process group {pgid} did not exit")
    drain_verified_process_group(
        pgid,
        term_seconds=0.0,
        kill_seconds=5.0,
    )
    if lease.exists():
        release_lease(lease, token)
    return update_json(
        run_dir / "run.json",
        status="stopped",
        stopped_at=utc_now(),
    )


def add_threshold_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--driver-reservation-mib",
        type=int,
        default=DEFAULT_DRIVER_RESERVATION_MIB,
    )
    parser.add_argument(
        "--max-idle-utilization",
        type=int,
        default=DEFAULT_MAX_IDLE_UTILIZATION,
    )
    parser.add_argument(
        "--min-host-available-gib",
        type=float,
        default=DEFAULT_MIN_HOST_AVAILABLE_GIB,
    )


def add_remote_script_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--remote-script",
        type=Path,
        default=Path(os.environ.get("TASKDEV_REMOTE_SCRIPT", Path(__file__).resolve())),
        help="Absolute shared-PFS path to this script on hp-13/hp-14.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    audit = subparsers.add_parser("audit", help="Read-only live resource audit.")
    audit.add_argument("--hosts", nargs="+", default=["hp-13", "hp-14"])
    add_remote_script_argument(audit)
    add_threshold_arguments(audit)

    launch = subparsers.add_parser("launch", help="Plan by default; execute only with --launch.")
    launch.add_argument("--host", required=True)
    launch.add_argument("--gpu", required=True, type=int)
    launch.add_argument("--task-id", required=True)
    launch.add_argument("--repo", required=True, type=Path)
    launch.add_argument("--base-ref", default="remote-task-dev-latest")
    launch.add_argument("--state-root", type=Path)
    launch.add_argument("--run-root", type=Path)
    launch.add_argument("--worktree-root", type=Path)
    launch.add_argument("--branch")
    launch.add_argument("--heartbeat-seconds", type=float, default=DEFAULT_HEARTBEAT_SECONDS)
    launch.add_argument("--max-runtime-seconds", type=float, default=DEFAULT_MAX_RUNTIME_SECONDS)
    launch.add_argument("--launch", action="store_true", help="Perform the launch.")
    add_remote_script_argument(launch)
    add_threshold_arguments(launch)
    launch.add_argument("command", nargs=argparse.REMAINDER)

    status = subparsers.add_parser("status", help="Read one run's metadata.")
    status.add_argument("--host", required=True)
    status.add_argument("--run-dir", required=True, type=Path)
    add_remote_script_argument(status)

    stop = subparsers.add_parser("stop", help="Stop only the run's owned process group.")
    stop.add_argument("--host", required=True)
    stop.add_argument("--run-dir", required=True, type=Path)
    stop.add_argument("--timeout-seconds", type=float, default=15.0)
    add_remote_script_argument(stop)

    local_audit = subparsers.add_parser("_audit-local")
    local_audit.add_argument("--host", required=True)
    add_threshold_arguments(local_audit)

    local_launch = subparsers.add_parser("_launch-local")
    local_launch.add_argument("--host", required=True)
    local_launch.add_argument("--gpu", required=True, type=int)
    local_launch.add_argument("--task-id", required=True)
    local_launch.add_argument("--repo", required=True)
    local_launch.add_argument("--base-ref", required=True)
    local_launch.add_argument("--state-root", required=True)
    local_launch.add_argument("--run-root", required=True)
    local_launch.add_argument("--worktree-root", required=True)
    local_launch.add_argument("--branch", required=True)
    local_launch.add_argument("--run-id", required=True)
    local_launch.add_argument("--token", required=True)
    local_launch.add_argument("--heartbeat-seconds", type=float, required=True)
    local_launch.add_argument("--max-runtime-seconds", type=float, required=True)
    add_threshold_arguments(local_launch)
    local_launch.add_argument("command", nargs=argparse.REMAINDER)

    worker = subparsers.add_parser("_worker")
    worker.add_argument("--run-dir", required=True)
    worker.add_argument("--lease-dir", required=True)
    worker.add_argument("--token", required=True)
    worker.add_argument("--heartbeat-seconds", type=float, required=True)

    namespace_exec = subparsers.add_parser("_gpu-namespace-exec")
    namespace_exec.add_argument("--physical-gpu", required=True, type=int)
    namespace_exec.add_argument("--run-dir", required=True, type=Path)
    namespace_exec.add_argument("command", nargs=argparse.REMAINDER)

    local_status = subparsers.add_parser("_status-local")
    local_status.add_argument("--host", required=True)
    local_status.add_argument("--run-dir", required=True, type=Path)

    local_stop = subparsers.add_parser("_stop-local")
    local_stop.add_argument("--host", required=True)
    local_stop.add_argument("--run-dir", required=True, type=Path)
    local_stop.add_argument("--timeout-seconds", type=float, required=True)
    return parser


def resolved_roots(args: argparse.Namespace, run_id: str) -> tuple[Path, Path, Path, Path, str]:
    repo = args.repo.resolve()
    parent = repo.parent
    state_root = (args.state_root or parent / ".handson-taskdev-cluster").resolve()
    run_root = (args.run_root or parent / "handson-taskdev-runs").resolve()
    worktree_root = (args.worktree_root or parent / "handson-taskdev-worktrees").resolve()
    branch = args.branch or f"taskdev/run/{task_slug(args.task_id)}/{run_id}"
    return repo, state_root, run_root, worktree_root, branch


def controller_launch(args: argparse.Namespace) -> dict[str, Any]:
    host = canonical_host(args.host)
    if args.gpu < 0:
        raise SchedulerError("GPU index must be non-negative")
    command = normalize_command(args.command)
    run_id = make_run_id(args.task_id)
    token = secrets.token_hex(32)
    repo, state_root, run_root, worktree_root, branch = resolved_roots(args, run_id)
    run_dir = run_root / run_id
    worktree = worktree_root / run_id
    # A controller audit gives useful dry-run feedback.  A real launch never
    # trusts it: _launch-local acquires the lease and audits again.
    snapshot = audit_remote(
        host,
        args.remote_script,
        driver_reservation_mib=args.driver_reservation_mib,
        max_idle_utilization=args.max_idle_utilization,
        min_host_available_gib=args.min_host_available_gib,
    )
    ensure_snapshot_available(snapshot, args.gpu)
    plan = {
        "mode": "launch" if args.launch else "dry-run",
        "host": host,
        "physical_gpu": args.gpu,
        "max_runtime_seconds": float(args.max_runtime_seconds),
        "task_id": args.task_id,
        "run_id": run_id,
        "repo": str(repo),
        "base_ref": args.base_ref,
        "branch": branch,
        "worktree": str(worktree),
        "run_dir": str(run_dir),
        "command": expand_command(
            command,
            gpu=args.gpu,
            task_id=args.task_id,
            worktree=worktree,
            run_dir=run_dir,
        ),
        "controller_audit": snapshot,
    }
    if not args.launch:
        return plan
    remote_argv = [
        "python3",
        str(args.remote_script),
        "_launch-local",
        "--host",
        host,
        "--gpu",
        str(args.gpu),
        "--task-id",
        args.task_id,
        "--repo",
        str(repo),
        "--base-ref",
        args.base_ref,
        "--state-root",
        str(state_root),
        "--run-root",
        str(run_root),
        "--worktree-root",
        str(worktree_root),
        "--branch",
        branch,
        "--run-id",
        run_id,
        "--token",
        token,
        "--heartbeat-seconds",
        str(args.heartbeat_seconds),
        "--max-runtime-seconds",
        str(args.max_runtime_seconds),
        "--driver-reservation-mib",
        str(args.driver_reservation_mib),
        "--max-idle-utilization",
        str(args.max_idle_utilization),
        "--min-host-available-gib",
        str(args.min_host_available_gib),
        "--",
        *command,
    ]
    return remote_json(host, remote_argv, timeout=360.0)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.subcommand == "audit":
            payload = {
                "schema_version": 1,
                "snapshots": [
                    audit_remote(
                        host,
                        args.remote_script,
                        driver_reservation_mib=args.driver_reservation_mib,
                        max_idle_utilization=args.max_idle_utilization,
                        min_host_available_gib=args.min_host_available_gib,
                    )
                    for host in args.hosts
                ],
            }
        elif args.subcommand == "launch":
            payload = controller_launch(args)
        elif args.subcommand == "status":
            payload = remote_json(
                args.host,
                (
                    "python3",
                    str(args.remote_script),
                    "_status-local",
                    "--host",
                    canonical_host(args.host),
                    "--run-dir",
                    str(args.run_dir),
                ),
            )
        elif args.subcommand == "stop":
            payload = remote_json(
                args.host,
                (
                    "python3",
                    str(args.remote_script),
                    "_stop-local",
                    "--host",
                    canonical_host(args.host),
                    "--run-dir",
                    str(args.run_dir),
                    "--timeout-seconds",
                    str(args.timeout_seconds),
                ),
                timeout=max(30.0, args.timeout_seconds + 15.0),
            )
        elif args.subcommand == "_audit-local":
            payload = jsonable(
                audit_local(
                    args.host,
                    driver_reservation_mib=args.driver_reservation_mib,
                    max_idle_utilization=args.max_idle_utilization,
                    min_host_available_gib=args.min_host_available_gib,
                )
            )
        elif args.subcommand == "_launch-local":
            payload = launch_local(args)
        elif args.subcommand == "_worker":
            return worker_main(args)
        elif args.subcommand == "_gpu-namespace-exec":
            return gpu_namespace_exec(args)
        elif args.subcommand == "_status-local":
            assert_running_on(args.host)
            payload = load_json(args.run_dir.resolve() / "run.json")
        elif args.subcommand == "_stop-local":
            assert_running_on(args.host)
            payload = stop_local(args.run_dir.resolve(), args.timeout_seconds)
        else:
            raise SchedulerError(f"unsupported subcommand {args.subcommand}")
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except SchedulerError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
