#!/usr/bin/env python3
"""Run one command with exactly one idle physical NVIDIA GPU visible.

Derived from the already validated GEN5 namespace helper (September 8, 2026).
This acceptance helper is intentionally independent of the taskdev scheduler,
whose current host allowlist only recognizes retired hp-* workers.  The caller
must already hold its own advisory lock; this program repeats the live idle
check immediately before creating a private mount namespace and hides every
non-selected NVIDIA and DRM device node before replacing itself with the
command.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile


MS_BIND = 4096
GPU_NODE_RE = re.compile(r"^nvidia([0-9]+)$")
DRM_NODE_RE = re.compile(r"^(?:card[0-9]+|renderD[0-9]+)$")


def _run(*args: str) -> str:
    completed = subprocess.run(
        args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _normalize_pci_bus_id(value: str) -> str:
    domain, bus, slot_function = value.strip().split(":")
    slot, function = slot_function.split(".")
    return (
        f"{int(domain, 16):04x}:{int(bus, 16):02x}:"
        f"{int(slot, 16):02x}.{int(function, 16):x}"
    )


def _gpu_pci_bus_ids() -> dict[int, str]:
    output = _run(
        "nvidia-smi",
        "--query-gpu=index,pci.bus_id",
        "--format=csv,noheader,nounits",
    )
    rows: dict[int, str] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise RuntimeError(f"unexpected nvidia-smi PCI row: {line!r}")
        rows[int(fields[0])] = _normalize_pci_bus_id(fields[1])
    return rows


def _gpu_nodes() -> dict[int, Path]:
    nodes: dict[int, Path] = {}
    for path in Path("/dev").glob("nvidia[0-9]*"):
        match = GPU_NODE_RE.match(path.name)
        if match is None:
            continue
        metadata = path.stat()
        if not stat.S_ISCHR(metadata.st_mode):
            raise RuntimeError(f"GPU device is not a character node: {path}")
        nodes[int(match.group(1))] = path
    if not nodes:
        raise RuntimeError("no /dev/nvidiaN GPU device nodes found")
    return nodes


def _drm_nodes_by_pci() -> dict[str, list[Path]]:
    rows: dict[str, list[Path]] = {}
    sysfs_root = Path("/sys/class/drm")
    device_root = Path("/dev/dri")
    for sysfs_path in sysfs_root.iterdir():
        if DRM_NODE_RE.fullmatch(sysfs_path.name) is None:
            continue
        device_path = device_root / sysfs_path.name
        if not device_path.exists():
            continue
        metadata = device_path.stat()
        if not stat.S_ISCHR(metadata.st_mode):
            raise RuntimeError(f"DRM device is not a character node: {device_path}")
        pci_bus_id = (sysfs_path / "device").resolve().name.lower()
        rows.setdefault(pci_bus_id, []).append(device_path)
    return rows


def _bind(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    mount = libc.mount
    mount.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    mount.restype = ctypes.c_int
    if mount(os.fsencode(source), os.fsencode(target), None, MS_BIND, None) != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            f"{source} -> {target}",
        )


def _bind_null(path: Path) -> None:
    _bind(Path("/dev/null"), path)
    hidden = path.stat()
    if not stat.S_ISCHR(hidden.st_mode) or (
        os.major(hidden.st_rdev), os.minor(hidden.st_rdev)
    ) != (1, 3):
        raise RuntimeError(f"failed to hide non-selected GPU node: {path}")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    import socket
    from validation_runtime import validate_gpu
    allocation = validate_gpu(
        _run("nvidia-smi", "-i", str(args.physical_gpu), "-q", "-x"),
        host=args.host, actual_host=socket.gethostname(),
        index=args.physical_gpu, uuid=args.expected_uuid,
    )
    uuid = allocation["uuid"]
    utilization = allocation["utilization_percent"]
    memory_mib = allocation["used_memory_mib"]
    processes = allocation["processes"]

    namespace_inode = os.readlink("/proc/self/ns/mnt")
    init_namespace_inode = os.readlink("/proc/1/ns/mnt")
    if namespace_inode == init_namespace_inode:
        raise RuntimeError("refusing to alter GPU nodes outside a private mount namespace")

    pci_by_index = _gpu_pci_bus_ids()
    selected_pci = pci_by_index.get(args.physical_gpu)
    if selected_pci is None:
        raise RuntimeError(f"missing PCI identity for GPU {args.physical_gpu}")

    nodes = _gpu_nodes()
    selected = nodes.get(args.physical_gpu)
    if selected is None:
        raise RuntimeError(f"missing selected GPU device /dev/nvidia{args.physical_gpu}")
    selected_metadata = selected.stat()
    logical_device = nodes.get(0)
    if logical_device is None:
        raise RuntimeError("missing logical target device /dev/nvidia0")
    if args.physical_gpu != 0:
        # NVIDIA's Vulkan ICD opens /dev/nvidia0 while establishing its first
        # physical device.  Alias the chosen card at that logical pathname in
        # this private namespace while retaining its kernel-native pathname;
        # both names refer to the same selected device and every other GPU is
        # hidden.
        _bind(selected, logical_device)
        logical_metadata = logical_device.stat()
        if (
            os.major(logical_metadata.st_rdev),
            os.minor(logical_metadata.st_rdev),
        ) != (
            os.major(selected_metadata.st_rdev),
            os.minor(selected_metadata.st_rdev),
        ):
            raise RuntimeError(
                f"failed to map {selected} onto logical device {logical_device}"
            )
    hidden: list[str] = []
    for index, path in sorted(nodes.items()):
        if index in {0, args.physical_gpu}:
            continue
        _bind_null(path)
        hidden.append(str(path))

    drm_by_pci = _drm_nodes_by_pci()
    selected_drm = sorted(drm_by_pci.get(selected_pci, []), key=str)
    if not selected_drm:
        raise RuntimeError(
            f"missing DRM device nodes for selected GPU PCI device {selected_pci}"
        )
    hidden_drm: list[str] = []
    for pci_bus_id, paths in sorted(drm_by_pci.items()):
        if pci_bus_id == selected_pci:
            continue
        for path in sorted(paths, key=str):
            _bind_null(path)
            hidden_drm.append(str(path))

    _atomic_json(
        args.run_dir / "gpu_namespace.json",
        {
            "schema_version": 1,
            "physical_gpu": args.physical_gpu,
            "logical_gpu": 0,
            "uuid": uuid,
            "pci_bus_id": selected_pci,
            "prelaunch_utilization_percent": utilization,
            "prelaunch_memory_used_mib": memory_mib,
            "prelaunch_compute_processes": processes,
            "mount_namespace_inode": namespace_inode,
            "physical_device": str(selected),
            "selected_device": str(logical_device),
            "selected_rdev": [
                os.major(selected_metadata.st_rdev),
                os.minor(selected_metadata.st_rdev),
            ],
            "hidden_devices": hidden,
            "selected_drm_devices": [str(path) for path in selected_drm],
            "hidden_drm_devices": hidden_drm,
            "command": command,
        },
    )

    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update(
        {
            "TASKDEV_PHYSICAL_GPU": str(args.physical_gpu),
            "TASKDEV_LOGICAL_GPU": "0",
            "HANDSON_GPU_DEVICE": "0",
            "HANDSON_ISAACSIM_RENDERER_GPU": "0",
            "HANDSON_ISAACLAB_DEVICE": "cuda:0",
            "HANDSON_VIDEO_GPU": "0",
            "MUJOCO_EGL_DEVICE_ID": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    os.execvpe(command[0], command, environment)
    return 127


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"simbench_gpu_namespace_exec: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
