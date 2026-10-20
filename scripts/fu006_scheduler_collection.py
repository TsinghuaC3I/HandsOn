#!/usr/bin/env python3
"""Auditable, fail-closed scheduler invocation for FU006 collection.

The default mode is CPU-only: it prints the complete scheduler argv and the
resolved path contract without contacting a cluster node.  ``--launch`` is the
only execution gate and must be used only inside an explicitly authorized GPU
window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Sequence


TASK_ID = "bench320-fu006-vertical-slice-collect"
CANONICAL_REPO = Path(
    "/mnt/pfs/luotianwei/handson-task-worktrees/"
    "t46_store_charger"
)
CANONICAL_STATE_ROOT = Path(
    "/mnt/pfs/luotianwei/.handson-taskdev-cluster"
)
CANONICAL_RUN_ROOT = Path("/mnt/pfs/luotianwei/handson-taskdev-runs")
CANONICAL_WORKTREE_ROOT = Path(
    "/mnt/pfs/luotianwei/handson-taskdev-worktrees"
)
REMOTE_SCHEDULER = Path(
    "/mnt/pfs/luotianwei/handson-taskdev-orchestration/"
    "taskdev_cluster_all_hp.py"
)
POLICY_PATH = Path(
    "/mnt/pfs/luotianwei/handson-taskdev-orchestration/"
    "b320_32_scene_wave_20260817/FULL_PIPELINE_RUN_POLICY.md"
)
EXPECTED_SCHEDULER_SHA256 = (
    "3bca73f695658fd4bbe8dcf4944754a595d662b472344c57ea230f30c5792d63"
)
EXPECTED_POLICY_SHA256 = (
    "aa2346d67200334db837decba8ec74477153e42df179be4dd9fcdaaafe418fd4"
)
FORBIDDEN_HOSTS = frozenset({"hp-1", "hp-6", "hp-17"})
MAX_RUNTIME_SECONDS = 1800


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_authorization(host: str, gpu: int, base_ref: str) -> None:
    if not re.fullmatch(r"hp-[0-9]+", host) or host in FORBIDDEN_HOSTS:
        raise ValueError(f"FU006 refuses unauthorized host {host!r}")
    if not 0 <= gpu <= 7:
        raise ValueError("FU006 physical GPU must be in [0, 7]")
    if not re.fullmatch(r"[0-9a-f]{40}", base_ref):
        raise ValueError("FU006 base-ref must be an exact 40-character commit")


def collection_command() -> list[str]:
    """Return the complete run command with scheduler-owned path templates."""
    shell_command = (
        'mkdir -p "{run_dir}/tmp" "{run_dir}/xdg_cache" '
        '"{run_dir}/xdg_config" "{run_dir}/xdg_data" && '
        "exec /mnt/pfs/luotianwei/envs/isaacsim-5.1-handson/bin/python "
        '"{worktree}/scripts/fu006_vertical_slice.py" collect '
        '--seed 320006 --output "{run_dir}/episode_candidate"'
    )
    return [
        "/usr/bin/env",
        "OMNI_KIT_ACCEPT_EULA=YES",
        "ACCEPT_EULA=Y",
        (
            "LD_LIBRARY_PATH=/mnt/pfs/luotianwei/envs/"
            "handson-openpi-jax062/lib:/mnt/pfs/luotianwei/"
            "handson-taskdev-runtime-libs/glvnd-jammy-1.4.0/"
            "usr/lib/x86_64-linux-gnu"
        ),
        "__NV_PRIME_RENDER_OFFLOAD=1",
        "__GLX_VENDOR_LIBRARY_NAME=nvidia",
        "__VK_LAYER_NV_optimus=NVIDIA_only",
        "VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json",
        "VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json",
        (
            "HANDSON_ISAACSIM_PYTHON=/mnt/pfs/luotianwei/envs/"
            "isaacsim-5.1-handson/bin/python"
        ),
        (
            "HANDSON_ISAACLAB_XML_DIR=/mnt/pfs/luotianwei/"
            "handson-taskdev-runtime-assets/isaac51-xml-overlay-v1"
        ),
        "HANDSON_ENABLE_ISAACSIM_NEWTON=0",
        "HANDSON_RECORD_PHYSX_GROUND_TRUTH=1",
        "ISAAC_SIM_DIR=/mnt/pfs/luotianwei/envs/isaacsim-5.1-handson",
        "ISAAC_LAB_DIR=/mnt/pfs/luotianwei/IsaacLab-v2.3.2",
        "WORKSPACE_DIR={run_dir}",
        "TMPDIR={run_dir}/tmp",
        "XDG_CACHE_HOME={run_dir}/xdg_cache",
        "XDG_CONFIG_HOME={run_dir}/xdg_config",
        "XDG_DATA_HOME={run_dir}/xdg_data",
        "PYTHONPATH={worktree}:{worktree}/handson",
        "/bin/bash",
        "-lc",
        shell_command,
    ]


def build_scheduler_argv(
    *,
    scheduler: Path,
    host: str,
    gpu: int,
    base_ref: str,
    launch: bool,
) -> list[str]:
    """Build the only accepted FU006 collection scheduler invocation."""
    _validate_authorization(host, gpu, base_ref)
    argv = [
        "python3",
        str(scheduler),
        "launch",
        "--host",
        host,
        "--gpu",
        str(gpu),
        "--task-id",
        TASK_ID,
        "--repo",
        str(CANONICAL_REPO),
        "--base-ref",
        base_ref,
        "--state-root",
        str(CANONICAL_STATE_ROOT),
        "--run-root",
        str(CANONICAL_RUN_ROOT),
        "--worktree-root",
        str(CANONICAL_WORKTREE_ROOT),
        "--max-runtime-seconds",
        str(MAX_RUNTIME_SECONDS),
        "--remote-script",
        str(REMOTE_SCHEDULER),
    ]
    if launch:
        argv.append("--launch")
    return [*argv, "--", *collection_command()]


def resolved_contract(run_id: str) -> dict[str, str]:
    """Resolve scheduler-owned paths for a proposed run without side effects."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]+", run_id):
        raise ValueError(f"invalid scheduler run id {run_id!r}")
    run_dir = CANONICAL_RUN_ROOT / run_id
    worktree = CANONICAL_WORKTREE_ROOT / run_id
    return {
        "state_root": str(CANONICAL_STATE_ROOT),
        "run_dir": str(run_dir),
        "worktree": str(worktree),
        "runner": str(worktree / "scripts/fu006_vertical_slice.py"),
        "episode_candidate": str(run_dir / "episode_candidate"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument(
        "--contract-run-id",
        default="fu006-contract-dry-run",
        help="CPU-only run id used to render the resolved path contract.",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Execute the scheduler; omit to print a CPU-only audit plan.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    scheduler_argv = build_scheduler_argv(
        scheduler=args.scheduler,
        host=args.host,
        gpu=args.gpu,
        base_ref=args.base_ref,
        launch=args.launch,
    )
    if not args.launch:
        print(
            json.dumps(
                {
                    "mode": "cpu_only_contract",
                    "scheduler_argv": scheduler_argv,
                    "resolved_contract": resolved_contract(args.contract_run_id),
                    "expected_scheduler_sha256": EXPECTED_SCHEDULER_SHA256,
                    "policy_path": str(POLICY_PATH),
                    "expected_policy_sha256": EXPECTED_POLICY_SHA256,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    scheduler_hash = _sha256(args.scheduler)
    if scheduler_hash != EXPECTED_SCHEDULER_SHA256:
        raise RuntimeError(
            "FU006 scheduler hash mismatch: "
            f"expected {EXPECTED_SCHEDULER_SHA256}, got {scheduler_hash}"
        )
    os.execvp(scheduler_argv[0], scheduler_argv)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
