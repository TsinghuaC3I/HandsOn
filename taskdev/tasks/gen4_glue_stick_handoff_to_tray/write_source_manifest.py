#!/usr/bin/env python3
"""Bind the current GEN4 source bytes for a future gated diagnostic launch.

This CPU-only manifest is source provenance, not a validation certificate.
Historical manifests remain immutable; generate again after source changes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task = Path(__file__).resolve().parent
    root = task.parents[2]
    historical = task / "evidence/cpu_implementation_manifest.json"
    output = args.output.resolve()
    protected = {historical.resolve(), *(task / "evidence").glob("supervisor_source_manifest*.json")}
    if output in protected:
        parser.error("Historical manifests must not be overwritten")
    inventory = json.loads(historical.read_text())
    names = set(inventory["files"])
    names.update(str((task / name).relative_to(root)) for name in (
        "probe_public_actions.py", "write_source_manifest.py", "audit_appearance_cpu.py", "capture_appearance.py",
        "simbench_gpu_namespace_exec.py", "validation_runtime.py",
    ))
    names.update((
        "handson/handson/tasks/eval_gen04_pass_glue_stick/visual_appearance.py",
        "handson/handson/sim/isaaclab/blender_visual_layer.py",
        "handson/handson/sim/isaaclab/task_visual_appearance.py",
        "handson/handson/sim/isaaclab/wrist_cable_visuals.py",
        "handson/handson/sim/isaaclab/bimanual_assembly_env.py",
    ))
    bundle = root / "handson/handson/sim/envs/usd/rokae_wuji1_real_v11"
    names.update(str(path.relative_to(root)) for path in bundle.rglob("*") if path.is_file())
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True,
    ).strip()
    if branch != "codex/gen4-glue-stick-handoff":
        parser.error(f"Expected GEN4 branch, got {branch!r}")
    manifest = {
        "format": "gen4-source-binding-v1",
        "branch": branch,
        "base_commit": inventory["base_commit"],
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        ).strip(),
        "source_tree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True,
        ).strip()),
        "scope": "Current source binding only; no CPU, GPU, or acceptance claim.",
        "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                  for name in sorted(names)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{output}: {len(manifest['files'])} files at {manifest['source_commit']}")


if __name__ == "__main__":
    main()
