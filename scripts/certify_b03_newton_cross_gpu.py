#!/usr/bin/env python3
"""Run B03 Newton certification on a host with a different GPU model.

Formal mode launches exactly three sequential fresh Python processes.  Each
process creates, replays, and destroys one Newton scene, then the parent binds
the immutable reports into ``newton_cross_gpu_acceptance.json``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "handson"
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from handson.data.newton_cross_gpu_certification import (
    ACCEPTANCE_NAME,
    REQUIRED_RUNS,
    RUN_DIRECTORY,
    build_acceptance,
)
from handson.sim.isaaclab.newton_determinism import (
    NEWTON_CROSS_GPU_PROFILE,
    NEWTON_REPLAY_PROFILE_ENV,
)


def _run(
    demo_dir: Path,
    *,
    python: Path,
    replay_script: Path,
    same_gpu_smoke: bool,
    overwrite: bool,
) -> Path:
    if same_gpu_smoke:
        report_paths = [demo_dir / "newton_same_gpu_smoke.json"]
    else:
        report_paths = [
            demo_dir / RUN_DIRECTORY / f"run_{index:03d}.json"
            for index in range(1, REQUIRED_RUNS + 1)
        ]
    protected = [*report_paths]
    if not same_gpu_smoke:
        protected.append(demo_dir / ACCEPTANCE_NAME)
    existing = [path for path in protected if path.exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise RuntimeError(
            "Certification outputs already exist; inspect them or rerun with "
            f"--overwrite: {paths}"
        )

    environment = os.environ.copy()
    environment[NEWTON_REPLAY_PROFILE_ENV] = NEWTON_CROSS_GPU_PROFILE
    environment["HANDSON_RECORD_NEWTON_GROUND_TRUTH"] = "1"
    environment["HANDSON_ISAACLAB_DIRECT_HAND_CONTROL"] = "0"
    environment["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] = "0"
    environment.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    for index, report_path in enumerate(report_paths, start=1):
        command = [
            str(python),
            str(replay_script),
            str(demo_dir),
            "--report-path",
            str(report_path),
        ]
        if same_gpu_smoke:
            command.append("--same-gpu-smoke")
        print(
            f"[B03 Newton certification] fresh run {index}/{len(report_paths)}",
            flush=True,
        )
        result = subprocess.run(
            command,
            cwd=str(replay_script.parents[1]),
            env=environment,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"fresh replay run {index} failed with exit code "
                f"{result.returncode}; see {report_path}"
            )

    if same_gpu_smoke:
        print(
            "[B03 Newton certification] local smoke passed; no acceptance "
            "was issued and upload remains blocked.",
            flush=True,
        )
        return report_paths[0]
    acceptance = build_acceptance(demo_dir, report_paths, write=True)
    acceptance_path = demo_dir / ACCEPTANCE_NAME
    print(
        "[B03 Newton certification] ACCEPTED: "
        f"runs={acceptance['completed_runs']}, path={acceptance_path}",
        flush=True,
    )
    return acceptance_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo_dir", type=Path)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Qualified Isaac Sim/Newton Python used for every fresh run.",
    )
    parser.add_argument(
        "--same-gpu-smoke",
        action="store_true",
        help="Run once without issuing publishable acceptance.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing run reports/acceptance after explicit review.",
    )
    args = parser.parse_args()
    demo_dir = args.demo_dir.expanduser().resolve()
    python = args.python.expanduser().resolve()
    replay_script = Path(__file__).resolve().with_name(
        "replay_b03_newton_physics.py"
    )
    if not demo_dir.is_dir():
        parser.error(f"demo directory does not exist: {demo_dir}")
    if not python.is_file() or not os.access(python, os.X_OK):
        parser.error(f"Python runtime is not executable: {python}")
    _run(
        demo_dir,
        python=python,
        replay_script=replay_script,
        same_gpu_smoke=bool(args.same_gpu_smoke),
        overwrite=bool(args.overwrite),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
