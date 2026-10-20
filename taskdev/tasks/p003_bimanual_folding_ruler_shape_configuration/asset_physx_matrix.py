#!/usr/bin/env python3
"""Run the historical immutable-source Ruler035 canaries in fresh processes.

This matrix only aggregates source-asset diagnostics.  It does not exercise
the current analytic collision overlay, passive hinges, tabletop runtime, or
success contract and therefore is not a current P0-03 physics gate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


POSE_KEYS = ("extended", "square_u", "zigzag_left", "zigzag_right")
MODULE = (
    "taskdev.tasks.p003_bimanual_folding_ruler_shape_configuration."
    "asset_physx_canary"
)


def _write(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows: list[dict[str, object]] = []
    for key in POSE_KEYS:
        output = output_dir / f"ruler035_{key}_physx_canary.json"
        log = output_dir / f"ruler035_{key}_physx_canary.log"
        command = [
            sys.executable,
            "-m",
            MODULE,
            "--pose-key",
            key,
            "--output",
            str(output),
        ]
        with log.open("wb") as stream:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=240.0,
            )
        payload: dict[str, object] = {}
        if output.is_file():
            payload = json.loads(output.read_text(encoding="utf-8"))
        rows.append(
            {
                "pose_key": key,
                "return_code": int(completed.returncode),
                "output": str(output),
                "log": str(log),
                "reported_source_canary_pass": (
                    payload.get("source_asset_canary_passed") is True
                ),
                "source_only_scope_confirmed": bool(
                    payload.get("evidence_scope")
                    == "immutable_source_asset_only"
                    and payload.get("current_tabletop_physics_gate") is False
                ),
                "runner_rc": payload.get("runner_rc"),
            }
        )
        if (
            completed.returncode != 0
            or payload.get("source_asset_canary_passed") is not True
            or payload.get("current_tabletop_physics_gate") is not False
        ):
            break
    source_asset_matrix_passed = bool(
        len(rows) == len(POSE_KEYS)
        and all(
            row["return_code"] == 0
            and row["reported_source_canary_pass"] is True
            and row["source_only_scope_confirmed"] is True
            for row in rows
        )
    )
    report = {
        "format": "handson.p003.ruler035_source_physx_matrix_v2",
        "evidence_scope": "immutable_source_asset_only",
        "evidence_status": "historical_non_runtime_acceptance",
        "current_tabletop_physics_gate": False,
        "current_physics_gate_references": [
            "handson/tests/test_p003_physics_overlay.py",
            (
                "taskdev/tasks/"
                "p003_bimanual_folding_ruler_shape_configuration/"
                "runtime_reset_smoke.py"
            ),
        ],
        "pose_keys": list(POSE_KEYS),
        "rows": rows,
        "wall_seconds": time.time() - started,
        "source_asset_matrix_passed": source_asset_matrix_passed,
        # Compatibility for archived tooling only; this remains source-only
        # historical evidence, never current tabletop-runtime acceptance.
        "passed": source_asset_matrix_passed,
    }
    _write(output_dir / "ruler035_physx_matrix.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if report["source_asset_matrix_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
