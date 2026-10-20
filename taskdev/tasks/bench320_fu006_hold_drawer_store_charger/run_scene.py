#!/usr/bin/env python3
"""Compatibility entry point for the FU006 Isaac Sim 5.1 live probe.

The implementation lives in ``scripts/fu006_vertical_slice.py`` so probe,
collection, and fresh replay share one runtime and one physical truth contract.
This process must itself be launched by the taskdev scheduler under its GPU
lease; it never starts or selects an unscheduled GPU.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stability-steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=320006)
    # Retain the old scheduler command-line surface; the scheduler owns device
    # names and the shared runtime is always headless with cameras enabled.
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable_cameras", action="store_true")
    args = parser.parse_args()
    if args.device != "cuda:0":
        raise ValueError("FU006 scheduler namespace must expose physical GPU1 as cuda:0")
    if os.environ.get("HANDSON_ENABLE_ISAACSIM_NEWTON", "0").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }:
        raise ValueError("FU006 forbids Newton")

    script = REPOSITORY_ROOT / "scripts" / "fu006_vertical_slice.py"
    os.execv(
        sys.executable,
        [
            sys.executable,
            str(script),
            "probe",
            "--output",
            str(args.output_dir),
            "--steps",
            str(args.stability_steps),
            "--seed",
            str(args.seed),
        ],
    )
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
