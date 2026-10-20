#!/usr/bin/env python3
"""Background batch runner for E/F garment IPC postprocessing."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPO_ROOT / "handson"
for _entry in (str(_REPO_ROOT), str(_PACKAGE_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from absl import flags as absl_flags

from scripts import record_demos_zarr as recorder


STATUS_FILE = "e01_postprocess.json"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(
        timespec="seconds"
    )


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _ensure_absl_flags_parsed() -> None:
    if not absl_flags.FLAGS.is_parsed():
        absl_flags.FLAGS([sys.argv[0]])


def _run_one(
    demo_dir: Path,
    *,
    exp_name: str,
    sim_backend: str,
    ipc_env_value: str,
    arm_env_value: str,
    video_fps: int,
    log_path: Path | None,
    force: bool,
) -> int:
    demo_path = demo_dir.expanduser().resolve()
    current = _read_json(demo_path / STATUS_FILE)
    if not force and str(current.get("status") or "").lower() == "complete":
        print(f"[E01 IPC postprocess worker] skip complete: {demo_path}", flush=True)
        return 0

    print(
        f"[E01 IPC postprocess worker] starting: {demo_path}",
        flush=True,
    )
    try:
        result = recorder._run_e01_postprocess_pipeline(
            demo_path,
            exp_name=exp_name,
            sim_backend=sim_backend,
            ipc_env_value=ipc_env_value,
            arm_env_value=arm_env_value,
            video_fps=max(1, int(video_fps)),
            background=True,
            log_path=str(log_path) if log_path is not None else "",
        )
        status = str(result.get("status") or "unknown")
        print(
            f"[E01 IPC postprocess worker] {status}: {demo_path}",
            flush=True,
        )
        return 0 if status == "complete" else 1
    except KeyboardInterrupt:
        recorder._write_e01_postprocess_document(
            demo_path,
            {
                "format": "handson.e01_postprocess",
                "version": 1,
                "status": "stopped",
                "started_at": current.get("started_at") or _now_iso(),
                "completed_at": _now_iso(),
                "background": True,
                "exp_name": exp_name,
                "sim_backend": sim_backend,
                "video_fps": max(1, int(video_fps)),
                "log_path": str(log_path) if log_path is not None else "",
                "stages": (
                    current.get("stages")
                    if isinstance(current.get("stages"), dict)
                    else {}
                ),
                "error": "stopped for collection",
            },
        )
        raise
    except Exception as exc:  # noqa: BLE001
        recorder._write_e01_postprocess_document(
            demo_path,
            {
                "format": "handson.e01_postprocess",
                "version": 1,
                "status": "failed",
                "started_at": current.get("started_at") or _now_iso(),
                "completed_at": _now_iso(),
                "background": True,
                "exp_name": exp_name,
                "sim_backend": sim_backend,
                "video_fps": max(1, int(video_fps)),
                "log_path": str(log_path) if log_path is not None else "",
                "stages": (
                    current.get("stages")
                    if isinstance(current.get("stages"), dict)
                    else {}
                ),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        print(
            "[E01 IPC postprocess worker] failed with exception:\n"
            + traceback.format_exc(),
            flush=True,
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo_dirs", nargs="+", help="Saved demo directories.")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--sim_backend", required=True)
    parser.add_argument("--ipc_env_value", default="1")
    parser.add_argument("--arm_env_value", default="1")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--log_path", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    _ensure_absl_flags_parsed()
    log_path = Path(args.log_path).expanduser().resolve() if args.log_path else None
    print(
        "[E01 IPC postprocess worker] batch start: "
        f"{len(args.demo_dirs)} demo(s), exp_name={args.exp_name}, log={log_path or ''}",
        flush=True,
    )
    failures = 0
    for demo in args.demo_dirs:
        if _run_one(
            Path(demo),
            exp_name=args.exp_name,
            sim_backend=args.sim_backend,
            ipc_env_value=args.ipc_env_value,
            arm_env_value=args.arm_env_value,
            video_fps=args.video_fps,
            log_path=log_path,
            force=bool(args.force),
        ) != 0:
            failures += 1
    print(
        f"[E01 IPC postprocess worker] batch done: failures={failures}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
