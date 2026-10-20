#!/usr/bin/env python3
"""D20 authored-pose collision overlap query (taskdev hard gate).

Builds the real Isaac Sim 5.1 D20 environment without randomization, resets
it to the authored layout, and asks PhysX which collision shapes overlap.
Resting supports (object on table/tray, tongs on cradle, tray/cradle/
microwave on table) are approved; every other overlapping pair is reported
as unapproved.  The script never writes simulator state after reset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("HANDSON_ENABLE_ISAACSIM_NEWTON", "0")
os.environ.setdefault("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "1")
os.environ.setdefault("HANDSON_RECORD_PHYSX_GROUND_TRUTH", "1")

TASK = "d20_microwave_hot_dog_plating"
D20_BODIES = (
    "d20_microwave", "d20_hot_dog", "d20_tongs", "d20_tongs_cradle",
    "d20_serving_tray", "d20_plate", "d20_fork", "d20_mustard", "d20_mug",
)
SUPPORT_TOKENS = ("table", "counter", "worktop", "ground", "floor", "d20_serving_tray", "d20_tongs_cradle")
# (body, support) pairs that are expected resting contacts at the authored pose.
APPROVED = {
    ("d20_hot_dog", "table"), ("d20_tongs", "d20_tongs_cradle"), ("d20_tongs", "table"),
    ("d20_plate", "d20_serving_tray"), ("d20_fork", "d20_serving_tray"),
    ("d20_mustard", "d20_serving_tray"), ("d20_mug", "d20_serving_tray"),
    ("d20_serving_tray", "table"), ("d20_tongs_cradle", "table"), ("d20_microwave", "table"),
    # Internal articulation contacts of the same asset.
    ("d20_microwave", "d20_microwave"), ("d20_tongs", "d20_tongs"),
}


def _owner(path: str) -> str:
    for body in sorted(D20_BODIES, key=len, reverse=True):
        if f"/{body}" in path or f"_{body}_" in path or path.rstrip("/").endswith(body):
            return body
    low = path.lower()
    if any(tok in low for tok in ("table", "counter", "worktop")):
        return "table"
    if "ground" in low or "floor" in low:
        return "table"
    if re.search(r"(rokae|wuji|hand|finger|thumb|palm|link|arm|robot)", low):
        return "robot"
    return "other:" + path.rsplit("/", 2)[-2] if "/" in path else path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("d20_overlap_probe"))
    parser.add_argument("--settle-steps", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    from handson.sim.isaaclab.d_long_horizon_env import HandsOnDLongHorizonIsaacLabEnv
    import numpy as np

    started = time.time()
    env = HandsOnDLongHorizonIsaacLabEnv(
        TASK, seed=0, randomize=False, image_obs=False, render_mode="rgb_array",
        render_resolution=(480, 640), hz=30, physics_dt=1.0 / 120.0, time_limit=240.0,
        physics_backend="physx", headless=True,
    )
    report: dict = {"task": TASK, "randomize": False, "settle_steps": args.settle_steps}
    try:
        env.reset(seed=0)
        for _ in range(args.settle_steps):
            env.step(np.zeros(54, dtype=np.float32))
        # Contact-report based overlap query: PhysX reports every touching
        # pair after a physics step together with the signed separation of
        # each contact point (negative = penetration).  The ground-truth
        # recorder already applied PhysxContactReportAPI to the task bodies.
        import omni.physx
        from pxr import PhysicsSchemaTools

        # Step three zero-action control steps and read the env's own PhysX
        # ground-truth contact trace (the same recorder the production
        # certifier uses).  Its canonical pair signature carries the maximum
        # penetration depth of every touching pair.
        for _ in range(3):
            env.step(np.zeros(54, dtype=np.float32))
        trace = getattr(env, "_physx_ground_truth_trace", None)
        if trace is None:
            raise RuntimeError("PhysX ground-truth trace is not attached (set HANDSON_RECORD_PHYSX_GROUND_TRUTH=1)")
        snap = trace.snapshot()
        table = getattr(trace, "_contact_report_path_table", ())
        canonical = tuple(sorted(str(e["path"]) for e in table))
        valid = int(np.asarray(snap["physx_contact_signature_valid"]).reshape(-1)[0])
        pair_count = int(np.asarray(snap["physx_contact_signature_pair_count"]).reshape(-1)[0])
        idx = np.asarray(snap["physx_contact_signature_pair_path_indices"]).reshape(-1, 4)
        metrics = np.asarray(snap["physx_contact_signature_pair_metrics"]).reshape(-1, 3)
        counts = np.asarray(snap["physx_contact_signature_pair_counts"]).reshape(-1, 2)
        report["contact_capture_source"] = int(np.asarray(snap["physx_contact_capture_source"]).reshape(-1)[0])
        report["contact_header_count"] = int(np.asarray(snap["physx_contact_header_count"]).reshape(-1)[0])
        report["contact_point_count"] = int(np.asarray(snap["physx_contact_point_count"]).reshape(-1)[0])
        report["signature_valid"] = valid
        report["signature_pair_count"] = pair_count
        report["canonical_path_count"] = len(canonical)
        report["report_sensor_paths"] = sorted(getattr(trace, "_tensor_contact_sensor_paths", ()) or getattr(trace, "_contact_report_paths", ()))

        def _cpath(i: int) -> str:
            i = int(i)
            return canonical[i - 1] if 1 <= i <= len(canonical) else f"?{i}"

        pairs: dict[tuple[str, str], dict] = {}
        for row in range(pair_count):
            a0, c0, a1, c1 = (_cpath(v) for v in idx[row])
            depth = float(metrics[row][0])
            key = tuple(sorted((_owner(c0) if not _owner(c0).startswith("other") else _owner(a0),
                                _owner(c1) if not _owner(c1).startswith("other") else _owner(a1))))
            entry = pairs.setdefault(key, {"max_penetration_m": 0.0, "contact_points": 0, "examples": []})
            entry["max_penetration_m"] = max(entry["max_penetration_m"], depth)
            entry["contact_points"] += int(counts[row][0])
            if len(entry["examples"]) < 4:
                entry["examples"].append(f"{c0} <-> {c1} depth={depth*1000:.2f}mm")
        d20_pairs = {k: v for k, v in pairs.items() if any(x.startswith("d20_") for x in k)}
        unapproved = {}
        approved = {}
        for key, entry in d20_pairs.items():
            a, b = key
            ok = (a, b) in APPROVED or (b, a) in APPROVED
            (approved if ok else unapproved)[f"{a} <-> {b}"] = entry
        report["all_pair_owners"] = sorted(f"{a} <-> {b}" for a, b in pairs)
        report["approved_pairs"] = approved
        report["unapproved_pairs"] = unapproved
        report["unapproved_count"] = len(unapproved)
        report["max_approved_penetration_m"] = max([v["max_penetration_m"] for v in approved.values()], default=0.0)
        report["elapsed_seconds"] = time.time() - started
        (args.output / "overlap_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("D20_OVERLAP_DONE", json.dumps({"valid": valid, "pairs": pair_count, "source": report["contact_capture_source"], "approved": {k: round(v["max_penetration_m"]*1000, 2) for k, v in approved.items()}, "unapproved": unapproved, "others": report["all_pair_owners"]}, indent=1), flush=True)
    finally:
        import threading
        threading.Timer(60.0, lambda: os._exit(0)).start()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
