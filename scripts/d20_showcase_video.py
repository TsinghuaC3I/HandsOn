#!/usr/bin/env python3
"""D20 showcase clip: what a recorded episode looks like from the three cameras.

Runs the real Isaac Sim 5.1 D20 environment with live RTX rendering and a
short, declared 54-D action script (hold -> right palm approaches the tongs
handle -> left palm approaches the microwave button -> right hand closes a
little -> hold).  Every control frame of head/left-wrist/right-wrist is saved
as PNG so ffmpeg can assemble an MP4.  This is a visualization aid only: no
simulator state is written after reset and no success is claimed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
import time

import numpy as np

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("HANDSON_ENABLE_ISAACSIM_NEWTON", "0")
os.environ.setdefault("HANDSON_RECORD_PHYSX_GROUND_TRUTH", "1")
os.environ.setdefault("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "0")

TASK = "d20_microwave_hot_dog_plating"
CAMERA_KEYS = ("random_camera", "wrist_left", "wrist_right")


def _grip() -> np.ndarray:
    try:
        from fu006_vertical_slice import _POWER_GRIP  # type: ignore

        return np.asarray(_POWER_GRIP, dtype=np.float64).reshape(20)
    except Exception:  # noqa: BLE001
        return np.full(20, 0.6, dtype=np.float64)


def _lerp_pose(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    out = a.copy()
    out[:3] = (1 - t) * a[:3] + t * b[:3]
    qa, qb = a[3:7] / np.linalg.norm(a[3:7]), b[3:7] / np.linalg.norm(b[3:7])
    if float(qa @ qb) < 0:
        qb = -qb
    q = (1 - t) * qa + t * qb
    out[3:7] = q / max(np.linalg.norm(q), 1e-9)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--output", type=Path, default=Path("d20_showcase"))
    parser.add_argument("--frames-dir", type=Path, default=None,
                        help="Fast local directory for the PNG frames (CPFS small-file writes are slow)")
    args = parser.parse_args()
    frames_dir = args.frames_dir or (args.output / "frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    from handson.sim.isaaclab import d20_microwave_hot_dog as d20
    from handson.sim.isaaclab.d_long_horizon_env import HandsOnDLongHorizonIsaacLabEnv

    env = HandsOnDLongHorizonIsaacLabEnv(
        TASK, seed=args.seed, randomize=True, image_obs=True, render_mode="rgb_array",
        render_resolution=(480, 640), hz=30, physics_dt=1.0 / 120.0, time_limit=240.0,
        physics_backend="physx", headless=True,
    )
    log: list[dict] = []
    try:
        obs, _ = env.reset(seed=args.seed)
        state = obs["state"]
        tcp = np.asarray(state["tcp_pose"], dtype=np.float64).reshape(14)
        right_home, left_home = tcp[:7].copy(), tcp[7:].copy()
        tongs = np.asarray(state["tongs_pose"], dtype=np.float64).reshape(7)
        microwave = np.asarray(state["microwave_pose"], dtype=np.float64).reshape(7)
        # Declared waypoints (world frame): above the tongs handle, and in
        # front of the start button (microwave frame rotated -90 deg about Z).
        right_goal = right_home.copy()
        right_goal[:3] = [tongs[0] - 0.10, tongs[1], tongs[2] + 0.14]
        left_goal = left_home.copy()
        left_goal[:3] = [microwave[0] - 0.2258 - 0.13, microwave[1] - 0.267 + 0.02, 1.06]
        grip = _grip()
        open_hand = np.zeros(20)
        phases = [
            ("hold_home", 45, right_home, left_home, open_hand, open_hand),
            ("right_to_tongs", 90, right_goal, left_home, open_hand, open_hand),
            ("left_to_button", 90, right_goal, left_goal, open_hand, open_hand),
            ("right_hand_close", 30, right_goal, left_goal, 0.5 * grip, open_hand),
            ("hold_end", 45, right_goal, left_goal, 0.5 * grip, open_hand),
        ]
        cur_r, cur_l, cur_rh, cur_lh = right_home, left_home, open_hand, open_hand
        frame = 0
        started = time.time()

        def save(observation: dict, index: int) -> None:
            for key in CAMERA_KEYS:
                arr = np.asarray(observation["images"][key])[..., :3].astype(np.uint8)
                Image.fromarray(arr).save(frames_dir / f"{key}_{index:04d}.png")

        save(obs, frame)
        for name, n, tr, tl, trh, tlh in phases:
            for i in range(1, n + 1):
                t = i / n
                action = np.concatenate([
                    _lerp_pose(cur_r, tr, t), _lerp_pose(cur_l, tl, t),
                    (1 - t) * cur_rh + t * trh, (1 - t) * cur_lh + t * tlh,
                ]).astype(np.float32)
                obs, _rew, term, trunc, info = env.step(action)
                frame += 1
                save(obs, frame)
                if term or trunc:
                    log.append({"phase": name, "frame": frame, "terminated": bool(term), "truncated": bool(trunc)})
                    break
            cur_r, cur_l, cur_rh, cur_lh = tr, tl, trh, tlh
            tcp_now = np.asarray(obs["state"]["tcp_pose"], dtype=np.float64).reshape(14)
            log.append({
                "phase": name, "frame": frame,
                "right_tcp": tcp_now[:3].round(3).tolist(), "left_tcp": tcp_now[7:10].round(3).tolist(),
                "right_goal": tr[:3].round(3).tolist(), "left_goal": tl[:3].round(3).tolist(),
                "tongs_state": np.asarray(obs["state"]["tongs_state"]).round(3).tolist(),
                "microwave_state": np.asarray(obs["state"]["microwave_state"]).round(3).tolist(),
                "d20": {k: (float(v) if np.isscalar(v) and not isinstance(v, str) else str(v)[:80]) for k, v in info.items() if k.startswith("d20_")},
            })
            print(json.dumps(log[-1]), flush=True)
        (args.output / "showcase_log.json").write_text(json.dumps({
            "task": TASK, "seed": args.seed, "frames": frame + 1, "fps": 30,
            "elapsed_seconds": time.time() - started, "phases": log,
        }, indent=2) + "\n")
        print("D20_SHOWCASE_DONE", frame + 1, flush=True)
    except BaseException:  # noqa: BLE001 - print before Kit teardown can hang
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        import threading

        threading.Timer(60.0, lambda: os._exit(0)).start()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
