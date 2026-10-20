#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "handson"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from handson.tasks.mappings import CONFIG_MAPPING


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the bimanual_assembly Isaac Sim backend."
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-mode", choices=["rgb_array", "human"], default="rgb_array")
    parser.add_argument("--arm-type", choices=["panda", "rokae"], default="rokae")
    parser.add_argument("--hand-type", choices=["allegro", "wuji1"], default="wuji1")
    parser.add_argument(
        "--interface-only",
        action="store_true",
        help="Do not launch Isaac Sim; validate the Gym/HandsOn interface only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIG_MAPPING["bimanual_assembly"]()
    env = config.get_native_environment(
        policy_mode=True,
        render_mode=args.render_mode,
        sim_backend="isaacsim",
        arm_type=args.arm_type,
        hand_type=args.hand_type,
        launch_isaac=not args.interface_only,
        seed=args.seed,
    )

    try:
        obs, info = env.reset(seed=args.seed)
        print(f"reset ok: state_shape={obs['state'].shape}, info={info}")
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        for index in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"episode ended at step {index}: reward={reward}, info={info}")
                break
        print(f"step ok: state_shape={obs['state'].shape}, last_info={info}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
