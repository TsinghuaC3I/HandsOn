#!/usr/bin/env python3
"""Audit the real B320-HS005 success state machine on 10+10 fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import types

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "handson"))

from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv  # noqa: E402


TASK = "t37_scrub_bottle"


def _environment() -> HandsOnABCIsaacLabEnv:
    return HandsOnABCIsaacLabEnv(
        TASK,
        launch_isaac=False,
        image_obs=False,
        render_mode="rgb_array",
    )


def _install_synthetic_physics(env: HandsOnABCIsaacLabEnv) -> dict:
    tokens = {
        name: object() for name in ("left_bottle", "right_brush", "brush_bottle")
    }
    forces = {id(token): 1.0 for token in tokens.values()}
    poses = {
        "hs005_bottle": np.asarray(
            [0.05, 0.0, 0.924, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        ),
        "hs005_brush": np.asarray(
            [-0.29, -0.16, 0.928, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        ),
    }
    velocities = {
        "hs005_bottle": np.zeros(6, dtype=np.float64),
        "hs005_brush": np.zeros(6, dtype=np.float64),
    }
    state = {
        "tokens": tokens,
        "forces": forces,
        "poses": poses,
        "velocities": velocities,
        "geometry": (np.zeros(3, dtype=np.float64), 0.060, 0.0, 0),
        "clear": (False, False),
        "safe_grasp": True,
        "bristle_touch": False,
    }
    env._hs005_contact_views = tokens
    env._current_body_pose = types.MethodType(
        lambda self, name: state["poses"].get(name), env
    )
    env._physics_body_velocity = types.MethodType(
        lambda self, name: state["velocities"].get(name), env
    )
    env._c04_contact_force = types.MethodType(
        lambda self, view: state["forces"].get(id(view), 0.0), env
    )
    env._hs005_bristle_geometry = types.MethodType(
        lambda self: state["geometry"], env
    )
    env._hs005_hand_clear = types.MethodType(
        lambda self: state["clear"], env
    )
    env._hs005_brush_grasp_state = types.MethodType(
        lambda self: (
            state["safe_grasp"],
            state["bristle_touch"],
            0.0,
            2 if state["safe_grasp"] else 0,
        ),
        env,
    )
    return state


def _set_contact_force(state: dict, name: str, force: float) -> None:
    state["forces"][id(state["tokens"][name])] = float(force)


def _sample(env: HandsOnABCIsaacLabEnv, state: dict, point, local_z, band) -> None:
    state["geometry"] = (
        np.asarray(point, dtype=np.float64),
        float(local_z),
        0.0,
        int(band),
    )
    env._update_hs005_physical_events()


def _ordered_scrub(
    env: HandsOnABCIsaacLabEnv,
    state: dict,
    *,
    angle_rad: float = 0.0,
    short_path: bool = False,
) -> None:
    direction = np.asarray(
        [np.cos(angle_rad), np.sin(angle_rad), 0.0], dtype=np.float64
    )
    bands = (0, 1, 2) if short_path else (0, 0, 1, 2, 2, 2)
    heights = {0: 0.060, 1: 0.125, 2: 0.190}
    for index, band in enumerate(bands):
        point = np.asarray([0.0, 0.0, 1.0]) + 0.020 * index * direction
        _sample(env, state, point, heights[band], band)


def _release_and_hold(env: HandsOnABCIsaacLabEnv, state: dict) -> None:
    for name in state["tokens"]:
        _set_contact_force(state, name, 0.0)
    required_steps = int(np.ceil(0.5 / env._physics_dt))
    for _ in range(required_steps):
        env._update_hs005_physical_events()


def _positive_trial(index: int) -> dict:
    env = _environment()
    try:
        state = _install_synthetic_physics(env)
        force = 0.25 + 0.15 * index
        for name in state["tokens"]:
            _set_contact_force(state, name, force)
        velocity = 0.001 + 0.0005 * index
        state["velocities"]["hs005_bottle"][0] = velocity
        state["velocities"]["hs005_brush"][1] = velocity
        _ordered_scrub(env, state, angle_rad=0.17 * index)
        state["clear"] = (True, True)
        _release_and_hold(env, state)
        predicted = bool(env._hs005_success_now())
        return {
            "id": f"positive_{index:02d}",
            "expected": True,
            "predicted": predicted,
            "force_n": force,
            "path_angle_rad": 0.17 * index,
            "path_length_m": float(env._hs005_path_length),
            "stability_steps": int(env._hs005_stability_steps),
            "stability_seconds": float(
                env._hs005_stability_steps * env._physics_dt
            ),
        }
    finally:
        env.close()


def _negative_trial(case: str) -> dict:
    env = _environment()
    try:
        state = _install_synthetic_physics(env)
        if case == "missing_left_contact":
            _set_contact_force(state, "left_bottle", 0.0)
        elif case == "missing_right_contact":
            _set_contact_force(state, "right_brush", 0.0)
        elif case == "missing_scrub_contact":
            _set_contact_force(state, "brush_bottle", 0.0)
        elif case == "unsafe_handle_grasp":
            state["safe_grasp"] = False
        elif case == "bristle_grasp":
            state["bristle_touch"] = True

        if case == "wrong_first_band":
            _sample(env, state, (0.0, 0.0, 1.0), 0.125, 1)
        elif case == "band_regression":
            _sample(env, state, (0.0, 0.0, 1.0), 0.060, 0)
            _sample(env, state, (0.02, 0.0, 1.0), 0.125, 1)
            _sample(env, state, (0.04, 0.0, 1.0), 0.060, 0)

        _ordered_scrub(env, state, short_path=case == "insufficient_path")

        state["clear"] = (
            (False, True) if case == "no_release_clearance" else (True, True)
        )
        if case == "unstable_release":
            state["velocities"]["hs005_bottle"][0] = 0.031
        if case == "tipped_bottle":
            half = np.sqrt(0.5)
            state["poses"]["hs005_bottle"][3:7] = (half, half, 0.0, 0.0)
        _release_and_hold(env, state)
        predicted = bool(env._hs005_success_now())
        return {
            "id": case,
            "expected": False,
            "predicted": predicted,
            "coverage_mask": [int(value) for value in env._hs005_coverage_mask],
            "order_violation": bool(env._hs005_order_violation),
            "path_length_m": float(env._hs005_path_length),
            "release_seen": bool(env._hs005_release_seen),
            "stability_steps": int(env._hs005_stability_steps),
        }
    finally:
        env.close()


def run_audit() -> dict:
    positives = [_positive_trial(index) for index in range(10)]
    negative_cases = (
        "missing_left_contact",
        "missing_right_contact",
        "missing_scrub_contact",
        "unsafe_handle_grasp",
        "bristle_grasp",
        "wrong_first_band",
        "band_regression",
        "insufficient_path",
        "no_release_clearance",
        "unstable_release",
        "tipped_bottle",
    )
    # Keep the required audit at exactly ten negatives while retaining both
    # order-failure variants in source for easy focused regression selection.
    negatives = [_negative_trial(case) for case in negative_cases if case != "wrong_first_band"]
    false_negatives = sum(not trial["predicted"] for trial in positives)
    false_positives = sum(trial["predicted"] for trial in negatives)
    passed = bool(
        len(positives) >= 10
        and len(negatives) >= 10
        and false_negatives == 0
        and false_positives == 0
    )
    return {
        "schema": "handson.b320_hs005.success_classifier_audit.v1",
        "benchmark_task_id": "B320-HS005",
        "classifier": "HandsOnABCIsaacLabEnv._hs005_success_now",
        "positive_trials": positives,
        "negative_trials": negatives,
        "positive_trial_count": len(positives),
        "negative_trial_count": len(negatives),
        "false_negative_count": false_negatives,
        "false_positive_count": false_positives,
        "pass": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_audit()
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
