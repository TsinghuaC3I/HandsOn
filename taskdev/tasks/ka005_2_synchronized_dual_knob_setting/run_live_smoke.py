#!/usr/bin/env python3
"""Run the B320-KA005-2 fixture through the real Isaac Sim 5.1 runtime.

This is intentionally a full collection-environment smoke test, not a USD-only
preview.  It verifies the named PhysX DOFs, all four hand-to-knob contact views,
the three-body/three-collider fixture, the low tabletop envelope, and passive
zero-action stability.  Run it only when enough GPU memory is available.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from handson.tasks.t38_set_dual_knobs.config import TaskConfig
from handson.sim.isaaclab.b320_bimanual_scene import (
    B320_KA005_2_TARGET_MARKER_PATHS,
)


TASK_ROOT = "/World/Task/B320Benchmark"
EXPECTED_BODIES = {"panel_base", "knob_left", "knob_right"}
EXPECTED_CONTACTS = {"LL", "RR", "LR", "RL"}
EXPECTED_DOFS = ("left_knob_joint", "right_knob_joint")
EXPECTED_COLLIDER_TYPES = {
    "panel_base": "Cube",
    "knob_left": "Cylinder",
    "knob_right": "Cylinder",
}
OPTIONAL_PRIOR_MODULE = "handson.sim.isaaclab." + "dual_knob_panel"


def _mark(message: str) -> None:
    print(f"[KA005-2 smoke] {message}", flush=True)


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    return value


def _assert_cleanroom_import_boundary() -> None:
    if OPTIONAL_PRIOR_MODULE in sys.modules:
        raise AssertionError(
            "clean-room launch imported the optional prior task module"
        )


def _assert_fixture_reset_isolation(
    raw: Any,
    fixture_paths: set[str],
) -> dict[str, Any]:
    reset_body_paths = {
        str(entry[0]) for entry in raw._task_reset_body_states
    }
    overlap = sorted(fixture_paths & reset_body_paths)
    if overlap:
        raise AssertionError(
            "articulated fixture bodies leaked into independent rigid-body reset "
            f"truth: {overlap}"
        )
    return {
        "fixture_paths": sorted(fixture_paths),
        "reset_body_paths": sorted(reset_body_paths),
        "overlap": overlap,
    }


def _quat_yaw_rad(orientation: Any) -> float:
    real = float(orientation.GetReal())
    imaginary = orientation.GetImaginary()
    x, y, z = (float(imaginary[index]) for index in range(3))
    return math.atan2(
        2.0 * (real * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _angle_error_rad(actual: float, expected: float) -> float:
    return math.atan2(
        math.sin(actual - expected),
        math.cos(actual - expected),
    )


def _assert_marker_targets(
    stage: Any,
    target_rad: tuple[float, float] | list[float] | np.ndarray,
) -> dict[str, Any]:
    marker_report: dict[str, Any] = {}
    for index, side in enumerate(("left", "right")):
        marker_path = B320_KA005_2_TARGET_MARKER_PATHS[side]
        marker = stage.GetPrimAtPath(marker_path)
        if not marker.IsValid():
            raise AssertionError(f"{side} target marker is missing: {marker_path}")
        expected = float(target_rad[index])
        custom_angle_value = marker.GetCustomDataByKey(
            "handson:targetAngleRad"
        )
        if custom_angle_value is None:
            raise AssertionError(
                f"{side} target marker has no targetAngleRad customData"
            )
        custom_angle = float(custom_angle_value)
        orient_attribute = marker.GetAttribute("xformOp:orient")
        orientation = orient_attribute.Get() if orient_attribute.IsValid() else None
        if orientation is None:
            raise AssertionError(
                f"{side} target marker has no authored xformOp:orient"
            )
        orient_angle = _quat_yaw_rad(orientation)
        custom_error = _angle_error_rad(custom_angle, expected)
        orient_error = _angle_error_rad(orient_angle, expected)
        if abs(custom_error) > 1.0e-8:
            raise AssertionError(
                f"{side} marker customData target mismatch: "
                f"expected={expected} actual={custom_angle}"
            )
        if abs(orient_error) > 1.0e-5:
            raise AssertionError(
                f"{side} marker orientation target mismatch: "
                f"expected={expected} actual={orient_angle}"
            )
        marker_report[side] = {
            "path": marker_path,
            "expected_target_rad": expected,
            "custom_data_target_rad": custom_angle,
            "orientation_target_rad": orient_angle,
            "custom_data_error_rad": custom_error,
            "orientation_error_rad": orient_error,
        }
    return marker_report


def _assert_analytic_colliders(
    stage: Any,
    spawn: Any,
    usd_module: Any,
    usd_physics_module: Any,
) -> tuple[dict[str, Any], list[str]]:
    collider_report: dict[str, Any] = {}
    all_collider_paths: list[str] = []
    for body_name, body_path in spawn.body_paths.items():
        body_prim = stage.GetPrimAtPath(body_path)
        colliders = [
            prim
            for prim in usd_module.PrimRange(body_prim)
            if prim.HasAPI(usd_physics_module.CollisionAPI)
        ]
        if len(colliders) != 1:
            raise AssertionError(
                f"{body_name} must have exactly one analytic collider, "
                f"found {[str(prim.GetPath()) for prim in colliders]}"
            )
        collider = colliders[0]
        collider_path = str(collider.GetPath())
        collider_type = str(collider.GetTypeName())
        expected_type = EXPECTED_COLLIDER_TYPES[body_name]
        if collider_type != expected_type:
            raise AssertionError(
                f"{body_name} collider must be {expected_type}, "
                f"found {collider_type} at {collider_path}"
            )
        collider_report[body_name] = {
            "path": collider_path,
            "type": collider_type,
        }
        all_collider_paths.append(collider_path)
    return collider_report, sorted(all_collider_paths)


def run(*, steps: int, seed: int, report: Path | None = None) -> dict[str, Any]:
    environment = None
    raw = None
    try:
        _mark("launching full collection environment")
        environment = TaskConfig().get_environment(
            policy_mode=True,
            render_mode="rgb_array",
            randomize=False,
            sim_backend="isaacsim",
            physics_backend="physx",
            launch_isaac=True,
            headless=True,
            image_obs=False,
            seed=seed,
            isaac_device="cuda:0",
        )
        _mark("environment constructed; resetting")
        observation, reset_info = environment.reset(seed=seed)
        raw = environment.unwrapped
        # Import Kit-provided USD modules only after the environment has
        # launched SimulationApp.
        from pxr import Usd, UsdGeom, UsdPhysics
        _mark("reset complete; checking timing and declared topology")

        _assert_cleanroom_import_boundary()

        if not math.isclose(raw._control_dt, 1.0 / 30.0, abs_tol=1.0e-12):
            raise AssertionError(f"unexpected control_dt={raw._control_dt}")
        if not math.isclose(raw._sim_dt, 1.0 / 120.0, abs_tol=1.0e-12):
            raise AssertionError(f"unexpected physics_dt={raw._sim_dt}")
        if int(raw._n_substeps) != 4:
            raise AssertionError(f"unexpected n_substeps={raw._n_substeps}")

        spawn = raw._b320_spawn_result
        if set(spawn.body_paths) != EXPECTED_BODIES:
            raise AssertionError(f"unexpected body roles={spawn.body_paths}")
        if tuple(spawn.dof_names) != EXPECTED_DOFS:
            raise AssertionError(f"unexpected spawn DOFs={spawn.dof_names}")
        if sum(int(value) for value in spawn.collider_counts.values()) != 3:
            raise AssertionError(f"unexpected collider counts={spawn.collider_counts}")
        if set(raw._ka005_2_contact_views) != EXPECTED_CONTACTS:
            raise AssertionError(
                f"four-pair contact truth is incomplete: {raw._ka005_2_contact_views}"
            )
        fixture_paths = set(spawn.body_paths.values())
        deterministic_reset_isolation = _assert_fixture_reset_isolation(
            raw,
            fixture_paths,
        )
        _mark("declared topology and four contact views are present")

        articulation_entry = raw._task_articulation_views.get("panel_base")
        if articulation_entry is None:
            raise AssertionError("panel_base articulation truth is missing")
        dof_indices = articulation_entry[0].get_metatype(0).dof_indices
        if set(EXPECTED_DOFS) - set(dof_indices):
            raise AssertionError(f"named PhysX DOFs are incomplete: {dof_indices}")
        _mark("named PhysX articulation DOFs are readable")

        stage = raw._stage
        body_prims = [stage.GetPrimAtPath(path) for path in spawn.body_paths.values()]
        rigid_paths = sorted(
            str(prim.GetPath())
            for prim in body_prims
            if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        collider_report, collider_paths = _assert_analytic_colliders(
            stage,
            spawn,
            Usd,
            UsdPhysics,
        )
        if len(rigid_paths) != 3 or len(collider_paths) != 3:
            raise AssertionError(
                f"fixture topology is not 3 bodies/3 colliders: "
                f"rigid={rigid_paths} colliders={collider_paths}"
            )
        _mark(
            "live USD topology is 3 rigid bodies / 3 analytic colliders "
            "(Cube/Cylinder/Cylinder)"
        )

        _mark("computing live fixture and marker bounds")
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
            ignoreVisibility=False,
        )
        body_boxes = [
            bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            for prim in body_prims
        ]
        world_min = np.min(
            [np.asarray(box.GetMin(), dtype=np.float64) for box in body_boxes], axis=0
        )
        world_max = np.max(
            [np.asarray(box.GetMax(), dtype=np.float64) for box in body_boxes], axis=0
        )
        if world_min[2] < 0.889 or world_max[2] > 0.995:
            raise AssertionError(
                f"fixture escaped the low tabletop envelope: min={world_min} max={world_max}"
            )

        for side in ("left", "right"):
            marker_path = B320_KA005_2_TARGET_MARKER_PATHS[side]
            marker = stage.GetPrimAtPath(marker_path)
            marker_visual = stage.GetPrimAtPath(f"{marker_path}/visual")
            if not marker.IsValid() or not marker_visual.IsValid():
                raise AssertionError(f"{side} target marker hierarchy is missing")
            marker_size = np.asarray(
                bbox_cache.ComputeLocalBound(marker_visual).GetRange().GetSize(),
                dtype=np.float64,
            )
            if float(marker_size.max()) > 0.020:
                raise AssertionError(
                    f"{side} target marker unexpectedly expanded: {marker_size}"
                )
        deterministic_reset_sample = raw._ka005_2_reset_sample.as_dict()
        deterministic_marker_report = _assert_marker_targets(
            stage,
            raw._ka005_2_reset_sample.target_rad,
        )
        _mark("fixture height, target-marker scale, and fixed target angles are valid")

        _mark(f"running randomized reset with seed {seed + 1}")
        raw.randomize = True
        randomized_observation, randomized_reset_info = environment.reset(
            seed=seed + 1
        )
        _assert_cleanroom_import_boundary()
        randomized_reset_isolation = _assert_fixture_reset_isolation(
            raw,
            fixture_paths,
        )
        randomized_reset_sample = raw._ka005_2_reset_sample.as_dict()
        if randomized_reset_sample["fallback_used"]:
            raise AssertionError("randomized reset unexpectedly used the fallback pair")
        randomized_marker_report = _assert_marker_targets(
            stage,
            raw._ka005_2_reset_sample.target_rad,
        )
        if set(raw._ka005_2_contact_views) != EXPECTED_CONTACTS:
            raise AssertionError(
                "four-pair contact truth changed after randomized reset: "
                f"{raw._ka005_2_contact_views}"
            )
        _mark("randomized reset and target-marker revalidation passed")

        q_initial, qd_initial, dof_available = raw._ka005_2_physics_dof_state()
        if not dof_available:
            raise AssertionError("initial knob DOF truth is unavailable")
        randomized_initial_error_rad = np.asarray(
            q_initial,
            dtype=np.float64,
        ) - np.asarray(
            raw._ka005_2_reset_sample.initial_rad,
            dtype=np.float64,
        )
        if np.any(np.abs(randomized_initial_error_rad) > 1.0e-5):
            raise AssertionError(
                "randomized reset did not reach its sampled initial angles: "
                f"error={randomized_initial_error_rad}"
            )
        base_initial = raw._pose("panel_base").copy()
        last_info = dict(randomized_reset_info)
        _mark(f"running {steps} zero-action control steps")
        for index in range(steps):
            _obs, _reward, terminated, truncated, last_info = raw.step(None)
            if terminated or truncated:
                raise AssertionError(
                    "zero-action smoke terminated unexpectedly: "
                    f"terminated={terminated} truncated={truncated} info={last_info}"
                )
            _mark(f"zero-action step {index + 1}/{steps} complete")

        q_final, qd_final, dof_available = raw._ka005_2_physics_dof_state()
        if not dof_available or not raw._ka005_2_state.contact_truth_available:
            raise AssertionError("live DOF/contact truth did not remain readable")
        base_final = raw._pose("panel_base").copy()
        base_drift_m = float(np.linalg.norm(base_final[:3] - base_initial[:3]))
        joint_drift_rad = np.abs(q_final - q_initial)
        if base_drift_m > 0.002:
            raise AssertionError(f"panel base drifted {base_drift_m} m")
        if np.any(joint_drift_rad > math.radians(0.25)):
            raise AssertionError(f"passive knob drift is too large: {joint_drift_rad}")
        _mark("zero-action stability checks passed")

        result = _as_jsonable({
            "schema": "handson_ka005_2_live_smoke_v2",
            "task_id": "B320-KA005-2",
            "task_name": "t38_set_dual_knobs",
            "isaac_sim_version": "5.1",
            "isaaclab_version": "2.3.2",
            "seed": seed,
            "steps": steps,
            "control_hz": 30,
            "physics_hz": 120,
            "body_paths": spawn.body_paths,
            "rigid_body_paths": rigid_paths,
            "collider_paths": collider_paths,
            "analytic_colliders": collider_report,
            "dof_names": list(EXPECTED_DOFS),
            "contact_pairs": sorted(raw._ka005_2_contact_views),
            "deterministic_reset": {
                "seed": seed,
                "sample": deterministic_reset_sample,
                "fixture_reset_isolation": deterministic_reset_isolation,
                "target_markers": deterministic_marker_report,
                "info": reset_info,
            },
            "randomized_reset": {
                "seed": seed + 1,
                "sample": randomized_reset_sample,
                "fixture_reset_isolation": randomized_reset_isolation,
                "target_markers": randomized_marker_report,
                "initial_joint_error_rad": randomized_initial_error_rad,
                "info": randomized_reset_info,
            },
            "fixture_world_min_m": world_min,
            "fixture_world_max_m": world_max,
            "base_drift_m": base_drift_m,
            "joint_initial_rad": q_initial,
            "joint_initial_velocity_rad_s": qd_initial,
            "joint_final_rad": q_final,
            "joint_final_velocity_rad_s": qd_final,
            "joint_drift_rad": joint_drift_rad,
            "observation_space_contains_reset": bool(
                environment.observation_space.contains(observation)
            ),
            "observation_space_contains_randomized_reset": bool(
                environment.observation_space.contains(randomized_observation)
            ),
            "success": True,
            "last_info": last_info,
        })
        # The supported Kit skip-cleanup shutdown can terminate the process
        # from SimulationApp.close().  Persist and print the result before
        # entering the finally block so successful evidence is never lost.
        if report is not None:
            report = report.resolve()
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _mark(f"wrote evidence report: {report}")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return result
    finally:
        if environment is not None:
            _mark("closing Isaac environment")
            # Headless/no-camera diagnostics can use the runtime's supported
            # skip-cleanup path.  SimulationContext.stop() is known to block
            # after some PhysX tensor sessions even when no cameras exist.
            if raw is not None:
                raw._fast_shutdown_on_close = True
            environment.close()
            _mark("Isaac environment closed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parent / "evidence/live_smoke.json",
    )
    arguments = parser.parse_args()
    if arguments.steps < 1:
        parser.error("--steps must be positive")
    run(
        steps=arguments.steps,
        seed=arguments.seed,
        report=arguments.report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
