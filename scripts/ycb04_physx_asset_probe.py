#!/usr/bin/env python3
"""Isaac Sim 5.1 settle and extraction-path probe for YCB04."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import threading
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "handson"))

from isaaclab.app import AppLauncher

from handson.sim.isaaclab.runtime_display import enforce_isaac_viewport_only, isaac_kit_args


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--mode",
    choices=(
        "settle",
        "path",
        "pose_sweep",
        "material_sweep",
        "contact_sweep",
        "yaw_sweep",
        "damping_sweep",
        "motion_window",
    ),
    default="settle",
)
parser.add_argument("--seeds", type=int, default=8)
parser.add_argument("--settle_steps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not any(argument == "--device" or argument.startswith("--device=") for argument in sys.argv[1:]):
    args_cli.device = "cuda:0"
args_cli.kit_args = isaac_kit_args(args_cli.kit_args, headless=args_cli.headless)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
enforce_isaac_viewport_only(simulation_app, headless=args_cli.headless)


import torch  # noqa: E402
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv  # noqa: E402
from handson.sim.isaaclab import ycb04_semantics as semantics  # noqa: E402


_TASK_NAME = "t41_sort_nested_cups"
_CUPS = ("ycb04_large_cup", "ycb04_medium_cup", "ycb04_small_cup")


def _poses(env: HandsOnABCIsaacLabEnv) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in _CUPS:
        pose = env._physics_body_pose(name)
        if pose is None:
            raise RuntimeError(f"Missing PhysX pose for {name}")
        result[name] = pose
    return result


def _velocities(env: HandsOnABCIsaacLabEnv) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in _CUPS:
        velocity = env._physics_body_velocity(name)
        if velocity is None:
            raise RuntimeError(f"Missing PhysX velocity for {name}")
        result[name] = velocity
    return result


def _upright_alignment(pose: np.ndarray) -> float:
    _w, x, y, _z = np.asarray(pose[3:7], dtype=np.float64)
    return float(1.0 - 2.0 * (x * x + y * y))


def _settle_probe(env: HandsOnABCIsaacLabEnv) -> bool:
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    passed = True
    xy_samples: list[np.ndarray] = []
    yaw_samples: list[float] = []
    for seed in range(args_cli.seeds):
        env.reset(seed=seed)
        initial = _poses(env)
        for _ in range(args_cli.settle_steps):
            _obs, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                raise RuntimeError(f"Episode ended during settle probe: seed={seed} info={info}")
        final = _poses(env)
        velocities = _velocities(env)
        evidence = env._ycb04_cups_metrics(update_events=True)

        common_xy = final[_CUPS[0]][:2] - np.asarray([-0.25, 0.0])
        xy_samples.append(common_xy)
        quaternion = final[_CUPS[0]][3:7]
        yaw_samples.append(
            math.atan2(
                2.0 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
                1.0 - 2.0 * (quaternion[2] ** 2 + quaternion[3] ** 2),
            )
        )
        translation = max(
            float(np.linalg.norm(final[name][:3] - initial[name][:3])) for name in _CUPS
        )
        linear_speeds = {
            name: float(np.linalg.norm(velocities[name][:3])) for name in _CUPS
        }
        angular_speeds = {
            name: float(np.linalg.norm(velocities[name][3:])) for name in _CUPS
        }
        speed = max(linear_speeds.values())
        angular_speed = max(angular_speeds.values())
        alignment = min(_upright_alignment(final[name]) for name in _CUPS)
        medium_large_xy = float(
            np.linalg.norm(final["ycb04_medium_cup"][:2] - final["ycb04_large_cup"][:2])
        )
        small_medium_xy = float(
            np.linalg.norm(final["ycb04_small_cup"][:2] - final["ycb04_medium_cup"][:2])
        )
        large_bottom = float(final["ycb04_large_cup"][2] - 0.032 - env._current_table_top_z())
        seed_passed = bool(
            np.isfinite(np.concatenate(tuple(final.values()))).all()
            and translation <= 0.012
            and speed < 0.02
            and angular_speed < 0.3
            and alignment >= math.cos(math.radians(10.0))
            and medium_large_xy <= 0.006
            and small_medium_xy <= 0.006
            and large_bottom >= -0.002
            and evidence["initial_nested_valid"]
            and evidence["contact_available"]
            and evidence["small_medium_contact_seen"]
            and evidence["medium_large_contact_seen"]
            and not evidence["small_extracted_from_medium"]
            and not evidence["medium_extracted_from_large"]
        )
        passed = passed and seed_passed
        print(
            f"[YCB04][seed={seed}] pass={seed_passed} translation_m={translation:.6f} "
            f"speed_mps={speed:.6f} angular_speed_radps={angular_speed:.6f} "
            f"linear_by_cup={linear_speeds} angular_by_cup={angular_speeds} "
            f"alignment={alignment:.6f} "
            f"medium_large_xy_m={medium_large_xy:.6f} "
            f"small_medium_xy_m={small_medium_xy:.6f} large_bottom_m={large_bottom:.6f} "
            f"initial_nested_valid={evidence['initial_nested_valid']} "
            f"contact_available={evidence['contact_available']} "
            f"pair_contact_seen=({evidence['small_medium_contact_seen']},"
            f"{evidence['medium_large_contact_seen']})",
            flush=True,
        )

    xy_array = np.asarray(xy_samples)
    xy_span = np.ptp(xy_array, axis=0)
    yaw_span = float(np.ptp(np.unwrap(np.asarray(yaw_samples))))
    randomization_passed = bool(np.all(xy_span > 1.0e-4) and yaw_span > math.radians(0.1))
    print(
        f"[YCB04] reset_xy_span_m={xy_span.tolist()} reset_yaw_span_rad={yaw_span:.6f} "
        f"randomization_pass={randomization_passed}",
        flush=True,
    )
    return passed and randomization_passed


def _set_disable_gravity(env: HandsOnABCIsaacLabEnv) -> None:
    for name in _CUPS:
        prim = env._find_rigid_body_prim(name)
        if prim is None:
            raise RuntimeError(f"Missing cup rigid body: {name}")
        api = PhysxSchema.PhysxRigidBodyAPI(prim)
        if not api:
            api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        api.CreateDisableGravityAttr().Set(True)


def _print_collider_contract(env: HandsOnABCIsaacLabEnv) -> bool:
    """Prove that each original YCB rim has an open-mesh collision recipe."""
    passed = True
    for name in _CUPS:
        body = env._find_rigid_body_prim(name)
        if body is None:
            print(f"[YCB04][collider={name}] pass=False reason=missing_body", flush=True)
            passed = False
            continue
        colliders = [
            prim
            for prim in Usd.PrimRange(body)
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        active_colliders = []
        for prim in colliders:
            attr = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
            if attr and attr.IsValid() and attr.Get() is False:
                continue
            active_colliders.append(prim)
        rims = []
        for prim in active_colliders:
            mesh = UsdGeom.Mesh(prim) if prim.IsA(UsdGeom.Mesh) else None
            points = mesh.GetPointsAttr().Get() if mesh else None
            if points is not None and len(points) == 864:
                rims.append(prim)
        approximations: list[str] = []
        for prim in rims:
            mesh_api = UsdPhysics.MeshCollisionAPI(prim)
            approximation = mesh_api.GetApproximationAttr().Get() if mesh_api else None
            approximations.append(str(approximation))
        articulation_root = body.HasAPI(UsdPhysics.ArticulationRootAPI)
        physx_articulation = body.HasAPI(PhysxSchema.PhysxArticulationAPI)
        active_approximations = []
        for prim in active_colliders:
            mesh_api = UsdPhysics.MeshCollisionAPI(prim)
            approximation = (
                mesh_api.GetApproximationAttr().Get() if mesh_api else None
            )
            active_approximations.append(str(approximation))
        cup_passed = bool(
            len(colliders) == 27
            and len(active_colliders) == 27
            and len(rims) == 1
            and approximations == ["convexDecomposition"]
            and not articulation_root
            and not physx_articulation
        )
        passed = passed and cup_passed
        print(
            f"[YCB04][collider={name}] pass={cup_passed} "
            f"count={len(colliders)} active_count={len(active_colliders)} "
            f"rim_paths={[str(prim.GetPath()) for prim in rims]} "
            f"approximation={approximations} "
            f"active_approximations={active_approximations} "
            f"articulation_root={articulation_root} "
            f"physx_articulation={physx_articulation}",
            flush=True,
        )
    return passed


def _path_probe(env: HandsOnABCIsaacLabEnv) -> bool:
    """Motor the grasped layer upward while the other hand holds the outer cup."""
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    _set_disable_gravity(env)
    passed = True
    scenarios = (
        ("small", ("ycb04_small_cup",)),
        ("medium_with_small", ("ycb04_medium_cup", "ycb04_small_cup")),
    )
    for label, names in scenarios:
        env.reset(seed=0)
        initial = _poses(env)
        held_names = tuple(name for name in _CUPS if name not in names)
        held_transforms = {
            name: env._task_body_views[name].get_transforms().clone()
            for name in held_names
        }
        checkpoints: dict[float, dict[str, float]] = {}

        def apply_controlled_extraction(
            _substep_index: int,
            _substeps: int,
        ) -> None:
            for name in _CUPS:
                view = env._task_body_views[name]
                velocities = torch.zeros_like(view.get_velocities())
                if name in names:
                    velocities[:, 2] = 0.12
                indices = torch.arange(
                    int(view.count),
                    dtype=torch.long,
                    device=velocities.device,
                )
                if name in held_names:
                    view.set_transforms(held_transforms[name], indices)
                view.set_velocities(velocities, indices)

        env._handson_pre_physics_substep_callback = apply_controlled_extraction
        try:
            for _ in range(120):
                _obs, _reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    raise RuntimeError(
                        f"Episode ended during path probe: label={label} info={info}"
                    )
                current = _poses(env)
                numeric = np.concatenate(tuple(current.values()))
                if not np.all(np.isfinite(numeric)):
                    raise RuntimeError(
                        f"Non-finite pose during path probe: label={label}"
                    )
                progress = min(
                    float(current[name][2] - initial[name][2]) for name in names
                )
                lateral = max(
                    float(np.linalg.norm(current[name][:2] - initial[name][:2]))
                    for name in names
                )
                held_displacement = max(
                    (
                        float(
                            np.linalg.norm(
                                current[name][:3] - initial[name][:3]
                            )
                        )
                        for name in held_names
                    ),
                    default=0.0,
                )
                for height in (0.01, 0.03, 0.07):
                    if progress >= height and height not in checkpoints:
                        checkpoints[height] = {
                            "progress": progress,
                            "lateral": lateral,
                            "held_displacement": held_displacement,
                        }
                if progress >= 0.075:
                    break
        finally:
            delattr(env, "_handson_pre_physics_substep_callback")

        final = _poses(env)
        progress = min(
            float(final[name][2] - initial[name][2]) for name in names
        )
        max_lateral = max(
            float(np.linalg.norm(final[name][:2] - initial[name][:2]))
            for name in names
        )
        held_displacement = max(
            (
                float(np.linalg.norm(final[name][:3] - initial[name][:3]))
                for name in held_names
            ),
            default=0.0,
        )
        if label == "small":
            fully_separated = semantics.pair_is_fully_separated(
                final["ycb04_small_cup"],
                semantics.SMALL_CUP,
                final["ycb04_medium_cup"],
                semantics.MEDIUM_CUP,
            )
        else:
            fully_separated = semantics.pair_is_fully_separated(
                final["ycb04_medium_cup"],
                semantics.MEDIUM_CUP,
                final["ycb04_large_cup"],
                semantics.LARGE_CUP,
            )
        scenario_passed = bool(
            set(checkpoints) == {0.01, 0.03, 0.07}
            and progress >= 0.075
            and max_lateral <= 0.005
            and held_displacement <= 0.005
            and fully_separated
        )
        passed = passed and scenario_passed
        for height in (0.01, 0.03, 0.07):
            reading = checkpoints.get(height, {})
            print(
                f"[YCB04][path={label}][height_m={height:.3f}] "
                f"crossed={bool(reading)} "
                f"progress_m={float(reading.get('progress', 0.0)):.6f} "
                f"lateral_m={float(reading.get('lateral', math.inf)):.6f} "
                f"held_displacement_m={float(reading.get('held_displacement', math.inf)):.6f}",
                flush=True,
            )
        print(
            f"[YCB04][path={label}] pass={scenario_passed} "
            f"progress_m={progress:.6f} max_lateral_m={max_lateral:.6f} "
            f"held_displacement_m={held_displacement:.6f} "
            f"fully_separated={fully_separated}",
            flush=True,
        )
    return passed


def _author_nested_heights(
    env: HandsOnABCIsaacLabEnv,
    medium_large_z: float,
    small_medium_z: float,
) -> dict[str, np.ndarray]:
    table_top = env._current_table_top_z()
    centers = {
        "ycb04_large_cup": table_top + semantics.LARGE_CUP.half_height,
        "ycb04_medium_cup": (
            table_top + semantics.LARGE_CUP.half_height + medium_large_z
        ),
        "ycb04_small_cup": (
            table_top
            + semantics.LARGE_CUP.half_height
            + medium_large_z
            + small_medium_z
        ),
    }
    for name, center_z in centers.items():
        view = env._task_body_views[name]
        transform = view.get_transforms().clone()
        transform[:, 2] = float(center_z)
        indices = torch.arange(
            int(view.count),
            dtype=torch.long,
            device=transform.device,
        )
        view.set_transforms(transform, indices)
        view.set_velocities(torch.zeros_like(view.get_velocities()), indices)
    if env._sim is not None:
        env._sim.forward()
    env._refresh_object_poses()
    authored = _poses(env)
    env._ycb04_initial_nested_valid = bool(
        semantics.pair_is_nested(
            authored["ycb04_small_cup"],
            semantics.SMALL_CUP,
            authored["ycb04_medium_cup"],
            semantics.MEDIUM_CUP,
        )
        and semantics.pair_is_nested(
            authored["ycb04_medium_cup"],
            semantics.MEDIUM_CUP,
            authored["ycb04_large_cup"],
            semantics.LARGE_CUP,
        )
    )
    env._ycb04_initial_evaluated = True
    env._ycb04_pair_contact_seen = {
        "small_medium": False,
        "medium_large": False,
    }
    env._ycb04_extracted = {
        "small_medium": False,
        "medium_large": False,
    }
    return authored


def _pose_sweep_probe(env: HandsOnABCIsaacLabEnv) -> bool:
    """Find a physical nested reset pose without changing asset geometry."""
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    candidates = (
        (medium_large_z, small_medium_z)
        for medium_large_z in (0.02018, 0.024, 0.028, 0.032)
        for small_medium_z in (0.00639, 0.010, 0.014, 0.018)
    )
    any_passed = False
    for medium_large_z, small_medium_z in candidates:
        env.reset(seed=0)
        authored = _author_nested_heights(
            env,
            medium_large_z,
            small_medium_z,
        )
        for _ in range(args_cli.settle_steps):
            _obs, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                raise RuntimeError(
                    "Episode ended during pose sweep: "
                    f"medium_large_z={medium_large_z} "
                    f"small_medium_z={small_medium_z} info={info}"
                )
        final = _poses(env)
        velocities = _velocities(env)
        metrics = env._ycb04_cups_metrics(update_events=True)
        linear_speed = max(
            float(np.linalg.norm(velocities[name][:3])) for name in _CUPS
        )
        angular_speed = max(
            float(np.linalg.norm(velocities[name][3:])) for name in _CUPS
        )
        displacement = max(
            float(np.linalg.norm(final[name][:3] - authored[name][:3]))
            for name in _CUPS
        )
        final_medium_large_z = float(
            final["ycb04_medium_cup"][2] - final["ycb04_large_cup"][2]
        )
        final_small_medium_z = float(
            final["ycb04_small_cup"][2] - final["ycb04_medium_cup"][2]
        )
        final_nested = bool(
            semantics.pair_is_nested(
                final["ycb04_small_cup"],
                semantics.SMALL_CUP,
                final["ycb04_medium_cup"],
                semantics.MEDIUM_CUP,
            )
            and semantics.pair_is_nested(
                final["ycb04_medium_cup"],
                semantics.MEDIUM_CUP,
                final["ycb04_large_cup"],
                semantics.LARGE_CUP,
            )
        )
        candidate_passed = bool(
            env._ycb04_initial_nested_valid
            and final_nested
            and metrics["contact_available"]
            and metrics["small_medium_contact_seen"]
            and metrics["medium_large_contact_seen"]
            and displacement <= 0.012
            and linear_speed < 0.02
            and angular_speed < 0.3
        )
        any_passed = any_passed or candidate_passed
        print(
            "[YCB04][pose_sweep] "
            f"pass={candidate_passed} authored_z=({medium_large_z:.5f},"
            f"{small_medium_z:.5f}) final_z=({final_medium_large_z:.5f},"
            f"{final_small_medium_z:.5f}) displacement_m={displacement:.6f} "
            f"linear_mps={linear_speed:.6f} angular_radps={angular_speed:.6f} "
            f"initial_nested={env._ycb04_initial_nested_valid} "
            f"final_nested={final_nested} "
            f"pair_contact_seen=({metrics['small_medium_contact_seen']},"
            f"{metrics['medium_large_contact_seen']})",
            flush=True,
        )
    return any_passed


def _set_cup_material_friction(
    env: HandsOnABCIsaacLabEnv,
    *,
    static_friction: float,
    dynamic_friction: float,
) -> None:
    if env._stage is None:
        raise RuntimeError("YCB04 material sweep requires a USD stage")
    material = env._stage.GetPrimAtPath(
        "/World/PhysicsMaterials/HandsOnYCB04Cups"
    )
    if not material.IsValid():
        raise RuntimeError("YCB04 cup physics material is missing")
    api = UsdPhysics.MaterialAPI(material)
    static_attr = api.GetStaticFrictionAttr()
    if not static_attr or not static_attr.IsValid():
        static_attr = api.CreateStaticFrictionAttr()
    dynamic_attr = api.GetDynamicFrictionAttr()
    if not dynamic_attr or not dynamic_attr.IsValid():
        dynamic_attr = api.CreateDynamicFrictionAttr()
    static_attr.Set(float(static_friction))
    dynamic_attr.Set(float(dynamic_friction))
    if env._sim is not None:
        env._sim.forward()


def _material_sweep_probe(env: HandsOnABCIsaacLabEnv) -> bool:
    """Select realistic plastic friction without altering cup geometry."""
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    candidates = (
        (0.95, 0.75),
        (0.70, 0.45),
        (0.50, 0.30),
        (0.35, 0.20),
        (0.25, 0.12),
    )
    any_passed = False
    for static_friction, dynamic_friction in candidates:
        _set_cup_material_friction(
            env,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
        )
        env.reset(seed=0)
        initial = _poses(env)
        for _ in range(args_cli.settle_steps):
            _obs, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                raise RuntimeError(
                    "Episode ended during material sweep: "
                    f"static={static_friction} dynamic={dynamic_friction} "
                    f"info={info}"
                )
        final = _poses(env)
        velocities = _velocities(env)
        metrics = env._ycb04_cups_metrics(update_events=True)
        displacement = max(
            float(np.linalg.norm(final[name][:3] - initial[name][:3]))
            for name in _CUPS
        )
        linear_speed = max(
            float(np.linalg.norm(velocities[name][:3])) for name in _CUPS
        )
        angular_speed = max(
            float(np.linalg.norm(velocities[name][3:])) for name in _CUPS
        )
        nested = bool(
            semantics.pair_is_nested(
                final["ycb04_small_cup"],
                semantics.SMALL_CUP,
                final["ycb04_medium_cup"],
                semantics.MEDIUM_CUP,
            )
            and semantics.pair_is_nested(
                final["ycb04_medium_cup"],
                semantics.MEDIUM_CUP,
                final["ycb04_large_cup"],
                semantics.LARGE_CUP,
            )
        )
        candidate_passed = bool(
            metrics["initial_nested_valid"]
            and nested
            and metrics["contact_available"]
            and metrics["small_medium_contact_seen"]
            and metrics["medium_large_contact_seen"]
            and displacement <= 0.012
            and linear_speed < 0.02
            and angular_speed < 0.3
        )
        any_passed = any_passed or candidate_passed
        print(
            "[YCB04][material_sweep] "
            f"pass={candidate_passed} static={static_friction:.2f} "
            f"dynamic={dynamic_friction:.2f} displacement_m={displacement:.6f} "
            f"linear_mps={linear_speed:.6f} angular_radps={angular_speed:.6f} "
            f"velocities={{{', '.join(f'{name}: {velocities[name].tolist()}' for name in _CUPS)}}} "
            f"initial_nested={metrics['initial_nested_valid']} final_nested={nested} "
            f"pair_contact_seen=({metrics['small_medium_contact_seen']},"
            f"{metrics['medium_large_contact_seen']})",
            flush=True,
        )
    return any_passed


def _set_cup_contact_offset(
    env: HandsOnABCIsaacLabEnv,
    contact_offset: float,
) -> int:
    if env._stage is None:
        raise RuntimeError("YCB04 contact sweep requires a USD stage")
    tuned = 0
    for name in _CUPS:
        body = env._find_rigid_body_prim(name)
        if body is None:
            continue
        for prim in Usd.PrimRange(body):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            api = PhysxSchema.PhysxCollisionAPI(prim)
            if not api:
                api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            attr = api.GetContactOffsetAttr()
            if not attr or not attr.IsValid():
                attr = api.CreateContactOffsetAttr()
            attr.Set(float(contact_offset))
            tuned += 1
    if tuned != 81:
        raise RuntimeError(
            f"Expected to tune 81 YCB04 colliders, tuned {tuned}"
        )
    if env._sim is not None:
        env._sim.forward()
    return tuned


def _contact_sweep_probe(env: HandsOnABCIsaacLabEnv) -> bool:
    """Find a persistent contact shell that still fits the 2.5 mm cup gaps."""
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    any_passed = False
    for contact_offset in (5.0e-5, 2.0e-4, 5.0e-4, 7.5e-4, 1.0e-3):
        _set_cup_contact_offset(env, contact_offset)
        env.reset(seed=0)
        initial = _poses(env)
        for _ in range(args_cli.settle_steps):
            _obs, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                raise RuntimeError(
                    "Episode ended during contact-offset sweep: "
                    f"contact_offset={contact_offset} info={info}"
                )
        final = _poses(env)
        velocities = _velocities(env)
        metrics = env._ycb04_cups_metrics(update_events=True)
        displacement = max(
            float(np.linalg.norm(final[name][:3] - initial[name][:3]))
            for name in _CUPS
        )
        linear_speed = max(
            float(np.linalg.norm(velocities[name][:3])) for name in _CUPS
        )
        angular_speed = max(
            float(np.linalg.norm(velocities[name][3:])) for name in _CUPS
        )
        nested = bool(
            semantics.pair_is_nested(
                final["ycb04_small_cup"],
                semantics.SMALL_CUP,
                final["ycb04_medium_cup"],
                semantics.MEDIUM_CUP,
            )
            and semantics.pair_is_nested(
                final["ycb04_medium_cup"],
                semantics.MEDIUM_CUP,
                final["ycb04_large_cup"],
                semantics.LARGE_CUP,
            )
        )
        candidate_passed = bool(
            metrics["initial_nested_valid"]
            and nested
            and metrics["contact_available"]
            and metrics["small_medium_contact_seen"]
            and metrics["medium_large_contact_seen"]
            and displacement <= 0.012
            and linear_speed < 0.02
            and angular_speed < 0.3
        )
        any_passed = any_passed or candidate_passed
        print(
            "[YCB04][contact_sweep] "
            f"pass={candidate_passed} contact_offset_m={contact_offset:.7f} "
            f"displacement_m={displacement:.6f} linear_mps={linear_speed:.6f} "
            f"angular_radps={angular_speed:.6f} "
            f"velocities={{{', '.join(f'{name}: {velocities[name].tolist()}' for name in _CUPS)}}} "
            f"initial_nested={metrics['initial_nested_valid']} final_nested={nested} "
            f"pair_contact_seen=({metrics['small_medium_contact_seen']},"
            f"{metrics['medium_large_contact_seen']})",
            flush=True,
        )
    return any_passed


def _author_nested_yaws(
    env: HandsOnABCIsaacLabEnv,
    medium_large_yaw: float,
    small_medium_yaw: float,
) -> dict[str, np.ndarray]:
    yaws = {
        "ycb04_large_cup": 0.0,
        "ycb04_medium_cup": medium_large_yaw,
        "ycb04_small_cup": medium_large_yaw + small_medium_yaw,
    }
    for name, yaw in yaws.items():
        view = env._task_body_views[name]
        transform = view.get_transforms().clone()
        transform[:, 3:7] = torch.as_tensor(
            [0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)],
            dtype=transform.dtype,
            device=transform.device,
        )
        indices = torch.arange(
            int(view.count),
            dtype=torch.long,
            device=transform.device,
        )
        view.set_velocities(torch.zeros_like(view.get_velocities()), indices)
        view.set_transforms(transform, indices)
    if env._sim is not None:
        env._sim.forward()
    env._refresh_object_poses()
    authored = _poses(env)
    env._ycb04_initial_nested_valid = bool(
        semantics.pair_is_nested(
            authored["ycb04_small_cup"],
            semantics.SMALL_CUP,
            authored["ycb04_medium_cup"],
            semantics.MEDIUM_CUP,
        )
        and semantics.pair_is_nested(
            authored["ycb04_medium_cup"],
            semantics.MEDIUM_CUP,
            authored["ycb04_large_cup"],
            semantics.LARGE_CUP,
        )
    )
    env._ycb04_initial_evaluated = True
    env._ycb04_pair_contact_seen = {
        "small_medium": False,
        "medium_large": False,
    }
    env._ycb04_extracted = {
        "small_medium": False,
        "medium_large": False,
    }
    return authored


def _yaw_sweep_probe(env: HandsOnABCIsaacLabEnv) -> bool:
    """Break discrete-facet symmetry while preserving the original meshes."""
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    degree_pairs = (
        (0.0, 0.0),
        (3.75, 0.0),
        (7.5, 0.0),
        (0.0, 3.75),
        (0.0, 7.5),
        (3.75, 3.75),
        (7.5, 7.5),
        (7.5, -7.5),
    )
    any_passed = False
    for medium_large_deg, small_medium_deg in degree_pairs:
        env.reset(seed=0)
        authored = _author_nested_yaws(
            env,
            math.radians(medium_large_deg),
            math.radians(small_medium_deg),
        )
        for _ in range(args_cli.settle_steps):
            _obs, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                raise RuntimeError(
                    "Episode ended during yaw sweep: "
                    f"medium_large_deg={medium_large_deg} "
                    f"small_medium_deg={small_medium_deg} info={info}"
                )
        final = _poses(env)
        velocities = _velocities(env)
        metrics = env._ycb04_cups_metrics(update_events=True)
        displacement = max(
            float(np.linalg.norm(final[name][:3] - authored[name][:3]))
            for name in _CUPS
        )
        linear_speed = max(
            float(np.linalg.norm(velocities[name][:3])) for name in _CUPS
        )
        angular_speed = max(
            float(np.linalg.norm(velocities[name][3:])) for name in _CUPS
        )
        nested = bool(
            semantics.pair_is_nested(
                final["ycb04_small_cup"],
                semantics.SMALL_CUP,
                final["ycb04_medium_cup"],
                semantics.MEDIUM_CUP,
            )
            and semantics.pair_is_nested(
                final["ycb04_medium_cup"],
                semantics.MEDIUM_CUP,
                final["ycb04_large_cup"],
                semantics.LARGE_CUP,
            )
        )
        candidate_passed = bool(
            env._ycb04_initial_nested_valid
            and nested
            and metrics["contact_available"]
            and metrics["small_medium_contact_seen"]
            and metrics["medium_large_contact_seen"]
            and displacement <= 0.012
            and linear_speed < 0.02
            and angular_speed < 0.3
        )
        any_passed = any_passed or candidate_passed
        print(
            "[YCB04][yaw_sweep] "
            f"pass={candidate_passed} yaw_deg=({medium_large_deg:.2f},"
            f"{small_medium_deg:.2f}) displacement_m={displacement:.6f} "
            f"linear_mps={linear_speed:.6f} angular_radps={angular_speed:.6f} "
            f"velocities={{{', '.join(f'{name}: {velocities[name].tolist()}' for name in _CUPS)}}} "
            f"initial_nested={env._ycb04_initial_nested_valid} final_nested={nested} "
            f"pair_contact_seen=({metrics['small_medium_contact_seen']},"
            f"{metrics['medium_large_contact_seen']})",
            flush=True,
        )
    return any_passed


def _set_cup_damping(
    env: HandsOnABCIsaacLabEnv,
    *,
    linear_damping: float,
    angular_damping: float,
) -> int:
    if env._stage is None:
        raise RuntimeError("YCB04 damping sweep requires a USD stage")
    tuned = 0
    for name in _CUPS:
        body = env._find_rigid_body_prim(name)
        if body is None:
            continue
        api = PhysxSchema.PhysxRigidBodyAPI(body)
        if not api:
            api = PhysxSchema.PhysxRigidBodyAPI.Apply(body)
        linear_attr = api.GetLinearDampingAttr()
        if not linear_attr or not linear_attr.IsValid():
            linear_attr = api.CreateLinearDampingAttr()
        angular_attr = api.GetAngularDampingAttr()
        if not angular_attr or not angular_attr.IsValid():
            angular_attr = api.CreateAngularDampingAttr()
        linear_attr.Set(float(linear_damping))
        angular_attr.Set(float(angular_damping))
        tuned += 1
    if tuned != 3:
        raise RuntimeError(f"Expected to tune 3 YCB04 bodies, tuned {tuned}")
    if env._sim is not None:
        env._sim.forward()
    return tuned


def _damping_sweep_probe(env: HandsOnABCIsaacLabEnv) -> bool:
    """Select the least ordinary rigid-body damping that settles the cups."""
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    any_passed = False
    for angular_damping in (2.0, 4.0, 6.0, 8.0, 12.0):
        _set_cup_damping(
            env,
            linear_damping=0.4,
            angular_damping=angular_damping,
        )
        env.reset(seed=0)
        initial = _poses(env)
        for _ in range(args_cli.settle_steps):
            _obs, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                raise RuntimeError(
                    "Episode ended during damping sweep: "
                    f"angular_damping={angular_damping} info={info}"
                )
        final = _poses(env)
        velocities = _velocities(env)
        metrics = env._ycb04_cups_metrics(update_events=True)
        displacement = max(
            float(np.linalg.norm(final[name][:3] - initial[name][:3]))
            for name in _CUPS
        )
        linear_speed = max(
            float(np.linalg.norm(velocities[name][:3])) for name in _CUPS
        )
        angular_speed = max(
            float(np.linalg.norm(velocities[name][3:])) for name in _CUPS
        )
        nested = bool(
            semantics.pair_is_nested(
                final["ycb04_small_cup"],
                semantics.SMALL_CUP,
                final["ycb04_medium_cup"],
                semantics.MEDIUM_CUP,
            )
            and semantics.pair_is_nested(
                final["ycb04_medium_cup"],
                semantics.MEDIUM_CUP,
                final["ycb04_large_cup"],
                semantics.LARGE_CUP,
            )
        )
        candidate_passed = bool(
            metrics["initial_nested_valid"]
            and nested
            and metrics["contact_available"]
            and metrics["small_medium_contact_seen"]
            and metrics["medium_large_contact_seen"]
            and displacement <= 0.012
            and linear_speed < 0.02
            and angular_speed < 0.3
        )
        any_passed = any_passed or candidate_passed
        print(
            "[YCB04][damping_sweep] "
            f"pass={candidate_passed} linear_damping=0.40 "
            f"angular_damping={angular_damping:.2f} "
            f"displacement_m={displacement:.6f} linear_mps={linear_speed:.6f} "
            f"angular_radps={angular_speed:.6f} "
            f"velocities={{{', '.join(f'{name}: {velocities[name].tolist()}' for name in _CUPS)}}} "
            f"initial_nested={metrics['initial_nested_valid']} final_nested={nested} "
            f"pair_contact_seen=({metrics['small_medium_contact_seen']},"
            f"{metrics['medium_large_contact_seen']})",
            flush=True,
        )
    return any_passed


def _quat_delta_angle(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(4)
    second = np.asarray(second, dtype=np.float64).reshape(4)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1.0e-12 or second_norm <= 1.0e-12:
        return math.inf
    cosine = float(
        np.clip(abs(np.dot(first / first_norm, second / second_norm)), 0.0, 1.0)
    )
    return float(2.0 * math.acos(cosine))


def _motion_window_probe(env: HandsOnABCIsaacLabEnv) -> bool:
    """Validate production 30 Hz motion against independent pose differencing."""
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    passed = True
    for seed in range(args_cli.seeds):
        env.reset(seed=seed)
        pose_history = [_poses(env)]
        production_velocity_history: list[dict[str, np.ndarray]] = []
        for _ in range(args_cli.settle_steps):
            _obs, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                raise RuntimeError(
                    f"Episode ended during motion-window probe: seed={seed} "
                    f"info={info}"
                )
            pose_history.append(_poses(env))
            production_velocity_history.append(
                {
                    name: velocity.copy()
                    for name, velocity in (
                        env._ycb04_observable_velocities.items()
                    )
                }
            )
        if len(pose_history) < 31:
            raise RuntimeError(
                "motion-window probe requires at least 30 control steps"
            )

        control_dt = 1.0 / 30.0
        recent = pose_history[-31:]
        recent_production = production_velocity_history[-30:]
        independent_linear: dict[str, list[float]] = {
            name: [] for name in _CUPS
        }
        independent_angular: dict[str, list[float]] = {
            name: [] for name in _CUPS
        }
        production_linear: dict[str, list[float]] = {
            name: [] for name in _CUPS
        }
        production_angular: dict[str, list[float]] = {
            name: [] for name in _CUPS
        }
        for sample_index, (previous, current) in enumerate(
            zip(recent, recent[1:])
        ):
            production = recent_production[sample_index]
            for name in _CUPS:
                independent_linear[name].append(
                    float(
                        np.linalg.norm(current[name][:3] - previous[name][:3])
                        / control_dt
                    )
                )
                independent_angular[name].append(
                    _quat_delta_angle(
                        previous[name][3:7], current[name][3:7]
                    )
                    / control_dt
                )
                production_linear[name].append(
                    float(np.linalg.norm(production[name][:3]))
                )
                production_angular[name].append(
                    float(np.linalg.norm(production[name][3:]))
                )
        max_linear = max(max(values) for values in production_linear.values())
        max_angular = max(
            max(values) for values in production_angular.values()
        )
        max_linear_error = max(
            abs(production_linear[name][index] - independent_linear[name][index])
            for name in _CUPS
            for index in range(30)
        )
        max_angular_error = max(
            abs(production_angular[name][index] - independent_angular[name][index])
            for name in _CUPS
            for index in range(30)
        )
        one_second_drift = {
            name: float(
                np.linalg.norm(recent[-1][name][:3] - recent[0][name][:3])
            )
            for name in _CUPS
        }
        raw_velocities = _velocities(env)
        metrics = env._ycb04_cups_metrics(update_events=False)
        final = recent[-1]
        nested = bool(
            semantics.pair_is_nested(
                final["ycb04_small_cup"],
                semantics.SMALL_CUP,
                final["ycb04_medium_cup"],
                semantics.MEDIUM_CUP,
            )
            and semantics.pair_is_nested(
                final["ycb04_medium_cup"],
                semantics.MEDIUM_CUP,
                final["ycb04_large_cup"],
                semantics.LARGE_CUP,
            )
        )
        seed_passed = bool(
            metrics["initial_nested_valid"]
            and nested
            and metrics["contact_available"]
            and metrics["small_medium_contact_seen"]
            and metrics["medium_large_contact_seen"]
            and metrics["stable"]
            and max_linear < 0.02
            and max_angular < 0.3
            and max_linear_error < 1.0e-8
            and max_angular_error < 1.0e-8
        )
        passed = passed and seed_passed
        print(
            f"[YCB04][motion_window][seed={seed}] pass={seed_passed} "
            f"max_observable_linear_mps={max_linear:.9f} "
            f"max_observable_angular_radps={max_angular:.9f} "
            f"max_independent_error=({max_linear_error:.3e},"
            f"{max_angular_error:.3e}) one_second_drift_m={one_second_drift} "
            f"raw_physx_velocities={{{', '.join(f'{name}: {raw_velocities[name].tolist()}' for name in _CUPS)}}} "
            f"initial_nested={metrics['initial_nested_valid']} "
            f"final_nested={nested} stable={metrics['stable']} "
            f"pair_contact_seen=({metrics['small_medium_contact_seen']},"
            f"{metrics['medium_large_contact_seen']})",
            flush=True,
        )
    return passed


def main() -> int:
    env = HandsOnABCIsaacLabEnv(
        _TASK_NAME,
        seed=0,
        control_dt=1.0 / 30.0,
        physics_dt=1.0 / 120.0,
        time_limit=120.0,
        render_mode="rgb_array",
        image_obs=False,
        randomize=args_cli.mode in {"settle", "motion_window"},
        headless=args_cli.headless,
        isaac_device=args_cli.device,
        existing_simulation_app=simulation_app,
        existing_app_launcher=app_launcher,
    )
    collider_passed = _print_collider_contract(env)
    probe = {
        "settle": _settle_probe,
        "path": _path_probe,
        "pose_sweep": _pose_sweep_probe,
        "material_sweep": _material_sweep_probe,
        "contact_sweep": _contact_sweep_probe,
        "yaw_sweep": _yaw_sweep_probe,
        "damping_sweep": _damping_sweep_probe,
        "motion_window": _motion_window_probe,
    }[args_cli.mode]
    passed = probe(env) and collider_passed
    print(f"YCB04_PHYSX_{args_cli.mode.upper()}={'PASS' if passed else 'FAIL'}", flush=True)
    result = 0 if passed else 1
    # Some Isaac builds can block indefinitely while tearing down an imported
    # MJCF scene.  Preserve the verdict first, then bound probe-only cleanup.
    close_guard = threading.Timer(5.0, lambda: os._exit(result))
    close_guard.daemon = True
    close_guard.start()
    try:
        env.close()
    finally:
        close_guard.cancel()
    return result


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        shutdown_guard = threading.Timer(5.0, lambda: os._exit(exit_code))
        shutdown_guard.daemon = True
        shutdown_guard.start()
        try:
            simulation_app.close(wait_for_replicator=False)
        finally:
            shutdown_guard.cancel()
    raise SystemExit(exit_code)
