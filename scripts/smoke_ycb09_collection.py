#!/usr/bin/env python3
"""Live Isaac Sim 5.1 collection and multi-pose physics smoke for YCB09.

The final-placement drops in this script are validation-only state writes. They
are deliberately kept out of the task runtime and out of recorded episodes.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import sys
import traceback

import imageio.v3 as iio
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "handson"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from handson.tasks.mappings import CONFIG_MAPPING
from handson.tasks.t29_place_wine_glass import semantics as ycb09


RESET_SEEDS = tuple(range(11, 19))
FINAL_DROP_OFFSETS_M = (0.0, 0.0125, 0.024)
PLATE_CENTRAL_FLAT_LOCAL_Z_M = -0.0087735
DROP_CLEARANCE_M = 0.004
MAX_SUPPORT_PENETRATION_M = 0.0015
MAX_RESET_DRIFT_M = 0.005
RECORDING_ORIGIN_MAX_LINEAR_SPEED_M_S = 0.001
RECORDING_ORIGIN_MAX_ANGULAR_SPEED_RAD_S = 0.002
FAST_EXIT_AFTER_PASS_ENV = "HANDSON_YCB09_SMOKE_FAST_EXIT_AFTER_PASS"


def _pose_matrix(pose: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(pose[3:7], dtype=np.float64)
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = pose[:3]
    return matrix


def _camera_report(frame: np.ndarray) -> dict[str, object]:
    frame = np.asarray(frame)
    return {
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "min": int(frame.min()),
        "max": int(frame.max()),
        "std": float(frame.std()),
        "nonconstant": bool(frame.min() != frame.max()),
    }


def _step_raw(raw: object, count: int) -> None:
    for _ in range(int(count)):
        raw._step_isaaclab()
    raw._refresh_object_poses()


def _force(view: object, dt: float) -> float:
    matrix = view.get_contact_force_matrix(float(dt))
    if hasattr(matrix, "detach"):
        matrix = matrix.detach().cpu().numpy()
    forces = np.asarray(matrix, dtype=np.float64)
    if forces.size == 0 or not np.all(np.isfinite(forces)):
        raise RuntimeError("contact force tensor is empty or non-finite")
    return float(np.linalg.norm(forces, axis=-1).sum())


def _make_glass_table_contact_view(raw: object) -> object:
    physics_view = raw._robots["right"]._physics_sim_view
    glass_prim = raw._find_rigid_body_prim(ycb09.GLASS_NAME)
    table_prim = raw._find_rigid_body_prim("table")
    if physics_view is None or glass_prim is None or table_prim is None:
        raise RuntimeError("glass/table contact truth could not be bound")
    view = physics_view.create_rigid_contact_view(
        str(glass_prim.GetPath()),
        filter_patterns=[str(table_prim.GetPath())],
        max_contact_data_count=64,
    )
    if view is None or int(view.sensor_count) < 1 or int(view.filter_count) < 1:
        raise RuntimeError("glass/table contact view is empty")
    return view


def _runtime_inertial_report(raw: object, name: str) -> dict[str, object]:
    from pxr import PhysxSchema, UsdPhysics

    prim = raw._find_rigid_body_prim(name)
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"runtime inertial body is unavailable: {name}")
    mass_api = UsdPhysics.MassAPI(prim)
    rigid_api = PhysxSchema.PhysxRigidBodyAPI(prim)
    center = mass_api.GetCenterOfMassAttr().Get()
    inertia = mass_api.GetDiagonalInertiaAttr().Get()
    axes = mass_api.GetPrincipalAxesAttr().Get()
    return {
        "mass_kg": float(mass_api.GetMassAttr().Get()),
        "center_of_mass_m": [float(value) for value in center],
        "diagonal_inertia_kg_m2": [float(value) for value in inertia],
        "principal_axes_wxyz": [
            float(axes.GetReal()),
            *[float(value) for value in axes.GetImaginary()],
        ],
        "linear_damping": float(rigid_api.GetLinearDampingAttr().Get()),
        "angular_damping": float(rigid_api.GetAngularDampingAttr().Get()),
        "sleep_threshold": float(rigid_api.GetSleepThresholdAttr().Get()),
        "stabilization_threshold": float(
            rigid_api.GetStabilizationThresholdAttr().Get()
        ),
        "max_depenetration_velocity_m_s": float(
            rigid_api.GetMaxDepenetrationVelocityAttr().Get()
        ),
    }


def _recording_origin_report(
    raw: object,
    seed: int,
    *,
    checkpoint_steps: tuple[int, ...] = (480,),
) -> dict[str, object]:
    checkpoint_steps = tuple(sorted(set(int(step) for step in checkpoint_steps)))
    if not checkpoint_steps or checkpoint_steps[0] <= 0:
        raise ValueError("recording-origin checkpoints must be positive")
    initial_poses = {
        name: np.asarray(raw._poses[name], dtype=np.float64).copy()
        for name in (ycb09.GLASS_NAME, ycb09.PLATE_NAME)
    }
    checkpoints: dict[str, object] = {}
    previous_step = 0
    for checkpoint_step in checkpoint_steps:
        _step_raw(raw, checkpoint_step - previous_step)
        previous_step = checkpoint_step
        bodies: dict[str, object] = {}
        for name in (ycb09.GLASS_NAME, ycb09.PLATE_NAME):
            velocity = raw._physics_body_velocity(name)
            if velocity is None:
                raise RuntimeError(
                    f"recording-origin velocity is unavailable: {name}"
                )
            velocity = np.asarray(velocity, dtype=np.float64)
            pose = np.asarray(raw._poses[name], dtype=np.float64)
            initial_pose = initial_poses[name]
            quaternion_dot = float(
                np.clip(abs(np.dot(initial_pose[3:7], pose[3:7])), 0.0, 1.0)
            )
            bodies[name] = {
                "velocity_linear_xyz_m_s": velocity[:3].tolist(),
                "velocity_angular_xyz_rad_s": velocity[3:].tolist(),
                "linear_speed_m_s": float(np.linalg.norm(velocity[:3])),
                "angular_speed_rad_s": float(np.linalg.norm(velocity[3:])),
                "translation_delta_m": float(
                    np.linalg.norm(pose[:3] - initial_pose[:3])
                ),
                "orientation_delta_deg": float(
                    np.degrees(2.0 * np.arccos(quaternion_dot))
                ),
                "inertial": _runtime_inertial_report(raw, name),
            }
        checkpoints[str(checkpoint_step)] = bodies

    final_step = checkpoint_steps[-1]
    final_bodies = checkpoints[str(final_step)]
    failures: list[str] = []
    for name, body in final_bodies.items():
        linear_speed = float(body["linear_speed_m_s"])
        angular_speed = float(body["angular_speed_rad_s"])
        if linear_speed >= RECORDING_ORIGIN_MAX_LINEAR_SPEED_M_S:
            failures.append(
                f"{name} recording-origin linear speed {linear_speed} exceeds "
                f"{RECORDING_ORIGIN_MAX_LINEAR_SPEED_M_S}"
            )
        if angular_speed >= RECORDING_ORIGIN_MAX_ANGULAR_SPEED_RAD_S:
            failures.append(
                f"{name} recording-origin angular speed {angular_speed} exceeds "
                f"{RECORDING_ORIGIN_MAX_ANGULAR_SPEED_RAD_S}"
            )
    return {
        "seed": int(seed),
        "physics_steps": final_step,
        "checkpoint_steps": list(checkpoint_steps),
        "max_linear_speed_m_s": RECORDING_ORIGIN_MAX_LINEAR_SPEED_M_S,
        "max_angular_speed_rad_s": RECORDING_ORIGIN_MAX_ANGULAR_SPEED_RAD_S,
        "bodies": final_bodies,
        "checkpoints": checkpoints,
        "passed": not failures,
        "failures": failures,
    }


def _settled_reset_report(raw: object, seed: int, glass_table_view: object) -> dict[str, object]:
    glass = np.asarray(raw._poses[ycb09.GLASS_NAME], dtype=np.float64)
    plate = np.asarray(raw._poses[ycb09.PLATE_NAME], dtype=np.float64)
    glass_reset = np.asarray(raw._ycb09_glass_reset_pose, dtype=np.float64)
    plate_reset = np.asarray(raw._ycb09_plate_reset_pose, dtype=np.float64)
    table_top = float(raw._current_table_top_z())
    glass_velocity = raw._physics_body_velocity(ycb09.GLASS_NAME)
    plate_velocity = raw._physics_body_velocity(ycb09.PLATE_NAME)
    if glass_velocity is None or plate_velocity is None:
        raise RuntimeError(f"seed {seed}: rigid-body velocity truth is unavailable")
    glass_velocity = np.asarray(glass_velocity, dtype=np.float64)
    plate_velocity = np.asarray(plate_velocity, dtype=np.float64)
    metrics = raw._ycb09_metrics(update_progress=False)
    glass_bottom = float(glass[2] + ycb09.GLASS_LOCAL_MIN_Z_M)
    plate_bottom = float(plate[2] + ycb09.PLATE_LOCAL_MIN_Z_M)
    center_distance = float(np.linalg.norm(glass[:2] - plate[:2]))
    surface_clearance = (
        center_distance
        - ycb09.GLASS_FOOTPRINT_RADIUS_M
        - ycb09.PLATE_FOOTPRINT_RADIUS_M
    )
    glass_drift = float(np.linalg.norm(glass[:3] - glass_reset[:3]))
    plate_drift = float(np.linalg.norm(plate[:3] - plate_reset[:3]))
    glass_table_force = _force(glass_table_view, raw._sim_dt)

    if not metrics["ycb09_finite_state"] or metrics["ycb09_invalid_reason"]:
        raise RuntimeError(f"seed {seed}: invalid reset metrics: {metrics}")
    if not metrics["ycb09_contact_truth_available"]:
        raise RuntimeError(f"seed {seed}: required contact truth is unavailable")
    if not metrics["ycb09_plate_table_supported"] or glass_table_force <= 0.0:
        raise RuntimeError(f"seed {seed}: table support contact is missing")
    if metrics["ycb09_plate_contact_force_n"] > ycb09.CONTACT_FORCE_THRESHOLD_N:
        raise RuntimeError(f"seed {seed}: glass starts in contact with the plate")
    if metrics["ycb09_hand_contact_force_n"] > ycb09.CONTACT_FORCE_THRESHOLD_N:
        raise RuntimeError(f"seed {seed}: glass starts in contact with the hand")
    if glass_bottom < table_top - MAX_SUPPORT_PENETRATION_M:
        raise RuntimeError(f"seed {seed}: glass penetrates the table")
    if plate_bottom < table_top - MAX_SUPPORT_PENETRATION_M:
        raise RuntimeError(f"seed {seed}: plate penetrates the table")
    if surface_clearance <= 0.0:
        raise RuntimeError(f"seed {seed}: reset objects overlap")
    if glass_drift > MAX_RESET_DRIFT_M or plate_drift > MAX_RESET_DRIFT_M:
        raise RuntimeError(
            f"seed {seed}: excessive settle drift glass={glass_drift} plate={plate_drift}"
        )
    for name, velocity in (("glass", glass_velocity), ("plate", plate_velocity)):
        if not np.all(np.isfinite(velocity)):
            raise RuntimeError(f"seed {seed}: non-finite {name} velocity")
        if np.linalg.norm(velocity[:3]) >= ycb09.MAX_LINEAR_SPEED_M_S:
            raise RuntimeError(f"seed {seed}: {name} linear velocity did not settle")
        if np.linalg.norm(velocity[3:]) >= ycb09.MAX_ANGULAR_SPEED_RAD_S:
            raise RuntimeError(f"seed {seed}: {name} angular velocity did not settle")
    if metrics["ycb09_tilt_deg"] > 1.0:
        raise RuntimeError(f"seed {seed}: glass reset tilt is too large")

    return {
        "seed": int(seed),
        "glass_pose": glass.tolist(),
        "plate_pose": plate.tolist(),
        "center_distance_m": center_distance,
        "swept_surface_clearance_m": surface_clearance,
        "glass_table_clearance_m": glass_bottom - table_top,
        "plate_table_clearance_m": plate_bottom - table_top,
        "glass_settle_drift_m": glass_drift,
        "plate_settle_drift_m": plate_drift,
        "glass_table_contact_force_n": glass_table_force,
        "plate_table_contact_force_n": float(metrics["ycb09_table_contact_force_n"]),
        "glass_linear_speed_m_s": float(np.linalg.norm(glass_velocity[:3])),
        "glass_angular_speed_rad_s": float(np.linalg.norm(glass_velocity[3:])),
        "plate_linear_speed_m_s": float(np.linalg.norm(plate_velocity[:3])),
        "plate_angular_speed_rad_s": float(np.linalg.norm(plate_velocity[3:])),
        "glass_tilt_deg": float(metrics["ycb09_tilt_deg"]),
    }


def _final_drop_report(raw: object, seed: int, offset_m: float) -> dict[str, object]:
    plate = np.asarray(raw._poses[ycb09.PLATE_NAME], dtype=np.float64)
    plate_matrix = _pose_matrix(plate)
    local_glass_center = np.asarray(
        [
            float(offset_m),
            0.0,
            PLATE_CENTRAL_FLAT_LOCAL_Z_M
            - ycb09.GLASS_LOCAL_MIN_Z_M
            + DROP_CLEARANCE_M,
            1.0,
        ],
        dtype=np.float64,
    )
    world_center = plate_matrix @ local_glass_center
    glass_pose = np.asarray(
        [*world_center[:3], *plate[3:7]],
        dtype=np.float64,
    )
    body_view = raw._task_body_views.get(ycb09.GLASS_NAME)
    if body_view is None or not raw._set_task_body_world_pose(body_view, glass_pose):
        raise RuntimeError(f"seed {seed}: could not author validation-only glass drop")
    _step_raw(raw, 240)
    metrics = raw._ycb09_metrics(update_progress=False)
    velocity = raw._physics_body_velocity(ycb09.GLASS_NAME)
    if velocity is None:
        raise RuntimeError(f"seed {seed}: glass velocity missing after final drop")
    velocity = np.asarray(velocity, dtype=np.float64)
    base_local_z = float(metrics["ycb09_glass_plate_local"][2])
    if not metrics["ycb09_centered"]:
        raise RuntimeError(f"seed {seed}: final drop is not centered: {metrics}")
    if not metrics["ycb09_plate_support"]:
        raise RuntimeError(f"seed {seed}: glass/plate support contact is missing")
    if not metrics["ycb09_plate_stationary"]:
        raise RuntimeError(f"seed {seed}: plate moved or lost table support")
    if not metrics["ycb09_upright"] or not metrics["ycb09_velocity_stable"]:
        raise RuntimeError(f"seed {seed}: final glass did not settle upright: {metrics}")
    if base_local_z < PLATE_CENTRAL_FLAT_LOCAL_Z_M - MAX_SUPPORT_PENETRATION_M:
        raise RuntimeError(f"seed {seed}: glass penetrates plate central support")
    return {
        "seed": int(seed),
        "authored_offset_m": float(offset_m),
        "settled_center_error_m": float(metrics["ycb09_center_error_m"]),
        "glass_plate_base_local_z_m": base_local_z,
        "plate_central_flat_local_z_m": PLATE_CENTRAL_FLAT_LOCAL_Z_M,
        "glass_plate_contact_force_n": float(metrics["ycb09_plate_contact_force_n"]),
        "plate_table_contact_force_n": float(metrics["ycb09_table_contact_force_n"]),
        "plate_displacement_m": float(metrics["ycb09_plate_displacement_m"]),
        "glass_linear_speed_m_s": float(np.linalg.norm(velocity[:3])),
        "glass_angular_speed_rad_s": float(np.linalg.norm(velocity[3:])),
        "glass_tilt_deg": float(metrics["ycb09_tilt_deg"]),
    }


def run(
    output_dir: Path,
    *,
    image_obs: bool,
    isaac_device: str,
    validate_drops: bool = True,
    origin_only: bool = False,
    origin_checkpoint_steps: tuple[int, ...] = (480,),
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = CONFIG_MAPPING[ycb09.TASK_NAME]()
    print(f"YCB09_SMOKE constructing image_obs={image_obs}", flush=True)
    env = config.get_native_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=True,
        sim_backend="isaacsim",
        arm_type="rokae",
        hand_type="wuji1",
        launch_isaac=True,
        image_obs=image_obs,
        headless=True,
        isaac_device=isaac_device,
        seed=RESET_SEEDS[0],
    )
    print("YCB09_SMOKE constructed", flush=True)
    raw = env.unwrapped
    try:
        if raw.task_name != ycb09.TASK_NAME:
            raise RuntimeError(f"wrong live task: {raw.task_name!r}")
        if not np.isclose(raw._control_dt, 1.0 / 30.0):
            raise RuntimeError(f"wrong control dt: {raw._control_dt}")
        if not np.isclose(raw._physics_dt, 1.0 / 120.0) or raw._n_substeps != 4:
            raise RuntimeError(
                f"wrong physics timing: dt={raw._physics_dt}, substeps={raw._n_substeps}"
            )
        if raw._time_limit != 60.0:
            raise RuntimeError(f"wrong task time limit: {raw._time_limit}")
        if not raw._ycb09_contact_pairs_available:
            raise RuntimeError(
                "YCB09 contact-pair ground truth was not created: "
                + "; ".join(raw._isaac_warnings)
            )
        if set(raw._ycb09_contact_views) != {
            "glass_hand",
            "glass_plate",
            "plate_table",
        }:
            raise RuntimeError(
                f"wrong YCB09 contact views: {sorted(raw._ycb09_contact_views)}"
            )
        glass_table_view = _make_glass_table_contact_view(raw)

        observation, reset_info = env.reset(seed=RESET_SEEDS[0])
        expected_recording_fields = {"ycb09_hidden_state"}
        if not expected_recording_fields.issubset(observation):
            raise RuntimeError(
                f"missing recording-only fields: {expected_recording_fields - set(observation)}"
            )
        if any(key in config.proprio_keys for key in expected_recording_fields):
            raise RuntimeError("hidden YCB09 fields leaked into policy proprioception")

        if origin_only:
            origin_report = _recording_origin_report(
                raw,
                RESET_SEEDS[0],
                checkpoint_steps=origin_checkpoint_steps,
            )
            report = {
                "status": "PASS" if origin_report["passed"] else "FAIL",
                "task_id": ycb09.TASK_ID,
                "task_name": ycb09.TASK_NAME,
                "runtime_class": type(raw).__name__,
                "recording_origin": origin_report,
                "synthetic_success": False,
                "state_writes": False,
            }
            (output_dir / "report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            passed = bool(origin_report["passed"])
            print(
                f"YCB09_ORIGIN_SMOKE_{'PASS' if passed else 'FAIL'} "
                + json.dumps(report, sort_keys=True),
                flush=True,
            )
            fast_exit = os.environ.get(
                FAST_EXIT_AFTER_PASS_ENV, ""
            ).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if fast_exit:
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0 if passed else 1)
            if not passed:
                raise RuntimeError("; ".join(origin_report["failures"]))
            return report

        camera_reports: dict[str, object] = {}
        if image_obs:
            for key in ("random_camera", "wrist_left", "wrist_right"):
                frame = np.asarray(observation[key])
                camera_reports[key] = _camera_report(frame)
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                    raise RuntimeError(f"invalid {key} camera frame: {frame.shape} {frame.dtype}")
                if frame.min() == frame.max():
                    raise RuntimeError(f"constant {key} camera frame")
                iio.imwrite(output_dir / f"{key}.png", frame)

        # Exercise the exact policy collection action path while holding the
        # current commanded TCP and open-hand targets.
        hold_action = np.concatenate(
            [raw._right_tcp_pose, raw._right_hand], axis=0
        ).astype(env.action_space.dtype)
        for _ in range(3):
            _, _, terminated, truncated, step_info = env.step(hold_action)
            if terminated or truncated:
                raise RuntimeError(
                    "baseline policy step ended episode: "
                    + str(step_info.get("ycb09_terminal_reason", ""))
                )

        reset_reports: list[dict[str, object]] = []
        final_drop_reports: list[dict[str, object]] = []
        sampled_glass_xy: list[list[float]] = []
        sampled_plate_xy: list[list[float]] = []
        for index, seed in enumerate(RESET_SEEDS):
            env.reset(seed=seed)
            _step_raw(raw, 120)
            reset_report = _settled_reset_report(raw, seed, glass_table_view)
            reset_reports.append(reset_report)
            sampled_glass_xy.append(reset_report["glass_pose"][:2])
            sampled_plate_xy.append(reset_report["plate_pose"][:2])
            if validate_drops and index < len(FINAL_DROP_OFFSETS_M):
                final_drop_reports.append(
                    _final_drop_report(raw, seed, FINAL_DROP_OFFSETS_M[index])
                )

        for name, values in (
            ("glass", np.asarray(sampled_glass_xy, dtype=np.float64)),
            ("plate", np.asarray(sampled_plate_xy, dtype=np.float64)),
        ):
            if np.ptp(values[:, 0]) <= 0.005 or np.ptp(values[:, 1]) <= 0.005:
                raise RuntimeError(f"{name} live reset randomization span is too small")

        report = {
            "status": "PASS",
            "task_id": ycb09.TASK_ID,
            "task_name": ycb09.TASK_NAME,
            "runtime_class": type(raw).__name__,
            "control_hz": 1.0 / raw._control_dt,
            "physics_hz": 1.0 / raw._physics_dt,
            "physics_substeps_per_control": raw._n_substeps,
            "time_limit_seconds": raw._time_limit,
            "contact_view_count": len(raw._ycb09_contact_views),
            "camera_reports": camera_reports,
            "reset_info_terminal_reason": reset_info.get("ycb09_terminal_reason", ""),
            "reset_reports": reset_reports,
            "final_drop_reports": final_drop_reports,
            "isaac_warnings": list(raw._isaac_warnings),
        }
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print("YCB09_SMOKE_PASS " + json.dumps(report, sort_keys=True), flush=True)
        if os.environ.get(FAST_EXIT_AFTER_PASS_ENV, "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            # This is a dedicated acceptance process.  Some headless
            # sim-bench driver stacks can block indefinitely in Kit's
            # SimulationContext.stop() callback after all evidence has been
            # durably written.  Match the repository's post-render worker:
            # flush the PASS receipt, then let process teardown release every
            # GPU/Kit resource instead of entering that callback.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        return report
    except BaseException as exc:
        print(f"YCB09_SMOKE_FAILED {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        env.close()


def main() -> None:
    faulthandler.enable()
    faulthandler.dump_traceback_later(90.0, repeat=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--skip-drops", action="store_true")
    parser.add_argument("--origin-only", action="store_true")
    parser.add_argument(
        "--origin-checkpoint-steps",
        type=int,
        nargs="+",
        default=(480,),
        help="Physics-step checkpoints recorded by --origin-only.",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(
        args.output_dir,
        image_obs=not args.no_cameras,
        isaac_device=args.device,
        validate_drops=not args.skip_drops,
        origin_only=args.origin_only,
        origin_checkpoint_steps=tuple(args.origin_checkpoint_steps),
    )


if __name__ == "__main__":
    main()
