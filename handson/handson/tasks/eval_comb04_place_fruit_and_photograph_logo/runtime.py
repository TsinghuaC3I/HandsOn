"""XY3-only hooks called by the existing ABC collection lifecycle."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from . import semantics as xy3


def attach_logo(root: ET.Element) -> None:
    from ...sim.isaaclab.abc_collection_env import _matrix_to_quat_wxyz
    bucket = root.find(".//body[@name='bucket']")
    if bucket is None:
        raise ValueError("XY3 requires A01's original bucket")
    if bucket.find("geom[@name='photograph_logo_disk']") is None:
        ET.SubElement(bucket, "geom", {
            "name": "photograph_logo_disk", "type": "cylinder",
            "pos": " ".join(map(str, xy3.LOGO_LOCAL)),
            "quat": " ".join(map(str, _matrix_to_quat_wxyz(xy3.LOGO_ROTATION_LOCAL))),
            "size": f"{xy3.LOGO_RADIUS} 0.0001", "material": "casia_logo_mat",
            "contype": "0", "conaffinity": "0", "mass": "0", "group": "1",
        })


def author_initial_layout(root: ET.Element) -> None:
    support_gap = 0.004
    for body in root.iter("body"):
        name = body.get("name")
        name = "camera" if name == "link_19" else name
        if name not in xy3.DEFAULT_XY:
            continue
        x, y = xy3.DEFAULT_XY[name]
        body.set(
            "pos",
            f"{x:.6f} {y:.6f} {0.89 - xy3.SUPPORT_MIN_Z[name] + support_gap:.6f}",
        )


def layout_is_clear(env, xy: dict, yaw: dict, hand_points=()) -> bool:
    """Joint object envelopes plus both robot-body clouds; live shape audit is separate."""
    # Reuse the atomic dimensions and distance implementations without
    # changing their task-specific samplers or physical thresholds.
    from ...sim.isaaclab.abc_collection_env import (
        _A01_OBJECT_FOOTPRINT_RADII, _A01_BUCKET_FOOTPRINT_HALF_EXTENTS,
    )
    half = _A01_BUCKET_FOOTPRINT_HALF_EXTENTS
    bucket_poly = xy["bucket"] + np.asarray(
        [(-half[0], -half[1]), (half[0], -half[1]),
         (half[0], half[1]), (-half[0], half[1])]
    )
    camera_poly = env._c06_camera_footprint(xy["camera"], yaw["camera"])
    if env._c01_convex_polygon_axis_margin(bucket_poly, camera_poly) < 0.010:
        return False
    for i, name in enumerate(xy3.FRUIT_NAMES):
        radius = _A01_OBJECT_FOOTPRINT_RADII[name]
        for polygon in (bucket_poly, camera_poly):
            if env._c01_point_polygon_distance(xy[name], polygon) < radius + 0.015:
                return False
        for other in xy3.FRUIT_NAMES[i + 1:]:
            if np.linalg.norm(xy[name] - xy[other]) < radius + _A01_OBJECT_FOOTPRINT_RADII[other] + 0.015:
                return False
    # This screening is deliberately conservative; exact all-shape overlap
    # before/after PhysX is still a required acceptance gate.
    for point in hand_points:
        point = np.asarray(point)
        for polygon, top_z in ((bucket_poly, 1.070), (camera_poly, 1.065)):
            if point[2] <= top_z + 0.060 and env._c01_point_polygon_distance(point[:2], polygon) < 0.045:
                return False
        for name in xy3.FRUIT_NAMES:
            if point[2] <= 1.030 and np.linalg.norm(point[:2] - xy[name]) < _A01_OBJECT_FOOTPRINT_RADII[name] + 0.045:
                return False
    return True


def apply_reset_layout(env) -> None:
    from ...sim.isaaclab.abc_collection_env import _randomized_physx_pose

    hands = (*env._reset_robot_body_points("right"), *env._reset_robot_body_points("left"))
    xy, yaw = xy3.sample_layout(
        env._random, randomize=env.randomize,
        is_clear=lambda positions, angles: layout_is_clear(env, positions, angles, hands),
    )
    rigid = {path: (view, indices, pose, velocity)
             for path, view, indices, pose, velocity in env._task_reset_body_states}
    articulated = {path: (view, indices, pose, velocity)
                   for path, view, indices, pose, velocity, _, _ in env._task_reset_articulation_states}
    support_gap = 0.004
    for name in xy3.DYNAMIC_NAMES:
        prim = env._find_rigid_body_prim(name)
        if prim is None:
            raise RuntimeError(f"XY3 reset entity missing: {name}")
        path = str(prim.GetPath())
        entry = env._reset_transform_state_for_path(path, rigid, articulated)
        if entry is None:
            raise RuntimeError(f"XY3 entity not registered for physical reset: {name} ({path})")
        kind, view, indices, pose, velocity = entry
        result = _randomized_physx_pose(
            pose.detach().cpu().numpy(), xy_offset=np.zeros(2),
            z_offset=0.0, yaw_degrees=yaw[name],
        )
        result[:, :2] = xy[name]
        result[:, 2] = 0.89 - xy3.SUPPORT_MIN_Z[name] + support_gap
        tensor = env._torch.as_tensor(result, dtype=pose.dtype, device=pose.device)
        zeros = env._torch.zeros_like(velocity)
        if kind == "articulation":
            view.set_root_transforms(tensor, indices)
            view.set_root_velocities(zeros, indices)
            env._record_authored_task_reset_write(path, transform_xyzw=tensor, velocity=zeros)
        else:
            env._write_rigid_reset_state(path, view, indices, tensor, zeros)
    env._xy3_reset_xy = {name: value.copy() for name, value in xy.items()}
    env._xy3_reset_yaw = dict(yaw)


def logo_pose(env) -> np.ndarray:
    from ...sim.isaaclab.abc_collection_env import _pose_vector_to_matrix, _matrix_to_quat_wxyz

    bucket = env._pose_for_role("receptacle")
    transform = _pose_vector_to_matrix(bucket)
    return np.concatenate(((transform @ np.r_[xy3.LOGO_LOCAL, 1.0])[:3],
                           _matrix_to_quat_wxyz(transform[:3, :3] @ xy3.LOGO_ROTATION_LOCAL)))


def photograph_metrics(env, *, advance_contact_grace: bool = True) -> tuple[dict, bool, bool]:
    from ...sim.isaaclab.abc_collection_env import _pose_vector_to_matrix

    camera = env._pose_for_role("camera")
    logo = logo_pose(env)
    # Updating C06 history in both stages prevents a pre-held shutter from
    # manufacturing a new press at the A->B boundary.
    metrics = env._c06_photograph_metrics(
        camera_pose=camera, logo_position=logo[:3],
        advance_contact_grace=advance_contact_grace,
    )
    normal = _pose_vector_to_matrix(env._pose_for_role("receptacle"))[:3, :3] @ xy3.LOGO_NORMAL_LOCAL
    front_visible = bool(np.dot(camera[:3] - logo[:3], normal) > 0.0)
    # The label lies on the outer face of the convex front wall, so an origin
    # in its front half-space cannot view it through that wall. Keep any
    # additional scene occlusion checks in the physical validation evidence.
    photo_valid = bool(metrics["c06_camera_aimed"] and metrics["c06_right_hand_contact"] and front_visible)
    return metrics, front_visible, photo_valid


def sample_photograph_substep(env) -> None:
    """Capture the real press and its validity at the same physical substep."""
    env._refresh_object_poses()
    # The inherited grace budget is expressed in control frames. Substep
    # sampling may refresh contact but must not spend that budget four times
    # per frame. compute_success spends it once at the control boundary.
    metrics, _, photo_valid = photograph_metrics(env, advance_contact_grace=False)
    if metrics["c06_shutter_press_edge"] and env._xy3_sequence.a_completed:
        env._xy3_sequence.observe_photo(
            a_valid=env._a01_fruit_bucket_success_now(),
            photo_valid=photo_valid, press_edge=True,
        )


def compute_success(env) -> bool:
    from ...sim.isaaclab.abc_collection_env import _A01_RELEASE_STABLE_SECONDS

    a_valid = env._a01_fruit_bucket_success_now()
    metrics, front_visible, photo_valid = photograph_metrics(env)
    required_steps = max(3, int(math.ceil(_A01_RELEASE_STABLE_SECONDS / env._control_dt)))
    success = env._xy3_sequence.update(
        a_valid=a_valid, required_steps=required_steps, photo_valid=photo_valid,
        press_edge=metrics["c06_shutter_press_edge"],
    )
    env._last_c06_success_metrics = metrics
    env._last_xy3_metrics = {
        **env._last_a01_success_metrics, **metrics,
        "xy3_logo_front_visible": front_visible,
        "xy3_a_stable_steps": env._xy3_sequence.a_stable_steps,
        "xy3_a_steps_required": required_steps,
        "xy3_a_completed": env._xy3_sequence.a_completed,
        "xy3_a_result_valid": a_valid,
        "xy3_photo_completed": env._xy3_sequence.photo_completed,
        "xy3_stage": "complete" if success else ("photograph" if env._xy3_sequence.a_completed else "fruit_to_bucket"),
        "xy3_success_now": success,
    }
    if success:
        env._xy3_success_latched = True
        env._xy3_terminal_a01_metrics = dict(env._last_a01_success_metrics)
        env._xy3_terminal_c06_metrics = dict(env._last_c06_success_metrics)
        env._xy3_terminal_metrics = dict(env._last_xy3_metrics)
    return success


def restore_hidden_state(env, state_parts) -> None:
    value = state_parts.get("xy3_hidden_state")
    if value is None:
        return
    state = np.asarray(value, dtype=np.float64)
    if state.shape != (len(xy3.HIDDEN_FIELDS),) or not np.all(np.isfinite(state)):
        raise ValueError("XY3 hidden state must be a finite five-value vector")
    env._xy3_sequence = xy3.SequenceState(int(state[0]), bool(state[1]), bool(state[2]))
    env._c06_shutter_was_down = bool(state[3])
    env._c06_right_contact_grace = int(state[4])
    env._last_xy3_metrics = {}
