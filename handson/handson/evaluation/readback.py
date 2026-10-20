"""Passive native readback from the paper evaluator."""
from collections.abc import Mapping
from .contracts import require


def physics_tick(env):
    """Read the simulation's real step counter across Isaac Lab versions."""
    sim = env._sim
    if hasattr(sim, "get_physics_step_count"):
        return int(sim.get_physics_step_count())
    return int(sim.current_time_step_index)

def plain(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy().tolist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, (str, float, int, bool)) or value is None:
        return value
    return repr(value)

def articulation_readback(env):
    """Read the full native articulation list, not only logical object aliases.

    Joint names come from the live PhysX metatype. Never invent positional names
    or use the saved reset states as the actual measurement.
    """
    result = {}
    for path, view, _indices, *_saved_reset in env._task_reset_articulation_states:
        require(path not in result, f'Duplicate native articulation path: {path}')
        positions = plain(view.get_dof_positions())
        velocities = plain(view.get_dof_velocities())
        count = len(positions[0])
        indices = getattr(view.get_metatype(0), 'dof_indices', None)
        require(isinstance(indices, Mapping), f'Missing articulation joint names: {path}')
        pairs = sorted((int(index), str(name)) for name, index in indices.items())
        require([index for index, _name in pairs] == list(range(count)),
                f'Incomplete articulation joint ordering: {path}')
        require(len(positions) == len(velocities) and
                all(len(row) == count for row in positions + velocities),
                f'Articulation joint shape mismatch: {path}')
        result[path] = dict(joint_names=[name for _index, name in pairs],
            joint_pos=positions, joint_vel=velocities,
            root_pose_xyzw=plain(view.get_root_transforms()),
            root_velocity=plain(view.get_root_velocities()),
            logical_aliases=sorted(name for name, pair in env._task_articulation_views.items()
                                   if pair[0] is view),
            readback='live_physx_tensor', recorded_state_used=False)
    return result

def capture(env):
    effective_physx = env._sim.cfg.physx
    return plain(dict(
        step=env.env_step, physics_tick=env._sim.current_time_step_index,
        control_dt=env._control_dt, physics_dt=env._physics_dt, substeps=env._n_substeps,
        target=env._target_object_name, table_height_delta=env.delta_h,
        robots={side: dict(joint_names=robot.joint_names, body_names=robot.body_names,
            joint_pos=robot.data.joint_pos, joint_vel=robot.data.joint_vel,
            body_pose_w=robot.data.body_pose_w, body_vel_w=robot.data.body_vel_w)
                for side, robot in env._robots.items()},
        objects={name: dict(pose_xyzw=view.get_transforms(), velocity=view.get_velocities())
                 for name, view in env._task_body_views.items()},
        articulations=articulation_readback(env),
        arm_ik_backend=env._arm_ik_backend,
        haorui_enabled=env._haorui_ik_enabled,
        control_modes={key: getattr(env, key, None) for key in (
            '_direct_hand_control', '_simplified_post_ik', '_deferred_arm_render',
            '_hand_target_control', '_hands_only', '_guidance_mode')},
        effective_physx={k: getattr(effective_physx, k) for k in (
            'solver_type', 'enable_enhanced_determinism', 'max_position_iteration_count',
            'max_velocity_iteration_count', 'bounce_threshold_velocity',
            'friction_offset_threshold', 'friction_correlation_distance',
            'gpu_max_rigid_contact_count', 'gpu_max_rigid_patch_count')},
        controller={key: value for key, value in vars(env).items()
                    if key.startswith(('_arm_target_', '_arm_max_', '_arm_force_'))},
        task_info='not_called_by_passive_capture; use native step return', warnings=env._isaac_warnings))

def verify_native_control(env, recorded):
    """Check live mode/joint ordering against recorded facts, not assumed defaults."""
    attributes = {'live_direct_hand_control': '_direct_hand_control',
                  'isaac_arm_render_deferred': '_deferred_arm_render',
                  'arm_ik_deferred': '_simplified_post_ik'}
    modes = {}
    for field, attr in attributes.items():
        actual = getattr(env, attr, None)
        expected = recorded['recording'].get(field)
        if expected is not None:
            require(actual is not None, f'Missing native mode readback: {field}')
            require(bool(actual) == expected, f'Native control mode mismatch: {field}')
        modes[field] = dict(recorded=expected, attribute=attr,
                           actual=None if actual is None else bool(actual),
                           compared=expected is not None)
    layout = recorded['robot_layout']
    require(layout and set(env._robots) == set(layout), 'Robot side layout differs')
    actual_names = {}
    for side, robot in env._robots.items():
        names = list(robot.joint_names)
        require(names == layout[side]['names'], f'Native joint ordering differs: {side}')
        actual_names[side] = names
    return dict(modes=modes, robot_joint_names=actual_names,
                missing_recording=[k for k, v in modes.items() if not v['compared']],
                scope='declared mode fields and ordered joints; not complete control equivalence')
