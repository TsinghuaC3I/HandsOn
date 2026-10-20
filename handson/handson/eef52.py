"""Independent measured-world-palm EEF52 boundary; no evaluator defaults.

Policy layout is right26,left26, each xyz/rotvec/hand20. Native commands
are named right/left arrays xyz/quaternion-wxyz/hand20, not flat54.
"""
import numpy as np
from scipy.spatial.transform import Rotation

SIDES = ('right', 'left')


def finite(value, shape, label):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'Invalid {label}: expected {shape}, got {value.shape}')
    return value


def hand_indices(names, side):
    if side not in SIDES or len(names) != len(set(names)):
        raise ValueError('Invalid side or duplicate joint names')
    prefix = 'r' if side == 'right' else 'l'
    aliases = [f'{prefix}_thumb_{part}' for part in ('cmc_flex', 'cmc_abd', 'mcp', 'ip')]
    for digit in ('index', 'middle', 'ring', 'pinky'):
        aliases.extend(f'{prefix}_{digit}_finger_{part}'
                       for part in ('mcp_flex', 'mcp_abd', 'pip', 'dip'))
    generic = [f'{side}_finger{f}_joint{j}' for f in range(1, 6) for j in range(1, 5)]
    indices = []
    for canonical, anatomical in zip(generic, aliases):
        matches = [names.index(n) for n in (canonical, anatomical) if n in names]
        if len(matches) != 1:
            raise ValueError(f'Missing/ambiguous control joint: {canonical}')
        indices.append(matches[0])
    return indices


def pose6_from_wxyz(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape[-1] != 7 or not np.isfinite(pose).all():
        raise ValueError('Invalid measured pose')
    quat = pose[..., [4, 5, 6, 3]]
    if np.any(np.linalg.norm(quat, axis=-1) < 1e-8):
        raise ValueError('Zero quaternion')
    # Match the actual training converter's matrix -> principal-rotvec path.
    matrix = Rotation.from_quat(quat.reshape(-1, 4)).as_matrix()
    rv = Rotation.from_matrix(matrix).as_rotvec().reshape(pose.shape[:-1] + (3,))
    return np.concatenate((pose[..., :3], rv), axis=-1)


def encode_measured(tcp_wxyz, joints, joint_names):
    tcp = np.asarray(tcp_wxyz, dtype=np.float64)
    q = np.asarray(joints, dtype=np.float64)
    if tcp.shape[-2:] != (2, 7) or q.shape[:-1] != tcp.shape[:-1]:
        raise ValueError('Measured side/pose/joint axes differ')
    hands = np.stack([q[..., i, hand_indices(list(joint_names[s]), s)]
                      for i, s in enumerate(SIDES)], axis=-2)
    state = np.concatenate((pose6_from_wxyz(tcp), hands), axis=-1)
    state = state.reshape(tcp.shape[:-2] + (52,)).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError('Nonfinite measured state')
    return state


def decode_command(action):
    """No delta, clipping, retiming, IK, side masking or frame change here."""
    action = finite(action, (52,), 'EEF52 action').reshape(2, 26)
    result = {}
    for i, side in enumerate(SIDES):
        quat = Rotation.from_rotvec(action[i, 3:6]).as_quat()[[3, 0, 1, 2]]
        result[side] = np.concatenate((action[i, :3], quat, action[i, 6:]))
    return result


def encode_commands(command_pose, command_hands, measured, active_sides):
    """Recorded command at t supervises action[t], not measured state[t+1]."""
    state = np.asarray(measured, dtype=np.float32)
    pose = finite(command_pose, state.shape[:-1] + (2, 7), 'recorded commands')
    hand = finite(command_hands, state.shape[:-1] + (2, 20), 'recorded hand commands')
    action = state.reshape(state.shape[:-1] + (2, 26)).copy()
    if not active_sides or not set(active_sides).issubset(SIDES):
        raise ValueError('Invalid active sides')
    for i, side in enumerate(SIDES):
        if side in active_sides:
            action[..., i, :6] = pose6_from_wxyz(pose[..., i, :])
            action[..., i, 6:] = hand[..., i, :]
    return action.reshape(state.shape)


def pose_matrix6(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-1] != 6 or not np.isfinite(value).all():
        raise ValueError('Invalid pose6')
    result = np.broadcast_to(np.eye(4), value.shape[:-1]+(4, 4)).copy()
    result[..., :3, 3] = value[..., :3]
    result[..., :3, :3] = Rotation.from_rotvec(value[..., 3:].reshape(-1, 3)).as_matrix().reshape(
        value.shape[:-1]+(3, 3))
    return result


def checked_transform(value):
    value = finite(value, (4, 4), 'recorded frame transform')
    if (not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8, rtol=0) or
        not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=2e-6, rtol=0) or
        abs(np.linalg.det(value[:3, :3])-1) >= 2e-6):
        raise ValueError('Not a rigid recorded frame transform')
    return value


def knots_to_world(knots, mapping):
    """Invert the *recorded* first@world_palm@last mapping; no real defaults.

    This converts saved labels only. It does not make DLS equivalent to the
    source's IK, ServoJ, retiming and hand-smoothing execution path.
    """
    knots = np.asarray(knots, dtype=np.float64)
    if knots.shape[-1] != 52 or not np.isfinite(knots).all():
        raise ValueError('Invalid policy knots')
    out = knots.copy()
    for i, side in enumerate(SIDES):
        first, last = [checked_transform(t) for t in mapping[side]]
        offset = i*26
        matrix = (np.linalg.inv(first) @ pose_matrix6(knots[..., offset:offset+6]) @
                  np.linalg.inv(last))
        out[..., offset:offset+3] = matrix[..., :3, 3]
        out[..., offset+3:offset+6] = Rotation.from_matrix(
            matrix[..., :3, :3].reshape(-1, 3, 3)).as_rotvec().reshape(knots.shape[:-1]+(3,))
    return out


def world_to_knots(actions, mapping):
    """Encode world-palm labels for the explicit recorded two-sided IK frame.

    No current camera pose, guessed real calibration, clipping, delta or hold
    substitution is allowed here. Execution is a separate controller layer.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape[-1] != 52 or not np.isfinite(actions).all():
        raise ValueError('Invalid world EEF52 actions')
    out = actions.copy()
    for i, side in enumerate(SIDES):
        first, last = [checked_transform(t) for t in mapping[side]]
        offset = i*26
        matrix = first @ pose_matrix6(actions[..., offset:offset+6]) @ last
        out[..., offset:offset+3] = matrix[..., :3, 3]
        out[..., offset+3:offset+6] = Rotation.from_matrix(
            matrix[..., :3, :3].reshape(-1, 3, 3)).as_rotvec().reshape(actions.shape[:-1]+(3,))
    return out


def host(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def measured_palm_pose(env, side):
    """Return a measured world pose in the public WXYZ convention.

    Lab 3 articulation buffers use XYZW; native command poses remain WXYZ.
    Keep this conversion at the read boundary, shared with wrist cameras.
    """
    robot = env._robots[side]
    body_name = f'{side}_palm_link'
    bodies = list(robot.body_names)
    if bodies.count(body_name) != 1:
        raise ValueError(f'Missing/ambiguous measured palm: {body_name}')
    index = bodies.index(body_name)
    pose = finite(host(robot.data.body_state_w)[0, index, :7], (7,), body_name).copy()
    if getattr(env, '_isaaclab_uses_xyzw', False):
        pose[3:] = pose[[6, 3, 4, 5]]
    return pose, index


def live_state(env):
    """Read the same palm body_state_w and joint_pos used by the recorder.

    Never fall back to obs['state']['tcp_pose'], which can be last commands.
    """
    poses, joints, names, evidence = [], [], {}, {}
    for side in SIDES:
        robot = env._robots[side]
        body_name = f'{side}_palm_link'
        pose, index = measured_palm_pose(env, side)
        poses.append(pose)
        names[side] = list(robot.joint_names)
        joints.append(finite(host(robot.data.joint_pos)[0], (len(names[side]),), side))
        ids = hand_indices(names[side], side)
        if ids != list(env._hand_joint_ids[side]):
            raise ValueError(f'Native hand target order differs from training: {side}')
        evidence[side] = dict(body=body_name, body_index=index, joint_names=names[side],
                              runtime_quaternion_order='xyzw' if getattr(env, '_isaaclab_uses_xyzw', False) else 'wxyz',
                              hand_control_indices=ids, palm_wxyz=poses[-1].tolist(),
                              joint_position=joints[-1].tolist())
    return encode_measured(np.stack(poses), np.stack(joints), names), evidence


def command_readback(env, requested):
    """Keep high-level command and actual bounded actuator targets distinct."""
    result = {}
    for side in SIDES:
        pose = finite(getattr(env, f'_{side}_tcp_pose'), (7,), side+' command pose')
        hand = finite(getattr(env, f'_{side}_hand'), (20,), side+' command hand')
        want = requested[side]
        angle = (Rotation.from_quat(pose[[4, 5, 6, 3]]).inv() *
                 Rotation.from_quat(want[[4, 5, 6, 3]])).magnitude()
        result[side] = dict(position_max_abs=float(np.max(np.abs(pose[:3]-want[:3]))),
            rotation_geodesic_rad=float(angle),
            hand_max_abs=float(np.max(np.abs(hand-want[7:]))),
            high_level_command=np.concatenate((pose, hand)).tolist(),
            actuator_joint_target=host(env._joint_targets[side])[0].tolist())
    return result


class EEF52Control:
    """One measured-state/action adapter for public environments and rollout."""
    def __init__(self, native):
        self.native = native
        self.last_command = None

    def state(self):
        return live_state(self.native)

    def step(self, action):
        self.last_command = decode_command(action)
        return self.native.step(self.last_command)
