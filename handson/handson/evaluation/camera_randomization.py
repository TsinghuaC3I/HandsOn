"""Episode camera draws from the recorded renderer."""
import hashlib
import json
import math
import random
PROFILE = "v14_camera_mount_jitter_v1"
TRANSLATION_M = 0.002
ROTATION_DEG = 0.3
CAMERAS = ("main", "wrist_left", "wrist_right")

def sample(task, storage_id, episode_seed):
    """Independent per-camera RNGs, unaffected by physics RNG or retry order."""
    result = {'profile': PROFILE, 'enabled': True, 'task': task,
              'storage_id': storage_id, 'episode_reset_seed': int(episode_seed),
              'translation_bounds_m_per_axis': [-TRANSLATION_M, TRANSLATION_M],
              'rotation_bounds_deg_per_axis': [-ROTATION_DEG, ROTATION_DEG],
              'composition': 'T_world_camera_nominal @ T_camera_delta',
              'coordinate_convention': 'OpenGL camera local: +X right, +Y up, -Z forward',
              'rotation_order': 'Rz(yaw) @ Ry(pitch) @ Rx(roll)',
              'sampling': 'once per trajectory, independently for each camera',
              'intrinsics_changed': False, 'cameras': {}}
    for name in CAMERAS:
        key = json.dumps([PROFILE, task, storage_id, int(episode_seed), name], separators=(',', ':'))
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')
        rng = random.Random(seed)
        translation = [rng.uniform(-TRANSLATION_M, TRANSLATION_M) for _ in range(3)]
        angles = [rng.uniform(-ROTATION_DEG, ROTATION_DEG) for _ in range(3)]
        x, y, z = [math.radians(a) / 2 for a in angles]
        cx, cy, cz = math.cos(x), math.cos(y), math.cos(z)
        sx, sy, sz = math.sin(x), math.sin(y), math.sin(z)
        quaternion = [cx*cy*cz + sx*sy*sz, sx*cy*cz - cx*sy*sz,
                      cx*sy*cz + sx*cy*sz, cx*cy*sz - sx*sy*cz]
        result['cameras'][name] = {'seed': seed, 'translation_m': translation,
                                  'roll_pitch_yaw_deg': angles,
                                  'quaternion_wxyz': quaternion}
    return result
