import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from handson.evaluation.eef52 import (SIDES, command_readback, decode_command, encode_commands,
                         encode_measured, hand_indices, live_state, pose6_from_wxyz,
                         knots_to_world, pose_matrix6, world_to_knots)


def names(side):
    return ([f'joint{j}_{side}' for j in range(1, 8)] +
            [f'{side}_finger{f}_joint{j}' for j in range(1, 5) for f in range(1, 6)])


class CodecTest(unittest.TestCase):
    def test_two_sided_recorded_frame_mapping_no_default(self):
        rng = np.random.default_rng(31)
        world = rng.normal(size=(7, 52))
        mapped = world.copy()
        mapping = {}
        for i, side in enumerate(SIDES):
            first = pose_matrix6(rng.normal(size=6))
            last = pose_matrix6(rng.normal(size=6))
            mapping[side] = [first, last]
            t = first @ pose_matrix6(world[:, i*26:i*26+6]) @ last
            mapped[:, i*26:i*26+3] = t[:, :3, 3]
            mapped[:, i*26+3:i*26+6] = Rotation.from_matrix(t[:, :3, :3]).as_rotvec()
        decoded = knots_to_world(mapped, mapping)
        encoded = world_to_knots(world, mapping)
        for i in range(2):
            np.testing.assert_allclose(pose_matrix6(encoded[:, i*26:i*26+6]),
                                       pose_matrix6(mapped[:, i*26:i*26+6]), atol=2e-14)
            np.testing.assert_allclose(pose_matrix6(decoded[:, i*26:i*26+6]),
                                       pose_matrix6(world[:, i*26:i*26+6]), atol=2e-14)
            np.testing.assert_array_equal(decoded[:, i*26+6:(i+1)*26],
                                          world[:, i*26+6:(i+1)*26])
            np.testing.assert_array_equal(encoded[:, i*26+6:(i+1)*26],
                                          world[:, i*26+6:(i+1)*26])
        mapping['right'][0][3, 0] = 1
        with self.assertRaisesRegex(ValueError, 'rigid'):
            knots_to_world(mapped, mapping)
        with self.assertRaisesRegex(ValueError, 'rigid'):
            world_to_knots(world, mapping)

    def test_permutation_not_contiguous_articulation_slice(self):
        index = hand_indices(names('right'), 'right')
        self.assertEqual(index[:8], [7, 12, 17, 22, 8, 13, 18, 23])
        pose = np.tile([.1, .2, .3, 1, 0, 0, 0], (3, 2, 1))
        q = np.arange(3*2*27).reshape(3, 2, 27)
        encoded = encode_measured(pose, q, {s: names(s) for s in SIDES})
        np.testing.assert_array_equal(encoded[:, 6:26], q[:, 0, index])
        np.testing.assert_array_equal(encoded[:, 32:52], q[:, 1, index])

    def test_anatomical_names_and_ambiguous_alias_rejected(self):
        canonical = names('right')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            hand_indices(canonical + [canonical[0]], 'right')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            hand_indices(canonical + ['r_thumb_cmc_flex'], 'right')
        alternate = canonical.copy()
        alternate[7] = 'r_thumb_cmc_flex'
        self.assertEqual(hand_indices(alternate, 'right'), hand_indices(canonical, 'right'))

    def test_decode_both_sides_without_delta_clip_or_mask(self):
        x = np.arange(52, dtype=float)/9
        decoded = decode_command(x)
        for i, side in enumerate(SIDES):
            want = x[i*26:(i+1)*26]
            np.testing.assert_array_equal(decoded[side][:3], want[:3])
            np.testing.assert_array_equal(decoded[side][7:], want[6:])
            matrix = Rotation.from_quat(decoded[side][[4, 5, 6, 3]]).as_matrix()
            np.testing.assert_allclose(matrix, Rotation.from_rotvec(want[3:6]).as_matrix(), atol=1e-14)

    def test_pi_boundary_and_zero_quaternion(self):
        q = Rotation.from_rotvec([np.pi, 0, 0]).as_quat()
        pose = np.r_[1, 2, 3, q[[3, 0, 1, 2]]]
        np.testing.assert_allclose(pose6_from_wxyz(pose)[3:], [np.pi, 0, 0])
        with self.assertRaisesRegex(ValueError, 'Zero quaternion'):
            pose6_from_wxyz(np.zeros(7))
        with self.assertRaises(ValueError):
            decode_command(np.zeros(54))
        with self.assertRaises(ValueError):
            decode_command(np.full(52, np.nan))

    def test_commands_use_same_time_and_measured_inactive_hold(self):
        state = np.arange(4*52, dtype=np.float32).reshape(4, 52)/100
        pose = np.tile([0, 0, 0, 1, 0, 0, 0], (4, 2, 1)).astype(float)
        pose[:, 0, 0] = [10, 20, 30, 40]
        hand = np.arange(4*2*20).reshape(4, 2, 20)/1000
        action = encode_commands(pose, hand, state, ['right'])
        np.testing.assert_array_equal(action[:, 0], [10, 20, 30, 40])
        np.testing.assert_array_equal(action[:, 26:], state[:, 26:])
        np.testing.assert_allclose(action[:, 6:26], hand[:, 0], atol=1e-6)

    def fixture(self):
        robots = {}
        for i, side in enumerate(SIDES):
            data = SimpleNamespace(body_state_w=np.array([[[i, .2, .3, 1, 0, 0, 0]+[0]*6]]),
                                   joint_pos=np.arange(27, dtype=float)[None]/100)
            robots[side] = SimpleNamespace(joint_names=names(side), body_names=[side+'_palm_link'], data=data)
        return SimpleNamespace(_robots=robots, _hand_joint_ids={s: hand_indices(names(s), s) for s in SIDES})

    def test_live_uses_measured_not_last_command(self):
        env = self.fixture()
        env._right_tcp_pose = np.full(7, 999)
        state, evidence = live_state(env)
        self.assertEqual(state[0], 0)
        self.assertEqual(state[26], 1)
        self.assertEqual(evidence['right']['body'], 'right_palm_link')

    def test_lab_quaternion_orders_give_identical_policy_state_and_camera_pose(self):
        from handson.eef52 import measured_palm_pose
        from handson.evaluation.cameras import pose_matrix
        env = self.fixture()
        rotation = Rotation.from_euler('xyz', [.35, -.72, 1.19])
        xyzw = rotation.as_quat()
        for side in SIDES:
            env._robots[side].data.body_state_w[0, 0, 3:7] = xyzw[[3, 0, 1, 2]]
        state_wxyz, _ = live_state(env)
        expected = {}
        for side in SIDES:
            pose, _ = measured_palm_pose(env, side)
            expected[side] = pose_matrix(pose[:3], pose[3:])
            np.testing.assert_allclose(expected[side][:3, :3], rotation.as_matrix(), atol=1e-14)
            env._robots[side].data.body_state_w[0, 0, 3:7] = xyzw
        env._isaaclab_uses_xyzw = True
        state_xyzw, evidence = live_state(env)
        np.testing.assert_array_equal(state_xyzw, state_wxyz)
        np.testing.assert_allclose(state_xyzw[3:6], rotation.as_rotvec(), atol=1e-7)
        for side in SIDES:
            pose, _ = measured_palm_pose(env, side)
            np.testing.assert_allclose(pose_matrix(pose[:3], pose[3:]), expected[side], atol=1e-14)
            np.testing.assert_array_equal(env._robots[side].data.body_state_w[0, 0, 3:7], xyzw)
            self.assertEqual(evidence[side]['runtime_quaternion_order'], 'xyzw')

    def test_live_missing_palm_or_wrong_controller_order_fails(self):
        env = self.fixture()
        env._robots['right'].body_names = ['not_palm']
        with self.assertRaisesRegex(ValueError, 'palm'):
            live_state(env)
        env = self.fixture()
        env._hand_joint_ids['left'] = list(range(7, 27))
        with self.assertRaisesRegex(ValueError, 'order differs'):
            live_state(env)

    def test_readback_keeps_actuator_clamp_distinct(self):
        x = np.arange(52)/100
        requested = decode_command(x)
        env = SimpleNamespace(_joint_targets={s: np.zeros((1, 27)) for s in SIDES})
        for side in SIDES:
            setattr(env, '_'+side+'_tcp_pose', requested[side][:7])
            setattr(env, '_'+side+'_hand', requested[side][7:])
        report = command_readback(env, requested)
        for side in SIDES:
            self.assertEqual(report[side]['hand_max_abs'], 0)
            self.assertEqual(report[side]['position_max_abs'], 0)
            self.assertAlmostEqual(report[side]['rotation_geodesic_rad'], 0)
            self.assertEqual(report[side]['actuator_joint_target'], [0]*27)


if __name__ == '__main__':
    unittest.main()
