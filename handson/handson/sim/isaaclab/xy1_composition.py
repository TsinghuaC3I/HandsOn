"""Small D03 + B05 composition layer; physics/control remain in the D runtime."""
from __future__ import annotations

import itertools
import math
import numpy as np

from ...tasks.eval_comb03_pass_block_and_close_box import contract as xy1


def sample_layout(rng, randomize: bool) -> dict[str, np.ndarray]:
    result = {}
    for name, position in xy1.NOMINAL_POSITIONS.items():
        xyz = np.asarray(position, dtype=np.float64).copy()
        yaw = math.radians(xy1.NOMINAL_YAW_DEG.get(name, 0.0))
        if randomize:
            extent = np.asarray(xy1.XY_JITTER[name])
            xyz[:2] += rng.uniform(-extent, extent)
            yaw += math.radians(float(rng.uniform(*xy1.YAW_BOUNDS_DEG[name])))
        result[name] = np.asarray([*xyz, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    return result


def prepare_scene(root) -> None:
    """Remove D03's duplicate fixture before conversion, preserving all assets."""
    for worldbody in root.findall('worldbody'):
        for body in list(worldbody):
            if body.tag == 'body' and body.get('name') == 'd03_box':
                worldbody.remove(body)
    for body in root.iter('body'):
        position = xy1.NOMINAL_POSITIONS.get(body.get('name'))
        if position is not None:
            body.set('pos', ' '.join(str(v) for v in position))
            yaw = math.radians(xy1.NOMINAL_YAW_DEG.get(body.get('name'), 0.0))
            body.set('quat', f'{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}')


class XY1CompositionMixin:
    """Named hooks used only by XY1; no task-name impersonation or state writes."""

    def _xy1_initialize(self):
        self._xy1_a_stable_frames = 0
        self._xy1_a_complete = False
        self._xy1_lid_ready = False
        self._xy1_right_lid_seen = False
        self._xy1_sampled_poses = {}
        self._xy1_failure_reason = None

    def _xy1_reset(self):
        self._xy1_initialize()
        self._lid_start_pose = None
        self._lid_was_moved = False
        self._last_b05_lid_success_metrics = {}
        self._xy1_sampled_poses = sample_layout(self._random, self.randomize)
        self._poses.update({n: p.copy() for n, p in self._xy1_sampled_poses.items()})

    def _xy1_apply_reset(self):
        for name, pose in self._xy1_sampled_poses.items():
            self._write_reset_pose(name, xy=pose[:2], z=float(pose[2]),
                                   orientation_xyzw=np.asarray([0., 0., 0., 1.]),
                                   yaw_degrees=math.degrees(2 * math.atan2(pose[6], pose[3])))

    def _xy1_block_inside(self, target, box):
        from .abc_collection_env import _pose_vector_to_matrix, _invert_pose_matrix
        relative = _invert_pose_matrix(_pose_vector_to_matrix(box)) @ _pose_vector_to_matrix(target)
        corners = np.asarray([(*c, 1.0) for c in itertools.product(
            (-xy1.BLOCK_HALF_EXTENT, xy1.BLOCK_HALF_EXTENT), repeat=3)])
        vertices = (relative @ corners.T).T[:, :3]
        half = np.asarray(xy1.INNER_WALL_HALF0) + (
            vertices[:, 2, None] - xy1.INNER_WALL_Z0
        ) * np.asarray(xy1.INNER_WALL_SLOPE)
        return bool(
            -0.058 <= relative[2, 3] <= 0.055
            and np.all(np.abs(vertices[:, :2]) <= half + xy1.CONTAINMENT_CONTACT_TOLERANCE)
            and np.min(vertices[:, 2]) >= -0.071314 - xy1.CONTAINMENT_CONTACT_TOLERANCE
            and np.max(vertices[:, 2]) <= 0.0745 + xy1.CONTAINMENT_CONTACT_TOLERANCE
        )

    def _xy1_compute_success(self):
        previous_stage = self._d03_stage
        a = self._d03_metrics(update=not self._xy1_a_complete,
                              placement_counter=self._xy1_a_stable_frames)
        if previous_stage in (1, 2) and self._d03_stage == 0:
            self._d03_left_streak = self._d03_both_streak = self._d03_right_streak = 0
        lid = self._poses[xy1.LID]
        box = self._poses[xy1.BOX]
        if not self._xy1_a_complete:
            valid = self._d03_stage >= 3 and a['placement_stable']
            self._xy1_a_stable_frames = self._xy1_a_stable_frames + 1 if valid else 0
            if self._xy1_a_stable_frames >= xy1.A_STABLE_FRAMES:
                self._xy1_a_complete = True
                self._d03_stage = 4
                self._lid_start_pose = lid.copy()
                self._lid_was_moved = False
            self._d_last_metrics = {**a, 'xy1_a_complete': self._xy1_a_complete,
                                    'xy1_right_lid_seen': False, 'xy1_success_now': False}
            return False

        # A preclosed lid is recoverable, but cannot receive credit without a
        # new, post-A pickup. Being near the rim while still uncovered is fine.
        from .abc_collection_env import (
            _pose_vector_to_matrix, _invert_pose_matrix, _LID_GOAL_LOCAL_Z,
        )
        lid_local = _invert_pose_matrix(_pose_vector_to_matrix(box)) @ _pose_vector_to_matrix(lid)
        separated = bool(
            not self._b05_full_coverage_metrics(lid, box)['full_coverage']
            or lid_local[2, 3] > _LID_GOAL_LOCAL_Z['t03_cover_box'] + .025
        )
        if separated and not self._xy1_lid_ready:
            self._xy1_lid_ready = True
        right_contact = self._d_hand_object_contact('right', xy1.LID, 0.065)
        left_contact = self._d_hand_object_contact('left', xy1.LID, 0.065)
        right_only = right_contact and not left_contact
        update_movement = False
        # The recorded right_lid_seen flag establishes a contact candidate;
        # lid_was_moved alone is pickup credit. A first right touch establishes
        # a NEW origin, never credit for movement since a left-owned sample.
        if left_contact and not right_contact:
            self._lid_start_pose = lid.copy()
            self._lid_was_moved = False
            self._xy1_right_lid_seen = False
        elif not self._lid_was_moved:
            if not right_only or not self._xy1_right_lid_seen:
                self._lid_start_pose = lid.copy()
                self._xy1_right_lid_seen = right_only
            else:
                update_movement = self._xy1_lid_ready
        # A brief second-hand touch after a real right pickup preserves its
        # credit; only a left-only takeover invalidates it. Before pickup,
        # mixed/no contact rebases the candidate so it cannot accumulate lift.
        if self._lid_start_pose is None:
            self._lid_start_pose = lid.copy()
        b_metrics = self._b05_lid_fit_metrics(
            lid,
            box,
            start_pose=self._lid_start_pose,
            was_moved=self._lid_was_moved,
            lid_name=xy1.LID,
        )
        self._lid_was_moved = bool(
            b_metrics.get("b05_moved", self._lid_was_moved)
            if update_movement
            else self._lid_was_moved
        )
        self._last_b05_lid_success_metrics = b_metrics
        b = bool(b_metrics.get("b05_success_now", False))
        success = bool(a['placement_stable'] and self._xy1_lid_ready
                       and self._lid_was_moved and b)
        self._d_last_metrics = {
            **a, **self._last_b05_lid_success_metrics,
            'xy1_a_complete': self._xy1_a_complete,
            'xy1_right_lid_seen': bool(self._lid_was_moved),
            'xy1_success_now': success,
        }
        return success

    def _xy1_hidden_state(self):
        scalars = [float(getattr(self, '_' + n)) for n in xy1.HIDDEN_FIELDS[:14]]
        pose = self._lid_start_pose
        return np.asarray([*scalars, float(pose is not None),
                           *(np.zeros(7) if pose is None else pose)], dtype=np.float64)

    def _xy1_restore_hidden_state(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(len(xy1.HIDDEN_FIELDS))
        bool_fields = {'d03_left_confirmed', 'd03_both_confirmed', 'd03_right_confirmed',
                       'd03_direct_right_seen', 'xy1_a_complete', 'xy1_lid_ready',
                       'xy1_right_lid_seen', 'lid_was_moved'}
        for name, value in zip(xy1.HIDDEN_FIELDS[:14], values[:14]):
            setattr(self, '_' + name, bool(value) if name in bool_fields else int(value))
        self._lid_start_pose = values[15:22].copy() if values[14] else None
        self._d_last_metrics = {}
        self._last_b05_lid_success_metrics = {}

    def _xy1_condition_rows(self):
        cached_metrics = self._d_last_metrics
        a = self._d03_metrics(update=False)
        self._d_last_metrics = cached_metrics
        b = self._last_b05_lid_success_metrics
        rows = [
            self._condition_row('xy1_left', 'Left hand lifted the block first', self._d03_left_confirmed),
            self._condition_row('xy1_both', 'Both hands contacted the block in the middle', self._d03_both_confirmed),
            self._condition_row('xy1_right', 'Right hand received after left release', self._d03_right_confirmed),
            self._condition_row('xy1_a', 'Handoff and released placement completed', self._xy1_a_complete),
            self._condition_row('xy1_lid_pickup', 'Right hand picked up lid after placement', self._lid_was_moved),
            self._condition_row('xy1_containment', 'Block remains released and stable inside box', a['placement_stable']),
        ]
        for key, label in (
            ('b05_full_coverage', 'Lid covers more than 90% of the opening'),
            ('b05_height_close', 'Lid at closing height'),
            ('b05_top_aligned', 'Lid lies flat'),
            ('b05_lid_stable', 'Lid stable'),
            ('b05_base_stable', 'Box stable'),
            ('b05_base_upright', 'Box upright'),
            ('b05_base_on_table', 'Box supported by table'),
            ('b05_hand_released', 'Both hands clear of lid and box'),
        ):
            rows.append(self._condition_row(key, label, bool(b.get(key, False))))
        return rows
