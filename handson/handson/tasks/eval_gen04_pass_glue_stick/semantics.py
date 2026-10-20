"""D03-ordered handoff with geometry supplied by the selected GEN4 assets.

No simulator state is written here. Live callers must supply real contacts,
support and velocities; these predicates cannot manufacture a grasp.
"""

from dataclasses import dataclass
import numpy as np

from .contract import OWNERSHIP_FRAMES, SUCCESS_FRAMES


@dataclass
class HandoffState:
    stage: int = 0
    left_streak: int = 0
    both_streak: int = 0
    right_streak: int = 0
    left_confirmed: bool = False
    both_confirmed: bool = False
    right_confirmed: bool = False
    direct_right_seen: bool = False

    def advance(self, *, left, right, contacts_available, lifted, transfer, stable, success_counter):
        if not contacts_available:
            # Unknown contact cannot establish ownership or a hand release.
            # Break the pending consecutive window, retaining proven stages so
            # ordinary sensor recovery can continue without a task reset.
            if self.stage in (0, 1, 2):
                pending_streak = ("left_streak", "both_streak", "right_streak")[self.stage]
                setattr(self, pending_streak, 0)
            return
        if self.stage == 0:
            self.direct_right_seen |= bool(right and not left)
            valid = left and not right and lifted
            self.left_streak = self.left_streak + 1 if valid else 0
            if self.left_streak >= OWNERSHIP_FRAMES:
                self.left_confirmed = True
                self.stage = 1
        elif self.stage == 1:
            valid = left and right and transfer
            self.both_streak = self.both_streak + 1 if valid else 0
            if self.both_streak >= OWNERSHIP_FRAMES:
                self.both_confirmed = True
                self.stage = 2
        elif self.stage == 2:
            valid = right and not left and transfer
            self.right_streak = self.right_streak + 1 if valid else 0
            if self.right_streak >= OWNERSHIP_FRAMES:
                self.right_confirmed = True
                self.stage = 3
        if self.stage in (1, 2) and not lifted and not (left or right):
            # A recoverable table drop starts a fresh ownership sequence.
            diagnostic = self.direct_right_seen
            self.__dict__.update(HandoffState().__dict__)
            self.direct_right_seen = diagnostic
        if self.stage == 3 and stable and success_counter >= SUCCESS_FRAMES - 1:
            self.stage = 4

    def vector(self):
        return np.asarray(list(self.__dict__.values()), dtype=np.float64)

    @classmethod
    def from_vector(cls, vector):
        values = np.asarray(vector, dtype=np.float64).reshape(8)
        if not np.isfinite(values).all() or not 0 <= values[0] <= 4:
            raise ValueError("Invalid GEN4 recorded handoff state")
        return cls(*(int(v) for v in values[:4]), *(bool(v) for v in values[4:]))


def transformed_points(points, pose):
    """Transform source-derived occupied vertices by a WXYZ pose."""
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    q = pose[3:]
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    rotation = np.asarray([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    return np.asarray(points) @ rotation.T + pose[:3]


def tray_geometry(points, glue_pose, tray_pose, interior_xy, floor_z):
    """Full footprint plus floor support band; standing above the rim is valid."""
    world = transformed_points(points, glue_pose)
    basis = transformed_points(np.eye(3), tray_pose) - np.asarray(tray_pose[:3])
    local = (world - np.asarray(tray_pose[:3])) @ basis.T
    low, high = np.asarray(interior_xy, dtype=np.float64)
    footprint_inside = bool(
        np.all(local[:, :2].min(axis=0) >= low - 1e-5)
        and np.all(local[:, :2].max(axis=0) <= high + 1e-5)
    )
    floor_gap = float(local[:, 2].min() - floor_z)
    return {
        "footprint_inside": footprint_inside,
        "floor_gap_m": floor_gap,
        "at_floor": -0.002 <= floor_gap <= 0.004,
        "lowest_world_z": float(world[:, 2].min()),
    }
