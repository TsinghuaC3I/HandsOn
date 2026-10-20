"""Contact-acquired, force-limited D20 tool grip and its replay state."""
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation

STATE_SIZE = 49
ACQUIRE_COMMAND = .45
RELEASE_COMMAND = .25
CONTACT_FORCE = .04
ACQUIRE_FRAMES = 2
BREAK_DISTANCE = .065


def hand_grip_command(values):
    """Wuji long-finger flexion, excluding thumb opposition and finger spread."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size != 20 or not np.isfinite(values).all():
        return 0.0
    flexion = np.mean(np.abs(values.reshape(5, 4)[1:, 1:]), axis=1)
    return float(np.mean(np.sort(flexion)[-2:]))


def limit_vector(value, maximum):
    value = np.asarray(value, dtype=np.float64)
    return value * min(1., maximum / max(float(np.linalg.norm(value)), 1e-12))


def grip_wrench(current, velocity, target, previous, *, dt, mass, payload=False):
    """Finite spring/damper forces: the solver retains contact and slip."""
    dt = max(float(dt), 1e-6)
    v = np.asarray(velocity, dtype=np.float64)
    linear = np.zeros(3)
    angular = np.zeros(3)
    if previous is not None:
        linear = limit_vector((target[:3, 3]-previous[:3, 3])/dt, .8)
        angular = limit_vector(Rotation.from_matrix(target[:3, :3]@previous[:3, :3].T).as_rotvec()/dt, 4.)
    error = target[:3, 3]-current[:3, 3]
    rotation = Rotation.from_matrix(target[:3, :3]@current[:3, :3].T).as_rotvec()
    kp, kd, kr, dr, force_limit, torque_limit = (
        (400., 12., .4, .05, 10., .3) if payload else (600., 18., .8, .08, 20., .6)
    )
    force = kp*error + kd*(linear-v[:3]) + np.array([0., 0., mass*9.81])
    torque = kr*rotation + dr*(angular-v[3:])
    return np.r_[limit_vector(force, force_limit), limit_vector(torque, torque_limit)]


@dataclass
class ToolGrip:
    side: str = ''
    offset: np.ndarray | None = None
    previous: np.ndarray | None = None
    contacts: np.ndarray = field(default_factory=lambda: np.zeros(2))
    tool_wrench: np.ndarray = field(default_factory=lambda: np.zeros(6))
    payload_wrench: np.ndarray = field(default_factory=lambda: np.zeros(6))

    def reset(self):
        self.side = ''; self.offset = None; self.previous = None
        self.contacts.fill(0); self.tool_wrench.fill(0); self.payload_wrench.fill(0)

    def observation(self):
        return np.r_[{'':0, 'right':1, 'left':2}[self.side], self.offset is not None,
            self.previous is not None, self.contacts,
            (self.offset if self.offset is not None else np.zeros((4,4))).ravel(),
            (self.previous if self.previous is not None else np.zeros((4,4))).ravel(),
            self.tool_wrench, self.payload_wrench].astype(np.float64)

    def restore(self, values):
        self.reset()
        a = np.asarray(values, dtype=np.float64).ravel()
        if a.size != STATE_SIZE or not np.isfinite(a).all():
            return
        self.side = {0:'', 1:'right', 2:'left'}.get(int(a[0]), '')
        self.offset = a[5:21].reshape(4,4).copy() if a[1] else None
        self.previous = a[21:37].reshape(4,4).copy() if a[2] else None
        self.contacts[:] = a[3:5]
        self.tool_wrench[:] = a[37:43]; self.payload_wrench[:] = a[43:49]
