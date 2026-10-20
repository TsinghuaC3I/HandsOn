"""EEF52 <-> Alpha80 boundary. No robot joint is compressed or discarded."""
import numpy as np
from scipy.spatial.transform import Rotation

HAND_NAMES = [f"finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]
ALPHA_VALID = np.zeros(80, dtype=bool)
for lo, hi in ((0, 9), (10, 30), (34, 43), (44, 64)):
    ALPHA_VALID[lo:hi] = True


def eef52_to_alpha80(x):
    x = np.asarray(x, np.float32)
    if x.shape[-1] != 52 or not np.isfinite(x).all():
        raise ValueError("Expected finite EEF52")
    result = np.zeros(x.shape[:-1] + (80,), np.float32)
    for src, dst, hand in ((26, 0, 10), (0, 34, 44)):
        result[..., dst:dst + 3] = x[..., src:src + 3]
        mat = Rotation.from_rotvec(x[..., src + 3:src + 6].reshape(-1, 3)).as_matrix()
        rot = np.concatenate((mat[:, :, 0], mat[:, :, 1]), axis=-1)
        result[..., dst + 3:dst + 9] = rot.reshape(x.shape[:-1] + (6,))
        result[..., hand:hand + 20] = x[..., src + 6:src + 26]
    return result


def rot6d_to_matrix(x):
    """Two-column Gram-Schmidt with deterministic fallbacks for degenerate outputs.

    A fresh checkpoint may generate zero/parallel columns. Define a valid rotation
    instead of producing NaNs; callers can count those cases as diagnostics.
    """
    x = np.asarray(x, np.float64)
    a, b = x[..., :3], x[..., 3:]
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    e1 = np.broadcast_to(np.array([1., 0., 0.]), a.shape)
    a = np.where(n > 1e-8, a / np.maximum(n, 1e-8), e1)
    b = b - (a * b).sum(-1, keepdims=True) * a
    n = np.linalg.norm(b, axis=-1, keepdims=True)
    basis = np.eye(3)[np.argmin(np.abs(a), axis=-1)]
    fallback = basis - (basis * a).sum(-1, keepdims=True) * a
    fallback /= np.linalg.norm(fallback, axis=-1, keepdims=True)
    b = np.where(n > 1e-8, b / np.maximum(n, 1e-8), fallback)
    return np.stack((a, b, np.cross(a, b)), axis=-1)


def alpha80_to_eef52(x):
    x = np.asarray(x, np.float32)
    if x.shape[-1] != 80 or not np.isfinite(x).all():
        raise ValueError("Expected finite Alpha80")
    result = np.zeros(x.shape[:-1] + (52,), np.float32)
    for dst, src, hand in ((26, 0, 10), (0, 34, 44)):
        result[..., dst:dst + 3] = x[..., src:src + 3]
        mat = rot6d_to_matrix(x[..., src + 3:src + 9])
        result[..., dst + 3:dst + 6] = Rotation.from_matrix(mat.reshape(-1, 3, 3)).as_rotvec().reshape(x.shape[:-1] + (3,))
        result[..., dst + 6:dst + 26] = x[..., hand:hand + 20]
    return result


class PolicyNormalizer:
    """The saved transforms are used by both training and inference."""
    def __init__(self, stats, policy):
        self.stats, self.policy = stats, policy

    def bounds(self, kind):
        # Official Motus reader uses action min/max for state as well.
        kind = "action" if self.policy == "motus" else kind
        return tuple(np.asarray(self.stats[kind][k], np.float64) for k in ("min", "max"))

    def normalize(self, x, kind):
        x = np.asarray(x, np.float64)
        lo, hi = self.bounds(kind)
        if self.policy == "motus":
            width = np.where(hi == lo, 1.0, hi - lo)
            return ((x - lo) / width).astype(np.float32)
        # Preserve rotations exactly; scatter applies their conversion afterward.
        y = np.clip(2 * (x - lo) / np.maximum(hi - lo, 1e-6) - 1, -1, 1)
        mapped = eef52_to_alpha80(x)
        for src, dst, hand in ((26, 0, 10), (0, 34, 44)):
            mapped[..., dst:dst + 3] = y[..., src:src + 3]
            mapped[..., hand:hand + 20] = y[..., src + 6:src + 26]
        return mapped

    def denormalize(self, x, kind="action"):
        lo, hi = self.bounds(kind)
        if self.policy == "motus":
            return (np.asarray(x) * (hi - lo) + lo).astype(np.float32)
        result = alpha80_to_eef52(x)
        # Native OpenWAM inverse does not clip model predictions.
        for start in (0, 26):
            for a, b in ((start, start + 3), (start + 6, start + 26)):
                width = np.maximum(hi[a:b] - lo[a:b], 1e-6)
                result[..., a:b] = (result[..., a:b] + 1) * .5 * width + lo[a:b]
        return result.astype(np.float32)

    @property
    def valid_mask(self):
        return np.ones(52, bool) if self.policy == "motus" else ALPHA_VALID.copy()
