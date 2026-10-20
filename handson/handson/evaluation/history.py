"""Episode-owned image history; recording never changes global functions."""
import numpy as np

from ..interfaces import RENDER_VIEWS
from .contracts import require


class ImageHistory:
    def __init__(self, frames=1, stride=30, horizon=30, render_stride=30):
        require(frames >= 1, "History frames must be positive")
        if frames > 1:
            require(stride > 0 and stride == render_stride and horizon % stride == 0,
                    "History stride must equal the render stride and divide the action horizon")
        self.frames, self.stride, self.horizon = frames, stride, horizon
        self._ring = {}

    def record(self, step, images):
        if self.frames > 1:
            self._ring[step] = {k: np.array(images[k], copy=True) for k in RENDER_VIEWS}
            first = max(0, step - self.stride * (self.frames - 1))
            for old in [s for s in self._ring if s < first]:
                del self._ring[old]

    def views(self, images, index):
        if self.frames == 1:
            return {k: images[k] for k in RENDER_VIEWS}
        step = index * self.horizon
        if index == 0:
            self._ring.clear()
            self.record(0, images)
        require(bool(self._ring) and max(self._ring) == step,
                f"History ring does not end at request {index}, step {step}")
        require(all(np.array_equal(self._ring[step][k], images[k]) for k in RENDER_VIEWS),
                "History ring differs from rollout frames")
        steps = [max(0, step - self.stride * (self.frames - 1 - j)) for j in range(self.frames)]
        require(all(s in self._ring for s in steps), f"Missing history frames at {steps}")
        return {k: np.stack([self._ring[s][k] for s in steps]) for k in RENDER_VIEWS}
