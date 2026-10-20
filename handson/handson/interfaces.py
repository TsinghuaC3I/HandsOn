"""Lightweight EEF52 boundary shared by simulation and model processes."""
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

STATE_DIM = ACTION_DIM = 52
CONTROL_HZ = 30
VIEWS = ("front", "wrist_left", "wrist_right")
RENDER_VIEWS = ("main", "wrist_left", "wrist_right")


@runtime_checkable
class Policy(Protocol):
    """Return the full native horizon in absolute world EEF52 units.

    Each side is xyz, rotation vector and twenty hand joints, right then left.
    Adapters own the model's original preprocessing and normalization.
    """
    def predict(self, state: np.ndarray, images: Mapping[str, np.ndarray],
                prompt: str, seed: int) -> np.ndarray: ...


def validate_observation(state, images, prompt, history_frames=1):
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError("Expected finite measured EEF52 state")
    if not isinstance(prompt, str) or not 0 < len(prompt.strip()) < 4000:
        raise ValueError("Invalid task instruction")
    if set(images) != set(VIEWS):
        raise ValueError("All three camera views are required")
    for name, frame in images.items():
        frame = np.asarray(frame)
        rank = 4 if history_frames > 1 else 3
        if frame.dtype != np.uint8 or frame.ndim != rank or frame.shape[-1] != 3:
            raise ValueError(f"{name}: expected uint8 RGB with rank {rank}")
        if not all(s > 0 for s in frame.shape):
            raise ValueError(f"{name}: empty image")
        if history_frames > 1 and frame.shape[0] != history_frames:
            raise ValueError(f"{name}: wrong history length")
    return state


def validate_actions(actions, horizon):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (horizon, ACTION_DIM) or not np.isfinite(actions).all():
        raise ValueError("Model must return finite physical world EEF52 commands "
                         f"with shape ({horizon}, {ACTION_DIM}); got {actions.shape}")
    return actions
