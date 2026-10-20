"""Small helpers for reading MuJoCo sensor values across numpy versions."""

from __future__ import annotations

from typing import Any

import numpy as np


def sensor_scalar(data: Any) -> float:
    """Return a single-value MuJoCo sensor as a Python float."""
    arr = np.asarray(data)
    if arr.size != 1:
        raise ValueError(f"expected single-value sensor data, got shape {arr.shape}")
    return float(arr.reshape(-1)[0])
