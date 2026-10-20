"""Pure pose-error bounds and legacy YCB11 envelope helpers.

The live environment preserves raw actions and uses ``limit_pose_motion``
only for measured-palm drive error. Legacy contact-distance constants remain
available for reading older recordings.
"""
from __future__ import annotations
import math
import numpy as np

YCB11_CONTACT_GUARD_DISTANCE_M = 0.030
YCB11_FREE_PALM_STEP_M = 0.003
YCB11_CONTACT_PALM_STEP_M = 0.0005
YCB11_FREE_ROTATION_STEP_DEG = 2.0
YCB11_CONTACT_ROTATION_STEP_DEG = 0.15
YCB11_FREE_HAND_STEP_RAD = 0.03
YCB11_CONTACT_HAND_STEP_RAD = 0.01
YCB11_CONTACT_HAND_TARGET_LEAD_RAD = 0.02
YCB11_CONTACT_PALM_TARGET_LEAD_M = 0.002
YCB11_CONTACT_ROTATION_TARGET_LEAD_DEG = 0.5


def contact_guard_active(clearance_m: float, *, readback_failed: bool = False) -> bool:
    return (readback_failed or not math.isfinite(float(clearance_m))
            or float(clearance_m) <= YCB11_CONTACT_GUARD_DISTANCE_M)


def limit_pose_motion(previous: np.ndarray, requested: np.ndarray,
                      translation_m: float, rotation_deg: float) -> np.ndarray:
    """Bound translation AND angular sweep; a translation-only clamp is unsafe."""
    previous = np.asarray(previous, dtype=np.float64).reshape(7)
    requested = np.asarray(requested, dtype=np.float64).reshape(7)
    if not np.all(np.isfinite(previous)) or np.linalg.norm(previous[3:]) < 1e-9:
        raise ValueError("YCB11 previous pose is invalid")
    if not np.all(np.isfinite(requested)) or np.linalg.norm(requested[3:]) < 1e-9:
        return previous.copy()
    result = requested.copy()
    delta = requested[:3] - previous[:3]
    length = float(np.linalg.norm(delta))
    result[:3] = previous[:3] + delta * min(1.0, translation_m / max(length, 1e-12))
    q0 = previous[3:] / np.linalg.norm(previous[3:])
    q1 = requested[3:] / np.linalg.norm(requested[3:])
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    half_angle = math.acos(float(np.clip(dot, -1.0, 1.0)))
    fraction = min(1.0, math.radians(rotation_deg) / max(2 * half_angle, 1e-12))
    if half_angle < 1e-8:
        result[3:] = q1
    else:
        result[3:] = (math.sin((1-fraction)*half_angle)*q0
                      + math.sin(fraction*half_angle)*q1) / math.sin(half_angle)
    return result
