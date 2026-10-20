"""Backend-independent physical success geometry for round-peg assembly."""

from __future__ import annotations

import numpy as np


PEG_COLLISION_LOCAL_POS = np.asarray([0.0, 0.0, 0.081], dtype=np.float64)
PEG_SHAFT_HALF_LENGTH = 0.0675
SOCKET_SITE_LOCAL_POS = np.asarray([0.0, 0.0, 0.027], dtype=np.float64)

# The socket's 20.25 mm inner half-width leaves roughly 2.4 mm radial
# clearance around the 17.85 mm peg.  Six millimetres tolerates contact solver
# compliance without accepting a peg resting beside the hole.
ASSEMBLY_RADIAL_TOLERANCE = 0.006
ASSEMBLY_MIN_INSERTION_DEPTH = 0.010
ASSEMBLY_MAX_INSERTION_DEPTH = 0.024
ASSEMBLY_AXIS_ALIGNMENT = float(np.cos(np.deg2rad(20.0)))
ASSEMBLY_HANDLE_MIN_OUTSIDE_DEPTH = 0.040
ASSEMBLY_STABLE_LINEAR_SPEED = 0.020
ASSEMBLY_STABLE_ANGULAR_SPEED = 0.30
ASSEMBLY_STABLE_HOLD_SECONDS = 0.50


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    value = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = value / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def assembly_geometry(
    peg_pose_wxyz: np.ndarray,
    socket_pose_wxyz: np.ndarray,
) -> dict[str, float | bool | np.ndarray]:
    """Measure the narrow peg tip at the socket-mouth plane."""

    peg = np.asarray(peg_pose_wxyz, dtype=np.float64).reshape(7)
    socket = np.asarray(socket_pose_wxyz, dtype=np.float64).reshape(7)
    peg_rot = _quat_wxyz_to_matrix(peg[3:7])
    socket_rot = _quat_wxyz_to_matrix(socket[3:7])

    socket_site = socket[:3] + socket_rot @ SOCKET_SITE_LOCAL_POS
    socket_axis = socket_rot[:, 2]
    socket_axis /= max(float(np.linalg.norm(socket_axis)), 1e-12)
    peg_center = peg[:3] + peg_rot @ PEG_COLLISION_LOCAL_POS
    peg_axis = peg_rot[:, 2]
    peg_axis /= max(float(np.linalg.norm(peg_axis)), 1e-12)

    center_delta = peg_center - socket_site
    center_axial = float(np.dot(center_delta, socket_axis))
    inserted_tip = peg_center - peg_axis * PEG_SHAFT_HALF_LENGTH
    handle_end = peg_center + peg_axis * PEG_SHAFT_HALF_LENGTH
    insertion_depth = float(np.dot(socket_site - inserted_tip, socket_axis))
    handle_end_depth = float(np.dot(socket_site - handle_end, socket_axis))

    peg_socket_axis_dot = float(np.dot(peg_axis, socket_axis))
    if abs(peg_socket_axis_dot) > 1e-12:
        mouth_offset = float(
            np.clip(
                -center_axial / peg_socket_axis_dot,
                -PEG_SHAFT_HALF_LENGTH,
                PEG_SHAFT_HALF_LENGTH,
            )
        )
        mouth_point = peg_center + mouth_offset * peg_axis
    else:
        mouth_point = inserted_tip.copy()
    mouth_delta = mouth_point - socket_site
    mouth_axial = float(np.dot(mouth_delta, socket_axis))
    radial_vector = mouth_delta - mouth_axial * socket_axis
    signed_alignment = float(np.dot(peg_axis, socket_axis))

    return {
        "socket_site": socket_site,
        "socket_axis": socket_axis,
        "peg_center": peg_center,
        "peg_axis": peg_axis,
        "inserted_tip": inserted_tip,
        "mouth_point": mouth_point,
        "radial_error": float(np.linalg.norm(radial_vector)),
        "insertion_depth": insertion_depth,
        "handle_end_depth": handle_end_depth,
        "axis_alignment": float(abs(signed_alignment)),
        "signed_axis_alignment": signed_alignment,
        "tip_end_correct": bool(signed_alignment > 0.0),
        "handle_outside": bool(
            handle_end_depth <= -ASSEMBLY_HANDLE_MIN_OUTSIDE_DEPTH
        ),
    }


def assembly_geometry_is_inserted(
    geometry: dict[str, float | bool | np.ndarray],
) -> bool:
    return bool(
        float(geometry["radial_error"]) <= ASSEMBLY_RADIAL_TOLERANCE
        and ASSEMBLY_MIN_INSERTION_DEPTH
        <= float(geometry["insertion_depth"])
        <= ASSEMBLY_MAX_INSERTION_DEPTH
        and float(geometry["axis_alignment"]) >= ASSEMBLY_AXIS_ALIGNMENT
        and bool(geometry["tip_end_correct"])
        and bool(geometry["handle_outside"])
    )

