"""XY6 composition of the unchanged BC014 and KA012 source assets."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np


TASK_NAME = "eval_comb06_season_and_stir"
TASK_ID = "XY6"
PROMPT = (
    "Use the left hand for salt and the right hand for pepper. Pour both into "
    "the same open pot together, then put down and release the jars. Hold the "
    "pot with the left hand and stir with the right-hand spoon through all six "
    "angular sectors and more than 650 degrees, keeping the pot stable."
)
OBJECT_NAMES = ("salt_jar", "pepper_jar", "dutch_oven", "dutch_oven_lid", "spoon")
# Half ranges about authored supported poses. Every free body translates in
# both tabletop axes; no height randomization that could bury a source mesh.
TRANSLATION_HALF_RANGES_M = {
    "salt_jar": (0.020, 0.020),
    "pepper_jar": (0.020, 0.020),
    "dutch_oven": (0.015, 0.015),
    "dutch_oven_lid": (0.020, 0.020),
    "spoon": (0.020, 0.015),
}
YAW_HALF_RANGES_DEG = {
    "salt_jar": 12.0, "pepper_jar": 12.0, "dutch_oven": 8.0,
    "dutch_oven_lid": 8.0, "spoon": 8.0,
}
# Source mesh inner lip radius >=81 mm, at uniform scale .78. A 58 mm
# receiving disk keeps >5 mm clearance inside that lip. Heights are in the
# scaled asset frame, whose +Y becomes world +Z under the original +90deg X.
# These are semantic regions, not replacement collision/visual geometry.
POT_OPENING_HEIGHT_M = 0.131555 * 0.78
POT_RECEIVER_HEIGHT_M = 0.122 * 0.78
POT_INNER_RADIUS_M = 0.058
RESET_MAX_ATTEMPTS = 128


def make_task_spec(seasoning, stirring):
    """Copy physical declarations, changing only identity and placement."""
    original = {s.name: s for s in (*seasoning.object_specs, *stirring.object_specs)}
    positions = {
        "salt_jar": (-0.35, 0.23, original["salt_jar"].home_translation[2]),
        "pepper_jar": (-0.35, -0.23, original["pepper_jar"].home_translation[2]),
        "dutch_oven": (-0.195, 0.06, original["dutch_oven"].home_translation[2]),
        "dutch_oven_lid": (0.09, 0.31, original["dutch_oven_lid"].home_translation[2]),
        "spoon": (-0.14, -0.22, original["spoon"].home_translation[2]),
    }
    return replace(
        stirring, task_id=TASK_ID, task_name=TASK_NAME,
        label="XY6 Synchronized Seasoning Then Stabilized Stirring", prompt=PROMPT,
        time_limit_seconds=90, left_role="salt_jar", right_role="pepper_jar",
        object_specs=tuple(replace(original[n], home_translation=positions[n]) for n in OBJECT_NAMES),
        goal_pose_wxyz=(*positions["dutch_oven"], *original["dutch_oven"].home_orientation_wxyz),
        success_labels=(
            "X: left salt and right pepper physically contacted",
            "X: salt hits the pot for at least 0.6 seconds",
            "X: pepper hits the pot for at least 0.6 seconds",
            "X: both streams hit together for 0.4 continuous seconds",
            "X complete; both jars released before stirring",
            *(label.replace("six ordered angular sectors", "six angular sectors")
              for label in stirring.success_labels),
        ),
        source_asset_keys=tuple(k for k in dict.fromkeys((*seasoning.source_asset_keys, *stirring.source_asset_keys)) if k != "target_marker"),
    )


def validate_asset_files(task_spec):
    """Reuse original bundle checks without requiring unrelated B320 props."""
    from .b320_bimanual_scene import B320_SOURCE_BUNDLE_SHA256, _source_bundle_sha256

    for spec in task_spec.object_specs:
        if not spec.source_path.is_file() or spec.source_path.stat().st_size < 1024:
            raise FileNotFoundError(f"XY6 source asset is missing or still an LFS pointer: {spec.source_path}")
    for key in task_spec.source_asset_keys:
        if key in B320_SOURCE_BUNDLE_SHA256:
            actual = _source_bundle_sha256(key)
            if actual != B320_SOURCE_BUNDLE_SHA256[key]:
                raise ValueError(f"XY6 reused B320 bundle hash mismatch: {key}")


def rotation_wxyz(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def pot_flow_projection(jar_pose, pot_pose, *, jar_height, tilt_threshold, maximum_length):
    """Cap ray must enter through the moving opening and reach the interior.

    The runtime additionally tests the ray segment against actual collision
    shapes so a lid, hand, jar or pot wall cannot be transparent to scoring.
    """
    jar_pose, pot_pose = np.asarray(jar_pose), np.asarray(pot_pose)
    cap_axis = rotation_wxyz(jar_pose[3:7])[:, 2]
    cap_position = jar_pose[:3] + cap_axis * jar_height
    pot_rotation = rotation_wxyz(pot_pose[3:7])
    local_cap = pot_rotation.T @ (cap_position - pot_pose[:3])
    local_axis = pot_rotation.T @ cap_axis
    tilt = math.degrees(math.acos(float(np.clip(cap_axis[2], -1.0, 1.0))))
    length = 0.0
    opening_length = 0.0
    valid = False
    if tilt >= tilt_threshold and cap_axis[2] < -1e-6 and local_axis[1] < -1e-6:
        length = float((POT_RECEIVER_HEIGHT_M - local_cap[1]) / local_axis[1])
        opening_length = float((POT_OPENING_HEIGHT_M - local_cap[1]) / local_axis[1])
        valid = bool(0.010 <= length <= maximum_length and opening_length >= 0.0)
    impact = cap_position + cap_axis * length if valid else cap_position.copy()
    hit_local = local_cap + local_axis * length
    opening_local = local_cap + local_axis * opening_length
    margin = POT_INNER_RADIUS_M - max(
        float(np.linalg.norm(hit_local[[0, 2]])),
        float(np.linalg.norm(opening_local[[0, 2]])),
    )
    return {
        "cap_axis": cap_axis, "cap_position": cap_position,
        "impact_position": impact, "stream_length_m": length if valid else 0.0,
        "tilt_deg": tilt, "ray_valid": valid,
        "zone_offset_xy_m": hit_local[[0, 2]], "zone_margin_m": margin,
        "projects_into_zone": bool(valid and margin >= 0.0),
    }
