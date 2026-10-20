"""Utilities for scaling task objects without touching robot bodies."""

from __future__ import annotations

import os
import weakref

import mujoco
import numpy as np


DEFAULT_TASK_OBJECT_SCALE = 0.8

_UNSCALED_YCB_OBJECT_ROOT_BODIES = frozenset(
    {
        # YCB task objects stay at their authored size.
        "011_banana",
        "apple",
        "orange",
        "pear",
        "003_cracker_box",
        "004_sugar_box",
        "008_pudding_box",
        "025_mug",
        "a_cup",
        "a_cups",
        "b_cups",
        "c_cups",
        "tennis_ball",
        "golf_ball",
        "mini_soccer_ball",
        "024_bowl",
        "rubiks_cube",
        "bucket",
        "plate",
        "tray",
    }
)

_SCALED_TASK_OBJECT_ROOT_BODIES = frozenset(
    {
        # bimanual_assembly
        "industreal_round_peg_8mm",
        "industreal_tray_insert_round_peg_8mm",
        # bimanual_hanoi
        "hanoi_base",
        "hanoi_disk_large",
        "hanoi_disk_medium",
        "hanoi_disk_small",
        # bimanual_photograph
        "link_19",
        # pick_bucket
        "bucket",
        "boxed_food_0",
        # YCB/HandsOn mixed A-series tasks. Actual YCB assets are excluded below.
        "011_banana",
        "apple",
        "orange",
        "pear",
        "003_cracker_box",
        "004_sugar_box",
        "008_pudding_box",
        "025_mug",
        "a_cup",
        "a_cups",
        "b_cups",
        "c_cups",
        "plate",
        "tray",
        "tennis_ball",
        "golf_ball",
        "mini_soccer_ball",
        "024_bowl",
        "rubiks_cube",
        # fold_glasses
        "glass",
        "open_box",
        # hammer_nail
        "hammer_body",
        "nail",
        "wood",
        # bimanual_unlock_ipad
        "ipad_body",
        "ipad_stand",
        # bimanual_microwave_cook
        "hot_dog",
        "microwave_object",
        # click_mouse
        "link_3",
        "display_link_1",
        "mousepad",
        # water_plant
        "plant",
        "link_2",
        # pinch_tongs
        "link_1",
    }
)

_TASK_OBJECT_ROOT_BODIES = (
    _SCALED_TASK_OBJECT_ROOT_BODIES - _UNSCALED_YCB_OBJECT_ROOT_BODIES
)

_TABLE_ATTACHED_OBJECT_GEOMS = frozenset(
    {
        "photograph_logo_disk",
    }
)

_APPLIED_MODEL_SCALES: weakref.WeakKeyDictionary[mujoco.MjModel, float] = (
    weakref.WeakKeyDictionary()
)


def task_object_scale_from_env() -> float:
    """Return the task-object scale requested by the environment."""
    value = os.environ.get("HANDSON_OBJECT_SCALE", "").strip()
    if not value:
        return DEFAULT_TASK_OBJECT_SCALE
    try:
        scale = float(value)
    except ValueError:
        return DEFAULT_TASK_OBJECT_SCALE
    return scale if scale > 0.0 else DEFAULT_TASK_OBJECT_SCALE


def apply_task_object_scale(
    model: mujoco.MjModel,
    scale: float,
) -> None:
    """Set known task objects in a compiled model to the requested scale.

    The selected roots are only task objects and task props. Robot arms, hands,
    the table, walls, mocap targets, and wrist cameras are intentionally left at
    their original size.
    """
    scale = float(scale)
    previous_scale = _APPLIED_MODEL_SCALES.get(model, 1.0)
    if np.isclose(scale, previous_scale):
        return
    relative_scale = scale / previous_scale

    root_body_ids = _top_level_named_body_ids(model, _TASK_OBJECT_ROOT_BODIES)
    object_body_ids = _descendant_body_ids(model, root_body_ids)
    object_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in object_body_ids
    ]
    table_attached_geom_ids = _named_geom_ids(model, _TABLE_ATTACHED_OBJECT_GEOMS)

    if not object_body_ids and not table_attached_geom_ids:
        return

    root_body_id_set = set(root_body_ids)
    for body_id in object_body_ids:
        if body_id not in root_body_id_set:
            model.body_pos[body_id] *= relative_scale

    for geom_id in object_geom_ids:
        model.geom_pos[geom_id] *= relative_scale
        model.geom_size[geom_id] *= relative_scale
        model.geom_margin[geom_id] *= relative_scale
        model.geom_gap[geom_id] *= relative_scale

    for geom_id in table_attached_geom_ids:
        model.geom_size[geom_id] *= relative_scale
        model.geom_margin[geom_id] *= relative_scale
        model.geom_gap[geom_id] *= relative_scale

    for site_id in range(model.nsite):
        if int(model.site_bodyid[site_id]) in object_body_ids:
            model.site_pos[site_id] *= relative_scale
            model.site_size[site_id] *= relative_scale

    for joint_id in range(model.njnt):
        if int(model.jnt_bodyid[joint_id]) not in object_body_ids:
            continue
        model.jnt_pos[joint_id] *= relative_scale
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_SLIDE):
            model.jnt_range[joint_id] *= relative_scale

    for camera_id in range(model.ncam):
        if int(model.cam_bodyid[camera_id]) in object_body_ids:
            model.cam_pos[camera_id] *= relative_scale

    for light_id in range(model.nlight):
        if int(model.light_bodyid[light_id]) in object_body_ids:
            model.light_pos[light_id] *= relative_scale

    mesh_ids = _mesh_ids_for_geoms(model, object_geom_ids + table_attached_geom_ids)
    for mesh_id in mesh_ids:
        vert_adr = int(model.mesh_vertadr[mesh_id])
        vert_num = int(model.mesh_vertnum[mesh_id])
        model.mesh_vert[vert_adr : vert_adr + vert_num] *= relative_scale

    # Recompute constants such as bounding radii after editing compiled arrays.
    scratch_data = mujoco.MjData(model)
    mujoco.mj_setConst(model, scratch_data)
    _APPLIED_MODEL_SCALES[model] = scale


def _top_level_named_body_ids(
    model: mujoco.MjModel,
    names: frozenset[str],
) -> tuple[int, ...]:
    body_ids = {
        int(body_id)
        for name in names
        if (body_id := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)) >= 0
    }
    top_level_ids = []
    for body_id in sorted(body_ids):
        parent_id = int(model.body_parentid[body_id])
        while parent_id > 0:
            if parent_id in body_ids:
                break
            parent_id = int(model.body_parentid[parent_id])
        else:
            top_level_ids.append(body_id)
    return tuple(top_level_ids)


def _descendant_body_ids(
    model: mujoco.MjModel,
    root_body_ids: tuple[int, ...],
) -> set[int]:
    root_body_id_set = set(root_body_ids)
    body_ids: set[int] = set()
    for body_id in range(1, model.nbody):
        ancestor_id = body_id
        while ancestor_id > 0:
            if ancestor_id in root_body_id_set:
                body_ids.add(body_id)
                break
            ancestor_id = int(model.body_parentid[ancestor_id])
    return body_ids


def _named_geom_ids(
    model: mujoco.MjModel,
    names: frozenset[str],
) -> list[int]:
    return [
        int(geom_id)
        for name in names
        if (geom_id := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0
    ]


def _mesh_ids_for_geoms(
    model: mujoco.MjModel,
    geom_ids: list[int],
) -> tuple[int, ...]:
    mesh_ids = set()
    for geom_id in geom_ids:
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id >= 0:
            mesh_ids.add(mesh_id)
    return tuple(sorted(mesh_ids))
