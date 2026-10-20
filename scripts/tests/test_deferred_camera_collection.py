from __future__ import annotations

import pytest

from handson.runtime.collection_support import (
    ISAACSIM_COLLECTION_TASKS,
    ISAACSIM_DEFERRED_CAMERA_RENDER_TASKS,
    ISAACSIM_FORMAL_COLLECTION_TASKS,
)
from handson.tasks.mappings import CONFIG_MAPPING
from scripts.record_demos_zarr import RecordingSpool, _online_camera_capture_scope

# The collection UI is not distributed with this repository; the post-render
# contract lives with it.
collection_ui = pytest.importorskip(
    "collection_ui.server",
    reason="collection_ui is not distributed with this repository",
)
canonical_camera_postrender_contract = collection_ui.canonical_camera_postrender_contract

TASKS = (
    "t35_rotate_propeller",
    "t41_sort_nested_cups",
    "t47_trigger_drill_on_block",
    "t29_place_wine_glass",
    "t33_reposition_marker_grasp",
    "eval_flex01_compress_sponge_into_mug",
)


@pytest.mark.parametrize("task_name", TASKS)
def test_merged_ycb_tasks_inherit_formal_camera_deferral(task_name, monkeypatch):
    assert task_name in ISAACSIM_COLLECTION_TASKS
    assert ISAACSIM_FORMAL_COLLECTION_TASKS == ISAACSIM_COLLECTION_TASKS
    assert ISAACSIM_DEFERRED_CAMERA_RENDER_TASKS == ISAACSIM_FORMAL_COLLECTION_TASKS
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "1")
    monkeypatch.setenv("HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY", "0")
    assert _online_camera_capture_scope(
        task_name, "isaacsim", deferred_arm_env_value="1",
        deferred_camera_env_value="1", live_main_env_value="0",
    ) == "all_canonical_postrender"
    env = CONFIG_MAPPING[task_name]().get_native_environment(
        policy_mode=True, sim_backend="isaacsim", launch_isaac=False, image_obs=True,
    )
    try:
        assert env.unwrapped._deferred_camera_render is True
        # The scene builder must return before creating any recording sensor.
        assert env.unwrapped._init_recording_cameras() is None
    finally:
        env.close()


def test_state_only_spool_needs_no_camera_products(tmp_path):
    spool = RecordingSpool(tmp_path, video_fps=30, encode_videos=False)
    try:
        spool.append_observation({"state": {}})
        assert spool.frame_count == 1
        assert not list(spool.root.rglob("*.mp4"))
    finally:
        spool.cleanup()


def test_mismatched_camera_declaration_is_rejected():
    declaration = {
        "online_camera_capture_scope": "all_canonical_postrender",
        "canonical_camera_postrender_required": True,
    }
    required, errors = canonical_camera_postrender_contract(declaration, declaration)
    assert required and not errors
    mismatched = {**declaration, "canonical_camera_postrender_required": False}
    _, errors = canonical_camera_postrender_contract(declaration, mismatched)
    assert errors
