"""Renaming a task must preserve its control and reset path."""
from handson.data.recording_origin_profile import is_abc_style_collection_task
from handson.runtime.collection_support import (
    uses_awake_contact_collection_origin, uses_wrist_camera_rerender_pipeline,
)
from handson.sim.isaaclab.abc_collection_env import _task_supports_deferred_arm_render
from handson.sim.isaaclab.bimanual_assembly_env import BimanualAssemblyIsaacLabEnv
from types import SimpleNamespace


def test_public_names_preserve_ordinary_and_benchmark_families():
    for name in ('t26_place_fruit_in_bucket', 't04_cover_pitcher', 't50_water_plant',
                 't10_lift_carton', 't16_open_laptop'):
        assert is_abc_style_collection_task(name)
        assert uses_awake_contact_collection_origin(name)
    for name in ('t26_place_fruit_in_bucket', 't34_roll_ball_through_gate', 't13_load_cans'):
        assert _task_supports_deferred_arm_render(name)
    assert not _task_supports_deferred_arm_render('t27_place_largest_fitting_ball')
    assert uses_wrist_camera_rerender_pipeline('t26_place_fruit_in_bucket')
    assert not uses_wrist_camera_rerender_pipeline('t16_open_laptop')
    assert BimanualAssemblyIsaacLabEnv._is_b320_collection_task(
        SimpleNamespace(task_name='t16_open_laptop'))
    assert not BimanualAssemblyIsaacLabEnv._is_b320_collection_task(
        SimpleNamespace(task_name='t26_place_fruit_in_bucket'))
