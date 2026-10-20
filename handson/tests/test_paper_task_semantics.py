"""Regression cases for the paper task predicates lost during packaging."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from handson.evaluation import settings
from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
from handson.sim.isaaclab.xy2_contract import SUCCESS_HOLD_STEPS, XY2Progress
from handson.tasks.eval_gen01_cover_lunchbox import geometry


def test_lunchbox_coverage_rejects_a_shifted_lid():
    box = np.array([0., 0., 1., 1., 0., 0., 0.])
    lid = box.copy()
    lid[2] += geometry.GOAL_LOCAL_Z
    assert geometry._coverage_metrics(lid, box)[1]
    lid[0] += .2
    assert not geometry._coverage_metrics(lid, box)[1]


def test_lunchbox_native_dispatch_uses_its_own_geometry():
    env = SimpleNamespace(task_name=geometry.TASK_NAME)
    for success in (False, True):
        with patch.object(geometry, 'success_metrics', return_value={'b05_success_now': success}):
            assert HandsOnABCIsaacLabEnv._compute_success_now(env) is success


def test_mouse_click_is_retained_after_release_but_not_after_leaving_pad():
    laptop = dict(finite=True, failure=False, hinge_target_ok=True,
                  base_stable=True, velocity_stable=True)
    mouse = dict(c05_failed_early_click=False, c05_mouse_on_pad=True,
                 c05_left_clicked=False)
    progress = XY2Progress(mouse_hold_steps=3)
    for frame in range(SUCCESS_HOLD_STEPS):
        assert not progress.update(frame=frame, laptop=laptop, mouse_xy=(0., 0.),
                                   mouse=mouse, buttons_neutral=True)
    assert progress.phase == 1
    mouse['c05_left_clicked'] = True
    assert not progress.update(frame=30, laptop=laptop, mouse_xy=(0., 0.),
                               mouse=mouse, buttons_neutral=False)
    mouse['c05_left_clicked'] = False
    assert not progress.update(frame=31, laptop=laptop, mouse_xy=(0., 0.),
                               mouse=mouse, buttons_neutral=True)
    assert progress.update(frame=32, laptop=laptop, mouse_xy=(0., 0.),
                           mouse=mouse, buttons_neutral=True)
    mouse['c05_mouse_on_pad'] = False
    assert not progress.update(frame=33, laptop=laptop, mouse_xy=(.2, 0.),
                               mouse=mouse, buttons_neutral=True)


def test_approved_drawer_setting_selects_matching_native_environment(tmp_path, monkeypatch):
    from handson.sim.isaaclab import b_eval_setting
    from handson.tasks.t15_open_drawers.config import TaskConfig
    task = 't15_open_drawers'
    expected = {'small_upper_d6': 'b320_bc017_small_d6_env',
                'large_upper_contact': 'b320_bc017_large_upper_contact_env',
                'large_six_contact': 'b320_bc017_env'}
    import json
    for setting, module in expected.items():
        selection = settings.select_setting(0, task, 0, forced_setting=setting)
        path = tmp_path / 'selection.json'
        path.write_text(json.dumps(selection))
        monkeypatch.setenv('HANDSON_B_EVAL_SELECTION', str(path))
        b_eval_setting.selected.cache_clear()
        wrapped = TaskConfig().get_environment(policy_mode=True, launch_isaac=False)
        try:
            assert type(wrapped.unwrapped).__module__.endswith('.' + module)
        finally:
            wrapped.close()
    b_eval_setting.selected.cache_clear()


def test_all_public_task_scene_xml_references_exist():
    import ast
    from handson.tasks import CONFIG_MAPPING
    import handson
    package = Path(handson.__file__).parent
    # Inspect the native registry without loading Kit's scene importer on CPU.
    tree = ast.parse((package / 'sim/isaaclab/task_preview.py').read_text())
    registry = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == '_TASK_SCENES'
                            for t in n.targets))
    scenes = {key.value: value.value for key, value in zip(registry.keys, registry.values)
              if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)}
    root = package / 'sim/envs/xmls'
    missing = {task: scene for task, scene in scenes.items()
               if task in CONFIG_MAPPING and not (root / scene).is_file()}
    assert missing == {}


def test_core_box_retains_paper_centering_predicate():
    # Core03 uses the recorded centering tolerance; Transfer lids have their
    # own coverage/contact predicates and must not replace this contract.
    from handson.sim.isaaclab.core_box_env import (
        HandsOnABCIsaacLabEnv as PaperBoxEnv, _LID_GOAL_LOCAL_Z,
    )
    lid = np.array([0., 0., 1., 1., 0., 0., 0.])
    local = np.array([.04, 0., _LID_GOAL_LOCAL_Z["t03_cover_box"]])
    env = SimpleNamespace(
        task_name="t03_cover_box", _target_object_name="Box_lid",
        _lid_start_pose=lid.copy(), _lid_was_moved=True,
        _pose_for_role=lambda role: lid,
        _target_local_to_receptacle=lambda: (local, None, None),
        _z_axis_alignment=lambda *args: 1.,
        _x_axis_abs_alignment=lambda *args: 1.,
        _physics_body_velocity=lambda name: np.zeros(6),
    )
    assert PaperBoxEnv._lid_fit_success(env)
    local[0] = .06
    assert not PaperBoxEnv._lid_fit_success(env)


def test_all_public_policy_factories_expose_eef52_without_starting_isaac():
    """Construct all real task classes, including inherited family factories."""
    from handson.tasks import CONFIG_MAPPING
    from handson.tasks.config import TaskConfigBase
    from handson.evaluation.native import check_task
    for task, config_type in CONFIG_MAPPING.items():
        assert config_type.get_environment is TaskConfigBase.get_environment, task
        for image_obs in (False, True):
            env = config_type().get_environment(policy_mode=True, launch_isaac=False,
                                                image_obs=image_obs, headless=True)
            try:
                assert env.action_space.shape == (52,), task
                assert env.observation_space['state'].shape == (52,), task
                expected = {'state', 'front', 'wrist_left', 'wrist_right'} if image_obs else {'state'}
                assert set(env.observation_space.spaces) == expected, task
            finally:
                env.close()
        assert check_task(task)['state_dim'] == 52
