"""Historical-input regressions, including different model quotas and old DP."""
from collections import Counter
import contextlib
import io
import json
import os
from unittest.mock import patch

import pytest

from handson.evaluation import cli, paper_trials, task_contracts


def test_model_specific_ball_quotas_and_language_draws_are_retained():
    task = 't24_place_ball_in_bowl'
    expected = {'pi05': {'golf ball':18, 'mini soccer ball':40, 'tennis ball':42},
                'openwam': {'golf ball':28, 'mini soccer ball':36, 'tennis ball':36}}
    for model, counts in expected.items():
        rows = [paper_trials.load(model, task, i) for i in range(100)]
        assert Counter(r['goal']['ball'] for r in rows) == counts
    assert paper_trials.load('openwam', task, 0)['language']['language_seed'] == 20260918
    assert paper_trials.load('pi05', task, 0)['language']['language_seed'] == 42


def test_rejected_reset_keeps_its_slot_goal_and_recorded_accepted_candidate():
    row = paper_trials.load('dm05', 't26_place_fruit_in_bucket', 1)
    assert row['index'] == 1 and row['candidate_seed_index'] == 101
    assert row['seed'] == row['reset_seed'] == 488775594
    assert row['reset_rejections'][0]['seed'] == 977439540
    assert row['goal'] == {'fruit':'apple'}
    assert row['prompt'] == 'Pick the apple and place it into the bucket.'


def test_old_dp_binding_is_not_relabelled_as_a_new_language_draw():
    row = paper_trials.load('dp', 't27_place_largest_fitting_ball', 0)
    assert row['language'] is None
    assert row['goal_route'] == 'legacy_task_binding'
    assert row['goal'] == {'requested_cup':'C'}


def test_historical_source_variants_remain_explicit():
    rows = [paper_trials.load(model, 't06_fit_device_cover', i)
            for model in paper_trials.manifest()['models'] for i in range(100)]
    assert sum(r['paper_variant'] == 'stabilize_body' for r in rows) == 7
    rows = [paper_trials.load('dp', 't27_place_largest_fitting_ball', i) for i in range(100)]
    assert Counter(r['constructor_overrides']['reset_clearance_scope'] for r in rows) == {
        'hand_bodies':66, 'arm_bodies':34}


def test_settings_use_the_recorded_index_and_complete_profile():
    row = paper_trials.load('pi05', 't38_set_dual_knobs', 0)
    selection = paper_trials.select_setting(row)
    assert selection['setting_id'] == 'knob_pair_53'
    assert selection['seeds']['reset'] == 327930728
    assert selection['profile']['initial_degrees'] == [0, 40]
    assert selection['profile']['target_degrees'] == [60, -60]
    row['setting']['episode_index'] += 1
    with pytest.raises(ValueError, match='Historical setting field differs'):
        paper_trials.select_setting(row)


def test_modified_inputs_fail_and_do_not_poison_cached_records():
    row = paper_trials.load('pi05', 't16_open_laptop', 0)
    row['goal']['target_angle_degrees'] = 120
    with pytest.raises(ValueError, match='modified'):
        paper_trials.validate(row, row['task'], row['seed'], 'physx', None)
    assert paper_trials.load('pi05', row['task'], 0)['goal']['target_angle_degrees'] == 30


def test_inspection_needs_no_assets_or_policy_server():
    argv = ['evaluation', '--task', 't24_place_ball_in_bowl', '--model', 'openwam',
            '--paper-trial', '0', '--inspect-paper-trial']
    out = io.StringIO()
    with patch('sys.argv', argv), contextlib.redirect_stdout(out), patch(
            'handson.robots.assets.require_benchmark_assets', side_effect=AssertionError('GPU/assets used')):
        cli.main()
    assert json.loads(out.getvalue())['goal'] == {'ball':'tennis ball'}


def test_recorded_control_caps_are_cleared_between_trial_contexts():
    task = 't29_place_wine_glass'
    row = paper_trials.load('dm05', task, 60)
    plain = paper_trials.load('dm05', task, 0)
    with patch.dict(os.environ, {}, clear=True):
        paper_trials.configure_environment(row, task_contracts.load(task))
        assert os.environ['HANDSON_PHYSX_MAX_CONTACT_POINTS'] == '4096'
        paper_trials.configure_environment(plain, task_contracts.load(task))
        assert 'HANDSON_PHYSX_MAX_CONTACT_POINTS' not in os.environ
