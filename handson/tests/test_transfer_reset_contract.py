"""Transfer reset stream must remain compatible with the paper runtime."""
import json
from pathlib import Path

import numpy as np
import pytest

from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
from handson.sim.isaaclab.d_collection_env import HandsOnDIsaacLabEnv
from handson.sim.isaaclab.lw_native_articulation_contract import GEN3_SPEC, sample_reset


def test_trash_bin_reset_matches_archived_paper_sampler():
    oracle = json.loads((Path(__file__).parent / "fixtures/transfer_gen02_reset.json").read_text())
    for row in oracle["rows"]:
        expected = dict(row["sample"], task_id=GEN3_SPEC.task_id, task_name=GEN3_SPEC.task_name)
        assert sample_reset(GEN3_SPEC, seed=row["seed"], randomize=True).as_dict() == expected


@pytest.mark.parametrize("seed", [0, 1, 1729530836])
def test_gen4_parent_preserves_task_layout_random_stream(monkeypatch, seed):
    calls = []

    def base_reset(self, *, clear_manual_request):
        calls.append(clear_manual_request)
        # Model the base sampler's legitimate draws; only the D parent must
        # leave the following task-specific samples unchanged.
        self._random.uniform(size=3)

    monkeypatch.setattr(HandsOnABCIsaacLabEnv, "_sample_reset_state", base_reset)
    env = object.__new__(HandsOnDIsaacLabEnv)
    env.task_name = "eval_gen04_pass_glue_stick"
    env.randomize = True
    env._random = np.random.RandomState(seed)
    env.delta_h = np.ones(3)
    env._d_last_metrics = {"stale": 1}
    env._d_prev_object_poses = {"stale": 1}
    expected = np.random.RandomState(seed)
    expected.uniform(size=3)

    env._sample_reset_state(clear_manual_request=True)

    assert calls == [True]
    np.testing.assert_array_equal(env.delta_h, 0)
    assert env._d_last_metrics == env._d_prev_object_poses == {}
    np.testing.assert_array_equal(env._random.uniform(size=32), expected.uniform(size=32))
