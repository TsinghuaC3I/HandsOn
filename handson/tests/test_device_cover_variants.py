"""Prevent the two paper tasks from being combined into one hybrid protocol."""
from types import SimpleNamespace

import pytest

from handson.evaluation.paper_variants import load, sample_language
from handson.tasks.registry import TASKS

TASK = "t06_fit_device_cover"
VARIANT = "stabilize_body"
VALUES = {"mode": "fixed_instruction_or_observed_visual_target"}


def test_variant_keeps_controller_origin_trace_and_camera_source_together():
    default = TASKS[TASK].native_contract()
    legacy = TASKS[TASK].native_contract(VARIANT)
    assert legacy["origin_implementation"]["source_sha256"].startswith("1b733a0f70d8")
    assert legacy["trace_implementation"]["source_sha256"].startswith("30896f62f36f")
    assert legacy["constructor"]["target_object"] == "device_cover"
    spec = TASKS[TASK].evaluation_adapter(VARIANT).spec
    assert spec.camera_environment_module == "core_device_cover_legacy_abc"
    assert spec.trace_environment_module == "core_device_cover_legacy_base"
    assert TASKS[TASK].native_contract() == default
    assert TASKS[TASK].evaluation_adapter().spec == TASKS[TASK]


@pytest.mark.parametrize("task,variant,backend", [
    ("t34_roll_ball_through_gate", VARIANT, "physx"),
    (TASK, "unknown", "physx"), (TASK, VARIANT, "newton"),
])
def test_no_silent_fallback_for_unsupported_cohort(task, variant, backend):
    with pytest.raises(ValueError, match="Unsupported paper variant"):
        load(task, variant, backend)


def test_old_task_uses_recovered_language_split_and_accepts_seven_paper_prompts():
    row = load(TASK, VARIANT)
    goal = row["language_goal"]
    assert len(goal["train"]) == 32 and len(goal["heldout"]) == 8
    assert not set(goal["train"]) & set(goal["heldout"])
    allowed = {goal["canonical"], *goal["train"]}
    assert len(row["historical_trials"]) == row["paper_trials"] == 7
    assert all(trial["prompt"] in allowed for trial in row["historical_trials"])
    samples = [sample_language(TASK, VARIANT, VALUES, seed) for seed in range(100)]
    assert samples == [sample_language(TASK, VARIANT, VALUES, seed) for seed in range(100)]
    assert all(s["prompt"] in allowed and not s["historical_draw_reproduced"] for s in samples)
    from handson.tasks.language import allowed_prompts
    assert not allowed & allowed_prompts(TASK)


def test_goal_binding_rejects_environment_language_mismatch():
    from handson.evaluation.goals import bind
    env = SimpleNamespace(env_step=0, paper_variant=VARIANT)
    with pytest.raises(Exception, match="source cohorts disagree"):
        bind(env, TASK, 0)


def test_policy_transport_carries_cohort_and_server_rejects_wrong_language():
    import io
    import json
    import numpy as np
    from handson.evaluation.policy import HTTPTransport
    from handson.policies.profiles import PROFILES
    from handson.policies.server import predict_request
    from handson.tasks.language import allowed_prompts
    profile = PROFILES["pi05"]
    seen = []
    policy = SimpleNamespace(predict=lambda state, images, prompt, seed:
                             seen.append(prompt) or np.zeros((profile.model_horizon, 52)))
    config = dict(endpoint="http://local.test", task=TASK, paper_variant=VARIANT,
                  prompt=load(TASK, VARIANT)["language_goal"]["canonical"])
    transport = HTTPTransport(config)
    def open_request(request, timeout):
        result = predict_request(policy, profile, "pi05", request.data)
        assert result["paper_variant"] == VARIANT
        return io.BytesIO(json.dumps(result).encode())
    transport.opener = SimpleNamespace(open=open_request)
    images = {k:np.zeros((20, 30, 3), np.uint8) for k in ["main", "wrist_left", "wrist_right"]}
    assert np.asarray(transport.predict(np.zeros(52), images, 0, 1)).shape == (profile.action_horizon, 52)
    assert seen == [config["prompt"]]
    config["prompt"] = next(iter(allowed_prompts(TASK)))
    with pytest.raises(ValueError, match="language bank"):
        transport.predict(np.zeros(52), images, 1, 1)
    assert len(seen) == 1


def test_both_native_factories_expose_the_expected_physics_and_distinct_modules():
    from handson.evaluation.native import check_task
    original = check_task(TASK)
    legacy = check_task(TASK, paper_variant=VARIANT)
    assert original["environment_module"].endswith(".t06_fit_device_cover_env")
    assert legacy["environment_module"].endswith(".core_device_cover_legacy_env")
    assert original["physics_dt"] == legacy["physics_dt"] == 1 / 120
    assert original["n_substeps"] == legacy["n_substeps"] == 4
    from handson.sim.isaaclab import core_device_cover_legacy_scene as old
    from handson.sim.isaaclab import t06_fit_device_cover_scene as new
    for name, old_mass, new_mass in [("device_body", .380, .450), ("device_cover", .120, .160)]:
        assert next(s.mass_kg for s in old.G02_ASSET_SPECS if s.name == name) == old_mass
        assert next(s.mass_kg for s in new.G02_ASSET_SPECS if s.name == name) == new_mass
