from __future__ import annotations

import numpy as np
import pytest

from handson.sim.isaaclab.t06_fit_device_cover_env import (
    HandsOnG02RepairCloseDeviceIsaacLabEnv,
    _G02_HAND_CONTACT_FORCE_THRESHOLD,
    _G02_OBJECT_CONTACT_FORCE_THRESHOLD,
)
from handson.sim.isaaclab.t06_fit_device_cover_scene import (
    G02_ASSET_SPECS,
    G02_DEVICE_BODY_HOME,
    G02_DEVICE_COVER_HOME,
    G02_STABLE_CONTROL_FRAMES,
    validate_g02_asset_files,
)
from handson.sim.isaaclab.recording_origin import (
    stable_recording_origin_kwargs_for_task,
)
from handson.tasks.t06_fit_device_cover.config import TaskConfig


EXPECTED_STATE_SHAPES = {
    "tcp_pose": (14,),
    "gripper_pose": (32,),
    "device_body_pose": (7,),
    "device_cover_pose": (7,),
    "cover_to_body": (7,),
    "alignment_metrics": (8,),
    "contact_metrics": (19,),
    "table_delta_height": (1,),
}


def _interface_env(*, randomize: bool = True):
    return HandsOnG02RepairCloseDeviceIsaacLabEnv(
        launch_isaac=False,
        image_obs=False,
        randomize=randomize,
    )


def test_g02_assets_use_reachable_hand_owned_pickup_zones() -> None:
    source_paths = validate_g02_asset_files()
    assert {path.name for path in source_paths} == {
        "workbench.usd",
        "device_body.usd",
        "device_cover.usd",
    }

    specs = {spec.name: spec for spec in G02_ASSET_SPECS}
    np.testing.assert_allclose(G02_DEVICE_BODY_HOME, (-0.10, 0.18, 0.852))
    np.testing.assert_allclose(G02_DEVICE_COVER_HOME, (-0.10, -0.18, 0.877))
    assert specs["device_body"].mass_kg == pytest.approx(0.450)
    assert specs["device_cover"].mass_kg == pytest.approx(0.160)
    for name in ("device_body", "device_cover"):
        assert specs[name].static_friction == pytest.approx(1.6)
        assert specs[name].dynamic_friction == pytest.approx(1.3)


def test_g02_randomized_resets_stay_close_and_on_the_correct_side() -> None:
    env = _interface_env()
    for seed in range(32):
        env._random = np.random.default_rng(seed)
        sampled = env._task_specific_reset_xy_positions()
        assert sampled is not None
        body_xy = sampled["device_body"]
        cover_xy = sampled["device_cover"]
        assert -0.44 <= body_xy[0] <= -0.04 and 0.08 <= body_xy[1] <= 0.30
        assert -0.44 <= cover_xy[0] <= -0.04 and -0.30 <= cover_xy[1] <= -0.08
        assert np.linalg.norm(body_xy - cover_xy) > 0.20
    env.close()


def test_g02_interface_and_recording_origin_contract() -> None:
    env = _interface_env()
    observation, info = env.reset(seed=17)
    assert env.action_space.shape == (54,)
    assert env.observation_space.contains(observation)
    assert {
        key: value.shape for key, value in observation["state"].items()
    } == EXPECTED_STATE_SHAPES
    assert info["task_id"] == "G02"
    assert info["g02_physical_replay_certified"] is False

    profile = stable_recording_origin_kwargs_for_task(
        "t06_fit_device_cover"
    )
    assert profile["post_settle_sleeping"] is False
    assert profile["stability_metric"] == "velocity_or_static_pose"
    assert profile["stability_sample_seconds"] == 0.0
    env.close()


def test_g02_success_needs_both_hand_roles_contact_and_eight_frames() -> None:
    env = _interface_env(randomize=False)
    env.reset(seed=0)
    body = env._fallback_g02_pose("device_body")
    cover = body.copy()
    cover[2] += 0.050
    env._poses["device_body"] = body
    env._poses["device_cover"] = cover
    env._g02_body_reset_pose = body.copy()
    env._g02_contact_available = {
        "body_cover": True,
        "left_body": True,
        "right_cover": True,
    }
    env._g02_contact_forces = {
        "body_cover": _G02_OBJECT_CONTACT_FORCE_THRESHOLD + 0.01,
        "left_body": _G02_HAND_CONTACT_FORCE_THRESHOLD + 0.01,
        "right_cover": _G02_HAND_CONTACT_FORCE_THRESHOLD + 0.01,
    }
    env._g02_dual_transport_completed = True
    env._g02_body_cover_penetration_available = True
    env._g02_body_cover_penetration_m = 0.0
    env._physics_body_velocity = lambda _: np.zeros(6)
    # Interface mode has no live PhysX views; retain the injected physical
    # truth while exercising the actual success predicate and hold counter.
    env._read_g02_contacts = lambda *, update_latches: (
        setattr(env, "_g02_left_body_contact_seen", True),
        setattr(env, "_g02_right_cover_contact_seen", True),
    )
    results = [env._compute_success() for _ in range(G02_STABLE_CONTROL_FRAMES)]
    assert results == [False] * (G02_STABLE_CONTROL_FRAMES - 1) + [True]
    # The final fit cannot erase missing transport, invalid depth or moving-body truth.
    for attribute, invalid in [("_g02_dual_transport_completed", False),
                               ("_g02_body_cover_penetration_available", False),
                               ("_g02_body_cover_penetration_m", 0.02)]:
        old = getattr(env, attribute)
        setattr(env, attribute, invalid)
        assert not env._compute_success_now()
        setattr(env, attribute, old)
    env._physics_body_velocity = lambda _: None
    assert not env._compute_success_now()
    env.close()


def test_g02_config_rejects_non_isaac_backends() -> None:
    with pytest.raises(ValueError, match="requires sim_backend='isaacsim'"):
        TaskConfig().get_environment(sim_backend="mujoco")
