from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import yaml

from handson.runtime.collection_support import (
    ISAACSIM_COLLECTION_UNVERIFIED_TASKS,
    ISAACSIM_FORMAL_COLLECTION_TASKS,
    ISAACSIM_NONSTANDARD_DEFERRED_RENDER_TASKS,
)
from handson.tasks import b320_common
from handson.tasks.t15_open_drawers.config import TaskConfig


_CANONICAL_PROMPT = (
    "Use each assigned hand to pull the highlighted drawer on its side beyond "
    "60 percent travel; each side independently selects one of three drawers "
    "per episode, and both thresholds must be crossed within 1.2 seconds."
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_bc017_config_declares_dedicated_runtime_and_recording_keys() -> None:
    config = TaskConfig()
    yaml_path = (
        _repository_root()
        / "configs"
        / "multi_task"
        / "t15_open_drawers.yaml"
    )
    parsed = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    assert parsed["env_name"] == config.task_name
    assert parsed["prompt"] == config.task_prompt
    assert config.task_prompt == _CANONICAL_PROMPT
    assert parsed["b320_runtime"] == "bc017"
    assert config.b320_runtime == "bc017"
    assert config.b320_proprio_keys[-5:] == [
        "bc017_drawer_state",
        "bc017_contact_state",
        "bc017_stage_state",
        "bc017_terminal_state",
        "bc017_hidden_state",
    ]
    assert config.task_name not in ISAACSIM_COLLECTION_UNVERIFIED_TASKS
    assert config.task_name in ISAACSIM_FORMAL_COLLECTION_TASKS
    assert config.task_name in ISAACSIM_NONSTANDARD_DEFERRED_RENDER_TASKS


def test_bc017_factory_routes_to_dedicated_env_without_launching_isaac(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeBC017Environment:
        def __init__(self, task_name: str, **kwargs) -> None:
            calls.append({"task_name": task_name, **kwargs})

    fake_module = ModuleType(
        "handson.sim.isaaclab.b320_bc017_env"
    )
    fake_module.HandsOnB320BC017IsaacLabEnv = FakeBC017Environment
    monkeypatch.setitem(
        sys.modules,
        "handson.sim.isaaclab.b320_bc017_env",
        fake_module,
    )
    monkeypatch.setattr(
        b320_common,
        "DualArmViveHandTeleopWrapper",
        lambda env, _teleop: env,
    )
    monkeypatch.setattr(
        b320_common,
        "HandsOnObsAdapter",
        lambda env, proprio_keys: (env, proprio_keys),
    )

    config = TaskConfig()
    env, proprio_keys = config.get_environment(
        render_mode="rgb_array",
        randomize=False,
        launch_isaac=False,
        image_obs=False,
    )

    assert isinstance(env, FakeBC017Environment)
    assert calls == [
        {
            "task_name": "t15_open_drawers",
            "render_mode": "rgb_array",
            "randomize": False,
            "launch_isaac": False,
            "image_obs": False,
            "hz": 30,
            "time_limit": 30.0,
            "hand_type": "wuji1",
            "arm_type": "rokae",
        }
    ]
    assert proprio_keys == config.b320_proprio_keys
