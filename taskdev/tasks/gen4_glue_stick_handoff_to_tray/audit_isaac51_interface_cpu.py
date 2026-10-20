#!/usr/bin/env python3
"""Run in the selected 5.1 interpreter without launching Kit or CUDA."""
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
from handson.sim.isaaclab.d_collection_env import HandsOnDIsaacLabEnv
from handson.sim.isaaclab.gen4_env import HandsOnGEN4IsaacLabEnv
from handson.tasks.eval_gen04_pass_glue_stick.config import TaskConfig
from handson.tasks.eval_gen04_pass_glue_stick.contract import TASK_NAME

reference = HandsOnDIsaacLabEnv("t21_pass_block", launch_isaac=False, image_obs=False)
env = HandsOnDIsaacLabEnv(TASK_NAME, launch_isaac=False, image_obs=False)
wrapped = TaskConfig().get_environment(policy_mode=True, sim_backend="isaacsim", launch_isaac=False, image_obs=False, render_mode="rgb_array")
try:
    assert isinstance(env, HandsOnGEN4IsaacLabEnv)
    assert env.action_space == reference.action_space
    assert env.observation_space["state"] == reference.observation_space["state"]
    obs, _ = wrapped.reset(seed=17)
    assert obs["state"].shape == (144,)
    assert obs["gen4_handoff_state"].shape == (10,)
    assert not any(m.startswith("omni.kit") for m in sys.modules)
    data = {
        "method": "Actual Isaac Sim 5.1 Python, launch_isaac=False, CUDA masked, no Kit started",
        "python": sys.executable, "isaacsim": md.version("isaacsim"),
        "isaaclab_release": Path("/opt/handson/IsaacLab-v2.3.2/VERSION").read_text().strip(),
        "isaaclab_source": json.loads(md.distribution("isaaclab").read_text("direct_url.json")),
        "task": env._task_info(), "action_shape": env.action_space.shape,
        "policy_state_shape": obs["state"].shape,
        "record_only_hidden_state_shape": obs["gen4_handoff_state"].shape,
        "d03_action_and_eight_state_keys_equal": True, "gpu_started": False,
    }
    assert data["isaacsim"] == "5.1.0.0" and data["isaaclab_release"] == "2.3.2"
    print(json.dumps(data, indent=2, default=str))
finally:
    env.close(); reference.close(); wrapped.close()
