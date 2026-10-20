"""Exact Isaac 5.1 interpreter checks with launch_isaac=False; no GPU or Kit."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isaaclab-root", type=Path, required=True)
    args = parser.parse_args()
    version = importlib.metadata.version("isaacsim")
    assert version.startswith("5.1."), version
    lab_version = (args.isaaclab_root / "VERSION").read_text().strip()
    assert lab_version == "2.3.2", lab_version
    from handson.tasks.eval_comb02_open_laptop_and_click_mouse.config import TaskConfig
    samples = []
    env = TaskConfig().get_native_environment(
        policy_mode=True, sim_backend="isaacsim", render_mode="rgb_array",
        launch_isaac=False, image_obs=False,
    )
    try:
        for angle in (30, 60, 90, 120):
            for seed in (0, 1, 7, 41, 42):
                first, _ = env.reset(seed=seed, target_label=str(angle))
                layout = env.unwrapped._xy2_layout.to_dict()
                second, _ = env.reset(seed=seed, target_label=str(angle))
                assert layout == env.unwrapped._xy2_layout.to_dict()
                assert env.observation_space.contains(first)
                assert env.observation_space.contains(second)
                assert np.array_equal(first["state"], second["state"])
                assert env.action_space.shape == (54,)
                assert first["state"].shape == (76,)
                assert env.unwrapped._physics_dt == 1 / 120
                assert env.unwrapped._control_dt == 1 / 30
                samples.append(layout)
    finally:
        env.close()
    assert not any(name == "omni.kit.app" or name.startswith("omni.kit.app.") for name in sys.modules)
    result = {
        "pass": True, "scope": "exact runtime Python; constructor/reset/interface only",
        "gpu_initialized": False, "isaac_physx_verified": False,
        "python": sys.version, "executable": sys.executable,
        "isaacsim": version, "isaaclab": lab_version,
        "action_shape": [54], "policy_state_shape": [76], "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}))


if __name__ == "__main__":
    main()
