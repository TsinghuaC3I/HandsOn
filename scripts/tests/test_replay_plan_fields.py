import os
from pathlib import Path
import subprocess
import sys


def test_replayed_episode_preserves_plan_recording_fields() -> None:
    # The recorder and replay CLIs intentionally share flag names, so import
    # the replay module in an isolated process just as the real CLI does.
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "handson"), str(root), env.get("PYTHONPATH", "")]
    )
    code = r'''
import numpy as np
from scripts.replay_demos_zarr import _episode_data_from_trajectory

trajectory = []
for step in range(2):
    trajectory.append(
        {
            "actions": np.zeros(27, dtype=np.float64),
            "observations": {
                "state": np.asarray([step, step + 1], dtype=np.float64),
                "plan_arm_qpos": np.arange(7, dtype=np.float64) + step,
                "plan_arm_success": np.asarray([1.0], dtype=np.float64),
                "plan_arm_error": np.asarray([0.001 * (step + 1)], dtype=np.float64),
                "plan_hand_qpos": np.arange(20, dtype=np.float64) + step,
                "front": np.zeros((32, 32, 3), dtype=np.uint8),
            },
        }
    )

episode = _episode_data_from_trajectory(trajectory, data_fps=20.0)
assert episode["state"].shape == (2, 2)
assert episode["plan_arm_qpos"].shape == (2, 7)
assert episode["plan_arm_success"].shape == (2, 1)
assert episode["plan_arm_error"].shape == (2, 1)
assert episode["plan_hand_qpos"].shape == (2, 20)
np.testing.assert_allclose(episode["timestamp"], [0.0, 0.05])
assert "front" not in episode
'''

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_isaac_replay_close_marks_wrapped_environment_for_fast_shutdown() -> None:
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "handson"), str(root), env.get("PYTHONPATH", "")]
    )
    code = r'''
from types import SimpleNamespace

from scripts.replay_demos_zarr import _enable_isaac_fast_shutdown

raw = SimpleNamespace(_sim=object(), _fast_shutdown_on_close=False)
wrapped = SimpleNamespace(unwrapped=raw)
assert _enable_isaac_fast_shutdown(wrapped) is True
assert raw._fast_shutdown_on_close is True
assert _enable_isaac_fast_shutdown(SimpleNamespace()) is False
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_isaac_recorded_state_is_split_by_cached_layout() -> None:
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "handson"), str(root), env.get("PYTHONPATH", "")]
    )
    code = r'''
import numpy as np
from scripts.replay_demos_zarr import _split_initial_state

class Config:
    task_name = "t04_cover_pitcher"

layout = (("tcp_pose", 7), ("object_pose", 7), ("table_delta_height", 1))
state = np.arange(15, dtype=np.float64)
parts = _split_initial_state(Config(), state, layout)
np.testing.assert_array_equal(parts["tcp_pose"], state[:7])
np.testing.assert_array_equal(parts["object_pose"], state[7:14])
np.testing.assert_array_equal(parts["table_delta_height"], state[14:])

legacy_parts = _split_initial_state(Config(), state[:-1], layout)
np.testing.assert_array_equal(legacy_parts["tcp_pose"], state[:7])
np.testing.assert_array_equal(legacy_parts["object_pose"], state[7:14])
assert "table_delta_height" not in legacy_parts

class RawEnv:
    def _compute_observation(self):
        return {"state": {"pose": np.zeros(7), "empty": np.zeros((0, 7)), "height": 0.0}}

class EmptyFieldConfig:
    proprio_keys = ["pose", "empty", "height"]

from scripts.replay_demos_zarr import _initial_state_layout
assert _initial_state_layout(RawEnv(), EmptyFieldConfig()) == (
    ("pose", 7), ("empty", 0), ("height", 1)
)
'''

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_replay_backend_defaults_to_recorded_engine() -> None:
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "handson"), str(root), env.get("PYTHONPATH", "")]
    )
    code = r'''
import json
import tempfile
from pathlib import Path

from scripts.replay_demos_zarr import _sim_backend_for_demo

with tempfile.TemporaryDirectory() as temp:
    demo = Path(temp)
    (demo / "collection_meta.json").write_text(
        json.dumps({"engine_code": "E02", "sim_backend": "isaacsim"}),
        encoding="utf-8",
    )
    assert _sim_backend_for_demo(demo, "auto") == "isaacsim"
    assert _sim_backend_for_demo(demo, "mujoco") == "mujoco"

    (demo / "collection_meta.json").write_text(
        json.dumps({"engine_code": "E01", "sim_backend": "mujoco"}),
        encoding="utf-8",
    )
    (demo / "replay_manifest.json").write_text(
        json.dumps({"recording": {"sim_backend": "isaacsim"}}),
        encoding="utf-8",
    )
    assert _sim_backend_for_demo(demo, "auto") == "isaacsim"

    (demo / "replay_manifest.json").unlink()
    assert _sim_backend_for_demo(demo, "auto") == "mujoco"

    (demo / "collection_meta.json").write_text(
        json.dumps({"engine_code": "E04", "sim_backend": "isaacsim_newton"}),
        encoding="utf-8",
    )
    assert _sim_backend_for_demo(demo, "auto") == "isaacsim_newton"

    (demo / "replay_manifest.json").write_text(
        json.dumps({"recording": {"sim_backend": "isaacsim_newton"}}),
        encoding="utf-8",
    )
    assert _sim_backend_for_demo(demo, "auto") == "isaacsim_newton"
'''

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_force_complete_replay_enables_accurate_wrist_visibility() -> None:
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "handson"), str(root), env.get("PYTHONPATH", "")]
    )
    code = r'''
import os

from scripts.replay_demos_zarr import _apply_isaac_render_overrides

keys = (
    "HANDSON_ISAACLAB_DEFER_ARM_RENDER",
    "HANDSON_ISAACLAB_DEFER_CAMERA_RENDER",
    "HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY",
    "HANDSON_ISAACSIM_ACCURATE_CAMERA_VISIBILITY",
    "HANDSON_ISAACLAB_WRIST_CAMERA_OUTPUT_SIZE",
)
for key in keys:
    os.environ.pop(key, None)

_apply_isaac_render_overrides(
    wrist_width=3840,
    wrist_height=2160,
    force_complete_arm_render=True,
)

assert os.environ["HANDSON_ISAACLAB_DEFER_ARM_RENDER"] == "0"
assert os.environ["HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"] == "0"
assert "HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY" not in os.environ
assert os.environ["HANDSON_ISAACSIM_ACCURATE_CAMERA_VISIBILITY"] == "1"
assert os.environ["HANDSON_ISAACLAB_WRIST_CAMERA_OUTPUT_SIZE"] == "3840x2160"
'''

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
