from pathlib import Path
import os
import subprocess
import sys

import pytest

from handson.runtime.task_camera_profile import (
    BC007_TASK_NAME, WRIST_OUTPUT_SIZE_ENV, configure_collection_task_camera_profile,
)


@pytest.mark.parametrize("args", [
    [f"--exp_name={BC007_TASK_NAME}"], ["--exp_name", BC007_TASK_NAME],
])
def test_bc007_overrides_fleet_default_before_camera_import(args):
    env = {WRIST_OUTPUT_SIZE_ENV: "848x480"}
    configure_collection_task_camera_profile(args, env)
    assert env[WRIST_OUTPUT_SIZE_ENV] == "640x480"


@pytest.mark.parametrize("args", [[], ["--exp_name=c09_staple_paper_sheets"],
    ["--", f"--exp_name={BC007_TASK_NAME}"], ["--exp_name"],
])
def test_other_tasks_and_positional_arguments_do_not_change_camera_profile(args):
    env = {WRIST_OUTPUT_SIZE_ENV: "848x480"}
    configure_collection_task_camera_profile(args, env)
    assert env[WRIST_OUTPUT_SIZE_ENV] == "848x480"


def test_real_recorder_import_selects_profile_before_scene_helpers():
    root = Path(__file__).resolve().parents[2]
    code = f'''
import runpy, sys
sys.argv = ["record_demos_zarr.py", "--exp_name={BC007_TASK_NAME}"]
runpy.run_path({str(root / "scripts/record_demos_zarr.py")!r}, run_name="camera_import_test")
from handson.sim.isaaclab.recording_cameras import WRIST_RECORDING_CAMERA_OUTPUT_SIZE
assert WRIST_RECORDING_CAMERA_OUTPUT_SIZE == (640, 480), WRIST_RECORDING_CAMERA_OUTPUT_SIZE
'''
    env = dict(os.environ)
    env[WRIST_OUTPUT_SIZE_ENV] = "848x480"
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
