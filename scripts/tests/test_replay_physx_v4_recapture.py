import os
from pathlib import Path
import subprocess
import sys


def test_physx_v4_recapture_writes_strict_current_next_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "handson"), str(root), env.get("PYTHONPATH", "")]
    )
    code = r'''
import json
import tempfile
from pathlib import Path

import numpy as np
import zarr

from handson.data.episode_store import ZarrEpisodeStore
from handson.data.recording_fields import write_recording_schema
from scripts.replay_demos_zarr import (
    _episode_data_from_trajectory,
    _write_physx_recapture_manifest,
)

field_shapes = {
    "physx_substep_count": [1],
    "physx_substep_robot_joint_position": [2, 2, 3],
    "physx_substep_robot_joint_velocity": [2, 2, 3],
    "physx_substep_robot_joint_target": [2, 2, 3],
    "physx_substep_robot_joint_valid": [2, 2, 3],
    "physx_substep_robot_joint_target_valid": [2, 2, 3],
}
schema = {
    "version": 4,
    "fields": {
        name: {
            "shape": shape,
            "dtype": "int64" if name.endswith("count") else "float64",
        }
        for name, shape in field_shapes.items()
    },
    "physx": {
        "physics_config": {"n_substeps": 2},
        "robot_sides": ["right", "left"],
    },
}

def observation(*, initial=False, value=0.0):
    result = {}
    for name, shape in field_shapes.items():
        if name.endswith("count"):
            result[name] = np.asarray(
                [0 if initial else 2],
                dtype=np.int64,
            )
        elif name.endswith("valid"):
            result[name] = np.zeros(shape) if initial else np.ones(shape)
        else:
            result[name] = np.full(shape, value, dtype=np.float64)
    return result

after0 = observation(value=1.0)
after1 = observation(value=2.0)
trajectory = [
    {
        "observations": observation(initial=True),
        "next_observations": after0,
        "actions": np.zeros(3),
        "dones": False,
        "infos": {"succeed": False},
    },
    {
        "observations": after0,
        "next_observations": after1,
        "actions": np.ones(3),
        "dones": True,
        "infos": {"succeed": True},
    },
]
episode_data = _episode_data_from_trajectory(
    trajectory,
    data_fps=60.0,
    environment_schema=schema,
)
for name in field_shapes:
    assert name in episode_data
    assert f"next_{name}" in episode_data
assert episode_data["terminal"].tolist() == [0, 1]
assert episode_data["succeed"].tolist() == [0, 1]

with tempfile.TemporaryDirectory() as temp:
    temp = Path(temp)
    source = temp / "source"
    demo = temp / "output"
    source.mkdir()
    demo.mkdir()
    (source / "replay_manifest.json").write_text(
        json.dumps(
            {
                "task": {"name": "t04_cover_pitcher"},
                "recording": {
                    "action_storage_layout": "single_pose_hand",
                },
            }
        )
    )
    store = ZarrEpisodeStore.create_empty(
        storage=zarr.DirectoryStore(str(demo / "replay.zarr"))
    )
    store.append_episode(episode_data, compressors=None)
    write_recording_schema(
        demo,
        store.root,
        episode_data,
        environment_schema=schema,
    )
    manifest_path = _write_physx_recapture_manifest(
        demo,
        source_demo_dir=source,
        environment_schema=schema,
        episode_data=episode_data,
        succeeded=True,
        data_fps=60.0,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["recording"]["physical_transition_contract"] == (
        "physx_state_t_plus_exact_robot_substep_path_t_"
        "produces_next_physx_state_t"
    )
    assert manifest["recapture"][
        "task_object_state_written_during_action_reexecution"
    ] is False
    assert json.loads(
        (demo / "recording_schema.json").read_text()
    )["environment_schema"]["version"] == 4
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
