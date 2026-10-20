from copy import deepcopy

import numpy as np
import pytest
import zarr

from handson.tasks.eval_comb01_cover_pitcher_and_box.recording import (
    TASK_NAME, hidden_field_schema, task_state_envelope,
)
from handson.sim.isaaclab.eval_comb01_cover_pitcher_and_box_env import XY5PitcherThenBoxLidEnv
from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv
from scripts.certify_physx_replay import _validated_task_state_schema, CertificationError


def test_xy5_current_and_next_judge_schema_is_compared_never_restored(monkeypatch):
    monkeypatch.setattr(HandsOnABCIsaacLabEnv, "get_recording_schema", lambda self: {"fields": {}})
    env = XY5PitcherThenBoxLidEnv(launch_isaac=False, image_obs=False)
    schema = env.get_recording_schema()
    assert schema["task_state"] == task_state_envelope()
    field = hidden_field_schema()
    stored = {name: {"shape_per_step": [21], "dtype": "float64", "semantics": field} for name in ("xy5_hidden_state", "next_xy5_hidden_state")}
    assert _validated_task_state_schema(TASK_NAME, schema, schema["fields"], stored) == {"xy5_hidden_state": field}
    damaged = deepcopy(stored)
    damaged.pop("next_xy5_hidden_state")
    with pytest.raises(CertificationError):
        _validated_task_state_schema(TASK_NAME, schema, schema["fields"], damaged)
    damaged_schema = deepcopy(schema)
    damaged_schema["fields"]["xy5_hidden_state"]["replay_restore_supported"] = True
    with pytest.raises(CertificationError):
        _validated_task_state_schema(TASK_NAME, damaged_schema, damaged_schema["fields"], stored)
    env.close()


def test_xy5_postrender_loader_preserves_both_extra_arrays(tmp_path):
    # Recorder and replay CLIs each own absl flags; use the real replay loader
    # in its normal separate process instead of sharing the CLI flag registry.
    import subprocess
    import sys
    code = r"""
from pathlib import Path
import sys
import numpy as np
import zarr
from scripts.replay_demos_zarr import _load_ycb02_recording_state_extras, _ycb02_recording_extras_at
path = Path(sys.argv[1]) / "replay.zarr"
data = zarr.open_group(str(path), mode="w").create_group("data")
hidden = np.arange(63, dtype=np.float64).reshape(3, 21)
poses = np.zeros((3, 4, 7), dtype=np.float64)
poses[..., 3] = 1
data["xy5_hidden_state"] = hidden
data["xy5_reset_poses"] = poses
kwargs = dict(expected_steps=3, field_shapes={"xy5_hidden_state": ((21,),), "xy5_reset_poses": ((4, 7),)}, task_label="XY5")
loaded = _load_ycb02_recording_state_extras(path, **kwargs)
at = _ycb02_recording_extras_at(loaded, 1)
np.testing.assert_array_equal(at["xy5_hidden_state"], hidden[1])
np.testing.assert_array_equal(at["xy5_reset_poses"], poses[1])
data["xy5_hidden_state"] = hidden[:2]
try:
    _load_ycb02_recording_state_extras(path, **kwargs)
except ValueError as exc:
    assert "expected exactly 3" in str(exc)
else:
    raise AssertionError("truncated XY5 history was accepted")
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True)
