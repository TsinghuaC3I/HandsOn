"""CPU checks of XY2's actual recorder schema and formal replay loader."""

import json

import numpy as np
import pytest
import zarr

from handson.data.physical_replay_certification import _recording_schema_report_contract
from handson.sim.isaaclab.bimanual_assembly_env import BimanualAssemblyIsaacLabEnv
from handson.sim.isaaclab.xy2_collection_env import HandsOnXY2IsaacLabEnv
from handson.sim.isaaclab.xy2_contract import TASK_NAME, RECORDING_LAYOUTS
from scripts.certify_physx_replay import (
    CertificationError, _initialization_field_partitions, compare_snapshot, load_demo,
)
from scripts.tests.test_certify_physx_replay import _write_demo


@pytest.fixture
def demo_dir(tmp_path, monkeypatch):
    # Keep the existing shared v6 physical-state fixture; extend it through
    # the real XY2 environment schema, then use the real on-disk loaders.
    demo = _write_demo(tmp_path)
    manifest_path = demo / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["task"]["name"] = TASK_NAME
    manifest["recording"]["stable_recording_origin"]["task_name"] = TASK_NAME
    manifest_path.write_text(json.dumps(manifest))
    schema_path = demo / "recording_schema.json"
    schema = json.loads(schema_path.read_text())
    monkeypatch.setattr(BimanualAssemblyIsaacLabEnv, "get_recording_schema",
                        lambda self: schema["environment_schema"])
    env = HandsOnXY2IsaacLabEnv(launch_isaac=False, image_obs=False)
    try:
        schema["environment_schema"] = env.get_recording_schema()
    finally:
        env.close()
    data = zarr.open_group(str(demo / "replay.zarr"), mode="a")["data"]
    for name, layout in RECORDING_LAYOUTS.items():
        semantics = schema["environment_schema"]["fields"][name]
        stored = {"dtype": "float64", "shape_per_step": [len(layout)], "semantics": semantics}
        schema["fields"][name] = stored
        schema["fields"][f"next_{name}"] = stored
        boundaries = np.arange(3 * len(layout), dtype=np.float64).reshape(3, -1)
        data.array(name, boundaries[:-1])
        data.array(f"next_{name}", boundaries[1:])
    schema_path.write_text(json.dumps(schema))
    return demo


def test_actual_xy2_schema_loads_and_compares_but_never_restores_history(demo_dir):
    demo = load_demo(demo_dir)
    restorable, compared, _ = _initialization_field_partitions(demo.field_specs)
    assert set(RECORDING_LAYOUTS).isdisjoint(spec.name for spec in restorable)
    assert set(RECORDING_LAYOUTS) <= {spec.name for spec in compared}
    atols, _, _, errors = _recording_schema_report_contract(demo_dir)
    assert errors == []
    assert set(RECORDING_LAYOUTS) <= set(atols)


@pytest.mark.parametrize("field", RECORDING_LAYOUTS)
def test_xy2_replay_detects_hidden_state_divergence_and_broken_boundary(demo_dir, field):
    demo = load_demo(demo_dir)
    recorded = {spec.name: np.asarray(demo.next_fields[spec.name][0]) for spec in demo.field_specs}
    actual = {name: value.copy() for name, value in recorded.items()}
    actual[field][0] += 1
    _, divergence = compare_snapshot(
        recorded, actual, demo.field_specs, phase="post_action", action_index=0,
        recorded_time_origin=0., actual_time_origin=0.,
    )
    assert divergence["field"] == field
    data = zarr.open_group(str(demo_dir / "replay.zarr"), mode="a")["data"]
    data[field][1, 0] += 1
    with pytest.raises(CertificationError, match=f"continuity is broken for {field}"):
        load_demo(demo_dir)
