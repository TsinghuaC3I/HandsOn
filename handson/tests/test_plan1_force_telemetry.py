from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import zarr

from handson.data.episode_store import ZarrEpisodeStore
from handson.data.plan1_force_telemetry import (
    MAX_FIELD_ELEMENTS,
    Plan1ForceTelemetry,
    ROLE_HAMMER,
)
from handson.data.recording_fields import (
    stack_transition_numeric_observation_fields,
    write_recording_schema,
)


def _telemetry_env(*, touching: bool, n_substeps: int = 2):
    hand_x = 0.15 if touching else 1.20
    model = mujoco.MjModel.from_xml_string(
        f"""
        <mujoco model="plan1_force_test">
          <option timestep="0.002" gravity="0 0 0" solver="Newton"
                  iterations="25" noslip_iterations="3"/>
          <worldbody>
            <body name="right_finger1" pos="{hand_x} 0 0.2">
              <joint name="right_finger1_joint1" type="hinge"/>
              <geom name="right_tip" type="sphere" size="0.1"/>
            </body>
            <body name="hammer_body" pos="0 0 0.2">
              <freejoint name="hammer_joint"/>
              <geom name="face" type="sphere" size="0.1" mass="1"/>
            </body>
            <body name="wood" pos="0.6 0 0.2">
              <freejoint name="wood_joint"/>
              <geom name="wood_collision" type="box" size="0.03 0.03 0.03"
                    mass="0.1"/>
            </body>
            <body name="nail" mocap="true" pos="0.7 0 0.2">
              <geom name="nail_head" type="sphere" size="0.01" mass="0.01"/>
            </body>
            <body name="table" pos="0 0 -1">
              <geom name="table_collision" type="box" size="1 1 0.1"/>
            </body>
          </worldbody>
          <actuator>
            <position name="right_finger1_joint1_actuator"
                      joint="right_finger1_joint1" kp="10"
                      ctrlrange="-1 1" forcerange="-4 4"/>
          </actuator>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    env = SimpleNamespace(
        _model=model,
        _data=data,
        _n_substeps=n_substeps,
        control_dt=n_substeps * float(model.opt.timestep),
        _hand=SimpleNamespace(
            joint_names=["right_finger1_joint1"],
            ctrl_ids=np.asarray([model.actuator("right_finger1_joint1_actuator").id]),
        ),
        _c01_hand_collision_geom_ids={
            "right": {model.geom("right_tip").id},
            "left": set(),
        },
        _hammer_geom_ids={model.geom("face").id},
        _wood_collision_geom_id=model.geom("wood_collision").id,
        _nail_geom_ids={model.geom("nail_head").id},
        _table_geom_id=model.geom("table_collision").id,
        _hammer_body_id=model.body("hammer_body").id,
        _hammer_joint_id=model.joint("hammer_joint").id,
        _wood_body_id=model.body("wood").id,
        _wood_joint_id=model.joint("wood_joint").id,
        _nail_body_id=model.body("nail").id,
        _nail_slide_joint_id=-1,
        _table_body_id=model.body("table").id,
    )
    return env


def _capture_frame(telemetry: Plan1ForceTelemetry) -> dict[str, np.ndarray]:
    telemetry.begin_frame()
    for _ in range(telemetry.n_substeps):
        telemetry.capture_pre_substep()
        mujoco.mj_step(telemetry.model, telemetry.data)
        telemetry.capture_substep()
    return telemetry.finalize_frame()


def test_fixed_shapes_dtypes_and_spaces_are_recorder_safe() -> None:
    telemetry = Plan1ForceTelemetry(_telemetry_env(touching=True))
    fields = _capture_frame(telemetry)
    spaces = telemetry.observation_spaces()

    assert fields["c01_force_contact_mask"].shape == (2, 128)
    assert fields["c01_force_contact_frame"].shape == (2, 128, 9)
    assert fields["c01_force_contact_ids"].dtype == np.int32
    assert fields["c01_force_contact_mask"].dtype == np.uint8
    assert fields["c01_force_contact_wrench_world_on_geom2"].dtype == np.float64
    assert fields["c01_force_substep_hand_joint"].shape == (2, 64, 9)
    assert fields["c01_force_substep_hand_actuator"].shape == (2, 64, 6)
    assert fields["c01_force_substep_pre_ctrl"].shape == (2, telemetry.model.nu)
    assert fields["c01_force_substep_pre_xfrc_applied"].shape == (
        2,
        telemetry.model.nbody,
        6,
    )
    assert fields["c01_force_sim_time"] == pytest.approx(
        np.asarray([telemetry.data.time])
    )

    assert set(spaces) == set(fields)
    for name, value in fields.items():
        assert value.ndim <= 3, name
        assert value.size <= MAX_FIELD_ELEMENTS, name
        assert spaces[name].contains(value), name

    assert fields["c01_force_hand_joint_mask"].sum() == 1
    assert fields["c01_force_hand_actuator_mask"].sum() == 1
    assert fields["c01_force_object_mask"].sum() == 4
    assert np.array_equal(fields["c01_force_qpos"], telemetry.data.qpos)
    assert np.array_equal(fields["c01_force_qvel"], telemetry.data.qvel)
    assert np.array_equal(fields["c01_force_ctrl"], telemetry.data.ctrl)
    assert np.array_equal(
        fields["c01_force_xfrc_applied"], telemetry.data.xfrc_applied
    )
    np.testing.assert_allclose(
        fields["c01_force_substep_hand_actuator"][-1, 0, 1],
        fields["c01_force_hand_actuator_state"][0, 1],
    )
    np.testing.assert_allclose(
        fields["c01_force_substep_hand_actuator"][-1, 0, 4:6],
        fields["c01_force_hand_actuator_dynamics"][0, 2:4],
    )
    np.testing.assert_allclose(
        fields["c01_force_substep_hand_joint"][-1, 0, 7:9],
        fields["c01_force_hand_joint_dynamics"][0, :2],
    )
    np.testing.assert_allclose(
        fields["c01_force_substep_pre_xfrc_applied"][-1],
        telemetry.data.xfrc_applied,
    )
    np.testing.assert_allclose(
        fields["c01_force_object_assist_wrench_world"][0],
        fields["c01_force_substep_pre_xfrc_applied"][-1][
            telemetry.model.body("hammer_body").id
        ],
    )


def test_contact_frame_transform_and_geom2_sign_are_exact() -> None:
    telemetry = Plan1ForceTelemetry(_telemetry_env(touching=True))
    fields = _capture_frame(telemetry)
    active = np.argwhere(fields["c01_force_contact_mask"] == 1)
    assert active.size > 0
    substep, contact_slot = active[0]

    frame = fields["c01_force_contact_frame"][substep, contact_slot].reshape(3, 3)
    local = fields["c01_force_contact_wrench_local_on_geom2"][
        substep, contact_slot
    ]
    world = fields["c01_force_contact_wrench_world_on_geom2"][
        substep, contact_slot
    ]
    ids = fields["c01_force_contact_ids"][substep, contact_slot]

    assert np.allclose(world[:3], frame.T @ local[:3])
    assert np.allclose(world[3:], frame.T @ local[3:])
    assert ids[6] == ROLE_HAMMER  # geom2 is the hammer in this authored model.
    assert local[0] > 0.0
    assert world[0] < 0.0  # The hammer is pushed away from the hand at +x.

    hammer_slot = 0
    assert np.allclose(
        fields["c01_force_object_contact_wrench_world"][hammer_slot, :3],
        fields["c01_force_contact_wrench_world_on_geom2"][-1, 0, :3],
    )


def test_no_contact_keeps_masks_and_wrenches_zero() -> None:
    telemetry = Plan1ForceTelemetry(_telemetry_env(touching=False))
    fields = _capture_frame(telemetry)

    assert fields["c01_force_substep_mask"].tolist() == [1, 1]
    assert not fields["c01_force_contact_mask"].any()
    assert not fields["c01_force_contact_overflow"].any()
    assert not fields["c01_force_contact_wrench_local_on_geom2"].any()
    assert not fields["c01_force_contact_wrench_world_on_geom2"].any()
    assert not fields["c01_force_object_contact_wrench_world"].any()
    assert not fields["c01_force_object_contact_impulse_world"].any()


def test_object_net_uses_last_applied_substep_wrench_not_next_queued_wrench() -> None:
    telemetry = Plan1ForceTelemetry(_telemetry_env(touching=False, n_substeps=1))
    hammer_body = telemetry.model.body("hammer_body").id
    applied = np.asarray([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    queued_next = np.asarray([7.0, 8.0, 9.0, 0.7, 0.8, 0.9])

    telemetry.begin_frame()
    telemetry.data.xfrc_applied[hammer_body] = applied
    telemetry.capture_pre_substep()
    mujoco.mj_step(telemetry.model, telemetry.data)
    telemetry.capture_substep()
    telemetry.data.xfrc_applied[hammer_body] = queued_next
    fields = telemetry.finalize_frame()

    np.testing.assert_allclose(
        fields["c01_force_object_assist_wrench_world"][0],
        applied,
    )
    np.testing.assert_allclose(
        fields["c01_force_xfrc_applied"][hammer_body],
        queued_next,
    )


def test_pre_solve_post_rows_preserve_mujoco_step_timing() -> None:
    telemetry = Plan1ForceTelemetry(_telemetry_env(touching=False, n_substeps=1))
    hammer_joint = telemetry.model.joint("hammer_joint")
    qpos_adr = int(telemetry.model.jnt_qposadr[hammer_joint.id])
    dof_adr = int(telemetry.model.jnt_dofadr[hammer_joint.id])
    actuator_id = telemetry.model.actuator("right_finger1_joint1_actuator").id

    telemetry.data.qvel[dof_adr] = 0.5
    telemetry.data.ctrl[actuator_id] = 0.4
    warmstart = np.linspace(0.1, 0.1 * telemetry.model.nv, telemetry.model.nv)
    telemetry.data.qacc_warmstart[:] = warmstart

    telemetry.begin_frame()
    telemetry.capture_pre_substep()
    pre_qpos = telemetry.data.qpos.copy()
    pre_qvel = telemetry.data.qvel.copy()
    mujoco.mj_step(telemetry.model, telemetry.data)
    solved_qacc = telemetry.data.qacc.copy()
    solved_actuator_force = telemetry.data.actuator_force.copy()
    post_qpos = telemetry.data.qpos.copy()
    post_qvel = telemetry.data.qvel.copy()
    telemetry.capture_substep()
    fields = telemetry.finalize_frame()

    assert fields["c01_force_substep_pre_time"][0] == pytest.approx(0.0)
    assert fields["c01_force_substep_post_time"][0] == pytest.approx(
        telemetry.model.opt.timestep
    )
    assert fields["c01_force_substep_time"][0] == pytest.approx(
        fields["c01_force_substep_post_time"][0]
    )
    np.testing.assert_array_equal(fields["c01_force_substep_pre_qpos"][0], pre_qpos)
    np.testing.assert_array_equal(fields["c01_force_substep_pre_qvel"][0], pre_qvel)
    np.testing.assert_array_equal(fields["c01_force_substep_post_qpos"][0], post_qpos)
    np.testing.assert_array_equal(fields["c01_force_substep_post_qvel"][0], post_qvel)
    np.testing.assert_array_equal(
        fields["c01_force_substep_pre_qacc_warmstart"][0], warmstart
    )
    np.testing.assert_array_equal(
        fields["c01_force_substep_solve_qacc"][0], solved_qacc
    )
    np.testing.assert_array_equal(
        fields["c01_force_substep_solve_actuator_force"][0],
        solved_actuator_force,
    )
    assert fields["c01_force_substep_pre_qpos"][0, qpos_adr] == pytest.approx(0.0)
    assert fields["c01_force_substep_post_qpos"][0, qpos_adr] > 0.0
    assert fields["c01_force_substep_pre_ctrl"][0, actuator_id] == pytest.approx(0.4)


def test_pre_and_post_capture_pairing_is_strict() -> None:
    telemetry = Plan1ForceTelemetry(_telemetry_env(touching=False, n_substeps=1))

    with pytest.raises(RuntimeError, match="requires capture_pre_substep"):
        telemetry.capture_substep()

    telemetry.capture_pre_substep()
    with pytest.raises(RuntimeError, match="called twice"):
        telemetry.capture_pre_substep()
    with pytest.raises(RuntimeError, match="without the matching"):
        telemetry.finalize_frame()


def test_schema_is_json_safe_and_describes_reproduction_contract() -> None:
    telemetry = Plan1ForceTelemetry(_telemetry_env(touching=True))
    schema = telemetry.schema()

    encoded = json.dumps(schema, sort_keys=True)
    assert "handson.plan1_force_telemetry" in encoded
    assert schema["role_codebook"][str(ROLE_HAMMER)] == "hammer"
    assert schema["mujoco"]["physics_dt_s"] == pytest.approx(0.002)
    assert schema["mujoco"]["n_substeps"] == 2
    assert len(schema["mujoco"]["compiled_model_sha256"]) == 64
    assert hashlib.sha256(telemetry.compiled_model_bytes()).hexdigest() == (
        schema["mujoco"]["compiled_model_sha256"]
    )
    assert schema["fields"]["c01_force_contact_frame"]["shape"] == [2, 128, 9]
    assert schema["fields"]["c01_force_contact_frame"]["layout"].startswith(
        "row-major"
    )
    assert "mass_and_inertia" in schema["dynamic_parameters"]
    assert schema["mappings"]["hand_joints"][0]["side"] == "right_hand"


def test_episode_model_snapshot_refreshes_after_reset_time_model_changes() -> None:
    telemetry = Plan1ForceTelemetry(_telemetry_env(touching=True))
    constructor_hash = telemetry.schema()["mujoco"]["compiled_model_sha256"]
    hammer_id = telemetry.model.body("hammer_body").id
    telemetry.model.body_mass[hammer_id] *= 1.5

    telemetry.refresh_episode_model_snapshot()

    episode_hash = telemetry.schema()["mujoco"]["compiled_model_sha256"]
    assert episode_hash != constructor_hash
    assert hashlib.sha256(telemetry.compiled_model_bytes()).hexdigest() == episode_hash


def test_contact_capacity_is_capped_at_128() -> None:
    env = _telemetry_env(touching=False)
    with pytest.raises(ValueError, match="max_contacts"):
        Plan1ForceTelemetry(env, max_contacts=129)


def test_full_declared_telemetry_roundtrips_with_action_to_next_alignment(
    tmp_path,
) -> None:
    telemetry = Plan1ForceTelemetry(_telemetry_env(touching=True))
    current = _capture_frame(telemetry)
    following = _capture_frame(telemetry)
    required = set(telemetry.schema()["fields"])

    stacked = stack_transition_numeric_observation_fields(
        [
            {
                "observations": current,
                "next_observations": following,
            }
        ],
        required_fields=required,
    )

    assert set(required) <= set(stacked)
    assert {f"next_{name}" for name in required} <= set(stacked)
    assert stacked["c01_force_contact_frame"].shape == (1, 2, 128, 9)
    assert stacked["next_c01_force_contact_frame"].shape == (1, 2, 128, 9)
    np.testing.assert_allclose(
        stacked["next_c01_force_sim_time"][:, 0],
        following["c01_force_sim_time"],
    )

    episode_data = {
        "action": np.zeros((1, 1), dtype=np.float32),
        "timestamp": current["c01_force_sim_time"].copy(),
        **stacked,
    }
    store = ZarrEpisodeStore.create_empty(storage=zarr.MemoryStore())
    store.append_episode(episode_data, compressors=None)
    schema_path = write_recording_schema(
        tmp_path,
        store.root,
        episode_data,
        environment_schema=telemetry.schema(),
    )

    assert schema_path.is_file()
    assert store.data["next_c01_force_contact_frame"].shape == (1, 2, 128, 9)
    assert (
        store.data["next_c01_force_contact_frame"].attrs["source"]
        == "next_observation"
    )
