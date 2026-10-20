from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import pytest
import zarr

from handson.data.episode_store import ZarrEpisodeStore
from handson.data.recording_fields import (
    RECORDER_MONOTONIC_TIME_KEY,
    build_recording_schema,
    declared_recording_field_names,
    effective_sampling_hz,
    resolve_environment_recording_schema,
    select_current_observation_for_recording,
    select_next_observation_for_recording,
    stack_numeric_observation_fields,
    stack_transition_numeric_observation_fields,
    transition_monotonic_timestamps,
    transition_timestamps,
    wall_sampling_contract,
    wall_sampling_contract_errors,
    write_recording_schema,
    write_replay_assets,
)


def test_episode_store_saves_zero_length_payload_dimensions(tmp_path) -> None:
    zarr_path = tmp_path / "replay.zarr"
    store = ZarrEpisodeStore.create_empty(
        storage=zarr.DirectoryStore(str(zarr_path))
    )
    first = {
        "action": np.zeros((3, 4), dtype=np.float32),
        "physx_task_articulation_dof_position": np.empty(
            (3, 4, 0),
            dtype=np.float64,
        ),
        "physx_task_rigid_body_state": np.empty(
            (3, 0, 13),
            dtype=np.float64,
        ),
    }
    second = {
        name: np.empty((2, *value.shape[1:]), dtype=value.dtype)
        for name, value in first.items()
    }

    store.append_episode(first, compressors="disk")
    store.append_episode(second, compressors="disk")

    assert store.total_steps == 5
    assert store.data["physx_task_articulation_dof_position"].shape == (
        5,
        4,
        0,
    )
    assert store.data["physx_task_rigid_body_state"].shape == (5, 0, 13)
    for dataset in store.data.values():
        assert all(size > 0 for size in dataset.chunks)

    reopened = zarr.open(str(zarr_path), mode="r")
    assert reopened["data"]["physx_task_articulation_dof_position"].shape == (
        5,
        4,
        0,
    )


def test_declared_zero_length_observation_fields_stack_for_saving() -> None:
    empty_dofs = np.empty((4, 0), dtype=np.float64)
    trajectory = [
        {
            "observations": {
                "physx_task_articulation_dof_position": empty_dofs.copy(),
            },
            "next_observations": {
                "physx_task_articulation_dof_position": empty_dofs.copy(),
            },
        }
        for _step in range(3)
    ]
    environment_schema = {
        "fields": {
            "physx_task_articulation_dof_position": {
                "shape": [4, 0],
                "dtype": "float64",
            },
        },
    }

    stacked = stack_transition_numeric_observation_fields(
        trajectory,
        required_fields={"physx_task_articulation_dof_position"},
        environment_schema=environment_schema,
    )

    assert stacked["physx_task_articulation_dof_position"].shape == (3, 4, 0)
    assert stacked["next_physx_task_articulation_dof_position"].shape == (
        3,
        4,
        0,
    )


def _trajectory(step_count: int = 3):
    result = []
    for step in range(step_count):
        result.append(
            {
                "observations": {
                    "state": np.full(4, step, dtype=np.float32),
                    "plan1_contact_wrench": np.full(
                        (32, 18), step + 0.25, dtype=np.float32
                    ),
                    "plan1_contact_valid": np.asarray(
                        [(slot + step) % 2 for slot in range(32)], dtype=np.uint8
                    ),
                    "long_joint_vector": np.arange(256, dtype=np.float64) + step,
                    "main": np.zeros((32, 32, 3), dtype=np.uint8),
                    "text": "not numeric telemetry",
                }
            }
        )
    return result


def test_stacks_force_matrices_masks_and_vectors_larger_than_legacy_limit() -> None:
    stacked = stack_numeric_observation_fields(_trajectory())

    assert set(stacked) == {
        "long_joint_vector",
        "plan1_contact_valid",
        "plan1_contact_wrench",
    }
    assert stacked["plan1_contact_wrench"].shape == (3, 32, 18)
    assert stacked["plan1_contact_wrench"].dtype == np.float32
    assert stacked["plan1_contact_valid"].shape == (3, 32)
    assert stacked["long_joint_vector"].shape == (3, 256)
    assert stacked["plan1_contact_wrench"].flags.c_contiguous
    np.testing.assert_allclose(stacked["plan1_contact_wrench"][2], 2.25)


def test_stacks_all_ten_physics_substeps_without_truncating_contacts() -> None:
    trajectory = [
        {
            "observations": {
                "c01_force_micro_contact_wrench": np.full(
                    (10, 128, 12),
                    step,
                    dtype=np.float32,
                ),
                # This has an image-like H×W×3 shape.  An explicit environment
                # schema declaration must take precedence over the heuristic.
                "c01_force_micro_contact_pos": np.full(
                    (10, 128, 3),
                    step,
                    dtype=np.float32,
                ),
            }
        }
        for step in range(3)
    ]

    stacked = stack_numeric_observation_fields(
        trajectory,
        required_fields={
            "c01_force_micro_contact_wrench",
            "c01_force_micro_contact_pos",
        },
    )

    assert stacked["c01_force_micro_contact_wrench"].shape == (3, 10, 128, 12)
    assert stacked["c01_force_micro_contact_wrench"].dtype == np.float32
    assert stacked["c01_force_micro_contact_pos"].shape == (3, 10, 128, 3)


def test_declared_force_field_cannot_silently_change_shape() -> None:
    trajectory = _trajectory()
    trajectory[1]["observations"]["plan1_contact_wrench"] = np.zeros(
        (31, 18), dtype=np.float32
    )

    with pytest.raises(ValueError, match="Required recording field.*changes shape"):
        stack_numeric_observation_fields(
            trajectory,
            required_fields={"plan1_contact_wrench"},
        )


def test_transition_fields_align_action_with_next_force_sample() -> None:
    trajectory = []
    for step in range(3):
        trajectory.append(
            {
                "observations": {
                    "c01_force_sim_time": np.asarray([0.02 * step]),
                },
                "next_observations": {
                    "c01_force_sim_time": np.asarray([0.02 * (step + 1)]),
                },
            }
        )

    stacked = stack_transition_numeric_observation_fields(
        trajectory,
        required_fields={"c01_force_sim_time"},
    )

    np.testing.assert_allclose(
        stacked["c01_force_sim_time"][:, 0],
        [0.0, 0.02, 0.04],
    )
    np.testing.assert_allclose(
        stacked["next_c01_force_sim_time"][:, 0],
        [0.02, 0.04, 0.06],
    )


def test_next_observation_selection_excludes_rgb_and_keeps_training_targets() -> None:
    state = np.arange(4, dtype=np.float64)
    force = np.ones((10, 128, 3), dtype=np.float64)
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    observation = {
        "state": state,
        "c01_force_contact_pos": force,
        "main": image,
        "wrist_left": image,
    }

    selected = select_next_observation_for_recording(
        observation,
        required_fields={"c01_force_contact_pos"},
    )

    assert set(selected) == {"state", "c01_force_contact_pos"}
    assert selected["state"] is state
    assert selected["c01_force_contact_pos"] is force
    assert "main" not in selected
    assert "wrist_left" not in selected


def test_current_force_is_stored_once_then_reconstructed_from_previous_next() -> None:
    required = {"c01_force_sim_time"}
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    def observation(step: int) -> dict[str, np.ndarray]:
        return {
            "state": np.asarray([step], dtype=np.float32),
            "main": image.copy(),
            "c01_force_sim_time": np.asarray([0.02 * step], dtype=np.float64),
        }

    observations = [observation(step) for step in range(3)]
    trajectory = [
        {
            "observations": select_current_observation_for_recording(
                observations[0],
                required_fields=required,
                include_required_fields=True,
            ),
            "next_observations": select_next_observation_for_recording(
                observations[1], required_fields=required
            ),
        },
        {
            "observations": select_current_observation_for_recording(
                observations[1],
                required_fields=required,
                include_required_fields=False,
            ),
            "next_observations": select_next_observation_for_recording(
                observations[2], required_fields=required
            ),
        },
    ]

    assert "main" in trajectory[1]["observations"]
    assert "c01_force_sim_time" not in trajectory[1]["observations"]
    stacked = stack_transition_numeric_observation_fields(
        trajectory,
        required_fields=required,
    )
    np.testing.assert_allclose(stacked["c01_force_sim_time"][:, 0], [0.0, 0.02])
    np.testing.assert_allclose(
        stacked["next_c01_force_sim_time"][:, 0],
        [0.02, 0.04],
    )


@pytest.mark.parametrize("transition_side", ["observations", "next_observations"])
@pytest.mark.parametrize(
    ("bad_value", "message"),
    [
        (np.zeros((3,), dtype=np.float64), "has shape"),
        (np.zeros((2,), dtype=np.float32), "has dtype"),
    ],
)
def test_declared_schema_rejects_shape_or_dtype_conflict_on_both_transition_sides(
    transition_side: str,
    bad_value: np.ndarray,
    message: str,
) -> None:
    field_name = "c01_force_sim_time"
    valid = np.zeros((2,), dtype=np.float64)
    transition = {
        "observations": {field_name: valid},
        "next_observations": {field_name: valid},
    }
    transition[transition_side] = {field_name: bad_value}
    environment_schema = {
        "fields": {
            field_name: {
                "shape": [2],
                "dtype": "float64",
            }
        }
    }

    with pytest.raises(ValueError, match=message):
        stack_transition_numeric_observation_fields(
            [transition],
            environment_schema=environment_schema,
        )


def test_schema_builder_refuses_declared_metadata_conflict() -> None:
    field_name = "c01_force_sim_time"
    with pytest.raises(ValueError, match="conflicts with the environment schema"):
        build_recording_schema(
            {
                "action": np.zeros((2, 1), dtype=np.float64),
                field_name: np.zeros((2, 1), dtype=np.float32),
            },
            environment_schema={
                "fields": {
                    field_name: {"shape": [1], "dtype": "float64"},
                }
            },
        )


def _substep_integrity_fixture():
    def frame(count: int, mask) -> dict[str, np.ndarray]:
        return {
            "c01_force_substep_count": np.asarray([count], dtype=np.int32),
            "c01_force_substep_mask": np.asarray(mask, dtype=np.uint8),
        }

    trajectory = [
        {
            "observations": frame(0, [0, 0]),
            "next_observations": frame(2, [1, 1]),
        },
        {
            "observations": frame(2, [1, 1]),
            "next_observations": frame(2, [1, 1]),
        },
    ]
    schema = {
        "mujoco": {"n_substeps": 2},
        "fields": {
            "c01_force_substep_count": {"shape": [1], "dtype": "int32"},
            "c01_force_substep_mask": {"shape": [2], "dtype": "uint8"},
        },
    }
    return trajectory, schema


def test_substep_integrity_accepts_reset_then_complete_current_and_next_frames() -> None:
    trajectory, schema = _substep_integrity_fixture()

    fields = stack_transition_numeric_observation_fields(
        trajectory,
        environment_schema=schema,
    )

    assert fields["c01_force_substep_count"][:, 0].tolist() == [0, 2]
    assert fields["next_c01_force_substep_count"][:, 0].tolist() == [2, 2]


@pytest.mark.parametrize(
    ("side", "step_idx", "count", "mask", "message"),
    [
        ("next_observations", 0, 1, [1, 0], "Incomplete force substep"),
        ("observations", 1, 1, [1, 0], "Incomplete force substep"),
        ("next_observations", 1, 2, [1, 0], "does not match"),
    ],
)
def test_substep_integrity_rejects_missing_or_inconsistent_substeps(
    side: str,
    step_idx: int,
    count: int,
    mask: list[int],
    message: str,
) -> None:
    trajectory, schema = _substep_integrity_fixture()
    trajectory[step_idx][side]["c01_force_substep_count"] = np.asarray(
        [count], dtype=np.int32
    )
    trajectory[step_idx][side]["c01_force_substep_mask"] = np.asarray(
        mask, dtype=np.uint8
    )

    with pytest.raises(ValueError, match=message):
        stack_transition_numeric_observation_fields(
            trajectory,
            environment_schema=schema,
        )


def _physx_v4_substep_integrity_fixture():
    n_substeps = 2
    robot_shape = (n_substeps, 2, 3)

    def frame(count: int, offset: float) -> dict[str, np.ndarray]:
        active = count > 0
        valid = np.ones(robot_shape, dtype=np.uint8)
        if not active:
            valid.fill(0)
        position = np.full(robot_shape, offset, dtype=np.float64)
        velocity = np.full(robot_shape, offset + 0.25, dtype=np.float64)
        target = np.full(robot_shape, offset + 0.5, dtype=np.float64)
        return {
            "physx_substep_count": np.asarray([count], dtype=np.int64),
            "physx_substep_robot_joint_position": position,
            "physx_substep_robot_joint_velocity": velocity,
            "physx_substep_robot_joint_target": target,
            "physx_substep_robot_joint_valid": valid,
            "physx_substep_robot_joint_target_valid": valid.copy(),
        }

    trajectory = [
        {
            "observations": frame(0, 0.0),
            "next_observations": frame(n_substeps, 1.0),
        },
        {
            # The production recorder omits repeated current telemetry and the
            # stacker reconstructs it from the preceding next observation.
            "observations": {},
            "next_observations": frame(n_substeps, 2.0),
        },
    ]
    fields = {
        "physx_substep_count": {"shape": [1], "dtype": "int64"},
    }
    for name in (
        "physx_substep_robot_joint_position",
        "physx_substep_robot_joint_velocity",
        "physx_substep_robot_joint_target",
    ):
        fields[name] = {"shape": list(robot_shape), "dtype": "float64"}
    for name in (
        "physx_substep_robot_joint_valid",
        "physx_substep_robot_joint_target_valid",
    ):
        fields[name] = {"shape": list(robot_shape), "dtype": "uint8"}
    schema = {"version": 4, "fields": fields}
    return trajectory, schema


def test_physx_v4_substep_integrity_accepts_reset_and_complete_next_frames() -> None:
    trajectory, schema = _physx_v4_substep_integrity_fixture()

    fields = stack_transition_numeric_observation_fields(
        trajectory,
        environment_schema=schema,
    )

    assert fields["physx_substep_count"][:, 0].tolist() == [0, 2]
    assert fields["next_physx_substep_count"][:, 0].tolist() == [2, 2]
    np.testing.assert_allclose(
        fields["physx_substep_robot_joint_position"][1],
        fields["next_physx_substep_robot_joint_position"][0],
    )


def test_physx_v4_partial_terminal_guard_requires_explicit_discard_opt_in() -> None:
    trajectory, schema = _physx_v4_substep_integrity_fixture()
    terminal_next = trajectory[-1]["next_observations"]
    terminal_next["physx_substep_count"][0] = 1
    for name, value in terminal_next.items():
        array = np.asarray(value)
        if name.startswith("physx_substep_") and array.shape[:1] == (2,):
            array[1:] = 0
    trajectory[-1]["infos"] = {
        "manual_reset": True,
        "numerical_guard_event": {
            "body": "task_body_alpha",
            "reason": "rigid_motion_spike",
            "physics_substep": 0,
        },
    }

    with pytest.raises(ValueError, match="Incomplete PhysX v4"):
        stack_transition_numeric_observation_fields(
            trajectory,
            environment_schema=schema,
        )

    fields = stack_transition_numeric_observation_fields(
        trajectory,
        environment_schema=schema,
        allow_partial_terminal_guard=True,
    )
    assert fields["next_physx_substep_count"][:, 0].tolist() == [2, 1]


@pytest.mark.parametrize("fault", ("missing_event", "nonzero_padding"))
def test_physx_v4_partial_terminal_guard_still_fails_closed(
    fault: str,
) -> None:
    trajectory, schema = _physx_v4_substep_integrity_fixture()
    terminal_next = trajectory[-1]["next_observations"]
    terminal_next["physx_substep_count"][0] = 1
    for name, value in terminal_next.items():
        array = np.asarray(value)
        if name.startswith("physx_substep_") and array.shape[:1] == (2,):
            array[1:] = 0
    trajectory[-1]["infos"] = {
        "manual_reset": True,
        "numerical_guard_event": {
            "body": "task_body_alpha",
            "reason": "rigid_motion_spike",
        },
    }
    if fault == "missing_event":
        trajectory[-1]["infos"].pop("numerical_guard_event")
    else:
        terminal_next["physx_substep_robot_joint_target"][1, 0, 0] = 1.0

    with pytest.raises(ValueError, match="Incomplete|non-zero padding"):
        stack_transition_numeric_observation_fields(
            trajectory,
            environment_schema=schema,
            allow_partial_terminal_guard=True,
        )


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("next_count", "Incomplete PhysX v4"),
        ("missing_schema_field", "schema is missing"),
        ("missing_observation_field", "is missing from observations"),
        ("target_valid_gap", "must exactly cover"),
        ("nonfinite_position", "contains non-finite"),
    ],
)
def test_physx_v4_substep_integrity_rejects_invalid_recordings(
    failure: str,
    message: str,
) -> None:
    trajectory, schema = _physx_v4_substep_integrity_fixture()
    if failure == "next_count":
        trajectory[0]["next_observations"]["physx_substep_count"][0] = 1
    elif failure == "missing_schema_field":
        del schema["fields"]["physx_substep_robot_joint_target_valid"]
    elif failure == "missing_observation_field":
        del trajectory[0]["next_observations"][
            "physx_substep_robot_joint_velocity"
        ]
    elif failure == "target_valid_gap":
        trajectory[0]["next_observations"][
            "physx_substep_robot_joint_target_valid"
        ][0, 0, 0] = 0
    elif failure == "nonfinite_position":
        trajectory[0]["next_observations"][
            "physx_substep_robot_joint_position"
        ][0, 0, 0] = np.nan

    with pytest.raises(ValueError, match=message):
        stack_transition_numeric_observation_fields(
            trajectory,
            environment_schema=schema,
        )


def test_transition_timestamps_prefer_complete_info_clock_then_fall_back() -> None:
    real_clock = transition_timestamps(
        [
            {"infos": {"timestamp": np.float64(101.25)}},
            {"info": {"time": 101.29}},
        ],
        fallback_hz=50.0,
    )
    np.testing.assert_allclose(real_clock, [101.25, 101.29], rtol=0.0, atol=0.0)

    fallback = transition_timestamps(
        [
            {"infos": {"timestamp": 101.25}},
            {"infos": {}},
        ],
        fallback_hz=50.0,
    )
    np.testing.assert_allclose(fallback, [0.0, 0.02], rtol=0.0, atol=0.0)


def test_wall_sampling_clock_is_real_monotonic_time_and_fail_closed() -> None:
    trajectory = [
        {RECORDER_MONOTONIC_TIME_KEY: 900.0 + index / 30.0}
        for index in range(31)
    ]
    timestamps = transition_monotonic_timestamps(trajectory)
    assert timestamps is not None
    np.testing.assert_allclose(
        timestamps,
        np.arange(31, dtype=np.float64) / 30.0,
        atol=1e-12,
    )
    assert effective_sampling_hz(timestamps, fallback_hz=1.0) == pytest.approx(
        30.0
    )
    contract = wall_sampling_contract(timestamps, target_hz=30.0)
    assert contract["qualified"] is True
    assert wall_sampling_contract_errors(contract) == []

    slow = wall_sampling_contract(
        np.arange(21, dtype=np.float64) / 20.0,
        target_hz=30.0,
    )
    assert slow["mean_hz"] == pytest.approx(20.0)
    assert slow["qualified"] is False
    assert "not production-qualified" in "; ".join(
        wall_sampling_contract_errors(slow)
    )

    with pytest.raises(ValueError, match="missing from part"):
        transition_monotonic_timestamps(
            [
                {RECORDER_MONOTONIC_TIME_KEY: 1.0},
                {},
            ]
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        transition_monotonic_timestamps(
            [
                {RECORDER_MONOTONIC_TIME_KEY: 1.0},
                {RECORDER_MONOTONIC_TIME_KEY: 1.0},
            ]
        )


def test_required_transition_force_rejects_legacy_shifted_layout() -> None:
    with pytest.raises(ValueError, match="requires next_observations"):
        stack_transition_numeric_observation_fields(
            [
                {
                    "observations": {
                        "c01_force_sim_time": np.asarray([0.0]),
                    }
                }
            ],
            required_fields={"c01_force_sim_time"},
        )


def test_force_transition_refuses_to_save_truncated_contacts() -> None:
    current = {
        "c01_force_contact_overflow": np.zeros(2, dtype=np.int32),
    }
    following = {
        "c01_force_contact_overflow": np.asarray([0, 1], dtype=np.int32),
    }

    with pytest.raises(ValueError, match="refusing to save.*incomplete"):
        stack_transition_numeric_observation_fields(
            [
                {
                    "observations": current,
                    "next_observations": following,
                }
            ],
            required_fields={"c01_force_contact_overflow"},
        )


def test_undeclared_inconsistent_numeric_field_is_warned_and_skipped() -> None:
    trajectory = _trajectory()
    del trajectory[1]["observations"]["long_joint_vector"]

    with pytest.warns(RuntimeWarning, match="long_joint_vector"):
        stacked = stack_numeric_observation_fields(trajectory)

    assert "long_joint_vector" not in stacked
    assert "plan1_contact_wrench" in stacked


def test_resolves_force_schema_through_environment_wrapper() -> None:
    class Inner:
        def get_force_recording_schema(self):
            return {
                "version": np.int64(1),
                "fields": {
                    "plan1_contact_wrench": {
                        "columns": np.asarray(["fx", "fy", "fz"]),
                        "units": "N",
                    }
                },
            }

    class Wrapper:
        def __init__(self):
            self.env = Inner()

    schema = resolve_environment_recording_schema(Wrapper())

    assert schema == {
        "version": 1,
        "fields": {
            "plan1_contact_wrench": {
                "columns": ["fx", "fy", "fz"],
                "units": "N",
            }
        },
    }
    assert declared_recording_field_names(schema) == {"plan1_contact_wrench"}


def test_schema_is_written_to_sidecar_root_attrs_and_dataset_attrs(tmp_path) -> None:
    episode_data = {
        "action": np.zeros((3, 4), dtype=np.float32),
        "timestamp": np.arange(3, dtype=np.float64) / 30.0,
        "plan1_contact_wrench": np.zeros((3, 32, 18), dtype=np.float32),
    }
    store = ZarrEpisodeStore.create_empty(storage=zarr.MemoryStore())
    store.append_episode(episode_data, compressors=None)
    environment_schema = {
        "version": 1,
        "fields": {
            "plan1_contact_wrench": {
                "columns": ["point_x", "point_y", "point_z", "force_x"],
                "units": ["m", "m", "m", "N"],
            }
        },
    }

    schema_path = write_recording_schema(
        tmp_path,
        store.root,
        episode_data,
        environment_schema=environment_schema,
    )

    with schema_path.open(encoding="utf-8") as stream:
        sidecar = json.load(stream)
    assert sidecar == store.root.attrs["recording_schema"]
    assert store.root.attrs["recording_schema_path"] == "../recording_schema.json"
    force_schema = sidecar["fields"]["plan1_contact_wrench"]
    assert force_schema["shape_per_step"] == [32, 18]
    assert force_schema["dtype"] == "float32"
    assert force_schema["semantics"]["units"][-1] == "N"
    assert store.data["plan1_contact_wrench"].attrs["time_axis"] == 0
    assert (
        store.data["plan1_contact_wrench"].attrs["semantics"]["columns"][-1]
        == "force_x"
    )


def test_next_force_field_reuses_declared_semantics_and_marks_alignment() -> None:
    episode_data = {
        "action": np.zeros((2, 4), dtype=np.float32),
        "next_c01_force_sim_time": np.asarray([[0.02], [0.04]], dtype=np.float64),
    }
    environment_schema = {
        "fields": {
            "c01_force_sim_time": {
                "columns": ["mujoco_time"],
                "units": ["s"],
            }
        }
    }

    schema = build_recording_schema(
        episode_data,
        environment_schema=environment_schema,
    )

    field = schema["fields"]["next_c01_force_sim_time"]
    assert field["source"] == "next_observation"
    assert field["semantics"]["units"] == ["s"]


def _tiny_compiled_model():
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco model="replay_asset_test">
          <worldbody>
            <body name="object" pos="0 0 0.2">
              <freejoint/>
              <geom type="box" size="0.02 0.03 0.04" mass="0.2"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def _compiled_model_sha256(model) -> str:
    return hashlib.sha256(_compiled_model_bytes(model)).hexdigest()


def _compiled_model_bytes(model) -> bytes:
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return buffer.tobytes()


def test_replay_assets_save_exact_model_hash_and_whitelisted_manifest(tmp_path) -> None:
    # Use a real small Git checkout: a multi-GB developer diff may exceed the
    # recorder's bounded provenance timeout, independently of manifest behavior.
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    source = repo / "source.py"
    source.write_text("version = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=HandsOn test",
                    "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
                    "-c", "core.hooksPath=/dev/null", "commit", "-qm", "fixture"], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"]).decode().strip()
    source.write_text("version = 2\n")
    model = _tiny_compiled_model()
    model_sha256 = _compiled_model_sha256(model)
    environment_schema = {
        "mujoco": {
            "compiled_model_sha256": model_sha256,
        }
    }
    schema_path = tmp_path / "recording_schema.json"
    schema_path.write_text(
        json.dumps({"environment_schema": environment_schema}) + "\n",
        encoding="utf-8",
    )

    class Inner:
        def __init__(self):
            self.model = model
            self._c01_force_telemetry = type(
                "Telemetry",
                (),
                {"compiled_model_bytes": lambda _self: _compiled_model_bytes(model)},
            )()

    class Wrapper:
        def __init__(self):
            self.env = Inner()

    manifest_path = write_replay_assets(
        tmp_path,
        Wrapper(),
        task_name="plan1_hammer_nail",
        recording_parameters={
            "step_count": np.int64(12),
            "data_fps": 50.0,
            "hand_type": "wuji1",
        },
        environment_schema=environment_schema,
        repo_root=repo,
        environ={
            "HANDSON_C01_PLAN_FORCE_GRASP": "1",
            "HANDSON_WUJI_GRASP_FORCE_SCALE": "2.5",
            "HANDSON_ARM_TYPE": "rokae",
            "HANDSON_HAND_DIRECT_QPOS": "0",
            "HANDSON_ROKAE_WUJI_ARM_ACTUATOR": "1",
            "HANDSON_WUJI_API_KEY": "must-not-be-recorded-either",
            "HAND": "wuji1",
            "OPENAI_API_KEY": "must-not-be-recorded",
            "PATH": "/also/not/recorded",
        },
    )

    model_path = tmp_path / "model.mjb"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == model_sha256
    loaded = mujoco.MjModel.from_binary_path(str(model_path))
    assert loaded.nq == model.nq
    assert loaded.nv == model.nv
    assert manifest["task"] == {"name": "plan1_hammer_nail"}
    assert manifest["recording"]["step_count"] == 12
    assert manifest["software"]["numpy"] == np.__version__
    assert manifest["software"]["git"]["commit"] == commit
    assert manifest["software"]["git"]["dirty"] is True
    assert manifest["software"]["git"]["changed_files"] == ["source.py"]
    assert len(manifest["software"]["git"]["diff_binary_sha256"]) == 64
    assert manifest["assets"]["recording_schema"] == {
        "path": "recording_schema.json",
        "sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
    }
    assert manifest["assets"]["compiled_model"] == {
        "status": "saved",
        "role": "schema_baseline",
        "path": "model.mjb",
        "sha256": model_sha256,
        "source": "force_telemetry_schema_baseline",
        "environment_schema_sha256": model_sha256,
        "matches_environment_schema": True,
    }
    assert manifest["assets"]["final_compiled_model"] == {
        "status": "same_as_baseline",
        "role": "runtime_final",
        "path": "model.mjb",
        "sha256": model_sha256,
    }
    assert manifest["environment"]["variables"] == {
        "HANDSON_ARM_TYPE": "rokae",
        "HANDSON_C01_PLAN_FORCE_GRASP": "1",
        "HANDSON_HAND_DIRECT_QPOS": "0",
        "HANDSON_ROKAE_WUJI_ARM_ACTUATOR": "1",
        "HANDSON_WUJI_GRASP_FORCE_SCALE": "2.5",
        "HAND": "wuji1",
    }
    assert not (tmp_path / ".replay_manifest.json.tmp").exists()
    assert not (tmp_path / ".model.mjb.tmp").exists()
    assert not (tmp_path / ".final_model.mjb.tmp").exists()


def test_replay_assets_gracefully_mark_non_mujoco_environment(tmp_path, monkeypatch) -> None:
    # CI may place TMPDIR inside its checkout. This fixture must remain a
    # non-repository even when Git could otherwise discover that parent.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    (tmp_path / "recording_schema.json").write_text("{}\n", encoding="utf-8")

    manifest_path = write_replay_assets(
        tmp_path,
        object(),
        task_name="non_mujoco_task",
        recording_parameters={"sim_backend": "isaacsim"},
        repo_root=tmp_path,
        environ={"UNRELATED_SECRET": "not-recorded"},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_asset = manifest["assets"]["compiled_model"]
    assert model_asset["status"] == "skipped"
    assert model_asset["path"] is None
    assert model_asset["sha256"] is None
    assert "No mujoco.MjModel" in model_asset["reason"]
    assert manifest["environment"]["variables"] == {}
    assert manifest["software"]["git"]["status"] == "unavailable"
    assert not (tmp_path / "model.mjb").exists()


def test_replay_assets_keep_schema_baseline_when_runtime_model_changes(tmp_path) -> None:
    model = _tiny_compiled_model()
    baseline_bytes = _compiled_model_bytes(model)
    baseline_sha256 = hashlib.sha256(baseline_bytes).hexdigest()
    (tmp_path / "recording_schema.json").write_text("{}\n", encoding="utf-8")

    class Telemetry:
        def compiled_model_bytes(self):
            return baseline_bytes

    class Environment:
        def __init__(self):
            self.model = model
            self._c01_force_telemetry = Telemetry()

    # Plan1 squeeze tuning can mutate compiled actuator/friction parameters.
    # Use timestep as a compact stand-in proving the final MJB differs.
    model.opt.timestep *= 2.0
    final_sha256 = _compiled_model_sha256(model)
    write_replay_assets(
        tmp_path,
        Environment(),
        task_name="plan1_hammer_nail",
        recording_parameters={},
        environment_schema={
            "mujoco": {"compiled_model_sha256": baseline_sha256}
        },
        repo_root=tmp_path,
        environ={},
    )

    manifest = json.loads(
        (tmp_path / "replay_manifest.json").read_text(encoding="utf-8")
    )
    assert hashlib.sha256((tmp_path / "model.mjb").read_bytes()).hexdigest() == baseline_sha256
    assert hashlib.sha256((tmp_path / "final_model.mjb").read_bytes()).hexdigest() == final_sha256
    assert manifest["assets"]["compiled_model"]["matches_environment_schema"] is True
    assert manifest["assets"]["final_compiled_model"] == {
        "status": "saved",
        "role": "runtime_final",
        "path": "final_model.mjb",
        "sha256": final_sha256,
    }


def test_replay_assets_reject_model_hash_mismatch_after_recording_it(tmp_path) -> None:
    model = _tiny_compiled_model()
    baseline_bytes = _compiled_model_bytes(model)
    (tmp_path / "recording_schema.json").write_text("{}\n", encoding="utf-8")

    class Telemetry:
        def compiled_model_bytes(self):
            return baseline_bytes

    class Environment:
        def __init__(self):
            self.model = model
            self._c01_force_telemetry = Telemetry()

    with pytest.raises(ValueError, match="does not match"):
        write_replay_assets(
            tmp_path,
            Environment(),
            task_name="plan1_hammer_nail",
            recording_parameters={},
            environment_schema={
                "mujoco": {"compiled_model_sha256": "0" * 64}
            },
            repo_root=tmp_path,
            environ={},
        )

    manifest = json.loads(
        (tmp_path / "replay_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["assets"]["compiled_model"]["matches_environment_schema"] is False
