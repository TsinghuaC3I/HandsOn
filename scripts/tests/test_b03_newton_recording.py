from scripts import record_demos_zarr as recorder
import types


def test_b03_is_a_newton_collection_task_without_removing_a01() -> None:
    assert recorder.NEWTON_COLLECTION_TASKS == frozenset(
        {"t26_place_fruit_in_bucket", "t09_insert_peg"}
    )


def test_b03_newton_replay_contract_uses_per_substep_drive_targets() -> None:
    contract = recorder._physical_replay_recording_contract(
        recorded_drive_targets=True,
        physics_backend="newton",
    )

    assert contract["physical_transition_contract"] == (
        "seeded_fresh_newton_reset_plus_per_substep_drive_targets_v1"
    )
    assert contract["trajectory_input_mode"] == (
        "recorded_newton_substep_robot_joint_position_targets"
    )
    assert contract["physical_replay_drive_target_source"] == (
        "exact_recorded_per_newton_substep_robot_position_targets"
    )


def test_b03_cross_gpu_contract_and_dataset_generation_are_isolated_from_v1() -> None:
    contract = recorder._physical_replay_recording_contract(
        recorded_drive_targets=True,
        physics_backend="newton",
        newton_cross_gpu=True,
    )
    env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            _physics_backend="newton",
            task_name="t09_insert_peg",
        )
    )

    generation = recorder._dataset_generation_for_env(env)

    assert contract["physical_transition_contract"].endswith("cross_gpu_v2")
    assert generation.name == "nvidia_newton_cross_gpu_v2"
    assert generation.directory_suffix == "__newton_v2"
    assert generation.marker_file == "DATASET_NEWTON_CROSS_GPU_V2.json"
    assert generation.training_status == "blocked_until_cross_gpu_certified"
    assert generation.directory_suffix in (
        recorder.DATASET_GENERATION_DIRECTORY_SUFFIXES
    )
    assert recorder._dataset_generation_suffix_for_episode(
        "b03_demo__newton_v2"
    ) == "__newton_v2"
