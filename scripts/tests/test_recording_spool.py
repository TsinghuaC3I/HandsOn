from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from absl.testing import flagsaver

from handson.runtime.collection_support import (
    ISAACSIM_COLLECTION_TASKS,
    ISAACSIM_DEFERRED_CAMERA_RENDER_TASKS,
    ISAACSIM_FORMAL_COLLECTION_TASKS,
)
from scripts import live_camera_viewer
from scripts import record_demos_zarr as recorder


def test_fu015_scripted_action_is_grouped_by_side_before_storage() -> None:
    action = np.arange(54, dtype=np.float64)

    grouped = recorder._group_fu015_scripted_action(
        action,
        hand_action_dim=20,
    )

    np.testing.assert_array_equal(
        grouped["right"],
        np.concatenate([action[:7], action[14:34]]),
    )
    np.testing.assert_array_equal(
        grouped["left"],
        np.concatenate([action[7:14], action[34:54]]),
    )
    stored = recorder._flatten_action_for_storage(grouped)
    assert recorder._action_storage_layout(stored.shape) == (
        "right_pose_hand_then_left_pose_hand"
    )
    np.testing.assert_array_equal(stored[:27], grouped["right"])
    np.testing.assert_array_equal(stored[27:], grouped["left"])


@pytest.mark.parametrize(
    ("preview_only", "backend", "ground_truth", "deferred", "fresh", "expected"),
    [
        (False, "isaacsim", True, True, False, True),
        (False, "isaacsim", True, False, True, True),
        (False, "isaacsim", True, False, False, False),
        (True, "isaacsim", True, False, True, False),
        (False, "isaacsim_newton", True, False, True, False),
        (False, "isaacsim", False, False, True, False),
    ],
)
def test_session_history_covers_deferred_and_fresh_certified_workers(
    preview_only: bool,
    backend: str,
    ground_truth: bool,
    deferred: bool,
    fresh: bool,
    expected: bool,
) -> None:
    assert recorder._physical_replay_session_history_required(
        preview_only=preview_only,
        sim_backend=backend,
        record_physx_ground_truth=ground_truth,
        deferred_certification=deferred,
        fresh_certified_worker=fresh,
    ) is expected


def _solid_rgb(value: int, *, height: int = 48, width: int = 64) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_isaac_recorder_marks_wrapped_environment_for_fast_shutdown() -> None:
    raw = SimpleNamespace(_simulation_app=object(), _fast_shutdown_on_close=False)
    wrapped = SimpleNamespace(unwrapped=raw)

    assert recorder._enable_isaac_fast_shutdown(wrapped) is True
    assert raw._fast_shutdown_on_close is True


def test_online_camera_capture_scope_is_fail_closed() -> None:
    assert recorder._online_camera_capture_scope(
        "t09_insert_peg",
        "isaacsim_newton",
        deferred_arm_env_value="0",
        live_main_env_value="1",
    ) == "main_live_all_canonical_postrender"
    assert recorder._online_camera_capture_scope(
        "t09_insert_peg",
        "isaacsim_newton",
        deferred_arm_env_value="1",
        live_main_env_value="0",
    ) == "all_canonical_live"
    assert recorder._online_camera_capture_scope(
        "t09_insert_peg",
        "isaacsim_newton",
        deferred_arm_env_value="0",
        deferred_camera_env_value="1",
        live_main_env_value="0",
    ) == "all_canonical_postrender"
    assert recorder._online_camera_capture_scope(
        "t21_pass_block",
        "isaacsim",
        deferred_arm_env_value="1",
        deferred_camera_env_value="1",
        live_main_env_value="0",
    ) == "all_canonical_postrender"
    assert recorder._online_camera_capture_scope(
        "t10_lift_carton",
        "isaacsim",
        deferred_arm_env_value="1",
        deferred_camera_env_value="1",
        live_main_env_value="0",
    ) == "all_canonical_postrender"
    assert recorder._uses_isaac_deferred_arm_pipeline(
        "t10_lift_carton",
        "isaacsim",
        "1",
    ) is True
    assert recorder._online_camera_capture_scope(
        "loaded_tray_through_narrow_gate",
        "isaacsim",
        deferred_arm_env_value="1",
        deferred_camera_env_value="1",
        live_main_env_value="0",
    ) == "all_canonical_postrender"
    assert recorder._uses_isaac_deferred_arm_pipeline(
        "loaded_tray_through_narrow_gate",
        "isaacsim",
        "1",
    ) is True
    assert recorder._online_camera_capture_scope(
        "t15_open_drawers",
        "isaacsim",
        deferred_arm_env_value="1",
        deferred_camera_env_value="1",
        live_main_env_value="0",
    ) == "all_canonical_postrender"
    assert recorder._uses_isaac_deferred_arm_pipeline(
        "t15_open_drawers",
        "isaacsim",
        "1",
    ) is True
    with pytest.raises(ValueError, match="cannot defer all"):
        recorder._online_camera_capture_scope(
            "t09_insert_peg",
            "isaacsim_newton",
            deferred_arm_env_value="0",
            deferred_camera_env_value="1",
            live_main_env_value="1",
        )
    with pytest.raises(ValueError, match="live-arm B03 Newton"):
        recorder._online_camera_capture_scope(
            "t09_insert_peg",
            "isaacsim_newton",
            deferred_arm_env_value="1",
            live_main_env_value="1",
        )
    with pytest.raises(ValueError, match="live-arm B03 Newton"):
        recorder._online_camera_capture_scope(
            "t26_place_fruit_in_bucket",
            "isaacsim_newton",
            deferred_arm_env_value="0",
            live_main_env_value="1",
        )
    assert recorder._online_camera_capture_scope(
        "t06_fit_device_cover",
        "isaacsim",
        deferred_arm_env_value="0",
        deferred_camera_env_value="1",
        live_main_env_value="0",
    ) == "all_canonical_postrender"
    with pytest.raises(ValueError, match="registered formal Isaac PhysX"):
        recorder._online_camera_capture_scope(
            "not_a_registered_task",
            "isaacsim",
            deferred_arm_env_value="0",
            deferred_camera_env_value="1",
            live_main_env_value="0",
        )
    with pytest.raises(ValueError, match="Unsupported writable"):
        recorder._validated_online_camera_capture_scope(
            "all_canoncal_postrender"
        )


def test_every_formal_isaac_task_uses_the_same_camera_postrender_contract() -> None:
    assert ISAACSIM_DEFERRED_CAMERA_RENDER_TASKS == (
        ISAACSIM_FORMAL_COLLECTION_TASKS
    )
    assert ISAACSIM_DEFERRED_CAMERA_RENDER_TASKS == ISAACSIM_COLLECTION_TASKS
    for task_name in sorted(ISAACSIM_FORMAL_COLLECTION_TASKS):
        assert recorder._online_camera_capture_scope(
            task_name,
            "isaacsim",
            deferred_arm_env_value="0",
            deferred_camera_env_value="1",
            live_main_env_value="0",
        ) == recorder.ONLINE_CAMERA_SCOPE_ALL_POSTRENDER


def test_online_camera_spool_contract_rejects_both_mode_mismatches() -> None:
    with pytest.raises(RuntimeError, match="does not match"):
        recorder._online_camera_spool_contract(
            recorder.ONLINE_CAMERA_SCOPE_ALL_POSTRENDER,
            SimpleNamespace(encode_videos=True),
            save_depth=False,
        )
    with pytest.raises(RuntimeError, match="does not match"):
        recorder._online_camera_spool_contract(
            recorder.ONLINE_CAMERA_SCOPE_ALL_LIVE,
            SimpleNamespace(encode_videos=False),
            save_depth=False,
        )


class _FakeWriter:
    def __init__(self):
        self.frames = []
        self.output = None
        self.stopped = False

    def start(self, output):
        self.output = Path(output)
        self.output.write_bytes(b"fake h264 mp4")

    def write_frame(self, frame):
        self.frames.append(np.asarray(frame).copy())

    def stop(self):
        self.stopped = True


class _PublishedOneFrameSpool:
    depth_frame_count = 0
    video_crf = 18

    def __init__(self, frame_count: int = 1):
        self.frame_count = int(frame_count)

    def publish_videos(self, videos_dir):
        videos_dir = Path(videos_dir)
        videos_dir.mkdir(parents=True)
        frame = _solid_rgb(10)
        sources = {}
        first_frames = {}
        for key in recorder.CANONICAL_CAMERA_KEYS:
            (videos_dir / f"{key}.mp4").write_bytes(b"complete video")
            sources[key] = key
            first_frames[key] = frame
        return sources, first_frames


def _episode_test_flags(*, sim_backend: str) -> SimpleNamespace:
    return SimpleNamespace(
        arm_type="rokae",
        auto_c01_arm_video=True,
        auto_plan_arm_video=True,
        c01_arm_collision_penetration_tolerance=0.001,
        c01_arm_collision_sweep_joint_step=0.05,
        c01_arm_max_fk_orientation_error=0.1,
        c01_arm_max_fk_position_error=0.01,
        camera_screen_effect=False,
        data_fps=30.0,
        fisheye_intrinsics_yaml="",
        hand_type="wuji1",
        ipad_screen_effect=False,
        randomize=False,
        randomize_dynamics=False,
        record_physx_ground_truth=False,
        render_mode="rgb_array",
        save_camera_info=False,
        save_depth=False,
        scripted_policy=False,
        sim_backend=sim_backend,
        target_label="",
    )


def _write_one_frame_test_episode(
    tmp_path: Path,
    *,
    task_name: str,
    manus_raw_events: list[dict[str, object]] | None = None,
    manus_raw_capture_summary: dict[str, object] | None = None,
    trajectory: list[dict[str, object]] | None = None,
) -> Path:
    env = _stable_v2_env(task_name=task_name)
    env.unwrapped.arm_type = "rokae"
    env.unwrapped.hand_type = "wuji1"
    env.unwrapped.image_obs = True
    generation = recorder._dataset_generation_for_env(env)
    episode_id = f"{task_name}_demo_1_timestamp__stable_v2"
    demo_dir = tmp_path / episode_id
    if trajectory is None:
        trajectory = [
            {
                "observations": {},
                "next_observations": {},
                "actions": np.zeros(3, dtype=np.float64),
                "dones": True,
                "infos": {"succeed": True},
            }
        ]
    recorder._write_demo_contents(
        trajectory,
        task_name,
        1,
        demo_dir,
        30,
        env=env,
        environment_schema={},
        required_recording_fields=set(),
        video_spool=_PublishedOneFrameSpool(len(trajectory)),
        collection_run_seed=7,
        episode_reset_seed=8,
        collection_seed_source=recorder.RANDOM_COLLECTION_SEED_SOURCE,
        episode_reset_ordinal=1,
        dataset_generation=generation,
        published_demo_name=episode_id,
        manus_raw_events=manus_raw_events,
        manus_raw_capture_summary=manus_raw_capture_summary,
    )
    return demo_dir


def test_episode_records_real_wall_cadence_and_simulation_data_rate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recorder,
        "FLAGS",
        _episode_test_flags(sim_backend="mujoco"),
    )
    trajectory = [
        {
            "observations": {},
            "next_observations": {},
            "actions": np.zeros(3, dtype=np.float64),
            "dones": index == 30,
            "infos": {
                "succeed": index == 30,
                "timestamp": index * 0.02,
            },
            recorder.RECORDER_MONOTONIC_TIME_KEY: 1000.0 + index / 20.0,
        }
        for index in range(31)
    ]

    demo_dir = _write_one_frame_test_episode(
        tmp_path,
        task_name="a01_coffee_pod",
        trajectory=trajectory,
    )

    metadata = json.loads(
        (demo_dir / "collection_meta.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (demo_dir / "replay_manifest.json").read_text(encoding="utf-8")
    )
    root = zarr.open_group(str(demo_dir / "replay.zarr"), mode="r")
    np.testing.assert_allclose(
        root["data/wall_timestamp"][:],
        np.arange(31, dtype=np.float64) / 20.0,
        atol=1e-12,
    )
    assert metadata["data_fps"] == pytest.approx(50.0)
    assert metadata["requested_data_fps"] == pytest.approx(30.0)
    assert metadata["timestamp_source"] == (
        "environment_info_or_fixed_rate_fallback"
    )
    assert metadata["wall_clock_sampling"]["mean_hz"] == pytest.approx(20.0)
    assert metadata["wall_clock_sampling"]["qualified"] is False
    assert manifest["recording"]["wall_clock_sampling"] == metadata[
        "wall_clock_sampling"
    ]


def test_incomplete_manus_sidecar_is_saved_without_rejecting_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recorder,
        "FLAGS",
        _episode_test_flags(sim_backend="mujoco"),
    )
    summary = {
        "format": recorder.MANUS_RAW_FORMAT,
        "version": recorder.MANUS_RAW_VERSION,
        "event_count": 0,
        "topics": [],
        "sides": [],
        "sequence_gap_count": 0,
        "sequence_gaps": [],
        "receiver_error_count": 1,
        "receiver_errors": ["port unavailable"],
        "complete": False,
        "status": "recorder_unavailable",
    }

    demo_dir = _write_one_frame_test_episode(
        tmp_path,
        task_name="a01_coffee_pod",
        manus_raw_events=[],
        manus_raw_capture_summary=summary,
    )

    manifest = json.loads(
        (demo_dir / "replay_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["recording"]["manus_raw_recording"]["complete"] is False
    assert manifest["recording"]["manus_raw_recording"]["status"] == (
        "recorder_unavailable"
    )
    assert (demo_dir / recorder.MANUS_RAW_SCHEMA_FILE).is_file()
    root = zarr.open_group(str(demo_dir / "replay.zarr"), mode="r")
    assert root["manus_raw/source_sequence"].shape == (0,)


def test_recorder_uses_authoritative_isaac_collection_allowlist() -> None:
    assert recorder.ISAACSIM_COLLECTION_TASKS is ISAACSIM_COLLECTION_TASKS
    assert "d01_bimanual_nut_bolt_assembly" in recorder.ISAACSIM_COLLECTION_TASKS


@pytest.mark.parametrize(
    ("sim_backend", "field_name"),
    [
        ("isaacsim", "physx_sim_time"),
        ("isaacsim_newton", "newton_sim_time"),
    ],
)
def test_recording_sim_time_uses_backend_specific_field(
    sim_backend: str,
    field_name: str,
) -> None:
    trace = SimpleNamespace(
        snapshot=lambda: {field_name: np.asarray([1.25], dtype=np.float64)}
    )
    raw_env = SimpleNamespace(_physx_ground_truth_trace=trace)

    assert recorder._recording_sim_time(
        raw_env,
        sim_backend=sim_backend,
    ) == pytest.approx(1.25)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"newton_sim_time": None},
        {"newton_sim_time": np.asarray(["not-a-number"], dtype=object)},
        {"newton_sim_time": np.asarray([1.0, 2.0])},
        {"newton_sim_time": np.asarray([np.nan])},
    ],
)
def test_recording_sim_time_rejects_missing_or_invalid_values(payload) -> None:
    trace = SimpleNamespace(snapshot=lambda: payload)
    raw_env = SimpleNamespace(_physx_ground_truth_trace=trace)

    assert recorder._recording_sim_time(
        raw_env,
        sim_backend="isaacsim_newton",
    ) is None


def test_recording_sim_time_treats_snapshot_failure_as_unavailable() -> None:
    def fail_snapshot():
        raise RuntimeError("trace is not ready")

    trace = SimpleNamespace(snapshot=fail_snapshot)
    raw_env = SimpleNamespace(_physx_ground_truth_trace=trace)

    assert recorder._recording_sim_time(
        raw_env,
        sim_backend="isaacsim_newton",
    ) is None


def test_newton_reset_accepts_newton_sim_time_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = SimpleNamespace(
        snapshot=lambda: {
            "newton_sim_time": np.asarray([2.5], dtype=np.float64)
        }
    )
    raw_env = SimpleNamespace(
        task_name="t09_insert_peg",
        _physx_ground_truth_trace=trace,
    )
    env = SimpleNamespace(unwrapped=raw_env)
    reset_calls = []

    def fake_reset(_env, kwargs):
        reset_calls.append(dict(kwargs))
        return {"reset": True}, {"reset": True}

    monkeypatch.setattr(recorder, "_reset_recording_episode", fake_reset)

    observation, info, next_index, episode_seed = (
        recorder._reset_recording_episode_with_stable_origin(
            env,
            {},
            sim_backend="isaacsim_newton",
            task_id="t09_insert_peg",
            collection_run_seed=100,
            reset_attempt_index=0,
            seed_resets=True,
            reset_attempt_observer=lambda _event: None,
        )
    )

    assert reset_calls == [{"seed": 100}]
    assert observation == {"reset": True}
    assert info == {"reset": True}
    assert next_index == 1
    assert episode_seed == 100


def test_genesis_engine_label_distinguishes_live_and_offline_phases() -> None:
    assert recorder.ENGINE_LABELS["E03"] == "Genesis（旧 PBD 诊断 / IPC 离线）"


def test_tasks_use_declared_stable_origin_profiles() -> None:
    expected_abc = {
        "stability_metric": "velocity",
        "initial_settle_seconds": 1.0,
        "post_settle_velocity_zeroing": True,
        "post_settle_sleeping": False,
        "stabilization_velocity_clamp": False,
        "stabilization_velocity_clamp_scope": "none",
        "stability_sample_seconds": 0.0,
        "reset_translation_drift_limit": 0.005,
        "reset_rotation_drift_limit": pytest.approx(0.05235987755982989),
        "require_valid_contact_report": True,
        "contact_penetration_limit": 0.0005,
        "timeout_seconds": 39.0,
    }
    expected_static_pose_fallback = {
        **expected_abc,
        "stability_metric": "velocity_or_static_pose",
        "pose_translation_limit": 1.0e-6,
        "pose_rotation_limit": 1.0e-5,
    }
    for task_name in (
        "t26_place_fruit_in_bucket",
        "t19_orient_cup_upright",
        "t04_cover_pitcher",
        "t44_stack_blocks",
        "t36_scoop_marbles",
        "t50_water_plant",
        "c07_pitcher_pour_cup",
        "c08_tongs_ice_cubes",
        "t11_line_up_blocks",
    ):
        assert (
            recorder.stable_recording_origin_kwargs_for_task(task_name)
            == expected_static_pose_fallback
        )
    assert recorder.stable_recording_origin_reset_attempts_for_task(
        "t26_place_fruit_in_bucket"
    ) == 12
    assert recorder.stable_recording_origin_kwargs_for_task(
        "t09_insert_peg"
    ) == expected_static_pose_fallback
    for task_name in ISAACSIM_COLLECTION_TASKS:
        profile = recorder.stable_recording_origin_kwargs_for_task(task_name)
        if task_name == "t20_pass_and_orient_pen":
            assert profile["initial_settle_seconds"] == pytest.approx(1.0 / 120.0)
            assert profile["stabilization_velocity_clamp"] is True
            assert profile["stabilization_velocity_clamp_scope"] == "gate_only"
        elif task_name in {
            "t45_stir_pot",
            "loaded_tray_through_narrow_gate",
            "t27_place_largest_fitting_ball",
            "t41_sort_nested_cups",
        }:
            assert profile["initial_settle_seconds"] == pytest.approx(1.0)
            assert profile["stabilization_velocity_clamp"] is True
            assert profile["stabilization_velocity_clamp_scope"] == "gate_only"
        else:
            assert profile["stabilization_velocity_clamp"] is False
            assert profile["stabilization_velocity_clamp_scope"] == "none"
    assert recorder.stable_recording_origin_kwargs_for_task(
        "t12_load_and_start_microwave"
    )["post_settle_sleeping"] is False
    assert recorder.stable_recording_origin_kwargs_for_task(
        "t21_pass_block"
    )["post_settle_sleeping"] is True
    for task_name in (
        "t42_sort_objects",
        "d07_organize_office_desk",
        "d08_toast_bread",
    ):
        profile = recorder.stable_recording_origin_kwargs_for_task(task_name)
        assert profile["post_settle_sleeping"] is False
        assert profile["stability_metric"] == "velocity_or_static_pose"
        assert profile["pose_translation_limit"] == pytest.approx(1.0e-6)
        assert profile["pose_rotation_limit"] == pytest.approx(1.0e-5)
        assert profile["timeout_seconds"] == pytest.approx(8.0)


def _stable_v2_env(
    *,
    observed: int = 60,
    stability_metric: str = "velocity",
    task_name: str = "t50_water_plant",
) -> SimpleNamespace:
    profile = recorder.stable_recording_origin_kwargs_for_task(task_name)
    post_settle_sleeping = bool(profile["post_settle_sleeping"])
    velocity_clamp = bool(profile["stabilization_velocity_clamp"])
    initial_settle_steps = 1 if velocity_clamp else 120
    physics_steps = 61 if velocity_clamp else 180
    velocity_clamp_steps = (
        physics_steps - initial_settle_steps if velocity_clamp else 0
    )
    stability_sample_steps = (
        1 if profile.get("stability_sample_seconds") == 0.0 else 6
    )
    sleep_report = {
        "rigid_bodies": 1,
        "body_paths": ["/World/Task/body"],
        "skipped_non_dynamic_or_non_path_names": [],
    }
    raw_env = SimpleNamespace(
        _physical_recording_origin={
            "strategy": recorder.STABLE_V2_ORIGIN_STRATEGY,
            "stability_metric": stability_metric,
            "task_name": task_name,
            "state_writes": True,
            "per_frame_state_writes": False,
            "stable_steps_required": 60,
            "stable_steps_observed": observed,
            "joint_velocity_limit": 0.001,
            "full_frame_pose_gate": (
                stability_metric == "velocity_or_static_pose"
            ),
            "stabilization_velocity_clamp": velocity_clamp,
            "stabilization_velocity_clamp_scope": profile[
                "stabilization_velocity_clamp_scope"
            ],
            "initialization_per_physics_step_velocity_writes": velocity_clamp,
            "physics_steps": physics_steps,
            "initial_settle_steps": initial_settle_steps,
            "stabilization_velocity_clamp_steps": velocity_clamp_steps,
            "stabilization_velocity_clamp_write": (
                {"rigid_bodies": 1} if velocity_clamp else None
            ),
            "post_settle_velocity_zeroing": True,
            "post_settle_sleeping": post_settle_sleeping,
            "post_settle_sleep": (
                sleep_report if post_settle_sleeping else None
            ),
            "final_boundary_sleep": (
                sleep_report if post_settle_sleeping else None
            ),
            "boundary_state_write_mode": "velocity_only",
            "contact_warm_start_preserved": not post_settle_sleeping,
            "final_boundary_task_and_joint_velocity_exact_zero": True,
            "stability_sample_steps": stability_sample_steps,
        }
    )
    return SimpleNamespace(unwrapped=raw_env)


def test_stable_v2_generation_requires_the_complete_stability_contract(
    tmp_path: Path,
) -> None:
    generation = recorder._dataset_generation_for_env(_stable_v2_env())

    assert generation.is_stable_v2 is True
    assert generation.name == "stable_origin_v2"
    assert generation.training_status == "new_requires_physical_certification"
    assert generation.directory_suffix == "__stable_v2"
    assert generation.marker_file == "DATASET_STABLE_V2.json"

    marker = recorder._write_dataset_generation_marker(
        tmp_path,
        generation,
        episode_id="t50_water_plant_demo_1_timestamp__stable_v2",
    )
    assert marker == tmp_path / "DATASET_STABLE_V2.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["dataset_generation"] == "stable_origin_v2"
    assert payload["is_stable_v2_data"] is True
    assert payload["old_data_policy"] == (
        "episodes_without_this_marker_are_legacy_invalid_do_not_train"
    )
    assert payload["stable_recording_origin"]["stable_steps_observed"] == 60

    incomplete = recorder._dataset_generation_for_env(
        _stable_v2_env(observed=59)
    )
    assert incomplete.is_stable_v2 is False
    assert incomplete.name == "legacy_pre_stable_v2"
    assert incomplete.training_status == "legacy_invalid_do_not_train"
    assert recorder._write_dataset_generation_marker(
        tmp_path / "legacy",
        incomplete,
        episode_id="t50_water_plant_demo_legacy",
    ) is None

    pose_origin = _stable_v2_env()
    pose_origin.unwrapped._physical_recording_origin[
        "stability_metric"
    ] = "pose"
    assert recorder._dataset_generation_for_env(
        pose_origin
    ).is_stable_v2 is False

    for task_name in recorder.STATIC_POSE_FALLBACK_TASKS:
        approved_hybrid = _stable_v2_env(
            stability_metric="velocity_or_static_pose",
            task_name=task_name,
        )
        assert (
            recorder._dataset_generation_for_env(approved_hybrid).is_stable_v2
            is True
        )

    unapproved_hybrid = _stable_v2_env(
        stability_metric="velocity_or_static_pose",
        task_name="b02_nut_bolt_assembly",
    )
    assert (
        recorder._dataset_generation_for_env(unapproved_hybrid).is_stable_v2
        is False
    )


def test_bc011_gate_only_velocity_zeroing_remains_stable_v2() -> None:
    env = _stable_v2_env(
        task_name="t20_pass_and_orient_pen"
    )
    origin = env.unwrapped._physical_recording_origin

    assert origin["initial_settle_steps"] == 1
    assert origin["stabilization_velocity_clamp_scope"] == "gate_only"
    assert origin["stabilization_velocity_clamp_steps"] == 60
    assert recorder._dataset_generation_for_env(env).is_stable_v2 is True


def test_manual_reset_auto_start_skips_the_teleop_delay(monkeypatch) -> None:
    calls = []

    class TeleopWrapper:
        intervened = False

        def set_intervened(self, enabled, *, start_delay=True):
            calls.append((enabled, start_delay))

    monkeypatch.setenv("HANDSON_AUTO_START_TELEOP", "1")

    recorder._maybe_auto_start_teleop(TeleopWrapper(), immediate=True)

    assert calls == [(True, False)]


def test_immediate_keyboard_reset_repeats_the_same_recorded_seed() -> None:
    class Environment:
        def __init__(self) -> None:
            self.calls = []

        def reset(self, **kwargs):
            self.calls.append(dict(kwargs))
            if len(self.calls) == 1:
                return {}, {"manual_reset_applied": True}
            return {}, {"manual_reset_applied": False}

    env = Environment()
    recorder._reset_recording_episode(
        env,
        {"seed": 123, "target_label": "target"},
    )

    assert env.calls == [
        {"seed": 123, "target_label": "target"},
        {"seed": 123, "target_label": "target"},
    ]


def test_fresh_attempt_rejects_manual_reset_during_initial_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class Environment:
        def reset(self, **kwargs):
            calls.append(dict(kwargs))
            return {}, {"manual_reset_applied": True}

    class FastExitCalled(RuntimeError):
        pass

    monkeypatch.setenv(
        recorder._FRESH_CERTIFIED_COLLECTION_WORKER_ENV,
        "1",
    )
    monkeypatch.setattr(recorder.os, "write", lambda *_args: 0)
    monkeypatch.setattr(
        recorder.os,
        "_exit",
        lambda code: (_ for _ in ()).throw(FastExitCalled(str(code))),
    )

    with pytest.raises(FastExitCalled):
        recorder._reset_recording_episode(Environment(), {"seed": 123})

    assert calls == [{"seed": 123}]


def _write_certification_parent_fixture(tmp_path: Path) -> Path:
    demo_dir = tmp_path / "t09_insert_peg_demo_1"
    demo_dir.mkdir()
    (demo_dir / "replay_manifest.json").write_text(
        json.dumps({"environment": {"variables": {}}}),
        encoding="utf-8",
    )
    return demo_dir


def test_certification_parent_requires_a_fresh_passed_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_dir = _write_certification_parent_fixture(tmp_path)
    stale_report = demo_dir / recorder.PHYSICAL_REPLAY_REPORT_NAME
    stale_report.write_text('{"status": "passed"}', encoding="utf-8")

    def fake_run(*_args, **_kwargs):
        assert not stale_report.exists()
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(recorder.subprocess, "run", fake_run)
    with flagsaver.flagsaver(physical_replay_certification_timeout_seconds=5):
        with pytest.raises(RuntimeError, match="certification failed"):
            recorder._certify_saved_physx_replays([demo_dir])


def test_certification_parent_accepts_valid_report_not_returncode_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_dir = _write_certification_parent_fixture(tmp_path)

    def fake_run(*_args, **_kwargs):
        (demo_dir / recorder.PHYSICAL_REPLAY_REPORT_NAME).write_text(
            json.dumps({"status": "passed"}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=23, stdout="")

    monkeypatch.setattr(recorder.subprocess, "run", fake_run)
    monkeypatch.setattr(
        recorder,
        "physical_replay_certification_upload_blocker",
        lambda _demo_dir: None,
    )
    with flagsaver.flagsaver(physical_replay_certification_timeout_seconds=5):
        recorder._certify_saved_physx_replays([demo_dir])


def test_physx_v3_generation_requires_fresh_scene_and_complete_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _stable_v2_env()
    env.unwrapped.hard_reset_physx_scene = lambda *, seed: {
        "scene_rebuilt": True,
        "scene_generation": 1,
        "seed": seed,
    }

    monkeypatch.setenv(recorder.ISAAC_IN_PROCESS_HARD_RESET_ENV, "1")
    assert recorder._dataset_generation_for_env(env).is_physx_v3 is False

    monkeypatch.setenv(
        recorder.DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_ENV,
        "1",
    )
    monkeypatch.setenv("HANDSON_RECORD_PHYSX_GROUND_TRUTH", "1")
    generation = recorder._dataset_generation_for_env(env)

    assert generation.is_stable_v2 is True
    assert generation.is_physx_v3 is True
    assert generation.name == "physx_fresh_scene_v3"
    assert generation.directory_suffix == "__physx_v3"
    assert generation.marker_file == "DATASET_PHYSX_V3.json"

    episode_id = "t26_place_fruit_in_bucket_demo_1_timestamp__physx_v3"
    marker = recorder._write_dataset_generation_marker(
        tmp_path,
        generation,
        episode_id=episode_id,
    )
    assert marker == tmp_path / "DATASET_PHYSX_V3.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["episode_id"] == episode_id
    assert payload["dataset_generation"] == "physx_fresh_scene_v3"
    assert payload["dataset_directory_suffix"] == "__physx_v3"
    assert payload["is_stable_v2_data"] is True
    assert payload["is_physx_v3_data"] is True
    assert payload["old_data_policy"] == (
        "episodes_without_DATASET_PHYSX_V3_are_pre_fresh_scene_data_do_not_mix"
    )


def test_task_guidance_can_never_be_classified_as_physx_v3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _stable_v2_env()
    env.unwrapped._guidance_mode = True
    env.unwrapped.hard_reset_physx_scene = lambda *, seed: {
        "scene_rebuilt": True,
        "scene_generation": 1,
        "seed": seed,
    }
    monkeypatch.setenv(recorder.ISAAC_IN_PROCESS_HARD_RESET_ENV, "1")
    monkeypatch.setenv(
        recorder.DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_ENV,
        "1",
    )
    monkeypatch.setenv("HANDSON_RECORD_PHYSX_GROUND_TRUTH", "1")

    generation = recorder._dataset_generation_for_env(env)

    assert generation.name == recorder.LEGACY_DATASET_GENERATION
    assert generation.is_physx_v3 is False
    assert generation.marker_file == ""


def test_fu006_persistent_worker_is_v3_after_verified_hard_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _stable_v2_env(
        task_name="t46_store_charger",
        stability_metric="velocity_or_static_pose",
    )
    env.unwrapped.task_name = "t46_store_charger"
    env.unwrapped.hard_reset_physx_scene = lambda *, seed: {
        "scene_rebuilt": True,
        "scene_generation": 1,
        "seed": seed,
    }

    monkeypatch.setenv(recorder.ISAAC_IN_PROCESS_HARD_RESET_ENV, "1")
    monkeypatch.setenv(
        recorder.DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_ENV,
        "1",
    )
    monkeypatch.setenv("HANDSON_RECORD_PHYSX_GROUND_TRUTH", "1")

    assert recorder._isaac_in_process_hard_reset_enabled_for_task(
        env.unwrapped.task_name
    ) is True
    generation = recorder._dataset_generation_for_env(env)

    assert generation.is_physx_v3 is True
    assert generation.directory_suffix == "__physx_v3"
    assert generation.marker_file == "DATASET_PHYSX_V3.json"

    # The supervised one-episode fresh-process route remains a valid fallback.
    monkeypatch.delenv(recorder.DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_ENV)
    monkeypatch.setenv(
        recorder._FRESH_CERTIFIED_COLLECTION_WORKER_ENV,
        "1",
    )
    generation = recorder._dataset_generation_for_env(env)

    assert generation.is_physx_v3 is True
    assert generation.directory_suffix == "__physx_v3"
    assert generation.marker_file == "DATASET_PHYSX_V3.json"


def test_direct_certified_isaac_recorder_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="record_physx_ground_truth=true"):
        recorder._enforce_fresh_process_certified_collection_entrypoint(
            preview_only=False,
            sim_backend="isaacsim",
            record_physx_ground_truth=False,
            certify_physical_replay=False,
            fresh_worker_authorized=False,
        )

    with pytest.raises(
        RuntimeError,
        match="fresh_process_certified_collection.py",
    ):
        recorder._enforce_fresh_process_certified_collection_entrypoint(
            preview_only=False,
            sim_backend="isaacsim",
            record_physx_ground_truth=True,
            certify_physical_replay=True,
            fresh_worker_authorized=False,
        )

    with pytest.raises(RuntimeError, match="HANDSON_DEFER_PHYSICAL_REPLAY_CERTIFICATION"):
        recorder._enforce_fresh_process_certified_collection_entrypoint(
            preview_only=False,
            sim_backend="isaacsim",
            record_physx_ground_truth=True,
            certify_physical_replay=False,
            fresh_worker_authorized=False,
        )

    monkeypatch.setenv("HANDSON_DEFER_PHYSICAL_REPLAY_CERTIFICATION", "1")
    recorder._enforce_fresh_process_certified_collection_entrypoint(
        preview_only=False,
        sim_backend="isaacsim",
        record_physx_ground_truth=True,
        certify_physical_replay=False,
        fresh_worker_authorized=False,
    )

    recorder._enforce_fresh_process_certified_collection_entrypoint(
        preview_only=False,
        sim_backend="isaacsim",
        record_physx_ground_truth=True,
        certify_physical_replay=False,
        fresh_worker_authorized=True,
    )

    recorder._enforce_fresh_process_certified_collection_entrypoint(
        preview_only=True,
        sim_backend="isaacsim",
        record_physx_ground_truth=False,
        certify_physical_replay=False,
        fresh_worker_authorized=False,
    )

    recorder._enforce_fresh_process_certified_collection_entrypoint(
        preview_only=False,
        sim_backend="isaacsim",
        record_physx_ground_truth=False,
        certify_physical_replay=False,
        fresh_worker_authorized=False,
        guidance_mode=True,
    )


@pytest.mark.parametrize(
    "task_name",
    [
        "t10_lift_carton",
        "b320_towel_corner_synchronized_spread",
    ],
)
def test_isaac_main_reaches_kit_before_deferred_supervisor(
    task_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Released tasks reach Isaac setup before the deferred supervisor."""

    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "0")
    setup_calls: list[str] = []
    deferred_calls: list[str] = []

    def setup_probe() -> None:
        setup_calls.append("isaac_setup")
        raise RuntimeError("preview reached Isaac setup probe")

    monkeypatch.setattr(
        recorder,
        "_configure_current_isaac_process_environment",
        setup_probe,
    )
    monkeypatch.setattr(
        recorder,
        "_run_isaac_deferred_collection_supervisor",
        lambda *args, **kwargs: deferred_calls.append("deferred_supervisor"),
    )

    with flagsaver.as_parsed(
        exp_name=task_name,
        sim_backend="isaacsim",
        arm_type="rokae",
        preview_only="false",
        record_physx_ground_truth="true",
    ):
        with pytest.raises(RuntimeError, match="preview reached Isaac setup probe"):
            recorder.main(None)
    assert setup_calls == ["isaac_setup"]
    assert deferred_calls == []

    with flagsaver.as_parsed(
        exp_name=task_name,
        sim_backend="isaacsim",
        arm_type="rokae",
        preview_only="true",
    ):
        with pytest.raises(RuntimeError, match="preview reached Isaac setup probe"):
            recorder.main(None)
    assert setup_calls == ["isaac_setup", "isaac_setup"]
    assert deferred_calls == []


def test_bc004_deferred_render_contract_routes_formal_main_to_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_calls: list[str] = []
    deferred_calls: list[dict[str, object]] = []
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_ARM_RENDER", "1")
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "1")
    monkeypatch.setenv(
        recorder.DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_ENV,
        "1",
    )
    monkeypatch.setattr(
        recorder,
        "_configure_current_isaac_process_environment",
        lambda: setup_calls.append("isaac_setup"),
    )
    monkeypatch.setattr(
        recorder,
        "_run_isaac_deferred_collection_supervisor",
        lambda **kwargs: deferred_calls.append(kwargs),
    )

    with flagsaver.as_parsed(
        exp_name="t10_lift_carton",
        sim_backend="isaacsim",
        arm_type="rokae",
        preview_only="false",
        record_physx_ground_truth="true",
        certify_physical_replay="false",
        save_depth="false",
        out_dir=str(tmp_path / "demos"),
        video_fps="30",
    ):
        recorder.main(None)

    assert setup_calls == ["isaac_setup"]
    assert len(deferred_calls) == 1
    assert deferred_calls[0]["exp_name"] == (
        "t10_lift_carton"
    )


def test_manual_only_postprocess_authorizes_deferred_recording_without_autostart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        recorder.DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_ENV,
        raising=False,
    )
    monkeypatch.setenv(recorder.POSTPROCESS_MANUAL_ONLY_ENV, "1")

    assert recorder._postprocess_manual_only_enabled() is True
    assert recorder._deferred_physical_replay_certification_enabled() is True
    assert recorder._deferred_physical_replay_autostart_enabled() is False
    recorder._enforce_fresh_process_certified_collection_entrypoint(
        preview_only=False,
        sim_backend="isaacsim",
        record_physx_ground_truth=True,
        certify_physical_replay=False,
        fresh_worker_authorized=False,
    )

    monkeypatch.delenv(recorder.POSTPROCESS_MANUAL_ONLY_ENV)
    monkeypatch.setenv(
        recorder.DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_ENV,
        "1",
    )
    assert recorder._deferred_physical_replay_autostart_enabled() is False

    monkeypatch.setenv(
        recorder.DEFERRED_PHYSICAL_REPLAY_AUTOSTART_ENV,
        "1",
    )
    assert recorder._deferred_physical_replay_autostart_enabled() is True

    monkeypatch.setenv(
        recorder.DEFERRED_PHYSICAL_REPLAY_AUTOSTART_ENV,
        "0",
    )
    assert recorder._deferred_physical_replay_autostart_enabled() is False


def test_verified_v3_declaration_exists_only_for_manual_postprocess() -> None:
    assert recorder._manual_postprocess_publication_declaration(
        enabled=True
    ) == {
        "publication_contract": "handson.verified_episode.v3",
        "postprocess_policy": "manual_review_render_upload_v1",
        "postprocess_manual_only": True,
    }
    assert recorder._manual_postprocess_publication_declaration(
        enabled=False
    ) == {}


@pytest.mark.parametrize(
    ("task_name", "expected_status"),
    [
        ("t08_hammer_nail", "pending"),
        ("plan1_hammer_nail", "pending"),
        ("plan8_peg_in_socket", "pending"),
        ("t26_place_fruit_in_bucket", "not_required"),
    ],
)
def test_manual_offline_arm_declaration_is_uniform_for_c01_and_plan(
    task_name: str,
    expected_status: str,
) -> None:
    declaration = recorder._offline_arm_render_declaration(
        task_name,
        sim_backend="mujoco",
        manual_only=True,
        c01_enabled=True,
        plan_enabled=True,
        c01_env_value="1",
        plan_env_value="1",
    )

    assert declaration == {
        "offline_arm_render_deferred": expected_status == "pending",
        "offline_arm_render_status": expected_status,
    }
    assert recorder._offline_arm_render_declaration(
        task_name,
        sim_backend="mujoco",
        manual_only=False,
        c01_enabled=True,
        plan_enabled=True,
        c01_env_value="1",
        plan_env_value="1",
    ) == {}


@pytest.mark.parametrize(
    ("task_name", "sim_backend"),
    [
        ("e01_fold_garment", "genesis"),
        ("f01_cotton_cloth_mvp", "isaacsim"),
    ],
)
def test_manual_only_never_autostarts_e_or_f_postprocess(
    task_name: str,
    sim_backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(recorder.POSTPROCESS_MANUAL_ONLY_ENV, "1")

    assert not recorder._should_autostart_e01_postprocess(
        task_name,
        sim_backend=sim_backend,
        ipc_env_value="1",
        arm_env_value="1",
    )

    monkeypatch.delenv(recorder.POSTPROCESS_MANUAL_ONLY_ENV)
    assert recorder._should_autostart_e01_postprocess(
        task_name,
        sim_backend=sim_backend,
        ipc_env_value="1",
        arm_env_value="1",
    )


def test_recorder_manifest_describes_recorded_drive_target_certifier() -> None:
    contract = recorder._physical_replay_recording_contract()

    assert contract["physical_transition_contract"] == (
        recorder.RECORDED_DRIVE_TARGET_TRANSITION_CONTRACT
    )
    assert contract["trajectory_input_mode"] == (
        recorder.RECORDED_DRIVE_TARGET_INPUT_MODE
    )
    assert contract["physical_replay_action_source"] == (
        "exact_raw_environment_action_executed"
    )
    assert contract["recorded_drive_target_replay_required"] is True
    assert "not_recomputed" in contract["physical_replay_ik"]


def test_specialized_controller_can_retain_explicit_legacy_contract() -> None:
    contract = recorder._physical_replay_recording_contract(
        recorded_drive_targets=False,
    )

    assert contract["physical_transition_contract"] == (
        recorder.LEGACY_RAW_ACTION_TRANSITION_CONTRACT
    )
    assert contract["trajectory_input_mode"] == (
        recorder.LEGACY_RAW_ACTION_INPUT_MODE
    )
    assert contract["recorded_drive_target_replay_required"] is False


def test_ycb06_reset_panel_reports_retry_and_clears_on_acceptance(monkeypatch):
    panels, observed = [], []
    raw = SimpleNamespace(
        task_name="t28_place_tableware",
        set_recording_origin_progress=panels.append,
    )
    env = SimpleNamespace(unwrapped=raw)
    monkeypatch.delenv(recorder.ISAAC_IN_PROCESS_HARD_RESET_ENV, raising=False)
    monkeypatch.setattr(recorder, "_reset_recording_episode", lambda env, kwargs: ({}, {}))
    calls = 0

    def stabilize(env, observation, info, **kwargs):
        nonlocal calls
        calls += 1
        kwargs["progress_callback"]({"physics_steps": 120})
        if calls == 1:
            raise recorder.RecordingOriginStabilityError("invalid first layout")
        return observation, info

    monkeypatch.setattr(recorder, "stabilize_recording_origin", stabilize)
    recorder._reset_recording_episode_with_stable_origin(
        env, {}, sim_backend="isaacsim", task_id=raw.task_name,
        collection_run_seed=123, reset_attempt_index=0, seed_resets=True,
        stabilization_progress_observer=observed.append,
    )
    assert [item["attempt"] for item in observed] == [1, 2]
    assert all(item["attempts_allowed"] == 8 for item in observed)
    assert panels == [*observed, None]


def test_unstable_a_series_origin_resamples_with_the_next_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_env = SimpleNamespace(task_name="t25_place_cup_on_plate")
    env = SimpleNamespace(unwrapped=raw_env)
    reset_seeds = []
    reset_events = []
    stabilization_calls = 0

    def fake_reset(_env, kwargs):
        reset_seeds.append(kwargs["seed"])
        return {"seed": kwargs["seed"]}, {}

    def fake_stabilize(_env, observation, info, **kwargs):
        nonlocal stabilization_calls
        stabilization_calls += 1
        assert kwargs == {
            "progress_callback": None,
            "stability_metric": "velocity_or_static_pose",
            "initial_settle_seconds": 1.0,
            "post_settle_velocity_zeroing": True,
            "post_settle_sleeping": False,
            "stabilization_velocity_clamp": False,
            "stabilization_velocity_clamp_scope": "none",
            "stability_sample_seconds": 0.0,
            "pose_translation_limit": 1.0e-6,
            "pose_rotation_limit": 1.0e-5,
            "reset_translation_drift_limit": 0.005,
            "reset_rotation_drift_limit": pytest.approx(
                0.05235987755982989
            ),
            "require_valid_contact_report": True,
            "contact_penetration_limit": 0.0005,
            "timeout_seconds": 39.0,
        }
        if stabilization_calls < 3:
            raise recorder.RecordingOriginStabilityError(
                "still moving",
                diagnostics={
                    "failure_phase": "stability_timeout",
                    "physics_steps": 240,
                    "stability_samples": 60,
                    "hold_target_sync_passes": 242,
                },
            )
        return observation, {
            **info,
            "physical_recording_origin": {
                "physics_steps": 123,
            },
        }

    monkeypatch.setattr(recorder, "_reset_recording_episode", fake_reset)
    monkeypatch.setattr(recorder, "stabilize_recording_origin", fake_stabilize)

    observation, info, next_index, episode_seed = (
        recorder._reset_recording_episode_with_stable_origin(
            env,
            {},
            sim_backend="isaacsim",
            task_id="t25_place_cup_on_plate",
            collection_run_seed=100,
            reset_attempt_index=0,
            seed_resets=True,
            reset_attempt_observer=reset_events.append,
        )
    )

    assert reset_seeds == [100, 101, 102]
    assert observation == {"seed": 102}
    assert next_index == 3
    assert episode_seed == 102
    origin = info["physical_recording_origin"]
    assert origin["physics_steps"] == 123
    assert origin["reset_stabilization_attempt"] == 3
    assert origin["reset_stabilization_attempts_allowed"] == 3
    assert origin["collection_run_seed"] == 100
    assert origin["collection_seed_source"] == "recorded_collection_seed"
    assert origin["episode_reset_ordinal"] == 2
    assert origin["episode_reset_seed"] == 102
    assert origin["episode_seed_derivation"] == (
        recorder.EPISODE_SEED_DERIVATION
    )
    assert isinstance(origin["episode_reset_requested_at"], str)
    assert origin["scene_rebuilt_before_reset"] is False
    assert origin["scene_generation"] == 0
    assert origin["scene_rebuild_report"] is None
    assert raw_env._physical_recording_origin == info[
        "physical_recording_origin"
    ]
    assert [event["outcome"] for event in reset_events] == [
        "rejected_unstable_origin",
        "rejected_unstable_origin",
        "accepted",
    ]
    assert [event["episode_reset_ordinal"] for event in reset_events] == [
        0,
        1,
        2,
    ]
    assert [event["episode_reset_seed"] for event in reset_events] == [
        100,
        101,
        102,
    ]
    assert all(
        event["stability_diagnostics"]["failure_phase"]
        == "stability_timeout"
        for event in reset_events[:2]
    )
    assert all(
        event["stable_origin_kwargs"]["initial_settle_seconds"] == 1.0
        for event in reset_events
    )
    assert all(event["scene_rebuild_report"] is None for event in reset_events)
    assert reset_events[-1]["stable_recording_origin"]["physics_steps"] == 123


def test_persistent_isaac_resample_rebuilds_every_later_physx_scene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rebuild_seeds: list[int] = []
    reset_seeds: list[int] = []
    reset_events: list[dict[str, object]] = []
    stabilization_calls = 0

    class RawEnvironment:
        task_name = "t25_place_cup_on_plate"
        _physx_scene_generation = 0

        def hard_reset_physx_scene(self, *, seed: int) -> dict[str, object]:
            rebuild_seeds.append(seed)
            self._physx_scene_generation += 1
            return {
                "scene_rebuilt": True,
                "scene_generation": self._physx_scene_generation,
                "seed": seed,
                "process_restarted": False,
                "simulation_app_reused": True,
            }

    raw_env = RawEnvironment()
    env = SimpleNamespace(unwrapped=raw_env)

    def fake_reset(_env, kwargs):
        reset_seeds.append(kwargs["seed"])
        return {"seed": kwargs["seed"]}, {}

    def fake_stabilize(_env, observation, info, **_kwargs):
        nonlocal stabilization_calls
        stabilization_calls += 1
        if stabilization_calls < 3:
            raise recorder.RecordingOriginStabilityError(
                "still moving",
                diagnostics={
                    "failure_phase": "stability_timeout",
                    "physics_steps": 240,
                    "stability_samples": 60,
                    "hold_target_sync_passes": 242,
                },
            )
        return observation, {
            **info,
            "physical_recording_origin": {"physics_steps": 123},
        }

    monkeypatch.setenv(recorder.ISAAC_IN_PROCESS_HARD_RESET_ENV, "1")
    monkeypatch.setattr(recorder, "_reset_recording_episode", fake_reset)
    monkeypatch.setattr(recorder, "stabilize_recording_origin", fake_stabilize)

    _observation, info, next_index, episode_seed = (
        recorder._reset_recording_episode_with_stable_origin(
            env,
            {},
            sim_backend="isaacsim",
            task_id="t25_place_cup_on_plate",
            collection_run_seed=100,
            reset_attempt_index=0,
            seed_resets=True,
            reset_attempt_observer=reset_events.append,
        )
    )

    assert reset_seeds == [100, 101, 102]
    assert rebuild_seeds == [101, 102]
    assert next_index == 3
    assert episode_seed == 102
    assert [event["scene_rebuilt_before_reset"] for event in reset_events] == [
        False,
        True,
        True,
    ]
    assert [event["scene_generation"] for event in reset_events] == [0, 1, 2]
    origin = info["physical_recording_origin"]
    assert origin["scene_rebuilt_before_reset"] is True
    assert origin["scene_generation"] == 2
    assert origin["scene_rebuild_report"] == {
        "scene_rebuilt": True,
        "scene_generation": 2,
        "seed": 102,
        "process_restarted": False,
        "simulation_app_reused": True,
    }


def test_unstable_fresh_worker_fast_exits_without_kit_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_env = SimpleNamespace(task_name="t25_place_cup_on_plate")
    env = SimpleNamespace(unwrapped=raw_env)
    diagnostics = []

    class FastExitCalled(RuntimeError):
        pass

    monkeypatch.setenv(
        recorder._FRESH_CERTIFIED_COLLECTION_WORKER_ENV,
        "1",
    )
    monkeypatch.setattr(
        recorder,
        "stable_recording_origin_reset_attempts_for_task",
        lambda _task_name: 1,
    )
    monkeypatch.setattr(
        recorder,
        "_reset_recording_episode",
        lambda _env, _kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        recorder,
        "stabilize_recording_origin",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            recorder.RecordingOriginStabilityError("right arm still moving")
        ),
    )
    monkeypatch.setattr(
        recorder.os,
        "write",
        lambda fd, value: diagnostics.append((fd, value)) or len(value),
    )

    def fake_exit(code: int) -> None:
        raise FastExitCalled(str(code))

    monkeypatch.setattr(recorder.os, "_exit", fake_exit)

    with pytest.raises(
        FastExitCalled,
        match=str(recorder._FRESH_WORKER_RETRYABLE_ORIGIN_EXIT_CODE),
    ):
        recorder._reset_recording_episode_with_stable_origin(
            env,
            {},
            sim_backend="isaacsim",
            task_id="t25_place_cup_on_plate",
            collection_run_seed=100,
            reset_attempt_index=0,
            seed_resets=True,
        )

    assert diagnostics
    assert diagnostics[0][0] == 2
    assert b"right arm still moving" in diagnostics[0][1]
    assert b"without Isaac/Kit teardown" in diagnostics[0][1]


@pytest.mark.parametrize(
    "reason",
    [
        "operator_keyboard_reset",
        "teleoperation_manual_reset",
        "unsuccessful_terminal_transition",
    ],
)
def test_fresh_worker_rejects_any_episode_reset_instead_of_reusing_scene(
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> None:
    diagnostics = []

    class FastExitCalled(RuntimeError):
        pass

    monkeypatch.setenv(
        recorder._FRESH_CERTIFIED_COLLECTION_WORKER_ENV,
        "1",
    )
    monkeypatch.setattr(
        recorder.os,
        "write",
        lambda fd, value: diagnostics.append((fd, value)) or len(value),
    )
    monkeypatch.setattr(
        recorder.os,
        "_exit",
        lambda code: (_ for _ in ()).throw(FastExitCalled(str(code))),
    )

    with pytest.raises(
        FastExitCalled,
        match=str(recorder._FRESH_WORKER_RETRYABLE_ATTEMPT_EXIT_CODE),
    ):
        recorder._fast_exit_fresh_worker_for_rejected_attempt(reason)

    assert diagnostics
    assert reason.encode("utf-8") in diagnostics[0][1]
    assert b"new process/PhysX scene" in diagnostics[0][1]


def test_nonfresh_recorder_may_use_ordinary_in_process_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        recorder._FRESH_CERTIFIED_COLLECTION_WORKER_ENV,
        raising=False,
    )

    assert (
        recorder._fast_exit_fresh_worker_for_rejected_attempt(
            "operator_keyboard_reset"
        )
        is None
    )


def test_session_history_binding_uses_output_root_relative_path(
    tmp_path: Path,
) -> None:
    pending = tmp_path / recorder.PHYSICAL_REPLAY_PENDING_DIRECTORY
    demo = pending / "a01_demo_1__stable_v2"
    demo.mkdir(parents=True)
    (demo / "replay_manifest.json").write_text(
        json.dumps({"recording": {"episode_index": 1}}),
        encoding="utf-8",
    )
    (demo / "collection_meta.json").write_text(
        json.dumps({"collection_session_id": "session/one"}),
        encoding="utf-8",
    )
    history_path = recorder._physical_replay_session_history_path(
        tmp_path,
        "session/one",
    )
    history_path.mkdir(parents=True)
    journal = SimpleNamespace(path=history_path, session_id="session/one")
    finalization = {
        "event_count": 2,
        "history_identity": {"sha256": "a" * 64},
        "accepted_demos": [
            {
                "demo_id": demo.name,
                "event_ordinal": 1,
                "reset_ordinal": 0,
            }
        ],
        "certified_through_event_ordinal": 1,
    }

    recorder._bind_physx_session_history_to_demos(
        [demo],
        output_root=tmp_path,
        journal=journal,
        finalization=finalization,
    )

    manifest = json.loads(
        (demo / "replay_manifest.json").read_text(encoding="utf-8")
    )
    binding = manifest["recording"]
    assert binding["physical_replay_certification_scope"] == (
        "ordered_collection_session"
    )
    assert binding["physical_replay_session_history_id"] == "session/one"
    assert binding[
        "physical_replay_session_history_root_relative_path"
    ].startswith(
        f"{recorder.PHYSICAL_REPLAY_PENDING_DIRECTORY}/"
        f"{recorder.PHYSICAL_REPLAY_SESSION_HISTORY_DIRECTORY}/"
    )
    assert binding["physical_replay_session_history_sha256"] == "a" * 64
    assert binding["physical_replay_session_history_event_count"] == 2
    assert binding[
        "physical_replay_session_history_accepted_event_ordinal"
    ] == 1
    metadata = json.loads(
        (demo / "collection_meta.json").read_text(encoding="utf-8")
    )
    assert metadata["physical_replay_session_history_sha256"] == "a" * 64


def test_unstable_nonfresh_recorder_keeps_normal_exception_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        recorder._FRESH_CERTIFIED_COLLECTION_WORKER_ENV,
        raising=False,
    )
    monkeypatch.setattr(
        recorder,
        "stable_recording_origin_reset_attempts_for_task",
        lambda _task_name: 1,
    )
    monkeypatch.setattr(
        recorder,
        "_reset_recording_episode",
        lambda _env, _kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        recorder,
        "stabilize_recording_origin",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            recorder.RecordingOriginStabilityError("still moving")
        ),
    )
    monkeypatch.setattr(
        recorder.os,
        "_exit",
        lambda _code: pytest.fail("non-fresh recorder must not hard-exit"),
    )

    with pytest.raises(
        recorder.RecordingOriginStabilityError,
        match="still moving",
    ):
        recorder._reset_recording_episode_with_stable_origin(
            SimpleNamespace(
                unwrapped=SimpleNamespace(task_name="t25_place_cup_on_plate")
            ),
            {},
            sim_backend="isaacsim",
            task_id="t25_place_cup_on_plate",
            collection_run_seed=100,
            reset_attempt_index=0,
            seed_resets=True,
        )


def test_session_journal_fails_closed_on_legacy_double_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        recorder._FRESH_CERTIFIED_COLLECTION_WORKER_ENV,
        raising=False,
    )

    class Environment:
        def __init__(self) -> None:
            self.unwrapped = SimpleNamespace(task_name="t25_place_cup_on_plate")
            self.reset_calls = 0

        def reset(self, **_kwargs):
            self.reset_calls += 1
            if self.reset_calls == 1:
                return {}, {"manual_reset_applied": True}
            return {}, {}

    env = Environment()
    with pytest.raises(RuntimeError, match="more than one env.reset"):
        recorder._reset_recording_episode_with_stable_origin(
            env,
            {},
            sim_backend="isaacsim",
            task_id="t25_place_cup_on_plate",
            collection_run_seed=100,
            reset_attempt_index=0,
            seed_resets=True,
            reset_attempt_observer=lambda _event: None,
        )

    assert env.reset_calls == 2


def test_camera_viewer_does_not_lock_user_resized_window(monkeypatch) -> None:
    calls = {"named": [], "shown": [], "resized_window": []}

    monkeypatch.setattr(
        recorder.cv2,
        "namedWindow",
        lambda name, mode: calls["named"].append((name, mode)),
    )
    monkeypatch.setattr(
        recorder.cv2,
        "imshow",
        lambda name, frame: calls["shown"].append((name, frame.shape)),
    )
    monkeypatch.setattr(
        recorder.cv2,
        "resizeWindow",
        lambda *args: calls["resized_window"].append(args),
    )
    monkeypatch.setattr(recorder.cv2, "waitKey", lambda _delay: -1)

    viewer = recorder.WristCameraViewer(display_scale=1.0, include_main=True)
    observation = {
        "front": _solid_rgb(10),
        "wrist_left": _solid_rgb(20),
        "wrist_right": _solid_rgb(30),
    }
    viewer.update_and_show(observation)

    assert [name for name, _mode in calls["named"]] == [
        "Main",
        "Wrist Left",
        "Wrist Right",
    ]
    assert [name for name, _shape in calls["shown"]] == [
        "Main",
        "Wrist Left",
        "Wrist Right",
    ]
    assert calls["resized_window"] == []


def test_camera_viewer_draws_c01_overlay_only_on_display_copy(monkeypatch) -> None:
    calls = {"named": [], "shown": []}

    monkeypatch.setattr(
        recorder.cv2,
        "namedWindow",
        lambda name, mode: calls["named"].append((name, mode)),
    )
    monkeypatch.setattr(
        recorder.cv2,
        "imshow",
        lambda name, frame: calls["shown"].append((name, frame.copy())),
    )
    monkeypatch.setattr(recorder.cv2, "waitKey", lambda _delay: -1)

    class FakeEnv:
        def __init__(self):
            self.camera_keys = []

        def c01_hammer_guide_pixels(self, camera_key, image_shape):
            self.camera_keys.append((camera_key, image_shape[:2]))
            return {
                "line_start": (4, 4),
                "line_end": (56, 32),
                "impact": (56, 32),
                "nail_target": (48, 28),
                "aligned": False,
            }

    env = FakeEnv()
    ego = _solid_rgb(10)
    wrist_left = _solid_rgb(20)
    wrist_right = _solid_rgb(30)
    observation = {
        "images": {
            "ego": ego,
            "wrist_left": wrist_left,
            "wrist_right": wrist_right,
        }
    }

    assert recorder._has_displayable_wrist_images(observation)
    viewer = recorder.WristCameraViewer(display_scale=1.0, include_main=True)
    viewer.update_and_show(observation, env=env)

    assert [name for name, _mode in calls["named"]] == [
        "Main",
        "Wrist Left",
        "Wrist Right",
    ]
    assert [key for key, _shape in env.camera_keys] == [
        "ego",
        "wrist_left",
        "wrist_right",
    ]
    assert np.all(ego == 10)
    assert np.all(wrist_left == 20)
    assert np.all(wrist_right == 30)
    assert any(not np.all(frame == frame[0, 0]) for _name, frame in calls["shown"])


def test_live_preview_draws_c01_overlay_only_on_preview_copy(tmp_path: Path) -> None:
    class FakeEnv:
        def __init__(self):
            self.camera_keys = []

        def c01_hammer_guide_pixels(self, camera_key, image_shape):
            self.camera_keys.append((camera_key, image_shape[:2]))
            return {
                "line_start": (4, 4),
                "line_end": (56, 32),
                "impact": (56, 32),
                "nail_target": (48, 28),
                "aligned": False,
            }

    env = FakeEnv()
    ego = _solid_rgb(10)
    preview_path = tmp_path / "live_mujoco.png"

    recorder._write_live_preview(
        {"images": {"ego": ego}},
        preview_path,
        env=env,
    )

    assert [key for key, _shape in env.camera_keys] == ["ego"]
    assert np.all(ego == 10)
    written = recorder.cv2.imread(str(preview_path), recorder.cv2.IMREAD_COLOR)
    assert written is not None
    assert not np.all(written == written[0, 0])


def test_external_live_preview_uses_topmost_viewer_and_closes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview_path = tmp_path / "live_mujoco.png"
    preview_path.write_bytes(b"png")
    calls: dict[str, object] = {}

    class Process:
        pid = 4321

        def poll(self):
            return None

        def terminate(self):
            calls["terminated"] = True

        def wait(self, timeout):
            calls["wait_timeout"] = timeout

        def kill(self):
            calls["killed"] = True

    def popen(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return Process()

    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.setattr(recorder.os, "getpid", lambda: 1234)
    monkeypatch.setattr(recorder.subprocess, "Popen", popen)

    process = recorder._launch_external_live_preview(preview_path)
    recorder._close_external_live_preview(process)

    assert calls["command"] == [
        "/usr/bin/python3",
        str(Path(recorder.__file__).with_name("live_camera_viewer.py")),
        "--image",
        str(preview_path),
        "--parent-pid",
        "1234",
        "--title",
        "HandsOn 实时仿真相机",
    ]
    assert calls["kwargs"]["stdout"] is recorder.subprocess.DEVNULL
    assert calls["kwargs"]["stderr"] is recorder.subprocess.DEVNULL
    assert calls["terminated"] is True
    assert calls["wait_timeout"] == 2.0


def test_live_camera_r_queues_sigusr1_in_the_recorder() -> None:
    calls: list[tuple[int, object]] = []

    assert live_camera_viewer._request_parent_reset(
        1234,
        kill=lambda pid, signum: calls.append((pid, signum)),
    )
    assert calls == [(1234, live_camera_viewer.signal.SIGUSR1)]
    assert not live_camera_viewer._request_parent_reset(
        0,
        kill=lambda *_args: pytest.fail("invalid parent must not be signalled"),
    )


def test_manual_reset_signal_bridge_only_sets_and_consumes_an_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed: list[tuple[object, object]] = []
    bridge = recorder.ManualResetSignalBridge()
    monkeypatch.setattr(
        recorder.signal,
        "signal",
        lambda signum, handler: installed.append((signum, handler)),
    )

    bridge.install()
    assert bridge.consume() is False
    bridge._handle(bridge.signum, None)
    assert bridge.consume() is True
    assert bridge.consume() is False
    bridge.restore()

    assert installed[0] == (bridge.signum, bridge._handle)
    assert installed[-1] == (bridge.signum, bridge.previous_handler)


def test_external_live_preview_requires_display_and_existing_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview_path = tmp_path / "missing.png"
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(
        recorder.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("viewer must not launch"),
    )

    assert recorder._launch_external_live_preview(preview_path) is None
    preview_path.write_bytes(b"png")
    assert recorder._launch_external_live_preview(preview_path) is None


def test_b03_newton_live_preview_publishes_every_control_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HANDSON_LIVE_PREVIEW_STRIDE", raising=False)

    assert recorder._live_preview_stride("isaacsim_newton", "t09_insert_peg") == 1
    assert recorder._live_preview_stride("isaacsim", "t09_insert_peg") == 3
    assert recorder._live_preview_stride("isaacsim_newton", "t26_place_fruit_in_bucket") == 3

    monkeypatch.setenv("HANDSON_LIVE_PREVIEW_STRIDE", "2")
    assert recorder._live_preview_stride("isaacsim_newton", "t09_insert_peg") == 1
    assert recorder._live_preview_stride("isaacsim", "t09_insert_peg") == 2

    monkeypatch.setenv("HANDSON_LIVE_PREVIEW_STRIDE", "invalid")
    assert recorder._live_preview_stride("isaacsim_newton", "t09_insert_peg") == 1

    assert recorder._async_live_preview_enabled(
        "isaacsim_newton", "t09_insert_peg"
    )
    assert not recorder._async_live_preview_enabled(
        "isaacsim", "t09_insert_peg"
    )


def test_deferred_camera_mode_disables_file_backed_live_preview(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "HANDSON_LIVE_PREVIEW_PATH",
        str(tmp_path / "live.png"),
    )
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "1")

    assert recorder._live_preview_path() is None

    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "0")
    assert recorder._live_preview_path() == tmp_path / "live.png"


def test_latest_frame_preview_publisher_copies_and_drops_stale_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_started = recorder.threading.Event()
    release_worker = recorder.threading.Event()
    written_values: list[int] = []

    def fake_write(_camera_key, frame, _preview_path) -> None:
        worker_started.set()
        assert release_worker.wait(timeout=2.0)
        written_values.append(int(frame[0, 0, 0]))

    monkeypatch.setattr(recorder, "_write_live_preview_frame", fake_write)
    publisher = recorder.LatestFramePreviewPublisher(tmp_path / "live.png")

    first = _solid_rgb(1)
    assert publisher.submit({"images": {"front": first}})
    assert worker_started.wait(timeout=2.0)
    first.fill(99)
    assert publisher.submit({"images": {"front": _solid_rgb(2)}})
    assert publisher.submit({"images": {"front": _solid_rgb(3)}})

    release_worker.set()
    publisher.close(timeout=2.0)
    publisher.close(timeout=2.0)

    assert written_values == [1, 3]
    assert not publisher._thread.is_alive()
    assert not publisher.submit({"images": {"front": _solid_rgb(4)}})


def test_latest_frame_preview_publisher_survives_worker_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_attempt = recorder.threading.Event()
    second_attempt = recorder.threading.Event()
    attempts = 0

    def flaky_write(_camera_key, _frame, _preview_path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            first_attempt.set()
            raise RuntimeError("synthetic encoder failure")
        second_attempt.set()

    monkeypatch.setattr(recorder, "_write_live_preview_frame", flaky_write)
    publisher = recorder.LatestFramePreviewPublisher(tmp_path / "live.png")

    assert publisher.submit({"images": {"front": _solid_rgb(1)}})
    assert first_attempt.wait(timeout=2.0)
    assert publisher.submit({"images": {"front": _solid_rgb(2)}})
    assert second_attempt.wait(timeout=2.0)
    publisher.close(timeout=2.0)

    assert attempts == 2
    assert not publisher._thread.is_alive()


def test_b03_newton_formal_monitor_includes_main_camera() -> None:
    assert recorder._sim_camera_viewer_includes_main(
        sim_backend="isaacsim_newton",
        task_id="t09_insert_peg",
        preview_only=False,
    )
    assert recorder._sim_camera_viewer_includes_main(
        sim_backend="isaacsim_newton",
        task_id="t09_insert_peg",
        preview_only=True,
    )
    assert not recorder._sim_camera_viewer_includes_main(
        sim_backend="isaacsim_newton",
        task_id="t26_place_fruit_in_bucket",
        preview_only=False,
    )
    assert recorder._has_displayable_sim_images(
        {"images": {"front": _solid_rgb(10)}},
        include_main=True,
    )
    assert not recorder._has_displayable_sim_images(
        {"images": {"front": _solid_rgb(10)}},
        include_main=False,
    )


def test_discarded_spool_aborts_video_encoders_without_flushing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writers = []

    class AbortAwareWriter(_FakeWriter):
        codec = "h264_nvenc"
        fallback_codec = None

        def __init__(self):
            super().__init__()
            self.aborted = False

        def abort(self):
            self.aborted = True

        def stop(self):
            raise AssertionError("discarded episodes must not flush encoders")

    def create_h264(**_kwargs):
        writer = AbortAwareWriter()
        writers.append(writer)
        return writer

    monkeypatch.setattr(recorder.Mp4VideoWriter, "create_h264", create_h264)
    frame = _solid_rgb(42)
    spool = recorder.RecordingSpool(tmp_path, video_fps=30)
    spool.append_observation(_three_camera_observation(frame, frame, frame))

    spool.cleanup()

    assert len(writers) == 3
    assert all(writer.aborted for writer in writers)
    assert not spool.root.exists()


def _install_fake_video_writers(monkeypatch: pytest.MonkeyPatch):
    writers = []

    def create_h264(**_kwargs):
        writer = _FakeWriter()
        writers.append(writer)
        return writer

    monkeypatch.setattr(recorder.Mp4VideoWriter, "create_h264", create_h264)
    return writers


def _three_camera_observation(front, wrist_left, wrist_right, **extra):
    return {
        "images": {
            "front": front,
            "wrist_left": wrist_left,
            "wrist_right": wrist_right,
        },
        **extra,
    }


def test_observation_rgb_encodes_once_and_trajectory_stays_image_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writers = _install_fake_video_writers(monkeypatch)
    front = _solid_rgb(37)
    wrist_left = _solid_rgb(181)
    wrist_right = _solid_rgb(92)
    state = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    declared_tensor = np.zeros((40, 40, 3), dtype=np.float32)

    spool = recorder.RecordingSpool(tmp_path, video_fps=30)
    root = spool.root
    source_observation = _three_camera_observation(
        front,
        wrist_left,
        wrist_right,
        state=state,
        declared_tensor=declared_tensor,
    )
    spool.append_observation(source_observation)
    observation = spool.compact_observation(
        source_observation,
        protected_keys={"declared_tensor"},
    )

    assert set(observation) == {"state", "declared_tensor"}
    assert observation["state"] is state
    assert observation["declared_tensor"] is declared_tensor
    videos_dir = tmp_path / "demo" / "videos"
    camera_sources, first_frames = spool.publish_videos(videos_dir)
    assert len(writers) == 3
    np.testing.assert_array_equal(writers[0].frames[0], front)
    np.testing.assert_array_equal(writers[1].frames[0], wrist_left)
    np.testing.assert_array_equal(writers[2].frames[0], wrist_right)
    assert camera_sources == {
        "main": "front",
        "wrist_left": "wrist_left",
        "wrist_right": "wrist_right",
    }
    np.testing.assert_array_equal(first_frames["main"], front)
    assert {path.name for path in videos_dir.glob("*.mp4")} == {
        "main.mp4",
        "wrist_left.mp4",
        "wrist_right.mp4",
    }
    assert all(writer.stopped for writer in writers)

    spool.cleanup()
    assert not root.exists()


def test_postrender_spool_counts_frames_without_encoding_placeholder_rgb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recorder.Mp4VideoWriter,
        "create_h264",
        lambda **_kwargs: pytest.fail(
            "camera-postrender collection must not create an online encoder"
        ),
    )
    spool = recorder.RecordingSpool(
        tmp_path,
        video_fps=30,
        encode_videos=False,
    )
    root = spool.root
    spool.append_observation({})
    spool.append_observation({"state": np.asarray([1.0])})

    assert spool.frame_count == 2
    videos_dir = tmp_path / "postrender-demo" / "videos"
    assert spool.publish_videos(videos_dir) == ({}, {})
    assert not videos_dir.exists()

    spool.cleanup()
    assert not root.exists()


def test_b03_newton_postrender_real_save_keeps_trace_without_encoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from handson.data.b03_newton_validity import (
        b03_newton_source_validity_gate_contract,
    )
    from handson.sim.isaaclab.newton_replay import RecordedNewtonDrivePath

    flags = _episode_test_flags(sim_backend="isaacsim_newton")
    flags.save_camera_info = True
    monkeypatch.setattr(recorder, "FLAGS", flags)
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_ARM_RENDER", "0")
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "1")
    monkeypatch.setenv("HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY", "0")
    monkeypatch.setenv("HANDSON_ISAACLAB_DIRECT_HAND_CONTROL", "0")
    monkeypatch.setattr(
        recorder.Mp4VideoWriter,
        "create_h264",
        lambda **_kwargs: pytest.fail(
            "camera-postrender collection must not create an online encoder"
        ),
    )

    field_specs = {
        "newton_sim_time": {"shape": [1], "dtype": "float64"},
        "newton_substep_robot_joint_target": {
            "shape": [2, 2, 27],
            "dtype": "float64",
        },
        "newton_substep_robot_joint_target_valid": {
            "shape": [2, 2, 27],
            "dtype": "uint8",
        },
        "newton_substep_robot_joint_velocity": {
            "shape": [2, 2, 27],
            "dtype": "float64",
        },
        "newton_substep_robot_joint_valid": {
            "shape": [2, 2, 27],
            "dtype": "uint8",
        },
        "newton_substep_hand_peg_contact_valid": {
            "shape": [2, 1],
            "dtype": "uint8",
        },
        "newton_substep_hand_peg_max_solid_penetration": {
            "shape": [2, 1],
            "dtype": "float64",
        },
        "newton_substep_count": {"shape": [1], "dtype": "int64"},
        "newton_robot_joint_target": {
            "shape": [2, 27],
            "dtype": "float64",
        },
        "newton_robot_joint_target_valid": {
            "shape": [2, 27],
            "dtype": "uint8",
        },
    }
    robot_joint_names = {
        side: [
            *(f"joint{joint}_{side}_1" for joint in range(1, 8)),
            *(
                f"{side}_finger{finger}_joint{joint}"
                for finger in range(1, 6)
                for joint in range(1, 5)
            ),
        ]
        for side in ("right", "left")
    }
    environment_schema = {
        "name": "handson.b03_newton_ground_truth",
        "version": 3,
        "fields": field_specs,
        "newton": {
            "n_substeps": 2,
            "robot_sides": ["right", "left"],
            "robot_joint_names": robot_joint_names,
            "source_validity_gate": (
                b03_newton_source_validity_gate_contract()
            ),
        },
    }
    raw_env = SimpleNamespace(
        _physics_backend="newton",
        task_name="t09_insert_peg",
        _newton_ground_truth_enabled=True,
        recorded_drive_target_replay_supported=True,
        _newton_replay_profile=recorder.NEWTON_CROSS_GPU_PROFILE,
        _direct_hand_control=False,
        arm_type="rokae",
        hand_type="wuji1",
        image_obs=True,
        recording_camera_profile={},
        get_physical_replay_reset_contract=lambda: {
            "strategy": "fresh_test_reset"
        },
    )
    env = SimpleNamespace(unwrapped=raw_env)
    required_fields = set(field_specs)

    samples = []
    for index in range(3):
        targets = (
            np.arange(108, dtype=np.float64).reshape(2, 2, 27)
            + index * 200
        )
        samples.append(
            {
                "state": np.asarray(
                    [index, index + 0.25, index + 0.5],
                    dtype=np.float64,
                ),
                "newton_sim_time": np.asarray(
                    [index / 30.0],
                    dtype=np.float64,
                ),
                "newton_substep_robot_joint_target": targets,
                "newton_substep_robot_joint_target_valid": np.ones(
                    (2, 2, 27),
                    dtype=np.uint8,
                ),
                "newton_substep_robot_joint_velocity": np.zeros(
                    (2, 2, 27),
                    dtype=np.float64,
                ),
                "newton_substep_robot_joint_valid": np.ones(
                    (2, 2, 27),
                    dtype=np.uint8,
                ),
                "newton_substep_hand_peg_contact_valid": np.ones(
                    (2, 1),
                    dtype=np.uint8,
                ),
                "newton_substep_hand_peg_max_solid_penetration": np.zeros(
                    (2, 1),
                    dtype=np.float64,
                ),
                "newton_substep_count": np.asarray([2], dtype=np.int64),
                "newton_robot_joint_target": targets[-1].copy(),
                "newton_robot_joint_target_valid": np.ones(
                    (2, 27),
                    dtype=np.uint8,
                ),
                "images": {
                    "front": _solid_rgb(1 + index),
                    "wrist_left": _solid_rgb(4 + index),
                    "wrist_right": _solid_rgb(7 + index),
                },
            }
        )

    spool = recorder.RecordingSpool(
        tmp_path,
        video_fps=30,
        encode_videos=False,
    )
    trajectory = []
    for index in range(2):
        spool.append_observation(samples[index], env=env)
        current = recorder.select_current_observation_for_recording(
            samples[index],
            required_fields=required_fields,
            include_required_fields=index == 0,
        )
        current = spool.compact_observation(
            current,
            protected_keys=required_fields,
        )
        next_observation = recorder.select_next_observation_for_recording(
            samples[index + 1],
            required_fields=required_fields,
        )
        trajectory.append(
            {
                "observations": current,
                "next_observations": next_observation,
                "actions": np.zeros(3, dtype=np.float64),
                "dones": index == 1,
                "infos": {"succeed": index == 1},
            }
        )

    assert all("images" not in step["observations"] for step in trajectory)
    assert (
        "newton_substep_robot_joint_target"
        not in trajectory[1]["observations"]
    )

    generation = recorder._dataset_generation_for_env(env)
    invalid_trajectory = copy.deepcopy(trajectory)
    invalid_trajectory[0]["next_observations"][
        "newton_substep_robot_joint_velocity"
    ][0, 0, 7] = 4.01
    invalid_output = tmp_path / "invalid-b03-output"
    invalid_spool = recorder.RecordingSpool(
        tmp_path,
        video_fps=30,
        encode_videos=False,
    )
    for sample in samples[:2]:
        invalid_spool.append_observation(sample, env=env)
    with pytest.raises(
        RuntimeError,
        match="Refusing to save physically invalid B03 Newton episode",
    ):
        recorder._write_demo_zarr_and_videos(
            invalid_trajectory,
            "t09_insert_peg",
            99,
            invalid_output,
            30,
            env=env,
            environment_schema=environment_schema,
            required_recording_fields=required_fields,
            video_spool=invalid_spool,
            collection_run_seed=7,
            episode_reset_seed=8,
            collection_seed_source=recorder.RANDOM_COLLECTION_SEED_SOURCE,
            episode_reset_ordinal=0,
            online_camera_capture_scope=(
                recorder.ONLINE_CAMERA_SCOPE_ALL_POSTRENDER
            ),
        )
    invalid_spool.cleanup()
    assert not list(invalid_output.glob("t09_insert_peg_demo_*"))
    assert not (invalid_output / ".recording_staging").exists()

    demo_dir = tmp_path / (
        "t09_insert_peg_demo_1_test"
        f"{generation.directory_suffix}"
    )
    frozen_camera_scope = recorder._runtime_online_camera_capture_scope(
        "t09_insert_peg",
        "isaacsim_newton",
    )
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "0")
    recorder._write_demo_contents(
        trajectory,
        "t09_insert_peg",
        1,
        demo_dir,
        30,
        env=env,
        environment_schema=environment_schema,
        required_recording_fields=required_fields,
        video_spool=spool,
        collection_run_seed=7,
        episode_reset_seed=8,
        collection_seed_source=recorder.RANDOM_COLLECTION_SEED_SOURCE,
        episode_reset_ordinal=0,
        dataset_generation=generation,
        published_demo_name=demo_dir.name,
        online_camera_capture_scope=frozen_camera_scope,
    )

    root = zarr.open_group(str(demo_dir / "replay.zarr"), mode="r")
    data = root["data"]
    np.testing.assert_array_equal(
        data["state"][:],
        np.stack([sample["state"] for sample in samples[:2]]),
    )
    np.testing.assert_array_equal(
        data["next_state"][:],
        np.stack([sample["state"] for sample in samples[1:]]),
    )
    for field_name in required_fields:
        np.testing.assert_array_equal(
            data[field_name][:],
            np.stack([sample[field_name] for sample in samples[:2]]),
        )
        np.testing.assert_array_equal(
            data[f"next_{field_name}"][:],
            np.stack([sample[field_name] for sample in samples[1:]]),
        )
    assert not (demo_dir / "videos").exists()
    assert not any(
        key in data for key in ("images", "front", "wrist_left", "wrist_right")
    )

    manifest = json.loads(
        (demo_dir / "replay_manifest.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (demo_dir / "collection_meta.json").read_text(encoding="utf-8")
    )
    for contract in (manifest["recording"], metadata):
        assert contract["online_camera_capture_scope"] == (
            recorder.ONLINE_CAMERA_SCOPE_ALL_POSTRENDER
        )
        assert contract["canonical_camera_postrender_required"] is True
        assert contract["online_video_encoding_enabled"] is False
        assert contract["source_physical_validity"]["valid"] is True
        assert contract["source_physical_validity"]["status"] == "valid"
        assert contract["source_physical_validity"][
            "evaluated_physics_substeps"
        ] == 4

    recorded_schema = json.loads(
        (demo_dir / "recording_schema.json").read_text(encoding="utf-8")
    )
    replay_path = RecordedNewtonDrivePath(recorded_schema)
    next_snapshot = {
        field_name: data[f"next_{field_name}"][0]
        for field_name in (
            "newton_substep_robot_joint_target",
            "newton_substep_robot_joint_target_valid",
            "newton_substep_count",
            "newton_robot_joint_target",
            "newton_robot_joint_target_valid",
        )
    }
    replay_step = replay_path.control_step(next_snapshot)
    assert replay_step.substep_count == 2
    assert replay_step.expected_joint_count == 54
    np.testing.assert_array_equal(
        replay_step.targets,
        samples[1]["newton_substep_robot_joint_target"],
    )

    spool.cleanup()


def test_recording_spool_applies_environment_frame_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writers = _install_fake_video_writers(monkeypatch)
    frame = _solid_rgb(37)
    calls = []

    class FilteredEnv:
        def postprocess_recording_frame(self, image, *, canonical_key, source_key):
            calls.append((canonical_key, source_key))
            output = np.asarray(image).copy()
            output[0, 0] = np.asarray([9, 8, 7], dtype=np.uint8)
            return output

    spool = recorder.RecordingSpool(tmp_path, video_fps=30)
    spool.append_observation(
        _three_camera_observation(frame, frame, frame),
        env=FilteredEnv(),
    )

    assert calls == [
        ("main", "front"),
        ("wrist_left", "wrist_left"),
        ("wrist_right", "wrist_right"),
    ]
    for writer in writers:
        np.testing.assert_array_equal(writer.frames[0][0, 0], [9, 8, 7])

    spool.cleanup()


def test_published_h264_spool_is_decodable_at_original_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import av

    monkeypatch.setenv("HANDSON_VIDEO_ENCODER", "cpu")
    first = _solid_rgb(24)
    second = _solid_rgb(196)
    with recorder.RecordingSpool(tmp_path, video_fps=30) as spool:
        spool.append_observation(
            _three_camera_observation(first, first, first)
        )
        spool.append_observation(
            _three_camera_observation(second, second, second)
        )
        videos_dir = tmp_path / "decoded-demo" / "videos"
        spool.publish_videos(videos_dir)

    for canonical_key in recorder.CANONICAL_CAMERA_KEYS:
        video_path = videos_dir / f"{canonical_key}.mp4"
        with av.open(str(video_path)) as container:
            decoded = list(container.decode(video=0))
        assert len(decoded) == 2
        assert (decoded[0].height, decoded[0].width) == first.shape[:2]


def test_nvenc_first_frame_failure_retries_with_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu_writers = []
    cpu_writers = []

    class RuntimeFailNvencWriter(_FakeWriter):
        codec = "h264_nvenc"
        fallback_codec = "h264"

        def __init__(self):
            super().__init__()
            self.codec_options = {}
            self.fallback_codec_options = {}

        def write_frame(self, frame):
            raise RuntimeError("NVENC session unavailable")

    class CpuWriter(_FakeWriter):
        codec = "h264"
        fallback_codec = None

        def __init__(self):
            super().__init__()
            self.codec_options = {}
            self.fallback_codec_options = {}

    def create_h264(**kwargs):
        if kwargs.get("encoder") == "cpu":
            writer = CpuWriter()
            cpu_writers.append(writer)
            return writer
        writer = RuntimeFailNvencWriter()
        gpu_writers.append(writer)
        return writer

    monkeypatch.setattr(recorder.Mp4VideoWriter, "create_h264", create_h264)
    frame = _solid_rgb(71)
    with recorder.RecordingSpool(tmp_path, video_fps=30) as spool:
        spool.append_observation(_three_camera_observation(frame, frame, frame))
        spool.publish_videos(tmp_path / "nvenc-fallback-demo" / "videos")

    assert len(gpu_writers) == 3
    assert len(cpu_writers) == 3
    assert all(writer.stopped for writer in gpu_writers + cpu_writers)
    assert [len(writer.frames) for writer in cpu_writers] == [1, 1, 1]
    assert all(
        writer.codec_options["preset"] == "ultrafast"
        for writer in cpu_writers
    )


def test_spool_uses_configured_output_disk_and_cleans_on_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool_parent = tmp_path / "camera-spool-disk"
    monkeypatch.setenv("HANDSON_RECORDING_SPOOL_DIR", str(spool_parent))
    monkeypatch.setenv("HANDSON_RECORDING_SPOOL_VIDEO_CRF", "17")
    writers = _install_fake_video_writers(monkeypatch)

    root = None
    with pytest.raises(RuntimeError, match="collection failed"):
        with recorder.RecordingSpool(
            tmp_path / "output",
            video_fps=30,
        ) as spool:
            root = spool.root
            assert spool.video_crf == 17
            assert root.parent == spool_parent
            frame = _solid_rgb(90)
            spool.append_observation(
                _three_camera_observation(frame, frame, frame)
            )
            raise RuntimeError("collection failed")

    assert root is not None
    assert not root.exists()
    # A configured directory can be shared across collectors and is retained.
    assert spool_parent.is_dir()
    assert all(writer.stopped for writer in writers)


def test_incomplete_rgb_bundle_fails_without_shifting_other_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writers = _install_fake_video_writers(monkeypatch)
    frame = _solid_rgb(80)
    spool = recorder.RecordingSpool(tmp_path, video_fps=30)
    spool.append_observation(_three_camera_observation(frame, frame, frame))

    with pytest.raises(RuntimeError, match="Incomplete canonical camera bundle"):
        spool.append_observation(
            {"images": {"front": frame, "wrist_right": frame}}
        )

    assert spool.frame_count == 1
    spool.publish_videos(tmp_path / "aligned-demo" / "videos")
    assert [len(writer.frames) for writer in writers] == [1, 1, 1]
    spool.cleanup()


def test_depth_bundle_requires_all_three_canonical_frames(tmp_path: Path) -> None:
    frame = np.ones((3, 4), dtype=np.float32)
    with recorder.RecordingSpool(tmp_path, video_fps=30) as spool:
        refs = spool.spool_depth_mapping(
            {key: frame for key in recorder.CANONICAL_CAMERA_KEYS}
        )
        assert set(refs) == set(recorder.CANONICAL_CAMERA_KEYS)
        assert spool.depth_frame_count == 1
        with pytest.raises(RuntimeError, match="Incomplete canonical depth bundle"):
            spool.spool_depth_mapping({"main": frame})
        assert spool.depth_frame_count == 1


def test_camera_metadata_uses_actual_observation_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []

    def describe(_env, source_keys):
        seen.append(source_keys)
        return {key: {"source": key} for key in source_keys}

    monkeypatch.setattr(recorder, "describe_camera_views", describe)
    views = recorder._canonical_camera_views(
        object(),
        {
            "main": "ego",
            "wrist_left": "ego_left",
            "wrist_right": "ego_right",
        },
    )
    assert seen == [["ego", "ego_left", "ego_right"]]
    assert views["main"]["source"] == "ego"
    assert views["wrist_left"]["source"] == "ego_left"
    assert views["wrist_right"]["source"] == "ego_right"
    assert views["main"]["observation_source_key"] == "ego"
    assert views["wrist_left"]["observation_source_key"] == "ego_left"
    assert views["wrist_right"]["observation_source_key"] == "ego_right"


def test_demo_staging_is_removed_when_save_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_after_replay(*_args, **kwargs):
        demo_dir = Path(_args[3])
        (demo_dir / "replay.zarr").mkdir(parents=True)
        raise RuntimeError("video finalize failed")

    monkeypatch.setattr(recorder, "_write_demo_contents", fail_after_replay)
    with pytest.raises(RuntimeError, match="video finalize failed"):
        recorder._write_demo_zarr_and_videos(
            [],
            "d02_test",
            1,
            tmp_path,
            30,
        )

    assert not list(tmp_path.glob("d02_test_demo_*"))
    assert not recorder._recording_staging_parent(tmp_path).exists()


def test_complete_demo_is_published_only_after_staging_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_staging_dirs = []

    def write_complete_demo(*args, **_kwargs):
        demo_dir = Path(args[3])
        observed_staging_dirs.append(demo_dir)
        (demo_dir / "replay.zarr").mkdir(parents=True)
        videos_dir = demo_dir / "videos"
        videos_dir.mkdir()
        for key in recorder.CANONICAL_CAMERA_KEYS:
            (videos_dir / f"{key}.mp4").write_bytes(b"complete video")
            (videos_dir / f"{key}_camera_info.json").write_text("{}")
        (videos_dir / "camera_info.json").write_text("{}")

    monkeypatch.setattr(recorder, "_write_demo_contents", write_complete_demo)
    monkeypatch.setattr(
        recorder,
        "FLAGS",
        type("Flags", (), {"save_camera_info": True, "save_depth": False})(),
    )
    published = Path(
        recorder._write_demo_zarr_and_videos(
            [],
            "d02_test",
            1,
            tmp_path,
            30,
        )
    )

    assert published.parent == tmp_path.resolve()
    assert (published / "replay.zarr").is_dir()
    assert all(
        (published / "videos" / f"{key}.mp4").is_file()
        for key in recorder.CANONICAL_CAMERA_KEYS
    )
    assert len(observed_staging_dirs) == 1
    assert observed_staging_dirs[0] != published
    assert not observed_staging_dirs[0].exists()
    assert not recorder._recording_staging_parent(tmp_path).exists()


def test_postrender_demo_publishes_without_placeholder_video_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def write_state_only_demo(*args, **kwargs):
        demo_dir = Path(args[3])
        observed["camera_scope"] = kwargs["online_camera_capture_scope"]
        (demo_dir / "replay.zarr").mkdir(parents=True)
        recorder._write_dataset_generation_marker(
            demo_dir,
            kwargs["dataset_generation"],
            episode_id=kwargs["published_demo_name"],
        )
        # Publishing must keep using the session's resolved contract even if
        # an environment variable changes while a long episode is being saved.
        monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "0")

    monkeypatch.setattr(
        recorder,
        "_write_demo_contents",
        write_state_only_demo,
    )
    monkeypatch.setattr(
        recorder,
        "FLAGS",
        type(
            "Flags",
            (),
            {
                "save_camera_info": True,
                "save_depth": False,
                "sim_backend": "isaacsim_newton",
                "record_physx_ground_truth": False,
            },
        )(),
    )
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_ARM_RENDER", "0")
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER", "1")
    monkeypatch.delenv(
        "HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY",
        raising=False,
    )
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            _physics_backend="newton",
            task_name="t09_insert_peg",
        )
    )

    published = Path(
        recorder._write_demo_zarr_and_videos(
            [],
            "t09_insert_peg",
            1,
            tmp_path,
            30,
            env=env,
        )
    )

    assert published.name.endswith("__newton_v2")
    assert observed["camera_scope"] == recorder.ONLINE_CAMERA_SCOPE_ALL_POSTRENDER
    assert (published / "replay.zarr").is_dir()
    assert (published / "DATASET_NEWTON_CROSS_GPU_V2.json").is_file()
    assert not (published / "videos").exists()
    assert not recorder._recording_staging_parent(tmp_path).exists()


def test_deferred_physx_candidate_stays_hidden_until_fresh_certification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def write_complete_demo(*args, **_kwargs):
        demo_dir = Path(args[3])
        (demo_dir / "replay.zarr").mkdir(parents=True)
        videos_dir = demo_dir / "videos"
        videos_dir.mkdir()
        for key in recorder.CANONICAL_CAMERA_KEYS:
            (videos_dir / f"{key}.mp4").write_bytes(b"complete video")

    monkeypatch.setattr(recorder, "_write_demo_contents", write_complete_demo)
    monkeypatch.setattr(
        recorder,
        "FLAGS",
        type(
            "Flags",
            (),
            {
                "save_camera_info": False,
                "save_depth": False,
                "sim_backend": "isaacsim",
                "record_physx_ground_truth": True,
            },
        )(),
    )
    monkeypatch.setenv(
        recorder.DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_ENV,
        "1",
    )
    candidate = Path(
        recorder._write_demo_zarr_and_videos(
            [],
            "d02_test",
            1,
            tmp_path,
            30,
        )
    )

    assert candidate.parent == recorder._physical_replay_pending_parent(tmp_path)
    assert (candidate / "replay.zarr").is_dir()
    assert not (tmp_path / candidate.name).exists()
    assert not list(tmp_path.glob("d02_test_demo_*"))


def test_deferred_certifier_launch_waits_on_recorded_source_pid_and_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = (
        recorder._physical_replay_pending_parent(tmp_path)
        / "b03_demo__stable_v2"
    )
    (demo / "replay.zarr").mkdir(parents=True)
    observed: dict[str, object] = {}

    def launch(command, **kwargs):
        observed["command"] = list(command)
        observed["env"] = dict(kwargs["env"])
        return SimpleNamespace(pid=9876)

    monkeypatch.setattr(recorder.subprocess, "Popen", launch)
    monkeypatch.setattr(recorder.os, "getpid", lambda: 4321)
    monkeypatch.setenv(
        recorder.DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_ENV,
        "1",
    )

    queued = recorder._launch_deferred_physical_replay_certification_batch(
        [demo],
        exp_name="t09_insert_peg",
        certification_runs=3,
        certification_timeout_seconds=123,
    )

    assert queued == 1
    command = observed["command"]
    assert "--certification-runs=3" in command
    assert "--certification-timeout-seconds=123" in command
    child_env = observed["env"]
    assert recorder.DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_ENV not in child_env
    assert child_env[recorder.DEFERRED_SOURCE_RECORDER_PID_ENV] == "4321"


def _pending_replay_candidate(
    tmp_path: Path,
    *,
    name: str = "t09_insert_peg_demo_1__stable_v2",
    session_id: str = "session-1",
) -> Path:
    demo = recorder._physical_replay_pending_parent(tmp_path) / name
    (demo / "replay.zarr").mkdir(parents=True)
    (demo / "collection_meta.json").write_text(
        json.dumps({"collection_session_id": session_id}),
        encoding="utf-8",
    )
    return demo.resolve()


def _write_ordered_session_binding(
    demo: Path,
    *,
    session_id: str,
    pending: bool,
) -> None:
    binding = {
        "physical_replay_certification_scope": "ordered_collection_session",
        "physical_replay_session_history_pending": pending,
        "physical_replay_session_history_id": session_id,
        "physical_replay_session_history_root_relative_path": (
            ".physical_replay_pending/.physx_session_history/session-journal"
        ),
        "physical_replay_session_history_sha256": "a" * 64,
        "physical_replay_session_history_event_count": 3,
        "physical_replay_session_history_accepted_event_ordinal": 1,
        "physical_replay_session_history_certified_through_event_ordinal": 2,
    }
    (demo / "replay_manifest.json").write_text(
        json.dumps({"recording": binding}),
        encoding="utf-8",
    )
    (demo / "collection_meta.json").write_text(
        json.dumps({"collection_session_id": session_id, **binding}),
        encoding="utf-8",
    )


def test_deferred_recovery_hides_demo_until_session_history_is_sealed(
    tmp_path: Path,
) -> None:
    demo = _pending_replay_candidate(tmp_path)
    _write_ordered_session_binding(
        demo,
        session_id="session-1",
        pending=True,
    )

    candidates = recorder._pending_deferred_physical_replay_candidates(
        tmp_path,
        exp_name="t09_insert_peg",
    )

    assert candidates == []


def test_deferred_recovery_discovers_only_complete_session_history_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _pending_replay_candidate(tmp_path)
    _write_ordered_session_binding(
        demo,
        session_id="session-1",
        pending=False,
    )
    history_path = (
        tmp_path
        / recorder.PHYSICAL_REPLAY_PENDING_DIRECTORY
        / recorder.PHYSICAL_REPLAY_SESSION_HISTORY_DIRECTORY
        / "session-journal"
    )
    history_path.mkdir(parents=True)
    monkeypatch.setattr(
        recorder,
        "load_session_history",
        lambda *_args, **_kwargs: {
            "finalization": {
                "history_identity": {"sha256": "a" * 64},
                "accepted_demos": [{"demo_id": demo.name}],
            }
        },
    )

    candidates = recorder._pending_deferred_physical_replay_candidates(
        tmp_path,
        exp_name="t09_insert_peg",
        session_id="session-1",
    )

    assert candidates == [demo]


def test_deferred_recovery_never_discovers_a_partially_bound_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _pending_replay_candidate(tmp_path)
    second = _pending_replay_candidate(
        tmp_path,
        name="t09_insert_peg_demo_2__stable_v2",
    )
    _write_ordered_session_binding(
        first,
        session_id="session-1",
        pending=False,
    )
    _write_ordered_session_binding(
        second,
        session_id="session-1",
        pending=True,
    )
    history_path = (
        tmp_path
        / recorder.PHYSICAL_REPLAY_PENDING_DIRECTORY
        / recorder.PHYSICAL_REPLAY_SESSION_HISTORY_DIRECTORY
        / "session-journal"
    )
    history_path.mkdir(parents=True)
    monkeypatch.setattr(
        recorder,
        "load_session_history",
        lambda *_args, **_kwargs: {
            "finalization": {
                "history_identity": {"sha256": "a" * 64},
                "accepted_demos": [
                    {"demo_id": first.name},
                    {"demo_id": second.name},
                ],
            }
        },
    )

    assert recorder._pending_deferred_physical_replay_candidates(
        tmp_path,
        exp_name="t09_insert_peg",
    ) == []

    _write_ordered_session_binding(
        second,
        session_id="session-1",
        pending=False,
    )
    assert recorder._pending_deferred_physical_replay_candidates(
        tmp_path,
        exp_name="t09_insert_peg",
    ) == [first, second]


@pytest.mark.parametrize("recorder_live", [False, True])
def test_deferred_recovery_only_finalizes_a_dead_recorder_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorder_live: bool,
) -> None:
    session_id = "orphan-session"
    demo = _pending_replay_candidate(
        tmp_path,
        session_id=session_id,
    )
    (demo / "replay_manifest.json").write_text(
        json.dumps(
            {
                "recording": {
                    "physical_replay_certification_scope": (
                        "ordered_collection_session"
                    ),
                    "physical_replay_session_history_pending": True,
                }
            }
        ),
        encoding="utf-8",
    )
    history_path = recorder._physical_replay_session_history_path(
        tmp_path,
        session_id,
    )
    journal = recorder.PhysxSessionHistoryJournal.create(
        history_path,
        session_id=session_id,
        task_name="t09_insert_peg",
        collection_run_seed=100,
        metadata={"recorder_pid": 4321},
    )
    journal.record_reset_attempt(
        seed=100,
        ordinal=0,
        outcome="accepted",
        stable_kwargs={},
        stable_origin={"physics_steps": 2},
        start_sim_time=0.0,
        end_sim_time=0.02,
    )
    journal.record_accepted_demo(
        {
            "action": np.zeros((1, 2), dtype=np.float64),
            "physx_sim_time": np.asarray([[0.02]], dtype=np.float64),
            "next_physx_sim_time": np.asarray([[0.04]], dtype=np.float64),
        },
        demo_dir=demo,
        reset_ordinal=0,
    )
    monkeypatch.setattr(
        recorder,
        "_process_command",
        lambda pid: (
            ("python", "/repo/scripts/record_demos_zarr.py")
            if recorder_live and pid == 4321
            else ()
        ),
    )

    repaired = recorder._recover_physx_session_history_bindings(
        tmp_path,
        exp_name="t09_insert_peg",
        session_id=session_id,
    )

    if recorder_live:
        assert repaired == 0
        assert not journal.finalization_path.exists()
        manifest = json.loads(
            (demo / "replay_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["recording"][
            "physical_replay_session_history_pending"
        ] is True
        return

    assert repaired == 1
    manifest = json.loads(
        (demo / "replay_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["recording"][
        "physical_replay_session_history_pending"
    ] is False
    assert journal.finalization_path.is_file()
    assert recorder._pending_deferred_physical_replay_candidates(
        tmp_path,
        exp_name="t09_insert_peg",
        session_id=session_id,
    ) == [demo]


def test_deferred_recovery_does_not_duplicate_a_live_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _pending_replay_candidate(tmp_path)
    recorder._write_json_atomic(
        recorder._deferred_physical_worker_state_path(demo),
        {"status": "running", "pid": 4321},
    )
    monkeypatch.setattr(
        recorder,
        "_deferred_physical_worker_command",
        lambda pid: (
            (
                "python",
                "deferred_physical_replay_certification_worker.py",
                str(demo),
            )
            if pid == 4321
            else ()
        ),
    )
    launch = monkeypatch.setattr(
        recorder,
        "_launch_deferred_physical_replay_certification_batch",
        lambda *_args, **_kwargs: pytest.fail("duplicate worker launch"),
    )

    recovered = recorder._ensure_deferred_physical_replay_certification(
        tmp_path,
        exp_name="t09_insert_peg",
        certification_runs=3,
        certification_timeout_seconds=123,
    )

    assert recovered == 0
    assert launch is None


def test_deferred_recovery_does_not_retry_a_versioned_invalid_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _pending_replay_candidate(tmp_path)
    proof_dir = demo / recorder.PHYSICAL_REPLAY_RUN_DIRECTORY
    proof_dir.mkdir()
    proof_path = proof_dir / "run_001.json"
    proof_path.write_text(
        json.dumps(
            {"status": "failed", "reason": "physical_state_diverged"}
        ),
        encoding="utf-8",
    )
    proof_sha256 = recorder._sha256_regular_file(proof_path)
    (demo / recorder.FRESH_PROCESS_ACCEPTANCE_NAME).write_text(
        json.dumps(
            {
                "format": recorder.DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_FORMAT,
                "version": (
                    recorder.DEFERRED_PHYSICAL_REPLAY_ACCEPTANCE_VERSION
                ),
                "mode": recorder.DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE,
                "status": "failed",
                "reason": "physical divergence",
                "physical_replay_outcome_version": (
                    recorder.PHYSICAL_REPLAY_OUTCOME_VERSION
                ),
                "physical_replay_outcome_code": (
                    recorder.PHYSICAL_REPLAY_INVALID_OUTCOME_CODE
                ),
                "review_status": recorder.PHYSICAL_REPLAY_REVIEW_INVALID,
                "certification_runs": [
                    {
                        "run": 1,
                        "report": (
                            "physical_replay_certification_runs/run_001.json"
                        ),
                        "report_sha256": proof_sha256,
                    }
                ],
                "outcome_evidence": {
                    "format": (
                        "handson.physical_replay_invalid_outcome_evidence"
                    ),
                    "version": recorder.PHYSICAL_REPLAY_OUTCOME_VERSION,
                    "run": 1,
                    "report": (
                        "physical_replay_certification_runs/run_001.json"
                    ),
                    "report_sha256": proof_sha256,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        recorder,
        "_launch_deferred_physical_replay_certification_batch",
        lambda *_args, **_kwargs: pytest.fail("final failure was retried"),
    )

    recovered = recorder._ensure_deferred_physical_replay_certification(
        tmp_path,
        exp_name="t09_insert_peg",
        certification_runs=3,
        certification_timeout_seconds=123,
    )

    assert recovered == 0


def test_deferred_recovery_launches_only_one_physx_session_per_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _pending_replay_candidate(
        tmp_path,
        name="t09_insert_peg_demo_1__stable_v2",
        session_id="session-1",
    )
    second = _pending_replay_candidate(
        tmp_path,
        name="t09_insert_peg_demo_2__stable_v2",
        session_id="session-2",
    )
    observed: list[list[Path]] = []

    monkeypatch.setattr(
        recorder,
        "_launch_deferred_physical_replay_certification_batch",
        lambda demos, **_kwargs: (observed.append(list(demos)) or len(demos)),
    )

    recovered = recorder._ensure_deferred_physical_replay_certification(
        tmp_path,
        exp_name="t09_insert_peg",
        certification_runs=3,
        certification_timeout_seconds=123,
        source_recorder_pid=0,
        allow_review_required_rerun=True,
        manual_review_run_id="review-1",
    )

    assert recovered == 1
    assert observed == [[first]]
    assert second not in observed[0]


def test_manual_review_run_skips_a_session_it_already_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _pending_replay_candidate(
        tmp_path,
        name="t09_insert_peg_demo_1__stable_v2",
        session_id="session-1",
    )
    second = _pending_replay_candidate(
        tmp_path,
        name="t09_insert_peg_demo_2__stable_v2",
        session_id="session-2",
    )
    state = {
        "status": "review_required",
        "terminal": False,
        "physical_replay_outcome_version": recorder.PHYSICAL_REPLAY_OUTCOME_VERSION,
        "physical_replay_outcome_code": recorder.PHYSICAL_REPLAY_REVIEW_REQUIRED_OUTCOME_CODE,
        "review_status": recorder.PHYSICAL_REPLAY_REVIEW_REQUIRED,
        "manual_review_run_id": "review-1",
        "launch_attempt_count": 1,
    }
    recorder._write_json_atomic(
        recorder._deferred_physical_worker_state_path(first), state
    )
    observed: list[list[Path]] = []
    monkeypatch.setattr(
        recorder,
        "_launch_deferred_physical_replay_certification_batch",
        lambda demos, **_kwargs: (observed.append(list(demos)) or len(demos)),
    )

    recovered = recorder._ensure_deferred_physical_replay_certification(
        tmp_path,
        exp_name="t09_insert_peg",
        certification_runs=3,
        certification_timeout_seconds=123,
        source_recorder_pid=0,
        allow_review_required_rerun=True,
        manual_review_run_id="review-1",
    )

    assert recovered == 1
    assert observed == [[second]]
    exhausted = json.loads(
        recorder._deferred_physical_worker_state_path(first).read_text(
            encoding="utf-8"
        )
    )
    assert exhausted["status"] == "recovery_exhausted"


def test_manual_review_run_does_not_repeat_invalid_without_physical_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _pending_replay_candidate(
        tmp_path,
        name="t09_insert_peg_demo_1__stable_v2",
        session_id="session-1",
    )
    second = _pending_replay_candidate(
        tmp_path,
        name="t09_insert_peg_demo_2__stable_v2",
        session_id="session-2",
    )
    recorder._write_json_atomic(
        recorder._deferred_physical_worker_state_path(first),
        {
            "status": "invalid",
            "terminal": True,
            "physical_replay_outcome_version": (
                recorder.PHYSICAL_REPLAY_OUTCOME_VERSION
            ),
            "physical_replay_outcome_code": (
                recorder.PHYSICAL_REPLAY_INVALID_OUTCOME_CODE
            ),
            "review_status": recorder.PHYSICAL_REPLAY_REVIEW_INVALID,
            "manual_review_run_id": "review-1",
            "launch_attempt_count": 1,
        },
    )
    observed: list[list[Path]] = []
    monkeypatch.setattr(
        recorder,
        "_launch_deferred_physical_replay_certification_batch",
        lambda demos, **_kwargs: (observed.append(list(demos)) or len(demos)),
    )

    recovered = recorder._ensure_deferred_physical_replay_certification(
        tmp_path,
        exp_name="t09_insert_peg",
        certification_runs=3,
        certification_timeout_seconds=123,
        source_recorder_pid=0,
        allow_review_required_rerun=True,
        manual_review_run_id="review-1",
    )

    assert recovered == 1
    assert observed == [[second]]
    exhausted = json.loads(
        recorder._deferred_physical_worker_state_path(first).read_text(
            encoding="utf-8"
        )
    )
    assert exhausted["status"] == "recovery_exhausted"
    assert exhausted["review_status"] == recorder.PHYSICAL_REPLAY_REVIEW_REQUIRED


def test_deferred_recovery_restarts_an_orphan_incomplete_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _pending_replay_candidate(tmp_path)
    recorder._write_json_atomic(
        recorder._deferred_physical_worker_state_path(demo),
        {"status": "running", "pid": 4321},
    )
    monkeypatch.setattr(
        recorder,
        "_deferred_physical_worker_command",
        lambda _pid: (),
    )
    observed: dict[str, object] = {}

    def launch(demos, **kwargs):
        observed["demos"] = list(demos)
        observed["kwargs"] = dict(kwargs)
        return len(demos)

    monkeypatch.setattr(
        recorder,
        "_launch_deferred_physical_replay_certification_batch",
        launch,
    )

    recovered = recorder._ensure_deferred_physical_replay_certification(
        tmp_path,
        exp_name="t09_insert_peg",
        certification_runs=3,
        certification_timeout_seconds=123,
        session_id="session-1",
        source_recorder_pid=9876,
    )

    assert recovered == 1
    assert observed["demos"] == [demo]
    assert observed["kwargs"] == {
        "exp_name": "t09_insert_peg",
        "certification_runs": 3,
        "certification_timeout_seconds": 123,
        "source_recorder_pid": 9876,
    }


def test_deferred_recovery_never_adopts_another_live_collector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _pending_replay_candidate(tmp_path)
    (demo / "replay_manifest.json").write_text(
        json.dumps({"recording": {"recorder_pid": 4321}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        recorder,
        "_process_command",
        lambda pid: (
            ("python", "/repo/scripts/record_demos_zarr.py")
            if pid == 4321
            else ()
        ),
    )
    monkeypatch.setattr(
        recorder,
        "_launch_deferred_physical_replay_certification_batch",
        lambda *_args, **_kwargs: pytest.fail(
            "one collector adopted another live collector's pending demo"
        ),
    )
    summary: dict[str, object] = {}

    recovered = recorder._ensure_deferred_physical_replay_certification(
        tmp_path,
        exp_name="t09_insert_peg",
        certification_runs=3,
        certification_timeout_seconds=123,
        source_recorder_pid=9876,
        recovery_summary=summary,
    )

    assert recovered == 0
    assert summary["live_source_skipped_count"] == 1


def test_stable_v2_demo_directory_and_marker_are_published_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def write_complete_demo(*args, **kwargs):
        demo_dir = Path(args[3])
        generation = kwargs["dataset_generation"]
        episode_id = kwargs["published_demo_name"]
        observed["generation"] = generation
        observed["episode_id"] = episode_id
        (demo_dir / "replay.zarr").mkdir(parents=True)
        videos_dir = demo_dir / "videos"
        videos_dir.mkdir()
        for key in recorder.CANONICAL_CAMERA_KEYS:
            (videos_dir / f"{key}.mp4").write_bytes(b"complete video")
        recorder._write_dataset_generation_marker(
            demo_dir,
            generation,
            episode_id=episode_id,
        )

    monkeypatch.setattr(recorder, "_write_demo_contents", write_complete_demo)
    monkeypatch.setattr(
        recorder,
        "FLAGS",
        type("Flags", (), {"save_camera_info": False, "save_depth": False})(),
    )

    published = Path(
        recorder._write_demo_zarr_and_videos(
            [],
            "t50_water_plant",
            1,
            tmp_path,
            30,
            env=_stable_v2_env(),
        )
    )

    assert published.name.startswith("t50_water_plant_demo_1_")
    assert published.name.endswith("__stable_v2")
    assert published.name.split("_demo_", 1)[0] == "t50_water_plant"
    assert observed["episode_id"] == published.name
    generation = observed["generation"]
    assert isinstance(generation, recorder.DatasetGeneration)
    assert generation.is_stable_v2 is True
    marker = json.loads(
        (published / "DATASET_STABLE_V2.json").read_text(encoding="utf-8")
    )
    assert marker["episode_id"] == published.name
    assert not recorder._recording_staging_parent(tmp_path).exists()


def test_physx_v3_demo_directory_has_a_distinct_visible_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def write_complete_demo(*args, **kwargs):
        demo_dir = Path(args[3])
        generation = kwargs["dataset_generation"]
        episode_id = kwargs["published_demo_name"]
        (demo_dir / "replay.zarr").mkdir(parents=True)
        videos_dir = demo_dir / "videos"
        videos_dir.mkdir()
        for key in recorder.CANONICAL_CAMERA_KEYS:
            (videos_dir / f"{key}.mp4").write_bytes(b"complete video")
        recorder._write_dataset_generation_marker(
            demo_dir,
            generation,
            episode_id=episode_id,
        )

    env = _stable_v2_env()
    env.unwrapped.hard_reset_physx_scene = lambda *, seed: {
        "scene_rebuilt": True,
        "scene_generation": 1,
        "seed": seed,
    }
    monkeypatch.setenv(recorder.ISAAC_IN_PROCESS_HARD_RESET_ENV, "1")
    monkeypatch.setenv(
        recorder.DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_ENV,
        "1",
    )
    monkeypatch.setenv("HANDSON_RECORD_PHYSX_GROUND_TRUTH", "1")
    monkeypatch.setattr(recorder, "_write_demo_contents", write_complete_demo)
    monkeypatch.setattr(
        recorder,
        "FLAGS",
        type("Flags", (), {"save_camera_info": False, "save_depth": False})(),
    )

    published = Path(
        recorder._write_demo_zarr_and_videos(
            [],
            "t26_place_fruit_in_bucket",
            1,
            tmp_path,
            30,
            env=env,
        )
    )

    assert published.name.startswith("t26_place_fruit_in_bucket_demo_1_")
    assert published.name.endswith("__physx_v3")
    marker = json.loads(
        (published / "DATASET_PHYSX_V3.json").read_text(encoding="utf-8")
    )
    assert marker["episode_id"] == published.name
    assert marker["is_physx_v3_data"] is True


def test_stable_v2_generation_is_repeated_in_all_episode_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PublishedSpool:
        frame_count = 1
        depth_frame_count = 0
        video_crf = 18

        def publish_videos(self, videos_dir):
            videos_dir = Path(videos_dir)
            videos_dir.mkdir(parents=True)
            frame = _solid_rgb(10)
            sources = {}
            first_frames = {}
            for key in recorder.CANONICAL_CAMERA_KEYS:
                (videos_dir / f"{key}.mp4").write_bytes(b"complete video")
                sources[key] = key
                first_frames[key] = frame
            return sources, first_frames

    monkeypatch.setattr(
        recorder,
        "FLAGS",
        SimpleNamespace(
            arm_type="rokae",
            auto_c01_arm_video=False,
            auto_plan_arm_video=False,
            c01_arm_collision_penetration_tolerance=0.001,
            c01_arm_collision_sweep_joint_step=0.05,
            c01_arm_max_fk_orientation_error=0.1,
            c01_arm_max_fk_position_error=0.01,
            camera_screen_effect=False,
            data_fps=30.0,
            fisheye_intrinsics_yaml="",
            hand_type="wuji1",
            ipad_screen_effect=False,
            randomize=False,
            randomize_dynamics=False,
            record_physx_ground_truth=True,
            render_mode="rgb_array",
            save_camera_info=False,
            save_depth=False,
            scripted_policy=False,
            sim_backend="isaacsim",
            target_label="",
        ),
    )
    env = _stable_v2_env()
    env.unwrapped.arm_type = "rokae"
    env.unwrapped.hand_type = "wuji1"
    env.unwrapped.image_obs = True
    env.unwrapped.recorded_drive_target_replay_supported = True
    generation = recorder._dataset_generation_for_env(env)
    episode_id = "d02_test_demo_1_timestamp__stable_v2"
    demo_dir = tmp_path / episode_id
    trajectory = [
        {
            "observations": {},
            "next_observations": {},
            "actions": np.zeros(3, dtype=np.float64),
            "dones": True,
            "infos": {"succeed": True},
        }
    ]

    monkeypatch.setenv(
        recorder.DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_ENV,
        "1",
    )
    monkeypatch.setenv("HANDSON_ISAACLAB_DEFER_ARM_RENDER", "1")
    recorder._write_demo_contents(
        trajectory,
        "d02_test",
        1,
        demo_dir,
        30,
        env=env,
        environment_schema={},
        required_recording_fields=set(),
        video_spool=PublishedSpool(),
        collection_run_seed=7,
        episode_reset_seed=8,
        collection_seed_source=recorder.RANDOM_COLLECTION_SEED_SOURCE,
        episode_reset_ordinal=1,
        dataset_generation=generation,
        published_demo_name=episode_id,
    )

    zarr_root = zarr.open_group(str(demo_dir / "replay.zarr"), mode="r")
    manifest = json.loads(
        (demo_dir / "replay_manifest.json").read_text(encoding="utf-8")
    )
    collection = json.loads(
        (demo_dir / "collection_meta.json").read_text(encoding="utf-8")
    )
    marker = json.loads(
        (demo_dir / "DATASET_STABLE_V2.json").read_text(encoding="utf-8")
    )
    for metadata in (
        dict(zarr_root.attrs),
        manifest["recording"],
        collection,
        marker,
    ):
        assert metadata["episode_id"] == episode_id
        assert metadata["dataset_generation"] == "stable_origin_v2"
        assert metadata["training_generation_status"] == (
            "new_requires_physical_certification"
        )
        assert metadata["dataset_directory_suffix"] == "__stable_v2"
        assert metadata["dataset_marker_file"] == "DATASET_STABLE_V2.json"
        assert metadata["is_stable_v2_data"] is True
    assert collection["data_code"] == episode_id
    recording = manifest["recording"]
    assert recording["physical_replay_certification_required"] is True
    assert recording["physical_replay_certification_mode"] == (
        recorder.DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE
    )
    assert collection["physical_replay_certification_required"] is True
    assert collection["physical_replay_certification_mode"] == (
        recorder.DEFERRED_PHYSICAL_REPLAY_CERTIFICATION_MODE
    )
    assert recording["recorder_lifecycle"] == "persistent_process"
    assert recording["source_process_reuse_enabled"] is True
    assert "source_process_reused_across_episodes" not in recording
    assert recording["fresh_recorder_process_per_episode_required"] is False
    assert recording["fresh_recorder_process_per_attempt_required"] is False
    assert recording["fresh_scene_replay_required"] is True
    assert recording["fresh_certifier_process_per_run_required"] is True
    assert recording["publish_only_after_fresh_certification"] is True
    assert recording["isaac_arm_render_deferred"] is True
    assert collection["isaac_arm_render_deferred"] is True
    assert recording["trajectory_input_mode"] == (
        recorder.RECORDED_DRIVE_TARGET_INPUT_MODE
    )
    assert recording["collection_seed_source"] == (
        recorder.RANDOM_COLLECTION_SEED_SOURCE
    )
    assert recording["episode_reset_ordinal"] == 1
    assert recording["episode_seed_derivation"] == (
        recorder.EPISODE_SEED_DERIVATION
    )
    assert recording["recorder_pid"] > 0
    for field in (
        "publication_contract",
        "postprocess_policy",
        "postprocess_manual_only",
        "offline_arm_render_deferred",
        "offline_arm_render_status",
    ):
        assert field not in recording
        assert field not in collection


@pytest.mark.parametrize(
    "task_name",
    ["t08_hammer_nail", "plan8_peg_in_socket"],
)
def test_manual_episode_binds_verified_v3_and_defers_offline_arm_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_name: str,
) -> None:
    monkeypatch.setattr(
        recorder,
        "FLAGS",
        _episode_test_flags(sim_backend="mujoco"),
    )
    monkeypatch.setenv(recorder.POSTPROCESS_MANUAL_ONLY_ENV, "1")
    original_run = recorder.subprocess.run

    def unexpected_subprocess(command, *args, **kwargs):
        rendered = " ".join(str(part) for part in command)
        if any(
            script in rendered
            for script in (
                "c01_compute_arm_plan_qpos.py",
                "c01_render_arm_replay.py",
                "plan_render_arm_replay.py",
            )
        ):
            raise AssertionError(
                "manual-only episode save started an arm renderer"
            )
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(recorder.subprocess, "run", unexpected_subprocess)
    demo_dir = _write_one_frame_test_episode(
        tmp_path,
        task_name=task_name,
    )

    manifest = json.loads(
        (demo_dir / "replay_manifest.json").read_text(encoding="utf-8")
    )
    collection = json.loads(
        (demo_dir / "collection_meta.json").read_text(encoding="utf-8")
    )
    recording = manifest["recording"]
    expected = {
        "publication_contract": "handson.verified_episode.v3",
        "postprocess_policy": "manual_review_render_upload_v1",
        "postprocess_manual_only": True,
        "offline_arm_render_deferred": True,
        "offline_arm_render_status": "pending",
    }
    for key, value in expected.items():
        assert recording[key] == value
        assert collection[key] == value
    assert not (demo_dir / "videos_arm").exists()


@pytest.mark.parametrize(
    ("task_name", "sim_backend", "expected_stages"),
    [
        (
            "e01_fold_garment",
            "genesis",
            ["ipc_replay", "ipc_render", "arm_ik"],
        ),
        (
            "f01_cotton_cloth_mvp",
            "isaacsim",
            ["ipc_replay", "ipc_render"],
        ),
    ],
)
def test_manual_episode_writes_pending_e_or_f_postprocess_declaration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_name: str,
    sim_backend: str,
    expected_stages: list[str],
) -> None:
    monkeypatch.setattr(
        recorder,
        "FLAGS",
        _episode_test_flags(sim_backend=sim_backend),
    )
    monkeypatch.setenv(recorder.POSTPROCESS_MANUAL_ONLY_ENV, "1")
    demo_dir = _write_one_frame_test_episode(
        tmp_path,
        task_name=task_name,
    )

    status = json.loads(
        (demo_dir / "e01_postprocess.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (demo_dir / "replay_manifest.json").read_text(encoding="utf-8")
    )
    collection = json.loads(
        (demo_dir / "collection_meta.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "pending"
    assert status["operator_deferred"] is True
    assert status["background"] is False
    assert status["requested_stages"] == expected_stages
    assert collection["e01_postprocess_status"] == "pending"
    assert collection["e01_postprocess"] == status
    assert collection["e01_postprocess_background"] is False
    for metadata in (manifest["recording"], collection):
        assert metadata["publication_contract"] == (
            "handson.verified_episode.v3"
        )
        assert metadata["postprocess_policy"] == (
            "manual_review_render_upload_v1"
        )
        assert metadata["postprocess_manual_only"] is True
        assert metadata["offline_arm_render_deferred"] is False
        assert metadata["offline_arm_render_status"] == "not_required"


def test_demo_staging_stays_on_output_root_for_atomic_publish(tmp_path: Path) -> None:
    staging_parent = recorder._recording_staging_parent(tmp_path)

    assert staging_parent == tmp_path.resolve() / ".recording_staging"
    assert staging_parent.parent.stat().st_dev == tmp_path.stat().st_dev


def test_depth_spool_restores_and_writes_npz_without_frame_stack_in_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeWriter:
        def __init__(self):
            self.frames = []
            self.output = None

        def start(self, output):
            self.output = Path(output)
            self.output.write_bytes(b"fake mp4")

        def write_frame(self, frame):
            self.frames.append(np.asarray(frame).copy())

        def stop(self):
            pass

    writer = FakeWriter()
    monkeypatch.setattr(
        recorder.Mp4VideoWriter,
        "create_h264",
        lambda **_kwargs: writer,
    )

    source_frames = [
        np.asarray([[0.5, 1.0], [1.5, np.inf]], dtype=np.float32),
        np.asarray([[0.75, 1.25], [1.75, 2.0]], dtype=np.float32),
    ]
    with recorder.RecordingSpool(tmp_path, video_fps=30) as spool:
        refs = [spool.spool_depth(frame) for frame in source_frames]
        assert all(isinstance(ref, recorder.SpooledFrameRef) for ref in refs)
        assert all(not isinstance(ref, np.ndarray) for ref in refs)
        outputs = recorder._write_spooled_depth_outputs(
            {"main": refs},
            tmp_path / "videos",
            video_fps=30,
            expected_frame_count=2,
        )

    npz_path = tmp_path / "videos" / "main_depth.npz"
    with np.load(npz_path) as archive:
        restored = archive["depth"]
    np.testing.assert_equal(restored, np.stack(source_frames, axis=0))
    assert restored.dtype == np.float32
    assert restored.shape == (2, 2, 2)
    assert len(writer.frames) == 2
    assert all(frame.shape == (2, 2, 3) for frame in writer.frames)
    assert any(str(npz_path) == output for output in outputs)
    assert not list((tmp_path / "videos").glob(".*depth*"))
