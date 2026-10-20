"""Record the validated P0-01 controller through the standard Zarr writer."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts import record_demos_zarr as recorder
from handson.sim.isaaclab.p001_contract import (
    P001_TASK_STATE_DIM,
    p001_action_to_side_major_storage,
)

import live_smoke as runner


def _required_path(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return Path(value)


_OUTPUT_ROOT = _required_path("P001_RECORDING_OUTPUT_ROOT")
_REPORT_PATH = _required_path("P001_RECORDING_REPORT")
_ORIGIN_DIAGNOSTIC_REPORT_PATH = _REPORT_PATH.with_name(
    f"{_REPORT_PATH.stem}.origin.json"
)
_TARGET_DEG = int(os.environ.get("P001_RECORDING_TARGET_DEG", "0"))
_SEED = int(os.environ.get("P001_RECORDING_SEED", "-1"))
_DIAGNOSTIC_ORIGIN = (
    os.environ.get("P001_RECORDING_DIAGNOSTIC_ORIGIN", "0").strip() == "1"
)
if _TARGET_DEG not in (60, 90, 120):
    raise ValueError("P001_RECORDING_TARGET_DEG must be 60, 90, or 120")
if _SEED < 0:
    raise ValueError("P001_RECORDING_SEED must be nonnegative")

os.environ["HANDSON_RECORD_PHYSX_GROUND_TRUTH"] = "1"
if _DIAGNOSTIC_ORIGIN:
    # The standard fresh-worker path deliberately uses os._exit(75), which is
    # correct for formal collection but bypasses Python exception reporting.
    # A diagnostic launch keeps every stability criterion unchanged and only
    # lets the final exception propagate so its structured contact evidence can
    # be persisted below.
    os.environ.pop("HANDSON_FRESH_CERTIFIED_COLLECTION_WORKER", None)
else:
    os.environ["HANDSON_FRESH_CERTIFIED_COLLECTION_WORKER"] = "1"
os.environ[recorder.STABLE_ORIGIN_ENV] = "1"
os.environ.setdefault("HANDSON_COLLECTOR_ID", "p001_scripted_validation")

if not recorder.FLAGS.is_parsed():
    recorder.FLAGS(
        [
            "p001_recording_live_smoke",
            "--exp_name=t48_unfold_fan",
            "--successes_needed=1",
            f"--out_dir={_OUTPUT_ROOT}",
            "--video_fps=30",
            "--data_fps=30",
            "--render_mode=rgb_array",
            "--randomize=true",
            "--randomize_dynamics=false",
            "--save_depth=false",
            "--save_camera_info=true",
            f"--target_label={_TARGET_DEG}deg",
            "--arm_type=rokae",
            "--hand_type=wuji1",
            "--sim_backend=isaacsim",
            "--record_physx_ground_truth=true",
            f"--seed={_SEED}",
            "--certify_physical_replay=false",
            "--show_sim_cameras=false",
            "--preview_only=false",
        ]
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class _RecordingEnvironment:
    def __init__(self, environment: Any) -> None:
        self._environment = environment
        self._current_observation: dict[str, Any] | None = None
        self._trajectory: list[dict[str, Any]] = []
        self._spool: Any | None = None
        self._environment_schema: dict[str, Any] | None = None
        self._required_fields: set[str] = set()
        self._final_info: dict[str, Any] = {}
        self._closed = False
        self._saved_demo_dir: str | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    @property
    def unwrapped(self) -> Any:
        return self._environment.unwrapped

    def _enrich(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        enriched = dict(observation)
        raw = self._environment.unwrapped
        positions, velocities = raw._active_joint_state()
        metrics = dict(raw._p001_last_metrics or raw._refresh_metrics())
        readback = raw._p001_last_readback
        target_rad = np.asarray(
            [np.deg2rad(float(raw._p001_reset.target_open_deg))],
            dtype=np.float64,
        )
        contacts = np.asarray(
            [
                float(readback.left_end_contact_force_n),
                float(readback.right_end_contact_force_n),
                float(readback.left_wrong_end_contact_force_n),
                float(readback.right_wrong_end_contact_force_n),
                float(readback.robot_table_contact_force_n),
            ],
            dtype=np.float64,
        )
        task_state = np.asarray(
            [
                float(raw._p001_reset.seed),
                float(raw._p001_reset.root_x_m),
                float(raw._p001_reset.root_y_m),
                float(raw._p001_reset.root_yaw_deg),
                float(raw._p001_reset.initial_span_deg),
                float(raw._p001_reset.target_open_deg),
                float(raw._p001_reset.friction),
                float(raw._p001_reset.mass_scale),
                float(metrics["span_deg"]),
                float(metrics["left_span_deg"]),
                float(metrics["right_span_deg"]),
                float(metrics["balance_error_deg"]),
                float(metrics["bilateral_expansion_fraction"]),
                float(metrics["root_displacement_m"]),
                float(metrics["planarity_error_m"]),
                float(metrics["left_end_contact"]),
                float(metrics["right_end_contact"]),
                float(metrics["velocity_stable"]),
                float(raw._success_counter),
                float(raw._p001_failure_latched),
                float(raw._p001_contact_seen_this_control_frame),
                float(raw._p001_robot_table_hard_contact_substeps),
                float(raw._p001_robot_table_hard_failure_latched),
            ],
            dtype=np.float64,
        )
        if task_state.shape != (P001_TASK_STATE_DIM,):
            raise RuntimeError(
                "P0-01 recording fan_task_state schema mismatch: "
                f"expected {(P001_TASK_STATE_DIM,)}, got {task_state.shape}"
            )
        succeeded = bool(
            raw._success_counter >= raw._success_steps_required()
            and not raw._p001_failure_latched
        )
        enriched.update(
            {
                "proprioception": np.asarray(enriched["state"], dtype=np.float64),
                "contacts": contacts,
                "fan_joint_positions": np.asarray(positions, dtype=np.float64),
                "fan_joint_velocities": np.asarray(velocities, dtype=np.float64),
                "fan_target_angle": target_rad,
                "fan_task_state": task_state,
                "success": np.asarray([succeeded], dtype=np.uint8),
            }
        )
        return enriched

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            raise TypeError("P0-01 recording reset accepts keyword arguments only")
        requested_seed = int(kwargs.pop("seed", _SEED))
        if requested_seed != _SEED:
            raise ValueError(
                f"Recording seed mismatch: runner={requested_seed}, wrapper={_SEED}"
            )
        reset_attempts: list[dict[str, Any]] = []
        reset_observer = (
            lambda report: reset_attempts.append(copy.deepcopy(dict(report)))
        ) if _DIAGNOSTIC_ORIGIN else None
        try:
            observation, reset_info, _attempt_index, episode_seed = (
                recorder._reset_recording_episode_with_stable_origin(
                    self._environment,
                    kwargs,
                    sim_backend="isaacsim",
                    task_id="t48_unfold_fan",
                    collection_run_seed=_SEED,
                    reset_attempt_index=0,
                    seed_resets=True,
                    collection_seed_source=(
                        recorder.EXPLICIT_COLLECTION_SEED_SOURCE
                    ),
                    reset_attempt_observer=reset_observer,
                )
            )
        except BaseException as exc:
            _write_json(
                _ORIGIN_DIAGNOSTIC_REPORT_PATH,
                {
                    "schema": "p001_scripted_standard_recording_v1",
                    "passed": False,
                    "diagnostic_origin": _DIAGNOSTIC_ORIGIN,
                    "target_deg": _TARGET_DEG,
                    "seed": _SEED,
                    "error": f"{type(exc).__name__}: {exc}",
                    "stability_diagnostics": copy.deepcopy(
                        getattr(exc, "diagnostics", None)
                    ),
                    "reset_attempts": reset_attempts,
                },
            )
            raise
        if int(episode_seed) != _SEED:
            raise RuntimeError(
                f"Stable-origin helper changed the requested seed to {episode_seed}"
            )
        self._current_observation = self._enrich(observation)
        self._environment_schema = recorder.resolve_environment_recording_schema(
            self._environment
        )
        self._required_fields = recorder.declared_recording_field_names(
            self._environment_schema
        )
        self._spool = recorder.RecordingSpool(_OUTPUT_ROOT, video_fps=30)
        return self._current_observation, reset_info

    def step(self, action: Any) -> Any:
        if self._current_observation is None or self._spool is None:
            raise RuntimeError("Recording environment must be reset before step")
        next_observation, reward, terminated, truncated, info = (
            self._environment.step(action)
        )
        next_observation = self._enrich(next_observation)
        terminal = bool(terminated or truncated)
        self._spool.append_observation(
            self._current_observation,
            env=self._environment,
        )
        selected = recorder.select_current_observation_for_recording(
            self._current_observation,
            required_fields=self._required_fields,
            include_required_fields=not self._trajectory,
        )
        selected = self._spool.compact_observation(
            selected,
            protected_keys=self._required_fields,
        )
        self._trajectory.append(
            copy.deepcopy(
                {
                    "observations": selected,
                    "next_observations": (
                        recorder.select_next_observation_for_recording(
                            next_observation,
                            required_fields=self._required_fields,
                        )
                    ),
                    "actions": p001_action_to_side_major_storage(action),
                    "dones": terminal,
                    "infos": dict(info),
                }
            )
        )
        self._current_observation = next_observation
        self._final_info = dict(info)
        return next_observation, reward, terminated, truncated, info

    def _save(self) -> None:
        if not bool(self._final_info.get("succeed", False)):
            raise RuntimeError("Refusing to save an unsuccessful P0-01 trajectory")
        if not self._trajectory or self._spool is None:
            raise RuntimeError("P0-01 recording trajectory is empty")
        self._saved_demo_dir = recorder._write_demo_zarr_and_videos(
            self._trajectory,
            "t48_unfold_fan",
            1,
            _OUTPUT_ROOT,
            30,
            env=self._environment,
            environment_schema=self._environment_schema,
            required_recording_fields=self._required_fields,
            video_spool=self._spool,
            collection_run_seed=_SEED,
            episode_reset_seed=_SEED,
            collection_seed_source=recorder.EXPLICIT_COLLECTION_SEED_SOURCE,
            episode_reset_ordinal=0,
        )
        demo_dir = Path(self._saved_demo_dir)
        _write_json(
            _REPORT_PATH,
            {
                "schema": "p001_scripted_standard_recording_v1",
                "passed": True,
                "target_deg": _TARGET_DEG,
                "seed": _SEED,
                "trajectory_steps": len(self._trajectory),
                "demo_dir": str(demo_dir),
                "replay_zarr": str(demo_dir / "replay.zarr"),
                "recording_schema": str(demo_dir / "recording_schema.json"),
                "replay_manifest": str(demo_dir / "replay_manifest.json"),
                "videos": {
                    key: str(demo_dir / "videos" / f"{key}.mp4")
                    for key in recorder.CANONICAL_CAMERA_KEYS
                },
            },
        )

    def close(self) -> Any:
        if self._closed:
            return None
        self._closed = True
        error: BaseException | None = None
        try:
            self._save()
        except BaseException as exc:
            error = exc
            _write_json(
                _REPORT_PATH,
                {
                    "schema": "p001_scripted_standard_recording_v1",
                    "passed": False,
                    "target_deg": _TARGET_DEG,
                    "seed": _SEED,
                    "trajectory_steps": len(self._trajectory),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        finally:
            if self._spool is not None:
                self._spool.cleanup()
            self._environment.close()
        if error is not None:
            raise error
        return None


_original_get_environment = runner.TaskConfig.get_environment


def _recording_get_environment(self: Any, *args: Any, **kwargs: Any) -> Any:
    environment = _original_get_environment(self, *args, **kwargs)
    return _RecordingEnvironment(environment)


runner.TaskConfig.get_environment = _recording_get_environment


if __name__ == "__main__":
    runner.main()
