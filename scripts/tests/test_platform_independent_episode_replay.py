from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from scripts import platform_independent_episode_replay as episode


@dataclass(frozen=True)
class FakeDemo:
    environment_variables: dict[str, str]
    source_scene_rebuild_per_reset: bool = True
    source_scene_generation: int = 153
    episode_reset_seed: int = 760963625
    image_obs: bool = True
    task_name: str = "t23_photograph_logo"
    actions: tuple[int, ...] = (1, 2, 3)


class FakeRawEnvironment:
    def __init__(self) -> None:
        self._control_dt = 1.0 / 30.0
        self._physx_scene_generation = 0
        self.image_obs = False
        self.hz = 0
        self.constructor_values: tuple[bool, int, int] | None = None

    def hard_reset_physx_scene(self, *, seed: int):
        generation = self._physx_scene_generation + 1
        self.constructor_values = (self.image_obs, self.hz, generation)
        # The real hard reset clears/reconstructs this object's attributes.
        self._physx_scene_generation = generation
        self.image_obs = True
        self.hz = 30
        return {
            "scene_rebuilt": True,
            "scene_generation": generation,
            "seed": seed,
            "process_restarted": False,
            "simulation_app_reused": True,
        }


def test_scene_rebuild_reproduces_only_the_target_episode_boundary() -> None:
    raw = FakeRawEnvironment()
    demo = FakeDemo(environment_variables={})

    report = episode.reproduce_recorded_scene_rebuild(raw, demo)

    assert raw.constructor_values == (True, 30, 153)
    assert raw._physx_scene_generation == 153
    assert raw.image_obs is False
    assert raw.hz == 0
    assert report == {
        "scene_rebuilt": True,
        "scene_generation": 153,
        "seed": 760963625,
        "process_restarted": False,
        "simulation_app_reused": True,
    }


def test_portable_environment_replaces_recording_computer_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "recorded-source"
    monkeypatch.setenv(
        "HANDSON_PLATFORM_REPLAY_ISAACSIM_PYTHON",
        "/local/isaac/python.sh",
    )
    demo = FakeDemo(
        environment_variables={
            "HANDSON_ROOT": "/recording/computer/handson",
            "HANDSON_FISHEYE_INTRINSICS_YAML": (
                "/recording/computer/handson/configs/camera.yaml"
            ),
            "HANDSON_COLLECTION_SESSION_ID": "old-session",
            "HANDSON_ISAAC_STARTUP_READY_FILE": "/tmp/old-ready.json",
            "HANDSON_ISAACLAB_CONTROL_HZ": "30",
        }
    )

    portable = episode._portable_demo_environment(demo, source_root)

    assert portable.environment_variables["HANDSON_ROOT"] == str(source_root)
    assert portable.environment_variables["HANDSON_FISHEYE_INTRINSICS_YAML"] == str(
        source_root / "configs/camera.yaml"
    )
    assert portable.environment_variables["HANDSON_ISAACSIM_PYTHON"] == (
        "/local/isaac/python.sh"
    )
    assert portable.environment_variables["HANDSON_AUTO_START_TELEOP"] == "0"
    assert portable.environment_variables["HANDSON_ISAACLAB_CONTROL_HZ"] == "30"
    assert "HANDSON_COLLECTION_SESSION_ID" not in portable.environment_variables
    assert "HANDSON_ISAAC_STARTUP_READY_FILE" not in portable.environment_variables


def test_legacy_recorded_certifier_runs_without_platform_review_keyword() -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def legacy_function(*args, **kwargs):
        calls.append((args, kwargs))
        return "legacy-result"

    result = episode._call_with_platform_review(
        legacy_function,
        "demo",
        required=True,
        on_environment_created="callback",
    )

    assert result == "legacy-result"
    assert calls == [
        (("demo",), {"on_environment_created": "callback"}),
    ]


def test_legacy_compatibility_bypasses_only_host_and_source_capture() -> None:
    recorded_git = {"commit": "a" * 40, "dirty": False}
    recorded_runtime = {"python": "recording-host"}
    demo = SimpleNamespace(
        recorded_git=recorded_git,
        recorded_runtime_identity=recorded_runtime,
    )

    def run_certification(_demo, *, on_environment_created=None):
        del on_environment_created

    original_git = lambda *_args, **_kwargs: {"commit": "current"}
    original_runtime = lambda: {"python": "current-host"}
    certifier = SimpleNamespace(
        run_certification=run_certification,
        capture_git_replay_state=original_git,
        capture_runtime_replay_identity=original_runtime,
    )

    with episode._legacy_nonphysical_gate_overrides(certifier, demo):
        assert certifier.capture_git_replay_state("ignored") == recorded_git
        assert certifier.capture_runtime_replay_identity() == recorded_runtime

    assert certifier.capture_git_replay_state is original_git
    assert certifier.capture_runtime_replay_identity is original_runtime


def test_episode_writes_report_and_result_before_closing_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    source_root = tmp_path / "source"
    source_root.mkdir()
    result_path = tmp_path / "result.json"
    report_path = demo_dir / episode.REPORT_NAME
    demo = FakeDemo(environment_variables={"HANDSON_ROOT": str(source_root)})
    raw = FakeRawEnvironment()

    class Environment:
        unwrapped = raw

        def close(self):
            assert report_path.is_file()
            assert result_path.is_file()

    environment = Environment()

    def load_demo(_path, *, platform_review):
        assert platform_review is True
        return demo

    def run_certification(
        loaded_demo,
        *,
        on_environment_created,
        platform_review,
    ):
        assert loaded_demo.task_name == demo.task_name
        assert platform_review is True
        on_environment_created(environment)
        return {
            "status": "passed",
            "reason": "continuous_physx_replay_matched",
            "task": demo.task_name,
            "steps_expected": 3,
            "steps_executed": 3,
            "task_success_replay": True,
            "final_succeed": True,
            "trace_match": True,
            "transition_outcomes_match": True,
        }

    def write_report_atomic(_demo_dir, report):
        episode._write_json_atomic(report_path, report)

    certifier = SimpleNamespace(
        load_demo=load_demo,
        run_certification=run_certification,
        write_report_atomic=write_report_atomic,
    )
    monkeypatch.setattr(episode, "_import_recorded_certifier", lambda _root: certifier)

    report = episode.replay_one_episode(
        demo_dir,
        source_root=source_root,
        recorded_commit="c" * 40,
        result_path=result_path,
    )

    envelope = json.loads(result_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["session_replay"]["session_history_used"] is False
    assert report["platform_source_replay"]["recorded_commit"] == "c" * 40
    assert report["platform_source_replay"]["scene_rebuild"][
        "scene_generation"
    ] == 153
    assert envelope["report"] == report
