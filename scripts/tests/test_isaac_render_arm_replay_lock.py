from __future__ import annotations

import multiprocessing
import json
import os
import subprocess
import sys
from pathlib import Path
from queue import Empty

from handson.data.episode_publication_lock import (
    INHERITED_EPISODE_PUBLICATION_LOCK_FD_ENV,
    episode_publication_lock,
    episode_publication_lock_path,
    episode_publication_scope,
)
from handson.data.deferred_arm_replay_artifacts import (
    deferred_arm_replay_artifact_errors,
)


def _enter_render_publication_scope(
    demo_dir: str,
    inherited_lock_fd: int | None,
    queue,
) -> None:
    environ: dict[str, str] = {}
    if inherited_lock_fd is not None:
        environ[INHERITED_EPISODE_PUBLICATION_LOCK_FD_ENV] = str(
            inherited_lock_fd
        )
    with episode_publication_scope(Path(demo_dir), environ=environ):
        queue.put("entered")


def test_recorded_replay_settings_preserve_randomized_episode_seed(
    tmp_path: Path,
) -> None:
    demo_dir = tmp_path / "randomized"
    demo_dir.mkdir()
    (demo_dir / "replay_manifest.json").write_text(
        json.dumps(
            {
                "recording": {
                    "randomize": True,
                    "collection_run_seed": 101,
                    "episode_reset_seed": 109,
                }
            }
        ),
        encoding="utf-8",
    )

    program = (
        "import json, sys; from pathlib import Path; "
        "from scripts.isaac_render_arm_replay import _recorded_replay_settings; "
        "print(json.dumps(_recorded_replay_settings(Path(sys.argv[1]))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(demo_dir)],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [True, 101, 109]


def test_randomized_replay_fails_closed_without_episode_seed(
    tmp_path: Path,
) -> None:
    demo_dir = tmp_path / "missing-seed"
    demo_dir.mkdir()
    (demo_dir / "replay_manifest.json").write_text(
        json.dumps({"recording": {"randomize": True}}),
        encoding="utf-8",
    )

    program = (
        "import sys; from pathlib import Path; "
        "from scripts.isaac_render_arm_replay import _recorded_replay_settings; "
        "_recorded_replay_settings(Path(sys.argv[1]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(demo_dir)],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode != 0
    assert "episode_reset_seed" in result.stderr


def test_canonical_camera_postrender_uses_visual_only_newton_mode() -> None:
    program = r'''
import json
import os

import scripts.isaac_render_arm_replay as renderer

renderer.FLAGS(["camera-postrender-test"])
os.environ["HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY"] = "1"
os.environ["HANDSON_BC011_COLLECTION_ANGLE_GUIDANCE"] = "1"
newton_mode = renderer._configure_canonical_camera_postrender(
    source_backend="isaacsim_newton"
)
newton = {
    "mode": newton_mode,
    "sim_backend": renderer.FLAGS.sim_backend,
    "restore_state": renderer.FLAGS.restore_state,
    "state_guided": renderer.FLAGS.state_guided,
    "kinematic_visual_replay": renderer.FLAGS.kinematic_visual_replay,
    "global_ik": os.environ.get("HANDSON_GLOBAL_IK_PLAN"),
    "defer_arm": os.environ.get("HANDSON_ISAACLAB_DEFER_ARM_RENDER"),
    "defer_camera": os.environ.get("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"),
    "direct_hand": os.environ.get("HANDSON_ISAACLAB_DIRECT_HAND_CONTROL"),
    "main_only_present": "HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY" in os.environ,
    "angle_guidance_present": (
        "HANDSON_BC011_COLLECTION_ANGLE_GUIDANCE" in os.environ
    ),
}
physx_mode = renderer._configure_canonical_camera_postrender(
    source_backend="isaacsim"
)
physx = {
    "mode": physx_mode,
    "state_guided": renderer.FLAGS.state_guided,
    "kinematic_visual_replay": renderer.FLAGS.kinematic_visual_replay,
}
simplified_mode = renderer._configure_canonical_camera_postrender(
    source_backend="isaacsim",
    simplified_post_ik=True,
)
simplified = {
    "mode": simplified_mode,
    "arm_type": os.environ.get("HANDSON_ARM_TYPE"),
    "hand_type": os.environ.get("HANDSON_HAND_TYPE"),
    "state_guided": renderer.FLAGS.state_guided,
    "kinematic_visual_replay": renderer.FLAGS.kinematic_visual_replay,
    "defer_arm": os.environ.get("HANDSON_ISAACLAB_DEFER_ARM_RENDER"),
    "defer_camera": os.environ.get("HANDSON_ISAACLAB_DEFER_CAMERA_RENDER"),
    "direct_hand": os.environ.get("HANDSON_ISAACLAB_DIRECT_HAND_CONTROL"),
}
as019_mode = renderer._configure_canonical_camera_postrender(
    source_backend="isaacsim",
    exact_physx_visual_state=True,
)
as019 = {
    "mode": as019_mode,
    "restore_state": renderer.FLAGS.restore_state,
    "state_guided": renderer.FLAGS.state_guided,
    "kinematic_visual_replay": renderer.FLAGS.kinematic_visual_replay,
    "ground_truth": os.environ.get("HANDSON_RECORD_PHYSX_GROUND_TRUTH"),
}
print(json.dumps({"newton": newton, "physx": physx, "simplified": simplified, "as019": as019}, sort_keys=True))
'''
    result = subprocess.run(
        [sys.executable, "-c", program],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["newton"] == {
        "mode": "kinematic_visual",
        "sim_backend": "isaacsim",
        "restore_state": True,
        "state_guided": False,
        "kinematic_visual_replay": True,
        "global_ik": "0",
        "defer_arm": "0",
        "defer_camera": "0",
        "direct_hand": "0",
        "main_only_present": False,
        "angle_guidance_present": False,
    }
    assert payload["physx"] == {
        "mode": "state_guided",
        "state_guided": True,
        "kinematic_visual_replay": False,
    }
    assert payload["simplified"] == {
        "mode": "kinematic_visual",
        "arm_type": "rokae",
        "hand_type": "wuji1",
        "state_guided": False,
        "kinematic_visual_replay": True,
        "defer_arm": "0",
        "defer_camera": "0",
        "direct_hand": "0",
    }
    assert payload["as019"] == {
        "mode": "physx_state_restore_visual",
        "restore_state": False,
        "state_guided": False,
        "kinematic_visual_replay": False,
        "ground_truth": "1",
    }


def test_as019_render_loads_and_passes_complete_physx_visual_trace() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts" / "isaac_render_arm_replay.py").read_text(
        encoding="utf-8"
    )
    assert "task_name == _AS019_TASK_NAME" in source
    assert "_load_physx_replay_trajectory" in source
    assert "physx_replay=physx_visual_replay" in source


def test_manual_render_entry_waits_for_episode_publication_lock(
    tmp_path: Path,
) -> None:
    demo_dir = tmp_path / "a01_demo"
    demo_dir.mkdir()
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    process = context.Process(
        target=_enter_render_publication_scope,
        args=(str(demo_dir), None, queue),
    )

    with episode_publication_lock(demo_dir, blocking=False):
        process.start()
        try:
            queue.get(timeout=0.3)
        except Empty:
            pass
        else:
            raise AssertionError("manual renderer bypassed the publication lock")
        assert process.is_alive()

    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
    assert process.exitcode == 0
    assert queue.get(timeout=1) == "entered"


def test_worker_child_inherits_verified_parent_lock_without_self_deadlock(
    tmp_path: Path,
) -> None:
    demo_dir = tmp_path / "a01_demo"
    demo_dir.mkdir()
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    with episode_publication_lock(
        demo_dir,
        blocking=False,
    ) as publication_lock:
        process = context.Process(
            target=_enter_render_publication_scope,
            args=(str(demo_dir), publication_lock.fileno(), queue),
        )
        process.start()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        assert process.exitcode == 0
        assert queue.get(timeout=1) == "entered"


def test_exec_child_accepts_exact_passed_locked_descriptor(
    tmp_path: Path,
) -> None:
    demo_dir = tmp_path / "a01_demo"
    demo_dir.mkdir()
    program = (
        "from pathlib import Path; "
        "from handson.data.episode_publication_lock import "
        "episode_publication_scope; "
        "import sys; "
        "scope = episode_publication_scope(Path(sys.argv[1])); "
        "scope.__enter__(); print('entered', flush=True); scope.__exit__(None, None, None)"
    )
    with episode_publication_lock(
        demo_dir,
        blocking=False,
    ) as publication_lock:
        inherited_fd = publication_lock.fileno()
        child_env = os.environ.copy()
        child_env[INHERITED_EPISODE_PUBLICATION_LOCK_FD_ENV] = str(
            inherited_fd
        )
        result = subprocess.run(
            [sys.executable, "-c", program, str(demo_dir)],
            env=child_env,
            pass_fds=(inherited_fd,),
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "entered"


def test_foreign_holder_and_forged_same_inode_fd_cannot_bypass_lock(
    tmp_path: Path,
) -> None:
    demo_dir = tmp_path / "a01_demo"
    demo_dir.mkdir()
    program = (
        "from pathlib import Path; "
        "from handson.data.episode_publication_lock import "
        "episode_publication_scope; "
        "import sys; "
        "scope = episode_publication_scope(Path(sys.argv[1])); "
        "scope.__enter__(); print('entered', flush=True); scope.__exit__(None, None, None)"
    )
    process: subprocess.Popen[str] | None = None

    with episode_publication_lock(demo_dir, blocking=False):
        # This descriptor identifies the exact lock-file inode but does not
        # own its flock.  The former PID+busy contract incorrectly trusted an
        # equivalent assertion.  The inherited-FD validator must reject it
        # while the foreign open-file description above owns the lock.
        forged_handle = episode_publication_lock_path(demo_dir).open("a+")
        child_env = os.environ.copy()
        child_env[INHERITED_EPISODE_PUBLICATION_LOCK_FD_ENV] = str(
            forged_handle.fileno()
        )
        process = subprocess.Popen(
            [sys.executable, "-c", program, str(demo_dir)],
            env=child_env,
            pass_fds=(forged_handle.fileno(),),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            process.communicate(timeout=0.3)
        except subprocess.TimeoutExpired:
            pass
        else:
            raise AssertionError("foreign-holder capability bypassed the lock")
        assert process.poll() is None

    assert process is not None
    try:
        stdout, stderr = process.communicate(timeout=5)
    finally:
        forged_handle.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)
    assert process.returncode == 0, stderr
    assert stdout.strip() == "entered"


def test_manual_renderer_invalidates_cloud_ack_and_records_final_video_proof(
    tmp_path: Path,
) -> None:
    demo_dir = tmp_path / "a01_demo"
    videos_dir = demo_dir / "videos"
    videos_dir.mkdir(parents=True)
    for name in ("main.mp4", "wrist_left.mp4", "wrist_right.mp4"):
        (videos_dir / name).write_bytes(f"final-{name}".encode())
    (demo_dir / "deferred_arm_replay_status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "artifact_validation_passed": True,
                "artifact_proof": {"files": [{"path": "stale"}]},
                "artifact_quality_profile": "stale-profile",
                "artifact_quality_evidence_sha256": "0" * 64,
                "error": "stale failure from an earlier attempt",
            }
        ),
        encoding="utf-8",
    )
    (demo_dir / "cloud_upload_meta.json").write_text(
        json.dumps(
            {
                "status": "uploaded",
                "cloud_ack": "confirmed",
                "cloud_ack_at": "old",
            }
        ),
        encoding="utf-8",
    )

    program = (
        "from pathlib import Path; import sys; "
        "from scripts.isaac_render_arm_replay import "
        "_prepare_arm_video_replacement, _mark_arm_render_complete; "
        "demo = Path(sys.argv[1]); "
        "_prepare_arm_video_replacement(demo); "
        "(demo / 'replacing_snapshot.json').write_text("
        "(demo / 'deferred_arm_replay_status.json').read_text()); "
        "_mark_arm_render_complete(demo)"
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(demo_dir)],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    replacing = json.loads((demo_dir / "replacing_snapshot.json").read_text())
    assert replacing["publication_phase"] == "replacing"
    assert replacing["artifact_validation_passed"] is False
    assert "artifact_proof" not in replacing
    assert "artifact_quality_profile" not in replacing
    assert "artifact_quality_evidence_sha256" not in replacing
    assert "error" not in replacing

    complete = json.loads(
        (demo_dir / "deferred_arm_replay_status.json").read_text()
    )
    assert complete["status"] == "complete"
    assert complete["publication_phase"] == "complete"
    assert complete["artifact_validation_passed"] is True
    assert "error" not in complete
    assert not deferred_arm_replay_artifact_errors(
        demo_dir,
        complete["artifact_proof"],
    )
    cloud = json.loads((demo_dir / "cloud_upload_meta.json").read_text())
    assert cloud["status"] == "stale_local_artifacts"
    assert cloud["cloud_ack"] == "stale"
    assert cloud["needs_reupload"] is True
    assert "cloud_ack_at" not in cloud
