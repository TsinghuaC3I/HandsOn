"""Exercise the recorder's real shutdown block without starting Kit."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from handson.data import physx_session_history as history
from scripts import record_demos_zarr as recorder


@pytest.mark.parametrize("with_error", [False, True])
@pytest.mark.parametrize("has_trajectory", [False, True])
def test_shutdown_preserves_unfinished_actions_before_closing_simulator(
    tmp_path: Path, with_error: bool, has_trajectory: bool,
) -> None:
    journal = history.PhysxSessionHistoryJournal.create(
        tmp_path / "history", session_id="z01-stop-test",
        task_name="z01_pipette", collection_run_seed=387808937,
    )
    journal.record_reset_attempt(
        seed=387808937, ordinal=0, outcome="accepted", stable_kwargs={},
        stable_origin={"physics_steps": 120}, start_sim_time=0., end_sim_time=1.,
    )
    actions = np.arange(3 * 54, dtype=np.float64).reshape(3, 54)
    def observation(index):
        targets = np.full(54, index / 100., dtype=np.float64)
        return {
            "physx_sim_time": np.array([1. + index / 30.]),
            "physx_robot_joint_target": targets,
            "physx_robot_joint_target_valid": np.ones(54, dtype=np.uint8),
            "physx_substep_count": np.array([4]),
            "physx_substep_robot_joint_target": np.tile(targets, (4, 1)),
            "physx_substep_robot_joint_target_valid": np.ones((4, 54), dtype=np.uint8),
            "physx_task_controller_state": np.full(12, index, dtype=np.float64),
        }
    transitions = [
        {"observations": observation(i), "next_observations": observation(i + 1),
         "actions": actions[i], "dones": False, "infos": {"succeed": False}}
        for i in range(3)
    ]
    if not has_trajectory:
        # A completed/rejected episode has already emptied the live buffer.
        journal.record_trajectory([], reason="operator_keyboard_reset", reset_ordinal=0,
                                  start_sim_time=1., end_sim_time=1.)
    original_error = RuntimeError("operator input failed") if with_error else None
    closed = []
    def close_env():
        # Persistence must be complete before Kit releases the in-memory data.
        assert (journal.path / "FINALIZED.json").is_file()
        closed.append(True)

    namespace = dict(vars(recorder))
    namespace.update(
        physx_session_journal=journal, base_out=tmp_path, trajectory=transitions if has_trajectory else [],
        guidance_mode=False, collection_error=original_error, episode_reset_ordinal=0,
        environment_recording_schema=None, required_recording_fields=set(observation(0)),
        manus_raw_recorder=None, trajectory_spool=None, pbar=None, wrist_cam_viewer=None,
        live_preview_publisher=None, external_live_preview_process=None, saved_demo_dirs=[],
        poisoned_isaac_backend=False, FLAGS=SimpleNamespace(sim_backend="isaacsim"),
        task_id="z01_pipette", env=SimpleNamespace(close=close_env),
        previous_stop_handlers={}, manual_reset_bridge=SimpleNamespace(restore=lambda: None),
        cv2=SimpleNamespace(destroyAllWindows=lambda: None, error=RuntimeError),
    )
    # Execute the production nested journal function and full finally block.
    # This catches omission/order regressions in main, while real Zarr writes
    # and session finalization exercise persistence rather than a mocked save.
    tree = ast.parse(Path(recorder.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    save = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "_journal_episode_history")
    shutdown = next(n for n in main.body if isinstance(n, ast.Try) and n.finalbody
                    and "manus_raw_recorder" in ast.unparse(n.finalbody[0]))
    code = ast.Module(body=[save, *shutdown.finalbody], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), recorder.__file__, "exec"), namespace)

    loaded = history.load_session_history(journal.path)
    assert closed == [True]
    assert namespace["collection_error"] is original_error
    assert len(loaded["events"]) == 2
    event = loaded["events"][-1]
    assert event["payload"]["accepted"] is False
    if has_trajectory:
        assert event["payload"]["reason"] == ("collection_error" if with_error else "collection_stopped")
        data = history.load_discarded_trajectory(journal.path, event["ordinal"])
        np.testing.assert_array_equal(data["action"], actions)
        np.testing.assert_array_equal(data["succeed"], 0)
        np.testing.assert_array_equal(data["terminal"], 0)
        np.testing.assert_array_equal(data["next_physx_substep_robot_joint_target"][2],
                                      observation(3)["physx_substep_robot_joint_target"])
        np.testing.assert_array_equal(data["next_physx_task_controller_state"][2], observation(3)["physx_task_controller_state"])
