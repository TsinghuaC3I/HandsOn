#!/usr/bin/env python3
"""Inspect GEN4 grasp trajectories through public TCP/hand actions only.

This paused command session is an engineering probe, not a formal recording or
human teleoperation. Each JSON inbox command contains public action phases;
normal environment reset is the only supported physical initialization path.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inbox = args.output / "commands"
    inbox.mkdir(exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    sys.argv = [sys.argv[0], "--portable-root", str(root / ".gen4-local/kit"),
                "--/log/file=" + str((args.output / "kit.log").resolve())]
    # Live cameras are explicitly for this diagnostic's requested snapshots.
    # Formal collection continues to defer all three canonical products.
    os.environ.update(HANDSON_ENABLE_ISAACSIM_NEWTON="0",
                      HANDSON_ISAACLAB_DEFER_CAMERA_RENDER="0",
                      HANDSON_ISAACLAB_LIVE_MAIN_CAMERA_ONLY="0",
                      HANDSON_RECORD_PHYSX_GROUND_TRUTH="1")
    from handson.sim.isaaclab.gen4_env import HandsOnGEN4IsaacLabEnv
    from handson.sim.isaaclab.bimanual_assembly_env import _matrix_to_quat_wxyz
    from handson.tasks.eval_gen04_pass_glue_stick import contract as c
    from shape_queries import query_final_shapes
    from PIL import Image

    class QueryEnabledEnv(HandsOnGEN4IsaacLabEnv):
        def _configure_task_simulation_settings(self):
            super()._configure_task_simulation_settings()
            self._sim.get_physics_context().set_enable_scene_query_support(True)

    def plain(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy().tolist()
        if hasattr(value, "tolist"):
            return value.tolist()
        raise TypeError(type(value).__name__)

    def write(path, value):
        path.write_text(json.dumps(value, indent=2, default=plain) + "\n")

    env = None
    result = {"task": "GEN4", "human_teleop": False,
              "source": "public TCP/hand action commands", "commands": [],
              "completed": False, "exceptions": []}
    try:
        env = QueryEnabledEnv(
            seed=args.seed, randomize=True, image_obs=True, headless=True,
            render_mode="rgb_array", isaac_device="cuda:0",
            arm_type="rokae", hand_type="wuji1", render_resolution=(576, 1024),
        )
        env.image_obs = False
        hand_targets = {side: np.zeros(20) for side in ("left", "right")}
        terminal = False

        def state():
            robots = {}
            for side, robot in env._robots.items():
                palm = env._robot_body_pose_matrix(robot, env._ee_body_indices[side])
                robots[side] = {
                    "tcp": np.r_[palm[:3, 3], _matrix_to_quat_wxyz(palm[:3, :3])],
                    "joint_names": list(robot.joint_names),
                    "joint_positions": plain(robot.data.joint_pos),
                    "body_matrices": {name: env._robot_body_pose_matrix(robot, i)
                                      for i, name in enumerate(robot.body_names)
                                      if "finger" in name or "palm" in name},
                }
            contact_shapes = {
                side: list(env._d_hand_contact_views[(side, c.GLUE_BODY)].get_contact_force_matrix(env._sim_dt).shape)
                for side in ("left", "right")
            }
            return {"glue_pose": env._poses[c.GLUE_BODY], "tray_pose": env._poses[c.TRAY_BODY],
                    "metrics": env._gen4_metrics(), "robots": robots,
                    "contact_matrix_shapes": contact_shapes,
                    "hidden_state": env._compute_observation()["gen4_handoff_state"],
                    "sim_time": float(env._sim.current_time), "terminal": terminal}

        def snapshots(prefix):
            for camera, rgb in zip(("main", "wrist_left", "wrist_right"), env.render()):
                Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(args.output / f"{prefix}_{camera}.png")

        write(args.output / "ready.json", state())
        snapshots("reset")
        print("[GEN4 actions] Ready", flush=True)
        processed = set()
        last_command = time.monotonic()
        while time.monotonic() - last_command < 900:
            pending = [p for p in sorted(inbox.glob("*.json")) if p.name not in processed]
            if not pending:
                time.sleep(0.2)
                continue
            path = pending[0]
            command = json.loads(path.read_text())
            processed.add(path.name)
            last_command = time.monotonic()
            if command.get("quit"):
                result["completed"] = True
                break
            if "reset_seed" in command:
                env.reset(seed=int(command["reset_seed"]))
                hand_targets = {side: np.zeros(20) for side in ("left", "right")}
                terminal = False
            report = {"command": command, "phases": [], "exceptions": []}
            records = []
            for phase in command.get("phases", []):
                if terminal:
                    raise RuntimeError("A terminal diagnostic trajectory needs a normal reset before another action")
                steps = int(phase.get("steps", 60))
                if not 1 <= steps <= 600:
                    raise ValueError("A diagnostic phase must contain 1--600 control steps")
                before = state()
                endpoints = {}
                starts = {}
                for side, target in phase.get("targets", {}).items():
                    if side not in hand_targets:
                        raise ValueError(f"Unknown hand: {side}")
                    starts[side] = np.r_[before["robots"][side]["tcp"], hand_targets[side]]
                    endpoint = starts[side].copy()
                    if "tcp" in target:
                        endpoint[:7] = target["tcp"]
                    if "position" in target:
                        endpoint[:3] = target["position"]
                    if "glue_offset" in target:
                        endpoint[:3] = env._poses[c.GLUE_BODY][:3] + np.asarray(target["glue_offset"])
                    if "orientation" in target:
                        endpoint[3:7] = target["orientation"]
                    if "hand" in target:
                        endpoint[7:] = target["hand"]
                    endpoint[3:7] /= np.linalg.norm(endpoint[3:7])
                    if np.dot(endpoint[3:7], starts[side][3:7]) < 0:
                        endpoint[3:7] *= -1
                    endpoints[side] = endpoint
                contact_frames = {"left": 0, "right": 0}
                peak_stage = 0
                for frame in range(1, steps + 1):
                    alpha = frame / steps
                    action = {}
                    for side, endpoint in endpoints.items():
                        value = (1 - alpha) * starts[side] + alpha * endpoint
                        value[3:7] /= np.linalg.norm(value[3:7])
                        action[side] = value
                    observation, reward, terminated, truncated, info = env.step(action)
                    metrics = env._gen4_metrics()
                    for side in contact_frames:
                        contact_frames[side] += int(metrics[f"{side}_target_contact"])
                    peak_stage = max(peak_stage, int(metrics["handoff_stage"]))
                    records.append({"phase": phase.get("name"), "frame": frame,
                                    "action": action, "glue_pose": env._poses[c.GLUE_BODY].copy(),
                                    "hidden_state": observation["gen4_handoff_state"].copy(),
                                    "metrics": metrics, "succeed": bool(info.get("succeed")),
                                    "terminated": bool(terminated), "truncated": bool(truncated)})
                    terminal = bool(terminated or truncated)
                    if terminal:
                        break
                for side, endpoint in endpoints.items():
                    hand_targets[side] = endpoint[7:].copy()
                phase_report = {"name": phase.get("name"), "contact_frames": contact_frames,
                                "peak_stage": peak_stage, "state": state()}
                report["phases"].append(phase_report)
                print("[GEN4 actions] " + json.dumps({"name": phase.get("name"),
                      "contacts": contact_frames, "peak_stage": peak_stage,
                      "metrics": env._gen4_metrics()}), flush=True)
                if terminal:
                    break
            if command.get("reader_fault_check", False):
                report["reader_fault_checks"] = []

                class FaultContactView:
                    def __init__(self, payload):
                        self.payload = payload

                    def get_contact_force_matrix(self, dt):
                        if isinstance(self.payload, Exception):
                            raise self.payload
                        return self.payload

                faults = {
                    "exception": RuntimeError("GEN4 diagnostic transient reader failure"),
                    "empty": np.empty((1, 0, 3)),
                    "malformed": np.zeros((1, 3)),
                    "nonfinite": np.asarray([[[np.nan, 0.0, 0.0]]]),
                    "overflow": np.asarray([[[1.0e308, 0.0, 0.0]]]),
                }
                # Faults affect readout only; physics and public hold actions
                # remain unchanged. CPU tests cover all ownership transitions.
                for side in ("left", "right"):
                    key = (side, c.GLUE_BODY)
                    healthy_view = env._d_hand_contact_views[key]
                    for label, payload in faults.items():
                        initial_stage = env._gen4_handoff.stage
                        try:
                            env._d_hand_contact_views[key] = FaultContactView(payload)
                            assert env._d_contact_reading(side, c.GLUE_BODY) == (0.0, False)
                            for _ in range(2):
                                _, _, terminated, truncated, info = env.step({})
                                assert not terminated and not truncated
                                assert not info.get("manual_reset")
                                assert env._gen4_handoff.stage == initial_stage
                        finally:
                            env._d_hand_contact_views[key] = healthy_view
                        force, available = env._d_contact_reading(side, c.GLUE_BODY)
                        assert available
                        report["reader_fault_checks"].append({
                            "side": side, "fault": label, "stage_preserved": initial_stage,
                            "recovered_available": bool(available), "recovered_force_n": force,
                        })
            if command.get("query", False):
                report["shape_query"] = query_final_shapes(env, path.stem, allowed_hand_contacts=("left", "right"))
                write(args.output / f"{path.stem}_result.json", report)
                query = report["shape_query"]
                if (not query["callback_count"] or query["missing_self_hit_shapes"]
                        or query["query_errors"] or query["unapproved_pairs"]):
                    raise RuntimeError("GEN4 shape query requires a positive self-hit for each input, no query errors, and no unapproved pairs")
            if command.get("snapshots", True):
                snapshots(path.stem)
            report["state"] = state()
            write(args.output / f"{path.stem}_result.json", report)
            write(args.output / f"{path.stem}_actions.json", records)
            write(args.output / "ready.json", state())
            result["commands"].append(path.name)
            write(args.output / "actions_session.json", result)
            last_command = time.monotonic()
    except BaseException:
        result["exceptions"].append(traceback.format_exc())
        raise
    finally:
        write(args.output / "actions_session.json", result)
        if env is not None:
            env._fast_shutdown_on_close = True
            env.close()


if __name__ == "__main__":
    main()
