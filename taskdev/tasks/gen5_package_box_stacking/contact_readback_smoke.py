"""Native sensor regression and bounded public-TCP touch; no task pose writes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "handson")]


def main():
    from isaaclab.app import AppLauncher
    from handson.sim.isaaclab.runtime_display import isaac_kit_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--audit-suite", default="contact")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless, args.enable_cameras = True, False
    args.kit_args = isaac_kit_args(args.kit_args, headless=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "scope": "Native GEN5 contact readback regression and bounded public-TCP touch; not a successful demonstration or replay proof",
        "task_body_state_writes_after_reset": 0,
        "actions": [], "faults": [], "checks": {}, "passed": False,
        "runtime": {"python": sys.executable, "isaacsim": importlib.metadata.version("isaacsim")},
        "source_sha256": hashlib.sha256((ROOT / "handson/handson/sim/isaaclab/abc_collection_env.py").read_bytes()).hexdigest(),
    }
    assert result["runtime"]["isaacsim"].startswith("5.1.")
    launcher = env = None
    try:
        launcher = AppLauncher(args)
        import numpy as np
        import torch
        import isaaclab
        from handson.sim.isaaclab import eval_gen07_stack_boxes as gen5
        from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv

        result["runtime"].update(isaaclab_file=isaaclab.__file__, isaaclab_version=getattr(isaaclab, "__version__", None), cuda_count=torch.cuda.device_count(), cuda_name=torch.cuda.get_device_name(0))
        print("[GEN5 contact] Runtime:", result["runtime"], flush=True)
        env = HandsOnABCIsaacLabEnv(gen5.TASK_NAME, seed=args.seed, control_dt=1/30,
            physics_dt=1/120, render_mode="rgb_array", image_obs=False,
            randomize=True, randomize_dynamics=False, hand_type="wuji1", arm_type="rokae",
            launch_isaac=True, headless=True, isaac_device=args.device, physics_backend="physx",
            existing_simulation_app=launcher.app, existing_app_launcher=launcher)
        env.reset(seed=args.seed)
        home = env._right_tcp_pose.copy()
        hand = env._right_hand.copy()
        command = home.copy()

        def array(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value)

        def read_frame(phase, action=None):
            observation, reward, terminated, truncated, info = env.step(action)
            force, available = env._b04_right_hand_contact_force()
            row = {"phase": phase, "action": None if action is None else {k: list(map(float, v)) for k,v in action.items()},
                "contact_force_n": force, "contact_available": available,
                "hand_released": info.get("stack_hand_released"), "hand_release_source": info.get("stack_hand_release_source"),
                "success_counter": int(env._success_counter), "succeed": bool(info.get("succeed")),
                "terminated": bool(terminated), "truncated": bool(truncated),
                "poses": {n: env._physics_body_pose(n).tolist() for n in gen5.BODY_NAMES},
                "velocities": {n: env._physics_body_velocity(n).tolist() for n in gen5.BODY_NAMES},
                "robot_joint_speed": {s: float(np.max(np.abs(array(r.data.joint_vel)))) for s,r in env._robots.items()},
                "robot_target_error": {s: float(np.max(np.abs(array(r.data.joint_pos)-array(r.data.joint_pos_target)))) for s,r in env._robots.items()},
            }
            result["actions"].append(row)
            assert not terminated and not truncated and not info.get("succeed"), row
            assert all(np.all(np.isfinite(row[k][n])) for k in ("poses", "velocities") for n in gen5.BODY_NAMES), row
            return row

        for _ in range(args.steps):
            read_frame("initial_hold")
        result["native_views"] = {n: {"sensor_count": int(v.sensor_count), "filter_count": int(v.filter_count),
            "matrix_shape": list(array(v.get_contact_force_matrix(env._sim_dt)).shape)} for n,v in env._b04_block_contact_views.items()}
        print("[GEN5 contact] Native views:", result["native_views"], flush=True)
        result["initial_frame"] = result["actions"][-1]

        class FaultView:
            def __init__(self, kind): self.kind = kind
            def get_contact_force_matrix(self, dt):
                if self.kind == "exception": raise RuntimeError("deliberate diagnostic contact readback fault")
                return np.empty((1, 0, 3)) if self.kind == "empty" else np.full((1, 1, 3), np.nan)

        for name in gen5.BODY_NAMES:
            for kind in ("exception", "empty", "nan"):
                original = env._b04_block_contact_views[name]
                try:
                    env._b04_block_contact_views[name] = FaultView(kind)
                    row = read_frame("fault_" + name + "_" + kind)
                    failed_closed = not row["contact_available"] and row["hand_release_source"] == "contact_unavailable" and not row["hand_released"] and row["success_counter"] == 0
                finally:
                    env._b04_block_contact_views[name] = original
                recovered = read_frame("recovery_" + name + "_" + kind)
                result["faults"].append({"body": name, "fault": kind, "failed_closed": failed_closed, "recovered": recovered["contact_available"]})
                assert failed_closed and recovered["contact_available"], result["faults"][-1]

        base_start = env._physics_body_pose(gen5.BASE_NAME).copy()
        tips = env._wuji_fingertip_positions("right")
        anchor = next((n for n in tips if "index" in n.lower()), next(iter(tips)))
        result["touch_target"] = {"body": gen5.BASE_NAME, "anchor": anchor, "initial_tips": {n: p.tolist() for n,p in tips.items()}}
        base_top = base_start[2] + gen5.half_extents()[gen5.BASE_NAME][2]

        def move(delta, phase, speed=0.0015):
            nonlocal command
            delta = np.asarray(delta, dtype=float)
            count = max(1, int(np.ceil(np.linalg.norm(delta)/speed)))
            start = command.copy()
            row = None
            for i in range(1, count+1):
                command = start.copy()
                command[:3] += delta*i/count
                row = read_frame(phase, {"right": np.r_[command, hand]})
                if phase.startswith("touch"):
                    moved = np.linalg.norm(np.asarray(row["poses"][gen5.BASE_NAME])[:3]-base_start[:3])
                    if row["contact_force_n"] >= 2.0 or moved > 0.010:
                        raise RuntimeError("Bounded touch force/displacement guard stopped the probe")
            return row

        # Translate an actual fingertip above the base before the downward approach.
        move([0, 0, 0.06], "lift")
        tip = env._wuji_fingertip_positions("right")[anchor]
        move([base_start[0]-tip[0], base_start[1]-tip[1], 0], "over_base")
        for _ in range(20): read_frame("over_base_hold", {"right": np.r_[command, hand]})
        tip = env._wuji_fingertip_positions("right")[anchor]
        descent = max(-0.22, min(0.0, base_top+0.012-tip[2]))
        move([0, 0, descent], "touch_approach", speed=0.001)
        contact_row = None
        for _ in range(48):
            row = move([0,0,-0.0005], "touch_seek", speed=0.0005)
            if row["contact_available"] and row["contact_force_n"] > 0.005:
                contact_row = row
                break
        hold = []
        if contact_row is not None:
            for _ in range(30):
                row = read_frame("touch_hold", {"right": np.r_[command, hand]})
                hold.append(row)
                if row["contact_force_n"] >= 2.0:
                    raise RuntimeError("Bounded contact hold force guard stopped the probe")
        result["touch"] = {"contact_observed": contact_row is not None, "hold_frames": len(hold),
            "hold_contact_frames": sum(row["contact_force_n"] > 0.001 for row in hold),
            "max_force_n": max((row["contact_force_n"] for row in result["actions"]), default=0.0)}
        move([0,0,0.10], "withdraw")
        move(home[:3]-command[:3], "return_home")
        for _ in range(30): read_frame("final_hold", {"right": np.r_[command, hand]})
        result["checks"] = {
            "four_real_valid_contact_views": set(result["native_views"]) == set(gen5.BODY_NAMES) and all(v["matrix_shape"] == [1,v["filter_count"],3] for v in result["native_views"].values()),
            "all_faults_fail_closed_and_recover": all(r["failed_closed"] and r["recovered"] for r in result["faults"]),
            "healthy_native_readbacks_stay_available": all(r["contact_available"] for r in result["actions"] if not r["phase"].startswith("fault_")),
            "no_unexpected_terminal_or_success": all(not r["terminated"] and not r["truncated"] and not r["succeed"] for r in result["actions"]),
            "deferred_cameras": env._deferred_camera_render and not env._cameras,
            "single_gpu": torch.cuda.device_count() == 1,
        }
        result["passed"] = all(result["checks"].values())
        print("[GEN5 contact] Checks:", result["checks"], "touch:", result["touch"], flush=True)
    except Exception:
        result["exception"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("[GEN5 contact] Result saved; closing native child.", flush=True)
        if env is not None:
            env._fast_shutdown_on_close = True
            env.close()
        if launcher is not None:
            launcher.app.close(wait_for_replicator=False, skip_cleanup=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
