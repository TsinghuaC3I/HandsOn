#!/usr/bin/env python3
"""Preview HandsOn's ROKAE + Wuji assembly task in native Isaac Lab."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
from pathlib import Path

# Make the source checkout importable without installing HandsOn (and without
# pulling the MuJoCo dependency set into the dedicated Isaac Sim environment).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "handson"))

from isaaclab.app import AppLauncher

from handson.sim.isaaclab.physics_config import gpu_rigid_buffer_capacities
from handson.sim.isaaclab.recording_cameras import (
    OPERATOR_VIEWPORT_CAMERA_EYE,
    OPERATOR_VIEWPORT_CAMERA_TARGET,
)
from handson.sim.isaaclab.runtime_display import enforce_isaac_viewport_only, isaac_kit_args


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1, help="Number of tiled scene instances.")
parser.add_argument(
    "--hand_type",
    choices=("wuji1", "wuji2"),
    default="wuji1",
    help="Wuji articulation to use for both ROKAE arms.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Stop after this many physics steps; 0 keeps the interactive preview open.",
)
parser.add_argument(
    "--status_interval",
    type=int,
    default=240,
    help="Print physical state every N steps; 0 disables it.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Isaac Sim's bundled PyTorch build on this workstation does not contain the
# RTX 5090 (sm_120) kernels.  A one-scene visual preview is fast enough on CPU;
# callers with a CUDA 12.8 PyTorch build can opt in via ``--device cuda:0``.
if not any(argument == "--device" or argument.startswith("--device=") for argument in sys.argv[1:]):
    args_cli.device = "cuda:0"
args_cli.kit_args = isaac_kit_args(args_cli.kit_args, headless=args_cli.headless)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
enforce_isaac_viewport_only(simulation_app, headless=args_cli.headless)


import torch  # noqa: E402

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

# The minimal headless Isaac Lab experience does not preload the MJCF importer.
# Enable it before any MjcfFileCfg is instantiated.
enable_extension("isaacsim.asset.importer.mjcf")

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from handson.sim.isaaclab.bimanual_assembly_scene import (  # noqa: E402
    apply_scene_visuals,
    fix_robot_bases,
    make_bimanual_assembly_scene_cfg,
    validate_asset_files,
)


def reset_scene(scene: InteractiveScene) -> None:
    """Restore authored physical state and start joint-position holding."""
    for name in ("right_robot", "left_robot"):
        robot = scene[name]
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)

    table = scene["table"]
    table_state = table.data.default_root_state.clone()
    table_state[:, :3] += scene.env_origins
    table.write_root_pose_to_sim(table_state[:, :7])

    for name in ("peg", "socket"):
        rigid_object = scene[name]
        root_state = rigid_object.data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        rigid_object.write_root_pose_to_sim(root_state[:, :7])
        rigid_object.write_root_velocity_to_sim(root_state[:, 7:])

    scene.reset()
    scene.write_data_to_sim()


def print_scene_summary(scene: InteractiveScene) -> None:
    print("[HandsOn/IsaacLab] Native bimanual assembly preview ready")
    for name in ("right_robot", "left_robot"):
        robot = scene[name]
        print(
            f"  {name}: {robot.num_joints} joints, {robot.num_bodies} bodies, "
            f"device={robot.device}"
        )
        print(f"    joints: {', '.join(robot.joint_names)}")
    print("  dynamic parts: peg, socket; kinematic fixture: table")


def print_physics_status(scene: InteractiveScene, step: int) -> None:
    peg = scene["peg"].data.root_pos_w[0]
    socket = scene["socket"].data.root_pos_w[0]
    separation = torch.linalg.vector_norm(peg - socket).item()
    finite = all(
        torch.isfinite(scene[name].data.root_pos_w).all().item()
        for name in ("right_robot", "left_robot", "peg", "socket")
    )
    print(
        f"[step {step:06d}] peg={tuple(round(v, 4) for v in peg.tolist())} "
        f"socket={tuple(round(v, 4) for v in socket.tolist())} "
        f"root_distance={separation:.4f}m finite={finite}"
    )


def main() -> int:
    validate_asset_files()
    rigid_contact_count, rigid_patch_count = gpu_rigid_buffer_capacities(
        args_cli.num_envs
    )

    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=1.0 / 120.0,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            max_position_iteration_count=64,
            max_velocity_iteration_count=4,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=rigid_contact_count,
            gpu_max_rigid_patch_count=rigid_patch_count,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(
        eye=OPERATOR_VIEWPORT_CAMERA_EYE,
        target=OPERATOR_VIEWPORT_CAMERA_TARGET,
    )

    print("[HandsOn/IsaacLab] Constructing scene...", flush=True)
    scene = InteractiveScene(
        make_bimanual_assembly_scene_cfg(
            num_envs=args_cli.num_envs,
            env_spacing=2.5,
            hand_type=args_cli.hand_type,
        )
    )
    apply_scene_visuals(args_cli.num_envs)
    fix_robot_bases(args_cli.num_envs)
    print("[HandsOn/IsaacLab] Starting PhysX...", flush=True)
    sim.reset()
    print("[HandsOn/IsaacLab] Resetting task state...", flush=True)
    reset_scene(scene)
    print_scene_summary(scene)

    step = 0
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running() and (args_cli.max_steps <= 0 or step < args_cli.max_steps):
        # The preview uses a genuine position-hold command on every articulation
        # DOF.  Nothing is teleported after reset.
        for name in ("right_robot", "left_robot"):
            robot = scene[name]
            robot.set_joint_position_target(robot.data.default_joint_pos)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step += 1

        if args_cli.status_interval > 0 and step % args_cli.status_interval == 0:
            print_physics_status(scene, step)

    if args_cli.status_interval <= 0 or step % args_cli.status_interval != 0:
        print_physics_status(scene, step)
    stable = True
    for name in ("right_robot", "left_robot", "peg", "socket"):
        stable = stable and bool(torch.isfinite(scene[name].data.root_state_w).all().item())
    if scene["peg"].data.root_pos_w[:, 2].min().item() < 0.85:
        stable = False
        print("[ERROR] Peg fell below the task work surface.")
    if scene["socket"].data.root_pos_w[:, 2].min().item() < 0.85:
        stable = False
        print("[ERROR] Socket fell below the task work surface.")

    print(f"[HandsOn/IsaacLab] Preview {'PASSED' if stable else 'FAILED'} after {step} steps")
    return 0 if stable else 1


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        # Isaac Sim 5.1 can stall in framework teardown after MJCF conversion.
        # Bound cleanup so finite headless smoke tests never leave Kit workers.
        sys.stdout.flush()
        sys.stderr.flush()
        shutdown_guard = threading.Timer(5.0, lambda: os._exit(exit_code))
        shutdown_guard.daemon = True
        shutdown_guard.start()
        try:
            simulation_app.close(wait_for_replicator=False)
        finally:
            shutdown_guard.cancel()
    raise SystemExit(exit_code)
