#!/usr/bin/env python3
"""Minimal Isaac Sim preview for the E01 cotton-garment cloth path.

The script intentionally keeps the scene small: a table, the GarmentLab E01
surface-deformable shirt, and two kinematic contact spheres that run a simple
pinch/lift trajectory.  With ``--with_arms`` it also imports the previous
ROKAE+Wuji E01 scene so the garment is shown in the real robot workspace.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import queue
import sys
import threading
import traceback
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
HANDSON_ROOT = REPO_ROOT / "handson"
if str(HANDSON_ROOT) not in sys.path:
    sys.path.insert(0, str(HANDSON_ROOT))

# Make the MVP start as a quiet tabletop garment unless the caller explicitly
# wants the asset-native hanging pose or the wrinkled rest pose.
os.environ.setdefault("HANDSON_E01_REST_SHAPE_MODE", "tabletop_flat")

from isaaclab.app import AppLauncher

from handson.sim.isaaclab.runtime_display import (
    enforce_isaac_viewport_only,
    isaac_kit_args,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Stop after N simulation steps. 0 keeps the interactive preview open.",
)
parser.add_argument(
    "--status_interval",
    type=int,
    default=120,
    help="Print a short status line every N steps. 0 disables periodic output.",
)
parser.add_argument(
    "--reset_at_step",
    type=int,
    default=-1,
    help="Testing hook: request one cloth reset after this many simulation steps.",
)
parser.add_argument(
    "--no_pinch",
    action="store_true",
    help="Keep the two contact spheres parked above the cloth. Deprecated; this is now the default.",
)
parser.add_argument(
    "--auto_pinch",
    action="store_true",
    help="Run the automatic pinch/lift trajectory. Default parks the spheres for a stable cloth preview.",
)
parser.add_argument(
    "--with_arms",
    action="store_true",
    help="Import the E01 ROKAE+Wuji scene so the cloth appears with the previous robot arms.",
)
parser.add_argument(
    "--no_manual_controls",
    action="store_true",
    help="Disable viewport keyboard control handles when --with_arms is active.",
)
parser.add_argument(
    "--manual_speed",
    type=float,
    default=0.22,
    help="Keyboard control handle speed in meters per second.",
)
parser.add_argument(
    "--sphere_radius",
    type=float,
    default=0.035,
    help="Radius in meters of each kinematic pinch sphere.",
)
parser.add_argument(
    "--save_usd",
    default="",
    help="Optional path to export the authored USD stage before shutdown.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not any(argument == "--device" or argument.startswith("--device=") for argument in sys.argv[1:]):
    args_cli.device = "cuda:0"
args_cli.kit_args = " ".join(
    filter(
        None,
        (
            str(args_cli.kit_args or ""),
            "--/persistent/physics/enableDeformableBeta=true",
        ),
    )
)
args_cli.kit_args = isaac_kit_args(args_cli.kit_args, headless=args_cli.headless)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
enforce_isaac_viewport_only(simulation_app, headless=args_cli.headless)

if args_cli.with_arms:
    from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

    enable_extension("isaacsim.asset.importer.mjcf")


import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
import omni.usd  # noqa: E402
from pxr import PhysxSchema, UsdGeom, UsdPhysics  # noqa: E402
import torch  # noqa: E402

from handson.sim.isaaclab.e_collection_scene import (  # noqa: E402
    E01_GARMENT_MATERIAL_PATH,
    E01_GARMENT_ROOT_PATH,
    E01_GARMENT_SIM_MESH_PATH,
    E01_TABLE_TOP_Z,
    selected_garmentlab_preset_metadata,
    spawn_e01_garmentlab_surface_deformable,
)
from handson.sim.isaaclab.physics_config import (  # noqa: E402
    SINGLE_ENV_GPU_MAX_RIGID_CONTACT_COUNT,
    SINGLE_ENV_GPU_MAX_RIGID_PATCH_COUNT,
)


PINCH_ROOT_PATH = "/World/CottonMvpPinch"
PINCH_PATHS = {
    "left": f"{PINCH_ROOT_PATH}/LeftSphere",
    "right": f"{PINCH_ROOT_PATH}/RightSphere",
}
SIM_DT = 1.0 / 120.0
TABLE_THICKNESS = 0.08


def _set_attr(prim: Any, name: str, value: Any) -> bool:
    attr = prim.GetAttribute(name)
    if not attr.IsValid():
        print(f"[cloth-mvp] missing attr {name} on {prim.GetPath()}", flush=True)
        return False
    attr.Set(value)
    return True


def _spawn_lights() -> None:
    dome_cfg = sim_utils.DomeLightCfg(intensity=560.0, color=(0.92, 0.94, 1.0))
    dome_cfg.func("/World/CottonMvpDomeLight", dome_cfg)
    key_cfg = sim_utils.SphereLightCfg(
        intensity=3600.0,
        color=(1.0, 0.96, 0.88),
        radius=0.18,
        treat_as_point=False,
    )
    key_cfg.func(
        "/World/CottonMvpKeyLight",
        key_cfg,
        translation=(-0.18, -0.62, 1.65),
    )


def _spawn_table() -> None:
    table_cfg = sim_utils.CuboidCfg(
        size=(1.60, 1.60, TABLE_THICKNESS),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
            max_depenetration_velocity=0.25,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.004,
            rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.60,
            dynamic_friction=1.25,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.56, 0.55, 0.51),
            roughness=0.92,
            metallic=0.0,
        ),
    )
    table_cfg.func(
        "/World/CottonMvpTable",
        table_cfg,
        translation=(0.50, 0.05, E01_TABLE_TOP_Z - 0.5 * TABLE_THICKNESS),
    )


def _spawn_contact_sphere(path: str, *, color: tuple[float, float, float], pos: tuple[float, float, float]) -> None:
    sphere_cfg = sim_utils.SphereCfg(
        radius=float(args_cli.sphere_radius),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=8,
            max_depenetration_velocity=0.35,
            enable_gyroscopic_forces=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.006,
            rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.45,
            dynamic_friction=1.12,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            roughness=0.50,
            metallic=0.0,
        ),
    )
    sphere_cfg.func(path, sphere_cfg, translation=pos)
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    physx_collision.CreateContactOffsetAttr().Set(0.006)
    physx_collision.CreateRestOffsetAttr().Set(0.0)
    physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rigid.CreateEnableSpeculativeCCDAttr().Set(True)
    physx_rigid.CreateMaxDepenetrationVelocityAttr().Set(0.35)


def _spawn_contact_spheres() -> None:
    UsdGeom.Xform.Define(omni.usd.get_context().get_stage(), PINCH_ROOT_PATH)
    park_z = E01_TABLE_TOP_Z + 0.34
    _spawn_contact_sphere(
        PINCH_PATHS["left"],
        color=(0.20, 0.72, 0.66),
        pos=(0.36, -0.10, park_z),
    )
    _spawn_contact_sphere(
        PINCH_PATHS["right"],
        color=(0.88, 0.66, 0.24),
        pos=(0.64, 0.18, park_z),
    )


def _configure_cotton_material(stage: Any) -> None:
    material_prim = stage.GetPrimAtPath(E01_GARMENT_MATERIAL_PATH)
    if not material_prim.IsValid():
        raise RuntimeError(f"Missing garment material prim: {E01_GARMENT_MATERIAL_PATH}")
    # A thicker, high-quality cotton jersey approximation.  These are PhysX
    # surface-deformable parameters, not render-only shader controls.
    overrides = {
        "omniphysics:density": 170.0,
        "omniphysics:staticFriction": 1.45,
        "omniphysics:dynamicFriction": 1.12,
        "omniphysics:surfaceThickness": 0.0012,
        "omniphysics:surfaceStretchStiffness": 1.20e4,
        "omniphysics:surfaceShearStiffness": 90.0,
        "omniphysics:surfaceBendStiffness": 0.55,
        "physxDeformableMaterial:elasticityDamping": 0.78,
        "physxDeformableMaterial:bendDamping": 0.72,
    }
    for name, value in overrides.items():
        _set_attr(material_prim, name, value)

    root_prim = stage.GetPrimAtPath(E01_GARMENT_ROOT_PATH)
    body_overrides = {
        "physxDeformableBody:selfCollision": True,
        "physxDeformableBody:enableSpeculativeCCD": True,
        "physxDeformableBody:solverPositionIterationCount": 32,
        "physxDeformableBody:collisionPairUpdateFrequency": 4,
        "physxDeformableBody:collisionIterationMultiplier": 4,
        "physxDeformableBody:maxLinearVelocity": 1.2,
        "physxDeformableBody:maxDepenetrationVelocity": 0.35,
    }
    for name, value in body_overrides.items():
        _set_attr(root_prim, name, value)

    sim_mesh_prim = stage.GetPrimAtPath(E01_GARMENT_SIM_MESH_PATH)
    if sim_mesh_prim.IsValid():
        sim_mesh_prim.ApplyAPI(PhysxSchema.PhysxCollisionAPI)
        collision_api = PhysxSchema.PhysxCollisionAPI(sim_mesh_prim)
        collision_api.GetRestOffsetAttr().Set(0.0010)
        collision_api.GetContactOffsetAttr().Set(0.0060)


def _configure_physx_scene(stage: Any) -> None:
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Scene):
            continue
        prim.ApplyAPI(PhysxSchema.PhysxSceneAPI)
        scene_api = PhysxSchema.PhysxSceneAPI(prim)
        scene_api.GetTimeStepsPerSecondAttr().Set(120)
        scene_api.GetGpuMaxDeformableSurfaceContactsAttr().Set(4 * 1_048_576)
        break


def _smoothstep(value: float) -> float:
    x = max(0.0, min(1.0, value))
    return x * x * (3.0 - 2.0 * x)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _pinch_pose(side: str, t: float) -> tuple[float, float, float]:
    radius = float(args_cli.sphere_radius)
    start = {
        "left": (0.34, -0.12, E01_TABLE_TOP_Z + 0.34),
        "right": (0.66, 0.20, E01_TABLE_TOP_Z + 0.34),
    }[side]
    hover = {
        "left": (0.405, -0.045, E01_TABLE_TOP_Z + 0.11),
        "right": (0.595, 0.135, E01_TABLE_TOP_Z + 0.11),
    }[side]
    contact = {
        "left": (0.455, 0.000, E01_TABLE_TOP_Z + radius + 0.018),
        "right": (0.545, 0.090, E01_TABLE_TOP_Z + radius + 0.018),
    }[side]
    lift = {
        "left": (0.420, -0.020, E01_TABLE_TOP_Z + 0.38),
        "right": (0.580, 0.120, E01_TABLE_TOP_Z + 0.38),
    }[side]
    release = {
        "left": (0.31, -0.19, E01_TABLE_TOP_Z + 0.42),
        "right": (0.69, 0.29, E01_TABLE_TOP_Z + 0.42),
    }[side]

    if args_cli.no_pinch or not args_cli.auto_pinch:
        return start
    if t < 1.0:
        return start
    if t < 2.2:
        u = _smoothstep((t - 1.0) / 1.2)
        return tuple(_lerp(start[i], hover[i], u) for i in range(3))
    if t < 3.4:
        u = _smoothstep((t - 2.2) / 1.2)
        return tuple(_lerp(hover[i], contact[i], u) for i in range(3))
    if t < 5.7:
        u = _smoothstep((t - 3.4) / 2.3)
        return tuple(_lerp(contact[i], lift[i], u) for i in range(3))
    if t < 7.2:
        return lift
    if t < 8.4:
        u = _smoothstep((t - 7.2) / 1.2)
        return tuple(_lerp(lift[i], release[i], u) for i in range(3))
    return release


class KinematicPinchController:
    """Writes PhysX kinematic transforms for the two pinch spheres."""

    def __init__(self, *, device: str, manual: bool = False, speed: float = 0.22):
        self._device = torch.device(device)
        self._manual = bool(manual)
        self._speed = max(0.02, float(speed))
        self._views: dict[str, Any] = {}
        self._indices: dict[str, Any] = {}
        self._initial_pose = {
            side: torch.tensor(_pinch_pose(side, 0.0), dtype=torch.float32, device=self._device)
            for side in PINCH_PATHS
        }
        self._manual_pose = {side: pose.clone() for side, pose in self._initial_pose.items()}
        self._last_pose = {side: pose.clone() for side, pose in self._initial_pose.items()}

    def bind(self) -> None:
        physics_sim_view = SimulationManager.get_physics_sim_view()
        for side, path in PINCH_PATHS.items():
            view = physics_sim_view.create_rigid_body_view(path)
            if int(view.count) < 1:
                raise RuntimeError(f"Could not bind kinematic pinch sphere: {path}")
            self._views[side] = view
            self._indices[side] = torch.arange(
                int(view.count),
                dtype=torch.long,
                device=self._device,
            )
        self.write(0.0, reset=True)

    def reset(self) -> None:
        self._manual_pose = {side: pose.clone() for side, pose in self._initial_pose.items()}
        self._last_pose = {side: pose.clone() for side, pose in self._initial_pose.items()}
        self.write(0.0, reset=True)

    def update_manual(self, controls: dict[str, Any], *, dt: float) -> None:
        if not self._manual:
            return
        pressed = set(controls.get("pressed") or ())
        active = str(controls.get("active") or "both")
        sides = ("left", "right") if active == "both" else (active,)
        direction = torch.zeros(3, dtype=torch.float32, device=self._device)
        if "w" in pressed or "up" in pressed:
            direction[0] += 1.0
        if "s" in pressed or "down" in pressed:
            direction[0] -= 1.0
        if "a" in pressed or "left_arrow" in pressed:
            direction[1] += 1.0
        if "d" in pressed or "right_arrow" in pressed:
            direction[1] -= 1.0
        if "e" in pressed:
            direction[2] += 1.0
        if "q" in pressed:
            direction[2] -= 1.0
        norm = torch.linalg.vector_norm(direction)
        speed = self._speed * (2.0 if {"shift", "left_shift", "right_shift"} & pressed else 1.0)
        if float(norm) > 1.0e-6:
            direction = direction / torch.clamp(norm, min=1.0)
            for side in sides:
                if side in self._manual_pose:
                    self._manual_pose[side] = self._manual_pose[side] + direction * speed * float(dt)

        pinch = 0.0
        if "z" in pressed:
            pinch += 1.0
        if "x" in pressed:
            pinch -= 1.0
        if pinch:
            center = 0.5 * (self._manual_pose["left"] + self._manual_pose["right"])
            for side in ("left", "right"):
                toward = center - self._manual_pose[side]
                toward[2] = 0.0
                toward_norm = torch.linalg.vector_norm(toward)
                if float(toward_norm) > 1.0e-6:
                    self._manual_pose[side] = (
                        self._manual_pose[side]
                        + pinch * toward / torch.clamp(toward_norm, min=1.0e-6) * speed * float(dt)
                    )

        radius = float(args_cli.sphere_radius)
        lower = torch.tensor(
            (0.24, -0.34, E01_TABLE_TOP_Z + radius + 0.010),
            dtype=torch.float32,
            device=self._device,
        )
        upper = torch.tensor(
            (0.78, 0.38, E01_TABLE_TOP_Z + 0.58),
            dtype=torch.float32,
            device=self._device,
        )
        for side in self._manual_pose:
            self._manual_pose[side] = torch.minimum(torch.maximum(self._manual_pose[side], lower), upper)

    def write(self, t: float, *, reset: bool = False) -> None:
        for side, view in self._views.items():
            indices = self._indices[side]
            if self._manual:
                pose = self._manual_pose[side]
            else:
                pose = torch.tensor(
                    _pinch_pose(side, t),
                    dtype=torch.float32,
                    device=self._device,
                )
            payload = torch.zeros((1, 7), dtype=torch.float32, device=self._device)
            payload[0, :3] = pose
            payload[0, 6] = 1.0
            view.set_transforms(payload, indices)

            velocities = torch.zeros_like(torch.as_tensor(view.get_velocities(), device=self._device))
            if not reset:
                velocities[:, :3] = (pose - self._last_pose[side]) / SIM_DT
            view.set_velocities(velocities, indices)
            self._last_pose[side] = pose


class GarmentResetController:
    """Caches and restores the E01 surface-deformable nodal state."""

    def __init__(self, *, device: str):
        self._device = torch.device(device)
        self._sim_view: Any | None = None
        self._view: Any | None = None
        self._indices: Any | None = None
        self._initial_positions: torch.Tensor | None = None
        self._initial_velocities: torch.Tensor | None = None

    def bind(self, *, capture_initial: bool) -> None:
        import omni.physics.tensors as physics_tensors

        self._sim_view = physics_tensors.create_simulation_view("torch")
        self._sim_view.set_subspace_roots("/")
        view = self._sim_view.create_surface_deformable_body_view(E01_GARMENT_ROOT_PATH)
        view_count = int(view.count)
        view_valid = bool(view.check()) if view_count else False
        if view_count != 1:
            raise RuntimeError(
                "Could not bind E01 garment surface-deformable view: "
                f"count={view_count}, check={view_valid}, root={E01_GARMENT_ROOT_PATH}"
            )
        positions = view.get_simulation_nodal_positions()
        velocities = view.get_simulation_nodal_velocities()
        if not hasattr(positions, "detach") or not hasattr(velocities, "detach"):
            raise RuntimeError("Expected torch-backed PhysX surface tensors for garment reset")
        self._view = view
        self._indices = torch.arange(view_count, dtype=torch.long, device=positions.device)
        if capture_initial:
            self._initial_positions = positions.detach().clone()
            self._initial_velocities = torch.zeros_like(velocities)
            print(
                "[cloth-mvp] cached reset cloth nodes: "
                f"{int(self._initial_positions.shape[1])} nodes",
                flush=True,
            )

    def restore(self) -> None:
        if self._initial_positions is None or self._initial_velocities is None:
            raise RuntimeError("Garment reset was requested before initial nodes were cached")
        if self._view is None or self._indices is None:
            self.bind(capture_initial=False)
        try:
            self._view.set_simulation_nodal_positions(
                self._initial_positions.clone(),
                self._indices,
            )
            self._view.set_simulation_nodal_velocities(
                self._initial_velocities.clone(),
                self._indices,
            )
        except Exception:
            self.bind(capture_initial=False)
            self._view.set_simulation_nodal_positions(
                self._initial_positions.clone(),
                self._indices,
            )
            self._view.set_simulation_nodal_velocities(
                self._initial_velocities.clone(),
                self._indices,
            )


class ResetRequests:
    """Collects viewport reset requests and manual handle key state."""

    def __init__(self, *, headless: bool):
        self._headless = bool(headless)
        self._queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._pressed: set[str] = set()
        self._active = "both"
        self._keyboard = None
        self._keyboard_input_interface = None
        self._keyboard_subscription = None

    def install(self) -> None:
        self._install_keyboard()
        self._install_terminal_reader()

    def close(self) -> None:
        if self._keyboard_subscription is not None and self._keyboard_input_interface is not None:
            try:
                self._keyboard_input_interface.unsubscribe_to_keyboard_events(
                    self._keyboard,
                    self._keyboard_subscription,
                )
            except Exception:
                pass
        self._keyboard = None
        self._keyboard_input_interface = None
        self._keyboard_subscription = None

    def consume(self) -> str | None:
        source = None
        while True:
            try:
                source = self._queue.get_nowait()
            except queue.Empty:
                return source

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"pressed": frozenset(self._pressed), "active": self._active}

    def _request(self, source: str) -> None:
        self._queue.put(source)

    def _set_active(self, active: str) -> None:
        if active not in {"left", "right", "both"}:
            return
        with self._lock:
            changed = active != self._active
            self._active = active
        if changed:
            print(f"[cloth-mvp] manual control target: {active}", flush=True)

    def _install_keyboard(self) -> None:
        if self._headless:
            return
        try:
            import carb.input
            import omni.appwindow

            app_window = omni.appwindow.get_default_app_window()
            if app_window is None:
                raise RuntimeError("default app window is unavailable")
            keyboard = app_window.get_keyboard()
            input_interface = carb.input.acquire_input_interface()
            self._keyboard = keyboard
            self._keyboard_input_interface = input_interface
            self._keyboard_subscription = input_interface.subscribe_to_keyboard_events(
                keyboard,
                lambda event, *args, **kwargs: self._on_keyboard_event(carb.input, event),
            )
        except Exception as exc:
            print(f"[cloth-mvp] viewport keyboard reset unavailable: {exc}", flush=True)

    def _install_terminal_reader(self) -> None:
        if sys.stdin is None or sys.stdin.closed or not sys.stdin.isatty():
            return

        def read_loop() -> None:
            while True:
                line = sys.stdin.readline()
                if not line:
                    return
                command = line.strip().lower()
                if command in {"r", "reset"}:
                    self._request("terminal")
                elif command in {"left", "right", "both"}:
                    self._set_active(command)

        thread = threading.Thread(target=read_loop, daemon=True)
        thread.start()

    def _on_keyboard_event(self, carb_input, event) -> bool:
        event_type = getattr(event, "type", None)
        key_input = getattr(event, "input", None)
        press_type = getattr(carb_input.KeyboardEventType, "KEY_PRESS", None)
        release_type = getattr(carb_input.KeyboardEventType, "KEY_RELEASE", None)
        event_type_name = str(event_type).lower()
        is_press = event_type == press_type or ("press" in event_type_name and "release" not in event_type_name)
        is_release = event_type == release_type or "release" in event_type_name
        key_name = self._key_token(key_input)
        is_r_key = key_input == getattr(carb_input.KeyboardInput, "R", None) or key_name == "r"
        if is_press and is_r_key:
            self._request("viewport")
            return True
        if is_press and key_name in {"1", "num1", "numpad1"}:
            self._set_active("left")
            return True
        if is_press and key_name in {"2", "num2", "numpad2"}:
            self._set_active("right")
            return True
        if is_press and key_name in {"3", "num3", "numpad3"}:
            self._set_active("both")
            return True
        movement_key = self._movement_key(key_name)
        if movement_key:
            with self._lock:
                if is_release:
                    self._pressed.discard(movement_key)
                elif is_press:
                    self._pressed.add(movement_key)
        return True

    @staticmethod
    def _key_token(key_input: Any) -> str:
        key_name = str(key_input).lower()
        key_name = key_name.rsplit(".", 1)[-1]
        for prefix in ("keyboardinput.", "key_", "keyboard_", "num_"):
            if key_name.startswith(prefix):
                key_name = key_name[len(prefix) :]
        key_name = key_name.replace("arrow_", "")
        key_name = key_name.replace("numpad_", "numpad")
        if key_name == "left":
            return "left_arrow"
        if key_name == "right":
            return "right_arrow"
        return key_name

    @staticmethod
    def _movement_key(key_name: str) -> str | None:
        aliases = {
            "w": "w",
            "s": "s",
            "a": "a",
            "d": "d",
            "q": "q",
            "e": "e",
            "z": "z",
            "x": "x",
            "up": "up",
            "down": "down",
            "left_arrow": "left_arrow",
            "right_arrow": "right_arrow",
            "shift": "shift",
            "left_shift": "left_shift",
            "right_shift": "right_shift",
        }
        return aliases.get(key_name)


def _count_surface_deformables(stage: Any) -> int:
    count = 0
    for prim in stage.Traverse():
        if prim.GetAttribute("physxDeformableBody:selfCollision").IsValid():
            count += 1
    return count


def _count_task_articulations(stage: Any) -> int:
    count = 0
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith("/World/Task") and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            count += 1
    return count


def main() -> int:
    metadata = selected_garmentlab_preset_metadata()
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=SIM_DT,
        render_interval=2,
        gravity=(0.0, 0.0, -10.0),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            enable_ccd=True,
            max_position_iteration_count=64,
            max_velocity_iteration_count=8,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=SINGLE_ENV_GPU_MAX_RIGID_CONTACT_COUNT,
            gpu_max_rigid_patch_count=SINGLE_ENV_GPU_MAX_RIGID_PATCH_COUNT,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.45,
            dynamic_friction=1.12,
        ),
    )
    sim = SimulationContext(sim_cfg)
    if args_cli.with_arms:
        from handson.sim.isaaclab.task_preview import validate_task_assets

        spec = validate_task_assets("e01_fold_garment")
        sim.set_camera_view(eye=spec.camera_eye, target=spec.camera_target)
    else:
        sim.set_camera_view(eye=(-0.22, -0.92, 1.42), target=(0.50, 0.05, 0.94))

    stage = omni.usd.get_context().get_stage()
    _spawn_lights()
    task_mjcf = None
    if args_cli.with_arms:
        from handson.sim.isaaclab.e_collection_scene import E01_TASK_NAME
        from handson.sim.isaaclab.task_preview import spawn_task_scene

        task_mjcf, (rebound, hidden) = spawn_task_scene(E01_TASK_NAME)
        print(
            "[cloth-mvp] imported E01 ROKAE+Wuji scene: "
            f"cached_mjcf={task_mjcf}, visual_repaired={rebound}, hidden_sites={hidden}",
            flush=True,
        )
    else:
        _spawn_table()
    manual_controls = bool(args_cli.with_arms and not args_cli.no_manual_controls and not args_cli.auto_pinch)
    pinch_enabled = (
        (not args_cli.with_arms)
        or manual_controls
        or (args_cli.auto_pinch and not args_cli.no_pinch)
    )
    if pinch_enabled:
        _spawn_contact_spheres()
    garment_asset = spawn_e01_garmentlab_surface_deformable(stage)
    _configure_cotton_material(stage)
    _configure_physx_scene(stage)
    sim_utils.update_stage()

    sim.reset()
    controller = None
    if pinch_enabled:
        controller = KinematicPinchController(
            device=args_cli.device,
            manual=manual_controls,
            speed=args_cli.manual_speed,
        )
        controller.bind()
    garment_reset = GarmentResetController(device=args_cli.device)
    garment_reset.bind(capture_initial=True)
    reset_requests = ResetRequests(headless=args_cli.headless)
    reset_requests.install()

    surface_count = _count_surface_deformables(stage)
    task_articulations = _count_task_articulations(stage)
    print(
        "[cloth-mvp] ready: "
        f"garment={metadata['label']} "
        f"{metadata['sim_vertex_count']}v/{metadata['sim_triangle_count']}t, "
        f"asset={garment_asset}, surface_deformables={surface_count}, "
        f"task_articulations={task_articulations}, "
        "gravity=-10.0m/s^2, cotton_thickness=1.2mm, "
        f"with_arms={'on' if args_cli.with_arms else 'off'}, "
        f"manual_controls={'on' if manual_controls else 'off'}, "
        f"auto_pinch={'on' if pinch_enabled and args_cli.auto_pinch and not args_cli.no_pinch else 'off'}",
        flush=True,
    )
    if manual_controls:
        print(
            "[cloth-mvp] controls: click the viewport; 1=left, 2=right, 3=both, "
            "W/S and A/D move on table, Q/E height, Z/X pinch, R resets cloth",
            flush=True,
        )
    else:
        print("[cloth-mvp] controls: press R in the Isaac window, or type r + Enter in this terminal, to reset", flush=True)

    step = 0
    forced_reset_done = False
    try:
        while simulation_app.is_running() and (args_cli.max_steps <= 0 or step < args_cli.max_steps):
            reset_source = reset_requests.consume()
            if (
                reset_source is None
                and not forced_reset_done
                and args_cli.reset_at_step >= 0
                and step >= args_cli.reset_at_step
            ):
                reset_source = "script"
                forced_reset_done = True
            if reset_source is not None:
                print(f"[cloth-mvp] reset requested from {reset_source}", flush=True)
                if not args_cli.with_arms:
                    sim.reset()
                garment_reset.restore()
                if controller is not None:
                    try:
                        controller.reset()
                    except Exception:
                        controller.bind()
                step = 0
                continue

            t = step * SIM_DT
            if controller is not None:
                if manual_controls:
                    controller.update_manual(reset_requests.snapshot(), dt=SIM_DT)
                controller.write(t)
            sim.step()
            step += 1
            if args_cli.status_interval > 0 and step % args_cli.status_interval == 0:
                if controller is None:
                    phase = "robot_scene"
                elif manual_controls:
                    phase = f"manual_{reset_requests.snapshot()['active']}"
                else:
                    parked = args_cli.no_pinch or not args_cli.auto_pinch
                    phase = "parked" if parked else f"pinch_t={t:.2f}s"
                print(f"[cloth-mvp step {step:06d}] {phase}", flush=True)
    finally:
        reset_requests.close()

    if args_cli.save_usd:
        save_path = Path(args_cli.save_usd).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        stage.GetRootLayer().Export(str(save_path))
        print(f"[cloth-mvp] exported stage: {save_path}", flush=True)

    print(f"[cloth-mvp] finished after {step} steps", flush=True)
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
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
