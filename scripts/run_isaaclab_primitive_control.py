#!/usr/bin/env python3
"""Control A01, B03, or D08 with keyboard and local primitive HTTP APIs."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np


os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "handson"))

from handson.runtime.primitive_control import (  # noqa: E402
    AdaptiveGraspExecutor,
    HandPrimitiveController,
    PrimitiveCommandBroker,
    PrimitiveCommandError,
    PrimitiveHTTPService,
    apply_primitive_command,
    available_grasp_presets,
)
from handson.sim.envs.hand_specs import (  # noqa: E402
    SIMPLIFIED_POST_IK_ARM_TYPE,
)


SUPPORTED_TASKS = (
    "t26_place_fruit_in_bucket",
    "t09_insert_peg",
    "d08_toast_bread",
)
_CONTINUOUS_KEYS = frozenset("wasdqeuojlik")


class KitKeyboardPrimitiveInput:
    """Collect Kit keyboard events without mutating PhysX from callbacks."""

    def __init__(self, broker: PrimitiveCommandBroker, *, enabled: bool) -> None:
        self.broker = broker
        self.enabled = bool(enabled)
        self.pressed: set[str] = set()
        self._input_interface = None
        self._keyboard = None
        self._subscription = None
        if self.enabled:
            self._subscribe()

    @staticmethod
    def _key_name(value: Any) -> str:
        name = str(value).strip().lower()
        if "." in name:
            name = name.rsplit(".", 1)[-1]
        aliases = {
            "escape": "esc",
            "esc": "esc",
            **{f"key_{value}": value for value in "0123456789"},
        }
        return aliases.get(name, name)

    def _subscribe(self) -> None:
        import carb.input
        import omni.appwindow

        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            raise RuntimeError("default Isaac app window is unavailable")
        self._keyboard = app_window.get_keyboard()
        self._input_interface = carb.input.acquire_input_interface()
        self._subscription = self._input_interface.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, **kwargs: self._on_event(carb.input, event),
        )

    def _on_event(self, carb_input: Any, event: Any) -> bool:
        key = self._key_name(getattr(event, "input", ""))
        event_type = getattr(event, "type", None)
        type_name = str(event_type).lower()
        press_type = getattr(carb_input.KeyboardEventType, "KEY_PRESS", None)
        repeat_type = getattr(carb_input.KeyboardEventType, "KEY_REPEAT", None)
        release_type = getattr(carb_input.KeyboardEventType, "KEY_RELEASE", None)
        is_press = event_type == press_type or (
            "press" in type_name and "release" not in type_name
        )
        is_repeat = event_type == repeat_type or "repeat" in type_name
        is_release = event_type == release_type or "release" in type_name

        if key in _CONTINUOUS_KEYS:
            if is_press or is_repeat:
                self.pressed.add(key)
            elif is_release:
                self.pressed.discard(key)
            return False

        if not is_press:
            return True
        operations = {
            "g": ("grasp", {"type": "power"}),
            "p": ("grasp", {"type": "pinch"}),
            "c": ("grasp", {"type": "cylindrical"}),
            "b": ("grasp", {"type": "spherical"}),
            "v": ("grasp", {"type": "tripod"}),
            "x": ("grasp", {"type": "lateral"}),
            "n": ("grasp", {"type": "hook"}),
            "m": ("grasp_adaptive", {"strategy": "auto"}),
            "0": ("grasp", {"type": "relax"}),
            "h": ("grasp", {"type": "open"}),
            "t": ("toggle_hand", {}),
            "r": ("reset", {}),
            "esc": ("quit", {}),
        }
        command = operations.get(key)
        if command is None:
            return True
        self.broker.submit(*command)
        return False

    def clear(self) -> None:
        self.pressed.clear()

    def apply_continuous(
        self,
        controller: HandPrimitiveController,
        *,
        dt: float,
        translation_speed_m_s: float,
        rotation_speed_deg_s: float,
    ) -> bool:
        changed = False
        linear = np.asarray(
            [
                float("w" in self.pressed) - float("s" in self.pressed),
                float("a" in self.pressed) - float("d" in self.pressed),
                float("e" in self.pressed) - float("q" in self.pressed),
            ],
            dtype=np.float64,
        )
        linear_norm = float(np.linalg.norm(linear))
        if linear_norm > 0.0:
            controller.move(
                linear / linear_norm * float(translation_speed_m_s) * float(dt),
                frame="world",
            )
            changed = True

        rpy = np.asarray(
            [
                float("u" in self.pressed) - float("o" in self.pressed),
                float("i" in self.pressed) - float("k" in self.pressed),
                float("j" in self.pressed) - float("l" in self.pressed),
            ],
            dtype=np.float64,
        )
        rotation_norm = float(np.linalg.norm(rpy))
        if rotation_norm > 0.0:
            controller.rotate_rpy(
                rpy / rotation_norm * float(rotation_speed_deg_s) * float(dt),
                frame="local",
            )
            changed = True
        return changed

    def close(self) -> None:
        if self._subscription is not None and self._input_interface is not None:
            try:
                self._input_interface.unsubscribe_to_keyboard_events(
                    self._keyboard, self._subscription
                )
            except Exception:
                pass
        self._subscription = None
        self._input_interface = None
        self._keyboard = None


def _make_environment(args: argparse.Namespace):
    common = {
        "seed": args.seed,
        "render_mode": "rgb_array" if args.headless else "human",
        "image_obs": False,
        "randomize": args.randomize,
        "hand_type": "wuji1",
        # This selects the repository's detached, state-driven Wuji hand
        # runtime.  Despite the historical profile name, no online IK or
        # simulated ROKAE arm is present while this process is running.
        "arm_type": SIMPLIFIED_POST_IK_ARM_TYPE,
        "launch_isaac": True,
        "headless": args.headless,
        "hz": args.hz,
        "isaac_device": args.device,
        "physics_backend": "physx",
    }
    if args.task == "d08_toast_bread":
        from handson.sim.isaaclab.d_long_horizon_env import (
            HandsOnDLongHorizonIsaacLabEnv,
        )

        return HandsOnDLongHorizonIsaacLabEnv(
            args.task,
            time_limit=210.0,
            **common,
        )

    from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv

    return HandsOnABCIsaacLabEnv(args.task, time_limit=120.0, **common)


def _compact_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, np.generic):
        return _compact_value(value.item())
    if isinstance(value, np.ndarray) and value.size <= 64:
        return value.tolist()
    if isinstance(value, (list, tuple)) and len(value) <= 64:
        compact_items = [_compact_value(item) for item in value]
        if all(item is not None for item in compact_items):
            return compact_items
    if isinstance(value, Mapping):
        compact = {
            str(key): _compact_value(item)
            for key, item in value.items()
            if not str(key).startswith("physx_")
        }
        return {key: item for key, item in compact.items() if item is not None}
    return None


def _status_state(
    task: str,
    controller: HandPrimitiveController,
    info: Mapping[str, Any],
    *,
    env_step: int,
    runtime: Mapping[str, Any],
    physical_hands: Mapping[str, Any] | None = None,
    objects: Mapping[str, Any] | None = None,
    adaptive: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ready": True,
        "task": task,
        "env_step": int(env_step),
        "succeed": bool(info.get("succeed", False)),
        "failed": bool(info.get("failed", False)),
        "runtime": dict(runtime),
        "controller": controller.state(),
        "physical_hands": _compact_value(dict(physical_hands or {})),
        "objects": _compact_value(dict(objects or {})),
        "adaptive": _compact_value(dict(adaptive or {})),
        "task_info": _compact_value(dict(info)),
    }


def _graspable_object_names(task: str, env: Any) -> tuple[str, ...]:
    if task == "d08_toast_bread":
        return ("d08_bread_0", "d08_bread_1")
    spec = getattr(env, "_spec", None)
    names = getattr(spec, "target_objects", ())
    return tuple(str(name) for name in names)


def _task_scene_state_names(task: str, env: Any) -> tuple[str, ...]:
    """Return dynamic objects plus fixtures needed by closed-loop planners."""

    extras = {
        "t26_place_fruit_in_bucket": ("bucket",),
        "t09_insert_peg": ("socket",),
        "d08_toast_bread": (
            "d08_toaster",
            "d08_toaster_button",
            "d08_bread_basket",
            "d08_plate",
        ),
    }.get(task, ())
    return tuple(dict.fromkeys((*_graspable_object_names(task, env), *extras)))


def _physical_object_state(task: str, env: Any) -> dict[str, Any]:
    """Read graspable object poses/velocities from PhysX tensor views."""

    state: dict[str, Any] = {}
    pose_reader = getattr(env, "_physics_body_pose", None)
    velocity_reader = getattr(env, "_physics_body_velocity", None)
    for name in _task_scene_state_names(task, env):
        pose = pose_reader(name) if callable(pose_reader) else None
        if pose is None:
            continue
        entry: dict[str, Any] = {
            "pose_wxyz": np.asarray(pose, dtype=np.float64).reshape(7).tolist(),
        }
        velocity = velocity_reader(name) if callable(velocity_reader) else None
        if velocity is not None:
            velocity_array = np.asarray(velocity, dtype=np.float64).reshape(6)
            entry["linear_velocity_m_s"] = velocity_array[:3].tolist()
            entry["angular_velocity_rad_s"] = velocity_array[3:].tolist()
        state[name] = entry
    return state


def _nearest_object_name(
    physical_hands: Mapping[str, Any],
    objects: Mapping[str, Any],
    side: str,
) -> str | None:
    hand = physical_hands.get(side, {})
    if not isinstance(hand, Mapping):
        return None
    pose = hand.get("actual_pose_wxyz")
    if pose is None:
        return None
    palm_position = np.asarray(pose, dtype=np.float64).reshape(7)[:3]
    candidates: list[tuple[float, str]] = []
    for name, value in objects.items():
        if not isinstance(value, Mapping) or value.get("pose_wxyz") is None:
            continue
        object_position = np.asarray(
            value["pose_wxyz"], dtype=np.float64
        ).reshape(7)[:3]
        candidates.append(
            (float(np.linalg.norm(object_position - palm_position)), str(name))
        )
    return min(candidates)[1] if candidates else None


def _assert_hands_only_runtime(env: Any) -> dict[str, Any]:
    """Fail closed if the environment silently falls back to arms or IK."""

    hands_only = bool(getattr(env, "_hands_only", False))
    direct_hand_control = bool(getattr(env, "_direct_hand_control", False))
    simplified_profile = bool(getattr(env, "_simplified_post_ik", False))
    physical_servo = bool(
        getattr(env, "_direct_physics_servo_enabled", False)
    )
    ik_controllers = getattr(env, "_ik_controllers", None)
    arm_joint_ids = getattr(env, "_arm_joint_ids", {})
    has_arm_joints = any(bool(ids) for ids in arm_joint_ids.values())
    robots = getattr(env, "_robots", {})
    finger_drives_ready = bool(robots) and all(
        bool(getattr(robot, "actuators", {}))
        for robot in robots.values()
    )
    if not (
        hands_only
        and direct_hand_control
        and simplified_profile
        and physical_servo
        and finger_drives_ready
        and ik_controllers == {}
        and not has_arm_joints
    ):
        raise RuntimeError(
            "hands-only safety check failed: expected finite-wrench detached "
            "hand servos, finger drives, zero arm joints, and zero online IK"
        )
    return {
        "control_mode": "physical_servo_hands_only",
        "physics_backend": "physx",
        "simulated_arms": 0,
        "simulated_hands": len(robots),
        "online_ik": False,
        "finite_wrench_palm_servo": True,
        "finite_effort_finger_drives": True,
        "state_writes_during_control": False,
        "physical_contacts": True,
        "physical_finger_feedback": True,
        "adaptive_grasp_lift_test": True,
    }


def _print_controls(api_address: tuple[str, int] | None) -> None:
    print(
        "\n[HandsOn/Primitive] 键盘控制（先点击 Isaac 窗口）：\n"
        "  W/S: 世界 X 前后    A/D: 世界 Y 左右    E/Q: 上/下\n"
        "  I/K: 手腕俯仰       J/L: 手腕偏航       U/O: 手腕滚转\n"
        "  G: 大抓取  P: 两指捏  C: 圆柱抓  B: 球形抓  V: 三指捏\n"
        "  X: 侧捏    N: 钩握    M: 自适应抓取并试抬  0: 放松  H: 张手\n"
        "  T: 切换左右手（D08） R: 复位              Esc: 退出\n",
        flush=True,
    )
    if api_address is not None:
        host, port = api_address
        base = f"http://{host}:{port}/v1"
        print(
            "[HandsOn/Primitive] 本机 API：\n"
            f"  GET  {base}/state\n"
            f"  POST {base}/move   "
            "{\"hand\":\"right\",\"delta_position_m\":[0.01,0,0]}\n"
            f"  POST {base}/rotate "
            "{\"hand\":\"right\",\"clockwise_degrees\":50}\n"
            f"  POST {base}/grasp  "
            "{\"hand\":\"right\",\"type\":\"power|cylindrical|spherical|"
            "pinch|tripod|lateral|hook|open\"}\n"
            f"  POST {base}/fingers "
            "{\"curls\":{\"thumb\":0.8,\"index\":1.0}}\n"
            f"  POST {base}/grasp/adaptive "
            "{\"object\":\"apple\",\"strategy\":\"auto\","
            "\"lift_distance_m\":0.03}\n"
            f"  POST {base}/lift_test "
            "{\"object\":\"apple\",\"z\":0.03}\n"
            f"  GET  {base}/primitives  # 查询全部原语/手势\n"
            f"  POST {base}/reset   {{}}\n",
            flush=True,
        )


def run(args: argparse.Namespace) -> int:
    env = None
    keyboard = None
    api = None
    broker = PrimitiveCommandBroker()
    adaptive = AdaptiveGraspExecutor()
    try:
        # Show the real Isaac Sim scene for this interactive primitive-control
        # launcher.  The repository's legacy external preview is a B03-only
        # MuJoCo proxy and cannot represent A01 or D08 faithfully.
        os.environ["HANDSON_ISAACLAB_DIRECT_EXTERNAL_PREVIEW"] = "0"
        # Opt into the contact-respecting finite-wrench palm servo.  The
        # runtime assertion below fails closed if the environment falls back
        # to the historical per-substep root/joint state writer.
        os.environ["HANDSON_ISAACLAB_PHYSICAL_HAND_SERVO"] = "1"
        # The detached palm servo represents the wrist/forearm support that
        # was intentionally removed from this runtime.  The conservative
        # migration defaults (2 N / 0.05 N m) cannot even hold a down-facing
        # Wuji hand while its fingers close, so contact reactions rotate the
        # palm away before a pinch forms.  These remain finite, overridable
        # wrench limits and PhysX/table contacts stay authoritative.
        os.environ.setdefault("HANDSON_DIRECT_HAND_LINEAR_STIFFNESS", "300")
        os.environ.setdefault("HANDSON_DIRECT_HAND_LINEAR_DAMPING", "20")
        os.environ.setdefault("HANDSON_DIRECT_HAND_MAX_FORCE", "15")
        os.environ.setdefault("HANDSON_DIRECT_HAND_ANGULAR_STIFFNESS", "80")
        os.environ.setdefault("HANDSON_DIRECT_HAND_ANGULAR_DAMPING", "4")
        os.environ.setdefault("HANDSON_DIRECT_HAND_MAX_TORQUE", "8")
        env = _make_environment(args)
        observation, info = env.reset(seed=args.seed)
        del observation
        runtime = _assert_hands_only_runtime(env)
        print(
            "[HandsOn/Primitive] runtime verified: physical-servo hands only, "
            f"hands={runtime['simulated_hands']}, arms=0, online_ik=false, "
            "state_writes_during_control=false, PhysX contacts=true",
            flush=True,
        )
        bimanual = bool(getattr(getattr(env, "_spec", None), "bimanual", False))
        controller = HandPrimitiveController(
            hand_type=getattr(env, "hand_type", "wuji1"),
            bimanual=bimanual,
            hand_slew_rate_rad_s=args.hand_slew_rate,
        )
        controller.sync_from_env(env)
        keyboard = KitKeyboardPrimitiveInput(broker, enabled=not args.headless)
        api_address = None
        if not args.no_api:
            api = PrimitiveHTTPService(
                broker, host=args.api_host, port=args.api_port
            )
            api_address = api.start()
        _print_controls(api_address)
        physical_hands = env.direct_hand_servo_state()
        objects = _physical_object_state(args.task, env)
        broker.update_state(
            _status_state(
                args.task,
                controller,
                info,
                env_step=getattr(env, "env_step", 0),
                runtime=runtime,
                physical_hands=physical_hands,
                objects=objects,
                adaptive=adaptive.state(),
            )
        )

        quit_requested = False
        success_reported = False
        control_dt = 1.0 / max(1, int(args.hz))
        step_count = 0
        simulation_app = getattr(env, "_simulation_app", None)
        while not quit_requested:
            is_running = getattr(simulation_app, "is_running", None)
            if callable(is_running) and not is_running():
                break

            reset_requested = bool(
                getattr(env, "_manual_reset_requested", False)
            )
            for command in broker.drain():
                try:
                    effects = apply_primitive_command(controller, command)
                except PrimitiveCommandError as exc:
                    print(
                        f"[HandsOn/Primitive] command {command.sequence} "
                        f"rejected: {exc}",
                        flush=True,
                    )
                    continue
                if command.operation in {
                    "move",
                    "rotate",
                    "set_pose",
                    "grasp",
                    "fingers",
                }:
                    adaptive.cancel()
                adaptive_payload = effects.get("adaptive_grasp")
                if adaptive_payload is not None:
                    payload = dict(adaptive_payload)
                    side = str(
                        payload.get("hand", controller.selected_hand)
                    ).strip().lower()
                    if not payload.get("object") and not payload.get("object_name"):
                        nearest = _nearest_object_name(
                            physical_hands, objects, side
                        )
                        if nearest is not None:
                            payload["object"] = nearest
                    adaptive.start(controller, payload)
                lift_payload = effects.get("lift_test")
                if lift_payload is not None:
                    payload = dict(lift_payload)
                    side = str(
                        payload.get("hand", controller.selected_hand)
                    ).strip().lower()
                    if not payload.get("object") and not payload.get("object_name"):
                        nearest = _nearest_object_name(
                            physical_hands, objects, side
                        )
                        if nearest is not None:
                            payload["object"] = nearest
                    adaptive.start_lift_test(controller, payload)
                if effects.get("cancel"):
                    adaptive.cancel()
                reset_requested = reset_requested or effects["reset"]
                quit_requested = quit_requested or effects["quit"]
            if quit_requested:
                break
            if reset_requested:
                _, info = env.reset(seed=args.seed)
                controller.sync_from_env(env)
                keyboard.clear()
                adaptive.cancel()
                physical_hands = env.direct_hand_servo_state()
                objects = _physical_object_state(args.task, env)
                success_reported = False
                print("[HandsOn/Primitive] task reset complete", flush=True)
            else:
                if keyboard.pressed:
                    adaptive.cancel()
                adaptive.update(
                    controller,
                    dt=control_dt,
                    physical_hands=physical_hands,
                    objects=objects,
                )
                keyboard.apply_continuous(
                    controller,
                    dt=control_dt,
                    translation_speed_m_s=args.translation_speed,
                    rotation_speed_deg_s=args.rotation_speed,
                )
                controller.advance(control_dt)
                _, _, _terminated, _truncated, info = env.step(
                    controller.action()
                )
                physical_hands = env.direct_hand_servo_state()
                objects = _physical_object_state(args.task, env)
                if not args.headless:
                    sim = getattr(env, "_sim", None)
                    if sim is not None:
                        sim.render()
                if info.get("succeed") and not success_reported:
                    success_reported = True
                    print(
                        f"[HandsOn/Primitive] SUCCESS task={args.task} "
                        f"step={getattr(env, 'env_step', step_count)}",
                        flush=True,
                    )

            broker.update_state(
                _status_state(
                    args.task,
                    controller,
                    info,
                    env_step=getattr(env, "env_step", step_count),
                    runtime=runtime,
                    physical_hands=physical_hands,
                    objects=objects,
                    adaptive=adaptive.state(),
                )
            )
            step_count += 1
            if args.max_steps > 0 and step_count >= args.max_steps:
                break
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        if keyboard is not None:
            keyboard.close()
        if api is not None:
            api.close()
        if env is not None:
            # Isaac Sim 5.1 can block in SimulationContext.stop() after an
            # imported MJCF scene has rendered.  The recorder uses this same
            # guarded skip-cleanup path for one-process-per-session workers.
            setattr(env, "_fast_shutdown_on_close", True)
            env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=SUPPORTED_TASKS, default=SUPPORTED_TASKS[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", "--max_steps", type=int, default=0)
    parser.add_argument("--hz", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--translation-speed", type=float, default=0.12)
    parser.add_argument("--rotation-speed", type=float, default=45.0)
    parser.add_argument("--hand-slew-rate", type=float, default=1.5)
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8765)
    parser.add_argument("--no-api", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
