from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from handson.sim.isaaclab.p002_contract import (
    ASSET_RELATIVE_PATH,
    ASSET_SHA256,
    CONTROL_HZ,
    DOOR_BODY_PATH,
    DOOR_ESCAPE_DEG,
    EXPLOSION_ANGULAR_SPEED_RADPS,
    FORBIDDEN_CONTACT_FORCE_N,
    FRONT_HANDLE_BODY_PATH,
    GRAVITY_DISABLED_BODY_PATHS,
    HANDLE_PRESS_DEG,
    HANDLE_PRESS_DWELL_STEPS,
    HANDLE_RELEASE_DEG,
    LatchPhase,
    MAX_PENETRATION_M,
    NOMINAL_ROOT_XYZ_M,
    P002LatchStateMachine,
    P002PhysicalReadback,
    P002ResetState,
    P002_ROLE_CONTRACT_VERSION,
    P002ScriptedPolicy,
    P002SuccessTracker,
    LEFT_HOME,
    REQUIRED_CONTACT_FORCE_N,
    RIGHT_HOME,
    RIGHT_HANDLE_PALM_NOMINAL_Y_M,
    REAR_HANDLE_BODY_PATH,
    ROOT_XY_HALF_RANGE_M,
    ROOT_YAW_HALF_RANGE_DEG,
    SUCCESS_HOLD_STEPS,
    SCRIPTED_DOOR_GOAL_DEG,
    TASK_ID,
    TASK_NAME,
    TASK_PROMPT,
    TARGET_DOOR_DEG,
    TIMEOUT_STEPS,
    _door_push_tcp_pose,
    _task_pose,
    evaluate_physical_readback,
    physical_failure_reasons,
    sample_reset_state,
)
from handson.sim.isaaclab.p001_contract import (
    LEFT_HOME as SHARED_STANDARD_LEFT_HOME,
    RIGHT_HOME as SHARED_STANDARD_RIGHT_HOME,
)
from handson.sim.isaaclab.abc_collection_env import (
    HandsOnABCIsaacLabEnv,
    _RECORDING_CAMERA_NAMES,
    _RECORDING_CAMERA_PRIM_PATHS,
    _TASKS_WITHOUT_CENTER_RIG,
)
from handson.sim.isaaclab.p002_collection_env import (
    HandsOnP002IsaacLabEnv,
    _P002_ARM_HOME_RAD,
    _P002_EXPECTED_ROBOT_BASE_POSES_WXYZ,
    _randomized_root_pose_xyzw,
)
from handson.sim.isaaclab.recording_origin import (
    stable_recording_origin_kwargs_for_task,
)


TASK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TASK_DIR.parents[2]


def _shared_bimanual_home_contract() -> tuple[dict[str, tuple], dict[str, tuple]]:
    source = (
        REPOSITORY_ROOT
        / "handson/handson/sim/isaaclab/bimanual_assembly_scene.py"
    ).read_text(encoding="utf-8")
    wanted = {"_RIGHT_ARM_HOME", "_LEFT_ARM_MIRROR_SIGNS", "_BASE_POSE"}
    values: dict[str, object] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            values[target.id] = ast.literal_eval(node.value)
    assert values.keys() == wanted
    right = tuple(values["_RIGHT_ARM_HOME"])
    signs = tuple(values["_LEFT_ARM_MIRROR_SIGNS"])
    homes = {
        "right": right,
        "left": tuple(sign * value for sign, value in zip(signs, right)),
    }
    return homes, dict(values["_BASE_POSE"])


def _readback(**overrides) -> P002PhysicalReadback:
    values = {
        "front_handle_angle_rad": math.radians(50.0),
        "rear_handle_angle_rad": 0.0,
        "door_angle_rad": math.radians(45.0),
        "door_angular_speed_radps": 0.0,
        "left_front_handle_contact_force_n": REQUIRED_CONTACT_FORCE_N + 0.1,
        "right_door_contact_force_n": REQUIRED_CONTACT_FORCE_N + 0.1,
        "robot_table_contact_force_n": 0.0,
        "robot_frame_contact_force_n": 0.0,
        "robot_rear_handle_contact_force_n": 0.0,
        "max_penetration_m": 0.0,
        "hard_failure_max_penetration_m": 0.0,
    }
    canonical_aliases = {
        "right_front_handle_contact_force_n": (
            "left_front_handle_contact_force_n"
        ),
        "left_door_contact_force_n": "right_door_contact_force_n",
    }
    for canonical, legacy_slot in canonical_aliases.items():
        if canonical in overrides:
            if legacy_slot in overrides:
                raise ValueError(f"duplicate P0-02 readback override: {canonical}")
            overrides[legacy_slot] = overrides.pop(canonical)
    values.update(overrides)
    return P002PhysicalReadback(**values)


def _unlock(latch: P002LatchStateMachine) -> None:
    for _ in range(HANDLE_PRESS_DWELL_STEPS):
        latch.update(_readback(door_angle_rad=0.0))
    assert latch.unlocked_seen and not latch.locked


def test_identity_and_exact_asset_hash() -> None:
    assert TASK_ID == "P0-02"
    assert TASK_NAME == "t17_open_latched_door"
    assert P002_ROLE_CONTRACT_VERSION == 4
    asset = REPOSITORY_ROOT / ASSET_RELATIVE_PATH
    assert asset.is_file()
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == ASSET_SHA256


@pytest.mark.parametrize("side", ["right", "left"])
def test_p002_robot_cfg_keeps_platform_base_and_standard_arm_home(side: str) -> None:
    shared_arm_home, shared_base_pose = _shared_bimanual_home_contract()
    assert _P002_ARM_HOME_RAD[side] == shared_arm_home[side]
    assert _P002_EXPECTED_ROBOT_BASE_POSES_WXYZ[side] == (
        *shared_base_pose[side][0],
        *shared_base_pose[side][1],
    )
    environment = object.__new__(HandsOnP002IsaacLabEnv)
    config = SimpleNamespace(
        init_state=SimpleNamespace(
            pos=(-0.75, -0.065 if side == "right" else 0.065, 1.34),
            joint_pos={
                f"joint{index}_{side}(_1)?": value
                for index, value in enumerate(_P002_ARM_HOME_RAD[side], start=1)
            },
        ),
        spawn=SimpleNamespace(),
    )
    configured = environment._configure_task_robot_cfg(config, side)
    assert configured.init_state.pos == pytest.approx(
        (-0.75, -0.065 if side == "right" else 0.065, 1.34)
    )
    assert [
        configured.init_state.joint_pos[f"joint{index}_{side}(_1)?"]
        for index in range(1, 8)
    ] == pytest.approx(_P002_ARM_HOME_RAD[side], abs=1.0e-12)
    wrong_base = SimpleNamespace(
        init_state=SimpleNamespace(
            pos=(-0.40, -0.065 if side == "right" else 0.065, 1.34),
            joint_pos=dict(configured.init_state.joint_pos),
        ),
        spawn=SimpleNamespace(),
    )
    with pytest.raises(RuntimeError, match="requires the standard .* robot base"):
        environment._configure_task_robot_cfg(wrong_base, side)


def test_live_acceptance_uses_the_runtime_base_contract() -> None:
    source = (TASK_DIR / "live_physics_acceptance.py").read_text(encoding="utf-8")
    assert "_P002_EXPECTED_ROBOT_BASE_POSES_WXYZ" in source
    assert "_P002_EXPECTED_PALM_POSES_WXYZ" in source
    assert '"bilateral_near_home_motion_targets_reachable"' in source
    assert "target[0] += 0.02" in source
    assert "\n_EXPECTED_ROBOT_BASE_POSES_WXYZ =" not in source
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_robot_initial_state_evidence"
    ]
    assert len(calls) == 2
    assert all(
        any(keyword.arg == "expected_base_poses_wxyz" for keyword in call.keywords)
        for call in calls
    )


def test_p002_declared_palms_match_the_shared_standard_physical_home() -> None:
    assert RIGHT_HOME == pytest.approx(SHARED_STANDARD_RIGHT_HOME)
    assert LEFT_HOME == pytest.approx(SHARED_STANDARD_LEFT_HOME)


def test_only_spring_returned_levers_disable_gravity() -> None:
    assert GRAVITY_DISABLED_BODY_PATHS == frozenset(
        {FRONT_HANDLE_BODY_PATH, REAR_HANDLE_BODY_PATH}
    )
    assert DOOR_BODY_PATH not in GRAVITY_DISABLED_BODY_PATHS


def test_p002_permanently_omits_the_nonsemantic_center_rig() -> None:
    assert TASK_NAME in _TASKS_WITHOUT_CENTER_RIG
    for script_name in ("live_physics_acceptance.py", "live_reachability_sweep.py"):
        source = (TASK_DIR / script_name).read_text(encoding="utf-8")
        assert "P002_SKIP_CENTER_RIG" not in source


def test_p002_actual_robot_gate_runs_before_collection_ready() -> None:
    source = inspect.getsource(HandsOnABCIsaacLabEnv._setup_isaaclab)
    gate = source.index("self._validate_task_initial_state_before_ready()")
    ready = source.index("Native ABC collection ready")
    assert gate < ready

    environment = object.__new__(HandsOnP002IsaacLabEnv)
    environment._robots = {"right": object(), "left": object()}
    environment._p002_articulation_view = object()
    robots = {"right": {"passed": True}, "left": {"passed": True}}
    task = {"passed": True}
    environment._validate_initial_robot_state_or_raise = lambda: robots
    environment._validate_initial_task_state_or_raise = lambda: task

    assert environment._validate_task_initial_state_before_ready() == {
        "robots": robots,
        "task": task,
    }
    assert environment._p002_initial_robot_state_report == robots
    assert environment._p002_initial_task_state_report == task


def test_p002_actual_task_gate_checks_root_joints_contacts_and_penetration() -> None:
    class _View:
        count = 1

        @staticmethod
        def get_root_transforms() -> np.ndarray:
            return np.asarray([[0.10, -0.20, 1.00, 0.0, 0.0, 0.0, 1.0]])

    environment = object.__new__(HandsOnP002IsaacLabEnv)
    environment._p002_articulation_view = _View()
    environment._p002_reset_frame_pose_wxyz = np.asarray(
        [0.10, -0.20, 1.00, 1.0, 0.0, 0.0, 0.0]
    )
    environment._joint_positions = lambda: np.zeros(3, dtype=np.float64)
    environment._joint_velocities = lambda: np.zeros(3, dtype=np.float64)
    clearance_readings = {
        "initial_robot_front_handle": (0.0, 0.0),
        "initial_robot_door": (0.0, 0.0),
        "robot_frame": (0.0, 0.0),
        "robot_rear_handle": (0.0, 0.0),
    }
    environment._contact_reading = lambda key: clearance_readings[key]
    clear = _readback(
        right_front_handle_contact_force_n=0.0,
        left_door_contact_force_n=0.0,
        robot_frame_contact_force_n=0.0,
        robot_rear_handle_contact_force_n=0.0,
    )
    environment._physical_readback = lambda: clear

    report = environment._validate_initial_task_state_or_raise()

    assert report["passed"] is True
    assert all(report["checks"].values())

    contact = _readback(
        right_front_handle_contact_force_n=0.0,
        left_door_contact_force_n=0.0,
        robot_frame_contact_force_n=0.0,
        robot_rear_handle_contact_force_n=FORBIDDEN_CONTACT_FORCE_N,
    )
    environment._physical_readback = lambda: contact
    with pytest.raises(RuntimeError, match="initial task pose/joint/contact gate"):
        environment._validate_initial_task_state_or_raise()

    environment._physical_readback = lambda: clear
    clearance_readings["initial_robot_door"] = (0.0, 1.0e-6)
    with pytest.raises(RuntimeError, match="initial task pose/joint/contact gate"):
        environment._validate_initial_task_state_or_raise()


def test_p002_actual_robot_gate_repeats_after_every_public_reset(
    monkeypatch,
) -> None:
    environment = object.__new__(HandsOnP002IsaacLabEnv)
    environment._p002_reset_serial = 8
    environment._p002_articulation_view = object()
    calls: list[str] = []
    environment._validate_task_initial_state_before_ready = (
        lambda: calls.append("actual_state_gate")
    )
    monkeypatch.setattr(
        HandsOnABCIsaacLabEnv,
        "reset",
        lambda _self, seed=None, **_kwargs: ("observation", {"seed": seed}),
    )

    result = environment.reset(seed=41)

    assert result == ("observation", {"seed": 41})
    assert environment._p002_reset_serial == -1
    assert calls == ["actual_state_gate"]


def test_p002_actual_robot_gate_repeats_at_stable_recording_origin(
    monkeypatch,
) -> None:
    environment = object.__new__(HandsOnP002IsaacLabEnv)
    robots = {"right": {"passed": True}, "left": {"passed": True}}
    task = {"passed": True}
    environment._validate_task_initial_state_before_ready = lambda: {
        "robots": robots,
        "task": task,
    }
    monkeypatch.setattr(
        HandsOnABCIsaacLabEnv,
        "accept_stable_recording_origin",
        lambda _self, _report: None,
    )

    acceptance = environment.accept_stable_recording_origin(
        {"physics_steps": 1200}
    )

    assert acceptance == {
        "accepted": True,
        "kind": "p002_actual_robot_and_task_state_boundary",
        "initial_robot_state": robots,
        "initial_task_state": task,
    }


def test_p002_production_origin_requires_ten_stable_seconds() -> None:
    profile = stable_recording_origin_kwargs_for_task(TASK_NAME)

    assert profile["stable_seconds"] == pytest.approx(10.0)
    assert profile["stability_sample_seconds"] == pytest.approx(0.0)
    assert profile["timeout_seconds"] >= profile["stable_seconds"]


def test_recording_cameras_do_not_depend_on_the_omitted_center_rig() -> None:
    assert _RECORDING_CAMERA_NAMES == (
        "head_camera",
        "wrist_left_camera",
        "wrist_right_camera",
    )
    assert set(_RECORDING_CAMERA_PRIM_PATHS) == set(_RECORDING_CAMERA_NAMES)
    assert all(
        path == f"/World/RecordCameras/{name}"
        for name, path in _RECORDING_CAMERA_PRIM_PATHS.items()
    )
    assert all("/CenterRig" not in path for path in _RECORDING_CAMERA_PRIM_PATHS.values())


def test_initial_overlay_is_ascii_descriptive_and_not_successful() -> None:
    environment = object.__new__(HandsOnP002IsaacLabEnv)
    environment.task_name = TASK_NAME
    environment._p002_last_metrics = {
        "phase": "locked",
        "front_handle_angle_deg": 0.0,
        "door_angle_deg": 0.0,
        "door_stable": True,
        "forbidden_clear": True,
        "penetration_ok": True,
    }
    environment._success_counter = 0

    status = environment._task_status_text()
    rows = environment._success_condition_rows()
    display_labels = [
        environment._success_condition_display_label(row) for row in rows
    ]

    assert status == "Latch locked  |  Right lever 0.0 deg  |  Left door 0.0 deg"
    assert status.isascii()
    assert "?" not in status
    assert [row["key"] for row in rows] == [
        "p002_right_handle",
        "p002_unlatched",
        "p002_left_door",
        "p002_target",
        "p002_stable",
        "p002_forbidden",
        "success_hold",
    ]
    assert [row["ok"] for row in rows] == [
        False,
        False,
        False,
        False,
        True,
        True,
        False,
    ]
    assert all(label.isascii() and "?" not in label for label in display_labels)
    assert all(not label.startswith("p002 ") for label in display_labels)
    assert display_labels[0] == "Right hand contacts and depresses the front lever"
    assert display_labels[2] == "Left hand contacts the panel after unlatching"
    assert display_labels[3] == "Door angle is at least 45 deg"
    assert display_labels[5] == "Rear handle clear; penetration <= 5 mm"


def test_runtime_contract_uses_right_handle_and_left_door_roles() -> None:
    assert TASK_PROMPT.startswith("Use the right hand to depress")
    assert "left hand pushes the door panel" in TASK_PROMPT
    assert "at least 45 degrees" in TASK_PROMPT

    readback = _readback(
        right_front_handle_contact_force_n=1.25,
        left_door_contact_force_n=2.5,
    )
    assert readback.right_front_handle_contact_force_n == pytest.approx(1.25)
    assert readback.left_door_contact_force_n == pytest.approx(2.5)
    # The original dataclass slots remain available for old recordings/code.
    assert readback.left_front_handle_contact_force_n == pytest.approx(1.25)
    assert readback.right_door_contact_force_n == pytest.approx(2.5)


def test_runtime_contact_views_filter_the_correct_hands() -> None:
    class _MetaType:
        dof_indices = {
            "Door_joint": 0,
            "Handle001_joint": 1,
            "Handle002_joint": 2,
        }

    class _Articulation:
        count = 1
        max_dofs = 3

        def get_metatype(self, _index: int) -> _MetaType:
            return _MetaType()

    class _RigidBody:
        count = 1

    class _PhysicsView:
        def create_articulation_view(self, _path: str) -> _Articulation:
            return _Articulation()

        def create_rigid_body_view(self, _path: str) -> _RigidBody:
            return _RigidBody()

    class _Robot:
        _physics_sim_view = _PhysicsView()

    environment = object.__new__(HandsOnP002IsaacLabEnv)
    environment._robots = {"left": _Robot()}
    environment._p002_body_views = {}
    environment._p002_contact_views = {}
    environment._p002_contact_paths = {}
    environment._robot_rigid_body_paths = lambda side, hand_only: [
        f"/{side}/{'hand' if hand_only else 'robot'}"
    ]
    created: list[tuple[str, str, tuple[str, ...]]] = []

    def create_contact_view(
        _physics: object,
        key: str,
        sensor_path: str,
        filter_paths: list[str],
    ) -> None:
        created.append((key, sensor_path, tuple(filter_paths)))

    environment._create_contact_view = create_contact_view
    environment._set_door_drive_locked = lambda _locked: None

    environment._init_task_specific_runtime()

    assert created == [
        ("right_front_handle", FRONT_HANDLE_BODY_PATH, ("/right/hand",)),
        ("left_door", DOOR_BODY_PATH, ("/left/hand",)),
        ("robot_frame", "/World/Task/P002Door/Door042", ("/left/robot", "/right/robot")),
        ("robot_rear_handle", REAR_HANDLE_BODY_PATH, ("/left/robot", "/right/robot")),
        (
            "initial_robot_front_handle",
            FRONT_HANDLE_BODY_PATH,
            ("/left/robot", "/right/robot"),
        ),
        (
            "initial_robot_door",
            DOOR_BODY_PATH,
            ("/left/robot", "/right/robot"),
        ),
    ]
    assert environment._p002_contact_contract_ready is True


def test_time_limit_is_reported_as_truncation_before_reset(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    environment = object.__new__(HandsOnP002IsaacLabEnv)
    environment.env_step = TIMEOUT_STEPS
    environment._control_dt = 1.0 / CONTROL_HZ
    environment._time_limit = TIMEOUT_STEPS / CONTROL_HZ
    environment._p002_terminal_reported = False

    def base_step(_self: object, *, start_time: float | None = None):
        del start_time
        return "observation", 0.0, True, False, {
            "succeed": False,
            "failed": False,
            "manual_reset": False,
        }

    monkeypatch.setattr(
        HandsOnABCIsaacLabEnv,
        "step_with_current_joint_targets",
        base_step,
    )

    observation, reward, terminated, truncated, info = (
        environment.step_with_current_joint_targets()
    )

    assert observation == "observation"
    assert reward == 0.0
    assert terminated is False
    assert truncated is True
    assert info["time_limit_reached"] is True
    assert info["truncated_reason"] == "time_limit"
    assert info["p002_terminal_reason"] == "time_limit"
    line = capsys.readouterr().out.strip()
    prefix = "[HandsOn/P002 terminal] "
    assert line.startswith(prefix)
    event = json.loads(line.removeprefix(prefix))
    assert event == {
        "door_angle_deg": None,
        "door_escape_seen": False,
        "elapsed_sim_time_seconds": TIMEOUT_STEPS / CONTROL_HZ,
        "env_step": TIMEOUT_STEPS,
        "event": "p002_first_terminal",
        "front_handle_angle_deg": None,
        "handle_press_threshold_deg": HANDLE_PRESS_DEG,
        "handle_release_threshold_deg": HANDLE_RELEASE_DEG,
        "latch_locked": True,
        "latch_phase": "unknown",
        "latch_press_steps": 0,
        "p002_drive_locked": True,
        "reason": "time_limit",
        "right_front_handle_contact_force_n": None,
        "terminated": False,
        "time_limit_seconds": TIMEOUT_STEPS / CONTROL_HZ,
        "timeout_steps": TIMEOUT_STEPS,
        "truncated": True,
        "unlocked_seen": False,
    }

    environment.step_with_current_joint_targets()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "terminal_info",
    [
        {"succeed": True, "failed": False, "manual_reset": False},
        {"succeed": False, "failed": False, "manual_reset": True},
    ],
)
def test_success_and_manual_reset_take_priority_over_time_limit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    terminal_info: dict[str, bool],
) -> None:
    environment = object.__new__(HandsOnP002IsaacLabEnv)
    environment.env_step = TIMEOUT_STEPS
    environment._control_dt = 1.0 / CONTROL_HZ
    environment._time_limit = TIMEOUT_STEPS / CONTROL_HZ
    environment._p002_terminal_reported = False

    monkeypatch.setattr(
        HandsOnABCIsaacLabEnv,
        "step_with_current_joint_targets",
        lambda _self, *, start_time=None: (
            "observation",
            0.0,
            True,
            False,
            dict(terminal_info),
        ),
    )

    _observation, _reward, terminated, truncated, info = (
        environment.step_with_current_joint_targets()
    )

    assert terminated is True
    assert truncated is False
    assert "time_limit_reached" not in info
    assert "truncated_reason" not in info
    assert capsys.readouterr().out == ""


def test_physical_failure_reasons_have_stable_priority_and_boundaries() -> None:
    assert physical_failure_reasons(_readback()) == ()
    assert physical_failure_reasons(
        _readback(door_angular_speed_radps=EXPLOSION_ANGULAR_SPEED_RADPS)
    ) == ()
    assert physical_failure_reasons(
        _readback(hard_failure_max_penetration_m=MAX_PENETRATION_M)
    ) == ()
    assert physical_failure_reasons(
        _readback(robot_table_contact_force_n=FORBIDDEN_CONTACT_FORCE_N)
    ) == ()
    assert physical_failure_reasons(
        _readback(robot_table_contact_force_n=float("nan"))
    ) == ()

    readback = _readback(
        door_angular_speed_radps=EXPLOSION_ANGULAR_SPEED_RADPS + 0.1,
        hard_failure_max_penetration_m=MAX_PENETRATION_M + 0.001,
        robot_table_contact_force_n=FORBIDDEN_CONTACT_FORCE_N + 0.01,
        robot_frame_contact_force_n=FORBIDDEN_CONTACT_FORCE_N + 0.02,
        robot_rear_handle_contact_force_n=FORBIDDEN_CONTACT_FORCE_N + 0.03,
    )
    expected = (
        "door_speed_explosion",
        "penetration_limit",
        "forbidden_rear_handle_contact",
    )
    assert physical_failure_reasons(readback) == expected
    metrics = evaluate_physical_readback(readback, P002LatchStateMachine())
    assert metrics["failed"] is True
    assert metrics["failure_reason"] == expected[0]
    assert metrics["failure_reasons"] == expected
    assert physical_failure_reasons(
        _readback(front_handle_angle_rad=float("nan"))
    ) == ("nonfinite_readback",)


def test_table_door_panel_and_frame_contact_are_diagnostic_only() -> None:
    readback = _readback(
        robot_table_contact_force_n=100.0,
        robot_frame_contact_force_n=100.0,
        max_penetration_m=0.100,
        hard_failure_max_penetration_m=0.0,
    )
    assert physical_failure_reasons(readback) == ()

    latch = P002LatchStateMachine()
    _unlock(latch)
    latch.update(readback)
    metrics = evaluate_physical_readback(readback, latch)
    assert metrics["success_now"] is True
    assert metrics["failed"] is False
    assert metrics["forbidden_clear"] is True
    assert metrics["penetration_ok"] is True


def test_runtime_excludes_table_door_and_frame_from_protected_penetration() -> None:
    environment = object.__new__(HandsOnP002IsaacLabEnv)
    environment._p002_dof_indices = {
        "front_handle": 0,
        "rear_handle": 1,
        "door": 2,
    }
    environment._joint_positions = lambda: np.asarray(
        [math.radians(50.0), 0.0, math.radians(45.0)], dtype=np.float64
    )
    environment._joint_velocities = lambda: np.zeros(3, dtype=np.float64)
    readings = {
        "right_front_handle": (REQUIRED_CONTACT_FORCE_N + 0.1, 0.001),
        "left_door": (500.0, 0.050),
        "robot_frame": (500.0, 0.060),
        "robot_rear_handle": (0.0, 0.003),
    }
    requested_keys = []

    def contact_reading(key: str) -> tuple[float, float]:
        requested_keys.append(key)
        return readings[key]

    environment._contact_reading = contact_reading

    readback = environment._physical_readback()

    assert requested_keys == [
        "right_front_handle",
        "left_door",
        "robot_frame",
        "robot_rear_handle",
    ]
    assert readback.max_penetration_m == pytest.approx(0.060)
    assert readback.hard_failure_max_penetration_m == pytest.approx(0.003)
    assert readback.robot_table_contact_force_n == 0.0
    assert readback.robot_frame_contact_force_n == pytest.approx(500.0)
    assert readback.left_door_contact_force_n == pytest.approx(500.0)
    assert readback.right_front_handle_contact_force_n == pytest.approx(
        REQUIRED_CONTACT_FORCE_N + 0.1
    )
    assert physical_failure_reasons(readback) == ()


def test_recorded_role_arrays_keep_their_existing_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = object.__new__(HandsOnP002IsaacLabEnv)
    environment._p002_last_readback = _readback(
        right_front_handle_contact_force_n=1.25,
        left_door_contact_force_n=2.5,
        robot_table_contact_force_n=0.0,
        robot_frame_contact_force_n=3.75,
        robot_rear_handle_contact_force_n=4.0,
    )
    environment._p002_last_metrics = {
        "target_ok": False,
        "door_stable": True,
        "forbidden_clear": False,
    }
    environment._p002_articulation_view = None
    environment._p002_body_views = {}
    environment._p002_latch = P002LatchStateMachine()
    environment._success_counter = 0
    environment._p002_reset = sample_reset_state(0)
    environment._p002_drive_locked = True
    environment._p002_scene_geometry_report = None
    monkeypatch.setattr(
        HandsOnABCIsaacLabEnv,
        "_compute_observation",
        lambda _self: {"state": {}},
    )

    state = environment._compute_observation()["state"]

    assert state["p002_role_contacts"].shape == (5,)
    assert state["p002_role_contacts"].tolist() == [1.25, 2.5, 0.0, 3.75, 4.0]
    assert state["p002_latch_state"].shape == (8,)
    assert state["p002_latch_state"][5] == 0.0
    assert state["p002_role_contract_version"].shape == (1,)
    assert state["p002_role_contract_version"].tolist() == [4.0]
    assert environment._task_info()["p002_role_contract_version"] == 4

    environment._p002_last_metrics["target_ok"] = True
    updated_state = environment._compute_observation()["state"]
    assert updated_state["p002_latch_state"].shape == (8,)
    assert updated_state["p002_latch_state"][5] == 1.0


def test_first_physical_failure_is_frozen_and_logged_once_before_reset(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    environment = object.__new__(HandsOnP002IsaacLabEnv)
    environment.env_step = TIMEOUT_STEPS - 1
    environment._control_dt = 1.0 / CONTROL_HZ
    environment._time_limit = TIMEOUT_STEPS / CONTROL_HZ
    environment._p002_reset = sample_reset_state(17)
    environment._p002_latch = P002LatchStateMachine()
    environment._p002_failure_latched = False
    environment._p002_first_failure_event = None
    environment._p002_terminal_reported = False
    environment._trajectory_action_phase_active = True
    environment._trajectory_action_phase_substep_index = 2
    first_readback = _readback(
        robot_rear_handle_contact_force_n=FORBIDDEN_CONTACT_FORCE_N + 0.25,
        right_front_handle_contact_force_n=1.25,
        left_door_contact_force_n=2.5,
    )
    environment._physical_readback = lambda: first_readback

    first_metrics = environment._refresh_metrics(update_latch=False)
    assert first_metrics["failure_reason"] == "forbidden_rear_handle_contact"
    assert environment._p002_failure_latched is True
    frozen_event = dict(environment._p002_first_failure_event)

    environment._physical_readback = _readback
    environment._refresh_metrics(update_latch=False)
    assert environment._p002_first_failure_event == frozen_event

    def base_step(_self: object, *, start_time: float | None = None):
        del start_time
        _self.env_step += 1
        return "observation", 0.0, True, False, {
            "succeed": False,
            "failed": True,
            "manual_reset": False,
        }

    monkeypatch.setattr(
        HandsOnABCIsaacLabEnv,
        "step_with_current_joint_targets",
        base_step,
    )
    observation, reward, terminated, truncated, info = (
        environment.step_with_current_joint_targets()
    )

    assert observation == "observation"
    assert reward == 0.0
    assert terminated is True
    assert truncated is False
    assert info["failure_reason"] == "forbidden_rear_handle_contact"
    assert info["p002_failure_reason"] == "forbidden_rear_handle_contact"
    assert info["p002_terminal_reason"] == "forbidden_rear_handle_contact"
    assert info["p002_failure_reasons"] == ["forbidden_rear_handle_contact"]
    assert "time_limit_reached" not in info
    assert "truncated_reason" not in info

    prefix = "[HandsOn/P002 terminal] "
    line = capsys.readouterr().out.strip()
    assert line.startswith(prefix)
    event = json.loads(line.removeprefix(prefix))
    assert event["event"] == "p002_first_terminal"
    assert event["reason"] == "forbidden_rear_handle_contact"
    assert event["failure_reasons"] == ["forbidden_rear_handle_contact"]
    assert event["failure_observed_env_step"] == TIMEOUT_STEPS - 1
    assert event["env_step"] == TIMEOUT_STEPS
    assert event["physics_substep_index"] == 2
    assert event["failure_observed_during_action"] is True
    assert event["robot_rear_handle_contact_force_n"] == pytest.approx(
        FORBIDDEN_CONTACT_FORCE_N + 0.25
    )
    assert event["robot_table_contact_force_n"] == 0.0
    assert event["right_front_handle_contact_force_n"] == pytest.approx(1.25)
    assert event["left_door_contact_force_n"] == pytest.approx(2.5)
    assert event["left_front_handle_contact_force_n"] == pytest.approx(1.25)
    assert event["right_door_contact_force_n"] == pytest.approx(2.5)
    assert event["forbidden_contact_force_threshold_n"] == (
        FORBIDDEN_CONTACT_FORCE_N
    )
    assert event["rear_handle_contact_force_threshold_n"] == (
        FORBIDDEN_CONTACT_FORCE_N
    )
    assert event["terminated"] is True
    assert event["truncated"] is False
    assert event["nonfinite_fields"] == []

    environment.step_with_current_joint_targets()
    assert capsys.readouterr().out == ""


def test_nonfinite_failure_snapshot_remains_strict_json() -> None:
    environment = object.__new__(HandsOnP002IsaacLabEnv)
    environment.env_step = 0
    environment._p002_latch = P002LatchStateMachine()
    environment._p002_last_readback = _readback(
        door_angular_speed_radps=float("nan")
    )
    environment._trajectory_action_phase_active = True
    environment._trajectory_action_phase_substep_index = 0
    metrics = evaluate_physical_readback(
        environment._p002_last_readback,
        environment._p002_latch,
    )

    event = environment._first_failure_snapshot(metrics)

    assert event["reason"] == "nonfinite_readback"
    assert event["door_angular_speed_radps"] == "nan"
    assert event["nonfinite_fields"] == ["door_angular_speed_radps"]
    json.dumps(event, allow_nan=False)


def test_sample_reset_clears_first_failure_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = object.__new__(HandsOnP002IsaacLabEnv)
    environment.delta_h = np.ones(1, dtype=np.float64)
    environment._seed = 31
    environment._p002_reset_serial = -1
    environment._p002_latch = P002LatchStateMachine()
    environment._p002_failure_latched = True
    environment._p002_first_failure_event = {"reason": "old_failure"}
    environment._p002_terminal_reported = True
    monkeypatch.setattr(
        HandsOnABCIsaacLabEnv,
        "_sample_reset_state",
        lambda _self, *, clear_manual_request: None,
    )

    environment._sample_reset_state(clear_manual_request=True)

    assert environment._p002_failure_latched is False
    assert environment._p002_first_failure_event is None
    assert environment._p002_terminal_reported is False


def test_recorded_initial_state_restore_clears_failure_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = object.__new__(HandsOnP002IsaacLabEnv)
    environment._p002_failure_latched = True
    environment._p002_first_failure_event = {"reason": "old_failure"}
    environment._p002_terminal_reported = True
    environment._p002_articulation_view = None
    environment._torch = None
    monkeypatch.setattr(
        HandsOnABCIsaacLabEnv,
        "restore_recorded_initial_state",
        lambda _self, state_parts: {"state": dict(state_parts)},
    )

    observation = environment.restore_recorded_initial_state({})

    assert observation == {"state": {}}
    assert environment._p002_failure_latched is False
    assert environment._p002_first_failure_event is None
    assert environment._p002_terminal_reported is False


def test_twenty_seeded_resets_are_deterministic_and_in_frozen_range() -> None:
    first = [sample_reset_state(seed) for seed in range(20)]
    assert first == [sample_reset_state(seed) for seed in range(20)]
    for reset in first:
        assert abs(reset.root_x_m - 0.01) <= ROOT_XY_HALF_RANGE_M
        assert abs(reset.root_y_m - 0.15) <= ROOT_XY_HALF_RANGE_M
        assert abs(reset.root_yaw_deg - 90.0) <= ROOT_YAW_HALF_RANGE_DEG


def test_closer_door_and_scripted_targets_share_one_task_frame() -> None:
    assert NOMINAL_ROOT_XYZ_M == pytest.approx([0.01, 0.15, 1.05])
    reset = P002ResetState(
        seed=0,
        root_x_m=0.01,
        root_y_m=0.15,
        root_z_m=1.05,
        root_yaw_deg=90.0,
    )

    action = P002ScriptedPolicy(reset).action(190)

    assert action[0] == pytest.approx(-0.18)
    assert action[7] == pytest.approx(-0.15)
    assert _door_push_tcp_pose(0.0)[0] == pytest.approx(-0.15)


def test_sampled_reset_delta_is_applied_from_nominal_composed_pose() -> None:
    composed = np.asarray(
        [0.0095971358, 0.15, 1.0500650406, 0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
        dtype=np.float64,
    )
    for seed in range(20):
        reset = sample_reset_state(seed)
        actual = _randomized_root_pose_xyzw(composed, reset)
        assert actual[0] - composed[0] == pytest.approx(reset.root_x_m - 0.01)
        assert actual[1] - composed[1] == pytest.approx(reset.root_y_m - 0.15)
        assert actual[2] - composed[2] == pytest.approx(reset.root_z_m - 1.05)
        expected_yaw = math.radians(reset.root_yaw_deg)
        assert actual[3:7] == pytest.approx(
            [0.0, 0.0, math.sin(expected_yaw / 2.0), math.cos(expected_yaw / 2.0)]
        )


def test_unlock_requires_contact_angle_and_exact_five_step_dwell() -> None:
    latch = P002LatchStateMachine()
    for _ in range(HANDLE_PRESS_DWELL_STEPS - 1):
        latch.update(_readback(door_angle_rad=0.0))
        assert latch.locked
    latch.update(_readback(door_angle_rad=0.0))
    assert not latch.locked
    assert latch.unlocked_seen
    assert latch.right_handle_contact_seen
    assert latch.left_handle_contact_seen  # legacy alias

    for bad in (
        _readback(front_handle_angle_rad=math.radians(HANDLE_PRESS_DEG - 1), door_angle_rad=0.0),
        _readback(right_front_handle_contact_force_n=0.0, door_angle_rad=0.0),
    ):
        latch = P002LatchStateMachine()
        for _ in range(HANDLE_PRESS_DWELL_STEPS + 2):
            latch.update(bad)
        assert latch.locked and not latch.unlocked_seen


def test_logged_helindu_operator_press_is_inside_the_unlock_band() -> None:
    assert HANDLE_PRESS_DEG == 25.0
    assert HANDLE_RELEASE_DEG == 15.0
    latch = P002LatchStateMachine()
    logged_press = _readback(
        front_handle_angle_rad=math.radians(28.168763939095815),
        right_front_handle_contact_force_n=61.19245529174805,
        door_angle_rad=0.0,
    )
    for _ in range(HANDLE_PRESS_DWELL_STEPS):
        latch.update(logged_press)
    assert latch.unlocked_seen and not latch.locked

    below_threshold = P002LatchStateMachine()
    for _ in range(HANDLE_PRESS_DWELL_STEPS + 1):
        below_threshold.update(
            _readback(
                front_handle_angle_rad=math.radians(HANDLE_PRESS_DEG - 0.1),
                right_front_handle_contact_force_n=61.19245529174805,
                door_angle_rad=0.0,
            )
        )
    assert below_threshold.locked and not below_threshold.unlocked_seen


def test_logged_helindu_press_requests_the_unlocked_door_drive() -> None:
    environment = object.__new__(HandsOnP002IsaacLabEnv)
    environment._p002_latch = P002LatchStateMachine()
    environment._p002_failure_latched = False
    environment._p002_first_failure_event = None
    environment._p002_last_readback = None
    environment._p002_last_metrics = {}
    environment._physical_readback = lambda: _readback(
        front_handle_angle_rad=math.radians(28.168763939095815),
        right_front_handle_contact_force_n=61.19245529174805,
        door_angle_rad=0.0,
    )
    drive_lock_commands: list[bool] = []
    environment._set_door_drive_locked = drive_lock_commands.append

    for _ in range(HANDLE_PRESS_DWELL_STEPS):
        environment._refresh_metrics(update_latch=True)

    assert environment._p002_latch.unlocked_seen
    assert drive_lock_commands == [False]


def test_left_door_contact_is_credited_only_after_unlock() -> None:
    latch = P002LatchStateMachine()
    latch.update(_readback(door_angle_rad=0.0))
    assert latch.left_door_contact_after_unlock_seen is False

    _unlock(latch)
    latch.update(_readback())

    assert latch.left_door_contact_after_unlock_seen is True
    assert latch.right_door_contact_after_unlock_seen is True  # legacy alias
    metrics = latch.to_dict()
    assert metrics["right_handle_contact_seen"] is True
    assert metrics["left_door_contact_after_unlock_seen"] is True
    assert metrics["left_handle_contact_seen"] is True
    assert metrics["right_door_contact_after_unlock_seen"] is True


def test_hysteretic_relock_and_escape_are_causal() -> None:
    latch = P002LatchStateMachine()
    _unlock(latch)
    latch.update(
        _readback(
            front_handle_angle_rad=math.radians(HANDLE_RELEASE_DEG - 1),
            door_angle_rad=math.radians(DOOR_ESCAPE_DEG - 1),
        )
    )
    assert latch.locked

    latch = P002LatchStateMachine()
    _unlock(latch)
    latch.update(
        _readback(
            front_handle_angle_rad=math.radians(HANDLE_RELEASE_DEG - 1),
            door_angle_rad=math.radians(DOOR_ESCAPE_DEG + 1),
        )
    )
    assert not latch.locked and latch.door_escape_seen


@pytest.mark.parametrize("trial", range(10))
def test_ten_positive_semantic_trials_have_zero_false_negatives(trial: int) -> None:
    latch = P002LatchStateMachine()
    _unlock(latch)
    latch.update(_readback())
    metrics = evaluate_physical_readback(_readback(), latch)
    assert metrics["success_now"] is True, trial
    assert metrics["failed"] is False


@pytest.mark.parametrize("door_angle_deg", [45.0, 52.0, 90.0])
def test_stable_door_angles_at_or_above_45_satisfy_the_minimum_angle_gate(
    door_angle_deg: float,
) -> None:
    latch = P002LatchStateMachine()
    _unlock(latch)
    readback = _readback(door_angle_rad=math.radians(door_angle_deg))
    latch.update(readback)

    metrics = evaluate_physical_readback(readback, latch)

    # ``target_ok`` is a compatibility key and the sixth existing latch-state
    # observation slot; its meaning is now this one-sided minimum-angle gate.
    assert metrics["target_ok"] is True
    assert metrics["success_now"] is True


def test_success_window_remains_fifteen_consecutive_stable_control_frames() -> None:
    environment = object.__new__(HandsOnP002IsaacLabEnv)
    assert environment._success_steps_required() == SUCCESS_HOLD_STEPS == 15

    tracker = P002SuccessTracker()
    assert tracker.hold_steps == SUCCESS_HOLD_STEPS
    for _ in range(SUCCESS_HOLD_STEPS - 1):
        assert tracker.update(True) is False
    assert tracker.update(True) is True

    # Any frame that loses the angle/speed/causality aggregate restarts the
    # consecutive window rather than terminating the episode.
    assert tracker.update(False) is False
    assert tracker.counter == 0


@pytest.mark.parametrize(
    "name,mutate_latch,overrides",
    [
        ("direct_push_no_handle", False, {}),
        ("no_right_handle_contact", True, {"right_front_handle_contact_force_n": 0.0}),
        ("short_press", False, {}),
        (
            "early_release_relock",
            True,
            {
                "front_handle_angle_rad": math.radians(20.0),
                "door_angle_rad": math.radians(5.0),
            },
        ),
        ("missing_left_door_contact", True, {"left_door_contact_force_n": 0.0}),
        (
            "wrong_far_handle",
            True,
            {
                "robot_rear_handle_contact_force_n": (
                    REQUIRED_CONTACT_FORCE_N + 0.1
                )
            },
        ),
        (
            "protected_penetration",
            True,
            {"hard_failure_max_penetration_m": MAX_PENETRATION_M + 0.001},
        ),
        ("target_low", True, {"door_angle_rad": math.radians(35.0)}),
        (
            "target_just_below_minimum",
            True,
            {"door_angle_rad": math.radians(45.0 - 1.0e-6)},
        ),
        ("unstable_target", True, {"door_angular_speed_radps": 0.6}),
    ],
)
def test_ten_negative_semantic_trials_have_zero_false_positives(
    name: str, mutate_latch: bool, overrides: dict[str, float]
) -> None:
    latch = P002LatchStateMachine()
    if mutate_latch:
        _unlock(latch)
        if name == "no_right_handle_contact":
            latch.phase = LatchPhase.UNLATCHED
            latch.unlocked_seen = True
            latch.right_handle_contact_seen = False
        latch.update(_readback(**overrides))
    elif name == "short_press":
        for _ in range(HANDLE_PRESS_DWELL_STEPS - 1):
            latch.update(_readback(door_angle_rad=0.0))
    readback = _readback(**overrides)
    metrics = evaluate_physical_readback(readback, latch)
    assert metrics["success_now"] is False, name


def test_scripted_policy_is_deterministic_finite_and_bimanual() -> None:
    reset = sample_reset_state(7)
    first = P002ScriptedPolicy(reset)
    second = P002ScriptedPolicy(reset)
    for step in (0, 30, 60, 90, 120, 160, 190, 260, 340, 500):
        action = first.action(step)
        assert action.shape == (54,)
        assert np.isfinite(action).all()
        assert np.array_equal(action, second.action(step))
        assert np.isclose(np.linalg.norm(action[3:7]), 1.0)
        assert np.isclose(np.linalg.norm(action[10:14]), 1.0)

    # The right-hand target stays on the operator-right side instead of
    # reusing the former left-hand +Y approach.
    assert RIGHT_HANDLE_PALM_NOMINAL_Y_M == -0.04


def test_scripted_policy_goal_has_headroom_above_human_success_minimum() -> None:
    assert TARGET_DOOR_DEG == 45.0
    assert SCRIPTED_DOOR_GOAL_DEG == 50.0
    assert TARGET_DOOR_DEG < SCRIPTED_DOOR_GOAL_DEG <= 90.0

    reset = sample_reset_state(7)
    latch = P002LatchStateMachine()
    _unlock(latch)
    at_human_minimum = _readback(
        door_angle_rad=math.radians(TARGET_DOOR_DEG)
    )

    action = P002ScriptedPolicy(reset).action(
        500,
        readback=at_human_minimum,
        latch=latch,
    )

    np.testing.assert_allclose(
        action[7:14],
        _task_pose(_door_push_tcp_pose(SCRIPTED_DOOR_GOAL_DEG), reset),
    )
    assert evaluate_physical_readback(at_human_minimum, latch)["target_ok"] is True
