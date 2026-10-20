from __future__ import annotations
import threading

import threading
from types import SimpleNamespace

import numpy as np

from handson.tasks import sim_teleop
from handson.tasks.sim_teleop import (
    BimanualTeleopConfig,
    DualArmViveHandTeleopWrapper,
    SingleArmTeleopConfig,
    SingleArmViveHandTeleopWrapper,
)


def _b02_wuji_env() -> SimpleNamespace:
    return SimpleNamespace(
        task_name=sim_teleop.B02_NUT_BOLT_TASK_NAME,
        hand_type="wuji1",
    )


def _representative_closed_wuji_hand() -> np.ndarray:
    return np.array(
        [
            1.263,
            -0.017,
            0.138,
            0.256,
            1.022,
            -0.039,
            0.232,
            0.980,
            1.109,
            -0.167,
            1.261,
            1.462,
            0.888,
            -0.150,
            1.440,
            1.627,
            0.407,
            -0.222,
            1.232,
            1.627,
        ],
        dtype=np.float64,
    )


def _translated_pose(x: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = float(x)
    return pose


def test_udp_receiver_discards_backlog_and_returns_newest_packet() -> None:
    class FakeSocket:
        def __init__(self) -> None:
            self.packets = [(b"old", "first"), (b"middle", "second"), (b"new", "third")]
            self.flags = []
            self.timeout = 0.1

        def recvfrom(self, _max_size, flags=0):
            self.flags.append(flags)
            if not self.packets:
                raise BlockingIOError
            return self.packets.pop(0)

        def gettimeout(self):
            return self.timeout

        def setblocking(self, enabled):
            assert enabled is False
            self.timeout = 0.0

        def settimeout(self, timeout):
            self.timeout = timeout

    sock = FakeSocket()

    data, address = sim_teleop._recv_latest_udp_datagram(sock, 4096)

    assert data == b"new"
    assert address == "third"
    assert sock.flags == [0, 0, 0, 0]
    assert sock.timeout == 0.1


def test_single_arm_manual_reset_rebases_without_start_delay(monkeypatch) -> None:
    sleeps = []
    monkeypatch.setattr(sim_teleop.time, "sleep", sleeps.append)
    wrapper = object.__new__(SingleArmViveHandTeleopWrapper)
    wrapper._tracker_lock = threading.RLock()
    wrapper.intervened = False
    wrapper.env = SimpleNamespace(get_end_effector_pose_matrix=lambda: np.eye(4))
    wrapper.intervened = False
    wrapper._tracker_lock = threading.Lock()
    wrapper.latest_tracker = _translated_pose(0.25)
    wrapper.tracker_start_world = None
    wrapper.ee_start = None
    wrapper._filtered_target_pose = None
    wrapper.config = SingleArmTeleopConfig()
    wrapper.target_frame = "tracker"

    wrapper.set_intervened(True, start_delay=False)

    assert sleeps == []
    assert wrapper.intervened
    np.testing.assert_allclose(
        wrapper.tracker_start_world,
        wrapper.latest_tracker,
        atol=1e-12,
    )
    np.testing.assert_allclose(wrapper.ee_start, np.eye(4), atol=1e-12)


def test_single_arm_isaac_wuji_holds_live_reset_hand_target_without_packets() -> None:
    reset_target = np.linspace(-0.2, 0.7, sim_teleop.WUJI_NATIVE_HAND_DIM)
    wrapper = object.__new__(SingleArmViveHandTeleopWrapper)
    wrapper._tracker_lock = threading.RLock()
    wrapper.intervened = False
    wrapper.env = SimpleNamespace(
        hand_type="wuji1",
        _right_hand=reset_target.copy(),
    )
    wrapper._single_wuji_native_hand = True
    wrapper._hand_action_dim = sim_teleop.WUJI_NATIVE_HAND_DIM

    held_target = wrapper._current_hand_action()

    np.testing.assert_allclose(held_target, reset_target, atol=1e-12)
    assert held_target is not wrapper.env._right_hand


def test_dual_arm_manual_reset_rebases_without_start_delay(monkeypatch) -> None:
    sleeps = []
    monkeypatch.setattr(sim_teleop.time, "sleep", sleeps.append)
    wrapper = object.__new__(DualArmViveHandTeleopWrapper)
    wrapper.env = SimpleNamespace(
        get_end_effector_pose_matrix=lambda: (np.eye(4), np.eye(4))
    )
    wrapper.intervened = False
    wrapper._tracker_lock = threading.Lock()
    wrapper.latest_tracker_right = _translated_pose(0.20)
    wrapper.latest_tracker_left = _translated_pose(-0.20)
    wrapper.tracker_start_world = None
    wrapper.ee_start = None
    wrapper._filtered_target_pose = {"right": None, "left": None}
    wrapper.config = BimanualTeleopConfig()
    wrapper.target_frame = "tracker"

    wrapper.set_intervened(True, start_delay=False)

    assert sleeps == []
    assert wrapper.intervened
    np.testing.assert_allclose(
        wrapper.tracker_start_world["right"],
        wrapper.latest_tracker_right,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        wrapper.tracker_start_world["left"],
        wrapper.latest_tracker_left,
        atol=1e-12,
    )


def test_dual_episode_baseline_uses_exact_bridge_ack_packet(monkeypatch) -> None:
    right = _translated_pose(0.37)
    left = _translated_pose(-0.42)
    packet = right[:3, :].tobytes() + left[:3, :].tobytes()
    monkeypatch.setattr(
        sim_teleop,
        "request_tracker_episode_baseline",
        lambda: packet,
    )

    wrapper = object.__new__(DualArmViveHandTeleopWrapper)
    wrapper._rokae_wuji_teleop = True
    wrapper._tracker_lock = threading.Lock()
    wrapper._tracker_history = sim_teleop.deque()
    wrapper._last_tracker_packet_time = 1.0
    wrapper.latest_tracker_right = _translated_pose(9.0)
    wrapper.latest_tracker_left = _translated_pose(-9.0)

    assert wrapper._establish_episode_tracker_baseline()

    np.testing.assert_allclose(wrapper.latest_tracker_right, right, atol=1e-12)
    np.testing.assert_allclose(wrapper.latest_tracker_left, left, atol=1e-12)
    assert wrapper._last_tracker_packet_time is not None
    assert len(wrapper._tracker_history) == 1


def test_single_z_rebase_maps_relocated_glove_to_current_sim_pose(monkeypatch) -> None:
    relocated_tracker = _translated_pose(1.25)
    packet = relocated_tracker[:3, :].tobytes()
    monkeypatch.setattr(
        sim_teleop,
        "request_tracker_episode_baseline",
        lambda: packet,
    )
    requests = iter((True, False))
    physical_pose = _translated_pose(0.42)
    wrapper = object.__new__(SingleArmViveHandTeleopWrapper)
    wrapper._tracker_lock = threading.RLock()
    wrapper.intervened = False
    wrapper.env = SimpleNamespace(
        get_end_effector_pose_matrix=lambda: physical_pose.copy(),
        consume_manual_tracker_rebase_request=lambda: next(requests),
    )
    wrapper.intervened = True
    wrapper._rokae_wuji_teleop = True
    wrapper._manual_tracker_rebase_requested = False
    wrapper._tracker_lock = threading.Lock()
    wrapper._tracker_history = sim_teleop.deque()
    wrapper._tracker_baseline_wait_started_at = 1.0
    wrapper._tracker_rebase_requested = True
    wrapper._tracker_rebase_hold_pending = True
    wrapper._last_commanded_tracker_pose = _translated_pose(0.10)
    wrapper._tracker_stationary_since = 1.0
    wrapper._tracker_stationary_drift_eligible = False
    wrapper.latest_tracker = _translated_pose(0.10)
    wrapper.tracker_start_world = _translated_pose(0.10)
    wrapper.ee_start = _translated_pose(0.20)
    wrapper._filtered_target_pose = _translated_pose(0.20)
    wrapper.hold_pose = _translated_pose(0.20)

    hold_action = wrapper._consume_manual_tracker_rebase_action()

    assert hold_action is not None
    np.testing.assert_allclose(hold_action[:3], physical_pose[:3, 3], atol=1e-12)
    np.testing.assert_allclose(
        wrapper.tracker_start_world,
        relocated_tracker,
        atol=1e-12,
    )
    np.testing.assert_allclose(wrapper.ee_start, physical_pose, atol=1e-12)
    assert wrapper._tracker_rebase_requested is False
    assert wrapper._tracker_rebase_hold_pending is False

    moved_tracker = relocated_tracker.copy()
    moved_tracker[0, 3] += 0.03
    target = sim_teleop._target_pose_from_bridge_world_pose(
        wrapper.ee_start,
        wrapper.tracker_start_world,
        moved_tracker,
        1.0,
    )
    np.testing.assert_allclose(
        target[0, 3], physical_pose[0, 3] + 0.03, atol=1e-12
    )


def test_dual_z_rebase_preserves_both_simulated_palms(monkeypatch) -> None:
    right_tracker = _translated_pose(1.10)
    left_tracker = _translated_pose(-1.30)
    packet = right_tracker[:3, :].tobytes() + left_tracker[:3, :].tobytes()
    monkeypatch.setattr(
        sim_teleop,
        "request_tracker_episode_baseline",
        lambda: packet,
    )
    right_palm = _translated_pose(0.31)
    left_palm = _translated_pose(-0.29)
    wrapper = object.__new__(DualArmViveHandTeleopWrapper)
    wrapper.env = SimpleNamespace(
        get_end_effector_pose_matrix=lambda: (
            right_palm.copy(),
            left_palm.copy(),
        ),
        consume_manual_tracker_rebase_request=lambda: True,
    )
    wrapper.intervened = True
    wrapper._rokae_wuji_teleop = True
    wrapper._manual_tracker_rebase_requested = False
    wrapper._tracker_lock = threading.Lock()
    wrapper._tracker_history = sim_teleop.deque()
    wrapper._tracker_baseline_wait_started_at = 1.0
    wrapper._tracker_rebase_requested = True
    wrapper._tracker_rebase_hold_pending = True
    wrapper.latest_tracker_right = _translated_pose(0.10)
    wrapper.latest_tracker_left = _translated_pose(-0.10)
    wrapper.tracker_start_world = {
        "right": wrapper.latest_tracker_right.copy(),
        "left": wrapper.latest_tracker_left.copy(),
    }
    wrapper.ee_start = {
        "right": _translated_pose(0.20),
        "left": _translated_pose(-0.20),
    }
    wrapper._filtered_target_pose = dict(wrapper.ee_start)
    wrapper.hold_pose = dict(wrapper.ee_start)

    hold_action = wrapper._consume_manual_tracker_rebase_action()

    assert hold_action is not None
    np.testing.assert_allclose(hold_action["right"][:3], right_palm[:3, 3])
    np.testing.assert_allclose(hold_action["left"][:3], left_palm[:3, 3])
    np.testing.assert_allclose(
        wrapper.tracker_start_world["right"], right_tracker, atol=1e-12
    )
    np.testing.assert_allclose(
        wrapper.tracker_start_world["left"], left_tracker, atol=1e-12
    )
    np.testing.assert_allclose(wrapper.ee_start["right"], right_palm, atol=1e-12)
    np.testing.assert_allclose(wrapper.ee_start["left"], left_palm, atol=1e-12)


def test_dual_wrapper_close_releases_bridge_episode_guard(monkeypatch) -> None:
    releases = []
    closed = []
    monkeypatch.setattr(
        sim_teleop,
        "release_tracker_episode_guard",
        lambda: releases.append(True) or True,
    )

    class IdleThread:
        @staticmethod
        def is_alive() -> bool:
            return False

    class FakeSocket:
        @staticmethod
        def close() -> None:
            closed.append("socket")

    wrapper = object.__new__(DualArmViveHandTeleopWrapper)
    wrapper._rokae_wuji_teleop = True
    wrapper._vive_thread = IdleThread()
    wrapper._hand_thread = IdleThread()
    wrapper._vive_socket = FakeSocket()
    wrapper._right_hand_socket = FakeSocket()
    wrapper._left_hand_socket = FakeSocket()
    wrapper.env = SimpleNamespace(close=lambda: closed.append("env"))

    wrapper.close()

    assert releases == [True]
    assert closed == ["socket", "socket", "socket", "env"]


def test_repeated_single_arm_resets_pause_then_rebase_active_teleoperation(
    monkeypatch,
) -> None:
    sleeps = []
    reset_calls = []
    monkeypatch.setattr(sim_teleop.time, "sleep", sleeps.append)
    wrapper = object.__new__(SingleArmViveHandTeleopWrapper)
    wrapper._tracker_lock = threading.RLock()
    wrapper.intervened = False
    wrapper.env = SimpleNamespace(
        reset=lambda **_kwargs: (reset_calls.append(True) or {"reset": True}, {}),
        get_end_effector_pose_matrix=lambda: np.eye(4),
    )
    wrapper.intervened = True
    wrapper.reset_trigger = False
    wrapper._tracker_lock = threading.Lock()
    wrapper._hand_lock = threading.Lock()
    wrapper.latest_tracker = _translated_pose(0.25)
    wrapper.tracker_start_world = wrapper.latest_tracker.copy()
    wrapper.ee_start = np.eye(4)
    wrapper._filtered_target_pose = np.eye(4)
    wrapper.latest_allegro_angles = np.zeros(20, dtype=np.float64)
    wrapper.config = SingleArmTeleopConfig()
    wrapper.target_frame = "tracker"
    wrapper._ensure_key_callback = lambda: None
    wrapper._capture_hold_pose = lambda: None
    wrapper._current_hand_action = lambda: np.zeros(20, dtype=np.float64)

    for index in range(6):
        wrapper.latest_tracker = _translated_pose(0.25 + index * 0.01)
        observation, _ = wrapper.reset()
        assert observation == {"reset": True}
        assert wrapper.intervened
        np.testing.assert_allclose(
            wrapper.tracker_start_world,
            wrapper.latest_tracker,
            atol=1e-12,
        )

    assert len(reset_calls) == 6
    assert sleeps == [sim_teleop.RESET_REBASE_DELAY_SECONDS] * 6


def test_dual_arm_reset_rebases_after_glove_reposition_pause(monkeypatch) -> None:
    wrapper = object.__new__(DualArmViveHandTeleopWrapper)
    wrapper.env = SimpleNamespace(
        reset=lambda **_kwargs: ({"reset": True}, {}),
        get_end_effector_pose_matrix=lambda: (np.eye(4), np.eye(4)),
    )
    wrapper.intervened = True
    wrapper.reset_trigger = False
    wrapper._tracker_lock = threading.Lock()
    wrapper._hand_lock = threading.Lock()
    wrapper.latest_tracker_right = _translated_pose(0.20)
    wrapper.latest_tracker_left = _translated_pose(-0.20)
    wrapper.tracker_start_world = {
        "right": wrapper.latest_tracker_right.copy(),
        "left": wrapper.latest_tracker_left.copy(),
    }
    wrapper.ee_start = {"right": np.eye(4), "left": np.eye(4)}
    wrapper._filtered_target_pose = {"right": np.eye(4), "left": np.eye(4)}
    wrapper.latest_allegro_angles_right = np.zeros(20, dtype=np.float64)
    wrapper.latest_allegro_angles_left = np.zeros(20, dtype=np.float64)
    wrapper._last_hand_packet_time_right = 10.0
    wrapper._last_hand_packet_time_left = 10.0
    wrapper.config = BimanualTeleopConfig()
    wrapper.target_frame = "tracker"
    wrapper._ensure_key_callback = lambda: None
    wrapper._capture_hold_pose = lambda: None
    wrapper._current_hand_action = lambda: (
        np.zeros(20, dtype=np.float64),
        np.zeros(20, dtype=np.float64),
    )

    def reposition_gloves_during_pause(seconds: float) -> None:
        assert seconds == sim_teleop.RESET_REBASE_DELAY_SECONDS
        wrapper.latest_tracker_right = _translated_pose(0.32)
        wrapper.latest_tracker_left = _translated_pose(-0.31)

    monkeypatch.setattr(sim_teleop.time, "sleep", reposition_gloves_during_pause)

    observation, _ = wrapper.reset()

    assert observation == {"reset": True}
    assert wrapper.intervened
    np.testing.assert_allclose(
        wrapper.tracker_start_world["right"],
        wrapper.latest_tracker_right,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        wrapper.tracker_start_world["left"],
        wrapper.latest_tracker_left,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        wrapper._commanded_hand_angles_right,
        np.zeros(20),
    )
    np.testing.assert_array_equal(
        wrapper._commanded_hand_angles_left,
        np.zeros(20),
    )
    assert wrapper._last_hand_packet_time_right is None
    assert wrapper._last_hand_packet_time_left is None


def test_non_b02_wuji_hand_is_not_conditioned() -> None:
    raw = _representative_closed_wuji_hand()
    env = SimpleNamespace(task_name="t36_scoop_marbles", hand_type="wuji1")

    corrected = sim_teleop._condition_b02_wuji_hand_action(env, raw)

    np.testing.assert_allclose(corrected, raw, atol=1e-12)


def test_single_arm_action_records_conditioned_b02_hand() -> None:
    raw_hand = _representative_closed_wuji_hand()
    wrapper = object.__new__(SingleArmViveHandTeleopWrapper)
    wrapper._tracker_lock = threading.RLock()
    wrapper.intervened = False
    wrapper.env = _b02_wuji_env()
    wrapper._vive_action = lambda: np.zeros(7, dtype=np.float64)
    wrapper._hold_pose_action = lambda: np.ones(7, dtype=np.float64)
    wrapper._hand_action = lambda _action: raw_hand.copy()

    action = SingleArmViveHandTeleopWrapper.action(wrapper, np.zeros(27, dtype=np.float64))

    np.testing.assert_allclose(action[:7], np.zeros(7, dtype=np.float64))
    np.testing.assert_allclose(
        action[7:],
        sim_teleop._condition_b02_wuji_hand_action(_b02_wuji_env(), raw_hand),
        atol=1e-12,
    )


def test_single_arm_action_appends_one_shot_task_operator_control() -> None:
    extras = iter((np.asarray([1.0]), np.asarray([0.0])))
    env = SimpleNamespace(
        task_name="c08_tongs_ice_cubes",
        hand_type="wuji1",
        consume_teleop_action_extras=lambda: next(extras),
    )
    wrapper = object.__new__(SingleArmViveHandTeleopWrapper)
    wrapper._tracker_lock = threading.RLock()
    wrapper.intervened = False
    wrapper.env = env
    wrapper._vive_action = lambda: np.zeros(7, dtype=np.float64)
    wrapper._hold_pose_action = lambda: np.ones(7, dtype=np.float64)
    wrapper._hand_action = lambda _action: np.zeros(20, dtype=np.float64)

    pressed = SingleArmViveHandTeleopWrapper.action(
        wrapper,
        np.zeros(28, dtype=np.float64),
    )
    released = SingleArmViveHandTeleopWrapper.action(
        wrapper,
        np.zeros(28, dtype=np.float64),
    )

    assert pressed.shape == (28,)
    assert pressed[-1] == 1.0
    assert released.shape == (28,)
    assert released[-1] == 0.0
