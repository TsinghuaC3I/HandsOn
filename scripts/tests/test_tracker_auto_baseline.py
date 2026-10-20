from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from handson.tasks import tracker_episode_protocol as protocol
from scripts.tests.test_manus_tracker_freshness import bridge


class ControlSocket:
    def __init__(self):
        self.requests = deque()
        self.sent = []

    def recvfrom(self, _size):
        if not self.requests:
            raise BlockingIOError
        return self.requests.popleft()

    def sendto(self, packet, target):
        self.sent.append((packet, target))


def pose(x=0.0):
    result = np.eye(4)
    result[0, 3] = x
    return result


def controller_for(monkeypatch, hands=("right", "left"), swap=False):
    clock = [100.0]
    monkeypatch.setattr(bridge.time, "monotonic", lambda: clock[0])
    controller = object.__new__(bridge.ManusWujiHandsOnBridge)
    controller.args = SimpleNamespace(send_tracker=True)
    controller.hands = hands
    controller.swap_arms = swap
    controller._calib_enabled = False
    controller._calib_phase = "done"
    controller._tracker_guard_armed = True
    controller._latest_tracker = {s: pose() for s in hands}
    controller._tracker_rebase_correction = {s: bridge.identity_tracker_pose_correction() for s in hands}
    controller._tracker_jump_reject_count = {s: 0 for s in hands}
    controller.tracker_control_socket = ControlSocket()
    controller.tracker_socket = ControlSocket()
    controller.tracker_target = ("127.0.0.1", 5012)
    controller._tracker_sent_count = 0
    controller.get_logger = lambda: SimpleNamespace(info=lambda _: None, warn=lambda _: None)
    matrices = {s: pose(.5 + .1 * i) for i, s in enumerate(hands)}
    controller._lookup_tracker_matrix = lambda side: matrices[side]
    return controller, clock, matrices


def request(controller, token=b"a" * 16, target=("127.0.0.1", 45678)):
    controller.tracker_control_socket.requests.append((protocol.TRACKER_AUTO_BASELINE_REQUEST + token, target))
    controller._consume_tracker_episode_baseline_requests()


@pytest.mark.parametrize("hands,swap", [(("right",), False), (("right", "left"), False), (("right", "left"), True)])
def test_bridge_acks_first_valid_pose_without_any_confirmation_delay(monkeypatch, hands, swap):
    controller, clock, matrices = controller_for(monkeypatch, hands, swap)
    request(controller)
    for s in hands:
        np.testing.assert_array_equal(controller._latest_tracker[s], pose())
    before = clock[0]
    controller._send_tracker()
    assert clock[0] == before
    assert len(controller.tracker_socket.sent) == 1
    packet = controller.tracker_socket.sent[0][0]
    order = tuple(reversed(hands)) if swap else hands
    assert packet == b"".join(matrices[s][:3, :].tobytes() for s in order)
    assert controller.tracker_control_socket.sent == [(protocol.TRACKER_AUTO_BASELINE_ACK + b"a" * 16 + packet, ("127.0.0.1", 45678))]
    assert controller._tracker_guard_armed
    assert not controller._pending_auto_baselines


@pytest.mark.parametrize("fault", ["missing", "nan", "rotation"])
def test_auto_request_waits_only_when_no_valid_source_pose_exists(monkeypatch, fault):
    controller, clock, matrices = controller_for(monkeypatch)
    request(controller)
    for i in range(9):
        clock[0] += .029
        matrices["right"] = pose(.5)
        if fault == "missing":
            matrices["right"] = None
        elif fault == "nan":
            matrices["right"][0, 3] = np.nan
        elif fault == "rotation":
            matrices["right"][0, 0] = 4
        # Renew the request while upstream has no usable pose.
        request(controller)
        controller._send_tracker()
        assert not controller.tracker_socket.sent
        assert not controller.tracker_control_socket.sent
        for s in controller.hands:
            np.testing.assert_array_equal(controller._latest_tracker[s], pose())
    matrices["right"] = pose(.6)
    before = clock[0]
    controller._send_tracker()
    assert clock[0] == before
    assert len(controller.tracker_socket.sent) == 1


def test_cancel_and_expiry_keep_original_reference(monkeypatch):
    controller, clock, _ = controller_for(monkeypatch)
    request(controller)
    controller.tracker_control_socket.requests.append((protocol.TRACKER_AUTO_BASELINE_CANCEL + b"b" * 16, ("127.0.0.1", 45678)))
    controller._consume_tracker_episode_baseline_requests()
    assert controller._pending_auto_baselines
    controller.tracker_control_socket.requests.append((protocol.TRACKER_AUTO_BASELINE_CANCEL + b"a" * 16, ("127.0.0.1", 45678)))
    controller._consume_tracker_episode_baseline_requests()
    assert not controller._pending_auto_baselines
    request(controller)
    clock[0] += 2.01
    controller._consume_tracker_episode_baseline_requests()
    assert not controller._pending_auto_baselines
    assert controller._tracker_guard_armed
    np.testing.assert_array_equal(controller._latest_tracker["right"], pose())


def test_auto_request_does_not_override_calibration_or_explicit_episode_reset(monkeypatch):
    controller, _, _ = controller_for(monkeypatch)
    controller._calib_enabled = True
    controller._calib_phase = "up"
    request(controller)
    assert not controller._pending_auto_baselines
    controller._calib_phase = "done"
    request(controller)
    controller._tracker_recovery_active = {s: False for s in controller.hands}
    controller.tracker_control_socket.requests.append((protocol.TRACKER_EPISODE_BASELINE_REQUEST, ("127.0.0.1", 45679)))
    controller._consume_tracker_episode_baseline_requests()
    assert not controller._pending_auto_baselines
    assert controller._tracker_episode_baseline_pending
    request(controller)
    assert not controller._pending_auto_baselines


@pytest.mark.parametrize("hands,swap", [(('right',), False), (('right', 'left'), False), (('right', 'left'), True)])
@pytest.mark.parametrize("fault", ["missing", "rotation"])
def test_episode_reset_resumes_on_first_valid_live_batch_after_input_returns(monkeypatch, hands, swap, fault):
    controller, clock, matrices = controller_for(monkeypatch, hands, swap)
    controller._tracker_recovery_active = {side: False for side in hands}
    controller._tracker_recovery_samples = {side: [] for side in hands}
    controller._tracker_missing_since = {side: None for side in hands}
    controller._last_tracker_missing_log = clock[0]
    controller._last_tracker_jump_log = {side: clock[0] for side in hands}
    controller.tracker_reconnect_stable_sec = .35
    controller.tracker_reconnect_max_position_spread_m = .008
    controller.tracker_reconnect_max_rotation_spread_deg = 2.0
    controller.tracker_preserve_baseline_on_reconnect = True
    controller.tracker_reacquire_reset_after_sec = .25
    controller._log_tracker_delta_debug = lambda _: None
    target = ("127.0.0.1", 45679)
    controller.tracker_control_socket.requests.append((protocol.TRACKER_EPISODE_BASELINE_REQUEST, target))

    matrices["right"] = None if fault == "missing" else pose(.5)
    if fault == "rotation":
        matrices["right"][0, 0] = 4.0
    controller._send_tracker()
    assert controller._tracker_episode_baseline_pending
    assert not controller.tracker_socket.sent
    assert not controller.tracker_control_socket.sent

    # The blocking episode request has already timed out; fresh, moving hands
    # must still release the reset, even though auto requests are held pending.
    clock[0] += 2.0
    matrices["right"] = pose(.8)
    controller._send_tracker()
    order = tuple(reversed(hands)) if swap else hands
    packet = b"".join(matrices[side][:3, :].tobytes() for side in order)
    assert controller.tracker_socket.sent == [(packet, controller.tracker_target)]
    assert controller.tracker_control_socket.sent == [(protocol.TRACKER_EPISODE_BASELINE_ACK + packet, target)]
    assert not controller._tracker_episode_baseline_pending
    assert not any(controller._tracker_recovery_active.values())
    assert controller._tracker_guard_armed


class ClientSocket:
    def __init__(self):
        self.received = deque()
        self.sent = []
        self.closed = False

    def connect(self, target):
        self.target = target

    def setblocking(self, value):
        assert value is False

    def recv(self, _size):
        if not self.received:
            raise BlockingIOError
        return self.received.popleft()

    def send(self, value):
        self.sent.append(value)

    def close(self):
        self.closed = True


def test_nonblocking_client_retries_and_rejects_other_freeze_or_old_ack(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(protocol.time, "monotonic", lambda: clock[0])
    sock = ClientSocket()
    monkeypatch.setattr(protocol.socket, "socket", lambda *_: sock)
    client = protocol.TrackerAutoBaselineClient()
    assert client.poll() is None
    assert len(sock.sent) == 1
    packet = pose(.5)[:3, :].tobytes()
    ack = protocol.TRACKER_AUTO_BASELINE_ACK + client._token + packet
    sock.received.append(protocol.TRACKER_AUTO_BASELINE_ACK + b"x" * 16 + packet)
    assert client.poll() is None
    assert len(sock.sent) == 1
    sock.received.append(ack)
    clock[0] += .30
    assert client.poll() is None  # Queued before a long simulation stall.
    assert len(sock.sent) == 2
    for _ in range(6):
        clock[0] += .1
        assert client.poll() is None
    assert len(sock.sent) == 3
    sock.received.append(ack)
    assert client.poll() == packet
    client.close()
    assert sock.closed
    assert sock.sent[-1] == protocol.TRACKER_AUTO_BASELINE_CANCEL + client._token
