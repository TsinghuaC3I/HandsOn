from __future__ import annotations

import numpy as np

from handson.sim.isaaclab import abc_collection_env as abc_env
from handson.sim.isaaclab.abc_collection_env import HandsOnABCIsaacLabEnv


def _interface_env() -> HandsOnABCIsaacLabEnv:
    return HandsOnABCIsaacLabEnv(
        "t36_scoop_marbles",
        launch_isaac=False,
        image_obs=False,
    )


def test_c02_direct_hand_marble_contact_fails_and_latches() -> None:
    env = _interface_env()
    env._success_counter = 7
    env._c02_hand_marble_contact_reading = lambda: (
        10.0 * abc_env._C02_DIRECT_HAND_CONTACT_FORCE,
        True,
    )

    assert env._compute_failure() is True
    assert env._success_counter == 0
    assert env._c02_failed_direct_marble_touch is True
    info = env._task_info()
    assert info["c02_failure_monitor_available"] is True
    assert info["c02_failed_direct_marble_touch"] is True


def test_c02_clear_contact_reading_does_not_fail() -> None:
    env = _interface_env()
    env._c02_hand_marble_contact_reading = lambda: (0.0, True)

    assert env._compute_failure() is False
    assert env._c02_failed_direct_marble_touch is False
    assert env._task_info()["c02_failure_monitor_available"] is True


def test_c02_missing_contact_channel_fails_closed() -> None:
    env = _interface_env()
    env._success_counter = 4
    env._c02_hand_marble_contact_reading = lambda: (0.0, False)

    assert env._compute_failure() is True
    assert env._success_counter == 0
    info = env._task_info()
    assert info["c02_failure_monitor_available"] is False
    assert info["c02_failed_direct_marble_touch"] is False


def test_c02_contact_reader_rejects_nonfinite_force_data() -> None:
    env = _interface_env()

    class FakeView:
        def get_contact_force_matrix(self, _dt: float) -> np.ndarray:
            return np.asarray([[[np.nan, 0.0, 0.0]]], dtype=np.float64)

    env._c02_marble_hand_contact_view = FakeView()
    assert env._c02_hand_marble_contact_reading() == (0.0, False)
