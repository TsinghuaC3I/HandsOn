from __future__ import annotations

import numpy as np
import pytest

from scripts import record_demos_zarr as recorder


def test_p003_scripted_action_is_stored_in_raw_per_side_layout() -> None:
    flat = np.arange(54, dtype=np.float64)
    grouped = recorder._group_p003_scripted_action(
        flat,
        hand_action_dim=20,
    )

    assert set(grouped) == {"right", "left"}
    assert grouped["right"].shape == (27,)
    assert grouped["left"].shape == (27,)
    assert np.array_equal(grouped["right"], np.r_[flat[:7], flat[14:34]])
    assert np.array_equal(grouped["left"], np.r_[flat[7:14], flat[34:54]])
    assert np.array_equal(
        recorder._flatten_action_for_storage(grouped),
        np.r_[flat[:7], flat[14:34], flat[7:14], flat[34:54]],
    )


def test_p003_scripted_action_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="expected"):
        recorder._group_p003_scripted_action(
            np.zeros(53, dtype=np.float64),
            hand_action_dim=20,
        )


def test_scripted_policy_flag_is_explicit_and_opt_in() -> None:
    assert recorder.FLAGS["scripted_policy"].default is False
