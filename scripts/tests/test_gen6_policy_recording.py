"""The saved GEN6 policy input must match the raw action the task executes."""
import gymnasium as gym
import numpy as np
import math

from handson.tasks.policy_wrappers import DualArmPolicyWrapper
from scripts import record_demos_zarr as recorder


def test_gen6_rate_contract_preserves_quantized_clock_measurement():
    control_dt = 1.0 / 30.0
    timestamps = (1.5 + np.arange(1018) * control_dt).astype(np.float32).astype(np.float64)
    original = timestamps.copy()
    measured = round(recorder.effective_sampling_hz(timestamps, fallback_hz=30.0), 9)
    assert not math.isclose(1.0 / measured, control_dt, rel_tol=0.0, abs_tol=1e-12)
    metadata = recorder._recorded_data_rate_metadata(
        "eval_gen05_set_mixer_knobs", measured,
        {"physx": {"physics_config": {"control_dt": control_dt}}},
        timestamp_source="physx_sim_time",
    )
    assert math.isclose(1.0 / metadata["data_fps"], control_dt, rel_tol=0.0, abs_tol=1e-12)
    assert metadata["observed_sim_clock_fps"] == measured
    np.testing.assert_array_equal(timestamps, original)
    assert recorder._recorded_data_rate_metadata(
        "t38_set_dual_knobs", measured, None,
        timestamp_source="physx_sim_time",
    ) == {"data_fps": measured}


def test_native_gen6_policy_storage_matches_executed_raw_action():
    class RawEnvironment(gym.Env):
        hand_type = "wuji1"
        action_space = gym.spaces.Box(-np.inf, np.inf, (54,), dtype=np.float32)
        observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)

        def step(self, action):
            self.executed = action
            return np.zeros(1, np.float32), 0., False, False, {}

    raw = RawEnvironment()
    wrapper = DualArmPolicyWrapper(raw)
    action = np.arange(54, dtype=np.float32) / 100
    wrapper.step(action)
    recorded = recorder._group_scripted_bimanual_action(
        action, hand_action_dim=20, task_label="GEN6"
    )
    for side in ("right", "left"):
        np.testing.assert_array_equal(recorded[side], raw.executed[side])
    assert recorder._action_storage_layout_for_action(recorded) == "right_pose_hand_then_left_pose_hand"
    np.testing.assert_array_equal(
        recorder._flatten_action_for_storage(recorded),
        np.concatenate((raw.executed["right"], raw.executed["left"])),
    )
