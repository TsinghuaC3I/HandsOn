import gymnasium as gym
import numpy as np

from ..sim.envs.hand_specs import (
    HAND_POLICY_DIM,
    WUJI_NATIVE_HAND_DIM,
    policy_side_to_native_hand_values,
)


_ARM_POSE_DIM = 7


def _flat_action_dim(env) -> int | None:
    shape = getattr(getattr(env, "action_space", None), "shape", None)
    if shape is None or len(shape) != 1:
        return None
    return int(shape[0])


def _is_wuji(env) -> bool:
    return str(getattr(env, "hand_type", "")).lower() == "wuji1"


def _single_arm_hand_dim(env) -> int:
    action_dim = _flat_action_dim(env)
    if action_dim is not None and action_dim >= _ARM_POSE_DIM:
        hand_dim = action_dim - _ARM_POSE_DIM
        if hand_dim in (HAND_POLICY_DIM, WUJI_NATIVE_HAND_DIM):
            return hand_dim
    return WUJI_NATIVE_HAND_DIM if _is_wuji(env) else HAND_POLICY_DIM


def _dual_arm_hand_dim(env) -> int:
    action_dim = _flat_action_dim(env)
    if action_dim is not None:
        # IsaacLab bimanual environments expose the complete flat action,
        # while older MuJoCo environments expose one side.  Recognize both.
        if action_dim >= 2 * _ARM_POSE_DIM:
            hand_values = action_dim - 2 * _ARM_POSE_DIM
            if hand_values % 2 == 0:
                hand_dim = hand_values // 2
                if hand_dim in (HAND_POLICY_DIM, WUJI_NATIVE_HAND_DIM):
                    return hand_dim
        hand_dim = action_dim - _ARM_POSE_DIM
        if hand_dim == WUJI_NATIVE_HAND_DIM:
            return hand_dim

    # Some bimanual envs still expose a 23-d side action space but can parse
    # native Wuji 27-d side actions. Prefer native only for those migrated envs.
    if _is_wuji(env) and hasattr(env, "_parse_bimanual_hand_angles"):
        return WUJI_NATIVE_HAND_DIM
    return HAND_POLICY_DIM


class SingleArmPolicyWrapper(gym.ActionWrapper):
    """Adapt single-arm policy actions to the raw single-arm task format.

    Policy action layout:
      - shape: (23,) for Allegro or (27,) for Wuji
      - action[0:3]: end-effector target position, xyz
      - action[3:7]: end-effector target quaternion, wxyz
      - action[7:]: hand joint targets, 16 Allegro-policy or 20 Wuji-native values

    Raw single-arm tasks consume the same flat array.
    """

    def __init__(self, env):
        super().__init__(env)
        self._hand_dim = _single_arm_hand_dim(env)
        self._action_dim = _ARM_POSE_DIM + self._hand_dim
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._action_dim,),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self._action_dim,):
            raise ValueError(
                "Expected single-arm policy action shape "
                f"({self._action_dim},), got {action.shape}."
            )
        return action

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class DualArmPolicyWrapper(gym.ActionWrapper):
    """Adapt bimanual policy actions to the raw bimanual task format.

    Policy action layout:
      - shape: (46,) for Allegro-policy hands or (54,) for native Wuji hands
      - action[0:7]: right arm target pose, [x, y, z, qw, qx, qy, qz]
      - action[7:14]: left arm target pose, [x, y, z, qw, qx, qy, qz]
      - action[14:14+H]: right hand joint targets
      - action[14+H:14+2H]: left hand joint targets

    Raw bimanual tasks consume a dictionary:
      - "right": shape (7+H,), [right_pose_7, right_hand_joints]
      - "left": shape (7+H,), [left_pose_7, left_hand_joints]
    """

    def __init__(self, env):
        super().__init__(env)
        self._hand_dim = _dual_arm_hand_dim(env)
        self._expand_policy_wuji_hands = bool(
            self._hand_dim == HAND_POLICY_DIM
            and _is_wuji(env)
            and int(getattr(env, "_hand_action_dim", HAND_POLICY_DIM))
            == WUJI_NATIVE_HAND_DIM
        )
        self._action_dim = 2 * _ARM_POSE_DIM + 2 * self._hand_dim
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._action_dim,),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self._action_dim,):
            raise ValueError(
                "Expected bimanual policy action shape "
                f"({self._action_dim},), got {action.shape}."
            )

        right_pose = action[0:_ARM_POSE_DIM]
        left_pose = action[_ARM_POSE_DIM : 2 * _ARM_POSE_DIM]
        hand_start = 2 * _ARM_POSE_DIM
        right_hand = action[hand_start : hand_start + self._hand_dim]
        left_hand = action[
            hand_start + self._hand_dim : hand_start + 2 * self._hand_dim
        ]
        if self._expand_policy_wuji_hands:
            right_hand = policy_side_to_native_hand_values(
                "wuji1", right_hand, side="right", dual_arm=True
            )
            left_hand = policy_side_to_native_hand_values(
                "wuji1", left_hand, side="left", dual_arm=True
            )

        return {
            "right": np.concatenate([right_pose, right_hand], axis=0),
            "left": np.concatenate([left_pose, left_hand], axis=0),
        }

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
