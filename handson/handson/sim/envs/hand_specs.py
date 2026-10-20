from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

try:
    import mujoco
except ImportError:  # Isaac Sim runtime does not need MuJoCo for hand specs.
    mujoco = None


HandType = Literal["allegro", "wuji1", "wuji2"]
SIMPLIFIED_POST_IK_ARM_TYPE = "rokae_simplified_post_ik"
ArmType = Literal["panda", "rokae", "rokae_simplified_post_ik"]

HAND_POLICY_DIM = 16
WUJI_NATIVE_HAND_DIM = 20
WUJI_HAND_TYPES = frozenset(("wuji1", "wuji2"))


def is_simplified_post_ik_arm_type(arm_type: str | None) -> bool:
    """Return whether collection should use detached hands plus offline IK."""

    value = str(arm_type or "").strip().lower().replace("-", "_")
    return value in {
        SIMPLIFIED_POST_IK_ARM_TYPE,
        "simplified_post_ik",
        "rokae_simplified",
    }


def is_rokae_arm_type(arm_type: str | None) -> bool:
    """Treat the simplified collection profile as a ROKAE asset family."""

    return normalize_arm_type(arm_type) == "rokae"


ROKAE_RIGHT_HOME = np.asarray(
    (-np.pi / 2.0, np.pi / 2.0, np.pi / 2.0, np.pi / 2.0, -np.pi / 2.0, 0.0, 0.0),
    dtype=np.float64,
)
# Elbow-up IK-equivalent of the old single-arm home; keeps the hand pose while
# giving the upper arm more table clearance.
ROKAE_SINGLE_RIGHT_HOME = np.asarray(
    (-1.22758, 0.88050, 0.67302, 1.80002, -0.81527, -0.80251, -0.33397),
    dtype=np.float64,
)
ROKAE_LEFT_HOME = np.asarray(
    (np.pi / 2.0, np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0, np.pi / 2.0, 0.0, 0.0),
    dtype=np.float64,
)

ALLEGRO_HOME = np.asarray(
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263, 0, 0, 0),
    dtype=np.float32,
)

_ALLEGRO_FINGER_RANGES = np.asarray(
    [
        [-0.47, 0.47],
        [-0.196, 1.61],
        [-0.174, 1.709],
        [-0.227, 1.618],
    ],
    dtype=np.float64,
)
_ALLEGRO_THUMB_RANGES = np.asarray(
    [
        [0.263, 1.396],
        [-0.105, 1.163],
        [-0.189, 1.644],
        [-0.162, 1.719],
    ],
    dtype=np.float64,
)
_ALLEGRO_RANGES = np.concatenate(
    [
        _ALLEGRO_FINGER_RANGES,
        _ALLEGRO_FINGER_RANGES,
        _ALLEGRO_FINGER_RANGES,
        _ALLEGRO_THUMB_RANGES,
    ],
    axis=0,
)

_WUJI_THUMB_RANGES = np.asarray(
    [
        [-1.187, 1.291],  # cmc_flex
        [-1.484, 0.698],  # cmc_abd
        [-1.047, 1.570],  # mcp
        [-1.047, 1.570],  # ip
    ],
    dtype=np.float64,
)
_WUJI1_THUMB_RANGES = np.asarray(
    [
        [0.0475, 1.6030],  # finger1_joint1
        [-0.1387, 0.9324],  # finger1_joint2
        [-0.4642, 1.5620],  # finger1_joint3
        [-0.4699, 1.5568],  # finger1_joint4
    ],
    dtype=np.float64,
)
_WUJI_FINGER_RANGES = np.asarray(
    [
        [-1.047, 1.570],  # mcp_flex
        [-0.698, 0.698],  # mcp_abd
        [-1.047, 2.094],  # pip
        [-1.047, 1.570],  # dip
    ],
    dtype=np.float64,
)
_WUJI1_FINGER_RANGES = np.asarray(
    [
        [-0.1644, 1.5580],  # joint1
        [-0.3700, 0.3700],  # joint2
        [-0.4777, 1.5490],  # joint3
        [-0.4777, 1.5750],  # joint4
    ],
    dtype=np.float64,
)
_WUJI_RANGES = np.concatenate(
    [
        _WUJI_THUMB_RANGES,
        _WUJI_FINGER_RANGES,
        _WUJI_FINGER_RANGES,
        _WUJI_FINGER_RANGES,
        _WUJI_FINGER_RANGES,
    ],
    axis=0,
)

_ALLEGRO_RIGHT_ORDER = ("ff", "mf", "rf", "th")
_ALLEGRO_LEFT_ORDER = ("rf", "mf", "ff", "th")
_ALLEGRO_ACTUATORS = {
    "ff": ("ffa0", "ffa1", "ffa2", "ffa3"),
    "mf": ("mfa0", "mfa1", "mfa2", "mfa3"),
    "rf": ("rfa0", "rfa1", "rfa2", "rfa3"),
    "th": ("tha0", "tha1", "tha2", "tha3"),
}
_ALLEGRO_JOINTS = {
    "ff": ("ffj0", "ffj1", "ffj2", "ffj3"),
    "mf": ("mfj0", "mfj1", "mfj2", "mfj3"),
    "rf": ("rfj0", "rfj1", "rfj2", "rfj3"),
    "th": ("thj0", "thj1", "thj2", "thj3"),
}

_WUJI_NATIVE_ORDER = (
    "thumb",
    "index_finger",
    "middle_finger",
    "ring_finger",
    "pinky",
)
_WUJI1_NATIVE_ORDER = ("finger1", "finger2", "finger3", "finger4", "finger5")
_WUJI_FINGER_TO_ACTUATOR = {
    "thumb": "finger1",
    "index_finger": "finger2",
    "middle_finger": "finger3",
    "ring_finger": "finger4",
    "pinky": "finger5",
}
_WUJI1_POLICY_TO_FINGER_RIGHT = {
    "ff": "finger2",
    "mf": "finger3",
    "rf": "finger4",
    "th": "finger1",
}
_WUJI1_POLICY_TO_FINGER_LEFT = {
    "rf": "finger4",
    "mf": "finger3",
    "ff": "finger2",
    "th": "finger1",
}
_WUJI_POLICY_TO_FINGER_RIGHT = {
    "ff": "index_finger",
    "mf": "middle_finger",
    "rf": "ring_finger",
    "th": "thumb",
}
_WUJI_POLICY_TO_FINGER_LEFT = {
    "rf": "ring_finger",
    "mf": "middle_finger",
    "ff": "index_finger",
    "th": "thumb",
}
_WUJI_PINKY_FOLLOW = np.asarray((0.8, 0.5, 0.8, 0.8), dtype=np.float64)


def normalize_hand_type(hand_type: str | None) -> HandType:
    value = (hand_type or "allegro").lower()
    aliases = {
        "allegro": "allegro",
        "wonik_allegro": "allegro",
        "wuji": "wuji1",
        "wuji1": "wuji1",
        "wuji_hand1": "wuji1",
        "wuji_hand_1": "wuji1",
        "wuji2": "wuji2",
        "wiji2": "wuji2",
        "wuji_hand2": "wuji2",
        "wuji_hand_2": "wuji2",
    }
    if value not in aliases:
        raise ValueError(
            f"Unsupported hand_type={hand_type!r}; expected 'allegro', 'wuji1', or 'wuji2'."
        )
    return aliases[value]  # type: ignore[return-value]


def hand_action_dim(hand_type: str | None) -> int:
    return WUJI_NATIVE_HAND_DIM if normalize_hand_type(hand_type) in WUJI_HAND_TYPES else HAND_POLICY_DIM


def is_wuji_hand_type(hand_type: str | None) -> bool:
    return normalize_hand_type(hand_type) in WUJI_HAND_TYPES


def resolve_hand_xml_path(xml_path: str | Path, hand_type: str | None) -> Path:
    hand = normalize_hand_type(hand_type)
    path = Path(xml_path)
    if hand == "allegro":
        return path

    arm_type = normalize_arm_type(os.environ.get("HANDSON_ARM_TYPE"))
    suffix = f"{arm_type}_{hand}" if arm_type != "panda" else hand
    wuji_path = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    if not wuji_path.exists():
        raise FileNotFoundError(
            f"{arm_type}/{hand} XML variant not found for {path.name}: {wuji_path}"
        )
    return wuji_path


def normalize_arm_type(arm_type: str | None) -> ArmType:
    value = (arm_type or "panda").strip().lower().replace("-", "_")
    aliases = {
        "panda": "panda",
        "franka": "panda",
        "franka_panda": "panda",
        "rokae": "rokae",
        "rokae_arm": "rokae",
        "ar5": "rokae",
        "ar5_5_07": "rokae",
        "ar5_5_08": "rokae",
        "luoshi": "rokae",
        # This is a collection/control profile rather than a different robot
        # asset.  MuJoCo and offline renderers still load the canonical ROKAE
        # XML while the live Isaac runtime may replace the articulated arms
        # with detached Wuji hands.
        SIMPLIFIED_POST_IK_ARM_TYPE: "rokae",
        "simplified_post_ik": "rokae",
        "rokae_simplified": "rokae",
    }
    if value not in aliases:
        raise ValueError(
            "Unsupported HANDSON_ARM_TYPE="
            f"{arm_type!r}; expected 'panda', 'rokae', or "
            f"'{SIMPLIFIED_POST_IK_ARM_TYPE}'."
        )
    return aliases[value]  # type: ignore[return-value]


def _require_mujoco():
    if mujoco is None:
        raise RuntimeError("MuJoCo is required for MuJoCo hand model lookup.")
    return mujoco


def _active_rokae_home(model, joint_ids: Sequence[int]) -> np.ndarray:
    mj = _require_mujoco()
    names = {
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, int(jid)) or ""
        for jid in joint_ids
    }
    if any(name.endswith("_left") for name in names):
        return ROKAE_LEFT_HOME.copy()
    if any(name.endswith("_right") for name in names):
        return ROKAE_RIGHT_HOME.copy()
    # Single-arm ROKAE tasks need a palm-down reset pose with extra table
    # clearance because those scenes randomize the table height independently.
    return ROKAE_SINGLE_RIGHT_HOME.copy()


def clipped_arm_home(model: mujoco.MjModel, joint_ids: Sequence[int], home: np.ndarray) -> np.ndarray:
    """Return a home qpos clipped to the active robot XML's joint ranges."""
    ids = np.asarray(joint_ids, dtype=int)
    values = np.asarray(home, dtype=np.float64).copy()
    ranges = model.jnt_range[ids]
    arm_type = normalize_arm_type(os.environ.get("HANDSON_ARM_TYPE"))
    if len(values) == len(ROKAE_RIGHT_HOME) and ranges.shape[0] == len(ROKAE_RIGHT_HOME):
        if arm_type == "rokae":
            values = _active_rokae_home(model, ids)
    return np.clip(values, ranges[:, 0], ranges[:, 1])


def _scale_range(values: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    u = (values - src[:, 0]) / (src[:, 1] - src[:, 0])
    mapped = dst[:, 0] + u * (dst[:, 1] - dst[:, 0])
    return np.clip(mapped, dst[:, 0], dst[:, 1])


def _allegro_order(side: str, dual_arm: bool) -> tuple[str, ...]:
    if dual_arm and side == "left":
        return _ALLEGRO_LEFT_ORDER
    return _ALLEGRO_RIGHT_ORDER


def _allegro_names(side: str, dual_arm: bool, kind: str) -> list[str]:
    groups = _ALLEGRO_ACTUATORS if kind == "actuator" else _ALLEGRO_JOINTS
    names: list[str] = []
    suffix = f"_{side}" if dual_arm else ""
    for finger in _allegro_order(side, dual_arm):
        names.extend(f"{name}{suffix}" for name in groups[finger])
    return names


def _allegro_sensor_names(side: str, dual_arm: bool) -> list[str]:
    sensor_side = side if dual_arm else "right"
    names: list[str] = []
    for finger in _allegro_order(side, dual_arm):
        names.extend(f"allegro_{sensor_side}/{joint}_pos" for joint in _ALLEGRO_JOINTS[finger])
    return names


def _wuji_side_prefix(side: str) -> str:
    return "l" if side == "left" else "r"


def _wuji_side_word(side: str) -> str:
    return "left" if side == "left" else "right"


def _wuji_native_order(hand_type: HandType) -> tuple[str, ...]:
    if hand_type == "wuji1":
        return _WUJI1_NATIVE_ORDER
    return _WUJI_NATIVE_ORDER


def _wuji_ranges(hand_type: HandType) -> tuple[np.ndarray, np.ndarray]:
    if hand_type == "wuji1":
        return _WUJI1_THUMB_RANGES, _WUJI1_FINGER_RANGES
    return _WUJI_THUMB_RANGES, _WUJI_FINGER_RANGES


def _wuji_finger_map(hand_type: HandType, side: str, dual_arm: bool) -> dict[str, str]:
    if hand_type == "wuji1":
        return (
            _WUJI1_POLICY_TO_FINGER_LEFT if dual_arm and side == "left"
            else _WUJI1_POLICY_TO_FINGER_RIGHT
        )
    return (
        _WUJI_POLICY_TO_FINGER_LEFT if dual_arm and side == "left"
        else _WUJI_POLICY_TO_FINGER_RIGHT
    )


def _wuji_joint_names(hand_type: HandType, side: str) -> list[str]:
    if hand_type == "wuji1":
        word = _wuji_side_word(side)
        return [
            f"{word}_finger{finger}_joint{joint}"
            for finger in range(1, 6)
            for joint in range(1, 5)
        ]

    prefix = _wuji_side_prefix(side)
    return [
        f"{prefix}_thumb_cmc_flex",
        f"{prefix}_thumb_cmc_abd",
        f"{prefix}_thumb_mcp",
        f"{prefix}_thumb_ip",
        f"{prefix}_index_finger_mcp_flex",
        f"{prefix}_index_finger_mcp_abd",
        f"{prefix}_index_finger_pip",
        f"{prefix}_index_finger_dip",
        f"{prefix}_middle_finger_mcp_flex",
        f"{prefix}_middle_finger_mcp_abd",
        f"{prefix}_middle_finger_pip",
        f"{prefix}_middle_finger_dip",
        f"{prefix}_ring_finger_mcp_flex",
        f"{prefix}_ring_finger_mcp_abd",
        f"{prefix}_ring_finger_pip",
        f"{prefix}_ring_finger_dip",
        f"{prefix}_pinky_mcp_flex",
        f"{prefix}_pinky_mcp_abd",
        f"{prefix}_pinky_pip",
        f"{prefix}_pinky_dip",
    ]


def native_hand_joint_names(
    hand_type: str | None,
    side: str,
) -> tuple[str, ...]:
    """Return the selected hand's native joint order for one robot side.

    Isaac Lab and MuJoCo must consume the same 20-value Wuji command packet.
    Keeping this public lookup beside the retargeting ranges prevents the
    simulator from silently retaining the Wuji Hand 1 joint names when Hand 2
    is selected.
    """

    hand = normalize_hand_type(hand_type)
    if side not in {"left", "right"}:
        raise ValueError(f"Unsupported hand side: {side!r}")
    if hand == "allegro":
        return tuple(_allegro_names(side, True, "joint"))
    return tuple(_wuji_joint_names(hand, side))


def _wuji_actuator_names(hand_type: HandType, side: str) -> list[str]:
    if hand_type == "wuji2":
        prefix = _wuji_side_prefix(side)
        return [
            f"{prefix}_{finger}J{joint}"
            for finger in ("TH", "FF", "MF", "RF", "LF")
            for joint in range(4)
        ]

    word = _wuji_side_word(side)
    names: list[str] = []
    for finger in _wuji_native_order(hand_type):
        actuator_finger = (
            finger if hand_type == "wuji1"
            else _WUJI_FINGER_TO_ACTUATOR[finger]
        )
        names.extend(f"{word}_{actuator_finger}_joint{i}_actuator" for i in range(1, 5))
    return names


def _wuji_sensor_names(hand_type: HandType, side: str) -> list[str]:
    word = _wuji_side_word(side)
    namespace = f"{hand_type}_{word}"
    return [f"{namespace}/{joint}_pos" for joint in _wuji_joint_names(hand_type, side)]


def policy_side_to_native_hand_values(
    hand_type: str | None,
    policy_values: Sequence[float],
    side: str = "right",
    dual_arm: bool = False,
) -> np.ndarray:
    hand = normalize_hand_type(hand_type)
    policy = np.asarray(policy_values, dtype=np.float64)
    if policy.shape[0] != HAND_POLICY_DIM:
        raise ValueError(
            f"Expected {HAND_POLICY_DIM} policy hand values, got {policy.shape[0]}."
        )
    if hand == "allegro":
        return policy.copy()

    order = _allegro_order(side, dual_arm)
    policy_groups = {
        finger: policy[i * 4 : (i + 1) * 4] for i, finger in enumerate(order)
    }
    finger_map = _wuji_finger_map(hand, side, dual_arm)
    thumb_ranges, finger_ranges = _wuji_ranges(hand)

    native_by_finger: dict[str, np.ndarray] = {}
    for policy_finger, wuji_finger in finger_map.items():
        src = policy_groups[policy_finger]
        if policy_finger == "th":
            native_by_finger[wuji_finger] = np.asarray(
                [
                    _scale_range(src[[1]], _ALLEGRO_THUMB_RANGES[[1]], thumb_ranges[[0]])[0],
                    _scale_range(src[[0]], _ALLEGRO_THUMB_RANGES[[0]], thumb_ranges[[1]])[0],
                    _scale_range(src[[2]], _ALLEGRO_THUMB_RANGES[[2]], thumb_ranges[[2]])[0],
                    _scale_range(src[[3]], _ALLEGRO_THUMB_RANGES[[3]], thumb_ranges[[3]])[0],
                ],
                dtype=np.float64,
            )
        else:
            native_by_finger[wuji_finger] = np.asarray(
                [
                    _scale_range(src[[1]], _ALLEGRO_FINGER_RANGES[[1]], finger_ranges[[0]])[0],
                    _scale_range(src[[0]], _ALLEGRO_FINGER_RANGES[[0]], finger_ranges[[1]])[0],
                    _scale_range(src[[2]], _ALLEGRO_FINGER_RANGES[[2]], finger_ranges[[2]])[0],
                    _scale_range(src[[3]], _ALLEGRO_FINGER_RANGES[[3]], finger_ranges[[3]])[0],
                ],
                dtype=np.float64,
            )

    ring_finger = finger_map["rf"]
    pinky_finger = "finger5" if hand == "wuji1" else "pinky"
    ring = native_by_finger[ring_finger]
    ring_home_src = ALLEGRO_HOME[8:12]
    ring_home = np.asarray(
        [
            _scale_range(ring_home_src[[1]], _ALLEGRO_FINGER_RANGES[[1]], finger_ranges[[0]])[0],
            _scale_range(ring_home_src[[0]], _ALLEGRO_FINGER_RANGES[[0]], finger_ranges[[1]])[0],
            _scale_range(ring_home_src[[2]], _ALLEGRO_FINGER_RANGES[[2]], finger_ranges[[2]])[0],
            _scale_range(ring_home_src[[3]], _ALLEGRO_FINGER_RANGES[[3]], finger_ranges[[3]])[0],
        ],
        dtype=np.float64,
    )
    native_by_finger[pinky_finger] = ring_home + _WUJI_PINKY_FOLLOW * (ring - ring_home)

    return np.concatenate(
        [native_by_finger[finger] for finger in _wuji_native_order(hand)]
    )


def policy_to_native_hand_values(
    hand_type: str | None,
    policy_values: Sequence[float],
    sides: Sequence[str] = ("right",),
    dual_arm: bool = False,
) -> np.ndarray:
    hand = normalize_hand_type(hand_type)
    values = np.asarray(policy_values, dtype=np.float64)
    side_tuple = tuple(sides)
    expected = HAND_POLICY_DIM * len(side_tuple)
    if values.shape[0] != expected:
        raise ValueError(f"Expected {expected} policy hand values, got {values.shape[0]}.")
    native_values = []
    for index, side in enumerate(side_tuple):
        start = index * HAND_POLICY_DIM
        native_values.append(
            policy_side_to_native_hand_values(
                hand,
                values[start : start + HAND_POLICY_DIM],
                side=side,
                dual_arm=dual_arm,
            )
        )
    return np.concatenate(native_values)


def _lookup_actuator_ids(model: mujoco.MjModel, names: Sequence[str]) -> np.ndarray:
    mj = _require_mujoco()
    ids = []
    for name in names:
        aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0 and not name.startswith("/"):
            aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"/{name}")
        if aid < 0:
            raise KeyError(f"Actuator {name!r} not found in MuJoCo model.")
        ids.append(aid)
    return np.asarray(ids, dtype=int)


def _lookup_qpos_ids(model: mujoco.MjModel, names: Sequence[str]) -> np.ndarray:
    mj = _require_mujoco()
    qpos_ids = []
    for name in names:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0 and not name.startswith("/"):
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"/{name}")
        if jid < 0:
            raise KeyError(f"Joint {name!r} not found in MuJoCo model.")
        qpos_ids.append(int(model.jnt_qposadr[jid]))
    return np.asarray(qpos_ids, dtype=int)


@dataclass(frozen=True)
class _SideHandles:
    side: str
    joint_names: tuple[str, ...]
    sensor_names: tuple[str, ...]
    ctrl_ids: np.ndarray
    qpos_ids: np.ndarray
    native_home: np.ndarray


class HandController:
    """Translate between policy-hand coordinates and the selected native hand."""

    def __init__(
        self,
        model: mujoco.MjModel,
        hand_type: str | None = "allegro",
        sides: Sequence[str] = ("right",),
        dual_arm: bool = False,
    ):
        self.hand_type = normalize_hand_type(hand_type)
        self.sides = tuple(sides)
        self.dual_arm = dual_arm
        self._side_handles = {
            side: self._build_side(model, side) for side in self.sides
        }

        self.ctrl_ids = np.concatenate([self._side_handles[s].ctrl_ids for s in self.sides])
        self.qpos_ids = np.concatenate([self._side_handles[s].qpos_ids for s in self.sides])
        self.joint_names = [
            name for side in self.sides for name in self._side_handles[side].joint_names
        ]
        self.side_qpos_ids = {
            side: self._side_handles[side].qpos_ids for side in self.sides
        }
        self.side_ctrl_ids = {
            side: self._side_handles[side].ctrl_ids for side in self.sides
        }
        self.home_qpos = np.concatenate(
            [self._side_handles[s].native_home for s in self.sides]
        )

    def _build_side(self, model: mujoco.MjModel, side: str) -> _SideHandles:
        if self.hand_type == "allegro":
            joint_names = _allegro_names(side, self.dual_arm, "joint")
            actuator_names = _allegro_names(side, self.dual_arm, "actuator")
            sensor_names = _allegro_sensor_names(side, self.dual_arm)
            native_home = ALLEGRO_HOME.astype(np.float64)
        else:
            joint_names = _wuji_joint_names(self.hand_type, side)
            actuator_names = _wuji_actuator_names(self.hand_type, side)
            sensor_names = _wuji_sensor_names(self.hand_type, side)
            native_home = self._policy_side_to_native(ALLEGRO_HOME, side)

        return _SideHandles(
            side=side,
            joint_names=tuple(joint_names),
            sensor_names=tuple(sensor_names),
            ctrl_ids=_lookup_actuator_ids(model, actuator_names),
            qpos_ids=_lookup_qpos_ids(model, joint_names),
            native_home=native_home,
        )

    def policy_to_native(self, policy_values: Sequence[float]) -> np.ndarray:
        values = np.asarray(policy_values, dtype=np.float64)
        expected = HAND_POLICY_DIM * len(self.sides)
        if values.shape[0] != expected:
            raise ValueError(
                f"Expected {expected} policy hand values, got {values.shape[0]}."
            )

        native_values = []
        for index, side in enumerate(self.sides):
            start = index * HAND_POLICY_DIM
            side_policy = values[start : start + HAND_POLICY_DIM]
            native_values.append(self._policy_side_to_native(side_policy, side))
        return np.concatenate(native_values)

    def _policy_side_to_native(self, policy_values: Sequence[float], side: str) -> np.ndarray:
        return policy_side_to_native_hand_values(
            self.hand_type,
            policy_values,
            side=side,
            dual_arm=self.dual_arm,
        )

    def native_to_policy(self, native_values: Sequence[float]) -> np.ndarray:
        values = np.asarray(native_values, dtype=np.float64)
        expected = len(self.qpos_ids)
        if values.shape[0] != expected:
            raise ValueError(f"Expected {expected} native hand values, got {values.shape[0]}.")

        policy_values = []
        offset = 0
        for side in self.sides:
            native_dim = len(self._side_handles[side].qpos_ids)
            side_native = values[offset : offset + native_dim]
            policy_values.append(self._native_side_to_policy(side_native, side))
            offset += native_dim
        return np.concatenate(policy_values).astype(np.float32)

    def _native_side_to_policy(self, native_values: Sequence[float], side: str) -> np.ndarray:
        native = np.asarray(native_values, dtype=np.float64)
        if self.hand_type == "allegro":
            return native.astype(np.float32)

        native_groups = {
            finger: native[i * 4 : (i + 1) * 4]
            for i, finger in enumerate(_wuji_native_order(self.hand_type))
        }
        finger_map = _wuji_finger_map(self.hand_type, side, self.dual_arm)
        thumb_ranges, finger_ranges = _wuji_ranges(self.hand_type)

        policy_by_finger: dict[str, np.ndarray] = {}
        for policy_finger, wuji_finger in finger_map.items():
            src = native_groups[wuji_finger]
            if policy_finger == "th":
                policy_by_finger[policy_finger] = np.asarray(
                    [
                        _scale_range(src[[1]], thumb_ranges[[1]], _ALLEGRO_THUMB_RANGES[[0]])[0],
                        _scale_range(src[[0]], thumb_ranges[[0]], _ALLEGRO_THUMB_RANGES[[1]])[0],
                        _scale_range(src[[2]], thumb_ranges[[2]], _ALLEGRO_THUMB_RANGES[[2]])[0],
                        _scale_range(src[[3]], thumb_ranges[[3]], _ALLEGRO_THUMB_RANGES[[3]])[0],
                    ],
                    dtype=np.float64,
                )
            else:
                policy_by_finger[policy_finger] = np.asarray(
                    [
                        _scale_range(src[[1]], finger_ranges[[1]], _ALLEGRO_FINGER_RANGES[[0]])[0],
                        _scale_range(src[[0]], finger_ranges[[0]], _ALLEGRO_FINGER_RANGES[[1]])[0],
                        _scale_range(src[[2]], finger_ranges[[2]], _ALLEGRO_FINGER_RANGES[[2]])[0],
                        _scale_range(src[[3]], finger_ranges[[3]], _ALLEGRO_FINGER_RANGES[[3]])[0],
                    ],
                    dtype=np.float64,
                )

        order = _allegro_order(side, self.dual_arm)
        return np.concatenate([policy_by_finger[finger] for finger in order]).astype(np.float32)

    def get_policy_qpos(self, data: mujoco.MjData) -> np.ndarray:
        native_qpos = data.qpos[self.qpos_ids]
        return self.native_to_policy(native_qpos)

    def set_home_qpos(
        self,
        data: mujoco.MjData,
        policy_pose: Sequence[float] | None = None,
    ) -> None:
        native_pose = (
            self.home_qpos
            if policy_pose is None
            else self.policy_to_native(policy_pose)
        )
        data.qpos[self.qpos_ids] = native_pose
        self.set_native_ctrl(data, native_pose)

    def set_ctrl(self, data: mujoco.MjData, policy_values: Sequence[float]) -> None:
        self.set_native_ctrl(data, self.policy_to_native(policy_values))

    def set_action_ctrl(self, data: mujoco.MjData, values: Sequence[float]) -> None:
        action_values = np.asarray(values, dtype=np.float64)
        if action_values.shape[0] == len(self.qpos_ids):
            self.set_native_ctrl(data, action_values)
        else:
            self.set_ctrl(data, action_values)

    def set_native_ctrl(self, data: mujoco.MjData, native_values: Sequence[float]) -> None:
        native = np.asarray(native_values, dtype=np.float64)
        valid = self.ctrl_ids >= 0
        data.ctrl[self.ctrl_ids[valid].astype(int)] = native[valid]
