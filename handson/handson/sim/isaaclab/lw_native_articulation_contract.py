"""Pure physical-success contracts for the LW native-USD task family.

This module intentionally imports neither Isaac Sim nor USD.  The live runtime
feeds it ordered PhysX articulation and assigned-contact truth at 120 Hz.  The
same state machines can therefore be audited in ordinary CPU-only tests.

The contracts are observers only: they never return actions, write joint
targets, or otherwise move an asset.  Most tasks require measured assigned-hand
contact, signed physical motion, target hold, release, and a post-release
stability window; task-specific flags can intentionally shorten that sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


PHYSICS_HZ = 120
NOMINAL_DT_S = 1.0 / PHYSICS_HZ

CONTACT_ON_N = 0.15
CONTACT_OFF_N = 0.05
TARGET_HOLD_S = 0.25
RELEASE_HOLD_S = 0.25
STABLE_HOLD_S = 0.50

# G08 uses a reset-positioned native fixed base.  The 20%-smaller composed box
# leaves enough tabletop and dual-hand reach margin for a visibly broad
# continuous XY/yaw distribution without changing roll/pitch or pre-opening
# the lid.
G08_ROOT_NOMINAL_X_M = -0.280
G08_ROOT_NOMINAL_Y_M = 0.0
G08_ROOT_RANDOM_X_HALF_RANGE_M = 0.120
G08_ROOT_RANDOM_Y_HALF_RANGE_M = 0.120
G08_ROOT_RANDOM_YAW_HALF_RANGE_DEG = 30.0

# The supplied box USD has a negative 90-degree hinge excursion, and the
# operator task intentionally uses that authored stop.  Keep the values in
# degrees here because USD revolute-joint limits are authored in degrees; the
# pure contract converts them to radians below.  The target window includes
# the physical stop so a live readback of -90 degrees is a valid completion
# state rather than an over-limit failure.
G08_LID_HARD_STOP_DEG = 90.0
G08_LID_TARGET_DEG = 80.0
G08_LID_OPEN_THRESHOLD_DEG = 75.0
G08_LID_TARGET_ENTER_DEG = (-90.0, -75.0)
G08_LID_TARGET_EXIT_DEG = (-90.0, -73.0)

# G10 is a free tabletop asset. These bounds keep randomized resets reachable
# while allowing the ruler to move during manipulation.
G10_ROOT_NOMINAL_X_M = -0.300
G10_ROOT_NOMINAL_Y_M = 0.0
G10_ROOT_RANDOM_X_HALF_RANGE_M = 0.060
G10_ROOT_RANDOM_Y_HALF_RANGE_M = 0.120
G10_ROOT_RANDOM_YAW_HALF_RANGE_DEG = 25.0

MAX_DOF_COUNT = 19
CONDITION_LABELS = (
    "truth_available",
    "left_contact_seen",
    "right_contact_seen",
    "left_role_qualified",
    "right_role_qualified",
    "synchronized_motion",
    "synchronized_target",
    "target_hold_complete",
    "release_complete",
    "success",
)
CONDITION_VECTOR_SIZE = len(CONDITION_LABELS)

# Four padded DOF vectors followed by the scalar block documented below.
_SCALAR_OBSERVATION_LABELS = (
    "dof_count",
    "left_contact_active",
    "right_contact_active",
    "left_force_n",
    "right_force_n",
    "left_contact_seen",
    "right_contact_seen",
    "left_role_qualified",
    "right_role_qualified",
    "left_target_seen",
    "right_target_seen",
    "dof_truth_available",
    "contact_truth_available",
    "root_translation_error_m",
    "root_rotation_error_deg",
    "start_skew_s",
    "target_skew_s",
    "overlap_s",
    "target_hold_fraction",
    "release_fraction",
    "stable_fraction",
    "stage",
    "success",
    "failed",
)
OBSERVATION_VECTOR_SIZE = 4 * MAX_DOF_COUNT + len(
    _SCALAR_OBSERVATION_LABELS
)

# Repeated additions of 1 / 120 can finish a few ulps below an authored time.
# This epsilon is far smaller than one physics substep, so it cannot make an
# N - 1 frame window pass as an N frame window.
_TIME_EPSILON_S = 1.0e-9


def _time_at_least(value_s: float, threshold_s: float) -> bool:
    return float(value_s) >= float(threshold_s) - _TIME_EPSILON_S


class LWTaskStage(IntEnum):
    WAIT_CONTACT = 0
    WAIT_MOTION = 1
    MOVING = 2
    TARGET_HOLD = 3
    WAIT_RELEASE = 4
    STABILIZE = 5
    COMPLETE = 6
    FAILED = 7


@dataclass(frozen=True)
class LWNativeTaskSpec:
    """Immutable physical contract and ordered articulation schema."""

    task_id: str
    task_name: str
    dof_names: tuple[str, ...]
    lower_limits_rad: tuple[float, ...]
    upper_limits_rad: tuple[float, ...]
    authored_q0_rad: tuple[float, ...]
    target_q_rad: tuple[float, ...]
    target_enter_lower_rad: tuple[float, ...]
    target_enter_upper_rad: tuple[float, ...]
    target_exit_lower_rad: tuple[float, ...]
    target_exit_upper_rad: tuple[float, ...]
    direction_by_dof: tuple[int, ...]
    left_dof_indices: tuple[int, ...]
    right_dof_indices: tuple[int, ...]
    side_target_min_count: tuple[int, int]
    global_target_min_count: int
    side_progress_min: tuple[float, float]
    motion_displacement_rad: tuple[float, float]
    wrong_direction_tolerance_rad: tuple[float, float]
    motion_speed_rad_s: float
    target_speed_rad_s: float
    stable_speed_rad_s: float
    start_skew_limit_s: float | None
    target_skew_limit_s: float | None
    overlap_required_s: float
    root_translation_limit_m: float
    root_rotation_limit_deg: float
    limit_failure_margin_rad: float
    # Whether ordinary task progress is constrained to the reset root pose.
    # Native LW tasks use this integrity check even when a task's success
    # predicate is deliberately reduced to a geometric end state.
    root_pose_constraint: bool = True
    # Fixed-root PhysX constraints can report a single overshoot frame when a
    # thin link is first pinched.  Keep the hard envelope, but require the
    # measured error to persist for this many seconds before declaring a
    # terminal drift.  A zero value preserves the historical fail-on-first-
    # frame behavior for tasks that do not need a debounce.
    root_translation_drift_hold_s: float = 0.0
    root_rotation_drift_hold_s: float = 0.0
    recoverable_root_drift: bool = False
    root_escape_translation_limit_m: float | None = None
    root_escape_rotation_limit_deg: float | None = None
    root_escape_hold_s: float = 0.0
    locked_dof_index: int | None = None
    locked_dof_tolerance_rad: float = 0.0
    left_is_stabilizer: bool = False
    release_left: bool = True
    release_right: bool = True
    keep_left_contact_after_target: bool = False
    # A task-local physics-time debounce for a left stabilizer witness.
    # Zero preserves immediate-loss behavior for tasks that do not use it.
    stabilizer_contact_loss_debounce_s: float = 0.0
    # A task may pause (but never credit) target/release hold time during a
    # short hard-stop speed spike.
    target_speed_loss_debounce_s: float = 0.0
    # When enabled, the required release hold is the terminal success event.
    # The release debounce prevents a single noisy sample from latching
    # success on tasks that use this mode.
    success_on_release: bool = False
    requires_support_contact: bool = False
    # Some tasks use the articulation only as a geometric end-state check.
    # Hand contact, signed motion, synchronization, release, and velocity
    # holds are intentionally not required when this flag is enabled.
    success_on_position_target: bool = False
    # G10 additionally treats +180 and -180 degrees as the same straight
    # physical hinge orientation when evaluating its position target.
    straightness_only: bool = False

    def __post_init__(self) -> None:
        count = len(self.dof_names)
        vector_fields = (
            self.lower_limits_rad,
            self.upper_limits_rad,
            self.authored_q0_rad,
            self.target_q_rad,
            self.target_enter_lower_rad,
            self.target_enter_upper_rad,
            self.target_exit_lower_rad,
            self.target_exit_upper_rad,
            self.direction_by_dof,
        )
        if not count or count > MAX_DOF_COUNT:
            raise ValueError(f"Invalid DOF count for {self.task_name}: {count}")
        if any(len(values) != count for values in vector_fields):
            raise ValueError(f"Inconsistent DOF schema for {self.task_name}")
        if any(direction not in (-1, 0, 1) for direction in self.direction_by_dof):
            raise ValueError(f"Invalid signed direction for {self.task_name}")
        all_side_indices = self.left_dof_indices + self.right_dof_indices
        if any(index < 0 or index >= count for index in all_side_indices):
            raise ValueError(f"Invalid side DOF index for {self.task_name}")
        if self.locked_dof_index is not None:
            if not 0 <= self.locked_dof_index < count:
                raise ValueError(f"Invalid locked DOF index for {self.task_name}")
            if self.direction_by_dof[self.locked_dof_index] != 0:
                raise ValueError(
                    f"Locked DOF must have zero direction: {self.task_name}"
                )
        if self.root_translation_limit_m <= 0.0:
            raise ValueError(f"Invalid root translation limit: {self.task_name}")
        if self.root_rotation_limit_deg <= 0.0:
            raise ValueError(f"Invalid root rotation limit: {self.task_name}")
        for value, label in (
            (self.root_translation_drift_hold_s, "translation"),
            (self.root_rotation_drift_hold_s, "rotation"),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(
                    f"Invalid root {label} drift hold: {self.task_name}"
                )
        if (
            not math.isfinite(float(self.stabilizer_contact_loss_debounce_s))
            or self.stabilizer_contact_loss_debounce_s < 0.0
        ):
            raise ValueError(
                f"Invalid stabilizer contact loss debounce: {self.task_name}"
            )
        if (
            not math.isfinite(float(self.target_speed_loss_debounce_s))
            or self.target_speed_loss_debounce_s < 0.0
        ):
            raise ValueError(
                f"Invalid target speed loss debounce: {self.task_name}"
            )
        if self.recoverable_root_drift:
            if (
                self.root_escape_translation_limit_m is None
                or self.root_escape_translation_limit_m
                <= self.root_translation_limit_m
            ):
                raise ValueError(
                    f"Recoverable root translation requires a larger escape "
                    f"limit: {self.task_name}"
                )
            if (
                self.root_escape_rotation_limit_deg is None
                or self.root_escape_rotation_limit_deg
                <= self.root_rotation_limit_deg
            ):
                raise ValueError(
                    f"Recoverable root rotation requires a larger escape "
                    f"limit: {self.task_name}"
                )
            if self.root_escape_hold_s <= 0.0:
                raise ValueError(
                    f"Recoverable root escape requires a positive hold: "
                    f"{self.task_name}"
                )

    @property
    def dof_count(self) -> int:
        return len(self.dof_names)


def _deg(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(math.radians(float(value)) for value in values)


_FAN_DOF_NAMES = tuple(f"Prop{index:03d}_joint" for index in range(1, 20))
_FAN_DIRECTIONS = (1,) * 9 + (0,) + (-1,) * 9
_FAN_TARGET_DEG = (8.0,) * 9 + (0.0,) + (-8.0,) * 9
_FAN_ENTER_LOWER_DEG = (6.5,) * 9 + (0.0,) + (-8.0,) * 9
_FAN_ENTER_UPPER_DEG = (8.0,) * 9 + (0.0,) + (-6.5,) * 9
_FAN_EXIT_LOWER_DEG = (6.0,) * 9 + (-0.1,) + (-8.0,) * 9
_FAN_EXIT_UPPER_DEG = (8.0,) * 9 + (0.1,) + (-6.0,) * 9

G05_SPEC = LWNativeTaskSpec(
    task_id="G05",
    task_name="g05_folding_fan_bimanual_open",
    dof_names=_FAN_DOF_NAMES,
    lower_limits_rad=_deg((0.0,) * 10 + (-8.0,) * 9),
    upper_limits_rad=_deg((8.0,) * 9 + (0.0,) * 10),
    # A fully closed fan leaves the two outer grip ribs only about 20 mm
    # apart.  Start from a coherent 2 degree opening per moving joint so the
    # two assigned Wuji hands have distinct, collision-clear grasp targets;
    # this remains far outside the 6.5 degree target window.
    authored_q0_rad=_deg((2.0,) * 9 + (0.0,) + (-2.0,) * 9),
    target_q_rad=_deg(_FAN_TARGET_DEG),
    target_enter_lower_rad=_deg(_FAN_ENTER_LOWER_DEG),
    target_enter_upper_rad=_deg(_FAN_ENTER_UPPER_DEG),
    target_exit_lower_rad=_deg(_FAN_EXIT_LOWER_DEG),
    target_exit_upper_rad=_deg(_FAN_EXIT_UPPER_DEG),
    direction_by_dof=_FAN_DIRECTIONS,
    left_dof_indices=tuple(range(0, 9)),
    right_dof_indices=tuple(range(10, 19)),
    side_target_min_count=(8, 8),
    global_target_min_count=16,
    side_progress_min=(0.85, 0.85),
    motion_displacement_rad=_deg((1.0, 1.0)),
    wrong_direction_tolerance_rad=_deg((5.0, 5.0)),
    motion_speed_rad_s=0.02,
    target_speed_rad_s=0.08,
    stable_speed_rad_s=0.08,
    start_skew_limit_s=1.00,
    target_skew_limit_s=0.75,
    overlap_required_s=0.30,
    root_translation_limit_m=0.002,
    root_rotation_limit_deg=1.0,
    limit_failure_margin_rad=math.radians(0.5),
    locked_dof_index=9,
    # Prop010 is authored as the fixed centre rib, but contact impulses from
    # opening both fan halves can leave a small PhysX constraint error.  Match
    # the normal joint-limit guard's 0.5 degree solver margin so expected
    # bimanual opening contact does not terminate the attempt.
    locked_dof_tolerance_rad=math.radians(0.5),
)

G08_SPEC = LWNativeTaskSpec(
    task_id="18",
    task_name="t18_open_micropipette_box",
    dof_names=("Lid001_joint",),
    lower_limits_rad=_deg((-G08_LID_HARD_STOP_DEG,)),
    upper_limits_rad=_deg((0.0,)),
    authored_q0_rad=_deg((0.0,)),
    target_q_rad=_deg((-G08_LID_TARGET_DEG,)),
    target_enter_lower_rad=_deg((G08_LID_TARGET_ENTER_DEG[0],)),
    target_enter_upper_rad=_deg((G08_LID_TARGET_ENTER_DEG[1],)),
    target_exit_lower_rad=_deg((G08_LID_TARGET_EXIT_DEG[0],)),
    target_exit_upper_rad=_deg((G08_LID_TARGET_EXIT_DEG[1],)),
    direction_by_dof=(-1,),
    left_dof_indices=(),
    right_dof_indices=(0,),
    side_target_min_count=(0, 1),
    global_target_min_count=1,
    side_progress_min=(0.0, 0.85),
    motion_displacement_rad=(0.0, math.radians(5.0)),
    wrong_direction_tolerance_rad=(0.0, math.radians(3.0)),
    motion_speed_rad_s=0.08,
    target_speed_rad_s=0.08,
    stable_speed_rad_s=0.05,
    start_skew_limit_s=None,
    target_skew_limit_s=None,
    overlap_required_s=0.0,
    root_translation_limit_m=0.030,
    root_rotation_limit_deg=8.0,
    # A quarter-degree solver/readback margin tolerates tiny quantization at
    # the authored -90-degree physical stop while still rejecting a real
    # excursion beyond the source limit.
    limit_failure_margin_rad=math.radians(0.25),
    # The live scene reset-positions and then fixes the base to the world.  The
    # root envelope remains an independent constraint-integrity audit.
    recoverable_root_drift=True,
    root_escape_translation_limit_m=0.200,
    root_escape_rotation_limit_deg=60.0,
    root_escape_hold_s=0.50,
    requires_support_contact=False,
    success_on_position_target=True,
)

G10_SPEC = LWNativeTaskSpec(
    task_id="49",
    task_name="t49_unfold_ruler",
    dof_names=("Prop001_joint", "Prop002_joint"),
    lower_limits_rad=_deg((-180.0, -180.0)),
    upper_limits_rad=_deg((180.0, 180.0)),
    # Start half-open so both outer links are separately graspable before
    # either is straightened. The composed root is free on the tabletop.
    authored_q0_rad=_deg((-90.0, 90.0)),
    target_q_rad=_deg((-180.0, 180.0)),
    target_enter_lower_rad=_deg((-180.0, 175.0)),
    target_enter_upper_rad=_deg((-175.0, 180.0)),
    target_exit_lower_rad=_deg((-180.0, 173.0)),
    target_exit_upper_rad=_deg((-173.0, 180.0)),
    direction_by_dof=(-1, 1),
    # Left/right indices retain source body ownership; straightness-only
    # success does not require a particular hand assignment.
    left_dof_indices=(0,),
    right_dof_indices=(1,),
    side_target_min_count=(1, 1),
    global_target_min_count=2,
    side_progress_min=(0.95, 0.95),
    motion_displacement_rad=_deg((5.0, 5.0)),
    wrong_direction_tolerance_rad=_deg((3.0, 3.0)),
    motion_speed_rad_s=0.12,
    target_speed_rad_s=0.08,
    stable_speed_rad_s=0.05,
    start_skew_limit_s=None,
    target_skew_limit_s=None,
    overlap_required_s=0.0,
    root_translation_limit_m=0.002,
    root_rotation_limit_deg=1.0,
    limit_failure_margin_rad=math.radians(1.0),
    # G10 has no root-pose success constraint. The broad escape guard is only
    # a safety bound against losing the asset entirely during manipulation.
    root_translation_drift_hold_s=0.0,
    root_rotation_drift_hold_s=0.0,
    root_pose_constraint=False,
    recoverable_root_drift=True,
    root_escape_translation_limit_m=0.750,
    root_escape_rotation_limit_deg=270.0,
    root_escape_hold_s=0.50,
    success_on_position_target=True,
    straightness_only=True,
)

# Asset-transfer task: preserve every G08 success/failure parameter, changing
# only canonical identity and the new asset's physical DOF name.
GEN3_SPEC = replace(
    G08_SPEC,
    task_id="EVAL-Gen02",
    task_name="eval_gen02_open_trash_bin",
    dof_names=("lid_hinge",),
)

_SPECS = (G05_SPEC, G08_SPEC, G10_SPEC, GEN3_SPEC)
TASK_SPECS = MappingProxyType({spec.task_name: spec for spec in _SPECS})
TASK_SPECS_BY_ID = MappingProxyType({spec.task_id: spec for spec in _SPECS})


def get_task_spec(task: str | LWNativeTaskSpec) -> LWNativeTaskSpec:
    if isinstance(task, LWNativeTaskSpec):
        return task
    key = str(task)
    if key in TASK_SPECS_BY_ID:
        return TASK_SPECS_BY_ID[key]
    if key in TASK_SPECS:
        return TASK_SPECS[key]
    raise KeyError(f"Unknown LW native-articulation task: {key}")


@dataclass(frozen=True)
class LWNativeResetSample:
    task_id: str
    task_name: str
    seed: int
    randomize: bool
    initial_q_rad: tuple[float, ...]
    target_q_rad: tuple[float, ...]
    root_x_m: float | None = None
    root_y_m: float | None = None
    root_yaw_deg: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "seed": int(self.seed),
            "randomize": bool(self.randomize),
            "initial_q_rad": list(self.initial_q_rad),
            "target_q_rad": list(self.target_q_rad),
            "root_x_m": (
                None if self.root_x_m is None else float(self.root_x_m)
            ),
            "root_y_m": (
                None if self.root_y_m is None else float(self.root_y_m)
            ),
            "root_yaw_deg": float(self.root_yaw_deg),
        }


def sample_reset(
    task: str | LWNativeTaskSpec,
    *,
    seed: int = 0,
    randomize: bool = False,
) -> LWNativeResetSample:
    """Return a deterministic, graspable, non-target reset sample.

    Non-random resets use the task's audited initial articulation coordinates.
    G08 preserves its closed hard-stop source pose, G05 uses a small coherent
    pre-opening, and G10 uses a half-open two-sided pose so thin outer links
    are separately graspable. Optional randomization stays limit-safe and
    never places a task in or near success.
    """

    spec = get_task_spec(task)
    normalized_seed = int(seed)
    q0 = np.asarray(spec.authored_q0_rad, dtype=np.float64)
    root_x_m: float | None = None
    root_y_m: float | None = None
    root_yaw_deg = 0.0
    if spec.task_id in ("18", "EVAL-Gen02"):
        root_x_m = G08_ROOT_NOMINAL_X_M
        root_y_m = G08_ROOT_NOMINAL_Y_M
    elif spec.task_id == "49":
        root_x_m = G10_ROOT_NOMINAL_X_M
        root_y_m = G10_ROOT_NOMINAL_Y_M
    if randomize:
        rng = np.random.default_rng(normalized_seed)
        if spec.task_id == "G05":
            # Keep all ribs coherently pre-opened and Prop010 exactly locked.
            # The 2.0--2.5 degree range preserves a distinct two-hand grasp
            # envelope while remaining well outside the open target window.
            opening_deg = float(rng.choice(np.asarray([2.0, 2.25, 2.5])))
            q0 = np.asarray(
                _deg((opening_deg,) * 9 + (0.0,) + (-opening_deg,) * 9),
                dtype=np.float64,
            )
            q0[9] = 0.0
        elif spec.task_id in ("18", "EVAL-Gen02"):
            # Keep the lid at its closed hard stop and randomize only the
            # reset-positioned fixed base.  This yields meaningful reach and
            # camera diversity without introducing pre-action hinge motion.
            q0 = np.asarray(spec.authored_q0_rad, dtype=np.float64)
            root_x_m = float(
                rng.uniform(
                    G08_ROOT_NOMINAL_X_M
                    - G08_ROOT_RANDOM_X_HALF_RANGE_M,
                    G08_ROOT_NOMINAL_X_M
                    + G08_ROOT_RANDOM_X_HALF_RANGE_M,
                )
            )
            root_y_m = float(
                rng.uniform(
                    G08_ROOT_NOMINAL_Y_M
                    - G08_ROOT_RANDOM_Y_HALF_RANGE_M,
                    G08_ROOT_NOMINAL_Y_M
                    + G08_ROOT_RANDOM_Y_HALF_RANGE_M,
                )
            )
            root_yaw_deg = float(
                rng.uniform(
                    -G08_ROOT_RANDOM_YAW_HALF_RANGE_DEG,
                    G08_ROOT_RANDOM_YAW_HALF_RANGE_DEG,
                )
            )
        elif spec.task_id == "49":
            # Keep a coherent half-open pose and randomize the free root below.
            # This is the audited passive equilibrium for the source's two real
            # hinges; link and velocity damping absorb the small landing impulse
            # without imposing a position target or disabling self-collision.
            opening_deg = 90.0
            q0 = _as_float_array(_deg((-opening_deg, opening_deg)), 2)
            root_x_m = float(
                rng.uniform(
                    G10_ROOT_NOMINAL_X_M - G10_ROOT_RANDOM_X_HALF_RANGE_M,
                    G10_ROOT_NOMINAL_X_M + G10_ROOT_RANDOM_X_HALF_RANGE_M,
                )
            )
            root_y_m = float(
                rng.uniform(
                    G10_ROOT_NOMINAL_Y_M - G10_ROOT_RANDOM_Y_HALF_RANGE_M,
                    G10_ROOT_NOMINAL_Y_M + G10_ROOT_RANDOM_Y_HALF_RANGE_M,
                )
            )
            root_yaw_deg = float(
                rng.uniform(
                    -G10_ROOT_RANDOM_YAW_HALF_RANGE_DEG,
                    G10_ROOT_RANDOM_YAW_HALF_RANGE_DEG,
                )
            )

    target = np.asarray(spec.target_q_rad, dtype=np.float64)
    if _is_global_target(spec, q0, np.zeros_like(q0), exit_window=False):
        raise RuntimeError(f"Reset sampler placed {spec.task_name} in target")
    if spec.locked_dof_index is not None and q0[spec.locked_dof_index] != 0.0:
        raise RuntimeError(f"Reset sampler moved locked DOF for {spec.task_name}")
    return LWNativeResetSample(
        task_id=spec.task_id,
        task_name=spec.task_name,
        seed=normalized_seed,
        randomize=bool(randomize),
        initial_q_rad=tuple(float(value) for value in q0),
        target_q_rad=tuple(float(value) for value in target),
        root_x_m=root_x_m,
        root_y_m=root_y_m,
        root_yaw_deg=root_yaw_deg,
    )


def _as_float_array(values: Any, expected_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != expected_size:
        raise ValueError(f"Expected {expected_size} values, received {array.size}")
    return array


def _target_mask(
    spec: LWNativeTaskSpec,
    q: np.ndarray,
    qd: np.ndarray,
    *,
    exit_window: bool,
) -> np.ndarray:
    lower = np.asarray(
        spec.target_exit_lower_rad if exit_window else spec.target_enter_lower_rad,
        dtype=np.float64,
    )
    upper = np.asarray(
        spec.target_exit_upper_rad if exit_window else spec.target_enter_upper_rad,
        dtype=np.float64,
    )
    return (q >= lower) & (q <= upper) & (
        np.abs(qd) <= spec.target_speed_rad_s
    )


def _position_target_mask(
    spec: LWNativeTaskSpec,
    q: np.ndarray,
    *,
    exit_window: bool,
) -> np.ndarray:
    """Return target-window membership without a velocity requirement."""

    lower = np.asarray(
        spec.target_exit_lower_rad if exit_window else spec.target_enter_lower_rad,
        dtype=np.float64,
    )
    upper = np.asarray(
        spec.target_exit_upper_rad if exit_window else spec.target_enter_upper_rad,
        dtype=np.float64,
    )
    if spec.straightness_only:
        # A straight hinge is periodic: +180 degrees and -180 degrees are the
        # same physical orientation. Compare the wrapped absolute angle so
        # either signed representation satisfies the straightness window.
        wrapped_abs = np.abs(np.arctan2(np.sin(q), np.cos(q)))
        lower_abs = np.minimum(np.abs(lower), np.abs(upper))
        upper_abs = np.maximum(np.abs(lower), np.abs(upper))
        return (wrapped_abs >= lower_abs) & (wrapped_abs <= upper_abs)
    # PhysX may report a hinge a few ulps beyond an authored hard stop.  Extend
    # only a target boundary that is also a physical limit; the actual opening
    # threshold remains exact, while the independent joint-limit guard still
    # rejects a real excursion beyond ``limit_failure_margin_rad``.
    physical_lower = np.asarray(spec.lower_limits_rad, dtype=np.float64)
    physical_upper = np.asarray(spec.upper_limits_rad, dtype=np.float64)
    lower = np.where(
        np.isclose(lower, physical_lower, atol=1.0e-12, rtol=0.0),
        lower - spec.limit_failure_margin_rad,
        lower,
    )
    upper = np.where(
        np.isclose(upper, physical_upper, atol=1.0e-12, rtol=0.0),
        upper + spec.limit_failure_margin_rad,
        upper,
    )
    return (q >= lower) & (q <= upper)


def _side_progress(
    spec: LWNativeTaskSpec,
    initial_q: np.ndarray,
    q: np.ndarray,
    indices: tuple[int, ...],
) -> float:
    if not indices:
        return 1.0
    index = np.asarray(indices, dtype=np.int64)
    direction = np.asarray(spec.direction_by_dof, dtype=np.float64)[index]
    numerator = direction * (q[index] - initial_q[index])
    target = np.asarray(spec.target_q_rad, dtype=np.float64)
    denominator = np.maximum(
        direction * (target[index] - initial_q[index]),
        1.0e-9,
    )
    return float(np.clip(np.mean(numerator / denominator), 0.0, 1.0))


def _is_global_target(
    spec: LWNativeTaskSpec,
    q: np.ndarray,
    qd: np.ndarray,
    *,
    exit_window: bool,
) -> bool:
    mask = _target_mask(spec, q, qd, exit_window=exit_window)
    moving = np.asarray(spec.direction_by_dof, dtype=np.int8) != 0
    return int(np.count_nonzero(mask & moving)) >= spec.global_target_min_count


def _is_global_position_target(
    spec: LWNativeTaskSpec,
    q: np.ndarray,
    *,
    exit_window: bool,
) -> bool:
    """Check a geometric target without speed, contact, or timing gates."""

    mask = _position_target_mask(spec, q, exit_window=exit_window)
    moving = np.asarray(spec.direction_by_dof, dtype=np.int8) != 0
    return int(np.count_nonzero(mask & moving)) >= spec.global_target_min_count


@dataclass(frozen=True)
class LWContractOutput:
    """Immutable per-substep view returned to the live environment."""

    conditions: tuple[bool, ...]
    observation: tuple[float, ...]
    metrics: Mapping[str, float | int | bool | str]


class LWNativeArticulationState:
    """No-assist 120 Hz state machine shared by G05, G08, and G10."""

    def __init__(
        self,
        task: str | LWNativeTaskSpec,
        sample: LWNativeResetSample | None = None,
    ) -> None:
        self.spec = get_task_spec(task)
        self.reset(sample or sample_reset(self.spec))

    def reset(self, sample: LWNativeResetSample | None = None) -> None:
        sample = sample or sample_reset(self.spec)
        if (
            sample.task_id != self.spec.task_id
            or sample.task_name != self.spec.task_name
        ):
            raise ValueError(
                f"Reset sample {sample.task_name} does not match {self.spec.task_name}"
            )
        initial_q = _as_float_array(sample.initial_q_rad, self.spec.dof_count)
        target_q = _as_float_array(sample.target_q_rad, self.spec.dof_count)
        if not np.all(np.isfinite(initial_q)) or not np.all(np.isfinite(target_q)):
            raise ValueError("Reset articulation coordinates must be finite")
        if not np.allclose(
            target_q,
            np.asarray(self.spec.target_q_rad, dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("Reset sample target does not match immutable task spec")
        if _is_global_target(
            self.spec,
            initial_q,
            np.zeros_like(initial_q),
            exit_window=False,
        ):
            raise ValueError("Reset articulation state is already in the target window")

        self.sample = sample
        self.initial_q = initial_q.copy()
        self.target_q = target_q.copy()
        self.last_q = initial_q.copy()
        self.last_qd = np.zeros(self.spec.dof_count, dtype=np.float64)
        self.sim_time_s = 0.0
        self.stage = LWTaskStage.WAIT_CONTACT
        self.success = False
        self.failure_reason = ""
        self.dof_truth_available = False
        self.contact_truth_available = False
        self.contact_active = np.zeros(2, dtype=bool)
        self.contact_force_n = np.zeros(2, dtype=np.float64)
        self.contact_seen = np.zeros(2, dtype=bool)
        self.role_qualified = np.zeros(2, dtype=bool)
        self.motion_candidate_time = np.full(2, np.nan, dtype=np.float64)
        self.first_motion_time = np.full(2, np.nan, dtype=np.float64)
        self.contact_progress_rad = np.zeros(2, dtype=np.float64)
        self.wrong_direction_rad = np.zeros(2, dtype=np.float64)
        self.target_seen = np.zeros(2, dtype=bool)
        self.target_entry_time = np.full(2, np.nan, dtype=np.float64)
        self.overlap_s = 0.0
        self.target_hold_s = 0.0
        self.release_s = 0.0
        self.stable_s = 0.0
        # These are diagnostic latches for the post-target gate.  They are
        # recomputed on every live substep (and reset on a recoverable retry)
        # so metrics can distinguish a target-window failure from a release
        # contact failure for tasks that use the staged contact/release path.
        self.exit_target_valid = False
        self.exit_target_position_valid = False
        # Wall-clock duration for which a configured left stabilizer has been
        # below the contact-off threshold. This is deliberately separate
        # from ``contact_active``: the latter is the debounced logical
        # witness consumed by the task predicates.
        self.stabilizer_contact_loss_s = 0.0
        self.target_speed_loss_s = 0.0
        self.target_hold_complete = False
        self.release_complete = False
        self.root_translation_error_m = 0.0
        self.root_rotation_error_deg = 0.0
        self.root_goal_valid = True
        # Accumulated over-limit time for fixed-root drift debouncing.  These
        # are independent channels so a transient translation spike cannot
        # hide a sustained rotation error (or vice versa).
        self.root_translation_drift_s = 0.0
        self.root_rotation_drift_s = 0.0
        self.root_translation_escape_s = 0.0
        self.root_rotation_escape_s = 0.0
        self.support_truth_available = not self.spec.requires_support_contact
        self.support_goal_valid = not self.spec.requires_support_contact
        self.task_goal_valid = not self.spec.requires_support_contact

    def _clear_recoverable_attempt(self) -> None:
        """Discard invalidated progress without forcing a physical reset.

        Current contact hysteresis and simulator time remain observable, while
        no invalid motion can be counted retroactively toward success.
        """

        self.stage = LWTaskStage.WAIT_CONTACT
        self.contact_seen[:] = False
        self.role_qualified[:] = False
        self.motion_candidate_time[:] = np.nan
        self.first_motion_time[:] = np.nan
        self.contact_progress_rad[:] = 0.0
        self.wrong_direction_rad[:] = 0.0
        self.target_seen[:] = False
        self.target_entry_time[:] = np.nan
        self.overlap_s = 0.0
        self.target_hold_s = 0.0
        self.release_s = 0.0
        self.stable_s = 0.0
        self.exit_target_valid = False
        self.exit_target_position_valid = False
        self.stabilizer_contact_loss_s = 0.0
        self.target_speed_loss_s = 0.0
        self.target_hold_complete = False
        self.release_complete = False
        self.root_translation_drift_s = 0.0
        self.root_rotation_drift_s = 0.0

    def _invalidate_attempt_or_fail(self, reason: str) -> None:
        """Restart a recoverable task, otherwise preserve its hard failure.

        Specs with recoverable root drift may retry after a staged predicate
        is invalidated; other specs keep fail-closed semantics. Callers invoke this only
        after ``last_q``/``last_qd`` have been synchronized for the current
        substep, so motion during the invalid attempt cannot be counted later.
        """

        if self.spec.recoverable_root_drift:
            self._clear_recoverable_attempt()
            return
        self._fail(reason)

    @property
    def start_skew_s(self) -> float:
        if not np.all(np.isfinite(self.first_motion_time)):
            return -1.0
        return float(abs(self.first_motion_time[0] - self.first_motion_time[1]))

    @property
    def target_skew_s(self) -> float:
        if not np.all(np.isfinite(self.target_entry_time)):
            return -1.0
        return float(abs(self.target_entry_time[0] - self.target_entry_time[1]))

    def _fail(self, reason: str) -> None:
        # Terminal first-cause latch: later callbacks cannot obscure the first
        # missing truth channel or physical violation.
        if not self.failure_reason:
            self.failure_reason = str(reason)
            self.success = False
            self.stage = LWTaskStage.FAILED

    def _update_contact_hysteresis(
        self,
        left_force: float,
        right_force: float,
        *,
        latch_seen: bool = True,
    ) -> None:
        self.contact_force_n[:] = (max(0.0, left_force), max(0.0, right_force))
        for side in range(2):
            force = self.contact_force_n[side]
            if self.contact_active[side]:
                self.contact_active[side] = force > CONTACT_OFF_N
            else:
                self.contact_active[side] = force >= CONTACT_ON_N
        if latch_seen:
            self.contact_seen |= self.contact_active

    def _update_stabilizer_contact_debounce(self, dt: float) -> bool:
        """Keep a brief configured stabilizer-contact gap recoverable.

        PhysX contact reports are sampled after each physics substep and may
        briefly dip below ``CONTACT_OFF_N`` while a hand is still bracing the
        object. Preserve the logical left-contact witness until a
        wall-clock debounce expires.  The helper returns whether a loss is
        currently pending; callers use that signal to prevent target/release
        hold time from being credited during the gap.  Once the debounce is
        exceeded, ``contact_active`` remains false and the normal recoverable
        attempt invalidation runs immediately afterward.
        """

        debounce_s = float(self.spec.stabilizer_contact_loss_debounce_s)
        if (
            debounce_s <= 0.0
            or not self.spec.left_is_stabilizer
            or not self.spec.keep_left_contact_after_target
            or not self.contact_seen[0]
        ):
            self.stabilizer_contact_loss_s = 0.0
            return False

        # ``_update_contact_hysteresis`` has already applied the normal
        # off/on thresholds.  During a pending loss, however, a force that is
        # merely back above the off threshold is enough to prove that the
        # contact never remained absent for the full debounce interval; do not
        # require a second on-threshold crossing after that one-frame dip.
        if self.contact_force_n[0] > CONTACT_OFF_N:
            self.contact_active[0] = True
            self.stabilizer_contact_loss_s = 0.0
            return False

        if self.contact_active[0]:
            self.stabilizer_contact_loss_s = 0.0
            return False

        self.stabilizer_contact_loss_s += float(dt)
        if not _time_at_least(self.stabilizer_contact_loss_s, debounce_s):
            # Restore the debounced logical witness for this short interval.
            # The raw force and loss timer remain observable in metrics, and
            # the hold/release predicates below refuse to credit this frame.
            self.contact_active[0] = True
            return True
        return False

    def _stabilizer_contact_loss_pending(self) -> bool:
        """Return whether a stabilizer gap is inside its debounce window."""

        debounce_s = float(self.spec.stabilizer_contact_loss_debounce_s)
        return bool(
            debounce_s > 0.0
            and self.stabilizer_contact_loss_s > 0.0
            and not _time_at_least(self.stabilizer_contact_loss_s, debounce_s)
        )

    def _update_target_speed_debounce(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        dt: float,
        *,
        exit_window: bool,
    ) -> tuple[bool, bool]:
        """Pause target/release hold for a short hard-stop speed spike.

        The position window remains mandatory.  Once a target has been seen,
        a transient ``|qd| > target_speed`` sample pauses elapsed hold time;
        it does not add credit.  If the same position-window speed violation
        persists for the configured debounce interval, the caller clears the
        recoverable attempt.  Other LW tasks retain their immediate,
        fail-closed speed semantics because their debounce is zero.
        """

        debounce_s = float(self.spec.target_speed_loss_debounce_s)
        if (
            debounce_s <= 0.0
            or self.stage < LWTaskStage.TARGET_HOLD
            or not np.all(
                _position_target_mask(self.spec, q, exit_window=exit_window)
                & (
                    np.asarray(self.spec.direction_by_dof, dtype=np.int8)
                    != 0
                )
            )
        ):
            self.target_speed_loss_s = 0.0
            return False, False

        moving = np.asarray(self.spec.direction_by_dof, dtype=np.int8) != 0
        speed_bad = bool(np.any(np.abs(qd[moving]) > self.spec.target_speed_rad_s))
        if not speed_bad:
            self.target_speed_loss_s = 0.0
            return False, False

        self.target_speed_loss_s += float(dt)
        expired = _time_at_least(self.target_speed_loss_s, debounce_s)
        return (not expired), expired

    def prime_recording_origin_truth(
        self,
        q: Sequence[float] | np.ndarray,
        qd: Sequence[float] | np.ndarray,
        left_force: float,
        right_force: float,
        dof_available: bool,
        contact_available: bool,
        support_available: bool = True,
        support_goal_valid: bool = True,
    ) -> LWContractOutput:
        """Publish accepted action-0 truth without advancing task state.

        Stable-origin settling happens before the operator trajectory.  Its
        final measured articulation/contact values belong in the action-0
        observation, but must not count as elapsed time, contact evidence,
        physical motion, target hold, release, or success progress.
        """

        if (
            self.sim_time_s != 0.0
            or self.stage != LWTaskStage.WAIT_CONTACT
            or bool(np.any(self.contact_seen))
            or bool(np.any(self.role_qualified))
            or self.success
            or bool(self.failure_reason)
        ):
            raise RuntimeError(
                "Recording-origin truth can prime only a fresh LW reset state"
            )
        try:
            q_array = _as_float_array(q, self.spec.dof_count)
            qd_array = _as_float_array(qd, self.spec.dof_count)
            left_force_value = float(left_force)
            right_force_value = float(right_force)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid recording-origin truth shape or type") from exc
        values = np.concatenate(
            (
                q_array,
                qd_array,
                np.asarray([left_force_value, right_force_value]),
            )
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Recording-origin truth must be finite")

        self.dof_truth_available = bool(dof_available)
        self.contact_truth_available = bool(contact_available)
        self.support_truth_available = bool(support_available)
        self.support_goal_valid = bool(support_goal_valid)
        self.task_goal_valid = bool(
            self.root_goal_valid
            and (
                not self.spec.requires_support_contact
                or (
                    self.support_truth_available
                    and self.support_goal_valid
                )
            )
        )
        self.last_q = q_array.copy()
        self.last_qd = qd_array.copy()
        if self.contact_truth_available:
            self._update_contact_hysteresis(
                left_force_value,
                right_force_value,
                latch_seen=False,
            )
        else:
            self.contact_force_n[:] = (
                max(0.0, left_force_value),
                max(0.0, right_force_value),
            )
            self.contact_active[:] = False
        return self._snapshot()

    def _side_motion(
        self,
        side: int,
        q: np.ndarray,
        qd: np.ndarray,
        delta_q: np.ndarray,
    ) -> bool:
        indices = (
            self.spec.left_dof_indices if side == 0 else self.spec.right_dof_indices
        )
        if not indices:
            return False
        assigned_contact = bool(self.contact_active[side])
        if self.spec.left_is_stabilizer and side == 1:
            assigned_contact = assigned_contact and bool(self.contact_active[0])
        index = np.asarray(indices, dtype=np.int64)
        direction = np.asarray(self.spec.direction_by_dof, dtype=np.float64)[index]
        directed_delta = direction * delta_q[index]
        directed_speed = float(np.mean(direction * qd[index]))
        if assigned_contact:
            self.contact_progress_rad[side] += float(
                np.mean(np.maximum(directed_delta, 0.0))
            )
            self.wrong_direction_rad[side] += float(
                np.mean(np.maximum(-directed_delta, 0.0))
            )
        moving = bool(
            assigned_contact and directed_speed > self.spec.motion_speed_rad_s
        )
        if moving:
            if not math.isfinite(self.motion_candidate_time[side]):
                self.motion_candidate_time[side] = self.sim_time_s
        elif not self.role_qualified[side]:
            self.motion_candidate_time[side] = np.nan

        if (
            not self.role_qualified[side]
            and moving
            and self.contact_progress_rad[side]
            >= self.spec.motion_displacement_rad[side] - 1.0e-12
        ):
            self.role_qualified[side] = True
            onset = self.motion_candidate_time[side]
            self.first_motion_time[side] = (
                float(onset) if math.isfinite(onset) else self.sim_time_s
            )
        return moving

    def _side_target_candidate(
        self,
        side: int,
        q: np.ndarray,
        qd: np.ndarray,
        *,
        exit_window: bool,
        require_stabilizer_contact: bool = True,
        require_speed: bool = True,
        require_assigned_contact: bool = False,
    ) -> bool:
        if self.spec.left_is_stabilizer and side == 0:
            # A configured left-hand stabilizer remains required during the
            # staged target hold unless the caller explicitly permits loss.
            return bool(
                self.role_qualified[0]
                and (
                    self.contact_active[0]
                    if require_stabilizer_contact
                    else True
                )
            )
        indices = (
            self.spec.left_dof_indices if side == 0 else self.spec.right_dof_indices
        )
        if not indices:
            return False
        # The moving-side target witness normally belongs to an assigned hand
        # contact. A staged release task may disable this check after the
        # target has been achieved so releasing the hand does not erase it.
        if require_assigned_contact and not self.contact_active[side]:
            return False
        mask = (
            _target_mask(self.spec, q, qd, exit_window=exit_window)
            if require_speed
            else _position_target_mask(self.spec, q, exit_window=exit_window)
        )
        near_count = int(np.count_nonzero(mask[np.asarray(indices, dtype=np.int64)]))
        progress = _side_progress(self.spec, self.initial_q, q, indices)
        return bool(
            near_count >= self.spec.side_target_min_count[side]
            and progress >= self.spec.side_progress_min[side]
        )

    def _snapshot(self) -> LWContractOutput:
        metrics = MappingProxyType(self.metrics())
        return LWContractOutput(
            conditions=self.condition_flags(),
            observation=tuple(float(value) for value in self.observation_vector()),
            metrics=metrics,
        )

    def update(
        self,
        q: Sequence[float] | np.ndarray,
        qd: Sequence[float] | np.ndarray,
        left_force: float,
        right_force: float,
        dof_available: bool,
        contact_available: bool,
        root_translation_error_m: float,
        root_rotation_error_deg: float,
        dt: float,
        support_available: bool = True,
        support_goal_valid: bool = True,
    ) -> LWContractOutput:
        """Advance one measured physics substep and return a fixed snapshot."""

        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"Invalid LW physics dt: {dt}")
        self.sim_time_s += dt
        if self.failure_reason or self.success:
            return self._snapshot()

        self.dof_truth_available = bool(dof_available)
        self.contact_truth_available = bool(contact_available)
        if not self.dof_truth_available:
            self._fail("dof_truth_unavailable")
            return self._snapshot()
        if (
            not self.contact_truth_available
            and not self.spec.success_on_position_target
        ):
            self._fail("contact_truth_unavailable")
            return self._snapshot()

        try:
            q_array = _as_float_array(q, self.spec.dof_count)
            qd_array = _as_float_array(qd, self.spec.dof_count)
            left_force_value = float(left_force)
            right_force_value = float(right_force)
            root_translation = float(root_translation_error_m)
            root_rotation = float(root_rotation_error_deg)
        except (TypeError, ValueError):
            self._fail("truth_shape_or_type_invalid")
            return self._snapshot()
        values = np.concatenate(
            (
                q_array,
                qd_array,
                np.asarray(
                    [
                        left_force_value,
                        right_force_value,
                        root_translation,
                        root_rotation,
                    ],
                    dtype=np.float64,
                ),
            )
        )
        if not np.all(np.isfinite(values)):
            self._fail("non_finite_truth")
            return self._snapshot()
        if root_translation < 0.0 or root_rotation < 0.0:
            self._fail("negative_root_error")
            return self._snapshot()

        self.root_translation_error_m = root_translation
        self.root_rotation_error_deg = root_rotation
        self._update_contact_hysteresis(left_force_value, right_force_value)

        if self.spec.locked_dof_index is not None:
            locked_index = self.spec.locked_dof_index
            if (
                abs(q_array[locked_index])
                > self.spec.locked_dof_tolerance_rad
            ):
                self._fail("locked_joint_moved")
                return self._snapshot()

        lower = np.asarray(self.spec.lower_limits_rad, dtype=np.float64)
        upper = np.asarray(self.spec.upper_limits_rad, dtype=np.float64)
        margin = self.spec.limit_failure_margin_rad
        if np.any(q_array < lower - margin) or np.any(q_array > upper + margin):
            self._fail("joint_limit_exceeded")
            return self._snapshot()

        self.root_goal_valid = bool(
            not self.spec.root_pose_constraint
            or (
                root_translation <= self.spec.root_translation_limit_m
                and root_rotation <= self.spec.root_rotation_limit_deg
            )
        )
        self.support_truth_available = bool(support_available)
        self.support_goal_valid = bool(support_goal_valid)
        self.task_goal_valid = bool(
            self.root_goal_valid
            and (
                not self.spec.requires_support_contact
                or (
                    self.support_truth_available
                    and self.support_goal_valid
                )
            )
        )
        if self.spec.recoverable_root_drift:
            translation_escape = bool(
                root_translation
                > float(self.spec.root_escape_translation_limit_m)
            )
            rotation_escape = bool(
                root_rotation > float(self.spec.root_escape_rotation_limit_deg)
            )
            self.root_translation_escape_s = (
                self.root_translation_escape_s + dt
                if translation_escape
                else 0.0
            )
            self.root_rotation_escape_s = (
                self.root_rotation_escape_s + dt if rotation_escape else 0.0
            )
            if _time_at_least(
                self.root_translation_escape_s,
                self.spec.root_escape_hold_s,
            ):
                self._fail("root_translation_escape")
                return self._snapshot()
            if _time_at_least(
                self.root_rotation_escape_s,
                self.spec.root_escape_hold_s,
            ):
                self._fail("root_rotation_escape")
                return self._snapshot()
            # Support/contact goal invalidation clears progress; fixed-root
            # tasks retain the historical soft-drift recovery behavior.
            soft_goal_invalid = bool(
                (self.spec.root_pose_constraint and not self.root_goal_valid)
                or (
                    self.spec.requires_support_contact
                    and not self.support_goal_valid
                )
            )
            if soft_goal_invalid:
                self._clear_recoverable_attempt()
                self.last_q = q_array.copy()
                self.last_qd = qd_array.copy()
                return self._snapshot()
        else:
            translation_over_limit = bool(
                self.spec.root_pose_constraint
                and root_translation > self.spec.root_translation_limit_m
            )
            self.root_translation_drift_s = (
                self.root_translation_drift_s + dt
                if translation_over_limit
                else 0.0
            )
            if translation_over_limit and (
                self.spec.root_translation_drift_hold_s <= 0.0
                or _time_at_least(
                    self.root_translation_drift_s,
                    self.spec.root_translation_drift_hold_s,
                )
            ):
                self._fail("root_translation_drift")
                return self._snapshot()

            rotation_over_limit = bool(
                self.spec.root_pose_constraint
                and root_rotation > self.spec.root_rotation_limit_deg
            )
            self.root_rotation_drift_s = (
                self.root_rotation_drift_s + dt
                if rotation_over_limit
                else 0.0
            )
            if rotation_over_limit and (
                self.spec.root_rotation_drift_hold_s <= 0.0
                or _time_at_least(
                    self.root_rotation_drift_s,
                    self.spec.root_rotation_drift_hold_s,
                )
            ):
                self._fail("root_rotation_drift")
                return self._snapshot()

        if self.spec.success_on_position_target and _is_global_position_target(
            self.spec,
            q_array,
            exit_window=False,
        ):
            # The physical articulation coordinate is the complete success
            # predicate for G08 and G10.  Truth, joint-limit, and root-safety
            # checks above remain enforced; contact, motion history, release,
            # speed, and hold time are intentionally irrelevant.
            self.target_seen[:] = True
            self.target_entry_time[:] = np.where(
                np.isfinite(self.target_entry_time),
                self.target_entry_time,
                self.sim_time_s,
            )
            self.target_hold_complete = True
            self.release_complete = True
            self.success = True
            self.stage = LWTaskStage.COMPLETE
            self.last_q = q_array.copy()
            self.last_qd = qd_array.copy()
            return self._snapshot()

        if self.spec.success_on_position_target:
            # Position-only tasks do not accumulate hand-role or signed-motion
            # state before reaching the goal.  This also prevents an ordinary
            # grasp impulse from tripping the generic wrong-direction guard.
            self.last_q = q_array.copy()
            self.last_qd = qd_array.copy()
            return self._snapshot()

        # A continuous stabilizer witness is not merely a contact that may be
        # reacquired before the next predicate.
        # A short physics-time gap is retained by the task-local debounce,
        # while a sustained hysteresis-level loss still invalidates the current
        # attempt.  This remains recoverable (the operator can re-contact and
        # retry in the same episode), while a right-only approach before the
        # left hand ever touches is simply left unqualified and is not erased.
        stabilizer_contact_loss_pending = (
            self._update_stabilizer_contact_debounce(dt)
        )
        if (
            self.spec.left_is_stabilizer
            and self.spec.keep_left_contact_after_target
            and self.contact_seen[0]
            and not self.contact_active[0]
        ):
            self._invalidate_attempt_or_fail("stabilizer_contact_lost")
            return self._snapshot()

        # Once G08 has reached its target stage, retain the position-window
        # state across a bounded hard-stop speed spike.  This is a pause, not
        # credit: target/release timers are advanced only on a frame whose
        # measured speed is below the ordinary 0.08 rad/s limit.
        target_speed_loss_pending, target_speed_loss_expired = (
            self._update_target_speed_debounce(
                q_array,
                qd_array,
                dt,
                exit_window=self.stage >= LWTaskStage.WAIT_RELEASE,
            )
        )
        if target_speed_loss_expired:
            self._invalidate_attempt_or_fail("target_speed_lost")
            return self._snapshot()

        delta_q = q_array - self.last_q
        self.last_q = q_array.copy()
        self.last_qd = qd_array.copy()

        moving_now = np.zeros(2, dtype=bool)
        if self.spec.left_is_stabilizer:
            moving_now[1] = self._side_motion(1, q_array, qd_array, delta_q)
            if self.role_qualified[1] and self.contact_active[0]:
                self.role_qualified[0] = True
                if not math.isfinite(self.first_motion_time[0]):
                    self.first_motion_time[0] = self.first_motion_time[1]
        else:
            moving_now[0] = self._side_motion(0, q_array, qd_array, delta_q)
            moving_now[1] = self._side_motion(1, q_array, qd_array, delta_q)

        for side in range(2):
            tolerance = self.spec.wrong_direction_tolerance_rad[side]
            if tolerance > 0.0 and self.wrong_direction_rad[side] > tolerance:
                self._invalidate_attempt_or_fail(
                    "left_wrong_direction"
                    if side == 0
                    else "right_wrong_direction"
                )
                return self._snapshot()

        if bool(np.any(self.contact_active)) and self.stage == LWTaskStage.WAIT_CONTACT:
            self.stage = LWTaskStage.WAIT_MOTION
        if bool(np.any(self.role_qualified)) and self.stage < LWTaskStage.MOVING:
            self.stage = LWTaskStage.MOVING

        if not self.spec.left_is_stabilizer and bool(np.all(moving_now)):
            self.overlap_s += dt

        if self.spec.start_skew_limit_s is not None and np.all(
            np.isfinite(self.first_motion_time)
        ):
            if self.start_skew_s > self.spec.start_skew_limit_s + _TIME_EPSILON_S:
                self._fail("start_skew_exceeded")
                return self._snapshot()

        roles_ready = bool(np.all(self.role_qualified))
        target_candidates = np.asarray(
            [
                self._side_target_candidate(
                    0,
                    q_array,
                    qd_array,
                    exit_window=False,
                    require_assigned_contact=self.spec.task_id == "18",
                ),
                self._side_target_candidate(
                    1,
                    q_array,
                    qd_array,
                    exit_window=False,
                    require_assigned_contact=self.spec.task_id == "18",
                ),
            ],
            dtype=bool,
        )
        global_target = _is_global_target(
            self.spec,
            q_array,
            qd_array,
            exit_window=False,
        )
        target_position_candidates = np.asarray(
            [
                self._side_target_candidate(
                    0,
                    q_array,
                    qd_array,
                    exit_window=False,
                    require_speed=False,
                    require_assigned_contact=self.spec.task_id == "18",
                ),
                self._side_target_candidate(
                    1,
                    q_array,
                    qd_array,
                    exit_window=False,
                    require_speed=False,
                    require_assigned_contact=self.spec.task_id == "18",
                ),
            ],
            dtype=bool,
        )
        global_position_target = _is_global_position_target(
            self.spec,
            q_array,
            exit_window=False,
        )
        if self.spec.left_is_stabilizer and roles_ready:
            # The stabilizing hand has no articulation DOF of its own.  Its
            # target event is therefore the same measured instant at which the
            # contacted lid first reaches the physical target window.
            if (
                bool(np.all(target_candidates))
                and global_target
                and not stabilizer_contact_loss_pending
            ):
                for side in range(2):
                    if not self.target_seen[side]:
                        self.target_seen[side] = True
                        self.target_entry_time[side] = self.sim_time_s
        elif roles_ready:
            for side in range(2):
                if target_candidates[side] and not self.target_seen[side]:
                    self.target_seen[side] = True
                    self.target_entry_time[side] = self.sim_time_s

        if self.spec.target_skew_limit_s is not None and np.all(self.target_seen):
            if self.target_skew_s > self.spec.target_skew_limit_s + _TIME_EPSILON_S:
                self._fail("target_skew_exceeded")
                return self._snapshot()

        overlap_ready = _time_at_least(
            self.overlap_s,
            self.spec.overlap_required_s,
        )
        assigned_contacts_for_hold = bool(np.all(self.contact_active))
        target_hold_valid = bool(
            roles_ready
            and np.all(target_candidates)
            and global_target
            and overlap_ready
            and assigned_contacts_for_hold
            and not stabilizer_contact_loss_pending
            and not target_speed_loss_pending
        )
        target_hold_position_valid = bool(
            roles_ready
            and np.all(target_position_candidates)
            and global_position_target
            and overlap_ready
            and assigned_contacts_for_hold
            and not stabilizer_contact_loss_pending
        )

        if self.stage <= LWTaskStage.TARGET_HOLD:
            if target_hold_valid:
                self.stage = LWTaskStage.TARGET_HOLD
                self.target_hold_s += dt
            elif self.stage == LWTaskStage.TARGET_HOLD:
                # A short speed-only loss is recoverable in place.  Any
                # position/contact loss still clears the attempt immediately.
                if not (
                    target_speed_loss_pending and target_hold_position_valid
                ):
                    self.target_hold_s = 0.0
                    self.stage = LWTaskStage.MOVING
            if _time_at_least(self.target_hold_s, TARGET_HOLD_S):
                self.target_hold_complete = True
                self.stage = LWTaskStage.WAIT_RELEASE
                return self._snapshot()

        exit_target = _is_global_target(
            self.spec,
            q_array,
            qd_array,
            exit_window=True,
        ) and all(
            self._side_target_candidate(
                side,
                q_array,
                qd_array,
                exit_window=True,
                require_stabilizer_contact=(
                    self.spec.keep_left_contact_after_target
                    if self.spec.left_is_stabilizer and side == 0
                    else False
                ),
                # G08's right hand is the releasing hand.  Its target
                # position/speed remains a valid exit witness after contact
                # drops to zero; the left stabilizer branch above still uses
                # ``keep_left_contact_after_target``.
                require_assigned_contact=(
                    self.spec.task_id == "18"
                    and side == 0
                    and self.spec.left_is_stabilizer
                    and self.spec.keep_left_contact_after_target
                ),
            )
            for side in range(2)
        )
        exit_target_position = _is_global_position_target(
            self.spec,
            q_array,
            exit_window=True,
        ) and all(
            self._side_target_candidate(
                side,
                q_array,
                qd_array,
                exit_window=True,
                require_speed=False,
                require_stabilizer_contact=(
                    self.spec.keep_left_contact_after_target
                    if self.spec.left_is_stabilizer and side == 0
                    else False
                ),
                # Keep this position-only fallback aligned with the speed-
                # checked exit path.  Otherwise a transient qdot debounce
                # would accidentally re-introduce a right-contact gate.
                require_assigned_contact=(
                    self.spec.task_id == "18"
                    and side == 0
                    and self.spec.left_is_stabilizer
                    and self.spec.keep_left_contact_after_target
                ),
            )
            for side in range(2)
        )
        self.exit_target_valid = bool(exit_target)
        self.exit_target_position_valid = bool(exit_target_position)

        if self.stage == LWTaskStage.WAIT_RELEASE:
            if self.spec.keep_left_contact_after_target and not self.contact_active[0]:
                self._invalidate_attempt_or_fail("stabilizer_contact_lost")
                return self._snapshot()
            if not exit_target and not (
                target_speed_loss_pending and exit_target_position
            ):
                self._invalidate_attempt_or_fail("target_lost_before_release")
                return self._snapshot()
            if stabilizer_contact_loss_pending or target_speed_loss_pending:
                # Do not credit the final release window while the stabilizer
                # contact is only being carried by the debounce.  Requiring a
                # fresh uninterrupted hold keeps the success predicate strict
                # without discarding the already-qualified roles.
                self.release_s = 0.0
                return self._snapshot()
            release_valid = bool(
                (not self.spec.release_left or not self.contact_active[0])
                and (not self.spec.release_right or not self.contact_active[1])
            )
            self.release_s = self.release_s + dt if release_valid else 0.0
            if _time_at_least(self.release_s, RELEASE_HOLD_S):
                self.release_complete = True
                if self.spec.success_on_release:
                    # Some tasks define the first valid release as the end
                    # event.  G08 intentionally stops here: the left hand
                    # has already been verified continuously through opening,
                    # target hold, and this release debounce.  It may remain
                    # in contact after the success latch; releasing it is not
                    # an additional condition.
                    self.success = True
                    self.stage = LWTaskStage.COMPLETE
                    return self._snapshot()
                self.stage = LWTaskStage.STABILIZE
                return self._snapshot()

        if self.stage == LWTaskStage.STABILIZE:
            if self.spec.keep_left_contact_after_target and not self.contact_active[0]:
                self._invalidate_attempt_or_fail("stabilizer_contact_lost")
                return self._snapshot()
            if not exit_target and not (
                target_speed_loss_pending and exit_target_position
            ):
                self._invalidate_attempt_or_fail("target_lost_after_release")
                return self._snapshot()
            release_valid = bool(
                (not self.spec.release_left or not self.contact_active[0])
                and (not self.spec.release_right or not self.contact_active[1])
            )
            stable_valid = bool(
                release_valid
                and np.all(np.abs(qd_array) <= self.spec.stable_speed_rad_s)
            )
            self.stable_s = self.stable_s + dt if stable_valid else 0.0
            if _time_at_least(self.stable_s, STABLE_HOLD_S):
                self.success = True
                self.stage = LWTaskStage.COMPLETE

        return self._snapshot()

    def condition_flags(self) -> tuple[bool, ...]:
        truth_available = bool(
            self.dof_truth_available and self.contact_truth_available
        )
        synchronized_motion = bool(
            np.all(self.role_qualified)
            and (
                self.spec.start_skew_limit_s is None
                or (
                    self.start_skew_s >= 0.0
                    and self.start_skew_s <= self.spec.start_skew_limit_s
                )
            )
            and _time_at_least(self.overlap_s, self.spec.overlap_required_s)
        )
        synchronized_target = bool(
            np.all(self.target_seen)
            and (
                self.spec.target_skew_limit_s is None
                or (
                    self.target_skew_s >= 0.0
                    and self.target_skew_s <= self.spec.target_skew_limit_s
                )
            )
        )
        return (
            truth_available,
            bool(self.contact_seen[0]),
            bool(self.contact_seen[1]),
            bool(self.role_qualified[0]),
            bool(self.role_qualified[1]),
            synchronized_motion,
            synchronized_target,
            bool(self.target_hold_complete),
            bool(self.release_complete),
            bool(self.success),
        )

    def metrics(self) -> dict[str, float | int | bool | str]:
        return {
            "task_id": self.spec.task_id,
            "task_name": self.spec.task_name,
            "success_on_position_target": bool(
                self.spec.success_on_position_target
            ),
            "straightness_only": bool(self.spec.straightness_only),
            "success_on_release": bool(self.spec.success_on_release),
            "stage": int(self.stage),
            "sim_time_s": float(self.sim_time_s),
            "dof_count": int(self.spec.dof_count),
            "dof_truth_available": bool(self.dof_truth_available),
            "contact_truth_available": bool(self.contact_truth_available),
            "left_contact_active": bool(self.contact_active[0]),
            "right_contact_active": bool(self.contact_active[1]),
            "left_contact_seen": bool(self.contact_seen[0]),
            "right_contact_seen": bool(self.contact_seen[1]),
            "left_role_qualified": bool(self.role_qualified[0]),
            "right_role_qualified": bool(self.role_qualified[1]),
            "left_target_seen": bool(self.target_seen[0]),
            "right_target_seen": bool(self.target_seen[1]),
            "start_skew_s": float(self.start_skew_s),
            "target_skew_s": float(self.target_skew_s),
            "overlap_s": float(self.overlap_s),
            "target_hold_s": float(self.target_hold_s),
            "release_s": float(self.release_s),
            "stable_s": float(self.stable_s),
            "exit_target_valid": bool(self.exit_target_valid),
            "exit_target_position_valid": bool(
                self.exit_target_position_valid
            ),
            "stabilizer_contact_loss_s": float(
                self.stabilizer_contact_loss_s
            ),
            "stabilizer_contact_loss_debounce_s": float(
                self.spec.stabilizer_contact_loss_debounce_s
            ),
            "stabilizer_contact_loss_pending": bool(
                self._stabilizer_contact_loss_pending()
            ),
            "target_speed_loss_s": float(self.target_speed_loss_s),
            "target_speed_loss_debounce_s": float(
                self.spec.target_speed_loss_debounce_s
            ),
            "target_speed_loss_pending": bool(
                self.spec.target_speed_loss_debounce_s > 0.0
                and self.target_speed_loss_s > 0.0
                and not _time_at_least(
                    self.target_speed_loss_s,
                    self.spec.target_speed_loss_debounce_s,
                )
            ),
            "root_translation_error_m": float(self.root_translation_error_m),
            "root_rotation_error_deg": float(self.root_rotation_error_deg),
            "root_pose_constraint": bool(self.spec.root_pose_constraint),
            "root_goal_valid": bool(self.root_goal_valid),
            "root_translation_drift_s": float(self.root_translation_drift_s),
            "root_rotation_drift_s": float(self.root_rotation_drift_s),
            "root_translation_drift_hold_s": float(
                self.spec.root_translation_drift_hold_s
            ),
            "root_rotation_drift_hold_s": float(
                self.spec.root_rotation_drift_hold_s
            ),
            "support_truth_available": bool(self.support_truth_available),
            "support_goal_valid": bool(self.support_goal_valid),
            "task_goal_valid": bool(self.task_goal_valid),
            "root_translation_escape_s": float(
                self.root_translation_escape_s
            ),
            "root_rotation_escape_s": float(self.root_rotation_escape_s),
            "root_escape_hold_s": float(self.spec.root_escape_hold_s),
            "target_hold_complete": bool(self.target_hold_complete),
            "release_complete": bool(self.release_complete),
            "success_now": bool(self.success),
            "failure_reason": self.failure_reason,
        }

    def observation_vector(self) -> np.ndarray:
        """Return the fixed 100-value observation shared by all three tasks.

        Layout: padded q[19], qd[19], target[19], target_error[19], then
        ``_SCALAR_OBSERVATION_LABELS``.  Padding is always zero.
        """

        q = np.zeros(MAX_DOF_COUNT, dtype=np.float64)
        qd = np.zeros(MAX_DOF_COUNT, dtype=np.float64)
        target = np.zeros(MAX_DOF_COUNT, dtype=np.float64)
        error = np.zeros(MAX_DOF_COUNT, dtype=np.float64)
        count = self.spec.dof_count
        q[:count] = self.last_q
        qd[:count] = self.last_qd
        target[:count] = self.target_q
        error[:count] = self.target_q - self.last_q
        scalars = np.asarray(
            [
                float(count),
                float(self.contact_active[0]),
                float(self.contact_active[1]),
                float(self.contact_force_n[0]),
                float(self.contact_force_n[1]),
                float(self.contact_seen[0]),
                float(self.contact_seen[1]),
                float(self.role_qualified[0]),
                float(self.role_qualified[1]),
                float(self.target_seen[0]),
                float(self.target_seen[1]),
                float(self.dof_truth_available),
                float(self.contact_truth_available),
                float(self.root_translation_error_m),
                float(self.root_rotation_error_deg),
                float(self.start_skew_s),
                float(self.target_skew_s),
                float(self.overlap_s),
                min(1.0, self.target_hold_s / TARGET_HOLD_S),
                min(1.0, self.release_s / RELEASE_HOLD_S),
                min(1.0, self.stable_s / STABLE_HOLD_S),
                float(self.stage),
                float(self.success),
                float(bool(self.failure_reason)),
            ],
            dtype=np.float64,
        )
        observation = np.concatenate((q, qd, target, error, scalars))
        if observation.shape != (OBSERVATION_VECTOR_SIZE,):
            raise RuntimeError("LW observation layout changed unexpectedly")
        return observation


__all__ = [
    "CONDITION_LABELS",
    "CONDITION_VECTOR_SIZE",
    "CONTACT_OFF_N",
    "CONTACT_ON_N",
    "G05_SPEC",
    "G08_ROOT_NOMINAL_X_M",
    "G08_ROOT_NOMINAL_Y_M",
    "G08_ROOT_RANDOM_X_HALF_RANGE_M",
    "G08_ROOT_RANDOM_Y_HALF_RANGE_M",
    "G08_ROOT_RANDOM_YAW_HALF_RANGE_DEG",
    "G08_LID_HARD_STOP_DEG",
    "G08_LID_OPEN_THRESHOLD_DEG",
    "G08_LID_TARGET_DEG",
    "G08_LID_TARGET_ENTER_DEG",
    "G08_LID_TARGET_EXIT_DEG",
    "G10_ROOT_NOMINAL_X_M",
    "G10_ROOT_NOMINAL_Y_M",
    "G10_ROOT_RANDOM_X_HALF_RANGE_M",
    "G10_ROOT_RANDOM_Y_HALF_RANGE_M",
    "G10_ROOT_RANDOM_YAW_HALF_RANGE_DEG",
    "G08_SPEC",
    "G10_SPEC",
    "LWContractOutput",
    "LWNativeArticulationState",
    "LWNativeResetSample",
    "LWNativeTaskSpec",
    "LWTaskStage",
    "MAX_DOF_COUNT",
    "NOMINAL_DT_S",
    "OBSERVATION_VECTOR_SIZE",
    "PHYSICS_HZ",
    "RELEASE_HOLD_S",
    "STABLE_HOLD_S",
    "TARGET_HOLD_S",
    "TASK_SPECS",
    "TASK_SPECS_BY_ID",
    "get_task_spec",
    "sample_reset",
]
