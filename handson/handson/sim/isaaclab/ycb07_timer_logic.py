"""Pure, replayable state machine for the YCB07 physical timer buttons."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np


BUTTON_NAMES = ("start_stop", "mode", "hr", "clear", "min", "memory", "sec")
BUTTON_INDEX = {name: index for index, name in enumerate(BUTTON_NAMES)}
TARGETS = ((1, 1), (1, 2), (2, 1), (2, 2))

STAGE_WAIT_BRACE = 0
STAGE_PROGRAM = 1
STAGE_READY = 2
STAGE_RUNNING = 3
STAGE_RELEASE_STABLE = 4
STAGE_SUCCESS = 5
STAGE_FAILED = 6

FAIL_NONE = 0
FAIL_WRONG_START = 1
FAIL_POST_START_BUTTON = 2
FAIL_MODE_SWITCH = 3
FAIL_OFF_TABLE = 4
FAIL_EXCESSIVE_TILT = 5
FAIL_PHYSICS = 6
FAIL_ROBOT_SAFETY = 7

ERROR_NONE = 0
ERROR_NO_RIGHT_CONTACT = 1
ERROR_NO_LEFT_BRACE = 2
ERROR_LEFT_BUTTON_CONTACT = 3
ERROR_SIMULTANEOUS = 4
ERROR_WRONG_FUNCTION = 5
ERROR_OVERSHOOT = 6
ERROR_LONG_HOLD = 7  # Legacy recordings only; new presses have no hold timeout.
ERROR_READBACK = 8

ERROR_REASONS = {
    ERROR_NONE: "none",
    ERROR_NO_RIGHT_CONTACT: "no right fingertip contact",
    ERROR_NO_LEFT_BRACE: "left hand was not bracing the case",
    ERROR_LEFT_BUTTON_CONTACT: "left hand touched a button",
    ERROR_SIMULTANEOUS: "multiple buttons pressed together",
    ERROR_WRONG_FUNCTION: "HR/MEMORY is not used in this task",
    ERROR_OVERSHOOT: "program exceeds the target",
    ERROR_LONG_HOLD: "button held longer than 1.5 seconds",
    ERROR_READBACK: "button/contact sensor readback unavailable",
}

LOGIC_SIZE = 18
LATCH_SIZE = 4 * len(BUTTON_NAMES)


@dataclass
class TimerProgrammingState:
    target_minutes: int = 1
    target_seconds: int = 1
    programmed_minutes: int = 0
    programmed_seconds: int = 0
    running: bool = False
    attempt_invalid: bool = False
    error_count: int = 0
    last_error_code: int = ERROR_NONE
    stage: int = STAGE_WAIT_BRACE
    run_elapsed_s: float = 0.0
    countdown_decrements: int = 0
    terminal_failure_code: int = FAIL_NONE
    terminal_failure_reason: str = ""
    start_press_seen: bool = False
    start_release_seen: bool = False
    left_brace_ever: bool = False
    last_button_index: int = -1
    brace_stable_steps: int = 0
    tilt_bad_steps: int = 0
    final_stable_steps: int = 0
    button_down: list[bool] = field(default_factory=lambda: [False] * len(BUTTON_NAMES))
    button_contact_seen: list[bool] = field(default_factory=lambda: [False] * len(BUTTON_NAMES))
    button_press_valid: list[bool] = field(default_factory=lambda: [False] * len(BUTTON_NAMES))
    button_hold_steps: list[int] = field(default_factory=lambda: [0] * len(BUTTON_NAMES))

    def reset(self, target: tuple[int, int]) -> None:
        self.target_minutes, self.target_seconds = map(int, target)
        self.programmed_minutes = 0
        self.programmed_seconds = 0
        self.running = False
        self.attempt_invalid = False
        self.error_count = 0
        self.last_error_code = ERROR_NONE
        self.stage = STAGE_WAIT_BRACE
        self.run_elapsed_s = 0.0
        self.countdown_decrements = 0
        self.terminal_failure_code = FAIL_NONE
        self.terminal_failure_reason = ""
        self.start_press_seen = False
        self.start_release_seen = False
        self.left_brace_ever = False
        self.last_button_index = -1
        self.brace_stable_steps = 0
        self.tilt_bad_steps = 0
        self.final_stable_steps = 0
        self.button_down[:] = [False] * len(BUTTON_NAMES)
        self.button_contact_seen[:] = [False] * len(BUTTON_NAMES)
        self.button_press_valid[:] = [False] * len(BUTTON_NAMES)
        self.button_hold_steps[:] = [0] * len(BUTTON_NAMES)

    @property
    def failed(self) -> bool:
        return self.terminal_failure_code != FAIL_NONE

    @property
    def error_reason(self) -> str:
        return ERROR_REASONS.get(self.last_error_code, "unknown press error")

    @property
    def target_total_seconds(self) -> int:
        return 60 * self.target_minutes + self.target_seconds

    @property
    def display_total_seconds(self) -> int:
        if not self.running:
            return 60 * self.programmed_minutes + self.programmed_seconds
        return max(0, self.target_total_seconds - self.countdown_decrements)

    @property
    def display(self) -> tuple[int, int]:
        total = self.display_total_seconds
        return total // 60, total % 60

    @property
    def programmed_target(self) -> bool:
        return (
            self.programmed_minutes == self.target_minutes
            and self.programmed_seconds == self.target_seconds
        )

    def fail(self, code: int, reason: str) -> None:
        if not self.failed:
            self.terminal_failure_code = int(code)
            self.terminal_failure_reason = str(reason)
            self.stage = STAGE_FAILED

    def invalidate(self, code: int) -> None:
        self.attempt_invalid = True
        self.error_count += 1
        self.last_error_code = int(code)
        if not self.running and not self.failed:
            self.stage = STAGE_PROGRAM

    def begin_press(
        self,
        index: int,
        *,
        right_contact: bool,
        left_button_contact: bool,
        left_brace_stable: bool,
        simultaneous: bool = False,
    ) -> None:
        index = int(index)
        if self.failed or self.button_down[index]:
            return
        self.button_down[index] = True
        self.button_hold_steps[index] = 1
        self.button_contact_seen[index] = bool(right_contact)
        self.last_button_index = index
        if self.running:
            # Spring/inertial travel alone is not a second operator press.
            # Keep the down latch until release, but require physical hand
            # contact before declaring a post-start operation.
            if right_contact or left_button_contact:
                self.fail(FAIL_POST_START_BUTTON, "button pressed after countdown start")
            return
        if index == BUTTON_INDEX["mode"]:
            self.fail(FAIL_MODE_SWITCH, "mode switch left TIMER")
            return
        if simultaneous:
            # Invalidate the already-held click too (including CLEAR), so its
            # later release cannot clear an actual overlapping-button error.
            self.button_press_valid[:] = [False] * len(BUTTON_NAMES)
            self.invalidate(ERROR_SIMULTANEOUS)
            return
        if left_button_contact:
            self.invalidate(ERROR_LEFT_BUTTON_CONTACT)
            return
        if not right_contact:
            self.invalidate(ERROR_NO_RIGHT_CONTACT)
            return
        if not left_brace_stable:
            self.invalidate(ERROR_NO_LEFT_BRACE)
            return
        self.left_brace_ever = True
        self.button_press_valid[index] = True
        if index == BUTTON_INDEX["start_stop"]:
            self.start_press_seen = True

    def hold_press(
        self,
        index: int,
        *,
        right_contact: bool,
        left_button_contact: bool,
        left_brace_stable: bool,
    ) -> None:
        index = int(index)
        if not self.button_down[index] or self.failed:
            return
        if self.running:
            if right_contact or left_button_contact:
                self.fail(FAIL_POST_START_BUTTON, "button pressed after countdown start")
            return
        self.button_contact_seen[index] = bool(
            self.button_contact_seen[index] or right_contact
        )
        self.button_hold_steps[index] += 1
        if self.button_press_valid[index] and left_button_contact:
            self.button_press_valid[index] = False
            self.invalidate(ERROR_LEFT_BUTTON_CONTACT)
        elif self.button_press_valid[index] and not left_brace_stable:
            self.button_press_valid[index] = False
            self.invalidate(ERROR_NO_LEFT_BRACE)

    def release_press(self, index: int) -> None:
        index = int(index)
        if not self.button_down[index]:
            return
        valid = bool(
            self.button_press_valid[index]
            and self.button_contact_seen[index]
            and not self.failed
        )
        self.button_down[index] = False
        self.button_contact_seen[index] = False
        self.button_press_valid[index] = False
        self.button_hold_steps[index] = 0
        if valid:
            self._apply_button(index)

    def _apply_button(self, index: int) -> None:
        name = BUTTON_NAMES[int(index)]
        if name == "clear":
            self.programmed_minutes = 0
            self.programmed_seconds = 0
            self.attempt_invalid = False
            self.last_error_code = ERROR_NONE
            self.start_press_seen = False
            self.start_release_seen = False
            self.stage = STAGE_PROGRAM
            return
        if name in {"hr", "memory"}:
            self.invalidate(ERROR_WRONG_FUNCTION)
            return
        if name in {"min", "sec"}:
            if self.attempt_invalid:
                return
            if name == "min":
                self.programmed_minutes += 1
            else:
                self.programmed_seconds += 1
            if (
                self.programmed_minutes > self.target_minutes
                or self.programmed_seconds > self.target_seconds
            ):
                self.invalidate(ERROR_OVERSHOOT)
            elif self.programmed_target:
                self.stage = STAGE_READY
            else:
                self.stage = STAGE_PROGRAM
            return
        if name == "start_stop":
            if self.attempt_invalid or not self.programmed_target:
                reason = (
                    f"CLEAR required: {self.error_reason}"
                    if self.attempt_invalid
                    else f"program {self.programmed_minutes:02d}:{self.programmed_seconds:02d} "
                         f"does not match target {self.target_minutes:02d}:{self.target_seconds:02d}"
                )
                self.fail(FAIL_WRONG_START, f"START/STOP pressed before a valid exact program; {reason}")
                return
            self.running = True
            self.start_release_seen = True
            self.run_elapsed_s = 0.0
            self.countdown_decrements = 0
            self.stage = STAGE_RUNNING

    def tick(self, dt: float) -> None:
        if not self.running or self.failed:
            return
        self.run_elapsed_s = max(0.0, self.run_elapsed_s + float(dt))
        self.countdown_decrements = min(
            self.target_total_seconds,
            int(math.floor(self.run_elapsed_s + 1e-9)),
        )

    def logic_vector(self) -> np.ndarray:
        return np.asarray(
            [
                float(self.running),
                float(self.attempt_invalid),
                float(self.error_count),
                float(self.last_error_code),
                float(self.stage),
                float(self.run_elapsed_s),
                float(self.countdown_decrements),
                float(self.terminal_failure_code),
                float(self.start_press_seen),
                float(self.start_release_seen),
                float(self.left_brace_ever),
                float(self.last_button_index),
                float(self.brace_stable_steps),
                float(self.tilt_bad_steps),
                float(self.final_stable_steps),
                float(self.programmed_minutes),
                float(self.programmed_seconds),
                float(self.failed),
            ],
            dtype=np.float64,
        )

    def restore_logic_vector(self, values: Iterable[float]) -> None:
        vector = np.asarray(tuple(values), dtype=np.float64).reshape(-1)
        if vector.size < LOGIC_SIZE:
            return
        self.running = bool(vector[0])
        self.attempt_invalid = bool(vector[1])
        self.error_count = int(round(vector[2]))
        self.last_error_code = int(round(vector[3]))
        self.stage = int(round(vector[4]))
        self.run_elapsed_s = float(vector[5])
        self.countdown_decrements = int(round(vector[6]))
        self.terminal_failure_code = int(round(vector[7]))
        self.start_press_seen = bool(vector[8])
        self.start_release_seen = bool(vector[9])
        self.left_brace_ever = bool(vector[10])
        self.last_button_index = int(round(vector[11]))
        self.brace_stable_steps = int(round(vector[12]))
        self.tilt_bad_steps = int(round(vector[13]))
        self.final_stable_steps = int(round(vector[14]))
        self.programmed_minutes = int(round(vector[15]))
        self.programmed_seconds = int(round(vector[16]))

    def latch_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(self.button_down, dtype=np.float64),
                np.asarray(self.button_contact_seen, dtype=np.float64),
                np.asarray(self.button_press_valid, dtype=np.float64),
                np.asarray(self.button_hold_steps, dtype=np.float64),
            ]
        )

    def restore_latch_vector(self, values: Iterable[float]) -> None:
        vector = np.asarray(tuple(values), dtype=np.float64).reshape(-1)
        if vector.size < LATCH_SIZE:
            return
        count = len(BUTTON_NAMES)
        self.button_down[:] = [bool(value) for value in vector[:count]]
        self.button_contact_seen[:] = [bool(value) for value in vector[count : 2 * count]]
        self.button_press_valid[:] = [bool(value) for value in vector[2 * count : 3 * count]]
        self.button_hold_steps[:] = [int(round(value)) for value in vector[3 * count : 4 * count]]


def seeded_target(random_state: np.random.RandomState) -> tuple[int, int]:
    return TARGETS[int(random_state.randint(0, len(TARGETS)))]
