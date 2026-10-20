"""Stage identities and physical completion predicates on the existing D20 scene."""
from __future__ import annotations

import numpy as np
from . import d20_microwave_hot_dog as d20

STAGE_TASKS = (
    "d20_1_load_and_start",
    "d20_1_5_set_controls",
    "d20_2_retrieve_and_plate",
    "d20_3_finish_table_setting",
)
TASK_NAMES = frozenset((d20.TASK_NAME, *STAGE_TASKS))
LABELS = ("D20-1 装入热狗并关门", "D20-1.5 设置中火3分钟", "D20-2 取出热狗并装盘", "D20-3 完成餐具摆放")
PROMPTS = (
    "Use the tongs to put the hot dog inside the microwave, release the food, put the tongs down, "
    "close the door. Keep the food and tongs released and stable for 0.5 seconds. This stage ends before any rotary control operation.",
    "Watch the locally saved D20-1 demonstration. Continue from that saved scene, set the upper knob to Medium and the lower timer to 3 minutes (150-210 simulation seconds accepted at release). While a hand is on either knob the clock is paused; after release, the acceptance window moves down with the countdown (for example, after 5 seconds it is 145-205). This stage completes once the hands are clear and the setting is stable.",
    "Watch the locally saved D20-1.5 demonstration. During stitching both hands stay fixed while the timer is fast-forwarded to zero. Continue from "
    "that saved scene: open the stopped microwave, use the tongs to take out the cooked hot dog, "
    "place it at the plate center and release it stably for 0.5 seconds.",
    "Watch the complete saved D20-2 workflow, including its D20-1 and D20-1.5 predecessors. Continue from the "
    "saved plated scene: put the fork left of the plate, mustard right, and mug upper right. "
    "Finish with the microwave stopped, door closed, both hands no longer touching the food, tongs or tableware, and all objects stable for 0.5 seconds.",
)
CONDITIONS = (
    ("Load the food with the tongs", "Food released inside", "Tongs put down", "Door closed", "Stable and released for 0.5 s"),
    ("Hot dog and tongs restored from D20-1", "Medium power and timer about 3 minutes (150-210 s at release; window follows countdown)", "Hands released after setting", "Stable and released for 0.5 s"),
    ("Cooking completed and heating stopped", "Retrieve the food with the tongs", "Hot dog centered on plate and released", "Plate and hot dog stable for 0.5 s"),
    (*d20.SUCCESS_CONDITION_LABELS[:-1],
     "10. Plate and tableware on tray; hands not touching food, tongs or tableware; objects stable for 0.5 s."),
)

# D20-1.5 accepts a three-minute medium-power setting with a half-minute
# adjustment tolerance.  The timer is paused while a hand is on either knob;
# releasing the hand leaves the setting as the successful stage boundary.  The
# retrieval continuation then owns the accelerated heating wait.
STAGE1_POWER_INDEX = 2  # 中火
STAGE1_TIMER_MIN_SECONDS = 150.0
STAGE1_TIMER_MAX_SECONDS = 210.0
STAGE1_TIMER_EPSILON = 1e-6


def stage_number(task_name: str) -> int:
    return STAGE_TASKS.index(task_name) + 1 if task_name in STAGE_TASKS else 0


def family_name(task_name: str) -> str:
    return d20.TASK_NAME if task_name in TASK_NAMES else task_name


class StageProgress:
    """The five values are saved with every current/next policy state."""
    SIZE = 5

    def __init__(self, stage: int = 0):
        self.stage = stage
        self.reset()

    def reset(self):
        self.grasp_seen = False
        self.loaded_with_tongs = False
        self.retrieved_with_tongs = False
        self.release_seconds = 0.0
        # Historical field name is retained in the five-value save schema;
        # it now means "accepted timer window seen" for the three-minute
        # medium-power control stage.
        self.one_minute_seen = False
        # Runtime-only timer observations; these are not part of the saved
        # stage schema because the control stage can be corrected live.
        self.initial_timer_seconds = None
        # Runtime-only countdown bookkeeping.  A failed release is
        # recoverable: the operator may adjust the knob again while the
        # acceptance window follows the time that has actually elapsed.
        self.countdown_elapsed_seconds = 0.0
        self.last_unheld_timer_seconds = None

    def observation(self):
        return np.asarray([self.grasp_seen, self.loaded_with_tongs, self.retrieved_with_tongs,
                           self.release_seconds, self.one_minute_seen], dtype=np.float32)

    def restore(self, values):
        values = np.asarray(values, dtype=float).ravel()
        if values.size != self.SIZE or not np.isfinite(values).all():
            raise ValueError("Invalid D20 stage save state")
        self.grasp_seen, self.loaded_with_tongs, self.retrieved_with_tongs = map(bool, values[:3])
        self.release_seconds = float(values[3])
        self.one_minute_seen = bool(values[4])
        self.initial_timer_seconds = None
        self.countdown_elapsed_seconds = 0.0
        self.last_unheld_timer_seconds = None

    def record_tongs_grasp(self, *, inside, cooked, stopped):
        """Latch physical grasp events before a later substep can release them."""
        self.grasp_seen = True
        self.loaded_with_tongs |= bool(inside)
        self.retrieved_with_tongs |= bool(cooked and stopped and not inside)

    def update(self, reading, metrics, *, update):
        if not self.stage:
            return metrics
        m = dict(metrics)
        inside = m["d20_hot_dog_inside"]
        if update:
            self.grasp_seen |= bool(reading.tongs_latched and reading.tongs_holding_side)
            self.loaded_with_tongs |= bool(inside and self.grasp_seen)
            self.retrieved_with_tongs |= bool(
                m["d20_cook_ok"] and m["d20_stop_ok"] and not inside
                and reading.tongs_latched and reading.tongs_holding_side)
            current_timer_seconds = m["d20_timer_minutes"] * 60.0
            current_timer_valid = bool(
                STAGE1_TIMER_MIN_SECONDS - STAGE1_TIMER_EPSILON
                <= current_timer_seconds
                <= STAGE1_TIMER_MAX_SECONDS + STAGE1_TIMER_EPSILON
            )
            # A hand can hold a knob while the clock is paused. Show the
            if (
                self.stage == 2
                and not reading.timer_two_finger_hold
                and not reading.tongs_latched
                and not reading.tongs_holding_side
            ):
                # Knob contact pauses the microwave.  Count only timer motion
                # observed after the controls are released, so rotating to a
                # new value while holding a knob does not consume tolerance.
                if self.initial_timer_seconds is None:
                    self.last_unheld_timer_seconds = current_timer_seconds
                    self.initial_timer_seconds = current_timer_seconds
                else:
                    self.countdown_elapsed_seconds += max(
                        0.0,
                        self.last_unheld_timer_seconds - current_timer_seconds,
                    )
                    self.last_unheld_timer_seconds = current_timer_seconds
                self.one_minute_seen = current_timer_valid
            elif self.stage == 2 and reading.timer_two_finger_hold:
                # A knob rotation while held is paused/manual adjustment, not
                # countdown time. Re-anchor before the next release.
                self.last_unheld_timer_seconds = current_timer_seconds
        # The held tool can remain in the operator's hand at the end of stage 2.
        payload_released = not reading.tongs_latched and not m.get("d20_tongs_payload_contact", False)
        tongs = reading.poses[d20.TONGS]
        tongs_down = bool(
            not reading.tongs_holding_side
            # The tongs root is above the table by its collider radius and
            # can rest at a different height when rotated.  Use a generous
            # support band; release/hand-clear checks below still gate success.
            and reading.table_top_z - .02 <= tongs[2] <= reading.table_top_z + .14
            and d20.is_stable(reading.velocities.get(d20.TONGS)))
        safe = bool(not m["d20_failed"] and m["d20_no_heated_touch"] and not m["d20_order_violation"])
        if self.stage == 1:
            rows = (
                self.loaded_with_tongs,
                inside,
                tongs_down,
                m["d20_door_closed"] and not m["d20_running"] and m["d20_timer_minutes"] == 0,
            )
            ready = all(rows) and payload_released and m["d20_stable"] and m["d20_hands_released"] and safe
        elif self.stage == 2:
            current_timer_seconds = m["d20_timer_minutes"] * 60.0
            # The microwave starts counting down as soon as the operator lets
            # go of the controls.  Keep the original 150-210 s setting window
            # anchored to that first released value, so a five-second
            # countdown changes the live acceptance window to 145-205 s.
            # ``initial_timer_seconds`` is only recorded after knob release;
            # this also prevents an overlong setting from becoming valid just
            # by counting down through the window.
            countdown_seconds = self.countdown_elapsed_seconds
            timer_min = STAGE1_TIMER_MIN_SECONDS - countdown_seconds
            timer_max = STAGE1_TIMER_MAX_SECONDS - countdown_seconds
            self.one_minute_seen = bool(
                timer_min - STAGE1_TIMER_EPSILON
                <= current_timer_seconds
                <= timer_max + STAGE1_TIMER_EPSILON
            )
            controls = bool(self.one_minute_seen and m["d20_power_index"] == STAGE1_POWER_INDEX
                            and m["d20_door_closed"] and m["d20_running"]
                            and timer_min - STAGE1_TIMER_EPSILON
                            <= current_timer_seconds
                            <= timer_max + STAGE1_TIMER_EPSILON)
            # For D20-1.5 the third visible condition is specifically release
            # from the two control knobs. A global 110 mm object-distance
            # proxy can stay false even when the hands have no knob contact.
            hands_released_after_setting = bool(
                m.get("d20_hand_pose_available", False)
                and not reading.timer_two_finger_hold
            )
            rows = (
                self.loaded_with_tongs and inside and tongs_down,
                controls,
                hands_released_after_setting,
            )
            ready = all(rows) and payload_released and m["d20_stable"] and safe
        elif self.stage == 3:
            unstable = [label for name, label in ((d20.HOT_DOG, "Hot dog"), (d20.PLATE, "Plate"))
                        if not d20.is_stable(reading.velocities.get(name))]
            stable = not unstable
            rows = (m["d20_cook_ok"] and m["d20_stop_ok"], self.retrieved_with_tongs,
                    m["d20_plated_now"] and payload_released and m["d20_plate_on_tray"])
            ready = all(rows) and stable and safe
            # The last row also includes historical safety guards. A hot-food
            # touch or lost contact evidence can block it even when every
            # placement row is green; expose that instead of only showing 0 s.
            if not safe:
                blocker = d20.operator_next_action(m)
            elif not all(rows):
                blocker = "Complete the first three steps."
            elif unstable:
                blocker = f"{' and '.join(unstable)} still moving; wait for it to settle."
            else:
                blocker = ""
            m["d20_stage_release_blocker"] = blocker
        elif self.stage == 4:
            rows = ()
            ready = False
        else:
            ready = False
            rows = ()
        if update and self.stage != 4:
            self.release_seconds = min(.5, self.release_seconds + reading.dt) if ready else 0.0
        complete = ready and self.release_seconds + 1e-9 >= .5
        if self.stage != 4:
            m["success_now"] = bool(complete)
        m.update(d20_stage=self.stage, d20_stage_rows=(*rows, complete),
                 d20_stage_release_seconds=self.release_seconds,
                 d20_grasp_seen=self.grasp_seen,
                 d20_tongs_down=tongs_down, d20_loaded_with_tongs=self.loaded_with_tongs,
                 d20_retrieved_with_tongs=self.retrieved_with_tongs)
        if self.stage == 2:
            m["d20_stage_timer_window"] = (timer_min, timer_max)
        return m


def retrieval_release_status(metrics):
    """Explain the existing D20-2 hold guard without changing task progress."""
    blocker = metrics.get("d20_stage_release_blocker")
    if blocker:
        return f"Release blocked: {blocker}"
    seconds = metrics.get("d20_stage_release_seconds", 0.0)
    return f"Plate and hot dog stable: {seconds:.2f}/0.50 s"


def operator_text(stage, metrics):
    if stage == 4:
        return "D20-3 | Continue saved table setting\n" + d20.operator_status_text(metrics)
    label = LABELS[stage - 1] if 1 <= stage <= len(LABELS) else f"D20 stage {stage}"
    power_hint = "Medium = 3" if stage == 2 else "High = 5"
    timer_window = metrics.get("d20_stage_timer_window")
    window_text = ""
    if stage == 2 and timer_window is not None:
        window_text = f" | Accepted now: {timer_window[0]:.0f}-{timer_window[1]:.0f} s"
    release_detail = f"\n{retrieval_release_status(metrics)}" if stage == 3 else ""
    return (f"{label} | {PROMPTS[stage-1]}\n"
            f"Tongs grasp recorded: {'YES' if metrics.get('d20_grasp_seen') else 'NO'} | "
            f"Power: {metrics.get('d20_power_index', 0) + 1}/5 ({power_hint}) | "
            f"Timer: {metrics.get('d20_timer_minutes', 0) * 60:.1f} s{window_text} | "
            f"Release: {metrics.get('d20_stage_release_seconds', 0):.1f}/0.5 s{release_detail}")
