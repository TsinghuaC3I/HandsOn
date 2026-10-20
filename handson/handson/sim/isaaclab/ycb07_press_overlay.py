"""Live MIN/SEC press counters, independent of the timer's countdown display."""

from __future__ import annotations

from typing import Any

from .ycb07_timer_logic import BUTTON_INDEX, TimerProgrammingState


def press_progress(state: TimerProgrammingState) -> list[dict[str, Any]]:
    rows = []
    for name, count, target in (
        ("min", state.programmed_minutes, state.target_minutes),
        ("sec", state.programmed_seconds, state.target_seconds),
    ):
        if count < target:
            detail = f"{target - count} more to go"
        elif count > target:
            detail = f"{count - target} over target - CLEAR"
        else:
            detail = "Target matched"
        if state.failed:
            detail = "FAILED - reset with R"
        elif state.attempt_invalid and count <= target:
            detail = "BLOCKED - press CLEAR"
        if state.button_down[BUTTON_INDEX[name]]:
            detail += " | DOWN"
        rows.append({
            "name": name,
            "count": count,
            "target": target,
            "matched": count == target and not state.attempt_invalid and not state.failed,
            "detail": detail,
        })
    return rows


def press_guidance(state: TimerProgrammingState) -> str:
    if state.failed:
        return f"FAILED: {state.terminal_failure_reason}. Reset with R."
    if state.attempt_invalid:
        return f"Blocked: {state.error_reason}. Press and release CLEAR."
    if state.running:
        minutes, seconds = state.display
        return f"Countdown {minutes:02d}:{seconds:02d} - withdraw both hands."
    if any(state.button_down):
        return "Button DOWN detected - release fully to count once."
    if state.programmed_target:
        return "Both targets matched - press and release START/STOP."
    return "Brace case with left hand. Right fingertip: press, then release to count."


def press_card_style(matched: bool) -> dict[str, Any]:
    return {
        "background_color": 0xE02A3820 if matched else 0xE0181C22,
        "border_color": 0xFF81EE80 if matched else 0x70FFFFFF,
        "border_width": 5 if matched else 2,
        "border_radius": 12,
    }


class TimerPressOverlay:
    """Use the same bottom-of-viewport card layout as the dual-knob task."""

    def __init__(self):
        self.frame = None
        self.cards = {}
        self.values = {}
        self.details = {}
        self.guidance = None

    def build(self, ui: Any, frame: Any) -> None:
        self.close()
        self.frame = frame
        frame.clear()
        frame.visible = True
        with frame:
            with ui.VStack():
                ui.Spacer()
                with ui.HStack(height=148):
                    ui.Spacer()
                    for index, name in enumerate(("min", "sec")):
                        if index:
                            ui.Spacer(width=32)
                        with ui.ZStack(width=250, height=140):
                            self.cards[name] = ui.Rectangle(style=press_card_style(False))
                            with ui.VStack(spacing=2, style={"margin": 10}):
                                ui.Label(name.upper(), height=22,
                                         alignment=ui.Alignment.CENTER,
                                         style={"font_size": 20, "color": 0xFFD7DEE8})
                                self.values[name] = ui.Label(
                                    "", height=44, alignment=ui.Alignment.CENTER,
                                    style={"font_size": 38, "color": 0xFFFFFFFF})
                                ui.Label("COUNTED / TARGET", height=18,
                                         alignment=ui.Alignment.CENTER,
                                         style={"font_size": 12, "color": 0xFFB8C3D0})
                                self.details[name] = ui.Label(
                                    "", height=22, alignment=ui.Alignment.CENTER,
                                    style={"font_size": 15, "color": 0xFFE1E7EF})
                    ui.Spacer()
                with ui.HStack(height=46):
                    ui.Spacer()
                    with ui.ZStack(width=620, height=46):
                        ui.Rectangle(style={"background_color": 0xE0181C22,
                                            "border_radius": 6})
                        self.guidance = ui.Label(
                            "", alignment=ui.Alignment.CENTER, word_wrap=True,
                            style={"font_size": 15, "color": 0xFFFFFFFF, "margin": 8})
                    ui.Spacer()
                ui.Spacer(height=28)

    def update(self, state: TimerProgrammingState) -> None:
        if self.frame is None:
            return
        self.frame.visible = True
        for row in press_progress(state):
            name = row["name"]
            self.values[name].text = f"{row['count']} / {row['target']}"
            self.details[name].text = row["detail"]
            self.cards[name].style = press_card_style(row["matched"])
        self.guidance.text = press_guidance(state)

    def close(self) -> None:
        if self.frame is not None:
            self.frame.visible = False
            self.frame.clear()
        self.frame = None
        self.cards.clear()
        self.values.clear()
        self.details.clear()
        self.guidance = None
