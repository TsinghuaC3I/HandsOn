"""Collection-only timer readout for the D20 microwave task.

The frame is attached to the live Isaac viewport, not to a camera or USD
prim.  It therefore helps an operator during guidance recording while staying
out of replay and offline re-render products.
"""

from __future__ import annotations

from typing import Any, Mapping


def _card_style(paused: bool) -> dict[str, Any]:
    return {
        "background_color": 0xE0181C22,
        "border_color": 0xFFFFC857 if paused else 0x70FFFFFF,
        "border_width": 5 if paused else 2,
        "border_radius": 12,
    }


class D20TimerOverlay:
    """Large lower-viewport timer card used only during live collection."""

    def __init__(self) -> None:
        self.frame = None
        self.card = None
        self.value = None
        self.detail = None
        self.guidance = None

    def build(self, ui: Any, frame: Any) -> None:
        self.close()
        self.frame = frame
        frame.clear()
        frame.visible = True
        with frame:
            with ui.VStack():
                ui.Spacer()
                with ui.HStack(height=118):
                    ui.Spacer()
                    with ui.ZStack(width=420, height=112):
                        self.card = ui.Rectangle(style=_card_style(False))
                        with ui.VStack(spacing=2, style={"margin": 10}):
                            ui.Label(
                                "MICROWAVE TIMER",
                                height=22,
                                alignment=ui.Alignment.CENTER,
                                style={"font_size": 18, "color": 0xFFD7DEE8},
                            )
                            self.value = ui.Label(
                                "00:00",
                                height=52,
                                alignment=ui.Alignment.CENTER,
                                style={"font_size": 42, "color": 0xFFFFFFFF},
                            )
                            self.detail = ui.Label(
                                "",
                                height=20,
                                alignment=ui.Alignment.CENTER,
                                style={"font_size": 15, "color": 0xFFE1E7EF},
                            )
                    ui.Spacer()
                with ui.HStack(height=34):
                    ui.Spacer()
                    with ui.ZStack(width=600, height=34):
                        ui.Rectangle(style={"background_color": 0xE0181C22, "border_radius": 6})
                        self.guidance = ui.Label(
                            "",
                            alignment=ui.Alignment.CENTER,
                            word_wrap=True,
                            style={"font_size": 14, "color": 0xFFFFFFFF, "margin": 6},
                        )
                    ui.Spacer()
                ui.Spacer(height=24)

    def update(self, metrics: Mapping[str, Any]) -> None:
        if self.frame is None:
            return
        self.frame.visible = True
        remaining = max(0, int(float(metrics.get("d20_timer_minutes", 0.0)) * 60.0 + 1.0e-6))
        paused = bool(metrics.get("d20_heating_paused", False))
        running = bool(metrics.get("d20_running", False))
        state = "PAUSED - hand on knob" if paused else ("HEATING" if running else "STOPPED")
        if self.value is not None:
            self.value.text = f"{remaining // 60:02d}:{remaining % 60:02d}"
        if self.detail is not None:
            self.detail.text = f"{state}  |  cook {100.0 * float(metrics.get('d20_cook_progress', 0.0)):.1f}%"
        if self.guidance is not None:
            self.guidance.text = (
                "Release the knob to resume the clockwork; hand contact pauses safely."
                if paused
                else "Set 01:00, then let the lower knob return to 00:00."
            )
        if self.card is not None:
            self.card.style = _card_style(paused)

    def close(self) -> None:
        if self.frame is not None:
            self.frame.visible = False
            self.frame.clear()
        self.frame = None
        self.card = None
        self.value = None
        self.detail = None
        self.guidance = None


__all__ = ["D20TimerOverlay"]
