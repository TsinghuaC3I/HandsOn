#!/usr/bin/env python3
"""Small always-on-top viewer for the recorder's atomically replaced PNG."""

from __future__ import annotations

import argparse
import os
import signal
import time
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk


def _parent_is_alive(parent_pid: int) -> bool:
    if parent_pid <= 0:
        return True
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _request_parent_reset(parent_pid: int, *, kill=None) -> bool:
    """Queue a reset in the recorder without resetting from the GUI thread."""

    reset_signal = getattr(signal, "SIGUSR1", None)
    if parent_pid <= 0 or reset_signal is None:
        return False
    if kill is None:
        kill = os.kill
    try:
        kill(int(parent_pid), reset_signal)
    except (OSError, ProcessLookupError, PermissionError):
        return False
    return True


class LiveCameraViewer:
    REFRESH_INTERVAL_MS = 33
    RESET_DEBOUNCE_SECONDS = 0.5

    def __init__(self, image_path: Path, *, parent_pid: int, title: str) -> None:
        self.image_path = image_path
        self.parent_pid = int(parent_pid)
        self._base_title = f"{title} · 按 R 重置"
        self._last_reset_request = float("-inf")
        self.root = tk.Tk(className="HandsOnLiveCamera")
        self.root.title(self._base_title)
        self.root.configure(background="#050706")
        self.root.attributes("-topmost", True)
        self.root.minsize(480, 480)
        self.label = tk.Label(
            self.root,
            background="#050706",
            foreground="#f3f6f4",
            text="正在等待 NewtonWarp 实时相机…",
        )
        self.label.pack(fill=tk.BOTH, expand=True)
        self._photo: ImageTk.PhotoImage | None = None
        self._last_signature: tuple[int, int] | None = None
        self.root.bind_all("<KeyPress-r>", self._request_reset)
        self.root.bind_all("<KeyPress-R>", self._request_reset)
        self.root.after(0, self._refresh)
        self.root.after(500, self._keep_above)

    def _restore_title(self) -> None:
        if self.root.winfo_exists():
            self.root.title(self._base_title)

    def _request_reset(self, _event=None) -> str:
        now = time.monotonic()
        if now - self._last_reset_request < self.RESET_DEBOUNCE_SECONDS:
            return "break"
        self._last_reset_request = now
        if _request_parent_reset(self.parent_pid):
            self.root.title(f"{self._base_title} · 已发送")
        else:
            self.root.title(f"{self._base_title} · 重置失败")
        self.root.after(900, self._restore_title)
        return "break"

    def _keep_above(self) -> None:
        if not self.root.winfo_exists():
            return
        self.root.attributes("-topmost", True)
        self.root.lift()
        self.root.after(1000, self._keep_above)

    def _refresh(self) -> None:
        if not _parent_is_alive(self.parent_pid):
            self.root.destroy()
            return
        try:
            stat = self.image_path.stat()
            signature = (int(stat.st_mtime_ns), int(stat.st_size))
            if signature != self._last_signature:
                with Image.open(self.image_path) as source:
                    image = source.convert("RGB")
                max_width = max(480, min(900, self.root.winfo_screenwidth() - 160))
                max_height = max(480, min(900, self.root.winfo_screenheight() - 200))
                image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                self._photo = ImageTk.PhotoImage(image)
                self.label.configure(image=self._photo, text="")
                self._last_signature = signature
        except (OSError, ValueError):
            # Atomic replacement can leave a very short interval in which the
            # path is absent.  Retain the last good frame and retry promptly.
            pass
        self.root.after(self.REFRESH_INTERVAL_MS, self._refresh)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--title", default="HandsOn 实时仿真相机")
    args = parser.parse_args()
    LiveCameraViewer(
        args.image.expanduser(),
        parent_pid=args.parent_pid,
        title=args.title,
    ).run()


if __name__ == "__main__":
    main()
