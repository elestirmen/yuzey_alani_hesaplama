"""Lightweight terminal progress reporting (no external deps)."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import TextIO


def _format_seconds(seconds: float) -> str:
    s = int(max(0.0, float(seconds)))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


@dataclass(slots=True)
class ProgressPrinter:
    """Single-line progress printer for terminal use.

    - Writes to stderr by default.
    - Uses carriage-return updates when running in a TTY.
    - Throttles updates to avoid flooding the console.
    """

    enabled: bool | None = None
    stream: TextIO = sys.stderr
    min_interval_s: float = 0.25
    min_delta_percent: int = 1

    _last_render: str = ""
    _last_update_t: float = 0.0
    _last_percent: int = -1
    _label: str = ""
    _stage_start_t: float = 0.0

    def __post_init__(self) -> None:
        if self.enabled is None:
            try:
                self.enabled = bool(self.stream.isatty())
            except Exception:
                self.enabled = False

    def _clear_line(self) -> None:
        if not self.enabled:
            return
        if not self._last_render:
            return
        self.stream.write("\r" + (" " * len(self._last_render)) + "\r")
        self.stream.flush()
        self._last_render = ""
        self._last_percent = -1

    def log(self, message: str) -> None:
        """Print a normal log line (and clear any progress line)."""
        if not self.enabled:
            print(message, file=self.stream)
            return
        self._clear_line()
        print(message, file=self.stream)
        self.stream.flush()

    def update(self, *, label: str, current: int, total: int) -> None:
        if not self.enabled:
            return
        if total <= 0:
            return

        current_i = int(max(0, min(int(current), int(total))))
        total_i = int(total)
        percent = int(round((100.0 * current_i) / float(total_i))) if total_i else 0

        now = time.monotonic()
        if label != self._label:
            self._label = label
            self._stage_start_t = now
            self._last_percent = -1

        if (now - self._last_update_t) < float(self.min_interval_s) and abs(percent - self._last_percent) < int(
            self.min_delta_percent
        ):
            return

        elapsed = now - float(self._stage_start_t)
        eta_s = (elapsed / current_i) * (total_i - current_i) if current_i > 0 else float("inf")
        eta_txt = _format_seconds(eta_s) if eta_s != float("inf") else "--:--"

        text = f"{label}: {percent:3d}% ({current_i}/{total_i}) ETA {eta_txt}"
        if len(text) < len(self._last_render):
            text = text + (" " * (len(self._last_render) - len(text)))

        self.stream.write("\r" + text)
        self.stream.flush()
        self._last_render = text
        self._last_update_t = now
        self._last_percent = percent

    def finish(self) -> None:
        """End the current progress line with a newline."""
        if not self.enabled:
            return
        if not self._last_render:
            return
        self.stream.write("\n")
        self.stream.flush()
        self._last_render = ""
        self._last_percent = -1

