from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import time


@dataclass(slots=True)
class ProgressLogger:
    path: Path
    interval_seconds: float = 30.0
    enabled: bool = True
    _run_start_monotonic: float = field(default=0.0, init=False)
    _last_emit_monotonic: float = field(default=0.0, init=False)

    def reset(self, **fields: object) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self._run_start_monotonic = time.monotonic()
        self._last_emit_monotonic = 0.0
        self.log("run_start", force=True, **fields)

    def maybe_log(self, event: str, **fields: object) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if self._last_emit_monotonic == 0.0 or (now - self._last_emit_monotonic) >= self.interval_seconds:
            self.log(event, force=True, now_monotonic=now, **fields)

    def log(self, event: str, force: bool = False, now_monotonic: float | None = None, **fields: object) -> None:
        if not self.enabled:
            return
        now = now_monotonic if now_monotonic is not None else time.monotonic()
        if not force and self._last_emit_monotonic != 0.0 and (now - self._last_emit_monotonic) < self.interval_seconds:
            return
        if self._run_start_monotonic == 0.0:
            self._run_start_monotonic = now
        elapsed = now - self._run_start_monotonic
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        payload = {"event": event, "elapsed_sec": f"{elapsed:.1f}", **fields}
        line = f"{timestamp} " + " ".join(f"{key}={self._format_value(val)}" for key, val in payload.items())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        self._last_emit_monotonic = now

    @staticmethod
    def _format_value(val: object) -> str:
        if val is None:
            return "none"
        text = str(val)
        return text.replace(" ", "_")
